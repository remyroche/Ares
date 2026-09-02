#!/usr/bin/env python3
"""Read-only audit of the Strict-R3 live hourly operational chain.

The audit deliberately consumes immutable receipts only.  It provides a
deterministic count toward the required consecutive-candle operational review
without fetching market data, scoring candidates, mutating live state, or
calling the exchange.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
REPORTS = ROOT / "data_perp" / "reports"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _producer_runtime(receipt: Path) -> str:
    match = re.match(
        r"^strict_r3_live_hourly_producer_(.+)_\d{8}T\d{6}Z_v\d+$",
        receipt.name,
    )
    if not match:
        raise ValueError(f"invalid producer receipt name: {receipt.name}")
    return match.group(1)


def _report_for(receipt: Path, decision: pd.Timestamp) -> Path | None:
    runtime = _producer_runtime(receipt)
    prefix = (
        f"strict_r3_live_candle_{runtime}_"
        f"{decision.strftime('%Y%m%dT%H%M%SZ')}_{receipt.name}"
    )
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for candidate in REPORTS.glob(f"{prefix}*.json"):
        try:
            payload = _load(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("producer_receipt") == str(receipt.relative_to(ROOT)):
            candidates.append((candidate, payload))
    if not candidates:
        return None
    # An immutable report made against a retired state remains diagnostic
    # evidence.  Prefer a later state-bound, clean reconciliation receipt for
    # the operational chain, never overwriting the original incident report.
    clean = [
        item for item in candidates
        if item[1].get("status") == "pass" and not item[1].get("irregularities")
    ]
    selected = clean or candidates
    return max(selected, key=lambda item: item[0].stat().st_mtime_ns)[0]


def _supervisor_for(report: Path, decision: pd.Timestamp) -> Path | None:
    payload = _load(report)
    runtime = str(payload.get("runtime_tag") or "")
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    expected_report = str(report.relative_to(ROOT))
    candidates: list[Path] = []
    for candidate in ARTIFACTS.glob(
        f"strict_r3_live_operations_supervisor_{runtime}_{tag}_*/run_manifest.json"
    ):
        try:
            supervisor = _load(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if supervisor.get("report") == expected_report:
            candidates.append(candidate)
    return max(candidates, key=lambda item: item.stat().st_mtime_ns) if candidates else None


def _successes_since(start: pd.Timestamp) -> list[tuple[pd.Timestamp, Path, dict[str, Any]]]:
    grouped: dict[pd.Timestamp, list[tuple[Path, dict[str, Any]]]] = {}
    for manifest in ARTIFACTS.glob("strict_r3_live_hourly_producer_*_v*/run_manifest.json"):
        try:
            payload = _load(manifest)
            decision = _utc(payload["decision_ts"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if decision < start:
            continue
        if not (
            payload.get("status") == "pass"
            and payload.get("mode") == "live"
            and bool(payload.get("exchange_order_submission"))
        ):
            continue
        grouped.setdefault(decision, []).append((manifest.parent, payload))
    selected: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for decision, values in grouped.items():
        # Multiple successful live producer receipts for one decision would be
        # an idempotency incident, not an arbitrary tie to hide.
        if len(values) != 1:
            selected.append((decision, Path(), {"_duplicate_successes": values}))
        else:
            receipt, payload = values[0]
            selected.append((decision, receipt, payload))
    return sorted(selected, key=lambda item: item[0])


def audit(*, start: pd.Timestamp, required: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    expected = start
    consecutive = 0
    for decision, receipt, producer in _successes_since(start):
        issues: list[str] = []
        if decision != expected:
            issues.append(f"nonconsecutive_decision_expected={expected.isoformat()}")
        if not receipt:
            issues.append("multiple_successful_producer_receipts")
        else:
            if not (receipt / "producer_lease.json").is_file():
                issues.append("producer_lease_missing")
            if not (receipt / "execution_attempt_started.json").is_file():
                issues.append("execution_intent_missing")
            report = _report_for(receipt, decision)
            if report is None:
                issues.append("report_missing")
            else:
                report_payload = _load(report)
                if report_payload.get("status") != "pass":
                    issues.append("report_not_pass")
                if report_payload.get("producer_receipt") != str(receipt.relative_to(ROOT)):
                    issues.append("report_producer_identity_mismatch")
                if report_payload.get("irregularities"):
                    issues.append("report_irregularities_present")
                monitor = report_payload.get("position_monitor") or {}
                if not monitor.get("receipt"):
                    issues.append("monitor_receipt_missing")
                supervisor = _supervisor_for(report, decision)
                if supervisor is None:
                    issues.append("supervisor_receipt_missing")
                else:
                    supervisor_payload = _load(supervisor)
                    if not supervisor_payload.get("terminal"):
                        issues.append("supervisor_not_terminal")
                    if supervisor_payload.get("report_status") != "pass":
                        issues.append("supervisor_report_not_pass")
        valid = not issues
        records.append({
            "decision_ts": decision.isoformat(),
            "producer_receipt": str(receipt.relative_to(ROOT)) if receipt else None,
            "valid": valid,
            "issues": issues,
        })
        if valid and decision == expected:
            consecutive += 1
            expected += pd.Timedelta(hours=1)
        else:
            break
    return {
        "schema": "strict_r3_live_operational_chain_audit_v1",
        "read_only": True,
        "start": start.isoformat(),
        "required_consecutive_candles": required,
        "valid_consecutive_candles": consecutive,
        "remaining": max(0, required - consecutive),
        "ready_for_completion": consecutive >= required,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--required", type=int, default=8)
    args = parser.parse_args()
    if args.required <= 0:
        raise ValueError("--required must be positive")
    print(json.dumps(audit(start=_utc(args.start), required=args.required), indent=2))


if __name__ == "__main__":
    main()
