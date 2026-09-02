#!/usr/bin/env python3
"""Audit a contiguous strict-R3 live operational chain from immutable receipts.

It is intentionally read-only with respect to live trading.  A completed
candle requires a terminal producer report, its matching read-only supervisor
receipt, full source readiness, and at least one successful one-minute
position-monitor receipt inside the candle hour.  Fewer than the requested
number of contiguous hours is `in_progress`, not a passing promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _monitor_ok(decision: pd.Timestamp) -> tuple[bool, str | None]:
    candidates: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for manifest in ARTIFACTS.glob("strict_r3_live_position_monitor_*/monitor_*/run_manifest.json"):
        try:
            payload = _load(manifest)
            observed = _utc(payload["observed_at"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if decision <= observed < decision + pd.Timedelta(hours=1):
            candidates.append((observed, manifest, payload))
    if not candidates:
        return False, None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return payload.get("status") in {None, "pass"}, str(path.relative_to(ROOT))


def _supervisor_receipt(runtime: str, decision: pd.Timestamp) -> Path | None:
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    matches = sorted(ARTIFACTS.glob(
        f"strict_r3_live_operations_supervisor_{runtime}_{tag}_v*/run_manifest.json"
    ))
    return matches[-1] if matches else None


def _one_hour(runtime: str, decision: pd.Timestamp) -> dict[str, Any]:
    supervisor_path = _supervisor_receipt(runtime, decision)
    if supervisor_path is None:
        return {"decision_ts": decision.isoformat(), "status": "missing_supervisor_receipt"}
    supervisor = _load(supervisor_path)
    report_path = ROOT / str(supervisor.get("report") or "")
    if not report_path.is_file():
        return {"decision_ts": decision.isoformat(), "status": "missing_report", "supervisor": str(supervisor_path.relative_to(ROOT))}
    report = _load(report_path)
    report_hash_ok = str(supervisor.get("report_sha256")) == _sha(report_path)
    monitor_ok, monitor_path = _monitor_ok(decision)
    funnel = dict(report.get("funnel") or {})
    checks = {
        "supervisor_terminal": bool(supervisor.get("terminal")),
        "supervisor_read_only": bool(supervisor.get("read_only")) and int(supervisor.get("exchange_calls", -1)) == 0 and int(supervisor.get("live_state_mutations", -1)) == 0,
        "report_hash_exact": report_hash_ok,
        "report_pass": report.get("status") == "pass",
        "source_complete": int(funnel.get("source_ready", -1)) == int(funnel.get("universe", -2)) and int(funnel.get("universe", 0)) > 0,
        "monitor_passed_in_hour": monitor_ok,
    }
    return {
        "decision_ts": decision.isoformat(),
        "status": "pass" if all(checks.values()) else "action_required",
        "checks": checks,
        "funnel": funnel,
        "report": str(report_path.relative_to(ROOT)),
        "supervisor": str(supervisor_path.relative_to(ROOT)),
        "monitor": monitor_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tag", required=True)
    parser.add_argument("--start", required=True, help="first decision timestamp")
    parser.add_argument("--hours", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.hours < 1:
        raise ValueError("--hours must be positive")
    start = _utc(args.start).floor("h")
    rows = [_one_hour(args.runtime_tag, start + pd.Timedelta(hours=index)) for index in range(args.hours)]
    completed = [row for row in rows if row["status"] in {"pass", "action_required"}]
    failures = [row for row in completed if row["status"] != "pass"]
    status = "action_required" if failures else ("pass" if len(completed) == args.hours else "in_progress")
    payload = {
        "schema": "strict_r3_live_operations_chain_audit_v1",
        "runtime_tag": args.runtime_tag,
        "start": start.isoformat(),
        "requested_hours": args.hours,
        "completed_hours": len(completed),
        "status": status,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "completed_hours", "requested_hours")}))


if __name__ == "__main__":
    main()
