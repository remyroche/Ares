#!/usr/bin/env python3
"""Consume an immutable strict-R3 live-candle report into a supervisor receipt.

This is deliberately read-only with respect to the live trader: it never
refreshes market data, computes a score, changes live state, calls Kraken, or
restarts services.  It provides the durable input for the operational review
loop: every producer report is classified once, including incident attempts
and later same-hour tested successors.
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runtime-tag", required=True)
    args = parser.parse_args()

    report_path = args.report.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text())
    if report.get("schema") != "strict_r3_live_candle_report_v2":
        raise ValueError("unknown strict-R3 live-candle report schema")
    decision = _utc(report["decision_ts"])
    if str(args.runtime_tag) not in {"any", "*"} and str(report.get("runtime_tag")) != str(args.runtime_tag):
        raise ValueError("report runtime tag does not match supervisor runtime")

    report_sha = _sha(report_path)
    attempt = report_path.stem.rsplit("_", 1)[-1]
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    out_dir = ARTIFACTS / (
        f"strict_r3_live_operations_supervisor_{report.get('runtime_tag')}_{tag}_{attempt}"
    )
    output = out_dir / "run_manifest.json"
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get("report_sha256") != report_sha:
            raise ValueError("immutable supervisor receipt conflicts with report hash")
        print(json.dumps(existing, sort_keys=True))
        return

    report_status = str(report.get("status") or "")
    irregularities = [str(value) for value in report.get("irregularities") or []]
    terminal = report_status == "pass"
    if terminal:
        disposition = "observe_next_candle"
    else:
        disposition = "investigate_root_cause_before_any_same-hour_successor"
    receipt = {
        "schema": "strict_r3_live_operations_supervisor_v1",
        "runtime_tag": str(report.get("runtime_tag")),
        "decision_ts": decision.isoformat(),
        "report": _rel(report_path),
        "report_sha256": report_sha,
        "report_status": report_status,
        "terminal": terminal,
        "disposition": disposition,
        "irregularities": irregularities,
        "funnel": dict(report.get("funnel") or {}),
        "producer_receipt": report.get("producer_receipt"),
        "hourly_run": report.get("hourly_run"),
        "read_only": True,
        "exchange_calls": 0,
        "live_state_mutations": 0,
    }
    _atomic_json(output, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
