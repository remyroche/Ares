#!/usr/bin/env python3
"""Recover persisted entry-time execution-adjusted predictions, fail closed.

Pre-telemetry strict-R3 live trades retained the exact execution economics in
the immutable producer ``execution.log`` but not in the close ledger.  This
read-only utility creates a separate sidecar mapping a ledger key to that
persisted entry-time prediction.  It does *not* reconstruct a score, rerun a
model, or use an outcome.  A match requires candidate id and actual fill time
to agree with the close ledger; conflicting matching receipts are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_execution_prediction_recovery_sidecar_v1"


def _utc(value: object) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _finite(value: object) -> float | None:
    result = pd.to_numeric(value, errors="coerce")
    return float(result) if np.isfinite(result) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ledger_targets(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = ledger.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("close-notification ledger lacks records")
    targets: list[dict[str, Any]] = []
    for record_key, record in records.items():
        if not isinstance(record, Mapping):
            continue
        telemetry = record.get("trade_telemetry")
        entry = telemetry.get("entry") if isinstance(telemetry, Mapping) else None
        fill_time = _utc(entry.get("entry_fill_time")) if isinstance(entry, Mapping) else None
        candidate_id = str(record.get("candidate_id") or "")
        if candidate_id and fill_time is not None:
            targets.append({
                "record_key": str(record_key),
                "candidate_id": candidate_id,
                "symbol": record.get("symbol"),
                "entry_fill_time": fill_time,
            })
    return targets


def _actions_from_receipt(path: Path) -> Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    """Yield (line number, top-level receipt, entry action) for JSON log lines."""
    actions: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                receipt = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(receipt, Mapping):
                continue
            for action in receipt.get("actions") or []:
                if isinstance(action, Mapping) and str(action.get("action") or "") == "entry":
                    actions.append((number, receipt, action))
    return actions


def recover(
    *,
    ledger: Mapping[str, Any],
    receipt_paths: Sequence[Path],
    fill_tolerance_seconds: int = 2,
) -> dict[str, Any]:
    """Recover only exact persisted action economics for each ledger trade."""
    if fill_tolerance_seconds < 0:
        raise ValueError("fill_tolerance_seconds must be non-negative")
    targets = _ledger_targets(ledger)
    by_candidate = {str(row["candidate_id"]): row for row in targets}
    matches: dict[str, list[dict[str, Any]]] = {str(row["record_key"]): [] for row in targets}
    scanned: list[dict[str, Any]] = []
    for path in sorted({Path(path) for path in receipt_paths}):
        if not path.is_file():
            continue
        source_hash = _sha256(path)
        scanned.append({"path": str(path), "sha256": source_hash})
        for line_number, receipt, action in _actions_from_receipt(path):
            target = by_candidate.get(str(action.get("candidate_id") or ""))
            if target is None:
                continue
            fill_time = _utc(action.get("actual_entry_fill_ts"))
            if fill_time is None or abs(fill_time - target["entry_fill_time"]) > pd.Timedelta(seconds=fill_tolerance_seconds):
                continue
            economics = action.get("execution_economics")
            adjusted = (
                _finite(economics.get("execution_adjusted_expected_net_bps"))
                if isinstance(economics, Mapping) else None
            )
            if adjusted is None:
                continue
            matches[str(target["record_key"])].append({
                "execution_adjusted_expected_net_bps": adjusted,
                "actual_entry_fill_ts": fill_time.isoformat(),
                "entry_order_id": action.get("entry_order_id"),
                "decision_ts": receipt.get("decision_ts"),
                "receipt_path": str(path),
                "receipt_sha256": source_hash,
                "receipt_line": int(line_number),
                "execution_economics": dict(economics),
                "inference_bundle_sha256": receipt.get("inference_bundle_sha256"),
                "exit_policy_sha256": receipt.get("exit_policy_sha256"),
            })

    rows: list[dict[str, Any]] = []
    for target in targets:
        record_key = str(target["record_key"])
        raw = matches[record_key]
        identities = {
            (
                item["actual_entry_fill_ts"],
                str(item.get("entry_order_id") or ""),
                round(float(item["execution_adjusted_expected_net_bps"]), 10),
            )
            for item in raw
        }
        row = {
            "record_key": record_key,
            "candidate_id": target["candidate_id"],
            "symbol": target.get("symbol"),
            "ledger_entry_fill_time": target["entry_fill_time"].isoformat(),
            "status": "unconfirmed",
            "reason": None,
            "matches": raw,
        }
        if not raw:
            row["reason"] = "no_exact_persisted_entry_economics_match"
        elif len(identities) != 1:
            row["reason"] = "conflicting_persisted_entry_economics_matches"
        else:
            row["status"] = "confirmed"
            row["reason"] = "exact_candidate_and_actual_fill_match"
            row["execution_adjusted_expected_net_bps"] = raw[0][
                "execution_adjusted_expected_net_bps"
            ]
        rows.append(row)
    return {
        "schema": SCHEMA,
        "source": {
            "source_type": "immutable_local_producer_receipts",
            "outcome_or_label_input": False,
            "fill_tolerance_seconds": int(fill_tolerance_seconds),
            "receipt_count_scanned": len(scanned),
            "receipts": scanned,
        },
        "confirmed_prediction_count": sum(row["status"] == "confirmed" for row in rows),
        "unconfirmed_prediction_count": sum(row["status"] != "confirmed" for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fill-tolerance-seconds", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("prediction sidecar path must be immutable")
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    receipts = list(args.receipt_root.rglob("execution.log"))
    result = recover(
        ledger=ledger,
        receipt_paths=receipts,
        fill_tolerance_seconds=args.fill_tolerance_seconds,
    )
    result["ledger_sha256"] = _sha256(args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "receipt_count_scanned": result["source"]["receipt_count_scanned"],
        "confirmed_prediction_count": result["confirmed_prediction_count"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
