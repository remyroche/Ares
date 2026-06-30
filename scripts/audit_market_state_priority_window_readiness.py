#!/usr/bin/env python3
"""Preflight audit for appending market-state priority shadow windows.

This audit is intentionally cheap: it inspects candidate ledgers and existing
shadow-window manifests before the expensive market-state scoring/replay path is
run.  Its job is to prevent confounded validation, especially accidental mixing
of timestamp-rank and global-rank candidate contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_EXISTING_MANIFEST = Path(
    "data_perp/reports/market_state_priority_shadow_windows_globalrank_safegrid_lgbm_20260626_v1"
    "/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_priority_window_readiness_20260626"
)
EXPECTED_ACTIVE_HEADS = ("short_asset", "short_boll")
EXPECTED_DISABLED_HEADS = ("long_bars", "long_dist")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").astype(str)
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    return "\n".join(lines)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_manifest_path(path: Path) -> Path | None:
    artifact_root = path.parent.parent if len(path.parents) >= 2 else path.parent
    names = [
        "t1_repaired_static_baseline_manifest.json",
        "t1_anchor_scored_candidate_manifest.json",
        "live_ledger_native_materialization_manifest.json",
    ]
    for name in names:
        candidate = artifact_root / name
        if candidate.exists():
            return candidate
    manifests = sorted(artifact_root.glob("*manifest*.json"))
    return manifests[0] if manifests else None


def _read_contract(path: Path) -> dict[str, Any]:
    manifest_path = _candidate_manifest_path(path)
    payload = _load_json(manifest_path)
    active_stack = dict(payload.get("active_stack") or {})
    validation = dict(payload.get("validation") or {})
    rank_reference = dict(validation.get("rank_reference_contract") or {})
    return {
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": _sha256(manifest_path),
        "generated_by": payload.get("generated_by"),
        "rank_contract": active_stack.get("rank_contract"),
        "rank_scope": active_stack.get("rank_scope"),
        "rank_reference_run_id": active_stack.get("rank_reference_run_id")
        or rank_reference.get("eval_rank_reference_run_id"),
        "promotion_status": active_stack.get("promotion_status"),
        "policy_variant": active_stack.get("policy_variant"),
        "auction": active_stack.get("auction"),
        "enabled_heads": sorted(map(str, active_stack.get("enabled_heads") or [])),
        "disabled_heads": sorted(map(str, active_stack.get("disabled_heads") or [])),
        "qfail_active": active_stack.get("qfail_active"),
        "head_health_active": active_stack.get("head_health_active"),
        "market_state_threshold_controller_active": active_stack.get(
            "market_state_threshold_controller_active"
        ),
    }


def _read_candidate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = pq.read_schema(path).names
    need = [c for c in ["timestamp", "head", "symbol", "strategy_id", "side"] if c in columns]
    frame = pd.read_parquet(path, columns=need)
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    ts = ts.dropna()
    heads = sorted(frame["head"].dropna().astype(str).unique()) if "head" in frame else []
    key_cols = [c for c in ["timestamp", "head", "symbol", "strategy_id", "side"] if c in frame.columns]
    duplicate_keys = int(frame.duplicated(key_cols).sum()) if key_cols else None
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "timestamp_count": int(ts.nunique()),
        "start": ts.min().isoformat() if not ts.empty else None,
        "end": ts.max().isoformat() if not ts.empty else None,
        "heads": heads,
        "duplicate_decision_key_count": duplicate_keys,
        "decision_key_columns": key_cols,
        "contract": _read_contract(path),
    }


def _existing_windows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in manifest.get("windows") or []:
        candidate = dict(item.get("candidate") or {})
        rows.append(
            {
                "label": item.get("label"),
                "path": candidate.get("path"),
                "sha256": candidate.get("sha256"),
                "rows": candidate.get("rows"),
                "timestamp_count": candidate.get("timestamp_count"),
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "heads": candidate.get("heads") or [],
                "contract": candidate.get("contract") or {},
            }
        )
    return rows


def _interval_overlap(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> bool:
    a0 = pd.Timestamp(a_start) if a_start else pd.NaT
    a1 = pd.Timestamp(a_end) if a_end else pd.NaT
    b0 = pd.Timestamp(b_start) if b_start else pd.NaT
    b1 = pd.Timestamp(b_end) if b_end else pd.NaT
    if pd.isna(a0) or pd.isna(a1) or pd.isna(b0) or pd.isna(b1):
        return False
    return bool(a0 <= b1 and b0 <= a1)


def _contract_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    contract = dict(row.get("contract") or {})
    return (
        contract.get("rank_contract"),
        contract.get("rank_scope"),
        contract.get("rank_reference_run_id"),
    )


def _expected_contract(existing_manifest: dict[str, Any]) -> dict[str, Any]:
    contract = dict(existing_manifest.get("contract") or {})
    candidates = list(contract.get("candidate_rank_contracts") or [])
    first = dict(candidates[0]) if candidates else {}
    return {
        "rank_contract": first.get("rank_contract")
        or (contract.get("candidate_rank_contract_names") or [None])[0],
        "rank_scope": first.get("rank_scope")
        or (contract.get("candidate_rank_scopes") or [None])[0],
        "rank_reference_run_id": first.get("rank_reference_run_id"),
        "policy_variant": None,
    }


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    existing: list[dict[str, Any]],
    expected: dict[str, Any],
    min_timestamp_count: int,
    min_rows: int,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if int(candidate.get("rows") or 0) < int(min_rows):
        failures.append("candidate_rows_below_minimum")
    if int(candidate.get("timestamp_count") or 0) < int(min_timestamp_count):
        failures.append("timestamp_count_below_minimum")
    if sorted(candidate.get("heads") or []) != sorted(EXPECTED_ACTIVE_HEADS):
        failures.append("active_heads_not_exactly_short_asset_short_boll")
    if int(candidate.get("duplicate_decision_key_count") or 0) > 0:
        failures.append("duplicate_candidate_decision_keys")

    contract = dict(candidate.get("contract") or {})
    disabled = sorted(contract.get("disabled_heads") or [])
    if disabled and disabled != sorted(EXPECTED_DISABLED_HEADS):
        failures.append("disabled_heads_contract_mismatch")
    if contract.get("qfail_active") is True:
        failures.append("qfail_active")
    if contract.get("head_health_active") is True:
        failures.append("head_health_active")
    if contract.get("market_state_threshold_controller_active") is True:
        failures.append("market_state_threshold_controller_active")

    for field in ["rank_contract", "rank_scope", "rank_reference_run_id"]:
        expected_value = expected.get(field)
        if expected_value and contract.get(field) != expected_value:
            failures.append(f"{field}_mismatch")
    if expected.get("policy_variant") and contract.get("policy_variant") != expected["policy_variant"]:
        failures.append("policy_variant_mismatch")
    if contract.get("promotion_status") != "rank_contract_challenger":
        warnings.append("candidate_not_marked_rank_contract_challenger")

    for prior in existing:
        if candidate.get("sha256") and candidate.get("sha256") == prior.get("sha256"):
            failures.append("candidate_sha_already_evaluated")
        if _interval_overlap(candidate.get("start"), candidate.get("end"), prior.get("start"), prior.get("end")):
            failures.append("candidate_window_overlaps_existing_shadow_window")

    return ("pass" if not failures else "fail"), sorted(set(failures + warnings))


def audit_window_readiness(
    *,
    candidates: list[Path],
    existing_manifest: Path,
    output_dir: Path,
    min_timestamp_count: int = 3,
    min_rows: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(existing_manifest)
    existing = _existing_windows(manifest)
    expected = _expected_contract(manifest)
    if not expected.get("policy_variant"):
        for row in existing:
            policy_variant = (row.get("contract") or {}).get("policy_variant")
            if policy_variant:
                expected["policy_variant"] = policy_variant
                break

    rows: list[dict[str, Any]] = []
    for path in candidates:
        candidate = _read_candidate(path)
        status, reasons = _evaluate_candidate(
            candidate,
            existing=existing,
            expected=expected,
            min_timestamp_count=min_timestamp_count,
            min_rows=min_rows,
        )
        contract = dict(candidate.get("contract") or {})
        rows.append(
            {
                "path": candidate["path"],
                "sha256": candidate["sha256"],
                "status": status,
                "reasons": ";".join(reasons),
                "rows": candidate["rows"],
                "timestamp_count": candidate["timestamp_count"],
                "start": candidate["start"],
                "end": candidate["end"],
                "heads": ",".join(candidate["heads"]),
                "duplicate_decision_key_count": candidate["duplicate_decision_key_count"],
                "rank_contract": contract.get("rank_contract"),
                "rank_scope": contract.get("rank_scope"),
                "rank_reference_run_id": contract.get("rank_reference_run_id"),
                "promotion_status": contract.get("promotion_status"),
                "policy_variant": contract.get("policy_variant"),
                "manifest_path": contract.get("manifest_path"),
            }
        )

    frame = pd.DataFrame(rows)
    passed = bool(not frame.empty and frame["status"].eq("pass").all())
    summary = {
        "generated_by": "audit_market_state_priority_window_readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "candidate_count": int(len(frame)),
        "passing_candidate_count": int(frame["status"].eq("pass").sum()) if not frame.empty else 0,
        "existing_manifest": str(existing_manifest),
        "existing_manifest_sha256": _sha256(existing_manifest),
        "existing_window_count": int(len(existing)),
        "expected_contract": expected,
        "min_timestamp_count": int(min_timestamp_count),
        "min_rows": int(min_rows),
        "candidate_rows": rows,
    }
    frame.to_csv(output_dir / "market_state_priority_window_readiness.csv", index=False)
    (output_dir / "market_state_priority_window_readiness.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Market-State Priority Window Readiness Audit",
        "",
        f"Passed: `{passed}`",
        "",
        "## Expected Contract",
        "",
        _markdown_table(pd.DataFrame([expected])),
        "",
        "## Candidate Windows",
        "",
        _markdown_table(frame) if not frame.empty else "_No candidates supplied._",
        "",
        "## Interpretation",
        "",
        (
            "Only `pass` rows should be appended to the fixed safe-grid shadow run. "
            "Failures indicate stale, overlapping, mixed-rank-contract, or otherwise "
            "non-comparable ledgers."
        ),
        "",
    ]
    (output_dir / "market_state_priority_window_readiness_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, default=DEFAULT_EXISTING_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-timestamp-count", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = audit_window_readiness(
        candidates=list(args.candidate),
        existing_manifest=args.existing_manifest,
        output_dir=args.output_dir,
        min_timestamp_count=int(args.min_timestamp_count),
        min_rows=int(args.min_rows),
    )
    print(json.dumps(_json_safe(summary), indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
