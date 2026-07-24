#!/usr/bin/env python3
"""Read-only provenance auditor for historical Pack-B versus locked DEC-09.

This program never loads models, parquet data, feature stores, or state objects.
It reports whether an old Pack-B artifact can be considered only as a historical
comparator under DEC-09; it does not regenerate, rescore, train, or publish OOF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS = ROOT / "config/full_pipeline_decisions_20260724.json"
DEFAULT_PACKB_REPORT = ROOT / (
    "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1"
)
DEFAULT_LABELS = ROOT / (
    "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
EXPECTED_OUTER_FOLDS = (
    ("2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),
    ("2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    ("2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("2026-07-01T00:00:00Z", "2026-07-11T00:00:00Z"),
)
EXPECTED_HORIZONS = {
    "packb_directional_base": "decision_timestamp + 24 hours (96 x 15-minute causal path bars)",
    "auxiliary_catboost_execution_ev_and_timing": "decision_timestamp + 12 hours",
}
EXPECTED_AUTHORIZATION = (
    "decision_timestamp < validation_start AND actual_stage_label_end < validation_start "
    "AND decision_timestamp < validation_start - 24 hours"
)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _fold_start(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or "_" not in value:
        raise ValueError(f"{name} must be YYYY-MM-DD_YYYY-MM-DD")
    return _utc(f"{value.split('_', 1)[0]}T00:00:00+00:00", name)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def parse_dec09(decisions_path: Path) -> Mapping[str, Any]:
    """Validate the immutable DEC-09 fields this audit relies on."""
    payload = _read_json(decisions_path)
    if payload.get("schema_version") != "full_pipeline_decisions_v1":
        raise ValueError("expected full_pipeline_decisions_v1")
    if payload.get("status") != "LOCKED_BEFORE_NEW_TRAINING":
        raise ValueError("DEC-09 decisions must be locked before new training")
    decisions = payload.get("decisions")
    decision = decisions.get("DEC-09") if isinstance(decisions, dict) else None
    if not isinstance(decision, dict):
        raise ValueError("decisions.DEC-09 is required")
    if decision.get("decision_timestamp") != "signal_timestamp + 1 hour":
        raise ValueError("DEC-09 decision timestamp contract changed")
    if decision.get("stage_label_horizons") != EXPECTED_HORIZONS:
        raise ValueError("DEC-09 stage label horizons changed")
    if decision.get("train_authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("DEC-09 train authorization changed")
    if decision.get("signal_timestamp_purge_hours") != 25:
        raise ValueError("DEC-09 signal_timestamp_purge_hours must be 25")
    if (
        decision.get("purge_hours") != 12
        or decision.get("post_label_embargo_hours") != 12
    ):
        raise ValueError("DEC-09 purge/embargo contract changed")
    outer_folds = tuple(
        tuple(fold)
        for fold in decision.get("outer_folds", ())
        if isinstance(fold, list)
    )
    if outer_folds != EXPECTED_OUTER_FOLDS:
        raise ValueError("DEC-09 outer folds changed")
    cutoff = _utc(
        decision.get("feature_selection_hpo_resolution_cutoff_utc"), "resolution cutoff"
    )
    return {
        "decision": decision,
        "cutoff_utc": cutoff,
        "sha256": _sha256(decisions_path),
    }


def _check_before_cutoff(
    name: str, value: Any, cutoff: datetime, parser: Any
) -> Mapping[str, Any]:
    try:
        resolved = parser(value, name)
    except ValueError as exc:
        return {"name": name, "pass": False, "value": value, "reason": str(exc)}
    passed = resolved < cutoff
    return {
        "name": name,
        "pass": passed,
        "value": value,
        "resolved_at_utc": _iso(resolved),
        "cutoff_utc": _iso(cutoff),
        "reason": "strictly_before_cutoff"
        if passed
        else "at_or_after_dec09_resolution_cutoff",
    }


def _label_inventory(labels_path: Path) -> Mapping[str, Any]:
    manifest = labels_path / "labels_manifest.json"
    if not manifest.is_file():
        return {
            "available": False,
            "path": str(manifest),
            "reason": "labels_manifest_missing",
        }
    payload = _read_json(manifest)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        return {
            "available": False,
            "path": str(manifest),
            "reason": "labels_manifest_datasets_missing",
        }
    try:
        rows = sum(
            int(item["rows"]) for item in datasets.values() if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "path": str(manifest),
            "reason": f"invalid_dataset_rows: {exc}",
        }
    tail = labels_path / "label_tail_append_2026_07.json"
    return {
        "available": True,
        "path": str(manifest),
        "sha256": _sha256(manifest),
        "run_id": payload.get("run_id"),
        "shard_count": len(datasets),
        "rows": rows,
        "tail_append_manifest": str(tail) if tail.is_file() else None,
        "tail_append_sha256": _sha256(tail) if tail.is_file() else None,
    }


def audit_historical_packb(
    decisions_path: Path = DEFAULT_DECISIONS,
    packb_report_dir: Path = DEFAULT_PACKB_REPORT,
    labels_path: Path = DEFAULT_LABELS,
) -> Mapping[str, Any]:
    """Audit one historical Pack-B report without opening any model or parquet file."""
    dec09 = parse_dec09(decisions_path)
    report_path = packb_report_dir / "manifest.json"
    state_path = (
        packb_report_dir
        / "models/final_all_rows/ae_gmm_state/source_state_manifest.json"
    )
    report, state = _read_json(report_path), _read_json(state_path)
    cutoff = dec09["cutoff_utc"]
    checks = [
        _check_before_cutoff(
            "feature_selection_calibration_fold_start",
            report.get("feature_selection_calibration_fold"),
            cutoff,
            _fold_start,
        ),
        _check_before_cutoff(
            "hpo_calibration_fold_start",
            report.get("hpo_calibration_fold"),
            cutoff,
            _fold_start,
        ),
        _check_before_cutoff(
            "ae_gmm_cycle_reference_end", state.get("cycle_reference_end"), cutoff, _utc
        ),
    ]
    scope = report.get("model_side_scope")
    checks.append(
        {
            "name": "model_side_scope_is_per_side",
            "pass": scope == "per_side",
            "value": scope,
            "reason": "per_side_required"
            if scope == "per_side"
            else "pooled_or_unknown_side_scope",
        }
    )
    inventory = _label_inventory(labels_path)
    report_rows = report.get("rows")
    rows_match = inventory.get("available") and report_rows == inventory.get("rows")
    checks.append(
        {
            "name": "label_shard_inventory_matches_report_rows",
            "pass": bool(rows_match),
            "report_rows": report_rows,
            "inventory_rows": inventory.get("rows"),
            "reason": "matches"
            if rows_match
            else "stale_or_unverifiable_label_shard_inventory",
        }
    )
    blockers = [check for check in checks if not check["pass"]]
    return {
        "schema_version": "packb_dec09_historical_comparator_audit_v1",
        "status": "BLOCKED_HISTORICAL_COMPARATOR_ONLY"
        if blockers
        else "HISTORICAL_COMPARATOR_PROVENANCE_COMPLETE",
        "canonical_oof_reuse_allowed": False,
        "purpose": "read_only_provenance_audit_not_regeneration_or_scoring",
        "dec09": {
            "decisions_path": str(decisions_path),
            "decisions_sha256": dec09["sha256"],
            "resolution_cutoff_utc": _iso(cutoff),
            "outer_folds": [list(fold) for fold in EXPECTED_OUTER_FOLDS],
        },
        "sources": {
            "packb_manifest": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
            "ae_gmm_source_state_manifest": {
                "path": str(state_path),
                "sha256": _sha256(state_path),
            },
            "label_inventory": inventory,
        },
        "checks": checks,
        "blockers": blockers,
    }


def run(**kwargs: Any) -> Mapping[str, Any]:
    """Compatibility entry point; deliberately limited to the read-only audit."""
    return audit_historical_packb(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--packb-report", type=Path, default=DEFAULT_PACKB_REPORT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, help="optional JSON destination")
    args = parser.parse_args(argv)
    try:
        audit = audit_historical_packb(args.decisions, args.packb_report, args.labels)
    except ValueError as exc:
        print(
            json.dumps({"status": "AUDIT_INPUT_ERROR", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 1
    text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
