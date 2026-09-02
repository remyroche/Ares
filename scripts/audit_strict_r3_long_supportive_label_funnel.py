#!/usr/bin/env python3
"""Audit immutable identity, chronology, and label separation for the long funnel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_CAUSAL_PATH_OUTPUTS = {"path_entropy", "path_max_probability"}
FORBIDDEN_FEATURE_TOKENS = ("supportive_", "path_arch_", "future_", "policy_net_bps", "h12_")


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _identity(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["candidate_id", "__decision_ts__", "fold", "cohort"]].reset_index(drop=True)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    predictions = json.loads((root / "stage1_oof_predictions_manifest.json").read_text())
    parts_root = Path(predictions["root"])
    if not parts_root.is_absolute():
        parts_root = (root / parts_root).resolve()
    by_fold: dict[str, list[Path]] = {}
    for relative in predictions["parts"]:
        path = parts_root / relative
        by_fold.setdefault(path.parent.name, []).append(path)
    checks: list[dict[str, Any]] = []
    outer = {item["name"]: item for item in manifest["outer_folds"]}
    per_fold: list[dict[str, Any]] = []
    all_columns: set[str] = set()
    for folder, paths in sorted(by_fold.items()):
        paths = sorted(paths)
        base = pd.read_parquet(paths[0], columns=["candidate_id", "__decision_ts__", "fold", "cohort"])
        reference = _identity(base)
        fold_name = str(reference["fold"].iloc[0]) if len(reference) else ""
        expected = outer.get(fold_name)
        start = pd.Timestamp(expected["start"]) if expected else pd.NaT
        end = pd.Timestamp(expected["end_exclusive"]) if expected else pd.NaT
        row = {
            "partition": folder,
            "fold": fold_name,
            "parts": int(len(paths)),
            "rows": int(len(reference)),
            "candidate_id_unique": bool(not reference["candidate_id"].duplicated().any()),
            "decision_range_valid": bool(reference["__decision_ts__"].ge(start).all() and reference["__decision_ts__"].lt(end).all()) if expected else False,
            "all_arm_identity_exact": True,
        }
        for path in paths:
            frame = pd.read_parquet(path)
            all_columns.update(frame.columns)
            same = _identity(frame).equals(reference)
            if not same:
                row["all_arm_identity_exact"] = False
        per_fold.append(row)
    checks.append(_check("six_outer_folds_present", len(per_fold) == len(outer), {"found": len(per_fold), "expected": len(outer)}))
    checks.append(_check("exactly_34_arms_per_fold", all(item["parts"] == 34 for item in per_fold), [item["parts"] for item in per_fold]))
    checks.append(_check("candidate_ids_unique_per_fold", all(item["candidate_id_unique"] for item in per_fold), per_fold))
    checks.append(_check("held_identity_exact_across_arms", all(item["all_arm_identity_exact"] for item in per_fold), per_fold))
    checks.append(_check("held_decisions_inside_declared_fold", all(item["decision_range_valid"] for item in per_fold), per_fold))
    illegal_manifest = [
        name for name in manifest.get("causal_features", [])
        if any(token in str(name).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    checks.append(_check("causal_feature_contract_excludes_future_labels", not illegal_manifest, illegal_manifest))
    forbidden_prediction = [
        name for name in all_columns
        if any(token in str(name).lower() for token in ("supportive_", "path_arch_", "future_", "h12_"))
    ]
    checks.append(_check("prediction_parts_exclude_raw_future_targets", not forbidden_prediction, forbidden_prediction))
    fold_audit = pd.read_parquet(root / "stage1_fold_audit.parquet")
    checks.append(_check("all_supervised_folds_strictly_completed", bool(fold_audit["status"].eq("ok").all()), fold_audit.to_dict("records")))
    if {"train_label_cutoff", "embargo_hours"}.issubset(fold_audit.columns):
        checks.append(_check("h12_embargo_recorded", bool(pd.to_numeric(fold_audit["embargo_hours"], errors="coerce").eq(12).all()), fold_audit[["fold", "train_label_cutoff", "embargo_hours"]].to_dict("records")))
    result = {
        "schema": "strict_r3_long_supportive_label_funnel_stage1_audit_v1",
        "root": str(root),
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
        "per_fold": per_fold,
        "prediction_columns": sorted(all_columns),
    }
    return result


def run_stage2(root: Path) -> dict[str, Any]:
    """Audit stable-geometry Stage 2 without treating its P1 outputs as labels."""
    root = root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    geometry_mode = str(manifest.get("geometry_mode", "frozen"))
    geometry = json.loads((root / "frozen_geometry_contract.json").read_text()) if geometry_mode == "frozen" else None
    predictions = json.loads((root / "stage2_oof_predictions_manifest.json").read_text())
    parts_root = Path(predictions["root"])
    if not parts_root.is_absolute():
        parts_root = (root / parts_root).resolve()
    by_fold: dict[str, list[Path]] = {}
    for relative in predictions["parts"]:
        path = parts_root / relative
        by_fold.setdefault(path.parent.name, []).append(path)
    outer = {item["name"]: item for item in manifest["outer_folds"]}
    per_fold: list[dict[str, Any]] = []
    columns: set[str] = set()
    for folder, paths in sorted(by_fold.items()):
        paths = sorted(paths)
        base = pd.read_parquet(paths[0], columns=["candidate_id", "__decision_ts__", "fold", "cohort"])
        reference = _identity(base)
        fold = str(reference["fold"].iloc[0])
        expected = outer.get(fold)
        start = pd.Timestamp(expected["start"]) if expected else pd.NaT
        end = pd.Timestamp(expected["end_exclusive"]) if expected else pd.NaT
        exact = True
        for path in paths:
            frame = pd.read_parquet(path)
            columns.update(frame.columns)
            exact &= _identity(frame).equals(reference)
        per_fold.append({
            "partition": folder,
            "fold": fold,
            "parts": len(paths),
            "rows": len(reference),
            "candidate_id_unique": bool(not reference["candidate_id"].duplicated().any()),
            "identity_exact": bool(exact),
            "decision_range_valid": bool(reference["__decision_ts__"].ge(start).all() and reference["__decision_ts__"].lt(end).all()),
        })
    predictor_illegal = [
        name for name in manifest.get("predictor_contract", [])
        if any(token in str(name).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    # ``frozen_path_p_*`` is a causal recogniser output, not a realised path
    # coordinate.  All raw supportive/path-architecture columns remain banned.
    prediction_illegal = [
        name for name in columns
        if any(token in str(name).lower() for token in ("supportive_", "path_arch_", "future_", "h12_"))
    ]
    fold_audit = pd.read_parquet(root / "stage2_fold_audit.parquet")
    if geometry_mode == "frozen":
        geometry_checks = [
            _check("frozen_geometry_covers_october_to_december", set(geometry.get("geometry_month_rows", {})) == {"2024-10", "2024-11", "2024-12"}, geometry.get("geometry_month_rows")),
            _check("geometry_precedes_supervised_population", pd.Timestamp(geometry["geometry_definition_end_exclusive"]) <= pd.Timestamp("2025-01-01T00:00:00Z"), geometry),
        ]
        geometry_detail: Any = geometry
    else:
        geometry_by_fold = list(manifest.get("geometry_by_fold", []))
        geometry_checks = [
            _check("rolling_geometry_has_one_bundle_per_fold", len(geometry_by_fold) == len(per_fold), geometry_by_fold),
            _check(
                "rolling_geometry_definition_precedes_same_bundle_training",
                all(pd.Timestamp(item["geometry_definition_end_exclusive"]) <= pd.Timestamp(item["supervised_start"]) for item in geometry_by_fold),
                geometry_by_fold,
            ),
            _check(
                "rolling_geometry_definition_spans_three_complete_months",
                all(len(item.get("geometry_month_rows", {})) == 3 for item in geometry_by_fold),
                geometry_by_fold,
            ),
        ]
        geometry_detail = geometry_by_fold
    checks = [
        _check("six_outer_folds_present", len(per_fold) == len(outer), {"found": len(per_fold), "expected": len(outer)}),
        _check("exactly_three_predeclared_arms_per_fold", all(item["parts"] == 3 for item in per_fold), [item["parts"] for item in per_fold]),
        _check("held_identity_exact_across_stage2_arms", all(item["identity_exact"] and item["candidate_id_unique"] for item in per_fold), per_fold),
        _check("held_decisions_inside_declared_fold", all(item["decision_range_valid"] for item in per_fold), per_fold),
        *geometry_checks,
        _check("geometry_definition_excluded_from_supervised_training", bool(manifest.get("geometry_definition_rows_excluded_from_supervised_training")), manifest.get("geometry_definition_rows_excluded_from_supervised_training")),
        _check("stage2_predictor_contract_excludes_future_labels", not predictor_illegal, predictor_illegal),
        _check("stage2_predictions_exclude_raw_future_targets", not prediction_illegal, prediction_illegal),
        _check("all_stage2_folds_completed", bool(fold_audit["status"].eq("ok").all()), fold_audit.to_dict("records")),
        _check("shared_residual_uses_inner_oof_rows", bool(fold_audit["residual_status"].eq("ok").all() and pd.to_numeric(fold_audit["residual_oof_rows"], errors="coerce").gt(0).all()), fold_audit[["fold", "residual_status", "residual_oof_rows"]].to_dict("records")),
    ]
    return {
        "schema": "strict_r3_long_supportive_label_stage2_audit_v1",
        "root": str(root),
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
        "per_fold": per_fold,
        "geometry_mode": geometry_mode,
        "geometry_detail": geometry_detail,
        "prediction_columns": sorted(columns),
    }


def run_stage3(root: Path) -> dict[str, Any]:
    """Audit the no-refit direct-support bps blend artifacts."""
    root = root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    parts_root = root / "oof_prediction_parts"
    by_fold: dict[str, list[Path]] = {}
    for path in sorted(parts_root.rglob("*.parquet")):
        by_fold.setdefault(path.parent.name, []).append(path)
    per_fold: list[dict[str, Any]] = []
    columns: set[str] = set()
    for folder, paths in sorted(by_fold.items()):
        base = pd.read_parquet(paths[0])
        reference = _identity(base)
        exact = True
        for path in paths:
            frame = pd.read_parquet(path)
            columns.update(frame.columns)
            exact &= _identity(frame).equals(reference)
        per_fold.append({
            "partition": folder,
            "fold": str(reference["fold"].iloc[0]),
            "parts": len(paths),
            "rows": len(reference),
            "candidate_id_unique": bool(not reference["candidate_id"].duplicated().any()),
            "identity_exact": bool(exact),
        })
    raw_future = [name for name in columns if any(token in str(name).lower() for token in ("supportive_", "path_arch_", "future_", "h12_"))]
    fold_audit = pd.read_parquet(root / "stage3_fold_audit.parquet")
    checks = [
        _check("six_outer_folds_present", len(per_fold) == 6, {"found": len(per_fold)}),
        _check("exactly_four_predeclared_blends_per_fold", all(item["parts"] == 4 for item in per_fold), [item["parts"] for item in per_fold]),
        _check("blend_inputs_have_exact_identity", all(item["candidate_id_unique"] and item["identity_exact"] for item in per_fold), per_fold),
        _check("blend_prediction_parts_exclude_raw_future_targets", not raw_future, raw_future),
        _check("all_blend_folds_completed", bool(fold_audit["status"].eq("ok").all()), fold_audit.to_dict("records")),
        _check("source_lineage_is_sealed_stage1_oof", "stage1" in str(manifest.get("source_stage1", "")) and bool(manifest.get("source_stage1_parquet_sha256")), manifest.get("source_stage1")),
    ]
    return {
        "schema": "strict_r3_long_direct_support_blends_audit_v1",
        "root": str(root),
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
        "per_fold": per_fold,
        "prediction_columns": sorted(columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--stage3", action="store_true")
    args = parser.parse_args()
    if args.stage2 and args.stage3:
        raise SystemExit("select at most one of --stage2 / --stage3")
    result = run_stage2(args.root) if args.stage2 else run_stage3(args.root) if args.stage3 else run(args.root)
    payload = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        args.out.write_text(payload)
    print(payload)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
