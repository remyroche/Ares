#!/usr/bin/env python3
"""Validate GMM smoke candidates before train_base -> train_meta handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_REPORT_DIR = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")
LEARNABILITY_CHECK_FILE = "gmm_train_base_learnability_check.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if pd.isna(value):
        return None
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def build_readiness_check(report_dir: Path) -> dict[str, Any]:
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    viability = _read_csv(report_dir / "gmm_label_viability_matrix.csv")
    readiness = _read_csv(report_dir / "gmm_train_meta_readiness.csv")

    errors: list[str] = []
    warnings: list[str] = []
    active_mask = _bool_column(viability, "active_label")
    active = viability[active_mask].copy()
    manifest_active = int((manifest.get("label_viability_matrix") or {}).get("active_rows", -1))
    readiness_rows = int(len(readiness))
    active_rows = int(len(active))
    if active_rows <= 0:
        errors.append("no_active_viability_rows")
    if readiness_rows != active_rows:
        errors.append(f"readiness_rows_mismatch:{readiness_rows}!={active_rows}")
    if manifest_active != active_rows:
        errors.append(f"manifest_active_rows_mismatch:{manifest_active}!={active_rows}")

    if readiness_rows:
        if "is_final_promotion_ready" in readiness.columns:
            final_ready = readiness["is_final_promotion_ready"].astype(bool)
            if bool(final_ready.any()):
                errors.append("readiness_marks_final_promotion_ready_before_required_next_checks")
        required = set()
        if "required_next_checks" in readiness.columns:
            for raw in readiness["required_next_checks"].dropna().astype(str):
                required.update(part for part in raw.split(";") if part)
        expected = {
            "train_base_oof_learnability",
            "train_meta_oos_profitability",
            "simple_policy_optimiser_exit_policy",
            "frozen_threshold_replay",
            "leakage_and_feature_parity_audit",
        }
        missing = sorted(expected - required)
        if missing:
            errors.append(f"missing_required_next_checks:{','.join(missing)}")

    next_stage_plan = [
        "train_base_oof_learnability",
        "train_meta_oos_profitability",
        "simple_policy_optimiser_exit_policy",
        "frozen_threshold_replay",
        "leakage_and_feature_parity_audit",
    ]
    learnability_path = report_dir / LEARNABILITY_CHECK_FILE
    learnability_check = _read_json_if_exists(learnability_path)
    passed_next_checks: list[str] = []
    failed_next_checks: list[str] = []
    if learnability_check is not None:
        learnability_status = str(learnability_check.get("status", "unknown"))
        if learnability_status == "pass":
            passed_next_checks.append("train_base_oof_learnability")
        elif learnability_status in {
            "candidate_for_train_meta_risk_filter_smoke",
            "candidate_for_train_meta_path_filter_smoke",
        }:
            passed_next_checks.append("train_base_oof_learnability")
            warnings.append("train_base_final_policy_readiness_failed_meta_filter_required")
        else:
            failed_next_checks.append("train_base_oof_learnability")
            errors.append(f"learnability_check_failed:{learnability_status}")
    pending_next_checks = [
        check for check in next_stage_plan if check not in set(passed_next_checks + failed_next_checks)
    ]

    failed_s15_s16 = viability[
        viability["cluster_policy"].astype(str).str.startswith(("s15_", "s16_"))
        & active_mask
    ]
    if not failed_s15_s16.empty:
        errors.append("s15_or_s16_unexpectedly_active")
    comparator_failures = viability[
        viability["cluster_policy"].astype(str).str.startswith(("s15_", "s16_"))
        & viability["first_failed_gate"].astype(str).ne("pass")
    ]
    if comparator_failures.empty:
        warnings.append("no_failed_s15_s16_comparators_found")

    candidate_records = readiness.to_dict(orient="records")
    source_paths = {}
    if candidate_records:
        row0 = candidate_records[0]
        source_paths = {
            "labels_path": row0.get("labels_path"),
            "feature_dir": row0.get("feature_dir"),
            "feature_list_csv": row0.get("feature_list_csv"),
        }

    if errors and failed_next_checks:
        status = "not_ready_for_train_meta_profitability"
    elif errors or readiness_rows <= 0:
        status = "not_ready"
    elif (
        learnability_check is not None
        and learnability_check.get("status")
        in {
            "candidate_for_train_meta_risk_filter_smoke",
            "candidate_for_train_meta_path_filter_smoke",
        }
    ):
        status = "candidate_for_train_meta_path_filter_smoke"
    elif "train_base_oof_learnability" in passed_next_checks:
        status = "candidate_for_train_meta_profitability_smoke"
    else:
        status = "candidate_for_train_base_meta_smoke"
    return {
        "status": status,
        "report_dir": str(report_dir),
        "manifest": str(manifest_path),
        "active_rows": active_rows,
        "readiness_rows": readiness_rows,
        "manifest_active_rows": manifest_active,
        "candidate_selectors": sorted(readiness["cluster_policy"].astype(str).unique().tolist())
        if "cluster_policy" in readiness.columns
        else [],
        "candidate_records": candidate_records,
        "comparator_failures": comparator_failures[
            [
                col
                for col in (
                    "cluster_policy",
                    "top_frac",
                    "label_viability_score",
                    "first_failed_gate",
                    "mean_u",
                    "bad_mae_1r_rate",
                    "max_month_bad_mae_1r_rate",
                    "timeout_rate",
                    "max_month_timeout_rate",
                    "final_stage_oracle_recall_mean",
                    "final_stage_june_oracle_recall",
                )
                if col in comparator_failures.columns
            ]
        ].to_dict(orient="records"),
        "source_paths": source_paths,
        "required_next_checks": (manifest.get("train_base_meta_readiness") or {}).get(
            "required_next_checks",
            [],
        ),
        "next_stage_plan": next_stage_plan,
        "passed_next_checks": passed_next_checks,
        "failed_next_checks": failed_next_checks,
        "pending_next_checks": pending_next_checks,
        "learnability_check": learnability_check,
        "base_gate_split": {
            "candidate_readiness": (
                learnability_check.get("gate_1a_train_base_candidate_readiness")
                if learnability_check
                else None
            ),
            "final_policy_readiness": (
                learnability_check.get("gate_1b_train_base_final_policy_readiness")
                if learnability_check
                else None
            ),
        },
        "meta_filter_handoff": (
            learnability_check.get("meta_filter_handoff") if learnability_check else None
        ),
        "learnability_check_path": str(learnability_path),
        "final_promotion_ready": False,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_readiness_check(args.report_dir)
    output = args.output or (args.report_dir / "gmm_train_base_meta_readiness_check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0 if report["status"] in {
        "candidate_for_train_base_meta_smoke",
        "candidate_for_train_meta_profitability_smoke",
        "candidate_for_train_meta_path_filter_smoke",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
