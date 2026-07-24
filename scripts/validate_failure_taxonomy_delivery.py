#!/usr/bin/env python3
"""Validate three-year failure taxonomy and prospective detector artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    is_batch_layout_dependent_ae_gmm_feature,
)

OUTCOME_CONTRACT = "hourly_close_policy_proxy_v2_activation_deadline"
FORBIDDEN_FEATURE_PREFIXES = (
    "availability__",
    "expost__",
    "target__",
    "label__",
    "outcome__",
    "future__",
    "realized__",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_mean(frame: pd.DataFrame, name: str) -> float:
    if name not in frame:
        return np.nan
    values = pd.to_numeric(frame[name], errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def run(args: argparse.Namespace) -> dict[str, Any]:
    backcast = Path(args.backcast)
    taxonomy = Path(args.taxonomy)
    prospective = Path(args.prospective)
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}

    monthly_manifests = sorted((backcast / "monthly").glob("*/manifest.json"))
    monthly = [_json(path) for path in monthly_manifests]
    checks["backcast_months_present"] = bool(monthly)
    checks["backcast_outcome_contract"] = bool(monthly) and all(
        item.get("outcome_contract_version") == OUTCOME_CONTRACT
        and int(item.get("policy_bar_minutes", 0)) == 15
        for item in monthly
    )
    checks["backcast_cost_once"] = bool(monthly) and all(
        bool(item.get("cost_counted_once"))
        and np.isclose(float(item.get("round_trip_cost", np.nan)), 0.01)
        for item in monthly
    )
    checks["backcast_path_coverage"] = bool(monthly) and all(
        item.get("path_stats")
        and all(
            float(value.get("coverage", 0.0)) >= float(args.min_path_coverage)
            for value in item["path_stats"].values()
        )
        for item in monthly
    )
    metrics["backcast_months"] = len(monthly)
    metrics["required_min_path_coverage"] = float(args.min_path_coverage)
    metrics["backcast_rows"] = int(sum(int(item.get("rows", 0)) for item in monthly))
    metrics["backcast_start"] = min(
        (str(item.get("start")) for item in monthly), default=""
    )
    metrics["backcast_end_exclusive"] = max(
        (str(item.get("end_exclusive")) for item in monthly), default=""
    )
    path_coverages = [
        float(value.get("coverage", np.nan))
        for item in monthly
        for value in (item.get("path_stats") or {}).values()
        if np.isfinite(float(value.get("coverage", np.nan)))
    ]
    metrics["minimum_path_coverage"] = (
        float(min(path_coverages)) if path_coverages else np.nan
    )
    metrics["path_coverage_below_90pct_count"] = int(
        sum(value < 0.90 for value in path_coverages)
    )

    taxonomy_manifest = _json(taxonomy / "manifest.json")
    coverage = pd.read_csv(taxonomy / "negative_pnl_day_coverage.csv")
    negative_modes = pd.read_csv(taxonomy / "negative_pnl_day_failure_modes.csv")
    checks["three_year_span"] = bool(
        taxonomy_manifest.get("source", {}).get("three_year_coverage_pass")
    )
    checks["every_negative_day_in_parent_episode"] = bool(
        len(coverage)
        and coverage["covered_by_parent_episode"].fillna(False).astype(bool).all()
    )
    checks["every_negative_day_has_parent_mode"] = bool(
        len(negative_modes)
        and negative_modes["parent_mode_assigned"].fillna(False).astype(bool).all()
    )
    checks["negative_day_catalog_complete"] = len(coverage) == len(negative_modes)
    coverage_days = pd.to_datetime(coverage.get("day"), utc=True, errors="coerce")
    mode_days = pd.to_datetime(negative_modes.get("day"), utc=True, errors="coerce")
    checks["negative_day_keys_exact"] = bool(
        coverage_days.notna().all()
        and mode_days.notna().all()
        and not coverage_days.duplicated().any()
        and not mode_days.duplicated().any()
        and set(coverage_days) == set(mode_days)
    )
    checks["negative_day_modes_use_frozen_taxonomy"] = (
        taxonomy_manifest.get("negative_day_mode_assignment_contract")
        == "frozen_reference_prototypes"
    )
    frozen_local = pd.read_parquet(
        taxonomy / "local_frozen_failure_mode_assignments.parquet"
    )
    frozen_parent = pd.read_parquet(
        taxonomy / "parent_frozen_failure_mode_assignments.parquet"
    )
    frozen_local_diagnostics = pd.read_csv(
        taxonomy / "local_frozen_failure_mode_diagnostics.csv"
    )
    frozen_parent_diagnostics = pd.read_csv(
        taxonomy / "parent_frozen_failure_mode_diagnostics.csv"
    )
    frozen_local_profiles = pd.read_csv(
        taxonomy / "local_frozen_failure_mode_profiles.csv"
    )
    frozen_parent_profiles = pd.read_csv(
        taxonomy / "parent_frozen_failure_mode_profiles.csv"
    )
    frozen_local_state = _json(
        taxonomy / "local_frozen_failure_taxonomy_state.json"
    )
    frozen_parent_state = _json(
        taxonomy / "parent_frozen_failure_taxonomy_state.json"
    )
    checks["frozen_taxonomies_populated"] = bool(
        len(frozen_local)
        and len(frozen_parent)
        and len(frozen_local_diagnostics)
        and len(frozen_parent_diagnostics)
        and len(frozen_local_profiles)
        and len(frozen_parent_profiles)
        and frozen_local_state.get("groups")
        and frozen_parent_state.get("groups")
    )
    if taxonomy_manifest.get("historical_meta_score_available") is False:
        local_calendar = pd.read_parquet(taxonomy / "local_adverse_calendar.parquet")
        unavailable_meta_columns = [
            name
            for name in local_calendar
            if name.startswith("expost__meta_")
            or name.startswith("expost__base_meta_")
        ]
        checks["missing_meta_evidence_preserved"] = bool(
            unavailable_meta_columns
            and all(local_calendar[name].isna().all() for name in unavailable_meta_columns)
            and int(
                taxonomy_manifest.get(
                    "frozen_semantic_label_changes_after_meta_missingness_repair", -1
                )
            )
            == 0
        )
    else:
        checks["missing_meta_evidence_preserved"] = True
    metrics["negative_pnl_days"] = int(len(coverage))
    metrics["negative_days_with_parent_mode"] = int(
        negative_modes.get("parent_mode_assigned", pd.Series(dtype=bool))
        .fillna(False)
        .sum()
    )
    metrics["negative_day_local_mode_full_coverage"] = float(
        negative_modes.get(
            "all_active_local_modes_assigned", pd.Series(dtype=bool)
        )
        .fillna(False)
        .mean()
    )

    local_profiles = pd.read_csv(taxonomy / "local_failure_mixture_profiles.csv")
    parent_profiles = pd.read_csv(taxonomy / "parent_failure_mixture_profiles.csv")
    frozen_local_profiles = pd.read_csv(
        taxonomy / "local_frozen_failure_mode_profiles.csv"
    )
    frozen_parent_profiles = pd.read_csv(
        taxonomy / "parent_frozen_failure_mode_profiles.csv"
    )
    local_stability = pd.read_csv(
        taxonomy / "local_failure_mode_temporal_stability.csv"
    )
    parent_stability = pd.read_csv(
        taxonomy / "parent_failure_mode_temporal_stability.csv"
    )
    local_nonredundancy = pd.read_csv(
        taxonomy / "local_failure_mixture_nonredundancy.csv"
    )
    parent_nonredundancy = pd.read_csv(
        taxonomy / "parent_failure_mixture_nonredundancy.csv"
    )
    checks["distinct_failure_modes_present"] = bool(
        len(local_profiles) >= 2 and len(parent_profiles) >= 2
    )
    checks["stability_audit_present"] = bool(
        len(local_stability) and len(parent_stability)
    )
    checks["nonredundancy_audit_present"] = bool(
        len(local_nonredundancy) and len(parent_nonredundancy)
    )
    # The prospective detector and negative-day catalog use the frozen
    # reference taxonomy. Keep descriptive full-period counts explicit rather
    # than reporting them under the canonical mode names.
    metrics["local_semantic_modes"] = int(
        frozen_local_profiles["semantic_label"].nunique()
    )
    metrics["parent_semantic_modes"] = int(
        frozen_parent_profiles["semantic_label"].nunique()
    )
    metrics["local_technical_modes"] = int(len(frozen_local_profiles))
    metrics["parent_technical_modes"] = int(len(frozen_parent_profiles))
    metrics["descriptive_local_semantic_modes"] = int(
        local_profiles["semantic_label"].nunique()
    )
    metrics["descriptive_parent_semantic_modes"] = int(
        parent_profiles["semantic_label"].nunique()
    )
    metrics["descriptive_local_technical_modes"] = int(len(local_profiles))
    metrics["descriptive_parent_technical_modes"] = int(len(parent_profiles))
    metrics["local_mode_mean_ev_range"] = float(
        frozen_local_profiles["mean_calendar_ev"].max()
        - frozen_local_profiles["mean_calendar_ev"].min()
    )
    metrics["parent_mode_mean_ev_range"] = float(
        frozen_parent_profiles["mean_calendar_ev"].max()
        - frozen_parent_profiles["mean_calendar_ev"].min()
    )
    metrics["descriptive_local_mode_mean_ev_range"] = float(
        local_profiles["mean_calendar_ev"].max()
        - local_profiles["mean_calendar_ev"].min()
    )
    metrics["descriptive_parent_mode_mean_ev_range"] = float(
        parent_profiles["mean_calendar_ev"].max()
        - parent_profiles["mean_calendar_ev"].min()
    )
    metrics["local_temporal_warning_rate"] = _finite_mean(
        local_stability, "temporal_stability_warning"
    )
    metrics["parent_temporal_warning_rate"] = _finite_mean(
        parent_stability, "temporal_stability_warning"
    )
    metrics["local_calendar_redundancy_warning_rate"] = _finite_mean(
        local_nonredundancy, "calendar_redundancy_warning"
    )
    metrics["parent_calendar_redundancy_warning_rate"] = _finite_mean(
        parent_nonredundancy, "calendar_redundancy_warning"
    )

    prospective_manifest = _json(prospective / "manifest.json")
    selections = []
    predictions = []
    detector_metrics = []
    for scope in ("local", "parent"):
        selections.append(pd.read_csv(prospective / f"{scope}_feature_selection.csv"))
        predictions.append(
            pd.read_parquet(prospective / f"{scope}_oos_predictions.parquet")
        )
        detector_metrics.append(
            pd.read_csv(prospective / f"{scope}_oos_metrics.csv")
        )
    selection = pd.concat(selections, ignore_index=True)
    prediction = pd.concat(predictions, ignore_index=True)
    detector = pd.concat(detector_metrics, ignore_index=True)
    leaked = sorted(
        {
            str(name)
            for name in selection.get("feature", pd.Series(dtype=str)).dropna()
            if str(name).startswith(FORBIDDEN_FEATURE_PREFIXES)
        }
    )
    checks["prospective_features_outcome_free"] = not leaked
    nonportable = sorted(
        {
            str(name)
            for name in selection.get("feature", pd.Series(dtype=str)).dropna()
            if is_batch_layout_dependent_ae_gmm_feature(str(name))
        }
    )
    checks["prospective_features_batch_layout_portable"] = not nonportable
    checks["prospective_predictions_chronological"] = bool(
        len(prediction)
        and (
            pd.to_datetime(prediction["day"], utc=True)
            >= pd.to_datetime(prediction["train_end"], utc=True)
        ).all()
    )
    checks["prospective_outputs_present"] = bool(len(prediction) and len(detector))
    checks["prospective_probabilities_bounded"] = bool(
        len(prediction)
        and pd.to_numeric(prediction["risk"], errors="coerce").between(0.0, 1.0).all()
    )
    metrics["prospective_prediction_rows"] = int(len(prediction))
    metrics["prospective_fold_arms"] = int(len(detector))
    metrics["prospective_mean_precision"] = _finite_mean(detector, "precision")
    metrics["prospective_mean_recall"] = _finite_mean(detector, "recall")
    metrics["prospective_mean_lift"] = _finite_mean(detector, "lift")
    metrics["prospective_mean_average_precision"] = _finite_mean(
        detector, "average_precision"
    )
    metrics["prospective_mean_severity_mae"] = _finite_mean(
        detector, "oos_failure_severity_mae"
    )
    metrics["prospective_manifest_status"] = prospective_manifest.get(
        "deployment_status"
    )
    metrics["leaked_features"] = leaked
    metrics["nonportable_ae_gmm_features"] = nonportable

    core_checks = (
        "backcast_months_present",
        "backcast_outcome_contract",
        "backcast_cost_once",
        "backcast_path_coverage",
        "three_year_span",
        "every_negative_day_in_parent_episode",
        "every_negative_day_has_parent_mode",
        "negative_day_catalog_complete",
        "negative_day_keys_exact",
        "negative_day_modes_use_frozen_taxonomy",
        "frozen_taxonomies_populated",
        "missing_meta_evidence_preserved",
        "distinct_failure_modes_present",
        "stability_audit_present",
        "nonredundancy_audit_present",
        "prospective_features_outcome_free",
        "prospective_features_batch_layout_portable",
        "prospective_predictions_chronological",
        "prospective_outputs_present",
        "prospective_probabilities_bounded",
    )
    report = {
        "schema": "failure_taxonomy_delivery_validation_v1",
        "passed": all(checks.get(name, False) for name in core_checks),
        "checks": checks,
        "metrics": metrics,
        "quality_note": (
            "Artifact integrity and leakage checks are gates. Prospective lift, "
            "stability and non-redundancy remain empirical evidence and are not "
            "converted into a pass merely because files exist."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backcast", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-path-coverage", type=float, default=0.90)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
