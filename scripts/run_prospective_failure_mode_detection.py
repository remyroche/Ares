#!/usr/bin/env python3
"""Run leakage-safe chronological local and parent failure-mode detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    ProspectiveFailureDetectorConfig,
    add_causal_state_dynamics,
    attach_failure_mode_targets,
    chronological_failure_detection,
    is_batch_layout_dependent_ae_gmm_feature,
)
from extreme_price_movements.unsupervised_regime_learning.failure_episodes import (
    validate_inference_feature_columns,
)

DEFAULT_TAXONOMY = Path("data_perp/reports/failure_episode_taxonomy_20260719_v5")
DEFAULT_OUTPUT = Path(
    "data_perp/reports/prospective_failure_mode_detection_20260719_v1"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_assignments(
    assignments: pd.DataFrame,
    *,
    state: dict[str, Any],
    path: Path,
) -> None:
    if assignments.empty:
        raise ValueError(f"Frozen failure assignments are empty: {path}")
    if not assignments.get("method", pd.Series(dtype=str)).eq(
        "frozen_consensus_prototype"
    ).all():
        raise ValueError(f"Prospective detector requires frozen assignments: {path}")
    expected = pd.Timestamp(state["reference_end"])
    observed = pd.to_datetime(
        assignments.get("taxonomy_reference_end"), utc=True, errors="coerce"
    )
    if observed.isna().any() or not observed.eq(expected).all():
        raise ValueError(f"Assignment/state reference cutoff mismatch: {path}")


def _run_scope(
    root: Path,
    *,
    prefix: str,
    state_name: str,
    calendar_name: str,
    assignments_name: str,
    output: Path,
    config: ProspectiveFailureDetectorConfig,
) -> dict[str, Any]:
    print(f"Prospective detector scope start: {prefix}", flush=True)
    state = pd.read_parquet(root / state_name)
    state = add_causal_state_dynamics(
        state,
        lookback_days=int(config.inner_validation_days),
        add_market_geometry=prefix == "local",
    )
    observable_features = [
        name
        for name in state
        if name not in {"day", "side_name", "archetype_policy_key"}
        and not is_batch_layout_dependent_ae_gmm_feature(name)
    ]
    excluded_nonportable_features = sorted(
        name
        for name in state
        if is_batch_layout_dependent_ae_gmm_feature(name)
    )
    validate_inference_feature_columns(observable_features)
    calendar = pd.read_parquet(root / calendar_name)
    assignments_path = root / assignments_name
    assignments = pd.read_parquet(assignments_path)
    state_path = root / f"{prefix}_frozen_failure_taxonomy_state.json"
    taxonomy_state = json.loads(state_path.read_text(encoding="utf-8"))
    _validate_frozen_assignments(
        assignments,
        state=taxonomy_state,
        path=assignments_path,
    )
    labelled = attach_failure_mode_targets(
        state,
        calendar,
        assignments,
        lead_days=config.lead_days,
    )
    predictions, metrics, selections = chronological_failure_detection(
        labelled,
        config=config,
        feature_columns=observable_features,
    )
    predictions.to_parquet(output / f"{prefix}_oos_predictions.parquet", index=False)
    metrics.to_csv(output / f"{prefix}_oos_metrics.csv", index=False)
    selections.to_csv(output / f"{prefix}_feature_selection.csv", index=False)
    valid = metrics.loc[metrics["oos_days"].gt(0)] if not metrics.empty else metrics
    any_failure = (
        valid.loc[valid["failure_mode"].eq("any_failure")] if not valid.empty else valid
    )
    summary = {
        "state_rows": int(len(state)),
        "observable_state_features": int(len(observable_features)),
        "observable_state_feature_names": observable_features,
        "excluded_nonportable_ae_gmm_features": excluded_nonportable_features,
        "labelled_rows": int(len(labelled)),
        "prediction_rows": int(len(predictions)),
        "detector_fold_arms": int(len(metrics)),
        "any_failure_fold_arms": int(len(any_failure)),
        "mean_any_failure_precision": float(any_failure["precision"].mean())
        if len(any_failure)
        else np.nan,
        "mean_any_failure_lift": float(any_failure["lift"].mean())
        if len(any_failure)
        else np.nan,
        "mean_any_failure_recall": float(any_failure["recall"].mean())
        if len(any_failure)
        else np.nan,
        "mean_expected_failure_severity": float(
            predictions["expected_failure_severity"].mean()
        )
        if len(predictions)
        else np.nan,
        "mean_realized_failure_severity": float(
            predictions["target_failure_severity"].mean()
        )
        if len(predictions)
        else np.nan,
        "mean_failure_severity_mae": float(
            metrics["oos_failure_severity_mae"].mean()
        )
        if len(metrics)
        else np.nan,
        "mean_risk_aleatoric_uncertainty": float(
            predictions["risk_aleatoric_uncertainty"].mean()
        )
        if len(predictions)
        else np.nan,
        "mean_risk_support_uncertainty": float(
            predictions["risk_support_uncertainty"].mean()
        )
        if len(predictions)
        else np.nan,
        "mode_count": int(
            metrics.loc[
                metrics["failure_mode"].ne("any_failure"), "failure_mode"
            ].nunique()
        )
        if len(metrics)
        else 0,
        "assignment_sha256": _sha256(assignments_path),
        "taxonomy_state_sha256": _sha256(state_path),
        "taxonomy_reference_end": taxonomy_state["reference_end"],
    }
    print(
        "Prospective detector scope complete: "
        f"{prefix} predictions={len(predictions)} arms={len(metrics)}",
        flush=True,
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.taxonomy)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    taxonomy_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state_references = []
    for scope in ("local", "parent"):
        state_payload = json.loads(
            (root / f"{scope}_frozen_failure_taxonomy_state.json").read_text(
                encoding="utf-8"
            )
        )
        state_references.append(pd.Timestamp(state_payload["reference_end"]))
    minimum_evaluation_start = max(state_references)
    evaluation_start = str(args.evaluation_start).strip() or str(
        taxonomy_manifest.get("prospective_taxonomy_reference_end", "")
    )
    evaluation_start_ts = pd.Timestamp(evaluation_start)
    if evaluation_start_ts.tzinfo is None:
        raise ValueError("evaluation_start must be timezone-aware")
    if evaluation_start_ts.tz_convert("UTC") < minimum_evaluation_start:
        raise ValueError(
            "evaluation_start precedes the frozen taxonomy reference cutoff"
        )
    config = ProspectiveFailureDetectorConfig(
        min_train_days=int(args.min_train_days),
        eval_days=int(args.eval_days),
        inner_validation_days=int(args.inner_validation_days),
        min_positive_days=int(args.min_positive_days),
        max_features=int(args.max_features),
        alert_quantile=float(args.alert_quantile),
        probability_calibration=str(args.probability_calibration),
        lead_days=tuple(
            int(value) for value in str(args.lead_days).split(",") if str(value).strip()
        ),
        embargo_days=int(args.embargo_days),
        evaluation_start=evaluation_start,
    )
    local = _run_scope(
        root,
        prefix="local",
        state_name="daily_observable_state.parquet",
        calendar_name="local_adverse_calendar.parquet",
        assignments_name=str(args.local_assignments),
        output=output,
        config=config,
    )
    parent = _run_scope(
        root,
        prefix="parent",
        state_name="daily_parent_market_state.parquet",
        calendar_name="parent_adverse_calendar.parquet",
        assignments_name=str(args.parent_assignments),
        output=output,
        config=config,
    )
    manifest = {
        "schema": "prospective_failure_mode_detection_v1",
        "taxonomy": str(root.resolve()),
        "config": config.__dict__,
        "local": local,
        "parent": parent,
        "leakage_contract": (
            "Feature screening, scaling, model fitting and alert thresholds use only "
            "rows before each OOS interval. OOS inputs are day-open observable state; "
            "expost__, target__, episode recovery and OOS ranks are excluded."
        ),
        "deployment_status": "research_only_not_promoted",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=_json_default), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train-days", type=int, default=120)
    parser.add_argument("--eval-days", type=int, default=45)
    parser.add_argument("--inner-validation-days", type=int, default=35)
    parser.add_argument("--min-positive-days", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--alert-quantile", type=float, default=0.95)
    parser.add_argument(
        "--probability-calibration",
        choices=("platt", "none"),
        default="platt",
    )
    parser.add_argument("--lead-days", default="1,3")
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument("--evaluation-start", default="")
    parser.add_argument(
        "--local-assignments",
        default="local_frozen_failure_mode_semantic_assignments.parquet",
    )
    parser.add_argument(
        "--parent-assignments",
        default="parent_frozen_failure_mode_semantic_assignments.parquet",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
