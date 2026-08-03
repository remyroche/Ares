#!/usr/bin/env python3
"""Ablate causal feature groups for H12 economic-conversion transitions.

This experiment reuses the immutable label/context sources and the fixed
CatBoost/fold contract from the canonical transition-head baseline.  It is a
diagnostic learnability comparison only: the latest fold is reported as a
stability gate and is not used to promote a trading score or policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from scripts.run_canonical_economic_conversion_transition_head_ablation import (
        CATBOOST_GEOMETRY,
        COHORT_KEY,
        CONTEXT_SOURCE,
        LABEL_SOURCE,
        PRIMARY_HORIZON,
        TARGETS,
        _aggregate_oof_metrics,
        _context_features,
        _fold_stability,
        _label_columns,
        _source_hashes,
        _safe,
        build_expanding_folds,
        fit_target_oof,
        prepare_population,
        sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_canonical_economic_conversion_transition_head_ablation import (
        CATBOOST_GEOMETRY,
        COHORT_KEY,
        CONTEXT_SOURCE,
        LABEL_SOURCE,
        PRIMARY_HORIZON,
        TARGETS,
        _aggregate_oof_metrics,
        _context_features,
        _fold_stability,
        _label_columns,
        _source_hashes,
        _safe,
        build_expanding_folds,
        fit_target_oof,
        prepare_population,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_economic_conversion_transition_feature_group_ablation_20260729_v1"
)
SCHEMA = "canonical_economic_conversion_transition_feature_group_ablation_v1"
EVALUATED_TARGETS = tuple(
    target for target in TARGETS if target.name != "exit_mixture_net"
)


def _feature_groups(features: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Build fixed semantic groups from the immutable context whitelist."""

    ordered = tuple(map(str, features))
    side_controls = tuple(
        column
        for column in ordered
        if column in ("context__side_sign", "context__frozen_base_score_decile")
    )
    score = tuple(
        column
        for column in ordered
        if column.startswith("context__base_")
        or column.startswith("context__selected_top40_")
    )
    market = tuple(
        column
        for column in ordered
        if column
        in (
            "context__range_24h_pct__mean",
            "context__meta_raw__volatility_zscore__mean",
            "context__trend_r2_24__mean",
            "context__jump_intensity__mean",
            "context__meta_raw__chop_score__mean",
        )
    )
    market_deltas = tuple(
        column
        for column in ordered
        if column.startswith("context__preentry_transition__")
        and "__regime_source_" not in column
    )
    regime_deltas = tuple(
        column
        for column in ordered
        if column.startswith("context__preentry_transition__")
        and "__regime_source_" in column
    )
    regime = tuple(
        column for column in ordered if column.startswith("context__regime_source_")
    )

    def combine(*parts: Iterable[str]) -> tuple[str, ...]:
        allowed = {column for part in parts for column in part}
        return tuple(column for column in ordered if column in allowed)

    groups = {
        "identity_only": side_controls,
        "score_only": combine(score, side_controls),
        "market_only": combine(market, market_deltas, side_controls),
        "regime_level_only": combine(regime, side_controls),
        "regime_transition_only": combine(regime_deltas, side_controls),
        "market_and_regime": combine(
            market, market_deltas, regime_deltas, regime, side_controls
        ),
        "score_and_regime": combine(score, regime_deltas, regime, side_controls),
        "full_context": ordered,
    }
    empty = [name for name, columns in groups.items() if not columns]
    if empty:
        raise ValueError(f"empty causal feature groups: {empty}")
    if set(groups["full_context"]) != set(ordered):
        raise AssertionError("full-context group does not preserve the immutable whitelist")
    if set(score) & set(market) or set(score) & set(market_deltas) or set(score) & set(regime):
        raise AssertionError("semantic feature groups overlap unexpectedly")
    semantic = (score, market, market_deltas, regime_deltas, regime)
    if any(set(left) & set(right) for index, left in enumerate(semantic) for right in semantic[index + 1 :]):
        raise AssertionError("semantic feature groups overlap unexpectedly")
    return groups


def _hash_tree(root: Path) -> dict[str, str]:
    required = (
        "manifest.json",
        "manifest.sha256",
        "oof_head_predictions.parquet",
        "per_fold_metrics.parquet",
        "oof_metrics.parquet",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"baseline head artifact is incomplete: {missing}")
    sidecar = (root / "manifest.sha256").read_text(encoding="utf-8").strip().split(maxsplit=1)
    if not sidecar or sidecar[0] != sha256(root / "manifest.json"):
        raise ValueError("baseline head manifest checksum mismatch")
    return {str(root / name): sha256(root / name) for name in required}


def _summarize_group(
    predictions: pd.DataFrame, per_fold: pd.DataFrame, feature_group: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = _aggregate_oof_metrics(predictions)
    aggregate.insert(0, "feature_group", feature_group)
    latest = (
        per_fold.sort_values(["horizon_hours", "target", "fold_id"], kind="stable")
        .groupby(["horizon_hours", "target"], observed=True, sort=False)
        .tail(1)
        .copy()
    )
    latest.insert(0, "feature_group", feature_group)
    return aggregate, latest


def _build_gate_table(aggregate: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    keys = ["feature_group", "horizon_hours", "target"]
    aggregate_columns = keys + [
        "model_regression_mae",
        "constant_regression_mae",
        "model_regression_rank_ic",
        "model_sign_auc",
        "model_sign_ap",
        "model_sign_brier",
        "model_sign_calibration_ece_10",
        "constant_sign_brier",
    ]
    latest_columns = keys + [
        "fold_id",
        "validation_start_utc",
        "validation_end_utc",
        "target_valid_rows",
        "model_regression_mae",
        "constant_regression_mae",
        "model_regression_rank_ic",
        "model_sign_auc",
        "model_sign_ap",
        "model_sign_brier",
        "model_sign_calibration_ece_10",
        "constant_sign_brier",
    ]
    left = aggregate.loc[:, aggregate_columns].rename(
        columns={
            column: f"aggregate__{column}"
            for column in aggregate_columns
            if column not in keys
        }
    )
    right = latest.loc[:, latest_columns].rename(
        columns={
            column: f"latest__{column}"
            for column in latest_columns
            if column not in keys
        }
    )
    result = left.merge(right, on=keys, how="inner", validate="one_to_one")
    result["aggregate_mae_gain"] = (
        result["aggregate__constant_regression_mae"]
        - result["aggregate__model_regression_mae"]
    )
    result["latest_mae_gain"] = (
        result["latest__constant_regression_mae"]
        - result["latest__model_regression_mae"]
    )
    result["aggregate_brier_gain"] = (
        result["aggregate__constant_sign_brier"] - result["aggregate__model_sign_brier"]
    )
    result["latest_brier_gain"] = (
        result["latest__constant_sign_brier"] - result["latest__model_sign_brier"]
    )
    result["passes_aggregate_baselines"] = (
        result["aggregate_mae_gain"].gt(0.0)
        & result["aggregate_brier_gain"].gt(0.0)
        & result["aggregate__model_regression_rank_ic"].gt(0.0)
        & result["aggregate__model_sign_auc"].gt(0.5)
    )
    result["passes_latest_baselines"] = (
        result["latest_mae_gain"].gt(0.0)
        & result["latest_brier_gain"].gt(0.0)
        & result["latest__model_regression_rank_ic"].gt(0.0)
        & result["latest__model_sign_auc"].gt(0.5)
    )
    result["passes_both_period_gates"] = (
        result["passes_aggregate_baselines"] & result["passes_latest_baselines"]
    )
    result["diagnostic_rank_within_target"] = (
        result.sort_values(
            [
                "target",
                "passes_both_period_gates",
                "latest_mae_gain",
                "latest__model_sign_auc",
                "aggregate_mae_gain",
            ],
            ascending=[True, False, False, False, False],
            kind="stable",
        )
        .groupby("target", observed=True)
        .cumcount()
        .add(1)
        .reindex(result.index)
        .astype(int)
    )
    return result.sort_values(
        ["target", "diagnostic_rank_within_target", "feature_group"], kind="stable"
    ).reset_index(drop=True)


def _recent_fold_gates(per_fold: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (feature_group, target), group in per_fold.groupby(
        ["feature_group", "target"], observed=True, sort=True
    ):
        recent_ids = sorted(group["fold_id"].unique())[-2:]
        recent = group.loc[group["fold_id"].isin(recent_ids)].copy()
        mae_gain = (
            recent["constant_regression_mae"] - recent["model_regression_mae"]
        )
        records.append(
            {
                "feature_group": feature_group,
                "horizon_hours": int(recent["horizon_hours"].iloc[0]),
                "target": target,
                "recent_fold_ids": ",".join(map(str, recent_ids)),
                "recent_folds": int(len(recent)),
                "recent_all_positive_rank_ic": bool(
                    recent["model_regression_rank_ic"].gt(0.0).all()
                ),
                "recent_all_mae_better_than_constant": bool(mae_gain.gt(0.0).all()),
                "recent_min_rank_ic": float(recent["model_regression_rank_ic"].min()),
                "recent_min_mae_gain": float(mae_gain.min()),
            }
        )
    return pd.DataFrame.from_records(records)


def _apply_predeclared_gates(
    gates: pd.DataFrame, recent: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = gates.merge(
        recent,
        on=["feature_group", "horizon_hours", "target"],
        how="left",
        validate="one_to_one",
    )
    full = result.loc[result["feature_group"].eq("full_context")].set_index("target")
    target_pass: list[bool] = []
    for row in result.itertuples(index=False):
        if row.target == "opportunity_probability_0bps":
            reference = full.loc[row.target]
            passed = (
                row.latest_mae_gain > 0.0
                and row.latest__model_regression_mae
                <= reference["latest__model_regression_mae"]
                and row.latest__model_regression_rank_ic
                >= reference["latest__model_regression_rank_ic"]
                and row.latest__model_sign_auc >= reference["latest__model_sign_auc"]
            )
        elif row.target == "direct_mean_net":
            reference = full.loc[row.target]
            passed = (
                row.latest_mae_gain > 0.0
                and row.latest__model_regression_rank_ic
                >= reference["latest__model_regression_rank_ic"]
                and row.latest__model_sign_auc >= reference["latest__model_sign_auc"]
                and row.latest__model_sign_calibration_ece_10 <= 0.05
            )
        elif row.target == "adverse_severity_robust_mean":
            reference = full.loc[row.target]
            passed = (
                row.latest_mae_gain > 0.0
                and row.latest__model_regression_rank_ic
                >= reference["latest__model_regression_rank_ic"]
                and row.latest__model_sign_auc >= reference["latest__model_sign_auc"]
            )
        else:
            passed = False
        aggregate_floor = (
            row.aggregate_mae_gain > 0.0
            and row.aggregate__model_regression_rank_ic
            >= 0.9 * full.loc[row.target, "aggregate__model_regression_rank_ic"]
        )
        target_pass.append(
            bool(
                passed
                and aggregate_floor
                and row.recent_all_positive_rank_ic
                and row.recent_all_mae_better_than_constant
            )
        )
    result["passes_predeclared_target_gate"] = target_pass
    primary = result.loc[
        result["target"].isin(
            ("opportunity_probability_0bps", "direct_mean_net")
        )
    ]
    group_rows: list[dict[str, Any]] = []
    for feature_group, group in primary.groupby("feature_group", sort=True):
        passed_targets = set(
            group.loc[group["passes_predeclared_target_gate"], "target"]
        )
        group_rows.append(
            {
                "feature_group": feature_group,
                "opportunity_gate": "opportunity_probability_0bps"
                in passed_targets,
                "direct_net_gate": "direct_mean_net" in passed_targets,
                "advances_to_frozen_ordering_diagnostic": {
                    "opportunity_probability_0bps",
                    "direct_mean_net",
                }.issubset(passed_targets),
                "status": "DIAGNOSTIC_ONLY_NOT_PROMOTION_ELIGIBLE",
            }
        )
    return result, pd.DataFrame.from_records(group_rows)


def _plan(
    context_source: Path,
    label_source: Path,
    baseline_source: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifests, hashes = _source_hashes(context_source, label_source)
    features = _context_features(manifests["context"])
    groups = _feature_groups(features)
    return {
        "action": "PLAN_ONLY_NO_TRAINING_OR_MATERIALIZATION",
        "schema": SCHEMA,
        "context_source": str(context_source),
        "label_source": str(label_source),
        "baseline_source": str(baseline_source),
        "output": str(output),
        "source_sha256": hashes,
        "baseline_sha256": _hash_tree(baseline_source),
        "horizon_hours": PRIMARY_HORIZON,
        "targets": [target.name for target in EVALUATED_TARGETS],
        "feature_groups": {name: list(columns) for name, columns in groups.items()},
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "fold_contract": {
            "minimum_calendar_history_days": int(args.min_train_days),
            "validation_days": int(args.validation_days),
            "training_availability": "actual after-target availability strictly precedes validation start",
        },
        "scope": "feature attribution and latest-fold stability only; no admission, PnL, policy, or portfolio replay",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context_source = Path(args.context_source)
    label_source = Path(args.label_source)
    baseline_source = Path(args.baseline_source)
    output = Path(args.output_dir)
    if args.plan_only:
        return _plan(context_source, label_source, baseline_source, output, args)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")

    manifests, hashes = _source_hashes(context_source, label_source)
    baseline_hashes = _hash_tree(baseline_source)
    features = _context_features(manifests["context"])
    groups = _feature_groups(features)
    context = pd.read_parquet(
        context_source / "cohort_transition_context.parquet",
        columns=[*COHORT_KEY, *features],
    )
    labels = pd.read_parquet(
        label_source / "cohort_transition_labels.parquet",
        columns=list(_label_columns()),
    )
    population = prepare_population(context, labels, features)
    population = population.loc[
        population["horizon_hours"].eq(PRIMARY_HORIZON)
    ].copy()
    folds = build_expanding_folds(
        population,
        min_train_days=int(args.min_train_days),
        validation_days=int(args.validation_days),
    )

    prediction_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    aggregate_parts: list[pd.DataFrame] = []
    latest_parts: list[pd.DataFrame] = []
    for group_index, (group_name, group_features) in enumerate(groups.items()):
        group_predictions: list[pd.DataFrame] = []
        group_folds: list[pd.DataFrame] = []
        for target in EVALUATED_TARGETS:
            predictions, per_fold = fit_target_oof(
                population,
                target=target,
                features=group_features,
                folds=folds,
                min_train_rows=int(args.min_train_rows),
                fit_budget_rows=int(args.fit_budget_rows),
                random_state=int(args.random_state),
                threads=int(args.threads),
            )
            predictions.insert(0, "feature_group", group_name)
            per_fold.insert(0, "feature_group", group_name)
            group_predictions.append(predictions)
            group_folds.append(per_fold)
        group_prediction_frame = pd.concat(group_predictions, ignore_index=True)
        group_fold_frame = pd.concat(group_folds, ignore_index=True)
        aggregate, latest = _summarize_group(
            group_prediction_frame.drop(columns=["feature_group"]),
            group_fold_frame.drop(columns=["feature_group"]),
            group_name,
        )
        prediction_parts.append(group_prediction_frame)
        fold_parts.append(group_fold_frame)
        aggregate_parts.append(aggregate)
        latest_parts.append(latest)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_fold = pd.concat(fold_parts, ignore_index=True)
    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    latest = pd.concat(latest_parts, ignore_index=True)
    gates = _build_gate_table(aggregate, latest)
    recent = _recent_fold_gates(per_fold)
    gates, group_gates = _apply_predeclared_gates(gates, recent)
    stability_parts: list[pd.DataFrame] = []
    for group_name, group in per_fold.groupby("feature_group", sort=False):
        stability = _fold_stability(group.drop(columns=["feature_group"]))
        stability.insert(0, "feature_group", group_name)
        stability_parts.append(stability)
    stability = pd.concat(stability_parts, ignore_index=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    frames = {
        "oof_feature_group_predictions.parquet": predictions,
        "per_fold_feature_group_metrics.parquet": per_fold,
        "aggregate_feature_group_metrics.parquet": aggregate,
        "latest_fold_feature_group_metrics.parquet": latest,
        "feature_group_gates.parquet": gates,
        "feature_group_recent_fold_gates.parquet": recent,
        "feature_group_advancement_gates.parquet": group_gates,
        "feature_group_fold_stability.parquet": stability,
    }
    for name, frame in frames.items():
        frame.to_parquet(temporary / name, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_DIAGNOSTIC_FEATURE_GROUP_ABLATION_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": hashes,
        "baseline_artifact_sha256": baseline_hashes,
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "horizon_hours": PRIMARY_HORIZON,
        "targets": [target.name for target in EVALUATED_TARGETS],
        "feature_groups": {name: list(columns) for name, columns in groups.items()},
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "fit_budget": {
            "minimum_prior_resolved_rows": int(args.min_train_rows),
            "maximum_fit_rows_per_target_fold": int(args.fit_budget_rows),
            "threads": int(args.threads),
            "random_state": int(args.random_state),
            "feature_selection": "disabled",
            "hyperparameter_optimization": "disabled",
        },
        "folds": folds,
        "contracts": {
            "training_availability": "every training target resolves strictly before validation start",
            "latest_fold": "reported as a diagnostic stability gate; not an untouched promotion fold",
            "selection": "opportunity and direct-net must beat predeclared latest/full and recent-fold gates; any advancing arm is diagnostic only",
            "duplicate_target": "exit-mixture net is accounting-identical to direct mean net in this artifact and is not refit or double-counted",
            "global_policy": "no score mapping, global top-k selection, PnL, admission routing, or portfolio replay",
        },
        "rows": {
            "population": int(len(population)),
            "oof_predictions": int(len(predictions)),
        },
        "outputs_sha256": {
            name: sha256(temporary / name) for name in sorted(frames)
        },
        "checksum_convention": "manifest.json is verified by detached manifest.sha256",
    }
    (temporary / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (temporary / "manifest.sha256").write_text(
        f"{sha256(temporary / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {
        "output": str(output),
        "feature_groups": len(groups),
        "targets": len(EVALUATED_TARGETS),
        "oof_predictions": int(len(predictions)),
        "advancing_feature_groups": int(
            group_gates["advances_to_frozen_ordering_diagnostic"].sum()
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument(
        "--baseline-source",
        type=Path,
        default=(
            ROOT
            / "data_perp/artifacts/"
            "canonical_economic_conversion_transition_head_ablation_20260729_v1"
        ),
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--min-train-days", type=int, default=28)
    result.add_argument("--validation-days", type=int, default=14)
    result.add_argument("--min-train-rows", type=int, default=1_500)
    result.add_argument("--fit-budget-rows", type=int, default=75_000)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--random-state", type=int, default=20260729)
    result.add_argument("--plan-only", action="store_true")
    return result


def main() -> None:
    print(json.dumps(_safe(run(parser().parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
