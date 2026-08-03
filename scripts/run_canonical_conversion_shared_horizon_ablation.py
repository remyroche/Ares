#!/usr/bin/env python3
"""Test shared H3/H12 auxiliary learning for conversion-transition heads.

Separate-H12 and pooled-H3/H12 models use identical causal context, folds and
fixed geometry.  Pooled regression targets are standardized per horizon from
permitted training rows only, and each horizon receives equal total loss
weight.  H3 outcomes are auxiliary training rows only; no same-anchor H3
label or prediction becomes an H12 feature.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
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
        _catboost_classifier,
        _catboost_regressor,
        _classification_metrics,
        _context_features,
        _normalise_context,
        _regression_metrics,
        _safe,
        _source_hashes,
        _spread_train_subset,
        build_expanding_folds,
        sha256,
    )
    from scripts.run_canonical_economic_conversion_transition_target_ablation import (
        CONTRIBUTION_SOURCE,
        _contribution_hashes,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_canonical_economic_conversion_transition_head_ablation import (
        CATBOOST_GEOMETRY,
        COHORT_KEY,
        CONTEXT_SOURCE,
        LABEL_SOURCE,
        PRIMARY_HORIZON,
        _catboost_classifier,
        _catboost_regressor,
        _classification_metrics,
        _context_features,
        _normalise_context,
        _regression_metrics,
        _safe,
        _source_hashes,
        _spread_train_subset,
        build_expanding_folds,
        sha256,
    )
    from run_canonical_economic_conversion_transition_target_ablation import (
        CONTRIBUTION_SOURCE,
        _contribution_hashes,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/canonical_conversion_shared_horizon_ablation_20260729_v1"
)
SCHEMA = "canonical_conversion_shared_horizon_ablation_v1"
KEY = (*COHORT_KEY, "horizon_hours")
ARMS = ("separate_h12", "shared_h3_h12")


@dataclass(frozen=True)
class SharedTarget:
    name: str
    column: str
    conditional_flags: tuple[str, ...] = ()


TARGETS = (
    SharedTarget(
        "opportunity_probability_0bps",
        "delta_opportunity_probability_0bps",
    ),
    SharedTarget("direct_mean_net", "delta_direct_mean_net"),
    SharedTarget(
        "adverse_severity_robust_mean",
        "delta_conditional_adverse_loss_robust_mean",
        (
            "before_adverse_loss_missing_support_flag",
            "after_adverse_loss_missing_support_flag",
        ),
    ),
    SharedTarget(
        "robust_unconditional_upside",
        "delta_positive_net_contribution_robust_mean",
    ),
)


def _training_scale(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        raise ValueError("cannot standardize an empty training target")
    center = float(np.median(numeric))
    scale = float(1.4826 * np.median(np.abs(numeric - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(numeric))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _equal_horizon_weights(horizons: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(horizons, errors="raise").to_numpy(int)
    unique, counts = np.unique(numeric, return_counts=True)
    if not len(unique):
        return np.asarray([], dtype=float)
    lookup = {
        int(horizon): float(len(numeric) / (len(unique) * count))
        for horizon, count in zip(unique, counts, strict=True)
    }
    return np.asarray([lookup[int(value)] for value in numeric], dtype=float)


def prepare_population(
    context: pd.DataFrame,
    base: pd.DataFrame,
    contribution: pd.DataFrame,
    features: Iterable[str],
) -> pd.DataFrame:
    features = tuple(features)
    context = _normalise_context(context, features)
    base_columns = [
        *KEY,
        "before_global_hour_complete_flag",
        "after_global_hour_complete_flag",
        "before_candidate_support",
        "after_candidate_support",
        "before_target_available_utc",
        "after_target_available_utc",
        "before_adverse_loss_missing_support_flag",
        "after_adverse_loss_missing_support_flag",
        "delta_opportunity_probability_0bps",
        "delta_direct_mean_net",
        "delta_conditional_adverse_loss_robust_mean",
    ]
    contribution_columns = [
        *KEY,
        "delta_positive_net_contribution_robust_mean",
    ]
    base = base.loc[:, base_columns].copy()
    contribution = contribution.loc[:, contribution_columns].copy()
    for frame in (base, contribution):
        frame["cohort_anchor_utc"] = pd.to_datetime(
            frame["cohort_anchor_utc"], utc=True, errors="raise"
        )
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if frame.duplicated(list(KEY)).any():
            raise ValueError("shared-horizon label identity is not one-to-one")
    for column in ("before_target_available_utc", "after_target_available_utc"):
        base[column] = pd.to_datetime(base[column], utc=True, errors="coerce")
    base["label_available_utc"] = pd.concat(
        [base["before_target_available_utc"], base["after_target_available_utc"]],
        axis=1,
    ).max(axis=1)
    population = (
        base.merge(contribution, on=list(KEY), how="inner", validate="one_to_one")
        .merge(context, on=list(COHORT_KEY), how="left", validate="many_to_one")
    )
    population["complete_target"] = (
        population["before_global_hour_complete_flag"].astype(bool)
        & population["after_global_hour_complete_flag"].astype(bool)
        & population["before_candidate_support"].gt(0)
        & population["after_candidate_support"].gt(0)
        & population["after_target_available_utc"].notna()
    )
    population["context__horizon_hours"] = population["horizon_hours"].astype(float)
    if population.loc[:, features].isna().all(axis=1).any():
        raise ValueError("shared-horizon population lacks causal context")
    return population.sort_values(
        ["horizon_hours", "cohort_anchor_utc", "side_name", "frozen_base_score_decile"],
        kind="stable",
    ).reset_index(drop=True)


def _target_valid(
    frame: pd.DataFrame, target: SharedTarget
) -> np.ndarray:
    valid = frame["complete_target"].to_numpy(bool)
    valid &= np.isfinite(
        pd.to_numeric(frame[target.column], errors="coerce").to_numpy(float)
    )
    for flag in target.conditional_flags:
        valid &= ~frame[flag].astype(bool).to_numpy()
    return valid


def _metrics(
    frame: pd.DataFrame,
    *,
    target: SharedTarget,
    arm: str,
    fold_id: int,
    period: str,
) -> dict[str, Any]:
    valid = frame["target_valid"].astype(bool)
    scored = frame.loc[valid]
    y = scored["target_delta"].to_numpy(float)
    result: dict[str, Any] = {
        "target": target.name,
        "arm": arm,
        "fold_id": int(fold_id),
        "period": period,
        "rows": int(len(scored)),
    }
    for prefix, values in (
        ("model", scored["delta_prediction"].to_numpy(float)),
        ("constant", scored["constant_delta_prediction"].to_numpy(float)),
    ):
        for name, value in _regression_metrics(y, values).items():
            if name != "rows":
                result[f"{prefix}_regression_{name}"] = value
    sign = (y > 0.0).astype(np.int8)
    for prefix, values in (
        ("model", scored["sign_probability"].to_numpy(float)),
        ("constant", scored["constant_sign_probability"].to_numpy(float)),
    ):
        for name, value in _classification_metrics(sign, values).items():
            if name != "rows":
                result[f"{prefix}_sign_{name}"] = value
    return result


def fit_shared_target_oof(
    population: pd.DataFrame,
    *,
    target: SharedTarget,
    arm: str,
    features: Iterable[str],
    folds: list[Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if arm not in ARMS:
        raise ValueError(f"unknown shared-horizon arm: {arm}")
    base_features = tuple(features)
    model_features = (
        base_features
        if arm == "separate_h12"
        else (*base_features, "context__horizon_hours")
    )
    latest_fold = max(int(fold["fold_id"]) for fold in folds)
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in folds:
        start = pd.Timestamp(fold["validation_start_utc"])
        end = pd.Timestamp(fold["validation_end_utc"])
        evaluation = population.loc[
            population["horizon_hours"].eq(PRIMARY_HORIZON)
            & population["cohort_anchor_utc"].ge(start)
            & population["cohort_anchor_utc"].lt(end)
        ].copy()
        allowed = population.loc[
            population["after_target_available_utc"].lt(start)
            & population["label_available_utc"].lt(start)
        ].copy()
        if arm == "separate_h12":
            allowed = allowed.loc[
                allowed["horizon_hours"].eq(PRIMARY_HORIZON)
            ].copy()
        train_valid = _target_valid(allowed, target)
        evaluation_valid = _target_valid(evaluation, target)
        train = allowed.loc[train_valid].copy()
        train["__target_raw__"] = pd.to_numeric(
            train[target.column], errors="coerce"
        ).to_numpy(float)
        scales: dict[int, tuple[float, float]] = {}
        for horizon, horizon_rows in train.groupby("horizon_hours", sort=True):
            scales[int(horizon)] = _training_scale(horizon_rows["__target_raw__"])
        train["__target_scaled__"] = [
            (value - scales[int(horizon)][0]) / scales[int(horizon)][1]
            for value, horizon in zip(
                train["__target_raw__"], train["horizon_hours"], strict=True
            )
        ]
        train["__weight__"] = (
            1.0
            if arm == "separate_h12"
            else _equal_horizon_weights(train["horizon_hours"])
        )
        train_subset = _spread_train_subset(train, int(args.fit_budget_rows))
        h12_train = train.loc[train["horizon_hours"].eq(PRIMARY_HORIZON)]
        constant = float(h12_train["__target_raw__"].mean())
        constant_sign = float((h12_train["__target_raw__"] > 0.0).mean())
        result = evaluation.loc[
            :,
            [*KEY, "after_target_available_utc", "label_available_utc"],
        ].copy()
        result["target"] = target.name
        result["arm"] = arm
        result["fold_id"] = int(fold["fold_id"])
        result["validation_start_utc"] = start
        result["validation_end_utc"] = end
        result["target_delta"] = pd.to_numeric(
            evaluation[target.column], errors="coerce"
        ).to_numpy(float)
        result["target_valid"] = evaluation_valid
        result["delta_prediction"] = np.nan
        result["sign_probability"] = np.nan
        result["constant_delta_prediction"] = np.where(
            evaluation_valid, constant, np.nan
        )
        result["constant_sign_probability"] = np.where(
            evaluation_valid, constant_sign, np.nan
        )
        if len(train_subset) >= int(args.min_train_rows) and evaluation_valid.any():
            x_train = train_subset.loc[:, model_features]
            x_eval = evaluation.loc[evaluation_valid, model_features]
            regressor = _catboost_regressor(
                seed=int(args.random_state + 10_000 * int(fold["fold_id"])),
                threads=int(args.threads),
            ).fit(
                x_train,
                train_subset["__target_scaled__"].to_numpy(float),
                sample_weight=train_subset["__weight__"].to_numpy(float),
            )
            standardized = np.asarray(regressor.predict(x_eval), dtype=float)
            h12_center, h12_scale = scales[PRIMARY_HORIZON]
            result.loc[evaluation_valid, "delta_prediction"] = (
                standardized * h12_scale + h12_center
            )
            sign_train = (
                train_subset["__target_raw__"].to_numpy(float) > 0.0
            ).astype(np.int8)
            if np.unique(sign_train).size == 2:
                classifier = _catboost_classifier(
                    seed=int(args.random_state + 20_000 * int(fold["fold_id"])),
                    threads=int(args.threads),
                ).fit(
                    x_train,
                    sign_train,
                    sample_weight=train_subset["__weight__"].to_numpy(float),
                )
                result.loc[evaluation_valid, "sign_probability"] = (
                    classifier.predict_proba(x_eval)[:, 1]
                )
            else:
                result.loc[evaluation_valid, "sign_probability"] = constant_sign
        else:
            result.loc[evaluation_valid, "delta_prediction"] = constant
            result.loc[evaluation_valid, "sign_probability"] = constant_sign
        result["training_rows"] = int(len(train))
        result["fitted_training_rows"] = int(len(train_subset))
        result["training_max_after_target_available_utc"] = train[
            "after_target_available_utc"
        ].max()
        result["h3_same_anchor_label_or_prediction_feature"] = False
        prediction_parts.append(result)
        metric_rows.append(
            _metrics(
                result,
                target=target,
                arm=arm,
                fold_id=int(fold["fold_id"]),
                period="development"
                if int(fold["fold_id"]) < latest_fold
                else "confirmation",
            )
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(metric_rows)


def _period_metrics(
    predictions: pd.DataFrame,
    targets: Iterable[SharedTarget],
) -> pd.DataFrame:
    lookup = {target.name: target for target in targets}
    latest = int(predictions["fold_id"].max())
    records: list[dict[str, Any]] = []
    for (target_name, arm), group in predictions.groupby(
        ["target", "arm"], observed=True, sort=True
    ):
        for period, mask in (
            ("development", group["fold_id"].lt(latest)),
            ("confirmation", group["fold_id"].eq(latest)),
            ("all_oof", pd.Series(True, index=group.index)),
        ):
            records.append(
                _metrics(
                    group.loc[mask],
                    target=lookup[target_name],
                    arm=arm,
                    fold_id=latest if period == "confirmation" else -1,
                    period=period,
                )
            )
    return pd.DataFrame.from_records(records)


def _comparison_gates(period_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target, group in period_metrics.groupby("target", observed=True, sort=True):
        index = group.set_index(["arm", "period"])
        separate_dev = index.loc[("separate_h12", "development")]
        shared_dev = index.loc[("shared_h3_h12", "development")]
        separate_confirmation = index.loc[("separate_h12", "confirmation")]
        shared_confirmation = index.loc[("shared_h3_h12", "confirmation")]
        dev_noninferior = bool(
            shared_dev["model_regression_mae"]
            <= separate_dev["model_regression_mae"]
            and shared_dev["model_regression_rank_ic"]
            >= separate_dev["model_regression_rank_ic"]
            and shared_dev["model_sign_auc"] >= separate_dev["model_sign_auc"]
            and shared_dev["model_sign_calibration_ece_10"]
            <= separate_dev["model_sign_calibration_ece_10"]
        )
        confirmation_noninferior = bool(
            shared_confirmation["model_regression_mae"]
            <= separate_confirmation["model_regression_mae"]
            and shared_confirmation["model_regression_rank_ic"]
            >= separate_confirmation["model_regression_rank_ic"]
            and shared_confirmation["model_sign_auc"]
            >= separate_confirmation["model_sign_auc"]
        )
        confirmation_repairs_constant = bool(
            shared_confirmation["model_regression_mae"]
            <= shared_confirmation["constant_regression_mae"]
        )
        records.append(
            {
                "target": target,
                "development_all_metric_noninferior": dev_noninferior,
                "confirmation_mae_ic_auc_noninferior": confirmation_noninferior,
                "confirmation_mae_beats_constant": confirmation_repairs_constant,
                "shared_horizon_passes": bool(
                    dev_noninferior
                    and confirmation_noninferior
                    and confirmation_repairs_constant
                ),
                "shared_minus_separate_development_mae": float(
                    shared_dev["model_regression_mae"]
                    - separate_dev["model_regression_mae"]
                ),
                "shared_minus_separate_development_ic": float(
                    shared_dev["model_regression_rank_ic"]
                    - separate_dev["model_regression_rank_ic"]
                ),
                "shared_minus_separate_confirmation_mae": float(
                    shared_confirmation["model_regression_mae"]
                    - separate_confirmation["model_regression_mae"]
                ),
                "shared_minus_separate_confirmation_ic": float(
                    shared_confirmation["model_regression_rank_ic"]
                    - separate_confirmation["model_regression_rank_ic"]
                ),
                "status": "DIAGNOSTIC_ONLY_NOT_PROMOTION_ELIGIBLE",
            }
        )
    return pd.DataFrame.from_records(records)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    manifests, hashes = _source_hashes(
        Path(args.context_source), Path(args.base_label_source)
    )
    contribution_manifest, contribution_hashes = _contribution_hashes(
        Path(args.contribution_source)
    )
    features = _context_features(manifests["context"])
    return {
        "action": "PLAN_ONLY_NO_TRAINING",
        "schema": SCHEMA,
        "source_sha256": {**hashes, **contribution_hashes},
        "contribution_schema": contribution_manifest["schema"],
        "targets": [target.name for target in TARGETS],
        "arms": ARMS,
        "base_feature_count": len(features),
        "shared_feature_addition": "horizon_hours only",
        "target_scaling": "training-only median/MAD per horizon",
        "loss_weighting": "equal total H3 and H12 mass",
        "forbidden": "same-anchor H3 label or prediction as an H12 feature",
        "scope": "component auxiliary-learning diagnostic only",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan_only:
        return plan(args)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    context_source = Path(args.context_source)
    base_source = Path(args.base_label_source)
    contribution_source = Path(args.contribution_source)
    manifests, hashes = _source_hashes(context_source, base_source)
    contribution_manifest, contribution_hashes = _contribution_hashes(
        contribution_source
    )
    features = _context_features(manifests["context"])
    context = pd.read_parquet(
        context_source / "cohort_transition_context.parquet",
        columns=[*COHORT_KEY, *features],
    )
    base = pd.read_parquet(base_source / "cohort_transition_labels.parquet")
    contribution = pd.read_parquet(
        contribution_source / "cohort_contribution_labels.parquet"
    )
    population = prepare_population(context, base, contribution, features)
    folds = build_expanding_folds(
        population,
        min_train_days=int(args.min_train_days),
        validation_days=int(args.validation_days),
    )
    prediction_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        for arm in ARMS:
            predictions, metrics = fit_shared_target_oof(
                population,
                target=target,
                arm=arm,
                features=features,
                folds=folds,
                args=args,
            )
            prediction_parts.append(predictions)
            fold_parts.append(metrics)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_fold = pd.concat(fold_parts, ignore_index=True)
    period = _period_metrics(predictions, TARGETS)
    gates = _comparison_gates(period)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    frames = {
        "oof_shared_horizon_predictions.parquet": predictions,
        "per_fold_shared_horizon_metrics.parquet": per_fold,
        "period_shared_horizon_metrics.parquet": period,
        "shared_horizon_gates.parquet": gates,
    }
    for name, frame in frames.items():
        frame.to_parquet(temporary / name, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_AUXILIARY_LEARNING_ABLATION_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": {**hashes, **contribution_hashes},
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "contribution_schema": contribution_manifest["schema"],
        "targets": [target.__dict__ for target in TARGETS],
        "arms": ARMS,
        "base_context_feature_columns": list(features),
        "shared_feature_addition": "context__horizon_hours",
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "folds": folds,
        "contracts": {
            "training_availability": "all H3/H12 labels resolve strictly before validation start",
            "scaling": "fold-local training median/MAD independently by horizon; H12 predictions transformed back to raw units",
            "weighting": "equal total H3/H12 training loss mass",
            "no_same_anchor_leakage": "H3 outcomes are auxiliary rows only; no H3 target or prediction is an H12 feature",
            "selection": "shared must be noninferior on development and confirmation and beat confirmation constant MAE",
            "scope": "component diagnostic only; no admission, PnL, policy, or replay",
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
        "targets": len(TARGETS),
        "oof_predictions": int(len(predictions)),
        "passing_shared_targets": int(gates["shared_horizon_passes"].sum()),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--context-source", type=Path, default=CONTEXT_SOURCE)
    result.add_argument("--base-label-source", type=Path, default=LABEL_SOURCE)
    result.add_argument("--contribution-source", type=Path, default=CONTRIBUTION_SOURCE)
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
