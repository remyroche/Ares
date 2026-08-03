#!/usr/bin/env python3
"""Ablate support-aware and economically reconciled conversion targets.

The experiment fixes the full 47-feature causal context, CatBoost geometry,
and chronological availability folds.  Folds 0--3 are the development
comparison; the truncated final fold is scored once as confirmation.  Results
remain component diagnostics and cannot promote admission or a policy.
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
        _source_hashes,
        _spread_train_subset,
        _safe,
        build_expanding_folds,
        sha256,
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
        _source_hashes,
        _spread_train_subset,
        _safe,
        build_expanding_folds,
        sha256,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTRIBUTION_SOURCE = (
    ROOT
    / "data_perp/artifacts/canonical_economic_conversion_contribution_labels_20260729_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/"
    "canonical_economic_conversion_transition_target_ablation_20260729_v1"
)
SCHEMA = "canonical_economic_conversion_transition_target_ablation_v1"
KEY = (*COHORT_KEY, "horizon_hours")
EB_PRIOR_SUPPORT = 16.0


@dataclass(frozen=True)
class TargetArm:
    name: str
    target_column: str
    family: str
    weighted: bool = False
    empirical_bayes: bool = False
    selection_target: bool = True


ARMS = (
    TargetArm(
        "A0_conditional_favorable_reference",
        "delta_conditional_favorable_net_robust_mean",
        "conditional_favorable",
    ),
    TargetArm(
        "A1_conditional_favorable_support_weighted",
        "delta_conditional_favorable_net_robust_mean",
        "conditional_favorable",
        weighted=True,
    ),
    TargetArm(
        "A2_conditional_favorable_empirical_bayes",
        "delta_conditional_favorable_net_robust_mean",
        "conditional_favorable",
        weighted=True,
        empirical_bayes=True,
    ),
    TargetArm(
        "B1_unconditional_positive_contribution",
        "delta_positive_net_contribution",
        "unconditional_upside",
    ),
    TargetArm(
        "B1R_robust_positive_contribution",
        "delta_positive_net_contribution_robust_mean",
        "unconditional_upside",
    ),
    TargetArm(
        "B2_soft_net_positive_rate",
        "delta_net_positive_rate",
        "soft_positive_rate",
    ),
    TargetArm(
        "B3_unconditional_loss_contribution",
        "delta_loss_net_contribution",
        "unconditional_downside",
        selection_target=False,
    ),
)


def _contribution_hashes(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256"
    parquet_path = root / "cohort_contribution_labels.parquet"
    for path in (manifest_path, sidecar_path, parquet_path):
        if not path.is_file():
            raise FileNotFoundError(f"contribution source is incomplete: {path}")
    if sidecar_path.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("contribution manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "canonical_economic_conversion_contribution_labels_v1":
        raise ValueError("unexpected contribution-label schema")
    return manifest, {
        str(manifest_path): sha256(manifest_path),
        str(sidecar_path): sha256(sidecar_path),
        str(parquet_path): sha256(parquet_path),
    }


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    value = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not valid.any():
        return float("nan")
    order = np.argsort(value[valid], kind="stable")
    ordered_value = value[valid][order]
    ordered_weight = weight[valid][order]
    cutoff = 0.5 * ordered_weight.sum()
    return float(ordered_value[np.searchsorted(np.cumsum(ordered_weight), cutoff, side="left")])


def _effective_support(frame: pd.DataFrame) -> np.ndarray:
    before = pd.to_numeric(frame["before_favorable_net_support"], errors="coerce").to_numpy(float)
    after = pd.to_numeric(frame["after_favorable_net_support"], errors="coerce").to_numpy(float)
    result = np.zeros(len(frame), dtype=float)
    valid = (before > 0.0) & (after > 0.0)
    result[valid] = 1.0 / (1.0 / before[valid] + 1.0 / after[valid])
    return result


def _fit_priors(train: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    priors: dict[tuple[str, int, str], float] = {}
    groups = ["side_name", "frozen_base_score_decile"]
    for (side, decile), group in train.groupby(groups, observed=True, sort=True):
        for phase in ("before", "after"):
            priors[(str(side), int(decile), phase)] = _weighted_median(
                group[f"{phase}_conditional_favorable_net_robust_mean"],
                group[f"{phase}_favorable_net_support"],
            )
    return priors


def _empirical_bayes_delta(
    frame: pd.DataFrame, priors: Mapping[tuple[str, int, str], float]
) -> np.ndarray:
    smoothed: dict[str, np.ndarray] = {}
    for phase in ("before", "after"):
        mean = pd.to_numeric(
            frame[f"{phase}_conditional_favorable_net_robust_mean"],
            errors="coerce",
        ).to_numpy(float)
        support = pd.to_numeric(
            frame[f"{phase}_favorable_net_support"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        prior = np.asarray(
            [
                priors.get((str(side), int(decile), phase), np.nan)
                for side, decile in zip(
                    frame["side_name"],
                    frame["frozen_base_score_decile"],
                    strict=True,
                )
            ],
            dtype=float,
        )
        observed = np.where(np.isfinite(mean), mean, prior)
        smoothed[phase] = (
            support * observed + EB_PRIOR_SUPPORT * prior
        ) / (support + EB_PRIOR_SUPPORT)
    return smoothed["after"] - smoothed["before"]


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
        "before_favorable_net_support",
        "after_favorable_net_support",
        "before_conditional_favorable_net_robust_mean",
        "after_conditional_favorable_net_robust_mean",
        "delta_conditional_favorable_net_robust_mean",
    ]
    contribution_columns = [
        *KEY,
        "delta_positive_net_contribution",
        "delta_positive_net_contribution_robust_mean",
        "delta_net_positive_rate",
        "delta_loss_net_contribution",
    ]
    base = base.loc[:, base_columns].copy()
    contribution = contribution.loc[:, contribution_columns].copy()
    for frame in (base, contribution):
        frame["cohort_anchor_utc"] = pd.to_datetime(
            frame["cohort_anchor_utc"], utc=True, errors="raise"
        )
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
    base["before_target_available_utc"] = pd.to_datetime(
        base["before_target_available_utc"], utc=True, errors="coerce"
    )
    base["after_target_available_utc"] = pd.to_datetime(
        base["after_target_available_utc"], utc=True, errors="coerce"
    )
    base["label_available_utc"] = pd.concat(
        [base["before_target_available_utc"], base["after_target_available_utc"]],
        axis=1,
    ).max(axis=1)
    if base.duplicated(list(KEY)).any() or contribution.duplicated(list(KEY)).any():
        raise ValueError("target-ablation label identity is not one-to-one")
    population = (
        base.merge(contribution, on=list(KEY), how="inner", validate="one_to_one")
        .merge(context, on=list(COHORT_KEY), how="left", validate="many_to_one")
    )
    population = population.loc[
        population["horizon_hours"].eq(PRIMARY_HORIZON)
    ].copy()
    population["complete_target"] = (
        population["before_global_hour_complete_flag"].astype(bool)
        & population["after_global_hour_complete_flag"].astype(bool)
        & population["before_candidate_support"].gt(0)
        & population["after_candidate_support"].gt(0)
        & population["after_target_available_utc"].notna()
    )
    population["conditional_favorable_valid"] = (
        population["complete_target"]
        & population["before_favorable_net_support"].gt(0)
        & population["after_favorable_net_support"].gt(0)
        & population[
            "delta_conditional_favorable_net_robust_mean"
        ].notna()
    )
    if population.loc[:, features].isna().all(axis=1).any():
        raise ValueError("target-ablation cohort lacks all causal context")
    return population.sort_values(
        ["cohort_anchor_utc", "side_name", "frozen_base_score_decile"],
        kind="stable",
    ).reset_index(drop=True)


def _target_for_arm(
    frame: pd.DataFrame,
    arm: TargetArm,
    priors: Mapping[tuple[str, int, str], float] | None,
) -> np.ndarray:
    if arm.empirical_bayes:
        if priors is None:
            raise ValueError("empirical-Bayes arm requires fold-local priors")
        return _empirical_bayes_delta(frame, priors)
    return pd.to_numeric(frame[arm.target_column], errors="coerce").to_numpy(float)


def _valid_for_arm(frame: pd.DataFrame, arm: TargetArm, target: np.ndarray) -> np.ndarray:
    valid = frame["complete_target"].to_numpy(bool) & np.isfinite(target)
    if arm.family == "conditional_favorable" and not arm.empirical_bayes:
        valid &= frame["conditional_favorable_valid"].to_numpy(bool)
    return valid


def _arm_weights(frame: pd.DataFrame, arm: TargetArm) -> np.ndarray:
    if not arm.weighted:
        return np.ones(len(frame), dtype=float)
    return np.minimum(1.0, _effective_support(frame) / EB_PRIOR_SUPPORT)


def _metric_record(
    frame: pd.DataFrame,
    *,
    arm: TargetArm,
    fold_id: int,
    period: str,
) -> dict[str, Any]:
    valid = frame["target_valid"].astype(bool).to_numpy()
    scored = frame.loc[valid]
    y = scored["target_delta"].to_numpy(float)
    record: dict[str, Any] = {
        "target_arm": arm.name,
        "target_family": arm.family,
        "selection_target": arm.selection_target,
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
                record[f"{prefix}_regression_{name}"] = value
    sign = (y > 0.0).astype(np.int8)
    for prefix, values in (
        ("model", scored["sign_probability"].to_numpy(float)),
        ("constant", scored["constant_sign_probability"].to_numpy(float)),
    ):
        for name, value in _classification_metrics(sign, values).items():
            if name != "rows":
                record[f"{prefix}_sign_{name}"] = value
    return record


def fit_arm_oof(
    population: pd.DataFrame,
    *,
    arm: TargetArm,
    features: Iterable[str],
    folds: Iterable[Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = tuple(features)
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for fold in folds:
        start = pd.Timestamp(fold["validation_start_utc"])
        end = pd.Timestamp(fold["validation_end_utc"])
        evaluation = population.loc[
            population["cohort_anchor_utc"].ge(start)
            & population["cohort_anchor_utc"].lt(end)
        ].copy()
        allowed = population.loc[
            population["after_target_available_utc"].lt(start)
            & population["label_available_utc"].lt(start)
        ].copy()
        if allowed.empty or evaluation.empty:
            continue
        priors = _fit_priors(allowed) if arm.empirical_bayes else None
        allowed["__target__"] = _target_for_arm(allowed, arm, priors)
        evaluation["__target__"] = _target_for_arm(evaluation, arm, priors)
        allowed_valid = _valid_for_arm(
            allowed, arm, allowed["__target__"].to_numpy(float)
        )
        evaluation_valid = _valid_for_arm(
            evaluation, arm, evaluation["__target__"].to_numpy(float)
        )
        train = allowed.loc[allowed_valid].copy()
        train["__weight__"] = _arm_weights(train, arm)
        train = train.loc[train["__weight__"].gt(0.0)].copy()
        train_subset = _spread_train_subset(train, int(args.fit_budget_rows))
        y_train = train_subset["__target__"].to_numpy(float)
        weight_train = train_subset["__weight__"].to_numpy(float)
        constant = float(np.average(y_train, weights=weight_train))
        constant_sign = float(
            np.average((y_train > 0.0).astype(float), weights=weight_train)
        )
        result = evaluation.loc[
            :,
            [*KEY, "after_target_available_utc", "label_available_utc"],
        ].copy()
        result["target_arm"] = arm.name
        result["target_family"] = arm.family
        result["fold_id"] = int(fold["fold_id"])
        result["validation_start_utc"] = start
        result["validation_end_utc"] = end
        result["target_delta"] = evaluation["__target__"].to_numpy(float)
        result["economic_reference_delta"] = pd.to_numeric(
            evaluation["delta_conditional_favorable_net_robust_mean"],
            errors="coerce",
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
            x_train = train_subset.loc[:, features]
            x_eval = evaluation.loc[evaluation_valid, features]
            regressor = _catboost_regressor(
                seed=int(args.random_state + 10_000 * int(fold["fold_id"])),
                threads=int(args.threads),
            ).fit(x_train, y_train, sample_weight=weight_train)
            result.loc[evaluation_valid, "delta_prediction"] = regressor.predict(
                x_eval
            )
            if np.unique(y_train > 0.0).size == 2:
                classifier = _catboost_classifier(
                    seed=int(args.random_state + 20_000 * int(fold["fold_id"])),
                    threads=int(args.threads),
                ).fit(
                    x_train,
                    (y_train > 0.0).astype(np.int8),
                    sample_weight=weight_train,
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
        prediction_parts.append(result)
        metric_rows.append(
            _metric_record(
                result,
                arm=arm,
                fold_id=int(fold["fold_id"]),
                period="development"
                if int(fold["fold_id"]) < max(item["fold_id"] for item in folds)
                else "confirmation",
            )
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(metric_rows)


def _aggregate_period_metrics(
    predictions: pd.DataFrame, arms: Iterable[TargetArm]
) -> pd.DataFrame:
    arm_lookup = {arm.name: arm for arm in arms}
    latest_fold = int(predictions["fold_id"].max())
    records: list[dict[str, Any]] = []
    periods = {
        "development": predictions["fold_id"].lt(latest_fold),
        "confirmation": predictions["fold_id"].eq(latest_fold),
        "all_oof": pd.Series(True, index=predictions.index),
    }
    for arm_name in arm_lookup:
        for period in periods:
            group = predictions.loc[
                predictions["target_arm"].eq(arm_name) & periods[period]
            ]
            records.append(
                _metric_record(
                    group,
                    arm=arm_lookup[arm_name],
                    fold_id=latest_fold if period == "confirmation" else -1,
                    period=period,
                )
            )
    return pd.DataFrame.from_records(records)


def _side_decile_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    latest = int(predictions["fold_id"].max())
    records: list[dict[str, Any]] = []
    for (arm, side), group in predictions.loc[
        predictions["fold_id"].eq(latest)
    ].groupby(["target_arm", "side_name"], observed=True, sort=True):
        valid = group["target_valid"].astype(bool)
        metrics = _regression_metrics(
            group.loc[valid, "target_delta"].to_numpy(float),
            group.loc[valid, "delta_prediction"].to_numpy(float),
        )
        records.append(
            {
                "target_arm": arm,
                "side_name": side,
                "rows": int(valid.sum()),
                "rank_ic": metrics["rank_ic"],
                "mae": metrics["mae"],
            }
        )
    return pd.DataFrame.from_records(records)


def _gates(
    aggregate: pd.DataFrame, per_fold: pd.DataFrame, side_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        development = aggregate.loc[
            aggregate["target_arm"].eq(arm.name)
            & aggregate["period"].eq("development")
        ].iloc[0]
        confirmation = aggregate.loc[
            aggregate["target_arm"].eq(arm.name)
            & aggregate["period"].eq("confirmation")
        ].iloc[0]
        development_folds = per_fold.loc[
            per_fold["target_arm"].eq(arm.name)
            & per_fold["period"].eq("development")
        ]
        side = side_metrics.loc[side_metrics["target_arm"].eq(arm.name)]
        dev_relative_mae_gain = (
            development["constant_regression_mae"]
            - development["model_regression_mae"]
        ) / development["constant_regression_mae"]
        passed = bool(
            arm.selection_target
            and dev_relative_mae_gain >= 0.03
            and development["model_regression_rank_ic"] >= 0.18
            and development_folds["model_regression_rank_ic"].gt(0.0).sum() >= 3
            and confirmation["model_regression_mae"]
            <= confirmation["constant_regression_mae"]
            and confirmation["model_regression_rank_ic"] >= 0.10
            and len(side) == 2
            and side["rank_ic"].gt(0.0).all()
        )
        rows.append(
            {
                "target_arm": arm.name,
                "target_family": arm.family,
                "selection_target": arm.selection_target,
                "development_relative_mae_gain": float(dev_relative_mae_gain),
                "development_rank_ic": float(
                    development["model_regression_rank_ic"]
                ),
                "development_positive_ic_folds": int(
                    development_folds["model_regression_rank_ic"].gt(0.0).sum()
                ),
                "confirmation_mae_gain": float(
                    confirmation["constant_regression_mae"]
                    - confirmation["model_regression_mae"]
                ),
                "confirmation_rank_ic": float(
                    confirmation["model_regression_rank_ic"]
                ),
                "confirmation_min_side_rank_ic": float(side["rank_ic"].min()),
                "passes_predeclared_component_gate": passed,
                "status": "DIAGNOSTIC_ONLY_NOT_PROMOTION_ELIGIBLE",
            }
        )
    result = pd.DataFrame.from_records(rows)
    result["diagnostic_rank"] = (
        result.sort_values(
            [
                "passes_predeclared_component_gate",
                "development_relative_mae_gain",
                "confirmation_rank_ic",
            ],
            ascending=[False, False, False],
            kind="stable",
        )
        .reset_index()
        .reset_index()
        .set_index("index")["level_0"]
        .add(1)
        .reindex(result.index)
        .astype(int)
    )
    return result.sort_values("diagnostic_rank", kind="stable").reset_index(drop=True)


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
        "horizon_hours": PRIMARY_HORIZON,
        "feature_count": len(features),
        "target_arms": [_safe(arm.__dict__) for arm in ARMS],
        "empirical_bayes_prior_support": EB_PRIOR_SUPPORT,
        "fold_selection": "folds 0-3 development; final truncated fold confirmation",
        "scope": "component target learnability only; no score routing, PnL, admission, or portfolio replay",
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
    for arm in ARMS:
        predictions, metrics = fit_arm_oof(
            population, arm=arm, features=features, folds=folds, args=args
        )
        prediction_parts.append(predictions)
        fold_parts.append(metrics)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    per_fold = pd.concat(fold_parts, ignore_index=True)
    aggregate = _aggregate_period_metrics(predictions, ARMS)
    side_metrics = _side_decile_metrics(predictions)
    gates = _gates(aggregate, per_fold, side_metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    frames = {
        "oof_target_predictions.parquet": predictions,
        "per_fold_target_metrics.parquet": per_fold,
        "period_target_metrics.parquet": aggregate,
        "confirmation_side_metrics.parquet": side_metrics,
        "target_gates.parquet": gates,
    }
    for name, frame in frames.items():
        frame.to_parquet(temporary / name, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "IMMUTABLE_COMPONENT_TARGET_ABLATION_NOT_PROMOTION_ELIGIBLE",
        "promotion_eligible": False,
        "source_artifacts_sha256": {**hashes, **contribution_hashes},
        "source_panel_identity_sha256": manifests["context"].get(
            "source_panel_identity_sha256"
        ),
        "contribution_schema": contribution_manifest["schema"],
        "horizon_hours": PRIMARY_HORIZON,
        "context_feature_columns": list(features),
        "target_arms": [_safe(arm.__dict__) for arm in ARMS],
        "fixed_catboost_geometry": dict(CATBOOST_GEOMETRY),
        "empirical_bayes_prior_support": EB_PRIOR_SUPPORT,
        "folds": folds,
        "contracts": {
            "training_availability": "every training label and every fold-local prior resolves strictly before validation start",
            "support_weight": "min(1, harmonic before/after positive support / 16); support is loss metadata, never a model feature",
            "empirical_bayes": "separate before/after side-decile weighted-median priors estimated from permitted training rows only",
            "selection": "folds before the truncated latest fold select; latest fold is confirmation; side metrics must not collapse",
            "scope": "component learnability only; no admission, PnL, policy, or portfolio replay",
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
        "target_arms": len(ARMS),
        "oof_predictions": int(len(predictions)),
        "passing_component_arms": int(
            gates["passes_predeclared_component_gate"].sum()
        ),
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
