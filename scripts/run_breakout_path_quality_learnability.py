#!/usr/bin/env python3
"""Chronologically test pre-entry learnability of refined breakout path labels.

This is deliberately a target-validation experiment, not a policy search. It
uses one frozen, observable feature set for all models and folds, applies
train-derived label thresholds to every scored fold, and reports ranking and
calibration without changing base/meta/policy behavior.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler

from extreme_price_movements.breakout_path_quality_labels import (
    fit_breakout_path_quality_thresholds,
    fit_severe_retention_threshold,
    materialize_breakout_path_quality_labels,
    materialize_severe_retention_failure,
)
from scripts.validate_breakout_path_quality_labels import _derive_outcomes, _quarter_starts


GROUP = ("short", "short_breakout_precision")
IDENTITY_COLUMNS = ("__symbol__", "candidate_id")
METADATA_COLUMNS = {
    "__ts__", "side_name", "__archetype_policy_key__", "side", "timeframe",
    "candidate_id", "G_VOL", "ret1h_G_VOL_0", "ret1h_G_VOL_1", "strategy_id", "source_tag",
}
OUTCOME_COLUMNS = {
    "__path_trailing_success__", "__first_touch_mfe_norm__",
    "__first_touch_full_path_mae_norm__", "__first_touch_mfe_to_tp__",
    "__path_post_mfe_drawdown_norm__", "__first_touch_capture_net__",
}
REQUIRED = [*METADATA_COLUMNS.intersection({"__ts__", "side_name", "__archetype_policy_key__"}), *OUTCOME_COLUMNS]
LAGGED_PATH_STATE_PREFIXES = (
    "dir_path_",
    "up_barrier_pressure_",
    "down_barrier_pressure_",
)
FEATURE_VARIANTS = ("all_observable", "exclude_lagged_path_state")


def _feature_columns(schema: list[str], variant: str) -> list[str]:
    if variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant {variant!r}; expected one of {FEATURE_VARIANTS}")
    output = sorted(
        column for column in schema
        if not column.startswith("__") and column not in METADATA_COLUMNS
    )
    if variant == "exclude_lagged_path_state":
        output = [
            column for column in output
            if not column.startswith(LAGGED_PATH_STATE_PREFIXES)
        ]
    return output


def _feature_provenance(features: list[str]) -> dict[str, object]:
    """Describe the fixed inference contract without treating lagged states as labels."""

    lagged_path_state = [
        column for column in features
        if column.startswith(LAGGED_PATH_STATE_PREFIXES)
    ]
    return {
        "feature_count": len(features),
        "lagged_path_state_features": lagged_path_state,
        "lagged_path_state_count": len(lagged_path_state),
        "outcome_or_target_named_features": [
            column for column in features
            if any(token in column.lower() for token in ("target", "label", "outcome", "mfe", "mae"))
        ],
        "contract": (
            "Lagged path-state features are previous-bar observable indicators, not realized "
            "outcomes for the current decision. The conservative ablation removes them to "
            "quantify dependence on short-horizon path persistence."
        ),
    }


def _partition_month(path: Path) -> pd.Period:
    """Return the UTC month encoded by the monthly label partition filename."""

    try:
        return pd.Period("-".join(path.stem.rsplit("_", 2)[-2:]), freq="M")
    except ValueError as exc:
        raise ValueError(f"Expected a YYYY_MM label partition suffix: {path.name}") from exc


def _eligible_paths(labels_dir: Path, end: pd.Timestamp) -> list[Path]:
    end_month = end.tz_localize(None).to_period("M")
    paths = [
        path for path in sorted(labels_dir.glob("train_global_short_5_*.parquet"))
        if _partition_month(path) < end_month
    ]
    if not paths:
        raise FileNotFoundError("No short global label partitions before evaluation end")
    return paths


def _common_features(paths: list[Path], feature_variant: str) -> list[str]:
    schema: list[str] | None = None
    for path in paths:
        names = pq.read_schema(path).names
        if schema is None:
            schema = names
        elif set(schema) != set(names):
            schema = sorted(set(schema).intersection(names))
    if schema is None:
        raise FileNotFoundError("No short global label partitions found")
    return _feature_columns(schema, feature_variant)


def _read_group(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only one monthly partition and retain the intended local group."""

    filters = [
        ("side_name", "==", GROUP[0]),
        ("__archetype_policy_key__", "==", GROUP[1]),
    ]
    frame = pd.read_parquet(path, columns=columns, filters=filters)
    # The filter is a performance hint; retain the contract if an engine ignores it.
    return frame.loc[
        frame["side_name"].eq(GROUP[0])
        & frame["__archetype_policy_key__"].eq(GROUP[1])
    ].reset_index(drop=True)


def _load_core(paths: list[Path]) -> pd.DataFrame:
    columns = [
        "__ts__", "side_name", "__archetype_policy_key__", *IDENTITY_COLUMNS,
        *OUTCOME_COLUMNS,
    ]
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_group(path, columns)
        if frame.empty:
            continue
        frame["__source_path__"] = str(path)
        frame["__source_row__"] = np.arange(len(frame), dtype=np.int32)
        parts.append(frame)
    if not parts:
        raise RuntimeError("No short-breakout rows with the required outcome schema")
    output = pd.concat(parts, ignore_index=True, copy=False)
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="coerce")
    return output.dropna(subset=["__ts__"]).sort_values("__ts__", kind="stable").reset_index(drop=True)


def _raw_matrix(core: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Stream feature columns from monthly partitions into a bounded float32 matrix."""

    values = np.full((len(core), len(features)), np.nan, dtype=np.float32)
    columns = ["side_name", "__archetype_policy_key__", *features]
    for source_path, group in core.groupby("__source_path__", sort=False, observed=True):
        frame = _read_group(Path(source_path), columns)
        local_rows = group["__source_row__"].to_numpy(np.int64, copy=False)
        positions = group.index.to_numpy(np.int64, copy=False)
        values[positions] = frame.iloc[local_rows].loc[:, features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32, copy=False)
        del frame
    return values


def _matrix(frame: pd.DataFrame, features: list[str], medians: np.ndarray) -> np.ndarray:
    values = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32, copy=False)
    return np.where(np.isfinite(values), values, medians).astype(np.float32, copy=False)


def _sample_train(n_rows: int, target: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Sample indices before materializing the dense feature matrix."""

    if n_rows <= maximum:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    pieces = np.array_split(np.arange(n_rows), 3)
    selected: list[np.ndarray] = []
    budget = max(maximum // 3, 1)
    for piece in pieces:
        if len(piece) <= budget:
            selected.append(piece)
            continue
        # Preserve both classes inside each chronological segment where possible.
        labels = target[piece]
        local: list[np.ndarray] = []
        for value in (0, 1):
            group = piece[labels == value]
            count = min(len(group), max(1, round(budget * len(group) / max(len(piece), 1))))
            if count:
                local.append(rng.choice(group, size=count, replace=False))
        selected.append(np.concatenate(local) if local else rng.choice(piece, size=budget, replace=False))
    return np.sort(np.concatenate(selected))[:maximum]


def _top_decile_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(score)
    target, score = target[valid], score[valid]
    if len(target) == 0 or np.unique(target).size < 2:
        return {"roc_auc": np.nan, "average_precision": np.nan, "top10_precision": np.nan, "top10_lift": np.nan, "brier": np.nan, "ece10": np.nan}
    size = max(1, int(np.ceil(len(target) * 0.10)))
    selected = np.argsort(-score, kind="stable")[:size]
    prevalence = float(target.mean())
    precision = float(target[selected].mean())
    bins = np.minimum((score * 10).astype(np.int8), 9)
    ece = 0.0
    for bucket in range(10):
        mask = bins == bucket
        if mask.any():
            ece += float(mask.mean() * abs(score[mask].mean() - target[mask].mean()))
    return {
        "roc_auc": float(roc_auc_score(target, score)),
        "average_precision": float(average_precision_score(target, score)),
        "top10_precision": precision,
        "top10_lift": precision / prevalence if prevalence else np.nan,
        "brier": float(brier_score_loss(target, score)),
        "ece10": ece,
    }


def _fit_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if model_name == "lgbm":
        positives = max(int(y_train.sum()), 1)
        negatives = max(int(len(y_train) - positives), 1)
        model = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss",
                "learning_rate": 0.04, "num_leaves": 7, "max_depth": 3,
                "min_data_in_leaf": 300, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 1, "lambda_l1": 1.0, "lambda_l2": 5.0,
                "scale_pos_weight": negatives / positives, "seed": seed,
                "feature_pre_filter": False, "verbosity": -1, "num_threads": -1,
            },
            lgb.Dataset(x_train, label=y_train, free_raw_data=True),
            num_boost_round=300,
        )
        importance = pd.DataFrame({"feature_index": np.arange(x_train.shape[1]), "importance": model.feature_importance(importance_type="gain").astype(float)})
        return model.predict(x_eval).astype(np.float32), model.predict(x_train).astype(np.float32), importance
    if model_name == "logistic":
        scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(x_train)
        model = LogisticRegression(
            C=0.03, penalty="l2", solver="liblinear", class_weight="balanced",
            max_iter=2_000, random_state=seed,
        )
        model.fit(scaler.transform(x_train), y_train)
        importance = pd.DataFrame({"feature_index": np.arange(x_train.shape[1]), "importance": np.abs(model.coef_[0])})
        return (
            model.predict_proba(scaler.transform(x_eval))[:, 1].astype(np.float32),
            model.predict_proba(scaler.transform(x_train))[:, 1].astype(np.float32),
            importance,
        )
    if model_name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        model = ExplainableBoostingClassifier(
            max_bins=32, max_rounds=120, min_samples_leaf=200, interactions=4,
            outer_bags=2, validation_size=0.15, random_state=seed, n_jobs=1,
        )
        model.fit(x_train, y_train)
        importance = pd.DataFrame({"feature_index": np.arange(x_train.shape[1]), "importance": np.asarray(model.term_importances())[:x_train.shape[1]]})
        return (
            model.predict_proba(x_eval)[:, 1].astype(np.float32),
            model.predict_proba(x_train)[:, 1].astype(np.float32),
            importance,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _targets(train: pd.DataFrame, scored: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    train_outcomes = _derive_outcomes(train)
    scored_outcomes = _derive_outcomes(scored)
    path_thresholds = fit_breakout_path_quality_thresholds(train_outcomes)
    train_path = materialize_breakout_path_quality_labels(train_outcomes, path_thresholds)
    score_path = materialize_breakout_path_quality_labels(scored_outcomes, path_thresholds)
    severe_threshold = fit_severe_retention_threshold(train["__first_touch_capture_net__"])
    return (
        {
            "rapid_reversal": train_path["breakout_rapid_reversal"].to_numpy(np.int8),
            "severe_retention": materialize_severe_retention_failure(train["__path_trailing_success__"], train["__first_touch_capture_net__"], severe_threshold).to_numpy(np.int8),
        },
        {
            "rapid_reversal": score_path["breakout_rapid_reversal"].to_numpy(np.int8),
            "severe_retention": materialize_severe_retention_failure(scored["__path_trailing_success__"], scored["__first_touch_capture_net__"], severe_threshold).to_numpy(np.int8),
        },
        {"rapid_reversal_high": path_thresholds.reversal_high, "severe_retention_capture_net_low": severe_threshold.capture_net_low},
    )


def _pre_entry_reliability(
    x_train: np.ndarray,
    x_score: np.ndarray,
    ebm_train_score: np.ndarray,
    model_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct inference-time confidence without using scored outcomes.

    Calibration errors are deliberately excluded: they must come from a nested
    chronological validation set, not from the OOS rows being emitted here.
    """

    median = np.median(x_train, axis=0)
    iqr = np.subtract(*np.percentile(x_train, [75.0, 25.0], axis=0))
    scale = np.maximum(iqr, np.float32(1e-3))
    normalized_distance = np.minimum(np.abs((x_score - median) / scale), 6.0).mean(axis=1) / 6.0
    distribution_reliability = np.clip(1.0 - normalized_distance, 0.0, 1.0).astype(np.float32)

    edges = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    train_bins = np.clip(np.digitize(ebm_train_score, edges, right=False) - 1, 0, 9)
    score_bins = np.clip(np.digitize(model_scores["ebm"], edges, right=False) - 1, 0, 9)
    support = np.bincount(train_bins, minlength=10).astype(np.float32)[score_bins]
    support_reliability = (support / (support + 200.0)).astype(np.float32)

    prediction_stack = np.vstack([model_scores[name] for name in sorted(model_scores)])
    disagreement = np.std(prediction_stack, axis=0).astype(np.float32)
    agreement_reliability = np.clip(1.0 - disagreement / 0.25, 0.0, 1.0).astype(np.float32)
    probability_reliability = np.sqrt(
        distribution_reliability * support_reliability
    ).astype(np.float32) * agreement_reliability
    return {
        "feature_distribution_reliability": distribution_reliability,
        "score_bin_support": support,
        "score_bin_support_reliability": support_reliability,
        "model_disagreement": disagreement,
        "model_agreement_reliability": agreement_reliability,
        "probability_reliability": probability_reliability.astype(np.float32),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.eval_start, tz="UTC"), pd.Timestamp(args.eval_end, tz="UTC")
    paths = _eligible_paths(args.labels_dir, end)
    features = _common_features(paths, args.feature_variant)
    values = _load_core(paths)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    requested_models = tuple(args.models.split(","))
    requested_targets = tuple(args.targets.split(","))
    for fold_start in _quarter_starts(start, end):
        fold_end = min(fold_start + pd.DateOffset(months=3), end)
        # Labels resolve after entry. Exclude the final authorized horizon before
        # each scored quarter so no train outcome can overlap the OOS label path.
        purge_start = fold_start - pd.Timedelta(hours=args.purge_hours)
        train = values.loc[values["__ts__"].lt(purge_start)].reset_index(drop=True)
        scored = values.loc[
            values["__ts__"].ge(fold_start) & values["__ts__"].lt(fold_end)
        ].reset_index(drop=True)
        if len(train) < args.minimum_train_rows or len(scored) < args.minimum_eval_rows:
            continue
        y_train, y_score, thresholds = _targets(train, scored)
        for target_name in requested_targets:
            target_train, target_score = y_train[target_name], y_score[target_name]
            if target_train.sum() < args.minimum_positive_rows or np.unique(target_score).size < 2:
                continue
            sample = _sample_train(len(train), target_train, args.max_train_rows, args.seed)
            x_train = _raw_matrix(train.iloc[sample].reset_index(drop=True), features)
            medians = np.nanmedian(x_train, axis=0).astype(np.float32, copy=False)
            medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32, copy=False)
            x_train = np.where(np.isfinite(x_train), x_train, medians).astype(np.float32, copy=False)
            x_score = _raw_matrix(scored, features)
            x_score = np.where(np.isfinite(x_score), x_score, medians).astype(np.float32, copy=False)
            model_scores: dict[str, np.ndarray] = {}
            train_scores: dict[str, np.ndarray] = {}
            model_importances: dict[str, pd.DataFrame] = {}
            for model_name in requested_models:
                score, train_score, importance = _fit_predict(
                    model_name, x_train, target_train[sample], x_score, args.seed
                )
                model_scores[model_name] = score
                train_scores[model_name] = train_score
                model_importances[model_name] = importance
            reliability = (
                _pre_entry_reliability(x_train, x_score, train_scores["ebm"], model_scores)
                if {"ebm", "lgbm", "logistic"}.issubset(model_scores)
                else {}
            )
            for model_name in requested_models:
                score = model_scores[model_name]
                importance = model_importances[model_name]
                metrics = _top_decile_metrics(target_score, score)
                rows.append({
                    "fold_start": fold_start, "fold_end": fold_end, "target": target_name,
                    "model": model_name, "train_rows": int(len(train)), "fit_rows": int(len(sample)),
                    "eval_rows": int(len(scored)), "train_prevalence": float(target_train.mean()),
                    "eval_prevalence": float(target_score.mean()), "purge_hours": int(args.purge_hours),
                    **thresholds, **metrics,
                })
                part = scored.loc[:, ["__ts__", *IDENTITY_COLUMNS, "side_name", "__archetype_policy_key__"]].copy()
                part["fold_start"], part["target"], part["model"] = fold_start, target_name, model_name
                part["target_realized"] = target_score
                part["prediction"] = score
                if reliability:
                    for name, value in reliability.items():
                        part[name] = value
                predictions.append(part)
                importance["feature"] = [features[index] for index in importance["feature_index"]]
                importance["fold_start"], importance["target"], importance["model"] = fold_start, target_name, model_name
                importances.append(importance.drop(columns="feature_index"))
            del x_train, x_score, model_scores, train_scores, model_importances
            gc.collect()
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "chronological_learnability_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True, copy=False).to_parquet(args.output / "oof_predictions.parquet", index=False, compression="zstd")
    if importances:
        pd.concat(importances, ignore_index=True, copy=False).to_csv(args.output / "feature_importance_by_fold.csv", index=False)
    summary = metrics.groupby(["target", "model"], observed=True).agg(
        folds=("fold_start", "nunique"), mean_auc=("roc_auc", "mean"), min_auc=("roc_auc", "min"),
        mean_top10_lift=("top10_lift", "mean"), min_top10_lift=("top10_lift", "min"),
        mean_brier=("brier", "mean"), mean_ece10=("ece10", "mean"),
    ).reset_index() if not metrics.empty else pd.DataFrame()
    if not summary.empty:
        summary["learnable"] = (
            summary["mean_auc"].gt(0.55)
            & summary["min_auc"].gt(0.50)
            & summary["mean_top10_lift"].gt(1.25)
            & summary["min_top10_lift"].gt(1.0)
        )
    summary.to_csv(args.output / "learnability_summary.csv", index=False)
    feature_hash = hashlib.sha256("\n".join(features).encode()).hexdigest()
    manifest = {
        "schema": "breakout_path_quality_learnability_v2",
        "status": "diagnostic_only_no_policy_effect",
        "scope": {"side": GROUP[0], "archetype": GROUP[1]},
        "targets": list(requested_targets), "models": list(requested_models),
        "feature_variant": args.feature_variant,
        "frozen_pre_entry_features": features, "feature_schema_hash": feature_hash,
        "feature_provenance": _feature_provenance(features),
        "probability_reliability_contract": (
            "When all three models are requested, reliability is the product of train-derived "
            "score-bin support, feature-distribution proximity, and EBM/LGBM/logistic agreement. "
            "It contains no scored-fold outcomes or post-hoc calibration error."
        ),
        "leakage_contract": (
            "Features exclude every __* realized outcome. Label cutoffs, imputers, scalers, "
            f"and models fit only before each scored fold; the final {int(args.purge_hours)}h "
            "before the scored fold is purged from training."
        ),
        "selection_contract": "No thresholds, ranks, base/meta predictions, or live policy are altered.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return {**manifest, "metric_rows": int(len(metrics)), "summary_rows": int(len(summary))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-start", default="2025-07-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--models", default="lgbm,logistic,ebm")
    parser.add_argument("--targets", default="rapid_reversal,severe_retention")
    parser.add_argument("--minimum-train-rows", type=int, default=10_000)
    parser.add_argument("--minimum-eval-rows", type=int, default=1_000)
    parser.add_argument("--minimum-positive-rows", type=int, default=100)
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--purge-hours", type=int, default=8)
    parser.add_argument("--feature-variant", choices=FEATURE_VARIANTS, default="all_observable")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
