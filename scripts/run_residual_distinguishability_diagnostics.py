#!/usr/bin/env python3
"""Test whether residual adverse episodes are distinguishable from benign lookalikes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
)
from scripts import run_meta_residual_event_balanced_error_overlay as overlay


TARGET_GROUPS = {
    ("long", "long_volcompression_wideslow_candidate"): 3,
    ("long", "long_breakout_diagnostic_candidate"): 2,
    ("short", "short_breakout_precision"): 2,
    ("short", "short_default_clean_path"): 5,
    ("short", "short_mixed_clean_path"): 3,
}
FOLD_STARTS = overlay.FOLD_STARTS
FINAL_TRAIN_END = pd.Timestamp("2026-04-01", tz="UTC")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _feature_schema_hash(features: list[str]) -> str:
    payload = json.dumps([str(feature) for feature in features], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return (-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))).astype(
        np.float32
    )


def _difficulty_label(
    ensemble_std_pct: np.ndarray,
    neighbor_entropy: np.ndarray,
    neighbor_distance_pct: np.ndarray,
) -> np.ndarray:
    std = np.asarray(ensemble_std_pct, dtype=np.float32)
    entropy = np.asarray(neighbor_entropy, dtype=np.float32)
    distance = np.asarray(neighbor_distance_pct, dtype=np.float32)
    result = np.full(len(std), "medium", dtype=object)
    easy = (std < 0.50) & (entropy < 0.50) & (distance < 0.75)
    hard = (std >= 0.75) | (entropy >= 0.75) | (distance >= 0.90)
    ambiguous = (std >= 0.75) & (entropy >= 0.80) & (distance < 0.90)
    result[easy] = "easy"
    result[hard] = "hard"
    result[ambiguous] = "ambiguous"
    return result


def _feature_catalog(model_report: Path) -> dict[tuple[str, str], list[str]]:
    report = pd.read_csv(model_report)
    final = report.loc[report["stage"].astype(str).eq("final")]
    result: dict[tuple[str, str], list[str]] = {}
    for row in final.itertuples(index=False):
        key = (str(row.side_name), str(row.archetype_policy_key))
        features = [
            name
            for name in str(row.selected_features).split("|")
            if name and name != "nan"
        ]
        result[key] = list(
            dict.fromkeys([*features, *NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS])
        )
    return result


def _robust_matrix(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_fit = fit[features].to_numpy(np.float32, copy=True)
    x_score = score[features].to_numpy(np.float32, copy=True)
    median = np.nanmedian(x_fit, axis=0).astype(np.float32)
    q25 = np.nanquantile(x_fit, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(x_fit, 0.75, axis=0).astype(np.float32)
    median = np.nan_to_num(median, nan=0.0)
    scale = np.maximum(q75 - q25, np.float32(1e-4))
    for matrix in (x_fit, x_score):
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(median, np.nonzero(missing)[1])
        matrix -= median
        matrix /= scale
        np.clip(matrix, -5.0, 5.0, out=matrix)
    return x_fit, x_score


def _ensemble_and_neighbors(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    *,
    seeds: int,
    seed: int,
    neighbor_count: int,
    neighbor_shrinkage: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    x_fit, x_score = _robust_matrix(fit, score, features)
    y = fit[overlay.TARGET].to_numpy(np.float32)
    base_weights = overlay._sample_weights(fit)
    days, day_codes = np.unique(fit["day"].to_numpy(), return_inverse=True)
    predictions: list[np.ndarray] = []
    importance_rows: list[dict[str, Any]] = []
    for model_index in range(seeds):
        rng = np.random.default_rng(seed + model_index)
        day_weight = rng.exponential(1.0, len(days)).astype(np.float32)
        weights = base_weights * day_weight[day_codes]
        weights /= max(float(weights.mean()), 1e-8)
        model = lgb.train(
            {
                "objective": "binary",
                "metric": "None",
                "learning_rate": 0.035,
                "max_depth": 3,
                "num_leaves": 7,
                "min_data_in_leaf": 50,
                "min_gain_to_split": 0.02,
                "lambda_l1": 1.0,
                "lambda_l2": 10.0,
                "feature_fraction": 0.80,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "seed": seed + model_index,
                "feature_fraction_seed": seed + model_index,
                "bagging_seed": seed + model_index,
                "verbosity": -1,
                "force_col_wise": True,
            },
            lgb.Dataset(x_fit, label=y, weight=weights, feature_name=features),
            num_boost_round=160,
        )
        predictions.append(np.asarray(model.predict(x_score), dtype=np.float32))
        gains = model.feature_importance(importance_type="gain")
        for name, gain in zip(features, gains, strict=True):
            importance_rows.append(
                {"model_index": model_index, "feature": name, "gain": float(gain)}
            )
    prediction_matrix = np.column_stack(predictions).astype(np.float32, copy=False)
    mean = prediction_matrix.mean(axis=1).astype(np.float32)
    std = prediction_matrix.std(axis=1).astype(np.float32)

    k = min(max(int(neighbor_count), 1), len(fit))
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1).fit(x_fit)
    distance, index = neighbors.kneighbors(x_score, return_distance=True)
    neighbor_target = y[index]
    neighbor_ev = fit["ev_after_1pct"].to_numpy(np.float32)[index]
    neighbor_rate = neighbor_target.mean(axis=1).astype(np.float32)
    bandwidth = np.maximum(
        np.nanmedian(distance, axis=1, keepdims=True), np.float32(1e-4)
    )
    neighbor_weight = np.exp(-np.square(distance / bandwidth)).astype(np.float32)
    weight_sum = np.maximum(neighbor_weight.sum(axis=1), np.float32(1e-8))
    normalized_weight = neighbor_weight / weight_sum[:, None]
    weighted_rate = np.sum(normalized_weight * neighbor_target, axis=1).astype(np.float32)
    weighted_ev_mean = np.sum(normalized_weight * neighbor_ev, axis=1).astype(np.float32)
    weighted_ev_var = np.sum(
        normalized_weight * np.square(neighbor_ev - weighted_ev_mean[:, None]), axis=1
    )
    weighted_ev_std = np.sqrt(np.maximum(weighted_ev_var, 0.0)).astype(np.float32)
    effective_count = np.square(weight_sum) / np.maximum(
        np.square(neighbor_weight).sum(axis=1), np.float32(1e-8)
    )
    support = effective_count / (
        effective_count + np.float32(max(neighbor_shrinkage, 0.0))
    )
    train_prior = np.float32(np.mean(y))
    shrunken_rate = (
        support * weighted_rate + (1.0 - support) * train_prior
    ).astype(np.float32)
    neighbor_rows: list[dict[str, Any]] = []
    event_positions = np.flatnonzero(score[overlay.EVENT].to_numpy(bool))
    fit_days = fit["day"].to_numpy()
    fit_timestamps = fit["__ts__"].to_numpy()
    fit_ev = fit["ev_after_1pct"].to_numpy(np.float32)
    fit_clean = fit["clean_exec"].to_numpy(np.float32)
    for score_position in event_positions:
        for neighbor_rank, fit_position in enumerate(index[score_position], start=1):
            neighbor_rows.append(
                {
                    "query_timestamp": score.iloc[score_position]["__ts__"],
                    "query_day": score.iloc[score_position]["day"],
                    "side_name": score.iloc[score_position]["side_name"],
                    "archetype_policy_key": score.iloc[score_position][
                        "archetype_policy_key"
                    ],
                    "neighbor_rank": neighbor_rank,
                    "neighbor_timestamp": fit_timestamps[fit_position],
                    "neighbor_day": fit_days[fit_position],
                    "neighbor_distance": float(distance[score_position, neighbor_rank - 1]),
                    "neighbor_adverse": int(y[fit_position]),
                    "neighbor_ev": float(fit_ev[fit_position]),
                    "neighbor_clean": float(fit_clean[fit_position]),
                }
            )
    result = score.loc[
        :,
        [
            "__ts__",
            "day",
            "side_name",
            "archetype_policy_key",
            "parent_rank_v9",
            "ev_after_1pct",
            "clean_exec",
            overlay.EVENT,
            overlay.TARGET,
        ],
    ].copy()
    result["ensemble_risk_mean"] = mean
    result["ensemble_risk_std"] = std
    result["ensemble_predictive_entropy"] = _binary_entropy(mean)
    result["neighbor_adverse_rate"] = neighbor_rate
    result["neighbor_outcome_entropy"] = _binary_entropy(neighbor_rate)
    result["neighbor_ev_mean"] = np.nanmean(neighbor_ev, axis=1).astype(np.float32)
    result["neighbor_ev_std"] = np.nanstd(neighbor_ev, axis=1).astype(np.float32)
    result["neighbor_weighted_adverse_rate"] = weighted_rate
    result["neighbor_shrunken_adverse_rate"] = shrunken_rate
    result["neighbor_weighted_outcome_entropy"] = _binary_entropy(shrunken_rate)
    result["neighbor_weighted_ev_mean"] = weighted_ev_mean
    result["neighbor_weighted_ev_std"] = weighted_ev_std
    result["neighbor_effective_count"] = effective_count.astype(np.float32)
    result["neighbor_reliability"] = support.astype(np.float32)
    result["neighbor_train_adverse_prior"] = train_prior
    result["nearest_neighbor_distance"] = distance[:, 0].astype(np.float32)
    result["neighbor_distance_mean"] = distance.mean(axis=1).astype(np.float32)
    return result, importance_rows, neighbor_rows


def _percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite = np.sort(reference[np.isfinite(reference)])
    if not len(finite):
        return np.full(len(values), 0.5, dtype=np.float32)
    return (
        np.searchsorted(finite, values, side="right") / max(len(finite), 1)
    ).astype(np.float32)


def _episode_frame(state: pd.DataFrame, features: list[str], days: int) -> pd.DataFrame:
    daily = (
        state.groupby("day", observed=True, sort=True)
        .agg(
            **{name: (name, "median") for name in features},
            adverse=(overlay.TARGET, "max"),
            mean_ev=("ev_after_1pct", "mean"),
        )
        .sort_index()
    )
    output = pd.DataFrame(index=daily.index)
    minimum = max(2, days // 2)
    for name in features:
        rolling = daily[name].rolling(days, min_periods=minimum)
        output[f"{name}__mean"] = rolling.mean()
        output[f"{name}__std"] = rolling.std(ddof=0)
        output[f"{name}__change"] = daily[name] - daily[name].shift(days - 1)
    output["adverse"] = daily["adverse"].astype(np.int8)
    output["mean_ev"] = daily["mean_ev"].astype(np.float32)
    return output.reset_index()


def _fit_episode_gmm(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    x_fit, x_score = _robust_matrix(fit, score, features)
    components = min(8, x_fit.shape[1], max(2, len(fit) // 20))
    pca = PCA(n_components=components, random_state=seed).fit(x_fit)
    z_fit = pca.transform(x_fit).astype(np.float32)
    z_score = pca.transform(x_score).astype(np.float32)
    candidates: list[tuple[float, GaussianMixture]] = []
    for clusters in range(2, min(5, len(fit) // 15) + 1):
        model = GaussianMixture(
            n_components=clusters,
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=3,
            max_iter=250,
            random_state=seed + clusters,
        ).fit(z_fit)
        candidates.append((float(model.bic(z_fit)), model))
    if not candidates:
        raise RuntimeError("Insufficient episode support for GMM")
    _, model = min(candidates, key=lambda item: item[0])
    fit_cluster = model.predict(z_fit)
    score_cluster = model.predict(z_score)
    posterior = model.predict_proba(z_score).astype(np.float32)
    global_rate = float(fit["adverse"].mean())
    priors: dict[int, float] = {}
    support: dict[int, int] = {}
    for cluster in range(model.n_components):
        local = fit_cluster == cluster
        n = int(local.sum())
        rate = float(fit.loc[local, "adverse"].mean()) if n else global_rate
        priors[cluster] = (n * rate + 20.0 * global_rate) / (n + 20.0)
        support[cluster] = n
    result = score.loc[:, ["day", "adverse", "mean_ev"]].copy()
    result["episode_cluster_id"] = score_cluster.astype(np.int16)
    result["episode_cluster_adverse_prior"] = np.asarray(
        [priors[int(cluster)] for cluster in score_cluster], dtype=np.float32
    )
    result["episode_cluster_support"] = np.asarray(
        [support[int(cluster)] for cluster in score_cluster], dtype=np.int32
    )
    result["episode_posterior_max"] = posterior.max(axis=1)
    result["episode_posterior_entropy"] = (
        -np.sum(posterior * np.log(np.clip(posterior, 1e-8, 1.0)), axis=1)
        / max(math.log(model.n_components), 1e-8)
    ).astype(np.float32)
    return result, {
        "clusters": model.n_components,
        "pca_components": components,
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "cluster_priors": priors,
        "cluster_support": support,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    config = overlay.Config(
        train_start=args.train_start,
        train_end=args.train_end,
        eval_end=args.eval_end,
        max_features=args.max_features,
        targeted_temporal_features=min(8, args.max_features),
        seed=args.seed,
    )
    train, valid, coverage = overlay._load_frames(args, config)
    events = overlay._load_event_cells(args.event_calendar, args.extension_calendar)
    train = overlay._attach_event_target(train, events)
    valid = overlay._attach_event_target(valid, events)
    expected = overlay._fit_expected_clean_baseline(train, top10_floor=0.90)
    train, _ = overlay._v9_residual_calendar(
        train, top10_floor=0.90, expected_clean_baseline=expected
    )
    valid, _ = overlay._v9_residual_calendar(
        valid, top10_floor=0.90, expected_clean_baseline=expected
    )
    catalog = _feature_catalog(args.model_report)
    oof_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    matched_neighbor_rows: list[dict[str, Any]] = []
    episode_parts: list[pd.DataFrame] = []
    episode_manifests: list[dict[str, Any]] = []
    feature_schema_rows: list[dict[str, Any]] = []
    neighbor_index_parts: list[pd.DataFrame] = []

    for group_index, (key, episode_days) in enumerate(TARGET_GROUPS.items()):
        side, archetype = key
        candidates = [
            name for name in catalog.get(key, overlay._candidate_features(train.columns))
            if name in train and name in valid
        ]
        local_train = train.loc[
            train["side_name"].astype(str).eq(side)
            & train["archetype_policy_key"].astype(str).eq(archetype)
            & train["parent_rank_v9"].ge(0.80)
        ].sort_values("__ts__", kind="stable")
        local_valid = valid.loc[
            valid["side_name"].astype(str).eq(side)
            & valid["archetype_policy_key"].astype(str).eq(archetype)
            & valid["parent_rank_v9"].ge(0.80)
        ].sort_values("__ts__", kind="stable")
        if local_train.empty or local_valid.empty or not candidates:
            continue
        train_state = overlay._timestamp_training_frame(
            local_train, candidates, target_column=overlay.TARGET, event_column=overlay.EVENT
        )
        valid_state = overlay._timestamp_training_frame(
            local_valid, candidates, target_column=overlay.TARGET, event_column=overlay.EVENT
        )
        for state, source in ((train_state, local_train), (valid_state, local_valid)):
            rank_by_timestamp = source.groupby("__ts__", observed=True)[
                "parent_rank_v9"
            ].median()
            state["side_name"] = side
            state["archetype_policy_key"] = archetype
            state["parent_rank_v9"] = state["__ts__"].map(rank_by_timestamp).astype(
                np.float32
            )
        final_selected: list[str] = []
        for fold_index, fold_start in enumerate(FOLD_STARTS):
            fold_end = (
                FOLD_STARTS[fold_index + 1]
                if fold_index + 1 < len(FOLD_STARTS)
                else FINAL_TRAIN_END
            )
            fit = train_state.loc[
                train_state["__ts__"].lt(fold_start - pd.Timedelta(days=2))
            ]
            score = train_state.loc[
                train_state["__ts__"].ge(fold_start)
                & train_state["__ts__"].lt(fold_end)
            ]
            if len(fit) < 500 or int(fit[overlay.TARGET].sum()) < 10 or score.empty:
                continue
            selected, _ = overlay._screen_features(
                fit, candidates, config, side=side, archetype=archetype
            )
            if not selected:
                continue
            feature_schema_rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "stage": "train_oof",
                    "fold_start": fold_start,
                    "feature_schema_hash": _feature_schema_hash(selected),
                    "feature_order_json": json.dumps(selected, separators=(",", ":")),
                    "transform_schema": "train_iqr_scale_clip5_v1",
                }
            )
            predictions, gains, neighbors = _ensemble_and_neighbors(
                fit,
                score,
                selected,
                seeds=args.ensemble_models,
                seed=args.seed + 1000 * group_index + 100 * fold_index,
                neighbor_count=args.neighbor_count,
                neighbor_shrinkage=args.neighbor_shrinkage,
            )
            predictions["stage"] = "train_oof"
            predictions["fold_start"] = fold_start
            oof_parts.append(predictions)
            matched_neighbor_rows.extend(neighbors)
            for row in gains:
                importance_rows.append(
                    {"side_name": side, "archetype_policy_key": archetype, "fold_start": fold_start, **row}
                )
        selected, _ = overlay._screen_features(
            train_state, candidates, config, side=side, archetype=archetype
        )
        final_selected = selected
        if selected:
            feature_schema_rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "stage": "eval_oos",
                    "fold_start": FINAL_TRAIN_END,
                    "feature_schema_hash": _feature_schema_hash(selected),
                    "feature_order_json": json.dumps(selected, separators=(",", ":")),
                    "transform_schema": "train_iqr_scale_clip5_v1",
                }
            )
            neighbor_index = train_state.loc[:, ["__ts__"]].copy()
            neighbor_index["side_name"] = side
            neighbor_index["archetype_policy_key"] = archetype
            neighbor_index_parts.append(neighbor_index)
            predictions, gains, neighbors = _ensemble_and_neighbors(
                train_state,
                valid_state,
                selected,
                seeds=args.ensemble_models,
                seed=args.seed + 10_000 + group_index,
                neighbor_count=args.neighbor_count,
                neighbor_shrinkage=args.neighbor_shrinkage,
            )
            predictions["stage"] = "eval_oos"
            predictions["fold_start"] = FINAL_TRAIN_END
            eval_parts.append(predictions)
            matched_neighbor_rows.extend(neighbors)
            for row in gains:
                importance_rows.append(
                    {"side_name": side, "archetype_policy_key": archetype, "fold_start": "final", **row}
                )

        episode_features = final_selected[: min(16, len(final_selected))]
        if not episode_features:
            continue
        train_episode = _episode_frame(train_state, episode_features, episode_days)
        valid_episode = _episode_frame(valid_state, episode_features, episode_days)
        episode_columns = [
            name for name in train_episode if name not in {"day", "adverse", "mean_ev"}
        ]
        for fold_index, fold_start in enumerate(FOLD_STARTS):
            fold_end = (
                FOLD_STARTS[fold_index + 1]
                if fold_index + 1 < len(FOLD_STARTS)
                else FINAL_TRAIN_END
            )
            fit = train_episode.loc[train_episode["day"].lt(fold_start - pd.Timedelta(days=2))].dropna(subset=episode_columns)
            score = train_episode.loc[
                train_episode["day"].ge(fold_start) & train_episode["day"].lt(fold_end)
            ].dropna(subset=episode_columns)
            if len(fit) < 45 or score.empty:
                continue
            result, state = _fit_episode_gmm(
                fit, score, episode_columns, args.seed + 20_000 + 100 * group_index + fold_index
            )
            result["side_name"] = side
            result["archetype_policy_key"] = archetype
            result["stage"] = "train_oof"
            result["fold_start"] = fold_start
            episode_parts.append(result)
            episode_manifests.append(
                {"side_name": side, "archetype_policy_key": archetype, "stage": "train_oof", "fold_start": fold_start, **state}
            )
        fit = train_episode.dropna(subset=episode_columns)
        score = valid_episode.dropna(subset=episode_columns)
        if len(fit) >= 45 and not score.empty:
            result, state = _fit_episode_gmm(
                fit, score, episode_columns, args.seed + 30_000 + group_index
            )
            result["side_name"] = side
            result["archetype_policy_key"] = archetype
            result["stage"] = "eval_oos"
            result["fold_start"] = FINAL_TRAIN_END
            episode_parts.append(result)
            episode_manifests.append(
                {"side_name": side, "archetype_policy_key": archetype, "stage": "eval_oos", "fold_start": FINAL_TRAIN_END, **state}
            )

    if not oof_parts or not eval_parts:
        raise RuntimeError("No chronological distinguishability predictions were generated")
    oof = pd.concat(oof_parts, ignore_index=True, copy=False)
    evaluated = pd.concat(eval_parts, ignore_index=True, copy=False)
    reference = oof.groupby(["side_name", "archetype_policy_key"], observed=True)
    enriched_parts: list[pd.DataFrame] = []
    for key, local in pd.concat([oof, evaluated], ignore_index=True, copy=False).groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        ref = reference.get_group(key)
        local = local.copy()
        local["ensemble_risk_std_percentile"] = _percentile(
            local["ensemble_risk_std"].to_numpy(np.float32),
            ref["ensemble_risk_std"].to_numpy(np.float32),
        )
        local["neighbor_distance_percentile"] = _percentile(
            local["neighbor_distance_mean"].to_numpy(np.float32),
            ref["neighbor_distance_mean"].to_numpy(np.float32),
        )
        local["difficulty_label"] = _difficulty_label(
            local["ensemble_risk_std_percentile"].to_numpy(np.float32),
            local["neighbor_outcome_entropy"].to_numpy(np.float32),
            local["neighbor_distance_percentile"].to_numpy(np.float32),
        )
        enriched_parts.append(local)
    diagnostics = pd.concat(enriched_parts, ignore_index=True, copy=False)
    diagnostics.to_parquet(
        args.output / "state_distinguishability_predictions.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame(importance_rows).to_csv(args.output / "ensemble_feature_gain.csv", index=False)
    pd.DataFrame(feature_schema_rows).to_csv(
        args.output / "feature_schemas.csv", index=False
    )
    pd.concat(neighbor_index_parts, ignore_index=True, copy=False).to_parquet(
        args.output / "neighbor_training_index.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(matched_neighbor_rows).to_parquet(
        args.output / "matched_adverse_event_neighbors.parquet",
        index=False,
        compression="zstd",
    )
    episodes = pd.concat(episode_parts, ignore_index=True, copy=False)
    episodes.to_parquet(args.output / "episode_cluster_assignments.parquet", index=False, compression="zstd")
    pd.DataFrame(episode_manifests).to_json(
        args.output / "episode_cluster_manifest.json", orient="records", indent=2
    )

    summary = (
        diagnostics.groupby(
            ["stage", "side_name", "archetype_policy_key", "difficulty_label"],
            observed=True,
            dropna=False,
        )
        .agg(
            rows=(overlay.TARGET, "size"),
            adverse_rate=(overlay.TARGET, "mean"),
            mean_ev=("ev_after_1pct", "mean"),
            ensemble_risk_mean=("ensemble_risk_mean", "mean"),
            ensemble_risk_std=("ensemble_risk_std", "mean"),
            neighbor_adverse_rate=("neighbor_adverse_rate", "mean"),
            neighbor_entropy=("neighbor_outcome_entropy", "mean"),
            neighbor_distance_pct=("neighbor_distance_percentile", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "difficulty_summary.csv", index=False)
    event_summary = (
        diagnostics.loc[diagnostics[overlay.EVENT].gt(0)]
        .groupby(["stage", "day", "side_name", "archetype_policy_key"], observed=True)
        .agg(
            mean_ev=("ev_after_1pct", "mean"),
            ensemble_risk_mean=("ensemble_risk_mean", "mean"),
            ensemble_risk_std=("ensemble_risk_std", "mean"),
            neighbor_adverse_rate=("neighbor_adverse_rate", "mean"),
            neighbor_entropy=("neighbor_outcome_entropy", "mean"),
            neighbor_distance_pct=("neighbor_distance_percentile", "mean"),
            dominant_difficulty=("difficulty_label", lambda values: values.value_counts().index[0]),
        )
        .reset_index()
    )
    event_summary.to_csv(args.output / "adverse_event_distinguishability.csv", index=False)
    manifest = {
        "schema": "residual_distinguishability_diagnostics_v1",
        "coverage": coverage,
        "ensemble_models": args.ensemble_models,
        "neighbor_count": args.neighbor_count,
        "neighbor_shrinkage": args.neighbor_shrinkage,
        "groups": [f"{side}::{archetype}" for side, archetype in TARGET_GROUPS],
        "train_period": [args.train_start, args.train_end],
        "eval_period": [args.train_end, args.eval_end],
        "rows": len(diagnostics),
        "episode_rows": len(episodes),
        "matched_neighbor_rows": len(matched_neighbor_rows),
        "feature_schema_contract": "feature_schemas.csv records exact ordered local inputs and robust transform semantics",
        "neighbor_training_index_contract": "neighbor_training_index.parquet records final train-only timestamp state rows",
        "activation": "diagnostic_only",
        "leakage_contract": (
            "Every OOF ensemble, robust transform, neighbor index, feature screen, PCA, "
            "GMM, and cluster outcome prior is fitted only on rows before its score fold. "
            "Two days are purged. April-June receives frozen assignments from training "
            "through March. Difficulty labels are diagnostics and are not inference inputs."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion-ledger",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/"
            "frozen_champion_single_source_ledger.parquet"
        ),
    )
    parser.add_argument(
        "--train-oof-predictions-dir",
        type=Path,
        default=Path(
            "data_perp/reports/s59_h5_2025start_monthly_v4_base_configfull_"
            "mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_"
            "payload300k_20260706/train_meta_regime_handoff_singlehead_base_soft_"
            "lgbmpipeline_auto_hpo150_oos15_top30_hpo45k_20260706_v5/"
            "best_full_oos_fixedfs_streamed_v1/prediction_shards"
        ),
    )
    parser.add_argument(
        "--train-oof-rank-cache",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_"
            "market_20260712_v3/meta_oof_global_rank_202504_202603.parquet"
        ),
    )
    parser.add_argument(
        "--state-artifact",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_archetype_true_base_oof_compactlocal_"
            "market_20260713_v4_predicted_damage/oos_residual_event_states.parquet"
        ),
    )
    parser.add_argument(
        "--parent-eval-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_baseline_"
            "globaloverlay_sparse_shock_composite/oos_predictions_historical_rank.parquet"
        ),
    )
    parser.add_argument(
        "--v9-predictions",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/oos_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--v9-manifest",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/manifest.json"
        ),
    )
    parser.add_argument(
        "--v9-selected-features",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_extreme_local_champion_overlay_"
            "ooftrain_tieaware_downonly_20260712_v9/selected_local_features_strict.csv"
        ),
    )
    parser.add_argument(
        "--negative-residual-features",
        type=Path,
        default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"),
    )
    parser.add_argument(
        "--temporal-state-features",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_"
            "support_fallback/oos_temporal_state_context_apr2025_july2026.parquet"
        ),
    )
    parser.add_argument(
        "--event-calendar",
        type=Path,
        default=Path(
            "data_perp/reports/residual_episode_recognition_calendar_20260712_v1/"
            "calendar_recognized_vs_ignored.csv"
        ),
    )
    parser.add_argument(
        "--extension-calendar",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_"
            "support_fallback/residual_event_calendar.csv"
        ),
    )
    parser.add_argument(
        "--model-report",
        type=Path,
        default=Path(
            "data_perp/reports/meta_residual_event_balanced_error_overlay_"
            "20260713_v11_predicted_damage/model_report.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_perp/reports/residual_distinguishability_20260713_v1"),
    )
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--max-features", type=int, default=32)
    parser.add_argument("--ensemble-models", type=int, default=7)
    parser.add_argument("--neighbor-count", type=int, default=50)
    parser.add_argument("--neighbor-shrinkage", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    print(json.dumps(_json_safe(run(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
