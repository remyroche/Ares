#!/usr/bin/env python3
"""Assess market transition features against residual-event calendar cells OOS.

The target is defined per side x archetype.  Market features remain observable
and market-wide; realized residual calendar labels are used only during fitting
and evaluation.  Every reported prediction comes from a later chronological
block than the rows used to fit its model and operating threshold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
)
from extreme_price_movements.market_regime_change_contract import (
    MARKET_REGIME_CHANGE_FEATURE_KEYS,
)
from extreme_price_movements.residual_event_archetypes import (
    RESIDUAL_EVENT_PREFIX,
    RESIDUAL_EVENT_TEMPORAL_SUFFIXES,
)


KEYS = ["day", "side_name", "archetype_policy_key"]


@dataclass(frozen=True)
class AssessmentConfig:
    train_start: str = "2025-01-01"
    eval_start: str = "2025-04-01"
    eval_end: str = "2026-07-13"
    risk_fraction: float = 0.10
    max_features: int = 20
    min_train_days: int = 75
    min_positive_days: int = 5
    min_eval_rows: int = 100
    seed: int = 20260713


def _binned_mi(values: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 100 or np.unique(target[finite]).size < 2:
        return 0.0
    x = values[finite].astype(np.float64, copy=False)
    y = target[finite].astype(np.int8, copy=False)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    code = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, bins - 1)
    joint = np.bincount(code * 2 + y, minlength=bins * 2).reshape(bins, 2)
    joint = joint.astype(np.float64) / max(float(joint.sum()), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    valid = (joint > 0.0) & (expected > 0.0)
    return float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))


def _screen_features(
    train: pd.DataFrame,
    target: np.ndarray,
    candidates: list[str],
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for feature in candidates:
        values = pd.to_numeric(train[feature], errors="coerce").to_numpy(np.float32)
        rows.append(
            {
                "feature": feature,
                "binned_mi": _binned_mi(values, target),
                "finite_rate": float(np.isfinite(values).mean()),
                "is_transition": feature in MARKET_REGIME_CHANGE_FEATURE_KEYS,
            }
        )
    relevance = pd.DataFrame(rows).sort_values(
        ["binned_mi", "finite_rate", "feature"],
        ascending=[False, False, True],
        kind="stable",
    )
    selected = relevance.loc[
        relevance["finite_rate"].ge(0.50) & relevance["binned_mi"].gt(0.0),
        "feature",
    ].head(max_features).tolist()
    relevance["selected"] = relevance["feature"].isin(selected)
    return selected, relevance


def _matrix(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[features].to_numpy(dtype=np.float32, copy=True)
    x_valid = valid[features].to_numpy(dtype=np.float32, copy=True)
    medians = np.nanmedian(np.where(np.isfinite(x_train), x_train, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    for matrix in (x_train, x_valid):
        bad = ~np.isfinite(matrix)
        if bad.any():
            matrix[bad] = np.take(medians, np.nonzero(bad)[1])
    return x_train, x_valid


def _fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_valid = _matrix(train, valid, features)
    y = train["event_label"].to_numpy(np.int8)
    positive = max(int(y.sum()), 1)
    negative = max(int(len(y) - positive), 1)
    weights = np.where(y > 0, 0.5 / positive, 0.5 / negative).astype(np.float32)
    weights *= np.float32(len(weights))
    dataset = lgb.Dataset(
        x_train,
        label=y.astype(np.float32, copy=False),
        weight=weights,
        feature_name=features,
        free_raw_data=True,
    )
    model = lgb.train(
        {
            "objective": "binary",
            "metric": "None",
            "learning_rate": 0.035,
            "max_depth": 3,
            "num_leaves": 7,
            "min_data_in_leaf": 10,
            "min_gain_to_split": 0.01,
            "lambda_l1": 1.0,
            "lambda_l2": 4.0,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "feature_fraction": 0.80,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "num_threads": -1,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=180,
    )
    return (
        np.asarray(model.predict(x_train), dtype=np.float32),
        np.asarray(model.predict(x_valid), dtype=np.float32),
    )


def _episode_ids(days: pd.Series, target: np.ndarray) -> np.ndarray:
    order = np.argsort(days.to_numpy())
    result = np.full(len(days), -1, dtype=np.int32)
    episode = -1
    previous: pd.Timestamp | None = None
    for position in order:
        if not bool(target[position]):
            previous = None
            continue
        day = pd.Timestamp(days.iloc[position])
        if previous is None or day - previous > pd.Timedelta(days=1):
            episode += 1
        result[position] = episode
        previous = day
    return result


def _daily_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["event_label"].to_numpy(np.int8)
    selected = frame["recognized"].to_numpy(bool)
    prevalence = float(target.mean()) if len(target) else np.nan
    precision = float(target[selected].mean()) if selected.any() else np.nan
    negative = target == 0
    fpr = float(selected[negative].mean()) if negative.any() else np.nan
    episode = _episode_ids(frame["day"], target)
    episodes = np.unique(episode[episode >= 0])
    episode_hits = [bool(selected[episode == value].any()) for value in episodes]
    long_episodes = [
        value for value in episodes if int(np.sum(episode == value)) >= 2
    ]
    long_hits = [bool(selected[episode == value].any()) for value in long_episodes]
    ap = (
        float(average_precision_score(target, frame["risk_score"]))
        if np.unique(target).size > 1
        else np.nan
    )
    return {
        "days": int(len(frame)),
        "adverse_days": int(target.sum()),
        "adverse_prevalence": prevalence,
        "average_precision": ap,
        "precision": precision,
        "lift": precision / max(prevalence, 1e-9) if np.isfinite(precision) else np.nan,
        "false_positive_rate": fpr,
        "episode_count": int(len(episodes)),
        "episode_recall": float(np.mean(episode_hits)) if episode_hits else np.nan,
        "persistent_episode_count": int(len(long_episodes)),
        "persistent_episode_recall": float(np.mean(long_hits)) if long_hits else np.nan,
    }


def _quarter_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    boundaries = list(pd.date_range(start=start, end=end, freq="QS", tz="UTC"))
    if not boundaries or boundaries[0] != start:
        boundaries.insert(0, start)
    if boundaries[-1] < end:
        boundaries.append(end)
    return [(left, right) for left, right in zip(boundaries[:-1], boundaries[1:])]


def run(args: argparse.Namespace) -> dict[str, object]:
    config = AssessmentConfig(
        train_start=args.train_start,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        risk_fraction=float(args.risk_fraction),
        max_features=int(args.max_features),
        seed=int(args.seed),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.daily_calendar)
    daily["day"] = pd.to_datetime(daily["day"], utc=True).dt.floor("D")
    daily = daily.loc[daily["scope"].eq("side_archetype"), KEYS + [
        "selected_rows", "mean_ev_after_1pct", "clean_exec_rate", "signed_surprise"
    ]]
    events = pd.read_csv(args.event_calendar)
    events["day"] = pd.to_datetime(events["day"], utc=True).dt.floor("D")
    if args.extension_calendar is not None and args.extension_calendar.exists():
        extension = pd.read_csv(args.extension_calendar)
        extension["day"] = pd.to_datetime(extension["day"], utc=True).dt.floor("D")
        extension_start = extension["day"].min()
        extension_daily = extension.rename(
            columns={
                "rows": "selected_rows",
                "clean_exec_precision": "clean_exec_rate",
                "mean_timestamp_neutral_surprise": "signed_surprise",
            }
        ).loc[:, KEYS + [
            "selected_rows", "mean_ev_after_1pct", "clean_exec_rate", "signed_surprise"
        ]]
        daily = pd.concat(
            [daily.loc[daily["day"].lt(extension_start)], extension_daily],
            ignore_index=True,
            copy=False,
        )
        extension_events = extension.loc[
            pd.to_numeric(extension["adverse_event_rows"], errors="coerce").fillna(0).gt(0),
            KEYS,
        ]
        events = pd.concat(
            [events.loc[events["day"].lt(extension_start)], extension_events],
            ignore_index=True,
            copy=False,
        )
    temporal_features: list[str] = []
    if args.temporal_state_features is not None and args.temporal_state_features.exists():
        state = pd.read_parquet(args.temporal_state_features)
        state["day"] = pd.to_datetime(state["__ts__"], utc=True).dt.floor("D")
        temporal_features = [
            f"{RESIDUAL_EVENT_PREFIX}{suffix}"
            for suffix in RESIDUAL_EVENT_TEMPORAL_SUFFIXES
            if f"{RESIDUAL_EVENT_PREFIX}{suffix}" in state.columns
        ]
        aggregations = {name: "max" for name in temporal_features}
        since_name = f"{RESIDUAL_EVENT_PREFIX}hours_since_ood_spike_96h_norm"
        if since_name in aggregations:
            aggregations[since_name] = "min"
        daily_state = (
            state.groupby(KEYS, observed=True, sort=True)
            .agg(aggregations)
            .reset_index()
        )
        daily = daily.merge(daily_state, on=KEYS, how="left", validate="one_to_one")
    event_keys = events.loc[:, KEYS].drop_duplicates()
    daily = daily.merge(event_keys.assign(event_label=np.int8(1)), on=KEYS, how="left")
    daily["event_label"] = daily["event_label"].fillna(0).astype(np.int8)
    market = pd.read_parquet(args.market_features, columns=NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    market.index = pd.to_datetime(market.index, utc=True, errors="coerce")
    market = market.loc[~market.index.duplicated(keep="last")].sort_index()
    market.index.name = "market_timestamp"
    market = market.reset_index()
    market["day"] = market["market_timestamp"].dt.floor("D")
    # Each OOS hour is scored causally. Day-level recognition uses the maximum
    # observed risk score only for retrospective event-recall assessment.
    frame = daily.merge(market, on="day", how="inner", validate="many_to_many")
    start = pd.Timestamp(config.train_start, tz="UTC")
    eval_start = pd.Timestamp(config.eval_start, tz="UTC")
    end = pd.Timestamp(config.eval_end, tz="UTC")
    frame = frame.loc[frame["day"].ge(start) & frame["day"].lt(end)].copy()
    static = [
        feature for feature in NEGATIVE_RESIDUAL_META_FEATURE_KEYS
        if feature not in MARKET_REGIME_CHANGE_FEATURE_KEYS and feature in frame.columns
    ]
    augmented = static + [
        feature for feature in MARKET_REGIME_CHANGE_FEATURE_KEYS if feature in frame.columns
    ]
    augmented_temporal = augmented + [
        feature for feature in temporal_features if feature in frame.columns
    ]
    predictions: list[pd.DataFrame] = []
    relevance_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    groups = frame.groupby(["side_name", "archetype_policy_key"], observed=True, sort=True)
    for group_index, ((side, archetype), local) in enumerate(groups):
        local = local.sort_values("day", kind="stable")
        for fold_index, (fold_start, fold_end) in enumerate(_quarter_boundaries(eval_start, end)):
            train = local.loc[local["day"].lt(fold_start)]
            valid = local.loc[local["day"].ge(fold_start) & local["day"].lt(fold_end)]
            train_days = int(train["day"].nunique())
            positive_days = int(train.loc[train["event_label"].gt(0), "day"].nunique())
            if (
                train_days < config.min_train_days
                or positive_days < config.min_positive_days
                or len(valid) < 5
                or np.unique(train["event_label"]).size < 2
            ):
                continue
            arm_predictions: dict[str, pd.DataFrame] = {}
            fold_row_positions: dict[str, int] = {}
            for arm_index, (arm, candidates) in enumerate(
                (
                    ("static", static),
                    ("static_plus_transitions", augmented),
                    (
                        "static_plus_transitions_temporal_state",
                        augmented_temporal,
                    ),
                )
            ):
                selected_features, relevance = _screen_features(
                    train,
                    train["event_label"].to_numpy(np.int8),
                    candidates,
                    config.max_features,
                )
                if not selected_features:
                    continue
                train_score, valid_score = _fit_predict(
                    train,
                    valid,
                    selected_features,
                    config.seed + 100 * group_index + 10 * fold_index + arm_index,
                )
                train_daily_score = pd.Series(train_score).groupby(
                    train["day"].reset_index(drop=True), sort=True
                ).max()
                valid_daily_score = pd.Series(valid_score).groupby(
                    valid["day"].reset_index(drop=True), sort=True
                ).max()
                train_threshold = float(
                    np.quantile(train_daily_score, 1.0 - config.risk_fraction)
                )
                # Ranking discrimination is assessed at a fixed OOS score budget.
                # This uses no outcomes, but is intentionally not an inference
                # threshold; frozen-threshold behavior is retained separately.
                threshold = float(
                    np.quantile(valid_daily_score, 1.0 - config.risk_fraction)
                )
                risk_budget = max(
                    1, int(np.ceil(len(valid_daily_score) * config.risk_fraction))
                )
                risk_order = np.argsort(
                    -valid_daily_score.to_numpy(np.float32), kind="stable"
                )
                recognized_days = np.zeros(len(valid_daily_score), dtype=bool)
                if float(np.nanstd(valid_daily_score)) > 1e-8:
                    recognized_days[risk_order[:risk_budget]] = True
                selected_day_index = valid_daily_score.index[recognized_days]
                pred = (
                    valid.loc[:, KEYS + [
                        "selected_rows", "mean_ev_after_1pct", "clean_exec_rate",
                        "signed_surprise", "event_label"
                    ]]
                    .drop_duplicates(KEYS, keep="last")
                    .sort_values("day", kind="stable")
                    .reset_index(drop=True)
                )
                pred["arm"] = arm
                pred["fold_start"] = fold_start
                pred["fold_end"] = fold_end
                pred["risk_score"] = pred["day"].map(valid_daily_score).astype(np.float32)
                pred["risk_threshold"] = threshold
                pred["recognized"] = pred["day"].isin(selected_day_index)
                pred["train_risk_threshold"] = train_threshold
                pred["frozen_threshold_recognized"] = pred["risk_score"].ge(
                    train_threshold
                )
                pred["selected_features"] = "|".join(selected_features)
                predictions.append(pred)
                arm_predictions[arm] = pred
                relevance.insert(0, "fold_start", fold_start)
                relevance.insert(0, "archetype_policy_key", archetype)
                relevance.insert(0, "side_name", side)
                relevance.insert(0, "arm", arm)
                relevance_rows.append(relevance)
                fold_rows.append(
                    {
                        "arm": arm,
                        "fold_start": fold_start,
                        "fold_end": fold_end,
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "train_days": train_days,
                        "train_adverse_days": positive_days,
                        "features": len(selected_features),
                        **_daily_metrics(pred),
                    }
                )
                fold_row_positions[arm] = len(fold_rows) - 1
            if "static" in arm_predictions:
                static_precision = _daily_metrics(arm_predictions["static"])["precision"]
                for arm, arm_prediction in arm_predictions.items():
                    if arm == "static":
                        continue
                    arm_precision = _daily_metrics(arm_prediction)["precision"]
                    fold_rows[fold_row_positions[arm]]["precision_delta_vs_static"] = (
                        arm_precision - static_precision
                        if np.isfinite(static_precision) and np.isfinite(arm_precision)
                        else np.nan
                    )
    prediction = pd.concat(predictions, ignore_index=True, copy=False)
    folds = pd.DataFrame(fold_rows)
    relevance = pd.concat(relevance_rows, ignore_index=True, copy=False)
    prediction.to_parquet(args.output / "oos_calendar_predictions.parquet", index=False, compression="zstd")
    folds.to_csv(args.output / "fold_metrics.csv", index=False)
    relevance.to_csv(args.output / "feature_relevance.csv", index=False)
    summary_rows = []
    for (arm, side, archetype), part in prediction.groupby(
        ["arm", "side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        local_folds = folds.loc[
            folds["arm"].eq(arm)
            & folds["side_name"].eq(side)
            & folds["archetype_policy_key"].eq(archetype)
        ]
        metric = _daily_metrics(part)
        summary_rows.append(
            {
                "scope": "side_archetype",
                "arm": arm,
                "side_name": side,
                "archetype_policy_key": archetype,
                **metric,
                "folds": int(len(local_folds)),
                "positive_precision_delta_folds": int(
                    pd.to_numeric(
                        local_folds.get("precision_delta_vs_static"), errors="coerce"
                    ).gt(0).sum()
                ),
                "lift_q25": float(pd.to_numeric(local_folds["lift"], errors="coerce").quantile(0.25)),
                "fpr_q75": float(pd.to_numeric(local_folds["false_positive_rate"], errors="coerce").quantile(0.75)),
            }
        )
    for arm, part in prediction.groupby("arm", observed=True, sort=True):
        summary_rows.append(
            {
                "scope": "global",
                "arm": arm,
                "side_name": "__all__",
                "archetype_policy_key": "__all__",
                **_daily_metrics(part),
                "folds": int(part["fold_start"].nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["promotion_candidate"] = (
        summary["scope"].eq("side_archetype")
        & summary["arm"].ne("static")
        & summary["lift_q25"].ge(1.5)
        & summary["fpr_q75"].le(0.15)
        & summary["positive_precision_delta_folds"].ge(3)
        & summary["episode_count"].ge(3)
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    recognized = prediction.loc[
        prediction["event_label"].gt(0),
        KEYS + ["arm", "fold_start", "risk_score", "risk_threshold", "recognized", "selected_features"],
    ].sort_values(KEYS + ["arm"], kind="stable")
    recognized.to_csv(args.output / "event_calendar_recognition.csv", index=False)
    manifest = {
        "schema": "market_regime_change_residual_calendar_assessment_v1",
        "config": asdict(config),
        "rows": int(len(frame)),
        "oos_prediction_rows": int(len(prediction)),
        "event_cells": int(
            frame.loc[frame["event_label"].gt(0), KEYS].drop_duplicates().shape[0]
        ),
        "static_features": len(static),
        "transition_features": len(augmented) - len(static),
        "temporal_state_features": len(temporal_features),
        "promotion_candidates": summary.loc[
            summary["promotion_candidate"], ["side_name", "archetype_policy_key"]
        ].to_dict("records"),
        "leakage_contract": (
            "Residual events are labels only. Feature screening and model fitting "
            "use rows strictly before each chronological OOS quarter. OOS "
            "top-decile recognition is a rank diagnostic; the separate frozen-threshold "
            "column uses only the train-score distribution."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-calendar",
        type=Path,
        default=Path(
            "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
            "champion_frozen_single_source_202501_20260710/daily_surprise_calendar_all_cells.csv"
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
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_v1/"
            "residual_event_calendar.csv"
        ),
    )
    parser.add_argument(
        "--market-features",
        type=Path,
        default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"),
    )
    parser.add_argument(
        "--temporal-state-features",
        type=Path,
        default=Path(
            "data_perp/reports/residual_event_target_transitions_july_oos_20260713_v2_support_fallback/"
            "oos_temporal_state_context_apr2025_july2026.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_perp/reports/market_regime_change_calendar_assessment_20260713_v1"),
    )
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--eval-start", default="2025-04-01")
    parser.add_argument("--eval-end", default="2026-07-13")
    parser.add_argument("--risk-fraction", type=float, default=0.10)
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
