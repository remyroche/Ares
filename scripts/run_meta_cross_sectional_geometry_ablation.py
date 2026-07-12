#!/usr/bin/env python3
"""Walk-forward market-state versus cross-sectional-geometry meta ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.meta_cross_sectional_geometry import (
    DEFAULT_RELATIVE_FEATURES,
    geometry_feature_names,
    materialize_cross_sectional_geometry,
)
from scripts.score_compare_meta_residual_july_oos import _append_store_features

KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
OUTCOMES = [
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
]
MARKET_FEATURES = [
    "mkt_median_oi_chg_1h_rz",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_1h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_pct_oi_drawdown_24h_lt_minus5pct",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_median_oi_recovery_fraction_24h",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_pct_price_down_oi_down_1h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_up_oi_down_4h",
    "mkt_median_long_flush_intensity_4h",
    "mkt_median_short_cover_intensity_1h",
    "market_breadth_recovery_from_24h_min",
    "market_breadth_drawdown_from_6h_max",
    "market_pct_recovering_from_24h_low",
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_24h",
    "market_pc1_variance_share_chg_4h",
    "market_downside_pairwise_corr_24h",
    "market_downside_corr_minus_unconditional_corr_24h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
]
ARMS = (
    "market_state",
    "cross_sectional_geometry",
    "joint_state_geometry",
    "joint_state_geometry_breakout_weighted",
    "joint_state_geometry_breakout_day_balanced",
)
ALPHAS = np.asarray((0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50), dtype=np.float32)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _columns(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema.names]


def _batch_rank(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return (
        pd.to_numeric(frame[score_col], errors="coerce")
        .groupby(frame["__ts__"], sort=False)
        .rank(method="average", pct=True)
        .astype(np.float32)
    )


def _prepare_historical(path: Path, cache_path: Path, force: bool) -> pd.DataFrame:
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    available = set(_columns(path))
    requested = list(
        dict.fromkeys(
            KEYS
            + ["score_meta_base_soft_label", "oos_fold"]
            + OUTCOMES
            + MARKET_FEATURES
            + list(DEFAULT_RELATIVE_FEATURES)
        )
    )
    frame = pd.read_parquet(
        path, columns=[name for name in requested if name in available]
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame.loc[frame["score_meta_base_soft_label"].notna()].copy()
    frame["model_score"] = pd.to_numeric(
        frame["score_meta_base_soft_label"], errors="coerce"
    ).astype(np.float32)
    frame["base_batch_rank"] = _batch_rank(frame, "model_score")
    geometry = materialize_cross_sectional_geometry(frame, score_col="model_score")
    frame[geometry.columns] = geometry.to_numpy(dtype=np.float32, copy=False)
    frame["calendar_month"] = frame["__ts__"].dt.strftime("%Y-%m")
    keep = list(
        dict.fromkeys(
            KEYS
            + ["model_score", "base_batch_rank", "calendar_month"]
            + OUTCOMES
            + MARKET_FEATURES
            + list(DEFAULT_RELATIVE_FEATURES)
            + geometry_feature_names()
        )
    )
    frame = frame.reindex(columns=[name for name in keep if name in frame.columns])
    for name in frame.columns:
        if name not in KEYS + ["calendar_month"]:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False, compression="zstd")
    return frame


def _prepare_july(
    aligned_path: Path,
    completed_path: Path,
    feature_root: Path,
    cache_path: Path,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path), {"cache_hit": 1.0}
    completed = pd.read_parquet(completed_path)
    aligned = pd.read_parquet(aligned_path)
    completed["__ts__"] = pd.to_datetime(completed["__ts__"], utc=True, errors="coerce")
    aligned["__ts__"] = pd.to_datetime(aligned["__ts__"], utc=True, errors="coerce")
    # Materialized research predictions are authoritative on overlap.
    frame = pd.concat([completed, aligned], ignore_index=True, sort=False, copy=False)
    frame = frame.sort_values(KEYS, kind="stable").drop_duplicates(KEYS, keep="last")
    frame = frame.loc[
        frame["__ts__"].ge(pd.Timestamp("2026-07-01", tz="UTC"))
        & frame["__ts__"].lt(pd.Timestamp("2026-07-11", tz="UTC"))
    ].reset_index(drop=True)
    frame["model_score"] = pd.to_numeric(
        frame["score_shock_adjusted"], errors="coerce"
    ).astype(np.float32)
    requested = list(dict.fromkeys(MARKET_FEATURES + list(DEFAULT_RELATIVE_FEATURES)))
    frame, coverage = _append_store_features(frame, feature_root, requested)
    frame["base_batch_rank"] = _batch_rank(frame, "model_score")
    geometry = materialize_cross_sectional_geometry(frame, score_col="model_score")
    frame[geometry.columns] = geometry.to_numpy(dtype=np.float32, copy=False)
    frame["calendar_month"] = "2026-07"
    keep = list(
        dict.fromkeys(
            KEYS
            + ["model_score", "base_batch_rank", "calendar_month"]
            + OUTCOMES
            + MARKET_FEATURES
            + list(DEFAULT_RELATIVE_FEATURES)
            + geometry_feature_names()
        )
    )
    frame = frame.reindex(columns=[name for name in keep if name in frame.columns])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False, compression="zstd")
    return frame, coverage


def _arm_features(arm: str, available: Sequence[str]) -> list[str]:
    common = ["model_score", "base_batch_rank"]
    geometry = geometry_feature_names()
    if arm == "market_state":
        requested = common + MARKET_FEATURES
    elif arm == "cross_sectional_geometry":
        requested = common + list(DEFAULT_RELATIVE_FEATURES) + geometry
    else:
        requested = (
            common + MARKET_FEATURES + list(DEFAULT_RELATIVE_FEATURES) + geometry
        )
    return [name for name in dict.fromkeys(requested) if name in available]


@dataclass
class _LocalModel:
    booster: Any
    features: list[str]
    median: np.ndarray
    low: np.ndarray
    high: np.ndarray
    prediction_mean: float
    prediction_std: float
    support_rows: int

    def predict_z(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.reindex(columns=self.features).to_numpy(
            dtype=np.float32, copy=True
        )
        values = np.where(np.isfinite(values), values, self.median)
        np.clip(values, self.low, self.high, out=values)
        prediction = np.asarray(self.booster.predict(values), dtype=np.float32)
        return np.clip(
            (prediction - np.float32(self.prediction_mean))
            / np.float32(max(self.prediction_std, 1e-4)),
            -4.0,
            4.0,
        ).astype(np.float32)


@dataclass
class _OverlayState:
    side_models: dict[str, _LocalModel]
    local_models: dict[tuple[str, str], _LocalModel]
    features: list[str]
    train_end: str

    def predict_z(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        output = np.zeros(len(frame), dtype=np.float32)
        local_used = np.zeros(len(frame), dtype=np.int8)
        side = frame["side_name"].astype(str).str.lower()
        archetype = frame["archetype_policy_key"].astype(str)
        groups = pd.DataFrame({"side": side, "archetype": archetype}, index=frame.index)
        for (side_key, archetype_key), index in groups.groupby(
            ["side", "archetype"], observed=True, sort=False
        ).groups.items():
            model = self.local_models.get((str(side_key), str(archetype_key)))
            is_local = model is not None
            if model is None:
                model = self.side_models.get(str(side_key))
            if model is None:
                continue
            positions = frame.index.get_indexer(index)
            output[positions] = model.predict_z(frame.loc[index])
            local_used[positions] = np.int8(is_local)
        return output, local_used


def _fit_one(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    seed: int,
    breakout_weighted: bool,
    day_balanced: bool,
) -> _LocalModel | None:
    work = frame.loc[
        frame["base_batch_rank"].ge(0.80)
        & pd.to_numeric(frame["ev_after_1pct"], errors="coerce").notna()
    ].copy()
    if len(work) < 1_200:
        return None
    columns = [name for name in features if name in work.columns]
    values = work.reindex(columns=columns).to_numpy(dtype=np.float32, copy=True)
    median = np.nanmedian(values, axis=0).astype(np.float32)
    median = np.nan_to_num(median, nan=0.0)
    values = np.where(np.isfinite(values), values, median)
    low = np.nanpercentile(values, 0.5, axis=0).astype(np.float32)
    high = np.nanpercentile(values, 99.5, axis=0).astype(np.float32)
    np.clip(values, low, high, out=values)
    ev = pd.to_numeric(work["ev_after_1pct"], errors="coerce").clip(-0.08, 0.08)
    target = ev.to_numpy(dtype=np.float32)
    top10 = work["base_batch_rank"].ge(0.90).to_numpy(dtype=bool)
    positive_band = (
        work["base_batch_rank"].between(0.80, 0.90, inclusive="left") & ev.gt(0.0)
    ).to_numpy(dtype=bool)
    negative_top = top10 & ev.le(0.0).to_numpy(dtype=bool)
    weights = np.ones(len(work), dtype=np.float32)
    weights += 1.5 * negative_top.astype(np.float32)
    weights += 1.0 * positive_band.astype(np.float32)
    if breakout_weighted:
        is_breakout = (
            work["side_name"].astype(str).str.lower().eq("short")
            & work["archetype_policy_key"]
            .astype(str)
            .str.contains("breakout", case=False, na=False)
        ).to_numpy(dtype=bool)
        weights += 2.0 * (is_breakout & negative_top).astype(np.float32)
        weights += 1.0 * (is_breakout & positive_band).astype(np.float32)
    if day_balanced:
        is_breakout = (
            work["side_name"].astype(str).str.lower().eq("short")
            & work["archetype_policy_key"]
            .astype(str)
            .str.contains("breakout", case=False, na=False)
        ).to_numpy(dtype=bool)
        day = pd.to_datetime(work["__ts__"], utc=True, errors="coerce").dt.floor("D")
        day_count = (
            day.groupby(day, sort=False).transform("size").to_numpy(dtype=np.float32)
        )
        reference = (
            float(np.nanmedian(day_count[is_breakout])) if is_breakout.any() else 1.0
        )
        balancing = np.sqrt(reference / np.maximum(day_count, 1.0)).astype(np.float32)
        weights *= np.where(is_breakout, np.clip(balancing, 0.35, 3.0), 1.0)
    weights = np.clip(weights, 0.5, 5.0)
    params = {
        "objective": "huber",
        "alpha": 0.85,
        "learning_rate": 0.035,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 120,
        "min_gain_to_split": 0.002,
        "lambda_l1": 0.10,
        "lambda_l2": 8.0,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "seed": int(seed),
        "num_threads": 2,
        "verbosity": -1,
        "force_col_wise": True,
    }
    dataset = lgb.Dataset(values, label=target, weight=weights, free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=120)
    prediction = np.asarray(booster.predict(values), dtype=np.float32)
    return _LocalModel(
        booster=booster,
        features=columns,
        median=median,
        low=low,
        high=high,
        prediction_mean=float(np.mean(prediction)),
        prediction_std=float(max(np.std(prediction), 1e-4)),
        support_rows=int(len(work)),
    )


def _fit_state(
    train: pd.DataFrame, arm: str, features: Sequence[str], seed: int
) -> _OverlayState:
    breakout_weighted = "breakout_" in arm
    day_balanced = arm.endswith("day_balanced")
    side_models: dict[str, _LocalModel] = {}
    local_models: dict[tuple[str, str], _LocalModel] = {}
    for side, index in train.groupby(
        train["side_name"].astype(str).str.lower(), sort=True
    ).groups.items():
        model = _fit_one(
            train.loc[index],
            features,
            seed=seed + len(side_models) * 101,
            breakout_weighted=breakout_weighted,
            day_balanced=day_balanced,
        )
        if model is not None:
            side_models[str(side)] = model
    groups = train.groupby(
        [
            train["side_name"].astype(str).str.lower(),
            train["archetype_policy_key"].astype(str),
        ],
        observed=True,
        sort=True,
    ).groups
    for (side, archetype), index in groups.items():
        model = _fit_one(
            train.loc[index],
            features,
            seed=seed + len(local_models) * 37 + 11,
            breakout_weighted=breakout_weighted,
            day_balanced=day_balanced,
        )
        if model is not None:
            local_models[(str(side), str(archetype))] = model
    return _OverlayState(
        side_models=side_models,
        local_models=local_models,
        features=list(features),
        train_end=str(train["__ts__"].max()),
    )


def _objective(frame: pd.DataFrame, rank_col: str) -> float:
    selected = frame.loc[
        frame[rank_col].ge(0.90) & frame["ev_after_1pct"].notna()
    ].copy()
    if selected.empty:
        return -np.inf
    selected["day"] = selected["__ts__"].dt.floor("D")
    selected["week"] = selected["day"] - pd.to_timedelta(
        selected["day"].dt.weekday, unit="D"
    )
    daily = selected.groupby("day", observed=True)["ev_after_1pct"].mean()
    weekly = selected.groupby("week", observed=True)["ev_after_1pct"].mean()
    return float(
        selected["ev_after_1pct"].mean()
        - 0.10 * daily.std(ddof=0)
        + 0.20 * daily.min()
        + 0.10 * weekly.min()
    )


def _choose_alpha(history: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    if history.empty:
        return 0.15, pd.DataFrame()
    cutoff = history["__ts__"].max() - pd.Timedelta(days=93)
    probe = history.loc[history["__ts__"].ge(cutoff)].copy()
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        probe["trial_score"] = (1.0 - float(alpha)) * probe["base_batch_rank"] + float(
            alpha
        ) * probe["overlay_rank"]
        probe["trial_rank"] = _batch_rank(probe, "trial_score")
        rows.append(
            {"alpha": float(alpha), "objective": _objective(probe, "trial_rank")}
        )
    search = pd.DataFrame(rows).sort_values(
        ["objective", "alpha"], ascending=[False, True], kind="stable"
    )
    return float(search.iloc[0]["alpha"]), search


def _score_fold(
    valid: pd.DataFrame, state: _OverlayState, alpha: float, arm: str
) -> pd.DataFrame:
    output = valid[
        KEYS + ["calendar_month", "model_score", "base_batch_rank"] + OUTCOMES
    ].copy()
    prediction_z, local_used = state.predict_z(valid)
    output["overlay_expected_ev_z"] = prediction_z
    output["overlay_local_model"] = local_used
    output["overlay_rank"] = (
        pd.Series(prediction_z, index=valid.index)
        .groupby(valid["__ts__"], sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )
    output["alpha"] = np.float32(alpha)
    output["score_adjusted"] = (
        (1.0 - float(alpha)) * output["base_batch_rank"]
        + float(alpha) * output["overlay_rank"]
    ).astype(np.float32)
    output["adjusted_batch_rank"] = _batch_rank(output, "score_adjusted")
    output["arm"] = arm
    return output


def _metric_rows(frame: pd.DataFrame, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selectors = (("baseline", "base_batch_rank"), (arm, "adjusted_batch_rank"))
    scopes = {
        "overall": [],
        "month": ["calendar_month"],
        "week": ["week_start"],
        "day": ["day"],
        "side": ["side_name"],
        "archetype": ["archetype_policy_key"],
        "month_side_archetype": ["calendar_month", "side_name", "archetype_policy_key"],
    }
    work = frame.copy()
    work["day"] = work["__ts__"].dt.floor("D")
    work["week_start"] = work["day"] - pd.to_timedelta(work["day"].dt.weekday, unit="D")
    for scope, groups in scopes.items():
        grouped = (
            [((), work)]
            if not groups
            else work.groupby(groups, observed=True, dropna=False, sort=True)
        )
        for key, part in grouped:
            values = key if isinstance(key, tuple) else (key,)
            for selector, rank_col in selectors:
                selected = part.loc[
                    part[rank_col].ge(0.90) & part["ev_after_1pct"].notna()
                ]
                row: dict[str, Any] = {
                    "arm": arm,
                    "selector": selector,
                    "scope": scope,
                    "candidate_rows": int(len(part)),
                    "selected_rows": int(len(selected)),
                    "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
                    "clean_exec_precision": float(selected["clean_exec"].mean()),
                    "full_path_bad_mae_rate": float(
                        selected["full_path_bad_mae_1r"].mean()
                    ),
                    "timeout_rate": float(selected["timeout"].mean()),
                }
                for name, value in zip(groups, values, strict=False):
                    row[name] = value
                rows.append(row)
    return rows


def _scorecard(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    frame = frame.copy()
    frame["day"] = frame["__ts__"].dt.floor("D")
    frame["week"] = frame["day"] - pd.to_timedelta(frame["day"].dt.weekday, unit="D")
    result: dict[str, Any] = {"arm": arm}
    for selector, rank_col in (
        ("baseline", "base_batch_rank"),
        ("adjusted", "adjusted_batch_rank"),
    ):
        selected = frame.loc[frame[rank_col].ge(0.90) & frame["ev_after_1pct"].notna()]
        daily = selected.groupby("day", observed=True)["ev_after_1pct"].mean()
        weekly = selected.groupby("week", observed=True)["ev_after_1pct"].mean()
        breakout = selected.loc[
            selected["side_name"].astype(str).str.lower().eq("short")
            & selected["archetype_policy_key"]
            .astype(str)
            .str.contains("breakout", case=False, na=False)
        ]
        result.update(
            {
                f"{selector}_rows": int(len(selected)),
                f"{selector}_mean_ev": float(selected["ev_after_1pct"].mean()),
                f"{selector}_clean": float(selected["clean_exec"].mean()),
                f"{selector}_worst_day_ev": float(daily.min()),
                f"{selector}_daily_std": float(daily.std(ddof=0)),
                f"{selector}_positive_days": int(daily.gt(0.0).sum()),
                f"{selector}_days": int(len(daily)),
                f"{selector}_worst_week_ev": float(weekly.min()),
                f"{selector}_breakout_ev": float(breakout["ev_after_1pct"].mean()),
                f"{selector}_breakout_rows": int(len(breakout)),
            }
        )
    result["mean_ev_delta"] = result["adjusted_mean_ev"] - result["baseline_mean_ev"]
    result["worst_day_delta"] = (
        result["adjusted_worst_day_ev"] - result["baseline_worst_day_ev"]
    )
    result["worst_week_delta"] = (
        result["adjusted_worst_week_ev"] - result["baseline_worst_week_ev"]
    )
    result["breakout_ev_delta"] = (
        result["adjusted_breakout_ev"] - result["baseline_breakout_ev"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--july-aligned", type=Path, required=True)
    parser.add_argument("--july-completed", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-start-month", default="2025-09")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="Optional subset of ablation arms to run.",
    )
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical = _prepare_historical(
        args.historical,
        args.output_dir / "cache" / "historical_geometry.parquet",
        args.force_features,
    )
    july, july_coverage = _prepare_july(
        args.july_aligned,
        args.july_completed,
        args.feature_root,
        args.output_dir / "cache" / "july_geometry.parquet",
        args.force_features,
    )
    historical = historical.sort_values("__ts__", kind="stable").reset_index(drop=True)
    july = july.sort_values("__ts__", kind="stable").reset_index(drop=True)
    months = [
        month
        for month in sorted(historical["calendar_month"].dropna().unique())
        if month >= args.eval_start_month and month <= "2026-06"
    ]
    all_metrics: list[dict[str, Any]] = []
    scorecards: list[dict[str, Any]] = []
    fold_manifests: list[dict[str, Any]] = []
    requested_arms = tuple(dict.fromkeys(args.arms))
    for arm_idx, arm in enumerate(requested_arms):
        features = _arm_features(arm, historical.columns)
        history = pd.DataFrame()
        arm_predictions: list[pd.DataFrame] = []
        for fold_idx, month in enumerate(months):
            start = pd.Timestamp(pd.Period(month).start_time, tz="UTC")
            end = pd.Timestamp((pd.Period(month) + 1).start_time, tz="UTC")
            train = historical.loc[historical["__ts__"].lt(start)]
            valid = historical.loc[
                historical["__ts__"].ge(start) & historical["__ts__"].lt(end)
            ].reset_index(drop=True)
            state = _fit_state(
                train, arm, features, 20260711 + arm_idx * 10_000 + fold_idx * 101
            )
            alpha, search = _choose_alpha(history)
            scored = _score_fold(valid, state, alpha, arm)
            arm_predictions.append(scored)
            history = pd.concat([history, scored], ignore_index=True, copy=False)
            fold_manifests.append(
                {
                    "arm": arm,
                    "month": month,
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "alpha": alpha,
                    "alpha_search_rows": int(len(search)),
                    "side_models": int(len(state.side_models)),
                    "local_models": int(len(state.local_models)),
                }
            )
            print(
                json.dumps({"event": "geometry_fold", **fold_manifests[-1]}), flush=True
            )

        final_state = _fit_state(
            historical.loc[
                historical["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC"))
            ],
            arm,
            features,
            20260711 + arm_idx * 10_000 + 999,
        )
        joblib.dump(
            final_state, args.output_dir / f"{arm}_final_state.joblib", compress=3
        )
        importance_rows: list[dict[str, Any]] = []
        for model_key, model in [
            *(
                (f"side::{key}", value)
                for key, value in final_state.side_models.items()
            ),
            *(
                (f"local::{key[0]}::{key[1]}", value)
                for key, value in final_state.local_models.items()
            ),
        ]:
            gain = model.booster.feature_importance(importance_type="gain")
            for feature, value in zip(model.features, gain, strict=False):
                importance_rows.append(
                    {
                        "model_key": model_key,
                        "feature": feature,
                        "gain": float(value),
                        "support_rows": int(model.support_rows),
                    }
                )
        pd.DataFrame(importance_rows).to_csv(
            args.output_dir / f"{arm}_feature_importance.csv", index=False
        )
        july_alpha, july_search = _choose_alpha(history)
        july_scored = _score_fold(
            july.reset_index(drop=True), final_state, july_alpha, arm
        )
        july_scored["evaluation_scope"] = "july_oos"
        historical_scored = pd.concat(arm_predictions, ignore_index=True)
        historical_scored["evaluation_scope"] = "historical_walkforward_oos"
        combined = pd.concat(
            [historical_scored, july_scored], ignore_index=True, copy=False
        )
        combined.to_parquet(
            args.output_dir / f"{arm}_predictions.parquet",
            index=False,
            compression="zstd",
        )
        metrics = _metric_rows(combined, arm)
        pd.DataFrame(metrics).to_csv(
            args.output_dir / f"{arm}_metrics_by_scope.csv", index=False
        )
        historical_card = _scorecard(historical_scored, arm)
        historical_card["evaluation_scope"] = "historical_walkforward_oos"
        july_card = _scorecard(july_scored, arm)
        july_card["evaluation_scope"] = "july_oos"
        scorecards.extend([historical_card, july_card])
        all_metrics.extend(metrics)
        (args.output_dir / f"{arm}_july_alpha_search.csv").write_text(
            july_search.to_csv(index=False), encoding="utf-8"
        )

    scorecard = pd.DataFrame(scorecards)
    scorecard.to_csv(args.output_dir / "scorecard.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(
        args.output_dir / "all_metrics_by_scope.csv", index=False
    )
    pd.DataFrame(fold_manifests).to_csv(
        args.output_dir / "fold_manifest.csv", index=False
    )
    manifest = {
        "schema": "meta_cross_sectional_geometry_ablation_v1",
        "arms": list(requested_arms),
        "eval_months": months,
        "historical_rows": int(len(historical)),
        "july_rows": int(len(july)),
        "july_feature_coverage": july_coverage,
        "selection_contract": "global within-timestamp top 10%; fixed activity denominator",
        "cost_contract": "ev_after_1pct includes 1% round-trip cost",
        "alpha_contract": "selected only from trailing prior OOS predictions; July alpha uses pre-July OOS",
        "model_contract": (
            "shallow LightGBM EV models are fitted separately by side x archetype with side fallback"
        ),
        "leakage_contract": (
            "Each fold fits only earlier frozen OOS rows. Geometry uses current cross-section and lagged "
            "membership only; July is never used for fitting or alpha selection."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nScorecard:")
    print(scorecard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
