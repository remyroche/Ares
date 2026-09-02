#!/usr/bin/env python3
"""In-process fast evaluator for long_dist label/weight Optuna trials.

This intentionally does not replace the full training-path evaluator. It is a
search accelerator: load the frozen deployed feature contract once, fit a
200-tree LightGBM candidate on a spread sample, score a spread holdout, and
write the same metrics JSON shape used by label_weight_optuna.py.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.label_weight_optuna import (
    apply_geometry_recipe_to_labels,
    apply_distillation_recipe,
    apply_label_recipe,
    apply_weight_recipe,
    build_native_base_sample_weight_from_frame,
    build_native_mfe_mae_soft_label_from_frame,
    load_recipe_from_env_or_cfg,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ID = (
    "dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_"
    "leverage_build_score_0_45107844_return_autocorr_48_1_18643_"
    "rolling_range_20_-0_25967735"
)
LABEL_PATH = (
    ROOT
    / "data_perp"
    / "artifacts"
    / "20260523_015947"
    / "labels"
    / f"train_{STRATEGY_ID}_5.parquet"
)
FEATURE_DIR = ROOT / "data_perp" / "features" / "20260523_015947"
MODEL_PATH = (
    ROOT
    / "data_perp"
    / "artifacts"
    / "20260525_010004_nopenalty"
    / "models"
    / "native"
    / f"long_{STRATEGY_ID}_H5"
    / "model.joblib"
)

LABEL_GENERATOR_KEYS = frozenset(
    {
        "lgbm_soft_label_costs",
        "lgbm_soft_label_min_opportunity_mult",
        "lgbm_soft_label_temperature",
        "net_executable_mae_lambda",
        "net_executable_center_vol",
        "net_executable_temperature_vol",
        "policy_label_sl_atr_mult",
        "policy_label_tp_sl_ratio",
        "policy_label_trailing_pct",
        "policy_label_max_hold_hours",
    }
)
WEIGHT_GENERATOR_KEYS = frozenset(
    {
        "timeout_weight",
        "outcome_weight_clip_min",
        "outcome_weight_clip_max",
        "mfe_mae_w_min",
        "mfe_mae_tau",
        "mfe_mae_cost_floor",
        "meta_weight_sigmoid_alpha",
        "meta_mfe_mae_tau",
    }
)


@dataclass
class FastEvalData:
    frame: pd.DataFrame
    x: np.ndarray
    selected_features: list[str]
    best_params: dict[str, Any]
    train_idx: np.ndarray
    eval_idx: np.ndarray
    y_hard: np.ndarray
    y_ret: np.ndarray
    base_weight: np.ndarray
    timestamps: pd.Series


def _symbol_to_feature_path(symbol: str) -> Path:
    return FEATURE_DIR / f"symbol={str(symbol).replace('/', '_')}.parquet"


def _spread_indices(n: int, cap: int, *, offset: int = 0, exclude: np.ndarray | None = None) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    allowed = np.ones(n, dtype=bool)
    if exclude is not None and len(exclude):
        allowed[np.asarray(exclude, dtype=np.int64)] = False
    pool = np.flatnonzero(allowed)
    if len(pool) <= cap:
        return pool.astype(np.int64)
    pos = np.linspace(0, len(pool) - 1, num=cap, dtype=np.int64)
    if offset:
        pos = np.unique(np.clip(pos + int(offset), 0, len(pool) - 1))
        if len(pos) < cap:
            missing = cap - len(pos)
            extra = np.linspace(0, len(pool) - 1, num=missing, dtype=np.int64)
            pos = np.unique(np.concatenate([pos, extra]))[:cap]
    return pool[pos].astype(np.int64)


def _spread_timestamp_group_indices(
    timestamps: pd.Series,
    target_rows: int,
    *,
    offset: int = 0,
    exclude_timestamps: set[pd.Timestamp] | None = None,
) -> np.ndarray:
    ts = pd.to_datetime(timestamps, utc=True)
    unique = pd.Index(ts.drop_duplicates())
    if exclude_timestamps:
        unique = pd.Index([t for t in unique if t not in exclude_timestamps])
    if len(unique) == 0:
        return np.asarray([], dtype=np.int64)
    counts = ts.value_counts(sort=False)
    median_count = max(1.0, float(np.nanmedian(counts.reindex(unique).fillna(0).to_numpy(dtype=float))))
    target_groups = int(np.clip(np.ceil(float(target_rows) / median_count), 1, len(unique)))
    pos = np.linspace(0, len(unique) - 1, num=target_groups, dtype=np.int64)
    if offset:
        pos = np.unique(np.clip(pos + int(offset), 0, len(unique) - 1))
        if len(pos) < target_groups:
            extra = np.linspace(0, len(unique) - 1, num=target_groups - len(pos), dtype=np.int64)
            pos = np.unique(np.concatenate([pos, extra]))[:target_groups]
    chosen = set(pd.Index(unique[pos]))
    return np.flatnonzero(ts.isin(chosen).to_numpy()).astype(np.int64)


def _current_soft_label(df: pd.DataFrame, y_hard: np.ndarray, *, cfg: dict[str, Any]) -> np.ndarray:
    soft, _ = build_native_mfe_mae_soft_label_from_frame(
        df,
        y_hard,
        cfg=cfg,
        stage="train_base",
        label="fast_native",
    )
    return np.asarray(soft, dtype=np.float32)


def _recipe_has_generator_overrides(cfg: dict[str, Any], keys: frozenset[str]) -> bool:
    recipe = load_recipe_from_env_or_cfg(cfg)
    if recipe is None:
        return False
    return any(getattr(recipe.generator, key, None) is not None for key in keys)


def _array_stats(values: np.ndarray, *, reference: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    out: dict[str, float] = {
        "count": float(arr.size),
        "finite_count": float(np.sum(finite)),
    }
    if not np.any(finite):
        out.update(
            {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p01": float("nan"),
                "p10": float("nan"),
                "p50": float("nan"),
                "p90": float("nan"),
                "p99": float("nan"),
                "max": float("nan"),
            }
        )
        return out
    clean = arr[finite]
    out.update(
        {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "min": float(np.min(clean)),
            "p01": float(np.percentile(clean, 1)),
            "p10": float(np.percentile(clean, 10)),
            "p50": float(np.percentile(clean, 50)),
            "p90": float(np.percentile(clean, 90)),
            "p99": float(np.percentile(clean, 99)),
            "max": float(np.max(clean)),
        }
    )
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        n = min(len(arr), len(ref))
        delta = arr[:n] - ref[:n]
        delta_finite = np.isfinite(delta)
        if np.any(delta_finite):
            d = delta[delta_finite]
            out.update(
                {
                    "delta_mean": float(np.mean(d)),
                    "delta_abs_mean": float(np.mean(np.abs(d))),
                    "delta_abs_p95": float(np.percentile(np.abs(d), 95)),
                    "changed_frac_gt_1e_6": float(np.mean(np.abs(d) > 1e-6)),
                }
            )
    return out


def _weight_diagnostics(values: np.ndarray, *, reference: np.ndarray | None = None) -> dict[str, float]:
    out = _array_stats(values, reference=reference)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr) & (arr > 0.0)
    if np.any(finite):
        clean = arr[finite]
        denom = float(np.sum(clean * clean))
        out["effective_sample_size"] = float((np.sum(clean) ** 2) / denom) if denom > 1e-12 else 0.0
        out["effective_sample_frac"] = float(out["effective_sample_size"] / max(len(arr), 1))
    else:
        out["effective_sample_size"] = 0.0
        out["effective_sample_frac"] = 0.0
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)
        n = min(len(arr), len(ref))
        if n >= 3:
            rank_frame = pd.DataFrame({"value": arr[:n], "reference": ref[:n]}).replace(
                [np.inf, -np.inf],
                np.nan,
            )
            rank_frame = rank_frame.dropna()
            if len(rank_frame) >= 3:
                corr = rank_frame["value"].rank().corr(rank_frame["reference"].rank())
                out["rank_corr_to_reference"] = float(corr) if pd.notna(corr) else 0.0
    return out


def _read_feature_block(symbol: str, timestamps: pd.Series, selected_features: list[str]) -> np.ndarray:
    path = _symbol_to_feature_path(symbol)
    out = np.full((len(timestamps), len(selected_features)), np.nan, dtype=np.float32)
    if not path.exists():
        return out
    try:
        feat = pd.read_parquet(path, columns=selected_features)
        available = list(selected_features)
    except Exception:
        try:
            import pyarrow.parquet as pq

            names = set(pq.read_schema(path).names)
            available = [c for c in selected_features if c in names]
            if not available:
                return out
            feat = pd.read_parquet(path, columns=available)
        except Exception:
            return out
    feat.index = pd.to_datetime(feat.index, utc=True)
    aligned = feat.reindex(pd.to_datetime(timestamps, utc=True))
    col_pos = {name: idx for idx, name in enumerate(selected_features)}
    for col in available:
        out[:, col_pos[col]] = aligned[col].to_numpy(dtype=np.float32, copy=False)
    return out


def _load_data() -> FastEvalData:
    t0 = time.monotonic()
    model = joblib.load(MODEL_PATH)
    selected = list(getattr(model, "selected_features", []) or [])
    if not selected:
        raise RuntimeError(f"No selected features found in deployed native model: {MODEL_PATH}")
    best_params = dict(getattr(model, "best_params", {}) or {})
    label_cols = [
        "__y_bin__",
        "__y_ret__",
        "__y_outcome__",
        "__w__",
        "__ts__",
        "__symbol__",
        "__mfe_ret__",
        "__mae_ret__",
        "__barrier_pct__",
        "__is_timeout__",
        "__mfe__",
        "__mae__",
        "__tp__",
        "__sl__",
        "__bars_to_mfe__",
        "__bars_to_mae__",
        "__bars_policy__",
    ]
    label_schema_names = set(pq.read_schema(LABEL_PATH).names)
    df = pd.read_parquet(LABEL_PATH, columns=[c for c in label_cols if c in label_schema_names])
    df["__ts__"] = pd.to_datetime(df["__ts__"], utc=True)
    recent_days = float(os.getenv("EPM_TRAIN_RECENT_DAYS", "730"))
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
    df = df[df["__ts__"] >= cutoff].copy()
    df.sort_values(["__ts__", "__symbol__"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    x = np.full((len(df), len(selected)), np.nan, dtype=np.float32)
    loaded_symbols = 0
    for symbol, idx in df.groupby("__symbol__", sort=False).indices.items():
        rows = np.asarray(idx, dtype=np.int64)
        block = _read_feature_block(str(symbol), df.loc[rows, "__ts__"], selected)
        x[rows, :] = block
        loaded_symbols += int(np.isfinite(block).any())
    finite = np.isfinite(x).all(axis=1)
    label_finite = np.isfinite(df["__y_bin__"].to_numpy(dtype=float)) & np.isfinite(df["__y_ret__"].to_numpy(dtype=float))
    keep = finite & label_finite
    df = df.loc[keep].reset_index(drop=True)
    x = x[keep].astype(np.float32, copy=False)
    if len(df) < 50_000:
        raise RuntimeError(f"Fast evaluator retained too few complete rows: {len(df)}")
    y_hard = df["__y_bin__"].to_numpy(dtype=np.float32)
    y_ret = df["__y_ret__"].to_numpy(dtype=np.float32)
    base_weight = (
        df["__w__"].to_numpy(dtype=np.float32)
        if "__w__" in df.columns
        else np.ones(len(df), dtype=np.float32)
    )
    order = np.argsort(df["__ts__"].astype("int64").to_numpy(), kind="mergesort")
    df = df.iloc[order].reset_index(drop=True)
    x = x[order]
    y_hard = y_hard[order]
    y_ret = y_ret[order]
    base_weight = base_weight[order]
    eval_cap = int(os.getenv("EPM_LABEL_WEIGHT_FAST_EVAL_ROWS", "80000"))
    train_cap = int(os.getenv("EPM_LABEL_WEIGHT_FAST_TRAIN_ROWS", "160000"))
    eval_idx = _spread_timestamp_group_indices(df["__ts__"], eval_cap, offset=0)
    eval_ts = set(pd.to_datetime(df["__ts__"].iloc[eval_idx], utc=True).drop_duplicates())
    train_idx = _spread_timestamp_group_indices(
        df["__ts__"],
        train_cap,
        offset=1,
        exclude_timestamps=eval_ts,
    )
    if len(eval_idx) == 0 or len(train_idx) == 0:
        eval_idx = _spread_indices(len(df), eval_cap, offset=0)
        train_idx = _spread_indices(len(df), train_cap, offset=max(1, len(df) // max(eval_cap, 1) // 2), exclude=eval_idx)
    print(
        "FAST_LABEL_WEIGHT_EVAL cache ready: "
        f"rows={len(df)} train={len(train_idx)} eval={len(eval_idx)} "
        f"features={len(selected)} loaded_symbols={loaded_symbols} "
        f"elapsed={time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return FastEvalData(
        frame=df,
        x=x,
        selected_features=selected,
        best_params=best_params,
        train_idx=train_idx,
        eval_idx=eval_idx,
        y_hard=y_hard,
        y_ret=y_ret,
        base_weight=base_weight,
        timestamps=df["__ts__"],
    )


def _fit_predict(data: FastEvalData, y_soft: np.ndarray, weights: np.ndarray, *, trial_number: int, phase: str) -> np.ndarray:
    import lightgbm as lgb

    params = dict(data.best_params)
    params["objective"] = "cross_entropy"
    num_boost_round = min(
        int(params.pop("n_estimators", 200)),
        int(os.getenv("EPM_LGBM_N_ESTIMATORS_CAP", "200")),
    )
    params["num_threads"] = int(os.getenv("EPM_LABEL_WEIGHT_FAST_N_JOBS", params.pop("n_jobs", 3)))
    params["verbosity"] = -1
    fixed_seed = int(os.getenv("EPM_LABEL_WEIGHT_FAST_FIXED_SEED", params.pop("random_state", 364)))
    params["seed"] = fixed_seed
    params["random_state"] = fixed_seed
    params["bagging_seed"] = fixed_seed
    params["feature_fraction_seed"] = fixed_seed
    params["data_random_seed"] = fixed_seed
    params["drop_seed"] = fixed_seed
    params.pop("scale_pos_weight", None)
    params.pop("class_weight", None)
    params.pop("is_unbalance", None)
    train_idx = data.train_idx
    eval_idx = data.eval_idx
    fit_weight = np.asarray(weights[train_idx], dtype=np.float32)
    if str(phase).strip().lower() == "distillation":
        first_ds = lgb.Dataset(
            data.x[train_idx],
            label=np.asarray(y_soft[train_idx], dtype=np.float32),
            weight=fit_weight,
            free_raw_data=True,
        )
        first = lgb.train(params, first_ds, num_boost_round=num_boost_round)
        pred_train = np.clip(first.predict(data.x[train_idx]), 0.0, 1.0)
        distill = np.ones(len(train_idx), dtype=np.float32)
        fp_weight = np.ones(len(train_idx), dtype=np.float32)
        distill, fp_weight = apply_distillation_recipe(
            distill,
            fp_weight,
            y_metric=y_soft[train_idx],
            pred=pred_train,
            returns=data.y_ret[train_idx],
            timestamps=data.timestamps.iloc[train_idx],
            objective_mode="train_base",
        )
        fit_weight = fit_weight * np.asarray(distill, dtype=np.float32) * np.asarray(fp_weight, dtype=np.float32)
        fit_weight = fit_weight / max(float(np.nanmean(fit_weight)), 1e-12)
    ds = lgb.Dataset(
        data.x[train_idx],
        label=np.asarray(y_soft[train_idx], dtype=np.float32),
        weight=fit_weight,
        free_raw_data=True,
    )
    model = lgb.train(params, ds, num_boost_round=num_boost_round)
    return np.clip(model.predict(data.x[eval_idx]), 0.0, 1.0).astype(np.float32)


def _hhi(values: pd.Series) -> float:
    if values.empty:
        return 1.0
    shares = values.astype(str).value_counts(normalize=True).to_numpy(dtype=float)
    return float(np.sum(np.square(shares))) if len(shares) else 1.0


def _weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if int(np.sum(finite)) < 3:
        return 0.0
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    weights = weights[finite].astype(float)
    weights = weights / max(float(np.sum(weights)), 1e-12)
    x_centered = x - float(np.sum(weights * x))
    y_centered = y - float(np.sum(weights * y))
    cov = float(np.sum(weights * x_centered * y_centered))
    x_var = float(np.sum(weights * x_centered * x_centered))
    y_var = float(np.sum(weights * y_centered * y_centered))
    denom = math.sqrt(max(x_var * y_var, 0.0))
    return float(cov / denom) if denom > 1e-12 else 0.0


def _weighted_spearman(score: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    return _weighted_corr(score_rank, target_rank, weights)


def _top_rank_weights(score_rank_pct: np.ndarray, *, top_frac: float) -> np.ndarray:
    threshold = max(0.0, min(1.0, 1.0 - float(top_frac)))
    width = max(float(top_frac), 1e-6)
    ramp = np.clip((score_rank_pct - threshold) / width, 0.0, 1.0)
    return 0.25 + 0.75 * ramp


def _topk_metrics(df: pd.DataFrame, *, start: pd.Timestamp | None, cost_bps: float, k: int) -> dict[str, float]:
    d = df if start is None else df[df["timestamp"] >= start]
    d = d[np.isfinite(d["oof_prob"]) & np.isfinite(d["y_ret"])]
    top_frac = float(np.clip(float(k) / 100.0, 0.0, 1.0))
    if d.empty:
        return {
            "n": 0.0,
            "top_fraction": top_frac,
            "top_count": 0.0,
            "net_hit": 0.0,
            "bps_weighted_hit": 0.0,
            "mean_net_bps": 0.0,
            "median_net_bps": 0.0,
            "avg_win_net_bps": 0.0,
            "avg_loss_net_bps": 0.0,
            "avg_stop_loss_bps": 0.0,
            "stop_hit_rate": 1.0,
            "unique_symbols": 0.0,
            "symbol_hhi": 1.0,
            "unique_weeks": 0.0,
            "week_hhi": 1.0,
        }
    top_n = max(1, int(math.ceil(len(d) * top_frac))) if top_frac > 0.0 else 0
    top = d.sort_values("oof_prob", ascending=False).head(top_n)
    net_bps = top["y_ret"].to_numpy(dtype=float) * 10_000.0 - cost_bps
    mae_bps = top.get("mae_ret", pd.Series(np.zeros(len(top)), index=top.index)).to_numpy(dtype=float) * 10_000.0
    win = net_bps > 0.0
    loss = net_bps <= 0.0
    bps_weighted_hit = float(
        np.nansum(np.maximum(net_bps, 0.0)) / max(float(np.nansum(np.abs(net_bps))), 1e-12)
    )
    if "exit_code" in top.columns:
        exit_code = pd.to_numeric(top["exit_code"], errors="coerce").to_numpy(dtype=float)
        stop = np.isfinite(exit_code) & (exit_code == 0.0)
    else:
        stop = mae_bps > 100.0
    stop_losses = np.maximum(-net_bps[stop], 0.0)
    weeks = pd.to_datetime(top["timestamp"], utc=True).dt.tz_convert(None).dt.to_period("W").astype(str)
    return {
        "n": float(len(top)),
        "top_fraction": top_frac,
        "top_count": float(len(top)),
        "net_hit": float(np.mean(win)),
        "bps_weighted_hit": bps_weighted_hit,
        "mean_net_bps": float(np.nanmean(net_bps)),
        "median_net_bps": float(np.nanmedian(net_bps)),
        "avg_win_net_bps": float(np.nanmean(net_bps[win])) if np.any(win) else 0.0,
        "avg_loss_net_bps": float(np.nanmean(-net_bps[loss])) if np.any(loss) else 0.0,
        "avg_stop_loss_bps": float(np.nanmean(stop_losses)) if len(stop_losses) else 0.0,
        "stop_hit_rate": float(np.mean(stop)),
        "unique_symbols": float(top["symbol"].nunique()) if "symbol" in top else 0.0,
        "symbol_hhi": _hhi(top["symbol"]) if "symbol" in top else 1.0,
        "unique_weeks": float(weeks.nunique()),
        "week_hhi": _hhi(weeks),
    }


def _ranking_surface_metrics(df: pd.DataFrame, *, start: pd.Timestamp | None, cost_bps: float) -> dict[str, float]:
    d = df if start is None else df[df["timestamp"] >= start]
    d = d[np.isfinite(d["oof_prob"]) & np.isfinite(d["y_ret"])]
    if d.empty:
        return {
            "prediction_score_std": 0.0,
            "prediction_score_iqr": 0.0,
            "score_gap_top10_to_30_40": 0.0,
            "economic_rank_ic": 0.0,
            "economic_weighted_ic": 0.0,
            "economic_weighted_ic_full": 0.0,
            "economic_weighted_ic_top30": 0.0,
            "economic_weighted_ic_top20": 0.0,
            "economic_weighted_ic_top10": 0.0,
            "economic_rank_monotonicity": 0.5,
            "economic_bucket_spread_bps": 0.0,
        }
    score = d["oof_prob"].to_numpy(dtype=float)
    net_bps = d["y_ret"].to_numpy(dtype=float) * 10_000.0 - cost_bps
    finite = np.isfinite(score) & np.isfinite(net_bps)
    score = score[finite]
    net_bps = net_bps[finite]
    if len(score) < 10:
        return {
            "prediction_score_std": float(np.nanstd(score)) if len(score) else 0.0,
            "prediction_score_iqr": 0.0,
            "score_gap_top10_to_30_40": 0.0,
            "economic_rank_ic": 0.0,
            "economic_weighted_ic": 0.0,
            "economic_weighted_ic_full": 0.0,
            "economic_weighted_ic_top30": 0.0,
            "economic_weighted_ic_top20": 0.0,
            "economic_weighted_ic_top10": 0.0,
            "economic_rank_monotonicity": 0.5,
            "economic_bucket_spread_bps": 0.0,
        }
    ranks = pd.Series(score).rank(pct=True, method="average").to_numpy(dtype=float)
    top10 = score[ranks >= 0.90]
    mid30_40 = score[(ranks >= 0.60) & (ranks < 0.70)]
    score_gap = (
        float(np.nanmean(top10) - np.nanmean(mid30_40))
        if len(top10) and len(mid30_40)
        else 0.0
    )
    rank_ic = pd.Series(score).corr(pd.Series(net_bps), method="spearman")
    full_weights = 0.50 + 0.50 * ranks
    weighted_ic_full = _weighted_spearman(score, net_bps, full_weights)
    weighted_ic_top30 = _weighted_spearman(score, net_bps, _top_rank_weights(ranks, top_frac=0.30))
    weighted_ic_top20 = _weighted_spearman(score, net_bps, _top_rank_weights(ranks, top_frac=0.20))
    weighted_ic_top10 = _weighted_spearman(score, net_bps, _top_rank_weights(ranks, top_frac=0.10))
    weighted_ic = (
        0.25 * weighted_ic_full
        + 0.20 * weighted_ic_top30
        + 0.15 * weighted_ic_top20
        + 0.10 * weighted_ic_top10
    ) / 0.70
    try:
        bucket = pd.qcut(score, q=min(10, len(np.unique(score))), labels=False, duplicates="drop")
    except ValueError:
        bucket = None
    if bucket is None or pd.isna(bucket).all():
        monotonicity = 0.5
        bucket_spread = 0.0
    else:
        by_bucket = pd.DataFrame({"bucket": bucket, "net_bps": net_bps}).dropna()
        means = by_bucket.groupby("bucket", sort=True)["net_bps"].mean().to_numpy(dtype=float)
        if len(means) < 2:
            monotonicity = 0.5
            bucket_spread = 0.0
        else:
            diffs = np.diff(means)
            monotonicity = float(np.mean(diffs >= 0.0))
            bucket_spread = float(means[-1] - means[0])
    return {
        "prediction_score_std": float(np.nanstd(score)),
        "prediction_score_iqr": float(np.nanpercentile(score, 75) - np.nanpercentile(score, 25)),
        "score_gap_top10_to_30_40": score_gap,
        "economic_rank_ic": float(rank_ic) if rank_ic is not None and math.isfinite(float(rank_ic)) else 0.0,
        "economic_weighted_ic": float(weighted_ic),
        "economic_weighted_ic_full": float(weighted_ic_full),
        "economic_weighted_ic_top30": float(weighted_ic_top30),
        "economic_weighted_ic_top20": float(weighted_ic_top20),
        "economic_weighted_ic_top10": float(weighted_ic_top10),
        "economic_rank_monotonicity": monotonicity,
        "economic_bucket_spread_bps": bucket_spread,
    }


def _unit_interval(value: float, *, floor: float, good: float) -> float:
    if not math.isfinite(float(value)) or abs(good - floor) <= 1e-12:
        return 0.0
    return float(np.clip((float(value) - floor) / (good - floor), 0.0, 1.0))


def _lgbm_style_j_proxy(metrics: dict[str, Any]) -> float:
    ic_component = _unit_interval(float(metrics.get("economic_weighted_ic", 0.0)), floor=-0.02, good=0.12)
    mono_component = float(np.clip(float(metrics.get("economic_rank_monotonicity", 0.5)), 0.0, 1.0))
    spread_component = _unit_interval(float(metrics.get("economic_bucket_spread_bps", 0.0)), floor=-25.0, good=100.0)
    std_component = _unit_interval(float(metrics.get("prediction_score_std", 0.0)), floor=0.01, good=0.10)
    return float(0.25 + 0.30 * ic_component + 0.25 * mono_component + 0.12 * spread_component + 0.08 * std_component)


class FastLongDistEvaluator:
    def __init__(self) -> None:
        self.data = _load_data()

    def evaluate(self, *, recipe_path: str, trial_number: int, phase: str, metrics_json: str | Path | None = None) -> dict[str, Any]:
        if str(recipe_path).strip() == "__no_recipe__":
            os.environ.pop("EPM_LABEL_WEIGHT_RECIPE", None)
            os.environ["EPM_LABEL_WEIGHT_USE_BEST_DEFAULT"] = "1"
            os.environ["EPM_LABEL_WEIGHT_BEST_RECIPE"] = str(ROOT / "reports_perp" / "label_weight_optuna" / "__missing_no_recipe__.json")
        else:
            os.environ["EPM_LABEL_WEIGHT_RECIPE"] = str(recipe_path)
            os.environ["EPM_LABEL_WEIGHT_USE_BEST_DEFAULT"] = "0"
        data = self.data
        eval_cfg = {"execution_aware_cost_bps": float(os.getenv("EPM_EXECUTION_AWARE_COST_BPS", "68.83"))}
        recipe_frame, recipe_y_hard, geometry_stats = apply_geometry_recipe_to_labels(
            data.frame,
            data.y_hard,
            cfg=eval_cfg,
            stage="train_base",
            label=f"fast_trial_{trial_number}",
        )
        current_soft, current_soft_stats = build_native_mfe_mae_soft_label_from_frame(
            recipe_frame,
            recipe_y_hard,
            cfg=eval_cfg,
            stage="train_base",
            label="fast_native",
        )
        current_soft = np.asarray(current_soft, dtype=np.float32)
        y_soft, _ = apply_label_recipe(
            recipe_frame,
            recipe_y_hard,
            current_soft,
            cfg=eval_cfg,
            stage="train_base",
            label=f"fast_trial_{trial_number}",
        )
        base_weight = data.base_weight
        native_weight_stats: dict[str, Any] = {
            "enabled": False,
            "reason": "no_weight_generator_overrides",
        }
        if _recipe_has_generator_overrides(eval_cfg, WEIGHT_GENERATOR_KEYS):
            base_weight, native_weight_stats = build_native_base_sample_weight_from_frame(
                recipe_frame,
                recipe_y_hard,
                data.y_ret,
                cfg=eval_cfg,
                stage="train_base",
            )
        weights, _ = apply_weight_recipe(
            recipe_frame,
            recipe_y_hard,
            y_soft,
            base_weight,
            cfg=eval_cfg,
            stage="train_base",
            label=f"fast_trial_{trial_number}",
            fit_indices=data.train_idx,
        )
        final_soft_stats = _array_stats(y_soft, reference=current_soft)
        final_weight_stats = _weight_diagnostics(weights, reference=base_weight)
        pred = _fit_predict(data, y_soft, weights, trial_number=trial_number, phase=phase)
        eval_idx = data.eval_idx
        scored = pd.DataFrame(
            {
                "timestamp": data.timestamps.iloc[eval_idx].to_numpy(),
                "symbol": data.frame["__symbol__"].iloc[eval_idx].to_numpy(),
                "oof_prob": pred,
                "y_ret": data.y_ret[eval_idx],
                "mae_ret": data.frame["__mae_ret__"].iloc[eval_idx].to_numpy(dtype=np.float32),
                "exit_code": data.frame.get("__y_outcome__", pd.Series(np.nan, index=data.frame.index)).iloc[
                    eval_idx
                ].to_numpy(),
            }
        )
        scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)
        max_ts = scored["timestamp"].max()
        cost_bps = float(os.getenv("EPM_EXECUTION_AWARE_COST_BPS", "68.83"))
        windows = {
            "full": None,
            "26w": max_ts - pd.Timedelta(weeks=26),
            "13w": max_ts - pd.Timedelta(weeks=13),
            "8w": max_ts - pd.Timedelta(weeks=8),
        }
        per_window: dict[str, dict[str, dict[str, float]]] = {}
        per_window_ranking: dict[str, dict[str, float]] = {}
        for win, start in windows.items():
            per_window[win] = {str(k): _topk_metrics(scored, start=start, cost_bps=cost_bps, k=k) for k in (10, 20, 30, 50)}
            per_window_ranking[win] = _ranking_surface_metrics(scored, start=start, cost_bps=cost_bps)
        weights_by_window = {"full": 0.20, "26w": 0.30, "13w": 0.30, "8w": 0.20}
        metrics: dict[str, Any] = {
            "model_stage": "base",
            "execution_cost_bps": cost_bps,
            "run_id": f"fast_label_weight_trial_{trial_number}",
            "fast_eval": True,
            "train_rows": int(len(data.train_idx)),
            "eval_rows": int(len(data.eval_idx)),
            "features": int(len(data.selected_features)),
            "per_window": per_window,
            "per_window_ranking": per_window_ranking,
            "label_generator_overrides_present": _recipe_has_generator_overrides(eval_cfg, LABEL_GENERATOR_KEYS),
            "weight_generator_overrides_present": _recipe_has_generator_overrides(eval_cfg, WEIGHT_GENERATOR_KEYS),
            "geometry_stats": geometry_stats,
            "native_label_generator_stats": current_soft_stats,
            "native_weight_generator_stats": native_weight_stats,
            "label_final_changed_frac": float(final_soft_stats.get("changed_frac_gt_1e_6", 0.0)),
            "label_final_delta_abs_mean": float(final_soft_stats.get("delta_abs_mean", 0.0)),
            "weight_final_changed_frac": float(final_weight_stats.get("changed_frac_gt_1e_6", 0.0)),
            "weight_final_delta_abs_mean": float(final_weight_stats.get("delta_abs_mean", 0.0)),
            "weight_rank_corr_to_baseline": float(final_weight_stats.get("rank_corr_to_reference", 1.0)),
            "effective_sample_frac": float(final_weight_stats.get("effective_sample_frac", 0.0)),
            "label_stats": {
                "source_hard": _array_stats(data.y_hard),
                "geometry_hard": _array_stats(recipe_y_hard, reference=data.y_hard),
                "native_soft": _array_stats(current_soft, reference=recipe_y_hard),
                "final_soft": final_soft_stats,
            },
            "weight_stats": {
                "source_base": _weight_diagnostics(data.base_weight),
                "native_base": _weight_diagnostics(base_weight, reference=data.base_weight),
                "final": final_weight_stats,
            },
        }
        for k in (10, 20, 30, 50):
            metrics[f"net_hit_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["net_hit"] for w in weights_by_window))
            metrics[f"bps_weighted_hit_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["bps_weighted_hit"] for w in weights_by_window))
            metrics[f"avg_win_net_bps_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["avg_win_net_bps"] for w in weights_by_window))
            metrics[f"avg_loss_net_bps_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["avg_loss_net_bps"] for w in weights_by_window))
            metrics[f"mean_net_bps_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["mean_net_bps"] for w in weights_by_window))
            metrics[f"median_net_bps_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["median_net_bps"] for w in weights_by_window))
            metrics[f"unique_symbols_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["unique_symbols"] for w in weights_by_window))
            metrics[f"symbol_concentration_hhi_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["symbol_hhi"] for w in weights_by_window))
            metrics[f"unique_weeks_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["unique_weeks"] for w in weights_by_window))
            metrics[f"week_concentration_hhi_at_{k}"] = float(sum(weights_by_window[w] * per_window[w][str(k)]["week_hhi"] for w in weights_by_window))
        metrics["avg_stop_loss_bps_at_20"] = float(sum(weights_by_window[w] * per_window[w]["20"]["avg_stop_loss_bps"] for w in weights_by_window))
        metrics["stop_hit_rate_at_20"] = float(sum(weights_by_window[w] * per_window[w]["20"]["stop_hit_rate"] for w in weights_by_window))
        metrics["prediction_instability"] = float(np.nanstd([per_window[w]["20"]["bps_weighted_hit"] for w in weights_by_window]))
        for name in (
            "prediction_score_std",
            "prediction_score_iqr",
            "score_gap_top10_to_30_40",
            "economic_rank_ic",
            "economic_weighted_ic",
            "economic_weighted_ic_full",
            "economic_weighted_ic_top30",
            "economic_weighted_ic_top20",
            "economic_weighted_ic_top10",
            "economic_rank_monotonicity",
            "economic_bucket_spread_bps",
        ):
            metrics[name] = float(sum(weights_by_window[w] * per_window_ranking[w][name] for w in weights_by_window))
        j_proxy = _lgbm_style_j_proxy(metrics)
        metrics["J_base"] = j_proxy
        metrics["J_final"] = j_proxy
        metrics["J_proxy"] = j_proxy
        metrics["J_source"] = "fast_label_weight_proxy"
        if metrics_json:
            out = Path(metrics_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metrics
