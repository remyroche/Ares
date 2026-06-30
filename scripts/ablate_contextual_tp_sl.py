#!/usr/bin/env python3
"""Report-only contextual TP/SL ablation for simple-policy candidates.

This script does not write deployment artifacts.  It tests whether take-profit
and stop-loss distances should be modulated by deployable diagnostics such as
rank, drift, OOD/novelty, and meta-model uncertainty.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

from extreme_price_movements.simple_policy_optimiser import (
    _apply_delayed_entry_execution_model,
    _fetch_policy_paths,
    _json_safe,
    _make_policy_replay_store,
    _path_take,
    _policy_path_finite_mask,
)
from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)


DEFAULT_BAR_MINUTES = 15

RANK_FEATURES = [
    "rank_pct",
    "strategy_rank_pct",
    "normalized_rank_score",
    "auction_rank_score",
    "calibrated_score",
    "simple_policy_calibrated_good_trade_prob",
]
UNCERTAINTY_FEATURES = [
    "oof_prob_uncertainty",
    "oof_contrib_entropy",
    "oof_rank_bin_se_oof",
    "oof_score_path_std",
    "oof_score_path_volatility",
    "oof_rank_path_std",
    "oof_score_reversal_count",
]
UNCERTAINTY_INVERSE_FEATURES = [
    "oof_score_margin_top10",
    "oof_score_margin_top20",
    "oof_score_margin_top30",
    "oof_rank_margin_top10",
    "oof_rank_margin_top20",
    "oof_rank_margin_top30",
]
DRIFT_FEATURES = [
    "oof_feature_drift_psi_core",
    "oof_feature_drift_ks_core",
    "oof_feature_drift_cov_shift",
]
OOD_FEATURES = [
    "oof_dae_reconstruction_error",
    "oof_dae_reconstruction_error_zscore",
    "oof_latent_mahalanobis_drift",
    "oof_support_gap",
    "oof_rare_leaf_fraction",
]
OOD_INVERSE_FEATURES = [
    "oof_leaf_count_mean",
    "oof_leaf_count_median",
    "oof_leaf_count_q25",
    "oof_leaf_count_p10",
    "oof_leaf_count_min",
    "oof_leaf_train_freq_mean",
    "oof_leaf_train_freq_p10",
    "oof_leaf_train_freq_min",
]


@dataclass
class FeatureState:
    columns: List[str]
    signs: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    groups: List[str]


def _side_code(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"-1", "short", "sell"} or text.startswith("short"):
        return -1.0
    return 1.0


def _prepare_rows(path: Path, *, min_rank: float) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    required = {"timestamp", "symbol", "strategy_id", "rank_pct", "barrier_pct"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    rows = rows.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol", "strategy_id"]).copy()
    rows["rank_pct"] = pd.to_numeric(rows["rank_pct"], errors="coerce")
    rows = rows.loc[rows["rank_pct"].ge(float(min_rank))].copy()
    rows["side"] = [_side_code(v) for v in rows.get("side", 1.0)]
    rows["symbol"] = rows["symbol"].astype(str)
    rows["strategy_id"] = rows["strategy_id"].astype(str)
    rows = rows.sort_values(["strategy_id", "timestamp", "symbol"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows left after rank_pct >= {min_rank}")
    return rows


def _chronological_split(
    rows: pd.DataFrame,
    *,
    validation_frac: float,
) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    ts_unique = (
        pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    if len(ts_unique) < 10:
        raise ValueError("Need at least ten timestamps for chronological split")
    cut_pos = int(np.floor(len(ts_unique) * (1.0 - float(validation_frac))))
    cut_pos = min(max(cut_pos, 1), len(ts_unique) - 1)
    cut_ts = pd.Timestamp(ts_unique[cut_pos])
    cut_ts = cut_ts.tz_localize("UTC") if cut_ts.tzinfo is None else cut_ts.tz_convert("UTC")
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    train_idx = np.flatnonzero(ts.lt(cut_ts).to_numpy()).astype(np.int64)
    val_idx = np.flatnonzero(ts.ge(cut_ts).to_numpy()).astype(np.int64)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Chronological split produced empty train or validation")
    return train_idx, val_idx, cut_ts


def _feature_columns(rows: pd.DataFrame, groups: Sequence[str]) -> Tuple[List[str], List[str], np.ndarray]:
    candidates: List[Tuple[str, str, float]] = []
    group_set = set(groups)
    if "rank" in group_set:
        candidates += [(c, "rank", 1.0) for c in RANK_FEATURES]
    if "uncertainty" in group_set:
        candidates += [(c, "uncertainty", 1.0) for c in UNCERTAINTY_FEATURES]
        candidates += [(c, "uncertainty", -1.0) for c in UNCERTAINTY_INVERSE_FEATURES]
    if "drift" in group_set:
        candidates += [(c, "drift", 1.0) for c in DRIFT_FEATURES]
    if "ood" in group_set:
        candidates += [(c, "ood", 1.0) for c in OOD_FEATURES]
        candidates += [(c, "ood", -1.0) for c in OOD_INVERSE_FEATURES]

    cols: List[str] = []
    out_groups: List[str] = []
    signs: List[float] = []
    for col, group, sign in candidates:
        if col not in rows.columns or col in cols:
            continue
        vals = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size < max(20, int(0.1 * len(rows))) or float(np.nanstd(finite)) <= 1e-12:
            continue
        cols.append(col)
        out_groups.append(group)
        signs.append(sign)
    return cols, out_groups, np.asarray(signs, dtype=np.float64)


def _fit_feature_state(
    train_rows: pd.DataFrame,
    all_rows: pd.DataFrame,
    groups: Sequence[str],
) -> FeatureState:
    cols, col_groups, signs = _feature_columns(all_rows, groups)
    if not cols:
        return FeatureState([], signs, np.array([], dtype=np.float64), np.array([], dtype=np.float64), [])
    x = train_rows[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    q75 = np.nanpercentile(x, 75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, np.nanstd(x, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    return FeatureState(cols, signs, med, scale, col_groups)


def _transform_feature_groups(rows: pd.DataFrame, state: FeatureState) -> Dict[str, np.ndarray]:
    if not state.columns:
        zeros = np.zeros(len(rows), dtype=np.float32)
        return {"rank": zeros, "uncertainty": zeros, "drift": zeros, "ood": zeros}
    x = rows[state.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x = np.where(np.isfinite(x), x, state.median)
    z = (x - state.median) / state.scale
    z = np.clip(z * state.signs, -6.0, 6.0)
    out: Dict[str, np.ndarray] = {}
    for group in ("rank", "uncertainty", "drift", "ood"):
        idx = [i for i, g in enumerate(state.groups) if g == group]
        if not idx:
            out[group] = np.zeros(len(rows), dtype=np.float32)
        else:
            out[group] = np.nanmean(z[:, idx], axis=1).astype(np.float32, copy=False)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))).astype(np.float32)


def _build_context_score(
    trial: Optional[optuna.Trial],
    group_features: Mapping[str, np.ndarray],
    *,
    arm: str,
    fixed_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    n = len(next(iter(group_features.values()))) if group_features else 0
    if arm == "static":
        return np.full(n, 0.5, dtype=np.float32)

    if arm == "rank_only":
        groups = ("rank",)
    else:
        groups = ("rank", "uncertainty", "drift", "ood")

    linear = np.zeros(n, dtype=np.float32)
    for group in groups:
        if fixed_params is not None:
            weight = float(fixed_params.get(f"w_{group}", 0.0))
        else:
            assert trial is not None
            weight = trial.suggest_float(f"w_{group}", -2.0, 2.0)
        linear = linear + np.float32(weight) * group_features[group]
    if fixed_params is not None:
        intercept = float(fixed_params.get("intercept", 0.0))
    else:
        assert trial is not None
        intercept = trial.suggest_float("intercept", -1.0, 1.0)
    return _sigmoid(linear + np.float32(intercept))


def _row_cost_pct(rows: pd.DataFrame) -> np.ndarray:
    bps = pd.to_numeric(
        rows.get("expected_friction_bps", pd.Series(np.nan, index=rows.index)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    fallback = (
        pd.to_numeric(rows.get("fees_bps", pd.Series(10.0, index=rows.index)), errors="coerce")
        .fillna(10.0)
        .to_numpy(dtype=np.float64)
        + pd.to_numeric(rows.get("entry_reanchor_bps", pd.Series(0.0, index=rows.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        + pd.to_numeric(rows.get("exit_spread_cost_bps", pd.Series(0.0, index=rows.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    bps = np.where(np.isfinite(bps) & (bps >= 0.0), bps, fallback)
    return np.maximum(bps, 0.0).astype(np.float32) / np.float32(10000.0)


def _simulate_tp_sl(
    rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    sl_mult: np.ndarray,
    tp_mult: np.ndarray,
) -> Dict[str, Any]:
    opens, highs, lows, closes = paths
    n, max_bars = opens.shape
    if n == 0:
        return {"net_pnl": 0.0, "n_trades": 0, "raw_gains": np.array([], dtype=np.float32)}

    entry = opens[:, 0].astype(np.float32, copy=False)
    side = rows["side"].to_numpy(dtype=np.float32, copy=False)
    barrier = pd.to_numeric(rows.get("policy_effective_barrier_pct", rows["barrier_pct"]), errors="coerce").to_numpy(dtype=np.float32)
    raw_barrier = pd.to_numeric(rows["barrier_pct"], errors="coerce").to_numpy(dtype=np.float32)
    barrier = np.where(np.isfinite(barrier) & (barrier > 0.0), barrier, raw_barrier)
    barrier = np.maximum(np.where(np.isfinite(barrier), barrier, 0.01), np.float32(1e-4))
    sl_ret = np.maximum(sl_mult.astype(np.float32) * barrier, np.float32(1e-4))
    tp_ret = np.maximum(tp_mult.astype(np.float32) * barrier, np.float32(1e-4))

    long_mask = side >= 0.0
    high_ret = np.where(
        long_mask[:, None],
        highs / np.maximum(entry[:, None], 1e-12) - 1.0,
        entry[:, None] / np.maximum(lows, 1e-12) - 1.0,
    )
    low_ret = np.where(
        long_mask[:, None],
        lows / np.maximum(entry[:, None], 1e-12) - 1.0,
        entry[:, None] / np.maximum(highs, 1e-12) - 1.0,
    )
    close_ret = np.where(
        long_mask[:, None],
        closes / np.maximum(entry[:, None], 1e-12) - 1.0,
        entry[:, None] / np.maximum(closes, 1e-12) - 1.0,
    )
    tp_hit = high_ret >= tp_ret[:, None]
    sl_hit = low_ret <= -sl_ret[:, None]
    any_hit = tp_hit | sl_hit
    first_hit = np.argmax(any_hit, axis=1)
    has_hit = np.any(any_hit, axis=1)
    row_idx = np.arange(n)
    hit_sl = np.zeros(n, dtype=bool)
    hit_tp = np.zeros(n, dtype=bool)
    same_bar_conflict = np.zeros(n, dtype=bool)
    hit_sl[has_hit] = sl_hit[row_idx[has_hit], first_hit[has_hit]]
    hit_tp[has_hit] = tp_hit[row_idx[has_hit], first_hit[has_hit]] & ~hit_sl[has_hit]
    same_bar_conflict[has_hit] = (
        sl_hit[row_idx[has_hit], first_hit[has_hit]]
        & tp_hit[row_idx[has_hit], first_hit[has_hit]]
    )
    last_valid = np.sum(np.isfinite(close_ret), axis=1) - 1
    last_valid = np.maximum(last_valid, 0)
    exit_bar = np.where(has_hit, first_hit, last_valid).astype(np.int32)
    gross = close_ret[row_idx, last_valid].astype(np.float32)
    gross = np.where(hit_tp, tp_ret, gross)
    gross = np.where(hit_sl, -sl_ret, gross)
    cost = _row_cost_pct(rows)
    net = gross - cost
    ts = pd.to_datetime(rows["timestamp"], utc=True)
    weeks = ts.dt.to_period("W").astype(str).to_numpy()
    days = ts.dt.date.astype(str).to_numpy()
    exit_price = entry * (1.0 + side * gross)
    exit_reason = np.where(hit_sl, "full_sl", np.where(hit_tp, "hard_tp", "timeout"))
    return {
        "net_pnl": float(np.nansum(net)),
        "gross_pnl": float(np.nansum(gross)),
        "cost_pnl": float(np.nansum(cost)),
        "mean_net_trade": float(np.nanmean(net)) if net.size else 0.0,
        "win_rate": float(np.nanmean(net > 0.0)) if net.size else 0.0,
        "n_trades": int(n),
        "tp_rate": float(np.nanmean(hit_tp)) if n else 0.0,
        "sl_rate": float(np.nanmean(hit_sl)) if n else 0.0,
        "timeout_rate": float(np.nanmean(~has_hit)) if n else 0.0,
        "same_bar_conflict_rate": float(np.nanmean(same_bar_conflict)) if n else 0.0,
        "raw_gains": net.astype(np.float32),
        "gross_returns": gross.astype(np.float32),
        "net_returns": net.astype(np.float32),
        "exit_bar": exit_bar,
        "exit_price": exit_price.astype(np.float32),
        "exit_reason": exit_reason,
        "same_bar_conflict": same_bar_conflict,
        "weeks": weeks,
        "days": days,
    }


def _period_stats(metrics: Mapping[str, Any]) -> Dict[str, float]:
    gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
    weeks = np.asarray(metrics.get("weeks", []))
    days = np.asarray(metrics.get("days", []))
    if gains.size == 0 or weeks.size == 0:
        return {
            "week_count": 0,
            "avg_week_pnl": 0.0,
            "q35_day_pnl": 0.0,
            "q20_day_pnl": 0.0,
            "min_week_pnl": 0.0,
            "positive_week_rate": 0.0,
        }
    week_vals = []
    for week in pd.unique(weeks):
        week_vals.append(float(np.nansum(gains[weeks == week])))
    week_arr = np.asarray(week_vals, dtype=np.float64)
    day_vals = []
    if days.size == gains.size:
        for day in pd.unique(days):
            day_vals.append(float(np.nansum(gains[days == day])))
    day_arr = np.asarray(day_vals, dtype=np.float64)
    return {
        "week_count": int(week_arr.size),
        "day_count": int(day_arr.size),
        "avg_week_pnl": float(np.nanmean(week_arr)),
        "q35_day_pnl": float(np.nanpercentile(day_arr, 35)) if day_arr.size else 0.0,
        "q20_day_pnl": float(np.nanpercentile(day_arr, 20)) if day_arr.size else 0.0,
        "min_week_pnl": float(np.nanmin(week_arr)),
        "positive_week_rate": float(np.nanmean(week_arr > 0.0)),
    }


def _objective(metrics: Mapping[str, Any]) -> float:
    stats = _period_stats(metrics)
    return float(
        stats["avg_week_pnl"]
        + 0.5 * stats["q35_day_pnl"]
        + 0.2 * stats["q20_day_pnl"]
    )


def _params_to_multipliers(
    params: Mapping[str, float],
    context_score: np.ndarray,
    *,
    arm: str,
) -> Tuple[np.ndarray, np.ndarray]:
    base_sl = float(params.get("base_sl_mult", 1.0))
    base_tp = float(params.get("base_tp_mult", 2.0))
    centered = context_score.astype(np.float32) - np.float32(0.5)
    if arm == "static":
        sl = np.full_like(centered, base_sl, dtype=np.float32)
        tp = np.full_like(centered, base_tp, dtype=np.float32)
    elif arm == "joint_all":
        strength = float(params.get("joint_strength", 0.0))
        mod = np.exp(np.clip(strength * centered, -1.5, 1.5)).astype(np.float32)
        sl = base_sl * mod
        tp = base_tp * mod
    else:
        sl_strength = float(params.get("sl_strength", 0.0))
        tp_strength = float(params.get("tp_strength", 0.0))
        sl = base_sl * np.exp(np.clip(sl_strength * centered, -1.5, 1.5)).astype(np.float32)
        tp = base_tp * np.exp(np.clip(tp_strength * centered, -1.5, 1.5)).astype(np.float32)
    sl = np.clip(sl, np.float32(0.25), np.float32(5.0))
    tp = np.clip(tp, np.float32(0.25), np.float32(8.0))
    return sl.astype(np.float32), tp.astype(np.float32)


def _suggest_params(trial: optuna.Trial, *, arm: str) -> Dict[str, float]:
    params = {
        "base_sl_mult": trial.suggest_float("base_sl_mult", 0.4, 3.5),
        "base_tp_mult": trial.suggest_float("base_tp_mult", 0.4, 6.0),
    }
    if arm == "joint_all":
        params["joint_strength"] = trial.suggest_float("joint_strength", -3.0, 3.0)
    elif arm in {"independent_all", "rank_only"}:
        params["sl_strength"] = trial.suggest_float("sl_strength", -3.0, 3.0)
        params["tp_strength"] = trial.suggest_float("tp_strength", -3.0, 3.0)
    return params


def _evaluate_arm(
    rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    group_features: Mapping[str, np.ndarray],
    *,
    arm: str,
    params: Mapping[str, float],
) -> Dict[str, Any]:
    context = _build_context_score(None, group_features, arm=arm, fixed_params=params)
    sl, tp = _params_to_multipliers(params, context, arm=arm)
    metrics = _simulate_tp_sl(rows, paths, sl_mult=sl, tp_mult=tp)
    out = {k: v for k, v in metrics.items() if k not in {"raw_gains", "weeks", "days"}}
    out.update(_period_stats(metrics))
    out["objective"] = _objective(metrics)
    out["context_score_mean"] = float(np.nanmean(context)) if context.size else 0.5
    out["context_score_p90"] = float(np.nanpercentile(context, 90)) if context.size else 0.5
    return out


def _optimise_arm(
    train_rows: pd.DataFrame,
    train_paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    train_features: Mapping[str, np.ndarray],
    *,
    arm: str,
    n_trials: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, arm=arm)
        if arm != "static":
            _build_context_score(trial, train_features, arm=arm)
            for key, value in trial.params.items():
                params.setdefault(key, value)
        metrics = _evaluate_arm(train_rows, train_paths, train_features, arm=arm, params=params)
        value = float(metrics["objective"])
        trial.set_user_attr("metrics", metrics)
        return value if np.isfinite(value) else -1e9

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_trial
    params = dict(best.params)
    params.update({k: float(v) for k, v in params.items()})
    metrics = dict(best.user_attrs.get("metrics", {}))
    return params, {
        "trials": int(len(study.trials)),
        "best_trial": int(best.number),
        "best_train_objective": float(best.value),
        "best_train_metrics": metrics,
    }


def _tail_metrics(
    rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    features: Mapping[str, np.ndarray],
    *,
    arm: str,
    params: Mapping[str, float],
    cutoffs: Sequence[float],
) -> Dict[str, Dict[str, Any]]:
    rank = pd.to_numeric(rows["rank_pct"], errors="coerce").to_numpy(dtype=np.float64)
    out: Dict[str, Dict[str, Any]] = {}
    for cutoff in cutoffs:
        mask = np.isfinite(rank) & (rank >= float(cutoff))
        key = f"rank_ge_{cutoff:.2f}"
        if not np.any(mask):
            out[key] = {"n_trades": 0, "net_pnl": 0.0, "objective": 0.0}
            continue
        idx = np.flatnonzero(mask).astype(np.int64)
        sub_features = {k: v[idx] for k, v in features.items()}
        out[key] = _evaluate_arm(
            rows.iloc[idx].copy().reset_index(drop=True),
            _path_take(paths, idx),
            sub_features,
            arm=arm,
            params=params,
        )
    return out


def _weekly_table(
    rows: pd.DataFrame,
    metrics_by_arm: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    records = []
    weeks = pd.to_datetime(rows["timestamp"], utc=True).dt.to_period("W").astype(str).to_numpy()
    for arm, payload in metrics_by_arm.items():
        gains = np.asarray(payload.get("raw_gains", []), dtype=np.float64)
        if gains.size != len(rows):
            continue
        for week in pd.unique(weeks):
            mask = weeks == week
            wg = gains[mask]
            records.append(
                {
                    "arm": arm,
                    "week": week,
                    "n_trades": int(mask.sum()),
                    "net_pnl": float(np.nansum(wg)),
                    "mean_net_trade": float(np.nanmean(wg)) if wg.size else 0.0,
                    "win_rate": float(np.nanmean(wg > 0.0)) if wg.size else 0.0,
                }
            )
    return pd.DataFrame(records)


def _head_name(strategy_id: str) -> str:
    if str(strategy_id).startswith("short_bollinger"):
        return "short_bollinger"
    parts = str(strategy_id).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(strategy_id)


def _portfolio_candidate_table(
    rows: pd.DataFrame,
    paths: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    features: Mapping[str, np.ndarray],
    *,
    arm: str,
    params: Mapping[str, float],
) -> pd.DataFrame:
    context = _build_context_score(None, features, arm=arm, fixed_params=params)
    sl, tp = _params_to_multipliers(params, context, arm=arm)
    metrics = _simulate_tp_sl(rows, paths, sl_mult=sl, tp_mult=tp)
    out = rows.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    exit_bar = np.asarray(metrics["exit_bar"], dtype=np.int64)
    out["exit_timestamp"] = ts + pd.to_timedelta(exit_bar * DEFAULT_BAR_MINUTES, unit="m")
    out["entry_price"] = np.asarray(paths[0], dtype=np.float32)[:, 0]
    out["policy_executable_entry_price"] = out["entry_price"]
    out["exit_price"] = np.asarray(metrics["exit_price"], dtype=np.float32)
    out["gross_return"] = np.asarray(metrics["gross_returns"], dtype=np.float32)
    out["net_return"] = np.asarray(metrics["net_returns"], dtype=np.float32)
    out["net_return_before_spread"] = out["net_return"]
    out["net_return_before_legacy_entry_spread_haircut"] = out["net_return"]
    out["holding_bars"] = np.maximum(exit_bar, 1)
    out["simple_policy_exit_reason"] = np.asarray(metrics["exit_reason"], dtype=object)
    out["contextual_tp_sl_arm"] = str(arm)
    out["contextual_tp_sl_score"] = context
    out["policy_sl_mult"] = sl
    out["policy_hard_tp_abs_pct"] = tp * pd.to_numeric(
        out.get("policy_effective_barrier_pct", out["barrier_pct"]),
        errors="coerce",
    ).fillna(pd.to_numeric(out["barrier_pct"], errors="coerce")).to_numpy(dtype=np.float32)
    out["contextual_tp_mult"] = tp
    out["contextual_same_bar_conflict"] = np.asarray(metrics["same_bar_conflict"], dtype=bool)
    if "base_strategy_threshold" not in out.columns:
        out["base_strategy_threshold"] = 0.70
    if "calibrated_score" not in out.columns:
        out["calibrated_score"] = pd.to_numeric(out["rank_pct"], errors="coerce")
    if "normalized_rank_score" not in out.columns:
        out["normalized_rank_score"] = pd.to_numeric(out["rank_pct"], errors="coerce")
    if "strategy_rank_pct" not in out.columns:
        out["strategy_rank_pct"] = pd.to_numeric(out["rank_pct"], errors="coerce")
    return out


def _summarise_accepted(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "accepted" not in decisions.columns:
        return pd.DataFrame()
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["week"] = pd.to_datetime(
        accepted["timestamp"], utc=True, errors="coerce"
    ).dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    rows: List[Dict[str, Any]] = []
    for keys, group in accepted.groupby(["week", "head"], dropna=False, sort=True):
        week, head = keys
        net = pd.to_numeric(group["position_net_return"], errors="coerce")
        gross = pd.to_numeric(group["position_gross_return"], errors="coerce")
        size = pd.to_numeric(group["position_size"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "week": week,
                "head": head,
                "accepted_trades": int(len(group)),
                "net_pnl": float((size * net.fillna(0.0)).sum()),
                "gross_pnl": float((size * gross.fillna(0.0)).sum()),
                "mean_net_return": float(net.mean()) if len(net) else 0.0,
                "hit_rate": float((net > 0.0).mean()) if len(net) else 0.0,
            }
        )
    global_rows: List[Dict[str, Any]] = []
    for week, group in accepted.groupby("week", dropna=False, sort=True):
        net = pd.to_numeric(group["position_net_return"], errors="coerce")
        gross = pd.to_numeric(group["position_gross_return"], errors="coerce")
        size = pd.to_numeric(group["position_size"], errors="coerce").fillna(0.0)
        global_rows.append(
            {
                "week": week,
                "head": "GLOBAL",
                "accepted_trades": int(len(group)),
                "net_pnl": float((size * net.fillna(0.0)).sum()),
                "gross_pnl": float((size * gross.fillna(0.0)).sum()),
                "mean_net_return": float(net.mean()) if len(net) else 0.0,
                "hit_rate": float((net > 0.0).mean()) if len(net) else 0.0,
            }
        )
    return pd.DataFrame(global_rows + rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data_perp/artifacts/20260629_050000_lgbm_mda/simple_policy_optimiser/simple_policy_candidates_broad.parquet"),
    )
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-rank", type=float, default=0.70)
    parser.add_argument("--validation-frac", type=float, default=0.30)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--strategy-ids", default="")
    parser.add_argument("--portfolio-replay", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _prepare_rows(args.candidates, min_rank=float(args.min_rank))
    if args.strategy_ids.strip():
        allowed = {s.strip() for s in args.strategy_ids.split(",") if s.strip()}
        rows = rows.loc[rows["strategy_id"].isin(allowed)].copy()
    if rows.empty:
        raise ValueError("No candidate rows after filtering")

    store = _make_policy_replay_store(args.data_root, args.market_mode)
    arms = {
        "static": ("rank",),
        "rank_only": ("rank",),
        "joint_all": ("rank", "uncertainty", "drift", "ood"),
        "independent_all": ("rank", "uncertainty", "drift", "ood"),
    }
    payload: Dict[str, Any] = {
        "generated_by": "ablate_contextual_tp_sl",
        "candidate_path": str(args.candidates),
        "data_root": str(args.data_root),
        "market_mode": str(args.market_mode),
        "min_rank": float(args.min_rank),
        "validation_frac": float(args.validation_frac),
        "n_trials": int(args.n_trials),
        "seed": int(args.seed),
        "arms": list(arms),
        "strategies": {},
    }
    summary_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    weekly_frames: List[pd.DataFrame] = []
    full_candidate_by_arm: Dict[str, List[pd.DataFrame]] = {
        "static": [],
        "rank_only": [],
        "joint_all": [],
        "independent_all": [],
        "best_by_head": [],
    }

    for strategy_idx, (strategy_id, group) in enumerate(rows.groupby("strategy_id", sort=True)):
        group = group.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        paths = _fetch_policy_paths(group, store, path_len=96)
        group, paths = _apply_delayed_entry_execution_model(
            group,
            paths,
            data_root=args.data_root,
            market_mode=args.market_mode,
        )
        finite = _policy_path_finite_mask(paths)
        group = group.loc[finite].reset_index(drop=True)
        paths = _path_take(paths, np.flatnonzero(finite))
        if len(group) < 100:
            payload["strategies"][strategy_id] = {
                "status": "skipped",
                "reason": "too_few_rows",
                "rows": int(len(group)),
            }
            continue

        train_idx, val_idx, cut_ts = _chronological_split(
            group, validation_frac=float(args.validation_frac)
        )
        train_rows = group.iloc[train_idx].copy().reset_index(drop=True)
        val_rows = group.iloc[val_idx].copy().reset_index(drop=True)
        train_paths = _path_take(paths, train_idx)
        val_paths = _path_take(paths, val_idx)
        strategy_payload: Dict[str, Any] = {
            "status": "ok",
            "rows": int(len(group)),
            "train_rows": int(len(train_rows)),
            "validation_rows": int(len(val_rows)),
            "period_start": str(pd.to_datetime(group["timestamp"], utc=True).min()),
            "period_end": str(pd.to_datetime(group["timestamp"], utc=True).max()),
            "validation_start": str(cut_ts),
            "arms": {},
        }
        val_raw_metrics: Dict[str, Dict[str, Any]] = {}
        strategy_full_candidates: Dict[str, pd.DataFrame] = {}

        for arm_idx, (arm, groups) in enumerate(arms.items()):
            state = _fit_feature_state(train_rows, group, groups)
            train_features = _transform_feature_groups(train_rows, state)
            val_features = _transform_feature_groups(val_rows, state)
            full_features = _transform_feature_groups(group, state)
            params, fit_summary = _optimise_arm(
                train_rows,
                train_paths,
                train_features,
                arm=arm,
                n_trials=int(args.n_trials),
                seed=int(args.seed) + 1000 * strategy_idx + 17 * arm_idx,
            )
            val_eval = _evaluate_arm(
                val_rows,
                val_paths,
                val_features,
                arm=arm,
                params=params,
            )
            context = _build_context_score(None, val_features, arm=arm, fixed_params=params)
            sl, tp = _params_to_multipliers(params, context, arm=arm)
            raw = _simulate_tp_sl(val_rows, val_paths, sl_mult=sl, tp_mult=tp)
            val_raw_metrics[arm] = raw
            tails = _tail_metrics(
                val_rows,
                val_paths,
                val_features,
                arm=arm,
                params=params,
                cutoffs=(0.70, 0.80, 0.90, 0.95),
            )
            strategy_payload["arms"][arm] = {
                "feature_columns": state.columns,
                "feature_groups": state.groups,
                "params": params,
                "fit_summary": fit_summary,
                "validation": val_eval,
                "validation_tail": tails,
            }
            if args.portfolio_replay:
                strategy_full_candidates[arm] = _portfolio_candidate_table(
                    group,
                    paths,
                    full_features,
                    arm=arm,
                    params=params,
                )
            rec = {
                "strategy_id": strategy_id,
                "arm": arm,
                "feature_count": int(len(state.columns)),
                **{f"validation_{k}": v for k, v in val_eval.items() if isinstance(v, (int, float))},
                **{f"param_{k}": v for k, v in params.items()},
            }
            for tail_key, tail_val in tails.items():
                for metric in ("net_pnl", "objective", "win_rate", "n_trades", "tp_rate", "sl_rate"):
                    if metric in tail_val:
                        rec[f"{tail_key}_{metric}"] = tail_val[metric]
            summary_rows.append(rec)

        base = strategy_payload["arms"]["static"]["validation"]
        for arm in arms:
            cur = strategy_payload["arms"][arm]["validation"]
            delta = {
                "strategy_id": strategy_id,
                "arm": arm,
                "delta_objective_vs_static": float(cur["objective"] - base["objective"]),
                "delta_net_pnl_vs_static": float(cur["net_pnl"] - base["net_pnl"]),
                "delta_win_rate_vs_static": float(cur["win_rate"] - base["win_rate"]),
                "delta_avg_week_pnl_vs_static": float(cur["avg_week_pnl"] - base["avg_week_pnl"]),
                "delta_q35_day_pnl_vs_static": float(cur["q35_day_pnl"] - base["q35_day_pnl"]),
                "delta_q20_day_pnl_vs_static": float(cur["q20_day_pnl"] - base["q20_day_pnl"]),
            }
            for cutoff in ("rank_ge_0.90", "rank_ge_0.95"):
                cur_tail = strategy_payload["arms"][arm]["validation_tail"][cutoff]
                base_tail = strategy_payload["arms"]["static"]["validation_tail"][cutoff]
                delta[f"delta_{cutoff}_net_pnl_vs_static"] = float(
                    cur_tail["net_pnl"] - base_tail["net_pnl"]
                )
                delta[f"delta_{cutoff}_win_rate_vs_static"] = float(
                    cur_tail["win_rate"] - base_tail["win_rate"]
                )
            delta_rows.append(delta)

        if args.portfolio_replay:
            best_arm = max(
                arms,
                key=lambda a: float(
                    strategy_payload["arms"][a]["validation"].get("objective", -np.inf)
                ),
            )
            strategy_payload["best_validation_arm"] = best_arm
            for arm, frame in strategy_full_candidates.items():
                full_candidate_by_arm[arm].append(frame)
            if best_arm in strategy_full_candidates:
                best_frame = strategy_full_candidates[best_arm].copy()
                best_frame["contextual_tp_sl_selected_arm"] = best_arm
                full_candidate_by_arm["best_by_head"].append(best_frame)

        w = _weekly_table(val_rows, val_raw_metrics)
        if not w.empty:
            w.insert(0, "strategy_id", strategy_id)
            weekly_frames.append(w)
        payload["strategies"][strategy_id] = strategy_payload

    summary = pd.DataFrame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    weekly = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    portfolio_summary_rows: List[Dict[str, Any]] = []
    portfolio_weekly_frames: List[pd.DataFrame] = []
    if args.portfolio_replay:
        portfolio_dir = args.out_dir / "portfolio_replay"
        portfolio_dir.mkdir(parents=True, exist_ok=True)
        for arm, frames in full_candidate_by_arm.items():
            if not frames:
                continue
            arm_candidates = pd.concat(frames, ignore_index=True)
            arm_candidates = arm_candidates.sort_values(
                ["timestamp", "strategy_id", "symbol"]
            ).reset_index(drop=True)
            ev_curve = fit_hierarchical_ev_curves(arm_candidates)
            params = PortfolioPolicyParams(global_threshold_floor=0.0)
            decisions, equity, metrics = replay_candidates(
                arm_candidates,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=args.market_mode,
            )
            arm_candidates.to_parquet(
                portfolio_dir / f"{arm}_contextual_tp_sl_candidates.parquet",
                index=False,
            )
            decisions.to_parquet(portfolio_dir / f"{arm}_decisions.parquet", index=False)
            equity.to_parquet(portfolio_dir / f"{arm}_equity.parquet", index=False)
            weekly_replay = _summarise_accepted(decisions)
            if not weekly_replay.empty:
                weekly_replay.insert(0, "arm", arm)
                portfolio_weekly_frames.append(weekly_replay)
            portfolio_summary_rows.append(
                {
                    "arm": arm,
                    "candidate_rows": int(len(arm_candidates)),
                    "candidate_start": str(
                        pd.to_datetime(arm_candidates["timestamp"], utc=True).min()
                    ),
                    "candidate_end": str(
                        pd.to_datetime(arm_candidates["timestamp"], utc=True).max()
                    ),
                    **{
                        f"portfolio_{k}": v
                        for k, v in metrics.items()
                        if isinstance(v, (int, float, str, bool))
                    },
                }
            )
        portfolio_summary = pd.DataFrame(portfolio_summary_rows)
        portfolio_weekly = (
            pd.concat(portfolio_weekly_frames, ignore_index=True)
            if portfolio_weekly_frames
            else pd.DataFrame()
        )
    else:
        portfolio_summary = pd.DataFrame()
        portfolio_weekly = pd.DataFrame()
    summary.to_csv(args.out_dir / "contextual_tp_sl_summary.csv", index=False)
    deltas.to_csv(args.out_dir / "contextual_tp_sl_deltas.csv", index=False)
    weekly.to_csv(args.out_dir / "contextual_tp_sl_weekly.csv", index=False)
    if args.portfolio_replay:
        portfolio_summary.to_csv(args.out_dir / "portfolio_replay_summary.csv", index=False)
        portfolio_weekly.to_csv(args.out_dir / "portfolio_replay_weekly.csv", index=False)
        payload["portfolio_replay"] = {
            "enabled": True,
            "arms": portfolio_summary_rows,
        }
    else:
        payload["portfolio_replay"] = {"enabled": False}
    (args.out_dir / "contextual_tp_sl_report.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Contextual TP/SL Ablation",
        "",
        f"Candidate path: `{args.candidates}`",
        f"Rows after min-rank filter: {len(rows)}",
        f"Min rank: {float(args.min_rank):.2f}",
        f"Trials per arm/head: {int(args.n_trials)}",
        "Objective: `avg_week_pnl + 0.5 * q35_day_pnl + 0.2 * q20_day_pnl`",
        "",
    ]
    if not summary.empty:
        keep = [
            "strategy_id",
            "arm",
            "feature_count",
            "validation_objective",
            "validation_net_pnl",
            "validation_win_rate",
            "validation_n_trades",
            "validation_avg_week_pnl",
            "validation_q35_day_pnl",
            "validation_q20_day_pnl",
            "rank_ge_0.90_net_pnl",
            "rank_ge_0.90_win_rate",
            "rank_ge_0.95_net_pnl",
            "rank_ge_0.95_win_rate",
        ]
        lines.append(summary[[c for c in keep if c in summary.columns]].to_markdown(index=False))
        lines.extend(["", "## Deltas vs Static", ""])
        keep_delta = [
            "strategy_id",
            "arm",
            "delta_objective_vs_static",
            "delta_net_pnl_vs_static",
            "delta_win_rate_vs_static",
            "delta_q35_day_pnl_vs_static",
            "delta_q20_day_pnl_vs_static",
            "delta_rank_ge_0.90_net_pnl_vs_static",
            "delta_rank_ge_0.95_net_pnl_vs_static",
        ]
        lines.append(deltas[[c for c in keep_delta if c in deltas.columns]].to_markdown(index=False))
    if args.portfolio_replay and not portfolio_summary.empty:
        lines.extend(["", "## Full Portfolio Replay", ""])
        keep_port = [
            "arm",
            "candidate_rows",
            "candidate_start",
            "candidate_end",
            "portfolio_objective",
            "portfolio_net_pnl",
            "portfolio_gross_pnl",
            "portfolio_trade_count",
            "portfolio_mean_net_return_per_trade",
            "portfolio_full_sl_rate",
            "portfolio_timeout_rate",
            "portfolio_max_drawdown",
            "portfolio_worst_week",
            "portfolio_strategy_concentration",
            "portfolio_side_concentration",
        ]
        lines.append(
            portfolio_summary[[c for c in keep_port if c in portfolio_summary.columns]]
            .to_markdown(index=False)
        )
    (args.out_dir / "contextual_tp_sl_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(rows)}), indent=2))


if __name__ == "__main__":
    main()
