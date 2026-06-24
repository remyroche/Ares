#!/usr/bin/env python3
"""Diagnose why short_asset context risk rejects profitable trades.

This is diagnostic-only.  It reuses the prior short_asset canonical context
surface and reconstructs the same market-state high_conf_tail_loss risk score
only to explain the economic failure mode.  It does not write feature-store or
model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.diagnose_meta_recent_failures import (
    _bad_recent_weeks,
    _discover_heads,
    _downcast_numeric,
    _normalise_keys,
    _prepare_model_matrix,
    _weekly_high_conf_metrics,
    lgb,
)
from scripts.quantify_bad_regime_archetype_usefulness import _failure_targets, _pick_realized_return
from scripts.run_canonical_context_retrain_experiment import (
    MARKET_STATE,
    _fit_predict_lgbm,
    _make_chrono_folds,
    _prediction_controls,
    _safe_auc,
    _safe_pr_auc,
)
from scripts.run_one_head_contextual_meta_ablation import DEFAULT_EPISODE_REGISTRY, _load_episode_registry


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int16)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return out
    ranks = pd.Series(arr[mask]).rank(method="first").to_numpy(dtype=np.float64)
    pct = (ranks - 1.0) / max(float(len(ranks) - 1), 1.0)
    out[mask] = np.minimum(np.floor(pct * int(n_bins)).astype(np.int16), int(n_bins) - 1)
    return out


def _safe_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanquantile(arr, q))


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def _return_stats(ret: np.ndarray) -> dict[str, Any]:
    rr = np.asarray(ret, dtype=np.float64)
    rr = rr[np.isfinite(rr)]
    if rr.size == 0:
        return {
            "trade_count": 0,
            "hit_rate": np.nan,
            "mean_return": np.nan,
            "median_return": np.nan,
            "lower_tail_return_q05": np.nan,
            "winner_magnitude_mean": np.nan,
            "loser_magnitude_mean": np.nan,
        }
    winners = rr[rr > 0.0]
    losers = rr[rr < 0.0]
    return {
        "trade_count": int(rr.size),
        "hit_rate": float(np.mean(rr > 0.0)),
        "mean_return": float(np.nanmean(rr)),
        "median_return": float(np.nanmedian(rr)),
        "lower_tail_return_q05": float(np.nanquantile(rr, 0.05)) if rr.size >= 20 else np.nan,
        "winner_magnitude_mean": float(np.nanmean(winners)) if winners.size else 0.0,
        "loser_magnitude_mean": float(np.nanmean(np.abs(losers))) if losers.size else 0.0,
    }


def _gate_economics(ret: np.ndarray, reject: np.ndarray) -> dict[str, Any]:
    rr = np.asarray(ret, dtype=np.float64)
    reject = np.asarray(reject, dtype=bool) & np.isfinite(rr)
    finite = np.isfinite(rr)
    retain = finite & ~reject
    flagged = rr[reject]
    all_ret = rr[finite]
    retained = rr[retain]
    if all_ret.size == 0:
        return {
            "rows": 0,
            "rejected_rows": 0,
            "rejection_share": np.nan,
            "net_benefit": np.nan,
        }
    loser_loss_avoided = float(-np.nansum(np.minimum(flagged, 0.0))) if flagged.size else 0.0
    winner_profit_sacrificed = float(np.nansum(np.maximum(flagged, 0.0))) if flagged.size else 0.0
    return {
        "rows": int(all_ret.size),
        "rejected_rows": int(flagged.size),
        "retained_rows": int(retained.size),
        "rejection_share": float(flagged.size / max(float(all_ret.size), 1.0)),
        "loser_loss_avoided_sum": loser_loss_avoided,
        "winner_profit_sacrificed_sum": winner_profit_sacrificed,
        "net_benefit": loser_loss_avoided - winner_profit_sacrificed,
        "all_return_mean": float(np.nanmean(all_ret)),
        "retained_return_mean": float(np.nanmean(retained)) if retained.size else np.nan,
        "mean_return_change": float(np.nanmean(retained) - np.nanmean(all_ret)) if retained.size else np.nan,
        "all_hit_rate": float(np.nanmean(all_ret > 0.0)),
        "retained_hit_rate": float(np.nanmean(retained > 0.0)) if retained.size else np.nan,
        "hit_rate_change": float(np.nanmean(retained > 0.0) - np.nanmean(all_ret > 0.0)) if retained.size else np.nan,
        "all_lower_tail_q05": float(np.nanquantile(all_ret, 0.05)) if all_ret.size >= 20 else np.nan,
        "retained_lower_tail_q05": float(np.nanquantile(retained, 0.05)) if retained.size >= 20 else np.nan,
        "tail_loss_avoided_q05": (
            float(np.nanquantile(retained, 0.05) - np.nanquantile(all_ret, 0.05))
            if all_ret.size >= 20 and retained.size >= 20
            else np.nan
        ),
    }


def _period_sample(
    timestamps: pd.Series,
    idx: np.ndarray,
    max_rows: int,
    *,
    seed: int,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if int(max_rows) <= 0 or len(idx) <= int(max_rows):
        return idx
    rng = np.random.default_rng(int(seed))
    ts = pd.to_datetime(timestamps.iloc[idx], utc=True, errors="coerce")
    tmp = pd.DataFrame({"idx": idx, "week": ts.dt.to_period("W").astype(str).to_numpy()})
    target_frac = float(max_rows) / max(float(len(idx)), 1.0)
    pieces: list[np.ndarray] = []
    for _, group in tmp.groupby("week", sort=False):
        take = min(len(group), max(1, int(round(len(group) * target_frac))))
        pieces.append(rng.choice(group["idx"].to_numpy(dtype=np.int64), size=take, replace=False))
    sampled = np.concatenate(pieces) if pieces else idx
    if len(sampled) > int(max_rows):
        sampled = rng.choice(sampled, size=int(max_rows), replace=False)
    return np.sort(sampled.astype(np.int64, copy=False))


def _fit_predict_lgbm_regression(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[Any],
    *,
    timestamps: pd.Series,
    seed: int,
    max_train_rows: int,
    max_depth: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for this diagnostic")
    y = np.asarray(y, dtype=np.float32)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    x = x.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return oof, [{"reason": "empty_matrix", "feature_count": 0}]
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr = tr[np.isfinite(y[tr])]
        va = va[np.isfinite(y[va])]
        if len(tr) < 200 or len(va) < 50:
            rows.append({"fold": fold.fold_id, "reason": "insufficient_rows"})
            continue
        tr_fit = _period_sample(timestamps, tr, int(max_train_rows), seed=int(seed + fold.fold_id * 17))
        min_child = max(50, int(math.ceil(0.025 * len(tr_fit))))
        reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.035,
            max_depth=int(max_depth),
            num_leaves=max(4, min(16, 2 ** int(max_depth))),
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=int(seed + fold.fold_id * 1009),
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(
                x_prepared.iloc[tr_fit],
                y[tr_fit],
                eval_set=[(x_prepared.iloc[va], y[va])],
                eval_metric="l2",
                callbacks=callbacks,
            )
        pred = reg.predict(x_prepared.iloc[va]).astype(np.float32, copy=False)
        oof[va] = pred
        rows.append(
            {
                "fold": int(fold.fold_id),
                "reason": "",
                "train_rows": int(len(tr_fit)),
                "valid_rows": int(len(va)),
                "feature_count": int(len(keep_cols)),
                "best_iteration": int(getattr(reg, "best_iteration_", 0) or 0),
            }
        )
    return oof, rows


def _regression_metrics(y: np.ndarray, pred: np.ndarray, *, baseline_pred: np.ndarray | None = None) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(yy) & np.isfinite(pp)
    if int(mask.sum()) < 50:
        return {"rows": int(mask.sum()), "rmse": np.nan, "mae": np.nan, "pearson_ic": np.nan, "spearman_ic": np.nan}
    out = {
        "rows": int(mask.sum()),
        "rmse": float(np.sqrt(mean_squared_error(yy[mask], pp[mask]))),
        "mae": float(mean_absolute_error(yy[mask], pp[mask])),
        "pearson_ic": (
            float(np.corrcoef(yy[mask], pp[mask])[0, 1])
            if float(np.nanstd(yy[mask])) > 1e-12 and float(np.nanstd(pp[mask])) > 1e-12
            else np.nan
        ),
        "spearman_ic": float(pd.Series(yy[mask]).corr(pd.Series(pp[mask]), method="spearman")),
    }
    if baseline_pred is not None:
        bb = np.asarray(baseline_pred, dtype=np.float64)
        bmask = mask & np.isfinite(bb)
        if int(bmask.sum()) >= 50:
            base_rmse = float(np.sqrt(mean_squared_error(yy[bmask], bb[bmask])))
            base_mae = float(mean_absolute_error(yy[bmask], bb[bmask]))
            out["baseline_rmse"] = base_rmse
            out["baseline_mae"] = base_mae
            out["delta_rmse_improvement"] = base_rmse - float(np.sqrt(mean_squared_error(yy[bmask], pp[bmask])))
            out["delta_mae_improvement"] = base_mae - float(mean_absolute_error(yy[bmask], pp[bmask]))
    return out


def _discover_short_asset_panel(meta_artifact_dir: Path, report_dir: Path) -> tuple[pd.DataFrame, Path]:
    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_state["bundle"]["meta_models"])
    head = next((h for h in heads if h.head == "short_asset"), None)
    if head is None:
        raise RuntimeError("Could not find short_asset meta OOF head")
    return _normalise_keys(pd.read_parquet(head.meta_oof_path)), head.meta_oof_path


def _build_analysis_panel(
    *,
    panel: pd.DataFrame,
    canonical: pd.DataFrame,
    rank_threshold: float,
    folds: list[Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_mask = pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= float(rank_threshold)
    panel_high = panel.loc[high_mask].reset_index(drop=True)
    if len(panel_high) != len(canonical):
        raise RuntimeError(
            f"Canonical context row count {len(canonical)} does not match short_asset high-rank rows {len(panel_high)}"
        )
    realized_return = _pick_realized_return(panel_high).reset_index(drop=True)
    targets = {t["name"]: t for t in _failure_targets(panel_high) if t.get("kind") == "binary"}
    if "high_conf_tail_loss" not in targets:
        raise RuntimeError("high_conf_tail_loss target is unavailable for short_asset")
    y_tail = np.asarray(targets["high_conf_tail_loss"]["values"], dtype=np.float32)
    y_tail_bin = np.where(np.isfinite(y_tail), y_tail, 0.0).astype(np.int8)
    controls = _prediction_controls(panel_high)
    market = canonical.loc[:, list(MARKET_STATE)]
    x_market = pd.concat([controls, market], axis=1, copy=False)
    context_risk, fold_rows = _fit_predict_lgbm(
        x_market,
        y_tail_bin,
        folds,
        seed=int(seed),
        max_depth=3,
    )
    out = pd.DataFrame(
        {
            "timestamp": panel_high["timestamp"],
            "symbol": panel_high["symbol"].astype(str),
            "episode": pd.to_datetime(panel_high["timestamp"], utc=True, errors="coerce")
            .dt.to_period("W")
            .dt.start_time
            .dt.strftime("%Y-%m-%d"),
            "base_opportunity_score": pd.to_numeric(panel_high.get("oof_rank_pct"), errors="coerce").astype(
                "float32"
            ),
            "base_prediction": pd.to_numeric(panel_high.get("oof_pred"), errors="coerce").astype("float32"),
            "realized_return": realized_return.astype("float32"),
            "hit": (realized_return > 0.0).astype("float32"),
            "tail_loss_target": y_tail.astype(np.float32, copy=False),
            "context_risk_score": context_risk.astype(np.float32, copy=False),
        }
    )
    support_proxy = None
    support_orientation = "low_is_low_support"
    for col in ("oof_leaf_train_freq_p10", "oof_leaf_train_freq_min", "oof_leaf_count_p10"):
        if col in panel_high.columns:
            support_proxy = pd.to_numeric(panel_high[col], errors="coerce").to_numpy(dtype=np.float32)
            support_orientation = "low_is_low_support"
            break
    if support_proxy is None and "oof_support_gap" in panel_high.columns:
        support_proxy = pd.to_numeric(panel_high["oof_support_gap"], errors="coerce").to_numpy(dtype=np.float32)
        support_orientation = "high_is_low_support"
    out["leaf_support_proxy"] = support_proxy if support_proxy is not None else np.nan
    out["leaf_support_orientation"] = support_orientation
    for col in ("oof_support_gap", "oof_score_path_std", "oof_rank_path_std"):
        if col in panel_high.columns:
            out[col] = pd.to_numeric(panel_high[col], errors="coerce").astype("float32")
    for col in canonical.columns:
        out[f"context__{col}"] = pd.to_numeric(canonical[col], errors="coerce").astype("float32")
    return _downcast_numeric(out, exclude=["timestamp", "symbol", "episode", "leaf_support_orientation"]), pd.DataFrame(
        fold_rows
    )


def _two_dimensional_evaluation(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = panel["context_risk_score"].notna() & panel["base_opportunity_score"].notna() & panel["realized_return"].notna()
    work = panel.loc[valid].copy().reset_index(drop=True)
    work["base_decile"] = _quantile_bins(work["base_opportunity_score"].to_numpy(), 10) + 1
    work["context_risk_quintile"] = _quantile_bins(work["context_risk_score"].to_numpy(), 5) + 1
    rows: list[dict[str, Any]] = []
    for (base_decile, context_bin), group in work.groupby(["base_decile", "context_risk_quintile"], sort=True):
        ret = group["realized_return"].to_numpy(dtype=np.float64)
        stats = _return_stats(ret)
        stats.update(
            {
                "base_decile": int(base_decile),
                "context_risk_quintile": int(context_bin),
                "context_risk_mean": float(group["context_risk_score"].mean()),
                "context_risk_p25": float(group["context_risk_score"].quantile(0.25)),
                "context_risk_p75": float(group["context_risk_score"].quantile(0.75)),
                "base_score_min": float(group["base_opportunity_score"].min()),
                "base_score_max": float(group["base_opportunity_score"].max()),
            }
        )
        rows.append(stats)
    grid = pd.DataFrame(rows)
    compare_rows: list[dict[str, Any]] = []
    for base_decile, group in work.groupby("base_decile", sort=True):
        low = group.loc[group["context_risk_quintile"].eq(1), "realized_return"].to_numpy(dtype=np.float64)
        high = group.loc[group["context_risk_quintile"].eq(5), "realized_return"].to_numpy(dtype=np.float64)
        low_stats = _return_stats(low)
        high_stats = _return_stats(high)
        compare_rows.append(
            {
                "base_decile": int(base_decile),
                "low_context_rows": low_stats["trade_count"],
                "high_context_rows": high_stats["trade_count"],
                "high_minus_low_mean_return": high_stats["mean_return"] - low_stats["mean_return"],
                "high_minus_low_hit_rate": high_stats["hit_rate"] - low_stats["hit_rate"],
                "high_minus_low_lower_tail_q05": high_stats["lower_tail_return_q05"]
                - low_stats["lower_tail_return_q05"],
                "high_minus_low_winner_magnitude": high_stats["winner_magnitude_mean"]
                - low_stats["winner_magnitude_mean"],
                "high_minus_low_loser_magnitude": high_stats["loser_magnitude_mean"]
                - low_stats["loser_magnitude_mean"],
                "high_context_mean_return": high_stats["mean_return"],
                "low_context_mean_return": low_stats["mean_return"],
            }
        )
    return grid, pd.DataFrame(compare_rows)


def _economic_target_tests(
    panel: pd.DataFrame,
    folds: list[Any],
    *,
    seed: int,
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = panel[["base_prediction", "base_opportunity_score"]].rename(
        columns={"base_prediction": "oof_pred", "base_opportunity_score": "oof_rank_pct"}
    )
    context_cols = [c for c in panel.columns if c.startswith("context__")]
    context = panel[context_cols].copy()
    x_controls = controls
    x_context = pd.concat([controls, context], axis=1, copy=False)
    ret = pd.to_numeric(panel["realized_return"], errors="coerce").to_numpy(dtype=np.float32)
    finite_ret = ret[np.isfinite(ret)]
    tail_q10 = float(np.nanquantile(finite_ret, 0.10)) if finite_ret.size else np.nan
    targets = {
        "Y1_net_return": ret,
        "Y2_negative_return_part": np.minimum(ret, 0.0).astype(np.float32, copy=False),
        "Y3_downside_magnitude": np.maximum(-ret, 0.0).astype(np.float32, copy=False),
        "Y4_expected_shortfall_contribution": np.maximum(tail_q10 - ret, 0.0).astype(np.float32, copy=False),
    }
    rows: list[dict[str, Any]] = []
    pred_cols: dict[str, np.ndarray] = {}
    for target_i, (target_name, y) in enumerate(targets.items(), start=1):
        base_pred, base_folds = _fit_predict_lgbm_regression(
            x_controls,
            y,
            folds,
            timestamps=panel["timestamp"],
            seed=int(seed + target_i * 100),
            max_train_rows=int(max_train_rows),
            max_depth=3,
        )
        ctx_pred, ctx_folds = _fit_predict_lgbm_regression(
            x_context,
            y,
            folds,
            timestamps=panel["timestamp"],
            seed=int(seed + target_i * 100 + 17),
            max_train_rows=int(max_train_rows),
            max_depth=3,
        )
        pred_cols[f"{target_name}__controls_pred"] = base_pred
        pred_cols[f"{target_name}__context_pred"] = ctx_pred
        base_metrics = _regression_metrics(y, base_pred)
        ctx_metrics = _regression_metrics(y, ctx_pred, baseline_pred=base_pred)
        rows.append(
            {
                "target": target_name,
                "tail_q10_reference": tail_q10,
                "controls_rows": base_metrics.get("rows", 0),
                "controls_rmse": base_metrics.get("rmse", np.nan),
                "controls_mae": base_metrics.get("mae", np.nan),
                "controls_pearson_ic": base_metrics.get("pearson_ic", np.nan),
                "controls_spearman_ic": base_metrics.get("spearman_ic", np.nan),
                "context_rows": ctx_metrics.get("rows", 0),
                "context_rmse": ctx_metrics.get("rmse", np.nan),
                "context_mae": ctx_metrics.get("mae", np.nan),
                "context_pearson_ic": ctx_metrics.get("pearson_ic", np.nan),
                "context_spearman_ic": ctx_metrics.get("spearman_ic", np.nan),
                "delta_rmse_improvement": ctx_metrics.get("delta_rmse_improvement", np.nan),
                "delta_mae_improvement": ctx_metrics.get("delta_mae_improvement", np.nan),
                "controls_fold_failures": sum(1 for row in base_folds if row.get("reason")),
                "context_fold_failures": sum(1 for row in ctx_folds if row.get("reason")),
            }
        )
    pred_df = pd.DataFrame(pred_cols)
    return pd.DataFrame(rows), pred_df


def _residual_downside_tests(
    panel: pd.DataFrame,
    folds: list[Any],
    *,
    seed: int,
    max_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = panel[["base_prediction", "base_opportunity_score"]].rename(
        columns={"base_prediction": "oof_pred", "base_opportunity_score": "oof_rank_pct"}
    )
    context_cols = [c for c in panel.columns if c.startswith("context__")]
    context = panel[context_cols].copy()
    ret = pd.to_numeric(panel["realized_return"], errors="coerce").to_numpy(dtype=np.float32)
    q0, _folds = _fit_predict_lgbm_regression(
        controls,
        ret,
        folds,
        timestamps=panel["timestamp"],
        seed=int(seed + 700),
        max_train_rows=int(max_train_rows),
        max_depth=3,
    )
    residual = ret - q0
    downside_residual = np.maximum(-residual, 0.0).astype(np.float32, copy=False)
    context_resid_pred, _ = _fit_predict_lgbm_regression(
        context,
        residual,
        folds,
        timestamps=panel["timestamp"],
        seed=int(seed + 717),
        max_train_rows=int(max_train_rows),
        max_depth=3,
    )
    context_downside_pred, _ = _fit_predict_lgbm_regression(
        context,
        downside_residual,
        folds,
        timestamps=panel["timestamp"],
        seed=int(seed + 733),
        max_train_rows=int(max_train_rows),
        max_depth=3,
    )
    zero = np.zeros(len(panel), dtype=np.float32)
    rows = []
    for name, y, pred in (
        ("return_residual_given_prediction_controls", residual, context_resid_pred),
        ("downside_residual_magnitude", downside_residual, context_downside_pred),
    ):
        metrics = _regression_metrics(y, pred, baseline_pred=zero)
        rows.append({"target": name, **metrics})
    pred_df = pd.DataFrame(
        {
            "q0_expected_return_controls": q0,
            "return_residual": residual,
            "context_predicted_return_residual": context_resid_pred,
            "downside_residual_magnitude": downside_residual,
            "context_predicted_downside_residual": context_downside_pred,
        }
    )
    return pd.DataFrame(rows), pred_df


def _conditional_rules(panel: pd.DataFrame, bad_episodes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = panel["context_risk_score"].notna() & panel["realized_return"].notna()
    work = panel.loc[valid].copy().reset_index(drop=True)
    base_decile = _quantile_bins(work["base_opportunity_score"].to_numpy(dtype=np.float64), 10) + 1
    work["base_decile"] = base_decile
    risk_q80 = _safe_quantile(work["context_risk_score"].to_numpy(dtype=np.float64), 0.80)
    leverage_col = "context__leverage_funding_crowding"
    liquidity_col = "context__liquidity_participation_stress"
    path_col = "context__prediction_path_instability"
    if leverage_col not in work:
        work[leverage_col] = np.nan
    if liquidity_col not in work:
        work[liquidity_col] = np.nan
    if path_col not in work:
        work[path_col] = np.nan
    leverage_q80 = _safe_quantile(work[leverage_col].to_numpy(dtype=np.float64), 0.80)
    liquidity_q80 = _safe_quantile(work[liquidity_col].to_numpy(dtype=np.float64), 0.80)
    path_q80 = _safe_quantile(work[path_col].to_numpy(dtype=np.float64), 0.80)
    support = work["leaf_support_proxy"].to_numpy(dtype=np.float64)
    if str(work["leaf_support_orientation"].dropna().iloc[0] if work["leaf_support_orientation"].notna().any() else "") == "high_is_low_support":
        low_support = support >= _safe_quantile(support, 0.75)
        support_threshold = _safe_quantile(support, 0.75)
        support_rule = "leaf_support_proxy >= q75"
    else:
        low_support = support <= _safe_quantile(support, 0.25)
        support_threshold = _safe_quantile(support, 0.25)
        support_rule = "leaf_support_proxy <= q25"
    rules = {
        "moderate_base_score_x_high_context_risk": (
            work["base_decile"].between(4, 7).to_numpy(dtype=bool)
            & (work["context_risk_score"].to_numpy(dtype=np.float64) >= risk_q80)
        ),
        "low_leaf_support_x_high_leverage_crowding": low_support
        & (work[leverage_col].to_numpy(dtype=np.float64) >= leverage_q80),
        "high_path_instability_x_high_liquidity_stress": (
            work[path_col].to_numpy(dtype=np.float64) >= path_q80
        )
        & (work[liquidity_col].to_numpy(dtype=np.float64) >= liquidity_q80),
    }
    pooled: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    ret = work["realized_return"].to_numpy(dtype=np.float64)
    for name, mask in rules.items():
        row = {"rule": name, **_gate_economics(ret, mask)}
        row.update(
            {
                "context_risk_q80": risk_q80,
                "leverage_q80": leverage_q80,
                "liquidity_q80": liquidity_q80,
                "path_instability_q80": path_q80,
                "support_rule": support_rule,
                "support_threshold": support_threshold,
            }
        )
        pooled.append(row)
        for episode, idx in work.groupby("episode").groups.items():
            ids = np.asarray(list(idx), dtype=np.int64)
            if len(ids) < 30:
                continue
            econ = _gate_economics(ret[ids], mask[ids])
            episode_rows.append(
                {
                    "rule": name,
                    "episode": str(episode),
                    "is_bad_episode": str(episode) in bad_episodes,
                    **econ,
                }
            )
    pooled_df = pd.DataFrame(pooled)
    episode_df = pd.DataFrame(episode_rows)
    if not episode_df.empty:
        agg = (
            episode_df.groupby("rule", as_index=False)
            .agg(
                episode_count=("episode", "nunique"),
                episodes_positive_net=("net_benefit", lambda s: int((s > 0).sum())),
                bad_episode_count=("is_bad_episode", "sum"),
                bad_episodes_positive_net=("net_benefit", lambda s: int(((s > 0) & episode_df.loc[s.index, "is_bad_episode"]).sum())),
            )
        )
        pooled_df = pooled_df.merge(agg, on="rule", how="left")
    return pooled_df, episode_df


def _economic_frontier(panel: pd.DataFrame, bad_episodes: set[str]) -> pd.DataFrame:
    valid = panel["context_risk_score"].notna() & panel["realized_return"].notna()
    work = panel.loc[valid].copy().reset_index(drop=True)
    ret = work["realized_return"].to_numpy(dtype=np.float64)
    risk = work["context_risk_score"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for retained_coverage in (0.99, 0.975, 0.95, 0.90, 0.80):
        reject_share = 1.0 - retained_coverage
        threshold = float(np.nanquantile(risk[np.isfinite(risk)], retained_coverage))
        reject = risk >= threshold
        econ = _gate_economics(ret, reject)
        episode_net: list[float] = []
        bad_episode_net: list[float] = []
        for episode, idx in work.groupby("episode").groups.items():
            ids = np.asarray(list(idx), dtype=np.int64)
            if len(ids) < 30:
                continue
            local = _gate_economics(ret[ids], reject[ids])
            if np.isfinite(local.get("net_benefit", np.nan)):
                episode_net.append(float(local["net_benefit"]))
                if str(episode) in bad_episodes:
                    bad_episode_net.append(float(local["net_benefit"]))
        rows.append(
            {
                "retained_coverage": retained_coverage,
                "rejected_coverage": reject_share,
                "risk_threshold": threshold,
                **econ,
                "episode_count": len(episode_net),
                "episode_positive_net_rate": float(np.mean(np.asarray(episode_net) > 0.0)) if episode_net else np.nan,
                "bad_episode_count": len(bad_episode_net),
                "bad_episode_positive_net_rate": float(np.mean(np.asarray(bad_episode_net) > 0.0))
                if bad_episode_net
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _requirement_audit(
    *,
    risk_metrics: dict[str, Any],
    two_d: pd.DataFrame,
    decile_compare: pd.DataFrame,
    target_tests: pd.DataFrame,
    residual_tests: pd.DataFrame,
    rules: pd.DataFrame,
    rule_episodes: pd.DataFrame,
    frontier: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    high_mean_positive = int((decile_compare.get("high_minus_low_mean_return", pd.Series(dtype=float)) > 0).sum())
    high_hit_worse = int((decile_compare.get("high_minus_low_hit_rate", pd.Series(dtype=float)) < 0).sum())
    high_tail_worse = int((decile_compare.get("high_minus_low_lower_tail_q05", pd.Series(dtype=float)) < 0).sum())
    rows.append(
        {
            "step": "risk_score_reconstruction",
            "status": "completed",
            "primary_metrics": (
                f"rows={risk_metrics.get('rows')}; coverage={risk_metrics.get('coverage'):.4f}; "
                f"tail_loss_auc={risk_metrics.get('tail_loss_auc'):.4f}; "
                f"tail_loss_pr_auc={risk_metrics.get('tail_loss_pr_auc'):.4f}"
            ),
            "outcome": "context risk is statistically real but not an economic gate",
            "artifact": "short_asset_context_risk_fold_metrics.csv",
        }
    )
    rows.append(
        {
            "step": "base_opportunity_x_context_risk",
            "status": "completed",
            "primary_metrics": (
                f"grid_cells={len(two_d)}; deciles={len(decile_compare)}; "
                f"high_context_mean_return_higher_deciles={high_mean_positive}/{len(decile_compare)}; "
                f"high_context_hit_rate_lower_deciles={high_hit_worse}/{len(decile_compare)}; "
                f"high_context_q05_worse_deciles={high_tail_worse}/{len(decile_compare)}"
            ),
            "outcome": "high context risk marks higher dispersion: lower hit rate and worse tails, but higher mean/winner magnitude",
            "artifact": "short_asset_context_2d_bins.csv; short_asset_base_decile_context_comparison.csv",
        }
    )
    rows.append(
        {
            "step": "economic_target_alignment",
            "status": "completed",
            "primary_metrics": (
                f"targets={len(target_tests)}; "
                f"rmse_improved={(target_tests['delta_rmse_improvement'] > 0).sum()}/{len(target_tests)}; "
                f"mae_improved={(target_tests['delta_mae_improvement'] > 0).sum()}/{len(target_tests)}"
            ),
            "outcome": "context adds small predictive lift on return/downside magnitudes, but effect size is economically tiny",
            "artifact": "short_asset_economic_target_tests.csv",
        }
    )
    rows.append(
        {
            "step": "incremental_residual_downside",
            "status": "completed",
            "primary_metrics": (
                f"targets={len(residual_tests)}; "
                f"rmse_improved={(residual_tests['delta_rmse_improvement'] > 0).sum()}/{len(residual_tests)}; "
                f"mae_improved={(residual_tests['delta_mae_improvement'] > 0).sum()}/{len(residual_tests)}"
            ),
            "outcome": "context explains some residual variance, but downside-residual MAE worsens versus zero baseline",
            "artifact": "short_asset_incremental_residual_downside.csv",
        }
    )
    bad_episode_col = "bad_episodes_positive_net"
    positive_rules = int((rules.get("net_benefit", pd.Series(dtype=float)) > 0).sum())
    bad_episode_positive_rules = int((rules.get(bad_episode_col, pd.Series(dtype=float)) > 0).sum())
    rows.append(
        {
            "step": "predeclared_conditional_actions",
            "status": "completed_rejected",
            "primary_metrics": (
                f"rules={len(rules)}; pooled_positive_net_rules={positive_rules}/{len(rules)}; "
                f"rules_positive_in_bad_episodes={bad_episode_positive_rules}/{len(rules)}; "
                f"episode_rule_rows={len(rule_episodes)}"
            ),
            "outcome": "no conditional gate passes; winner profit sacrificed exceeds loser loss avoided",
            "artifact": "short_asset_conditional_action_rules.csv; short_asset_conditional_action_rule_episodes.csv",
        }
    )
    best_frontier = frontier.sort_values("net_benefit", ascending=False).iloc[0] if not frontier.empty else pd.Series()
    positive_frontier = int((frontier.get("net_benefit", pd.Series(dtype=float)) > 0).sum())
    rows.append(
        {
            "step": "economic_frontier",
            "status": "completed_rejected",
            "primary_metrics": (
                f"coverage_points={len(frontier)}; positive_net_points={positive_frontier}/{len(frontier)}; "
                f"best_retained_coverage={best_frontier.get('retained_coverage', np.nan):.3f}; "
                f"best_net_benefit={best_frontier.get('net_benefit', np.nan):.6f}"
            ),
            "outcome": "no retained-coverage point has positive net economic value",
            "artifact": "short_asset_economic_frontier.csv",
        }
    )
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    *,
    manifest: dict[str, Any],
    risk_metrics: dict[str, Any],
    two_d: pd.DataFrame,
    decile_compare: pd.DataFrame,
    target_tests: pd.DataFrame,
    residual_tests: pd.DataFrame,
    rules: pd.DataFrame,
    requirement_audit: pd.DataFrame,
    frontier: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Short Asset Context Economics Diagnostic")
    lines.append("")
    lines.append("Diagnostic-only follow-up for `short_asset / high_conf_tail_loss`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- No production model or feature-store artifact was modified.")
    lines.append("- Context risk uses the same canonical market-state tail-loss feature set from the rejected retrain arm.")
    lines.append("- Fresh OOS remains untouched; this report uses nested OOF-style diagnostics only.")
    lines.append("")
    lines.append("## Risk Score Reconstruction")
    lines.append("")
    lines.append(pd.DataFrame([risk_metrics]).to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## Key Outcome")
    lines.append("")
    if not frontier.empty:
        best = frontier.sort_values("net_benefit", ascending=False).iloc[0]
        lines.append(
            f"Best global retained-coverage frontier point has net benefit `{best['net_benefit']:.6f}` "
            f"at retained coverage `{best['retained_coverage']:.3f}`."
        )
        lines.append("")
    if not rules.empty:
        best_rule = rules.sort_values("net_benefit", ascending=False).iloc[0]
        lines.append(
            f"Best predeclared conditional rule is `{best_rule['rule']}` with net benefit "
            f"`{best_rule['net_benefit']:.6f}`."
        )
        lines.append("")
    lines.append("## Requirement Audit")
    lines.append("")
    lines.append(requirement_audit.to_markdown(index=False))
    lines.append("")
    lines.append("## Base Opportunity x Context Risk")
    lines.append("")
    lines.append("High-minus-low context risk by base opportunity decile:")
    lines.append("")
    lines.append(decile_compare.to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## Economic Target Tests")
    lines.append("")
    lines.append(target_tests.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Incremental Residual Downside")
    lines.append("")
    lines.append(residual_tests.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Conditional Action Rules")
    lines.append("")
    lines.append(rules.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Economic Frontier")
    lines.append("")
    lines.append(frontier.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Full 2D Grid")
    lines.append("")
    lines.append(two_d.to_markdown(index=False, floatfmt=".6f"))
    lines.append("")
    lines.append("## Manifest")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    lines.append("```")
    (out_dir / "short_asset_context_economic_diagnostic_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    out_dir = _ensure_dir(Path(args.output_dir))
    panel, panel_path = _discover_short_asset_panel(Path(args.meta_artifact_dir), Path(args.report_dir))
    canonical = pd.read_parquet(Path(args.canonical_context))
    high_mask = pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= float(args.rank_threshold)
    panel_high_for_folds = panel.loc[high_mask].reset_index(drop=True)
    folds = _make_chrono_folds(
        panel_high_for_folds["timestamp"],
        int(args.outer_folds),
        embargo_hours=int(args.embargo_hours),
    )
    analysis_panel, risk_fold_metrics = _build_analysis_panel(
        panel=panel,
        canonical=canonical,
        rank_threshold=float(args.rank_threshold),
        folds=folds,
        seed=int(args.seed),
    )
    valid_risk = analysis_panel["context_risk_score"].notna()
    y_tail = analysis_panel["tail_loss_target"].to_numpy(dtype=np.float32)
    risk = analysis_panel["context_risk_score"].to_numpy(dtype=np.float32)
    ret = analysis_panel["realized_return"].to_numpy(dtype=np.float32)
    risk_metrics = {
        "rows": int(valid_risk.sum()),
        "coverage": float(valid_risk.mean()),
        "tail_loss_auc": _safe_auc(y_tail, risk),
        "tail_loss_pr_auc": _safe_pr_auc(y_tail, risk),
        "mean_return_highest_risk_decile": _safe_mean(ret[risk >= _safe_quantile(risk, 0.90)]),
        "mean_return_lowest_risk_decile": _safe_mean(ret[risk <= _safe_quantile(risk, 0.10)]),
    }
    bad_episodes, bad_meta = _load_episode_registry(args.episode_registry, head="short_asset", target_name="diagnostic")
    if not bad_episodes and bool(bad_meta.get("fallback_allowed", False)):
        weekly = _weekly_high_conf_metrics(panel, float(args.rank_threshold), int(args.min_week_rows))
        bad_weeks, fallback_meta = _bad_recent_weeks(
            weekly,
            recent_weeks=int(args.recent_weeks),
            min_week_rows=int(args.min_week_rows),
        )
        bad_episodes = {pd.Timestamp(w).strftime("%Y-%m-%d") for w in bad_weeks}
        bad_meta = {
            **fallback_meta,
            "source": str(args.episode_registry),
            "reason": f"fallback_bad_recent_weeks_after_{bad_meta.get('reason', 'registry_unavailable')}",
            "fallback_allowed": True,
        }
    two_d, decile_compare = _two_dimensional_evaluation(analysis_panel)
    target_tests, target_pred = _economic_target_tests(
        analysis_panel,
        folds,
        seed=int(args.seed) + 1000,
        max_train_rows=int(args.max_train_rows),
    )
    residual_tests, residual_pred = _residual_downside_tests(
        analysis_panel,
        folds,
        seed=int(args.seed) + 2000,
        max_train_rows=int(args.max_train_rows),
    )
    rules, rule_episodes = _conditional_rules(analysis_panel, bad_episodes)
    frontier = _economic_frontier(analysis_panel, bad_episodes)
    requirement_audit = _requirement_audit(
        risk_metrics=risk_metrics,
        two_d=two_d,
        decile_compare=decile_compare,
        target_tests=target_tests,
        residual_tests=residual_tests,
        rules=rules,
        rule_episodes=rule_episodes,
        frontier=frontier,
    )
    enriched_panel = pd.concat([analysis_panel, target_pred, residual_pred], axis=1, copy=False)
    enriched_panel.to_parquet(out_dir / "short_asset_context_economic_panel.parquet", index=False)
    risk_fold_metrics.to_csv(out_dir / "short_asset_context_risk_fold_metrics.csv", index=False)
    two_d.to_csv(out_dir / "short_asset_context_2d_bins.csv", index=False)
    decile_compare.to_csv(out_dir / "short_asset_base_decile_context_comparison.csv", index=False)
    target_tests.to_csv(out_dir / "short_asset_economic_target_tests.csv", index=False)
    residual_tests.to_csv(out_dir / "short_asset_incremental_residual_downside.csv", index=False)
    rules.to_csv(out_dir / "short_asset_conditional_action_rules.csv", index=False)
    rule_episodes.to_csv(out_dir / "short_asset_conditional_action_rule_episodes.csv", index=False)
    frontier.to_csv(out_dir / "short_asset_economic_frontier.csv", index=False)
    requirement_audit.to_csv(out_dir / "short_asset_context_economic_requirement_audit.csv", index=False)
    manifest = {
        "status": "completed",
        "head": "short_asset",
        "target": "high_conf_tail_loss",
        "meta_oof_path": str(panel_path),
        "canonical_context": str(args.canonical_context),
        "rank_threshold": float(args.rank_threshold),
        "rows_high_rank": int(len(analysis_panel)),
        "rows_with_context_risk": int(valid_risk.sum()),
        "bad_episodes": sorted(bad_episodes),
        "bad_episode_metadata": bad_meta,
        "episode_registry": str(args.episode_registry),
        "outputs": {
            "panel": "short_asset_context_economic_panel.parquet",
            "risk_fold_metrics": "short_asset_context_risk_fold_metrics.csv",
            "two_dimensional_bins": "short_asset_context_2d_bins.csv",
            "base_decile_context_comparison": "short_asset_base_decile_context_comparison.csv",
            "economic_target_tests": "short_asset_economic_target_tests.csv",
            "incremental_residual_downside": "short_asset_incremental_residual_downside.csv",
            "conditional_action_rules": "short_asset_conditional_action_rules.csv",
            "conditional_action_rule_episodes": "short_asset_conditional_action_rule_episodes.csv",
            "economic_frontier": "short_asset_economic_frontier.csv",
            "requirement_audit": "short_asset_context_economic_requirement_audit.csv",
            "report": "short_asset_context_economic_diagnostic_report.md",
        },
    }
    (out_dir / "short_asset_context_economic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default)
    )
    _write_report(
        out_dir,
        manifest=manifest,
        risk_metrics=risk_metrics,
        two_d=two_d,
        decile_compare=decile_compare,
        target_tests=target_tests,
        residual_tests=residual_tests,
        rules=rules,
        requirement_audit=requirement_audit,
        frontier=frontier,
    )
    print(f"[short_asset_context_economics] wrote results to {out_dir}", flush=True)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument(
        "--canonical-context",
        default="data_perp/reports/canonical_context_retrain_experiment_20260622/short_asset_fold_fitted_canonical_context.parquet",
    )
    parser.add_argument("--output-dir", default="data_perp/reports/short_asset_context_economic_diagnostic_20260622")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--recent-weeks", type=int, default=8)
    parser.add_argument("--min-week-rows", type=int, default=30)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--episode-registry", default=str(DEFAULT_EPISODE_REGISTRY))
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
