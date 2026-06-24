#!/usr/bin/env python3
"""Quantify whether bad-regime archetype scores explain meta OOF failures.

This is intentionally diagnostic-only.  It does not write to the feature store
and does not mutate trained model artifacts.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from extreme_price_movements.unsupervised_regime_learning.bad_regime_archetypes import (
    BadRegimeArchetypeFeatureConfig,
    build_bad_regime_archetype_feature_frame,
    load_bad_regime_archetype_definitions,
)

from scripts.diagnose_meta_recent_failures import (
    _adversarial_diagnostics,
    _assemble_selected_matrix,
    _bad_recent_weeks,
    _base_models_for_head,
    _candidate_feature_contract,
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _fit_lgbm_cv,
    _known_export_features,
    _latest_regime_context,
    _merge_feature_candidates,
    _normalise_keys,
    _period_stratified_sample,
    _prepare_model_matrix,
    _read_regime_features,
    _weekly_high_conf_metrics,
    lgb,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _blank_reason_mask(frame: pd.DataFrame, column: str, *, default_blank: bool = True) -> pd.Series:
    """Return True for blank reason fields, including CSV-round-tripped NaN."""
    if column not in frame.columns:
        return pd.Series(bool(default_blank), index=frame.index)
    raw = frame[column]
    return raw.isna() | raw.astype(str).str.strip().eq("")


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(y)
    if int(mask.sum()) < 50:
        return float("nan")
    yy = np.asarray(y[mask], dtype=np.int8)
    if len(np.unique(yy)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, score[mask]))
    except Exception:
        return float("nan")


def _safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(y)
    if int(mask.sum()) < 50:
        return float("nan")
    yy = np.asarray(y[mask], dtype=np.int8)
    if len(np.unique(yy)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(yy, score[mask]))
    except Exception:
        return float("nan")


def _pick_realized_return(panel: pd.DataFrame) -> pd.Series:
    for col in ("net_return", "net_ret", "return", "gross_return", "ret"):
        if col in panel.columns:
            return pd.to_numeric(panel[col], errors="coerce")
    return pd.Series(np.nan, index=panel.index, dtype="float32")


def _classification_and_economic_metrics(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    realized_return: np.ndarray,
    prefix: str = "",
) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(pred)
    yy = np.asarray(y[mask], dtype=np.int8)
    pp = np.clip(np.asarray(pred[mask], dtype=np.float64), 1e-6, 1.0 - 1e-6)
    rr = np.asarray(realized_return[mask], dtype=np.float64)
    out: dict[str, Any] = {
        f"{prefix}roc_auc": _safe_auc(yy.astype(np.float32), pp.astype(np.float32)),
        f"{prefix}pr_auc": _safe_pr_auc(yy.astype(np.float32), pp.astype(np.float32)),
        f"{prefix}log_loss": float(log_loss(yy, pp, labels=[0, 1])) if len(np.unique(yy)) >= 2 else np.nan,
        f"{prefix}brier": float(brier_score_loss(yy, pp)) if len(np.unique(yy)) >= 2 else np.nan,
    }
    total_failures = max(float(np.sum(yy > 0)), 1.0)
    finite_ret = np.isfinite(rr)
    all_ret_mean = float(np.nanmean(rr[finite_ret])) if finite_ret.any() else np.nan
    all_tail = (
        float(np.nanmean(rr[finite_ret & (rr <= np.nanquantile(rr[finite_ret], 0.05))]))
        if finite_ret.sum() >= 50
        else np.nan
    )
    for pct in (0.05, 0.10, 0.20):
        n_reject = max(1, int(math.ceil(pct * len(pp))))
        order = np.argsort(pp)[::-1]
        reject = np.zeros(len(pp), dtype=bool)
        reject[order[:n_reject]] = True
        retain = ~reject
        suffix = f"at_{int(pct * 100)}pct_abstain"
        out[f"{prefix}failure_capture_{suffix}"] = float(np.sum(yy[reject] > 0) / total_failures)
        out[f"{prefix}rejected_failure_rate_{suffix}"] = float(np.mean(yy[reject])) if reject.any() else np.nan
        out[f"{prefix}retained_failure_rate_{suffix}"] = float(np.mean(yy[retain])) if retain.any() else np.nan
        ret_retain = rr[retain & np.isfinite(rr)]
        ret_reject = rr[reject & np.isfinite(rr)]
        out[f"{prefix}retained_return_mean_{suffix}"] = float(np.nanmean(ret_retain)) if ret_retain.size else np.nan
        out[f"{prefix}all_return_mean"] = all_ret_mean
        if ret_retain.size >= 50 and np.isfinite(all_tail):
            retained_tail = float(np.nanmean(ret_retain[ret_retain <= np.nanquantile(ret_retain, 0.05)]))
            out[f"{prefix}tail_loss_avoided_{suffix}"] = retained_tail - all_tail
        else:
            out[f"{prefix}tail_loss_avoided_{suffix}"] = np.nan
        rejected_winners = ret_reject[ret_reject > 0]
        out[f"{prefix}rejected_winner_cost_{suffix}"] = (
            float(np.nanmean(rejected_winners)) if rejected_winners.size else 0.0
        )
        out[f"{prefix}rejected_winner_share_{suffix}"] = (
            float(rejected_winners.size / max(float(reject.sum()), 1.0)) if reject.any() else np.nan
        )
    return out


def _rank_to_unit(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return out
    ranks = pd.Series(arr[mask]).rank(method="average").to_numpy(dtype=np.float64)
    denom = max(float(len(ranks) - 1), 1.0)
    out[mask] = ((ranks - 1.0) / denom).astype(np.float32, copy=False)
    return out


def _continuous_utility_metrics(
    *,
    y: np.ndarray,
    pred: np.ndarray,
    prefix: str = "",
) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(yy) & np.isfinite(pp)
    if int(mask.sum()) < 50:
        return {
            f"{prefix}rmse": np.nan,
            f"{prefix}mae": np.nan,
            f"{prefix}pearson_ic": np.nan,
            f"{prefix}spearman_ic": np.nan,
        }
    err = pp[mask] - yy[mask]
    out: dict[str, Any] = {
        f"{prefix}rmse": float(np.sqrt(np.nanmean(err * err))),
        f"{prefix}mae": float(np.nanmean(np.abs(err))),
        f"{prefix}pearson_ic": float(np.corrcoef(pp[mask], yy[mask])[0, 1])
        if int(mask.sum()) >= 3 and float(np.nanstd(pp[mask])) > 1e-12 and float(np.nanstd(yy[mask])) > 1e-12
        else np.nan,
        f"{prefix}spearman_ic": float(pd.Series(pp[mask]).corr(pd.Series(yy[mask]), method="spearman")),
    }
    risk_score = _rank_to_unit(-pp)
    y_loss = (yy < 0.0).astype(np.int8)
    out.update(
        _classification_and_economic_metrics(
            y=y_loss.astype(np.float32),
            pred=risk_score,
            realized_return=yy,
            prefix=f"{prefix}risk_gate_",
        )
    )
    return out


def _period_sample_indices(
    frame: pd.DataFrame,
    max_rows: int,
    *,
    seed: int = 7,
    period: str = "W",
) -> np.ndarray:
    n = len(frame)
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    tmp = pd.DataFrame(
        {
            "idx": np.arange(n, dtype=np.int64),
            "period": pd.to_datetime(frame["timestamp"], utc=True).dt.to_period(period).astype(str).to_numpy(),
        }
    )
    samples: list[np.ndarray] = []
    target_frac = max_rows / max(n, 1)
    for _, group in tmp.groupby("period", sort=False):
        take = max(1, int(round(len(group) * target_frac)))
        take = min(take, len(group))
        samples.append(rng.choice(group["idx"].to_numpy(), size=take, replace=False))
    idx = np.concatenate(samples) if samples else np.arange(n, dtype=np.int64)
    if idx.size > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _fit_lgbm_cv_detailed(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    realized_return: pd.Series,
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for these diagnostics")
    y_raw = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(y_raw)
    x = x.loc[valid].reset_index(drop=True)
    ts = pd.to_datetime(timestamps.loc[valid], utc=True, errors="coerce").reset_index(drop=True)
    ret = pd.to_numeric(realized_return.loc[valid], errors="coerce").reset_index(drop=True)
    y = (y_raw[valid] > 0.5).astype(np.int8, copy=False)
    if len(np.unique(y)) < 2 or len(y) < 200:
        return {
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "pr_auc": np.nan,
            "log_loss": np.nan,
            "brier": np.nan,
            "folds": 0,
            "rows": int(len(y)),
            "positive_rate": float(np.mean(y)) if len(y) else np.nan,
            "reason": "insufficient_classes_or_rows",
        }

    frame = pd.DataFrame({"timestamp": ts})
    sample_idx = _period_stratified_sample(frame, y, max_rows=max_rows, seed=seed)
    x = x.iloc[sample_idx].reset_index(drop=True)
    y = y[sample_idx]
    ts = ts.iloc[sample_idx].reset_index(drop=True)
    ret = ret.iloc[sample_idx].reset_index(drop=True)
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    x = _prepare_model_matrix(x.iloc[order].reset_index(drop=True))
    y = y[order]
    ts = ts.iloc[order].reset_index(drop=True)
    ret = ret.iloc[order].reset_index(drop=True)

    n_splits = min(5, max(2, len(y) // 5000))
    splitter = TimeSeriesSplit(n_splits=n_splits).split(x)
    aucs: list[float] = []
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(splitter):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        min_child = max(50, int(math.ceil(0.025 * len(train_idx))))
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=8,
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(
                x.iloc[train_idx],
                y[train_idx],
                eval_set=[(x.iloc[test_idx], y[test_idx])],
                eval_metric="auc",
                callbacks=callbacks,
            )
        pred = clf.predict_proba(x.iloc[test_idx])[:, 1]
        oof[test_idx] = pred.astype(np.float32, copy=False)
        aucs.append(float(roc_auc_score(y[test_idx], pred)))

    if not aucs:
        return {
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "pr_auc": np.nan,
            "log_loss": np.nan,
            "brier": np.nan,
            "folds": 0,
            "rows": int(len(y)),
            "positive_rate": float(np.mean(y)) if len(y) else np.nan,
            "reason": "no_valid_cv_folds",
        }
    metrics = _classification_and_economic_metrics(
        y=y.astype(np.float32),
        pred=oof,
        realized_return=ret.to_numpy(dtype=np.float64, copy=False),
    )
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "folds": int(len(aucs)),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "reason": "",
        **metrics,
    }


def _fit_lgbm_cv_regression_detailed(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for these diagnostics")
    y_raw = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(y_raw)
    x = x.loc[valid].reset_index(drop=True)
    ts = pd.to_datetime(timestamps.loc[valid], utc=True, errors="coerce").reset_index(drop=True)
    yv = y_raw[valid].astype(np.float32, copy=False)
    if len(yv) < 200 or float(np.nanstd(yv)) <= 1e-12:
        return {
            "folds": 0,
            "rows": int(len(yv)),
            "target_mean": float(np.nanmean(yv)) if len(yv) else np.nan,
            "target_std": float(np.nanstd(yv)) if len(yv) else np.nan,
            "reason": "insufficient_rows_or_target_variance",
        }

    sample_idx = _period_sample_indices(pd.DataFrame({"timestamp": ts}), max_rows=max_rows, seed=seed)
    x = x.iloc[sample_idx].reset_index(drop=True)
    yv = yv[sample_idx]
    ts = ts.iloc[sample_idx].reset_index(drop=True)
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    x = _prepare_model_matrix(x.iloc[order].reset_index(drop=True))
    yv = yv[order]

    n_splits = min(5, max(2, len(yv) // 5000))
    splitter = TimeSeriesSplit(n_splits=n_splits).split(x)
    oof = np.full(len(yv), np.nan, dtype=np.float32)
    fold_count = 0
    for fold, (train_idx, test_idx) in enumerate(splitter):
        if len(train_idx) < 100 or len(test_idx) < 50 or float(np.nanstd(yv[train_idx])) <= 1e-12:
            continue
        min_child = max(50, int(math.ceil(0.025 * len(train_idx))))
        reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=500,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=8,
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(
                x.iloc[train_idx],
                yv[train_idx],
                eval_set=[(x.iloc[test_idx], yv[test_idx])],
                eval_metric="l2",
                callbacks=callbacks,
            )
        oof[test_idx] = reg.predict(x.iloc[test_idx]).astype(np.float32, copy=False)
        fold_count += 1

    if fold_count == 0:
        return {
            "folds": 0,
            "rows": int(len(yv)),
            "target_mean": float(np.nanmean(yv)) if len(yv) else np.nan,
            "target_std": float(np.nanstd(yv)) if len(yv) else np.nan,
            "reason": "no_valid_cv_folds",
        }
    return {
        "folds": int(fold_count),
        "rows": int(len(yv)),
        "target_mean": float(np.nanmean(yv)),
        "target_std": float(np.nanstd(yv)),
        "reason": "",
        **_continuous_utility_metrics(y=yv, pred=oof, prefix="utility_"),
    }


def _failure_targets(panel_high: pd.DataFrame) -> list[dict[str, Any]]:
    y_bin = pd.to_numeric(panel_high.get("y_bin", pd.Series(np.nan, index=panel_high.index)), errors="coerce")
    pred = pd.to_numeric(panel_high.get("oof_pred", pd.Series(np.nan, index=panel_high.index)), errors="coerce")
    realized_return = _pick_realized_return(panel_high)
    targets: list[dict[str, Any]] = []

    if y_bin.notna().sum() >= 50:
        y_values = np.full(len(panel_high), np.nan, dtype=np.float32)
        y_mask = y_bin.notna().to_numpy()
        y_values[y_mask] = (y_bin.to_numpy(dtype=np.float32, copy=False)[y_mask] <= 0.0).astype(np.float32)
        targets.append(
            {
                "name": "high_conf_miss",
                "kind": "binary",
                "values": y_values,
                "definition": "y_bin <= 0 inside rank-filtered rows",
            }
        )
    finite_ret = realized_return[np.isfinite(realized_return)]
    if len(finite_ret) >= 50:
        ret_values = realized_return.to_numpy(dtype=np.float32, copy=False)
        ret_mask = np.isfinite(ret_values)
        negative_values = np.full(len(panel_high), np.nan, dtype=np.float32)
        negative_values[ret_mask] = (ret_values[ret_mask] < 0.0).astype(np.float32)
        targets.append(
            {
                "name": "high_conf_negative_net_pnl",
                "kind": "binary",
                "values": negative_values,
                "definition": "realized return < 0 inside rank-filtered rows",
            }
        )
        tail_threshold = float(np.nanquantile(finite_ret, 0.10))
        tail_values = np.full(len(panel_high), np.nan, dtype=np.float32)
        tail_values[ret_mask] = (ret_values[ret_mask] <= tail_threshold).astype(np.float32)
        targets.append(
            {
                "name": "high_conf_tail_loss",
                "kind": "binary",
                "values": tail_values,
                "definition": "realized return in the bottom decile inside rank-filtered rows",
                "target_threshold": tail_threshold,
            }
        )
        targets.append(
            {
                "name": "continuous_net_utility",
                "kind": "continuous",
                "values": ret_values.astype(np.float32, copy=False),
                "definition": "continuous realized return / net utility inside rank-filtered rows",
            }
        )
    residual = pred - y_bin
    finite_resid = residual[np.isfinite(residual)]
    if len(finite_resid) >= 50:
        positive_resid = finite_resid[finite_resid > 0.0]
        if len(positive_resid) >= 50:
            overprediction_threshold = float(np.nanquantile(positive_resid, 0.75))
            threshold_definition = "upper quartile of positive oof_pred - y_bin residuals"
        else:
            overprediction_threshold = max(float(np.nanquantile(finite_resid, 0.75)), 0.0)
            threshold_definition = "upper quartile of oof_pred - y_bin residuals, floored at zero"
        residual_values = residual.to_numpy(dtype=np.float32, copy=False)
        residual_mask = np.isfinite(residual_values)
        overprediction_values = np.full(len(panel_high), np.nan, dtype=np.float32)
        overprediction_values[residual_mask] = (residual_values[residual_mask] >= overprediction_threshold).astype(
            np.float32
        )
        targets.append(
            {
                "name": "prediction_minus_outcome",
                "kind": "binary",
                "values": overprediction_values,
                "definition": threshold_definition,
                "target_threshold": overprediction_threshold,
            }
        )
    return targets


def _univariate_archetype_rows(
    *,
    head: str,
    y: np.ndarray,
    archetypes: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = np.asarray(y, dtype=np.float32)
    for col in archetypes.columns:
        score = pd.to_numeric(archetypes[col], errors="coerce").to_numpy(dtype=np.float32)
        mask = np.isfinite(score) & np.isfinite(y)
        if int(mask.sum()) < 50 or len(np.unique(y[mask].astype(np.int8))) < 2:
            continue
        auc = _safe_auc(y, score)
        q80 = float(np.nanquantile(score[mask], 0.80))
        q20 = float(np.nanquantile(score[mask], 0.20))
        top = mask & (score >= q80)
        bottom = mask & (score <= q20)
        rest = mask & (score < q80)
        fail_top = float(np.nanmean(y[top])) if bool(top.any()) else float("nan")
        fail_rest = float(np.nanmean(y[rest])) if bool(rest.any()) else float("nan")
        fail_bottom = float(np.nanmean(y[bottom])) if bool(bottom.any()) else float("nan")
        bad = score[mask & (y > 0.5)]
        good = score[mask & (y <= 0.5)]
        pooled = float(np.nanstd(score[mask]))
        effect = (
            float((np.nanmean(bad) - np.nanmean(good)) / max(pooled, 1e-8))
            if len(bad) and len(good)
            else float("nan")
        )
        rows.append(
            {
                "head": head,
                "feature": col,
                "rows": int(mask.sum()),
                "finite_fraction": float(mask.mean()),
                "auc_high_score_bad": auc,
                "directional_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan"),
                "auc_lift_abs": abs(auc - 0.5) * 2.0 if np.isfinite(auc) else float("nan"),
                "direction": "high_score_bad" if np.isfinite(auc) and auc >= 0.5 else "low_score_bad",
                "fail_rate_top20_score": fail_top,
                "fail_rate_rest": fail_rest,
                "fail_rate_bottom20_score": fail_bottom,
                "top20_minus_rest_fail_rate": fail_top - fail_rest,
                "top20_minus_bottom20_fail_rate": fail_top - fail_bottom,
                "bad_minus_good_std_effect": effect,
                "score_mean": float(np.nanmean(score[mask])),
                "score_p90": float(np.nanpercentile(score[mask], 90.0)),
            }
        )
    return rows


def _prediction_controls(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    for col in ("oof_pred", "oof_rank_pct", "oof_p_move", "oof_meta_clf", "oof_base_clf"):
        if col in panel.columns:
            out[col] = pd.to_numeric(panel[col], errors="coerce").astype(np.float32)
    return _downcast_numeric(out)


def _nuisance_controls(panel: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy(dtype=np.float64, copy=False)
    finite_ts = np.isfinite(ts_ns)
    if finite_ts.any():
        denom = max(float(np.nanmax(ts_ns[finite_ts]) - np.nanmin(ts_ns[finite_ts])), 1.0)
        time_ordinal = (ts_ns - np.nanmin(ts_ns[finite_ts])) / denom
    else:
        time_ordinal = np.zeros(len(panel), dtype=np.float64)
    rows_per_ts = ts.groupby(ts).transform("size").to_numpy(dtype=np.float32, copy=False)
    out = pd.DataFrame(index=panel.index)
    out["time_ordinal_nuisance"] = np.asarray(time_ordinal, dtype=np.float32)
    out["rows_per_timestamp_nuisance"] = rows_per_ts.astype(np.float32, copy=False)
    hour = ts.dt.hour.fillna(0).to_numpy(dtype=np.float32, copy=False)
    out["session_sin_nuisance"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32, copy=False)
    out["session_cos_nuisance"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32, copy=False)
    return _downcast_numeric(out)


def _select_columns_by_patterns(frame: pd.DataFrame, patterns: tuple[str, ...], *, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    cols = [
        c
        for c in frame.columns
        if any(pattern in c.lower() for pattern in patterns)
        and not any(pattern in c.lower() for pattern in exclude)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return frame.loc[:, list(dict.fromkeys(cols))]


def _zscore_frame(frame: pd.DataFrame, max_cols: int) -> pd.DataFrame:
    cols = list(frame.columns[:max_cols])
    data: dict[str, np.ndarray] = {}
    for col in cols:
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float32)
        mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.0
        std = float(np.nanstd(arr)) if np.isfinite(arr).any() else 0.0
        if not np.isfinite(std) or std <= 1e-8:
            continue
        data[col] = ((arr - mean) / std).astype(np.float32, copy=False)
    return pd.DataFrame(data, index=frame.index)


def _interaction_frame(left: pd.DataFrame, right: pd.DataFrame, *, max_left: int = 8, max_right: int = 10) -> pd.DataFrame:
    left_z = _zscore_frame(left, max_cols=max_left)
    right_z = _zscore_frame(right, max_cols=max_right)
    data: dict[str, np.ndarray] = {}
    for lc in left_z.columns:
        for rc in right_z.columns:
            data[f"int__{lc}__x__{rc}"] = (
                left_z[lc].to_numpy(dtype=np.float32, copy=False)
                * right_z[rc].to_numpy(dtype=np.float32, copy=False)
            ).astype(np.float32, copy=False)
    return pd.DataFrame(data, index=left.index)


def _model_family_matrices(
    *,
    panel_high: pd.DataFrame,
    archetypes_high: pd.DataFrame,
    candidate_x_high: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    prediction_controls = _prediction_controls(panel_high)
    nuisance_controls = _nuisance_controls(panel_high)
    archetypes = archetypes_high.loc[
        :,
        [c for c in archetypes_high.columns if archetypes_high[c].notna().mean() > 0.02],
    ]
    candidate_x_high = candidate_x_high.loc[
        :,
        [c for c in candidate_x_high.columns if candidate_x_high[c].notna().mean() > 0.02],
    ]
    model_support = _select_columns_by_patterns(
        pd.concat([candidate_x_high, archetypes], axis=1, copy=False),
        ("support", "leaf", "score_path", "score_early", "rank_100_minus", "dae", "gmm", "cluster", "centroid"),
    )
    market_state = _select_columns_by_patterns(
        archetypes,
        ("leverage", "crowding", "liquidity", "tail", "volatility", "relative", "breadth", "network"),
        exclude=("model_path", "support"),
    )
    support_x_market = _interaction_frame(model_support, market_state)
    return {
        "prediction_controls_only": prediction_controls,
        "nuisance_controls_only": nuisance_controls,
        "prediction_plus_nuisance": pd.concat(
            [
                prediction_controls,
                nuisance_controls.loc[:, [c for c in nuisance_controls.columns if c not in prediction_controls.columns]],
            ],
            axis=1,
            copy=False,
        ),
        "clean_reconstructed_features": candidate_x_high,
        "model_support_variables": model_support,
        "market_state_archetypes": market_state,
        "archetype_only": archetypes,
        "prediction_plus_archetype": pd.concat(
            [
                prediction_controls,
                archetypes.loc[:, [c for c in archetypes.columns if c not in prediction_controls.columns]],
            ],
            axis=1,
            copy=False,
        ),
        "nuisance_plus_archetype": pd.concat(
            [
                nuisance_controls,
                archetypes.loc[:, [c for c in archetypes.columns if c not in nuisance_controls.columns]],
            ],
            axis=1,
            copy=False,
        ),
        "support_x_market_interactions": pd.concat(
            [
                prediction_controls,
                model_support.loc[:, list(model_support.columns[:12])],
                market_state.loc[:, list(market_state.columns[:12])],
                support_x_market,
            ],
            axis=1,
            copy=False,
        ),
    }


def _score_models(
    *,
    head: str,
    panel_high: pd.DataFrame,
    target_name: str,
    target_kind: str,
    y: np.ndarray,
    target_meta: dict[str, Any],
    archetypes_high: pd.DataFrame,
    candidate_x_high: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices = _model_family_matrices(
        panel_high=panel_high,
        archetypes_high=archetypes_high,
        candidate_x_high=candidate_x_high,
    )
    realized_return = _pick_realized_return(panel_high)
    for model_name, x in matrices.items():
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.loc[:, [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]]
        if x.empty:
            rows.append(
                {
                    "head": head,
                    "target": target_name,
                    "target_kind": target_kind,
                    "model": model_name,
                    "auc_mean": np.nan,
                    "auc_std": np.nan,
                    "folds": 0,
                    "rows": int(len(panel_high)),
                    "positive_rate": float(np.nanmean(y)) if len(y) else np.nan,
                    "feature_count": 0,
                    "reason": "empty_matrix",
                    **target_meta,
                }
            )
            continue
        if target_kind == "continuous":
            summary = _fit_lgbm_cv_regression_detailed(
                x=x,
                y=np.asarray(y, dtype=np.float32),
                timestamps=panel_high["timestamp"],
                max_rows=max_rows,
                seed=seed,
            )
        else:
            summary = _fit_lgbm_cv_detailed(
                x=x,
                y=np.asarray(y, dtype=np.float32),
                timestamps=panel_high["timestamp"],
                realized_return=realized_return,
                max_rows=max_rows,
                seed=seed,
            )
        rows.append(
            {
                "head": head,
                "target": target_name,
                "target_kind": target_kind,
                "model": model_name,
                "feature_count": int(x.shape[1]),
                **target_meta,
                **summary,
            }
        )
    return rows


def _safe_week_start(values: Any) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts.dt.to_period("W").dt.start_time


def _clean_binary_matrix(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = _prepare_model_matrix(x_train).replace([np.inf, -np.inf], np.nan)
    test = _prepare_model_matrix(x_test).replace([np.inf, -np.inf], np.nan)
    keep: list[str] = []
    for col in train.columns:
        ser = pd.to_numeric(train[col], errors="coerce")
        finite_rate = float(ser.notna().mean())
        if finite_rate <= 0.02:
            continue
        arr = ser.to_numpy(dtype=np.float64, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size < 20 or float(np.nanstd(finite)) <= 1e-12:
            continue
        keep.append(col)
    if not keep:
        return pd.DataFrame(index=train.index), pd.DataFrame(index=test.index), []
    return (
        train.loc[:, keep].reset_index(drop=True),
        test.reindex(columns=keep).reset_index(drop=True),
        keep,
    )


def _fit_lgbm_episode_transfer_binary(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    realized_return: pd.Series,
    episode_labels: pd.Series,
    heldout_episode: Any,
    max_train_rows: int,
    seed: int,
) -> dict[str, Any]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for these diagnostics")
    y_raw = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(y_raw)
    episodes = pd.Series(episode_labels).reset_index(drop=True)
    holdout = episodes.eq(heldout_episode).to_numpy(dtype=bool)
    train_mask = valid & ~holdout
    test_mask = valid & holdout
    if int(train_mask.sum()) < 200 or int(test_mask.sum()) < 50:
        return {
            "transfer_rows_train": int(train_mask.sum()),
            "transfer_rows_test": int(test_mask.sum()),
            "transfer_reason": "insufficient_train_or_test_rows",
        }
    y_train = (y_raw[train_mask] > 0.5).astype(np.int8, copy=False)
    y_test = (y_raw[test_mask] > 0.5).astype(np.int8, copy=False)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "transfer_rows_train": int(train_mask.sum()),
            "transfer_rows_test": int(test_mask.sum()),
            "transfer_train_positive_rate": float(np.mean(y_train)) if len(y_train) else np.nan,
            "transfer_test_positive_rate": float(np.mean(y_test)) if len(y_test) else np.nan,
            "transfer_reason": "insufficient_train_or_test_classes",
        }
    train_idx = np.flatnonzero(train_mask)
    if max_train_rows > 0 and len(train_idx) > max_train_rows:
        local = _period_stratified_sample(
            pd.DataFrame({"timestamp": pd.to_datetime(timestamps.iloc[train_idx], utc=True, errors="coerce")}),
            y_train,
            max_rows=max_train_rows,
            seed=seed,
        )
        train_idx = train_idx[local]
        y_train = (y_raw[train_idx] > 0.5).astype(np.int8, copy=False)
        if len(np.unique(y_train)) < 2:
            return {
                "transfer_rows_train": int(len(train_idx)),
                "transfer_rows_test": int(test_mask.sum()),
                "transfer_reason": "subsample_removed_train_class",
            }
    test_idx = np.flatnonzero(test_mask)
    x_train, x_test, keep = _clean_binary_matrix(x.iloc[train_idx], x.iloc[test_idx])
    if not keep:
        return {
            "transfer_rows_train": int(len(train_idx)),
            "transfer_rows_test": int(len(test_idx)),
            "transfer_reason": "empty_matrix",
            "transfer_feature_count": 0,
        }
    min_child = max(50, int(math.ceil(0.025 * len(train_idx))))
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.035,
        max_depth=3,
        num_leaves=8,
        min_child_samples=min_child,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_test)[:, 1].astype(np.float32, copy=False)
    ret = pd.to_numeric(realized_return.iloc[test_idx], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    metrics = _classification_and_economic_metrics(
        y=y_test.astype(np.float32),
        pred=pred,
        realized_return=ret,
        prefix="transfer_",
    )
    return {
        "transfer_rows_train": int(len(train_idx)),
        "transfer_rows_test": int(len(test_idx)),
        "transfer_train_positive_rate": float(np.mean(y_train)),
        "transfer_test_positive_rate": float(np.mean(y_test)),
        "transfer_feature_count": int(len(keep)),
        "transfer_reason": "",
        **metrics,
    }


def _leave_one_episode_transfer_rows(
    *,
    head: str,
    panel: pd.DataFrame,
    panel_high: pd.DataFrame,
    target_name: str,
    target_kind: str,
    y: np.ndarray,
    archetypes_high: pd.DataFrame,
    candidate_x_high: pd.DataFrame,
    rank_threshold: float,
    recent_weeks: int,
    min_week_rows: int,
    max_train_rows: int,
    max_episodes: int,
    model_allowlist: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    if target_kind != "binary":
        return []
    weekly = _weekly_high_conf_metrics(panel, rank_threshold, min_week_rows)
    bad_weeks, bad_meta = _bad_recent_weeks(
        weekly,
        recent_weeks=recent_weeks,
        min_week_rows=min_week_rows,
    )
    if not bad_weeks:
        return [
            {
                "head": head,
                "target": target_name,
                "model": "__summary__",
                "heldout_episode": "",
                "bad_episode_count": 0,
                "transfer_reason": str(bad_meta.get("reason", "no_bad_weeks")),
            }
        ]
    high_week = _safe_week_start(panel_high["timestamp"])
    bad_set = {pd.Timestamp(w).tz_localize(None) for w in bad_weeks}
    episodes = [
        episode
        for episode in high_week.dropna().drop_duplicates().tolist()
        if pd.Timestamp(episode).tz_localize(None) in bad_set
    ]
    episode_counts = high_week.value_counts().to_dict()
    episodes = [
        episode
        for episode in episodes
        if int(episode_counts.get(episode, 0)) >= int(min_week_rows)
    ]
    if max_episodes > 0:
        episodes = episodes[-int(max_episodes):]
    if len(episodes) < 2:
        return [
            {
                "head": head,
                "target": target_name,
                "model": "__summary__",
                "heldout_episode": str(episodes[0]) if episodes else "",
                "bad_episode_count": int(len(episodes)),
                "transfer_reason": "insufficient_bad_episodes_for_leave_one_out",
                "bad_week_reason": str(bad_meta.get("reason", "")),
            }
        ]
    matrices = _model_family_matrices(
        panel_high=panel_high,
        archetypes_high=archetypes_high,
        candidate_x_high=candidate_x_high,
    )
    realized_return = _pick_realized_return(panel_high).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for model_name, x in matrices.items():
        if model_allowlist and model_name not in model_allowlist:
            continue
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.loc[:, [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]]
        if x.empty:
            rows.append(
                {
                    "head": head,
                    "target": target_name,
                    "model": model_name,
                    "bad_episode_count": int(len(episodes)),
                    "heldout_episode": "__all__",
                    "transfer_reason": "empty_matrix",
                    "transfer_feature_count": 0,
                }
            )
            continue
        for episode_i, episode in enumerate(episodes, start=1):
            summary = _fit_lgbm_episode_transfer_binary(
                x=x.reset_index(drop=True),
                y=np.asarray(y, dtype=np.float32),
                timestamps=panel_high["timestamp"].reset_index(drop=True),
                realized_return=realized_return,
                episode_labels=high_week.reset_index(drop=True),
                heldout_episode=episode,
                max_train_rows=max_train_rows,
                seed=seed
                + episode_i * 1009
                + (
                    sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(model_name)))
                    % 997
                ),
            )
            rows.append(
                {
                    "head": head,
                    "target": target_name,
                    "target_kind": target_kind,
                    "model": model_name,
                    "bad_episode_count": int(len(episodes)),
                    "heldout_episode": str(episode),
                    "bad_week_reason": str(bad_meta.get("reason", "")),
                    **summary,
                }
            )
    return rows


def _fit_shadow_failure_risk_oof(
    *,
    head: str,
    target_name: str,
    model_name: str,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    realized_return: pd.Series,
    max_rows: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if lgb is None:
        raise RuntimeError("lightgbm is required for these diagnostics")
    y_raw = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(y_raw)
    x = x.loc[valid].reset_index(drop=True)
    ts = pd.to_datetime(timestamps.loc[valid], utc=True, errors="coerce").reset_index(drop=True)
    ret = pd.to_numeric(realized_return.loc[valid], errors="coerce").reset_index(drop=True)
    original_row = np.flatnonzero(valid).astype(np.int32)
    y_bin = (y_raw[valid] > 0.5).astype(np.int8, copy=False)
    if len(y_bin) < 300 or len(np.unique(y_bin)) < 2:
        return (
            {
                "head": head,
                "target": target_name,
                "model": model_name,
                "rows": int(len(y_bin)),
                "folds": 0,
                "reason": "insufficient_classes_or_rows",
            },
            pd.DataFrame(),
        )
    sample_idx = _period_stratified_sample(
        pd.DataFrame({"timestamp": ts}),
        y_bin,
        max_rows=max_rows,
        seed=seed,
    )
    x = x.iloc[sample_idx].reset_index(drop=True)
    y_bin = y_bin[sample_idx]
    ts = ts.iloc[sample_idx].reset_index(drop=True)
    ret = ret.iloc[sample_idx].reset_index(drop=True)
    original_row = original_row[sample_idx]
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    x = x.iloc[order].reset_index(drop=True)
    y_bin = y_bin[order]
    ts = ts.iloc[order].reset_index(drop=True)
    ret = ret.iloc[order].reset_index(drop=True)
    original_row = original_row[order]
    if len(y_bin) < 300 or len(np.unique(y_bin)) < 2:
        return (
            {
                "head": head,
                "target": target_name,
                "model": model_name,
                "rows": int(len(y_bin)),
                "folds": 0,
                "reason": "insufficient_classes_or_rows_after_sampling",
            },
            pd.DataFrame(),
        )

    n_splits = min(5, max(2, len(y_bin) // 5000))
    oof = np.full(len(y_bin), np.nan, dtype=np.float32)
    fold_ids = np.full(len(y_bin), -1, dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(x), start=1):
        if len(train_idx) < 200 or len(test_idx) < 50:
            continue
        if len(np.unique(y_bin[train_idx])) < 2 or len(np.unique(y_bin[test_idx])) < 2:
            continue
        x_train, x_test, keep = _clean_binary_matrix(x.iloc[train_idx], x.iloc[test_idx])
        if not keep:
            continue
        min_child = max(50, int(math.ceil(0.025 * len(train_idx))))
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=450,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=8,
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold * 271,
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(x_train, y_bin[train_idx])
        pred = clf.predict_proba(x_test)[:, 1].astype(np.float32, copy=False)
        oof[test_idx] = pred
        fold_ids[test_idx] = np.int16(fold)
        fold_rows.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "feature_count": int(len(keep)),
                "test_auc": _safe_auc(y_bin[test_idx].astype(np.float32), pred),
                "test_positive_rate": float(np.mean(y_bin[test_idx])),
            }
        )
    if not fold_rows or int(np.isfinite(oof).sum()) < 100:
        return (
            {
                "head": head,
                "target": target_name,
                "model": model_name,
                "rows": int(len(y_bin)),
                "folds": int(len(fold_rows)),
                "reason": "no_valid_shadow_oof_folds",
            },
            pd.DataFrame(),
        )
    metrics = _classification_and_economic_metrics(
        y=y_bin.astype(np.float32),
        pred=oof,
        realized_return=ret.to_numpy(dtype=np.float64, copy=False),
        prefix="shadow_",
    )
    oof_df = pd.DataFrame(
        {
            "head": head,
            "target": target_name,
            "model": model_name,
            "row_index": original_row.astype(np.int32),
            "timestamp": ts,
            "realized_return": ret.to_numpy(dtype=np.float32, copy=False),
            "failure_target": y_bin.astype(np.int8, copy=False),
            "shadow_failure_risk": oof,
            "shadow_fold": fold_ids,
        }
    )
    oof_df = oof_df.loc[np.isfinite(oof_df["shadow_failure_risk"])].reset_index(drop=True)
    return (
        {
            "head": head,
            "target": target_name,
            "model": model_name,
            "rows": int(len(y_bin)),
            "oof_rows": int(len(oof_df)),
            "folds": int(len(fold_rows)),
            "positive_rate": float(np.mean(y_bin)),
            "reason": "",
            "folds_detail": fold_rows,
            **metrics,
        },
        oof_df,
    )


def _smooth_risk_scalers(risk: np.ndarray) -> dict[str, np.ndarray]:
    rr = _rank_to_unit(risk)
    rr = np.nan_to_num(rr, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    out: dict[str, np.ndarray] = {}
    for floor in (0.10, 0.25, 0.40):
        for alpha in (0.50, 0.75, 1.00):
            size = np.clip(1.0 - alpha * rr, floor, 1.0).astype(np.float32, copy=False)
            out[f"linear_floor{int(floor * 100)}_alpha{int(alpha * 100)}"] = size
    for floor in (0.10, 0.25, 0.40):
        for center in (0.70, 0.80):
            steep = 10.0
            high_risk = 1.0 / (1.0 + np.exp(-steep * (rr - center)))
            size = (floor + (1.0 - floor) * (1.0 - high_risk)).astype(np.float32, copy=False)
            out[f"logistic_floor{int(floor * 100)}_center{int(center * 100)}"] = size
    return out


def _tail_mean(values: np.ndarray, q: float = 0.05) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 50:
        return float("nan")
    threshold = float(np.nanquantile(arr, q))
    tail = arr[arr <= threshold]
    return float(np.nanmean(tail)) if tail.size else float("nan")


def _evaluate_smooth_risk_scalers(oof_scores: pd.DataFrame) -> pd.DataFrame:
    if oof_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["head", "target", "model"]
    for group_key, group in oof_scores.groupby(group_cols, sort=False):
        head, target, model_name = group_key
        risk = pd.to_numeric(group["shadow_failure_risk"], errors="coerce").to_numpy(dtype=np.float32)
        ret = pd.to_numeric(group["realized_return"], errors="coerce").to_numpy(dtype=np.float64)
        y = pd.to_numeric(group["failure_target"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(risk) & np.isfinite(ret)
        if int(mask.sum()) < 50:
            continue
        risk = risk[mask]
        ret = ret[mask]
        y = y[mask]
        unsized_mean = float(np.nanmean(ret))
        unsized_sum = float(np.nansum(ret))
        unsized_tail = _tail_mean(ret, 0.05)
        unsized_loss_mean = float(np.nanmean(ret[ret < 0.0])) if np.any(ret < 0.0) else np.nan
        scalers = _smooth_risk_scalers(risk)
        for policy, size in scalers.items():
            size = np.asarray(size, dtype=np.float64)
            sized_ret = ret * size
            winners = ret > 0.0
            losers = ret < 0.0
            failure = y > 0.5
            success = y <= 0.5
            sized_tail = _tail_mean(sized_ret, 0.05)
            sized_loss_mean = float(np.nanmean(sized_ret[losers])) if np.any(losers) else np.nan
            winner_haircut = float(np.nanmean((1.0 - size[winners]) * ret[winners])) if np.any(winners) else 0.0
            loser_loss_reduction = (
                float(np.nanmean((1.0 - size[losers]) * np.abs(ret[losers])))
                if np.any(losers)
                else 0.0
            )
            failure_exposure = float(np.nanmean(size[failure])) if np.any(failure) else np.nan
            success_exposure = float(np.nanmean(size[success])) if np.any(success) else np.nan
            rows.append(
                {
                    "head": head,
                    "target": target,
                    "model": model_name,
                    "policy": policy,
                    "rows": int(len(ret)),
                    "avg_size": float(np.nanmean(size)),
                    "p10_size": float(np.nanpercentile(size, 10.0)),
                    "p50_size": float(np.nanpercentile(size, 50.0)),
                    "failure_exposure": failure_exposure,
                    "success_exposure": success_exposure,
                    "success_minus_failure_exposure": (
                        success_exposure - failure_exposure
                        if np.isfinite(success_exposure) and np.isfinite(failure_exposure)
                        else np.nan
                    ),
                    "unsized_return_mean": unsized_mean,
                    "sized_return_mean": float(np.nanmean(sized_ret)),
                    "return_mean_delta": float(np.nanmean(sized_ret) - unsized_mean),
                    "unsized_return_sum": unsized_sum,
                    "sized_return_sum": float(np.nansum(sized_ret)),
                    "return_sum_delta": float(np.nansum(sized_ret) - unsized_sum),
                    "unsized_tail_mean_q05": unsized_tail,
                    "sized_tail_mean_q05": sized_tail,
                    "tail_loss_delta_q05": (
                        sized_tail - unsized_tail
                        if np.isfinite(sized_tail) and np.isfinite(unsized_tail)
                        else np.nan
                    ),
                    "unsized_loss_mean": unsized_loss_mean,
                    "sized_loss_mean": sized_loss_mean,
                    "loss_mean_delta": (
                        sized_loss_mean - unsized_loss_mean
                        if np.isfinite(sized_loss_mean) and np.isfinite(unsized_loss_mean)
                        else np.nan
                    ),
                    "winner_haircut_mean": winner_haircut,
                    "loser_loss_reduction_mean": loser_loss_reduction,
                    "risk_sizing_score": float(
                        (0.50 * (sized_tail - unsized_tail) if np.isfinite(sized_tail) and np.isfinite(unsized_tail) else 0.0)
                        + 0.30 * loser_loss_reduction
                        - 0.20 * winner_haircut
                    ),
                }
            )
    return pd.DataFrame(rows)


def _training_intervention_recommendations(
    *,
    model_rows: pd.DataFrame,
    transfer: pd.DataFrame,
    shadow_policy: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> pd.DataFrame:
    """Conservative promotion gates for diagnostic-only intervention candidates."""
    rows: list[dict[str, Any]] = []

    transfer_summary = pd.DataFrame()
    if not transfer.empty and {"head", "target", "model"}.issubset(transfer.columns):
        valid_transfer = transfer.loc[
            transfer["model"].astype(str).ne("__summary__")
            & _blank_reason_mask(transfer, "transfer_reason")
        ].copy()
        if not valid_transfer.empty:
            transfer_summary = (
                valid_transfer.groupby(["head", "target", "model"], dropna=False)
                .agg(
                    transfer_episodes=("heldout_episode", "nunique"),
                    median_transfer_auc=("transfer_roc_auc", "median"),
                    mean_transfer_auc=("transfer_roc_auc", "mean"),
                    transfer_auc_positive_rate=(
                        "transfer_roc_auc",
                        lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.5)),
                    ),
                    median_transfer_failure_capture_10pct=(
                        "transfer_failure_capture_at_10pct_abstain",
                        "median",
                    ),
                    median_transfer_retained_return_10pct=(
                        "transfer_retained_return_mean_at_10pct_abstain",
                        "median",
                    ),
                    median_transfer_rejected_winner_cost_10pct=(
                        "transfer_rejected_winner_cost_at_10pct_abstain",
                        "median",
                    ),
                )
                .reset_index()
            )

    incremental_rows: list[dict[str, Any]] = []
    if not model_rows.empty and {"head", "target", "model", "auc_mean"}.issubset(model_rows.columns):
        binary = model_rows.loc[model_rows.get("target_kind", "binary").astype(str).eq("binary")].copy()
        if not binary.empty:
            pivot = binary.pivot_table(
                index=["head", "target"],
                columns="model",
                values="auc_mean",
                aggfunc="first",
            )
            for (head, target), row in pivot.iterrows():
                pred_base = float(row.get("prediction_controls_only", np.nan))
                nuisance_base = float(row.get("nuisance_controls_only", np.nan))
                candidates = [
                    "prediction_plus_archetype",
                    "market_state_archetypes",
                    "archetype_only",
                    "support_x_market_interactions",
                ]
                best_model = ""
                best_auc = float("nan")
                best_delta_pred = float("-inf")
                best_delta_nuisance = float("nan")
                for model_name in candidates:
                    auc = float(row.get(model_name, np.nan))
                    if not np.isfinite(auc):
                        continue
                    delta_pred = auc - pred_base if np.isfinite(pred_base) else np.nan
                    score_delta = delta_pred if np.isfinite(delta_pred) else -np.inf
                    if score_delta > best_delta_pred:
                        best_model = model_name
                        best_auc = auc
                        best_delta_pred = score_delta
                        best_delta_nuisance = auc - nuisance_base if np.isfinite(nuisance_base) else np.nan
                incremental_rows.append(
                    {
                        "head": head,
                        "target": target,
                        "best_incremental_model": best_model,
                        "best_incremental_auc": best_auc,
                        "prediction_controls_auc": pred_base,
                        "nuisance_controls_auc": nuisance_base,
                        "delta_vs_prediction_controls": (
                            best_delta_pred if np.isfinite(best_delta_pred) else np.nan
                        ),
                        "delta_vs_nuisance_controls": best_delta_nuisance,
                    }
                )
    incremental = pd.DataFrame(incremental_rows)

    decomposition_summary = pd.DataFrame()
    if not decomposition.empty and {"head", "delta_full_vs_prediction"}.issubset(decomposition.columns):
        decomposition_summary = (
            decomposition.groupby("head", dropna=False)
            .agg(
                best_period_within_delta=("delta_full_vs_prediction", "max"),
                median_period_within_delta=("delta_full_vs_prediction", "median"),
                positive_period_within_share=(
                    "delta_full_vs_prediction",
                    lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.0)),
                ),
            )
            .reset_index()
        )

    if not shadow_policy.empty and {"head", "target", "model"}.issubset(shadow_policy.columns):
        shadow_rank = shadow_policy.copy()
        shadow_rank["risk_sizing_score"] = pd.to_numeric(
            shadow_rank.get("risk_sizing_score", np.nan),
            errors="coerce",
        )
        shadow_rank = shadow_rank.sort_values(
            ["head", "target", "risk_sizing_score", "tail_loss_delta_q05"],
            ascending=[True, True, False, False],
        )
        best_shadow = shadow_rank.groupby(["head", "target"], as_index=False).head(1)
        for _, rec in best_shadow.iterrows():
            risk_score = float(rec.get("risk_sizing_score", np.nan))
            tail_delta = float(rec.get("tail_loss_delta_q05", np.nan))
            exposure_gap = float(rec.get("success_minus_failure_exposure", np.nan))
            winner_haircut = float(rec.get("winner_haircut_mean", np.nan))
            loser_reduction = float(rec.get("loser_loss_reduction_mean", np.nan))
            avg_size = float(rec.get("avg_size", np.nan))
            economic_pass = (
                np.isfinite(risk_score)
                and risk_score > 0.0
                and np.isfinite(tail_delta)
                and tail_delta >= 0.0
                and np.isfinite(exposure_gap)
                and exposure_gap > 0.0
                and (not np.isfinite(avg_size) or avg_size >= 0.35)
                and (
                    not np.isfinite(winner_haircut)
                    or not np.isfinite(loser_reduction)
                    or loser_reduction >= 0.75 * winner_haircut
                )
            )
            rows.append(
                {
                    "head": rec.get("head"),
                    "target": rec.get("target"),
                    "action": "shadow_smooth_risk_sizing",
                    "model": rec.get("model"),
                    "policy": rec.get("policy"),
                    "recommendation": "candidate" if economic_pass else "reject",
                    "decision_reason": (
                        "smooth sizing reduces risk exposure with acceptable winner haircut"
                        if economic_pass
                        else "risk sizing evidence is not yet economically clean"
                    ),
                    "economic_pass": bool(economic_pass),
                    "recurrence_pass": np.nan,
                    "incremental_lift_pass": np.nan,
                    "risk_sizing_score": risk_score,
                    "tail_loss_delta_q05": tail_delta,
                    "success_minus_failure_exposure": exposure_gap,
                    "winner_haircut_mean": winner_haircut,
                    "loser_loss_reduction_mean": loser_reduction,
                    "avg_size": avg_size,
                }
            )

    if not incremental.empty:
        for _, inc in incremental.iterrows():
            head = str(inc.get("head", ""))
            target = str(inc.get("target", ""))
            best_model = str(inc.get("best_incremental_model", ""))
            transfer_match = (
                transfer_summary.loc[
                    (transfer_summary["head"].astype(str) == head)
                    & (transfer_summary["target"].astype(str) == target)
                    & (transfer_summary["model"].astype(str).isin({best_model, "prediction_plus_archetype", "support_x_market_interactions"}))
                ]
                if not transfer_summary.empty
                else pd.DataFrame()
            )
            if not transfer_match.empty:
                transfer_match = transfer_match.sort_values(
                    ["median_transfer_auc", "transfer_auc_positive_rate"],
                    ascending=False,
                )
                trans = transfer_match.iloc[0]
            else:
                trans = pd.Series(dtype=object)
            decomp_match = (
                decomposition_summary.loc[decomposition_summary["head"].astype(str) == head]
                if not decomposition_summary.empty
                else pd.DataFrame()
            )
            decomp = decomp_match.iloc[0] if not decomp_match.empty else pd.Series(dtype=object)
            delta_pred = float(inc.get("delta_vs_prediction_controls", np.nan))
            delta_nuisance = float(inc.get("delta_vs_nuisance_controls", np.nan))
            transfer_auc = float(trans.get("median_transfer_auc", np.nan))
            transfer_hit = float(trans.get("transfer_auc_positive_rate", np.nan))
            transfer_episodes = int(trans.get("transfer_episodes", 0) or 0)
            decomp_delta = float(decomp.get("best_period_within_delta", np.nan))
            recurrence_pass = (
                transfer_episodes >= 2
                and np.isfinite(transfer_auc)
                and transfer_auc >= 0.55
                and np.isfinite(transfer_hit)
                and transfer_hit >= 0.60
            )
            incremental_pass = (
                np.isfinite(delta_pred)
                and delta_pred >= 0.01
                and (
                    not np.isfinite(delta_nuisance)
                    or delta_nuisance >= 0.005
                    or (np.isfinite(decomp_delta) and decomp_delta >= 0.01)
                )
            )
            recommendation = "candidate" if recurrence_pass and incremental_pass else "reject"
            if recommendation == "candidate":
                reason = "recurring transfer and incremental context lift are present"
            elif not recurrence_pass and not incremental_pass:
                reason = "missing recurrence and incremental lift"
            elif not recurrence_pass:
                reason = "missing leave-one-episode recurrence"
            else:
                reason = "missing incremental lift beyond prediction/nuisance controls"
            rows.append(
                {
                    "head": head,
                    "target": target,
                    "action": "archetype_aware_meta_retrain",
                    "model": best_model,
                    "policy": "",
                    "recommendation": recommendation,
                    "decision_reason": reason,
                    "economic_pass": np.nan,
                    "recurrence_pass": bool(recurrence_pass),
                    "incremental_lift_pass": bool(incremental_pass),
                    "prediction_controls_auc": inc.get("prediction_controls_auc", np.nan),
                    "nuisance_controls_auc": inc.get("nuisance_controls_auc", np.nan),
                    "best_incremental_auc": inc.get("best_incremental_auc", np.nan),
                    "delta_vs_prediction_controls": delta_pred,
                    "delta_vs_nuisance_controls": delta_nuisance,
                    "transfer_model": trans.get("model", ""),
                    "transfer_episodes": transfer_episodes,
                    "median_transfer_auc": transfer_auc,
                    "transfer_auc_positive_rate": transfer_hit,
                    "best_period_within_delta": decomp_delta,
                }
            )

    return pd.DataFrame(rows)


def _weekly_rows(
    *,
    head: str,
    panel: pd.DataFrame,
    archetypes: pd.DataFrame,
    rank_threshold: float,
    min_week_rows: int,
    recent_weeks: int,
) -> list[dict[str, Any]]:
    weekly = _weekly_high_conf_metrics(panel, rank_threshold, min_week_rows)
    bad_weeks, bad_meta = _bad_recent_weeks(
        weekly,
        recent_weeks=recent_weeks,
        min_week_rows=min_week_rows,
    )
    high_mask = pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= rank_threshold
    if not bool(high_mask.any()) or not bad_weeks:
        return [
            {
                "head": head,
                "feature": "__summary__",
                "bad_week_count": int(len(bad_weeks)),
                "reason": bad_meta.get("reason", "no_bad_weeks"),
            }
        ]
    data = panel.loc[high_mask, ["timestamp"]].copy()
    data["week"] = pd.to_datetime(data["timestamp"], utc=True).dt.to_period("W").dt.start_time
    bad_set = {pd.Timestamp(w).tz_localize(None) for w in bad_weeks}
    y_week = data["week"].map(lambda v: pd.Timestamp(v).tz_localize(None) in bad_set).to_numpy(dtype=np.int8)
    a = archetypes.loc[high_mask].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for col in a.columns:
        score = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=np.float32)
        mask = np.isfinite(score)
        if int(mask.sum()) < 50 or len(np.unique(y_week[mask])) < 2:
            continue
        auc = _safe_auc(y_week.astype(np.float32), score)
        bad_mean = float(np.nanmean(score[mask & (y_week == 1)]))
        normal_mean = float(np.nanmean(score[mask & (y_week == 0)]))
        rows.append(
            {
                "head": head,
                "feature": col,
                "bad_week_count": int(len(bad_weeks)),
                "bad_week_reason": bad_meta.get("reason", ""),
                "bad_week_auc_high_score_bad": auc,
                "directional_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                "bad_week_score_mean": bad_mean,
                "normal_week_score_mean": normal_mean,
                "bad_minus_normal_score": bad_mean - normal_mean,
            }
        )
    return rows


def _fit_logit_temporal_auc(
    *,
    x: pd.DataFrame,
    y: np.ndarray,
    timestamps: pd.Series,
    max_rows: int,
    seed: int,
) -> float:
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=np.int8)
    valid = np.isfinite(y)
    x = x.loc[valid].reset_index(drop=True)
    ts = pd.to_datetime(timestamps.loc[valid], utc=True, errors="coerce").reset_index(drop=True)
    y = y[valid]
    if len(y) < 200 or len(np.unique(y)) < 2 or x.empty:
        return float("nan")
    sample_idx = _period_stratified_sample(pd.DataFrame({"timestamp": ts}), y, max_rows=max_rows, seed=seed)
    x = x.iloc[sample_idx].reset_index(drop=True)
    y = y[sample_idx]
    ts = ts.iloc[sample_idx].reset_index(drop=True)
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]", copy=False), kind="mergesort")
    x = _prepare_model_matrix(x.iloc[order].reset_index(drop=True))
    y = y[order]
    splitter = TimeSeriesSplit(n_splits=min(5, max(2, len(y) // 5000))).split(x)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(splitter):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        train = x.iloc[train_idx].copy()
        test = x.iloc[test_idx].copy()
        med = train.median(numeric_only=True)
        train = train.fillna(med).fillna(0.0)
        test = test.fillna(med).fillna(0.0)
        mean = train.mean(axis=0)
        std = train.std(axis=0).replace(0.0, 1.0)
        train = (train - mean) / std
        test = (test - mean) / std
        clf = LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=300,
            class_weight="balanced",
            random_state=seed + fold,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(train, y[train_idx])
        oof[test_idx] = clf.predict_proba(test)[:, 1].astype(np.float32, copy=False)
    return _safe_auc(y.astype(np.float32), oof)


def _period_within_rows(
    *,
    head: str,
    panel_high: pd.DataFrame,
    y: np.ndarray,
    archetypes_high: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    prediction_controls = _prediction_controls(panel_high)
    if prediction_controls.empty:
        prediction_controls = pd.DataFrame({"constant_prediction_control": np.zeros(len(panel_high), dtype=np.float32)})
    ts = pd.to_datetime(panel_high["timestamp"], utc=True, errors="coerce")
    pred_ref = (
        pd.to_numeric(panel_high["oof_rank_pct"], errors="coerce")
        if "oof_rank_pct" in panel_high.columns
        else pd.Series(np.nan, index=panel_high.index)
    )
    pred_z = pred_ref.to_numpy(dtype=np.float32, copy=False)
    if np.isfinite(pred_z).any():
        pred_z = ((pred_z - np.nanmean(pred_z)) / max(float(np.nanstd(pred_z)), 1e-8)).astype(np.float32, copy=False)
    else:
        pred_z = np.zeros(len(panel_high), dtype=np.float32)

    rows: list[dict[str, Any]] = []
    candidate_cols = [
        c
        for c in archetypes_high.columns
        if c.endswith("_score") or c.endswith("_probability") or c in {"dominant_archetype_score", "archetype_uncertainty"}
    ]
    for idx, col in enumerate(candidate_cols):
        score = pd.to_numeric(archetypes_high[col], errors="coerce").astype("float32")
        if score.notna().mean() < 0.02 or float(score.std(skipna=True) or 0.0) <= 1e-8:
            continue
        period = score.groupby(ts).transform("mean").astype("float32")
        within = (score - period).astype("float32")
        interaction = (period.to_numpy(dtype=np.float32, copy=False) * pred_z).astype(np.float32, copy=False)
        base = prediction_controls.copy()
        matrices = {
            "prediction_only": base,
            "period_state": pd.concat([base, pd.DataFrame({f"{col}__period_mean": period})], axis=1),
            "within_timestamp_state": pd.concat([base, pd.DataFrame({f"{col}__within_timestamp": within})], axis=1),
            "period_x_prediction": pd.concat([base, pd.DataFrame({f"{col}__period_x_rank": interaction})], axis=1),
            "full_period_within_interaction": pd.concat(
                [
                    base,
                    pd.DataFrame(
                        {
                            f"{col}__period_mean": period,
                            f"{col}__within_timestamp": within,
                            f"{col}__period_x_rank": interaction,
                        }
                    ),
                ],
                axis=1,
            ),
        }
        aucs = {
            name: _fit_logit_temporal_auc(
                x=x,
                y=y,
                timestamps=panel_high["timestamp"],
                max_rows=max_rows,
                seed=seed + idx * 13 + offset,
            )
            for offset, (name, x) in enumerate(matrices.items())
        }
        rows.append(
            {
                "head": head,
                "feature": col,
                "prediction_only_auc": aucs.get("prediction_only", np.nan),
                "period_state_auc": aucs.get("period_state", np.nan),
                "within_timestamp_state_auc": aucs.get("within_timestamp_state", np.nan),
                "period_x_prediction_auc": aucs.get("period_x_prediction", np.nan),
                "full_period_within_interaction_auc": aucs.get("full_period_within_interaction", np.nan),
                "delta_period_vs_prediction": aucs.get("period_state", np.nan) - aucs.get("prediction_only", np.nan),
                "delta_within_vs_prediction": aucs.get("within_timestamp_state", np.nan)
                - aucs.get("prediction_only", np.nan),
                "delta_interaction_vs_prediction": aucs.get("period_x_prediction", np.nan)
                - aucs.get("prediction_only", np.nan),
                "delta_full_vs_prediction": aucs.get("full_period_within_interaction", np.nan)
                - aucs.get("prediction_only", np.nan),
            }
        )
    return rows


def _duplicate_groups(frame: pd.DataFrame, *, method: str) -> dict[str, str]:
    numeric = frame.select_dtypes(include=[np.number])
    groups: dict[str, str] = {}
    if numeric.empty:
        return groups
    if method == "exact":
        hashes: dict[tuple[float, ...], list[str]] = {}
        for col in numeric.columns:
            arr = pd.to_numeric(numeric[col], errors="coerce").fillna(-999999.12345).to_numpy(dtype=np.float32)
            key = tuple(np.round(arr, 7).tolist())
            hashes.setdefault(key, []).append(col)
        for group_id, cols in enumerate([v for v in hashes.values() if len(v) > 1], start=1):
            label = f"exact_dup_{group_id}"
            for col in cols:
                groups[col] = label
        return groups
    corr = numeric.corr(method=method).abs()
    assigned: set[str] = set()
    group_id = 0
    for col in corr.columns:
        if col in assigned:
            continue
        peers = [peer for peer in corr.columns if peer != col and float(corr.loc[col, peer]) >= 0.999]
        if peers:
            group_id += 1
            label = f"{method}_dup_{group_id}"
            for peer in [col, *peers]:
                groups[peer] = label
                assigned.add(peer)
    return groups


def _archetype_alias_audit_rows(
    *,
    head: str,
    archetypes: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    exact = _duplicate_groups(archetypes, method="exact")
    pearson = _duplicate_groups(archetypes, method="pearson")
    spearman = _duplicate_groups(archetypes, method="spearman")
    archetype_diag = diagnostics.get("archetypes", {}) if isinstance(diagnostics, dict) else {}
    parent_by_column: dict[str, dict[str, Any]] = {}
    for archetype_name, payload in archetype_diag.items():
        if not isinstance(payload, dict):
            continue
        resolved_map = payload.get("resolved_feature_map", {}) if isinstance(payload.get("resolved_feature_map", {}), dict) else {}
        requested_features = float(payload.get("requested_features", 0) or 0)
        resolved_features = float(payload.get("resolved_features", 0) or 0)
        resolved_fraction = resolved_features / max(requested_features, 1.0)
        unresolved_fraction = max(0.0, 1.0 - resolved_fraction)
        fallback_fraction = 1.0 if not bool(payload.get("active", False)) else unresolved_fraction
        column_keys = ("score_column", "support_column", "probability_column")
        mapped_columns: list[str] = []
        for key in column_keys:
            col = str(payload.get(key, ""))
            if col:
                mapped_columns.append(col)
        for col in payload.get("alias_columns", []) or []:
            if str(col):
                mapped_columns.append(str(col))
        for col in dict.fromkeys(mapped_columns):
            parent_by_column[col] = {
                "archetype": archetype_name,
                "parents": sorted(set(map(str, resolved_map.values()))),
                "requested_features": int(requested_features),
                "resolved_features": int(resolved_features),
                "resolved_fraction": resolved_fraction,
                "fallback_fraction": fallback_fraction,
                "active": bool(payload.get("active", False)),
            }
    rows: list[dict[str, Any]] = []
    for col in archetypes.columns:
        ser = pd.to_numeric(archetypes[col], errors="coerce")
        parent = parent_by_column.get(col, {})
        rows.append(
            {
                "head": head,
                "output_feature": col,
                "resolved_parents": ",".join(parent.get("parents", [])),
                "source_archetype": parent.get("archetype", ""),
                "requested_features": parent.get("requested_features", np.nan),
                "resolved_features": parent.get("resolved_features", np.nan),
                "resolved_fraction": parent.get("resolved_fraction", np.nan),
                "fallback_fraction": parent.get("fallback_fraction", np.nan),
                "active_archetype": parent.get("active", np.nan),
                "exact_duplicate_group": exact.get(col, ""),
                "pearson_duplicate_group": pearson.get(col, ""),
                "spearman_duplicate_group": spearman.get(col, ""),
                "unique_values": int(ser.nunique(dropna=True)),
                "variance": float(ser.var(skipna=True)) if ser.notna().any() else np.nan,
                "heads_available": 1,
                "available_before_trade": True,
                "outcome_independent": True,
                "fold_fitted": False,
                "live_equivalent": True,
                "train_live_parity_validated": False,
            }
        )
    return rows


def _canonical_reduction_rows(definitions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "model_path_fragility_archetype": ("prediction_path_instability", "model_state"),
        "leverage_crowding_archetype": ("leverage_funding_crowding", "market_state"),
        "liquidity_stress_archetype": ("liquidity_participation_stress", "market_state"),
        "tail_volatility_stress_archetype": ("tail_volatility_stress", "market_state"),
        "relative_value_path_archetype": ("relative_value_dislocation", "market_state"),
        "market_breadth_archetype": ("breadth_market_state", "market_state"),
        "network_concentration_archetype": ("network_concentration", "market_state"),
    }
    rows: list[dict[str, Any]] = []
    for archetype_name, payload in definitions.items():
        canonical, family = mapping.get(archetype_name, (archetype_name.replace("_archetype", ""), "market_state"))
        deployable = payload.get("deployable_features", []) if isinstance(payload, dict) else []
        top_features = payload.get("top_features", []) if isinstance(payload, dict) else []
        rows.append(
            {
                "canonical_variable": canonical,
                "state_family": family,
                "source_archetype": archetype_name,
                "mechanism_channel": str(payload.get("mechanism_channel", "")) if isinstance(payload, dict) else "",
                "deployable_aliases": ",".join(map(str, deployable)),
                "top_parent_features": ",".join(map(str, top_features[:12])),
                "recommended_for_training": bool(canonical in {v[0] for v in mapping.values()}),
            }
        )
    extra_model_rows = [
        {
            "canonical_variable": "prediction_support_quality",
            "state_family": "model_state",
            "source_archetype": "model_path_fragility_archetype",
            "mechanism_channel": "model_path_fragility",
            "deployable_aliases": "model_support_gap_score",
            "top_parent_features": "leaf_count,leaf_support,support_gap",
            "recommended_for_training": True,
        },
        {
            "canonical_variable": "prediction_reconstruction_anomaly",
            "state_family": "model_state",
            "source_archetype": "model_path_fragility_archetype",
            "mechanism_channel": "model_path_fragility",
            "deployable_aliases": "model_reconstruction_anomaly_score",
            "top_parent_features": "dae_reconstruction_error,dae_b16,gmm_mahal",
            "recommended_for_training": True,
        },
        {
            "canonical_variable": "regime_similarity_or_novelty",
            "state_family": "model_state",
            "source_archetype": "model_path_fragility_archetype",
            "mechanism_channel": "model_path_fragility",
            "deployable_aliases": "regime_centroid_similarity_or_novelty",
            "top_parent_features": "regime_centroid_similarity_train,gmm_dist_center,cluster_entropy",
            "recommended_for_training": True,
        },
    ]
    return [*rows, *extra_model_rows]


def _summary_report(
    out_dir: Path,
    model_rows: pd.DataFrame,
    uni: pd.DataFrame,
    weekly: pd.DataFrame,
    diagnostics: list[dict[str, Any]],
    decomposition: pd.DataFrame,
    transfer: pd.DataFrame,
    shadow_policy: pd.DataFrame,
    intervention_recs: pd.DataFrame,
    audit: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Bad-Regime Archetype Usefulness")
    lines.append("")
    lines.append("Diagnostic-only report. No feature-store or model artifacts were modified.")
    lines.append("")
    if not model_rows.empty:
        binary_rows = (
            model_rows.loc[model_rows["target_kind"].astype(str).eq("binary")].copy()
            if "target_kind" in model_rows.columns
            else model_rows
        )
        pivot_index = ["target", "head"] if "target" in binary_rows.columns else ["head"]
        pivot = binary_rows.pivot_table(index=pivot_index, columns="model", values="auc_mean", aggfunc="first")
        if {"prediction_controls_only", "prediction_plus_archetype"}.issubset(pivot.columns):
            pivot["delta_vs_prediction_controls"] = pivot["prediction_plus_archetype"] - pivot["prediction_controls_only"]
        if {"nuisance_controls_only", "nuisance_plus_archetype"}.issubset(pivot.columns):
            pivot["delta_vs_nuisance_controls"] = pivot["nuisance_plus_archetype"] - pivot["nuisance_controls_only"]
        lines.append("## Incremental Binary Classifier Signal")
        lines.append("")
        lines.append(pivot.reset_index().to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        continuous_rows = (
            model_rows.loc[model_rows["target_kind"].astype(str).eq("continuous")].copy()
            if "target_kind" in model_rows.columns
            else pd.DataFrame()
        )
        if not continuous_rows.empty:
            utility_cols = [
                "target",
                "head",
                "model",
                "utility_spearman_ic",
                "utility_pearson_ic",
                "utility_risk_gate_roc_auc",
                "utility_risk_gate_failure_capture_at_10pct_abstain",
                "utility_risk_gate_retained_return_mean_at_10pct_abstain",
                "utility_risk_gate_tail_loss_avoided_at_10pct_abstain",
                "utility_risk_gate_rejected_winner_cost_at_10pct_abstain",
            ]
            available_utility_cols = [c for c in utility_cols if c in continuous_rows.columns]
            lines.append("## Continuous Net Utility Signal")
            lines.append("")
            lines.append(continuous_rows[available_utility_cols].to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
        metric_cols = [
            "target",
            "target_kind",
            "head",
            "model",
            "auc_mean",
            "pr_auc",
            "log_loss",
            "brier",
            "utility_spearman_ic",
            "utility_risk_gate_roc_auc",
            "failure_capture_at_10pct_abstain",
            "utility_risk_gate_failure_capture_at_10pct_abstain",
            "retained_return_mean_at_10pct_abstain",
            "utility_risk_gate_retained_return_mean_at_10pct_abstain",
            "tail_loss_avoided_at_10pct_abstain",
            "utility_risk_gate_tail_loss_avoided_at_10pct_abstain",
            "rejected_winner_cost_at_10pct_abstain",
            "utility_risk_gate_rejected_winner_cost_at_10pct_abstain",
        ]
        available_metric_cols = [c for c in metric_cols if c in model_rows.columns]
        lines.append("## Failure-gate acceptance metrics")
        lines.append("")
        lines.append(model_rows[available_metric_cols].to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    if not transfer.empty:
        valid_transfer = transfer.loc[
            transfer["model"].astype(str).ne("__summary__")
            & _blank_reason_mask(transfer, "transfer_reason")
        ].copy()
        if not valid_transfer.empty:
            grouped = (
                valid_transfer.groupby(["target", "head", "model"], dropna=False)
                .agg(
                    heldout_episodes=("heldout_episode", "nunique"),
                    median_transfer_auc=("transfer_roc_auc", "median"),
                    mean_transfer_auc=("transfer_roc_auc", "mean"),
                    transfer_auc_positive_rate=(
                        "transfer_roc_auc",
                        lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") > 0.5)),
                    ),
                    median_failure_capture_10pct=(
                        "transfer_failure_capture_at_10pct_abstain",
                        "median",
                    ),
                    median_retained_return_10pct=(
                        "transfer_retained_return_mean_at_10pct_abstain",
                        "median",
                    ),
                    median_rejected_winner_cost_10pct=(
                        "transfer_rejected_winner_cost_at_10pct_abstain",
                        "median",
                    ),
                )
                .reset_index()
            )
            lines.append("## Leave-One-Episode-Out Transfer")
            lines.append("")
            lines.append(
                grouped.sort_values(
                    ["median_transfer_auc", "transfer_auc_positive_rate"],
                    ascending=False,
                )
                .head(40)
                .to_markdown(index=False, floatfmt=".4f")
            )
            lines.append("")
        summaries = transfer.loc[transfer["model"].astype(str).eq("__summary__")].copy()
        if not summaries.empty:
            lines.append("## Leave-One-Episode-Out Coverage Notes")
            lines.append("")
            lines.append(
                summaries[
                    [
                        c
                        for c in (
                            "head",
                            "target",
                            "bad_episode_count",
                            "heldout_episode",
                            "transfer_reason",
                        )
                        if c in summaries.columns
                    ]
                ].to_markdown(index=False)
            )
            lines.append("")
    if not shadow_policy.empty:
        top_shadow = (
            shadow_policy.sort_values(
                ["risk_sizing_score", "tail_loss_delta_q05", "success_minus_failure_exposure"],
                ascending=False,
            )
            .groupby(["head", "target"], as_index=False)
            .head(5)
        )
        lines.append("## Shadow Failure-Risk Sizing")
        lines.append("")
        lines.append(
            top_shadow[
                [
                    c
                    for c in (
                        "head",
                        "target",
                        "model",
                        "policy",
                        "avg_size",
                        "success_minus_failure_exposure",
                        "return_mean_delta",
                        "tail_loss_delta_q05",
                        "winner_haircut_mean",
                        "loser_loss_reduction_mean",
                        "risk_sizing_score",
                    )
                    if c in top_shadow.columns
                ]
            ].to_markdown(index=False, floatfmt=".5f")
        )
        lines.append("")
    if not intervention_recs.empty:
        lines.append("## Training Intervention Recommendations")
        lines.append("")
        lines.append(
            intervention_recs[
                [
                    c
                    for c in (
                        "head",
                        "target",
                        "action",
                        "model",
                        "policy",
                        "recommendation",
                        "decision_reason",
                        "recurrence_pass",
                        "incremental_lift_pass",
                        "economic_pass",
                        "delta_vs_prediction_controls",
                        "median_transfer_auc",
                        "risk_sizing_score",
                        "tail_loss_delta_q05",
                    )
                    if c in intervention_recs.columns
                ]
            ].to_markdown(index=False, floatfmt=".5f")
        )
        lines.append("")
    if not uni.empty:
        top = (
            uni.sort_values(["auc_lift_abs", "top20_minus_rest_fail_rate"], ascending=False)
            .groupby("head", as_index=False)
            .head(8)
        )
        lines.append("## Top univariate archetype scores")
        lines.append("")
        lines.append(
            top[
                [
                    "head",
                    "feature",
                    "auc_high_score_bad",
                    "auc_lift_abs",
                    "top20_minus_rest_fail_rate",
                    "direction",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")
    if not weekly.empty:
        wk = weekly.loc[weekly["feature"] != "__summary__"].copy()
        if not wk.empty:
            top_wk = wk.sort_values(["directional_auc", "bad_minus_normal_score"], ascending=False).groupby("head", as_index=False).head(5)
            lines.append("## Bad-week separation")
            lines.append("")
            lines.append(
                top_wk[
                    [
                        "head",
                        "feature",
                        "bad_week_count",
                        "bad_week_auc_high_score_bad",
                        "bad_minus_normal_score",
                    ]
                ].to_markdown(index=False, floatfmt=".4f")
            )
            lines.append("")
    if diagnostics:
        diag_df = pd.DataFrame(diagnostics)
        lines.append("## Archetype resolution")
        lines.append("")
        lines.append(
            diag_df[
                [
                    "head",
                    "output_feature_count",
                    "active_archetypes",
                    "inactive_archetypes",
                    "mean_resolved_fraction",
                ]
            ].to_markdown(index=False, floatfmt=".3f")
        )
        lines.append("")
    if not decomposition.empty:
        top_dec = decomposition.sort_values("delta_full_vs_prediction", ascending=False).groupby("head", as_index=False).head(6)
        lines.append("## Period Versus Within-Period Decomposition")
        lines.append("")
        lines.append(
            top_dec[
                [
                    "head",
                    "feature",
                    "prediction_only_auc",
                    "period_state_auc",
                    "within_timestamp_state_auc",
                    "period_x_prediction_auc",
                    "full_period_within_interaction_auc",
                    "delta_full_vs_prediction",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")
    if not audit.empty:
        dup = audit.loc[
            audit["exact_duplicate_group"].astype(str).ne("")
            | audit["pearson_duplicate_group"].astype(str).ne("")
            | audit["spearman_duplicate_group"].astype(str).ne("")
        ]
        if not dup.empty:
            lines.append("## Alias / Duplicate Audit")
            lines.append("")
            lines.append(
                dup[
                    [
                        "head",
                        "output_feature",
                        "exact_duplicate_group",
                        "pearson_duplicate_group",
                        "spearman_duplicate_group",
                        "unique_values",
                        "variance",
                    ]
                ]
                .head(30)
                .to_markdown(index=False, floatfmt=".6f")
            )
            lines.append("")
    if not canonical.empty:
        lines.append("## Canonical Reduction")
        lines.append("")
        lines.append(
            canonical[
                [
                    "canonical_variable",
                    "state_family",
                    "source_archetype",
                    "recommended_for_training",
                ]
            ].to_markdown(index=False)
        )
        lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("")
    lines.append("- These are post-hoc diagnostic CV scores on OOF rows, not final production validation.")
    lines.append("- Prediction controls are deployable-ish OOF prediction/rank features; nuisance controls add time ordinal and rows per timestamp.")
    lines.append("- A useful archetype should improve `prediction_plus_archetype` over `prediction_controls_only`; robustness is stronger if it also improves over nuisance controls.")
    lines.append("- Alias columns and ranked aliases should not be fed directly into training; use the canonical reduction table.")
    lines.append("- `fold_fitted=False` in the alias audit means these post-hoc scores still need train-fold fitting before production use.")
    (out_dir / "bad_regime_archetype_usefulness_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument("--transform-cache", default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet")
    parser.add_argument("--regime-root", default="extreme_price_movements/unsupervised_regime_learning")
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--definitions", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_synthesis_clean_contract_v1/soft_archetype_definitions.json")
    parser.add_argument("--output-dir", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument(
        "--failure-targets",
        default="high_conf_miss,high_conf_negative_net_pnl,high_conf_tail_loss,prediction_minus_outcome,continuous_net_utility",
        help="Comma-separated failure targets to estimate, or 'all'.",
    )
    parser.add_argument("--classifier-max-rows", type=int, default=80000)
    parser.add_argument(
        "--skip-leave-one-episode-out",
        action="store_true",
        help="Skip leave-one-bad-episode-out transfer diagnostics.",
    )
    parser.add_argument(
        "--episode-transfer-max-train-rows",
        type=int,
        default=60000,
        help="Maximum non-heldout high-confidence rows used to fit each transfer model.",
    )
    parser.add_argument(
        "--episode-transfer-max-episodes",
        type=int,
        default=8,
        help="Maximum recent bad episodes to hold out one at a time.",
    )
    parser.add_argument(
        "--episode-transfer-targets",
        default="high_conf_miss,high_conf_tail_loss,prediction_minus_outcome",
        help="Comma-separated binary targets for leave-one-episode-out transfer, or 'all'.",
    )
    parser.add_argument(
        "--episode-transfer-models",
        default="prediction_controls_only,prediction_plus_archetype,market_state_archetypes,model_support_variables,support_x_market_interactions",
        help="Comma-separated model families for leave-one-episode-out transfer, or 'all'.",
    )
    parser.add_argument(
        "--skip-shadow-risk-head",
        action="store_true",
        help="Skip diagnostic shadow failure-risk head and smooth sizing evaluation.",
    )
    parser.add_argument(
        "--shadow-risk-max-rows",
        type=int,
        default=80000,
        help="Maximum high-confidence rows used for shadow risk-head OOF scoring.",
    )
    parser.add_argument(
        "--shadow-risk-targets",
        default="high_conf_miss,high_conf_tail_loss,high_conf_negative_net_pnl",
        help="Comma-separated binary targets for shadow failure-risk heads, or 'all'.",
    )
    parser.add_argument(
        "--shadow-risk-models",
        default="prediction_plus_archetype,model_support_variables,market_state_archetypes,support_x_market_interactions",
        help="Comma-separated model families for shadow risk heads, or 'all'.",
    )
    parser.add_argument("--recent-weeks", type=int, default=8)
    parser.add_argument("--min-week-rows", type=int, default=200)
    parser.add_argument("--max-regime-columns", type=int, default=80)
    parser.add_argument("--only-head", nargs="*", default=[])
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = _ensure_dir(Path(args.output_dir))
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    report_dir = Path(args.report_dir)
    feature_dir = Path(args.feature_dir)
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    regime_context = Path(args.regime_context) if args.regime_context else _latest_regime_context(Path(args.regime_root))
    definitions = load_bad_regime_archetype_definitions(args.definitions)
    if not definitions:
        raise SystemExit(f"No archetype definitions loaded from {args.definitions}")

    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    if args.only_head:
        wanted = {str(v).strip() for v in args.only_head if str(v).strip()}
        heads = [
            h
            for h in heads
            if h.head in wanted
            or h.strategy_id in wanted
            or h.meta_key in wanted
            or any(token in h.head or token in h.strategy_id or token in h.meta_key for token in wanted)
        ]
    requested_targets = {
        str(value).strip()
        for value in str(args.failure_targets).split(",")
        if str(value).strip()
    }
    if not requested_targets or "all" in requested_targets:
        requested_targets = {
            "high_conf_miss",
            "high_conf_negative_net_pnl",
            "high_conf_tail_loss",
            "prediction_minus_outcome",
            "continuous_net_utility",
        }
    episode_transfer_targets = {
        str(value).strip()
        for value in str(args.episode_transfer_targets).split(",")
        if str(value).strip()
    }
    if not episode_transfer_targets or "all" in episode_transfer_targets:
        episode_transfer_targets = set(requested_targets)
    episode_transfer_models = {
        str(value).strip()
        for value in str(args.episode_transfer_models).split(",")
        if str(value).strip()
    }
    if "all" in episode_transfer_models:
        episode_transfer_models = set()
    shadow_risk_targets = {
        str(value).strip()
        for value in str(args.shadow_risk_targets).split(",")
        if str(value).strip()
    }
    if not shadow_risk_targets or "all" in shadow_risk_targets:
        shadow_risk_targets = set(requested_targets)
    shadow_risk_models = {
        str(value).strip()
        for value in str(args.shadow_risk_models).split(",")
        if str(value).strip()
    }
    if "all" in shadow_risk_models:
        shadow_risk_models = set()

    base_bundle: dict[str, Any] | None = None
    base_path = baseline_artifact_dir / "base_models_intermediate.pkl"
    if base_path.exists():
        with base_path.open("rb") as fh:
            base_bundle = pickle.load(fh)

    symbol_columns = _feature_store_union(feature_dir)
    model_rows: list[dict[str, Any]] = []
    univariate_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    shadow_metric_rows: list[dict[str, Any]] = []
    shadow_oof_frames: list[pd.DataFrame] = []
    alias_audit_rows: list[dict[str, Any]] = []
    config = BadRegimeArchetypeFeatureConfig(
        trailing_window=int(args.trailing_window),
        min_periods=int(args.min_periods),
        min_resolved_features=int(args.min_resolved_features),
    )

    for head in heads:
        print(f"[archetype_usefulness] processing {head.head}", flush=True)
        panel = pd.read_parquet(head.meta_oof_path)
        panel = _normalise_keys(panel)
        panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
        race = meta_models[head.meta_key]
        selected_x, coverage, coverage_summary = _assemble_selected_matrix(
            panel=panel,
            race=race,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
        coverage.insert(0, "head", head.head)
        coverage.to_csv(out_dir / f"{head.head}_meta_selected_feature_coverage.csv", index=False)

        base_selected_x = pd.DataFrame(index=panel.index)
        if base_bundle is not None:
            _, base_features = _base_models_for_head(base_bundle, head)
            if base_features:
                fake_race = type("FakeRace", (), {})()
                fake_best = type("FakeBest", (), {})()
                fake_best.selected_features = list(base_features)
                fake_best.get_training_meta_features = lambda: pd.DataFrame(index=panel.index)
                fake_best.model_effectiveness_history_defaults_ = {}
                fake_best.feature_stats_train = {}
                fake_race.best_model = fake_best
                base_selected_x, base_cov, _base_cov_summary = _assemble_selected_matrix(
                    panel=panel,
                    race=fake_race,
                    feature_dir=feature_dir,
                    transform_cache=transform_cache,
                    symbol_columns=symbol_columns,
                )
                base_cov.insert(0, "head", head.head)
                base_cov.to_csv(out_dir / f"{head.head}_base_selected_feature_coverage.csv", index=False)

        export_x = _known_export_features(panel)
        regime_x = _read_regime_features(regime_context, panel[["timestamp", "symbol"]], args.max_regime_columns)
        selected_parts = [selected_x]
        if not base_selected_x.empty:
            extra_base = [c for c in base_selected_x.columns if c not in selected_x.columns]
            if extra_base:
                selected_parts.append(base_selected_x[extra_base])
        candidate_x = _merge_feature_candidates(
            pd.concat(selected_parts, axis=1, copy=False),
            export_x,
            regime_x,
        )
        feature_contract = _candidate_feature_contract(candidate_x)
        feature_contract.insert(0, "head", head.head)
        feature_contract.insert(1, "strategy_id", head.strategy_id)
        feature_contract.to_csv(out_dir / f"{head.head}_candidate_feature_contract.csv", index=False)
        score_input = pd.concat(
            [panel[["timestamp", "symbol"]].reset_index(drop=True), candidate_x.reset_index(drop=True)],
            axis=1,
            copy=False,
        )
        archetypes, diagnostics = build_bad_regime_archetype_feature_frame(
            score_input,
            definitions,
            config=config,
        )
        archetypes.to_parquet(out_dir / f"{head.head}_archetype_scores.parquet", index=False, compression="zstd")
        alias_audit_rows.extend(
            _archetype_alias_audit_rows(
                head=head.head,
                archetypes=archetypes,
                diagnostics=diagnostics,
            )
        )
        arch_diag = diagnostics.get("archetypes", {}) if isinstance(diagnostics, dict) else {}
        active = [k for k, v in arch_diag.items() if isinstance(v, dict) and bool(v.get("active", False))]
        resolved_fracs = [
            float(v.get("resolved_features", 0)) / max(float(v.get("requested_features", 1)), 1.0)
            for v in arch_diag.values()
            if isinstance(v, dict)
        ]
        diagnostics_rows.append(
            {
                "head": head.head,
                "strategy_id": head.strategy_id,
                "candidate_feature_count": int(candidate_x.shape[1]),
                "output_feature_count": int(archetypes.shape[1]),
                "active_archetypes": int(len(active)),
                "inactive_archetypes": int(len(arch_diag) - len(active)),
                "mean_resolved_fraction": float(np.nanmean(resolved_fracs)) if resolved_fracs else np.nan,
                "coverage_mean_finite_fraction": float(coverage_summary.get("mean_finite_fraction", np.nan)),
            }
        )
        (out_dir / f"{head.head}_archetype_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True, default=_json_default)
        )

        high_mask = pd.to_numeric(panel["oof_rank_pct"], errors="coerce") >= float(args.rank_threshold)
        if not bool(high_mask.any()):
            continue
        panel_high = panel.loc[high_mask].reset_index(drop=True)
        archetypes_high = archetypes.loc[high_mask].reset_index(drop=True)
        candidate_x_high = candidate_x.loc[high_mask].reset_index(drop=True)
        targets = [target for target in _failure_targets(panel_high) if str(target.get("name")) in requested_targets]
        target_by_name = {str(target.get("name")): target for target in targets}
        primary = target_by_name.get("high_conf_miss") or (targets[0] if targets else None)
        if primary is None:
            continue
        y = np.asarray(primary["values"], dtype=np.float32)
        univariate_rows.extend(
            _univariate_archetype_rows(
                head=head.head,
                y=y,
                archetypes=archetypes_high,
            )
        )
        for target in targets:
            target_meta = {
                "target_definition": str(target.get("definition", "")),
                "target_threshold": target.get("target_threshold", np.nan),
            }
            model_rows.extend(
                _score_models(
                    head=head.head,
                    panel_high=panel_high,
                    target_name=str(target.get("name", "")),
                    target_kind=str(target.get("kind", "binary")),
                    y=np.asarray(target["values"], dtype=np.float32),
                    target_meta=target_meta,
                    archetypes_high=archetypes_high,
                    candidate_x_high=candidate_x_high,
                    max_rows=int(args.classifier_max_rows),
                    seed=int(args.seed),
                )
            )
            if (
                not bool(args.skip_leave_one_episode_out)
                and str(target.get("kind", "binary")) == "binary"
                and str(target.get("name", "")) in episode_transfer_targets
            ):
                transfer_rows.extend(
                    _leave_one_episode_transfer_rows(
                        head=head.head,
                        panel=panel,
                        panel_high=panel_high,
                        target_name=str(target.get("name", "")),
                        target_kind=str(target.get("kind", "binary")),
                        y=np.asarray(target["values"], dtype=np.float32),
                        archetypes_high=archetypes_high,
                        candidate_x_high=candidate_x_high,
                        rank_threshold=float(args.rank_threshold),
                        recent_weeks=int(args.recent_weeks),
                        min_week_rows=int(args.min_week_rows),
                        max_train_rows=int(args.episode_transfer_max_train_rows),
                        max_episodes=int(args.episode_transfer_max_episodes),
                        model_allowlist=set(episode_transfer_models),
                        seed=int(args.seed),
                    )
                )
            if (
                not bool(args.skip_shadow_risk_head)
                and str(target.get("kind", "binary")) == "binary"
                and str(target.get("name", "")) in shadow_risk_targets
            ):
                matrices = _model_family_matrices(
                    panel_high=panel_high,
                    archetypes_high=archetypes_high,
                    candidate_x_high=candidate_x_high,
                )
                realized_return = _pick_realized_return(panel_high)
                for model_name, x_shadow in matrices.items():
                    if shadow_risk_models and model_name not in shadow_risk_models:
                        continue
                    x_shadow = x_shadow.replace([np.inf, -np.inf], np.nan)
                    x_shadow = x_shadow.loc[
                        :,
                        [
                            c
                            for c in x_shadow.columns
                            if pd.to_numeric(x_shadow[c], errors="coerce").notna().mean() > 0.02
                        ],
                    ]
                    if x_shadow.empty:
                        shadow_metric_rows.append(
                            {
                                "head": head.head,
                                "target": str(target.get("name", "")),
                                "model": model_name,
                                "rows": int(len(panel_high)),
                                "folds": 0,
                                "reason": "empty_matrix",
                            }
                        )
                        continue
                    shadow_summary, shadow_oof = _fit_shadow_failure_risk_oof(
                        head=head.head,
                        target_name=str(target.get("name", "")),
                        model_name=model_name,
                        x=x_shadow,
                        y=np.asarray(target["values"], dtype=np.float32),
                        timestamps=panel_high["timestamp"],
                        realized_return=realized_return,
                        max_rows=int(args.shadow_risk_max_rows),
                        seed=int(args.seed)
                        + (
                            sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(model_name)))
                            % 997
                        ),
                    )
                    shadow_metric_rows.append(shadow_summary)
                    if not shadow_oof.empty:
                        shadow_oof_frames.append(shadow_oof)
        decomposition_rows.extend(
            _period_within_rows(
                head=head.head,
                panel_high=panel_high,
                y=y,
                archetypes_high=archetypes_high,
                max_rows=int(args.classifier_max_rows),
                seed=int(args.seed),
            )
        )
        weekly_rows.extend(
            _weekly_rows(
                head=head.head,
                panel=panel,
                archetypes=archetypes,
                rank_threshold=float(args.rank_threshold),
                min_week_rows=int(args.min_week_rows),
                recent_weeks=int(args.recent_weeks),
            )
        )

    model_df = pd.DataFrame(model_rows)
    uni_df = pd.DataFrame(univariate_rows)
    weekly_df = pd.DataFrame(weekly_rows)
    diag_df = pd.DataFrame(diagnostics_rows)
    decomposition_df = pd.DataFrame(decomposition_rows)
    transfer_df = pd.DataFrame(transfer_rows)
    shadow_metrics_df = pd.DataFrame(shadow_metric_rows)
    shadow_oof_df = pd.concat(shadow_oof_frames, axis=0, ignore_index=True) if shadow_oof_frames else pd.DataFrame()
    shadow_policy_df = _evaluate_smooth_risk_scalers(shadow_oof_df)
    intervention_recs_df = _training_intervention_recommendations(
        model_rows=model_df,
        transfer=transfer_df,
        shadow_policy=shadow_policy_df,
        decomposition=decomposition_df,
    )
    audit_df = pd.DataFrame(alias_audit_rows)
    if not audit_df.empty and {"output_feature", "head"}.issubset(audit_df.columns):
        heads_available = audit_df.groupby("output_feature")["head"].nunique().to_dict()
        audit_df["heads_available"] = audit_df["output_feature"].map(heads_available).astype("int16")
    canonical_df = pd.DataFrame(_canonical_reduction_rows(definitions))
    model_df.to_csv(out_dir / "archetype_model_auc_summary.csv", index=False)
    uni_df.to_csv(out_dir / "archetype_univariate_usefulness.csv", index=False)
    weekly_df.to_csv(out_dir / "archetype_bad_week_separation.csv", index=False)
    diag_df.to_csv(out_dir / "archetype_resolution_summary.csv", index=False)
    decomposition_df.to_csv(out_dir / "archetype_period_within_decomposition.csv", index=False)
    transfer_df.to_csv(out_dir / "archetype_leave_one_episode_transfer.csv", index=False)
    shadow_metrics_df.to_csv(out_dir / "shadow_failure_risk_head_summary.csv", index=False)
    if not shadow_oof_df.empty:
        shadow_oof_df.to_parquet(out_dir / "shadow_failure_risk_oof_scores.parquet", index=False, compression="zstd")
    else:
        pd.DataFrame().to_parquet(out_dir / "shadow_failure_risk_oof_scores.parquet", index=False)
    shadow_policy_df.to_csv(out_dir / "shadow_failure_risk_policy_eval.csv", index=False)
    intervention_recs_df.to_csv(out_dir / "training_intervention_recommendations.csv", index=False)
    audit_df.to_csv(out_dir / "archetype_alias_resolution_audit.csv", index=False)
    canonical_df.to_csv(out_dir / "canonical_archetype_reduction.csv", index=False)
    _summary_report(
        out_dir,
        model_df,
        uni_df,
        weekly_df,
        diagnostics_rows,
        decomposition_df,
        transfer_df,
        shadow_policy_df,
        intervention_recs_df,
        audit_df,
        canonical_df,
    )
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=_json_default))
    print(f"[archetype_usefulness] wrote report to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
