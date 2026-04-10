"""
simple_position_sizer.py

A lightweight, diagnostic-first module that evaluates whether upstream meta-model heads
are economically sufficient before any complex policy optimization. It answers:
"Can these upstream inputs be used to generate a profit proxy?"

Focuses on:
1. Stage 1: head-level diagnostics
2. Stage 2: small-combo race
3. Simple Ridge-based sizer using only meta-model heads as inputs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, Ridge, RidgeClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler

from extreme_price_movements.metrics import _stable_equity_and_drawdown
from extreme_price_movements.offline_optimisers.params_store import (
    load_inference_candidate_mask_params_per_bucket,
)
from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlanner,
    SlicePlannerConfig,
)
from extreme_price_movements.position_sizer_v2_metrics import (
    compute_bucket_monotonicity,
    compute_false_safe_rate,
    compute_top_slice_metrics,
)
from extreme_price_movements.run_ridge_sizer import (
    load_base_oof_predictions,
    load_meta_oof_predictions,
    load_trade_outcomes,
)
from extreme_price_movements.src_utils_tprint import tprint

logger = logging.getLogger(__name__)


def _strategy_params_path(data_root: str, run_id: str) -> Path:
    return (
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "strategy_params.json"
    )


def _frozen_strategy_thresholds_path(data_root: str, run_id: str) -> Path:
    return (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "frozen_strategy_thresholds.json"
    )


def _extract_strategy_params_payload(
    *,
    run_id: str,
    cost_pct: float,
    strategy_results: Dict[str, Any],
) -> Dict[str, Any]:
    strategies: List[Dict[str, Any]] = []
    for strategy_id, res in strategy_results.items():
        opt_table = res.get("profit_proxy_table_", pd.DataFrame())
        if not isinstance(opt_table, pd.DataFrame) or opt_table.empty:
            continue
        if "is_optimal" in opt_table.columns:
            opt_rows = opt_table[opt_table["is_optimal"]]
            opt_row = opt_rows.iloc[0] if not opt_rows.empty else opt_table.iloc[0]
        else:
            opt_row = opt_table.sort_values("wallet_pnl", ascending=False).iloc[0]

        strategy_meta = res.get("_strategy_meta_", {})
        side = str(strategy_meta.get("trade_side", strategy_meta.get("side", "")) or "")
        source_target = str(strategy_meta.get("source_target", "") or "")
        source_horizon = strategy_meta.get("source_horizon", np.nan)
        row = {
            "strategy_id": str(strategy_id),
            "side": side,
            "threshold_pct": _to_float_or_nan(opt_row.get("threshold_pct", np.nan)),
            "selection_frac": _to_float_or_nan(opt_row.get("selection_frac", np.nan)),
            "wallet_pnl": _to_float_or_nan(opt_row.get("wallet_pnl", np.nan)),
            "net_pnl": _to_float_or_nan(opt_row.get("net_pnl", np.nan)),
            "pnl_per_trade": _to_float_or_nan(opt_row.get("pnl_per_trade", np.nan)),
            "profit_factor": _to_float_or_nan(opt_row.get("profit_factor", np.nan)),
            "hit_rate": _to_float_or_nan(opt_row.get("hit_rate", np.nan)),
            "trades_per_day": _to_float_or_nan(opt_row.get("trades_per_day", np.nan)),
            "monthly_sortino": _to_float_or_nan(opt_row.get("monthly_sortino", np.nan)),
            "monthly_pnl_std": _to_float_or_nan(opt_row.get("monthly_pnl_std", np.nan)),
            "monthly_group_cv_pnl": _to_float_or_nan(
                opt_row.get("monthly_group_cv_pnl", np.nan)
            ),
            "asset_group_cv_pnl": _to_float_or_nan(
                opt_row.get("asset_group_cv_pnl", np.nan)
            ),
            "asset_group_positive_share": _to_float_or_nan(
                opt_row.get("asset_group_positive_share", np.nan)
            ),
            "stability": _to_float_or_nan(opt_row.get("stability", np.nan)),
            "max_drawdown": _to_float_or_nan(opt_row.get("max_drawdown", np.nan)),
            "source_target": source_target,
            "source_horizon": _to_float_or_nan(source_horizon),
        }
        strategies.append(row)

    strategies = sorted(
        strategies,
        key=lambda x: (
            float(x.get("net_pnl", float("-inf"))),
            float(x.get("profit_factor", float("-inf"))),
            float(x.get("hit_rate", float("-inf"))),
        ),
        reverse=True,
    )

    payload = {
        "schema_version": "v1",
        "generated_by": "simple_position_sizer",
        "run_id": str(run_id),
        "fee_pct": float(cost_pct),
        "strategies": strategies,
        "buckets": {str(row["strategy_id"]): dict(row) for row in strategies},
        "best_strategy_id": strategies[0]["strategy_id"] if strategies else None,
        "best_threshold_pct": float(strategies[0]["threshold_pct"])
        if strategies
        else None,
    }
    return payload


def _save_strategy_params_payload(
    *,
    data_root: str,
    run_id: str,
    cost_pct: float,
    strategy_results: Dict[str, Any],
) -> Path | None:
    payload = _extract_strategy_params_payload(
        run_id=run_id,
        cost_pct=cost_pct,
        strategy_results=strategy_results,
    )
    if not payload["strategies"]:
        return None
    for path in (
        _strategy_params_path(data_root, run_id),
        _frozen_strategy_thresholds_path(data_root, run_id),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return _strategy_params_path(data_root, run_id)


def _persist_head_to_head_winner(
    data_root: str,
    run_id: str,
    strategy_results: Dict[str, Any],
) -> None:
    comparison_rows: List[Dict[str, Any]] = []
    et_winner_rows: List[Dict[str, Any]] = []
    for strategy_id, res in strategy_results.items():
        comp = res.get("comparison_", {})
        if not comp:
            continue
        comparison_rows.append({"strategy_id": strategy_id, **comp})
        if comp.get("winner") == "et":
            et_profit = res.get("et_profit_proxy_table_", pd.DataFrame())
            if et_profit.empty:
                continue
            if "is_optimal" in et_profit.columns:
                opt = et_profit[et_profit["is_optimal"]].iloc[0]
            else:
                opt = et_profit.sort_values("wallet_pnl", ascending=False).iloc[0]
            meta = res.get("_strategy_meta_", {})
            et_winner_rows.append(
                {
                    "strategy_id": str(strategy_id),
                    "side": str(meta.get("trade_side", "")),
                    "threshold_pct": _to_float_or_nan(opt.get("threshold_pct", np.nan)),
                    "selection_frac": _to_float_or_nan(
                        opt.get("selection_frac", np.nan)
                    ),
                    "wallet_pnl": _to_float_or_nan(opt.get("wallet_pnl", np.nan)),
                    "net_pnl": _to_float_or_nan(opt.get("net_pnl", np.nan)),
                    "profit_factor": _to_float_or_nan(opt.get("profit_factor", np.nan)),
                    "hit_rate": _to_float_or_nan(opt.get("hit_rate", np.nan)),
                    "model_source": "et",
                }
            )

    if comparison_rows:
        comp_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "ridge_sizer"
            / "head_to_head_comparison.json"
        )
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        comp_path.write_text(json.dumps(comparison_rows, indent=2, default=str))

    if et_winner_rows:
        et_params_path = (
            Path(data_root) / "artifacts" / run_id / "et_sizer" / "strategy_params.json"
        )
        et_params_path.parent.mkdir(parents=True, exist_ok=True)
        et_payload = {
            "schema_version": "v1",
            "generated_by": "simple_position_sizer",
            "run_id": run_id,
            "fee_pct": 0.003,
            "strategies": et_winner_rows,
        }
        et_params_path.write_text(json.dumps(et_payload, indent=2))
        logger.info(
            f"Persisted ET winner params ({len(et_winner_rows)} strategies) to {et_params_path}"
        )


def _to_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        if not s:
            return float("nan")
        try:
            return float(s)
        except Exception:
            return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


_RIDGE_HEAD_EXACT = {
    "reg",
    "reg_mean",
    "reg_std",
    "reg_range",
    "reg_sign_agree",
    "reg_clf_agree",
    "oof_pred",
    "oof_pred_oriented",
    "clf",
    "oof_ev",
    "oof_u_hat",
    "oof_log_mae_q70_hat",
    "oof_log_mfe_hat",
    "oof_asym_hat",
}
_RIDGE_HEAD_PREFIXES = (
    "base_h",
    "tbm_",
    "mae_h",
    "mfe_h",
    "oof_p_",
)


def collect_ridge_head_columns(df: pd.DataFrame) -> List[str]:
    """Return prediction-head columns that should feed the Ridge stack."""
    cols: List[str] = []
    for col in df.columns:
        if col in _RIDGE_HEAD_EXACT:
            cols.append(col)
            continue
        col_l = col.lower()
        if any(col_l.startswith(prefix) for prefix in _RIDGE_HEAD_PREFIXES):
            cols.append(col)
    return cols


def detect_meta_head_keys(
    feature_dict: Dict[str, np.ndarray], config_overrides: Optional[List[str]] = None
) -> Dict[str, str]:
    """Detects likely meta-model heads from the feature dictionary and classifies them."""
    if config_overrides:
        keys = [k for k in config_overrides if k in feature_dict]
    else:
        keys = list(feature_dict.keys())

    heads = {}
    for k in keys:
        kl = k.lower()
        # Exact OOF head names first so we do not confuse them with regime features
        # like `regime_*`.
        if kl in {
            "reg",
            "reg_mean",
            "reg_std",
            "oof_pred",
            "oof_pred_oriented",
            "reg_range",
            "reg_sign_agree",
            "reg_clf_agree",
            "oof_ev",
            "oof_u_hat",
            "oof_log_mae_q70_hat",
            "oof_log_mfe_hat",
            "oof_asym_hat",
        } or kl.startswith("base_h"):
            heads[k] = "return-like"
        elif kl in {"clf", "oof_p_tp", "oof_p_to", "oof_p_sl"}:
            heads[k] = "classification-like"
        if (
            "edge" in kl
            or "expected_return" in kl
            or "regressor" in kl
            or "reg_head" in kl
        ):
            heads[k] = "return-like"
        elif "mae" in kl or "downside" in kl or "risk" in kl:
            heads[k] = "risk-like"
        elif "mfe" in kl or "upside" in kl:
            heads[k] = "upside-like"
        elif "asym" in kl:
            heads[k] = "asymmetry-like"
        elif "uncert" in kl or "confid" in kl:
            heads[k] = "uncertainty-like"
        elif (
            "prob" in kl
            or "logit" in kl
            or "class" in kl
            or "meta_clf" in kl
            or "multi_obj" in kl
        ):
            heads[k] = "classification-like"

    # Include keys that were requested via override but missed the heuristic (if any).
    if config_overrides:
        for k in config_overrides:
            if k in feature_dict and k not in heads:
                heads[k] = "unknown"

    return heads


def clean_and_standardize(
    X: np.ndarray,
    fit_medians: Optional[np.ndarray] = None,
    scaler: Optional[RobustScaler] = None,
    center_1d: Optional[float] = None,
    scale_1d: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    """Standardizes features safely handling NaNs and Infs, using robust statistics."""
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    if fit_medians is None:
        fit_medians = np.nanmedian(X_clean, axis=0)
        if np.isscalar(fit_medians):
            if np.isnan(fit_medians):
                fit_medians = 0.0
        else:
            fit_medians[np.isnan(fit_medians)] = 0.0

    if X_clean.ndim == 1:
        inds = np.isnan(X_clean)
        X_clean[inds] = fit_medians

        if center_1d is None or scale_1d is None:
            center_1d = np.median(X_clean)
            q75, q25 = np.percentile(X_clean, [75, 25])
            scale_1d = q75 - q25

        if scale_1d > 1e-9:
            X_clean = (X_clean - center_1d) / scale_1d
        else:
            X_clean = X_clean - center_1d
    else:
        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(fit_medians, inds[1])

        if scaler is None:
            scaler = RobustScaler()
            X_clean = scaler.fit_transform(X_clean)
        else:
            X_clean = scaler.transform(X_clean)

    return X_clean, fit_medians, scaler, center_1d, scale_1d


def walk_forward_temporal_splits(
    timestamps: Optional[np.ndarray],
    n_samples: int,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
    embargo_pct: float = 0.01,
    symbols: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates strict walk-forward temporal splits using SlicePlanner.
    Ensures zero temporal leakage with proper purge and symbol-aware grouping.
    """
    if n_samples <= 0:
        return []

    ts = timestamps if timestamps is not None and len(timestamps) > 0 else None
    if ts is None:
        indices = np.arange(n_samples)
        logger.warning("No timestamps provided; falling back to positional splits.")
        splits = []
        start_idx = int(n_samples * min_train_frac)
        test_chunk_size = (n_samples - start_idx) // n_splits
        for i in range(n_splits):
            test_start = start_idx + i * test_chunk_size
            test_end = test_start + test_chunk_size if i < n_splits - 1 else n_samples
            if test_start < n_samples:
                splits.append((indices[:test_start], indices[test_start:test_end]))
        return (
            splits
            if splits
            else [(indices[: int(n_samples * 0.8)], indices[int(n_samples * 0.8) :])]
        )

    ts_parsed = pd.to_datetime(ts, utc=True, errors="coerce")
    sym_vals = symbols if symbols is not None else np.repeat("ALL", n_samples)

    events = pd.DataFrame(
        {
            "event_id": np.arange(n_samples, dtype=np.int64),
            "symbol": sym_vals,
            "t0": ts_parsed,
            "t1": ts_parsed + pd.Timedelta(hours=6),
        }
    )

    p_cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
    p_cfg = p_cfg.__class__(
        **{
            **p_cfg.__dict__,
            "preset": p_cfg.preset.__class__(
                preset_name=p_cfg.preset.preset_name,
                outer=p_cfg.preset.outer,
                inner=p_cfg.preset.inner.__class__(n_splits=max(2, n_splits)),
                sampling=p_cfg.preset.sampling,
                symbol_policy=p_cfg.preset.symbol_policy,
                purge_policy=p_cfg.preset.purge_policy,
            ),
            "silent": True,
            "min_rows_per_fold": 1,
            "min_symbols_per_fold": 1,
        }
    )

    try:
        bundle = SlicePlanner(p_cfg).build(events)
        splits = [
            (plan.fit_idx, plan.predict_idx)
            for plan in bundle["consumer_plans"]["ridge_sizer_fit"]
            if plan.tag == "predict_outer_test"
            and plan.fit_idx.size > 0
            and plan.predict_idx.size > 0
        ]
        if splits:
            return splits
    except Exception as e:
        logger.warning(f"SlicePlanner failed ({e}), falling back to positional splits.")

    indices = np.argsort(ts_parsed.values)
    start_idx = int(n_samples * min_train_frac)
    test_chunk_size = (n_samples - start_idx) // n_splits
    embargo_size = int(n_samples * embargo_pct)
    splits = []
    for i in range(n_splits):
        test_start = start_idx + i * test_chunk_size
        test_end = test_start + test_chunk_size if i < n_splits - 1 else n_samples
        train_end = test_start - embargo_size
        if train_end > 0 and test_start < n_samples:
            splits.append((indices[:train_end], indices[test_start:test_end]))
    if not splits:
        train_end = int(n_samples * 0.8)
        splits.append((indices[:train_end], indices[train_end:]))
    return splits


def evaluate_signal(
    name: str,
    scores: np.ndarray,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    directionality: str,
) -> Dict[str, Any]:
    """
    Evaluates a single signal or formula.
    Inverts the signal if it's risk-like so that higher score always means better expected outcome.
    """
    eval_scores = scores.copy()
    if directionality == "risk-like":
        # Lower risk is better, so invert it for evaluation
        eval_scores = -eval_scores

    metrics = {"head_name": name, "directionality": directionality}

    # Spearman with returns
    try:
        corr, _ = spearmanr(eval_scores, y_raw_net_return, nan_policy="omit")
        metrics["spearman_ret"] = float(corr) if pd.notna(corr) else 0.0
    except Exception:
        metrics["spearman_ret"] = 0.0

    # Top slice metrics
    top_metrics = compute_top_slice_metrics(
        eval_scores, y_raw_net_return, top_fracs=(0.1, 0.2)
    )
    metrics.update(top_metrics)

    # Bucket monotonicity
    metrics["monotonicity"] = compute_bucket_monotonicity(
        eval_scores, y_raw_net_return, n_buckets=10
    )

    # Downside false safe
    # Note: For false safe, we want to know if "safe" predictions (high eval_score) lead to high downside.
    # We pass -eval_scores so that lower values mean "predicted safe" for the helper logic
    # which assumes 'lower predicted downside == safer'.
    metrics["false_safe_rate"] = compute_false_safe_rate(
        -eval_scores, y_downside, low_q=0.2, high_q=0.8
    )

    # Calculate simple utility score for ranking:
    # Reward high top 10% returns, high monotonicity, low false safe rate.
    # Normalizing top 10% returns heuristically
    top10_ret = metrics.get("top_10_mean_net", 0.0)
    mono = max(0.0, metrics["monotonicity"])
    fs_penalty = metrics["false_safe_rate"]

    # Very simple empirical utility proxy:
    utility = (np.sign(top10_ret) * (np.abs(top10_ret) ** 0.5) * 10) + mono - fs_penalty
    metrics["utility_score"] = float(utility)

    return metrics


def compute_period_aggregated_stats(
    trade_rets: np.ndarray,
    trade_ts: Optional[np.ndarray],
    freq: str,
) -> Tuple[float, float]:
    """Return Sortino and standard deviation of period-aggregated PnL."""
    if trade_ts is None or len(trade_ts) == 0 or len(trade_rets) == 0:
        return 0.0, 0.0
    try:
        ts = pd.to_datetime(trade_ts, utc=True, errors="coerce")
        if isinstance(ts, pd.Series):
            valid = ts.notna().values
            ts_idx = pd.DatetimeIndex(ts[valid])
        else:
            valid = pd.notna(ts)
            ts_idx = pd.DatetimeIndex(ts[valid])
        if np.sum(valid) == 0:
            return 0.0, 0.0
        rets = np.asarray(trade_rets, dtype=float)[valid]
        period_vals = pd.Series(rets).groupby(ts_idx.to_period(freq)).sum().values
        if len(period_vals) == 0:
            return 0.0, 0.0
        neg = period_vals[period_vals < 0]
        downside_std = float(np.std(neg)) if len(neg) > 0 else 1e-6
        mean_ret = float(np.mean(period_vals))
        sortino = mean_ret / downside_std if downside_std > 1e-12 else 0.0
        return sortino, float(np.std(period_vals))
    except Exception:
        return 0.0, 0.0


def compute_group_stability_stats(
    trade_rets: np.ndarray,
    group_labels: Optional[np.ndarray],
) -> Dict[str, float]:
    """Return dispersion and consistency metrics for grouped trade PnL."""
    if group_labels is None or len(trade_rets) == 0:
        return {
            "group_mean_pnl": 0.0,
            "group_std_pnl": 0.0,
            "group_cv_pnl": 0.0,
            "group_positive_share": 0.0,
            "group_pf_std": 0.0,
        }

    try:
        vals = np.asarray(trade_rets, dtype=float)
        labels = np.asarray(group_labels)
        valid = np.isfinite(vals) & pd.notna(labels)
        if not np.any(valid):
            return {
                "group_mean_pnl": 0.0,
                "group_std_pnl": 0.0,
                "group_cv_pnl": 0.0,
                "group_positive_share": 0.0,
                "group_pf_std": 0.0,
            }
        vals = vals[valid]
        labels = labels[valid]
        group_df = pd.DataFrame({"label": labels, "ret": vals})
        agg = group_df.groupby("label")["ret"].agg(
            group_pnl="sum",
            group_hit_rate=lambda x: float(np.mean(x > 0)) if len(x) else 0.0,
            group_pf=lambda x: (
                float(np.sum(x[x > 0])) / float(np.abs(np.sum(x[x < 0])))
                if np.abs(np.sum(x[x < 0])) > 0
                else float(np.sum(x[x > 0]))
            ),
        )
        group_pnl = agg["group_pnl"].values.astype(float)
        if len(group_pnl) == 0:
            return {
                "group_mean_pnl": 0.0,
                "group_std_pnl": 0.0,
                "group_cv_pnl": 0.0,
                "group_positive_share": 0.0,
                "group_pf_std": 0.0,
            }
        group_mean = float(np.mean(group_pnl))
        group_std = float(np.std(group_pnl))
        group_cv = float(group_std / max(abs(group_mean), 1e-12))
        return {
            "group_mean_pnl": group_mean,
            "group_std_pnl": group_std,
            "group_cv_pnl": group_cv,
            "group_positive_share": float(np.mean(group_pnl > 0)),
            "group_pf_std": float(np.std(agg["group_pf"].values.astype(float))),
        }
    except Exception:
        return {
            "group_mean_pnl": 0.0,
            "group_std_pnl": 0.0,
            "group_cv_pnl": 0.0,
            "group_positive_share": 0.0,
            "group_pf_std": 0.0,
        }


def run_stage_1_diagnostics(
    feature_dict: Dict[str, np.ndarray],
    detected_heads: Dict[str, str],
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
) -> pd.DataFrame:
    """Runs single-head diagnostics for all detected meta heads."""
    results = []
    for head_key, head_type in detected_heads.items():
        if head_key not in feature_dict:
            continue
        scores = feature_dict[head_key]
        if len(scores) != len(y_raw_net_return):
            continue

        metrics = evaluate_signal(
            head_key, scores, y_raw_net_return, y_downside, head_type
        )
        results.append(metrics)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="utility_score", ascending=False).reset_index(drop=True)
    return df


def build_combo_candidates(
    feature_dict: Dict[str, np.ndarray],
    detected_heads: Dict[str, str],
    lambda_grid: List[float] = [0.25, 0.5, 1.0, 2.0],
) -> Dict[str, np.ndarray]:
    """
    Generates a small family of fixed-form score combinations from available heads.
    Uses basic normalization to combine disparate scales safely.
    """
    candidates = {}

    # Organize heads by type
    edge_heads = [
        k for k, v in detected_heads.items() if v == "return-like" and k in feature_dict
    ]
    mae_heads = [
        k for k, v in detected_heads.items() if v == "risk-like" and k in feature_dict
    ]
    mfe_heads = [
        k for k, v in detected_heads.items() if v == "upside-like" and k in feature_dict
    ]
    clf_heads = [
        k
        for k, v in detected_heads.items()
        if v == "classification-like" and k in feature_dict
    ]
    asym_heads = [
        k
        for k, v in detected_heads.items()
        if v == "asymmetry-like" and k in feature_dict
    ]

    def _norm(x):
        x_c, _, _, _, _ = clean_and_standardize(x)
        return x_c

    # 1. Base edge heads (always include if present)
    for eh in edge_heads:
        candidates[f"base_{eh}"] = _norm(feature_dict[eh])

    # 2. Edge - lambda * MAE
    if edge_heads and mae_heads:
        for eh in edge_heads:
            for mh in mae_heads:
                n_eh = _norm(feature_dict[eh])
                n_mh = _norm(feature_dict[mh])
                for lam in lambda_grid:
                    # Note: MAE is risk-like, so higher MAE score means higher predicted risk.
                    # We subtract lambda * MAE to penalize risk.
                    candidates[f"combo: {eh} - {lam}*{mh}"] = n_eh - lam * n_mh

    # 3. MFE - lambda * MAE
    if mfe_heads and mae_heads:
        for mfe_h in mfe_heads:
            for mae_h in mae_heads:
                n_mfe = _norm(feature_dict[mfe_h])
                n_mae = _norm(feature_dict[mae_h])
                for lam in lambda_grid:
                    candidates[f"combo: {mfe_h} - {lam}*{mae_h}"] = n_mfe - lam * n_mae

    # 4. Edge / (MAE + eps) style ratio (using standardized but shifted positive)
    if edge_heads and mae_heads:
        for eh in edge_heads:
            for mh in mae_heads:
                n_eh = _norm(feature_dict[eh])
                n_mh = _norm(feature_dict[mh])
                # Shift mae to be positive so ratio makes sense
                pos_mh = n_mh - np.min(n_mh) + 1.0
                candidates[f"combo: {eh} / ({mh}+eps)"] = n_eh / pos_mh

    # 5. Edge + alpha * classification
    if edge_heads and clf_heads:
        for eh in edge_heads:
            for ch in clf_heads:
                n_eh = _norm(feature_dict[eh])
                n_ch = _norm(feature_dict[ch])
                for lam in lambda_grid:
                    candidates[f"combo: {eh} + {lam}*{ch}"] = n_eh + lam * n_ch

    # 6. Include asym combinations if present
    if edge_heads and asym_heads:
        for eh in edge_heads:
            for ah in asym_heads:
                n_eh = _norm(feature_dict[eh])
                n_ah = _norm(feature_dict[ah])
                for lam in lambda_grid:
                    # Assuming higher asym prediction is better
                    candidates[f"combo: {eh} + {lam}*{ah}"] = n_eh + lam * n_ah

    return candidates


def run_stage_2_combo_race(
    candidates: Dict[str, np.ndarray],
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs race evaluation across all combinations."""
    results = []

    for name, scores in candidates.items():
        if len(scores) != len(y_raw_net_return):
            continue

        # Combos are pre-aligned so higher score = better expected outcome.
        # We pass directionality "return-like" because we built them that way.
        metrics = evaluate_signal(
            name, scores, y_raw_net_return, y_downside, directionality="return-like"
        )

        # Calculate fold-level stability
        if splits:
            fold_spearmans = []
            for tr_idx, te_idx in splits:
                if len(te_idx) > 0:
                    corr, _ = spearmanr(
                        scores[te_idx], y_raw_net_return[te_idx], nan_policy="omit"
                    )
                    if pd.notna(corr):
                        fold_spearmans.append(float(corr))
            if fold_spearmans:
                metrics["fold_spearman_mean"] = float(np.mean(fold_spearmans))
                metrics["fold_spearman_std"] = float(np.std(fold_spearmans))
            else:
                metrics["fold_spearman_mean"] = 0.0
                metrics["fold_spearman_std"] = 0.0

        # Rename head_name to combo_name for clarity
        metrics["combo_name"] = metrics.pop("head_name")
        results.append(metrics)

    df = pd.DataFrame(results)
    best_combo = {}
    if not df.empty:
        df = df.sort_values(by="utility_score", ascending=False).reset_index(drop=True)
        best_combo = df.iloc[0].to_dict()

    return df, best_combo


class SimpleHeadRidgeSizer:
    """
    A compact experimental component that tests if a linear model using
    only meta heads can beat fixed formulas.
    """

    def __init__(self, model=None):
        from sklearn.linear_model import Ridge

        self.model = model or Ridge(alpha=1.0)
        self.fold_coefs = []
        self.feature_names = []

    def fit_predict_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        feature_names: List[str] = None,
    ):
        """
        Fits locally on each train fold and predicts the next test fold.
        Stores coefficients for interpretability.
        """
        n_samples = len(y)
        oof_preds = np.zeros(n_samples)
        self.fold_coefs = []
        self.feature_names = feature_names or [f"head_{i}" for i in range(X.shape[1])]

        for tr_idx, te_idx in splits:
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te = X[te_idx]

            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
                continue

            # Fold-local scaling and NaN cleaning
            X_tr_clean, medians, scaler, center_1d, scale_1d = clean_and_standardize(
                X_tr
            )
            X_te_clean, _, _, _, _ = clean_and_standardize(
                X_te,
                fit_medians=medians,
                scaler=scaler,
                center_1d=center_1d,
                scale_1d=scale_1d,
            )

            # Fit & predict
            self.model.fit(X_tr_clean, y_tr)
            self.fold_coefs.append(self.model.coef_)
            oof_preds[te_idx] = self.model.predict(X_te_clean)

        return oof_preds

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns the mean weight (coefficient) per meta-head across folds."""
        if not self.fold_coefs:
            return pd.DataFrame()

        coef_matrix = np.array(self.fold_coefs)
        mean_coefs = np.mean(coef_matrix, axis=0)
        std_coefs = np.std(coef_matrix, axis=0)

        df = pd.DataFrame(
            {
                "head_name": self.feature_names,
                "mean_weight": mean_coefs,
                "std_weight": std_coefs,
                "abs_weight": np.abs(mean_coefs),
            }
        )
        return df.sort_values("abs_weight", ascending=False)


class SimpleHeadBarrierClassifier:
    """Dual-head classifier predicting triple-barrier outcomes: 0=SL, 1=TIME, 2=TP."""

    def __init__(
        self,
        alpha: float = 1.0,
        class_weight: Optional[str] = "balanced",
    ):
        self.alpha = alpha
        self.class_weight = class_weight
        self.fold_coefs: List[np.ndarray] = []
        self.feature_names: List[str] = []
        self.n_classes_: int = 3

    def fit_predict_oof_proba(
        self,
        X: np.ndarray,
        y_labels: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """OOF probability predictions for each barrier class.

        Returns:
            oof_proba: shape (n_samples, n_classes) — class probabilities
            oof_preds: shape (n_samples,) — hard class predictions
        """
        n_samples = len(y_labels)
        classes = np.unique(y_labels)
        self.n_classes_ = max(3, int(classes.max()) + 1)
        oof_proba = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        oof_preds = np.full(n_samples, -1, dtype=np.int8)
        self.fold_coefs = []
        self.feature_names = feature_names or [f"head_{i}" for i in range(X.shape[1])]

        for tr_idx, te_idx in splits:
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            X_tr, y_tr = X[tr_idx], y_labels[tr_idx]
            X_te = X[te_idx]
            if X_tr.shape[0] < 5 or X_te.shape[0] == 0:
                continue

            X_tr_clean, medians, scaler, center_1d, scale_1d = clean_and_standardize(
                X_tr
            )
            X_te_clean, _, _, _, _ = clean_and_standardize(
                X_te,
                fit_medians=medians,
                scaler=scaler,
                center_1d=center_1d,
                scale_1d=scale_1d,
            )

            base_clf = RidgeClassifier(alpha=self.alpha, class_weight=self.class_weight)
            calibrated = CalibratedClassifierCV(
                estimator=base_clf, cv=3, method="sigmoid"
            )
            try:
                calibrated.fit(X_tr_clean, y_tr)
                proba = calibrated.predict_proba(X_te_clean)
                preds = calibrated.predict(X_te_clean)
                if proba.shape[1] == self.n_classes_:
                    oof_proba[te_idx] = proba.astype(np.float32)
                else:
                    full_proba = np.zeros(
                        (len(te_idx), self.n_classes_), dtype=np.float32
                    )
                    for ci, cls in enumerate(calibrated.classes_):
                        col = int(cls)
                        if 0 <= col < self.n_classes_:
                            full_proba[:, col] = proba[:, ci]
                    oof_proba[te_idx] = full_proba
                oof_preds[te_idx] = preds.astype(np.int8)
                if hasattr(calibrated, "estimators_"):
                    est = calibrated.estimators_[0]
                    if hasattr(est, "coef_"):
                        self.fold_coefs.append(est.coef_)
            except Exception as e:
                logger.warning(f"Barrier classifier fold failed: {e}")
                oof_proba[te_idx, 1] = 1.0
                oof_preds[te_idx] = 1

        row_sums = oof_proba.sum(axis=1, keepdims=True)
        oof_proba = np.where(row_sums > 1e-6, oof_proba / row_sums, oof_proba)
        return oof_proba, oof_preds

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.fold_coefs:
            return pd.DataFrame()
        coef_matrix = np.array(self.fold_coefs)
        if coef_matrix.ndim == 3:
            mean_coefs = np.mean(np.abs(coef_matrix), axis=(0, 1))
        else:
            mean_coefs = np.mean(np.abs(coef_matrix), axis=0)
        df = pd.DataFrame(
            {
                "head_name": self.feature_names,
                "mean_abs_weight": mean_coefs,
            }
        )
        return df.sort_values("mean_abs_weight", ascending=False)


def calibrate_barrier_probabilities(
    oof_proba: np.ndarray,
    y_labels: np.ndarray,
    method: str = "platt",
) -> np.ndarray:
    """Post-hoc calibration of barrier probabilities.

    Applies Platt (sigmoid) scaling per-class using the full OOF predictions.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    n_samples, n_classes = oof_proba.shape
    calibrated = np.zeros_like(oof_proba)
    for cls in range(n_classes):
        y_binary = (y_labels == cls).astype(np.int32)
        pos_count = int(y_binary.sum())
        neg_count = n_samples - pos_count
        if pos_count < 10 or neg_count < 10:
            calibrated[:, cls] = oof_proba[:, cls]
            continue
        try:
            if method == "isotonic" and n_samples > 500:
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(oof_proba[:, cls], y_binary)
                calibrated[:, cls] = np.clip(cal.transform(oof_proba[:, cls]), 0.0, 1.0)
            else:
                cal = LogisticRegression(C=1.0, max_iter=500)
                cal.fit(oof_proba[:, cls].reshape(-1, 1), y_binary)
                calibrated[:, cls] = cal.predict_proba(
                    oof_proba[:, cls].reshape(-1, 1)
                )[:, 1]
        except Exception:
            calibrated[:, cls] = oof_proba[:, cls]
    row_sums = calibrated.sum(axis=1, keepdims=True)
    calibrated = np.where(row_sums > 1e-6, calibrated / row_sums, calibrated)
    return calibrated.astype(np.float32)


def evaluate_selection_profit_proxy(
    scores: np.ndarray,
    y_raw_net_return: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    top_fracs: List[float] = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3],
    start_equity: float = 100000.0,
    cost_pct: float = 0.003,
    n_days: float = 365.0,
    wallet_range: Tuple[float, float] = (0.05, 0.15),
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Evaluates "Could this generate profit?" using:
    1. A confidence rank threshold grid (95% down to 70%).
    2. Variable position sizing (e.g. 5% to 15% wallet allocation).
    """
    results = []
    n_samples = len(scores)

    if n_samples == 0:
        return pd.DataFrame(), np.array([]), np.array([])

    for frac in top_fracs:
        k = max(1, int(n_samples * frac))
        idx = np.argpartition(scores, -k)[-k:]

        selected_rets = y_raw_net_return[idx]
        selected_ts = (
            timestamps[idx]
            if timestamps is not None and len(timestamps) == n_samples
            else None
        )
        selected_syms = (
            symbols[idx] if symbols is not None and len(symbols) == n_samples else None
        )
        hit_rate = float(np.mean(selected_rets > 0)) if len(selected_rets) > 0 else 0.0

        # Apply cost (fees + slippage)
        sized_rets = selected_rets - cost_pct

        # --- 🏆 Advanced Sizing: Confidence-Weighted Allocation (5-15%) ---
        # We assign higher size to the highest scores within the k-selection.
        # Use rank-based sizing to be robust to score distributions.
        slice_scores = scores[idx]
        sorted_args = np.argsort(slice_scores)  # idx of scores in ascending order

        # Rankings from 5% (min score in slice) to 15% (max score in slice)
        allocations = np.linspace(wallet_range[0], wallet_range[1], len(idx))

        # Apply allocations to the sorted returns
        sorted_rets_net = selected_rets[sorted_args] - cost_pct
        wallet_rets = sorted_rets_net * allocations

        net_wallet_pnl_pct = float(np.sum(wallet_rets))

        wallet_sensitivity = {}
        for _wr_label, _wr in [
            ("narrow", (0.05, 0.10)),
            ("wide", (0.05, 0.20)),
            ("flat", (0.10, 0.10)),
        ]:
            _allocs = np.linspace(_wr[0], _wr[1], len(idx))
            _wallet_rets_s = sorted_rets_net * _allocs
            wallet_sensitivity[f"wallet_pnl_{_wr_label}"] = float(
                np.sum(_wallet_rets_s)
            )

        _, dd_series = _stable_equity_and_drawdown(sized_rets)
        mdd_pct = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
        net_pnl = float(np.sum(sized_rets))

        # Basic PF
        gross_profit = float(np.sum(sized_rets[sized_rets > 0]))
        gross_loss = float(np.abs(np.sum(sized_rets[sized_rets < 0])))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float(gross_profit)
        )

        # 📈 Advanced Risk Metrics

        # 1. Sortino Ratio (Downside-aware)
        downside_rets = sized_rets[sized_rets < 0]
        downside_std = np.std(downside_rets) if len(downside_rets) > 0 else 1e-6
        mean_ret = np.mean(sized_rets)
        sortino = mean_ret / downside_std

        weekly_sortino, weekly_pnl_std = compute_period_aggregated_stats(
            sized_rets, selected_ts, "W"
        )
        monthly_sortino, monthly_pnl_std = compute_period_aggregated_stats(
            sized_rets, selected_ts, "M"
        )
        month_groups = None
        if selected_ts is not None:
            ts_idx = pd.to_datetime(selected_ts, utc=True, errors="coerce")
            if isinstance(ts_idx, pd.Series):
                ts_idx = pd.DatetimeIndex(ts_idx[ts_idx.notna()])
            else:
                ts_idx = pd.DatetimeIndex(ts_idx[pd.notna(ts_idx)])
            if len(ts_idx) == len(sized_rets):
                month_groups = ts_idx.to_period("M").astype(str)
        month_stability = compute_group_stability_stats(sized_rets, month_groups)
        asset_stability = compute_group_stability_stats(sized_rets, selected_syms)

        # 2. PnL Stability (R-squared of linear equity curve)
        equity = np.cumsum(sized_rets)
        if len(equity) > 5:
            _, _, r_val, _, _ = linregress(np.arange(len(equity)), equity)
            stability = float(r_val**2)
        else:
            stability = 0.0

        # 3. Frequency & Efficiency
        pnl_per_trade_bps = (
            (net_pnl / len(sized_rets)) * 10000 if len(sized_rets) > 0 else 0.0
        )
        trades_per_day = len(sized_rets) / n_days if n_days > 0 else 0.0

        results.append(
            {
                "selection_frac": frac,
                "threshold_pct": f"{100*(1-frac):.1f}%",
                "net_pnl": net_pnl,
                "wallet_pnl": net_wallet_pnl_pct,
                "pnl_per_trade": pnl_per_trade_bps,
                "trades_per_day": trades_per_day,
                "hit_rate": hit_rate,
                "profit_factor": profit_factor,
                "sortino": sortino,
                "weekly_sortino": weekly_sortino,
                "monthly_sortino": monthly_sortino,
                "weekly_pnl_std": weekly_pnl_std,
                "monthly_pnl_std": monthly_pnl_std,
                "monthly_group_std_pnl": month_stability["group_std_pnl"],
                "monthly_group_cv_pnl": month_stability["group_cv_pnl"],
                "monthly_group_pf_std": month_stability["group_pf_std"],
                "asset_group_std_pnl": asset_stability["group_std_pnl"],
                "asset_group_cv_pnl": asset_stability["group_cv_pnl"],
                "asset_group_positive_share": asset_stability["group_positive_share"],
                "asset_group_pf_std": asset_stability["group_pf_std"],
                "stability": stability,
                "max_drawdown": mdd_pct,
                "trades_selected": len(sized_rets),
                **wallet_sensitivity,
            }
        )

    df = pd.DataFrame(results)

    # Identify optimal threshold by Wallet PnL (since user asked for 5-15% sizing)
    opt_rets = np.array([])
    opt_ts = np.array([])
    if not df.empty:
        opt_idx = df["wallet_pnl"].idxmax()
        df["is_optimal"] = False
        df.loc[opt_idx, "is_optimal"] = True

        # Recalculate k for the optimal frac to get indexed returns
        frac_opt = df.loc[opt_idx, "selection_frac"]
        k_opt = max(1, int(n_samples * frac_opt))
        idx_opt = np.argpartition(scores, -k_opt)[-k_opt:]
        opt_rets = y_raw_net_return[idx_opt] - cost_pct
        if timestamps is not None and len(timestamps) == n_samples:
            opt_ts = np.asarray(timestamps)[idx_opt]

    return df, opt_rets, opt_ts


def run_simple_position_sizer(
    feature_dict: Dict[str, np.ndarray],
    trade_outcomes: pd.DataFrame,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    timestamps: np.ndarray,
    symbols: Optional[np.ndarray] = None,
    bucket_labels: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    start_equity: float = 100000.0,
    cost_pct: float = 0.003,
    lambda_grid: Optional[List[float]] = None,
    top_fracs: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2),
    use_ridge_head_sizer: bool = True,
    use_et_head_sizer: bool = True,
    use_barrier_classifier: bool = True,
    config_feature_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for the simple position sizer diagnostic framework.
    By default runs both Ridge and ExtraTrees, compares them, and selects the best.
    """
    if lambda_grid is None:
        lambda_grid = [0.25, 0.5, 1.0, 2.0]

    detected_heads = detect_meta_head_keys(
        feature_dict, config_overrides=config_feature_keys
    )
    used_keys = [k for k in detected_heads.keys() if k in feature_dict]
    missing_keys = [k for k in detected_heads.keys() if k not in feature_dict]

    feature_coverage_report = {
        "detected_candidates": list(detected_heads.keys()),
        "used_heads": used_keys,
        "missing_heads": missing_keys,
        "head_classification": detected_heads,
    }

    stage_1_df = run_stage_1_diagnostics(
        feature_dict, detected_heads, y_raw_net_return, y_downside
    )

    n_samples = len(y_raw_net_return)
    splits = walk_forward_temporal_splits(
        timestamps, n_samples, n_splits=5, symbols=symbols
    )

    combo_candidates = build_combo_candidates(feature_dict, detected_heads, lambda_grid)
    stage_2_df, best_combo = run_stage_2_combo_race(
        combo_candidates, y_raw_net_return, y_downside, splits
    )

    best_simple_score = None
    best_simple_score_name = None

    if not stage_2_df.empty:
        best_simple_score_name = best_combo["combo_name"]
        best_simple_score = combo_candidates[best_simple_score_name]

    results: Dict[str, Any] = {}
    t_diff = np.max(timestamps) - np.min(timestamps)
    if (
        hasattr(t_diff, "astype")
        and not isinstance(t_diff, float)
        and not isinstance(t_diff, int)
        and not isinstance(t_diff, np.integer)
    ):
        try:
            n_days = float(t_diff / np.timedelta64(1, "D"))
        except Exception:
            n_days = float(t_diff) / 86400.0 if len(timestamps) > 1 else 0.0
    else:
        n_days = float(t_diff) / 86400.0 if len(timestamps) > 1 else 0.0

    _sym_vals = (
        trade_outcomes["symbol"].values if "symbol" in trade_outcomes.columns else None
    )

    # --- Ridge Sizer ---
    ridge_sizer_eval: Dict[str, Any] = {}
    ridge_importance_df = pd.DataFrame()
    ridge_profit_proxy_df = pd.DataFrame()

    if use_ridge_head_sizer and used_keys:
        X_heads = np.column_stack([feature_dict[k] for k in used_keys])
        sizer = SimpleHeadRidgeSizer()
        ridge_oof_preds = sizer.fit_predict_oof(
            X_heads, y_raw_net_return, splits, feature_names=used_keys
        )
        ridge_importance_df = sizer.get_feature_importance()
        ridge_metrics = evaluate_signal(
            "Ridge_Head_Sizer",
            ridge_oof_preds,
            y_raw_net_return,
            y_downside,
            directionality="return-like",
        )
        ridge_sizer_eval = ridge_metrics
        (
            ridge_profit_proxy_df,
            ridge_opt_rets,
            ridge_opt_ts,
        ) = evaluate_selection_profit_proxy(
            ridge_oof_preds,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        results["ridge_sizer_scores_"] = ridge_oof_preds
        results["ridge_importance_table_"] = ridge_importance_df
        results["ridge_profit_proxy_table_"] = ridge_profit_proxy_df
        results["ridge_opt_rets_"] = ridge_opt_rets
        results["ridge_opt_ts_"] = ridge_opt_ts
        if not best_combo or ridge_metrics.get("utility_score", 0) > best_combo.get(
            "utility_score", -9999
        ):
            best_simple_score = ridge_oof_preds
            best_simple_score_name = "Ridge_Head_Sizer"

    # --- Barrier Classifier (dual-head) ---
    barrier_clf_importance_df = pd.DataFrame()
    barrier_clf_oof_proba: Optional[np.ndarray] = None
    barrier_clf_oof_preds: Optional[np.ndarray] = None

    _tbm_labels = None
    if "tbm_label" in trade_outcomes.columns:
        _tbm_labels = np.asarray(trade_outcomes["tbm_label"].values, dtype=np.int8)
    elif all(c in trade_outcomes.columns for c in ("mfe_ret", "mae_ret", "return")):
        _mfe = np.abs(np.asarray(trade_outcomes["mfe_ret"].values, dtype=np.float32))
        _mae = np.abs(np.asarray(trade_outcomes["mae_ret"].values, dtype=np.float32))
        _barrier = np.clip(np.maximum(_mae * 2.5, 1e-4), 0.005, 0.2)
        _tp_dist = 0.50 * _barrier - cost_pct
        _sl_dist = 0.18 * _barrier + cost_pct
        _is_tp = _mfe >= np.maximum(_tp_dist, 1e-6)
        _is_sl = _mae >= np.maximum(_sl_dist, 1e-6)
        _tbm_labels = np.ones(n_samples, dtype=np.int8)
        _tbm_labels[_is_sl & ~_is_tp] = 0
        _tbm_labels[_is_tp] = 2

    if (
        use_barrier_classifier
        and used_keys
        and _tbm_labels is not None
        and len(np.unique(_tbm_labels)) >= 2
    ):
        if "X_heads" not in dir() or X_heads is None:
            X_heads = np.column_stack([feature_dict[k] for k in used_keys])
        barrier_clf = SimpleHeadBarrierClassifier()
        (
            barrier_clf_oof_proba,
            barrier_clf_oof_preds,
        ) = barrier_clf.fit_predict_oof_proba(
            X_heads, _tbm_labels, splits, feature_names=used_keys
        )
        barrier_clf_importance_df = barrier_clf.get_feature_importance()
        barrier_clf_oof_proba = calibrate_barrier_probabilities(
            barrier_clf_oof_proba, _tbm_labels
        )
        results["barrier_clf_oof_proba_"] = barrier_clf_oof_proba
        results["barrier_clf_oof_preds_"] = barrier_clf_oof_preds
        results["barrier_clf_importance_"] = barrier_clf_importance_df
        results["oof_p_tp_"] = barrier_clf_oof_proba[:, 2].astype(np.float32)
        results["oof_p_sl_"] = barrier_clf_oof_proba[:, 0].astype(np.float32)
        results["oof_p_time_"] = barrier_clf_oof_proba[:, 1].astype(np.float32)
        _class_counts = np.bincount(_tbm_labels.astype(int), minlength=3)
        tprint(
            f"Barrier Classifier: classes={dict(zip(['SL', 'TIME', 'TP'], _class_counts))} | "
            f"mean P(TP)={np.mean(barrier_clf_oof_proba[:, 2]):.3f} "
            f"P(SL)={np.mean(barrier_clf_oof_proba[:, 0]):.3f} "
            f"P(TIME)={np.mean(barrier_clf_oof_proba[:, 1]):.3f}"
        )
    et_sizer_eval: Dict[str, Any] = {}
    et_importance_df = pd.DataFrame()
    et_profit_proxy_df = pd.DataFrame()

    if use_et_head_sizer and used_keys:
        from extreme_price_movements.extratrees_position_sizer import (
            SimpleHeadExtraTreesSizer,
        )

        X_heads = (
            X_heads
            if use_ridge_head_sizer and used_keys
            else np.column_stack([feature_dict[k] for k in used_keys])
        )
        et_model = ExtraTreesRegressor(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=30,
            min_samples_split=60,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        et_sizer = SimpleHeadExtraTreesSizer(
            model=et_model, calibration_method="isotonic"
        )
        et_oof_preds = et_sizer.fit_predict_oof(
            X_heads, y_raw_net_return, splits, feature_names=used_keys
        )
        et_importance_df = et_sizer.get_feature_importance()
        et_metrics = evaluate_signal(
            "ExtraTrees_Head_Sizer",
            et_oof_preds,
            y_raw_net_return,
            y_downside,
            directionality="return-like",
        )
        et_sizer_eval = et_metrics
        et_profit_proxy_df, et_opt_rets, et_opt_ts = evaluate_selection_profit_proxy(
            et_oof_preds,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        results["et_sizer_scores_"] = et_oof_preds
        results["et_importance_table_"] = et_importance_df
        results["et_profit_proxy_table_"] = et_profit_proxy_df
        results["et_opt_rets_"] = et_opt_rets
        results["et_opt_ts_"] = et_opt_ts
        if et_metrics.get("utility_score", 0) > (
            ridge_sizer_eval.get("utility_score", -9999) if ridge_sizer_eval else -9999
        ):
            if not best_combo or et_metrics.get("utility_score", 0) > best_combo.get(
                "utility_score", -9999
            ):
                best_simple_score = et_oof_preds
                best_simple_score_name = "ExtraTrees_Head_Sizer"

    # --- Head-to-Head Comparison ---
    comparison: Dict[str, Any] = {}
    if ridge_sizer_eval and et_sizer_eval:
        ridge_util = ridge_sizer_eval.get("utility_score", 0.0)
        et_util = et_sizer_eval.get("utility_score", 0.0)
        ridge_wallet = (
            float(ridge_profit_proxy_df["wallet_pnl"].max())
            if not ridge_profit_proxy_df.empty
            else 0.0
        )
        et_wallet = (
            float(et_profit_proxy_df["wallet_pnl"].max())
            if not et_profit_proxy_df.empty
            else 0.0
        )
        comparison = {
            "ridge_utility": ridge_util,
            "et_utility": et_util,
            "ridge_best_wallet_pnl": ridge_wallet,
            "et_best_wallet_pnl": et_wallet,
            "winner": "ridge" if ridge_util >= et_util else "et",
            "wallet_winner": "ridge" if ridge_wallet >= et_wallet else "et",
        }
        tprint(
            f"Head-to-Head: Ridge util={ridge_util:.4f} wallet={ridge_wallet:.4f} | ET util={et_util:.4f} wallet={et_wallet:.4f} | Winner={comparison['winner']}"
        )
    results["comparison_"] = comparison

    # --- Profit Proxy on Best Score ---
    profit_proxy_df = pd.DataFrame()
    best_opt_rets = np.array([])
    best_opt_ts = np.array([])
    if best_simple_score is not None:
        profit_proxy_df, best_opt_rets, best_opt_ts = evaluate_selection_profit_proxy(
            best_simple_score,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs) + [0.3],
            start_equity=start_equity,
            cost_pct=cost_pct,
        )

    return {
        "feature_coverage_report_": feature_coverage_report,
        "head_diagnostics_table_": stage_1_df,
        "combo_race_table_": stage_2_df,
        "best_combo_": best_combo,
        "ridge_sizer_eval_": ridge_sizer_eval,
        "ridge_importance_table_": ridge_importance_df,
        "ridge_profit_proxy_table_": ridge_profit_proxy_df,
        "et_sizer_eval_": et_sizer_eval,
        "et_importance_table_": et_importance_df,
        "et_profit_proxy_table_": et_profit_proxy_df,
        "comparison_": comparison,
        "best_simple_score_": best_simple_score,
        "best_simple_score_name_": best_simple_score_name,
        "profit_proxy_table_": profit_proxy_df
        if not profit_proxy_df.empty
        else pd.DataFrame(),
        "opt_rets_": best_opt_rets,
        "opt_ts_": best_opt_ts,
        "barrier_clf_importance_": barrier_clf_importance_df,
        "barrier_clf_oof_proba_": barrier_clf_oof_proba,
        "oof_p_tp_": results.get("oof_p_tp_"),
        "oof_p_sl_": results.get("oof_p_sl_"),
        "oof_p_time_": results.get("oof_p_time_"),
    }


def run_bucketed_simple_position_sizer(
    feature_dict: Dict[str, np.ndarray],
    trade_outcomes: pd.DataFrame,
    y_raw_net_return: np.ndarray,
    y_downside: np.ndarray,
    timestamps: np.ndarray,
    bucket_labels: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    min_bucket_samples: int = 50,
    **kwargs,
) -> Dict[str, Any]:
    """
    Runs the simple position sizer independently per bucket.
    """
    # Run global first
    global_results = run_simple_position_sizer(
        feature_dict,
        trade_outcomes,
        y_raw_net_return,
        y_downside,
        timestamps,
        trade_outcomes["symbol"].values if "symbol" in trade_outcomes.columns else None,
        bucket_labels=None,
        sample_weight=sample_weight,
        **kwargs,
    )

    bucket_results = {}
    summary_rows = []

    unique_buckets = np.unique(bucket_labels[~pd.isna(bucket_labels)])

    for b in unique_buckets:
        mask = bucket_labels == b
        if np.sum(mask) < min_bucket_samples:
            continue

        b_feature_dict = {k: v[mask] for k, v in feature_dict.items()}
        b_trade_outcomes = trade_outcomes.iloc[mask].reset_index(drop=True)
        b_y_raw_net_return = y_raw_net_return[mask]
        b_y_downside = y_downside[mask]
        b_timestamps = timestamps[mask]
        b_sample_weight = sample_weight[mask] if sample_weight is not None else None

        b_res = run_simple_position_sizer(
            b_feature_dict,
            b_trade_outcomes,
            b_y_raw_net_return,
            b_y_downside,
            b_timestamps,
            b_trade_outcomes["symbol"].values
            if "symbol" in b_trade_outcomes.columns
            else None,
            bucket_labels=None,
            sample_weight=b_sample_weight,
            **kwargs,
        )
        bucket_results[b] = b_res

        # Build summary row
        summary_rows.append(
            {
                "bucket": b,
                "samples": np.sum(mask),
                "best_model_name": b_res.get("best_simple_score_name_"),
                "best_utility": b_res.get("best_combo_", {}).get("utility_score", 0.0),
            }
        )

    global_results["bucket_results"] = bucket_results
    global_results["bucket_summary_table_"] = pd.DataFrame(summary_rows)

    return global_results


def run_simple_position_sizer_from_artifacts(
    data_root: str,
    run_id: str,
    top_fracs: Tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2),
    use_ridge_head_sizer: bool = True,
    use_et_head_sizer: bool = True,
    top_n_strategies: int = 4,
    time_filter: Optional[Tuple[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs the simple position sizer directly on pipeline artifacts, executing the pipeline
    once per strategy loaded.

    Loads base model OOF predictions and filters strictly to the exact strategy mask
    (as optimized per-bucket) before running diagnostics independently for each strategy.

    Returns a dictionary mapping strategy_id to its respective position sizer results.
    """
    from extreme_price_movements.run_ridge_sizer import (
        load_meta_oof_predictions,
        load_trade_outcomes,
    )

    # Load dynamic strategies (which rules are active per bucket)
    # load_inference_candidate_mask_params_per_bucket returns top_n PER (side, horizon) group.
    # We load a large pool then take the true global top-N by score_for_best_params.
    _pool = load_inference_candidate_mask_params_per_bucket(
        top_n=99, ranking_metric="score_for_best_params"
    )

    if not _pool:
        logger.warning("No strategies loaded from params_store.")
        return {}

    # Deduplicate by strategy_id and take global top-N
    _seen_ids: set = set()
    strategies = []
    for s in _pool:
        sid = s.get("strategy_id", "")
        if sid and sid not in _seen_ids:
            _seen_ids.add(sid)
            strategies.append(s)
    strategies = strategies[:top_n_strategies]

    logger.info(
        f"Loaded {len(strategies)} strategies (global top-{top_n_strategies}). IDs: {[s.get('strategy_id', '')[:40] for s in strategies]}"
    )

    # Load base and meta OOFs
    base_oofs = load_base_oof_predictions(data_root, run_id)
    try:
        meta_oofs = load_meta_oof_predictions(data_root, run_id)
    except Exception as e:
        logger.warning(f"Could not load meta OOFs: {e}. Falling back to base-only.")
        meta_oofs = {}

    if not base_oofs and not meta_oofs:
        logger.warning(
            f"No OOFs found in {data_root}/artifacts/{run_id}. Checking both base/ and meta_oof/."
        )
        return {}

    # Supplement params_store strategies with any OOF-artifact buckets not already matched.
    # This ensures isolated runs (e.g. single-symbol or small-universe) are evaluated even
    # when their strategy IDs are absent from the full-universe params CSV.
    import re as _re_sizer

    _known_ids = {s.get("strategy_id", "") for s in strategies}
    _all_oof_keys = set(base_oofs.keys()) | set(meta_oofs.keys())
    for _oof_key in sorted(_all_oof_keys):
        _stripped = _re_sizer.sub(r"^(long|short)_", "", _oof_key)
        _side = (
            "long"
            if _oof_key.startswith("long_")
            else ("short" if _oof_key.startswith("short_") else "")
        )
        if _stripped not in _known_ids and _oof_key not in _known_ids:
            strategies.append({"strategy_id": _stripped, "trade_side": _side})
            _known_ids.add(_stripped)
            logger.info(
                f"OOF-derived strategy injected: side={_side!r} id={_stripped[:60]}"
            )

    strategy_results = {}

    for strategy in strategies:
        strategy_id = strategy.get("strategy_id", "")
        if not strategy_id:
            continue
        # --- 🚀 RESTORE FULL MASK COVERAGE ---
        # 1. Load the Base Labels (The ground-truth of discovery)
        labels_dir = Path(data_root) / "artifacts" / run_id / "labels"
        logger.warning(
            f"DEBUG_PATH labels_dir: {labels_dir.absolute()} (Exists: {labels_dir.exists()})"
        )
        strat_id_raw = strategy.get("strategy_id", "")

        # FUZZY RESOLVER: Find label file on disk
        # Names on disk use _ instead of . and may have different rounding
        full_df = pd.DataFrame()
        label_file = None

        if labels_dir.exists():
            all_label_files = list(labels_dir.glob("train_*.parquet"))
            import re

            def normalize(s):
                return re.sub(r"[^a-z0-9]", "", s.lower())

            target_norm = normalize(strat_id_raw)

            for f in all_label_files:
                if "_tight" in f.name or "_wide" in f.name or "_balanced" in f.name:
                    continue

                f_name_norm = normalize(f.stem.replace("train_", ""))
                # Check for inclusion or high similarity
                if target_norm in f_name_norm or f_name_norm in target_norm:
                    label_file = f
                    logger.warning(
                        f"Fuzzy matched (norm): {f.name} for {strat_id_raw[:40]}..."
                    )
                    break

            if not label_file:
                logger.warning(
                    f"Failed norm match for {strat_id_raw[:40]}... scanning tokens."
                )
                # Fallback to token intersection
                tokens = set(
                    normalize(strat_id_raw)
                )  # This is too granular, let's keep previous tokens but normalize them
                tokens = set(re.split(r"[^a-z0-9]", strat_id_raw.lower()))
                tokens.discard("")

                max_overlap = 0
                for f in all_label_files:
                    if "_tight" in f.name or "_wide" in f.name or "_balanced" in f.name:
                        continue
                    f_tokens = set(
                        re.split(r"[^a-z0-9]", f.stem.lower().replace("train_", ""))
                    )
                    f_tokens.discard("")
                    overlap = len(tokens.intersection(f_tokens))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_match = f

                min_recall = 0.6  # at least 60% of strategy tokens must appear in the label filename
                if (
                    best_match
                    and len(tokens) > 0
                    and (max_overlap / len(tokens)) >= min_recall
                ):
                    label_file = best_match
                    logger.warning(
                        f"Fuzzy matched (token): {label_file.name} (Overlap: {max_overlap})"
                    )

        if label_file and label_file.exists():
            full_df = pd.read_parquet(label_file)
            logger.info(f"Loaded full label coverage: {len(full_df)} rows.")
            if "__r_policy_net__" in full_df.columns:
                full_df["return"] = full_df["__r_policy_net__"]
            if "__ts__" in full_df.columns:
                full_df["timestamp"] = full_df["__ts__"]
            if "__symbol__" in full_df.columns:
                full_df["symbol"] = full_df["__symbol__"]
            if "__index__" in full_df.columns:
                full_df["index"] = full_df["__index__"]

            if time_filter is not None and "timestamp" in full_df.columns:
                _tf_start, _tf_end = time_filter
                _tf_ts = pd.to_datetime(full_df["timestamp"], utc=True, errors="coerce")
                _tf_mask = np.ones(len(full_df), dtype=bool)
                if _tf_start is not None:
                    _tf_mask &= _tf_ts >= pd.Timestamp(_tf_start)
                if _tf_end is not None:
                    _tf_mask &= _tf_ts < pd.Timestamp(_tf_end)
                full_df = full_df.loc[_tf_mask].reset_index(drop=True)
                logger.info(
                    f"Time filter [{_tf_start}, {_tf_end}): {len(full_df)} rows retained"
                )

        trade_side = str(strategy.get("trade_side", "") or "")

        # 2. Resolve OOF bucket by matching strategy_id prefix against meta_oof keys
        # meta_oofs keys look like: "short_compression_ratio_..." or "compression_ratio_..."
        # strategy_id looks like: "compression_ratio_1_0376017_dist_ema_fast_..."
        oof_df = pd.DataFrame()
        resolved_meta_key = None

        # Priority 1: exact side-prefixed match (e.g. "short_<strategy_id>")
        prefixed = f"{trade_side}_{strategy_id}" if trade_side else strategy_id
        if prefixed in meta_oofs:
            resolved_meta_key = prefixed
        else:
            # Priority 2: longest common prefix between strategy_id and each meta key (strip side prefix)
            import re as _re

            def _strip_side(k):
                return _re.sub(r"^(long|short)_", "", k)

            strat_norm = _re.sub(r"[^a-z0-9]", "", strategy_id.lower())
            best_key, best_score = None, 0
            for mk in meta_oofs.keys():
                mk_norm = _re.sub(r"[^a-z0-9]", "", _strip_side(mk).lower())
                # Score = length of matching prefix
                plen = 0
                for a, b in zip(strat_norm, mk_norm):
                    if a == b:
                        plen += 1
                    else:
                        break
                if plen > best_score:
                    best_score = plen
                    best_key = mk
            if best_key and best_score >= 20:  # require substantial prefix overlap
                resolved_meta_key = best_key
                logger.info(
                    f"Fuzzy matched strategy '{strategy_id[:40]}' to meta_oof key '{best_key[:40]}' (prefix_len={best_score})"
                )

        if resolved_meta_key:
            oof_df = meta_oofs[resolved_meta_key]
            inferred_side = str(oof_df.attrs.get("trade_side", "") or "")
            if not inferred_side:
                inferred_side = (
                    "long"
                    if str(resolved_meta_key).startswith("long_")
                    else "short"
                    if str(resolved_meta_key).startswith("short_")
                    else ""
                )
            if inferred_side and not trade_side:
                trade_side = inferred_side

        # 3. Join OOF onto the Full Labels to ensure 100% coverage
        # When full_df is available, it is always the left side (preserves all 1592 label hits)
        # OOF enriches it for the 267 scored rows; the rest get NaN prediction columns.
        if not full_df.empty:
            if not oof_df.empty and "index" in oof_df.columns:
                # Align on row-position index: meta_oof index 0..266 maps to its own rows
                # We join on timestamp+symbol when available for a robust link
                join_cols = [
                    c
                    for c in ["timestamp", "symbol"]
                    if c in full_df.columns and c in oof_df.columns
                ]
                if join_cols:
                    oof_clean = oof_df.drop(
                        columns=[
                            c
                            for c in ["return", "y_ret", "y_bin"]
                            if c in full_df.columns
                        ],
                        errors="ignore",
                    )
                    # Normalize timestamp timezones to avoid datetime64 tz mismatch
                    if "timestamp" in join_cols:
                        for _df in [full_df, oof_clean]:
                            if (
                                "timestamp" in _df.columns
                                and hasattr(_df["timestamp"].dtype, "tz")
                                and _df["timestamp"].dt.tz is not None
                            ):
                                _df["timestamp"] = _df["timestamp"].dt.tz_localize(None)
                    # If OOF covers only one symbol, restrict label rows to that symbol
                    # and join on timestamp alone (symbol formats may differ, e.g. BTC/USDT vs BTCUSDT)
                    if "symbol" in join_cols and "symbol" in oof_clean.columns:
                        oof_syms = set(oof_clean["symbol"].dropna().unique())
                        if len(oof_syms) == 1:
                            oof_sym = next(iter(oof_syms))

                            # Normalise symbol for comparison (strip / and spaces)
                            def _norm_sym(s):
                                return str(s).replace("/", "").replace(" ", "").upper()

                            oof_sym_norm = _norm_sym(oof_sym)
                            label_sym_col = (
                                full_df["symbol"]
                                if "symbol" in full_df.columns
                                else (
                                    full_df["__symbol__"]
                                    if "__symbol__" in full_df.columns
                                    else None
                                )
                            )
                            if hasattr(label_sym_col, "map"):
                                sym_mask = label_sym_col.map(_norm_sym) == oof_sym_norm
                                if sym_mask.sum() > 0:
                                    full_df = full_df[sym_mask].copy()
                                    logger.info(
                                        f"Single-symbol OOF ({oof_sym}): filtered labels to {sym_mask.sum()} matching rows"
                                    )
                            join_cols = ["timestamp"]  # drop symbol from join key
                    active_df = pd.merge(
                        full_df,
                        oof_clean,
                        on=join_cols,
                        how="left",
                        suffixes=("", "_oof"),
                    )
                else:
                    # Positional join — label rows in same order as oof
                    oof_clean = oof_df.drop(
                        columns=[
                            c
                            for c in ["return", "y_ret", "y_bin"]
                            if c in full_df.columns
                        ],
                        errors="ignore",
                    )
                    active_df = full_df.copy()
                    for col in oof_clean.columns:
                        if col not in active_df.columns:
                            vals = np.full(len(active_df), np.nan)
                            vals[: min(len(oof_clean), len(active_df))] = oof_clean[
                                col
                            ].values[: len(active_df)]
                            active_df[col] = vals
            else:
                # No OOF found — use full labels as-is (no prediction columns)
                active_df = full_df.copy()
        else:
            active_df = oof_df

        if active_df.empty:
            logger.warning(
                f"Could not resolve data for strategy {strategy_id[:60]}. Skipping."
            )
            continue

        if time_filter is not None and "timestamp" in active_df.columns:
            _tf_start, _tf_end = time_filter
            _tf_ts = pd.to_datetime(active_df["timestamp"], utc=True, errors="coerce")
            _tf_mask = np.ones(len(active_df), dtype=bool)
            if _tf_start is not None:
                _tf_mask &= _tf_ts >= pd.Timestamp(_tf_start)
            if _tf_end is not None:
                _tf_mask &= _tf_ts < pd.Timestamp(_tf_end)
            active_df = active_df.loc[_tf_mask].reset_index(drop=True)
            logger.info(
                f"OOF time filter [{_tf_start}, {_tf_end}): {len(active_df)} rows retained"
            )

        active_joined_df = active_df

        _sizer_cap = 50000
        if _sizer_cap > 0 and len(active_joined_df) > _sizer_cap:
            from extreme_price_movements.training import subsample_symbol_balanced

            _pre_sizer = len(active_joined_df)
            _sym_col = (
                "symbol" if "symbol" in active_joined_df.columns else "__symbol__"
            )
            _ts_col = (
                "timestamp" if "timestamp" in active_joined_df.columns else "__ts__"
            )
            active_joined_df = subsample_symbol_balanced(
                active_joined_df, _sizer_cap, symbol_col=_sym_col, ts_col=_ts_col
            )
            logger.info(
                f"Sizer symbol-balanced subsample {_pre_sizer} -> {len(active_joined_df)} (cap={_sizer_cap})"
            )

        # Get target outcomes
        trade_outcomes = load_trade_outcomes(data_root, run_id, active_joined_df)
        if (
            trade_outcomes is None
            or "return" not in trade_outcomes.columns
            or len(trade_outcomes) == 0
        ):
            logger.info(
                f"Skipping strategy {strategy_id}: could not load trade outcomes."
            )
            continue

        # Load and Filter OOF
        # Diagnostic: Strategy Funnel
        # 1. Labels Discovery (100% of discovered hits)
        # 2. OOF Scorable (only the hits used for training/validation OOF)
        # 3. Unscored Loss (dropout due to subsampling or non-resolution)

        n_raw_labels = len(full_df) if not full_df.empty else len(active_joined_df)
        n_matched_labels = len(active_joined_df)
        # Detect any OOF prediction column available (reg, clf, oof_pred, oof_prob, etc.)
        _oof_score_cols = [
            c
            for c in active_joined_df.columns
            if c in ("oof_prob", "oof_pred", "reg", "clf")
            or c.startswith(("tbm_", "mae_h", "mfe_h"))
        ]
        if _oof_score_cols:
            n_scorable = int(active_joined_df[_oof_score_cols[0]].notna().sum())
            scorable_mask = active_joined_df[_oof_score_cols[0]].notna()
            active_scored_df = active_joined_df[scorable_mask].copy()
        else:
            n_scorable = len(active_joined_df)
            scorable_mask = np.ones(len(active_joined_df), dtype=bool)
            active_scored_df = active_joined_df.copy()

        # Diagnostic: Mask Pass Rate
        print(f"\n" + "=" * 80)
        print(f" TARGETING STRATEGY: {strategy_id[:65]}...")
        print(f" " + "-" * 78)
        raw_to_matched = n_matched_labels / max(n_raw_labels, 1)
        matched_to_scorable = n_scorable / max(n_matched_labels, 1)
        print(f" STRATEGY FUNNEL (Coverage Restoration):")
        print(f"   [1] Raw label rows:    {n_raw_labels}")
        print(
            f"   [1b] Matched rows:     {n_matched_labels} "
            f"({raw_to_matched:.1%} of raw labels)"
        )
        print(
            f"   [2] OOF scorable rows: {n_scorable} "
            f"({matched_to_scorable:.1%} of matched rows)"
        )
        subsampling_loss = max(0, n_matched_labels - n_scorable)
        extra_scored = max(0, n_scorable - n_matched_labels)
        print(
            f"   [!] Subsampling Loss:  {subsampling_loss} trades dropped by technical subsampling (Step 3)."
        )
        if extra_scored > 0:
            print(
                f"   [!] Matched rows warning: {extra_scored} scored rows exceeded the matched label rows after join."
            )
        if n_matched_labels > n_raw_labels:
            print(
                f"   [!] Join expansion warning: matched rows exceeded raw label rows "
                f"by {n_matched_labels - n_raw_labels}. This usually means the join "
                f"key is not unique or the OOF frame was broadcast positionally."
            )
        # Detect symbol universe mismatch between labels and OOF
        _label_syms = (
            set(active_joined_df["symbol"].dropna().unique())
            if "symbol" in active_joined_df.columns
            else set()
        )
        _oof_syms = (
            set(oof_df["symbol"].dropna().unique())
            if (not oof_df.empty and "symbol" in oof_df.columns)
            else set()
        )
        _sym_overlap = _label_syms & _oof_syms
        if _oof_syms and not _sym_overlap:
            logger.warning(
                f"Strategy {strategy_id[:50]}: SYMBOL UNIVERSE MISMATCH — "
                f"label symbols {sorted(_label_syms)[:3]} vs OOF symbols {sorted(_oof_syms)[:3]}. "
                f"Meta OOF was trained on a different symbol scope. OOF predictions cannot be joined."
            )
        elif n_scorable < 500:
            logger.warning(
                f"Strategy {strategy_id[:50]}: only {n_scorable} OOF-scored rows out of "
                f"{n_matched_labels} matched label rows. Meta model was likely trained on a heavily "
                f"subsampled dataset — diagnostic results will have wide confidence intervals."
            )
        print(f"=" * 80 + "\n")

        # Now pass the SCORED df to the sizer logic.
        # Keep trade outcomes aligned with the same scored rows so score/return
        # arrays refer to the exact same universe.
        active_df = active_scored_df.reset_index(drop=True)
        trade_outcomes = trade_outcomes.loc[np.asarray(scorable_mask)].reset_index(
            drop=True
        )

        # Identify columns to use as heads.
        # STRICT RULE: never use realized-outcome columns as predictive heads — they are
        # hindsight by construction (MFE, MAE, bars_to_mfe, policy returns, labels).
        _HINDSIGHT_COLS = {
            "return",
            "is_long",
            "y_ret",
            "y_bin",
            "exit_code",
            "bars_to_mfe",
            "mae_ret",
            "mfe_ret",
            "u_policy",
            "u_policy_net",
        }

        # Any column wrapped in __ is a label-side realized outcome
        def _is_hindsight(col: str) -> bool:
            return col in _HINDSIGHT_COLS or (
                col.startswith("__") and col.endswith("__")
            )

        head_cols = [
            c for c in collect_ridge_head_columns(active_df) if not _is_hindsight(c)
        ]

        expected_families = {
            "base": any(c.lower().startswith("base_h") for c in head_cols),
            "classifier": any(
                c == "clf" or c.lower().startswith("oof_p_") for c in head_cols
            ),
            "mae": any(
                c.lower().startswith("mae_h") or c == "oof_log_mae_q70_hat"
                for c in head_cols
            ),
            "mfe": any(
                c.lower().startswith("mfe_h") or c == "oof_log_mfe_hat"
                for c in head_cols
            ),
            "reg": any(
                c in {"reg", "reg_mean", "oof_pred", "oof_pred_oriented"}
                for c in head_cols
            ),
            "asym": any(c == "oof_asym_hat" for c in head_cols),
        }

        # Remove columns that are all-NaN (e.g. unmatched OOF columns after left-join)
        head_cols = [c for c in head_cols if active_df[c].notna().any()]

        if not head_cols:
            logger.warning(
                f"Strategy {strategy_id[:50]}: no OOF prediction columns found after join. "
                f"This strategy has no meta_oof match — skipping to avoid hindsight evaluation."
            )
            continue
        missing_fams = [name for name, ok in expected_families.items() if not ok]
        if missing_fams:
            logger.warning(
                f"Strategy {strategy_id[:50]}: missing expected head families after join: {missing_fams}"
            )

        _MIN_SCORED_ROWS = 30
        if len(active_df) < _MIN_SCORED_ROWS:
            logger.warning(
                f"Strategy {strategy_id[:50]}: only {len(active_df)} scored rows after join "
                f"(minimum {_MIN_SCORED_ROWS} required). Likely a symbol universe mismatch — skipping."
            )
            continue

        y_raw_net_return = trade_outcomes["return"].values
        if y_raw_net_return.size == 0:
            logger.info(
                f"Skipping strategy {strategy_id}: empty aligned trade outcomes."
            )
            continue

        if "downside" in trade_outcomes.columns:
            y_downside = trade_outcomes["downside"].values
        elif "mae" in active_df.columns:
            y_downside = active_df["mae"].values
        else:
            y_downside = np.zeros_like(y_raw_net_return)

        timestamps = (
            active_df["timestamp"].values
            if "timestamp" in active_df.columns
            else np.zeros(len(y_raw_net_return))
        )

        feature_dict = {col: active_df[col].values for col in head_cols}

        # Run the pipeline for this specific strategy
        res = run_simple_position_sizer(
            feature_dict=feature_dict,
            trade_outcomes=trade_outcomes,
            y_raw_net_return=y_raw_net_return,
            y_downside=y_downside,
            timestamps=timestamps,
            symbols=trade_outcomes["symbol"].values
            if "symbol" in trade_outcomes.columns
            else None,
            bucket_labels=None,  # Running independently, no bucketing needed
            top_fracs=top_fracs,
            use_ridge_head_sizer=use_ridge_head_sizer,
            use_et_head_sizer=use_et_head_sizer,
        )

        res["_strategy_meta_"] = {
            "trade_side": trade_side,
            "source_target": strategy.get("source_target", ""),
            "source_horizon": strategy.get("source_horizon", np.nan),
        }
        strategy_results[strategy_id] = res

    strategy_params_path = _save_strategy_params_payload(
        data_root=data_root,
        run_id=run_id,
        cost_pct=0.003,
        strategy_results=strategy_results,
    )
    if strategy_params_path is not None:
        logger.info(f"Saved strategy params to {strategy_params_path}")

    _persist_head_to_head_winner(data_root, run_id, strategy_results)

    # Print Strategy Leaderboard after all strategies are processed
    if strategy_results:
        leaderboard_rows = []
        for sid, res in strategy_results.items():
            opt_table = res.get("profit_proxy_table_", pd.DataFrame())
            if not opt_table.empty:
                # Use the row marked as optimal
                if "is_optimal" in opt_table.columns:
                    opt_row = opt_table[opt_table["is_optimal"]].iloc[0]
                else:
                    opt_row = opt_table.iloc[0]

                leaderboard_rows.append(
                    {
                        "strategy_id": sid[:40] + "...",
                        "threshold": opt_row["threshold_pct"],
                        "wallet_pnl": opt_row["wallet_pnl"],
                        "net_pnl": opt_row["net_pnl"],
                        "pnl/trade(bps)": opt_row["pnl_per_trade"],
                        "trades/day": opt_row["trades_per_day"],
                        "hit_rate": opt_row["hit_rate"],
                        "pf": opt_row["profit_factor"],
                        "monthly_sortino": opt_row.get("monthly_sortino", np.nan),
                        "monthly_pnl_std": opt_row.get("monthly_pnl_std", np.nan),
                        "monthly_group_cv_pnl": opt_row.get(
                            "monthly_group_cv_pnl", np.nan
                        ),
                        "asset_group_cv_pnl": opt_row.get("asset_group_cv_pnl", np.nan),
                        "asset_group_positive_share": opt_row.get(
                            "asset_group_positive_share", np.nan
                        ),
                        "stability": opt_row["stability"],
                        "mdd": opt_row["max_drawdown"],
                    }
                )

        if leaderboard_rows:
            # --- PORTFOLIO AGGREGATION ---
            # Collect returns for all strategies with PF > 1.0
            portfolio_rets_list = []
            portfolio_ts_list = []
            portfolio_wallet_pnls = []
            max_days = 0
            for sid, res in strategy_results.items():
                pf = 0
                opt_table = res.get("profit_proxy_table_", pd.DataFrame())
                if not opt_table.empty:
                    opt_row = opt_table[opt_table["is_optimal"]].iloc[0]
                    pf = opt_row["profit_factor"]
                    w_pnl = opt_row["wallet_pnl"]

                if pf > 1.0:
                    strat_rets = res.get("opt_rets_", np.array([]))
                    strat_ts = res.get("opt_ts_", np.array([]))
                    if len(strat_rets) > 0:
                        portfolio_rets_list.append(strat_rets)
                        portfolio_wallet_pnls.append(w_pnl)
                        if len(strat_ts) == len(strat_rets):
                            portfolio_ts_list.append(strat_ts)

            if portfolio_rets_list:
                all_portfolio_rets = np.concatenate(portfolio_rets_list)
                all_portfolio_ts = (
                    np.concatenate(portfolio_ts_list)
                    if portfolio_ts_list
                    and len(portfolio_ts_list) == len(portfolio_rets_list)
                    else None
                )
                p_wallet_pnl_total = sum(portfolio_wallet_pnls)

                # Portfolio PnL
                p_pnl = float(np.sum(all_portfolio_rets))
                p_hit = float(np.mean(all_portfolio_rets > 0))
                p_gp = float(np.sum(all_portfolio_rets[all_portfolio_rets > 0]))
                p_gl = float(np.abs(np.sum(all_portfolio_rets[all_portfolio_rets < 0])))
                p_pf = p_gp / p_gl if p_gl > 0 else p_gp

                # Stability (approximate on concatenated curve)
                p_equity = np.cumsum(all_portfolio_rets)
                p_stab = 0.0
                if len(p_equity) > 5:
                    _, _, r_val, _, _ = linregress(np.arange(len(p_equity)), p_equity)
                    p_stab = float(r_val**2)

                # MDD
                _, p_dd_series = _stable_equity_and_drawdown(all_portfolio_rets)
                p_mdd = float(np.max(p_dd_series)) if len(p_dd_series) > 0 else 0.0

                p_weekly_sortino, p_weekly_pnl_std = compute_period_aggregated_stats(
                    all_portfolio_rets, all_portfolio_ts, "W"
                )
                p_monthly_sortino, p_monthly_pnl_std = compute_period_aggregated_stats(
                    all_portfolio_rets, all_portfolio_ts, "M"
                )

                leaderboard_rows.append(
                    {
                        "strategy_id": "[PORTFOLIO - Positive PF Only]",
                        "threshold": "Mixed",
                        "wallet_pnl": p_wallet_pnl_total,  # Need to calculate this
                        "net_pnl": p_pnl,
                        "pnl/trade(bps)": (p_pnl / len(all_portfolio_rets)) * 10000,
                        "trades/day": len(all_portfolio_rets) / 725.0,
                        "hit_rate": p_hit,
                        "pf": p_pf,
                        "monthly_sortino": p_monthly_sortino,
                        "monthly_pnl_std": p_monthly_pnl_std,
                        "monthly_group_cv_pnl": np.nan,
                        "asset_group_cv_pnl": np.nan,
                        "asset_group_positive_share": np.nan,
                        "stability": p_stab,
                        "mdd": p_mdd,
                    }
                )

            print("\n" + "=" * 110)
            print(
                " STRATEGY LEADERBOARD (Sorted by Net PnL - Optimal Confidence Threshold)"
            )
            print("=" * 110)
            leaderboard_df = pd.DataFrame(leaderboard_rows)
            # Sort non-portfolio rows only?
            is_port = leaderboard_df["strategy_id"].str.contains("PORTFOLIO")
            sorted_v = leaderboard_df[~is_port].sort_values("net_pnl", ascending=False)
            leaderboard_df = pd.concat([sorted_v, leaderboard_df[is_port]])

            print(leaderboard_df.to_string(index=False))
            print("=" * 110 + "\n")

    return strategy_results


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from extreme_price_movements.run_ridge_sizer import find_latest_run_id
    from extreme_price_movements.src_utils_tprint import tprint

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Run Simple Position Sizer Diagnostics"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to analyze (defaults to latest in artifacts/)",
    )
    parser.add_argument(
        "--data-root", type=str, default=".", help="Root directory for data/artifacts"
    )
    parser.add_argument(
        "--cost-pct",
        type=float,
        default=0.003,
        help="Cost per trade in decimal (default: 0.003 for 30bps)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help="Top N strategies to evaluate from params store",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ridge", action="store_true", help="Use only Ridge model")
    group.add_argument("--et", action="store_true", help="Use only ExtraTrees model")
    group.add_argument(
        "--compare", action="store_true", help="Run both and compare (default)"
    )

    args = parser.parse_args()

    # Determine which models to run (default to compare if none specified)
    use_ridge = (
        args.ridge
        or args.compare
        or (not args.ridge and not args.et and not args.compare)
    )
    use_et = (
        args.et or args.compare or (not args.ridge and not args.et and not args.compare)
    )

    data_root = args.data_root
    run_id = args.run_id

    if not run_id:
        try:
            run_id = find_latest_run_id(data_root)
            tprint(f"Detected latest run: {run_id}")
        except Exception as e:
            tprint(f"Error detecting latest run: {e}")
            sys.exit(1)

    tprint(f"Starting Simple Position Sizer for run: {run_id} (data_root: {data_root})")

    try:
        results = run_simple_position_sizer_from_artifacts(
            data_root=data_root,
            run_id=run_id,
            top_n_strategies=args.top_n,
            use_ridge_head_sizer=use_ridge,
            use_et_head_sizer=use_et,
        )

        if not results:
            tprint(
                "No strategy results produced. Check if base OOFs and params_store are populated."
            )
            sys.exit(0)

        for strategy_id, res in results.items():
            print("-" * 60)
            print(f"\n============================================================")
            print(f" STRATEGY: {strategy_id}")
            print(f"============================================================\n")

            if "head_diagnostics_table_" in res:
                print(f"\nTop 5 Meta-Heads by Utility:")
                print(res["head_diagnostics_table_"].head(5).to_string(index=False))

            if "ridge_importance_table_" in res:
                print(
                    f"\nMeta-Head Importance (Ridge Weights - Strictly Walk-Forward OOF):"
                )
                print(res["ridge_importance_table_"].to_string(index=False))

            print(f"\nBest Combo Found: {res.get('best_combo_name_', 'N/A')}")
            print(
                f"  Utility Score: {res.get('best_combo_', {}).get('utility_score', 0.0):.4f}"
            )
            print(
                f"  Spearman (Ret): {res.get('best_combo_', {}).get('spearman_ret', 0.0):.4f}"
            )

            cost_display = args.cost_pct if "args" in locals() else 0.003
            print(f"\nConfidence Grid (Sizing: 5% to 15% Rank-Based):")
            print(res.get("profit_proxy_table_", pd.DataFrame()).to_string(index=False))
            print("-" * 60)

    except KeyboardInterrupt:
        tprint("Execution interrupted by user.")
    except Exception as e:
        tprint(f"CRITICAL ERROR: {e}")
        import traceback

        tprint(traceback.format_exc())
        sys.exit(1)
