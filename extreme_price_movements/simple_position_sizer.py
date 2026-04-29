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
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from scipy.stats import linregress, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge, RidgeClassifier
from sklearn.preprocessing import RobustScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    import lightgbm.sklearn as _lgbm_sklearn

    def _lgbm_check_xy_compat(X: Any, y: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _lgbm_check_xy_original(X, y, **kwargs)

    def _lgbm_check_array_compat(array: Any, **kwargs: Any) -> Any:
        if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _lgbm_check_array_original(array, **kwargs)

    _lgbm_check_xy_original = _lgbm_sklearn._LGBMCheckXY
    _lgbm_check_array_original = _lgbm_sklearn._LGBMCheckArray
    _lgbm_sklearn._LGBMCheckXY = _lgbm_check_xy_compat
    _lgbm_sklearn._LGBMCheckArray = _lgbm_check_array_compat
except ImportError:
    LGBMRegressor = None  # type: ignore[assignment, misc]

from extreme_price_movements.meta_training.trade_filtering import select_top_rank_mask
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
MIN_POLICY_AVG_PNL_PER_TRADE_PCT = 0.2


def _rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling median absolute deviation."""
    arr = np.asarray(values, dtype=np.float32)
    out = np.full(len(arr), np.nan, dtype=np.float32)
    window = max(1, int(window))
    for i in range(len(arr)):
        start = max(0, i + 1 - window)
        sample = arr[start : i + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            continue
        med = float(np.median(sample))
        out[i] = float(np.median(np.abs(sample - med)))
    return out


def _compute_true_range_atr_pct(
    df: pd.DataFrame,
    *,
    horizon: int,
    symbol_col: str = "symbol",
) -> np.ndarray:
    """Compute ATR(4*h)/close in row order, grouped by symbol when available."""
    window = max(1, int(4 * max(1, horizon)))
    if not {"high", "low", "close"}.issubset(df.columns):
        return np.full(len(df), np.nan, dtype=np.float32)

    work = df[["high", "low", "close"]].copy()
    work["_pos"] = np.arange(len(work), dtype=np.int32)
    if symbol_col in df.columns:
        work[symbol_col] = df[symbol_col].values
    groups = (
        work.groupby(symbol_col, sort=False)
        if symbol_col in work.columns
        else [(None, work)]
    )
    atr_pct = np.full(len(df), np.nan, dtype=np.float32)
    for _, grp in groups:
        idx = grp["_pos"].to_numpy(dtype=np.int32)
        high = pd.to_numeric(grp["high"], errors="coerce").astype(float)
        low = pd.to_numeric(grp["low"], errors="coerce").astype(float)
        close = pd.to_numeric(grp["close"], errors="coerce").astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=window, min_periods=max(1, window // 2)).mean()
        atr_pct[idx] = (atr / close.replace(0.0, np.nan)).to_numpy(dtype=np.float32)
    return atr_pct


def _build_blended_sizer_target(
    df: pd.DataFrame,
    *,
    horizon: int,
    alpha: float = 0.7,
    symbol_col: str = "symbol",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Build blended close/Kalman forward-return target normalized by ATR."""
    if not {"close", "kalman_price"}.issubset(df.columns):
        return None, None, None

    h = max(1, int(horizon))
    work = df[["close", "kalman_price"]].copy()
    work["_pos"] = np.arange(len(work), dtype=np.int32)
    if symbol_col in df.columns:
        work[symbol_col] = df[symbol_col].values
    groups = (
        work.groupby(symbol_col, sort=False)
        if symbol_col in work.columns
        else [(None, work)]
    )
    y_raw = np.full(len(df), np.nan, dtype=np.float32)

    for _, grp in groups:
        idx = grp["_pos"].to_numpy(dtype=np.int32)
        close = pd.to_numeric(grp["close"], errors="coerce").astype(float)
        kalman = pd.to_numeric(grp["kalman_price"], errors="coerce").astype(float)
        log_close = np.log(close.where(close > 0.0))
        r_close_fwd = log_close.shift(-h) - log_close
        r_kalman_fwd = kalman.shift(-h) - kalman
        y_raw[idx] = (
            float(alpha) * r_close_fwd + (1.0 - float(alpha)) * r_kalman_fwd
        ).to_numpy(dtype=np.float32)

    atr_pct = _compute_true_range_atr_pct(df, horizon=h, symbol_col=symbol_col)
    y = y_raw / (atr_pct + 1e-8)
    y = np.clip(y, -5.0, 5.0).astype(np.float32)
    y_final = (0.65 * np.arcsinh(y) + 0.35 * y).astype(np.float32)
    valid = np.isfinite(y_final)
    if valid.mean() < 0.5:
        return None, None, None
    return y_final, atr_pct.astype(np.float32), f"atr_{4 * h}_over_close"


def _infer_side_label(
    *,
    strategy_id: str,
    strategy_meta: Optional[Dict[str, Any]] = None,
    trade_outcomes: Optional[pd.DataFrame] = None,
) -> str:
    side = ""
    if strategy_meta:
        side = str(strategy_meta.get("trade_side", strategy_meta.get("side", "")) or "")
        if side in {"long", "short"}:
            return side

    sid = str(strategy_id or "")
    if sid.startswith("long_"):
        return "long"
    if sid.startswith("short_"):
        return "short"

    if trade_outcomes is not None and "is_long" in trade_outcomes.columns:
        vals = np.asarray(trade_outcomes["is_long"].values, dtype=float)
        finite = np.isfinite(vals)
        if finite.any():
            return "long" if float(np.nanmean(vals[finite])) >= 0.5 else "short"

    return side


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
        side = _infer_side_label(strategy_id=strategy_id, strategy_meta=strategy_meta)
        source_target = str(strategy_meta.get("source_target", "") or "")
        source_horizon = strategy_meta.get("source_horizon", np.nan)
        net_pnl_per_trade_pct = _to_float_or_nan(opt_row.get("pnl_per_trade", np.nan))
        avg_wallet_growth_per_month_pct = float("nan")
        opt_ts = np.asarray(res.get("opt_ts_", np.array([])))
        wallet_pnl = _to_float_or_nan(opt_row.get("wallet_pnl", np.nan))
        if len(opt_ts) > 0 and np.isfinite(wallet_pnl):
            ts = pd.to_datetime(opt_ts, utc=True, errors="coerce")
            ts = pd.DatetimeIndex(ts[pd.notna(ts)])
            if len(ts) > 0:
                n_months = max(
                    1,
                    len(
                        pd.period_range(
                            ts.min().to_period("M"), ts.max().to_period("M"), freq="M"
                        )
                    ),
                )
                avg_wallet_growth_per_month_pct = float(wallet_pnl / n_months * 100.0)
        row = {
            "strategy_id": str(strategy_id),
            "side": side,
            "threshold_pct": _to_float_or_nan(opt_row.get("threshold_pct", np.nan)),
            "selection_frac": _to_float_or_nan(opt_row.get("selection_frac", np.nan)),
            "wallet_pnl": wallet_pnl,
            "net_pnl": _to_float_or_nan(opt_row.get("net_pnl", np.nan)),
            "pnl_per_trade": net_pnl_per_trade_pct,
            "net_pnl_per_trade_pct": net_pnl_per_trade_pct,
            "avg_wallet_growth_per_month_pct": avg_wallet_growth_per_month_pct,
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
            "calmar_ratio": _to_float_or_nan(opt_row.get("calmar_ratio", np.nan)),
            "expectancy_tstat": _to_float_or_nan(
                opt_row.get("expectancy_tstat", np.nan)
            ),
            "source_target": source_target,
            "source_horizon": _to_float_or_nan(source_horizon),
        }
        row["profitable_for_downstream"] = bool(
            np.isfinite(row["wallet_pnl"])
            and np.isfinite(row["net_pnl"])
            and np.isfinite(row["net_pnl_per_trade_pct"])
            and float(row["wallet_pnl"]) > 0.0
            and float(row["net_pnl"]) > 0.0
            and float(row["net_pnl_per_trade_pct"])
            > MIN_POLICY_AVG_PNL_PER_TRADE_PCT
        )
        row["allow_downstream"] = bool(row["profitable_for_downstream"])
        row["downstream_min_pnl_per_trade_pct"] = float(
            MIN_POLICY_AVG_PNL_PER_TRADE_PCT
        )
        reject_reasons: List[str] = []
        if not (np.isfinite(row["wallet_pnl"]) and float(row["wallet_pnl"]) > 0.0):
            reject_reasons.append("wallet_pnl_not_positive")
        if not (np.isfinite(row["net_pnl"]) and float(row["net_pnl"]) > 0.0):
            reject_reasons.append("net_pnl_not_positive")
        if not (
            np.isfinite(row["net_pnl_per_trade_pct"])
            and float(row["net_pnl_per_trade_pct"])
            > MIN_POLICY_AVG_PNL_PER_TRADE_PCT
        ):
            reject_reasons.append("avg_pnl_per_trade_below_0_2pct")
        row["downstream_reject_reasons"] = reject_reasons
        row["downstream_filter_tag"] = (
            "profitable"
            if row["profitable_for_downstream"]
            else "blocked_avg_pnl_per_trade_below_0_2pct"
        )
        row["feature_importance"] = _summarize_strategy_feature_importance(res)
        _mix_best = res.get("mix_grid_best_", {})
        if _mix_best:
            row["sizer_mix_ridge_w"] = float(_mix_best.get("ridge_w", 1.0))
            row["sizer_mix_booster_w"] = float(_mix_best.get("booster_w", 0.5))
            row["sizer_mix_conf_mult"] = float(_mix_best.get("conf_mult", 1.0))
            row["sizer_mix_winner"] = str(_mix_best.get("winner", ""))
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
        "best_threshold_pct": (
            float(strategies[0]["threshold_pct"]) if strategies else None
        ),
    }
    return payload


def _normalize_importance_table(
    df: pd.DataFrame | None,
    *,
    default_metric: str,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "feature" not in out.columns:
        if "head_name" in out.columns:
            out["feature"] = out["head_name"].astype(str)
        else:
            out["feature"] = [f"feature_{i}" for i in range(len(out))]

    metric_col = None
    for cand in (
        "mean_importance",
        "importance",
        "mean_abs_weight",
        "abs_weight",
        "mean_weight",
    ):
        if cand in out.columns:
            metric_col = cand
            break
    if metric_col is None:
        return pd.DataFrame()

    out["importance_value"] = pd.to_numeric(out[metric_col], errors="coerce").astype(
        float
    )
    if metric_col in {"mean_weight"} and "abs_weight" in out.columns:
        out["importance_value"] = pd.to_numeric(
            out["abs_weight"], errors="coerce"
        ).astype(float)
    out["importance_metric"] = str(default_metric)

    std_col = None
    for cand in ("std_importance", "std_weight"):
        if cand in out.columns:
            std_col = cand
            break
    out["importance_std"] = (
        pd.to_numeric(out[std_col], errors="coerce").astype(float)
        if std_col is not None
        else np.nan
    )
    out = out[np.isfinite(out["importance_value"])].copy()
    if out.empty:
        return pd.DataFrame()
    return out.sort_values("importance_value", ascending=False).reset_index(drop=True)


def _summarize_strategy_feature_importance(res: Dict[str, Any]) -> Dict[str, Any]:
    ridge_df = _normalize_importance_table(
        res.get("ridge_importance_table_"), default_metric="abs_weight"
    )
    et_df = _normalize_importance_table(
        res.get("et_importance_table_"), default_metric="gain"
    )
    barrier_df = _normalize_importance_table(
        res.get("barrier_clf_importance_"), default_metric="abs_weight"
    )

    comparison = dict(res.get("comparison_", {}) or {})
    winner = str(
        comparison.get(
            "wallet_winner",
            comparison.get("winner", "ridge" if not ridge_df.empty else "et"),
        )
    )
    winner_key = "ridge" if winner.startswith("ridge") else "et"
    winner_df = ridge_df if winner_key == "ridge" else et_df

    def _topn(df: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        rows = []
        for _, row in df.head(n).iterrows():
            rows.append(
                {
                    "feature": str(row.get("feature", "")),
                    "importance": float(row.get("importance_value", np.nan)),
                    "std": (
                        float(row.get("importance_std", np.nan))
                        if np.isfinite(row.get("importance_std", np.nan))
                        else None
                    ),
                    "metric": str(row.get("importance_metric", "")),
                }
            )
        return rows

    def _pred_prefix_share(df: pd.DataFrame) -> float | None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        vals = np.asarray(df["importance_value"], dtype=float)
        total = float(np.nansum(np.abs(vals)))
        if total <= 0.0:
            return None
        mask = df["feature"].astype(str).str.startswith("oof_")
        share = float(np.nansum(np.abs(vals[mask.to_numpy()])) / total)
        return share

    return {
        "winner_model": winner_key,
        "winner_top10": _topn(winner_df, 10),
        "winner_pred_prefix_share": _pred_prefix_share(winner_df),
        "elasticnet_top10": _topn(ridge_df, 10),
        "extratrees_top10": _topn(et_df, 10),
        "barrier_clf_top10": _topn(barrier_df, 10),
    }


def _persist_feature_importance_summary(
    data_root: str,
    run_id: str,
    strategy_results: Dict[str, Any],
) -> Path | None:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": "v1",
        "generated_by": "simple_position_sizer",
        "run_id": str(run_id),
        "strategies": {},
    }
    for strategy_id, res in strategy_results.items():
        summary = _summarize_strategy_feature_importance(res)
        payload["strategies"][str(strategy_id)] = summary
        for model_name, top_key in (
            ("elasticnet", "elasticnet_top10"),
            ("extratrees", "extratrees_top10"),
            ("barrier_clf", "barrier_clf_top10"),
        ):
            for rank, item in enumerate(summary.get(top_key, []), start=1):
                rows.append(
                    {
                        "strategy_id": str(strategy_id),
                        "model": model_name,
                        "rank": int(rank),
                        "feature": str(item.get("feature", "")),
                        "importance": float(item.get("importance", np.nan)),
                        "importance_std": item.get("std"),
                        "importance_metric": str(item.get("metric", "")),
                        "winner_model": str(summary.get("winner_model", "")),
                    }
                )

    out_dir = Path(data_root) / "artifacts" / run_id / "ridge_sizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "feature_importance_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "feature_importance_summary.csv", index=False
        )
    return json_path


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


# =============================================================================
# Confidence Calibration (Full Curve)
# =============================================================================


def compute_full_calibration_curves(
    oof_predictions: pd.DataFrame,
    realized_returns: pd.DataFrame,
    strategy_col: str = "strategy",
    score_col: str = "sizer_score",
    return_col: str = "realized_return",
    n_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Compute full calibration curves for each strategy using isotonic regression.

    This computes well-calibrated probability estimates from raw scores by
    learning a monotonic mapping from scores to observed win frequencies.

    Args:
        oof_predictions: DataFrame with OOF predictions per strategy
        realized_returns: DataFrame with realized returns aligned to predictions
        strategy_col: Column name for strategy identifier
        score_col: Column name for the raw score/prediction
        return_col: Column name for realized returns
        n_bins: Number of bins for empirical calibration curve

    Returns:
        Dict mapping strategy_id -> calibration data:
        {
            "strategy_id": str,
            "raw_scores": List[float],  # All historical scores
            "isotonic_scores": List[float],  # Calibrated scores
            "score_range": (min, max),
            "bin_edges": List[float],  # For discretized calibration
            "bin_centers": List[float],
            "bin_frequencies": List[float],  # Observed win rates per bin
            "bin_counts": List[int],  # Sample count per bin
            "p75_threshold": float,  # 75th percentile of isotonic scores
            "p90_threshold": float,  # 90th percentile for reference
            "calibration_curve": List[Tuple[float, float]],  # (raw, calibrated) pairs
        }
    """
    from sklearn.isotonic import IsotonicRegression

    calibration_data: Dict[str, Dict[str, Any]] = {}

    # Ensure aligned data
    if strategy_col not in oof_predictions.columns:
        logger.warning(f"[Calibration] {strategy_col} not in OOF predictions")
        return calibration_data

    # Merge predictions with realized returns
    merged = oof_predictions.merge(
        realized_returns[["symbol", "timestamp", return_col]],
        on=["symbol", "timestamp"],
        how="inner",
    )

    if merged.empty:
        logger.warning("[Calibration] No aligned data after merge")
        return calibration_data

    # Group by strategy
    for strategy_id, group in merged.groupby(strategy_col):
        if len(group) < n_bins * 5:  # Need enough samples
            logger.warning(
                f"[Calibration] Insufficient samples for {strategy_id}: {len(group)}"
            )
            continue

        raw_scores = group[score_col].values
        returns = group[return_col].values

        # Binary outcomes (win/loss)
        was_win = (returns > 0).astype(float)

        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        calibrated_scores = iso_reg.fit_transform(raw_scores, was_win)

        # Compute empirical calibration curve (binned)
        sorted_indices = np.argsort(raw_scores)
        sorted_scores = raw_scores[sorted_indices]
        sorted_calibrated = calibrated_scores[sorted_indices]

        # Create bins
        bin_edges = np.linspace(sorted_scores.min(), sorted_scores.max(), n_bins + 1)
        bin_centers = []
        bin_frequencies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (sorted_scores >= bin_edges[i]) & (sorted_scores < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (sorted_scores >= bin_edges[i]) & (
                    sorted_scores <= bin_edges[i + 1]
                )

            bin_scores = sorted_scores[mask]
            bin_calibrated = sorted_calibrated[mask]

            if len(bin_scores) > 0:
                bin_centers.append(float(np.mean(bin_scores)))
                bin_frequencies.append(float(np.mean(bin_calibrated)))
                bin_counts.append(int(len(bin_scores)))

        # Compute percentiles on calibrated scores
        p75 = float(np.percentile(calibrated_scores, 75))
        p90 = float(np.percentile(calibrated_scores, 90))

        # Store calibration data
        calibration_data[str(strategy_id)] = {
            "strategy_id": str(strategy_id),
            "n_samples": int(len(group)),
            "raw_score_range": (float(raw_scores.min()), float(raw_scores.max())),
            "calibrated_score_range": (
                float(calibrated_scores.min()),
                float(calibrated_scores.max()),
            ),
            "bin_edges": [float(x) for x in bin_edges],
            "bin_centers": bin_centers,
            "bin_frequencies": bin_frequencies,
            "bin_counts": bin_counts,
            "p75_threshold": p75,
            "p90_threshold": p90,
            "isotonic_regression": {
                "X_min": float(iso_reg.X_min_),
                "X_max": float(iso_reg.X_max_),
                "increasing": bool(iso_reg.increasing),
            },
            # Store full calibration curve (subsampled for efficiency)
            "calibration_curve": [
                (float(raw_scores[i]), float(calibrated_scores[i]))
                for i in range(0, len(raw_scores), max(1, len(raw_scores) // 1000))
            ],
        }

    return calibration_data


def compute_oof_rank_calibration_curves(
    oof_predictions: pd.DataFrame,
    strategy_col: str = "strategy_id",
    score_col: str = "sizer_score_oof",
    n_points: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """Calibrate raw OOF sizer scores to empirical within-strategy rank percentiles.

    This fallback is used when the full realized-return calibration inputs are
    not persisted but OOF sizer scores are. It preserves cross-strategy
    comparability for inference gating by mapping each strategy's raw score
    distribution to the same [0, 1] rank scale.
    """
    calibration_data: Dict[str, Dict[str, Any]] = {}
    if oof_predictions.empty or strategy_col not in oof_predictions.columns:
        return calibration_data
    if score_col not in oof_predictions.columns:
        return calibration_data

    for strategy_id, group in oof_predictions.groupby(strategy_col):
        scores = pd.to_numeric(group[score_col], errors="coerce").dropna()
        if len(scores) < 50:
            continue
        sorted_scores = np.sort(scores.to_numpy(dtype=np.float64))
        if sorted_scores.size == 0:
            continue
        ranks = np.linspace(0.0, 1.0, sorted_scores.size, dtype=np.float64)
        stride = max(1, int(np.ceil(sorted_scores.size / max(1, int(n_points)))))
        curve = [
            (float(sorted_scores[i]), float(ranks[i]))
            for i in range(0, sorted_scores.size, stride)
        ]
        if curve[-1][0] != float(sorted_scores[-1]):
            curve.append((float(sorted_scores[-1]), 1.0))
        calibration_data[str(strategy_id)] = {
            "strategy_id": str(strategy_id),
            "n_samples": int(sorted_scores.size),
            "raw_score_range": (
                float(sorted_scores[0]),
                float(sorted_scores[-1]),
            ),
            "calibrated_score_range": (0.0, 1.0),
            "p75_threshold": 0.75,
            "p90_threshold": 0.90,
            "calibration_curve": curve,
            "calibration_method": "empirical_oof_rank_percentile",
        }
    return calibration_data


def save_calibration_curves(
    calibration_data: Dict[str, Dict[str, Any]],
    data_root: str,
    run_id: str,
    *,
    calibration_method: str = "isotonic_win_probability",
) -> Path:
    """Save calibration curves as JSON artifact."""
    path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "confidence_calibration.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    payload = {
        "schema_version": "v1",
        "generated_by": "simple_position_sizer",
        "run_id": run_id,
        "n_strategies": len(calibration_data),
        "strategies": calibration_data,
    }

    path.write_text(json.dumps(payload, indent=2))
    contract_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "confidence_calibration.contract.json"
    )
    contract_payload = {
        "schema_version": "v1",
        "generated_by": "simple_position_sizer",
        "run_id": run_id,
        "required_strategy_fields": ["p75_threshold", "calibration_curve"],
        "rank_semantics": (
            "empirical_oof_rank_percentile"
            if calibration_method == "empirical_oof_rank_percentile"
            else "calibrated_p75_threshold"
        ),
        "calibration_method": calibration_method,
        "disagreement_feature_contract": "engine._calculate_disagreement_features",
    }
    contract_path.write_text(json.dumps(contract_payload, indent=2))
    tprint(
        f"[Calibration] Saved calibration curves for {len(calibration_data)} strategies to {path}"
    )
    return path


def load_calibration_curves(
    data_root: str,
    run_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Load calibration curves from JSON artifact."""
    path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "confidence_calibration.json"
    )
    if not path.exists():
        return {}

    payload = json.loads(path.read_text())
    return payload.get("strategies", {})


def load_calibration_contract(
    data_root: str,
    run_id: str,
) -> Dict[str, Any]:
    """Load calibration contract metadata used by inference parity checks."""
    path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "confidence_calibration.contract.json"
    )
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def calibrate_score(
    raw_score: float,
    strategy_id: str,
    calibration_data: Dict[str, Dict[str, Any]],
) -> float:
    """Calibrate a single raw score using pre-computed calibration curve.

    Uses linear interpolation between calibration curve points.

    Args:
        raw_score: Raw confidence score
        strategy_id: Strategy identifier
        calibration_data: Calibration data from compute_full_calibration_curves()

    Returns:
        Calibrated score (well-calibrated probability estimate)
    """
    if strategy_id not in calibration_data:
        return raw_score  # No calibration available

    strat_calib = calibration_data[strategy_id]
    curve = strat_calib.get("calibration_curve", [])

    if not curve:
        return raw_score

    # Sort by raw score
    sorted_curve = sorted(curve, key=lambda x: x[0])
    raw_points = [x[0] for x in sorted_curve]
    calib_points = [x[1] for x in sorted_curve]

    # Find interpolation position
    if raw_score <= raw_points[0]:
        return calib_points[0]
    if raw_score >= raw_points[-1]:
        return calib_points[-1]

    # Linear interpolation
    for i in range(len(raw_points) - 1):
        if raw_points[i] <= raw_score <= raw_points[i + 1]:
            # Linear interpolation
            t = (raw_score - raw_points[i]) / (raw_points[i + 1] - raw_points[i])
            return calib_points[i] + t * (calib_points[i + 1] - calib_points[i])

    return calib_points[-1]


def filter_by_calibrated_confidence(
    df: pd.DataFrame,
    calibration_data: Dict[str, Dict[str, Any]],
    percentile_threshold: float = 75.0,
    strategy_col: str = "strategy",
    score_col: str = "sizer_score",
    calibrated_col: str = "calibrated_score",
) -> pd.DataFrame:
    """Filter trades where calibrated confidence ranks below threshold percentile.

    Args:
        df: DataFrame with trades/scores
        calibration_data: Strategy calibration data
        percentile_threshold: Percentile cutoff (default: 75.0 for top 25%)
        strategy_col: Strategy identifier column
        score_col: Raw score column
        calibrated_col: Column name to store calibrated scores

    Returns:
        Filtered DataFrame with only passing trades
    """
    if not calibration_data or df.empty:
        return df

    # Add calibrated scores
    def get_threshold(row):
        sid = row[strategy_col]
        calib = calibration_data.get(sid, {})
        # Get threshold for specified percentile
        pct_key = f"p{int(percentile_threshold)}_threshold"
        return calib.get(pct_key, calib.get("p75_threshold", 0.5))

    df["_threshold"] = df.apply(get_threshold, axis=1)
    df[calibrated_col] = df.apply(
        lambda row: calibrate_score(
            row[score_col], row[strategy_col], calibration_data
        ),
        axis=1,
    )

    # Filter: keep only if calibrated score >= threshold
    mask = df[calibrated_col] >= df["_threshold"]
    filtered = df[mask].copy()

    n_before = len(df)
    n_after = len(filtered)
    tprint(
        f"[Calibration] Filtered {n_before} -> {n_after} trades ({n_after/n_before*100:.1f}% kept)"
    )

    # Clean up temp column
    filtered = filtered.drop(columns=["_threshold"])

    return filtered


def filter_qualified_strategies(
    strategies: List[Dict[str, Any]],
    *,
    min_profit_factor: float = 1.3,
    min_stability: float = 0.7,
    min_monthly_sortino: float = 1.0,
    min_calmar_ratio: float = 1.0,
    min_expectancy_tstat: float = 2.0,
) -> List[Dict[str, Any]]:
    qualified: List[Dict[str, Any]] = []
    for s in strategies:
        pf = float(s.get("profit_factor", 0.0) or 0.0)
        stab = float(s.get("stability", 0.0) or 0.0)
        ms = float(s.get("monthly_sortino", 0.0) or 0.0)
        calmar = float(s.get("calmar_ratio", 0.0) or 0.0)
        etstat = float(s.get("expectancy_tstat", 0.0) or 0.0)
        if pf < min_profit_factor:
            continue
        if stab < min_stability:
            continue
        if ms < min_monthly_sortino:
            continue
        if calmar < min_calmar_ratio:
            continue
        if etstat < min_expectancy_tstat:
            continue
        qualified.append(s)
    return qualified


def write_holdout_multi_metrics(
    data_root: str,
    run_id: str,
) -> List[Dict[str, Any]]:
    params_path = _strategy_params_path(data_root, run_id)
    if not params_path.exists():
        return []
    payload = json.loads(params_path.read_text())
    all_strategies = payload.get("strategies", [])
    qualified = filter_qualified_strategies(all_strategies)
    out_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "holdout_multi_metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qualified, indent=2, sort_keys=True, default=str))
    tprint(
        f"Holdout multi-metrics: {len(qualified)}/{len(all_strategies)} strategies "
        f"passed quality gates -> {out_path}"
    )
    return qualified


def _persist_sizer_oof_predictions(
    data_root: str,
    run_id: str,
    strategy_results: Dict[str, Any],
) -> None:
    """Save sizer OOF predictions to parquet for policy_optimiser consumption.

    Saves predictions indexed by row position (assumes consistent ordering across strategies).
    Also persists Total_Confidence and inference metadata for live residualization.
    """
    all_rows = []
    row_idx = 0

    for strategy_id, res in strategy_results.items():
        ridge_preds = res.get("ridge_sizer_scores_")
        et_preds = res.get("et_sizer_scores_")
        lgbm_preds = res.get("lgbm_sizer_scores_")
        total_conf = res.get("total_confidence_")
        total_conf_lgbm = res.get("total_confidence_lgbm_")
        ctx = res.get("_sizer_oof_context_", {})

        preds = total_conf
        if preds is None:
            preds = (
                ridge_preds
                if ridge_preds is not None
                else (
                    total_conf_lgbm
                    if total_conf_lgbm is not None
                    else (lgbm_preds if lgbm_preds is not None else et_preds)
                )
            )
        if preds is None:
            continue

        n_preds = len(preds)

        comp = res.get("comparison_", {})
        if comp:
            winner = comp.get("winner", "elasticnet")
        else:
            winner = "extratrees" if et_preds is not None else "elasticnet"

        for idx in range(n_preds):
            score = float(preds[idx]) if np.isfinite(preds[idx]) else np.nan
            timestamp_val = np.nan
            symbol_val = np.nan
            if isinstance(ctx, dict):
                if "timestamp" in ctx and idx < len(ctx["timestamp"]):
                    timestamp_val = ctx["timestamp"][idx]
                if "symbol" in ctx and idx < len(ctx["symbol"]):
                    symbol_val = ctx["symbol"][idx]
            row_data = {
                "row_idx": row_idx + idx,
                "strategy_id": strategy_id,
                "timestamp": timestamp_val,
                "symbol": symbol_val,
                "sizer_score_oof": score,
                "total_confidence": score,
                "model": winner,
            }
            if ridge_preds is not None and idx < len(ridge_preds):
                row_data["ridge_score"] = (
                    float(ridge_preds[idx]) if np.isfinite(ridge_preds[idx]) else np.nan
                )
            if et_preds is not None and idx < len(et_preds):
                row_data["et_score"] = (
                    float(et_preds[idx]) if np.isfinite(et_preds[idx]) else np.nan
                )
            if lgbm_preds is not None and idx < len(lgbm_preds):
                row_data["lgbm_score"] = (
                    float(lgbm_preds[idx])
                    if np.isfinite(lgbm_preds[idx])
                    else np.nan
                )
            all_rows.append(row_data)

        row_idx += n_preds

    if all_rows:
        df = pd.DataFrame(all_rows)
        missing_context = int(
            df["timestamp"].isna().sum() + df["symbol"].isna().sum()
        )
        if missing_context:
            logger.warning(
                "simple_position_sizer OOF context has %d missing timestamp/symbol "
                "values; policy optimiser will drop unmatched rows.",
                missing_context,
            )
        out_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "oof"
            / "simple_sizer_oof_all.parquet"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved simple_position_sizer OOF predictions to {out_path}")

    for strategy_id, res in strategy_results.items():
        calibration_isotonic = res.get("calibration_isotonic_")
        ridge_resid_mean = res.get("ridge_resid_mean_")
        ridge_resid_std = res.get("ridge_resid_std_")
        et_best_params = res.get("et_best_params_")
        lgbm_best_params = res.get("lgbm_best_params_")
        cal_iso_et = res.get("calibration_isotonic_")
        cal_iso_lgbm = res.get("calibration_isotonic_lgbm_")

        if cal_iso_et is not None or et_best_params is not None:
            meta_path = (
                Path(data_root)
                / "artifacts"
                / run_id
                / "et_sizer"
                / "inference_metadata.json"
            )
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            import pickle

            if cal_iso_et is not None:
                with open(meta_path.parent / "calibration_isotonic.pkl", "wb") as f:
                    pickle.dump(cal_iso_et, f)
            meta_payload = {
                "ridge_resid_mean": ridge_resid_mean,
                "ridge_resid_std": ridge_resid_std,
                "et_best_params": et_best_params,
                "has_calibration": cal_iso_et is not None,
                "atr_column": res.get("_strategy_meta_", {}).get("atr_column"),
            }
            with open(meta_path, "w") as f:
                json.dump(meta_payload, f, indent=2)

        if cal_iso_lgbm is not None or lgbm_best_params is not None:
            meta_path_lgbm = (
                Path(data_root)
                / "artifacts"
                / run_id
                / "lgbm_sizer"
                / "inference_metadata.json"
            )
            meta_path_lgbm.parent.mkdir(parents=True, exist_ok=True)
            import pickle

            if cal_iso_lgbm is not None:
                with open(
                    meta_path_lgbm.parent / "calibration_isotonic.pkl", "wb"
                ) as f:
                    pickle.dump(cal_iso_lgbm, f)
            meta_payload_lgbm = {
                "ridge_resid_mean": ridge_resid_mean,
                "ridge_resid_std": ridge_resid_std,
                "lgbm_best_params": lgbm_best_params,
                "has_calibration": cal_iso_lgbm is not None,
                "atr_column": res.get("_strategy_meta_", {}).get("atr_column"),
            }
            with open(meta_path_lgbm, "w") as f:
                json.dump(meta_payload_lgbm, f, indent=2)

        break


def _persist_booster_bundle(
    data_root: str,
    run_id: str,
    strategy_id: str,
    res: Dict[str, Any],
) -> None:
    import pickle

    out_dir = Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "booster_bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = strategy_id[:80]
    winner = res.get("mix_grid_best_", {}).get("winner", "")
    bundle: Dict[str, Any] = {
        "strategy_id": strategy_id,
        "winner": winner,
        "ridge_resid_mean": res.get("ridge_resid_mean_"),
        "ridge_resid_std": res.get("ridge_resid_std_"),
    }
    if winner == "ridge_plus_et":
        bundle["fold_models"] = res.get("fold_et_models_", [])
        bundle["feature_keys"] = res.get("et_feature_keys_", [])
        bundle["calibration_isotonic"] = res.get("calibration_isotonic_")
    elif winner == "ridge_plus_lgbm":
        bundle["fold_models"] = res.get("fold_lgbm_models_", [])
        bundle["feature_keys"] = res.get("lgbm_feature_keys_", [])
        bundle["calibration_isotonic"] = res.get("calibration_isotonic_lgbm_")
    elif winner == "ridge_plus_lgbm_clf":
        bundle["fold_models"] = res.get("fold_lgbm_clf_models_", [])
        bundle["feature_keys"] = res.get("lgbm_clf_feature_keys_", [])
        bundle["calibration_isotonic"] = None
    else:
        return
    if not bundle.get("fold_models"):
        return
    pkl_path = out_dir / f"{safe_id}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)


def _persist_head_to_head_winner(
    data_root: str,
    run_id: str,
    strategy_results: Dict[str, Any],
) -> None:
    comparison_rows: List[Dict[str, Any]] = []
    et_winner_rows: List[Dict[str, Any]] = []
    lgbm_winner_rows: List[Dict[str, Any]] = []

    def _winner_policy_row(
        *,
        strategy_id: str,
        meta: Dict[str, Any],
        opt: Any,
        model_source: str,
    ) -> Dict[str, Any]:
        wallet_pnl = _to_float_or_nan(opt.get("wallet_pnl", np.nan))
        net_pnl = _to_float_or_nan(opt.get("net_pnl", np.nan))
        pnl_per_trade_pct = _to_float_or_nan(
            opt.get("pnl_per_trade_pct", opt.get("pnl_per_trade", np.nan))
        )
        reject_reasons: List[str] = []
        if not (np.isfinite(wallet_pnl) and wallet_pnl > 0.0):
            reject_reasons.append("wallet_pnl_not_positive")
        if not (np.isfinite(net_pnl) and net_pnl > 0.0):
            reject_reasons.append("net_pnl_not_positive")
        if not (
            np.isfinite(pnl_per_trade_pct)
            and pnl_per_trade_pct > MIN_POLICY_AVG_PNL_PER_TRADE_PCT
        ):
            reject_reasons.append("avg_pnl_per_trade_below_0_2pct")
        profitable_for_downstream = len(reject_reasons) == 0
        return {
            "strategy_id": str(strategy_id),
            "side": _infer_side_label(
                strategy_id=str(strategy_id), strategy_meta=meta
            ),
            "threshold_pct": _to_float_or_nan(opt.get("threshold_pct", np.nan)),
            "selection_frac": _to_float_or_nan(opt.get("selection_frac", np.nan)),
            "wallet_pnl": wallet_pnl,
            "net_pnl": net_pnl,
            "pnl_per_trade": pnl_per_trade_pct,
            "net_pnl_per_trade_pct": pnl_per_trade_pct,
            "profit_factor": _to_float_or_nan(opt.get("profit_factor", np.nan)),
            "hit_rate": _to_float_or_nan(opt.get("hit_rate", np.nan)),
            "trades_per_day": _to_float_or_nan(opt.get("trades_per_day", np.nan)),
            "trades_selected": _to_float_or_nan(opt.get("trades_selected", np.nan)),
            "model_source": model_source,
            "profitable_for_downstream": bool(profitable_for_downstream),
            "allow_downstream": bool(profitable_for_downstream),
            "downstream_min_pnl_per_trade_pct": float(
                MIN_POLICY_AVG_PNL_PER_TRADE_PCT
            ),
            "downstream_reject_reasons": reject_reasons,
            "downstream_filter_tag": (
                "profitable"
                if profitable_for_downstream
                else "blocked_avg_pnl_per_trade_below_0_2pct"
            ),
        }

    for strategy_id, res in strategy_results.items():
        comp = res.get("comparison_", {})
        if not comp:
            continue
        comparison_rows.append({"strategy_id": strategy_id, **comp})
        winner = comp.get("winner", "")
        if winner == "ridge_plus_et":
            profit_df = res.get("et_profit_proxy_table_", pd.DataFrame())
            if profit_df.empty:
                continue
            opt = (
                profit_df[profit_df["is_optimal"]].iloc[0]
                if "is_optimal" in profit_df.columns
                else profit_df.sort_values("wallet_pnl", ascending=False).iloc[0]
            )
            meta = res.get("_strategy_meta_", {})
            et_winner_rows.append(
                _winner_policy_row(
                    strategy_id=str(strategy_id),
                    meta=meta,
                    opt=opt,
                    model_source="ridge_plus_et",
                )
            )
        elif winner == "ridge_plus_lgbm":
            profit_df = res.get("lgbm_profit_proxy_table_", pd.DataFrame())
            if profit_df.empty:
                continue
            opt = (
                profit_df[profit_df["is_optimal"]].iloc[0]
                if "is_optimal" in profit_df.columns
                else profit_df.sort_values("wallet_pnl", ascending=False).iloc[0]
            )
            meta = res.get("_strategy_meta_", {})
            lgbm_winner_rows.append(
                _winner_policy_row(
                    strategy_id=str(strategy_id),
                    meta=meta,
                    opt=opt,
                    model_source="ridge_plus_lgbm",
                )
            )
        elif winner == "ridge_plus_lgbm_clf":
            profit_df = res.get("lgbm_clf_profit_proxy_table_", pd.DataFrame())
            if profit_df.empty:
                continue
            opt = (
                profit_df[profit_df["is_optimal"]].iloc[0]
                if "is_optimal" in profit_df.columns
                else profit_df.sort_values("wallet_pnl", ascending=False).iloc[0]
            )
            meta = res.get("_strategy_meta_", {})
            comparison_rows.append(
                {
                    "strategy_id": str(strategy_id),
                    "wallet_pnl": _to_float_or_nan(opt.get("wallet_pnl", np.nan)),
                    "net_pnl": _to_float_or_nan(opt.get("net_pnl", np.nan)),
                    "profit_factor": _to_float_or_nan(opt.get("profit_factor", np.nan)),
                    "hit_rate": _to_float_or_nan(opt.get("hit_rate", np.nan)),
                    "model_source": "ridge_plus_lgbm_clf",
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

    for model_name, winner_rows in [
        ("et_sizer", et_winner_rows),
        ("lgbm_sizer", lgbm_winner_rows),
    ]:
        if not winner_rows:
            continue
        best_row = max(
            winner_rows,
            key=lambda r: (
                float(r.get("net_pnl", float("-inf"))),
                float(r.get("profit_factor", float("-inf"))),
                float(r.get("hit_rate", float("-inf"))),
            ),
        )
        params_path = (
            Path(data_root) / "artifacts" / run_id / model_name / "strategy_params.json"
        )
        params_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "v1",
            "generated_by": "simple_position_sizer",
            "run_id": run_id,
            "fee_pct": 0.003,
            "strategies": winner_rows,
            "buckets": {row["strategy_id"]: dict(row) for row in winner_rows},
            "best_strategy_id": best_row["strategy_id"],
            "best_threshold_pct": best_row["threshold_pct"],
        }
        params_path.write_text(json.dumps(payload, indent=2))
        logger.info(
            f"Persisted {model_name} winner params ({len(winner_rows)} strategies) to {params_path}"
        )

    comp_table = pd.DataFrame()
    for strategy_id, res in strategy_results.items():
        ct = res.get("comparison_table_", pd.DataFrame())
        if not ct.empty:
            comp_table = pd.concat(
                [comp_table, ct.assign(strategy_id=strategy_id)], ignore_index=True
            )
    if not comp_table.empty:
        table_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "ridge_sizer"
            / "pipeline_comparison_table.csv"
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        comp_table.to_csv(table_path, index=False)
        logger.info(f"Persisted pipeline comparison table to {table_path}")


def _persist_mix_grid(
    data_root: str,
    run_id: str,
    strategy_results: Dict[str, Any],
) -> None:
    rows: List[Dict[str, Any]] = []
    for strategy_id, res in strategy_results.items():
        _best = res.get("mix_grid_best_", {})
        if _best:
            _best["strategy_id"] = strategy_id
            rows.append(_best)
        _tbl = res.get("mix_grid_table_", pd.DataFrame())
        if not _tbl.empty:
            _tbl = _tbl.copy()
            _tbl["strategy_id"] = strategy_id
            rows.extend(_tbl.to_dict("records"))
    if not rows:
        return
    out_path = (
        Path(data_root) / "artifacts" / run_id / "ridge_sizer" / "mix_grid_results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _persist_detailed_model_metrics(
    data_root: str,
    run_id: str,
    strategy_results: Dict[str, Any],
) -> None:
    rows: List[Dict[str, Any]] = []
    for strategy_id, res in strategy_results.items():
        for model_tag, eval_key, profit_key in [
            ("ridge", "ridge_sizer_eval_", "ridge_profit_proxy_table_"),
            ("et", "et_sizer_eval_", "et_profit_proxy_table_"),
            ("lgbm", "lgbm_sizer_eval_", "lgbm_profit_proxy_table_"),
            ("lgbm_clf", "lgbm_clf_sizer_eval_", "lgbm_clf_profit_proxy_table_"),
        ]:
            eval_dict = res.get(eval_key, {})
            profit_df = res.get(profit_key, pd.DataFrame())
            row: Dict[str, Any] = {
                "strategy_id": strategy_id,
                "model": model_tag,
                "spearman_ic": eval_dict.get("spearman_ret", float("nan")),
                "top_1_mean_net": eval_dict.get("top_1_mean_net", float("nan")),
                "top_1_hit_rate": eval_dict.get("top_1_hit_rate", float("nan")),
                "top_2_5_mean_net": eval_dict.get("top_2_5_mean_net", float("nan")),
                "top_2_5_hit_rate": eval_dict.get("top_2_5_hit_rate", float("nan")),
                "top_5_mean_net": eval_dict.get("top_5_mean_net", float("nan")),
                "top_5_hit_rate": eval_dict.get("top_5_hit_rate", float("nan")),
                "top_10_mean_net": eval_dict.get("top_10_mean_net", float("nan")),
                "top_10_hit_rate": eval_dict.get("top_10_hit_rate", float("nan")),
                "top_20_mean_net": eval_dict.get("top_20_mean_net", float("nan")),
                "top_20_hit_rate": eval_dict.get("top_20_hit_rate", float("nan")),
                "monotonicity": eval_dict.get("monotonicity", float("nan")),
                "false_safe_rate": eval_dict.get("false_safe_rate", float("nan")),
                "utility_score": eval_dict.get("utility_score", float("nan")),
            }
            if not profit_df.empty and "is_optimal" in profit_df.columns:
                opt = profit_df[profit_df["is_optimal"]].iloc[0]
                row["optimal_threshold_pct"] = opt.get("threshold_pct", float("nan"))
                row["optimal_selection_frac"] = opt.get("selection_frac", float("nan"))
                row["optimal_wallet_pnl"] = opt.get("wallet_pnl", float("nan"))
                row["optimal_net_pnl"] = opt.get("net_pnl", float("nan"))
                row["optimal_hit_rate"] = opt.get("hit_rate", float("nan"))
                row["optimal_pf"] = opt.get("profit_factor", float("nan"))
                row["optimal_sortino"] = opt.get("sortino", float("nan"))
                row["optimal_monthly_sortino"] = opt.get(
                    "monthly_sortino", float("nan")
                )
                row["optimal_stability"] = opt.get("stability", float("nan"))
                row["optimal_mdd"] = opt.get("max_drawdown", float("nan"))
                row["optimal_trades_per_day"] = opt.get("trades_per_day", float("nan"))
                row["optimal_pnl_per_trade"] = opt.get(
                    "pnl_per_trade_pct", float("nan")
                )
            rows.append(row)
    if not rows:
        return
    out_path = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "detailed_model_metrics.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


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
    "oof_p_move",
    "utility",
    "mae_q70",
    "mfe",
    "early_inval",
    "mae_mean",
    "mae_std",
    "mfe_mean",
    "mfe_std",
    "tbm_mean",
    "tbm_std",
    "risk_reward_ratio",
    "risk_adjusted_pred",
    "high_utility_pred",
    "utility_disagreement",
    "robust_sigma_meta_reg",
    "robust_sigma_meta_clf",
    "cv_meta_reg",
    "cv_meta_clf",
    "avg_robust_sigma_meta",
    "avg_cv_meta",
    "meta_avg",
    "meta_diff",
    "meta_abs_diff",
    "meta_rel_diff",
    "meta_agreement_strength",
    "meta_reliability",
    "clf_center",
    "clf_entropy",
    "clf_prefix_std",
    "reg_pred",
    "reg_prefix_std",
    "clf_leaf_support_q25",
    "reg_leaf_support_q25",
    "clf_leaf_target_iqr_mean",
    "reg_leaf_target_iqr_mean",
    "sign_agree",
    "joint_confidence",
    "conflict_score",
    "joint_instability",
    "edge_unc_pen",
    "edge_support_pen",
    "edge_noise_pen",
    "Upside",
    "Downside",
    "EdgeSharpe",
    "winrate_20",
    "winrate_50",
    "brier_50",
    "logloss_50",
    "rank_ic_50",
    "consecutive_losses",
    "edge_tbm",
    "edge_x_winrate_20",
    "edge_x_rank_ic_50",
    "edge_x_brier_50",
    "edge_x_volatility_zscore",
    "edge_x_amihud_z",
    "edge_x_vol_of_vol",
}
_RIDGE_HEAD_PREFIXES = (
    "base_h",
    "reg_h",
    "tbm_",
    "mae_h",
    "mfe_h",
    "asym_h",
    "oof_p_",
)

_NON_HEAD_DROP_FEATURES = {
    "impact_z",
    "dv_z",
    "rng_z",
    "score",
    "early_inval",
    "oof_p_tp",
    "oof_p_sl",
    "oof_p_time",
    "vol_concentration_12",
    "volume_entropy_12",
    "clv_t",
    "body_ratio_15m",
    "rejection_proxy",
    "range_norm_12",
    "sv_imb_24",
    "press_24",
    "impact_24",
    "ts_24",
    "atr_12_15m",
    "clf_center",
    "clf_entropy",
    "base_clf_centered",
    "oof_base_meta_correctness_prob",
    "clf_prefix_std",
    "clf_leaf_support_q25",
    "clf_leaf_target_iqr_mean",
    "reg",
    "reg_mean",
    "reg_std",
    "reg_range",
    "reg_pred",
    "reg_prefix_std",
    "reg_leaf_support_q25",
    "reg_leaf_target_iqr_mean",
    "utility",
    "mae_q70",
    "mfe",
    "oof_u_hat",
    "oof_log_mae_q70_hat",
    "oof_log_mfe_hat",
    "oof_asym_hat",
    "Upside",
    "Downside",
    "EdgeSharpe",
    "risk_reward_ratio",
    "high_utility_pred",
    "risk_adjusted_pred",
    "utility_disagreement",
    "sign_agree",
    "joint_confidence",
    "conflict_score",
    "joint_instability",
    "edge_unc_pen",
    "edge_support_pen",
    "edge_noise_pen",
    "vol_of_vol",
    "volatility_of_volatility_48",
    "vov_ratio",
    "vov_fast_slow_ratio",
    "rvol_hod_base",
    "rvol_z",
    "volume_price_corr_10h",
    "variance_ratio_10_48",
    "hurst_proxy_24",
    "dist_vwap_norm",
    "z_vwap_24",
    "climax_range_12",
    "climax_vol_12",
}

_LIVE_UNAVAILABLE_SIZER_FEATURES = {
    "reg_gate_target",
    "reg_train_target",
    "reg_target_positive",
    "reg_raw_vol_norm",
    "y_move",
    "y_move_soft",
    "move_threshold",
    "barrier_pct",
    "bars_to_mfe",
    "reg_weight",
}


def _config_sizer_features() -> set:
    from extreme_price_movements.config import CFG

    return set(CFG.get("position_sizer_features", []))


def collect_ridge_head_columns(df: pd.DataFrame) -> List[str]:
    """Return columns the sizer model should use: config regime features + meta heads + base OOF heads."""
    config_feats = _config_sizer_features()
    cols: List[str] = []
    for col in df.columns:
        if col in _RIDGE_HEAD_EXACT:
            cols.append(col)
            continue
        col_l = col.lower()
        if any(col_l.startswith(prefix) for prefix in _RIDGE_HEAD_PREFIXES):
            cols.append(col)
            continue
        if col in config_feats:
            if np.issubdtype(df[col].dtype, np.number):
                cols.append(col)
    return cols


def _is_classifier_head_feature(col: str) -> bool:
    c = str(col).lower()
    return (
        c == "clf"
        or c.startswith("oof_p_")
        or "meta_clf" in c
        or c in {"oof_ebm_raw", "oof_ebm_en", "oof_ebm_uncertainty_weighted"}
    )


def _augment_meta_clf_reliability_features(
    active_df: pd.DataFrame, trade_outcomes: pd.DataFrame
) -> pd.DataFrame:
    out = active_df.copy()
    edge_col = next(
        (c for c in ("oof_p_tp", "clf", "oof_p_move") if c in out.columns),
        None,
    )
    if edge_col is None:
        return out
    edge = np.asarray(out[edge_col], dtype=np.float64)
    valid_edge = np.isfinite(edge)
    if int(np.sum(valid_edge)) < 30:
        return out
    q90 = float(np.nanquantile(edge[valid_edge], 0.90))
    top_mask = valid_edge & (edge >= q90)

    mfe = (
        np.asarray(trade_outcomes["mfe_ret"], dtype=np.float64)
        if "mfe_ret" in trade_outcomes.columns
        else np.full(len(out), np.nan, dtype=np.float64)
    )
    tp = np.full(len(out), 0.02, dtype=np.float64)
    for c in ("__barrier_pct__", "barrier_pct", "tp", "tp_pct"):
        if c in out.columns:
            tp = np.asarray(out[c], dtype=np.float64)
            break
        if c in trade_outcomes.columns:
            tp = np.asarray(trade_outcomes[c], dtype=np.float64)
            break
    y_win = (mfe > tp).astype(np.float64)
    y_win[~np.isfinite(mfe) | ~np.isfinite(tp)] = np.nan

    winrate_20 = np.full(len(out), np.nan, dtype=np.float64)
    winrate_50 = np.full(len(out), np.nan, dtype=np.float64)
    brier_50 = np.full(len(out), np.nan, dtype=np.float64)
    logloss_50 = np.full(len(out), np.nan, dtype=np.float64)
    rank_ic_50 = np.full(len(out), np.nan, dtype=np.float64)
    consecutive_losses = np.zeros(len(out), dtype=np.float64)

    top_idx = np.where(top_mask)[0]
    seen_idx: list[int] = []
    current_losses = 0.0
    for idx in top_idx:
        seen_idx.append(int(idx))
        recent20 = seen_idx[-20:]
        recent50 = seen_idx[-50:]
        y20 = y_win[recent20]
        y50 = y_win[recent50]
        p50 = np.clip(edge[recent50], 1e-6, 1.0 - 1e-6)
        m20 = np.isfinite(y20)
        m50 = np.isfinite(y50) & np.isfinite(p50)
        if int(np.sum(m20)) > 0:
            winrate_20[idx] = float(np.mean(y20[m20]))
        if int(np.sum(m50)) > 0:
            yv = y50[m50]
            pv = p50[m50]
            winrate_50[idx] = float(np.mean(yv))
            brier_50[idx] = float(np.mean((pv - yv) ** 2))
            logloss_50[idx] = float(
                -np.mean(yv * np.log(pv) + (1.0 - yv) * np.log(1.0 - pv))
            )
            if len(yv) >= 8 and np.std(yv) > 1e-12 and np.std(pv) > 1e-12:
                rank_ic_50[idx] = float(spearmanr(pv, yv).correlation)
        if np.isfinite(y_win[idx]) and y_win[idx] < 0.5:
            current_losses += 1.0
        else:
            current_losses = 0.0
        consecutive_losses[idx] = current_losses

    for arr, name in (
        (winrate_20, "winrate_20"),
        (winrate_50, "winrate_50"),
        (brier_50, "brier_50"),
        (logloss_50, "logloss_50"),
        (rank_ic_50, "rank_ic_50"),
        (consecutive_losses, "consecutive_losses"),
    ):
        s = pd.Series(arr)
        out[name] = s.ffill().fillna(0.0).to_numpy(dtype=np.float32)

    edge_safe = np.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out["edge_tbm"] = edge_safe
    out["edge_x_winrate_20"] = edge_safe * np.asarray(
        out["winrate_20"], dtype=np.float32
    )
    out["edge_x_rank_ic_50"] = edge_safe * np.asarray(
        out["rank_ic_50"], dtype=np.float32
    )
    out["edge_x_brier_50"] = edge_safe * np.asarray(out["brier_50"], dtype=np.float32)
    for src, dst in (
        ("volatility_zscore", "edge_x_volatility_zscore"),
        ("amihud_z", "edge_x_amihud_z"),
        ("vol_of_vol", "edge_x_vol_of_vol"),
    ):
        if src in out.columns:
            out[dst] = edge_safe * np.asarray(out[src], dtype=np.float32)
    return out


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
            "reg_pred",
            "oof_ev",
            "oof_u_hat",
            "oof_log_mae_q70_hat",
            "oof_log_mfe_hat",
            "oof_asym_hat",
            "meta_avg",
            "meta_diff",
            "meta_abs_diff",
            "meta_rel_diff",
            "edge_unc_pen",
            "edge_support_pen",
            "edge_noise_pen",
        } or kl.startswith("base_h"):
            heads[k] = "return-like"
        elif kl in {
            "clf",
            "oof_p_tp",
            "oof_p_to",
            "oof_p_sl",
            "clf_center",
            "clf_entropy",
        }:
            heads[k] = "classification-like"
        elif (
            kl.startswith("robust_sigma_meta")
            or kl.startswith("cv_meta")
            or kl
            in {
                "avg_robust_sigma_meta",
                "avg_cv_meta",
                "meta_agreement_strength",
                "meta_reliability",
                "clf_prefix_std",
                "reg_prefix_std",
                "clf_leaf_support_q25",
                "reg_leaf_support_q25",
                "clf_leaf_target_iqr_mean",
                "reg_leaf_target_iqr_mean",
                "sign_agree",
                "joint_confidence",
                "conflict_score",
                "joint_instability",
            }
        ):
            heads[k] = "uncertainty-like"
        elif kl in {"meta_avg", "meta_diff", "meta_abs_diff", "meta_rel_diff"}:
            heads[k] = "return-like"
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

    if X_clean.dtype != np.float32:
        X_clean = X_clean.astype(np.float32)
    return X_clean, fit_medians, scaler, center_1d, scale_1d


def _precompute_fold_standardization(
    X: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]]
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in splits:
        X_tr = X[tr_idx]
        X_tr_clean, medians, scaler, center_1d, scale_1d = clean_and_standardize(X_tr)
        X_te_clean, _, _, _, _ = clean_and_standardize(
            X[te_idx],
            fit_medians=medians,
            scaler=scaler,
            center_1d=center_1d,
            scale_1d=scale_1d,
        )
        scaled_folds.append((tr_idx, te_idx, X_tr_clean, X_te_clean))
    return scaled_folds


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
        eval_scores, y_raw_net_return, top_fracs=(0.01, 0.025, 0.05, 0.1, 0.2)
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

    # Very simple empirical utility proxy.
    # Keep it for diagnostics, but do not use it as the HPO target.
    utility = (np.sign(top10_ret) * (np.abs(top10_ret) ** 0.5) * 10) + mono - fs_penalty
    metrics["utility_score"] = float(utility)

    return metrics


def _fit_predict_oof_regressor_with_pruning(
    *,
    X: np.ndarray,
    y: np.ndarray,
    y_downside: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_factory,
    feature_names: Optional[List[str]] = None,
    trial: Optional[optuna.trial.Trial] = None,
    trial_name: str = "",
    min_report_frac: float = 0.2,
    calibration_method: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fit a fold-local regressor with optional Optuna pruning and OOF calibration."""
    n_samples = len(y)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    observed_mask = np.zeros(n_samples, dtype=bool)
    fold_importances: List[np.ndarray] = []
    feature_names = feature_names or [f"head_{i}" for i in range(X.shape[1])]

    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[te_idx]
        w_tr = (
            np.asarray(sample_weight, dtype=np.float32)[tr_idx]
            if sample_weight is not None
            else None
        )
        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            continue

        X_tr_clean, medians, scaler, center_1d, scale_1d = clean_and_standardize(X_tr)
        X_te_clean, _, _, _, _ = clean_and_standardize(
            X_te,
            fit_medians=medians,
            scaler=scaler,
            center_1d=center_1d,
            scale_1d=scale_1d,
        )

        model = model_factory()
        if w_tr is not None:
            model.fit(X_tr_clean, y_tr, sample_weight=w_tr)
        else:
            model.fit(X_tr_clean, y_tr)
        preds = np.asarray(model.predict(X_te_clean), dtype=np.float32)
        oof_preds[te_idx] = preds
        observed_mask[te_idx] = True

        if hasattr(model, "coef_"):
            fold_importances.append(np.asarray(model.coef_, dtype=np.float32))
        elif hasattr(model, "feature_importances_"):
            fold_importances.append(
                np.asarray(model.feature_importances_, dtype=np.float32)
            )

        if trial is not None:
            seen_idx = np.flatnonzero(observed_mask)
            min_seen = max(20, int(np.ceil(min_report_frac * n_samples)))
            if seen_idx.size >= min_seen:
                interim_metrics = evaluate_signal(
                    f"{trial_name}_fold{fold_idx}",
                    oof_preds[seen_idx],
                    y[seen_idx],
                    y_downside[seen_idx],
                    directionality="return-like",
                )
                trial.report(
                    float(interim_metrics.get("utility_score", -np.inf)), step=fold_idx
                )
                if trial.should_prune():
                    raise optuna.TrialPruned()

    if calibration_method == "isotonic" and np.any(observed_mask):
        try:
            valid = observed_mask & np.isfinite(oof_preds) & np.isfinite(y)
            if valid.sum() >= 20:
                from sklearn.isotonic import IsotonicRegression

                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(oof_preds[valid], y[valid])
                oof_preds[valid] = calibrator.transform(oof_preds[valid])
        except Exception as exc:
            logger.warning(f"OOF calibration failed: {exc}")

    if fold_importances:
        importance_matrix = np.asarray(fold_importances, dtype=np.float32)
        mean_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        importance_df = pd.DataFrame(
            {
                "head_name": feature_names,
                "mean_importance": mean_importance,
                "std_importance": std_importance,
                "importance_rank": pd.Series(mean_importance)
                .rank(ascending=False)
                .values,
            }
        ).sort_values("mean_importance", ascending=False)
    else:
        importance_df = pd.DataFrame()

    return oof_preds, importance_df


def compute_period_aggregated_stats(
    trade_rets: np.ndarray,
    trade_ts: Optional[np.ndarray],
    freq: str,
) -> Tuple[float, float, float, float, float, float]:
    """Return Sortino, standard deviation, TUW, Ulcer, % negative, and worst PnL of period-aggregated PnL."""
    if trade_ts is None or len(trade_ts) == 0 or len(trade_rets) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        ts = pd.to_datetime(trade_ts, utc=True, errors="coerce")
        if isinstance(ts, pd.Series):
            valid = ts.notna().values
            ts_idx = pd.DatetimeIndex(ts[valid])
        else:
            valid = pd.notna(ts)
            ts_idx = pd.DatetimeIndex(ts[valid])
        if np.sum(valid) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rets = np.asarray(trade_rets, dtype=float)[valid]
        period_vals = pd.Series(rets).groupby(ts_idx.to_period(freq)).sum().values
        if len(period_vals) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        neg = period_vals[period_vals < 0]
        downside_std = float(np.std(neg)) if len(neg) > 0 else 1e-6
        mean_ret = float(np.mean(period_vals))
        sortino = mean_ret / downside_std if downside_std > 1e-12 else 0.0
        pnl_std = float(np.std(period_vals))

        _, dd_series = _stable_equity_and_drawdown(period_vals)
        tuw = float(np.mean(dd_series > 1e-12)) if dd_series.size else 1.0
        ulcer = (
            float(np.sqrt(np.mean(np.square(dd_series * 100.0))))
            if dd_series.size
            else 100.0
        )
        pct_negative = float(np.mean(period_vals < 0)) if len(period_vals) > 0 else 0.0
        worst_pnl = float(np.min(period_vals))

        return sortino, pnl_std, tuw, ulcer, pct_negative, worst_pnl
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


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


def _select_topk_non_concurrent(
    scores: np.ndarray,
    k: int,
    timestamps: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    horizon_hours: float = 4.0,
    max_global_concurrent: int = 3,
) -> np.ndarray:
    n = len(scores)
    if n <= 0:
        return np.array([], dtype=np.int32)
    k = max(1, min(int(k), n))
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    if (
        timestamps is None
        or symbols is None
        or len(timestamps) != n
        or len(symbols) != n
    ):
        return order[:k].astype(np.int32)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    syms = np.asarray(symbols).astype(str)
    horizon = pd.Timedelta(hours=max(1e-6, float(horizon_hours)))
    max_global = max(1, int(max_global_concurrent))
    active_trades: List[Tuple[pd.Timestamp, str]] = []
    out: List[int] = []
    for idx in order:
        if len(out) >= k:
            break
        t = ts[idx]
        if pd.isna(t):
            continue
        s = syms[idx]
        active_trades = [(e, sym) for e, sym in active_trades if e > t]
        symbol_already_active = any(sym == s for _, sym in active_trades)
        if symbol_already_active:
            continue
        if len(active_trades) >= max_global:
            continue
        out.append(int(idx))
        active_trades.append((t + horizon, s))
    if len(out) < k:
        chosen = set(out)
        for idx in order:
            if len(out) >= k:
                break
            if int(idx) not in chosen:
                out.append(int(idx))
    return np.asarray(out, dtype=np.int32)


def _weighted_period_std(
    values: np.ndarray, weights: np.ndarray, timestamps: Optional[np.ndarray], freq: str
) -> float:
    if timestamps is None or len(values) == 0:
        return 0.0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0) & pd.notna(ts)
    if int(np.sum(m)) < 4:
        return 0.0
    df = pd.DataFrame({"ts": ts[m], "v": v[m], "w": w[m]})
    grouped = df.groupby(df["ts"].dt.to_period(freq), sort=False)
    vals: List[float] = []
    for _, g in grouped:
        sw = float(np.sum(g["w"].values))
        if sw <= 1e-12:
            continue
        vals.append(float(np.sum(g["v"].values * g["w"].values) / sw))
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0


def _weighted_group_std(
    values: np.ndarray, weights: np.ndarray, groups: np.ndarray
) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    g = np.asarray(groups)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if int(np.sum(m)) < 4:
        return 0.0
    g_codes, _ = pd.factorize(g[m], sort=False)
    if len(g_codes) == 0:
        return 0.0
    w_m = w[m]
    v_m = v[m]
    sum_w = np.bincount(g_codes, weights=w_m)
    sum_vw = np.bincount(g_codes, weights=v_m * w_m)
    valid = sum_w > 1e-12
    if int(np.sum(valid)) < 2:
        return 0.0
    group_ret = np.divide(
        sum_vw[valid],
        sum_w[valid],
        out=np.zeros_like(sum_vw[valid]),
        where=sum_w[valid] > 1e-12,
    )
    group_w = sum_w[valid]
    mu = float(np.sum(group_ret * group_w) / (np.sum(group_w) + 1e-12))
    var = float(np.sum(group_w * (group_ret - mu) ** 2) / (np.sum(group_w) + 1e-12))
    return float(np.sqrt(max(var, 0.0)))


def _make_tail_weight_from_base_pred(base_pred: np.ndarray) -> np.ndarray:
    rank = (
        pd.Series(np.asarray(base_pred, dtype=np.float64))
        .rank(pct=True)
        .to_numpy(dtype=np.float64)
    )
    tail_weight = np.clip((rank - 0.80) / 0.15, 0.0, 1.0)
    tail_weight = np.where(rank >= 0.98, 1.0, tail_weight)
    return tail_weight.astype(np.float32)


def _normalized_tail_fit_weight(
    base_pred: np.ndarray, min_weight: float = 0.15
) -> np.ndarray:
    w = np.asarray(_make_tail_weight_from_base_pred(base_pred), dtype=np.float64)
    w = np.clip(w, float(min_weight), 1.0)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    w /= max(float(np.mean(w)), 1e-9)
    return w.astype(np.float32)


def _score_tail_20(
    y: np.ndarray,
    pred: np.ndarray,
    week_id: np.ndarray,
    month_id: np.ndarray,
    fixed_tail_weight: Optional[np.ndarray] = None,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    tail_weight = (
        _make_tail_weight_from_base_pred(pred)
        if fixed_tail_weight is None
        else np.asarray(fixed_tail_weight, dtype=np.float64)
    )
    avg_return_tail = float(np.sum(y * tail_weight) / (np.sum(tail_weight) + 1e-12))
    std_weekly_tail = _weighted_group_std(y, tail_weight, np.asarray(week_id))
    std_monthly_tail = _weighted_group_std(y, tail_weight, np.asarray(month_id))
    return float(avg_return_tail - 0.25 * std_weekly_tail - 0.15 * std_monthly_tail)


def _rank_weighted_tail_components(
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: Optional[np.ndarray],
    horizon_hours: float,
) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    m = np.isfinite(s) & np.isfinite(r)
    if int(np.sum(m)) < 20:
        return {
            "avg_top10_return": 0.0,
            "avg_top30_return": 0.0,
            "std_weekly_top10": 0.0,
            "std_weekly_top30": 0.0,
            "std_monthly_top10": 0.0,
            "std_monthly_top30": 0.0,
            "score_tail_20": 0.0,
            "score": 0.0,
        }
    s = s[m]
    r = r[m]
    ts = (
        np.asarray(timestamps)[m]
        if timestamps is not None and len(timestamps) == len(m)
        else None
    )
    sy = (
        np.asarray(symbols)[m]
        if symbols is not None and len(symbols) == len(m)
        else None
    )
    q = pd.Series(s).rank(pct=True, method="average").to_numpy(dtype=np.float64)

    def _band_stats(mask: np.ndarray) -> Tuple[float, float, float]:
        if int(np.sum(mask)) < 5:
            return 0.0, 0.0, 0.0
        idx = np.where(mask)[0]
        idx = idx[
            _select_topk_non_concurrent(
                s[idx],
                len(idx),
                timestamps=ts[idx] if ts is not None else None,
                symbols=sy[idx] if sy is not None else None,
                horizon_hours=horizon_hours,
            )
        ]
        qq = q[idx]
        lo, hi = float(np.min(qq)), float(np.max(qq))
        u = np.clip((qq - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        w = np.clip(np.power(u, 1.5), 1e-4, 1.0)
        avg = float(np.sum(r[idx] * w) / max(float(np.sum(w)), 1e-9))
        w_std = _weighted_period_std(
            r[idx], w, ts[idx] if ts is not None else None, "W"
        )
        m_std = _weighted_period_std(
            r[idx], w, ts[idx] if ts is not None else None, "M"
        )
        return avg, w_std, m_std

    avg30, w30, m30 = _band_stats((q >= 0.70) & (q <= 0.99))
    avg10, w10, m10 = _band_stats((q >= 0.90) & (q <= 0.99))
    score = avg10 + 0.75 * avg30 - (0.25 * w10 + 0.25 * w30) - (0.15 * m10 + 0.15 * m30)
    return {
        "avg_top10_return": float(avg10),
        "avg_top30_return": float(avg30),
        "std_weekly_top10": float(w10),
        "std_weekly_top30": float(w30),
        "std_monthly_top10": float(m10),
        "std_monthly_top30": float(m30),
        "score_tail_20": float(score),
        "score": float(score),
    }


def _update_ridge_round_weights(
    current_weight: np.ndarray,
    oof_prediction: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    w = np.asarray(current_weight, dtype=np.float64).reshape(-1)
    p = np.asarray(oof_prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    n = min(len(w), len(p), len(y))
    if n <= 0:
        return np.asarray(current_weight, dtype=np.float32)
    w, p, y = w[:n], p[:n], y[:n]
    top_focus = np.clip(
        np.asarray(_make_tail_weight_from_base_pred(p), dtype=np.float64),
        0.15,
        1.0,
    )
    confidence = np.abs(p - 0.5) * 2.0
    confidence = np.clip(confidence, 0.0, 1.0)
    strength = confidence * top_focus
    correct_mult = 1.0 + 0.15 * strength
    wrong_mult = 1.0 + 0.45 * strength
    pred_weight = np.where(y > 0.0, correct_mult, wrong_mult)
    new_weight = w * top_focus * confidence * pred_weight
    new_weight = np.nan_to_num(new_weight, nan=1.0, posinf=1.0, neginf=1.0)
    new_weight = np.clip(new_weight, 1e-3, 50.0)
    new_weight /= max(float(np.mean(new_weight)), 1e-9)
    return new_weight.astype(np.float32)


def _lightweight_hpo_pnl_eval(
    scores: np.ndarray,
    y_raw_net_return: np.ndarray,
    top_fracs: List[float],
    cost_pct: float = 0.003,
    n_days: float = 365.0,
    timestamps: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    horizon_hours: float = 4.0,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    n = len(scores)
    if n == 0:
        return results

    frac = 0.05
    k = max(1, int(n * frac))
    idx = _select_topk_non_concurrent(
        scores, k, timestamps=timestamps, symbols=symbols, horizon_hours=horizon_hours
    )
    rets = y_raw_net_return[idx]
    net_rets = rets - cost_pct

    sorted_args = np.argsort(scores[idx])
    allocations = np.linspace(0.05, 0.15, len(idx))
    wallet_rets = (rets[sorted_args] - cost_pct) * allocations
    wallet_pnl = float(np.sum(wallet_rets))

    gross_win = float(np.sum(net_rets[net_rets > 0]))
    gross_loss = float(np.abs(np.sum(net_rets[net_rets < 0])))
    pf = (
        gross_win / gross_loss
        if gross_loss > 1e-12
        else (100.0 if gross_win > 0 else 0.0)
    )

    ds = net_rets[net_rets < 0]
    ds_std = float(np.std(ds)) if len(ds) > 1 else 1e-4
    sortino = float(np.mean(net_rets)) / ds_std if ds_std > 1e-12 else 0.0

    results["wallet_pnl_0.050"] = wallet_pnl
    results["pf_0.050"] = pf
    results["sortino_0.050"] = sortino
    results["tpd_0.050"] = float(len(rets)) / n_days if n_days > 0 else 0.0

    return results


def evaluate_selection_profit_proxy(
    scores: np.ndarray,
    y_raw_net_return: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    symbols: Optional[np.ndarray] = None,
    top_fracs: List[float] = [
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
    ],
    start_equity: float = 100000.0,
    cost_pct: float = 0.003,
    n_days: float = 365.0,
    wallet_range: Tuple[float, float] = (0.05, 0.15),
    horizon_hours: float = 4.0,
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
        idx = _select_topk_non_concurrent(
            scores=scores,
            k=k,
            timestamps=timestamps,
            symbols=symbols,
            horizon_hours=horizon_hours,
        )

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

        (
            weekly_sortino,
            weekly_pnl_std,
            weekly_tuw,
            weekly_ulcer,
            weekly_pct_negative,
            weekly_worst_pnl,
        ) = compute_period_aggregated_stats(sized_rets, selected_ts, "W")
        (
            monthly_sortino,
            monthly_pnl_std,
            monthly_tuw,
            monthly_ulcer,
            monthly_pct_negative,
            monthly_worst_pnl,
        ) = compute_period_aggregated_stats(sized_rets, selected_ts, "M")
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
        pnl_per_trade_pct = (
            (net_pnl / len(sized_rets)) * 100.0 if len(sized_rets) > 0 else 0.0
        )
        trades_per_day = len(sized_rets) / n_days if n_days > 0 else 0.0

        results.append(
            {
                "selection_frac": frac,
                "threshold_pct": f"{100*(1-frac):.1f}%",
                "net_pnl": net_pnl,
                "wallet_pnl": net_wallet_pnl_pct,
                "pnl_per_trade": pnl_per_trade_pct,
                "pnl_per_trade_pct": pnl_per_trade_pct,
                "trades_per_day": trades_per_day,
                "hit_rate": hit_rate,
                "profit_factor": profit_factor,
                "sortino": sortino,
                "weekly_sortino": weekly_sortino,
                "monthly_sortino": monthly_sortino,
                "weekly_pnl_std": weekly_pnl_std,
                "monthly_pnl_std": monthly_pnl_std,
                "weekly_tuw": weekly_tuw,
                "weekly_ulcer": weekly_ulcer,
                "weekly_pct_negative": weekly_pct_negative,
                "weekly_worst_pnl": weekly_worst_pnl,
                "monthly_tuw": monthly_tuw,
                "monthly_ulcer": monthly_ulcer,
                "monthly_pct_negative": monthly_pct_negative,
                "monthly_worst_pnl": monthly_worst_pnl,
                "monthly_group_std_pnl": month_stability["group_std_pnl"],
                "monthly_group_cv_pnl": month_stability["group_cv_pnl"],
                "monthly_group_pf_std": month_stability["group_pf_std"],
                "asset_group_std_pnl": asset_stability["group_std_pnl"],
                "asset_group_cv_pnl": asset_stability["group_cv_pnl"],
                "asset_group_positive_share": asset_stability["group_positive_share"],
                "asset_group_pf_std": asset_stability["group_pf_std"],
                "stability": stability,
                "max_drawdown": mdd_pct,
                "calmar_ratio": net_pnl / mdd_pct if mdd_pct > 1e-9 else float("inf"),
                "expectancy_tstat": (
                    mean_ret / float(np.std(sized_rets))
                    if len(sized_rets) > 1 and float(np.std(sized_rets)) > 1e-9
                    else 0.0
                ),
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
        idx_opt = _select_topk_non_concurrent(
            scores=scores,
            k=k_opt,
            timestamps=timestamps,
            symbols=symbols,
            horizon_hours=horizon_hours,
        )
        opt_rets = y_raw_net_return[idx_opt] - cost_pct
        if timestamps is not None and len(timestamps) == n_samples:
            opt_ts = np.asarray(timestamps)[idx_opt]

    return df, opt_rets, opt_ts


def _mdi_select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_factory: Callable[[], Any],
    n_max: int = 60,
    n_min: int = 10,
    cumulative_threshold: float = 0.99,
) -> List[str]:
    _valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean, y_clean = X[_valid_mask], y[_valid_mask]
    n_mdi = min(len(X_clean), 15000)
    if n_mdi < len(X_clean):
        _mdi_rng = np.random.RandomState(42)
        _mdi_idx = _mdi_rng.choice(len(X_clean), n_mdi, replace=False)
        X_mdi, y_mdi = X_clean[_mdi_idx], y_clean[_mdi_idx]
    else:
        X_mdi, y_mdi = X_clean, y_clean
    if len(X_mdi) < 100:
        return list(feature_names[:n_min])
    preliminary = model_factory()
    try:
        preliminary.fit(X_mdi, y_mdi)
    except TypeError as exc:
        if "force_all_finite" not in str(exc):
            raise
        tprint(
            "  MDI feature selection: LightGBM/sklearn validation API mismatch; "
            "falling back to ExtraTrees importance."
        )
        preliminary = ExtraTreesRegressor(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=max(20, int(0.01 * len(X_mdi))),
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        preliminary.fit(X_mdi, y_mdi)
    if not hasattr(preliminary, "feature_importances_"):
        return list(feature_names)
    importances = np.asarray(preliminary.feature_importances_, dtype=np.float64)
    n_max = min(n_max, len(feature_names))
    n_min = min(n_min, n_max)
    order = np.argsort(importances)[::-1]
    sorted_imp = importances[order]
    total = sorted_imp.sum()
    if total < 1e-12:
        return list(feature_names[:n_min])
    cumsum = np.cumsum(sorted_imp) / total
    elbow_idx = int(np.searchsorted(cumsum, cumulative_threshold)) + 1
    n_select = max(n_min, min(elbow_idx, n_max))
    top_idx = order[:n_select]
    return [feature_names[i] for i in sorted(top_idx)]


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
    use_et_head_sizer: bool = False,
    use_lgbm_head_sizer: bool = True,
    use_barrier_classifier: bool = True,
    config_feature_keys: Optional[List[str]] = None,
    y_atr_target: Optional[np.ndarray] = None,
    atr_vals: Optional[np.ndarray] = None,
    target_horizon: int = 1,
) -> Dict[str, Any]:
    """
    Main orchestrator for the simple position sizer diagnostic framework.
    By default runs both Ridge and ExtraTrees, compares them, and selects the best.
    """
    if lambda_grid is None:
        lambda_grid = [0.25, 0.5, 1.0, 2.0]

    y_model_target = y_atr_target if y_atr_target is not None else y_raw_net_return
    _atr_safe: np.ndarray = (
        np.where(
            np.isfinite(atr_vals) & (np.abs(atr_vals) > 1e-8), atr_vals, 1.0
        ).astype(np.float32)
        if atr_vals is not None
        else np.ones(len(y_raw_net_return), dtype=np.float32)
    )
    _use_atr = atr_vals is not None and y_atr_target is not None
    if y_atr_target is not None:
        tprint(
            f"  ATR-normalized target: "
            f"mean={float(np.nanmean(y_model_target)):.6f}, "
            f"std={float(np.nanstd(y_model_target)):.6f}, "
            f"raw mean={float(np.nanmean(y_raw_net_return)):.6f}, "
            f"ATR mean={float(np.mean(_atr_safe)):.6f}"
        )

    # ExtraTrees head race removed by design (keep flag for backward compatibility).
    use_et_head_sizer = False

    detected_heads = detect_meta_head_keys(
        feature_dict, config_overrides=config_feature_keys
    )
    used_keys = [
        k
        for k in detected_heads.keys()
        if k in feature_dict and k not in _LIVE_UNAVAILABLE_SIZER_FEATURES
    ]
    missing_keys = [k for k in detected_heads.keys() if k not in feature_dict]
    dropped_live_unavailable = [
        k for k in detected_heads.keys() if k in _LIVE_UNAVAILABLE_SIZER_FEATURES
    ]
    if dropped_live_unavailable:
        tprint(
            "  Dropped live-unavailable target-derived sizer features: "
            f"{dropped_live_unavailable}"
        )

    feature_coverage_report = {
        "detected_candidates": list(detected_heads.keys()),
        "used_heads": used_keys,
        "missing_heads": missing_keys,
        "dropped_live_unavailable": dropped_live_unavailable,
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
    best_combo_profit_proxy_df = pd.DataFrame()
    best_combo_objective = float("-inf")
    combined_et_objective = float("-inf")

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

    if best_simple_score is not None:
        best_combo_profit_proxy_df, _, _ = evaluate_selection_profit_proxy(
            best_simple_score,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        if not best_combo_profit_proxy_df.empty:
            best_combo_objective = float(best_combo_profit_proxy_df["wallet_pnl"].max())

    # --- ElasticNet Sizer (replaces Ridge) ---
    ridge_sizer_eval: Dict[str, Any] = {}
    ridge_importance_df = pd.DataFrame()
    ridge_profit_proxy_df = pd.DataFrame()

    _hpo_max_rows_per_fold = 12000
    _hpo_rng = np.random.RandomState(42)
    _hpo_tr_idx_subs: List[np.ndarray] = []
    for _tr_idx, _te_idx in splits:
        if len(_tr_idx) > _hpo_max_rows_per_fold:
            _hpo_tr_idx_subs.append(
                _hpo_rng.choice(_tr_idx, _hpo_max_rows_per_fold, replace=False)
            )
        else:
            _hpo_tr_idx_subs.append(_tr_idx)

    if use_ridge_head_sizer and used_keys:
        X_heads = np.column_stack([feature_dict[k] for k in used_keys])
        ridge_round_weights = (
            np.ones(len(y_raw_net_return), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32)
        )
        _n_constant = 0
        _n_near_constant = 0
        for _fi, _fk in enumerate(used_keys):
            _col = X_heads[:, _fi]
            _std = float(np.nanstd(_col))
            _n_uniq = int(len(np.unique(_col[~np.isnan(_col)])))
            if _std < 1e-12 or _n_uniq <= 1:
                tprint(
                    f"  WARNING: feature '{_fk}' is constant (std={_std:.2e}, unique={_n_uniq})"
                )
                _n_constant += 1
            elif _n_uniq <= 3:
                tprint(
                    f"  WARNING: feature '{_fk}' is near-constant (std={_std:.4f}, unique={_n_uniq})"
                )
                _n_near_constant += 1
        if _n_constant > 0 or _n_near_constant > 0:
            tprint(
                f"  Feature quality: {_n_constant} constant, {_n_near_constant} near-constant out of {len(used_keys)} features"
            )
        best_ridge_utility = -np.inf
        best_ridge_alpha = 1.0
        best_ridge_l1_ratio = 0.0
        best_ridge_preds = None
        best_ridge_importance = pd.DataFrame()
        best_ridge_metrics = {}
        best_ridge_profit_proxy = pd.DataFrame()
        best_ridge_opt_rets = np.array([])
        best_ridge_opt_ts = np.array([])

        ridge_trials = 40
        optuna_patience_trials = 40

        def _make_patience_callback(
            *, patience: int, label: str, min_trials: int = 0
        ) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
            best_value = float("-inf")
            best_trial_number = -1
            meaningful_improvement_threshold = 1.005
            last_meaningful_improvement_trial = -1
            best_value_at_meaningful_check = float("-inf")

            def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                nonlocal best_value, best_trial_number, last_meaningful_improvement_trial, best_value_at_meaningful_check
                values = trial.values or []
                current_trial_number = int(trial.number)

                if current_trial_number < min_trials:
                    return

                if values:
                    current = float(values[0])
                    if np.isfinite(current):
                        if current > best_value:
                            best_value = current
                            best_trial_number = current_trial_number
                        if (
                            current
                            > best_value_at_meaningful_check
                            * meaningful_improvement_threshold
                        ):
                            best_value_at_meaningful_check = current
                            last_meaningful_improvement_trial = current_trial_number

                if (
                    best_trial_number >= 0
                    and (current_trial_number - best_trial_number) >= patience
                ):
                    tprint(
                        f"{label}: early stopping after {patience} trials without improvement "
                        f"(best={best_value:.6f}, last_improved_trial={best_trial_number})"
                    )
                    study.stop()
                    return

                extended_patience = int(patience * 1.5)
                if (
                    last_meaningful_improvement_trial >= 0
                    and (current_trial_number - last_meaningful_improvement_trial)
                    >= extended_patience
                ):
                    tprint(
                        f"{label}: early stopping after {extended_patience} trials without meaningful "
                        f"improvement (>0.5% gain) "
                        f"(best={best_value:.6f}, last_meaningful_trial={last_meaningful_improvement_trial})"
                    )
                    study.stop()

            return _callback

        def _make_median_pruner() -> MedianPruner:
            return MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=0,
                interval_steps=1,
                n_min_trials=5,
            )

        ridge_alpha_choices = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 5.0]
        ridge_l1_choices = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]

        def _best_ridge_model_factory(alpha: float, l1_ratio: float) -> ElasticNet:
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=10000,
                tol=1e-4,
                random_state=42,
                selection="cyclic",
            )

        def _build_sub_scaled_folds(
            x_round: np.ndarray,
        ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for i, (_tr_idx, te_idx) in enumerate(splits):
                sub_tr = _hpo_tr_idx_subs[i]
                x_sub = x_round[sub_tr]
                x_sub_clean, med, sc, c1d, s1d = clean_and_standardize(x_sub)
                x_te_clean, _, _, _, _ = clean_and_standardize(
                    x_round[te_idx],
                    fit_medians=med,
                    scaler=sc,
                    center_1d=c1d,
                    scale_1d=s1d,
                )
                out.append((sub_tr, te_idx, x_sub_clean, x_te_clean))
            return out

        def _optimize_ridge_hpo(
            x_round: np.ndarray,
            feature_count: int,
            round_weights: np.ndarray,
            round_label: str,
        ) -> Tuple[float, float]:
            sub_scaled = _build_sub_scaled_folds(x_round)
            round_study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42, multivariate=True, group=True),
                pruner=_make_median_pruner(),
            )

            def _ridge_objective(trial: optuna.trial.Trial) -> float:
                alpha = float(trial.suggest_categorical("alpha", ridge_alpha_choices))
                l1_ratio = float(
                    trial.suggest_categorical("l1_ratio", ridge_l1_choices)
                )
                fold_scores: List[float] = []
                for fold_idx in range(len(splits)):
                    sub_tr, te_idx, x_tr_clean, x_te_clean = sub_scaled[fold_idx]
                    if len(sub_tr) == 0 or len(te_idx) == 0:
                        continue
                    model = _best_ridge_model_factory(alpha=alpha, l1_ratio=l1_ratio)
                    model.fit(
                        x_tr_clean,
                        y_model_target[sub_tr],
                        sample_weight=np.asarray(
                            round_weights[sub_tr], dtype=np.float32
                        ),
                    )
                    fold_preds = np.asarray(model.predict(x_te_clean), dtype=np.float32)
                    tail = _rank_weighted_tail_components(
                        scores=fold_preds,
                        returns=y_raw_net_return[te_idx],
                        timestamps=np.asarray(timestamps)[te_idx],
                        symbols=(
                            np.asarray(_sym_vals)[te_idx]
                            if _sym_vals is not None
                            and len(_sym_vals) == len(y_raw_net_return)
                            else None
                        ),
                        horizon_hours=float(max(1, target_horizon)),
                    )
                    fold_scores.append(float(tail.get("score_tail_20", 0.0)))
                    if trial is not None and fold_idx >= len(splits) // 2:
                        trial.report(
                            float(np.mean(fold_scores)) if fold_scores else -1e9,
                            step=fold_idx,
                        )
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                if not fold_scores:
                    return -1e9
                composite = float(np.mean(fold_scores))
                trial.set_user_attr("feature_count", int(feature_count))
                trial.set_user_attr("score_tail_20_mean", composite)
                trial.set_user_attr("composite_score", composite)
                return composite

            round_study.optimize(
                _ridge_objective,
                n_trials=ridge_trials,
                gc_after_trial=True,
                callbacks=[
                    _make_patience_callback(
                        patience=optuna_patience_trials,
                        label=f"ElasticNet HPO {round_label}",
                    )
                ],
            )
            if round_study.best_trial is None:
                return 1.0, 0.0
            return (
                float(round_study.best_trial.params.get("alpha", 1.0)),
                float(round_study.best_trial.params.get("l1_ratio", 0.0)),
            )

        # 1) Feature-selection fit (pre-HPO), then weight update.
        seed_alpha = 0.1
        seed_l1_ratio = 0.05
        tprint("  ElasticNet pre-HPO feature-selection fit...")
        (
            fs_oof_preds,
            fs_importance,
        ) = _fit_predict_oof_regressor_with_pruning(
            X=X_heads,
            y=y_model_target,
            y_downside=y_downside,
            splits=splits,
            model_factory=lambda: _best_ridge_model_factory(
                alpha=seed_alpha, l1_ratio=seed_l1_ratio
            ),
            feature_names=list(used_keys),
            calibration_method=None,
            sample_weight=ridge_round_weights,
        )
        ridge_round_weights = _update_ridge_round_weights(
            ridge_round_weights, fs_oof_preds, y_raw_net_return
        )
        round_feature_names = list(used_keys)
        if not fs_importance.empty and "head_name" in fs_importance.columns:
            imp_df = fs_importance.copy()
            if "abs_weight" in imp_df.columns:
                imp_df = imp_df.sort_values("abs_weight", ascending=False)
                non_zero = (
                    imp_df[imp_df["abs_weight"] > 1e-8]["head_name"]
                    .astype(str)
                    .tolist()
                )
                if len(non_zero) >= 5:
                    round_feature_names = non_zero
        round_feature_idx = [
            used_keys.index(k) for k in round_feature_names if k in used_keys
        ]
        if len(round_feature_idx) == 0:
            round_feature_idx = list(range(X_heads.shape[1]))
            round_feature_names = list(used_keys)
        X_heads_selected = X_heads[:, round_feature_idx]

        # 2) Single HPO on selected features with distilled weights.
        best_ridge_alpha, best_ridge_l1_ratio = _optimize_ridge_hpo(
            X_heads_selected, len(round_feature_names), ridge_round_weights, "Final"
        )

        # 3) Two refits with weight updates between rounds.
        tprint("  Starting ElasticNet OOF refit Round-1...")
        (
            best_ridge_preds,
            best_ridge_importance,
        ) = _fit_predict_oof_regressor_with_pruning(
            X=X_heads_selected,
            y=y_model_target,
            y_downside=y_downside,
            splits=splits,
            model_factory=lambda: _best_ridge_model_factory(
                alpha=best_ridge_alpha, l1_ratio=best_ridge_l1_ratio
            ),
            feature_names=round_feature_names,
            calibration_method=None,
            sample_weight=ridge_round_weights,
        )
        tprint("  ElasticNet OOF refit Round-1 complete.")
        ridge_round1_tail = _rank_weighted_tail_components(
            scores=best_ridge_preds,
            returns=y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            horizon_hours=float(max(1, target_horizon)),
        )
        tprint(
            f"  ElasticNet Round-1 score_tail_20={ridge_round1_tail['score_tail_20']:.6f}"
        )
        ridge_round_weights = _update_ridge_round_weights(
            ridge_round_weights, best_ridge_preds, y_raw_net_return
        )
        tprint("  Starting ElasticNet Round-2 OOF refit (distilled weights)...")
        (
            best_ridge_preds,
            best_ridge_importance,
        ) = _fit_predict_oof_regressor_with_pruning(
            X=X_heads_selected,
            y=y_model_target,
            y_downside=y_downside,
            splits=splits,
            model_factory=lambda: _best_ridge_model_factory(
                alpha=best_ridge_alpha, l1_ratio=best_ridge_l1_ratio
            ),
            feature_names=round_feature_names,
            calibration_method=None,
            sample_weight=ridge_round_weights,
        )
        ridge_round2_tail = _rank_weighted_tail_components(
            scores=best_ridge_preds,
            returns=y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            horizon_hours=float(max(1, target_horizon)),
        )
        tprint(
            f"  ElasticNet Round-2 score_tail_20={ridge_round2_tail['score_tail_20']:.6f}"
        )
        ridge_round_weights = _update_ridge_round_weights(
            ridge_round_weights, best_ridge_preds, y_raw_net_return
        )
        # Persist final distilled model fitted on full data for downstream usage.
        (
            X_full_clean,
            full_medians,
            full_scaler,
            full_center_1d,
            full_scale_1d,
        ) = clean_and_standardize(X_heads_selected)
        ridge_final_model = _best_ridge_model_factory(
            alpha=best_ridge_alpha, l1_ratio=best_ridge_l1_ratio
        )
        ridge_final_model.fit(
            X_full_clean,
            y_model_target,
            sample_weight=np.asarray(ridge_round_weights, dtype=np.float32),
        )
        best_ridge_metrics = evaluate_signal(
            f"ElasticNet_Head_Sizer(a={best_ridge_alpha},l1={best_ridge_l1_ratio})",
            best_ridge_preds,
            y_raw_net_return,
            y_downside,
            directionality="return-like",
        )
        best_ridge_utility = best_ridge_metrics.get("utility_score", -np.inf)
        (
            best_ridge_profit_proxy,
            best_ridge_opt_rets,
            best_ridge_opt_ts,
        ) = evaluate_selection_profit_proxy(
            best_ridge_preds,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        best_ridge_objective = (
            float(best_ridge_profit_proxy["wallet_pnl"].max())
            if not best_ridge_profit_proxy.empty
            else float("-inf")
        )

        tprint(
            f"ElasticNet HPO winner: alpha={best_ridge_alpha}, l1_ratio={best_ridge_l1_ratio} "
            f"(utility={best_ridge_utility:.4f}, wallet_pnl={best_ridge_objective:.4f})"
        )
        if not best_ridge_importance.empty:
            tprint("=== ElasticNet Feature Importance (top 20) ===")
            _imp = best_ridge_importance.copy()
            if "abs_weight" not in _imp.columns:
                if "mean_weight" in _imp.columns:
                    _imp["abs_weight"] = _imp["mean_weight"].abs()
                elif "mean_importance" in _imp.columns:
                    _imp["abs_weight"] = _imp["mean_importance"].abs()
                else:
                    _imp["abs_weight"] = 0.0
            name_col = "head_name" if "head_name" in _imp.columns else "feature"
            weight_col = (
                "mean_weight"
                if "mean_weight" in _imp.columns
                else (
                    "mean_importance"
                    if "mean_importance" in _imp.columns
                    else "abs_weight"
                )
            )
            std_col = "std_weight" if "std_weight" in _imp.columns else None
            _imp = _imp.sort_values("abs_weight", ascending=False).head(20)
            for _i, _row in _imp.iterrows():
                std_str = ""
                if std_col is not None and pd.notna(_row.get(std_col, np.nan)):
                    std_str = f"  std={float(_row[std_col]):.6f}"
                tprint(
                    f"  {_i+1:>3}. {_row[name_col]:<40} "
                    f"weight={float(_row[weight_col]):+.6f}{std_str}"
                )
            tprint("=== End ElasticNet Importance ===")
        ridge_oof_preds = best_ridge_preds
        ridge_sizer_eval = best_ridge_metrics
        ridge_importance_df = best_ridge_importance
        ridge_profit_proxy_df = best_ridge_profit_proxy
        results["ridge_sizer_scores_"] = ridge_oof_preds
        results["ridge_importance_table_"] = ridge_importance_df
        results["ridge_profit_proxy_table_"] = ridge_profit_proxy_df
        results["ridge_opt_rets_"] = best_ridge_opt_rets
        results["ridge_opt_ts_"] = best_ridge_opt_ts
        results["ridge_score_tail_20_round1_"] = float(
            ridge_round1_tail["score_tail_20"]
        )
        results["ridge_score_tail_20_round2_"] = float(
            ridge_round2_tail["score_tail_20"]
        )
        results["ridge_tail_score_round1_"] = float(ridge_round1_tail["score_tail_20"])
        results["ridge_tail_score_round2_"] = float(ridge_round2_tail["score_tail_20"])
        results["ridge_distilled_round_weights_"] = np.asarray(
            ridge_round_weights, dtype=np.float32
        )
        results["ridge_selected_feature_names_"] = list(round_feature_names)
        results["ridge_selected_feature_idx_"] = np.asarray(
            round_feature_idx, dtype=np.int32
        )
        results["ridge_final_model_"] = ridge_final_model
        results["ridge_final_model_preproc_"] = {
            "medians": full_medians,
            "scaler": full_scaler,
            "center_1d": full_center_1d,
            "scale_1d": full_scale_1d,
        }
        if not best_combo or best_ridge_objective > best_combo_objective:
            best_simple_score = ridge_oof_preds
            best_simple_score_name = "ElasticNet_Head_Sizer"

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
        tprint("  Starting Barrier Classifier OOF...")
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
    lgbm_sizer_eval: Dict[str, Any] = {}
    lgbm_importance_df = pd.DataFrame()
    lgbm_profit_proxy_df = pd.DataFrame()
    lgbm_clf_sizer_eval: Dict[str, Any] = {}
    lgbm_clf_importance_df = pd.DataFrame()
    lgbm_clf_profit_proxy_df = pd.DataFrame()

    _use_any_booster = (use_et_head_sizer or use_lgbm_head_sizer) and used_keys

    if _use_any_booster:
        X_heads = (
            X_heads
            if use_ridge_head_sizer and used_keys
            else np.column_stack([feature_dict[k] for k in used_keys])
        )

        ridge_oof_available = (
            use_ridge_head_sizer
            and ridge_oof_preds is not None
            and len(ridge_oof_preds) == len(y_raw_net_return)
        )

        if ridge_oof_available:
            ridge_resid = np.asarray(y_raw_net_return, dtype=np.float32) - np.asarray(
                ridge_oof_preds, dtype=np.float32
            )
            resid_atr = ridge_resid / (_atr_safe + 1e-8)
            resid_mad = _rolling_mad(
                resid_atr,
                window=max(1, int(3 * max(1, target_horizon))),
            )
            resid_z = resid_atr / (resid_mad + 1e-8)
            resid_z = np.clip(resid_z, -5.0, 5.0).astype(np.float32)
            if not np.isfinite(resid_z).all():
                finite_resid_z = resid_z[np.isfinite(resid_z)]
                fill_resid_z = (
                    float(np.nanmedian(finite_resid_z)) if finite_resid_z.size else 0.0
                )
                resid_z = np.where(np.isfinite(resid_z), resid_z, fill_resid_z).astype(
                    np.float32
                )
            y_resid_target = (0.7 * np.arcsinh(resid_z) + 0.3 * resid_z).astype(
                np.float32
            )
            ridge_resid_mean = float(np.nanmean(ridge_resid))
            ridge_resid_std = float(np.nanstd(ridge_resid)) + 1e-6
            tprint(
                f"  Residual target: resid_mean={ridge_resid_mean:.6f}, "
                f"resid_std={ridge_resid_std:.6f}, "
                f"target_mean={float(np.nanmean(y_resid_target)):.6f}"
            )
        else:
            y_resid_target = y_model_target
            ridge_resid_mean = 0.0
            ridge_resid_std = 1.0
            tprint("  No Ridge OOF available — training boosters on raw target")

        X_heads_clean_m, medians_m, scaler_m, c1d_m, s1d_m = clean_and_standardize(
            X_heads
        )

    def _quantile_scale_to_band(
        arr: np.ndarray, low: float = 0.7, high: float = 1.3
    ) -> np.ndarray:
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.float32)
        ranks = np.argsort(np.argsort(arr)).astype(np.float64) / max(n - 1, 1)
        delta_r = ranks - 0.5
        mapped = 1.0 + np.sign(delta_r) * 1.2 * (delta_r**2)
        mapped = np.clip(mapped, low, high)
        median_val = float(np.median(mapped))
        if abs(median_val) > 1e-12:
            mapped = mapped * (1.0 / median_val)
        mapped = np.clip(mapped, low, high)
        return mapped.astype(np.float32)

    if use_et_head_sizer and used_keys:
        best_et_utility = -np.inf
        best_et_preds = None
        best_et_importance = pd.DataFrame()
        best_et_metrics = {}
        best_et_profit_proxy = pd.DataFrame()
        best_et_opt_rets = np.array([])
        best_et_opt_ts = np.array([])
        best_et_params = {}

        et_selected_keys = _mdi_select_features(
            X_heads_clean_m,
            y_resid_target,
            used_keys,
            model_factory=lambda: ExtraTreesRegressor(
                n_estimators=200, max_depth=7, random_state=42, n_jobs=2
            ),
        )
        et_selected_idx = [
            used_keys.index(k) for k in et_selected_keys if k in used_keys
        ]
        X_et = X_heads[:, et_selected_idx]
        tprint(
            f"  ET MDI feature selection: {len(et_selected_keys)}/{len(used_keys)} features retained"
        )

        et_sub_scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i, (tr_idx, te_idx) in enumerate(splits):
            sub_tr = _hpo_tr_idx_subs[i]
            X_sub = X_et[sub_tr]
            X_sub_clean, med, sc, c1d, s1d = clean_and_standardize(X_sub)
            X_te_clean, _, _, _, _ = clean_and_standardize(
                X_et[te_idx], fit_medians=med, scaler=sc, center_1d=c1d, scale_1d=s1d
            )
            et_sub_scaled_folds.append((sub_tr, te_idx, X_sub_clean, X_te_clean))

        et_trials = 150
        et_sampler = TPESampler(seed=42, multivariate=True, group=True)
        et_pruner = _make_median_pruner()
        et_study = optuna.create_study(
            direction="maximize", sampler=et_sampler, pruner=et_pruner
        )

        et_n_estimators_choices = [200, 300, 400, 500, 600, 800, 1000]
        et_max_depth_choices = [3, 4, 5, 6, 7]
        et_criterion_choices = ["absolute_error"]

        def _compute_et_confidence(
            model: ExtraTreesRegressor,
            X: np.ndarray,
        ) -> np.ndarray:
            n_trees = len(model.estimators_)
            n_samples = X.shape[0]
            leaf_ids = np.empty((n_samples, n_trees), dtype=np.int32)
            tree_preds = np.empty((n_samples, n_trees), dtype=np.float32)
            for t_idx, tree in enumerate(model.estimators_):
                leaf_ids[:, t_idx] = tree.apply(X)
                tree_preds[:, t_idx] = tree.predict(X).astype(np.float32)

            tree_mean = np.mean(tree_preds, axis=1)
            tree_std = np.std(tree_preds, axis=1).astype(np.float32)
            tree_disagreement = tree_std / (np.abs(tree_mean) + 1e-6)

            leaf_support = np.zeros(n_samples, dtype=np.float32)
            centroid_dist = np.zeros(n_samples, dtype=np.float32)
            for t_idx in range(n_trees):
                leaves = leaf_ids[:, t_idx]
                leaves_offset = leaves - leaves.min()
                n_bins = int(leaves_offset.max()) + 1
                counts = np.bincount(leaves_offset, minlength=n_bins).astype(np.float32)
                sums = np.bincount(
                    leaves_offset, weights=tree_preds[:, t_idx], minlength=n_bins
                )
                leaf_means = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)
                sample_counts = counts[leaves_offset]
                sample_means = leaf_means[leaves_offset]
                leaf_support += sample_counts
                centroid_dist += np.abs(tree_preds[:, t_idx] - sample_means)
            leaf_support /= n_trees
            centroid_dist /= n_trees

            td_inv = 1.0 / (tree_disagreement + 1e-12)
            cd_inv = 1.0 / (centroid_dist + 1e-12)
            td_scaled = _quantile_scale_to_band(td_inv)
            ls_scaled = _quantile_scale_to_band(leaf_support)
            cd_scaled = _quantile_scale_to_band(cd_inv)

            log_conf = (
                3.0 * np.log(td_scaled + 1e-12)
                + 2.0 * np.log(ls_scaled + 1e-12)
                + 1.0 * np.log(cd_scaled + 1e-12)
            ) / 6.0
            raw_wgm = np.exp(log_conf).astype(np.float32)
            median_wgm = float(np.median(raw_wgm))
            if abs(median_wgm) > 1e-12:
                et_confidence = np.clip(raw_wgm / median_wgm, 0.7, 1.3).astype(
                    np.float32
                )
            else:
                et_confidence = np.ones_like(raw_wgm)
            return et_confidence

        def _et_objective(trial: optuna.trial.Trial) -> float:
            n_estimators = int(
                trial.suggest_categorical("n_estimators", et_n_estimators_choices)
            )
            max_depth = int(
                trial.suggest_categorical("max_depth", et_max_depth_choices)
            )
            max_features = float(trial.suggest_float("max_features", 0.2, 0.7))
            ccp_alpha = 1e-6
            min_impurity_decrease = 1e-6
            criterion = trial.suggest_categorical("criterion", et_criterion_choices)
            min_samples_leaf_frac = float(
                trial.suggest_float("min_samples_leaf_frac", 0.01, 0.05)
            )
            min_samples_split_frac = float(
                trial.suggest_float("min_samples_split_frac", 0.01, 0.05)
            )
            min_samples_leaf = max(
                1, int(np.ceil(min_samples_leaf_frac * X_et.shape[0]))
            )
            min_samples_split = max(
                min_samples_leaf + 1,
                int(np.ceil(min_samples_split_frac * X_et.shape[0])),
            )

            fold_pnl_5: List[float] = []
            fold_sortino_5: List[float] = []

            for fold_idx in range(len(splits)):
                sub_tr, te_idx, X_tr_clean, X_te_clean = et_sub_scaled_folds[fold_idx]
                if len(sub_tr) == 0 or len(te_idx) == 0:
                    continue
                y_tr = y_resid_target[sub_tr]
                y_te = y_raw_net_return[te_idx]

                model = ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    criterion=criterion,
                    ccp_alpha=ccp_alpha,
                    min_impurity_decrease=min_impurity_decrease,
                    bootstrap=True,
                    oob_score=False,
                    random_state=42,
                    n_jobs=2,
                    verbose=0,
                )
                model.fit(X_tr_clean, y_tr)
                et_fold_preds = np.asarray(model.predict(X_te_clean), dtype=np.float32)
                et_conf_fold = _compute_et_confidence(model, X_te_clean)

                if ridge_oof_available:
                    ridge_fold_preds = ridge_oof_preds[te_idx]
                    _combined_hpo = (
                        ridge_fold_preds + 0.5 * et_fold_preds * et_conf_fold
                    )
                    if _use_atr:
                        raw_conf = np.clip(
                            _combined_hpo * _atr_safe[te_idx],
                            0.0,
                            3.0 * _atr_safe[te_idx],
                        ).astype(np.float32)
                    else:
                        raw_conf = np.maximum(
                            0.0,
                            _combined_hpo * ridge_resid_std,
                        ).astype(np.float32)
                else:
                    raw_conf = (
                        et_fold_preds * _atr_safe[te_idx] if _use_atr else et_fold_preds
                    )

                n_days_fold = max(1.0, len(te_idx) / 96.0)
                lw = _lightweight_hpo_pnl_eval(
                    raw_conf,
                    y_raw_net_return[te_idx],
                    top_fracs=[0.05],
                    cost_pct=cost_pct,
                    n_days=n_days_fold,
                )
                if "wallet_pnl_0.050" in lw:
                    fold_pnl_5.append(lw["wallet_pnl_0.050"])
                if "sortino_0.050" in lw:
                    fold_sortino_5.append(lw["sortino_0.050"])

                if trial is not None and fold_idx >= len(splits) // 2:
                    trial.report(
                        float(np.mean(fold_pnl_5)) if fold_pnl_5 else -1e9,
                        step=fold_idx,
                    )
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if not fold_pnl_5:
                return -1e9

            mean_pnl_5 = float(np.mean(fold_pnl_5))
            std_pnl_5 = float(np.std(fold_pnl_5))
            sortino_5 = float(np.mean(fold_sortino_5)) if fold_sortino_5 else 0.0

            composite = 0.75 * mean_pnl_5 - 0.50 * std_pnl_5 + 0.25 * sortino_5

            trial.set_user_attr("feature_count", int(X_heads.shape[1]))
            trial.set_user_attr("mean_pnl_5", mean_pnl_5)
            trial.set_user_attr("std_pnl_5", std_pnl_5)
            trial.set_user_attr("sortino_5", sortino_5)
            trial.set_user_attr("composite_score", composite)

            return float(composite)

        et_study.optimize(
            _et_objective,
            n_trials=et_trials,
            gc_after_trial=True,
            callbacks=[
                _make_patience_callback(
                    patience=optuna_patience_trials, label="ExtraTrees Residual HPO"
                )
            ],
        )
        if et_study.best_trial is not None:
            best_et_params = dict(et_study.best_trial.params)

        best_et_n_estimators = int(best_et_params.get("n_estimators", 200))
        best_et_max_depth = int(best_et_params.get("max_depth", 5))
        best_et_max_features = best_et_params.get("max_features", "sqrt")
        best_et_ccp_alpha = float(best_et_params.get("ccp_alpha", 1e-5))
        best_et_min_impurity = float(best_et_params.get("min_impurity_decrease", 1e-6))
        best_et_criterion = str(best_et_params.get("criterion", "absolute_error"))
        best_et_min_samples_leaf = max(
            1,
            int(
                np.ceil(
                    float(best_et_params.get("min_samples_leaf_frac", 0.02))
                    * X_et.shape[0]
                )
            ),
        )
        best_et_min_samples_split = max(
            best_et_min_samples_leaf + 1,
            int(
                np.ceil(
                    float(best_et_params.get("min_samples_split_frac", 0.02))
                    * X_et.shape[0]
                )
            ),
        )

        def _best_et_model_factory() -> ExtraTreesRegressor:
            return ExtraTreesRegressor(
                n_estimators=best_et_n_estimators,
                max_depth=best_et_max_depth,
                min_samples_leaf=best_et_min_samples_leaf,
                min_samples_split=best_et_min_samples_split,
                max_features=best_et_max_features,
                criterion=best_et_criterion,
                ccp_alpha=best_et_ccp_alpha,
                min_impurity_decrease=best_et_min_impurity,
                bootstrap=True,
                oob_score=False,
                random_state=42,
                n_jobs=2,
                verbose=0,
            )

        et_oof_preds_raw = np.zeros(len(y_raw_net_return), dtype=np.float32)
        et_confidence_all = np.ones(len(y_raw_net_return), dtype=np.float32)
        et_fold_importances: List[np.ndarray] = []
        observed_mask = np.zeros(len(y_raw_net_return), dtype=bool)
        fold_et_models = []

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            tprint(
                f"  ET refit fold {fold_idx+1}/{len(splits)}: train={len(tr_idx)} test={len(te_idx)}"
            )
            X_tr, y_tr = X_et[tr_idx], y_resid_target[tr_idx]
            X_te = X_et[te_idx]
            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
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

            model = _best_et_model_factory()
            model.fit(X_tr_clean, y_tr)
            preds = np.asarray(model.predict(X_te_clean), dtype=np.float32)
            et_oof_preds_raw[te_idx] = preds
            observed_mask[te_idx] = True

            conf = _compute_et_confidence(model, X_te_clean)
            et_confidence_all[te_idx] = conf

            if hasattr(model, "feature_importances_"):
                et_fold_importances.append(
                    np.asarray(model.feature_importances_, dtype=np.float32)
                )

            fold_et_models.append(
                {
                    "fold_idx": fold_idx,
                    "model": model,
                    "medians": medians,
                    "scaler": scaler,
                    "center_1d": center_1d,
                    "scale_1d": scale_1d,
                    "tr_idx": tr_idx,
                    "te_idx": te_idx,
                }
            )

        if ridge_oof_available:
            _combined_atr = ridge_oof_preds + 0.5 * et_oof_preds_raw * et_confidence_all
            if _use_atr:
                raw_conf_all = np.clip(
                    _combined_atr * _atr_safe, 0.0, 3.0 * _atr_safe
                ).astype(np.float32)
            else:
                raw_conf_all = np.maximum(
                    0.0,
                    _combined_atr * ridge_resid_std,
                ).astype(np.float32)
        else:
            raw_conf_all = (
                et_oof_preds_raw * _atr_safe if _use_atr else et_oof_preds_raw
            )

        total_confidence = raw_conf_all.copy()
        calibration_isotonic = None
        valid = (
            observed_mask
            & np.isfinite(total_confidence)
            & np.isfinite(y_raw_net_return)
        )
        if valid.sum() >= 30:
            from sklearn.isotonic import IsotonicRegression as IsoReg

            calibration_isotonic = IsoReg(out_of_bounds="clip")
            calibration_isotonic.fit(total_confidence[valid], y_raw_net_return[valid])
            total_confidence_calibrated = calibration_isotonic.transform(
                total_confidence.astype(np.float64)
            ).astype(np.float32)
            tprint(
                f"  Isotonic calibration fitted on {valid.sum()} samples: "
                f"raw_conf range=[{float(np.min(raw_conf_all[valid])):.4f}, "
                f"{float(np.max(raw_conf_all[valid])):.4f}] -> "
                f"calibrated range=[{float(np.min(total_confidence_calibrated)):.4f}, "
                f"{float(np.max(total_confidence_calibrated)):.4f}]"
            )
        else:
            total_confidence_calibrated = total_confidence

        best_et_preds = total_confidence_calibrated

        if et_fold_importances:
            importance_matrix = np.asarray(et_fold_importances, dtype=np.float32)
            mean_importance = np.mean(importance_matrix, axis=0)
            std_importance = np.std(importance_matrix, axis=0)
            best_et_importance = pd.DataFrame(
                {
                    "head_name": et_selected_keys,
                    "mean_importance": mean_importance,
                    "std_importance": std_importance,
                    "importance_rank": pd.Series(mean_importance)
                    .rank(ascending=False)
                    .values,
                }
            ).sort_values("mean_importance", ascending=False)
        else:
            best_et_importance = pd.DataFrame()

        best_et_metrics = evaluate_signal(
            "ET_Residual_HPO",
            best_et_preds,
            y_raw_net_return,
            y_downside,
            directionality="return-like",
        )
        best_et_utility = best_et_metrics.get("utility_score", -np.inf)
        (
            best_et_profit_proxy,
            best_et_opt_rets,
            best_et_opt_ts,
        ) = evaluate_selection_profit_proxy(
            best_et_preds,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        best_et_objective = (
            float(best_et_profit_proxy["wallet_pnl"].max())
            if not best_et_profit_proxy.empty
            else float("-inf")
        )

        tprint(
            "ET Residual HPO winner: "
            f"{best_et_params} (utility={best_et_utility:.4f}, "
            f"wallet_pnl={best_et_objective:.4f}, "
            f"n_estimators={best_et_n_estimators}, n_jobs=2, "
            f"ridge_resid_std={ridge_resid_std:.6f})"
        )
        if not best_et_importance.empty:
            tprint("=== ExtraTrees Feature Importance (top 20) ===")
            _imp_et = best_et_importance.copy()
            if "importance" not in _imp_et.columns:
                if "mean_importance" in _imp_et.columns:
                    _imp_et["importance"] = _imp_et["mean_importance"]
                else:
                    _imp_et["importance"] = 0.0
            feature_col = "feature" if "feature" in _imp_et.columns else "head_name"
            _imp_et = _imp_et.sort_values("importance", ascending=False).head(20)
            for _i, _row in _imp_et.iterrows():
                tprint(
                    f"  {_i+1:>3}. {_row[feature_col]:<40} importance={float(_row['importance']):.6f}"
                )
            tprint("=== End ExtraTrees Importance ===")
        et_oof_preds = best_et_preds
        et_sizer_eval = best_et_metrics
        et_importance_df = best_et_importance
        et_profit_proxy_df = best_et_profit_proxy
        results["et_sizer_scores_"] = et_oof_preds
        results["et_importance_table_"] = et_importance_df
        results["et_profit_proxy_table_"] = et_profit_proxy_df
        results["et_opt_rets_"] = best_et_opt_rets
        results["et_opt_ts_"] = best_et_opt_ts
        results["et_raw_residual_preds_"] = et_oof_preds_raw
        results["et_confidence_"] = et_confidence_all
        results["raw_confidence_"] = raw_conf_all
        results["total_confidence_"] = best_et_preds
        results["calibration_isotonic_"] = calibration_isotonic
        results["ridge_resid_mean_"] = ridge_resid_mean
        results["ridge_resid_std_"] = ridge_resid_std
        results["fold_et_models_"] = fold_et_models
        results["et_feature_keys_"] = et_selected_keys
        results["et_best_params_"] = best_et_params
        combined_et_objective = best_et_objective
        if not best_combo or combined_et_objective > best_combo_objective:
            best_simple_score = best_et_preds
            best_simple_score_name = "RidgePlusET_TotalConfidence"

    # --- LGBM Residual Sizer ---
    if (
        use_lgbm_head_sizer
        and used_keys
        and _use_any_booster
        and LGBMRegressor is not None
    ):
        best_lgbm_utility = -np.inf
        best_lgbm_preds = None
        best_lgbm_importance = pd.DataFrame()
        best_lgbm_metrics = {}
        best_lgbm_profit_proxy = pd.DataFrame()
        best_lgbm_opt_rets = np.array([])
        best_lgbm_opt_ts = np.array([])
        best_lgbm_params = {}

        lgbm_selected_keys = _mdi_select_features(
            X_heads_clean_m,
            y_resid_target,
            used_keys,
            model_factory=lambda: LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                num_leaves=31,
                objective="mae",
                random_state=42,
                n_jobs=2,
                verbose=-1,
            ),
        )
        lgbm_selected_idx = [
            used_keys.index(k) for k in lgbm_selected_keys if k in used_keys
        ]
        X_lgbm = X_heads[:, lgbm_selected_idx]
        lgbm_tail_weight_full = _normalized_tail_fit_weight(
            ridge_oof_preds
            if ridge_oof_available and ridge_oof_preds is not None
            else y_model_target
        )
        tprint(
            f"  LGBM MDI feature selection: {len(lgbm_selected_keys)}/{len(used_keys)} features retained"
        )

        lgbm_sub_scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i, (tr_idx, te_idx) in enumerate(splits):
            sub_tr = _hpo_tr_idx_subs[i]
            X_sub = X_lgbm[sub_tr]
            X_sub_clean, med, sc, c1d, s1d = clean_and_standardize(X_sub)
            X_te_clean, _, _, _, _ = clean_and_standardize(
                X_lgbm[te_idx],
                fit_medians=med,
                scaler=sc,
                center_1d=c1d,
                scale_1d=s1d,
            )
            lgbm_sub_scaled_folds.append((sub_tr, te_idx, X_sub_clean, X_te_clean))

        lgbm_trials = 150
        lgbm_sampler = TPESampler(seed=42, multivariate=True, group=True)
        lgbm_pruner = _make_median_pruner()
        lgbm_study = optuna.create_study(
            direction="maximize", sampler=lgbm_sampler, pruner=lgbm_pruner
        )

        def _as_2d_leaf_predictions(values: np.ndarray) -> np.ndarray:
            arr = np.asarray(values, dtype=np.int32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr

        def _compute_lgbm_confidence(
            model: LGBMRegressor,
            X: np.ndarray,
            X_raw: np.ndarray,
            y_train_raw: np.ndarray,
        ) -> np.ndarray:
            n_samples = X.shape[0]
            leaf_preds = _as_2d_leaf_predictions(
                model.predict(X_raw, pred_leaf=True)
            )
            n_trees = leaf_preds.shape[1]

            leaf_counts_arr = np.zeros((n_samples, n_trees), dtype=np.float32)
            leaf_var_arr = np.zeros((n_samples, n_trees), dtype=np.float32)
            tree_preds_arr = np.zeros((n_samples, n_trees), dtype=np.float32)
            for t_idx in range(n_trees):
                leaf_ids = leaf_preds[:, t_idx]
                train_leaf_ids = leaf_preds_train_cache[:, t_idx]
                leaves_offset = leaf_ids - leaf_ids.min()
                n_bins = int(leaves_offset.max()) + 1
                counts = np.bincount(leaves_offset, minlength=n_bins).astype(np.float32)
                sample_counts = counts[leaves_offset]
                leaf_counts_arr[:, t_idx] = sample_counts

                train_offset = train_leaf_ids - train_leaf_ids.min()
                train_n_bins = int(train_offset.max()) + 1
                train_counts = np.bincount(train_offset, minlength=train_n_bins).astype(
                    np.float64
                )
                train_sum = np.bincount(
                    train_offset,
                    weights=y_train_raw.astype(np.float64),
                    minlength=train_n_bins,
                )
                train_sq_sum = np.bincount(
                    train_offset,
                    weights=(y_train_raw.astype(np.float64)) ** 2,
                    minlength=train_n_bins,
                )
                train_mean = np.where(train_counts > 0, train_sum / train_counts, 0.0)
                train_var = np.where(
                    train_counts > 1,
                    train_sq_sum / train_counts - train_mean**2,
                    0.0,
                ).astype(np.float32)
                train_pred_sum = np.bincount(
                    train_offset,
                    weights=tree_preds_train_cache[:, t_idx].astype(np.float64),
                    minlength=train_n_bins,
                )
                train_pred_mean = np.where(
                    train_counts > 0, train_pred_sum / train_counts, 0.0
                ).astype(np.float32)

                remap = np.zeros(n_bins, dtype=np.int32)
                leaf_min = int(leaf_ids.min())
                train_min = int(train_leaf_ids.min())
                valid_bins = np.arange(n_bins) + leaf_min
                in_range = (valid_bins >= train_min) & (
                    valid_bins < train_min + train_n_bins
                )
                remap[in_range] = valid_bins[in_range] - train_min
                remap[~in_range] = 0

                leaf_var_arr[:, t_idx] = train_var[remap[leaves_offset]]
                tree_preds_arr[:, t_idx] = train_pred_mean[remap[leaves_offset]]

            n_leaf_avg = np.mean(leaf_counts_arr, axis=1)
            centroid_dist = np.mean(
                np.abs(
                    model.predict(X_raw).astype(np.float32)[:, None] - tree_preds_arr
                ),
                axis=1,
            )
            local_variance = centroid_dist / (np.sqrt(np.log(1.0 + n_leaf_avg) + 1e-12))
            leaf_variance = np.mean(leaf_var_arr, axis=1)

            lv_scaled = _quantile_scale_to_band(local_variance)
            lfv_scaled = _quantile_scale_to_band(leaf_variance)

            log_conf = (
                1.0 * np.log(lv_scaled + 1e-12) + 2.0 * np.log(lfv_scaled + 1e-12)
            ) / 3.0
            raw_wgm = np.exp(log_conf).astype(np.float32)
            median_wgm = float(np.median(raw_wgm))
            if abs(median_wgm) > 1e-12:
                lgbm_confidence = np.clip(raw_wgm / median_wgm, 0.7, 1.3).astype(
                    np.float32
                )
            else:
                lgbm_confidence = np.ones_like(raw_wgm)
            return lgbm_confidence

        def _lgbm_objective(trial: optuna.trial.Trial) -> float:
            n_estimators = 3000
            num_leaves = int(trial.suggest_int("num_leaves", 8, 128))
            max_depth = int(trial.suggest_categorical("max_depth", [2, 3, 4]))
            min_data_frac = float(trial.suggest_float("min_data_frac", 0.005, 0.03))
            min_data_in_leaf = max(1, int(np.ceil(min_data_frac * X_lgbm.shape[0])))
            min_sum_hessian = float(
                trial.suggest_float("min_sum_hessian", 1e-3, 10.0, log=True)
            )
            feature_fraction = float(trial.suggest_float("feature_fraction", 0.4, 1.0))
            bagging_fraction = float(trial.suggest_float("bagging_fraction", 0.7, 1.0))
            bagging_freq = int(trial.suggest_int("bagging_freq", 0, 10))
            _lgbm_hpo_rng = np.random.RandomState(42 + trial.number)
            lambda_l1 = float(10.0 ** _lgbm_hpo_rng.uniform(-2.0, np.log10(5.0)))
            lambda_l2 = float(10.0 ** _lgbm_hpo_rng.uniform(-2.0, np.log10(10.0)))
            min_gain_to_split = float(
                trial.suggest_float("min_gain_to_split", 1e-4, 1e-1, log=True)
            )

            fold_pnl_5: List[float] = []
            fold_sortino_5: List[float] = []
            fold_ic_30: List[float] = []

            _lgbm_es_rounds = 50

            for fold_idx in range(len(splits)):
                (
                    sub_tr,
                    te_idx,
                    X_tr_clean,
                    X_te_clean,
                ) = lgbm_sub_scaled_folds[fold_idx]
                if len(sub_tr) == 0 or len(te_idx) == 0:
                    continue
                y_tr = y_resid_target[sub_tr]
                y_te = y_raw_net_return[te_idx]

                _es_split = max(1, len(sub_tr) // 5)
                _es_tr = sub_tr[:-_es_split]
                _es_val = sub_tr[-_es_split:]

                model = LGBMRegressor(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_data_in_leaf,
                    min_sum_hessian_in_leaf=min_sum_hessian,
                    colsample_bytree=feature_fraction,
                    subsample=bagging_fraction,
                    subsample_freq=bagging_freq,
                    reg_alpha=lambda_l1,
                    reg_lambda=lambda_l2,
                    min_gain_to_split=min_gain_to_split,
                    objective="mae",
                    random_state=42,
                    n_jobs=2,
                    verbose=-1,
                )
                model.fit(
                    X_tr_clean[:-_es_split],
                    y_tr[:-_es_split],
                    sample_weight=np.asarray(
                        lgbm_tail_weight_full[sub_tr[:-_es_split]], dtype=np.float32
                    ),
                    eval_set=[(X_tr_clean[-_es_split:], y_tr[-_es_split:])],
                    callbacks=[
                        early_stopping(_lgbm_es_rounds, verbose=False),
                        log_evaluation(period=0),
                    ],
                )
                _best_it = getattr(model, "best_iteration_", n_estimators - 1) + 1
                lgbm_fold_preds = np.asarray(
                    model.predict(X_te_clean, num_iteration=_best_it), dtype=np.float32
                )

                if ridge_oof_available:
                    ridge_fold_preds = ridge_oof_preds[te_idx]
                    _combined_hpo_l = ridge_fold_preds + 0.5 * lgbm_fold_preds
                    if _use_atr:
                        raw_conf = np.clip(
                            _combined_hpo_l * _atr_safe[te_idx],
                            0.0,
                            3.0 * _atr_safe[te_idx],
                        ).astype(np.float32)
                    else:
                        raw_conf = np.maximum(
                            0.0,
                            _combined_hpo_l * ridge_resid_std,
                        ).astype(np.float32)
                else:
                    raw_conf = (
                        lgbm_fold_preds * _atr_safe[te_idx]
                        if _use_atr
                        else lgbm_fold_preds
                    )

                n_days_fold = max(1.0, len(te_idx) / 96.0)
                lw = _lightweight_hpo_pnl_eval(
                    raw_conf,
                    y_te,
                    top_fracs=[0.05],
                    cost_pct=cost_pct,
                    n_days=n_days_fold,
                )
                if "wallet_pnl_0.050" in lw:
                    fold_pnl_5.append(lw["wallet_pnl_0.050"])
                if "sortino_0.050" in lw:
                    fold_sortino_5.append(lw["sortino_0.050"])

                _combined_raw = raw_conf
                k_30 = max(1, int(len(_combined_raw) * 0.30))
                _top_idx = np.argpartition(_combined_raw, -k_30)[-k_30:]
                try:
                    _ic_30, _ = spearmanr(
                        _combined_raw[_top_idx],
                        y_te[_top_idx],
                        nan_policy="omit",
                    )
                    fold_ic_30.append(float(_ic_30) if pd.notna(_ic_30) else 0.0)
                except Exception:
                    fold_ic_30.append(0.0)

                if trial is not None and fold_idx >= len(splits) // 2:
                    trial.report(
                        float(np.mean(fold_pnl_5)) if fold_pnl_5 else -1e9,
                        step=fold_idx,
                    )
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if not fold_pnl_5:
                return -1e9

            mean_pnl_5 = float(np.mean(fold_pnl_5))
            std_pnl_5 = float(np.std(fold_pnl_5))
            sortino_5 = float(np.mean(fold_sortino_5)) if fold_sortino_5 else 0.0
            mean_ic_30 = float(np.mean(fold_ic_30)) if fold_ic_30 else 0.0

            composite = (
                0.50 * mean_pnl_5
                - 0.30 * std_pnl_5
                + 0.17 * sortino_5
                + 0.33 * mean_ic_30
            )

            trial.set_user_attr("feature_count", int(X_lgbm.shape[1]))
            trial.set_user_attr("best_n_estimators", _best_it)
            trial.set_user_attr("mean_pnl_5", mean_pnl_5)
            trial.set_user_attr("std_pnl_5", std_pnl_5)
            trial.set_user_attr("sortino_5", sortino_5)
            trial.set_user_attr("mean_ic_30", mean_ic_30)
            trial.set_user_attr("composite_score", composite)

            return float(composite)

        lgbm_study.optimize(
            _lgbm_objective,
            n_trials=lgbm_trials,
            gc_after_trial=True,
            callbacks=[
                _make_patience_callback(
                    patience=30, label="LGBM Residual HPO", min_trials=50
                )
            ],
        )
        _lgbm_hpo_cache_path = Path(".cache") / "lgbm_residual_hpo_round1.json"
        _lgbm_hpo_cache_path.parent.mkdir(parents=True, exist_ok=True)
        if lgbm_study.best_trial is not None:
            best_lgbm_params = dict(lgbm_study.best_trial.params)
            try:
                _lgbm_hpo_cache_path.write_text(
                    json.dumps(
                        {
                            "feature_count": int(X_lgbm.shape[1]),
                            "params": best_lgbm_params,
                            "best_n_estimators": int(
                                lgbm_study.best_trial.user_attrs.get(
                                    "best_n_estimators", 3000
                                )
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            except Exception as _cache_exc:
                tprint(f"  LGBM round-1 HPO cache write skipped: {_cache_exc}")

        # Round-2 HPO: same search/pruning rules with round-1 JSON params as warm-start.
        lgbm_study_round2 = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=43, multivariate=True, group=True),
            pruner=_make_median_pruner(),
        )
        try:
            if _lgbm_hpo_cache_path.exists():
                _payload = json.loads(_lgbm_hpo_cache_path.read_text())
                _cached_params = _payload.get("params", {})
                if isinstance(_cached_params, dict) and _cached_params:
                    lgbm_study_round2.enqueue_trial(_cached_params)
        except Exception as _cache_exc:
            tprint(f"  LGBM round-2 HPO cache read skipped: {_cache_exc}")

        lgbm_study_round2.optimize(
            _lgbm_objective,
            n_trials=max(40, lgbm_trials // 2),
            gc_after_trial=True,
            callbacks=[
                _make_patience_callback(
                    patience=30, label="LGBM Residual HPO Round-2", min_trials=30
                )
            ],
        )
        if lgbm_study_round2.best_trial is not None:
            best_lgbm_params = dict(lgbm_study_round2.best_trial.params)
            lgbm_study = lgbm_study_round2

        _lgbm_best_n_estimators = 3000
        if lgbm_study.best_trial is not None:
            _lgbm_best_n_estimators = int(
                lgbm_study.best_trial.user_attrs.get("best_n_estimators", 3000)
            )

        # LGBM residual feature diagnostics (2 folds x 4 variants).
        lgbm_feature_diag_df = pd.DataFrame()
        lgbm_feature_diag_meta: Dict[str, Any] = {}
        try:
            _diag_n = min(len(y_raw_net_return), 10000)
            _diag_idx = (
                np.random.RandomState(42).choice(
                    len(y_raw_net_return), _diag_n, replace=False
                )
                if _diag_n < len(y_raw_net_return)
                else np.arange(len(y_raw_net_return))
            )
            _diag_idx = np.sort(_diag_idx)
            _diag_X = X_lgbm[_diag_idx]
            _diag_y = y_raw_net_return[_diag_idx]
            _diag_ridge = (
                np.asarray(ridge_oof_preds[_diag_idx], dtype=np.float32)
                if ridge_oof_available and ridge_oof_preds is not None
                else np.zeros(_diag_n, dtype=np.float32)
            )
            _diag_ts = pd.to_datetime(
                np.asarray(timestamps)[_diag_idx], utc=True, errors="coerce"
            )
            _diag_week = _diag_ts.to_period("W").astype(str)
            _diag_month = _diag_ts.to_period("M").astype(str)
            _split = _diag_n // 2
            _diag_folds = [np.arange(0, _split), np.arange(_split, _diag_n)]
            _base_l1 = float(best_lgbm_params.get("lambda_l1", 0.1))
            _base_ff = float(best_lgbm_params.get("feature_fraction", 0.8))
            _variants = [
                {"name": "base", "l1": _base_l1, "ff": _base_ff},
                {"name": "l1_up", "l1": _base_l1 * 1.5, "ff": _base_ff},
                {"name": "ff_down", "l1": _base_l1, "ff": _base_ff * 0.66},
                {"name": "l1_up_ff_down", "l1": _base_l1 * 1.5, "ff": _base_ff * 0.66},
            ]
            _diag_rows: List[Dict[str, Any]] = []
            _presence = np.zeros(len(lgbm_selected_keys), dtype=np.int32)
            _gain_sum = np.zeros(len(lgbm_selected_keys), dtype=np.float64)
            _split_sum = np.zeros(len(lgbm_selected_keys), dtype=np.float64)
            for _fid, _val_idx in enumerate(_diag_folds):
                _tr_idx = np.setdiff1d(np.arange(_diag_n), _val_idx)
                x_tr, y_tr = _diag_X[_tr_idx], y_resid_target[_diag_idx][_tr_idx]
                x_val = _diag_X[_val_idx]
                x_tr_clean, med, sc, c1d, s1d = clean_and_standardize(x_tr)
                x_val_clean, _, _, _, _ = clean_and_standardize(
                    x_val, fit_medians=med, scaler=sc, center_1d=c1d, scale_1d=s1d
                )
                for _v in _variants:
                    _model = LGBMRegressor(
                        n_estimators=_lgbm_best_n_estimators,
                        num_leaves=int(best_lgbm_params.get("num_leaves", 31)),
                        max_depth=int(best_lgbm_params.get("max_depth", 3)),
                        min_child_samples=max(
                            1,
                            int(
                                np.ceil(
                                    float(best_lgbm_params.get("min_data_frac", 0.02))
                                    * len(_tr_idx)
                                )
                            ),
                        ),
                        min_sum_hessian_in_leaf=float(
                            best_lgbm_params.get("min_sum_hessian", 1e-3)
                        ),
                        colsample_bytree=float(np.clip(_v["ff"], 0.05, 1.0)),
                        subsample=float(best_lgbm_params.get("bagging_fraction", 0.8)),
                        subsample_freq=int(best_lgbm_params.get("bagging_freq", 1)),
                        reg_alpha=float(max(_v["l1"], 0.0)),
                        reg_lambda=float(best_lgbm_params.get("lambda_l2", 1.0)),
                        min_gain_to_split=float(
                            best_lgbm_params.get("min_gain_to_split", 1e-4)
                        ),
                        objective="mae",
                        random_state=42 + _fid,
                        n_jobs=2,
                        verbose=-1,
                    )
                    _diag_w = np.asarray(
                        lgbm_tail_weight_full[_diag_idx][_tr_idx], dtype=np.float32
                    )
                    _model.fit(x_tr_clean, y_tr, sample_weight=_diag_w)
                    _pred = np.asarray(_model.predict(x_val_clean), dtype=np.float32)
                    _combined = _diag_ridge[_val_idx] + 0.8 * _pred
                    _score = _score_tail_20(
                        y=_diag_y[_val_idx],
                        pred=_combined,
                        week_id=np.asarray(_diag_week)[_val_idx],
                        month_id=np.asarray(_diag_month)[_val_idx],
                    )
                    _diag_rows.append(
                        {
                            "fold": _fid,
                            "variant": _v["name"],
                            "score_tail_20": float(_score),
                        }
                    )
                    _g = np.asarray(_model.feature_importances_, dtype=np.float64)
                    _s = np.asarray(
                        _model.booster_.feature_importance(importance_type="split"),
                        dtype=np.float64,
                    )
                    _presence += (_s > 0).astype(np.int32)
                    _gain_sum += _g
                    _split_sum += _s
            lgbm_feature_diag_df = pd.DataFrame(_diag_rows)
            lgbm_feature_diag_meta = {
                "feature_presence": pd.DataFrame(
                    {
                        "feature": lgbm_selected_keys,
                        "model_presence_count": _presence,
                        "model_presence_rate": _presence
                        / max(1, len(_diag_folds) * len(_variants)),
                        "mean_gain": _gain_sum
                        / max(1, len(_diag_folds) * len(_variants)),
                        "mean_split_count": _split_sum
                        / max(1, len(_diag_folds) * len(_variants)),
                    }
                ).sort_values("model_presence_count", ascending=False)
            }
        except Exception as _diag_exc:
            tprint(f"  LGBM feature diagnostics skipped: {_diag_exc}")

        def _best_lgbm_model_factory() -> LGBMRegressor:
            _n_est = _lgbm_best_n_estimators
            _nl = int(best_lgbm_params.get("num_leaves", 31))
            _md = int(best_lgbm_params.get("max_depth", 5))
            _mdf = float(best_lgbm_params.get("min_data_frac", 0.02))
            _mdil = max(1, int(np.ceil(_mdf * X_lgbm.shape[0])))
            return LGBMRegressor(
                n_estimators=_n_est,
                num_leaves=_nl,
                max_depth=_md,
                min_child_samples=_mdil,
                min_sum_hessian_in_leaf=float(
                    best_lgbm_params.get("min_sum_hessian", 1e-3)
                ),
                colsample_bytree=float(best_lgbm_params.get("feature_fraction", 0.9)),
                subsample=float(best_lgbm_params.get("bagging_fraction", 0.8)),
                subsample_freq=int(best_lgbm_params.get("bagging_freq", 1)),
                reg_alpha=float(best_lgbm_params.get("lambda_l1", 1.0)),
                reg_lambda=float(best_lgbm_params.get("lambda_l2", 1.0)),
                min_gain_to_split=float(
                    best_lgbm_params.get("min_gain_to_split", 1e-4)
                ),
                objective="mae",
                random_state=42,
                n_jobs=2,
                verbose=-1,
            )

        lgbm_oof_preds_raw = np.zeros(len(y_raw_net_return), dtype=np.float32)
        lgbm_confidence_all = np.ones(len(y_raw_net_return), dtype=np.float32)
        lgbm_fold_importances: List[np.ndarray] = []
        observed_mask_lgbm = np.zeros(len(y_raw_net_return), dtype=bool)
        fold_lgbm_models = []

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            if len(tr_idx) == 0 or len(te_idx) == 0:
                continue
            tprint(
                f"  LGBM refit fold {fold_idx+1}/{len(splits)}: train={len(tr_idx)} test={len(te_idx)}"
            )
            X_tr, y_tr = X_lgbm[tr_idx], y_resid_target[tr_idx]
            X_te = X_lgbm[te_idx]
            if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
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

            model = _best_lgbm_model_factory()
            _es_split = max(1, len(tr_idx) // 5)
            model.fit(
                X_tr_clean[:-_es_split],
                y_tr[:-_es_split],
                sample_weight=np.asarray(
                    lgbm_tail_weight_full[tr_idx[:-_es_split]], dtype=np.float32
                ),
                eval_set=[(X_tr_clean[-_es_split:], y_tr[-_es_split:])],
                callbacks=[
                    early_stopping(50, verbose=False),
                    log_evaluation(period=0),
                ],
            )
            _refit_best_it = (
                getattr(model, "best_iteration_", _lgbm_best_n_estimators - 1) + 1
            )
            preds = np.asarray(
                model.predict(X_te_clean, num_iteration=_refit_best_it),
                dtype=np.float32,
            )
            lgbm_oof_preds_raw[te_idx] = preds
            observed_mask_lgbm[te_idx] = True

            leaf_preds_train_cache = _as_2d_leaf_predictions(
                model.predict(X_tr_clean, pred_leaf=True)
            )
            n_trees_actual = leaf_preds_train_cache.shape[1]
            _max_trees_for_cache = 100
            _tree_step = max(1, n_trees_actual // _max_trees_for_cache)
            _tree_indices = list(range(0, n_trees_actual, _tree_step))
            tree_preds_train_cache = np.zeros(
                (X_tr_clean.shape[0], n_trees_actual),
                dtype=np.float32,
            )
            booster = model.booster_
            prev = np.zeros(X_tr_clean.shape[0], dtype=np.float64)
            for t_idx in _tree_indices:
                cur = booster.predict(X_tr_clean, num_iteration=t_idx + 1).astype(
                    np.float64
                )
                tree_preds_train_cache[:, t_idx] = (cur - prev).astype(np.float32)
                prev = cur.copy()

            conf = _compute_lgbm_confidence(model, X_te_clean, X_te, y_tr)
            lgbm_confidence_all[te_idx] = conf

            if hasattr(model, "feature_importances_"):
                lgbm_fold_importances.append(
                    np.asarray(model.feature_importances_, dtype=np.float32)
                )

            fold_lgbm_models.append(
                {
                    "fold_idx": fold_idx,
                    "model": model,
                    "medians": medians,
                    "scaler": scaler,
                    "center_1d": center_1d,
                    "scale_1d": scale_1d,
                    "tr_idx": tr_idx,
                    "te_idx": te_idx,
                }
            )

        if ridge_oof_available:
            _combined_lgbm_atr = (
                ridge_oof_preds + 0.5 * lgbm_oof_preds_raw * lgbm_confidence_all
            )
            if _use_atr:
                raw_conf_lgbm_all = np.clip(
                    _combined_lgbm_atr * _atr_safe, 0.0, 3.0 * _atr_safe
                ).astype(np.float32)
            else:
                raw_conf_lgbm_all = np.maximum(
                    0.0,
                    _combined_lgbm_atr * ridge_resid_std,
                ).astype(np.float32)
        else:
            raw_conf_lgbm_all = (
                lgbm_oof_preds_raw * _atr_safe if _use_atr else lgbm_oof_preds_raw
            )

        total_confidence_lgbm = raw_conf_lgbm_all.copy()
        calibration_isotonic_lgbm = None
        valid_lgbm = (
            observed_mask_lgbm
            & np.isfinite(total_confidence_lgbm)
            & np.isfinite(y_raw_net_return)
        )
        if valid_lgbm.sum() >= 30:
            from sklearn.isotonic import IsotonicRegression as IsoReg

            calibration_isotonic_lgbm = IsoReg(out_of_bounds="clip")
            calibration_isotonic_lgbm.fit(
                total_confidence_lgbm[valid_lgbm],
                y_raw_net_return[valid_lgbm],
            )
            total_confidence_lgbm_cal = calibration_isotonic_lgbm.transform(
                total_confidence_lgbm.astype(np.float64)
            ).astype(np.float32)
            tprint(f"  LGBM isotonic calibration fitted on {valid_lgbm.sum()} samples")
        else:
            total_confidence_lgbm_cal = total_confidence_lgbm

        best_lgbm_preds = total_confidence_lgbm_cal

        if lgbm_fold_importances:
            importance_matrix_l = np.asarray(lgbm_fold_importances, dtype=np.float32)
            mean_importance_l = np.mean(importance_matrix_l, axis=0)
            std_importance_l = np.std(importance_matrix_l, axis=0)
            best_lgbm_importance = pd.DataFrame(
                {
                    "head_name": lgbm_selected_keys,
                    "mean_importance": mean_importance_l,
                    "std_importance": std_importance_l,
                    "importance_rank": pd.Series(mean_importance_l)
                    .rank(ascending=False)
                    .values,
                }
            ).sort_values("mean_importance", ascending=False)
        else:
            best_lgbm_importance = pd.DataFrame()

        best_lgbm_metrics = evaluate_signal(
            "LGBM_Residual_HPO",
            best_lgbm_preds,
            y_raw_net_return,
            y_downside,
            directionality="return-like",
        )
        best_lgbm_utility = best_lgbm_metrics.get("utility_score", -np.inf)
        (
            best_lgbm_profit_proxy,
            best_lgbm_opt_rets,
            best_lgbm_opt_ts,
        ) = evaluate_selection_profit_proxy(
            best_lgbm_preds,
            y_raw_net_return,
            timestamps=timestamps,
            symbols=_sym_vals,
            top_fracs=list(top_fracs),
            cost_pct=cost_pct,
            n_days=n_days,
        )
        best_lgbm_objective = (
            float(best_lgbm_profit_proxy["wallet_pnl"].max())
            if not best_lgbm_profit_proxy.empty
            else float("-inf")
        )

        tprint(
            f"LGBM Residual HPO winner: "
            f"{best_lgbm_params} (utility={best_lgbm_utility:.4f}, "
            f"wallet_pnl={best_lgbm_objective:.4f}, "
            f"ridge_resid_std={ridge_resid_std:.6f})"
        )
        if not best_lgbm_importance.empty:
            tprint("=== LGBM Feature Importance (top 20) ===")
            _imp_l = best_lgbm_importance.copy()
            if "importance" not in _imp_l.columns:
                _imp_l["importance"] = _imp_l.get("mean_importance", 0.0)
            feature_col = "feature" if "feature" in _imp_l.columns else "head_name"
            _imp_l = _imp_l.sort_values("importance", ascending=False).head(20)
            for _i, _row in _imp_l.iterrows():
                tprint(
                    f"  {_i+1:>3}. {_row[feature_col]:<40} importance={float(_row['importance']):.6f}"
                )
            tprint("=== End LGBM Importance ===")
        lgbm_oof_preds = best_lgbm_preds
        lgbm_sizer_eval = best_lgbm_metrics
        lgbm_importance_df = best_lgbm_importance
        lgbm_profit_proxy_df = best_lgbm_profit_proxy
        results["lgbm_sizer_scores_"] = lgbm_oof_preds
        results["lgbm_importance_table_"] = lgbm_importance_df
        results["lgbm_profit_proxy_table_"] = lgbm_profit_proxy_df
        results["lgbm_opt_rets_"] = best_lgbm_opt_rets
        results["lgbm_opt_ts_"] = best_lgbm_opt_ts
        results["lgbm_raw_residual_preds_"] = lgbm_oof_preds_raw
        results["lgbm_confidence_"] = lgbm_confidence_all
        results["raw_confidence_lgbm_"] = raw_conf_lgbm_all
        results["total_confidence_lgbm_"] = best_lgbm_preds
        results["calibration_isotonic_lgbm_"] = calibration_isotonic_lgbm
        results["fold_lgbm_models_"] = fold_lgbm_models
        results["lgbm_feature_keys_"] = lgbm_selected_keys
        results["lgbm_best_params_"] = best_lgbm_params
        results["lgbm_feature_diag_table_"] = lgbm_feature_diag_df
        results["lgbm_feature_diag_meta_"] = lgbm_feature_diag_meta
        results["lgbm_uncertainty_aware_oof_"] = best_lgbm_preds
        if (
            isinstance(best_lgbm_profit_proxy, pd.DataFrame)
            and not best_lgbm_profit_proxy.empty
            and "is_optimal" in best_lgbm_profit_proxy.columns
        ):
            _opt_lgbm = best_lgbm_profit_proxy[best_lgbm_profit_proxy["is_optimal"]]
            if not _opt_lgbm.empty:
                results["lgbm_optimal_threshold_"] = {
                    "selection_frac": float(
                        _opt_lgbm.iloc[0].get("selection_frac", np.nan)
                    ),
                    "threshold_pct": str(_opt_lgbm.iloc[0].get("threshold_pct", "")),
                    "wallet_pnl": float(_opt_lgbm.iloc[0].get("wallet_pnl", np.nan)),
                    "net_pnl": float(_opt_lgbm.iloc[0].get("net_pnl", np.nan)),
                }
        combined_lgbm_objective = best_lgbm_objective
        if not best_combo or combined_lgbm_objective > best_combo_objective:
            if (
                best_simple_score is None
                or combined_lgbm_objective > combined_et_objective
            ):
                best_simple_score = best_lgbm_preds
                best_simple_score_name = "RidgePlusLGBM_TotalConfidence"
    elif use_lgbm_head_sizer and LGBMRegressor is None:
        tprint("  LGBM skipped: lightgbm not installed")

    # --- LGBM Classifier (3-class Ridge residual direction) ---
    if (
        use_lgbm_head_sizer
        and LGBMRegressor is not None
        and ridge_oof_available
        and _use_any_booster
        and len(used_keys) > 0
    ):
        from lightgbm import LGBMClassifier as _LGBMClassifier

        tprint("  LGBM Classifier: building 3-class residual labels...")
        _resid_atr = ridge_resid / (_atr_safe + 1e-8)
        _p35 = float(np.percentile(_resid_atr, 35))
        _p65 = float(np.percentile(_resid_atr, 65))
        _clf_y = np.ones(len(_resid_atr), dtype=np.int32)
        _clf_y[_resid_atr < _p35] = 0
        _clf_y[_resid_atr >= _p65] = 2
        lgbm_clf_tail_weight_full = _normalized_tail_fit_weight(
            ridge_oof_preds if ridge_oof_preds is not None else y_model_target
        )
        tprint(
            f"  LGBM Classifier labels: under={np.sum(_clf_y == 0)} "
            f"ok={np.sum(_clf_y == 1)} over={np.sum(_clf_y == 2)}"
        )

        lgbm_clf_selected_keys = _mdi_select_features(
            X_heads,
            _clf_y.astype(np.float32),
            used_keys,
            model_factory=lambda: ExtraTreesRegressor(
                n_estimators=200,
                max_depth=5,
                random_state=42,
                n_jobs=2,
            ),
        )
        _lgbm_clf_idx = [
            used_keys.index(k) for k in lgbm_clf_selected_keys if k in used_keys
        ]
        X_lgbm_clf = X_heads[:, _lgbm_clf_idx]
        tprint(
            f"  LGBM Clf MDI: {len(lgbm_clf_selected_keys)}/{len(used_keys)} features retained"
        )

        lgbm_clf_sub_scaled_folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i, (tr_idx, te_idx) in enumerate(splits):
            sub_tr = _hpo_tr_idx_subs[i]
            X_sub = X_lgbm_clf[sub_tr]
            X_sub_clean, med, sc, c1d, s1d = clean_and_standardize(X_sub)
            X_te_clean, _, _, _, _ = clean_and_standardize(
                X_lgbm_clf[te_idx],
                fit_medians=med,
                scaler=sc,
                center_1d=c1d,
                scale_1d=s1d,
            )
            lgbm_clf_sub_scaled_folds.append((sub_tr, te_idx, X_sub_clean, X_te_clean))

        lgbm_clf_trials = 150
        lgbm_clf_sampler = TPESampler(seed=42, multivariate=True, group=True)
        lgbm_clf_study = optuna.create_study(
            direction="maximize",
            sampler=lgbm_clf_sampler,
            pruner=_make_median_pruner(),
        )

        def _lgbm_clf_objective(trial: optuna.trial.Trial) -> float:
            n_estimators = 3000
            num_leaves = int(trial.suggest_int("num_leaves", 8, 64))
            max_depth = int(trial.suggest_categorical("max_depth", [2, 3, 4]))
            min_data_frac = float(trial.suggest_float("min_data_frac", 0.005, 0.03))
            min_data_in_leaf = max(1, int(np.ceil(min_data_frac * X_lgbm_clf.shape[0])))
            feature_fraction = float(trial.suggest_float("feature_fraction", 0.4, 1.0))
            bagging_fraction = float(trial.suggest_float("bagging_fraction", 0.7, 1.0))
            bagging_freq = int(trial.suggest_int("bagging_freq", 1, 10))
            _rng_lc = np.random.RandomState(42 + trial.number)
            lambda_l1 = float(10.0 ** _rng_lc.uniform(-2.0, np.log10(5.0)))
            lambda_l2 = float(10.0 ** _rng_lc.uniform(-2.0, np.log10(10.0)))

            fold_pnl_5: List[float] = []
            fold_sortino_5: List[float] = []
            fold_ic_30: List[float] = []

            for fold_idx in range(len(splits)):
                sub_tr, te_idx, X_tr_clean, X_te_clean = lgbm_clf_sub_scaled_folds[
                    fold_idx
                ]
                if len(sub_tr) == 0 or len(te_idx) == 0:
                    continue
                y_tr = _clf_y[sub_tr]
                y_te = y_raw_net_return[te_idx]

                _es_split = max(1, len(sub_tr) // 5)
                model = _LGBMClassifier(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_data_in_leaf,
                    colsample_bytree=feature_fraction,
                    subsample=bagging_fraction,
                    subsample_freq=bagging_freq,
                    reg_alpha=lambda_l1,
                    reg_lambda=lambda_l2,
                    objective="multiclass",
                    num_class=3,
                    random_state=42,
                    n_jobs=2,
                    verbose=-1,
                )
                model.fit(
                    X_tr_clean[:-_es_split],
                    y_tr[:-_es_split],
                    sample_weight=np.asarray(
                        lgbm_clf_tail_weight_full[sub_tr[:-_es_split]],
                        dtype=np.float32,
                    ),
                    eval_set=[(X_tr_clean[-_es_split:], y_tr[-_es_split:])],
                    callbacks=[
                        early_stopping(50, verbose=False),
                        log_evaluation(period=0),
                    ],
                )
                _best_it = getattr(model, "best_iteration_", n_estimators - 1) + 1
                _proba = np.asarray(
                    model.predict_proba(X_te_clean, num_iteration=_best_it),
                    dtype=np.float32,
                )
                _clf_score = _proba[:, 0] - _proba[:, 2]

                ridge_fold_preds = ridge_oof_preds[te_idx]
                _combined = ridge_fold_preds + 0.5 * _clf_score
                if _use_atr:
                    raw_conf = np.clip(
                        _combined * _atr_safe[te_idx],
                        0.0,
                        3.0 * _atr_safe[te_idx],
                    ).astype(np.float32)
                else:
                    raw_conf = np.maximum(0.0, _combined * ridge_resid_std).astype(
                        np.float32
                    )

                n_days_fold = max(1.0, len(te_idx) / 96.0)
                lw = _lightweight_hpo_pnl_eval(
                    raw_conf,
                    y_te,
                    top_fracs=[0.05],
                    cost_pct=cost_pct,
                    n_days=n_days_fold,
                )
                if "wallet_pnl_0.050" in lw:
                    fold_pnl_5.append(lw["wallet_pnl_0.050"])
                if "sortino_0.050" in lw:
                    fold_sortino_5.append(lw["sortino_0.050"])

                k_30 = max(1, int(len(raw_conf) * 0.30))
                _top_idx = np.argpartition(raw_conf, -k_30)[-k_30:]
                try:
                    _ic_30, _ = spearmanr(
                        raw_conf[_top_idx], y_te[_top_idx], nan_policy="omit"
                    )
                    fold_ic_30.append(float(_ic_30) if pd.notna(_ic_30) else 0.0)
                except Exception:
                    fold_ic_30.append(0.0)

                if trial is not None and fold_idx >= len(splits) // 2:
                    trial.report(
                        float(np.mean(fold_pnl_5)) if fold_pnl_5 else -1e9,
                        step=fold_idx,
                    )
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            if not fold_pnl_5:
                return -1e9
            mean_pnl_5 = float(np.mean(fold_pnl_5))
            std_pnl_5 = float(np.std(fold_pnl_5))
            sortino_5 = float(np.mean(fold_sortino_5)) if fold_sortino_5 else 0.0
            mean_ic_30 = float(np.mean(fold_ic_30)) if fold_ic_30 else 0.0
            composite = (
                0.50 * mean_pnl_5
                - 0.30 * std_pnl_5
                + 0.17 * sortino_5
                + 0.33 * mean_ic_30
            )
            trial.set_user_attr("mean_pnl_5", mean_pnl_5)
            trial.set_user_attr("mean_ic_30", mean_ic_30)
            trial.set_user_attr("composite_score", composite)
            return float(composite)

        lgbm_clf_study.optimize(
            _lgbm_clf_objective,
            n_trials=lgbm_clf_trials,
            catch=(Exception,),
            show_progress_bar=False,
        )

        if lgbm_clf_study.best_trial is not None:
            best_lgbm_clf_params = dict(lgbm_clf_study.best_params)
            tprint(
                f"  LGBM Clf HPO winner: composite={lgbm_clf_study.best_value:.4f} "
                f"params={best_lgbm_clf_params}"
            )

            _lgbm_clf_best_n = 3000
            _lgbm_clf_best_it_attr = lgbm_clf_study.best_trial.user_attrs.get(
                "best_n_estimators", 3000
            )
            if _lgbm_clf_best_it_attr < _lgbm_clf_best_n:
                _lgbm_clf_best_n = int(_lgbm_clf_best_it_attr)

            def _best_lgbm_clf_factory() -> _LGBMClassifier:
                _rng_f = np.random.RandomState(42)
                return _LGBMClassifier(
                    n_estimators=_lgbm_clf_best_n,
                    num_leaves=int(best_lgbm_clf_params.get("num_leaves", 31)),
                    max_depth=int(best_lgbm_clf_params.get("max_depth", 3)),
                    min_child_samples=max(
                        1,
                        int(
                            np.ceil(
                                float(best_lgbm_clf_params.get("min_data_frac", 0.01))
                                * X_lgbm_clf.shape[0]
                            )
                        ),
                    ),
                    colsample_bytree=float(
                        best_lgbm_clf_params.get("feature_fraction", 0.7)
                    ),
                    subsample=float(best_lgbm_clf_params.get("bagging_fraction", 0.8)),
                    subsample_freq=int(best_lgbm_clf_params.get("bagging_freq", 1)),
                    reg_alpha=float(10.0 ** _rng_f.uniform(-2.0, np.log10(5.0))),
                    reg_lambda=float(10.0 ** _rng_f.uniform(-2.0, np.log10(10.0))),
                    objective="multiclass",
                    num_class=3,
                    random_state=42,
                    n_jobs=2,
                    verbose=-1,
                )

            # Full OOF refit
            lgbm_clf_oof_preds_raw = np.full(
                len(y_raw_net_return), np.nan, dtype=np.float32
            )
            fold_lgbm_clf_models: List[Any] = []
            for fold_idx, (tr_idx, te_idx) in enumerate(splits):
                if len(tr_idx) == 0 or len(te_idx) == 0:
                    continue
                tprint(
                    f"  LGBM Clf refit fold {fold_idx+1}/{len(splits)}: "
                    f"train={len(tr_idx)} test={len(te_idx)}"
                )
                (
                    X_tr_clean,
                    medians,
                    scaler,
                    center_1d,
                    scale_1d,
                ) = clean_and_standardize(X_lgbm_clf[tr_idx])
                X_te_clean, _, _, _, _ = clean_and_standardize(
                    X_lgbm_clf[te_idx],
                    fit_medians=medians,
                    scaler=scaler,
                    center_1d=center_1d,
                    scale_1d=scale_1d,
                )
                y_tr = _clf_y[tr_idx]
                model = _best_lgbm_clf_factory()
                _es_split = max(1, len(tr_idx) // 5)
                model.fit(
                    X_tr_clean[:-_es_split],
                    y_tr[:-_es_split],
                    sample_weight=np.asarray(
                        lgbm_clf_tail_weight_full[tr_idx[:-_es_split]],
                        dtype=np.float32,
                    ),
                    eval_set=[(X_tr_clean[-_es_split:], y_tr[-_es_split:])],
                    callbacks=[
                        early_stopping(50, verbose=False),
                        log_evaluation(period=0),
                    ],
                )
                _refit_best_it = (
                    getattr(model, "best_iteration_", _lgbm_clf_best_n - 1) + 1
                )
                _proba = np.asarray(
                    model.predict_proba(X_te_clean, num_iteration=_refit_best_it),
                    dtype=np.float32,
                )
                _score = (_proba[:, 0] - _proba[:, 2]).astype(np.float32)
                lgbm_clf_oof_preds_raw[te_idx] = _score
                fold_lgbm_clf_models.append(
                    {
                        "fold_idx": fold_idx,
                        "model": model,
                        "medians": medians,
                        "scaler": scaler,
                        "center_1d": center_1d,
                        "scale_1d": scale_1d,
                    }
                )

            _valid_clf = ~np.isnan(lgbm_clf_oof_preds_raw)
            if _valid_clf.sum() > 0:
                best_lgbm_clf_preds = lgbm_clf_oof_preds_raw.copy()
                _combined_clf = ridge_oof_preds + 0.5 * best_lgbm_clf_preds
                if _use_atr:
                    raw_conf_clf = np.clip(
                        _combined_clf * _atr_safe, 0.0, 3.0 * _atr_safe
                    ).astype(np.float32)
                else:
                    raw_conf_clf = np.maximum(
                        0.0, _combined_clf * ridge_resid_std
                    ).astype(np.float32)

                best_lgbm_clf_metrics = evaluate_signal(
                    "LGBM_Clf_Score(under-over)",
                    raw_conf_clf,
                    y_raw_net_return,
                    y_downside,
                    directionality="return-like",
                )
                lgbm_clf_sizer_eval = best_lgbm_clf_metrics
                best_lgbm_clf_utility = best_lgbm_clf_metrics.get(
                    "utility_score", -np.inf
                )

                lgbm_clf_profit_proxy_df, _, _ = evaluate_selection_profit_proxy(
                    raw_conf_clf,
                    y_raw_net_return,
                    timestamps=timestamps,
                    symbols=_sym_vals,
                    top_fracs=list(top_fracs),
                    cost_pct=cost_pct,
                    n_days=n_days,
                )
                best_lgbm_clf_objective = (
                    float(lgbm_clf_profit_proxy_df["wallet_pnl"].max())
                    if not lgbm_clf_profit_proxy_df.empty
                    else float("-inf")
                )
                tprint(
                    f"  LGBM Clf: utility={best_lgbm_clf_utility:.4f}, "
                    f"wallet_pnl={best_lgbm_clf_objective:.4f}"
                )

                # --- LGBM Classifier confidence/uncertainty ---
                def _compute_lgbm_clf_confidence(
                    fold_dicts: List[Dict[str, Any]],
                    X_full: np.ndarray,
                    splits_list: List[Tuple[np.ndarray, np.ndarray]],
                ) -> np.ndarray:
                    n = X_full.shape[0]
                    proba_stacks = np.zeros((n, 3, 0), dtype=np.float32)
                    for fd in fold_dicts:
                        te_idx = fd["te_idx"]
                        m = fd["model"]
                        X_te_clean, _, _, _, _ = clean_and_standardize(
                            X_full[te_idx],
                            fit_medians=fd["medians"],
                            scaler=fd["scaler"],
                            center_1d=fd["center_1d"],
                            scale_1d=fd["scale_1d"],
                        )
                        p = np.asarray(m.predict_proba(X_te_clean), dtype=np.float32)
                        _stack = np.zeros((n, 3, 1), dtype=np.float32)
                        _stack[te_idx, :, 0] = p
                        proba_stacks = np.concatenate([proba_stacks, _stack], axis=2)

                    n_obs = proba_stacks.shape[2]
                    conf_arr = np.ones(n, dtype=np.float32)
                    under_over_arr = np.zeros(n, dtype=np.float32)
                    entropy_arr = np.zeros(n, dtype=np.float32)
                    p_under_var_arr = np.zeros(n, dtype=np.float32)
                    for i in range(n):
                        mask = proba_stacks[i, 0, :] > 0
                        if mask.sum() < 2:
                            continue
                        p_under = proba_stacks[i, 0, mask]
                        p_over = proba_stacks[i, 2, mask]
                        p_ok = proba_stacks[i, 1, mask]
                        uo = float(np.mean(p_under - p_over))
                        under_over_arr[i] = uo
                        p_mean = np.mean(proba_stacks[i, :, mask], axis=1)
                        p_mean_clipped = np.clip(p_mean, 1e-12, 1.0)
                        entropy_arr[i] = float(
                            -np.sum(p_mean_clipped * np.log(p_mean_clipped))
                        )
                        p_under_var_arr[i] = float(np.var(p_under))

                    p_under_over_scaled = _quantile_scale_to_band(under_over_arr)
                    entropy_inv = _quantile_scale_to_band(1.0 / (entropy_arr + 1e-6))
                    p_under_var_scaled = _quantile_scale_to_band(p_under_var_arr)

                    log_conf = (
                        2.0 * np.log(p_under_over_scaled + 1e-12)
                        + 1.0 * np.log(entropy_inv + 1e-12)
                        + 1.0 * np.log(p_under_var_scaled + 1e-12)
                    ) / 4.0
                    raw_wgm = np.exp(log_conf).astype(np.float32)
                    median_wgm = float(np.median(raw_wgm))
                    if abs(median_wgm) > 1e-12:
                        conf_arr = np.clip(raw_wgm / median_wgm, 0.7, 1.3).astype(
                            np.float32
                        )
                    return conf_arr, under_over_arr, entropy_arr

                lgbm_clf_confidence_all = np.ones(
                    len(y_raw_net_return), dtype=np.float32
                )
                lgbm_clf_under_over = np.zeros(len(y_raw_net_return), dtype=np.float32)
                lgbm_clf_entropy = np.zeros(len(y_raw_net_return), dtype=np.float32)
                try:
                    (
                        lgbm_clf_confidence_all,
                        lgbm_clf_under_over,
                        lgbm_clf_entropy,
                    ) = _compute_lgbm_clf_confidence(
                        fold_lgbm_clf_models, X_lgbm_clf, splits
                    )
                except Exception as _e_clf_conf:
                    tprint(f"  LGBM Clf confidence failed: {_e_clf_conf}")

                results["lgbm_clf_oof_score_"] = best_lgbm_clf_preds
                results["lgbm_clf_confidence_"] = lgbm_clf_confidence_all
                results["lgbm_clf_under_over_"] = lgbm_clf_under_over
                results["lgbm_clf_entropy_"] = lgbm_clf_entropy
                results["fold_lgbm_clf_models_"] = fold_lgbm_clf_models
                results["lgbm_clf_feature_keys_"] = lgbm_clf_selected_keys
                results["lgbm_clf_best_params_"] = best_lgbm_clf_params

                combined_lgbm_clf_objective = best_lgbm_clf_objective
                if not best_combo or combined_lgbm_clf_objective > best_combo_objective:
                    if (
                        best_simple_score is None
                        or combined_lgbm_clf_objective > combined_et_objective
                    ):
                        best_simple_score = raw_conf_clf
                        best_simple_score_name = "RidgePlusLGBM_Clf_TotalConfidence"

    # --- 3-Way Pipeline Comparison ---
    comparison: Dict[str, Any] = {}
    comparison_rows = []
    ridge_util = ridge_sizer_eval.get("utility_score", 0.0) if ridge_sizer_eval else 0.0
    ridge_wallet = (
        float(ridge_profit_proxy_df["wallet_pnl"].max())
        if not ridge_profit_proxy_df.empty
        else 0.0
    )
    comparison_rows.append(
        {
            "pipeline": "ridge_only",
            "utility": ridge_util,
            "wallet_pnl": ridge_wallet,
        }
    )
    if et_sizer_eval:
        et_util = et_sizer_eval.get("utility_score", 0.0)
        et_wallet = (
            float(et_profit_proxy_df["wallet_pnl"].max())
            if not et_profit_proxy_df.empty
            else 0.0
        )
        comparison_rows.append(
            {
                "pipeline": "ridge_plus_et",
                "utility": et_util,
                "wallet_pnl": et_wallet,
            }
        )
    if lgbm_sizer_eval:
        lgbm_util = lgbm_sizer_eval.get("utility_score", 0.0)
        lgbm_wallet = (
            float(lgbm_profit_proxy_df["wallet_pnl"].max())
            if not lgbm_profit_proxy_df.empty
            else 0.0
        )
        comparison_rows.append(
            {
                "pipeline": "ridge_plus_lgbm",
                "utility": lgbm_util,
                "wallet_pnl": lgbm_wallet,
            }
        )
    if lgbm_clf_sizer_eval:
        lgbm_clf_util = lgbm_clf_sizer_eval.get("utility_score", 0.0)
        lgbm_clf_wallet = (
            float(lgbm_clf_profit_proxy_df["wallet_pnl"].max())
            if not lgbm_clf_profit_proxy_df.empty
            else 0.0
        )
        comparison_rows.append(
            {
                "pipeline": "ridge_plus_lgbm_clf",
                "utility": lgbm_clf_util,
                "wallet_pnl": lgbm_clf_wallet,
            }
        )
    if comparison_rows:
        best_row = max(comparison_rows, key=lambda r: r["wallet_pnl"])
        comparison = {
            "rows": comparison_rows,
            "winner": best_row["pipeline"],
            "winner_wallet_pnl": best_row["wallet_pnl"],
            "winner_utility": best_row["utility"],
        }
        tprint(
            f"4-Way Pipeline Comparison: "
            + " | ".join(
                f"{r['pipeline']}={r['wallet_pnl']:.4f}" for r in comparison_rows
            )
            + f" | winner={comparison['winner']}"
        )
    results["comparison_"] = comparison
    results["comparison_table_"] = (
        pd.DataFrame(comparison_rows) if comparison_rows else pd.DataFrame()
    )

    # --- Grid search: ridge/booster ratio + confidence multiplier ---
    _mix_grid_results: List[Dict[str, Any]] = []
    _winner_name = comparison.get("winner", "")
    if _winner_name and _winner_name != "ridge_only" and ridge_oof_available:
        _booster_raw = None
        _booster_conf = None
        _winner_profit_df = pd.DataFrame()
        if _winner_name == "ridge_plus_et":
            _booster_raw = results.get("et_raw_residual_preds_")
            _booster_conf = results.get("et_confidence_")
            _winner_profit_df = et_profit_proxy_df
        elif _winner_name == "ridge_plus_lgbm":
            _booster_raw = results.get("lgbm_raw_residual_preds_")
            _booster_conf = results.get("lgbm_confidence_")
            _winner_profit_df = lgbm_profit_proxy_df
        elif _winner_name == "ridge_plus_lgbm_clf":
            _booster_raw = results.get("lgbm_clf_oof_score_")
            _booster_conf = results.get("lgbm_clf_confidence_")
            _winner_profit_df = lgbm_clf_profit_proxy_df

        if (
            _booster_raw is not None
            and _booster_conf is not None
            and not _winner_profit_df.empty
        ):
            if "is_optimal" in _winner_profit_df.columns:
                _opt_row = _winner_profit_df[_winner_profit_df["is_optimal"]].iloc[0]
            else:
                _opt_row = _winner_profit_df.sort_values(
                    "wallet_pnl", ascending=False
                ).iloc[0]
            _opt_frac = float(_opt_row.get("selection_frac", 0.10))
            _k_opt = max(1, int(len(y_raw_net_return) * _opt_frac))

            _ridge_w_grid = [0.5, 0.7, 0.85, 1.0]
            _booster_w_grid = [0.1, 0.25, 0.5, 0.75]
            _conf_m_grid = [0.5, 0.75, 1.0, 1.25]

            for _rw in _ridge_w_grid:
                for _bw in _booster_w_grid:
                    for _cm in _conf_m_grid:
                        _combined = _rw * ridge_oof_preds + _bw * _booster_raw * (
                            _booster_conf * _cm
                        )
                        if _use_atr:
                            _score = np.clip(
                                _combined * _atr_safe,
                                0.0,
                                3.0 * _atr_safe,
                            ).astype(np.float32)
                        else:
                            _score = np.maximum(
                                0.0, _combined * ridge_resid_std
                            ).astype(np.float32)

                        _idx = np.argpartition(_score, -_k_opt)[-_k_opt:]
                        _sel_rets = y_raw_net_return[_idx]
                        _sorted_args = np.argsort(_score[_idx])
                        _allocs = np.linspace(0.05, 0.15, len(_idx))
                        _w_rets = (_sel_rets[_sorted_args] - cost_pct) * _allocs
                        _wp = float(np.sum(_w_rets))

                        _net = _sel_rets - cost_pct
                        _gross_win = float(np.sum(_net[_net > 0]))
                        _gross_loss = float(np.abs(np.sum(_net[_net < 0])))
                        _pf = (
                            _gross_win / _gross_loss
                            if _gross_loss > 1e-12
                            else (100.0 if _gross_win > 0 else 0.0)
                        )

                        _mix_grid_results.append(
                            {
                                "winner": _winner_name,
                                "ridge_w": _rw,
                                "booster_w": _bw,
                                "conf_mult": _cm,
                                "threshold_frac": _opt_frac,
                                "wallet_pnl": _wp,
                                "profit_factor": _pf,
                                "hit_rate": float(np.mean(_sel_rets > 0)),
                                "trades": len(_sel_rets),
                            }
                        )

        if _mix_grid_results:
            _mix_df = pd.DataFrame(_mix_grid_results)
            _best_mix = _mix_df.loc[_mix_df["wallet_pnl"].idxmax()]
            tprint(
                f"  Mix grid winner: ridge_w={_best_mix['ridge_w']:.2f} "
                f"booster_w={_best_mix['booster_w']:.2f} "
                f"conf_mult={_best_mix['conf_mult']:.2f} "
                f"wallet_pnl={_best_mix['wallet_pnl']:.4f} "
                f"PF={_best_mix['profit_factor']:.2f} "
                f"hit={_best_mix['hit_rate']:.2f}"
            )
            results["mix_grid_table_"] = _mix_df
            results["mix_grid_best_"] = {
                "winner": _best_mix["winner"],
                "ridge_w": float(_best_mix["ridge_w"]),
                "booster_w": float(_best_mix["booster_w"]),
                "conf_mult": float(_best_mix["conf_mult"]),
                "threshold_frac": float(_best_mix["threshold_frac"]),
                "wallet_pnl": float(_best_mix["wallet_pnl"]),
                "profit_factor": float(_best_mix["profit_factor"]),
            }
        else:
            results["mix_grid_table_"] = pd.DataFrame()
            results["mix_grid_best_"] = {}
    else:
        results["mix_grid_table_"] = pd.DataFrame()
        results["mix_grid_best_"] = {}

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
        "mix_grid_table_": results.get("mix_grid_table_", pd.DataFrame()),
        "mix_grid_best_": results.get("mix_grid_best_", {}),
        "ridge_sizer_eval_": ridge_sizer_eval,
        "ridge_importance_table_": ridge_importance_df,
        "ridge_profit_proxy_table_": ridge_profit_proxy_df,
        "et_sizer_eval_": et_sizer_eval,
        "et_importance_table_": et_importance_df,
        "et_profit_proxy_table_": et_profit_proxy_df,
        "lgbm_sizer_eval_": lgbm_sizer_eval,
        "lgbm_importance_table_": lgbm_importance_df,
        "lgbm_profit_proxy_table_": lgbm_profit_proxy_df,
        "lgbm_clf_sizer_eval_": lgbm_clf_sizer_eval,
        "lgbm_clf_importance_table_": lgbm_clf_importance_df,
        "lgbm_clf_profit_proxy_table_": lgbm_clf_profit_proxy_df,
        "comparison_": comparison,
        "comparison_table_": results.get("comparison_table_", pd.DataFrame()),
        "best_simple_score_": best_simple_score,
        "best_simple_score_name_": best_simple_score_name,
        "best_combo_profit_proxy_table_": best_combo_profit_proxy_df,
        "profit_proxy_table_": (
            profit_proxy_df if not profit_proxy_df.empty else pd.DataFrame()
        ),
        "opt_rets_": best_opt_rets,
        "opt_ts_": best_opt_ts,
        "barrier_clf_importance_": barrier_clf_importance_df,
        "barrier_clf_oof_proba_": barrier_clf_oof_proba,
        "oof_p_tp_": results.get("oof_p_tp_"),
        "oof_p_sl_": results.get("oof_p_sl_"),
        "oof_p_time_": results.get("oof_p_time_"),
        "ridge_sizer_scores_": results.get("ridge_sizer_scores_"),
        "et_raw_residual_preds_": results.get("et_raw_residual_preds_"),
        "et_sizer_scores_": results.get("et_sizer_scores_"),
        "et_confidence_": results.get("et_confidence_"),
        "raw_confidence_": results.get("raw_confidence_"),
        "total_confidence_": results.get("total_confidence_"),
        "calibration_isotonic_": results.get("calibration_isotonic_"),
        "ridge_resid_mean_": results.get("ridge_resid_mean_"),
        "ridge_resid_std_": results.get("ridge_resid_std_"),
        "fold_et_models_": results.get("fold_et_models_"),
        "et_best_params_": results.get("et_best_params_"),
        "lgbm_raw_residual_preds_": results.get("lgbm_raw_residual_preds_"),
        "lgbm_sizer_scores_": results.get("lgbm_sizer_scores_"),
        "lgbm_confidence_": results.get("lgbm_confidence_"),
        "raw_confidence_lgbm_": results.get("raw_confidence_lgbm_"),
        "total_confidence_lgbm_": results.get("total_confidence_lgbm_"),
        "calibration_isotonic_lgbm_": results.get("calibration_isotonic_lgbm_"),
        "fold_lgbm_models_": results.get("fold_lgbm_models_"),
        "lgbm_best_params_": results.get("lgbm_best_params_"),
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
        b_kwargs = dict(kwargs)
        if "y_atr_target" in b_kwargs and b_kwargs["y_atr_target"] is not None:
            b_kwargs["y_atr_target"] = b_kwargs["y_atr_target"][mask]
        if "atr_vals" in b_kwargs and b_kwargs["atr_vals"] is not None:
            b_kwargs["atr_vals"] = b_kwargs["atr_vals"][mask]

        b_res = run_simple_position_sizer(
            b_feature_dict,
            b_trade_outcomes,
            b_y_raw_net_return,
            b_y_downside,
            b_timestamps,
            (
                b_trade_outcomes["symbol"].values
                if "symbol" in b_trade_outcomes.columns
                else None
            ),
            bucket_labels=None,
            sample_weight=b_sample_weight,
            **b_kwargs,
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
    use_et_head_sizer: bool = False,
    use_lgbm_head_sizer: bool = True,
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

    # Load dynamic strategies (which rules are active per bucket)
    # load_inference_candidate_mask_params_per_bucket returns top_n PER (side, horizon) group.
    # We load a large pool then take the true global top-N by score_for_best_params.
    _pool = load_inference_candidate_mask_params_per_bucket(
        top_n=99, ranking_metric="score_for_best_params"
    )

    if not _pool:
        fallback_path = (
            Path(data_root)
            / "artifacts"
            / run_id
            / "ridge_sizer"
            / "strategy_params.json"
        )
        if fallback_path.exists():
            try:
                payload = json.loads(fallback_path.read_text())
                buckets = payload.get("buckets", {})
                for sid, row in buckets.items():
                    _pool.append({"strategy_id": sid, **row})
                logger.info(
                    f"Fallback: loaded {len(_pool)} strategies from {fallback_path}"
                )
            except Exception as e:
                logger.warning(f"Fallback strategy load failed: {e}")

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

    # Load base and meta OOFs — BOTH required
    base_oofs = load_base_oof_predictions(data_root, run_id)
    try:
        meta_oofs = load_meta_oof_predictions(data_root, run_id)
    except Exception as e:
        logger.error(
            f"Could not load meta OOFs: {e}. Aborting — both base and meta OOFs are required."
        )
        return {}

    if not base_oofs:
        logger.error(
            f"No base OOFs found in {data_root}/artifacts/{run_id}/oof/. Aborting."
        )
        return {}
    if not meta_oofs:
        logger.error(
            f"No meta OOFs found in {data_root}/artifacts/{run_id}/meta_oof/. Aborting."
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

    if len(strategies) > top_n_strategies:
        logger.info(
            f"Capping {len(strategies)} strategies to top-{top_n_strategies} after OOF injection"
        )
        strategies = strategies[:top_n_strategies]

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
                best_match = None
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
                _tf_start_ts = (
                    pd.to_datetime(_tf_start, utc=True)
                    if _tf_start is not None
                    else None
                )
                _tf_end_ts = (
                    pd.to_datetime(_tf_end, utc=True) if _tf_end is not None else None
                )
                _tf_mask = np.ones(len(full_df), dtype=bool)
                if _tf_start_ts is not None:
                    _tf_mask &= _tf_ts >= _tf_start_ts
                if _tf_end_ts is not None:
                    _tf_mask &= _tf_ts < _tf_end_ts
                full_df = full_df.loc[_tf_mask].reset_index(drop=True)
                logger.info(
                    f"Time filter [{_tf_start}, {_tf_end}): {len(full_df)} rows retained"
                )

        trade_side = str(strategy.get("trade_side", "") or "")

        # 2. Resolve OOF bucket by matching strategy_id against meta_oof AND base_oof keys
        oof_df = pd.DataFrame()
        resolved_meta_key = None
        resolved_base_key = None

        import re as _re_oof

        def _strip_side(k):
            return _re_oof.sub(r"^(long|short)_", "", k)

        def _best_prefix_match(strategy_id: str, keys) -> tuple:
            strat_norm = _re_oof.sub(r"[^a-z0-9]", "", strategy_id.lower())
            best_k, best_s = None, 0
            for k in keys:
                k_norm = _re_oof.sub(r"[^a-z0-9]", "", _strip_side(k).lower())
                plen = 0
                for a, b in zip(strat_norm, k_norm):
                    if a == b:
                        plen += 1
                    else:
                        break
                if plen > best_s:
                    best_s = plen
                    best_k = k
            return (best_k, best_s) if best_s >= 20 else (None, 0)

        prefixed = f"{trade_side}_{strategy_id}" if trade_side else strategy_id
        if prefixed in meta_oofs:
            resolved_meta_key = prefixed
        elif strategy_id in meta_oofs:
            resolved_meta_key = strategy_id
        else:
            resolved_meta_key, _ = _best_prefix_match(strategy_id, meta_oofs.keys())

        if prefixed in base_oofs:
            resolved_base_key = prefixed
        elif strategy_id in base_oofs:
            resolved_base_key = strategy_id
        else:
            resolved_base_key, _ = _best_prefix_match(strategy_id, base_oofs.keys())

        if resolved_meta_key:
            oof_df = meta_oofs[resolved_meta_key]
            logger.info(
                f"Resolved meta OOF: '{strategy_id[:40]}' -> '{resolved_meta_key[:40]}'"
            )

        if resolved_base_key and base_oofs[resolved_base_key] is not None:
            base_part = base_oofs[resolved_base_key]
            if not isinstance(base_part, pd.DataFrame):
                base_part = pd.DataFrame(base_part)
            if oof_df.empty:
                oof_df = base_part
                logger.info(
                    f"Resolved base OOF (no meta): '{strategy_id[:40]}' -> '{resolved_base_key[:40]}'"
                )
            else:
                base_cols = [c for c in base_part.columns if c not in oof_df.columns]
                if base_cols:
                    n = min(len(oof_df), len(base_part))
                    for c in base_cols:
                        oof_df[c] = base_part[c].values[:n]
                    logger.info(
                        f"Merged {len(base_cols)} base columns into OOF from '{resolved_base_key[:40]}'"
                    )

        if resolved_meta_key or resolved_base_key:
            src_key = resolved_meta_key or resolved_base_key
            inferred_side = str(oof_df.attrs.get("trade_side", "") or "")
            if not inferred_side:
                inferred_side = (
                    "long"
                    if str(src_key).startswith("long_")
                    else "short" if str(src_key).startswith("short_") else ""
                )
            if inferred_side and not trade_side:
                trade_side = inferred_side

        if not trade_side:
            if "is_long" in full_df.columns:
                _is_long_vals = np.asarray(full_df["is_long"].values, dtype=float)
                _finite_long = np.isfinite(_is_long_vals)
                if _finite_long.any():
                    trade_side = (
                        "long"
                        if float(np.nanmean(_is_long_vals[_finite_long])) >= 0.5
                        else "short"
                    )
            if not trade_side:
                trade_side = (
                    "long"
                    if strategy_id.startswith("long_")
                    else "short" if strategy_id.startswith("short_") else ""
                )

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
            _tf_start_ts = (
                pd.to_datetime(_tf_start, utc=True) if _tf_start is not None else None
            )
            _tf_end_ts = (
                pd.to_datetime(_tf_end, utc=True) if _tf_end is not None else None
            )
            _tf_mask = np.ones(len(active_df), dtype=bool)
            if _tf_start_ts is not None:
                _tf_mask &= _tf_ts >= _tf_start_ts
            if _tf_end_ts is not None:
                _tf_mask &= _tf_ts < _tf_end_ts
            active_df = active_df.loc[_tf_mask].reset_index(drop=True)
            logger.info(
                f"OOF time filter [{_tf_start}, {_tf_end}): {len(active_df)} rows retained"
            )

        active_joined_df = active_df

        _sizer_cap = 65000
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
            or c.startswith(("tbm_", "mae_h", "mfe_h", "asym_h"))
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
        print("\n" + "=" * 80)
        print(f" TARGETING STRATEGY: {strategy_id[:65]}...")
        print(" " + "-" * 78)
        raw_to_matched = n_matched_labels / max(n_raw_labels, 1)
        matched_to_scorable = n_scorable / max(n_matched_labels, 1)
        print(" STRATEGY FUNNEL (Coverage Restoration):")
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
        print("=" * 80 + "\n")

        # Now pass the SCORED df to the sizer logic.
        # Keep trade outcomes aligned with the same scored rows so score/return
        # arrays refer to the exact same universe.
        active_df = active_scored_df.reset_index(drop=True)
        trade_outcomes = trade_outcomes.loc[np.asarray(scorable_mask)].reset_index(
            drop=True
        )
        active_df = _augment_meta_clf_reliability_features(active_df, trade_outcomes)

        _base_prob_col = next(
            (
                c
                for c in [
                    "oof_prob",
                    "clf",
                    "pred",
                    "oof_pred",
                    "base_h4",
                    "base_h2",
                ]
                if c in active_df.columns
            ),
            None,
        )
        if _base_prob_col is not None and len(active_df) == len(trade_outcomes):
            _asset_col = (
                "symbol"
                if "symbol" in active_df.columns
                else "__symbol__" if "__symbol__" in active_df.columns else None
            )
            _ts_col = (
                "timestamp"
                if "timestamp" in active_df.columns
                else "__ts__" if "__ts__" in active_df.columns else None
            )
            _side_ret = np.asarray(trade_outcomes["return"].values, dtype=np.float32)
            _y_bin = (_side_ret > 0).astype(np.float32)
            _mask_res = select_top_rank_mask(
                base_prob=np.asarray(
                    active_df[_base_prob_col].values, dtype=np.float32
                ),
                strategy_mask=np.ones(len(active_df), dtype=bool),
                symbols=(
                    np.asarray(active_df[_asset_col].astype(str).values)
                    if _asset_col is not None
                    else np.asarray(np.repeat("all", len(active_df)))
                ),
                timestamps=(
                    pd.to_datetime(active_df[_ts_col], errors="coerce")
                    if _ts_col is not None
                    else None
                ),
                outcomes=_y_bin,
                mfe=(
                    np.asarray(trade_outcomes["mfe_ret"].values, dtype=np.float32)
                    if "mfe_ret" in trade_outcomes.columns
                    else None
                ),
                mae=(
                    np.asarray(trade_outcomes["mae_ret"].values, dtype=np.float32)
                    if "mae_ret" in trade_outcomes.columns
                    else None
                ),
                t_mfe=(
                    np.asarray(trade_outcomes["t_mfe"].values, dtype=np.float32)
                    if "t_mfe" in trade_outcomes.columns
                    else None
                ),
                t_mae=(
                    np.asarray(trade_outcomes["t_mae"].values, dtype=np.float32)
                    if "t_mae" in trade_outcomes.columns
                    else None
                ),
                tp=(
                    np.asarray(active_df["__barrier_pct__"].values, dtype=np.float32)
                    if "__barrier_pct__" in active_df.columns
                    else np.full(len(active_df), 0.02, dtype=np.float32)
                ),
            )
            _rank_mask = np.asarray(_mask_res.mask, dtype=bool)
            if int(np.sum(_rank_mask)) > 50:
                active_df = active_df.loc[_rank_mask].reset_index(drop=True)
                trade_outcomes = trade_outcomes.loc[_rank_mask].reset_index(drop=True)
                logger.info(
                    f"Applied top-rank trade mask in sizer: topx={_mask_res.chosen_topx}% "
                    f"coverage={_mask_res.coverage:.3f} kept={int(np.sum(_rank_mask))}/{len(_rank_mask)}"
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
        head_cols = [
            c
            for c in head_cols
            if (c not in _NON_HEAD_DROP_FEATURES) or _is_classifier_head_feature(c)
        ]

        numeric_head_cols: List[str] = []
        dropped_non_numeric_heads: List[str] = []
        for col in head_cols:
            try:
                if pd.api.types.is_numeric_dtype(active_df[col]):
                    numeric_head_cols.append(col)
                else:
                    dropped_non_numeric_heads.append(col)
            except Exception:
                dropped_non_numeric_heads.append(col)
        if dropped_non_numeric_heads:
            logger.info(
                f"Strategy {strategy_id[:50]}: dropping non-numeric heads from sizer inputs: "
                f"{dropped_non_numeric_heads}"
            )
        head_cols = numeric_head_cols

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
            "asym": any(
                c.lower().startswith("asym_h") or c == "oof_asym_hat" for c in head_cols
            ),
        }

        def _family_available_in_sources(family: str) -> bool:
            source_frames: List[pd.DataFrame] = list(base_oofs.values()) + list(
                meta_oofs.values()
            )
            if family == "base":
                return any(
                    any(c.lower().startswith("base_h") for c in src.columns)
                    for src in source_frames
                )
            if family == "classifier":
                return any(
                    any(
                        c == "clf" or c.lower().startswith("oof_p_")
                        for c in src.columns
                    )
                    for src in source_frames
                )
            if family == "mae":
                return any(
                    any(
                        c.lower().startswith("mae_h")
                        or c in {"mae_mean", "mae_std", "oof_log_mae_q70_hat"}
                        or c == "oof_mae_q70_hat"
                        for c in src.columns
                    )
                    for src in source_frames
                )
            if family == "mfe":
                return any(
                    any(
                        c.lower().startswith("mfe_h")
                        or c in {"mfe_mean", "mfe_std", "oof_log_mfe_hat"}
                        or c == "oof_mfe_hat"
                        for c in src.columns
                    )
                    for src in source_frames
                )
            if family == "reg":
                return any(
                    any(
                        c in {"reg", "reg_mean", "oof_pred", "oof_pred_oriented"}
                        or c.lower().startswith("reg_h")
                        for c in src.columns
                    )
                    for src in source_frames
                )
            if family == "asym":
                return any(
                    any(
                        c.lower().startswith("asym_h")
                        or c == "oof_asym_hat"
                        or "asym" in c.lower()
                        for c in src.columns
                    )
                    for src in source_frames
                )
            return False

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
            join_gap = [
                fam for fam in missing_fams if _family_available_in_sources(fam)
            ]
            export_gap = [fam for fam in missing_fams if fam not in join_gap]
            if join_gap:
                logger.warning(
                    f"Strategy {strategy_id[:50]}: missing expected head families after join: {join_gap}"
                )
            if export_gap:
                logger.info(
                    f"Strategy {strategy_id[:50]}: no exported heads found for families {export_gap} in this run; they were not available to join"
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

        try:
            target_horizon = max(1, int(float(strategy.get("source_horizon", 1))))
        except (TypeError, ValueError):
            target_horizon = 1

        y_atr_target, atr_vals, atr_col = _build_blended_sizer_target(
            active_df,
            horizon=target_horizon,
        )
        if y_atr_target is not None:
            logger.info(
                f"Blended sizer target via close/kalman_price: horizon={target_horizon}, "
                f"valid={np.isfinite(y_atr_target).mean():.1%}, "
                f"target std={float(np.nanstd(y_atr_target)):.6f}"
            )

        if y_atr_target is None:
            for _atr_c in ("atr_12_15m", "atr_pct"):
                if _atr_c in trade_outcomes.columns:
                    atr_col = _atr_c
                    break
                if _atr_c in active_df.columns:
                    atr_col = _atr_c
                    break
        if y_atr_target is None and atr_col is not None:
            _atr_raw = (
                trade_outcomes[atr_col].values
                if atr_col in trade_outcomes.columns
                else active_df[atr_col].values
            )
            atr_safe = np.where(np.abs(_atr_raw) > 1e-8, _atr_raw, np.nan)
            y_atr_target = np.asarray(y_raw_net_return / atr_safe, dtype=np.float32)
            valid_atr = np.isfinite(y_atr_target)
            if valid_atr.mean() < 0.5:
                logger.warning(
                    f"ATR normalization: only {valid_atr.mean():.1%} valid values "
                    f"from '{atr_col}' — falling back to raw returns"
                )
                y_atr_target = None
            else:
                atr_vals = _atr_raw.astype(np.float32)
                logger.info(
                    f"ATR normalization via '{atr_col}': "
                    f"{valid_atr.mean():.1%} valid, "
                    f"target std={float(np.nanstd(y_atr_target)):.6f}"
                )

        feature_dict = {col: active_df[col].values for col in head_cols}

        res = run_simple_position_sizer(
            feature_dict=feature_dict,
            trade_outcomes=trade_outcomes,
            y_raw_net_return=y_raw_net_return,
            y_downside=y_downside,
            timestamps=timestamps,
            symbols=(
                trade_outcomes["symbol"].values
                if "symbol" in trade_outcomes.columns
                else None
            ),
            bucket_labels=None,
            top_fracs=top_fracs,
            use_ridge_head_sizer=use_ridge_head_sizer,
            use_et_head_sizer=use_et_head_sizer,
            use_lgbm_head_sizer=use_lgbm_head_sizer,
            y_atr_target=y_atr_target,
            atr_vals=atr_vals,
            target_horizon=target_horizon,
        )

        res["_strategy_meta_"] = {
            "trade_side": trade_side,
            "source_target": strategy.get("source_target", ""),
            "source_horizon": strategy.get("source_horizon", np.nan),
            "atr_column": atr_col,
        }
        _score_ref = res.get("total_confidence_")
        if _score_ref is None:
            _score_ref = res.get("ridge_sizer_scores_")
        if _score_ref is None:
            _score_ref = res.get("lgbm_sizer_scores_")
        if _score_ref is None:
            _score_ref = res.get("et_sizer_scores_")
        _n_score = len(_score_ref) if _score_ref is not None else len(trade_outcomes)
        _ctx: Dict[str, Any] = {}
        if "timestamp" in active_df.columns:
            _ctx["timestamp"] = (
                pd.to_datetime(active_df["timestamp"], utc=True, errors="coerce")
                .astype(str)
                .to_numpy()[:_n_score]
            )
        elif "timestamp" in trade_outcomes.columns:
            _ctx["timestamp"] = (
                pd.to_datetime(trade_outcomes["timestamp"], utc=True, errors="coerce")
                .astype(str)
                .to_numpy()[:_n_score]
            )
        if "symbol" in active_df.columns:
            _ctx["symbol"] = active_df["symbol"].astype(str).to_numpy()[:_n_score]
        elif "symbol" in trade_outcomes.columns:
            _ctx["symbol"] = (
                trade_outcomes["symbol"].astype(str).to_numpy()[:_n_score]
            )
        if _ctx:
            res["_sizer_oof_context_"] = _ctx
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
    _persist_detailed_model_metrics(data_root, run_id, strategy_results)
    _persist_mix_grid(data_root, run_id, strategy_results)
    for _sid, _sres in strategy_results.items():
        _persist_booster_bundle(data_root, run_id, _sid, _sres)
    try:
        _fi_path = _persist_feature_importance_summary(
            data_root, run_id, strategy_results
        )
        if _fi_path is not None:
            logger.info(f"Saved sizer feature importance summary to {_fi_path}")
    except Exception as e:
        logger.warning(f"Could not save feature importance summary: {e}")

    # Save OOF predictions to parquet for policy_optimiser
    if strategy_results:
        try:
            _persist_sizer_oof_predictions(data_root, run_id, strategy_results)
        except Exception as e:
            logger.warning(f"Could not save OOF predictions: {e}")

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
                        "pnl/trade(%)": opt_row["pnl_per_trade"],
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

                (
                    p_weekly_sortino,
                    p_weekly_pnl_std,
                    p_weekly_tuw,
                    p_weekly_ulcer,
                    p_weekly_pct_negative,
                    p_weekly_worst_pnl,
                ) = compute_period_aggregated_stats(
                    all_portfolio_rets, all_portfolio_ts, "W"
                )
                (
                    p_monthly_sortino,
                    p_monthly_pnl_std,
                    p_monthly_tuw,
                    p_monthly_ulcer,
                    p_monthly_pct_negative,
                    p_monthly_worst_pnl,
                ) = compute_period_aggregated_stats(
                    all_portfolio_rets, all_portfolio_ts, "M"
                )

                leaderboard_rows.append(
                    {
                        "strategy_id": "[PORTFOLIO - Positive PF Only]",
                        "threshold": "Mixed",
                        "wallet_pnl": p_wallet_pnl_total,  # Need to calculate this
                        "net_pnl": p_pnl,
                        "pnl/trade(%)": (p_pnl / len(all_portfolio_rets)) * 100.0,
                        "trades/day": len(all_portfolio_rets) / 725.0,
                        "hit_rate": p_hit,
                        "pf": p_pf,
                        "weekly_sortino": p_weekly_sortino,
                        "monthly_sortino": p_monthly_sortino,
                        "weekly_pnl_std": p_weekly_pnl_std,
                        "monthly_pnl_std": p_monthly_pnl_std,
                        "weekly_tuw": p_weekly_tuw,
                        "weekly_ulcer": p_weekly_ulcer,
                        "weekly_pct_negative": p_weekly_pct_negative,
                        "weekly_worst_pnl": p_weekly_worst_pnl,
                        "monthly_tuw": p_monthly_tuw,
                        "monthly_ulcer": p_monthly_ulcer,
                        "monthly_pct_negative": p_monthly_pct_negative,
                        "monthly_worst_pnl": p_monthly_worst_pnl,
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

    # =============================================================================
    # Compute and Save Full Calibration Curves
    # =============================================================================
    try:
        # Collect all OOF predictions for calibration
        all_oof_preds = []
        for strategy_id, res in strategy_results.items():
            if "profit_proxy_table_" in res:
                table = res["profit_proxy_table_"]
                if isinstance(table, pd.DataFrame) and not table.empty:
                    # Extract scored rows with predictions
                    table["strategy"] = strategy_id
                    all_oof_preds.append(table)

        if all_oof_preds:
            oof_df = pd.concat(all_oof_preds, ignore_index=True)

            # Build realized returns DataFrame from trade_outcomes
            realized_returns_df = pd.DataFrame()
            if trade_outcomes is not None and not trade_outcomes.empty:
                realized_returns_df = trade_outcomes[
                    ["symbol", "timestamp", "return"]
                ].copy()
                realized_returns_df = realized_returns_df.rename(
                    columns={"return": "realized_return"}
                )

            if not oof_df.empty and not realized_returns_df.empty:
                tprint(
                    f"[Calibration] Computing full calibration curves for {len(strategy_results)} strategies..."
                )

                calibration_data = compute_full_calibration_curves(
                    oof_predictions=oof_df,
                    realized_returns=realized_returns_df,
                    strategy_col="strategy",
                    score_col=(
                        "sizer_score"
                        if "sizer_score" in oof_df.columns
                        else "trading_score"
                    ),
                    return_col="realized_return",
                    n_bins=10,
                )

                if calibration_data:
                    save_calibration_curves(calibration_data, data_root, run_id)
                    tprint(
                        f"[Calibration] Saved calibration curves for {len(calibration_data)} strategies"
                    )
                else:
                    tprint(
                        "[Calibration] No calibration data computed (insufficient samples)"
                    )
            else:
                tprint(
                    "[Calibration] Skipping calibration: missing OOF or realized returns data"
                )
        else:
            tprint("[Calibration] No OOF predictions available for calibration")
    except Exception as e:
        logger.warning(f"[Calibration] Failed to compute calibration curves: {e}")

    return strategy_results


if __name__ == "__main__":
    import argparse
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
            print("\n============================================================")
            print(f" STRATEGY: {strategy_id}")
            print("============================================================\n")

            if "head_diagnostics_table_" in res:
                print("\nTop 5 Meta-Heads by Utility:")
                print(res["head_diagnostics_table_"].head(5).to_string(index=False))

            if "ridge_importance_table_" in res:
                print(
                    "\nMeta-Head Importance (Ridge Weights - Strictly Walk-Forward OOF):"
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
            print("\nConfidence Grid (Sizing: 5% to 15% Rank-Based):")
            print(res.get("profit_proxy_table_", pd.DataFrame()).to_string(index=False))
            print("-" * 60)

    except KeyboardInterrupt:
        tprint("Execution interrupted by user.")
    except Exception as e:
        tprint(f"CRITICAL ERROR: {e}")
        import traceback

        tprint(traceback.format_exc())
        sys.exit(1)
