"""
Weighted Meta-Labeling Step (Production Pipeline).

This step extends the feature_generation_meta_labeling_step with sample weighting
capabilities discovered by meta_labeling_hpo_sample_weighted.

Key additions:
1. Loads optimal weighting parameters from HPO output
2. Applies generate_weights_per_label for sample weighting during training
3. Uses calibration-adjusted position sizing for evaluation
4. Integrates with the weighted HPO pipeline

Usage:
    python src/launcher/ares_launcher.py \\
        --step weighted_meta_labeling \\
        --symbol ETHUSDT --exchange binance --timeframe 15m --direction long \\
        --execution-mode full

Prerequisites:
    Run meta_labeling_hpo_sample_weighted first to generate optimal parameters.
"""


import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

# Import core functionality from feature_generation_meta_labeling_step
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    # Core labeling functions
    compute_realized_returns,
    generate_primary_signals,
    kalman_smooth_labels,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
    create_regime_aware_quantile_labels_from_vol_scaled_returns,
    attach_rolling_hmm_regimes_to_market_data,
    # Feature engineering
    create_meta_features,
    build_meta_features_for_model,
    # Model training
    train_bagged_lgbm_with_kfold,
    create_base_models,
    # Calibration
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    # Diagnostics
    generate_diagnostics_report,
    compute_learnability_score,
    compute_label_entropy_score,
    calculate_calibration_metrics,
    # Constants
    DEFAULT_PROFIT_THRESHOLD,
    DEFAULT_STOP_THRESHOLD,
    ECON_MIN_RETURN_MULTIPLE,
    # The base step class
    FeatureGenerationMetaLabelingStep,
)
from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

from src.training.steps.labeling.label_based_layer_1 import run_layer1_optimization
# Import sample weighting utilities
from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_horizon_consistency,
    compute_uniqueness,
)

# Import Kalman multi-triple-barrier labeling system
from src.training.steps.labeling.multi_label_voting_utils import (
    kalman_multi_triple_barrier_labels,
    compute_kalman_smoothed_price_and_volatility,
)


# Import Kalman/RTS functions from HPO module
from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import (
    generate_kalman_features,
    rts_smoother_1d,
    kalman_filter_1d,
    smooth_prices_rts,
    # Feature selection functions (De Prado pipeline)
    select_features_with_quality,
    calculate_feature_quality,
    calculate_time_robust_quality,
    calculate_all_feature_qualities,
    reduce_features_by_correlation,
    select_features_hierarchical,
    lgbm_magnitude_sweep,
    generate_multi_horizon_features,
    # Cross-feature interactions
    generate_cross_features,
    get_cross_feature_inventory,
    get_feature_inventory,
    # Caching utilities
    load_cached_feature_selection,
    save_feature_selection_cache,
    invalidate_feature_selection_cache,
    # Diagnostics utilities
    run_lag1_stress_test,
    compute_dummy_baseline_auc,
)


def _load_latest_hpo_feature_selection(
    symbol: str,
    timeframe: str,
    outcomes_dir: Path = Path("outcomes"),
) -> Optional[Dict[str, Any]]:
    try:
        if not outcomes_dir.exists():
            return None
        pattern = f"hpo_feature_selection_{symbol}_{timeframe}_*.json"
        candidates = list(outcomes_dir.glob(pattern))
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(latest, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        payload["_path"] = str(latest)
        return payload
    except Exception:
        return None


# Default weighting parameters (fallback if HPO not run)
DEFAULT_WEIGHTING_PARAMS = {
    'mag_compression': 0.8,
    'learn_slope': 10.0,
    'learn_center': 0.4,
    'uniq_intensity': 1.0,
    'exp_mag': 1.0,
    'exp_learn': 1.0,
    'exp_uniq': 1.0,
    'exp_cross': 1.0,
    'downside_multiplier': 1.0,
}

# Default Kalman/RTS parameters (fallback if HPO not run)
DEFAULT_KALMAN_PARAMS = {
    'kalman_Q': 1e-4,  # Process noise
    'kalman_R': 0.01,   # Measurement noise
}


def _compute_date_range_days(index: pd.Index) -> Optional[float]:
    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
            delta = index[-1] - index[0]
            return max(float(delta.total_seconds() / 86400.0), 1.0)
    except Exception:
        return None
    return None


def _assign_feature_group(feature_name: str) -> str:
    name = str(feature_name)
    lname = name.lower()

    if lname.startswith("shap_int__"):
        return "interaction"

    if lname.startswith("mr_"):
        return "mr"

    if name.startswith(("KF_", "KC_", "PV_", "VN_", "XH_", "RC_")):
        return "kalman"

    if lname.startswith(("risk_", "path_", "smc_", "breakout_", "liquidity_")):
        return "specialists"

    if name in {
        "resistance_scalar",
        "breakout_scalar_resistance",
        "support_scalar",
        "breakout_scalar_support",
        "breakout_success_prob",
        "breakout_high_conf_signal",
    }:
        return "specialists"

    if any(
        tok in lname
        for tok in (
            "regime",
            "hmm",
            "volatility",
            "atr",
            "meta_vol",
            "volume",
            "amihud",
            "kyle",
            "rolls",
            "spread",
            "parkinson",
        )
    ):
        return "vol_regime"

    if any(
        tok in lname
        for tok in (
            "rsi",
            "bb_distance",
            "bollinger",
            "zscore",
            "sma_distance",
            "ema_distance",
            "price_vs_sma",
            "meanrev",
            "reversion",
        )
    ):
        return "mr"

    if any(
        tok in lname
        for tok in (
            "momentum",
            "roc",
            "trend",
            "macd",
            "kaufman",
            "slope",
            "acceleration",
            "strength",
            "persistence",
        )
    ):
        return "trend"

    return "other"


def _safe_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        mask = ~(np.isnan(y) | np.isnan(p))
        if int(mask.sum()) < 10:
            return None
        y_clean = y[mask]
        p_clean = p[mask]
        if np.unique(y_clean).size < 2:
            return None
        return float(roc_auc_score(y_clean, p_clean))
    except Exception:
        return None


def _safe_pearson_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        if int(mask.sum()) < 3:
            return None
        aa = aa[mask]
        bb = bb[mask]
        if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
            return 0.0
        c = float(np.corrcoef(aa, bb)[0, 1])
        return c if np.isfinite(c) else None
    except Exception:
        return None


def _safe_spearman_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        mask = np.isfinite(aa) & np.isfinite(bb)
        if int(mask.sum()) < 3:
            return None
        aa = aa[mask]
        bb = bb[mask]
        if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
            return 0.0
        ra = pd.Series(aa).rank(method="average").to_numpy(dtype=float)
        rb = pd.Series(bb).rank(method="average").to_numpy(dtype=float)
        c = float(np.corrcoef(ra, rb)[0, 1])
        return c if np.isfinite(c) else None
    except Exception:
        return None


def _split_time_bins(index: pd.Index, n_bins: int) -> List[np.ndarray]:
    n = int(len(index))
    if n <= 0:
        return []
    n_bins = int(max(1, n_bins))
    if n_bins == 1:
        return [np.arange(n, dtype=int)]
    return [np.asarray(a, dtype=int) for a in np.array_split(np.arange(n, dtype=int), n_bins) if len(a) > 0]


def _compute_ic_diagnostics(
    *,
    X: pd.DataFrame,
    y_bin: pd.Series,
    y_cont: pd.Series,
    feature_groups: Optional[pd.Series] = None,
    n_bins: int = 8,
    max_samples: int = 50000,
    min_bin_samples: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if X is None or getattr(X, "empty", True):
        return pd.DataFrame(), pd.DataFrame(), {"enabled": True, "reason": "empty_X"}

    idx = X.index
    if not isinstance(y_bin, pd.Series):
        y_bin = pd.Series(y_bin, index=idx)
    if not isinstance(y_cont, pd.Series):
        y_cont = pd.Series(y_cont, index=idx)

    y_bin = y_bin.reindex(idx)
    y_cont = y_cont.reindex(idx)

    base_mask = y_bin.notna() & y_cont.notna()
    if int(base_mask.sum()) < 200:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {"enabled": True, "reason": "insufficient_samples", "n_valid": int(base_mask.sum())},
        )

    Xv = X.loc[base_mask]
    yb = y_bin.loc[base_mask]
    yc = y_cont.loc[base_mask]

    n = int(len(Xv))
    if max_samples is not None and int(max_samples) > 0 and n > int(max_samples):
        sel = np.linspace(0, n - 1, int(max_samples)).astype(int)
        Xv = Xv.iloc[sel]
        yb = yb.iloc[sel]
        yc = yc.iloc[sel]

    bins = _split_time_bins(Xv.index, int(n_bins))
    if not bins:
        bins = [np.arange(int(len(Xv)), dtype=int)]

    rows = []
    bin_rows = []
    for col in list(Xv.columns):
        try:
            x = pd.to_numeric(Xv[col], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue

        y_cont_arr = pd.to_numeric(yc, errors="coerce").to_numpy(dtype=float)
        y_bin_arr = pd.to_numeric(yb, errors="coerce").to_numpy(dtype=float)

        ic_ret_s = _safe_spearman_corr(x, y_cont_arr)
        ic_ret_p = _safe_pearson_corr(x, y_cont_arr)
        ic_lbl_s = _safe_spearman_corr(x, y_bin_arr)
        ic_lbl_p = _safe_pearson_corr(x, y_bin_arr)

        ic_ret_bins = []
        ic_lbl_bins = []
        for bi, pos in enumerate(bins):
            if int(len(pos)) < int(min_bin_samples):
                continue
            xs = x[pos]
            yr = y_cont_arr[pos]
            yk = y_bin_arr[pos]
            ics_r = _safe_spearman_corr(xs, yr)
            icp_r = _safe_pearson_corr(xs, yr)
            ics_k = _safe_spearman_corr(xs, yk)
            icp_k = _safe_pearson_corr(xs, yk)

            if ics_r is not None and np.isfinite(ics_r):
                ic_ret_bins.append(float(ics_r))
            if ics_k is not None and np.isfinite(ics_k):
                ic_lbl_bins.append(float(ics_k))

            bin_rows.append(
                {
                    "feature_name": str(col),
                    "feature_group": str(feature_groups.get(col)) if isinstance(feature_groups, pd.Series) and col in feature_groups.index else None,
                    "bin_idx": int(bi),
                    "n_samples": int(len(pos)),
                    "ic_spearman_return": float(ics_r) if ics_r is not None and np.isfinite(ics_r) else None,
                    "ic_pearson_return": float(icp_r) if icp_r is not None and np.isfinite(icp_r) else None,
                    "ic_spearman_label": float(ics_k) if ics_k is not None and np.isfinite(ics_k) else None,
                    "ic_pearson_label": float(icp_k) if icp_k is not None and np.isfinite(icp_k) else None,
                }
            )

        def _summ(vs: List[float], overall: Optional[float]) -> Dict[str, Any]:
            arr = np.asarray(vs, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return {
                    "ic_bins": 0,
                    "ic_mean_bins": None,
                    "ic_std_bins": None,
                    "ic_ir_bins": None,
                    "ic_sign_consistency": None,
                    "ic_min_bins": None,
                    "ic_max_bins": None,
                }
            mu = float(np.mean(arr))
            sd = float(np.std(arr)) if arr.size > 1 else 0.0
            denom = sd if sd > 1e-12 else 1e-12
            ir = float(mu / denom)
            ref = float(overall) if (overall is not None and np.isfinite(overall)) else mu
            sign_cons = float(np.mean(np.sign(arr) == np.sign(ref)))
            return {
                "ic_bins": int(arr.size),
                "ic_mean_bins": mu,
                "ic_std_bins": sd,
                "ic_ir_bins": ir,
                "ic_sign_consistency": sign_cons,
                "ic_min_bins": float(np.min(arr)),
                "ic_max_bins": float(np.max(arr)),
            }

        ret_stats = _summ(ic_ret_bins, ic_ret_s)
        lbl_stats = _summ(ic_lbl_bins, ic_lbl_s)

        rows.append(
            {
                "feature_name": str(col),
                "feature_group": str(feature_groups.get(col)) if isinstance(feature_groups, pd.Series) and col in feature_groups.index else None,
                "n_samples_used": int(len(Xv)),
                "n_bins": int(n_bins),
                "ic_spearman_return": float(ic_ret_s) if ic_ret_s is not None and np.isfinite(ic_ret_s) else None,
                "ic_pearson_return": float(ic_ret_p) if ic_ret_p is not None and np.isfinite(ic_ret_p) else None,
                "ic_spearman_label": float(ic_lbl_s) if ic_lbl_s is not None and np.isfinite(ic_lbl_s) else None,
                "ic_pearson_label": float(ic_lbl_p) if ic_lbl_p is not None and np.isfinite(ic_lbl_p) else None,
                "ic_bins_return": ret_stats["ic_bins"],
                "ic_mean_bins_return": ret_stats["ic_mean_bins"],
                "ic_std_bins_return": ret_stats["ic_std_bins"],
                "ic_ir_bins_return": ret_stats["ic_ir_bins"],
                "ic_sign_consistency_return": ret_stats["ic_sign_consistency"],
                "ic_min_bins_return": ret_stats["ic_min_bins"],
                "ic_max_bins_return": ret_stats["ic_max_bins"],
                "ic_bins_label": lbl_stats["ic_bins"],
                "ic_mean_bins_label": lbl_stats["ic_mean_bins"],
                "ic_std_bins_label": lbl_stats["ic_std_bins"],
                "ic_ir_bins_label": lbl_stats["ic_ir_bins"],
                "ic_sign_consistency_label": lbl_stats["ic_sign_consistency"],
                "ic_min_bins_label": lbl_stats["ic_min_bins"],
                "ic_max_bins_label": lbl_stats["ic_max_bins"],
            }
        )

    out_table = pd.DataFrame(rows)
    out_bins = pd.DataFrame(bin_rows)
    summary = {
        "enabled": True,
        "n_features": int(out_table.shape[0]) if out_table is not None else 0,
        "n_samples_used": int(len(Xv)),
        "n_bins": int(n_bins),
        "min_bin_samples": int(min_bin_samples),
    }
    return out_table, out_bins, summary


def _train_weighted_lgbm_oof(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    n_splits: int,
    params: Dict[str, Any],
) -> Tuple[pd.Series, List[Any]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_probs = np.full(len(y), np.nan, dtype=float)
    models: List[Any] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        y_train = y.iloc[train_idx]
        if np.unique(y_train.dropna()).size < 2:
            continue
        model = lgb.LGBMClassifier(**params)
        try:
            model.fit(X.iloc[train_idx], y_train, sample_weight=sample_weights[train_idx])
            oof_probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
            models.append(model)
        except Exception:
            continue

    return pd.Series(oof_probs, index=X.index), models


def _compute_threshold_trade_stats(
    realized_returns: pd.Series,
    prob_series: pd.Series,
    threshold: float,
    date_range_days: Optional[float],
) -> Dict[str, Any]:
    mask = (~realized_returns.isna()) & (~prob_series.isna()) & (prob_series >= float(threshold))
    n_trades = int(mask.sum())
    mean_ret = float(realized_returns.loc[mask].mean()) if n_trades > 0 else float("nan")
    sum_ret = float(realized_returns.loc[mask].sum()) if n_trades > 0 else 0.0
    trades_per_day = float(n_trades / date_range_days) if (date_range_days is not None and date_range_days > 0) else float("nan")
    return {
        "n_trades": n_trades,
        "trades_per_day": trades_per_day,
        "mean_return": mean_ret,
        "sum_return": sum_ret,
    }


def _find_latest_outcomes_file(outcomes_dir: Path, pattern: str) -> Optional[Path]:
    try:
        paths = sorted(outcomes_dir.glob(pattern))
        return paths[-1] if paths else None
    except Exception:
        return None


def _metric_param_corr_matrix(
    df: pd.DataFrame,
    metric_cols: List[str],
    param_cols: List[str],
    method: str = "spearman",
) -> Optional[pd.DataFrame]:
    metrics_present = [c for c in metric_cols if c in df.columns]
    params_present = [c for c in param_cols if c in df.columns]
    if not metrics_present or not params_present:
        return None

    sub = df[metrics_present + params_present].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    good_cols: List[str] = []
    for c in sub.columns:
        arr = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if int(arr.size) < 3:
            continue
        if int(np.unique(arr).size) < 2:
            continue
        good_cols.append(c)

    if not good_cols:
        return None
    sub = sub[good_cols]

    metrics_present = [c for c in metrics_present if c in sub.columns]
    params_present = [c for c in params_present if c in sub.columns]
    if not metrics_present or not params_present:
        return None

    try:
        corr = sub.corr(method=method)
    except Exception:
        corr = sub.corr()

    try:
        return corr.loc[metrics_present, params_present]
    except Exception:
        return None


def _save_corr_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    if matrix is None or matrix.empty:
        return

    if plt is None or sns is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(8.0, 0.55 * float(len(matrix.columns)) + 3.0)
    fig_h = max(4.0, 0.45 * float(len(matrix.index)) + 2.0)

    try:
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            matrix,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            linewidths=0.2,
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    finally:
        try:
            plt.close()
        except Exception:
            pass


def _generate_hpo_correlation_artifacts(
    outcomes_dir: Path,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    timestamp: str,
    method: str = "spearman",
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"method": method}
    if not outcomes_dir.exists():
        return results

    stage0_path = _find_latest_outcomes_file(
        outcomes_dir, f"hpo_stage0_kalman_trials_{symbol}_{timeframe}_*.csv"
    )
    if stage0_path is not None:
        try:
            df0 = pd.read_csv(stage0_path)
            matrix0 = _metric_param_corr_matrix(
                df=df0,
                metric_cols=["score", "loss", "smooth", "track", "amp", "amp_ratio"],
                param_cols=["kalman_Q", "kalman_R"],
                method=method,
            )
            if matrix0 is not None and not matrix0.empty:
                out_csv = outcomes_dir / (
                    f"weighted_meta_hpo_corr_stage0_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                )
                out_png = outcomes_dir / (
                    f"weighted_meta_hpo_corr_stage0_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.png"
                )
                matrix0.to_csv(out_csv)
                _save_corr_heatmap(matrix0, out_png, title="Stage 0 (Kalman): metrics vs params")
                results["stage0"] = {
                    "trials_csv": str(stage0_path),
                    "n_rows": int(len(df0)),
                    "corr_csv": str(out_csv),
                    "heatmap_png": str(out_png),
                    "metrics": list(matrix0.index),
                    "params": list(matrix0.columns),
                }
        except Exception:
            pass

    layer1_path = _find_latest_outcomes_file(
        outcomes_dir, f"hpo_layer1_trials_{symbol}_{timeframe}_*.csv"
    )
    if layer1_path is not None:
        try:
            df1 = pd.read_csv(layer1_path)
            param_cols_1 = [c for c in df1.columns if str(c).startswith("param_")]
            matrix1 = _metric_param_corr_matrix(
                df=df1,
                metric_cols=[
                    "score",
                    "weights_entropy_norm",
                    "weights_mean",
                    "weights_min",
                    "weights_max",
                    "mas",
                    "wes",
                    "nwp",
                    "uop_penalty",
                    "vdp_penalty",
                    "n_events",
                ],
                param_cols=param_cols_1,
                method=method,
            )
            if matrix1 is not None and not matrix1.empty:
                out_csv = outcomes_dir / (
                    f"weighted_meta_hpo_corr_layer1_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                )
                out_png = outcomes_dir / (
                    f"weighted_meta_hpo_corr_layer1_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.png"
                )
                matrix1.to_csv(out_csv)
                _save_corr_heatmap(matrix1, out_png, title="Layer 1 (Weighting): metrics vs params")
                results["layer1"] = {
                    "trials_csv": str(layer1_path),
                    "n_rows": int(len(df1)),
                    "corr_csv": str(out_csv),
                    "heatmap_png": str(out_png),
                    "metrics": list(matrix1.index),
                    "params": list(matrix1.columns),
                }
        except Exception:
            pass

    layer2_path = _find_latest_outcomes_file(
        outcomes_dir, f"hpo_layer2_trials_{symbol}_{timeframe}_*.csv"
    )
    if layer2_path is not None:
        try:
            df2 = pd.read_csv(layer2_path)
            param_cols_2 = [c for c in df2.columns if str(c).startswith("param_")]
            matrix2 = _metric_param_corr_matrix(
                df=df2,
                metric_cols=[
                    "utility",
                    "auc",
                    "trades_per_day",
                    "sharpe_mean",
                    "sharpe_std",
                    "base_score",
                    "phi_auc",
                    "phi_density",
                    "modifier",
                ],
                param_cols=param_cols_2,
                method=method,
            )
            if matrix2 is not None and not matrix2.empty:
                out_csv = outcomes_dir / (
                    f"weighted_meta_hpo_corr_layer2_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                )
                out_png = outcomes_dir / (
                    f"weighted_meta_hpo_corr_layer2_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.png"
                )
                matrix2.to_csv(out_csv)
                _save_corr_heatmap(matrix2, out_png, title="Layer 2 (Trading): metrics vs params")
                results["layer2"] = {
                    "trials_csv": str(layer2_path),
                    "n_rows": int(len(df2)),
                    "corr_csv": str(out_csv),
                    "heatmap_png": str(out_png),
                    "metrics": list(matrix2.index),
                    "params": list(matrix2.columns),
                }
        except Exception:
            pass

    layer3_path = _find_latest_outcomes_file(
        outcomes_dir, f"hpo_layer3_trials_{symbol}_{timeframe}_*.csv"
    )
    if layer3_path is not None:
        try:
            df3 = pd.read_csv(layer3_path)
            param_cols_3 = [c for c in df3.columns if str(c).startswith("param_")]
            matrix3 = _metric_param_corr_matrix(
                df=df3,
                metric_cols=[
                    "utility",
                    "mean_auc",
                    "trades_per_day",
                    "sharpe_mean",
                    "sharpe_std",
                    "base_score",
                    "phi_auc",
                    "phi_density",
                    "modifier",
                ],
                param_cols=param_cols_3,
                method=method,
            )
            if matrix3 is not None and not matrix3.empty:
                out_csv = outcomes_dir / (
                    f"weighted_meta_hpo_corr_layer3_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                )
                out_png = outcomes_dir / (
                    f"weighted_meta_hpo_corr_layer3_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.png"
                )
                matrix3.to_csv(out_csv)
                _save_corr_heatmap(matrix3, out_png, title="Layer 3 (Model): metrics vs params")
                results["layer3"] = {
                    "trials_csv": str(layer3_path),
                    "n_rows": int(len(df3)),
                    "corr_csv": str(out_csv),
                    "heatmap_png": str(out_png),
                    "metrics": list(matrix3.index),
                    "params": list(matrix3.columns),
                }
        except Exception:
            pass

    return results


def _load_weighting_params_from_hpo(
    symbol: str,
    timeframe: str,
    direction: str = "long",
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Path]]:
    """Load weighting and Kalman parameters from the multi-stage HPO output.
    
    Searches for files matching:
        outcomes/hpo_multi_stage_best_params_{symbol}_*.json
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        direction: Trading direction (long/short)
        
    Returns:
        Tuple of (weighting_params, kalman_params, file_path)
    """
    outcomes_dir = Path("outcomes")
    if not outcomes_dir.exists():
        return DEFAULT_WEIGHTING_PARAMS.copy(), DEFAULT_KALMAN_PARAMS.copy(), None
    
    # Look for multi-stage HPO output first
    pattern = f"hpo_multi_stage_best_params_{symbol}_*.json"
    candidates = sorted(outcomes_dir.glob(pattern))
    
    # Also check for standard HPO output
    if not candidates:
        pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json"
        candidates = sorted(outcomes_dir.glob(pattern))
    
    if not candidates:
        return DEFAULT_WEIGHTING_PARAMS.copy(), DEFAULT_KALMAN_PARAMS.copy(), None
    
    latest = candidates[-1]
    try:
        with open(latest, "r") as f:
            hpo_cfg = json.load(f)
        
        # Extract weighting params from the full config
        weighting_params = {}
        weighting_keys = list(DEFAULT_WEIGHTING_PARAMS.keys())
        
        for key in weighting_keys:
            if key in hpo_cfg:
                weighting_params[key] = float(hpo_cfg[key])
        
        # Extract Kalman params
        kalman_params = DEFAULT_KALMAN_PARAMS.copy()
        if 'kalman_Q' in hpo_cfg:
            kalman_params['kalman_Q'] = float(hpo_cfg['kalman_Q'])
        if 'kalman_R' in hpo_cfg:
            kalman_params['kalman_R'] = float(hpo_cfg['kalman_R'])
        
        # If we found at least some params, use them (fill missing with defaults)
        merged_weighting = DEFAULT_WEIGHTING_PARAMS.copy()
        if weighting_params:
            merged_weighting.update(weighting_params)
        
        tprint_info(f"📊 Loaded params from {latest}")
        tprint_info(f"   Kalman: Q={kalman_params['kalman_Q']:.2e}, R={kalman_params['kalman_R']:.2e}")
        return merged_weighting, kalman_params, latest
        
    except Exception as e:
        tprint_warning(f"⚠️ Failed to load params from {latest}: {e}")
    
    return DEFAULT_WEIGHTING_PARAMS.copy(), DEFAULT_KALMAN_PARAMS.copy(), None


def compute_sample_weights_for_events(
    realized_returns: pd.Series,
    market_data: pd.DataFrame,
    weighting_params: Dict[str, Any],
    horizon: int = 12,
) -> np.ndarray:
    """Compute sample weights for labeled events using the weighted pipeline.
    
    Args:
        realized_returns: Series of realized returns (only labeled events)
        market_data: Full market data for computing bar-level features
        weighting_params: Parameters for generate_weights_per_label
        horizon: Lookahead horizon for consistency calculation
        
    Returns:
        Array of sample weights aligned with realized_returns
    """
    # Filter to valid (labeled) events
    valid_mask = ~realized_returns.isna()
    valid_returns = realized_returns[valid_mask]
    
    if len(valid_returns) < 10:
        return np.ones(len(realized_returns))
    
    t_events = valid_returns.index
    
    # Compute bar-level features
    close_series = market_data["close"]
    returns_series = close_series.pct_change().fillna(0.0)
    
    # Pre-calculate heavy features
    full_consistency = compute_horizon_consistency(close_series, horizon=horizon)
    full_volatility = returns_series.rolling(20).std().fillna(0.0)
    
    # Create t_events Series for uniqueness (with estimated end times)
    try:
        t_events_series = pd.Series(
            index=t_events,
            data=t_events + pd.Timedelta(minutes=15 * horizon)  # Assuming 15m bars
        )
    except Exception:
        t_events_series = pd.Series(index=t_events, data=t_events)
    
    # Align features to event timestamps
    consistency_aligned = full_consistency.reindex(t_events).fillna(0.0).values
    volatility_aligned = full_volatility.reindex(t_events).fillna(0.0).values
    uniqueness_aligned = compute_uniqueness(
        t_events_series,
        events_index=t_events_series.index,
        market_index=market_data.index
    )
    
    if isinstance(uniqueness_aligned, pd.Series):
        uniqueness_aligned = uniqueness_aligned.values
    
    # Generate weights
    weighting_params_local = dict(weighting_params) if isinstance(weighting_params, dict) else {}
    if "transaction_cost" not in weighting_params_local:
        weighting_params_local["transaction_cost"] = 0.0
    weights = generate_weights_per_label(
        returns=valid_returns.values,
        t_events=t_events,
        close_series=None,
        consistency_scores=consistency_aligned,
        uniqueness_scores=uniqueness_aligned,
        vol_proxy=volatility_aligned,
        **weighting_params_local
    )
    
    # Map back to full index (non-labeled events get weight 0)
    full_weights = np.zeros(len(realized_returns))
    full_weights[valid_mask.values] = weights
    
    return full_weights


def train_weighted_bagged_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray,
    n_splits: int = 5,
    n_bags: int = 10,
    base_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Any]]:
    """Train bagged LightGBM with sample weighting.
    
    Args:
        X: Feature matrix
        y: Binary labels
        sample_weights: Sample weights from generate_weights_per_label
        n_splits: Number of CV splits
        n_bags: Number of bagged estimators
        base_params: LightGBM parameters
        
    Returns:
        Tuple of (OOF predictions DataFrame, trained models list)
    """
    tprint_info("🔧 train_weighted_bagged_lgbm() called")
    tprint_info(f"   X_shape={X.shape}, n_splits={n_splits}, n_bags={n_bags}")
    
    if base_params is None:
        base_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': -1,
            'verbose': -1,
            'random_state': 42,
        }
    
    # Prepare output
    oof_probs = np.full(len(y), np.nan, dtype=float)
    models = []
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights[train_idx]
        
        # Skip if insufficient class variety
        if len(np.unique(y_train.dropna())) < 2:
            continue
        
        # Train bagged ensemble for this fold
        fold_probs = []
        for bag_idx in range(n_bags):
            # Bootstrap sample
            rng = np.random.RandomState(42 + fold_idx * 100 + bag_idx)
            n_train = len(X_train)
            boot_idx = rng.choice(n_train, size=n_train, replace=True)
            
            # Weighted bootstrap
            X_boot = X_train.iloc[boot_idx]
            y_boot = y_train.iloc[boot_idx]
            w_boot = w_train[boot_idx]
            
            # Train model
            model = lgb.LGBMClassifier(**base_params)
            try:
                model.fit(X_boot, y_boot, sample_weight=w_boot)
                probs = model.predict_proba(X_val)[:, 1]
                fold_probs.append(probs)
                models.append(model)
            except Exception as e:
                tprint_warning(f"⚠️ Bag {bag_idx} fold {fold_idx} failed: {e}")
                continue
        
        # Average predictions across bags
        if fold_probs:
            oof_probs[val_idx] = np.mean(fold_probs, axis=0)
    
    # Build output DataFrame
    oof_df = pd.DataFrame({
        'lgbm_bag_mean': oof_probs,
        'lgbm_bag_lower': oof_probs * 0.9,  # Approximate lower bound
    }, index=X.index)
    
    return oof_df, models


class WeightedMetaLabelingStep(FeatureGenerationMetaLabelingStep):
    """Production meta-labeling step with sample weighting from HPO.
    
    This step extends FeatureGenerationMetaLabelingStep by:
    1. Loading optimal weighting and Kalman parameters from HPO output
    2. Computing sample weights using generate_weights_per_label
    3. Generating Kalman-based features (KF_Close, KF_High, KF_Low: Filtered OHLC using causal Kalman filter)
    4. Training weighted bagged LightGBM models
    5. Using calibration-adjusted position sizing
    
    Kalman Features Added:
    - KF_Close, KF_High, KF_Low: Filtered OHLC using causal Kalman filter
    - KF_Velocity, KF_Acceleration: 1st/2nd derivatives of filtered close
    - KF_Slope: Rolling slope of filtered close
    - KF_P: Error covariance (uncertainty)
    - KF_RSI: RSI computed on filtered close
    - KF_BB_Distance: Distance from Kalman Bollinger Band
    - KF_Volume, KF_LogVolume_Slope, KF_Volume_Zscore, KF_Volume_Ratio, KF_Volume_P
    
    Config keys (in addition to base class):
    - use_hpo_weighting: bool - Whether to use HPO weighting params (default: True)
    - weighting_params: dict - Override weighting params (optional)
    - kalman_params: dict - Override Kalman Q/R params (optional)
    - weight_optimization_enabled: bool - Run Layer 1 optimization if HPO not found
    """
    
    def __init__(self, step_name: str = "weighted_meta_labeling") -> None:
        super().__init__(step_name)
        self.weighting_params = DEFAULT_WEIGHTING_PARAMS.copy()
        self.kalman_params = DEFAULT_KALMAN_PARAMS.copy()
        self.weighting_source = None
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weighted meta-labeling pipeline.
        
        Steps:
        1. Load market data and generate primary signals
        2. Load or compute weighting parameters
        3. Generate labels with triple-barrier method
        4. Compute sample weights for training
        5. Train weighted bagged LGBM models
        6. Generate meta-probability outputs
        7. Save labeled data artifacts
        """
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")

        try:
            self.set_context(
                symbol=str(symbol),
                exchange=str(exchange),
                timeframe=str(timeframe),
                direction=str(direction),
                model=str(config.get("model", "analyst")),
            )
        except Exception:
            pass
        
        tprint_info(
            f"🚀 Starting Weighted Meta-Labeling for {symbol}/{exchange} [{timeframe}] ({direction})"
        )
        
        # ------------------------------------------------------------------
        # 1. Load weighting and Kalman parameters from HPO
        # ------------------------------------------------------------------
        use_hpo_weighting = config.get("use_hpo_weighting", True)
        
        if use_hpo_weighting:
            self.weighting_params, self.kalman_params, hpo_path = _load_weighting_params_from_hpo(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
            )
            if hpo_path:
                self.weighting_source = str(hpo_path)
                tprint_success(f"✅ Using params from: {hpo_path}")
                # Load consensus_threshold from HPO params if available
                consensus_threshold = self.weighting_params.get('consensus_threshold', 0.6)
            else:
                tprint_warning("⚠️ No HPO params found, using defaults")
                self.weighting_source = "defaults"
                consensus_threshold = 0.6
        else:
            self.weighting_source = "config"
            if "weighting_params" in config:
                self.weighting_params.update(config["weighting_params"])
            if "kalman_params" in config:
                self.kalman_params.update(config["kalman_params"])
            consensus_threshold = config.get("consensus_threshold", 0.6)

        tprint_info(f"   Weighting params: {self.weighting_params}")
        tprint_info(f"   Kalman params: Q={self.kalman_params['kalman_Q']:.2e}, R={self.kalman_params['kalman_R']:.2e}")
        tprint_info(f"   Consensus threshold: {consensus_threshold}")
        
        # ------------------------------------------------------------------
        # 2. Load market data (delegate to base class)
        # ------------------------------------------------------------------
        pipeline_state: Dict[str, Any] = {}
        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=True,
            skip_artifacts=True,
        )
        
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            msg = "❌ No market data available for weighted meta-labeling"
            tprint_error(msg)
            return {"success": False, "error": msg, "metrics": {}, "artifacts": {}}
        
        tprint_info(f"   Loaded {len(market_data)} bars from {source}")
        
        # ------------------------------------------------------------------
        # 3. Generate primary signals
        # ------------------------------------------------------------------
        try:
            primary_signals = generate_primary_signals(market_data.copy())
            signals_mask = (primary_signals != 0)
            try:
                n_signals = int(np.asarray(signals_mask).sum())
            except Exception:
                n_signals = int(signals_mask.sum().sum() if hasattr(signals_mask, "sum") else 0)
            tprint_info(f"   Generated {n_signals} primary signals")
        except Exception as e:
            tprint_error(f"❌ Primary signal generation failed: {e}")
            return {"success": False, "error": str(e), "metrics": {}, "artifacts": {}}
        
        # ------------------------------------------------------------------
        # 4. Attach regimes (optional)
        # ------------------------------------------------------------------
        try:
            try:
                market_data = attach_rolling_hmm_regimes_to_market_data(
                    step=self,
                    market_data=market_data,
                    config=config,
                )
            except TypeError:
                # Backward compatibility if signature differs
                market_data = attach_rolling_hmm_regimes_to_market_data(
                    self,
                    market_data,
                    config,
                )
            tprint_info("   Regimes attached successfully")
        except Exception as e:
            tprint_warning(f"⚠️ Regime attachment failed: {e}")

        # ------------------------------------------------------------------
        # 4b. Compute per-regime label diagnostics (if regimes available)
        # ------------------------------------------------------------------
        regime_columns = [col for col in market_data.columns if 'regime' in col.lower()]
        if regime_columns:
            tprint_info("   Computing per-regime label diagnostics...")

            # Create volatility buckets for additional stratification
            if 'close' in market_data.columns:
                log_ret = np.log(market_data["close"]).diff()

                # Multiple volatility measures
                vol_5 = log_ret.rolling(5).std()
                vol_20 = log_ret.rolling(20).std()
                vol_50 = log_ret.rolling(50).std()

                # Volatility-normalized momentum (returns scaled by volatility)
                for period in [1, 3, 5, 10, 20]:
                    momentum = market_data["close"].pct_change(period)
                    # Normalize by different volatility horizons
                    market_data[f'momentum_{period}d_vol_norm_5'] = momentum / (vol_5 + 1e-8)
                    market_data[f'momentum_{period}d_vol_norm_20'] = momentum / (vol_20 + 1e-8)
                    market_data[f'momentum_{period}d_vol_norm_50'] = momentum / (vol_50 + 1e-8)

                # Mean reversion features (deviation from moving averages)
                for period in [10, 20, 50]:
                    ma = market_data["close"].rolling(period).mean()
                    std = market_data["close"].rolling(period).std()
                    z_score = (market_data["close"] - ma) / (std + 1e-8)

                    # Volatility-normalized z-score
                    vol_norm_factor = vol_20 / vol_20.rolling(period).mean()
                    market_data[f'mean_revert_z_{period}_vol_norm'] = z_score / (vol_norm_factor + 1e-8)

                    # Bollinger band position (volatility-adjusted)
                    bb_upper = ma + 2 * std
                    bb_lower = ma - 2 * std
                    bb_position = (market_data["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    market_data[f'bb_position_{period}_vol_adj'] = bb_position * vol_norm_factor

                tprint_info(f"   ✅ Added volatility-normalized momentum and mean-reversion features")

            # Adaptive lookback features based on current volatility
            try:
                # Calculate adaptive lookback periods based on volatility regime
                vol_regime = pd.qcut(vol_20, q=3, labels=['low_vol', 'med_vol', 'high_vol'])

                # Adaptive RSI (shorter lookback in high vol, longer in low vol)
                adaptive_rsi_periods = vol_regime.map({'low_vol': 21, 'med_vol': 14, 'high_vol': 9})
                adaptive_rsi = pd.Series(index=market_data.index, dtype=float)

                # Adaptive MACD (shorter fast/slow in high vol)
                adaptive_fast_periods = vol_regime.map({'low_vol': 26, 'med_vol': 12, 'high_vol': 8})
                adaptive_slow_periods = vol_regime.map({'low_vol': 52, 'med_vol': 26, 'high_vol': 17})

                for i in range(len(market_data)):
                    if i < 50:  # Need some history
                        continue

                    # Adaptive RSI
                    period = adaptive_rsi_periods.iloc[i]
                    if i >= period:
                        prices = market_data["close"].iloc[i-period:i+1]
                        gains = prices.diff().clip(lower=0)
                        losses = -prices.diff().clip(upper=0)
                        avg_gain = gains.rolling(period, min_periods=1).mean().iloc[-1]
                        avg_loss = losses.rolling(period, min_periods=1).mean().iloc[-1]
                        rs = avg_gain / (avg_loss + 1e-8)
                        adaptive_rsi.iloc[i] = 100 - (100 / (1 + rs))

                    # Adaptive MACD
                    fast_period = adaptive_fast_periods.iloc[i]
                    slow_period = adaptive_slow_periods.iloc[i]
                    if i >= slow_period:
                        fast_ema = market_data["close"].iloc[i-fast_period:i+1].ewm(span=fast_period, adjust=False).mean().iloc[-1]
                        slow_ema = market_data["close"].iloc[i-slow_period:i+1].ewm(span=slow_period, adjust=False).mean().iloc[-1]
                        market_data.loc[market_data.index[i], 'adaptive_macd'] = fast_ema - slow_ema

                # Only keep RSI where calculated
                market_data['adaptive_rsi'] = adaptive_rsi

                tprint_info(f"   ✅ Added adaptive lookback features (RSI, MACD)")

            except Exception as e:
                tprint_warning(f"   ⚠️ Adaptive lookback features failed: {e}")

            # Trend persistence features (consecutive bars in same direction)
            try:
                # Calculate consecutive up/down bars
                price_changes = market_data["close"].diff()
                bar_direction = (price_changes > 0).astype(int)  # 1 for up, 0 for down

                # Count consecutive bars in same direction
                consecutive_count = pd.Series(index=market_data.index, dtype=int)
                current_streak = 0
                last_direction = None

                for i, dir_val in enumerate(bar_direction):
                    if pd.isna(dir_val):
                        consecutive_count.iloc[i] = 0
                        continue

                    if last_direction is None or dir_val != last_direction:
                        current_streak = 1
                    else:
                        current_streak += 1

                    consecutive_count.iloc[i] = current_streak
                    last_direction = dir_val

                market_data['trend_persistence_up'] = consecutive_count * (bar_direction == 1)
                market_data['trend_persistence_down'] = consecutive_count * (bar_direction == 0)
                market_data['trend_persistence_total'] = consecutive_count

                # Additional trend features
                market_data['trend_strength_5'] = abs(market_data["close"].pct_change(5))
                market_data['trend_strength_10'] = abs(market_data["close"].pct_change(10))
                market_data['trend_acceleration'] = market_data["close"].pct_change(1) - market_data["close"].pct_change(1).shift(1)

                tprint_info(f"   ✅ Added trend persistence features")

            except Exception as e:
                tprint_warning(f"   ⚠️ Trend persistence features failed: {e}")

            # Store regime diagnostics for later use in reports
            self.regime_diagnostics = {}
            for regime_col in regime_columns + ['volatility_bucket']:
                if regime_col not in market_data.columns:
                    continue

                valid_regime_mask = ~market_data[regime_col].isna()
                if valid_regime_mask.sum() == 0:
                    continue

                regime_values = market_data.loc[valid_regime_mask, regime_col].unique()
                self.regime_diagnostics[regime_col] = {}

                for regime_val in regime_values:
                    regime_mask = (market_data[regime_col] == regime_val) & valid_regime_mask
                    regime_signals = primary_signals[regime_mask]

                    try:
                        n_regime_signals = int(np.asarray(regime_signals != 0).sum())
                    except Exception:
                        n_regime_signals = 0

                    if n_regime_signals == 0:
                        continue
                    self.regime_diagnostics[regime_col][str(regime_val)] = {
                        'n_signals': n_regime_signals,
                        'signal_rate': n_regime_signals / len(regime_mask) * 100
                    }

                tprint_info(f"   {regime_col}: {len(self.regime_diagnostics[regime_col])} regimes found")
        else:
            self.regime_diagnostics = {}
            tprint_info("   No regime columns found - skipping regime diagnostics")

        # ------------------------------------------------------------------
        # 5. Generate labels with Kalman multi-triple-barrier system
        # ------------------------------------------------------------------
        # Use Kalman-smoothed multi-triple-barrier with consensus averaging
        # This replaces individual TP/SL/horizon optimization with ensemble approach
        kalman_process_noise = float(config.get("kalman_process_noise", 1e-5))
        kalman_measurement_noise = float(config.get("kalman_measurement_noise", 1e-3))
        vol_window = int(config.get("vol_window", 20))
        tx_cost = float(config.get("transaction_cost", DEFAULT_TRANSACTION_COST))

        # Multi-triple-barrier configuration (ensemble across multiple settings)
        sl_multipliers = config.get("sl_multipliers", [0.6, 0.9, 1.2])  # Modest stops (sigma units)
        if "tp_multipliers" in config:
            tp_multipliers = config.get("tp_multipliers")
        else:
            try:
                tp_multipliers = [2.0 * float(x) for x in list(sl_multipliers)]
            except Exception:
                tp_multipliers = [1.2, 1.8, 2.4]
        horizons = config.get("horizons", [4, 8, 12])  # 1h, 2h, 3h horizons (15min bars)
        economic_floor_multiplier = float(config.get("economic_floor_multiplier", 0.25))
        # consensus_threshold is now loaded from HPO params above

        # Set horizon for compatibility (use max horizon from ensemble)
        horizon = max(horizons) if horizons else 12

        # Balance improvement: Add guardrails for better label entropy
        target_pos_rate_min = config.get("target_pos_rate_min", 0.35)  # 35% minimum positive rate
        target_pos_rate_max = config.get("target_pos_rate_max", 0.45)  # 45% maximum positive rate
        target_entropy_min = config.get("target_entropy_min", 0.4)    # Minimum entropy for balance
        
        # Generate labels using Kalman multi-triple-barrier system
        tprint_info(f"   Applying Kalman multi-triple-barrier labeling:")
        tprint_info(f"     TP multipliers: {tp_multipliers}")
        tprint_info(f"     SL multipliers: {sl_multipliers}")
        tprint_info(f"     Horizons: {horizons}")
        tprint_info(f"     Consensus threshold: {consensus_threshold}")
        tprint_info(f"     Economic floor multiplier: {economic_floor_multiplier}")

        # Apply Kalman multi-triple-barrier labeling
        consensus_labels, sample_weights, detailed_results = kalman_multi_triple_barrier_labels(
            market_data=market_data,
            primary_signals=primary_signals,
            tp_multipliers=tp_multipliers,
            sl_multipliers=sl_multipliers,
            horizons=horizons,
            kalman_process_noise=kalman_process_noise,
            kalman_measurement_noise=kalman_measurement_noise,
            vol_window=vol_window,
            transaction_cost=tx_cost,
            economic_floor_multiplier=economic_floor_multiplier,
            consensus_threshold=consensus_threshold,  # Now from HPO params
            return_detailed_results=True
        )

        # Extract results for compatibility with downstream processing
        # Note: This system generates {-1, 0, 1} labels directly, not separate binary labels
        consensus_full = pd.Series(0.0, index=market_data.index, dtype=float)
        try:
            consensus_full.loc[consensus_labels.index] = consensus_labels.astype(float)
        except Exception:
            pass

        sample_weights_full = pd.Series(0.0, index=market_data.index, name='sample_weights', dtype=float)
        try:
            sample_weights_full.loc[sample_weights.index] = sample_weights.astype(float)
        except Exception:
            pass

        # Binary labels: NaN for abstentions (consensus==0), 1.0 for positive, 0.0 for negative.
        binary_labels = pd.Series(np.nan, index=market_data.index, dtype=float)
        binary_labels.loc[consensus_full == 1.0] = 1.0
        binary_labels.loc[consensus_full == -1.0] = 0.0

        binary_labels_long = pd.Series(np.nan, index=market_data.index, dtype=float)
        binary_labels_short = pd.Series(np.nan, index=market_data.index, dtype=float)

        try:
            signal_direction = primary_signals["consensus"].reindex(market_data.index).fillna(0.0)
        except Exception:
            signal_direction = pd.Series(0.0, index=market_data.index, dtype=float)

        long_mask = signal_direction > 0
        short_mask = signal_direction < 0

        binary_labels_long.loc[long_mask] = binary_labels.loc[long_mask]
        binary_labels_short.loc[short_mask] = binary_labels.loc[short_mask]

        # Aggregate realized returns / exit reasons across triple-barrier configurations.
        # These are required for aleatoric filtering and label-quality metrics.
        tb_results = detailed_results.get('tb_results') if isinstance(detailed_results, dict) else None

        returns_list: List[pd.Series] = []
        exit_reasons_list: List[pd.Series] = []

        if isinstance(tb_results, list) and tb_results:
            for r in tb_results:
                if not isinstance(r, dict):
                    continue
                rr = r.get('returns')
                if isinstance(rr, pd.Series):
                    returns_list.append(rr.reindex(market_data.index))
                er = r.get('exit_reasons')
                if isinstance(er, pd.Series):
                    exit_reasons_list.append(er.reindex(market_data.index))

        if returns_list:
            returns_df = pd.concat(returns_list, axis=1)
            agg = str(config.get('tb_returns_aggregation', 'median')).lower()
            if agg == 'mean':
                realized_returns = returns_df.mean(axis=1).rename('realized_returns')
            elif agg == 'max':
                realized_returns = returns_df.max(axis=1).rename('realized_returns')
            else:
                realized_returns = returns_df.median(axis=1).rename('realized_returns')
        else:
            realized_returns = pd.Series(np.nan, index=binary_labels.index, name='realized_returns')

        if exit_reasons_list:
            exit_df = pd.concat(exit_reasons_list, axis=1)
            try:
                exit_reasons = exit_df.mode(axis=1, dropna=True).iloc[:, 0].rename('exit_reasons')
            except Exception:
                exit_reasons = exit_df.bfill(axis=1).iloc[:, 0].rename('exit_reasons')
        else:
            exit_reasons = pd.Series(pd.NA, index=market_data.index, name='exit_reasons')

        # Placeholders (not returned by multi-triple-barrier yet)
        event_durations = pd.Series(1.0, index=binary_labels.index, name='event_durations')
        mfe_series = pd.Series(np.nan, index=binary_labels.index, name='mfe')
        mae_series = pd.Series(np.nan, index=binary_labels.index, name='mae')

        # Adaptive thresholds are required by downstream meta-feature generation.
        # The Kalman multi-triple-barrier system does not currently return per-event
        # stop/profit thresholds, so we provide conservative constant series.
        base_profit_thr = float(config.get('profit_threshold', DEFAULT_PROFIT_THRESHOLD))
        base_stop_thr = float(config.get('stop_threshold', DEFAULT_STOP_THRESHOLD))
        adaptive_profit = pd.Series(base_profit_thr, index=market_data.index, name='adaptive_profit')
        adaptive_stop = pd.Series(base_stop_thr, index=market_data.index, name='adaptive_stop')

        kalman_price = detailed_results.get('kalman_price') if isinstance(detailed_results, dict) else None
        kalman_volatility = detailed_results.get('kalman_volatility') if isinstance(detailed_results, dict) else None
        if not isinstance(kalman_price, pd.Series):
            kalman_price = pd.Series(np.nan, index=market_data.index, name='kalman_price')
        if not isinstance(kalman_volatility, pd.Series):
            kalman_volatility = pd.Series(np.nan, index=market_data.index, name='kalman_volatility')

        self.kalman_diagnostics = {
            'kalman_price_mean': float(kalman_price.mean()),
            'kalman_volatility_mean': float(kalman_volatility.mean()),
            'kalman_volatility_std': float(kalman_volatility.std()),
            'tp_multipliers_used': tp_multipliers,
            'sl_multipliers_used': sl_multipliers,
            'horizons_used': horizons,
            'consensus_threshold': consensus_threshold,
            'economic_floor_multiplier': economic_floor_multiplier,
            'n_configurations': len(detailed_results.get('configs', []) if isinstance(detailed_results, dict) else []),
            'sample_weight_mean': float(sample_weights_full.mean()),
            'sample_weight_std': float(sample_weights_full.std()),
        }

        tprint_info(
            f"   Kalman diagnostics: volatility σ={kalman_volatility.mean():.6f}, "
            f"{len(detailed_results.get('configs', []) if isinstance(detailed_results, dict) else [])} configurations tested"
        )

        # Diagnostics container for this run
        diagnostics: Dict[str, Any] = {}

        try:
            enable_aleatoric_filter = bool(config.get("enable_aleatoric_filter", True))
            aleatoric_action = str(config.get("aleatoric_action", "abstain")).lower()
            aleatoric_cost_mult = float(config.get("aleatoric_cost_multiplier", 1.0))
            aleatoric_cost_add = float(config.get("aleatoric_cost_add", 0.0))
            aleatoric_downweight_mult = float(config.get("aleatoric_downweight_multiplier", 0.25))

            if enable_aleatoric_filter:
                near_cost_threshold = float(abs(tx_cost) * aleatoric_cost_mult + aleatoric_cost_add)
                near_cost_mask = (
                    (~realized_returns.isna())
                    & (realized_returns.abs() <= near_cost_threshold)
                    & (~binary_labels.isna())
                )

                n_near_cost = int(near_cost_mask.sum())
                diagnostics["aleatoric_filter"] = {
                    "enabled": True,
                    "action": aleatoric_action,
                    "near_cost_threshold": near_cost_threshold,
                    "n_near_cost_events": n_near_cost,
                    "downweight_multiplier": aleatoric_downweight_mult,
                }

                if n_near_cost > 0 and ("abstain" in aleatoric_action or "drop" in aleatoric_action or aleatoric_action == "both"):
                    binary_labels = binary_labels.mask(near_cost_mask)
                    binary_labels_long = binary_labels_long.mask(near_cost_mask)
                    binary_labels_short = binary_labels_short.mask(near_cost_mask)
                    realized_returns = realized_returns.mask(near_cost_mask)
                    exit_reasons = exit_reasons.mask(near_cost_mask)
                    event_durations = event_durations.mask(near_cost_mask)
                    mfe_series = mfe_series.mask(near_cost_mask)
                    mae_series = mae_series.mask(near_cost_mask)

                    tprint_info(
                        f"   Aleatoric abstention: removed {n_near_cost} near-cost events "
                        f"(|ret| <= {near_cost_threshold:.6f})"
                    )

                self._near_cost_mask_for_downweight = near_cost_mask if ("downweight" in aleatoric_action or aleatoric_action == "both") else None
                self._aleatoric_downweight_multiplier = aleatoric_downweight_mult
            else:
                diagnostics["aleatoric_filter"] = {"enabled": False}
                self._near_cost_mask_for_downweight = None
                self._aleatoric_downweight_multiplier = 1.0
        except Exception as e_aleatoric:
            tprint_warning(f"   ⚠️ Aleatoric filter failed: {e_aleatoric}")
            self._near_cost_mask_for_downweight = None
            self._aleatoric_downweight_multiplier = 1.0
        
        labeled_mask = ~binary_labels.isna()
        n_events = int(labeled_mask.sum())
        tprint_info(f"   Labeled events: {n_events}")
        
        if n_events < 100:
            tprint_error(f"❌ Insufficient labeled events ({n_events})")
            return {"success": False, "error": "insufficient_events", "metrics": {}, "artifacts": {}}

        # ------------------------------------------------------------------
        # 5b. Check label balance and quality metrics
        # ------------------------------------------------------------------
        valid_labels = binary_labels[labeled_mask]
        pos_rate = (valid_labels == 1).mean()
        neg_rate = (valid_labels == 0).mean()
        label_entropy = -pos_rate * np.log(pos_rate + 1e-8) - neg_rate * np.log(neg_rate + 1e-8)
        label_entropy = label_entropy / np.log(2)  # Normalize to 0-1 scale

        # Compute SNR and other quality metrics
        valid_returns = realized_returns[labeled_mask]
        pos_returns = valid_returns[valid_labels == 1]
        neg_returns = valid_returns[valid_labels == 0]

        if len(pos_returns) > 0 and len(neg_returns) > 0:
            pos_mean = pos_returns.mean()
            neg_mean = neg_returns.mean()
            pos_std = pos_returns.std()
            neg_std = neg_returns.std()

            if pos_std > 0:
                snr = (pos_mean - neg_mean) / pos_std
                effect_size = (pos_mean - neg_mean) / np.sqrt((pos_std**2 + neg_std**2) / 2)
            else:
                snr = 0
                effect_size = 0

            # Check for mis-signed P&L (returns with wrong sign)
            pos_returns_clean = pos_returns - tx_cost
            neg_returns_clean = neg_returns + tx_cost
            mis_signed = ((pos_returns_clean < 0).sum() + (neg_returns_clean > 0).sum()) / len(valid_returns)

            # Aleatoric uncertainty (fraction of events with |return| < cost)
            aleatoric_uncertainty = ((valid_returns.abs() < tx_cost).sum()) / len(valid_returns)
        else:
            snr = 0
            effect_size = 0
            mis_signed = 1.0
            aleatoric_uncertainty = 1.0

        # Balance guardrails with fail-fast
        balance_issues = []
        if pos_rate < target_pos_rate_min or pos_rate > target_pos_rate_max:
            balance_issues.append(
                f"Positive rate out of range: pos_rate={pos_rate:.3%} (target {target_pos_rate_min:.1%}-{target_pos_rate_max:.1%})"
            )
        if label_entropy < target_entropy_min:
            balance_issues.append(
                f"Low label entropy: entropy={label_entropy:.3f} (min {target_entropy_min:.3f})"
            )
        if snr < 3.0:
            balance_issues.append(
                f"Low SNR: snr={float(snr):.3f} (target >= 3.000)"
            )
        if mis_signed > 0.07:  # >7% mis-signed
            balance_issues.append(
                f"High mis-signed P&L: mis_signed={float(mis_signed):.3%} (max 7.000%)"
            )
        if aleatoric_uncertainty > 0.08:  # >8% aleatoric
            balance_issues.append(
                f"High aleatoric uncertainty: aleatoric={float(aleatoric_uncertainty):.3%} (max 8.000%)"
            )

        tprint_info("   Label quality metrics:")
        tprint_info(f"     Entropy (0-1): {label_entropy:.3f}")
        tprint_info(f"     Positive rate: {pos_rate:.3%}")
        tprint_info(f"     Negative rate: {neg_rate:.3%}")
        tprint_info(f"     SNR: {float(snr):.3f}")
        tprint_info(f"     Effect size (Cohen d): {float(effect_size):.3f}")
        tprint_info(f"     Mis-signed P&L: {float(mis_signed):.3%}")
        tprint_info(f"     Aleatoric (|ret|<cost): {float(aleatoric_uncertainty):.3%}")

        if balance_issues:
            tprint_warning("⚠️ Label quality issues detected - may impact learnability:")
            for issue in balance_issues:
                tprint_warning(f"   {issue}")
            # Don't fail-fast for now, just warn and log
            tprint_warning("   Proceeding with current labels but consider parameter tuning")
        
        # ------------------------------------------------------------------
        # 6. Use sample weights from Kalman multi-triple-barrier system
        # ------------------------------------------------------------------
        # Sample weights are already computed by the Kalman system based on
        # averaged absolute returns across multiple barrier configurations
        tprint_info("   Using sample weights from Kalman multi-triple-barrier system...")

        # Convert weights to numpy array aligned to market_data.index for downstream processing
        try:
            sample_weights = sample_weights_full.reindex(market_data.index).fillna(0.0).to_numpy(dtype=float)
        except Exception:
            sample_weights = np.asarray(sample_weights_full.values, dtype=float)

        try:
            near_cost_mask_dw = getattr(self, "_near_cost_mask_for_downweight", None)
            dw_mult = float(getattr(self, "_aleatoric_downweight_multiplier", 1.0))
            if isinstance(near_cost_mask_dw, pd.Series) and dw_mult < 1.0:
                aligned = near_cost_mask_dw.reindex(market_data.index).fillna(False)
                sample_weights = np.asarray(sample_weights, dtype=float)
                sample_weights[aligned.values] = sample_weights[aligned.values] * dw_mult
                diagnostics.setdefault("aleatoric_filter", {})["downweight_applied"] = True
        except Exception as e_dw:
            tprint_warning(f"   ⚠️ Aleatoric downweighting failed: {e_dw}")
        
        # Summarize weights
        valid_weights = sample_weights[labeled_mask.values]
        tprint_info(
            f"   Weight stats: mean={np.mean(valid_weights):.3f}, "
            f"std={np.std(valid_weights):.3f}, "
            f"min={np.min(valid_weights):.3f}, max={np.max(valid_weights):.3f}"
        )
        
        # ------------------------------------------------------------------
        # 6b. Apply confident learning noise filter (if enabled)
        # ------------------------------------------------------------------
        if config.get("enable_confident_learning_filter", True):
            tprint_info("   Applying confident learning noise filter...")
            try:
                from .snr_diagnostics import _apply_confident_learning_noise_filter
                
                # Create temporary DataFrame for filtering
                temp_df = market_data.copy()
                temp_df['binary_label'] = binary_labels
                temp_df['realized_return'] = realized_returns
                temp_df['sample_weight'] = sample_weights
                
                # Apply noise filter
                filter_result = _apply_confident_learning_noise_filter(
                    temp_df,
                    y_true_col="binary_label",
                    y_proba_col="meta_probability",  # Will be populated later if available
                    threshold_confident=0.9,
                    verbose=False
                )
                
                if filter_result["applied_filter"]:
                    # Update data with filtered results
                    filtered_indices = set(filter_result["filtered_df"].index)
                    original_indices = set(market_data.index)
                    
                    # Filter all series to match filtered DataFrame
                    mask_keep = market_data.index.isin(filtered_indices)
                    market_data = market_data[mask_keep]
                    binary_labels = binary_labels[mask_keep]
                    realized_returns = realized_returns[mask_keep]
                    sample_weights = sample_weights[mask_keep]
                    primary_signals = primary_signals[mask_keep]
                    event_durations = event_durations[mask_keep]
                    mfe_series = mfe_series[mask_keep]
                    mae_series = mae_series[mask_keep]
                    adaptive_profit = adaptive_profit[mask_keep]
                    adaptive_stop = adaptive_stop[mask_keep]
                    
                    # Update volatility diagnostics
                    if hasattr(self, 'volatility_diagnostics'):
                        self.volatility_diagnostics['noise_filter_applied'] = True
                        self.volatility_diagnostics['noise_filter_stats'] = filter_result["noise_stats"]
                        self.volatility_diagnostics['snr_improvement'] = filter_result.get("snr_improvement", {})
                    
                    tprint_info(f"   Removed {len(filter_result.get('indices_removed', []))} mislabeled samples")
                    if filter_result.get("snr_improvement"):
                        improvement = filter_result["snr_improvement"]
                        tprint_info(f"   SNR improvement: {improvement.get('snr_delta', 0):+.3f} ({improvement.get('snr_pct_change', 0):+.1f}%)")
                else:
                    tprint_info("   No noise filtering applied (no mislabeled candidates detected)")
                    
            except ImportError:
                tprint_warning("   Confident learning filter not available - skipping")
            except Exception as e:
                tprint_warning(f"   Confident learning filter failed: {e}")
        else:
            tprint_info("   Confident learning filter disabled")
        
        # ------------------------------------------------------------------
        # 7a. Add OHLCV-only regime/context features
        # ------------------------------------------------------------------
        if config.get("enable_ohlcv_regime_features", True):
            tprint_info("   Adding OHLCV-only regime features...")
            try:
                from .ohlcv_regime_features import add_ohlcv_regime_features
                
                # Add regime features to market data
                market_data = add_ohlcv_regime_features(
                    market_data,
                    config=config.get("ohlcv_regime_config", {}),
                    verbose=True
                )
                
                tprint_info(f"   Market data now has {len(market_data.columns)} columns")
                
            except ImportError:
                tprint_warning("   OHLCV regime features module not available - skipping")
            except Exception as e:
                tprint_warning(f"   OHLCV regime features failed: {e}")
        else:
            tprint_info("   OHLCV regime features disabled")
        
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        
        # ------------------------------------------------------------------
        # 7. Add advanced feature engineering
        # ------------------------------------------------------------------
        tprint_info("   Adding advanced feature engineering...")

        # Volatility-normalized momentum features
        if 'close' in market_data.columns:
            try:
                # Get volatility for normalization
                close_series = market_data['close']
                log_returns = np.log(close_series).diff()

                # Multiple volatility measures
                vol_5 = log_returns.rolling(5).std()
                vol_20 = log_returns.rolling(20).std()
                vol_50 = log_returns.rolling(50).std()

                # Volatility-normalized momentum (returns scaled by volatility)
                for period in [1, 3, 5, 10, 20]:
                    momentum = close_series.pct_change(period)
                    # Normalize by different volatility horizons
                    market_data[f'momentum_{period}d_vol_norm_5'] = momentum / (vol_5 + 1e-8)
                    market_data[f'momentum_{period}d_vol_norm_20'] = momentum / (vol_20 + 1e-8)
                    market_data[f'momentum_{period}d_vol_norm_50'] = momentum / (vol_50 + 1e-8)

                # Mean reversion features (deviation from moving averages)
                for period in [10, 20, 50]:
                    ma = close_series.rolling(period).mean()
                    std = close_series.rolling(period).std()
                    z_score = (close_series - ma) / (std + 1e-8)

                    # Volatility-normalized z-score
                    vol_norm_factor = vol_20 / vol_20.rolling(period).mean()
                    market_data[f'mean_revert_z_{period}_vol_norm'] = z_score / (vol_norm_factor + 1e-8)

                    # Bollinger band position (volatility-adjusted)
                    bb_upper = ma + 2 * std
                    bb_lower = ma - 2 * std
                    bb_position = (close_series - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    market_data[f'bb_position_{period}_vol_adj'] = bb_position * vol_norm_factor

                tprint_info(f"   ✅ Added volatility-normalized momentum and mean-reversion features")

            except Exception as e:
                tprint_warning(f"   ⚠️ Advanced volatility features failed: {e}")

        # Adaptive lookback features based on current volatility
        try:
            # Calculate adaptive lookback periods based on volatility regime
            vol_regime = pd.qcut(vol_20, q=3, labels=['low_vol', 'med_vol', 'high_vol'])

            # Adaptive RSI (shorter lookback in high vol, longer in low vol)
            adaptive_rsi_periods = vol_regime.map({'low_vol': 21, 'med_vol': 14, 'high_vol': 9})
            adaptive_rsi = pd.Series(index=market_data.index, dtype=float)

            # Adaptive MACD (shorter fast/slow in high vol)
            adaptive_fast_periods = vol_regime.map({'low_vol': 26, 'med_vol': 12, 'high_vol': 8})
            adaptive_slow_periods = vol_regime.map({'low_vol': 52, 'med_vol': 26, 'high_vol': 17})

            for i in range(len(market_data)):
                if i < 50:  # Need some history
                    continue

                # Adaptive RSI
                period = adaptive_rsi_periods.iloc[i]
                if i >= period:
                    prices = close_series.iloc[i-period:i+1]
                    gains = prices.diff().clip(lower=0)
                    losses = -prices.diff().clip(upper=0)
                    avg_gain = gains.rolling(period, min_periods=1).mean().iloc[-1]
                    avg_loss = losses.rolling(period, min_periods=1).mean().iloc[-1]
                    rs = avg_gain / (avg_loss + 1e-8)
                    adaptive_rsi.iloc[i] = 100 - (100 / (1 + rs))

                # Adaptive MACD
                fast_period = adaptive_fast_periods.iloc[i]
                slow_period = adaptive_slow_periods.iloc[i]
                if i >= slow_period:
                    fast_ema = close_series.iloc[i-fast_period:i+1].ewm(span=fast_period, adjust=False).mean().iloc[-1]
                    slow_ema = close_series.iloc[i-slow_period:i+1].ewm(span=slow_period, adjust=False).mean().iloc[-1]
                    market_data.loc[market_data.index[i], 'adaptive_macd'] = fast_ema - slow_ema

            # Only keep RSI where calculated
            market_data['adaptive_rsi'] = adaptive_rsi

            tprint_info(f"   ✅ Added adaptive lookback features (RSI, MACD)")

        except Exception as e:
            tprint_warning(f"   ⚠️ Adaptive lookback features failed: {e}")

        # Trend persistence features (consecutive bars in same direction)
        try:
            # Calculate consecutive up/down bars
            price_changes = close_series.diff()
            bar_direction = (price_changes > 0).astype(int)  # 1 for up, 0 for down

            # Count consecutive bars in same direction
            consecutive_count = pd.Series(index=market_data.index, dtype=int)
            current_streak = 0
            last_direction = None

            for i, dir_val in enumerate(bar_direction):
                if pd.isna(dir_val):
                    consecutive_count.iloc[i] = 0
                    continue

                if last_direction is None or dir_val != last_direction:
                    current_streak = 1
                else:
                    current_streak += 1

                consecutive_count.iloc[i] = current_streak
                last_direction = dir_val

            market_data['trend_persistence_up'] = consecutive_count * (bar_direction == 1)
            market_data['trend_persistence_down'] = consecutive_count * (bar_direction == 0)
            market_data['trend_persistence_total'] = consecutive_count

            # Additional trend features
            market_data['trend_strength_5'] = abs(close_series.pct_change(5))
            market_data['trend_strength_10'] = abs(close_series.pct_change(10))
            market_data['trend_acceleration'] = close_series.pct_change(1) - close_series.pct_change(1).shift(1)

            tprint_info(f"   ✅ Added trend persistence features")

        except Exception as e:
            tprint_warning(f"   ⚠️ Trend persistence features failed: {e}")

        # ------------------------------------------------------------------
        # 7. Build meta-features
        # ------------------------------------------------------------------
        tprint_info("   Building meta-features...")
        volume_available = "volume" in market_data.columns
        
        _, meta_features, _, _ = build_meta_features_for_model(
            market_data=market_data,
            primary_signals=primary_signals,
            realized_returns=realized_returns,
            binary_labels=binary_labels,
            event_durations=event_durations,
            mfe_series=mfe_series,
            mae_series=mae_series,
            adaptive_stop_threshold=adaptive_stop,
            horizon=horizon,
            volume_available=volume_available,
            meta_feature_cfg=config.get("meta_feature_engineering", {}),
        )
        
        n_base_features = meta_features.shape[1]
        tprint_info(f"   Built {n_base_features} base features")
        
        # ------------------------------------------------------------------
        # 7b. Add Kalman-based features (WEIGHTED PIPELINE ONLY)
        # ------------------------------------------------------------------
        # Uses CAUSAL Kalman Filter for live-compatible features
        # (RTS is acausal and only used for label generation in HPO)
        tprint_info("   Generating Kalman-based features...")
        
        try:
            kalman_features = generate_kalman_features(
                market_data=market_data,
                kalman_Q=self.kalman_params['kalman_Q'],
                kalman_R=self.kalman_params['kalman_R'],
            )
            
            # Align indices and merge
            kalman_features_aligned = kalman_features.reindex(meta_features.index).fillna(0)
            
            for col in kalman_features_aligned.columns:
                meta_features[col] = kalman_features_aligned[col]
            
            n_kalman_features = len(kalman_features.columns)
            tprint_success(f"   ✅ Added {n_kalman_features} Kalman features")
        except Exception as kf_exc:
            tprint_warning(f"   ⚠️ Kalman feature generation failed: {kf_exc}")
            n_kalman_features = 0
        
        tprint_info(f"   Total raw features: {meta_features.shape[1]} ({n_base_features} base + {n_kalman_features} Kalman)")

        # ------------------------------------------------------------------
        # 7c(0). Prefer HPO-selected feature set (from meta_labeling_hpo_sample_weighted)
        # ------------------------------------------------------------------
        prefer_hpo_feature_selection = bool(config.get("prefer_hpo_feature_selection", True))
        hpo_feature_selection_applied = False
        if prefer_hpo_feature_selection:
            try:
                hpo_payload = _load_latest_hpo_feature_selection(
                    symbol=str(symbol),
                    timeframe=str(timeframe),
                )
                hpo_selected = hpo_payload.get("selected_features") if isinstance(hpo_payload, dict) else None
                if isinstance(hpo_selected, list) and hpo_selected:
                    # HPO-selected features are typically based on the expanded (multi-horizon + cross)
                    # feature space. Expand locally first so the names match.
                    df_expanded = meta_features.copy()

                    horizon_config = config.get(
                        "feature_horizon_config",
                        {
                            "Short": 5,
                            "Medium": 20,
                            "Long": 60,
                        },
                    )

                    if config.get("enable_multi_horizon_features", True):
                        try:
                            n_before = int(df_expanded.shape[1])
                            df_expanded = generate_multi_horizon_features(df_expanded, horizon_config)
                            tprint_info(
                                f"   Expanded features for HPO application: {n_before} → {int(df_expanded.shape[1])}"
                            )
                        except Exception as e_expand:
                            tprint_warning(f"   ⚠️ Failed to expand features for HPO application: {e_expand}")

                    if config.get("enable_cross_features", True):
                        try:
                            kalman_cols = [c for c in df_expanded.columns if str(c).startswith("KF_")]
                            base_cols = [c for c in df_expanded.columns if not str(c).startswith("KF_")]

                            kalman_features_df = (
                                df_expanded[kalman_cols] if kalman_cols else pd.DataFrame(index=df_expanded.index)
                            )
                            base_features_df = (
                                df_expanded[base_cols] if base_cols else pd.DataFrame(index=df_expanded.index)
                            )

                            cross_features_df = generate_cross_features(
                                base_features=base_features_df,
                                kalman_features=kalman_features_df,
                                market_data=market_data if market_data is not None else pd.DataFrame(index=df_expanded.index),
                            )
                            for col in cross_features_df.columns:
                                if col not in df_expanded.columns:
                                    df_expanded[col] = cross_features_df[col]
                        except Exception as e_cross:
                            tprint_warning(f"   ⚠️ Failed to generate cross-features for HPO application: {e_cross}")

                    selected_ordered = [str(c) for c in hpo_selected]
                    available_cols = [c for c in selected_ordered if c in df_expanded.columns]

                    min_features = int(config.get("hpo_feature_selection_min_features", 30))
                    min_keep_ratio = float(config.get("hpo_feature_selection_min_keep_ratio", 0.5))

                    if available_cols and len(available_cols) >= min_features and (
                        len(available_cols) / max(1, len(selected_ordered))
                    ) >= min_keep_ratio:
                        # Ensure monotone-constraint features are preserved when present
                        for feat in ['atr_14', 'momentum_30', 'rolling_sharpe', 'kaufman_efficiency_ratio']:
                            if feat in df_expanded.columns and feat not in available_cols:
                                available_cols.append(feat)

                        meta_features = df_expanded[available_cols].copy()
                        hpo_feature_selection_applied = True
                        try:
                            qs = hpo_payload.get("quality_scores") if isinstance(hpo_payload, dict) else None
                            if isinstance(qs, dict):
                                self._feature_quality_scores = {c: float(qs.get(c, 0.0)) for c in available_cols}
                        except Exception:
                            pass
                        tprint_info(
                            f"   ✅ Using HPO-selected feature set ({len(available_cols)} features) from {hpo_payload.get('_path') if isinstance(hpo_payload, dict) else 'unknown'}"
                        )
            except Exception as e_hpo_fs:
                tprint_warning(f"   ⚠️ Failed to apply HPO feature set; continuing with in-step selection: {e_hpo_fs}")
        
        # ------------------------------------------------------------------
        # 7c. Quality-based feature selection
        # ------------------------------------------------------------------
        # Solves circular dependency: select features using unsupervised
        # Signal-to-Noise ratio rather than label-dependent metrics.
        # Elbow method determines optimal feature count (no longer arbitrary "70").
        target_feature_count = int(config.get("target_feature_count", 50))  # Fallback for non-Elbow methods
        feature_correlation_threshold = float(config.get("feature_correlation_threshold", 0.85))
        enable_multi_horizon = config.get("enable_multi_horizon_features", True)
        enable_cross_features = config.get("enable_cross_features", True)
        use_hierarchical_selection = config.get("use_hierarchical_selection", True)
        use_lgbm_sweep = config.get("use_lgbm_sweep", True)
        lgbm_lookahead = int(config.get("lgbm_sweep_lookahead", 4))
        lgbm_max_features = int(config.get("lgbm_max_features", 300))
        quality_drop_percentile = float(config.get("quality_drop_percentile", 20.0))
        use_feature_cache = config.get("use_feature_selection_cache", True)
        force_recompute_features = config.get("force_recompute_features", False)
        
        horizon_config = config.get("feature_horizon_config", {
            "Short": 5,
            "Medium": 20,
            "Long": 60,
        })
        
        if not hpo_feature_selection_applied:
            tprint_info("   Running De Prado feature selection pipeline...")
            try:
                meta_features, self._feature_quality_scores = select_features_with_quality(
                    df_features=meta_features,
                    target_n=target_feature_count,
                    correlation_threshold=feature_correlation_threshold,
                    generate_horizons=enable_multi_horizon,
                    horizon_config=horizon_config,
                    enable_cross_features=enable_cross_features,
                    market_data=market_data,
                    config=config,
                    # De Prado pipeline parameters
                    use_hierarchical=use_hierarchical_selection,
                    use_lgbm_sweep=use_lgbm_sweep,
                    lgbm_lookahead=lgbm_lookahead,
                    lgbm_max_features=lgbm_max_features,
                    quality_drop_percentile=quality_drop_percentile,
                    # Caching parameters
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    use_cache=use_feature_cache,
                    force_recompute=force_recompute_features,
                )
                tprint_success(f"   ✅ Selected {meta_features.shape[1]} features (Elbow method optimization)")
            except Exception as fs_exc:
                tprint_warning(f"   ⚠️ Feature selection failed: {fs_exc}. Using all features.")
                self._feature_quality_scores = {}
        else:
            tprint_info("   Skipping De Prado feature selection (HPO feature set applied)")
        
        # ------------------------------------------------------------------
        # 7c. MDA/SHAP Feature Selection (between Layers 2 and 3)
        # ------------------------------------------------------------------
        # Uses Elbow method to determine optimal feature count instead of arbitrary targets
        if config.get("enable_mda_shap_selection", True):
            tprint_info("   🧬 MDA/SHAP Feature Selection...")
            try:
                from .mda_shap_feature_selection import run_mda_shap_feature_selection

                # Get labeled data for feature selection
                labeled_mask = binary_labels.notna()
                if labeled_mask.sum() > 200:  # Need sufficient samples for CV
                    # Setup pre-filters as requested
                    pre_filter_config = {
                        "enable_lgbm_mdi_filter": True,      # LGBM MDI pre-filter (Elbow method determines final count)
                        "enable_correlation_filter": True,    # Drop features with corr > 0.95
                        "enable_variance_filter": True,       # Low variance filter
                        "enable_anova_filter": True           # ANOVA F-test top 75th percentile
                    }

                    # Configure MDA/SHAP selection
                    extractor_cfg = (
                        config.get("regime_leaf_extractor_config")
                        if isinstance(config.get("regime_leaf_extractor_config"), dict)
                        else {}
                    )
                    try:
                        if isinstance(extractor_cfg, dict):
                            extractor_cfg.setdefault(
                                "enabled_targets",
                                [
                                    "regime_liquidity",
                                    "regime_volatility",
                                    "regime_macro_trend",
                                    "regime_trend_efficiency",
                                    "regime_memory",
                                ],
                            )

                            rep_cfg = extractor_cfg.setdefault("reporting", {})
                            if isinstance(rep_cfg, dict):
                                rep_cfg["enabled"] = True

                            onehot_cfg = extractor_cfg.setdefault("onehot", {})
                            if isinstance(onehot_cfg, dict):
                                onehot_cfg.setdefault("enabled", False)

                            inter_cfg = extractor_cfg.setdefault("interaction_feature", {})
                            if isinstance(inter_cfg, dict):
                                inter_cfg.setdefault("enabled", True)
                                inter_cfg.setdefault("include_base", True)
                    except Exception:
                        pass
                    mda_shap_config = {
                        "model_type": "rf",  # Use RandomForest for robustness
                        "n_folds": 5,        # Time-series CV folds
                        "pre_filters": pre_filter_config,
                        "corr_threshold": 0.85,    # Cluster correlation threshold
                        "top_clusters": 8,         # Consider more clusters for Elbow method selection
                        "shap_sample_size": min(1000, labeled_mask.sum()),
                        "regime_leaf_config": {
                            "enabled": bool(config.get("enable_regime_leaf_features", True)),
                            "market_data": market_data,
                            "X_base": None,
                            "extractor_config": extractor_cfg,
                            "random_state": int(config.get("random_state", 42)),
                            "verbose": True,
                        },
                        "enable_shap_interaction_features": bool(
                            config.get("enable_shap_interaction_features", False)
                        ),
                        "shap_interaction_config": (
                            config.get("shap_interaction_config")
                            if isinstance(config.get("shap_interaction_config"), dict)
                            else {
                                "top_main_features": 25,
                                "max_pairs": 30,
                                "max_new_features": 20,
                                "transforms": ["prod"],
                                "fillna_value": 0.0,
                            }
                        ),
                        "verbose": True
                    }

                    # Run MDA/SHAP feature selection with target sample weights
                    selected_features, selection_results = run_mda_shap_feature_selection(
                        X=meta_features.loc[labeled_mask],
                        y=binary_labels[labeled_mask],
                        target_sample_weight=sample_weights[labeled_mask.values],
                        config=mda_shap_config,
                        artifact_router=self.artifact_router,
                        pipeline_context={
                            "symbol": config.get("symbol"),
                            "exchange": config.get("exchange"),
                            "timeframe": config.get("timeframe"),
                            "direction": config.get("direction"),
                        },
                    )

                    try:
                        pre_counts = selection_results.get("prefilter_counts", {}) if isinstance(selection_results, dict) else {}
                        if isinstance(pre_counts, dict) and pre_counts:
                            tprint_info(
                                "   [MDA/SHAP prefilters] "
                                + ", ".join([f"{k}={int(v)}" for k, v in pre_counts.items() if v is not None])
                            )
                    except Exception:
                        pass

                    try:
                        shap_inter = (
                            selection_results.get("shap_interaction_features", {})
                            if isinstance(selection_results, dict)
                            else {}
                        )
                        inter_defs = shap_inter.get("interaction_defs", []) if isinstance(shap_inter, dict) else []
                        if inter_defs:
                            from .shap_interaction_feature_mining import apply_interaction_definitions

                            fillna_value = 0.0
                            try:
                                fillna_value = float(
                                    (mda_shap_config.get("shap_interaction_config") or {}).get("fillna_value", 0.0)
                                )
                            except Exception:
                                fillna_value = 0.0

                            inter_df_full = apply_interaction_definitions(
                                meta_features,
                                inter_defs,
                                fillna_value=fillna_value,
                            )
                            if inter_df_full is not None and not inter_df_full.empty:
                                meta_features = pd.concat([meta_features, inter_df_full], axis=1)
                                tprint_info(f"   🧩 Added SHAP interaction features: {int(inter_df_full.shape[1])}")
                    except Exception:
                        pass

                    # Apply feature selection
                    if selected_features and len(selected_features) > 0:
                        original_count = len(meta_features.columns)
                        meta_features = meta_features[selected_features].copy()

                        tprint_success(f"   ✅ MDA/SHAP selection: {original_count} → {len(selected_features)} features")

                        try:
                            tprint_info(
                                "   [MDA/SHAP selected features] "
                                + ", ".join(list(selected_features))
                            )
                        except Exception:
                            pass

                        # Store results for diagnostics
                        if hasattr(self, 'volatility_diagnostics'):
                            self.volatility_diagnostics['mda_shap_selection_applied'] = True
                            self.volatility_diagnostics['mda_shap_results'] = selection_results
                            self.volatility_diagnostics['selected_features_mda_shap'] = selected_features

                        # Log cluster and SHAP insights
                        if 'importance_rankings' in selection_results:
                            rankings = selection_results['importance_rankings']

                            # Show top MDA clusters
                            if 'mda_clusters' in rankings:
                                top_clusters = list(rankings['mda_clusters'].keys())[:3]
                                tprint_info(f"   🏆 Top MDA clusters: {', '.join(top_clusters)}")

                            # Show top SHAP features
                            if 'shap_features' in rankings:
                                top_shap = list(rankings['shap_features'].keys())[:5]
                                tprint_info(f"   🎯 Top SHAP features: {', '.join(top_shap)}")

                    else:
                        tprint_warning("   ⚠️ MDA/SHAP selection returned no features, keeping all")

                else:
                    tprint_warning("   Insufficient labeled samples for MDA/SHAP selection (need >200)")

            except ImportError:
                tprint_warning("   MDA/SHAP feature selection module not available - skipping")
            except Exception as e:
                tprint_warning(f"   MDA/SHAP feature selection failed: {e}")
            
            # Record MDA/SHAP diagnostics
            try:
                diagnostics["mda_shap"] = {
                    "enabled": True,
                    "selected_features": selected_features if 'selected_features' in locals() else [],
                    "selection_results": selection_results if 'selection_results' in locals() else {},
                }
            except Exception:
                pass
        else:
            tprint_info("   MDA/SHAP feature selection disabled")
            diagnostics["mda_shap"] = {"enabled": False}

        # ------------------------------------------------------------------
        # 7d. Feature bagging to reduce importance concentration
        # ------------------------------------------------------------------
        if config.get("enable_feature_bagging", True):
            tprint_info("   Applying feature bagging to reduce importance concentration...")
            try:
                from .feature_bagging import reduce_importance_concentration
                
                # Get labels for feature bagging
                labeled_mask = binary_labels.notna()
                if labeled_mask.sum() > 100:  # Minimum samples for bagging
                    bagging_config = config.get("feature_bagging_config", {
                        "bag_n_estimators": 30,
                        "bag_sample_fraction": 0.7,
                        "bag_feature_fraction": 0.8,
                        "bag_importance_threshold": 0.01,
                        "bag_min_selection_frequency": 0.3
                    })
                    
                    # Apply feature bagging
                    bagged_features, bagging_results = reduce_importance_concentration(
                        meta_features,
                        binary_labels[labeled_mask],
                        method=config.get("feature_bagging_method", "bagging"),
                        config=bagging_config,
                        verbose=True
                    )
                    
                    # Update meta_features with bagged selection
                    original_features = list(meta_features.columns)
                    meta_features = meta_features[bagged_features].copy()
                    
                    # Store bagging results
                    if hasattr(self, 'volatility_diagnostics'):
                        self.volatility_diagnostics['feature_bagging_applied'] = True
                        self.volatility_diagnostics['bagging_method'] = bagging_results.get('method', 'bagging')
                        self.volatility_diagnostics['bagging_results'] = bagging_results
                        self.volatility_diagnostics['original_feature_count'] = len(original_features)
                        self.volatility_diagnostics['bagged_feature_count'] = len(bagged_features)
                    
                    tprint_info(f"   Feature bagging: {len(original_features)} → {len(bagged_features)} features")
                    
                    # Log concentration metrics
                    if 'concentration_metrics' in bagging_results:
                        metrics = bagging_results['concentration_metrics']
                        tprint_info(f"   Importance concentration: {metrics.get('importance_concentration', 0):.3f}")
                        tprint_info(f"   Top-10 concentration: {metrics.get('top_10_concentration', 0):.3f}")
                    
                    diagnostics["feature_bagging"] = {
                        "enabled": True,
                        "method": bagging_results.get("method", "bagging"),
                        "concentration_metrics": bagging_results.get("concentration_metrics", {}),
                        "selected_features": bagged_features,
                        "original_feature_count": len(original_features),
                        "bagged_feature_count": len(bagged_features),
                    }
                else:
                    tprint_warning("   Insufficient labeled samples for feature bagging")
                    diagnostics["feature_bagging"] = {
                        "enabled": True,
                        "reason": "insufficient_samples",
                    }
                    
            except ImportError:
                tprint_warning("   Feature bagging module not available - skipping")
                diagnostics["feature_bagging"] = {
                    "enabled": False,
                    "reason": "import_error",
                }
            except Exception as e:
                tprint_warning(f"   Feature bagging failed: {e}")
                diagnostics["feature_bagging"] = {
                    "enabled": True,
                    "error": str(e),
                }
        else:
            tprint_info("   Feature bagging disabled")
            diagnostics["feature_bagging"] = {"enabled": False}

        labeled_mask = binary_labels.notna()
        X = meta_features.loc[labeled_mask].fillna(0)
        y = binary_labels[labeled_mask]
        w = sample_weights[labeled_mask.values]

        # ------------------------------------------------------------------
        # 8. Train enhanced models (shallow/regularized/monotone GBDT + calibrated logistic)
        # ------------------------------------------------------------------
        use_enhanced_models = config.get("use_enhanced_models", False)
        
        if use_enhanced_models:
            tprint_info("   Training enhanced models (shallow/regularized/monotone GBDT + calibrated logistic)...")
            try:
                from .enhanced_model_training import train_enhanced_models
                
                enhanced_config = config.get("enhanced_model_config", {
                    "enhanced_model_type": "shallow_gbdt",
                    "calibration_method": "isotonic",
                    "compare_models": True,
                    "cv_splits": 5,
                    "n_bags": 8
                })
                
                # Train enhanced models
                enhanced_results = train_enhanced_models(
                    X=X,
                    y=y,
                    sample_weights=w,
                    config=enhanced_config,
                    verbose=True
                )
                
                # Use primary enhanced model for meta_probability
                primary_model = enhanced_results['primary_model']
                oof_df = primary_model['oof_predictions']
                models = primary_model['models']
                
                # Store enhanced model results
                if hasattr(self, 'volatility_diagnostics'):
                    self.volatility_diagnostics['enhanced_models_applied'] = True
                    self.volatility_diagnostics['enhanced_model_type'] = primary_model['type']
                    self.volatility_diagnostics['enhanced_model_metrics'] = primary_model['metrics']
                    self.volatility_diagnostics['model_comparison'] = {
                        model_type: result.get('metrics', {}) 
                        for model_type, result in enhanced_results['comparison'].items()
                    }
                
                tprint_success(f"   Enhanced model training completed: {primary_model['type']}")
                
            except ImportError:
                tprint_warning("   Enhanced model training module not available - falling back to standard")
                use_enhanced_models = False
            except Exception as e:
                tprint_warning(f"   Enhanced model training failed: {e} - falling back to standard")
                use_enhanced_models = False
        
        if not use_enhanced_models:
            # ------------------------------------------------------------------
            # 8a. Standard weighted bagged LGBM training
            # ------------------------------------------------------------------
            tprint_info("   Training standard weighted bagged LGBM...")
            
            oof_df, models = train_weighted_bagged_lgbm(
                X=X,
                y=y,
                sample_weights=w,
                n_splits=config.get("cv_splits", 5),
                n_bags=config.get("n_bags", 10),
            )
        
        # Compute AUC and calibration metrics
        valid_oof = ~oof_df['lgbm_bag_mean'].isna()
        if valid_oof.sum() > 50 and len(y[valid_oof].unique()) >= 2:
            oof_auc = roc_auc_score(y[valid_oof], oof_df.loc[valid_oof, 'lgbm_bag_mean'])
            tprint_success(f"   ✅ OOF AUC: {oof_auc:.4f}")

            # Compute calibration quality metrics
            y_true_cal = y[valid_oof].values
            y_prob_cal = oof_df.loc[valid_oof, 'lgbm_bag_mean'].values

            calibration_metrics = calculate_calibration_metrics(y_true_cal, y_prob_cal)
            ece = calibration_metrics.get('ece', float('nan'))
            brier_skill_score = calibration_metrics.get('brier_skill_score', float('nan'))

            if not np.isnan(ece):
                tprint_info(f"   ECE: {ece:.3f}")
            if not np.isnan(brier_skill_score):
                tprint_info(f"   Brier Skill Score: {brier_skill_score:.3f}")

            # Calibration gates for artifact quality
            calibration_passed = True
            if not np.isnan(ece) and ece > 0.05:  # ECE > 5% indicates poor calibration
                tprint_warning(f"   ⚠️ High ECE ({ece:.3f}) - probabilities poorly calibrated")
                calibration_passed = False
            if not np.isnan(brier_skill_score) and brier_skill_score < 0.1:  # BSL < 10% vs baseline
                tprint_warning(f"   ⚠️ Low Brier Skill Score ({brier_skill_score:.3f}) - poor discriminative ability")
                calibration_passed = False

            if calibration_passed:
                tprint_success("   ✅ Calibration quality checks passed")
            else:
                tprint_warning("   ⚠️ Calibration issues detected - consider model recalibration")

        else:
            oof_auc = 0.5
            calibration_metrics = {}
            ece = float("nan")
            brier_skill_score = float("nan")
            tprint_warning("   ⚠️ Insufficient OOF predictions for AUC and calibration calculation")
        
        # ------------------------------------------------------------------
        # 8b. Diagnostics: lag-1 stress test and dummy volatility baseline
        # ------------------------------------------------------------------
        try:
            labeled_X = meta_features.loc[labeled_mask].copy()
            labeled_y = binary_labels[labeled_mask].copy()
            if len(labeled_y.dropna()) >= 100 and labeled_X.shape[1] > 0:
                # Lag-1 stress test (lookahead suspicion)
                lag_diag = run_lag1_stress_test(X=labeled_X, y=labeled_y)
                diagnostics["lag1_stress_test"] = lag_diag
                if lag_diag.get("auc_base") and lag_diag.get("auc_lag1"):
                    tprint_info(
                        f"   Lag1 stress: base AUC={lag_diag.get('auc_base'):.3f}, "
                        f"lag1 AUC={lag_diag.get('auc_lag1'):.3f}, "
                        f"diff={lag_diag.get('auc_diff'):.3f}, "
                        f"lookahead={lag_diag.get('lookahead_suspected')}"
                    )
                # Dummy volatility baseline
                if "close" in market_data.columns:
                    vol_proxy = market_data["close"].pct_change().rolling(96).std()
                    dummy_diag = compute_dummy_baseline_auc(volatility=vol_proxy, y=labeled_y)
                    diagnostics["dummy_vol_baseline"] = dummy_diag
                    if dummy_diag.get("auc_dummy") is not None:
                        tprint_info(
                            f"   Dummy vol baseline AUC={dummy_diag.get('auc_dummy'):.3f} "
                            f"(raw={dummy_diag.get('auc_dummy_raw'):.3f}, n={dummy_diag.get('n_samples')})"
                        )
        except Exception as diag_exc:
            tprint_warning(f"   ⚠️ Diagnostics failed: {diag_exc}")

        try:
            if bool(config.get("enable_failure_point_diagnostics", True)):
                diag_thresholds = config.get("failure_diag_thresholds", [0.5, 0.55, 0.6, 0.7])
                diag_thresholds = [float(t) for t in diag_thresholds if t is not None]
                date_range_days = _compute_date_range_days(market_data.index)

                prob_series = oof_df.get('lgbm_bag_mean') if isinstance(oof_df, pd.DataFrame) else None
                if isinstance(prob_series, pd.Series):
                    prob_series = prob_series.copy()

                realized_aligned = (
                    realized_returns.reindex(prob_series.index)
                    if isinstance(prob_series, pd.Series)
                    else realized_returns.loc[labeled_mask]
                )

                thr_stats = {}
                if isinstance(prob_series, pd.Series) and isinstance(realized_aligned, pd.Series):
                    for thr in diag_thresholds:
                        thr_stats[f"thr_{thr}"] = _compute_threshold_trade_stats(
                            realized_returns=realized_aligned,
                            prob_series=prob_series,
                            threshold=thr,
                            date_range_days=date_range_days,
                        )

                learnability = None
                probe_mean_auc = None
                try:
                    probe_max = int(config.get("failure_diag_probe_max_samples", 20000))
                    labeled_mask_probe = binary_labels.notna()
                    X_probe = meta_features.loc[labeled_mask_probe].select_dtypes(include=[np.number])
                    y_probe = binary_labels.loc[labeled_mask_probe]
                    if int(len(y_probe)) > probe_max:
                        X_probe = X_probe.iloc[-probe_max:]
                        y_probe = y_probe.iloc[-probe_max:]
                    learnability, probe_mean_auc = compute_learnability_score(X_probe, y_probe, cv_splits=3, time_aware_cv=True)
                except Exception:
                    learnability, probe_mean_auc = None, None

                exit_reason_counts = None
                try:
                    exit_reason_counts = (
                        exit_reasons.loc[labeled_mask].value_counts(dropna=True).head(20).to_dict()
                        if isinstance(exit_reasons, pd.Series)
                        else None
                    )
                except Exception:
                    exit_reason_counts = None

                failure_flags: List[str] = []
                if float(oof_auc) <= 0.53:
                    failure_flags.append("auc_low")

                if (not np.isnan(ece)) and float(ece) > 0.05:
                    failure_flags.append("calibration_ece_high")
                if (not np.isnan(brier_skill_score)) and float(brier_skill_score) < 0.10:
                    failure_flags.append("calibration_brier_skill_low")

                if isinstance(thr_stats, dict) and thr_stats:
                    try:
                        ref_thr = float(config.get("failure_diag_reference_threshold", 0.55))
                        ref_key = f"thr_{ref_thr}"
                        if ref_key in thr_stats:
                            trades_per_day = thr_stats[ref_key].get("trades_per_day")
                            if trades_per_day is not None and np.isfinite(trades_per_day):
                                if float(trades_per_day) < float(config.get("failure_diag_min_trades_per_day", 1.0)):
                                    failure_flags.append("trades_per_day_low")
                                if float(trades_per_day) > float(config.get("failure_diag_max_trades_per_day", 200.0)):
                                    failure_flags.append("trades_per_day_high")
                    except Exception:
                        pass

                lag_diag = diagnostics.get("lag1_stress_test") if isinstance(diagnostics, dict) else None
                if isinstance(lag_diag, dict) and lag_diag.get("lookahead_suspected"):
                    failure_flags.append("lookahead_suspected")

                dummy_diag = diagnostics.get("dummy_vol_baseline") if isinstance(diagnostics, dict) else None
                if isinstance(dummy_diag, dict):
                    auc_dummy = dummy_diag.get("auc_dummy")
                    if auc_dummy is not None and np.isfinite(auc_dummy) and float(oof_auc) - float(auc_dummy) < 0.02:
                        failure_flags.append("model_not_beating_dummy_vol")

                diagnostics["failure_point_diagnostics"] = {
                    "n_events": int(n_events),
                    "event_coverage": float(n_events / max(1, len(market_data))),
                    "pos_rate": float(pos_rate),
                    "neg_rate": float(neg_rate),
                    "label_entropy": float(label_entropy),
                    "snr": float(snr),
                    "effect_size": float(effect_size),
                    "mis_signed": float(mis_signed),
                    "aleatoric_uncertainty": float(aleatoric_uncertainty),
                    "oof_auc": float(oof_auc),
                    "calibration": {
                        "ece": float(ece) if not np.isnan(ece) else None,
                        "brier_skill_score": float(brier_skill_score) if not np.isnan(brier_skill_score) else None,
                    },
                    "threshold_trade_stats": thr_stats,
                    "learnability_probe": {
                        "learnability": float(learnability) if learnability is not None else None,
                        "mean_auc": float(probe_mean_auc) if probe_mean_auc is not None else None,
                    },
                    "exit_reason_counts": exit_reason_counts,
                    "flags": failure_flags,
                }
        except Exception as failure_diag_exc:
            tprint_warning(f"   ⚠️ Failure-point diagnostics failed: {failure_diag_exc}")
        
        # ------------------------------------------------------------------
        # 9. Assemble labeled data output
        # ------------------------------------------------------------------
        tprint_info("   Assembling labeled data...")
        
        labeled_data = market_data.copy()
        labeled_data["realized_return"] = realized_returns
        labeled_data["binary_label"] = binary_labels
        labeled_data["binary_label_long"] = binary_labels_long
        labeled_data["binary_label_short"] = binary_labels_short
        labeled_data["exit_reason"] = exit_reasons
        labeled_data["event_duration_bars"] = event_durations
        labeled_data["target_sample_weight"] = sample_weights
        
        # Add meta-probability from weighted model
        labeled_data["meta_probability"] = np.nan
        labeled_data.loc[oof_df.index, "meta_probability"] = oof_df['lgbm_bag_mean']
        
        # ------------------------------------------------------------------
        # 8. Calibrate meta_probabilities using isotonic/Platt calibration
        # ------------------------------------------------------------------
        if config.get("enable_probability_calibration", True):
            tprint_info("   Calibrating meta_probabilities...")
            try:
                from .probability_calibration import calibrate_meta_probabilities
                
                # Apply calibration
                calibration_result = calibrate_meta_probabilities(
                    df=labeled_data,
                    y_true_col="binary_label",
                    y_proba_col="meta_probability",
                    method=config.get("calibration_method", "isotonic"),
                    cv_folds=config.get("calibration_cv_folds", 5),
                    min_samples=config.get("calibration_min_samples", 100),
                    plot_dir=config.get("calibration_plot_dir", "./calibration_plots"),
                    verbose=True
                )
                
                if calibration_result["applied"]:
                    # Update labeled data with calibrated probabilities
                    labeled_data = calibration_result["calibrated_df"]
                    calibrated_col = calibration_result["calibrated_column"]
                    
                    # Store calibration metrics
                    if hasattr(self, 'volatility_diagnostics'):
                        self.volatility_diagnostics['probability_calibration_applied'] = True
                        self.volatility_diagnostics['calibration_method'] = calibration_result["method"]
                        self.volatility_diagnostics['calibration_metrics'] = calibration_result["metrics"]
                    
                    tprint_info(f"   Calibration applied: {calibrated_col}")
                    metrics = calibration_result["metrics"]
                    tprint_info(f"   Brier improvement: {metrics.get('brier_improvement', 0):.4f}")
                    tprint_info(f"   ECE improvement: {metrics.get('ece_improvement', 0):.4f}")
                else:
                    tprint_warning("   Probability calibration failed or insufficient data")
                    
            except ImportError:
                tprint_warning("   Probability calibration module not available - skipping")
            except Exception as e:
                tprint_warning(f"   Probability calibration failed: {e}")
        else:
            tprint_info("   Probability calibration disabled")
        
        # Add bagged variants
        labeled_data["meta_probability_lgbm_bag_mean"] = np.nan
        labeled_data.loc[oof_df.index, "meta_probability_lgbm_bag_mean"] = oof_df['lgbm_bag_mean']
        
        labeled_data["meta_probability_lgbm_bag_lower"] = np.nan
        labeled_data.loc[oof_df.index, "meta_probability_lgbm_bag_lower"] = oof_df['lgbm_bag_lower']
        
        # Add metadata columns
        labeled_data["meta_probability_source"] = "weighted_lgbm_bag"
        labeled_data["labeled_data_schema_version"] = "2.0_weighted"
        labeled_data["labeling_timestamp"] = datetime.utcnow().isoformat()
        labeled_data["labeling_method_id"] = f"weighted_meta_labeling_{symbol}_{timeframe}"
        
        # ------------------------------------------------------------------
        # 10. Save artifacts
        # ------------------------------------------------------------------
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        csv_path = None
        labeled_data_summary_path = outcomes_dir / (
            f"weighted_labeled_data_summary_{symbol}_{timeframe}_{timestamp}.json"
        )
        if config.get("save_labeled_data_csv", False):
            csv_path = outcomes_dir / f"weighted_labeled_data_{symbol}_{timeframe}_{timestamp}.csv"
            labeled_data.to_csv(csv_path)
            tprint_success(f"   ✅ Saved labeled data to {csv_path}")
        else:
            try:
                labeled_mask = pd.Series(binary_labels).notna()
                summary_payload = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "timestamp_utc": timestamp,
                    "n_rows": int(len(labeled_data)),
                    "n_labeled": int(labeled_mask.sum()),
                    "label_rate": float(labeled_mask.mean()) if int(len(labeled_mask)) > 0 else 0.0,
                    "n_columns": int(labeled_data.shape[1]),
                    "columns": list(map(str, labeled_data.columns)),
                }
                if isinstance(labeled_data.index, pd.DatetimeIndex) and len(labeled_data.index) > 0:
                    summary_payload["start_time"] = labeled_data.index.min().isoformat()
                    summary_payload["end_time"] = labeled_data.index.max().isoformat()
                with open(labeled_data_summary_path, "w") as f:
                    json.dump(summary_payload, f, indent=2)
                tprint_success(f"   ✅ Saved labeled data summary to {labeled_data_summary_path}")
            except Exception as e_summary:
                tprint_warning(f"   ⚠️ Failed to save labeled data summary: {e_summary}")

        try:
            artifact_name = f"labeled_data_{symbol}_{exchange}_{timeframe}_{direction}"
            labeled_data_artifact_path = self._save_artifact(
                data=labeled_data,
                artifact_name=artifact_name,
                artifact_type="data",
                compression="auto",
                data_category="features",
                metadata={
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "n_samples": int(len(labeled_data)),
                    "n_labeled": int(binary_labels.notna().sum()),
                    "n_features": int(meta_features.shape[1]),
                },
            )
        except Exception as e_save_labeled:
            labeled_data_artifact_path = None
            tprint_warning(f"   ⚠️ Failed to save labeled_data artifact: {e_save_labeled}")

        try:
            legacy_artifact_name = f"labeled_data_{symbol}_{timeframe}"
            self._save_artifact(
                data=labeled_data,
                artifact_name=legacy_artifact_name,
                artifact_type="data",
                compression="auto",
                data_category="features",
            )
        except Exception:
            pass

        try:
            diagnostics_dir = outcomes_dir
            labeled_mask = binary_labels.notna()
            X_raw = meta_features.loc[labeled_mask]
            X_train = X_raw.fillna(0.0)
            y_train = binary_labels.loc[labeled_mask].astype(int)
            r_train = realized_returns.loc[labeled_mask]
            w_train = sample_weights[labeled_mask.values]

            training_matrix = X_train.copy()
            training_matrix["binary_label"] = y_train
            training_matrix["realized_return"] = r_train
            training_matrix["target_sample_weight"] = w_train

            train_matrix_path = diagnostics_dir / (
                f"weighted_meta_model_training_matrix_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
            )
            training_matrix.to_csv(train_matrix_path)

            try:
                self._save_artifact(
                    data=training_matrix,
                    artifact_name=f"weighted_meta_model_training_matrix_{symbol}_{exchange}_{timeframe}_{direction}",
                    artifact_type="data",
                    compression="auto",
                    data_category="features",
                    metadata={
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "direction": direction,
                        "timestamp_utc": timestamp,
                        "n_rows": int(training_matrix.shape[0]),
                        "n_features": int(X_train.shape[1]),
                    },
                )
            except Exception:
                pass

            importances_list = []
            for m in models:
                try:
                    imp = getattr(m, "feature_importances_", None)
                    if imp is None:
                        continue
                    imp_arr = np.asarray(imp, dtype=float)
                    if imp_arr.shape[0] == X_train.shape[1]:
                        importances_list.append(imp_arr)
                except Exception:
                    continue

            feature_importance_df = None
            if importances_list:
                imp_stack = np.vstack(importances_list)
                mean_imp = np.nanmean(imp_stack, axis=0)
                std_imp = np.nanstd(imp_stack, axis=0)
                total_imp = float(np.nansum(mean_imp))
                if total_imp <= 0.0:
                    total_imp = 1.0
                share = mean_imp / total_imp
                feature_importance_df = pd.DataFrame(
                    {
                        "feature_name": list(X_train.columns),
                        "feature_group": [_assign_feature_group(c) for c in X_train.columns],
                        "mean_importance": mean_imp,
                        "std_importance": std_imp,
                        "importance_share": share,
                    }
                ).sort_values("mean_importance", ascending=False)

                fi_path = diagnostics_dir / (
                    f"weighted_meta_model_feature_importance_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                )
                feature_importance_df.to_csv(fi_path, index=False)

            missingness_df = None
            try:
                missing_rate = X_raw.isna().mean().rename("missing_rate")
                missingness_df = pd.DataFrame(
                    {
                        "feature_name": missing_rate.index,
                        "feature_group": [_assign_feature_group(c) for c in missing_rate.index],
                        "missing_rate": missing_rate.values,
                        "n_missing": X_raw.isna().sum().reindex(missing_rate.index).values,
                        "n_total": int(X_raw.shape[0]),
                    }
                ).sort_values("missing_rate", ascending=False)

                missing_path = diagnostics_dir / (
                    f"weighted_meta_model_feature_missingness_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                )
                missingness_df.to_csv(missing_path, index=False)
            except Exception:
                missing_path = None

            group_importance_path = None
            importance_concentration = None
            group_importance = None
            if feature_importance_df is not None and not feature_importance_df.empty:
                try:
                    group_importance = (
                        feature_importance_df.groupby("feature_group")["importance_share"].sum().sort_values(ascending=False)
                    )
                    group_importance_path = diagnostics_dir / (
                        f"weighted_meta_model_feature_group_importance_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                    )
                    group_importance.reset_index().rename(columns={"importance_share": "importance_share_sum"}).to_csv(
                        group_importance_path, index=False
                    )
                except Exception:
                    group_importance_path = None

                try:
                    shares = feature_importance_df["importance_share"].to_numpy(dtype=float)
                    shares = shares[np.isfinite(shares)]
                    shares = shares[shares > 0]
                    if shares.size:
                        shares_sorted = np.sort(shares)[::-1]
                        hhi = float(np.sum(shares_sorted**2))
                        importance_concentration = {
                            "top_5_share": float(np.sum(shares_sorted[:5])) if shares_sorted.size >= 1 else None,
                            "top_10_share": float(np.sum(shares_sorted[:10])) if shares_sorted.size >= 1 else None,
                            "top_20_share": float(np.sum(shares_sorted[:20])) if shares_sorted.size >= 1 else None,
                            "hhi": hhi,
                            "effective_n": float(1.0 / hhi) if hhi > 0 else None,
                        }
                except Exception:
                    importance_concentration = None

            group_map = pd.Series({c: _assign_feature_group(c) for c in X_train.columns})
            group_map_path = diagnostics_dir / (
                f"weighted_meta_model_feature_groups_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
            )
            pd.DataFrame({"feature_name": group_map.index, "feature_group": group_map.values}).to_csv(
                group_map_path, index=False
            )

            date_range_days = _compute_date_range_days(market_data.index)
            ablation_params = {
                "n_estimators": int(config.get("feature_ablation_n_estimators", 200)),
                "max_depth": int(config.get("feature_ablation_max_depth", 4)),
                "learning_rate": float(config.get("feature_ablation_learning_rate", 0.05)),
                "num_leaves": int(config.get("feature_ablation_num_leaves", 31)),
                "subsample": float(config.get("feature_ablation_subsample", 0.8)),
                "colsample_bytree": float(config.get("feature_ablation_colsample", 0.8)),
                "reg_alpha": float(config.get("feature_ablation_reg_alpha", 0.1)),
                "reg_lambda": float(config.get("feature_ablation_reg_lambda", 0.1)),
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42,
            }
            ablation_splits = int(config.get("feature_ablation_cv_splits", 3))
            thresholds = [0.5, 0.55, 0.6, 0.7]

            baseline_probs, _ = _train_weighted_lgbm_oof(
                X=X_train,
                y=y_train,
                sample_weights=w_train,
                n_splits=ablation_splits,
                params=ablation_params,
            )

            baseline_auc = _safe_auc(y_train.to_numpy(dtype=float), baseline_probs.to_numpy(dtype=float))
            baseline_stats = {
                f"thr_{thr}": _compute_threshold_trade_stats(r_train, baseline_probs, thr, date_range_days)
                for thr in thresholds
            }

            rows: List[Dict[str, Any]] = []
            base_row: Dict[str, Any] = {
                "variant": "full",
                "group": "all",
                "n_features": int(X_train.shape[1]),
                "auc": baseline_auc,
            }
            for thr in thresholds:
                s = baseline_stats[f"thr_{thr}"]
                base_row[f"n_trades_at_{thr}"] = s["n_trades"]
                base_row[f"trades_per_day_at_{thr}"] = s["trades_per_day"]
                base_row[f"mean_return_at_{thr}"] = s["mean_return"]
                base_row[f"sum_return_at_{thr}"] = s["sum_return"]
            rows.append(base_row)

            groups = ["trend", "mr", "vol_regime", "specialists", "kalman"]
            for g in groups:
                group_features = [c for c in X_train.columns if group_map.get(c) == g]
                if not group_features:
                    continue

                X_only = X_train[group_features]
                only_probs, _ = _train_weighted_lgbm_oof(
                    X=X_only,
                    y=y_train,
                    sample_weights=w_train,
                    n_splits=ablation_splits,
                    params=ablation_params,
                )
                only_auc = _safe_auc(y_train.to_numpy(dtype=float), only_probs.to_numpy(dtype=float))
                only_row: Dict[str, Any] = {
                    "variant": "only",
                    "group": g,
                    "n_features": int(X_only.shape[1]),
                    "auc": only_auc,
                }
                for thr in thresholds:
                    s = _compute_threshold_trade_stats(r_train, only_probs, thr, date_range_days)
                    only_row[f"n_trades_at_{thr}"] = s["n_trades"]
                    only_row[f"trades_per_day_at_{thr}"] = s["trades_per_day"]
                    only_row[f"mean_return_at_{thr}"] = s["mean_return"]
                    only_row[f"sum_return_at_{thr}"] = s["sum_return"]
                rows.append(only_row)

                drop_features = [c for c in X_train.columns if c not in set(group_features)]
                if not drop_features:
                    continue
                X_drop = X_train[drop_features]
                drop_probs, _ = _train_weighted_lgbm_oof(
                    X=X_drop,
                    y=y_train,
                    sample_weights=w_train,
                    n_splits=ablation_splits,
                    params=ablation_params,
                )
                drop_auc = _safe_auc(y_train.to_numpy(dtype=float), drop_probs.to_numpy(dtype=float))
                drop_row: Dict[str, Any] = {
                    "variant": "drop",
                    "group": g,
                    "n_features": int(X_drop.shape[1]),
                    "auc": drop_auc,
                    "delta_auc_vs_full": (float(drop_auc) - float(baseline_auc))
                    if (drop_auc is not None and baseline_auc is not None)
                    else None,
                }
                for thr in thresholds:
                    s = _compute_threshold_trade_stats(r_train, drop_probs, thr, date_range_days)
                    drop_row[f"n_trades_at_{thr}"] = s["n_trades"]
                    drop_row[f"trades_per_day_at_{thr}"] = s["trades_per_day"]
                    drop_row[f"mean_return_at_{thr}"] = s["mean_return"]
                    drop_row[f"sum_return_at_{thr}"] = s["sum_return"]
                    base_s = baseline_stats[f"thr_{thr}"]
                    drop_row[f"delta_mean_return_at_{thr}_vs_full"] = (
                        float(s["mean_return"]) - float(base_s["mean_return"])
                        if (np.isfinite(s["mean_return"]) and np.isfinite(base_s["mean_return"]))
                        else float("nan")
                    )
                rows.append(drop_row)

            ablation_df = pd.DataFrame(rows)
            ablation_path = diagnostics_dir / (
                f"weighted_meta_model_feature_group_ablation_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
            )
            ablation_df.to_csv(ablation_path, index=False)

            group_counts = None
            missing_groups = None
            prefix_presence = None
            try:
                group_counts = group_map.value_counts().to_dict() if isinstance(group_map, pd.Series) else None
                expected_groups = ["trend", "mr", "vol_regime", "specialists", "kalman"]
                missing_groups = [g for g in expected_groups if (group_counts or {}).get(g, 0) == 0]
                prefix_presence = {
                    "KF_": bool(any(str(c).startswith("KF_") for c in X_train.columns)),
                    "PV_": bool(any(str(c).startswith("PV_") for c in X_train.columns)),
                    "VN_": bool(any(str(c).startswith("VN_") for c in X_train.columns)),
                    "XH_": bool(any(str(c).startswith("XH_") for c in X_train.columns)),
                    "RC_": bool(any(str(c).startswith("RC_") for c in X_train.columns)),
                    "KC_": bool(any(str(c).startswith("KC_") for c in X_train.columns)),
                    "PATH_": bool(any(str(c).startswith("PATH_") for c in X_train.columns)),
                    "ENT_": bool(any(str(c).startswith("ENT_") for c in X_train.columns)),
                    "LIQ_": bool(any(str(c).startswith("LIQ_") for c in X_train.columns)),
                }
            except Exception:
                group_counts, missing_groups, prefix_presence = None, None, None

            diagnostics["feature_diagnostics"] = {
                "training_matrix_csv": str(train_matrix_path),
                "feature_groups_csv": str(group_map_path),
                "feature_importance_csv": str(fi_path) if feature_importance_df is not None else None,
                "feature_missingness_csv": str(missing_path) if missing_path is not None else None,
                "feature_group_importance_csv": str(group_importance_path) if group_importance_path is not None else None,
                "feature_groups_present": group_counts,
                "feature_groups_missing": missing_groups,
                "prefix_presence": prefix_presence,
                "importance_concentration": importance_concentration,
                "feature_group_ablation_csv": str(ablation_path),
            }

            try:
                ic_cfg = config.get("ic_diagnostics") if isinstance(config.get("ic_diagnostics"), dict) else {}
                if bool(ic_cfg.get("enabled", True)):
                    n_bins = int(ic_cfg.get("n_bins", 8))
                    max_samples = int(ic_cfg.get("max_samples", 50000))
                    min_bin_samples = int(ic_cfg.get("min_bin_samples", 200))

                    X_ic = X_train.select_dtypes(include=[np.number]).copy()
                    ic_table, ic_bins, ic_summary = _compute_ic_diagnostics(
                        X=X_ic,
                        y_bin=y_train,
                        y_cont=r_train,
                        feature_groups=group_map.reindex(X_ic.columns) if isinstance(group_map, pd.Series) else None,
                        n_bins=n_bins,
                        max_samples=max_samples,
                        min_bin_samples=min_bin_samples,
                    )

                    ic_table_path = None
                    ic_bins_path = None
                    if ic_table is not None and not ic_table.empty:
                        ic_table_path = diagnostics_dir / (
                            f"weighted_meta_model_ic_table_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                        )
                        ic_table.to_csv(ic_table_path, index=False)

                    if ic_bins is not None and not ic_bins.empty:
                        ic_bins_path = diagnostics_dir / (
                            f"weighted_meta_model_ic_bins_{symbol}_{exchange}_{timeframe}_{direction}_{timestamp}.csv"
                        )
                        ic_bins.to_csv(ic_bins_path, index=False)

                    ic_meta = {}
                    try:
                        prob_series = oof_df.get("lgbm_bag_mean") if isinstance(oof_df, pd.DataFrame) else None
                        if isinstance(prob_series, pd.Series):
                            r_align = realized_returns.reindex(prob_series.index)
                            y_align = binary_labels.reindex(prob_series.index)
                            ic_meta = {
                                "meta_prob_ic_spearman_return": _safe_spearman_corr(
                                    pd.to_numeric(prob_series, errors="coerce").to_numpy(dtype=float),
                                    pd.to_numeric(r_align, errors="coerce").to_numpy(dtype=float),
                                ),
                                "meta_prob_ic_pearson_return": _safe_pearson_corr(
                                    pd.to_numeric(prob_series, errors="coerce").to_numpy(dtype=float),
                                    pd.to_numeric(r_align, errors="coerce").to_numpy(dtype=float),
                                ),
                                "meta_prob_ic_spearman_label": _safe_spearman_corr(
                                    pd.to_numeric(prob_series, errors="coerce").to_numpy(dtype=float),
                                    pd.to_numeric(y_align, errors="coerce").to_numpy(dtype=float),
                                ),
                                "meta_prob_ic_pearson_label": _safe_pearson_corr(
                                    pd.to_numeric(prob_series, errors="coerce").to_numpy(dtype=float),
                                    pd.to_numeric(y_align, errors="coerce").to_numpy(dtype=float),
                                ),
                                "n_oof": int((~prob_series.isna()).sum()),
                            }
                    except Exception:
                        ic_meta = {}

                    diagnostics["ic_diagnostics"] = {
                        "summary": ic_summary,
                        "ic_table_csv": str(ic_table_path) if ic_table_path is not None else None,
                        "ic_bins_csv": str(ic_bins_path) if ic_bins_path is not None else None,
                        "meta_probability": ic_meta,
                    }
            except Exception as ic_exc:
                tprint_warning(f"   ⚠️ IC diagnostics failed: {ic_exc}")
        except Exception as feature_diag_exc:
            tprint_warning(f"   ⚠️ Feature diagnostics failed: {feature_diag_exc}")
        
        try:
            hpo_corr = _generate_hpo_correlation_artifacts(
                outcomes_dir=outcomes_dir,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                timestamp=timestamp,
            )
            if hpo_corr:
                diagnostics["hpo_correlation"] = hpo_corr
        except Exception as corr_exc:
            tprint_warning(f"   ⚠️ HPO correlation analysis failed: {corr_exc}")
         
        # Compile metrics
        metrics = {
            "oof_auc": float(oof_auc),
            "n_events": n_events,
            "n_features": meta_features.shape[1],
            "weighting_source": self.weighting_source,
            "calibration_metrics": calibration_metrics,
            "weighting_params": self.weighting_params,
            "diagnostics": diagnostics,
        }
        
        artifacts = {
            "labeled_data_csv": str(csv_path) if csv_path is not None else None,
            "labeled_data_summary": str(labeled_data_summary_path) if labeled_data_summary_path else None,
            "labeled_data_artifact": labeled_data_artifact_path,
        }
        
        tprint_success(f"✅ Weighted Meta-Labeling complete (AUC={oof_auc:.4f}, events={n_events})")
        
        return {
            "success": True,
            "metrics": metrics,
            "artifacts": artifacts,
            "labeled_data": labeled_data,
        }


def register_weighted_meta_labeling_step() -> None:
    """Register the weighted meta-labeling step in the step registry."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("weighted_meta_labeling", WeightedMetaLabelingStep)
    step_registry.register("weighted_meta_labeling_step", WeightedMetaLabelingStep)
    
    tprint(
        "✅ Weighted meta-labeling step registered "
        "(aliases: weighted_meta_labeling, weighted_meta_labeling_step)",
        "SUCCESS"
    )


# Auto-register when module is imported
register_weighted_meta_labeling_step()
