"""Meta-Labeling HPO Experiment Step.

This offline step performs hierarchical hyperparameter optimization over
labeling-specific parameters (triple-barrier / TPSL, horizon, and target
clipping) using the HierarchicalParameterOptimizer.

It is intentionally decoupled from standard training runs. Invoke it
explicitly via the launcher with an appropriate config. A simple config
flag `enable_labeling_hpo` can be used to disable the optimization and
exit early if desired.

Post-HPO Model Evaluation (NEW):
After HPO completes, trains multiple ML models for SNR diagnostics:
1. Simple LGBM (baseline)
2. Logistic Regression (linear benchmark)
3. LGBM Bagged with Diversity Defense
"""

from __future__ import annotations

from src.training.steps.labeling.multi_label_voting_utils import (
    TripleBarrierConfig,
    compute_multi_triple_barrier_outcomes_vectorized,
    compute_kalman_smoothed_price_and_volatility,
)


from typing import Any, Dict, List, Tuple, Optional
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, pearsonr, rankdata
import lightgbm as lgb

# Post-HPO evaluation imports
from src.training.steps.labeling.post_hpo_model_evaluation import (
    run_post_hpo_evaluation,
    compute_parameter_outcome_correlations,
    generate_correlation_report,
    compute_calibration_metrics,
    compute_snr_diagnostics,
    compute_backtest_metrics,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_meta_features,
    build_meta_features_for_model,
    create_quantile_labels_from_vol_scaled_returns,
    create_regime_aware_quantile_labels_from_vol_scaled_returns,
    create_rolling_quantile_labels_from_vol_scaled_returns,
    create_rolling_regime_aware_quantile_labels_from_vol_scaled_returns,
)
from src.training.steps.labeling.mda_shap_feature_selection import (
    run_mda_shap_feature_selection,
)
from src.training.steps.labeling.generate_weights_per_label import (
    generate_weights_per_label,
    compute_horizon_consistency,
    compute_uniqueness,
    run_layer1_optimization,
)
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    OptimizationStage,
    create_param_group,
)


def _subsample_rows_for_proxy(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if max_rows <= 0:
        return df
    n_rows = len(df)
    if n_rows <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n_rows, size=max_rows, replace=False)
    return df.iloc[sample_idx]


def _rank_signature(
    col_values: np.ndarray,
    *,
    n_bins: int = 64,
) -> bytes:
    if col_values.size == 0:
        return b""

    x = np.asarray(col_values, dtype=np.float32)
    finite = np.isfinite(x)
    n_finite = int(finite.sum())
    if n_finite <= 1:
        binned = np.zeros_like(x, dtype=np.uint8)
    else:
        x_f = x[finite]
        ranks = rankdata(x_f, method="average").astype(np.float32)
        denom = max(float(n_finite - 1), 1.0)
        scaled = (ranks - 1.0) / denom
        binned_f = np.clip(np.floor(scaled * float(n_bins)).astype(np.int32), 0, n_bins - 1).astype(np.uint8)
        binned = np.zeros_like(x, dtype=np.uint8)
        binned[finite] = binned_f
        if not finite.all():
            fill_bin = int(np.median(binned_f)) if binned_f.size else 0
            binned[~finite] = np.uint8(fill_bin)

    h = hashlib.blake2b(binned.tobytes(), digest_size=8)
    return h.digest()


def preprune_by_rank_signature(
    df_features: pd.DataFrame,
    quality_scores: Dict[str, float],
    *,
    target_n: int = 128,
    max_rows: int = 5000,
    n_bins: int = 64,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df_features.empty:
        return df_features, quality_scores
    if len(df_features.columns) <= target_n:
        return df_features, {c: quality_scores.get(c, 0.0) for c in df_features.columns}

    df_sample = _subsample_rows_for_proxy(df_features, max_rows=max_rows, seed=seed)

    bucket_best: Dict[bytes, str] = {}
    for col in df_features.columns:
        sig = _rank_signature(df_sample[col].values, n_bins=n_bins)
        prev = bucket_best.get(sig)
        if prev is None:
            bucket_best[sig] = col
        else:
            if float(quality_scores.get(col, 0.0)) > float(quality_scores.get(prev, 0.0)):
                bucket_best[sig] = col

    candidates = list(bucket_best.values())
    candidates_sorted = sorted(candidates, key=lambda c: float(quality_scores.get(c, 0.0)), reverse=True)
    keep = candidates_sorted[: max(1, min(target_n, len(candidates_sorted)))]
    return df_features[keep], {c: quality_scores.get(c, 0.0) for c in keep}


def _standardize_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(sig > 1e-12, sig, 1.0)
    Xz = (X - mu) / sig
    return np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)


def select_by_anchor_farthest_first(
    df_features: pd.DataFrame,
    quality_scores: Dict[str, float],
    *,
    target_n: int = 70,
    n_anchors: int = 64,
    max_rows: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df_features.empty:
        return df_features, quality_scores

    cols = list(df_features.columns)
    if len(cols) <= target_n:
        return df_features, {c: quality_scores.get(c, 0.0) for c in cols}

    df_sample = _subsample_rows_for_proxy(df_features, max_rows=max_rows, seed=seed)

    cols_sorted = sorted(cols, key=lambda c: float(quality_scores.get(c, 0.0)), reverse=True)
    anchor_cols = cols_sorted[: max(2, min(n_anchors, len(cols_sorted)))]

    X = _standardize_matrix(df_sample[cols_sorted].values)
    Xa = _standardize_matrix(df_sample[anchor_cols].values)

    n = float(max(X.shape[0] - 1, 1))
    fp = (X.T @ Xa) / n
    fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0)

    # Farthest-first selection in fingerprint space (maximize min-distance to selected)
    selected_idx: List[int] = []
    selected_set: set = set()

    # Seed with best-quality feature
    seed_i = 0
    selected_idx.append(seed_i)
    selected_set.add(seed_i)

    min_dist = np.full(fp.shape[0], np.inf, dtype=np.float32)
    for _ in range(1, target_n):
        last = selected_idx[-1]
        d = np.linalg.norm(fp - fp[last:last + 1, :], axis=1)
        min_dist = np.minimum(min_dist, d.astype(np.float32))
        min_dist[list(selected_set)] = -1.0
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] < 0:
            break
        selected_idx.append(nxt)
        selected_set.add(nxt)

    selected_cols = [cols_sorted[i] for i in selected_idx]
    return df_features[selected_cols], {c: quality_scores.get(c, 0.0) for c in selected_cols}


def _find_latest_path(outcomes_dir: Path, pattern: str) -> Optional[Path]:
    try:
        candidates = list(outcomes_dir.glob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def _sanitize_json_value(obj: Any) -> Any:
    if obj is None:
        return None
    try:
        if isinstance(obj, (np.floating,)):
            obj = float(obj)
        if isinstance(obj, (np.integer,)):
            obj = int(obj)
    except Exception:
        pass

    if isinstance(obj, float):
        return float(obj) if np.isfinite(obj) else None

    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(v) for v in obj]

    return obj


def _write_hpo_stage_report(
    *,
    outcomes_dir: Path,
    run_timestamp: str,
    stage_id: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    best_params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
    trials_csv_path: Optional[Path] = None,
    history_json_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outcomes_dir.mkdir(parents=True, exist_ok=True)

    best_params = _sanitize_json_value(best_params or {}) or {}
    metrics = _sanitize_json_value(metrics or {}) or {}
    search_space = _sanitize_json_value(search_space) if search_space is not None else None
    extra = _sanitize_json_value(extra or {}) or {}

    safe_stage = str(stage_id).lower().replace(" ", "_").replace("/", "_").replace("-", "_")

    trials_summary: Dict[str, Any] = {}
    if trials_csv_path is not None and trials_csv_path.exists():
        try:
            df = pd.read_csv(trials_csv_path)
            trials_summary["n_trials"] = int(df.shape[0])

            objective_candidates = [
                "utility",
                "score",
                "best_value",
                "mean_auc",
                "auc",
                "edge",
                "combined",
                "loss",
            ]
            objective_col = next((c for c in objective_candidates if c in df.columns), None)

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            metric_cols = [
                c
                for c in (
                    "utility",
                    "score",
                    "mean_auc",
                    "auc",
                    "trades_per_day",
                    "sharpe_mean",
                    "sharpe_std",
                    "loss",
                    "valid_events",
                    "valid_folds",
                )
                if c in df.columns and c in numeric_cols
            ]

            quantiles = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
            trials_summary["metric_quantiles"] = {
                c: {str(q): float(df[c].quantile(q)) for q in quantiles if np.isfinite(df[c].quantile(q))}
                for c in metric_cols
            }

            # Full Spearman correlation matrix across all numeric columns (all trials)
            try:
                corr_df_full = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                corr_df_full = corr_df_full.dropna(axis=1, how="all")
                corr_cols_full = [
                    c
                    for c in corr_df_full.columns
                    if int(corr_df_full[c].dropna().nunique()) > 1
                ]
                corr_df_full = corr_df_full[corr_cols_full]
                if corr_df_full.shape[1] >= 2:
                    corr_matrix = corr_df_full.corr(method="spearman", min_periods=5)
                    counts_matrix = (
                        corr_df_full.notna().astype(int).T.dot(corr_df_full.notna().astype(int)).astype(int)
                    )

                    corr_path = outcomes_dir / (
                        f"hpo_stage_corr_spearman_{safe_stage}_{symbol}_{exchange}_{timeframe}_{direction}_{run_timestamp}.csv"
                    )
                    counts_path = outcomes_dir / (
                        f"hpo_stage_corr_spearman_counts_{safe_stage}_{symbol}_{exchange}_{timeframe}_{direction}_{run_timestamp}.csv"
                    )

                    corr_matrix.to_csv(corr_path)
                    counts_matrix.to_csv(counts_path)

                    trials_summary["spearman_corr_matrix_csv"] = str(corr_path)
                    trials_summary["spearman_corr_counts_csv"] = str(counts_path)

                    try:
                        corr_arr, p_arr = spearmanr(
                            corr_df_full.to_numpy(dtype=float),
                            axis=0,
                            nan_policy="omit",
                        )
                        if (
                            isinstance(p_arr, np.ndarray)
                            and p_arr.ndim == 2
                            and int(p_arr.shape[0]) == int(len(corr_cols_full))
                        ):
                            pvals_df = pd.DataFrame(p_arr, index=corr_cols_full, columns=corr_cols_full)
                            pvals_path = outcomes_dir / (
                                f"hpo_stage_corr_spearman_pvalues_{safe_stage}_{symbol}_{exchange}_{timeframe}_{direction}_{run_timestamp}.csv"
                            )
                            pvals_df.to_csv(pvals_path)
                            trials_summary["spearman_corr_pvalues_csv"] = str(pvals_path)
                    except Exception:
                        pass
            except Exception as corr_exc:
                trials_summary["spearman_corr_error"] = str(corr_exc)

            param_cols = [c for c in df.columns if str(c).startswith("param_")]
            if not param_cols:
                param_cols = [
                    c
                    for c in df.columns
                    if c
                    in (
                        "kalman_Q",
                        "kalman_R",
                        "sl_atr_mult",
                        "risk_reward_ratio",
                        "trail_distance_atr_mult",
                    )
                    and c in numeric_cols
                ]

            if objective_col is not None and objective_col in numeric_cols:
                df_corr = df[[objective_col] + [c for c in param_cols if c in numeric_cols]].copy()
                df_corr = df_corr.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
                if df_corr.shape[0] >= 5 and df_corr.shape[1] >= 2:
                    corr_series = df_corr.corr(method="spearman")[objective_col].drop(labels=[objective_col])
                    corr_dict = {
                        str(k): float(v)
                        for k, v in corr_series.sort_values(ascending=False).items()
                        if np.isfinite(v)
                    }
                    trials_summary["param_spearman"] = corr_dict

                try:
                    if objective_col != "loss":
                        best_idx = int(df[objective_col].astype(float).idxmax())
                    else:
                        best_idx = int(df[objective_col].astype(float).idxmin())
                    trials_summary["best_trial_row"] = _sanitize_json_value(df.loc[best_idx].to_dict())
                except Exception:
                    pass

                try:
                    sort_ascending = bool(objective_col == "loss")
                    top_df = df.sort_values(objective_col, ascending=sort_ascending).head(20)
                    trials_summary["top_trials"] = _sanitize_json_value(top_df.to_dict(orient="records"))
                except Exception:
                    pass
        except Exception as e_trials:
            trials_summary["error"] = str(e_trials)

    trials_summary = _sanitize_json_value(trials_summary) or {}

    payload: Dict[str, Any] = {
        "stage_id": stage_id,
        "run_timestamp": run_timestamp,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "direction": direction,
        "best_params": best_params,
        "metrics": metrics,
        "search_space": search_space,
        "trials_csv": str(trials_csv_path) if trials_csv_path is not None else None,
        "history_json": str(history_json_path) if history_json_path is not None else None,
        "trials_summary": trials_summary,
        "extra": extra,
    }

    payload = _sanitize_json_value(payload) or payload
    json_path = outcomes_dir / (
        f"hpo_stage_report_{safe_stage}_{symbol}_{exchange}_{timeframe}_{direction}_{run_timestamp}.json"
    )
    md_path = outcomes_dir / (
        f"hpo_stage_report_{safe_stage}_{symbol}_{exchange}_{timeframe}_{direction}_{run_timestamp}.md"
    )

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write(f"# HPO Stage Report: {stage_id}\\n\\n")
        f.write(f"- symbol: {symbol}\\n")
        f.write(f"- exchange: {exchange}\\n")
        f.write(f"- timeframe: {timeframe}\\n")
        f.write(f"- direction: {direction}\\n")
        f.write(f"- run_timestamp: {run_timestamp}\\n\\n")

        f.write("## Best params\\n")
        f.write(f"```json\\n{json.dumps(best_params, indent=2, default=str)}\\n```\\n\\n")

        f.write("## Metrics\\n")
        f.write(f"```json\\n{json.dumps(metrics, indent=2, default=str)}\\n```\\n\\n")

        if search_space is not None:
            f.write("## Search space\\n")
            f.write(f"```json\\n{json.dumps(search_space, indent=2, default=str)}\\n```\\n\\n")

        f.write("## Trial artifacts\\n")
        f.write(f"- trials_csv: {str(trials_csv_path) if trials_csv_path is not None else None}\\n")
        f.write(f"- history_json: {str(history_json_path) if history_json_path is not None else None}\\n\\n")

        if trials_summary:
            f.write("## Trials summary\\n")
            f.write(f"```json\\n{json.dumps(trials_summary, indent=2, default=str)[:20000]}\\n```\\n")

    return {
        "report_json": str(json_path),
        "report_md": str(md_path),
        "trials_csv": str(trials_csv_path) if trials_csv_path is not None else None,
        "history_json": str(history_json_path) if history_json_path is not None else None,
    }

def compute_linear_regime_adaptive_threshold(
    base_threshold: float,
    regime_score: pd.Series,
    min_threshold: float = 0.3,
    max_threshold: float = 0.8,
    sensitivity: float = 0.2,
) -> pd.Series:
    """
    Compute linear regime-adaptive probability thresholds.

    Args:
        base_threshold: Base probability threshold (e.g., 0.55)
        regime_score: Continuous regime score (e.g., volatility z-score, HMM state normalized)
        min_threshold: Minimum allowed threshold
        max_threshold: Maximum allowed threshold
        sensitivity: How much regime affects threshold (0-1)

    Returns:
        Series of adaptive thresholds aligned with regime_score
    """
    # Normalize regime_score to [0, 1] using robust statistics
    regime_median = regime_score.median()
    regime_std = regime_score.std()
    if regime_std == 0:
        regime_std = 1.0

    # Z-score normalization clipped to [-2, 2] then mapped to [0, 1]
    regime_z = (regime_score - regime_median) / regime_std
    regime_z = np.clip(regime_z, -2, 2)
    regime_norm = (regime_z + 2) / 4  # Maps [-2, 2] to [0, 1]

    # Linear adjustment: lower threshold in favorable regimes, higher in unfavorable
    # regime_norm=0 (unfavorable) → threshold = base_threshold + sensitivity
    # regime_norm=1 (favorable) → threshold = base_threshold - sensitivity
    adaptive_threshold = base_threshold + sensitivity * (0.5 - regime_norm)

    # Clip to bounds
    adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)

    return adaptive_threshold


def compute_regime_aware_trade_simulation(
    probabilities: pd.Series,
    realized_returns: pd.Series,
    regime_score: Optional[pd.Series] = None,
    base_threshold: float = 0.55,
    use_linear_adaptive: bool = True,
    min_threshold: float = 0.3,
    max_threshold: float = 0.8,
    sensitivity: float = 0.2,
    transaction_cost: float = 0.003,
) -> Dict[str, Any]:
    """
    Compute trade simulation metrics with optional regime-adaptive gating.

    Args:
        probabilities: Meta-model probabilities
        realized_returns: Realized returns for events
        regime_score: Optional regime score for adaptive gating
        base_threshold: Base probability threshold
        use_linear_adaptive: Whether to use linear regime-adaptive thresholds
        min_threshold: Minimum adaptive threshold
        max_threshold: Maximum adaptive threshold
        sensitivity: Regime sensitivity parameter
        transaction_cost: Transaction cost per trade

    Returns:
        Dictionary with trade simulation metrics
    """
    results = {}

    if use_linear_adaptive and regime_score is not None:
        # Linear regime-adaptive thresholds
        adaptive_thresholds = compute_linear_regime_adaptive_threshold(
            base_threshold=base_threshold,
            regime_score=regime_score,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            sensitivity=sensitivity,
        )

        # Apply adaptive thresholds
        trade_mask = probabilities >= adaptive_thresholds
        results['adaptive_thresholds_used'] = True
        results['mean_adaptive_threshold'] = float(adaptive_thresholds.mean())
        results['std_adaptive_threshold'] = float(adaptive_thresholds.std())
    else:
        # Static threshold
        trade_mask = probabilities >= base_threshold
        results['adaptive_thresholds_used'] = False
        results['mean_adaptive_threshold'] = base_threshold
        results['std_adaptive_threshold'] = 0.0

    # Compute trade metrics
    n_trades = int(trade_mask.sum())
    total_trades = len(probabilities)

    if n_trades > 0:
        trade_returns = realized_returns[trade_mask]
        net_returns = trade_returns  # realized_returns already includes transaction costs

        results['n_trades'] = n_trades
        results['trade_rate'] = float(n_trades / total_trades)
        results['avg_return_per_trade'] = float(net_returns.mean())
        results['total_return'] = float(net_returns.sum())
        results['win_rate'] = float((net_returns > 0).mean())
        results['sharpe_ratio'] = float(net_returns.mean() / (net_returns.std() + 1e-8))
        results['max_drawdown'] = float(net_returns.cumsum().min())
    else:
        results.update({
            'n_trades': 0,
            'trade_rate': 0.0,
            'avg_return_per_trade': 0.0,
            'total_return': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
        })

    return results


def smoothed_brier_lgb_objective(y_pred, dtrain):
    if hasattr(dtrain, "get_label"):
        y_true = dtrain.get_label()
    else:
        y_true = dtrain
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    smoothing_factor = 0.1
    y_smooth = y_true * (1.0 - smoothing_factor) + (smoothing_factor / 2.0)
    grad = 2.0 * p * (1.0 - p) * (p - y_smooth)
    hess = 2.0 * p * (1.0 - p) * (
        p * (1.0 - p) + (1.0 - 2.0 * p) * (p - y_smooth)
    )
    hess = np.maximum(hess, 1e-6)
    return grad, hess


def compute_brier_and_ece(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Brier score and a simple Expected Calibration Error (ECE)."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        return None, None

    y = y[mask]
    p = p[mask]

    brier = float(np.mean((p - y) ** 2))

    # ECE via uniform probability bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)
        if not idx.any():
            continue
        p_bin = p[idx]
        y_bin = y[idx]
        bin_frac = float(idx.mean())
        ece += bin_frac * abs(float(p_bin.mean()) - float(y_bin.mean()))

    return brier, float(ece)


def _cross_val_predict_proba_weighted(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    """Time-series cross-validated predict_proba with sample_weight support.

    This avoids relying on sklearn.cross_val_predict's fit_params API, which
    is version-sensitive, by explicitly looping over TimeSeriesSplit folds.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)

    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    w_arr = np.asarray(sample_weight) if sample_weight is not None else None
    preds = np.zeros(len(y_arr), dtype=float)

    for train_idx, test_idx in cv.split(X, y_arr):
        est = clone(estimator)
        fit_kwargs = {}
        if w_arr is not None:
            fit_kwargs["sample_weight"] = w_arr[train_idx]

        est.fit(X.iloc[train_idx], y_arr[train_idx], **fit_kwargs)
        prob = est.predict_proba(X.iloc[test_idx])[:, 1]
        preds[test_idx] = prob

    return preds


def _cross_val_predict_proba_and_fold_sharpes_weighted(
    *,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[np.ndarray],
    n_splits: int,
    returns: np.ndarray,
    direction: str,
    prob_thr: float = 0.5,
    use_calibration: bool = True,
    enable_ev_gating: bool = False,
    ev_margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """TimeSeriesSplit CV that returns OOF probabilities, fold Sharpes (sized via calibrated probs), and calibration metrics."""
    cv = TimeSeriesSplit(n_splits=n_splits)

    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    w_arr = np.asarray(sample_weight) if sample_weight is not None else None
    ret_arr = np.asarray(returns, dtype=float)
    preds = np.zeros(len(y_arr), dtype=float)
    fold_sharpes: List[float] = []
    fold_briers: List[float] = []
    fold_eces: List[float] = []

    from sklearn.isotonic import IsotonicRegression

    for train_idx, test_idx in cv.split(X, y_arr):
        est = clone(estimator)
        fit_kwargs: Dict[str, Any] = {}
        if w_arr is not None:
            fit_kwargs["sample_weight"] = w_arr[train_idx]

        est.fit(X.iloc[train_idx], y_arr[train_idx], **fit_kwargs)
        prob_test = est.predict_proba(X.iloc[test_idx])[:, 1]

        # Train-only calibration for sizing and calibration metrics
        prob_test_cal = prob_test
        if use_calibration:
            try:
                if len(np.unique(y_arr[train_idx])) >= 2:
                    prob_train = est.predict_proba(X.iloc[train_idx])[:, 1]
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(prob_train, y_arr[train_idx])
                    prob_test_cal = iso.predict(prob_test.astype(float))
            except Exception:
                prob_test_cal = prob_test

        # Store calibrated probabilities for downstream gating/density calculations.
        # Isotonic is monotonic so AUC is typically preserved, and this keeps
        # the OOF probability stream consistent with sizing.
        preds[test_idx] = prob_test_cal

        ev_gate_enabled = bool(enable_ev_gating)
        try:
            ev_margin_f = float(ev_margin)
        except Exception:
            ev_margin_f = 0.0

        e_win = None
        e_loss = None
        if ev_gate_enabled:
            try:
                y_tr = np.asarray(y_arr[train_idx], dtype=float)
                r_tr = np.asarray(ret_arr[train_idx], dtype=float)

                win_mask = (y_tr >= 0.5) & np.isfinite(r_tr)
                loss_mask = (y_tr < 0.5) & np.isfinite(r_tr)

                wins = r_tr[win_mask]
                losses = r_tr[loss_mask]

                if wins.size >= 5:
                    e_win = float(np.mean(wins))
                elif wins.size > 0:
                    e_win = float(np.mean(wins))
                else:
                    e_win = None

                if losses.size >= 5:
                    neg_losses = losses[losses < 0]
                    if neg_losses.size >= 5:
                        e_loss = float(abs(np.mean(neg_losses)))
                    else:
                        e_loss = float(abs(np.mean(losses)))
                elif losses.size > 0:
                    e_loss = float(abs(np.mean(losses)))
                else:
                    e_loss = None

                if e_win is not None and (not np.isfinite(e_win) or e_win <= 0.0):
                    e_win = None
                if e_loss is not None and (not np.isfinite(e_loss) or e_loss <= 0.0):
                    e_loss = None
            except Exception:
                e_win = None
                e_loss = None

        # Sized returns for Sharpe (direction-aware)
        sized = []
        for p, r in zip(prob_test_cal, ret_arr[test_idx]):
            if ev_gate_enabled and e_win is not None and e_loss is not None:
                try:
                    p_f = float(p)
                    ev_hat = (p_f * float(e_win)) - ((1.0 - p_f) * float(e_loss))
                    if not np.isfinite(ev_hat) or (ev_hat <= ev_margin_f):
                        sized.append(0.0)
                        continue
                except Exception:
                    pass
            sz = directional_size_from_prob(
                float(p),
                direction=direction,
                thr=prob_thr,
                max_exposure=1.0,
                scale=1.0,
            )
            sized.append(sz * float(r))
        sized_arr = np.asarray(sized, dtype=float)
        sized_arr = np.asarray(sized, dtype=float)
        if sized_arr.size > 1:
            m_val = float(np.mean(sized_arr))
            s_val = float(np.std(sized_arr))
            # Regularize to prevent infinite Sharpe on zero-variance folds
            adj_std = max(s_val, 1e-5)
            raw_sharpe = m_val / adj_std
            # Cap to avoid artifacts dominating the mean
            fold_sharpes.append(float(np.clip(raw_sharpe, -20.0, 20.0)))
        else:
            fold_sharpes.append(0.0)

        # Calibration metrics on test fold (using calibrated probs)
        try:
            brier, ece = compute_brier_and_ece(y_arr[test_idx], prob_test_cal)
            if brier is not None and np.isfinite(brier):
                fold_briers.append(float(brier))
            if ece is not None and np.isfinite(ece):
                fold_eces.append(float(ece))
        except Exception:
            pass

    mean_brier = float(np.mean(fold_briers)) if fold_briers else None
    mean_ece = float(np.mean(fold_eces)) if fold_eces else None
    return preds, np.asarray(fold_sharpes, dtype=float), mean_brier, mean_ece


from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
logger = system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_success
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs


class TwoStageBaggedMetaModel(BaseEstimator, ClassifierMixin):
    """Two-stage meta-model: activity gate + bagged directional predictor.

    Stage 1: single LightGBM classifier predicts event activity (non-timeout).
    Stage 2: bagged LightGBM ensemble predicts direction conditional on activity.

    Labels convention for y:
        1  -> profit event
        -1 -> stop event
        0  -> timeout / inactive
    """

    def __init__(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        n_bagging: int = 10,
        bagging_fraction: float = 0.7,
        random_state: int = 42,
    ) -> None:
        self.base_params = {} if base_params is None else dict(base_params)
        self.n_bagging = int(n_bagging)
        self.bagging_fraction = float(bagging_fraction)
        self.random_state = int(random_state)

        # Phase 2 Optimization: Force balanced class weights by default
        # to handle extreme label imbalance (e.g. 5% positive rate)
        # DISABLE for Phase 3: Let HPO decide or default to None. 'balanced' causes degenerate 0.5 preds on weak features.
        # if "class_weight" not in self.base_params and "is_unbalance" not in self.base_params:
        #     self.base_params["class_weight"] = "balanced"


        # Ensure random_state is set in base_params if not already present
        if "random_state" not in self.base_params:
            self.base_params["random_state"] = self.random_state
        
        self.stage1_model: Optional[lgb.LGBMClassifier] = lgb.LGBMClassifier(**self.base_params)
        self.stage2_ensemble: Optional[BaggingClassifier] = BaggingClassifier(
            estimator=lgb.LGBMClassifier(**self.base_params),
            n_estimators=self.n_bagging,
            max_samples=self.bagging_fraction,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state,
        )

    def fit(self, X: Any, y: Any) -> "TwoStageBaggedMetaModel":
        y_arr = np.asarray(y)

        # Stage 1: activity gate (non-timeout vs timeout)
        y_activity = (y_arr != 0).astype(int)
        self.stage1_model.fit(X, y_activity)

        # Stage 2: direction among active events only
        active_mask = (y_arr != 0)
        if not np.any(active_mask):
            self.stage2_ensemble = None
            return self

        X_dir = X[active_mask]
        y_dir_raw = y_arr[active_mask]

        # Map profit vs stop to {1, 0}
        y_dir = (y_dir_raw == 1).astype(int)

        if X_dir.shape[0] > 50 and np.unique(y_dir).size >= 2:
            self.stage2_ensemble.fit(X_dir, y_dir)
        else:
            self.stage2_ensemble = None

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        # P(active)
        p_active = self.stage1_model.predict_proba(X)[:, 1]

        # P(win | active)
        if self.stage2_ensemble is not None:
            p_win_conditional = self.stage2_ensemble.predict_proba(X)[:, 1]
        else:
            p_win_conditional = np.full(X.shape[0], 0.5, dtype=float)

        final_score = p_active * p_win_conditional
        proba_1 = final_score
        proba_0 = 1.0 - proba_1
        return np.vstack([proba_0, proba_1]).T

def generate_trinary_labels(
    events_df: pd.DataFrame,
    outcomes_series: pd.Series,
    vertical_barrier_col: str = "t1",
) -> pd.Series:
    """Map standard triple-barrier outcomes to {1, -1, 0} for two-stage modeling.

    1  = Profit Target Hit
    -1 = Stop Loss Hit
    0  = Time Limit Exceeded (Vertical Barrier / timeout)

    Args:
        events_df: DataFrame with at least a 'ret' column containing realized returns per event.
        outcomes_series: Binary outcomes (0/1) or similar event labels aligned to events_df.
        vertical_barrier_col: Placeholder for future use when dynamic vertical barriers are explicit.
    """
    # Initialize all events as timeouts (0)
    y_trinary = pd.Series(index=events_df.index, data=0)

    # 1. Profits: outcome==1
    mask_profit = (outcomes_series == 1)
    y_trinary[mask_profit] = 1

    # 2. Among non-profits (outcome==0), distinguish stop vs timeout using returns
    mask_loss_or_timeout = (outcomes_series == 0)

    if "ret" in events_df.columns:
        epsilon = 0.0005  # 5 bps buffer for fees/slippage
        mask_stop = mask_loss_or_timeout & (events_df["ret"] < -epsilon)
        y_trinary[mask_stop] = -1
        # Remaining loss/timeout events stay at 0 (timeouts / mild losses)
    else:
        raise ValueError("Need 'ret' column to distinguish Stop from Timeout.")

    return y_trinary.astype(int)


# Reuse core labeling utilities from the production meta-labeling step
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    compute_realized_returns,
    kalman_smooth_labels,
    fit_probability_to_return_mapping,
    translate_to_targets_with_isotonic,
    generate_primary_signals,
    DEFAULT_TRANSACTION_COST,
    ECON_MIN_RETURN_MULTIPLE,
    build_meta_features_for_model,
    compute_label_entropy_score,
    generate_diagnostics_report,
    compute_vol_scaled_returns_for_events,
    create_quantile_labels_from_vol_scaled_returns,
    attach_rolling_hmm_regimes_to_market_data,
    create_regime_aware_quantile_labels_from_vol_scaled_returns,
)
from src.training.steps.labeling.lgbm_feature_selection import FeatureSetPersistence
from src.training.steps.labeling.label_config import (
    build_label_config,
    compute_label_config_id,
)

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.utils.ml_common.optimization.pareto import (
    Solution,
    ParetoFront,
    compute_pareto_front,
    select_knee_point,
)




# Optional diagnostics for the recommended configuration can be useful but are
# not required for the HPO step to function. They have occasionally triggered
# pandas categorical setitem issues in some environments. To keep the HPO step
# robust, we disable these diagnostics by default and gate them behind this
# constant, which can be flipped to True if deeper investigation is needed.
GENERATE_RECOMMENDED_DIAGNOSTICS: bool = False

# Toggle for underfit diagnostics - computes learning curves, feature importance
# concentration, and probe vs deeper model comparisons. Adds computational cost.
ENABLE_UNDERFIT_DIAGNOSTICS: bool = True

# Minimum sample requirements for HPO phases
MIN_EVENTS_PHASE1 = 200

# ============================================================================
# REPRODUCIBILITY & CONFIGURATION CONSTANTS
# ============================================================================
# Centralized random seed for all random operations to ensure reproducibility
DEFAULT_RANDOM_SEED: int = 42

# CV embargo period (bars) to prevent look-ahead bias in time-series splits
# Default: 1 day worth of bars at 15m timeframe (~96 bars)
DEFAULT_CV_EMBARGO_BARS: int = 96

# Minimum gap between train/test splits to prevent leakage
DEFAULT_CV_GAP_BARS: int = 24  # ~6 hours at 15m

# ============================================================================
# CACHING & PERFORMANCE HELPERS
# ============================================================================
class ObjectiveComputationCache:
    """Cache for expensive computations in labeling_objective to avoid recomputation.
    
    Caches invariant computations that don't depend on HPO parameters:
    - ATR series (depends only on market_data)
    - Trend strength (depends only on market_data and config)
    - Volatility baseline (for fixed vol_baseline_window)
    - Primary signals (for default cusum_threshold)
    """
    def __init__(self):
        self._atr_cache: Optional[pd.Series] = None
        self._trend_strength_cache: Optional[pd.Series] = None
        self._vol_baseline_cache: Dict[int, pd.Series] = {}  # key: vol_baseline_window
        self._primary_signals_cache: Optional[pd.DataFrame] = None
        self._default_cusum: float = 0.015
        
    def get_atr(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Get or compute ATR series."""
        if self._atr_cache is None:
            high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
            low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
            close_prices = market_data["close"]
            tr1 = high_prices - low_prices
            tr2 = (high_prices - close_prices.shift(1)).abs()
            tr3 = (low_prices - close_prices.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            trend_atr_window = int(config.get("trend_strength_atr_window", 14))
            self._atr_cache = true_range.rolling(window=trend_atr_window, min_periods=1).mean()
        return self._atr_cache
    
    def get_trend_strength(self, market_data: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
        """Get or compute trend strength series."""
        if self._trend_strength_cache is None:
            atr_series = self.get_atr(market_data, config)
            close_prices = market_data["close"]
            trend_delta_lookback = int(config.get("trend_strength_delta_lookback", 4))
            price_delta = close_prices.diff(trend_delta_lookback).abs()
            trend_strength = (price_delta / (atr_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
            trend_strength = trend_strength.clip(
                lower=0.0,
                upper=float(config.get("trend_strength_clip", 5.0)),
            ).fillna(0.0)
            self._trend_strength_cache = trend_strength
        return self._trend_strength_cache
    
    def get_vol_baseline(self, volatility_1d: pd.Series, vol_baseline_window: int) -> pd.Series:
        """Get or compute volatility baseline for given window."""
        if vol_baseline_window not in self._vol_baseline_cache:
            self._vol_baseline_cache[vol_baseline_window] = volatility_1d.rolling(vol_baseline_window).mean()
        return self._vol_baseline_cache[vol_baseline_window]
    
    def get_primary_signals(self, market_data: pd.DataFrame, cusum_threshold: float, 
                           target_signal_density: float, default_cusum: float = 0.015) -> pd.DataFrame:
        """Get cached primary signals or regenerate if threshold differs."""
        if abs(cusum_threshold - default_cusum) <= 0.001:
            if self._primary_signals_cache is None:
                from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
                self._primary_signals_cache = generate_primary_signals(
                    market_data.copy(),
                    cusum_threshold=default_cusum,
                    target_trades_per_day=target_signal_density,
                )
            return self._primary_signals_cache
        else:
            # Regenerate for non-default threshold
            from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
            return generate_primary_signals(
                market_data.copy(),
                cusum_threshold=cusum_threshold,
                target_trades_per_day=target_signal_density,
            )
    
    def clear(self):
        """Clear all caches."""
        self._atr_cache = None
        self._trend_strength_cache = None
        self._vol_baseline_cache.clear()
        self._primary_signals_cache = None


def safe_config_copy(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a deep copy of config to avoid mutating caller's dict.
    
    Args:
        config: Original configuration dictionary
        
    Returns:
        Deep copy of config that can be safely modified
    """
    import copy
    return copy.deepcopy(config)


def validate_sample_weight_alignment(
    weights: Optional[np.ndarray],
    labels: pd.Series,
    indices: Optional[np.ndarray] = None,
) -> bool:
    """Validate that sample weights align with labels/index.
    
    Args:
        weights: Sample weights array (can be None)
        labels: Label series
        indices: Optional index array for validation
        
    Returns:
        True if alignment is valid, False otherwise
    """
    if weights is None:
        return True
    
    if not isinstance(weights, np.ndarray):
        return False
    
    expected_len = len(labels) if indices is None else len(indices)
    if len(weights) != expected_len:
        return False
    
    if not np.all(np.isfinite(weights)):
        return False
    
    return True


def create_time_aware_cv_splits(
    n_splits: int,
    n_samples: int,
    embargo_bars: int = DEFAULT_CV_EMBARGO_BARS,
    gap_bars: int = DEFAULT_CV_GAP_BARS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time-aware CV splits with embargo to prevent look-ahead bias.
    
    Args:
        n_splits: Number of CV folds
        n_samples: Total number of samples
        embargo_bars: Number of bars to embargo after test set
        gap_bars: Minimum gap between train and test sets
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    # Use TimeSeriesSplit which respects temporal order
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        # Apply embargo: remove samples immediately after test set
        if len(test_idx) > 0:
            max_test_idx = test_idx.max()
            embargo_end = min(max_test_idx + embargo_bars, n_samples)
            # Remove embargoed samples from train set
            train_idx = train_idx[train_idx < max_test_idx - gap_bars]
        
        splits.append((train_idx, test_idx))
    
    return splits


def get_reproducible_random_state(base_seed: int = DEFAULT_RANDOM_SEED, 
                                  offset: int = 0) -> int:
    """Get a reproducible random state seed.
    
    Args:
        base_seed: Base seed value
        offset: Optional offset for different components
        
    Returns:
        Reproducible random seed
    """
    return base_seed + offset  # Minimum events for Phase 1 (sample count optimization)
MIN_EVENTS_PHASE2 = 300  # Minimum events for Phase 2 (edge refinement)

# Multi-stage HPO configuration defaults
# TWO-PHASE HPO DESIGN:
# Phase 1 (Stages 1-2): Optimize for SAMPLE COUNT + basic AUC
#   - Primary objective: n_events >= MIN_EVENTS_PHASE1
#   - Secondary objective: AUC > 0.52
# Phase 2 (Stages 3-4): Optimize for EDGE on sufficient samples
#   - Only runs if Phase 1 achieves MIN_EVENTS_PHASE2
#   - Primary objective: realistic P&L edge
STAGE_CONFIGS: List[OptimizationStage] = [
    {
        "name": "Stage 1 (Sample Count Screening)",
        "complexity": "fast",
        "n_trials": 60,  # Reduced from 100 - focus on finding configs with enough samples
        "top_k_to_pass": 30,
        "phase": 1,  # Phase 1: sample count optimization
        "min_events_required": MIN_EVENTS_PHASE1,
        "model_params": {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "cv_splits": 3,
        },
    },
    {
        "name": "Stage 2 (Sample Count Refinement)",
        "complexity": "medium",
        "n_trials": 30,  # Reduced from 50
        "top_k_to_pass": 10,
        "phase": 1,  # Still Phase 1
        "min_events_required": MIN_EVENTS_PHASE1,
        "model_params": {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "cv_splits": 3,
        },
    },
    {
        "name": "Stage 3 (Edge Optimization)",
        "complexity": "strong",
        "n_trials": 30,  # Increased from 35 for better edge exploration
        "top_k_to_pass": 5,  # Pass more to Stage 4 for refinement
        "phase": 2,  # Phase 2: edge optimization (only if sufficient samples)
        "min_events_required": MIN_EVENTS_PHASE2,
        "model_params": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.01,
            "cv_splits": 3,
            "use_feature_selection": True,
            "use_resampling": True,
        },
    },
    {
        "name": "Stage 4 (Edge Refinement)",
        "complexity": "strong",
        "n_trials": 30,  # Increased from 45
        "top_k_to_pass": 1,
        "phase": 2,  # Phase 2
        "min_events_required": MIN_EVENTS_PHASE2,
        "model_params": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.01,
            "cv_splits": 3,
            "use_feature_selection": True,
            "use_resampling": True,
        },
    },
    {
        "name": "Stage 5 (Smart Walker - Model HPO)",
        "complexity": "strong",
        "n_trials": 100,  # Walker will stop early if needed
        "top_k_to_pass": 1,
        "phase": 2,
        "min_events_required": MIN_EVENTS_PHASE2,
        "optimization_stage": OptimizationStage.SMART_WALKER, # Explicitly use Smart Walker
        "model_params": {
            "cv_splits": 3,
            "use_feature_selection": True,
            "use_resampling": True,
        },
    },
]


def _build_t1_aware_purged_splits_for_events(
    y: pd.Series,
    event_durations: Optional[pd.Series],
    market_index: pd.DatetimeIndex,
    cv_splits: int,
    base_horizon_bars: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build purged, embargoed K-fold splits using event [t0, t1] windows.

    This helper operates in *event space* (one row per labeled event) and
    removes from the training set any events whose [t0, t1] window overlaps
    the validation window extended by +/- horizon.

    Args:
        y: Clean label series indexed by event timestamps (no NaNs).
        event_durations: Per-event durations in bars (same index as
            ``market_index``). When missing, ``base_horizon_bars`` is used.
        market_index: Full market_data index used to map bar positions to
            timestamps.
        cv_splits: Number of folds.
        base_horizon_bars: Fallback vertical barrier in bars when event
            durations are missing.

    Returns:
        List of (train_idx, val_idx) index arrays in event space.
    """

    n_samples = len(y)
    if cv_splits < 2 or n_samples < cv_splits * 2:
        # Fallback: naive sequential splits without purging when data is
        # too small to support robust purged CV.
        fold_sizes = np.full(cv_splits, n_samples // cv_splits, dtype=int)
        fold_sizes[: n_samples % cv_splits] += 1
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = np.arange(start, stop)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            splits.append((train_idx, val_idx))
            current = stop
        return splits

    labels_index = y.index

    # Map event timestamps to integer positions in the full market index.
    pos = market_index.get_indexer(labels_index)
    valid_pos_mask = pos >= 0
    if not np.all(valid_pos_mask):
        # Drop any events that cannot be mapped back to market_data.
        labels_index = labels_index[valid_pos_mask]
        y = y.iloc[valid_pos_mask]
        pos = pos[valid_pos_mask]
        n_samples = len(labels_index)
        if cv_splits < 2 or n_samples < cv_splits * 2:
            fold_sizes = np.full(cv_splits, n_samples // cv_splits, dtype=int)
            fold_sizes[: n_samples % cv_splits] += 1
            splits = []
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                val_idx = np.arange(start, stop)
                train_mask = np.ones(n_samples, dtype=bool)
                train_mask[val_idx] = False
                train_idx = np.nonzero(train_mask)[0]
                splits.append((train_idx, val_idx))
                current = stop
            return splits

    # Derive per-event durations in bars.
    if event_durations is not None:
        dur = event_durations.reindex(labels_index).fillna(base_horizon_bars or 1)
    else:
        dur = pd.Series(base_horizon_bars or 1, index=labels_index)
    dur = dur.astype(int).clip(lower=1)

    end_pos = pos + dur.to_numpy()
    end_pos = np.clip(end_pos, 0, len(market_index) - 1)

    t0_times = market_index[pos]
    t1_times = market_index[end_pos]

    # Approximate single-bar delta from market_index; fallback to 15m.
    if len(market_index) >= 2:
        bar_delta = market_index[1] - market_index[0]
    else:
        bar_delta = pd.Timedelta(minutes=15)

    horizon_bars = max(int(base_horizon_bars or 1), 1)
    horizon_delta = bar_delta * horizon_bars
    purge_delta = horizon_delta
    embargo_delta = horizon_delta

    # Sequential folds in event order.
    fold_sizes = np.full(cv_splits, n_samples // cv_splits, dtype=int)
    fold_sizes[: n_samples % cv_splits] += 1

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = np.arange(start, stop)

        val_start_time = t0_times[start]
        val_end_time = t0_times[stop - 1]
        window_start = val_start_time - purge_delta
        window_end = val_end_time + embargo_delta

        # Train events whose [t0, t1] overlaps the extended validation
        # window are removed from training.
        overlap = (t1_times >= window_start) & (t0_times <= window_end)
        train_mask = ~overlap
        train_mask[val_idx] = False
        train_idx = np.nonzero(train_mask)[0]

        splits.append((train_idx, val_idx))
        current = stop

    return splits


def compute_learnability_with_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    model_complexity: str = "fast",
    cv_splits: int = 3,
    time_aware_cv: bool = True,
    use_ensemble: bool = False,
    signal_strength_scale_max: float = 1.5,
    event_durations: Optional[pd.Series] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
    base_horizon_bars: Optional[int] = None,
    use_smoothed_brier_objective_lgbm: bool = False,
    scale_pos_weight: Optional[float] = None,
    use_feature_selection: bool = False,
    use_resampling: bool = False,
) -> Tuple[float, float, np.ndarray, Optional[IsotonicRegression], np.ndarray, np.ndarray]:
    """Compute learnability score with isotonic calibration for accurate P&L estimation.

    Unlike the basic compute_learnability_score, this function:
    1. Uses model complexity levels (fast/medium/strong)
    2. Returns calibrated probabilities via isotonic regression
    3. Supports ensemble models for strong complexity

    Args:
        X: Feature matrix
        y: Binary labels
        realized_returns: Realized returns for isotonic calibration
        model_complexity: "fast", "medium", or "strong"
        cv_splits: Number of CV splits
        time_aware_cv: Use TimeSeriesSplit instead of KFold
        use_ensemble: Whether to use ensemble of models (for strong complexity)

    Returns:
        Tuple of (learnability_score, mean_auc, calibrated_probabilities, isotonic_regressor, fold_aucs_array, oof_probs_full)
    """
    tprint_info(f"🔧 compute_learnability_with_calibration() called")
    tprint_info(f"   model_complexity={model_complexity}, cv_splits={cv_splits}, X_shape={X.shape}")
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    # Remove NaN labels
    valid_mask = ~y.isna()
    X_num = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
    if isinstance(X_num, pd.DataFrame) and X_num.empty:
        return 0.0, 0.5, np.array([]), None, np.array([]), np.array([])

    X_clean = X_num[valid_mask].fillna(0)
    if isinstance(X_clean, pd.DataFrame) and X_clean.shape[1] == 0:
        return 0.0, 0.5, np.array([]), None, np.array([]), np.array([])
    y_clean = y[valid_mask]
    returns_clean = realized_returns[valid_mask]

    try:
        y_clean_counts = y_clean.value_counts(dropna=False).to_dict()
    except Exception:
        y_clean_counts = {}
    tprint(
        f"[LEARNABILITY_LABEL_STATS] n_clean={len(y_clean)}, value_counts={y_clean_counts}",
        "INFO",
    )

    if len(y_clean) < 50:
        return 0.0, 0.5, np.array([]), None, np.array([]), np.array([])

    if len(y_clean.unique()) < 2:
        return 0.0, 0.5, np.array([]), None, np.array([]), np.array([])

    # Additional guard: if all numeric features are constant (or effectively
    # collapsed to a single value) after cleaning, many tree learners will
    # refuse to train and LightGBM can surface num_features()==0 errors.
    # Detect and short-circuit early in that case.
    if isinstance(X_clean, pd.DataFrame):
        try:
            nunique = X_clean.nunique(dropna=False)
            non_constant_cols = nunique[nunique > 1].index
            if len(non_constant_cols) == 0:
                tprint(
                    f"[LEARNABILITY_EMPTY_FEATURES] All features constant for learnability "
                    f"(n_clean={len(y_clean)}, y_counts={y_clean_counts}); returning defaults",
                    "WARNING",
                )
                return 0.0, 0.5, np.array([]), None, np.array([]), np.array([])
            if len(non_constant_cols) < X_clean.shape[1]:
                X_clean = X_clean.loc[:, non_constant_cols]
        except Exception:
            # If anything goes wrong during constant-feature detection, fall back
            # to the original X_clean; downstream guards and the outer try/except
            # will still prevent hard failures.
            pass

    # Select model based on complexity
    if isinstance(X_clean, pd.DataFrame):
        n_features_clean = X_clean.shape[1]
    else:
        X_clean_arr = np.asarray(X_clean)
        if X_clean_arr.ndim == 1:
            n_features_clean = 1
        else:
            n_features_clean = X_clean_arr.shape[1]

    # ===== FEATURE SELECTION (Fast RFE-like) =====
    if use_feature_selection and n_features_clean > 10:
        try:
            # Train a quick fast model to get feature importance
            selector = lgb.LGBMClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1, 
                random_state=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=1), verbose=-1
            )
            selector.fit(X_clean, y_clean)
            importances = selector.feature_importances_

            # Select top 60% features or at least top 10
            indices_sorted = np.argsort(importances)[::-1]
            n_keep = max(10, int(n_features_clean * 0.6))
            selected_indices = indices_sorted[:n_keep]

            if isinstance(X_clean, pd.DataFrame):
                X_clean = X_clean.iloc[:, selected_indices]
                tprint(f"[RFE] Selected {n_keep}/{n_features_clean} features", "INFO")
                n_features_clean = X_clean.shape[1]
            else:
                X_clean = X_clean_arr[:, selected_indices]
                tprint(f"[RFE] Selected {n_keep}/{n_features_clean} features (numpy)", "INFO")
                n_features_clean = X_clean.shape[1]
        except Exception as e:
            tprint(f"⚠️ Feature selection failed: {e}", "WARNING")
            pass

    if n_features_clean <= 1:
        models = [
            LogisticRegression(
                max_iter=400,
                n_jobs=-1,
                penalty="l2",
                solver="lbfgs",
            )
        ]
    else:
        if model_complexity == "fast":
            models = [lgb.LGBMClassifier(
                boosting_type='gbdt',
                objective='binary',
                n_estimators=20,
                max_depth=2,
                num_leaves=4,  # Explicitly set satisfies 2^depth condition
                learning_rate=0.15,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                n_jobs=-1,
                verbose=-1,
                random_state=42,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                feature_pre_filter=False,
                min_data_in_bin=1,
            )]

        elif model_complexity == "medium":
            models = [lgb.LGBMClassifier(
                boosting_type='gbdt',
                objective=smoothed_brier_lgb_objective if use_smoothed_brier_objective_lgbm else 'binary',
                max_depth=3,
                num_leaves=8, # 2^3
                n_estimators=150,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,  # Relaxed from 60 to allow splits on small/weak signal subsets
                reg_alpha=0.1,         # Relaxed from 0.2
                reg_lambda=0.1,        # Relaxed from 0.7
                n_jobs=-1,
                verbose=-1,
                random_state=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=2),
                feature_pre_filter=False,
                min_data_in_bin=1,
                scale_pos_weight=scale_pos_weight,
            )]

        else:  # strong
            models = [
                lgb.LGBMClassifier(
                    boosting_type='gbdt',
                    objective=smoothed_brier_lgb_objective if use_smoothed_brier_objective_lgbm else 'binary',
                    max_depth=6,
                    num_leaves=64, # 2^6
                    n_estimators=220,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=80,
                    reg_alpha=0.3,
                    reg_lambda=0.9,
                n_jobs=-1,
                verbose=-1,
                random_state=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=3),
                feature_pre_filter=False,
                min_data_in_bin=1,
                )
            ]

            # Add XGBoost and RF for ensemble if available and requested
            if use_ensemble:
                if XGBOOST_AVAILABLE:
                    models.append(xgb.XGBClassifier(
                        max_depth=4,
                        n_estimators=200,
                        learning_rate=0.02,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.2,
                        reg_lambda=0.8,
                        n_jobs=-1,
                        verbosity=0,
                        random_state=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=4)
                    ))

                models.append(RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=42
                ))

    # Time-aware CV: use t1-aware purged K-fold splits with an embargo
    # proportional to the labeling horizon. For non-time-aware CV, fall
    # back to standard shuffled KFold.

    # Adaptive CV: If samples are scarce, reduce splits to prevent starvation
    if cv_splits > 2 and len(X_clean) < 200:
        new_splits = max(2, int(len(X_clean) / 50))  # Ensure at least 50 samples per fold roughly
        if new_splits < cv_splits:
            tprint_warning(f"⚠️ Adaptive CV: Reduced splits {cv_splits}→{new_splits} due to low n_samples={len(X_clean)}")
            cv_splits = new_splits

    if time_aware_cv:
        if event_durations is not None and market_index is not None and base_horizon_bars is not None:
            cv_splits_indices = _build_t1_aware_purged_splits_for_events(
                y=y_clean,
                event_durations=event_durations,
                market_index=market_index,
                cv_splits=cv_splits,
                base_horizon_bars=base_horizon_bars,
            )
        else:
            from src.utils.ml_common.labeling.meta_labeling import purged_kfold_splits

            n_samples = len(X_clean)
            embargo = base_horizon_bars or 5
            cv_splits_indices = purged_kfold_splits(
                n_samples=n_samples,
                n_splits=cv_splits,
                embargo=int(embargo),
            )
    else:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_splits, shuffle=True, 
                   random_state=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=6))
        cv_splits_indices = list(kf.split(X_clean))

    # Cost/return-aware sample weights with slight positive class bias (1.2x)
    returns_array = returns_clean.fillna(0.0).to_numpy(dtype=float)
    y_array = y_clean.to_numpy(dtype=float)

    sample_weights = np.ones_like(returns_array, dtype=float)

    # Conservative positive class up-weighting
    pos_mask = (y_array == 1.0)
    sample_weights[pos_mask] *= 1.2

    # Return-based weighting for label=1: linear in realized return, clipped
    try:
        finite_returns = returns_clean.replace([np.inf, -np.inf], np.nan).dropna().values
        if finite_returns.size >= 50:
            ret_clip = float(np.nanpercentile(np.abs(finite_returns), 95))
            ret_clip = max(ret_clip, 1e-4)
        else:
            ret_clip = 0.02
    except Exception:
        ret_clip = 0.02

    if ret_clip > 0:
        ret_for_weight = np.clip(np.maximum(returns_array, 0.0), 0.0, ret_clip)
        weight_factor = 1.0 + (ret_for_weight / ret_clip)
        sample_weights[pos_mask] *= weight_factor[pos_mask]

    # Optional: scale positive-class weights by signal strength so that
    # high-confidence signal configurations receive slightly higher weight
    # in the learnability scorer. We use the "signal_strength_all" meta-feature
    # when present in X_clean, normalised and clipped for robustness. The
    # overall strength of this effect is controlled by ``signal_strength_scale_max``.
    signal_strength = None
    if isinstance(X_clean, pd.DataFrame) and "signal_strength_all" in X_clean.columns:
        try:
            s = X_clean.loc[valid_mask, "signal_strength_all"].to_numpy(dtype=float)
            s = np.abs(s)
            # Robust scaling: use 90th percentile to avoid extreme values
            if np.isfinite(s).any():
                s_clean = s[np.isfinite(s)]
                if s_clean.size >= 10:
                    s_clip = float(np.nanpercentile(s_clean, 90))
                    s_clip = max(s_clip, 1e-6)
                else:
                    s_clip = float(np.nanmax(s_clean)) if s_clean.size > 0 else 1.0
                if s_clip <= 0:
                    s_clip = 1.0
                strength_norm = np.clip(s / s_clip, 0.0, 1.0)
                # Map to [1.0, signal_strength_scale_max] so HPO can tune the
                # influence of signal strength on sample weighting.
                scale_max = max(1.0, float(signal_strength_scale_max))
                scale_range = max(0.0, scale_max - 1.0)
                signal_weight = 1.0 + scale_range * strength_norm
                sample_weights[pos_mask] *= signal_weight[pos_mask]
        except Exception:
            # If anything goes wrong, fall back to return-only weighting.
            pass

    # Normalize weights for numerical stability
    mean_w = float(sample_weights.mean()) if sample_weights.size > 0 else 1.0
    if mean_w > 0:
        sample_weights = sample_weights / mean_w

    oof_probs_full = np.full(len(X_clean), np.nan, dtype=float)

    def _normalize_probabilities_or_scores(pred_array: np.ndarray) -> np.ndarray:
        """Map model outputs to well-behaved probabilities in [0, 1].

        This helper is robust to the case where ``predict_proba`` returns
        raw scores/logits (e.g. when using a custom LightGBM objective).
        - If the vector already looks like probabilities (within [0, 1]),
          it is simply clipped to [0, 1].
        - Otherwise it is treated as scores, robustly rescaled to avoid
          extreme magnitudes, then passed through a bounded sigmoid before
          clamping to [0, 1].
        """

        arr = np.asarray(pred_array, dtype=float).ravel()
        if arr.size == 0:
            return arr

        try:
            raw_min = float(np.nanmin(arr))
            raw_max = float(np.nanmax(arr))
        except Exception:
            # If anything goes wrong computing the range, fall back to the
            # raw array; downstream guards and diagnostics will still apply.
            return arr

        # Case 1: already looks like a probability vector
        if raw_min >= -1e-6 and raw_max <= 1.0 + 1e-6:
            return np.clip(arr, 0.0, 1.0)

        # Case 2: treat as scores/logits.
        # FIX (Step 373): Do NOT scale scores. Raw logits like -8e15 mean "Very Sure Zero".
        # Scaling them to -1.0 (relative to max) converts them to "Unsure" (0.27).
        # We simply clip them to a safe range for sigmoid.
        scores = np.clip(arr, -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-scores))
        probs = np.clip(probs, 0.0, 1.0)
        try:
            prob_min = float(np.nanmin(probs))
            prob_max = float(np.nanmax(probs))
        except Exception:
            return probs
        if prob_max - prob_min < 1e-3:
            finite_mask = np.isfinite(arr)
            if finite_mask.any():
                arr_finite = arr[finite_mask]
                try:
                    score_min = float(np.nanmin(arr_finite))
                    score_max = float(np.nanmax(arr_finite))
                except Exception:
                    score_min = 0.0
                    score_max = 0.0
                if score_max - score_min > 1e-6:
                    order = np.argsort(np.argsort(arr_finite))
                    denom = float(max(len(order) - 1, 1))
                    ranks = order.astype(float) / denom
                    probs_new = np.full_like(arr, 0.5, dtype=float)
                    probs_new[finite_mask] = ranks
                    probs = probs_new
                else:
                    probs = np.full_like(arr, 0.5, dtype=float)
            else:
                probs = np.full_like(arr, 0.5, dtype=float)
        return probs

    try:
        all_full_probs = []
        all_aucs = []
        all_model_oof_preds: List[np.ndarray] = []
        oof_indices: np.ndarray = np.array([], dtype=int)
        reference_fold_aucs: List[float] = []

        for i, model in enumerate(models):
            fold_aucs: List[float] = []
            model_oof_parts: List[np.ndarray] = []
            current_oof_indices: List[np.ndarray] = []

            for train_idx, test_idx in cv_splits_indices:
                X_train_cv = X_clean.iloc[train_idx]
                y_train_cv = y_clean.iloc[train_idx]
                X_test_cv = X_clean.iloc[test_idx]
                y_test_cv = y_clean.iloc[test_idx]

                # Per-fold constant-feature guard: if the training subset has
                # no non-constant numeric columns, LightGBM will effectively
                # see zero usable features and raise num_features()==0. Detect
                # this early and skip the fold instead.
                if isinstance(X_train_cv, pd.DataFrame):
                    try:
                        nunique_fold = X_train_cv.nunique(dropna=False)
                        non_constant_cols_fold = nunique_fold[nunique_fold > 1].index
                        if len(non_constant_cols_fold) == 0:
                            try:
                                y_train_counts = y_train_cv.value_counts(dropna=False).to_dict()
                            except Exception:
                                y_train_counts = {}
                            tprint(
                                "[LEARNABILITY_FOLD_EMPTY_FEATURES] Skipping CV fold with all-constant "
                                f"features (n_train={len(y_train_cv)}, y_counts={y_train_counts})",
                                "WARNING",
                            )
                            continue
                        if len(non_constant_cols_fold) < X_train_cv.shape[1]:
                            X_train_cv = X_train_cv.loc[:, non_constant_cols_fold]
                            X_test_cv = X_test_cv.loc[:, non_constant_cols_fold]
                    except Exception:
                        # If anything goes wrong here, fall back to using the
                        # original X_train_cv/X_test_cv; downstream guards and
                        # the outer try/except will still prevent hard failure.
                        pass

                w_train_cv = sample_weights[train_idx]

                # Ensure LightGBM and other models always see a purely numeric
                # matrix, independent of pandas dtypes.
                if isinstance(X_train_cv, pd.DataFrame):
                    X_train_mat = X_train_cv.to_numpy(dtype=float)
                    X_test_mat = X_test_cv.to_numpy(dtype=float)
                else:
                    X_train_mat = np.asarray(X_train_cv, dtype=float)
                    X_test_mat = np.asarray(X_test_cv, dtype=float)

                # ===== RESAMPLING (SMOTE / RandomOverSampler) =====
                if use_resampling:
                    try:
                        # Only resample if we have enough samples and imbalance
                        n_pos = np.sum(y_train_cv == 1)
                        n_neg = np.sum(y_train_cv == 0)
                        if n_pos > 10 and n_neg > 10:
                            # Try importing imblearn
                            try:
                                from imblearn.over_sampling import SMOTE
                                from imblearn.under_sampling import RandomUnderSampler
                                from imblearn.pipeline import Pipeline as ImbPipeline

                                # Hybrid strategy: SMOTE to 50% balance, then UnderSample majority to 1:1 if needed
                                # But simpler: Just SMOTE to 1.0 (balanced)
                                # Note: SMOTE can be slow on large dims.
                                resampler = SMOTE(random_state=42, k_neighbors=min(n_pos-1, 5))
                            except ImportError:
                                # DIY Random Oversampling
                                resampler = None

                            if resampler:
                                X_res, y_res = resampler.fit_resample(X_train_mat, y_train_cv)
                                X_train_mat = X_res
                                y_train_cv = y_res
                                # Reset weights for resampled data (assume uniform or re-calculate)
                                w_train_cv = np.ones(len(y_res), dtype=float)
                            else:
                                # Manual oversampling of minority class
                                minor_cls = 1 if n_pos < n_neg else 0
                                minor_indices = np.where(y_train_cv == minor_cls)[0]
                                major_indices = np.where(y_train_cv != minor_cls)[0]
                                diff = len(major_indices) - len(minor_indices)
                                if diff > 0:
                                    add_indices = np.random.choice(minor_indices, size=diff, replace=True)
                                    final_indices = np.concatenate([np.arange(len(y_train_cv)), add_indices])
                                    X_train_mat = X_train_mat[final_indices]
                                    y_train_cv = y_train_cv.iloc[final_indices] if isinstance(y_train_cv, pd.Series) else y_train_cv[final_indices]
                                    w_train_cv = np.ones(len(y_train_cv), dtype=float)

                    except Exception as e:
                        tprint(f"⚠️ Resampling failed in fold, continuing without: {e}", "WARNING")

                # Drop zero-variance columns (LightGBM can strip them and end up with 0 features)
                col_std = np.nanstd(X_train_mat, axis=0)
                nonzero_mask = col_std > 0
                if not np.any(nonzero_mask):
                    tprint(
                        f"[LEARNABILITY_FOLD_ZERO_VAR] All features zero-variance in fold; skipping (n_train={len(y_train_cv)})",
                        "WARNING",
                    )
                    continue
                if np.any(~nonzero_mask):
                    X_train_mat = X_train_mat[:, nonzero_mask]
                    X_test_mat = X_test_mat[:, nonzero_mask]

                # Final guard: check if the numeric matrix is empty or constant
                tprint(
                    f"[LEARNABILITY_DEBUG] X_train_mat shape: {X_train_mat.shape}, "
                    f"unique_vals_per_col: {[np.unique(X_train_mat[:, i]).size for i in range(min(3, X_train_mat.shape[1]))]}",
                    "INFO",
                )

                if X_train_mat.shape[1] == 0:
                    tprint(
                        f"[LEARNABILITY_EMPTY_MATRIX] X_train_mat has zero columns; skipping fold",
                        "WARNING",
                    )
                    continue

                # Check for NaN or infinite values that can cause LightGBM to fail
                if np.any(~np.isfinite(X_train_mat)):
                    nan_count = np.sum(~np.isfinite(X_train_mat))
                    tprint(
                        f"[LEARNABILITY_INVALID_VALUES] Found {nan_count} NaN/inf values in X_train_mat; cleaning and skipping fold",
                        "WARNING",
                    )
                    continue

                # Check if all features are constant in the training matrix
                if np.all(X_train_mat == X_train_mat[0, :], axis=0).all():
                    tprint(
                        f"[LEARNABILITY_CONSTANT_MATRIX] All features constant in X_train_mat; skipping fold",
                        "WARNING",
                    )
                    continue

                # LightGBM probe: verify the matrix is acceptable before main fit (looser params)
                try:
                    probe = lgb.LGBMClassifier(
                        boosting_type='gbdt',
                        objective='binary',
                        n_estimators=10,
                        max_depth=-1,
                        min_data_in_bin=1,
                        min_data_in_leaf=1,
                        learning_rate=0.2,
                        n_jobs=-1,
                        verbose=-1,
                        random_state=42,
                    )
                    probe.fit(X_train_mat, y_train_cv)
                except Exception as e_probe:
                    tprint(
                        f"[LEARNABILITY_LGB_PROBE_FAIL] LightGBM probe rejected fold: {str(e_probe)[:120]}...; skipping fold",
                        "WARNING",
                    )
                    continue

                try:
                    model.fit(X_train_mat, y_train_cv, sample_weight=w_train_cv)
                except TypeError:
                    model.fit(X_train_mat, y_train_cv)
                except Exception as e:
                    # Catch LightGBM errors and provide graceful fallback
                    if "num_features" in str(e) or "Check failed" in str(e):
                        tprint(
                            f"[LEARNABILITY_LIGHTGBM_ERROR] LightGBM feature error: {str(e)[:120]}...; trying logistic fallback",
                            "WARNING",
                        )
                        try:
                            fallback = LogisticRegression(
                                max_iter=200,
                                n_jobs=1,
                                penalty="l2",
                                solver="lbfgs",
                            )
                            fallback.fit(X_train_mat, y_train_cv)
                            model = fallback
                        except Exception as e_fallback:
                            tprint(
                                f"[LEARNABILITY_FALLBACK_FAIL] Logistic fallback also failed: {str(e_fallback)[:120]}...; skipping fold",
                                "WARNING",
                            )
                            continue
                    else:
                        # Re-raise other unexpected errors
                        raise

                proba_cv = model.predict_proba(X_test_mat)
                proba_cv = np.asarray(proba_cv)
                if proba_cv.ndim == 2:
                    if proba_cv.shape[1] >= 2:
                        y_proba_cv = proba_cv[:, 1]
                    else:
                        y_proba_cv = proba_cv[:, 0]
                else:
                    y_proba_cv = proba_cv.ravel()

                # Ensure fold-level predictions are well-scaled
                y_proba_cv = _normalize_probabilities_or_scores(y_proba_cv)

                model_oof_parts.append(y_proba_cv)
                if i == 0:
                    current_oof_indices.append(test_idx)

                try:
                    fold_auc = roc_auc_score(y_test_cv, y_proba_cv)
                    fold_aucs.append(fold_auc)
                except Exception:
                    pass

            if fold_aucs:
                mean_auc = float(np.mean(fold_aucs))
            else:
                mean_auc = 0.5
                tprint(
                    f"[LEARNABILITY_FALLBACK] No valid CV folds for AUC; setting mean_auc=0.5 "
                    f"(n_clean={len(y_clean)}, y_counts={y_clean_counts})",
                    "WARNING",
                )

            if model_oof_parts:
                all_model_oof_preds.append(np.concatenate(model_oof_parts))

            if i == 0 and current_oof_indices:
                oof_indices = np.concatenate(current_oof_indices)
                reference_fold_aucs = list(fold_aucs)

            # Global fit on full cleaned feature matrix. As above, always
            # convert to a dense numeric NumPy array before passing to the
            # underlying model to avoid any surprises with dtypes.
            if isinstance(X_clean, pd.DataFrame):
                X_full_mat = X_clean.to_numpy(dtype=float)
            else:
                X_full_mat = np.asarray(X_clean, dtype=float)

            # Final guard: check if the numeric matrix is empty or constant
            if X_full_mat.shape[1] == 0:
                tprint(
                    f"[LEARNABILITY_EMPTY_GLOBAL_MATRIX] X_full_mat has zero columns; skipping global fit",
                    "WARNING",
                )
                continue

            # Check if all features are constant in the full matrix
            if np.all(X_full_mat == X_full_mat[0, :], axis=0).all():
                tprint(
                    f"[LEARNABILITY_CONSTANT_GLOBAL_MATRIX] All features constant in X_full_mat; skipping global fit",
                    "WARNING",
                )
                continue

            # Drop zero-variance columns at global level (LightGBM may strip them)
            col_std_full = np.nanstd(X_full_mat, axis=0)
            nonzero_mask_full = col_std_full > 0
            if not np.any(nonzero_mask_full):
                tprint(
                    f"[LEARNABILITY_GLOBAL_ZERO_VAR] All features zero-variance in global matrix; skipping global fit",
                    "WARNING",
                )
                # Fallback: use constant 0.5 probabilities and AUC=0.5
                all_full_probs.append(np.full(len(y_clean), 0.5, dtype=float))
                all_aucs.append(0.5)
                continue
            if np.any(~nonzero_mask_full):
                X_full_mat = X_full_mat[:, nonzero_mask_full]

            # Guard against NaN/inf in global matrix
            if np.any(~np.isfinite(X_full_mat)):
                nan_count_full = int(np.sum(~np.isfinite(X_full_mat)))
                tprint(
                    f"[LEARNABILITY_GLOBAL_INVALID_VALUES] Found {nan_count_full} NaN/inf values in X_full_mat; skipping global fit",
                    "WARNING",
                )
                all_full_probs.append(np.full(len(y_clean), 0.5, dtype=float))
                all_aucs.append(0.5)
                continue

            # LightGBM probe on full matrix before global fit
            try:
                probe_full = lgb.LGBMClassifier(
                    boosting_type="gbdt",
                    objective="binary",
                    n_estimators=10,
                    max_depth=-1,
                    min_data_in_bin=1,
                    min_data_in_leaf=1,
                    learning_rate=0.2,
                    n_jobs=-1,
                    verbose=-1,
                    random_state=42,
                )
                probe_full.fit(X_full_mat, y_clean)
            except Exception as e_probe_full:
                tprint(
                    f"[LEARNABILITY_LGB_GLOBAL_PROBE_FAIL] LightGBM probe rejected global fit: {str(e_probe_full)[:120]}...; using constant probs",
                    "WARNING",
                )
                all_full_probs.append(np.full(len(y_clean), 0.5, dtype=float))
                all_aucs.append(0.5)
                continue

            try:
                model.fit(X_full_mat, y_clean, sample_weight=sample_weights)
            except TypeError:
                model.fit(X_full_mat, y_clean)
            except Exception as e_global:
                if "num_features" in str(e_global) or "Check failed" in str(e_global):
                    tprint(
                        f"[LEARNABILITY_LIGHTGBM_GLOBAL_ERROR] LightGBM global fit error: {str(e_global)[:120]}...; using constant probs",
                        "WARNING",
                    )
                    all_full_probs.append(np.full(len(y_clean), 0.5, dtype=float))
                    all_aucs.append(0.5)
                    continue
                else:
                    raise

            full_proba = model.predict_proba(X_full_mat)
            full_proba = np.asarray(full_proba)
            if full_proba.ndim == 2:
                if full_proba.shape[1] >= 2:
                    full_probs = full_proba[:, 1]
                else:
                    full_probs = full_proba[:, 0]
            else:
                full_probs = full_proba.ravel()

            # Normalize global predictions before any downstream calibration
            full_probs = _normalize_probabilities_or_scores(full_probs)

            all_full_probs.append(full_probs)
            all_aucs.append(mean_auc)

        if len(models) > 1:
            full_probs_array = np.array(all_full_probs)
            disagreement = np.std(full_probs_array, axis=0)
            avg_probs = np.mean(full_probs_array, axis=0)
            confidence_penalty = 1.0 - (disagreement * 0.5)
            final_probs = avg_probs * confidence_penalty + (1 - confidence_penalty) * 0.5
            final_probs = np.clip(final_probs, 0.0, 1.0)
            mean_auc = np.mean(all_aucs)
        else:
            final_probs = all_full_probs[0]
            mean_auc = all_aucs[0]

        if reference_fold_aucs:
            fold_aucs_array = np.asarray(reference_fold_aucs, dtype=float)
            std_auc_cv = float(np.std(fold_aucs_array))
        else:
            fold_aucs_array = np.asarray([], dtype=float)
            std_auc_cv = 0.0

        iso_reg = None
        calibrated_probs = final_probs

        # Move OOF construction earlier so it can be used for calibration
        if all_model_oof_preds and oof_indices.size > 0:
            try:
                oof_probs_array = np.array(all_model_oof_preds)
                if len(models) > 1:
                    avg_oof_probs = np.mean(oof_probs_array, axis=0)
                    disagreement_oof = np.std(oof_probs_array, axis=0)
                    confidence_penalty_oof = 1.0 - (disagreement_oof * 0.5)
                    final_oof_probs = avg_oof_probs * confidence_penalty_oof + (1 - confidence_penalty_oof) * 0.5
                    final_oof_probs = np.clip(final_oof_probs, 0.0, 1.0)
                else:
                    final_oof_probs = oof_probs_array[0]

                if final_oof_probs.shape[0] == oof_indices.shape[0]:
                    oof_probs_full = np.full(len(X_clean), np.nan, dtype=float)
                    oof_probs_full[oof_indices] = final_oof_probs
            except Exception:
                oof_probs_full = np.full(len(X_clean), np.nan, dtype=float)
        else:
            oof_probs_full = np.full(len(X_clean), np.nan, dtype=float)

        if model_complexity in ["medium", "strong"] and all_model_oof_preds and len(oof_indices) > 50:
            try:
                # Reuse final_oof_probs computed above
                y_oof = y_clean.iloc[oof_indices].to_numpy(dtype=float)

                iso_reg = IsotonicRegression(out_of_bounds='clip')

                # Prepare sample weights for OOF indices if available
                weights_oof = None
                if sample_weights is not None and len(sample_weights) == len(y_clean):
                    try:
                        weights_oof = sample_weights[oof_indices]
                    except Exception:
                        weights_oof = None

                valid_for_iso = np.isfinite(y_oof) & np.isfinite(final_oof_probs)
                if np.sum(valid_for_iso) > 50:
                    # Compute DYNAMIC balanced weights just for this fold/slice to force 50/50 prior
                    y_iso_fit = y_oof[valid_for_iso]
                    n_iso = len(y_iso_fit)
                    n_pos = np.sum(y_iso_fit == 1.0)
                    n_neg = np.sum(y_iso_fit == 0.0)

                    weights_iso = None

                    iso_reg.fit(final_oof_probs[valid_for_iso], y_iso_fit, sample_weight=weights_iso)

                    # FIX: Use OOF predictions for calibrated_probs to ensure valid diagnostics (No Leakage)
                    # We populate calibrated_probs with the calibrated OOF predictions.
                    # Indices not in OOF (e.g. purged) will remain NaN or be filled with in-sample fallback.
                    calibrated_probs = np.full(len(X_clean), np.nan, dtype=float)

                    # Calibrate the OOF predictions
                    calibrated_oof = iso_reg.predict(final_oof_probs)
                    calibrated_probs[oof_indices] = calibrated_oof

                    # Fallback for any missing indices (purged gaps): use in-sample predictions for continuity,
                    # but diagnostics usually respect NaNs or mask them.
                    missing_mask = np.isnan(calibrated_probs)
                    if np.any(missing_mask):
                        calibrated_probs[missing_mask] = iso_reg.predict(final_probs[missing_mask])

                    # Check for degenerate calibration (all same value)
                    if (calibrated_probs.max() - calibrated_probs.min()) < 0.05:
                        # Fallback to OOF probs (uncalibrated) if calibration collapsed
                         if np.any(~missing_mask):
                            calibrated_probs[~missing_mask] = final_oof_probs
                         if np.any(missing_mask):
                            calibrated_probs[missing_mask] = final_probs[missing_mask]

                else:
                    # Fallback to In-Sample if OOF didn't have enough valid samples (rare)
                    y_array_full = y_clean.to_numpy(dtype=float)
                    valid_in_sample = np.isfinite(y_array_full) & np.isfinite(final_probs)
                    if np.sum(valid_in_sample) > 50:
                        y_iso_fit = y_array_full[valid_in_sample]
                        # ... (same balancing logic as original) ...
                        n_iso = len(y_iso_fit)
                        n_pos = np.sum(y_iso_fit == 1.0)
                        n_neg = np.sum(y_iso_fit == 0.0)

                        weights_iso = None

                        iso_reg.fit(final_probs[valid_in_sample], y_iso_fit, sample_weight=weights_iso)
                        calibrated_probs = iso_reg.predict(final_probs)
                        if (calibrated_probs.max() - calibrated_probs.min()) < 0.05:
                            calibrated_probs = final_probs
            except Exception:
                calibrated_probs = final_probs
                iso_reg = None

        # Final safety: if calibration collapses to a near-constant vector while
        # mean_auc indicates a non-trivial signal, fall back to the uncalibrated
        # probabilities for all downstream diagnostics (MI, calibration curves,
        # meta-gating search). This prevents "Degenerate (Single Bin)" outputs
        # when the underlying model is actually informative.
        try:
            if isinstance(calibrated_probs, np.ndarray) and calibrated_probs.size > 0:
                prob_min = float(np.nanmin(calibrated_probs))
                prob_max = float(np.nanmax(calibrated_probs))
                prob_range = prob_max - prob_min
                if prob_range < 0.02 and mean_auc > 0.55:
                    tprint_warning(
                        "[CALIBRATION_DEGEN] Calibrated probabilities collapsed to near-constant "
                        f"(range={prob_range:.6f}) despite mean_auc={mean_auc:.3f}; "
                        "reverting to uncalibrated probabilities for diagnostics",
                    )
                    calibrated_probs = final_probs
                    iso_reg = None
                    candidates = []
                    try:
                        candidates.append(np.asarray(final_probs, dtype=float))
                    except Exception:
                        pass
                    try:
                        if oof_probs_full is not None:
                            candidates.append(np.asarray(oof_probs_full, dtype=float))
                    except Exception:
                        pass
                    for cand in candidates:
                        try:
                            finite_mask_cand = np.isfinite(cand)
                            if not finite_mask_cand.any():
                                continue
                            cand_min = float(np.nanmin(cand[finite_mask_cand]))
                            cand_max = float(np.nanmax(cand[finite_mask_cand]))
                            if cand_max - cand_min >= 1e-3:
                                calibrated_probs = cand
                                break
                        except Exception:
                            continue
        except Exception:
            pass

        learnability = mean_auc - (0.5 * std_auc_cv)

        return learnability, mean_auc, calibrated_probs, iso_reg, fold_aucs_array, oof_probs_full

    except Exception as e:
        tprint(f"[LEARNABILITY_EXCEPTION] Calibrated learnability scoring failed: {e}", "WARNING")
        return 0.0, 0.5, np.array([]), None, np.array([]), np.array([])


def run_leakage_sanity_check(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    top_k: int = 3,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "baseline_auc": None,
        "dropped_auc": None,
        "delta_auc": None,
        "top_features": [],
        "top_importances": [],
        "top_importance": None,
        "second_importance": None,
        "god_feature_suspected": False,
    }
    try:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            return result
        y_clean = y.dropna()
        if len(y_clean) < 100 or len(y_clean.unique()) < 2:
            return result
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return result
        X_aligned = X_num.reindex(y_clean.index).fillna(0.0)
        if X_aligned.shape[1] == 0:
            return result

        base_model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=120,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=60,
            reg_alpha=0.3,
            reg_lambda=0.9,
            feature_pre_filter=False,
            min_data_in_bin=3,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
        )
        base_model.fit(X_aligned, y_clean)
        preds_base = base_model.predict_proba(X_aligned)[:, 1]
        baseline_auc = float(roc_auc_score(y_clean, preds_base))

        imp = permutation_importance(
            base_model,
            X_aligned,
            y_clean,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state,
        )
        importances_mean = imp.importances_mean
        if importances_mean is None or importances_mean.size == 0:
            return result

        indices = np.argsort(importances_mean)[::-1]
        k = min(max(1, top_k), indices.size)
        top_idx = indices[:k]
        feature_names = X_aligned.columns.to_list()
        top_features = [feature_names[i] for i in top_idx]
        top_importances = [float(importances_mean[i]) for i in top_idx]

        X_dropped = X_aligned.drop(columns=top_features, errors="ignore")
        if X_dropped.shape[1] == 0:
            return {
                "baseline_auc": baseline_auc,
                "dropped_auc": None,
                "delta_auc": None,
                "top_features": top_features,
            }

        drop_model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=120,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=60,
            reg_alpha=0.3,
            reg_lambda=0.9,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
        )
        drop_model.fit(X_dropped, y_clean)
        preds_drop = drop_model.predict_proba(X_dropped)[:, 1]
        dropped_auc = float(roc_auc_score(y_clean, preds_drop))
        delta_auc = baseline_auc - dropped_auc

        result["baseline_auc"] = baseline_auc
        result["dropped_auc"] = dropped_auc
        result["delta_auc"] = delta_auc
        result["top_features"] = top_features
        result["top_importances"] = top_importances
        if top_importances:
            result["top_importance"] = float(top_importances[0])
            if len(top_importances) > 1:
                result["second_importance"] = float(top_importances[1])
        top_imp_val = result["top_importance"]
        second_imp_val = result["second_importance"]
        if top_imp_val is not None and delta_auc is not None:
            god_by_level = top_imp_val >= 0.25
            if second_imp_val is not None and second_imp_val > 0:
                ratio = top_imp_val / second_imp_val
            else:
                ratio = float("inf")
            god_by_ratio = ratio >= 3.0
            min_delta_auc = 0.01
            result["god_feature_suspected"] = bool((god_by_level or god_by_ratio) and (delta_auc >= min_delta_auc))
        return result
    except Exception:
        return result


def run_lag1_stress_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "auc_base": None,
        "auc_lag1": None,
        "auc_diff": None,
        "lookahead_suspected": False,
    }
    try:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            return result
        y_clean = y.dropna()
        if len(y_clean) < 100 or len(y_clean.unique()) < 2:
            return result
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            return result
        X_aligned = X_num.reindex(y_clean.index).fillna(0.0)
        if X_aligned.shape[1] == 0:
            return result

        base_model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=120,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=60,
            reg_alpha=0.3,
            reg_lambda=0.9,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
        )
        base_model.fit(X_aligned, y_clean)
        base_probs = base_model.predict_proba(X_aligned)[:, 1]
        auc_base = float(roc_auc_score(y_clean, base_probs))
        X_lag = X_aligned.shift(1).dropna()
        if X_lag.empty:
            return result
        y_lag = y_clean.reindex(X_lag.index).dropna()
        X_lag = X_lag.reindex(y_lag.index)
        if len(y_lag) < 100 or len(y_lag.unique()) < 2:
            return result

        lag_model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            max_depth=3,
            n_estimators=120,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=60,
            reg_alpha=0.3,
            reg_lambda=0.9,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state,
        )
        lag_model.fit(X_lag, y_lag)
        lag_probs = lag_model.predict_proba(X_lag)[:, 1]
        auc_lag1 = float(roc_auc_score(y_lag, lag_probs))
        auc_diff = auc_base - auc_lag1
        result["auc_base"] = auc_base
        result["auc_lag1"] = auc_lag1
        result["auc_diff"] = auc_diff
        lookahead = bool(auc_base >= 0.7 and auc_diff >= 0.1)
        result["lookahead_suspected"] = lookahead
        return result
    except Exception:
        return result


def compute_dummy_baseline_auc(
    volatility: pd.Series,
    y: pd.Series,
    window: int = 64,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "auc_dummy": None,
        "auc_dummy_raw": None,
        "n_samples": 0,
    }
    try:
        if not isinstance(volatility, pd.Series) or not isinstance(y, pd.Series):
            return result
        y_clean = y.dropna()
        if len(y_clean) < 50 or len(y_clean.unique()) < 2:
            return result
        vol_aligned = volatility.reindex(y_clean.index)
        if vol_aligned.isna().all():
            return result
        vol_aligned = vol_aligned.astype(float)
        vol_aligned = vol_aligned.fillna(method="ffill").fillna(method="bfill")
        if vol_aligned.isna().all():
            return result
        min_periods = max(10, window // 4)
        vol_ma = vol_aligned.rolling(window, min_periods=min_periods).mean()
        score = (vol_aligned - vol_ma).fillna(0.0)
        auc_raw = float(roc_auc_score(y_clean, score))
        auc_abs = auc_raw
        if auc_abs < 0.5:
            auc_abs = 1.0 - auc_abs
        result["auc_dummy_raw"] = auc_raw
        result["auc_dummy"] = auc_abs
        result["n_samples"] = int(len(y_clean))
        return result
    except Exception:
        return result


def _discrete_mi(x: pd.Series, y: pd.Series) -> float:
    """Mutual information for discrete variables with robust guards.

    Returns 0.0 (instead of NaN) for degenerate or empty cases so that
    downstream diagnostics never propagate NaNs.
    """
    valid = x.notna() & y.notna()
    if not valid.any():
        return 0.0

    xv = x.loc[valid]
    yv = y.loc[valid]

    # If either side has fewer than 2 unique values, MI is zero by definition
    if xv.nunique() < 2 or yv.nunique() < 2:
        return 0.0

    joint = pd.crosstab(xv, yv, normalize=True)
    if joint.empty:
        return 0.0

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi_val = 0.0
    for xi in joint.index:
        for yi in joint.columns:
            pxy = float(joint.loc[xi, yi])
            if pxy <= 0.0:
                continue
            denom = float(px[xi] * py[yi])
            if denom <= 0.0:
                continue
            mi_val += pxy * np.log(pxy / denom)

    if not np.isfinite(mi_val):
        return 0.0
    return float(mi_val)


def compute_underfit_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    time_aware_cv: bool = True,
    use_purged_splits: bool = False,
    event_durations: Optional[pd.Series] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
    base_horizon_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute underfit diagnostics to assess room for model improvement.

    Computes:
    1. Learning curves at different data fractions
    2. Learning curves at different depths
    3. Feature importance concentration
    4. Probe vs deeper model AUC comparison

    Args:
        X: Feature matrix
        y: Binary labels
        cv_splits: Number of CV splits
        time_aware_cv: Use TimeSeriesSplit

    Returns:
        Dictionary with diagnostic metrics
    """
    from sklearn.model_selection import cross_val_score, learning_curve

    diagnostics = {
        "learning_curve_fractions": {},
        "learning_curve_depths": {},
        "feature_importance_concentration": None,
        "feature_group_importance": None,
        "top_feature_importances": [],
        "probe_vs_deep_auc_diff": None,
        "is_underfit": False,
        "underfit_indicators": [],
    }

    # Remove NaN labels
    valid_mask = ~y.isna()
    X_num = X.select_dtypes(include=[np.number]) if isinstance(X, pd.DataFrame) else X
    if isinstance(X_num, pd.DataFrame) and X_num.empty:
        return diagnostics

    X_clean = X_num[valid_mask].fillna(0)
    y_clean = y[valid_mask]

    if len(y_clean) < 100 or len(y_clean.unique()) < 2:
        return diagnostics

    # Optional: reuse t1-aware purged splits from HPO when event information is provided.
    # Default to False to avoid NameError downstream; caller can opt-in.
    if 'use_purged_splits' not in locals():
        use_purged_splits = False

    purged_splits = None
    if (
        use_purged_splits
        and event_durations is not None
        and market_index is not None
        and base_horizon_bars is not None
    ):
        try:
            purged_splits = _build_t1_aware_purged_splits_for_events(
                y=y_clean,
                event_durations=event_durations,
                market_index=market_index,
                cv_splits=cv_splits,
                base_horizon_bars=base_horizon_bars,
            )
        except Exception:
            purged_splits = None

    # Time-aware CV (fallback when purged splits are not requested or unavailable)
    if time_aware_cv and purged_splits is None:
        cv = TimeSeriesSplit(n_splits=cv_splits)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    try:
        # 1. Learning curves with data fractions (20%, 40%, 60%, 80%, 100%)
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        fraction_aucs = []

        probe_model = lgb.LGBMClassifier(
            max_depth=3, n_estimators=50, learning_rate=0.1,
            n_jobs=-1, verbose=-1, random_state=42
        )

        for frac in fractions:
            n_samples = int(len(X_clean) * frac)
            if n_samples < 50:
                continue

            X_frac = X_clean.iloc[:n_samples]
            y_frac = y_clean.iloc[:n_samples]

            if len(y_frac.unique()) < 2:
                continue

            try:
                # For fraction curves we keep simple CV; purged splits are defined on
                # the full event set and are not trivially compatible with subsets.
                scores = cross_val_score(
                    probe_model, X_frac, y_frac,
                    cv=min(cv_splits, len(y_frac) // 20),
                    scoring='roc_auc', n_jobs=-1
                )
                diagnostics["learning_curve_fractions"][frac] = float(scores.mean())
                fraction_aucs.append(scores.mean())
            except Exception:
                pass

        # Check if AUC keeps rising without plateau (underfit indicator)
        if len(fraction_aucs) >= 3:
            # If last improvement > 2%, likely underfit
            if fraction_aucs[-1] - fraction_aucs[-2] > 0.02:
                diagnostics["is_underfit"] = True
                diagnostics["underfit_indicators"].append("AUC still rising with more data")

        # 2. Learning curves with different depths (3, 5, 7)
        depths = [3, 5, 7]
        depth_aucs = []

        for depth in depths:
            model = lgb.LGBMClassifier(
                max_depth=depth, n_estimators=100, learning_rate=0.05,
                n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                cv_arg = purged_splits if purged_splits is not None else cv
                scores = cross_val_score(model, X_clean, y_clean, cv=cv_arg, scoring='roc_auc', n_jobs=-1)
                auc = float(scores.mean())
                diagnostics["learning_curve_depths"][depth] = auc
                depth_aucs.append(auc)
            except Exception:
                pass

        # 3. Probe vs deeper model comparison
        if len(depth_aucs) >= 2:
            probe_auc = depth_aucs[0]  # depth=3
            best_deep_auc = max(depth_aucs[1:])  # best of deeper models

            auc_diff = best_deep_auc - probe_auc
            diagnostics["probe_vs_deep_auc_diff"] = float(auc_diff)

            # If deeper model improves >5%, probe is underfit
            if auc_diff > 0.05:
                diagnostics["is_underfit"] = True
                diagnostics["underfit_indicators"].append(f"Deeper model +{auc_diff:.1%} AUC")

        # 4. Feature importance concentration (top 5 features)
        try:
            final_model = lgb.LGBMClassifier(
                max_depth=5, n_estimators=100, learning_rate=0.05,
                n_jobs=-1, verbose=-1, random_state=42
            )
            final_model.fit(X_clean, y_clean)

            importances = final_model.feature_importances_
            sorted_imp = np.sort(importances)[::-1]
            total_imp = importances.sum()
            feature_names = list(X_clean.columns) if isinstance(X_clean, pd.DataFrame) else None

            if total_imp > 0:
                top_5_concentration = sorted_imp[:5].sum() / total_imp
                diagnostics["feature_importance_concentration"] = float(top_5_concentration)

                if feature_names is not None and len(feature_names) == len(importances):
                    group_totals = {"volatility": 0.0, "signal": 0.0, "other": 0.0}
                    top_features: list[dict[str, Any]] = []
                    idx_sorted = np.argsort(importances)[::-1]
                    top_k = min(20, len(idx_sorted))
                    for idx in idx_sorted[:top_k]:
                        name = str(feature_names[idx])
                        share = float(importances[idx]) / float(total_imp)
                        lname = name.lower()
                        if any(tok in lname for tok in ("vol", "atr", "std", "var", "range")):
                            group = "volatility"
                        elif any(tok in lname for tok in ("signal", "alpha", "entry", "meta", "prob")):
                            group = "signal"
                        else:
                            group = "other"
                        group_totals[group] += share
                        top_features.append({"name": name, "importance": share})
                    diagnostics["feature_group_importance"] = group_totals
                    diagnostics["top_feature_importances"] = top_features

                # If top 5 features dominate 80%+, model is only scratching surface
                if top_5_concentration > 0.8:
                    diagnostics["is_underfit"] = True
                    diagnostics["underfit_indicators"].append(
                        f"Top 5 features = {top_5_concentration:.1%} importance"
                    )
        except Exception:
            pass

    except Exception as e:
        tprint(f"⚠️ Underfit diagnostics failed: {e}", "WARNING")

    return diagnostics


def _label_terciles_causal(x: pd.Series, *, train_frac: float = 0.7) -> Tuple[pd.Series, Dict[str, float]]:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    non_null = s.dropna()
    if non_null.empty:
        return pd.Series(index=s.index, dtype=object), {"q33": float("nan"), "q67": float("nan")}

    split_idx = int(max(1, min(len(non_null), round(len(non_null) * float(train_frac)))))
    train_slice = non_null.iloc[:split_idx]
    try:
        q33 = float(train_slice.quantile(1.0 / 3.0))
        q67 = float(train_slice.quantile(2.0 / 3.0))
    except Exception:
        q33 = float(non_null.quantile(1.0 / 3.0))
        q67 = float(non_null.quantile(2.0 / 3.0))

    labels = pd.Series(index=s.index, dtype=object)
    try:
        labels.loc[s <= q33] = "low"
        labels.loc[(s > q33) & (s <= q67)] = "medium"
        labels.loc[s > q67] = "high"
    except Exception:
        pass
    return labels, {"q33": q33, "q67": q67}


def _compute_volatility_1d_from_market(market_data: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(market_data.get("close"), errors="coerce")
    log_ret = np.log(close).diff()
    return log_ret.rolling(window=96, min_periods=16).std()


def _compute_trend_strength_from_market(market_data: pd.DataFrame, *, horizon_bars: int = 96) -> pd.Series:
    close = pd.to_numeric(market_data.get("close"), errors="coerce")
    horizon_bars = int(max(2, horizon_bars))
    return close.pct_change(horizon_bars).abs()


def _build_event_regime_labels(
    *,
    market_data: pd.DataFrame,
    event_index: pd.Index,
    config: Dict[str, Any],
) -> Dict[str, pd.Series]:
    train_frac = float(config.get("regime_label_train_frac", 0.7))
    trend_h = int(config.get("trend_regime_horizon_bars", 96))

    vol_1d = _compute_volatility_1d_from_market(market_data)
    trend_strength = _compute_trend_strength_from_market(market_data, horizon_bars=trend_h)

    vol_regime_all, _ = _label_terciles_causal(vol_1d, train_frac=train_frac)
    trend_regime_all, _ = _label_terciles_causal(trend_strength, train_frac=train_frac)

    vol_regime = vol_regime_all.reindex(event_index)
    trend_regime = trend_regime_all.reindex(event_index)

    combined = pd.Series(index=pd.Index(event_index), dtype=object)
    try:
        v = vol_regime.astype(object)
        t = trend_regime.astype(object)
        combined = ("vol_" + v.fillna("na").astype(str) + "__trend_" + t.fillna("na").astype(str)).astype(object)
        combined.index = pd.Index(event_index)
    except Exception:
        pass

    return {
        "volatility_regime": vol_regime,
        "trend_regime": trend_regime,
        "combined_regime": combined,
    }


def _compute_fold_metrics_from_oof(
    *,
    X: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    returns: np.ndarray,
    threshold: float,
    days_span: float,
    transaction_cost: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        cv = TimeSeriesSplit(n_splits=5)
        for fold_idx, (_, te_idx) in enumerate(cv.split(X)):
            try:
                te_idx = np.asarray(te_idx, dtype=int)
                p = np.asarray(probs, dtype=float)[te_idx]
                y = np.asarray(y_true, dtype=float)[te_idx]
                r = np.asarray(returns, dtype=float)[te_idx]
                mask = np.isfinite(p) & np.isfinite(y) & np.isfinite(r)
                if int(np.sum(mask)) < 20:
                    continue
                p = p[mask]
                y = y[mask]
                r = r[mask]
                auc = float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else None
                trade_mask = p >= float(threshold)
                n_trades = int(np.sum(trade_mask))
                mean_ret = float(np.mean(r[trade_mask])) if n_trades > 0 else 0.0
                net_mean_ret = float(mean_ret - float(transaction_cost)) if n_trades > 0 else 0.0
                win_rate = float(np.mean(r[trade_mask] > 0.0)) if n_trades > 0 else 0.0
                trades_per_day = float(n_trades) / float(max(days_span, 1.0))
                out.append(
                    {
                        "fold": int(fold_idx),
                        "auc": auc,
                        "n_test": int(len(p)),
                        "n_trades": int(n_trades),
                        "trades_per_day": float(trades_per_day),
                        "mean_return": float(mean_ret),
                        "net_pnl_per_trade": float(net_mean_ret),
                        "win_rate": float(win_rate),
                    }
                )
            except Exception:
                continue
    except Exception:
        return out
    return out


def _compute_metrics_by_regime(
    *,
    y_true: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    returns: np.ndarray,
    base_thr: float,
    transaction_cost: float,
    regime_labels: pd.Series,
    days_span: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if regime_labels is None or regime_labels.empty:
        return out

    lab = regime_labels.astype(object)
    for reg_val in pd.unique(lab.dropna()):
        try:
            mask = (lab == reg_val).to_numpy(dtype=bool)
            n_events = int(np.sum(mask))
            if n_events < 20:
                continue

            r_r = np.asarray(returns, dtype=float)[mask]
            valid = np.isfinite(r_r)

            y_r = None
            p_r = None
            if y_true is not None:
                y_r = np.asarray(y_true, dtype=float)[mask]
                valid = valid & np.isfinite(y_r)
            if probs is not None:
                p_r = np.asarray(probs, dtype=float)[mask]
                valid = valid & np.isfinite(p_r)

            if int(np.sum(valid)) < 20:
                continue
            r_r = r_r[valid]
            if y_r is not None:
                y_r = y_r[valid]
            if p_r is not None:
                p_r = p_r[valid]

            trade_mask = np.full(len(r_r), True, dtype=bool)
            if p_r is not None:
                trade_mask = p_r >= float(base_thr)

            n_trades = int(np.sum(trade_mask))
            if n_trades <= 0:
                out[str(reg_val)] = {"n_events": int(n_events), "n_trades": 0}
                continue

            trade_returns = r_r[trade_mask]
            trade_returns = trade_returns[np.isfinite(trade_returns)]
            if trade_returns.size <= 0:
                out[str(reg_val)] = {"n_events": int(n_events), "n_trades": 0}
                continue
            mean_ret = float(np.mean(trade_returns))
            net_mean_ret = float(mean_ret - float(transaction_cost))
            win_rate = float(np.mean(trade_returns > 0.0))

            sharpe = 0.0
            if trade_returns.size > 1:
                tr_std = float(np.std(trade_returns, ddof=1))
                if tr_std > 0.0:
                    trades_per_day = float(n_trades) / float(max(days_span, 1.0))
                    trades_per_year = float(trades_per_day) * 365.0
                    sharpe = float(mean_ret / (tr_std + 1e-8) * np.sqrt(max(trades_per_year, 1e-8)))
            try:
                sharpe = float(np.clip(float(sharpe), -20.0, 20.0))
            except Exception:
                sharpe = 0.0

            auc_r = None
            if y_r is not None and p_r is not None:
                try:
                    if len(np.unique(y_r)) >= 2:
                        auc_r = float(roc_auc_score(y_r, p_r))
                except Exception:
                    auc_r = None

            out[str(reg_val)] = {
                "n_events": int(n_events),
                "n_trades": int(n_trades),
                "trades_per_day": float(n_trades) / float(max(days_span, 1.0)),
                "mean_return": float(mean_ret),
                "net_pnl_per_trade": float(net_mean_ret),
                "win_rate": float(win_rate),
                "sharpe": float(sharpe),
                "auc": auc_r,
            }
        except Exception:
            continue
    return out


def compute_filtering_inflation_diagnostics(
    X: pd.DataFrame,
    y_full: pd.Series,
    y_filtered: pd.Series,
    realized_returns: pd.Series,
    volatility: pd.Series,
    probabilities: np.ndarray,
    cv_splits: int = 3,
    time_aware_cv: bool = True,
    event_durations: Optional[pd.Series] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
    base_horizon_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute diagnostics to detect AUC inflation from aggressive filtering.

    Tests:
    1. AUC on full vs filtered labels - if filtered AUC >> full AUC, filtering inflates
    2. AUC by return magnitude bucket - if AUC high only in large-move bins, labels dominated
    3. Class overlap metrics - how separable are retained vs discarded events
    4. Precision@K drop - if precision collapses as K increases, model only good on easy cases

    Args:
        X: Feature matrix (aligned to full label space)
        y_full: Labels before filtering (pre-quantile, using economic floor only)
        y_filtered: Labels after all filtering (quantile-based, final)
        realized_returns: Realized returns per event
        volatility: Volatility series for normalization
        probabilities: Model predicted probabilities (aligned to filtered labels)
        cv_splits: Number of CV splits for AUC computation
        time_aware_cv: Use TimeSeriesSplit

    Returns:
        Dictionary with diagnostic metrics
    """
    from sklearn.metrics import roc_auc_score, precision_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_predict

    diagnostics = {
        # 1. Full vs Filtered AUC comparison
        "auc_full": None,
        "auc_filtered": None,
        "auc_inflation": None,  # filtered - full
        "filtering_is_major_contributor": False,  # True if inflation > 0.08

        # 2. AUC by return magnitude bucket
        "auc_by_return_bucket": {},
        "auc_dominated_by_large_moves": False,

        # 3. Retention metrics
        "n_full_events": 0,
        "n_filtered_events": 0,
        "retention_rate": 0.0,

        # 4. Precision@K analysis
        "precision_at_k": {},
        "precision_collapse_detected": False,
    }

    try:
        # Align indices
        common_idx = y_full.index.intersection(y_filtered.index)
        if len(common_idx) < 50:
            return diagnostics

        # ===== 1. AUC on Full vs Filtered Labels =====
        full_mask = ~y_full.isna()
        filtered_mask = ~y_filtered.isna()

        n_full = int(full_mask.sum())
        n_filtered = int(filtered_mask.sum())
        diagnostics["n_full_events"] = n_full
        diagnostics["n_filtered_events"] = n_filtered
        diagnostics["retention_rate"] = n_filtered / max(n_full, 1)

        # Compute AUC on filtered labels (what we normally report)
        if n_filtered >= 50 and len(np.unique(y_filtered[filtered_mask])) >= 2:
            if probabilities is not None and len(probabilities) == len(y_filtered):
                try:
                    probs_filtered = probabilities[filtered_mask.values]
                    y_filt_vals = y_filtered[filtered_mask].values
                    diagnostics["auc_filtered"] = float(roc_auc_score(y_filt_vals, probs_filtered))
                except Exception:
                    pass

        # Compute AUC on full labels (before quantile filtering)
        # Need to retrain/predict on full label space
        if n_full >= 50 and len(np.unique(y_full[full_mask])) >= 2:
            try:
                X_num = X.select_dtypes(include=[np.number])
                X_full = X_num[full_mask].fillna(0)
                y_full_vals = y_full[full_mask]

                if len(y_full_vals) >= 50 and len(y_full_vals.unique()) >= 2:
                    probe = lgb.LGBMClassifier(
                        max_depth=3, n_estimators=50, learning_rate=0.1,
                        n_jobs=-1, verbose=-1, random_state=42
                    )

                    auc_full_local = None
                    min_full_predictions = 20

                    if time_aware_cv and event_durations is not None and market_index is not None and base_horizon_bars is not None:
                        splits = _build_t1_aware_purged_splits_for_events(
                            y=y_full_vals,
                            event_durations=event_durations,
                            market_index=market_index,
                            cv_splits=cv_splits,
                            base_horizon_bars=base_horizon_bars,
                        )
                        probs_full = np.full(len(y_full_vals), np.nan, dtype=float)
                        for train_idx, test_idx in splits:
                            X_train_cv = X_full.iloc[train_idx]
                            y_train_cv = y_full_vals.iloc[train_idx]
                            X_test_cv = X_full.iloc[test_idx]
                            y_test_cv = y_full_vals.iloc[test_idx]
                            if len(np.unique(y_train_cv)) < 2 or len(np.unique(y_test_cv)) < 2:
                                continue
                            probe.fit(X_train_cv, y_train_cv)
                            probs_full[test_idx] = probe.predict_proba(X_test_cv)[:, 1]
                        valid_mask_full = np.isfinite(probs_full)
                        n_valid = int(valid_mask_full.sum())
                        if n_valid >= min_full_predictions:
                            auc_full_local = float(
                                roc_auc_score(y_full_vals.iloc[valid_mask_full], probs_full[valid_mask_full])
                            )
                        else:
                            tprint_info(
                                f" Filtering diagnostics: insufficient t1-aware predictions for full-label AUC "
                                f"(n_valid={n_valid}, min_required={min_full_predictions}); falling back to standard CV"
                            )

                    if auc_full_local is None:
                        if time_aware_cv:
                            cv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, len(y_full_vals) // 20)))
                        else:
                            from sklearn.model_selection import KFold
                            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
                        probs_full = cross_val_predict(probe, X_full, y_full_vals, cv=cv, method='predict_proba')[:, 1]
                        if len(np.unique(y_full_vals)) >= 2:
                            auc_full_local = float(roc_auc_score(y_full_vals, probs_full))

                    if auc_full_local is not None:
                        diagnostics["auc_full"] = auc_full_local
            except Exception as exc:
                tprint_warning(f"6a0 Filtering inflation full-label AUC computation failed: {exc}")

        # Compute inflation
        if diagnostics["auc_filtered"] is not None and diagnostics["auc_full"] is not None:
            diagnostics["auc_inflation"] = diagnostics["auc_filtered"] - diagnostics["auc_full"]
            diagnostics["filtering_is_major_contributor"] = diagnostics["auc_inflation"] > 0.08

        # ===== 2. AUC by Return Magnitude Bucket =====
        try:
            vol_aligned = volatility.reindex(realized_returns.index).fillna(volatility.median())
            vol_scaled = realized_returns / (vol_aligned.abs() + 1e-8)
            abs_vol_scaled = vol_scaled.abs()

            # Define buckets by sigma
            buckets = [
                ("0-0.1σ", 0.0, 0.1),
                ("0.1-0.3σ", 0.1, 0.3),
                ("0.3-0.8σ", 0.3, 0.8),
                ("0.8-1.5σ", 0.8, 1.5),
                ("1.5σ+", 1.5, float('inf')),
            ]

            bucket_aucs = {}
            small_move_aucs = []
            large_move_aucs = []

            for bucket_name, low, high in buckets:
                bucket_mask = (abs_vol_scaled >= low) & (abs_vol_scaled < high) & filtered_mask
                n_bucket = int(bucket_mask.sum())

                if n_bucket >= 30:
                    try:
                        y_bucket = y_filtered[bucket_mask]
                        if len(y_bucket.unique()) >= 2 and probabilities is not None:
                            probs_bucket = probabilities[bucket_mask.values]
                            auc_bucket = float(roc_auc_score(y_bucket.values, probs_bucket))
                            bucket_aucs[bucket_name] = {"auc": auc_bucket, "n": n_bucket}

                            # Track for dominance check
                            if high <= 0.3:
                                small_move_aucs.append(auc_bucket)
                            elif low >= 0.8:
                                large_move_aucs.append(auc_bucket)
                    except Exception:
                        bucket_aucs[bucket_name] = {"auc": None, "n": n_bucket}
                else:
                    bucket_aucs[bucket_name] = {"auc": None, "n": n_bucket}

            diagnostics["auc_by_return_bucket"] = bucket_aucs

            # Check if dominated by large moves
            if small_move_aucs and large_move_aucs:
                avg_small = np.mean(small_move_aucs)
                avg_large = np.mean(large_move_aucs)
                # Dominated if small AUC is 0.52-0.56 and large is 0.65+
                if avg_small < 0.58 and avg_large > 0.63:
                    diagnostics["auc_dominated_by_large_moves"] = True

        except Exception:
            pass

        # ===== 4. Precision@K Drop Analysis =====
        try:
            if probabilities is not None and n_filtered >= 50:
                y_filt = y_filtered[filtered_mask].values
                probs_filt = probabilities[filtered_mask.values]

                # Sort by probability descending
                sorted_idx = np.argsort(probs_filt)[::-1]
                y_sorted = y_filt[sorted_idx]

                # Compute precision at different K percentiles
                precision_at_k = {}
                for k_pct in [10, 20, 30, 50, 70]:
                    k = max(1, int(len(y_sorted) * k_pct / 100))
                    precision_k = float(y_sorted[:k].mean())
                    precision_at_k[f"top_{k_pct}%"] = precision_k

                diagnostics["precision_at_k"] = precision_at_k

                # Check for collapse: top 10% precision is good but top 50% collapses
                if "top_10%" in precision_at_k and "top_50%" in precision_at_k:
                    if precision_at_k["top_10%"] > 0.65 and precision_at_k["top_50%"] < 0.55:
                        diagnostics["precision_collapse_detected"] = True

        except Exception:
            pass

    except Exception as e:
        tprint(f"⚠️ Filtering inflation diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    realized_returns: np.ndarray,
    transaction_cost: float = 0.003,
    n_bins: int = 10,
    regime_score: Optional[np.ndarray] = None,
    use_linear_adaptive_gating: bool = True,
) -> Dict[str, Any]:
    """Compute calibration diagnostics for the meta-model.

    Includes:
    - Brier score
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Reliability diagram data
    - Precision per probability bin (especially ≥0.6, ≥0.7, ≥0.8)
    - Expected net P&L by probability bucket

    Args:
        y_true: True binary labels
        probabilities: Model predicted probabilities
        realized_returns: Realized returns for P&L calculation
        transaction_cost: Transaction cost per trade
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics
    """
    diagnostics = {
        "brier_score": None,
        "ece": None,  # Expected Calibration Error
        "mce": None,  # Maximum Calibration Error
        "reliability_diagram": [],  # List of (mean_pred, mean_actual, count) per bin
        "precision_per_bin": {},
        "fpr_per_bin": {},
        "expected_pnl_per_bin": {},
        "is_well_calibrated": False,
        "prob_range_raw": None,
        "prob_range_clamped": None,
        "degenerate_calibration": False,
    }

    try:
        # Coerce inputs to float arrays and align lengths safely
        if probabilities is None:
            return diagnostics

        y_arr = np.asarray(y_true, dtype=float).ravel()
        probs_arr = np.asarray(probabilities, dtype=float).ravel()
        returns_arr = None
        if realized_returns is not None:
            returns_arr = np.asarray(realized_returns, dtype=float).ravel()
        regime_arr = None
        if regime_score is not None:
            regime_arr = np.asarray(regime_score).ravel()

        n = min(len(y_arr), len(probs_arr))
        if returns_arr is not None:
            n = min(n, len(returns_arr))
        if regime_arr is not None:
            n = min(n, len(regime_arr))
        if n < 50:
            return diagnostics

        y_arr = y_arr[:n]
        probs_arr = probs_arr[:n]
        if returns_arr is not None:
            returns_arr = returns_arr[:n]
        if regime_arr is not None:
            regime_arr = regime_arr[:n]

        # Remove NaN/inf values and align all arrays
        valid_mask = np.isfinite(y_arr) & np.isfinite(probs_arr)
        if returns_arr is not None:
            valid_mask &= np.isfinite(returns_arr)
        if regime_arr is not None:
            valid_mask &= np.isfinite(regime_arr)

        y = y_arr[valid_mask]
        probs = probs_arr[valid_mask]
        returns = returns_arr[valid_mask] if returns_arr is not None else None
        regimes = regime_arr[valid_mask] if regime_arr is not None else None

        if len(y) < 50:
            return diagnostics

        # Track raw probability range before any transformation
        raw_min = float(np.nanmin(probs)) if probs.size > 0 else None
        raw_max = float(np.nanmax(probs)) if probs.size > 0 else None
        diagnostics["prob_range_raw"] = {
            "min": raw_min,
            "max": raw_max,
        }

        # If inputs look like scores/logits (far outside [0, 1]), map them
        # through a bounded sigmoid before treating them as probabilities.
        if raw_min is not None and raw_max is not None:
            if raw_min < -1e-3 or raw_max > 1.0 + 1e-3:
                scores = np.clip(probs, -20.0, 20.0)
                probs = 1.0 / (1.0 + np.exp(-scores))

        # Clamp probabilities to [0, 1] to avoid pathological Brier scores
        probs = np.clip(probs, 0.0, 1.0)
        diagnostics["prob_range_clamped"] = {
            "min": float(np.nanmin(probs)) if probs.size > 0 else None,
            "max": float(np.nanmax(probs)) if probs.size > 0 else None,
        }

        # Detect near-constant probability vectors as degenerate calibration
        if probs.size == 0 or (
            float(np.nanmax(probs)) - float(np.nanmin(probs)) < 1e-3
        ):
            diagnostics["degenerate_calibration"] = True

        # ===== Brier Score =====
        diagnostics["brier_score"] = float(np.mean((probs - y) ** 2))

        # ===== Calibration Error (ECE, MCE) and Reliability Diagram =====
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        reliability_data = []

        for i in range(n_bins):
            bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:  # Include 1.0 in last bin
                bin_mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

            n_in_bin = int(bin_mask.sum())
            if n_in_bin > 0:
                mean_pred = float(probs[bin_mask].mean())
                mean_actual = float(y[bin_mask].mean())
                calibration_error = abs(mean_pred - mean_actual)

                ece += (n_in_bin / len(y)) * calibration_error
                mce = max(mce, calibration_error)

                reliability_data.append({
                    "bin_start": float(bin_edges[i]),
                    "bin_end": float(bin_edges[i + 1]),
                    "mean_predicted": mean_pred,
                    "mean_actual": mean_actual,
                    "count": n_in_bin,
                    "calibration_error": calibration_error,
                })

        if len(reliability_data) == 0 or len(reliability_data) == 1:
            diagnostics["ece"] = None  # None indicates degenerate/undefined
            diagnostics["mce"] = None
            diagnostics["reliability_diagram"] = reliability_data
            diagnostics["is_well_calibrated"] = False
            diagnostics["degenerate_calibration"] = True
            diagnostics["degenerate_calibration"] = True
        else:
            diagnostics["ece"] = float(ece)
            diagnostics["mce"] = float(mce)
            diagnostics["reliability_diagram"] = reliability_data
            diagnostics["is_well_calibrated"] = ece < 0.05
            diagnostics["degenerate_calibration"] = diagnostics.get("degenerate_calibration", False)

        # ===== Precision Per Bin (especially high-probability regions) =====
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            if use_linear_adaptive_gating and regimes is not None and thresh == 0.5:
                # Use regime-adaptive gating for 0.5 threshold based on
                # the sanitized probability vector.
                probs_series = pd.Series(probs)
                returns_series = pd.Series(returns) if returns is not None else pd.Series([], dtype=float)
                regime_series = pd.Series(regimes)

                adaptive_results = compute_regime_aware_trade_simulation(
                    probabilities=probs_series,
                    realized_returns=returns_series,
                    regime_score=regime_series,
                    base_threshold=thresh,
                    use_linear_adaptive=True,
                    transaction_cost=transaction_cost,
                )

                diagnostics["precision_per_bin"][f"≥{thresh}_adaptive"] = {
                    "precision": adaptive_results["win_rate"],
                    "count": adaptive_results["n_trades"],
                    "avg_return": adaptive_results["avg_return_per_trade"],
                    "adaptive_threshold_used": True,
                }
            else:
                # Standard static threshold
                mask = probs >= thresh
                n_above = int(mask.sum())
                if n_above >= 10:
                    precision = float(y[mask].mean())
                    fpr = float((1 - y[mask]).mean())  # False positive rate
                    diagnostics["precision_per_bin"][f"≥{thresh}"] = {
                        "precision": precision,
                        "count": n_above,
                    }
                    diagnostics["fpr_per_bin"][f"≥{thresh}"] = {
                        "fpr": fpr,
                        "count": n_above,
                    }

        # ===== Expected Net P&L by Probability Bucket =====
        if returns is not None:
            pnl_bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
            for low, high in pnl_bins:
                mask = (probs >= low) & (probs < high)
                n_in_bin = int(mask.sum())
                if n_in_bin >= 10:
                    bin_returns = returns[mask]
                    net_pnl = float(bin_returns.mean() - transaction_cost)
                    pnl_per_trade = net_pnl
                    expected_total = net_pnl * n_in_bin

                    diagnostics["expected_pnl_per_bin"][f"{low:.1f}-{high:.1f}"] = {
                        "mean_return": float(bin_returns.mean()),
                        "net_pnl_per_trade": pnl_per_trade,
                        "n_trades": n_in_bin,
                        "expected_total_pnl": expected_total,
                        "is_profitable": pnl_per_trade > 0,
                    }

            try:
                from sklearn.isotonic import IsotonicRegression
                if len(y) >= 100:
                    pnl_target = returns - transaction_cost
                    iso_mask = np.isfinite(probs) & np.isfinite(pnl_target)
                    if np.sum(iso_mask) >= 50:
                        iso_reg_pnl = IsotonicRegression(out_of_bounds="clip")
                        iso_reg_pnl.fit(probs[iso_mask], pnl_target[iso_mask])
                        grid = np.linspace(0.0, 1.0, 11, dtype=float)
                        expected_net = iso_reg_pnl.predict(grid)
                        diagnostics["pnl_calibration_curve"] = {
                            "prob_grid": grid.tolist(),
                            "expected_net_return": [float(v) for v in expected_net],
                        }
            except Exception:
                pass

    except Exception as e:
        tprint(f"⚠️ Calibration diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_robustness_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    realized_returns: pd.Series,
    regimes: Optional[pd.Series] = None,
    volatility: Optional[pd.Series] = None,
    n_folds: int = 5,
    transaction_cost: float = 0.003,
    time_aware_cv: bool = True,
    event_durations: Optional[pd.Series] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
    base_horizon_bars: Optional[int] = None,
    use_purged_splits: bool = True,
) -> Dict[str, Any]:
    """Compute robustness diagnostics across time and regimes.

    Tests:
    - Per-fold metrics (AUC, ECE, precision@threshold, net P&L)
    - Worst-fold performance and CV dispersion
    - Performance vs volatility regimes (low, median, high)
    - Per-regime breakdown if regimes provided

    Args:
        X: Feature matrix
        y: Binary labels
        realized_returns: Realized returns
        regimes: Optional regime labels
        volatility: Optional volatility series for regime splitting
        n_folds: Number of CV folds
        transaction_cost: Transaction cost

    Returns:
        Dictionary with robustness metrics
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    diagnostics = {
        "per_fold_metrics": [],
        "worst_fold_auc": None,
        "best_fold_auc": None,
        "auc_cv_std": None,
        "auc_cv_coefficient_of_variation": None,
        "per_volatility_regime": {},
        "per_regime_metrics": {},
        "is_robust": False,
    }

    # Clean data
    valid_mask = ~y.isna()
    X_num = X.select_dtypes(include=[np.number])
    X_clean = X_num[valid_mask].fillna(0)
    y_clean = y[valid_mask]
    returns_clean = realized_returns[valid_mask] if realized_returns is not None else None

    if len(y_clean) < 100 or len(y_clean.unique()) < 2:
        return diagnostics

    # Build CV splits: prefer t1-aware purged splits when event information is provided.
    purged_splits = None
    if (
        use_purged_splits
        and time_aware_cv
        and event_durations is not None
        and market_index is not None
        and base_horizon_bars is not None
    ):
        try:
            purged_splits = _build_t1_aware_purged_splits_for_events(
                y=y_clean,
                event_durations=event_durations,
                market_index=market_index,
                cv_splits=n_folds,
                base_horizon_bars=base_horizon_bars,
            )
        except Exception:
            purged_splits = None

    if purged_splits is not None:
        cv_splits_iter = list(purged_splits)
    elif time_aware_cv:
        cv = TimeSeriesSplit(n_splits=n_folds)
        cv_splits_iter = list(cv.split(X_clean))
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_splits_iter = list(cv.split(X_clean))

    try:
        # 1. Per-Fold Metrics
        fold_aucs = []
        fold_metrics = []

        model = lgb.LGBMClassifier(
            max_depth=5, n_estimators=100, learning_rate=0.05,
            n_jobs=-1, verbose=-1, random_state=42
        )

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits_iter):
            try:
                X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
                y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

                if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
                    continue

                model.fit(X_train, y_train)
                probs = model.predict_proba(X_test)[:, 1]

                # AUC
                auc = float(roc_auc_score(y_test, probs))
                fold_aucs.append(auc)

                # ECE (Expected Calibration Error) using probability bins
                ece = 0.0
                bin_edges = np.linspace(0, 1, 11)
                y_test_np = y_test.to_numpy(dtype=float)
                for i in range(10):
                    mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
                    if mask.sum() > 0:
                        mean_pred = float(probs[mask].mean())
                        mean_actual = float(y_test_np[mask].mean())
                        ece += (mask.sum() / len(probs)) * abs(mean_pred - mean_actual)

                # Precision at 0.6
                precision_06 = float(y_test[probs >= 0.6].mean()) if (probs >= 0.6).sum() > 0 else None

                # Net P&L
                net_pnl = None
                if returns_clean is not None:
                    returns_test = returns_clean.iloc[test_idx]
                    trade_mask = probs >= 0.5
                    if trade_mask.sum() > 0:
                        net_pnl = float(returns_test[trade_mask].mean() - transaction_cost)

                fold_metrics.append({
                    "fold": fold_idx,
                    "auc": auc,
                    "ece": float(ece),
                    "precision_at_0.6": precision_06,
                    "net_pnl_per_trade": net_pnl,
                    "n_test": len(y_test),
                })

            except Exception:
                continue

        diagnostics["per_fold_metrics"] = fold_metrics

        if fold_aucs:
            diagnostics["worst_fold_auc"] = float(min(fold_aucs))
            diagnostics["best_fold_auc"] = float(max(fold_aucs))
            diagnostics["auc_cv_std"] = float(np.std(fold_aucs))
            mean_auc = float(np.mean(fold_aucs))
            diagnostics["auc_cv_coefficient_of_variation"] = float(np.std(fold_aucs) / mean_auc) if mean_auc > 0 else None

            # Robust if CV std < 0.05 and worst fold > 0.52
            diagnostics["is_robust"] = (diagnostics["auc_cv_std"] < 0.05 and diagnostics["worst_fold_auc"] > 0.52)

        # ===== Per Volatility Regime (using OOF predictions) =====
        # Build out-of-fold predictions first to avoid in-sample leakage
        oof_probs = np.full(len(y_clean), np.nan)
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits_iter):
                X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
                y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
                if len(y_train.unique()) < 2:
                    continue
                model.fit(X_train, y_train)
                oof_probs[test_idx] = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        oof_valid_mask = ~np.isnan(oof_probs)

        if volatility is not None:
            try:
                vol_aligned = volatility.reindex(y_clean.index)
                vol_clean = vol_aligned[~vol_aligned.isna()]

                if len(vol_clean) >= 50:
                    vol_low_thresh = vol_clean.quantile(0.33)
                    vol_high_thresh = vol_clean.quantile(0.67)

                    vol_regimes = {
                        "low_vol": vol_aligned <= vol_low_thresh,
                        "medium_vol": (vol_aligned > vol_low_thresh) & (vol_aligned <= vol_high_thresh),
                        "high_vol": vol_aligned > vol_high_thresh,
                    }

                    for regime_name, regime_mask in vol_regimes.items():
                        regime_mask = regime_mask.reindex(y_clean.index, fill_value=False)
                        # Combine with OOF validity
                        combined_mask = regime_mask.values & oof_valid_mask
                        n_regime = int(combined_mask.sum())

                        if n_regime >= 30 and len(np.unique(y_clean.values[combined_mask])) >= 2:
                            try:
                                # Use OOF predictions (out-of-sample) for regime AUC
                                probs_regime = oof_probs[combined_mask]
                                y_regime = y_clean.values[combined_mask]
                                auc_regime = float(roc_auc_score(y_regime, probs_regime))

                                diagnostics["per_volatility_regime"][regime_name] = {
                                    "auc": auc_regime,
                                    "n_events": n_regime,
                                }
                            except Exception:
                                diagnostics["per_volatility_regime"][regime_name] = {"auc": None, "n_events": n_regime}
                        else:
                            diagnostics["per_volatility_regime"][regime_name] = {"auc": None, "n_events": n_regime}

            except Exception:
                pass

        # ===== Per Regime (if provided) - using OOF predictions =====
        if regimes is not None:
            try:
                regimes_aligned = regimes.reindex(y_clean.index)
                unique_regimes = pd.unique(regimes_aligned.dropna())

                for reg_val in unique_regimes:
                    regime_mask = (regimes_aligned == reg_val).values
                    # Combine with OOF validity
                    combined_mask = regime_mask & oof_valid_mask
                    n_regime = int(combined_mask.sum())

                    if n_regime >= 30 and len(np.unique(y_clean.values[combined_mask])) >= 2:
                        try:
                            # Use OOF predictions (out-of-sample) for regime AUC
                            probs_regime = oof_probs[combined_mask]
                            y_regime = y_clean.values[combined_mask]
                            auc_regime = float(roc_auc_score(y_regime, probs_regime))

                            # Net P&L in regime
                            net_pnl = None
                            if returns_clean is not None:
                                returns_regime = returns_clean.values[combined_mask]
                                trade_mask = probs_regime >= 0.5
                                if trade_mask.sum() > 0:
                                    net_pnl = float(returns_regime[trade_mask].mean() - transaction_cost)

                            diagnostics["per_regime_metrics"][str(reg_val)] = {
                                "auc": auc_regime,
                                "n_events": n_regime,
                                "net_pnl_per_trade": net_pnl,
                            }
                        except Exception:
                            diagnostics["per_regime_metrics"][str(reg_val)] = {"auc": None, "n_events": n_regime}
                    else:
                        diagnostics["per_regime_metrics"][str(reg_val)] = {"auc": None, "n_events": n_regime}

            except Exception:
                pass

    except Exception as e:
        tprint(f"⚠️ Robustness diagnostics failed: {e}", "WARNING")

    return diagnostics


def compute_class_overlap_features(
    X: pd.DataFrame,
    retained_mask: pd.Series,
    top_k_features: int = 10,
) -> Dict[str, Any]:
    """Compute class overlap visualization data for retained vs discarded events.

    Args:
        X: Feature matrix
        retained_mask: Boolean mask - True for retained events
        top_k_features: Number of top features to analyze

    Returns:
        Dictionary with overlap metrics and distribution data
    """
    diagnostics = {
        "feature_distributions": {},
        "overlap_scores": {},
        "retained_cluster_tightness": None,
        "easy_problem_detected": False,
    }

    try:
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        if X_num.empty or len(retained_mask) != len(X_num):
            return diagnostics

        retained = X_num[retained_mask]
        discarded = X_num[~retained_mask]

        if len(retained) < 20 or len(discarded) < 20:
            return diagnostics

        # Compute feature importance to select top features
        from sklearn.ensemble import RandomForestClassifier
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
            rf.fit(X_num, retained_mask.astype(int))
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:top_k_features]
            top_features = [X_num.columns[i] for i in top_idx]
        except Exception:
            top_features = list(X_num.columns[:top_k_features])

        # For each top feature, compute distribution stats
        overlap_scores = []
        for feat in top_features:
            try:
                ret_vals = retained[feat].values
                disc_vals = discarded[feat].values

                # Compute overlap using Bhattacharyya coefficient approximation
                ret_mean, ret_std = ret_vals.mean(), ret_vals.std() + 1e-8
                disc_mean, disc_std = disc_vals.mean(), disc_vals.std() + 1e-8

                # Bhattacharyya distance (lower = more overlap)
                bd = 0.25 * np.log(0.25 * ((ret_std/disc_std)**2 + (disc_std/ret_std)**2 + 2)) + \
                     0.25 * ((ret_mean - disc_mean)**2 / (ret_std**2 + disc_std**2))

                overlap_score = np.exp(-bd)  # Convert to 0-1, higher = more overlap
                overlap_scores.append(overlap_score)

                diagnostics["feature_distributions"][feat] = {
                    "retained_mean": float(ret_mean),
                    "retained_std": float(ret_std),
                    "discarded_mean": float(disc_mean),
                    "discarded_std": float(disc_std),
                    "overlap_score": float(overlap_score),
                }
                diagnostics["overlap_scores"][feat] = float(overlap_score)

            except Exception:
                continue

        # Overall cluster tightness (variance ratio)
        try:
            ret_var = retained.var().mean()
            disc_var = discarded.var().mean()
            full_var = X_num.var().mean()

            # If retained events have much lower variance, they're clustered
            diagnostics["retained_cluster_tightness"] = float(ret_var / (full_var + 1e-8))

            # Easy problem if: low overlap AND tight clustering
            avg_overlap = np.mean(overlap_scores) if overlap_scores else 1.0
            if avg_overlap < 0.4 and diagnostics["retained_cluster_tightness"] < 0.7:
                diagnostics["easy_problem_detected"] = True

        except Exception:
            pass

    except Exception as e:
        tprint(f"⚠️ Class overlap diagnostics failed: {e}", "WARNING")

    return diagnostics


def shrink_search_space(
    original_space: Dict[str, Any],
    previous_results: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, Any]:
    """Narrow the search space around the top-K best performing parameters.

    Args:
        original_space: Original parameter search space
        previous_results: List of results from previous stage (must have 'params' and 'combined' keys)
        top_k: Number of top candidates to consider

    Returns:
        Narrowed search space
    """
    if not previous_results:
        return original_space.copy()

    # Sort by objective score (prefer edge, fallback to combined) and take top-K
    sorted_results = sorted(previous_results, key=lambda x: x.get('edge', x.get('combined', 0)), reverse=True)
    best_candidates = sorted_results[:top_k]

    if not best_candidates:
        return original_space.copy()

    new_space = {}

    for param_name, config in original_space.items():
        # Extract values for this param from best candidates
        values = []
        for c in best_candidates:
            if 'params' in c and param_name in c['params']:
                values.append(c['params'][param_name])

        if not values:
            new_space[param_name] = config.copy()
            continue

        param_type = config.get('type', 'float')

        if param_type in ['float', 'int']:
            min_val = min(values)
            max_val = max(values)

            # Add 10% buffer so we don't over-collapse
            spread = max_val - min_val
            if spread == 0:
                spread = (config.get('high', 1) - config.get('low', 0)) * 0.1

            buffer = spread * 0.1
            new_low = max(config.get('low', 0), min_val - buffer)
            new_high = min(config.get('high', 1), max_val + buffer)

            # Ensure we don't collapse completely
            if new_high <= new_low:
                new_low = config.get('low', 0)
                new_high = config.get('high', 1)

            new_config = config.copy()
            new_config['low'] = new_low if param_type == 'float' else int(new_low)
            new_config['high'] = new_high if param_type == 'float' else int(max(new_high, new_low + 1))
            new_space[param_name] = new_config

        else:
            # For categorical params, keep all or filter to frequent ones
            new_space[param_name] = config.copy()

    return new_space


def compute_economic_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray
) -> float:
    """Helper: Weighted AUC based on log-return magnitude."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    # Weight by log-magnitude of returns to prioritize economically significant events
    weights = np.log1p(np.abs(returns))
    try:
        return roc_auc_score(y_true, y_prob, sample_weight=weights)
    except Exception:
        return 0.5


def sigmoid_gate(x: float, threshold: float, sharpness: float = 10.0, lower_bound: float = 0.0) -> float:
    """Helper: Smooth transition from lower_bound to 1.0."""
    try:
        sigmoid = 1.0 / (1.0 + np.exp(-sharpness * (x - threshold)))
    except OverflowError:
        sigmoid = 0.0 if x < threshold else 1.0
    return lower_bound + (1.0 - lower_bound) * sigmoid


# ============================================================================
# RTS SMOOTHER & KALMAN FILTER FUNCTIONS (For Label & Feature Generation)
# ============================================================================

def rts_smoother_1d(
    prices: np.ndarray,
    Q: float,
    R: float,
    init_val: float = None,
    init_cov: float = 1.0,
) -> tuple:
    """
    Implements a 1D Rauch-Tung-Striebel Smoother (Local Level Model).
    This is an ACAUSAL (zero-lag) smoother ideal for label generation.
    
    Model: x_t = x_{t-1} + w_t  (Process Noise Q)
           z_t = x_t + v_t      (Measurement Noise R)
    
    Args:
        prices: Raw price series
        Q: Process noise variance (higher = more responsive, less smooth)
        R: Measurement noise variance (higher = more smooth)
        init_val: Initial state value (default: first observation)
        init_cov: Initial covariance (default: 1.0)
    
    Returns:
        Tuple of (smoothed_state, smoothed_covariance)
    """
    n = len(prices)
    obs = np.asarray(prices, dtype=np.float64)
    
    # --- Forward Pass (Standard Kalman Filter) ---
    m = np.zeros(n)  # State means
    P = np.zeros(n)  # State covariances
    
    # Initialization
    m[0] = init_val if init_val is not None else obs[0]
    P[0] = init_cov
    
    for t in range(1, n):
        # Time Update (Prediction)
        m_minus = m[t-1]
        P_minus = P[t-1] + Q
        
        # Measurement Update (Correction)
        K = P_minus / (P_minus + R)  # Kalman Gain
        m[t] = m_minus + K * (obs[t] - m_minus)
        P[t] = (1 - K) * P_minus
        
    # --- Backward Pass (RTS Smoothing) ---
    s_m = np.zeros(n)  # Smoothed means
    s_P = np.zeros(n)  # Smoothed covariances
    
    # Last step is same as filter
    s_m[-1] = m[-1]
    s_P[-1] = P[-1]
    
    for t in range(n-2, -1, -1):
        # Smoothing Gain
        P_pred = P[t] + Q
        J = P[t] / P_pred if P_pred > 1e-12 else 0.0
        
        # State Update (look-ahead correction)
        s_m[t] = m[t] + J * (s_m[t+1] - m[t])
        s_P[t] = P[t] + (J**2) * (s_P[t+1] - P_pred)
    
    return s_m, s_P


def kalman_filter_1d(
    prices: np.ndarray,
    Q: float,
    R: float,
    init_val: float = None,
    init_cov: float = 1.0,
) -> tuple:
    """
    Standard 1D Kalman Filter (CAUSAL) for live feature generation.
    
    Model: x_t = x_{t-1} + w_t  (Process Noise Q)
           z_t = x_t + v_t      (Measurement Noise R)
    
    Args:
        prices: Raw price series
        Q: Process noise variance
        R: Measurement noise variance
        init_val: Initial state value (default: first observation)
        init_cov: Initial covariance (default: 1.0)
    
    Returns:
        Tuple of (filtered_state, filtered_covariance, kalman_gain)
    """
    n = len(prices)
    obs = np.asarray(prices, dtype=np.float64)
    
    m = np.zeros(n)  # State means
    P = np.zeros(n)  # State covariances
    K_arr = np.zeros(n)  # Kalman gains
    
    # Initialization
    m[0] = init_val if init_val is not None else obs[0]
    P[0] = init_cov
    K_arr[0] = 0.5
    
    for t in range(1, n):
        # Time Update (Prediction)
        m_minus = m[t-1]
        P_minus = P[t-1] + Q
        
        # Measurement Update (Correction)
        K = P_minus / (P_minus + R) if (P_minus + R) > 1e-12 else 0.5
        K_arr[t] = K
        m[t] = m_minus + K * (obs[t] - m_minus)
        P[t] = (1 - K) * P_minus
    
    return m, P, K_arr


def robust_labeling_loss(
    smoothed: np.ndarray,
    raw: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    is_acausal: bool = True,
) -> tuple:
    """
    Compute loss for labeling optimization (RTS Smoother tuning).
    
    Optimizes Signal-to-Noise Ratio by balancing:
    1. Smoothness (minimal wiggle / 2nd derivative)
    2. Tracking Error (RMSE from raw prices)
    3. Amplitude Fidelity (preserve ~95% of price volatility)
    
    Args:
        smoothed: Smoothed price series from RTS
        raw: Raw price series
        alpha: Weight for smoothness penalty
        beta: Weight for tracking error penalty
        gamma: Weight for amplitude error penalty
        is_acausal: If True (RTS), enforces zero-lag checking
    
    Returns:
        Tuple of (total_loss, details_dict)
    """
    s = np.asarray(smoothed, dtype=np.float64)
    r = np.asarray(raw, dtype=np.float64)
    
    # Returns for stationarity
    s_ret = np.diff(s)
    r_ret = np.diff(r)
    raw_vol = np.std(r_ret) + 1e-9

    # --- Component 1: Smoothness (Normalized) ---
    # Goal: Minimal "wiggle" (2nd derivative)
    second_diff = np.diff(s, n=2)
    smooth_error = np.mean(second_diff**2) / (raw_vol**2)

    # --- Component 2: Tracking Error ---
    if is_acausal:
        # RTS: Direct RMSE (no lag expected)
        rmse = np.sqrt(np.mean((s - r)**2))
        tracking_error = rmse / raw_vol
    else:
        # Causal filter: Allow 1-bar lag
        tau = 1
        rmse = np.sqrt(np.mean((s[:-tau] - r[tau:])**2))
        tracking_error = rmse / raw_vol

    # --- Component 3: Amplitude Fidelity ---
    # CRITICAL FOR LABELS: Ensure we don't "shrink" the events.
    std_s = np.std(s_ret)
    std_r = np.std(r_ret)
    
    # Target 95% volatility retention
    # Penalty for over-smoothing (ratio < 0.95) or noise amplification (ratio > 1.05)
    amp_ratio = std_s / (std_r + 1e-9)
    amp_error = (amp_ratio - 0.95)**2

    # --- Total Loss ---
    total_loss = (alpha * smooth_error) + (beta * tracking_error) + (gamma * amp_error)
    
    return total_loss, {
        "loss": total_loss,
        "smooth": smooth_error,
        "track": tracking_error,
        "amp": amp_error,
        "amp_ratio": amp_ratio,
    }


def smooth_prices_rts(
    prices: pd.Series,
    Q: float,
    R: float,
) -> pd.Series:
    """
    Smooth a price series using RTS Smoother for label generation.
    
    RTS is ACAUSAL (uses future data) - only for training labels, NOT live features.
    For live features, use kalman_filter_1d instead.
    
    Args:
        prices: Raw price series
        Q: Process noise variance (from Stage 0 optimization)
        R: Measurement noise variance (from Stage 0 optimization)
    
    Returns:
        Smoothed price series as pandas Series
    """
    smoothed, _ = rts_smoother_1d(
        prices=prices.values,
        Q=Q,
        R=R,
        init_val=None,
        init_cov=1.0,
    )
    return pd.Series(smoothed, index=prices.index, name="rts_smoothed_close")


def _log_normalize(arr: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Log-normalize an array: sign(x) * log(1 + |x|).
    Preserves sign while compressing magnitude for better ML performance.
    """
    return np.sign(arr) * np.log1p(np.abs(arr) + epsilon)


# ============================================================================
# FEATURE QUALITY & SELECTION FUNCTIONS (De Prado-Inspired)
# ============================================================================

def _base_quality_score(series: np.ndarray) -> float:
    """
    Base quality score calculation engine.
    
    High Stability (low std of score), High Signal (mean), Low Fat Tails (kurtosis).
    Uses Sharpe-like ratio penalized by kurtosis for robustness.
    
    Args:
        series: Feature values as numpy array
    
    Returns:
        Quality score (higher = better). Returns 0.0 for flat/useless features.
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    
    # 1. Handle constant or empty data
    if len(series) < 10:
        return 0.0
    
    sigma = np.std(series)
    if sigma < 1e-9:
        return 0.0
    
    # 2. Calculate components
    mu = np.mean(series)
    
    # Fisher kurtosis: normal distribution = 0.0
    try:
        kurt = scipy_kurtosis(series, fisher=True, nan_policy='omit')
        if not np.isfinite(kurt):
            kurt = 0.0
    except Exception:
        kurt = 0.0
    
    # 3. Formulate Score
    # Signal-to-noise ratio penalized by fat tails
    signal_to_noise = np.abs(mu / sigma)
    
    # Penalty: high kurtosis (fat tails) = unstable feature
    # (1 + abs(kurt)) ensures strictly positive denominator
    penalty = 1.0 + np.abs(kurt)
    
    return float(signal_to_noise / penalty)


def calculate_time_robust_quality(series: np.ndarray, chunk_size: int = 2000) -> float:
    """
    Calculate Quality Score in chunks and return the WORST-CASE (10th percentile).
    
    This prevents features that are only good in specific market regimes from
    dominating the selection. A feature must be consistently useful across time.
    
    Args:
        series: Feature values as numpy array
        chunk_size: Size of each evaluation chunk (default 2000 bars ~ 20 days at 15m)
    
    Returns:
        Conservative (10th percentile) quality score across time chunks
    """
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(series)
    
    # Fallback for short series: just calculate global score
    if n < chunk_size:
        return _base_quality_score(series)
    
    scores = []
    
    # Rolling/Chunked Evaluation
    for i in range(0, n, chunk_size):
        chunk = series[i : i + chunk_size]
        
        # Skip small incomplete chunks at the end
        if len(chunk) < chunk_size // 2:
            continue
        
        q_score = _base_quality_score(chunk)
        scores.append(q_score)
    
    if not scores:
        return _base_quality_score(series)
    
    # Return 10th Percentile (Conservative worst-case)
    return float(np.percentile(scores, 10))


def calculate_feature_quality(series: np.ndarray) -> float:
    """
    Calculate time-robust quality score for a feature (unsupervised).
    
    This is a wrapper that uses the time-robust chunked evaluation by default.
    
    Args:
        series: Feature values as numpy array
    
    Returns:
        Quality score (higher = better). Returns 0.0 for flat/useless features.
    """
    return calculate_time_robust_quality(series)


def calculate_all_feature_qualities(
    df_features: pd.DataFrame,
    use_time_robust: bool = True,
    chunk_size: int = 2000,
) -> Dict[str, float]:
    """
    Calculate quality scores for all feature columns.
    
    Args:
        df_features: DataFrame with feature columns
        use_time_robust: Whether to use time-robust chunked evaluation (default True)
        chunk_size: Chunk size for time-robust evaluation
    
    Returns:
        Dict mapping column name to quality score
    """
    quality_map = {}
    for col in df_features.columns:
        try:
            if use_time_robust:
                quality_map[col] = calculate_time_robust_quality(
                    df_features[col].values, chunk_size=chunk_size
                )
            else:
                quality_map[col] = _base_quality_score(df_features[col].values)
        except Exception:
            quality_map[col] = 0.0
    return quality_map


# ============================================================================
# LGBM MAGNITUDE SWEEP (Structure Detection)
# ============================================================================

def lgbm_magnitude_sweep(
    df_features: pd.DataFrame,
    market_data: pd.DataFrame,
    lookahead: int = 4,
    max_features: int = 200,
    importance_threshold: float = 1.0,
    price_col: str = 'close',
) -> pd.DataFrame:
    """
    Use 'Future Volatility/Magnitude' as a proxy target to prune features
    that contain no structural market information.
    
    This is a CHEAP unsupervised filter that removes noise features before
    the expensive hierarchical clustering step.
    
    Target = Absolute value of the next N-bar return (Volatility Proxy).
    
    Args:
        df_features: DataFrame with feature columns
        market_data: Market data with price column
        lookahead: Number of bars to look ahead for magnitude (default 4)
        max_features: Maximum features to keep (default 200)
        importance_threshold: Minimum importance % to keep (default 1.0%)
        price_col: Column name for price data
    
    Returns:
        DataFrame with features that have structural market information
    """
    import lightgbm as lgb
    
    tprint_info(f"🔬 LGBM Magnitude Sweep: {len(df_features.columns)} features, lookahead={lookahead} bars")
    
    # 1. Create Proxy Target: Absolute Future Return (Magnitude)
    if price_col not in market_data.columns:
        tprint_warning(f"   ⚠️ Price column '{price_col}' not found, skipping magnitude sweep")
        return df_features
    
    # Target: The MAGNITUDE of the move over the next 'lookahead' bars (Cumulative)
    # Corrected: Use cumulative return over N bars rather than return of the N-th bar
    future_price = market_data[price_col].shift(-lookahead)
    current_price = market_data[price_col]
    
    # abs(ln(P_{t+N} / P_t))
    proxy_target = np.abs(np.log(future_price / current_price))
    
    # Align indices
    common_idx = df_features.index.intersection(proxy_target.dropna().index)
    if len(common_idx) < 500:
        tprint_warning(f"   ⚠️ Insufficient data for magnitude sweep ({len(common_idx)} rows)")
        return df_features
    
    X = df_features.loc[common_idx].fillna(0)
    y = proxy_target.loc[common_idx]
    
    # Remove any remaining NaN/Inf
    valid_mask = np.isfinite(y.values)
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    if len(X) < 500:
        tprint_warning(f"   ⚠️ Insufficient valid data for magnitude sweep")
        return df_features
    
    tprint_info(f"   Training shadow LGBM on {X.shape[1]} features ({len(X)} samples)...")
    
    # 2. Train Fast LGBM (L1 regression for robustness)
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    
    params = {
        'objective': 'regression_l1',  # Mean Absolute Error - robust to spikes
        'boosting': 'gbdt',
        'verbosity': -1,
        'num_leaves': 15,             # Very simple trees
        'max_depth': 3,               # Prevent overfitting
        'feature_fraction': 0.7,      # Force trees to look at different features
        'min_data_in_leaf': 100,
        'learning_rate': 0.1,
        'seed': 42,
    }
    
    # Train heavily regularized model
    model = lgb.train(params, dtrain, num_boost_round=150)
    
    # 3. Get Feature Importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = np.array(model.feature_name())
    
    # Handle case where no features have importance
    if importance.max() < 1e-9:
        tprint_warning("   ⚠️ No features showed importance, keeping all")
        return df_features
    
    # Normalize importance (0-100)
    importance_normalized = 100 * (importance / importance.max())
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_normalized}
    ).sort_values("importance", ascending=False)
    
    keep_count = min(max_features, len(importance_df))
    above_threshold = importance_df[importance_df["importance"] >= importance_threshold]
    
    if len(above_threshold) < keep_count:
        tprint_warning(
            f"   ⚠️ Only {len(above_threshold)} features meet the "
            f"{importance_threshold:.2f}% importance threshold; keeping top {keep_count} instead"
        )
        keep_df = importance_df.head(keep_count)
    else:
        keep_df = above_threshold.head(keep_count)
    
    useful_feats = keep_df["feature"].to_numpy()
    dropped_count = len(feature_names) - len(useful_feats)
    min_kept_importance = float(keep_df["importance"].min()) if not keep_df.empty else 0.0
    
    tprint_info(f"   → Dropped {dropped_count} features that failed to predict market magnitude")
    tprint_info(
        f"   → Kept {len(useful_feats)} structure-bearing features "
        f"(min kept importance {min_kept_importance:.2f}%)"
    )
    
    top_preview = keep_df.head(10)
    if not top_preview.empty:
        preview_pairs = [f"{row.feature}:{row.importance:.1f}%" for row in top_preview.itertuples(index=False)]
        tprint_info(f"   → Top features: {preview_pairs}")
    
    # Return only useful features (preserve original column order where possible)
    useful_feats_set = set(useful_feats)
    ordered_feats = [c for c in df_features.columns if c in useful_feats_set]
    
    return df_features[ordered_feats]


# ============================================================================
# HIERARCHICAL FEATURE SELECTION (De Prado's Method)
# ============================================================================

def select_features_hierarchical(
    df_features: pd.DataFrame,
    quality_scores: Dict[str, float],
    target_n: int = 70,
    target_series: Optional[Union[pd.Series, np.ndarray]] = None,
    autocorr_penalty_weight: float = 0.3,
    snr_weight: float = 0.2,
    enable_snr_objective: bool = True,
) -> pd.DataFrame:
    """
    De Prado's Hierarchical Feature Selection (Optimized).
    
    Guarantees diversity by picking BEST-IN-CLASS from N DISTINCT clusters.
    This prevents concept dominance where correlated features crowd out
    diverse information sources.
    
    Algorithm:
    1. Pre-trim to max 5k features by quality (optimization)
    2. Compute correlation matrix (Spearman for robustness, optimized)
    3. Convert to distance: d = sqrt(2(1-|rho|))
    4. Hierarchical clustering (Ward's method)
    5. Cut tree into target_n clusters
    6. Select highest-quality feature from each cluster
    
    Args:
        df_features: DataFrame with feature columns
        quality_scores: Dict mapping column name to quality score
        target_n: Target number of features (= number of clusters)
    
    Returns:
        DataFrame with target_n orthogonal features
    """
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    # 0. Safety: Drop constant columns to prevent NaN correlations
    df_clean = df_features.loc[:, (df_features != df_features.iloc[0]).any()].copy()
    
    # Also drop columns with very low std
    col_stds = df_clean.std()
    valid_cols = col_stds[col_stds > 1e-9].index.tolist()
    df_clean = df_clean[valid_cols]
    
    n_features = len(df_clean.columns)
    
    if n_features == 0:
        tprint_warning("⚠️ No valid features for hierarchical selection")
        return df_features.iloc[:, :min(target_n, len(df_features.columns))]
    
    if n_features <= target_n:
        tprint_info(f"   Hierarchical: Only {n_features} features, keeping all")
        return df_clean
    
    # OPTIMIZATION 1: Pre-trim to max 5k features by quality to reduce O(n^2) cost
    MAX_FEATURES_FOR_CLUSTERING = 5000
    if n_features > MAX_FEATURES_FOR_CLUSTERING:
        if quality_scores:
            # Sort by quality and keep top 5k
            sorted_cols = sorted(
                df_clean.columns, 
                key=lambda c: quality_scores.get(c, 0.0), 
                reverse=True
            )
            trimmed_cols = sorted_cols[:MAX_FEATURES_FOR_CLUSTERING]
            df_clean = df_clean[trimmed_cols]
            n_features = len(df_clean.columns)
            tprint_info(f"   Pre-trimmed to top {n_features} features by quality (cap={MAX_FEATURES_FOR_CLUSTERING})")
        else:
            # No quality scores, just take first 5k
            df_clean = df_clean.iloc[:, :MAX_FEATURES_FOR_CLUSTERING]
            n_features = len(df_clean.columns)
            tprint_info(f"   Pre-trimmed to first {n_features} features (no quality scores, cap={MAX_FEATURES_FOR_CLUSTERING})")
    
    tprint_info(f"   Hierarchical clustering: {n_features} → {target_n} features")
    
    # OPTIMIZATION 2: Faster Spearman correlation using float32, rank-transform, optional row subsampling
    # Subsample rows if > 20k for faster computation (approximation acceptable)
    MAX_ROWS_FOR_CORR = 5000
    n_rows = len(df_clean)
    if n_rows > MAX_ROWS_FOR_CORR:
        # Use random sampling to preserve distribution
        np.random.seed(42)  # Reproducible
        sample_idx = np.random.choice(n_rows, size=MAX_ROWS_FOR_CORR, replace=False)
        df_sample = df_clean.iloc[sample_idx]
        tprint_info(f"   Subsampled rows: {n_rows} → {MAX_ROWS_FOR_CORR} for correlation computation")
    else:
        df_sample = df_clean
    
    # Convert to float32 numpy array for memory efficiency
    feature_array = df_sample.values.astype(np.float32)
    n_cols = feature_array.shape[1]
    
    # Rank-transform each column (Spearman correlation = Pearson on ranks)
    ranked_array = np.zeros_like(feature_array, dtype=np.float32)
    for i in range(n_cols):
        col_data = feature_array[:, i]
        # Handle NaN/inf - replace with column mean for ranking
        valid_mask = np.isfinite(col_data)
        if valid_mask.sum() > 1:
            # Rank valid values, fill invalid with mean rank
            ranks = rankdata(col_data[valid_mask], method='average').astype(np.float32)
            ranked_array[valid_mask, i] = ranks
            if not valid_mask.all():
                mean_rank = ranks.mean()
                ranked_array[~valid_mask, i] = mean_rank
        elif valid_mask.sum() == 1:
            # Single valid value, set all to same rank
            ranked_array[:, i] = 1.0
        else:
            # No valid values, set to 0 (will be handled in correlation)
            ranked_array[:, i] = 0.0
    
    # Compute correlation matrix using np.corrcoef (faster than pandas)
    corr_matrix = np.corrcoef(ranked_array.T)
    
    # Replace NaN/Inf with 0 (same as pandas fillna(0))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure symmetric and bounded [-1, 1]
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    
    # 2. Convert to Distance: d = sqrt(2(1-|rho|))
    # Clip to avoid negative values from float precision errors
    dist_matrix = np.sqrt(np.clip(2 * (1 - np.abs(corr_matrix)), 0, None))
    
    # Ensure diagonal is exactly 0
    np.fill_diagonal(dist_matrix, 0)
    
    # Convert to condensed form for scipy
    try:
        dist_array = squareform(dist_matrix, checks=False)
    except Exception as e:
        tprint_warning(f"   ⚠️ Distance matrix conversion failed: {e}")
        # Fallback to greedy selection
        sorted_cols = sorted(df_clean.columns, key=lambda c: quality_scores.get(c, 0.0), reverse=True)
        return df_clean[sorted_cols[:target_n]]
    
    # 3. Hierarchical Clustering (Ward's method for balanced clusters)
    try:
        linkage_matrix = hierarchy.linkage(dist_array, method='ward')
    except Exception as e:
        tprint_warning(f"   ⚠️ Hierarchical clustering failed: {e}")
        sorted_cols = sorted(df_clean.columns, key=lambda c: quality_scores.get(c, 0.0), reverse=True)
        return df_clean[sorted_cols[:target_n]]
    
    # 4. Form Clusters (cut tree into target_n clusters)
    cluster_labels = hierarchy.fcluster(linkage_matrix, t=target_n, criterion='maxclust')
    
    # 5. Select Best Feature per Cluster
    selected_feats = []
    feature_names = df_clean.columns.tolist()
    
    for cluster_id in range(1, target_n + 1):
        # Get all features in this cluster
        members = [feature_names[i] for i in range(len(feature_names)) 
                   if cluster_labels[i] == cluster_id]
        
        if not members:
            continue
        
        # Pick the one with highest Quality Score
        best_in_cluster = max(members, key=lambda x: quality_scores.get(x, 0.0))
        selected_feats.append(best_in_cluster)
    
    tprint_info(f"   → Selected {len(selected_feats)} orthogonal features from {target_n} clusters")
    
    return df_clean[selected_feats]


# ============================================================================
# LEGACY GREEDY SELECTION (Kept for comparison/fallback)
# ============================================================================

def reduce_features_by_correlation(
    df_features: pd.DataFrame,
    quality_scores: Dict[str, float],
    target_n: int = 70,
    correlation_threshold: float = 0.85,
    min_quality_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    [LEGACY] Greedy correlation-based feature reduction.
    
    WARNING: This method suffers from "Concept Dominance" where features from
    the same family (e.g., multiple RSI variants) can crowd out diverse features.
    
    For production use, prefer select_features_hierarchical() which guarantees
    diversity through clustering.
    
    Args:
        df_features: DataFrame with all features
        quality_scores: Dict mapping column name to quality score
        target_n: Target number of features to keep
        correlation_threshold: Max allowed |correlation| between features
        min_quality_threshold: Minimum quality score to consider (hard cutoff)
    
    Returns:
        DataFrame with reduced feature set
    """
    # 1. Filter out low-quality features first
    valid_cols = [
        col for col in df_features.columns 
        if quality_scores.get(col, 0.0) > min_quality_threshold
    ]
    
    if len(valid_cols) == 0:
        tprint_warning("⚠️ No features passed quality threshold, using all features")
        valid_cols = list(df_features.columns)
    
    # 2. Sort by quality (descending)
    sorted_cols = sorted(valid_cols, key=lambda c: quality_scores.get(c, 0.0), reverse=True)
    
    # 3. Compute correlation matrix (only for valid columns)
    df_valid = df_features[sorted_cols].copy()
    corr_matrix = df_valid.corr().abs()
    
    # 4. Greedy selection: add features if not too correlated with selected ones
    selected_features = []
    
    for col in sorted_cols:
        if len(selected_features) >= target_n:
            break
        
        # Check correlation with already selected features
        is_correlated = False
        for selected_col in selected_features:
            if corr_matrix.loc[col, selected_col] > correlation_threshold:
                is_correlated = True
                break
        
        if not is_correlated:
            selected_features.append(col)
    
    # If we don't have enough features, lower correlation threshold
    if len(selected_features) < target_n:
        remaining_cols = [c for c in sorted_cols if c not in selected_features]
        relaxed_threshold = min(0.95, correlation_threshold + 0.1)
        
        for col in remaining_cols:
            if len(selected_features) >= target_n:
                break
            
            is_correlated = False
            for selected_col in selected_features:
                if corr_matrix.loc[col, selected_col] > relaxed_threshold:
                    is_correlated = True
                    break
            
            if not is_correlated:
                selected_features.append(col)
    
    tprint_info(
        f"   Greedy reduction: {len(df_features.columns)} → {len(selected_features)} "
        f"(target={target_n}, corr_threshold={correlation_threshold})"
    )
    
    return df_features[selected_features]


def generate_multi_horizon_features(
    base_features: pd.DataFrame,
    horizons: Dict[str, int] = None,
    include_base: bool = True,
) -> pd.DataFrame:
    """
    Generate multi-horizon versions of features (short, medium, long).
    
    For each feature, creates smoothed versions at different lookback windows
    to capture different time scales of the same signal.
    
    NOTE: This creates EMA-smoothed versions of ALREADY COMPUTED features.
    For truly configurable base timeframes, use generate_configurable_features().
    
    Args:
        base_features: DataFrame with base feature columns
        horizons: Dict mapping horizon name to lookback bars
                  Default: {"Short": 5, "Medium": 20, "Long": 60}
        include_base: Whether to include original features (default True)
    
    Returns:
        DataFrame with multi-horizon features added
        
    Features created per input column:
        - {col}_Short: EMA(5) smoothed
        - {col}_Medium: EMA(20) smoothed
        - {col}_Long: EMA(60) smoothed
        - {col}_Short_Diff: Momentum vs short EMA
        - {col}_Medium_Diff: Momentum vs medium EMA
        - {col}_Long_Diff: Momentum vs long EMA
    
    Total: 6 new columns per input feature (+ original if include_base=True)
    """
    if horizons is None:
        horizons = {
            "Short": 5,    # ~1.25 hours at 15m
            "Medium": 20,  # ~5 hours at 15m
            "Long": 60,    # ~15 hours at 15m
        }
    
    if include_base:
        result = base_features.copy()
    else:
        result = pd.DataFrame(index=base_features.index)
    
    for col in base_features.columns:
        series = base_features[col]
        
        for horizon_name, lookback in horizons.items():
            # Create smoothed version using EMA
            smoothed = series.ewm(span=lookback, adjust=False).mean()
            
            # Feature: Smoothed value
            new_col_name = f"{col}_{horizon_name}"
            result[new_col_name] = smoothed
            
            # Feature: Difference from base (momentum at this horizon)
            diff_col_name = f"{col}_{horizon_name}_Diff"
            result[diff_col_name] = _log_normalize((series - smoothed).values)
    
    return result


# ============================================================================
# CONFIGURABLE BASE FEATURE GENERATION (Horizon-Aware)
# ============================================================================

# Default horizon multipliers for different feature types
# Base unit is the "Medium" horizon (e.g., 20 bars at 15m = 5 hours)
HORIZON_MULTIPLIERS = {
    "Short": 0.25,   # 5 bars at 15m
    "Medium": 1.0,   # 20 bars at 15m (base)
    "Long": 3.0,     # 60 bars at 15m
}

# Features that should NOT be horizon-adjusted (specialist features)
FIXED_HORIZON_FEATURES = {
    "volatility_1d",      # Always 1-day (96 bars at 15m)
    "volatility_1w",      # Always 1-week
    "daily_range",        # Daily high-low
    "overnight_gap",      # Session-based
}


def generate_configurable_technical_features(
    market_data: pd.DataFrame,
    base_horizon: int = 20,
    horizons: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Generate technical features with CONFIGURABLE base timeframes.
    
    All lookback windows are derived from `base_horizon` multiplied by
    horizon multipliers, allowing bulk adjustment of feature timeframes.
    
    Args:
        market_data: DataFrame with OHLCV columns
        base_horizon: Base lookback in bars (default 20 = ~5 hours at 15m)
        horizons: Multipliers for Short/Medium/Long (default from HORIZON_MULTIPLIERS)
    
    Returns:
        DataFrame with configurable-horizon technical features
        
    Feature List (per horizon):
        1. RSI_{horizon}: Relative Strength Index
        2. ATR_{horizon}: Average True Range (normalized)
        3. BB_Distance_{horizon}: Bollinger Band position
        4. SMA_Distance_{horizon}: Distance from SMA
        5. EMA_Distance_{horizon}: Distance from EMA
        6. ROC_{horizon}: Rate of Change
        7. Momentum_{horizon}: Price momentum
        8. Volatility_{horizon}: Rolling volatility
        9. Volume_SMA_Ratio_{horizon}: Volume vs SMA (if volume available)
        10. High_Low_Range_{horizon}: Normalized range
        
    Total: 10 features x 3 horizons = 30 configurable features
    Plus fixed features (volatility_1d, etc.)
    """
    if horizons is None:
        horizons = HORIZON_MULTIPLIERS.copy()
    
    features = pd.DataFrame(index=market_data.index)
    
    close = market_data["close"]
    high = market_data.get("high", close)
    low = market_data.get("low", close)
    volume = market_data.get("volume", None)
    
    # Generate features for each horizon
    for horizon_name, multiplier in horizons.items():
        lookback = max(2, int(base_horizon * multiplier))
        suffix = f"_{horizon_name}"
        
        # 1. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        features[f"RSI{suffix}"] = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # 2. ATR (normalized by price)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(lookback).mean()
        features[f"ATR{suffix}"] = _log_normalize((atr / close).values)
        
        # 3. Bollinger Band Distance
        sma = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        bb_width = bb_upper - bb_lower
        features[f"BB_Distance{suffix}"] = ((close - sma) / (bb_width / 2 + 1e-9)).values
        
        # 4. SMA Distance
        features[f"SMA_Distance{suffix}"] = _log_normalize(((close - sma) / close).values)
        
        # 5. EMA Distance
        ema = close.ewm(span=lookback, adjust=False).mean()
        features[f"EMA_Distance{suffix}"] = _log_normalize(((close - ema) / close).values)
        
        # 6. Rate of Change (ROC)
        roc = (close - close.shift(lookback)) / (close.shift(lookback) + 1e-9)
        features[f"ROC{suffix}"] = _log_normalize(roc.values)
        
        # 7. Momentum
        momentum = close - close.shift(lookback)
        features[f"Momentum{suffix}"] = _log_normalize((momentum / close).values)
        
        # 8. Rolling Volatility
        returns = close.pct_change()
        vol = returns.rolling(lookback).std()
        features[f"Volatility{suffix}"] = _log_normalize(vol.values)
        
        # 9. Volume SMA Ratio (if available)
        if volume is not None:
            vol_sma = volume.rolling(lookback).mean()
            vol_ratio = volume / (vol_sma + 1e-9)
            features[f"Volume_SMA_Ratio{suffix}"] = _log_normalize((vol_ratio - 1).values)
        
        # 10. High-Low Range (normalized)
        hl_range = (high - low) / close
        hl_range_ma = hl_range.rolling(lookback).mean()
        features[f"HL_Range{suffix}"] = _log_normalize(hl_range_ma.values)
    
    # Add FIXED horizon features (specialist features)
    # These are NOT adjusted by horizon config
    
    # Volatility 1D (always 96 bars at 15m)
    returns = close.pct_change()
    features["Volatility_1D"] = _log_normalize(returns.rolling(96).std().values)
    
    # Volatility 1W (always 672 bars at 15m)
    features["Volatility_1W"] = _log_normalize(returns.rolling(672).std().values)
    
    # Daily Range (24-hour high-low, 96 bars at 15m)
    daily_high = high.rolling(96).max()
    daily_low = low.rolling(96).min()
    features["Daily_Range"] = _log_normalize(((daily_high - daily_low) / close).values)
    
    # Fill NaN and replace infinities
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    
    return features


def get_feature_inventory() -> Dict[str, List[str]]:
    """
    Return a complete inventory of features generated by the pipeline.
    
    Returns:
        Dict with categories and their feature lists
    """
    inventory = {
        "configurable_technical": [
            # Per horizon (Short, Medium, Long) = 3 versions each
            "RSI",
            "ATR",
            "BB_Distance",
            "SMA_Distance",
            "EMA_Distance",
            "ROC",
            "Momentum",
            "Volatility",
            "Volume_SMA_Ratio",
            "HL_Range",
        ],
        "fixed_specialist": [
            "Volatility_1D",
            "Volatility_1W",
            "Daily_Range",
        ],
        "kalman_price": [
            "KF_Close_LogRet",
            "KF_High_LogRet",
            "KF_Low_LogRet",
            "KF_Velocity",
            "KF_Acceleration",
            "KF_Slope",
            "KF_P",
            "KF_RSI",
            "KF_BB_Distance",
            "KF_ATR",
            "KF_ATR_Ratio",
        ],
        "kalman_vwap": [
            "KF_VWAP_Distance",
            "KF_VWAP_Slope",
            "KF_VWAP_Zscore",
        ],
        "kalman_volume": [
            "KF_Volume_Diff",
            "KF_LogVolume_Slope",
            "KF_Volume_Zscore",
            "KF_Volume_Ratio",
            "KF_Volume_P",
        ],
        # Cross-feature interactions (added for sophisticated signal combinations)
        "cross_price_volume": [
            "PV_Velocity_x_VolSlope",      # KF_Velocity × KF_LogVolume_Slope
            "PV_VWAP_x_VolZscore",         # KF_VWAP_Distance × KF_Volume_Zscore
            "PV_Return_x_VolRatio",        # KF_Close_LogRet × KF_Volume_Ratio
            "PV_SMADist_x_VolRatio",       # SMA_Distance × Volume_SMA_Ratio
            "PV_ROC_x_VolP",               # ROC × KF_Volume_P
        ],
        "cross_volatility_normalized": [
            "VN_Velocity_per_ATR",         # KF_Velocity / KF_ATR
            "VN_Mom5_per_Vol5",            # Momentum_5 / volatility_5
            "VN_Accel_per_ATRRatio",       # KF_Acceleration / KF_ATR_Ratio
            "VN_ROC_x_VolRegime",          # ROC × Volatility_Regime
            "VN_Slope_x_KalmanP",          # KF_Slope × KF_P
        ],
        "cross_horizon_divergence": [
            "XH_RSI_Divergence",           # RSI_Short - RSI_Long
            "XH_Momentum_Ratio",           # Momentum_Short / Momentum_Long
            "XH_ATR_Ratio",                # ATR_Short / ATR_Long
            "XH_SMADist_Divergence",       # SMA_Distance_Short - SMA_Distance_Long
            "XH_BBDist_Divergence",        # BB_Distance_Short - BB_Distance_Long
        ],
        "cross_regime_conditional": [
            "RC_Mom_x_MetaVolRegime",      # Momentum × meta_volatility_regime
            "RC_VolSlope_x_MetaVolShock",  # KF_LogVolume_Slope × meta_volume_shock
            "RC_PriceSMA_x_Trendiness",    # Price_vs_SMA20 × meta_trendiness
            "RC_VWAPSlope_x_Trendiness",   # KF_VWAP_Slope × meta_trendiness
            "RC_ATRRatio_x_MetaVolRegime", # KF_ATR_Ratio × meta_volatility_regime
        ],
        "cross_kalman": [
            "KC_FilteredRawATR_Ratio",     # KF_ATR / ATR_14
            "KC_VolSlope_per_Vol5",        # KF_LogVolume_Slope / volatility_5
            "KC_Velocity_per_KalmanP",     # KF_Velocity / KF_P
            "KC_VWAPZscore_x_VolRatio",    # KF_VWAP_Zscore × KF_Volume_Ratio
            "KC_MomPerVol_x_VolSlope",     # Momentum_per_vol × KF_LogVolume_Slope
        ],
        # Path, Entropy, and Liquidity features
        "cross_path_efficiency": [
            "PATH_ER_x_Momentum",          # Kaufman ER × Momentum
            "PATH_ER_x_Volatility",        # Kaufman ER × Volatility
            "PATH_ER_x_VolRatio",          # Kaufman ER × Volume Ratio
            "PATH_Efficiency_10",          # 10-bar path efficiency
            "PATH_Efficiency_30",          # 30-bar path efficiency
            "PATH_Efficiency_Divergence",  # Short vs long path efficiency
        ],
        "cross_entropy_complexity": [
            "ENT_Return_x_Momentum",       # Return entropy × Momentum
            "ENT_Return_x_Volatility",     # Return entropy × Volatility
            "ENT_ApproxEntropy_20",        # Approximate entropy proxy
            "ENT_PermEntropy_Proxy",       # Permutation entropy proxy
            "ENT_PathComplexity",          # Price path complexity
        ],
        "cross_liquidity_proxy": [
            "LIQ_Imbalance_x_Momentum",    # Volume imbalance × Momentum
            "LIQ_Imbalance_x_ATR",         # Volume imbalance × ATR
            "LIQ_Amihud_Ratio",            # Amihud illiquidity ratio
            "LIQ_KyleLambda",              # Kyle's lambda (price impact)
            "LIQ_RollsSpread",             # Roll's spread estimator
            "LIQ_VolPressure",             # Volume pressure ratio
            "LIQ_HLSpread_Ratio",          # High-Low spread ratio
            "LIQ_ParkinsonVol",            # Parkinson volatility
        ],
        "cross_pel_interactions": [
            "PEL_PathEff_x_Entropy",       # Path efficiency × Entropy
            "PEL_Amihud_x_Entropy",        # Amihud × Entropy
            "PEL_PathEff_x_Liquidity",     # Path efficiency × Liquidity
        ],
    }
    
    # Calculate totals
    configurable_count = len(inventory["configurable_technical"]) * 3  # 3 horizons
    fixed_count = len(inventory["fixed_specialist"])
    kalman_count = (
        len(inventory["kalman_price"]) + 
        len(inventory["kalman_vwap"]) + 
        len(inventory["kalman_volume"])
    )
    cross_feature_count = (
        len(inventory["cross_price_volume"]) +
        len(inventory["cross_volatility_normalized"]) +
        len(inventory["cross_horizon_divergence"]) +
        len(inventory["cross_regime_conditional"]) +
        len(inventory["cross_kalman"]) +
        len(inventory["cross_path_efficiency"]) +
        len(inventory["cross_entropy_complexity"]) +
        len(inventory["cross_liquidity_proxy"]) +
        len(inventory["cross_pel_interactions"])
    )
    
    inventory["_counts"] = {
        "configurable_technical": configurable_count,
        "fixed_specialist": fixed_count,
        "kalman_features": kalman_count,
        "cross_features": cross_feature_count,
        "path_entropy_liquidity": (
            len(inventory["cross_path_efficiency"]) +
            len(inventory["cross_entropy_complexity"]) +
            len(inventory["cross_liquidity_proxy"]) +
            len(inventory["cross_pel_interactions"])
        ),
        "total_base": configurable_count + fixed_count + kalman_count + cross_feature_count,
        "with_multi_horizon": (configurable_count + fixed_count + kalman_count) * 7 + cross_feature_count,
    }
    
    return inventory


# ============================================================================
# FEATURE SELECTION CACHING
# ============================================================================

# Global cache for feature selection results
_FEATURE_SELECTION_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_feature_selection_cache_key(
    symbol: str,
    exchange: str,
    timeframe: str,
) -> str:
    """Generate cache key for feature selection results."""
    return f"{symbol}_{exchange}_{timeframe}"


def _get_feature_selection_cache_path(
    symbol: str,
    exchange: str,
    timeframe: str,
) -> Path:
    """Get file path for cached feature selection results."""
    cache_dir = Path("cache") / "feature_selection"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"feature_selection_{symbol}_{exchange}_{timeframe}.json"


def load_cached_feature_selection(
    symbol: str,
    exchange: str,
    timeframe: str,
) -> Optional[Dict[str, Any]]:
    """
    Load cached feature selection results for an asset/exchange/timeframe.
    
    Returns:
        Dict with 'selected_features' (list) and 'quality_scores' (dict), or None if not cached
    """
    cache_key = _get_feature_selection_cache_key(symbol, exchange, timeframe)
    
    # Check in-memory cache first
    if cache_key in _FEATURE_SELECTION_CACHE:
        tprint_info(f"   📦 Loaded feature selection from memory cache")
        return _FEATURE_SELECTION_CACHE[cache_key]
    
    # Check file cache
    cache_path = _get_feature_selection_cache_path(symbol, exchange, timeframe)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            _FEATURE_SELECTION_CACHE[cache_key] = cached
            tprint_info(f"   📦 Loaded feature selection from {cache_path}")
            return cached
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to load cache: {e}")
    
    return None


def save_feature_selection_cache(
    symbol: str,
    exchange: str,
    timeframe: str,
    selected_features: List[str],
    quality_scores: Dict[str, float],
) -> None:
    """
    Save feature selection results to cache.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        selected_features: List of selected feature names
        quality_scores: Dict mapping feature name to quality score
    """
    cache_key = _get_feature_selection_cache_key(symbol, exchange, timeframe)
    
    cache_data = {
        'selected_features': selected_features,
        'quality_scores': quality_scores,
        'timestamp': datetime.now().isoformat(),
        'n_features': len(selected_features),
    }
    
    # Save to memory
    _FEATURE_SELECTION_CACHE[cache_key] = cache_data
    
    # Save to file
    cache_path = _get_feature_selection_cache_path(symbol, exchange, timeframe)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        tprint_info(f"   💾 Saved feature selection to {cache_path}")
    except Exception as e:
        tprint_warning(f"   ⚠️ Failed to save cache: {e}")


def invalidate_feature_selection_cache(
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> int:
    """
    Invalidate feature selection cache.
    
    Args:
        symbol: If provided, only invalidate for this symbol
        exchange: If provided, only invalidate for this exchange
        timeframe: If provided, only invalidate for this timeframe
        
    Returns:
        Number of cache entries invalidated
    """
    global _FEATURE_SELECTION_CACHE
    
    if symbol is None and exchange is None and timeframe is None:
        # Clear all
        count = len(_FEATURE_SELECTION_CACHE)
        _FEATURE_SELECTION_CACHE = {}
        return count
    
    # Selective invalidation
    keys_to_remove = []
    for key in _FEATURE_SELECTION_CACHE:
        parts = key.split('_')
        if len(parts) >= 3:
            k_symbol, k_exchange, k_timeframe = parts[0], parts[1], '_'.join(parts[2:])
            if ((symbol is None or symbol == k_symbol) and
                (exchange is None or exchange == k_exchange) and
                (timeframe is None or timeframe == k_timeframe)):
                keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del _FEATURE_SELECTION_CACHE[key]
    
    return len(keys_to_remove)


# ============================================================================
# MAIN FEATURE SELECTION PIPELINE (De Prado-Inspired)
# ============================================================================

def select_features_with_quality(
    df_features: pd.DataFrame,
    target_n: int = 70,
    correlation_threshold: float = 0.85,
    generate_horizons: bool = True,
    horizon_config: Dict[str, int] = None,
    enable_cross_features: bool = True,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    # New De Prado pipeline parameters
    use_hierarchical: bool = True,
    use_lgbm_sweep: bool = True,
    lgbm_lookahead: int = 4,
    lgbm_max_features: int = 300,
    quality_drop_percentile: float = 20.0,
    min_std_threshold: float = 1e-9,
    # Caching parameters
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframe: Optional[str] = None,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Complete feature selection pipeline with De Prado's hierarchical method.
    
    PIPELINE (Anti-Concept-Dominance):
    ==================================
    1. Generate multi-horizon features
    2. Generate cross-feature interactions
    3. DROP features with std < min_std_threshold (constant/near-constant)
    4. Calculate time-robust quality scores (worst-case across chunks)
    5. DROP bottom quality_drop_percentile% by quality
    6. LGBM Magnitude Sweep: Keep top lgbm_max_features that predict future volatility
    7. Hierarchical Clustering: Select target_n features from N distinct clusters
    
    The hierarchical clustering GUARANTEES diversity by:
    - Grouping correlated features into clusters
    - Selecting ONLY the best feature from each cluster
    - This prevents "concept dominance" where RSI variants crowd out volume features
    
    CACHING:
    ========
    Results are cached per symbol/exchange/timeframe to avoid recomputation.
    Set use_cache=False or force_recompute=True to bypass caching.
    
    Args:
        df_features: DataFrame with raw features (base + Kalman merged)
        target_n: Target number of features to select
        correlation_threshold: Correlation threshold (used in fallback greedy method)
        generate_horizons: Whether to create multi-horizon versions
        horizon_config: Custom horizon configuration
        enable_cross_features: Whether to generate cross-feature interactions
        market_data: Original market data (required for LGBM sweep and cross-features)
        use_hierarchical: Use De Prado hierarchical selection (default True)
        use_lgbm_sweep: Use LGBM magnitude sweep pre-filter (default True)
        lgbm_lookahead: Bars to look ahead for magnitude proxy (default 4)
        lgbm_max_features: Max features to keep after LGBM sweep (default 300)
        quality_drop_percentile: Drop bottom X% by quality (default 20%)
        min_std_threshold: Drop features with std below this (default 1e-9)
        symbol: Trading symbol (for caching)
        exchange: Exchange name (for caching)
        timeframe: Timeframe (for caching)
        use_cache: Whether to use cached results (default True)
        force_recompute: Force recomputation even if cached (default False)
    
    Returns:
        Tuple of (reduced_features_df, quality_scores_dict)
    """
    tprint_info("🔍 Starting De Prado feature selection pipeline...")

    cfg = config if isinstance(config, dict) else {}
    enable_proxy_two_stage = bool(cfg.get("feature_selection_proxy_two_stage", True))
    proxy_stage1_target = int(cfg.get("feature_selection_proxy_stage1_target", 128))
    proxy_signature_bins = int(cfg.get("feature_selection_proxy_signature_bins", 64))
    proxy_anchor_count = int(cfg.get("feature_selection_proxy_anchor_count", 64))
    proxy_max_rows = int(cfg.get("feature_selection_proxy_max_rows", 5000))
    
    # =========================================================================
    # STEP 0: Check cache (load now, validate after expansion)
    # =========================================================================
    cached = None
    if use_cache and not force_recompute and symbol and exchange and timeframe:
        cached = load_cached_feature_selection(symbol, exchange, timeframe)
        if cached:
            tprint_info("   📦 Cached selection loaded; will validate after expansion")
    
    # =========================================================================
    # STEP 1: Generate multi-horizon features
    # =========================================================================
    if generate_horizons:
        tprint_info(f"   [1/7] Generating multi-horizon features...")
        initial_cols = len(df_features.columns)
        df_expanded = generate_multi_horizon_features(df_features, horizon_config)
        tprint_info(f"         Expanded: {initial_cols} → {len(df_expanded.columns)} features")
    else:
        df_expanded = df_features.copy()
    
    # Validate cache against the expanded feature set (post-horizon generation).
    if cached:
        selected_cols = cached['selected_features']
        quality_scores = cached['quality_scores']
        available_cols = [c for c in selected_cols if c in df_expanded.columns]
        if len(available_cols) >= target_n * 0.8:  # Allow 20% missing
            tprint_success(f"✅ Using cached selection: {len(available_cols)} features")
            return df_expanded[available_cols], {c: quality_scores.get(c, 0.0) for c in available_cols}
        tprint_warning(f"   ⚠️ Cache invalid: only {len(available_cols)}/{len(selected_cols)} features found")
        missing_cols = [c for c in selected_cols if c not in available_cols]
        if missing_cols:
            tprint_info(
                f"   Missing cached features (sample): {missing_cols[:10]} (total missing={len(missing_cols)})"
            )
    
    # =========================================================================
    # STEP 2: Generate cross-feature interactions
    # =========================================================================
    if enable_cross_features:
        tprint_info("   [2/7] Generating cross-feature interactions...")
        try:
            kalman_cols = [c for c in df_expanded.columns if c.startswith("KF_")]
            base_cols = [c for c in df_expanded.columns if not c.startswith("KF_")]
            
            kalman_features_df = df_expanded[kalman_cols] if kalman_cols else pd.DataFrame(index=df_expanded.index)
            base_features_df = df_expanded[base_cols] if base_cols else pd.DataFrame(index=df_expanded.index)
            
            cross_features_df = generate_cross_features(
                base_features=base_features_df,
                kalman_features=kalman_features_df,
                market_data=market_data if market_data is not None else pd.DataFrame(index=df_expanded.index),
            )
            
            n_cross = len(cross_features_df.columns)
            for col in cross_features_df.columns:
                if col not in df_expanded.columns:
                    df_expanded[col] = cross_features_df[col]
            
            tprint_info(f"         Added {n_cross} cross-feature interactions")
        except Exception as e:
            tprint_warning(f"         ⚠️ Cross-feature generation failed: {e}")
    
    n_after_expansion = len(df_expanded.columns)
    
    # =========================================================================
    # STEP 3: Drop constant/near-constant features (std < threshold)
    # =========================================================================
    tprint_info(f"   [3/7] Dropping constant features (std < {min_std_threshold})...")
    col_stds = df_expanded.std()
    valid_std_cols = col_stds[col_stds >= min_std_threshold].index.tolist()
    dropped_constant_cols = [c for c in df_expanded.columns if col_stds.get(c, 0.0) < min_std_threshold]
    
    n_dropped_std = len(dropped_constant_cols)
    if n_dropped_std > 0:
        sample_constant = dropped_constant_cols[:10]
        tprint_info(f"         Dropped {n_dropped_std} constant features: {sample_constant}")
        df_expanded = df_expanded[valid_std_cols]
    
    # =========================================================================
    # STEP 4: Calculate time-robust quality scores
    # =========================================================================
    tprint_info("   [4/7] Calculating time-robust quality scores...")
    quality_scores = calculate_all_feature_qualities(df_expanded, use_time_robust=True)
    
    # Log top/bottom quality features
    sorted_by_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_by_quality:
        top_3 = sorted_by_quality[:3]
        bottom_3 = sorted_by_quality[-3:]
        tprint_info(f"         Top 3: {[(n, f'{q:.3f}') for n, q in top_3]}")
        tprint_info(f"         Bottom 3: {[(n, f'{q:.3f}') for n, q in bottom_3]}")
    
    # =========================================================================
    # STEP 5: Drop bottom quality_drop_percentile% by quality
    # =========================================================================
    tprint_info(f"   [5/7] Dropping bottom {quality_drop_percentile}% by quality...")
    
    if sorted_by_quality:
        n_total = len(sorted_by_quality)
        n_drop_target = int(np.floor(n_total * (quality_drop_percentile / 100.0)))

        # Ensure we always keep at least one feature if any exist
        n_drop_target = max(0, min(n_drop_target, n_total - 1))

        if n_drop_target > 0:
            kept_by_rank = sorted_by_quality[:-n_drop_target]
            dropped_by_rank = sorted_by_quality[-n_drop_target:]
            quality_filtered_cols = [col for col, _ in kept_by_rank]
            quality_threshold = kept_by_rank[-1][1] if kept_by_rank else dropped_by_rank[-1][1]
            n_dropped_quality = n_drop_target
        else:
            quality_filtered_cols = [col for col, _ in sorted_by_quality]
            quality_threshold = sorted_by_quality[-1][1]
            n_dropped_quality = 0
        
        if n_dropped_quality > 0:
            dropped_quality_cols = [c for c, _ in dropped_by_rank]
            sample_quality = [(c, quality_scores.get(c, 0.0)) for c in dropped_quality_cols[:10]]
            tprint_info(
                f"         Dropped {n_dropped_quality} low-quality features "
                f"(threshold={quality_threshold:.4f}); sample={sample_quality}"
            )
            df_expanded = df_expanded[quality_filtered_cols]
            quality_scores = {c: quality_scores[c] for c in quality_filtered_cols}
    
    # =========================================================================
    # STEP 6: LGBM Magnitude Sweep
    # =========================================================================
    if use_lgbm_sweep and market_data is not None and len(df_expanded.columns) > lgbm_max_features:
        tprint_info(f"   [6/7] LGBM Magnitude Sweep (lookahead={lgbm_lookahead}, max={lgbm_max_features})...")
        try:
            df_expanded = lgbm_magnitude_sweep(
                df_features=df_expanded,
                market_data=market_data,
                lookahead=lgbm_lookahead,
                max_features=lgbm_max_features,
            )
            # Update quality scores for remaining features
            quality_scores = {c: quality_scores.get(c, 0.0) for c in df_expanded.columns}
        except Exception as e:
            tprint_warning(f"         ⚠️ LGBM sweep failed: {e}")
    else:
        tprint_info(f"   [6/7] Skipping LGBM sweep (features={len(df_expanded.columns)} <= {lgbm_max_features})")

    # =========================================================================
    # OPTIONAL FAST PROXY SELECTION (Two-stage)
    # =========================================================================
    if enable_proxy_two_stage and len(df_expanded.columns) > target_n:
        tprint_info(
            f"   [7/7] Proxy selection (rank-signature→{proxy_stage1_target}, anchors→{target_n}; rows≤{proxy_max_rows})..."
        )
        try:
            df_stage1, q_stage1 = preprune_by_rank_signature(
                df_features=df_expanded,
                quality_scores=quality_scores,
                target_n=max(target_n, proxy_stage1_target),
                max_rows=proxy_max_rows,
                n_bins=proxy_signature_bins,
                seed=42,
            )
            tprint_info(f"         Stage1 signatures: {len(df_expanded.columns)} → {len(df_stage1.columns)}")

            df_reduced, selected_quality = select_by_anchor_farthest_first(
                df_features=df_stage1,
                quality_scores=q_stage1,
                target_n=target_n,
                n_anchors=proxy_anchor_count,
                max_rows=proxy_max_rows,
                seed=42,
            )
            tprint_success(f"✅ Proxy selection complete: {len(df_expanded.columns)} → {len(df_reduced.columns)}")
            # Save to cache
            if use_cache and symbol and exchange and timeframe:
                save_feature_selection_cache(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    selected_features=list(df_reduced.columns),
                    quality_scores=selected_quality,
                )
            return df_reduced, selected_quality
        except Exception as proxy_exc:
            tprint_warning(f"         ⚠️ Proxy selection failed: {proxy_exc}. Falling back to hierarchical/greedy.")
    
    # =========================================================================
    # STEP 7: Final Selection (Hierarchical or Greedy)
    # =========================================================================
    if use_hierarchical and len(df_expanded.columns) > target_n:
        tprint_info(f"   [7/7] Hierarchical clustering selection (target={target_n})...")
        try:
            df_reduced = select_features_hierarchical(
                df_features=df_expanded,
                quality_scores=quality_scores,
                target_n=target_n,
                target_series=None,  # Target data not available at HPO level
                autocorr_penalty_weight=0.0,  # Disable autocorr penalty at HPO level
            )
        except Exception as e:
            tprint_warning(f"         ⚠️ Hierarchical failed: {e}, falling back to greedy")
            df_reduced = reduce_features_by_correlation(
                df_features=df_expanded,
                quality_scores=quality_scores,
                target_n=target_n,
                correlation_threshold=correlation_threshold,
            )
    else:
        tprint_info(f"   [7/7] Greedy correlation reduction (target={target_n})...")
        df_reduced = reduce_features_by_correlation(
            df_features=df_expanded,
            quality_scores=quality_scores,
            target_n=target_n,
            correlation_threshold=correlation_threshold,
        )
    
    # Final quality scores for selected features
    selected_quality = {col: quality_scores.get(col, 0.0) for col in df_reduced.columns}
    
    # =========================================================================
    # Save to cache
    # =========================================================================
    if use_cache and symbol and exchange and timeframe:
        save_feature_selection_cache(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            selected_features=list(df_reduced.columns),
            quality_scores=selected_quality,
        )
    
    tprint_success(
        f"✅ Feature selection complete: {n_after_expansion} → {len(df_reduced.columns)} features"
    )
    
    return df_reduced, selected_quality


def generate_kalman_features(
    market_data: pd.DataFrame,
    kalman_Q: float,
    kalman_R: float,
) -> pd.DataFrame:
    """
    Generate Kalman-based features for the weighted pipeline.
    
    Uses CAUSAL Kalman Filter (not RTS) for features that can be used in live trading.
    RTS is only used for label generation (acausal, look-ahead).
    
    All price-derivative features are LOG-NORMALIZED for better ML performance.
    
    Features generated:
    - KF_Close, KF_High, KF_Low: Filtered OHLC (log-returns vs raw)
    - KF_Velocity: 1st derivative of filtered close (log-normalized)
    - KF_Acceleration: 2nd derivative of filtered close (log-normalized)
    - KF_Slope: Rolling slope of filtered close (log-normalized)
    - KF_P: Error covariance (log-normalized)
    - KF_RSI: RSI computed on filtered close (already bounded 0-100)
    - KF_BB_Distance: Distance from Kalman Bollinger Band (normalized)
    - KF_ATR: Kalman-filtered ATR (log-normalized)
    - KF_VWAP: Kalman-filtered VWAP distance (normalized)
    - KF_LogVolume: Filtered log volume
    - KF_LogVolume_Slope: Slope of filtered log volume
    - KF_Volume_Zscore: Standardized volume innovation
    - KF_Volume_Ratio: Current vs filtered volume ratio (log-normalized)
    - KF_Volume_P: Volume error covariance (log-normalized)
    
    Args:
        market_data: DataFrame with OHLCV data
        kalman_Q: Process noise (from Stage 0 optimization)
        kalman_R: Measurement noise (from Stage 0 optimization)
    
    Returns:
        DataFrame with Kalman features (log-normalized where appropriate)
    """
    features = pd.DataFrame(index=market_data.index)
    
    # Extract price series
    close = market_data["close"].values
    high = market_data.get("high", market_data["close"]).values
    low = market_data.get("low", market_data["close"]).values
    open_price = market_data.get("open", market_data["close"]).values
    
    # --- Price-Based Kalman Features ---
    
    # 1. Filtered OHLC
    kf_close, kf_close_P, _ = kalman_filter_1d(close, Q=kalman_Q, R=kalman_R)
    kf_high, kf_high_P, _ = kalman_filter_1d(high, Q=kalman_Q, R=kalman_R)
    kf_low, kf_low_P, _ = kalman_filter_1d(low, Q=kalman_Q, R=kalman_R)
    
    # Store as log-returns relative to raw (normalized difference)
    features["KF_Close_LogRet"] = _log_normalize((kf_close - close) / (close + 1e-9))
    features["KF_High_LogRet"] = _log_normalize((kf_high - high) / (high + 1e-9))
    features["KF_Low_LogRet"] = _log_normalize((kf_low - low) / (low + 1e-9))
    
    # 2. Kalman Velocity (1st derivative) - LOG-NORMALIZED
    kf_velocity = np.zeros_like(kf_close)
    kf_velocity[1:] = np.diff(kf_close) / (kf_close[:-1] + 1e-9)  # Percent change
    features["KF_Velocity"] = _log_normalize(kf_velocity)
    
    # 3. Kalman Acceleration (2nd derivative) - LOG-NORMALIZED
    kf_accel = np.zeros_like(kf_close)
    kf_accel[2:] = np.diff(kf_velocity[1:])  # Change in velocity
    features["KF_Acceleration"] = _log_normalize(kf_accel)
    
    # 4. Kalman Slope (rolling regression slope) - LOG-NORMALIZED
    kf_slope = np.zeros_like(kf_close)
    slope_window = 10
    for i in range(slope_window, len(kf_close)):
        y = kf_close[i-slope_window:i]
        x = np.arange(slope_window)
        if np.std(y) > 1e-9:
            slope = np.polyfit(x, y, 1)[0]
            # Normalize by price level
            kf_slope[i] = slope / (kf_close[i] + 1e-9)
    features["KF_Slope"] = _log_normalize(kf_slope)
    
    # 5. Kalman P (Error Covariance / Uncertainty) - LOG-NORMALIZED
    features["KF_P"] = _log_normalize(kf_close_P)
    
    # 6. Kalman RSI (RSI on filtered close) - Already bounded 0-100
    kf_returns = np.diff(kf_close, prepend=kf_close[0])
    gains = np.where(kf_returns > 0, kf_returns, 0)
    losses = np.where(kf_returns < 0, -kf_returns, 0)
    
    # Exponential moving average for RSI
    rsi_period = 14
    avg_gain = pd.Series(gains).ewm(span=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-9)
    kf_rsi = 100 - (100 / (1 + rs))
    # Normalize RSI to [-1, 1] range for consistency
    features["KF_RSI"] = (kf_rsi - 50) / 50
    
    # 7. KF_Close - KF_Bollinger (distance from Kalman Bollinger Band)
    bb_window = 20
    kf_close_series = pd.Series(kf_close)
    kf_ma = kf_close_series.rolling(bb_window).mean()
    kf_std = kf_close_series.rolling(bb_window).std()
    kf_upper_bb = kf_ma + 2 * kf_std
    kf_lower_bb = kf_ma - 2 * kf_std
    
    # Normalized distance from center band (already normalized by band width)
    bb_width = (kf_upper_bb - kf_lower_bb).replace(0, np.nan)
    kf_bb_distance = (kf_close_series - kf_ma) / (bb_width / 2 + 1e-9)
    features["KF_BB_Distance"] = kf_bb_distance.values
    
    # --- NEW: Kalman ATR ---
    # 8. Compute True Range and filter with Kalman
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # Handle first element
    tr3[0] = tr1[0]
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Filter True Range with Kalman (use higher R for smoothing)
    kf_tr, _, _ = kalman_filter_1d(true_range, Q=kalman_Q * 0.5, R=kalman_R * 3.0)
    
    # ATR as rolling mean of filtered TR
    kf_atr = pd.Series(kf_tr).rolling(14).mean().values
    
    # Normalize ATR by price level and log-transform
    kf_atr_normalized = kf_atr / (close + 1e-9)
    features["KF_ATR"] = _log_normalize(kf_atr_normalized)
    
    # Also store raw ATR ratio (KF_ATR vs standard ATR)
    raw_atr = pd.Series(true_range).rolling(14).mean().values
    atr_ratio = kf_atr / (raw_atr + 1e-9)
    features["KF_ATR_Ratio"] = _log_normalize(atr_ratio - 1.0)  # Center around 0
    
    # --- NEW: Kalman VWAP ---
    # 9. Volume-Weighted Average Price with Kalman filtering
    if "volume" in market_data.columns:
        volume = market_data["volume"].values
        
        # Typical price
        typical_price = (high + low + close) / 3.0
        
        # Cumulative VWAP components
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)
        
        # Raw VWAP
        raw_vwap = cumulative_tp_vol / (cumulative_vol + 1e-9)
        
        # Filter VWAP with Kalman
        kf_vwap, kf_vwap_P, _ = kalman_filter_1d(raw_vwap, Q=kalman_Q * 0.3, R=kalman_R * 2.0)
        
        # Distance from VWAP (normalized by price)
        vwap_distance = (close - kf_vwap) / (close + 1e-9)
        features["KF_VWAP_Distance"] = _log_normalize(vwap_distance)
        
        # VWAP slope (momentum of VWAP)
        kf_vwap_slope = np.zeros_like(kf_vwap)
        kf_vwap_slope[1:] = np.diff(kf_vwap) / (kf_vwap[:-1] + 1e-9)
        features["KF_VWAP_Slope"] = _log_normalize(kf_vwap_slope)
        
        # Price position relative to VWAP bands
        vwap_std = pd.Series(close - kf_vwap).rolling(20).std().values
        vwap_zscore = (close - kf_vwap) / (vwap_std + 1e-9)
        features["KF_VWAP_Zscore"] = np.clip(vwap_zscore, -5, 5) / 5  # Normalize to [-1, 1]
    
    # --- Volume-Based Kalman Features ---
    if "volume" in market_data.columns:
        volume = market_data["volume"].values
        
        # Transform to log space
        log_volume = np.log(volume + 1)
        
        # Run Kalman Filter on log(Volume + 1)
        kf_log_vol, kf_vol_P, kf_vol_K = kalman_filter_1d(
            log_volume, Q=kalman_Q * 0.1, R=kalman_R * 2.0  # Different Q/R for volume
        )
        
        # 10. Smoothed Volume (back to linear space, then log-normalize the ratio)
        kf_volume_linear = np.exp(kf_log_vol)
        vol_diff = (kf_volume_linear - volume) / (volume + 1e-9)
        features["KF_Volume_Diff"] = _log_normalize(vol_diff)
        
        # 11. KF_LogVolume_Slope - already in log space
        kf_log_vol_slope = np.zeros_like(kf_log_vol)
        kf_log_vol_slope[1:] = np.diff(kf_log_vol)
        features["KF_LogVolume_Slope"] = kf_log_vol_slope  # Already log-scale
        
        # 12. Volume Z-score: (LogVolume - KF_Predicted_LogVolume) / sqrt(KF_Covariance)
        vol_innovation = log_volume - kf_log_vol
        vol_zscore = vol_innovation / (np.sqrt(kf_vol_P) + 1e-9)
        features["KF_Volume_Zscore"] = np.clip(vol_zscore, -5, 5) / 5  # Normalize
        
        # 13. Volume Ratio: log of ratio for symmetry
        vol_ratio = volume / (kf_volume_linear + 1e-9)
        features["KF_Volume_Ratio"] = _log_normalize(vol_ratio - 1.0)  # Center around 0
        
        # 14. Volume Error Covariance - LOG-NORMALIZED
        features["KF_Volume_P"] = _log_normalize(kf_vol_P)
    
    # Fill NaN values
    features = features.fillna(0)
    
    # Final cleanup: replace infinities with 0
    features = features.replace([np.inf, -np.inf], 0)
    
    return features


# ============================================================================
# CROSS-FEATURE INTERACTIONS (Price-Volume, Volatility-Normalized, etc.)
# ============================================================================

def generate_cross_features(
    base_features: pd.DataFrame,
    kalman_features: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate sophisticated cross-feature interactions combining price, volume,
    volatility, and regime signals.
    
    These features capture non-linear relationships that single features miss:
    - Price-Volume Interactions: Momentum scaled by participation
    - Volatility-Normalized Features: Signals adjusted for current volatility
    - Cross-Horizon Divergence: Short vs long-term signal disagreements
    - Regime-Conditional Features: Signals weighted by market regime
    - Kalman Cross-Features: Filtered signal combinations
    
    Args:
        base_features: DataFrame with base meta-features (from create_meta_features)
        kalman_features: DataFrame with Kalman-filtered features (from generate_kalman_features)
        market_data: Original OHLCV data for any additional calculations
    
    Returns:
        DataFrame with cross-feature interactions (log-normalized where appropriate)
    """
    tprint_info("🔧 generate_cross_features() called")
    tprint_info(f"   base_features: {base_features.shape}, kalman_features: {kalman_features.shape}")
    
    cross = pd.DataFrame(index=base_features.index)
    
    # Helper to safely get feature, returning zeros if not found
    def _safe_get(df: pd.DataFrame, col: str) -> np.ndarray:
        if col in df.columns:
            return df[col].fillna(0).values
        return np.zeros(len(df))
    
    # Merge available features for easier access
    all_features = pd.concat([base_features, kalman_features], axis=1)
    
    # =========================================================================
    # 1. PRICE-VOLUME INTERACTIONS
    # =========================================================================
    # These capture momentum scaled by market participation/volume trends
    
    # KF_Velocity × KF_LogVolume_Slope: Momentum scaled by participation trend
    kf_velocity = _safe_get(all_features, "KF_Velocity")
    kf_logvol_slope = _safe_get(all_features, "KF_LogVolume_Slope")
    cross["PV_Velocity_x_VolSlope"] = _log_normalize(kf_velocity * kf_logvol_slope)
    
    # KF_VWAP_Distance × KF_Volume_Zscore: Price deviation from VWAP weighted by volume surprise
    kf_vwap_dist = _safe_get(all_features, "KF_VWAP_Distance")
    kf_vol_zscore = _safe_get(all_features, "KF_Volume_Zscore")
    cross["PV_VWAP_x_VolZscore"] = _log_normalize(kf_vwap_dist * kf_vol_zscore)
    
    # KF_Close_LogRet × KF_Volume_Ratio: Return scaled by abnormal volume
    kf_close_logret = _safe_get(all_features, "KF_Close_LogRet")
    kf_vol_ratio = _safe_get(all_features, "KF_Volume_Ratio")
    cross["PV_Return_x_VolRatio"] = _log_normalize(kf_close_logret * kf_vol_ratio)
    
    # SMA_Distance × Volume_SMA_Ratio: Price deviation from SMA weighted by volume trend
    # Try multiple horizon variants
    for horizon in ["_Short", "_Medium", "_Long", ""]:
        sma_dist_col = f"SMA_Distance{horizon}" if horizon else "SMA_Distance_Medium"
        vol_sma_col = f"Volume_SMA_Ratio{horizon}" if horizon else "Volume_SMA_Ratio_Medium"
        sma_dist = _safe_get(all_features, sma_dist_col)
        vol_sma_ratio = _safe_get(all_features, vol_sma_col)
        if np.any(sma_dist != 0) and np.any(vol_sma_ratio != 0):
            cross[f"PV_SMADist_x_VolRatio{horizon}"] = _log_normalize(sma_dist * vol_sma_ratio)
            break  # Use first available
    
    # ROC × KF_Volume_P: Rate-of-change scaled by volume uncertainty
    for horizon in ["_Short", "_Medium", "_Long", ""]:
        roc_col = f"ROC{horizon}" if horizon else "ROC_Medium"
        roc = _safe_get(all_features, roc_col)
        kf_vol_p = _safe_get(all_features, "KF_Volume_P")
        if np.any(roc != 0):
            cross["PV_ROC_x_VolP"] = _log_normalize(roc * kf_vol_p)
            break
    
    # =========================================================================
    # 2. VOLATILITY-NORMALIZED FEATURES
    # =========================================================================
    # These normalize signals by current volatility regime
    
    # KF_Velocity / KF_ATR: Normalized velocity relative to volatility
    kf_atr = _safe_get(all_features, "KF_ATR")
    safe_atr = np.where(np.abs(kf_atr) > 1e-9, kf_atr, 1e-9)
    cross["VN_Velocity_per_ATR"] = _log_normalize(kf_velocity / safe_atr)
    
    # Momentum_5 / volatility_5: Short-term momentum scaled by short-term volatility
    momentum_5 = _safe_get(all_features, "momentum_5")
    volatility_5 = _safe_get(all_features, "volatility_5")
    safe_vol5 = np.where(np.abs(volatility_5) > 1e-9, volatility_5, 1e-9)
    cross["VN_Mom5_per_Vol5"] = _log_normalize(momentum_5 / safe_vol5)
    
    # KF_Acceleration / KF_ATR_Ratio: Trend change relative to filtered volatility
    kf_accel = _safe_get(all_features, "KF_Acceleration")
    kf_atr_ratio = _safe_get(all_features, "KF_ATR_Ratio")
    safe_atr_ratio = np.where(np.abs(kf_atr_ratio) > 1e-9, kf_atr_ratio, 1e-9)
    cross["VN_Accel_per_ATRRatio"] = _log_normalize(kf_accel / safe_atr_ratio)
    
    # ROC × Volatility_Regime: Rate-of-change weighted by market volatility regime
    # Use vol_regime_high as a continuous proxy (0 or 1)
    vol_regime_high = _safe_get(all_features, "vol_regime_high")
    vol_regime_med = _safe_get(all_features, "vol_regime_medium")
    # Create regime score: 0 (low), 0.5 (medium), 1.0 (high)
    regime_score = vol_regime_med * 0.5 + vol_regime_high * 1.0
    for horizon in ["_Short", "_Medium", "_Long", ""]:
        roc_col = f"ROC{horizon}" if horizon else "ROC_Medium"
        roc = _safe_get(all_features, roc_col)
        if np.any(roc != 0):
            cross["VN_ROC_x_VolRegime"] = _log_normalize(roc * (1.0 + regime_score))
            break
    
    # KF_Slope × KF_P: Price slope weighted by Kalman state uncertainty
    kf_slope = _safe_get(all_features, "KF_Slope")
    kf_p = _safe_get(all_features, "KF_P")
    cross["VN_Slope_x_KalmanP"] = _log_normalize(kf_slope * kf_p)
    
    # =========================================================================
    # 3. CROSS-HORIZON DIVERGENCE FEATURES
    # =========================================================================
    # These capture disagreements between short and long-term signals
    
    # RSI_Short - RSI_Long: Overbought/oversold divergence across horizons
    rsi_short = _safe_get(all_features, "RSI_Short")
    rsi_long = _safe_get(all_features, "RSI_Long")
    if np.any(rsi_short != 0) or np.any(rsi_long != 0):
        cross["XH_RSI_Divergence"] = rsi_short - rsi_long  # Already normalized [-1, 1]
    
    # Velocity_Short / Velocity_Long: Short-term vs long-term momentum ratio
    # Use Momentum features as velocity proxies
    mom_short = _safe_get(all_features, "Momentum_Short")
    mom_long = _safe_get(all_features, "Momentum_Long")
    safe_mom_long = np.where(np.abs(mom_long) > 1e-9, mom_long, np.sign(mom_long) * 1e-9)
    safe_mom_long = np.where(safe_mom_long == 0, 1e-9, safe_mom_long)
    if np.any(mom_short != 0) or np.any(mom_long != 0):
        cross["XH_Momentum_Ratio"] = _log_normalize(mom_short / safe_mom_long)
    
    # ATR_Short / ATR_Long: Short-term vs long-term volatility ratio
    atr_short = _safe_get(all_features, "ATR_Short")
    atr_long = _safe_get(all_features, "ATR_Long")
    safe_atr_long = np.where(np.abs(atr_long) > 1e-9, atr_long, 1e-9)
    if np.any(atr_short != 0) or np.any(atr_long != 0):
        cross["XH_ATR_Ratio"] = _log_normalize(atr_short / safe_atr_long)
    
    # SMA_Distance_Short - SMA_Distance_Long: Mean-reversion signals across horizons
    sma_dist_short = _safe_get(all_features, "SMA_Distance_Short")
    sma_dist_long = _safe_get(all_features, "SMA_Distance_Long")
    if np.any(sma_dist_short != 0) or np.any(sma_dist_long != 0):
        cross["XH_SMADist_Divergence"] = _log_normalize(sma_dist_short - sma_dist_long)
    
    # BB_Distance_Short - BB_Distance_Long: Relative band positioning between horizons
    bb_dist_short = _safe_get(all_features, "BB_Distance_Short")
    bb_dist_long = _safe_get(all_features, "BB_Distance_Long")
    if np.any(bb_dist_short != 0) or np.any(bb_dist_long != 0):
        cross["XH_BBDist_Divergence"] = bb_dist_short - bb_dist_long  # Already normalized
    
    # =========================================================================
    # 4. REGIME-CONDITIONAL FEATURES
    # =========================================================================
    # These weight signals by the current market regime state
    
    # Momentum × meta_volatility_regime: Trend strength conditional on volatility regime
    meta_vol_regime = _safe_get(all_features, "meta_volatility_regime")
    momentum_10 = _safe_get(all_features, "momentum_10")
    if np.any(meta_vol_regime != 0):
        cross["RC_Mom_x_MetaVolRegime"] = _log_normalize(momentum_10 * meta_vol_regime)
    
    # KF_LogVolume_Slope × meta_volume_shock: Participation trend weighted by volume shock
    meta_vol_shock = _safe_get(all_features, "meta_volume_shock")
    if np.any(meta_vol_shock != 0):
        cross["RC_VolSlope_x_MetaVolShock"] = _log_normalize(kf_logvol_slope * meta_vol_shock)
    
    # Price_vs_SMA20 × meta_trendiness: Price deviation scaled by trendiness score
    price_vs_sma20 = _safe_get(all_features, "price_vs_sma20")
    meta_trendiness = _safe_get(all_features, "meta_trendiness")
    if np.any(meta_trendiness != 0):
        cross["RC_PriceSMA_x_Trendiness"] = _log_normalize(price_vs_sma20 * meta_trendiness)
    
    # KF_VWAP_Slope × meta_trendiness: VWAP trend scaled by regime trendiness
    kf_vwap_slope = _safe_get(all_features, "KF_VWAP_Slope")
    if np.any(meta_trendiness != 0) and np.any(kf_vwap_slope != 0):
        cross["RC_VWAPSlope_x_Trendiness"] = _log_normalize(kf_vwap_slope * meta_trendiness)
    
    # KF_ATR_Ratio × meta_volatility_regime: Volatility relative to baseline under current regime
    if np.any(meta_vol_regime != 0):
        cross["RC_ATRRatio_x_MetaVolRegime"] = _log_normalize(kf_atr_ratio * meta_vol_regime)
    
    # =========================================================================
    # 5. KALMAN CROSS-FEATURES
    # =========================================================================
    # These combine Kalman-filtered signals for enhanced signal quality
    
    # KF_ATR / ATR_14: Filtered vs raw volatility ratio
    atr_14 = _safe_get(all_features, "atr_14")
    safe_atr_14 = np.where(np.abs(atr_14) > 1e-9, atr_14, 1e-9)
    if np.any(kf_atr != 0):
        cross["KC_FilteredRawATR_Ratio"] = _log_normalize(kf_atr / safe_atr_14)
    
    # KF_LogVolume_Slope / volatility_5: Relative volume trend vs recent volatility
    if np.any(kf_logvol_slope != 0):
        cross["KC_VolSlope_per_Vol5"] = _log_normalize(kf_logvol_slope / safe_vol5)
    
    # KF_Velocity / KF_P: Velocity normalized by Kalman state uncertainty
    safe_kf_p = np.where(np.abs(kf_p) > 1e-9, kf_p, 1e-9)
    cross["KC_Velocity_per_KalmanP"] = _log_normalize(kf_velocity / safe_kf_p)
    
    # KF_VWAP_Zscore × KF_Volume_Ratio: Standardized VWAP innovation weighted by volume ratio
    kf_vwap_zscore = _safe_get(all_features, "KF_VWAP_Zscore")
    cross["KC_VWAPZscore_x_VolRatio"] = _log_normalize(kf_vwap_zscore * kf_vol_ratio)
    
    # Momentum_per_vol × KF_LogVolume_Slope: Momentum per unit volatility scaled by participation
    momentum_per_vol = _safe_get(all_features, "momentum_per_vol")
    cross["KC_MomPerVol_x_VolSlope"] = _log_normalize(momentum_per_vol * kf_logvol_slope)
    
    # =========================================================================
    # 6. PATH EFFICIENCY FEATURES
    # =========================================================================
    # These measure path quality - directness of price movement
    
    # Kaufman Efficiency Ratio: Already computed, but create interactions
    kaufman_er = _safe_get(all_features, "kaufman_efficiency_ratio")
    if np.any(kaufman_er != 0):
        # Path efficiency × momentum: Strong trends with efficient paths
        cross["PATH_ER_x_Momentum"] = _log_normalize(kaufman_er * momentum_10)
        
        # Path efficiency × volatility: Efficient paths in volatile markets
        cross["PATH_ER_x_Volatility"] = _log_normalize(kaufman_er * safe_vol5)
        
        # Path efficiency × volume: Efficient paths with volume confirmation
        vol_ratio = _safe_get(all_features, "volume_ratio")
        cross["PATH_ER_x_VolRatio"] = _log_normalize(kaufman_er * vol_ratio)
    
    # Compute path efficiency from market data if available
    if market_data is not None and 'close' in market_data.columns:
        close = market_data['close'].reindex(base_features.index)
        
        # Path Efficiency (10-bar): |Net Change| / Sum(|Changes|)
        net_change_10 = close.diff(10).abs()
        path_length_10 = close.diff().abs().rolling(10).sum()
        path_eff_10 = (net_change_10 / (path_length_10 + 1e-9)).fillna(0).values
        cross["PATH_Efficiency_10"] = path_eff_10
        
        # Path Efficiency (30-bar): Longer-term path quality
        net_change_30 = close.diff(30).abs()
        path_length_30 = close.diff().abs().rolling(30).sum()
        path_eff_30 = (net_change_30 / (path_length_30 + 1e-9)).fillna(0).values
        cross["PATH_Efficiency_30"] = path_eff_30
        
        # Path divergence: Short-term vs long-term efficiency
        cross["PATH_Efficiency_Divergence"] = _log_normalize(path_eff_10 - path_eff_30)
    
    # =========================================================================
    # 7. ENTROPY FEATURES (Predictability/Complexity)
    # =========================================================================
    # These measure market complexity and predictability
    
    # Returns entropy: Already computed in base features, create interactions
    returns_entropy = _safe_get(all_features, "returns_entropy")
    if np.any(returns_entropy != 0):
        # Entropy × momentum: Trend strength in complex markets
        cross["ENT_Return_x_Momentum"] = _log_normalize(returns_entropy * momentum_10)
        
        # Entropy × volatility: Complexity under volatile conditions
        cross["ENT_Return_x_Volatility"] = _log_normalize(returns_entropy * safe_vol5)
    
    # Compute additional entropy features from market data
    if market_data is not None and 'close' in market_data.columns:
        returns = market_data['close'].pct_change().reindex(base_features.index).fillna(0)
        
        # Approximate Entropy proxy: Rolling std of |returns| (simpler than true ApEn)
        returns_abs = returns.abs()
        approx_entropy_20 = returns_abs.rolling(20).std().fillna(0).values
        cross["ENT_ApproxEntropy_20"] = _log_normalize(approx_entropy_20)
        
        # Permutation Entropy proxy: Rank correlation volatility
        # (True permutation entropy is expensive, this is a fast approximation)
        rank_changes = returns.rolling(5).apply(
            lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1] if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        perm_entropy_proxy = rank_changes.rolling(20).std().fillna(0).values
        cross["ENT_PermEntropy_Proxy"] = _log_normalize(perm_entropy_proxy)
        
        # Price path complexity: Second derivative variance
        price_accel = returns.diff()
        path_complexity = price_accel.rolling(20).std().fillna(0).values
        cross["ENT_PathComplexity"] = _log_normalize(path_complexity)
    
    # =========================================================================
    # 8. LIQUIDITY PROXY FEATURES (Microstructure)
    # =========================================================================
    # These approximate liquidity conditions without order book data
    
    # Volume imbalance: Already computed, create interactions
    vol_imbalance = _safe_get(all_features, "volume_imbalance")
    if np.any(vol_imbalance != 0):
        # Volume imbalance × momentum: Directional pressure with volume confirmation
        cross["LIQ_Imbalance_x_Momentum"] = _log_normalize(vol_imbalance * momentum_10)
        
        # Volume imbalance × ATR: Liquidity pressure under volatile conditions
        cross["LIQ_Imbalance_x_ATR"] = _log_normalize(vol_imbalance * safe_atr)
    
    if market_data is not None and 'close' in market_data.columns:
        close = market_data['close'].reindex(base_features.index)
        returns = close.pct_change().fillna(0)
        
        # Amihud Illiquidity Proxy: |Return| / Volume
        # Higher = less liquid (price moves more per unit volume)
        if 'volume' in market_data.columns:
            volume = market_data['volume'].reindex(base_features.index).fillna(1)
            amihud_raw = returns.abs() / (volume + 1e-9)
            amihud_20 = amihud_raw.rolling(20).mean().fillna(0)
            amihud_baseline = amihud_20.rolling(96).median()
            amihud_ratio = (amihud_20 / (amihud_baseline + 1e-9)).fillna(1).values
            cross["LIQ_Amihud_Ratio"] = _log_normalize(amihud_ratio - 1.0)
            
            # Kyle's Lambda Proxy: Price impact per unit volume flow
            # Higher = larger price impact from volume
            price_change_6 = close.diff(6)
            signed_volume_6 = (volume * np.sign(returns)).rolling(6).sum()
            kyles_lambda = (price_change_6 / (signed_volume_6 + 1e-9)).fillna(0)
            kyles_lambda_smoothed = kyles_lambda.ewm(span=6).mean().fillna(0).values
            cross["LIQ_KyleLambda"] = _log_normalize(kyles_lambda_smoothed)
            
            # Roll's Spread Estimator Proxy: 2 * sqrt(-cov(r_t, r_{t-1}))
            # Negative autocovariance implies bid-ask bounce
            return_autocov = returns.rolling(20).apply(
                lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0,
                raw=False
            ).fillna(0)
            # Only use when autocov is negative (as expected for bid-ask bounce)
            rolls_spread = 2 * np.sqrt(np.maximum(-return_autocov, 0)).fillna(0).values
            cross["LIQ_RollsSpread"] = _log_normalize(rolls_spread)
            
            # Volume Pressure Ratio: Recent volume vs baseline
            vol_short = volume.rolling(5).mean()
            vol_long = volume.rolling(50).mean()
            vol_pressure = ((vol_short / (vol_long + 1e-9)) - 1).fillna(0).values
            cross["LIQ_VolPressure"] = _log_normalize(vol_pressure)
        
        # High-Low Spread Proxy: (High - Low) / Close
        # Proxy for intraday volatility/liquidity
        if 'high' in market_data.columns and 'low' in market_data.columns:
            high = market_data['high'].reindex(base_features.index)
            low = market_data['low'].reindex(base_features.index)
            hl_spread = ((high - low) / (close + 1e-9)).fillna(0)
            hl_spread_norm = (hl_spread / hl_spread.rolling(50).mean()).fillna(1).values
            cross["LIQ_HLSpread_Ratio"] = _log_normalize(hl_spread_norm - 1.0)
            
            # Parkinson Volatility: More efficient volatility estimator
            parkinson_vol = np.sqrt((np.log(high / low) ** 2).rolling(20).mean() / (4 * np.log(2))).fillna(0).values
            cross["LIQ_ParkinsonVol"] = _log_normalize(parkinson_vol)
    
    # =========================================================================
    # 9. PATH-ENTROPY-LIQUIDITY INTERACTIONS
    # =========================================================================
    # These combine the three concept families for rich signal extraction
    
    # Path efficiency × Entropy: Efficient paths in predictable markets
    if "PATH_Efficiency_10" in cross.columns and np.any(returns_entropy != 0):
        cross["PEL_PathEff_x_Entropy"] = _log_normalize(cross["PATH_Efficiency_10"].values * returns_entropy)
    
    # Liquidity × Entropy: Market microstructure under uncertainty
    if "LIQ_Amihud_Ratio" in cross.columns and np.any(returns_entropy != 0):
        cross["PEL_Amihud_x_Entropy"] = _log_normalize(cross["LIQ_Amihud_Ratio"].values * returns_entropy)
    
    # Path efficiency × Liquidity: Path quality under different liquidity conditions
    if "PATH_Efficiency_10" in cross.columns and "LIQ_VolPressure" in cross.columns:
        cross["PEL_PathEff_x_Liquidity"] = _log_normalize(
            cross["PATH_Efficiency_10"].values * cross["LIQ_VolPressure"].values
        )
    
    # Fill NaN and replace infinities
    cross = cross.fillna(0).replace([np.inf, -np.inf], 0)
    
    return cross


def get_cross_feature_inventory() -> Dict[str, List[str]]:
    """
    Return inventory of cross-features generated by generate_cross_features().
    
    Returns:
        Dict with categories and their feature lists
    """
    return {
        "price_volume_interactions": [
            "PV_Velocity_x_VolSlope",      # KF_Velocity × KF_LogVolume_Slope
            "PV_VWAP_x_VolZscore",         # KF_VWAP_Distance × KF_Volume_Zscore
            "PV_Return_x_VolRatio",        # KF_Close_LogRet × KF_Volume_Ratio
            "PV_SMADist_x_VolRatio",       # SMA_Distance × Volume_SMA_Ratio (horizon variant)
            "PV_ROC_x_VolP",               # ROC × KF_Volume_P
        ],
        "volatility_normalized": [
            "VN_Velocity_per_ATR",         # KF_Velocity / KF_ATR
            "VN_Mom5_per_Vol5",            # Momentum_5 / volatility_5
            "VN_Accel_per_ATRRatio",       # KF_Acceleration / KF_ATR_Ratio
            "VN_ROC_x_VolRegime",          # ROC × Volatility_Regime
            "VN_Slope_x_KalmanP",          # KF_Slope × KF_P
        ],
        "cross_horizon_divergence": [
            "XH_RSI_Divergence",           # RSI_Short - RSI_Long
            "XH_Momentum_Ratio",           # Momentum_Short / Momentum_Long
            "XH_ATR_Ratio",                # ATR_Short / ATR_Long
            "XH_SMADist_Divergence",       # SMA_Distance_Short - SMA_Distance_Long
            "XH_BBDist_Divergence",        # BB_Distance_Short - BB_Distance_Long
        ],
        "regime_conditional": [
            "RC_Mom_x_MetaVolRegime",      # Momentum × meta_volatility_regime
            "RC_VolSlope_x_MetaVolShock",  # KF_LogVolume_Slope × meta_volume_shock
            "RC_PriceSMA_x_Trendiness",    # Price_vs_SMA20 × meta_trendiness
            "RC_VWAPSlope_x_Trendiness",   # KF_VWAP_Slope × meta_trendiness
            "RC_ATRRatio_x_MetaVolRegime", # KF_ATR_Ratio × meta_volatility_regime
        ],
        "kalman_cross": [
            "KC_FilteredRawATR_Ratio",     # KF_ATR / ATR_14
            "KC_VolSlope_per_Vol5",        # KF_LogVolume_Slope / volatility_5
            "KC_Velocity_per_KalmanP",     # KF_Velocity / KF_P
            "KC_VWAPZscore_x_VolRatio",    # KF_VWAP_Zscore × KF_Volume_Ratio
            "KC_MomPerVol_x_VolSlope",     # Momentum_per_vol × KF_LogVolume_Slope
        ],
        "path_efficiency": [
            "PATH_ER_x_Momentum",          # Kaufman ER × Momentum
            "PATH_ER_x_Volatility",        # Kaufman ER × Volatility
            "PATH_ER_x_VolRatio",          # Kaufman ER × Volume Ratio
            "PATH_Efficiency_10",          # 10-bar path efficiency
            "PATH_Efficiency_30",          # 30-bar path efficiency
            "PATH_Efficiency_Divergence",  # Short vs long path efficiency
        ],
        "entropy_complexity": [
            "ENT_Return_x_Momentum",       # Return entropy × Momentum
            "ENT_Return_x_Volatility",     # Return entropy × Volatility
            "ENT_ApproxEntropy_20",        # Approximate entropy proxy
            "ENT_PermEntropy_Proxy",       # Permutation entropy proxy
            "ENT_PathComplexity",          # Price path complexity
        ],
        "liquidity_proxy": [
            "LIQ_Imbalance_x_Momentum",    # Volume imbalance × Momentum
            "LIQ_Imbalance_x_ATR",         # Volume imbalance × ATR
            "LIQ_Amihud_Ratio",            # Amihud illiquidity ratio
            "LIQ_KyleLambda",              # Kyle's lambda (price impact)
            "LIQ_RollsSpread",             # Roll's spread estimator
            "LIQ_VolPressure",             # Volume pressure ratio
            "LIQ_HLSpread_Ratio",          # High-Low spread ratio
            "LIQ_ParkinsonVol",            # Parkinson volatility
        ],
        "path_entropy_liquidity": [
            "PEL_PathEff_x_Entropy",       # Path efficiency × Entropy
            "PEL_Amihud_x_Entropy",        # Amihud × Entropy
            "PEL_PathEff_x_Liquidity",     # Path efficiency × Liquidity
        ],
    }


# ============================================================================
# HPO UTILITY FUNCTIONS (Trapezoidal Gate & Stability-Adjusted Utility)
# ============================================================================

def trapezoidal_gate(x: float, lower: float, sweet_spot: tuple, upper: float) -> float:
    """
    Returns a score [0, 1] based on trapezoidal membership.
    - Below lower: soft floor (0.01)
    - Between lower and sweet_spot[0]: Ramp up
    - Inside sweet_spot: 1.0
    - Between sweet_spot[1] and upper: Ramp down
    - Above upper: soft floor (leakage penalty)
    
    Args:
        x: The value to score
        lower: Lower bound (below this = rejection territory)
        sweet_spot: Tuple (min, max) for the ideal range
        upper: Upper bound (above this = leakage/overfit territory)
    
    Returns:
        Score in [0.2, 1.0]
    """
    s_min, s_max = sweet_spot
    
    if x < lower or x > upper:
        return 0.2
    elif s_min <= x <= s_max:
        return 1.0
    elif lower <= x < s_min:
        # Ramp up
        return (x - lower) / (s_min - lower)
    elif s_max < x <= upper:
        # Ramp down
        return (upper - x) / (upper - s_max)
    return 0.2


def calculate_hpo_utility(
    folds_sharpe: np.ndarray,
    auc: float,
    trades_per_day: float,
    lambda_vol: float = 1.2,
    w_auc: float = 1.0,
    w_den: float = 0.5,
    calibration_brier: Optional[float] = None,
    calibration_ece: Optional[float] = None,
    w_cal: float = 0.0,
) -> float:
    """
    Compute a stable utility for HPO combining Sharpe stability, AUC gate, and trade density.
    
    Args:
        folds_sharpe: Array of per-fold Sharpe ratios
        auc: Mean AUC across folds
        trades_per_day: Average trades per day
        lambda_vol: Penalty weight for Sharpe volatility across folds (default 1.2)
        w_auc: Weight exponent for AUC gate (default 1.0 = strict)
        w_den: Weight exponent for density modifier (default 0.5)
    
    Returns:
        Utility score. Returns -1.0 for rejection.
    """
    sharpe_arr = np.asarray(folds_sharpe, dtype=float).reshape(-1)
    sharpe_arr = sharpe_arr[np.isfinite(sharpe_arr)]
    if sharpe_arr.size < 1:
        return -1.0

    avg_sharpe = float(np.mean(sharpe_arr))
    vol_sharpe = float(np.std(sharpe_arr, ddof=1)) if sharpe_arr.size > 1 else 0.0
    if not (np.isfinite(avg_sharpe) and np.isfinite(vol_sharpe)):
        return -1.0

    base_score = avg_sharpe - (lambda_vol * vol_sharpe)
    try:
        base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
    except Exception:
        base_norm = 0.0
    if not np.isfinite(base_norm):
        base_norm = 0.0

    # 2. Auxiliary modifiers (0..1)
    
    # AUC Gate: Returns 0.01 to 1.0
    # Loosened band to reduce premature penalty and allow moderate gains
    phi_auc = trapezoidal_gate(auc, lower=0.52, sweet_spot=(0.56, 0.66), upper=0.72)

    # Density gate: penalize over-trading by ramping down above 5 trades/day
    phi_density = trapezoidal_gate(
        float(trades_per_day),
        lower=0.5,
        sweet_spot=(1.5, 5.0),
        upper=8.0,
    )

    # 3. Weighted Geometric Combination
    # If phi_auc is near 0 (leakage/random), the whole score collapses
    try:
        modifier = float((phi_auc ** w_auc) * (phi_density ** w_den))
    except Exception:
        modifier = 0.0
    if not np.isfinite(modifier):
        modifier = 0.0

    # Optional: calibration quality modifier (model-dependent; useful for Layer 3)
    if w_cal and w_cal > 0.0:
        cal = None
        if calibration_brier is not None and np.isfinite(calibration_brier):
            cal = float(calibration_brier)
        elif calibration_ece is not None and np.isfinite(calibration_ece):
            cal = float(calibration_ece)
        if cal is not None:
            phi_cal = float(np.clip(1.0 - (cal / 1.0), 0.0, 1.0))
            try:
                modifier *= float(phi_cal) ** float(w_cal)
            except Exception:
                modifier *= 0.0

    utility = float(np.clip(float(base_norm) * float(modifier), -1.0, 10.0))
    if not np.isfinite(utility):
        return -1.0
    return float(utility)


def _apply_hpo_quality_penalty(
    *,
    utility: float,
    returns: np.ndarray,
    labels: np.ndarray,
    exit_reasons: Optional[np.ndarray] = None,
    durations: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
    tx_cost: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    cfg = config or {}
    enabled = bool(cfg.get("hpo_quality_penalty_enabled", True))
    if not enabled:
        return utility, {"enabled": False}

    if not np.isfinite(utility) or utility <= 0.0:
        return utility, {"enabled": True, "skipped": True}

    ret = np.asarray(returns, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(ret) & np.isfinite(y)
    if int(mask.sum()) < 20:
        return utility, {"enabled": True, "skipped": True, "reason": "insufficient_samples"}

    ret = ret[mask]
    y = y[mask]

    eps_mult = float(cfg.get("hpo_aleatoric_cost_multiplier", 1.0))
    eps_add = float(cfg.get("hpo_aleatoric_cost_add", 0.0))
    eps = float(abs(tx_cost) * eps_mult + eps_add)
    if eps <= 0.0:
        eps = float(abs(tx_cost))

    aleatoric_rate = float((np.abs(ret) <= eps).mean())

    timeout_rate = float("nan")
    if exit_reasons is not None:
        ex = np.asarray(exit_reasons).astype(object)
        try:
            ex_mask = ex[mask]
            timeout_rate = float((pd.Series(ex_mask).astype(str).str.contains("timeout", case=False, na=False)).mean())
        except Exception:
            timeout_rate = float("nan")

    if not np.isfinite(timeout_rate):
        if durations is not None and horizon is not None:
            try:
                dur = np.asarray(durations, dtype=float)
                dur = dur[mask]
                timeout_rate = float((dur >= float(horizon)).mean())
            except Exception:
                timeout_rate = float("nan")

    if not np.isfinite(timeout_rate):
        timeout_rate = 0.0

    pos_rate = float((y >= 0.5).mean())
    target_center = float(cfg.get("hpo_target_pos_rate", 0.40))
    target_band = float(cfg.get("hpo_target_pos_band", 0.10))
    balance_dev = max(0.0, abs(pos_rate - target_center) - target_band)
    balance_penalty = float(balance_dev / max(1e-9, (0.5 - target_band)))

    w_aleatoric = float(cfg.get("hpo_penalty_w_aleatoric", 1.25))
    w_timeout = float(cfg.get("hpo_penalty_w_timeout", 1.0))
    w_balance = float(cfg.get("hpo_penalty_w_balance", 0.75))

    q_aleatoric = float(np.clip(1.0 - aleatoric_rate, 0.0, 1.0)) ** w_aleatoric
    q_timeout = float(np.clip(1.0 - timeout_rate, 0.0, 1.0)) ** w_timeout
    q_balance = float(np.exp(-w_balance * balance_penalty))

    quality_multiplier = float(np.clip(q_aleatoric * q_timeout * q_balance, 0.0, 1.0))
    utility_adj = float(np.clip(utility * quality_multiplier, -1.0, 10.0))

    details = {
        "enabled": True,
        "tx_cost": float(tx_cost),
        "near_cost_threshold": float(eps),
        "aleatoric_rate": aleatoric_rate,
        "timeout_rate": float(timeout_rate),
        "pos_rate": pos_rate,
        "balance_penalty": balance_penalty,
        "quality_multiplier": quality_multiplier,
        "weights": {
            "w_aleatoric": w_aleatoric,
            "w_timeout": w_timeout,
            "w_balance": w_balance,
        },
    }
    return utility_adj, details


def linear_size_from_prob(
    p: float,
    max_exposure: float = 1.0,
    min_prob: float = 0.5,
    scale: float = 1.0,
) -> float:
    """
    Compute long/short signed size in [-max_exposure, +max_exposure] from probability.
    
    Args:
        p: Predicted probability of a positive outcome
        min_prob: Neutral threshold (default 0.5)
        scale: Multiplier to control aggressiveness
        max_exposure: Maximum allowed exposure magnitude
    
    Returns:
        Position size in [-max_exposure, +max_exposure]
    """
    # confidence in [-1, 1]: maps 0.5->0, 1.0->1.0, 0.0->-1.0
    conf = (p - min_prob) / (1.0 - min_prob) if min_prob < 1.0 else 0.0
    conf = np.clip(conf, -1.0, 1.0)
    size = scale * conf
    # clamp to allowed exposure
    size = np.clip(size, -max_exposure, max_exposure)
    return float(size)


def directional_size_from_prob(
    p: float,
    *,
    direction: str,
    thr: float = 0.5,
    max_exposure: float = 1.0,
    scale: float = 1.0,
) -> float:
    d = str(direction or "").lower()
    p = float(p)
    thr = float(thr)
    if d == "long":
        # Long-only sizing
        conf = (p - thr) / max(1e-12, (1.0 - thr))
        size = max(0.0, conf)
        return float(np.clip(scale * size, 0.0, max_exposure))
    if d == "short":
        # Short-only sizing (negative exposure)
        conf = (thr - p) / max(1e-12, thr)
        size = max(0.0, conf)
        return float(-np.clip(scale * size, 0.0, max_exposure))
    # Fallback: signed sizing
    return linear_size_from_prob(p, max_exposure=max_exposure, min_prob=thr, scale=scale)


def calibrate_probabilities_isotonic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cv_folds: int = 3,
) -> np.ndarray:
    """
    Calibrate probabilities using isotonic regression with cross-validation.
    
    Args:
        y_true: True binary labels
        y_prob: Uncalibrated predicted probabilities
        cv_folds: Number of cross-validation folds
    
    Returns:
        Calibrated probabilities
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    
    calibrated = np.zeros_like(y_prob, dtype=float)
    kf = KFold(n_splits=cv_folds, shuffle=False)
    
    for train_idx, val_idx in kf.split(y_prob):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_prob[train_idx], y_true[train_idx])
        calibrated[val_idx] = iso.predict(y_prob[val_idx])
    
    return calibrated


def compute_fold_sharpe_ratios(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    n_folds: int = 5,
    use_calibration: bool = True,
) -> np.ndarray:
    """
    Compute per-fold Sharpe ratios for stability assessment.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        returns: Realized returns per sample
        n_folds: Number of folds
        use_calibration: Whether to calibrate probabilities before sizing
    
    Returns:
        Array of per-fold Sharpe ratios
    """
    from sklearn.model_selection import KFold
    
    # Pre-calibrate if requested
    if use_calibration:
        y_prob_cal = calibrate_probabilities_isotonic(y_true, y_prob, cv_folds=min(3, n_folds))
    else:
        y_prob_cal = y_prob
    
    fold_sharpes = []
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    for _, fold_idx in kf.split(returns):
        fold_probs = y_prob_cal[fold_idx]
        fold_returns = returns[fold_idx]
        
        # Compute sized returns: position_size * realized_return
        sized_returns = []
        for prob, ret in zip(fold_probs, fold_returns):
            size = linear_size_from_prob(prob, max_exposure=1.0, min_prob=0.5, scale=1.0)
            sized_returns.append(size * ret)
        
        sized_returns = np.array(sized_returns)
        
        if len(sized_returns) > 1 and np.std(sized_returns) > 1e-9:
            fold_sharpe = np.mean(sized_returns) / np.std(sized_returns)
        else:
            fold_sharpe = 0.0
        
        fold_sharpes.append(fold_sharpe)
    
    return np.array(fold_sharpes)


def compute_robust_expectancy(returns: np.ndarray, cap_quantile: float = 0.95) -> float:
    """Calculates the Edge in a way that works for BOTH Trend (Skewed) and Mean Reversion (Normal) strategies.

    Args:
        returns: Array of trade returns (floats, e.g., 0.01 for 1%).
        cap_quantile: The boundary for 'Lucky' trades (default 95th percentile).
    """
    if len(returns) == 0:
        return 0.0

    returns = np.asarray(returns, dtype=float)

    # 1. Determine Dynamic Caps (Winsorization Limits)
    # We define 'Extreme' based on the distribution of THIS strategy.
    # If the strategy naturally has huge wins, the cap will be higher.
    try:
        win_cap = np.percentile(returns, cap_quantile * 100)
        loss_cap = np.percentile(returns, (1 - cap_quantile) * 100)
    except Exception:
        return float(np.mean(returns))

    # 2. Apply Caps (Winsorize)
    # Instead of deleting outliers (Trimming), we clamp them.
    # A +50% trade becomes a +Win_Cap% trade.
    # This preserves the "Win" signal but limits the magnitude impact.
    capped_returns = np.clip(returns, loss_cap, win_cap)

    # 3. Calculate Expectancy
    # This is the "True" mathematical edge, cleaned of luck.
    robust_edge = float(np.mean(capped_returns))

    return robust_edge


def compute_robust_hpo_objective(
    stats: Dict[str, Any],
    df_results: pd.DataFrame,
    regime_col: str = 'regime',
    min_trades_per_day: float = 0.5,
    target_trades_per_day: float = 2.5,
    max_trades_per_day: float = 8.0,
) -> float:
    """Comprehensive HPO objective optimizing for ROBUSTNESS, EDGE, and TRADE QUALITY.

    This is the CANONICAL HPO objective function that combines:
    1. P&L Edge (calibration-adjusted position sizing via probability weighting)
    2. Minimum trades/day hard gate (reject if below threshold)
    3. Trade density bonus (reward 1.5-5 trades/day sweet spot)
    4. Sharpe ratio component (risk-adjusted returns)
    5. AUC component when y_true/y_prob available (learnability)
    6. Robust expectancy (winsorized to remove outlier luck)

    Formula:
        score = robust_edge * sqrt(N) * trade_density_factor * sharpe_factor * auc_bonus

    Args:
        stats: Dict with 'num_trades', 'trades_per_day', 'sharpe_ratio', 'mean_auc' (optional).
        df_results: DataFrame containing per-trade results:
                    ['y_true', 'y_prob', 'ret_bps', 'regime']
                    - y_true: Binary label (0/1)
                    - y_prob: Model predicted probability (used for position sizing)
                    - ret_bps: Realized return as float (e.g., 0.01 = 1%)
                    - regime: Optional regime label for stratified analysis
        regime_col: Column name for regime labels
        min_trades_per_day: Hard minimum (default 0.5) - reject if below
        target_trades_per_day: Optimal trades/day for bonus (default 2.5)
        max_trades_per_day: Upper bound for density penalty (default 8.0)

    Returns:
        Comprehensive objective score (higher is better)
    """
    if df_results.empty or 'ret_bps' not in df_results.columns:
        return 0.0

    # Extract core stats
    num_trades = stats.get('num_trades', len(df_results))
    trades_per_day = stats.get('trades_per_day', 0.0)
    sharpe_ratio = stats.get('sharpe_ratio', 0.0)
    mean_auc = stats.get('mean_auc', None)

    # --- HARD GATE: Minimum trades per day ---
    # Reject configurations that don't generate enough trading opportunities
    if trades_per_day < min_trades_per_day:
        return -1e6  # Strong penalty to ensure rejection

    # --- 1. Robust Edge Calculation (Winsorised Expectancy) ---
    returns_arr = df_results['ret_bps'].to_numpy(dtype=float)
    robust_edge_val = compute_robust_expectancy(returns_arr)

    # --- 2. Calibration-Adjusted P&L (Position Sizing by Probability) ---
    # If y_prob available, compute probability-weighted returns
    calib_adjusted_edge = robust_edge_val
    if 'y_prob' in df_results.columns:
        try:
            probs = df_results['y_prob'].to_numpy(dtype=float)
            # Position size = probability (Kelly-like sizing)
            # Calibration-adjusted return = prob * actual_return
            valid_mask = np.isfinite(probs) & np.isfinite(returns_arr)
            if valid_mask.sum() > 10:
                calib_returns = probs[valid_mask] * returns_arr[valid_mask]
                calib_adjusted_edge = compute_robust_expectancy(calib_returns)
        except Exception:
            pass

    # --- 3. Trade Density Factor ---
    # Reward trades in sweet spot (1.5-5/day), penalize extremes
    density_factor = 1.0
    if trades_per_day < 1.5:
        # Below sweet spot: linear ramp from min_trades to 1.5
        density_factor = 0.5 + 0.5 * (trades_per_day - min_trades_per_day) / max(1.5 - min_trades_per_day, 0.1)
    elif trades_per_day > 5.0:
        # Above sweet spot: gradual penalty
        excess = (trades_per_day - 5.0) / max(max_trades_per_day - 5.0, 1.0)
        density_factor = max(0.5, 1.0 - 0.5 * min(1.0, excess))
    else:
        # In sweet spot: bonus for being near target
        proximity = 1.0 - abs(trades_per_day - target_trades_per_day) / 3.5
        density_factor = 1.0 + 0.2 * max(0, proximity)

    # --- 4. Sharpe Ratio Factor ---
    # Reward positive Sharpe, penalize negative
    sharpe_factor = 1.0
    if sharpe_ratio is not None and np.isfinite(sharpe_ratio):
        if sharpe_ratio > 0:
            # Bonus for positive Sharpe (up to 50% boost at Sharpe=2)
            sharpe_factor = 1.0 + min(0.5, sharpe_ratio * 0.25)
        else:
            # Penalty for negative Sharpe
            sharpe_factor = max(0.3, 1.0 + sharpe_ratio * 0.3)

    # --- 5. AUC Bonus (Learnability) ---
    # If AUC available, reward AUC > 0.55, penalize AUC < 0.52
    auc_bonus = 1.0
    if mean_auc is not None and np.isfinite(mean_auc):
        if mean_auc >= 0.55:
            # Bonus: up to 30% for AUC approaching 0.70
            auc_bonus = 1.0 + min(0.3, (mean_auc - 0.55) * 2.0)
        elif mean_auc < 0.52:
            # Penalty for near-random AUC
            auc_bonus = max(0.5, 1.0 - (0.52 - mean_auc) * 5.0)

    # --- 6. Sample Size Scaling (t-stat like) ---
    # Reward more samples for statistical significance
    sample_factor = np.sqrt(max(1, num_trades))

    # --- FINAL COMPOSITE SCORE ---
    # Combine all components multiplicatively
    robust_score = (
        calib_adjusted_edge 
        * sample_factor 
        * density_factor 
        * sharpe_factor 
        * auc_bonus
    )

    return robust_score


def simulate_concurrent_trades(
    events_df: pd.DataFrame,
    max_concurrency: int = 1,
    transaction_cost: float = 0.003,
) -> pd.DataFrame:
    """Simulate trades with concurrency constraint (FIFO slot blocking).

    Logic: If Active_Trades >= Max_Concurrent_Positions, ignore new signals until a slot frees up.
    Bet Sizing: Size = 1.0 * CalibratedProb.

    Args:
        events_df: DataFrame with columns ['entry_time', 'exit_time', 'prob', 'realized_return', 'y_true', 'regime']
                   Must be sorted by entry_time.
        max_concurrency: Max simultaneous positions (Hardcoded to 1 for this task).
        transaction_cost: Transaction cost to ensure net returns are correct.

    Returns:
        DataFrame of executed trades.
    """
    if events_df.empty:
        return pd.DataFrame()

    # Ensure sorted by entry time
    events_sorted = events_df.sort_values('entry_time').reset_index(drop=True)

    executed_indices = []
    # Track end times of active trades. For concurrency=1, this is just a scalar or single-element list.
    # We use a min-heap or sorted list to track earliest exit if concurrency > 1.
    # Since concurrency is hardcoded to 1, a simple scalar variable suffices.
    active_trade_end_time = pd.Timestamp.min

    for idx, row in events_sorted.iterrows():
        entry_t = row['entry_time']

        # Check if slot is free
        # For concurrency=1: if current entry time < last accepted trade's exit time, BLOCK.
        if entry_t < active_trade_end_time:
            continue

        # Take trade
        executed_indices.append(idx)
        active_trade_end_time = row['exit_time']

    if not executed_indices:
        return pd.DataFrame()

    executed_df = events_sorted.iloc[executed_indices].copy()

    # Bet Sizing: Size = 1.0 * CalibratedProb
    # Apply sizing to return: Adjusted_Return = Size * Realized_Return
    # Note: Realized_Return in input is already net of costs per unit.
    # We assume 'realized_return' column holds the unit return.

    base_size = 1.0
    sizes = base_size * executed_df['prob']
    executed_df['position_size'] = sizes

    # Scale return by position size.
    # Logic: If we bet 0.6 units and get 10% return, PnL impact is 0.06 units (6%).
    executed_df['ret_bps'] = executed_df['realized_return'] * sizes

    return executed_df


class MetaLabelingHPOSampleWeightedStep(BaseStep):
    """Offline HPO step to optimize labeling parameters with sample weighting.

    This step does *not* run as part of standard training. It must be
    invoked explicitly (e.g. via the launcher). It reuses the existing
    labeling utilities but searches over a small parameter space:

    - Event definition / TPSL (profit threshold, stop ratio, horizon 2–12,
      min spacing)
    - Target transformation (symmetric probability clipping and
      symmetric quantile clipping of target magnitudes)

    The objective focuses on label quality and economic separation
    between positive/negative labels using realized returns.
    """

    def __init__(self, step_name: str = "meta_labeling_hpo_sample_weighted") -> None:
        super().__init__(step_name, use_versioned_artifacts=False)
        self.logger = logger

    def _create_selection_subsample(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a subsampled dataset for feature selection/discovery phases.

        Logic:
        1. Identify the selection window (default: last 6 months).
        2. Divide this window into N segments (default: 4).
        3. From each segment, take the last M days (default: 30 days).
        4. Concatenate these chunks.

        If the total dataset is smaller than the selection window, the entire
        dataset is used (with potential further subsampling if it's still huge).

        Args:
            features: Full feature DataFrame.
            targets: Full target DataFrame (aligned with features).
            config: Step configuration.

        Returns:
            Tuple of (subsampled_features, subsampled_targets).
        """
        # Default config: Selection window = 180 days (6 months)
        selection_window_days = int(config.get("selection_window_days", 180))
        # Default config: 4 segments
        subsample_count = int(config.get("subsample_count", 4))
        # Default config: 30 days per segment
        subsample_days = int(config.get("subsample_period_days", 30))
        # Minimum total rows required to trigger subsampling (e.g. < 6 months data -> no subsampling)
        min_rows_threshold = 20000

        if len(features) < min_rows_threshold:
            tprint_info(
                f"📊 Dataset size ({len(features)}) < threshold ({min_rows_threshold}); "
                "skipping subsampling for final selection"
            )
            return features, targets

        try:
            # Prefer time-based subsampling when index is a DatetimeIndex
            if not isinstance(features.index, pd.DatetimeIndex):
                # Fallback to row-based slicing if no datetime index
                tprint_warning(
                    "⚠️ Features index is not DatetimeIndex; "
                    "falling back to row-based subsampling for final selection"
                )
                total_rows = len(features)
                rows_per_day = 96  # approx 15m bars
                selection_window_rows = selection_window_days * rows_per_day
                start_idx = max(0, total_rows - selection_window_rows)

                # Slice to selection window
                features_window = features.iloc[start_idx:]
                targets_window = targets.iloc[start_idx:]

                # Split into segments
                segment_size = max(1, len(features_window) // subsample_count)
                subsample_size = subsample_days * rows_per_day

                indices_to_keep: List[int] = []
                for i in range(subsample_count):
                    seg_start = i * segment_size
                    seg_end = min(len(features_window), (i + 1) * segment_size)
                    if seg_start >= seg_end:
                        continue
                    # Take last chunk of segment
                    chunk_start = max(seg_start, seg_end - subsample_size)
                    indices_to_keep.extend(
                        range(start_idx + chunk_start, start_idx + seg_end)
                    )

                # Ensure unique and sorted
                indices_to_keep = sorted(set(indices_to_keep))
                if not indices_to_keep:
                    return features, targets

                sub_feats = features.iloc[indices_to_keep]
                sub_targs = targets.iloc[indices_to_keep]

                tprint_info(
                    "📊 Row-based subsampling (final FS): "
                    f"{len(features)} → {len(sub_feats)} rows "
                    f"({len(indices_to_keep)/len(features):.1%})"
                )
                return sub_feats, sub_targs

            # Time-based slicing
            end_ts = features.index.max()
            start_ts = end_ts - pd.Timedelta(days=selection_window_days)

            # Slice to selection window
            mask_window = (features.index >= start_ts) & (features.index <= end_ts)
            features_window = features.loc[mask_window]
            targets_window = targets.loc[mask_window]

            if features_window.empty:
                tprint_warning(
                    "⚠️ Selection window for final FS empty; using full dataset"
                )
                return features, targets

            window_duration = (
                features_window.index.max() - features_window.index.min()
            )
            if subsample_count <= 0 or window_duration <= pd.Timedelta(0):
                return features_window, targets_window

            segment_duration = window_duration / subsample_count
            subsample_duration = pd.Timedelta(days=subsample_days)

            chunks_features: List[pd.DataFrame] = []
            chunks_targets: List[pd.DataFrame] = []

            tprint_info(
                "📊 Subsampling for final FS from last "
                f"{selection_window_days} days (window: {start_ts} to {end_ts})"
            )

            for i in range(subsample_count):
                seg_start_ts = features_window.index.min() + i * segment_duration
                seg_end_ts = seg_start_ts + segment_duration

                # Define subsample range: [segment_end - subsample_days, segment_end]
                sub_end_ts = seg_end_ts
                sub_start_ts = max(seg_start_ts, sub_end_ts - subsample_duration)

                mask_sub = (features.index >= sub_start_ts) & (
                    features.index < sub_end_ts
                )
                chunk_f = features.loc[mask_sub]
                chunk_t = targets.loc[mask_sub]

                if not chunk_f.empty:
                    chunks_features.append(chunk_f)
                    chunks_targets.append(chunk_t)

            if not chunks_features:
                tprint_warning(
                    "⚠️ No chunks generated for final FS subsampling; using full dataset"
                )
                return features, targets

            sub_feats = pd.concat(chunks_features).sort_index()
            sub_targs = pd.concat(chunks_targets).sort_index()

            tprint_info(
                "📊 Time-based subsampling (final FS): "
                f"{len(features)} → {len(sub_feats)} rows "
                f"({len(sub_feats)/len(features):.1%})"
            )
            tprint_info(
                f"   Using {subsample_count} chunks of ~{subsample_days} days "
                f"from the last {selection_window_days} days"
            )

            return sub_feats, sub_targs

        except Exception as e:
            tprint_warning(
                f"⚠️ Error in final FS subsampling: {e}; using full dataset"
            )
            return features, targets


    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hierarchical HPO over labeling parameters.

        Config keys (non-exhaustive):
        - symbol, exchange, timeframe: market context
        - enable_labeling_hpo: if False, step exits early
        - execution_mode: 'full'/'light'/'blank' for data loading scope
        """
        # Shared run timestamp for all stage artifacts (placed before early exits)
        run_ts: str = config.setdefault(
            "run_timestamp",
            datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        )

        if not config.get("enable_labeling_hpo", True):
            tprint("ℹ️ Labeling HPO disabled via config.enable_labeling_hpo", "INFO")
            return {"success": True, "metrics": {}, "artifacts": {}, "skipped": True}

        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")

        force_hpo = bool(config.get("force_hpo", False))
        if force_hpo:
            # Force full recomputation: disable caches for features/HPO
            config["force_recompute_features"] = True
            config["force_recompute"] = True
            config["use_feature_selection_cache"] = False

        start_at_raw = config.get("labeling_hpo_start_at")
        start_at_norm = str(start_at_raw).strip().lower() if start_at_raw is not None else "layer0"
        start_rank = 0
        if start_at_norm in {"0", "stage0", "layer0", "kalman"}:
            start_rank = 0
        elif start_at_norm in {"1", "layer1", "weighting"}:
            start_rank = 1
        elif start_at_norm in {"2", "layer2", "trading"}:
            start_rank = 2
        elif start_at_norm in {"feature_selection", "fs", "feature-selection"}:
            start_rank = 3
        elif start_at_norm in {"3", "layer3", "model"}:
            start_rank = 4
        else:
            tprint_warning(
                f"⚠️ Unknown labeling_hpo_start_at='{start_at_raw}'. Defaulting to 'layer0'."
            )
            start_rank = 0

        # Only allow full cached early-exit when starting from layer0.
        if (not force_hpo) and (start_rank == 0):
            try:
                outcomes_dir = Path("outcomes")
                if outcomes_dir.exists():
                    pattern = f"hpo_multi_stage_best_params_{symbol}_*.json"
                    candidates = sorted(outcomes_dir.glob(pattern))
                    if candidates:
                        latest = candidates[-1]
                        with open(latest, "r") as f:
                            cached_params = json.load(f)
                        tprint_info(
                            f"♻️ Reusing cached multi-stage HPO best params from {latest} "
                            "(set config.force_hpo=True or use --force-hpo to recompute)."
                        )
                        metrics = {
                            "best_params": cached_params,
                            "hpo_cached": True,
                            "best_params_path": str(latest),
                        }
                        return {
                            "success": True,
                            "metrics": metrics,
                            "artifacts": {"best_params_json": str(latest)},
                            "skipped": True,
                        }
            except Exception as e:
                tprint_warning(
                    f"⚠️ Failed to load cached multi-stage HPO params; running full HPO instead: {e}"
                )

        hpo_feature_set: Optional[List[str]] = None
        try:
            persistence = FeatureSetPersistence()
            # Prefer the 70-feature production meta-labeling set when available.
            if not force_hpo:
                feature_names_70 = persistence.get_feature_set(
                    symbol=symbol,
                    exchange=exchange,
                    family="meta_labeling",
                    size=70,
                )
                if feature_names_70:
                    hpo_feature_set = feature_names_70
                    tprint_info(
                        f"📊 HPO: using precomputed LGBM meta-labeling feature set (70 features) for {symbol} on {exchange}"
                    )
        except Exception as e:
            tprint_warning(f"[HPO_FEATURES] Could not load precomputed feature set: {e}")

        tprint_info(
            f"🚀 Starting Meta-Labeling HPO experiment for {symbol}/{exchange} [{timeframe}] ({direction})"
        )

        use_smoothed_brier_objective_lgbm: bool = bool(
            config.get("use_smoothed_brier_objective_lgbm", True)
        )

        # ------------------------------------------------------------------
        # 1) Load market data once and generate primary signals
        # ------------------------------------------------------------------
        # Create safe copy of config to avoid mutating caller's dict
        config = safe_config_copy(config)
        
        pipeline_state: Dict[str, Any] = {}
        if config.get("execution_mode") == "full":
            # Force sufficient history for HPO in full mode
            # This overrides potentially small defaults in BaseStep/ExecutionModeConfig
            current_lookback = config.get("lookback_days", 0)
            if not current_lookback or int(current_lookback) < 365:
                 tprint(f"🔧 HPO {config.get('execution_mode')} Mode: Forcing 1095 days lookback to ensure sufficient data", "INFO")
                 config["lookback_days"] = 1095
                 # Also disable Stage 1 subsampling cap (default 180) to allow full dataset
                 config["stage1_default_cap_days"] = 0

        market_data, source = self.load_market_data_or_fail(
            config,
            pipeline_state,
            allow_config_override=True,
            skip_artifacts=True,
        )

        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            msg = "❌ No market data available for labeling HPO"
            tprint(msg, "ERROR")
            return {"success": False, "error": msg, "metrics": {}, "artifacts": {}}

        # Honour centralized lookback_days from the launcher for full/blank modes
        # by trimming to the last N days on the DateTimeIndex. Light-mode behavior
        # (shorter windows) is handled via BaseStep._apply_light_mode_filter.
        lookback_days = config.get("lookback_days")
        if lookback_days and int(lookback_days) > 0 and isinstance(market_data.index, pd.DatetimeIndex):
            end_ts = market_data.index.max()
            start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
            market_data = market_data[market_data.index >= start_ts].copy()
            tprint_info(f"   ✂️ Trimmed market_data to last {lookback_days} days (N={len(market_data)})")

        try:
            primary_signals = generate_primary_signals(market_data.copy())
        except Exception as e:
            msg = f"❌ Primary signal generation failed: {e}"
            tprint(msg, "ERROR")
            return {"success": False, "error": msg, "metrics": {}, "artifacts": {}}

        # Attach regimes (HMM + Volatility) to market data
        try:
            market_data_reg = attach_rolling_hmm_regimes_to_market_data(
                step=self,
                market_data=market_data,
                config=config,
            )
            if market_data_reg is not None and not market_data_reg.empty:
                market_data = market_data_reg
        except Exception as e:
            tprint_warning(f"⚠️ Regime attachment failed: {e}")

        # Compute log-returns and volatility for later use
        log_ret = np.log(market_data["close"]).diff()
        volatility_1d = log_ret.rolling(96).std()

        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        return self.run_hpo_loop(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            market_data=market_data,
            primary_signals=primary_signals,
            market_data_full_for_diagnostics=market_data,
            config=config,
            outcomes_dir=outcomes_dir,
        )

    def run_hpo_loop(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        direction: str,
        market_data: pd.DataFrame,
        primary_signals: pd.DataFrame,
        market_data_full_for_diagnostics: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        outcomes_dir: Path = Path("outcomes"),
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the Multi-Stage HPO:
        Stage 0: Signal/Feature HPO (Kalman Tuning)
        Layer 1: Weighting Params Optimization
        Layer 2: Trading Params Optimization (Dynamic events, dynamic weighting)
        Layer 3: Model Hyperparams Optimization
        """
        tprint_info(f"Starting Multi-Stage HPO for {symbol} {timeframe} {direction}")
        # Create safe copy of config to avoid mutations
        config = safe_config_copy(config) if config else {}

        start_at_raw = config.get("labeling_hpo_start_at")
        start_at_norm = str(start_at_raw).strip().lower() if start_at_raw is not None else "layer0"

        stage_rank = {
            "stage0": 0,
            "layer1": 1,
            "layer2": 2,
            "feature_selection": 3,
            "layer3": 4,
        }

        if start_at_norm in {"0", "stage0", "layer0", "kalman"}:
            start_rank = 0
            start_at_canonical = "layer0"
        elif start_at_norm in {"1", "layer1", "weighting"}:
            start_rank = 1
            start_at_canonical = "layer1"
        elif start_at_norm in {"2", "layer2", "trading"}:
            start_rank = 2
            start_at_canonical = "layer2"
        elif start_at_norm in {"feature_selection", "fs", "feature-selection"}:
            start_rank = 3
            start_at_canonical = "feature_selection"
        elif start_at_norm in {"3", "layer3", "model"}:
            start_rank = 4
            start_at_canonical = "layer3"
        else:
            start_rank = 0
            start_at_canonical = "layer0"
            tprint_warning(
                f"⚠️ Unknown labeling_hpo_start_at='{start_at_raw}'. Defaulting to 'layer0'."
            )

        tprint_info(
            f"🔁 HPO start control: start_at={start_at_canonical} (rank={start_rank})"
        )

        def _load_latest_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
            if path is None:
                return None
            try:
                if not path.exists():
                    return None
                with open(path, "r") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else None
            except Exception:
                return None

        def _load_latest_multi_stage_best_params() -> Tuple[Dict[str, Any], Optional[Path]]:
            try:
                p = _find_latest_path(Path("outcomes"), f"hpo_multi_stage_best_params_{symbol}_*.json")
                data = _load_latest_json(p)
                if isinstance(data, dict):
                    return data, p
            except Exception:
                pass
            return {}, None

        def _load_stage_best_params(stage: str) -> Tuple[Dict[str, Any], Optional[Path]]:
            """Load latest best params for a stage from outcomes artifacts."""
            outcomes = Path("outcomes")

            multi_best, multi_path = _load_latest_multi_stage_best_params()

            if stage == "stage0":
                p = _find_latest_path(
                    outcomes,
                    f"hpo_stage_report_stage0_kalman_{symbol}_{exchange}_{timeframe}_{direction}_*.json",
                )
                data = _load_latest_json(p)
                if isinstance(data, dict) and isinstance(data.get("best_params"), dict):
                    return dict(data.get("best_params") or {}), p
                # Fallback to multi-stage
                return {
                    "kalman_Q": multi_best.get("kalman_Q"),
                    "kalman_R": multi_best.get("kalman_R"),
                }, multi_path

            if stage == "layer1":
                p = _find_latest_path(outcomes, f"hpo_layer1_best_params_{symbol}_{timeframe}_*.json")
                data = _load_latest_json(p)
                if isinstance(data, dict) and isinstance(data.get("best_params"), dict):
                    return dict(data.get("best_params") or {}), p
                return {
                    k: v
                    for k, v in multi_best.items()
                    if k
                    in {
                        "mag_compression",
                        "learn_slope",
                        "learn_center",
                        "uniq_intensity",
                        "exp_mag",
                        "exp_learn",
                        "exp_uniq",
                        "exp_cross",
                        "downside_multiplier",
                        "time_decay_halflife",
                        "mag_clip_pct",
                    }
                }, multi_path

            if stage == "layer2":
                p = _find_latest_path(outcomes, f"hpo_layer2_best_params_{symbol}_{timeframe}_*.json")
                data = _load_latest_json(p)
                if isinstance(data, dict) and isinstance(data.get("best_params"), dict):
                    return dict(data.get("best_params") or {}), p
                # Fallback to multi-stage
                return {
                    k: v
                    for k, v in multi_best.items()
                    if k
                    in {
                        "sl_atr_mult",
                        "risk_reward_ratio",
                        "trail_distance_atr_mult",
                        "w_scalp",
                        "w_swing",
                        "w_trend",
                        "consensus_threshold",
                        "consensus_quantile",
                        "horizon_bars",
                        "min_event_spacing",
                    }
                }, multi_path

            return {}, None
        
        # Initialize computation cache for expensive operations
        computation_cache = ObjectiveComputationCache()
        
        # Extract configuration with defaults
        stage1_enable_subsample = config.get("stage1_enable_subsample", False)
        stage1_subsample_window_days = config.get("stage1_subsample_window_days", 180)

        # ------------------------------------------------------------------
        # 0. SETUP & PRE-CALCULATION
        # ------------------------------------------------------------------
        close_series = market_data["close"]
        close_prices = close_series  # Alias for compatibility
        returns_series = close_series.pct_change().fillna(0.0)

        # Compute log-returns and volatility for later use
        log_ret = np.log(close_series).diff()
        volatility_1d = log_ret.rolling(96).std()

        # Calculate days span from market data
        try:
            days_span = max(1, (market_data.index.max() - market_data.index.min()).days)
        except Exception:
            days_span = max(1, len(market_data) // 96)  # Assume ~96 bars per day as fallback

        # Heavy Lifting: Pre-calculate bar-level features for weighting
        tprint_info("🏗️ Pre-calculating bar-level features...")
        full_consistency = compute_horizon_consistency(close_series, horizon=12)
        full_volatility = returns_series.rolling(20).std().fillna(0.0)

        # ATR series for TBM
        high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
        low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
        tr1 = high_prices - low_prices
        tr2 = (high_prices - close_prices.shift(1)).abs()
        tr3 = (low_prices - close_prices.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=14, min_periods=1).mean()
        atr_frac = (
            atr_series / (close_series.abs() + 1e-8)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        try:
            layer2_tp_target = float(config.get("layer2_tp_target", 0.015))
        except Exception:
            layer2_tp_target = 0.015
        try:
            layer2_sl_target = float(config.get("layer2_sl_target", 0.006))
        except Exception:
            layer2_sl_target = 0.006
        try:
            atr_ref = float(np.nanmedian(pd.to_numeric(atr_frac, errors="coerce").values))
        except Exception:
            atr_ref = 0.0
        if (not np.isfinite(atr_ref)) or atr_ref <= 0.0:
            atr_ref = float(np.nanmean(pd.to_numeric(atr_frac, errors="coerce").values))
        if (not np.isfinite(atr_ref)) or atr_ref <= 0.0:
            atr_ref = 1e-3
        atr_scale = (atr_frac / (float(atr_ref) + 1e-12)).astype(float)
        atr_scale = atr_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        try:
            atr_scale_min = float(config.get("layer2_atr_scale_min", 0.25))
            atr_scale_max = float(config.get("layer2_atr_scale_max", 4.0))
            if np.isfinite(atr_scale_min) and np.isfinite(atr_scale_max) and atr_scale_max > atr_scale_min > 0:
                atr_scale = atr_scale.clip(lower=float(atr_scale_min), upper=float(atr_scale_max))
        except Exception:
            pass
        min_profit_floor = float(DEFAULT_TRANSACTION_COST) * 1.05
        fixed_layer2_profit_thr = (float(layer2_tp_target) * atr_scale).astype(float).clip(lower=min_profit_floor)
        fixed_layer2_stop_thr = (float(layer2_sl_target) * atr_scale).astype(float).clip(lower=min_profit_floor / 2.0)

        stage1_market_data = market_data
        stage1_primary_signals = primary_signals
        stage1_volatility_1d = volatility_1d
        stage1_atr_series = atr_series
        stage1_days_span = days_span

        if stage1_enable_subsample:
            try:
                start_ts = market_data.index.min()
                end_ts = start_ts + pd.Timedelta(days=stage1_subsample_window_days)
                mask = (market_data.index >= start_ts) & (market_data.index <= end_ts)
                if int(mask.sum()) > 0:
                    stage1_market_data = market_data.loc[mask].copy()
                    stage1_primary_signals = primary_signals.loc[mask].copy()
                    stage1_volatility_1d = volatility_1d.loc[mask].copy()
                    stage1_atr_series = atr_series.loc[mask].copy()
                    try:
                        stage1_days_span = max(
                            1,
                            (stage1_market_data.index.max() - stage1_market_data.index.min()).days,
                        )
                    except Exception:
                        stage1_days_span = days_span
                else:
                    stage1_enable_subsample = False
            except Exception:
                stage1_enable_subsample = False
                stage1_market_data = market_data
                stage1_primary_signals = primary_signals
                stage1_volatility_1d = volatility_1d
                stage1_atr_series = atr_series
                stage1_days_span = days_span

        # Build simple arrays for the optimizer API (they are not used in
        # the objective itself but provide shapes/logging)
        # Build proper feature matrix using production logic
        # This replaces the dummy placeholder aligned with user request (70+ features vs 1)
        tprint_info("🔧 Generating production meta-features for HPO...")
        try:
            X_features = create_meta_features(
                df=market_data,
                signals=primary_signals,
                volume_available="volume" in market_data.columns,
                include_raw_signals=False,
                use_kalman=True
            )
            # Ensure index alignment
            common_idx = X_features.index.intersection(market_data.index)
            X_dummy = X_features.loc[common_idx]
            # Ensure float32 for memory efficiency, filtering non-numeric cols first
            numeric_cols = X_dummy.select_dtypes(include=[np.number]).columns
            X_dummy = X_dummy[numeric_cols].astype("float32")
            base_feat_count = X_dummy.shape[1]

            # Expand with multi-horizon smoothed variants to mimic production breadth
            try:
                horizon_cfg = {"Short": 5, "Medium": 20, "Long": 60}
                X_dummy = generate_multi_horizon_features(
                    base_features=X_dummy,
                    horizons=horizon_cfg,
                    include_base=True,
                )
                tprint_info(
                    f"   Multi-horizon expansion: {base_feat_count} → {X_dummy.shape[1]} cols "
                    f"(horizons={list(horizon_cfg.keys())})"
                )
            except Exception as mh_exc:
                tprint_warning(f"   ⚠️ Multi-horizon expansion failed: {mh_exc}")

            # Add cross-feature interactions (base × Kalman / volatility-normalised)
            try:
                kalman_cols = [c for c in X_dummy.columns if str(c).startswith("KF_")]
                base_cols = [c for c in X_dummy.columns if not str(c).startswith("KF_")]
                kalman_df = X_dummy[kalman_cols] if kalman_cols else pd.DataFrame(index=X_dummy.index)
                base_df = X_dummy[base_cols] if base_cols else pd.DataFrame(index=X_dummy.index)
                cross_df = generate_cross_features(
                    base_features=base_df,
                    kalman_features=kalman_df,
                    market_data=market_data if market_data is not None else pd.DataFrame(index=X_dummy.index),
                )
                if cross_df is not None and not cross_df.empty:
                    X_dummy = pd.concat([X_dummy, cross_df], axis=1)
                    tprint_info(f"   Added {len(cross_df.columns)} cross features")
            except Exception as cross_exc:
                tprint_warning(f"   ⚠️ Cross-feature generation failed: {cross_exc}")

            # Final dtype cast and log
            X_dummy = X_dummy.astype("float32")
            tprint_success(f"✅ Generated {X_dummy.shape[1]} features (rows={len(X_dummy)})")
        except Exception as feat_e:
            tprint_warning(f"⚠️ Feature generation failed: {feat_e}. Using fallback.")
            X_dummy = market_data[["close"]].dropna()

        y_dummy = np.zeros(len(X_dummy), dtype="float32")

        # ------------------------------------------------------------------
        # 2) Define parameter groups for hierarchical HPO
        # ------------------------------------------------------------------
        param_groups = [
            # Group 1: Signal Structure (3 params)
            # RELAXED: Lower CUSUM threshold (0.006-0.025) and min_event_spacing (0)
            # to generate more events and avoid sample starvation
            create_param_group(
                name="signal_structure",
                params={
                    "cusum_threshold": {
                        "type": "float",
                        "low": 0.010,
                        "high": 0.06,
                    },
                    "target_signal_density": {
                        "type": "float",
                        "low": 10.0,
                        "high": 40.0,
                    },
                    "min_event_spacing": {
                        "type": "int",
                        "low": 0,
                        "high": 10,
                    },
                },
                priority=1,
                description="Signal generation and event spacing",
            ),
            # Group 2: Event Geometry (4 params) - Depends on Signal Structure
            create_param_group(
                name="event_geometry",
                params={
                    "horizon_bars": {
                        "type": "int",
                        "low": 16,
                        "high": 28,
                        "step": 2,
                    },
                    # NOTE: profit_thr_base and stop_to_profit_ratio are now redundant
                    # The Kalman multi-triple-barrier system handles multiple TP/SL configurations internally
                    # "profit_thr_base": {
                    #     "type": "float",
                    #     "low": 0.010,
                    #     "high": 0.025,
                    # },
                    # "stop_to_profit_ratio": {
                    #     "type": "float",
                    #     "low": 0.3,
                    #     "high": 0.67,
                    # },
                    "trail_distance": {
                        "type": "float",
                        "low": 0.6,
                        "high": 3.0,
                    },
                    "consensus_threshold": {
                        "type": "float",
                        "low": 0.4,
                        "high": 0.8,
                        "step": 0.05,
                    },
                },
                priority=2,
                depends_on=["signal_structure"],
                description="Triple-barrier shape and trailing stop logic",
            ),
            # Group 3: Volatility Adaptation (5 params) - Depends on Event Geometry
            create_param_group(
                name="volatility_adaptation",
                params={
                    "vol_baseline_window": {
                        "type": "int",
                        "low": 48,
                        "high": 192,
                    },
                    "profit_mult_min": {
                        "type": "float",
                        "low": 0.7,
                        "high": 1.0,
                    },
                    "profit_mult_max": {
                        "type": "float",
                        "low": 1.0,
                        "high": 2.0,
                    },
                    "stop_mult_min": {
                        "type": "float",
                        "low": 0.5,
                        "high": 1.0,
                    },
                    "stop_mult_max": {
                        "type": "float",
                        "low": 1.0,
                        "high": 1.5,
                    },
                },
                priority=3,
                depends_on=["event_geometry"],
                description="Volatility adaptation baseline and multipliers",
            ),
            # Group 4: Label Definition (3 params) - Depends on Volatility Adaptation
            # WIDENED: label quantile ranges to capture more samples (20-80% instead of 25-45/55-80)
            create_param_group(
                name="label_definition",
                params={
                    "label_low_q": {
                        "type": "float",
                        "low": 0.15,  # WIDENED from 0.25 to capture more negative samples
                        "high": 0.40,  # WIDENED from 0.45
                    },
                    "label_high_q": {
                        "type": "float",
                        "low": 0.60,  # WIDENED from 0.55
                        "high": 0.85,  # WIDENED from 0.80 to capture more positive samples
                    },
                    "econ_min_return_multiple": {
                        "type": "float",
                        "low": 1.0,
                        "high": 2.0,
                    },
                },
                priority=4,
                depends_on=["volatility_adaptation"],
                description="Quantile-based label definition",
            ),
            # Group 5: Target Engineering (5 params) - Depends on Label Definition
            create_param_group(
                name="target_engineering",
                params={
                    "iso_min_prob": {
                        "type": "float",
                        "low": 0.05,
                        "high": 0.15,
                    },
                    "target_clip_high_q": {
                        "type": "float",
                        "low": 0.90,
                        "high": 0.98,
                    },
                    "signal_strength_scale_max": {
                        "type": "float",
                        "low": 1.2,
                        "high": 2.0,
                    },
                    "r_multiple_pos_threshold": {
                        "type": "float",
                        "low": 0.3,
                        "high": 1.0,
                    },
                    "transaction_cost_mult": {
                        "type": "float",
                        "low": 1.0,
                        "high": 1.2,
                    },
                    "scale_pos_weight": {
                        "type": "float",
                        "low": 1.0,
                        "high": 10.0,
                    },
                },
                priority=5,
                depends_on=["label_definition"],
                description="Target transformation and trade filters",
            ),
            # Group 6: Smoothing (2 params) - Independent/Late stage
            create_param_group(
                name="kalman_smoothing",
                params={
                    "kalman_Q": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-3,
                        "log": True,
                    },
                    "kalman_R": {
                        "type": "float",
                        "low": 1e-3,
                        "high": 0.1,
                        "log": True,
                    },
                },
                priority=6,
                depends_on=["event_geometry"],
                description="Kalman smoothing noise parameters",
            ),
            # Group 7: Model Hyperparameters (Smart Walker)
            create_param_group(
                name="model_hyperparameters",
                params={
                    "num_leaves": {
                        "type": "int",
                        "low": 16,
                        "high": 256,
                        "walker_type": "geometric",
                        "step": 1.5,
                        "anchor": 31,
                    },
                    "min_data_in_leaf": {
                        "type": "int",
                        "low": 10,
                        "high": 200,
                        "walker_type": "geometric",
                        "step": 1.5,
                        "anchor": 20,
                    },
                    "max_depth": {
                        "type": "int",
                        "low": 3,
                        "high": 12,
                        "walker_type": "arithmetic",
                        "step": 1,
                        "anchor": 6,
                    },
                    "min_gain_to_split": {
                        "type": "float",
                        "low": 0.0,
                        "high": 1.0,
                        "walker_type": "arithmetic",
                        # Start from moderate split gain and walk in small steps
                        "step": 0.05,
                        "anchor": 0.1,
                    },
                    "lambda_l1": {
                        "type": "float",
                        "low": 0.0,
                        "high": 10.0,
                        "walker_type": "log_step",
                        # Explore moderate L1 regularization: 0.1 → 0.3 → 0.9 → 2.7 → 8.1
                        "step": 3.0,
                        "anchor": 0.1,
                    },
                    "lambda_l2": {
                        "type": "float",
                        "low": 0.0,
                        "high": 10.0,
                        "walker_type": "log_step",
                        # Same schedule as lambda_l1
                        "step": 3.0,
                        "anchor": 0.1,
                    },
                    "path_smooth": {
                        "type": "float",
                        "low": 0.0,
                        "high": 10.0,
                        "walker_type": "log_step",
                        # Smoothing strength: prefer non-zero baseline and moderate growth
                        "step": 3.0,
                        "anchor": 0.1,
                    },
                    "learning_rate": {
                        "type": "float",
                        "low": 0.01,
                        "high": 0.2,
                        "walker_type": "log_step",
                        "step": 2.0,
                        "anchor": 0.05,
                    },
                    "subsample": {
                        "type": "float",
                        "low": 0.6,
                        "high": 1.0,
                        "walker_type": "arithmetic",
                        "step": 0.1,
                        "anchor": 0.8,
                    },
                    "colsample_bytree": {
                        "type": "float",
                        "low": 0.6,
                        "high": 1.0,
                        "walker_type": "arithmetic",
                        "step": 0.1,
                        "anchor": 0.8,
                    },
                    "n_estimators": {
                        "type": "int",
                        "low": 150,
                        "high": 800,
                        "walker_type": "arithmetic",
                        "step": 50,
                        "anchor": 300,
                    },
                },
                priority=7,
                # Independent of label defs, but affects modeling.
                # Usually we want these after signal/targets are fixed.
                depends_on=["target_engineering"], 
                description="LGBM Model Hyperparameters (Smart Walker)",
            ),
        ]

        warm_start_best_params: Dict[str, Any] = {}
        warm_start_candidates_df: Optional[pd.DataFrame] = None
        outcomes_dir = Path("outcomes")
        try:
            json_pattern = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{direction}_*.json"
            json_paths = sorted(outcomes_dir.glob(json_pattern))
            if json_paths:
                latest_json = json_paths[-1]
                with open(latest_json, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        best_params_data = data.get("best_params", {})
                        if isinstance(best_params_data, dict):
                            warm_start_best_params = best_params_data
        except Exception:
            warm_start_best_params = {}
        try:
            csv_pattern = f"meta_labeling_hpo_candidate_pool_{symbol}_{timeframe}_{direction}_*.csv"
            csv_paths = sorted(outcomes_dir.glob(csv_pattern))
            if csv_paths:
                latest_csv = csv_paths[-1]
                warm_start_candidates_df = pd.read_csv(latest_csv)
        except Exception:
            warm_start_candidates_df = None

        calibrated_horizon: Optional[int] = None
        if stage1_enable_subsample:
            def _evaluate_horizon_candidate(h: int) -> Dict[str, float]:
                (
                    realized_returns_h,
                    binary_labels_h,
                    exit_reasons_h,
                    event_durations_h,
                    mfe_h,
                    mae_h,
                    _binary_labels_long_h,  # Not used in HPO scoring
                    _binary_labels_short_h,  # Not used in HPO scoring
                ) = compute_realized_returns(
                    stage1_market_data,
                    stage1_primary_signals,
                    profit_threshold=float(warm_start_best_params.get("profit_thr_base", 0.012)),
                    stop_threshold=float(warm_start_best_params.get("profit_thr_base", 0.012)) * float(warm_start_best_params.get("stop_to_profit_ratio", 0.5)),
                    horizon=int(h),
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=int(warm_start_best_params.get("min_event_spacing", 0)),
                    atr_series=stage1_atr_series,
                    trail_distance_atr_mult=float(warm_start_best_params.get("trail_distance", 0.0)),
                )
                labeled_mask_h = ~binary_labels_h.isna()
                n_events_h = int(labeled_mask_h.sum())
                if n_events_h <= 0 or stage1_days_span <= 0:
                    return {
                        "trades_per_day": 0.0,
                        "risk_reward": 0.0,
                        "profit_potential": 0.0,
                    }
                trades_per_day_h = n_events_h / float(stage1_days_span)
                returns_labeled_h = realized_returns_h[labeled_mask_h].dropna()
                if len(returns_labeled_h) == 0:
                    return {
                        "trades_per_day": trades_per_day_h,
                        "risk_reward": 0.0,
                        "profit_potential": 0.0,
                    }
                labels_labeled_h = binary_labels_h[labeled_mask_h]
                r_pos_h = returns_labeled_h[labels_labeled_h == 1]
                r_neg_h = returns_labeled_h[labels_labeled_h == 0]
                mean_pos_h = float(r_pos_h.mean()) if len(r_pos_h) > 0 else 0.0
                mean_neg_h = float(r_neg_h.mean()) if len(r_neg_h) > 0 else 0.0
                mean_loss_h = abs(mean_neg_h) if mean_neg_h < 0 else (abs(float(r_neg_h.mean())) if len(r_neg_h) > 0 else 0.0)
                mean_win_h = mean_pos_h
                risk_reward_h = mean_win_h / (mean_loss_h + 1e-8) if mean_loss_h > 0 else 0.0
                mean_return_h = float(returns_labeled_h.mean()) if len(returns_labeled_h) > 0 else 0.0
                profit_potential_h = trades_per_day_h * mean_return_h
                return {
                    "trades_per_day": trades_per_day_h,
                    "risk_reward": risk_reward_h,
                    "profit_potential": profit_potential_h,
                }
            try:
                # Use the event_geometry group (index 1) for horizon calibration so that
                # the calibrated horizon respects the same search-space bounds as HPO.
                event_group = param_groups[1]
                horizon_spec = event_group.params.get("horizon_bars", {})
                h_low = int(horizon_spec.get("low", 8))
                h_high = int(horizon_spec.get("high", 56))
                h_step = int(horizon_spec.get("step", 2)) or 1
                candidate_horizons = list(range(h_low, h_high + 1, h_step))
                best_h = None
                best_potential = float("-inf")
                best_trades = 0.0
                best_rr = 0.0
                for h in candidate_horizons:
                    metrics_h = _evaluate_horizon_candidate(h)
                    trades_h = metrics_h["trades_per_day"]
                    rr_h = metrics_h["risk_reward"]
                    # RELAXED: Allow wider trade density range (0.2 - 6.0/day)
                    # to avoid prematurely filtering out potentially good horizons
                    if trades_h < 0.2 or trades_h > 6.0:
                        continue
                    # RELAXED: Lower R/R threshold (1.0 instead of 1.2)
                    # since HPO will refine the TPSL params later
                    if rr_h < 1.0:
                        continue
                    if metrics_h["profit_potential"] > best_potential:
                        best_potential = metrics_h["profit_potential"]
                        best_h = h
                        best_trades = trades_h
                        best_rr = rr_h
                if best_h is None and candidate_horizons:
                    best_h = int(np.median(candidate_horizons))
                calibrated_horizon = int(best_h) if best_h is not None else None
                if calibrated_horizon is not None:
                    tprint_info(
                        f"📏 Using calibrated horizon_bars={calibrated_horizon} on subsample (trades_per_day≈{best_trades:.3f}, rr≈{best_rr:.3f})"
                    )
            except Exception:
                calibrated_horizon = None
        else:
            calibrated_horizon = None

        # Storage for candidate label configurations
        candidate_pool: List[Dict[str, Any]] = []

        # Lightweight debug sampling for objective diagnostics
        debug_sample_limit = 50
        debug_sample_count = 0

        gate_stats: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # STAGE 0: SIGNAL/FEATURE HPO (KALMAN TUNING)
        # ------------------------------------------------------------------
        tprint_info("🧪 Stage 0: Optimizing Kalman Signal Parameters...")

        kalman_search_space = {
            # Widened to explore both smoother and more reactive regimes
            "kalman_Q": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
            "kalman_R": {"type": "float", "low": 1e-4, "high": 2e-1, "log": True},
        }
        def labeling_objective(
            params: Dict[str, Any],
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            model: Any | None = None,
            cv_folds: int = 1,
            scoring_metric: str = "custom_balanced_score",
            model_complexity: str = "fast",  # NEW: Model complexity level
            use_ensemble: bool = False,  # NEW: Use ensemble for strong complexity
            compute_diagnostics: bool = False,  # NEW: Whether to compute underfit diagnostics
            use_feature_selection: bool = False,
            use_resampling: bool = False,
            **kwargs: Any,
        ) -> Dict[str, float]:
            """Evaluate one labeling configuration with multi-objective scoring.

            This function:
            - Recomputes realized returns & binary labels with candidate TPSL parameters
            - Smooths labels via Kalman filter
            - Creates meta-features for learnability assessment
            - Computes learnability score with isotonic calibration (cross-validated AUC)
            - Computes realistic P&L edge metric
            - Applies regularization checks (temporal stability, regime consistency)
            - Optionally computes underfit diagnostics
            - Returns dict of objectives for Pareto frontier

            Args:
                model_complexity: "fast", "medium", or "strong" - controls model capacity
                use_ensemble: Whether to use model ensemble (for strong complexity)
                compute_diagnostics: Whether to compute underfit diagnostics

            Returns:
                Dict with keys: 'learnability', 'profitability', 'combined', 'edge'
            """
            # Determine CV splits based on model complexity
            cv_splits_map = {"fast": 3, "medium": 3, "strong": 5}  # Increased strong to 5 to match diagnostics
            cv_splits = cv_splits_map.get(model_complexity, 3)

            try:
                nonlocal debug_sample_count, gate_stats
                # NOTE: profit_thr_base/stop_to_profit_ratio constraints are now redundant
                # The Kalman multi-triple-barrier system handles multiple TP/SL configurations internally
                # Keeping only trail_distance for trailing profit parameters
                trail_dist = float(params.get("trail_distance", 0.0))

                # Removed profit/stop ratio constraints - handled by ensemble averaging
                # if profit_thr_base < 1.5 * stop_thr_base:
                #     # Early exit: invalid RR geometry
                #     tprint_warning(
                #         f"[EARLY_EXIT_RR] Config rejected: profit {profit_thr_base:.4f} < 2x stop {stop_thr_base:.4f}"
                #     )
                #     gate_stats["rr_profit_vs_stop"] = gate_stats.get("rr_profit_vs_stop", 0) + 1
                #     return {
                #         'learnability': 0.0,
                #         'profitability': -1e9,
                #         'edge': -1e9,
                #         'combined': -1e9,
                #     }

                # Extract parameters
                horizon = int(params["horizon_bars"])
                min_spacing = int(params["min_event_spacing"])

                if horizon <= 0:
                    gate_stats["invalid_horizon"] = gate_stats.get("invalid_horizon", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                spacing_ratio = float(min_spacing) / float(horizon)
                max_spacing_ratio = float(config.get("hpo_max_spacing_horizon_ratio", 8.0))
                if spacing_ratio > max_spacing_ratio:
                    gate_stats["spacing_vs_horizon"] = gate_stats.get("spacing_vs_horizon", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                kalman_Q = float(params.get("kalman_Q", 1e-4))
                kalman_R = float(params.get("kalman_R", 0.01))
                vol_baseline_window = int(params.get("vol_baseline_window", 96))

                # TPSL multipliers: use current params if present, otherwise fall back
                # to broad but reasonable defaults (kept consistent with diagnostics).
                profit_mult_min = float(params.get("profit_mult_min", 0.5))
                profit_mult_max = float(params.get("profit_mult_max", 2.0))
                stop_mult_min = float(params.get("stop_mult_min", 0.5))
                stop_mult_max = float(params.get("stop_mult_max", 2.0))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                # Hard RR constraint: even in the worst-case (smallest profit, largest stop),
                # require a minimum RR ~1.2 (relaxed for better balance while maintaining positive expectancy).
                worst_rr = (profit_thr_base * profit_mult_min) / max(stop_thr_base * stop_mult_max, 1e-8)
                if worst_rr < 1.2:
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(
                            f"[EARLY_EXIT_RR] Rejecting config due to worst_rr={worst_rr:.3f} < 1.5 "
                            f"(profit_thr_base={profit_thr_base:.4f}, stop_thr_base={stop_thr_base:.4f}, "
                            f"profit_mult_min={profit_mult_min:.3f}, stop_mult_max={stop_mult_max:.3f})"
                        )
                        debug_sample_count += 1
                    gate_stats["rr_worst_rr"] = gate_stats.get("rr_worst_rr", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # Use safe defaults when target_transform params are not part of the current group
                iso_min_prob = float(params.get("iso_min_prob", 0.05))
                # Allow slightly stronger clipping on both tails; keep symmetric band.
                iso_min_prob = max(0.05, min(0.15, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.85, min(1.0, iso_max_prob))

                q_high = float(params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.98, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                # Economic floor multiplier for vol-scaled labels and isotonic mapping
                econ_min_mult = float(params.get("econ_min_return_multiple", 1.0))
                if not np.isfinite(econ_min_mult) or econ_min_mult <= 0:
                    econ_min_mult = 1.0
                # Clamp economic multiplier to prevent zero-target degeneracy
                econ_min_mult = max(1.0, min(econ_min_mult, 2.0))

                # Label quantile thresholds (regime-aware when regimes are present).
                # Default to a slightly denser configuration (~40% positives target):
                # bottom 40% vs top 40% tails.
                label_low_q = float(params.get("label_low_q", 0.40))
                label_high_q = float(params.get("label_high_q", 0.60))
                # Guard-rail: ensure a proper ordering and keep them away from extremes.
                label_low_q = max(0.10, min(0.60, label_low_q))
                label_high_q = max(0.40, min(0.90, label_high_q))
                if label_high_q <= label_low_q:
                    label_low_q, label_high_q = 0.40, 0.60

                min_label_band_width = float(config.get("hpo_min_label_band_width", 0.15))
                if (label_high_q - label_low_q) < min_label_band_width:
                    gate_stats["label_band_too_narrow"] = gate_stats.get("label_band_too_narrow", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # NEW: R-multiple threshold for labeling - controls trade velocity filter
                r_multiple_threshold = float(params.get("r_multiple_pos_threshold", 0.5))
                # Relax lower bound to allow almost no trade-velocity filtering if features are good enough
                r_multiple_threshold = max(0.01, min(1.2, r_multiple_threshold))

                # NEW: Transaction cost multiplier - allows HPO to explore cost sensitivity
                tx_cost_mult = float(params.get("transaction_cost_mult", 1.0))
                tx_cost_mult = max(1.0, min(1.2, tx_cost_mult))
                effective_tx_cost = DEFAULT_TRANSACTION_COST * tx_cost_mult

                # NEW: Signal generation parameters
                cusum_threshold = float(params.get("cusum_threshold", 0.015))
                cusum_threshold = max(0.010, min(0.035, cusum_threshold))
                target_signal_density = float(params.get("target_signal_density", 20.0))
                target_signal_density = max(5.0, min(50.0, target_signal_density))

                # --- Recompute realized returns ---
                # NO FUTURE LEAKAGE in volatility-based thresholds:
                # - volatility_1d is backward-looking (rolling 96-bar std)
                # - vol_baseline is backward-looking (rolling mean of past volatility)
                # - vol_factor at time T uses only volatility from T-vol_baseline_window to T
                # Use cache for vol_baseline computation
                vol_baseline = computation_cache.get_vol_baseline(volatility_1d, vol_baseline_window)
                vol_factor = volatility_1d / (vol_baseline + 1e-8)

                # Use cache for ATR and trend strength (invariant to HPO params)
                atr_series = computation_cache.get_atr(market_data, config)
                trend_strength = computation_cache.get_trend_strength(market_data, config)

                trend_alpha = float(config.get("trend_strength_alpha_profit", 0.5))
                trend_beta = float(config.get("trend_strength_beta_stop", 0.5))

                profit_factor = 1.0 + trend_alpha * trend_strength
                stop_factor = 1.0 + trend_beta * trend_strength

                adaptive_profit = profit_thr_base * vol_factor * profit_factor
                adaptive_stop = stop_thr_base * vol_factor * stop_factor
                adaptive_profit = adaptive_profit.clip(
                    lower=profit_thr_base * profit_mult_min,
                    upper=profit_thr_base * profit_mult_max,
                )
                adaptive_stop = adaptive_stop.clip(
                    lower=stop_thr_base * stop_mult_min,
                    upper=stop_thr_base * stop_mult_max,
                )

                # Use cache for primary signals (only regenerate if cusum_threshold differs)
                default_cusum = 0.015
                try:
                    signals_to_use = computation_cache.get_primary_signals(
                        market_data,
                        cusum_threshold,
                        target_signal_density,
                        default_cusum=default_cusum,
                    )
                except Exception:
                    signals_to_use = primary_signals  # Fallback if cache/regeneration fails

                (
                    realized_returns,
                    binary_labels,
                    exit_reasons,
                    event_durations,
                    mfe_series,
                    mae_series,
                    _binary_labels_long,  # Not used in HPO objective function
                    _binary_labels_short,  # Not used in HPO objective function
                ) = compute_realized_returns(
                    market_data,
                    signals_to_use,
                    profit_threshold=adaptive_profit,
                    stop_threshold=adaptive_stop,
                    horizon=horizon,
                    transaction_cost=effective_tx_cost,  # Use HPO-tunable tx cost
                    min_event_spacing=min_spacing,
                    atr_series=atr_series,
                    trail_distance_atr_mult=trail_dist,
                )

                # Basic diagnostics on raw realized returns and labels before
                # any vol-scaling or quantile-based relabeling.
                # Reduced logging frequency to avoid noise in hot loop
                n_raw_events = len(realized_returns)
                n_raw_labeled = int((~binary_labels.isna()).sum())
                # Only log every 10th sample to reduce noise
                if debug_sample_count < debug_sample_limit and debug_sample_count % 10 == 0:
                    tprint_info(
                        f"[HPO_DEBUG] events={n_raw_events}, labeled={n_raw_labeled}, "
                        f"profit={profit_thr_base:.4f}, stop={stop_thr_base:.4f}",
                    )

                # Replace legacy R-multiple based labels with quantile-based labels
                # derived from volatility-scaled realized returns, to improve label
                # balance and economic relevance in HPO scoring.
                #
                # NO FUTURE VOLATILITY LEAKAGE:
                # - volatility_1d at time T uses only data from T-96 to T-1 (backward-looking)
                # - realized_returns at time T uses future prices (expected for labeling)
                # - vol_scaled = realized_returns / past_volatility (no future vol leakage)
                #
                # NOTE: Quantile thresholds are computed from ALL training data, which is
                # acceptable for HPO/training. In production, use expanding/rolling quantiles.
                vol_scaled_returns = compute_vol_scaled_returns_for_events(
                    realized_returns=realized_returns,
                    volatility=volatility_1d,
                    econ_min_return_multiple=econ_min_mult,
                )

                n_vol_scaled_events = int(vol_scaled_returns.dropna().size)
                # Reduced logging frequency
                if debug_sample_count < debug_sample_limit and debug_sample_count % 10 == 0:
                    tprint_info(f"[HPO_DEBUG] vol_scaled_events={n_vol_scaled_events}")

                # Decide whether to use regime-aware quantiles based on the
                # attached HMM regimes (typically 1h) on market_data.
                regimes_for_labeling = None
                if config.get("enable_regime_aware_quantiles", True) and "hmm_regime_label_1h" in market_data.columns:
                    regimes_for_labeling = market_data["hmm_regime_label_1h"]

                # Use rolling quantiles by default to match production and avoid look-ahead bias
                # ENFORCE rolling quantiles for HPO to prevent leakage
                use_rolling = config.get("use_rolling_quantiles", True)
                if not use_rolling:
                    tprint_warning(
                        "⚠️ use_rolling_quantiles=False detected in HPO. "
                        "Forcing rolling quantiles to prevent look-ahead bias."
                    )
                    use_rolling = True
                rolling_lookback = int(config.get("rolling_quantile_lookback_bars", 3000))
                rolling_min_periods = int(config.get("rolling_quantile_min_periods", 300))
                
                # Validate rolling parameters are reasonable
                if rolling_lookback < 100:
                    tprint_warning(
                        f"⚠️ rolling_quantile_lookback_bars={rolling_lookback} is too small. "
                        f"Using minimum of 100 bars."
                    )
                    rolling_lookback = max(100, rolling_lookback)

                def _make_quantile_labels(vol_scaled_series: pd.Series) -> pd.Series:
                    """Helper to create (regime-aware) quantile labels from a score series."""
                    if use_rolling:
                        if regimes_for_labeling is not None:
                            return create_rolling_regime_aware_quantile_labels_from_vol_scaled_returns(
                                vol_scaled=vol_scaled_series,
                                regimes=regimes_for_labeling,
                                low_q=label_low_q,
                                high_q=label_high_q,
                                lookback_bars=rolling_lookback,
                                min_periods=rolling_min_periods,
                            )
                        return create_rolling_quantile_labels_from_vol_scaled_returns(
                            vol_scaled=vol_scaled_series,
                            low_q=label_low_q,
                            high_q=label_high_q,
                            lookback_bars=rolling_lookback,
                            min_periods=rolling_min_periods,
                        )
                    else:
                        # Legacy global quantiles (has look-ahead bias)
                        if regimes_for_labeling is not None:
                            return create_regime_aware_quantile_labels_from_vol_scaled_returns(
                                vol_scaled=vol_scaled_series,
                                regimes=regimes_for_labeling,
                                low_q=label_low_q,
                                high_q=label_high_q,
                            )
                        return create_quantile_labels_from_vol_scaled_returns(
                            vol_scaled=vol_scaled_series,
                            low_q=label_low_q,
                            high_q=label_high_q,
                        )

                # Primary quantile labels on vol-scaled returns with the
                # HPO-chosen economic floor.
                quantile_labels = _make_quantile_labels(vol_scaled_returns)
                binary_labels = quantile_labels

            except Exception as quantile_exc:
                # Handle quantile label generation errors gracefully
                tprint_warning(f"[QUANTILE_LABEL_ERROR] {quantile_exc}")
                gate_stats["quantile_error"] = gate_stats.get("quantile_error", 0) + 1
                return {
                    'learnability': 0.0,
                    'profitability': -1e9,
                    'edge': -1e9,
                    'combined': -1e9,
                }

        def kalman_objective(params: Dict[str, Any]) -> float:
            """
            Stage 0: RTS Smoother Optimization for Label Generation.
            
            Objective: Maximize Signal-to-Noise Ratio (SNR) of the raw price series.
            Uses RTS (Rauch-Tung-Striebel) smoother which is ACAUSAL (zero-lag) - 
            ideal for generating training labels.
            
            The optimized Q and R values will also be used in the standard (causal)
            Kalman Filter for live feature generation.
            
            Loss components:
            1. Smoothness: Minimal "wiggle" (2nd derivative)
            2. Tracking Error: RMSE from raw prices (bias/oversmoothing penalty)
            3. Amplitude Fidelity: Preserve ~95% of price volatility
            """
            Q = params['kalman_Q']
            R = params['kalman_R']
            
            # Get raw close prices
            raw_close = close_series.values
            
            if len(raw_close) < 100:
                return -10.0  # Reject if insufficient data
            
            try:
                # Run RTS Smoother (acausal, zero-lag)
                smoothed_close, smoothed_cov = rts_smoother_1d(
                    prices=raw_close,
                    Q=Q,
                    R=R,
                    init_val=None,
                    init_cov=1.0,
                )
                
                # Compute robust labeling loss
                # Loss weights: alpha=smoothness, beta=tracking, gamma=amplitude
                loss, details = robust_labeling_loss(
                    smoothed=smoothed_close,
                    raw=raw_close,
                    alpha=1.0,   # Smoothness weight
                    beta=1.0,    # Tracking error weight
                    gamma=1.0,   # Amplitude fidelity weight
                    is_acausal=True,  # RTS is acausal
                )
                
                # Optimizer maximizes, so return negative loss
                # Also add bonus for amplitude ratio being close to 0.95
                amp_ratio = details.get("amp_ratio", 0.95)
                amp_bonus = max(0, 0.1 - abs(amp_ratio - 0.95))  # Small bonus for good amplitude
                
                score = -loss + amp_bonus
                
                return float(score) if np.isfinite(score) else -10.0
                
            except Exception as e:
                tprint_warning(f"[KALMAN_OBJ_ERROR] {e}")
                return -10.0

        # Run Stage 0 optimization
        kalman_loss: float = float("nan")
        kalman_loss_details: Dict[str, Any] = {}

        stage0_loaded_from: Optional[str] = None
        if stage_rank["stage0"] < start_rank:
            loaded_params, loaded_path = _load_stage_best_params("stage0")
            best_kalman_params = dict(loaded_params or {})
            stage0_loaded_from = str(loaded_path) if loaded_path is not None else None
            if not best_kalman_params:
                best_kalman_params = {"kalman_Q": 1e-4, "kalman_R": 0.01}
            kalman_result = {"best_params": dict(best_kalman_params), "best_value": 0.0, "history": []}
            tprint_info(
                f"♻️ Stage 0 skipped (start_at={start_at_canonical}); loaded best params from {stage0_loaded_from}"
            )
        else:
            kalman_optimizer = BayesianTPEOptimizer(
                config=OptimizationConfig(
                    n_trials=60,
                    execution_mode="full",
                    direction="maximize",
                    seed=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=0)
                )
            )
            kalman_result = kalman_optimizer.optimize(objective=kalman_objective, search_space=kalman_search_space)
            best_kalman_params = kalman_result.get("best_params", {})

        # Log the results with loss details
        best_Q = best_kalman_params.get("kalman_Q")
        best_R = best_kalman_params.get("kalman_R")
        try:
            best_Q = float(best_Q) if best_Q is not None else float("nan")
        except Exception:
            best_Q = float("nan")
        try:
            best_R = float(best_R) if best_R is not None else float("nan")
        except Exception:
            best_R = float("nan")
        if not np.isfinite(best_Q) or best_Q <= 0.0:
            best_Q = 1e-4
        if not np.isfinite(best_R) or best_R <= 0.0:
            best_R = 0.01
        # Compute final loss details for logging
        try:
            final_smoothed, _ = rts_smoother_1d(close_series.values, Q=best_Q, R=best_R)
            final_loss, final_details = robust_labeling_loss(final_smoothed, close_series.values, is_acausal=True)
            kalman_loss = float(final_loss)
            kalman_loss_details = final_details or {}
            tprint_success(
                f"✅ Stage 0 Complete. Loss: {final_loss:.4f} "
                f"(smooth={final_details['smooth']:.4f}, track={final_details['track']:.4f}, "
                f"amp={final_details['amp']:.4f}, amp_ratio={final_details['amp_ratio']:.3f})"
            )
        except Exception:
            try:
                bv = kalman_result.get("best_value", 0.0) if isinstance(kalman_result, dict) else 0.0
                bv = float(bv) if bv is not None and np.isfinite(float(bv)) else 0.0
            except Exception:
                bv = 0.0
            tprint_success(f"✅ Stage 0 Complete. Best Score: {bv:.4f}")

        tprint_info(f"   Best RTS/Kalman Params: Q={best_Q:.2e}, R={best_R:.2e}")
        tprint_info("   Note: RTS (acausal) for labels, Kalman (causal) for live features")

        hpo_stage_reports: Dict[str, Any] = {}

        # Persist Stage 0 trial diagnostics for offline analysis
        stage0_csv: Optional[Path] = None
        try:
            kalman_history = kalman_result.get("history", []) if isinstance(kalman_result, dict) else []
            stage0_rows = []
            for trial in kalman_history:
                params = trial.get("params", {}) if isinstance(trial, dict) else {}
                q_val = float(params.get("kalman_Q", 1e-4))
                r_val = float(params.get("kalman_R", 0.01))

                # Recompute loss components for this (Q, R) pair
                try:
                    smoothed_trial, _ = rts_smoother_1d(
                        prices=close_series.values,
                        Q=q_val,
                        R=r_val,
                        init_val=None,
                        init_cov=1.0,
                    )
                    loss_trial, details_trial = robust_labeling_loss(
                        smoothed=smoothed_trial,
                        raw=close_series.values,
                        is_acausal=True,
                    )
                except Exception:
                    loss_trial, details_trial = float("nan"), {}

                row = {
                    "trial_number": trial.get("trial_number") if isinstance(trial, dict) else None,
                    "kalman_Q": q_val,
                    "kalman_R": r_val,
                    "score": float(trial.get("value", float("nan"))) if isinstance(trial, dict) else float("nan"),
                    "loss": float(loss_trial),
                    "smooth": float(details_trial.get("smooth", float("nan"))),
                    "track": float(details_trial.get("track", float("nan"))),
                    "amp": float(details_trial.get("amp", float("nan"))),
                    "amp_ratio": float(details_trial.get("amp_ratio", float("nan"))),
                }
                stage0_rows.append(row)

            if stage0_rows:
                ts_stage0 = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                stage0_csv = outcomes_dir / f"hpo_stage0_kalman_trials_{symbol}_{timeframe}_{ts_stage0}.csv"
                pd.DataFrame(stage0_rows).to_csv(stage0_csv, index=False)
                tprint_info(f"   💾 Saved Stage 0 Kalman trial diagnostics to {stage0_csv}")
        except Exception as stage0_exc:
            tprint_warning(f"   ⚠️ Failed to save Stage 0 Kalman trial diagnostics: {stage0_exc}")

        try:
            stage0_report = _write_hpo_stage_report(
                outcomes_dir=outcomes_dir,
                run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
                stage_id="stage0_kalman",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                best_params=dict(best_kalman_params) if isinstance(best_kalman_params, dict) else {},
                metrics={
                    "best_value": kalman_result.get("best_value", None),
                    "loss": kalman_loss,
                    "loss_details": kalman_loss_details,
                },
                search_space=kalman_search_space,
                trials_csv_path=stage0_csv,
                history_json_path=None,
            )
            hpo_stage_reports["stage0"] = stage0_report
        except Exception as stage0_report_exc:
            tprint_warning(f"   ⚠️ Failed to write Stage 0 report: {stage0_report_exc}")

        # ------------------------------------------------------------------
        # 1. LAYER 1: WEIGHTING OPTIMIZATION
        # ------------------------------------------------------------------
        tprint_info("🧪 Layer 1: Optimizing Sample Weighting Parameters...")

# ... (rest of the code remains the same)
        # Generate baseline events using defaults (as TBM params are Layer 2)
        baseline_profit = (atr_frac * 2.0).astype(float).clip(lower=0.008)
        baseline_stop = (atr_frac * 1.0).astype(float).clip(lower=0.004)

        (
            baseline_returns,
            binary_labels_raw,
            exit_reasons_raw,
            event_durations_raw,
            mfe_raw,
            mae_raw,
            _, _,
        ) = compute_realized_returns(
            market_data,
            primary_signals,
            profit_threshold=baseline_profit,
            stop_threshold=baseline_stop,
            horizon=12,
            transaction_cost=DEFAULT_TRANSACTION_COST,
            min_event_spacing=2,
        )

        # Create baseline quantile labels (outer-scope) so Layer 2 feature selection
        # and post-HPO evaluation have a valid `binary_labels` series.
        try:
            label_low_q_baseline = float(config.get("label_low_q", 0.40))
            label_high_q_baseline = float(config.get("label_high_q", 0.60))
            vol_scaled_baseline = compute_vol_scaled_returns_for_events(
                realized_returns=baseline_returns,
                volatility=volatility_1d,
                econ_min_return_multiple=float(config.get("econ_min_return_multiple", 1.0)),
            )
            binary_labels = create_rolling_quantile_labels_from_vol_scaled_returns(
                vol_scaled=vol_scaled_baseline,
                low_q=label_low_q_baseline,
                high_q=label_high_q_baseline,
                lookback_bars=int(config.get("rolling_quantile_lookback_bars", 3000)),
                min_periods=int(config.get("rolling_quantile_min_periods", 300)),
            )
        except Exception:
            # Fallback: directional labels from returns when quantile labeling fails.
            binary_labels = pd.Series(np.nan, index=baseline_returns.index)
            mask_evt = ~baseline_returns.isna()
            binary_labels.loc[mask_evt & (baseline_returns > 0)] = 1.0
            binary_labels.loc[mask_evt & (baseline_returns <= 0)] = 0.0

        valid_mask = ~baseline_returns.isna()
        baseline_t_events = baseline_returns.index[valid_mask]
        baseline_returns_clean = baseline_returns[valid_mask]

        committee_agreement_scores_l1: Optional[np.ndarray] = None
        committee_mag_factors_l1: Optional[np.ndarray] = None

        try:
            enable_committee_weight_factor_for_l1 = bool(
                config.get("layer1_optimize_committee_weight_factor", True)
            ) and bool(config.get("enable_committee_weight_factor", True))
        except Exception:
            enable_committee_weight_factor_for_l1 = False

        if enable_committee_weight_factor_for_l1 and len(baseline_t_events) >= 50:
            try:
                best_Q_l1 = best_kalman_params.get('kalman_Q', 1e-4)
                best_R_l1 = best_kalman_params.get('kalman_R', 0.01)

                mk_data_voting_l1 = market_data.copy()
                kalman_price_smooth_l1, kalman_vol_smooth_l1 = compute_kalman_smoothed_price_and_volatility(
                    prices=market_data['close'],
                    process_noise=best_Q_l1,
                    measurement_noise=best_R_l1,
                    vol_window=20,
                )
                mk_data_voting_l1['kalman_price'] = kalman_price_smooth_l1
                mk_data_voting_l1['kalman_volatility'] = kalman_vol_smooth_l1

                base_profiles_l1 = {
                    "scalp": (1.2, 0.6, 8),
                    "swing": (1.8, 0.9, 12),
                    "trend": (2.4, 1.2, 24),
                }
                vol_scalars_l1 = {"lower": 0.8, "upper": 1.2}

                committee_configs_l1: List[TripleBarrierConfig] = []
                committee_names_l1: List[str] = []
                for p_name, (tp_base, sl_base, h_base) in base_profiles_l1.items():
                    for v_name, v_scalar in vol_scalars_l1.items():
                        committee_configs_l1.append(
                            TripleBarrierConfig(
                                tp_multiplier=tp_base * v_scalar,
                                sl_multiplier=sl_base * v_scalar,
                                horizon=h_base,
                            )
                        )
                        committee_names_l1.append(f"{p_name}_{v_name}")

                committee_results_l1 = compute_multi_triple_barrier_outcomes_vectorized(
                    market_data=mk_data_voting_l1,
                    primary_signals=primary_signals,
                    configs=committee_configs_l1,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                )

                event_mask_l1 = primary_signals['consensus'] != 0
                event_idx_l1 = primary_signals[event_mask_l1].index
                if len(event_idx_l1) > 0:
                    label_matrix_l1 = np.zeros((len(event_idx_l1), len(committee_configs_l1)), dtype=np.int8)
                    returns_matrix_l1 = np.full((len(event_idx_l1), len(committee_configs_l1)), np.nan, dtype=np.float32)

                    for i, res in enumerate(committee_results_l1):
                        lbls = res['labels'].reindex(event_idx_l1).fillna(0).values.astype(int)
                        rets = res['returns'].reindex(event_idx_l1).values.astype(np.float32)
                        label_matrix_l1[:, i] = lbls
                        returns_matrix_l1[:, i] = rets

                    fired_mask_l1 = (label_matrix_l1 != 0)
                    ret_mat_l1 = np.asarray(returns_matrix_l1, dtype=float)
                    ret_mat_l1 = np.where(fired_mask_l1, ret_mat_l1, np.nan)

                    abs_ret_l1 = np.abs(ret_mat_l1)
                    abs_ret_mean_l1 = np.nanmean(abs_ret_l1, axis=1)
                    abs_ret_mean_l1 = np.where(np.isfinite(abs_ret_mean_l1), abs_ret_mean_l1, 0.0)

                    positive_abs_l1 = abs_ret_mean_l1[abs_ret_mean_l1 > 0]
                    abs_med_l1 = float(np.nanmedian(positive_abs_l1)) if positive_abs_l1.size > 0 else 0.0
                    if np.isfinite(abs_med_l1) and abs_med_l1 > 0:
                        mag_factor_l1 = abs_ret_mean_l1 / (abs_med_l1 + 1e-12)
                    else:
                        mag_factor_l1 = np.ones_like(abs_ret_mean_l1, dtype=float)

                    sign_mat_l1 = np.asarray(label_matrix_l1, dtype=float)
                    sign_mat_l1 = np.where(fired_mask_l1, np.sign(sign_mat_l1), np.nan)
                    mean_sign_l1 = np.nanmean(sign_mat_l1, axis=1)
                    agree_l1 = np.abs(mean_sign_l1)
                    agree_l1 = np.where(np.isfinite(agree_l1), agree_l1, 0.0)
                    agree_l1 = np.clip(agree_l1, 0.0, 1.0)

                    committee_agreement_scores_l1 = (
                        pd.Series(agree_l1, index=event_idx_l1)
                        .reindex(baseline_t_events)
                        .fillna(0.0)
                        .values.astype(float)
                    )
                    committee_mag_factors_l1 = (
                        pd.Series(mag_factor_l1, index=event_idx_l1)
                        .reindex(baseline_t_events)
                        .fillna(1.0)
                        .values.astype(float)
                    )
            except Exception:
                committee_agreement_scores_l1 = None
                committee_mag_factors_l1 = None

        layer1_loaded_from: Optional[str] = None
        if stage_rank["layer1"] < start_rank:
            loaded_params, loaded_path = _load_stage_best_params("layer1")
            best_weighting_params = dict(loaded_params or {})
            layer1_loaded_from = str(loaded_path) if loaded_path is not None else None
            tprint_info(
                f"♻️ Layer 1 skipped (start_at={start_at_canonical}); loaded best params from {layer1_loaded_from}"
            )
        else:
            if len(baseline_t_events) < 50:
                tprint_warning(f"⚠️ Too few baseline events ({len(baseline_t_events)}) for Layer 1. Using defaults.")
                best_weighting_params = {
                    'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                    'uniq_intensity': 1.0, 'exp_mag': 1.0, 'exp_learn': 1.0,
                    'exp_uniq': 1.0, 'exp_cross': 1.0, 'downside_multiplier': 1.0
                }
            else:
                try:
                    from src.training.steps.labeling.generate_weights_per_label import run_layer1_optimization
                    best_weighting_params = run_layer1_optimization(
                        symbol=symbol,
                        timeframe=timeframe,
                        market_data=market_data,
                        labels=baseline_returns_clean,
                        committee_agreement_scores=committee_agreement_scores_l1,
                        committee_mag_factors=committee_mag_factors_l1,
                    )
                except Exception as e:
                    tprint_warning(f"⚠️ Layer 1 optimization failed: {e}. Using defaults.")
                    best_weighting_params = {
                        'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                        'uniq_intensity': 1.0, 'exp_mag': 1.0, 'exp_learn': 1.0,
                        'exp_uniq': 1.0, 'exp_cross': 1.0, 'downside_multiplier': 1.0
                    }

        tprint_success(f"✅ Layer 1 Complete. Best Weighting Params: {best_weighting_params}")

        # Persist Layer 1 params immediately
        l1_path: Optional[Path] = None
        try:
            ts = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            l1_path = Path("outcomes") / f"hpo_layer1_best_params_{symbol}_{timeframe}_{ts}.json"
            l1_payload = {
                "best_params": best_weighting_params,
                "timestamp": ts,
            }
            l1_path.parent.mkdir(parents=True, exist_ok=True)
            with open(l1_path, "w") as f:
                json.dump(l1_payload, f, indent=2, default=str)
            tprint_info(f"   💾 Saved Layer 1 best params to {l1_path}")
        except Exception as l1_exc:
            tprint_warning(f"   ⚠️ Failed to save Layer 1 params: {l1_exc}")

        try:
            l1_trials_csv = _find_latest_path(
                outcomes_dir=outcomes_dir,
                pattern=f"hpo_layer1_trials_{symbol}_{timeframe}_*.csv",
            )
            l1_report = _write_hpo_stage_report(
                outcomes_dir=outcomes_dir,
                run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
                stage_id="layer1_weighting",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                best_params=dict(best_weighting_params) if isinstance(best_weighting_params, dict) else {},
                metrics={
                    "best_params_path": str(l1_path) if l1_path is not None else None,
                },
                search_space=None,
                trials_csv_path=l1_trials_csv,
                history_json_path=None,
            )
            hpo_stage_reports["layer1"] = l1_report
        except Exception as l1_report_exc:
            tprint_warning(f"   ⚠️ Failed to write Layer 1 report: {l1_report_exc}")

        enable_committee_voting_hpo = bool(config.get("enable_committee_voting_hpo", True))
        enable_committee_weight_factor = bool(config.get("enable_committee_weight_factor", True))
        allow_committee_fallback_to_standard = bool(
            config.get("allow_committee_fallback_to_standard", False)
        )
        layer2_mode = "committee" if enable_committee_voting_hpo else "standard"
        tprint_info(f"🔧 Layer 2 mode: {layer2_mode} (enable_committee_voting_hpo={enable_committee_voting_hpo})")

        committee_weight_factor_series: Optional[pd.Series] = None

        if enable_committee_voting_hpo or enable_committee_weight_factor:
            if enable_committee_voting_hpo:
                tprint_info("🧪 Layer 2: Optimizing Committee Consensus Weights...")

            # A. PRE-COMPUTE COMMITTEE LABEL MATRIX (The "Expert Panel")
            tprint_info("🏗️ Pre-computing Committee of 6 Label Matrix...")

            # 1. Define the 6 Profiles
            # Scalp: Tight (TP 1.2 / SL 0.6)
            # Swing: Balanced (TP 2.0 / SL 1.0)
            # Trend: Looser (TP 3.0 / SL 1.5)
            # Multipliers: Lower (0.8x), Upper (1.2x)

            # Base Multipliers (TP, SL)
            # Scalp: 1.2/0.6
            # Swing: 2.0/1.0
            # Trend: 3.0/1.5
            base_profiles = {
                "scalp": (1.2, 0.6, 8),
                "swing": (1.8, 0.9, 12),
                "trend": (2.4, 1.2, 24),
            }
            vol_scalars = {"lower": 0.8, "upper": 1.2}

            committee_configs = []
            committee_names = []

            for p_name, (tp_base, sl_base, h_base) in base_profiles.items():
                for v_name, v_scalar in vol_scalars.items():
                    config_id = f"{p_name}_{v_name}"
                    # Scale multipliers (effectively scaling volatility assumption)
                    committee_configs.append(
                        TripleBarrierConfig(
                            tp_multiplier=tp_base * v_scalar,
                            sl_multiplier=sl_base * v_scalar,
                            horizon=h_base,
                        )
                    )
                    committee_names.append(config_id)

            # 2. Vectorized computation of all 6 outcomes
            try:
                # Add Kalman columns if not present (Stage 0 result)
                best_Q = best_kalman_params.get('kalman_Q', 1e-4)
                best_R = best_kalman_params.get('kalman_R', 0.01)

                # Re-compute Kalman smooth data for labeling (acausal/smooth for labeling)
                # using the optimized parameters
                kalman_price_smooth, kalman_vol_smooth = compute_kalman_smoothed_price_and_volatility(
                    prices=market_data['close'],
                    process_noise=best_Q,
                    measurement_noise=best_R,
                    vol_window=20
                )

                mk_data_voting = market_data.copy()
                mk_data_voting['kalman_price'] = kalman_price_smooth
                mk_data_voting['kalman_volatility'] = kalman_vol_smooth

                committee_results = compute_multi_triple_barrier_outcomes_vectorized(
                    market_data=mk_data_voting,
                    primary_signals=primary_signals,
                    configs=committee_configs,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                )

                # 3. Assemble Label Matrix (Rows=Events, Cols=6 Experts)
                # Find common events (primary_signals != 0)
                event_mask = primary_signals['consensus'] != 0
                event_idx = primary_signals[event_mask].index

                # Initialize matrices
                label_matrix_values = np.zeros((len(event_idx), len(committee_configs)), dtype=np.int8)
                returns_matrix_values = np.full((len(event_idx), len(committee_configs)), np.nan, dtype=np.float32)

                for i, res in enumerate(committee_results):
                    # align to event_idx
                    lbls = res['labels'].reindex(event_idx).fillna(0).values.astype(int)
                    rets = res['returns'].reindex(event_idx).values.astype(np.float32)

                    label_matrix_values[:, i] = lbls
                    returns_matrix_values[:, i] = rets

                tprint_success(f"✅ Committee Matrices Built: {label_matrix_values.shape} (Events x Experts)")

                try:
                    n_ev = int(label_matrix_values.shape[0])
                    for j, name in enumerate(list(committee_names)):
                        col = np.asarray(label_matrix_values[:, j], dtype=float)
                        if col.size <= 0:
                            continue
                        frac_pos = float(np.mean(col > 0.0))
                        frac_neg = float(np.mean(col < 0.0))
                        frac_zero = float(np.mean(col == 0.0))
                        tprint_info(
                            f"   [committee expert] {name}: +={frac_pos:.2%}, -={frac_neg:.2%}, 0={frac_zero:.2%} (n={n_ev})"
                        )
                except Exception:
                    pass

                try:
                    fired_mask = (label_matrix_values != 0)
                    ret_mat = np.asarray(returns_matrix_values, dtype=float)
                    ret_mat = np.where(fired_mask, ret_mat, np.nan)

                    abs_ret = np.abs(ret_mat)
                    abs_ret_mean = np.nanmean(abs_ret, axis=1)
                    abs_ret_mean = np.where(np.isfinite(abs_ret_mean), abs_ret_mean, 0.0)

                    positive_abs = abs_ret_mean[abs_ret_mean > 0]
                    abs_med = float(np.nanmedian(positive_abs)) if positive_abs.size > 0 else 0.0
                    if np.isfinite(abs_med) and abs_med > 0:
                        mag_factor = abs_ret_mean / (abs_med + 1e-12)
                    else:
                        mag_factor = np.ones_like(abs_ret_mean, dtype=float)

                    sign_mat = np.asarray(label_matrix_values, dtype=float)
                    sign_mat = np.where(fired_mask, np.sign(sign_mat), np.nan)
                    mean_sign = np.nanmean(sign_mat, axis=1)
                    agree = np.abs(mean_sign)
                    agree = np.where(np.isfinite(agree), agree, 0.0)
                    agree = np.clip(agree, 0.0, 1.0)

                    alpha = float(
                        best_weighting_params.get(
                            "committee_agreement_alpha",
                            config.get("committee_agreement_alpha", 0.5),
                        )
                    )
                    mag_clip = float(
                        best_weighting_params.get(
                            "committee_mag_clip",
                            config.get("committee_mag_clip", 5.0),
                        )
                    )
                    mag_factor = np.where(np.isfinite(mag_factor), mag_factor, 1.0)
                    mag_factor = np.clip(mag_factor, 0.0, mag_clip)

                    factor = (1.0 + alpha * agree) * mag_factor
                    factor_mean = float(np.nanmean(factor[np.isfinite(factor)])) if np.isfinite(factor).any() else 1.0
                    if np.isfinite(factor_mean) and factor_mean > 0:
                        factor = factor / factor_mean
                    else:
                        factor = np.ones_like(factor, dtype=float)

                    committee_weight_factor_series = pd.Series(factor, index=event_idx)

                    try:
                        if bool(config.get("log_committee_weight_factor", True)):
                            v = committee_weight_factor_series.values.astype(float)
                            v = v[np.isfinite(v)]
                            if v.size > 0:
                                tprint_info(
                                    "   [committee weight factor] "
                                    f"n={int(v.size)}, mean={float(np.mean(v)):.4f}, min={float(np.min(v)):.4f}, max={float(np.max(v)):.4f}"
                                )
                    except Exception:
                        pass
                except Exception:
                    committee_weight_factor_series = None

            except Exception as e:
                if allow_committee_fallback_to_standard:
                    tprint_warning(
                        f"⚠️ Committee pre-computation failed: {e}. Falling back to standard HPO."
                    )
                    layer2_mode = "standard"
                else:
                    tprint_error(
                        f"❌ Committee pre-computation failed: {e}. Aborting (set allow_committee_fallback_to_standard=true to fallback)."
                    )
                    return {"success": False, "error": str(e)}

        if layer2_mode == "committee":
            # Updated search space for Consensus Voting
            layer2_search_space = {
                "w_scalp": {"type": "float", "low": 0.0, "high": 2.0},
                "w_swing": {"type": "float", "low": 0.0, "high": 2.0},
                "w_trend": {"type": "float", "low": 0.0, "high": 2.0},
                "consensus_quantile": {"type": "float", "low": 0.50, "high": 0.85},
            }
        else:
            tprint_info("🧪 Layer 2: Optimizing Trading Parameters...")
            layer2_search_space = {
                "trail_distance_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
            }

        if layer2_mode == "committee":
            l2_trial_counter = 0

            def layer2_objective(trial_params: Dict[str, Any]) -> float:
                """
                Layer 2 objective using Committee Voting:
                - Optimizes weights for [Scalp, Swing, Trend] experts.
                - Computes consensus score.
                - Thresholds for binary labels.
                """
                nonlocal l2_trial_counter
                l2_trial_counter += 1

                w_scalp = trial_params["w_scalp"]
                w_swing = trial_params["w_swing"]
                w_trend = trial_params["w_trend"]
                threshold = float(trial_params.get("consensus_threshold", 0.5))
                cq = trial_params.get("consensus_quantile", None)

                try:
                    metrics_trial = _compute_layer2_metrics_committee(trial_params)
                    utility = float(metrics_trial.get("utility", -1.0))
                    if not np.isfinite(float(utility)):
                        utility = -1.0
                except Exception:
                    metrics_trial = {}
                    utility = -1.0

                try:
                    if bool(config.get("layer2_log_trials", True)):
                        every = int(config.get("layer2_trial_log_every", 1))
                        if every <= 0:
                            every = 1
                        if (l2_trial_counter % every) == 0:
                            try:
                                n_trades = int(metrics_trial.get("n_trades") or 0)
                            except Exception:
                                n_trades = 0

                            try:
                                trades_per_day = float(metrics_trial.get("trades_per_day"))
                            except Exception:
                                trades_per_day = float("nan")

                            try:
                                sharpe_mean = float(metrics_trial.get("sharpe_mean"))
                            except Exception:
                                sharpe_mean = float("nan")

                            try:
                                take_rate = float(metrics_trial.get("take_rate"))
                            except Exception:
                                take_rate = float("nan")

                            try:
                                cs_p50 = float(metrics_trial.get("consensus_p50"))
                            except Exception:
                                cs_p50 = float("nan")

                            try:
                                cs_p90 = float(metrics_trial.get("consensus_p90"))
                            except Exception:
                                cs_p90 = float("nan")

                            try:
                                trade_mean = float(metrics_trial.get("trade_mean_return"))
                            except Exception:
                                trade_mean = float("nan")

                            try:
                                trade_win = float(metrics_trial.get("trade_win_rate"))
                            except Exception:
                                trade_win = float("nan")

                            tprint_info(
                                "   [L2 committee trial] "
                                f"trial={l2_trial_counter}, utility={utility:.4f}, sharpe={sharpe_mean:.4f}, "
                                f"n_trades={n_trades}, tpd={trades_per_day:.2f}, "
                                f"thr={threshold:.3f}, q={cq}, thr_eff={metrics_trial.get('consensus_threshold_effective')}, "
                                f"w=({w_scalp:.3f},{w_swing:.3f},{w_trend:.3f}), "
                                f"take_rate={take_rate:.3f}, cs_p50={cs_p50:.3f}, cs_p90={cs_p90:.3f}, "
                                f"trade_mean={trade_mean:.4%}, win={trade_win:.2%}"
                            )
                except Exception:
                    pass

                return float(utility)

        else:
            def layer2_objective(trial_params: Dict[str, Any]) -> float:
                try:
                    metrics_trial = _compute_layer2_metrics(trial_params)
                    return float(metrics_trial.get("utility", -1.0))
                except Exception:
                    return -1.0

        meta_feature_cfg = config.get("meta_feature_engineering", {})
        volume_available = "volume" in market_data.columns

        # ------------------------------------------------------------------
        # OPTION: Use cached features from labeled_data artifact (Phase 3)
        # ------------------------------------------------------------------
        use_cached_features = bool(config.get("hpo_use_cached_features", False))
        cached_features_loaded = False
        
        if use_cached_features:
            try:
                # Attempt to load features from labeled_data artifact
                from src.artifacts.versioned_artifact_store import VersionedArtifactStore
                store = VersionedArtifactStore()
                labeled_data = store.get_artifact(
                    f"labeled_data_{symbol}_{timeframe}",
                    version="latest"
                )
                if labeled_data is not None and hasattr(labeled_data, 'columns'):
                    # Extract feature columns (exclude label/return columns)
                    exclude_cols = {
                        'binary_label', 'realized_return', 'target', 'target_long', 'target_short',
                        'meta_probability', 'meta_probability_ensemble', 'exit_reason', 'duration'
                    }
                    feature_cols = [c for c in labeled_data.columns if c not in exclude_cols]
                    if len(feature_cols) > 10:
                        meta_features_full = labeled_data[feature_cols].copy()
                        cached_features_loaded = True
                        tprint_success(f"✅ Loaded {len(feature_cols)} cached features from labeled_data artifact")
            except Exception as cache_exc:
                tprint_warning(f"⚠️ Failed to load cached features: {cache_exc}. Regenerating.")

        if not cached_features_loaded:
            # PRE-CALCULATE META-FEATURES ONCE (Performance Optimization)
            # Use baseline returns/labels as proxy. The goal is to get X features.
            # Note: If meta-features rely heavily on exact realized_return of the specific TBM,
            # this is an approximation. But for HPO speed, it is necessary.
            # Most features (technicals, regime, kalman) depend only on market_data/signals.
            tprint_info("🏗️ Layer 2: Pre-calculating meta-features with optimized Kalman params...")
        mf_config_opt = meta_feature_cfg.copy()
        try:
            hpo_use_full_feature_set = bool(config.get("hpo_use_full_feature_set", True))
            if hpo_use_full_feature_set:
                mf_config_opt["enable_feature_selection"] = False
                if "max_features" in mf_config_opt:
                    mf_config_opt.pop("max_features", None)
        except Exception:
            pass
        mf_config_opt['kalman_Q'] = best_kalman_params.get('kalman_Q', 1e-4)
        mf_config_opt['kalman_R'] = best_kalman_params.get('kalman_R', 0.01)

        # Generate dummy stop threshold for feature generation (won't affect independent features)
        dummy_stop_thr = (atr_frac * 1.0).astype(float).clip(lower=0.002)
        dummy_profit_thr = (atr_frac * 2.0).astype(float).clip(lower=0.008)

        _, meta_features_full, _, _ = build_meta_features_for_model(
            market_data=market_data,
            primary_signals=primary_signals,
            realized_returns=baseline_returns,
            binary_labels=binary_labels,
            event_durations=event_durations_raw,
            mfe_series=mfe_raw,
            mae_series=mae_raw,
            adaptive_stop_threshold=baseline_stop.reindex(market_data.index),
            horizon=12,
            volume_available=volume_available,
            meta_feature_cfg=mf_config_opt,
        )

        # ------------------------------------------------------------------
        # WEIGHTED PIPELINE: Add Kalman-based Features
        # ------------------------------------------------------------------
        # Uses the optimized Q and R from Stage 0 (RTS) in a CAUSAL Kalman Filter
        # for features that can be used in live trading.
        tprint_info("🏗️ Generating Kalman-based features (weighted pipeline)...")
        
        kalman_Q_opt = best_kalman_params.get('kalman_Q', 1e-4)
        kalman_R_opt = best_kalman_params.get('kalman_R', 0.01)
        
        try:
            kalman_features = generate_kalman_features(
                market_data=market_data,
                kalman_Q=kalman_Q_opt,
                kalman_R=kalman_R_opt,
            )
            
            # Merge Kalman features with existing meta features
            # Align indices and handle any missing data
            kalman_features_aligned = kalman_features.reindex(meta_features_full.index).fillna(0)
            
            # Add Kalman features to meta_features_full
            for col in kalman_features_aligned.columns:
                meta_features_full[col] = kalman_features_aligned[col]
            
            tprint_success(f"✅ Added {len(kalman_features.columns)} Kalman features")
        except Exception as kf_exc:
            tprint_warning(f"⚠️ Kalman feature generation failed: {kf_exc}. Continuing without Kalman features.")
        
        tprint_success(f"✅ Meta-features pre-calculated: {meta_features_full.shape[1]} columns")

        meta_features_full_raw = meta_features_full.copy()
        
        # ------------------------------------------------------------------
        # QUALITY-BASED FEATURE SELECTION (After Layer 0, Before HPO Loop)
        # ------------------------------------------------------------------
        # This solves the circular dependency: features are selected based on
        # unsupervised quality metrics (Signal/Noise ratio) rather than labels.
        #
        # Pipeline:
        # 1. Generate multi-horizon versions (Short/Medium/Long) for cross-timeframe
        # 2. Calculate Signal-to-Noise ratio for all features
        # 3. Reduce by correlation, keeping highest quality features
        #
        target_feature_count = int(config.get("target_feature_count", 70))
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
        
        # Custom horizon configuration (can be overridden in config)
        horizon_config = config.get("feature_horizon_config", {
            "Short": 5,    # ~1.25 hours at 15m (fast signals)
            "Medium": 20,  # ~5 hours at 15m (medium signals)
            "Long": 60,    # ~15 hours at 15m (slow signals)
        })
        
        tprint_info("🔬 Running De Prado feature selection pipeline...")
        try:
            meta_features_full, feature_quality_scores = select_features_with_quality(
                df_features=meta_features_full,
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
            
            # Store quality scores for potential later use
            self._feature_quality_scores = feature_quality_scores
            
            tprint_success(
                f"✅ Feature selection complete: {len(meta_features_full.columns)} features "
                f"(target={target_feature_count})"
            )

            # Persist feature selection results immediately
            try:
                ts_fs = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                fs_artifact_path = Path("outcomes") / f"hpo_feature_selection_{symbol}_{timeframe}_{ts_fs}.json"
                fs_payload = {
                    "selected_features": list(meta_features_full.columns),
                    "quality_scores": feature_quality_scores,
                    "target": target_feature_count,
                    "timestamp": ts_fs,
                }
                fs_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                with open(fs_artifact_path, "w") as f:
                    json.dump(fs_payload, f, indent=2, default=str)
                tprint_info(f"   💾 Saved feature selection stage to {fs_artifact_path}")
            except Exception as fs_save_exc:
                tprint_warning(f"   ⚠️ Failed to save feature selection artifact: {fs_save_exc}")
        except Exception as fs_exc:
            tprint_warning(f"⚠️ Feature selection failed: {fs_exc}. Using all features.")
            self._feature_quality_scores = {}
            meta_features_full = meta_features_full_raw

        if False:
            # --- LEGACY RETENTION FOR INTERFACE COMPATIBILITY ---
            # Construct dummy objects if needed by downstream
            # Use Swing Upper (Index 3) as the " Representative" return for downstream analysis if needed,
            # but the HPO was driven by the Weighted P&L.
            l2_returns_clean = committee_results[3]['returns'].reindex(event_idx).fillna(0.0)
            valid_idx = np.ones(len(pnl_series), dtype=bool) # All events valid
            l2_labels_clean = pd.Series(l2_binary_labels, index=event_idx)
            l2_t_events = event_idx
            # ----------------------------------------------------


            # B. DYNAMIC WEIGHT GENERATION
            # Construct t1 (end times) Series for compute_uniqueness
            # Map start timestamps to integer locations
            t0_locs = pd.Series(np.arange(len(market_data)), index=market_data.index)
            start_locs = t0_locs.loc[l2_t_events].values
            # Get durations for these specific events
            dur_vals = l2_durations.loc[l2_t_events].values.astype(int)
            end_locs = np.minimum(start_locs + dur_vals, len(market_data) - 1)
            t1_vals = market_data.index[end_locs]
            t1_series = pd.Series(t1_vals, index=l2_t_events)

            batch_consistency = full_consistency.reindex(l2_t_events).fillna(1.0).values
            batch_volatility = full_volatility.reindex(l2_t_events).fillna(0).values
            batch_uniqueness = compute_uniqueness(t1_series, market_index=market_data.index)

            sample_weights = generate_weights_per_label(
                returns=l2_returns_clean.values,
                t_events=l2_t_events,
                close_series=None,
                consistency_scores=batch_consistency,
                uniqueness_scores=batch_uniqueness.values,
                vol_proxy=batch_volatility,
                **best_weighting_params
            )

            try:
                if committee_weight_factor_series is not None:
                    cf = committee_weight_factor_series.reindex(l2_t_events).fillna(1.0).values.astype(float)
                    cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                    sample_weights = np.asarray(sample_weights, dtype=float) * cf
                    sw_mean = float(np.mean(sample_weights)) if sample_weights.size else 1.0
                    if np.isfinite(sw_mean) and sw_mean > 0:
                        sample_weights = sample_weights / sw_mean
            except Exception:
                pass

            # C. SUBSET META-FEATURES (Fast)
            X_trial = meta_features_full.loc[valid_idx].fillna(0)

            # D. FAST MODEL TRAINING WITH CV
            n_cv_folds = 5
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                cv_preds, folds_sharpe, mean_brier, mean_ece = _cross_val_predict_proba_and_fold_sharpes_weighted(
                    estimator=fast_model,
                    X=X_trial,
                    y=l2_labels_clean,
                    sample_weight=sample_weights,
                    n_splits=n_cv_folds,
                    returns=l2_returns_clean.values.astype(float),
                    direction=direction,
                    prob_thr=0.5,
                    use_calibration=True,
                    enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                    ev_margin=config.get("ev_margin", 0.0),
                )
            except Exception:
                return -1.0

            # E. COMPUTE AUC (for trapezoidal gate)
            try:
                mean_auc = roc_auc_score(l2_labels_clean.values, cv_preds)
            except Exception:
                mean_auc = 0.5

            y_true_arr = l2_labels_clean.values.astype(float)
            returns_arr = l2_returns_clean.values.astype(float)

            # G. COMPUTE TRADES PER DAY (from predicted trades, not event count)
            # Use the same threshold as sizing to avoid density gate being constant.
            try:
                pred_trade_mask = np.asarray(cv_preds, dtype=float) >= 0.5
                n_pred_trades = int(np.sum(pred_trade_mask))
            except Exception:
                n_pred_trades = int(len(l2_returns_clean))
            trades_per_day = float(n_pred_trades) / float(max(days_span, 1))

            # H. CALCULATE UTILITY (Trapezoidal Gate + Stability)
            utility = calculate_hpo_utility(
                folds_sharpe=folds_sharpe,
                auc=mean_auc,
                trades_per_day=trades_per_day,
                lambda_vol=1.2,   # Penalty for Sharpe volatility across folds
                w_auc=0.8,        # Slightly looser AUC weighting
                w_den=0.5,        # Moderate density weight
                calibration_brier=mean_brier,
                calibration_ece=mean_ece,
                w_cal=0.0,
            )

            try:
                utility, q_details = _apply_hpo_quality_penalty(
                    utility=utility,
                    returns=l2_returns_clean.values,
                    labels=l2_labels_clean.values,
                    exit_reasons=l2_exit_reasons.loc[l2_t_events].values if l2_exit_reasons is not None else None,
                    durations=l2_durations.loc[l2_t_events].values if l2_durations is not None else None,
                    horizon=12,
                    tx_cost=float(DEFAULT_TRANSACTION_COST),
                    config=config,
                )
            except Exception:
                q_details = {}

            # Log objective components for traceability
            try:
                tprint_info(
                    "   [L2 objective] "
                    f"utility={utility:.4f}, auc={mean_auc:.4f}, "
                    f"trades_per_day={trades_per_day:.2f}, "
                    f"folds_sharpe_mean={float(np.mean(folds_sharpe)):.4f}, "
                    f"folds_sharpe_std={float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe)>1 else 0.0:.4f}"
                )
            except Exception:
                pass

            return utility

        def _compute_layer2_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
            """Single-shot computation of Layer 2 metrics for reporting."""
            trail_dist = float(params.get("trail_distance_atr_mult", 0.0))

            prof_thr = fixed_layer2_profit_thr
            stop_thr = fixed_layer2_stop_thr

            (
                l2_returns,
                l2_labels,
                l2_exit_reasons,
                l2_durations,
                l2_mfe,
                l2_mae,
                _, _
            ) = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=prof_thr,
                stop_threshold=stop_thr,
                horizon=12,
                transaction_cost=DEFAULT_TRANSACTION_COST,
                min_event_spacing=2,
                trail_distance_atr_mult=trail_dist,
                atr_series=atr_series,
            )

            valid_idx = ~l2_labels.isna()
            if valid_idx.sum() < 50:
                return {"valid_events": int(valid_idx.sum()), "utility": -1.0}

            l2_t_events = l2_returns.index[valid_idx]
            l2_returns_clean = l2_returns[valid_idx]
            l2_labels_clean = l2_labels[valid_idx]

            t0_locs = pd.Series(np.arange(len(market_data)), index=market_data.index)
            start_locs = t0_locs.loc[l2_t_events].values
            dur_vals = l2_durations.loc[l2_t_events].values.astype(int)
            end_locs = np.minimum(start_locs + dur_vals, len(market_data) - 1)
            t1_vals = market_data.index[end_locs]
            t1_series = pd.Series(t1_vals, index=l2_t_events)

            batch_consistency = full_consistency.reindex(l2_t_events).fillna(1.0).values
            batch_volatility = full_volatility.reindex(l2_t_events).fillna(0).values
            batch_uniqueness = compute_uniqueness(t1_series, market_index=market_data.index)

            sample_weights = generate_weights_per_label(
                returns=l2_returns_clean.values,
                t_events=l2_t_events,
                close_series=None,
                consistency_scores=batch_consistency,
                uniqueness_scores=batch_uniqueness.values,
                vol_proxy=batch_volatility,
                **best_weighting_params
            )

            try:
                if committee_weight_factor_series is not None:
                    cf = committee_weight_factor_series.reindex(l2_t_events).fillna(1.0).values.astype(float)
                    cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                    sample_weights = np.asarray(sample_weights, dtype=float) * cf
                    sw_mean = float(np.mean(sample_weights)) if sample_weights.size else 1.0
                    if np.isfinite(sw_mean) and sw_mean > 0:
                        sample_weights = sample_weights / sw_mean
            except Exception:
                pass

            X_trial = meta_features_full.loc[valid_idx].fillna(0)
            n_cv_folds = 5
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                cv_preds, folds_sharpe, mean_brier, mean_ece = _cross_val_predict_proba_and_fold_sharpes_weighted(
                    estimator=fast_model,
                    X=X_trial,
                    y=l2_labels_clean,
                    sample_weight=sample_weights,
                    n_splits=n_cv_folds,
                    returns=l2_returns_clean.values.astype(float),
                    direction=direction,
                    prob_thr=0.5,
                    use_calibration=True,
                    enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                    ev_margin=config.get("ev_margin", 0.0),
                )
            except Exception:
                return {"valid_events": int(valid_idx.sum()), "utility": -1.0}

            try:
                mean_auc = roc_auc_score(l2_labels_clean.values, cv_preds)
            except Exception:
                mean_auc = 0.5

            y_true_arr = l2_labels_clean.values.astype(float)
            returns_arr = l2_returns_clean.values.astype(float)

            # Use predicted trade frequency (not raw event count) so density gating
            # reflects actual model aggressiveness.
            try:
                pred_trade_mask = np.asarray(cv_preds, dtype=float) >= 0.5
                n_pred_trades = int(np.sum(pred_trade_mask))
            except Exception:
                n_pred_trades = int(len(l2_returns_clean))
            trades_per_day = float(n_pred_trades) / float(max(days_span, 1))
            lambda_vol = 1.2
            w_auc = 1.0
            w_den = 0.5

            per_fold_metrics = []
            try:
                per_fold_metrics = _compute_fold_metrics_from_oof(
                    X=X_trial,
                    y_true=y_true_arr,
                    probs=np.asarray(cv_preds, dtype=float),
                    returns=np.asarray(returns_arr, dtype=float),
                    threshold=0.5,
                    days_span=float(days_span),
                    transaction_cost=0.0,
                )
            except Exception:
                per_fold_metrics = []

            per_regime_metrics: Dict[str, Any] = {}
            try:
                regime_labels = _build_event_regime_labels(
                    market_data=market_data,
                    event_index=l2_t_events,
                    config=config,
                )
                per_regime_metrics = {
                    "volatility": _compute_metrics_by_regime(
                        y_true=y_true_arr,
                        probs=np.asarray(cv_preds, dtype=float),
                        returns=np.asarray(returns_arr, dtype=float),
                        base_thr=0.5,
                        transaction_cost=0.0,
                        regime_labels=regime_labels.get("volatility_regime"),
                        days_span=float(days_span),
                    ),
                    "trend": _compute_metrics_by_regime(
                        y_true=y_true_arr,
                        probs=np.asarray(cv_preds, dtype=float),
                        returns=np.asarray(returns_arr, dtype=float),
                        base_thr=0.5,
                        transaction_cost=0.0,
                        regime_labels=regime_labels.get("trend_regime"),
                        days_span=float(days_span),
                    ),
                    "combined": _compute_metrics_by_regime(
                        y_true=y_true_arr,
                        probs=np.asarray(cv_preds, dtype=float),
                        returns=np.asarray(returns_arr, dtype=float),
                        base_thr=0.5,
                        transaction_cost=0.0,
                        regime_labels=regime_labels.get("combined_regime"),
                        days_span=float(days_span),
                    ),
                }
            except Exception:
                per_regime_metrics = {}

            avg_sharpe = float(np.mean(folds_sharpe))
            vol_sharpe = float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe) > 1 else 0.0
            base_score = avg_sharpe - (lambda_vol * vol_sharpe)
            try:
                base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
            except Exception:
                base_norm = 0.0
            if not np.isfinite(base_norm):
                base_norm = 0.0
            phi_auc = trapezoidal_gate(mean_auc, lower=0.52, sweet_spot=(0.56, 0.66), upper=0.72)
            phi_density = trapezoidal_gate(
                float(trades_per_day),
                lower=0.5,
                sweet_spot=(1.5, 5.0),
                upper=8.0,
            )
            modifier = (phi_auc ** w_auc) * (phi_density ** w_den)

            utility = calculate_hpo_utility(
                folds_sharpe=folds_sharpe,
                auc=mean_auc,
                trades_per_day=trades_per_day,
                lambda_vol=lambda_vol,
                w_auc=w_auc,
                w_den=w_den,
                calibration_brier=mean_brier,
                calibration_ece=mean_ece,
                w_cal=0.0,
            )

            q_details: Dict[str, Any] = {}
            try:
                utility, q_details = _apply_hpo_quality_penalty(
                    utility=float(utility),
                    returns=returns_arr,
                    labels=y_true_arr,
                    exit_reasons=l2_exit_reasons.loc[l2_t_events].values if l2_exit_reasons is not None else None,
                    durations=l2_durations.loc[l2_t_events].values if l2_durations is not None else None,
                    horizon=12,
                    tx_cost=float(DEFAULT_TRANSACTION_COST),
                    config=config,
                )
            except Exception:
                pass

            return {
                "valid_events": int(valid_idx.sum()),
                "utility": float(utility),
                "quality_penalty": q_details,
                "auc": float(mean_auc),
                "trades_per_day": float(trades_per_day),
                "calibration_brier": float(mean_brier) if mean_brier is not None else None,
                "calibration_ece": float(mean_ece) if mean_ece is not None else None,
                "sharpe_mean": float(np.mean(folds_sharpe)),
                "sharpe_std": float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe) > 1 else 0.0,
                "sharpe_min": float(np.min(folds_sharpe)),
                "sharpe_max": float(np.max(folds_sharpe)),
                "folds_sharpe_values": [float(v) for v in folds_sharpe.tolist()] if isinstance(folds_sharpe, np.ndarray) else [],
                "per_fold_metrics": per_fold_metrics,
                "per_regime_metrics": per_regime_metrics,
                "lambda_vol": lambda_vol,
                "w_auc": w_auc,
                "w_den": w_den,
                "avg_sharpe": avg_sharpe,
                "vol_sharpe": vol_sharpe,
                "base_score": float(base_score),
                "base_norm": float(base_norm) if np.isfinite(base_norm) else float("nan"),
                "phi_auc": float(phi_auc),
                "phi_density": float(phi_density),
                "modifier": float(modifier),
            }

        def _compute_layer2_metrics_committee(params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                w_scalp = float(params.get("w_scalp", 0.0))
                w_swing = float(params.get("w_swing", 0.0))
                w_trend = float(params.get("w_trend", 0.0))
                threshold = float(params.get("consensus_threshold", 0.5))
                consensus_quantile = params.get("consensus_quantile", None)
                consensus_quantile = float(consensus_quantile) if consensus_quantile is not None else None
            except Exception:
                return {"valid_events": int(len(event_idx)), "utility": -1.0}

            weights_vec = np.array(
                [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend],
                dtype=float,
            )
            total_weight = float(np.sum(weights_vec)) + 1e-8
            if (not np.isfinite(total_weight)) or (total_weight <= 1e-8):
                return {"valid_events": int(len(event_idx)), "utility": -1.0}

            consensus_score = label_matrix_values.dot(weights_vec) / total_weight
            try:
                cs = np.asarray(consensus_score, dtype=float)
                cs = cs[np.isfinite(cs)]
                consensus_mean = float(np.mean(cs)) if cs.size > 0 else float("nan")
                consensus_std = float(np.std(cs, ddof=1)) if cs.size > 1 else 0.0
                consensus_p10 = float(np.quantile(cs, 0.10)) if cs.size > 0 else float("nan")
                consensus_p50 = float(np.quantile(cs, 0.50)) if cs.size > 0 else float("nan")
                consensus_p90 = float(np.quantile(cs, 0.90)) if cs.size > 0 else float("nan")
                consensus_p99 = float(np.quantile(cs, 0.99)) if cs.size > 0 else float("nan")
                consensus_min = float(np.min(cs)) if cs.size > 0 else float("nan")
                consensus_max = float(np.max(cs)) if cs.size > 0 else float("nan")
                frac_pos = float(np.mean(cs > 0.0)) if cs.size > 0 else float("nan")
                frac_neg = float(np.mean(cs < 0.0)) if cs.size > 0 else float("nan")
            except Exception:
                consensus_mean = float("nan")
                consensus_std = float("nan")
                consensus_p10 = float("nan")
                consensus_p50 = float("nan")
                consensus_p90 = float("nan")
                consensus_p99 = float("nan")
                consensus_min = float("nan")
                consensus_max = float("nan")
                frac_pos = float("nan")
                frac_neg = float("nan")

            thr_effective = float(threshold)
            try:
                cs_full = np.asarray(consensus_score, dtype=float)
                cs_full = np.where(np.isfinite(cs_full), cs_full, -np.inf)
                if consensus_quantile is not None and np.isfinite(consensus_quantile):
                    q = float(np.clip(consensus_quantile, 0.0, 0.999999))
                    n = int(cs_full.size)
                    if n > 0:
                        k = int(np.ceil((1.0 - q) * float(n)))
                        k = int(np.clip(k, 1, n))
                        top_idx = np.argpartition(cs_full, n - k)[n - k :]
                        take_mask = np.zeros(n, dtype=bool)
                        take_mask[top_idx] = True
                        try:
                            thr_effective = float(np.min(cs_full[top_idx])) if top_idx.size > 0 else float(threshold)
                        except Exception:
                            thr_effective = float(threshold)
                    else:
                        take_mask = np.zeros(0, dtype=bool)
                else:
                    take_mask = cs_full > float(threshold)
            except Exception:
                take_mask = np.asarray(consensus_score, dtype=float) > float(threshold)

            n_trades = int(np.sum(take_mask))
            take_rate = float(n_trades) / float(len(event_idx)) if len(event_idx) > 0 else 0.0
            trades_per_day = float(n_trades) / float(max(days_span, 1))

            committee_expert_stats: Dict[str, Any] = {}
            sanity_checks: Dict[str, Any] = {"violations": [], "debug_tables": {}}
            try:
                ev_idx = pd.DatetimeIndex(event_idx)
                consensus_arr = np.asarray(consensus_score, dtype=float)
                for j, name in enumerate(list(committee_names)):
                    lbl_col = np.asarray(label_matrix_values[:, j], dtype=float)
                    ret_col = np.asarray(returns_matrix_values[:, j], dtype=float)
                    fired = lbl_col != 0.0
                    n_fired = int(np.sum(fired))
                    pos_mask = lbl_col > 0.0
                    neg_mask = lbl_col < 0.0
                    out = {
                        "n_events": int(lbl_col.size),
                        "n_fired": int(n_fired),
                        "frac_fired": float(n_fired) / float(max(int(lbl_col.size), 1)),
                        "frac_pos": float(np.mean(pos_mask)) if lbl_col.size else 0.0,
                        "frac_neg": float(np.mean(neg_mask)) if lbl_col.size else 0.0,
                    }
                    if n_fired > 0:
                        r = ret_col[fired]
                        r = r[np.isfinite(r)]
                        out["mean_return_on_fired"] = float(np.mean(r)) if r.size else 0.0
                        out["win_rate_on_fired"] = float(np.mean(r > 0.0)) if r.size else 0.0
                    else:
                        out["mean_return_on_fired"] = 0.0
                        out["win_rate_on_fired"] = 0.0

                    r_pos = ret_col[pos_mask]
                    r_pos = r_pos[np.isfinite(r_pos)]
                    out["n_pos"] = int(np.sum(pos_mask))
                    out["mean_return_on_pos"] = float(np.mean(r_pos)) if r_pos.size else 0.0
                    out["win_rate_on_pos"] = float(np.mean(r_pos > 0.0)) if r_pos.size else 0.0

                    r_neg = ret_col[neg_mask]
                    r_neg = r_neg[np.isfinite(r_neg)]
                    out["n_neg"] = int(np.sum(neg_mask))
                    out["mean_return_on_neg"] = float(np.mean(r_neg)) if r_neg.size else 0.0
                    out["win_rate_on_neg"] = float(np.mean(r_neg > 0.0)) if r_neg.size else 0.0

                    try:
                        n_pos = int(out.get("n_pos", 0))
                        n_neg = int(out.get("n_neg", 0))
                        mean_pos = float(out.get("mean_return_on_pos", 0.0))
                        mean_neg = float(out.get("mean_return_on_neg", 0.0))
                        if n_pos >= 20 and n_neg >= 20 and np.isfinite(mean_pos) and np.isfinite(mean_neg) and mean_pos < mean_neg:
                            sanity_checks["violations"].append(
                                {
                                    "expert": str(name),
                                    "n_pos": int(n_pos),
                                    "n_neg": int(n_neg),
                                    "mean_return_on_pos": float(mean_pos),
                                    "mean_return_on_neg": float(mean_neg),
                                }
                            )

                            dbg_n = int(config.get("layer2_sanity_debug_rows", 15))
                            bad_idx = np.where((lbl_col > 0.0) & np.isfinite(ret_col) & (ret_col < 0.0))[0]
                            if bad_idx.size > 0:
                                dbg_df = pd.DataFrame(
                                    {
                                        "timestamp": ev_idx[bad_idx].astype(str),
                                        "consensus": consensus_arr[bad_idx].astype(float),
                                        "realized_return": ret_col[bad_idx].astype(float),
                                        "costs": float(DEFAULT_TRANSACTION_COST),
                                    }
                                )
                                dbg_df = dbg_df.sort_values("realized_return", ascending=True).head(int(max(1, dbg_n)))
                                sanity_checks["debug_tables"][str(name)] = dbg_df.to_dict(orient="records")
                                try:
                                    tprint_warning(
                                        f"⚠️ Layer2 sanity: inverted pos/neg returns for expert={str(name)} "
                                        f"mean_pos={mean_pos:.6f} < mean_neg={mean_neg:.6f}. Sample bad pos trades:\n"
                                        + dbg_df.to_string(index=False)
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    committee_expert_stats[str(name)] = out
            except Exception:
                committee_expert_stats = {}

            committee_overlap: Dict[str, Any] = {}
            try:
                n_exp = int(label_matrix_values.shape[1])
                for i in range(n_exp):
                    for j in range(i + 1, n_exp):
                        name_i = str(list(committee_names)[i])
                        name_j = str(list(committee_names)[j])
                        li = np.asarray(label_matrix_values[:, i], dtype=float)
                        lj = np.asarray(label_matrix_values[:, j], dtype=float)
                        fi = li != 0.0
                        fj = lj != 0.0
                        inter = fi & fj
                        union = fi | fj
                        n_inter = int(np.sum(inter))
                        n_union = int(np.sum(union))
                        jacc = float(n_inter) / float(max(n_union, 1))
                        sign_agree = float(np.mean(np.sign(li[inter]) == np.sign(lj[inter]))) if n_inter > 0 else 0.0
                        committee_overlap[f"{name_i}__{name_j}"] = {
                            "n_intersection": int(n_inter),
                            "n_union": int(n_union),
                            "jaccard": float(jacc),
                            "sign_agreement": float(sign_agree),
                        }
            except Exception:
                committee_overlap = {}

            committee_drivers: Dict[str, Any] = {}
            try:
                take_mask_arr = np.asarray(take_mask, dtype=bool)
                for j, name in enumerate(list(committee_names)):
                    lbl_col = np.asarray(label_matrix_values[:, j], dtype=float)
                    pos_take = int(np.sum((lbl_col > 0.0) & take_mask_arr))
                    neg_take = int(np.sum((lbl_col < 0.0) & take_mask_arr))
                    fired_take = int(np.sum((lbl_col != 0.0) & take_mask_arr))
                    committee_drivers[str(name)] = {
                        "pos_on_taken": int(pos_take),
                        "neg_on_taken": int(neg_take),
                        "fired_on_taken": int(fired_take),
                        "share_fired_on_taken": float(fired_take) / float(max(n_trades, 1)),
                    }
            except Exception:
                committee_drivers = {}

            try:
                ret_mat = np.asarray(returns_matrix_values, dtype=float)
                finite_mask = np.isfinite(ret_mat)

                w_row = np.asarray(weights_vec, dtype=float).reshape(1, -1)
                denom = np.sum(finite_mask * w_row, axis=1).astype(float) + 1e-8
                numer = np.sum(np.where(finite_mask, ret_mat, 0.0) * w_row, axis=1).astype(float)
                weighted_returns = numer / denom
                weighted_returns = np.where(np.isfinite(weighted_returns), weighted_returns, 0.0)
            except Exception:
                weighted_returns = np.zeros(int(len(event_idx)), dtype=float)

            trade_returns = np.asarray(weighted_returns, dtype=float)[np.asarray(take_mask, dtype=bool)]
            trade_returns = trade_returns[np.isfinite(trade_returns)]

            per_fold_metrics: List[Dict[str, Any]] = []
            fold_sharpes: List[float] = []
            fold_aucs: List[float] = []
            try:
                cv_local = TimeSeriesSplit(n_splits=5)
                for fold_idx, (_, te_idx) in enumerate(cv_local.split(np.arange(len(event_idx)))):
                    te_idx = np.asarray(te_idx, dtype=int)
                    if te_idx.size <= 0:
                        continue
                    tr_mask = np.asarray(take_mask, dtype=bool)[te_idx]
                    tr_returns = np.asarray(weighted_returns, dtype=float)[te_idx]
                    tr_returns = tr_returns[np.isfinite(tr_returns)]
                    n_trades_fold = int(np.sum(tr_mask))

                    fold_auc = 0.5
                    try:
                        score_fold = np.asarray(consensus_score, dtype=float)[te_idx]
                        ret_fold = np.asarray(weighted_returns, dtype=float)[te_idx]
                        mm_auc = np.isfinite(score_fold) & np.isfinite(ret_fold)
                        if int(np.sum(mm_auc)) >= 20:
                            y_auc = (ret_fold[mm_auc] > 0.0).astype(int)
                            if int(np.unique(y_auc).size) >= 2:
                                fold_auc = float(roc_auc_score(y_auc, score_fold[mm_auc]))
                    except Exception:
                        fold_auc = 0.5

                    days_span_fold = 1.0
                    try:
                        idx_fold = pd.DatetimeIndex(event_idx[te_idx])
                        if len(idx_fold) >= 2:
                            days_span_fold = max(
                                1.0,
                                float((idx_fold.max() - idx_fold.min()).total_seconds() / 86400.0),
                            )
                    except Exception:
                        days_span_fold = float(max(days_span, 1))

                    if n_trades_fold <= 0:
                        per_fold_metrics.append(
                            {
                                "fold": int(fold_idx),
                                "auc": float(fold_auc),
                                "n_test": int(len(te_idx)),
                                "n_trades": 0,
                                "trades_per_day": 0.0,
                                "mean_return": 0.0,
                                "net_pnl_per_trade": 0.0,
                                "win_rate": 0.0,
                                "sharpe": 0.0,
                            }
                        )
                        fold_aucs.append(float(fold_auc))
                        fold_sharpes.append(0.0)
                        continue

                    fold_trade_returns = np.asarray(weighted_returns, dtype=float)[te_idx][tr_mask]
                    fold_trade_returns = fold_trade_returns[np.isfinite(fold_trade_returns)]
                    mean_ret = float(np.mean(fold_trade_returns)) if fold_trade_returns.size > 0 else 0.0
                    win_rate = float(np.mean(fold_trade_returns > 0)) if fold_trade_returns.size > 0 else 0.0
                    sharpe_fold = 0.0
                    try:
                        idx_te = pd.DatetimeIndex(event_idx[te_idx])
                        idx_tr = idx_te[np.asarray(tr_mask, dtype=bool)]
                        if fold_trade_returns.size > 0 and len(idx_tr) == int(fold_trade_returns.size):
                            day_index = pd.date_range(
                                start=idx_te.min().normalize(),
                                end=idx_te.max().normalize(),
                                freq="D",
                            )
                            daily_pnl = pd.Series(fold_trade_returns, index=idx_tr).groupby(idx_tr.normalize()).sum()
                            daily_pnl = daily_pnl.reindex(day_index, fill_value=0.0)

                            daily_log = np.log1p(daily_pnl.astype(float).values)
                            daily_log = daily_log[np.isfinite(daily_log)]
                            if int(daily_log.size) > 1:
                                mu = float(np.mean(daily_log))
                                sd = float(np.std(daily_log, ddof=1))
                                if sd > 1e-12:
                                    sharpe_fold = float(np.clip(mu / sd * np.sqrt(365.0), -20.0, 20.0))
                    except Exception:
                        sharpe_fold = 0.0

                    per_fold_metrics.append(
                        {
                            "fold": int(fold_idx),
                            "auc": float(fold_auc),
                            "n_test": int(len(te_idx)),
                            "n_trades": int(n_trades_fold),
                            "trades_per_day": float(n_trades_fold) / float(max(days_span_fold, 1.0)),
                            "mean_return": float(mean_ret),
                            "net_pnl_per_trade": float(mean_ret),
                            "win_rate": float(win_rate),
                            "sharpe": float(sharpe_fold),
                        }
                    )
                    fold_aucs.append(float(fold_auc))
                    fold_sharpes.append(float(sharpe_fold))
            except Exception:
                per_fold_metrics = []
                fold_sharpes = []
                fold_aucs = []

            per_regime_metrics: Dict[str, Any] = {}
            try:
                regime_labels = _build_event_regime_labels(
                    market_data=market_data,
                    event_index=event_idx,
                    config=config,
                )

                def _by_regime(reg: pd.Series) -> Dict[str, Any]:
                    out: Dict[str, Any] = {}
                    if reg is None or reg.empty:
                        return out
                    lab = reg.astype(object)
                    for rv in pd.unique(lab.dropna()):
                        rm = (lab == rv).to_numpy(dtype=bool)
                        n_events_r = int(np.sum(rm))
                        if n_events_r < 20:
                            continue
                        tm = np.asarray(take_mask, dtype=bool) & rm
                        n_trades_r = int(np.sum(tm))
                        if n_trades_r <= 0:
                            out[str(rv)] = {"n_events": n_events_r, "n_trades": 0}
                            continue
                        rvals = np.asarray(weighted_returns, dtype=float)[tm]
                        rvals = rvals[np.isfinite(rvals)]
                        if rvals.size <= 0:
                            out[str(rv)] = {"n_events": n_events_r, "n_trades": 0}
                            continue
                        mean_r = float(np.mean(rvals))
                        win_r = float(np.mean(rvals > 0.0))
                        sharpe_r = 0.0
                        try:
                            idx_r = pd.DatetimeIndex(event_idx[tm])
                            if int(rvals.size) > 0 and int(idx_r.size) == int(rvals.size):
                                day_index_r = pd.date_range(
                                    start=idx_r.min().normalize(),
                                    end=idx_r.max().normalize(),
                                    freq="D",
                                )
                                daily_pnl_r = pd.Series(rvals, index=idx_r).groupby(idx_r.normalize()).sum()
                                daily_pnl_r = daily_pnl_r.reindex(day_index_r, fill_value=0.0)

                                daily_log_r = np.log1p(daily_pnl_r.astype(float).values)
                                daily_log_r = daily_log_r[np.isfinite(daily_log_r)]
                                if int(daily_log_r.size) > 1:
                                    mu_r = float(np.mean(daily_log_r))
                                    sd_r = float(np.std(daily_log_r, ddof=1))
                                    if sd_r > 1e-12:
                                        sharpe_r = mu_r / sd_r * np.sqrt(365.0)
                        except Exception:
                            sharpe_r = 0.0
                        out[str(rv)] = {
                            "n_events": int(n_events_r),
                            "n_trades": int(n_trades_r),
                            "trades_per_day": float(n_trades_r) / float(max(days_span, 1.0)),
                            "mean_return": float(mean_r),
                            "net_pnl_per_trade": float(mean_r),
                            "win_rate": float(win_r),
                            "sharpe": float(sharpe_r),
                        }
                    return out

                per_regime_metrics = {
                    "volatility": _by_regime(regime_labels.get("volatility_regime")),
                    "trend": _by_regime(regime_labels.get("trend_regime")),
                    "combined": _by_regime(regime_labels.get("combined_regime")),
                }
            except Exception:
                per_regime_metrics = {}

            sharpe = 0.0
            try:
                tm_all = np.asarray(take_mask, dtype=bool)
                wr_all = np.asarray(weighted_returns, dtype=float)
                tm_fin = tm_all & np.isfinite(wr_all)
                trade_idx_all = pd.DatetimeIndex(event_idx[tm_fin])
                trade_ret_all = wr_all[tm_fin]
                if int(trade_ret_all.size) > 0 and int(trade_idx_all.size) == int(trade_ret_all.size):
                    day_index_all = pd.date_range(
                        start=trade_idx_all.min().normalize(),
                        end=trade_idx_all.max().normalize(),
                        freq="D",
                    )
                    daily_pnl_all = pd.Series(trade_ret_all, index=trade_idx_all).groupby(trade_idx_all.normalize()).sum()
                    daily_pnl_all = daily_pnl_all.reindex(day_index_all, fill_value=0.0)

                    daily_log_all = np.log1p(daily_pnl_all.astype(float).values)
                    daily_log_all = daily_log_all[np.isfinite(daily_log_all)]
                    if int(daily_log_all.size) > 1:
                        mu_all = float(np.mean(daily_log_all))
                        sd_all = float(np.std(daily_log_all, ddof=1))
                        if sd_all > 1e-12:
                            sharpe = float(mu_all / sd_all * np.sqrt(365.0))
            except Exception:
                sharpe = 0.0
            sharpe = float(np.clip(float(sharpe), -20.0, 20.0))

            utility = float(sharpe) * float(np.log1p(n_trades)) if n_trades >= 50 else -1.0
            if n_trades >= 50 and n_trades < 100:
                utility = float(utility) * (float(n_trades) / 100.0)
            if not np.isfinite(float(utility)):
                utility = -1.0

            try:
                trade_mean = float(np.mean(trade_returns)) if trade_returns.size > 0 else 0.0
                trade_win_rate = float(np.mean(trade_returns > 0.0)) if trade_returns.size > 0 else 0.0
            except Exception:
                trade_mean = 0.0
                trade_win_rate = 0.0

            auc_val = 0.5
            try:
                score_auc = np.asarray(consensus_score, dtype=float)
                wr = np.asarray(weighted_returns, dtype=float)

                m = np.isfinite(score_auc) & np.isfinite(wr)
                if int(np.sum(m)) >= 20:
                    y_true_auc = (wr[m] > 0.0).astype(int)
                    if int(np.unique(y_true_auc).size) >= 2:
                        auc_val = float(roc_auc_score(y_true_auc, score_auc[m]))
                    else:
                        auc_val = 0.5
                else:
                    auc_val = 0.5
            except Exception:
                auc_val = 0.5

            mean_auc = float(auc_val)
            try:
                fau = np.asarray(fold_aucs, dtype=float)
                fau = fau[np.isfinite(fau)]
                if fau.size > 0:
                    mean_auc = float(np.mean(fau))
            except Exception:
                pass

            folds_sharpe_arr = np.asarray(fold_sharpes, dtype=float)
            folds_sharpe_arr = folds_sharpe_arr[np.isfinite(folds_sharpe_arr)]
            if folds_sharpe_arr.size <= 0:
                folds_sharpe_arr = np.asarray([float(sharpe)], dtype=float)

            lambda_vol = 1.2
            w_auc = 1.0
            w_den = 0.5

            avg_sharpe = float(np.mean(folds_sharpe_arr))
            vol_sharpe = float(np.std(folds_sharpe_arr, ddof=1)) if folds_sharpe_arr.size > 1 else 0.0
            base_score = avg_sharpe - (lambda_vol * vol_sharpe)
            try:
                base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
            except Exception:
                base_norm = 0.0
            if not np.isfinite(base_norm):
                base_norm = 0.0

            phi_auc = trapezoidal_gate(mean_auc, lower=0.52, sweet_spot=(0.56, 0.66), upper=0.72)
            phi_density = trapezoidal_gate(
                float(trades_per_day),
                lower=0.5,
                sweet_spot=(1.5, 5.0),
                upper=8.0,
            )
            try:
                modifier = float((phi_auc ** w_auc) * (phi_density ** w_den))
            except Exception:
                modifier = 0.0
            if not np.isfinite(modifier):
                modifier = 0.0

            return {
                "valid_events": int(len(event_idx)),
                "utility": float(utility),
                "auc": float(mean_auc) if np.isfinite(float(mean_auc)) else 0.5,
                "trades_per_day": float(trades_per_day),
                "calibration_brier": None,
                "calibration_ece": None,
                "sharpe_mean": float(np.mean(folds_sharpe_arr)),
                "sharpe_std": float(np.std(folds_sharpe_arr, ddof=1)) if folds_sharpe_arr.size > 1 else 0.0,
                "sharpe_min": float(np.min(folds_sharpe_arr)),
                "sharpe_max": float(np.max(folds_sharpe_arr)),
                "folds_sharpe_values": [float(v) for v in folds_sharpe_arr.tolist()],
                "per_fold_metrics": per_fold_metrics,
                "per_regime_metrics": per_regime_metrics,
                "n_trades": int(n_trades),
                "trade_mean_return": float(trade_mean),
                "trade_win_rate": float(trade_win_rate),
                "take_rate": float(take_rate),
                "consensus_mean": float(consensus_mean),
                "consensus_std": float(consensus_std),
                "consensus_p10": float(consensus_p10),
                "consensus_p50": float(consensus_p50),
                "consensus_p90": float(consensus_p90),
                "consensus_p99": float(consensus_p99),
                "consensus_min": float(consensus_min),
                "consensus_max": float(consensus_max),
                "consensus_frac_pos": float(frac_pos),
                "consensus_frac_neg": float(frac_neg),
                "consensus_threshold_effective": float(thr_effective),
                "consensus_quantile": float(consensus_quantile) if consensus_quantile is not None else None,
                "committee_expert_stats": committee_expert_stats,
                "sanity_checks": sanity_checks,
                "committee_overlap": committee_overlap,
                "committee_drivers": committee_drivers,
                "lambda_vol": float(lambda_vol),
                "w_auc": float(w_auc),
                "w_den": float(w_den),
                "avg_sharpe": float(avg_sharpe),
                "vol_sharpe": float(vol_sharpe),
                "base_score": float(base_score),
                "base_norm": float(base_norm),
                "phi_auc": float(phi_auc),
                "phi_density": float(phi_density),
                "modifier": float(modifier),
            }

        layer2_loaded_from: Optional[str] = None
        if stage_rank["layer2"] < start_rank:
            loaded_params, loaded_path = _load_stage_best_params("layer2")
            best_trading_params = dict(loaded_params or {})
            layer2_loaded_from = str(loaded_path) if loaded_path is not None else None
            l2_result = {"best_params": dict(best_trading_params), "best_value": None, "history": []}
            best_l2_score = float("nan")
            tprint_info(
                f"♻️ Layer 2 skipped (start_at={start_at_canonical}); loaded best params from {layer2_loaded_from}"
            )
        else:
            l2_optimizer = BayesianTPEOptimizer(
                config=OptimizationConfig(
                    n_trials=20,  # was 2; expand search
                    execution_mode="full",
                    direction="maximize",
                    seed=42,
                    enable_staged_optimization=False,
                    enable_adaptive_grid_refinement=False,
                    enable_adaptive_optimization=False,
                    enable_vectorbt_optimization=False,
                    enable_hardware_optimization=False,
                    n_startup_trials=5,
                    tpe_trials=20,
                )
            )
            l2_result = l2_optimizer.optimize(objective=layer2_objective, search_space=layer2_search_space)
            best_trading_params = l2_result.get("best_params", {})
            best_l2_score = l2_result.get("best_value", 0.0)
        ts_l2 = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        try:
            if layer2_mode == "committee":
                l2_metrics = _compute_layer2_metrics_committee(best_trading_params)
            else:
                l2_metrics = _compute_layer2_metrics(best_trading_params)
            try:
                if isinstance(l2_metrics, dict):
                    l2_metrics["layer2_mode"] = layer2_mode
            except Exception:
                pass
            tprint_info(
                "   Layer 2 metrics: "
                f"utility={l2_metrics.get('utility', 0.0):.4f}, "
                f"auc={l2_metrics.get('auc', 0.0):.4f}, "
                f"trades_per_day={l2_metrics.get('trades_per_day', 0.0):.2f}, "
                f"sharpe_mean={l2_metrics.get('sharpe_mean', 0.0):.4f}, "
                f"sharpe_std={l2_metrics.get('sharpe_std', 0.0):.4f}"
            )
            tprint_info(
                "   Layer 2 gates: "
                f"base_score={l2_metrics.get('base_score', 0.0):.4f}, "
                f"phi_auc={l2_metrics.get('phi_auc', 0.0):.4f}, "
                f"phi_density={l2_metrics.get('phi_density', 0.0):.4f}, "
                f"modifier={l2_metrics.get('modifier', 0.0):.4f}"
            )
        except Exception as l2_diag_exc:
            l2_metrics = {}
            tprint_warning(f"   ⚠️ Failed to compute Layer 2 metrics breakdown: {l2_diag_exc}")
        tprint_success(f"✅ Layer 2 Complete. Best Score: {best_l2_score:.4f}")
        tprint_info(f"   Best Trading Params: {best_trading_params}")

        # Persist Layer 2 params immediately
        try:
            l2_path = Path("outcomes") / f"hpo_layer2_best_params_{symbol}_{timeframe}_{ts_l2}.json"
            l2_payload = {
                "best_params": best_trading_params,
                "best_score": best_l2_score,
                "timestamp": ts_l2,
            }
            l2_path.parent.mkdir(parents=True, exist_ok=True)
            with open(l2_path, "w") as f:
                json.dump(l2_payload, f, indent=2, default=str)
            tprint_info(f"   💾 Saved Layer 2 best params to {l2_path}")
        except Exception as l2_exc:
            tprint_warning(f"   ⚠️ Failed to save Layer 2 params: {l2_exc}")

        # Persist Layer 2 trial metrics for correlation analysis
        l2_trials_path: Optional[Path] = None
        try:
            trial_rows = []
            for trial in l2_result.get("history", []):
                params = trial.get("params", {}) if isinstance(trial, dict) else {}
                if layer2_mode == "committee":
                    metrics_trial = _compute_layer2_metrics_committee(params)
                else:
                    metrics_trial = _compute_layer2_metrics(params)
                row = {
                    "valid_events": metrics_trial.get("valid_events"),
                    "utility": metrics_trial.get("utility"),
                    "auc": metrics_trial.get("auc"),
                    "trades_per_day": metrics_trial.get("trades_per_day"),
                    "calibration_brier": metrics_trial.get("calibration_brier"),
                    "calibration_ece": metrics_trial.get("calibration_ece"),
                    "sharpe_mean": metrics_trial.get("sharpe_mean"),
                    "sharpe_std": metrics_trial.get("sharpe_std"),
                    "sharpe_min": metrics_trial.get("sharpe_min"),
                    "sharpe_max": metrics_trial.get("sharpe_max"),
                    "n_trades": metrics_trial.get("n_trades"),
                    "trade_mean_return": metrics_trial.get("trade_mean_return"),
                    "trade_win_rate": metrics_trial.get("trade_win_rate"),
                    "take_rate": metrics_trial.get("take_rate"),
                    "consensus_mean": metrics_trial.get("consensus_mean"),
                    "consensus_std": metrics_trial.get("consensus_std"),
                    "consensus_p10": metrics_trial.get("consensus_p10"),
                    "consensus_p50": metrics_trial.get("consensus_p50"),
                    "consensus_p90": metrics_trial.get("consensus_p90"),
                    # Optional per-fold Sharpe values for deeper correlation
                    "folds_sharpe_values": json.dumps(metrics_trial.get("folds_sharpe_values", [])),
                    "lambda_vol": metrics_trial.get("lambda_vol"),
                    "w_auc": metrics_trial.get("w_auc"),
                    "w_den": metrics_trial.get("w_den"),
                    "avg_sharpe": metrics_trial.get("avg_sharpe"),
                    "vol_sharpe": metrics_trial.get("vol_sharpe"),
                    "base_score": metrics_trial.get("base_score"),
                    "base_norm": metrics_trial.get("base_norm"),
                    "phi_auc": metrics_trial.get("phi_auc"),
                    "phi_density": metrics_trial.get("phi_density"),
                    "modifier": metrics_trial.get("modifier"),
                }
                for k, v in params.items():
                    row[f"param_{k}"] = v
                trial_rows.append(row)

            if trial_rows:
                l2_trials_path = Path("outcomes") / f"hpo_layer2_trials_{symbol}_{timeframe}_{ts_l2}.csv"
                pd.DataFrame(trial_rows).to_csv(l2_trials_path, index=False)
                tprint_info(f"   💾 Saved Layer 2 trial metrics to {l2_trials_path}")
        except Exception as l2_trials_exc:
            tprint_warning(f"   ⚠️ Failed to save Layer 2 trial metrics: {l2_trials_exc}")

        # ------------------------------------------------------------------
        # Layer 2 Debug Diagnostics (single re-evaluation with best params)
        # ------------------------------------------------------------------
        try:
            tprint_info("   🔍 Layer 2 debug: re-evaluating best params with diagnostics...")
            debug_trail = float(best_trading_params.get("trail_distance_atr_mult", 0.0))
            debug_prof_thr = fixed_layer2_profit_thr
            debug_stop_thr = fixed_layer2_stop_thr
            (
                dbg_returns,
                dbg_labels,
                _,
                dbg_durations,
                dbg_mfe,
                dbg_mae,
                _, _
            ) = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=debug_prof_thr,
                stop_threshold=debug_stop_thr,
                horizon=12,
                transaction_cost=DEFAULT_TRANSACTION_COST,
                min_event_spacing=2,
                trail_distance_atr_mult=debug_trail,
                atr_series=atr_series,
            )
            dbg_valid_idx = ~dbg_labels.isna()
            dbg_valid_events = int(dbg_valid_idx.sum())
            if dbg_valid_events < 50:
                tprint_info(
                    f"   Layer 2 debug: valid_events={dbg_valid_events} (<50), "
                    "objective would early-return -1.0"
                )
            else:
                dbg_t_events = dbg_returns.index[dbg_valid_idx]
                dbg_returns_clean = dbg_returns[dbg_valid_idx]
                dbg_labels_clean = dbg_labels[dbg_valid_idx]
                # Rebuild event horizons and uniqueness
                t0_locs_dbg = pd.Series(np.arange(len(market_data)), index=market_data.index)
                start_locs_dbg = t0_locs_dbg.loc[dbg_t_events].values
                dur_vals_dbg = dbg_durations.loc[dbg_t_events].values.astype(int)
                end_locs_dbg = np.minimum(start_locs_dbg + dur_vals_dbg, len(market_data) - 1)
                t1_vals_dbg = market_data.index[end_locs_dbg]
                t1_series_dbg = pd.Series(t1_vals_dbg, index=dbg_t_events)
                batch_consistency_dbg = full_consistency.reindex(dbg_t_events).fillna(1.0).values
                batch_volatility_dbg = full_volatility.reindex(dbg_t_events).fillna(0).values
                batch_uniqueness_dbg = compute_uniqueness(t1_series_dbg, market_index=market_data.index)
                sample_weights_dbg = generate_weights_per_label(
                    returns=dbg_returns_clean.values,
                    t_events=dbg_t_events,
                    close_series=None,
                    consistency_scores=batch_consistency_dbg,
                    uniqueness_scores=batch_uniqueness_dbg.values,
                    vol_proxy=batch_volatility_dbg,
                    **best_weighting_params,
                )
                # Subset meta-features
                X_dbg = meta_features_full.loc[dbg_valid_idx].fillna(0)
                # Fast model + CV
                n_cv_folds_dbg = 5
                fast_model_dbg = lgb.LGBMClassifier(
                    n_estimators=60,
                    max_depth=3,
                    learning_rate=0.1,
                    n_jobs=-1,
                    verbose=-1,
                    random_state=42,
                )
                try:
                    cv_preds_dbg, folds_sharpe_dbg, mean_brier_dbg, mean_ece_dbg = _cross_val_predict_proba_and_fold_sharpes_weighted(
                        estimator=fast_model_dbg,
                        X=X_dbg,
                        y=dbg_labels_clean,
                        sample_weight=sample_weights_dbg,
                        n_splits=n_cv_folds_dbg,
                        returns=dbg_returns_clean.values.astype(float),
                        direction=direction,
                        prob_thr=0.5,
                        use_calibration=True,
                        enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                        ev_margin=config.get("ev_margin", 0.0),
                    )
                except Exception as dbg_cv_exc:
                    tprint_warning(f"   ⚠️ Layer 2 debug: CV failed: {dbg_cv_exc}")
                    cv_preds_dbg = np.full(dbg_valid_events, 0.5, dtype=float)
                    folds_sharpe_dbg = np.array([0.0], dtype=float)
                    mean_brier_dbg = None
                    mean_ece_dbg = None
                # AUC
                try:
                    mean_auc_dbg = roc_auc_score(dbg_labels_clean.values, cv_preds_dbg)
                except Exception:
                    mean_auc_dbg = 0.5
                trades_per_day_dbg = len(dbg_returns_clean) / max(days_span, 1)
                # Reconstruct base_score as in calculate_hpo_utility
                avg_sharpe_dbg = float(np.mean(folds_sharpe_dbg))
                vol_sharpe_dbg = float(np.std(folds_sharpe_dbg, ddof=1)) if len(folds_sharpe_dbg) > 1 else 0.0
                base_score_dbg = avg_sharpe_dbg - 1.2 * vol_sharpe_dbg
                utility_dbg = calculate_hpo_utility(
                    folds_sharpe=folds_sharpe_dbg,
                    auc=mean_auc_dbg,
                    trades_per_day=trades_per_day_dbg,
                    lambda_vol=1.2,
                    w_auc=1.0,
                    w_den=0.5,
                    calibration_brier=mean_brier_dbg,
                    calibration_ece=mean_ece_dbg,
                    w_cal=0.0,
                )
                tprint_info(
                    "   Layer 2 debug: "
                    f"valid_events={dbg_valid_events}, "
                    f"AUC={mean_auc_dbg:.4f}, "
                    f"trades_per_day={trades_per_day_dbg:.2f}"
                )
                tprint_info(
                    "   Layer 2 debug: "
                    f"folds_sharpe={folds_sharpe_dbg.tolist()}, "
                    f"base_score={base_score_dbg:.4f}, "
                    f"utility={utility_dbg:.4f}"
                )
        except Exception as dbg_exc:
            tprint_warning(f"   ⚠️ Layer 2 debug diagnostics failed: {dbg_exc}")


        # Save Layer 2 History
        l2_history_path: Optional[Path] = None
        try:
            ts_l2_hist = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            l2_history_path = Path("outcomes") / f"hpo_layer2_history_{symbol}_{timeframe}_{ts_l2_hist}.json"
            with open(l2_history_path, "w") as f:
                json.dump(l2_result.get("history", []), f, default=str, indent=4)
            tprint_info(f"   💾 Saved Layer 2 history to {l2_history_path}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to save Layer 2 history: {e}")

        try:
            l2_report = _write_hpo_stage_report(
                outcomes_dir=outcomes_dir,
                run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
                stage_id="layer2_trading",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                best_params=dict(best_trading_params) if isinstance(best_trading_params, dict) else {},
                metrics={
                    "best_score": best_l2_score,
                    **(l2_metrics if isinstance(l2_metrics, dict) else {}),
                },
                search_space=layer2_search_space,
                trials_csv_path=l2_trials_path,
                history_json_path=l2_history_path,
            )
            hpo_stage_reports["layer2"] = l2_report
        except Exception as l2_report_exc:
            tprint_warning(f"   ⚠️ Failed to write Layer 2 report: {l2_report_exc}")

        # ------------------------------------------------------------------
        # 3. LAYER 3: MODEL HYPERPARAMETER OPTIMIZATION
        # ------------------------------------------------------------------
        tprint_info("🧪 Layer 3: Optimizing Model Hyperparameters...")

        final_exit_reasons = None
        final_mfe = None
        final_mae = None
        committee_gate_series_for_l3: Optional[pd.Series] = None

        if isinstance(best_trading_params, dict) and layer2_mode != "committee":
            final_trail = float(best_trading_params.get("trail_distance_atr_mult", 0.0))

            final_prof_thr = fixed_layer2_profit_thr.reindex(market_data.index).fillna(float(layer2_tp_target))
            final_stop_thr = fixed_layer2_stop_thr.reindex(market_data.index).fillna(float(layer2_sl_target))

            (
                final_returns,
                final_labels,
                final_exit_reasons,
                final_durations,
                final_mfe,
                final_mae,
                _,
                _,
            ) = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=final_prof_thr,
                stop_threshold=final_stop_thr,
                horizon=12,
                transaction_cost=DEFAULT_TRANSACTION_COST,
                min_event_spacing=2,
                trail_distance_atr_mult=final_trail,
                atr_series=atr_series,
            )
        else:
            try:
                w_scalp = float(best_trading_params.get("w_scalp", 0.0))
                w_swing = float(best_trading_params.get("w_swing", 0.0))
                w_trend = float(best_trading_params.get("w_trend", 0.0))
                threshold = float(best_trading_params.get("consensus_threshold", 0.5))
                consensus_quantile = best_trading_params.get("consensus_quantile", None)
                consensus_quantile = float(consensus_quantile) if consensus_quantile is not None else None

                weights_vec = np.array([w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend], dtype=float)
                total_weight = float(weights_vec.sum()) + 1e-8

                weighted_sum = label_matrix_values.dot(weights_vec)
                consensus_score = weighted_sum / total_weight
                scores_arr = np.asarray(consensus_score, dtype=float)
                scores_arr = np.where(np.isfinite(scores_arr), scores_arr, -np.inf)

                try:
                    if consensus_quantile is not None and np.isfinite(consensus_quantile):
                        q = float(np.clip(consensus_quantile, 0.0, 0.999999))
                        n = int(scores_arr.size)
                        if n > 0:
                            k = int(np.ceil((1.0 - q) * float(n)))
                            k = int(np.clip(k, 1, n))
                            top_idx = np.argpartition(scores_arr, n - k)[n - k :]
                            take_mask = np.zeros(n, dtype=bool)
                            take_mask[top_idx] = True
                        else:
                            take_mask = np.zeros(0, dtype=bool)
                    else:
                        take_mask = scores_arr >= float(threshold)
                except Exception:
                    take_mask = scores_arr >= float(threshold)

                final_t_events_all = pd.DatetimeIndex(event_idx)
                committee_gate_series_for_l3 = pd.Series(np.asarray(take_mask, dtype=bool), index=final_t_events_all)

                ret_mat = np.asarray(returns_matrix_values, dtype=float)
                finite_mask = np.isfinite(ret_mat)

                w_row = np.asarray(weights_vec, dtype=float).reshape(1, -1)
                denom = np.sum(finite_mask * w_row, axis=1).astype(float) + 1e-8
                numer = np.sum(np.where(finite_mask, ret_mat, 0.0) * w_row, axis=1).astype(float)
                weighted_returns = numer / denom
                weighted_returns = np.where(np.isfinite(weighted_returns), weighted_returns, 0.0)

                try:
                    require_hit = bool(config.get("layer3_committee_train_requires_barrier_hit", True))
                except Exception:
                    require_hit = True

                train_mask = np.ones(len(final_t_events_all), dtype=bool)
                try:
                    take_mask_arr = np.asarray(take_mask, dtype=bool)
                    if take_mask_arr.size == train_mask.size:
                        train_mask = take_mask_arr
                except Exception:
                    pass

                try:
                    min_events_l3 = int(config.get("layer3_min_events", 500))
                except Exception:
                    min_events_l3 = 500
                if int(np.sum(train_mask)) < int(min(50, max(10, min_events_l3))):
                    train_mask = np.ones(len(final_t_events_all), dtype=bool)

                final_returns = pd.Series(weighted_returns.astype(float), index=final_t_events_all).iloc[train_mask]
                final_labels = pd.Series((weighted_returns > 0).astype(float), index=final_t_events_all).iloc[train_mask]

                horizon_bars_l3 = int(best_trading_params.get("horizon_bars", 12))
                final_durations = pd.Series(
                    np.full(len(final_t_events_all), np.nan, dtype=float),
                    index=final_t_events_all,
                )
                final_durations = final_durations.iloc[train_mask]
            except Exception as committee_l3_exc:
                tprint_warning(f"⚠️ Layer 3: failed to reconstruct committee returns/labels: {committee_l3_exc}")
                return {"success": False}

        valid_final_mask = ~final_labels.isna()
        if valid_final_mask.sum() < 50:
            tprint_warning("⚠️ Layer 3: Insufficient events. Aborting.")
            return {"success": False}

        final_t_events = final_returns.index[valid_final_mask]
        X_final = meta_features_full.reindex(final_t_events).fillna(0)
        y_final = final_labels[valid_final_mask]

        committee_gate_arr_for_l3: Optional[np.ndarray] = None
        try:
            if committee_gate_series_for_l3 is not None:
                committee_gate_arr_for_l3 = (
                    committee_gate_series_for_l3.reindex(final_t_events).fillna(False).values.astype(bool)
                )
        except Exception:
            committee_gate_arr_for_l3 = None

        try:
            x_idx = pd.DatetimeIndex(X_final.index)
            y_idx = pd.DatetimeIndex(y_final.index)
            common_idx = x_idx.intersection(y_idx)
            if len(common_idx) < len(x_idx):
                X_final = X_final.loc[common_idx]
            if len(common_idx) < len(y_idx):
                y_final = y_final.loc[common_idx]
            final_t_events = common_idx
        except Exception:
            pass

        # Explicitly define X_final_full for feature selection usage
        X_final_full = X_final.copy()

        # ------------------------------------------------------------------
        # DIAGNOSTIC: Log HPO feature/label summary for comparison with SNR probe
        # ------------------------------------------------------------------
        try:
            n_feat = int(X_final_full.shape[1])
            n_samples = int(X_final_full.shape[0])
            pos_rate = float(y_final.mean()) if len(y_final) > 0 else 0.0
            n_pos = int((y_final == 1).sum())
            n_neg = int((y_final == 0).sum())
            sample_cols = list(X_final_full.columns[:10])
            tprint_info("=" * 60)
            tprint_info(f"📊 [HPO DIAGNOSTIC] Layer 3 Dataset Summary:")
            tprint_info(f"   Features: {n_feat}, Samples: {n_samples}")
            tprint_info(f"   Labels: pos={n_pos}, neg={n_neg}, pos_rate={pos_rate:.3f}")
            tprint_info(f"   Sample cols: {sample_cols}")
            tprint_info("=" * 60)
        except Exception as diag_exc:
            tprint_warning(f"⚠️ Diagnostic logging failed: {diag_exc}")

        # ------------------------------------------------------------------
        # SANITY AUC: Quick probe to verify HPO data can achieve similar AUC to SNR
        # ------------------------------------------------------------------
        try:
            if len(y_final) >= 100 and len(np.unique(y_final)) >= 2:
                from sklearn.model_selection import TimeSeriesSplit as _TimeSeriesSplit
                sanity_kf = _TimeSeriesSplit(n_splits=3)
                sanity_aucs = []
                sanity_model = lgb.LGBMClassifier(
                    n_estimators=50, max_depth=4, num_leaves=16,
                    verbose=-1, n_jobs=-1, random_state=42
                )
                for tr_idx, te_idx in sanity_kf.split(X_final_full):
                    X_tr, X_te = X_final_full.iloc[tr_idx], X_final_full.iloc[te_idx]
                    y_tr, y_te = y_final.iloc[tr_idx], y_final.iloc[te_idx]
                    if len(np.unique(y_tr)) < 2:
                        continue
                    sanity_model.fit(X_tr, y_tr)
                    preds = sanity_model.predict_proba(X_te)[:, 1]
                    try:
                        from sklearn.metrics import roc_auc_score
                        sanity_aucs.append(float(roc_auc_score(y_te, preds)))
                    except Exception:
                        pass
                if sanity_aucs:
                    sanity_mean_auc = float(np.mean(sanity_aucs))
                    tprint_info(f"📊 [SANITY CHECK] Quick probe AUC (3-fold): {sanity_mean_auc:.4f}")
                    if sanity_mean_auc < 0.55:
                        tprint_warning(f"   ⚠️ SANITY FAIL: Probe AUC {sanity_mean_auc:.4f} < 0.55 — data may be misaligned!")
                    elif sanity_mean_auc >= 0.70:
                        tprint_success(f"   ✅ SANITY PASS: Probe AUC {sanity_mean_auc:.4f} >= 0.70 — data looks consistent.")
        except Exception as sanity_exc:
            tprint_warning(f"⚠️ Sanity AUC check failed: {sanity_exc}")

        try:
            t0_locs = pd.Series(np.arange(len(market_data)), index=market_data.index)
            start_locs = t0_locs.loc[final_t_events].values
            dur_vals = final_durations.loc[final_t_events].values.astype(int)
            end_locs = np.minimum(start_locs + dur_vals, len(market_data) - 1)
            t1_vals = market_data.index[end_locs]
            t1_series = pd.Series(t1_vals, index=final_t_events)

            batch_con_final = full_consistency.reindex(final_t_events).fillna(1.0).values
            batch_vol_final = full_volatility.reindex(final_t_events).fillna(0).values
            batch_uniq_final = compute_uniqueness(t1_series, market_index=market_data.index)

            final_weights = generate_weights_per_label(
                returns=final_returns[valid_final_mask].reindex(final_t_events).values,
                t_events=final_t_events,
                close_series=None,
                consistency_scores=batch_con_final,
                uniqueness_scores=batch_uniq_final.values,
                vol_proxy=batch_vol_final,
                **best_weighting_params
            )

            try:
                if committee_weight_factor_series is not None:
                    cf = committee_weight_factor_series.reindex(final_t_events).fillna(1.0).values.astype(float)
                    cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                    final_weights = np.asarray(final_weights, dtype=float) * cf
                    fw_mean = float(np.mean(final_weights)) if final_weights.size else 1.0
                    if np.isfinite(fw_mean) and fw_mean > 0:
                        final_weights = final_weights / fw_mean
            except Exception:
                pass
        except Exception:
            final_weights = np.ones(len(final_t_events), dtype=float)

        tprint_info(
            f"[L3 data check] events={len(final_t_events)}, "
            f"features={X_final.shape[1]}, labels_valid={valid_final_mask.sum()}"
        )

        l3_feature_selection_info: Dict[str, Any] = {
            "enabled": bool(config.get("enable_mda_shap_selection_layer3", True)),
            "selected_features": [],
            "dropped_features": [],
            "prefilter_counts": {},
            "prefilter_features": [],
            "n_features_before": 0,
            "n_features_after": 0,
            "artifact_path": None,
        }

        # Backward/alternate switch name support
        disable_fs_alias = bool(config.get("disable_mda_shap_selection_layer3", False))
        enable_fs = bool(config.get("enable_mda_shap_selection_layer3", True)) and (not disable_fs_alias)
        try:
            l3_feature_selection_info["enabled"] = bool(enable_fs)
        except Exception:
            pass

        reuse_cached_fs = bool(enable_fs) and (stage_rank["feature_selection"] < start_rank)
        if reuse_cached_fs:
            try:
                fs_path = _find_latest_path(
                    Path("outcomes"),
                    f"hpo_layer3_feature_selection_{symbol}_{timeframe}_*.json",
                )
                fs_data = _load_latest_json(fs_path)
                cached_selected = None
                if isinstance(fs_data, dict):
                    cached_selected = fs_data.get("selected_features")
                if isinstance(cached_selected, list) and cached_selected:
                    cached_selected = [f for f in cached_selected if f in X_final_full.columns]
                if isinstance(cached_selected, list) and cached_selected:
                    before_n = int(X_final_full.shape[1])
                    X_final = X_final_full[cached_selected].copy()
                    l3_feature_selection_info["selected_features"] = list(cached_selected)
                    l3_feature_selection_info["dropped_features"] = sorted(
                        list(set(X_final_full.columns) - set(cached_selected))
                    )
                    l3_feature_selection_info["n_features_before"] = int(before_n)
                    l3_feature_selection_info["n_features_after"] = int(len(cached_selected))
                    l3_feature_selection_info["artifact_path"] = str(fs_path) if fs_path is not None else None
                    tprint_info(
                        f"♻️ Feature selection skipped (start_at={start_at_canonical}); "
                        f"reusing cached selection {before_n} → {len(cached_selected)} from {fs_path}"
                    )
                    enable_fs = False
            except Exception as reuse_fs_exc:
                tprint_warning(f"⚠️ Failed to reuse cached feature selection: {reuse_fs_exc}")

        # ------------------------------------------------------------------
        # MDA/SHAP FEATURE SELECTION (between Layer 2 and Layer 3)
        # ------------------------------------------------------------------
        # Always record counts even if feature selection is disabled or skipped.
        try:
            l3_feature_selection_info["n_features_before"] = int(X_final_full.shape[1])
            l3_feature_selection_info["n_features_after"] = int(X_final.shape[1])
        except Exception:
            pass

        if enable_fs:
            try:
                min_events_fs = int(config.get("layer3_mda_shap_min_events", config.get("layer3_min_events", 500)))
                try:
                    l3_feature_selection_info["gate_min_events"] = int(min_events_fs)
                    l3_feature_selection_info["gate_events"] = int(len(final_t_events))
                    l3_feature_selection_info["gate_feature_count"] = int(X_final.shape[1])
                except Exception:
                    pass
                try:
                    tprint_info(
                        f"   [Layer3 MDA/SHAP gate] events={int(len(final_t_events))}, min_events={int(min_events_fs)}, features={int(X_final.shape[1])}"
                    )
                except Exception:
                    pass

                gate_passed = bool(len(final_t_events) >= min_events_fs and X_final.shape[1] > 10)
                try:
                    l3_feature_selection_info["gate_passed"] = bool(gate_passed)
                except Exception:
                    pass

                if gate_passed:
                    try:
                        use_full_scope = bool(config.get("layer3_mda_shap_use_full_feature_scope", True))
                        if use_full_scope and 'X_dummy' in locals() and isinstance(X_dummy, pd.DataFrame) and not X_dummy.empty:
                            X_final_full = X_dummy.reindex(final_t_events).fillna(0)
                            try:
                                l3_feature_selection_info["input_feature_scope"] = "create_meta_features_expanded"
                            except Exception:
                                pass
                        elif use_full_scope and 'X_features' in locals() and isinstance(X_features, pd.DataFrame) and not X_features.empty:
                            X_final_full = X_features.reindex(final_t_events).fillna(0)
                            try:
                                l3_feature_selection_info["input_feature_scope"] = "create_meta_features"
                            except Exception:
                                pass
                        else:
                            # Fallback: expand the meta feature scope (multi-horizon + cross)
                            X_scope = meta_features_full_raw
                            try:
                                horizon_config = config.get(
                                    "feature_horizon_config",
                                    {
                                        "Short": 5,
                                        "Medium": 20,
                                        "Long": 60,
                                    },
                                )
                                X_scope = generate_multi_horizon_features(X_scope, horizon_config)
                            except Exception:
                                X_scope = meta_features_full_raw

                            try:
                                kalman_cols = [c for c in X_scope.columns if c.startswith("KF_")]
                                base_cols = [c for c in X_scope.columns if not c.startswith("KF_")]
                                kalman_features_df = X_scope[kalman_cols] if kalman_cols else pd.DataFrame(index=X_scope.index)
                                base_features_df = X_scope[base_cols] if base_cols else pd.DataFrame(index=X_scope.index)
                                cross_features_df = generate_cross_features(
                                    base_features=base_features_df,
                                    kalman_features=kalman_features_df,
                                    market_data=market_data if market_data is not None else pd.DataFrame(index=X_scope.index),
                                )
                                if cross_features_df is not None and not cross_features_df.empty:
                                    for col in cross_features_df.columns:
                                        if col not in X_scope.columns:
                                            X_scope[col] = cross_features_df[col]
                            except Exception:
                                pass

                            X_final_full = X_scope.reindex(final_t_events).fillna(0)
                            try:
                                l3_feature_selection_info["input_feature_scope"] = "expanded_meta_features"
                            except Exception:
                                pass
                    except Exception:
                        X_final_full = X_final

                    # Record pre-selection feature count (actual input to selector)
                    try:
                        l3_feature_selection_info["n_features_before"] = int(X_final_full.shape[1])
                    except Exception:
                        pass

                    try:
                        ts_l3_inputs = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        l3_inputs_path = outcomes_dir / f"hpo_layer3_input_feature_names_{symbol}_{timeframe}_{ts_l3_inputs}.json"
                        payload = {
                            "n_features": int(X_final_full.shape[1]),
                            "features": list(X_final_full.columns),
                        }
                        with open(l3_inputs_path, "w") as f:
                            json.dump(payload, f, indent=2, default=str)
                        l3_feature_selection_info["input_feature_names_path"] = str(l3_inputs_path)
                        tprint_info(
                            f"   [Layer3 MDA/SHAP input features] n={int(X_final_full.shape[1])}; "
                            f"saved={l3_inputs_path}"
                        )
                    except Exception:
                        pass

                    tprint_info(
                        f"   [Layer3 MDA/SHAP] Selecting features for Layer 3 "
                        f"(events={len(final_t_events)}, features={int(X_final_full.shape[1])})"
                    )

                    provided_l3_cfg = isinstance(config.get("mda_shap_config_layer3"), dict)
                    mda_shap_cfg = config.get(
                        "mda_shap_config_layer3",
                        {
                            "model_type": "rf",
                            "n_folds": int(config.get("cv_splits", 5)),
                            "pre_filters": {
                                "enable_lgbm_mdi_filter": True,
                                "enable_correlation_filter": True,
                                "enable_variance_filter": True,
                                "enable_anova_filter": True,
                            },
                            "corr_threshold": 0.85,
                            "top_clusters": 8,
                            "shap_sample_size": min(1000, len(y_final)),
                            "verbose": True,
                        },
                    )

                    try:
                        if isinstance(mda_shap_cfg, dict):
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

                            mda_shap_cfg.setdefault(
                                "regime_leaf_config",
                                {
                                    "enabled": bool(config.get("enable_regime_leaf_features", True)),
                                    "market_data": market_data,
                                    "X_base": None,
                                    "extractor_config": extractor_cfg,
                                    "random_state": int(config.get("random_state", 42)),
                                    "verbose": True,
                                },
                            )

                            # Raise elbow minimum base-feature selection for Layer3.
                            mda_shap_cfg.setdefault("elbow_min_features", 40)

                            if provided_l3_cfg:
                                mda_shap_cfg.setdefault(
                                    "enable_shap_interaction_features",
                                    bool(config.get("enable_shap_interaction_features", False)),
                                )
                                mda_shap_cfg.setdefault(
                                    "shap_interaction_config",
                                    (
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
                                )
                            else:
                                mda_shap_cfg["enable_shap_interaction_features"] = bool(
                                    config.get("enable_shap_interaction_features", False)
                                )
                                if isinstance(config.get("shap_interaction_config"), dict):
                                    mda_shap_cfg["shap_interaction_config"] = config.get("shap_interaction_config")
                    except Exception:
                        pass

                    w_series = None
                    try:
                        w_series = pd.Series(final_weights, index=final_t_events).reindex(X_final_full.index).fillna(1.0)
                    except Exception:
                        w_series = None

                    selected_feats_l3, selection_results_l3 = run_mda_shap_feature_selection(
                        X=X_final_full,
                        y=y_final,
                        target_sample_weight=w_series,
                        config=mda_shap_cfg,
                        artifact_router=self.artifact_router,
                        pipeline_context={
                            "symbol": config.get("symbol"),
                            "exchange": config.get("exchange"),
                            "timeframe": config.get("timeframe"),
                            "direction": config.get("direction"),
                        },
                    )

                    try:
                        shap_inter = (
                            selection_results_l3.get("shap_interaction_features", {})
                            if isinstance(selection_results_l3, dict)
                            else {}
                        )
                        inter_defs = shap_inter.get("interaction_defs", []) if isinstance(shap_inter, dict) else []
                        if inter_defs:
                            from .shap_interaction_feature_mining import apply_interaction_definitions

                            fillna_value = 0.0
                            try:
                                fillna_value = float(
                                    (mda_shap_cfg.get("shap_interaction_config") or {}).get("fillna_value", 0.0)
                                )
                            except Exception:
                                fillna_value = 0.0

                            inter_df_full = apply_interaction_definitions(
                                X_final_full,
                                inter_defs,
                                fillna_value=fillna_value,
                            )
                            if inter_df_full is not None and not inter_df_full.empty:
                                X_final_full = pd.concat([X_final_full, inter_df_full], axis=1)
                                tprint_info(f"   🧩 Added SHAP interaction features (Layer3): {int(inter_df_full.shape[1])}")
                    except Exception:
                        pass

                    try:
                        pre_counts = selection_results_l3.get("prefilter_counts", {}) if isinstance(selection_results_l3, dict) else {}
                        if isinstance(pre_counts, dict) and pre_counts:
                            tprint_info(
                                "   [Layer3 MDA/SHAP prefilters] "
                                + ", ".join([f"{k}={int(v)}" for k, v in pre_counts.items() if v is not None])
                            )
                    except Exception:
                        pass

                    try:
                        if isinstance(selection_results_l3, dict):
                            l3_feature_selection_info["prefilter_counts"] = selection_results_l3.get("prefilter_counts", {})
                            l3_feature_selection_info["n_features_original"] = selection_results_l3.get("n_features_original")
                            l3_feature_selection_info["n_features_after_prefilters"] = selection_results_l3.get("n_features_after_prefilters")
                            l3_feature_selection_info["n_features_selected"] = selection_results_l3.get("n_features_selected")
                            l3_feature_selection_info["prefilter_features"] = selection_results_l3.get("prefilter_features", [])

                            rl = selection_results_l3.get("regime_leaf_features")
                            if isinstance(rl, dict):
                                l3_feature_selection_info["regime_leaf_features"] = rl
                    except Exception:
                        pass

                    if selected_feats_l3:
                        # The selector can internally append regime-leaf features for scoring/selection.
                        # However, the pipeline's X_final_full does not include those columns unless we
                        # explicitly extract and concat them here.
                        try:
                            missing = [
                                f for f in list(selected_feats_l3)
                                if isinstance(f, str) and f not in X_final_full.columns
                            ]
                            needs_regime = any(
                                isinstance(f, str) and f.startswith("regime_leaf_") for f in missing
                            )
                            if needs_regime and isinstance(mda_shap_cfg, dict):
                                rl_cfg = mda_shap_cfg.get("regime_leaf_config")
                                if isinstance(rl_cfg, dict) and bool(rl_cfg.get("enabled", False)):
                                    extractor_cfg = rl_cfg.get("extractor_config")
                                    if isinstance(extractor_cfg, dict):
                                        from .regime_leaf_feature_extractor import extract_regime_leaf_onehot_features

                                        rl_df = extract_regime_leaf_onehot_features(
                                            X=X_final_full,
                                            market_data=market_data,
                                            config=extractor_cfg,
                                            random_state=int(rl_cfg.get("random_state", 42)),
                                            verbose=bool(rl_cfg.get("verbose", True)),
                                        )
                                        if rl_df is not None and not getattr(rl_df, "empty", True):
                                            rl_df = rl_df.reindex(X_final_full.index).fillna(0.0)
                                            X_final_full = pd.concat([X_final_full, rl_df], axis=1)
                        except Exception:
                            pass

                        final_selected = [f for f in list(selected_feats_l3) if f in X_final_full.columns]

                        before_n = int(X_final_full.shape[1])
                        X_final = X_final_full[final_selected].copy()
                        tprint_success(f"   Layer3 MDA/SHAP: {before_n} → {len(final_selected)} features")
                        try:
                            tprint_info(
                                "   [Layer3 MDA/SHAP selected features] "
                                + ", ".join(list(final_selected))
                            )
                        except Exception:
                            pass

                        try:
                            l3_feature_selection_info["selected_features"] = list(final_selected)
                            
                            # Calculate dropped features (Input - Selected)
                            input_set = set(X_final_full.columns)
                            selected_set = set(final_selected)
                            dropped_set = input_set - selected_set
                            l3_feature_selection_info["dropped_features"] = sorted(list(dropped_set))
                            
                            l3_feature_selection_info["n_features_before"] = int(before_n)
                            l3_feature_selection_info["n_features_after"] = int(len(final_selected))
                        except Exception:
                            pass
                    else:
                        tprint_warning("   ⚠️ Layer3 MDA/SHAP returned no features; keeping all")
                        try:
                            l3_feature_selection_info["n_features_before"] = int(X_final_full.shape[1])
                            l3_feature_selection_info["n_features_after"] = int(X_final_full.shape[1])
                            l3_feature_selection_info["dropped_features"] = []
                        except Exception:
                            pass
                else:
                    tprint_warning("   ⚠️ Layer3 MDA/SHAP skipped (insufficient events/features)")
                    try:
                        l3_feature_selection_info["skipped_reason"] = "insufficient_events_or_features"
                    except Exception:
                        pass
            except Exception as e_l3mda:
                tprint_warning(f"   ⚠️ Layer3 MDA/SHAP selection failed: {e_l3mda}")
                try:
                    l3_feature_selection_info["error"] = str(e_l3mda)
                except Exception:
                    pass
                # Ensure counts remain coherent on failure
                try:
                    l3_feature_selection_info["n_features_before"] = int(X_final_full.shape[1])
                    l3_feature_selection_info["n_features_after"] = int(X_final.shape[1])
                except Exception:
                    pass
        else:
            # Feature selection explicitly disabled
            try:
                l3_feature_selection_info["n_features_before"] = int(X_final_full.shape[1])
                l3_feature_selection_info["n_features_after"] = int(X_final.shape[1])
                l3_feature_selection_info["dropped_features"] = []
            except Exception:
                pass

        try:
            ts_l3_final_feats = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            l3_final_feats_path = outcomes_dir / f"hpo_layer3_feature_names_{symbol}_{timeframe}_{ts_l3_final_feats}.json"
            payload = {
                "n_features": int(X_final.shape[1]),
                "features": list(X_final.columns),
            }
            with open(l3_final_feats_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            l3_feature_selection_info["final_feature_names_path"] = str(l3_final_feats_path)
        except Exception:
            pass

        try:
            ts_l3_fs = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            l3_fs_path = outcomes_dir / f"hpo_layer3_feature_selection_{symbol}_{timeframe}_{ts_l3_fs}.json"
            try:
                l3_feature_selection_info["artifact_path"] = str(l3_fs_path)
            except Exception:
                pass
            with open(l3_fs_path, "w") as f:
                json.dump(l3_feature_selection_info, f, indent=2, default=str)
        except Exception:
            pass

        target_sample_weight = final_weights

        layer3_search_space = {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "num_leaves": {"type": "int", "low": 8, "high": 64},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        }

        # Precompute values needed for Layer 3 utility calculation (aligned to final_t_events)
        final_returns_arr = final_returns.reindex(final_t_events).values.astype(float)
        final_labels_arr = y_final.values.astype(float)
        final_exit_reasons_arr = None
        try:
            if final_exit_reasons is not None:
                final_exit_reasons_arr = final_exit_reasons.loc[final_t_events].astype(object).values
        except Exception:
            final_exit_reasons_arr = None
        final_durations_arr = None
        try:
            if final_durations is not None:
                final_durations_arr = final_durations.loc[final_t_events].astype(float).values
        except Exception:
            final_durations_arr = None

        def _compute_layer3_metrics(model_params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
            """Run Layer 3 CV and return utility plus component diagnostics.

            Never surfaces -inf/inf: any failure is converted to a finite sentinel utility.
            """
            try:
                model = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42, **model_params)

                n_cv_folds = 5
                kf = TimeSeriesSplit(n_splits=n_cv_folds)

                fold_aucs: List[float] = []
                fold_sharpes: List[float] = []
                fold_briers: List[float] = []
                fold_eces: List[float] = []

                # Store OOF calibrated predictions for density reporting
                oof_pred_cal = np.full(len(y_final), np.nan, dtype=float)
                per_fold_metrics: List[Dict[str, Any]] = []

                for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_final)):
                    X_tr, X_te = X_final.iloc[tr_idx], X_final.iloc[te_idx]
                    y_tr, y_te = y_final.iloc[tr_idx], y_final.iloc[te_idx]
                    w_tr = final_weights[tr_idx]

                    # Handle degenerate folds (single-class train split) without crashing.
                    # This can happen in time-series CV when positives are sparse.
                    y_tr_unique = np.unique(y_tr.values)
                    if len(y_tr_unique) < 2:
                        try:
                            prior = float(np.mean(y_tr.values.astype(float)))
                        except Exception:
                            prior = 0.5
                        prior = float(np.clip(prior, 0.0, 1.0)) if np.isfinite(prior) else 0.5
                        preds = np.full(int(len(X_te)), prior, dtype=float)
                        preds_cal = preds
                        try:
                            oof_pred_cal[te_idx] = preds_cal
                        except Exception:
                            pass

                        # Fold AUC: only meaningful if y_te has 2 classes.
                        try:
                            if len(np.unique(y_te.values)) >= 2:
                                fold_auc_local = float(roc_auc_score(y_te, preds))
                                fold_aucs.append(float(fold_auc_local))
                            else:
                                tprint_warning(f"   ⚠️ Fold {fold_idx}: Single class in test set (y_unique={np.unique(y_te.values)}). AUC undefined.")
                        except Exception as auc_exc:
                            tprint_warning(f"   ⚠️ Fold {fold_idx}: AUC calculation failed: {auc_exc}")
                            pass

                        # Sharpe from sized returns.
                        try:
                            ret_te = final_returns_arr[te_idx]
                            gate_te = None
                            try:
                                if committee_gate_arr_for_l3 is not None:
                                    gate_te = np.asarray(committee_gate_arr_for_l3, dtype=bool)[te_idx]
                            except Exception:
                                gate_te = None
                            sized_returns = []
                            for ii, (prob, ret) in enumerate(zip(preds_cal, ret_te)):
                                size = directional_size_from_prob(
                                    float(prob),
                                    direction=direction,
                                    thr=0.5,
                                    max_exposure=1.0,
                                    scale=1.0,
                                )
                                if gate_te is not None and (not bool(gate_te[ii])):
                                    size = 0.0
                                sized_returns.append(size * float(ret))
                            sized_returns_arr = np.asarray(sized_returns, dtype=float)
                            sized_returns_arr = sized_returns_arr[np.isfinite(sized_returns_arr)]
                            if sized_returns_arr.size > 1 and float(np.std(sized_returns_arr)) > 1e-9:
                                fold_sharpe = float(np.mean(sized_returns_arr) / np.std(sized_returns_arr))
                            else:
                                fold_sharpe = 0.0
                            try:
                                fold_sharpe = float(np.clip(float(fold_sharpe), -20.0, 20.0))
                            except Exception:
                                fold_sharpe = 0.0
                            fold_sharpes.append(float(fold_sharpe))
                        except Exception:
                            fold_sharpes.append(0.0)

                        # Per-fold metrics (trade stats).
                        try:
                            trade_mask_fold = np.asarray(preds_cal, dtype=float) >= 0.5
                            try:
                                if gate_te is not None:
                                    trade_mask_fold = trade_mask_fold & np.asarray(gate_te, dtype=bool)
                            except Exception:
                                pass
                            ret_te = np.asarray(final_returns_arr[te_idx], dtype=float)
                            ret_sel = ret_te[trade_mask_fold]
                            ret_sel = ret_sel[np.isfinite(ret_sel)]
                            n_trades_fold = int(ret_sel.size)
                            mean_ret_fold = float(np.mean(ret_sel)) if n_trades_fold > 0 else 0.0
                            win_rate_fold = float(np.mean(ret_sel > 0.0)) if n_trades_fold > 0 else 0.0
                            per_fold_metrics.append(
                                {
                                    "fold": int(fold_idx),
                                    "auc": None,
                                    "n_test": int(len(te_idx)),
                                    "n_trades": int(n_trades_fold),
                                    "trades_per_day": float(n_trades_fold) / float(max(days_span, 1.0)),
                                    "mean_return": float(mean_ret_fold),
                                    "net_pnl_per_trade": float(mean_ret_fold),
                                    "win_rate": float(win_rate_fold),
                                }
                            )
                        except Exception:
                            pass

                        # Skip calibration metrics for degenerate folds.
                        continue

                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    proba = model.predict_proba(X_te)
                    proba_pos = None
                    try:
                        arr = np.asarray(proba)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            proba_pos = arr[:, 1]
                        elif arr.ndim == 2 and arr.shape[1] == 1:
                            proba_pos = arr[:, 0]
                        elif arr.ndim == 1:
                            proba_pos = arr
                    except Exception:
                        proba_pos = None
                    if proba_pos is None:
                        proba_pos = np.full(int(len(X_te)), 0.5, dtype=float)
                    preds = np.asarray(proba_pos, dtype=float)

                    fold_auc_local = None
                    try:
                        if len(np.unique(y_te.values)) >= 2:
                            fold_auc_local = float(roc_auc_score(y_te, preds))
                            fold_aucs.append(float(fold_auc_local))
                    except Exception:
                        fold_auc_local = None

                    try:
                        preds_arr = preds.astype(float)
                        ret_te = final_returns_arr[te_idx]

                        gate_te = None
                        try:
                            if committee_gate_arr_for_l3 is not None:
                                gate_te = np.asarray(committee_gate_arr_for_l3, dtype=bool)[te_idx]
                        except Exception:
                            gate_te = None

                        from sklearn.isotonic import IsotonicRegression
                        iso = IsotonicRegression(out_of_bounds='clip')
                        train_proba = model.predict_proba(X_tr)
                        train_pos = None
                        try:
                            train_arr = np.asarray(train_proba)
                            if train_arr.ndim == 2 and train_arr.shape[1] >= 2:
                                train_pos = train_arr[:, 1]
                            elif train_arr.ndim == 2 and train_arr.shape[1] == 1:
                                train_pos = train_arr[:, 0]
                            elif train_arr.ndim == 1:
                                train_pos = train_arr
                        except Exception:
                            train_pos = None
                        if train_pos is None:
                            train_pos = np.full(int(len(X_tr)), float(np.mean(y_tr.values.astype(float))), dtype=float)
                        iso.fit(np.asarray(train_pos, dtype=float), y_tr.values)
                        preds_cal = iso.predict(preds_arr)

                        # Save calibrated OOF preds
                        try:
                            oof_pred_cal[te_idx] = preds_cal
                        except Exception:
                            pass

                        try:
                            trade_mask_fold = np.asarray(preds_cal, dtype=float) >= 0.5
                            try:
                                if gate_te is not None:
                                    trade_mask_fold = trade_mask_fold & np.asarray(gate_te, dtype=bool)
                            except Exception:
                                pass
                            ret_sel = np.asarray(ret_te, dtype=float)[trade_mask_fold]
                            ret_sel = ret_sel[np.isfinite(ret_sel)]
                            n_trades_fold = int(ret_sel.size)
                            mean_ret_fold = float(np.mean(ret_sel)) if n_trades_fold > 0 else 0.0
                            win_rate_fold = float(np.mean(ret_sel > 0.0)) if n_trades_fold > 0 else 0.0
                            per_fold_metrics.append(
                                {
                                    "fold": int(fold_idx),
                                    "auc": float(fold_auc_local) if fold_auc_local is not None else None,
                                    "n_test": int(len(te_idx)),
                                    "n_trades": int(n_trades_fold),
                                    "trades_per_day": float(n_trades_fold) / float(max(days_span, 1.0)),
                                    "mean_return": float(mean_ret_fold),
                                    "net_pnl_per_trade": float(mean_ret_fold),
                                    "win_rate": float(win_rate_fold),
                                }
                            )
                        except Exception:
                            pass

                        # Track calibration metrics (model-dependent penalty)
                        try:
                            brier, ece = compute_brier_and_ece(y_te.values, preds_cal)
                            if brier is not None and np.isfinite(brier):
                                fold_briers.append(float(brier))
                            if ece is not None and np.isfinite(ece):
                                fold_eces.append(float(ece))
                        except Exception:
                            pass

                        sized_returns = []
                        for ii, (prob, ret) in enumerate(zip(preds_cal, ret_te)):
                            size = directional_size_from_prob(
                                float(prob),
                                direction=direction,
                                thr=0.5,
                                max_exposure=1.0,
                                scale=1.0,
                            )
                            if gate_te is not None and (not bool(gate_te[ii])):
                                size = 0.0
                            sized_returns.append(size * float(ret))

                        sized_returns_arr = np.asarray(sized_returns, dtype=float)
                        sized_returns_arr = sized_returns_arr[np.isfinite(sized_returns_arr)]
                        if sized_returns_arr.size > 1 and float(np.std(sized_returns_arr)) > 1e-9:
                            fold_sharpe = float(np.mean(sized_returns_arr) / np.std(sized_returns_arr))
                        else:
                            fold_sharpe = 0.0
                        try:
                            fold_sharpe = float(np.clip(float(fold_sharpe), -20.0, 20.0))
                        except Exception:
                            fold_sharpe = 0.0

                        fold_sharpes.append(float(fold_sharpe))
                    except Exception:
                        pass

                if len(fold_sharpes) < 2:
                    # Not enough folds for stable Sharpe; still return a structured payload.
                    return -1.0, {
                        "valid_folds": int(len(fold_sharpes)),
                        "fold_aucs": [float(v) for v in fold_aucs],
                        "fold_sharpes": [float(v) for v in fold_sharpes],
                        "per_fold_metrics": per_fold_metrics,
                        "per_regime_metrics": {},
                    }

                mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5

                # Predicted trade density (same threshold as sizing) for reporting.
                try:
                    pred_trade_mask = np.isfinite(oof_pred_cal) & (oof_pred_cal >= 0.5)
                    try:
                        if committee_gate_arr_for_l3 is not None:
                            pred_trade_mask = pred_trade_mask & np.asarray(committee_gate_arr_for_l3, dtype=bool)
                    except Exception:
                        pass
                    n_pred_trades = int(np.sum(pred_trade_mask))
                except Exception:
                    n_pred_trades = int(len(final_returns_arr))
                trades_per_day = float(n_pred_trades) / float(max(days_span, 1))

                mean_brier = float(np.mean(fold_briers)) if fold_briers else None
                mean_ece = float(np.mean(fold_eces)) if fold_eces else None

                per_regime_metrics: Dict[str, Any] = {}
                try:
                    regime_labels = _build_event_regime_labels(
                        market_data=market_data,
                        event_index=y_final.index,
                        config=config,
                    )
                    per_regime_metrics = {
                        "volatility": _compute_metrics_by_regime(
                            y_true=final_labels_arr,
                            probs=np.asarray(oof_pred_cal, dtype=float),
                            returns=np.asarray(final_returns_arr, dtype=float),
                            base_thr=0.5,
                            transaction_cost=0.0,
                            regime_labels=regime_labels.get("volatility_regime"),
                            days_span=float(days_span),
                        ),
                        "trend": _compute_metrics_by_regime(
                            y_true=final_labels_arr,
                            probs=np.asarray(oof_pred_cal, dtype=float),
                            returns=np.asarray(final_returns_arr, dtype=float),
                            base_thr=0.5,
                            transaction_cost=0.0,
                            regime_labels=regime_labels.get("trend_regime"),
                            days_span=float(days_span),
                        ),
                        "combined": _compute_metrics_by_regime(
                            y_true=final_labels_arr,
                            probs=np.asarray(oof_pred_cal, dtype=float),
                            returns=np.asarray(final_returns_arr, dtype=float),
                            base_thr=0.5,
                            transaction_cost=0.0,
                            regime_labels=regime_labels.get("combined_regime"),
                            days_span=float(days_span),
                        ),
                    }
                except Exception:
                    per_regime_metrics = {}

                sharpe_arr = np.asarray(fold_sharpes, dtype=float)
                sharpe_mean = float(np.mean(sharpe_arr))
                sharpe_std = float(np.std(sharpe_arr, ddof=1)) if len(sharpe_arr) > 1 else 0.0
                sharpe_min = float(np.min(sharpe_arr))
                sharpe_max = float(np.max(sharpe_arr))

                lambda_vol = 1.2
                w_auc = 1.0
                w_den = 0.0
                w_cal = 1.0

                base_score = sharpe_mean - (lambda_vol * sharpe_std)
                try:
                    base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
                except Exception:
                    base_norm = 0.0
                if not np.isfinite(base_norm):
                    base_norm = 0.0

                # Keep this aligned with calculate_hpo_utility() so diagnostics match the optimized objective.
                phi_auc = trapezoidal_gate(mean_auc, lower=0.52, sweet_spot=(0.56, 0.66), upper=0.72)
                # Density term is constant w.r.t model params; disable it in objective.
                phi_density = 1.0

                # Calibration modifier for diagnostics (kept consistent with calculate_hpo_utility)
                phi_cal = float("nan")
                cal = None
                if mean_brier is not None and np.isfinite(mean_brier):
                    cal = float(mean_brier)
                elif mean_ece is not None and np.isfinite(mean_ece):
                    cal = float(mean_ece)
                if cal is not None:
                    phi_cal = float(np.clip(1.0 - (cal / 1.0), 0.0, 1.0))
                modifier = (phi_auc ** w_auc) * ((phi_cal ** w_cal) if np.isfinite(phi_cal) else 1.0)

                utility = calculate_hpo_utility(
                    folds_sharpe=sharpe_arr,
                    auc=mean_auc,
                    trades_per_day=trades_per_day,
                    lambda_vol=lambda_vol,
                    w_auc=w_auc,
                    w_den=w_den,
                    calibration_brier=mean_brier,
                    calibration_ece=mean_ece,
                    w_cal=w_cal,
                )

                if not np.isfinite(float(utility)):
                    utility = -1.0

                q_details: Dict[str, Any] = {}
                try:
                    utility_adj, q_details = _apply_hpo_quality_penalty(
                        utility=float(utility),
                        returns=final_returns_arr,
                        labels=final_labels_arr,
                        exit_reasons=final_exit_reasons_arr,
                        durations=final_durations_arr,
                        horizon=12,
                        tx_cost=float(DEFAULT_TRANSACTION_COST),
                        config=config,
                    )
                    if bool(config.get("hpo_apply_quality_penalty_to_layer3", False)):
                        utility = utility_adj
                except Exception:
                    pass

                details = {
                    "utility": float(utility),
                    "quality_penalty": q_details,
                    "mean_auc": mean_auc,
                    "fold_aucs": fold_aucs,
                    "fold_sharpes": fold_sharpes,
                    "per_fold_metrics": per_fold_metrics,
                    "per_regime_metrics": per_regime_metrics,
                    "calibration_brier": mean_brier,
                    "calibration_ece": mean_ece,
                    "sharpe_mean": sharpe_mean,
                    "sharpe_std": sharpe_std,
                    "sharpe_min": sharpe_min,
                    "sharpe_max": sharpe_max,
                    "trades_per_day": trades_per_day,
                    "valid_folds": min(len(fold_aucs), len(fold_sharpes)),
                    "lambda_vol": lambda_vol,
                    "w_auc": w_auc,
                    "w_den": w_den,
                    "w_cal": w_cal,
                    "base_score": float(base_score),
                    "phi_auc": float(phi_auc),
                    "phi_density": float(phi_density),
                    "phi_cal": float(phi_cal) if np.isfinite(phi_cal) else float("nan"),
                    "modifier": float(modifier),
                }
                return float(utility), details
            except Exception as exc:
                try:
                    tprint_warning(f"⚠️ Layer 3 metrics computation failed: {exc}")
                except Exception:
                    pass
                return -1.0, {"error": str(exc), "utility": -1.0}

        def layer3_objective(model_params: Dict[str, Any]) -> float:
            utility, _ = _compute_layer3_metrics(model_params)
            return utility

        # Wrapper to adapt layer3_objective to HierarchicalParameterOptimizer signature
        def layer3_objective_wrapper(
            params: Dict[str, Any],
            X_train=None,
            y_train=None,
            X_val=None,
            y_val=None,
            model=None,
            cv_folds: int = 1,
            scoring_metric: str = "custom_balanced_score",
            **kwargs,
        ) -> float:
            return layer3_objective(params)

        tprint_info("🚀 Layer 3: Optimizing Model Hyperparameters using Smart Walker...")

        # Find the Smart Walker parameter group
        try:
            model_group = next(g for g in param_groups if g.name == "model_hyperparameters")
        except StopIteration:
            tprint_warning("⚠️ 'model_hyperparameters' group not found. Fallback to default params.")
            model_group = None

        if model_group:
            from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import StageConfig, OptimizationStage

            # For the standalone Smart Walker call, use a dependency-free copy of
            # the model_hyperparameters group so HierarchicalParameterOptimizer
            # does not enforce the original depends_on=["target_engineering"].
            walker_model_group = create_param_group(
                name=model_group.name,
                params=model_group.params,
                priority=model_group.priority,
                depends_on=[],
                description=model_group.description,
            )

            # Configure Smart Walker stage
            walker_stage_config = StageConfig(
                stage=OptimizationStage.SMART_WALKER,
                n_trials=150,
                enable_pruning=False
            )

            # Instantiate HierarchicalParameterOptimizer
            walker_optimizer = HierarchicalParameterOptimizer(
                param_groups=[walker_model_group],
                objective_func=layer3_objective_wrapper,
                stages=[OptimizationStage.SMART_WALKER],
                stage_configs={OptimizationStage.SMART_WALKER: walker_stage_config},
                cv_folds=3,
                direction="maximize",
                n_rounds=2,
                enable_final_refinement=False,
                verbose=True,
            )
            
            # Execute optimization
            # We pass X_final/y_final for shape diagnostics, but objective uses closures internally
            hpo_results = walker_optimizer.optimize(
                X_train=X_final,
                y_train=y_final,
                X_val=None,
                y_val=None,
                model=None,
            )
            
            # Extract results into compatible format
            all_history = []
            if hpo_results.group_results:
                 for gr in hpo_results.group_results:
                     all_history.extend(gr.all_trials)
            
            l3_result = {
                "best_params": hpo_results.best_params,
                "best_value": hpo_results.best_score,
                "history": all_history
            }
            
            best_model_params = l3_result.get("best_params", {})
            try:
                l3_best_value = float(l3_result.get("best_value", float("-inf")))
                if not np.isfinite(l3_best_value):
                    tprint_error(
                        "❌ Layer 3 (Smart Walker) produced non-finite best utility; treating as failure "
                        "(objective likely errored)."
                    )
                    return {"success": False, "error": "layer3_non_finite_utility"}
            except Exception:
                tprint_error(
                    "❌ Layer 3 (Smart Walker) best utility could not be validated; treating as failure."
                )
                return {"success": False, "error": "layer3_best_value_validation_failed"}
            tprint_success(
                f"✅ Layer 3 (Smart Walker) Complete. Best utility: {l3_result.get('best_value', 0):.4f}"
            )
        else:
             l3_result = {'best_value': -1.0, 'history': []}
             best_model_params = {}

        # Log component metrics for the best Layer 3 params
        best_metrics: Dict[str, Any] = {}
        try:
            if best_model_params:
                best_utility, best_metrics = _compute_layer3_metrics(best_model_params)
                try:
                    if not np.isfinite(float(best_utility)):
                        tprint_error(
                            "❌ Layer 3 best params produced non-finite utility during breakdown; treating as failure."
                        )
                        return {"success": False, "error": "layer3_best_params_non_finite"}
                except Exception:
                    tprint_error(
                        "❌ Layer 3 best params utility could not be validated during breakdown; treating as failure."
                    )
                    return {"success": False, "error": "layer3_best_params_validation_failed"}
                tprint_info(
                    "   Layer 3 metrics: "
                    f"utility={best_utility:.4f}, "
                    f"auc={best_metrics.get('mean_auc', 0.0):.4f}, "
                    f"folds={best_metrics.get('valid_folds', 0)}, "
                    f"trades_per_day={best_metrics.get('trades_per_day', 0.0):.2f}"
                )
                tprint_info(
                    "   Layer 3 Sharpe stats: "
                    f"mean={best_metrics.get('sharpe_mean', 0.0):.4f}, "
                    f"std={best_metrics.get('sharpe_std', 0.0):.4f}, "
                    f"min={best_metrics.get('sharpe_min', 0.0):.4f}, "
                    f"max={best_metrics.get('sharpe_max', 0.0):.4f}"
                )
                tprint_info(
                    "   Layer 3 gates: "
                    f"base_score={best_metrics.get('base_score', 0.0):.4f}, "
                    f"phi_auc={best_metrics.get('phi_auc', 0.0):.4f}, "
                    f"phi_density={best_metrics.get('phi_density', 0.0):.4f}, "
                    f"phi_cal={best_metrics.get('phi_cal', float('nan')):.4f}, "
                    f"modifier={best_metrics.get('modifier', 0.0):.4f}"
                )
                tprint_info(
                    "   Layer 3 calibration: "
                    f"brier={best_metrics.get('calibration_brier', float('nan'))}, "
                    f"ece={best_metrics.get('calibration_ece', float('nan'))}"
                )
        except Exception as l3_diag_exc:
            tprint_warning(f"   ⚠️ Failed to compute Layer 3 metrics breakdown: {l3_diag_exc}")

        # Save Layer 3 History
        l3_history_path: Optional[Path] = None
        l3_trials_path: Optional[Path] = None
        try:
            ts_l3 = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            l3_history_path = Path("outcomes") / f"hpo_layer3_history_{symbol}_{timeframe}_{ts_l3}.json"
            with open(l3_history_path, "w") as f:
                json.dump(l3_result.get("history", []), f, default=str, indent=4)
            tprint_info(f"   💾 Saved Layer 3 history to {l3_history_path}")

            # Persist per-trial metrics for correlation analysis
            try:
                trial_rows = []
                for trial in l3_result.get("history", []):
                    params = trial.get("params", {}) if isinstance(trial, dict) else {}
                    utility_t, metrics_t = _compute_layer3_metrics(params)
                    row = {
                        "utility": utility_t,
                        "mean_auc": metrics_t.get("mean_auc"),
                        "trades_per_day": metrics_t.get("trades_per_day"),
                        "calibration_brier": metrics_t.get("calibration_brier"),
                        "calibration_ece": metrics_t.get("calibration_ece"),
                        "sharpe_mean": metrics_t.get("sharpe_mean"),
                        "sharpe_std": metrics_t.get("sharpe_std"),
                        "sharpe_min": metrics_t.get("sharpe_min"),
                        "sharpe_max": metrics_t.get("sharpe_max"),
                        "valid_folds": metrics_t.get("valid_folds"),
                        "lambda_vol": metrics_t.get("lambda_vol"),
                        "w_auc": metrics_t.get("w_auc"),
                        "w_den": metrics_t.get("w_den"),
                        "w_cal": metrics_t.get("w_cal"),
                        "base_score": metrics_t.get("base_score"),
                        "base_norm": metrics_t.get("base_norm"),
                        "phi_auc": metrics_t.get("phi_auc"),
                        "phi_density": metrics_t.get("phi_density"),
                        "phi_cal": metrics_t.get("phi_cal"),
                        "modifier": metrics_t.get("modifier"),
                        "fold_aucs": json.dumps(metrics_t.get("fold_aucs", [])),
                        "fold_sharpes": json.dumps(metrics_t.get("fold_sharpes", [])),
                    }
                    for k, v in params.items():
                        row[f"param_{k}"] = v
                    trial_rows.append(row)

                if trial_rows:
                    trials_df = pd.DataFrame(trial_rows)
                    l3_trials_path = Path("outcomes") / f"hpo_layer3_trials_{symbol}_{timeframe}_{ts_l3}.csv"
                    trials_df.to_csv(l3_trials_path, index=False)
                    tprint_info(f"   💾 Saved Layer 3 trial metrics to {l3_trials_path}")
            except Exception as l3_trial_exc:
                tprint_warning(f"   ⚠️ Failed to save Layer 3 trial metrics: {l3_trial_exc}")
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to save Layer 3 history: {e}")

        try:
            l3_report = _write_hpo_stage_report(
                outcomes_dir=outcomes_dir,
                run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
                stage_id="layer3_model",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                best_params=dict(best_model_params) if isinstance(best_model_params, dict) else {},
                metrics={
                    **(dict(best_metrics) if isinstance(best_metrics, dict) else {}),
                    "feature_selection": dict(l3_feature_selection_info) if isinstance(l3_feature_selection_info, dict) else {},
                },
                search_space=layer3_search_space,
                trials_csv_path=l3_trials_path,
                history_json_path=l3_history_path,
            )
            hpo_stage_reports["layer3"] = l3_report
        except Exception as l3_report_exc:
            tprint_warning(f"   ⚠️ Failed to write Layer 3 report: {l3_report_exc}")

        # ------------------------------------------------------------------
        # 4. FINAL COMPILATION & REPORTING
        # ------------------------------------------------------------------
        full_best_params = {}
        full_best_params.update(best_kalman_params)
        full_best_params.update(best_weighting_params)
        full_best_params.update(best_trading_params)
        full_best_params.update(best_model_params)
        full_best_params["horizon_bars"] = 12
        full_best_params["min_event_spacing"] = 2

        # Layer 1 weights diagnostics based on final weights used for model training
        layer1_weights_mean = float("nan")
        layer1_weights_min = float("nan")
        layer1_weights_max = float("nan")
        layer1_weights_entropy = float("nan")
        layer1_weights_entropy_norm = float("nan")
        try:
            w_arr = np.asarray(final_weights, dtype=float)
            valid_mask = np.isfinite(w_arr)
            if valid_mask.any():
                w_valid = w_arr[valid_mask]
                layer1_weights_mean = float(w_valid.mean())
                layer1_weights_min = float(w_valid.min())
                layer1_weights_max = float(w_valid.max())
                w_sum = float(w_valid.sum())
                if w_sum > 0 and len(w_valid) > 1:
                    w_norm = w_valid / w_sum
                    entropy = -float(np.sum(w_norm * np.log(w_norm + 1e-12)))
                    max_entropy = np.log(float(len(w_norm)))
                    entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else float("nan")
                    layer1_weights_entropy = entropy
                    layer1_weights_entropy_norm = entropy_norm
            tprint_info(
                "   Layer 1 weights diagnostics — "
                f"mean={layer1_weights_mean:.4f}, min={layer1_weights_min:.4f}, "
                f"max={layer1_weights_max:.4f}, entropy={layer1_weights_entropy:.4f}, "
                f"entropy_norm={layer1_weights_entropy_norm:.4f}"
            )
        except Exception:
            tprint_warning("⚠️ Failed to compute Layer 1 weights diagnostics.")

        summary_data = [
            {
                "Layer": "0. Kalman",
                "Score": kalman_result.get("best_value", 0),
                "AUC": None,
                "Trades/Day": None,
                "SharpeMean": None,
                "SharpeStd": None,
                "Loss": kalman_loss,
                "Params": str(best_kalman_params),
            },
            {
                "Layer": "1. Weighting",
                "Score": None,
                "AUC": None,
                "Trades/Day": None,
                "SharpeMean": None,
                "SharpeStd": None,
                "Params": str(best_weighting_params),
            },
            {
                "Layer": "2. Trading",
                "Score": l2_metrics.get("utility", best_l2_score),
                "AUC": l2_metrics.get("auc"),
                "Trades/Day": l2_metrics.get("trades_per_day"),
                "SharpeMean": l2_metrics.get("sharpe_mean"),
                "SharpeStd": l2_metrics.get("sharpe_std"),
                "Params": str(best_trading_params),
            },
            {
                "Layer": "3. Model",
                "Score": best_metrics.get("utility", l3_result.get("best_value", 0) if 'best_metrics' in locals() else 0),
                "AUC": best_metrics.get("mean_auc", l3_result.get("best_value", 0) if 'best_metrics' in locals() else 0),
                "Trades/Day": best_metrics.get("trades_per_day", None),
                "SharpeMean": best_metrics.get("sharpe_mean", None),
                "SharpeStd": best_metrics.get("sharpe_std", None),
                "Params": str(best_model_params),
            },
        ]
        ts_run = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # Persist CSV/MD reports with richer metrics
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(outcomes_dir / f"hpo_multi_stage_summary_{symbol}_{ts_run}.csv", index=False)

        md_path = outcomes_dir / f"hpo_multi_stage_report_{symbol}_{ts_run}.md"
        with open(md_path, "w") as f:
            f.write(f"# Multi-Stage HPO Report: {symbol}\n\n")
            kalman_score = kalman_result.get("best_value", 0)
            try:
                kalman_score_str = f"{float(kalman_score):.4f}"
            except Exception:
                kalman_score_str = "n/a"
            f.write(f"## Stage 0 (Kalman)\n")
            f.write(f"- score (optimizer best_value = -loss+amp_bonus): {kalman_score_str}\n")
            try:
                kalman_loss_str = f"{float(kalman_loss):.4f}"
            except Exception:
                kalman_loss_str = "n/a"
            f.write(f"- loss: {kalman_loss_str}\n")
            f.write(
                f"- smooth: {kalman_loss_details.get('smooth', float('nan')):.4f}, "
                f"track: {kalman_loss_details.get('track', float('nan')):.4f}, "
                f"amp: {kalman_loss_details.get('amp', float('nan')):.4f}, "
                f"amp_ratio: {kalman_loss_details.get('amp_ratio', float('nan')):.3f}\n"
            )
            f.write(f"```json\n{json.dumps(best_kalman_params, indent=2)}\n```\n\n")

            f.write(f"## Layer 1 (Weighting)\n")
            f.write(f"```json\n{json.dumps(best_weighting_params, indent=2)}\n```\n")
            f.write(
                f"- weights_mean: {layer1_weights_mean:.4f}, "
                f"min: {layer1_weights_min:.4f}, max: {layer1_weights_max:.4f}\n"
            )
            f.write(
                f"- weights_entropy: {layer1_weights_entropy:.4f}, "
                f"entropy_norm: {layer1_weights_entropy_norm:.4f}\n\n"
            )

            f.write(f"## Layer 2 (Trading)\n")
            f.write(f"- utility: {l2_metrics.get('utility', best_l2_score):.4f}\n")
            f.write(f"- auc: {l2_metrics.get('auc', 0.0):.4f}\n")
            f.write(f"- trades_per_day: {l2_metrics.get('trades_per_day', 0.0):.2f}\n")
            f.write(
                f"- sharpe_mean: {l2_metrics.get('sharpe_mean', 0.0):.4f}, "
                f"sharpe_std: {l2_metrics.get('sharpe_std', 0.0):.4f}, "
                f"sharpe_min: {l2_metrics.get('sharpe_min', 0.0):.4f}, "
                f"sharpe_max: {l2_metrics.get('sharpe_max', 0.0):.4f}\n"
            )
            f.write(f"```json\n{json.dumps(best_trading_params, indent=2)}\n```\n\n")

            try:
                pf = l2_metrics.get("per_fold_metrics")
                if isinstance(pf, list) and pf:
                    f.write("### Layer 2 - Per-fold metrics\n\n")
                    f.write("| fold | auc | n_test | n_trades | trades_per_day | mean_return | win_rate | sharpe |\n")
                    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                    for row in pf:
                        if not isinstance(row, dict):
                            continue
                        f.write(
                            f"| {int(row.get('fold', -1))} | "
                            f"{row.get('auc', float('nan')) if row.get('auc') is not None else float('nan'):.4f} | "
                            f"{int(row.get('n_test', 0))} | {int(row.get('n_trades', 0))} | "
                            f"{float(row.get('trades_per_day', 0.0)):.3f} | {float(row.get('mean_return', 0.0)):.6f} | "
                            f"{float(row.get('win_rate', 0.0)):.3f} | {float(row.get('sharpe', 0.0)):.3f} |\n"
                        )
                    f.write("\n")
            except Exception:
                pass

            try:
                prm = l2_metrics.get("per_regime_metrics")
                if isinstance(prm, dict) and prm:
                    for group_name in ["volatility", "trend", "combined"]:
                        grp = prm.get(group_name)
                        if not isinstance(grp, dict) or not grp:
                            continue
                        f.write(f"### Layer 2 - Per-regime ({group_name})\n\n")
                        f.write("| regime | n_events | n_trades | trades_per_day | mean_return | win_rate | sharpe | auc |\n")
                        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                        for reg, row in grp.items():
                            if not isinstance(row, dict):
                                continue
                            auc_val = row.get('auc')
                            auc_str = f"{float(auc_val):.4f}" if auc_val is not None and np.isfinite(float(auc_val)) else ""
                            f.write(
                                f"| {str(reg)} | {int(row.get('n_events', 0))} | {int(row.get('n_trades', 0))} | "
                                f"{float(row.get('trades_per_day', 0.0)):.3f} | {float(row.get('mean_return', 0.0)):.6f} | "
                                f"{float(row.get('win_rate', 0.0)):.3f} | {float(row.get('sharpe', 0.0)):.3f} | {auc_str} |\n"
                            )
                        f.write("\n")
            except Exception:
                pass

            f.write(f"## Layer 3 (Model)\n")
            f.write(
                f"- utility: {best_metrics.get('utility', l3_result.get('best_value', 0)):.4f}\n"
                f"- auc: {best_metrics.get('mean_auc', 0.0):.4f}\n"
                f"- folds: {best_metrics.get('valid_folds', 0)}\n"
                f"- trades_per_day: {best_metrics.get('trades_per_day', 0.0):.2f}\n"
                f"- sharpe_mean: {best_metrics.get('sharpe_mean', 0.0):.4f}, "
                f"sharpe_std: {best_metrics.get('sharpe_std', 0.0):.4f}, "
                f"sharpe_min: {best_metrics.get('sharpe_min', 0.0):.4f}, "
                f"sharpe_max: {best_metrics.get('sharpe_max', 0.0):.4f}\n"
            )
            try:
                fs_info = l3_feature_selection_info if isinstance(l3_feature_selection_info, dict) else {}
                f.write(
                    f"- feature_selection_enabled: {bool(fs_info.get('enabled', False))}\n"
                    f"- feature_selection_before: {int(fs_info.get('n_features_before', 0))}\n"
                    f"- feature_selection_after: {int(fs_info.get('n_features_after', 0))}\n"
                )
                f.write(f"```json\n{json.dumps(fs_info, indent=2)}\n```\n\n")
            except Exception:
                pass
            f.write(f"```json\n{json.dumps(best_model_params, indent=2)}\n```\n\n")

            try:
                pf = best_metrics.get("per_fold_metrics")
                if isinstance(pf, list) and pf:
                    f.write("### Layer 3 - Per-fold metrics\n\n")
                    f.write("| fold | auc | n_test | n_trades | trades_per_day | mean_return | win_rate |\n")
                    f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
                    for row in pf:
                        if not isinstance(row, dict):
                            continue
                        f.write(
                            f"| {int(row.get('fold', -1))} | "
                            f"{row.get('auc', float('nan')) if row.get('auc') is not None else float('nan'):.4f} | "
                            f"{int(row.get('n_test', 0))} | {int(row.get('n_trades', 0))} | "
                            f"{float(row.get('trades_per_day', 0.0)):.3f} | {float(row.get('mean_return', 0.0)):.6f} | "
                            f"{float(row.get('win_rate', 0.0)):.3f} |\n"
                        )
                    f.write("\n")
            except Exception:
                pass

            try:
                prm = best_metrics.get("per_regime_metrics")
                if isinstance(prm, dict) and prm:
                    for group_name in ["volatility", "trend", "combined"]:
                        grp = prm.get(group_name)
                        if not isinstance(grp, dict) or not grp:
                            continue
                        f.write(f"### Layer 3 - Per-regime ({group_name})\n\n")
                        f.write("| regime | n_events | n_trades | trades_per_day | mean_return | win_rate | sharpe | auc |\n")
                        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                        for reg, row in grp.items():
                            if not isinstance(row, dict):
                                continue
                            auc_val = row.get('auc')
                            auc_str = f"{float(auc_val):.4f}" if auc_val is not None and np.isfinite(float(auc_val)) else ""
                            f.write(
                                f"| {str(reg)} | {int(row.get('n_events', 0))} | {int(row.get('n_trades', 0))} | "
                                f"{float(row.get('trades_per_day', 0.0)):.3f} | {float(row.get('mean_return', 0.0)):.6f} | "
                                f"{float(row.get('win_rate', 0.0)):.3f} | {float(row.get('sharpe', 0.0)):.3f} | {auc_str} |\n"
                            )
                        f.write("\n")
            except Exception:
                pass

        json_path = outcomes_dir / f"hpo_multi_stage_best_params_{symbol}_{ts_run}.json"
        with open(json_path, "w") as f:
            json.dump(full_best_params, f, indent=2, default=str)

        metrics = {
            "layer0_ic": kalman_result.get("best_value", 0),
            "layer1_weights_mean": layer1_weights_mean,
            "layer1_weights_min": layer1_weights_min,
            "layer1_weights_max": layer1_weights_max,
            "layer1_weights_entropy": layer1_weights_entropy,
            "layer1_weights_entropy_norm": layer1_weights_entropy_norm,
            "layer2_score": best_l2_score,
            "layer3_auc": l3_result.get("best_value", 0),
            "layer3_feature_selection": l3_feature_selection_info,
            "best_params": full_best_params,
            "stage_reports": hpo_stage_reports,
        }

        tprint_success(f"✅ Multi-Layer HPO Complete. Final metrics: {metrics}")
        # Multi-layer HPO returns early with its results
        return {"success": True, "metrics": metrics, "artifacts": {"best_params_json": str(json_path)}}

        # ------------------------------------------------------------------
        # 4) Multi-objective wrapper for single-objective optimizers
        # ------------------------------------------------------------------
        def create_scalar_objective_wrapper(
            model_complexity: str = "fast",
            use_ensemble: bool = False,
            compute_diagnostics: bool = False,
        ) -> callable:
            """Create a wrapper with specific complexity settings.

            BayesianTPEOptimizer always calls the objective as objective(params), so
            we close over X_dummy / y_dummy and complexity settings.
            """
            def wrapper(params: Dict[str, Any]) -> float:
                result = labeling_objective(
                    params, X_dummy, y_dummy,
                    model_complexity=model_complexity,
                    use_ensemble=use_ensemble,
                    compute_diagnostics=compute_diagnostics,
                )
                if isinstance(result, dict):
                    return float(result.get('edge', result.get('combined', 0.0)))
                return float(result)
            return wrapper

        # ------------------------------------------------------------------
        # 5) Convert param_groups to search space
        # ------------------------------------------------------------------
        tprint_info("🚀 Using Multi-Stage Bayesian TPE optimization")

        # Convert param_groups (ParameterGroup instances) to a flat Optuna-style search space
        initial_search_space: Dict[str, Dict[str, Any]] = {}
        for group in param_groups:
            for param_name, param_spec in group.params.items():
                initial_search_space[param_name] = param_spec

        if warm_start_candidates_df is not None and not warm_start_candidates_df.empty:
            try:
                shrinkable_params = [
                    "profit_thr_base",
                    "stop_to_profit_ratio",
                    "horizon_bars",
                    "min_event_spacing",
                    "vol_baseline_window",
                    "profit_mult_min",
                    "profit_mult_max",
                    "stop_mult_min",
                    "stop_mult_max",
                    "iso_min_prob",
                    "target_clip_high_q",
                    "econ_min_return_multiple",
                    "label_low_q",
                    "label_high_q",
                    "signal_strength_scale_max",
                    # NEW parameters for trade count control
                    "cusum_threshold",
                    "target_signal_density",
                    "r_multiple_pos_threshold",
                    "transaction_cost_mult",
                    "kalman_Q",
                    "kalman_R",
                ]
                for p in shrinkable_params:
                    if p not in initial_search_space:
                        continue
                    if p not in warm_start_candidates_df.columns:
                        continue
                    spec = initial_search_space.get(p, {})
                    if not isinstance(spec, dict):
                        continue
                    ptype = spec.get("type", "float")
                    if ptype not in ["float", "int"]:
                        continue
                    series = warm_start_candidates_df[p].dropna()
                    if series.empty or len(series) < 20:
                        continue
                    try:
                        q_low = float(series.quantile(0.10))
                        q_high = float(series.quantile(0.90))
                    except Exception:
                        continue
                    low = spec.get("low")
                    high = spec.get("high")
                    if low is None or high is None:
                        continue
                    new_low = max(float(low), q_low)
                    new_high = min(float(high), q_high)
                    if new_high <= new_low:
                        continue
                    new_spec = spec.copy()
                    if ptype == "int":
                        new_spec["low"] = int(new_low)
                        new_spec["high"] = int(max(new_high, new_low + 1))
                    else:
                        new_spec["low"] = float(new_low)
                        new_spec["high"] = float(new_high)
                    initial_search_space[p] = new_spec
                tprint_info("📌 Warm-start: narrowed search space around previous candidate quantiles")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 6) Multi-Stage HPO Execution Loop
        # ------------------------------------------------------------------
        # Parameters optimized per stage:
        # Stage 1: horizon_bars, min_event_spacing, target_transform (iso/clipping,
        #          econ_min_return_multiple, label quantiles, signal strength
        #          scaling) + kalman_Q, kalman_R, vol_baseline_window,
        #          profit_mult_min/max, stop_mult_min/max
        # Stage 2: kalman_Q, kalman_R, vol_baseline_window, profit_mult_min/max,
        #          stop_mult_min/max
        # Stage 3: All parameters (iso_min_prob, target_transform refinements, etc.)
        #          NOTE: profit_thr_base, stop_to_profit_ratio now handled by Kalman ensemble
        # Stage 4: Filtering parameters only (label_low_q, label_high_q, econ_min_return_multiple, etc.)
        #          with fixed structural parameters.
        #
        # The multi-stage process progressively increases model complexity to find
        # configurations that are both profitable AND learnable by production models.

        # Define which parameters to optimize at each stage
        if calibrated_horizon is not None:
            stage_1_params = [
                'min_event_spacing',
                'cusum_threshold', 'target_signal_density',
                # 'profit_thr_base', 'stop_to_profit_ratio',  # Now handled by Kalman ensemble
                'trail_distance', 'consensus_threshold',
                'iso_min_prob', 'target_clip_high_q',
                'econ_min_return_multiple', 'label_low_q', 'label_high_q',
                'signal_strength_scale_max',
                'r_multiple_pos_threshold', 'transaction_cost_mult',
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max',
            ]
            stage_2_params = [
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'
            ]
        else:
            stage_1_params = [
                'horizon_bars', 'min_event_spacing',
                'cusum_threshold', 'target_signal_density',
                # 'profit_thr_base', 'stop_to_profit_ratio',  # Now handled by Kalman ensemble
                'trail_distance', 'consensus_threshold',
                'iso_min_prob', 'target_clip_high_q',
                'econ_min_return_multiple', 'label_low_q', 'label_high_q',
                'signal_strength_scale_max',
                'r_multiple_pos_threshold', 'transaction_cost_mult',
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max',
            ]
            stage_2_params = [
                'kalman_Q', 'kalman_R', 'vol_baseline_window',
                'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'
            ]

        # Stage 4 specific params (filtering/labeling only)
        # ENHANCED: Include all trade count-affecting parameters for final refinement
        stage_4_params = [
            'label_low_q', 'label_high_q',
            'econ_min_return_multiple',
            'iso_min_prob', 'target_clip_high_q',
            'signal_strength_scale_max',
            'r_multiple_pos_threshold',  # NEW: R-multiple threshold
            'transaction_cost_mult',  # NEW: transaction cost sensitivity
            'cusum_threshold', 'target_signal_density',  # NEW: signal density
        ]

        # Stage 3 uses all parameters (optionally treating horizon_bars as fixed when calibrated)

        stages = STAGE_CONFIGS
        stage_results: List[Dict[str, Any]] = []
        all_trials_count = 0
        best_overall_score = float('-inf')
        best_overall_params = {}

        # Track best params from each stage to use as defaults for fixed params
        accumulated_best_params: Dict[str, Any] = {}
        if calibrated_horizon is not None:
            accumulated_best_params["horizon_bars"] = int(calibrated_horizon)

        for stage_idx, stage in enumerate(stages):
            tprint_info(f"🚀 Starting {stage['name']} with complexity={stage['complexity']}...")

            # Determine which parameters to optimize in this stage
            if stage_idx == 0:  # Stage 1
                active_params = stage_1_params
            elif stage_idx == 1:  # Stage 2
                active_params = stage_2_params
            elif stage_idx == 2:  # Stage 3 - all parameters
                if calibrated_horizon is not None:
                    active_params = [
                        k for k in initial_search_space.keys()
                        if k != 'horizon_bars'
                    ]
                else:
                    active_params = list(initial_search_space.keys())
            else:  # Stage 4 - Labeling Refinement
                active_params = stage_4_params

            # Create stage-specific search space
            current_search_space = {
                k: v for k, v in initial_search_space.items()
                if k in active_params
            }

            tprint_info(f"   Optimizing parameters: {list(current_search_space.keys())}")

            # Create objective wrapper that merges optimized params with fixed params from previous stages
            def create_stage_objective_wrapper(
                model_complexity: str,
                use_ensemble: bool,
                compute_diagnostics: bool,
                fixed_params: Dict[str, Any],
                use_stage1_subsample: bool,
                stage_name: str,
            ) -> callable:
                """Create a wrapper that injects fixed params from previous stages."""
                def wrapper(params: Dict[str, Any]) -> float:
                    nonlocal market_data, primary_signals, volatility_1d, days_span, atr_series
                    # Track candidate_pool length before/after to detect append vs skip
                    pool_size_before = len(candidate_pool)
                    if use_stage1_subsample and model_complexity == "fast" and stage1_enable_subsample:
                        md_backup = market_data
                        ps_backup = primary_signals
                        vol_backup = volatility_1d
                        atr_backup = atr_series
                        days_backup = days_span
                        try:
                            market_data = stage1_market_data
                            primary_signals = stage1_primary_signals
                            volatility_1d = stage1_volatility_1d
                            atr_series = stage1_atr_series
                            days_span = stage1_days_span
                            full_params = {**fixed_params, **params}
                            result = labeling_objective(
                                full_params, X_dummy, y_dummy,
                                model_complexity=model_complexity,
                                use_ensemble=use_ensemble,
                                compute_diagnostics=compute_diagnostics,
                            )
                        finally:
                            market_data = md_backup
                            primary_signals = ps_backup
                            volatility_1d = vol_backup
                            atr_series = atr_backup
                            days_span = days_backup
                    else:
                        full_params = {**fixed_params, **params}
                        result = labeling_objective(
                            full_params, X_dummy, y_dummy,
                            model_complexity=model_complexity,
                            use_ensemble=use_ensemble,
                            compute_diagnostics=compute_diagnostics,
                        )

                    # Derive scalar objective value (edge-first, fallback to combined)
                    if isinstance(result, dict):
                        edge_val = float(result.get('edge', result.get('combined', 0.0)))
                        combined_val = float(result.get('combined', edge_val))
                    else:
                        edge_val = float(result)
                        combined_val = float('nan')

                    # Inspect candidate_pool to see if this trial produced a candidate
                    pool_size_after = len(candidate_pool)
                    if pool_size_after > pool_size_before:
                        # Last entry should correspond to this trial
                        last_cand = candidate_pool[-1]
                        mean_auc_cand = last_cand.get('mean_auc')
                        n_events_cand = last_cand.get('n_events')
                        tprint_info(
                            f"[HPO_TRIAL] stage={stage_name} complexity={model_complexity} "
                            f"decision=APPEND edge={edge_val:.6f} combined={combined_val:.6f} "
                            f"mean_auc={mean_auc_cand:.6f} n_events={n_events_cand} "
                            f"pool_size={pool_size_after}"
                        )
                    else:
                        tprint_info(
                            f"[HPO_TRIAL] stage={stage_name} complexity={model_complexity} "
                            f"decision=SKIP edge={edge_val:.6f} combined={combined_val:.6f} "
                            f"pool_size={pool_size_after}"
                        )

                    return edge_val
                return wrapper

            if stage_idx in (0, 1):
                stage_param_groups: list[list[str]] = []
                if stage_idx == 0:
                    # Stage 1 (fast model, subsampled data):
                    # Group A – event shape / density + signal generation
                    if calibrated_horizon is not None:
                        stage_param_groups.append(['min_event_spacing', 'cusum_threshold', 'target_signal_density', 'trail_distance'])
                    else:
                        stage_param_groups.append(['horizon_bars', 'min_event_spacing', 'cusum_threshold', 'target_signal_density', 'trail_distance'])
                    # Group B – TPSL geometry
                    stage_param_groups.append(['profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'])
                    # Group C – smoothing
                    stage_param_groups.append(['kalman_Q', 'kalman_R'])
                    # Group D – target transform (clipping, econ floor, label quantiles,
                    # and signal-strength weighting strength) + trade filters
                    stage_param_groups.append([
                        'iso_min_prob', 'target_clip_high_q',
                        'econ_min_return_multiple', 'label_low_q', 'label_high_q',
                        'signal_strength_scale_max',
                        'r_multiple_pos_threshold', 'transaction_cost_mult',  # NEW
                    ])
                else:
                    # Stage 2 (medium model, full data): smoothing + TPSL/vol
                    stage_param_groups.append(['kalman_Q', 'kalman_R'])
                    stage_param_groups.append(['vol_baseline_window', 'profit_mult_min', 'profit_mult_max', 'stop_mult_min', 'stop_mult_max'])

                stage_best_score = float('-inf')
                stage_best_params: Dict[str, Any] = {}
                stage_trials_total = 0

                n_groups = max(1, len(stage_param_groups))

                for group_idx, group_params in enumerate(stage_param_groups):
                    group_search_space = {
                        k: v for k, v in current_search_space.items()
                        if k in group_params
                    }
                    if not group_search_space:
                        continue

                    # Fixed params: best from previous stages and previous groups in this stage
                    group_fixed_params = {
                        k: v for k, v in accumulated_best_params.items()
                        if k not in group_search_space
                    }

                    group_objective = create_stage_objective_wrapper(
                        model_complexity=stage["complexity"],
                        use_ensemble=(stage["complexity"] == "strong"),
                        compute_diagnostics=(stage_idx == len(stages) - 1 and group_idx == len(stage_param_groups) - 1),
                        fixed_params=group_fixed_params,
                        use_stage1_subsample=(stage_idx == 0),
                        stage_name=stage["name"],
                    )

                    group_n_trials = max(5, stage["n_trials"] // n_groups)

                    group_config = OptimizationConfig(
                        n_trials=group_n_trials,
                        execution_mode=config.get("execution_mode", "full"),
                        direction='maximize',
                        # For small groups, use direct TPE (no internal grid) for speed
                        enable_staged_optimization=False,
                        coarse_grid_trials=0,
                        fine_grid_trials=0,
                        tpe_trials=group_n_trials,
                        enable_hardware_optimization=False,
                        enable_vectorbt_optimization=False,
                        early_stopping_patience=max(5, group_n_trials // 3),
                        early_stopping_threshold=None,
                        seed=42 + stage_idx * 10 + group_idx,
                    )

                    group_optimizer = BayesianTPEOptimizer(config=group_config)

                    tprint_info(
                        f"   → Group {group_idx + 1}/{len(stage_param_groups)} params={list(group_search_space.keys())}, "
                        f"trials={group_n_trials}"
                    )

                    try:
                        group_result = group_optimizer.optimize(
                            objective=group_objective,
                            search_space=group_search_space,
                        )
                        group_best_params = group_result.get('best_params', {})
                        group_best_score = group_result.get('best_value', 0.0)
                        group_trials = int(group_result.get('n_trials', group_n_trials))

                        stage_trials_total += group_trials

                        if group_best_score > stage_best_score:
                            stage_best_score = group_best_score

                        # Accumulate params from this group
                        accumulated_best_params.update(group_best_params)
                        stage_best_params.update(group_best_params)

                        tprint_success(
                            f"   ✅ Group {group_idx + 1}/{len(stage_param_groups)} complete: "
                            f"best_score={group_best_score:.4f}, trials={group_trials}"
                        )

                    except Exception as group_exc:
                        tprint_warning(
                            f"   ⚠️ Stage {stage['name']} group {group_idx + 1} failed: {group_exc}"
                        )
                        import traceback
                        traceback.print_exc()
                        continue

                if stage_best_score == float('-inf'):
                    # No successful groups; fall back to next stage
                    continue

                all_trials_count += stage_trials_total

                if stage_best_score > best_overall_score:
                    best_overall_score = stage_best_score
                    best_overall_params = stage_best_params.copy()

                stage_results.append({
                    'stage': stage['name'],
                    'complexity': stage['complexity'],
                    'best_score': stage_best_score,
                    'best_params': stage_best_params,
                    'trials': stage_trials_total,
                })

                # For Stage 1 (index 0), shrink Stage 2 search space using fast-model candidates
                if stage_idx == 0:
                    stage_candidates_fast = [
                        c for c in candidate_pool
                        if c.get('model_complexity') == stage['complexity']
                    ]
                    if stage_candidates_fast:
                        try:
                            initial_search_space = shrink_search_space(
                                original_space=initial_search_space,
                                previous_results=stage_candidates_fast,
                                top_k=stage['top_k_to_pass'],
                            )
                            tprint_info(
                                f"   📉 Narrowed Stage 2 search space based on "
                                f"Top {min(len(stage_candidates_fast), stage['top_k_to_pass'])} candidates"
                            )
                        except Exception:
                            pass

                # For Stage 2 (index 1), shrink Stage 3 search space using medium-model candidates
                if stage_idx == 1:
                    stage_candidates = [
                        c for c in candidate_pool
                        if c.get('model_complexity') == stage['complexity']
                    ]
                    if stage_candidates:
                        try:
                            initial_search_space = shrink_search_space(
                                original_space=initial_search_space,
                                previous_results=stage_candidates,
                                top_k=stage['top_k_to_pass'],
                            )
                            tprint_info(
                                f"   📉 Narrowed Stage 3 search space based on "
                                f"Top {min(len(stage_candidates), stage['top_k_to_pass'])} candidates"
                            )
                        except Exception:
                            pass

                tprint_success(
                    f"   ✅ {stage['name']} complete (hierarchical groups): "
                    f"best_score={stage_best_score:.4f}, trials={stage_trials_total}"
                )

                # Done with this stage; move to next global stage
                continue

            # Get fixed params: best values from previous stages for params not being optimized
            fixed_params = {
                k: v for k, v in accumulated_best_params.items()
                if k not in current_search_space
            }

            # Create stage-level objective for non-grouped stages
            stage_objective = create_stage_objective_wrapper(
                model_complexity=stage["complexity"],
                use_ensemble=(stage["complexity"] == "strong"),
                compute_diagnostics=(stage_idx == len(stages) - 1),
                fixed_params=fixed_params,
                use_stage1_subsample=False,
                stage_name=stage["name"],
            )

            bayesian_config = OptimizationConfig(
                n_trials=stage["n_trials"],
                execution_mode=config.get("execution_mode", "full"),
                direction='maximize',
                # Staged optimization settings (use TPE directly for speed)
                enable_staged_optimization=(stage_idx == 0),  # Only use grid in first stage
                coarse_grid_trials=min(15, stage["n_trials"] // 5) if stage_idx == 0 else 0,
                fine_grid_trials=min(10, stage["n_trials"] // 10) if stage_idx == 0 else 0,
                tpe_trials=stage["n_trials"] - (25 if stage_idx == 0 else 0),
                # Disable hardware/VectorBT-specific acceleration
                enable_hardware_optimization=False,
                enable_vectorbt_optimization=False,
                # Early stopping per trial
                early_stopping_patience=max(5, stage["n_trials"] // 5),
                early_stopping_threshold=None,
                # Reproducibility
                seed=42 + stage_idx,
            )

            optimizer = BayesianTPEOptimizer(config=bayesian_config)

            # Run optimization for this stage
            tprint_info(f"   Running {stage['n_trials']} trials with {stage['complexity']} model...")

            try:
                result = optimizer.optimize(
                    objective=stage_objective,
                    search_space=current_search_space,
                )

                stage_best_params = result.get('best_params', {})
                stage_best_score = result.get('best_value', 0.0)
                stage_trials = result.get('total_trials', stage["n_trials"])

                all_trials_count += stage_trials

                # Track best overall
                if stage_best_score > best_overall_score:
                    best_overall_score = stage_best_score
                    best_overall_params = stage_best_params.copy()

                # Accumulate best params from this stage for use in subsequent stages
                accumulated_best_params.update(stage_best_params)

                tprint_success(
                    f"   ✅ {stage['name']} complete: "
                    f"best_score={stage_best_score:.4f}, trials={stage_trials}"
                )

                # Store stage results
                stage_results.append({
                    'stage': stage['name'],
                    'complexity': stage['complexity'],
                    'best_score': stage_best_score,
                    'best_params': stage_best_params,
                    'trials': stage_trials,
                })

                # Get candidates from this stage
                stage_candidates = [
                    c for c in candidate_pool
                    if c.get('model_complexity') == stage['complexity']
                ]

                # Identify best candidate from this stage for explicit reporting
                if stage_candidates:
                    best_cand = max(stage_candidates, key=lambda x: x.get('edge', x.get('combined', 0)))
                    tprint_info(
                        f"   🏆 Best candidate {stage['name']}: "
                        f"Edge={best_cand.get('edge', 0):.4f}, "
                        f"AUC={best_cand.get('mean_auc', 0):.3f}, "
                        f"Trades/Day={best_cand.get('trades_per_day', 0):.2f}, "
                        f"Raw={best_cand.get('n_raw_events', 0)} → Vol={best_cand.get('n_vol_scaled_events', 0)} → Final={best_cand.get('n_events', 0)}"
                    )

                # For 4-stage setup:
                # - After Stage 1 (index 0): Shrink space for Stage 2 using fast-model candidates.
                # - After Stage 2 (index 1): Shrink space for Stage 3 (handled above in group block or here if not group)
                # - After Stage 3 (index 2): No shrinking needed for Stage 4 because Stage 4 uses a specific subset
                #   of parameters (labeling only), and structural params are fixed via accumulated_best_params.
                if stage_idx == 0 and stage_candidates:
                    try:
                        initial_search_space = shrink_search_space(
                            original_space=initial_search_space,
                            previous_results=stage_candidates,
                            top_k=stage['top_k_to_pass'],
                        )
                        tprint_info(
                            f"   📉 Narrowed Stage 2 search space based on "
                            f"Top {min(len(stage_candidates), stage['top_k_to_pass'])} candidates"
                        )
                    except Exception:
                        pass

                if stage_idx == 1 and stage_candidates:
                    try:
                        initial_search_space = shrink_search_space(
                            original_space=initial_search_space,
                            previous_results=stage_candidates,
                            top_k=stage['top_k_to_pass'],
                        )
                        tprint_info(
                            f"   📉 Narrowed Stage 3 search space based on "
                            f"Top {min(len(stage_candidates), stage['top_k_to_pass'])} candidates"
                        )
                    except Exception:
                        pass

                # Per-stage early stopping: if no improvement in this stage, move to next
                # (This is handled by the optimizer's early_stopping_patience setting)

            except Exception as stage_exc:
                tprint_warning(f"   ⚠️ Stage {stage['name']} failed: {stage_exc}")
                import traceback
                traceback.print_exc()
                # Continue to next stage or use best so far
                continue

        # Final best from all stages
        best_params = best_overall_params
        best_score = best_overall_score

        tprint_info(f"   Total trials across all stages: {all_trials_count}")

        tprint_success(f"✅ Labeling HPO completed. Best edge={best_score:.6f}")
        tprint_info(f"Best parameters: {best_params}")

        # Extract best candidate's detailed metrics for summary
        best_candidate_metrics = {}
        try:
            if candidate_pool:
                best_candidate = None

                for cand in candidate_pool:
                    cand_params = cand.get("params", {})
                    if cand_params == best_params:
                        best_candidate = cand
                        break

                if best_candidate is None:
                    try:
                        target_edge = float(best_score)
                        tol = max(1e-9, abs(target_edge) * 1e-6)
                        for cand in candidate_pool:
                            edge_val = float(cand.get("edge", float("nan")))
                            if np.isfinite(edge_val) and abs(edge_val - target_edge) <= tol:
                                best_candidate = cand
                                break
                    except Exception:
                        best_candidate = None

                if best_candidate is None:
                    try:
                        best_candidate = max(
                            candidate_pool,
                            key=lambda c: c.get("edge", c.get("combined", float("-inf"))),
                        )
                    except Exception:
                        best_candidate = None

                if best_candidate is not None:
                    best_candidate_metrics = {
                        "mean_auc": float(best_candidate.get("mean_auc", 0.5)),
                        "trades_per_day": float(best_candidate.get("trades_per_day", 0.0)),
                        "learnability": float(best_candidate.get("learnability", 0.0)),
                        "profitability": float(best_candidate.get("profitability", 0.0)),
                        "sharpe_pos": float(best_candidate.get("sharpe_pos", 0.0)),
                        "balance_score": float(best_candidate.get("balance_score", 0.0)),
                        "n_events": int(best_candidate.get("n_events", 0)),
                    }
        except Exception as metric_exc:
            tprint_warning(f"⚠️ Failed to extract best candidate metrics: {metric_exc}")
            best_candidate_metrics = {}

        # ------------------------------------------------------------------
        # HPO METRIC CORRELATION ANALYSIS (User Request)
        # ------------------------------------------------------------------
        try:
            if len(candidate_pool) >= 10:
                tprint_info("📊 Computing HPO Metric Correlations (Fidelity vs Downstream Proxy)...")
                # Extract key metrics
                corr_data = []
                for c in candidate_pool:
                    c_metrics = {
                        'IC': float(c.get('ic', 0.0)),
                        'Fidelity': float(c.get('fidelity_score', 0.0)),
                        'AUC': float(c.get('mean_auc', 0.5)),
                        'Calibration': float(c.get('calibration_score', 0.0)),
                        'Edge': float(c.get('edge', 0.0)),
                        'Profit': float(c.get('profitability', 0.0)),
                        'Learnability': float(c.get('learnability', 0.0)),
                        'Combined': float(c.get('combined', 0.0)),
                    }
                    corr_data.append(c_metrics)

                df_corr = pd.DataFrame(corr_data)
                # Compute correlation matrix
                corr_matrix = df_corr.corr()

                tprint_info("\n" + "="*50)
                tprint_info("🎯 HPO OBJECTIVE CORRELATION MATRIX")
                tprint_info("="*50)
                tprint_info(str(corr_matrix.round(3)))

                # Highlight key relationships
                ic_edge = corr_matrix.loc['IC', 'Edge']
                auc_edge = corr_matrix.loc['AUC', 'Edge']
                calib_edge = corr_matrix.loc['Calibration', 'Edge']

                tprint_info("-" * 40)
                tprint_info(f"Does Fidelity predict Profitability (Edge)?")
                tprint_info(f"  • IC vs Edge:          {ic_edge:.3f} " + ("✅ Strong positive" if ic_edge > 0.3 else "⚠️ Weak/Negative" if ic_edge < 0 else ""))
                tprint_info(f"  • AUC vs Edge:         {auc_edge:.3f}")
                tprint_info(f"  • Calibration vs Edge: {calib_edge:.3f}")
                tprint_info("-" * 40 + "\n")
        except Exception as corr_exc:
            tprint_warning(f"⚠️ Correlation analysis failed: {corr_exc}")

        # ------------------------------------------------------------------
        # SAVE FULL TRIALS REPORT (User Request)
        # ------------------------------------------------------------------
        try:
            if candidate_pool:
                tprint_info("💾 Saving full HPO trials report...")
                # Flatten params into columns
                report_data = []
                for c in candidate_pool:
                    row = c.copy()
                    # Flatten params dict
                    if 'params' in row:
                        p_dict = row.pop('params')
                        for k, v in p_dict.items():
                            row[k] = v
                    report_data.append(row)

                df_report = pd.DataFrame(report_data)

                # Timestamped filename
                timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                report_path = outcomes_dir / f"meta_labeling_hpo_trials_{symbol}_{timestamp_str}.csv"
                df_report.to_csv(report_path, index=False)
                tprint_success(f"✅ Saved HPO trials report to: {report_path}")
        except Exception as report_exc:
            tprint_warning(f"⚠️ Failed to save trials report: {report_exc}")

        # ------------------------------------------------------------------
        # 7) COMPREHENSIVE DIAGNOSTICS FOR BEST CONFIG
        # ------------------------------------------------------------------
        # These diagnostics are computed ONLY for the best config, not during HPO
        tprint_info("🔍 Computing comprehensive diagnostics for best configuration...")

        best_config_diagnostics = {}
        try:
            # Re-run labeling with best params to get intermediate data
            diag_params = best_params.copy()

            # For diagnostics (including two-stage bagged meta-model), prefer the
            # full lookback window if available, even when HPO itself used a
            # shorter multi-slice subset.
            market_data_diag = None
            try:
                if "market_data_full_for_diagnostics" in locals() and isinstance(
                    market_data_full_for_diagnostics, pd.DataFrame
                ):
                    market_data_diag = market_data_full_for_diagnostics.copy()
            except Exception:
                market_data_diag = None

            if market_data_diag is None:
                market_data_diag = market_data.copy()

            try:
                primary_signals_diag = generate_primary_signals(market_data_diag.copy())
            except Exception:
                try:
                    primary_signals_diag = primary_signals.reindex(market_data_diag.index)
                except Exception:
                    primary_signals_diag = pd.Series(0, index=market_data_diag.index)

            log_ret_diag = np.log(market_data_diag["close"]).diff()
            volatility_1d = log_ret_diag.rolling(96).std()

            # Override shared variables so that subsequent diagnostics operate
            # on the full diagnostics window.
            market_data = market_data_diag
            primary_signals = primary_signals_diag

            # Extract parameters
            profit_thr_base = float(diag_params.get("profit_thr_base", 0.012))
            stop_ratio = float(diag_params.get("stop_to_profit_ratio", 0.5))
            stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)
            horizon = int(diag_params.get("horizon_bars", 24))
            min_spacing = int(diag_params.get("min_event_spacing", 0))
            econ_min_mult = float(diag_params.get("econ_min_return_multiple", 1.5))
            label_low_q = float(diag_params.get("label_low_q", 0.40))
            label_high_q = float(diag_params.get("label_high_q", 0.60))
            tx_cost_mult = float(diag_params.get("transaction_cost_mult", 1.0))
            tx_cost_mult = max(1.0, min(1.2, tx_cost_mult))
            effective_tx_cost = DEFAULT_TRANSACTION_COST * tx_cost_mult
            vol_baseline_window = int(diag_params.get("vol_baseline_window", 96))
            profit_mult_min = float(diag_params.get("profit_mult_min", 0.5))
            profit_mult_max = float(diag_params.get("profit_mult_max", 2.0))
            stop_mult_min = float(diag_params.get("stop_mult_min", 0.5))
            stop_mult_max = float(diag_params.get("stop_mult_max", 2.0))

            # Extract trailing distance if available
            trail_dist_diag = float(diag_params.get("trail_distance", 0.0))

            # Recompute ATR for diagnostics
            high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
            low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
            close_prices = market_data["close"]
            tr1 = high_prices - low_prices
            tr2 = (high_prices - close_prices.shift(1)).abs()
            tr3 = (low_prices - close_prices.shift(1)).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_series_diag = true_range.rolling(window=14, min_periods=1).mean()

            # Compute adaptive thresholds
            vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
            vol_factor = volatility_1d / (vol_baseline + 1e-8)

            adaptive_profit = profit_thr_base * vol_factor
            adaptive_stop = stop_thr_base * vol_factor
            adaptive_profit = adaptive_profit.clip(
                lower=profit_thr_base * profit_mult_min,
                upper=profit_thr_base * profit_mult_max,
            )
            adaptive_stop = adaptive_stop.clip(
                lower=stop_thr_base * stop_mult_min,
                upper=stop_thr_base * stop_mult_max,
            )

            # Compute realized returns
            (
                realized_returns_diag,
                binary_labels_raw,
                exit_reasons_diag,
                event_durations_diag,
                mfe_diag,
                mae_diag,
                _, _
            ) = compute_realized_returns(
                market_data,
                primary_signals,
                profit_threshold=adaptive_profit,
                stop_threshold=adaptive_stop,
                horizon=horizon,
                transaction_cost=effective_tx_cost,
                min_event_spacing=min_spacing,
                atr_series=atr_series_diag,
                trail_distance_atr_mult=trail_dist_diag,
            )

            # Vol-scaled returns and quantile labels
            vol_scaled_diag = compute_vol_scaled_returns_for_events(
                realized_returns=realized_returns_diag,
                volatility=volatility_1d,
                econ_min_return_multiple=econ_min_mult,
            )

            regimes_diag = None
            try:
                # Prefer causal volatility regimes over HMM regimes for diagnostics
                if "volatility_regime" in market_data.columns:
                    regimes_diag = market_data["volatility_regime"]
                elif "vol_regime_high" in market_data.columns or "vol_regime_medium" in market_data.columns:
                    # Derive simple categorical regimes from volatility dummies
                    regime_labels: list[str] = []
                    has_high = "vol_regime_high" in market_data.columns
                    has_med = "vol_regime_medium" in market_data.columns
                    for idx in market_data.index:
                        if has_high and market_data.at[idx, "vol_regime_high"] == 1:
                            regime_labels.append("high")
                        elif has_med and market_data.at[idx, "vol_regime_medium"] == 1:
                            regime_labels.append("medium")
                        else:
                            regime_labels.append("low")
                    regimes_diag = pd.Series(regime_labels, index=market_data.index)
                elif "hmm_regime_label_1h" in market_data.columns:
                    # Fallback to HMM regimes only if volatility regimes are unavailable
                    regimes_diag = market_data["hmm_regime_label_1h"]
            except Exception:
                regimes_diag = None

            if regimes_diag is not None:
                quantile_labels_diag = create_regime_aware_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_diag,
                    regimes=regimes_diag,
                    low_q=label_low_q,
                    high_q=label_high_q,
                )
            else:
                quantile_labels_diag = create_quantile_labels_from_vol_scaled_returns(
                    vol_scaled=vol_scaled_diag,
                    low_q=label_low_q,
                    high_q=label_high_q,
                )

            # Tail exit-return diagnostics: mean net_return by exit_reason
            # for positive (label=1) and negative (label=0) quantile tails.
            try:
                tail_exit_stats: Dict[str, Any] = {}
                df_tail = pd.DataFrame(
                    {
                        "label": quantile_labels_diag,
                        "ret": realized_returns_diag,
                        "exit": exit_reasons_diag,
                    }
                )
                df_tail = df_tail.dropna(subset=["label", "ret", "exit"])

                for tail_label, tail_name in [(1.0, "positive"), (0.0, "negative")]:
                    mask_tail = df_tail["label"] == tail_label
                    if not mask_tail.any():
                        continue
                    grouped = df_tail.loc[mask_tail].groupby("exit")["ret"].agg(["mean", "count"])
                    stats: Dict[str, Any] = {}
                    for exit_reason, row in grouped.iterrows():
                        stats[str(exit_reason)] = {
                            "mean": float(row["mean"]),
                            "n": int(row["count"]),
                        }
                    tail_exit_stats[tail_name] = stats

                if tail_exit_stats:
                    best_config_diagnostics["tail_exit_stats"] = tail_exit_stats
            except Exception:
                # Diagnostics are best-effort; ignore failures here.
                pass

            labeled_mask_diag = ~quantile_labels_diag.isna()

            # Build meta-features
            meta_feature_cfg = config.get("meta_feature_engineering", {})
            volume_available = "volume" in market_data.columns

            meta_features_diag, meta_features_processed, _, _ = build_meta_features_for_model(
                market_data=market_data,
                primary_signals=primary_signals,
                realized_returns=realized_returns_diag,
                binary_labels=quantile_labels_diag,
                event_durations=event_durations_diag,
                mfe_series=mfe_diag,
                mae_series=mae_diag,
                adaptive_stop_threshold=adaptive_stop,
                horizon=horizon,
                volume_available=volume_available,
                meta_feature_cfg=meta_feature_cfg,
            )

            # Compute calibrated probabilities with t1-aware CV
            _, mean_auc_diag, calibrated_probs_diag, _, fold_aucs_diag, oof_probs_diag = compute_learnability_with_calibration(
                X=meta_features_processed,
                y=quantile_labels_diag,
                realized_returns=realized_returns_diag,
                model_complexity="strong",
                cv_splits=5,
                time_aware_cv=True,
                use_ensemble=False,
                event_durations=event_durations_diag,
                market_index=market_data.index,
                base_horizon_bars=horizon,
                use_smoothed_brier_objective_lgbm=use_smoothed_brier_objective_lgbm,
            )

            # Align calibrated probabilities and out-of-fold probabilities to labeled events (non-NaN labels)
            try:
                valid_mask_probs = ~quantile_labels_diag.isna()
                probs_series_diag = None
                oof_probs_series_diag = None
                if isinstance(calibrated_probs_diag, np.ndarray) and len(calibrated_probs_diag) == int(valid_mask_probs.sum()):
                    probs_series_diag = pd.Series(calibrated_probs_diag, index=quantile_labels_diag.index[valid_mask_probs])
                if isinstance(oof_probs_diag, np.ndarray) and len(oof_probs_diag) == int(valid_mask_probs.sum()):
                    oof_probs_series_diag = pd.Series(oof_probs_diag, index=quantile_labels_diag.index[valid_mask_probs])
            except Exception:
                probs_series_diag = None
                oof_probs_series_diag = None

            # Create "full" labels (before quantile filtering)
            econ_floor = econ_min_mult * effective_tx_cost
            y_full_diag = pd.Series(np.nan, index=realized_returns_diag.index)
            full_mask = ~realized_returns_diag.isna() & (realized_returns_diag.abs() >= econ_floor)
            y_full_diag[full_mask & (realized_returns_diag > 0)] = 1.0
            y_full_diag[full_mask & (realized_returns_diag <= 0)] = 0.0

            # Two-stage bagged meta-model diagnostics (activity gate + direction)
            tprint_info("  → Training two-stage bagged meta-model (activity + direction)...")
            try:
                # Build events DataFrame with realized returns for stop/timeout split
                events_df_diag = pd.DataFrame(index=realized_returns_diag.index)
                events_df_diag["ret"] = realized_returns_diag

                # Use raw binary labels from compute_realized_returns (1=profit, 0=loss/timeout)
                outcomes_binary_diag = binary_labels_raw

                y_trinary_diag = generate_trinary_labels(events_df_diag, outcomes_binary_diag)

                # Restrict to indices where we have both features and trinary labels
                if isinstance(meta_features_processed, pd.DataFrame):
                    X_two_stage = meta_features_processed
                else:
                    X_two_stage = pd.DataFrame(meta_features_processed, index=realized_returns_diag.index)

                mask_two_stage = (
                    ~y_trinary_diag.isna()
                    & ~realized_returns_diag.isna()
                )

                n_two_stage_events = int(mask_two_stage.sum())
                if n_two_stage_events >= 50:
                    X_ts = X_two_stage.loc[mask_two_stage]
                    y_ts = y_trinary_diag.loc[mask_two_stage]

                    base_lgb_params = {
                        "boosting_type": "gbdt",
                        "objective": "binary",
                        "max_depth": 4,
                        "n_estimators": 220,
                        "learning_rate": 0.02,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_samples": 80,
                        "reg_alpha": 0.3,
                        "reg_lambda": 0.9,
                        "n_jobs": -1,
                        "verbose": -1,
                        "random_state": 42,
                    }

                    two_stage_model = TwoStageBaggedMetaModel(
                        base_params=base_lgb_params,
                        n_bagging=10,
                        bagging_fraction=0.7,
                        random_state=42,
                    )
                    two_stage_model.fit(X_ts, y_ts.to_numpy())

                    # Activity AUC: active (profit or stop) vs timeout
                    y_activity_diag = (y_ts != 0).astype(int)
                    p_active_diag = two_stage_model.stage1_model.predict_proba(X_ts)[:, 1]
                    activity_auc = None
                    activity_brier = None
                    activity_ece = None
                    try:
                        activity_auc = roc_auc_score(y_activity_diag, p_active_diag)
                        activity_brier, activity_ece = compute_brier_and_ece(y_activity_diag, p_active_diag)
                    except Exception:
                        activity_auc = None

                    # Direction AUC: among active events only
                    direction_auc = None
                    direction_brier = None
                    direction_ece = None
                    mask_active = (y_ts != 0)
                    if mask_active.sum() >= 10 and two_stage_model.stage2_ensemble is not None:
                        y_dir_diag = (y_ts[mask_active] == 1).astype(int)
                        p_win_conditional_diag = two_stage_model.stage2_ensemble.predict_proba(
                            X_ts.loc[mask_active]
                        )[:, 1]
                        try:
                            direction_auc = roc_auc_score(y_dir_diag, p_win_conditional_diag)
                            direction_brier, direction_ece = compute_brier_and_ece(
                                y_dir_diag, p_win_conditional_diag
                            )
                        except Exception:
                            direction_auc = None

                    # Combined AUC: profit vs others using final score
                    combined_auc = None
                    global_rank_auc = None
                    regime_weighted_edge = None
                    per_regime_edge: Dict[str, Any] = {}
                    precision_top20 = None
                    base_rate = None
                    trades_per_day_two_stage = None
                    mean_return_all = None
                    mean_return_win = None
                    mean_return_loss = None

                    # Final score for two-stage model
                    if two_stage_model.stage2_ensemble is not None:
                        p_win_conditional_full = two_stage_model.stage2_ensemble.predict_proba(X_ts)[:, 1]
                    else:
                        p_win_conditional_full = np.full(X_ts.shape[0], 0.5, dtype=float)
                    final_score_diag = p_active_diag * p_win_conditional_full
                    y_final_diag = (y_ts == 1).astype(int)

                    # Global combined AUC
                    try:
                        if np.unique(y_final_diag).size >= 2:
                            combined_auc = roc_auc_score(y_final_diag, final_score_diag)
                    except Exception:
                        combined_auc = None

                    # Global Rank AUC via regime-wise percentile ranks and regime-weighted edge
                    regimes_events = None
                    try:
                        if "regimes_diag" in locals() and regimes_diag is not None:
                            if isinstance(regimes_diag, pd.Series):
                                regimes_events = regimes_diag.reindex(X_ts.index)
                    except Exception:
                        regimes_events = None

                    if regimes_events is not None:
                        try:
                            # Build a regime-wise ranking DataFrame for robust computation
                            df_rank = pd.DataFrame(
                                {
                                    "score": final_score_diag,
                                    "y": y_final_diag,
                                    "regime": regimes_events,
                                },
                                index=X_ts.index,
                            ).dropna(subset=["score", "y", "regime"])

                            if not df_rank.empty and df_rank["y"].nunique() >= 2:
                                # Rank scores within each regime
                                df_rank["regime_rank"] = df_rank.groupby("regime")["score"].rank(
                                    pct=True,
                                    method="average",
                                )

                                try:
                                    global_rank_auc = float(
                                        roc_auc_score(df_rank["y"].values, df_rank["regime_rank"].values)
                                    )
                                except Exception:
                                    global_rank_auc = None

                                # Regime-weighted edge for top 10% per regime using realized returns
                                total_top = 0
                                weighted_edge_sum = 0.0
                                per_regime_edge = {}

                                returns_aligned = realized_returns_diag.reindex(df_rank.index)
                                for reg_val, g in df_rank.groupby("regime"):
                                    g_top = g[g["regime_rank"] >= 0.9]
                                    n_top_reg = int(len(g_top))
                                    if n_top_reg < 10:
                                        continue

                                    ret_top = returns_aligned.reindex(g_top.index)
                                    if ret_top is not None and not ret_top.dropna().empty:
                                        edge_reg = float(ret_top.mean())
                                    else:
                                        edge_reg = 0.0

                                    key = str(reg_val)
                                    per_regime_edge[key] = {
                                        "edge_top10": edge_reg,
                                        "n_top": n_top_reg,
                                    }
                                    total_top += n_top_reg
                                    weighted_edge_sum += edge_reg * n_top_reg

                                if total_top > 0:
                                    regime_weighted_edge = float(weighted_edge_sum / total_top)

                                # Precision@Top20% and base rate (student quality)
                                top20_mask = df_rank["regime_rank"] >= 0.8
                                n_top20 = int(top20_mask.sum())
                                if n_top20 > 0:
                                    winners_top20 = int((df_rank.loc[top20_mask, "y"] == 1).sum())
                                    precision_top20 = float(winners_top20 / n_top20)
                                base_rate = float((df_rank["y"] == 1).mean())
                        except Exception:
                            # Leave global_rank_auc and per_regime_edge as-is on failure
                            pass

                    # Trades/day and mean returns per trade (for diagnostics context)
                    try:
                        # Use ACTIVE events (profit/stop) over the actual event span
                        if isinstance(realized_returns_diag.index, pd.DatetimeIndex) and n_two_stage_events > 0:
                            event_idx = realized_returns_diag.index[mask_two_stage]
                            if len(event_idx) >= 2:
                                days_span_events = (
                                    event_idx.max() - event_idx.min()
                                ).total_seconds() / 86400.0
                                if days_span_events > 0:
                                    n_active_events = int((y_ts != 0).sum())
                                    if n_active_events > 0:
                                        trades_per_day_two_stage = float(n_active_events / days_span_events)
                                    else:
                                        trades_per_day_two_stage = float(n_two_stage_events / days_span_events)
                    except Exception:
                        trades_per_day_two_stage = None

                    try:
                        ret_all = realized_returns_diag.loc[mask_two_stage]
                        if not ret_all.empty:
                            mean_return_all = float(ret_all.mean())
                        ret_win = realized_returns_diag.loc[mask_two_stage & (y_ts == 1)]
                        if not ret_win.empty:
                            mean_return_win = float(ret_win.mean())
                        ret_loss = realized_returns_diag.loc[mask_two_stage & (y_ts != 1)]
                        if not ret_loss.empty:
                            mean_return_loss = float(ret_loss.mean())
                    except Exception:
                        mean_return_all = mean_return_all

                    two_stage_diag = {
                        "n_events": n_two_stage_events,
                        "activity_auc": float(activity_auc) if activity_auc is not None else None,
                        "direction_auc": float(direction_auc) if direction_auc is not None else None,
                        "combined_auc": float(combined_auc) if combined_auc is not None else None,
                        "global_rank_auc": float(global_rank_auc) if global_rank_auc is not None else None,
                        "regime_weighted_edge": float(regime_weighted_edge) if regime_weighted_edge is not None else None,
                        "per_regime_edge": per_regime_edge,
                        "precision_top20": float(precision_top20) if precision_top20 is not None else None,
                        "base_rate": float(base_rate) if base_rate is not None else None,
                        "activity_brier": float(activity_brier) if activity_brier is not None else None,
                        "activity_ece": float(activity_ece) if activity_ece is not None else None,
                        "direction_brier": float(direction_brier) if direction_brier is not None else None,
                        "direction_ece": float(direction_ece) if direction_ece is not None else None,
                        "trades_per_day": float(trades_per_day_two_stage) if trades_per_day_two_stage is not None else None,
                        "mean_return_all": float(mean_return_all) if mean_return_all is not None else None,
                        "mean_return_win": float(mean_return_win) if mean_return_win is not None else None,
                        "mean_return_loss": float(mean_return_loss) if mean_return_loss is not None else None,
                    }
                    best_config_diagnostics["two_stage_meta_model"] = two_stage_diag

                    msg_parts: List[str] = []
                    if activity_auc is not None:
                        msg_parts.append(f"activity={activity_auc:.3f}")
                    if direction_auc is not None:
                        msg_parts.append(f"direction={direction_auc:.3f}")
                    if combined_auc is not None:
                        msg_parts.append(f"combined={combined_auc:.3f}")
                    if global_rank_auc is not None:
                        msg_parts.append(f"rank={global_rank_auc:.3f}")
                    if msg_parts:
                        tprint_info("  → Two-stage meta-model AUCs: " + ", ".join(msg_parts))
                else:
                    tprint_warning(
                        f"  ⚠️ Two-stage meta-model skipped: insufficient events (n={n_two_stage_events})"
                    )
            except Exception as e_two_stage:
                tprint_warning(f"  ⚠️ Two-stage meta-model diagnostics failed: {e_two_stage}")

            # Attach signal funnel statistics from the primary signal generator,
            # if available, so that HPO diagnostics can inspect raw vs final
            # signal counts.
            try:
                signal_funnel_diag = primary_signals.attrs.get('signal_funnel', {})  # type: ignore[attr-defined]
            except Exception:
                signal_funnel_diag = {}
            if signal_funnel_diag:
                best_config_diagnostics['signal_funnel'] = signal_funnel_diag
            # 1. Filtering inflation diagnostics
            tprint_info("  → Computing filtering inflation diagnostics...")
            try:
                probabilities_for_filter = None
                # Prefer out-of-fold probabilities when available to avoid in-sample optimism
                source_series_for_filter = oof_probs_series_diag if "oof_probs_series_diag" in locals() and oof_probs_series_diag is not None else probs_series_diag
                if source_series_for_filter is not None:
                    probs_full_aligned = pd.Series(np.nan, index=quantile_labels_diag.index, dtype=float)
                    try:
                        probs_full_aligned.loc[source_series_for_filter.index] = source_series_for_filter
                        probabilities_for_filter = probs_full_aligned.to_numpy(dtype=float)
                    except Exception:
                        probabilities_for_filter = None

                filtering_diag = compute_filtering_inflation_diagnostics(
                    X=meta_features_processed,
                    y_full=y_full_diag,
                    y_filtered=quantile_labels_diag,
                    realized_returns=realized_returns_diag,
                    volatility=volatility_1d,
                    probabilities=probabilities_for_filter,
                    cv_splits=5,
                    time_aware_cv=True,
                    event_durations=event_durations_diag,
                    market_index=market_data.index,
                    base_horizon_bars=horizon,
                )
                best_config_diagnostics['filtering_diagnostics'] = filtering_diag

                if filtering_diag.get('filtering_is_major_contributor'):
                    tprint_warning("  ⚠️ WARNING: Filtering is a major contributor to AUC inflation")
                if filtering_diag.get('auc_dominated_by_large_moves'):
                    tprint_warning("  ⚠️ WARNING: AUC dominated by large-move events")
                if filtering_diag.get('precision_collapse_detected'):
                    tprint_warning("  ⚠️ WARNING: Precision collapse detected (model only good on easy cases)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Filtering diagnostics failed: {e}")

            # 2. Calibration diagnostics + Mutual Information diagnostics
            tprint_info("  → Computing calibration diagnostics...")
            try:
                idx_labeled = quantile_labels_diag.index[labeled_mask_diag]
                y_calib = quantile_labels_diag.loc[idx_labeled].values

                # Prefer out-of-fold probabilities when available; fall back to full-sample calibrated
                probs_calib = None
                if "oof_probs_series_diag" in locals() and oof_probs_series_diag is not None:
                    try:
                        probs_calib_series = oof_probs_series_diag.reindex(idx_labeled)
                        probs_calib = probs_calib_series.to_numpy(dtype=float)
                    except Exception:
                        probs_calib = None

                if probs_calib is None and probs_series_diag is not None:
                    try:
                        probs_calib_series = probs_series_diag.reindex(idx_labeled)
                        probs_calib = probs_calib_series.to_numpy(dtype=float)
                    except Exception:
                        probs_calib = None

                if probs_calib is None and isinstance(calibrated_probs_diag, np.ndarray) and calibrated_probs_diag.size > 0:
                    probs_calib = calibrated_probs_diag if len(calibrated_probs_diag) == len(y_calib) else calibrated_probs_diag[: len(y_calib)]

                returns_calib = realized_returns_diag.loc[idx_labeled].values

                calib_diag = compute_calibration_diagnostics(
                    y_true=y_calib,
                    probabilities=probs_calib,
                    realized_returns=returns_calib,
                    transaction_cost=effective_tx_cost,
                    n_bins=10,
                    regime_score=volatility_1d.loc[idx_labeled].values if volatility_1d is not None else None,
                    use_linear_adaptive_gating=True,
                )
                best_config_diagnostics["calibration_diagnostics"] = calib_diag

                if calib_diag.get("brier_score") is not None:
                    tprint_info(f"  → Brier Score: {calib_diag['brier_score']:.4f}")

                ece_val = calib_diag.get("ece")
                if ece_val is not None:
                    tprint_info(f"  → ECE: {ece_val:.4f}")

                # Distinguish between genuine miscalibration and degenerate /
                # uninformative calibration curves (e.g. all mass in one bin).
                if not calib_diag.get("is_well_calibrated", True):
                    if calib_diag.get("degenerate_calibration", False):
                        tprint_warning(
                            "  ⚠️ WARNING: Calibration diagnostics are degenerate "
                            "(probabilities highly concentrated; treating as not well calibrated)"
                        )
                    elif ece_val is not None:
                        tprint_warning(
                            f"  ⚠️ WARNING: Model is miscalibrated (ECE={ece_val:.4f} ≥ 0.05)"
                        )
                    else:
                        tprint_warning(
                            "  ⚠️ WARNING: Model is miscalibrated (insufficient calibration data)"
                        )

                # 2b. Mutual information diagnostics (probability deciles vs label and return sign)
                tprint_info("  → Computing mutual information diagnostics...")
                mi_diag: Dict[str, Any] = {}
                try:
                    if probs_calib is not None and len(probs_calib) >= 50:
                        probs_series = pd.Series(probs_calib, index=idx_labeled).clip(0.0, 1.0)
                        try:
                            prob_decile = pd.qcut(
                                probs_series,
                                q=10,
                                labels=False,
                                duplicates="drop",
                            )
                            # If qcut collapsed to < 3 bins, fall back to fixed-width bins
                            if prob_decile is not None and prob_decile.nunique() < 3:
                                prob_decile = pd.cut(
                                    probs_series,
                                    bins=10,
                                    labels=False,
                                    include_lowest=True,
                                )
                        except Exception:
                            # Fallback to fixed-width bins if qcut fails entirely
                            try:
                                prob_decile = pd.cut(
                                    probs_series,
                                    bins=10,
                                    labels=False,
                                    include_lowest=True,
                                )
                            except Exception:
                                prob_decile = None

                        if prob_decile is not None:
                            y_series = pd.Series(y_calib, index=idx_labeled)

                            # Track basic MI input statistics for debugging
                            try:
                                mi_diag["n_prob_bins"] = int(pd.Series(prob_decile).nunique(dropna=True))
                            except Exception:
                                mi_diag["n_prob_bins"] = None
                            try:
                                mi_diag["n_label_classes"] = int(y_series.nunique(dropna=True))
                            except Exception:
                                mi_diag["n_label_classes"] = None

                            # MI between probability deciles and label
                            mi_nats_label = _discrete_mi(prob_decile, y_series)
                            mi_bits_label = (
                                mi_nats_label / np.log(2.0)
                                if np.isfinite(mi_nats_label)
                                else float("nan")
                            )
                            mi_diag["mi_prob_label_nats"] = mi_nats_label
                            mi_diag["mi_prob_label_bits"] = mi_bits_label

                            # MI between probability deciles and realized return sign
                            returns_series = pd.Series(returns_calib, index=idx_labeled)
                            dir_series = (returns_series > 0).astype(int)
                            try:
                                mi_diag["n_return_sign_classes"] = int(
                                    dir_series.nunique(dropna=True)
                                )
                            except Exception:
                                mi_diag["n_return_sign_classes"] = None

                            mi_nats_dir = _discrete_mi(prob_decile, dir_series)
                            mi_bits_dir = (
                                mi_nats_dir / np.log(2.0)
                                if np.isfinite(mi_nats_dir)
                                else float("nan")
                            )
                            mi_diag["mi_prob_return_sign_nats"] = mi_nats_dir
                            mi_diag["mi_prob_return_sign_bits"] = mi_bits_dir

                    if mi_diag:
                        best_config_diagnostics["mi_diagnostics"] = mi_diag
                        if "mi_prob_label_bits" in mi_diag and "mi_prob_return_sign_bits" in mi_diag:
                            tprint_info(
                                f"  → MI(bits): prob→label={mi_diag['mi_prob_label_bits']:.4f}, "
                                f"prob→ret_sign={mi_diag['mi_prob_return_sign_bits']:.4f}"
                            )
                            # If MI is numerically ~0 despite a non-trivial number of
                            # bins/classes, log a hint so we can inspect discretisation
                            # in future runs.
                            try:
                                if (
                                    mi_diag.get("mi_prob_label_bits") is not None
                                    and mi_diag.get("mi_prob_return_sign_bits") is not None
                                ):
                                    mi_label_abs = abs(float(mi_diag["mi_prob_label_bits"]))
                                    mi_dir_abs = abs(float(mi_diag["mi_prob_return_sign_bits"]))
                                    n_bins = mi_diag.get("n_prob_bins") or 0
                                    n_lbl = mi_diag.get("n_label_classes") or 0
                                    if (
                                        mi_label_abs < 1e-4
                                        and mi_dir_abs < 1e-4
                                        and n_bins >= 2
                                        and n_lbl >= 2
                                    ):
                                        tprint_warning(
                                            "  ⚠️ WARNING: MI≈0.0 despite multiple bins/classes; "
                                            "check probability discretisation or label alignment"
                                        )
                            except Exception:
                                pass
                except Exception as mi_exc:
                    tprint_warning(f"  ⚠️ MI diagnostics failed: {mi_exc}")

                regime_rank_diag: Dict[str, Any] = {}
                try:
                    if regimes_diag is not None and probs_calib is not None and len(probs_calib) == len(y_calib):
                        regimes_for_calib = regimes_diag.reindex(idx_labeled)
                        if isinstance(regimes_for_calib, pd.Series):
                            valid_reg_mask = ~regimes_for_calib.isna()
                            valid_reg_mask = valid_reg_mask.to_numpy()
                            if isinstance(probs_calib, np.ndarray):
                                valid_reg_mask = valid_reg_mask & np.isfinite(probs_calib)

                            if valid_reg_mask.any():
                                y_rr = y_calib[valid_reg_mask]
                                if len(y_rr) >= 50 and np.unique(y_rr).size >= 2:
                                    probs_rr = probs_calib[valid_reg_mask]
                                    regs_rr = regimes_for_calib.iloc[valid_reg_mask]

                                    df_rr = pd.DataFrame(
                                        {"prob": probs_rr, "regime": regs_rr.values, "y": y_rr},
                                        index=regs_rr.index,
                                    )
                                    df_rr = df_rr.dropna(subset=["prob", "regime", "y"])

                                    if not df_rr.empty and df_rr["y"].nunique() >= 2:
                                        df_rr["regime_rank"] = df_rr.groupby("regime")["prob"].rank(
                                            pct=True,
                                            method="average",
                                        )

                                        try:
                                            global_auc_raw = float(
                                                roc_auc_score(df_rr["y"].values, df_rr["prob"].values)
                                            )
                                        except Exception:
                                            global_auc_raw = None

                                        try:
                                            global_auc_rank = float(
                                                roc_auc_score(df_rr["y"].values, df_rr["regime_rank"].values)
                                            )
                                        except Exception:
                                            global_auc_rank = None

                                        per_regime_auc_rank: Dict[str, Any] = {}
                                        for reg_val, g in df_rr.groupby("regime"):
                                            y_g = g["y"].values
                                            r_g = g["regime_rank"].values
                                            auc_g = None
                                            if len(y_g) >= 30 and np.unique(y_g).size >= 2:
                                                try:
                                                    auc_g = float(roc_auc_score(y_g, r_g))
                                                except Exception:
                                                    auc_g = None
                                            per_regime_auc_rank[str(reg_val)] = {
                                                "auc": auc_g,
                                                "n_events": int(len(g)),
                                            }

                                        regime_rank_diag = {
                                            "global_auc_raw": global_auc_raw,
                                            "global_auc_regime_rank": global_auc_rank,
                                            "per_regime_auc_rank": per_regime_auc_rank,
                                        }
                                        best_config_diagnostics["regime_rank_diagnostics"] = regime_rank_diag

                                        if global_auc_raw is not None and global_auc_rank is not None:
                                            tprint_info(
                                                f"  → Regime-rank AUC: raw={global_auc_raw:.3f}, "
                                                f"regime_rank={global_auc_rank:.3f}"
                                            )
                except Exception as rr_exc:
                    tprint_warning(f"  ⚠️ Regime-rank diagnostics failed: {rr_exc}")

            except Exception as e:
                tprint_warning(f"  ⚠️ Calibration diagnostics failed: {e}")

            # 3. Robustness diagnostics
            tprint_info("  → Computing robustness diagnostics...")
            try:
                robust_diag = compute_robustness_diagnostics(
                    X=meta_features_processed,
                    y=quantile_labels_diag,
                    realized_returns=realized_returns_diag,
                    regimes=regimes_diag,
                    volatility=volatility_1d,
                    n_folds=5,
                    transaction_cost=effective_tx_cost,
                    time_aware_cv=True,
                    event_durations=event_durations_diag,
                    market_index=market_data.index,
                    base_horizon_bars=horizon,
                    use_purged_splits=True,
                )
                best_config_diagnostics['robustness_diagnostics'] = robust_diag

                if robust_diag.get('worst_fold_auc') is not None:
                    tprint_info(f"  → Worst-fold AUC: {robust_diag['worst_fold_auc']:.3f}")
                if robust_diag.get('auc_cv_std') is not None:
                    tprint_info(f"  → AUC CV Std: {robust_diag['auc_cv_std']:.4f}")
                # Additional visibility into regime / time sensitivity
                if robust_diag.get('best_fold_auc') is not None:
                    tprint_info(f"  → Best-fold AUC: {robust_diag['best_fold_auc']:.3f}")
                if robust_diag.get('auc_cv_coefficient_of_variation') is not None:
                    tprint_info(
                        f"  → AUC CV Coefficient of Variation: "
                        f"{robust_diag['auc_cv_coefficient_of_variation']:.4f}"
                    )
                if robust_diag.get('per_volatility_regime'):
                    try:
                        for reg_name, m in robust_diag['per_volatility_regime'].items():
                            auc_r = m.get('auc')
                            n_ev = m.get('n_events')
                            if auc_r is not None and n_ev is not None:
                                tprint_info(
                                    f"    · Volatility regime {reg_name}: "
                                    f"AUC={auc_r:.3f}, n_events={int(n_ev)}"
                                )
                    except Exception:
                        pass
                if robust_diag.get('per_regime_metrics'):
                    try:
                        # Log a compact per-regime summary to highlight any
                        # particularly weak or strong regime cluster.
                        for reg_name, m in robust_diag['per_regime_metrics'].items():
                            auc_r = m.get('auc')
                            n_ev = m.get('n_events')
                            net_pnl = m.get('net_pnl_per_trade')
                            if auc_r is None or n_ev is None:
                                continue
                            suffix = (
                                f", net_pnl_per_trade={net_pnl:.5f}"
                                if net_pnl is not None
                                else ""
                            )
                            tprint_info(
                                f"    · Regime {reg_name}: AUC={auc_r:.3f}, "
                                f"n_events={int(n_ev)}{suffix}"
                            )
                    except Exception:
                        pass
                if not robust_diag.get('is_robust', True):
                    tprint_warning("  ⚠️ WARNING: Model not robust (high CV variance or poor worst-fold)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Robustness diagnostics failed: {e}")

            # 4. Class overlap diagnostics
            tprint_info("  → Computing class overlap diagnostics...")
            try:
                overlap_diag = compute_class_overlap_features(
                    X=meta_features_processed,
                    retained_mask=labeled_mask_diag,
                    top_k_features=10,
                )
                best_config_diagnostics['class_overlap_diagnostics'] = overlap_diag
                if overlap_diag.get('easy_problem_detected'):
                    tprint_warning("  ⚠️ WARNING: Easy problem detected (retained events form tight cluster)")
            except Exception as e:
                tprint_warning(f"  ⚠️ Class overlap diagnostics failed: {e}")

            # 5. Permutation-importance leakage diagnostics (god feature detection)
            tprint_info("  → Computing permutation-importance leakage diagnostics...")
            try:
                leakage_diag = run_leakage_sanity_check(
                    X=meta_features_processed,
                    y=quantile_labels_diag,
                    random_state=42,
                    top_k=5,
                    n_repeats=5,
                )
                if leakage_diag:
                    best_config_diagnostics["leakage_diagnostics"] = leakage_diag
                    if leakage_diag.get("god_feature_suspected"):
                        tprint_warning(
                            "  ⚠️ WARNING: Permutation-importance suggests a god feature (possible leakage)"
                        )
                        # Surface the actual feature names and impact so we can
                        # quickly inspect potential sources of leakage.
                        try:
                            top_feats = leakage_diag.get("top_features") or []
                            top_imps = leakage_diag.get("top_importances") or []
                            baseline_auc = leakage_diag.get("baseline_auc")
                            dropped_auc = leakage_diag.get("dropped_auc")
                            delta_auc = leakage_diag.get("delta_auc")

                            if top_feats:
                                primary = top_feats[0]
                                primary_imp = (
                                    float(top_imps[0])
                                    if top_imps and top_imps[0] is not None
                                    else None
                                )
                                baseline_str = (
                                    f"{float(baseline_auc):.4f}"
                                    if baseline_auc is not None
                                    else "nan"
                                )
                                dropped_str = (
                                    f"{float(dropped_auc):.4f}"
                                    if dropped_auc is not None
                                    else "nan"
                                )
                                delta_str = (
                                    f"{float(delta_auc):.4f}"
                                    if delta_auc is not None
                                    else "nan"
                                )
                                imp_str = (
                                    f"{primary_imp:.4f}" if primary_imp is not None else "nan"
                                )
                                tprint_warning(
                                    "    Top leakage candidate: "
                                    f"{primary} (baseline_auc={baseline_str}, "
                                    f"dropped_auc={dropped_str}, delta_auc={delta_str}, "
                                    f"perm_importance={imp_str})"
                                )
                                if len(top_feats) > 1:
                                    others = ", ".join(map(str, top_feats[1:]))
                                    tprint_info(
                                        f"    Other high-importance features: {others}"
                                    )
                        except Exception:
                            pass
            except Exception as e:
                tprint_warning(f"  ⚠️ Leakage diagnostics failed: {e}")

            # 6. Lag-1 stress test for look-ahead bias
            tprint_info("  → Computing lag-1 stress test diagnostics...")
            try:
                lag_diag = run_lag1_stress_test(
                    X=meta_features_processed,
                    y=quantile_labels_diag,
                    random_state=42,
                )
                if lag_diag:
                    best_config_diagnostics["lag1_stress_test"] = lag_diag
                    if lag_diag.get("lookahead_suspected"):
                        tprint_warning(
                            "  ⚠️ WARNING: Lag-1 stress test suggests look-ahead bias (AUC drops sharply when lagged)"
                        )
            except Exception as e:
                tprint_warning(f"  ⚠️ Lag-1 stress test diagnostics failed: {e}")

            # 7. Dummy-rule volatility baseline AUC
            tprint_info("  → Computing dummy-rule volatility baseline AUC...")
            try:
                dummy_diag = compute_dummy_baseline_auc(
                    volatility=volatility_1d,
                    y=quantile_labels_diag,
                    window=64,
                )
                if dummy_diag:
                    best_config_diagnostics["dummy_baseline_diagnostics"] = dummy_diag
            except Exception as e:
                tprint_warning(f"  ⚠️ Dummy baseline diagnostics failed: {e}")

            # 8. Y-shuffle sanity test (labels shuffled, features intact)
            tprint_info("  → Running Y-shuffle sanity test on meta-features...")
            try:
                if isinstance(quantile_labels_diag, pd.Series):
                    y_vals = quantile_labels_diag.to_numpy(dtype=float)
                    rng = np.random.RandomState(42)
                    rng.shuffle(y_vals)
                    y_shuffled = pd.Series(y_vals, index=quantile_labels_diag.index)

                    _, auc_y_shuffle, _, _, _, _ = compute_learnability_with_calibration(
                        X=meta_features_processed,
                        y=y_shuffled,
                        realized_returns=realized_returns_diag,
                        model_complexity="strong",
                        cv_splits=5,
                        time_aware_cv=True,
                        use_ensemble=False,
                        event_durations=event_durations_diag,
                        market_index=market_data.index,
                        base_horizon_bars=horizon,
                        use_smoothed_brier_objective_lgbm=use_smoothed_brier_objective_lgbm,
                    )

                    best_config_diagnostics["y_shuffle_test"] = {
                        "auc": float(auc_y_shuffle),
                    }
            except Exception as e:
                tprint_warning(f"  ⚠️ Y-shuffle test failed: {e}")

            # 9. Single-feature baseline: momentum_10_x_regime_high
            tprint_info("  → Running single-feature baseline with momentum_10_x_regime_high...")
            try:
                single_name = "momentum_10_x_regime_high"
                if isinstance(meta_features_processed, pd.DataFrame) and single_name in meta_features_processed.columns:
                    X_single = meta_features_processed[[single_name]]

                    _, auc_single, _, _, _, _ = compute_learnability_with_calibration(
                        X=X_single,
                        y=quantile_labels_diag,
                        realized_returns=realized_returns_diag,
                        model_complexity="strong",
                        cv_splits=5,
                        time_aware_cv=True,
                        use_ensemble=False,
                        event_durations=event_durations_diag,
                        market_index=market_data.index,
                        base_horizon_bars=horizon,
                        use_smoothed_brier_objective_lgbm=use_smoothed_brier_objective_lgbm,
                    )

                    best_config_diagnostics["single_feature_momentum_test"] = {
                        "feature": single_name,
                        "auc": float(auc_single),
                    }
                else:
                    tprint_warning("  ⚠️ Single-feature test skipped: momentum_10_x_regime_high not in feature set")
            except Exception as e:
                tprint_warning(f"  ⚠️ Single-feature momentum test failed: {e}")

            # Nested sr_labeling_xgb XGB HPO diagnostics were previously stubbed
            # behind the enable_xgb_model_hpo flag. To keep this step focused on
            # labeling-HPO only and reduce computational overhead, we no longer
            # invoke or track any nested XGB HPO here.

            tprint_success("✅ Comprehensive diagnostics completed for best configuration")

        except Exception as diag_exc:
            tprint_warning(f"⚠️ Failed to compute comprehensive diagnostics: {diag_exc}")
            import traceback
            traceback.print_exc()

        # ------------------------------------------------------------------
        # 8) (Disabled) Pareto frontier and knee-point logic
        # ------------------------------------------------------------------
        pareto_solutions: list[Solution] = []
        pareto_front: list[Solution] = []
        knee_solution = None
        knee_params = best_params

        try:
            if candidate_pool:
                auc_values = [float(c.get("mean_auc", np.nan)) for c in candidate_pool]
                auc_values_clean = [v for v in auc_values if np.isfinite(v)]
                auc_median = float(np.median(auc_values_clean)) if auc_values_clean else 0.6

                k_auc = 10.0

                for cand in candidate_pool:
                    edge_val = float(cand.get("edge", 0.0))
                    mean_auc_raw = float(cand.get("mean_auc", 0.0))
                    auc_centered = mean_auc_raw - auc_median
                    if np.isfinite(auc_centered):
                        smooth_auc = float(1.0 / (1.0 + np.exp(-k_auc * auc_centered)))
                    else:
                        smooth_auc = 0.5

                    metrics = {
                        "edge": edge_val,
                        "mean_auc_smooth": smooth_auc,
                        "mean_auc_raw": mean_auc_raw,
                        "learnability": float(cand.get("learnability", 0.0)),
                        "profitability": float(cand.get("profitability", 0.0)),
                        "sharpe_pos": float(cand.get("sharpe_pos", 0.0)),
                    }
                    params_for_sol = cand.get("params") or {}
                    pareto_solutions.append(Solution(metrics=metrics, params=params_for_sol))

                objectives = {"edge": "max", "mean_auc_smooth": "max"}
                pareto_front = compute_pareto_front(pareto_solutions, objectives, use_gpu=True, use_vectorbt=True)
                knee_solution = select_knee_point(pareto_front, objectives)
                if knee_solution and knee_solution.params:
                    tmp_params = best_params.copy()
                    tmp_params.update(knee_solution.params)
                    knee_params = tmp_params
        except Exception as pareto_exc:
            tprint_warning(f"⚠️ Pareto frontier construction failed: {pareto_exc}")
            pareto_solutions = []
            pareto_front = []
            knee_solution = None
            knee_params = best_params

        # Compact run summary for quick log scanning
        try:
            round_results = getattr(optimizer, "round_results", [])
            n_rounds = len(round_results) if isinstance(round_results, list) else None
            total_trials = sum(r.get("trials", 0) for r in round_results) if isinstance(round_results, list) else None
        except Exception:
            n_rounds = None
            total_trials = None

        # Build comprehensive summary with key quality metrics
        mean_auc = best_candidate_metrics.get("mean_auc", 0.5)
        trades_per_day = best_candidate_metrics.get("trades_per_day", 0.0)
        learnability = best_candidate_metrics.get("learnability", 0.0)
        profitability = best_candidate_metrics.get("profitability", 0.0)
        sharpe_pos = best_candidate_metrics.get("sharpe_pos", 0.0)
        n_events = best_candidate_metrics.get("n_events", 0)

        tprint_info(
            "HPO summary → "
            f"symbol={symbol}, timeframe={timeframe}, "
            f"best_score(edge)={best_score:.6f}, "
            f"mean_auc={mean_auc:.4f}, "
            f"trades_per_day={trades_per_day:.3f}, "
            f"learnability={learnability:.4f}, "
            f"profitability={profitability:.4f}, "
            f"sharpe_pos={sharpe_pos:.4f}, "
            f"n_events={n_events}, "
            f"rounds={n_rounds}, trials={total_trials}",
        )

        # Persist best parameters and candidate pool to outcomes/
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # ===== OPTIONAL: Generate diagnostics for recommended configuration =====
        diagnostics_path: str | None = None
        try:
            # Prefer knee point if available, otherwise fall back to best_params
            diag_params: Dict[str, Any] = knee_params if knee_solution else best_params
            # Only run diagnostics when explicitly enabled via the module-level
            # constant to avoid categorical setitem issues in some environments.
            if diag_params and GENERATE_RECOMMENDED_DIAGNOSTICS:
                tprint_info("📊 Generating meta-labeling diagnostics for recommended configuration...")

                # Reconstruct labeling parameters (consistent with labeling_objective)
                profit_thr_base = float(diag_params["profit_thr_base"])
                stop_ratio = float(diag_params["stop_to_profit_ratio"])
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)

                horizon = int(diag_params["horizon_bars"])
                # Use safer get() with default 2 if key missing (e.g. older artifact)
                min_spacing = int(diag_params.get("min_event_spacing", 0))

                kalman_Q = float(diag_params.get("kalman_Q", 1e-4))
                kalman_R = float(diag_params.get("kalman_R", 0.01))
                vol_baseline_window = int(diag_params.get("vol_baseline_window", 96))
                profit_mult_min = float(diag_params.get("profit_mult_min", 0.5))
                profit_mult_max = float(diag_params.get("profit_mult_max", 2.0))
                stop_mult_min = float(diag_params.get("stop_mult_min", 0.5))
                stop_mult_max = float(diag_params.get("stop_mult_max", 2.0))

                # Apply same constraints as HPO objective (short intraday horizons)
                horizon = max(8, min(32, horizon))
                if horizon % 2 != 0:
                    horizon = (horizon // 2) * 2
                min_spacing = max(1, min(16, min_spacing))
                vol_baseline_window = max(8, min(512, vol_baseline_window))

                if profit_mult_min > profit_mult_max:
                    profit_mult_min, profit_mult_max = profit_mult_max, profit_mult_min
                if stop_mult_min > stop_mult_max:
                    stop_mult_min, stop_mult_max = stop_mult_max, stop_mult_min

                iso_min_prob = float(diag_params.get("iso_min_prob", 0.05))
                iso_min_prob = max(0.05, min(0.15, iso_min_prob))
                iso_max_prob = 1.0 - iso_min_prob
                iso_max_prob = max(0.85, min(1.0, iso_max_prob))

                q_high = float(diag_params.get("target_clip_high_q", 0.95))
                q_high = max(0.90, min(0.98, q_high))
                q_low = max(0.0, min(0.5, 1.0 - q_high))

                econ_min_mult = float(diag_params.get("econ_min_return_multiple", ECON_MIN_RETURN_MULTIPLE))
                if not np.isfinite(econ_min_mult) or econ_min_mult <= 0:
                    econ_min_mult = float(ECON_MIN_RETURN_MULTIPLE)

                # Recompute adaptive profit/stop thresholds
                vol_baseline = volatility_1d.rolling(vol_baseline_window).mean()
                vol_factor = volatility_1d / (vol_baseline + 1e-8)

                high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
                low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
                close_prices = market_data["close"]

                tr1 = high_prices - low_prices
                tr2 = (high_prices - close_prices.shift(1)).abs()
                tr3 = (low_prices - close_prices.shift(1)).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                trend_atr_window = int(config.get("trend_strength_atr_window", 14))
                atr_series = true_range.rolling(window=trend_atr_window, min_periods=1).mean()

                trend_delta_lookback = int(config.get("trend_strength_delta_lookback", 4))
                price_delta = close_prices.diff(trend_delta_lookback).abs()

                trend_strength = (price_delta / (atr_series + 1e-8)).replace([np.inf, -np.inf], np.nan)
                trend_strength = trend_strength.clip(
                    lower=0.0,
                    upper=float(config.get("trend_strength_clip", 5.0)),
                ).fillna(0.0)

                trend_alpha = float(config.get("trend_strength_alpha_profit", 0.5))
                trend_beta = float(config.get("trend_strength_beta_stop", 0.5))

                profit_factor = 1.0 + trend_alpha * trend_strength
                stop_factor = 1.0 + trend_beta * trend_strength

                adaptive_profit = profit_thr_base * vol_factor * profit_factor
                adaptive_stop = stop_thr_base * vol_factor * stop_factor
                adaptive_profit = adaptive_profit.clip(
                    lower=profit_thr_base * profit_mult_min,
                    upper=profit_thr_base * profit_mult_max,
                )
                adaptive_stop = adaptive_stop.clip(
                    lower=stop_thr_base * stop_mult_min,
                    upper=stop_thr_base * stop_mult_max,
                )

                (
                    realized_returns,
                    binary_labels,
                    exit_reasons,
                    event_durations,
                    mfe_series_diag,
                    mae_series_diag,
                    _binary_labels_long_diag,  # Not used in diagnostics
                    _binary_labels_short_diag,  # Not used in diagnostics
                ) = compute_realized_returns(
                    market_data,
                    primary_signals,
                    profit_threshold=adaptive_profit,
                    stop_threshold=adaptive_stop,
                    horizon=horizon,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=min_spacing,
                )

                # Guard: if too few events, skip diagnostics
                labeled_mask = ~binary_labels.isna()
                if int(labeled_mask.sum()) < 100:
                    tprint_warning(
                        "⚠️ Recommended config produced too few events for diagnostics; skipping report generation",
                        "WARNING",
                    )
                else:
                    # Kalman smoothing
                    smoothed_labels, _ = kalman_smooth_labels(
                        binary_labels,
                        Q=kalman_Q,
                        R=kalman_R,
                        volatility=volatility_1d,
                    )

                    prob_series = smoothed_labels.clip(0.0, 1.0)
                    prob_clipped = prob_series.clip(iso_min_prob, iso_max_prob)

                    # Fit probability→expected-return mapping
                    iso_reg = fit_probability_to_return_mapping(
                        probabilities=prob_clipped.values,
                        realized_returns=realized_returns.values,
                        method="isotonic",
                        econ_min_return_multiple=econ_min_mult,
                    )

                    # Translate to long/short targets
                    target_long, target_short = translate_to_targets_with_isotonic(
                        realized_returns=realized_returns,
                        probabilities=prob_clipped.values,
                        signals=primary_signals,
                        iso_regressor=iso_reg,
                    )

                    # Build labeled_data in the same spirit as production step
                    labeled_data = market_data.copy()

                    # Drop any existing derived columns that might carry categorical dtypes
                    # so that we can safely assign fresh non-categorical Series
                    derived_cols = [
                        "log_ret",
                        "volatility_1d",
                        "realized_return",
                        "binary_label",
                        "smoothed_label",
                        "meta_probability",
                        "exit_reason",
                        "event_duration_bars",
                        "target_long",
                        "target_short",
                        "primary_signal",
                    ]
                    labeled_data = labeled_data.drop(columns=[c for c in derived_cols if c in labeled_data.columns], errors="ignore")

                    log_ret = np.log(market_data["close"]).diff()
                    labeled_data["log_ret"] = log_ret
                    labeled_data["volatility_1d"] = volatility_1d
                    labeled_data["realized_return"] = realized_returns
                    labeled_data["binary_label"] = binary_labels
                    labeled_data["smoothed_label"] = smoothed_labels
                    labeled_data["meta_probability"] = prob_clipped.values
                    labeled_data["exit_reason"] = exit_reasons
                    labeled_data["event_duration_bars"] = event_durations
                    labeled_data["target_long"] = target_long
                    labeled_data["target_short"] = target_short
                    labeled_data["primary_signal"] = primary_signals["consensus"]

                    # Meta-features for diagnostics (same helper as production)
                    meta_feature_cfg = config.get("meta_feature_engineering", {})
                    volume_available = "volume" in market_data.columns

                    meta_features_diag, meta_features_model_diag, _, _ = build_meta_features_for_model(
                        market_data=market_data,
                        primary_signals=primary_signals,
                        realized_returns=realized_returns,
                        binary_labels=binary_labels,
                        event_durations=event_durations,
                        mfe_series=mfe_series_diag,
                        mae_series=mae_series_diag,
                        adaptive_stop_threshold=adaptive_stop,
                        horizon=horizon,
                        volume_available=volume_available,
                        meta_feature_cfg=meta_feature_cfg,
                    )

                    # Simple RF meta-model for feature importances
                    X_diag = meta_features_model_diag[labeled_mask].fillna(0)
                    y_diag = binary_labels[labeled_mask]
                    final_model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        random_state=42,
                    )
                    if len(y_diag.unique()) >= 2 and len(y_diag) >= 50:
                        final_model.fit(X_diag, y_diag)
                    else:
                        # Fallback: still fit to avoid attribute errors in diagnostics
                        final_model.fit(X_diag, y_diag)

                    # Slightly enriched config for diagnostics
                    diag_config = dict(config)
                    diag_config["horizon"] = horizon
                    diag_config["profit_thr_base"] = profit_thr_base
                    diag_config["stop_thr_base"] = stop_thr_base

                    # Sanitize any categorical columns to avoid setitem/category issues
                    labeled_data_for_diag = labeled_data.copy()
                    cat_cols = labeled_data_for_diag.select_dtypes(include=["category"]).columns
                    if len(cat_cols) > 0:
                        for col in cat_cols:
                            labeled_data_for_diag[col] = labeled_data_for_diag[col].astype(object)

                    # Also ensure core Series are not categorical
                    binary_labels_diag = pd.Series(
                        binary_labels.astype(float).values,
                        index=binary_labels.index,
                    )
                    exit_reasons_diag = None
                    event_durations_diag = None
                    if exit_reasons is not None:
                        exit_reasons_diag = pd.Series(
                            exit_reasons.astype(object).values,
                            index=exit_reasons.index,
                        )
                    if event_durations is not None:
                        event_durations_diag = pd.Series(
                            event_durations.astype(float).values,
                            index=event_durations.index,
                        )

                    diagnostics_path_obj = generate_diagnostics_report(
                        labeled_data=labeled_data_for_diag,
                        meta_features=meta_features_diag,
                        binary_labels=binary_labels_diag,
                        realized_returns=realized_returns,
                        smoothed_labels=smoothed_labels,
                        probabilities=prob_clipped.values,
                        final_model=final_model,
                        config=diag_config,
                        output_dir=outcomes_dir,
                        exit_reasons=exit_reasons_diag,
                        event_durations=event_durations_diag,
                        mfe_series=mfe_series_diag,
                        mae_series=mae_series_diag,
                        target_long=target_long,
                        target_short=target_short,
                        selected_feature_names=selected_feature_names,
                    )
                    diagnostics_path = str(diagnostics_path_obj)

                    tprint_success(
                        f"📊 Saved diagnostics for recommended labeling configuration to {diagnostics_path}",
                    )
        except Exception as diag_exc:
            tprint_warning(f"⚠️ Failed to generate diagnostics for recommended configuration: {diag_exc}")

        # ===== SAVE BEST PARAMS JSON =====
        json_name = f"meta_labeling_hpo_best_params_{symbol}_{timeframe}_{direction}_{timestamp}.json"
        json_path = outcomes_dir / json_name
        standardized_json_path = None

        try:
            # Also save a copy to standardized reports/post_hpo_evaluation for downstream consumers (e.g., meta_gated_backtest)
            try:
                from src.utils.pipeline_standards import PipelineStandards
                base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
                standardized_dir = Path(base_dir) / "post_hpo_evaluation"
                standardized_dir.mkdir(parents=True, exist_ok=True)
                standardized_json_path = standardized_dir / json_name
            except Exception as std_exc:
                tprint_warning(f"⚠️ Failed to build standardized HPO output path: {std_exc}")
            # Get best edge from the best candidate
            best_candidate_edge = 0.0
            if candidate_pool:
                sorted_candidates = sorted(candidate_pool, key=lambda x: x.get('edge', x.get('combined', 0)), reverse=True)
                if sorted_candidates:
                    best_candidate_edge = sorted_candidates[0].get('edge', 0.0)

            # Build output dict with diagnostics
            output_dict = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "best_score": best_score,
                "best_edge": best_candidate_edge,
                "best_params": best_params,
                "knee_params": knee_params,
                "pareto_front_size": len(pareto_front),
                "total_trials": all_trials_count,
                "stage_results": [
                    {
                        "stage": sr["stage"],
                        "complexity": sr["complexity"],
                        "best_score": sr["best_score"],
                        "trials": sr["trials"],
                    }
                    for sr in stage_results
                ],
                # Gate usage statistics across all evaluated configs
                "gate_stats": gate_stats,
                # Full candidate history for post-hoc analysis (the "matrix")
                "all_candidates": candidate_pool,
            }

            try:
                best_label_config_id = None
                knee_label_config_id = None

                if isinstance(best_params, dict) and best_params:
                    best_cfg = build_label_config(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        params=best_params,
                        extra=None,
                    )
                    best_label_config_id = compute_label_config_id(best_cfg)

                if isinstance(knee_params, dict) and knee_params:
                    knee_cfg = build_label_config(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        params=knee_params,
                        extra=None,
                    )
                    knee_label_config_id = compute_label_config_id(knee_cfg)

                output_dict["best_label_config_id"] = best_label_config_id
                output_dict["knee_label_config_id"] = knee_label_config_id
            except Exception:
                # If anything goes wrong computing IDs, proceed without them.
                pass

            # Add comprehensive diagnostics for best config
            if best_config_diagnostics:
                output_dict["best_config_diagnostics"] = best_config_diagnostics

                # Add summary warnings as top-level fields
                filtering_diag = best_config_diagnostics.get("filtering_diagnostics", {})
                calib_diag = best_config_diagnostics.get("calibration_diagnostics", {})
                robust_diag = best_config_diagnostics.get("robustness_diagnostics", {})
                overlap_diag = best_config_diagnostics.get("class_overlap_diagnostics", {})
                xgb_diag = best_config_diagnostics.get("sr_labeling_xgb", {})
                mi_diag = best_config_diagnostics.get("mi_diagnostics", {})
                leakage_diag = best_config_diagnostics.get("leakage_diagnostics", {})
                lag_diag = best_config_diagnostics.get("lag1_stress_test", {})
                dummy_diag = best_config_diagnostics.get("dummy_baseline_diagnostics", {})

                output_dict["diagnostics_summary"] = {
                    # Filtering / label inflation
                    "filtering_is_major_contributor": filtering_diag.get(
                        "filtering_is_major_contributor", False
                    ),
                    "auc_dominated_by_large_moves": filtering_diag.get(
                        "auc_dominated_by_large_moves", False
                    ),
                    "precision_collapse_detected": filtering_diag.get(
                        "precision_collapse_detected", False
                    ),
                    "auc_full": filtering_diag.get("auc_full"),
                    "auc_filtered": filtering_diag.get("auc_filtered"),
                    "auc_inflation": filtering_diag.get("auc_inflation"),
                    # Calibration metrics
                    "is_well_calibrated": calib_diag.get("is_well_calibrated", True),
                    "brier_score": calib_diag.get("brier_score"),
                    "ece": calib_diag.get("ece"),
                    "mce": calib_diag.get("mce"),
                    # Robustness
                    "is_robust": robust_diag.get("is_robust", True),
                    "worst_fold_auc": robust_diag.get("worst_fold_auc"),
                    "auc_cv_std": robust_diag.get("auc_cv_std"),
                    # Class overlap / problem difficulty
                    "easy_problem_detected": overlap_diag.get("easy_problem_detected", False),
                    # Mutual information (bits)
                    "mi_prob_label_bits": mi_diag.get("mi_prob_label_bits"),
                    "mi_prob_return_sign_bits": mi_diag.get("mi_prob_return_sign_bits"),
                    # Permutation-importance leakage diagnostics
                    "god_feature_suspected": leakage_diag.get("god_feature_suspected", False),
                    "leakage_baseline_auc": leakage_diag.get("baseline_auc"),
                    "leakage_delta_auc": leakage_diag.get("delta_auc"),
                    "leakage_top_feature": (leakage_diag.get("top_features") or [None])[0],
                    # Lag-1 stress test
                    "lag1_auc_base": lag_diag.get("auc_base"),
                    "lag1_auc_lag1": lag_diag.get("auc_lag1"),
                    "lag1_auc_diff": lag_diag.get("auc_diff"),
                    "lookahead_suspected": lag_diag.get("lookahead_suspected", False),
                    # Dummy-rule baseline
                    "auc_dummy": dummy_diag.get("auc_dummy"),
                    "auc_dummy_raw": dummy_diag.get("auc_dummy_raw"),
                    # sr_labeling_xgb meta-model diagnostics are no longer
                    # populated by this step; keys retained for backward
                    # compatibility but will typically be None.
                    "sr_labeling_xgb_best_auc": xgb_diag.get("best_auc"),
                    "sr_labeling_xgb_auc_improvement": xgb_diag.get("auc_improvement"),
                }

            with open(json_path, "w") as f:
                json.dump(output_dict, f, indent=2, default=str)
            tprint_success(f"💾 Saved best labeling HPO params to {json_path}")

            if standardized_json_path is not None:
                try:
                    with open(standardized_json_path, "w") as f_std:
                        json.dump(output_dict, f_std, indent=2, default=str)
                    tprint_success(f"💾 Saved standardized best params to {standardized_json_path}")
                except Exception as std_write_exc:
                    tprint_warning(f"⚠️ Failed to save standardized best params copy: {std_write_exc}")
        except Exception as save_exc:
            tprint_warning(f"⚠️ Failed to save best_params JSON: {save_exc}")
            json_path = None

        # ===== SAVE CANDIDATE POOL CSV =====
        csv_name = f"meta_labeling_hpo_candidate_pool_{symbol}_{timeframe}_{direction}_{timestamp}.csv"
        csv_path = outcomes_dir / csv_name

        try:
            if not candidate_pool:
                tprint_warning("⚠️ Candidate pool is empty; skipping candidate CSV export")
                csv_path = None
            else:
                candidate_df = pd.DataFrame(candidate_pool)

                if 'params' in candidate_df.columns:
                    params_df = pd.json_normalize(candidate_df['params'])
                    candidate_df = candidate_df.drop(columns=['params'])
                    candidate_df = pd.concat([candidate_df, params_df], axis=1)

                if 'edge' in candidate_df.columns:
                    candidate_df = candidate_df.sort_values('edge', ascending=False)
                elif 'combined' in candidate_df.columns:
                    candidate_df = candidate_df.sort_values('combined', ascending=False)

                candidate_df.to_csv(csv_path, index=False, float_format='%.6f')
                tprint_success(f"💾 Saved {len(candidate_pool)} candidate configs to {csv_path}")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save candidate pool CSV: {csv_exc}")
            csv_path = None

        # ===== SAVE PARETO FRONTIER CSV =====
        pareto_csv_name = f"meta_labeling_hpo_pareto_front_{symbol}_{timeframe}_{direction}_{timestamp}.csv"
        pareto_csv_path = outcomes_dir / pareto_csv_name

        try:
            if not pareto_front:
                pareto_csv_path = None
            else:
                pareto_data: list[dict[str, Any]] = []
                for sol in pareto_front:
                    row = dict(sol.metrics)
                    if sol.params:
                        row.update(sol.params)
                    pareto_data.append(row)

                pareto_df = pd.DataFrame(pareto_data)
                if 'edge' in pareto_df.columns:
                    pareto_df = pareto_df.sort_values('edge', ascending=False)
                elif 'combined' in pareto_df.columns:
                    pareto_df = pareto_df.sort_values('combined', ascending=False)

                pareto_df.to_csv(pareto_csv_path, index=False, float_format='%.6f')
                tprint_success(f"💾 Saved {len(pareto_front)} Pareto solutions to {pareto_csv_path}")
        except Exception as pareto_exc:
            tprint_warning(f"⚠️ Failed to save Pareto frontier CSV: {pareto_exc}")
            pareto_csv_path = None

        # ===== SAVE COMPREHENSIVE MARKDOWN REPORT =====
        md_name = f"meta_labeling_hpo_report_{symbol}_{timeframe}_{direction}_{timestamp}.md"
        md_path = outcomes_dir / md_name

        try:
            with open(md_path, "w") as f:
                f.write(f"# Meta-Labeling HPO Report\n\n")
                f.write(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
                f.write(f"**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe} | **Direction:** {direction}\n\n")
                f.write(f"---\n\n")

                # Summary
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Configurations Evaluated:** {len(candidate_pool)}\n")
                f.write(f"- **Total Trials:** {all_trials_count}\n")
                f.write(f"- **Best Edge:** {best_score:.6f}\n")
                f.write(f"- **Optimization Method:** Multi-Stage Bayesian TPE with Isotonic Calibration\n\n")

                # Multi-Stage Results
                f.write(f"## Multi-Stage HPO Results\n\n")
                f.write(f"| Stage | Complexity | Best Score | Trials |\n")
                f.write(f"|-------|------------|------------|--------|\n")
                for sr in stage_results:
                    f.write(f"| {sr['stage']} | {sr['complexity']} | {sr['best_score']:.4f} | {sr['trials']} |\n")
                f.write(f"\n")

                # Best Parameters
                f.write(f"## Best Parameters (Highest Edge)\n\n")
                f.write(f"```json\n")
                f.write(json.dumps(best_params, indent=2))
                f.write(f"\n```\n\n")

                # Per-regime metrics for best-edge configuration, if available
                best_regime_metrics = None
                try:
                    if candidate_pool:
                        best_candidate = max(
                            candidate_pool,
                            key=lambda x: x.get('edge', x.get('combined', 0)),
                        )
                        best_regime_metrics = best_candidate.get('per_regime_metrics')
                except Exception:
                    best_regime_metrics = None

                if best_regime_metrics:
                    f.write(f"## Per-Regime Metrics (Best Edge Configuration)\n\n")
                    f.write(f"| Regime | n_events | trades_per_day | mean_pos | mean_neg | edge | AUC |\n")
                    f.write(f"|--------|----------|----------------|----------|----------|------|-----|\n")
                    for reg_key, m in best_regime_metrics.items():
                        try:
                            f.write(
                                f"| {reg_key} | {int(m.get('n_events', 0))} | "
                                f"{float(m.get('trades_per_day', 0.0)):.3f} | "
                                f"{float(m.get('mean_pos', 0.0)):.5f} | "
                                f"{float(m.get('mean_neg', 0.0)):.5f} | "
                                f"{float(m.get('edge', 0.0)):.6f} | "
                                f"{float(m.get('auc', 0.5)):.3f} |\n"
                            )
                        except Exception:
                            continue
                    f.write("\n")

                # Knee Point Parameters
                if knee_solution:
                    f.write(f"## Recommended Parameters (Pareto Knee Point)\n\n")
                    f.write(f"Balanced trade-off between learnability and profitability:\n\n")
                    f.write(f"- **Learnability:** {knee_solution.metrics['learnability']:.4f}\n")
                    f.write(f"- **Profitability:** {knee_solution.metrics['profitability']:.4f}\n")
                    f.write(f"- **Mean AUC:** {knee_solution.metrics.get('mean_auc_raw', knee_solution.metrics.get('mean_auc_smooth', 0)):.4f}\n")
                    f.write(f"- **Sharpe (Winners):** {knee_solution.metrics.get('sharpe_pos', 0):.4f}\n\n")
                    f.write(f"```json\n")
                    f.write(json.dumps(knee_params, indent=2))
                    f.write(f"\n```\n\n")

                # Diagnostics Summary for Best Configuration
                if best_config_diagnostics:
                    f.write(f"## Diagnostics Summary (Best Configuration)\n\n")

                    filtering_diag = best_config_diagnostics.get("filtering_diagnostics", {})
                    calib_diag = best_config_diagnostics.get("calibration_diagnostics", {})
                    robust_diag = best_config_diagnostics.get("robustness_diagnostics", {})
                    overlap_diag = best_config_diagnostics.get("class_overlap_diagnostics", {})
                    mi_diag = best_config_diagnostics.get("mi_diagnostics", {})
                    xgb_diag = best_config_diagnostics.get("sr_labeling_xgb", {})
                    leakage_diag = best_config_diagnostics.get("leakage_diagnostics", {})
                    lag_diag = best_config_diagnostics.get("lag1_stress_test", {})
                    dummy_diag = best_config_diagnostics.get("dummy_baseline_diagnostics", {})
                    y_shuffle_diag = best_config_diagnostics.get("y_shuffle_test", {})

                    # Helper formatting
                    def _fmt_val(val: Any, digits: int = 4) -> str:
                        if val is None:
                            return "N/A"
                        try:
                            f_val = float(val)
                            if not np.isfinite(f_val):
                                return "nan"
                            return f"{f_val:.{digits}f}"
                        except Exception:
                            return "N/A"

                    # Explicit handler for degenerate calibration
                    def _fmt_calib(val: Any, is_degenerate: bool) -> str:
                        if is_degenerate and val is None:
                            return "Degenerate (Single Bin)"
                        return _fmt_val(val)

                    # Filtering diagnostics
                    f.write("### Filtering & AUC Inflation\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| AUC (full labels) | {_fmt_val(filtering_diag.get('auc_full'))} |\n")
                    f.write(f"| AUC (filtered labels) | {_fmt_val(filtering_diag.get('auc_filtered'))} |\n")
                    f.write(f"| AUC inflation (filtered - full) | {_fmt_val(filtering_diag.get('auc_inflation'))} |\n")
                    f.write(f"| Filtering is major contributor | {bool(filtering_diag.get('filtering_is_major_contributor', False))} |\n")
                    f.write(f"| AUC dominated by large moves | {bool(filtering_diag.get('auc_dominated_by_large_moves', False))} |\n")
                    f.write(f"| Precision collapse detected | {bool(filtering_diag.get('precision_collapse_detected', False))} |\n\n")

                    # Leakage diagnostics
                    if leakage_diag:
                        f.write("### Permutation-Importance Leakage Diagnostics\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(
                            f"| Baseline AUC (probe) | {_fmt_val(leakage_diag.get('baseline_auc'))} |\n"
                        )
                        f.write(
                            f"| AUC after dropping top-k features | {_fmt_val(leakage_diag.get('dropped_auc'))} |\n"
                        )
                        f.write(
                            f"| Delta AUC (baseline - dropped) | {_fmt_val(leakage_diag.get('delta_auc'))} |\n"
                        )
                        f.write(
                            f"| God feature suspected | {bool(leakage_diag.get('god_feature_suspected', False))} |\n"
                        )
                        top_feats = leakage_diag.get('top_features') or []
                        if top_feats:
                            f.write(
                                f"| Top features | {', '.join(map(str, top_feats))} |\n"
                            )
                        f.write("\n")

                    # Raw Metrics (Uncalibrated)
                    f.write("### Raw Metrics (Uncalibrated)\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Raw Probability Range | {_fmt_val(calib_diag.get('prob_range_raw', {}).get('max'))} - {_fmt_val(calib_diag.get('prob_range_raw', {}).get('min'))} |\n")
                    f.write(f"| Raw Brier Score | {_fmt_val(calib_diag.get('brier_score'))} |\n")
                    f.write(f"| Degenerate Calibration | {bool(calib_diag.get('degenerate_calibration'))} |\n\n")

                    # Calibration diagnostics
                    f.write("### Calibration Diagnostics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    is_degen = bool(calib_diag.get('degenerate_calibration', False))
                    f.write(f"| Well calibrated | {bool(calib_diag.get('is_well_calibrated', True))} |\n")
                    f.write(f"| Brier score | {_fmt_val(calib_diag.get('brier_score'))} |\n")
                    f.write(f"| Expected Calibration Error (ECE) | {_fmt_calib(calib_diag.get('ece'), is_degen)} |\n")
                    f.write(f"| Maximum Calibration Error (MCE) | {_fmt_calib(calib_diag.get('mce'), is_degen)} |\n\n")

                    # Mutual information
                    if mi_diag:
                        f.write("### Mutual Information (Meta-Score vs Targets)\n\n")
                        f.write("| Relationship | MI (bits) |\n")
                        f.write("|--------------|-----------|\n")
                        f.write(
                            f"| Probabilities → Label | {_fmt_val(mi_diag.get('mi_prob_label_bits'))} |\n"
                        )
                        f.write(
                            f"| Probabilities → Return sign | {_fmt_val(mi_diag.get('mi_prob_return_sign_bits'))} |\n\n"
                        )

                    # Robustness and class overlap
                    f.write("### Robustness & Class Overlap\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Robust across folds | {bool(robust_diag.get('is_robust', True))} |\n")
                    f.write(f"| Worst fold AUC | {_fmt_val(robust_diag.get('worst_fold_auc'))} |\n")
                    f.write(f"| AUC CV std | {_fmt_val(robust_diag.get('auc_cv_std'))} |\n")
                    f.write(f"| Easy problem detected | {bool(overlap_diag.get('easy_problem_detected', False))} |\n\n")

                    # Per-fold AUC summary (if available)
                    per_fold = robust_diag.get("per_fold_metrics") or []
                    if per_fold:
                        try:
                            f.write("#### Per-Fold AUC Summary\n\n")
                            f.write("| Fold | AUC | n_test | ECE | Net P&L per trade |\n")
                            f.write("|------|-----|--------|-----|-------------------|\n")
                            for fm in per_fold:
                                try:
                                    f.write(
                                        f"| {int(fm.get('fold', 0))} | "
                                        f"{_fmt_val(fm.get('auc'))} | "
                                        f"{int(fm.get('n_test', 0))} | "
                                        f"{_fmt_val(fm.get('ece'))} | "
                                        f"{_fmt_val(fm.get('net_pnl_per_trade'))} |\n"
                                    )
                                except Exception:
                                    continue
                            f.write("\n")
                        except Exception:
                            pass

                    # Volatility and regime-wise robustness
                    per_vol = robust_diag.get("per_volatility_regime") or {}
                    if per_vol:
                        f.write("#### AUC by Volatility Regime\n\n")
                        f.write("| Regime | AUC | n_events |\n")
                        f.write("|--------|-----|----------|\n")
                        for rname, rm in per_vol.items():
                            try:
                                f.write(
                                    f"| {rname} | {_fmt_val(rm.get('auc'))} | {int(rm.get('n_events', 0))} |\n"
                                )
                            except Exception:
                                continue
                        f.write("\n")

                    per_reg = robust_diag.get("per_regime_metrics") or {}
                    if per_reg:
                        f.write("#### AUC by Regime Label\n\n")
                        f.write("| Regime | AUC | n_events | Net P&L per trade |\n")
                        f.write("|--------|-----|----------|-------------------|\n")
                        for rname, rm in per_reg.items():
                            try:
                                f.write(
                                    f"| {rname} | {_fmt_val(rm.get('auc'))} | "
                                    f"{int(rm.get('n_events', 0))} | "
                                    f"{_fmt_val(rm.get('net_pnl_per_trade'))} |\n"
                                )
                            except Exception:
                                continue
                        f.write("\n")

                    # Y-shuffle sanity check
                    if y_shuffle_diag:
                        f.write("### Y-Shuffle Sanity Test\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(
                            f"| AUC with shuffled labels | {_fmt_val(y_shuffle_diag.get('auc'))} |\n"
                        )
                        f.write(
                            "\nA well-behaved model should have AUC≈0.5 under label shuffling; "
                            "any materially higher value would indicate leakage or mis-specification.\n\n"
                        )

                    # Lag-1 stress test
                    if lag_diag:
                        f.write("### Lag-1 Stress Test (Look-Ahead Bias)\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(
                            f"| AUC (base, t features) | {_fmt_val(lag_diag.get('auc_base'))} |\n"
                        )
                        f.write(
                            f"| AUC (lag-1 features) | {_fmt_val(lag_diag.get('auc_lag1'))} |\n"
                        )
                        f.write(
                            f"| AUC difference (base - lag1) | {_fmt_val(lag_diag.get('auc_diff'))} |\n"
                        )
                        f.write(
                            f"| Look-ahead suspected | {bool(lag_diag.get('lookahead_suspected', False))} |\n\n"
                        )

                    # Dummy-rule baseline
                    if dummy_diag:
                        f.write("### Dummy-Rule Volatility Baseline\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(
                            f"| AUC (raw, signed) | {_fmt_val(dummy_diag.get('auc_dummy_raw'))} |\n"
                        )
                        f.write(
                            f"| AUC (absolute, best side) | {_fmt_val(dummy_diag.get('auc_dummy'))} |\n"
                        )
                        f.write(
                            f"| Samples used | {int(dummy_diag.get('n_samples', 0))} |\n\n"
                        )

                    # sr_labeling_xgb diagnostics section intentionally omitted to
                    # avoid nested XGB HPO; downstream reports may still read
                    # sr_labeling_xgb_* keys from the JSON summary when present.

                # Underfit Diagnostics (if available)
                final_stage_candidates = [
                    c for c in candidate_pool
                    if c.get('model_complexity') == 'strong' and c.get('underfit_diagnostics')
                ]
                if final_stage_candidates:
                    f.write(f"## Underfit Diagnostics (Final Stage)\n\n")

                    # Get diagnostics from best strong candidate
                    best_strong = sorted(final_stage_candidates, key=lambda x: x['combined'], reverse=True)[0]
                    diag = best_strong.get('underfit_diagnostics', {})

                    if diag.get('is_underfit'):
                        f.write(f"**⚠️ Underfit Detected:** The model shows signs of underfitting.\n\n")
                        f.write(f"Indicators:\n")
                        for indicator in diag.get('underfit_indicators', []):
                            f.write(f"- {indicator}\n")
                        f.write(f"\n")
                    else:
                        f.write(f"**✅ No Significant Underfit:** The model appears well-fitted.\n\n")

                    # Learning curves by data fraction
                    if diag.get('learning_curve_fractions'):
                        f.write(f"### Learning Curves (Data Fractions)\n\n")
                        f.write(f"| Data Fraction | AUC |\n")
                        f.write(f"|---------------|-----|\n")
                        for frac, auc in sorted(diag['learning_curve_fractions'].items()):
                            f.write(f"| {frac:.0%} | {auc:.4f} |\n")
                        f.write(f"\n")

                    # Learning curves by depth
                    if diag.get('learning_curve_depths'):
                        f.write(f"### Learning Curves (Model Depths)\n\n")
                        f.write(f"| Depth | AUC |\n")
                        f.write(f"|-------|-----|\n")
                        for depth, auc in sorted(diag['learning_curve_depths'].items()):
                            f.write(f"| {depth} | {auc:.4f} |\n")
                        f.write(f"\n")

                    # Feature importance concentration
                    if diag.get('feature_importance_concentration') is not None:
                        f.write(f"**Feature Importance Concentration (Top 5):** {diag['feature_importance_concentration']:.1%}\n\n")

                    if diag.get('feature_group_importance'):
                        groups = diag['feature_group_importance']
                        f.write("**Feature Group Importance:**\n\n")
                        for g, share in groups.items():
                            f.write(f"- {g}: {share:.1%}\n")
                        f.write("\n")

                    if diag.get('top_feature_importances'):
                        f.write("**Top Features by Importance:**\n\n")
                        for item in diag['top_feature_importances'][:10]:
                            try:
                                name = item.get('name', '')
                                imp = float(item.get('importance', 0.0))
                                f.write(f"- {name}: {imp:.2%}\n")
                            except Exception:
                                continue
                        f.write("\n")

                    # Probe vs deep AUC diff
                    if diag.get('probe_vs_deep_auc_diff') is not None:
                        f.write(f"**Probe vs Deep Model AUC Improvement:** {diag['probe_vs_deep_auc_diff']:.1%}\n\n")

                # Regularization Checks Summary
                f.write(f"## Regularization & Scoring\n\n")
                f.write(f"### Realistic P&L Edge Metric\n\n")
                f.write(f"The primary scoring metric is the **Realistic P&L Edge**:\n\n")
                f.write(f"```\n")
                f.write(f"Edge = (Mean_Return_Label1 - Transaction_Cost) × max(0, 2×AUC - 1)\n")
                f.write(f"```\n\n")
                f.write(f"This metric penalizes 'profitable but unlearnable' strategies more realistically:\n")
                f.write(f"- If AUC = 0.5 (random), Edge = 0 regardless of profitability\n")
                f.write(f"- If AUC = 1.0 (perfect), you capture full mean return minus cost\n\n")
                f.write(f"### Regularization Checks\n\n")
                f.write(f"All configurations were evaluated with:\n\n")
                f.write(f"1. **Isotonic Calibration:** Probabilities calibrated to align with real expected returns\n")
                f.write(f"2. **Temporal Stability:** Rolling window AUC variance penalty\n")
                f.write(f"3. **Learnability Threshold:** Mean AUC < 0.7 heavily penalized\n")
                f.write(f"4. **Profit/Stop Constraint:** Profit threshold must be ≥ 1.5× stop threshold\n")
                f.write(f"5. **Label Balance:** Entropy-based balance scoring\n")
                f.write(f"6. **Early Stopping:** Per-trial and global early stopping to prevent overfitting\n\n")

                # Gate usage and artifacts
                if gate_stats:
                    f.write(f"## Gate Usage & Early-Exit Statistics\n\n")
                    f.write("| Gate | Count |\n")
                    f.write("|------|-------|\n")
                    for k, v in sorted(gate_stats.items()):
                        if k == "last_exception":
                            continue
                        try:
                            count_val = int(v)
                        except Exception:
                            continue
                        f.write(f"| {k} | {count_val} |\n")
                    f.write("\n")

                    if gate_stats.get("last_exception"):
                        f.write(
                            f"**Last exception (if any):** `{gate_stats['last_exception']}`\n\n"
                        )

                f.write(f"## Artifacts\n\n")
                f.write(f"- **Best Params JSON:** `{json_path.name if json_path else 'N/A'}`\n")
                f.write(f"- Y-shuffle sanity tests to ensure no trivial leakage\n")
                f.write(f"- Robustness diagnostics across CV folds and volatility regimes\n")
                f.write(f"- Dummy volatility baseline AUC to benchmark meta-model value-add\n")

                # Recommended next-step validation: meta-gated backtest
                f.write("\n## Recommended Next Step: Meta-Gated Backtest\n\n")
                f.write(
                    "To validate that the meta-labeling AUC and edge translate into tradable "
                    "performance, run the meta-gated backtest using the meta_gating_config "
                    "produced by feature_generation_meta_labeling_step. For example:\n\n"
                )
                f.write(
                    "```bash\n"
                    "python3 src/launcher/ares_launcher.py \\\n"
                    "  --step meta_gated_backtest \\\n"
                    f"  --symbol {symbol} --exchange {exchange} --timeframe {timeframe} --direction long --execution-mode full\n"
                    "```\n\n"
                )
                f.write(
                    "The meta-gated backtest report (meta_gated_backtest_report_*.md) provides "
                    "event-level P&L, trades-per-day, drawdowns, and cost stress tests for the "
                    "diagnostic gate implied by the best HPO configuration.\n"
                )

            tprint_success(f"📄 Saved comprehensive report to {md_path}")
        except Exception as md_exc:
            tprint_warning(f"⚠️ Failed to save markdown report: {md_exc}")
            md_path = None

        # Persist per-round HPO metrics to CSV for analysis
        csv_path = None
        try:
            round_results = getattr(optimizer, "round_results", [])
            if isinstance(round_results, list) and round_results:
                rows: list[dict[str, Any]] = []
                for rr in round_results:
                    rows.append(
                        {
                            "round": rr.get("round"),
                            "best_score": rr.get("best_score"),
                            "improvement": rr.get("improvement"),
                            "time_seconds": rr.get("time"),
                            "trials": rr.get("trials"),
                        }
                    )

                df_rounds = pd.DataFrame(rows)
                csv_name = (
                    f"meta_labeling_hpo_round_metrics_{symbol}_{timeframe}_{direction}_{timestamp}.csv"
                )
                csv_path = outcomes_dir / csv_name
                df_rounds.to_csv(csv_path, index=False)
                tprint_success(f"💾 Saved HPO round metrics to {csv_path}")
        except Exception as csv_exc:
            tprint_warning(f"⚠️ Failed to save HPO round metrics CSV: {csv_exc}")

        # Compute best edge from candidate pool
        best_edge = 0.0
        if candidate_pool:
            best_candidate = max(candidate_pool, key=lambda x: x.get('edge', x.get('combined', 0)))
            best_edge = best_candidate.get('edge', 0.0)

        # =========================================================================
        # POST-HPO MULTI-MODEL EVALUATION (NEW)
        # Train multiple ML models for SNR diagnostics and extensive backtesting
        # =========================================================================
        tprint_info("\n" + "=" * 70)
        tprint_info("📊 POST-HPO MULTI-MODEL EVALUATION")
        tprint_info("=" * 70)
        
        post_hpo_evaluation_results = {}
        try:
            run_post_hpo_models = config.get("run_post_hpo_models", True)
            
            if run_post_hpo_models and 'meta_features_diag' in dir() and meta_features_diag is not None:
                tprint_info("🔬 Running post-HPO model evaluation...")
                
                # Prepare data for evaluation
                labeled_mask_eval = binary_labels.notna()
                X_eval = meta_features_diag.loc[labeled_mask_eval].fillna(0)
                y_eval = binary_labels[labeled_mask_eval]
                returns_eval = realized_returns[labeled_mask_eval]

                t1_eval = None
                try:
                    if 'event_durations' in dir() and event_durations is not None:
                        dur_eval = (
                            event_durations.reindex(y_eval.index)
                            .fillna(1)
                            .astype(int)
                            .clip(lower=1)
                        )
                        t0_locs_eval = pd.Series(np.arange(len(market_data)), index=market_data.index)
                        start_locs_eval = t0_locs_eval.loc[y_eval.index].values
                        end_locs_eval = np.minimum(start_locs_eval + dur_eval.values, len(market_data) - 1)
                        t1_eval = pd.Series(market_data.index[end_locs_eval], index=y_eval.index)
                except Exception:
                    t1_eval = None
                
                # Sample weights if available - validate alignment
                weights_eval = None
                if 'sample_weights' in dir() and sample_weights is not None:
                    try:
                        if isinstance(sample_weights, pd.Series):
                            w_series = sample_weights
                        else:
                            w_series = None
                            try:
                                if 'l2_t_events' in dir() and l2_t_events is not None and len(sample_weights) == len(l2_t_events):
                                    w_series = pd.Series(np.asarray(sample_weights, dtype=float), index=l2_t_events)
                                elif 'final_t_events' in dir() and final_t_events is not None and len(sample_weights) == len(final_t_events):
                                    w_series = pd.Series(np.asarray(sample_weights, dtype=float), index=final_t_events)
                            except Exception:
                                w_series = None

                        if w_series is not None:
                            weights_eval = w_series.reindex(y_eval.index).values
                        elif isinstance(sample_weights, np.ndarray) and len(sample_weights) == len(y_eval):
                            weights_eval = np.asarray(sample_weights, dtype=float)
                        else:
                            weights_eval = None

                        if weights_eval is not None and not validate_sample_weight_alignment(weights_eval, y_eval):
                            tprint_warning(
                                "⚠️ Sample weights alignment validation failed in post-HPO evaluation. "
                                "Using unweighted evaluation."
                            )
                            weights_eval = None
                    except Exception as weight_exc:
                        tprint_warning(f"⚠️ Failed to extract sample weights for post-HPO eval: {weight_exc}")
                        weights_eval = None
                
                post_hpo_evaluation_results = run_post_hpo_evaluation(
                    X=X_eval,
                    y=y_eval,
                    realized_returns=returns_eval,
                    sample_weights=weights_eval,
                    t1=t1_eval,
                    n_splits=config.get("cv_splits", 5),
                    n_bags=config.get("n_bags", 10),
                    probability_threshold=config.get("probability_threshold", 0.5),
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction,
                    save_artifacts=True,
                    optimize_thresholds=config.get("optimize_post_hpo_thresholds", True),
                    enable_calibration=config.get("enable_post_hpo_calibration", True),
                    transaction_cost=float(DEFAULT_TRANSACTION_COST),
                )
                
                tprint_success("✅ Post-HPO model evaluation complete!")
            else:
                tprint_info("ℹ️ Skipping post-HPO model evaluation (disabled or no features available)")
        except Exception as post_hpo_exc:
            tprint_warning(f"⚠️ Post-HPO model evaluation failed: {post_hpo_exc}")
            post_hpo_evaluation_results = {"error": str(post_hpo_exc)}

        # =========================================================================
        # PARAMETER-OUTCOME CORRELATION ANALYSIS
        # =========================================================================
        tprint_info("\n" + "=" * 70)
        tprint_info("📈 PARAMETER-OUTCOME CORRELATION ANALYSIS")
        tprint_info("=" * 70)
        
        correlation_report = ""
        backtest_metrics_dict = None
        try:
            # Try to load meta_gated_backtest results if available
            tprint_info("🔍 Attempting to load meta_gated_backtest results for correlation...")
            try:
                from src.utils.pipeline_standards import PipelineStandards
                base_dir = PipelineStandards.build_path('reports', exchange=exchange, asset=symbol)
                backtest_dir = Path(base_dir) / "meta_gated_backtest"
            except Exception:
                backtest_dir = Path("outcomes")
            
            # Look for latest backtest metrics JSON
            backtest_pattern = f"meta_gated_backtest_metrics_{symbol}_{exchange}_{timeframe}_{direction}_*.json"
            backtest_files = sorted(backtest_dir.glob(backtest_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            
            if backtest_files:
                latest_backtest = backtest_files[0]
                tprint_info(f"   📂 Found backtest results: {latest_backtest.name}")
                try:
                    with open(latest_backtest, "r") as f:
                        backtest_data = json.load(f)
                    
                    # Map backtest metrics to candidate configs
                    # For now, we'll match the best candidate (since backtest uses best params)
                    # In future, could match by params hash if multiple configs were backtested
                    backtest_metrics_dict = {}
                    best_candidate = max(candidate_pool, key=lambda x: x.get('edge', x.get('combined', 0))) if candidate_pool else None
                    if best_candidate:
                        config_id = best_candidate.get('config_id', 'best')
                        backtest_metrics_dict[config_id] = {
                            'sharpe_trade': backtest_data.get('sharpe_trade'),
                            'mean_return_gated': backtest_data.get('mean_return_gated'),
                            'max_drawdown_event_time': backtest_data.get('max_drawdown_event_time'),
                            'hit_rate_gated': backtest_data.get('hit_rate_gated'),
                            'trades_per_day': backtest_data.get('trades_per_day'),
                            'coverage_gated': backtest_data.get('coverage_gated'),
                            'cost_adjusted_sharpe': backtest_data.get('enhanced_backtest', {}).get('cost_adjusted_sharpe'),
                            'cost_adjusted_return': backtest_data.get('enhanced_backtest', {}).get('cost_adjusted_return'),
                            'snr_positive': backtest_data.get('enhanced_snr', {}).get('snr_positive'),
                            'information_coefficient': backtest_data.get('enhanced_snr', {}).get('information_coefficient'),
                        }
                        tprint_success(f"   ✅ Loaded backtest metrics for correlation analysis")
                except Exception as bt_load_exc:
                    tprint_warning(f"   ⚠️ Failed to load backtest metrics: {bt_load_exc}")
            else:
                tprint_info("   ℹ️ No backtest results found (run meta_gated_backtest to enable correlation)")
        except Exception as bt_search_exc:
            tprint_warning(f"   ⚠️ Backtest search failed: {bt_search_exc}")
        
        try:
            if candidate_pool:
                corr_df, pval_df = compute_parameter_outcome_correlations(
                    candidate_pool,
                    backtest_metrics=backtest_metrics_dict
                )
                
                if not corr_df.empty:
                    correlation_report = generate_correlation_report(corr_df, pval_df)
                    tprint_info(correlation_report)
                    
                    # Save correlation matrix
                    corr_csv_path = outcomes_dir / f"hpo_param_outcome_correlations_{symbol}_{timeframe}_{timestamp}.csv"
                    corr_df.to_csv(corr_csv_path)
                    tprint_success(f"💾 Saved correlation matrix to {corr_csv_path}")
                    
                    # Save extended correlation matrix with backtest metrics if available
                    if backtest_metrics_dict:
                        backtest_corr_path = outcomes_dir / f"hpo_param_outcome_correlations_with_backtest_{symbol}_{timeframe}_{timestamp}.csv"
                        corr_df.to_csv(backtest_corr_path)
                        tprint_success(f"💾 Saved extended correlation matrix (with backtest) to {backtest_corr_path}")
                else:
                    tprint_warning("⚠️ No valid correlations computed")
        except Exception as corr_exc:
            tprint_warning(f"⚠️ Correlation analysis failed: {corr_exc}")

        best_params_path = standardized_json_path or json_path

        metrics: Dict[str, Any] = {
            "best_score": best_score,
            "best_edge": best_edge,
            "best_params": best_params,
            "best_params_json": str(best_params_path) if best_params_path is not None else None,
            "round_metrics_csv": str(csv_path) if csv_path is not None else None,
            "post_hpo_evaluation": post_hpo_evaluation_results.get("model_comparison", []),
            "recommended_diagnostics_path": diagnostics_path,
            "total_trials": all_trials_count,
            "stage_results": stage_results,
            "pareto_frontier_size": len(pareto_front),
            "candidate_pool_size": len(candidate_pool),
        }
        artifacts = {
            "best_params_json": str(best_params_path) if best_params_path is not None else None,
            "report_md": str(md_path)
        }
        return {"success": True, "metrics": metrics, "artifacts": artifacts}

def register_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the meta-labeling HPO sample weighted step in the registry.
    
    This is the CANONICAL HPO entry point for meta-labeling. All HPO step aliases
    route to MetaLabelingHPOSampleWeightedStep for consistency.
    """
    from src.training.steps.base_step import step_registry

    # Primary registration
    step_registry.register("meta_labeling_hpo_sample_weighted", MetaLabelingHPOSampleWeightedStep)
    
    # Backward compatibility aliases - route old names to weighted version
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb_weighted", MetaLabelingHPOSampleWeightedStep)
    
    tprint(
        "✅ Meta-labeling HPO sample weighted step registered as CANONICAL entry point "
        "(aliases: meta_labeling_hpo_sample_weighted, meta_labeling_hpo_experiment, "
        "sr_labeling_xgb, sr_labeling_xgb_weighted)",
        "SUCCESS"
    )


# Auto-register when module is imported
register_meta_labeling_hpo_sample_weighted_step()

