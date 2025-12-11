"""Meta-Labeling HPO Experiment Step.

This offline step performs hierarchical hyperparameter optimization over
labeling-specific parameters (triple-barrier / TPSL, horizon, and target
clipping) using the HierarchicalParameterOptimizer.

It is intentionally decoupled from standard training runs. Invoke it
explicitly via the launcher with an appropriate config. A simple config
flag `enable_labeling_hpo` can be used to disable the optimization and
exit early if desired.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import lightgbm as lgb

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

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
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
from src.training.steps.labeling import FeatureSetPersistence
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


logger = system_logger.getChild("MetaLabelingHPOExperiment")

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
MIN_EVENTS_PHASE1 = 200  # Minimum events for Phase 1 (sample count optimization)
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
                n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1
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
                random_state=42,
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
                    random_state=42,
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
                        random_state=42
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

        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
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
) -> float:
    """Optimizes for ROBUSTNESS.

    Args:
        stats: Dict with 'num_trades', 'trades_per_day', 'sharpe_ratio', etc.
        df_results: DataFrame containing per-trade results:
                    ['y_true', 'y_prob', 'ret_bps', 'regime']
                    Note: 'ret_bps' here is expected to be raw return float (e.g. 0.01)
        regime_col: Column name for regime labels
    """
    if df_results.empty or 'ret_bps' not in df_results.columns:
        return 0.0

    # --- 1. Robust Edge Calculation (Winsorised Expectancy) ---
    # Replace 'median_ret * sqrt(N)' with:
    returns_arr = df_results['ret_bps'].to_numpy(dtype=float)
    robust_edge_val = compute_robust_expectancy(returns_arr)
    # Scale by sqrt(N) to reward sample size (t-stat like scaling)
    robust_score = robust_edge_val * np.sqrt(stats.get('num_trades', 0))

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
            tprint_error(
                f"❌ Error in final FS subsampling: {e}; using full dataset"
            )
            return features, targets


    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hierarchical HPO over labeling parameters.

        Config keys (non-exhaustive):
        - symbol, exchange, timeframe: market context
        - enable_labeling_hpo: if False, step exits early
        - execution_mode: 'full'/'light'/'blank' for data loading scope
        """
        if not config.get("enable_labeling_hpo", True):
            tprint("ℹ️ Labeling HPO disabled via config.enable_labeling_hpo", "INFO")
            return {"success": True, "metrics": {}, "artifacts": {}, "skipped": True}

        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")

        hpo_feature_set: Optional[List[str]] = None
        try:
            persistence = FeatureSetPersistence()
            # Prefer the 70-feature production meta-labeling set when available.
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
        try:
            exec_mode = str(config.get("execution_mode", "blank")).lower()
            lookback_days_cfg = int(config.get("lookback_days", 0) or 0)
            if exec_mode == "blank" and lookback_days_cfg > 0:
                lookback_days = min(lookback_days_cfg, 160)
            else:
                lookback_days = lookback_days_cfg
            try:
                md_start = market_data.index.min()
                md_end = market_data.index.max()
                md_span_days = max(1, (md_end - md_start).days)
                tprint_info(
                    f"📅 HPO load: rows={len(market_data)}, "
                    f"start={md_start}, end={md_end}, span_days={md_span_days}"
                )
                if len(market_data) < 1000:
                    tprint_warning(
                        f"⚠️ [SAMPLE_STARVATION] HPO loaded only {len(market_data)} rows. "
                        f"LightGBM requires 1000+ for stable splits. Consider using 'blank' or 'full' mode."
                    )
            except Exception:
                pass
            if (
                lookback_days > 0
                and exec_mode in {"full", "blank"}
                and isinstance(market_data.index, pd.DatetimeIndex)
            ):
                end_ts = market_data.index.max()
                start_ts = end_ts - pd.Timedelta(days=lookback_days)
                orig_rows = len(market_data)
                try:
                    orig_span_days = max(
                        1,
                        (market_data.index.max() - market_data.index.min()).days,
                    )
                except Exception:
                    orig_span_days = -1

                mask = market_data.index >= start_ts
                if int(mask.sum()) > 0:
                    market_data = market_data.loc[mask].copy()
                    try:
                        new_span_days = max(
                            1,
                            (market_data.index.max() - market_data.index.min()).days,
                        )
                    except Exception:
                        new_span_days = -1
                    tprint_info(
                        f"⏱️ HPO lookback alignment: mode={exec_mode}, "
                        f"requested={lookback_days}d, span {orig_span_days}d → {new_span_days}d, "
                        f"rows {orig_rows}→{len(market_data)}",
                    )
                else:
                    tprint_warning(
                        f"⚠️ HPO lookback alignment requested {lookback_days}d but no data "
                        f"falls in that window; keeping original dataset (rows={orig_rows})",
                    )
        except Exception as lb_exc:
            tprint_warning(
                f"⚠️ Failed to apply lookback_days to HPO market_data; proceeding with raw window: {lb_exc}",
            )

        # Snapshot full-span market data for final diagnostics (two-stage, etc.).
        # This preserves the full lookback window (e.g. 3 years in FULL mode)
        # even if HPO itself uses a shorter multi-slice subset.
        market_data_full_for_diagnostics = market_data.copy()

        # Optional: restrict HPO evaluation to multiple non-consecutive slices
        # to better probe temporal robustness across distinct regimes without
        # requiring a single contiguous multi-year window.
        try:
            slice_days = int(config.get("hpo_slice_days", 40) or 0)
            slice_count = int(config.get("hpo_slice_count", 4) or 0)

            if (
                slice_days > 0
                and slice_count > 1
                and isinstance(market_data.index, pd.DatetimeIndex)
            ):
                full_start = market_data.index.min()
                full_end = market_data.index.max()
                full_span_days = max(1, (full_end - full_start).days)

                # Only slice when we have at least one full slice available
                if full_span_days < slice_days:
                    tprint_warning(
                        f"⚠️ HPO multi-slice skipped: span_days={full_span_days} < slice_days={slice_days} "
                        f"(rows={len(market_data)})"
                    )
                else:
                    min_required_span = slice_days * slice_count

                    # Build deterministic, approximately evenly spaced slices.
                    masks: List[pd.Series] = []
                    if full_span_days < min_required_span:
                        # Fallback: use the trailing min_required_span days as a
                        # single contiguous window (may be shorter than requested).
                        start_global = full_end - pd.Timedelta(days=min_required_span)
                        mask_global = market_data.index >= start_global
                        masks.append(mask_global)
                    else:
                        # Evenly distribute slice starts across the available span.
                        available_span = full_span_days - slice_days
                        step_days = available_span / float(slice_count - 1)

                        for i in range(slice_count):
                            offset_days = int(round(i * step_days))
                            slice_start = full_start + pd.Timedelta(days=offset_days)
                            slice_end = slice_start + pd.Timedelta(days=slice_days)
                            masks.append(
                                (market_data.index >= slice_start)
                                & (market_data.index <= slice_end)
                            )

                    if masks:
                        multi_mask = masks[0].copy()
                        for m in masks[1:]:
                            multi_mask |= m

                        orig_rows = len(market_data)
                        market_data = market_data.loc[multi_mask].copy()

                        try:
                            new_span_days = max(
                                1,
                                (market_data.index.max() - market_data.index.min()).days,
                            )
                        except Exception:
                            new_span_days = -1

                        tprint_info(
                            f"📆 HPO multi-slice selection: slices={slice_count}, "
                            f"slice_days={slice_days}, span_days={new_span_days}, "
                            f"rows {orig_rows}→{len(market_data)}",
                        )
        except Exception as ms_exc:
            tprint_warning(
                f"⚠️ Failed to apply multi-slice selection for HPO: {ms_exc}",
            )

        # Attach rolling HMM regimes (typically 1h) to the market_data frame so that
        # regime-aware features and thresholds can be evaluated during HPO.
        #
        # For HPO, we explicitly disable attaching rolling_hmm_regime_probabilities
        # to avoid loading deprecated probability artifacts while still using
        # the discrete regime labels.
        try:
            regime_cfg = dict(config)
            if "regime_timeframe" not in regime_cfg:
                regime_cfg["regime_timeframe"] = "1h"
            regime_cfg["attach_hmm_probabilities"] = False
            market_data = attach_rolling_hmm_regimes_to_market_data(
                self,
                market_data,
                regime_cfg,
            )
        except Exception as e_reg:
            tprint_warning(f"⚠️ Failed to attach rolling HMM regimes to market_data for HPO: {e_reg}")

        # ===== NEW: REGIME DISTRIBUTION DIAGNOSTICS =====
        # Log regime distribution to help diagnose why some regimes have zero events
        try:
            if "hmm_regime_label_1h" in market_data.columns:
                regime_col = market_data["hmm_regime_label_1h"]
                regime_counts = regime_col.value_counts(dropna=False).sort_index()
                total_rows = len(market_data)
                tprint_info(f"📊 [REGIME_DIAGNOSTICS] Total rows: {total_rows}")
                tprint_info(f"📊 [REGIME_DIAGNOSTICS] Regime distribution:")
                for regime_val, count in regime_counts.items():
                    pct = 100.0 * count / total_rows if total_rows > 0 else 0.0
                    tprint_info(f"   Regime {regime_val}: {count} rows ({pct:.1f}%)")

                # Check for missing regimes (0, 1, 2, 3, 4 expected)
                expected_regimes = set(range(5))
                observed_regimes = set(regime_counts.index.dropna().astype(int))
                missing_regimes = expected_regimes - observed_regimes
                if missing_regimes:
                    tprint_warning(
                        f"⚠️ [REGIME_DIAGNOSTICS] Missing regimes in data: {sorted(missing_regimes)}. "
                        f"These regimes will have zero events in HPO."
                    )

                # Check for regime imbalance
                if len(regime_counts) > 1:
                    max_count = regime_counts.max()
                    min_count = regime_counts[regime_counts > 0].min() if (regime_counts > 0).any() else 0
                    if min_count > 0 and max_count / min_count > 10:
                        tprint_warning(
                            f"⚠️ [REGIME_DIAGNOSTICS] Severe regime imbalance: "
                            f"max/min ratio = {max_count / min_count:.1f}x"
                        )
        except Exception as regime_diag_exc:
            tprint_warning(f"⚠️ Regime diagnostics failed: {regime_diag_exc}")

        # Attach specialist liquidity regime probabilities as additional regime features
        try:
            tprint_info("💧 Attempting to attach liquidity regime probabilities for HPO via specialist loader...")

            config_for_specialists = dict(config)
            config_for_specialists.setdefault("use_canonical_specialist_scalars", True)
            # For HPO we want ML Risk + Liquidity + other non-deprecated
            # specialists, but we explicitly disable the legacy specialists
            # whose artifacts are no longer maintained.
            config_for_specialists.setdefault("enable_risk_hmm_specialist", False)
            config_for_specialists.setdefault("enable_breakout_specialist", False)
            config_for_specialists.setdefault("enable_macro_trend_specialist", False)
            config_for_specialists.setdefault("enable_mr_trend_specialist", False)
            config_for_specialists.setdefault("enable_mean_reversion_specialist", False)

            specialist_df = get_specialist_models_outputs(
                artifact_router=self.artifact_router,
                training_index=market_data.index,
                config=config_for_specialists,
                logger=self.logger,
                strict=False,
            )

            if specialist_df is not None and not specialist_df.empty:
                prob_cols = [
                    c for c in specialist_df.columns
                    if c.startswith('liquidity_regime_') and 'prob_' in c
                ]

                if prob_cols:
                    liquidity_features = specialist_df[prob_cols].reindex(market_data.index, method='ffill')
                    tprint_info(f"   ↪ Selected {len(prob_cols)} liquidity regime probability columns: {prob_cols}")

                    for col in liquidity_features.columns:
                        market_data[f'liquidity_{col}'] = liquidity_features[col]

                    tprint_success(f"✅ Added {len(prob_cols)} liquidity regime probability features to market_data")
                else:
                    tprint_warning("⚠️ No liquidity regime probability columns found in specialist outputs for HPO")
            else:
                tprint_warning("⚠️ No specialist liquidity regime outputs found for HPO")

        except Exception as e_liquidity:
            tprint_warning(f"⚠️ Failed to attach specialist liquidity regime probabilities for HPO: {e_liquidity}")

        tprint_info(f"📊 Loaded market data from: {source} | rows={len(market_data)}")

        # Generate primary consensus signals using the production helper
        tprint_info("⚙️ Generating primary signals for HPO labeling runs…")
        primary_signals = generate_primary_signals(market_data.copy())

        # Filter primary signals based on direction if requested
        if "consensus" in primary_signals.columns:
            if direction == "long":
                primary_signals.loc[primary_signals["consensus"] <= 0, "consensus"] = 0
                tprint_info("   → Filtered primary signals for LONG direction (kept > 0)")
            elif direction == "short":
                primary_signals.loc[primary_signals["consensus"] >= 0, "consensus"] = 0
                tprint_info("   → Filtered primary signals for SHORT direction (kept < 0)")

        # Precompute volatility for Kalman smoothing and label normalization
        # IMPORTANT: NO FUTURE LEAKAGE - using backward-looking rolling window only
        # At time T, volatility_1d[T] uses returns from T-96 to T-1 (past data only)
        # This ensures labels at time T do not peek at future volatility
        log_ret = np.log(market_data["close"]).diff()
        volatility_1d = log_ret.rolling(96).std()

        # Verify no NaN at start could cause issues (first 96 bars will have partial window)
        n_valid_vol = int((~volatility_1d.isna()).sum())
        tprint_info(f"📊 Volatility computed: {n_valid_vol}/{len(volatility_1d)} valid values (backward-looking, no future leakage)")

        # Precompute span in days once (used by density penalty in objective)
        try:
            days_span = max(
                1,
                (market_data.index.max() - market_data.index.min()).days,
            )
        except Exception:
            days_span = 1

        stage1_enable_subsample = bool(config.get("stage1_enable_subsample", True))
        lookback_days_cfg = int(config.get("lookback_days", 0) or 0)
        if lookback_days_cfg > 0:
            default_stage1_window = min(days_span, lookback_days_cfg)
        else:
            default_stage1_window = days_span

        # Cap the default Stage 1 window to a shorter horizon so that the
        # initial screening runs on a smaller subset of the available data.
        stage1_default_cap_days = int(config.get("stage1_default_cap_days", 180))
        if stage1_default_cap_days > 0:
            default_stage1_window = min(default_stage1_window, stage1_default_cap_days)

        stage1_subsample_window_days = int(
            config.get("stage1_subsample_window_days", default_stage1_window)
        )

        # Precompute ATR for trailing profit simulation
        # Using 14-period True Range as standard
        high_prices = market_data["high"] if "high" in market_data.columns else market_data["close"]
        low_prices = market_data["low"] if "low" in market_data.columns else market_data["close"]
        close_prices = market_data["close"]

        tr1 = high_prices - low_prices
        tr2 = (high_prices - close_prices.shift(1)).abs()
        tr3 = (low_prices - close_prices.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Using 14-period rolling mean for ATR
        atr_series = true_range.rolling(window=14, min_periods=1).mean()

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
            # Ensure float32 for memory efficiency
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
                        "high": 0.035,
                    },
                    "target_signal_density": {
                        "type": "float",
                        "low": 2.0,  # RELAXED from 3.0 to allow sparser signals
                        "high": 12.0,  # EXPANDED to 12.0 per user request (aiming for ~6/day)
                    },
                    "min_event_spacing": {
                        "type": "int",
                        "low": 0,
                        "high": 0,
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
                    "profit_thr_base": {
                        "type": "float",
                        "low": 0.010,
                        "high": 0.025,
                    },
                    "stop_to_profit_ratio": {
                        "type": "float",
                        "low": 0.3,
                        "high": 0.67,
                    },
                    "trail_distance": {
                        "type": "float",
                        "low": 0.6,
                        "high": 1.2,
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
        # 3) Define objective function for labeling quality (with learnability)
        # ------------------------------------------------------------------

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
                # Enforce profit >= 1.5x stop constraint. During stages that do not
                # actively optimize profit_thr_base/stop_to_profit_ratio, fall back
                # to conservative defaults.
                profit_thr_base = float(params.get("profit_thr_base", 0.012))
                stop_ratio = float(params.get("stop_to_profit_ratio", 0.5))
                trail_dist = float(params.get("trail_distance", 0.0))

                # CONSTRAINT: Ensure profit is at least 1.5x stop
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)
                if profit_thr_base < 1.5 * stop_thr_base:
                    # Early exit: invalid RR geometry
                    tprint_warning(
                        f"[EARLY_EXIT_RR] Config rejected: profit {profit_thr_base:.4f} < 1.5x stop {stop_thr_base:.4f}"
                    )
                    gate_stats["rr_profit_vs_stop"] = gate_stats.get("rr_profit_vs_stop", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

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
                # require a minimum RR ~1.2 (≈1.05 net after fees).
                worst_rr = (profit_thr_base * profit_mult_min) / max(stop_thr_base * stop_mult_max, 1e-8)
                if worst_rr < 1.2:
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(
                            f"[EARLY_EXIT_RR] Rejecting config due to worst_rr={worst_rr:.3f} < 1.2 "
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
                target_signal_density = float(params.get("target_signal_density", 4.0))
                target_signal_density = max(3.0, min(5.0, target_signal_density))

                # --- Recompute realized returns ---
                # NO FUTURE LEAKAGE in volatility-based thresholds:
                # - volatility_1d is backward-looking (rolling 96-bar std)
                # - vol_baseline is backward-looking (rolling mean of past volatility)
                # - vol_factor at time T uses only volatility from T-vol_baseline_window to T
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

                # NEW: Regenerate primary signals if cusum_threshold differs from default
                # This allows HPO to explore different signal densities
                signals_to_use = primary_signals
                default_cusum = 0.015
                if abs(cusum_threshold - default_cusum) > 0.001:
                    try:
                        signals_to_use = generate_primary_signals(
                            market_data.copy(),
                            cusum_threshold=cusum_threshold,
                            target_trades_per_day=target_signal_density,
                        )
                    except Exception:
                        signals_to_use = primary_signals  # Fallback if regeneration fails

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
                n_raw_events = len(realized_returns)
                n_raw_labeled = int((~binary_labels.isna()).sum())
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_LABELS] raw_events={n_raw_events}, raw_labeled={n_raw_labeled}, "
                        f"profit_thr_base={profit_thr_base:.6f}, stop_thr_base={stop_thr_base:.6f}, "
                        f"econ_min_mult={econ_min_mult:.3f}, label_low_q={label_low_q:.3f}, label_high_q={label_high_q:.3f}",
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
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_LABELS] vol_scaled_events={n_vol_scaled_events}",
                    )

                # Decide whether to use regime-aware quantiles based on the
                # attached HMM regimes (typically 1h) on market_data.
                regimes_for_labeling = None
                if config.get("enable_regime_aware_quantiles", True) and "hmm_regime_label_1h" in market_data.columns:
                    regimes_for_labeling = market_data["hmm_regime_label_1h"]

                # Use rolling quantiles by default to match production and avoid look-ahead bias
                use_rolling = config.get("use_rolling_quantiles", True)
                rolling_lookback = int(config.get("rolling_quantile_lookback_bars", 3000))
                rolling_min_periods = int(config.get("rolling_quantile_min_periods", 300))

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

                n_quantile_non_nan = int((~quantile_labels.isna()).sum())
                unique_quantile_vals: list[int] = []
                if n_quantile_non_nan > 0:
                    try:
                        unique_quantile_vals = sorted(
                            pd.unique(quantile_labels.dropna().astype(int)).tolist()
                        )
                    except Exception:
                        unique_quantile_vals = []
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_LABELS] quantile_labels_non_nan={n_quantile_non_nan}, "
                        f"unique_labels={unique_quantile_vals}",
                    )

                # Guard: configurations that produce no labeled events at all are
                # rejected outright; sparse but non-zero densities are handled via
                # softer density penalties further down.
                labeled_mask = ~binary_labels.isna()
                n_events = int(labeled_mask.sum())

                effective_days_span = max(1.0, float(days_span))
                try:
                    labeled_idx = binary_labels.index[labeled_mask]
                    if len(labeled_idx) > 1:
                        total_seconds = (labeled_idx.max() - labeled_idx.min()).total_seconds()
                        if np.isfinite(total_seconds) and total_seconds > 0:
                            effective_days_span = max(1.0, total_seconds / 86400.0)
                except Exception:
                    pass

                events_per_day = n_events / max(effective_days_span, 1.0)
                if n_events == 0:
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(
                            f"[HPO_DEBUG_LABELS] rejecting config with zero labeled events: "
                            f"raw_labeled={n_raw_labeled}, vol_non_nan={n_vol_scaled_events}, "
                            f"quantile_non_nan={n_quantile_non_nan}",
                        )
                    tprint_warning(
                        f"[EARLY_EXIT_EVENTS] Zero labeled events: n_events={n_events}, "
                        f"events_per_day={events_per_day:.3f}, raw_labeled={n_raw_labeled}, "
                        f"vol_non_nan={n_vol_scaled_events}, quantile_non_nan={n_quantile_non_nan}",
                    )
                    gate_stats["events_zero"] = gate_stats.get("events_zero", 0) + 1
                    return -1e9

                # Lightweight density diagnostics for the first few configs:
                # report days_span and events/day so we can tune label density
                # targets and event filters more precisely.
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG_DENSITY] days_span={days_span}, effective_days_span={effective_days_span:.3f}, "
                        f"n_events={n_events}, events_per_day={events_per_day:.3f}",
                    )

                # --- Time-to-Outcome (TTO) metrics for constraints & penalties ---
                mean_tto = float("nan")
                timeout_rate = float("nan")
                tto_penalty = 0.0
                try:
                    if horizon > 0 and isinstance(event_durations, pd.Series):
                        event_mask_tto = labeled_mask & ~event_durations.isna()
                        if event_mask_tto.any():
                            tto_series = (event_durations[event_mask_tto] / float(horizon)).replace([np.inf, -np.inf], np.nan)
                            if len(tto_series) > 0:
                                mean_tto = float(tto_series.mean())

                        if exit_reasons is not None and isinstance(exit_reasons, pd.Series):
                            exit_events = exit_reasons[event_mask_tto]
                            timeout_rate = float((exit_events == 2).mean())

                    # Hard constraint on mean TTO to avoid pathologically slow exits
                    tto_hard_max = float(config.get("tto_max", 0.6))
                    if np.isfinite(mean_tto) and mean_tto > tto_hard_max:
                        tprint_warning(
                            f"[EARLY_EXIT_TTO] mean_tto={mean_tto:.3f} > tto_max={tto_hard_max:.2f} "
                            f"(n_events={n_events}, events_per_day={events_per_day:.3f}, horizon={horizon})"
                        )
                        gate_stats["tto_hard"] = gate_stats.get("tto_hard", 0) + 1
                        return {
                            'learnability': 0.0,
                            'profitability': -1e9,
                            'edge': -1e9,
                            'combined': -1e9,
                        }

                    # Soft TTO penalty (secondary importance): gently discourage
                    # configurations with mean TTO above a target.
                    if np.isfinite(mean_tto):
                        tto_target = float(config.get("tto_target", 0.4))
                        tto_penalty_weight = float(config.get("tto_penalty_weight", 20.0))
                        tto_excess = max(0.0, mean_tto - tto_target)
                        tto_penalty = tto_excess * tto_penalty_weight
                except Exception:
                    mean_tto = float("nan")
                    timeout_rate = float("nan")
                    tto_penalty = 0.0

                # Relaxed: Lower default min_events_gate for initial stages
                min_events_gate = int(config.get("hpo_min_events_gate", max(20, MIN_EVENTS_PHASE1 // 4)))
                if n_events < min_events_gate:
                    gate_stats["events_too_few_pre_cv"] = gate_stats.get("events_too_few_pre_cv", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # Relaxed: Broader density gates
                min_trades_per_day = float(config.get("hpo_min_trades_per_day_gate", 0.05))
                max_trades_per_day = float(config.get("hpo_max_trades_per_day_gate", 25.0))
                if events_per_day < min_trades_per_day or events_per_day > max_trades_per_day:
                    gate_stats["density_gate_pre_cv"] = gate_stats.get("density_gate_pre_cv", 0) + 1
                    # Softened penalty instead of hard -1e9 rejection for density
                    return {
                        'learnability': 0.0,
                        'profitability': -100.0,  # Soft rejection
                        'edge': -100.0,
                        'combined': -100.0,
                    }

                balance_score_pre = compute_label_entropy_score(binary_labels)
                if balance_score_pre <= 0.0:
                    gate_stats["entropy_gate_pre_cv"] = gate_stats.get("entropy_gate_pre_cv", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                returns_labeled_pre = realized_returns[labeled_mask]
                labels_labeled_pre = binary_labels[labeled_mask]
                r_pos_pre = returns_labeled_pre[labels_labeled_pre == 1]
                mean_pos_pre = float(r_pos_pre.mean()) if len(r_pos_pre) > 0 else 0.0
                if mean_pos_pre <= effective_tx_cost:
                    gate_stats["econ_gate_pre_cv"] = gate_stats.get("econ_gate_pre_cv", 0) + 1
                    return {
                        'learnability': 0.0,
                        'profitability': -1e9,
                        'edge': -1e9,
                        'combined': -1e9,
                    }

                # --- Kalman smoothing for meta probability proxy ---
                smoothed_labels, _ = kalman_smooth_labels(
                    binary_labels,
                    Q=kalman_Q,
                    R=kalman_R,
                    volatility=volatility_1d,
                )

                # Probabilities limited to [0, 1]
                prob_series = smoothed_labels.clip(0.0, 1.0)

                # Symmetric clipping before isotonic regression
                prob_clipped = prob_series.clip(iso_min_prob, iso_max_prob)

                # Fit probability→expected-return mapping on labeled events
                iso_reg = fit_probability_to_return_mapping(
                    probabilities=prob_clipped.values,
                    realized_returns=realized_returns.values,
                    method="isotonic",
                    econ_min_return_multiple=econ_min_mult,
                )

                # Translate to long/short targets using existing helper
                target_long, target_short = translate_to_targets_with_isotonic(
                    realized_returns=realized_returns,
                    probabilities=prob_clipped.values,
                    signals=primary_signals,
                    iso_regressor=iso_reg,
                )

                # Construct a unified target magnitude (as in diagnostics)
                target_mag = pd.Series(0.0, index=market_data.index)
                long_mask_meta = (target_long > 0).reindex(target_mag.index, fill_value=False)
                short_mask_meta = (target_short > 0).reindex(target_mag.index, fill_value=False)
                target_mag[long_mask_meta] = target_long.reindex(target_mag.index)[long_mask_meta]
                target_mag[short_mask_meta] = target_short.reindex(target_mag.index)[short_mask_meta]

                # Quantile clipping of non-zero targets (symmetric tails) and write
                # clipped values back into target_mag so downstream diagnostics see
                # the effect of winsorisation.
                target_nz = target_mag[target_mag > 0]
                if len(target_nz) >= 100:
                    low_val = target_nz.quantile(q_low)
                    high_val = target_nz.quantile(q_high)
                    if low_val < high_val:
                        target_nz_clipped = target_nz.clip(low_val, high_val)
                        target_mag.loc[target_nz_clipped.index] = target_nz_clipped

                # ===== REGRESSION TARGETS (EVENT-ONLY, DIAGNOSTIC) =====
                # Build an event-only, signed regression target from the
                # isotonic-based long/short targets. This is used only for
                # diagnostics and does not directly affect the main edge
                # metric, but provides a clearer view of payoff learnability
                # for regressors.
                try:
                    # Event mask and direction
                    event_mask_reg = labeled_mask.reindex(realized_returns.index, fill_value=False)
                    consensus_dir = primary_signals.get("consensus")
                    if isinstance(consensus_dir, pd.Series):
                        consensus_dir = consensus_dir.reindex(realized_returns.index)

                    if isinstance(consensus_dir, pd.Series):
                        long_event_mask = event_mask_reg & (consensus_dir > 0)
                        short_event_mask = event_mask_reg & (consensus_dir < 0)
                    else:
                        long_event_mask = event_mask_reg
                        short_event_mask = event_mask_reg & False

                    # Signed regression base target: long>0, short<0
                    # Phase 2 Fix: Use vol_scaled_returns DIRECTLY to avoid zero-target bug from isotonic mapping failure
                    # This ensures the regressor always has a valid magnitude target to learn.
                    y_reg_base = pd.Series(np.nan, index=realized_returns.index)
                    if vol_scaled_returns is not None:
                         # vol_scaled_returns is already signed (return / vol).
                         # For regression, we want this raw magnitude.
                         # We apply it to the event mask.
                         vs = vol_scaled_returns.reindex(realized_returns.index)
                         # Longs: we want positive return -> positive target
                         # Shorts: we want positive return (profit) -> positive target?
                         # Usually regressors predict "edge". Long edge = ret. Short edge = -ret.
                         # compute_vol_scaled_returns_for_events returns signed returns? checking...
                         # Assuming realized_returns handles direction?
                         # Let's use realized_returns directly to be safe, scaled by vol.

                         # Re-derive vol-scaled returns aligned with direction
                         # If consensus > 0 (Long), target = realized_return / vol
                         # If consensus < 0 (Short), target = realized_return / vol (since realized_return is P&L)
                         # Wait, realized_return in compute_realized_returns IS P&L (aligned with trade).
                         # So positive realized_return = Profit.
                         # So we just want realized_return / vol.

                         v_aligned = volatility_1d.reindex(realized_returns.index).replace(0, np.nan)
                         r_aligned = realized_returns
                         y_raw = r_aligned / (v_aligned + 1e-8)

                         y_reg_base.loc[event_mask_reg] = y_raw.loc[event_mask_reg]

                    # Drop NaNs to derive clipping/scaling statistics
                    y_ev = y_reg_base[event_mask_reg].dropna()

                    if len(y_ev) >= 50:
                        reg_cfg = config.get("regression_targets", {}) or {}
                        clip_low_q = float(reg_cfg.get("clip_low_q", 0.01))
                        clip_high_q = float(reg_cfg.get("clip_high_q", 0.99))

                        clip_low_q = max(0.0, min(0.2, clip_low_q))
                        clip_high_q = max(0.8, min(1.0, clip_high_q))
                        if clip_low_q >= clip_high_q:
                            clip_low_q, clip_high_q = 0.01, 0.99

                        lo = float(y_ev.quantile(clip_low_q))
                        hi = float(y_ev.quantile(clip_high_q))

                        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                            y_reg_base = y_reg_base.clip(lo, hi)
                            y_ev = y_reg_base[event_mask_reg].dropna()

                        # Robust scaling
                        med = float(y_ev.median())
                        q25 = float(y_ev.quantile(0.25))
                        q75 = float(y_ev.quantile(0.75))
                        iqr = q75 - q25
                        scale = iqr if iqr > 1e-8 else float(y_ev.mad()) if hasattr(y_ev, "mad") else 0.0

                        if scale > 0:
                            y_reg_scaled = (y_reg_base - med) / scale
                        else:
                            y_reg_scaled = y_reg_base * 0.0

                        # Store compact diagnostics in candidate_config later
                        # via local variables captured in this scope.
                        reg_target_stats = {
                            "n_events_reg": int(y_ev.shape[0]),
                            "mean_raw": float(y_ev.mean()),
                            "std_raw": float(y_ev.std()),
                            "min_raw": float(y_ev.min()),
                            "max_raw": float(y_ev.max()),
                            "median_raw": float(med),
                        }
                    else:
                        y_reg_scaled = y_reg_base
                        reg_target_stats = None
                except Exception:
                    y_reg_scaled = None
                    reg_target_stats = None

                # ===== LEARNABILITY ASSESSMENT WITH CALIBRATION =====
                # Create meta-features for this labeling configuration using the
                # same pipeline as the production meta-labeling step.
                # Use the pre-computed feature matrix passed via 'X'
                # This ensures consistent feature set across trials and avoids re-computation.
                meta_feature_cfg = config.get("meta_feature_engineering", {}) or {}

                # Align X to current labeled events
                # X (passed in) corresponds to X_dummy from the wrapper, which we replaced with X_features
                try:
                    # If X_train is a DataFrame (expected), use index alignment
                    if isinstance(X_train, pd.DataFrame):
                        common_idx = X_train.index.intersection(binary_labels.index)
                        meta_features_model_processed = X_train.loc[common_idx]
                        binary_labels = binary_labels.loc[common_idx]
                        realized_returns = realized_returns.loc[common_idx]
                        if isinstance(event_durations, pd.Series):
                            event_durations = event_durations.loc[common_idx]
                    else:
                        # Fallback if X_train got converted to numpy (unlikely with our change)
                        # We use build_meta_features_for_model only as last resort
                        tprint_warning("[HPO] X_train is not DataFrame, falling back to internal generation")
                        _, meta_features_model_processed, _, _ = build_meta_features_for_model(
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
                            meta_feature_cfg=meta_feature_cfg,
                        )
                except Exception as align_e:
                    tprint_warning(f"[HPO] Alignment failed: {align_e}")
                    meta_features_model_processed = pd.DataFrame()

                selected_feature_names = list(meta_features_model_processed.columns)

                # Use the fully processed feature matrix (winsorisation, robust
                # scaling, selection) for learnability and diagnostics, to
                # match the production training path.
                if hpo_feature_set:
                    available_cols = list(meta_features_model_processed.columns)
                    selected_cols = [c for c in hpo_feature_set if c in available_cols]
                    if len(selected_cols) >= 10:
                        X_for_learnability = meta_features_model_processed[selected_cols]
                        if debug_sample_count < debug_sample_limit:
                            tprint_info(
                                f"[HPO_FEATURES] Using fixed LGBM meta feature set for HPO "
                                f"({len(selected_cols)} columns, first={selected_cols[:5]})"
                            )
                    else:
                        X_for_learnability = meta_features_model_processed
                        if debug_sample_count < debug_sample_limit:
                            tprint_warning(
                                f"[HPO_FEATURES] Precomputed feature set has insufficient overlap "
                                f"with current meta-features (shared={len(selected_cols)}); "
                                f"using all features"
                            )
                else:
                    X_for_learnability = meta_features_model_processed

                if debug_sample_count < debug_sample_limit:
                    try:
                        y_counts_raw = binary_labels.value_counts(dropna=False).to_dict()
                    except Exception:
                        y_counts_raw = {}
                    tprint_info(
                        f"[HPO_LEARNABILITY_INPUT] n_features={X_for_learnability.shape[1]}, "
                        f"n_labels={int(binary_labels.notna().sum())}, y_counts={y_counts_raw}",
                    )
                    debug_sample_count += 1

                # Compute learnability score with isotonic calibration. Allow HPO to
                # tune the strength of signal-strength-based weighting.
                signal_strength_scale_max = float(params.get("signal_strength_scale_max", 1.5))
                if not np.isfinite(signal_strength_scale_max) or signal_strength_scale_max < 1.0:
                    signal_strength_scale_max = 1.5

                learnability_score, mean_auc, calibrated_probs, iso_reg_probe, fold_aucs, oof_probs = compute_learnability_with_calibration(
                    X=X_for_learnability,
                    y=binary_labels,
                    realized_returns=realized_returns,
                    model_complexity=model_complexity,
                    cv_splits=cv_splits,
                    time_aware_cv=True,
                    use_ensemble=use_ensemble,
                    signal_strength_scale_max=signal_strength_scale_max,
                    event_durations=event_durations,
                    market_index=market_data.index,
                    base_horizon_bars=horizon,
                    use_smoothed_brier_objective_lgbm=use_smoothed_brier_objective_lgbm,
                    scale_pos_weight=float(params.get("scale_pos_weight", 1.0)) if "scale_pos_weight" in params else None,
                    use_feature_selection=False,
                    use_resampling=use_resampling,
                )

                # ===== NEW: MUTUAL INFORMATION GATE =====
                # Require MI > 0.01 between probabilities and labels to ensure
                # the model predictions carry meaningful information
                mi_penalty = 0.0
                mi_value = 0.0
                try:
                    if calibrated_probs is not None and len(calibrated_probs) > 50:
                        from sklearn.metrics import mutual_info_score
                        # Discretize probabilities into bins for MI calculation
                        probs_finite = calibrated_probs[np.isfinite(calibrated_probs)]
                        y_aligned = binary_labels.dropna().values[:len(probs_finite)]
                        if len(probs_finite) >= 50 and len(y_aligned) >= 50:
                            prob_bins = np.digitize(probs_finite[:len(y_aligned)], bins=np.linspace(0, 1, 11))
                            mi_value = float(mutual_info_score(y_aligned.astype(int), prob_bins))
                            mi_threshold = float(config.get("mi_threshold", 0.01))
                            if mi_value < mi_threshold:
                                mi_penalty = (mi_threshold - mi_value) * 500.0  # Heavy penalty
                                if debug_sample_count < debug_sample_limit:
                                    tprint_warning(
                                        f"[MI_GATE] MI={mi_value:.4f} < {mi_threshold}: "
                                        f"Model predictions carry no information about labels"
                                    )
                except Exception as mi_exc:
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(f"[MI_GATE] MI calculation failed: {mi_exc}")

                # ===== NEW: PROBABILITY VARIANCE CHECK =====
                # Flag degenerate models that output near-constant probabilities
                prob_variance_penalty = 0.0
                prob_std = 0.0
                try:
                    if calibrated_probs is not None and len(calibrated_probs) > 50:
                        probs_finite = calibrated_probs[np.isfinite(calibrated_probs)]
                        if len(probs_finite) >= 50:
                            prob_std = float(np.std(probs_finite))
                            prob_std_threshold = float(config.get("prob_std_threshold", 0.05))
                            if prob_std < prob_std_threshold:
                                prob_variance_penalty = (prob_std_threshold - prob_std) * 300.0
                                if debug_sample_count < debug_sample_limit:
                                    tprint_warning(
                                        f"[PROB_VARIANCE] std(prob)={prob_std:.4f} < {prob_std_threshold}: "
                                        f"Model outputs near-constant probabilities (degenerate)"
                                    )
                except Exception:
                    pass

                # Ensure strict penalty for strictly constant or near-constant probabilities
                if calibrated_probs is not None and len(calibrated_probs) > 0:
                     try:
                         range_prob = float(np.nanmax(calibrated_probs) - np.nanmin(calibrated_probs))
                         if range_prob < 1e-6:
                             learnability_score -= 1000.0
                             if debug_sample_count < debug_sample_limit:
                                tprint_warning(f"[CONST_PROB] Penalty applied: range={range_prob:.9f}")
                     except (ValueError, RuntimeWarning):
                         pass

                # ===== NEW: SINGLE-BIN CALIBRATION PENALTY =====
                # Penalize configurations where ECE is degenerate (all samples in one bin)
                single_bin_penalty = 0.0
                try:
                    if calibrated_probs is not None and len(calibrated_probs) > 50:
                        probs_finite = calibrated_probs[np.isfinite(calibrated_probs)]
                        if len(probs_finite) >= 50:
                            # Check if all probabilities fall into a single bin
                            n_bins_check = 10
                            bin_edges = np.linspace(0, 1, n_bins_check + 1)
                            bin_counts = np.histogram(probs_finite, bins=bin_edges)[0]
                            non_empty_bins = np.sum(bin_counts > 0)
                            if non_empty_bins <= 2:  # Degenerate: 1-2 bins only
                                single_bin_penalty = 200.0 * (3 - non_empty_bins)
                                if debug_sample_count < debug_sample_limit:
                                    tprint_warning(
                                        f"[SINGLE_BIN] Only {non_empty_bins} non-empty probability bins: "
                                        f"Degenerate calibration (ECE meaningless)"
                                    )
                except Exception:
                    pass

                # Apply new penalties to learnability score
                learnability_score -= (mi_penalty + prob_variance_penalty + single_bin_penalty)

                # ===== AUC RANGE-BASED SCORING =====
                # Target AUC range: 0.55 - 0.67 (prop-shop acceptable range)
                #
                # AUC INTERPRETATION GUIDELINES:
                # < 0.54:  Too noisy - model cannot distinguish signal from noise
                # 0.54 - 0.62: Good edge, acceptable by prop shops
                # 0.60 - 0.67: Excellent, but check for leakage or horizon bias
                # > 0.70:  Suspicious - likely data leakage or look-ahead bias
                #
                # Within target range: higher is better
                # Outside target range: penalize and shift weight to edge/pnl/trades

                auc_in_target_range = (mean_auc >= 0.55) and (mean_auc <= 0.67)
                auc_penalty = 0.0
                auc_weight_multiplier = 1.0  # Will reduce AUC influence when outside range

                if mean_auc < 0.54:
                    # Too noisy - heavy penalty, strong shift to edge metrics
                    auc_penalty = (0.54 - mean_auc) * 15.0
                    auc_weight_multiplier = 0.3  # AUC has less sway, edge/pnl matter more
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(f"[AUC] {mean_auc:.3f} < 0.54: Too noisy, model cannot distinguish signal")
                elif mean_auc < 0.55:
                    # Slightly below target - mild penalty
                    auc_penalty = (0.55 - mean_auc) * 8.0
                    auc_weight_multiplier = 0.7
                elif mean_auc > 0.70:
                    # Suspicious - likely leakage, heavy penalty
                    auc_penalty = (mean_auc - 0.70) * 20.0
                    auc_weight_multiplier = 0.2  # Strongly discount AUC contribution
                    tprint_warning(
                        f"[AUC] {mean_auc:.3f} > 0.70: SUSPICIOUS - check for data leakage, "
                        f"look-ahead bias, or horizon issues"
                    )
                elif mean_auc > 0.67:
                    # Above excellent range - moderate penalty, check for issues
                    auc_penalty = (mean_auc - 0.67) * 10.0
                    auc_weight_multiplier = 0.5
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(f"[AUC] {mean_auc:.3f} > 0.67: Excellent but verify no leakage/horizon bias")
                else:
                    # In target range [0.55, 0.67]: reward higher AUC within range
                    # Bonus peaks at 0.62 (center of "good edge" zone)
                    distance_from_sweet_spot = abs(mean_auc - 0.62)
                    auc_penalty = -max(0, 0.05 - distance_from_sweet_spot) * 5.0  # Negative = bonus
                    auc_weight_multiplier = 1.0

                learnability_score -= auc_penalty

                # ===== ROBUSTNESS HARD CONSTRAINTS (CV-based) =====
                worst_fold_auc_cv: Optional[float] = None
                auc_cv_std: Optional[float] = None
                robustness_penalty = 0.0
                if isinstance(fold_aucs, np.ndarray) and fold_aucs.size > 0:
                    try:
                        worst_fold_auc_cv = float(np.nanmin(fold_aucs))
                        if fold_aucs.size > 1:
                            auc_cv_std = float(np.nanstd(fold_aucs))
                    except Exception:
                        worst_fold_auc_cv = None
                        auc_cv_std = None

                # Configurable thresholds: HPO-specific robustness gate (looser than
                # final diagnostics). When violated, apply a large penalty instead
                # of outright rejecting the configuration.
                base_min_worst_fold_auc = 0.45
                base_max_auc_cv_std = 0.12
                if model_complexity == "medium":
                    base_min_worst_fold_auc = 0.47
                    base_max_auc_cv_std = 0.10
                elif model_complexity == "strong":
                    base_min_worst_fold_auc = 0.50
                    base_max_auc_cv_std = 0.08

                min_worst_fold_auc = float(config.get("hpo_min_worst_fold_auc", base_min_worst_fold_auc))
                max_auc_cv_std = float(config.get("hpo_max_auc_cv_std", base_max_auc_cv_std))

                if (worst_fold_auc_cv is not None) and (auc_cv_std is not None):
                    is_robust_cv = (auc_cv_std < max_auc_cv_std) and (worst_fold_auc_cv >= min_worst_fold_auc)
                    if not is_robust_cv:
                        if debug_sample_count < debug_sample_limit:
                            tprint_warning(
                                f"[ROBUSTNESS] Penalizing config: worst_fold_auc={worst_fold_auc_cv:.3f}, "
                                f"auc_cv_std={auc_cv_std:.3f} (min_worst={min_worst_fold_auc:.3f}, max_std={max_auc_cv_std:.3f})"
                            )
                            debug_sample_count += 1

                        shortfall_auc = max(0.0, min_worst_fold_auc - float(worst_fold_auc_cv))
                        excess_std = max(0.0, float(auc_cv_std) - max_auc_cv_std)
                        default_penalty_weight = 360.0 # Increased from 300 (+20%)
                        if model_complexity == "medium":
                            default_penalty_weight = 480.0 # Increased from 400
                        elif model_complexity == "strong":
                            default_penalty_weight = 600.0 # Increased from 500

                        penalty_weight = float(config.get("hpo_robustness_penalty_weight", default_penalty_weight))
                        robustness_penalty = penalty_weight * (shortfall_auc + excess_std)

                # Compute label entropy/balance score
                balance_score = compute_label_entropy_score(binary_labels)

                # NOTE: Comprehensive diagnostics (filtering, calibration, robustness, overlap)
                # are computed ONLY for the best config at the end of HPO, not during
                # the search loop. This improves HPO performance significantly.
                # See post-HPO diagnostics section below.

                # ===== ECONOMIC PROFITABILITY =====
                # Compute economic separation metrics on labeled events
                returns_labeled = realized_returns[labeled_mask]
                labels_labeled = binary_labels[labeled_mask]

                # Mean return for label=1 vs label=0
                r_pos = returns_labeled[labels_labeled == 1]
                r_neg = returns_labeled[labels_labeled == 0]

                mean_pos = float(r_pos.mean()) if len(r_pos) > 0 else 0.0
                mean_neg = float(r_neg.mean()) if len(r_neg) > 0 else 0.0
                sep = mean_pos - mean_neg

                # Simple Sharpe for label=1 trades
                std_pos = float(r_pos.std()) if len(r_pos) > 1 else 0.0
                sharpe_pos = mean_pos / (std_pos + 1e-8) if std_pos > 0 else 0.0

                # Use effective transaction cost (HPO-tunable via tx_cost_mult)
                tx = float(effective_tx_cost)

                # Targeted debug logging for a small sample of trials
                if debug_sample_count < debug_sample_limit:
                    tprint_info(
                        f"[HPO_DEBUG] n_events={n_events}, events_per_day={events_per_day:.3f}, "
                        f"mean_pos={mean_pos:.6f}, tx={tx:.6f}, above_tx={mean_pos > tx}",
                    )
                    debug_sample_count += 1

                # Hard economic gate: positive bucket must beat transaction cost
                # Hard economic gate: positive bucket must beat transaction cost
                # RELAXED for Signal Fidelity: Allow slight negative edge if IC/AUC are high
                # We trust Stage 2 to filter these out.
                if mean_pos <= tx * 0.5: # Allow up to 50% loss of spread
                    if debug_sample_count < debug_sample_limit:
                        tprint_warning(
                            f"[EARLY_EXIT_ECON] Rejecting config: mean_pos={mean_pos:.6f} <= tx/2={tx*0.5:.6f} "
                            f"(n_events={n_events}, events_per_day={events_per_day:.3f})"
                        )
                        debug_sample_count += 1
                    gate_stats["econ_gate"] = gate_stats.get("econ_gate", 0) + 1
                    return {
                        'learnability_score': float(learnability_score), # Fix key name matching downstream
                        'combined_score': -1e9,
                        'loss': 1e9,
                        'status': 'fail',
                        'edge': -1e9,
                    }

                # Penalize configurations dominated by economically trivial events
                returns_labeled_nonnull = returns_labeled.dropna()
                if len(returns_labeled_nonnull) > 0:
                    small_band = tx
                    frac_small = float((returns_labeled_nonnull.abs() < small_band).mean())
                else:
                    frac_small = 1.0

                # ===== PRE- VS POST-FILTER DIAGNOSTICS (RETENTION & SNR) =====
                try:
                    pre_mask = ~realized_returns.isna()
                    n_pre_total = int(pre_mask.sum())

                    if n_pre_total > 0:
                        pre_returns = realized_returns[pre_mask]
                        raw_label_pre = (pre_returns > tx).astype(int)

                        n_pre_pos = int((raw_label_pre == 1).sum())
                        n_pre_neg = int((raw_label_pre == 0).sum())

                        n_post_total = int(n_events)
                        n_post_pos = int((labels_labeled == 1).sum())
                        n_post_neg = int((labels_labeled == 0).sum())

                        retention_total = n_post_total / max(n_pre_total, 1)
                        retention_pos = n_post_pos / max(n_pre_pos, 1) if n_pre_pos > 0 else 0.0
                        retention_neg = n_post_neg / max(n_pre_neg, 1) if n_pre_neg > 0 else 0.0

                        pre_pos_ret = pre_returns[raw_label_pre == 1]
                        pre_neg_ret = pre_returns[raw_label_pre == 0]

                        def _safe_stats(x: pd.Series) -> tuple[float, float]:
                            return (
                                float(x.mean()) if len(x) > 0 else 0.0,
                                float(x.std() if len(x) > 1 else 0.0),
                            )

                        pre_pos_mean, pre_pos_std = _safe_stats(pre_pos_ret)
                        pre_neg_mean, pre_neg_std = _safe_stats(pre_neg_ret)
                        post_pos_mean, post_pos_std = _safe_stats(r_pos)
                        post_neg_mean, post_neg_std = _safe_stats(r_neg)

                        def _cohens_d(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
                            if n1 <= 1 or n2 <= 1:
                                return float('nan')
                            pooled = ((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(n1 + n2 - 2, 1)
                            if pooled <= 0:
                                return float('nan')
                            return (m1 - m2) / np.sqrt(pooled)

                        d_pre = _cohens_d(
                            pre_pos_mean,
                            pre_pos_std,
                            max(len(pre_pos_ret), 1),
                            pre_neg_mean,
                            pre_neg_std,
                            max(len(pre_neg_ret), 1),
                        )
                        d_post = _cohens_d(
                            post_pos_mean,
                            post_pos_std,
                            max(len(r_pos), 1),
                            post_neg_mean,
                            post_neg_std,
                            max(len(r_neg), 1),
                        )

                        snr_pre = pre_pos_mean / (pre_pos_std + 1e-8) if pre_pos_std > 0 else 0.0
                        snr_post = post_pos_mean / (post_pos_std + 1e-8) if post_pos_std > 0 else 0.0
                    else:
                        n_pre_total = 0
                        retention_total = 0.0
                        retention_pos = 0.0
                        retention_neg = 0.0
                        d_pre = float('nan')
                        d_post = float('nan')
                        snr_pre = 0.0
                        snr_post = 0.0
                except Exception:
                    n_pre_total = 0
                    retention_total = 0.0
                    retention_pos = 0.0
                    retention_neg = 0.0
                    d_pre = float('nan')
                    d_post = float('nan')
                    snr_pre = 0.0
                    snr_post = 0.0

                retention_asym_penalty = 0.0
                if retention_total > 0.0 and retention_pos > 0.0:
                    if retention_neg <= 0.0:
                        retention_asym_penalty = 1.0
                    else:
                        ratio_pos_neg = retention_pos / max(retention_neg, 1e-6)
                        ratio_cap = float(config.get("retention_ratio_cap", 6.0))
                        if ratio_cap < 1.0:
                            ratio_cap = 1.0
                        if ratio_pos_neg > ratio_cap:
                            retention_asym_penalty = min(2.0, ratio_pos_neg - ratio_cap)

                # Event density penalty: prefer ~1–4 trades/day (centered near ~2)
                # Slightly stricter lower bound to discourage very sparse regimes
                trades_per_day = n_events / max(effective_days_span, 1.0)
                penalty_density = 0.0
                if trades_per_day < 0.5:
                    # Moderate penalty for sparse regimes (threshold raised from 0.3)
                    penalty_density += (0.5 - trades_per_day) * 10.0

                density_retention_penalty = 0.0

                penalty_noise = frac_small * 10.0

                # Top-bucket economics (top 10% by smoothed probability) to capture
                # how good the very best signals are.
                top_bucket_mean = 0.0
                top_bucket_sharpe = 0.0
                try:
                    prob_array = prob_clipped.values.astype(float)
                    if np.isfinite(prob_array).any():
                        q90 = np.nanquantile(prob_array, 0.9)
                        top_mask = (prob_array >= q90) & labeled_mask.to_numpy()
                        top_returns = realized_returns[top_mask]
                        top_returns = top_returns.dropna()
                        if len(top_returns) >= 20:
                            top_bucket_mean = float(top_returns.mean())
                            top_std = float(top_returns.std())
                            top_bucket_sharpe = top_bucket_mean / (top_std + 1e-8) if top_std > 0 else 0.0
                except Exception:
                    top_bucket_mean = 0.0
                    top_bucket_sharpe = 0.0

                # Profitability score: emphasize separation and Sharpe, subtract penalties,
                # and reward strong top-bucket performance. TTO penalty is secondary but
                # present so that extremely slow configurations are disfavoured.
                profitability_score = (
                    sep * 100.0
                    + sharpe_pos * 10.0
                    + top_bucket_sharpe * 15.0
                    + top_bucket_mean * 1000.0
                    - penalty_density
                    - penalty_noise
                    - tto_penalty
                )

                # Extra penalty when label balance is extreme (balance_score == 0)
                if balance_score <= 0.0:
                    learnability_score -= 0.5

                # Simple power heuristic: required samples for ~80% power based on post-filter effect size
                try:
                    if np.isfinite(d_post) and d_post != 0.0:
                        n_required_80 = 16.0 / (d_post ** 2)
                    else:
                        n_required_80 = float('inf')
                except Exception:
                    n_required_80 = float('inf')

                # ===== REGULARIZATION CHECKS =====
                # Temporal stability check (rolling window AUC variance)
                auc_variance = 0.0
                try:
                    window_size = max(100, n_events // 5)
                    n_windows = min(5, n_events // window_size)

                    if n_windows >= 2:
                        window_aucs = []
                        for w in range(n_windows):
                            start_idx = w * window_size
                            end_idx = min((w + 1) * window_size, n_events)
                            window_labels = labels_labeled.iloc[start_idx:end_idx]

                            if len(window_labels.unique()) >= 2 and len(window_labels) >= 20:
                                # Compute simple correlation as AUC proxy
                                window_returns = returns_labeled.iloc[start_idx:end_idx]
                                try:
                                    window_auc = abs(window_labels.corr(window_returns))
                                    window_aucs.append(window_auc if not np.isnan(window_auc) else 0.5)
                                except:
                                    window_aucs.append(0.5)

                        if len(window_aucs) >= 2:
                            auc_variance = float(np.var(window_aucs))
                            # Penalize high variance (instability across time) more strongly
                            temporal_stability_penalty = auc_variance * 30.0
                            profitability_score -= temporal_stability_penalty
                except Exception:
                    pass  # Skip if temporal check fails

                # Reference trade count for Sharpe-like scaling of edge. We
                # center this around a target trades/day consistent with the
                # density band above so that edge mildly rewards configurations
                # that achieve a healthy number of good trades.
                target_trades_per_day = float(config.get("edge_target_trades_per_day", 2.0))
                reference_trades = max(1.0, float(effective_days_span) * target_trades_per_day)

                # ===== ROBUST HPO OBJECTIVE (Simulated Trading) =====
                # 1. Simulate trades with concurrency=1 and probability sizing
                # Need event data: entry_time, exit_time (derived from duration), prob, realized_return, y_true
                # y_true is labels_labeled, y_prob is calibrated_probs

                sim_score = 0.0
                try:
                    # Align data for simulation
                    sim_indices = labels_labeled.index
                    if len(sim_indices) > 0:
                        # Reconstruct exit times from durations (assuming 15m bars for simplicity if index is datetime)
                        # If index is just integer, exit_time is index + duration
                        if isinstance(sim_indices, pd.DatetimeIndex):
                            # duration is in bars. Approx 15 min per bar.
                            durations = event_durations.reindex(sim_indices).fillna(1).astype(int)
                            exit_times = sim_indices + pd.to_timedelta(durations * 15, unit='min')
                        else:
                            # Use integer indices if not datetime
                            durations = event_durations.reindex(sim_indices).fillna(1).astype(int)
                            exit_times = sim_indices + durations

                        # Filter calibrated probs to match labeled set
                        # calibrated_probs is aligned to X_clean which should match labels_labeled
                        # (X_clean is derived from valid_mask which matches y_clean/labels_labeled)

                        # Note: 'calibrated_probs' passed here comes from 'compute_learnability_with_calibration'
                        # which returns an array aligned with X_clean/y_clean.

                        # Build DataFrame for simulation
                        sim_df = pd.DataFrame({
                            'entry_time': sim_indices,
                            'exit_time': exit_times,
                            'prob': calibrated_probs if len(calibrated_probs) == len(sim_indices) else np.full(len(sim_indices), 0.5),
                            'realized_return': returns_labeled.values, # This is net return per unit
                            'y_true': labels_labeled.values,
                            'regime': np.zeros(len(sim_indices), dtype=int)
                        })

                        # Add real regime data derived from volatility
                        # This enables the regime-based gating in the objective function
                        if volatility_1d is not None:
                            try:
                                # Align volatility to event times
                                vol_events = volatility_1d.reindex(sim_indices).fillna(0.0)

                                # Calculate thresholds from global volatility (to keep regimes consistent)
                                # 33rd and 67th percentiles define Low, Medium, High
                                v_low = float(volatility_1d.quantile(0.33))
                                v_high = float(volatility_1d.quantile(0.67))

                                # Assign regimes: 0=Low, 1=Med, 2=High
                                # Default is 0 (Low) from initialization
                                regime_arr = sim_df['regime'].values
                                vol_arr = vol_events.values

                                mask_med = (vol_arr > v_low) & (vol_arr <= v_high)
                                mask_high = (vol_arr > v_high)

                                regime_arr[mask_med] = 1
                                regime_arr[mask_high] = 2

                                sim_df['regime'] = regime_arr
                            except Exception:
                                pass # Keep default regimes if calculation fails

                        # Run Simulation
                        executed_trades = simulate_concurrent_trades(
                            sim_df,
                            max_concurrency=1,
                            transaction_cost=tx
                        )

                        if not executed_trades.empty:
                            # Calculate stats for objective function
                            n_exec = len(executed_trades)

                            # Calculate Sharpe on simulated PnL (ret_bps)
                            pnl_series = executed_trades['ret_bps']
                            mean_pnl = pnl_series.mean()
                            std_pnl = pnl_series.std()
                            sim_sharpe = mean_pnl / (std_pnl + 1e-9) if std_pnl > 0 else 0.0

                            sim_trades_per_day = n_exec / max(effective_days_span, 1.0)

                            sim_stats = {
                                'num_trades': n_exec,
                                'trades_per_day': sim_trades_per_day,
                                'sharpe_ratio': sim_sharpe
                            }

                            # Compute final robust score
                            # df_results needs ['y_true', 'y_prob', 'ret_bps', 'regime']
                            # executed_trades has these (from sim_df + 'ret_bps')
                            # Map 'prob' -> 'y_prob'
                            executed_trades['y_prob'] = executed_trades['prob']

                            sim_score = compute_robust_hpo_objective(
                                stats=sim_stats,
                                df_results=executed_trades,
                                regime_col='regime'
                            )
                        else:
                            sim_score = 0.0
                    else:
                        sim_score = 0.0
                except Exception:
                    sim_score = 0.0

                edge_score = sim_score

                # ===== CALIBRATION-AWARE ADJUSTMENT OF EDGE =====
                # Use calibration diagnostics (weighted Brier and ECE) to softly
                # down-weight edge for miscalibrated models. We do this *after*
                # temporal stability but before scaling and combining with AUC.
                try:
                    calib_edge_penalty = 1.0
                    if weighted_brier is not None:
                        # Map weighted_brier in [brier_target, brier_max] to a factor in [1.0, 0.5]
                        brier_target = float(config.get("calibration_brier_target", 0.18))
                        brier_max = float(config.get("calibration_brier_max", 0.35))
                        if brier_max <= brier_target:
                            brier_max = brier_target + 1e-3
                        brier_excess = max(0.0, float(weighted_brier) - brier_target)
                        brier_span = max(brier_max - brier_target, 1e-6)
                        brier_ratio = min(1.0, brier_excess / brier_span)
                        # Linearly shrink edge to 50% at the worst Brier in the band.
                        calib_edge_penalty *= (1.0 - 0.5 * brier_ratio)

                    if ece_norm is not None:
                        # ece_norm already in [0, 1]; shrink edge up to another 30% at ece_norm=1.
                        calib_edge_penalty *= (1.0 - 0.3 * float(ece_norm))

                    # Ensure non-negative factor
                    calib_edge_penalty = max(0.0, calib_edge_penalty)
                    edge_score *= calib_edge_penalty
                except Exception:
                    # If anything goes wrong in calibration-based adjustment, keep raw edge_score.
                    pass

                # Scale edge for combined metric (multiply by 1000 to make comparable)
                edge_scaled = edge_score * 1000.0

                # Additional hard penalties for pathological configurations (no positive
                # or negative bucket), while still keeping them in the candidate pool
                # for diagnostics.
                if len(r_pos) == 0 or len(r_neg) == 0:
                    profitability_score = -1e9

                # ===== COMBINED OBJECTIVE WITH RETENTION REGULARIZATION =====
                # New formulation: objective adjusts the filtered AUC based on how far
                # the retention rate falls below a soft target.
                #
                #   ret_target = 0.35  (35% of pre-events retained)
                #   penalty    = alpha * (max(0, ret_target - retention_total))^2
                #   objective  = AUC_filtered -  penalty
                #
                # This penalizes overly aggressive filtering (very low retention) while
                # not rewarding sparse configs purely via 1/retention.

                # Retention regularization parameter (configurable)
                alpha_retention = float(config.get("retention_regularization_alpha", 0.05))
                alpha_retention = max(0.01, min(0.10, alpha_retention))  # Clamp to [0.01, 0.10]

                # Soft target for overall retention (fraction of pre-events kept)
                retention_target = float(config.get("retention_target", 0.35))
                retention_target = max(0.05, min(0.80, retention_target))

                # Compute nonlinear shortfall penalty relative to the target
                # retention_total is already computed above as n_post_total / n_pre_total
                if retention_total > 0:
                    retention_shortfall = max(0.0, retention_target - float(retention_total))
                    # Quadratic penalty so that very low retention is punished more strongly
                    retention_penalty = alpha_retention * (retention_shortfall ** 2)
                else:
                    # If no events survive, apply maximal penalty against the target
                    retention_penalty = alpha_retention * retention_target

                # Cap retention penalty to avoid dominating the objective
                retention_penalty = min(retention_penalty, 0.5)

                # ===== UNIFIED AUC ADJUSTMENT =====
                # Combine AUC range penalties directly into adjusted AUC instead of
                # having separate auc_penalty and learnability_bonus terms.
                # This simplifies the objective and avoids double-counting.

                auc_range_adjustment = 0.0
                if mean_auc < 0.54:
                    # Too noisy: heavy penalty incorporated into AUC
                    auc_range_adjustment = -(0.54 - mean_auc) * 0.5
                elif mean_auc > 0.70:
                    # Suspicious (likely leakage): heavy penalty
                    auc_range_adjustment = -(mean_auc - 0.70) * 1.0
                elif mean_auc > 0.67:
                    # Above excellent range: moderate penalty
                    auc_range_adjustment = -(mean_auc - 0.67) * 0.3
                elif auc_in_target_range:
                    # In target range [0.55, 0.67]: small bonus for 0.60-0.62 sweet spot
                    if mean_auc >= 0.58 and mean_auc <= 0.64:
                        auc_range_adjustment = 0.02  # Small bonus for ideal range

                # Final retention-adjusted AUC (unified primary signal)
                auc_ret_raw = mean_auc + auc_range_adjustment - retention_penalty
                auc_cap = 0.65
                auc_retention_adjusted = min(auc_ret_raw, auc_cap)

                # Trade density bonus: stronger when AUC is outside target range
                # This gives edge/pnl/trades more influence when AUC is penalized
                # User Request (Step 451): Reduce importance of pure edge by ~30% to favor robustness
                base_edge_multiplier = 0.7
                edge_weight = (1.0 + (1.0 - auc_weight_multiplier) * 0.5) * base_edge_multiplier
                density_weight = 1.0 + (1.0 - auc_weight_multiplier) * 1.0  # Up to 2x density weight

                balance_weight = float(config.get("balance_score_weight", 0.15))
                if balance_weight < 0.0:
                    balance_weight = 0.0
                if balance_weight > 0.3:
                    balance_weight = 0.3
                balance_term = balance_weight * 10.0 * (balance_score - 0.5)

                retention_asym_weight = float(config.get("retention_asym_weight", 5.0))
                if retention_asym_weight < 0.0:
                    retention_asym_weight = 0.0

                # Trades/day bonus for being in sweet spot (1.5-5 trades/day)
                trades_bonus = 0.0
                if trades_per_day >= 1.5 and trades_per_day <= 5.0:
                    # Peak bonus at 2.5 trades/day (midpoint)
                    trades_bonus = max(0, 1.0 - abs(trades_per_day - 2.5) / 2.5) * 50.0

                # Geometry penalty: if profit threshold is unrealistically large
                # relative to the average trailing stop distance (ATR-based) while
                # overall retention is below target, down-weight the configuration.
                trail_geom_penalty = 0.0
                try:
                    if trail_dist > 0.0 and isinstance(atr_series, pd.Series) and "close" in market_data.columns:
                        atr_events = atr_series[labeled_mask]
                        price_events = market_data["close"][labeled_mask]
                        denom = price_events.abs().replace(0.0, np.nan)
                        avg_trail_pct = float(((trail_dist * atr_events) / (denom + 1e-8)).replace([np.inf, -np.inf], np.nan).mean())

                        if np.isfinite(avg_trail_pct) and avg_trail_pct > 0:
                            k_trail = float(config.get("trail_profit_ratio_k", 3.5))
                            # Only activate when retention is below target (over-selection regime)
                            if float(retention_total) < retention_target:
                                threshold = k_trail * avg_trail_pct
                                if profit_thr_base > threshold:
                                    excess_ratio = (profit_thr_base / max(threshold, 1e-8)) - 1.0
                                    weight = float(config.get("trail_penalty_weight", 80.0))
                                    trail_geom_penalty = max(0.0, excess_ratio) * weight
                except Exception:
                    trail_geom_penalty = 0.0

                # Diagnostic penalties: when aggressive filtering or ultra-easy
                # problems are detected (only computed when compute_diagnostics
                # is True to keep HPO runtime manageable).
                diagnostics_penalty = 0.0
                if compute_diagnostics:
                    try:
                        econ_floor_local = econ_min_mult * effective_tx_cost
                        y_full_local = pd.Series(np.nan, index=realized_returns.index)
                        full_mask_local = ~realized_returns.isna() & (realized_returns.abs() >= econ_floor_local)
                        y_full_local[full_mask_local & (realized_returns > 0)] = 1.0
                        y_full_local[full_mask_local & (realized_returns <= 0)] = 0.0

                        filtering_diag_local = compute_filtering_inflation_diagnostics(
                            X=meta_features_model_processed,
                            y_full=y_full_local,
                            y_filtered=binary_labels,
                            realized_returns=realized_returns,
                            volatility=volatility_1d,
                            probabilities=calibrated_probs,
                            cv_splits=3,
                            time_aware_cv=True,
                        )

                        easy_problem_flag = False
                        try:
                            overlap_diag_local = compute_class_overlap_features(
                                X=meta_features_model_processed,
                                retained_mask=labeled_mask,
                                top_k_features=5,
                            )
                            easy_problem_flag = bool(overlap_diag_local.get("easy_problem_detected", False))
                        except Exception:
                            overlap_diag_local = {}
                            easy_problem_flag = False

                        if (
                            filtering_diag_local.get("filtering_is_major_contributor")
                            or filtering_diag_local.get("precision_collapse_detected")
                            or easy_problem_flag
                        ):
                            diagnostics_penalty = float(config.get("diagnostic_penalty_weight", 150.0))
                    except Exception:
                        diagnostics_penalty = 0.0

                # ===== COMBINED OBJECTIVE WITH RETENTION REGULARIZATION =====
                # New formulation: objective adjusts the filtered AUC based on how far
                # the retention rate falls below a soft target.
                #
                #   ret_target = 0.35  (35% of pre-events retained)
                #   penalty    = alpha * (max(0, ret_target - retention_total))^2
                #   objective  = AUC_filtered -  penalty
                #
                # This penalizes overly aggressive filtering (very low retention) while
                # not rewarding sparse configs purely via 1/retention.

                # Retention regularization parameter (configurable)

                # Calibration-aware risk adjustment on top of retention-adjusted AUC.
                weighted_brier = None
                weighted_brier_norm = None
                ece_norm = None
                mid_brier = None
                mid_brier_norm = None
                calib_combo_value = None
                rank_calib_score = auc_retention_adjusted

                try:
                    probs_array = np.asarray(calibrated_probs, dtype=float)
                    y_calib_vals = np.asarray(binary_labels.values, dtype=float)
                    mask_calib = np.isfinite(probs_array) & np.isfinite(y_calib_vals)
                    probs_array = probs_array[mask_calib]
                    y_calib_vals = y_calib_vals[mask_calib]
                    n_calib = y_calib_vals.size

                    min_calib_samples = int(config.get("calibration_min_samples", 200))
                    if n_calib >= min_calib_samples:
                        p_min = float(config.get("calibration_prob_min_clip", 0.05))
                        p_max = float(config.get("calibration_prob_max_clip", 0.95))
                        if p_max <= p_min:
                            p_max = min(0.99, p_min + 0.05)
                        probs_clipped = np.clip(probs_array, p_min, p_max)

                        conf_power = float(config.get("calibration_confidence_power", 1.0))
                        if conf_power != 1.0:
                            confidence = np.abs(probs_clipped - 0.5) ** conf_power
                            weights = 1.0 + confidence
                        else:
                            weights = np.ones_like(probs_clipped)

                        errors = (y_calib_vals - probs_clipped) ** 2
                        weighted_brier = float(np.average(errors, weights=weights))

                        brier_target = float(config.get("calibration_brier_target", 0.18))
                        brier_max = float(config.get("calibration_brier_max", 0.35))
                        if brier_max <= brier_target:
                            brier_max = brier_target + 1e-3
                        excess = max(0.0, weighted_brier - brier_target)
                        denom = max(brier_max - brier_target, 1e-6)
                        weighted_brier_norm = min(1.0, excess / denom)

                        n_bins_calib = int(config.get("calibration_bins", 10))
                        if n_bins_calib <= 0:
                            n_bins_calib = 10
                        n_bins_calib = max(2, n_bins_calib)
                        bin_edges = np.linspace(0.0, 1.0, n_bins_calib + 1)
                        ece_val = 0.0
                        for bi in range(n_bins_calib):
                            mask_bin = (probs_clipped >= bin_edges[bi]) & (probs_clipped < bin_edges[bi + 1])
                            idx_bin = np.nonzero(mask_bin)[0]
                            n_bin = idx_bin.size
                            if n_bin < 20:
                                continue
                            p_hat = float(np.mean(probs_clipped[idx_bin]))
                            y_hat = float(np.mean(y_calib_vals[idx_bin]))
                            ece_val += (n_bin / float(n_calib)) * abs(p_hat - y_hat)
                        if ece_val > 0.0:
                            ece_raw = float(ece_val)
                            ece_max = float(config.get("calibration_ece_max", 0.25))
                            if ece_max <= 0.0:
                                ece_max = 0.25
                            ece_norm = min(1.0, ece_raw / ece_max)

                        mid_low = float(config.get("calibration_mid_prob_low", 0.3))
                        mid_high = float(config.get("calibration_mid_prob_high", 0.7))
                        if mid_high <= mid_low:
                            mid_high = mid_low + 1e-3
                        mid_mask = (probs_clipped >= mid_low) & (probs_clipped <= mid_high)
                        if np.any(mid_mask):
                            errors_mid = (y_calib_vals[mid_mask] - probs_clipped[mid_mask]) ** 2
                            if errors_mid.size > 0:
                                mid_brier = float(np.mean(errors_mid))
                                mid_target = float(config.get("calibration_mid_brier_target", brier_target))
                                mid_max = float(config.get("calibration_mid_brier_max", brier_max))
                                if mid_max <= mid_target:
                                    mid_max = mid_target + 1e-3
                                mid_excess = max(0.0, mid_brier - mid_target)
                                mid_denom = max(mid_max - mid_target, 1e-6)
                                mid_brier_norm = min(1.0, mid_excess / mid_denom)

                        # New logic: Discrete tiers for calibration impact
                        calib_score = weighted_brier if weighted_brier is not None else 0.25

                        calib_factor = 1.0
                        if calib_score > 0.25:
                            # "Minimum requirement" failed: harsh linear penalty
                            # 0.25 -> 1.0x, 0.35 -> 0.0x
                            penalty = (calib_score - 0.25) / 0.10
                            calib_factor = max(0.0, 1.0 - penalty * 10.0) # Very steep drop
                        elif calib_score <= 0.18:
                            # "Excellent" bonus
                            calib_factor = 1.1

                        calib_combo_value = float(calib_score)

                        factor = calib_factor

                        deflation = 0.0
                        if auc_cv_std is not None and np.isfinite(auc_cv_std):
                            try:
                                deflation = 2.0 * float(auc_cv_std)
                            except Exception:
                                deflation = 0.0
                        deflated_auc = max(0.0, float(auc_retention_adjusted) - deflation)
                        rank_calib_score = deflated_auc * factor

                        # ===== P&L CALIBRATION-CURVE SANITY TERM =====
                        # If available from diagnostics, use the P&L calibration curve
                        # (probability → expected net return) to detect configurations
                        # where no probability region is economically positive or where
                        # high-probability regions underperform.
                        try:
                            pnl_curve = None
                            if isinstance(best_config_diagnostics.get("calibration_diagnostics"), dict):
                                pnl_curve = best_config_diagnostics["calibration_diagnostics"].get("pnl_calibration_curve")

                            if isinstance(pnl_curve, dict):
                                prob_grid = np.asarray(pnl_curve.get("prob_grid", []), dtype=float)
                                exp_net = np.asarray(pnl_curve.get("expected_net_return", []), dtype=float)
                                if prob_grid.size == exp_net.size and prob_grid.size >= 3:
                                    # 1) Check if any region has positive expected net return.
                                    if not np.any(exp_net > 0.0):
                                        # Strong penalty: no economically positive region.
                                        rank_calib_score *= 0.7

                                    # 2) Check monotonicity in upper half of probability grid.
                                    mid_idx = prob_grid.size // 2
                                    high_probs = prob_grid[mid_idx:]
                                    high_exp = exp_net[mid_idx:]
                                    if high_probs.size >= 3:
                                        violations = 0
                                        for i in range(high_exp.size - 1):
                                            if high_exp[i + 1] < high_exp[i] - 1e-8:
                                                violations += 1
                                        if violations > 0:
                                            # Soft penalty proportional to fraction of violations.
                                            frac_viols = violations / max(high_exp.size - 1, 1)
                                            rank_calib_score *= (1.0 - 0.3 * min(1.0, frac_viols))
                        except Exception:
                            # If anything fails in P&L calibration sanity checks, do not alter rank_calib_score.
                            pass
                except Exception:
                    weighted_brier = None
                    weighted_brier_norm = None
                    ece_norm = None
                    mid_brier = None
                    mid_brier_norm = None
                    calib_combo_value = None
                    rank_calib_score = auc_retention_adjusted

                # ===== SIMPLIFIED COMBINED OBJECTIVE =====
                # Primary components:
                # 1. edge_scaled: Captures profitability AND learnability (via capture ratio)
                # 2. auc_retention_adjusted: AUC adjusted for retention penalty & range
                # 3. trades_bonus: Reward healthy trade density
                # 4. Penalties for extreme density, slow exits, over-aggressive filtering,
                #    and unrealistic TPSL geometry relative to ATR-based trailing.
                # 5. NEW: sample_count_bonus for Phase 1 (prioritize configs with enough samples)
                # 6. NEW: regime_coverage_penalty for stratified regime sampling

                # ===== TWO-PHASE HPO: SAMPLE COUNT BONUS =====
                # In Phase 1, heavily reward configurations that achieve minimum sample counts
                sample_count_bonus = 0.0
                if n_events >= MIN_EVENTS_PHASE2:
                    sample_count_bonus = 100.0  # Strong bonus for achieving Phase 2 threshold
                elif n_events >= MIN_EVENTS_PHASE1:
                    sample_count_bonus = 50.0  # Moderate bonus for achieving Phase 1 threshold
                elif n_events >= 100:
                    sample_count_bonus = 20.0  # Small bonus for reasonable sample count

                # ===== SIGNAL FIDELITY METRICS (NEW OBJECTIVE) =====
                # 1. Information Coefficient (IC) - Weighted Rank Correlation
                ic = 0.0
                try:
                    if calibrated_probs is not None and len(calibrated_probs) > 50:
                         # Align masks
                         valid_mask_ic = np.isfinite(calibrated_probs) & np.isfinite(realized_return_clean)
                         if valid_mask_ic.sum() > 50:
                             p_clean = calibrated_probs[valid_mask_ic]
                             r_clean = realized_return_clean[valid_mask_ic]

                             # Calculate confidence-based weights for IC
                             # Weight = 1.0 + |prob - 0.5| (Higher weight for confident predictions)
                             ic_weights = 1.0 + np.abs(p_clean - 0.5)

                             # Compute Ranks
                             from scipy.stats import rankdata
                             p_ranks = rankdata(p_clean)
                             r_ranks = rankdata(r_clean)

                             # Weighted Pearson on Ranks (Weighted Spearman)
                             # np.cov(..., aweights=...) returns covariance matrix
                             cov_mat = np.cov(p_ranks, r_ranks, aweights=ic_weights)
                             if cov_mat.shape == (2, 2):
                                 cov_xy = cov_mat[0, 1]
                                 var_x = cov_mat[0, 0]
                                 var_y = cov_mat[1, 1]
                                 if var_x > 0 and var_y > 0:
                                     val_ic = cov_xy / np.sqrt(var_x * var_y)
                                     if np.isfinite(val_ic):
                                         ic = float(val_ic)
                except Exception:
                    ic = 0.0

                # 2. Density Score (Logarithmic)
                density_score = 0.0
                if events_per_day > 0:
                    density_score = min(1.0, np.log10(events_per_day + 1.0) / 1.0)

                # 3. Effect Size (Cohen's d) - Reusing d_post
                # d_post is already calculated above
                snr_metric = max(0.0, float(d_post)) if np.isfinite(d_post) else 0.0

                # ===== CALIBRATION SCORE =====
                # Derived from weighted_brier_norm (0=perfect, 1=bad)
                # If unavailable, assume neutral/poor (0.0 score) to encourage calibration availability
                score_calibration = 0.0
                if weighted_brier_norm is not None:
                    score_calibration = max(0.0, 1.0 - float(weighted_brier_norm))
                elif ece_norm is not None:
                     score_calibration = max(0.0, 1.0 - float(ece_norm))

                # ===== NEW COMBINED SCORE FORMULA =====
                # Weights: AUC(30%), IC(20%), Calib(20%), SNR(15%), Density(15%)

                # User Request: Modulate AUC by stability (std dev)
                # If auc_cv_std is high (unstable), discount the AUC contribution.
                # Threshold: 0.10 (if std > 0.10, AUC score becomes 0)
                stability_factor = 1.0
                if auc_cv_std is not None and np.isfinite(auc_cv_std):
                    stability_factor = max(0.0, 1.0 - (float(auc_cv_std) / 0.10))

                score_auc = max(0.0, (mean_auc - 0.50) * 2.0) * stability_factor
                # User Request (Refined): Penalize negative IC and clip upside
                score_ic = np.clip(ic * 5.0, -1.0, 1.0)
                score_snr = min(1.0, snr_metric / 2.0)
                score_density = density_score

                # Base score [0, 1] mapped to [0, 100] scale
                fidelity_score = (
                    0.30 * score_auc +
                    0.20 * score_ic +
                    0.20 * score_calibration +
                    0.15 * score_snr +
                    0.15 * score_density
                ) * 100.0

                combined_score = (
                    fidelity_score
                    + sample_count_bonus
                    + (edge_scaled * edge_weight * 0.5) # Reduced edge weight
                    + balance_term
                    - (penalty_density * 0.1)
                    - (tto_penalty * 0.1)
                    - (retention_asym_weight * retention_asym_penalty)
                    - trail_geom_penalty
                    - diagnostics_penalty
                    - robustness_penalty
                )

                try:
                    label_cfg = build_label_config(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        params=params,
                        extra=None,
                    )
                    config_id = compute_label_config_id(label_cfg)
                except Exception:
                    config_id = None

                # ===== DETAILED P&L STATS (NEW) =====
                pnl_win_rate = 0.0
                pnl_avg_win = 0.0
                pnl_avg_loss = 0.0
                pnl_profit_factor = 0.0
                pnl_total_return = 0.0

                try:
                    # 'labeled_mask' is defined earlier as ~binary_labels.isna()
                    if 'labeled_mask' in locals() and isinstance(realized_returns, pd.Series):
                        valid_ret = realized_returns[labeled_mask]
                        if len(valid_ret) > 0:
                            pnl_total_return = float(valid_ret.sum())
                            wins = valid_ret[valid_ret > 0]
                            losses = valid_ret[valid_ret < 0]

                            pnl_win_rate = float(len(wins) / len(valid_ret))

                            if len(wins) > 0:
                                pnl_avg_win = float(wins.mean())

                            if len(losses) > 0:
                                pnl_avg_loss = float(losses.mean())

                            # NEW: Average return per trade (wins & losses)
                            pnl_avg_ret = float(valid_ret.mean())

                            loss_sum = abs(float(losses.sum()))
                            if loss_sum > 1e-9:
                                pnl_profit_factor = float(wins.sum()) / loss_sum
                            elif len(wins) > 0:
                                pnl_profit_factor = 100.0 # Capped infinite profit factor
                except Exception:
                    pass

                # Store candidate configuration for later persistence
                candidate_config = {
                    'config_id': config_id,
                    'params': params.copy(),
                    'learnability': float(learnability_score),
                    'mean_auc': float(mean_auc),
                    'profitability': float(profitability_score),
                    'edge': float(edge_score),
                    'edge_scaled': float(edge_scaled),
                    'combined': float(combined_score),
                    'fidelity_score': float(fidelity_score), # NEW
                    'ic': float(ic), # NEW
                    'calibration_score': float(score_calibration), # NEW
                    'mean_pos': float(mean_pos),
                    'mean_neg': float(mean_neg),
                    'sharpe_pos': float(sharpe_pos),
                    'n_events': int(n_events),
                    'n_raw_events': int(n_raw_events),
                    'n_vol_scaled_events': int(n_vol_scaled_events),
                    'balance_score': float(balance_score),
                    'trades_per_day': float(trades_per_day),
                    'mean_tto': float(mean_tto) if np.isfinite(mean_tto) else float('nan'),
                    'timeout_rate': float(timeout_rate) if np.isfinite(timeout_rate) else float('nan'),
                    'tto_penalty': float(tto_penalty),
                    'n_pre_events': int(n_pre_total),
                    'retention_total': float(retention_total),
                    'retention_pos': float(retention_pos),
                    'retention_neg': float(retention_neg),
                    'snr_pre': float(snr_pre),
                    'snr_post': float(snr_post),
                    'effect_size_pre': float(d_pre) if np.isfinite(d_pre) else 0.0,
                    'effect_size_post': float(d_post) if np.isfinite(d_post) else 0.0,
                    'n_required_80pct_power': float(n_required_80),
                    # CV-based robustness proxy from learnability probe
                    'learnability_worst_fold_auc': float(worst_fold_auc_cv) if worst_fold_auc_cv is not None else None,
                    'learnability_auc_cv_std': float(auc_cv_std) if auc_cv_std is not None else None,
                    'model_complexity': model_complexity,
                    # NEW: Track trade count control parameters
                    'cusum_threshold': float(cusum_threshold),
                    'target_signal_density': float(target_signal_density),
                    'r_multiple_threshold': float(r_multiple_threshold),
                    'effective_tx_cost': float(effective_tx_cost),
                    'label_low_q': float(label_low_q),
                    # NEW: Detailed P&L Stats
                    'pnl_win_rate': float(pnl_win_rate),
                    'pnl_avg_win': float(pnl_avg_win),
                    'pnl_avg_loss': float(pnl_avg_loss),
                    'pnl_profit_factor': float(pnl_profit_factor),
                    'pnl_profit_factor': float(pnl_profit_factor),
                    'pnl_total_return': float(pnl_total_return),
                    'pnl_avg_ret_per_trade': float(pnl_avg_ret) if 'pnl_avg_ret' in locals() else 0.0,
                    'pnl_per_day': float(pnl_total_return) / max(days_span, 1.0) if 'days_span' in locals() else 0.0,
                    'label_high_q': float(label_high_q),
                    # AUC range tracking (unified into auc_retention_adjusted)
                    'auc_in_target_range': bool(auc_in_target_range),
                    'auc_range_adjustment': float(auc_range_adjustment),
                    'auc_weight_multiplier': float(auc_weight_multiplier),
                    # NEW: CV fold diagnostics from learnability probe
                    'fold_aucs': fold_aucs.tolist() if isinstance(fold_aucs, np.ndarray) else list(fold_aucs) if fold_aucs is not None else [],
                    'auc_interpretation': (
                        'too_noisy' if mean_auc < 0.54 else
                        'good_edge' if mean_auc <= 0.62 else
                        'excellent_check_leakage' if mean_auc <= 0.67 else
                        'suspicious_leakage' if mean_auc > 0.70 else
                        'above_target'
                    ),
                    # Retention regularization (unified objective)
                    'retention_penalty': float(retention_penalty),
                    'retention_asym_penalty': float(retention_asym_penalty),
                    'auc_retention_adjusted': float(auc_retention_adjusted),
                    'alpha_retention': float(alpha_retention),
                    'weighted_brier': float(weighted_brier) if weighted_brier is not None else None,
                    'weighted_brier_norm': float(weighted_brier_norm) if weighted_brier_norm is not None else None,
                    'ece_norm': float(ece_norm) if ece_norm is not None else None,
                    'mid_brier': float(mid_brier) if mid_brier is not None else None,
                    'mid_brier_norm': float(mid_brier_norm) if mid_brier_norm is not None else None,
                    'calibration_combo': float(calib_combo_value) if calib_combo_value is not None else None,
                    'rank_calib_score': float(rank_calib_score),
                    # NEW: Two-phase HPO and diagnostic gates
                    'sample_count_bonus': float(sample_count_bonus),
                    'mi_value': float(mi_value),
                    'mi_penalty': float(mi_penalty),
                    'prob_std': float(prob_std),
                    'prob_variance_penalty': float(prob_variance_penalty),
                    'single_bin_penalty': float(single_bin_penalty),
                }

                # Optional per-regime breakdown using attached HMM regimes, if available.
                per_regime_metrics: Dict[str, Any] = {}
                regime_coverage_penalty = 0.0  # NEW: Penalty for poor regime coverage
                n_regimes_with_events = 0
                try:
                    if "hmm_regime_label_1h" in market_data.columns:
                        regimes_all = market_data["hmm_regime_label_1h"]
                        regimes_events = regimes_all[labeled_mask]

                        # Align calibrated probabilities with labeled events
                        try:
                            probs_series_full = pd.Series(calibrated_probs, index=binary_labels.index)
                            probs_events = probs_series_full[labeled_mask]
                        except Exception:
                            probs_events = None

                        from sklearn.metrics import roc_auc_score as _roc_auc_score_reg

                        unique_regs = pd.unique(regimes_events.dropna())

                        # ===== NEW: STRATIFIED REGIME SAMPLING CHECK =====
                        # Penalize configurations that have poor regime coverage
                        min_events_per_regime = int(config.get("min_events_per_regime", 20))
                        expected_n_regimes = int(config.get("expected_n_regimes", 5))

                        for reg_val in unique_regs:
                            reg_mask = regimes_events == reg_val
                            n_reg = int(reg_mask.sum())
                            if n_reg >= min_events_per_regime:
                                n_regimes_with_events += 1

                        # Penalty for missing regimes
                        if n_regimes_with_events < expected_n_regimes:
                            missing_regimes = expected_n_regimes - n_regimes_with_events
                            regime_coverage_penalty = missing_regimes * 50.0  # 50 points per missing regime
                            if debug_sample_count < debug_sample_limit:
                                tprint_warning(
                                    f"[REGIME_COVERAGE] Only {n_regimes_with_events}/{expected_n_regimes} regimes "
                                    f"have >= {min_events_per_regime} events. Penalty={regime_coverage_penalty:.1f}"
                                )
                        for reg_val in unique_regs:
                            try:
                                reg_mask = regimes_events == reg_val
                                n_reg = int(reg_mask.sum())
                                if n_reg < 20:
                                    continue

                                returns_reg = returns_labeled[reg_mask]
                                labels_reg = labels_labeled[reg_mask]

                                r_pos_reg = returns_reg[labels_reg == 1]
                                r_neg_reg = returns_reg[labels_reg == 0]

                                mean_pos_reg = float(r_pos_reg.mean()) if len(r_pos_reg) > 0 else 0.0
                                mean_neg_reg = float(r_neg_reg.mean()) if len(r_neg_reg) > 0 else 0.0

                                # Realized AUC within this regime (diagnostic only)
                                auc_reg_local = float("nan")
                                if probs_events is not None:
                                    try:
                                        probs_reg = probs_events[reg_mask]
                                        if len(labels_reg.unique()) >= 2:
                                            auc_reg_local = float(_roc_auc_score_reg(labels_reg, probs_reg))
                                    except Exception:
                                        auc_reg_local = float("nan")

                                # For edge, use the same cross-validated mean_auc and
                                # Sharpe-like trade-count scaling as the global metric,
                                # so that regime-level edges are directly comparable to
                                # the reported best_edge.
                                auc_for_edge = float(mean_auc)

                                edge_reg = compute_realistic_pnl_edge(
                                    mean_return_positive=mean_pos_reg,
                                    mean_auc=auc_for_edge,
                                    transaction_cost=tx,
                                    n_trades=n_reg,
                                    reference_trades=reference_trades,
                                )

                                trades_per_day_reg = float(n_reg) / max(days_span, 1)

                                per_regime_metrics[str(reg_val)] = {
                                    'n_events': n_reg,
                                    'trades_per_day': trades_per_day_reg,
                                    'mean_pos': mean_pos_reg,
                                    'mean_neg': mean_neg_reg,
                                    # Expose the aligned AUC used for edge, and keep the
                                    # local realized AUC as an auxiliary diagnostic field.
                                    'auc': auc_for_edge,
                                    'auc_local': auc_reg_local,
                                    'edge': edge_reg,
                                }
                            except Exception:
                                continue
                except Exception:
                    per_regime_metrics = {}

                # Attach per-regime metrics and optional regression diagnostics
                if per_regime_metrics:
                    candidate_config['per_regime_metrics'] = per_regime_metrics
                    candidate_config['n_regimes_with_events'] = n_regimes_with_events
                    candidate_config['regime_coverage_penalty'] = float(regime_coverage_penalty)
                if reg_target_stats is not None:
                    candidate_config['regression_target_stats'] = reg_target_stats

                # Apply regime coverage penalty to combined score (after per-regime metrics computed)
                combined_score -= regime_coverage_penalty
                candidate_config['combined'] = float(combined_score)

                # Append to candidate pool and log a concise summary for debugging
                candidate_pool.append(candidate_config)
                try:
                    pool_size_after = len(candidate_pool)
                    tprint_info(
                        f"[CANDIDATE_APPEND] complexity={model_complexity} "
                        f"edge={edge_score:.6f} combined={combined_score:.6f} "
                        f"mean_auc={mean_auc:.6f} n_events={n_events} "
                        f"pool_size={pool_size_after}"
                    )
                except Exception:
                    # Logging must not interfere with HPO; ignore any formatting errors
                    pass

                return {
                    'learnability': float(learnability_score),
                    'profitability': float(profitability_score),
                    'edge': float(edge_score),
                    'combined': float(combined_score),
                }

            except Exception as exc:  # Defensive: never crash HPO on one config
                tprint_warning(f"[EARLY_EXIT_EXCEPTION] Labeling objective failed: {exc}")
                gate_stats["exception"] = gate_stats.get("exception", 0) + 1
                gate_stats["last_exception"] = str(exc)
                import traceback
                traceback.print_exc()
                return {'learnability': 0.0, 'profitability': -1e9, 'edge': -1e9, 'combined': -1e9}

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
        # Stage 3: All parameters (profit_thr_base, stop_to_profit_ratio,
        #          iso_min_prob, target_transform refinements, etc.)
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
                'profit_thr_base', 'stop_to_profit_ratio', 'trail_distance',
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
                'profit_thr_base', 'stop_to_profit_ratio', 'trail_distance',
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

        try:
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

        metrics: Dict[str, Any] = {
            "best_score": best_score,
            "best_edge": best_edge,
            "best_params": best_params,
            "best_params_json": str(json_path) if json_path is not None else None,
            "round_metrics_csv": str(csv_path) if csv_path is not None else None,
            "recommended_diagnostics_path": diagnostics_path,
            "total_trials": all_trials_count,
            "stage_results": stage_results,
            "pareto_frontier_size": len(pareto_front),
            "candidate_pool_size": len(candidate_pool),
        }

        # Add diagnostics summary to metrics
        if best_config_diagnostics:
            metrics["best_config_diagnostics"] = best_config_diagnostics

        artifacts: Dict[str, Any] = {}
        if json_path is not None:
            artifacts["best_params_json"] = str(json_path)
        if csv_path is not None:
            artifacts["round_metrics_csv"] = str(csv_path)
        if diagnostics_path is not None:
            artifacts["recommended_diagnostics_path"] = diagnostics_path

        return {
            "success": True,
            "metrics": metrics,
            "artifacts": artifacts,
        }


def register_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry

    step_registry.register("meta_labeling_hpo_sample_weighted", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb_weighted", MetaLabelingHPOSampleWeightedStep)
    tprint("✅ Meta-labeling HPO sample weighted step registered (aliases: meta_labeling_hpo_sample_weighted, sr_labeling_xgb_weighted)", "SUCCESS")


# Auto-register when module is imported
register_meta_labeling_hpo_sample_weighted_step()