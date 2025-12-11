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
# FEATURE QUALITY & SELECTION FUNCTIONS
# ============================================================================

def calculate_feature_quality(series: np.ndarray) -> float:
    """
    Unsupervised Signal-to-Noise Ratio for feature quality assessment.
    
    Higher Score = Better Feature (high signal, low noise).
    
    Quality = Signal Power (Variance) / Noise Power (Smoothness Error)
    
    Logic:
    - We want features that move a lot (High Variance)
    - But represent clean trends (Low Wiggle/Smoothness Error)
    
    Args:
        series: Feature values as numpy array
    
    Returns:
        Quality score (higher = better). Returns 0.0 for flat/useless features.
    """
    # Handle NaN/Inf
    clean_series = series[np.isfinite(series)]
    if len(clean_series) < 10:
        return 0.0
    
    # 1. Signal Power (Standard Deviation)
    signal_power = np.std(clean_series)
    
    if signal_power < 1e-12:
        return 0.0  # Flatline feature, useless
    
    # 2. Noise Estimate (Mean Squared Second Difference)
    # "How jagged is the curve?"
    second_diff = np.diff(clean_series, n=2)
    noise_power = np.mean(second_diff**2)
    
    # Avoid division by zero
    if noise_power < 1e-12:
        # Very smooth feature - high quality
        return signal_power * 1e6  # Large but finite
    
    # Quality = Signal / Noise
    return float(signal_power / noise_power)


def calculate_all_feature_qualities(df_features: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Signal-to-Noise quality scores for all feature columns.
    
    This is a lightweight, unsupervised operation that runs in milliseconds.
    
    Args:
        df_features: DataFrame with feature columns
    
    Returns:
        Dict mapping column name to quality score
    """
    quality_map = {}
    for col in df_features.columns:
        try:
            quality_map[col] = calculate_feature_quality(df_features[col].values)
        except Exception:
            quality_map[col] = 0.0
    return quality_map


def reduce_features_by_correlation(
    df_features: pd.DataFrame,
    quality_scores: Dict[str, float],
    target_n: int = 70,
    correlation_threshold: float = 0.85,
    min_quality_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Reduce features by removing correlated ones, keeping higher quality features.
    
    Algorithm:
    1. Remove features below minimum quality threshold
    2. Sort features by quality (descending)
    3. Iteratively add features if not highly correlated with already selected
    4. Use quality score as tie-breaker when removing correlated features
    
    Args:
        df_features: DataFrame with all features
        quality_scores: Dict mapping column name to quality score
        target_n: Target number of features to keep
        correlation_threshold: Max allowed correlation between features
        min_quality_threshold: Minimum quality score to consider
    
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
        # Second pass with relaxed threshold
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
        f"   Feature reduction: {len(df_features.columns)} → {len(selected_features)} "
        f"(target={target_n}, corr_threshold={correlation_threshold})"
    )
    
    return df_features[selected_features]


def generate_multi_horizon_features(
    base_features: pd.DataFrame,
    horizons: Dict[str, int] = None,
) -> pd.DataFrame:
    """
    Generate multi-horizon versions of features (short, medium, long).
    
    For each feature, creates smoothed versions at different lookback windows
    to capture different time scales of the same signal.
    
    Args:
        base_features: DataFrame with base feature columns
        horizons: Dict mapping horizon name to lookback bars
                  Default: {"Short": 5, "Medium": 20, "Long": 60}
    
    Returns:
        DataFrame with multi-horizon features added
    """
    if horizons is None:
        horizons = {
            "Short": 5,    # ~1.25 hours at 15m
            "Medium": 20,  # ~5 hours at 15m
            "Long": 60,    # ~15 hours at 15m
        }
    
    result = base_features.copy()
    
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


def select_features_with_quality(
    df_features: pd.DataFrame,
    target_n: int = 70,
    correlation_threshold: float = 0.85,
    generate_horizons: bool = True,
    horizon_config: Dict[str, int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Complete feature selection pipeline with quality scoring.
    
    Pipeline:
    1. (Optional) Generate multi-horizon versions of features
    2. Calculate quality scores for all features
    3. Reduce features by correlation, using quality as tie-breaker
    
    This runs AFTER Layer 0 (Kalman Q/R optimization) but BEFORE the main HPO loop.
    
    Args:
        df_features: DataFrame with raw features
        target_n: Target number of features to select
        correlation_threshold: Max correlation between selected features
        generate_horizons: Whether to create multi-horizon versions
        horizon_config: Custom horizon configuration
    
    Returns:
        Tuple of (reduced_features_df, quality_scores_dict)
    """
    tprint_info("🔍 Starting quality-based feature selection...")
    
    # 1. Generate multi-horizon features (if enabled)
    if generate_horizons:
        tprint_info(f"   Generating multi-horizon features...")
        initial_cols = len(df_features.columns)
        df_expanded = generate_multi_horizon_features(df_features, horizon_config)
        tprint_info(f"   Expanded: {initial_cols} → {len(df_expanded.columns)} features")
    else:
        df_expanded = df_features.copy()
    
    # 2. Calculate quality scores for all features
    tprint_info("   Calculating feature quality scores (Signal/Noise ratio)...")
    quality_scores = calculate_all_feature_qualities(df_expanded)
    
    # Log top/bottom quality features for debugging
    sorted_by_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_by_quality[:5]
    bottom_5 = sorted_by_quality[-5:]
    
    tprint_info(f"   Top 5 quality: {[(n, f'{q:.2f}') for n, q in top_5]}")
    tprint_info(f"   Bottom 5 quality: {[(n, f'{q:.2f}') for n, q in bottom_5]}")
    
    # 3. Reduce by correlation with quality tie-breaker
    tprint_info(f"   Reducing to {target_n} features by correlation...")
    df_reduced = reduce_features_by_correlation(
        df_features=df_expanded,
        quality_scores=quality_scores,
        target_n=target_n,
        correlation_threshold=correlation_threshold,
    )
    
    # Return only quality scores for selected features
    selected_quality = {col: quality_scores[col] for col in df_reduced.columns}
    
    tprint_success(f"✅ Feature selection complete: {len(df_reduced.columns)} features selected")
    
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
        Score in [0.01, 1.0]
    """
    s_min, s_max = sweet_spot
    
    if x < lower or x > upper:
        return 0.01  # Soft floor to avoid log(0) errors if used later
    elif s_min <= x <= s_max:
        return 1.0
    elif lower <= x < s_min:
        # Ramp up
        return (x - lower) / (s_min - lower)
    elif s_max < x <= upper:
        # Ramp down
        return (upper - x) / (upper - s_max)
    return 0.01


def calculate_hpo_utility(
    folds_sharpe: np.ndarray,
    auc: float,
    trades_per_day: float,
    lambda_vol: float = 1.2,
    w_auc: float = 1.0,
    w_den: float = 0.5,
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
    # 1. Stability-adjusted Sharpe (base)
    avg_sharpe = float(np.mean(folds_sharpe))
    # Safety check for single-fold runs
    vol_sharpe = float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe) > 1 else 0.0
    
    # Penalize volatility across folds
    base_score = avg_sharpe - (lambda_vol * vol_sharpe)

    # Hard early reject for non-positive structural performance
    if base_score <= 0.0:
        return -1.0

    # Bound the base score (diminishing returns on super-high Sharpes)
    base_norm = np.log1p(base_score)

    # 2. Auxiliary modifiers (0..1)
    
    # AUC Gate: Returns 0.01 to 1.0
    # Strict penalty for leakage (AUC > 0.70) or randomness (AUC < 0.54)
    phi_auc = trapezoidal_gate(auc, lower=0.54, sweet_spot=(0.58, 0.64), upper=0.70)

    # Density: Adjusted Sigmoid
    # Shifted center to 1.0 so that at 2.0 trades/day, score is ~0.73
    # At 3.0 trades/day, score is ~0.88. At 5.0, score is ~0.98.
    phi_density = 1.0 / (1.0 + np.exp(-(trades_per_day - 1.0)))

    # 3. Weighted Geometric Combination
    # If phi_auc is near 0 (leakage/random), the whole score collapses
    modifier = (phi_auc ** w_auc) * (phi_density ** w_den)

    utility = float(np.clip(base_norm * modifier, -1.0, 10.0))
    return utility


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
        self.logger.info(f"Starting Multi-Stage HPO for {symbol} {timeframe} {direction}")
        config = config or {}

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
                        "low": 10.0,
                        "high": 40.0,
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
        # STAGE 0: SIGNAL/FEATURE HPO (KALMAN TUNING)
        # ------------------------------------------------------------------
        tprint_info("🧪 Stage 0: Optimizing Kalman Signal Parameters...")

        kalman_search_space = {
            "kalman_Q": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "kalman_R": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
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
                # Enforce profit >= 2x stop constraint. During stages that do not
                # actively optimize profit_thr_base/stop_to_profit_ratio, fall back
                # to conservative defaults.
                profit_thr_base = float(params.get("profit_thr_base", 0.012))
                stop_ratio = float(params.get("stop_to_profit_ratio", 0.5))
                trail_dist = float(params.get("trail_distance", 0.0))

                # CONSTRAINT: Ensure profit is at least 2x stop (RR >= 2:1)
                stop_thr_base = max(0.0005, profit_thr_base * stop_ratio)
                if profit_thr_base < 2.0 * stop_thr_base:
                    # Early exit: invalid RR geometry
                    tprint_warning(
                        f"[EARLY_EXIT_RR] Config rejected: profit {profit_thr_base:.4f} < 2x stop {stop_thr_base:.4f}"
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
                # require a minimum RR ~1.5 (after adaptive multipliers, ensures net positive expectancy).
                worst_rr = (profit_thr_base * profit_mult_min) / max(stop_thr_base * stop_mult_max, 1e-8)
                if worst_rr < 1.5:
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
        kalman_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(n_trials=30, execution_mode="full", direction="maximize", seed=42)
        )
        kalman_result = kalman_optimizer.optimize(objective=kalman_objective, search_space=kalman_search_space)
        best_kalman_params = kalman_result.get("best_params", {})
        
        # Log the results with loss details
        best_Q = best_kalman_params.get('kalman_Q', 1e-4)
        best_R = best_kalman_params.get('kalman_R', 0.01)
        
        # Compute final loss details for logging
        try:
            final_smoothed, _ = rts_smoother_1d(close_series.values, Q=best_Q, R=best_R)
            final_loss, final_details = robust_labeling_loss(final_smoothed, close_series.values, is_acausal=True)
            tprint_success(
                f"✅ Stage 0 Complete. Loss: {final_loss:.4f} "
                f"(smooth={final_details['smooth']:.4f}, track={final_details['track']:.4f}, "
                f"amp={final_details['amp']:.4f}, amp_ratio={final_details['amp_ratio']:.3f})"
            )
        except Exception:
            tprint_success(f"✅ Stage 0 Complete. Best Score: {kalman_result.get('best_value', 0):.4f}")
        
        tprint_info(f"   Best RTS/Kalman Params: Q={best_Q:.2e}, R={best_R:.2e}")
        tprint_info("   Note: RTS (acausal) for labels, Kalman (causal) for live features")

        # ------------------------------------------------------------------
        # 1. LAYER 1: WEIGHTING OPTIMIZATION
        # ------------------------------------------------------------------
        tprint_info("🧪 Layer 1: Optimizing Sample Weighting Parameters...")

        # Generate baseline events using defaults (as TBM params are Layer 2)
        baseline_profit = pd.Series(np.maximum(0.008, atr_series * 2.0), index=market_data.index)
        baseline_stop = pd.Series(np.maximum(0.004, atr_series * 1.0), index=market_data.index)

        (
            baseline_returns,
            _, _, _, _, _, _, _
        ) = compute_realized_returns(
            market_data,
            primary_signals,
            profit_threshold=baseline_profit,
            stop_threshold=baseline_stop,
            horizon=12,
            transaction_cost=DEFAULT_TRANSACTION_COST,
            min_event_spacing=2,
        )

        valid_mask = ~baseline_returns.isna()
        baseline_t_events = baseline_returns.index[valid_mask]
        baseline_returns_clean = baseline_returns[valid_mask]

        if len(baseline_t_events) < 50:
            tprint_warning(f"⚠️ Too few baseline events ({len(baseline_t_events)}) for Layer 1. Using defaults.")
            best_weighting_params = {
                'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                'uniq_intensity': 1.0, 'exp_mag': 1.0, 'exp_learn': 1.0,
                'exp_uniq': 1.0, 'exp_cross': 1.0, 'downside_multiplier': 1.0
            }
        else:
            try:
                best_weighting_params = run_layer1_optimization(
                    df=market_data,
                    returns=baseline_returns_clean,
                    t_events=baseline_t_events
                )
            except Exception as e:
                tprint_warning(f"⚠️ Layer 1 optimization failed: {e}. Using defaults.")
                best_weighting_params = {
                    'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                    'uniq_intensity': 1.0, 'exp_mag': 1.0, 'exp_learn': 1.0,
                    'exp_uniq': 1.0, 'exp_cross': 1.0, 'downside_multiplier': 1.0
                }

        tprint_success(f"✅ Layer 1 Complete. Best Weighting Params: {best_weighting_params}")

        # ------------------------------------------------------------------
        # 2. LAYER 2: TRADING PARAMETER OPTIMIZATION
        # ------------------------------------------------------------------
        tprint_info("🧪 Layer 2: Optimizing Trading Parameters...")

        # Updated search space with Constructive Sampling
        layer2_search_space = {
            "sl_atr_mult": {"type": "float", "low": 0.5, "high": 2.5},
            "risk_reward_ratio": {"type": "float", "low": 2.0, "high": 5.0}, # Constrains TP >= 2*SL
            "trail_distance_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
            # Kalman params are fixed from Stage 0
        }

        meta_feature_cfg = config.get("meta_feature_engineering", {})
        volume_available = "volume" in market_data.columns

        # PRE-CALCULATE META-FEATURES ONCE (Performance Optimization)
        # Use baseline returns/labels as proxy. The goal is to get X features.
        # Note: If meta-features rely heavily on exact realized_return of the specific TBM,
        # this is an approximation. But for HPO speed, it is necessary.
        # Most features (technicals, regime, kalman) depend only on market_data/signals.
        tprint_info("🏗️ Layer 2: Pre-calculating meta-features with optimized Kalman params...")
        mf_config_opt = meta_feature_cfg.copy()
        mf_config_opt['kalman_Q'] = best_kalman_params.get('kalman_Q', 1e-4)
        mf_config_opt['kalman_R'] = best_kalman_params.get('kalman_R', 0.01)

        # Generate dummy stop threshold for feature generation (won't affect independent features)
        dummy_stop_thr = np.maximum(0.002, atr_series * 1.0)

        _, meta_features_full, _, _ = build_meta_features_for_model(
            market_data=market_data,
            primary_signals=primary_signals,
            realized_returns=baseline_returns, # Proxy
            binary_labels=pd.Series(np.nan, index=market_data.index), # Proxy
            event_durations=pd.Series(12, index=market_data.index), # Proxy
            mfe_series=None,
            mae_series=None,
            adaptive_stop_threshold=dummy_stop_thr,
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
        
        # Custom horizon configuration (can be overridden in config)
        horizon_config = config.get("feature_horizon_config", {
            "Short": 5,    # ~1.25 hours at 15m (fast signals)
            "Medium": 20,  # ~5 hours at 15m (medium signals)
            "Long": 60,    # ~15 hours at 15m (slow signals)
        })
        
        tprint_info("🔬 Running quality-based feature selection...")
        try:
            meta_features_full, feature_quality_scores = select_features_with_quality(
                df_features=meta_features_full,
                target_n=target_feature_count,
                correlation_threshold=feature_correlation_threshold,
                generate_horizons=enable_multi_horizon,
                horizon_config=horizon_config,
            )
            
            # Store quality scores for potential later use
            self._feature_quality_scores = feature_quality_scores
            
            tprint_success(
                f"✅ Feature selection complete: {len(meta_features_full.columns)} features "
                f"(target={target_feature_count})"
            )
        except Exception as fs_exc:
            tprint_warning(f"⚠️ Feature selection failed: {fs_exc}. Using all features.")
            self._feature_quality_scores = {}

        def layer2_objective(trial_params: Dict[str, Any]) -> float:
            """
            Layer 2 objective using stability-adjusted utility with:
            - Pre-calibrated probabilities for position sizing
            - Per-fold Sharpe stability assessment
            - Trapezoidal AUC gate (penalizes leakage and randomness)
            - Trade density modifier
            """
            # A. TBM SIMULATION (Constructive)
            sl_mult = trial_params["sl_atr_mult"]
            rr = trial_params["risk_reward_ratio"]
            tp_mult = sl_mult * rr  # Guarantee: TP >= 2 * SL

            trail_dist = trial_params.get("trail_distance_atr_mult", 0.0)

            prof_thr = np.maximum(0.008, atr_series * tp_mult)
            stop_thr = np.maximum(0.002, atr_series * sl_mult)

            (
                l2_returns,
                l2_labels,
                _,
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
                atr_series=atr_series  # Pass for trailing logic
            )

            valid_idx = ~l2_labels.isna()
            if valid_idx.sum() < 50:
                return -1.0

            l2_t_events = l2_returns.index[valid_idx]
            l2_returns_clean = l2_returns[valid_idx]
            l2_labels_clean = l2_labels[valid_idx]

            # B. DYNAMIC WEIGHT GENERATION
            batch_consistency = full_consistency.reindex(l2_t_events).fillna(0).values
            batch_volatility = full_volatility.reindex(l2_t_events).fillna(0).values
            batch_uniqueness = compute_uniqueness(l2_t_events, market_data.index)

            sample_weights = generate_weights_per_label(
                returns=l2_returns_clean.values,
                t_events=l2_t_events,
                close_series=None,
                consistency_scores=batch_consistency,
                uniqueness_scores=batch_uniqueness.values,
                vol_proxy=batch_volatility,
                **best_weighting_params
            )

            # C. SUBSET META-FEATURES (Fast)
            X_trial = meta_features_full.loc[valid_idx].fillna(0)

            # D. FAST MODEL TRAINING WITH CV
            n_cv_folds = 5
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                cv_preds = cross_val_predict(
                    fast_model, X_trial, l2_labels_clean, cv=n_cv_folds,
                    method='predict_proba', fit_params={'sample_weight': sample_weights}, n_jobs=-1
                )[:, 1]
            except Exception:
                return -1.0

            # E. COMPUTE AUC (for trapezoidal gate)
            try:
                mean_auc = roc_auc_score(l2_labels_clean.values, cv_preds)
            except Exception:
                mean_auc = 0.5

            # F. COMPUTE FOLD SHARPE RATIOS (with pre-calibration)
            # Pre-calibrate probabilities using isotonic regression
            y_true_arr = l2_labels_clean.values.astype(float)
            y_prob_arr = cv_preds.astype(float)
            returns_arr = l2_returns_clean.values.astype(float)

            try:
                folds_sharpe = compute_fold_sharpe_ratios(
                    y_true=y_true_arr,
                    y_prob=y_prob_arr,
                    returns=returns_arr,
                    n_folds=n_cv_folds,
                    use_calibration=True,  # Pre-calibrate before sizing
                )
            except Exception:
                # Fallback to simple Sharpe if fold computation fails
                simple_sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-9)
                folds_sharpe = np.array([simple_sharpe])

            # G. COMPUTE TRADES PER DAY
            trades_per_day = len(l2_returns_clean) / max(days_span, 1)

            # H. CALCULATE UTILITY (Trapezoidal Gate + Stability)
            utility = calculate_hpo_utility(
                folds_sharpe=folds_sharpe,
                auc=mean_auc,
                trades_per_day=trades_per_day,
                lambda_vol=1.2,   # Penalty for Sharpe volatility across folds
                w_auc=1.0,        # Strict AUC gate
                w_den=0.5,        # Moderate density weight
            )

            return utility

        l2_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(n_trials=40, execution_mode="full", direction="maximize", seed=42)
        )
        l2_result = l2_optimizer.optimize(objective=layer2_objective, search_space=layer2_search_space)
        best_trading_params = l2_result.get("best_params", {})
        best_l2_score = l2_result.get("best_value", 0.0)
        tprint_success(f"✅ Layer 2 Complete. Best Score: {best_l2_score:.4f}")
        tprint_info(f"   Best Trading Params: {best_trading_params}")

        # ------------------------------------------------------------------
        # 3. LAYER 3: MODEL HYPERPARAMETER OPTIMIZATION
        # ------------------------------------------------------------------
        tprint_info("🧪 Layer 3: Optimizing Model Hyperparameters...")

        # Reconstruct optimal TBM setup
        final_sl_mult = best_trading_params["sl_atr_mult"]
        final_rr = best_trading_params["risk_reward_ratio"]
        final_tp_mult = final_sl_mult * final_rr
        final_trail = best_trading_params.get("trail_distance_atr_mult", 0.0)

        final_prof_thr = np.maximum(0.008, atr_series * final_tp_mult)
        final_stop_thr = np.maximum(0.002, atr_series * final_sl_mult)

        (
            final_returns, final_labels, _, final_durations, final_mfe, final_mae, _, _
        ) = compute_realized_returns(
            market_data, primary_signals,
            profit_threshold=final_prof_thr, stop_threshold=final_stop_thr,
            horizon=12, transaction_cost=DEFAULT_TRANSACTION_COST,
            min_event_spacing=2, trail_distance_atr_mult=final_trail,
            atr_series=atr_series
        )

        valid_final_mask = ~final_labels.isna()
        if valid_final_mask.sum() < 50:
            tprint_warning("⚠️ Layer 3: Insufficient events. Aborting.")
            return {"success": False}

        final_t_events = final_returns.index[valid_final_mask]

        batch_con_final = full_consistency.reindex(final_t_events).fillna(0).values
        batch_vol_final = full_volatility.reindex(final_t_events).fillna(0).values
        batch_uniq_final = compute_uniqueness(final_t_events, market_data.index)

        final_weights = generate_weights_per_label(
            returns=final_returns[valid_final_mask].values,
            t_events=final_t_events,
            close_series=None,
            consistency_scores=batch_con_final,
            uniqueness_scores=batch_uniq_final.values,
            vol_proxy=batch_vol_final,
            **best_weighting_params
        )

        mf_config_final = meta_feature_cfg.copy()
        mf_config_final['kalman_Q'] = best_kalman_params.get('kalman_Q', 1e-4)
        mf_config_final['kalman_R'] = best_kalman_params.get('kalman_R', 0.01)

        _, final_meta_feats, _, _ = build_meta_features_for_model(
            market_data=market_data, primary_signals=primary_signals,
            realized_returns=final_returns, binary_labels=final_labels,
            event_durations=final_durations, mfe_series=final_mfe, mae_series=final_mae,
            adaptive_stop_threshold=final_stop_thr, horizon=12,
            volume_available=volume_available, meta_feature_cfg=mf_config_final,
        )

        X_final = final_meta_feats.loc[valid_final_mask].fillna(0)
        y_final = final_labels[valid_final_mask]

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

        # Precompute values needed for Layer 3 utility calculation
        final_returns_arr = final_returns[valid_final_mask].values.astype(float)
        final_labels_arr = y_final.values.astype(float)

        def layer3_objective(model_params: Dict[str, Any]) -> float:
            """
            Layer 3 objective using stability-adjusted utility with:
            - Pre-calibrated probabilities for position sizing
            - Per-fold Sharpe stability assessment
            - Trapezoidal AUC gate (penalizes leakage and randomness)
            - Trade density modifier
            """
            model = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42, **model_params)
            
            n_cv_folds = 5
            kf = TimeSeriesSplit(n_splits=n_cv_folds)
            
            # Collect per-fold predictions and metrics
            all_preds = np.full(len(X_final), np.nan)
            fold_aucs = []
            fold_sharpes = []
            
            for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_final)):
                X_tr, X_te = X_final.iloc[tr_idx], X_final.iloc[te_idx]
                y_tr, y_te = y_final.iloc[tr_idx], y_final.iloc[te_idx]
                w_tr = final_weights[tr_idx]
                
                if len(np.unique(y_tr)) < 2:
                    continue
                
                # Train model
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                preds = model.predict_proba(X_te)[:, 1]
                all_preds[te_idx] = preds
                
                # Compute fold AUC
                try:
                    fold_auc = roc_auc_score(y_te, preds)
                    fold_aucs.append(fold_auc)
                except Exception:
                    pass
                
                # Calibrate predictions and compute fold Sharpe
                try:
                    y_te_arr = y_te.values.astype(float)
                    preds_arr = preds.astype(float)
                    ret_te = final_returns_arr[te_idx]
                    
                    # Isotonic calibration on this fold (using train data for calibrator)
                    from sklearn.isotonic import IsotonicRegression
                    iso = IsotonicRegression(out_of_bounds='clip')
                    train_preds = model.predict_proba(X_tr)[:, 1]
                    iso.fit(train_preds, y_tr.values)
                    preds_cal = iso.predict(preds_arr)
                    
                    # Compute sized returns using calibrated probabilities
                    sized_returns = []
                    for prob, ret in zip(preds_cal, ret_te):
                        size = linear_size_from_prob(prob, max_exposure=1.0, min_prob=0.5, scale=1.0)
                        sized_returns.append(size * ret)
                    
                    sized_returns = np.array(sized_returns)
                    if len(sized_returns) > 1 and np.std(sized_returns) > 1e-9:
                        fold_sharpe = np.mean(sized_returns) / np.std(sized_returns)
                    else:
                        fold_sharpe = 0.0
                    
                    fold_sharpes.append(fold_sharpe)
                except Exception:
                    pass
            
            # Check minimum data quality
            if len(fold_aucs) < 2 or len(fold_sharpes) < 2:
                return -1.0
            
            # Compute mean AUC
            mean_auc = float(np.mean(fold_aucs))
            
            # Compute trades per day
            trades_per_day = len(final_returns_arr) / max(days_span, 1)
            
            # Calculate utility using trapezoidal gate
            utility = calculate_hpo_utility(
                folds_sharpe=np.array(fold_sharpes),
                auc=mean_auc,
                trades_per_day=trades_per_day,
                lambda_vol=1.2,   # Penalty for Sharpe volatility across folds
                w_auc=1.0,        # Strict AUC gate
                w_den=0.5,        # Moderate density weight
            )
            
            return utility

        l3_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(n_trials=30, execution_mode="full", direction="maximize", seed=42)
        )
        l3_result = l3_optimizer.optimize(objective=layer3_objective, search_space=layer3_search_space)
        best_model_params = l3_result.get("best_params", {})
        tprint_success(f"✅ Layer 3 Complete. Best AUC: {l3_result.get('best_value', 0):.4f}")

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

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        summary_data = [
            {"Layer": "0. Kalman", "Best Score": kalman_result.get("best_value", 0), "Params": str(best_kalman_params)},
            {"Layer": "1. Weighting", "Best Score": "N/A", "Params": str(best_weighting_params)},
            {"Layer": "2. Trading", "Best Score": best_l2_score, "Params": str(best_trading_params)},
            {"Layer": "3. Model", "Best Score": l3_result.get("best_value", 0), "Params": str(best_model_params)}
        ]
        pd.DataFrame(summary_data).to_csv(outcomes_dir / f"hpo_multi_stage_summary_{symbol}_{timestamp}.csv", index=False)

        md_path = outcomes_dir / f"hpo_multi_stage_report_{symbol}_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(f"# Multi-Stage HPO Report: {symbol}\n\n")
            f.write(f"**Stage 0 (Kalman):** Optimized signal features (IC={kalman_result.get('best_value', 0):.4f}).\n")
            f.write(f"```json\n{json.dumps(best_kalman_params, indent=2)}\n```\n\n")
            f.write(f"**Layer 1 (Weighting):** Optimized sample weights.\n")
            f.write(f"```json\n{json.dumps(best_weighting_params, indent=2)}\n```\n\n")
            f.write(f"**Layer 2 (Trading):** Optimized TP/SL/Trail with dynamic TBM & Weighting.\n")
            f.write(f"- Robust Edge Score: {best_l2_score:.4f}\n")
            f.write(f"```json\n{json.dumps(best_trading_params, indent=2)}\n```\n\n")
            f.write(f"**Layer 3 (Model):** Tuned LightGBM hyperparameters.\n")
            f.write(f"- Mean AUC: {l3_result.get('best_value', 0):.4f}\n")
            f.write(f"```json\n{json.dumps(best_model_params, indent=2)}\n```\n\n")

        json_path = outcomes_dir / f"hpo_multi_stage_best_params_{symbol}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(full_best_params, f, indent=2, default=str)

        metrics = {
            "layer0_ic": kalman_result.get("best_value", 0),
            "layer2_score": best_l2_score,
            "layer3_auc": l3_result.get("best_value", 0),
            "best_params": full_best_params,
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
        artifacts = {
            "best_params_json": str(json_path),
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
