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

    # --- 2. Regime-Based Learnability Gate (The "Min-Max" Logic) ---
    # We compute EconAUC for each regime separately.
    regime_aucs = []
    if regime_col in df_results.columns:
        regimes = df_results[regime_col].unique()
        for r in regimes:
            mask = df_results[regime_col] == r
            # Safety check: Ignore regimes with statistically insignificant sample size
            if mask.sum() < 20:
                continue

            try:
                r_auc = compute_economic_auc(
                    df_results.loc[mask, 'y_true'].to_numpy(dtype=float),
                    df_results.loc[mask, 'y_prob'].to_numpy(dtype=float),
                    df_results.loc[mask, 'ret_bps'].to_numpy(dtype=float)
                )
                regime_aucs.append(r_auc)
            except Exception:
                pass

    # CRITICAL: We gate based on the WORST performing regime.
    # If it fails in "Low Vol", the whole strategy is penalized.
    # Default to 0.5 if no valid regimes found.
    worst_case_auc = np.min(regime_aucs) if regime_aucs else 0.5

    # Gate opens at 0.52 (better than random), closes hard below that.
    gate_auc = sigmoid_gate(worst_case_auc, threshold=0.52, sharpness=15.0)

    # --- 3. Stability Gates (Sharpe & Freq) ---
    # Standard Sharpe gate
    sharpe_val = stats.get('sharpe_ratio', 0.0)
    gate_sharpe = sigmoid_gate(sharpe_val, threshold=1.0, sharpness=2.0)

    # Frequency gate (Soft penalty for starvation)
    trades_per_day = stats.get('trades_per_day', 0.0)
    gate_freq = sigmoid_gate(trades_per_day, threshold=1.5, sharpness=0.5)

    # --- 4. Final Robust Score ---
    final_score = robust_score * gate_auc * gate_sharpe * gate_freq

    return final_score


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
        returns_series = close_series.pct_change().fillna(0.0)

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

        # ------------------------------------------------------------------
        # STAGE 0: SIGNAL/FEATURE HPO (KALMAN TUNING)
        # ------------------------------------------------------------------
        tprint_info("🧪 Stage 0: Optimizing Kalman Signal Parameters...")

        kalman_search_space = {
            "kalman_Q": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "kalman_R": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
        }

        def kalman_objective(params: Dict[str, Any]) -> float:
            # 1. Generate Smoothed Signals (Primary)
            # We can't easily re-run generate_primary_signals as it's complex.
            # Instead, let's optimize a specific feature's predictive power.
            # "Does this setting make the indicator actually predict price direction?"

            # Use Kalman Smoothed Labels on raw binary labels from a default TBM
            # This checks if the smoothing improves target alignment.

            # Default TBM for Stage 0
            def_prof = np.maximum(0.008, atr_series * 2.0)
            def_stop = np.maximum(0.004, atr_series * 1.0)

            _, stage0_labels, _, _, _, _, _, _ = compute_realized_returns(
                market_data, primary_signals,
                profit_threshold=def_prof, stop_threshold=def_stop,
                horizon=12, transaction_cost=DEFAULT_TRANSACTION_COST
            )

            valid_mask = ~stage0_labels.isna()
            if valid_mask.sum() < 50: return -1.0

            y_raw = stage0_labels[valid_mask]

            # Kalman Smooth the Binary Labels
            smoothed, _ = kalman_smooth_labels(
                stage0_labels,
                Q=params['kalman_Q'],
                R=params['kalman_R'],
                volatility=full_volatility
            )

            y_smooth = smoothed[valid_mask]

            # Metric: IC between Smoothed Label and Future Return (forward 12 bars)
            fwd_ret = returns_series.rolling(12).sum().shift(-12)[valid_mask]

            if len(y_smooth) != len(fwd_ret): return -1.0

            # Clean NaNs
            mask2 = ~np.isnan(y_smooth) & ~np.isnan(fwd_ret)
            if mask2.sum() < 20: return -1.0

            ic, _ = spearmanr(y_smooth[mask2], fwd_ret[mask2])
            return float(ic) if np.isfinite(ic) else -1.0

        kalman_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(n_trials=30, execution_mode="full", direction="maximize", seed=42)
        )
        kalman_result = kalman_optimizer.optimize(objective=kalman_objective, search_space=kalman_search_space)
        best_kalman_params = kalman_result.get("best_params", {})
        tprint_success(f"✅ Stage 0 Complete. Best IC: {kalman_result.get('best_value', 0):.4f}")
        tprint_info(f"   Best Kalman Params: {best_kalman_params}")

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
        tprint_success("✅ Meta-features pre-calculated.")

        def layer2_objective(trial_params: Dict[str, Any]) -> float:
            # A. TBM SIMULATION (Constructive)
            sl_mult = trial_params["sl_atr_mult"]
            rr = trial_params["risk_reward_ratio"]
            tp_mult = sl_mult * rr # Guarantee: TP >= 2 * SL

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
                atr_series=atr_series # Pass for trailing logic
            )

            valid_idx = ~l2_labels.isna()
            if valid_idx.sum() < 50: return -1.0

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

            # D. FAST MODEL TRAINING
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                cv_preds = cross_val_predict(
                    fast_model, X_trial, l2_labels_clean, cv=3,
                    method='predict_proba', fit_params={'sample_weight': sample_weights}, n_jobs=-1
                )[:, 1]
            except Exception:
                return -1.0

            # E. SCORING (Robust Edge)
            df_res = pd.DataFrame({
                'y_true': l2_labels_clean.values,
                'y_prob': cv_preds,
                'ret_bps': l2_returns_clean.values
            })
            stats = {
                'num_trades': len(df_res),
                'sharpe_ratio': df_res['ret_bps'].mean() / (df_res['ret_bps'].std() + 1e-9),
                'trades_per_day': len(df_res) / (len(market_data)/96.0)
            }
            return compute_robust_hpo_objective(stats, df_res, regime_col='none')

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

        def layer3_objective(model_params: Dict[str, Any]) -> float:
            model = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42, **model_params)
            cv_scores = []
            kf = TimeSeriesSplit(n_splits=5)
            for tr_idx, te_idx in kf.split(X_final):
                X_tr, X_te = X_final.iloc[tr_idx], X_final.iloc[te_idx]
                y_tr, y_te = y_final.iloc[tr_idx], y_final.iloc[te_idx]
                w_tr = final_weights[tr_idx]
                if len(np.unique(y_tr)) < 2: continue
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                preds = model.predict_proba(X_te)[:, 1]
                try: cv_scores.append(roc_auc_score(y_te, preds))
                except: pass
            return np.mean(cv_scores) if cv_scores else 0.0

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
            "best_params": full_best_params
        }
        artifacts = {
            "best_params_json": str(json_path),
            "report_md": str(md_path)
        }
        return {"success": True, "metrics": metrics, "artifacts": artifacts}

def register_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry

    step_registry.register("meta_labeling_hpo_sample_weighted", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb_weighted", MetaLabelingHPOSampleWeightedStep)
    tprint("✅ Meta-labeling HPO sample weighted step registered (aliases: meta_labeling_hpo_sample_weighted, sr_labeling_xgb_weighted)", "SUCCESS")


# Auto-register when module is imported
register_meta_labeling_hpo_sample_weighted_step()
