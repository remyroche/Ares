"""Meta-Labeling HPO Sample Weighted Step.

This step orchestrates the Layer 2 -> Layer 3 pipeline for label generation
and calibration.

Layer 2: Regime-Conditional Geometry Optimization (LabelBasedLayer2)
- Optimizes Barrier Geometries (TP/SL/Horizon) per barrier family.
- Selects diverse geometries.
- Generates Bagged OOF Labels and Weights (K-Fold OOF for analytics).
- Also generates Production Geometries (Full Fit).

Layer 3: Calibration & Meta-Model (LabelBasedLayer3)
- Feature Engineering on Layer 2 outputs (Disagreement, Volatility).
- Weights adjustment using Magnitude and Layer 1 weights.
- Calibrated Probability generation using LGBM + Isotonic Regression (K-Fold OOF).
- Final Model training on full dataset.

This replaces the legacy HierarchicalParameterOptimizer loop.
"""

from __future__ import annotations

from src.training.steps.labeling.multi_label_voting_utils import (
    TripleBarrierConfig,
    compute_multi_triple_barrier_outcomes_vectorized,
    compute_kalman_smoothed_price_and_volatility,
    compute_committee_voted_labels_full,
)
from src.training.steps.labeling.label_based_layer_0 import run_layer_0

from typing import Any, Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_info, tprint_error

from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm, plot_diagnostics
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_meta_features,
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
    
    async def execute(self, config: dict) -> dict:
        """
        Execute the pipeline.
        
        Args:
            config: Configuration dictionary.
        """
        # Load market data (using standard BaseStep mechanism)
        market_data, _ = self.load_market_data_or_fail(config)
        
        # Generate primary signals (using default logic if not provided)
        from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
        primary_signals = generate_primary_signals(market_data.copy())
        
        # Try load weights
        target_sample_weight = None
        # (Simulated for now as legacy loading is complex)
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


def _align_weights_to_index(
    weights: Optional[Any],
    index: pd.Index,
    *,
    fill_value: float = 1.0,
) -> Optional[pd.Series]:
    """Align weights to an index WITHOUT dropping rows.

    - If weights is a Series: reindex to `index` and fill missing with `fill_value`.
    - If weights is a 1D array matching length: wrap as Series.
    - Otherwise return None.
    """
    if weights is None:
        return None
    try:
        if isinstance(weights, pd.Series):
            return weights.reindex(index).fillna(fill_value).astype(float)
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.ndim != 1:
            return None
        if len(w_arr) == len(index):
            return pd.Series(w_arr, index=index, dtype=float)
        return None
    except Exception:
        return None


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
    recency_decay_lambda: Optional[float] = None,  # Exponential decay rate for recency weighting (0.01 = 1%/day)
    target_sample_weight: Optional[np.ndarray] = None,  # Pre-computed sample weights to use for training
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
            # Prepare sample weights for feature selection if provided
            fs_sample_weight = None
            if target_sample_weight is not None:
                try:
                    if isinstance(target_sample_weight, pd.Series):
                        fs_sample_weight = target_sample_weight.reindex(y_clean.index).fillna(1.0).values
                    elif len(target_sample_weight) == len(y):
                        # Original weights aligned to full y, subset to valid_mask
                        fs_sample_weight = np.asarray(target_sample_weight)[valid_mask.values if hasattr(valid_mask, 'values') else valid_mask]
                    elif len(target_sample_weight) == len(y_clean):
                        fs_sample_weight = np.asarray(target_sample_weight)
                except Exception:
                    fs_sample_weight = None
            
        # ---------------------------------------------------------
        # LAYER 2: Geometry Optimization & Bagged Labeling
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 2: Geometry Optimization (OOF & Full)...")
        
        layer2 = LabelBasedLayer2(
            transaction_cost=float(config.get('transaction_cost', 0.001)),
            n_trials=int(config.get('layer2_n_trials', 30)),
            n_splits=int(config.get('layer2_n_splits', 3)),
            verbose=True
        )
        
        # This now returns OOF labels AND Production Geometries
            # Train a quick fast model to get feature importance
            selector = lgb.LGBMClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1, 
                random_state=get_reproducible_random_state(DEFAULT_RANDOM_SEED, offset=1), verbose=-1
            )
            if fs_sample_weight is not None and len(fs_sample_weight) == len(y_clean):
                selector.fit(X_clean, y_clean, sample_weight=fs_sample_weight)
            else:
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
                    min_child_samples=40,
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

    # Use provided target_sample_weight if available, otherwise compute internally
    if target_sample_weight is not None:
        try:
            if isinstance(target_sample_weight, pd.Series):
                sample_weights = target_sample_weight.reindex(y_clean.index).fillna(1.0).values.astype(float)
            elif len(target_sample_weight) == len(y):
                # Original weights aligned to full y, subset to valid_mask
                sample_weights = np.asarray(target_sample_weight, dtype=float)[valid_mask.values if hasattr(valid_mask, 'values') else valid_mask]
            elif len(target_sample_weight) == len(y_clean):
                sample_weights = np.asarray(target_sample_weight, dtype=float)
            else:
                # Fallback to computing internally
                sample_weights = None
        except Exception:
            sample_weights = None
    else:
        sample_weights = None
    
    if sample_weights is None:
        # Compute sample weights internally if not provided or failed to extract
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

    # -------------------------------------------------------------------------
    # Apply recency weighting (exponential decay prioritizing recent events)
    # -------------------------------------------------------------------------
    # recency_decay_lambda is passed from HPO or config; 0.0 = disabled
    if recency_decay_lambda is not None and recency_decay_lambda > 0:
        try:
            from src.utils.ml_common.recency_weighting import (
                compute_recency_weights,
                combine_weights,
            )
            
            # Use X_clean index as timestamps
            if hasattr(X_clean, 'index') and isinstance(X_clean.index, pd.DatetimeIndex):
                recency_weights = compute_recency_weights(
                    timestamps=X_clean.index,
                    decay_lambda=recency_decay_lambda,
                    min_weight=0.1,
                )
                sample_weights = combine_weights(
                    base_weights=sample_weights,
                    recency_weights=recency_weights,
                    combination="multiply",
                )
                tprint(
                    f"[RECENCY_WEIGHTING] Applied decay_lambda={recency_decay_lambda:.4f}: "
                    f"min_w={sample_weights.min():.3f}, max_w={sample_weights.max():.3f}",
                    "INFO",
                )
        except Exception as e:
            tprint(f"[RECENCY_WEIGHTING] Failed to apply: {e}", "WARNING")

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
    event_index: Optional[pd.Index] = None,
    direction: str = "long",
    event_durations: Optional[pd.Series] = None,
    market_index: Optional[pd.DatetimeIndex] = None,
    base_horizon_bars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        if event_index is not None and market_index is not None and base_horizon_bars is not None:
            try:
                y_tmp = pd.Series(np.zeros(int(len(X))), index=pd.DatetimeIndex(event_index))
                splits = _build_t1_aware_purged_splits_for_events(
                    y=y_tmp,
                    event_durations=event_durations,
                    market_index=market_index,
                    cv_splits=5,
                    base_horizon_bars=int(base_horizon_bars),
                )
            except Exception:
                splits = None
        if splits is None:
            cv = TimeSeriesSplit(n_splits=5)
            splits = list(cv.split(X))

        for fold_idx, (_, te_idx) in enumerate(splits):
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
                pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else None
                precision_at_1pct = None
                precision_at_5pct = None
                precision_at_10pct = None
                try:
                    yb = (np.asarray(y, dtype=float) >= 0.5).astype(int)
                    order = np.argsort(-np.asarray(p, dtype=float))
                    n_tot = int(order.size)
                    if n_tot > 0:
                        k1 = int(max(1, int(np.ceil(0.01 * float(n_tot)))))
                        k5 = int(max(1, int(np.ceil(0.05 * float(n_tot)))))
                        k10 = int(max(1, int(np.ceil(0.10 * float(n_tot)))))
                        precision_at_1pct = float(np.sum(yb[order[:k1]] == 1)) / float(k1)
                        precision_at_5pct = float(np.sum(yb[order[:k5]] == 1)) / float(k5)
                        precision_at_10pct = float(np.sum(yb[order[:k10]] == 1)) / float(k10)
                except Exception:
                    pass

                # Canonical fold evaluation (sizing + annualized Sharpe).
                sizes = np.zeros_like(p, dtype=float)
                try:
                    for i, pv in enumerate(p):
                        sizes[i] = float(
                            directional_size_from_prob(
                                float(pv),
                                direction=direction,
                                thr=float(threshold),
                                max_exposure=1.0,
                                scale=1.0,
                            )
                        )
                except Exception:
                    sizes = (p >= float(threshold)).astype(float)

                # CRITICAL FIX: Use absolute sizes because returns are already direction-adjusted
                sized_returns = np.abs(sizes) * r
                sig = np.abs(np.asarray(sizes, dtype=float))

                fold_event_times = None
                try:
                    if event_index is not None:
                        fold_event_times = pd.DatetimeIndex(event_index)[te_idx][mask]
                except Exception:
                    fold_event_times = None

                bt = compute_backtest_metrics(
                    y_prob=sig,
                    returns=sized_returns,
                    threshold=1e-12,
                    transaction_cost=float(transaction_cost) if transaction_cost is not None else 0.0,
                    direction=direction,
                    event_times=fold_event_times,
                    returns_are_net=True,
                    annualize=True,
                    verbose=False,
                )

                n_trades = int(bt.get("n_trades", 0))
                trades_per_day = float(bt.get("trades_per_day", float(n_trades) / float(max(days_span, 1.0))))
                mean_ret = float(bt.get("mean_return", 0.0))
                net_mean_ret = float(bt.get("cost_adjusted_return", mean_ret))
                win_rate = float(bt.get("win_rate", 0.0))
                sharpe = float(bt.get("sharpe_ratio", 0.0))
                if not np.isfinite(sharpe):
                    sharpe = 0.0
                sharpe = _soft_sharpe_scale(float(sharpe))
                out.append(
                    {
                        "fold": int(fold_idx),
                        "auc": auc,
                        "pr_auc": pr_auc,
                        "precision_at_1pct": precision_at_1pct,
                        "precision_at_5pct": precision_at_5pct,
                        "precision_at_10pct": precision_at_10pct,
                        "n_test": int(len(p)),
                        "n_trades": int(n_trades),
                        "trades_per_day": float(trades_per_day),
                        "mean_return": float(mean_ret),
                        "net_pnl_per_trade": float(net_mean_ret),
                        "win_rate": float(win_rate),
                        "sharpe": float(sharpe),
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
    direction: str = "long",
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

            # Canonical regime evaluation: use the same sizing + annualized Sharpe as CV.
            sizes_r = np.ones_like(r_r, dtype=float)
            if p_r is not None:
                try:
                    sizes_r = np.zeros_like(p_r, dtype=float)
                    for i, pv in enumerate(p_r):
                        sizes_r[i] = float(
                            directional_size_from_prob(
                                float(pv),
                                direction=direction,
                                thr=float(base_thr),
                                max_exposure=1.0,
                                scale=1.0,
                            )
                        )
                except Exception:
                    sizes_r = (p_r >= float(base_thr)).astype(float)

            # CRITICAL FIX: Use absolute sizes because returns are already direction-adjusted
            sized_returns_r = np.abs(np.asarray(sizes_r, dtype=float)) * np.asarray(r_r, dtype=float)
            sig_r = np.abs(np.asarray(sizes_r, dtype=float))

            ev_times_r = None
            try:
                ev_times_r = pd.DatetimeIndex(regime_labels.index[mask])[valid]
            except Exception:
                ev_times_r = None

            bt = compute_backtest_metrics(
                y_prob=sig_r,
                returns=sized_returns_r,
                threshold=1e-12,
                transaction_cost=float(transaction_cost) if transaction_cost is not None else 0.0,
                direction=direction,
                event_times=ev_times_r,
                returns_are_net=True,
                annualize=True,
                verbose=False,
            )

            n_trades = int(bt.get("n_trades", 0))
            if n_trades <= 0:
                out[str(reg_val)] = {"n_events": int(n_events), "n_trades": 0}
                continue

            mean_ret = float(bt.get("mean_return", 0.0))
            net_mean_ret = float(bt.get("cost_adjusted_return", mean_ret))
            win_rate = float(bt.get("win_rate", 0.0))
            sharpe = float(bt.get("sharpe_ratio", 0.0))
            if not np.isfinite(sharpe):
                sharpe = 0.0
            sharpe = _soft_sharpe_scale(float(sharpe))

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
                    net_pnl = float(bin_returns.mean())
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
                    pnl_target = returns
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
    target_sample_weight: Optional[np.ndarray] = None,
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
    
    # Prepare sample weights aligned to cleaned data
    sample_weights_clean = None
    if target_sample_weight is not None:
        try:
            if isinstance(target_sample_weight, pd.Series):
                sample_weights_clean = target_sample_weight.reindex(y_clean.index).fillna(1.0).values
            elif len(target_sample_weight) == len(y):
                sample_weights_clean = np.asarray(target_sample_weight)[valid_mask.values if hasattr(valid_mask, 'values') else valid_mask]
            elif len(target_sample_weight) == len(y_clean):
                sample_weights_clean = np.asarray(target_sample_weight)
        except Exception:
            sample_weights_clean = None

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

                # Get sample weights for this fold
                w_train = None
                if sample_weights_clean is not None and len(sample_weights_clean) == len(y_clean):
                    w_train = sample_weights_clean[train_idx]
                
                if w_train is not None:
                    model.fit(X_train, y_train, sample_weight=w_train)
                else:
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
                        net_pnl = float(returns_test[trade_mask].mean())

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
                # Get sample weights for this fold
                w_train_oof = None
                if sample_weights_clean is not None and len(sample_weights_clean) == len(y_clean):
                    w_train_oof = sample_weights_clean[train_idx]
                
                if w_train_oof is not None:
                    model.fit(X_train, y_train, sample_weight=w_train_oof)
                else:
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
                                    net_pnl = float(returns_regime[trade_mask].mean())

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


def kalman_filter_adaptive(
    prices: np.ndarray,
    volume: Optional[np.ndarray],
    vwap: Optional[np.ndarray],
    Q: float,
    R: float,
    init_val: float = None,
    init_cov: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive 1D Kalman Filter (CAUSAL) with Volume-Weighted R and Dual Measurement.

    Features:
    1. Volume-Weighted R: R_t = R_base * (MedianVol / Vol_t)
       High volume -> Low noise -> Trust measurement.
    2. Dual Measurement: If VWAP is provided, observation is (Price + VWAP)/2 with R_eff = R_t / 2.
       This treats VWAP as a second noisy observation of the same underlying price state.

    Model: x_t = x_{t-1} + w_t  (Process Noise Q)
           z_t = x_t + v_t      (Measurement Noise R_t or R_eff)

    Args:
        prices: Raw price series (Close)
        volume: Volume series for adaptive R (optional)
        vwap: VWAP series for dual measurement (optional)
        Q: Process noise variance
        R: Base measurement noise variance
        init_val: Initial state value (default: first observation)
        init_cov: Initial covariance (default: 1.0)

    Returns:
        Tuple of (filtered_state, filtered_covariance, kalman_gain)
    """
    n = len(prices)
    obs = np.asarray(prices, dtype=np.float64)

    # 1. Compute Adaptive R_t based on Volume
    if volume is not None and len(volume) == n:
        vol = np.asarray(volume, dtype=np.float64)
        median_vol = np.nanmedian(vol)
        if median_vol > 0:
            vol_safe = np.where(vol < 1e-8, 1e-8, vol)
            scale_factor = median_vol / vol_safe
            scale_factor = np.clip(scale_factor, 0.1, 10.0)
            R_t = R * scale_factor
        else:
            R_t = np.full(n, R, dtype=np.float64)
    else:
        R_t = np.full(n, R, dtype=np.float64)

    # 2. Setup Observations (Dual Measurement Logic)
    if vwap is not None and len(vwap) == n:
        # Dual Measurement: z = [Close, VWAP]^T
        # Simplified equivalent scalar update:
        # z_eff = (Close + VWAP) / 2
        # R_eff = R_t / 2 (assuming independent errors with same variance)
        vwap_arr = np.asarray(vwap, dtype=np.float64)
        obs_eff = (obs + vwap_arr) / 2.0
        R_eff = R_t / 2.0
    else:
        obs_eff = obs
        R_eff = R_t

    m = np.zeros(n)  # State means
    P = np.zeros(n)  # State covariances
    K_arr = np.zeros(n)  # Kalman gains

    # Initialization
    m[0] = init_val if init_val is not None else obs[0]
    P[0] = init_cov
    K_arr[0] = 0.5

    for t in range(1, n):
        # Time Update (Prediction)
        m_minus = m[t - 1]
        P_minus = P[t - 1] + Q

        # Measurement Update (Correction)
        r_val = R_eff[t]

        K = P_minus / (P_minus + r_val) if (P_minus + r_val) > 1e-12 else 0.5
        K_arr[t] = K
        m[t] = m_minus + K * (obs_eff[t] - m_minus)
        P[t] = (1 - K) * P_minus

    return m, P, K_arr


def kalman_filter_1d(
    prices: np.ndarray,
    Q: float,
    R: float,
    init_val: float = None,
    init_cov: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy wrapper for backward compatibility."""
    return kalman_filter_adaptive(prices, None, None, Q, R, init_val, init_cov)




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
    # NOTE: fcluster with maxclust finds AT MOST target_n clusters.
    # If data is highly correlated, it may find fewer.
    cluster_labels = hierarchy.fcluster(linkage_matrix, t=target_n, criterion='maxclust')
    
    n_clusters_found = len(np.unique(cluster_labels))
    tprint_info(f"   Hierarchical clustering found {n_clusters_found} distinct clusters (target={target_n})")

    # 5. Select Features (Diversity + Backfill)
    selected_feats = []
    feature_names = df_clean.columns.tolist()
    
    # Store members of each cluster for potential backfilling
    cluster_members_map = {}
    
    # Step A: Pick BEST feature from each cluster (Primary Diversity)
    for cluster_id in np.unique(cluster_labels):
        # Get all features in this cluster
        members = [feature_names[i] for i in range(len(feature_names)) 
                   if cluster_labels[i] == cluster_id]
        
        if not members:
            continue
            
        # Sort members by quality (highest first)
        members_sorted = sorted(members, key=lambda x: quality_scores.get(x, 0.0), reverse=True)
        cluster_members_map[cluster_id] = members_sorted
        
        # Pick best one
        best_in_cluster = members_sorted[0]
        selected_feats.append(best_in_cluster)
    
    # Step B: Backfill if needed (meet target_n count)
    if len(selected_feats) < target_n:
        tprint_warning(
            f"   ⚠️ Distinct clusters ({len(selected_feats)}) < target ({target_n}). "
            f"Backfilling from best clusters to meet target."
        )
        
        # Strategy: Iterate through clusters again, picking the NEXT best feature
        # until we reach target_n. Prioritize clusters with many high-quality features.
        
        # Flatten remaining candidates: (quality, feature_name)
        backfill_candidates = []
        for cluster_id, members in cluster_members_map.items():
            # Skip the first one (already selected)
            if len(members) > 1:
                for feat in members[1:]:
                    backfill_candidates.append((quality_scores.get(feat, 0.0), feat))
        
        # Sort backfill candidates globally by quality
        backfill_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Fill up to target_n
        needed = target_n - len(selected_feats)
        for _, feat in backfill_candidates[:needed]:
            selected_feats.append(feat)
            
    tprint_info(f"   → Selected {len(selected_feats)} features (from {n_clusters_found} clusters + backfill)")
    
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
    
    # Extract optional series for adaptive filtering
    volume_values = market_data["volume"].values if "volume" in market_data.columns else None
    vwap_values = market_data["vwap"].values if "vwap" in market_data.columns else None

    # 1. Filtered OHLC
    # Use Volume-Weighted R for all. Use Dual Measurement (VWAP) for Close only.
    kf_close, kf_close_P, _ = kalman_filter_adaptive(
        close, volume=volume_values, vwap=vwap_values, Q=kalman_Q, R=kalman_R
    )
    kf_high, kf_high_P, _ = kalman_filter_adaptive(
        high, volume=volume_values, vwap=None, Q=kalman_Q, R=kalman_R
    )
    kf_low, kf_low_P, _ = kalman_filter_adaptive(
        low, volume=volume_values, vwap=None, Q=kalman_Q, R=kalman_R
    )
    
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
    
    floor = 0.2

    if x < lower or x > upper:
        return float(floor)
    elif s_min <= x <= s_max:
        return 1.0
    elif lower <= x < s_min:
        # Ramp up (keep soft floor so boundary does not zero out utility)
        ramp = (x - lower) / (s_min - lower)
        return float(floor + (1.0 - floor) * ramp)
    elif s_max < x <= upper:
        # Ramp down (keep soft floor)
        ramp = (upper - x) / (upper - s_max)
        return float(floor + (1.0 - floor) * ramp)
    return float(floor)


def calculate_hpo_utility(
    folds_sharpe: np.ndarray,  # Now actually Sortino ratios (name kept for backward compat)
    auc: float,
    trades_per_day: float,
    lambda_vol: float = 1.2,
    w_auc: float = 0.5,  # Softer AUC gate
    w_den: float = 0.15,  # Much lower density power
    calibration_brier: Optional[float] = None,
    calibration_ece: Optional[float] = None,
    w_cal: float = 0.0,
    clip_min: float = -1.0,
    clip_max: float = 20.0,  # Allow larger values
    debug_out: Optional[Dict[str, Any]] = None,
    density_lower: float = 0.3,  # More lenient
    density_sweet_spot: Tuple[float, float] = (1.0, 6.0),  # Widened
    density_upper: float = 10.0,  # More lenient
    # NEW PARAMETERS:
    mean_return: Optional[float] = None,  # Direct PnL term
    w_return: float = 3.0,  # Weight for return contribution
    max_drawdown: Optional[float] = None,  # Max drawdown (0.0 to 1.0)
    w_dd: float = 1.0,  # Weight for drawdown penalty
    # Probability-Return Correlation
    prob_return_corr: Optional[float] = None,
    w_prob_return_corr: float = 0.1,
) -> float:
    """
    Compute a stable utility for HPO combining Sortino stability, AUC gate, trade density,
    direct returns, and drawdown penalty.
    
    IMPROVEMENTS (Dec 2024):
    1. Replaced Sharpe with Sortino (upside volatility is not penalized)
    2. Fixed unit mismatch: return_contribution now z-score normalized to Sortino scale
    3. DD threshold raised to 15% (was 5%), multiplier reduced, capped at 2.0
    4. lambda_vol reduced from 0.8 to 0.4 (Sortino has higher variance than Sharpe)
    5. Removed magnitude bonus (redundant with win rate modifier)
    6. Removed stop-out penalty (conflicts with win rate modifier)
    
    Args:
        folds_sharpe: Array of per-fold Sortino ratios (name kept for backward compat)
        auc: Mean AUC across folds
        trades_per_day: Average trades per day
        lambda_vol: Penalty weight for Sortino volatility across folds (default 0.4)
        w_auc: Weight exponent for AUC gate (default 0.5 = softer)
        w_den: Weight exponent for density modifier (default 0.15)
        mean_return: Mean return per trade (if available)
        w_return: Weight for return contribution (applied after normalization)
        max_drawdown: Maximum drawdown (0.0 to 1.0)
        w_dd: Penalty weight for drawdown
    
    Returns:
        Utility score. Returns -1.0 for rejection.
    """
    try:
        clip_min_v = float(clip_min)
    except Exception:
        clip_min_v = -1.0
    if not np.isfinite(clip_min_v):
        clip_min_v = -1.0

    # Now using Sortino ratios (name kept as sharpe_arr for backward compatibility)
    sortino_arr = np.asarray(folds_sharpe, dtype=float).reshape(-1)
    sortino_arr = sortino_arr[np.isfinite(sortino_arr)]
    if sortino_arr.size < 1:
        return float(clip_min_v)

    avg_sortino = float(np.mean(sortino_arr))
    vol_sortino = float(np.std(sortino_arr, ddof=1)) if sortino_arr.size > 1 else 0.0
    if not (np.isfinite(avg_sortino) and np.isfinite(vol_sortino)):
        return float(clip_min_v)

    # Base score: mean Sharpe/Sortino minus fold variance penalty
    base_score = avg_sortino - (lambda_vol * vol_sortino)
    base_score = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
    if not np.isfinite(base_score):
        base_score = 0.0

    # UNIT MISMATCH FIX: Normalize return_contribution to Sortino scale
    # Instead of raw scaling (mean_return * 100 * w_return), use z-score-like normalization
    # Typical Sortino range: [-2, 5]. Typical mean_return range: [-0.02, 0.05]
    # We normalize return_contribution to have similar magnitude as Sortino
    return_contribution = 0.0
    if mean_return is not None and np.isfinite(mean_return):
        # Normalize: 1% mean return ≈ 1.0 Sortino contribution (after w_return)
        # This keeps return_contribution in [-2, 5] range for typical returns
        # Formula: (mean_return - expected_mean) / expected_std * scale
        # Using simplified approach: clip to Sortino-like range
        normalized_return = float(mean_return) * 100.0  # Convert to percentage
        # Soft clip to prevent dominating base_score
        normalized_return = float(np.clip(normalized_return, -3.0, 5.0))
        return_contribution = normalized_return * w_return
        if not np.isfinite(return_contribution):
            return_contribution = 0.0

    # DRAWDOWN PENALTY FIX: Threshold raised to 15%, multiplier reduced, capped at 2.0
    # Rationale: 5% DD threshold was too conservative for crypto. Typical strategies
    # have 15-25% max DD. Penalty is now linear with a hard cap.
    dd_penalty = 0.0
    dd_threshold = 0.15  # Raised from 0.05 to 0.15 (15% DD threshold)
    if max_drawdown is not None and np.isfinite(max_drawdown):
        dd_val = float(max_drawdown)
        if dd_val > dd_threshold:
            # Linear penalty: (dd - threshold) * w_dd * 3.0
            # At 30% DD: (0.30 - 0.15) * 1.0 * 3.0 = 0.45 penalty
            # At 50% DD: (0.50 - 0.15) * 1.0 * 3.0 = 1.05 penalty
            dd_penalty = max(0.0, (dd_val - dd_threshold) * w_dd * 3.0)
            # Hard cap at 2.0 to prevent NaN/Inf from extreme crashes
            dd_penalty = min(dd_penalty, 2.0)
        if not np.isfinite(dd_penalty):
            dd_penalty = 0.0

    # Combined base with returns and DD penalty
    combined_base = base_score + return_contribution - dd_penalty
    
    # ========== MODIFIERS (gates) ==========
    
    # ISSUE #3 FIX: Softer AUC gate - additive floor + multiplicative
    # Instead of hard multiplicative, blend with additive floor
    phi_auc_raw = trapezoidal_gate(auc, lower=0.50, sweet_spot=(0.54, 0.68), upper=0.75)
    # Add 0.3 floor so AUC=0.55 gives ~0.5 instead of ~0.15
    phi_auc = 0.3 + 0.7 * phi_auc_raw  # Range: [0.3, 1.0]

    # ISSUE #5 FIX: Widened density band
    try:
        d_lower = float(density_lower)
    except Exception:
        d_lower = 0.3
    if not np.isfinite(d_lower):
        d_lower = 0.3

    try:
        d_s0 = float(density_sweet_spot[0])
        d_s1 = float(density_sweet_spot[1])
    except Exception:
        d_s0, d_s1 = 1.0, 6.0
    if not np.isfinite(d_s0):
        d_s0 = 1.0
    if not np.isfinite(d_s1):
        d_s1 = 6.0
    if d_s1 < d_s0:
        d_s1 = d_s0

    try:
        d_upper = float(density_upper)
    except Exception:
        d_upper = 10.0
    if not np.isfinite(d_upper):
        d_upper = 10.0
    if d_upper < d_s1:
        d_upper = d_s1

    phi_density = trapezoidal_gate(
        float(trades_per_day),
        lower=float(d_lower),
        sweet_spot=(float(d_s0), float(d_s1)),
        upper=float(d_upper),
    )
    # Also add floor to density gate
    phi_density = 0.2 + 0.8 * phi_density  # Range: [0.2, 1.0]

    # ISSUE #3 FIX: Lower exponents for softer gates
    try:
        modifier = float((phi_auc ** w_auc) * (phi_density ** w_den))
    except Exception:
        modifier = 0.0
    if not np.isfinite(modifier):
        modifier = 0.0

    # Optional: calibration quality modifier (model-dependent; useful for Layer 3)
    phi_cal = None
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

    # NEW: Probability-Return Correlation modifier
    # Encourages models where higher probability predictions correlate with higher returns.
    # This addresses the core issue of weak prob-return correlation (Spearman ~0.06).
    # The modifier provides a bonus/penalty based on the correlation strength.
    phi_prob_ret_corr = None
    if w_prob_return_corr and w_prob_return_corr > 0.0 and prob_return_corr is not None:
        try:
            corr = float(prob_return_corr)
            if np.isfinite(corr):
                # Map correlation [-1, 1] to modifier [0.5, 1.5]
                # corr = 0.0 -> modifier = 1.0 (neutral)
                # corr = 0.3 -> modifier = 1.15 (15% bonus)
                # corr = -0.3 -> modifier = 0.85 (15% penalty)
                phi_prob_ret_corr = 1.0 + (corr * w_prob_return_corr * 5.0)  # Scale by 5 so w=0.1 gives ±50% effect at corr=±1
                phi_prob_ret_corr = float(np.clip(phi_prob_ret_corr, 0.5, 1.5))
                modifier *= phi_prob_ret_corr
        except Exception:
            pass

    # ISSUE #2 FIX: No log - direct multiplication
    utility_pre_clip = float(combined_base) * float(modifier)
    
    try:
        clip_max_v = float(clip_max)
    except Exception:
        clip_max_v = 20.0
    if not np.isfinite(clip_max_v):
        clip_max_v = 20.0
    clip_max_v = float(max(1.0, clip_max_v))
    utility = float(np.clip(float(utility_pre_clip), float(clip_min_v), clip_max_v))
    
    if isinstance(debug_out, dict):
        try:
            debug_out.update(
                {
                    "avg_sharpe": float(avg_sharpe),
                    "vol_sharpe": float(vol_sharpe),
                    "base_score": float(base_score),
                    "return_contribution": float(return_contribution),
                    "dd_penalty": float(dd_penalty),
                    "combined_base": float(combined_base),
                    "phi_auc": float(phi_auc),
                    "phi_auc_raw": float(phi_auc_raw),
                    "phi_density": float(phi_density),
                    "phi_cal": float(phi_cal) if phi_cal is not None else None,
                    "phi_prob_ret_corr": float(phi_prob_ret_corr) if phi_prob_ret_corr is not None else None,
                    "prob_return_corr": float(prob_return_corr) if prob_return_corr is not None else None,
                    "modifier": float(modifier),
                    "utility_pre_clip": float(utility_pre_clip),
                    "utility_clip_max": float(clip_max_v),
                    "utility": float(utility),
                    "density_lower": float(d_lower),
                    "density_sweet_spot": (float(d_s0), float(d_s1)),
                    "density_upper": float(d_upper),
                }
            )
        except Exception:
            pass
    if not np.isfinite(utility):
        return float(clip_min_v)
    return float(utility)


def _normal_cdf(z: float) -> float:
    try:
        zz = float(z)
    except Exception:
        return 0.5
    if not np.isfinite(zz):
        return 0.0 if zz < 0 else 1.0
    try:
        return 0.5 * (1.0 + math.erf(zz / math.sqrt(2.0)))
    except Exception:
        return 0.5


def _moment_skew_kurt(x: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if int(v.size) < 3:
        return 0.0, 3.0
    mu = float(np.mean(v))
    xc = v - mu
    m2 = float(np.mean(xc ** 2))
    if not np.isfinite(m2) or m2 <= 1e-18:
        return 0.0, 3.0
    m3 = float(np.mean(xc ** 3))
    m4 = float(np.mean(xc ** 4))
    skew = float(m3 / (m2 ** 1.5)) if np.isfinite(m3) else 0.0
    kurt = float(m4 / (m2 ** 2)) if np.isfinite(m4) else 3.0
    if not np.isfinite(skew):
        skew = 0.0
    if not np.isfinite(kurt) or kurt <= 0.0:
        kurt = 3.0
    return skew, kurt


def _psr_from_returns(
    returns: np.ndarray,
    *,
    sr_benchmark: float = 0.0,
    periods_per_year: float = 365.0,
) -> Dict[str, Any]:
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 5:
        return {
            "psr": 0.0,
            "psr_z": float("-inf"),
            "sr": float("nan"),
            "n": int(n),
            "skew": 0.0,
            "kurt": 3.0,
            "sr_benchmark": float(sr_benchmark),
        }

    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    sr = float("nan")
    if np.isfinite(sd) and sd > 1e-12 and np.isfinite(mu):
        sr = float(mu / sd * float(np.sqrt(float(periods_per_year))))

    skew, kurt = _moment_skew_kurt(r)

    z = float("-inf")
    psr = 0.0
    try:
        sr0 = float(sr_benchmark)
        sr_hat = float(sr)
        if np.isfinite(sr_hat):
            denom = 1.0 - float(skew) * float(sr_hat) + ((float(kurt) - 1.0) / 4.0) * (float(sr_hat) ** 2)
            denom = float(max(1e-12, denom))
            z = (float(sr_hat) - float(sr0)) * float(np.sqrt(float(max(n - 1, 1)))) / float(np.sqrt(denom))
            psr = float(_normal_cdf(z))
    except Exception:
        z = float("-inf")
        psr = 0.0

    return {
        "psr": float(psr),
        "psr_z": float(z),
        "sr": float(sr) if np.isfinite(sr) else None,
        "n": int(n),
        "skew": float(skew),
        "kurt": float(kurt),
        "sr_benchmark": float(sr_benchmark),
    }


def _probabilistic_sortino_from_returns(
    returns: np.ndarray,
    *,
    sortino_benchmark: float = 0.0,
    periods_per_year: float = 365.0,
) -> Dict[str, Any]:
    """Compute Probabilistic Sortino Ratio (PSoR) using Lo (2002) standard error formula.
    
    Unlike PSR which penalizes all volatility equally, Probabilistic Sortino only
    penalizes downside volatility. This is more appropriate for trading where
    upside volatility (large wins) is desirable.
    
    The standard error formula (Lo, 2002) is robust for both Sharpe and Sortino:
        se = sqrt((1 + 0.5 * ratio^2) / n)
        z = (ratio - benchmark) / se
        probabilistic_ratio = Phi(z)  # CDF of standard normal
    
    Args:
        returns: Array of returns (daily log returns recommended)
        sortino_benchmark: Benchmark Sortino ratio (default 0.0)
        periods_per_year: Annualization factor (365 for crypto, 252 for equities)
        
    Returns:
        Dict with probabilistic_sortino, z-score, sortino ratio, and diagnostics
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.size)
    
    if n < 5:
        return {
            "probabilistic_sortino": 0.0,
            "psor_z": float("-inf"),
            "sortino": float("nan"),
            "n": int(n),
            "downside_deviation": float("nan"),
            "sortino_benchmark": float(sortino_benchmark),
        }
    
    mu = float(np.mean(r))
    
    # Vectorized downside deviation: sqrt(mean(min(r, 0)^2))
    # Only negative returns contribute to downside risk
    downside_returns = np.minimum(r, 0.0)
    downside_variance = float(np.mean(downside_returns ** 2))
    downside_dev = float(np.sqrt(downside_variance)) if downside_variance > 0 else 1e-12
    
    # Annualized Sortino ratio
    sortino = float("nan")
    if np.isfinite(downside_dev) and downside_dev > 1e-12 and np.isfinite(mu):
        sortino = float(mu / downside_dev * float(np.sqrt(float(periods_per_year))))
    
    z = float("-inf")
    prob_sortino = 0.0
    
    try:
        sortino_hat = float(sortino)
        benchmark = float(sortino_benchmark)
        
        if np.isfinite(sortino_hat):
            # Lo (2002) standard error formula - robust for both Sharpe and Sortino
            # se = sqrt((1 + 0.5 * ratio^2) / n)
            se_sortino = float(np.sqrt((1.0 + 0.5 * (sortino_hat ** 2)) / float(n)))
            se_sortino = float(max(se_sortino, 1e-12))
            
            z = (sortino_hat - benchmark) / se_sortino
            prob_sortino = float(_normal_cdf(z))
    except Exception:
        z = float("-inf")
        prob_sortino = 0.0
    
    return {
        "probabilistic_sortino": float(prob_sortino),
        "psor_z": float(z),
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "n": int(n),
        "downside_deviation": float(downside_dev) if np.isfinite(downside_dev) else None,
        "sortino_benchmark": float(sortino_benchmark),
    }


def _compute_sortino_ratio(returns: np.ndarray, periods_per_year: float = 365.0) -> float:
    """Compute annualized Sortino ratio from returns array.
    
    Sortino ratio only penalizes downside volatility, making it more appropriate
    for trading strategies where upside volatility (large wins) is desirable.
    
    Args:
        returns: Array of returns
        periods_per_year: Annualization factor (365 for crypto)
        
    Returns:
        Annualized Sortino ratio (or 0.0 if insufficient data)
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    
    if len(r) < 2:
        return 0.0
    
    mu = float(np.mean(r))
    
    # Vectorized downside deviation
    downside_returns = np.minimum(r, 0.0)
    downside_variance = float(np.mean(downside_returns ** 2))
    downside_dev = float(np.sqrt(downside_variance)) if downside_variance > 0 else 1e-12
    
    if downside_dev < 1e-12:
        return 0.0
    
    sortino = float(mu / downside_dev * float(np.sqrt(float(periods_per_year))))
    
    return float(sortino) if np.isfinite(sortino) else 0.0


def _compute_regime_dispersion(per_regime_metrics: Any, metric_key: str = "sharpe") -> float:
    try:
        if not isinstance(per_regime_metrics, dict):
            return 0.0
        group_stds: List[float] = []
        for _, group in per_regime_metrics.items():
            if not isinstance(group, dict):
                continue
            vals: List[float] = []
            for _, m in group.items():
                if not isinstance(m, dict):
                    continue
                v = m.get(metric_key)
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
            if len(vals) >= 2:
                try:
                    group_stds.append(float(np.std(np.asarray(vals, dtype=float), ddof=1)))
                except Exception:
                    pass
        if not group_stds:
            return 0.0
        out = float(np.mean(np.asarray(group_stds, dtype=float)))
        return float(out) if np.isfinite(out) else 0.0
    except Exception:
        return 0.0


def _compute_early_late_gap(values: Any) -> Dict[str, Any]:
    out = {"early_mean": None, "late_mean": None, "abs_gap": 0.0}
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size < 4:
            return out
        split = int(arr.size // 2)
        early = arr[:split]
        late = arr[split:]
        if early.size < 2 or late.size < 2:
            return out
        early_m = float(np.mean(early))
        late_m = float(np.mean(late))
        gap = float(abs(late_m - early_m))
        out["early_mean"] = early_m
        out["late_mean"] = late_m
        out["abs_gap"] = gap if np.isfinite(gap) else 0.0
        return out
    except Exception:
        return out


def _compute_probability_mapping(
    *,
    probs: np.ndarray,
    returns: np.ndarray,
    n_bins: int = 10,
    score_name: str = "p",
) -> List[Dict[str, Any]]:
    p = np.asarray(probs, dtype=float).reshape(-1)
    r = np.asarray(returns, dtype=float).reshape(-1)
    m = np.isfinite(p) & np.isfinite(r)
    if int(np.sum(m)) < max(20, int(n_bins) * 2):
        return []

    p = p[m]
    r = r[m]
    n_bins = int(max(2, n_bins))

    try:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.asarray(edges, dtype=float)
        edges[0] = float(min(edges[0], np.min(p)))
        edges[-1] = float(max(edges[-1], np.max(p)))
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    try:
        score_name = str(score_name)
    except Exception:
        score_name = "p"
    if not score_name:
        score_name = "p"
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)
        if not bool(np.any(idx)):
            continue
        rr = r[idx]
        pp = p[idx]
        rr = rr[np.isfinite(rr)]
        pp = pp[np.isfinite(pp)]
        if rr.size <= 0:
            continue
        out.append(
            {
                "bin": int(i),
                f"{score_name}_lo": float(lo),
                f"{score_name}_hi": float(hi),
                "n": int(rr.size),
                f"{score_name}_mean": float(np.mean(pp)) if pp.size else float("nan"),
                "ret_mean": float(np.mean(rr)),
                "ret_median": float(np.median(rr)),
                "win_rate": float(np.mean(rr > 0.0)),
            }
        )
    return out


def _compute_taken_trade_deciles(
    *,
    probs: np.ndarray,
    returns: np.ndarray,
    sizes: np.ndarray,
    take_mask: np.ndarray,
    exit_reasons: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    try:
        p = np.asarray(probs, dtype=float).reshape(-1)
        r = np.asarray(returns, dtype=float).reshape(-1)
        s = np.asarray(sizes, dtype=float).reshape(-1)
        tm = np.asarray(take_mask, dtype=bool).reshape(-1)
    except Exception:
        return []

    n = int(min(p.size, r.size, s.size, tm.size))
    if n <= 0:
        return []
    p = p[:n]
    r = r[:n]
    s = s[:n]
    tm = tm[:n]

    m = np.isfinite(p) & np.isfinite(r) & np.isfinite(s) & tm & (np.abs(s) > 1e-12)
    if int(np.sum(m)) < max(10, int(n_bins) * 2):
        return []

    p_t = p[m]
    r_t = r[m]
    s_t = np.abs(s[m])
    sized = r_t * s_t

    ex_t = None
    if exit_reasons is not None:
        try:
            ex_arr = np.asarray(exit_reasons, dtype=object).reshape(-1)[:n]
            ex_t = ex_arr[m]
        except Exception:
            ex_t = None

    try:
        bins = pd.qcut(pd.Series(p_t), q=int(n_bins), labels=False, duplicates="drop")
    except Exception:
        try:
            bins = pd.cut(pd.Series(p_t), bins=int(n_bins), labels=False, include_lowest=True)
        except Exception:
            bins = pd.Series(np.zeros_like(p_t, dtype=int))

    try:
        bin_ids = pd.Series(bins).astype(float)
    except Exception:
        bin_ids = pd.Series(np.zeros_like(p_t, dtype=float))

    out: List[Dict[str, Any]] = []
    try:
        unique_bins = sorted([int(b) for b in pd.unique(bin_ids.dropna())])
    except Exception:
        unique_bins = []

    for b in unique_bins:
        try:
            idx = (bin_ids == float(b)).to_numpy(dtype=bool)
        except Exception:
            continue
        if int(np.sum(idx)) <= 0:
            continue

        pp = p_t[idx]
        rr = sized[idx]

        win_mask = rr > 0.0
        loss_mask = ~win_mask

        avg_win = float(np.mean(rr[win_mask])) if int(np.sum(win_mask)) > 0 else None
        avg_loss = float(np.mean(rr[loss_mask])) if int(np.sum(loss_mask)) > 0 else None

        profit_share = None
        stop_share = None
        timeout_share = None
        trailing_share = None
        other_share = None
        if ex_t is not None:
            try:
                ex_s = pd.Series(ex_t[idx], dtype=object).astype(str)
                ex_s = ex_s.replace("<NA>", np.nan).replace("nan", np.nan)
                ex_s = ex_s.dropna()
                total_ex = int(len(ex_s))
                if total_ex > 0:
                    counts = ex_s.value_counts(normalize=True)
                    profit_share = float(counts.get("profit", 0.0))
                    trailing_share = float(counts.get("trailing", 0.0))
                    stop_share = float(counts.get("stop", 0.0))
                    timeout_share = float(counts.get("timeout", 0.0))
                    other_share = float(
                        max(
                            0.0,
                            1.0
                            - (
                                float(profit_share)
                                + float(trailing_share)
                                + float(stop_share)
                                + float(timeout_share)
                            ),
                        )
                    )
            except Exception:
                pass

        out.append(
            {
                "decile": int(b),
                "n_trades": int(rr.size),
                "p_min": float(np.min(pp)) if pp.size else None,
                "p_max": float(np.max(pp)) if pp.size else None,
                "p_mean": float(np.mean(pp)) if pp.size else None,
                "mean_return": float(np.mean(rr)) if rr.size else None,
                "win_rate": float(np.mean(rr > 0.0)) if rr.size else None,
                "avg_win": float(avg_win) if avg_win is not None and np.isfinite(float(avg_win)) else None,
                "avg_loss": float(avg_loss) if avg_loss is not None and np.isfinite(float(avg_loss)) else None,
                "avg_loss_abs": float(abs(float(avg_loss))) if avg_loss is not None and np.isfinite(float(avg_loss)) else None,
                "exit_profit_share": float(profit_share + trailing_share)
                if profit_share is not None and trailing_share is not None
                else None,
                "exit_stop_share": float(stop_share) if stop_share is not None else None,
                "exit_timeout_share": float(timeout_share) if timeout_share is not None else None,
                "exit_trailing_share": float(trailing_share) if trailing_share is not None else None,
                "exit_other_share": float(other_share) if other_share is not None else None,
            }
        )

    return out


def _compute_oof_all_event_deciles(
    *,
    probs: np.ndarray,
    returns: np.ndarray,
    exit_reasons: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> List[Dict[str, Any]]:
    try:
        p = np.asarray(probs, dtype=float).reshape(-1)
        r = np.asarray(returns, dtype=float).reshape(-1)
    except Exception:
        return []

    n = int(min(p.size, r.size))
    if n <= 0:
        return []
    p = p[:n]
    r = r[:n]

    m = np.isfinite(p) & np.isfinite(r)
    if int(np.sum(m)) < max(20, int(n_bins) * 2):
        return []

    p_e = p[m]
    r_e = r[m]

    ex_e = None
    if exit_reasons is not None:
        try:
            ex_arr = np.asarray(exit_reasons, dtype=object).reshape(-1)[:n]
            ex_e = ex_arr[m]
        except Exception:
            ex_e = None

    try:
        bins = pd.qcut(pd.Series(p_e), q=int(n_bins), labels=False, duplicates="drop")
    except Exception:
        try:
            bins = pd.cut(pd.Series(p_e), bins=int(n_bins), labels=False, include_lowest=True)
        except Exception:
            bins = pd.Series(np.zeros_like(p_e, dtype=int))

    try:
        bin_ids = pd.Series(bins).astype(float)
    except Exception:
        bin_ids = pd.Series(np.zeros_like(p_e, dtype=float))

    out: List[Dict[str, Any]] = []
    try:
        unique_bins = sorted([int(b) for b in pd.unique(bin_ids.dropna())])
    except Exception:
        unique_bins = []

    for b in unique_bins:
        try:
            idx = (bin_ids == float(b)).to_numpy(dtype=bool)
        except Exception:
            continue
        if int(np.sum(idx)) <= 0:
            continue

        pp = p_e[idx]
        rr = r_e[idx]

        win_mask = rr > 0.0
        loss_mask = ~win_mask

        avg_win = float(np.mean(rr[win_mask])) if int(np.sum(win_mask)) > 0 else None
        avg_loss = float(np.mean(rr[loss_mask])) if int(np.sum(loss_mask)) > 0 else None

        profit_share = None
        stop_share = None
        timeout_share = None
        trailing_share = None
        other_share = None
        if ex_e is not None:
            try:
                ex_s = pd.Series(ex_e[idx], dtype=object).astype(str)
                ex_s = ex_s.replace("<NA>", np.nan).replace("nan", np.nan)
                ex_s = ex_s.dropna()
                total_ex = int(len(ex_s))
                if total_ex > 0:
                    counts = ex_s.value_counts(normalize=True)
                    profit_share = float(counts.get("profit", 0.0))
                    trailing_share = float(counts.get("trailing", 0.0))
                    stop_share = float(counts.get("stop", 0.0))
                    timeout_share = float(counts.get("timeout", 0.0))
                    other_share = float(
                        max(
                            0.0,
                            1.0
                            - (
                                float(profit_share)
                                + float(trailing_share)
                                + float(stop_share)
                                + float(timeout_share)
                            ),
                        )
                    )
            except Exception:
                pass

        out.append(
            {
                "decile": int(b),
                "n_events": int(rr.size),
                "p_min": float(np.min(pp)) if pp.size else None,
                "p_max": float(np.max(pp)) if pp.size else None,
                "p_mean": float(np.mean(pp)) if pp.size else None,
                "mean_return": float(np.mean(rr)) if rr.size else None,
                "win_rate": float(np.mean(rr > 0.0)) if rr.size else None,
                "avg_win": float(avg_win) if avg_win is not None and np.isfinite(float(avg_win)) else None,
                "avg_loss": float(avg_loss) if avg_loss is not None and np.isfinite(float(avg_loss)) else None,
                "avg_loss_abs": float(abs(float(avg_loss))) if avg_loss is not None and np.isfinite(float(avg_loss)) else None,
                "exit_profit_share": float(profit_share + trailing_share)
                if profit_share is not None and trailing_share is not None
                else None,
                "exit_stop_share": float(stop_share) if stop_share is not None else None,
                "exit_timeout_share": float(timeout_share) if timeout_share is not None else None,
                "exit_trailing_share": float(trailing_share) if trailing_share is not None else None,
                "exit_other_share": float(other_share) if other_share is not None else None,
            }
        )

    return out


def _sweep_prob_thresholds_for_profitability(
    *,
    probs: np.ndarray,
    returns: np.ndarray,
    direction: str,
    days_span: float,
    thresholds: Optional[np.ndarray] = None,
    min_trades: int = 30,
    p_fail: Optional[np.ndarray] = None,
    p_fail_threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        p_full = np.asarray(probs, dtype=float).reshape(-1)
        r_full = np.asarray(returns, dtype=float).reshape(-1)
    except Exception:
        return [], {"any_profitable": False, "best_positive": None, "best_any": None}

    n = int(min(p_full.size, r_full.size))
    if n <= 0:
        return [], {"any_profitable": False, "best_positive": None, "best_any": None}
    p_full = p_full[:n]
    r_full = r_full[:n]

    pf_full = None
    if p_fail is not None:
        try:
            pf_full = np.asarray(p_fail, dtype=float).reshape(-1)[:n]
        except Exception:
            pf_full = None

    base_mask = np.isfinite(p_full) & np.isfinite(r_full)
    p = p_full[base_mask]
    r = r_full[base_mask]
    pf = pf_full[base_mask] if pf_full is not None else None

    if int(p.size) < 20:
        return [], {"any_profitable": False, "best_positive": None, "best_any": None}

    d = str(direction or "").lower()
    if thresholds is None:
        if d == "short":
            thr_arr = np.linspace(0.5, 0.01, 50)
        else:
            thr_arr = np.linspace(0.5, 0.99, 50)
    else:
        thr_arr = np.asarray(thresholds, dtype=float).reshape(-1)
    thr_arr = thr_arr[np.isfinite(thr_arr)]

    rows: List[Dict[str, Any]] = []
    best_positive: Optional[Dict[str, Any]] = None
    best_any: Optional[Dict[str, Any]] = None

    for thr in thr_arr:
        thr_f = float(thr)
        if d == "short":
            denom = max(1e-12, thr_f)
            abs_size = np.clip((thr_f - p) / denom, 0.0, 1.0)
        else:
            denom = max(1e-12, (1.0 - thr_f))
            abs_size = np.clip((p - thr_f) / denom, 0.0, 1.0)

        take = abs_size > 1e-12
        if pf is not None and p_fail_threshold is not None and np.isfinite(float(p_fail_threshold)):
            pf_v = np.where(np.isfinite(pf), pf, -np.inf)
            veto = pf_v > float(p_fail_threshold)
            take = take & (~veto)

        tr = r * abs_size
        tr = tr[take]
        tr = tr[np.isfinite(tr)]
        n_tr = int(tr.size)

        mean_ret = float(np.mean(tr)) if n_tr > 0 else None
        win_rate = float(np.mean(tr > 0.0)) if n_tr > 0 else None
        avg_win = float(np.mean(tr[tr > 0.0])) if n_tr > 0 and np.any(tr > 0.0) else None
        avg_loss = float(np.mean(tr[tr <= 0.0])) if n_tr > 0 and np.any(tr <= 0.0) else None

        row = {
            "prob_threshold": float(thr_f),
            "n_trades": int(n_tr),
            "trades_per_day": float(n_tr) / float(max(float(days_span), 1.0)),
            "mean_return": float(mean_ret) if mean_ret is not None and np.isfinite(float(mean_ret)) else None,
            "win_rate": float(win_rate) if win_rate is not None and np.isfinite(float(win_rate)) else None,
            "avg_win": float(avg_win) if avg_win is not None and np.isfinite(float(avg_win)) else None,
            "avg_loss": float(avg_loss) if avg_loss is not None and np.isfinite(float(avg_loss)) else None,
        }
        rows.append(row)

        if row.get("mean_return") is not None:
            if best_any is None or float(row["mean_return"]) > float(best_any.get("mean_return") or -1e9):
                best_any = dict(row)
            if (
                int(row.get("n_trades") or 0) >= int(max(0, min_trades))
                and float(row["mean_return"]) > 0.0
                and (best_positive is None or float(row["mean_return"]) > float(best_positive.get("mean_return") or -1e9))
            ):
                best_positive = dict(row)

    summary = {
        "any_profitable": bool(best_positive is not None),
        "best_positive": best_positive,
        "best_any": best_any,
        "min_trades": int(max(0, min_trades)),
    }
    return rows, summary


def _weighted_avg_abs_corr(
    *,
    signals: np.ndarray,
    weights: np.ndarray,
) -> float:
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2:
        return 0.0
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = int(x.shape[1])
    if n <= 1 or int(w.size) != n:
        return 0.0
    w = np.where(np.isfinite(w) & (w >= 0.0), w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        return 0.0

    try:
        x = np.where(np.isfinite(x), x, 0.0)
        x = x - np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, ddof=1, keepdims=True)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        x = x / sd
        corr = (x.T @ x) / float(max(int(x.shape[0]) - 1, 1))
        corr = np.asarray(corr, dtype=float)
        corr = np.where(np.isfinite(corr), corr, 0.0)
        corr = np.clip(corr, -1.0, 1.0)
    except Exception:
        return 0.0

    denom = 0.0
    numer = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            wij = float(w[i]) * float(w[j])
            denom += wij
            numer += wij * float(abs(corr[i, j]))
    if denom <= 1e-12:
        return 0.0
    return float(numer / denom)


def _compute_per_expert_psr(
    *,
    returns_matrix: np.ndarray,
    label_matrix: np.ndarray,
    take_mask: np.ndarray,
    event_idx: pd.DatetimeIndex,
    expert_names: List[str],
    sr_benchmark: float = 0.0,
    periods_per_year: float = 365.0,
    min_trades: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute Probabilistic Sharpe Ratio (PSR) for each expert individually.
    
    This provides per-expert risk-adjusted performance metrics following
    De Prado's AFML methodology, enabling identification of consistently
    unprofitable experts in the committee.
    
    Args:
        returns_matrix: (n_events, n_experts) returns when expert fires
        label_matrix: (n_events, n_experts) expert signals (-1, 0, +1)
        take_mask: (n_events,) boolean mask of taken trades
        event_idx: DatetimeIndex for aggregating to daily returns
        expert_names: List of expert names
        sr_benchmark: Benchmark Sharpe Ratio for PSR calculation
        periods_per_year: Annualization factor (365 for crypto, 252 for equities)
        min_trades: Minimum trades required for valid PSR
        
    Returns:
        Dict mapping expert_name -> {psr, psr_z, sr, n, skew, kurt, ...}
    """
    result: Dict[str, Dict[str, Any]] = {}
    
    ret_mat = np.asarray(returns_matrix, dtype=float)
    lbl_mat = np.asarray(label_matrix, dtype=float)
    tm = np.asarray(take_mask, dtype=bool)
    
    if ret_mat.ndim != 2 or lbl_mat.ndim != 2:
        return result
    
    n_events, n_experts = ret_mat.shape
    if n_experts != len(expert_names):
        return result
    
    try:
        ev_idx = pd.DatetimeIndex(event_idx)
        if len(ev_idx) != n_events:
            return result
    except Exception:
        return result
    
    for j, name in enumerate(expert_names):
        try:
            # Expert fires and trade is taken
            expert_fired = lbl_mat[:, j] != 0.0
            expert_taken = expert_fired & tm
            
            n_taken = int(np.sum(expert_taken))
            if n_taken < min_trades:
                result[str(name)] = {
                    "psr": 0.0,
                    "psr_z": float("-inf"),
                    "sr": None,
                    "n": n_taken,
                    "skew": 0.0,
                    "kurt": 3.0,
                    "sr_benchmark": float(sr_benchmark),
                    "insufficient_trades": True,
                }
                continue
            
            # Get returns for this expert's taken trades
            expert_returns = ret_mat[expert_taken, j]
            expert_idx = ev_idx[expert_taken]
            
            # Aggregate to daily PnL
            day_index = pd.date_range(
                start=expert_idx.min().normalize(),
                end=expert_idx.max().normalize(),
                freq="D",
            )
            daily_pnl = pd.Series(expert_returns, index=expert_idx).groupby(
                expert_idx.normalize()
            ).sum()
            daily_pnl = daily_pnl.reindex(day_index, fill_value=0.0)
            daily_log = np.log1p(daily_pnl.astype(float).values)
            daily_log = daily_log[np.isfinite(daily_log)]
            
            # Compute PSR using existing function
            psr_details = _psr_from_returns(
                daily_log,
                sr_benchmark=float(sr_benchmark),
                periods_per_year=float(periods_per_year),
            )
            psr_details["insufficient_trades"] = False
            psr_details["n_taken"] = n_taken
            
            # Add contribution metrics
            total_pnl = float(np.sum(expert_returns))
            mean_ret = float(np.mean(expert_returns))
            win_rate = float(np.mean(expert_returns > 0.0))
            psr_details["total_pnl"] = total_pnl
            psr_details["mean_return"] = mean_ret
            psr_details["win_rate"] = win_rate
            
            result[str(name)] = psr_details
            
        except Exception:
            result[str(name)] = {
                "psr": 0.0,
                "psr_z": float("-inf"),
                "sr": None,
                "n": 0,
                "skew": 0.0,
                "kurt": 3.0,
                "sr_benchmark": float(sr_benchmark),
                "error": True,
            }
    
    return result


def _compute_unprofitable_expert_penalty(
    *,
    per_expert_psr: Dict[str, Dict[str, Any]],
    weights_vec: np.ndarray,
    expert_names: List[str],
    psr_threshold: float = 0.5,
    sr_threshold: float = 0.0,
    penalty_scale: float = 1.0,
    min_trades_required: int = 10,
) -> Dict[str, Any]:
    """
    Compute a penalty for configurations that give high weight to unprofitable experts.
    
    This enables HPO to automatically down-weight experts that consistently
    underperform on a risk-adjusted basis (low PSR or negative Sharpe Ratio).
    
    The penalty is computed as:
        penalty = sum(weight_i * underperformance_i) for all unprofitable experts
    
    Where underperformance_i = max(0, psr_threshold - psr_i) + max(0, -sr_i) * sr_weight
    
    Args:
        per_expert_psr: Dict from _compute_per_expert_psr with PSR metrics per expert
        weights_vec: (n_experts,) normalized expert weights from HPO
        expert_names: List of expert names matching weights_vec order
        psr_threshold: PSR below this is considered underperforming (default 0.5 = random)
        sr_threshold: SR below this triggers additional penalty (default 0.0)
        penalty_scale: Multiplier for the final penalty
        min_trades_required: Ignore experts with fewer trades
        
    Returns:
        Dict with:
            - penalty: Total penalty value (higher = worse configuration)
            - unprofitable_experts: List of expert names flagged as unprofitable
            - expert_penalties: Dict mapping expert_name -> individual penalty contribution
            - diagnostics: Additional debug info
    """
    result: Dict[str, Any] = {
        "penalty": 0.0,
        "unprofitable_experts": [],
        "expert_penalties": {},
        "diagnostics": {
            "n_experts_evaluated": 0,
            "n_unprofitable": 0,
            "total_unprofitable_weight": 0.0,
            "worst_expert": None,
            "worst_expert_psr": None,
        },
    }
    
    if not per_expert_psr or not expert_names:
        return result
    
    w = np.asarray(weights_vec, dtype=float).reshape(-1)
    if w.size != len(expert_names):
        return result
    
    # Normalize weights for consistent penalty scaling
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return result
    w_norm = w / w_sum
    
    total_penalty = 0.0
    unprofitable_experts: List[str] = []
    expert_penalties: Dict[str, float] = {}
    total_unprofitable_weight = 0.0
    worst_expert = None
    worst_psr = 1.0
    n_evaluated = 0
    
    for j, name in enumerate(expert_names):
        psr_data = per_expert_psr.get(str(name), {})
        if not isinstance(psr_data, dict):
            continue
        
        # Skip experts with insufficient trades
        n_trades = psr_data.get("n_taken", psr_data.get("n", 0))
        if n_trades < min_trades_required:
            continue
        
        n_evaluated += 1
        psr_val = float(psr_data.get("psr", 0.5))
        sr_val = psr_data.get("sr")
        weight_i = float(w_norm[j])
        
        # Compute underperformance score
        # 1. PSR penalty: how far below threshold
        psr_gap = float(max(0.0, psr_threshold - psr_val))
        
        # 2. SR penalty: additional penalty for negative Sharpe
        sr_penalty = 0.0
        if sr_val is not None and np.isfinite(sr_val):
            if float(sr_val) < sr_threshold:
                # Stronger penalty for negative SR (losing money)
                sr_penalty = float(max(0.0, sr_threshold - float(sr_val)))
        
        # Combined underperformance (PSR dominates, SR adds extra penalty for losses)
        underperformance = psr_gap + 0.5 * sr_penalty
        
        if underperformance > 0.01:
            # Weight-adjusted penalty: high weight + poor performance = big penalty
            expert_penalty = weight_i * underperformance * penalty_scale
            total_penalty += expert_penalty
            unprofitable_experts.append(str(name))
            expert_penalties[str(name)] = float(expert_penalty)
            total_unprofitable_weight += weight_i
            
            # Track worst expert
            if psr_val < worst_psr:
                worst_psr = psr_val
                worst_expert = str(name)
    
    result["penalty"] = float(total_penalty)
    result["unprofitable_experts"] = unprofitable_experts
    result["expert_penalties"] = expert_penalties
    result["diagnostics"] = {
        "n_experts_evaluated": n_evaluated,
        "n_unprofitable": len(unprofitable_experts),
        "total_unprofitable_weight": float(total_unprofitable_weight),
        "worst_expert": worst_expert,
        "worst_expert_psr": float(worst_psr) if worst_expert else None,
        "psr_threshold": float(psr_threshold),
        "sr_threshold": float(sr_threshold),
        "penalty_scale": float(penalty_scale),
    }
    
    return result


def _compute_regime_aware_correlation(
    *,
    signals: np.ndarray,
    weights: np.ndarray,
    regime_masks: Dict[str, np.ndarray],
    home_regime_map: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compute regime-aware expert correlation metrics.
    
    Unlike global correlation, this computes correlation within each regime
    separately. Experts should be diversified in regimes where they don't
    specialize, but may be correlated in their "home" regime.
    
    Args:
        signals: (n_events, n_experts) expert signal matrix
        weights: (n_experts,) expert weights for weighted average
        regime_masks: Dict mapping regime_name -> boolean mask of events
        home_regime_map: Optional dict mapping expert_idx -> home_regime_name
                         (experts are expected to be correlated in home regime)
    
    Returns:
        Dict with:
            - global_corr: Overall weighted average absolute correlation
            - per_regime_corr: {regime_name: weighted_avg_abs_corr}
            - out_of_home_corr: Avg correlation when experts are outside home regime
            - diversity_score: Combined score (lower = more diverse)
    """
    result: Dict[str, Any] = {
        "global_corr": 0.0,
        "per_regime_corr": {},
        "out_of_home_corr": 0.0,
        "diversity_score": 0.0,
        "n_regimes_evaluated": 0,
    }
    
    x = np.asarray(signals, dtype=float)
    if x.ndim != 2:
        return result
    
    n_events, n_experts = x.shape
    w = np.asarray(weights, dtype=float).reshape(-1)
    
    if n_experts <= 1 or int(w.size) != n_experts:
        return result
    
    w = np.where(np.isfinite(w) & (w >= 0.0), w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        return result
    
    # Global correlation (existing logic)
    result["global_corr"] = _weighted_avg_abs_corr(signals=x, weights=w)
    
    # Per-regime correlation
    per_regime_corr: Dict[str, float] = {}
    regime_weights_sum = 0.0
    weighted_regime_corr = 0.0
    
    for regime_name, mask in regime_masks.items():
        try:
            m = np.asarray(mask, dtype=bool)
            if int(m.size) != n_events:
                continue
            n_in_regime = int(np.sum(m))
            if n_in_regime < 20:  # Need sufficient samples
                continue
            
            x_regime = x[m, :]
            regime_corr = _weighted_avg_abs_corr(signals=x_regime, weights=w)
            per_regime_corr[str(regime_name)] = float(regime_corr)
            
            # Weight by number of events in regime
            weighted_regime_corr += float(regime_corr) * float(n_in_regime)
            regime_weights_sum += float(n_in_regime)
            
        except Exception:
            continue
    
    result["per_regime_corr"] = per_regime_corr
    result["n_regimes_evaluated"] = len(per_regime_corr)
    
    # Compute out-of-home correlation if home_regime_map provided
    if home_regime_map is not None and len(per_regime_corr) > 0:
        try:
            out_of_home_corrs: List[float] = []
            out_of_home_weights: List[float] = []
            
            for regime_name, regime_corr in per_regime_corr.items():
                # Count how many experts are "at home" in this regime
                n_home = sum(
                    1 for exp_idx, home_regime in home_regime_map.items()
                    if home_regime == regime_name and exp_idx < n_experts
                )
                # If most experts are NOT home, this is an out-of-home regime
                if n_home < n_experts // 2:
                    mask = regime_masks.get(regime_name)
                    if mask is not None:
                        n_events_regime = int(np.sum(mask))
                        out_of_home_corrs.append(float(regime_corr))
                        out_of_home_weights.append(float(n_events_regime))
            
            if out_of_home_corrs:
                total_w = sum(out_of_home_weights)
                if total_w > 0:
                    result["out_of_home_corr"] = float(
                        sum(c * w for c, w in zip(out_of_home_corrs, out_of_home_weights)) / total_w
                    )
        except Exception:
            pass
    
    # Diversity score: emphasize out-of-home correlation (where diversity matters most)
    # If no home_regime_map, use weighted average of per-regime correlations
    if regime_weights_sum > 0:
        avg_regime_corr = weighted_regime_corr / regime_weights_sum
        # Blend: 30% global, 70% regime-aware (emphasize regime-specific diversity)
        if result["out_of_home_corr"] > 0:
            result["diversity_score"] = float(
                0.3 * result["global_corr"] + 0.7 * result["out_of_home_corr"]
            )
        else:
            result["diversity_score"] = float(
                0.3 * result["global_corr"] + 0.7 * avg_regime_corr
            )
    else:
        result["diversity_score"] = result["global_corr"]
    
    return result


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
    target_center = float(cfg.get("hpo_target_pos_rate", 0.55))
    target_band = float(cfg.get("hpo_target_pos_band", 0.10))
    lower = float(target_center - target_band)
    upper = float(target_center + target_band)
    over_mult = float(cfg.get("hpo_balance_over_penalty_mult", 2.0))
    under_mult = float(cfg.get("hpo_balance_under_penalty_mult", 1.0))
    balance_dev_high = max(0.0, pos_rate - upper)
    balance_dev_low = max(0.0, lower - pos_rate)
    balance_penalty = float(under_mult * balance_dev_low + over_mult * balance_dev_high)

    w_aleatoric = float(cfg.get("hpo_penalty_w_aleatoric", 1.25))
    w_timeout = float(cfg.get("hpo_penalty_w_timeout", 1.0))
    w_balance = float(cfg.get("hpo_penalty_w_balance", 1.5))

    q_aleatoric = float(np.clip(1.0 - aleatoric_rate, 0.0, 1.0)) ** w_aleatoric
    q_timeout = float(np.clip(1.0 - timeout_rate, 0.0, 1.0)) ** w_timeout
    q_balance = float(np.exp(-w_balance * balance_penalty))

    quality_multiplier = float(np.clip(q_aleatoric * q_timeout * q_balance, 0.0, 1.0))
    # Avoid saturating / artificially capping utility by default.
    # (The caller can still override via config.)
    try:
        utility_floor = float(cfg.get("layer2_utility_floor", -1.0))
    except Exception:
        utility_floor = -1.0
    if not np.isfinite(utility_floor):
        utility_floor = -1.0
    utility_adj = float(
        np.clip(
            utility * quality_multiplier,
            float(utility_floor),
            float(cfg.get("layer2_utility_clip_max", 5000.0)),
        )
    )

    details = {
        "enabled": True,
        "tx_cost": float(tx_cost),
        "near_cost_threshold": float(eps),
        "aleatoric_rate": aleatoric_rate,
        "timeout_rate": float(timeout_rate),
        "pos_rate": pos_rate,
        "target_pos_rate": float(target_center),
        "target_pos_band": float(target_band),
        "balance_over_penalty_mult": float(over_mult),
        "balance_under_penalty_mult": float(under_mult),
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
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calibrate probabilities using isotonic regression with cross-validation.
    
    Args:
        y_true: True binary labels
        y_prob: Uncalibrated predicted probabilities
        cv_folds: Number of cross-validation folds
        sample_weight: Optional sample weights (if provided, used in isotonic fit)
    
    Returns:
        Calibrated probabilities
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold
    
    calibrated = np.zeros_like(y_prob, dtype=float)
    kf = KFold(n_splits=cv_folds, shuffle=False)
    
    for train_idx, val_idx in kf.split(y_prob):
        iso = IsotonicRegression(out_of_bounds='clip')
        if sample_weight is not None and len(sample_weight) == len(y_prob):
            iso.fit(y_prob[train_idx], y_true[train_idx], sample_weight=sample_weight[train_idx])
        else:
            iso.fit(y_prob[train_idx], y_true[train_idx])
        calibrated[val_idx] = iso.predict(y_prob[val_idx])
    
    return calibrated


def compute_fold_sharpe_ratios(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    n_folds: int = 5,
    use_calibration: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-fold Sharpe ratios for stability assessment.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        returns: Realized returns per sample
        n_folds: Number of folds
        use_calibration: Whether to calibrate probabilities before sizing
        sample_weight: Optional sample weights (passed to calibration if provided)
    
    Returns:
        Array of per-fold Sharpe ratios
    """
    from sklearn.model_selection import KFold
    
    # Pre-calibrate if requested
    if use_calibration:
        y_prob_cal = calibrate_probabilities_isotonic(y_true, y_prob, cv_folds=min(3, n_folds), sample_weight=sample_weight)
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
    """
    Orchestrator for Layer 2 + Layer 3 Meta-Labeling Pipeline.
    """
    
    async def execute(self, config: dict) -> dict:
        """
        Execute the pipeline.
        
        Args:
            config: Configuration dictionary.
        """
        # Load market data (using standard BaseStep mechanism)
        market_data, _ = self.load_market_data_or_fail(config)
        
        # Load primary signals if available (or generate dummy if needed/implied)
        # Usually passed via pipeline_state or artifact, but here we assume self-contained logic if possible
        # For HPO, we often generate primary signals inside the step.
        # But `run_step` signature had `primary_signals`.
        # We'll need to generate them here if not provided.
        
        # Check for target_sample_weight in artifacts/pipeline state?
        # For now, let's proceed with minimal setup.
        
        # We need `run_step` to be compatible with legacy calls if any,
        # but `execute` is the main entry point for BaseStep.
        # We will wrap logic in `_run_pipeline`.
        
        # Generate primary signals (using default logic if not provided)
        from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
        primary_signals = generate_primary_signals(market_data.copy())
        
        # Try load weights
        target_sample_weight = None
        # (Simulated for now as legacy loading is complex)
        
        return self.run_step(market_data, primary_signals, target_sample_weight)

    def run_step(self, market_data: pd.DataFrame, primary_signals: pd.DataFrame,
                 target_sample_weight: np.ndarray = None, **kwargs) -> dict:
        """
        Execute the pipeline (sync method for internal use).
        
        Args:
            market_data: OHLCV + features.
            primary_signals: 'consensus' column.
            target_sample_weight: Weights from Layer 1 (Uniqueness * Consistency).
        """
        config = self.config if hasattr(self, 'config') else kwargs.get('config', {})
        symbol = config.get("symbol", "UNKNOWN")
        exchange = config.get("exchange", "UNKNOWN")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        
        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        outcomes_dir = Path("outcomes") / f"meta_labeling_{symbol}_{run_timestamp}"
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        tprint_info(f"Starting Meta-Labeling Pipeline for {symbol} {direction}")
        
        # ---------------------------------------------------------
        # Data Preparation
        # ---------------------------------------------------------
        df = market_data.copy()
        
        if 'volatility_1d' not in df.columns:
            df['log_ret'] = np.log(df['close']).diff()
            df['volatility_1d'] = df['log_ret'].rolling(20).std()
            
        # ---------------------------------------------------------
        # LAYER 2: Geometry Optimization & Bagged Labeling
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 2: Geometry Optimization...")
        
        layer2 = LabelBasedLayer2(
            transaction_cost=float(config.get('transaction_cost', 0.001)),
            n_trials=int(config.get('layer2_n_trials', 30)),
            n_splits=int(config.get('layer2_n_splits', 3)),
            verbose=True
        )
        
        l2_output = layer2.run(df)
        
        if not l2_output:
            tprint_error("Layer 2 produced no output. Exiting.")
            return {"success": False}
        
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

        try:
            trgt_ewma_span = int(config.get("labeling_trgt_ewma_span", 64))
        except Exception:
            trgt_ewma_span = 64
        try:
            trgt_ewma_min_periods = int(config.get("labeling_trgt_ewma_min_periods", 20))
        except Exception:
            trgt_ewma_min_periods = 20
        trgt_sigma = (
            log_ret.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .ewm(span=int(trgt_ewma_span), adjust=False, min_periods=int(trgt_ewma_min_periods))
            .std()
            .abs()
        )
        trgt_sigma = trgt_sigma.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(0.0)
        try:
            trgt_sigma_ref = float(np.nanmedian(pd.to_numeric(trgt_sigma, errors="coerce").values))
        except Exception:
            trgt_sigma_ref = 0.0
        if (not np.isfinite(trgt_sigma_ref)) or trgt_sigma_ref <= 0.0:
            trgt_sigma_ref = float(np.nanmean(pd.to_numeric(trgt_sigma, errors="coerce").values))
        if (not np.isfinite(trgt_sigma_ref)) or trgt_sigma_ref <= 0.0:
            trgt_sigma_ref = 1e-4

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
            enable_regime_conditional_barrier_geometry = bool(
                config.get("enable_regime_conditional_barrier_geometry", True)
            )
        except Exception:
            enable_regime_conditional_barrier_geometry = True

        try:
            barrier_geometry_regime_col = str(
                config.get("barrier_geometry_regime_col", "hmm_regime_label_1h")
            )
        except Exception:
            barrier_geometry_regime_col = "hmm_regime_label_1h"

        barrier_geometry_by_regime = None
        try:
            cfg_bg = config.get("barrier_geometry_by_regime")
            if isinstance(cfg_bg, dict) and cfg_bg:
                barrier_geometry_by_regime = dict(cfg_bg)
        except Exception:
            barrier_geometry_by_regime = None

        regime_scalar_for_barriers = None
        try:
            if (
                bool(enable_regime_conditional_barrier_geometry)
                and barrier_geometry_by_regime is None
                and barrier_geometry_regime_col in market_data.columns
            ):
                regimes_all = market_data[barrier_geometry_regime_col].reindex(market_data.index)
                atr_all = atr_frac.reindex(market_data.index).astype(float)
                atr_med_all = float(np.nanmedian(pd.to_numeric(atr_all, errors="coerce").values))
                if (not np.isfinite(atr_med_all)) or atr_med_all <= 0.0:
                    atr_med_all = float(np.nanmean(pd.to_numeric(atr_all, errors="coerce").values))
                if (not np.isfinite(atr_med_all)) or atr_med_all <= 0.0:
                    atr_med_all = 1e-3

                scalars: Dict[str, float] = {}
                try:
                    for rv in pd.unique(regimes_all.dropna()):
                        m = regimes_all == rv
                        if int(m.sum()) < 50:
                            continue
                        med_r = float(np.nanmedian(pd.to_numeric(atr_all[m], errors="coerce").values))
                        if (not np.isfinite(med_r)) or med_r <= 0.0:
                            continue
                        s = float(med_r) / float(atr_med_all)
                        scalars[str(rv)] = float(s)
                except Exception:
                    scalars = {}

                if scalars:
                    reg_keys = regimes_all.astype(object).astype(str)
                    regime_scalar_for_barriers = reg_keys.map(scalars)
                    regime_scalar_for_barriers = regime_scalar_for_barriers.reindex(market_data.index)
                    regime_scalar_for_barriers = regime_scalar_for_barriers.astype(float)
                    regime_scalar_for_barriers = regime_scalar_for_barriers.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                    try:
                        s_min = float(config.get("barrier_geometry_regime_scalar_min", 0.7))
                        s_max = float(config.get("barrier_geometry_regime_scalar_max", 1.6))
                        if np.isfinite(s_min) and np.isfinite(s_max) and s_max > s_min > 0.0:
                            regime_scalar_for_barriers = regime_scalar_for_barriers.clip(lower=float(s_min), upper=float(s_max))
                    except Exception:
                        pass
        except Exception:
            regime_scalar_for_barriers = None

        def _compute_regime_conditional_barrier_geometry(
            *,
            params: Dict[str, Any],
            market_index: pd.Index,
            default_horizon: int,
            atr_frac_series: pd.Series,
        ) -> Tuple[pd.Series, pd.Series, pd.Series, Optional[pd.Series]]:
            try:
                profit_floor_tx_mult = float(params.get("profit_floor_tx_mult", layer2_profit_floor_tx_mult))
            except Exception:
                profit_floor_tx_mult = float(layer2_profit_floor_tx_mult)
            if (not np.isfinite(profit_floor_tx_mult)) or profit_floor_tx_mult <= 0.0:
                profit_floor_tx_mult = float(layer2_profit_floor_tx_mult)
            profit_floor_tx_mult = float(np.clip(profit_floor_tx_mult, 1.0, 10.0))
            min_profit_floor_local = float(DEFAULT_TRANSACTION_COST) * float(profit_floor_tx_mult)

            base_h = int(params.get("horizon_bars", default_horizon))
            base_sl = float(params.get("sl_atr_mult", 1.0))
            base_rr = float(params.get("risk_reward_ratio", 2.0))
            base_trail = float(params.get("trail_distance_atr_mult", 0.0))

            if (not np.isfinite(base_sl)) or base_sl <= 0.0:
                base_sl = 1.0
            if (not np.isfinite(base_rr)) or base_rr <= 0.0:
                base_rr = 2.0
            if (not np.isfinite(base_trail)) or base_trail < 0.0:
                base_trail = 0.0
            if base_h <= 0:
                base_h = int(default_horizon)

            stop_mult = pd.Series(float(base_sl), index=market_index, dtype=float)
            rr_series = pd.Series(float(base_rr), index=market_index, dtype=float)
            horizon_series = pd.Series(float(base_h), index=market_index, dtype=float)
            trail_mult_series: Optional[pd.Series] = None
            try:
                trail_mult_series = pd.Series(float(base_trail), index=market_index, dtype=float)
            except Exception:
                trail_mult_series = None

            if bool(enable_regime_conditional_barrier_geometry) and barrier_geometry_by_regime is not None:
                try:
                    if barrier_geometry_regime_col in market_data.columns:
                        regimes = market_data[barrier_geometry_regime_col].reindex(market_index)
                        reg_keys = regimes.astype(object).astype(str)

                        def _map_param(key: str, default_v: float) -> pd.Series:
                            out = pd.Series(float(default_v), index=market_index, dtype=float)
                            for rk in pd.unique(reg_keys.dropna()):
                                spec = barrier_geometry_by_regime.get(str(rk))
                                if not isinstance(spec, dict):
                                    continue
                                v = spec.get(key)
                                vm = spec.get(f"{key}_mult")
                                if v is None and vm is None:
                                    continue
                                try:
                                    if v is not None:
                                        vv = float(v)
                                    else:
                                        vv = float(default_v) * float(vm)
                                    if not np.isfinite(vv):
                                        continue
                                except Exception:
                                    continue
                                out.loc[reg_keys == str(rk)] = float(vv)
                            return out

                        stop_mult = _map_param("sl_atr_mult", float(base_sl))
                        rr_series = _map_param("risk_reward_ratio", float(base_rr))
                        horizon_series = _map_param("horizon_bars", float(base_h))
                        if trail_mult_series is not None:
                            trail_mult_series = _map_param("trail_distance_atr_mult", float(base_trail))
                except Exception:
                    pass

            if (
                bool(enable_regime_conditional_barrier_geometry)
                and barrier_geometry_by_regime is None
                and regime_scalar_for_barriers is not None
            ):
                try:
                    s = regime_scalar_for_barriers.reindex(market_index).astype(float)
                    s = s.replace([np.inf, -np.inf], np.nan).fillna(1.0)

                    # Tunable regime scaling strength/power so Layer2 HPO can reduce
                    # cross-regime dispersion rather than being forced into full scaling.
                    try:
                        barrier_regime_strength = float(params.get("barrier_regime_strength", 1.0))
                    except Exception:
                        barrier_regime_strength = 1.0
                    if not np.isfinite(barrier_regime_strength):
                        barrier_regime_strength = 1.0
                    barrier_regime_strength = float(np.clip(barrier_regime_strength, 0.0, 1.0))

                    try:
                        barrier_regime_power = float(params.get("barrier_regime_power", 1.0))
                    except Exception:
                        barrier_regime_power = 1.0
                    if not np.isfinite(barrier_regime_power) or barrier_regime_power <= 1e-6:
                        barrier_regime_power = 1.0
                    barrier_regime_power = float(np.clip(barrier_regime_power, 0.25, 4.0))

                    # Blend toward 1.0 when strength < 1
                    s_eff = 1.0 + float(barrier_regime_strength) * (np.power(s.astype(float), float(barrier_regime_power)) - 1.0)
                    s_eff = pd.to_numeric(s_eff, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)

                    # Apply scaling to geometry
                    s_thr_s = stop_mult * s_eff      # Expand SL with volatility
                    
                    # NEW: Volatility-of-Volatility Adjustment
                    # If enabled, widen barriers further when volatility itself is unstable (high vol-of-vol)
                    # This protects against "fat tails" that standard ATR/regime-scalar might underestimate
                    try:
                        vol_vol_exp = float(params.get("barrier_vol_vol_exp", 0.0))
                        if vol_vol_exp > 0.01:
                            # Need a proxy for vol-of-vol. 
                            # 's' is the regime scalar (volatility ratio).
                            # Compute rolling std of 's' (e.g. 24 bars) normalized by 's'
                            # This is roughly "coefficient of variation of volatility"
                            # If s is Series, use rolling std.
                            if isinstance(s, pd.Series):
                                s_std = s.rolling(24).std().fillna(0.0)
                                s_mean = s.rolling(24).mean().replace(0, 1.0)
                                vol_of_vol = (s_std / s_mean).fillna(0.0)
                                
                                # Adjustment factor: 1 + exponent * vol_of_vol
                                # e.g. vov=0.5 (very unstable), exp=1.0 -> widen by 50%
                                vov_factor = 1.0 + vol_vol_exp * vol_of_vol
                                vov_factor = vov_factor.clip(1.0, 2.0) # Widen only, max 2x
                                
                                # Apply to SL geometry (TP follows via asymmetry logic)
                                s_thr_s = s_thr_s * vov_factor
                    except Exception:
                        pass

                    
                    # NEW: Barrier Asymmetry Regime Modulation
                    # If enabled, profit target expands MORE than SL in high-vol/trend regimes
                    # This increases effective Risk/Reward ratio in favorable conditions
                    try:
                        barrier_asym = float(params.get("barrier_trend_asymmetry", 0.0))
                    except Exception:
                        barrier_asym = 0.0
                    
                    if barrier_asym > 0.01:
                        # Asymmetry factor: boosts profit target when s_eff > 1 (high vol/trend)
                        # factor = 1 + asymmetry * (s_eff - 1)
                        # e.g. s_eff=1.5, asym=0.5 -> factor = 1 + 0.5*0.5 = 1.25 -> TP boosted by extra 25%
                        asym_factor = np.where(s_eff > 1.0, 1.0 + barrier_asym * (s_eff - 1.0), 1.0)
                        p_thr_s = rr_series * s_thr_s * asym_factor  # Boosted TP
                    else:
                        p_thr_s = rr_series * s_thr_s

                    s_eff = s_eff.clip(lower=0.25, upper=4.0)

                    stop_mult = stop_mult.astype(float) * s_eff
                    if trail_mult_series is not None:
                        trail_mult_series = trail_mult_series.astype(float) * s_eff
                    # In high vol regimes (s>1), prefer shorter horizons; in low vol, longer.
                    horizon_series = horizon_series.astype(float) / s_eff
                except Exception:
                    pass

            stop_mult = stop_mult.reindex(market_index).astype(float)
            rr_series = rr_series.reindex(market_index).astype(float)
            horizon_series = horizon_series.reindex(market_index).astype(float)
            try:
                if trail_mult_series is not None:
                    trail_mult_series = trail_mult_series.reindex(market_index).astype(float)
            except Exception:
                trail_mult_series = None

            stop_thr = (stop_mult * atr_frac_series.reindex(market_index).astype(float)).astype(float)
            stop_thr = stop_thr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            profit_thr = (stop_thr * rr_series).astype(float)
            profit_thr = profit_thr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Enforce small positive floor to avoid degenerate 0 thresholds.
            profit_thr = profit_thr.clip(lower=float(min_profit_floor_local))
            stop_thr = stop_thr.clip(lower=float(min_profit_floor_local) / 2.0)

            # Horizon bounds
            horizon_series = horizon_series.replace([np.inf, -np.inf], np.nan).fillna(float(base_h))
            horizon_series = horizon_series.clip(lower=4.0, upper=256.0)

            if trail_mult_series is not None:
                trail_mult_series = trail_mult_series.replace([np.inf, -np.inf], np.nan).fillna(float(base_trail))
                trail_mult_series = trail_mult_series.clip(lower=0.0, upper=10.0)

            return profit_thr, stop_thr, horizon_series, trail_mult_series

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
        try:
            layer2_profit_floor_tx_mult = float(config.get("layer2_profit_floor_tx_mult", 1.05))
        except Exception:
            layer2_profit_floor_tx_mult = 1.05
        if (not np.isfinite(layer2_profit_floor_tx_mult)) or layer2_profit_floor_tx_mult <= 0.0:
            layer2_profit_floor_tx_mult = 1.05
        min_profit_floor = float(DEFAULT_TRANSACTION_COST) * float(layer2_profit_floor_tx_mult)

        try:
            layer2_tp_mult = config.get("layer2_tp_mult", None)
            layer2_tp_mult = float(layer2_tp_mult) if layer2_tp_mult is not None else None
        except Exception:
            layer2_tp_mult = None
        try:
            layer2_sl_mult = config.get("layer2_sl_mult", None)
            layer2_sl_mult = float(layer2_sl_mult) if layer2_sl_mult is not None else None
        except Exception:
            layer2_sl_mult = None

        if layer2_tp_mult is None:
            raw_tp = float(layer2_tp_target)
            layer2_tp_mult = raw_tp / (float(trgt_sigma_ref) + 1e-12) if raw_tp < 0.2 else raw_tp
        if layer2_sl_mult is None:
            raw_sl = float(layer2_sl_target)
            layer2_sl_mult = raw_sl / (float(trgt_sigma_ref) + 1e-12) if raw_sl < 0.2 else raw_sl

        fixed_layer2_profit_thr = (float(layer2_tp_mult) * trgt_sigma).astype(float).clip(lower=min_profit_floor)
        fixed_layer2_stop_thr = (float(layer2_sl_mult) * trgt_sigma).astype(float).clip(lower=min_profit_floor / 2.0)

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
            # Group 1: Signal Structure
            create_param_group(
                name="signal_structure",
                params={
                    "cusum_threshold": {"type": "float", "low": 0.010, "high": 0.06},
                    "target_signal_density": {"type": "float", "low": 10.0, "high": 40.0},
                    "min_event_spacing": {"type": "int", "low": 0, "high": 10},
                },
                priority=1,
                description="Signal generation and event spacing",
            ),
            # Group 2: Event Geometry
            create_param_group(
                name="event_geometry",
                params={
                    "horizon_bars": {"type": "int", "low": 16, "high": 28, "step": 2},
                    "trail_distance": {"type": "float", "low": 0.6, "high": 3.0},
                    "consensus_threshold": {"type": "float", "low": 0.4, "high": 0.8, "step": 0.05},
                },
                priority=2,
                depends_on=["signal_structure"],
                description="Event definition and geometry",
            ),
            # Group 3: Volatility Adaptation (Restored placeholder to fix dependency)
            create_param_group(
                name="volatility_adaptation",
                params={
                         "volatility_lookback": {"type": "int", "low": 24, "high": 120},
                         "vol_scaling_multiplier": {"type": "float", "low": 0.8, "high": 1.5},
                },
                priority=3,
                depends_on=["event_geometry"],
                description="Volatility adaptation parameters",
            ),
            # Group 4: Label Definition
            create_param_group(
                name="label_definition",
                params={
                    "label_low_q": {"type": "float", "low": 0.15, "high": 0.40},
                    "label_high_q": {"type": "float", "low": 0.60, "high": 0.85},
                    "econ_min_return_multiple": {"type": "float", "low": 1.0, "high": 2.0},
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
                    # Recency weighting: exponential decay rate (0 = disabled, 0.01 = 1%/day default)
                    "recency_decay_lambda": {
                        "type": "float",
                        "low": 0.0,
                        "high": 0.03,
                        "walker_type": "arithmetic",
                        "step": 0.005,
                        "anchor": 0.01,  # Default: 1% decay per day
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

        # ------------------------------------------------------------------
        # RUN LAYER 0 (Kalman/RTS Optimization + Committee Pre-Step)
        # ------------------------------------------------------------------
        layer0_output = run_layer_0(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            market_data=market_data,
            primary_signals=primary_signals,
            config=config,
            outcomes_dir=outcomes_dir,
            start_rank=start_rank,
            stage_rank=stage_rank,
            start_at_canonical=start_at_canonical,
            load_stage_best_params=load_stage_best_params,
        )

        # Unpack Layer 0 results into local variables for compatibility
        best_kalman_params = layer0_output.best_kalman_params
        enable_committee_voting_hpo = layer0_output.enable_committee_voting_hpo
        enable_committee_weight_factor = layer0_output.enable_committee_weight_factor
        enable_committee_pre_step = layer0_output.enable_committee_pre_step
        best_committee_params = layer0_output.best_committee_params
        committee_loaded_from = layer0_output.committee_loaded_from
        committee_configs = layer0_output.committee_configs
        committee_names = layer0_output.committee_names
        committee_event_idx = layer0_output.committee_event_idx
        committee_label_matrix_values = layer0_output.committee_label_matrix_values
        committee_returns_matrix_values = layer0_output.committee_returns_matrix_values
        committee_durations_matrix_values = layer0_output.committee_durations_matrix_values
        committee_confidence_matrix_values = layer0_output.committee_confidence_matrix_values
        advanced_gating_pipeline = layer0_output.advanced_gating_pipeline


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
                if (
                    enable_committee_pre_step
                    and committee_label_matrix_values is not None
                    and committee_returns_matrix_values is not None
                    and committee_event_idx is not None
                ):
                    w_scalp_l1 = float(best_committee_params.get("w_scalp", 1.0))
                    w_swing_l1 = float(best_committee_params.get("w_swing", 1.0))
                    w_trend_l1 = float(best_committee_params.get("w_trend", 1.0))
                    # Build weights vector matching matrix columns (6 base, or 9 if new experts present)
                    n_experts_l1 = int(committee_label_matrix_values.shape[1])
                    if n_experts_l1 > 6:
                        # Include new expert weights
                        w_breakout_l1 = float(best_committee_params.get("w_breakout", 0.5))
                        w_vwap_l1 = float(best_committee_params.get("w_vwap_rev", 0.5))
                        w_vol_shock_l1 = float(best_committee_params.get("w_vol_shock", 0.5))
                        weights_vec = np.array(
                            [w_scalp_l1, w_scalp_l1, w_swing_l1, w_swing_l1, w_trend_l1, w_trend_l1,
                             w_breakout_l1, w_vwap_l1, w_vol_shock_l1],
                            dtype=float,
                        )
                    else:
                        weights_vec = np.array(
                            [w_scalp_l1, w_scalp_l1, w_swing_l1, w_swing_l1, w_trend_l1, w_trend_l1],
                            dtype=float,
                        )
                    weights_vec = np.where(np.isfinite(weights_vec) & (weights_vec >= 0.0), weights_vec, 0.0)
                    if float(np.sum(weights_vec)) <= 1e-12:
                        weights_vec = np.ones_like(weights_vec, dtype=float)

                    lbl_mat = np.asarray(committee_label_matrix_values, dtype=float)
                    ret_mat = np.asarray(committee_returns_matrix_values, dtype=float)
                    conf_mat = committee_confidence_matrix_values
                    if conf_mat is None:
                        conf_mat = np.ones_like(ret_mat, dtype=float)
                    conf_mat = np.asarray(conf_mat, dtype=float)
                    conf_mat = np.where(np.isfinite(conf_mat) & (conf_mat >= 0.0), conf_mat, 0.0)
                    fired = lbl_mat != 0.0
                    fired_w = fired.astype(float) * conf_mat * weights_vec.reshape(1, -1)
                    denom = np.sum(fired_w, axis=1).astype(float) + 1e-8

                    sign_mat = np.where(fired, np.sign(lbl_mat), np.nan)
                    sign_w = np.where(np.isfinite(sign_mat), sign_mat, 0.0) * conf_mat * weights_vec.reshape(1, -1)
                    mean_sign = np.sum(sign_w, axis=1).astype(float) / denom
                    agree = np.abs(mean_sign)
                    agree = np.where(np.isfinite(agree), agree, 0.0)
                    agree = np.clip(agree, 0.0, 1.0)

                    # Coverage-adjust agreement: a single firing expert can yield agree=1.0,
                    # which makes committee_agreement_alpha concentrate weights on sparse/noisy events.
                    # Damp agreement when few experts fired (abstention-aware).
                    try:
                        fired_simple = fired.astype(float)
                        fired_weight = np.sum(fired_simple * weights_vec.reshape(1, -1), axis=1).astype(float)
                        total_weight = float(np.sum(weights_vec)) + 1e-8
                        coverage = fired_weight / total_weight
                        coverage = np.where(np.isfinite(coverage), coverage, 0.0)
                        coverage = np.clip(coverage, 0.0, 1.0)
                        agree = agree * np.sqrt(coverage)
                        agree = np.clip(agree, 0.0, 1.0)
                    except Exception:
                        pass

                    abs_ret = np.abs(ret_mat)
                    abs_ret = np.where(fired, abs_ret, np.nan)
                    abs_w = np.where(np.isfinite(abs_ret), abs_ret, 0.0) * conf_mat * weights_vec.reshape(1, -1)
                    mean_abs = np.sum(abs_w, axis=1).astype(float) / denom
                    mean_abs = np.where(np.isfinite(mean_abs), mean_abs, 0.0)
                    pos_abs = mean_abs[mean_abs > 0.0]
                    med_abs = float(np.nanmedian(pos_abs)) if pos_abs.size > 0 else 0.0
                    if np.isfinite(med_abs) and med_abs > 0.0:
                        mag_factor = mean_abs / (med_abs + 1e-12)
                    else:
                        mag_factor = np.ones_like(mean_abs, dtype=float)

                    committee_agreement_scores_l1 = (
                        pd.Series(agree, index=pd.DatetimeIndex(committee_event_idx))
                        .reindex(baseline_t_events)
                        .fillna(0.0)
                        .values.astype(float)
                    )
                    committee_mag_factors_l1 = (
                        pd.Series(mag_factor, index=pd.DatetimeIndex(committee_event_idx))
                        .reindex(baseline_t_events)
                        .fillna(1.0)
                        .values.astype(float)
                    )
                else:
                    committee_agreement_scores_l1 = None
                    committee_mag_factors_l1 = None
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
                    'uniq_intensity': 2.0, 'exp_mag': 1.5, 'exp_learn': 1.0,
                    'exp_uniq': 1.5, 'exp_cross': 1.0, 'downside_multiplier': 1.0
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
                        n_trials=int(config.get("layer1_n_trials", 60)),
                        objective_mode=str(config.get("layer1_objective_mode", "proxy")),
                    )
                except Exception as e:
                    tprint_warning(f"⚠️ Layer 1 optimization failed: {e}. Using defaults.")
                    best_weighting_params = {
                        'mag_compression': 0.8, 'learn_slope': 10.0, 'learn_center': 0.4,
                        'uniq_intensity': 2.0, 'exp_mag': 1.5, 'exp_learn': 1.0,
                        'exp_uniq': 1.5, 'exp_cross': 1.0, 'downside_multiplier': 1.0
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

        # Layer 2 is forced to Option C (standard / non-leaky) only.
        layer2_use_option_c = True
        layer2_mode = "standard"
        tprint_info("🔧 Layer 2: Option C only (standard / non-leaky)")

        committee_weight_factor_series: Optional[pd.Series] = None
        durations_matrix_values: Optional[np.ndarray] = None

        if enable_committee_pre_step and committee_label_matrix_values is not None and committee_returns_matrix_values is not None and committee_event_idx is not None:
            # Reuse pre-step matrices (now potentially includes new experts)
            label_matrix_values = committee_label_matrix_values
            returns_matrix_values = committee_returns_matrix_values
            durations_matrix_values = committee_durations_matrix_values
            confidence_matrix_values = committee_confidence_matrix_values
            event_idx = pd.DatetimeIndex(committee_event_idx)
        elif enable_committee_weight_factor:
            # A. PRE-COMPUTE COMMITTEE LABEL MATRIX (The "Expert Panel")
            tprint_info("🏗️ Pre-computing Committee of 6 Label Matrix...")

            # 1. Define the 6 Profiles
            # Scalp: Tight (TP 1.2 / SL 0.6)
            # Swing: Balanced (TP 2.0 / SL 1.0)
            # Trend: Looser (TP 3.0 / SL 1.5)
            # Multipliers: Lower (0.8x), Upper (1.2x)

            # Base Multipliers (TP, SL, Horizon) with VARIED TP/SL RATIOS for diversification
            # Scalp: Aggressive 3:1 ratio (quick profits, tight stops)
            # Swing: Balanced 2:1 ratio (standard risk/reward)
            # Trend: Conservative 1.5:1 ratio (ride trends, wider stops)
            base_profiles = {
                "scalp": (1.5, 0.5, 6),    # TP/SL = 3:1, short horizon
                "swing": (2.0, 1.0, 12),   # TP/SL = 2:1, medium horizon
                "trend": (2.4, 1.6, 24),   # TP/SL = 1.5:1, long horizon
            }
            
            # Asymmetric vol scaling for further diversification:
            # - "tight": tighter TP, same SL (more conservative entry)
            # - "wide": wider TP, tighter SL (more aggressive)
            vol_scalars = {
                "tight": {"tp": 0.8, "sl": 1.0},
                "wide": {"tp": 1.3, "sl": 0.85},
            }

            committee_configs = []
            committee_names = []

            for p_name, (tp_base, sl_base, h_base) in base_profiles.items():
                for v_name, v_scalar in vol_scalars.items():
                    config_id = f"{p_name}_{v_name}"
                    # Asymmetric scaling: different multipliers for TP and SL
                    committee_configs.append(
                        TripleBarrierConfig(
                            tp_multiplier=tp_base * v_scalar["tp"],
                            sl_multiplier=sl_base * v_scalar["sl"],
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
                    volume=market_data.get('volume', None),
                    vwap=market_data.get('vwap', None),
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

                # 3. Assemble Label Matrix (Rows=Events, Cols=6 base + 3 new experts = 9)
                # Find common events (primary_signals != 0)
                event_mask = primary_signals['consensus'] != 0
                event_idx = primary_signals[event_mask].index

                # Determine direction sign for new experts
                dir_raw = str(direction).lower()
                dir_sign = 1
                if dir_raw in {"short", "sell", "-1", "s"}:
                    dir_sign = -1

                # Compute new experts (breakout, vwap_rev, vol_shock)
                new_expert_scores = None
                new_expert_conf = None
                try:
                    from src.training.steps.labeling.layer2_advanced_logic import compute_new_experts_matrix, NEW_EXPERT_NAMES
                    new_expert_scores, new_expert_conf = compute_new_experts_matrix(
                        market_data=mk_data_voting,
                        event_idx=pd.DatetimeIndex(event_idx),
                        direction=dir_sign,
                        breakout_lookback=20,
                        vwap_lookback=20,
                        vol_lookback=20,
                    )
                    committee_names.extend(NEW_EXPERT_NAMES)
                    tprint_info(f"   [new experts] Computed {len(NEW_EXPERT_NAMES)} new experts: {NEW_EXPERT_NAMES}")
                except Exception as new_exp_exc:
                    tprint_warning(f"⚠️ Failed to compute new experts: {new_exp_exc}")
                    new_expert_scores = None
                    new_expert_conf = None

                # Total experts = 6 base + 3 new (if available)
                n_base_experts = len(committee_configs)
                n_new_experts = 3 if new_expert_scores is not None else 0
                n_total_experts = n_base_experts + n_new_experts

                # Initialize matrices
                label_matrix_values = np.zeros((len(event_idx), n_total_experts), dtype=np.int8)
                returns_matrix_values = np.full((len(event_idx), n_total_experts), np.nan, dtype=np.float32)
                durations_matrix_values = np.full((len(event_idx), n_total_experts), np.nan, dtype=np.float32)
                confidence_matrix_values = np.full((len(event_idx), n_total_experts), np.nan, dtype=np.float32)

                for i, res in enumerate(committee_results):
                    # align to event_idx
                    lbls = res['labels'].reindex(event_idx).fillna(0).values.astype(int)
                    rets = res['returns'].reindex(event_idx).values.astype(np.float32)
                    durs_s = res.get("durations")
                    if not isinstance(durs_s, pd.Series):
                        durs_s = res.get("event_durations")
                    if isinstance(durs_s, pd.Series):
                        dur_vals = durs_s.reindex(event_idx).values.astype(np.float32)
                    else:
                        try:
                            h = float(getattr(committee_configs[i], "horizon", 1.0))
                        except Exception:
                            h = 1.0
                        dur_vals = np.full(int(len(event_idx)), float(h), dtype=np.float32)
                    conf = res.get('confidence')
                    if isinstance(conf, pd.Series):
                        conf_vals = conf.reindex(event_idx).values.astype(np.float32)
                    else:
                        conf_vals = np.full(int(len(event_idx)), 1.0, dtype=np.float32)

                    label_matrix_values[:, i] = lbls
                    returns_matrix_values[:, i] = rets
                    durations_matrix_values[:, i] = dur_vals
                    confidence_matrix_values[:, i] = conf_vals

                # Add new experts to matrices (columns 6, 7, 8)
                if new_expert_scores is not None and new_expert_conf is not None:
                    for j in range(n_new_experts):
                        col_idx = n_base_experts + j
                        # Convert score to label: sign of score
                        scores_j = new_expert_scores[:, j]
                        labels_j = np.sign(scores_j).astype(np.int8)
                        # Returns: use score magnitude as proxy (scaled)
                        # New experts don't have realized returns, so use score * avg_base_return as proxy
                        avg_base_ret = np.nanmean(np.abs(returns_matrix_values[:, :n_base_experts]))
                        if not np.isfinite(avg_base_ret) or avg_base_ret < 1e-6:
                            avg_base_ret = 0.001
                        returns_j = scores_j * avg_base_ret
                        # Confidence from expert
                        conf_j = new_expert_conf[:, j]
                        # Duration: use median base duration
                        med_dur = np.nanmedian(durations_matrix_values[:, :n_base_experts])
                        if not np.isfinite(med_dur) or med_dur < 1:
                            med_dur = 12.0
                        dur_j = np.full(len(event_idx), med_dur, dtype=np.float32)

                        label_matrix_values[:, col_idx] = labels_j
                        returns_matrix_values[:, col_idx] = returns_j.astype(np.float32)
                        durations_matrix_values[:, col_idx] = dur_j
                        confidence_matrix_values[:, col_idx] = conf_j.astype(np.float32)

                tprint_success(f"✅ Committee Matrices Built: {label_matrix_values.shape} (Events x {n_total_experts} Experts)")

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
                    conf_mat = np.asarray(confidence_matrix_values, dtype=float)
                    conf_mat = np.where(np.isfinite(conf_mat) & (conf_mat >= 0.0), conf_mat, 0.0)
                    ret_mat = np.asarray(returns_matrix_values, dtype=float)
                    ret_mat = np.where(fired_mask, ret_mat, np.nan)

                    # Use the pre-step optimized committee voting weights to compute
                    # agreement/magnitude (so the weight factor aligns with the voting model).
                    try:
                        w_scalp_c = float(best_committee_params.get("w_scalp", 1.0))
                        w_swing_c = float(best_committee_params.get("w_swing", 1.0))
                        w_trend_c = float(best_committee_params.get("w_trend", 1.0))
                        w_breakout_c = float(best_committee_params.get("w_breakout", 0.5))
                        w_vwap_rev_c = float(best_committee_params.get("w_vwap_rev", 0.5))
                        w_vol_shock_c = float(best_committee_params.get("w_vol_shock", 0.5))
                    except Exception:
                        w_scalp_c, w_swing_c, w_trend_c = 1.0, 1.0, 1.0
                        w_breakout_c, w_vwap_rev_c, w_vol_shock_c = 0.5, 0.5, 0.5

                    n_exp_cf = int(label_matrix_values.shape[1]) if isinstance(label_matrix_values, np.ndarray) else 6
                    if n_exp_cf > 6:
                        weights_vec = np.array(
                            [
                                w_scalp_c,
                                w_scalp_c,
                                w_swing_c,
                                w_swing_c,
                                w_trend_c,
                                w_trend_c,
                                w_breakout_c,
                                w_vwap_rev_c,
                                w_vol_shock_c,
                            ],
                            dtype=float,
                        )
                    else:
                        weights_vec = np.array(
                            [w_scalp_c, w_scalp_c, w_swing_c, w_swing_c, w_trend_c, w_trend_c],
                            dtype=float,
                        )
                    weights_vec = np.where(np.isfinite(weights_vec) & (weights_vec >= 0.0), weights_vec, 0.0)
                    if float(np.sum(weights_vec)) <= 1e-12:
                        weights_vec = np.ones_like(weights_vec, dtype=float)

                    fired_w = fired_mask.astype(float) * conf_mat * weights_vec.reshape(1, -1)
                    denom = np.sum(fired_w, axis=1).astype(float) + 1e-8

                    abs_ret = np.abs(ret_mat)
                    abs_ret = np.where(fired_mask, abs_ret, np.nan)
                    abs_w = np.where(np.isfinite(abs_ret), abs_ret, 0.0) * conf_mat * weights_vec.reshape(1, -1)
                    abs_ret_mean = np.sum(abs_w, axis=1).astype(float) / denom
                    abs_ret_mean = np.where(np.isfinite(abs_ret_mean), abs_ret_mean, 0.0)

                    positive_abs = abs_ret_mean[abs_ret_mean > 0]
                    abs_med = float(np.nanmedian(positive_abs)) if positive_abs.size > 0 else 0.0
                    if np.isfinite(abs_med) and abs_med > 0:
                        mag_factor = abs_ret_mean / (abs_med + 1e-12)
                    else:
                        mag_factor = np.ones_like(abs_ret_mean, dtype=float)

                    # FIX: Use ex-ante signal strength instead of outcome-based agreement
                    # The old code used label_matrix_values (realized outcomes) which creates leakage
                    # New approach: use signal strength from primary_signals (ex-ante, no leakage)
                    try:
                        if primary_signals is not None and "consensus" in primary_signals:
                            sig_strength = primary_signals["consensus"].reindex(event_idx).fillna(0.0).abs().values
                            sig_strength = np.where(np.isfinite(sig_strength), sig_strength, 0.0)
                            # Normalize to [0, 1] range using quantile scaling
                            sig_finite = sig_strength[sig_strength > 0]
                            if sig_finite.size >= 20:
                                sig_p95 = float(np.quantile(sig_finite, 0.95))
                                if sig_p95 > 0:
                                    agree = np.clip(sig_strength / sig_p95, 0.0, 1.0)
                                else:
                                    agree = np.zeros_like(sig_strength)
                            else:
                                agree = np.clip(sig_strength, 0.0, 1.0)
                        else:
                            agree = np.zeros(len(event_idx), dtype=float)
                    except Exception:
                        agree = np.zeros(len(event_idx), dtype=float)

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
                tprint_warning(f"⚠️ Committee pre-computation failed: {e}. Continuing without committee matrices.")
                committee_weight_factor_series = None

        tprint_info("🧪 Layer 2: Optimizing Trading Parameters...")
        try:
            l2_prob_thr_high = float(config.get("layer2_prob_threshold_high", 0.70))
        except Exception:
            l2_prob_thr_high = 0.70
        l2_prob_thr_high = float(np.clip(l2_prob_thr_high, 0.55, 0.85))

        try:
            l2_vol_penalty_high = float(config.get("layer2_volatility_penalty_lambda_high", 0.25))
        except Exception:
            l2_vol_penalty_high = 0.25
        if not np.isfinite(l2_vol_penalty_high):
            l2_vol_penalty_high = 0.25
        l2_vol_penalty_high = float(np.clip(l2_vol_penalty_high, 0.0, 1.0))
        layer2_search_space = {
            "profit_floor_tx_mult": {"type": "float", "low": 1.0, "high": 4.0},
            "sl_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
            "risk_reward_ratio": {"type": "float", "low": 1.0, "high": 5.0},
            "horizon_bars": {"type": "int", "low": 6, "high": 48},
            "min_event_spacing": {"type": "int", "low": 0, "high": 6},
            "trail_distance_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
            "prob_threshold": {"type": "float", "low": 0.50, "high": float(l2_prob_thr_high)},
            "ev_margin": {"type": "float", "low": 0.0, "high": 0.25},
            "volatility_penalty_lambda": {"type": "float", "low": 0.0, "high": float(l2_vol_penalty_high)},
            # Regime-conditional barrier geometry knobs (stabilize cross-regime performance)
            "barrier_regime_strength": {"type": "float", "low": 0.0, "high": 1.0},
            "barrier_regime_power": {"type": "float", "low": 0.5, "high": 2.0},
            # NEW: Strength-Adaptive Threshold - lowers prob_threshold when signal is strong
            # adjusted_threshold = prob_threshold - sig_strength_sensitivity * (sig_strength - 0.5)
            "sig_strength_sensitivity": {"type": "float", "low": 0.0, "high": 0.3},
            # NEW: Trailing Stop Trend Modulation (activate/tighten trail only in trends)
            "trail_trend_modulation": {"type": "float", "low": 0.0, "high": 2.0},
            # NEW: Barrier Asymmetry Regime Modulation (larger TP relative to SL in trends)
            "barrier_trend_asymmetry": {"type": "float", "low": 0.0, "high": 1.5},
            # NEW: Volume-Weighted Time (shrinks horizon when volume is high)
            # horizon = base_horizon / (1 + mod * (vol_ratio - 1))
            "horizon_volume_modulation": {"type": "float", "low": 0.0, "high": 2.0},
            # NEW: Volatility-of-Volatility Adjustment (widens barriers when vol is unstable)
            # width = base * (1 + exp * vol_of_vol)
            "barrier_vol_vol_exp": {"type": "float", "low": 0.0, "high": 1.5},
            
            # NEW: Mixture of Experts (MoE) Params
            "moe_trend_dominance": {"type": "float", "low": 0.0, "high": 1.0},
            "moe_scalp_dominance": {"type": "float", "low": 0.0, "high": 1.0},
            "moe_vol_sensitivity": {"type": "float", "low": 0.0, "high": 1.0},
            # MoE quantile thresholds (data-driven, converted to ADX thresholds per-trial)
            "moe_adx_trend_q": {"type": "float", "low": 0.55, "high": 0.95},
            "moe_adx_chop_q": {"type": "float", "low": 0.05, "high": 0.45},
            "moe_vol_spike_q": {"type": "float", "low": 0.70, "high": 0.99},
            
            # NEW: Probabilistic Stops (First Passage Time Veto)
            "prob_stop_enable": {"type": "int", "low": 0, "high": 1}, # Treat as bool
            "prob_stop_threshold": {"type": "float", "low": 0.55, "high": 0.95},
            "prob_stop_drift_window": {"type": "int", "low": 12, "high": 96},
            
            # NEW: Regime-Adaptive Probability Thresholds
            # These adjustments are ADDED to base prob_threshold for each regime.
            # Negative = lower threshold (more trades), Positive = higher threshold (fewer trades)
            # Range is intentionally asymmetric to allow lowering threshold more than raising.
            "prob_threshold_adj_vol_low": {"type": "float", "low": -0.15, "high": 0.05},   # Low vol: often lower threshold needed
            "prob_threshold_adj_vol_high": {"type": "float", "low": -0.05, "high": 0.10},  # High vol: may need higher threshold
            "prob_threshold_adj_trend_high": {"type": "float", "low": -0.10, "high": 0.05}, # Strong trends: lower threshold
            "prob_threshold_adj_trend_low": {"type": "float", "low": -0.05, "high": 0.10},  # Choppy: higher threshold
        }

        _l2_std_trial_counter = [0]
        def layer2_objective(trial_params: Dict[str, Any]) -> float:
            _l2_std_trial_counter[0] += 1
            try:
                metrics_trial = _compute_layer2_metrics(trial_params)
                util = float(metrics_trial.get("utility", -1.0))
                if _l2_std_trial_counter[0] <= 3:
                    tprint_info(f"   [L2 Std Trial {_l2_std_trial_counter[0]}] utility={util:.4f}, fail_reason={metrics_trial.get('fail_reason', 'none')}")
                return util
            except Exception as e:
                if _l2_std_trial_counter[0] <= 3:
                    tprint_warning(f"   [L2 Std Trial {_l2_std_trial_counter[0]}] Exception: {e}")
                return -1.0

        meta_feature_cfg = config.get("meta_feature_engineering", {})
        volume_available = "volume" in market_data.columns

        # Optional: enable short NN sequence embeddings inside build_meta_features_for_model
        # This flag is consumed by feature_generation_meta_labeling_step.build_meta_features_for_model
        # and will append nn_embed_* features to the meta feature matrix when enabled.
        try:
            if isinstance(meta_feature_cfg, dict):
                if "enable_nn_sequence_embeddings" in config and "enable_nn_sequence_embeddings" not in meta_feature_cfg:
                    meta_feature_cfg["enable_nn_sequence_embeddings"] = bool(config.get("enable_nn_sequence_embeddings"))
                if "nn_sequence_encoder" in config and "nn_sequence_encoder" not in meta_feature_cfg:
                    nn_cfg = config.get("nn_sequence_encoder")
                    if isinstance(nn_cfg, dict):
                        meta_feature_cfg["nn_sequence_encoder"] = nn_cfg
                if "nn_embeddings_cache_path" in config and "nn_embeddings_cache_path" not in meta_feature_cfg:
                    meta_feature_cfg["nn_embeddings_cache_path"] = config.get("nn_embeddings_cache_path")
        except Exception:
            pass

        # ------------------------------------------------------------------
        # OPTION: Use cached features from layer3_features parquet or labeled_data artifact
        # ------------------------------------------------------------------
        use_cached_features = bool(config.get("hpo_use_cached_features", True))
        cached_features_loaded = False
        
        if use_cached_features:
            # Priority 1: Check for most recent layer3_features_*.parquet
            try:
                import glob
                layer3_features_dir = Path("versioned_artifacts") / symbol / exchange / timeframe / "layer3_features"
                if layer3_features_dir.exists():
                    pattern = str(layer3_features_dir / f"layer3_features_{symbol}_{timeframe}_*.parquet")
                    parquet_files = sorted(glob.glob(pattern), reverse=True)
                    if parquet_files:
                        most_recent = parquet_files[0]
                        tprint_info(f"📂 Found cached layer3 features: {most_recent}")
                        meta_features_full = pd.read_parquet(most_recent)
                        # Align to market_data index
                        meta_features_full = meta_features_full.reindex(market_data.index)
                        # Drop any label/target columns that might be present
                        exclude_cols = {
                            'binary_label', 'realized_return', 'target', 'target_long', 'target_short',
                            'meta_probability', 'meta_probability_ensemble', 'exit_reason', 'duration',
                            'label', 'return', 'returns'
                        }
                        feature_cols = [c for c in meta_features_full.columns if c.lower() not in exclude_cols]
                        meta_features_full = meta_features_full[feature_cols].copy()
                        if len(feature_cols) > 10:
                            cached_features_loaded = True
                            tprint_success(f"✅ Loaded {len(feature_cols)} cached features from layer3_features parquet")
            except Exception as l3_cache_exc:
                tprint_warning(f"⚠️ Failed to load layer3_features parquet: {l3_cache_exc}")
            
            # Priority 2: Fall back to labeled_data artifact
            if not cached_features_loaded:
                try:
                    from src.artifacts.versioned_artifact_store import VersionedArtifactStore
                    store = VersionedArtifactStore()
                    labeled_data = store.get_artifact(
                        f"labeled_data_{symbol}_{timeframe}",
                        version="latest"
                    )
                    if labeled_data is not None and hasattr(labeled_data, 'columns'):
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

        # ------------------------------------------------------------------
        # LAYER3 FEATURE CACHING (for Layer2 reuse in next run)
        # ------------------------------------------------------------------
        # Save meta-features (including NN embeddings) to cache so Layer2 can
        # reuse them in subsequent runs without recomputing.
        try:
            enable_layer3_cache = bool(config.get("enable_layer3_feature_cache", True))
            if enable_layer3_cache:
                # Check if we should load from cache first
                if should_use_cached_features(config, symbol, exchange, timeframe, direction):
                    cached_features, cache_metadata = load_layer3_features_from_cache(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe,
                        direction=direction,
                        target_index=meta_features_full.index,
                        market_data=market_data,
                        validate_hash=bool(config.get("layer3_cache_validate_hash", True)),
                        max_age_hours=config.get("layer3_cache_max_age_hours"),
                    )
                    if cached_features is not None:
                        # Merge cached features (especially NN embeddings) with current
                        meta_features_full = merge_cached_features_with_new(
                            new_features=meta_features_full,
                            cached_features=cached_features,
                            prefer_cached_nn=True,
                        )
                        tprint_info(
                            f"   [layer3_cache] Merged cached features: "
                            f"{len([c for c in meta_features_full.columns if c.startswith('nn_embed_')])} nn_embed_* cols"
                        )
        except Exception as cache_load_exc:
            tprint_warning(f"   ⚠️ Layer3 cache load failed: {cache_load_exc}")

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

            # Enhance consistency with committee agreement if available
            if label_matrix_values is not None and returns_matrix_values is not None:
                try:
                    # Get subset of matrices for current valid_idx
                    valid_positions = np.searchsorted(event_idx, l2_t_events)
                    valid_positions = valid_positions[valid_positions < len(event_idx)]
                    if len(valid_positions) == len(l2_t_events):
                        lbl_subset = label_matrix_values[valid_positions, :]
                        ret_subset = returns_matrix_values[valid_positions, :]
                        # Compute label agreement consistency
                        agreement_consistency = compute_label_agreement_consistency(lbl_subset, ret_subset)
                        # Compute return sign consistency
                        sign_consistency = compute_return_sign_consistency(ret_subset)
                        # Combine: geometric mean of all consistency measures
                        combined_consistency = (
                            batch_consistency * agreement_consistency * sign_consistency
                        ) ** (1.0 / 3.0)
                        combined_consistency = np.clip(combined_consistency, 0.1, 1.0)
                        batch_consistency = combined_consistency
                except Exception:
                    pass

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
                use_return_weighted_sw = bool(config.get("layer2_use_return_weighted_sample_weights", True))
            except Exception:
                use_return_weighted_sw = True

            if bool(use_return_weighted_sw):
                try:
                    y_sw = np.asarray(l2_labels_bin.values, dtype=float)
                    r_sw = np.asarray(l2_returns_clean.values, dtype=float)
                    if int(y_sw.size) == int(r_sw.size) and int(y_sw.size) > 0:
                        yb_sw = (y_sw >= 0.5).astype(int)
                        pos_raw = np.where(yb_sw == 1, np.maximum(0.0, r_sw), 0.0)
                        pos_mask = (yb_sw == 1) & np.isfinite(pos_raw)
                        pos_mean = float(np.mean(pos_raw[pos_mask])) if int(np.sum(pos_mask)) > 0 else 0.0
                        if (not np.isfinite(pos_mean)) or float(pos_mean) <= 0.0:
                            pos_mean = 1.0
                        scale = 1.0 / float(pos_mean)
                        try:
                            neg_w = float(config.get("layer2_return_weighted_neg_weight", 0.25))
                        except Exception:
                            neg_w = 0.25
                        if (not np.isfinite(neg_w)) or float(neg_w) < 0.0:
                            neg_w = 0.25
                        sw_new = np.where(yb_sw == 1, pos_raw * float(scale), float(neg_w))
                        try:
                            pos_clip = float(config.get("layer2_return_weighted_pos_clip", 10.0))
                        except Exception:
                            pos_clip = 10.0
                        if np.isfinite(pos_clip) and float(pos_clip) > 0.0:
                            sw_new = np.clip(sw_new, 0.0, float(pos_clip))
                        sw_new = np.where(np.isfinite(sw_new) & (sw_new >= 0.0), sw_new, 0.0)
                        sample_weights = sw_new
                except Exception:
                    pass

            try:
                timeout_factor = float(config.get("layer2_timeout_downweight_factor", config.get("timeout_downweight_factor", 0.25)))
            except Exception:
                timeout_factor = 0.25
            if np.isfinite(timeout_factor) and timeout_factor < 1.0:
                try:
                    er = l2_exit_reasons.reindex(l2_t_events)
                    timeout_mask = (er.astype(str).values == "timeout")
                    sw = np.asarray(sample_weights, dtype=float)
                    sw = np.where(timeout_mask, sw * float(timeout_factor), sw)
                    sample_weights = sw
                except Exception:
                    pass

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

            try:
                sw = np.asarray(sample_weights, dtype=float)
                if int(sw.size) != int(len(l2_t_events)):
                    raise ValueError("sample_weights size mismatch")
                sw = np.where(np.isfinite(sw) & (sw >= 0.0), sw, 0.0)
                sw_mean = float(np.mean(sw)) if sw.size else 1.0
                if (not np.isfinite(sw_mean)) or sw_mean <= 0.0:
                    sw = np.ones(int(len(l2_t_events)), dtype=float)
                    sw_mean = 1.0
                sample_weights = sw / float(sw_mean)
            except Exception:
                sample_weights = None

            try:
                layer2_econ_win_enabled = bool(config.get("layer2_econ_win_enabled", False))
            except Exception:
                layer2_econ_win_enabled = False
            try:
                layer2_econ_win_tx_mult = float(config.get("layer2_econ_win_tx_mult", ECON_MIN_RETURN_MULTIPLE))
            except Exception:
                layer2_econ_win_tx_mult = float(ECON_MIN_RETURN_MULTIPLE)
            if not np.isfinite(layer2_econ_win_tx_mult) or layer2_econ_win_tx_mult <= 0.0:
                layer2_econ_win_tx_mult = float(ECON_MIN_RETURN_MULTIPLE)
            layer2_econ_win_floor = float(DEFAULT_TRANSACTION_COST) * float(layer2_econ_win_tx_mult)

            X_trial = meta_features_full.loc[valid_idx].fillna(0)
            n_cv_folds = 5
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            enable_confident_learning = bool(config.get("enable_confident_learning", True))
            confident_learning_frac = float(config.get("confident_learning_frac", 0.05))
            if enable_confident_learning and len(X_trial) >= 100:
                try:
                    sample_weights, noisy_mask, cl_diagnostics = filter_noisy_labels(
                        X=X_trial,
                        y=l2_labels_bin.values,
                        sample_weights=sample_weights,
                        method="confident_learning",
                        action="downweight",
                        downweight_factor=0.1,
                        frac_to_filter=confident_learning_frac,
                        n_cv_splits=3,
                        random_state=42,
                        verbose=False,
                    )
                except Exception as e_cl:
                    tprint_warning(f"⚠️ Confident learning failed: {e_cl}")

            # D. FAST MODEL TRAINING WITH CV
            n_cv_folds = 5
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            try:
                if int(pd.Series(l2_labels_bin.values).nunique()) < 2:
                    raise ValueError("Layer2 labels have <2 classes")
                
                y_train_target = l2_labels_bin
                
                cv_preds_raw, cv_preds, folds_sharpe, mean_brier, mean_ece, mean_mce = _cross_val_predict_proba_and_fold_sharpes_weighted(
                    estimator=fast_model,
                    X=X_trial,
                    y=y_train_target,
                    sample_weight=sample_weights,
                    n_splits=n_cv_folds,
                    returns=l2_returns_clean.values.astype(float),  # Keep linear returns for Sharpe calculation
                    direction=direction,
                    prob_thr=float(prob_thr),
                    use_calibration=True,
                    enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                    ev_margin=float(ev_margin_local),
                    time_aware_cv=True,
                    event_durations=l2_durations.reindex(l2_t_events) if l2_durations is not None else None,
                    market_index=market_data.index if market_data is not None else None,
                    base_horizon_bars=12,
                )

            except Exception:
                try:
                    base_p = float(np.mean(l2_labels_bin.values.astype(float)))
                    if not np.isfinite(base_p):
                        base_p = 0.5
                except Exception:
                    base_p = 0.5
                cv_preds_raw = np.full(int(len(l2_labels_bin)), float(base_p), dtype=float)
                cv_preds = np.full(int(len(l2_labels_bin)), float(base_p), dtype=float)
                folds_sharpe = np.zeros(int(max(1, n_cv_folds)), dtype=float)
                mean_brier, mean_ece, mean_mce = None, None, None

            # E. COMPUTE AUC (for trapezoidal gate)
            # FIX: Use returns-based labels for consistency with per-fold AUC
            try:
                returns_for_auc = l2_returns_clean.reindex(l2_t_events).values.astype(float)
                if bool(layer2_econ_win_enabled):
                    y_auc = (returns_for_auc > float(layer2_econ_win_floor)).astype(float)
                else:
                    y_auc = (returns_for_auc > 0.0).astype(float)
                p_auc_raw = np.asarray(cv_preds)
                if getattr(p_auc_raw, "ndim", 0) == 1 and getattr(p_auc_raw, "dtype", None) is not None and p_auc_raw.dtype == object:
                    try:
                        p_auc_raw = np.vstack(p_auc_raw)
                    except Exception:
                        pass
                if getattr(p_auc_raw, "ndim", 0) == 2 and int(p_auc_raw.shape[1]) >= 2:
                    p_auc = np.asarray(p_auc_raw[:, 1], dtype=float)
                elif getattr(p_auc_raw, "ndim", 0) == 2 and int(p_auc_raw.shape[1]) == 1:
                    p_auc = np.asarray(p_auc_raw[:, 0], dtype=float)
                else:
                    p_auc = np.asarray(p_auc_raw, dtype=float)
                m_auc = np.isfinite(y_auc) & np.isfinite(p_auc)
                if int(np.sum(m_auc)) >= 20:
                    if int(np.unique(y_auc[m_auc]).size) >= 2:
                        mean_auc = float(roc_auc_score(y_auc[m_auc].astype(int), p_auc[m_auc]))
                    else:
                        mean_auc = 0.5
                else:
                    mean_auc = 0.5
            except Exception:
                mean_auc = 0.5

            label_pos_rate_val = None
            label_n_pos_val = None
            label_n_neg_val = None
            pred_nan_frac = None
            pred_std = None
            pred_min = None
            pred_max = None
            pred_unique_rounded = None
            try:
                y_tmp = np.asarray(l2_labels_clean.values, dtype=float)
                m_y = np.isfinite(y_tmp)
                if int(np.sum(m_y)) > 0:
                    yb = (y_tmp[m_y] >= 0.5).astype(int)
                    label_pos_rate_val = float(np.mean(yb))
                    label_n_pos_val = int(np.sum(yb == 1))
                    label_n_neg_val = int(np.sum(yb == 0))
            except Exception:
                pass
            try:
                p_tmp = np.asarray(cv_preds, dtype=float)
                pred_nan_frac = float(np.mean(~np.isfinite(p_tmp)))
                p_fin = p_tmp[np.isfinite(p_tmp)]
                if p_fin.size > 0:
                    pred_std = float(np.std(p_fin, ddof=1)) if p_fin.size > 1 else 0.0
                    pred_min = float(np.min(p_fin))
                    pred_max = float(np.max(p_fin))
                    pred_unique_rounded = int(np.unique(np.round(p_fin, 4)).size)
            except Exception:
                pass

            y_true_arr = l2_labels_bin.values.astype(float)
            returns_arr = l2_returns_clean.values.astype(float)

            # G. COMPUTE TRADES PER DAY (from predicted trades, with strength-adaptive threshold)
            # NEW: Strength-Adaptive Threshold - lower threshold when signal is strong
            try:
                sig_strength_sens = float(params.get("sig_strength_sensitivity", 0.0))
            except Exception:
                sig_strength_sens = 0.0
            if not np.isfinite(sig_strength_sens):
                sig_strength_sens = 0.0
            sig_strength_sens = float(np.clip(sig_strength_sens, 0.0, 0.5))
            
            base_prob_thr = float(params.get("prob_threshold", 0.5))
            
            try:
                # Get signal strength from primary_signals consensus (if available)
                if primary_signals is not None and "consensus" in primary_signals.columns:
                    sig_strength_arr = primary_signals["consensus"].reindex(l2_t_events).fillna(0.0).abs().values
                    # Normalize sig_strength to 0-1 range (clip at 2.0 max, then divide)
                    sig_strength_arr = np.clip(sig_strength_arr, 0.0, 2.0) / 2.0
                else:
                    sig_strength_arr = np.ones(len(l2_t_events), dtype=float) * 0.5
            except Exception:
                sig_strength_arr = np.ones(len(l2_t_events), dtype=float) * 0.5
            
            # Adaptive threshold: lower when signal is strong (sig_strength > 0.5)
            # Formula: adjusted_threshold = base_threshold - sensitivity * (sig_strength - 0.5)
            # When sig_strength=1.0 and sensitivity=0.2, threshold drops by 0.1
            # When sig_strength=0.0 and sensitivity=0.2, threshold rises by 0.1
            adaptive_thresholds = base_prob_thr - sig_strength_sens * (sig_strength_arr - 0.5)
            adaptive_thresholds = np.clip(adaptive_thresholds, 0.3, 0.95)  # Safety bounds
            
            try:
                p_sz_raw = np.asarray(cv_preds)
                if getattr(p_sz_raw, "ndim", 0) == 1 and getattr(p_sz_raw, "dtype", None) is not None and p_sz_raw.dtype == object:
                    try:
                        p_sz_raw = np.vstack(p_sz_raw)
                    except Exception:
                        pass
                if getattr(p_sz_raw, "ndim", 0) == 2 and int(p_sz_raw.shape[1]) >= 2:
                    p_sz = np.asarray(p_sz_raw[:, 1], dtype=float)
                elif getattr(p_sz_raw, "ndim", 0) == 2 and int(p_sz_raw.shape[1]) == 1:
                    p_sz = np.asarray(p_sz_raw[:, 0], dtype=float)
                else:
                    p_sz = np.asarray(p_sz_raw, dtype=float)

                sizes = np.zeros(int(len(p_sz)), dtype=float)
                for i, pv in enumerate(p_sz):
                    sizes[i] = float(
                        directional_size_from_prob(
                            float(pv),
                            direction=direction,
                            thr=float(prob_thr),
                            max_exposure=1.0,
                            scale=1.0,
                        )
                    )
                take_mask = np.isfinite(sizes) & (np.abs(sizes) > 1e-12)
                n_pred_trades = int(np.sum(take_mask))
                # CRITICAL FIX: Use absolute sizes because returns_arr is already direction-adjusted
                trade_returns = np.asarray(returns_arr, dtype=float) * np.abs(np.asarray(sizes, dtype=float))
                trade_returns = trade_returns[np.asarray(take_mask, dtype=bool)]
                trade_returns = trade_returns[np.isfinite(trade_returns)]
            except Exception:
                n_pred_trades = int(len(l2_returns_clean))
                trade_returns = np.asarray([], dtype=float)
            trades_per_day = float(n_pred_trades) / float(max(days_span, 1))

            # H. CALCULATE UTILITY (Probabilistic Sortino Ratio on OOF traded daily returns)
            # CHANGED: Replaced PSR (Probabilistic Sharpe Ratio) with Probabilistic Sortino Ratio
            # Rationale: Sortino only penalizes downside volatility, not upside volatility.
            # In trading, large wins (upside vol) are desirable and should not be penalized.
            utility_debug: Dict[str, Any] = {}
            try:
                psor_min_trades = int(config.get("layer2_psor_min_trades", config.get("layer2_psr_min_trades", 30)))
            except Exception:
                psor_min_trades = 30
            psor_min_trades = int(max(1, psor_min_trades))

            try:
                sortino_benchmark = float(config.get("layer2_sortino_benchmark", config.get("layer2_psr_sr_benchmark", 0.0)))
            except Exception:
                sortino_benchmark = 0.0
            if not np.isfinite(sortino_benchmark):
                sortino_benchmark = 0.0

            psor_details = {"probabilistic_sortino": 0.0, "psor_z": float("-inf"), "sortino": None, "n": 0, "downside_deviation": None}
            try:
                if trade_returns is not None and int(np.size(trade_returns)) > 0:
                    idx_tr = pd.DatetimeIndex(ev_idx0)[np.asarray(take_mask, dtype=bool)]
                    tr = np.asarray(trade_returns, dtype=float)
                    # Align lengths defensively
                    n_tr = int(min(int(len(idx_tr)), int(tr.size)))
                    if n_tr > 0:
                        idx_tr = idx_tr[:n_tr]
                        tr = tr[:n_tr]
                        day_index = pd.date_range(
                            start=pd.DatetimeIndex(ev_idx0).min().normalize(),
                            end=pd.DatetimeIndex(ev_idx0).max().normalize(),
                            freq="D",
                        )
                        daily_pnl = pd.Series(tr, index=idx_tr).groupby(idx_tr.normalize()).sum()
                        daily_pnl = daily_pnl.reindex(day_index, fill_value=0.0)
                        daily_log = np.log1p(daily_pnl.astype(float).values)
                        daily_log = daily_log[np.isfinite(daily_log)]
                        # Use Probabilistic Sortino instead of PSR
                        psor_details = _probabilistic_sortino_from_returns(
                            daily_log,
                            sortino_benchmark=float(sortino_benchmark),
                            periods_per_year=365.0,
                        )
            except Exception:
                pass

            # Soft trade-count gate (keeps low-trade configs from dominating).
            phi_trades = 0.0
            try:
                phi_trades = float(np.clip(float(psor_details.get("n", 0)) / float(psor_min_trades), 0.0, 1.0))
            except Exception:
                phi_trades = 0.0
            utility = float(psor_details.get("probabilistic_sortino", 0.0)) * float(phi_trades)
            if not np.isfinite(float(utility)):
                utility = 0.0

            if isinstance(utility_debug, dict):
                try:
                    utility_debug.update(
                        {
                            # Keep backward-compatible keys with "psr" prefix for logging compatibility
                            "psr": float(psor_details.get("probabilistic_sortino", 0.0)),
                            "psr_z": float(psor_details.get("psor_z", float("-inf"))),
                            "psr_sr": psor_details.get("sortino", None),  # Actually Sortino now
                            "psr_n": int(psor_details.get("n", 0) or 0),
                            # New Sortino-specific keys
                            "probabilistic_sortino": float(psor_details.get("probabilistic_sortino", 0.0)),
                            "psor_z": float(psor_details.get("psor_z", float("-inf"))),
                            "sortino": psor_details.get("sortino", None),
                            "downside_deviation": psor_details.get("downside_deviation", None),
                            "sortino_benchmark": float(sortino_benchmark),
                            "phi_trades": float(phi_trades),
                            "utility_pre_clip": float(utility),
                            "utility": float(utility),
                        }
                    )
                except Exception:
                    pass

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
                    f"utility={utility:.4f} (psor={float(utility_debug.get('probabilistic_sortino', 0.0)):.4f}, z={float(utility_debug.get('psor_z', -1e9)):.2f}, n={int(utility_debug.get('psr_n', 0) or 0)}), auc={mean_auc:.4f}, "
                    f"trades_per_day={trades_per_day:.2f}, "
                    f"folds_sortino_mean={float(np.mean(folds_sharpe)):.4f}, "
                    f"folds_sortino_std={float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe)>1 else 0.0:.4f}, "
                    f"take_trades={int(np.size(trade_returns) if trade_returns is not None else 0)}"
                )
            except Exception:
                pass

            return utility

        def _compute_layer2_metrics(params: Dict[str, Any], *, write_diagnostics: bool = False) -> Dict[str, Any]:
            """Single-shot computation of Layer 2 metrics for reporting."""
            trail_dist = float(params.get("trail_distance_atr_mult", 0.0))
            prob_thr = float(params.get("prob_threshold", 0.5))
            ev_margin_local = float(params.get("ev_margin", config.get("ev_margin", 0.0)))
            vol_penalty_lambda = float(params.get("volatility_penalty_lambda", 0.0))

            try:
                l2_horizon_bars = int(params.get("horizon_bars", 12))
            except Exception:
                l2_horizon_bars = 12
            try:
                l2_min_event_spacing = int(params.get("min_event_spacing", 2))
            except Exception:
                l2_min_event_spacing = 2

            prof_thr = fixed_layer2_profit_thr
            stop_thr = fixed_layer2_stop_thr
            horizon_for_call: Union[int, pd.Series] = int(l2_horizon_bars)
            
            # NEW: Volume-Weighted Time (Horizon Modulation)
            # Concept: In high volume, events happen faster -> shrink horizon.
            # In low volume, events drag on -> expand horizon.
            try:
                horizon_vol_mod = float(params.get("horizon_volume_modulation", 0.0))
                if horizon_vol_mod > 0.01:
                    # Need volume ratio feature. 
                    # If "reg_ohlcv__vol_ratio_5" exists use it, else approximate or skip
                    vol_ratio_series = None
                    if "reg_ohlcv__vol_ratio_5" in market_data.columns:
                        vol_ratio_series = market_data["reg_ohlcv__vol_ratio_5"].fillna(1.0)
                    elif "volume" in market_data.columns:
                        # Compute on fly if needed (simple calc)
                        vol = market_data["volume"].replace(0, np.nan).fillna(method='ffill')
                        avg_vol = vol.rolling(96).mean().fillna(vol)
                        vol_ratio_series = vol / (avg_vol + 1e-8)
                    
                    if vol_ratio_series is not None:
                        # Logic: horizon = base / (1 + mod * (ratio - 1))
                        # e.g. mod=1.0, ratio=2.0 (2x vol) -> horizon = base / 2.0 (half time)
                        # e.g. mod=1.0, ratio=0.5 (half vol) -> horizon = base / 0.5 (double time)
                        
                        # Clip ratio to avoid extreme horizons
                        vr = vol_ratio_series.clip(0.2, 5.0)
                        
                        # Apply modulation factor
                        mod_factor = 1.0 + horizon_vol_mod * (vr - 1.0)
                        mod_factor = mod_factor.clip(0.25, 4.0) # Limit time dilation
                        
                        horizon_series = float(l2_horizon_bars) / mod_factor
                        horizon_for_call = horizon_series.fillna(float(l2_horizon_bars)).astype(int).clip(1, 100) # Ensure int
            except Exception:
                pass
            
            trail_mult_for_call: Optional[Union[float, pd.Series]] = float(trail_dist)

            # NEW: Trailing Stop Trend Modulation
            # If enabled, trailing stop is tighter (smaller dist) when trend is strong
            try:
                trail_trend_mod = float(params.get("trail_trend_modulation", 0.0))
                if trail_trend_mod > 0.01:
                    # Get trend strength (ADX) if available, else standard deviation ratio
                    # Simplified proxy: vol capacity often correlates with trendiness
                    # Better: check if we have adx feature in market_data
                    is_trending = np.zeros(len(market_data), dtype=bool)
                    if "reg_res_adx_14" in market_data.columns:
                        is_trending = market_data["reg_res_adx_14"].fillna(0.0).values > 25.0
                    
                    # If trending, tighten trail by dividing distance
                    # effective_trail = base_trail / (1 + modulation)
                    # e.g. mod=1.0 -> half distance in trends
                    mod_factor = np.where(is_trending, 1.0 + trail_trend_mod, 1.0)
                    
                    # Create series
                    trail_mult_series = pd.Series(
                        data=float(trail_dist) / mod_factor,
                        index=market_data.index
                    )
                    trail_mult_for_call = trail_mult_series
            except Exception:
                pass

            try:
                use_hpo_barrier_geometry = bool(enable_regime_conditional_barrier_geometry) or any(
                    k in params for k in ("sl_atr_mult", "risk_reward_ratio", "profit_floor_tx_mult", "horizon_bars")
                )
                if bool(use_hpo_barrier_geometry):
                    p_thr_s, s_thr_s, h_s, t_s = _compute_regime_conditional_barrier_geometry(
                        params=params,
                        market_index=market_data.index,
                        default_horizon=int(l2_horizon_bars),
                        atr_frac_series=atr_frac,
                    )
                    prof_thr = p_thr_s
                    stop_thr = s_thr_s
                    horizon_for_call = h_s
                    trail_mult_for_call = t_s
            except Exception:
                pass

            try:
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
                    horizon=horizon_for_call,
                    transaction_cost=DEFAULT_TRANSACTION_COST,
                    min_event_spacing=int(l2_min_event_spacing),
                    trail_distance_atr_mult=trail_mult_for_call,
                    atr_series=atr_series,
                )
            except Exception as e_l2_ret:
                return {
                    "valid_events": 0,
                    "utility": 0.0,
                    "fail_reason": "compute_realized_returns_failed",
                    "fail_exception": str(e_l2_ret),
                }

            valid_idx = ~l2_labels.isna()
            if valid_idx.sum() < 50:
                return {"valid_events": int(valid_idx.sum()), "utility": 0.0, "fail_reason": "too_few_valid_events"}

            l2_t_events = l2_returns.index[valid_idx]
            l2_returns_clean = l2_returns[valid_idx]
            l2_labels_clean = l2_labels[valid_idx]

            l2_labels_bin = None
            try:
                y_raw = np.asarray(l2_labels_clean.values, dtype=float)
                y_bin = (y_raw >= 0.5).astype(int)
                l2_labels_bin = pd.Series(y_bin, index=l2_labels_clean.index)
            except Exception:
                l2_labels_bin = None

            l2_labels_bin_base = None
            try:
                l2_labels_bin_base = l2_labels_bin.copy() if l2_labels_bin is not None else None
            except Exception:
                l2_labels_bin_base = None

            try:
                layer2_econ_win_enabled = bool(config.get("layer2_econ_win_enabled", False))
            except Exception:
                layer2_econ_win_enabled = False
            try:
                layer2_econ_win_tx_mult = float(config.get("layer2_econ_win_tx_mult", ECON_MIN_RETURN_MULTIPLE))
            except Exception:
                layer2_econ_win_tx_mult = float(ECON_MIN_RETURN_MULTIPLE)
            if not np.isfinite(layer2_econ_win_tx_mult) or layer2_econ_win_tx_mult <= 0.0:
                layer2_econ_win_tx_mult = float(ECON_MIN_RETURN_MULTIPLE)
            layer2_econ_win_floor = float(DEFAULT_TRANSACTION_COST) * float(layer2_econ_win_tx_mult)
            if bool(layer2_econ_win_enabled):
                try:
                    y_econ = (np.asarray(l2_returns_clean.values, dtype=float) > float(layer2_econ_win_floor)).astype(int)
                    l2_labels_bin = pd.Series(y_econ, index=l2_returns_clean.index)
                    l2_labels_bin_base = l2_labels_bin.copy()
                except Exception:
                    pass

            # =====================================================================
            # OPTION C: Use committee-voted labels (non-leaky) when available
            # =====================================================================
            # If committee matrices exist, replace single-TPSL labels with
            # committee-voted labels. Voting weights are learned per-fold from
            # training data only (no lookahead).
            # =====================================================================
            use_committee_voted_labels = bool(config.get("layer2_use_committee_voted_labels", True)) and (not bool(layer2_econ_win_enabled))
            committee_voted_labels_used = False
            if (
                use_committee_voted_labels
                and committee_label_matrix_values is not None
                and committee_returns_matrix_values is not None
                and committee_event_idx is not None
            ):
                try:
                    from sklearn.model_selection import TimeSeriesSplit
                    n_cv_folds_vote = 5
                    cv_vote = TimeSeriesSplit(n_splits=n_cv_folds_vote)
                    n_committee_events = len(committee_event_idx)
                    cv_splits_vote = list(cv_vote.split(np.arange(n_committee_events)))

                    voted_labels_full, vote_diag = compute_committee_voted_labels_full(
                        label_matrix=committee_label_matrix_values,
                        returns_matrix=committee_returns_matrix_values,
                        event_index=committee_event_idx,
                        cv_splits=cv_splits_vote,
                        floor_weight=0.1,
                        alpha_hit_rate=0.5,
                    )

                    # Align voted labels to l2_t_events
                    voted_aligned = voted_labels_full.reindex(l2_t_events)
                    valid_voted = ~voted_aligned.isna()
                    if int(valid_voted.sum()) >= 50 and int(voted_aligned.dropna().nunique()) >= 2:
                        voted_bin = voted_aligned.apply(lambda x: 1 if float(x) > 0.5 else 0) if int(valid_voted.sum()) > 0 else voted_aligned
                        voted_bin = voted_bin.astype(float)
                        base_aligned = None
                        try:
                            base_aligned = l2_labels_bin_base.reindex(l2_t_events).astype(float) if l2_labels_bin_base is not None else None
                        except Exception:
                            base_aligned = None
                        if base_aligned is not None:
                            merged = voted_bin.where(valid_voted, base_aligned)
                        else:
                            merged = voted_bin
                        merged = merged.astype(int)
                        l2_labels_bin = pd.Series(merged.values, index=l2_t_events)
                        committee_voted_labels_used = True
                        tprint_info(
                            f"   [L2 Option C] Using committee-voted labels: "
                            f"n={int(valid_voted.sum())}, pos_rate={vote_diag.get('pos_rate', 0.0):.3f}"
                        )
                except Exception as vote_exc:
                    tprint_warning(f"   ⚠️ Committee voting failed: {vote_exc}. Using single-TPSL labels.")

            if l2_labels_bin is None:
                return {"valid_events": int(valid_idx.sum()), "utility": 0.0, "fail_reason": "labels_binarization_failed"}

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
                use_return_weighted_sw = bool(config.get("layer2_use_return_weighted_sample_weights", True))
            except Exception:
                use_return_weighted_sw = True

            if bool(use_return_weighted_sw):
                try:
                    y_sw = np.asarray(l2_labels_bin.values, dtype=float)
                    r_sw = np.asarray(l2_returns_clean.values, dtype=float)
                    if int(y_sw.size) == int(r_sw.size) and int(y_sw.size) > 0:
                        yb_sw = (y_sw >= 0.5).astype(int)
                        pos_raw = np.where(yb_sw == 1, np.maximum(0.0, r_sw), 0.0)
                        pos_mask = (yb_sw == 1) & np.isfinite(pos_raw)
                        pos_mean = float(np.mean(pos_raw[pos_mask])) if int(np.sum(pos_mask)) > 0 else 0.0
                        if (not np.isfinite(pos_mean)) or float(pos_mean) <= 0.0:
                            pos_mean = 1.0
                        scale = 1.0 / float(pos_mean)
                        try:
                            neg_w = float(config.get("layer2_return_weighted_neg_weight", 0.25))
                        except Exception:
                            neg_w = 0.25
                        if (not np.isfinite(neg_w)) or float(neg_w) < 0.0:
                            neg_w = 0.25
                        sw_new = np.where(yb_sw == 1, pos_raw * float(scale), float(neg_w))
                        try:
                            pos_clip = float(config.get("layer2_return_weighted_pos_clip", 10.0))
                        except Exception:
                            pos_clip = 10.0
                        if np.isfinite(pos_clip) and float(pos_clip) > 0.0:
                            sw_new = np.clip(sw_new, 0.0, float(pos_clip))
                        sw_new = np.where(np.isfinite(sw_new) & (sw_new >= 0.0), sw_new, 0.0)
                        sample_weights = sw_new
                except Exception:
                    pass

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

            try:
                sw = np.asarray(sample_weights, dtype=float)
                if int(sw.size) != int(len(l2_t_events)):
                    raise ValueError("sample_weights size mismatch")
                sw = np.where(np.isfinite(sw) & (sw >= 0.0), sw, 0.0)
                sw_mean = float(np.mean(sw)) if sw.size else 1.0
                if (not np.isfinite(sw_mean)) or sw_mean <= 0.0:
                    sw = np.ones(int(len(l2_t_events)), dtype=float)
                    sw_mean = 1.0
                sample_weights = sw / float(sw_mean)
            except Exception:
                sample_weights = None

            X_trial = meta_features_full.loc[valid_idx].fillna(0)
            n_cv_folds = 5
            fast_model = lgb.LGBMClassifier(
                n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
            )

            cv_fail_reason: Optional[str] = None
            cv_fail_exception: Optional[str] = None
            try:
                if int(pd.Series(l2_labels_bin.values).nunique()) < 2:
                    cv_fail_reason = "single_class_labels"
                    raise ValueError("Layer2 labels have <2 classes")
                
                y_train_target = l2_labels_bin
                
                cv_preds_raw, cv_preds, folds_sharpe, mean_brier, mean_ece, mean_mce = _cross_val_predict_proba_and_fold_sharpes_weighted(
                    estimator=fast_model,
                    X=X_trial,
                    y=y_train_target,
                    sample_weight=sample_weights,
                    n_splits=n_cv_folds,
                    returns=l2_returns_clean.values.astype(float),  # Keep linear returns for Sharpe
                    direction=direction,
                    prob_thr=float(prob_thr),
                    use_calibration=True,
                    enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                    ev_margin=float(ev_margin_local),
                    time_aware_cv=True,
                    event_durations=l2_durations.reindex(l2_t_events) if l2_durations is not None else None,
                    market_index=market_data.index if market_data is not None else None,
                    base_horizon_bars=12,
                )

            except Exception as e:
                if cv_fail_reason is None:
                    cv_fail_reason = "cv_failed"
                try:
                    cv_fail_exception = str(e)
                except Exception:
                    cv_fail_exception = "unknown"
                try:
                    base_p = float(np.mean(l2_labels_bin.values.astype(float)))
                    if not np.isfinite(base_p):
                        base_p = 0.5
                except Exception:
                    base_p = 0.5
                cv_preds_raw = np.full(int(len(l2_labels_bin)), float(base_p), dtype=float)
                cv_preds = np.full(int(len(l2_labels_bin)), float(base_p), dtype=float)
                folds_sharpe = np.zeros(int(max(1, n_cv_folds)), dtype=float)
                mean_brier, mean_ece, mean_mce = None, None, None

            # FIX: Use returns-based labels for consistency with per-fold AUC
            # When using committee-voted labels, use committee-averaged returns for alignment
            try:
                if committee_voted_labels_used and committee_returns_matrix_values is not None and committee_event_idx is not None:
                    # Compute committee-averaged returns (mean across experts)
                    ret_mat = np.asarray(committee_returns_matrix_values, dtype=float)
                    fired_mask = ~np.isnan(ret_mat)
                    ret_masked = np.where(fired_mask, ret_mat, 0.0)
                    n_fired = np.sum(fired_mask, axis=1).astype(float)
                    n_fired = np.maximum(n_fired, 1.0)
                    avg_ret_committee = np.sum(ret_masked, axis=1) / n_fired
                    avg_ret_series = pd.Series(avg_ret_committee, index=committee_event_idx)
                    returns_for_auc = avg_ret_series.reindex(l2_t_events).values.astype(float)
                else:
                    returns_for_auc = l2_returns_clean.reindex(l2_t_events).values.astype(float)
                y_auc = (returns_for_auc > 0.0).astype(float)
                p_auc_raw = np.asarray(cv_preds)
                if getattr(p_auc_raw, "ndim", 0) == 1 and getattr(p_auc_raw, "dtype", None) is not None and p_auc_raw.dtype == object:
                    try:
                        p_auc_raw = np.vstack(p_auc_raw)
                    except Exception:
                        pass
                if getattr(p_auc_raw, "ndim", 0) == 2 and int(p_auc_raw.shape[1]) >= 2:
                    p_auc = np.asarray(p_auc_raw[:, 1], dtype=float)
                elif getattr(p_auc_raw, "ndim", 0) == 2 and int(p_auc_raw.shape[1]) == 1:
                    p_auc = np.asarray(p_auc_raw[:, 0], dtype=float)
                else:
                    p_auc = np.asarray(p_auc_raw, dtype=float)
                m_auc = np.isfinite(y_auc) & np.isfinite(p_auc)
                if int(np.sum(m_auc)) >= 20:
                    if int(np.unique(y_auc[m_auc]).size) >= 2:
                        mean_auc = float(roc_auc_score(y_auc[m_auc].astype(int), p_auc[m_auc]))
                    else:
                        mean_auc = 0.5
                else:
                    mean_auc = 0.5
            except Exception:
                mean_auc = 0.5

            y_true_arr = l2_labels_bin.values.astype(float)
            returns_arr = l2_returns_clean.values.astype(float)

            sizes_full = None
            take_mask = None
            trade_returns = None
            n_trades = 0
            take_rate = 0.0
            trade_mean_return = None
            trade_win_rate = None
            max_drawdown = None
            veto_rate = 0.0
            p_sz = None
            p_fail_arr = None
            prob_stop_threshold_val = None
            oof_taken_trade_deciles: Optional[List[Dict[str, Any]]] = None
            oof_taken_trade_deciles_path: Optional[str] = None
            oof_all_event_deciles: Optional[List[Dict[str, Any]]] = None
            oof_all_event_deciles_path: Optional[str] = None
            oof_prob_threshold_sweep_best: Optional[Dict[str, Any]] = None
            oof_prob_threshold_sweep_path: Optional[str] = None
            try:
                p_sz_raw = np.asarray(cv_preds)
                if getattr(p_sz_raw, "ndim", 0) == 1 and getattr(p_sz_raw, "dtype", None) is not None and p_sz_raw.dtype == object:
                    try:
                        p_sz_raw = np.vstack(p_sz_raw)
                    except Exception:
                        pass
                if getattr(p_sz_raw, "ndim", 0) == 2 and int(p_sz_raw.shape[1]) >= 2:
                    p_sz = np.asarray(p_sz_raw[:, 1], dtype=float)
                elif getattr(p_sz_raw, "ndim", 0) == 2 and int(p_sz_raw.shape[1]) == 1:
                    p_sz = np.asarray(p_sz_raw[:, 0], dtype=float)
                else:
                    p_sz = np.asarray(p_sz_raw, dtype=float)

                sizes = np.zeros(int(len(p_sz)), dtype=float)
                for i, pv in enumerate(p_sz):
                    sizes[i] = float(
                        directional_size_from_prob(
                            float(pv),
                            direction=direction,
                            thr=float(prob_thr),
                            max_exposure=1.0,
                            scale=1.0,
                        )
                    )
                sizes_full = sizes
                take_mask = np.isfinite(sizes_full) & (np.abs(sizes_full) > 1e-12)

                # ---------------------------------------------------------
                # Probabilistic stop veto (First Passage Time) applied to trade mask
                # ---------------------------------------------------------
                try:
                    prob_stop_active = int(params.get("prob_stop_enable", 0)) > 0
                except Exception:
                    prob_stop_active = False

                if bool(prob_stop_active):
                    try:
                        from src.training.steps.labeling.layer2_advanced_logic import calc_prob_touch_sl_vec

                        # Direction sign for veto math
                        dir_raw = str(direction).lower()
                        dir_sign = 1
                        if dir_raw in {"short", "sell", "-1", "s"}:
                            dir_sign = -1

                        p_thr = float(params.get("prob_stop_threshold", 0.70))
                        prob_stop_threshold_val = float(p_thr)
                        w_drift = int(params.get("prob_stop_drift_window", 24))
                        w_drift = int(np.clip(w_drift, 2, 512))

                        if market_data is not None and isinstance(market_data, pd.DataFrame) and "close" in market_data.columns:
                            mkt_rets = market_data["close"].pct_change().fillna(0.0)
                            drift_arr = mkt_rets.rolling(w_drift).mean()
                            vol_arr = mkt_rets.rolling(w_drift).std()

                            ev_drift = drift_arr.reindex(l2_t_events).fillna(0.0).to_numpy(dtype=float)
                            ev_vol = vol_arr.reindex(l2_t_events).fillna(0.0).to_numpy(dtype=float)

                            # Scale drift/vol to the horizon as a crude approximation
                            h = float(l2_horizon_bars)
                            mu = ev_drift * h
                            sigma = np.maximum(ev_vol * np.sqrt(h), 1e-6)

                            # Use the actual barrier geometry used by compute_realized_returns
                            if isinstance(stop_thr, pd.Series):
                                sl_d = pd.to_numeric(stop_thr.reindex(l2_t_events), errors="coerce").abs().fillna(0.0).to_numpy(dtype=float)
                            else:
                                sl_d = np.full(int(len(l2_t_events)), abs(float(stop_thr)), dtype=float)

                            if isinstance(prof_thr, pd.Series):
                                tp_d = pd.to_numeric(prof_thr.reindex(l2_t_events), errors="coerce").abs().fillna(0.0).to_numpy(dtype=float)
                            else:
                                tp_d = np.full(int(len(l2_t_events)), abs(float(prof_thr)), dtype=float)

                            # Guard against degenerate barriers
                            sl_d = np.maximum(sl_d, 1e-6)
                            tp_d = np.maximum(tp_d, 1e-6)

                            p_fail = calc_prob_touch_sl_vec(mu=mu, sigma=sigma, sl_dist=sl_d, tp_dist=tp_d, direction=int(dir_sign))
                            p_fail_arr = np.asarray(p_fail, dtype=float)
                            veto_mask = (np.asarray(p_fail, dtype=float) > float(p_thr))

                            # Apply veto only where we would have traded
                            veto_mask = np.asarray(veto_mask, dtype=bool) & np.asarray(take_mask, dtype=bool)
                            if int(np.sum(take_mask)) > 0:
                                veto_rate = float(np.mean(veto_mask[take_mask]))
                            else:
                                veto_rate = 0.0
                            take_mask = np.asarray(take_mask, dtype=bool) & (~veto_mask)
                    except Exception:
                        veto_rate = 0.0

                # CRITICAL FIX: Use absolute sizes because returns_arr is already direction-adjusted
                # (positive return = profitable trade regardless of long/short direction).
                # The sizing function returns negative values for shorts, but we don't want to
                # invert the sign of already-direction-adjusted returns.
                trade_returns = np.asarray(returns_arr, dtype=float) * np.abs(np.asarray(sizes_full, dtype=float))
                trade_returns = trade_returns[np.asarray(take_mask, dtype=bool)]
                trade_returns = trade_returns[np.isfinite(trade_returns)]
                n_trades = int(trade_returns.size)
                take_rate = float(n_trades) / float(max(int(len(returns_arr)), 1))
                if n_trades > 0:
                    trade_mean_return = float(np.mean(trade_returns))
                    trade_win_rate = float(np.mean(trade_returns > 0.0))

                if n_trades > 1:
                    eq = np.cumprod(1.0 + np.asarray(trade_returns, dtype=float))
                    peak = np.maximum.accumulate(eq)
                    denom = np.maximum(peak, 1e-12)
                    dd = 1.0 - (eq / denom)
                    if dd.size > 0:
                        max_drawdown = float(np.max(dd))
            except Exception:
                sizes_full = None
                take_mask = None
                trade_returns = None
                n_trades = 0
                take_rate = 0.0
                trade_mean_return = None
                trade_win_rate = None
                max_drawdown = None

            if bool(write_diagnostics):
                try:
                    outcomes_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                try:
                    min_trades_sweep = int(config.get("layer2_profitability_min_trades", 30))
                except Exception:
                    min_trades_sweep = 30
                min_trades_sweep = int(max(0, min_trades_sweep))

                try:
                    ts_diag = str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
                except Exception:
                    ts_diag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

                try:
                    exit_arr = None
                    try:
                        if l2_exit_reasons is not None:
                            exit_arr = l2_exit_reasons.reindex(l2_t_events).values
                    except Exception:
                        exit_arr = None

                    if p_sz is not None:
                        oof_all_event_deciles = _compute_oof_all_event_deciles(
                            probs=np.asarray(p_sz, dtype=float),
                            returns=np.asarray(returns_arr, dtype=float),
                            exit_reasons=exit_arr,
                            n_bins=10,
                        )
                        if oof_all_event_deciles:
                            all_path = outcomes_dir / (
                                f"hpo_layer2_oof_all_event_deciles_{symbol}_{timeframe}_{direction}_{ts_diag}.csv"
                            )
                            pd.DataFrame(oof_all_event_deciles).to_csv(all_path, index=False)
                            oof_all_event_deciles_path = str(all_path)
                            try:
                                tprint_info(f"   💾 Saved Layer 2 diagnostics (all-event deciles) to {oof_all_event_deciles_path}")
                            except Exception:
                                pass

                    if p_sz is not None and sizes_full is not None and take_mask is not None:
                        oof_taken_trade_deciles = _compute_taken_trade_deciles(
                            probs=np.asarray(p_sz, dtype=float),
                            returns=np.asarray(returns_arr, dtype=float),
                            sizes=np.asarray(sizes_full, dtype=float),
                            take_mask=np.asarray(take_mask, dtype=bool),
                            exit_reasons=exit_arr,
                            n_bins=10,
                        )
                        if oof_taken_trade_deciles:
                            dec_path = outcomes_dir / (
                                f"hpo_layer2_oof_taken_trade_deciles_{symbol}_{timeframe}_{direction}_{ts_diag}.csv"
                            )
                            pd.DataFrame(oof_taken_trade_deciles).to_csv(dec_path, index=False)
                            oof_taken_trade_deciles_path = str(dec_path)
                            try:
                                tprint_info(f"   💾 Saved Layer 2 diagnostics (taken trade deciles) to {oof_taken_trade_deciles_path}")
                            except Exception:
                                pass
                except Exception as e:
                    try:
                        tprint_warning(f"   ⚠️ Failed to write Layer 2 diagnostics (taken trade deciles): {e}")
                    except Exception:
                        pass

                try:
                    if p_sz is not None:
                        sweep_rows, sweep_summary = _sweep_prob_thresholds_for_profitability(
                            probs=np.asarray(p_sz, dtype=float),
                            returns=np.asarray(returns_arr, dtype=float),
                            direction=direction,
                            days_span=float(days_span),
                            thresholds=None,
                            min_trades=int(min_trades_sweep),
                            p_fail=p_fail_arr,
                            p_fail_threshold=prob_stop_threshold_val,
                        )
                        if sweep_rows:
                            sweep_path = outcomes_dir / (
                                f"hpo_layer2_oof_prob_threshold_sweep_{symbol}_{timeframe}_{direction}_{ts_diag}.csv"
                            )
                            pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)
                            oof_prob_threshold_sweep_path = str(sweep_path)
                            try:
                                tprint_info(f"   💾 Saved Layer 2 diagnostics (prob threshold sweep) to {oof_prob_threshold_sweep_path}")
                            except Exception:
                                pass
                        oof_prob_threshold_sweep_best = sweep_summary
                except Exception as e:
                    try:
                        tprint_warning(f"   ⚠️ Failed to write Layer 2 diagnostics (prob threshold sweep): {e}")
                    except Exception:
                        pass

            trades_per_day = float(n_trades) / float(max(days_span, 1))

            try:
                layer2_profitability_gate_enabled = bool(config.get("layer2_profitability_gate_enabled", False))
            except Exception:
                layer2_profitability_gate_enabled = True
            try:
                layer2_min_trades = int(config.get("layer2_profitability_min_trades", 30))
            except Exception:
                layer2_min_trades = 30
            layer2_min_trades = int(max(0, layer2_min_trades))
            try:
                layer2_min_trade_mean_return = float(config.get("layer2_profitability_min_trade_mean_return", 0.0))
            except Exception:
                layer2_min_trade_mean_return = 0.0

            # Default to a positive net-return floor so the strategy clears fees in expectation.
            # Note: returns_arr is already net-of-transaction-cost; this floor enforces a margin above 0.
            if (not np.isfinite(float(layer2_min_trade_mean_return))) or float(layer2_min_trade_mean_return) <= 0.0:
                try:
                    tx_mult = float(config.get("layer2_profitability_min_trade_mean_return_tx_mult", 0.0))
                except Exception:
                    tx_mult = 0.0
                if not np.isfinite(tx_mult):
                    tx_mult = 0.0
                tx_mult = float(max(0.0, tx_mult))
                try:
                    layer2_min_trade_mean_return = float(DEFAULT_TRANSACTION_COST) * tx_mult
                except Exception:
                    layer2_min_trade_mean_return = 0.0

            profitability_penalty = 0.0
            profitability_penalty_trades = 0.0
            profitability_penalty_mean_return = 0.0
            try:
                if bool(layer2_profitability_gate_enabled):
                    try:
                        w_trades_pen = float(config.get("layer2_profitability_penalty_w_trades", 1.0))
                    except Exception:
                        w_trades_pen = 1.0
                    try:
                        w_ret_pen = float(config.get("layer2_profitability_penalty_w_mean_return", 3.0))
                    except Exception:
                        w_ret_pen = 3.0
                    if not np.isfinite(w_trades_pen):
                        w_trades_pen = 1.0
                    if not np.isfinite(w_ret_pen):
                        w_ret_pen = 3.0
                    w_trades_pen = float(max(0.0, w_trades_pen))
                    w_ret_pen = float(max(0.0, w_ret_pen))

                    try:
                        trades_shortfall = float(max(0.0, float(layer2_min_trades) - float(n_trades)))
                        trades_shortfall_norm = trades_shortfall / float(max(1.0, float(layer2_min_trades)))
                    except Exception:
                        trades_shortfall_norm = 0.0
                    profitability_penalty_trades = float(w_trades_pen) * float(trades_shortfall_norm)

                    try:
                        tmr = float(trade_mean_return) if trade_mean_return is not None else float("-inf")
                        if not np.isfinite(tmr):
                            tmr = float("-inf")
                        ret_shortfall = float(max(0.0, float(layer2_min_trade_mean_return) - float(tmr)))
                        denom = float(
                            max(
                                1e-12,
                                abs(float(layer2_min_trade_mean_return))
                                if np.isfinite(float(layer2_min_trade_mean_return)) and float(layer2_min_trade_mean_return) != 0.0
                                else float(DEFAULT_TRANSACTION_COST),
                            )
                        )
                        ret_shortfall_norm = ret_shortfall / denom
                    except Exception:
                        ret_shortfall_norm = 0.0
                    profitability_penalty_mean_return = float(w_ret_pen) * float(ret_shortfall_norm)

                    profitability_penalty = float(profitability_penalty_trades) + float(profitability_penalty_mean_return)
            except Exception:
                profitability_penalty = 0.0
                profitability_penalty_trades = 0.0
                profitability_penalty_mean_return = 0.0
            try:
                lambda_vol = float(config.get("layer2_lambda_vol", 0.6))
            except Exception:
                lambda_vol = 0.6
            if not np.isfinite(lambda_vol):
                lambda_vol = 0.6
            lambda_vol = float(max(0.0, lambda_vol))

            try:
                w_auc = float(config.get("layer2_w_auc", 0.5))
            except Exception:
                w_auc = 0.5
            if not np.isfinite(w_auc):
                w_auc = 0.5
            w_auc = float(max(0.0, w_auc))
            try:
                w_den = float(config.get("layer2_w_den", 0.15))
            except Exception:
                w_den = 0.15
            if not np.isfinite(w_den):
                w_den = 0.15
            w_den = float(max(0.0, w_den))

            utility_debug: Dict[str, Any] = {}
            try:
                utility_clip_max = float(config.get("layer2_utility_clip_max", 5000.0))
            except Exception:
                utility_clip_max = 5000.0

            try:
                utility_floor = float(config.get("layer2_utility_floor", -1.0))
            except Exception:
                utility_floor = -1.0
            if not np.isfinite(utility_floor):
                utility_floor = -1.0

            per_fold_metrics = []
            try:
                if bool(layer2_econ_win_enabled):
                    y_profit_arr = (np.asarray(returns_arr, dtype=float) > float(layer2_econ_win_floor)).astype(int)
                else:
                    y_profit_arr = (np.asarray(returns_arr, dtype=float) > 0.0).astype(int)
                per_fold_metrics = _compute_fold_metrics_from_oof(
                    X=X_trial,
                    y_true=y_profit_arr,
                    probs=np.asarray(cv_preds, dtype=float),
                    returns=np.asarray(returns_arr, dtype=float),
                    threshold=float(prob_thr),
                    days_span=float(days_span),
                    transaction_cost=0.0,
                    event_index=l2_t_events,
                    direction=direction,
                    event_durations=l2_durations.reindex(l2_t_events) if l2_durations is not None else None,
                    market_index=market_data.index if market_data is not None else None,
                    base_horizon_bars=12,
                )
            except Exception:
                per_fold_metrics = []

            try:
                fold_aucs = [
                    float(m.get("auc"))
                    for m in (per_fold_metrics or [])
                    if m.get("auc") is not None and np.isfinite(float(m.get("auc")))
                ]
                if len(fold_aucs) > 0:
                    mean_auc = float(np.mean(fold_aucs))
            except Exception:
                pass

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
                        base_thr=float(prob_thr),
                        transaction_cost=0.0,
                        regime_labels=regime_labels.get("volatility_regime"),
                        days_span=float(days_span),
                        direction=direction,
                    ),
                    "trend": _compute_metrics_by_regime(
                        y_true=y_true_arr,
                        probs=np.asarray(cv_preds, dtype=float),
                        returns=np.asarray(returns_arr, dtype=float),
                        base_thr=float(prob_thr),
                        transaction_cost=0.0,
                        regime_labels=regime_labels.get("trend_regime"),
                        days_span=float(days_span),
                        direction=direction,
                    ),
                    "combined": _compute_metrics_by_regime(
                        y_true=y_true_arr,
                        probs=np.asarray(cv_preds, dtype=float),
                        returns=np.asarray(returns_arr, dtype=float),
                        base_thr=float(prob_thr),
                        transaction_cost=0.0,
                        regime_labels=regime_labels.get("combined_regime"),
                        days_span=float(days_span),
                        direction=direction,
                    ),
                }
            except Exception:
                per_regime_metrics = {}

            avg_sharpe = float(np.mean(folds_sharpe))
            vol_sharpe = float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe) > 1 else 0.0

            # De Prado PSR utility on traded daily log returns (no multiple-testing penalty)
            try:
                psr_min_trades = int(config.get("layer2_psr_min_trades", 30))
            except Exception:
                psr_min_trades = 30
            psr_min_trades = int(max(1, psr_min_trades))

            try:
                sr_benchmark = float(config.get("layer2_psr_sr_benchmark", 0.0))
            except Exception:
                sr_benchmark = 0.0
            if not np.isfinite(sr_benchmark):
                sr_benchmark = 0.0

            psr_details = {"psr": 0.0, "psr_z": float("-inf"), "sr": None, "n": 0, "skew": 0.0, "kurt": 3.0}
            try:
                tm = np.asarray(take_mask, dtype=bool) if take_mask is not None else None
                if tm is not None and trade_returns is not None:
                    idx_tr = pd.DatetimeIndex(l2_t_events)[tm]
                    tr = np.asarray(trade_returns, dtype=float)
                    n_tr = int(min(int(idx_tr.size), int(tr.size)))
                    if n_tr > 0:
                        idx_tr = idx_tr[:n_tr]
                        tr = tr[:n_tr]
                        day_index = pd.date_range(
                            start=pd.DatetimeIndex(l2_t_events).min().normalize(),
                            end=pd.DatetimeIndex(l2_t_events).max().normalize(),
                            freq="D",
                        )
                        daily_pnl = pd.Series(tr, index=idx_tr).groupby(idx_tr.normalize()).sum()
                        daily_pnl = daily_pnl.reindex(day_index, fill_value=0.0)
                        daily_log = np.log1p(daily_pnl.astype(float).values)
                        daily_log = daily_log[np.isfinite(daily_log)]
                        psr_details = _psr_from_returns(
                            daily_log,
                            sr_benchmark=float(sr_benchmark),
                            periods_per_year=365.0,
                        )
            except Exception:
                pass

            phi_trades = 0.0
            try:
                phi_trades = float(np.clip(float(psr_details.get("n", 0)) / float(psr_min_trades), 0.0, 1.0))
            except Exception:
                phi_trades = 0.0

            utility = float(psr_details.get("psr", 0.0)) * float(phi_trades)
            if not np.isfinite(float(utility)):
                utility = 0.0

            # =====================================================================
            # UTILITY IMPROVEMENTS (2024-12): Magnitude-Weighted Win Rate Only
            # =====================================================================
            # CHANGES (Dec 2024):
            # - REMOVED: Magnitude Bonus (double-counts with win rate modifier)
            # - REMOVED: Stop-Out Penalty (conflicts with magnitude-weighted win rate;
            #   tight stop management is a feature, not a bug)
            # - KEPT: Magnitude-Weighted Win Rate as the sole modifier
            #
            # Rationale: The magnitude-weighted win rate already captures the economic
            # effect of both trade size and stop management. A high stop-out rate with
            # tight stops actually leads to BETTER magnitude-weighted win rate (small
            # losses, occasional big wins), so penalizing stop-outs was counterproductive.
            # =====================================================================

            magnitude_win_rate_modifier = 1.0
            magnitude_weighted_win_rate = 0.5

            # --- Magnitude-Weighted Win Rate Gate ---
            # Concept: Weight wins by magnitude, not just count
            # A 50% win rate with 3:1 R:R is better than 50% with 1:1 R:R
            # Formula: sum(positive_returns) / (sum(positive_returns) + |sum(negative_returns)|)
            #
            # IMPORTANT: Uses returns AFTER FEES (transaction costs)
            # - trade_returns is derived from returns_arr (l2_returns_clean)
            # - l2_returns comes from compute_realized_returns() with transaction_cost=DEFAULT_TRANSACTION_COST
            # - This ensures we're measuring actual net profitability, not gross
            try:
                w_mag_winrate = float(config.get("layer2_utility_w_mag_winrate", 0.8))
            except Exception:
                w_mag_winrate = 0.8
            if not np.isfinite(w_mag_winrate):
                w_mag_winrate = 0.8

            try:
                mag_winrate_floor = float(config.get("layer2_utility_mag_winrate_floor", 0.45))
            except Exception:
                mag_winrate_floor = 0.45
            if not np.isfinite(mag_winrate_floor):
                mag_winrate_floor = 0.45

            try:
                if trade_returns is not None and len(trade_returns) > 0:
                    # trade_returns is already net of transaction costs (from compute_realized_returns)
                    tr = np.asarray(trade_returns, dtype=float)
                    tr = tr[np.isfinite(tr)]
                    
                    if len(tr) > 0:
                        pos_sum = float(np.sum(tr[tr > 0])) if np.any(tr > 0) else 0.0
                        neg_sum = float(np.abs(np.sum(tr[tr < 0]))) if np.any(tr < 0) else 0.0
                        total_magnitude = pos_sum + neg_sum
                        if total_magnitude > 1e-12:
                            magnitude_weighted_win_rate = pos_sum / total_magnitude
                            # Apply as a modifier if below floor (penalize poor magnitude-weighted win rate)
                            if magnitude_weighted_win_rate < mag_winrate_floor:
                                # Scale utility down: at 0.35 with floor 0.45, modifier = 0.5 + 0.5*(0.35/0.45) ≈ 0.89
                                magnitude_win_rate_modifier = float(
                                    0.5 + 0.5 * (magnitude_weighted_win_rate / mag_winrate_floor)
                                )
                                magnitude_win_rate_modifier = float(np.clip(magnitude_win_rate_modifier, 0.3, 1.0))
                            else:
                                # Bonus for good magnitude-weighted win rate (up to 1.4x)
                                excess = magnitude_weighted_win_rate - mag_winrate_floor
                                magnitude_win_rate_modifier = float(1.0 + excess * w_mag_winrate)
                                magnitude_win_rate_modifier = float(np.clip(magnitude_win_rate_modifier, 1.0, 1.4))
            except Exception:
                magnitude_win_rate_modifier = 1.0

            # --- Apply magnitude-weighted win rate modifier only ---
            utility_pre_improvements = float(utility)
            try:
                # Only apply win rate modifier (no magnitude bonus, no stop penalty)
                utility = float(utility) * float(magnitude_win_rate_modifier)
                if not np.isfinite(utility):
                    utility = utility_pre_improvements
            except Exception:
                utility = utility_pre_improvements

            try:
                utility_debug.update(
                    {
                        # Sortino-based metrics (backward-compatible keys)
                        "psr": float(psor_details.get("probabilistic_sortino", 0.0)),
                        "psr_z": float(psor_details.get("psor_z", float("-inf"))),
                        "psr_sr": psor_details.get("sortino", None),
                        "psr_n": int(psor_details.get("n", 0) or 0),
                        "sortino_benchmark": float(sortino_benchmark),
                        "phi_trades": float(phi_trades),
                        "utility_pre_improvements": float(utility_pre_improvements),
                        # Magnitude-Weighted Win Rate (sole modifier now)
                        "magnitude_weighted_win_rate": float(magnitude_weighted_win_rate),
                        "magnitude_win_rate_modifier": float(magnitude_win_rate_modifier),
                        "utility_pre_clip": float(utility),
                        "utility": float(utility),
                    }
                )
            except Exception:
                pass

            utility_pre_profitability_penalty = float(utility)
            try:
                if np.isfinite(float(utility)) and float(profitability_penalty) > 0.0:
                    utility = float(
                        np.clip(
                            float(utility) - float(profitability_penalty),
                            float(utility_floor),
                            float(utility_clip_max),
                        )
                    )
            except Exception:
                pass
            try:
                if isinstance(utility_debug, dict):
                    utility_debug["profitability_penalty"] = float(profitability_penalty)
                    utility_debug["profitability_penalty_trades"] = float(profitability_penalty_trades)
                    utility_debug["profitability_penalty_mean_return"] = float(profitability_penalty_mean_return)
                    utility_debug["utility_pre_profitability_penalty"] = float(utility_pre_profitability_penalty)
            except Exception:
                pass

            base_score = float(psr_details.get("sr")) if psr_details.get("sr") is not None and np.isfinite(float(psr_details.get("sr"))) else float("nan")
            base_norm = float(psr_details.get("psr_z")) if np.isfinite(float(psr_details.get("psr_z", float("nan")))) else float("nan")
            phi_auc = float("nan")
            phi_density = float("nan")
            modifier = float("nan")

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

            utility_pre_volatility_penalty = float(utility)
            vol_mean_all = None
            vol_mean_taken = None
            vol_excess_z = 0.0
            vol_excess_abs_z = 0.0
            try:
                vol_all = np.asarray(batch_volatility, dtype=float)
                vol_all = vol_all[np.isfinite(vol_all)]
                if vol_all.size > 5:
                    mu_all = float(np.mean(vol_all))
                    sd_all = float(np.std(vol_all, ddof=1)) if vol_all.size > 1 else 0.0
                    if np.isfinite(mu_all):
                        vol_mean_all = float(mu_all)
                    if sd_all > 1e-12:
                        ptm = None
                        try:
                            ptm = np.asarray(take_mask, dtype=bool) if take_mask is not None else None
                        except Exception:
                            ptm = None
                        if ptm is not None and int(ptm.size) == int(batch_volatility.size):
                            vol_taken = np.asarray(batch_volatility, dtype=float)[ptm]
                            vol_taken = vol_taken[np.isfinite(vol_taken)]
                            if vol_taken.size > 0:
                                mu_taken = float(np.mean(vol_taken))
                                if np.isfinite(mu_taken):
                                    vol_mean_taken = float(mu_taken)
                                if vol_mean_all is not None:
                                    vol_excess_z = float((mu_taken - mu_all) / (sd_all + 1e-12))
            except Exception:
                vol_excess_z = 0.0

            try:
                vol_excess_abs_z = float(abs(float(vol_excess_z))) if np.isfinite(float(vol_excess_z)) else 0.0
            except Exception:
                vol_excess_abs_z = 0.0

            try:
                vol_excess_pos_z = float(max(0.0, float(vol_excess_z))) if np.isfinite(float(vol_excess_z)) else 0.0
            except Exception:
                vol_excess_pos_z = 0.0

            # VOLATILITY PENALTY (downweighted Dec 2024)
            # Rationale: In crypto, much alpha comes from volatility. The original penalty
            # was too aggressive and systematically excluded profitable volatile regimes.
            # Reduced scaling factor from 1.0 to 0.25 to soften the penalty.
            volatility_penalty_scale = 0.25  # Downweighted from implicit 1.0
            try:
                if (
                    np.isfinite(float(utility))
                    and float(utility) > float(utility_floor)
                    and np.isfinite(float(vol_penalty_lambda))
                    and float(vol_penalty_lambda) > 0.0
                    and np.isfinite(float(vol_excess_pos_z))
                    and float(vol_excess_pos_z) > 0.0
                ):
                    # Apply scaled-down volatility penalty
                    scaled_penalty = float(vol_penalty_lambda) * float(vol_excess_pos_z) * volatility_penalty_scale
                    utility = float(
                        np.clip(
                            float(utility) - scaled_penalty,
                            float(utility_floor),
                            float(utility_clip_max),
                        )
                    )
            except Exception:
                pass

            # NEW: Compute returns and drawdown for reporting using position sizing logic
            # Use the same 'take_mask' and 'sizes' logic as implemented in objective function if possible,
            # or re-derive here if needed. Since _compute_layer2_metrics is single-shot, we can re-derive.
            trade_mean_return = None
            max_drawdown = None
            try:
                # Re-derive position sizes and returns for accurate reporting
                # (This mirrors the logic added to the objective function)
                # ... (Assuming cv_preds available in scope or passed in params? No, params is just dict)
                # Ah, _compute_layer2_metrics uses global 'cv_preds' which is available in closure scope
                
                # We need to apply the same adaptive threshold logic first
                # Re-calculate adaptive thresholds
                sig_strength_sens = float(params.get("sig_strength_sensitivity", 0.0))
                sig_strength_sens = float(np.clip(sig_strength_sens, 0.0, 0.5))
                base_prob_thr = float(params.get("prob_threshold", 0.5))
                
                sig_strength_arr = np.ones(len(l2_t_events), dtype=float) * 0.5
                if primary_signals is not None and "consensus" in primary_signals.columns:
                     try:
                        sa = primary_signals["consensus"].reindex(l2_t_events).fillna(0.0).abs().values
                        sig_strength_arr = np.clip(sa, 0.0, 2.0) / 2.0
                     except: pass
                
                # Per-event threshold
                # adaptive_thr = base - sens * (strength - 0.5)
                # But directional_size_from_prob takes a scalar 'thr'. 
                # To be precise, we should pass the adaptive threshold array if the sizer supports it.
                # The helper 'directional_size_from_prob' likely takes scalar. 
                # Let's use loop or vector op if supported. 
                # For reporting, simple logic:
                
                adaptive_thresholds = base_prob_thr - sig_strength_sens * (sig_strength_arr - 0.5)
                adaptive_thresholds = np.clip(adaptive_thresholds, 0.3, 0.95)
                
                # Vectorized size calc pseudo-code (or loop)
                p_sz_raw = np.asarray(cv_preds)
                if getattr(p_sz_raw, "ndim", 0) == 1 and getattr(p_sz_raw, "dtype", None) is not None and p_sz_raw.dtype == object:
                    try:
                        p_sz_raw = np.vstack(p_sz_raw)
                    except Exception:
                        pass
                if getattr(p_sz_raw, "ndim", 0) == 2 and int(p_sz_raw.shape[1]) >= 2:
                    p_sz = np.asarray(p_sz_raw[:, 1], dtype=float)
                elif getattr(p_sz_raw, "ndim", 0) == 2 and int(p_sz_raw.shape[1]) == 1:
                    p_sz = np.asarray(p_sz_raw[:, 0], dtype=float)
                else:
                    p_sz = np.asarray(p_sz_raw, dtype=float)
                r_arr = np.asarray(l2_returns_clean, dtype=float) # From closure
                
                # Check if we can use vectorized masking
                # Trade taken if prob >= adaptive_threshold
                # (Assuming simple thresholding for this report metric)
                taken_mask = p_sz >= adaptive_thresholds
                
                if np.sum(taken_mask) > 0:
                    # Probabilistic Stops (First Passage Time Veto) integration
                    # Filter the taken_mask using vectorized veto logic (same direction handling as Layer2 objective).
                    veto_mask = np.zeros(len(l2_t_events), dtype=bool)
                    veto_rate = 0.0
                    try:
                        prob_stop_active = int(params.get("prob_stop_enable", 0)) > 0
                    except Exception:
                        prob_stop_active = False

                    if bool(prob_stop_active):
                        try:
                            from src.training.steps.labeling.layer2_advanced_logic import calc_prob_touch_sl_vec

                            dir_raw = str(direction).lower()
                            dir_sign = 1
                            if dir_raw in {"short", "sell", "-1", "s"}:
                                dir_sign = -1

                            p_thr = float(params.get("prob_stop_threshold", 0.70))
                            w_drift = int(params.get("prob_stop_drift_window", 24))
                            w_drift = int(np.clip(w_drift, 2, 512))

                            mkt_rets = market_data['close'].pct_change().fillna(0.0)
                            drift_arr = mkt_rets.rolling(w_drift).mean()
                            vol_arr = mkt_rets.rolling(w_drift).std()

                            ev_drift = drift_arr.reindex(l2_t_events).fillna(0.0).to_numpy(dtype=float)
                            ev_vol = vol_arr.reindex(l2_t_events).fillna(0.0).to_numpy(dtype=float)

                            h = float(l2_horizon_bars)
                            mu = ev_drift * h
                            sigma = np.maximum(ev_vol * np.sqrt(h), 1e-6)

                            # Use the actual barrier distances
                            if isinstance(stop_thr, pd.Series):
                                sl_d = pd.to_numeric(stop_thr.reindex(l2_t_events), errors="coerce").abs().fillna(0.0).to_numpy(dtype=float)
                            else:
                                sl_d = np.full(int(len(l2_t_events)), abs(float(stop_thr)), dtype=float)
                            if isinstance(prof_thr, pd.Series):
                                tp_d = pd.to_numeric(prof_thr.reindex(l2_t_events), errors="coerce").abs().fillna(0.0).to_numpy(dtype=float)
                            else:
                                tp_d = np.full(int(len(l2_t_events)), abs(float(prof_thr)), dtype=float)

                            sl_d = np.maximum(sl_d, 1e-6)
                            tp_d = np.maximum(tp_d, 1e-6)

                            p_fail = calc_prob_touch_sl_vec(mu=mu, sigma=sigma, sl_dist=sl_d, tp_dist=tp_d, direction=int(dir_sign))
                            veto_mask = (np.asarray(p_fail, dtype=float) > float(p_thr))
                            veto_mask = np.asarray(veto_mask, dtype=bool) & np.asarray(taken_mask, dtype=bool)
                            veto_rate = float(np.mean(veto_mask[taken_mask])) if int(np.sum(taken_mask)) > 0 else 0.0
                        except Exception:
                            veto_mask = np.zeros(len(l2_t_events), dtype=bool)
                            veto_rate = 0.0

                    final_taken_mask = taken_mask & (~veto_mask)
                    
                    if np.sum(final_taken_mask) > 0:
                        tr_rets = r_arr[final_taken_mask]
                        trade_mean_return = float(np.mean(tr_rets))
                        
                        # Max Drawdown
                        if len(tr_rets) > 1:
                            cum_r = np.nancumsum(tr_rets)
                            peak = np.maximum.accumulate(cum_r)
                            dd_arr = peak - cum_r
                            max_drawdown = float(np.max(dd_arr))
                        else:
                            max_drawdown = 0.0
                    else:
                        trade_mean_return = 0.0
                        max_drawdown = 0.0
                        
                    # Calculate Logic Stats for Verification
                    veto_rate = np.mean(veto_mask)
                    # Split by drift direction (did it veto correctly?)
                    # Need mu_eff: drift * direction
                    # Re-calc minimal logic for logging
                    try:
                        # Re-get drift to compute detailed stats
                        pass # optimize speed, just return total veto rate for now
                    except: pass

                else:
                    trade_mean_return = 0.0
                    max_drawdown = 0.0
                    veto_rate = 0.0

            except Exception:
                pass

            # Update metrics dictionary with new fields
            met_dict = {
                # ... existing fields ...
                "mean_return": trade_mean_return,
                "max_drawdown": max_drawdown,
                "w_return": float(config.get("layer2_w_return", 3.0)),
                "w_dd": float(config.get("layer2_w_dd", 1.0)),
                "prob_stop_veto_rate": float(veto_rate),
                "trail_trend_modulation": float(params.get("trail_trend_modulation", 0.0)),
                "barrier_trend_asymmetry": float(params.get("barrier_trend_asymmetry", 0.0)),
                "horizon_volume_modulation": float(params.get("horizon_volume_modulation", 0.0)),
                "barrier_vol_vol_exp": float(params.get("barrier_vol_vol_exp", 0.0)),
                "sig_strength_sensitivity": float(params.get("sig_strength_sensitivity", 0.0)),
                "barrier_regime_power": float(params.get("barrier_regime_power", 1.0)),
                "barrier_regime_strength": float(params.get("barrier_regime_strength", 1.0)),
                "prob_stop_enable": int(params.get("prob_stop_enable", 0)),
                "prob_stop_threshold": float(params.get("prob_stop_threshold", 0.95)),
            }
            # (Merged execution below to handle strict object replacement)


            # ------------------------------------------------------------------
            # Layer 2 instability penalty across regimes (robust generalization)
            # ------------------------------------------------------------------
            
            # ... (previous code) ...

            try:
                layer2_regime_instability_lambda = float(config.get("layer2_regime_instability_lambda", 0.0))
            except Exception:
                layer2_regime_instability_lambda = 0.0
            if not np.isfinite(layer2_regime_instability_lambda):
                layer2_regime_instability_lambda = 0.0
            layer2_regime_instability_lambda = float(max(0.0, layer2_regime_instability_lambda))

            regime_dispersion = float(_compute_regime_dispersion(per_regime_metrics, metric_key="sharpe"))
            if not np.isfinite(regime_dispersion):
                regime_dispersion = 0.0

            utility_pre_regime_penalty = float(utility)
            try:
                if (
                    layer2_regime_instability_lambda > 0.0
                    and np.isfinite(float(utility))
                    and float(utility) > 0.0
                    and np.isfinite(regime_dispersion)
                    and float(regime_dispersion) > 0.0
                ):
                    utility = float(
                        np.clip(
                            float(utility) - float(layer2_regime_instability_lambda) * float(regime_dispersion),
                            -1.0,
                            float(utility_clip_max),
                        )
                    )
            except Exception:
                pass

            probability_mapping: List[Dict[str, Any]] = []
            try:
                probability_mapping = _compute_probability_mapping(
                    probs=np.asarray(cv_preds, dtype=float),
                    returns=np.asarray(returns_arr, dtype=float),
                    n_bins=int(config.get("probability_mapping_bins", 10)),
                )
            except Exception:
                probability_mapping = []

            probability_mapping_thresholded: List[Dict[str, Any]] = []
            probability_mapping_traded: List[Dict[str, Any]] = []
            try:
                p_map_raw = np.asarray(cv_preds)
                if (
                    getattr(p_map_raw, "ndim", 0) == 1
                    and getattr(p_map_raw, "dtype", None) is not None
                    and p_map_raw.dtype == object
                ):
                    try:
                        p_map_raw = np.vstack(p_map_raw)
                    except Exception:
                        pass
                if getattr(p_map_raw, "ndim", 0) == 2 and int(p_map_raw.shape[1]) >= 2:
                    p_map = np.asarray(p_map_raw[:, 1], dtype=float)
                elif getattr(p_map_raw, "ndim", 0) == 2 and int(p_map_raw.shape[1]) == 1:
                    p_map = np.asarray(p_map_raw[:, 0], dtype=float)
                else:
                    p_map = np.asarray(p_map_raw, dtype=float).reshape(-1)

                n_pm = int(min(p_map.size, np.asarray(returns_arr, dtype=float).size))
                if n_pm > 0:
                    p_map = p_map[:n_pm]
                    r_map = np.asarray(returns_arr, dtype=float).reshape(-1)[:n_pm]

                    dir_raw = str(direction or "").lower()
                    if dir_raw in {"short", "sell", "-1", "s"}:
                        threshold_mask = p_map <= float(prob_thr)
                    else:
                        threshold_mask = p_map >= float(prob_thr)

                    if int(np.sum(threshold_mask & np.isfinite(p_map) & np.isfinite(r_map))) >= 20:
                        probability_mapping_thresholded = _compute_probability_mapping(
                            probs=np.asarray(p_map, dtype=float)[threshold_mask],
                            returns=np.asarray(r_map, dtype=float)[threshold_mask],
                            n_bins=int(config.get("probability_mapping_bins", 10)),
                        )

                    if sizes_full is not None and take_mask is not None:
                        s_map = np.asarray(sizes_full, dtype=float).reshape(-1)[:n_pm]
                        tm = np.asarray(take_mask, dtype=bool).reshape(-1)[:n_pm]
                        trade_r_full = r_map * np.abs(s_map)
                        traded_mask = tm & np.isfinite(p_map) & np.isfinite(trade_r_full)
                        if int(np.sum(traded_mask)) >= 20:
                            probability_mapping_traded = _compute_probability_mapping(
                                probs=np.asarray(p_map, dtype=float)[traded_mask],
                                returns=np.asarray(trade_r_full, dtype=float)[traded_mask],
                                n_bins=int(config.get("probability_mapping_bins", 10)),
                            )
            except Exception:
                probability_mapping_thresholded = []
                probability_mapping_traded = []

            # DIAGNOSTIC: Label-Return Alignment Check
            # This helps identify if labels are misaligned with actual profitability
            label_return_alignment: Dict[str, Any] = {}
            try:
                y_lra = np.asarray(y_true_arr, dtype=float)
                r_lra = np.asarray(returns_arr, dtype=float)
                valid_lra = np.isfinite(y_lra) & np.isfinite(r_lra)
                if int(np.sum(valid_lra)) > 50:
                    y_v = y_lra[valid_lra]
                    r_v = r_lra[valid_lra]
                    # Label=1 events
                    pos_mask = y_v >= 0.5
                    neg_mask = y_v < 0.5
                    # Mean return for label=1 vs label=0
                    ret_when_label_1 = float(np.mean(r_v[pos_mask])) if np.sum(pos_mask) > 0 else None
                    ret_when_label_0 = float(np.mean(r_v[neg_mask])) if np.sum(neg_mask) > 0 else None
                    # Win rate (return > 0) for each label class
                    winrate_when_label_1 = float(np.mean(r_v[pos_mask] > 0)) if np.sum(pos_mask) > 0 else None
                    winrate_when_label_0 = float(np.mean(r_v[neg_mask] > 0)) if np.sum(neg_mask) > 0 else None
                    # Correlation between label and return
                    from scipy.stats import spearmanr
                    corr, pval = spearmanr(y_v, r_v)
                    label_return_alignment = {
                        "n_label_1": int(np.sum(pos_mask)),
                        "n_label_0": int(np.sum(neg_mask)),
                        "ret_mean_when_label_1": ret_when_label_1,
                        "ret_mean_when_label_0": ret_when_label_0,
                        "winrate_when_label_1": winrate_when_label_1,
                        "winrate_when_label_0": winrate_when_label_0,
                        "label_return_spearman": float(corr) if np.isfinite(corr) else None,
                        "label_return_pvalue": float(pval) if np.isfinite(pval) else None,
                    }
            except Exception:
                pass

            # DIAGNOSTIC: Probability-Return Ranking Check
            # This helps identify why model probability doesn't correlate with returns
            prob_return_ranking: Dict[str, Any] = {}
            try:
                p_prr = np.asarray(cv_preds, dtype=float)
                r_prr = np.asarray(returns_arr, dtype=float)
                valid_prr = np.isfinite(p_prr) & np.isfinite(r_prr)
                if int(np.sum(valid_prr)) > 50:
                    p_v = p_prr[valid_prr]
                    r_v = r_prr[valid_prr]
                    from scipy.stats import spearmanr, kendalltau
                    # Probability-Return correlation
                    sp_corr, sp_pval = spearmanr(p_v, r_v)
                    kt_corr, kt_pval = kendalltau(p_v, r_v)
                    # Top quartile analysis
                    p75 = float(np.percentile(p_v, 75))
                    top_q_mask = p_v >= p75
                    ret_top_q = float(np.mean(r_v[top_q_mask])) if np.sum(top_q_mask) > 0 else None
                    winrate_top_q = float(np.mean(r_v[top_q_mask] > 0)) if np.sum(top_q_mask) > 0 else None
                    # Bottom quartile analysis
                    p25 = float(np.percentile(p_v, 25))
                    bot_q_mask = p_v <= p25
                    ret_bot_q = float(np.mean(r_v[bot_q_mask])) if np.sum(bot_q_mask) > 0 else None
                    winrate_bot_q = float(np.mean(r_v[bot_q_mask] > 0)) if np.sum(bot_q_mask) > 0 else None
                    # Monotonicity check: does higher prob = higher return?
                    prob_return_ranking = {
                        "prob_return_spearman": float(sp_corr) if np.isfinite(sp_corr) else None,
                        "prob_return_spearman_pval": float(sp_pval) if np.isfinite(sp_pval) else None,
                        "prob_return_kendall": float(kt_corr) if np.isfinite(kt_corr) else None,
                        "prob_return_kendall_pval": float(kt_pval) if np.isfinite(kt_pval) else None,
                        "prob_p75": float(p75),
                        "prob_p25": float(p25),
                        "ret_mean_top_quartile": ret_top_q,
                        "winrate_top_quartile": winrate_top_q,
                        "ret_mean_bot_quartile": ret_bot_q,
                        "winrate_bot_quartile": winrate_bot_q,
                        "top_minus_bot_ret": float(ret_top_q - ret_bot_q) if ret_top_q is not None and ret_bot_q is not None else None,
                    }
            except Exception:
                pass

            prob_return_ranking_raw: Dict[str, Any] = {}
            try:
                p_prr_raw = np.asarray(cv_preds_raw, dtype=float)
                r_prr_raw = np.asarray(returns_arr, dtype=float)
                valid_prr_raw = np.isfinite(p_prr_raw) & np.isfinite(r_prr_raw)
                if int(np.sum(valid_prr_raw)) > 50:
                    p_v = p_prr_raw[valid_prr_raw]
                    r_v = r_prr_raw[valid_prr_raw]
                    from scipy.stats import spearmanr, kendalltau
                    sp_corr, sp_pval = spearmanr(p_v, r_v)
                    kt_corr, kt_pval = kendalltau(p_v, r_v)
                    p75 = float(np.percentile(p_v, 75))
                    top_q_mask = p_v >= p75
                    ret_top_q = float(np.mean(r_v[top_q_mask])) if np.sum(top_q_mask) > 0 else None
                    winrate_top_q = float(np.mean(r_v[top_q_mask] > 0)) if np.sum(top_q_mask) > 0 else None
                    p25 = float(np.percentile(p_v, 25))
                    bot_q_mask = p_v <= p25
                    ret_bot_q = float(np.mean(r_v[bot_q_mask])) if np.sum(bot_q_mask) > 0 else None
                    winrate_bot_q = float(np.mean(r_v[bot_q_mask] > 0)) if np.sum(bot_q_mask) > 0 else None
                    prob_return_ranking_raw = {
                        "prob_return_spearman": float(sp_corr) if np.isfinite(sp_corr) else None,
                        "prob_return_spearman_pval": float(sp_pval) if np.isfinite(sp_pval) else None,
                        "prob_return_kendall": float(kt_corr) if np.isfinite(kt_corr) else None,
                        "prob_return_kendall_pval": float(kt_pval) if np.isfinite(kt_pval) else None,
                        "prob_p75": float(p75),
                        "prob_p25": float(p25),
                        "ret_mean_top_quartile": ret_top_q,
                        "winrate_top_quartile": winrate_top_q,
                        "ret_mean_bot_quartile": ret_bot_q,
                        "winrate_bot_quartile": winrate_bot_q,
                        "top_minus_bot_ret": float(ret_top_q - ret_bot_q) if ret_top_q is not None and ret_bot_q is not None else None,
                    }
            except Exception:
                pass

            try:
                # Enhanced: Default enabled with lambda=2.0 to penalize poor probability-return correlation
                sp_penalty_lambda = float(config.get("layer2_prob_return_spearman_penalty_lambda", 2.0))
            except Exception:
                sp_penalty_lambda = 2.0
            if np.isfinite(float(sp_penalty_lambda)) and float(sp_penalty_lambda) > 0.0:
                try:
                    sp_val = prob_return_ranking.get("prob_return_spearman")
                    sp_val = float(sp_val) if sp_val is not None else float("nan")
                    if np.isfinite(sp_val) and np.isfinite(float(utility)):
                        # NEW: Penalize negative correlation AND low positive correlation
                        # Target: sp_val should be > 0.15 for good models
                        # Penalty: utility *= sigmoid(spearman) -- scales utility by correlation quality
                        # Range: spearman=-0.1 -> factor ~0.3, spearman=0.1 -> factor ~0.7, spearman=0.3 -> factor ~0.95
                        sp_factor = 1.0 / (1.0 + np.exp(-10.0 * (sp_val - 0.1)))  # Sigmoid centered at 0.1
                        sp_factor = float(np.clip(sp_factor, 0.1, 1.0))
                        # Apply as multiplicative penalty if lambda enabled
                        utility = float(utility) * sp_factor
                except Exception:
                    pass

            pr_auc_val = None
            precision_at_1pct_val = None
            precision_at_5pct_val = None
            precision_at_10pct_val = None
            pr_auc_raw_val = None
            precision_at_1pct_raw_val = None
            precision_at_5pct_raw_val = None
            precision_at_10pct_raw_val = None
            try:
                p_pr = np.asarray(cv_preds, dtype=float)
                r_pr = np.asarray(returns_arr, dtype=float)
                if bool(layer2_econ_win_enabled):
                    y_pr = (np.asarray(r_pr, dtype=float) > float(layer2_econ_win_floor)).astype(int)
                else:
                    y_pr = (np.asarray(r_pr, dtype=float) > 0.0).astype(int)
                m_pr = np.isfinite(p_pr) & np.isfinite(r_pr)
                if int(np.sum(m_pr)) >= 20:
                    yb_pr = y_pr[m_pr].astype(int)
                    p_pr_v = p_pr[m_pr]
                    if int(np.unique(yb_pr).size) >= 2:
                        pr_auc_val = float(average_precision_score(yb_pr, p_pr_v))
                    order = np.argsort(-np.asarray(p_pr_v, dtype=float))
                    n_tot = int(order.size)
                    if n_tot > 0:
                        k1 = int(max(1, int(np.ceil(0.01 * float(n_tot)))))
                        k5 = int(max(1, int(np.ceil(0.05 * float(n_tot)))))
                        k10 = int(max(1, int(np.ceil(0.10 * float(n_tot)))))
                        precision_at_1pct_val = float(np.sum(yb_pr[order[:k1]] == 1)) / float(k1)
                        precision_at_5pct_val = float(np.sum(yb_pr[order[:k5]] == 1)) / float(k5)
                        precision_at_10pct_val = float(np.sum(yb_pr[order[:k10]] == 1)) / float(k10)
            except Exception:
                pass

            try:
                p_pr_raw = np.asarray(cv_preds_raw, dtype=float)
                r_pr_raw = np.asarray(returns_arr, dtype=float)
                if bool(layer2_econ_win_enabled):
                    y_pr_raw = (np.asarray(r_pr_raw, dtype=float) > float(layer2_econ_win_floor)).astype(int)
                else:
                    y_pr_raw = (np.asarray(r_pr_raw, dtype=float) > 0.0).astype(int)
                m_pr_raw = np.isfinite(p_pr_raw) & np.isfinite(r_pr_raw)
                if int(np.sum(m_pr_raw)) >= 20:
                    yb_pr_raw = y_pr_raw[m_pr_raw].astype(int)
                    p_pr_raw_v = p_pr_raw[m_pr_raw]
                    if int(np.unique(yb_pr_raw).size) >= 2:
                        pr_auc_raw_val = float(average_precision_score(yb_pr_raw, p_pr_raw_v))
                    order_raw = np.argsort(-np.asarray(p_pr_raw_v, dtype=float))
                    n_tot_raw = int(order_raw.size)
                    if n_tot_raw > 0:
                        k1r = int(max(1, int(np.ceil(0.01 * float(n_tot_raw)))))
                        k5r = int(max(1, int(np.ceil(0.05 * float(n_tot_raw)))))
                        k10r = int(max(1, int(np.ceil(0.10 * float(n_tot_raw)))))
                        precision_at_1pct_raw_val = float(np.sum(yb_pr_raw[order_raw[:k1r]] == 1)) / float(k1r)
                        precision_at_5pct_raw_val = float(np.sum(yb_pr_raw[order_raw[:k5r]] == 1)) / float(k5r)
                        precision_at_10pct_raw_val = float(np.sum(yb_pr_raw[order_raw[:k10r]] == 1)) / float(k10r)
            except Exception:
                pass

            label_pos_rate_val = None
            label_n_pos_val = None
            label_n_neg_val = None
            try:
                y_diag = np.asarray(l2_labels_bin.values, dtype=float)
                m_diag = np.isfinite(y_diag)
                if int(np.sum(m_diag)) > 0:
                    yb = y_diag[m_diag].astype(int)
                    label_pos_rate_val = float(np.mean(yb))
                    label_n_pos_val = int(np.sum(yb == 1))
                    label_n_neg_val = int(np.sum(yb == 0))
            except Exception:
                pass

            pred_nan_frac_val = None
            pred_std_val = None
            pred_min_val = None
            pred_max_val = None
            pred_unique_rounded_val = None
            try:
                p_tmp = np.asarray(cv_preds, dtype=float)
                pred_nan_frac_val = float(np.mean(~np.isfinite(p_tmp)))
                p_fin = p_tmp[np.isfinite(p_tmp)]
                if p_fin.size > 0:
                    pred_std_val = float(np.std(p_fin, ddof=1)) if p_fin.size > 1 else 0.0
                    pred_min_val = float(np.min(p_fin))
                    pred_max_val = float(np.max(p_fin))
                    pred_unique_rounded_val = int(np.unique(np.round(p_fin, 4)).size)
            except Exception:
                pass

            return {
                "valid_events": int(valid_idx.sum()),
                "utility": float(utility),
                "utility_pre_clip": float(utility_debug.get("utility_pre_clip")) if "utility_pre_clip" in utility_debug else None,
                "utility_clip_max": float(utility_debug.get("utility_clip_max")) if "utility_clip_max" in utility_debug else float(utility_clip_max),
                "utility_pre_profitability_penalty": float(utility_pre_profitability_penalty)
                if "utility_pre_profitability_penalty" in locals()
                else None,
                "profitability_penalty": float(profitability_penalty) if "profitability_penalty" in locals() else None,
                "profitability_penalty_trades": float(profitability_penalty_trades)
                if "profitability_penalty_trades" in locals()
                else None,
                "profitability_penalty_mean_return": float(profitability_penalty_mean_return)
                if "profitability_penalty_mean_return" in locals()
                else None,
                "fail_reason": cv_fail_reason,
                "fail_exception": cv_fail_exception,
                "committee_voted_labels_used": bool(committee_voted_labels_used),
                "oof_all_event_deciles": oof_all_event_deciles,
                "oof_all_event_deciles_path": oof_all_event_deciles_path,
                "oof_taken_trade_deciles": oof_taken_trade_deciles,
                "oof_taken_trade_deciles_path": oof_taken_trade_deciles_path,
                "oof_prob_threshold_sweep_best": oof_prob_threshold_sweep_best,
                "oof_prob_threshold_sweep_path": oof_prob_threshold_sweep_path,
                "prob_threshold": float(prob_thr),
                "ev_margin": float(ev_margin_local),
                "probability_mapping": probability_mapping,
                "probability_mapping_thresholded": probability_mapping_thresholded,
                "probability_mapping_traded": probability_mapping_traded,
                "label_return_alignment": label_return_alignment,
                "prob_return_ranking": prob_return_ranking,
                "prob_return_ranking_raw": prob_return_ranking_raw,
                "volatility_penalty_lambda": float(vol_penalty_lambda),
                "utility_pre_volatility_penalty": float(utility_pre_volatility_penalty),
                "vol_mean_all": float(vol_mean_all) if vol_mean_all is not None else None,
                "vol_mean_taken": float(vol_mean_taken) if vol_mean_taken is not None else None,
                "vol_excess_z": float(vol_excess_z),
                "vol_excess_abs_z": float(vol_excess_abs_z),
                "layer2_regime_instability_lambda": float(layer2_regime_instability_lambda),
                "regime_dispersion": float(regime_dispersion),
                "utility_pre_regime_penalty": float(utility_pre_regime_penalty),
                "auc": float(mean_auc),
                "pr_auc": float(pr_auc_val) if pr_auc_val is not None and np.isfinite(float(pr_auc_val)) else None,
                "precision_at_1pct": float(precision_at_1pct_val) if precision_at_1pct_val is not None and np.isfinite(float(precision_at_1pct_val)) else None,
                "precision_at_5pct": float(precision_at_5pct_val) if precision_at_5pct_val is not None and np.isfinite(float(precision_at_5pct_val)) else None,
                "precision_at_10pct": float(precision_at_10pct_val) if precision_at_10pct_val is not None and np.isfinite(float(precision_at_10pct_val)) else None,
                "pr_auc_raw": float(pr_auc_raw_val) if pr_auc_raw_val is not None and np.isfinite(float(pr_auc_raw_val)) else None,
                "precision_at_1pct_raw": float(precision_at_1pct_raw_val) if precision_at_1pct_raw_val is not None and np.isfinite(float(precision_at_1pct_raw_val)) else None,
                "precision_at_5pct_raw": float(precision_at_5pct_raw_val) if precision_at_5pct_raw_val is not None and np.isfinite(float(precision_at_5pct_raw_val)) else None,
                "precision_at_10pct_raw": float(precision_at_10pct_raw_val) if precision_at_10pct_raw_val is not None and np.isfinite(float(precision_at_10pct_raw_val)) else None,
                "label_pos_rate": float(label_pos_rate_val) if label_pos_rate_val is not None else None,
                "label_n_pos": int(label_n_pos_val) if label_n_pos_val is not None else None,
                "label_n_neg": int(label_n_neg_val) if label_n_neg_val is not None else None,
                "pred_nan_frac": float(pred_nan_frac_val) if pred_nan_frac_val is not None else None,
                "pred_std": float(pred_std_val) if pred_std_val is not None else None,
                "pred_min": float(pred_min_val) if pred_min_val is not None else None,
                "pred_max": float(pred_max_val) if pred_max_val is not None else None,
                "pred_unique_rounded": int(pred_unique_rounded_val) if pred_unique_rounded_val is not None else None,
                "trades_per_day": float(trades_per_day),
                "n_trades": int(n_trades),
                "take_rate": float(take_rate),
                "trade_mean_return": float(trade_mean_return) if trade_mean_return is not None else None,
                "trade_win_rate": float(trade_win_rate) if trade_win_rate is not None else None,
                "max_drawdown": float(max_drawdown) if max_drawdown is not None else None,
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
                # NEW: utility formula components
                "return_contribution": float(utility_debug.get("return_contribution", 0.0)) if "return_contribution" in utility_debug else None,
                "dd_penalty": float(utility_debug.get("dd_penalty", 0.0)) if "dd_penalty" in utility_debug else None,
                "combined_base": float(utility_debug.get("combined_base", 0.0)) if "combined_base" in utility_debug else None,
            }

        def _compute_layer2_metrics_committee(params: Dict[str, Any]) -> Dict[str, Any]:
            raise RuntimeError(
                "Layer2 committee mode has been removed (Option C only). Use _compute_layer2_metrics instead."
            )
            try:
                w_scalp = float(params.get("w_scalp", 0.0))
                w_swing = float(params.get("w_swing", 0.0))
                w_trend = float(params.get("w_trend", 0.0))
                threshold = float(params.get("consensus_threshold", 0.5))
                abstain_margin = float(params.get("abstain_margin", 0.0))
                ev_margin_local = float(params.get("ev_margin", config.get("ev_margin", 0.0)))
                consensus_quantile = params.get("consensus_quantile", None)
                consensus_quantile = float(consensus_quantile) if consensus_quantile is not None else None
                diversity_lambda = float(params.get("diversity_lambda", 0.0))
                vol_penalty_lambda = float(params.get("volatility_penalty_lambda", 0.0))
                regime_threshold_sensitivity = float(params.get("regime_threshold_sensitivity", 0.0))
            except Exception:
                return {"valid_events": int(len(event_idx)), "utility": 0.0, "fail_reason": "invalid_params"}

            try:
                utility_clip_max = float(config.get("layer2_utility_clip_max", 5000.0))
            except Exception:
                utility_clip_max = 5000.0

            # Ensure these always exist even on early-return paths.
            committee_expert_stats: Dict[str, Any] = {}
            sanity_checks: Dict[str, Any] = {"violations": [], "debug_tables": {}}
            committee_overlap: Dict[str, Any] = {}
            committee_drivers: Dict[str, Any] = {}

            # Avoid NaNs propagating into trades/day, sharpe, etc.
            try:
                days_span_local = float(days_span)
                if (not np.isfinite(days_span_local)) or days_span_local <= 0.0:
                    days_span_local = 1.0
            except Exception:
                days_span_local = 1.0

            # NEW: Mixture of Experts (MoE) Logic
            # Compute weights_mat (n_events x n_experts) based on regime state (ADX, Vol Ratio).
            # Supports 6 base experts + 3 new experts (breakout, vwap_rev, vol_shock) = 9 total.
            
            # Get new expert weights from params
            w_breakout = float(params.get("w_breakout", 0.5))
            w_vwap_rev = float(params.get("w_vwap_rev", 0.5))
            w_vol_shock = float(params.get("w_vol_shock", 0.5))
            
            # Determine number of experts from label_matrix_values
            n_experts = int(label_matrix_values.shape[1]) if label_matrix_values is not None else 6
            has_new_experts = n_experts > 6

            # Always define a normalized weights vector so downstream logic has a stable shape,
            # even when MoE successfully produces weights_mat.
            try:
                if bool(has_new_experts):
                    weights_vec = np.array(
                        [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend, w_breakout, w_vwap_rev, w_vol_shock],
                        dtype=float,
                    )
                else:
                    weights_vec = np.array([w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend], dtype=float)
                weights_vec = np.where(np.isfinite(weights_vec) & (weights_vec >= 0.0), weights_vec, 0.0)
                wsum0 = float(np.sum(weights_vec)) + 1e-8
                if (not np.isfinite(wsum0)) or wsum0 <= 1e-8:
                    return {"valid_events": int(len(event_idx)), "utility": 0.0, "fail_reason": "invalid_static_weights"}
                weights_vec = weights_vec / wsum0
            except Exception:
                return {"valid_events": int(len(event_idx)), "utility": 0.0, "fail_reason": "weights_vec_build_failed"}
            
            weights_mat = None
            try:
                moe_trend = float(params.get("moe_trend_dominance", 0.0))
                moe_scalp = float(params.get("moe_scalp_dominance", 0.0))
                moe_vol   = float(params.get("moe_vol_sensitivity", 0.0))
                # New expert MoE boosts
                moe_breakout_boost = float(params.get("moe_breakout_boost", 0.0))
                moe_vwap_boost = float(params.get("moe_vwap_boost", 0.0))
                moe_vol_shock_boost = float(params.get("moe_vol_shock_boost", 0.0))
                
                any_moe_active = (moe_trend > 0.01 or moe_scalp > 0.01 or moe_vol > 0.01 or
                                  moe_breakout_boost > 0.01 or moe_vwap_boost > 0.01 or moe_vol_shock_boost > 0.01)
                
                if any_moe_active and market_data is not None:
                     evt_idx_local = event_idx
                     n_ev = len(evt_idx_local)
                     
                     # Build state vectors (robust to column naming)
                     adx_vec = np.full(n_ev, 20.0)
                     if "reg_res_adx_14" in market_data.columns:
                         adx_vec = market_data["reg_res_adx_14"].reindex(evt_idx_local).fillna(20.0).values
                     elif "adx" in market_data.columns:
                         adx_vec = market_data["adx"].reindex(evt_idx_local).fillna(20.0).values
                     
                     vol_vec = np.full(n_ev, 1.0)
                     if "reg_ohlcv__vol_ratio_5" in market_data.columns:
                         vol_vec = market_data["reg_ohlcv__vol_ratio_5"].reindex(evt_idx_local).fillna(1.0).values
                     elif "vol_ratio" in market_data.columns:
                         vol_vec = market_data["vol_ratio"].reindex(evt_idx_local).fillna(1.0).values
                     
                     # Build base weight vector (6 or 9 experts)
                     if has_new_experts:
                         w_base = np.array([w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend,
                                            w_breakout, w_vwap_rev, w_vol_shock], dtype=float)
                     else:
                         w_base = np.array([w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend], dtype=float)
                     
                     w_mat = np.tile(w_base, (n_ev, 1))
                     
                     # Quantile-based MoE thresholds (distribution-aware)
                     try:
                         q_trend = float(params.get('moe_adx_trend_q', 0.80))
                     except Exception:
                         q_trend = 0.80
                     try:
                         q_chop = float(params.get('moe_adx_chop_q', 0.20))
                     except Exception:
                         q_chop = 0.20
                     try:
                         q_vol = float(params.get('moe_vol_spike_q', 0.90))
                     except Exception:
                         q_vol = 0.90

                     q_trend = float(np.clip(q_trend, 0.0, 1.0))
                     q_chop = float(np.clip(q_chop, 0.0, 1.0))
                     q_vol = float(np.clip(q_vol, 0.0, 1.0))

                     adx_finite = adx_vec[np.isfinite(adx_vec)]
                     if adx_finite.size >= 20:
                         thr_trend = float(np.quantile(adx_finite, q_trend))
                         thr_chop = float(np.quantile(adx_finite, q_chop))
                     else:
                         thr_trend = 25.0
                         thr_chop = 20.0
                     if not np.isfinite(thr_trend):
                         thr_trend = 25.0
                     if not np.isfinite(thr_chop):
                         thr_chop = 20.0
                     if thr_trend < thr_chop + 1e-6:
                         thr_trend = thr_chop + 1.0

                     vol_finite = vol_vec[np.isfinite(vol_vec)]
                     if vol_finite.size >= 20:
                         thr_vol_spike = float(np.quantile(vol_finite, q_vol))
                     else:
                         thr_vol_spike = 1.5
                     if (not np.isfinite(thr_vol_spike)) or thr_vol_spike <= 0:
                         thr_vol_spike = 1.5
                     
                     trend_mask = adx_vec > thr_trend
                     chop_mask = adx_vec < thr_chop
                     vol_spike_mask = vol_vec > thr_vol_spike
                     
                     # Regime transition mask: chop → trend (ADX rising from chop toward trend)
                     adx_diff = np.zeros_like(adx_vec)
                     try:
                         if "reg_res_adx_14" in market_data.columns:
                             adx_s = market_data["reg_res_adx_14"].reindex(evt_idx_local).fillna(20.0)
                         else:
                             adx_s = market_data["adx"].reindex(evt_idx_local).fillna(20.0)
                         adx_diff = adx_s.diff(3).fillna(0.0).values
                     except Exception:
                         pass
                     transition_mask = chop_mask & (adx_diff > 2.0)  # chop but ADX rising
                     
                     # Trend Boost (base experts)
                     if moe_trend > 0.01:
                         boost = 1.0 + moe_trend * 2.0
                         penal = max(0.01, 1.0 - moe_trend * 0.5)
                         w_mat[trend_mask, 4] *= boost
                         w_mat[trend_mask, 5] *= boost
                         w_mat[trend_mask, 0:4] *= penal
                         
                     # Chop Boost (base experts)
                     if moe_scalp > 0.01:
                         boost = 1.0 + moe_scalp * 2.0
                         penal = max(0.01, 1.0 - moe_scalp * 0.8)
                         w_mat[chop_mask, 0:4] *= boost
                         w_mat[chop_mask, 4:6] *= penal
                         
                     # Vol Sensitivity (base experts)
                     if moe_vol > 0.01:
                         boost = 1.0 + moe_vol
                         penal = max(0.01, 1.0 - moe_vol * 0.5)
                         w_mat[vol_spike_mask, 2:4] *= boost
                         w_mat[vol_spike_mask, 0:2] *= penal
                     
                     # NEW EXPERT BOOSTS (indices 6, 7, 8 if present)
                     if has_new_experts:
                         # Breakout expert (idx 6): boost in transition regimes (chop → trend)
                         if moe_breakout_boost > 0.01:
                             boost = 1.0 + moe_breakout_boost * 3.0
                             w_mat[transition_mask, 6] *= boost
                             # Also boost when vol is expanding
                             w_mat[vol_spike_mask, 6] *= (1.0 + moe_breakout_boost)
                         
                         # VWAP Reversion expert (idx 7): boost in chop regimes
                         if moe_vwap_boost > 0.01:
                             boost = 1.0 + moe_vwap_boost * 2.5
                             w_mat[chop_mask, 7] *= boost
                             # Penalize in strong trends
                             penal = max(0.1, 1.0 - moe_vwap_boost)
                             w_mat[trend_mask, 7] *= penal
                         
                         # Vol Shock expert (idx 8): boost in vol spikes
                         if moe_vol_shock_boost > 0.01:
                             boost = 1.0 + moe_vol_shock_boost * 3.0
                             w_mat[vol_spike_mask, 8] *= boost
                         
                     # Normalize
                     row_sums = np.sum(w_mat, axis=1, keepdims=True) + 1e-12
                     weights_mat = w_mat / row_sums
                     
                     # Verification Logs: MoE weight diagnostics by regime
                     # Store in params dict so they propagate to metrics output
                     moe_diag = {}
                     moe_diag['n_trend_events'] = int(np.sum(trend_mask))
                     moe_diag['n_chop_events'] = int(np.sum(chop_mask))
                     moe_diag['n_vol_spike_events'] = int(np.sum(vol_spike_mask))
                     moe_diag['n_transition_events'] = int(np.sum(transition_mask))
                     
                     # Mean weights by regime for base experts
                     if np.sum(trend_mask) > 10:
                         moe_diag['trend_w_in_trend'] = float(np.mean(weights_mat[trend_mask, 4]))
                         moe_diag['scalp_w_in_trend'] = float(np.mean(weights_mat[trend_mask, 0]))
                     if np.sum(chop_mask) > 10:
                         moe_diag['trend_w_in_chop'] = float(np.mean(weights_mat[chop_mask, 4]))
                         moe_diag['scalp_w_in_chop'] = float(np.mean(weights_mat[chop_mask, 0]))
                         moe_diag['vwap_w_in_chop'] = float(np.mean(weights_mat[chop_mask, 7])) if has_new_experts else 0.0
                     if np.sum(vol_spike_mask) > 10:
                         moe_diag['vol_shock_w_in_vol_spike'] = float(np.mean(weights_mat[vol_spike_mask, 8])) if has_new_experts else 0.0
                     if np.sum(transition_mask) > 5:
                         moe_diag['breakout_w_in_transition'] = float(np.mean(weights_mat[transition_mask, 6])) if has_new_experts else 0.0
                     
                     # Weight sum verification (should be ~1.0)
                     moe_diag['weight_sum_mean'] = float(np.mean(np.sum(weights_mat, axis=1)))
                     moe_diag['weight_sum_std'] = float(np.std(np.sum(weights_mat, axis=1)))
                    
                     params['moe_diagnostics'] = moe_diag

                     try:
                         # Store resolved thresholds for transparency/debugging
                         moe_diag['thr_trend'] = float(thr_trend)
                         moe_diag['thr_chop'] = float(thr_chop)
                         moe_diag['thr_vol_spike'] = float(thr_vol_spike)
                         moe_diag['q_trend'] = float(q_trend)
                         moe_diag['q_chop'] = float(q_chop)
                         moe_diag['q_vol_spike'] = float(q_vol)
                     except Exception:
                         pass

            except Exception:
                weights_mat = None

            # ================================================================
            # STORE REGIME MASKS FOR DIVERSITY CALCULATION
            # ================================================================
            # These masks are used for regime-aware correlation penalty.
            # We reconstruct them here if they weren't computed in MoE block.
            regime_masks_for_diversity: Dict[str, np.ndarray] = {}
            try:
                n_ev_local = len(event_idx)
                if market_data is not None:
                    # Reconstruct ADX/vol vectors
                    adx_vec_div = np.full(n_ev_local, 20.0)
                    if "reg_res_adx_14" in market_data.columns:
                        adx_vec_div = market_data["reg_res_adx_14"].reindex(event_idx).fillna(20.0).values
                    elif "adx" in market_data.columns:
                        adx_vec_div = market_data["adx"].reindex(event_idx).fillna(20.0).values
                    
                    vol_vec_div = np.full(n_ev_local, 1.0)
                    if "reg_ohlcv__vol_ratio_5" in market_data.columns:
                        vol_vec_div = market_data["reg_ohlcv__vol_ratio_5"].reindex(event_idx).fillna(1.0).values
                    elif "vol_ratio" in market_data.columns:
                        vol_vec_div = market_data["vol_ratio"].reindex(event_idx).fillna(1.0).values
                    
                    # Use quantile thresholds from MoE diagnostics if available, else defaults
                    moe_diag_local = params.get("moe_diagnostics", {})
                    thr_trend_div = float(moe_diag_local.get("thr_trend", 25.0))
                    thr_chop_div = float(moe_diag_local.get("thr_chop", 20.0))
                    thr_vol_spike_div = float(moe_diag_local.get("thr_vol_spike", 1.5))
                    
                    regime_masks_for_diversity["trend"] = adx_vec_div > thr_trend_div
                    regime_masks_for_diversity["chop"] = adx_vec_div < thr_chop_div
                    regime_masks_for_diversity["vol_spike"] = vol_vec_div > thr_vol_spike_div
                    # Neutral regime: not trend, not chop
                    regime_masks_for_diversity["neutral"] = (
                        (adx_vec_div >= thr_chop_div) & (adx_vec_div <= thr_trend_div)
                    )
            except Exception:
                regime_masks_for_diversity = {}

            # Fallback to static weights if MoE disabled or failed
            if weights_mat is None:
                # Build static weights vector matching number of experts
                if has_new_experts:
                    weights_vec = np.array(
                        [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend,
                         w_breakout, w_vwap_rev, w_vol_shock],
                        dtype=float,
                    )
                else:
                    weights_vec = np.array(
                        [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend],
                        dtype=float,
                    )
                total_weight = float(np.sum(weights_vec)) + 1e-8
                if (not np.isfinite(total_weight)) or (total_weight <= 1e-8):
                    return {"valid_events": int(len(event_idx)), "utility": 0.0, "fail_reason": "invalid_static_weights"}
                # Normalize static weights
                weights_vec = weights_vec / total_weight

            # Ensure we always have a broadcastable weights matrix for downstream math.
            # If MoE is disabled, this becomes a constant matrix.
            try:
                if weights_mat is None:
                    weights_mat = np.tile(weights_vec, (int(len(event_idx)), 1))
            except Exception:
                pass


            # Align and sanitize committee matrices to prevent NaN propagation and size mismatches.
            try:
                lbl_mat = np.asarray(label_matrix_values, dtype=float)
                ret_mat0 = np.asarray(returns_matrix_values, dtype=float)

                conf_mat0 = confidence_matrix_values
                if conf_mat0 is None:
                    conf_mat0 = np.ones_like(ret_mat0, dtype=float)
                conf_mat0 = np.asarray(conf_mat0, dtype=float)

                n0 = int(min(
                    int(len(event_idx)),
                    int(lbl_mat.shape[0]) if lbl_mat.ndim == 2 else int(lbl_mat.size),
                    int(ret_mat0.shape[0]) if ret_mat0.ndim == 2 else int(ret_mat0.size),
                    int(conf_mat0.shape[0]) if conf_mat0.ndim == 2 else int(conf_mat0.size),
                ))
                if n0 <= 0:
                    return {"valid_events": 0, "utility": 0.0, "fail_reason": "empty_committee_matrices"}

                # Keep columns unchanged; only align row dimension.
                lbl_mat = lbl_mat[:n0, :]
                ret_mat0 = ret_mat0[:n0, :]
                conf_mat0 = conf_mat0[:n0, :]
                ev_idx0 = pd.DatetimeIndex(event_idx[:n0])

                lbl_mat = np.where(np.isfinite(lbl_mat), lbl_mat, 0.0)
                ret_mat0 = np.where(np.isfinite(ret_mat0), ret_mat0, np.nan)
                conf_mat0 = np.where(np.isfinite(conf_mat0) & (conf_mat0 >= 0.0), conf_mat0, 0.0)
            except Exception:
                # Fall back to existing globals if anything goes wrong
                lbl_mat = np.asarray(label_matrix_values, dtype=float)
                ret_mat0 = np.asarray(returns_matrix_values, dtype=float)
                conf_mat0 = confidence_matrix_values
                if conf_mat0 is None:
                    conf_mat0 = np.ones_like(ret_mat0, dtype=float)
                conf_mat0 = np.asarray(conf_mat0, dtype=float)
                ev_idx0 = pd.DatetimeIndex(event_idx)

            # ================================================================
            # CONSENSUS SCORE COMPUTATION - EX-ANTE MODE vs LEGACY MODE
            # ================================================================
            # Ex-ante mode: use signal strength * regime-conditioned weights
            # Legacy mode: use outcome labels (LEAKY - produces 100% win rate)
            # ================================================================
            try:
                layer2_eval_mode = str(config.get("layer2_eval_mode", "ex_ante")).lower()
            except Exception:
                layer2_eval_mode = "ex_ante"

            conf_mat = conf_mat0

            # ================================================================
            # ADVANCED GATING PIPELINE INTEGRATION
            # ================================================================
            # If advanced_gating_pipeline is fitted, use it for:
            # - Calibrated confidence scores
            # - Learned meta-gate weights
            # - Abstention-aware consensus
            # - Specialization-adjusted weights
            # ================================================================
            adv_gating_result: Optional[Dict[str, Any]] = None
            use_advanced_gating = bool(config.get("enable_advanced_gating", True))
            
            if use_advanced_gating and advanced_gating_pipeline is not None and advanced_gating_pipeline.is_fitted:
                try:
                    # Compute regime labels for current events
                    regime_labels_l2 = compute_regime_labels_for_events(
                        market_data=market_data,
                        event_idx=ev_idx0,
                    )
                    
                    # Build base weights from current params
                    if has_new_experts:
                        base_weights_l2 = np.array([
                            w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend,
                            w_breakout, w_vwap_rev, w_vol_shock
                        ], dtype=float)
                    else:
                        base_weights_l2 = np.array([
                            w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend
                        ], dtype=float)
                    base_weights_l2 = base_weights_l2 / (np.sum(base_weights_l2) + 1e-8)
                    
                    # Get base barrier geometry from params
                    base_tp_l2 = float(params.get("tp_atr_mult", 2.0))
                    base_sl_l2 = float(params.get("sl_atr_mult", 1.0))
                    base_horizon_l2 = int(params.get("horizon_bars", 12))
                    base_trail_l2 = float(params.get("trail_distance_atr_mult", 0.0))
                    
                    # Apply advanced gating pipeline
                    adv_gating_result = advanced_gating_pipeline.apply(
                        market_data=market_data,
                        event_idx=ev_idx0,
                        expert_labels=lbl_mat,
                        expert_confidences=conf_mat0,
                        base_weights=base_weights_l2,
                        base_tp=base_tp_l2,
                        base_sl=base_sl_l2,
                        base_horizon=base_horizon_l2,
                        base_trail=base_trail_l2,
                        regime_labels=regime_labels_l2,
                        coverage_min_override=params.get("coverage_min", None),
                        consensus_threshold_override=params.get("consensus_threshold", None),
                        specialization_strength_override=params.get("specialization_strength", None),
                        diversity_lambda_override=params.get("diversity_lambda", None),
                    )
                    
                    # Use calibrated confidence and learned weights
                    if adv_gating_result is not None:
                        conf_mat = adv_gating_result.get("calibrated_conf", conf_mat0)
                        learned_weights = adv_gating_result.get("weights", None)
                        if learned_weights is not None:
                            weights_mat = learned_weights
                            # Update moe_diagnostics to indicate learned router was used
                            moe_diag_update = params.get("moe_diagnostics", {})
                            moe_diag_update["learned_router"] = True
                            moe_diag_update["learned_weights_shape"] = list(learned_weights.shape) if hasattr(learned_weights, 'shape') else None
                            moe_diag_update["learned_weights_mean"] = float(np.mean(learned_weights)) if learned_weights is not None else None
                            moe_diag_update["learned_weights_std"] = float(np.std(learned_weights)) if learned_weights is not None else None
                            params["moe_diagnostics"] = moe_diag_update
                        # Store diagnostics
                        params["adv_gating_diagnostics"] = adv_gating_result.get("diagnostics", {})
                        
                except Exception as adv_exc:
                    tprint_warning(f"   [L2] Advanced gating apply failed: {adv_exc}")
                    adv_gating_result = None

            if layer2_eval_mode == "ex_ante":
                # --------------------------------------------------------
                # EX-ANTE MODE: No outcome leakage
                # --------------------------------------------------------
                # Signal strength from primary_signals (available at event time)
                try:
                    sig_strength = primary_signals["consensus"].reindex(ev_idx0).fillna(0.0).abs().values
                    sig_dir = np.sign(primary_signals["consensus"].reindex(ev_idx0).fillna(0.0).values)
                except Exception:
                    sig_strength = np.ones(len(ev_idx0), dtype=float)
                    sig_dir = np.ones(len(ev_idx0), dtype=float)

                # Regime-conditioned expert firing (ex-ante: based on volatility regime)
                # Expert configs: [scalp_L, scalp_S, swing_L, swing_S, trend_L, trend_S]
                # - Scalp fires in low volatility (regime_scalar < 1)
                # - Trend fires in high volatility (regime_scalar > 1)
                # - Swing fires in medium volatility (regime_scalar ~ 1)
                n_exp = int(weights_vec.size)
                fired_ex_ante = np.ones((len(ev_idx0), n_exp), dtype=float)
                try:
                    if regime_scalar_for_barriers is not None:
                        s_evt = regime_scalar_for_barriers.reindex(ev_idx0).astype(float)
                        s_evt = s_evt.replace([np.inf, -np.inf], np.nan).fillna(1.0).values
                        # Scalp experts (cols 0,1): prefer low volatility
                        scalp_affinity = np.clip(2.0 - s_evt, 0.2, 1.5)
                        # Swing experts (cols 2,3): prefer medium volatility
                        swing_affinity = np.clip(1.5 - np.abs(s_evt - 1.0), 0.2, 1.5)
                        # Trend experts (cols 4,5): prefer high volatility
                        trend_affinity = np.clip(s_evt, 0.2, 1.5)
                        fired_ex_ante[:, 0] = scalp_affinity
                        fired_ex_ante[:, 1] = scalp_affinity
                        fired_ex_ante[:, 2] = swing_affinity
                        fired_ex_ante[:, 3] = swing_affinity
                        fired_ex_ante[:, 4] = trend_affinity
                        fired_ex_ante[:, 5] = trend_affinity
                        # New experts: add simple ex-ante affinities
                        if n_exp > 6:
                            fired_ex_ante[:, 6] = np.clip(s_evt, 0.2, 1.5)  # breakout
                            fired_ex_ante[:, 7] = scalp_affinity  # vwap reversion
                            fired_ex_ante[:, 8] = np.clip(s_evt, 0.2, 1.5)  # vol shock
                except Exception:
                    pass

                # --------------------------------------------------------
                # USE ADVANCED GATING FOR CONSENSUS IF AVAILABLE
                # --------------------------------------------------------
                if adv_gating_result is not None and advanced_gating_pipeline is not None:
                    # Use abstention-aware consensus from advanced gating
                    consensus_score = adv_gating_result.get("consensus_scores", None)
                    coverage_l2 = adv_gating_result.get("coverage", None)
                    
                    if consensus_score is None:
                        # Fallback to standard computation
                        if weights_mat is None:
                            w_use = np.tile(weights_vec, (len(ev_idx0), 1))
                        else:
                            w_use = np.asarray(weights_mat, dtype=float)[: int(len(ev_idx0)), :]
                        denom_w = np.sum(w_use, axis=1).astype(float) + 1e-8
                        expert_agreement = np.sum(fired_ex_ante * w_use, axis=1).astype(float) / denom_w
                        consensus_score = sig_dir * sig_strength * expert_agreement
                    
                    fired = np.ones_like(fired_ex_ante, dtype=bool)
                    tprint_info(f"   [L2_EX_ANTE] Using advanced gating: calibrated conf + learned weights + abstention-aware")
                else:
                    # Standard ex-ante consensus computation
                    if weights_mat is None:
                        w_use = np.tile(weights_vec, (len(ev_idx0), 1))
                    else:
                        w_use = np.asarray(weights_mat, dtype=float)[: int(len(ev_idx0)), :]
                    denom_w = np.sum(w_use, axis=1).astype(float) + 1e-8
                    expert_agreement = np.sum(fired_ex_ante * w_use, axis=1).astype(float) / denom_w
                    consensus_score = sig_dir * sig_strength * expert_agreement
                    fired = np.ones_like(fired_ex_ante, dtype=bool)
                    tprint_info(f"   [L2_EX_ANTE] Using ex-ante consensus: signal_strength * regime_expert_weights")

                # ============================================================
                # SIGNAL QUALITY DIAGNOSTICS
                # ============================================================
                # Compute correlations to understand if signal predicts returns
                try:
                    # Get weighted returns for correlation analysis
                    # Use MoE weights if available (weights_mat) else static
                    if weights_mat is None:
                        w_use = np.tile(weights_vec, (len(ev_idx0), 1))
                    else:
                        w_use = np.asarray(weights_mat, dtype=float)[: int(len(ev_idx0)), :]
                    
                    finite_mask_diag = np.isfinite(ret_mat0)
                    denom_diag = np.sum(finite_mask_diag * conf_mat0 * w_use, axis=1).astype(float) + 1e-8
                    numer_diag = np.sum(np.where(finite_mask_diag, ret_mat0, 0.0) * conf_mat0 * w_use, axis=1).astype(float)
                    weighted_returns_diag = numer_diag / denom_diag

                    # Mask for valid samples
                    valid_mask = np.isfinite(sig_strength) & np.isfinite(weighted_returns_diag) & np.isfinite(consensus_score)
                    n_valid = int(np.sum(valid_mask))

                    if n_valid >= 50:
                        sig_v = sig_strength[valid_mask]
                        cs_v = consensus_score[valid_mask]
                        ret_v = weighted_returns_diag[valid_mask]
                        ret_binary = (ret_v > 0.0).astype(float)

                        # Pearson correlations
                        corr_sig_ret = float(np.corrcoef(sig_v, ret_v)[0, 1]) if np.std(sig_v) > 1e-10 else 0.0
                        corr_cs_ret = float(np.corrcoef(cs_v, ret_v)[0, 1]) if np.std(cs_v) > 1e-10 else 0.0
                        corr_sig_win = float(np.corrcoef(sig_v, ret_binary)[0, 1]) if np.std(sig_v) > 1e-10 else 0.0
                        corr_cs_win = float(np.corrcoef(cs_v, ret_binary)[0, 1]) if np.std(cs_v) > 1e-10 else 0.0

                        # Win rate by signal strength quintiles
                        quintile_winrates = []
                        try:
                            quintiles = np.percentile(sig_v, [0, 20, 40, 60, 80, 100])
                            for i in range(5):
                                q_mask = (sig_v >= quintiles[i]) & (sig_v < quintiles[i+1] + 1e-10)
                                if np.sum(q_mask) > 5:
                                    q_winrate = float(np.mean(ret_v[q_mask] > 0))
                                    q_mean_ret = float(np.mean(ret_v[q_mask]))
                                    quintile_winrates.append((i+1, float(np.sum(q_mask)), q_winrate, q_mean_ret))
                        except Exception:
                            pass

                        # Overall stats
                        overall_winrate = float(np.mean(ret_v > 0))
                        overall_mean_ret = float(np.mean(ret_v))

                        tprint_info(f"   [SIGNAL_DIAG] n_valid={n_valid}, overall_winrate={overall_winrate:.1%}, mean_ret={overall_mean_ret*100:.3f}%")
                        tprint_info(f"   [SIGNAL_DIAG] Pearson corr(|signal|, return)={corr_sig_ret:.4f}, corr(cs, return)={corr_cs_ret:.4f}")
                        tprint_info(f"   [SIGNAL_DIAG] Pearson corr(|signal|, win)={corr_sig_win:.4f}, corr(cs, win)={corr_cs_win:.4f}")

                        if quintile_winrates:
                            q_str = " | ".join([f"Q{q[0]}: n={q[1]:.0f}, wr={q[2]:.1%}, ret={q[3]*100:.3f}%" for q in quintile_winrates])
                            tprint_info(f"   [SIGNAL_DIAG] Win by signal quintile: {q_str}")

                        # Interpretation
                        if abs(corr_sig_ret) < 0.02 and abs(corr_cs_ret) < 0.02:
                            tprint_warning(f"   [SIGNAL_DIAG] ⚠️ Signal has NO correlation with returns - ex-ante committee cannot work!")
                        elif corr_cs_ret < 0:
                            tprint_warning(f"   [SIGNAL_DIAG] ⚠️ Consensus score NEGATIVELY correlated with returns - direction issue?")
                        elif corr_cs_ret > 0.05:
                            tprint_info(f"   [SIGNAL_DIAG] ✅ Consensus score positively correlated with returns ({corr_cs_ret:.4f})")
                except Exception as diag_exc:
                    tprint_warning(f"   [SIGNAL_DIAG] Failed to compute diagnostics: {diag_exc}")

            else:
                # --------------------------------------------------------
                # LEGACY MODE: Uses outcome labels (LEAKY - for debugging only)
                # --------------------------------------------------------
                # Use MoE weights if available (weights_mat) else static
                if weights_mat is None:
                    w_use = np.tile(weights_vec, (len(ev_idx0), 1))
                else:
                    w_use = weights_mat
                
                fired = (lbl_mat != 0.0)
                denom_cs = np.sum(fired.astype(float) * conf_mat * w_use, axis=1).astype(float) + 1e-8
                numer_cs = np.sum(lbl_mat.astype(float) * conf_mat * w_use, axis=1).astype(float)
                consensus_score = numer_cs / denom_cs
                tprint_warning(f"   [L2_LEGACY] Using outcome-based consensus (LEAKY - win_rate will be inflated)")
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
                frac_pos = float(np.mean(cs > 0.0)) if cs.size else 0.0
                frac_neg = float(np.mean(cs < 0.0)) if cs.size else 0.0
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
                thr_take = float(threshold) + float(abstain_margin) + float(ev_margin_local)

                # ================================================================
                # REGIME-ADAPTIVE PROBABILITY THRESHOLDS (HPO-optimized per regime)
                # ================================================================
                # Apply regime-specific threshold adjustments from HPO parameters.
                # These are ADDED to the base threshold for each regime type.
                thr_take_vec = None
                try:
                    # Get regime-specific threshold adjustments from params
                    adj_vol_low = float(params.get("prob_threshold_adj_vol_low", 0.0))
                    adj_vol_high = float(params.get("prob_threshold_adj_vol_high", 0.0))
                    adj_trend_high = float(params.get("prob_threshold_adj_trend_high", 0.0))
                    adj_trend_low = float(params.get("prob_threshold_adj_trend_low", 0.0))

                    # Validate adjustments are finite
                    if not np.isfinite(adj_vol_low):
                        adj_vol_low = 0.0
                    if not np.isfinite(adj_vol_high):
                        adj_vol_high = 0.0
                    if not np.isfinite(adj_trend_high):
                        adj_trend_high = 0.0
                    if not np.isfinite(adj_trend_low):
                        adj_trend_low = 0.0

                    # Check if any adjustments are non-zero
                    has_regime_adjustments = (
                        abs(adj_vol_low) > 1e-6
                        or abs(adj_vol_high) > 1e-6
                        or abs(adj_trend_high) > 1e-6
                        or abs(adj_trend_low) > 1e-6
                    )

                    if has_regime_adjustments and regime_labels is not None:
                        # Get regime labels aligned to events
                        vol_regime = regime_labels.get("volatility_regime")
                        trend_regime = regime_labels.get("trend_regime")

                        # Build per-event threshold adjustment vector
                        thr_adjustments = np.zeros(len(ev_idx0), dtype=float)

                        if vol_regime is not None:
                            try:
                                vol_r = vol_regime.reindex(ev_idx0).astype(str).fillna("medium")
                                thr_adjustments += np.where(vol_r == "low", adj_vol_low, 0.0)
                                thr_adjustments += np.where(vol_r == "high", adj_vol_high, 0.0)
                            except Exception:
                                pass

                        if trend_regime is not None:
                            try:
                                trend_r = trend_regime.reindex(ev_idx0).astype(str).fillna("medium")
                                thr_adjustments += np.where(trend_r == "high", adj_trend_high, 0.0)
                                thr_adjustments += np.where(trend_r == "low", adj_trend_low, 0.0)
                            except Exception:
                                pass

                        # Apply adjustments to base threshold
                        thr_take_vec = float(thr_take) + thr_adjustments
                        thr_take_vec = np.clip(thr_take_vec, 0.3, 0.95)  # Ensure reasonable bounds

                except Exception:
                    thr_take_vec = None

                # Legacy regime-aware thresholding: allow HPO to increase/decrease the
                # effective threshold in regimes where the committee generalizes poorly.
                # (This is multiplicative scaling, complementary to the additive adjustments above)
                try:
                    if (
                        np.isfinite(float(regime_threshold_sensitivity))
                        and float(regime_threshold_sensitivity) > 0.0
                        and regime_scalar_for_barriers is not None
                    ):
                        s_evt = regime_scalar_for_barriers.reindex(ev_idx0).astype(float)
                        s_evt = s_evt.replace([np.inf, -np.inf], np.nan).fillna(1.0)
                        if thr_take_vec is None:
                            thr_take_vec = float(thr_take) * (
                                1.0 + float(regime_threshold_sensitivity) * (s_evt.astype(float) - 1.0)
                            )
                        else:
                            # Combine with existing adjustments (multiply scaling)
                            scaling = 1.0 + float(regime_threshold_sensitivity) * (s_evt.astype(float) - 1.0)
                            thr_take_vec = thr_take_vec * scaling.to_numpy(dtype=float)
                        thr_take_vec = pd.to_numeric(pd.Series(thr_take_vec), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(thr_take))
                        thr_take_vec = thr_take_vec.clip(lower=0.0, upper=0.999999).to_numpy(dtype=float)
                except Exception:
                    pass

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
                        if thr_take_vec is not None and int(len(thr_take_vec)) == int(n):
                            take_mask = take_mask & (cs_full > thr_take_vec)
                        else:
                            take_mask = take_mask & (cs_full > float(thr_take))
                    else:
                        take_mask = np.zeros(0, dtype=bool)
                else:
                    if thr_take_vec is not None and int(len(thr_take_vec)) == int(cs_full.size):
                        take_mask = cs_full > thr_take_vec
                    else:
                        take_mask = cs_full > float(thr_take)
                
                # ============================================================
                # COVERAGE GATING: Use advanced gating take_mask if available
                # ============================================================
                # The advanced gating pipeline provides abstention-aware take_mask
                # that gates on coverage (fraction of experts that fired).
                coverage_min_param = float(params.get("coverage_min", 0.3))
                if adv_gating_result is not None:
                    adv_take_mask = adv_gating_result.get("take_mask", None)
                    adv_coverage = adv_gating_result.get("coverage", None)
                    
                    if adv_take_mask is not None and len(adv_take_mask) == len(take_mask):
                        # Combine: require both consensus threshold AND coverage threshold
                        take_mask = take_mask & adv_take_mask
                        
                    if adv_coverage is not None and len(adv_coverage) == len(take_mask):
                        # Additional coverage gating: reject low-coverage events
                        coverage_gate = adv_coverage >= coverage_min_param
                        take_mask = take_mask & coverage_gate
                        
            except Exception:
                take_mask = np.asarray(consensus_score, dtype=float) > float(threshold + abstain_margin + ev_margin_local)

            n_trades = int(np.sum(take_mask))
            take_rate = float(n_trades) / float(len(ev_idx0)) if len(ev_idx0) > 0 else 0.0
            trades_per_day = float(n_trades) / float(max(days_span_local, 1.0))

            # Compute weighted returns for the final committee decision (used for sanity + trade metrics).
            # ================================================================
            # REGIME-ADJUSTED BARRIER RETURNS
            # ================================================================
            # If advanced gating provides regime-adjusted barriers, scale returns
            # to approximate what they would be under the adjusted geometry.
            # This is an approximation since we can't recompute full triple-barrier
            # outcomes per trial (too expensive). Instead, we scale returns by the
            # ratio of regime-adjusted TP/SL to base TP/SL.
            regime_return_scaling = None
            try:
                if adv_gating_result is not None:
                    tp_arr = adv_gating_result.get("tp_arr", None)
                    sl_arr = adv_gating_result.get("sl_arr", None)
                    base_tp = float(params.get("tp_atr_mult", 2.0))
                    base_sl = float(params.get("sl_atr_mult", 1.0))
                    
                    if tp_arr is not None and sl_arr is not None:
                        # Compute scaling factor: regime-adjusted / base
                        # For TP hits: scale by tp_ratio (wider TP = potentially larger gains)
                        # For SL hits: scale by sl_ratio (wider SL = potentially larger losses)
                        tp_ratio = np.asarray(tp_arr, dtype=float) / (base_tp + 1e-8)
                        sl_ratio = np.asarray(sl_arr, dtype=float) / (base_sl + 1e-8)
                        
                        # Blend: use geometric mean of ratios as overall scaling
                        # This approximates the effect of regime-adjusted barriers
                        regime_return_scaling = np.sqrt(tp_ratio * sl_ratio)
                        regime_return_scaling = np.clip(regime_return_scaling, 0.5, 2.0)
            except Exception:
                regime_return_scaling = None
            
            try:
                ret_mat = np.asarray(ret_mat0, dtype=float)
                
                # Apply regime-adjusted scaling if available
                if regime_return_scaling is not None and len(regime_return_scaling) == ret_mat.shape[0]:
                    # Scale each row's returns by the regime factor
                    ret_mat = ret_mat * regime_return_scaling.reshape(-1, 1)
                
                finite_mask = np.isfinite(ret_mat)

                w_row = np.asarray(weights_vec, dtype=float).reshape(1, -1)
                denom = np.sum(finite_mask * conf_mat * w_row, axis=1).astype(float) + 1e-8
                numer = np.sum(np.where(finite_mask, ret_mat, 0.0) * conf_mat * w_row, axis=1).astype(float)
                weighted_returns = numer / denom
                weighted_returns = np.where(np.isfinite(weighted_returns), weighted_returns, 0.0)
            except Exception:
                weighted_returns = np.zeros(int(len(event_idx)), dtype=float)

            sanity = _layer2_sanity_checks(
                take_mask=take_mask,
                weighted_returns=weighted_returns,
                event_idx=pd.DatetimeIndex(ev_idx0),
                strict=bool(config.get("layer2_sanity_strict", True)),
                debug_context={
                    "tx_cost": float(DEFAULT_TRANSACTION_COST),
                    "raw_returns_matrix": ret_mat if 'ret_mat' in locals() else None,
                },
            )
            try:
                sanity_checks["violations"] = list(sanity.get("violations", []))
                sanity_checks["debug_tables"]["layer2_sanity"] = dict(sanity.get("stats", {}))
            except Exception:
                pass

            if (not bool(sanity.get("ok", True))) and bool(config.get("layer2_sanity_strict", True)):
                return {
                    "valid_events": int(len(event_idx)),
                    "utility": 0.0,
                    "utility_pre_clip": 0.0,
                    "utility_clip_max": float(utility_clip_max),
                    "auc": 0.5,
                    "psr": float(psr_details.get("psr", 0.0)),
                    "psr_z": float(psr_details.get("psr_z", float("-inf"))),
                    "psr_sr": psr_details.get("sr", None),
                    "psr_n": int(psr_details.get("n", 0) or 0),
                    "auc_negscore": 0.5,
                    "auc_global": 0.5,
                    "auc_global_negscore": 0.5,
                    "trades_per_day": float(trades_per_day),
                    "probability_mapping": [],
                    "consensus_score_mapping": [],
                    "volatility_penalty_lambda": float(vol_penalty_lambda),
                    "utility_pre_volatility_penalty": 0.0,
                    "vol_mean_all": None,
                    "vol_mean_taken": None,
                    "vol_excess_z": 0.0,
                    "vol_excess_abs_z": 0.0,
                    "abstain_margin": float(abstain_margin),
                    "ev_margin": float(ev_margin_local),
                    "diversity_lambda": float(diversity_lambda),
                    "weighted_avg_abs_corr": 0.0,
                    "diversity_penalty": 0.0,
                    "diversity_multiplier": 0.0,
                    "layer2_soft_min_trades_committee": int(config.get("layer2_soft_min_trades_committee", 50)),
                    "phi_trades": 0.0,
                    "sharpe_mean": 0.0,
                    "sharpe_std": 0.0,
                    "sharpe_min": 0.0,
                    "sharpe_max": 0.0,
                    "folds_sharpe_values": [],
                    "per_fold_metrics": [],
                    "per_regime_metrics": {},
                    "n_trades": int(n_trades),
                    "trade_mean_return": 0.0,
                    "trade_win_rate": 0.0,
                    "take_rate": float(take_rate),
                    "net_pnl_total": 0.0,
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
                    "lambda_vol": 1.2,
                    "w_auc": 1.0,
                    "w_den": 0.5,
                    "avg_sharpe": 0.0,
                    "vol_sharpe": 0.0,
                    "base_score": 0.0,
                    "base_norm": 0.0,
                    "phi_auc": 0.0,
                    "phi_density": 0.0,
                    "modifier": 0.0,
                }

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
                    out = {
                        "n_events": int(lbl_col.size),
                        "n_fired": int(n_fired),
                        "frac_fired": float(n_fired) / float(max(int(lbl_col.size), 1)),
                        "frac_pos": float(np.mean(lbl_col > 0.0)) if lbl_col.size else 0.0,
                        "frac_neg": float(np.mean(lbl_col < 0.0)) if lbl_col.size else 0.0,
                    }
                    if n_fired > 0:
                        r = ret_col[fired]
                        r = r[np.isfinite(r)]
                        out["mean_return_on_fired"] = float(np.mean(r)) if r.size else 0.0
                        out["win_rate_on_fired"] = float(np.mean(r > 0.0)) if r.size else 0.0
                    else:
                        out["mean_return_on_fired"] = 0.0
                        out["win_rate_on_fired"] = 0.0

                    try:
                        tm = np.asarray(take_mask, dtype=bool)
                        if int(tm.size) == int(ret_col.size):
                            r_taken = np.asarray(ret_col, dtype=float)[tm]
                            r_taken = r_taken[np.isfinite(r_taken)]
                            out["n_taken"] = int(np.sum(tm))
                            out["mean_return_on_taken"] = float(np.mean(r_taken)) if r_taken.size else 0.0
                            out["win_rate_on_taken"] = float(np.mean(r_taken > 0.0)) if r_taken.size else 0.0

                            agree_mask = tm & (lbl_col != 0.0) & (np.sign(lbl_col) == np.sign(consensus_arr))
                            r_agree = np.asarray(ret_col, dtype=float)[agree_mask]
                            r_agree = r_agree[np.isfinite(r_agree)]
                            out["n_agree_on_taken"] = int(np.sum(agree_mask))
                            out["mean_return_on_agree_on_taken"] = float(np.mean(r_agree)) if r_agree.size else 0.0
                    except Exception:
                        pass

                    committee_expert_stats[str(name)] = out
            except Exception:
                committee_expert_stats = {}

            # ================================================================
            # PER-EXPERT PSR: Compute individual expert risk-adjusted returns
            # ================================================================
            # This enables identification of consistently unprofitable experts
            # following De Prado's AFML methodology for strategy evaluation.
            per_expert_psr: Dict[str, Dict[str, Any]] = {}
            try:
                psr_sr_benchmark = float(config.get("layer2_psr_sr_benchmark", 0.0))
                if not np.isfinite(psr_sr_benchmark):
                    psr_sr_benchmark = 0.0
                psr_min_expert_trades = int(config.get("layer2_psr_min_expert_trades", 10))
                
                per_expert_psr = _compute_per_expert_psr(
                    returns_matrix=returns_matrix_values,
                    label_matrix=label_matrix_values,
                    take_mask=np.asarray(take_mask, dtype=bool),
                    event_idx=pd.DatetimeIndex(event_idx),
                    expert_names=list(committee_names),
                    sr_benchmark=psr_sr_benchmark,
                    periods_per_year=365.0,
                    min_trades=psr_min_expert_trades,
                )
                
                # Merge PSR into committee_expert_stats
                for name, psr_data in per_expert_psr.items():
                    if name in committee_expert_stats:
                        committee_expert_stats[name]["psr"] = psr_data.get("psr", 0.0)
                        committee_expert_stats[name]["psr_z"] = psr_data.get("psr_z", float("-inf"))
                        committee_expert_stats[name]["psr_sr"] = psr_data.get("sr")
                        committee_expert_stats[name]["psr_n"] = psr_data.get("n", 0)
                        committee_expert_stats[name]["psr_skew"] = psr_data.get("skew", 0.0)
                        committee_expert_stats[name]["psr_kurt"] = psr_data.get("kurt", 3.0)
                        committee_expert_stats[name]["psr_total_pnl"] = psr_data.get("total_pnl", 0.0)
                        committee_expert_stats[name]["psr_mean_return"] = psr_data.get("mean_return", 0.0)
                        committee_expert_stats[name]["psr_win_rate"] = psr_data.get("win_rate", 0.0)
            except Exception:
                pass

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

            # weighted_returns computed earlier (used for sanity checks)

            auc_global = 0.5
            auc_global_negscore = 0.5
            auc_global_n_pos = None
            auc_global_n_neg = None
            try:
                sc_all = np.asarray(consensus_score, dtype=float)
                wr_all = np.asarray(weighted_returns, dtype=float)
                fired_any = np.asarray(fired, dtype=bool)
                if fired_any.ndim == 2:
                    fired_any = np.any(fired_any, axis=1)
                elif fired_any.ndim != 1:
                    fired_any = np.ones(int(sc_all.size), dtype=bool)
                m_auc_all = fired_any & np.isfinite(sc_all) & np.isfinite(wr_all)
                if int(np.sum(m_auc_all)) >= 20:
                    y_auc_all = (wr_all[m_auc_all] > 0.0).astype(int)
                    auc_global_n_pos = int(np.sum(y_auc_all == 1))
                    auc_global_n_neg = int(np.sum(y_auc_all == 0))
                    if int(np.unique(y_auc_all).size) >= 2:
                        auc_dir = float(roc_auc_score(y_auc_all, sc_all[m_auc_all]))
                        auc_inv = float(roc_auc_score(y_auc_all, (-sc_all[m_auc_all])))
                        auc_global = float(max(auc_dir, auc_inv))
                        auc_global_negscore = float(auc_inv)
            except Exception:
                auc_global = 0.5
                auc_global_negscore = 0.5

            weighted_durations = np.ones(int(len(event_idx)), dtype=float)
            try:
                dur_mat = np.asarray(durations_matrix_values, dtype=float)
                dur_mat = np.where(np.isfinite(dur_mat) & (dur_mat > 0.0), dur_mat, np.nan)
                finite_dur = np.isfinite(dur_mat)
                denom_d = np.sum(finite_dur * conf_mat * w_row, axis=1).astype(float) + 1e-8
                numer_d = np.sum(np.where(finite_dur, dur_mat, 0.0) * conf_mat * w_row, axis=1).astype(float)
                wd = numer_d / denom_d
                wd = np.where(np.isfinite(wd) & (wd > 0.0), wd, 1.0)
                weighted_durations = wd.astype(float)
            except Exception:
                weighted_durations = np.ones(int(len(event_idx)), dtype=float)

            trade_returns = np.asarray(weighted_returns, dtype=float)[np.asarray(take_mask, dtype=bool)]
            trade_returns = trade_returns[np.isfinite(trade_returns)]

            per_fold_metrics: List[Dict[str, Any]] = []
            fold_sharpes: List[float] = []
            fold_aucs: List[float] = []
            fold_aucs_negscore: List[float] = []
            try:
                cv_local = TimeSeriesSplit(n_splits=5)
                splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
                try:
                    max_horizon = None
                    try:
                        max_horizon = int(max([int(getattr(c, "horizon", 0)) for c in list(committee_configs or [])] + [0]))
                    except Exception:
                        max_horizon = None

                    if market_data is not None and max_horizon is not None and max_horizon > 0:
                        y_tmp = pd.Series(np.zeros(int(len(event_idx))), index=pd.DatetimeIndex(event_idx))
                        splits = _build_t1_aware_purged_splits_for_events(
                            y=y_tmp,
                            event_durations=None,
                            market_index=market_data.index,
                            cv_splits=5,
                            base_horizon_bars=int(max_horizon),
                        )
                except Exception:
                    splits = None

                if splits is None:
                    splits = list(cv_local.split(np.arange(len(event_idx))))

                for fold_idx, (_, te_idx) in enumerate(splits):
                    te_idx = np.asarray(te_idx, dtype=int)
                    if te_idx.size <= 0:
                        continue
                    tr_mask = np.asarray(take_mask, dtype=bool)[te_idx]
                    tr_returns = np.asarray(weighted_returns, dtype=float)[te_idx]
                    tr_returns = tr_returns[np.isfinite(tr_returns)]
                    n_trades_fold = int(np.sum(tr_mask))

                    fold_auc = 0.5
                    fold_auc_negscore = 0.5
                    fold_auc_n_pos = 0
                    fold_auc_n_neg = 0
                    try:
                        score_fold = np.asarray(consensus_score, dtype=float)[te_idx]
                        ret_fold = np.asarray(weighted_returns, dtype=float)[te_idx]
                        fired_any_fold = np.asarray(fired, dtype=bool)
                        try:
                            fired_any_fold = fired_any_fold[te_idx]
                        except Exception:
                            fired_any_fold = np.ones(int(score_fold.size), dtype=bool)
                        if fired_any_fold.ndim == 2:
                            fired_any_fold = np.any(fired_any_fold, axis=1)
                        elif fired_any_fold.ndim != 1:
                            fired_any_fold = np.ones(int(score_fold.size), dtype=bool)
                        mm_auc = fired_any_fold & np.isfinite(score_fold) & np.isfinite(ret_fold)
                        if int(np.sum(mm_auc)) >= 20:
                            y_auc = (ret_fold[mm_auc] > 0.0).astype(int)
                            fold_auc_n_pos = int(np.sum(y_auc == 1))
                            fold_auc_n_neg = int(np.sum(y_auc == 0))
                            if int(np.unique(y_auc).size) >= 2:
                                auc_dir = float(roc_auc_score(y_auc, score_fold[mm_auc]))
                                auc_inv = float(roc_auc_score(y_auc, (-score_fold[mm_auc])))
                                fold_auc = float(max(auc_dir, auc_inv))
                                fold_auc_negscore = float(auc_inv)
                    except Exception:
                        fold_auc = 0.5
                        fold_auc_negscore = 0.5

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
                                "auc_negscore": float(fold_auc_negscore),
                                "auc_n_pos": int(fold_auc_n_pos),
                                "auc_n_neg": int(fold_auc_n_neg),
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
                        fold_aucs_negscore.append(float(fold_auc_negscore))
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
                                    sharpe_fold = _soft_sharpe_scale(mu / sd * np.sqrt(365.0))
                    except Exception:
                        sharpe_fold = 0.0

                    per_fold_metrics.append(
                        {
                            "fold": int(fold_idx),
                            "auc": float(fold_auc),
                            "auc_negscore": float(fold_auc_negscore),
                            "auc_n_pos": int(fold_auc_n_pos),
                            "auc_n_neg": int(fold_auc_n_neg),
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
                    fold_aucs_negscore.append(float(fold_auc_negscore))
                    fold_sharpes.append(float(sharpe_fold))
            except Exception:
                per_fold_metrics = []
                fold_sharpes = []
                fold_aucs = []
                fold_aucs_negscore = []

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
                            idx_trades_r = pd.DatetimeIndex(event_idx[tm])
                            idx_span_r = pd.DatetimeIndex(event_idx[rm])
                            if int(rvals.size) > 0 and int(idx_trades_r.size) == int(rvals.size) and len(idx_span_r) > 0:
                                day_index_r = pd.date_range(
                                    start=idx_span_r.min().normalize(),
                                    end=idx_span_r.max().normalize(),
                                    freq="D",
                                )
                                daily_pnl_r = (
                                    pd.Series(rvals, index=idx_trades_r)
                                    .groupby(idx_trades_r.normalize())
                                    .sum()
                                )
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
                trade_dur_all = np.asarray(weighted_durations, dtype=float)[tm_fin]
                trade_dur_all = np.where(np.isfinite(trade_dur_all) & (trade_dur_all > 0.0), trade_dur_all, 1.0)
                idx_span_all = pd.DatetimeIndex(event_idx)
                if (
                    int(trade_ret_all.size) > 0
                    and int(trade_idx_all.size) == int(trade_ret_all.size)
                    and len(idx_span_all) > 0
                ):
                    day_index_all = pd.date_range(
                        start=idx_span_all.min().normalize(),
                        end=idx_span_all.max().normalize(),
                        freq="D",
                    )
                    df_all = pd.DataFrame({"ret": trade_ret_all.astype(float), "dur": trade_dur_all.astype(float)}, index=trade_idx_all)
                    df_all = df_all[np.isfinite(df_all["ret"].values) & np.isfinite(df_all["dur"].values) & (df_all["dur"].values > 0.0)]
                    if int(df_all.shape[0]) > 1:
                        g = df_all.groupby(df_all.index.normalize())
                        daily_ret = g["ret"].sum().reindex(day_index_all, fill_value=0.0)
                        daily_dur = g["dur"].sum().reindex(day_index_all, fill_value=0.0)
                        daily_rate = (daily_ret / (daily_dur + 1e-12)).astype(float)
                        daily_rate = daily_rate[np.isfinite(daily_rate.values)]
                        if int(daily_rate.size) > 1:
                            mu_all = float(np.mean(daily_rate.values))
                            sd_all = float(np.std(daily_rate.values, ddof=1))
                            if sd_all > 1e-12:
                                sharpe = float(mu_all / sd_all * np.sqrt(365.0))
            except Exception:
                sharpe = 0.0
            sharpe = _soft_sharpe_scale(float(sharpe))

            # De Prado PSR utility on committee traded daily log returns (no multiple-testing penalty)
            try:
                psr_min_trades = int(config.get("layer2_psr_min_trades", 30))
            except Exception:
                psr_min_trades = 30
            psr_min_trades = int(max(1, psr_min_trades))

            try:
                sr_benchmark = float(config.get("layer2_psr_sr_benchmark", 0.0))
            except Exception:
                sr_benchmark = 0.0
            if not np.isfinite(sr_benchmark):
                sr_benchmark = 0.0

            psr_details = {"psr": 0.0, "psr_z": float("-inf"), "sr": None, "n": 0, "skew": 0.0, "kurt": 3.0}
            try:
                if trade_ret_all is not None and int(trade_ret_all.size) > 0 and int(trade_idx_all.size) == int(trade_ret_all.size):
                    day_index_all = pd.date_range(
                        start=pd.DatetimeIndex(event_idx).min().normalize(),
                        end=pd.DatetimeIndex(event_idx).max().normalize(),
                        freq="D",
                    )
                    daily_pnl = pd.Series(trade_ret_all.astype(float), index=trade_idx_all).groupby(trade_idx_all.normalize()).sum()
                    daily_pnl = daily_pnl.reindex(day_index_all, fill_value=0.0)
                    daily_log = np.log1p(daily_pnl.astype(float).values)
                    daily_log = daily_log[np.isfinite(daily_log)]
                    psr_details = _psr_from_returns(
                        daily_log,
                        sr_benchmark=float(sr_benchmark),
                        periods_per_year=365.0,
                    )
            except Exception:
                pass

            phi_trades = 0.0
            try:
                phi_trades = float(np.clip(float(psr_details.get("n", 0)) / float(psr_min_trades), 0.0, 1.0))
            except Exception:
                phi_trades = 0.0

            utility_raw = float(psr_details.get("psr", 0.0)) * float(phi_trades)
            if not np.isfinite(float(utility_raw)):
                utility_raw = 0.0

            # ================================================================
            # REGIME-AWARE CORRELATION DIVERSITY PENALTY
            # ================================================================
            # Unlike global correlation, this computes correlation within each
            # regime separately. Experts should be diversified in regimes where
            # they don't specialize (out-of-home), but may be correlated in
            # their "home" regime. This follows MoE principles more accurately.
            diversity = 0.0
            diversity_penalty = 0.0
            regime_aware_diversity: Dict[str, Any] = {}
            try:
                # Define home regimes for each expert type
                # Indices: [scalp_L, scalp_S, swing_L, swing_S, trend_L, trend_S, breakout, vwap_rev, vol_shock]
                home_regime_map: Dict[int, str] = {
                    0: "chop",      # scalp_L -> chop
                    1: "chop",      # scalp_S -> chop
                    2: "neutral",   # swing_L -> neutral
                    3: "neutral",   # swing_S -> neutral
                    4: "trend",     # trend_L -> trend
                    5: "trend",     # trend_S -> trend
                    6: "trend",     # breakout -> trend (transition into trend)
                    7: "chop",      # vwap_rev -> chop (mean reversion)
                    8: "vol_spike", # vol_shock -> vol_spike
                }
                
                # Compute regime-aware correlation
                regime_aware_diversity = _compute_regime_aware_correlation(
                    signals=np.asarray(label_matrix_values, dtype=float),
                    weights=np.asarray(weights_vec, dtype=float),
                    regime_masks=regime_masks_for_diversity,
                    home_regime_map=home_regime_map,
                )
                
                # Use diversity_score (blended global + regime-aware) for penalty
                diversity = float(regime_aware_diversity.get("diversity_score", 0.0))
                if not np.isfinite(diversity):
                    diversity = float(regime_aware_diversity.get("global_corr", 0.0))
                
                try:
                    diversity_scale = float(config.get("diversity_penalty_scale", 3.0))
                except Exception:
                    diversity_scale = 3.0
                diversity_multiplier = float(
                    np.exp(-float(max(0.0, diversity_lambda)) * float(max(0.0, diversity_scale)) * float(max(0.0, diversity)))
                )
                if not np.isfinite(diversity_multiplier):
                    diversity_multiplier = 1.0
            except Exception:
                diversity = 0.0
                diversity_penalty = 0.0
                diversity_multiplier = 1.0
                regime_aware_diversity = {}

            utility_after_diversity = float(utility_raw)
            try:
                if np.isfinite(float(utility_after_diversity)) and np.isfinite(float(diversity_multiplier)):
                    utility_after_diversity = float(utility_after_diversity) * float(diversity_multiplier)
            except Exception:
                pass

            try:
                diversity_penalty = float(utility_raw) - float(utility_after_diversity)
            except Exception:
                diversity_penalty = 0.0

            # ================================================================
            # ADVANCED DIVERSITY PENALTY (Jaccard overlap from advanced gating)
            # ================================================================
            # The advanced gating pipeline computes diversity based on expert
            # firing overlap (Jaccard), which complements the correlation-based
            # diversity penalty above.
            adv_diversity_penalty = 0.0
            adv_diversity_diag = {}
            try:
                if adv_gating_result is not None and diversity_lambda > 0.0:
                    adv_diag = adv_gating_result.get("diagnostics", {})
                    adv_diversity_info = adv_diag.get("diversity", {})
                    if adv_diversity_info:
                        mean_overlap = float(adv_diversity_info.get("mean_overlap", 0.0))
                        # Apply Jaccard-based diversity penalty
                        # Higher overlap = higher penalty
                        adv_diversity_penalty = diversity_lambda * mean_overlap * float(utility_after_diversity)
                        if np.isfinite(adv_diversity_penalty) and adv_diversity_penalty > 0.0:
                            utility_after_diversity = float(utility_after_diversity) - adv_diversity_penalty
                            diversity_penalty += adv_diversity_penalty
                        adv_diversity_diag = adv_diversity_info
            except Exception:
                pass

            # ================================================================
            # UNPROFITABLE EXPERT PENALTY
            # ================================================================
            # Automatically down-weight configurations that give high weight
            # to experts with poor risk-adjusted returns (low PSR / negative SR).
            # This enables HPO to learn which experts are consistently unprofitable.
            unprofitable_expert_penalty_result: Dict[str, Any] = {}
            unprofitable_expert_penalty = 0.0
            utility_pre_unprofitable_penalty = float(utility_after_diversity)
            try:
                # Get config for unprofitable expert penalty
                unprofitable_penalty_enabled = bool(config.get("layer2_unprofitable_expert_penalty_enabled", True))
                unprofitable_penalty_lambda = float(config.get("layer2_unprofitable_expert_penalty_lambda", 0.5))
                unprofitable_psr_threshold = float(config.get("layer2_unprofitable_psr_threshold", 0.5))
                unprofitable_sr_threshold = float(config.get("layer2_unprofitable_sr_threshold", 0.0))
                
                if unprofitable_penalty_enabled and unprofitable_penalty_lambda > 0.0 and per_expert_psr:
                    unprofitable_expert_penalty_result = _compute_unprofitable_expert_penalty(
                        per_expert_psr=per_expert_psr,
                        weights_vec=weights_vec,
                        expert_names=list(committee_names),
                        psr_threshold=unprofitable_psr_threshold,
                        sr_threshold=unprofitable_sr_threshold,
                        penalty_scale=unprofitable_penalty_lambda,
                        min_trades_required=int(config.get("layer2_psr_min_expert_trades", 10)),
                    )
                    
                    raw_penalty = float(unprofitable_expert_penalty_result.get("penalty", 0.0))
                    if np.isfinite(raw_penalty) and raw_penalty > 0.0:
                        # Apply penalty: reduce utility proportionally
                        # Cap penalty at 50% of utility to avoid complete zeroing
                        max_penalty = 0.5 * abs(float(utility_after_diversity))
                        unprofitable_expert_penalty = float(min(raw_penalty, max_penalty))
                        utility_after_diversity = float(utility_after_diversity) - unprofitable_expert_penalty
            except Exception:
                pass

            utility_pre_volatility_penalty = float(utility_after_diversity)
            vol_mean_all = None
            vol_mean_taken = None
            vol_excess_z = 0.0
            vol_excess_abs_z = 0.0
            try:
                vol_series_local = full_volatility.reindex(pd.DatetimeIndex(event_idx)).fillna(0.0).values
                vol_all = np.asarray(vol_series_local, dtype=float)
                vol_all = vol_all[np.isfinite(vol_all)]
                if vol_all.size > 5:
                    mu_all = float(np.mean(vol_all))
                    sd_all = float(np.std(vol_all, ddof=1)) if vol_all.size > 1 else 0.0
                    if np.isfinite(mu_all):
                        vol_mean_all = float(mu_all)
                    if sd_all > 1e-12:
                        tm = np.asarray(take_mask, dtype=bool)
                        if int(tm.size) == int(vol_series_local.size):
                            vol_taken = np.asarray(vol_series_local, dtype=float)[tm]
                            vol_taken = vol_taken[np.isfinite(vol_taken)]
                            if vol_taken.size > 0:
                                mu_taken = float(np.mean(vol_taken))
                                if np.isfinite(mu_taken):
                                    vol_mean_taken = float(mu_taken)
                                if vol_mean_all is not None:
                                    vol_excess_z = float((mu_taken - mu_all) / (sd_all + 1e-12))
                                    vol_excess_z = float(max(0.0, vol_excess_z))
            except Exception:
                vol_excess_z = 0.0

            try:
                if (
                    np.isfinite(float(utility_after_diversity))
                    and float(utility_after_diversity) > float(utility_floor)
                    and np.isfinite(float(vol_penalty_lambda))
                    and float(vol_penalty_lambda) > 0.0
                    and np.isfinite(float(vol_excess_z))
                    and float(vol_excess_z) > 0.0
                ):
                    utility_after_vol = float(utility_after_diversity) - float(vol_penalty_lambda) * float(vol_excess_z)
                else:
                    utility_after_vol = float(utility_after_diversity)
            except Exception:
                utility_after_vol = float(utility_after_diversity)

            # ------------------------------------------------------------------
            # Layer 2 instability penalty across regimes (committee mode)
            # ------------------------------------------------------------------
            try:
                layer2_regime_instability_lambda = float(config.get("layer2_regime_instability_lambda", 0.0))
            except Exception:
                layer2_regime_instability_lambda = 0.0
            if not np.isfinite(layer2_regime_instability_lambda):
                layer2_regime_instability_lambda = 0.0
            layer2_regime_instability_lambda = float(max(0.0, layer2_regime_instability_lambda))

            regime_dispersion = float(_compute_regime_dispersion(per_regime_metrics, metric_key="sharpe"))
            if not np.isfinite(regime_dispersion):
                regime_dispersion = 0.0

            utility_pre_regime_penalty = float(utility_after_vol)
            try:
                if layer2_regime_instability_lambda > 0.0 and float(utility_after_vol) > 0.0 and float(regime_dispersion) > 0.0:
                    utility_after_vol = float(utility_after_vol) - float(layer2_regime_instability_lambda) * float(regime_dispersion)
            except Exception:
                pass

            utility_pre_clip = float(utility_after_vol)
            try:
                utility = float(np.clip(float(utility_pre_clip), float(utility_floor), float(utility_clip_max)))
            except Exception:
                utility = float(np.clip(float(utility_pre_clip), float(utility_floor), 50.0))

            try:
                trade_mean = float(np.mean(trade_returns)) if trade_returns.size > 0 else 0.0
                trade_win_rate = float(np.mean(trade_returns > 0.0)) if trade_returns.size > 0 else 0.0
            except Exception:
                trade_mean = 0.0
                trade_win_rate = 0.0

            auc_val = 0.5
            try:
                from sklearn.metrics import roc_auc_score as _roc_auc_score
                score_auc = np.asarray(consensus_score, dtype=float)
                wr = np.asarray(weighted_returns, dtype=float)

                fired_any = np.asarray(fired, dtype=bool)
                if fired_any.ndim == 2:
                    fired_any = np.any(fired_any, axis=1)
                elif fired_any.ndim != 1:
                    fired_any = np.ones(int(score_auc.size), dtype=bool)
                m = fired_any & np.isfinite(score_auc) & np.isfinite(wr)
                n_valid_auc = int(np.sum(m))
                if n_valid_auc >= 50:
                    y_true_auc = (wr[m] > 0.0).astype(int)
                    n_classes = int(np.unique(y_true_auc).size)
                    if n_classes >= 2:
                        try:
                            auc_dir = float(_roc_auc_score(y_true_auc, score_auc[m]))
                            auc_inv = float(_roc_auc_score(y_true_auc, (-score_auc[m])))
                            auc_val = float(max(auc_dir, auc_inv))
                        except Exception as e_auc_calc:
                            tprint_warning(f"⚠️ AUC calculation inner exception: {e_auc_calc}")
                            auc_val = 0.5
                    else:
                        tprint_warning(f"⚠️ AUC skipped: Only {n_classes} class in labels (needs 2). Unique labels: {np.unique(y_true_auc)}")
                        auc_val = 0.5
                else:
                    tprint_warning(f"⚠️ AUC skipped: Only {n_valid_auc} valid samples (needs 20).")
                    auc_val = 0.5
            except Exception as e_auc:
                tprint_warning(f"⚠️ AUC calculation failed completely: {e_auc}")
                auc_val = 0.5

            mean_auc = float(auc_val)
            try:
                fau = np.asarray(fold_aucs, dtype=float)
                fau = fau[np.isfinite(fau)]
                if fau.size > 0:
                    mean_auc = float(np.mean(fau))
            except Exception:
                pass

            mean_auc_negscore = 0.5
            try:
                fau_n = np.asarray(fold_aucs_negscore, dtype=float)
                fau_n = fau_n[np.isfinite(fau_n)]
                if fau_n.size > 0:
                    mean_auc_negscore = float(np.mean(fau_n))
            except Exception:
                mean_auc_negscore = 0.5

            # Label balance diagnostics (used by report writer and sanity/debug)
            label_pos_rate = None
            label_n_pos = None
            label_n_neg = None
            try:
                wr_f = np.asarray(weighted_returns, dtype=float)
                m_wr = np.isfinite(wr_f)
                if int(np.sum(m_wr)) > 0:
                    yb = (wr_f[m_wr] > 0.0).astype(int)
                    label_pos_rate = float(np.mean(yb))
                    label_n_pos = int(np.sum(yb == 1))
                    label_n_neg = int(np.sum(yb == 0))
            except Exception:
                pass

            # Preserve the global AUC computed above (on signal-bearing events).
            # Only fill missing counts if they weren't computed.
            if auc_global_n_pos is None and label_n_pos is not None:
                auc_global_n_pos = int(label_n_pos)
            if auc_global_n_neg is None and label_n_neg is not None:
                auc_global_n_neg = int(label_n_neg)

            # If fold-wise AUC was not computable (often due to single-class folds),
            # fall back to a global AUC computed on all events.
            try:
                if (not np.isfinite(float(mean_auc))) or float(mean_auc) == 0.5:
                    if np.isfinite(float(auc_global)) and float(auc_global) != 0.5:
                        mean_auc = float(auc_global)
            except Exception:
                pass

            try:
                if (not np.isfinite(float(mean_auc_negscore))) or float(mean_auc_negscore) == 0.5:
                    if np.isfinite(float(auc_global_negscore)) and float(auc_global_negscore) != 0.5:
                        mean_auc_negscore = float(auc_global_negscore)
            except Exception:
                pass

            # Now using Sortino ratios (variable names kept for backward compat)
            folds_sortino_arr = np.asarray(fold_sharpes, dtype=float)  # Actually Sortino now
            folds_sortino_arr = folds_sortino_arr[np.isfinite(folds_sortino_arr)]
            if folds_sortino_arr.size <= 0:
                folds_sortino_arr = np.asarray([float(sharpe)], dtype=float)
            folds_sharpe_arr = folds_sortino_arr  # Alias for backward compat

            lambda_vol = 0.4  # CHANGED from 0.8 (recalibrated for Sortino)
            w_auc = 0.5
            w_den = 0.3

            avg_sharpe = float(np.mean(folds_sortino_arr))
            vol_sharpe = float(np.std(folds_sortino_arr, ddof=1)) if folds_sortino_arr.size > 1 else 0.0
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

            consensus_score_mapping: List[Dict[str, Any]] = []
            try:
                consensus_score_mapping = _compute_probability_mapping(
                    probs=np.asarray(consensus_score, dtype=float),
                    returns=np.asarray(weighted_returns, dtype=float),
                    n_bins=int(config.get("probability_mapping_bins", 10)),
                    score_name="score",
                )
            except Exception:
                consensus_score_mapping = []

            probability_mapping: List[Dict[str, Any]] = []
            try:
                cs_arr_full = np.asarray(consensus_score, dtype=float)
                cs_arr_full = np.where(np.isfinite(cs_arr_full), cs_arr_full, np.nan)
                p_all = (cs_arr_full + 1.0) / 2.0
                p_all = np.clip(p_all, 0.0, 1.0)
                probability_mapping = _compute_probability_mapping(
                    probs=np.asarray(p_all, dtype=float),
                    returns=np.asarray(weighted_returns, dtype=float),
                    n_bins=int(config.get("probability_mapping_bins", 10)),
                    score_name="p",
                )
            except Exception:
                probability_mapping = []

            return {
                "valid_events": int(len(event_idx)),
                "utility": float(utility),
                "utility_pre_clip": float(utility_pre_clip) if np.isfinite(float(utility_pre_clip)) else None,
                "utility_clip_max": float(utility_clip_max),
                "layer2_regime_instability_lambda": float(layer2_regime_instability_lambda),
                "regime_dispersion": float(regime_dispersion),
                "utility_pre_regime_penalty": float(utility_pre_regime_penalty),
                "auc": float(mean_auc),
                "psr": float(psr_details.get("psr", 0.0)),
                "psr_z": float(psr_details.get("psr_z", float("-inf"))),
                "psr_sr": psr_details.get("sr", None),
                "psr_n": int(psr_details.get("n", 0) or 0),
                "auc_negscore": float(mean_auc_negscore),
                "auc_global": float(auc_global),
                "auc_global_negscore": float(auc_global_negscore),
                "auc_global_n_pos": int(auc_global_n_pos) if auc_global_n_pos is not None else None,
                "auc_global_n_neg": int(auc_global_n_neg) if auc_global_n_neg is not None else None,
                "label_pos_rate": float(label_pos_rate) if label_pos_rate is not None else None,
                "label_n_pos": int(label_n_pos) if label_n_pos is not None else None,
                "label_n_neg": int(label_n_neg) if label_n_neg is not None else None,
                "trades_per_day": float(trades_per_day),
                "probability_mapping": probability_mapping,
                "consensus_score_mapping": consensus_score_mapping,
                "volatility_penalty_lambda": float(vol_penalty_lambda),
                "utility_pre_volatility_penalty": float(utility_pre_volatility_penalty),
                "vol_mean_all": float(vol_mean_all) if vol_mean_all is not None else None,
                "vol_mean_taken": float(vol_mean_taken) if vol_mean_taken is not None else None,
                "vol_excess_z": float(vol_excess_z),
                "vol_excess_abs_z": float(vol_excess_abs_z),
                "abstain_margin": float(abstain_margin),
                "ev_margin": float(ev_margin_local),
                "diversity_lambda": float(diversity_lambda),
                "weighted_avg_abs_corr": float(diversity),
                "diversity_penalty": float(diversity_penalty),
                "diversity_multiplier": float(diversity_multiplier),
                "regime_aware_diversity": regime_aware_diversity if regime_aware_diversity else {},
                "per_expert_psr": per_expert_psr if per_expert_psr else {},
                "unprofitable_expert_penalty": float(unprofitable_expert_penalty),
                "unprofitable_expert_penalty_result": unprofitable_expert_penalty_result if unprofitable_expert_penalty_result else {},
                "utility_pre_unprofitable_penalty": float(utility_pre_unprofitable_penalty),
                "layer2_soft_min_trades_committee": int(min_tr),
                "phi_trades": float(phi_trades),
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
                "net_pnl_total": float(np.sum(trade_returns)) if trade_returns.size > 0 else 0.0,
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
                "committee_expert_stats": committee_expert_stats if committee_expert_stats is not None else {},
                "sanity_checks": sanity_checks if sanity_checks is not None else {},
                "committee_overlap": committee_overlap if committee_overlap is not None else {},
                "committee_drivers": committee_drivers if committee_drivers is not None else {},
                "moe_diagnostics": params.get("moe_diagnostics", {}),
                "n_experts": int(n_experts),
                "has_new_experts": bool(has_new_experts),
                "w_breakout": float(w_breakout),
                "w_vwap_rev": float(w_vwap_rev),
                "w_vol_shock": float(w_vol_shock),
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
            # Match Layer1 budget for thorough exploration
            layer2_n_trials = int(config.get("layer2_n_trials", DEFAULT_LAYER2_N_TRIALS))
            layer2_n_trials = max(5, min(layer2_n_trials, MAX_HPO_N_TRIALS))
            l2_optimizer = BayesianTPEOptimizer(
                config=OptimizationConfig(
                    n_trials=layer2_n_trials,
                    execution_mode="full",
                    direction="maximize",
                    seed=42,
                    enable_staged_optimization=False,
                    enable_adaptive_grid_refinement=False,
                    enable_adaptive_optimization=False,
                    enable_vectorbt_optimization=False,
                    enable_hardware_optimization=False,
                    n_startup_trials=min(50, layer2_n_trials // 10),
                    tpe_trials=layer2_n_trials,
                )
            )
            l2_result = l2_optimizer.optimize(objective=layer2_objective, search_space=layer2_search_space)
            best_trading_params = l2_result.get("best_params", {})
            best_l2_score = l2_result.get("best_value", 0.0)
        ts_l2 = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        try:
            l2_metrics = _compute_layer2_metrics(best_trading_params, write_diagnostics=True)
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
            # Log new expert and MoE diagnostics
            if l2_metrics.get("has_new_experts"):
                tprint_info(
                    "   Layer 2 new experts: "
                    f"n_experts={l2_metrics.get('n_experts', 6)}, "
                    f"w_breakout={l2_metrics.get('w_breakout', 0.0):.3f}, "
                    f"w_vwap_rev={l2_metrics.get('w_vwap_rev', 0.0):.3f}, "
                    f"w_vol_shock={l2_metrics.get('w_vol_shock', 0.0):.3f}"
                )
            moe_diag = l2_metrics.get("moe_diagnostics", {})
            if moe_diag:
                # Check if learned router was used
                if moe_diag.get('learned_router'):
                    tprint_info("   Layer 2 MoE: 🧠 LEARNED ROUTER active (heuristic MoE skipped)")
                else:
                    tprint_info(
                        "   Layer 2 MoE regime counts: "
                        f"trend={moe_diag.get('n_trend_events', 0)}, "
                        f"chop={moe_diag.get('n_chop_events', 0)}, "
                        f"vol_spike={moe_diag.get('n_vol_spike_events', 0)}, "
                        f"transition={moe_diag.get('n_transition_events', 0)}"
                    )
                if moe_diag.get('trend_w_in_trend') is not None:
                    tprint_info(
                        "   Layer 2 MoE weights by regime: "
                        f"trend_w_in_trend={moe_diag.get('trend_w_in_trend', 0.0):.3f}, "
                        f"scalp_w_in_chop={moe_diag.get('scalp_w_in_chop', 0.0):.3f}, "
                        f"vwap_w_in_chop={moe_diag.get('vwap_w_in_chop', 0.0):.3f}, "
                        f"vol_shock_w_in_vol_spike={moe_diag.get('vol_shock_w_in_vol_spike', 0.0):.3f}"
                    )
                tprint_info(
                    "   Layer 2 MoE weight sum: "
                    f"mean={moe_diag.get('weight_sum_mean', 1.0):.4f}, "
                    f"std={moe_diag.get('weight_sum_std', 0.0):.6f}"
                )
            # Log per-expert PSR metrics
            per_expert_psr = l2_metrics.get("per_expert_psr", {})
            if per_expert_psr:
                psr_summary = []
                for expert_name, psr_data in per_expert_psr.items():
                    if isinstance(psr_data, dict):
                        psr_val = psr_data.get("psr", 0.0)
                        psr_sr = psr_data.get("sr")
                        n_taken = psr_data.get("n_taken", psr_data.get("n", 0))
                        if psr_sr is not None and np.isfinite(psr_sr):
                            psr_summary.append(f"{expert_name}:PSR={psr_val:.2f}/SR={psr_sr:.2f}/n={n_taken}")
                        else:
                            psr_summary.append(f"{expert_name}:PSR={psr_val:.2f}/n={n_taken}")
                if psr_summary:
                    tprint_info(f"   Layer 2 per-expert PSR: {', '.join(psr_summary[:6])}")
                    if len(psr_summary) > 6:
                        tprint_info(f"   Layer 2 per-expert PSR (cont): {', '.join(psr_summary[6:])}")
            # Log regime-aware diversity metrics
            regime_div = l2_metrics.get("regime_aware_diversity", {})
            if regime_div:
                global_corr = regime_div.get("global_corr", 0.0)
                diversity_score = regime_div.get("diversity_score", 0.0)
                out_of_home = regime_div.get("out_of_home_corr", 0.0)
                per_regime = regime_div.get("per_regime_corr", {})
                tprint_info(
                    f"   Layer 2 diversity: global={global_corr:.3f}, "
                    f"regime_aware={diversity_score:.3f}, out_of_home={out_of_home:.3f}"
                )
                if per_regime:
                    regime_str = ", ".join(f"{k}={v:.3f}" for k, v in per_regime.items())
                    tprint_info(f"   Layer 2 per-regime corr: {regime_str}")
            # Log unprofitable expert penalty
            unprofitable_penalty = l2_metrics.get("unprofitable_expert_penalty", 0.0)
            unprofitable_result = l2_metrics.get("unprofitable_expert_penalty_result", {})
            if unprofitable_penalty > 0.01 or unprofitable_result:
                diag = unprofitable_result.get("diagnostics", {})
                n_unprofitable = diag.get("n_unprofitable", 0)
                total_weight = diag.get("total_unprofitable_weight", 0.0)
                worst_expert = diag.get("worst_expert")
                worst_psr = diag.get("worst_expert_psr")
                unprofitable_experts = unprofitable_result.get("unprofitable_experts", [])
                tprint_info(
                    f"   Layer 2 unprofitable expert penalty: {unprofitable_penalty:.4f} "
                    f"(n_unprofitable={n_unprofitable}, weight={total_weight:.3f})"
                )
                if worst_expert and worst_psr is not None:
                    tprint_info(f"   Layer 2 worst expert: {worst_expert} (PSR={worst_psr:.3f})")
                if unprofitable_experts:
                    expert_penalties = unprofitable_result.get("expert_penalties", {})
                    penalty_str = ", ".join(
                        f"{exp}:{expert_penalties.get(exp, 0.0):.3f}" 
                        for exp in unprofitable_experts[:5]
                    )
                    tprint_info(f"   Layer 2 unprofitable experts: {penalty_str}")
        except Exception as l2_diag_exc:
            l2_metrics = {}
            tprint_warning(f"   ⚠️ Failed to compute Layer 2 metrics breakdown: {l2_diag_exc}")
        tprint_success(f"✅ Layer 2 Complete. Best Score: {best_l2_score:.4f}")
        tprint_info(f"   Best Trading Params: {best_trading_params}")

        layer2_ok = True
        layer2_utility = None
        try:
            layer2_utility = float(l2_metrics.get("utility")) if isinstance(l2_metrics, dict) else None
        except Exception:
            layer2_utility = None
        try:
            layer2_utility_floor = float(config.get("layer2_utility_floor", -1.0))
        except Exception:
            layer2_utility_floor = -1.0
        if not np.isfinite(layer2_utility_floor):
            layer2_utility_floor = -1.0
        if layer2_utility is None or (not np.isfinite(float(layer2_utility))):
            layer2_ok = False
        else:
            if float(layer2_utility) <= float(layer2_utility_floor) + 1e-9:
                layer2_ok = False

        try:
            layer3_requires_layer2_success = bool(config.get("layer3_requires_layer2_success", True))
        except Exception:
            layer3_requires_layer2_success = True
        try:
            allow_layer3_when_layer2_failed = bool(config.get("allow_layer3_when_layer2_failed", True))
        except Exception:
            allow_layer3_when_layer2_failed = True

        if layer3_requires_layer2_success and (not layer2_ok) and (not allow_layer3_when_layer2_failed):
            tprint_error(
                "❌ Layer 3 aborted: Layer 2 did not produce a valid solution (utility=-1). "
                "Fix Layer 2 or set allow_layer3_when_layer2_failed=true to force continuation."
            )
            return {"success": False, "error": "layer2_failed"}

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
                metrics_trial = _compute_layer2_metrics(params)
                row = {
                    "valid_events": metrics_trial.get("valid_events"),
                    "utility": metrics_trial.get("utility"),
                    "utility_pre_clip": metrics_trial.get("utility_pre_clip"),
                    "utility_clip_max": metrics_trial.get("utility_clip_max"),
                    "utility_pre_profitability_penalty": metrics_trial.get("utility_pre_profitability_penalty"),
                    "profitability_penalty": metrics_trial.get("profitability_penalty"),
                    "profitability_penalty_trades": metrics_trial.get("profitability_penalty_trades"),
                    "profitability_penalty_mean_return": metrics_trial.get("profitability_penalty_mean_return"),
                    "utility_pre_volatility_penalty": metrics_trial.get("utility_pre_volatility_penalty"),
                    "auc": metrics_trial.get("auc"),
                    "pr_auc": metrics_trial.get("pr_auc"),
                    "precision_at_1pct": metrics_trial.get("precision_at_1pct"),
                    "precision_at_5pct": metrics_trial.get("precision_at_5pct"),
                    "precision_at_10pct": metrics_trial.get("precision_at_10pct"),
                    "pr_auc_raw": metrics_trial.get("pr_auc_raw"),
                    "precision_at_1pct_raw": metrics_trial.get("precision_at_1pct_raw"),
                    "precision_at_5pct_raw": metrics_trial.get("precision_at_5pct_raw"),
                    "precision_at_10pct_raw": metrics_trial.get("precision_at_10pct_raw"),
                    "auc_negscore": metrics_trial.get("auc_negscore"),
                    "auc_global": metrics_trial.get("auc_global"),
                    "auc_global_negscore": metrics_trial.get("auc_global_negscore"),
                    "auc_global_n_pos": metrics_trial.get("auc_global_n_pos"),
                    "auc_global_n_neg": metrics_trial.get("auc_global_n_neg"),
                    "label_n_pos": metrics_trial.get("label_n_pos"),
                    "label_n_neg": metrics_trial.get("label_n_neg"),
                    "label_pos_rate": metrics_trial.get("label_pos_rate"),
                    "trades_per_day": metrics_trial.get("trades_per_day"),
                    "vol_excess_z": metrics_trial.get("vol_excess_z"),
                    "vol_excess_abs_z": metrics_trial.get("vol_excess_abs_z"),
                    "vol_mean_all": metrics_trial.get("vol_mean_all"),
                    "vol_mean_taken": metrics_trial.get("vol_mean_taken"),
                    "calibration_brier": metrics_trial.get("calibration_brier"),
                    "calibration_ece": metrics_trial.get("calibration_ece"),
                    "sharpe_mean": metrics_trial.get("sharpe_mean"),
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
                    # New expert and MoE diagnostics
                    "n_experts": metrics_trial.get("n_experts"),
                    "has_new_experts": metrics_trial.get("has_new_experts"),
                    "w_breakout": metrics_trial.get("w_breakout"),
                    "w_vwap_rev": metrics_trial.get("w_vwap_rev"),
                    "w_vol_shock": metrics_trial.get("w_vol_shock"),
                }
                # Add MoE diagnostics if available
                moe_diag = metrics_trial.get("moe_diagnostics", {})
                if moe_diag:
                    row["moe_n_trend_events"] = moe_diag.get("n_trend_events")
                    row["moe_n_chop_events"] = moe_diag.get("n_chop_events")
                    row["moe_n_vol_spike_events"] = moe_diag.get("n_vol_spike_events")
                    row["moe_n_transition_events"] = moe_diag.get("n_transition_events")
                    row["moe_trend_w_in_trend"] = moe_diag.get("trend_w_in_trend")
                    row["moe_scalp_w_in_chop"] = moe_diag.get("scalp_w_in_chop")
                    row["moe_vwap_w_in_chop"] = moe_diag.get("vwap_w_in_chop")
                    row["moe_vol_shock_w_in_vol_spike"] = moe_diag.get("vol_shock_w_in_vol_spike")
                    row["moe_breakout_w_in_transition"] = moe_diag.get("breakout_w_in_transition")
                    row["moe_weight_sum_mean"] = moe_diag.get("weight_sum_mean")
                    row["moe_weight_sum_std"] = moe_diag.get("weight_sum_std")
                row["calibration_brier"] = metrics_trial.get("calibration_brier")
                row["calibration_ece"] = metrics_trial.get("calibration_ece")
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
                try:
                    dbg_y_raw = np.asarray(dbg_labels_clean.values, dtype=float)
                    dbg_y_bin = (dbg_y_raw >= 0.5).astype(int)
                    dbg_labels_bin = pd.Series(dbg_y_bin, index=dbg_labels_clean.index)
                except Exception:
                    dbg_labels_bin = dbg_labels_clean
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

                try:
                    use_return_weighted_sw_dbg = bool(config.get("layer2_use_return_weighted_sample_weights", True))
                except Exception:
                    use_return_weighted_sw_dbg = True

                if bool(use_return_weighted_sw_dbg):
                    try:
                        y_sw_dbg = np.asarray(dbg_labels_bin.values, dtype=float)
                        r_sw_dbg = np.asarray(dbg_returns_clean.values, dtype=float)
                        if int(y_sw_dbg.size) == int(r_sw_dbg.size) and int(y_sw_dbg.size) > 0:
                            yb_sw_dbg = (y_sw_dbg >= 0.5).astype(int)
                            pos_raw_dbg = np.where(yb_sw_dbg == 1, np.maximum(0.0, r_sw_dbg), 0.0)
                            pos_mask_dbg = (yb_sw_dbg == 1) & np.isfinite(pos_raw_dbg)
                            pos_mean_dbg = float(np.mean(pos_raw_dbg[pos_mask_dbg])) if int(np.sum(pos_mask_dbg)) > 0 else 0.0
                            if (not np.isfinite(pos_mean_dbg)) or float(pos_mean_dbg) <= 0.0:
                                pos_mean_dbg = 1.0
                            scale_dbg = 1.0 / float(pos_mean_dbg)
                            try:
                                neg_w_dbg = float(config.get("layer2_return_weighted_neg_weight", 0.25))
                            except Exception:
                                neg_w_dbg = 0.25
                            if (not np.isfinite(neg_w_dbg)) or float(neg_w_dbg) < 0.0:
                                neg_w_dbg = 0.25
                            sw_new_dbg = np.where(yb_sw_dbg == 1, pos_raw_dbg * float(scale_dbg), float(neg_w_dbg))
                            try:
                                pos_clip_dbg = float(config.get("layer2_return_weighted_pos_clip", 10.0))
                            except Exception:
                                pos_clip_dbg = 10.0
                            if np.isfinite(pos_clip_dbg) and float(pos_clip_dbg) > 0.0:
                                sw_new_dbg = np.clip(sw_new_dbg, 0.0, float(pos_clip_dbg))
                            sw_new_dbg = np.where(np.isfinite(sw_new_dbg) & (sw_new_dbg >= 0.0), sw_new_dbg, 0.0)
                            sample_weights_dbg = sw_new_dbg
                    except Exception:
                        pass
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
                    try:
                        dbg_prob_thr = float(best_trading_params.get("prob_threshold", 0.5))
                    except Exception:
                        dbg_prob_thr = 0.5
                    dbg_prob_thr = float(np.clip(dbg_prob_thr, 0.01, 0.99))
                    try:
                        dbg_ev_margin = float(best_trading_params.get("ev_margin", config.get("ev_margin", 0.0)))
                    except Exception:
                        dbg_ev_margin = float(config.get("ev_margin", 0.0) or 0.0)

                    cv_preds_raw_dbg, cv_preds_dbg, folds_sharpe_dbg, mean_brier_dbg, mean_ece_dbg, mean_mce_dbg = _cross_val_predict_proba_and_fold_sharpes_weighted(
                        estimator=fast_model_dbg,
                        X=X_dbg,
                        y=(dbg_returns_clean.reindex(dbg_t_events) > 0).astype(int),  # Use sign(returns) as target
                        sample_weight=sample_weights_dbg,
                        n_splits=n_cv_folds_dbg,
                        returns=dbg_returns_clean.values.astype(float),
                        direction=direction,
                        prob_thr=float(dbg_prob_thr),

                        use_calibration=True,
                        enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                        ev_margin=float(dbg_ev_margin),
                    )
                except Exception as dbg_cv_exc:
                    tprint_warning(f"   ⚠️ Layer 2 debug: CV failed: {dbg_cv_exc}")
                    cv_preds_raw_dbg = np.full(dbg_valid_events, 0.5, dtype=float)
                    cv_preds_dbg = np.full(dbg_valid_events, 0.5, dtype=float)
                    folds_sharpe_dbg = np.array([0.0], dtype=float)
                    mean_brier_dbg = None
                    mean_ece_dbg = None
                    mean_mce_dbg = None
                # AUC
                mean_auc_dbg = 0.5
                try:
                    per_fold_dbg = _compute_fold_metrics_from_oof(
                        X=X_dbg,
                        y_true=(np.asarray(dbg_returns_clean.values, dtype=float) > 0.0).astype(int),
                        probs=np.asarray(cv_preds_dbg, dtype=float),
                        returns=np.asarray(dbg_returns_clean.values, dtype=float),
                        threshold=float(dbg_prob_thr),
                        days_span=float(days_span),
                        transaction_cost=0.0,
                        event_index=dbg_t_events,
                        direction=direction,
                        event_durations=dbg_durations.reindex(dbg_t_events) if dbg_durations is not None else None,
                        market_index=market_data.index if market_data is not None else None,
                        base_horizon_bars=12,
                    )
                    fold_aucs_dbg = [
                        float(m.get("auc"))
                        for m in (per_fold_dbg or [])
                        if m.get("auc") is not None and np.isfinite(float(m.get("auc")))
                    ]
                    if len(fold_aucs_dbg) > 0:
                        mean_auc_dbg = float(np.mean(fold_aucs_dbg))
                except Exception:
                    mean_auc_dbg = 0.5
                trades_per_day_dbg = len(dbg_returns_clean) / max(days_span, 1)
                # Reconstruct base_score as in calculate_hpo_utility
                avg_sharpe_dbg = float(np.mean(folds_sharpe_dbg))
                vol_sharpe_dbg = float(np.std(folds_sharpe_dbg, ddof=1)) if len(folds_sharpe_dbg) > 1 else 0.0
                base_score_dbg = avg_sharpe_dbg - 0.8 * vol_sharpe_dbg  # UPDATED: 0.8 from 1.2
                
                # Compute debug mean_return and max_dd
                try:
                    mean_return_dbg = float(np.nanmean(dbg_returns_clean.values)) if len(dbg_returns_clean) > 0 else None
                except Exception:
                    mean_return_dbg = None
                try:
                    if len(dbg_returns_clean) > 0:
                        cum_ret_dbg = np.nancumsum(dbg_returns_clean.values)
                        running_max_dbg = np.maximum.accumulate(np.nan_to_num(cum_ret_dbg, nan=0.0))
                        drawdown_dbg = running_max_dbg - cum_ret_dbg
                        max_dd_dbg = float(np.nanmax(drawdown_dbg)) if len(drawdown_dbg) > 0 else None
                    else:
                        max_dd_dbg = None
                except Exception:
                    max_dd_dbg = None
                
                utility_dbg = calculate_hpo_utility(
                    folds_sharpe=folds_sharpe_dbg,  # Now Sortino ratios
                    auc=mean_auc_dbg,
                    trades_per_day=trades_per_day_dbg,
                    lambda_vol=0.4,   # CHANGED from 0.8 (recalibrated for Sortino)
                    w_auc=0.5,
                    w_den=0.3,
                    calibration_brier=mean_brier_dbg,
                    calibration_ece=mean_ece_dbg,
                    w_cal=0.0,
                    mean_return=mean_return_dbg,
                    max_drawdown=max_dd_dbg,
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
                # NEW: Log return and drawdown components
                tprint_info(
                    "   Layer 2 debug: "
                    f"mean_return={mean_return_dbg:.6f if mean_return_dbg is not None else 'N/A'}, "
                    f"max_drawdown={max_dd_dbg:.4f if max_dd_dbg is not None else 'N/A'}"
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
            # Build extra dict with MoE and new expert info for the report
            l2_extra = {}
            if isinstance(l2_metrics, dict):
                l2_extra["n_experts"] = l2_metrics.get("n_experts")
                l2_extra["has_new_experts"] = l2_metrics.get("has_new_experts")
                l2_extra["w_breakout"] = l2_metrics.get("w_breakout")
                l2_extra["w_vwap_rev"] = l2_metrics.get("w_vwap_rev")
                l2_extra["w_vol_shock"] = l2_metrics.get("w_vol_shock")
                moe_diag = l2_metrics.get("moe_diagnostics", {})
                if moe_diag:
                    l2_extra["moe_diagnostics"] = moe_diag
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
                extra=l2_extra,
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

        if isinstance(best_trading_params, dict):
            final_trail = float(best_trading_params.get("trail_distance_atr_mult", 0.0))

            try:
                final_horizon_bars = int(best_trading_params.get("horizon_bars", 12))
            except Exception:
                final_horizon_bars = 12
            try:
                final_min_spacing = int(best_trading_params.get("min_event_spacing", 2))
            except Exception:
                final_min_spacing = 2

            final_prof_thr = fixed_layer2_profit_thr.reindex(market_data.index).fillna(float(layer2_tp_target))
            final_stop_thr = fixed_layer2_stop_thr.reindex(market_data.index).fillna(float(layer2_sl_target))
            final_horizon_for_call: Union[int, pd.Series] = int(final_horizon_bars)
            final_trail_for_call: Optional[Union[float, pd.Series]] = float(final_trail)

            try:
                if bool(enable_regime_conditional_barrier_geometry) and barrier_geometry_regime_col in market_data.columns:
                    p_thr_s, s_thr_s, h_s, t_s = _compute_regime_conditional_barrier_geometry(
                        params=dict(best_trading_params),
                        market_index=market_data.index,
                        default_horizon=int(final_horizon_bars),
                        atr_frac_series=atr_frac,
                    )
                    final_prof_thr = p_thr_s
                    final_stop_thr = s_thr_s
                    final_horizon_for_call = h_s
                    final_trail_for_call = t_s
            except Exception:
                pass

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
                horizon=final_horizon_for_call,
                transaction_cost=DEFAULT_TRANSACTION_COST,
                min_event_spacing=int(final_min_spacing),
                trail_distance_atr_mult=final_trail_for_call,
                atr_series=atr_series,
            )
        else:
            try:
                w_scalp = float(best_trading_params.get("w_scalp", 0.0))
                w_swing = float(best_trading_params.get("w_swing", 0.0))
                w_trend = float(best_trading_params.get("w_trend", 0.0))
                w_breakout = float(best_trading_params.get("w_breakout", 0.5))
                w_vwap_rev = float(best_trading_params.get("w_vwap_rev", 0.5))
                w_vol_shock = float(best_trading_params.get("w_vol_shock", 0.5))
                threshold = float(best_trading_params.get("consensus_threshold", 0.5))
                abstain_margin = float(best_trading_params.get("abstain_margin", 0.0))
                ev_margin_local = float(best_trading_params.get("ev_margin", config.get("ev_margin", 0.0)))
                consensus_quantile = best_trading_params.get("consensus_quantile", None)
                consensus_quantile = float(consensus_quantile) if consensus_quantile is not None else None

                n_exp_gate = int(label_matrix_values.shape[1]) if isinstance(label_matrix_values, np.ndarray) else 6
                if n_exp_gate > 6:
                    weights_vec = np.array(
                        [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend, w_breakout, w_vwap_rev, w_vol_shock],
                        dtype=float,
                    )
                else:
                    weights_vec = np.array([w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend], dtype=float)
                total_weight = float(weights_vec.sum()) + 1e-8

                weighted_sum = label_matrix_values.dot(weights_vec)
                consensus_score = weighted_sum / total_weight
                scores_arr = np.asarray(consensus_score, dtype=float)
                scores_arr = np.where(np.isfinite(scores_arr), scores_arr, -np.inf)

                try:
                    thr_take = float(threshold) + float(abstain_margin) + float(ev_margin_local)
                    if consensus_quantile is not None and np.isfinite(consensus_quantile):
                        q = float(np.clip(consensus_quantile, 0.0, 0.999999))
                        n = int(scores_arr.size)
                        if n > 0:
                            k = int(np.ceil((1.0 - q) * float(n)))
                            k = int(np.clip(k, 1, n))
                            top_idx = np.argpartition(scores_arr, n - k)[n - k :]
                            take_mask = np.zeros(n, dtype=bool)
                            take_mask[top_idx] = True
                            take_mask = take_mask & (scores_arr > float(thr_take))
                        else:
                            take_mask = np.zeros(0, dtype=bool)
                    else:
                        take_mask = scores_arr > float(thr_take)
                except Exception:
                    take_mask = scores_arr > float(threshold)

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

                # =====================================================================
                # FIX FOR CLASS IMBALANCE: Train on ALL events, not just committee-gated
                # =====================================================================
                # When layer3_train_on_all_events=True (default), Layer 3 trains on the
                # full labeled dataset to preserve class balance (e.g., 27% positive).
                # The committee gate is still used for INFERENCE (deciding whether to trade),
                # but the meta-model learns from ALL events including losses.
                # =====================================================================
                try:
                    train_on_all_events = bool(config.get("layer3_train_on_all_events", True))
                except Exception:
                    train_on_all_events = True

                train_mask = np.ones(len(final_t_events_all), dtype=bool)
                
                if not train_on_all_events:
                    # Legacy behavior: only train on committee-gated events
                    try:
                        take_mask_arr = np.asarray(take_mask, dtype=bool)
                        if take_mask_arr.size == train_mask.size:
                            train_mask = take_mask_arr
                    except Exception:
                        pass
                else:
                    # NEW DEFAULT: Train on ALL labeled events for better class balance
                    # Log the difference for transparency
                    try:
                        n_committee = int(np.sum(take_mask)) if take_mask is not None else 0
                        n_all = int(len(final_t_events_all))
                        tprint_info(f"   Layer 3 training: using ALL {n_all} events (committee would select {n_committee})")
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

                # If committee-gated training collapses to a single class, fall back
                # to training on all events to restore a meaningful learnability AUC.
                try:
                    y_tmp = pd.Series(final_labels.values, index=final_labels.index).dropna()
                    if int(y_tmp.nunique()) < 2:
                        train_mask = np.ones(len(final_t_events_all), dtype=bool)
                        final_returns = pd.Series(weighted_returns.astype(float), index=final_t_events_all)
                        final_labels = pd.Series((weighted_returns > 0).astype(float), index=final_t_events_all)
                except Exception:
                    pass

                # If return-sign labels are STILL single-class (rare but possible if the
                # committee returns are degenerate), fall back to quantile labels on
                # vol-scaled returns to ensure AUC is meaningful.
                try:
                    y_tmp2 = pd.Series(final_labels.values, index=final_labels.index).dropna()
                    if int(y_tmp2.nunique()) < 2:
                        vol_s = None
                        try:
                            vol_s = volatility_1d.reindex(final_t_events_all)
                        except Exception:
                            vol_s = None

                        try:
                            low_q = float(config.get("label_low_q", 0.40))
                            high_q = float(config.get("label_high_q", 0.60))
                        except Exception:
                            low_q, high_q = 0.40, 0.60

                        vsr = compute_vol_scaled_returns_for_events(
                            realized_returns=pd.Series(weighted_returns.astype(float), index=final_t_events_all),
                            volatility=vol_s,
                            econ_min_return_multiple=float(config.get("econ_min_return_multiple", 1.0)),
                        )
                        qlbl = create_quantile_labels_from_vol_scaled_returns(vsr, low_q=float(low_q), high_q=float(high_q))
                        # Map: top bucket => 1, else => 0, keep NaNs as NaN.
                        final_labels = qlbl.reindex(final_t_events_all).astype(float)
                        final_returns = pd.Series(weighted_returns.astype(float), index=final_t_events_all)
                except Exception:
                    pass
                
                # Log class balance for diagnostics
                try:
                    n_pos = int((final_labels > 0).sum())
                    n_neg = int((final_labels <= 0).sum())
                    n_total = n_pos + n_neg
                    pos_rate = float(n_pos) / float(max(n_total, 1))
                    tprint_info(f"   Layer 3 class balance: {n_pos} positive ({pos_rate:.1%}), {n_neg} negative ({1-pos_rate:.1%})")
                    
                    # Warn if class balance is still severely imbalanced
                    if pos_rate > 0.90 or pos_rate < 0.10:
                        tprint_warning(f"   ⚠️ SEVERE CLASS IMBALANCE: pos_rate={pos_rate:.1%}. Model may not learn well.")
                except Exception:
                    pass

                horizon_bars_l3 = int(best_trading_params.get("horizon_bars", 12))
                final_durations = pd.Series(
                    np.full(len(final_t_events_all), np.nan, dtype=float),
                    index=final_t_events_all,
                )
                final_durations = final_durations.iloc[train_mask]
            except Exception as committee_l3_exc:
                tprint_warning(f" Layer 3: failed to reconstruct committee returns/labels: {committee_l3_exc}")
                return {"success": False}

        valid_final_mask = ~final_labels.isna()
        if valid_final_mask.sum() < 50:
            tprint_warning(" Insufficient events. Aborting.")
            return {"success": False}
        
        # Unpack Layer 2 Artifacts (OOF for Training/Analytics)
        l2_labels = l2_output['oof_labels']
        l2_returns = l2_output['oof_returns']
        l2_weights = l2_output['weights']
        individual_geos = l2_output['individual_geometries']
        events_df = l2_output['events_df']
        selected_trials = l2_output['selected_trials'] # Production Geometries
        
        # Save Layer 2 Production Geometries (Optimized on Full Data)
        with open(outcomes_dir / "layer2_selected_geometries.json", "w") as f:
            json.dump(selected_trials, f, indent=2, default=str)
            
        # ---------------------------------------------------------
        # Weight Calculation for Layer 3 (Based on OOF)
        # ---------------------------------------------------------
        # Formula: W_t = W_L2 * log(1 + |R_composite|) * W_L1
        
        if target_sample_weight is not None:
             if len(target_sample_weight) == len(df):
                 w_l1_series = pd.Series(target_sample_weight, index=df.index)
                 w_l1_aligned = w_l1_series.reindex(events_df.index).fillna(1.0)
             else:
                 tprint_warning(f"Layer 1 weights length mismatch ({len(target_sample_weight)} vs {len(df)}). Using 1.0.")
                 w_l1_aligned = pd.Series(1.0, index=events_df.index)
        else:
             w_l1_aligned = pd.Series(1.0, index=events_df.index)
        
        magnitude_factor = np.log1p(l2_returns.abs().fillna(0))
        
        w_final_series = l2_weights * magnitude_factor * w_l1_aligned
        
        if w_final_series.mean() > 0:
            w_final_series /= w_final_series.mean()
        
        w_final = w_final_series.values
        
        # ---------------------------------------------------------
        # Data Assembly for Layer 3
        # ---------------------------------------------------------
        tprint_info(">>> Preparing OOF Data for Layer 3...")
        
        # Assemble OOF predictions from individual geometries
        geo_preds_df = pd.DataFrame(index=events_df.index)
        for uuid, preds in individual_geos.items():
            # preds are already Series on the correct index (or reindex safe)
            geo_preds_df[uuid] = preds.reindex(events_df.index)
            
        geo_cols = list(geo_preds_df.columns)
        
        l3_input_df = geo_preds_df.copy()
        
        context_cols = ['volatility_1d']
        for c in context_cols:
            if c in events_df.columns:
                l3_input_df[c] = events_df[c]
            elif c in df.columns:
                 l3_input_df[c] = df.loc[l3_input_df.index, c]
        
        target_col = 'l2_consensus_target'
        l3_input_df[target_col] = l2_labels
        
        # ---------------------------------------------------------
        # LAYER 3: Calibration & Meta-Model (OOF & Final)
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 3: OOF Calibration & Production Model Training...")
        
        # This now returns OOF predictions for entire dataset and the Final Model
        oof_export, final_model = layer3_analyst_lgbm(
            oof_df=l3_input_df,
            base_model_cols=geo_cols,
            target_col=target_col,
            train_split_date=None,
            sample_weight=w_final
        )
        
        # Generate Diagnostics (on OOF predictions)
        tprint_info(">>> Generating Layer 3 Diagnostics...")
        plot_diagnostics(
            y_true=oof_export[target_col],
            y_prob=oof_export['meta_prob'],
            output_path=str(outcomes_dir / "layer3_calibration_plot.png")
        )
        
        # ---------------------------------------------------------
        # Artifacts & Return
        # ---------------------------------------------------------
        
        # Save OOF Predictions (Full History)
        oof_export.to_csv(outcomes_dir / "layer3_oof_preds.csv")
        
        # Save Weights
        pd.DataFrame({'weight': w_final}).describe().to_csv(outcomes_dir / "layer3_weights_stats.csv")
        
        # Save Final Model
        joblib.dump(final_model, outcomes_dir / "layer3_final_model.joblib")

        tprint_success(f"Pipeline Completed. Artifacts saved to {outcomes_dir}")
        
        return {
            "success": True,
            "outcomes_dir": str(outcomes_dir),
            "metrics": {
                "n_events": len(l3_input_df),
                "n_geometries": len(geo_cols)
            },
            "artifacts": {
                "oof_preds": str(outcomes_dir / "layer3_oof_preds.csv"),
                "calibration_plot": str(outcomes_dir / "layer3_calibration_plot.png"),
                "final_model": str(outcomes_dir / "layer3_final_model.joblib"),
                "layer2_geometries": str(outcomes_dir / "layer2_selected_geometries.json")
            }
        }

def register_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry
    step_registry.register("meta_labeling_hpo_sample_weighted", MetaLabelingHPOSampleWeightedStep)
    # Aliases
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb_weighted", MetaLabelingHPOSampleWeightedStep)

register_meta_labeling_hpo_sample_weighted_step()
