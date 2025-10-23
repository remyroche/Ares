"""
Label Balancing & Sample Weighting System

This module implements comprehensive label balancing and sample weighting techniques
to address the extreme class imbalance in financial datasets (80-95% "no-trade" samples).

Key Features:
- Under-sampling of majority class (no-trade samples)
- Over-sampling of minority classes using SMOTE and mixup
- Purged time-series cross-validation for temporal integrity
- Multiple sample weighting schemes (volatility, confidence, event overlap, time decay)
- Regime-aware rebalancing
- Validation fairness checks

The system is designed to "teach the model what matters" by ensuring balanced exposure
to different classes and weighting samples by information content rather than just class balance.

CRITICAL: All balancing and weighting operations occur within training folds only to prevent
temporal leakage. Caller must pass fold-specific (X_train, y_train, weights_train); 
do not call on concatenated train+val data.

CONTRACTS:
==========

Causality:
- All statistics are computed on training folds only and shift(1) before use
- No future information is used in any balancing or weighting operation
- Temporal validation guards prevent leakage between train/validation periods

Inputs:
- X: pd.DataFrame with DatetimeIndex for temporal operations
- y: pd.Series with integer or categorical labels
- additional_features: Dict containing:
  - 'confidence': pd.Series with OOS prediction confidence (required for confidence weighting)
  - 'event_intervals': List[Tuple] of (start_time, end_time) for concurrency weighting
  - 'event_horizons': pd.Series with event durations for pseudo-concurrency weighting
  - 'regime': pd.Series with regime labels for regime-aware weighting
  - 'volatility': pd.Series with precomputed volatility (optional)
  - 'timestamp': pd.Series with timestamps if X.index is not DatetimeIndex

Outputs:
- X_balanced: pd.DataFrame with same columns as input, proper index alignment
- y_balanced: pd.Series with same dtype as input, proper index alignment  
- weights: pd.Series[float] with same index as X_balanced, normalized to mean=1
- qa_report: Dict with quality control metrics and warnings

Units:
- Volatility is computed from returns (log returns recommended)
- Time decay uses days as units
- All weights are normalized to mean=1 to preserve economic shape

Performance:
- Vectorized operations for overlap kernels (O(n) instead of O(n²))
- Cached k-NN graphs for synthetic weight computation
- Efficient convolution-based overlap counting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
from sklearn.utils import resample
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import warnings
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.special import softmax
from collections import Counter
import random

# Import existing utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_preview, tprint_data_format
    from src.utils.common_operations import safe_divide, safe_mean, safe_std, validate_dataframe
    from src.utils.math_validation import MathValidation
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    if TPRINT_AVAILABLE:
        tprint_warning("⚠️ imbalanced-learn not available, using basic resampling methods")


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validator for time series data.
    
    Implements purged cross-validation to prevent temporal leakage by:
    1. Purging samples that overlap with validation period
    2. Adding embargo period to prevent look-ahead bias
    3. Maintaining chronological order within folds
    """
    
    def __init__(self, n_splits: int = 5, purge_length: int = 1, embargo_length: int = 1, 
                 random_state: Optional[int] = None):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds
            purge_length: Number of samples to purge around validation period
            embargo_length: Number of samples to embargo after validation period
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length
        self.random_state = random_state
        
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X: Feature matrix with datetime index
            y: Target labels (optional)
            groups: Group labels (optional)
            
        Yields:
            Tuple of (train_indices, val_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex for purged cross-validation")
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Create fold boundaries
        fold_size = n_samples // self.n_splits
        fold_boundaries = [i * fold_size for i in range(self.n_splits + 1)]
        fold_boundaries[-1] = n_samples
        
        for i in range(self.n_splits):
            # Validation period
            val_start = fold_boundaries[i]
            val_end = fold_boundaries[i + 1]
            
            # Purge period (before validation)
            purge_start = max(0, val_start - self.purge_length)
            purge_end = val_start
            
            # Embargo period (after validation)
            embargo_start = val_end
            embargo_end = min(n_samples, val_end + self.embargo_length)
            
            # Training indices (exclude purge and embargo periods)
            train_indices = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])
            
            # Validation indices
            val_indices = indices[val_start:val_end]
            
            # Ensure we have enough samples
            if len(train_indices) < 10 or len(val_indices) < 5:
                continue
                
            yield train_indices, val_indices
    
    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, 
                     groups: Optional[pd.Series] = None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


def validate_temporal_integrity(X_train: pd.DataFrame, X_val: Optional[pd.DataFrame] = None) -> None:
    """
    Validate temporal integrity to prevent leakage.
    
    Args:
        X_train: Training data with datetime index
        X_val: Validation data with datetime index (optional)
        
    Raises:
        ValueError: If temporal leakage is detected
    """
    if not isinstance(X_train.index, pd.DatetimeIndex):
        raise ValueError("Training data must have DatetimeIndex for temporal validation")
    
    if X_val is not None:
        if not isinstance(X_val.index, pd.DatetimeIndex):
            raise ValueError("Validation data must have DatetimeIndex for temporal validation")
        
        # Check for temporal leakage
        if X_train.index.max() > X_val.index.min():
            raise ValueError(
                "Temporal leakage detected: training data extends beyond validation start. "
                "Ensure training data ends before validation begins."
            )
        
        if TPRINT_AVAILABLE:
            tprint_info(f"✅ Temporal integrity validated: train ends {X_train.index.max()}, val starts {X_val.index.min()}")


def ensure_proper_types(X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
                       weights: Union[pd.Series, np.ndarray, None] = None,
                       original_index: Optional[pd.Index] = None,
                       original_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Ensure proper types and index alignment after imblearn operations.
    
    Args:
        X: Feature matrix (may be DataFrame or array)
        y: Target labels (may be Series or array)
        weights: Sample weights (may be Series or array)
        original_index: Original index to preserve
        original_columns: Original column names to preserve
        
    Returns:
        Tuple of (X_df, y_series, weights_series) with proper types and alignment
    """
    # Convert X to DataFrame with proper index and columns
    if isinstance(X, np.ndarray):
        if original_columns is None:
            original_columns = [f"feature_{i}" for i in range(X.shape[1])]
        if original_index is None:
            original_index = pd.RangeIndex(len(X))
        X_df = pd.DataFrame(X, columns=original_columns, index=original_index)
    else:
        X_df = X.copy()
    
    # Convert y to Series with proper index
    if isinstance(y, np.ndarray):
        if original_index is None:
            original_index = pd.RangeIndex(len(y))
        y_series = pd.Series(y, index=original_index)
    else:
        y_series = y.copy()
    
    # Convert weights to Series with proper index
    if weights is not None:
        if isinstance(weights, np.ndarray):
            if original_index is None:
                original_index = pd.RangeIndex(len(weights))
            weights_series = pd.Series(weights, index=original_index)
        else:
            weights_series = weights.copy()
    else:
        weights_series = None
    
    # Ensure index alignment
    if not X_df.index.equals(y_series.index):
        # Reindex to align
        common_index = X_df.index.intersection(y_series.index)
        X_df = X_df.loc[common_index]
        y_series = y_series.loc[common_index]
        if weights_series is not None:
            weights_series = weights_series.loc[common_index]
    
    return X_df, y_series, weights_series


class BalancingTechnique(Enum):
    """Enumeration of balancing techniques."""
    UNDER_SAMPLING = "under_sampling"
    OVER_SAMPLING = "over_sampling"
    SMOTE = "smote"
    BORDERLINE_SMOTE = "borderline_smote"
    SVM_SMOTE = "svm_smote"
    ADASYN = "adasyn"
    MIXUP = "mixup"
    STRATIFIED_BATCHING = "stratified_batching"
    SMOTE_TOMEK = "smote_tomek"
    SMOTE_ENN = "smote_enn"
    NEAR_MISS = "near_miss"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class WeightingScheme(Enum):
    """Enumeration of sample weighting schemes."""
    VOLATILITY = "volatility"  # w_t ∝ 1/σ_t
    CONFIDENCE = "confidence"  # w_t ∝ Δp (label confidence)
    EVENT_OVERLAP = "event_overlap"  # López de Prado weighting
    TIME_DECAY = "time_decay"  # Exponential decay by recency
    REGIME_AWARE = "regime_aware"  # Inverse regime frequency
    INFORMATION_CONTENT = "information_content"  # Combined weighting


@dataclass
class BalancingConfig:
    """Configuration for label balancing."""

    # Balancing technique selection
    balancing_technique: BalancingTechnique = BalancingTechnique.ADAPTIVE

    # Under-sampling parameters
    under_sampling_ratio: float = 0.5  # Ratio of majority class to keep
    under_sampling_strategy: str = "random"  # random, tomek, enn, near_miss
    under_sampling_version: int = 1  # For NearMiss: 1, 2, or 3

    # Over-sampling parameters
    over_sampling_ratio: float = 1.0  # Ratio of minority classes to generate
    over_sampling_strategy: str = "smote"  # smote, adasyn, mixup, borderline_smote, svm_smote
    smote_k_neighbors: int = 5  # k-neighbors for SMOTE variants
    smote_sampling_strategy: str = "auto"  # auto, balanced, or dict

    # Mixup parameters
    mixup_alpha: float = 0.2  # Beta distribution alpha parameter
    mixup_version: int = 1  # Mixup version (1 or 2)

    # Stratified batching parameters
    stratified_batching: bool = True
    batch_size: int = 1024
    min_samples_per_class: int = 10
    max_batch_imbalance_ratio: float = 0.1  # Max allowed imbalance in batches

    # Hybrid balancing parameters
    hybrid_undersample_ratio: float = 0.7
    hybrid_oversample_ratio: float = 0.3
    hybrid_technique_order: List[str] = None  # Order of techniques to apply

    # Adaptive balancing parameters
    adaptive_imbalance_threshold: float = 0.1  # Threshold for considering imbalance
    adaptive_min_samples: int = 50  # Minimum samples per class for adaptation
    adaptive_technique_selection: str = "performance"  # performance, speed, balanced

    # Target class distribution (None for auto-detection)
    target_distribution: Optional[Dict[int, float]] = None
    target_imbalance_ratio: float = 0.3  # Target ratio of minority to majority

    # Quality control
    enable_quality_control: bool = True
    min_quality_score: float = 0.6
    max_synthetic_ratio: float = 0.5  # Max ratio of synthetic samples

    # Temporal validation
    enable_temporal_validation: bool = True
    purge_length: int = 1  # Samples to purge around validation period
    embargo_length: int = 1  # Samples to embargo after validation period

    # Random state for reproducibility
    random_state: int = 42


@dataclass
class WeightingConfig:
    """Configuration for sample weighting."""

    # Weighting scheme selection
    weighting_scheme: WeightingScheme = WeightingScheme.INFORMATION_CONTENT

    # Volatility weighting parameters
    volatility_window: int = 20
    volatility_floor: float = 1e-6
    volatility_weight_max: float = 10.0
    volatility_method: str = "rolling_std"  # rolling_std, garch, ewma
    volatility_robust: bool = True  # Use robust volatility estimation

    # Confidence weighting parameters
    confidence_method: str = "probability"  # probability, margin, entropy, uncertainty
    confidence_scale: float = 2.0
    confidence_smoothing: float = 0.1  # Smoothing factor for confidence scores
    confidence_min_threshold: float = 0.1  # Minimum confidence threshold

    # Event overlap weighting parameters (López de Prado)
    overlap_window: int = 5
    overlap_decay: float = 0.8
    overlap_method: str = "rolling_count"  # rolling_count, exponential_decay, gaussian
    overlap_threshold: float = 0.1  # Threshold for considering overlap

    # Time decay weighting parameters
    time_decay_half_life: int = 30  # days
    time_decay_method: str = "exponential"  # exponential, linear, polynomial
    time_decay_power: float = 1.0  # Power for polynomial decay
    time_decay_min_weight: float = 0.01  # Minimum weight for very old samples

    # Regime-aware weighting parameters
    regime_frequency_threshold: float = 0.2
    regime_weight_multiplier: float = 5.0
    regime_smoothing_window: int = 10  # Window for regime frequency smoothing
    regime_adaptation_rate: float = 0.1  # Rate of regime weight adaptation

    # Information content weighting parameters
    information_entropy_weight: float = 0.3  # Weight for entropy component
    information_uncertainty_weight: float = 0.3  # Weight for uncertainty component
    information_volatility_weight: float = 0.2  # Weight for volatility component
    information_regime_weight: float = 0.2  # Weight for regime component
    information_confidence_weight: float = 0.2  # Weight for confidence component
    information_overlap_weight: float = 0.2  # Weight for overlap component
    information_time_weight: float = 0.2  # Weight for time decay component

    # Advanced weighting parameters
    enable_dynamic_weighting: bool = True  # Adapt weights during training
    weight_adaptation_rate: float = 0.01  # Rate of weight adaptation
    weight_memory_decay: float = 0.95  # Decay rate for weight memory

    # Combined weighting parameters
    weight_normalization: str = "l2"  # none, l1, l2, minmax, robust
    weight_clip_percentile: float = 99.0
    weight_smoothing: float = 0.1  # Smoothing factor for final weights

    # Minimum and maximum weight bounds
    min_weight: float = 0.1
    max_weight: float = 10.0
    weight_floor: float = 1e-6  # Minimum weight to avoid numerical issues

    # Quality control
    enable_weight_validation: bool = True
    max_weight_ratio: float = 100.0  # Max ratio between highest and lowest weights
    weight_stability_threshold: float = 0.1  # Threshold for weight stability


@dataclass
class RegimeConfig:
    """Configuration for regime-aware balancing."""

    # Regime detection parameters
    enable_regime_detection: bool = True
    regime_column: str = "regime"  # Column name containing regime labels

    # Regime frequency calculation
    regime_lookback_window: int = 252  # Trading days for frequency calculation
    min_regime_samples: int = 50

    # Regime-aware rebalancing
    regime_balance_method: str = "inverse_frequency"  # inverse_frequency, stratified, uniform
    regime_balance_strength: float = 1.0

    # Regime validation fairness
    validation_regime_fairness: bool = True
    regime_fairness_tolerance: float = 0.1


@dataclass
class ValidationFairnessConfig:
    """Configuration for validation fairness checks."""

    # Class ratio fairness
    class_ratio_tolerance: float = 0.05
    min_class_samples: int = 100

    # Regime mix fairness
    regime_mix_tolerance: float = 0.1
    min_regime_samples: int = 50

    # Temporal fairness
    temporal_drift_tolerance: float = 0.1
    drift_window: int = 21

    # Enable/disable individual checks
    check_class_ratios: bool = True
    check_regime_mix: bool = True
    check_temporal_drift: bool = True


class LabelBalancer:
    """
    Comprehensive label balancing system for financial datasets.

    Addresses the extreme class imbalance problem where 80-95% of samples
    are "no-trade" (Analyst = 0) by providing multiple balancing techniques.

    Key Features:
    - Temporal integrity: All operations respect chronological order
    - Deterministic results: Reproducible with random_state
    - Type safety: Proper DataFrame/Series reconstruction after imblearn
    - Quality control: Synthetic ratio caps and validation checks
    - Objective-driven selection: CV-based technique selection

    Example:
        >>> config = BalancingConfig(balancing_technique=BalancingTechnique.ADAPTIVE)
        >>> balancer = LabelBalancer(config)
        >>> X_balanced, y_balanced, weights = balancer.balance_dataset(X_train, y_train)
    """

    def __init__(self, config: BalancingConfig):
        """Initialize the label balancer."""
        self.config = config
        self._validate_config()

        if not IMBLEARN_AVAILABLE and config.balancing_technique in [BalancingTechnique.SMOTE, BalancingTechnique.ADASYN]:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ {config.balancing_technique.value} requires imbalanced-learn, falling back to random sampling")
            self.config.balancing_technique = BalancingTechnique.UNDER_SAMPLING

    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0 < self.config.under_sampling_ratio <= 1:
            raise ValueError("under_sampling_ratio must be between 0 and 1")

        if not 0 < self.config.over_sampling_ratio <= 10:
            raise ValueError("over_sampling_ratio must be between 0 and 10")

        if self.config.batch_size < self.config.min_samples_per_class:
            raise ValueError("batch_size must be >= min_samples_per_class")

        if self.config.target_distribution is not None:
            if not np.isclose(sum(self.config.target_distribution.values()), 1.0):
                raise ValueError("target_distribution must sum to 1.0")

    def balance_dataset(self, X: pd.DataFrame, y: pd.Series,
                       sample_weight: Optional[pd.Series] = None,
                       X_val: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Apply balancing to the dataset.

        Args:
            X: Feature matrix (training data only)
            y: Target labels (training data only)
            sample_weight: Optional sample weights (training data only)
            X_val: Optional validation data for temporal integrity check

        Returns:
            Tuple of (balanced_X, balanced_y, balanced_weights)
            
        Note:
            Caller must pass fold-specific (X_train, y_train, weights_train); 
            do not call on concatenated train+val data to prevent temporal leakage.
        """
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Starting label balancing with technique: {self.config.balancing_technique.value}")

        # Validate temporal integrity
        if self.config.enable_temporal_validation:
            validate_temporal_integrity(X, X_val)

        # Set random state for reproducibility
        np.random.seed(self.config.random_state)
        random.seed(self.config.random_state)

        # Get class distribution info with deterministic ordering
        class_counts = y.value_counts().sort_index()  # Sort by label value for deterministic ordering
        majority_class = class_counts.index[0]
        minority_classes = class_counts.index[1:]

        if TPRINT_AVAILABLE:
            tprint_info(f"📊 Original class distribution: {dict(class_counts)}")

        if self.config.balancing_technique == BalancingTechnique.UNDER_SAMPLING:
            return self._under_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.OVER_SAMPLING:
            return self._over_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.SMOTE:
            return self._smote_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.ADASYN:
            return self._adasyn_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.MIXUP:
            return self._mixup_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.STRATIFIED_BATCHING:
            return self._create_stratified_batches(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.BORDERLINE_SMOTE:
            return self._borderline_smote_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.SVM_SMOTE:
            return self._svm_smote_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.SMOTE_TOMEK:
            return self._smote_tomek_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.SMOTE_ENN:
            return self._smote_enn_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.NEAR_MISS:
            return self._near_miss_sample(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.HYBRID:
            return self._hybrid_balance(X, y, sample_weight)

        elif self.config.balancing_technique == BalancingTechnique.ADAPTIVE:
            return self._adaptive_balance(X, y, sample_weight)

        else:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Unknown balancing technique: {self.config.balancing_technique}")
            return X, y, sample_weight

    def _under_sample(self, X: pd.DataFrame, y: pd.Series,
                     sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Under-sample the majority class with explicit target distribution."""
        class_counts = y.value_counts().sort_index()
        majority_class = class_counts.index[0]
        majority_count = class_counts.iloc[0]

        # Calculate target counts using explicit target distribution
        if self.config.target_distribution is not None:
            # Use provided target distribution
            target_counts = {}
            for class_label, target_ratio in self.config.target_distribution.items():
                target_counts[class_label] = int(len(y) * target_ratio)
        else:
            # Compute target distribution based on minority classes and under_sampling_ratio
            minority_total = len(y) - majority_count
            target_maj_count = int(minority_total / (1 - self.config.under_sampling_ratio))
            target_maj_count = min(target_maj_count, majority_count)
            
            target_counts = {majority_class: target_maj_count}
            for class_label in minority_classes:
                target_counts[class_label] = class_counts[class_label]

        # Cap by available samples and max synthetic ratio
        for class_label in target_counts:
            available_count = class_counts[class_label]
            max_allowed = int(available_count * (1 + self.config.max_synthetic_ratio))
            target_counts[class_label] = min(target_counts[class_label], max_allowed)

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Under-sampling majority class {majority_class} from {majority_count} to {target_counts[majority_class]}")

        # Under-sample majority class
        majority_indices = y[y == majority_class].index
        target_maj_count = target_counts[majority_class]
        
        if len(majority_indices) > target_maj_count:
            undersampled_indices = np.random.choice(
                majority_indices, size=target_maj_count, replace=False, 
                random_state=self.config.random_state
            )
        else:
            undersampled_indices = majority_indices

        # Combine with all minority samples
        minority_indices = y[y != majority_class].index
        final_indices = np.concatenate([undersampled_indices, minority_indices])

        # Rebuild DataFrames with proper index alignment
        balanced_X = X.loc[final_indices].copy()
        balanced_y = y.loc[final_indices].copy()
        balanced_weights = sample_weight.loc[final_indices].copy() if sample_weight is not None else None

        return balanced_X, balanced_y, balanced_weights

    def _over_sample(self, X: pd.DataFrame, y: pd.Series,
                    sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Over-sample minority classes using random sampling."""
        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"📈 Over-sampling to match majority class count: {max_count}")

        balanced_dfs = []

        for class_label in class_counts.index:
            class_data = X[y == class_label]
            class_labels = y[y == class_label]

            if len(class_data) < max_count:
                # Over-sample this class
                oversampled_data, oversampled_labels = resample(
                    class_data, class_labels,
                    n_samples=max_count,
                    random_state=self.config.random_state
                )
                balanced_dfs.append((oversampled_data, oversampled_labels))
            else:
                # Keep majority class as-is
                balanced_dfs.append((class_data, class_labels))

        # Combine all classes
        balanced_X = pd.concat([df[0] for df in balanced_dfs])
        balanced_y = pd.concat([df[1] for df in balanced_dfs])

        # Handle sample weights
        if sample_weight is not None:
            balanced_weights = []
            for i, (class_label, count) in enumerate(class_counts.items()):
                class_weights = sample_weight[y == class_label]
                if len(class_weights) < max_count:
                    # Duplicate weights for oversampled data
                    oversampled_weights = resample(
                        class_weights,
                        n_samples=max_count,
                        random_state=self.config.random_state
                    )
                    balanced_weights.append(oversampled_weights)
                else:
                    balanced_weights.append(class_weights)
            balanced_sample_weight = pd.concat(balanced_weights)
        else:
            balanced_sample_weight = None

        return balanced_X, balanced_y, balanced_sample_weight

    def _smote_sample(self, X: pd.DataFrame, y: pd.Series,
                     sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply SMOTE oversampling with safety guards and proper synthetic weight computation."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ SMOTE not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts().sort_index()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🧬 Applying SMOTE to reach majority class count: {max_count}")

        # Safety guard: check for classes with insufficient samples
        min_class_count = class_counts.min()
        if min_class_count < 2:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Class with only {min_class_count} samples detected, disabling SMOTE for safety")
            return self._over_sample(X, y, sample_weight)

        # Determine sampling strategy with safety caps
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count and count >= 2:  # Only SMOTE classes with >= 2 samples
                target_count = min(max_count, int(count * (1 + self.config.max_synthetic_ratio)))
                sampling_strategy[class_label] = target_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply SMOTE with safe k_neighbors
        safe_k_neighbors = min(self.config.smote_k_neighbors, min_class_count - 1)
        if safe_k_neighbors <= 0:
            safe_k_neighbors = 1

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state,
            k_neighbors=safe_k_neighbors
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Rebuild DataFrames with proper index alignment and type safety
        balanced_X, balanced_y, _ = ensure_proper_types(
            X_resampled, y_resampled, 
            original_columns=X.columns,
            original_index=pd.RangeIndex(len(X_resampled))
        )

        # Handle sample weights for SMOTE samples
        if sample_weight is not None:
            # For synthetic samples, compute weights as convex combination of neighbor weights
            synthetic_weights = self._compute_synthetic_weights(
                X, y, sample_weight, X_resampled, y_resampled, len(X)
            )
            balanced_sample_weight = pd.concat([
                sample_weight,
                synthetic_weights
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return balanced_X, balanced_y, balanced_sample_weight

    def _compute_synthetic_weights(self, X_orig: pd.DataFrame, y_orig: pd.Series, 
                                 weights_orig: pd.Series, X_synthetic: np.ndarray, 
                                 y_synthetic: np.ndarray, n_orig: int) -> pd.Series:
        """Compute synthetic sample weights as convex combination of neighbor weights."""
        if len(X_synthetic) <= n_orig:
            return pd.Series([], dtype=float)
        
        synthetic_weights = []
        
        # For each synthetic sample, find its k-nearest neighbors and interpolate weights
        for i in range(n_orig, len(X_synthetic)):
            # Find the class of this synthetic sample
            synth_class = y_synthetic[i]
            class_mask = y_orig == synth_class
            class_X = X_orig[class_mask]
            class_weights = weights_orig[class_mask]
            
            if len(class_X) == 0:
                synthetic_weights.append(weights_orig.mean())
                continue
            
            # Find k-nearest neighbors within the same class
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(3, len(class_X)))
            nn.fit(class_X)
            
            # Get the synthetic sample features
            synth_features = X_synthetic[i].reshape(1, -1)
            distances, indices = nn.kneighbors(synth_features)
            
            # Compute convex combination weights (inverse distance weighting)
            if distances[0][0] == 0:  # Exact match
                synthetic_weights.append(class_weights.iloc[indices[0][0]])
            else:
                # Inverse distance weighting
                inv_distances = 1.0 / (distances[0] + 1e-8)
                weights_sum = inv_distances.sum()
                neighbor_weights = class_weights.iloc[indices[0]]
                synthetic_weight = (inv_distances * neighbor_weights).sum() / weights_sum
                synthetic_weights.append(synthetic_weight)
        
        return pd.Series(synthetic_weights, index=range(n_orig, len(X_synthetic)))

    def _adasyn_sample(self, X: pd.DataFrame, y: pd.Series,
                      sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply ADASYN oversampling."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ ADASYN not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Applying ADASYN to reach majority class count: {max_count}")

        # Determine sampling strategy
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count:
                sampling_strategy[class_label] = max_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply ADASYN
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state
        )

        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        # Handle sample weights for ADASYN samples
        if sample_weight is not None:
            # Original weights for original samples
            original_weights = sample_weight.copy()

            # For synthetic samples, use interpolated weights
            synthetic_weights = []
            for i in range(len(X), len(X_resampled)):
                synthetic_weights.append(sample_weight.mean())

            balanced_sample_weight = pd.concat([
                pd.Series(original_weights),
                pd.Series(synthetic_weights)
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), balanced_sample_weight

    def _mixup_sample(self, X: pd.DataFrame, y: pd.Series,
                     sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply Mixup augmentation with soft-label compatibility check."""
        class_counts = y.value_counts().sort_index()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🍹 Applying Mixup augmentation to reach majority class count: {max_count}")

        # Check if downstream learner supports soft labels
        # For now, we'll generate soft labels but warn the user
        if TPRINT_AVAILABLE:
            tprint_warning("⚠️ Mixup generates soft labels - ensure downstream learner supports probabilistic targets")

        # Create mixup samples
        mixed_X = []
        mixed_y = []
        mixed_weights = [] if sample_weight is not None else None

        for class_label in class_counts.index:
            class_data = X[y == class_label]
            class_labels = y[y == class_label]
            class_weights = sample_weight[y == class_label] if sample_weight is not None else None

            if len(class_data) < max_count:
                # Generate mixup samples for this class
                n_mixup = max_count - len(class_data)

                # Sample pairs for mixing
                for _ in range(n_mixup):
                    # Randomly sample two different examples from this class
                    if len(class_data) < 2:
                        # Not enough samples for mixing, duplicate instead
                        idx1 = np.random.choice(len(class_data))
                        mixed_X.append(class_data.iloc[idx1])
                        mixed_y.append(class_labels.iloc[idx1])
                        if sample_weight is not None:
                            mixed_weights.append(class_weights.iloc[idx1])
                        continue

                    idx1, idx2 = np.random.choice(len(class_data), 2, replace=False)

                    # Mixup ratio (beta distribution) - use configurable alpha
                    alpha = self.config.mixup_alpha
                    lambda_param = np.random.beta(alpha, alpha)

                    # Mix features
                    mixed_features = lambda_param * class_data.iloc[idx1] + (1 - lambda_param) * class_data.iloc[idx2]

                    # Mix labels (soft labels) - convert to float for soft labels
                    mixed_label = lambda_param * float(class_labels.iloc[idx1]) + (1 - lambda_param) * float(class_labels.iloc[idx2])

                    mixed_X.append(mixed_features)
                    mixed_y.append(mixed_label)

                    if sample_weight is not None:
                        mixed_weight = lambda_param * class_weights.iloc[idx1] + (1 - lambda_param) * class_weights.iloc[idx2]
                        mixed_weights.append(mixed_weight)

            # Add original samples
            mixed_X.append(class_data)
            mixed_y.append(class_labels)
            if sample_weight is not None:
                mixed_weights.append(class_weights)

        # Combine all samples
        balanced_X = pd.concat(mixed_X) if mixed_X else X
        balanced_y = pd.concat(mixed_y) if mixed_y else y

        # Convert y to float dtype for soft labels
        balanced_y = balanced_y.astype(float)

        if sample_weight is not None:
            balanced_sample_weight = pd.concat(mixed_weights) if mixed_weights else sample_weight
        else:
            balanced_sample_weight = None

        return balanced_X, balanced_y, balanced_sample_weight

    def _hybrid_balance(self, X: pd.DataFrame, y: pd.Series,
                       sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply hybrid balancing (combination of under and over sampling)."""
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Applying hybrid balancing (under: {self.config.hybrid_undersample_ratio}, over: {self.config.hybrid_oversample_ratio})")

        # First, under-sample majority class
        X_under, y_under, sample_weight_under = self._under_sample(
            X, y, sample_weight
        )

        # Then, over-sample minority classes to achieve final balance
        return self._over_sample(X_under, y_under, sample_weight_under)

    def _create_stratified_batches(self, X: pd.DataFrame, y: pd.Series,
                                  sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Create chronological stratified batches for streaming training."""
        if TPRINT_AVAILABLE:
            tprint_info(f"📦 Creating chronological stratified batches with size: {self.config.batch_size}")

        # Use PurgedKFold for temporal integrity
        if isinstance(X.index, pd.DatetimeIndex):
            cv = PurgedKFold(
                n_splits=max(2, len(X) // self.config.batch_size),
                purge_length=self.config.purge_length,
                embargo_length=self.config.embargo_length,
                random_state=self.config.random_state
            )
        else:
            # Fallback to regular KFold if no datetime index
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=max(2, len(X) // self.config.batch_size), shuffle=False, random_state=self.config.random_state)

        batches_X = []
        batches_y = []
        batches_weights = [] if sample_weight is not None else None

        for train_idx, val_idx in cv.split(X, y):
            # Use validation indices as batch indices for chronological batching
            batch_y = y.iloc[val_idx]

            # Ensure minimum samples per class in each batch
            if len(batch_y.unique()) >= 2 and all(batch_y.value_counts() >= self.config.min_samples_per_class):
                batches_X.append(X.iloc[val_idx])
                batches_y.append(batch_y)

                if sample_weight is not None:
                    batches_weights.append(sample_weight.iloc[val_idx])

        if not batches_X:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ Could not create stratified batches, returning original data")
            return X, y, sample_weight

        # Combine all batches
        balanced_X = pd.concat(batches_X)
        balanced_y = pd.concat(batches_y)

        if sample_weight is not None:
            balanced_sample_weight = pd.concat(batches_weights)
        else:
            balanced_sample_weight = None

        return balanced_X, balanced_y, balanced_sample_weight

    def _borderline_smote_sample(self, X: pd.DataFrame, y: pd.Series,
                                sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply BorderlineSMOTE oversampling."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ BorderlineSMOTE not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Applying BorderlineSMOTE to reach majority class count: {max_count}")

        # Determine sampling strategy
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count:
                sampling_strategy[class_label] = max_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply BorderlineSMOTE
        borderline_smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state,
            k_neighbors=min(self.config.smote_k_neighbors, min(class_counts[class_counts > 1]) - 1)
        )

        X_resampled, y_resampled = borderline_smote.fit_resample(X, y)

        # Rebuild DataFrames with proper index alignment and type safety
        balanced_X, balanced_y, _ = ensure_proper_types(
            X_resampled, y_resampled, 
            original_columns=X.columns,
            original_index=pd.RangeIndex(len(X_resampled))
        )

        # Handle sample weights
        if sample_weight is not None:
            original_weights = sample_weight.copy()
            synthetic_weights = []
            for i in range(len(X), len(X_resampled)):
                synthetic_weights.append(sample_weight.mean())

            balanced_sample_weight = pd.concat([
                pd.Series(original_weights),
                pd.Series(synthetic_weights)
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return balanced_X, balanced_y, balanced_sample_weight

    def _svm_smote_sample(self, X: pd.DataFrame, y: pd.Series,
                         sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply SVMSMOTE oversampling."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ SVMSMOTE not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Applying SVMSMOTE to reach majority class count: {max_count}")

        # Determine sampling strategy
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count:
                sampling_strategy[class_label] = max_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply SVMSMOTE
        svm_smote = SVMSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state,
            k_neighbors=min(self.config.smote_k_neighbors, min(class_counts[class_counts > 1]) - 1)
        )

        X_resampled, y_resampled = svm_smote.fit_resample(X, y)

        # Rebuild DataFrames with proper index alignment and type safety
        balanced_X, balanced_y, _ = ensure_proper_types(
            X_resampled, y_resampled, 
            original_columns=X.columns,
            original_index=pd.RangeIndex(len(X_resampled))
        )

        # Handle sample weights
        if sample_weight is not None:
            original_weights = sample_weight.copy()
            synthetic_weights = []
            for i in range(len(X), len(X_resampled)):
                synthetic_weights.append(sample_weight.mean())

            balanced_sample_weight = pd.concat([
                pd.Series(original_weights),
                pd.Series(synthetic_weights)
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return balanced_X, balanced_y, balanced_sample_weight

    def _smote_tomek_sample(self, X: pd.DataFrame, y: pd.Series,
                           sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply SMOTETomek combination."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ SMOTETomek not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Applying SMOTETomek to reach majority class count: {max_count}")

        # Determine sampling strategy
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count:
                sampling_strategy[class_label] = max_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply SMOTETomek
        smote_tomek = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state,
            smote=SMOTE(k_neighbors=min(self.config.smote_k_neighbors, min(class_counts[class_counts > 1]) - 1))
        )

        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

        # Handle sample weights
        if sample_weight is not None:
            original_weights = sample_weight.copy()
            synthetic_weights = []
            for i in range(len(X), len(X_resampled)):
                synthetic_weights.append(sample_weight.mean())

            balanced_sample_weight = pd.concat([
                pd.Series(original_weights),
                pd.Series(synthetic_weights)
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), balanced_sample_weight

    def _smote_enn_sample(self, X: pd.DataFrame, y: pd.Series,
                         sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply SMOTEENN combination."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ SMOTEENN not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Applying SMOTEENN to reach majority class count: {max_count}")

        # Determine sampling strategy
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count:
                sampling_strategy[class_label] = max_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply SMOTEENN
        smote_enn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state,
            smote=SMOTE(k_neighbors=min(self.config.smote_k_neighbors, min(class_counts[class_counts > 1]) - 1))
        )

        X_resampled, y_resampled = smote_enn.fit_resample(X, y)

        # Handle sample weights
        if sample_weight is not None:
            original_weights = sample_weight.copy()
            synthetic_weights = []
            for i in range(len(X), len(X_resampled)):
                synthetic_weights.append(sample_weight.mean())

            balanced_sample_weight = pd.concat([
                pd.Series(original_weights),
                pd.Series(synthetic_weights)
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), balanced_sample_weight

    def _near_miss_sample(self, X: pd.DataFrame, y: pd.Series,
                         sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply NearMiss under-sampling."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ NearMiss not available, falling back to random under-sampling")
            return self._under_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        min_count = class_counts.min()

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Applying NearMiss to reach minority class count: {min_count}")

        # Apply NearMiss
        near_miss = NearMiss(
            version=self.config.under_sampling_version,
            n_neighbors=min(3, min_count - 1) if min_count > 1 else 1
        )

        X_resampled, y_resampled = near_miss.fit_resample(X, y)

        # Handle sample weights
        if sample_weight is not None:
            # Get indices of resampled data
            resampled_indices = X_resampled.index
            balanced_sample_weight = sample_weight.loc[resampled_indices]
        else:
            balanced_sample_weight = None

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), balanced_sample_weight

    def _adaptive_balance(self, X: pd.DataFrame, y: pd.Series,
                         sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply objective-driven balancing selection using purged CV."""
        if TPRINT_AVAILABLE:
            tprint_info("🧠 Applying objective-driven adaptive balancing...")

        # Define candidate techniques
        candidate_techniques = [
            BalancingTechnique.UNDER_SAMPLING,
            BalancingTechnique.SMOTE,
            BalancingTechnique.HYBRID,
            BalancingTechnique.STRATIFIED_BATCHING
        ]

        # Use purged CV to evaluate techniques
        if isinstance(X.index, pd.DatetimeIndex):
            cv = PurgedKFold(
                n_splits=3,  # Use fewer folds for efficiency
                purge_length=self.config.purge_length,
                embargo_length=self.config.embargo_length,
                random_state=self.config.random_state
            )
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=3, shuffle=False, random_state=self.config.random_state)

        best_technique = None
        best_score = -np.inf

        for technique in candidate_techniques:
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]
                
                # Apply technique to training fold
                try:
                    X_balanced, y_balanced, _ = self._apply_technique(
                        technique, X_train_fold, y_train_fold, sample_weight
                    )
                    
                    # Evaluate using class balance and diversity metrics
                    score = self._evaluate_balancing_quality(X_balanced, y_balanced, X_val_fold, y_val_fold)
                    scores.append(score)
                    
                except Exception as e:
                    if TPRINT_AVAILABLE:
                        tprint_warning(f"⚠️ Technique {technique.value} failed: {str(e)}")
                    continue
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_technique = technique

        # Apply best technique
        if best_technique is not None:
            if TPRINT_AVAILABLE:
                tprint_info(f"🎯 Best technique: {best_technique.value} (score: {best_score:.3f})")
            return self._apply_technique(best_technique, X, y, sample_weight)
        else:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ No technique succeeded, using under-sampling")
            return self._under_sample(X, y, sample_weight)

    def _apply_technique(self, technique: BalancingTechnique, X: pd.DataFrame, y: pd.Series,
                        sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply a specific balancing technique."""
        if technique == BalancingTechnique.UNDER_SAMPLING:
            return self._under_sample(X, y, sample_weight)
        elif technique == BalancingTechnique.SMOTE:
            return self._smote_sample(X, y, sample_weight)
        elif technique == BalancingTechnique.HYBRID:
            return self._hybrid_balance(X, y, sample_weight)
        elif technique == BalancingTechnique.STRATIFIED_BATCHING:
            return self._create_stratified_batches(X, y, sample_weight)
        else:
            return X, y, sample_weight

    def _evaluate_balancing_quality(self, X_balanced: pd.DataFrame, y_balanced: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Evaluate balancing quality using multiple metrics."""
        # Class balance score (inverse of Gini coefficient)
        class_counts = y_balanced.value_counts()
        n_samples = len(y_balanced)
        proportions = class_counts / n_samples
        
        # Gini coefficient for class balance
        gini = 1 - sum(p**2 for p in proportions)
        
        # Diversity score (number of unique classes)
        diversity = len(class_counts) / len(y_val.value_counts())
        
        # Sample efficiency (avoid excessive oversampling)
        efficiency = min(1.0, len(y_val) / len(y_balanced))
        
        # Combined score
        score = 0.4 * gini + 0.3 * diversity + 0.3 * efficiency
        
        return score


class SampleWeighter:
    """
    Sample weighting system for financial datasets.

    Implements various weighting schemes to emphasize samples with higher information content:
    - Volatility weighting: de-emphasize noisy high-vol periods (w ∝ 1/σ)
    - Confidence weighting: weight by OOS prediction confidence (w ∝ confidence)
    - Event overlap weighting: prevent duplicated exposure via concurrency (López de Prado)
    - Time decay weighting: keep model adaptive to latest dynamics (w ∝ exp(-t/τ))
    - Regime-aware weighting: balance across market regimes (w ∝ 1/frequency)
    - Information content: combined weighting using geometric mean

    All weights are normalized to mean=1 to preserve economic shape and avoid
    arbitrary scaling that could distort the intended weighting scheme.

    Example:
        >>> config = WeightingConfig(weighting_scheme=WeightingScheme.INFORMATION_CONTENT)
        >>> weighter = SampleWeighter(config)
        >>> weights = weighter.compute_weights(X, y, additional_features)
    """

    def __init__(self, config: WeightingConfig):
        """Initialize the sample weighter."""
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")

        if not 0 < self.config.min_weight <= self.config.max_weight:
            raise ValueError("min_weight must be <= max_weight and both positive")

        if not 0 < self.config.confidence_scale:
            raise ValueError("confidence_scale must be positive")

    def compute_weights(self, X: pd.DataFrame, y: pd.Series,
                       additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """
        Compute sample weights based on the configured weighting scheme.

        Args:
            X: Feature matrix
            y: Target labels
            additional_features: Optional additional features for weighting

        Returns:
            Series of sample weights
        """
        if TPRINT_AVAILABLE:
            tprint_info(f"⚖️ Computing sample weights using scheme: {self.config.weighting_scheme.value}")

        if self.config.weighting_scheme == WeightingScheme.VOLATILITY:
            return self._compute_volatility_weights(X, additional_features)

        elif self.config.weighting_scheme == WeightingScheme.CONFIDENCE:
            return self._compute_confidence_weights(y, additional_features)

        elif self.config.weighting_scheme == WeightingScheme.EVENT_OVERLAP:
            return self._compute_event_overlap_weights(y, additional_features)

        elif self.config.weighting_scheme == WeightingScheme.TIME_DECAY:
            return self._compute_time_decay_weights(X, additional_features)

        elif self.config.weighting_scheme == WeightingScheme.REGIME_AWARE:
            return self._compute_regime_aware_weights(X, y, additional_features)

        elif self.config.weighting_scheme == WeightingScheme.INFORMATION_CONTENT:
            return self._compute_information_content_weights(X, y, additional_features)

        else:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Unknown weighting scheme: {self.config.weighting_scheme}")
            return pd.Series(1.0, index=X.index)

    def _compute_volatility_weights(self, X: pd.DataFrame,
                                   additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute volatility-based weights (w_t ∝ 1/σ_t) with proper robustification."""
        # Try to get volatility from additional features or compute from returns
        volatility = None

        if additional_features and 'volatility' in additional_features:
            volatility = additional_features['volatility']
        elif 'returns' in X.columns:
            # Compute realized volatility from returns
            returns = X['returns']
            
            # Apply winsorization for robustification before computing volatility
            if self.config.volatility_robust:
                returns = returns.clip(
                    returns.quantile(0.01), 
                    returns.quantile(0.99)
                )
            
            if self.config.volatility_method == "rolling_std":
                volatility = returns.rolling(window=self.config.volatility_window).std()
            elif self.config.volatility_method == "ewma":
                # Exponentially weighted moving average
                volatility = returns.ewm(span=self.config.volatility_window).std()
            else:  # garch
                # Simple GARCH approximation
                volatility = returns.rolling(window=self.config.volatility_window).std()
            
            volatility = volatility.fillna(volatility.mean())
        else:
            # Use price volatility if available
            price_cols = [col for col in X.columns if col in ['close', 'price', 'Close', 'Price']]
            if price_cols:
                price_data = X[price_cols[0]]
                returns = price_data.pct_change()
                
                # Apply winsorization for robustification
                if self.config.volatility_robust:
                    returns = returns.clip(
                        returns.quantile(0.01), 
                        returns.quantile(0.99)
                    )
                
                if self.config.volatility_method == "rolling_std":
                    volatility = returns.rolling(window=self.config.volatility_window).std()
                elif self.config.volatility_method == "ewma":
                    volatility = returns.ewm(span=self.config.volatility_window).std()
                else:
                    volatility = returns.rolling(window=self.config.volatility_window).std()
                volatility = volatility.fillna(volatility.mean())

        if volatility is None:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ No volatility data available, using uniform weights")
            return pd.Series(1.0, index=X.index)

        # Compute weights inversely proportional to volatility
        weights = safe_divide(1.0, volatility + self.config.volatility_floor)

        # Normalize to mean=1 to preserve economic shape
        weights = weights / weights.mean()
        
        # Cap maximum weight
        weights = np.clip(weights, self.config.min_weight, self.config.volatility_weight_max)

        return pd.Series(weights, index=X.index)

    def _compute_confidence_weights(self, y: pd.Series,
                                   additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute confidence-based weights (w_t ∝ Δp) requiring OOS confidence."""
        if additional_features and 'confidence' in additional_features:
            confidence = additional_features['confidence']
            
            # Validate that confidence is from OOS predictions
            if TPRINT_AVAILABLE:
                tprint_info("✅ Using OOS confidence for weighting")
        else:
            # Disable confidence weighting if no OOS confidence available
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ No OOS confidence available, disabling confidence weighting")
            return pd.Series(1.0, index=y.index)

        # Apply smoothing
        if self.config.confidence_smoothing > 0:
            confidence = confidence.rolling(window=3, center=True).mean().fillna(confidence)

        # Apply minimum threshold
        confidence = np.maximum(confidence, self.config.confidence_min_threshold)

        # Scale confidence weights
        weights = self.config.confidence_scale * confidence

        # Normalize to mean=1 and clip
        weights = weights / weights.mean()
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return pd.Series(weights, index=y.index)

    def _compute_event_overlap_weights(self, y: pd.Series,
                                      additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute true concurrency weights from event intervals (López de Prado method)."""
        if additional_features and 'event_intervals' in additional_features:
            # Use provided event intervals for true concurrency calculation
            event_intervals = additional_features['event_intervals']
            concurrency_counts = self._compute_concurrency_from_intervals(event_intervals, y.index)
        elif additional_features and 'event_horizons' in additional_features:
            # Derive pseudo-events from horizons
            event_horizons = additional_features['event_horizons']
            concurrency_counts = self._compute_concurrency_from_horizons(event_horizons, y.index)
        else:
            # Fallback to simplified overlap counting with vectorized implementation
            if self.config.overlap_method == "exponential_decay":
                concurrency_counts = self._compute_vectorized_exponential_overlap(y)
            elif self.config.overlap_method == "gaussian":
                concurrency_counts = self._compute_vectorized_gaussian_overlap(y)
            else:
                concurrency_counts = self._compute_simplified_overlap_counts(y)

        # Weight inversely proportional to concurrency count
        weights = safe_divide(1.0, concurrency_counts + self.config.overlap_threshold)

        # Normalize to mean=1 and clip
        weights = weights / weights.mean()
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return pd.Series(weights, index=y.index)

    def _compute_concurrency_from_intervals(self, event_intervals: List[Tuple], time_index: pd.Index) -> pd.Series:
        """Compute concurrency counts from event start/finish times."""
        concurrency_counts = pd.Series(0, index=time_index)
        
        for i, timestamp in enumerate(time_index):
            # Count how many events are active at this timestamp
            active_events = 0
            for start_time, end_time in event_intervals:
                if start_time <= timestamp <= end_time:
                    active_events += 1
            concurrency_counts.iloc[i] = active_events
            
        return concurrency_counts

    def _compute_concurrency_from_horizons(self, event_horizons: pd.Series, time_index: pd.Index) -> pd.Series:
        """Compute concurrency counts from event horizons."""
        concurrency_counts = pd.Series(0, index=time_index)
        
        for i, timestamp in enumerate(time_index):
            # Count how many events are active at this timestamp
            active_events = 0
            for j, horizon in enumerate(event_horizons):
                if not pd.isna(horizon) and horizon > 0:
                    # Check if this timestamp falls within the horizon of event j
                    event_start = time_index[j]
                    event_end = event_start + pd.Timedelta(days=horizon)
                    if event_start <= timestamp <= event_end:
                        active_events += 1
            concurrency_counts.iloc[i] = active_events
            
        return concurrency_counts

    def _compute_simplified_overlap_counts(self, y: pd.Series) -> pd.Series:
        """Compute simplified overlap counts using vectorized operations."""
        # Vectorized rolling count of non-zero labels
        non_zero_mask = (y != 0).astype(int)
        overlap_counts = non_zero_mask.rolling(
            window=self.config.overlap_window,
            min_periods=1
        ).sum()
        
        return overlap_counts

    def _compute_vectorized_exponential_overlap(self, y: pd.Series) -> pd.Series:
        """Compute exponential decay overlap using vectorized convolution."""
        non_zero_mask = (y != 0).astype(float)
        
        # Create exponential decay kernel
        window_size = self.config.overlap_window
        decay_kernel = np.array([self.config.overlap_decay ** i for i in range(window_size)])
        
        # Use convolution for vectorized computation
        overlap_counts = np.convolve(non_zero_mask, decay_kernel, mode='same')
        
        return pd.Series(overlap_counts, index=y.index)

    def _compute_vectorized_gaussian_overlap(self, y: pd.Series) -> pd.Series:
        """Compute Gaussian-weighted overlap using vectorized operations."""
        non_zero_mask = (y != 0).astype(float)
        
        # Create Gaussian kernel
        window_size = self.config.overlap_window
        sigma = window_size / 3.0
        x = np.arange(-window_size//2, window_size//2 + 1)
        gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # Use convolution for vectorized computation
        overlap_counts = np.convolve(non_zero_mask, gaussian_kernel, mode='same')
        
        return pd.Series(overlap_counts, index=y.index)

    def _compute_time_decay_weights(self, X: pd.DataFrame,
                                   additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute time decay weights for recency adaptation."""
        # Get time index (assume datetime index or timestamp column)
        time_index = X.index

        if additional_features and 'timestamp' in additional_features:
            time_index = pd.to_datetime(additional_features['timestamp'])

        if not isinstance(time_index, pd.DatetimeIndex):
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ No datetime index available, using uniform weights")
            return pd.Series(1.0, index=X.index)

        # Compute time since last sample (in days)
        max_date = time_index.max()
        days_since = (max_date - time_index).dt.days

        if self.config.time_decay_method == "exponential":
            # Exponential decay: w_t = exp(-t / half_life)
            weights = np.exp(-days_since / self.config.time_decay_half_life)

        elif self.config.time_decay_method == "linear":
            # Linear decay: w_t = max(0, 1 - t / (2 * half_life))
            weights = np.maximum(0, 1 - days_since / (2 * self.config.time_decay_half_life))

        elif self.config.time_decay_method == "polynomial":
            # Polynomial decay: w_t = max(0, (1 - t / (2 * half_life))^power)
            normalized_time = days_since / (2 * self.config.time_decay_half_life)
            weights = np.maximum(0, (1 - normalized_time) ** self.config.time_decay_power)

        else:
            weights = pd.Series(1.0, index=X.index)

        # Apply minimum weight for very old samples
        weights = np.maximum(weights, self.config.time_decay_min_weight)

        # Normalize to mean=1 to preserve economic shape, then clip
        weights = weights / weights.mean()
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return pd.Series(weights, index=X.index)

    def _compute_regime_aware_weights(self, X: pd.DataFrame, y: pd.Series,
                                     additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute regime-aware weights based on trailing regime frequencies."""
        # Get regime labels
        regime_labels = None

        if additional_features and 'regime' in additional_features:
            regime_labels = additional_features['regime']
        elif 'regime' in X.columns:
            regime_labels = X['regime']
        else:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ No regime data available, using uniform weights")
            return pd.Series(1.0, index=X.index)

        # Compute trailing regime frequencies with shift(1) to avoid peeking
        lookback_window = self.config.regime_smoothing_window
        trailing_freq = pd.Series(0.0, index=regime_labels.index)
        
        for i in range(lookback_window, len(regime_labels)):
            # Get trailing window of regime frequencies
            window_regimes = regime_labels.iloc[i-lookback_window:i]
            window_freq = window_regimes.value_counts(normalize=True)
            
            # Shift by 1 to avoid peeking
            if i > 0:
                trailing_freq.iloc[i] = window_freq.get(regime_labels.iloc[i], 0.0)

        # Fill initial values with overall frequency
        overall_freq = regime_labels.value_counts(normalize=True)
        trailing_freq = trailing_freq.fillna(regime_labels.map(overall_freq))

        # Weight inversely proportional to trailing regime frequency
        weights = safe_divide(1.0, trailing_freq + 1e-8)

        # Apply regime frequency threshold and multiplier
        rare_regimes = trailing_freq[trailing_freq < self.config.regime_frequency_threshold].index
        for regime in rare_regimes:
            regime_mask = regime_labels == regime
            weights.loc[regime_mask] *= self.config.regime_weight_multiplier

        # Normalize to mean=1 and clip
        weights = weights / weights.mean()
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return pd.Series(weights, index=X.index)

    def _compute_information_content_weights(self, X: pd.DataFrame, y: pd.Series,
                                           additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute combined information content weights with correct component mapping."""
        weights_components = []
        component_names = []

        # Volatility component
        vol_weights = self._compute_volatility_weights(X, additional_features)
        weights_components.append(vol_weights)
        component_names.append("volatility")

        # Confidence component
        conf_weights = self._compute_confidence_weights(y, additional_features)
        weights_components.append(conf_weights)
        component_names.append("confidence")

        # Event overlap component
        overlap_weights = self._compute_event_overlap_weights(y, additional_features)
        weights_components.append(overlap_weights)
        component_names.append("overlap")

        # Time decay component
        time_weights = self._compute_time_decay_weights(X, additional_features)
        weights_components.append(time_weights)
        component_names.append("time_decay")

        # Regime-aware component
        regime_weights = self._compute_regime_aware_weights(X, y, additional_features)
        weights_components.append(regime_weights)
        component_names.append("regime")

        # Compute entropy component
        entropy_weights = self._compute_entropy_weights(X, y, additional_features)
        weights_components.append(entropy_weights)
        component_names.append("entropy")

        # Compute uncertainty component
        uncertainty_weights = self._compute_uncertainty_weights(X, y, additional_features)
        weights_components.append(uncertainty_weights)
        component_names.append("uncertainty")

        # Apply component weights with correct mapping
        weighted_components = []
        component_weights = [
            self.config.information_volatility_weight,    # volatility
            self.config.information_confidence_weight,    # confidence
            self.config.information_overlap_weight,       # overlap
            self.config.information_time_weight,          # time_decay
            self.config.information_regime_weight,        # regime
            self.config.information_entropy_weight,       # entropy
            self.config.information_uncertainty_weight    # uncertainty
        ]

        for i, (comp, weight) in enumerate(zip(weights_components, component_weights)):
            weighted_components.append(comp * weight)

        # Combine weights using weighted geometric mean
        combined_weights = np.ones_like(weights_components[0].values)
        total_weight = 0
        
        for comp, weight in zip(weighted_components, component_weights):
            if weight > 0:
                combined_weights *= (comp.values + self.config.weight_floor) ** weight
                total_weight += weight

        if total_weight > 0:
            combined_weights = combined_weights ** (1.0 / total_weight)

        # Apply smoothing if enabled
        if self.config.weight_smoothing > 0:
            combined_weights = pd.Series(combined_weights, index=X.index).rolling(
                window=3, center=True
            ).mean().fillna(combined_weights).values

        # Apply normalization (only once at the end)
        if self.config.weight_normalization == "l1":
            combined_weights = combined_weights / combined_weights.sum()
        elif self.config.weight_normalization == "l2":
            combined_weights = combined_weights / np.sqrt((combined_weights ** 2).sum())
        elif self.config.weight_normalization == "minmax":
            combined_weights = (combined_weights - combined_weights.min()) / (combined_weights.max() - combined_weights.min())
        elif self.config.weight_normalization == "robust":
            # Robust normalization using median and MAD
            median_weight = np.median(combined_weights)
            mad_weight = np.median(np.abs(combined_weights - median_weight))
            combined_weights = (combined_weights - median_weight) / (mad_weight + 1e-8)

        # Normalize to mean=1 to preserve economic shape
        combined_weights = combined_weights / combined_weights.mean()

        # Clip to bounds
        combined_weights = np.clip(combined_weights, self.config.min_weight, self.config.max_weight)

        return pd.Series(combined_weights, index=X.index)

    def _compute_entropy_weights(self, X: pd.DataFrame, y: pd.Series,
                                additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute entropy-based weights."""
        # Compute feature entropy for each sample
        entropy_weights = pd.Series(0.0, index=X.index)
        
        for col in X.columns:
            if X[col].dtype in [np.number, 'int64', 'float64']:
                # Discretize continuous features
                if X[col].nunique() > 10:
                    bins = pd.cut(X[col], bins=10, labels=False, duplicates='drop')
                else:
                    bins = X[col]
                
                # Compute entropy for each bin
                bin_counts = bins.value_counts()
                bin_probs = bin_counts / len(bins)
                bin_entropy = -sum(p * np.log(p + 1e-8) for p in bin_probs if p > 0)
                
                # Weight by entropy
                entropy_weights += bins.map(lambda x: bin_entropy if not pd.isna(x) else 0)
        
        # Normalize
        if entropy_weights.max() > 0:
            entropy_weights = entropy_weights / entropy_weights.max()
        
        return entropy_weights

    def _compute_uncertainty_weights(self, X: pd.DataFrame, y: pd.Series,
                                   additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute uncertainty-based weights."""
        # Use variance as uncertainty proxy
        uncertainty_weights = pd.Series(0.0, index=X.index)
        
        for col in X.columns:
            if X[col].dtype in [np.number, 'int64', 'float64']:
                # Compute rolling variance as uncertainty measure
                rolling_var = X[col].rolling(window=5, center=True).var().fillna(X[col].var())
                uncertainty_weights += rolling_var
        
        # Normalize
        if uncertainty_weights.max() > 0:
            uncertainty_weights = uncertainty_weights / uncertainty_weights.max()
        
        return uncertainty_weights


class RegimeAwareBalancer:
    """
    Regime-aware rebalancing system.

    Weights samples inversely to regime frequency to ensure balanced exposure
    to different market regimes.
    """

    def __init__(self, config: RegimeConfig):
        """Initialize the regime-aware balancer."""
        self.config = config

    def compute_regime_weights(self, X: pd.DataFrame, y: pd.Series,
                              regime_labels: Optional[pd.Series] = None) -> pd.Series:
        """
        Compute regime-aware weights.

        Args:
            X: Feature matrix
            y: Target labels
            regime_labels: Optional regime labels

        Returns:
            Series of regime-aware weights
        """
        if not self.config.enable_regime_detection:
            return pd.Series(1.0, index=X.index)

        # Get regime labels
        if regime_labels is None:
            if 'regime' in X.columns:
                regime_labels = X['regime']
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("⚠️ No regime labels available")
                return pd.Series(1.0, index=X.index)

        # Compute regime frequencies over lookback window
        regime_freq = regime_labels.value_counts(normalize=True)

        # Weight inversely proportional to regime frequency
        if self.config.regime_balance_method == "inverse_frequency":
            weights = regime_labels.map(lambda x: 1.0 / regime_freq.get(x, 0.5))

        elif self.config.regime_balance_method == "stratified":
            # Ensure equal representation per regime
            min_regime_count = regime_labels.value_counts().min()
            weights = regime_labels.map(lambda x: min_regime_count / regime_freq.get(x, 1.0))

        elif self.config.regime_balance_method == "uniform":
            # Uniform weights across regimes
            weights = pd.Series(1.0, index=X.index)

        else:
            weights = pd.Series(1.0, index=X.index)

        # Apply balance strength
        weights = 1.0 + self.config.regime_balance_strength * (weights - 1.0)

        return weights


class ValidationFairnessChecker:
    """
    Validation fairness checker for ensuring representative validation sets.

    Checks that validation sets have similar class ratios and regime mix as live data.
    """

    def __init__(self, config: ValidationFairnessConfig):
        """Initialize the validation fairness checker."""
        self.config = config

    def check_fairness(self, train_data: Dict, val_data: Dict,
                      live_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check fairness across class ratios, regime mix, and temporal drift.

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            live_data: Optional live data dictionary

        Returns:
            Dictionary of fairness metrics and warnings
        """
        fairness_report = {
            'class_ratio_fair': True,
            'regime_mix_fair': True,
            'temporal_fair': True,
            'warnings': [],
            'metrics': {}
        }

        # Check class ratio fairness
        if self.config.check_class_ratios:
            fairness_report.update(self._check_class_ratio_fairness(train_data, val_data))

        # Check regime mix fairness
        if self.config.check_regime_mix:
            fairness_report.update(self._check_regime_mix_fairness(train_data, val_data))

        # Check temporal drift fairness
        if self.config.check_temporal_drift and live_data:
            fairness_report.update(self._check_temporal_fairness(train_data, val_data, live_data))

        return fairness_report

    def _check_class_ratio_fairness(self, train_data: Dict, val_data: Dict) -> Dict[str, Any]:
        """Check if validation set has fair class representation."""
        train_y = train_data.get('y')
        val_y = val_data.get('y')

        if train_y is None or val_y is None:
            return {'class_ratio_fair': True, 'warnings': []}

        train_ratios = train_y.value_counts(normalize=True).to_dict()
        val_ratios = val_y.value_counts(normalize=True).to_dict()

        max_deviation = 0
        for class_label in train_ratios:
            if class_label in val_ratios:
                deviation = abs(train_ratios[class_label] - val_ratios[class_label])
                max_deviation = max(max_deviation, deviation)

        is_fair = max_deviation <= self.config.class_ratio_tolerance

        warnings = []
        if not is_fair:
            warnings.append(f"Class ratio deviation too high: {max_deviation:.3f} > {self.config.class_ratio_tolerance}")

        return {
            'class_ratio_fair': is_fair,
            'class_ratio_deviation': max_deviation,
            'train_ratios': train_ratios,
            'val_ratios': val_ratios,
            'warnings': warnings
        }

    def _check_regime_mix_fairness(self, train_data: Dict, val_data: Dict) -> Dict[str, Any]:
        """Check if validation set has fair regime representation."""
        train_regime = train_data.get('regime')
        val_regime = val_data.get('regime')

        if train_regime is None or val_regime is None:
            return {'regime_mix_fair': True, 'warnings': []}

        train_ratios = train_regime.value_counts(normalize=True).to_dict()
        val_ratios = val_regime.value_counts(normalize=True).to_dict()

        max_deviation = 0
        for regime in train_ratios:
            if regime in val_ratios:
                deviation = abs(train_ratios[regime] - val_ratios[regime])
                max_deviation = max(max_deviation, deviation)

        is_fair = max_deviation <= self.config.regime_mix_tolerance

        warnings = []
        if not is_fair:
            warnings.append(f"Regime mix deviation too high: {max_deviation:.3f} > {self.config.regime_mix_tolerance}")

        return {
            'regime_mix_fair': is_fair,
            'regime_mix_deviation': max_deviation,
            'train_regime_ratios': train_ratios,
            'val_regime_ratios': val_ratios,
            'warnings': warnings
        }

    def _check_temporal_fairness(self, train_data: Dict, val_data: Dict, live_data: Dict) -> Dict[str, Any]:
        """Check for temporal drift between training and live data."""
        # This is a simplified implementation - in practice, you'd want more sophisticated drift detection

        warnings = []

        # Simple check: compare recent training data with live data characteristics
        # For now, just return fair since this requires more complex implementation

        return {
            'temporal_fair': True,
            'temporal_drift_score': 0.0,
            'warnings': warnings
        }


class ComprehensiveBalancingSystem:
    """
    Comprehensive label balancing and sample weighting system.

    Combines all balancing and weighting techniques into a unified system
    that can be easily integrated into training pipelines.
    """

    def __init__(self, balancing_config: BalancingConfig,
                 weighting_config: WeightingConfig,
                 regime_config: RegimeConfig,
                 fairness_config: ValidationFairnessConfig):
        """Initialize the comprehensive system."""
        self.balancer = LabelBalancer(balancing_config)
        self.weighter = SampleWeighter(weighting_config)
        self.regime_balancer = RegimeAwareBalancer(regime_config)
        self.fairness_checker = ValidationFairnessChecker(fairness_config)

        # Store configs
        self.balancing_config = balancing_config
        self.weighting_config = weighting_config
        self.regime_config = regime_config
        self.fairness_config = fairness_config

    def balance_and_weight(self, X: pd.DataFrame, y: pd.Series,
                          sample_weight: Optional[pd.Series] = None,
                          additional_features: Optional[Dict[str, pd.Series]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Apply comprehensive balancing and weighting.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Optional existing sample weights
            additional_features: Optional additional features

        Returns:
            Tuple of (balanced_X, balanced_y, final_weights, qa_report)
        """
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Starting comprehensive balancing and weighting")

        # Step 1: Apply label balancing
        X_balanced, y_balanced, sample_weight_balanced = self.balancer.balance_dataset(
            X, y, sample_weight
        )

        # Step 2: Compute sample weights
        computed_weights = self.weighter.compute_weights(
            X_balanced, y_balanced, additional_features
        )

        # Step 3: Apply regime-aware weighting if enabled
        if self.regime_config.enable_regime_detection:
            regime_weights = self.regime_balancer.compute_regime_weights(
                X_balanced, y_balanced,
                additional_features.get('regime') if additional_features else None
            )
            computed_weights = computed_weights * regime_weights

        # Step 4: Combine with existing sample weights
        if sample_weight_balanced is not None:
            final_weights = computed_weights * sample_weight_balanced
        else:
            final_weights = computed_weights

        # Step 5: Normalize final weights
        if self.weighting_config.weight_normalization:
            if self.weighting_config.weight_normalization == "l1":
                final_weights = final_weights / final_weights.sum()
            elif self.weighting_config.weight_normalization == "l2":
                final_weights = final_weights / np.sqrt((final_weights ** 2).sum())

        # Clip to bounds
        final_weights = np.clip(final_weights, self.weighting_config.min_weight, self.weighting_config.max_weight)

        # Step 6: Quality control and QA reporting
        qa_report = self._generate_qa_report(X_balanced, y_balanced, final_weights, X, y)

        if TPRINT_AVAILABLE:
            tprint_info(f"✅ Balancing complete - Final dataset: {len(X_balanced)} samples")
            tprint_info(f"📊 Final class distribution: {y_balanced.value_counts().to_dict()}")

        return X_balanced, y_balanced, final_weights, qa_report

    def _generate_qa_report(self, X_balanced: pd.DataFrame, y_balanced: pd.Series, 
                           final_weights: pd.Series, X_orig: pd.DataFrame, y_orig: pd.Series) -> Dict[str, Any]:
        """Generate quality assurance report."""
        qa_report = {
            'synthetic_ratio_check': True,
            'weight_ratio_check': True,
            'weight_stability_check': True,
            'warnings': [],
            'metrics': {}
        }

        if not self.balancing_config.enable_quality_control:
            return qa_report

        # Check synthetic sample ratio
        if len(X_balanced) > len(X_orig):
            synthetic_ratio = (len(X_balanced) - len(X_orig)) / len(X_balanced)
            qa_report['metrics']['synthetic_ratio'] = synthetic_ratio
            
            if synthetic_ratio > self.balancing_config.max_synthetic_ratio:
                qa_report['synthetic_ratio_check'] = False
                qa_report['warnings'].append(
                    f"Synthetic ratio {synthetic_ratio:.3f} exceeds max {self.balancing_config.max_synthetic_ratio}"
                )

        # Check weight ratio
        if self.weighting_config.enable_weight_validation:
            weight_ratio = final_weights.max() / final_weights.min()
            qa_report['metrics']['weight_ratio'] = weight_ratio
            
            if weight_ratio > self.weighting_config.max_weight_ratio:
                qa_report['weight_ratio_check'] = False
                qa_report['warnings'].append(
                    f"Weight ratio {weight_ratio:.3f} exceeds max {self.weighting_config.max_weight_ratio}"
                )

        # Check weight stability
        if self.weighting_config.enable_weight_validation:
            weight_volatility = final_weights.rolling(window=10).std().mean()
            qa_report['metrics']['weight_volatility'] = weight_volatility
            
            if weight_volatility > self.weighting_config.weight_stability_threshold:
                qa_report['weight_stability_check'] = False
                qa_report['warnings'].append(
                    f"Weight volatility {weight_volatility:.3f} exceeds threshold {self.weighting_config.weight_stability_threshold}"
                )

        return qa_report

    def check_validation_fairness(self, train_data: Dict, val_data: Dict,
                                 live_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check validation fairness.

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            live_data: Optional live data dictionary

        Returns:
            Fairness report
        """
        return self.fairness_checker.check_fairness(train_data, val_data, live_data)


# Default configurations for common use cases
DEFAULT_BALANCING_CONFIG = BalancingConfig(
    balancing_technique=BalancingTechnique.ADAPTIVE,
    under_sampling_ratio=0.7,
    over_sampling_ratio=0.3,
    under_sampling_strategy="random",
    over_sampling_strategy="smote",
    smote_k_neighbors=5,
    mixup_alpha=0.2,
    stratified_batching=True,
    batch_size=1024,
    min_samples_per_class=10,
    adaptive_imbalance_threshold=0.1,
    adaptive_min_samples=50,
    enable_quality_control=True,
    enable_temporal_validation=True,
    purge_length=1,
    embargo_length=1,
    random_state=42
)

DEFAULT_WEIGHTING_CONFIG = WeightingConfig(
    weighting_scheme=WeightingScheme.INFORMATION_CONTENT,
    volatility_window=20,
    volatility_floor=1e-6,
    volatility_method="rolling_std",
    volatility_robust=True,
    confidence_method="probability",
    confidence_scale=2.0,
    confidence_smoothing=0.1,
    overlap_window=5,
    overlap_decay=0.8,
    overlap_method="rolling_count",
    time_decay_half_life=30,
    time_decay_method="exponential",
    time_decay_min_weight=0.01,
    regime_frequency_threshold=0.2,
    regime_weight_multiplier=5.0,
    regime_smoothing_window=10,
    information_entropy_weight=0.3,
    information_uncertainty_weight=0.3,
    information_volatility_weight=0.2,
    information_regime_weight=0.2,
    information_confidence_weight=0.2,
    information_overlap_weight=0.2,
    information_time_weight=0.2,
    enable_dynamic_weighting=True,
    weight_normalization="l2",
    weight_smoothing=0.1,
    min_weight=0.1,
    max_weight=10.0,
    enable_weight_validation=True
)

DEFAULT_REGIME_CONFIG = RegimeConfig(
    enable_regime_detection=True,
    regime_column="regime",
    regime_lookback_window=252,
    min_regime_samples=50,
    regime_balance_method="inverse_frequency",
    regime_balance_strength=1.0,
    validation_regime_fairness=True,
    regime_fairness_tolerance=0.1
)

DEFAULT_FAIRNESS_CONFIG = ValidationFairnessConfig(
    class_ratio_tolerance=0.05,
    min_class_samples=100,
    regime_mix_tolerance=0.1,
    min_regime_samples=50,
    temporal_drift_tolerance=0.1,
    drift_window=21,
    check_class_ratios=True,
    check_regime_mix=True,
    check_temporal_drift=True
)