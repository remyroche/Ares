"""
Label Balancing & Sample Weighting System

This module implements comprehensive label balancing and sample weighting techniques
to address the extreme class imbalance in financial datasets (80-95% "no-trade" samples).

Key Features:
- Under-sampling of majority class (no-trade samples)
- Over-sampling of minority classes using SMOTE and mixup
- Stratified batching for streaming training
- Multiple sample weighting schemes (volatility, confidence, event overlap, time decay)
- Regime-aware rebalancing
- Validation fairness checks

The system is designed to "teach the model what matters" by ensuring balanced exposure
to different classes and weighting samples by information content rather than just class balance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
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
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
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

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    if TPRINT_AVAILABLE:
        tprint_warning("⚠️ imbalanced-learn not available, using basic resampling methods")


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
                       sample_weight: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """
        Apply balancing to the dataset.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Optional sample weights

        Returns:
            Tuple of (balanced_X, balanced_y, balanced_weights)
        """
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Starting label balancing with technique: {self.config.balancing_technique.value}")

        # Get class distribution info
        class_counts = y.value_counts()
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
        """Under-sample the majority class."""
        class_counts = y.value_counts()
        majority_class = class_counts.index[0]
        majority_count = class_counts.iloc[0]

        # Calculate target count for majority class
        if self.config.target_distribution is not None:
            target_maj_count = int(len(y) * self.config.target_distribution.get(majority_class, 0.5))
        else:
            # Maintain ratio based on minority classes
            minority_total = len(y) - majority_count
            target_maj_count = int(minority_total / (1 - self.config.under_sampling_ratio))

        target_maj_count = min(target_maj_count, majority_count)

        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Under-sampling majority class {majority_class} from {majority_count} to {target_maj_count}")

        # Under-sample majority class
        majority_indices = y[y == majority_class].index
        if len(majority_indices) > target_maj_count:
            undersampled_indices = np.random.choice(
                majority_indices, size=target_maj_count, replace=False
            )
        else:
            undersampled_indices = majority_indices

        # Combine with all minority samples
        minority_indices = y[y != majority_class].index
        final_indices = np.concatenate([undersampled_indices, minority_indices])

        return X.loc[final_indices], y.loc[final_indices], sample_weight.loc[final_indices] if sample_weight is not None else None

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
        """Apply SMOTE oversampling."""
        if not IMBLEARN_AVAILABLE:
            if TPRINT_AVAILABLE:
                tprint_warning("⚠️ SMOTE not available, falling back to random oversampling")
            return self._over_sample(X, y, sample_weight)

        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🧬 Applying SMOTE to reach majority class count: {max_count}")

        # Determine sampling strategy
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < max_count:
                sampling_strategy[class_label] = max_count

        if not sampling_strategy:
            return X, y, sample_weight

        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.config.random_state,
            k_neighbors=min(5, min(class_counts[class_counts > 1]) - 1)
        )

        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Handle sample weights for SMOTE samples (use original weights)
        if sample_weight is not None:
            # Original weights for original samples
            original_weights = sample_weight.copy()

            # For synthetic samples, use average weight of k-nearest neighbors
            synthetic_weights = []
            for i in range(len(X), len(X_resampled)):
                # This is a simplified approach - in practice, you'd want to
                # interpolate weights from the k-nearest neighbors
                synthetic_weights.append(sample_weight.mean())

            balanced_sample_weight = pd.concat([
                pd.Series(original_weights),
                pd.Series(synthetic_weights)
            ]).reset_index(drop=True)
        else:
            balanced_sample_weight = None

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled), balanced_sample_weight

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
        """Apply Mixup augmentation."""
        class_counts = y.value_counts()
        max_count = class_counts.max()

        if TPRINT_AVAILABLE:
            tprint_info(f"🍹 Applying Mixup augmentation to reach majority class count: {max_count}")

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
                    idx1, idx2 = np.random.choice(len(class_data), 2, replace=False)

                    # Mixup ratio (beta distribution)
                    alpha = 0.2  # Mixup alpha parameter
                    lambda_param = np.random.beta(alpha, alpha)

                    # Mix features
                    mixed_features = lambda_param * class_data.iloc[idx1] + (1 - lambda_param) * class_data.iloc[idx2]

                    # Mix labels (soft labels)
                    mixed_label = lambda_param * class_labels.iloc[idx1] + (1 - lambda_param) * class_labels.iloc[idx2]

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
        """Create stratified batches for streaming training."""
        if TPRINT_AVAILABLE:
            tprint_info(f"📦 Creating stratified batches with size: {self.config.batch_size}")

        # Create stratified batches ensuring each batch has representation from all classes
        skf = StratifiedKFold(n_splits=max(2, len(X) // self.config.batch_size), shuffle=True, random_state=self.config.random_state)

        batches_X = []
        batches_y = []
        batches_weights = [] if sample_weight is not None else None

        for train_idx, test_idx in skf.split(X, y):
            # Ensure minimum samples per class in each batch
            batch_y = y.iloc[test_idx]

            if len(batch_y.unique()) >= 2 and all(batch_y.value_counts() >= self.config.min_samples_per_class):
                batches_X.append(X.iloc[test_idx])
                batches_y.append(batch_y)

                if sample_weight is not None:
                    batches_weights.append(sample_weight.iloc[test_idx])

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
        """Apply adaptive balancing based on dataset characteristics."""
        if TPRINT_AVAILABLE:
            tprint_info("🧠 Applying adaptive balancing...")

        # Analyze dataset characteristics
        class_counts = y.value_counts()
        n_classes = len(class_counts)
        total_samples = len(y)
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = min_count / max_count

        if TPRINT_AVAILABLE:
            tprint_info(f"📊 Dataset analysis: {n_classes} classes, {total_samples} samples")
            tprint_info(f"   → Imbalance ratio: {imbalance_ratio:.3f}")
            tprint_info(f"   → Min samples: {min_count}, Max samples: {max_count}")

        # Select technique based on characteristics
        if imbalance_ratio < self.config.adaptive_imbalance_threshold:
            if min_count < self.config.adaptive_min_samples:
                # Very imbalanced with few minority samples - use SMOTE
                if TPRINT_AVAILABLE:
                    tprint_info("🎯 Very imbalanced dataset - using SMOTE")
                return self._smote_sample(X, y, sample_weight)
            else:
                # Moderately imbalanced - use hybrid approach
                if TPRINT_AVAILABLE:
                    tprint_info("🎯 Moderately imbalanced dataset - using hybrid approach")
                return self._hybrid_balance(X, y, sample_weight)
        else:
            # Well balanced - use stratified batching
            if TPRINT_AVAILABLE:
                tprint_info("🎯 Well balanced dataset - using stratified batching")
            return self._create_stratified_batches(X, y, sample_weight)


class SampleWeighter:
    """
    Sample weighting system for financial datasets.

    Implements various weighting schemes to emphasize samples with higher information content:
    - Volatility weighting: de-emphasize noisy high-vol periods
    - Confidence weighting: weight by label confidence
    - Event overlap weighting: prevent duplicated exposure (López de Prado)
    - Time decay weighting: keep model adaptive to latest dynamics
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
        """Compute volatility-based weights using VectorBT optimization (w_t ∝ 1/σ_t)."""
        # Try to get volatility from additional features or compute from returns
        volatility = None

        if additional_features and 'volatility' in additional_features:
            volatility = additional_features['volatility']
        elif 'returns' in X.columns:
            # Compute realized volatility from returns using VectorBT
            returns = X['returns']
            if VECTORBT_AVAILABLE and self._should_use_vectorbt(returns):
                tprint_info("📊 Using VectorBT for volatility weight calculation")
                if self.config.volatility_method == "rolling_std":
                    volatility = rolling_std(returns, window=self.config.volatility_window)
                elif self.config.volatility_method == "ewma":
                    # Use VectorBT for EWMA
                    volatility = vbt.rolling_apply(
                        returns, 
                        lambda x: x.ewm(span=self.config.volatility_window).std().iloc[-1],
                        window=self.config.volatility_window
                    )
                else:  # garch
                    volatility = rolling_std(returns, window=self.config.volatility_window)
            else:
                # Fallback to pandas
                if self.config.volatility_method == "rolling_std":
                    volatility = returns.rolling(window=self.config.volatility_window).std()
                elif self.config.volatility_method == "ewma":
                    volatility = returns.ewm(span=self.config.volatility_window).std()
                else:  # garch
                    volatility = returns.rolling(window=self.config.volatility_window).std()
            
            volatility = volatility.fillna(volatility.mean())
        else:
            # Use price volatility if available
            price_cols = [col for col in X.columns if col in ['close', 'price', 'Close', 'Price']]
            if price_cols:
                price_data = X[price_cols[0]]
                returns = price_data.pct_change()
                if VECTORBT_AVAILABLE and self._should_use_vectorbt(returns):
                    if self.config.volatility_method == "rolling_std":
                        volatility = rolling_std(returns, window=self.config.volatility_window)
                    elif self.config.volatility_method == "ewma":
                        volatility = vbt.rolling_apply(
                            returns, 
                            lambda x: x.ewm(span=self.config.volatility_window).std().iloc[-1],
                            window=self.config.volatility_window
                        )
                    else:
                        volatility = rolling_std(returns, window=self.config.volatility_window)
                else:
                    # Fallback to pandas
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

        # Apply robust volatility estimation if enabled
        if self.config.volatility_robust:
            # Use median absolute deviation for robust estimation
            mad = np.median(np.abs(volatility - np.median(volatility)))
            volatility = np.maximum(volatility, mad * 1.4826)  # Scale factor for normal distribution

        # Compute weights inversely proportional to volatility using VectorBT
        if VECTORBT_AVAILABLE and self._should_use_vectorbt(volatility):
            weights = 1.0 / (volatility + self.config.volatility_floor)
        else:
            weights = safe_divide(1.0, volatility + self.config.volatility_floor)

        # Cap maximum weight
        weights = np.clip(weights, 0, self.config.volatility_weight_max)

        return weights
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))

    def _compute_confidence_weights(self, y: pd.Series,
                                   additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute confidence-based weights (w_t ∝ Δp)."""
        if additional_features and 'confidence' in additional_features:
            confidence = pd.Series(additional_features['confidence'], index=y.index)
        else:
            # Compute confidence based on different methods
            if self.config.confidence_method == "probability":
                # Use inverse of class frequency as confidence proxy
                class_freq = y.value_counts(normalize=True)
                confidence = y.map(lambda x: 1.0 / class_freq.get(x, 0.5))
            elif self.config.confidence_method == "margin":
                # Use margin-based confidence (distance from decision boundary)
                class_freq = y.value_counts(normalize=True)
                max_freq = class_freq.max()
                confidence = y.map(lambda x: abs(class_freq.get(x, 0.5) - max_freq) + 0.1)
            elif self.config.confidence_method == "entropy":
                # Use entropy-based confidence
                class_freq = y.value_counts(normalize=True)
                entropy = -sum(p * np.log(p + 1e-8) for p in class_freq.values())
                confidence = y.map(lambda x: entropy / (class_freq.get(x, 0.5) + 1e-8))
            elif self.config.confidence_method == "uncertainty":
                # Use uncertainty-based confidence
                class_freq = y.value_counts(normalize=True)
                uncertainty = 1.0 - class_freq.max()
                confidence = y.map(lambda x: uncertainty / (class_freq.get(x, 0.5) + 1e-8))
            else:
                # Default to probability method
                class_freq = y.value_counts(normalize=True)
                confidence = y.map(lambda x: 1.0 / class_freq.get(x, 0.5))

        confidence = pd.Series(confidence, index=y.index, dtype=float)

        # Apply smoothing
        if self.config.confidence_smoothing > 0:
            confidence = confidence.rolling(window=3, min_periods=1).mean()

        # Apply minimum threshold
        confidence = confidence.clip(lower=self.config.confidence_min_threshold)

        # Scale confidence weights
        weights = self.config.confidence_scale * confidence

        # Normalize and clip
        weights_min = weights.min()
        weights_range = weights.max() - weights_min

        if weights_range > 0:
            weights = (weights - weights_min) / (weights_range + 1e-8)
        else:
            weights = pd.Series(1.0, index=weights.index)

        weights = weights.clip(lower=self.config.min_weight, upper=self.config.max_weight)

        return weights

    def _compute_event_overlap_weights(self, y: pd.Series,
                                      additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute event overlap weights (López de Prado method)."""
        if additional_features and 'overlap_count' in additional_features:
            overlap_counts = additional_features['overlap_count']
        else:
            # Compute overlap counts based on different methods
            if self.config.overlap_method == "rolling_count":
                # Simple rolling count of non-zero labels
                non_zero_mask = (y != 0).astype(int)
                overlap_counts = non_zero_mask.rolling(
                    window=self.config.overlap_window,
                    min_periods=1
                ).sum()
            elif self.config.overlap_method == "exponential_decay":
                # Exponential decay for distant events
                non_zero_mask = (y != 0).astype(int)
                overlap_counts = pd.Series(0.0, index=y.index)
                for i in range(len(y)):
                    if non_zero_mask.iloc[i] == 1:
                        # Apply exponential decay to future events
                        future_window = min(self.config.overlap_window, len(y) - i)
                        for j in range(1, future_window):
                            if i + j < len(y):
                                decay_factor = self.config.overlap_decay ** j
                                overlap_counts.iloc[i + j] += decay_factor
            elif self.config.overlap_method == "gaussian":
                # Gaussian-weighted overlap counting
                non_zero_mask = (y != 0).astype(int)
                overlap_counts = pd.Series(0.0, index=y.index)
                for i in range(len(y)):
                    if non_zero_mask.iloc[i] == 1:
                        # Apply Gaussian weights to nearby events
                        for j in range(max(0, i - self.config.overlap_window), 
                                     min(len(y), i + self.config.overlap_window + 1)):
                            if j != i:
                                distance = abs(j - i)
                                gaussian_weight = np.exp(-0.5 * (distance / (self.config.overlap_window / 3)) ** 2)
                                overlap_counts.iloc[j] += gaussian_weight
            else:
                # Default to rolling count
                non_zero_mask = (y != 0).astype(int)
                overlap_counts = non_zero_mask.rolling(
                    window=self.config.overlap_window,
                    min_periods=1
                ).sum()

        # Apply threshold filtering
        overlap_counts = np.maximum(overlap_counts, self.config.overlap_threshold)

        # Weight inversely proportional to overlap count
        weights = safe_divide(1.0, overlap_counts)

        # Apply exponential decay for distant overlaps
        if self.config.overlap_method != "exponential_decay":
            decay_weights = self.config.overlap_decay ** (overlap_counts - 1)
            weights = weights * decay_weights

        # Normalize and clip
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return weights

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

        # Normalize and clip
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return weights

    def _compute_regime_aware_weights(self, X: pd.DataFrame, y: pd.Series,
                                     additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute regime-aware weights based on inverse regime frequency."""
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

        # Compute regime frequencies with smoothing
        regime_freq = regime_labels.value_counts(normalize=True)
        
        # Apply smoothing window if specified
        if self.config.regime_smoothing_window > 1:
            # Simple smoothing by averaging with nearby regimes
            regime_freq_smoothed = regime_freq.copy()
            for regime in regime_freq.index:
                # Find similar regimes (simplified approach)
                similar_regimes = [r for r in regime_freq.index if abs(regime_freq[r] - regime_freq[regime]) < 0.1]
                if len(similar_regimes) > 1:
                    regime_freq_smoothed[regime] = regime_freq[similar_regimes].mean()
            regime_freq = regime_freq_smoothed

        # Weight inversely proportional to regime frequency
        weights = regime_labels.map(lambda x: 1.0 / regime_freq.get(x, 0.5))

        # Apply regime frequency threshold and multiplier
        rare_regimes = regime_freq[regime_freq < self.config.regime_frequency_threshold].index
        for i, regime in enumerate(regime_labels):
            if regime in rare_regimes:
                weights.iloc[i] *= self.config.regime_weight_multiplier

        # Apply adaptation rate for dynamic weighting
        if self.config.regime_adaptation_rate > 0:
            # Simple adaptation: gradually adjust weights based on recent regime frequency
            recent_regime_freq = regime_labels.tail(self.config.regime_smoothing_window).value_counts(normalize=True)
            for regime in recent_regime_freq.index:
                regime_mask = regime_labels == regime
                if regime in regime_freq.index:
                    adaptation_factor = 1.0 + self.config.regime_adaptation_rate * (recent_regime_freq[regime] - regime_freq[regime])
                    weights.loc[regime_mask] *= adaptation_factor

        # Normalize and clip
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)

        return weights

    def _compute_information_content_weights(self, X: pd.DataFrame, y: pd.Series,
                                           additional_features: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """Compute combined information content weights."""
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

        # Apply component weights
        weighted_components = []
        component_weights = [
            self.config.information_volatility_weight,
            self.config.information_entropy_weight,
            self.config.information_uncertainty_weight,
            self.config.information_entropy_weight,  # time decay
            self.config.information_regime_weight,
            self.config.information_entropy_weight,
            self.config.information_uncertainty_weight
        ]

        for i, (comp, weight) in enumerate(zip(weights_components, component_weights)):
            weighted_components.append(comp * weight)

        # Combine weights using weighted geometric mean
        combined_weights = np.ones_like(weights_components[0])
        total_weight = 0
        
        for comp, weight in zip(weighted_components, component_weights):
            if weight > 0:
                combined_weights *= (comp + self.config.weight_floor) ** weight
                total_weight += weight

        if total_weight > 0:
            combined_weights = combined_weights ** (1.0 / total_weight)

        # Apply smoothing if enabled
        if self.config.weight_smoothing > 0:
            base_index = None
            if isinstance(weights_components[0], pd.Series):
                base_index = weights_components[0].index

            combined_series = pd.Series(combined_weights, index=base_index)
            combined_series = combined_series.rolling(window=3, min_periods=1).mean()
            combined_weights = combined_series.values if base_index is not None else combined_series.to_numpy()

        # Apply normalization
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

        # Clip to bounds
        combined_weights = np.clip(combined_weights, self.config.min_weight, self.config.max_weight)

        return combined_weights

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
                # Compute rolling variance as uncertainty measure using trailing window
                rolling_var = X[col].rolling(window=5, min_periods=1).var(ddof=0)
                expanding_var = X[col].expanding(min_periods=1).var(ddof=0)
                combined_var = rolling_var.fillna(expanding_var).fillna(0.0)
                uncertainty_weights = uncertainty_weights.add(combined_var, fill_value=0.0)
        
        # Normalize using trailing maximum to avoid future leakage
        if not uncertainty_weights.empty:
            expanding_max = uncertainty_weights.expanding(min_periods=1).max()
            scaled_weights = uncertainty_weights / expanding_max.replace(0, np.nan)
            uncertainty_weights = scaled_weights.fillna(0.0)

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
                          additional_features: Optional[Dict[str, pd.Series]] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Apply comprehensive balancing and weighting.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weight: Optional existing sample weights
            additional_features: Optional additional features

        Returns:
            Tuple of (balanced_X, balanced_y, final_weights)
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

        if TPRINT_AVAILABLE:
            tprint_info(f"✅ Balancing complete - Final dataset: {len(X_balanced)} samples")
            tprint_info(f"📊 Final class distribution: {y_balanced.value_counts().to_dict()}")

        return X_balanced, y_balanced, final_weights

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

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
