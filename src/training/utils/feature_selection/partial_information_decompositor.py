"""
Partial Information Decompositor Module

This module provides partial information decomposition (PID) capabilities for feature selection
and feature engineering. It can identify synergistic, redundant, and unique information
between features and create cross-timeframe, polynomial, and interaction features.

Key capabilities:
- Calculate redundancy, synergy, and unique information between features
- Detect meaningful feature interactions
- Create polynomial features based on PID analysis
- Analyze cross-timeframe dependencies
- Generate interaction features for machine learning models

Author: Feature Selection Framework Team
Version: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import logging
import os
from datetime import datetime
import time
import warnings
from itertools import combinations, product
from dataclasses import dataclass, field

# Import utilities
try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile,
        MathValidationError, validate_positive, validate_range
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Math validation utilities not available: {e}")
    MATH_VALIDATION_AVAILABLE = False
    # Create fallback implementations
    def safe_divide(a, b): return a / b if b != 0 else 0
    def safe_log(x): return np.log(np.maximum(x, 1e-10))
    def safe_sqrt(x): return np.sqrt(np.maximum(x, 0))
    def safe_power(x, p): return np.power(np.maximum(x, 0), p)
    def validate_finite(x): return np.isfinite(x).all()
    def safe_correlation(x, y): return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    def safe_covariance(x, y): return np.cov(x, y)[0, 1] if len(x) > 1 else 0
    def safe_mean(x): return np.mean(x) if len(x) > 0 else 0
    def safe_std(x): return np.std(x) if len(x) > 1 else 0
    def safe_percentile(x, p): return np.percentile(x, p) if len(x) > 0 else 0

# Import common operations and utilities
try:
    from src.utils.common_operations import (
        create_fallback_logger, create_fallback_decorator,
        safe_dataframe_operation, get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        create_directory_safe, get_file_size, validate_file_path
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Common operations not available: {e}")
    COMMON_OPERATIONS_AVAILABLE = False

# Import serialization utilities
try:
    from src.utils.serialization_utils import JSONSerializer, PickleSerializer
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Serialization utilities not available: {e}")
    SERIALIZATION_AVAILABLE = False

# Import parquet utilities
try:
    from src.utils.parquet_utils import ParquetUtils
    PARQUET_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Parquet utilities not available: {e}")
    PARQUET_AVAILABLE = False

# Import matrix operations
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Matrix operations not available: {e}")
    MATRIX_OPERATIONS_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common import tprint
    from src.utils.lookahead_bias_detector import (
        get_global_detector, validate_no_future_data, LookaheadBiasError
    )
    from src.utils.ml_common.validation.thresholding import AdaptiveThresholding
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# Enhanced dependency management
try:
    from src.utils.logger import get_logger
    _LOGGER = get_logger("FeatureSelection.PartialInformationDecompositor")
    if COMMON_OPERATIONS_AVAILABLE:
        _LOGGER = create_fallback_logger(_LOGGER, "FeatureSelection.PartialInformationDecompositor")
except Exception as e:
    if COMMON_OPERATIONS_AVAILABLE:
        _LOGGER = create_fallback_logger(None, "FeatureSelection.PartialInformationDecompositor")
    else:
        _LOGGER = logging.getLogger("FeatureSelection.PartialInformationDecompositor")
        _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

# Try to import sklearn for mutual information calculations
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import entropy
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited PID functionality")

@dataclass
class PIDConfig:
    """Configuration for partial information decomposition."""
    # Dynamic threshold configuration (percentile-based selection)
    use_dynamic_thresholds: bool = True
    top_percentile: float = 0.25  # Select top 25% of feature pairs
    min_pairs_threshold: int = 5   # Minimum number of pairs to select regardless of percentile

    # Fallback static thresholds (used when dynamic thresholds disabled)
    synergy_threshold: float = 0.0001  # Optimized for multi-horizon probability targets
    redundancy_threshold: float = 0.005  # Optimized for multi-horizon probability targets
    unique_info_threshold: float = 0.0001  # Optimized for multi-horizon probability targets

    # Multi-horizon specific thresholds
    multi_horizon_mode: bool = False  # Enable multi-horizon optimizations
    directional_synergy_boost: float = 1.5  # Boost synergy for directional targets
    probability_sensitivity: float = 0.8  # Sensitivity to probability variations
    time_horizon_weight: float = 0.3  # Weight for time horizon considerations

    # Cross-timeframe analysis
    cross_timeframe_threshold: float = 0.15
    max_timeframe_lag: int = 5

    # Polynomial feature creation
    max_polynomial_degree: int = 3
    max_interaction_features: int = 50

    # Computational limits
    max_features_for_full_pid: int = 20
    max_interaction_order: int = 3
    convergence_threshold: float = 1e-6
    max_iterations: int = 100

    # Sampling for large datasets
    sample_size: Optional[int] = None
    random_state: int = 42

    def __post_init__(self):
        """Validate configuration parameters using math validation utilities."""
        if MATH_VALIDATION_AVAILABLE:
            try:
                # Validate thresholds are in valid ranges
                self.synergy_threshold = validate_range(
                    self.synergy_threshold, 0.0, 1.0, "synergy_threshold"
                )
                self.redundancy_threshold = validate_range(
                    self.redundancy_threshold, 0.0, 1.0, "redundancy_threshold"
                )
                self.unique_info_threshold = validate_range(
                    self.unique_info_threshold, 0.0, 1.0, "unique_info_threshold"
                )
                self.cross_timeframe_threshold = validate_range(
                    self.cross_timeframe_threshold, 0.0, 1.0, "cross_timeframe_threshold"
                )

                # Validate positive integers
                self.max_timeframe_lag = validate_positive(
                    self.max_timeframe_lag, "max_timeframe_lag"
                )
                self.max_polynomial_degree = validate_positive(
                    self.max_polynomial_degree, "max_polynomial_degree"
                )
                self.max_interaction_features = validate_positive(
                    self.max_interaction_features, "max_interaction_features"
                )
                self.max_features_for_full_pid = validate_positive(
                    self.max_features_for_full_pid, "max_features_for_full_pid"
                )
                self.max_interaction_order = validate_positive(
                    self.max_interaction_order, "max_interaction_order"
                )
                self.max_iterations = validate_positive(
                    self.max_iterations, "max_iterations"
                )

                # Validate convergence threshold
                self.convergence_threshold = validate_finite(
                    self.convergence_threshold, "convergence_threshold"
                )

                if self.sample_size is not None:
                    self.sample_size = validate_positive(
                        self.sample_size, "sample_size"
                    )

                self.random_state = validate_finite(
                    self.random_state, "random_state"
                )

            except MathValidationError as e:
                logger.warning(f"⚠️ Configuration validation warning: {e}")
                # Use default values for invalid parameters (adjusted for continuous targets)
                self.synergy_threshold = 0.0001
                self.redundancy_threshold = 0.005
                self.unique_info_threshold = 0.0001

@dataclass
class PIDResult:
    """Result of partial information decomposition analysis."""
    # Information measures
    redundancy: Dict[Tuple[str, str], float] = field(default_factory=dict)
    synergy: Dict[Tuple[str, str], float] = field(default_factory=dict)
    unique_info: Dict[str, float] = field(default_factory=dict)

    # Interaction features
    polynomial_features: List[str] = field(default_factory=list)
    interaction_features: List[str] = field(default_factory=list)
    cross_timeframe_features: List[str] = field(default_factory=list)

    # Analysis metadata
    feature_pairs_analyzed: int = 0
    significant_interactions: int = 0
    significant_pairs: List[Tuple[str, str]] = field(default_factory=list)  # Store actual significant pairs
    execution_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)

class PartialInformationDecompositor:
    """Partial Information Decompositor for feature analysis and creation."""

    def __init__(self, config: Optional[PIDConfig] = None):
        """Initialize the partial information decompositor."""
        self.config = config or PIDConfig()
        self.logger = logger.getChild('PartialInformationDecompositor')

        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()

        # Initialize matrix operations
        self._initialize_matrix_operations()

        # Initialize ML utilities
        self._initialize_ml_utilities()

        _LOGGER.info("🔍 PartialInformationDecompositor initialized")
        if self.config.use_dynamic_thresholds:
            _LOGGER.info(f"⚙️ Dynamic thresholds: top {self.config.top_percentile*100:.1f}% of pairs (min: {self.config.min_pairs_threshold})")
        else:
            _LOGGER.info(f"⚙️ Static thresholds - Synergy: {self.config.synergy_threshold}, Redundancy: {self.config.redundancy_threshold}")
        _LOGGER.info(f"⚙️ Max polynomial degree: {self.config.max_polynomial_degree}")
        _LOGGER.info(f"⚙️ Max interaction features: {self.config.max_interaction_features}")

        if self.config.multi_horizon_mode:
            _LOGGER.info(f"🎯 Multi-horizon mode enabled - directional boost: {self.config.directional_synergy_boost}x")
            _LOGGER.info(f"🎯 Probability sensitivity: {self.config.probability_sensitivity}")
            _LOGGER.info(f"🎯 Time horizon weight: {self.config.time_horizon_weight}")

    def _initialize_hardware_optimizations(self):
        """Initialize hardware optimization utilities."""
        try:
            if COMMON_OPERATIONS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                if self.gpu_manager:
                    _LOGGER.info("✅ M1 GPU manager initialized")
                if self.memory_optimizer:
                    _LOGGER.info("✅ M1 memory optimizer initialized")
                if self.cpu_optimizer:
                    _LOGGER.info("✅ M1 CPU optimizer initialized")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    def _initialize_matrix_operations(self):
        """Initialize matrix operations utilities."""
        try:
            if MATRIX_OPERATIONS_AVAILABLE:
                self.matrix_ops = get_unified_matrix_operations()
                _LOGGER.info("✅ Unified matrix operations initialized")
            else:
                self.matrix_ops = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ Matrix operations initialization failed: {e}")
            self.matrix_ops = None

    def _initialize_ml_utilities(self):
        """Initialize ML utilities."""
        try:
            if ML_COMMON_AVAILABLE:
                self.lookahead_detector = get_global_detector()
                self.adaptive_thresholding = AdaptiveThresholding()
                _LOGGER.info("✅ ML utilities initialized")
            else:
                self.lookahead_detector = None
                self.adaptive_thresholding = None
        except Exception as e:
            _LOGGER.warning(f"⚠️ ML utilities initialization failed: {e}")
            self.lookahead_detector = None
            self.adaptive_thresholding = None

    def decompose_information(self, X: np.ndarray, y: np.ndarray,
                            feature_names: List[str]) -> PIDResult:
        """
        Perform partial information decomposition on the feature set.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names

        Returns:
            PIDResult containing decomposition analysis and suggested features
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting partial information decomposition...")
        _LOGGER.info(f"📊 Data shape: {X.shape}, Features: {len(feature_names)}")

        result = PIDResult()

        try:
            # Preprocess data with enhanced error handling
            _LOGGER.info("🔧 Preprocessing data for PID analysis...")
            try:
                X_processed, y_processed = self._preprocess_data(X, y)
                _LOGGER.info(f"✅ Data preprocessing completed: X{X_processed.shape}, y{y_processed.shape}")
            except Exception as preprocess_error:
                _LOGGER.error(f"❌ Data preprocessing failed: {preprocess_error}")
                raise

            # Calculate pairwise information measures with enhanced error handling
            _LOGGER.info("📊 Calculating pairwise information measures...")
            try:
                redundancy, synergy, unique_info = self._calculate_pairwise_pid(
                    X_processed, y_processed, feature_names
                )
                _LOGGER.info(f"✅ PID calculations completed: {len(redundancy)} pairs analyzed")
            except Exception as pid_error:
                _LOGGER.error(f"❌ PID calculations failed: {pid_error}")
                # Return empty results instead of crashing
                redundancy, synergy, unique_info = {}, {}, {}
                _LOGGER.warning("⚠️ Using empty PID results due to calculation failure")

            result.redundancy = redundancy
            result.synergy = synergy
            result.unique_info = unique_info

            # Detect significant interactions
            _LOGGER.info("🔍 Detecting significant interactions...")
            significant_pairs = self._detect_significant_interactions(
                synergy, redundancy, feature_names
            )

            # Create polynomial features
            _LOGGER.info("🔧 Creating polynomial features...")
            polynomial_features = self._create_polynomial_features(
                X_processed, feature_names, significant_pairs
            )

            # Create interaction features
            _LOGGER.info("🔧 Creating interaction features...")
            interaction_features = self._create_interaction_features(
                X_processed, feature_names, significant_pairs
            )

            # Analyze cross-timeframe dependencies
            _LOGGER.info("⏰ Analyzing cross-timeframe dependencies...")
            cross_timeframe_features = self._analyze_cross_timeframe_dependencies(
                X_processed, feature_names, significant_pairs
            )

            # Compile results
            result.polynomial_features = polynomial_features
            result.interaction_features = interaction_features
            result.cross_timeframe_features = cross_timeframe_features
            result.feature_pairs_analyzed = len(list(combinations(feature_names, 2)))
            result.significant_interactions = len(significant_pairs)
            result.significant_pairs = significant_pairs  # Store the actual pairs

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            _LOGGER.info(f"✅ PID analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Significant interactions found: {result.significant_interactions}")
            _LOGGER.info(f"🔧 Generated features: {len(polynomial_features)} polynomial, "
                        f"{len(interaction_features)} interaction, "
                        f"{len(cross_timeframe_features)} cross-timeframe")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ PID analysis failed: {e}")
            result.execution_time = time.time() - start_time
            return result

    def _preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for PID analysis with hardware optimizations."""
        try:
            # Debug input data
            _LOGGER.debug(f"🔍 Input data debug: X.shape={X.shape}, X.dtype={X.dtype}")
            _LOGGER.debug(f"🔍 Input target debug: y.shape={y.shape}, y.dtype={y.dtype}")

            # Memory optimization check
            if self.memory_optimizer:
                try:
                    memory_status = self.memory_optimizer.check_memory_status()
                    if memory_status.get('pressure', False):
                        _LOGGER.info("🧠 Memory pressure detected, using optimized preprocessing")
                except Exception as mem_error:
                    _LOGGER.warning(f"⚠️ Memory optimization check failed: {mem_error}")
        except Exception as debug_error:
            _LOGGER.error(f"❌ Initial debugging failed: {debug_error}")
            raise

        # Handle infinity and NaN values with math validation
        if MATH_VALIDATION_AVAILABLE:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            y_clean = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)

            # Validate finite values (check for any non-finite values)
            x_finite = np.isfinite(X_clean)
            y_finite = np.isfinite(y_clean)
            if not x_finite.all():
                _LOGGER.warning("⚠️ Non-finite values detected in X after cleaning")
            if not y_finite.all():
                _LOGGER.warning("⚠️ Non-finite values detected in y after cleaning")
        else:
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            y_clean = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)

        # Lookahead bias detection
        if self.lookahead_detector:
            try:
                validate_no_future_data(X_clean, y_clean)
                _LOGGER.debug("✅ No lookahead bias detected")
            except LookaheadBiasError as e:
                _LOGGER.warning(f"⚠️ Lookahead bias detected: {e}")

        # Standardize features with simple numpy operations (bypass problematic safe math)
        try:
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                _LOGGER.debug("✅ Used sklearn StandardScaler for feature scaling")
            else:
                # Use simple numpy operations
                X_scaled = (X_clean - np.mean(X_clean, axis=0)) / (np.std(X_clean, axis=0) + 1e-10)
                _LOGGER.debug("✅ Used numpy operations for feature scaling")
        except Exception as scaling_error:
            _LOGGER.error(f"❌ Feature scaling failed: {scaling_error}")
            # Use original data if scaling fails
            X_scaled = X_clean
            _LOGGER.warning("⚠️ Using unscaled data due to scaling failure")

        # Sample data if too large (with memory optimization)
        if self.config.sample_size and len(X_scaled) > self.config.sample_size:
            if self.memory_optimizer:
                # Use memory-optimized sampling
                optimal_sample_size = self.memory_optimizer.get_optimal_sample_size(
                    len(X_scaled), self.config.sample_size
                )
                sample_size = min(self.config.sample_size, optimal_sample_size)
            else:
                sample_size = self.config.sample_size

            np.random.seed(self.config.random_state)
            indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_scaled = X_scaled[indices]
            y_clean = y_clean[indices]
            _LOGGER.info(f"📊 Sampled data to {len(X_scaled)} samples")

        return X_scaled, y_clean

    def _calculate_pairwise_pid(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> Tuple[Dict, Dict, Dict]:
        """Calculate pairwise partial information decomposition."""
        redundancy = {}
        synergy = {}
        unique_info = {}

        n_features = len(feature_names)
        pairs_analyzed = 0

        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    feature_i, feature_j = feature_names[i], feature_names[j]

                    # Calculate mutual information with safe scalar conversion
                    mi_i_y = float(self._calculate_mutual_info(X[:, i], y))
                    mi_j_y = float(self._calculate_mutual_info(X[:, j], y))
                    mi_ij_y = float(self._calculate_mutual_info(np.column_stack([X[:, i], X[:, j]]), y))

                    # Calculate redundancy (minimum of individual MIs)
                    redundancy[(feature_i, feature_j)] = float(min(mi_i_y, mi_j_y))

                    # Calculate synergy (interaction information)
                    synergy_value = mi_ij_y - mi_i_y - mi_j_y + redundancy[(feature_i, feature_j)]
                    synergy[(feature_i, feature_j)] = float(synergy_value)

                    # Calculate unique information
                    unique_i = float(mi_i_y - redundancy[(feature_i, feature_j)])
                    unique_j = float(mi_j_y - redundancy[(feature_i, feature_j)])

                    unique_info[feature_i] = unique_info.get(feature_i, 0) + unique_i
                    unique_info[feature_j] = unique_info.get(feature_j, 0) + unique_j

                    pairs_analyzed += 1

                    if pairs_analyzed % 50 == 0:
                        _LOGGER.debug(f"📊 Analyzed {pairs_analyzed} feature pairs...")

                except Exception as e:
                    _LOGGER.warning(f"⚠️ Failed to analyze pair ({feature_names[i]}, {feature_names[j]}): {e}")
                    continue

        _LOGGER.info(f"📊 Completed pairwise PID analysis: {pairs_analyzed} pairs")
        return redundancy, synergy, unique_info

    def _detect_multi_horizon_target(self, y: np.ndarray, target_name: str = None) -> Dict[str, Any]:
        """Detect if target is from multi-horizon labeling and optimize accordingly."""
        target_analysis = {
            'is_multi_horizon': False,
            'is_directional': False,
            'is_probability': False,
            'target_type': 'unknown',
            'optimization_factor': 1.0
        }

        try:
            # Check if target name indicates multi-horizon origin
            if target_name:
                if any(keyword in target_name.lower() for keyword in ['long_', 'short_', 'opportunity', 'horizon', 'leverage']):
                    target_analysis['is_multi_horizon'] = True
                    _LOGGER.info(f"🎯 Detected multi-horizon target: {target_name}")

                if 'long_' in target_name.lower() or 'short_' in target_name.lower():
                    target_analysis['is_directional'] = True
                    target_analysis['optimization_factor'] = self.config.directional_synergy_boost
                    _LOGGER.info(f"🎯 Detected directional target with {target_analysis['optimization_factor']}x synergy boost")

            # Check if target values are probability-like (0-1 range with continuous values)
            y_min, y_max = np.min(y), np.max(y)
            unique_vals = len(np.unique(y))

            if 0 <= y_min and y_max <= 1 and unique_vals > 10:
                target_analysis['is_probability'] = True
                target_analysis['target_type'] = 'continuous_probability'
                _LOGGER.info(f"🎯 Detected probability target: range [{y_min:.3f}, {y_max:.3f}], {unique_vals} unique values")
            elif unique_vals <= 10:
                target_analysis['target_type'] = 'categorical'
            else:
                target_analysis['target_type'] = 'continuous'

            # Enable multi-horizon mode if detected
            if target_analysis['is_multi_horizon'] and not self.config.multi_horizon_mode:
                _LOGGER.info("🎯 Auto-enabling multi-horizon mode based on target detection")
                self.config.multi_horizon_mode = True

        except Exception as e:
            _LOGGER.warning(f"⚠️ Target analysis failed: {e}")

        return target_analysis

    def _calculate_mutual_info(self, X: np.ndarray, y: np.ndarray, target_name: str = None) -> float:
        """Calculate mutual information between features and target with multi-horizon optimizations."""
        try:
            # Analyze target characteristics for optimization
            if self.config.multi_horizon_mode or target_name:
                target_analysis = self._detect_multi_horizon_target(y, target_name)
            else:
                target_analysis = {'optimization_factor': 1.0, 'is_probability': False}

            if not SKLEARN_AVAILABLE:
                # Fallback to correlation-based approximation
                if X.ndim == 1:
                    base_mi = abs(safe_correlation(X, y))
                else:
                    # For multi-dimensional X, use average correlation
                    correlations = [abs(safe_correlation(X[:, i], y)) for i in range(X.shape[1])]
                    base_mi = safe_mean(correlations)

                # Apply multi-horizon optimization
                return base_mi * target_analysis.get('optimization_factor', 1.0)

            # Use sklearn for accurate MI calculation
            if X.ndim == 1:
                X_reshaped = X.reshape(-1, 1)
            else:
                X_reshaped = X

            # Determine if classification or regression with safe handling
            try:
                # Ensure y is a proper numpy array
                y_array = np.asarray(y).flatten()
                unique_vals = np.unique(y_array)
                unique_y = len(unique_vals)

                # For continuous targets (like multi-horizon probabilities), use regression
                # Use target analysis to determine best MI method
                if target_analysis.get('is_probability', False) or unique_y > 10:  # Regression for probabilities
                    _LOGGER.debug(f"Using regression MI for {'probability' if target_analysis.get('is_probability') else 'continuous'} target: {unique_y} unique values")
                    mi_result = mutual_info_regression(X_reshaped, y_array)
                else:  # Classification
                    _LOGGER.debug(f"Using classification MI: {unique_y} unique values")
                    mi_result = mutual_info_classif(X_reshaped, y_array)

                # Handle the result safely
                if hasattr(mi_result, '__len__') and len(mi_result) > 0:
                    base_mi = mi_result[0]
                else:
                    base_mi = mi_result

                # Apply multi-horizon optimization factor
                mi = base_mi * target_analysis.get('optimization_factor', 1.0)

                if target_analysis.get('optimization_factor', 1.0) != 1.0:
                    _LOGGER.debug(f"Applied {target_analysis['optimization_factor']}x optimization: {base_mi:.6f} -> {mi:.6f}")

            except Exception as unique_error:
                _LOGGER.warning(f"⚠️ Unique value calculation failed: {unique_error}")
                # Default to regression for continuous targets
                try:
                    y_array = np.asarray(y).flatten()
                    mi_result = mutual_info_regression(X_reshaped, y_array)
                    mi = mi_result[0] if hasattr(mi_result, '__len__') and len(mi_result) > 0 else mi_result
                except Exception as mi_error:
                    _LOGGER.warning(f"⚠️ Fallback MI calculation failed: {mi_error}")
                    return 0.0

            # Ensure we return a scalar float
            if hasattr(mi, '__len__'):
                mi = float(mi[0]) if len(mi) > 0 else 0.0
            else:
                mi = float(mi)

            return mi

        except Exception as e:
            _LOGGER.warning(f"⚠️ MI calculation failed: {e}")
            return 0.0

    def _detect_significant_interactions(self, synergy: Dict, redundancy: Dict,
                                       feature_names: List[str]) -> List[Tuple[str, str]]:
        """Detect significant feature interactions using dynamic thresholds."""
        if not synergy:
            _LOGGER.info("📊 No synergy data available")
            return []

        if self.config.use_dynamic_thresholds:
            return self._detect_interactions_dynamic_threshold(synergy, redundancy, feature_names)
        else:
            return self._detect_interactions_static_threshold(synergy, redundancy, feature_names)

    def _detect_interactions_dynamic_threshold(self, synergy: Dict, redundancy: Dict,
                                             feature_names: List[str]) -> List[Tuple[str, str]]:
        """Detect interactions using dynamic percentile-based thresholds."""
        # Calculate combined score for each pair (synergy + redundancy)
        pair_scores = {}
        for pair, syn_value in synergy.items():
            red_value = redundancy.get(pair, 0)
            # Combined score: prioritize synergy but also consider redundancy
            combined_score = syn_value + (red_value * 0.3)  # Weight redundancy at 30%
            pair_scores[pair] = {
                'combined_score': combined_score,
                'synergy': syn_value,
                'redundancy': red_value
            }

        # Sort pairs by combined score
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)

        # Calculate dynamic threshold based on percentile
        total_pairs = len(sorted_pairs)
        target_count = max(
            int(total_pairs * self.config.top_percentile),
            self.config.min_pairs_threshold
        )
        target_count = min(target_count, self.config.max_interaction_features, total_pairs)

        # Select top pairs
        significant_pairs = []
        selected_pairs = sorted_pairs[:target_count]

        for pair, scores in selected_pairs:
            significant_pairs.append(pair)

        # Debug information
        if selected_pairs:
            top_score = selected_pairs[0][1]['combined_score']
            bottom_score = selected_pairs[-1][1]['combined_score'] if len(selected_pairs) > 1 else top_score
            _LOGGER.info(f"📊 Dynamic threshold: selected top {len(significant_pairs)}/{total_pairs} pairs ({self.config.top_percentile*100:.1f}%)")
            _LOGGER.info(f"📊 Score range: {bottom_score:.6f} - {top_score:.6f}")

            # Show top 3 selected pairs
            for i, (pair, scores) in enumerate(selected_pairs[:3]):
                feat1, feat2 = pair
                _LOGGER.info(f"🔍 #{i+1}: {feat1} & {feat2} = {scores['combined_score']:.6f} (syn:{scores['synergy']:.6f}, red:{scores['redundancy']:.6f})")

        _LOGGER.info(f"📊 Found {len(significant_pairs)} significant interactions using dynamic thresholds")
        return significant_pairs

    def _detect_interactions_static_threshold(self, synergy: Dict, redundancy: Dict,
                                            feature_names: List[str]) -> List[Tuple[str, str]]:
        """Detect interactions using static absolute thresholds (fallback method)."""
        significant_pairs = []

        for (feat1, feat2), syn_value in synergy.items():
            red_value = redundancy.get((feat1, feat2), 0)

            # Check synergy threshold
            if syn_value > self.config.synergy_threshold:
                significant_pairs.append((feat1, feat2))
                _LOGGER.debug(f"🔍 Significant synergy: {feat1} & {feat2} = {syn_value:.4f}")

            # Check redundancy threshold
            elif red_value > self.config.redundancy_threshold:
                _LOGGER.debug(f"🔍 High redundancy: {feat1} & {feat2} = {red_value:.4f}")

        # Sort by synergy value
        significant_pairs.sort(key=lambda x: synergy.get(x, 0), reverse=True)

        # Limit number of interactions
        max_interactions = min(len(significant_pairs), self.config.max_interaction_features)
        significant_pairs = significant_pairs[:max_interactions]

        # Debug: Show actual values vs thresholds
        if len(synergy) > 0:
            max_synergy = max(synergy.values())
            max_redundancy = max(redundancy.values()) if redundancy else 0
            _LOGGER.info(f"📊 Max synergy found: {max_synergy:.6f} (threshold: {self.config.synergy_threshold})")
            _LOGGER.info(f"📊 Max redundancy found: {max_redundancy:.6f} (threshold: {self.config.redundancy_threshold})")

        _LOGGER.info(f"📊 Found {len(significant_pairs)} significant interactions using static thresholds")
        return significant_pairs

    def _create_polynomial_features(self, X: np.ndarray, feature_names: List[str],
                                  significant_pairs: List[Tuple[str, str]]) -> List[str]:
        """Create polynomial features based on significant interactions."""
        polynomial_features = []

        # Create polynomial features for significant pairs
        for feat1, feat2 in significant_pairs:
            try:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)

                x1, x2 = X[:, idx1], X[:, idx2]

                # Create polynomial features up to max degree
                for degree in range(2, self.config.max_polynomial_degree + 1):
                    # x1^degree
                    poly_feat_name = f"{feat1}_pow_{degree}"
                    polynomial_features.append(poly_feat_name)

                    # x2^degree
                    poly_feat_name = f"{feat2}_pow_{degree}"
                    polynomial_features.append(poly_feat_name)

                    # x1 * x2^(degree-1)
                    poly_feat_name = f"{feat1}_x_{feat2}_pow_{degree-1}"
                    polynomial_features.append(poly_feat_name)

                    # x2 * x1^(degree-1)
                    poly_feat_name = f"{feat2}_x_{feat1}_pow_{degree-1}"
                    polynomial_features.append(poly_feat_name)

            except ValueError:
                continue

        # Remove duplicates and limit
        polynomial_features = list(set(polynomial_features))
        polynomial_features = polynomial_features[:self.config.max_interaction_features]

        _LOGGER.info(f"🔧 Generated {len(polynomial_features)} polynomial feature names")
        return polynomial_features

    def _create_interaction_features(self, X: np.ndarray, feature_names: List[str],
                                   significant_pairs: List[Tuple[str, str]]) -> List[str]:
        """Create interaction features based on significant pairs."""
        interaction_features = []

        for feat1, feat2 in significant_pairs:
            try:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)

                x1, x2 = X[:, idx1], X[:, idx2]

                # Basic interactions
                interaction_features.extend([
                    f"{feat1}_x_{feat2}",
                    f"{feat1}_plus_{feat2}",
                    f"{feat1}_minus_{feat2}",
                    f"{feat1}_ratio_{feat2}",
                    f"sqrt_{feat1}_x_{feat2}",
                    f"log_{feat1}_x_{feat2}"
                ])

                # Statistical interactions
                interaction_features.extend([
                    f"{feat1}_x_{feat2}_norm",
                    f"{feat1}_x_{feat2}_std",
                    f"{feat1}_rank_x_{feat2}_rank"
                ])

            except ValueError:
                continue

        # Remove duplicates and limit
        interaction_features = list(set(interaction_features))
        interaction_features = interaction_features[:self.config.max_interaction_features]

        _LOGGER.info(f"🔧 Generated {len(interaction_features)} interaction feature names")
        return interaction_features

    def _analyze_cross_timeframe_dependencies(self, X: np.ndarray, feature_names: List[str],
                                            significant_pairs: List[Tuple[str, str]]) -> List[str]:
        """Analyze cross-timeframe dependencies and create relevant features."""
        cross_timeframe_features = []

        # Look for timeframe-related features
        timeframe_features = [f for f in feature_names if any(tf in f.lower() for tf in
                           ['1m', '5m', '15m', '1h', '4h', '1d', '1w', 'timeframe', 'tf'])]

        if len(timeframe_features) < 2:
            _LOGGER.info("📊 No timeframe features detected for cross-timeframe analysis")
            return cross_timeframe_features

        # Create cross-timeframe features
        for i, feat1 in enumerate(timeframe_features):
            for feat2 in timeframe_features[i+1:]:
                # Extract timeframes
                tf1 = self._extract_timeframe(feat1)
                tf2 = self._extract_timeframe(feat2)

                if tf1 and tf2 and tf1 != tf2:
                    base_name1 = self._remove_timeframe_from_name(feat1)
                    base_name2 = self._remove_timeframe_from_name(feat2)

                    if base_name1 == base_name2:  # Same feature, different timeframes
                        cross_timeframe_features.extend([
                            f"{base_name1}_{tf1}_to_{tf2}_ratio",
                            f"{base_name1}_{tf1}_to_{tf2}_diff",
                            f"{base_name1}_{tf1}_to_{tf2}_corr",
                            f"{base_name1}_{tf1}_x_{tf2}",
                            f"{base_name1}_{tf1}_plus_{tf2}"
                        ])

        # Create lag-based features for significant pairs
        for feat1, feat2 in significant_pairs[:10]:  # Limit to top 10
            for lag in range(1, min(self.config.max_timeframe_lag + 1, 4)):
                cross_timeframe_features.extend([
                    f"{feat1}_lag_{lag}_x_{feat2}",
                    f"{feat2}_lag_{lag}_x_{feat1}",
                    f"{feat1}_x_{feat2}_lag_{lag}"
                ])

        # Remove duplicates and limit
        cross_timeframe_features = list(set(cross_timeframe_features))
        cross_timeframe_features = cross_timeframe_features[:self.config.max_interaction_features]

        _LOGGER.info(f"🔧 Generated {len(cross_timeframe_features)} cross-timeframe feature names")
        return cross_timeframe_features

    def _extract_timeframe(self, feature_name: str) -> Optional[str]:
        """Extract timeframe from feature name."""
        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        for tf in timeframes:
            if tf in feature_name.lower():
                return tf
        return None

    def _remove_timeframe_from_name(self, feature_name: str) -> str:
        """Remove timeframe suffix from feature name."""
        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        for tf in timeframes:
            if feature_name.lower().endswith(f"_{tf}"):
                return feature_name[:-len(f"_{tf}")]
        return feature_name

    def generate_feature_matrix(self, X: np.ndarray, feature_names: List[str],
                              pid_result: PIDResult) -> Tuple[np.ndarray, List[str]]:
        """
        Generate expanded feature matrix with polynomial, interaction, and cross-timeframe features.

        Args:
            X: Original feature matrix
            feature_names: Original feature names
            pid_result: PID analysis result

        Returns:
            Tuple of (expanded_feature_matrix, expanded_feature_names)
        """
        _LOGGER.info("🔧 Generating expanded feature matrix...")

        # Start with original features
        expanded_features = [X]
        expanded_names = feature_names.copy()

        try:
            # Add polynomial features
            if pid_result.polynomial_features:
                poly_matrix = self._create_polynomial_matrix(X, feature_names, pid_result.polynomial_features)
                if poly_matrix is not None:
                    expanded_features.append(poly_matrix)
                    expanded_names.extend(pid_result.polynomial_features)

            # Add interaction features
            if pid_result.interaction_features:
                interaction_matrix = self._create_interaction_matrix(X, feature_names, pid_result.interaction_features)
                if interaction_matrix is not None:
                    expanded_features.append(interaction_matrix)
                    expanded_names.extend(pid_result.interaction_features)

            # Add cross-timeframe features
            if pid_result.cross_timeframe_features:
                cross_tf_matrix = self._create_cross_timeframe_matrix(X, feature_names, pid_result.cross_timeframe_features)
                if cross_tf_matrix is not None:
                    expanded_features.append(cross_tf_matrix)
                    expanded_names.extend(pid_result.cross_timeframe_features)

            # Combine all features
            if len(expanded_features) > 1:
                expanded_X = np.column_stack(expanded_features)
            else:
                expanded_X = X

            _LOGGER.info(f"✅ Expanded feature matrix: {X.shape} → {expanded_X.shape}")
            _LOGGER.info(f"📊 Total features: {len(expanded_names)}")

            return expanded_X, expanded_names

        except Exception as e:
            _LOGGER.error(f"❌ Feature matrix generation failed: {e}")
            return X, feature_names

    def _create_polynomial_matrix(self, X: np.ndarray, feature_names: List[str],
                                polynomial_features: List[str]) -> Optional[np.ndarray]:
        """Create polynomial feature matrix."""
        try:
            poly_features = []

            for poly_name in polynomial_features:
                try:
                    # Parse polynomial feature name
                    if '_pow_' in poly_name:
                        # Single feature power: feature_pow_degree
                        feat_name, degree_str = poly_name.split('_pow_')
                        degree = int(degree_str)

                        if feat_name in feature_names:
                            idx = feature_names.index(feat_name)
                            poly_feat = np.power(X[:, idx], degree)
                            poly_features.append(poly_feat)

                    elif '_x_' in poly_name and '_pow_' in poly_name:
                        # Cross feature with power: feat1_x_feat2_pow_degree
                        parts = poly_name.split('_x_')
                        if len(parts) == 2:
                            feat1 = parts[0]
                            feat2_pow = parts[1]

                            if '_pow_' in feat2_pow:
                                feat2, degree_str = feat2_pow.split('_pow_')
                                degree = int(degree_str)

                                if feat1 in feature_names and feat2 in feature_names:
                                    idx1 = feature_names.index(feat1)
                                    idx2 = feature_names.index(feat2)
                                    poly_feat = X[:, idx1] * np.power(X[:, idx2], degree)
                                    poly_features.append(poly_feat)

                except (ValueError, IndexError) as e:
                    _LOGGER.warning(f"⚠️ Failed to create polynomial feature {poly_name}: {e}")
                    continue

            if poly_features:
                return np.column_stack(poly_features)
            else:
                return None

        except Exception as e:
            _LOGGER.error(f"❌ Polynomial matrix creation failed: {e}")
            return None

    def _create_interaction_matrix(self, X: np.ndarray, feature_names: List[str],
                                 interaction_features: List[str]) -> Optional[np.ndarray]:
        """Create interaction feature matrix."""
        try:
            interaction_feats = []

            for interaction_name in interaction_features:
                try:
                    if '_x_' in interaction_name:
                        # Basic multiplication interaction
                        feat1, feat2 = interaction_name.split('_x_', 1)

                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = X[:, idx1] * X[:, idx2]
                            interaction_feats.append(interaction_feat)

                    elif '_plus_' in interaction_name:
                        # Addition interaction
                        feat1, feat2 = interaction_name.split('_plus_', 1)

                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = X[:, idx1] + X[:, idx2]
                            interaction_feats.append(interaction_feat)

                    elif '_minus_' in interaction_name:
                        # Subtraction interaction
                        feat1, feat2 = interaction_name.split('_minus_', 1)

                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = X[:, idx1] - X[:, idx2]
                            interaction_feats.append(interaction_feat)

                    elif '_ratio_' in interaction_name:
                        # Ratio interaction
                        feat1, feat2 = interaction_name.split('_ratio_', 1)

                        if feat1 in feature_names and feat2 in feature_names:
                            idx1 = feature_names.index(feat1)
                            idx2 = feature_names.index(feat2)
                            interaction_feat = safe_divide(X[:, idx1], X[:, idx2])
                            interaction_feats.append(interaction_feat)

                except (ValueError, IndexError) as e:
                    _LOGGER.warning(f"⚠️ Failed to create interaction feature {interaction_name}: {e}")
                    continue

            if interaction_feats:
                return np.column_stack(interaction_feats)
            else:
                return None

        except Exception as e:
            _LOGGER.error(f"❌ Interaction matrix creation failed: {e}")
            return None

    def _create_cross_timeframe_matrix(self, X: np.ndarray, feature_names: List[str],
                                     cross_timeframe_features: List[str]) -> Optional[np.ndarray]:
        """Create cross-timeframe feature matrix."""
        try:
            cross_tf_feats = []

            for cross_tf_name in cross_timeframe_features:
                try:
                    if '_to_' in cross_tf_name and '_ratio' in cross_tf_name:
                        # Ratio between timeframes
                        parts = cross_tf_name.split('_to_')
                        if len(parts) == 2:
                            feat1_part = parts[0]
                            feat2_part = parts[1].replace('_ratio', '')

                            # Find matching features
                            matching_feats = self._find_matching_features(feat1_part, feat2_part, feature_names)
                            if len(matching_feats) == 2:
                                idx1 = feature_names.index(matching_feats[0])
                                idx2 = feature_names.index(matching_feats[1])
                                cross_tf_feat = safe_divide(X[:, idx1], X[:, idx2])
                                cross_tf_feats.append(cross_tf_feat)

                    elif '_lag_' in cross_tf_name and '_x_' in cross_tf_name:
                        # Lag-based interaction
                        parts = cross_tf_name.split('_x_')
                        if len(parts) == 2:
                            lag_part, feat2 = parts[0], parts[1]

                            if '_lag_' in lag_part:
                                feat1, lag_str = lag_part.split('_lag_')
                                lag = int(lag_str)

                                if feat1 in feature_names and feat2 in feature_names:
                                    idx1 = feature_names.index(feat1)
                                    idx2 = feature_names.index(feat2)

                                    # Create lagged version (simple shift)
                                    if lag < X.shape[0]:
                                        lagged_feat1 = np.roll(X[:, idx1], lag)
                                        lagged_feat1[:lag] = 0  # Zero out the rolled values
                                        cross_tf_feat = lagged_feat1 * X[:, idx2]
                                        cross_tf_feats.append(cross_tf_feat)

                except (ValueError, IndexError) as e:
                    _LOGGER.warning(f"⚠️ Failed to create cross-timeframe feature {cross_tf_name}: {e}")
                    continue

            if cross_tf_feats:
                return np.column_stack(cross_tf_feats)
            else:
                return None

        except Exception as e:
            _LOGGER.error(f"❌ Cross-timeframe matrix creation failed: {e}")
            return None

    def _find_matching_features(self, feat1_part: str, feat2_part: str,
                              feature_names: List[str]) -> List[str]:
        """Find features matching the given patterns."""
        matching = []

        for feat_name in feature_names:
            if feat1_part in feat_name:
                matching.append(feat_name)
                break

        for feat_name in feature_names:
            if feat2_part in feat_name and feat_name not in matching:
                matching.append(feat_name)
                break

        return matching

    def get_feature_importance_scores(self, pid_result: PIDResult) -> Dict[str, float]:
        """Get feature importance scores based on PID analysis."""
        importance_scores = {}

        # Unique information scores
        for feature, unique_score in pid_result.unique_info.items():
            importance_scores[feature] = unique_score

        # Synergy scores (average for each feature)
        synergy_scores = {}
        for (feat1, feat2), syn_score in pid_result.synergy.items():
            synergy_scores[feat1] = synergy_scores.get(feat1, 0) + syn_score
            synergy_scores[feat2] = synergy_scores.get(feat2, 0) + syn_score

        # Normalize synergy scores
        for feature in synergy_scores:
            synergy_scores[feature] /= max(1, len([p for p in pid_result.synergy.keys() if feature in p]))
            importance_scores[feature] = importance_scores.get(feature, 0) + synergy_scores[feature]

        return importance_scores

    def save_analysis_results(self, pid_result: PIDResult, output_path: str = None):
        """Save PID analysis results to file with datetime in filename."""
        try:

            # Generate filename with datetime if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"pid_analysis_results_{timestamp}.json"

            # Ensure output directory exists
            if COMMON_OPERATIONS_AVAILABLE:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    create_directory_safe(output_dir)

            # Convert results to serializable format
            results_dict = {
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': pid_result.execution_time,
                    'feature_pairs_analyzed': pid_result.feature_pairs_analyzed,
                    'significant_interactions': pid_result.significant_interactions,
                    'config_used': {
                        'synergy_threshold': self.config.synergy_threshold,
                        'redundancy_threshold': self.config.redundancy_threshold,
                        'unique_info_threshold': self.config.unique_info_threshold,
                        'max_polynomial_degree': self.config.max_polynomial_degree,
                        'max_interaction_features': self.config.max_interaction_features,
                        'cross_timeframe_threshold': self.config.cross_timeframe_threshold
                    }
                },
                'information_measures': {
                    'redundancy': {f"{k[0]}_{k[1]}": v for k, v in pid_result.redundancy.items()},
                    'synergy': {f"{k[0]}_{k[1]}": v for k, v in pid_result.synergy.items()},
                    'unique_info': pid_result.unique_info
                },
                'generated_features': {
                    'polynomial_features': pid_result.polynomial_features,
                    'interaction_features': pid_result.interaction_features,
                    'cross_timeframe_features': pid_result.cross_timeframe_features,
                    'total_generated': (len(pid_result.polynomial_features) +
                                      len(pid_result.interaction_features) +
                                      len(pid_result.cross_timeframe_features))
                },
                'convergence_info': pid_result.convergence_info
            }

            # Use serialization utilities if available
            if SERIALIZATION_AVAILABLE:
                success = JSONSerializer.save(results_dict, output_path)
                if success:
                    _LOGGER.info(f"💾 PID analysis results saved to {output_path}")
                    return output_path
                else:
                    _LOGGER.error(f"❌ Failed to save PID results using JSONSerializer")
                    return None
            else:
                # Fallback to manual JSON writing
                import json
                with open(output_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                _LOGGER.info(f"💾 PID analysis results saved to {output_path}")
                return output_path

        except Exception as e:
            _LOGGER.error(f"❌ Failed to save PID results: {e}")
            return None

    def save_feature_matrix_artifact(self, X: np.ndarray, feature_names: List[str],
                                   pid_result: PIDResult, output_path: str = None):
        """Save expanded feature matrix as artifact with datetime."""
        try:

            # Generate expanded feature matrix
            expanded_X, expanded_names = self.generate_feature_matrix(X, feature_names, pid_result)

            # Generate filename with datetime if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"expanded_feature_matrix_{timestamp}.parquet"

            # Ensure output directory exists
            if COMMON_OPERATIONS_AVAILABLE:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    create_directory_safe(output_dir)

            # Create DataFrame with feature names
            df = pd.DataFrame(expanded_X, columns=expanded_names)

            # Add metadata
            df.attrs = {
                'original_features': len(feature_names),
                'expanded_features': len(expanded_names),
                'polynomial_features': len(pid_result.polynomial_features),
                'interaction_features': len(pid_result.interaction_features),
                'cross_timeframe_features': len(pid_result.cross_timeframe_features),
                'significant_interactions': pid_result.significant_interactions,
                'creation_timestamp': datetime.now().isoformat(),
                'pid_config': {
                    'synergy_threshold': self.config.synergy_threshold,
                    'redundancy_threshold': self.config.redundancy_threshold,
                    'max_polynomial_degree': self.config.max_polynomial_degree,
                    'max_interaction_features': self.config.max_interaction_features
                }
            }

            # Save as parquet with validation
            if PARQUET_AVAILABLE:
                parquet_utils = ParquetUtils()

                # Validate before saving
                validation_result = parquet_utils.validate_dataframe_for_parquet(df)
                if validation_result.get('valid', True):
                    df.to_parquet(output_path, index=False)

                    # Validate saved file
                    saved_validation = parquet_utils.validate_parquet_file(output_path)
                    if saved_validation.get('valid', False):
                        _LOGGER.info(f"💾 Expanded feature matrix saved and validated: {output_path}")
                    else:
                        _LOGGER.warning(f"⚠️ Saved parquet file validation failed: {saved_validation.get('error')}")
                else:
                    _LOGGER.warning(f"⚠️ DataFrame validation failed: {validation_result.get('error')}")
                    # Save anyway as fallback
                    df.to_parquet(output_path, index=False)
            else:
                # Fallback to direct parquet saving
                df.to_parquet(output_path, index=False)

            _LOGGER.info(f"📊 Matrix shape: {X.shape} → {expanded_X.shape}")
            _LOGGER.info(f"🔧 Generated features: {len(expanded_names) - len(feature_names)} new features")

            return output_path

        except Exception as e:
            _LOGGER.error(f"❌ Failed to save feature matrix artifact: {e}")
            return None

    def create_comprehensive_artifact(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str], pid_result: PIDResult,
                                    output_dir: str = "pid_artifacts") -> Dict[str, str]:
        """Create comprehensive artifacts with datetime in filenames."""
        try:

            # Create output directory if it doesn't exist
            if COMMON_OPERATIONS_AVAILABLE:
                create_directory_safe(output_dir)
            else:
                os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            artifacts = {}

            # 1. Save PID analysis results
            analysis_path = os.path.join(output_dir, f"pid_analysis_{timestamp}.json")
            artifacts['analysis_results'] = self.save_analysis_results(pid_result, analysis_path)

            # 2. Save expanded feature matrix
            matrix_path = os.path.join(output_dir, f"expanded_features_{timestamp}.parquet")
            artifacts['feature_matrix'] = self.save_feature_matrix_artifact(X, feature_names, pid_result, matrix_path)

            # 3. Save feature importance scores
            importance_scores = self.get_feature_importance_scores(pid_result)
            importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.json")

            importance_data = {
                'timestamp': datetime.now().isoformat(),
                'feature_importance_scores': importance_scores,
                'top_features': sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:20],
                'pid_config': {
                    'synergy_threshold': self.config.synergy_threshold,
                    'redundancy_threshold': self.config.redundancy_threshold,
                    'unique_info_threshold': self.config.unique_info_threshold
                }
            }

            # Use serialization utilities if available
            if SERIALIZATION_AVAILABLE:
                success = JSONSerializer.save(importance_data, importance_path)
                if success:
                    artifacts['feature_importance'] = importance_path
                else:
                    _LOGGER.warning("⚠️ Failed to save feature importance using JSONSerializer")
            else:
                with open(importance_path, 'w') as f:
                    json.dump(importance_data, f, indent=2)
                artifacts['feature_importance'] = importance_path

            # 4. Save interaction summary
            summary_path = os.path.join(output_dir, f"interaction_summary_{timestamp}.json")
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'significant_interactions': [
                    {
                        'features': f"{pair[0]}_{pair[1]}",
                        'synergy_score': score,
                        'redundancy_score': pid_result.redundancy.get(pair, 0)
                    }
                    for pair, score in sorted(pid_result.synergy.items(), key=lambda x: x[1], reverse=True)
                    if score > self.config.synergy_threshold
                ],
                'feature_generation_summary': {
                    'polynomial_features_count': len(pid_result.polynomial_features),
                    'interaction_features_count': len(pid_result.interaction_features),
                    'cross_timeframe_features_count': len(pid_result.cross_timeframe_features),
                    'total_new_features': (len(pid_result.polynomial_features) +
                                         len(pid_result.interaction_features) +
                                         len(pid_result.cross_timeframe_features))
                }
            }

            # Use serialization utilities if available
            if SERIALIZATION_AVAILABLE:
                success = JSONSerializer.save(summary_data, summary_path)
                if success:
                    artifacts['interaction_summary'] = summary_path
                else:
                    _LOGGER.warning("⚠️ Failed to save interaction summary using JSONSerializer")
            else:
                with open(summary_path, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                artifacts['interaction_summary'] = summary_path

            _LOGGER.info(f"🎉 Comprehensive PID artifacts created in {output_dir}")
            _LOGGER.info(f"📁 Generated {len(artifacts)} artifact files with timestamp {timestamp}")

            return artifacts

        except Exception as e:
            _LOGGER.error(f"❌ Failed to create comprehensive artifacts: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics using hardware optimization utilities."""
        metrics = {
            'hardware_optimizations': {
                'gpu_available': self.gpu_manager is not None,
                'memory_optimizer_available': self.memory_optimizer is not None,
                'cpu_optimizer_available': self.cpu_optimizer is not None,
                'matrix_ops_available': self.matrix_ops is not None
            },
            'utility_availability': {
                'math_validation': MATH_VALIDATION_AVAILABLE,
                'common_operations': COMMON_OPERATIONS_AVAILABLE,
                'serialization': SERIALIZATION_AVAILABLE,
                'parquet': PARQUET_AVAILABLE,
                'ml_common': ML_COMMON_AVAILABLE
            }
        }

        # Get memory status if available
        if self.memory_optimizer:
            try:
                memory_status = self.memory_optimizer.check_memory_status()
                metrics['memory_status'] = memory_status
            except Exception as e:
                _LOGGER.warning(f"⚠️ Failed to get memory status: {e}")

        # Get GPU status if available
        if self.gpu_manager:
            try:
                gpu_status = self.gpu_manager.get_gpu_status()
                metrics['gpu_status'] = gpu_status
            except Exception as e:
                _LOGGER.warning(f"⚠️ Failed to get GPU status: {e}")

        return metrics

    def calculate_interaction_scores(self, pid_result: PIDResult, feature_names: List[str]) -> Dict[str, float]:
        """Calculate interaction scores for each feature based on PID analysis."""
        try:
            interaction_scores = {}

            # Initialize all features with 0 score
            for feature in feature_names:
                interaction_scores[feature] = 0.0

            # Calculate scores based on redundancy, synergy, and unique information
            feature_redundancy_scores = {}
            feature_synergy_scores = {}
            feature_unique_scores = {}

            # Process redundancy scores (lower is better - less redundant)
            for (feat1, feat2), redundancy in pid_result.redundancy.items():
                if feat1 in feature_names:
                    if feat1 not in feature_redundancy_scores:
                        feature_redundancy_scores[feat1] = []
                    feature_redundancy_scores[feat1].append(redundancy)

                if feat2 in feature_names:
                    if feat2 not in feature_redundancy_scores:
                        feature_redundancy_scores[feat2] = []
                    feature_redundancy_scores[feat2].append(redundancy)

            # Process synergy scores (higher is better - more synergistic)
            for (feat1, feat2), synergy in pid_result.synergy.items():
                if feat1 in feature_names:
                    if feat1 not in feature_synergy_scores:
                        feature_synergy_scores[feat1] = []
                    feature_synergy_scores[feat1].append(synergy)

                if feat2 in feature_names:
                    if feat2 not in feature_synergy_scores:
                        feature_synergy_scores[feat2] = []
                    feature_synergy_scores[feat2].append(synergy)

            # Process unique information scores (higher is better - more unique)
            for feature, unique_info in pid_result.unique_info.items():
                if feature in feature_names:
                    feature_unique_scores[feature] = unique_info

            # Calculate final interaction scores for each feature
            for feature in feature_names:
                # Redundancy component (inverted - lower redundancy is better)
                if feature in feature_redundancy_scores:
                    avg_redundancy = np.mean(feature_redundancy_scores[feature])
                    redundancy_score = 1.0 - min(1.0, avg_redundancy)  # Invert so lower redundancy = higher score
                else:
                    redundancy_score = 1.0  # No redundancy data = perfect score

                # Synergy component (higher synergy is better)
                if feature in feature_synergy_scores:
                    avg_synergy = np.mean(feature_synergy_scores[feature])
                    synergy_score = min(1.0, avg_synergy)
                else:
                    synergy_score = 0.0  # No synergy data = neutral score

                # Unique information component (higher unique info is better)
                if feature in feature_unique_scores:
                    unique_score = min(1.0, feature_unique_scores[feature])
                else:
                    unique_score = 0.0  # No unique info data = neutral score

                # Combine components with weights
                # Weight: redundancy (40%), synergy (35%), unique info (25%)
                interaction_score = (
                    redundancy_score * 0.4 +
                    synergy_score * 0.35 +
                    unique_score * 0.25
                )

                interaction_scores[feature] = interaction_score

            # Normalize scores to 0-1 range
            if interaction_scores:
                max_score = max(interaction_scores.values())
                min_score = min(interaction_scores.values())
                if max_score > min_score:
                    for feature in interaction_scores:
                        interaction_scores[feature] = (interaction_scores[feature] - min_score) / (max_score - min_score)

            _LOGGER.info(f"📊 Calculated interaction scores for {len(interaction_scores)} features")
            return interaction_scores

        except Exception as e:
            _LOGGER.error(f"❌ Interaction score calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}
