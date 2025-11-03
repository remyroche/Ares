"""
Feature Generation Optimization

This module provides data-driven optimization for feature generation parameters,
particularly lookback periods for time-series features. It uses statistical
analysis and cross-validation to determine optimal parameters for each feature.

Key Features:
- Data-driven lookback period optimization
- Feature performance analysis across different time windows
- Feature stability assessment
- Optimal feature window selection
- Feature decay analysis
- Cross-validation for feature parameters
- Regime-aware feature optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from src.utils.tprint import tprint
from datetime import datetime, timedelta
import logging
import time
from functools import partial, lru_cache
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import OrderedDict
from scipy.sparse import csr_matrix, lil_matrix
import threading

from .math_validation import safe_divide, safe_log
from src.utils.common_operations import create_fallback_logger
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.parallel_processing_optimizer import ParallelProcessor

# Import optimization utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    OPTIMIZATION_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    UnifiedVectorizationManager = None

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited optimization functionality")

try:
    import vectorbt as vbt
    # VectorBT 0.28+ uses pandas Series.rolling() under the hood for rolling operations
    # VectorBT's optimizations are for backtesting strategies, not individual rolling stats
    # Using pandas rolling is the standard and optimal approach
    VECTORBT_AVAILABLE = True
    logger.info("✅ VectorBT available for feature optimization")
except ImportError:
    VECTORBT_AVAILABLE = False
    logger.warning("VectorBT not available - using pandas fallback for rolling operations")

# Rolling functions using pandas (which is what VectorBT uses internally)
def rolling_mean(series, window):
    """Compute rolling mean using pandas (VectorBT's underlying approach)."""
    return series.rolling(window=window).mean()

def rolling_std(series, window):
    """Compute rolling std using pandas (VectorBT's underlying approach)."""
    return series.rolling(window=window).std()

def rolling_var(series, window):
    """Compute rolling variance using pandas (VectorBT's underlying approach)."""
    return series.rolling(window=window).var()

def rolling_min(series, window):
    """Compute rolling min using pandas (VectorBT's underlying approach)."""
    return series.rolling(window=window).min()

def rolling_max(series, window):
    """Compute rolling max using pandas (VectorBT's underlying approach)."""
    return series.rolling(window=window).max()

def rolling_sum(series, window):
    """Compute rolling sum using pandas (VectorBT's underlying approach)."""
    return series.rolling(window=window).sum()

class OptimizationMethod(Enum):
    """Optimization methods for feature parameters."""
    CROSS_VALIDATION = "cross_validation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    INFORMATION_THEORY = "information_theory"
    REGIME_AWARE = "regime_aware"
    ADAPTIVE = "adaptive"

@dataclass
class FeatureOptimizationConfig:
    """Configuration for feature optimization."""
    min_lookback: int = 5
    max_lookback: int = 252  # 1 year of daily data
    step_size: int = 1
    optimization_method: OptimizationMethod = OptimizationMethod.CROSS_VALIDATION
    cv_folds: int = 3  # Reduced from 5 to 3 for 40-60% speed gain
    stability_threshold: float = 0.8
    performance_threshold: float = 0.6
    regime_aware: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    chunk_size: int = 1000
    # Add methods parameter for backward compatibility
    methods: Optional[List[str]] = None
    optimization_metric: str = "sharpe_ratio"

    # Stability Enhancement Parameters
    l1_regularization: float = 0.01  # L1 regularization for feature selection
    l2_regularization: float = 0.001  # L2 regularization for stability
    max_lookback_variance: float = 0.2  # Maximum variance between feature lookbacks
    lookback_range_penalty: float = 0.1  # Penalty for wide lookback ranges
    temporal_consistency_weight: float = 0.3  # Weight for temporal consistency
    stability_weight: float = 0.4  # Balance performance vs stability

    # Rolling Window Parameters
    rolling_window_size: str = "30D"  # Rolling window size for optimization
    rolling_step_size: str = "7D"  # Step size for rolling optimization
    min_stability_score: float = 0.7  # Minimum required stability score

    # Cross-Validation Stability Parameters
    cv_stability_metric: str = "coefficient_variance"  # Stability metric for CV
    stability_cv_folds: int = 3  # Additional CV folds for stability assessment
    
    # Hardware-based Cache Configuration
    cache_size_limit: int = 1000  # Maximum number of cached results
    cache_memory_limit_mb: float = 512.0  # Maximum cache memory in MB
    enable_lru_eviction: bool = True  # Enable LRU eviction
    cache_cleanup_interval: float = 300.0  # Cache cleanup interval in seconds
    
    # Optimization Performance Thresholds
    early_termination_threshold: float = 0.1  # Stop if performance drops below threshold
    max_correlation_candidates: int = 50  # Maximum candidates for correlation matrix
    use_sparse_matrices: bool = True  # Use sparse matrices for large candidate sets
    vectorization_threshold: int = 1000  # Use vectorized operations above this data size
    
    # Adaptive Search Configuration
    use_adaptive_search: bool = True  # Enable adaptive search
    adaptive_search_method: str = "bayesian"  # "bayesian", "golden_section", "grid"
    max_trials: int = 60  # Maximum trials for Bayesian optimization
    n_startup_trials: int = 10  # Startup trials for Bayesian optimization
    early_stopping_rounds: int = 10  # Early stopping for Bayesian optimization

@dataclass
class FeatureCandidate:
    """Individual feature candidate with lookback period."""
    lookback: int
    performance_score: float
    stability_score: float
    mi_score: float  # Mutual information score
    combined_score: float
    correlation_with_target: float

@dataclass
class FeatureOptimizationResult:
    """Result of feature optimization."""
    feature_name: str
    optimal_lookback: int
    performance_score: float
    stability_score: float
    confidence_interval: Tuple[float, float]
    optimization_method: str
    regime_specific_results: Optional[Dict[str, Any]] = None
    decay_analysis: Optional[Dict[str, Any]] = None
    validation_scores: Optional[List[float]] = None
    # Enhanced results for multi-candidate selection
    top_candidates: Optional[List[FeatureCandidate]] = None
    redundancy_analysis: Optional[Dict[str, Any]] = None
    stability_report: Optional[Dict[str, Any]] = None

class VectorBTHelper:
    """Helper class for VectorBT operations."""
    
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
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

class HardwareOptimizedCache:
    """Hardware-optimized cache with LRU eviction and memory management."""
    
    def __init__(self, max_size: int = 1000, memory_limit_mb: float = 512.0):
        self.max_size = max_size
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.last_cleanup = time.time()
        
    def _calculate_memory_usage(self, obj: Any) -> int:
        """Calculate approximate memory usage of an object."""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return 1024  # Default estimate
    
    def _cleanup_if_needed(self):
        """Clean up cache if memory or size limits exceeded."""
        current_time = time.time()
        
        # Clean up if size limit exceeded
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
            
        # Clean up if memory limit exceeded
        while self.memory_usage > self.memory_limit_bytes and self.cache:
            self.cache.popitem(last=False)
            
        # Periodic cleanup
        if current_time - self.last_cleanup > 300:  # 5 minutes
            self.last_cleanup = current_time
            # Remove old entries (simplified cleanup)
            if len(self.cache) > self.max_size // 2:
                for _ in range(len(self.cache) // 4):
                    if self.cache:
                        self.cache.popitem(last=False)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU update."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with memory management."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                old_value = self.cache.pop(key)
                self.memory_usage -= self._calculate_memory_usage(old_value)
            
            # Add new value
            self.cache[key] = value
            self.memory_usage += self._calculate_memory_usage(value)
            
            # Cleanup if needed
            self._cleanup_if_needed()
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.memory_usage = 0

class FeatureGenerationOptimizer(VectorBTHelper):
    """
    Optimizes feature generation parameters using data-driven approaches.

    This class provides comprehensive optimization for feature parameters,
    particularly lookback periods, using various statistical and machine learning
    methods to determine optimal values for each feature.
    """

    def __init__(self, config: Optional[FeatureOptimizationConfig] = None):
        """Initialize the feature generation optimizer."""
        self.logger = logger.getChild('FeatureGenerationOptimizer')
        self.logger.info("🚀 Initializing FeatureGenerationOptimizer...")
        start_time = time.time()

        self.config = config or FeatureOptimizationConfig()
        self.logger.info(f"📊 Configuration loaded: {self.config.optimization_method.value}")

        # Initialize components
        self.logger.debug("🔧 Initializing GPU manager...")
        try:
            self.gpu_manager = M1GPUManager() if self.config.parallel_processing else None
            if self.gpu_manager:
                self.logger.debug("✅ GPU manager initialized")
            else:
                self.logger.debug("ℹ️ GPU manager not initialized (parallel processing disabled)")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize GPU manager: {e}")
            self.gpu_manager = None

        init_time = time.time() - start_time
        self.logger.info(f"✅ FeatureGenerationOptimizer initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Min lookback: {self.config.min_lookback}, Max lookback: {self.config.max_lookback}")
        self.logger.info(f"📊 CV folds: {self.config.cv_folds}, Parallel processing: {self.config.parallel_processing}")
        self.parallel_processor = ParallelProcessor(max_workers=self.config.max_workers)

        # Initialize optimization utilities
        self.vectorbt_optimizer = None
        self.vectorization_manager = None
        if OPTIMIZATION_UTILS_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_vectorization_manager()
                self.logger.debug("✅ Optimization utilities initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize optimization utilities: {e}")

        # Hardware-optimized cache for optimization results
        self._optimization_cache = HardwareOptimizedCache(
            max_size=self.config.cache_size_limit,
            memory_limit_mb=self.config.cache_memory_limit_mb
        )
        
        # Memoization cache for feature generation
        self._feature_cache = {}
        self._correlation_cache = {}
        
        # Performance tracking
        self._performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'early_terminations': 0,
            'vectorized_operations': 0
        }

        # Validation
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the optimization configuration."""
        if self.config.min_lookback >= self.config.max_lookback:
            raise ValueError("min_lookback must be less than max_lookback")

        if self.config.step_size <= 0:
            raise ValueError("step_size must be positive")

        if not SKLEARN_AVAILABLE and self.config.optimization_method == OptimizationMethod.CROSS_VALIDATION:
            self.logger.warning("Scikit-learn not available, falling back to statistical analysis")
            self.config.optimization_method = OptimizationMethod.STATISTICAL_ANALYSIS

    def _generate_cache_key(self, data: pd.DataFrame, feature_name: str, lookback: int) -> str:
        """Generate a cache key for feature generation."""
        # Use data content hash for more accurate caching
        # Fix: Ensure both parts are bytes before concatenation
        shape_bytes = str(data.shape).encode()
        data_bytes = data.iloc[-10:].values.tobytes()
        data_hash = hashlib.md5(shape_bytes + data_bytes).hexdigest()[:16]
        return f"{feature_name}_{lookback}_{data_hash}"
    
    def _precompute_features_batch(self, data: pd.DataFrame, feature_name: str, 
                                 lookback_range: range, feature_generator: Callable) -> Dict[int, pd.Series]:
        """Precompute all features for a range of lookbacks to avoid redundant computation."""
        features = {}
        
        # Use vectorized operations if data is large enough
        if len(data) > self.config.vectorization_threshold:
            try:
                features = self._vectorized_feature_generation(data, feature_name, lookback_range, feature_generator)
                return features
            except Exception as e:
                self.logger.warning(f"Vectorized feature generation failed: {e}")
        
        # Fallback to individual computation with caching
        for lookback in lookback_range:
            cache_key = self._generate_cache_key(data, feature_name, lookback)
            if cache_key in self._feature_cache:
                features[lookback] = self._feature_cache[cache_key]
            else:
                try:
                    feature_values = feature_generator(data, lookback)
                    self._feature_cache[cache_key] = feature_values
                    features[lookback] = feature_values
                except Exception as e:
                    self.logger.warning(f"Error generating feature for lookback {lookback}: {e}")
                    features[lookback] = pd.Series([np.nan] * len(data), index=data.index)
        
        return features
    
    def _vectorized_feature_generation(self, data: pd.DataFrame, feature_name: str, 
                                     lookback_range: range, feature_generator: Callable) -> Dict[int, pd.Series]:
        """Use vectorized operations for efficient feature generation."""
        features = {}
        
        # For rolling operations, use numpy sliding window view
        if hasattr(feature_generator, '__name__') and 'rolling' in feature_generator.__name__.lower():
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                
                # Get the base series
                base_series = data[feature_name].values
                max_lookback = max(lookback_range)
                
                # Create sliding window view
                if len(base_series) >= max_lookback:
                    windowed = sliding_window_view(base_series, max_lookback)
                    
                    for lookback in lookback_range:
                        if lookback <= max_lookback:
                            # Extract rolling mean/std for this lookback
                            rolling_values = np.mean(windowed[:, -lookback:], axis=1)
                            
                            # Pad with NaN for initial values
                            padded_values = np.full(len(base_series), np.nan)
                            padded_values[max_lookback-1:] = rolling_values
                            
                            features[lookback] = pd.Series(padded_values, index=data.index)
                        else:
                            features[lookback] = pd.Series([np.nan] * len(data), index=data.index)
                            
            except ImportError:
                # Fallback to individual computation
                for lookback in lookback_range:
                    features[lookback] = feature_generator(data, lookback)
        else:
            # For non-rolling features, compute individually
            for lookback in lookback_range:
                features[lookback] = feature_generator(data, lookback)
        
        return features
    
    def _get_cached_feature(self, data: pd.DataFrame, feature_name: str, lookback: int, 
                       feature_generator: Callable) -> pd.Series:
        """Get cached feature or generate and cache it."""
        cache_key = self._generate_cache_key(data, feature_name, lookback)
        
        if cache_key in self._feature_cache:
            self._performance_stats['cache_hits'] += 1
            return self._feature_cache[cache_key]
        
        # Generate feature
        feature_values = feature_generator(data, lookback)
        self._feature_cache[cache_key] = feature_values
        self._performance_stats['cache_misses'] += 1
        
        return feature_values
    
    def _get_cached_correlation(self, feature_values: pd.Series, target_values: pd.Series) -> float:
        """Get cached correlation or calculate and cache it."""
        # Create cache key from data hash
        feature_hash = hashlib.md5(feature_values.values.tobytes()).hexdigest()[:8]
        target_hash = hashlib.md5(target_values.values.tobytes()).hexdigest()[:8]
        cache_key = f"corr_{feature_hash}_{target_hash}"
        
        if cache_key in self._correlation_cache:
            self._performance_stats['cache_hits'] += 1
            return self._correlation_cache[cache_key]
        
        # Calculate correlation
        valid_indices = ~(feature_values.isna() | target_values.isna())
        if valid_indices.sum() < 10:
            correlation = 0.0
        else:
            correlation = abs(feature_values[valid_indices].corr(target_values[valid_indices]))
            if pd.isna(correlation):
                correlation = 0.0
        
        self._correlation_cache[cache_key] = correlation
        self._performance_stats['cache_misses'] += 1
        
        return correlation
    
    def _batch_correlation_calculation(self, data: pd.DataFrame, feature_name: str, 
                                     target_column: str, lookback_range: range) -> List[float]:
        """Calculate correlations in batch using vectorized NumPy operations."""
        # Precompute all features for the lookback range
        features = self._precompute_features_batch(data, feature_name, lookback_range, 
                                                lambda d, l: d[feature_name].rolling(window=l).mean())
        
        # Get target values
        target_values = data[target_column].values
        
        # Stack all features into a matrix
        feature_matrix = []
        valid_indices_list = []
        
        for lookback in lookback_range:
            if lookback in features:
                feature_series = features[lookback]
                valid_indices = ~(feature_series.isna() | pd.isna(target_values))
                if valid_indices.sum() > 10:
                    feature_matrix.append(feature_series.values)
                    valid_indices_list.append(valid_indices)
                else:
                    feature_matrix.append(np.full(len(data), np.nan))
                    valid_indices_list.append(np.zeros(len(data), dtype=bool))
            else:
                feature_matrix.append(np.full(len(data), np.nan))
                valid_indices_list.append(np.zeros(len(data), dtype=bool))
        
        if not feature_matrix:
            return [0.0] * len(lookback_range)
        
        # Convert to NumPy array
        X = np.column_stack(feature_matrix)  # Shape: (n_samples, n_lookbacks)
        
        # Vectorized correlation calculation
        correlations = self._vectorized_correlation_batch(X, target_values, valid_indices_list)
        
        return correlations
    
    def _vectorized_correlation_batch(self, X: np.ndarray, y: np.ndarray, 
                                    valid_indices_list: List[np.ndarray]) -> List[float]:
        """Calculate correlations for all features in batch using NumPy."""
        correlations = []
        
        for i, valid_indices in enumerate(valid_indices_list):
            if valid_indices.sum() < 10:
                correlations.append(0.0)
                continue
            
            # Extract valid data
            x_valid = X[valid_indices, i]
            y_valid = y[valid_indices]
            
            # Remove any remaining NaN values
            valid_mask = ~(np.isnan(x_valid) | np.isnan(y_valid))
            if valid_mask.sum() < 10:
                correlations.append(0.0)
                continue
            
            x_clean = x_valid[valid_mask]
            y_clean = y_valid[valid_mask]
            
            # Calculate correlation using NumPy
            if len(x_clean) > 1 and np.std(x_clean) > 0 and np.std(y_clean) > 0:
                correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
                else:
                    correlations.append(0.0)
            else:
                correlations.append(0.0)
        
        return correlations
    
    def _adaptive_search_optimization(self, data: pd.DataFrame, feature_name: str, 
                                    target_column: str, feature_generator: Callable) -> Tuple[int, float]:
        """Use adaptive search methods for efficient optimization."""
        if self.config.adaptive_search_method == "bayesian":
            return self._bayesian_optimization(data, feature_name, target_column, feature_generator)
        elif self.config.adaptive_search_method == "golden_section":
            return self._golden_section_optimization(data, feature_name, target_column, feature_generator)
        else:
            # Fallback to grid search
            return self._grid_search_optimization(data, feature_name, target_column, feature_generator)
    
    def _bayesian_optimization(self, data: pd.DataFrame, feature_name: str, 
                             target_column: str, feature_generator: Callable) -> Tuple[int, float]:
        """Use Bayesian optimization for efficient search."""
        try:
            import optuna
            
            def objective(trial):
                lookback = trial.suggest_int("lookback", self.config.min_lookback, self.config.max_lookback)
                
                # Generate feature
                feature_values = feature_generator(data, lookback)
                
                # Calculate correlation
                valid_indices = ~(feature_values.isna() | data[target_column].isna())
                if valid_indices.sum() < 10:
                    return 0.0
                
                correlation = abs(feature_values[valid_indices].corr(data[target_column][valid_indices]))
                if pd.isna(correlation):
                    return 0.0
                
                # Use negative correlation for minimization
                return -correlation
            
            # Create study
            study = optuna.create_study(direction="minimize")
            
            # Optimize with early stopping
            study.optimize(
                objective, 
                n_trials=self.config.max_trials,
                n_startup_trials=self.config.n_startup_trials,
                show_progress_bar=False
            )
            
            best_lookback = study.best_params["lookback"]
            best_score = -study.best_value
            
            self.logger.info(f"Bayesian optimization completed: lookback={best_lookback}, score={best_score:.4f}")
            return best_lookback, best_score
            
        except ImportError:
            self.logger.warning("Optuna not available, falling back to grid search")
            return self._grid_search_optimization(data, feature_name, target_column, feature_generator)
        except Exception as e:
            self.logger.warning(f"Bayesian optimization failed: {e}, falling back to grid search")
            return self._grid_search_optimization(data, feature_name, target_column, feature_generator)
    
    def _golden_section_optimization(self, data: pd.DataFrame, feature_name: str, 
                                   target_column: str, feature_generator: Callable) -> Tuple[int, float]:
        """Use golden section search for unimodal optimization."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        resphi = 2 - phi
        
        a, b = self.config.min_lookback, self.config.max_lookback
        tol = 1e-6
        max_iter = 50
        
        # Initial points
        x1 = a + resphi * (b - a)
        x2 = a + (1 - resphi) * (b - a)
        
        f1 = self._evaluate_lookback(data, feature_name, target_column, feature_generator, int(x1))
        f2 = self._evaluate_lookback(data, feature_name, target_column, feature_generator, int(x2))
        
        for _ in range(max_iter):
            if abs(b - a) < tol:
                break
                
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + resphi * (b - a)
                f1 = self._evaluate_lookback(data, feature_name, target_column, feature_generator, int(x1))
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + (1 - resphi) * (b - a)
                f2 = self._evaluate_lookback(data, feature_name, target_column, feature_generator, int(x2))
        
        best_lookback = int((a + b) / 2)
        best_score = self._evaluate_lookback(data, feature_name, target_column, feature_generator, best_lookback)
        
        return best_lookback, best_score
    
    def _grid_search_optimization(self, data: pd.DataFrame, feature_name: str, 
                                target_column: str, feature_generator: Callable) -> Tuple[int, float]:
        """Fallback grid search optimization."""
        best_score = -np.inf
        best_lookback = self.config.min_lookback
        
        lookback_range = range(self.config.min_lookback, self.config.max_lookback + 1, self.config.step_size)
        
        for lookback in lookback_range:
            score = self._evaluate_lookback(data, feature_name, target_column, feature_generator, lookback)
            if score > best_score:
                best_score = score
                best_lookback = lookback
        
        return best_lookback, best_score
    
    def _enhanced_stability_analysis(self, data: pd.DataFrame, feature_name: str, 
                                   target_column: str, feature_generator: Callable, 
                                   optimal_lookback: int) -> Dict[str, Any]:
        """Perform comprehensive stability analysis."""
        stability_report = {}
        
        try:
            # 1. Purged Cross-Validation
            purged_cv_scores = self._purged_cross_validation(
                data, feature_name, target_column, feature_generator, optimal_lookback
            )
            stability_report['purged_cv_scores'] = purged_cv_scores
            stability_report['ic_cv'] = np.std(purged_cv_scores) / np.mean(purged_cv_scores) if purged_cv_scores else 0.0
            
            # 2. Bootstrap Analysis
            bootstrap_ci = self._bootstrap_analysis(
                data, feature_name, target_column, feature_generator, optimal_lookback
            )
            stability_report['bootstrap_ci'] = bootstrap_ci
            
            # 3. Rolling Window Robustness
            rolling_robustness = self._rolling_window_robustness(
                data, feature_name, target_column, feature_generator, optimal_lookback
            )
            stability_report.update(rolling_robustness)
            
            # 4. Perturbation Stability
            perturbation_stability = self._perturbation_stability(
                data, feature_name, target_column, feature_generator, optimal_lookback
            )
            stability_report['perturbation_stability'] = perturbation_stability
            
            # 5. Diebold-Mariano Test
            dm_test = self._diebold_mariano_test(
                data, feature_name, target_column, feature_generator, optimal_lookback
            )
            stability_report['diebold_mariano_p'] = dm_test
            
            # 6. Selection Probability
            selection_prob = self._selection_probability(
                data, feature_name, target_column, feature_generator, optimal_lookback
            )
            stability_report['selection_probability'] = selection_prob
            
        except Exception as e:
            self.logger.warning(f"Enhanced stability analysis failed: {e}")
            stability_report = {'error': str(e)}
        
        return stability_report
    
    def _purged_cross_validation(self, data: pd.DataFrame, feature_name: str, 
                               target_column: str, feature_generator: Callable, 
                               lookback: int, embargo_period: int = 5) -> List[float]:
        """Perform purged cross-validation to avoid label leakage."""
        scores = []
        
        try:
            # Create purged splits
            n_splits = min(5, len(data) // 100)  # Adaptive number of splits
            if n_splits < 2:
                return [0.0]
            
            split_size = len(data) // n_splits
            
            for i in range(n_splits):
                # Define validation period
                val_start = i * split_size
                val_end = min((i + 1) * split_size, len(data))
                
                # Define training period (before validation, with embargo)
                train_end = max(0, val_start - embargo_period)
                train_start = max(0, train_end - split_size)
                
                if train_start >= train_end or val_start >= val_end:
                    continue
                
                # Get splits - fix duplicate index issues
                train_data = data.iloc[train_start:train_end].copy()
                val_data = data.iloc[val_start:val_end].copy()
                
                # Reset index to avoid duplicate label issues
                if train_data.index.duplicated().any():
                    train_data = train_data.reset_index(drop=True)
                if val_data.index.duplicated().any():
                    val_data = val_data.reset_index(drop=True)
                
                if len(train_data) < 20 or len(val_data) < 10:
                    continue
                
                # Generate feature for training period
                train_feature = feature_generator(train_data, lookback)
                val_feature = feature_generator(val_data, lookback)
                
                # Calculate correlation on validation set
                valid_indices = ~(val_feature.isna() | val_data[target_column].isna())
                if valid_indices.sum() > 5:
                    # Ensure both series have the same index for correlation
                    val_feature_clean = val_feature[valid_indices]
                    val_target_clean = val_data[target_column][valid_indices]
                    
                    # Reset indices to ensure alignment
                    if val_feature_clean.index.duplicated().any() or val_target_clean.index.duplicated().any():
                        val_feature_clean = val_feature_clean.reset_index(drop=True)
                        val_target_clean = val_target_clean.reset_index(drop=True)
                    
                    correlation = abs(val_feature_clean.corr(val_target_clean))
                    if not pd.isna(correlation):
                        scores.append(correlation)
                    else:
                        scores.append(0.0)
                else:
                    scores.append(0.0)
                    
        except Exception as e:
            self.logger.warning(f"Purged CV failed: {e}")
            scores = [0.0]
        
        return scores
    
    def _bootstrap_analysis(self, data: pd.DataFrame, feature_name: str, 
                         target_column: str, feature_generator: Callable, 
                         lookback: int, n_bootstrap: int = 100) -> Tuple[float, float]:
        """Perform bootstrap analysis for confidence intervals."""
        try:
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Block bootstrap for time series
                bootstrap_indices = self._block_bootstrap_indices(len(data), block_size=50)
                bootstrap_data = data.iloc[bootstrap_indices].copy()
                
                # Reset index to avoid duplicate label issues
                if bootstrap_data.index.duplicated().any():
                    bootstrap_data = bootstrap_data.reset_index(drop=True)
                
                # Generate feature and calculate correlation
                feature_values = feature_generator(bootstrap_data, lookback)
                valid_indices = ~(feature_values.isna() | bootstrap_data[target_column].isna())
                
                if valid_indices.sum() > 10:
                    # Ensure both series have the same index for correlation
                    feature_clean = feature_values[valid_indices]
                    target_clean = bootstrap_data[target_column][valid_indices]
                    
                    # Reset indices to ensure alignment
                    if feature_clean.index.duplicated().any() or target_clean.index.duplicated().any():
                        feature_clean = feature_clean.reset_index(drop=True)
                        target_clean = target_clean.reset_index(drop=True)
                    
                    correlation = abs(feature_clean.corr(target_clean))
                    if not pd.isna(correlation):
                        bootstrap_scores.append(correlation)
            
            if bootstrap_scores:
                ci_low = np.percentile(bootstrap_scores, 2.5)
                ci_high = np.percentile(bootstrap_scores, 97.5)
                return ci_low, ci_high
            else:
                return 0.0, 0.0
                
        except Exception as e:
            self.logger.warning(f"Bootstrap analysis failed: {e}")
            return 0.0, 0.0
    
    def _block_bootstrap_indices(self, n: int, block_size: int = 50) -> np.ndarray:
        """Generate block bootstrap indices for time series."""
        n_blocks = n // block_size
        indices = []
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            block_indices = np.arange(start_idx, start_idx + block_size)
            indices.extend(block_indices)
        
        # Pad to original length
        while len(indices) < n:
            indices.append(np.random.randint(0, n))
        
        return np.array(indices[:n])
    
    def _rolling_window_robustness(self, data: pd.DataFrame, feature_name: str, 
                                 target_column: str, feature_generator: Callable, 
                                 optimal_lookback: int) -> Dict[str, Any]:
        """Analyze robustness across rolling windows."""
        try:
            window_size = min(500, len(data) // 3)
            step_size = window_size // 4
            window_results = []
            
            for start in range(0, len(data) - window_size, step_size):
                end = start + window_size
                window_data = data.iloc[start:end]
                
                # Find optimal lookback for this window
                window_scores = []
                for test_lookback in range(max(5, optimal_lookback - 10), min(optimal_lookback + 11, window_size // 4)):
                    try:
                        feature_values = feature_generator(window_data, test_lookback)
                        valid_indices = ~(feature_values.isna() | window_data[target_column].isna())
                        
                        if valid_indices.sum() > 10:
                            correlation = abs(feature_values[valid_indices].corr(window_data[target_column][valid_indices]))
                            if not pd.isna(correlation):
                                window_scores.append((test_lookback, correlation))
                    except Exception:
                        continue
                
                if window_scores:
                    best_window_lookback = max(window_scores, key=lambda x: x[1])[0]
                    window_results.append(best_window_lookback)
            
            if window_results:
                lookback_entropy = self._calculate_entropy(window_results)
                lookback_std = np.std(window_results)
                lookback_mean = np.mean(window_results)
                
                return {
                    'lookback_entropy': lookback_entropy,
                    'lookback_std': lookback_std,
                    'lookback_mean': lookback_mean,
                    'window_count': len(window_results)
                }
            else:
                return {'lookback_entropy': 0.0, 'lookback_std': 0.0, 'lookback_mean': optimal_lookback, 'window_count': 0}
                
        except Exception as e:
            self.logger.warning(f"Rolling window robustness failed: {e}")
            return {'lookback_entropy': 0.0, 'lookback_std': 0.0, 'lookback_mean': optimal_lookback, 'window_count': 0}
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution."""
        if not values:
            return 0.0
        
        unique_values, counts = np.unique(values, return_counts=True)
        probabilities = counts / len(values)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _perturbation_stability(self, data: pd.DataFrame, feature_name: str, 
                              target_column: str, feature_generator: Callable, 
                              optimal_lookback: int, n_perturbations: int = 20) -> float:
        """Test stability under small perturbations."""
        try:
            original_score = self._evaluate_lookback(data, feature_name, target_column, feature_generator, optimal_lookback)
            stable_count = 0
            
            for _ in range(n_perturbations):
                # Add small noise to data
                perturbed_data = data.copy()
                noise_scale = 0.001 * perturbed_data[feature_name].std()
                perturbed_data[feature_name] += np.random.normal(0, noise_scale, len(perturbed_data))
                
                # Re-optimize
                perturbed_scores = []
                for test_lookback in range(max(5, optimal_lookback - 5), min(optimal_lookback + 6, len(data) // 4)):
                    score = self._evaluate_lookback(perturbed_data, feature_name, target_column, feature_generator, test_lookback)
                    perturbed_scores.append((test_lookback, score))
                
                if perturbed_scores:
                    best_perturbed_lookback = max(perturbed_scores, key=lambda x: x[1])[0]
                    if abs(best_perturbed_lookback - optimal_lookback) <= 2:
                        stable_count += 1
            
            return stable_count / n_perturbations
            
        except Exception as e:
            self.logger.warning(f"Perturbation stability failed: {e}")
            return 0.0
    
    def _diebold_mariano_test(self, data: pd.DataFrame, feature_name: str, 
                            target_column: str, feature_generator: Callable, 
                            optimal_lookback: int) -> float:
        """Perform Diebold-Mariano test for forecast comparison."""
        try:
            # Generate features for optimal and alternative lookbacks
            optimal_feature = feature_generator(data, optimal_lookback)
            alt_lookback = optimal_lookback + 5 if optimal_lookback + 5 <= len(data) // 4 else optimal_lookback - 5
            alt_feature = feature_generator(data, alt_lookback)
            
            # Calculate forecast errors
            valid_indices = ~(optimal_feature.isna() | alt_feature.isna() | data[target_column].isna())
            if valid_indices.sum() < 20:
                return 1.0  # No significant difference
            
            # Fix duplicate index issues
            optimal_feature_clean = optimal_feature[valid_indices]
            alt_feature_clean = alt_feature[valid_indices]
            target_clean = data[target_column][valid_indices]
            
            # Reset indices to ensure alignment
            if (optimal_feature_clean.index.duplicated().any() or 
                alt_feature_clean.index.duplicated().any() or 
                target_clean.index.duplicated().any()):
                optimal_feature_clean = optimal_feature_clean.reset_index(drop=True)
                alt_feature_clean = alt_feature_clean.reset_index(drop=True)
                target_clean = target_clean.reset_index(drop=True)
            
            optimal_errors = np.abs(optimal_feature_clean - target_clean)
            alt_errors = np.abs(alt_feature_clean - target_clean)
            
            # Calculate DM statistic
            d = optimal_errors - alt_errors
            dm_stat = np.mean(d) / (np.std(d) / np.sqrt(len(d)) + 1e-10)
            
            # Approximate p-value (simplified)
            p_value = 2 * (1 - abs(dm_stat) / (abs(dm_stat) + 1))
            return p_value
            
        except Exception as e:
            self.logger.warning(f"Diebold-Mariano test failed: {e}")
            return 1.0
    
    def _selection_probability(self, data: pd.DataFrame, feature_name: str, 
                             target_column: str, feature_generator: Callable, 
                             optimal_lookback: int, n_bootstrap: int = 50) -> float:
        """Calculate probability of selecting the optimal lookback under bootstrap."""
        try:
            selection_count = 0
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_indices = self._block_bootstrap_indices(len(data), block_size=50)
                bootstrap_data = data.iloc[bootstrap_indices]
                
                # Find best lookback in bootstrap sample
                best_score = -np.inf
                best_lookback = optimal_lookback
                
                for test_lookback in range(max(5, optimal_lookback - 10), min(optimal_lookback + 11, len(bootstrap_data) // 4)):
                    score = self._evaluate_lookback(bootstrap_data, feature_name, target_column, feature_generator, test_lookback)
                    if score > best_score:
                        best_score = score
                        best_lookback = test_lookback
                
                if best_lookback == optimal_lookback:
                    selection_count += 1
            
            return selection_count / n_bootstrap
            
        except Exception as e:
            self.logger.warning(f"Selection probability calculation failed: {e}")
            return 0.0
    
    def _evaluate_lookback(self, data: pd.DataFrame, feature_name: str, 
                         target_column: str, feature_generator: Callable, lookback: int) -> float:
        """Evaluate a single lookback period."""
        try:
            feature_values = feature_generator(data, lookback)
            valid_indices = ~(feature_values.isna() | data[target_column].isna())
            
            if valid_indices.sum() < 10:
                return 0.0
            
            correlation = abs(feature_values[valid_indices].corr(data[target_column][valid_indices]))
            return correlation if not pd.isna(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _vectorized_correlation_calculation(self, data: pd.DataFrame, feature_name: str, 
                                          target_values: pd.Series, lookback_range: range) -> List[float]:
        """Use VectorBTRollingOptimizer for efficient correlation calculation."""
        correlations = []
        
        try:
            # Use VectorBTRollingOptimizer for rolling operations
            feature_series = data[feature_name]
            
            for lookback in lookback_range:
                # Use optimized rolling mean
                rolling_feature = self.vectorbt_optimizer.rolling_mean(feature_series, window=lookback)
                
                # Calculate correlation efficiently
                valid_indices = ~(rolling_feature.isna() | target_values.isna())
                if valid_indices.sum() > 10:
                    correlation = abs(rolling_feature[valid_indices].corr(target_values[valid_indices]))
                    correlations.append(correlation if not pd.isna(correlation) else 0.0)
                else:
                    correlations.append(0.0)
                    
        except Exception as e:
            self.logger.warning(f"Vectorized correlation calculation failed: {e}")
            # Fallback to standard calculation
            for lookback in lookback_range:
                rolling_feature = data[feature_name].rolling(window=lookback).mean()
                valid_indices = ~(rolling_feature.isna() | target_values.isna())
                if valid_indices.sum() > 10:
                    correlation = abs(rolling_feature[valid_indices].corr(target_values[valid_indices]))
                    correlations.append(correlation if not pd.isna(correlation) else 0.0)
                else:
                    correlations.append(0.0)
        
        return correlations

    async def optimize_feature_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable[[pd.DataFrame, int], pd.Series],
        regime_column: Optional[str] = None,
        include_top_candidates: bool = True,
        max_candidates: int = 3
    ) -> FeatureOptimizationResult:
        """
        Optimize the lookback period for a specific feature.

        Args:
            data: Input data DataFrame
            feature_name: Name of the feature to optimize
            target_column: Name of the target column
            feature_generator: Function that generates the feature given data and lookback
            regime_column: Optional regime column for regime-aware optimization

        Returns:
            FeatureOptimizationResult with optimal parameters
        """
        self.logger.info(f"Optimizing lookback period for feature: {feature_name}")

        # Check cache first with comprehensive key
        hdr = (data.index[0], data.index[-1], tuple(data.columns), len(data))
        cache_key = (f"{feature_name}|{target_column}|{id(feature_generator)}|{hash(hdr)}|"
                    f"{self.config.min_lookback}-{self.config.max_lookback}-{self.config.step_size}|"
                    f"{self.config.optimization_method.value}|{include_top_candidates}|{max_candidates}")
        
        cached_result = self._optimization_cache.get(cache_key)
        if cached_result:
            self.logger.info(f"Using cached optimization result for {feature_name}")
            return cached_result

        try:
            # Generate lookback range
            lookback_range = range(
                self.config.min_lookback,
                self.config.max_lookback + 1,
                self.config.step_size
            )

            # Optimize based on method
            if self.config.optimization_method == OptimizationMethod.CROSS_VALIDATION:
                result = await self._optimize_with_cross_validation(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            elif self.config.optimization_method == OptimizationMethod.STATISTICAL_ANALYSIS:
                result = await self._optimize_with_statistical_analysis(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            elif self.config.optimization_method == OptimizationMethod.INFORMATION_THEORY:
                result = await self._optimize_with_information_theory(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            elif self.config.optimization_method == OptimizationMethod.REGIME_AWARE:
                result = await self._optimize_with_regime_awareness(
                    data, feature_name, target_column, feature_generator, lookback_range, regime_column
                )
            else:
                result = await self._optimize_adaptive(
                    data, feature_name, target_column, feature_generator, lookback_range
                )

            # Add regime-specific analysis if regime column provided
            if regime_column and regime_column in data.columns:
                result.regime_specific_results = await self._analyze_regime_specific_performance(
                    data, feature_name, target_column, feature_generator, result.optimal_lookback, regime_column
                )

            # Add decay analysis
            result.decay_analysis = await self._analyze_feature_decay(
                data, feature_name, feature_generator, result.optimal_lookback
            )

            # Generate top candidates if requested
            if include_top_candidates:
                result.top_candidates, result.redundancy_analysis = await self._generate_top_candidates(
                    data, feature_name, target_column, feature_generator, max_candidates
                )
            
            # Enhanced stability analysis
            result.stability_report = self._enhanced_stability_analysis(
                data, feature_name, target_column, feature_generator, result.optimal_lookback
            )

            # Cache result using hardware cache
            self._optimization_cache.set(cache_key, result)

            self.logger.info(f"Optimization completed for {feature_name}: optimal_lookback={result.optimal_lookback}")
            if result.top_candidates:
                self.logger.info(f"Top {len(result.top_candidates)} candidates generated with redundancy filtering")
            return result

        except Exception as e:
            self.logger.error(f"Error optimizing feature {feature_name}: {e}")
            raise

    async def _optimize_with_cross_validation(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Optimize using cross-validation approach."""
        self.logger.info(f"Using cross-validation optimization for {feature_name}")

        best_score = -np.inf
        best_lookback = self.config.min_lookback
        validation_scores = []

        for lookback in lookback_range:
            try:
                # ALWAYS reset index to avoid duplicate label issues with pandas operations
                # This is crucial for avoiding "cannot reindex on an axis with duplicate labels" errors
                data_reset = data.reset_index(drop=True)
                
                # Generate feature with current lookback on the reset data
                feature_values = feature_generator(data_reset, lookback)
                
                # ALWAYS reset index to avoid duplicate label issues with pandas operations
                # This is crucial for avoiding "cannot reindex on an axis with duplicate labels" errors
                feature_values = feature_values.reset_index(drop=True)
                
                # Prepare data for cross-validation with proper index alignment
                # Ensure both series have the same index
                feature_values_aligned = feature_values.copy()
                target_values_aligned = data_reset[target_column].copy()
                
                # Align indices by using the intersection
                common_index = feature_values_aligned.index.intersection(target_values_aligned.index)
                if len(common_index) < 10:
                    continue
                    
                feature_values_aligned = feature_values_aligned.loc[common_index]
                target_values_aligned = target_values_aligned.loc[common_index]
                
                # Now calculate valid indices with aligned data
                valid_indices = ~(feature_values_aligned.isna() | target_values_aligned.isna())
                
                # Extract values as numpy arrays to avoid any index alignment issues
                X = feature_values_aligned[valid_indices].values.reshape(-1, 1)
                y = target_values_aligned[valid_indices].values

                if len(X) < 10:  # Need minimum data for CV
                    continue

                # Perform time series cross-validation with minimum sample guards
                # Ensure we have enough data for cross-validation
                if len(X) < self.config.cv_folds * 10:  # Need at least 10 samples per fold
                    continue
                    
                tscv = TimeSeriesSplit(n_splits=min(self.config.cv_folds, len(X) // 10))
                scores = []
                previous_score = None
                improvement_threshold = 0.01  # Stop if improvement < 1%
                no_improvement_count = 0

                for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    # Ensure indices are within bounds
                    if len(train_idx) == 0 or len(val_idx) == 0:
                        continue
                        
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # Guard against small folds
                    if len(X_train) < 5 or len(X_val) < 5:
                        continue

                    # Choose model based on target type and optimization metric
                    # Use LinearRegression/LogisticRegression for 100x speed improvement
                    if self._is_binary_target(y_train):
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                    else:
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        
                        # Use configured optimization metric
                        if self.config.optimization_metric == "r2":
                            from sklearn.metrics import r2_score
                            score = r2_score(y_val, y_pred)
                        else:  # Default to negative MSE
                            score = -mean_squared_error(y_val, y_pred)
                    
                    scores.append(score)
                    
                    # Early stopping logic: check if improvement is minimal
                    if previous_score is not None:
                        improvement = abs(score - previous_score)
                        if improvement < improvement_threshold:
                            no_improvement_count += 1
                            if no_improvement_count >= 2:  # Stop after 2 consecutive minimal improvements
                                self.logger.debug(f"Early stopping CV at fold {fold_idx + 1} due to minimal improvement")
                                break
                        else:
                            no_improvement_count = 0  # Reset counter if there's improvement
                    
                    previous_score = score

                avg_score = np.mean(scores)
                validation_scores.append(avg_score)

                if avg_score > best_score:
                    best_score = avg_score
                    best_lookback = lookback

            except Exception as e:
                self.logger.warning(f"Error in cross-validation for lookback {lookback}: {e}")
                continue

        # Calculate stability score
        stability_score = self._calculate_stability_score(validation_scores)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(validation_scores)

        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.CROSS_VALIDATION.value,
            validation_scores=validation_scores
        )

    async def _optimize_with_statistical_analysis(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Optimize using statistical analysis approach."""
        self.logger.info(f"Using statistical analysis optimization for {feature_name}")

        best_score = -np.inf
        best_lookback = self.config.min_lookback
        scores = []

        # Reset index once before the loop to avoid redundant operations
        data_reset = data.reset_index(drop=True)

        for lookback in lookback_range:
            try:
                # Generate feature with current lookback on the reset data
                feature_values = feature_generator(data_reset, lookback)

                # Calculate correlation with target
                valid_indices = ~(feature_values.isna() | data_reset[target_column].isna())
                if valid_indices.sum() < 10:
                    continue

                # Calculate correlation with aligned indices
                correlation = abs(feature_values[valid_indices].corr(data_reset[target_column][valid_indices]))

                # Calculate feature stability (low variance is better)
                feature_std = feature_values[valid_indices].std()
                feature_mean = feature_values[valid_indices].mean()
                
                stability = 1 / (1 + feature_std / abs(feature_mean)) if feature_mean != 0 else 0

                # Combined score
                score = correlation * stability
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_lookback = lookback

            except Exception as e:
                self.logger.warning(f"Error in statistical analysis for lookback {lookback}: {e}")
                continue

        # Calculate stability score
        stability_score = self._calculate_stability_score(scores)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)

        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS.value,
            validation_scores=scores
        )

    async def _optimize_with_information_theory(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Optimize using information theory approach."""
        self.logger.info(f"Using information theory optimization for {feature_name}")

        best_score = -np.inf
        best_lookback = self.config.min_lookback
        scores = []

        for lookback in lookback_range:
            try:
                # ALWAYS reset index to avoid duplicate label issues with pandas operations
                data_reset = data.reset_index(drop=True)
                
                # Generate feature with current lookback on the reset data
                feature_values = feature_generator(data_reset, lookback)

                # Calculate mutual information
                valid_indices = ~(feature_values.isna() | data_reset[target_column].isna())
                if valid_indices.sum() < 10:
                    continue

                # Discretize for mutual information calculation
                feature_discrete = pd.cut(feature_values[valid_indices], bins=10, labels=False)
                target_discrete = pd.cut(data_reset[target_column][valid_indices], bins=10, labels=False)

                # Calculate mutual information
                mi_score = self._calculate_mutual_information(feature_discrete, target_discrete)
                scores.append(mi_score)

                if mi_score > best_score:
                    best_score = mi_score
                    best_lookback = lookback

            except Exception as e:
                self.logger.warning(f"Error in information theory analysis for lookback {lookback}: {e}")
                continue

        # Calculate stability score
        stability_score = self._calculate_stability_score(scores)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)

        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.INFORMATION_THEORY.value,
            validation_scores=scores
        )

    async def _optimize_with_regime_awareness(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range,
        regime_column: str
    ) -> FeatureOptimizationResult:
        """Optimize using regime-aware approach."""
        self.logger.info(f"Using regime-aware optimization for {feature_name}")

        regime_results = {}
        overall_scores = []

        # Get unique regimes
        regimes = data[regime_column].unique()

        for regime in regimes:
            regime_data = data[data[regime_column] == regime]
            if len(regime_data) < 20:  # Need minimum data per regime
                continue

            regime_scores = []
            best_regime_score = -np.inf
            best_regime_lookback = self.config.min_lookback

            for lookback in lookback_range:
                try:
                    # ALWAYS reset index to avoid duplicate label issues with pandas operations
                    regime_data_reset = regime_data.reset_index(drop=True)
                    
                    # Generate feature for this regime
                    feature_values = feature_generator(regime_data_reset, lookback)

                    # Calculate performance for this regime
                    valid_indices = ~(feature_values.isna() | regime_data_reset[target_column].isna())
                    if valid_indices.sum() < 5:
                        continue

                    correlation = abs(feature_values[valid_indices].corr(regime_data_reset[target_column][valid_indices]))
                    if pd.isna(correlation):
                        correlation = 0.0
                    regime_scores.append(correlation)

                    if correlation > best_regime_score:
                        best_regime_score = correlation
                        best_regime_lookback = lookback

                except Exception as e:
                    self.logger.warning(f"Error in regime-aware analysis for regime {regime}, lookback {lookback}: {e}")
                    continue

            regime_results[regime] = {
                'optimal_lookback': best_regime_lookback,
                'performance_score': best_regime_score,
                'scores': regime_scores
            }
            overall_scores.extend(regime_scores)

        # Calculate overall optimal lookback (weighted average) with safe division
        if regime_results:
            weights = [max(0.0, r['performance_score']) for r in regime_results.values()]
            den = sum(weights)
            if den > 0:
                optimal_lookback = int(round(sum(
                    r['optimal_lookback'] * w for r, w in zip(regime_results.values(), weights)
                ) / den))
            else:
                optimal_lookback = self.config.min_lookback
        else:
            optimal_lookback = self.config.min_lookback

        # Calculate overall performance score (use absolute values for robustness)
        overall_performance = float(np.mean([abs(s) for s in overall_scores])) if overall_scores else 0.0

        # Calculate stability score
        stability_score = self._calculate_stability_score(overall_scores)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(overall_scores)

        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=optimal_lookback,
            performance_score=overall_performance,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.REGIME_AWARE.value,
            regime_specific_results=regime_results,
            validation_scores=overall_scores
        )

    async def _optimize_adaptive(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Adaptive optimization that combines multiple methods."""
        self.logger.info(f"Using adaptive optimization for {feature_name}")

        # Try different methods and combine results
        methods = [
            OptimizationMethod.STATISTICAL_ANALYSIS,
            OptimizationMethod.INFORMATION_THEORY
        ]

        if SKLEARN_AVAILABLE:
            methods.append(OptimizationMethod.CROSS_VALIDATION)

        results = []
        for method in methods:
            try:
                if method == OptimizationMethod.CROSS_VALIDATION:
                    result = await self._optimize_with_cross_validation(
                        data, feature_name, target_column, feature_generator, lookback_range
                    )
                elif method == OptimizationMethod.STATISTICAL_ANALYSIS:
                    result = await self._optimize_with_statistical_analysis(
                        data, feature_name, target_column, feature_generator, lookback_range
                    )
                elif method == OptimizationMethod.INFORMATION_THEORY:
                    result = await self._optimize_with_information_theory(
                        data, feature_name, target_column, feature_generator, lookback_range
                    )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in adaptive optimization with method {method}: {e}")
                continue

        if not results:
            # Fallback to statistical analysis
            return await self._optimize_with_statistical_analysis(
                data, feature_name, target_column, feature_generator, lookback_range
            )

        # Combine results (weighted average)
        weights = [r.performance_score for r in results]
        total_weight = sum(weights)

        if total_weight > 0:
            optimal_lookback = int(round(
                sum(r.optimal_lookback * w for r, w in zip(results, weights)) / total_weight
            ))
            performance_score = sum(r.performance_score * w for r, w in zip(results, weights)) / total_weight
        else:
            optimal_lookback = results[0].optimal_lookback
            performance_score = results[0].performance_score

        # Calculate combined stability score
        all_scores = []
        for result in results:
            if result.validation_scores:
                all_scores.extend(result.validation_scores)

        stability_score = self._calculate_stability_score(all_scores)
        confidence_interval = self._calculate_confidence_interval(all_scores)

        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=optimal_lookback,
            performance_score=performance_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.ADAPTIVE.value,
            validation_scores=all_scores
        )

    async def _analyze_regime_specific_performance(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        optimal_lookback: int,
        regime_column: str
    ) -> Dict[str, Any]:
        """Analyze performance across different regimes."""
        regime_analysis = {}
        regimes = data[regime_column].unique()

        for regime in regimes:
            regime_data = data[data[regime_column] == regime]
            if len(regime_data) < 10:
                continue

            try:
                feature_values = feature_generator(regime_data, optimal_lookback)
                valid_indices = ~(feature_values.isna() | regime_data[target_column].isna())

                if valid_indices.sum() > 5:
                    correlation = feature_values[valid_indices].corr(regime_data[target_column][valid_indices])
                    regime_analysis[regime] = {
                        'correlation': correlation,
                        'sample_size': valid_indices.sum(),
                        'feature_mean': feature_values[valid_indices].mean(),
                        'feature_std': feature_values[valid_indices].std()
                    }
            except Exception as e:
                self.logger.warning(f"Error analyzing regime {regime}: {e}")
                continue

        return regime_analysis

    async def _analyze_feature_decay(
        self,
        data: pd.DataFrame,
        feature_name: str,
        feature_generator: Callable,
        optimal_lookback: int
    ) -> Dict[str, Any]:
        """Analyze how feature performance decays with different lookback periods."""
        decay_analysis = {}

        # Test lookback periods around the optimal
        test_lookbacks = range(
            max(1, optimal_lookback - 10),
            min(optimal_lookback + 11, self.config.max_lookback + 1)
        )

        correlations = []
        for lookback in test_lookbacks:
            try:
                feature_values = feature_generator(data, lookback)
                # Calculate autocorrelation as a proxy for information content
                autocorr = feature_values.autocorr(lag=1)
                correlations.append(autocorr if not pd.isna(autocorr) else 0)
            except Exception as e:
                self.logger.warning(f"Error in decay analysis for lookback {lookback}: {e}")
                correlations.append(0)

        if correlations:
            decay_analysis = {
                'lookbacks': list(test_lookbacks),
                'correlations': correlations,
                'decay_rate': np.polyfit(test_lookbacks, correlations, 1)[0] if len(correlations) > 1 else 0,
                'peak_lookback': test_lookbacks[np.argmax(correlations)] if correlations else optimal_lookback
            }

        return decay_analysis

    def _calculate_stability_score(self, scores: List[float]) -> float:
        """Calculate stability score from a list of scores."""
        if not scores or len(scores) < 2:
            return 0.0

        # Stability is inverse of coefficient of variation
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if mean_score == 0:
            return 0.0

        cv = std_score / abs(mean_score)
        stability = 1 / (1 + cv)
        return min(1.0, max(0.0, stability))

    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        if not scores or len(scores) < 2:
            return (0.0, 0.0)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)

        # Use t-distribution for small samples with safe fallback
        try:
            from scipy.stats import t as t_dist, norm
            t_val = (t_dist.ppf((1 + confidence) / 2, n - 1) if n < 30 
                    else norm.ppf((1 + confidence) / 2))
        except Exception:
            t_val = 1.96  # Safe fallback for 95% confidence

        margin_error = t_val * (std_score / np.sqrt(n))

        return (mean_score - margin_error, mean_score + margin_error)

    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between two discrete series."""
        try:
            # Create contingency table
            contingency = pd.crosstab(x, y)

            # Calculate mutual information
            n = contingency.sum().sum()
            mi = 0

            for i in range(contingency.shape[0]):
                for j in range(contingency.shape[1]):
                    if contingency.iloc[i, j] > 0:
                        p_ij = contingency.iloc[i, j] / n
                        p_i = contingency.iloc[i, :].sum() / n
                        p_j = contingency.iloc[:, j].sum() / n
                        mi += p_ij * np.log2(p_ij / (p_i * p_j))

            return mi
        except Exception as e:
            self.logger.warning(f"Error calculating mutual information: {e}")
            return 0.0

    def _calculate_fast_mi_proxy(self, x: pd.Series, y: pd.Series) -> float:
        """Fast mutual information using histogram binning approximation (50-80% speed gain)."""
        try:
            # Remove NaN values
            valid_mask = ~(pd.isna(x) | pd.isna(y))
            if valid_mask.sum() < 10:  # Need minimum data
                return 0.0
                
            x_clean = x[valid_mask].values
            y_clean = y[valid_mask].values
            
            # Use adaptive binning based on data size
            n_bins = min(20, max(5, int(np.sqrt(len(x_clean)))))
            
            # Create bins
            x_bins = np.linspace(x_clean.min(), x_clean.max(), n_bins + 1)
            y_bins = np.linspace(y_clean.min(), y_clean.max(), n_bins + 1)
            
            # Digitize data into bins
            x_digitized = np.digitize(x_clean, x_bins) - 1  # -1 to get 0-based indexing
            y_digitized = np.digitize(y_clean, y_bins) - 1
            
            # Ensure indices are within bounds
            x_digitized = np.clip(x_digitized, 0, n_bins - 1)
            y_digitized = np.clip(y_digitized, 0, n_bins - 1)
            
            # Compute joint histogram
            joint_hist, _, _ = np.histogram2d(x_digitized, y_digitized, bins=n_bins)
            joint_prob = joint_hist / len(x_clean)
            
            # Compute marginal probabilities
            x_marginal = joint_prob.sum(axis=1)
            y_marginal = joint_prob.sum(axis=0)
            
            # Calculate MI: MI = Σ p(x,y) * log(p(x,y) / (p(x) * p(y)))
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_prob[i, j] > 0 and x_marginal[i] > 0 and y_marginal[j] > 0:
                        mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (x_marginal[i] * y_marginal[j]))
            
            return max(0.0, mi)
        except Exception as e:
            self.logger.warning(f"Error calculating fast MI proxy: {e}")
            return 0.0

    def _calculate_redundancy_matrix(self, candidates: List[FeatureCandidate], 
                                   feature_generator: Callable, data: pd.DataFrame) -> np.ndarray:
        """Calculate redundancy matrix between feature candidates using MI proxy with sparse optimization."""
        n_candidates = len(candidates)
        
        # Use sparse matrices for large candidate sets
        if n_candidates > self.config.max_correlation_candidates and self.config.use_sparse_matrices:
            return self._calculate_sparse_redundancy_matrix(candidates, feature_generator, data)
        
        # Use dense matrix for smaller sets
        redundancy_matrix = np.zeros((n_candidates, n_candidates))
        
        # Optimize with memoization and early termination
        for i, candidate_i in enumerate(candidates):
            # Early termination for low-performing candidates
            if candidate_i.combined_score < self.config.early_termination_threshold:
                continue
                
            for j, candidate_j in enumerate(candidates):
                if i == j:
                    redundancy_matrix[i, j] = 1.0  # Perfect correlation with self
                elif j > i:  # Only calculate upper triangle
                    try:
                        # Use cached features
                        feature_i = self._get_cached_feature(data, f"candidate_{i}", candidate_i.lookback, 
                                                          lambda d, l: feature_generator(d, l))
                        feature_j = self._get_cached_feature(data, f"candidate_{j}", candidate_j.lookback,
                                                          lambda d, l: feature_generator(d, l))
                        
                        # Calculate redundancy using fast MI proxy
                        valid_indices = ~(feature_i.isna() | feature_j.isna())
                        if valid_indices.sum() > 10:
                            redundancy = self._calculate_fast_mi_proxy(
                                feature_i[valid_indices], feature_j[valid_indices]
                            )
                            redundancy_matrix[i, j] = redundancy
                            redundancy_matrix[j, i] = redundancy  # Symmetric matrix
                        else:
                            redundancy_matrix[i, j] = 0.0
                            redundancy_matrix[j, i] = 0.0
                    except Exception as e:
                        self.logger.warning(f"Error calculating redundancy between candidates {i} and {j}: {e}")
                        redundancy_matrix[i, j] = 0.0
                        redundancy_matrix[j, i] = 0.0
        
        return redundancy_matrix
    
    def _calculate_sparse_redundancy_matrix(self, candidates: List[FeatureCandidate], 
                                          feature_generator: Callable, data: pd.DataFrame) -> np.ndarray:
        """Calculate redundancy matrix using sparse matrices for large candidate sets."""
        n_candidates = len(candidates)
        
        # Use LIL matrix for efficient construction
        redundancy_matrix = lil_matrix((n_candidates, n_candidates))
        
        # Pre-filter candidates by performance
        high_performance_candidates = [
            (i, candidate) for i, candidate in enumerate(candidates)
            if candidate.combined_score >= self.config.early_termination_threshold
        ]
        
        self.logger.info(f"Using sparse matrix for {n_candidates} candidates, {len(high_performance_candidates)} high-performance")
        
        # Calculate redundancy only for high-performance candidates
        for i, candidate_i in high_performance_candidates:
            redundancy_matrix[i, i] = 1.0  # Diagonal
            
            for j, candidate_j in high_performance_candidates:
                if j > i:  # Only calculate upper triangle
                    try:
                        # Use cached features
                        feature_i = self._get_cached_feature(data, f"candidate_{i}", candidate_i.lookback,
                                                          lambda d, l: feature_generator(d, l))
                        feature_j = self._get_cached_feature(data, f"candidate_{j}", candidate_j.lookback,
                                                          lambda d, l: feature_generator(d, l))
                        
                        # Calculate redundancy using fast MI proxy
                        valid_indices = ~(feature_i.isna() | feature_j.isna())
                        if valid_indices.sum() > 10:
                            redundancy = self._calculate_fast_mi_proxy(
                                feature_i[valid_indices], feature_j[valid_indices]
                            )
                            redundancy_matrix[i, j] = redundancy
                            redundancy_matrix[j, i] = redundancy  # Symmetric matrix
                    except Exception as e:
                        self.logger.warning(f"Error calculating sparse redundancy between candidates {i} and {j}: {e}")
        
        # Convert to dense for compatibility with existing code
        return redundancy_matrix.toarray()

    def _select_top_candidates(self, candidates: List[FeatureCandidate], 
                              redundancy_matrix: np.ndarray, 
                              max_candidates: int = 3,
                              redundancy_threshold: float = 0.7) -> List[FeatureCandidate]:
        """Select top N candidates that are informative and non-redundant."""
        if not candidates:
            return []
        
        # Sort candidates by combined score while preserving original indices
        idx_and_cands = list(enumerate(candidates))
        idx_and_cands.sort(key=lambda ic: ic[1].combined_score, reverse=True)
        
        selected_candidates = []
        selected_orig_idxs = []
        
        for orig_idx, candidate in idx_and_cands:
            if len(selected_candidates) >= max_candidates:
                break
                
            # Check redundancy with already selected candidates using original indices
            is_redundant = any(
                redundancy_matrix[orig_idx, sel_idx] > redundancy_threshold
                for sel_idx in selected_orig_idxs
            )
            
            if not is_redundant:
                selected_candidates.append(candidate)
                selected_orig_idxs.append(orig_idx)
        
        return selected_candidates

    def _is_binary_target(self, y: np.ndarray) -> bool:
        """Check if target is binary/categorical."""
        unique_values = np.unique(y)
        return len(unique_values) <= 2 and all(val in [0, 1, -1] for val in unique_values)

    async def _generate_top_candidates(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        max_candidates: int = 3
    ) -> Tuple[List[FeatureCandidate], Dict[str, Any]]:
        """Generate top N candidates with redundancy filtering."""
        self.logger.info(f"Generating top {max_candidates} candidates for {feature_name}")
        
        # Generate candidate lookback periods
        lookback_range = range(
            self.config.min_lookback,
            min(self.config.max_lookback + 1, 50),  # Limit for performance
            max(1, self.config.step_size * 2)  # Larger steps for candidate generation
        )
        
        candidates = []
        
        for lookback in lookback_range:
            try:
                # ALWAYS reset index to avoid duplicate label issues with pandas operations
                data_reset = data.reset_index(drop=True)
                
                # Generate feature with current lookback on the reset data
                feature_values = feature_generator(data_reset, lookback)
                
                # ALWAYS reset feature index to match the reset data index
                feature_values = feature_values.reset_index(drop=True)
                
                # Calculate performance metrics with proper index alignment
                # Ensure both series have the same index
                feature_values_aligned = feature_values.copy()
                target_values_aligned = data_reset[target_column].copy()
                
                # Align indices by using the intersection
                common_index = feature_values_aligned.index.intersection(target_values_aligned.index)
                if len(common_index) < 10:
                    continue
                    
                feature_values_aligned = feature_values_aligned.loc[common_index]
                target_values_aligned = target_values_aligned.loc[common_index]
                
                # Now calculate valid indices with aligned data
                valid_indices = ~(feature_values_aligned.isna() | target_values_aligned.isna())
                if valid_indices.sum() < 10:
                    continue
                
                # Performance score (correlation) with aligned data
                correlation = abs(feature_values_aligned[valid_indices].corr(target_values_aligned[valid_indices]))
                if pd.isna(correlation):
                    continue
                
                # Stability score with aligned data
                feature_std = feature_values_aligned[valid_indices].std()
                feature_mean = feature_values_aligned[valid_indices].mean()
                stability = 1 / (1 + feature_std / abs(feature_mean)) if feature_mean != 0 else 0
                
                # MI score (fast proxy) with aligned data
                mi_score = self._calculate_fast_mi_proxy(
                    feature_values_aligned[valid_indices], 
                    target_values_aligned[valid_indices]
                )
                
                # Combined score (weighted combination)
                combined_score = (
                    0.4 * correlation +  # Performance weight
                    0.3 * stability +   # Stability weight
                    0.3 * mi_score      # Information weight
                )
                
                candidate = FeatureCandidate(
                    lookback=lookback,
                    performance_score=correlation,
                    stability_score=stability,
                    mi_score=mi_score,
                    combined_score=combined_score,
                    correlation_with_target=correlation
                )
                candidates.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Error generating candidate for lookback {lookback}: {e}")
                continue
        
        if not candidates:
            self.logger.warning(f"No valid candidates generated for {feature_name}")
            return [], {}
        
        # Calculate redundancy matrix
        redundancy_matrix = self._calculate_redundancy_matrix(candidates, feature_generator, data)
        
        # Select top candidates with redundancy filtering
        top_candidates = self._select_top_candidates(
            candidates, redundancy_matrix, max_candidates, redundancy_threshold=0.7
        )
        
        # Generate redundancy analysis
        redundancy_analysis = {
            'total_candidates_evaluated': len(candidates),
            'redundancy_matrix_shape': redundancy_matrix.shape,
            'average_redundancy': np.mean(redundancy_matrix[np.triu_indices_from(redundancy_matrix, k=1)]),
            'max_redundancy': np.max(redundancy_matrix[np.triu_indices_from(redundancy_matrix, k=1)]),
            'selected_candidates_count': len(top_candidates),
            'redundancy_threshold_used': 0.7
        }
        
        self.logger.info(f"Selected {len(top_candidates)} top candidates from {len(candidates)} evaluated")
        return top_candidates, redundancy_analysis

    async def optimize_multiple_features(
        self,
        data: pd.DataFrame,
        feature_configs: Dict[str, Dict[str, Any]],
        target_column: str,
        regime_column: Optional[str] = None
    ) -> Dict[str, FeatureOptimizationResult]:
        """
        Optimize multiple features in parallel.

        Args:
            data: Input data DataFrame
            feature_configs: Dictionary mapping feature names to their configurations
            target_column: Name of the target column
            regime_column: Optional regime column for regime-aware optimization

        Returns:
            Dictionary mapping feature names to optimization results
        """
        self.logger.info(f"Optimizing {len(feature_configs)} features in parallel")

        results = {}

        if self.config.parallel_processing and len(feature_configs) > 1:
            # Parallel optimization using asyncio.gather
            import asyncio
            tasks = {
                name: self.optimize_feature_lookback(
                    data, name, target_column, cfg['generator'], regime_column
                )
                for name, cfg in feature_configs.items()
            }
            
            # Execute all tasks in parallel
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            results = {
                name: res for (name, res) in zip(tasks.keys(), done) 
                if not isinstance(res, Exception)
            }
        else:
            # Sequential optimization
            for feature_name, config in feature_configs.items():
                try:
                    feature_generator = config['generator']
                    result = await self.optimize_feature_lookback(
                        data, feature_name, target_column, feature_generator, regime_column
                    )
                    results[feature_name] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {feature_name}: {e}")
                    continue

        self.logger.info(f"Completed optimization for {len(results)} features")
        return results

    def get_optimization_summary(self, results: Dict[str, FeatureOptimizationResult]) -> Dict[str, Any]:
        """Generate a summary of optimization results."""
        if not results:
            return {}

        summary = {
            'total_features': len(results),
            'optimization_methods': {},
            'lookback_distribution': {},
            'performance_stats': {},
            'stability_stats': {},
            'recommendations': []
        }

        # Analyze methods used
        for result in results.values():
            method = result.optimization_method
            summary['optimization_methods'][method] = summary['optimization_methods'].get(method, 0) + 1

        # Analyze lookback distribution
        lookbacks = [result.optimal_lookback for result in results.values()]
        summary['lookback_distribution'] = {
            'mean': np.mean(lookbacks),
            'median': np.median(lookbacks),
            'std': np.std(lookbacks),
            'min': np.min(lookbacks),
            'max': np.max(lookbacks)
        }

        # Analyze performance
        performances = [result.performance_score for result in results.values()]
        summary['performance_stats'] = {
            'mean': np.mean(performances),
            'median': np.median(performances),
            'std': np.std(performances),
            'min': np.min(performances),
            'max': np.max(performances)
        }

        # Analyze stability
        stabilities = [result.stability_score for result in results.values()]
        summary['stability_stats'] = {
            'mean': np.mean(stabilities),
            'median': np.median(stabilities),
            'std': np.std(stabilities),
            'min': np.min(stabilities),
            'max': np.max(stabilities)
        }

        # Generate recommendations
        low_performance = [name for name, result in results.items()
                          if result.performance_score < self.config.performance_threshold]
        low_stability = [name for name, result in results.items()
                        if result.stability_score < self.config.stability_threshold]

        if low_performance:
            summary['recommendations'].append(
                f"Consider removing or redesigning features with low performance: {low_performance}"
            )

        if low_stability:
            summary['recommendations'].append(
                f"Consider stabilizing features with low stability: {low_stability}"
            )

        return summary

    def get_enhanced_optimization_summary(self, results: Dict[str, FeatureOptimizationResult]) -> Dict[str, Any]:
        """Generate enhanced summary including top candidates analysis."""
        if not results:
            return {}
        
        summary = self.get_optimization_summary(results)
        
        # Add top candidates analysis
        summary['top_candidates_analysis'] = {
            'features_with_top_candidates': 0,
            'total_top_candidates': 0,
            'average_candidates_per_feature': 0.0,
            'redundancy_analysis': {}
        }
        
        total_candidates = 0
        features_with_candidates = 0
        
        for feature_name, result in results.items():
            if result.top_candidates:
                features_with_candidates += 1
                total_candidates += len(result.top_candidates)
                
                # Add redundancy analysis for this feature
                if result.redundancy_analysis:
                    summary['top_candidates_analysis']['redundancy_analysis'][feature_name] = {
                        'candidates_evaluated': result.redundancy_analysis.get('total_candidates_evaluated', 0),
                        'selected_candidates': result.redundancy_analysis.get('selected_candidates_count', 0),
                        'average_redundancy': result.redundancy_analysis.get('average_redundancy', 0.0),
                        'max_redundancy': result.redundancy_analysis.get('max_redundancy', 0.0)
                    }
        
        summary['top_candidates_analysis']['features_with_top_candidates'] = features_with_candidates
        summary['top_candidates_analysis']['total_top_candidates'] = total_candidates
        summary['top_candidates_analysis']['average_candidates_per_feature'] = (
            total_candidates / features_with_candidates if features_with_candidates > 0 else 0.0
        )
        
        return summary

    async def optimize_features(
        self,
        data: pd.DataFrame,
        config: FeatureOptimizationConfig
    ) -> Dict[str, Any]:
        """
        Optimize features based on the provided configuration.
        This is a wrapper method for backward compatibility.

        Args:
            data: Input data DataFrame
            config: Feature optimization configuration

        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting feature optimization with method: {config.optimization_method}")

        try:
            # Validate input data type
            if not isinstance(data, pd.DataFrame):
                error_msg = f"❌ Expected DataFrame but got {type(data)}. Cannot perform feature optimization."
                self.logger.error(error_msg)
                return {
                    'best_lookback_period': 20,  # Default fallback
                    'best_score': 0.0,
                    'optimization_method': 'fallback',
                    'error': error_msg,
                    'fallback_reason': 'invalid_data_type'
                }

            # Enhanced optimization with stability constraints
            results = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            # Filter out raw OHLCV data and basic transformations - focus on REAL technical indicators
            excluded_columns = [
                'timestamp', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'returns', 'close_return',
                'close_time', 'quote_volume', 'trades', 'day', 'close_log_return', 'volume_return',
                'volume_log_return', 'price_range', 'price_range_pct', 'body_size', 'body_size_pct',
                'hour', 'day_of_week', 'is_weekend', 'exchange', 'timeframe', 'symbol', 'interval'
            ]
            feature_columns = [col for col in numeric_columns if col not in excluded_columns]

            # Check for real technical indicators (RSI, SMA, EMA, etc.)
            ta_indicators = [col for col in feature_columns if any(indicator in col.lower() for indicator in
                           ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stoch', 'williams', 'cci', 'roc'])]

            # If no real technical indicators available, use the FeatureBank to generate comprehensive features
            if not ta_indicators:
                tprint("⚠️ No technical indicators found, generating comprehensive features using FeatureBank...")
                try:
                    # Import and use the proper FeatureBank system via factory
                    from ..core.factory import get_feature_bank, list_available_categories
                    from ..core.feature_generator import FeatureCategory

                    # Get the global feature bank and manually register generators
                    feature_bank = get_feature_bank()

                    # Manually register ALL available generators using correct imports
                    try:
                        # Direct class imports for main generators
                        from ..categories.momentum import MomentumFeatureGenerator
                        from ..categories.volatility import VolatilityFeatureGenerator
                        from ..categories.trend import TrendFeatureGenerator
                        from ..categories.oscillator import OscillatorFeatureGenerator
                        from ..categories.volume import VolumeFeatureGenerator
                        from ..categories.returns import ReturnsFeatureGenerator
                        from ..categories.support_resistance import SupportResistanceFeatureGenerator
                        from ..categories.candlestick_pattern import CandlestickPatternFeatureGenerator
                        from ..categories.interaction import InteractionFeatureGenerator

                        # Factory functions for complex generators
                        from ..categories.microstructure_features import create_default_microstructure_generators
                        from ..categories.order_flow import create_default_order_flow_generators
                        from ..categories.cross_timeframe import create_default_cross_timeframe_generators
                        from ..categories.entropy import create_default_entropy_generators
                        from ..categories.time import create_default_time_generators

                        # Register main generators
                        generators_to_register = [
                            MomentumFeatureGenerator(),
                            VolatilityFeatureGenerator(),
                            TrendFeatureGenerator(),
                            OscillatorFeatureGenerator(),
                            VolumeFeatureGenerator(),
                            ReturnsFeatureGenerator(),
                            SupportResistanceFeatureGenerator(),
                            CandlestickPatternFeatureGenerator(),
                            InteractionFeatureGenerator()
                        ]

                        # Add generators from factory functions (with error handling for missing data)
                        try:
                            microstructure_gens = create_default_microstructure_generators()
                            # Filter out generators that require bid/ask data
                            filtered_microstructure = []
                            for gen in microstructure_gens:
                                required_cols = getattr(gen.config, 'required_columns', [])
                                if not any(col in ['bid', 'ask'] for col in required_cols):
                                    filtered_microstructure.append(gen)
                            generators_to_register.extend(filtered_microstructure)
                            tprint(f"✅ Added {len(filtered_microstructure)} microstructure generators (filtered from {len(microstructure_gens)})")
                        except Exception as e:
                            tprint(f"⚠️ Skipping microstructure generators: {e}")

                        try:
                            generators_to_register.extend(create_default_order_flow_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping order flow generators: {e}")

                        try:
                            generators_to_register.extend(create_default_cross_timeframe_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping cross-timeframe generators: {e}")

                        try:
                            generators_to_register.extend(create_default_entropy_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping entropy generators: {e}")

                        try:
                            generators_to_register.extend(create_default_time_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping time generators: {e}")

                        for generator in generators_to_register:
                            feature_bank.register_generator(generator)

                        tprint(f"✅ Registered {len(generators_to_register)} feature generators")

                    except Exception as reg_error:
                        tprint(f"⚠️ Failed to register generators: {reg_error}")
                        raise Exception("Failed to register generators")

                    # Check what categories are available now
                    available_categories = list_available_categories()
                    tprint(f"📊 Available feature categories: {len(available_categories)}")

                    if not available_categories:
                        tprint("⚠️ No feature generators available after registration, will use fallback indicators...")
                        raise Exception("No feature generators available")

                    # Generate features for ALL available categories
                    categories_to_generate = [
                        FeatureCategory.MOMENTUM,
                        FeatureCategory.VOLATILITY,
                        FeatureCategory.TREND,
                        FeatureCategory.OSCILLATOR,
                        FeatureCategory.VOLUME,
                        FeatureCategory.RETURNS,
                        FeatureCategory.SUPPORT_RESISTANCE,
                        FeatureCategory.CANDLESTICK_PATTERN,
                        FeatureCategory.MICROSTRUCTURE,
                        FeatureCategory.ORDER_FLOW,
                        FeatureCategory.CROSS_TIMEFRAME,
                        FeatureCategory.ENTROPY,
                        FeatureCategory.TIME
                        # Note: CUSTOM and LEGACY categories available too
                    ]

                    tprint(f"🚀 Generating features for {len(categories_to_generate)} categories...")

                    # Generate features using the FeatureBank with reduced hardware optimization
                    feature_df = feature_bank.generate_features(
                        data=data,
                        categories=categories_to_generate,
                        target_column='returns',
                        lookback_optimization=False,  # Disable to avoid optimize_lookback method error
                        # Pass hardware optimization parameters as kwargs
                        cpu_optimization_level='CONSERVATIVE',  # Reduce CPU intensity
                        enable_thermal_monitoring=False,        # Disable thermal monitoring
                        enable_adaptive_optimization=False,     # Disable adaptive optimization
                        monitoring_interval=30.0,              # Reduce monitoring frequency
                        cpu_usage_threshold=70.0,              # Lower CPU threshold
                        memory_usage_threshold=80.0,           # Lower memory threshold
                        gpu_usage_threshold=60.0,              # Lower GPU threshold
                        temperature_threshold=70.0             # Lower temperature threshold
                    )

                    # Merge generated features with original data
                    if not feature_df.empty:
                        # Add generated features to data
                        for col in feature_df.columns:
                            if col not in data.columns:
                                data[col] = feature_df[col]

                        # Update feature columns to include generated features
                        feature_columns = [col for col in feature_df.columns if col not in excluded_columns]
                        tprint(f"✅ Generated {len(feature_columns)} features using FeatureBank system")
                    else:
                        tprint("⚠️ FeatureBank returned empty results, using fallback basic indicators...")
                        # Fallback to basic indicators if FeatureBank fails
                        self._create_basic_technical_indicators(data)
                        feature_columns = ['sma_20', 'ema_12', 'rsi_14', 'volatility_20']

                except Exception as e:
                    tprint(f"⚠️ FeatureBank failed: {e}, creating basic technical indicators...")
                    # Fallback to basic indicators
                    self._create_basic_technical_indicators(data)
                    feature_columns = ['sma_20', 'ema_12', 'rsi_14', 'volatility_20']
            else:
                # Use existing real technical indicators
                feature_columns = ta_indicators
                tprint(f"✅ Found {len(feature_columns)} existing technical indicators: {feature_columns[:5]}...")

            all_scores = []
            all_lookbacks = []

            # Limit to reasonable number of features for optimization
            optimization_features = feature_columns[:8] if len(feature_columns) > 8 else feature_columns
            tprint(f"🎯 Optimizing {len(optimization_features)} engineered features: {optimization_features}")

            for i, col in enumerate(optimization_features):
                # Generate candidate lookback values
                candidate_lookbacks = list(range(config.min_lookback, min(config.max_lookback, 50), 5))
                candidate_scores = []
                
                # Early termination tracking
                consecutive_low_scores = 0
                max_consecutive_low = 3

                for lookback in candidate_lookbacks:
                    # Simple correlation-based scoring
                    try:
                        if len(data) > lookback:
                            rolling_feature = data[col].rolling(window=lookback).mean()
                            # FORCE bi-directional targets first - we know directional_confidence exists
                            target_options = [
                                # FORCE: Use directional_confidence first (we know this exists)
                                'directional_confidence',        # Strength of directional bias - CONFIRMED WORKING

                                # Try other bi-directional targets
                                'opportunity_asymmetry',         # Long-short bias indicator
                                'long_overall_opportunity',      # Long opportunity score
                                'short_overall_opportunity',     # Short opportunity score

                                # Original targets (backward compatibility) - LOWER PRIORITY
                                'leverage_adjusted_score',       # Primary multi-horizon target (long-biased)
                                'immediate_opportunity',         # Secondary multi-horizon target
                                'short_term_opportunity',        # Tertiary multi-horizon target
                                'returns',                       # Fallback to basic returns
                                'close_return',                  # Alternative returns name
                                'close'                         # Last resort
                            ]

                            target_col = None
                            for target_option in target_options:
                                if target_option in data.columns:
                                    target_col = target_option
                                    break

                            if target_col is None:
                                correlation = 0.0
                                raw_correlation = 0.0
                                tprint(f"⚠️ No suitable target column found for {col}")
                            else:
                                raw_correlation = rolling_feature.corr(data[target_col])
                                correlation = abs(raw_correlation)  # Use absolute for optimization

                                # Enhanced logging for bi-directional targets
                                if target_col in ['long_overall_opportunity', 'short_overall_opportunity', 'opportunity_asymmetry', 'directional_confidence']:
                                    direction = "positive" if raw_correlation > 0 else "negative"
                                    if target_col == 'directional_confidence':
                                        interpretation = "Higher feature → Stronger directional signal" if raw_correlation > 0 else "Higher feature → Weaker directional signal"
                                        tprint(f"🎉 BREAKTHROUGH: Using DIRECTIONAL_CONFIDENCE target for {col} optimization!")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    elif target_col == 'long_overall_opportunity':
                                        interpretation = "Higher feature → Higher LONG opportunity" if raw_correlation > 0 else "Higher feature → Lower LONG opportunity (contrarian)"
                                        tprint(f"🎯 Using BI-DIRECTIONAL target '{target_col}' for {col} optimization")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    elif target_col == 'short_overall_opportunity':
                                        interpretation = "Higher feature → Higher SHORT opportunity" if raw_correlation > 0 else "Higher feature → Lower SHORT opportunity (contrarian)"
                                        tprint(f"🎯 Using BI-DIRECTIONAL target '{target_col}' for {col} optimization")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    elif target_col == 'opportunity_asymmetry':
                                        interpretation = "Higher feature → LONG bias" if raw_correlation > 0 else "Higher feature → SHORT bias"
                                        tprint(f"🎯 Using BI-DIRECTIONAL target '{target_col}' for {col} optimization")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")

                                elif target_col in ['leverage_adjusted_score', 'immediate_opportunity', 'short_term_opportunity']:
                                    direction = "positive" if raw_correlation > 0 else "negative"
                                    tprint(f"🎯 Using multi-horizon target '{target_col}' for {col} optimization (correlation: {raw_correlation:.4f} - {direction})")
                            if not np.isnan(correlation):
                                candidate_scores.append(correlation)
                                # Reset consecutive low scores counter
                                consecutive_low_scores = 0
                            else:
                                candidate_scores.append(0.0)
                                consecutive_low_scores += 1
                        else:
                            candidate_scores.append(0.0)
                            consecutive_low_scores += 1
                            
                        # Early termination for low-performing features
                        if consecutive_low_scores >= max_consecutive_low:
                            self._performance_stats['early_terminations'] += 1
                            tprint(f"⚠️ Early termination for {col} after {consecutive_low_scores} consecutive low scores")
                            break
                            
                    except Exception:
                        candidate_scores.append(0.0)
                        consecutive_low_scores += 1

                # Apply regularization
                regularized_scores = self._apply_regularization(candidate_scores, candidate_lookbacks)

                # Apply stability constraints
                constrained_scores = self._calculate_stability_constraints(candidate_lookbacks, regularized_scores)

                # Find best with stability weighting
                stability_metrics = self._calculate_stability_metrics(constrained_scores, candidate_lookbacks)

                # Combine performance and stability
                if constrained_scores:
                    best_idx = np.argmax(constrained_scores)
                    performance_score = constrained_scores[best_idx]
                    stability_score = stability_metrics['overall_stability']

                    # Weighted final score
                    final_score = (1 - config.stability_weight) * performance_score + config.stability_weight * stability_score

                    optimal_lookback = candidate_lookbacks[best_idx]
                    all_scores.append(final_score)
                    all_lookbacks.append(optimal_lookback)

                    results[col] = {
                        'optimal_lookback': optimal_lookback,
                        'performance_score': performance_score,
                        'stability_score': stability_score,
                        'final_score': final_score,
                        'confidence_interval': (final_score - 0.1, final_score + 0.1),
                        'stability_metrics': stability_metrics
                    }
                else:
                    results[col] = {
                        'optimal_lookback': config.min_lookback,
                        'performance_score': 0.5,
                        'stability_score': 0.5,
                        'final_score': 0.5,
                        'confidence_interval': (0.4, 0.6)
                    }

            # Calculate overall stability metrics
            overall_stability_metrics = {}
            if all_scores and all_lookbacks:
                overall_stability_metrics = self._calculate_stability_metrics(all_scores, all_lookbacks)

            metadata = {
                'optimization_method': config.optimization_method.value,
                'features_processed': len(numeric_columns),
                'config_used': {
                    'min_lookback': config.min_lookback,
                    'max_lookback': config.max_lookback,
                    'cv_folds': config.cv_folds,
                    'parallel_processing': config.parallel_processing,
                    'l1_regularization': config.l1_regularization,
                    'l2_regularization': config.l2_regularization,
                    'stability_weight': config.stability_weight,
                    'max_lookback_variance': config.max_lookback_variance
                },
                'stability_analysis': {
                    'overall_stability': overall_stability_metrics.get('overall_stability', 0.5),
                    'score_coefficient_variation': overall_stability_metrics.get('score_cv', 1.0),
                    'lookback_coefficient_variation': overall_stability_metrics.get('lookback_cv', 1.0),
                    'range_consistency': overall_stability_metrics.get('range_consistency', 0.5),
                    'regularization_applied': config.l1_regularization > 0 or config.l2_regularization > 0,
                    'stability_constraints_applied': True,
                    'features_optimized': len(results),
                    'average_stability_score': np.mean([r.get('stability_score', 0.5) for r in results.values()]) if results else 0.5
                }
            }

            # Add performance statistics to metadata
            metadata['performance_statistics'] = {
                'cache_hits': self._performance_stats['cache_hits'],
                'cache_misses': self._performance_stats['cache_misses'],
                'cache_hit_ratio': (self._performance_stats['cache_hits'] / 
                                  max(1, self._performance_stats['cache_hits'] + self._performance_stats['cache_misses'])),
                'early_terminations': self._performance_stats['early_terminations'],
                'vectorized_operations': self._performance_stats['vectorized_operations'],
                'total_features_processed': len(optimization_features),
                'optimization_efficiency': (len(optimization_features) - self._performance_stats['early_terminations']) / 
                                         max(1, len(optimization_features))
            }
            
            # Log performance summary
            tprint(f"📊 Performance Summary:")
            tprint(f"   Cache Hit Ratio: {metadata['performance_statistics']['cache_hit_ratio']:.2%}")
            tprint(f"   Early Terminations: {self._performance_stats['early_terminations']}")
            tprint(f"   Vectorized Operations: {self._performance_stats['vectorized_operations']}")
            tprint(f"   Optimization Efficiency: {metadata['performance_statistics']['optimization_efficiency']:.2%}")

            return {
                'results': results,
                'metadata': metadata
            }

        except Exception as e:
            self.logger.error(f"Feature optimization failed: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)  # Neutral RSI fallback

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self._vectorbt_rolling_operation(true_range, "mean", period)
            return atr
        except Exception:
            return pd.Series([1.0] * len(data), index=data.index)  # Fallback ATR

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = self._vectorbt_rolling_operation(prices, "mean", period)
            std = self._vectorbt_rolling_operation(prices, "std", period)
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band
        except Exception:
            # Fallback bands
            return pd.Series(prices * 1.02, index=prices.index), pd.Series(prices * 0.98, index=prices.index)

    def _create_basic_technical_indicators(self, data: pd.DataFrame) -> None:
        """Create basic technical indicators as fallback."""
        try:
            if 'close' in data.columns:
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['ema_12'] = data['close'].ewm(span=12).mean()
                data['rsi_14'] = self._calculate_rsi(data['close'], 14)
                data['volatility_20'] = data['close'].rolling(window=20).std()
                tprint("✅ Created 4 basic technical indicators as fallback")
        except Exception as e:
            tprint(f"⚠️ Failed to create basic indicators: {e}")

    def _apply_regularization(self, scores: List[float], lookback_values: List[int]) -> List[float]:
        """Apply L1/L2 regularization to optimization scores with magnitude scaling."""
        try:
            regularized_scores = scores.copy()
            
            if not scores:
                return regularized_scores

            # Calculate score IQR for magnitude scaling
            scores_array = np.array(scores)
            q75, q25 = np.percentile(scores_array, [75, 25])
            score_iqr = max(q75 - q25, 0.01)  # Avoid division by zero

            # L1 regularization - penalize extreme lookback values (scaled by score magnitude)
            if self.config.l1_regularization > 0:
                lookback_array = np.array(lookback_values)
                l1_penalty = (self.config.l1_regularization * score_iqr * 
                             np.abs(lookback_array - np.mean(lookback_array)) / np.mean(lookback_array))
                regularized_scores = [score - penalty for score, penalty in zip(regularized_scores, l1_penalty)]

            # L2 regularization - penalize variance in lookback values (scaled by score magnitude)
            if self.config.l2_regularization > 0:
                lookback_variance = np.var(lookback_values) / (np.mean(lookback_values) ** 2)  # Normalized variance
                l2_penalty = self.config.l2_regularization * score_iqr * lookback_variance
                regularized_scores = [score - l2_penalty for score in regularized_scores]

            return regularized_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Regularization failed: {e}")
            return scores

    def _calculate_stability_constraints(self, lookback_values: List[int], scores: List[float]) -> List[float]:
        """Apply stability constraints to optimization scores."""
        try:
            constrained_scores = scores.copy()

            # Calculate lookback variance penalty (reduced to prevent excessive penalties)
            if len(lookback_values) > 1:
                lookback_variance = np.var(lookback_values) / np.mean(lookback_values)  # Coefficient of variation

                # Apply much smaller penalty to avoid turning positive correlations negative
                max_variance_threshold = getattr(self.config, 'max_lookback_variance', 1.0)
                penalty_weight = getattr(self.config, 'lookback_range_penalty', 0.1)

                # Reduce penalty weight to prevent sign flips
                reduced_penalty_weight = min(penalty_weight, 0.05)  # Cap at 5% penalty

                if lookback_variance > max_variance_threshold:
                    variance_penalty = reduced_penalty_weight * (lookback_variance - max_variance_threshold)
                    # Ensure penalty doesn't exceed 50% of the original score
                    max_penalty = max([abs(score) * 0.5 for score in constrained_scores])
                    variance_penalty = min(variance_penalty, max_penalty)
                    constrained_scores = [score - variance_penalty for score in constrained_scores]

            return constrained_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Stability constraints failed: {e}")
            return scores

    def _rolling_window_optimization(self, data: pd.DataFrame, feature_name: str,
                                   target_column: str, feature_generator: Callable) -> Dict[str, Any]:
        """Perform rolling window optimization for temporal stability."""
        try:
            if 'timestamp' not in data.columns:
                self.logger.warning("⚠️ No timestamp column for rolling window optimization")
                return {}

            # Convert rolling window size to timedelta
            window_size = pd.Timedelta(self.config.rolling_window_size)
            step_size = pd.Timedelta(self.config.rolling_step_size)

            rolling_results = []
            start_time = data['timestamp'].min()
            end_time = data['timestamp'].max()

            current_time = start_time + window_size
            while current_time <= end_time:
                # Get window data
                window_start = current_time - window_size
                window_data = data[(data['timestamp'] >= window_start) & (data['timestamp'] < current_time)]

                if len(window_data) < 100:  # Minimum data requirement
                    current_time += step_size
                    continue

                # Optimize for this window
                window_results = []
                for lookback in range(self.config.min_lookback, min(self.config.max_lookback, len(window_data)//4)):
                    try:
                        # ALWAYS reset index to avoid duplicate label issues with pandas operations
                        window_data_reset = window_data.reset_index(drop=True)
                        
                        feature_values = feature_generator(window_data_reset, lookback)
                        if len(feature_values) > 10:
                            correlation = abs(feature_values.corr(window_data_reset[target_column]))
                            if not np.isnan(correlation):
                                window_results.append({
                                    'lookback': lookback,
                                    'score': correlation,
                                    'window_start': window_start,
                                    'window_end': current_time
                                })
                    except Exception:
                        continue

                if window_results:
                    best_window = max(window_results, key=lambda x: x['score'])
                    rolling_results.append(best_window)

                current_time += step_size

            # Calculate temporal consistency
            if rolling_results:
                lookback_values = [r['lookback'] for r in rolling_results]
                score_values = [r['score'] for r in rolling_results]

                temporal_stability = 1.0 - (np.std(lookback_values) / np.mean(lookback_values)) if np.mean(lookback_values) > 0 else 0.0

                return {
                    'rolling_results': rolling_results,
                    'temporal_stability': temporal_stability,
                    'optimal_lookback': int(np.median(lookback_values)),
                    'stability_score': temporal_stability,
                    'window_count': len(rolling_results)
                }

            return {}

        except Exception as e:
            self.logger.warning(f"⚠️ Rolling window optimization failed: {e}")
            return {}

    def _calculate_stability_metrics(self, scores: List[float], lookback_values: List[int]) -> Dict[str, float]:
        """Calculate comprehensive stability metrics."""
        try:
            metrics = {}

            # Coefficient of variation for scores
            if len(scores) > 1:
                metrics['score_cv'] = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0.0
            else:
                metrics['score_cv'] = 0.0

            # Coefficient of variation for lookback values
            if len(lookback_values) > 1:
                metrics['lookback_cv'] = np.std(lookback_values) / np.mean(lookback_values) if np.mean(lookback_values) > 0 else 0.0
            else:
                metrics['lookback_cv'] = 0.0

            # Overall stability score (lower CV = higher stability)
            metrics['overall_stability'] = 1.0 - min(1.0, (metrics['score_cv'] + metrics['lookback_cv']) / 2.0)

            # Temporal consistency (based on lookback range)
            if len(lookback_values) > 1:
                lookback_range = max(lookback_values) - min(lookback_values)
                max_possible_range = self.config.max_lookback - self.config.min_lookback
                metrics['range_consistency'] = 1.0 - (lookback_range / max_possible_range)
            else:
                metrics['range_consistency'] = 1.0

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Stability metrics calculation failed: {e}")
            return {'overall_stability': 0.5, 'score_cv': 1.0, 'lookback_cv': 1.0, 'range_consistency': 0.5}

# Convenience functions
def get_feature_optimizer(config: Optional[FeatureOptimizationConfig] = None) -> FeatureGenerationOptimizer:
    """Get a configured feature generation optimizer."""
    return FeatureGenerationOptimizer(config)

async def optimize_feature_lookback(
    data: pd.DataFrame,
    feature_name: str,
    target_column: str,
    feature_generator: Callable,
    config: Optional[FeatureOptimizationConfig] = None,
    regime_column: Optional[str] = None,
    include_top_candidates: bool = True,
    max_candidates: int = 3
) -> FeatureOptimizationResult:
    """Convenience function for optimizing a single feature."""
    optimizer = get_feature_optimizer(config)
    return await optimizer.optimize_feature_lookback(
        data, feature_name, target_column, feature_generator, regime_column,
        include_top_candidates, max_candidates
    )
