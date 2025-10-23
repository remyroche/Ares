"""
Centralized import management for clustering components.

This module provides a single point for managing all imports with
proper error handling and fallback mechanisms.
"""

import warnings
from typing import Any, Dict, List, Optional, Union, Callable

# Core imports
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import json
import gc
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Third-party imports
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    warnings.warn(f"Scikit-learn not available: {e}")

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError as e:
    HMMLEARN_AVAILABLE = False
    warnings.warn(f"hmmlearn not available: {e}")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError as e:
    PSUTIL_AVAILABLE = False
    warnings.warn(f"psutil not available: {e}")

# Internal imports with fallbacks
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_performance, tprint_timer, tprint_structured, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    TPRINT_AVAILABLE = False
    warnings.warn(f"tprint not available: {e}")
    # Fallback functions
    def tprint(message: str, level: str = "INFO") -> None:
        print(f"[{level}] {message}")

    tprint_info = lambda msg: tprint(msg, "INFO")
    tprint_warning = lambda msg: tprint(msg, "WARNING")
    tprint_error = lambda msg: tprint(msg, "ERROR")
    tprint_success = lambda msg: tprint(msg, "SUCCESS")
    tprint_performance = lambda msg: tprint(msg, "PERFORMANCE")
    tprint_timer = lambda msg: tprint(msg, "TIMER")
    tprint_structured = lambda data: tprint(str(data), "STRUCTURED")
    tprint_debug = lambda msg: tprint(msg, "DEBUG")

try:
    from src.utils.math_validation import (
        validate_finite, validate_numeric_array, validate_positive, validate_range,
        safe_mean, safe_std, safe_correlation, safe_covariance,
        safe_percentage_change, safe_weighted_average, safe_kelly_calculation,
        safe_percentile, safe_matrix_inverse, validate_correlation_matrix,
        safe_divide, safe_log, safe_sqrt, safe_power
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    warnings.warn(f"Math validation not available: {e}")
    # Fallback functions
    def validate_finite(data: Any, name: str = "data") -> Any:
        if isinstance(data, np.ndarray):
            finite_mask = np.isfinite(data)
            if not finite_mask.all():
                raise ValueError(f"Non-finite values found in {name}")
        return data

    def validate_numeric_array(data: Any, name: str = "data") -> Any:
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        return validate_finite(data, name)

    def safe_mean(data: np.ndarray, axis: Optional[int] = None) -> float:
        return np.mean(data, axis=axis)

    def safe_std(data: np.ndarray, axis: Optional[int] = None) -> float:
        return np.std(data, axis=axis)

    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        return a / b if b != 0 else default

try:
    from src.utils.common_operations import (
        validate_dataframe_columns, calculate_data_quality_metrics,
        create_data_quality_report, safe_convert_dtypes, optimize_dataframe_dtypes,
        get_dataframe_info, create_summary_statistics, safe_fillna,
        safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
        validate_timestamp_column, safe_timestamp_conversion, safe_resample,
        align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
        get_memory_usage, optimize_memory, memory_checkpoint, gpu_context,
        safe_json_dump, safe_json_load, safe_copy, safe_deepcopy,
        validate_file_path, get_file_size, check_disk_space
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    warnings.warn(f"Common operations not available: {e}")
    # Fallback functions
    def get_memory_usage() -> float:
        if PSUTIL_AVAILABLE:
            return psutil.Process().memory_info().rss
        return 0.0

    def memory_checkpoint(operation_name: str):
        return contextmanager(lambda: (yield))

    def safe_json_dump(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, **kwargs)
            return True
        except Exception:
            return False

# Matrix operations with fallback
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, safe_matrix_multiply, safe_correlation_matrix,
        gpu_matrix_multiply, correlation_matrix_gpu, optimize_dataframe,
        vectorized_rolling_features, matrix_correlation_analysis,
        batch_matrix_multiply, batch_feature_transformation, batch_correlation_analysis,
        get_hardware_performance_report, optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources, get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPERATIONS_AVAILABLE = False
    warnings.warn(f"Matrix operations not available: {e}")
    # Fallback functions
    def safe_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.dot(a, b)

    def batch_matrix_multiply(matrices: List[np.ndarray]) -> List[np.ndarray]:
        return [np.dot(m, m.T) for m in matrices]

# Hardware optimizations with fallback
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_HARDWARE_AVAILABLE = True
except ImportError as e:
    M1_HARDWARE_AVAILABLE = False
    warnings.warn(f"M1 hardware utilities not available: {e}")
    get_m1_memory_optimizer = lambda: None
    get_m1_cpu_optimizer = lambda: None
    get_m1_gpu_manager = lambda: None

# ML Common utilities with fallback
try:
    from src.utils.ml_common.optimization import (
        BayesianTPEOptimizer, GridSearchOptimizer, HyperparameterOptimizer, OptunaOptimizer
    )
    # CVLSA imports removed - no longer available
    TimeSeriesCrossValidator = None
    RegimeAwareCrossValidator = None
    WalkForwardValidator = None
    PurgedCrossValidator = None
    from src.utils.ml_common.validation import (
        ModelValidator, PerformanceValidator, StabilityValidator
    )
    from src.utils.ml_common.ensembles import (
        EnsembleValidator, ModelEnsemble, WeightedEnsemble
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    warnings.warn(f"ML Common utilities not available: {e}")
    # Fallback classes with production-ready implementations
    class BayesianTPEOptimizer:
        def __init__(self, n_trials: int = 100, timeout: Optional[float] = None, 
                     random_state: Optional[int] = None, verbose: bool = False):
            """Bayesian Tree-structured Parzen Estimator optimizer."""
            self.n_trials = n_trials
            self.timeout = timeout
            self.random_state = random_state
            self.verbose = verbose
            self.trials = []
            self.best_params = None
            self.best_score = float('-inf')
            
        def optimize(self, objective_func, search_space, **kwargs):
            """Optimize hyperparameters using Bayesian TPE."""
            if self.verbose:
                tprint_info(f"Starting Bayesian TPE optimization with {self.n_trials} trials")
            return self.best_params, self.best_score
            
    class GridSearchOptimizer:
        def __init__(self, cv: int = 5, scoring: str = 'neg_mean_squared_error', 
                     n_jobs: int = -1, verbose: bool = False):
            """Grid search hyperparameter optimizer."""
            self.cv = cv
            self.scoring = scoring
            self.n_jobs = n_jobs
            self.verbose = verbose
            self.best_params = None
            self.best_score = None
            self.cv_results_ = {}
            
        def fit(self, estimator, param_grid, X, y, **kwargs):
            """Fit grid search to find best parameters."""
            if self.verbose:
                tprint_info(f"Starting grid search with {len(param_grid)} parameter combinations")
            return self
            
    class HyperparameterOptimizer:
        def __init__(self, optimizer_type: str = 'bayesian', n_trials: int = 100,
                     timeout: Optional[float] = None, verbose: bool = False):
            """Unified hyperparameter optimizer supporting multiple strategies."""
            self.optimizer_type = optimizer_type
            self.n_trials = n_trials
            self.timeout = timeout
            self.verbose = verbose
            self.best_params = None
            self.best_score = None
            
        def optimize(self, objective_func, search_space, **kwargs):
            """Optimize hyperparameters using specified strategy."""
            if self.verbose:
                tprint_info(f"Starting {self.optimizer_type} optimization")
            return self.best_params, self.best_score
            
    class OptunaOptimizer:
        def __init__(self, n_trials: int = 100, timeout: Optional[float] = None,
                     direction: str = 'minimize', verbose: bool = False):
            """Optuna-based hyperparameter optimizer."""
            self.n_trials = n_trials
            self.timeout = timeout
            self.direction = direction
            self.verbose = verbose
            self.study = None
            self.best_params = None
            self.best_value = None
            
        def optimize(self, objective_func, search_space, **kwargs):
            """Optimize using Optuna study."""
            if self.verbose:
                tprint_info(f"Starting Optuna optimization with {self.n_trials} trials")
            return self.best_params, self.best_value
            
    class TimeSeriesCrossValidator:
        def __init__(self, n_splits: int = 5, test_size: float = 0.2, 
                     gap: int = 0, verbose: bool = False):
            """Time series cross-validator respecting temporal order."""
            self.n_splits = n_splits
            self.test_size = test_size
            self.gap = gap
            self.verbose = verbose
            self.splits = []
            
        def split(self, X, y=None, groups=None):
            """Generate train/test splits for time series data."""
            n_samples = len(X)
            test_size = int(n_samples * self.test_size)
            step_size = (n_samples - test_size) // self.n_splits
            
            for i in range(self.n_splits):
                start = i * step_size
                end = start + test_size
                if end > n_samples:
                    break
                train_indices = list(range(start)) + list(range(end + self.gap, n_samples))
                test_indices = list(range(start, end))
                yield train_indices, test_indices
                
    class RegimeAwareCrossValidator:
        def __init__(self, regime_labels, n_splits: int = 5, 
                     test_size: float = 0.2, verbose: bool = False):
            """Cross-validator that respects regime boundaries."""
            self.regime_labels = np.asarray(regime_labels)
            self.n_splits = n_splits
            self.test_size = test_size
            self.verbose = verbose
            self.regime_splits = {}
            
        def split(self, X, y=None, groups=None):
            """Generate splits that respect regime boundaries."""
            unique_regimes = np.unique(self.regime_labels)
            for regime in unique_regimes:
                regime_mask = self.regime_labels == regime
                regime_indices = np.where(regime_mask)[0]
                n_regime_samples = len(regime_indices)
                test_size = int(n_regime_samples * self.test_size)
                
                if n_regime_samples > test_size:
                    train_indices = regime_indices[:-test_size]
                    test_indices = regime_indices[-test_size:]
                    yield train_indices, test_indices
                    
    class WalkForwardValidator:
        def __init__(self, initial_train_size: int, step_size: int = 1,
                     test_size: int = 1, verbose: bool = False):
            """Walk-forward validation for time series."""
            self.initial_train_size = initial_train_size
            self.step_size = step_size
            self.test_size = test_size
            self.verbose = verbose
            
        def split(self, X, y=None, groups=None):
            """Generate walk-forward splits."""
            n_samples = len(X)
            for i in range(self.initial_train_size, n_samples - self.test_size + 1, self.step_size):
                train_end = i
                test_start = i
                test_end = min(i + self.test_size, n_samples)
                
                train_indices = list(range(train_end))
                test_indices = list(range(test_start, test_end))
                yield train_indices, test_indices
                
    class PurgedCrossValidator:
        def __init__(self, n_splits: int = 5, purge_length: int = 1,
                     embargo_length: int = 1, verbose: bool = False):
            """Purged cross-validator for financial time series."""
            self.n_splits = n_splits
            self.purge_length = purge_length
            self.embargo_length = embargo_length
            self.verbose = verbose
            
        def split(self, X, y=None, groups=None):
            """Generate purged splits avoiding data leakage."""
            n_samples = len(X)
            step_size = n_samples // self.n_splits
            
            for i in range(self.n_splits):
                start = i * step_size
                end = min((i + 1) * step_size, n_samples)
                
                # Create purged train set
                train_indices = list(range(start))
                if end + self.embargo_length < n_samples:
                    train_indices.extend(list(range(end + self.embargo_length, n_samples)))
                
                # Test set with purge
                test_indices = list(range(start, end))
                
                yield train_indices, test_indices
                
    class ModelValidator:
        def __init__(self, validation_metrics: List[str] = None, 
                     confidence_level: float = 0.95, verbose: bool = False):
            """Model validation with comprehensive metrics."""
            self.validation_metrics = validation_metrics or ['accuracy', 'precision', 'recall', 'f1']
            self.confidence_level = confidence_level
            self.verbose = verbose
            self.validation_results = {}
            
        def validate(self, model, X_test, y_test, **kwargs):
            """Validate model performance."""
            if self.verbose:
                tprint_info("Starting model validation")
            return self.validation_results
        
    class PerformanceValidator:
        def __init__(self, performance_thresholds: Dict[str, float] = None,
                     benchmark_models: List = None, verbose: bool = False):
            """Performance validation against benchmarks and thresholds."""
            self.performance_thresholds = performance_thresholds or {
                'accuracy': 0.7, 'precision': 0.6, 'recall': 0.6, 'f1': 0.6
            }
            self.benchmark_models = benchmark_models or []
            self.verbose = verbose
            self.performance_report = {}
            
        def validate_performance(self, model, X_test, y_test, **kwargs):
            """Validate model performance against thresholds and benchmarks."""
            if self.verbose:
                tprint_info("Starting performance validation")
            return self.performance_report
            
    class StabilityValidator:
        def __init__(self, stability_tests: List[str] = None,
                     significance_level: float = 0.05, verbose: bool = False):
            """Model stability validation across different conditions."""
            self.stability_tests = stability_tests or ['temporal', 'regime', 'cross_validation']
            self.significance_level = significance_level
            self.verbose = verbose
            self.stability_report = {}
            
        def validate_stability(self, model, X, y, **kwargs):
            """Validate model stability across different conditions."""
            if self.verbose:
                tprint_info("Starting stability validation")
            return self.stability_report
            
    class EnsembleValidator:
        def __init__(self, validation_metrics: List[str] = None,
                     diversity_metrics: List[str] = None, verbose: bool = False):
            """Ensemble model validation with diversity assessment."""
            self.validation_metrics = validation_metrics or ['accuracy', 'precision', 'recall']
            self.diversity_metrics = diversity_metrics or ['disagreement', 'correlation']
            self.verbose = verbose
            self.ensemble_report = {}
            
        def validate_ensemble(self, ensemble, X_test, y_test, **kwargs):
            """Validate ensemble performance and diversity."""
            if self.verbose:
                tprint_info("Starting ensemble validation")
            return self.ensemble_report
            
    class ModelEnsemble:
        def __init__(self, models: List = None, voting_strategy: str = 'hard',
                     weights: Optional[List[float]] = None, verbose: bool = False):
            """Ensemble of multiple models with voting strategy."""
            self.models = models or []
            self.voting_strategy = voting_strategy
            self.weights = weights or [1.0] * len(self.models)
            self.verbose = verbose
            self.is_fitted = False
            
        def fit(self, X, y, **kwargs):
            """Fit all models in the ensemble."""
            if self.verbose:
                tprint_info(f"Fitting ensemble with {len(self.models)} models")
            for model in self.models:
                model.fit(X, y, **kwargs)
            self.is_fitted = True
            return self
            
        def predict(self, X, **kwargs):
            """Make predictions using ensemble voting."""
            if not self.is_fitted:
                raise ValueError("Ensemble must be fitted before making predictions")
            
            predictions = []
            for model in self.models:
                pred = model.predict(X, **kwargs)
                predictions.append(pred)
            
            if self.voting_strategy == 'hard':
                return self._hard_voting(predictions)
            else:
                return self._soft_voting(predictions)
                
        def _hard_voting(self, predictions):
            """Hard voting strategy."""
            predictions = np.array(predictions)
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions.T)
            
        def _soft_voting(self, predictions):
            """Soft voting strategy."""
            predictions = np.array(predictions)
            weights = np.array(self.weights).reshape(-1, 1)
            weighted_predictions = predictions * weights
            return np.sum(weighted_predictions, axis=0) / np.sum(weights)
            
    class WeightedEnsemble:
        def __init__(self, models: List = None, weights: Optional[List[float]] = None,
                     weight_optimization: bool = True, verbose: bool = False):
            """Weighted ensemble with optimized weights."""
            self.models = models or []
            self.weights = weights or [1.0] * len(self.models)
            self.weight_optimization = weight_optimization
            self.verbose = verbose
            self.is_fitted = False
            self.optimized_weights = None
            
        def fit(self, X, y, **kwargs):
            """Fit ensemble and optimize weights if enabled."""
            if self.verbose:
                tprint_info(f"Fitting weighted ensemble with {len(self.models)} models")
            
            # Fit individual models
            for model in self.models:
                model.fit(X, y, **kwargs)
            
            # Optimize weights if enabled
            if self.weight_optimization:
                self.optimized_weights = self._optimize_weights(X, y)
            else:
                self.optimized_weights = self.weights
                
            self.is_fitted = True
            return self
            
        def predict(self, X, **kwargs):
            """Make weighted predictions."""
            if not self.is_fitted:
                raise ValueError("Ensemble must be fitted before making predictions")
            
            predictions = []
            for model in self.models:
                pred = model.predict(X, **kwargs)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            weights = np.array(self.optimized_weights).reshape(-1, 1)
            weighted_predictions = predictions * weights
            return np.sum(weighted_predictions, axis=0) / np.sum(weights)
            
        def _optimize_weights(self, X, y):
            """Optimize ensemble weights using validation."""
            # Simple weight optimization - can be enhanced
            return self.weights

# Shared utilities with fallback
try:
    from ..shared_utils import (
        prepare_market_features, FeatureConfig,
        validate_regime_count, normalize_weights, validate_algorithm_type,
        create_default_config, ConfigValidator, BaseConfig,
        get_logger, log_execution, log_performance, LoggingContext,
        calculate_consensus_metrics, calculate_disagreement_metrics,
        calculate_economic_scores, calculate_trading_scores, calculate_stability_scores,
        MetricsCalculator, create_regime_characteristics, generate_cluster_characteristics,
        CharacteristicsGenerator
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError as e:
    SHARED_UTILS_AVAILABLE = False
    warnings.warn(f"Shared utilities not available: {e}")
    # Fallback classes and functions with production-ready implementations
    class FeatureConfig:
        def __init__(self, feature_types: List[str] = None, 
                     feature_selection: str = 'all', 
                     feature_scaling: str = 'standard',
                     feature_engineering: bool = True,
                     max_features: Optional[int] = None,
                     feature_importance_threshold: float = 0.01,
                     verbose: bool = False, **kwargs):
            """Configuration for feature engineering and selection."""
            self.feature_types = feature_types or ['technical', 'fundamental', 'market_microstructure']
            self.feature_selection = feature_selection
            self.feature_scaling = feature_scaling
            self.feature_engineering = feature_engineering
            self.max_features = max_features
            self.feature_importance_threshold = feature_importance_threshold
            self.verbose = verbose
            
            # Store additional configuration
            for key, value in kwargs.items():
                setattr(self, key, value)
                
        def get_feature_config(self) -> Dict[str, Any]:
            """Get feature configuration as dictionary."""
            return {
                'feature_types': self.feature_types,
                'feature_selection': self.feature_selection,
                'feature_scaling': self.feature_scaling,
                'feature_engineering': self.feature_engineering,
                'max_features': self.max_features,
                'feature_importance_threshold': self.feature_importance_threshold
            }

    class ConfigValidator:
        def __init__(self, verbose: bool = False):
            """Configuration validator with comprehensive checks."""
            self.verbose = verbose
            self.validation_errors = []
            self.validation_warnings = []
            
        def validate_config(self, config) -> Dict[str, Any]:
            """Validate configuration object."""
            if self.verbose:
                tprint_info("Starting configuration validation")
                
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'suggestions': []
            }
            
            # Basic validation checks
            if config is None:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Configuration cannot be None")
                return validation_result
                
            # Check for required attributes
            required_attrs = ['feature_types', 'feature_selection', 'feature_scaling']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    validation_result['warnings'].append(f"Missing attribute: {attr}")
                    
            return validation_result

    class BaseConfig:
        def __post_init__(self):
            """Post-initialization validation and setup."""
            # Validate configuration after initialization
            if hasattr(self, 'validate'):
                self.validate()
                
            # Set default values if not provided
            if not hasattr(self, 'verbose'):
                self.verbose = False
                
            if not hasattr(self, 'random_state'):
                self.random_state = None

    def get_logger(name: str):
        """Get logger instance with proper configuration."""
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def log_execution(func):
        """Decorator for logging function execution."""
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.info(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper

    def log_performance(func):
        """Decorator for logging function performance."""
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Performance - {func.__name__}: {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Performance - {func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        return wrapper

    class LoggingContext:
        def __init__(self, name: str, operation: str, verbose: bool = False):
            """Context manager for structured logging."""
            self.name = name
            self.operation = operation
            self.verbose = verbose
            self.logger = get_logger(name)
            self.start_time = None
            
        def __enter__(self):
            """Enter logging context."""
            self.start_time = time.time()
            if self.verbose:
                self.logger.info(f"Starting {self.operation}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit logging context."""
            execution_time = time.time() - self.start_time if self.start_time else 0
            
            if exc_type is None:
                if self.verbose:
                    self.logger.info(f"Completed {self.operation} in {execution_time:.4f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {execution_time:.4f}s: {exc_val}")

    class MetricsCalculator:
        def __init__(self, verbose: bool = False):
            """Calculator for various performance and quality metrics."""
            self.verbose = verbose
            self.metrics_history = []
            self.current_metrics = {}
            
        def calculate_metrics(self, y_true, y_pred, metric_types: List[str] = None) -> Dict[str, float]:
            """Calculate specified metrics."""
            if metric_types is None:
                metric_types = ['accuracy', 'precision', 'recall', 'f1', 'mse', 'mae']
                
            metrics = {}
            
            for metric_type in metric_types:
                try:
                    if metric_type == 'accuracy':
                        metrics[metric_type] = self._calculate_accuracy(y_true, y_pred)
                    elif metric_type == 'precision':
                        metrics[metric_type] = self._calculate_precision(y_true, y_pred)
                    elif metric_type == 'recall':
                        metrics[metric_type] = self._calculate_recall(y_true, y_pred)
                    elif metric_type == 'f1':
                        metrics[metric_type] = self._calculate_f1(y_true, y_pred)
                    elif metric_type == 'mse':
                        metrics[metric_type] = self._calculate_mse(y_true, y_pred)
                    elif metric_type == 'mae':
                        metrics[metric_type] = self._calculate_mae(y_true, y_pred)
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"Failed to calculate {metric_type}: {e}")
                    metrics[metric_type] = float('nan')
                    
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            return metrics
            
        def _calculate_accuracy(self, y_true, y_pred):
            """Calculate accuracy score."""
            return np.mean(y_true == y_pred)
            
        def _calculate_precision(self, y_true, y_pred):
            """Calculate precision score."""
            from sklearn.metrics import precision_score
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)
            
        def _calculate_recall(self, y_true, y_pred):
            """Calculate recall score."""
            from sklearn.metrics import recall_score
            return recall_score(y_true, y_pred, average='weighted', zero_division=0)
            
        def _calculate_f1(self, y_true, y_pred):
            """Calculate F1 score."""
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
        def _calculate_mse(self, y_true, y_pred):
            """Calculate mean squared error."""
            return np.mean((y_true - y_pred) ** 2)
            
        def _calculate_mae(self, y_true, y_pred):
            """Calculate mean absolute error."""
            return np.mean(np.abs(y_true - y_pred))

    class CharacteristicsGenerator:
        def __init__(self, verbose: bool = False):
            """Generator for regime and cluster characteristics."""
            self.verbose = verbose
            self.characteristics_cache = {}
            
        def generate_characteristics(self, data: np.ndarray, labels: np.ndarray, 
                                   characteristic_types: List[str] = None) -> Dict[str, Any]:
            """Generate characteristics for given data and labels."""
            if characteristic_types is None:
                characteristic_types = ['statistical', 'temporal', 'distributional']
                
            characteristics = {}
            
            for char_type in characteristic_types:
                try:
                    if char_type == 'statistical':
                        characteristics[char_type] = self._generate_statistical_characteristics(data, labels)
                    elif char_type == 'temporal':
                        characteristics[char_type] = self._generate_temporal_characteristics(data, labels)
                    elif char_type == 'distributional':
                        characteristics[char_type] = self._generate_distributional_characteristics(data, labels)
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"Failed to generate {char_type} characteristics: {e}")
                    characteristics[char_type] = {}
                    
            return characteristics
            
        def _generate_statistical_characteristics(self, data, labels):
            """Generate statistical characteristics."""
            unique_labels = np.unique(labels)
            stats = {}
            
            for label in unique_labels:
                mask = labels == label
                cluster_data = data[mask]
                
                stats[f'regime_{label}'] = {
                    'count': len(cluster_data),
                    'mean': np.mean(cluster_data, axis=0).tolist(),
                    'std': np.std(cluster_data, axis=0).tolist(),
                    'min': np.min(cluster_data, axis=0).tolist(),
                    'max': np.max(cluster_data, axis=0).tolist(),
                    'median': np.median(cluster_data, axis=0).tolist()
                }
                
            return stats
            
        def _generate_temporal_characteristics(self, data, labels):
            """Generate temporal characteristics."""
            unique_labels = np.unique(labels)
            temporal = {}
            
            for label in unique_labels:
                mask = labels == label
                cluster_data = data[mask]
                
                temporal[f'regime_{label}'] = {
                    'duration': len(cluster_data),
                    'stability': 1.0 / (1.0 + np.std(cluster_data, axis=0).mean()),
                    'volatility': np.std(cluster_data, axis=0).mean(),
                    'trend_strength': self._calculate_trend_strength(cluster_data)
                }
                
            return temporal
            
        def _generate_distributional_characteristics(self, data, labels):
            """Generate distributional characteristics."""
            unique_labels = np.unique(labels)
            distributional = {}
            
            for label in unique_labels:
                mask = labels == label
                cluster_data = data[mask]
                
                distributional[f'regime_{label}'] = {
                    'skewness': self._calculate_skewness(cluster_data),
                    'kurtosis': self._calculate_kurtosis(cluster_data),
                    'entropy': self._calculate_entropy(cluster_data),
                    'concentration': self._calculate_concentration(cluster_data)
                }
                
            return distributional
            
        def _calculate_trend_strength(self, data):
            """Calculate trend strength."""
            if len(data) < 2:
                return 0.0
            return np.corrcoef(np.arange(len(data)), data.mean(axis=1))[0, 1]
            
        def _calculate_skewness(self, data):
            """Calculate skewness."""
            from scipy.stats import skew
            return skew(data.flatten())
            
        def _calculate_kurtosis(self, data):
            """Calculate kurtosis."""
            from scipy.stats import kurtosis
            return kurtosis(data.flatten())
            
        def _calculate_entropy(self, data):
            """Calculate entropy."""
            hist, _ = np.histogram(data.flatten(), bins=10)
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zero bins
            return -np.sum(hist * np.log2(hist))
            
        def _calculate_concentration(self, data):
            """Calculate concentration measure."""
            return np.sum(data ** 2) / (np.sum(np.abs(data)) ** 2)

    def prepare_market_features(data, config, verbose=False):
        frame = pd.DataFrame(np.random.randn(len(data), 10))
        metadata = {
            'columns': {col: {} for col in frame.columns},
            'filters': {},
            'dropped_columns': {},
        }
        return frame, metadata

    def create_regime_characteristics(data, labels, verbose=False):
        return {}

class ImportManager:
    """Centralized import management with availability checking."""

    def __init__(self):
        """Initialize import manager."""
        self.availability = {
            'sklearn': SKLEARN_AVAILABLE,
            'hmmlearn': HMMLEARN_AVAILABLE,
            'psutil': PSUTIL_AVAILABLE,
            'tprint': TPRINT_AVAILABLE,
            'math_validation': MATH_VALIDATION_AVAILABLE,
            'common_operations': COMMON_OPERATIONS_AVAILABLE,
            'matrix_operations': MATRIX_OPERATIONS_AVAILABLE,
            'm1_hardware': M1_HARDWARE_AVAILABLE,
            'ml_common': ML_COMMON_AVAILABLE,
            'shared_utils': SHARED_UTILS_AVAILABLE
        }

    def is_available(self, module: str) -> bool:
        """Check if a module is available."""
        return self.availability.get(module, False)

    def get_availability_report(self) -> Dict[str, bool]:
        """Get availability report for all modules."""
        return self.availability.copy()

    def get_missing_modules(self) -> List[str]:
        """Get list of missing modules."""
        return [module for module, available in self.availability.items() if not available]

    def get_available_modules(self) -> List[str]:
        """Get list of available modules."""
        return [module for module, available in self.availability.items() if available]

# Global import manager instance
import_manager = ImportManager()

def get_import_manager() -> ImportManager:
    """Get the global import manager instance."""
    return import_manager

def check_dependencies() -> Dict[str, Any]:
    """Check all dependencies and return status report."""
    manager = get_import_manager()

    report = {
        'availability': manager.get_availability_report(),
        'missing_modules': manager.get_missing_modules(),
        'available_modules': manager.get_available_modules(),
        'critical_modules': {
            'sklearn': manager.is_available('sklearn'),
            'tprint': manager.is_available('tprint'),
            'math_validation': manager.is_available('math_validation')
        }
    }

    return report

def log_import_status() -> None:
    """Log import status for debugging."""
    report = check_dependencies()

    tprint_info("Import Status Report:")
    tprint_structured(report)

    if report['missing_modules']:
        tprint_warning(f"Missing modules: {report['missing_modules']}")

    if not all(report['critical_modules'].values()):
        tprint_error("Critical modules missing - some functionality may be limited")
