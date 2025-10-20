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
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
        EnsembleValidator, ModelEnsemble, WeightedEnsemble
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    warnings.warn(f"ML Common utilities not available: {e}")
    # Fallback classes
    class BayesianTPEOptimizer:
        def __init__(self): pass
    class GridSearchOptimizer:
        def __init__(self): pass
    class HyperparameterOptimizer:
        def __init__(self): pass
    class OptunaOptimizer:
        def __init__(self): pass
    class TimeSeriesCrossValidator:
        def __init__(self): pass
    class RegimeAwareCrossValidator:
        def __init__(self): pass
    class WalkForwardValidator:
        def __init__(self): pass
    class PurgedCrossValidator:
        def __init__(self): pass
    class ModelValidator:
        def __init__(self): pass
    class PerformanceValidator:
        def __init__(self): pass
    class StabilityValidator:
        def __init__(self): pass
    class EnsembleValidator:
        def __init__(self): pass
    class ModelEnsemble:
        def __init__(self): pass
    class WeightedEnsemble:
        def __init__(self): pass

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
    # Fallback classes and functions
    class FeatureConfig:
        def __init__(self, **kwargs): pass

    class ConfigValidator:
        def __init__(self, verbose: bool = False): pass
        def validate_config(self, config): pass

    class BaseConfig:
        def __post_init__(self): pass

    def get_logger(name: str):
        return None

    def log_execution(func): return func

    def log_performance(func): return func

    class LoggingContext:
        def __init__(self, name: str, operation: str, verbose: bool = False): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    class MetricsCalculator:
        def __init__(self, verbose: bool = False): pass

    class CharacteristicsGenerator:
        def __init__(self, verbose: bool = False): pass

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
