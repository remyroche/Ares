"""
Unified Utilities for NAS and TAS Systems

This module consolidates all utility functions used by both
Neural Architecture Search (NAS) and Tree Architecture Search (TAS) systems.
It provides common functionality for data processing, validation, and operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import warnings
import json
import pickle
from pathlib import Path
from collections import OrderedDict
import sys
import threading
from typing import TYPE_CHECKING, cast

try:  # Optional dependency used for memory-aware caching
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - psutil is optional
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

# Lazily imported utilities to avoid circular dependencies
if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from src.utils.common_operations import optimize_dataframe_dtypes, get_memory_usage
    from src.utils.hardware import get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
else:
    optimize_dataframe_dtypes = None  # type: ignore
    get_memory_usage = None  # type: ignore
    get_m1_gpu_manager = None  # type: ignore
    get_m1_memory_optimizer = None  # type: ignore
    get_m1_cpu_optimizer = None  # type: ignore

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UnifiedUtilityOperationError(RuntimeError):
    """Raised when a protected operation executed by UnifiedUtilities fails."""


class UnifiedCacheWarning(UserWarning):
    """Warning emitted when cache pressure forces aggressive eviction."""


def _load_optimize_dataframe_dtypes() -> Callable[[pd.DataFrame], pd.DataFrame]:
    global optimize_dataframe_dtypes
    if optimize_dataframe_dtypes is None:  # pragma: no branch - simple guard
        from src.utils.common_operations import optimize_dataframe_dtypes as _impl

        optimize_dataframe_dtypes = _impl
    return cast(Callable[[pd.DataFrame], pd.DataFrame], optimize_dataframe_dtypes)


def _load_get_memory_usage() -> Callable[[], float]:
    global get_memory_usage
    if get_memory_usage is None:  # pragma: no branch - simple guard
        from src.utils.common_operations import get_memory_usage as _impl

        get_memory_usage = _impl
    return cast(Callable[[], float], get_memory_usage)


def _load_m1_gpu_manager() -> Callable[..., Any]:
    global get_m1_gpu_manager
    if get_m1_gpu_manager is None:  # pragma: no branch - simple guard
        from src.utils.hardware import get_m1_gpu_manager as _impl

        get_m1_gpu_manager = _impl
    return cast(Callable[..., Any], get_m1_gpu_manager)


def _load_m1_memory_optimizer() -> Callable[..., Any]:
    global get_m1_memory_optimizer
    if get_m1_memory_optimizer is None:  # pragma: no branch - simple guard
        from src.utils.hardware import get_m1_memory_optimizer as _impl

        get_m1_memory_optimizer = _impl
    return cast(Callable[..., Any], get_m1_memory_optimizer)


def _load_m1_cpu_optimizer() -> Callable[..., Any]:
    global get_m1_cpu_optimizer
    if get_m1_cpu_optimizer is None:  # pragma: no branch - simple guard
        from src.utils.hardware import get_m1_cpu_optimizer as _impl

        get_m1_cpu_optimizer = _impl
    return cast(Callable[..., Any], get_m1_cpu_optimizer)

class ArchitectureType(Enum):
    """Types of architectures supported."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"

class DataType(Enum):
    """Types of data supported."""
    MARKET_DATA = "market_data"
    REGIME_DATA = "regime_data"
    PREDICTION_DATA = "prediction_data"
    EVALUATION_DATA = "evaluation_data"

@dataclass
class UnifiedUtilityConfig:
    """Configuration for unified utilities."""
    
    # Data processing parameters
    enable_data_validation: bool = True
    enable_memory_optimization: bool = True
    enable_hardware_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Validation parameters
    strict_validation: bool = False
    auto_fix_issues: bool = True
    validation_threshold: float = 0.95
    
    # Performance parameters
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    
    # Logging parameters
    enable_logging: bool = True
    log_level: str = 'INFO'
    enable_progress_tracking: bool = True

class UnifiedUtilities:
    """
    Unified utilities class that consolidates all utility functions
    for both NAS and TAS systems.
    """
    
    def __init__(self, config: Optional[UnifiedUtilityConfig] = None):
        """Initialize unified utilities."""
        self.config = config or UnifiedUtilityConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware optimizers
        self.hardware_optimizers: Dict[str, Any] = {}
        if self.config.enable_hardware_optimization:
            self._initialize_hardware_optimizers()

        # Initialize caching with memory-awareness
        self._cache_lock = threading.RLock()
        self.cache: "OrderedDict[str, Tuple[Any, int]]" = OrderedDict()
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._cache_bytes = 0
        self._max_cache_bytes = max(1, int(self.config.cache_size_mb * 1024 * 1024))
        self._cache_pressure_notified = False

        # Performance monitoring
        self.operation_history = []
        
        tprint_info(f"🚀 Unified Utilities initialized")
        tprint_info(f"   Data validation: {'Enabled' if self.config.enable_data_validation else 'Disabled'}")
        tprint_info(f"   Memory optimization: {'Enabled' if self.config.enable_memory_optimization else 'Disabled'}")
        tprint_info(f"   Hardware optimization: {'Enabled' if self.config.enable_hardware_optimization else 'Disabled'}")
    
    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            if self.config.memory_limit_gb:
                memory_optimizer_factory = _load_m1_memory_optimizer()
                self.hardware_optimizers['memory'] = memory_optimizer_factory(self.config.memory_limit_gb)

            try:
                gpu_manager_factory = _load_m1_gpu_manager()
                self.hardware_optimizers['gpu'] = gpu_manager_factory()
            except Exception:  # pragma: no cover - GPU optional
                self.logger.debug("GPU optimizer unavailable", exc_info=True)

            cpu_optimizer_factory = _load_m1_cpu_optimizer()
            self.hardware_optimizers['cpu'] = cpu_optimizer_factory()

            tprint_success("✅ Hardware optimizers initialized")

        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization setup failed: {e}")
            self.hardware_optimizers = {}
    
    def validate_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     data_type: DataType, 
                     architecture_type: ArchitectureType) -> Dict[str, Any]:
        """
        Validate data for the specified architecture type.
        
        Args:
            data: Data to validate
            data_type: Type of data
            architecture_type: Type of architecture
            
        Returns:
            Validation result dictionary
        """
        try:
            tprint_info(f"🔍 Validating {data_type.value} data for {architecture_type.value} architecture...")
            
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'data_quality_score': 0.0,
                'recommendations': []
            }
            
            if isinstance(data, pd.DataFrame):
                validation_result = self._validate_dataframe(data, data_type, architecture_type)
            elif isinstance(data, np.ndarray):
                validation_result = self._validate_array(data, data_type, architecture_type)
            else:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Unsupported data type: {type(data)}")
            
            # Calculate overall validation score
            validation_result['data_quality_score'] = self._calculate_validation_score(validation_result)
            
            tprint_success(f"✅ Data validation completed - Score: {validation_result['data_quality_score']:.4f}")
            
            return validation_result
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return {
                'is_valid': False,
                'warnings': [],
                'errors': [str(e)],
                'data_quality_score': 0.0,
                'recommendations': ['Fix validation errors before proceeding']
            }
    
    def _validate_dataframe(self, data: pd.DataFrame, data_type: DataType, 
                           architecture_type: ArchitectureType) -> Dict[str, Any]:
        """Validate DataFrame data."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Basic validation
        if len(data) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("DataFrame is empty")
            return validation_result
        
        # Check required columns based on data type
        required_columns = self._get_required_columns(data_type, architecture_type)
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            if self.config.strict_validation:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            else:
                validation_result['warnings'].append(f"Missing recommended columns: {missing_columns}")
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > (1 - self.config.validation_threshold):
            validation_result['warnings'].append(f"High missing value ratio: {missing_ratio:.4f}")
            validation_result['recommendations'].append("Consider imputing missing values")
        
        # Check data types
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0 and data_type in [DataType.MARKET_DATA, DataType.EVALUATION_DATA]:
            validation_result['warnings'].append("No numeric columns found")
            validation_result['recommendations'].append("Ensure numeric columns are properly formatted")
        
        # Architecture-specific validation
        if architecture_type == ArchitectureType.NEURAL:
            validation_result = self._validate_for_neural_architecture(data, validation_result)
        elif architecture_type == ArchitectureType.TREE:
            validation_result = self._validate_for_tree_architecture(data, validation_result)
        elif architecture_type == ArchitectureType.HYBRID:
            validation_result = self._validate_for_hybrid_architecture(data, validation_result)
        
        return validation_result
    
    def _validate_array(self, data: np.ndarray, data_type: DataType, 
                       architecture_type: ArchitectureType) -> Dict[str, Any]:
        """Validate numpy array data."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Basic validation
        if len(data) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Array is empty")
            return validation_result
        
        # Check for infinite or NaN values
        if not np.all(np.isfinite(data)):
            validation_result['warnings'].append("Array contains infinite or NaN values")
            validation_result['recommendations'].append("Clean infinite and NaN values")
        
        # Check data shape
        if len(data.shape) != 2 and data_type in [DataType.MARKET_DATA, DataType.EVALUATION_DATA]:
            validation_result['warnings'].append("Expected 2D array for this data type")
            validation_result['recommendations'].append("Reshape data to 2D if needed")
        
        # Architecture-specific validation
        if architecture_type == ArchitectureType.NEURAL:
            if len(data.shape) == 2 and data.shape[1] == 0:
                validation_result['warnings'].append("Empty feature dimension for neural architecture")
        
        return validation_result
    
    def _get_required_columns(self, data_type: DataType, architecture_type: ArchitectureType) -> List[str]:
        """Get required columns for data type and architecture."""
        base_columns = []
        
        if data_type == DataType.MARKET_DATA:
            base_columns = ['close']  # Minimum required
        elif data_type == DataType.REGIME_DATA:
            base_columns = ['regime']
        elif data_type == DataType.PREDICTION_DATA:
            base_columns = ['prediction']
        elif data_type == DataType.EVALUATION_DATA:
            base_columns = ['actual', 'predicted']
        
        # Architecture-specific requirements
        if architecture_type == ArchitectureType.NEURAL:
            # Neural networks benefit from more features
            if data_type == DataType.MARKET_DATA:
                base_columns.extend(['open', 'high', 'low', 'volume'])
        elif architecture_type == ArchitectureType.TREE:
            # Trees are more robust to missing features
            pass  # Use base columns only
        elif architecture_type == ArchitectureType.HYBRID:
            # Hybrid approach needs comprehensive features
            if data_type == DataType.MARKET_DATA:
                base_columns.extend(['open', 'high', 'low', 'volume'])
        
        return base_columns
    
    def _validate_for_neural_architecture(self, data: pd.DataFrame, 
                                         validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data specifically for neural architectures."""
        # Check for sufficient data
        if len(data) < 100:
            validation_result['warnings'].append("Neural architectures typically need more data")
            validation_result['recommendations'].append("Consider collecting more data or using data augmentation")
        
        # Check feature scaling
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            feature_ranges = data[numeric_columns].max() - data[numeric_columns].min()
            large_ranges = feature_ranges[feature_ranges > 1000]
            if len(large_ranges) > 0:
                validation_result['warnings'].append("Large feature ranges detected")
                validation_result['recommendations'].append("Consider feature scaling for neural architectures")
        
        return validation_result
    
    def _validate_for_tree_architecture(self, data: pd.DataFrame, 
                                       validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data specifically for tree architectures."""
        # Trees are generally more robust
        # Check for categorical features
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            validation_result['recommendations'].append("Consider encoding categorical features for trees")
        
        return validation_result
    
    def _validate_for_hybrid_architecture(self, data: pd.DataFrame, 
                                         validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data specifically for hybrid architectures."""
        # Combine validation for both neural and tree components
        validation_result = self._validate_for_neural_architecture(data, validation_result)
        validation_result = self._validate_for_tree_architecture(data, validation_result)
        
        # Hybrid-specific checks
        if len(data) < 200:
            validation_result['warnings'].append("Hybrid architectures may need more data for optimal performance")
        
        return validation_result
    
    def _calculate_validation_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        score = 1.0
        
        # Deduct for errors
        score -= len(validation_result['errors']) * 0.3
        
        # Deduct for warnings
        score -= len(validation_result['warnings']) * 0.1
        
        return max(0.0, score)
    
    def optimize_data(self, data: Union[pd.DataFrame, np.ndarray], 
                     architecture_type: ArchitectureType,
                     data_type: DataType) -> Union[pd.DataFrame, np.ndarray]:
        """
        Optimize data for the specified architecture type.
        
        Args:
            data: Data to optimize
            architecture_type: Type of architecture
            data_type: Type of data
            
        Returns:
            Optimized data
        """
        try:
            tprint_info(f"⚡ Optimizing data for {architecture_type.value} architecture...")
            
            optimized_data = data.copy()
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                optimized_data = self._optimize_memory_usage(optimized_data)
            
            # Architecture-specific optimization
            if architecture_type == ArchitectureType.NEURAL:
                optimized_data = self._optimize_for_neural(optimized_data, data_type)
            elif architecture_type == ArchitectureType.TREE:
                optimized_data = self._optimize_for_tree(optimized_data, data_type)
            elif architecture_type == ArchitectureType.HYBRID:
                optimized_data = self._optimize_for_hybrid(optimized_data, data_type)
            
            tprint_success("✅ Data optimization completed")
            
            return optimized_data
            
        except Exception as e:
            tprint_error(f"❌ Data optimization failed: {e}")
            return data  # Return original data on failure
    
    def _optimize_memory_usage(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """Optimize memory usage of data."""
        if isinstance(data, pd.DataFrame):
            # Optimize DataFrame dtypes using the lazily imported helper
            optimizer = _load_optimize_dataframe_dtypes()
            return optimizer(data)
        elif isinstance(data, np.ndarray):
            # Optimize array dtypes
            if data.dtype == np.float64:
                return data.astype(np.float32)
            elif data.dtype == np.int64:
                return data.astype(np.int32)
        
        return data
    
    def _optimize_for_neural(self, data: Union[pd.DataFrame, np.ndarray], 
                           data_type: DataType) -> Union[pd.DataFrame, np.ndarray]:
        """Optimize data for neural architectures."""
        if isinstance(data, pd.DataFrame):
            # Scale features for neural networks
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        
        return data
    
    def _optimize_for_tree(self, data: Union[pd.DataFrame, np.ndarray], 
                          data_type: DataType) -> Union[pd.DataFrame, np.ndarray]:
        """Optimize data for tree architectures."""
        # Trees are generally robust to data format
        # Just ensure proper encoding of categorical variables
        if isinstance(data, pd.DataFrame):
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                data[col] = pd.Categorical(data[col]).codes
        
        return data
    
    def _optimize_for_hybrid(self, data: Union[pd.DataFrame, np.ndarray], 
                           data_type: DataType) -> Union[pd.DataFrame, np.ndarray]:
        """Optimize data for hybrid architectures."""
        # Apply both neural and tree optimizations
        data = self._optimize_for_neural(data, data_type)
        data = self._optimize_for_tree(data, data_type)
        
        return data
    
    def create_performance_report(self, operation_name: str, 
                                start_time: float, 
                                end_time: float,
                                additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a performance report for an operation.
        
        Args:
            operation_name: Name of the operation
            start_time: Start time of operation
            end_time: End time of operation
            additional_metrics: Additional metrics to include
            
        Returns:
            Performance report dictionary
        """
        execution_time = end_time - start_time
        
        memory_usage_fn = _load_get_memory_usage()

        report = {
            'operation_name': operation_name,
            'execution_time': execution_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'memory_usage': memory_usage_fn(),
            'cache_bytes': self._cache_bytes,
            'additional_metrics': additional_metrics or {}
        }
        
        # Add to operation history
        self.operation_history.append(report)
        
        # Keep only recent operations
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-100:]
        
        return report
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all operations."""
        if not self.operation_history:
            return {'total_operations': 0}
        
        recent_operations = self.operation_history[-20:]  # Last 20 operations
        
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        cache_hit_rate = self.cache_stats['hits'] / total_requests if total_requests else 0.0

        summary = {
            'total_operations': len(self.operation_history),
            'avg_execution_time': np.mean([op['execution_time'] for op in recent_operations]),
            'max_execution_time': max([op['execution_time'] for op in recent_operations]),
            'min_execution_time': min([op['execution_time'] for op in recent_operations]),
            'cache_hit_rate': cache_hit_rate,
            'cache_evictions': self.cache_stats['evictions'],
            'cache_memory_mb': round(self._cache_bytes / (1024 * 1024), 3),
        }

        return summary
    
    def safe_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Safely execute an operation with error handling, logging, and propagation.

        Args:
            operation: Function to execute
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation.

        Raises:
            UnifiedUtilityOperationError: If the wrapped operation fails.
        """
        start_time = time.time()

        try:
            result = operation(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging path
            end_time = time.time()
            self.create_performance_report(
                getattr(operation, '__name__', repr(operation)),
                start_time,
                end_time,
                {'success': False, 'error': str(exc)}
            )
            tprint_error(f"❌ Operation {getattr(operation, '__name__', repr(operation))} failed: {exc}")
            raise UnifiedUtilityOperationError(str(exc)) from exc

        end_time = time.time()
        self.create_performance_report(
            getattr(operation, '__name__', repr(operation)),
            start_time,
            end_time,
            {'success': True}
        )

        tprint_success(
            f"✅ Operation {getattr(operation, '__name__', repr(operation))} completed successfully"
        )

        return result
    
    def _estimate_cache_entry_size(self, value: Any) -> int:
        """Estimate the memory footprint of a cache entry."""
        if isinstance(value, pd.DataFrame):
            return int(value.memory_usage(deep=True).sum())
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if hasattr(value, "nbytes"):
            try:
                return int(value.nbytes)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            return sys.getsizeof(value)
        except Exception:  # pragma: no cover - fallback when getsizeof fails
            return 0

    def _memory_pressure_exceeded(self) -> bool:
        """Determine if system memory usage warrants cache eviction."""
        if not PSUTIL_AVAILABLE:  # pragma: no cover - psutil optional
            return False

        try:
            virtual_memory = psutil.virtual_memory()
        except Exception:  # pragma: no cover - psutil failure safeguard
            return False

        return virtual_memory.percent >= 90 or virtual_memory.available < 0.1 * virtual_memory.total

    def _enforce_cache_limits(self) -> None:
        """Evict cache entries until limits are satisfied."""
        over_limit = lambda: self._cache_bytes > self._max_cache_bytes or self._memory_pressure_exceeded()

        while self.cache and over_limit():
            key, (_, entry_size) = self.cache.popitem(last=False)
            self._cache_bytes = max(0, self._cache_bytes - entry_size)
            self.cache_stats['evictions'] += 1

        if self._memory_pressure_exceeded():
            if not self._cache_pressure_notified:
                warnings.warn(
                    "Unified utilities cache trimmed due to detected memory pressure",
                    UnifiedCacheWarning,
                )
                self._cache_pressure_notified = True
        else:
            self._cache_pressure_notified = False

    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available."""
        with self._cache_lock:
            if key in self.cache:
                value, entry_size = self.cache.pop(key)
                # Reinsert to maintain LRU ordering
                self.cache[key] = (value, entry_size)
                self.cache_stats['hits'] += 1
                return value

            self.cache_stats['misses'] += 1
            return None

    def cache_result(self, key: str, result: Any) -> None:
        """Cache a result with memory-aware eviction."""
        entry_size = self._estimate_cache_entry_size(result)

        with self._cache_lock:
            if key in self.cache:
                _, existing_size = self.cache.pop(key)
                self._cache_bytes = max(0, self._cache_bytes - existing_size)

            self.cache[key] = (result, entry_size)
            self._cache_bytes += entry_size
            self._enforce_cache_limits()

def create_unified_utilities(config: Optional[UnifiedUtilityConfig] = None) -> UnifiedUtilities:
    """Create unified utilities with specified configuration."""
    return UnifiedUtilities(config)

# Convenience functions
def quick_data_validation(data: Union[pd.DataFrame, np.ndarray], 
                         data_type: DataType, 
                         architecture_type: ArchitectureType) -> Dict[str, Any]:
    """Quick data validation using default configuration."""
    utilities = UnifiedUtilities()
    return utilities.validate_data(data, data_type, architecture_type)

def quick_data_optimization(data: Union[pd.DataFrame, np.ndarray], 
                           architecture_type: ArchitectureType,
                           data_type: DataType) -> Union[pd.DataFrame, np.ndarray]:
    """Quick data optimization using default configuration."""
    utilities = UnifiedUtilities()
    return utilities.optimize_data(data, architecture_type, data_type)

# Export main classes and functions
__all__ = [
    'UnifiedUtilities',
    'UnifiedUtilityConfig',
    'ArchitectureType',
    'DataType',
    'create_unified_utilities',
    'quick_data_validation',
    'quick_data_optimization'
]