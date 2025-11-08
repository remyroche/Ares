"""
Unified Vectorization Manager

This module provides a centralized vectorization management system that unifies
VectorBT optimizations, rolling operations, and batch processing for maximum
performance in feature generation.

Key Features:
- Unified interface for all vectorization operations
- VectorBTRollingOptimizer integration
- VectorBTBatchProcessor integration
- Memory-efficient processing
- Performance monitoring and statistics
-
- Parallel processing capabilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import warnings

# Enhanced logging with tprint
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions for when tprint is not available
    def tprint(*args, **kwargs): 
        print(f"[TPRINT] {' '.join(map(str, args))}")
    def tprint_debug(*args, **kwargs): 
        print(f"[DEBUG] {' '.join(map(str, args))}")
    def tprint_info(*args, **kwargs): 
        print(f"[INFO] {' '.join(map(str, args))}")
    def tprint_warning(*args, **kwargs): 
        print(f"[WARNING] {' '.join(map(str, args))}")
    def tprint_error(*args, **kwargs): 
        print(f"[ERROR] {' '.join(map(str, args))}")
    def tprint_success(*args, **kwargs): 
        print(f"[SUCCESS] {' '.join(map(str, args))}")
    def tprint_performance(*args, **kwargs): 
        print(f"[PERF] {' '.join(map(str, args))}")
    def tprint_timer(*args, **kwargs): 
        print(f"[TIMER] {' '.join(map(str, args))}")

# VectorBT imports
try:
    import vectorbt as vbt
    # Import available VectorBT functions
    from vectorbt import FMEAN, FSTD, MEANLB, MSTD, RollingSplitter
    # Use pandas rolling functions as fallback for missing VectorBT functions
    import pandas as pd
    import numpy as np

    # Create wrapper functions for compatibility
    def rolling_mean(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).mean()
        return pd.Series(data).rolling(window, **kwargs).mean()

    def rolling_std(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).std()
        return pd.Series(data).rolling(window, **kwargs).std()

    def rolling_var(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).var()
        return pd.Series(data).rolling(window, **kwargs).var()

    def rolling_min(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).min()
        return pd.Series(data).rolling(window, **kwargs).min()

    def rolling_max(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).max()
        return pd.Series(data).rolling(window, **kwargs).max()

    def rolling_sum(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).sum()
        return pd.Series(data).rolling(window, **kwargs).sum()

    def rolling_apply(data, func, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).apply(func)
        return pd.Series(data).rolling(window, **kwargs).apply(func)

    def rolling_corr(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).corr()
        return pd.Series(data).rolling(window, **kwargs).corr()

    def rolling_cov(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).cov()
        return pd.Series(data).rolling(window, **kwargs).cov()

    def rolling_quantile(data, window, q, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).quantile(q)
        return pd.Series(data).rolling(window, **kwargs).quantile(q)

    def rolling_skew(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).skew()
        return pd.Series(data).rolling(window, **kwargs).skew()

    def rolling_kurt(data, window, **kwargs):
        if hasattr(data, 'rolling'):
            return data.rolling(window, **kwargs).kurt()
        return pd.Series(data).rolling(window, **kwargs).kurt()

    # Scaling functions using pandas/numpy
    def scale(data, **kwargs):
        return (data - data.mean()) / data.std()

    def rank(data, **kwargs):
        return data.rank(**kwargs)

    def zscore(data, **kwargs):
        return (data - data.mean()) / data.std()

    def winsorize(data, limits=None, **kwargs):
        if limits is None:
            limits = (0.05, 0.05)
        from scipy.stats import mstats
        return pd.Series(mstats.winsorize(data, limits=limits), index=data.index)

    def clip(data, lower=None, upper=None, **kwargs):
        return data.clip(lower=lower, upper=upper, **kwargs)

    def quantile(data, q, **kwargs):
        return data.quantile(q, **kwargs)

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
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import our optimization modules - using lazy imports to avoid circular imports
VectorBTRollingOptimizer = None
get_vectorbt_rolling_optimizer = None
VectorBTBatchProcessor = None
BatchProcessingConfig = None

# Hardware optimization imports
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager,
        get_unified_hardware_manager,
        WorkloadType,
        OptimizationLevel,
        HardwareConfig
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    UnifiedHardwareManager = None
    get_unified_hardware_manager = None
    WorkloadType = None
    OptimizationLevel = None
    HardwareConfig = None
    warnings.warn("UnifiedHardwareManager not available. Install hardware optimization components for enhanced performance")

logger = logging.getLogger(__name__)

def _lazy_import_optimization_modules():
    """Lazy import optimization modules to avoid circular imports."""
    global VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer, VectorBTBatchProcessor, BatchProcessingConfig

    if VectorBTRollingOptimizer is None:
        try:
            from .vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        except ImportError:
            try:
                from vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
            except ImportError:
                VectorBTRollingOptimizer = None
                get_vectorbt_rolling_optimizer = None

    if VectorBTBatchProcessor is None:
        try:
            from src.feature_generation.core.vectorbt_batch_processor import VectorBTBatchProcessor, BatchProcessingConfig
        except ImportError:
            VectorBTBatchProcessor = None
            BatchProcessingConfig = None

# Operation types and configuration classes
class OperationType(Enum):
    """Types of operations that can be optimized."""
    FEATURE_ENGINEERING = "feature_engineering"
    CROSS_VALIDATION = "cross_validation"
    BACKTESTING = "backtesting"
    HMM_TRAINING = "hmm_training"
    MODEL_TRAINING = "model_training"
    FEATURE_SELECTION = "feature_selection"
    TECHNICAL_INDICATORS = "technical_indicators"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    STATISTICAL_COMPUTATION = "statistical_computation"
    # VectorBT-specific operations
    VECTORBT_BACKTESTING = "vectorbt_backtesting"
    VECTORBT_METRICS = "vectorbt_metrics"
    VECTORBT_PORTFOLIO_OPTIMIZATION = "vectorbt_portfolio_optimization"
    VECTORBT_TECHNICAL_ANALYSIS = "vectorbt_technical_analysis"

class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    VECTORIZED_CPU = "vectorized_cpu"
    GPU_ACCELERATED = "gpu_accelerated"
    PARALLEL_PROCESSING = "parallel_processing"
    HYBRID_OPTIMIZATION = "hybrid_optimization"
    MEMORY_OPTIMIZED = "memory_optimized"
    FALLBACK = "fallback"
    # VectorBT-specific strategies
    VECTORBT_CPU = "vectorbt_cpu"
    VECTORBT_GPU = "vectorbt_gpu"
    VECTORBT_PARALLEL = "vectorbt_parallel"

@dataclass
class StrategySelectionConfig:
    """Configuration for strategy selection thresholds."""
    # Data size thresholds
    gpu_data_size_threshold: int = 10000
    parallel_data_size_threshold: int = 5000
    vectorbt_data_size_threshold: int = 100
    vectorbt_gpu_threshold: int = 5000
    vectorbt_parallel_threshold: int = 1000

    # Memory thresholds
    memory_optimization_threshold_mb: float = 512.0
    chunking_data_size_threshold: int = 1000

    # CPU core thresholds
    parallel_cpu_cores_threshold: int = 4
    vectorbt_parallel_cpu_cores_threshold: int = 2

@dataclass
class OperationConfig:
    """Configuration for operation optimization."""
    operation_type: OperationType
    data_size: int
    data_dimensions: Tuple[int, ...]
    memory_budget_mb: float = 1024.0
    time_budget_seconds: float = 300.0
    precision_requirement: str = "medium"  # "low", "medium", "high"
    parallel_workers: Optional[int] = None
    # Performance baselines (in seconds)
    baseline_times: Optional[Dict[OperationType, float]] = None
    # Strategy selection configuration
    strategy_config: Optional[StrategySelectionConfig] = None

@dataclass
class OptimizationResult:
    """Result of an optimized operation."""
    result: Any
    strategy_used: OptimizationStrategy
    computation_time: float
    memory_used_mb: float
    performance_gain: float
    metadata: Dict[str, Any]

# Enhanced error handling with fast failing
class UnifiedVectorizationError(Exception):
    """Custom exception for unified vectorization errors with detailed context."""
    def __init__(self, message: str, operation: str = None, data_shape: tuple = None,
                 config: str = None, original_error: Exception = None):
        self.operation = operation
        self.data_shape = data_shape
        self.config = config
        self.original_error = original_error

        # Build detailed error message
        context_parts = []
        if operation:
            context_parts.append(f"Operation: {operation}")
        if data_shape:
            context_parts.append(f"Data shape: {data_shape}")
        if config:
            context_parts.append(f"Config: {config}")

        context_str = ", ".join(context_parts)
        full_message = f"{message}"
        if context_str:
            full_message += f" (Context: {context_str})"
        if original_error:
            full_message += f" (Original: {str(original_error)})"

        super().__init__(full_message)

class VectorizationValidationError(Exception):
    """Custom exception for vectorization validation errors."""
    def __init__(self, message: str, validation_type: str = None, value: Any = None):
        self.validation_type = validation_type
        self.value = value
        full_message = f"{message}"
        if validation_type:
            full_message += f" (Validation: {validation_type})"
        if value is not None:
            full_message += f" (Value: {value})"
        super().__init__(full_message)

@dataclass
class VectorizationConfig:
    """Configuration for unified vectorization."""
    # VectorBT settings
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True

    # Hardware optimization
    enable_hardware_optimization: bool = True
    workload_type: WorkloadType = None
    optimization_level: OptimizationLevel = None

    # Memory management
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    memory_optimization_threshold_mb: float = 512.0
    memory_budget_mb: float = 8192.0

    # Performance monitoring
    enable_monitoring: bool = True
    enable_profiling: bool = False

    # Batch processing
    batch_size: int = 10000
    enable_batch_processing: bool = True

    # Rolling operations
    rolling_optimization_threshold: int = 1000
    enable_rolling_optimization: bool = True
    
    # Parallel processing thresholds
    parallel_cpu_cores_threshold: int = 4
    gpu_data_size_threshold: int = 10000

    def __post_init__(self):
        if not VECTORBT_AVAILABLE:
            self.enable_vectorbt = False
            logger.warning("VectorBT not available, disabling vectorization optimizations")

        if self.enable_gpu and True:
            self.enable_gpu = False
            logger.warning("GPU not available, disabling GPU optimizations")

class UnifiedVectorizationManager:
    """
    Unified manager for all vectorization operations using VectorBT optimizations.

    This class provides a single interface for:
    - VectorBT rolling operations
    - Batch processing
    - Memory optimization
    - Performance monitoring
    -
    - Parallel processing
    """

    def __init__(self, config: Optional[VectorizationConfig] = None,
                 fast_fail: bool = True, enable_logging: bool = True, verbose: bool = False):
        """
        Initialize unified vectorization manager with enhanced logging and fast failing.

        Args:
            config: Vectorization configuration
            fast_fail: Enable fast failing instead of silent fallbacks
            enable_logging: Enable comprehensive logging with tprint
            verbose: Enable verbose success messages (default: False for reduced output)
        """
        tprint_info("🚀 Initializing UnifiedVectorizationManager with enhanced logging and fast failing")

        self.config = config or VectorizationConfig()
        self.fast_fail = fast_fail
        self.enable_logging = enable_logging
        self.verbose = verbose

        # Validate configuration
        self._validate_config(self.config)

        # Initialize hardware manager
        self.hardware_manager = None
        if self.config.enable_hardware_optimization and HARDWARE_AVAILABLE:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                if self.config.workload_type:
                    self.hardware_manager.optimize_for_workload(
                        self.config.workload_type,
                        self.config.optimization_level or OptimizationLevel.BALANCED
                    )
                tprint_success("✅ Hardware manager initialized and optimized")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware manager initialization failed: {e}")
                self.hardware_manager = None

        # Initialize components with error handling
        tprint_info("🔧 Initializing vectorization components")

        # Lazy import optimization modules to avoid circular imports
        _lazy_import_optimization_modules()

        try:
            if get_vectorbt_rolling_optimizer is not None:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    memory_efficient=self.config.memory_efficient,
                    chunk_size=self.config.chunk_size,
                    fast_fail=self.fast_fail,
                    enable_logging=self.enable_logging,
                    verbose=self.verbose
                )
                tprint_success("✅ Rolling optimizer initialized")
            else:
                tprint_warning("⚠️ Rolling optimizer not available")
                self.rolling_optimizer = None
        except Exception as e:
            error_msg = f"Failed to initialize rolling optimizer: {e}"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="initialization", original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, continuing without rolling optimizer")
                self.rolling_optimizer = None

        # Initialize batch processor with error handling
        # Lazy import optimization modules again to ensure they're available
        _lazy_import_optimization_modules()

        if VectorBTBatchProcessor is not None and BatchProcessingConfig is not None:
            try:
                batch_config = BatchProcessingConfig(
                    batch_size=self.config.batch_size,
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    max_memory_gb=self.config.max_memory_gb,
                    chunk_size=self.config.chunk_size,
                    enable_memory_optimization=self.config.memory_efficient,
                    enable_progress_tracking=self.config.enable_monitoring
                )
                self.batch_processor = VectorBTBatchProcessor(batch_config)
                tprint_success("✅ Batch processor initialized")
            except Exception as e:
                error_msg = f"Failed to initialize batch processor: {e}"
                tprint_error(f"❌ {error_msg}")
                if self.fast_fail:
                    raise UnifiedVectorizationError(error_msg, operation="initialization", original_error=e)
                else:
                    tprint_warning("⚠️ Fast fail disabled, continuing without batch processor")
                    self.batch_processor = None
        else:
            tprint_warning("⚠️ VectorBTBatchProcessor not available, continuing without batch processor")
            self.batch_processor = None

        # Enhanced performance tracking with error tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }

        # Enhanced intelligent caching system with advanced strategies
        self._result_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 1000
        self._cache_access_times = {}  # Track access times for LRU
        self._cache_memory_usage = 0
        self._max_cache_memory_mb = 100  # Maximum cache memory in MB
        self._cache_hit_rates = {}  # Track hit rates per operation type

        # Advanced cache features
        self._cache_compression_enabled = True
        # In-process default: avoid serialization to reduce CPU/memory overhead
        self._cache_serialization_enabled = False
        self._cache_ttl = {}  # Time-to-live for cache entries
        self._cache_priority = {}  # Priority levels for cache entries
        self._cache_access_frequency = {}  # Access frequency tracking
        self._cache_creation_times = {}  # Creation time tracking

        # Memory management
        self._memory_pool = {}  # Memory pool for pre-allocated objects
        self._memory_usage_history = []  # Memory usage history
        self._memory_peak_usage = 0
        self._memory_cleanup_threshold = 0.8  # Cleanup when 80% of max memory used
        self._memory_profiling_enabled = True

        # Cache statistics
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'decompressions': 0,
            'serializations': 0,
            'deserializations': 0,
            'memory_savings': 0,
            'total_operations': 0
        }

        # Only log initialization once per session to reduce verbosity
        if not hasattr(UnifiedVectorizationManager, '_logged_initialization'):
            tprint_success(f"✅ UnifiedVectorizationManager initialized: VectorBT={self.config.enable_vectorbt}, GPU={self.config.enable_gpu}, Memory={self.config.memory_efficient}, FastFail={self.fast_fail}")
            logger.info(f"UnifiedVectorizationManager initialized: VectorBT={self.config.enable_vectorbt}, GPU={self.config.enable_gpu}, Memory={self.config.memory_efficient}")
            UnifiedVectorizationManager._logged_initialization = True

    def rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str,
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation with enhanced logging and validation.

        Args:
            data: Input data
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', etc.)
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['rolling_operations'] += 1

        tprint_debug(f"🔄 Starting rolling operation: {operation}, window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")

        # Validate inputs
        self._validate_rolling_inputs(data, operation, window)

        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key(data, operation, window, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                tprint_debug("💾 Cache hit for rolling operation")
                return cached_result
            self.performance_stats['cache_misses'] += 1
            tprint_debug("💾 Cache miss for rolling operation")

        # Ensure lazy imports are loaded
        _lazy_import_optimization_modules()

        # Check if rolling optimizer is available
        if self.rolling_optimizer is None:
            error_msg = "Rolling optimizer not available"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                return self._pandas_fallback_rolling(data, operation, window, **kwargs)

        try:
            # Use VectorBT rolling optimizer with detailed logging
            tprint_debug(f"🎯 Executing rolling {operation} with VectorBT optimizer")

            if operation == 'mean':
                result = self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                result = self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                result = self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                result = self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                result = self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                result = self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            elif operation == 'quantile':
                q = kwargs.pop('q', 0.5)
                result = self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            elif operation == 'skew':
                result = self.rolling_optimizer.rolling_skew(data, window, **kwargs)
            elif operation == 'kurt':
                result = self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
            elif operation == 'corr':
                other = kwargs.pop('other', None)
                result = self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
            elif operation == 'cov':
                other = kwargs.pop('other', None)
                result = self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
            elif operation == 'apply':
                func = kwargs.pop('func', None)
                result = self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
            else:
                error_msg = f"Unsupported rolling operation: {operation}"
                tprint_error(f"❌ {error_msg}")
                if self.fast_fail:
                    raise UnifiedVectorizationError(error_msg, operation=operation)
                else:
                    tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                    return self._pandas_fallback_rolling(data, operation, window, **kwargs)

            # Update stats
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            if rolling_stats.get('vectorbt_operations', 0) > 0:
                self.performance_stats['vectorbt_operations'] += 1
            if rolling_stats.get('gpu_operations', 0) > 0:
                self.performance_stats['gpu_operations'] += 1
            if rolling_stats.get('memory_optimizations', 0) > 0:
                self.performance_stats['memory_optimizations'] += 1

            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
                tprint_debug("💾 Result cached successfully")

            if self.verbose:
                tprint_success(f"✅ Rolling {operation} completed successfully")
            return result

        except Exception as e:
            error_msg = f"Rolling operation {operation} failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1

            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1
                return self._pandas_fallback_rolling(data, operation, window, **kwargs)

        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Rolling {operation}", execution_time)

    def process_feature_batch(self, batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of features leveraging the rolling optimizer's batch analysis."""

        required_keys = {'features', 'data', 'target_column'}
        missing = [key for key in required_keys if key not in batch_config]
        if missing:
            raise UnifiedVectorizationError(
                f"Batch configuration missing required keys: {missing}",
                operation="feature_batch"
            )

        features: List[str] = batch_config['features']
        data: pd.DataFrame = batch_config['data']
        target_column: str = batch_config['target_column']
        lookback_ranges: List[int] = batch_config.get('lookback_ranges') or list(range(5, 51, 5))
        metrics: List[str] = batch_config.get('metrics') or ['corr', 'std', 'var']

        if not isinstance(data, pd.DataFrame):
            raise UnifiedVectorizationError("data must be a pandas DataFrame", operation="feature_batch")

        if target_column not in data.columns:
            raise UnifiedVectorizationError(
                f"Target column '{target_column}' not present in data",
                operation="feature_batch"
            )

        available_features = [feature for feature in features if feature in data.columns]
        if not available_features:
            missing_preview = features[:5]
            tprint_warning(
                f"⚠️ Batch request rejected: none of the requested features exist in data. Sample missing: {missing_preview}"
            )
            return {'results': [], 'errors': ['No requested features found in data'], 'missing_features': features}

        feature_df = data[available_features].copy()
        target_series = data[target_column].copy()

        missing_features = [feature for feature in features if feature not in available_features]
        if missing_features:
            tprint_warning(
                f"⚠️ {len(missing_features)}/ {len(features)} requested features missing in data. Sample: {missing_features[:5]}"
            )

        tprint_debug(
            f"🔬 Batch processing summary: available={len(available_features)}, missing={len(missing_features)}, lookbacks={len(lookback_ranges)}"
        )
        tprint_debug(f"🔎 Available feature sample: {available_features[:5]}")

        batch_analysis: Dict[str, Dict[str, Dict[str, float]]] = {}
        errors: List[str] = []

        if self.rolling_optimizer and hasattr(self.rolling_optimizer, 'batch_multi_feature_analysis'):
            try:
                batch_analysis = self.rolling_optimizer.batch_multi_feature_analysis(
                    feature_df,
                    target_series,
                    lookback_ranges,
                    metrics
                )
            except Exception as batch_err:
                error_msg = f"VectorBT batch analysis failed: {batch_err}"
                tprint_warning(f"⚠️ {error_msg}")
                errors.append(error_msg)

        if not batch_analysis:
            try:
                batch_analysis = self._fallback_batch_feature_analysis(
                    feature_df,
                    target_series,
                    lookback_ranges,
                    metrics
                )
            except Exception as fallback_err:
                error_msg = f"Fallback batch analysis failed: {fallback_err}"
                tprint_error(f"❌ {error_msg}")
                errors.append(error_msg)
                batch_analysis = {}

        results: List[Dict[str, Any]] = []

        for feature_name in available_features:
            feature_results = batch_analysis.get(feature_name)
            if not feature_results:
                tprint_warning(f"⚠️ No batch metrics returned for feature '{feature_name}'")
                errors.append(f"No batch results for feature {feature_name}")
                continue

            try:
                optimal_key = max(
                    feature_results.keys(),
                    key=lambda lb_key: feature_results[lb_key].get('combined_score', 0.0)
                )
                optimal_lookback = int(optimal_key)
                optimal_stats = feature_results[optimal_key]

                performance_score = float(optimal_stats.get('combined_score', 0.0))

                feature_std = float(optimal_stats.get('feature_std', 0.0))
                feature_global_std = float(optimal_stats.get('feature_global_std', 0.0))
                if feature_global_std <= 1e-9:
                    stability_score = 0.0
                else:
                    stability_score = 1.0 - min(1.0, feature_std / (feature_global_std + 1e-9))
                    stability_score = float(max(0.0, min(stability_score, 1.0)))

                results.append({
                    'feature_name': feature_name,
                    'optimal_lookback': optimal_lookback,
                    'performance_score': performance_score,
                    'stability_score': stability_score,
                    'optimization_method': 'vectorbt_batch',
                    'lookback_range': f"{min(lookback_ranges)}-{max(lookback_ranges)}",
                    'cv_folds': batch_config.get('cv_folds', 2),
                    'optimization_time': 0.0,
                    'memory_usage': 0.0,
                    'success': True,
                    'lookback_results': feature_results
                })

            except Exception as feature_err:
                error_msg = f"Failed to summarize batch results for {feature_name}: {feature_err}"
                tprint_warning(f"⚠️ {error_msg}")
                errors.append(error_msg)

        if not results:
            errors.append('Batch processing returned zero feature results')
            tprint_warning("⚠️ Batch processing produced zero feature results")

        return {'results': results, 'errors': errors, 'missing_features': missing_features}

    def _fallback_batch_feature_analysis(self,
                                         feature_df: pd.DataFrame,
                                         target_series: pd.Series,
                                         lookbacks: List[int],
                                         metrics: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Fallback batch analysis using pandas/numpy when VectorBT batch is unavailable."""

        analysis: Dict[str, Dict[str, Dict[str, float]]] = {}

        if feature_df.empty or target_series.empty:
            return analysis

        sorted_lookbacks = sorted({int(lb) for lb in lookbacks if lb and lb > 0})
        if not sorted_lookbacks:
            return analysis

        for feature_name in feature_df.columns:
            feature_series = feature_df[feature_name]
            aligned = pd.concat([feature_series, target_series], axis=1, join='inner').dropna()
            if aligned.empty:
                continue

            feature_aligned = aligned.iloc[:, 0]
            target_aligned = aligned.iloc[:, 1]
            max_lookback = max(sorted_lookbacks)
            if len(feature_aligned) <= max_lookback:
                continue

            feature_global_std = float(feature_aligned.std(ddof=0) if feature_aligned.std(ddof=0) is not None else 0.0)
            target_global_std = float(target_aligned.std(ddof=0) if target_aligned.std(ddof=0) is not None else 0.0)

            lookback_stats: Dict[str, Dict[str, float]] = {}

            for lookback in sorted_lookbacks:
                if len(feature_aligned) <= lookback:
                    continue

                window_key = str(int(lookback))
                stats: Dict[str, float] = {
                    'combined_score': 0.0,
                    'feature_global_std': feature_global_std,
                    'target_global_std': target_global_std
                }

                if 'corr' in metrics:
                    corr_series = feature_aligned.rolling(window=lookback).corr(target_aligned)
                    corr_value = float(corr_series.abs().mean(skipna=True) or 0.0)
                else:
                    corr_value = 0.0

                if 'std' in metrics:
                    feature_std_series = feature_aligned.rolling(window=lookback).std()
                    target_std_series = target_aligned.rolling(window=lookback).std()
                    feature_std = float(feature_std_series.abs().mean(skipna=True) or 0.0)
                    target_std = float(target_std_series.abs().mean(skipna=True) or 0.0)
                else:
                    feature_std = 0.0
                    target_std = 0.0

                if 'var' in metrics:
                    feature_var_series = feature_aligned.rolling(window=lookback).var()
                    target_var_series = target_aligned.rolling(window=lookback).var()
                    feature_var = float(feature_var_series.abs().mean(skipna=True) or feature_std ** 2)
                    target_var = float(target_var_series.abs().mean(skipna=True) or target_std ** 2)
                else:
                    feature_var = feature_std ** 2
                    target_var = target_std ** 2

                stats.update({
                    'correlation': float(max(corr_value, 0.0)),
                    'feature_std': float(max(feature_std, 0.0)),
                    'target_std': float(max(target_std, 0.0)),
                    'feature_var': float(max(feature_var, 0.0)),
                    'target_var': float(max(target_var, 0.0))
                })

                stats['combined_score'] = float(min(max(corr_value, 0.0), 1.0))

                lookback_stats[window_key] = stats

            if lookback_stats:
                analysis[feature_name] = lookback_stats

        return analysis

    def scale_data(self, data: Union[pd.Series, pd.DataFrame],
                   method: str = 'zscore', **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Scale data using VectorBT scaling functions with enhanced logging and validation.

        Args:
            data: Input data
            method: Scaling method ('zscore', 'minmax', 'robust', 'quantile', 'winsorize')
            **kwargs: Additional parameters

        Returns:
            Scaled data
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['scaling_operations'] += 1

        tprint_debug(f"🔄 Starting data scaling: method={method}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")

        # Validate inputs
        self._validate_scaling_inputs(data, method)

        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
            tprint_warning("⚠️ VectorBT not available, using pandas fallback for scaling")
            return self._pandas_fallback_scaling(data, method, **kwargs)

        try:
            tprint_debug(f"🎯 Executing {method} scaling with VectorBT")

            if method == 'zscore':
                result = zscore(data, **kwargs)
            elif method == 'minmax':
                result = scale(data, method='minmax', **kwargs)
            elif method == 'robust':
                result = scale(data, method='robust', **kwargs)
            elif method == 'quantile':
                result = quantile(data, **kwargs)
            elif method == 'winsorize':
                result = winsorize(data, **kwargs)
            elif method == 'rank':
                result = rank(data, **kwargs)
            elif method == 'clip':
                result = clip(data, **kwargs)
            else:
                error_msg = f"Unsupported scaling method: {method}"
                tprint_error(f"❌ {error_msg}")
                if self.fast_fail:
                    raise UnifiedVectorizationError(error_msg, operation="scaling", original_error=ValueError(error_msg))
                else:
                    tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                    return self._pandas_fallback_scaling(data, method, **kwargs)

            self.performance_stats['vectorbt_operations'] += 1
            tprint_success(f"✅ {method} scaling completed successfully")
            return result

        except Exception as e:
            error_msg = f"VectorBT scaling failed for {method}"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1

            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="scaling", data_shape=data.shape if hasattr(data, 'shape') else None, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                return self._pandas_fallback_scaling(data, method, **kwargs)

        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Scaling {method}", execution_time)

    def parallel_process_features(self, data: pd.DataFrame,
                                feature_configs: List[Dict[str, Any]],
                                max_workers: int = None) -> pd.DataFrame:
        """
        Process multiple features in parallel with enhanced error recovery.

        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries
            max_workers: Maximum number of parallel workers (if None, auto-detect)

        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['batch_operations'] += 1

        tprint_info(f"🔄 Starting parallel feature processing: {len(feature_configs)} features, data_shape={data.shape}")

        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            error_msg = "Data must be a pandas DataFrame"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="parallel_processing", data_shape=data.shape if hasattr(data, 'shape') else None)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()

        if not isinstance(feature_configs, list) or len(feature_configs) == 0:
            error_msg = "feature_configs must be a non-empty list"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="parallel_processing")
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()

        # Determine optimal number of workers
        if max_workers is None:
            import multiprocessing
            max_workers = min(len(feature_configs), multiprocessing.cpu_count())

        tprint_debug(f"🎯 Using {max_workers} parallel workers")

        try:
            # Use adaptive batch sizing
            adaptive_batch_size = self.adaptive_batch_size(len(data), 'custom')

            # Split features into batches for parallel processing
            feature_batches = [feature_configs[i:i + adaptive_batch_size]
                             for i in range(0, len(feature_configs), adaptive_batch_size)]

            results = {}
            successful_features = 0
            failed_features = 0

            # Process batches in parallel
            if len(feature_batches) > 1 and max_workers > 1:
                tprint_debug(f"🔄 Processing {len(feature_batches)} batches in parallel")
                # For now, process sequentially but with enhanced error recovery
                # In a full implementation, this would use multiprocessing
                for batch_idx, batch in enumerate(feature_batches):
                    tprint_debug(f"🔄 Processing batch {batch_idx + 1}/{len(feature_batches)}")
                    batch_results = self._process_feature_batch_with_recovery(data, batch)
                    results.update(batch_results)
            else:
                # Process all features at once
                results = self._process_feature_batch_with_recovery(data, feature_configs)

            # Count successful and failed features
            for feature_name, result in results.items():
                if result is not None and not (isinstance(result, pd.Series) and result.isna().all()):
                    successful_features += 1
                else:
                    failed_features += 1

            tprint_success(f"✅ Parallel processing completed: {successful_features} successful, {failed_features} failed")
            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            error_msg = f"Parallel feature processing failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.performance_stats['errors'] += 1

            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="parallel_processing", data_shape=data.shape, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()

        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Parallel processing ({len(feature_configs)} features)", execution_time)

    def _process_feature_batch_with_recovery(self, data: pd.DataFrame,
                                           feature_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of features with enhanced error recovery and retry mechanisms."""
        results = {}

        for i, config in enumerate(feature_configs):
            feature_name = config.get('name', f'feature_{i}')
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    tprint_debug(f"🔄 Processing feature: {feature_name} (attempt {retry_count + 1})")

                    # Process single feature with error recovery
                    result = self._process_single_feature_with_recovery(data, config)
                    results[feature_name] = result
                    break  # Success, exit retry loop

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Feature {feature_name} failed (attempt {retry_count}): {e}"
                    tprint_warning(f"⚠️ {error_msg}")

                    if retry_count >= max_retries:
                        tprint_error(f"❌ Feature {feature_name} failed after {max_retries} attempts")
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                        self.performance_stats['errors'] += 1
                    else:
                        # Wait before retry (exponential backoff)
                        import time
                        time.sleep(0.1 * (2 ** retry_count))

        return results

    def _process_single_feature_with_recovery(self, data: pd.DataFrame, config: Dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Process a single feature with comprehensive error recovery."""
        feature_name = config.get('name', 'unknown')
        feature_type = config.get('type', 'rolling')
        params = config.get('params', {})

        try:
            if feature_type == 'rolling':
                operation = params.get('operation', 'mean')
                window = params.get('window', 20)
                column = params.get('column', 'close')

                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in data")

                rolling_params = {k: v for k, v in params.items() if k not in ['operation', 'window', 'column']}
                return self.rolling_operation(data[column], operation, window, **rolling_params)

            elif feature_type == 'scaling':
                method = params.get('method', 'zscore')
                column = params.get('column', 'close')

                if column not in data.columns:
                    raise ValueError(f"Column '{column}' not found in data")

                scaling_params = {k: v for k, v in params.items() if k not in ['method', 'column']}
                return self.scale_data(data[column], method, **scaling_params)

            elif feature_type == 'custom':
                func = params.get('function')
                if not callable(func):
                    raise ValueError(f"Custom function is not callable")

                return func(data, **params)

            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

        except Exception as e:
            # Enhanced error recovery
            if "memory" in str(e).lower():
                tprint_warning(f"⚠️ Memory error for {feature_name}, trying with smaller chunk")
                # Try with smaller data chunk
                chunk_size = len(data) // 2
                return self._process_single_feature_with_recovery(data.iloc[:chunk_size], config)
            elif "gpu" in str(e).lower():
                tprint_warning(f"⚠️ GPU error for {feature_name}, falling back to CPU")
                # Disable GPU and retry
                original_gpu = self.config.enable_gpu
                self.config.enable_gpu = False
                try:
                    result = self._process_single_feature_with_recovery(data, config)
                    return result
                finally:
                    self.config.enable_gpu = original_gpu
            else:
                raise  # Re-raise other errors

    def batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute batch operations (legacy method for compatibility).
        
        Args:
            operations: List of operation configuration dictionaries
            
        Returns:
            List of operation results
        """
        try:
            # This is a legacy method - operations should be processed individually
            # or converted to use batch_process_features
            results = []
            for i, op in enumerate(operations):
                try:
                    # Process individual operation
                    if op.get('operation') == 'rolling_mean':
                        data = op.get('data')
                        window = op.get('window', 20)
                        result = self.rolling_operation(data, 'mean', window)
                        results.append({
                            'success': True,
                            'feature_name': op.get('feature_name', f'operation_{i}'),
                            'result': result
                        })
                    elif op.get('operation') == 'rolling_std':
                        data = op.get('data')
                        window = op.get('window', 20)
                        result = self.rolling_operation(data, 'std', window)
                        results.append({
                            'success': True,
                            'feature_name': op.get('feature_name', f'operation_{i}'),
                            'result': result
                        })
                    else:
                        results.append({
                            'success': False,
                            'feature_name': op.get('feature_name', f'operation_{i}'),
                            'error': f"Unsupported operation: {op.get('operation')}"
                        })
                except Exception as e:
                    results.append({
                        'success': False,
                        'feature_name': op.get('feature_name', f'operation_{i}'),
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            tprint_error(f"Batch operations failed: {e}")
            return []

    def batch_process_features(self, data: pd.DataFrame,
                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process multiple features in batch with optimization and enhanced logging.

        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['batch_operations'] += 1

        tprint_info(f"🔄 Starting batch feature processing: {len(feature_configs)} features, data_shape={data.shape}")

        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            error_msg = "Data must be a pandas DataFrame"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape if hasattr(data, 'shape') else None)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()

        if not isinstance(feature_configs, list) or len(feature_configs) == 0:
            error_msg = "feature_configs must be a non-empty list"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="batch_processing")
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()

        try:
            # Use VectorBT batch processor
            results = {}
            successful_features = 0
            failed_features = 0

            tprint_debug(f"🎯 Processing {len(feature_configs)} features")

            for i, config in enumerate(feature_configs):
                feature_name = config.get('name', f'feature_{i}')
                feature_type = config.get('type', 'rolling')
                params = config.get('params', {})

                tprint_debug(f"🔄 Processing feature {i+1}/{len(feature_configs)}: {feature_name} ({feature_type})")

                try:
                    if feature_type == 'rolling':
                        operation = params.get('operation', 'mean')
                        window = params.get('window', 20)
                        column = params.get('column', 'close')

                        if column not in data.columns:
                            error_msg = f"Column '{column}' not found in data. Available: {list(data.columns)}"
                            tprint_error(f"❌ {error_msg}")
                            if self.fast_fail:
                                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape)
                            else:
                                tprint_warning("⚠️ Fast fail disabled, skipping feature")
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                                failed_features += 1
                                continue

                        # Remove operation and window from params to avoid conflicts
                        rolling_params = {k: v for k, v in params.items() if k not in ['operation', 'window', 'column']}
                        results[feature_name] = self.rolling_operation(
                            data[column], operation, window, **rolling_params
                        )
                        successful_features += 1
                        tprint_success(f"✅ Feature {feature_name} completed successfully")

                    elif feature_type == 'scaling':
                        method = params.get('method', 'zscore')
                        column = params.get('column', 'close')

                        if column not in data.columns:
                            error_msg = f"Column '{column}' not found in data. Available: {list(data.columns)}"
                            tprint_error(f"❌ {error_msg}")
                            if self.fast_fail:
                                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape)
                            else:
                                tprint_warning("⚠️ Fast fail disabled, skipping feature")
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                                failed_features += 1
                                continue

                        # Remove method and column from params to avoid conflicts
                        scaling_params = {k: v for k, v in params.items() if k not in ['method', 'column']}
                        results[feature_name] = self.scale_data(
                            data[column], method, **scaling_params
                        )
                        successful_features += 1
                        tprint_success(f"✅ Feature {feature_name} completed successfully")

                    elif feature_type == 'custom':
                        func = params.get('function')
                        if not callable(func):
                            error_msg = f"Custom function for {feature_name} is not callable"
                            tprint_error(f"❌ {error_msg}")
                            if self.fast_fail:
                                raise UnifiedVectorizationError(error_msg, operation="batch_processing")
                            else:
                                tprint_warning("⚠️ Fast fail disabled, skipping feature")
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                                failed_features += 1
                                continue

                        results[feature_name] = func(data, **params)
                        successful_features += 1
                        tprint_success(f"✅ Feature {feature_name} completed successfully")

                    else:
                        error_msg = f"Unsupported feature type: {feature_type}"
                        tprint_error(f"❌ {error_msg}")
                        if self.fast_fail:
                            raise UnifiedVectorizationError(error_msg, operation="batch_processing")
                        else:
                            tprint_warning("⚠️ Fast fail disabled, skipping feature")
                            results[feature_name] = pd.Series(np.nan, index=data.index)
                            failed_features += 1
                            continue

                except Exception as e:
                    error_msg = f"Feature {feature_name} failed: {e}"
                    tprint_error(f"❌ {error_msg}")
                    self.performance_stats['errors'] += 1

                    if self.fast_fail:
                        self.performance_stats['fast_failures'] += 1
                        raise UnifiedVectorizationError(error_msg, operation="batch_processing", original_error=e)
                    else:
                        tprint_warning("⚠️ Fast fail disabled, using NaN for failed feature")
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                        failed_features += 1

            tprint_success(f"✅ Batch processing completed: {successful_features} successful, {failed_features} failed")
            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            error_msg = f"Batch feature processing failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.performance_stats['errors'] += 1

            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()

        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Batch processing ({len(feature_configs)} features)", execution_time)

    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for memory efficiency and VectorBT processing.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        if not self.config.memory_efficient:
            return data

        try:
            optimized_data = data.copy()

            # Optimize data types
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1

                elif optimized_data[column].dtype == 'int64':
                    if (optimized_data[column].min() >= np.iinfo(np.int32).min and
                        optimized_data[column].max() <= np.iinfo(np.int32).max):
                        optimized_data[column] = optimized_data[column].astype(np.int32)
                        self.performance_stats['memory_optimizations'] += 1

            # Calculate memory savings
            original_memory = data.memory_usage(deep=True).sum()
            optimized_memory = optimized_data.memory_usage(deep=True).sum()
            memory_savings = (original_memory - optimized_memory) / original_memory * 100
            self.performance_stats['memory_savings'] += memory_savings

            return optimized_data

        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return data

    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for processing (alias for optimize_dataframe for compatibility).

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        return self.optimize_dataframe(data)

    def _pandas_fallback_rolling(self, data: Union[pd.Series, pd.DataFrame],
                                operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)

        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")

    def _pandas_fallback_scaling(self, data: Union[pd.Series, pd.DataFrame],
                                method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback scaling using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def _generate_cache_key(self, data: Union[pd.Series, pd.DataFrame],
                           operation: str, window: int, **kwargs) -> str:
        """Generate cache key for operation."""
        import hashlib

        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]

        return f"{operation}_{window}_{data_hash}_{params_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """Get result from cache with intelligent LRU eviction and TTL support."""
        if not self._cache_enabled:
            return None

        try:
            if cache_key in self._result_cache:
                # Check TTL (Time To Live)
                if cache_key in self._cache_ttl:
                    if time.time() > self._cache_ttl[cache_key]:
                        tprint_debug("💾 Cache entry expired")
                        self._remove_from_cache(cache_key)
                        self._cache_stats['misses'] += 1
                        return None

                # Update access tracking
                current_time = time.time()
                self._cache_access_times[cache_key] = current_time
                self._cache_access_frequency[cache_key] = self._cache_access_frequency.get(cache_key, 0) + 1

                # Decompress if needed
                result = self._result_cache[cache_key]
                if self._cache_compression_enabled and isinstance(result, bytes):
                    result = self._decompress_cache_entry(result)
                    self._cache_stats['decompressions'] += 1

                # Deserialize if needed
                if self._cache_serialization_enabled and isinstance(result, bytes):
                    result = self._deserialize_cache_entry(result)
                    self._cache_stats['deserializations'] += 1

                self._cache_stats['hits'] += 1
                tprint_debug("💾 Cache hit")
                return result
        except Exception as e:
            tprint_warning(f"⚠️ Cache retrieval failed: {e}")
            self._cache_stats['misses'] += 1

        tprint_debug("💾 Cache miss")
        self._cache_stats['misses'] += 1
        return None

    def _put_in_cache(self, cache_key: str, result: Union[pd.Series, pd.DataFrame],
                     ttl_seconds: int = 3600, priority: int = 1):
        """Put result in cache with intelligent memory management and compression."""
        if not self._cache_enabled:
            return

        try:
            # Calculate memory usage of the result
            original_memory = self._estimate_result_memory(result)

            # Apply compression if enabled
            # Compress only for sizable entries to avoid CPU overhead on tiny payloads (>1MB)
            if self._cache_compression_enabled and original_memory > (1 * 1024 * 1024):
                result = self._compress_cache_entry(result)
                self._cache_stats['compressions'] += 1
                compressed_memory = self._estimate_result_memory(result)
                memory_savings = original_memory - compressed_memory
                self._cache_stats['memory_savings'] += memory_savings
                tprint_debug(f"💾 Compressed cache entry: {memory_savings / (1024*1024):.2f}MB saved")

            # Apply serialization if enabled
            if self._cache_serialization_enabled:
                result = self._serialize_cache_entry(result)
                self._cache_stats['serializations'] += 1

            result_memory = self._estimate_result_memory(result)

            # Check if we need to evict items
            while (len(self._result_cache) >= self._max_cache_size or
                   self._cache_memory_usage + result_memory > self._max_cache_memory_mb * 1024 * 1024):
                self._evict_least_recently_used()

            # Store result with metadata
            self._result_cache[cache_key] = result
            current_time = time.time()
            self._cache_access_times[cache_key] = current_time
            self._cache_creation_times[cache_key] = current_time
            self._cache_priority[cache_key] = priority
            self._cache_access_frequency[cache_key] = 1

            # Set TTL
            if ttl_seconds > 0:
                self._cache_ttl[cache_key] = current_time + ttl_seconds

            self._cache_memory_usage += result_memory
            self._cache_stats['total_operations'] += 1

            tprint_debug(f"💾 Result cached: {result_memory / (1024*1024):.2f}MB (priority: {priority})")

        except Exception as e:
            tprint_warning(f"⚠️ Cache storage failed: {e}")

    def _compress_cache_entry(self, data: Union[pd.Series, pd.DataFrame]) -> bytes:
        """Compress cache entry using pickle and gzip."""
        try:
            import pickle
            import gzip

            # Serialize first
            serialized = pickle.dumps(data)

            # Compress
            compressed = gzip.compress(serialized)

            return compressed
        except Exception as e:
            tprint_warning(f"⚠️ Cache compression failed: {e}")
            return data  # Return original if compression fails

    def _decompress_cache_entry(self, compressed_data: bytes) -> Union[pd.Series, pd.DataFrame]:
        """Decompress cache entry."""
        try:
            import pickle
            import gzip

            # Decompress
            serialized = gzip.decompress(compressed_data)

            # Deserialize
            data = pickle.loads(serialized)

            return data
        except Exception as e:
            tprint_warning(f"⚠️ Cache decompression failed: {e}")
            return compressed_data  # Return original if decompression fails

    def _serialize_cache_entry(self, data: Union[pd.Series, pd.DataFrame]) -> bytes:
        """Serialize cache entry using pickle."""
        try:
            import pickle
            return pickle.dumps(data)
        except Exception as e:
            tprint_warning(f"⚠️ Cache serialization failed: {e}")
            return data  # Return original if serialization fails

    def _deserialize_cache_entry(self, serialized_data: bytes) -> Union[pd.Series, pd.DataFrame]:
        """Deserialize cache entry."""
        try:
            import pickle
            return pickle.loads(serialized_data)
        except Exception as e:
            tprint_warning(f"⚠️ Cache deserialization failed: {e}")
            return serialized_data  # Return original if deserialization fails

    def _remove_from_cache(self, cache_key: str):
        """Remove entry from cache and update memory usage."""
        if cache_key in self._result_cache:
            # Calculate memory being freed
            freed_memory = self._estimate_result_memory(self._result_cache[cache_key])

            # Remove from all tracking dictionaries
            del self._result_cache[cache_key]
            self._cache_access_times.pop(cache_key, None)
            self._cache_creation_times.pop(cache_key, None)
            self._cache_priority.pop(cache_key, None)
            self._cache_access_frequency.pop(cache_key, None)
            self._cache_ttl.pop(cache_key, None)

            # Update memory usage
            self._cache_memory_usage = max(0, self._cache_memory_usage - freed_memory)
            self._cache_stats['evictions'] += 1

            tprint_debug(f"💾 Cache entry removed: {freed_memory / (1024*1024):.2f}MB freed")

    def _estimate_result_memory(self, result: Union[pd.Series, pd.DataFrame]) -> int:
        """Estimate memory usage of a result."""
        try:
            if hasattr(result, 'memory_usage'):
                return result.memory_usage(deep=True).sum()
            else:
                # Rough estimate
                return len(str(result)) * 8  # Assume 8 bytes per character
        except:
            return 1024  # Default estimate

    def _evict_least_recently_used(self):
        """Evict least recently used cache entry with priority consideration."""
        if not self._cache_access_times:
            return

        # Enhanced eviction strategy: consider priority, frequency, and recency
        def eviction_score(key):
            # Lower score = more likely to be evicted
            recency_score = self._cache_access_times.get(key, 0)
            frequency_score = self._cache_access_frequency.get(key, 1)
            priority_score = self._cache_priority.get(key, 1)

            # Weighted combination: recency (70%), frequency (20%), priority (10%)
            return (recency_score * 0.7) + (frequency_score * 0.2) + (priority_score * 0.1)

        # Find least valuable key
        lru_key = min(self._cache_access_times.keys(), key=eviction_score)

        # Remove from cache
        self._remove_from_cache(lru_key)
        tprint_debug(f"💾 Evicted entry: {lru_key}")

    def _evict_expired_entries(self):
        """Evict expired cache entries based on TTL."""
        current_time = time.time()
        expired_keys = [key for key, expiry_time in self._cache_ttl.items()
                       if current_time > expiry_time]

        for key in expired_keys:
            self._remove_from_cache(key)
            tprint_debug(f"💾 Evicted expired entry: {key}")

    def _evict_low_priority_entries(self, target_memory_mb: float):
        """Evict low priority entries to free up memory."""
        # Sort by priority (ascending) and recency (ascending)
        sorted_keys = sorted(self._cache_access_times.keys(),
                           key=lambda k: (self._cache_priority.get(k, 1),
                                        self._cache_access_times[k]))

        freed_memory = 0
        for key in sorted_keys:
            if freed_memory >= target_memory_mb * 1024 * 1024:
                break

            entry_memory = self._estimate_result_memory(self._result_cache[key])
            self._remove_from_cache(key)
            freed_memory += entry_memory
            tprint_debug(f"💾 Evicted low priority entry: {key}")

    def _cleanup_memory(self):
        """Clean up memory when usage exceeds threshold."""
        current_memory_usage = self._get_current_memory_usage()
        self._memory_usage_history.append(current_memory_usage)

        # Keep only last 20 memory readings (reduced from 100 to prevent accumulation)
        if len(self._memory_usage_history) > 20:
            self._memory_usage_history = self._memory_usage_history[-20:]

        # Update peak usage
        self._memory_peak_usage = max(self._memory_peak_usage, current_memory_usage)

        # Check if cleanup is needed
        memory_ratio = current_memory_usage / (self._max_cache_memory_mb * 1024 * 1024)
        if memory_ratio > self._memory_cleanup_threshold:
            tprint_info(f"🧹 Memory cleanup triggered: {memory_ratio:.1%} usage")

            # Evict expired entries first
            self._evict_expired_entries()

            # If still over threshold, evict low priority entries
            if self._cache_memory_usage > self._max_cache_memory_mb * 1024 * 1024 * self._memory_cleanup_threshold:
                target_memory = self._max_cache_memory_mb * 1024 * 1024 * 0.5  # Target 50% usage
                self._evict_low_priority_entries(target_memory)

            tprint_success(f"✅ Memory cleanup completed: {self._cache_memory_usage / (1024*1024):.2f}MB used")

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return self._cache_memory_usage / (1024 * 1024)  # Fallback to cache memory

    def _initialize_memory_pool(self, pool_size: int = 10):
        """Initialize memory pool with pre-allocated objects."""
        tprint_debug(f"🔄 Initializing memory pool with {pool_size} objects")

        try:
            # Pre-allocate common data structures
            for i in range(pool_size):
                # Pre-allocate empty DataFrames
                self._memory_pool[f'df_{i}'] = pd.DataFrame()

                # Pre-allocate empty Series
                self._memory_pool[f'series_{i}'] = pd.Series(dtype=float)

                # Pre-allocate numpy arrays
                self._memory_pool[f'array_{i}'] = np.empty(1000, dtype=np.float64)

            tprint_success(f"✅ Memory pool initialized with {len(self._memory_pool)} objects")

        except Exception as e:
            tprint_warning(f"⚠️ Memory pool initialization failed: {e}")

    def _get_from_memory_pool(self, object_type: str) -> Any:
        """Get object from memory pool."""
        if object_type in self._memory_pool:
            return self._memory_pool[object_type].copy()
        return None

    def _return_to_memory_pool(self, object_type: str, obj: Any):
        """Return object to memory pool."""
        if object_type in self._memory_pool:
            # Clear the object and return to pool
            if hasattr(obj, 'clear'):
                obj.clear()
            self._memory_pool[object_type] = obj

    def get_memory_profiling(self) -> Dict[str, Any]:
        """Get comprehensive memory profiling information."""
        if not self._memory_profiling_enabled:
            return {"profiling_disabled": True}

        current_memory = self._get_current_memory_usage()

        profiling = {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self._memory_peak_usage,
            'cache_memory_mb': self._cache_memory_usage / (1024 * 1024),
            'cache_entries': len(self._result_cache),
            'memory_pool_objects': len(self._memory_pool),
            'memory_usage_history': self._memory_usage_history[-10:],  # Last 10 readings
            'memory_trend': self._calculate_memory_trend(),
            'memory_leaks': self._detect_memory_leaks(),
            'optimization_recommendations': self._get_memory_optimization_recommendations()
        }

        return profiling

    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        if len(self._memory_usage_history) < 10:
            return "insufficient_data"

        recent = self._memory_usage_history[-5:]
        older = self._memory_usage_history[-10:-5]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _detect_memory_leaks(self) -> List[str]:
        """Detect potential memory leaks."""
        leaks = []

        if len(self._memory_usage_history) < 20:
            return leaks

        # Check for consistent memory growth
        recent_trend = self._memory_usage_history[-10:]
        if all(recent_trend[i] < recent_trend[i+1] for i in range(len(recent_trend)-1)):
            leaks.append("Consistent memory growth detected")

        # Check for high memory usage
        if self._memory_peak_usage > self._max_cache_memory_mb * 2:
            leaks.append("Peak memory usage exceeds 2x cache limit")

        # Check for cache bloat
        if len(self._result_cache) > self._max_cache_size * 0.9:
            leaks.append("Cache size approaching limit")

        return leaks

    def _get_memory_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        current_memory = self._get_current_memory_usage()
        memory_ratio = current_memory / self._max_cache_memory_mb

        if memory_ratio > 0.8:
            recommendations.append("Consider increasing cache memory limit")

        if len(self._result_cache) > self._max_cache_size * 0.8:
            recommendations.append("Consider increasing cache size limit")

        if self._cache_stats['evictions'] > self._cache_stats['hits'] * 0.5:
            recommendations.append("High eviction rate - consider increasing cache size or memory")

        if self._cache_stats['memory_savings'] < 1024 * 1024:  # Less than 1MB saved
            recommendations.append("Low compression savings - consider disabling compression for small objects")

        return recommendations

    def force_garbage_collection(self):
        """Force garbage collection and memory cleanup."""
        tprint_info("🧹 Forcing garbage collection and memory cleanup")

        try:
            import gc

            # Clear cache if memory usage is high
            current_memory = self._get_current_memory_usage()
            if current_memory > self._max_cache_memory_mb * 1.5:
                tprint_warning("⚠️ High memory usage, clearing cache")
                self.clear_cache()

            # Force garbage collection
            collected = gc.collect()
            tprint_success(f"✅ Garbage collection completed: {collected} objects collected")

            # Update memory usage
            new_memory = self._get_current_memory_usage()
            memory_freed = current_memory - new_memory
            if memory_freed > 0:
                tprint_success(f"✅ Memory freed: {memory_freed:.2f}MB")

        except Exception as e:
            tprint_error(f"❌ Garbage collection failed: {e}")

    def clear_cache(self):
        """Clear all cache entries."""
        tprint_info("🧹 Clearing all cache entries")

        # Clear all cache-related dictionaries
        self._result_cache.clear()
        self._cache_access_times.clear()
        self._cache_creation_times.clear()
        self._cache_priority.clear()
        self._cache_access_frequency.clear()
        self._cache_ttl.clear()

        # Reset memory usage
        self._cache_memory_usage = 0

        # Reset cache statistics
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'decompressions': 0,
            'serializations': 0,
            'deserializations': 0,
            'memory_savings': 0,
            'total_operations': 0
        }

        tprint_success("✅ Cache cleared successfully")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_operations = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_operations * 100) if total_operations > 0 else 0

        stats = {
            'cache_enabled': self._cache_enabled,
            'cache_size': len(self._result_cache),
            'max_cache_size': self._max_cache_size,
            'cache_memory_mb': self._cache_memory_usage / (1024 * 1024),
            'max_cache_memory_mb': self._max_cache_memory_mb,
            'hit_rate_percent': hit_rate,
            'total_operations': total_operations,
            'compression_enabled': self._cache_compression_enabled,
            'serialization_enabled': self._cache_serialization_enabled,
            'memory_savings_mb': self._cache_stats['memory_savings'] / (1024 * 1024),
            'detailed_stats': self._cache_stats.copy()
        }

        return stats

    def adaptive_batch_size(self, data_size: int, operation_type: str,
                           available_memory_mb: float = None) -> int:
        """
        Calculate adaptive batch size based on data characteristics and available resources.

        Args:
            data_size: Size of the data to process
            operation_type: Type of operation being performed
            available_memory_mb: Available memory in MB (if None, will estimate)

        Returns:
            Optimal batch size
        """
        tprint_debug(f"🔄 Calculating adaptive batch size: data_size={data_size}, operation_type={operation_type}")

        try:
            # Estimate available memory if not provided
            if available_memory_mb is None:
                import psutil
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)

            # Base batch size based on operation type
            base_batch_sizes = {
                'rolling': 1000,
                'scaling': 5000,
                'correlation': 500,
                'covariance': 500,
                'custom': 2000
            }

            base_batch_size = base_batch_sizes.get(operation_type, 1000)

            # Adjust based on available memory
            memory_factor = min(1.0, available_memory_mb / 1024)  # Normalize to 1GB
            memory_adjusted_size = int(base_batch_size * memory_factor)

            # Adjust based on data size
            if data_size < 1000:
                size_factor = 1.0
            elif data_size < 10000:
                size_factor = 0.8
            elif data_size < 100000:
                size_factor = 0.6
            else:
                size_factor = 0.4

            size_adjusted_size = int(memory_adjusted_size * size_factor)

            # Ensure reasonable bounds
            adaptive_batch_size = max(100, min(size_adjusted_size, data_size))

            tprint_success(f"✅ Adaptive batch size calculated: {base_batch_size} -> {adaptive_batch_size} (memory: {available_memory_mb:.0f}MB)")
            return adaptive_batch_size

        except Exception as e:
            tprint_warning(f"⚠️ Adaptive batch size calculation failed: {e}, using default")
            return self.config.batch_size

    def validate_data_quality(self, data: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive data validation and quality checks.

        Args:
            data: Input data to validate

        Returns:
            Dictionary with validation results and recommendations
        """
        tprint_debug("🔍 Starting comprehensive data validation")

        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'quality_score': 0.0,
            'data_characteristics': {}
        }

        try:
            # Basic data type validation
            if not isinstance(data, (pd.Series, pd.DataFrame)):
                validation_results['errors'].append("Data must be a pandas Series or DataFrame")
                validation_results['is_valid'] = False
                return validation_results

            # Check for empty data
            if len(data) == 0:
                validation_results['errors'].append("Data is empty")
                validation_results['is_valid'] = False
                return validation_results

            # Data characteristics
            validation_results['data_characteristics'] = {
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
                'dtypes': data.dtypes.to_dict() if isinstance(data, pd.DataFrame) else {data.name: data.dtype}
            }

            # Check for missing values
            if isinstance(data, pd.Series):
                missing_count = data.isnull().sum()
                missing_pct = (missing_count / len(data)) * 100
                if missing_pct > 0:
                    validation_results['warnings'].append(f"Series has {missing_pct:.1f}% missing values")
                    if missing_pct > 50:
                        validation_results['errors'].append("Too many missing values (>50%)")
                        validation_results['is_valid'] = False
            else:  # DataFrame
                missing_counts = data.isnull().sum()
                missing_pcts = (missing_counts / len(data)) * 100
                high_missing_cols = missing_pcts[missing_pcts > 50].index.tolist()
                if high_missing_cols:
                    validation_results['errors'].append(f"Columns with >50% missing values: {high_missing_cols}")
                    validation_results['is_valid'] = False
                elif missing_counts.any():
                    validation_results['warnings'].append(f"DataFrame has missing values in {missing_counts[missing_counts > 0].count()} columns")

            # Check for infinite values
            if isinstance(data, pd.Series):
                inf_count = np.isinf(data).sum()
                if inf_count > 0:
                    validation_results['warnings'].append(f"Series has {inf_count} infinite values")
            else:  # DataFrame
                inf_counts = np.isinf(data).sum()
                inf_cols = inf_counts[inf_counts > 0].index.tolist()
                if inf_cols:
                    validation_results['warnings'].append(f"DataFrame has infinite values in columns: {inf_cols}")

            # Check for constant columns (DataFrame only)
            if isinstance(data, pd.DataFrame):
                constant_cols = data.nunique() == 1
                if constant_cols.any():
                    validation_results['warnings'].append(f"Constant columns detected: {constant_cols[constant_cols].index.tolist()}")
                    validation_results['recommendations'].append("Consider removing constant columns")

            # Check for duplicate rows (DataFrame only)
            if isinstance(data, pd.DataFrame):
                duplicate_count = data.duplicated().sum()
                if duplicate_count > 0:
                    validation_results['warnings'].append(f"DataFrame has {duplicate_count} duplicate rows")
                    validation_results['recommendations'].append("Consider removing duplicate rows")

            # Check data ranges for numeric data
            numeric_cols = data.select_dtypes(include=[np.number]).columns if isinstance(data, pd.DataFrame) else [data.name] if pd.api.types.is_numeric_dtype(data) else []
            for col in numeric_cols:
                if isinstance(data, pd.DataFrame):
                    col_data = data[col]
                else:
                    col_data = data

                if len(col_data) > 0:
                    min_val, max_val = col_data.min(), col_data.max()
                    if np.isnan(min_val) or np.isnan(max_val):
                        validation_results['warnings'].append(f"Column {col} contains only NaN values")
                    elif min_val == max_val:
                        validation_results['warnings'].append(f"Column {col} has no variation (constant value: {min_val})")
                    elif abs(max_val - min_val) < 1e-10:
                        validation_results['warnings'].append(f"Column {col} has very small range: {max_val - min_val}")

            # Calculate quality score
            quality_score = 100.0
            quality_score -= len(validation_results['errors']) * 20  # -20 for each error
            quality_score -= len(validation_results['warnings']) * 5  # -5 for each warning
            quality_score = max(0, quality_score)
            validation_results['quality_score'] = quality_score

            # Add recommendations based on findings
            if validation_results['quality_score'] < 80:
                validation_results['recommendations'].append("Consider data cleaning and preprocessing")
            if validation_results['data_characteristics']['memory_usage_mb'] > 100:
                validation_results['recommendations'].append("Consider memory optimization for large datasets")

            tprint_success(f"✅ Data validation completed: quality_score={quality_score:.1f}")

        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
            validation_results['is_valid'] = False
            tprint_error(f"❌ Data validation failed: {e}")

        return validation_results

    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics and optimization recommendations."""
        tprint_debug("📊 Generating performance analytics")

        stats = self.get_performance_stats()
        analytics = {
            'performance_summary': {
                'total_operations': stats.get('total_operations', 0),
                'total_time': stats.get('total_time', 0),
                'average_operation_time': stats.get('average_operation_time', 0),
                'error_rate': stats.get('errors', 0) / max(1, stats.get('total_operations', 1))
            },
            'efficiency_analysis': {
                'vectorbt_usage_rate': stats.get('vectorbt_usage_rate', 0),
                'gpu_usage_rate': stats.get('gpu_usage_rate', 0),
                'cache_hit_rate': stats.get('cache_hit_rate', 0),
                'memory_optimization_rate': stats.get('memory_optimizations', 0) / max(1, stats.get('total_operations', 1))
            },
            'bottleneck_analysis': self._analyze_bottlenecks(stats),
            'optimization_recommendations': self._generate_optimization_recommendations(stats),
            'resource_utilization': self._analyze_resource_utilization(stats)
        }

        return analytics

    def _analyze_bottlenecks(self, stats: Dict[str, Any]) -> List[str]:
        """Analyze performance bottlenecks."""
        bottlenecks = []

        total_ops = stats.get('total_operations', 0)
        if total_ops == 0:
            return bottlenecks

        # Check VectorBT usage
        vectorbt_rate = stats.get('vectorbt_usage_rate', 0)
        if vectorbt_rate < 0.5:
            bottlenecks.append(f"Low VectorBT usage ({vectorbt_rate:.1%}) - consider enabling VectorBT optimizations")

        # Check GPU usage
        gpu_rate = stats.get('gpu_usage_rate', 0)
        if self.config.enable_gpu and gpu_rate < 0.1:
            bottlenecks.append(f"Low GPU usage ({gpu_rate:.1%}) - consider using GPU for large datasets")

        # Check cache performance
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.3:
            bottlenecks.append(f"Low cache hit rate ({cache_hit_rate:.1%}) - consider increasing cache size")

        # Check error rate
        error_rate = stats.get('error_rate', 0)
        if error_rate > 0.1:
            bottlenecks.append(f"High error rate ({error_rate:.1%}) - check data quality and parameters")

        # Check memory optimizations
        memory_opt_rate = stats.get('memory_optimizations', 0) / total_ops
        if memory_opt_rate > 0.8:
            bottlenecks.append("Frequent memory optimizations - consider increasing memory budget or chunk size")

        return bottlenecks

    def _generate_optimization_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []

        total_ops = stats.get('total_operations', 0)
        if total_ops == 0:
            return recommendations

        # VectorBT recommendations
        vectorbt_rate = stats.get('vectorbt_usage_rate', 0)
        if vectorbt_rate < 0.5:
            recommendations.append("Enable VectorBT for better performance on financial data operations")

        # GPU recommendations
        if not self.config.enable_gpu and total_ops > 1000:
            recommendations.append("Consider enabling GPU acceleration for large-scale operations")

        # Memory recommendations
        memory_usage = stats.get('memory_optimizations', 0)
        if memory_usage / total_ops > 0.5:
            recommendations.append("Increase memory budget or enable more aggressive memory optimization")

        # Caching recommendations
        cache_hit_rate = stats.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.3:
            recommendations.append("Increase cache size or improve cache key generation")

        # Parallel processing recommendations
        if total_ops > 100 and not hasattr(self, 'parallel_processing_enabled'):
            recommendations.append("Consider enabling parallel processing for independent operations")

        return recommendations

    def _analyze_resource_utilization(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        utilization = {
            'cpu_efficiency': 0.0,
            'memory_efficiency': 0.0,
            'gpu_efficiency': 0.0,
            'cache_efficiency': 0.0
        }

        total_ops = stats.get('total_operations', 0)
        if total_ops == 0:
            return utilization

        # CPU efficiency (based on VectorBT usage)
        utilization['cpu_efficiency'] = stats.get('vectorbt_usage_rate', 0)

        # Memory efficiency (based on optimization rate)
        memory_opt_rate = stats.get('memory_optimizations', 0) / total_ops
        utilization['memory_efficiency'] = 1.0 - min(1.0, memory_opt_rate * 2)  # Lower is better

        # GPU efficiency
        utilization['gpu_efficiency'] = stats.get('gpu_usage_rate', 0)

        # Cache efficiency
        utilization['cache_efficiency'] = stats.get('cache_hit_rate', 0) / 100

        return utilization

    def optimize_operation(self, operation_func: Callable, config: Optional[OperationConfig] = None) -> OptimizationResult:
        """
        Optimize an operation using the unified vectorization manager.
        
        Args:
            operation_func: Function to optimize
            config: Operation configuration
            
        Returns:
            OptimizationResult with the operation result
        """
        start_time = time.time()
        
        try:
            # Execute the operation
            result = operation_func()
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            memory_used = self._get_current_memory_usage()
            
            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_time'] += execution_time
            
            return OptimizationResult(
                result=result,
                strategy_used=OptimizationStrategy.VECTORIZED_CPU,
                computation_time=execution_time,
                memory_used_mb=memory_used,
                performance_gain=1.0,
                metadata={'operation_type': 'custom', 'optimized': True}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats['errors'] += 1
            
            # Return error result
            return OptimizationResult(
                result=None,
                strategy_used=OptimizationStrategy.FALLBACK,
                computation_time=execution_time,
                memory_used_mb=0.0,
                performance_gain=0.0,
                metadata={'error': str(e), 'operation_type': 'custom', 'optimized': False}
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Ensure lazy imports are loaded
        _lazy_import_optimization_modules()

        stats = self.performance_stats.copy()

        # Add rolling optimizer stats
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats.update(rolling_stats)

        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            stats['rolling_usage_rate'] = stats['rolling_operations'] / stats['total_operations']
            stats['scaling_usage_rate'] = stats['scaling_operations'] / stats['total_operations']

            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['average_operation_time'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['gpu_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
            stats['rolling_usage_rate'] = 0
            stats['scaling_usage_rate'] = 0
            stats['cache_hit_rate'] = 0

        return stats

    def _validate_config(self, config: VectorizationConfig):
        """Validate configuration parameters with detailed error reporting."""
        tprint_debug("🔍 Validating UnifiedVectorizationManager configuration")

        if not isinstance(config, VectorizationConfig):
            raise VectorizationValidationError("Config must be a VectorizationConfig instance", "type_check", type(config))

        if not isinstance(config.enable_vectorbt, bool):
            raise VectorizationValidationError("enable_vectorbt must be a boolean", "type_check", config.enable_vectorbt)

        if not isinstance(config.enable_gpu, bool):
            raise VectorizationValidationError("enable_gpu must be a boolean", "type_check", config.enable_gpu)

        if not isinstance(config.memory_efficient, bool):
            raise VectorizationValidationError("memory_efficient must be a boolean", "type_check", config.memory_efficient)

        if not isinstance(config.chunk_size, int) or config.chunk_size <= 0:
            raise VectorizationValidationError("chunk_size must be a positive integer", "range_check", config.chunk_size)

        if not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise VectorizationValidationError("batch_size must be a positive integer", "range_check", config.batch_size)

        if config.max_memory_gb <= 0:
            raise VectorizationValidationError("max_memory_gb must be positive", "range_check", config.max_memory_gb)

        tprint_success("✅ Configuration validated successfully")

    def _validate_rolling_inputs(self, data: Union[pd.Series, pd.DataFrame],
                                operation: str, window: int):
        """Validate rolling operation inputs with comprehensive checks."""
        tprint_debug(f"🔍 Validating rolling inputs for {operation}")

        # Check data type
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise VectorizationValidationError("Data must be a pandas Series or DataFrame", "type_check", type(data))

        # Check data is not empty
        if len(data) == 0:
            raise VectorizationValidationError("Data cannot be empty", "empty_check", len(data))

        # Check window size
        if not isinstance(window, int) or window <= 0:
            raise VectorizationValidationError("Window must be a positive integer", "range_check", window)

        if window > len(data):
            raise VectorizationValidationError(f"Window size ({window}) cannot be larger than data length ({len(data)})", "range_check", window)

        # Check for supported operations
        supported_operations = ['mean', 'std', 'var', 'min', 'max', 'sum', 'quantile', 'skew', 'kurt', 'corr', 'cov', 'apply']
        if operation not in supported_operations:
            raise VectorizationValidationError(f"Unsupported operation: {operation}. Supported: {supported_operations}", "operation_check", operation)

        tprint_success(f"✅ Rolling inputs validated for {operation}")

    def _validate_scaling_inputs(self, data: Union[pd.Series, pd.DataFrame], method: str):
        """Validate scaling operation inputs."""
        tprint_debug(f"🔍 Validating scaling inputs for {method}")

        # Check data type
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise VectorizationValidationError("Data must be a pandas Series or DataFrame", "type_check", type(data))

        # Check data is not empty
        if len(data) == 0:
            raise VectorizationValidationError("Data cannot be empty", "empty_check", len(data))

        # Check for supported methods
        supported_methods = ['zscore', 'minmax', 'robust', 'quantile', 'winsorize', 'rank', 'clip']
        if method not in supported_methods:
            raise VectorizationValidationError(f"Unsupported scaling method: {method}. Supported: {supported_methods}", "method_check", method)

        tprint_success(f"✅ Scaling inputs validated for {method}")

    def reset_stats(self):
        """Reset all performance statistics."""
        tprint_info("🔄 Resetting UnifiedVectorizationManager performance statistics")

        # Ensure lazy imports are loaded
        _lazy_import_optimization_modules()

        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }

        if self.rolling_optimizer:
            self.rolling_optimizer.reset_stats()
        self._result_cache.clear()
        tprint_success("✅ Performance statistics reset")

    def optimize_for_workload(self, workload_type: WorkloadType, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        """Optimize vectorization operations for specific workload type."""
        if not self.config.enable_hardware_optimization or not self.hardware_manager:
            tprint_warning("⚠️ Hardware optimization not available")
            return

        try:
            self.config.workload_type = workload_type
            self.config.optimization_level = optimization_level
            self.hardware_manager.optimize_for_workload(workload_type, optimization_level)
            
            # Adjust vectorization settings based on workload
            if workload_type == WorkloadType.FEATURE_ENGINEERING:
                self.config.chunk_size = 1000
                self.config.batch_size = 10000
                self.config.enable_parallel = True
                tprint_info("🔧 Optimized for feature engineering workload")
            elif workload_type == WorkloadType.MODEL_TRAINING:
                self.config.chunk_size = 5000
                self.config.batch_size = 50000
                self.config.enable_parallel = False
                tprint_info("🔧 Optimized for model training workload")
            elif workload_type == WorkloadType.BACKTESTING:
                self.config.chunk_size = 2000
                self.config.batch_size = 20000
                self.config.enable_parallel = True
                tprint_info("🔧 Optimized for backtesting workload")
            
            tprint_success(f"✅ Optimized for {workload_type.value} workload")
            
        except Exception as e:
            tprint_warning(f"⚠️ Workload optimization failed: {e}")

    @contextmanager
    def hardware_optimization_context(self, workload_type: WorkloadType = None, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        """Context manager for hardware optimization during operations."""
        if not self.config.enable_hardware_optimization or not self.hardware_manager:
            yield
            return

        try:
            # Set optimization context
            if workload_type:
                self.optimize_for_workload(workload_type, optimization_level)
            
            # Enter hardware optimization context
            with self.hardware_manager.optimization_context(
                self.config.workload_type or WorkloadType.FEATURE_ENGINEERING,
                optimization_level
            ):
                yield
                
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization context failed: {e}")
            yield

    def get_hardware_status(self) -> Dict[str, Any]:
        """Get hardware optimization status and metrics."""
        if not self.config.enable_hardware_optimization or not self.hardware_manager:
            return {
                'hardware_optimization_enabled': False,
                'hardware_manager_available': False,
                'workload_type': None,
                'gpu_available': False,
                'memory_optimization': False
            }

        try:
            system_status = self.hardware_manager.get_system_status()
            return {
                'hardware_optimization_enabled': True,
                'hardware_manager_available': True,
                'workload_type': self.config.workload_type.value if self.config.workload_type else None,
                'optimization_level': self.config.optimization_level.value if self.config.optimization_level else None,
                'gpu_available': system_status.get('gpu_available', False),
                'memory_optimization': self.config.memory_efficient,
                'chunk_size': self.config.chunk_size,
                'batch_size': self.config.batch_size,
                'parallel_processing': self.config.enable_parallel,
                'system_status': system_status
            }
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get hardware status: {e}")
            return {
                'hardware_optimization_enabled': True,
                'hardware_manager_available': False,
                'error': str(e)
            }

    def _apply_hardware_optimizations(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply hardware-specific optimizations to data."""
        if not self.config.enable_hardware_optimization or not self.hardware_manager:
            return data

        try:
            # Apply memory optimizations
            if self.config.memory_efficient:
                optimized_data = self.hardware_manager.optimize_memory_usage(data)
                self.performance_stats['memory_optimizations'] += 1
                return optimized_data
            return data
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization failed: {e}")
            return data

    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        if not self.config.enable_monitoring:
            yield
            return

        start_time = time.time()
        start_memory = 0  # Could add memory monitoring here

        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(f"Operation {operation_name}: {execution_time:.3f}s")

# Global instance
_global_vectorization_manager = None

def get_unified_vectorization_manager(config: Optional[VectorizationConfig] = None) -> UnifiedVectorizationManager:
    """Get global unified vectorization manager instance."""
    global _global_vectorization_manager
    if _global_vectorization_manager is None:
        _global_vectorization_manager = UnifiedVectorizationManager(config)
    return _global_vectorization_manager

def create_optimized_vectorization_pipeline(enable_gpu: bool = False,
                                          memory_efficient: bool = True,
                                          enable_hardware_optimization: bool = True,
                                          workload_type: WorkloadType = None) -> UnifiedVectorizationManager:
    """
    Create an optimized vectorization pipeline.

    Args:
        enable_gpu: Enable GPU acceleration
        memory_efficient: Enable memory optimization
        enable_hardware_optimization: Enable hardware optimization integration
        workload_type: Workload type for hardware optimization

    Returns:
        Unified vectorization manager
    """
    config = VectorizationConfig(
        enable_vectorbt=True,
        enable_gpu=enable_gpu,
        enable_parallel=True,
        enable_hardware_optimization=enable_hardware_optimization,
        workload_type=workload_type or (WorkloadType.FEATURE_ENGINEERING if HARDWARE_AVAILABLE else None),
        optimization_level=OptimizationLevel.BALANCED if HARDWARE_AVAILABLE else None,
        memory_efficient=memory_efficient,
        max_memory_gb=8.0,
        chunk_size=1000,
        enable_monitoring=True,
        enable_profiling=False,
        batch_size=10000,
        enable_batch_processing=True,
        rolling_optimization_threshold=1000,
        enable_rolling_optimization=True
    )

    return UnifiedVectorizationManager(config)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(10000) * 0.01),
        'volume': np.random.randint(1000, 10000, 10000),
        'high': 100 + np.cumsum(np.random.randn(10000) * 0.01) + np.abs(np.random.randn(10000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(10000) * 0.01) - np.abs(np.random.randn(10000) * 0.5)
    })

    print("Original data shape:", data.shape)
    print("Original memory usage:", data.memory_usage(deep=True).sum() / (1024**3), "GB")

    # Create unified vectorization manager
    manager = get_unified_vectorization_manager(
        VectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
    )

    # Test rolling operations
    print("\nTesting rolling operations...")
    rolling_mean = manager.rolling_operation(data['close'], 'mean', window=20)
    rolling_std = manager.rolling_operation(data['close'], 'std', window=20)
    print(f"Rolling mean shape: {rolling_mean.shape}")
    print(f"Rolling std shape: {rolling_std.shape}")

    # Test scaling
    print("\nTesting scaling...")
    scaled_close = manager.scale_data(data['close'], method='zscore')
    print(f"Scaled close shape: {scaled_close.shape}")

    # Test batch processing
    print("\nTesting batch processing...")
    feature_configs = [
        {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
        {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
        {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
        {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}},
        {'name': 'volume_scaled', 'type': 'scaling', 'params': {'method': 'minmax', 'column': 'volume'}}
    ]

    features = manager.batch_process_features(data, feature_configs)
    print(f"Generated features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")

    # Test memory optimization
    print("\nTesting memory optimization...")
    optimized_data = manager.optimize_dataframe(data)
    print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum() / (1024**3):.3f}GB")

    # Get performance stats
    stats = manager.get_performance_stats()
    print(f"\nPerformance stats: {stats}")

    print("\nUnified vectorization pipeline test completed successfully!")
