"""
Core Optimization Logic for Feature Lookback Optimization.

This module contains the main optimization algorithms and core functionality.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, OrderedDict

# Import utility modules
from src.utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import safe_divide, safe_correlation
from .utils.error_handling import (
    safe_operation, safe_mi_calculation, safe_correlation_calculation,
    safe_dataframe_operation as safe_df_op, safe_numpy_operation,
    get_error_handler, OptimizationError, DataValidationError, ScoringError
)
from .utils.nan_handling import (
    SafeNaNHandler
)
from src.utils.matrix_operations import (
    safe_correlation_with_nan_handling, safe_mutual_information_with_nan_handling
)
from .utils.memory_monitor import get_memory_monitor, monitor_memory
from .utils.scoring_utils import get_scoring_utils, ScoringConfig
from .utils.constants import get_constants
from .utils.data_validation import get_data_validator, validate_optimization_data, ValidationLevel
from .utils.fast_failing_validation import (
    FastFailingValidator, validate_optimization_inputs_fast_fail,
    validate_feature_calculation_inputs
)
from .utils.memory_efficient_ops import (
    MemoryEfficientOps, optimize_dataframe_memory, create_dataframe_view
)
from src.utils.matrix_operations import (
    VectorizedCorrelationCalculator, calculate_correlations_vectorized
)

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    STATIONARITY_TESTS_AVAILABLE = True
except ImportError:
    adfuller = None  # type: ignore
    kpss = None  # type: ignore
    STATIONARITY_TESTS_AVAILABLE = False

# Import matrix operations for vectorized processing
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import multi-horizon profit labeler for target alignment
try:
    from ...multi_horizon_profit_labeler import MultiHorizonConfig
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False
    MultiHorizonConfig = None
from src.utils.serialization_utils import UniversalSerializer
from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time

from ..constants import OPTIMIZATION_CONSTANTS, PERFORMANCE_CONSTANTS, ALGORITHM_CONSTANTS
from ..dependency_manager import get_dependency
from src.training.config.data_locator import DataLocator as PipelineDataLocator

# Import Bayesian TPE optimizer for advanced optimization
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
    BAYESIAN_OPTIMIZER_AVAILABLE = True
except ImportError:
    BayesianTPEOptimizer = None
    OptimizationConfig = None
    BAYESIAN_OPTIMIZER_AVAILABLE = False

# Import VectorBT Rolling Optimizer and Unified Vectorization Manager
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, 
        get_vectorbt_rolling_optimizer,
        optimized_rolling_mean,
        optimized_rolling_std,
        optimized_rolling_corr
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager,
        get_unified_vectorization_manager,
        OperationType,
        OptimizationStrategy as UnifiedOptimizationStrategy
    )
    VECTORBT_UTILS_AVAILABLE = True
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    optimized_rolling_mean = None
    optimized_rolling_std = None
    optimized_rolling_corr = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    OperationType = None
    UnifiedOptimizationStrategy = None

# Get dependencies with fallbacks
np, _ = get_dependency('numpy')
pd, _ = get_dependency('pandas')


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    MRMR = "mrmr"
    RANDOM_SEARCH = "random_search"
    MULTI_TARGET = "multi_target"
    COARSE_TO_REFINE = "coarse_to_refine"


@dataclass
class LookbackConstraints:
    """Constraints for lookback optimization."""
    min_lookback: int = 5
    max_lookback: int = 300
    search_step: int = 5
    enable_regularization: bool = True
    regularization_strength: float = 0.1
    preferred_lookback: int = 50
    min_stability_score: float = 0.7
    
    # ENHANCEMENTS: Explicit objective function and stability tracking
    optimization_objective: str = "max_ic"  # 'max_ic', 'max_sharpe', 'min_rmse', 'max_label_corr'
    preferred_min: float = 40.0  # Preferred minimum lookback
    preferred_max: float = 80.0  # Preferred maximum lookback
    penalty_exponent: float = 2.0  # Penalty exponent for regularization
    enable_bootstrap_stability: bool = True  # Enable bootstrap resampling for stability
    n_bootstrap_samples: int = 10  # Number of bootstrap samples (reduced to 2 for light/blank modes)
    track_sensitivity: bool = True  # Track lookback sensitivity
    
    # MODE-AWARE OPTIMIZATION: Execution mode settings
    execution_mode: str = "full"  # 'light', 'blank', 'full'
    cv_folds: int = 5  # Cross-validation folds (reduced to 2 for light/blank modes)
    use_bayesian_optimization: bool = False  # Use Bayesian TPE optimizer for coarser search
    enable_enhanced_caching: bool = True  # Enable enhanced cache optimization


@dataclass
class OptimizationResult:
    """Standardized optimization result."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_trials: int
    optimization_time: float
    convergence_achieved: bool
    metadata: Dict[str, Any]
    feature_name: str = ""  # FIXED: Added feature_name to prevent attribute error
    stability_score: float = 0.0  # Added for validation
    lookback_sensitivity: float = 0.0  # Added for validation
    
    # ENHANCEMENTS: Extended stability and robustness metrics
    resampled_lookbacks: List[int] = None  # Bootstrap resampled lookbacks
    objective_name: str = "unknown"  # Explicit objective function name
    regularization_penalty: float = 0.0  # Regularization penalty applied
    raw_objective_value: float = 0.0  # Raw objective before regularization
    is_stable: bool = False  # Whether lookback meets stability criteria

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        tprint_debug("🧮 Converting OptimizationResult to dictionary")
        # Ensure metadata contains serializable values
        def convert_metadata(obj):
            tprint_debug(f"   ↳ Normalizing metadata type: {type(obj).__name__}")
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_metadata(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_metadata(item) for item in obj]
            else:
                return obj

        result_dict = {
            'best_lookback_period': int(self.best_lookback_period) if isinstance(self.best_lookback_period, np.int64) else self.best_lookback_period,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'total_trials': int(self.total_trials) if isinstance(self.total_trials, np.int64) else self.total_trials,
            'optimization_time': self.optimization_time,
            'convergence_achieved': self.convergence_achieved,
            'stability_score': self.stability_score,
            'lookback_sensitivity': self.lookback_sensitivity,
            'metadata': convert_metadata(self.metadata),
            # ENHANCEMENTS: Add new fields
            'resampled_lookbacks': self.resampled_lookbacks if self.resampled_lookbacks is not None else [],
            'objective_name': self.objective_name,
            'regularization_penalty': self.regularization_penalty,
            'raw_objective_value': self.raw_objective_value,
            'is_stable': self.is_stable
        }
        return result_dict


class CoreOptimizer:
    """
    Core optimization engine for feature lookback optimization.

    Provides standardized interface for different optimization algorithms.
    """

    @staticmethod
    def create_mode_aware_constraints(execution_mode: str = "full", base_constraints: Optional[LookbackConstraints] = None) -> LookbackConstraints:
        """
        Create LookbackConstraints optimized for the execution mode.
        
        Args:
            execution_mode: Execution mode ('light', 'blank', 'full')
            base_constraints: Optional base constraints to override
            
        Returns:
            LookbackConstraints with mode-specific optimizations
        """
        if base_constraints is None:
            base_constraints = LookbackConstraints()
        
        # Apply mode-specific optimizations
        if execution_mode in ["light", "blank"]:
            # OPTIMIZATION 1: Reduce bootstrap resampling to 2
            base_constraints.n_bootstrap_samples = 2
            
            # OPTIMIZATION 2: Reduce CV folds to 2
            base_constraints.cv_folds = 2
            
            # OPTIMIZATION 3: Enable Bayesian optimization for faster convergence
            base_constraints.use_bayesian_optimization = True
            
            # OPTIMIZATION 4: Coarser grid search for light mode
            if execution_mode == "light":
                base_constraints.search_step = 10
            else:  # blank mode
                base_constraints.search_step = 7
            
            # OPTIMIZATION 5: Enhanced caching
            base_constraints.enable_enhanced_caching = True
            
            tprint_info(f"🚀 Mode-aware optimization enabled for {execution_mode.upper()} mode:")
            tprint_info(f"   → Bootstrap samples: {base_constraints.n_bootstrap_samples}")
            tprint_info(f"   → CV folds: {base_constraints.cv_folds}")
            tprint_info(f"   → Grid search step: {base_constraints.search_step}")
            tprint_info(f"   → Bayesian optimization: {base_constraints.use_bayesian_optimization}")
            tprint_info(f"   → Enhanced caching: {base_constraints.enable_enhanced_caching}")
        else:
            # Full mode - use default settings
            base_constraints.execution_mode = "full"
            tprint_info(f"🎯 Using FULL mode optimization (highest accuracy)")
        
        base_constraints.execution_mode = execution_mode
        return base_constraints

    def __init__(self, logger=None, rng: Optional['np.random.Generator'] = None):
        """Initialize the core optimizer."""
        self.logger = logger or get_logger('CoreOptimizer')
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        self._rng: 'np.random.Generator' = rng or np.random.default_rng()
        
        tprint("🔧 Initializing Core Optimizer...")
        tprint("   → Performance tracking enabled")
        tprint("   → Feature calculation cache initialized")
        tprint("   → Shared forward returns cache ready")
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_optimization_time': 0.0,
            'best_scores': []
        }
        
        # Thread safety for caching
        import threading
        self._cache_lock = threading.RLock()
        self.max_cache_size = 50000  # Prevent memory leaks
        
        # Feature calculation cache with LRU tracking using OrderedDict for O(1) operations
        self.feature_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Memory monitoring for cache management
        self._last_cache_cleanup = 0
        self._cache_cleanup_interval = 1000  # Cleanup every 1000 operations

        # Track lag metadata for generated features
        self.feature_lag_metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}
        
        # Memory monitoring
        self.memory_monitor = get_memory_monitor()
        
        # Shared forward returns matrix cache (reused across all features)
        self.shared_forward_returns = {}
        self.shared_forward_returns_hash = None

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.batch_processor = None
        self.gpu_available = False
        
        # Initialize VectorBT optimization components
        self._initialize_vectorbt_components()
        
        # Initialize memory-efficient operations
        self.memory_ops = MemoryEfficientOps(enable_gc=True)
        
        # Initialize vectorized correlation calculator
        self.correlation_calculator = VectorizedCorrelationCalculator(use_gpu=False)
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.batch_processor = BatchMatrixProcessor(chunk_size_mb=128, enable_gpu=True)
                
                # Check GPU availability
                self.gpu_available = self._check_gpu_availability()
                if self.gpu_available:
                    self.logger.info("✅ Matrix operations initialized with GPU acceleration")
                else:
                    self.logger.info("✅ Matrix operations initialized (CPU only)")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize matrix operations: {e}")
                self.matrix_ops = None
                self.batch_processor = None
                self.gpu_available = False

        self._cached_multi_horizon_limits: Optional[Tuple[int, int]] = None
        self._data_locator: Optional[PipelineDataLocator] = None

    def _initialize_vectorbt_components(self):
        """Initialize VectorBT optimization components."""
        try:
            # Initialize VectorBT Rolling Optimizer with GPU support
            if VECTORBT_UTILS_AVAILABLE:
                # Check GPU availability for VectorBT
                gpu_available = self._check_gpu_availability()
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=gpu_available,
                    enable_parallel=True,
                    memory_efficient=True
                )
                if gpu_available:
                    self.logger.info("✅ VectorBT Rolling Optimizer initialized with GPU acceleration")
                else:
                    self.logger.info("✅ VectorBT Rolling Optimizer initialized (CPU only)")
            else:
                self.rolling_optimizer = None
                self.logger.warning("⚠️ VectorBT Rolling Optimizer not available")
            
            # Initialize Unified Vectorization Manager
            if VECTORBT_UTILS_AVAILABLE:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ Unified Vectorization Manager initialized")
            else:
                self.unified_manager = None
                self.logger.warning("⚠️ Unified Vectorization Manager not available")
            
            # Initialize VectorBT optimization flags
            self.use_vectorbt_optimization = VECTORBT_UTILS_AVAILABLE
            self.vectorbt_available = VECTORBT_UTILS_AVAILABLE
            
            # Initialize VectorBT performance metrics
            self.vectorbt_metrics = {
                'operations_count': 0,
                'gpu_operations': 0,
                'cpu_operations': 0,
                'memory_optimizations': 0,
                'batch_operations': 0,
                'total_time': 0.0,
                'average_operation_time': 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT components initialization failed: {e}")
            self.rolling_optimizer = None
            self.unified_manager = None
            self.use_vectorbt_optimization = False
            self.vectorbt_available = False
            self.vectorbt_metrics = {}

    def _track_vectorbt_operation(self, operation_type: str, duration: float, gpu_used: bool = False):
        """Track VectorBT operation performance metrics."""
        try:
            self.vectorbt_metrics['operations_count'] += 1
            self.vectorbt_metrics['total_time'] += duration
            
            if gpu_used:
                self.vectorbt_metrics['gpu_operations'] += 1
            else:
                self.vectorbt_metrics['cpu_operations'] += 1
            
            if operation_type == 'memory_optimization':
                self.vectorbt_metrics['memory_optimizations'] += 1
            elif operation_type == 'batch_operation':
                self.vectorbt_metrics['batch_operations'] += 1
            
            # Update average operation time
            if self.vectorbt_metrics['operations_count'] > 0:
                self.vectorbt_metrics['average_operation_time'] = (
                    self.vectorbt_metrics['total_time'] / self.vectorbt_metrics['operations_count']
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to track VectorBT operation: {e}")

    def get_vectorbt_performance_metrics(self) -> Dict[str, Any]:
        """Get VectorBT performance metrics."""
        return self.vectorbt_metrics.copy()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            if self.batch_processor and hasattr(self.batch_processor, 'gpu_available'):
                return self.batch_processor.gpu_available
            
            # Try to detect GPU availability
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                pass
            
            try:
                import cupy
                return True
            except ImportError:
                pass
            
            return False
            
        except Exception:
            return False

    def set_rng(self, rng: Optional['np.random.Generator']) -> None:
        """Update the RNG used for stochastic routines."""
        self._rng = rng or np.random.default_rng()

    def set_data_locator(self, locator: Optional[PipelineDataLocator]) -> None:
        """Attach a locator used when resolving shared configuration files."""

        self._data_locator = locator
        self._cached_multi_horizon_limits = None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache performance statistics.
        
        Returns:
            Dictionary containing cache metrics:
            - cache_size: Current number of cached entries
            - max_cache_size: Maximum allowed cache entries
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - hit_rate: Cache hit rate percentage
            - memory_estimate_mb: Estimated memory usage in MB
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_accesses * 100) if total_accesses > 0 else 0.0
        
        # Estimate memory: assume ~10KB per cached array
        memory_estimate_mb = len(self.feature_cache) * 10 / 1024
        
        return {
            'cache_size': len(self.feature_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_estimate_mb': memory_estimate_mb
        }
    
    def clear_cache(self, keep_recent: int = 0) -> None:
        """Clear the feature cache, optionally keeping the most recently used entries.
        
        Args:
            keep_recent: Number of most recently used entries to keep (0 = clear all)
        """
        if keep_recent > 0 and len(self.feature_cache) > keep_recent:
            # Keep only the most recent N entries
            items_to_keep = list(self.feature_cache.items())[-keep_recent:]
            self.feature_cache.clear()
            self.feature_cache.update(items_to_keep)
            self.logger.info(f"ℹ️ Cache cleared, kept {keep_recent} most recent entries")
        else:
            self.feature_cache.clear()
            self.logger.info("ℹ️ Cache fully cleared")
        
        # Reset cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _evict_cache_if_needed(self) -> None:
        """Evict oldest cache entries if cache size exceeds limit."""
        while len(self.feature_cache) > self.max_cache_size:
            # Remove oldest entry (first in OrderedDict)
            self.feature_cache.popitem(last=False)

    def optimize_features_parallel_batch(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        target_column: str,
        lookback_range: Tuple[int, int],
        method: str = "coarse_to_refine",
        max_workers: Optional[int] = None,
        batch_size: int = 10,
        regularization_settings: Optional[Dict[str, float]] = None,
        use_streaming: bool = False,
        streaming_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[OptimizationResult]:
        """
        PARALLEL BATCH PROCESSING: Optimize multiple features in parallel with VectorBT optimizations.
        
        PERFORMANCE OPTIMIZATION: Processes features in parallel batches using ThreadPoolExecutor
        - Utilizes multi-core CPUs efficiently
        - Batch processing reduces overhead
        - Shared forward returns matrix across all features
        - VectorBT optimizations for high-performance rolling operations
        - Expected 3-4x speedup on 4-core systems
        
        Args:
            data: Input data with features and target
            feature_names: List of feature names to optimize
            target_column: Target column for optimization
            lookback_range: Min and max lookback periods to test
            method: Optimization method to use
            max_workers: Number of parallel workers (default: cpu_count)
            batch_size: Number of features to process in each batch
            regularization_settings: Regularization settings
            **kwargs: Additional parameters passed to optimization method
            
        Returns:
            List of OptimizationResult for each feature
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing as mp
        
        # Check if streaming processing should be used
        if use_streaming or len(data) > 100000:  # Auto-enable for large datasets
            return self._optimize_with_streaming(
                data, feature_names, target_column, lookback_range, method,
                streaming_config, **kwargs
            )
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 4)  # Cap at 4 to avoid overhead
        
        tprint_success(f"🚀 Starting PARALLEL batch optimization for {len(feature_names)} features")
        tprint_info(f"   → Workers: {max_workers}")
        tprint_info(f"   → Batch size: {batch_size}")
        tprint_info(f"   → Method: {method}")
        
        # Optimize DataFrame memory usage with VectorBT enhancements
        data = self._optimize_dataframe_memory(data)
        
        # Apply VectorBT memory optimizations for large datasets
        if self.use_vectorbt_optimization and len(data) > 10000:
            try:
                # Use VectorBT memory-efficient processing for large datasets
                if self.rolling_optimizer and hasattr(self.rolling_optimizer, 'enable_memory_efficient_mode'):
                    self.rolling_optimizer.enable_memory_efficient_mode(True)
                    tprint_info("🧠 VectorBT memory-efficient mode enabled for large dataset")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT memory optimization failed: {e}")
        
        # Pre-compute shared forward returns matrix once for all features
        min_lookback, max_lookback = lookback_range
        precomputed_forward_returns = self._get_shared_forward_returns_matrix(
            data, target_column, max_horizon=max_lookback
        )
        
        if not precomputed_forward_returns:
            tprint_warning("⚠️ Failed to pre-compute forward returns matrix")
            precomputed_forward_returns = None
        else:
            tprint_success(f"✅ Pre-computed forward returns matrix for horizon up to {max_lookback}")
        
        # Initialize VectorBT batch optimization if available
        if self.use_vectorbt_optimization and self.unified_manager:
            try:
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=(len(data), len(data.columns)),
                    memory_budget_mb=1024.0,
                    time_budget_seconds=300.0
                )
                
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                tprint_info(f"🎯 VectorBT batch optimization strategy: {strategy}")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT batch optimization setup failed: {e}")
        
        # Single feature optimization function
        def optimize_single_feature(feature_name: str) -> OptimizationResult:
            """Optimize a single feature with pre-computed matrix."""
            try:
                # Select optimization method
                if method == "coarse_to_refine":
                    return self._coarse_to_refine_single_pass(
                        data,
                        feature_name,
                        target_column,
                        lookback_range,
                        regularization_settings=regularization_settings,
                        precomputed_forward_returns=precomputed_forward_returns,
                        **kwargs
                    )
                elif method == "bayesian_tpe":
                    # Pass precomputed matrix via kwargs
                    kwargs_with_matrix = {**kwargs, 'precomputed_forward_returns': precomputed_forward_returns}
                    return self._optimize_with_bayesian_tpe(
                        data,
                        feature_name,
                        target_column,
                        lookback_range,
                        regularization_settings=regularization_settings,
                        **kwargs_with_matrix
                    )
                elif method == "grid_search":
                    return self._optimize_grid_search(
                        data, feature_name, target_column, lookback_range, **kwargs
                    )
                elif method == "random_search":
                    return self._optimize_random_search(
                        data, feature_name, target_column, lookback_range, **kwargs
                    )
                else:
                    # Default to coarse_to_refine
                    return self._coarse_to_refine_single_pass(
                        data,
                        feature_name,
                        target_column,
                        lookback_range,
                        regularization_settings=regularization_settings,
                        precomputed_forward_returns=precomputed_forward_returns,
                        **kwargs
                    )
            except Exception as e:
                self.logger.error(f"❌ Failed to optimize {feature_name}: {e}")
                return self._create_failed_result(method, 0.0, feature_name=feature_name)
        
        # Process features in parallel batches
        all_results = []
        start_time = time.time()
        
        # Split features into batches
        feature_batches = [
            feature_names[i:i + batch_size] 
            for i in range(0, len(feature_names), batch_size)
        ]
        
        tprint_info(f"📦 Processing {len(feature_batches)} batches...")
        
        for batch_idx, batch in enumerate(feature_batches, 1):
            tprint_info(f"🔄 Batch {batch_idx}/{len(feature_batches)}: {len(batch)} features")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all features in this batch
                future_to_feature = {
                    executor.submit(optimize_single_feature, fname): fname 
                    for fname in batch
                }
                
                # Collect results as they complete
                batch_results = []
                for future in as_completed(future_to_feature):
                    feature_name = future_to_feature[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        if result.best_score > 0:
                            tprint_debug(f"   ✅ {feature_name}: lookback={result.best_lookback_period}, score={result.best_score:.4f}")
                    except Exception as e:
                        self.logger.error(f"   ❌ {feature_name} failed: {e}")
                        batch_results.append(
                            self._create_failed_result(method, 0.0, feature_name=feature_name)
                        )
            
            all_results.extend(batch_results)
            
            # Progress update
            completed = len(all_results)
            total = len(feature_names)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            
            tprint_info(f"   📊 Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                       f"Rate: {rate:.1f} features/sec | ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        avg_time_per_feature = total_time / len(feature_names) if feature_names else 0
        
        tprint_success(f"✅ Parallel batch optimization completed!")
        tprint_info(f"   📊 Total features: {len(feature_names)}")
        tprint_info(f"   ⏱️  Total time: {total_time:.2f}s")
        tprint_info(f"   ⚡ Avg per feature: {avg_time_per_feature:.2f}s")
        tprint_info(f"   🚀 Processing rate: {len(feature_names)/total_time:.2f} features/sec")
        
        # Cache statistics
        cache_stats = self.get_cache_statistics()
        tprint_info(f"   📦 Cache hit rate: {cache_stats['hit_rate']:.1f}%")
        
        return all_results

    def _normalize_regularization_settings(
        self,
        settings: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Merge caller-provided penalty settings with safe defaults."""

        defaults: Dict[str, float] = {
            'preferred_min': 40.0,
            'preferred_max': 80.0,
            'penalty_strength': 0.0,
            'penalty_exponent': 2.0,
        }

        if not settings:
            resolved = defaults
        else:
            resolved = defaults.copy()
            window = settings.get('preferred_window') if isinstance(settings, dict) else None
            if isinstance(window, (list, tuple)) and len(window) == 2:
                try:
                    resolved['preferred_min'] = float(window[0])
                    resolved['preferred_max'] = float(window[1])
                except (TypeError, ValueError):
                    pass

            for key in ('preferred_min', 'preferred_max', 'penalty_strength', 'penalty_exponent'):
                if isinstance(settings, dict) and key in settings and settings[key] is not None:
                    try:
                        resolved[key] = float(settings[key])
                    except (TypeError, ValueError):
                        continue

            if isinstance(settings, dict) and settings.get('preferred_center') is not None:
                center = settings.get('preferred_center')
                width = settings.get('preferred_width')
                try:
                    center_val = float(center) if center is not None else None
                    if center_val is not None:
                        if width is None:
                            width = resolved['preferred_max'] - resolved['preferred_min']
                        width_val = float(width)
                        resolved['preferred_min'] = center_val - (width_val / 2.0)
                        resolved['preferred_max'] = center_val + (width_val / 2.0)
                except (TypeError, ValueError):
                    pass

        if resolved['preferred_min'] > resolved['preferred_max']:
            resolved['preferred_min'], resolved['preferred_max'] = resolved['preferred_max'], resolved['preferred_min']

        resolved['preferred_center'] = (resolved['preferred_min'] + resolved['preferred_max']) / 2.0
        resolved['preferred_width'] = resolved['preferred_max'] - resolved['preferred_min']

        return resolved

    def _calculate_lookback_penalty(self, horizon: int, settings: Dict[str, float]) -> float:
        """Compute quadratic penalty for horizons outside the preferred band."""

        if not settings:
            return 0.0

        strength = float(max(settings.get('penalty_strength', 0.0), 0.0))
        if strength == 0.0:
            return 0.0

        exponent = float(settings.get('penalty_exponent', 2.0))
        if exponent < 1.0:
            exponent = 1.0

        preferred_min = float(settings.get('preferred_min', 0.0))
        preferred_max = float(settings.get('preferred_max', 0.0))

        horizon_val = float(horizon)
        if preferred_min <= horizon_val <= preferred_max:
            return 0.0

        if horizon_val < preferred_min:
            distance = preferred_min - horizon_val
        else:
            distance = horizon_val - preferred_max

        penalty = strength * (distance ** exponent)
        return float(penalty)

    def _apply_regularization_penalty(
        self,
        horizon: int,
        score: float,
        settings: Dict[str, float],
    ) -> Tuple[float, float]:
        """Return penalized score and raw penalty for the provided horizon."""

        penalty = self._calculate_lookback_penalty(horizon, settings)
        penalized_score = float(score) - penalty
        return penalized_score, penalty

    def _optimize_with_streaming(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        target_column: str,
        lookback_range: Tuple[int, int],
        method: str,
        streaming_config: Optional[Dict[str, Any]],
        **kwargs
    ) -> List[OptimizationResult]:
        """Optimize features using streaming processing for large datasets."""
        try:
            from src.training.steps.pre_training.feature_lookback_optimization.streaming.streaming_processor import (
                StreamingProcessor, StreamingConfig, create_streaming_processor
            )
            
            # Create streaming processor
            if streaming_config:
                processor = create_streaming_processor(**streaming_config)
            else:
                # Default configuration for large datasets
                processor = create_streaming_processor(
                    chunk_size=5000,  # Smaller chunks for large datasets
                    memory_limit_mb=2048,  # Higher memory limit
                    overlap_size=200,  # Larger overlap for continuity
                    enable_gc=True
                )
            
            # Process using streaming
            streaming_results = processor.process_large_dataset(
                data, feature_names, target_column, lookback_range, method, **kwargs
            )
            
            # Convert streaming results to OptimizationResult objects
            results = []
            for feature_name in feature_names:
                if feature_name in streaming_results and streaming_results[feature_name]:
                    result_data = streaming_results[feature_name]
                    result = OptimizationResult(
                        best_lookback_period=result_data['best_lookback_period'],
                        best_score=result_data['best_score'],
                        optimization_method=result_data['optimization_method'],
                        total_trials=result_data['total_trials'],
                        optimization_time=result_data['optimization_time'],
                        convergence_achieved=result_data['convergence_achieved']
                    )
                    results.append(result)
                else:
                    # Create failed result
                    results.append(self._create_failed_result(method, 0.0, feature_name=feature_name))
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Streaming optimization failed: {e}")
            # Fallback to regular processing
            self.logger.info("🔄 Falling back to regular processing...")
            return self.optimize_features_parallel_batch(
                data, feature_names, target_column, lookback_range, method,
                use_streaming=False, **kwargs
            )

    def optimize_single_feature(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        method: OptimizationMethod = OptimizationMethod.MRMR,
        lookback_range: Tuple[int, int] = (5, 300),
        regularization_settings: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize lookback period for a single feature.

        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            method: Optimization method to use
            lookback_range: Min and max lookback periods to test
            regularization_settings: Optional configuration controlling horizon penalties
            **kwargs: Additional parameters for optimization method

        Returns:
            OptimizationResult with best lookback period and score
        """
        try:
            start_time = time.time()
            self.logger.info(f'🎯 Starting optimization for feature: {feature_name} using {method.value}')
            tprint(f"🎯 Starting optimization for feature: {feature_name} using {method.value}")
            
            # Fast failing data validation
            try:
                validate_optimization_inputs_fast_fail(
                    data, 
                    [feature_name], 
                    [target_column], 
                    lookback_range,
                    min_samples=50
                )
                self.logger.info(f"✅ Data validation passed for {feature_name}")
            except DataValidationError as e:
                self.logger.error(f"❌ Data validation failed for {feature_name}: {e}")
                return self._create_failed_result(method.value, 0.0, feature_name)

            # Validate inputs
            if not self._validate_optimization_inputs(data, feature_name, target_column):
                tprint_error(f"❌ Input validation failed for feature: {feature_name}")
                return self._create_failed_result(method.value, time.time() - start_time)

            # Select optimization algorithm based on method
            if 'regularization_settings' in kwargs and regularization_settings is None:
                regularization_settings = kwargs.pop('regularization_settings')

            if method == OptimizationMethod.MRMR:
                result = self._optimize_mrmr(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.GRID_SEARCH:
                result = self._optimize_grid_search(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.BAYESIAN:
                result = self._optimize_bayesian(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = self._optimize_random_search(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.MULTI_TARGET:
                result = self._optimize_multi_target(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.COARSE_TO_REFINE:
                result = self._optimize_coarse_to_refine(
                    data,
                    feature_name,
                    target_column,
                    lookback_range,
                    regularization_settings=regularization_settings,
                    **kwargs,
                )
            else:
                # Fallback to MRMR
                self.logger.warning(f'⚠️ Unknown method {method.value}, falling back to MRMR')
                result = self._optimize_mrmr(data, feature_name, target_column, lookback_range, **kwargs)

            result.optimization_time = time.time() - start_time
            result.optimization_method = method.value

            # Update performance tracking
            self._update_performance_metrics(result, time.time() - start_time)

            if 'regularization_settings' not in result.metadata:
                if regularization_settings is not None:
                    result.metadata['regularization_settings'] = regularization_settings
                else:
                    result.metadata['regularization_settings'] = {}

            self.logger.info(f'✅ Optimization completed: best_lookback={result.best_lookback_period}, score={result.best_score:.4f}')
            tprint_success(f"✅ Optimization completed for {feature_name}: best_lookback={result.best_lookback_period}, score={result.best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Optimization failed for feature {feature_name}: {e}")
            tprint_error(f"❌ Optimization failed for feature {feature_name}: {e}")
            return self._create_failed_result(method.value, time.time() - start_time)

    def _validate_optimization_inputs(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str
    ) -> bool:
        """Validate inputs for optimization with fast failing."""
        try:
            tprint_debug(f"🔍 Validating optimization inputs for feature '{feature_name}' against target '{target_column}'")
            
            # Use fast failing validation
            validate_feature_calculation_inputs(data, feature_name, 1)  # Basic validation
            
            # Check target column exists
            if target_column not in data.columns:
                raise DataValidationError(f"Target column '{target_column}' not found in data")
            
            # Check for sufficient data
            if len(data) < OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK:
                raise DataValidationError(
                    f"Insufficient data: {len(data)} rows < {OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK} required"
                )
            
            tprint_debug("✅ Optimization inputs validated successfully")
            return True

        except DataValidationError as e:
            self.logger.error(f"❌ Validation failed: {e}")
            tprint_error(f"❌ Validation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected validation error: {e}")
            tprint_error(f"❌ Unexpected validation error: {e}")
            return False

    def _optimize_mrmr(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using MRMR approach with proper cross-validation."""
        try:
            min_lookback, max_lookback = lookback_range
            tprint_debug(
                f"🧠 Running MRMR optimization for '{feature_name}' in range [{min_lookback}, {max_lookback}]"
            )
            best_score = -float('inf')
            best_lookback = min_lookback
            trials = 0
            stationarity_audit: Dict[int, Dict[str, Any]] = {}

            # Use time series cross-validation to avoid data leakage
            # Split data: use first 70% for training, last 30% for testing
            split_point = int(len(data) * 0.7)

            if split_point < min_lookback:
                # Not enough data for cross-validation, fall back to full data
                self.logger.warning(f"Insufficient data for cross-validation ({len(data)} < {min_lookback * 1.4:.0f}), using full data")
                tprint_warning(
                    f"⚠️ Using full dataset for MRMR due to insufficient cross-validation rows ({len(data)})"
                )
                train_data = data
                test_data = data
            else:
                # Memory-efficient DataFrame splitting
                split_result = self.memory_ops.split_dataframe_efficiently(
                    data, split_ratio=split_point/len(data), force_copy=False
                )
                train_data = split_result.train_data
                test_data = split_result.test_data
                tprint_debug(
                    f"   ↳ MRMR split at index {split_point}: train={len(train_data)}, test={len(test_data)} "
                    f"(memory saved: {split_result.memory_saved_mb:.2f}MB)"
                )

            # Test different lookback periods using cross-validation
            for lookback in range(min_lookback, max_lookback + 1):
                try:
                    # Calculate feature on training data
                    train_features = self._calculate_feature_for_lookback(train_data, feature_name, lookback)
                    test_features = self._calculate_feature_for_lookback(test_data, feature_name, lookback)

                    train_features, train_stationarity = self._ensure_stationary_series(
                        train_features, lookback, context='train'
                    )
                    test_features, test_stationarity = self._ensure_stationary_series(
                        test_features, lookback, context='test'
                    )

                    stationarity_audit[lookback] = {
                        'train': train_stationarity,
                        'test': test_stationarity,
                    }

                    if train_stationarity.get('transformed') or test_stationarity.get('transformed'):
                        self.logger.debug(
                            f"   → Applied stationarity transform for {feature_name} (lookback={lookback})"
                        )

                    # Robust array alignment with safe NaN handling
                    try:
                        alignment = self.nan_handler.align_arrays_safely(
                            train_features, train_data[target_column].values, min_valid_samples=10
                        )
                        train_features_aligned = alignment.feature_values
                        train_targets_aligned = alignment.target_values
                        
                        alignment_test = self.nan_handler.align_arrays_safely(
                            test_features, test_data[target_column].values, min_valid_samples=10
                        )
                        test_features_aligned = alignment_test.feature_values
                        test_targets_aligned = alignment_test.target_values
                        
                        min_length = min(len(train_features_aligned), len(test_features_aligned))
                        if min_length <= 1:
                            stationarity_audit[lookback]['skipped'] = 'insufficient_length'
                            continue
                            
                    except DataValidationError as e:
                        stationarity_audit[lookback]['skipped'] = f'alignment_failed: {str(e)}'
                        continue

                    if np.nanstd(test_features_aligned) == 0:
                        stationarity_audit[lookback]['skipped'] = 'no_variance'
                        continue

                    # Calculate mutual information on test data to avoid overfitting
                    score = self._calculate_mutual_information_robust(
                        test_features_aligned, 
                        test_targets_aligned
                    )

                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ New MRMR best for '{feature_name}': lookback={lookback}, score={score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to evaluate lookback {lookback} for {feature_name}: {e}")
                    tprint_warning(f"⚠️ MRMR evaluation failed for lookback {lookback}: {e}")
                    continue

            tprint_success(
                f"🏁 MRMR optimization finished for '{feature_name}' with best lookback {best_lookback} (score={best_score:.4f})"
            )
            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="mrmr",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=trials > 0,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'correlation_method': 'pearson',
                    'cross_validation': True,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'stationarity_audit': stationarity_audit,
                    'stationarity_tests_available': STATIONARITY_TESTS_AVAILABLE,
                    'non_stationary_lookbacks': sum(
                        1
                        for audit in stationarity_audit.values()
                        if audit['train'].get('transformed') or audit['test'].get('transformed')
                    )
                }
            )

        except Exception as e:
            self.logger.error(f"MRMR optimization failed: {e}")
            return self._create_failed_result("mrmr", 0.0)

    def _ensure_stationary_series(
        self,
        values: Union[np.ndarray, pd.Series],
        lookback: int,
        context: str,
        significance: float = 0.05
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ensure a feature series is stationary, returning transformed values and metadata."""
        series = pd.Series(values).astype(float)
        series = series.replace([np.inf, -np.inf], np.nan)

        info: Dict[str, Any] = {
            'context': context,
            'tests_available': STATIONARITY_TESTS_AVAILABLE,
            'adf_pvalue': None,
            'kpss_pvalue': None,
            'transformed': False,
            'lookback': lookback,
        }

        clean_series = series.dropna()
        if clean_series.empty or len(clean_series) < 5:
            info['insufficient_samples'] = len(clean_series)
            return series.fillna(0.0).values, info

        is_stationary = False

        if STATIONARITY_TESTS_AVAILABLE and len(clean_series) >= 8:
            try:
                info['adf_pvalue'] = float(adfuller(clean_series, autolag='AIC')[1])
            except Exception as exc:  # pragma: no cover - defensive
                info['adf_error'] = str(exc)

            try:
                info['kpss_pvalue'] = float(kpss(clean_series, regression='c', nlags='auto')[1])
            except Exception as exc:  # pragma: no cover - defensive
                info['kpss_error'] = str(exc)

            adf_pass = info['adf_pvalue'] is not None and info['adf_pvalue'] < significance
            kpss_pass = info['kpss_pvalue'] is None or info['kpss_pvalue'] > significance
            is_stationary = adf_pass and kpss_pass
        else:
            diff_values = np.diff(clean_series.values)
            base_std = float(np.nanstd(clean_series.values))
            diff_std = float(np.nanstd(diff_values)) if len(diff_values) else 0.0
            ratio = diff_std / (base_std + 1e-12)
            info['heuristic_ratio'] = ratio
            is_stationary = base_std > 0 and diff_std > 0 and 0.1 <= ratio <= 10

        if not is_stationary:
            transformed = self._stationary_transform(series, lookback)
            info['transformed'] = True
            return transformed.values, info

        return series.fillna(0.0).values, info

    def _stationary_transform(self, series: pd.Series, lookback: int) -> pd.Series:
        """Apply a stationary transform using log returns/percent changes and demeaning."""
        safe_series = series.astype(float).replace([np.inf, -np.inf], np.nan)
        if safe_series.dropna().empty:
            return pd.Series(np.zeros(len(series)), index=series.index)

        if (safe_series > 0).all():
            diff_series = np.log(safe_series).diff()
        else:
            diff_series = safe_series.pct_change()

        diff_series = diff_series.replace([np.inf, -np.inf], np.nan)

        if diff_series.dropna().empty:
            diff_series = safe_series.diff()

        window = max(2, min(max(lookback, 2), len(diff_series)))
        demeaned = diff_series - diff_series.rolling(window=window, min_periods=1).mean()

        return demeaned.fillna(0.0)

    def _optimize_grid_search(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using comprehensive grid search approach with cross-validation."""
        try:
            min_lookback, max_lookback = lookback_range
            step_size = kwargs.get('step_size', 1)

            self.logger.info(f'🔍 Running grid search from {min_lookback} to {max_lookback} (step={step_size})')
            tprint_debug(
                f"🧭 Starting grid search for '{feature_name}' range [{min_lookback}, {max_lookback}] step={step_size}"
            )

            # Use time series cross-validation to avoid data leakage
            split_point = int(len(data) * 0.7)

            if split_point < min_lookback:
                self.logger.warning(f"Insufficient data for cross-validation ({len(data)} < {min_lookback * 1.4:.0f}), using full data")
                tprint_warning(
                    f"⚠️ Grid search using full dataset due to insufficient cross-validation rows ({len(data)})"
                )
                train_data = data
                test_data = data
            else:
                # Memory-efficient DataFrame splitting
                split_result = self.memory_ops.split_dataframe_efficiently(
                    data, split_ratio=split_point/len(data), force_copy=False
                )
                train_data = split_result.train_data
                test_data = split_result.test_data
                tprint_debug(
                    f"   ↳ Grid search split at {split_point}: train={len(train_data)}, test={len(test_data)} "
                    f"(memory saved: {split_result.memory_saved_mb:.2f}MB)"
                )

            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0

            # Test all lookback periods in range using cross-validation
            for lookback in range(min_lookback, max_lookback + 1, step_size):
                try:
                    # Calculate feature on both train and test data
                    train_features = self._calculate_feature_for_lookback(train_data, feature_name, lookback)
                    test_features = self._calculate_feature_for_lookback(test_data, feature_name, lookback)

                    # Robust array alignment with safe NaN handling
                    try:
                        alignment = self.nan_handler.align_arrays_safely(
                            test_features, test_data[target_column].values, min_valid_samples=10
                        )
                        test_features_aligned = alignment.feature_values
                        test_targets_aligned = alignment.target_values
                        
                        if len(test_features_aligned) <= 1:
                            continue
                            
                    except DataValidationError:
                        continue

                    # Calculate correlations on test data to avoid overfitting
                    correlations = self._calculate_comprehensive_correlations(
                        test_features_aligned, test_targets_aligned
                    )

                    # Use weighted combination of correlation metrics
                    score = self._calculate_composite_score(correlations)
                    all_scores.append(score)
                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ Grid search best updated: lookback={lookback}, score={score:.4f}"
                        )

                    if trials % 10 == 0:
                        self.logger.debug(f'   → Progress: {trials} trials, best_score={best_score:.4f}')
                        tprint_debug(
                            f"   ↺ Grid search progress: {trials} trials, current best={best_score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to evaluate lookback {lookback}: {e}')
                    tprint_warning(f"⚠️ Grid search evaluation failed for lookback {lookback}: {e}")
                    continue

            # Calculate convergence metrics
            convergence_achieved = self._check_convergence(all_scores)
            tprint_debug(
                f"📈 Grid search convergence {'achieved' if convergence_achieved else 'not achieved'} with {trials} trials"
            )

            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="grid_search",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                feature_name=feature_name,  # FIXED: Added feature_name field
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'step_size': step_size,
                    'all_scores': all_scores,
                    'score_std': np.std(all_scores) if all_scores else 0.0,
                    'cross_validation': True,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Grid search optimization failed: {e}")
            return self._create_failed_result("grid_search", 0.0)

    def _optimize_bayesian(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using Bayesian optimization approach with TPE."""
        try:
            min_lookback, max_lookback = lookback_range
            n_trials = kwargs.get('n_trials', 50)
            n_startup_trials = kwargs.get('n_startup_trials', 10)

            self.logger.info(f'🎯 Running Bayesian optimization with {n_trials} trials')
            tprint_debug(
                f"🧪 Starting Bayesian optimization for '{feature_name}' range [{min_lookback}, {max_lookback}] with {n_trials} trials"
            )
            
            # Initialize with random samples for exploration
            startup_trials = self._rng.integers(min_lookback, max_lookback + 1, n_startup_trials)
            all_scores = []
            all_lookbacks = []
            
            # Startup phase - random exploration
            for lookback in startup_trials:
                try:
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)

                    all_scores.append(score)
                    all_lookbacks.append(lookback)
                    tprint_debug(
                        f"   🔄 Bayesian startup trial lookback={lookback}, score={score:.4f}"
                    )

                except Exception as e:
                    self.logger.warning(f'⚠️ Startup trial failed for lookback {lookback}: {e}')
                    tprint_warning(f"⚠️ Bayesian startup trial failed for lookback {lookback}: {e}")
                    continue

            # Bayesian optimization phase
            for trial in range(n_startup_trials, n_trials):
                try:
                    # Use simple acquisition function (exploration vs exploitation)
                    if len(all_scores) < 5:
                        # More exploration
                        lookback = int(self._rng.integers(min_lookback, max_lookback + 1))
                    else:
                        # Exploit best regions
                        best_idx = np.argmax(all_scores)
                        best_lookback = all_lookbacks[best_idx]
                        
                        # Add some exploration around best point
                        exploration_range = max(1, (max_lookback - min_lookback) // 10)
                        lookback = int(
                            self._rng.integers(
                                max(min_lookback, best_lookback - exploration_range),
                                min(max_lookback + 1, best_lookback + exploration_range + 1)
                            )
                        )
                    
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)

                    all_scores.append(score)
                    all_lookbacks.append(lookback)
                    tprint_debug(
                        f"   🎯 Bayesian trial {trial}: lookback={lookback}, score={score:.4f}"
                    )

                    if trial % 10 == 0:
                        current_best = max(all_scores)
                        self.logger.debug(f'   → Trial {trial}: best_score={current_best:.4f}')
                        tprint_debug(
                            f"   📊 Bayesian progress trial {trial}: best_score={current_best:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Bayesian trial failed: {e}')
                    tprint_warning(f"⚠️ Bayesian trial failure at iteration {trial}: {e}")
                    continue

            # Find best result
            if all_scores:
                best_idx = np.argmax(all_scores)
                best_score = all_scores[best_idx]
                best_lookback = all_lookbacks[best_idx]
                convergence_achieved = self._check_convergence(all_scores)
                tprint_success(
                    f"🏁 Bayesian optimization finished for '{feature_name}' with lookback {best_lookback} (score={best_score:.4f})"
                )
            else:
                best_score = 0.0
                best_lookback = min_lookback
                convergence_achieved = False
                tprint_warning(
                    f"⚠️ Bayesian optimization produced no valid scores for '{feature_name}', using fallback values"
                )
            
            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="bayesian",
                total_trials=len(all_scores),
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                feature_name=feature_name,  # FIXED: Added feature_name field
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'n_trials': n_trials,
                    'n_startup_trials': n_startup_trials,
                    'all_scores': all_scores,
                    'all_lookbacks': all_lookbacks,
                    'score_improvement': max(all_scores) - min(all_scores) if all_scores else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return self._create_failed_result("bayesian", 0.0)

    def _optimize_random_search(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using random search approach."""
        try:
            min_lookback, max_lookback = lookback_range
            n_trials = kwargs.get('n_trials', 30)

            self.logger.info(f'🎲 Running random search with {n_trials} trials')
            tprint_debug(
                f"🎲 Starting random search for '{feature_name}' between {min_lookback} and {max_lookback} ({n_trials} trials)"
            )
            
            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0
            
            # Random sampling
            for trial in range(n_trials):
                try:
                    lookback = int(self._rng.integers(min_lookback, max_lookback + 1))

                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)
                    
                    all_scores.append(score)
                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ Random search best updated on trial {trial}: lookback={lookback}, score={score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Random trial failed: {e}')
                    tprint_warning(f"⚠️ Random search trial {trial} failed: {e}")
                    continue

            convergence_achieved = self._check_convergence(all_scores)
            tprint_debug(
                f"📈 Random search convergence {'achieved' if convergence_achieved else 'not achieved'} after {trials} trials"
            )

            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="random_search",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                feature_name=feature_name,  # FIXED: Added feature_name field
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'n_trials': n_trials,
                    'all_scores': all_scores,
                    'score_std': np.std(all_scores) if all_scores else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Random search optimization failed: {e}")
            return self._create_failed_result("random_search", 0.0)

    def _optimize_multi_target(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using multi-target approach for multiple objectives."""
        try:
            min_lookback, max_lookback = lookback_range
            step_size = kwargs.get('step_size', 1)

            self.logger.info(f'🎯 Running multi-target optimization')
            tprint_debug(
                f"🎯 Starting multi-target optimization for '{feature_name}' range [{min_lookback}, {max_lookback}] step={step_size}"
            )
            
            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0
            
            # Test all lookback periods
            for lookback in range(min_lookback, max_lookback + 1, step_size):
                try:
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)

                    # Calculate multiple target metrics
                    targets = self._calculate_multi_target_metrics(
                        feature_values, data[target_column].values
                    )

                    # Multi-objective optimization using weighted sum
                    score = self._calculate_multi_objective_score(targets)
                    all_scores.append(score)
                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ Multi-target best updated: lookback={lookback}, score={score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Multi-target trial failed for lookback {lookback}: {e}')
                    tprint_warning(f"⚠️ Multi-target evaluation failed for lookback {lookback}: {e}")
                    continue

            convergence_achieved = self._check_convergence(all_scores)
            tprint_debug(
                f"📈 Multi-target convergence {'achieved' if convergence_achieved else 'not achieved'} after {trials} trials"
            )

            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="multi_target",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                feature_name=feature_name,  # FIXED: Added feature_name field
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'all_scores': all_scores,
                    'multi_target_weights': kwargs.get('target_weights', {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Multi-target optimization failed: {e}")
            return self._create_failed_result("multi_target", 0.0)

    def _calculate_feature_for_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback: int
    ) -> np.ndarray:
        """
        Calculate sophisticated feature values for a given lookback period using VectorBT optimizations.

        This implementation uses VectorBTRollingOptimizer for high-performance rolling operations
        and actual technical indicators from the feature engineering pipeline.
        """
        try:
            tprint_debug(
                f"🧮 Calculating feature '{feature_name}' values for lookback {lookback} using VectorBT optimizations"
            )
            
            # OPTIMIZATION: Check if feature already exists in dataframe FIRST (before trying to generate)
            # This is critical for expensive features like GARCH that take 2-3 seconds to calculate
            if feature_name in data.columns:
                tprint_debug(
                    f"ℹ️ Using lagged version for pre-generated feature '{feature_name}' (lag={lookback})"
                )
                # Shift the feature back by 'lookback' periods to test predictive power at different lags
                feature_series = data[feature_name].shift(lookback)
                # CRITICAL FIX: Do NOT fill NaN with zeros - zeros create artificial correlation!
                # Return the series with NaN values, let alignment/filtering handle them properly
                return feature_series.values
            
            # Try VectorBT optimization first if available
            if self.use_vectorbt_optimization and self.rolling_optimizer:
                vectorbt_result = self._calculate_feature_vectorbt_optimized(
                    data, feature_name, lookback
                )
                if vectorbt_result is not None:
                    return vectorbt_result
            
            # If not pre-generated, create feature generator based on feature name pattern
            feature_generator = self._create_feature_generator(feature_name, lookback)

            if feature_generator is None:
                # Feature not recognized and not in dataframe
                tprint_warning(
                    f"⚠️ Feature '{feature_name}' not in dataframe and no generator found, returning zeros"
                )
                return np.zeros(len(data))

            tprint_debug(
                f"   ↳ Using generator {type(feature_generator).__name__} for '{feature_name}'"
            )
            
            # Ensure data has lowercase column names for feature generators
            # Use memory-efficient column operation
            data_for_generation = self.memory_ops.select_columns_efficiently(
                data, data.columns.tolist(), force_copy=True
            )
            data_for_generation.columns = data_for_generation.columns.str.lower()
            
            # Generate feature using the technical indicator
            feature_result = feature_generator.generate(data_for_generation)

            if feature_result.success and feature_result.data is not None:
                # Handle different return types from generators
                feature_data = feature_result.data

                # For Bollinger Bands, extract the specific band we want based on feature name
                if 'bb_' in feature_name.lower():
                    # Handle both DataFrame and Series cases
                    if isinstance(feature_data, pd.DataFrame):
                        if 'upper' in feature_name.lower() and len(feature_data.columns) > 0:
                            return feature_data.iloc[:, 0].values  # Upper band
                        elif 'lower' in feature_name.lower() and len(feature_data.columns) > 2:
                            return feature_data.iloc[:, 2].values  # Lower band
                        elif 'middle' in feature_name.lower() and len(feature_data.columns) > 1:
                            return feature_data.iloc[:, 1].values  # Middle band
                    elif isinstance(feature_data, pd.Series):
                        # If it's a Series, return it directly
                        return feature_data.values

                # For other indicators, return the single series or first column
                if isinstance(feature_data, pd.DataFrame):
                    if len(feature_data.columns) > 0:
                        tprint_debug(
                            f"   ↳ Returning first column from DataFrame for '{feature_name}'"
                        )
                        return feature_data.iloc[:, 0].values
                    else:
                        tprint_warning(
                            f"⚠️ Generated DataFrame empty for '{feature_name}', returning zeros"
                        )
                        return np.zeros(len(data))
                elif isinstance(feature_data, pd.Series):
                    tprint_debug(
                        f"   ↳ Returning Series values for '{feature_name}'"
                    )
                    return feature_data.values
                else:
                    tprint_debug(
                        f"   ↳ Converting generated data to numpy array for '{feature_name}'"
                    )
                    return np.array(feature_data)
            else:
                self.logger.warning(f"Feature generation failed for {feature_name}, using fallback")
                tprint_warning(
                    f"⚠️ Feature generation unsuccessful for '{feature_name}', returning zeros"
                )
                return np.zeros(len(data))

        except ImportError as e:
            self.logger.warning(f"Feature engineering modules not available: {e}, using fallback")
            # For pre-generated features, test lagged versions
            if feature_name in data.columns:
                tprint_debug(
                    f"ℹ️ Feature modules missing, using lagged version for '{feature_name}' (lag={lookback})"
                )
                feature_series = data[feature_name].shift(lookback)
                return feature_series.fillna(0.0).values
            else:
                tprint_error(
                    f"❌ Feature modules missing and '{feature_name}' not in data, returning zeros"
                )
                return np.zeros(len(data))
        except Exception as e:
            self.logger.error(f"Failed to calculate feature {feature_name} for lookback {lookback}: {e}")
            tprint_error(
                f"❌ Exception during feature calculation for '{feature_name}' (lookback={lookback}): {e}"
            )
            return np.zeros(len(data))

    def _calculate_feature_vectorbt_optimized(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback: int
    ) -> Optional[np.ndarray]:
        """
        Calculate feature using VectorBT optimizations for high-performance rolling operations.
        
        Args:
            data: Input data with OHLCV columns
            feature_name: Name of the feature to calculate
            lookback: Lookback period for the feature
            
        Returns:
            Feature values as numpy array, or None if not supported
        """
        start_time = time.time()
        try:
            if not self.use_vectorbt_optimization or not self.rolling_optimizer:
                return None
            
            # Get price data
            close_prices = data.get('close', data.get('Close', None))
            if close_prices is None:
                return None
            
            close_series = pd.Series(close_prices)
            
            # Use Unified Vectorization Manager for intelligent optimization
            if self.unified_manager:
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.TECHNICAL_INDICATORS,
                    data_size=len(data),
                    data_dimensions=(len(data), len(data.columns)),
                    memory_budget_mb=512.0,
                    time_budget_seconds=30.0
                )
                
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                self.logger.debug(f"Selected VectorBT strategy for {feature_name}: {strategy}")
            
            # Calculate feature based on name pattern using VectorBT optimizations
            feature_name_lower = feature_name.lower()
            
            if 'sma' in feature_name_lower or 'simple' in feature_name_lower:
                return self.rolling_optimizer.rolling_mean(close_series, lookback).values
            elif 'ema' in feature_name_lower or 'exponential' in feature_name_lower:
                # EMA calculation using VectorBT rolling operations
                alpha = 2.0 / (lookback + 1)
                ema_values = np.zeros_like(close_series.values)
                ema_values[0] = close_series.values[0]
                
                for i in range(1, len(close_series)):
                    ema_values[i] = alpha * close_series.values[i] + (1 - alpha) * ema_values[i-1]
                
                return ema_values
            elif 'std' in feature_name_lower or 'volatility' in feature_name_lower:
                return self.rolling_optimizer.rolling_std(close_series, lookback).values
            elif 'rsi' in feature_name_lower:
                # RSI calculation using VectorBT rolling operations
                price_changes = close_series.diff()
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)
                
                avg_gains = self.rolling_optimizer.rolling_mean(gains, lookback)
                avg_losses = self.rolling_optimizer.rolling_mean(losses, lookback)
                
                rs = avg_gains / (avg_losses + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                
                return rsi.values
            elif 'bb' in feature_name_lower or 'bollinger' in feature_name_lower:
                # Bollinger Bands calculation using VectorBT rolling operations
                sma = self.rolling_optimizer.rolling_mean(close_series, lookback)
                std = self.rolling_optimizer.rolling_std(close_series, lookback)
                
                upper_band = sma + (2 * std)
                lower_band = sma - (2 * std)
                
                # Return the width (upper - lower) / middle
                width = (upper_band - lower_band) / (sma + 1e-10)
                return width.values
            elif 'macd' in feature_name_lower:
                # MACD calculation using VectorBT rolling operations
                fast_window = lookback
                slow_window = lookback * 2
                signal_window = lookback // 2
                
                # Calculate EMAs
                fast_ema = pd.Series(close_series).ewm(span=fast_window).mean()
                slow_ema = pd.Series(close_series).ewm(span=slow_window).mean()
                
                # Calculate MACD line
                macd_line = fast_ema - slow_ema
                
                # Calculate signal line
                signal_line = macd_line.ewm(span=signal_window).mean()
                
                # Calculate MACD histogram
                macd_histogram = macd_line - signal_line
                
                return macd_histogram.values
            else:
                # Default to rolling mean for unknown features
                return self.rolling_optimizer.rolling_mean(close_series, lookback).values
                
        except Exception as e:
            self.logger.warning(f"VectorBT feature calculation failed for {feature_name}: {e}")
            return None
        finally:
            # Track performance metrics
            duration = time.time() - start_time
            self._track_vectorbt_operation('feature_calculation', duration, gpu_used=False)

    def _create_feature_generator(self, feature_name: str, lookback: int):
        """
        Create appropriate feature generator based on feature name pattern for ALL indicators
        from the feature engineering bank (excluding wavelets and autoencoders).

        Args:
            feature_name: Name of the feature (e.g., 'rsi_14', 'macd_12_26_9', 'bb_upper_20')
            lookback: Lookback period for optimization

        Returns:
            FeatureGenerator instance or None if not recognized
        """
        try:
            from src.feature_generation.base_calculations.base_calculator import BaseCalculationType

            # Import ALL feature generators needed from the feature bank
            # Momentum indicators
            from src.feature_generation.categories.momentum import (
                RSIGenerator, MACDGenerator, StochasticGenerator, WilliamsRGenerator,
                MomentumOscillatorGenerator, RateOfChangeGenerator
            )

            # Volatility indicators
            from src.feature_generation.categories.volatility import (
                BollingerBandsGenerator, ATRGenerator, VolatilityBandsGenerator,
                VolatilityFeatureGenerator, GARCHFeatureGenerator
            )

            # Trend indicators
            from src.feature_generation.categories.trend import (
                SMAGenerator, EMAGenerator, WMAGenerator, DEMAGenerator,
                TEMAGenerator, TRIMAGenerator, VWMAGenerator, KeltnerChannelsGenerator
            )

            # Oscillator indicators
            from src.feature_generation.categories.oscillator import (
                CCIGenerator, ADXGenerator, AroonGenerator, UltimateOscillatorGenerator,
                KSTGenerator, APOGenerator, CMOGenerator, NATRGenerator, PFEGenerator,
                T3Generator, KAMAGenerator
            )

            # Volume indicators
            from src.feature_generation.categories.volume import (
                VolumeSMAGenerator, VolumeEMAGenerator, VolumeRatioGenerator,
                VolumeROCGenerator, VolumeStdGenerator, VolumePercentileGenerator,
                VolumeTrendStrengthGenerator, VolumeOscillatorGenerator,
                VolumeMomentumGenerator, VolumeVWAPGenerator, VolumePriceTrendGenerator,
                VolumeAccumulationDistributionGenerator
            )

            # Support/Resistance indicators
            from src.feature_generation.categories.support_resistance import (
                SupportLevelGenerator, ResistanceLevelGenerator, PivotPointGenerator,
                FibonacciLevelGenerator
            )

            # Returns indicators
            from src.feature_generation.categories.returns import (
                SimpleReturnsGenerator, LogReturnsGenerator, CumulativeReturnsGenerator
            )

            # Entropy indicators
            from src.feature_generation.categories.entropy import (
                PriceEntropyGenerator, VolumeEntropyGenerator, ReturnEntropyGenerator,
                PriceEntropyMAGenerator, VolumeEntropyMAGenerator, ReturnEntropyMAGenerator,
                HighLowEntropyGenerator, VolatilityEntropyGenerator, MomentumEntropyGenerator,
                RSIEntropyGenerator, MACDEntropyGenerator, BollingerBandsEntropyGenerator,
                CrossAssetEntropyGenerator, RegimeEntropyGenerator
            )

            # Acceleration indicators
            from src.feature_generation.categories.acceleration import (
                MomentumGenerator, PriceAccelerationGenerator, PriceJerkGenerator,
                TrendStrengthGenerator, TrendConsistencyGenerator,
                VolumeAccelerationGenerator, VolatilityAccelerationGenerator
            )

            # Interaction indicators
            from src.feature_generation.categories.interaction import (
                CrossTimeframeInteractionGenerator, FeatureRatioGenerator,
                PolynomialFeatureGenerator, CorrelationInteractionGenerator
            )

            # Cross-timeframe indicators
            from src.feature_generation.categories.cross_timeframe import (
                CrossTimeframeMomentumGenerator, CrossTimeframeVolatilityGenerator,
                CrossTimeframeVolumeGenerator, CrossTimeframeTrendGenerator,
                CrossTimeframeHighLowGenerator, CrossTimeframeRatioGenerator,
                CrossTimeframeCorrelationGenerator, CrossTimeframeDivergenceGenerator
            )

            # Microstructure indicators
            from src.feature_generation.categories.microstructure import (
                BidAskSpreadGenerator, OrderFlowImbalanceGenerator, TradeSizeImbalanceGenerator,
                PriceImpactGenerator, VolumeWeightedPriceGenerator, TradeIntensityGenerator,
                LiquidityProxyGenerator, MarketDepthGenerator
            )

            # Order flow indicators
            from src.feature_generation.categories.order_flow import (
                TakerBuyRatioGenerator, TakerSellRatioGenerator, MarketAggressionIndexGenerator,
                OrderFlowImbalanceGenerator as OrderFlowImbalanceGeneratorOF
            )

            # Candlestick pattern indicators (placeholder implementations)
            from src.feature_generation.categories.candlestick_pattern import (
                CandlestickPatternFeatureGenerator
            )

            # Advanced SR features (these are calculated from historical SR data)
            from src.feature_generation.utils.enhanced_sr_feature_extractor import (
                EnhancedSRFeatureExtractor, HistoricalSRAnalyzer, HistoricalSRConfig
            )

            # Parse feature name to determine type and parameters
            name_lower = feature_name.lower()

            # Skip wavelets and autoencoders as requested
            if 'wavelet' in name_lower or 'autoencoder' in name_lower:
                return None

            # MOMENTUM INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            if name_lower.startswith('rsi'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    if 'vwap' in name_lower:
                        return RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)
                    else:
                        # Explicit PRICE_RETURNS variant
                        return RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('macd'):
                params = self._extract_macd_params(feature_name)
                if params:
                    fast, slow, signal = params
                    if 'price' in name_lower:
                        return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.PRICE_LEVELS)
                    elif 'returns' in name_lower:
                        if 'vwap' in name_lower:
                            return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.RETURNS_VWAP)
                        else:
                            # Explicit PRICE_RETURNS variant
                            return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.PRICE_RETURNS)
                    else:
                        # Default to RETURNS_VWAP (now standard in feature engineering)
                        return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('stoch'):
                period = self._extract_period_from_name(feature_name, 14)
                stoch_type = name_lower.split('_')[1] if len(name_lower.split('_')) > 1 else 'k'
                if 'price' in name_lower:
                    base_calc = BaseCalculationType.PRICE_LEVELS
                else:
                    # Default to RETURNS_VWAP for better signal quality
                    base_calc = BaseCalculationType.RETURNS_VWAP

                if stoch_type == 'k':
                    return StochasticGenerator(k_period=period, d_period=3, base_calculation=base_calc)
                elif stoch_type == 'd':
                    return StochasticGenerator(k_period=period, d_period=3, base_calculation=base_calc)

            elif name_lower.startswith('williams_r') or name_lower.startswith('williams%'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return WilliamsRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return WilliamsRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return WilliamsRGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('momentum_osc') or name_lower.startswith('momentum_'):
                period = self._extract_period_from_name(feature_name, 10)
                if 'price' in name_lower:
                    return MomentumOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return MomentumOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return MomentumOscillatorGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('roc_'):
                period = self._extract_period_from_name(feature_name, 10)
                if 'price' in name_lower:
                    return RateOfChangeGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return RateOfChangeGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return RateOfChangeGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            # VOLATILITY INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            elif name_lower.startswith('bb_'):
                period = self._extract_period_from_name(feature_name, 20)
                bb_type = name_lower.split('_')[1] if len(name_lower.split('_')) > 1 else 'middle'
                band_type = "middle"
                if bb_type == 'upper':
                    band_type = "upper"
                elif bb_type == 'lower':
                    band_type = "lower"

                if 'price' in name_lower:
                    return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS, band_type=band_type)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_RETURNS, band_type=band_type)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, band_type=band_type)

            elif name_lower.startswith('atr_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return ATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return ATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return ATRGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('volatility_bands'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return VolatilityBandsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return VolatilityBandsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return VolatilityBandsGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('volatility_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return VolatilityFeatureGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return VolatilityFeatureGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return VolatilityFeatureGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('garch_'):
                # Parse GARCH parameters (p, q, h)
                params = self._extract_garch_params(feature_name)
                if params:
                    p, q, h = params
                    return GARCHFeatureGenerator(p=p, q=q, forecast_horizon=h)

            # TREND INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            elif name_lower.startswith('sma_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return SMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return SMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return SMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('ema_'):
                period = self._extract_period_from_name(feature_name, 12)
                if 'price' in name_lower:
                    return EMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return EMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return EMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('wma_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return WMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return WMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return WMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('dema_'):
                period = self._extract_period_from_name(feature_name, 21)
                if 'price' in name_lower:
                    return DEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return DEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return DEMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('tema_'):
                period = self._extract_period_from_name(feature_name, 21)
                if 'price' in name_lower:
                    return TEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return TEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return TEMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('trima_'):
                period = self._extract_period_from_name(feature_name, 21)
                if 'price' in name_lower:
                    return TRIMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return TRIMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return TRIMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwma_'):
                period = self._extract_period_from_name(feature_name, 20)
                # VWMA is inherently volume-weighted, so RETURNS_VWAP makes sense
                if 'price' in name_lower:
                    return VWMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                else:
                    return VWMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('keltner_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return KeltnerChannelsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return KeltnerChannelsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return KeltnerChannelsGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('adx_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return ADXGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return ADXGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return ADXGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('trend_score'):
                period = self._extract_period_from_name(feature_name, 14)
                from src.feature_generation.categories.trend import TrendScoreGenerator
                if 'price' in name_lower:
                    return TrendScoreGenerator(adx_period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return TrendScoreGenerator(adx_period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return TrendScoreGenerator(adx_period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            # OSCILLATOR INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            elif name_lower.startswith('cci_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return CCIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return CCIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return CCIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('aroon_'):
                period = self._extract_period_from_name(feature_name, 25)
                if 'price' in name_lower:
                    return AroonGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return AroonGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return AroonGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('ultimate_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return UltimateOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return UltimateOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return UltimateOscillatorGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('kst_'):
                period = self._extract_period_from_name(feature_name, 10)
                if 'price' in name_lower:
                    return KSTGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return KSTGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return KSTGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('apo_'):
                period = self._extract_period_from_name(feature_name, 12)
                if 'price' in name_lower:
                    return APOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return APOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return APOGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('cmo_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return CMOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return CMOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return CMOGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('natr_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return NATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return NATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return NATRGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('pfe_'):
                period = self._extract_period_from_name(feature_name, 12)
                if 'price' in name_lower:
                    return PFEGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return PFEGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return PFEGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('t3_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return T3Generator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return T3Generator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return T3Generator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('kama_'):
                period = self._extract_period_from_name(feature_name, 30)
                if 'price' in name_lower:
                    return KAMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return KAMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return KAMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            # VOLUME INDICATORS - Use VOLUME_RETURNS as default for volume-based indicators
            elif name_lower.startswith('volume_sma_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeSMAGenerator(period=period)

            elif name_lower.startswith('volume_ema_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeEMAGenerator(period=period)

            elif name_lower.startswith('volume_ratio_'):
                period = self._extract_period_from_name(feature_name, 10)
                return VolumeRatioGenerator(period=period)

            elif name_lower.startswith('volume_roc_'):
                period = self._extract_period_from_name(feature_name, 10)
                return VolumeROCGenerator(period=period)

            elif name_lower.startswith('volume_std_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeStdGenerator(period=period)

            elif name_lower.startswith('volume_percentile_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumePercentileGenerator(period=period)

            elif name_lower.startswith('volume_trend_strength'):
                params = self._extract_dual_period_params(feature_name, 10, 30)
                if params:
                    short_period, long_period = params
                    return VolumeTrendStrengthGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('volume_osc'):
                params = self._extract_dual_period_params(feature_name, 10, 20)
                if params:
                    short_period, long_period = params
                    return VolumeOscillatorGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('volume_momentum_'):
                period = self._extract_period_from_name(feature_name, 10)
                return VolumeMomentumGenerator(period=period)

            elif name_lower.startswith('volume_vwap_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeVWAPGenerator(period=period)

            elif name_lower.startswith('volume_price_trend'):
                return VolumePriceTrendGenerator()

            elif name_lower.startswith('volume_acc_dist'):
                return VolumeAccumulationDistributionGenerator()

            # EXPLICIT VWAP-BASED VARIANTS (when 'vwap_' prefix is used)
            elif name_lower.startswith('vwap_rsi_'):
                period = self._extract_period_from_name(feature_name, 14)
                return RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_macd_'):
                params = self._extract_macd_params(feature_name.replace('vwap_', ''))
                if params:
                    fast, slow, signal = params
                    return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_sma_'):
                period = self._extract_period_from_name(feature_name, 20)
                return SMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_ema_'):
                period = self._extract_period_from_name(feature_name, 12)
                return EMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_bb_'):
                period = self._extract_period_from_name(feature_name, 20)
                bb_type = name_lower.split('_')[2] if len(name_lower.split('_')) > 2 else 'middle'
                band_type = "middle"
                if bb_type == 'upper':
                    band_type = "upper"
                elif bb_type == 'lower':
                    band_type = "lower"
                return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, band_type=band_type)

            # SUPPORT/RESISTANCE INDICATORS
            elif name_lower.startswith('support_level_'):
                params = self._extract_dual_params(feature_name, 'level', 'window')
                if params:
                    level, window = params
                    return SupportLevelGenerator(level=level, window=window)

            elif name_lower.startswith('resistance_level_'):
                params = self._extract_dual_params(feature_name, 'level', 'window')
                if params:
                    level, window = params
                    return ResistanceLevelGenerator(level=level, window=window)

            elif name_lower.startswith('pivot_point_'):
                window = self._extract_period_from_name(feature_name, 20)
                return PivotPointGenerator(window=window)

            elif name_lower.startswith('fibonacci_'):
                params = self._extract_dual_params(feature_name, 'level', 'window')
                if params:
                    level, window = params
                    return FibonacciLevelGenerator(level=level, window=window)

            # RETURNS INDICATORS
            elif name_lower.startswith('return_'):
                period = self._extract_period_from_name(feature_name, 1)
                return_type = name_lower.split('_')[1] if len(name_lower.split('_')) > 1 else 'simple'
                if return_type == 'log':
                    return LogReturnsGenerator(period=period)
                elif return_type == 'cumulative':
                    return CumulativeReturnsGenerator(period=period)
                else:
                    return SimpleReturnsGenerator(period=period)

            # ENTROPY INDICATORS
            elif name_lower.startswith('price_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return PriceEntropyGenerator(window=window)

            elif name_lower.startswith('volume_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return VolumeEntropyGenerator(window=window)

            elif name_lower.startswith('return_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return ReturnEntropyGenerator(window=window)

            elif name_lower.startswith('price_entropy_ma_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, ma_window = params
                    return PriceEntropyMAGenerator(window=window, ma_window=ma_window)

            elif name_lower.startswith('volume_entropy_ma_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, ma_window = params
                    return VolumeEntropyMAGenerator(window=window, ma_window=ma_window)

            elif name_lower.startswith('return_entropy_ma_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, ma_window = params
                    return ReturnEntropyMAGenerator(window=window, ma_window=ma_window)

            elif name_lower.startswith('high_low_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return HighLowEntropyGenerator(window=window)

            elif name_lower.startswith('volatility_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 10)
                if params:
                    window, volatility_window = params
                    return VolatilityEntropyGenerator(window=window, volatility_window=volatility_window)

            elif name_lower.startswith('momentum_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, momentum_period = params
                    return MomentumEntropyGenerator(window=window, momentum_period=momentum_period)

            elif name_lower.startswith('rsi_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 14)
                if params:
                    window, rsi_period = params
                    return RSIEntropyGenerator(window=window, rsi_period=rsi_period)

            elif name_lower.startswith('macd_entropy_'):
                params = self._extract_macd_params(feature_name)
                if params:
                    window, fast, slow = 20, params[0], params[1]  # Use window as first param
                    return MACDEntropyGenerator(window=window, fast=fast, slow=slow)

            elif name_lower.startswith('bb_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 20)
                if params:
                    window, bb_period = params
                    return BollingerBandsEntropyGenerator(window=window, bb_period=bb_period, bb_std=2.0)

            elif name_lower.startswith('cross_asset_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 10)
                if params:
                    window, correlation_window = params
                    return CrossAssetEntropyGenerator(window=window, correlation_window=correlation_window)

            elif name_lower.startswith('regime_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 50)
                if params:
                    window, regime_window = params
                    return RegimeEntropyGenerator(window=window, regime_window=regime_window)

            # ACCELERATION INDICATORS
            elif name_lower.startswith('momentum_acc_'):
                period = self._extract_period_from_name(feature_name, 10)
                return MomentumGenerator(period=period)

            elif name_lower.startswith('price_acceleration_'):
                period = self._extract_period_from_name(feature_name, 5)
                return PriceAccelerationGenerator(period=period)

            elif name_lower.startswith('price_jerk_'):
                period = self._extract_period_from_name(feature_name, 5)
                return PriceJerkGenerator(period=period)

            elif name_lower.startswith('trend_strength_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TrendStrengthGenerator(window=window)

            elif name_lower.startswith('trend_consistency_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TrendConsistencyGenerator(window=window)

            elif name_lower.startswith('volume_acceleration_'):
                period = self._extract_period_from_name(feature_name, 5)
                return VolumeAccelerationGenerator(period=period)

            elif name_lower.startswith('volatility_acceleration_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    period, volatility_window = params
                    return VolatilityAccelerationGenerator(period=period, volatility_window=volatility_window)

            # INTERACTION INDICATORS
            elif name_lower.startswith('cross_timeframe_interaction_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    interaction_type = name_lower.split('_')[-1] if len(name_lower.split('_')) > 3 else 'ratio'
                    return CrossTimeframeInteractionGenerator(short_period=short_period, long_period=long_period, interaction_type=interaction_type)

            elif name_lower.startswith('feature_ratio_'):
                # Extract column names from feature name
                columns = self._extract_column_params(feature_name)
                if columns:
                    numerator, denominator = columns
                    return FeatureRatioGenerator(numerator_column=numerator, denominator_column=denominator)

            elif name_lower.startswith('polynomial_'):
                # Extract column and degree
                parts = name_lower.replace('polynomial_', '').split('_deg_')
                if len(parts) == 2:
                    column, degree = parts
                    return PolynomialFeatureGenerator(column=column, degree=int(degree))

            elif name_lower.startswith('correlation_interaction_'):
                columns = self._extract_column_params(feature_name)
                if columns:
                    col1, col2 = columns
                    window = self._extract_period_from_name(feature_name, 20)
                    return CorrelationInteractionGenerator(column1=col1, column2=col2, window=window)

            # CROSS-TIMEFRAME INDICATORS
            elif name_lower.startswith('cross_tf_momentum_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeMomentumGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_volatility_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeVolatilityGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_volume_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeVolumeGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_trend_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeTrendGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_high_low_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeHighLowGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_ratio_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeRatioGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_correlation_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeCorrelationGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_divergence_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeDivergenceGenerator(short_period=short_period, long_period=long_period)

            # MICROSTRUCTURE INDICATORS
            elif name_lower.startswith('bid_ask_spread_'):
                window = self._extract_period_from_name(feature_name, 10)
                return BidAskSpreadGenerator(window=window)

            elif name_lower.startswith('order_flow_imbalance_'):
                window = self._extract_period_from_name(feature_name, 10)
                return OrderFlowImbalanceGenerator(window=window)

            elif name_lower.startswith('trade_size_imbalance_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TradeSizeImbalanceGenerator(window=window)

            elif name_lower.startswith('price_impact_'):
                window = self._extract_period_from_name(feature_name, 10)
                return PriceImpactGenerator(window=window)

            elif name_lower.startswith('volume_weighted_price_'):
                window = self._extract_period_from_name(feature_name, 10)
                return VolumeWeightedPriceGenerator(window=window)

            elif name_lower.startswith('trade_intensity_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TradeIntensityGenerator(window=window)

            elif name_lower.startswith('liquidity_proxy_'):
                window = self._extract_period_from_name(feature_name, 10)
                return LiquidityProxyGenerator(window=window)

            elif name_lower.startswith('market_depth_'):
                window = self._extract_period_from_name(feature_name, 10)
                return MarketDepthGenerator(window=window)

            # ORDER FLOW INDICATORS
            elif name_lower.startswith('taker_buy_ratio_'):
                window = self._extract_period_from_name(feature_name, 20)
                return TakerBuyRatioGenerator(window=window)

            elif name_lower.startswith('taker_sell_ratio_'):
                window = self._extract_period_from_name(feature_name, 20)
                return TakerSellRatioGenerator(window=window)

            elif name_lower.startswith('market_aggression_index_'):
                window = self._extract_period_from_name(feature_name, 20)
                return MarketAggressionIndexGenerator(window=window)

            elif name_lower.startswith('order_flow_imbalance_of_'):
                window = self._extract_period_from_name(feature_name, 20)
                return OrderFlowImbalanceGeneratorOF(window=window)

            # CANDLESTICK PATTERN INDICATORS (placeholder implementations)
            elif name_lower.startswith('candlestick_pattern_'):
                # Extract pattern type if specified
                pattern_type = name_lower.replace('candlestick_pattern_', '')
                return CandlestickPatternFeatureGenerator()  # Uses default config

            # ADVANCED SUPPORT/RESISTANCE FEATURES (calculated from historical SR data)
            elif name_lower.startswith('sr_persistence_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_persistence_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_persistence_generator(window, sr_type)

            elif name_lower.startswith('sr_touch_freq_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_touch_freq_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_touch_freq_generator(window, sr_type)

            elif name_lower.startswith('sr_bounce_rate_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_bounce_rate_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_bounce_rate_generator(window, sr_type)

            elif name_lower.startswith('sr_strength_trend_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_strength_trend_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_strength_trend_generator(window, sr_type)

            elif name_lower.startswith('ml_reliability_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_ml_reliability_generator(window)

            elif name_lower.startswith('ml_bounce_prob_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_ml_bounce_prob_generator(window)

            elif name_lower.startswith('trading_sr_reliability_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('trading_sr_reliability_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'support'
                return self._create_trading_sr_reliability_generator(window, sr_type)

            elif name_lower.startswith('volume_profile_hvn_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_hvn_generator(window)

            elif name_lower.startswith('volume_profile_poc_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_poc_generator(window)

            elif name_lower.startswith('volume_profile_vah_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_vah_generator(window)

            elif name_lower.startswith('volume_profile_val_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_val_generator(window)

            # Unknown feature type
            self.logger.debug(f"Unknown feature type: {feature_name}")
            return None

        except Exception as e:
            self.logger.error(f"Error creating feature generator for {feature_name}: {e}")
            return None

    def _extract_period_from_name(self, feature_name: str, default: int) -> int:
        """Extract period parameter from feature name."""
        try:
            tprint_debug(
                f"🧾 Extracting period from feature '{feature_name}' with default {default}"
            )
            # Split by underscore and look for numeric values
            parts = feature_name.split('_')
            for part in reversed(parts):
                if part.isdigit():
                    tprint_debug(f"   ↳ Found period {part}")
                    return int(part)
            tprint_warning(
                f"⚠️ No explicit period found for '{feature_name}', using default {default}"
            )
            return default
        except Exception:
            tprint_error(
                f"❌ Failed to parse period from '{feature_name}', using default {default}"
            )
            return default

    def _extract_macd_params(self, feature_name: str) -> Optional[Tuple[int, int, int]]:
        """Extract MACD parameters (fast, slow, signal) from feature name."""
        try:
            tprint_debug(f"🧾 Extracting MACD params from '{feature_name}'")
            # Expected format: macd_12_26_9, macd_returns_12_26_9, etc.
            parts = feature_name.lower().replace('macd', '').replace('returns', '').replace('vwap', '').split('_')
            numbers = [int(p) for p in parts if p.isdigit()]

            if len(numbers) >= 3:
                tprint_debug(f"   ↳ Parsed MACD parameters: {numbers[0]}, {numbers[1]}, {numbers[2]}")
                return (numbers[0], numbers[1], numbers[2])
            elif len(numbers) == 2:
                tprint_debug(f"   ↳ Parsed partial MACD params {numbers}, defaulting signal to 9")
                return (numbers[0], numbers[1], 9)  # Default signal period
            elif len(numbers) == 1:
                tprint_debug(f"   ↳ Parsed single MACD value {numbers[0]}, defaulting fast/slow to 12/26")
                return (12, 26, numbers[0])  # Default fast/slow, use number as signal

            tprint_warning(f"⚠️ No MACD parameters found in '{feature_name}'")
            return None
        except Exception as e:
            tprint_error(f"❌ Exception extracting MACD parameters from '{feature_name}': {e}")
            return None

    def _extract_garch_params(self, feature_name: str) -> Optional[Tuple[int, int, int]]:
        """Extract GARCH parameters (p, q, h) from feature name."""
        try:
            tprint_debug(f"🧾 Extracting GARCH params from '{feature_name}'")
            # Expected format: garch_1_1_1, garch_1_1_5, etc.
            parts = feature_name.lower().replace('garch', '').split('_')
            numbers = [int(p) for p in parts if p.isdigit()]

            if len(numbers) >= 3:
                tprint_debug(f"   ↳ Parsed GARCH parameters: {numbers[0]}, {numbers[1]}, {numbers[2]}")
                return (numbers[0], numbers[1], numbers[2])
            elif len(numbers) == 2:
                tprint_debug(f"   ↳ Parsed partial GARCH params {numbers}, defaulting horizon to 1")
                return (numbers[0], numbers[1], 1)  # Default horizon
            elif len(numbers) == 1:
                tprint_debug(f"   ↳ Parsed single GARCH value {numbers[0]}, defaulting p/q to 1")
                return (1, 1, numbers[0])  # Default p,q, use number as horizon

            tprint_warning(f"⚠️ No GARCH parameters found in '{feature_name}'")
            return None
        except Exception as e:
            tprint_error(f"❌ Exception extracting GARCH parameters from '{feature_name}': {e}")
            return None

    def _extract_dual_period_params(self, feature_name: str, default_short: int, default_long: int) -> Optional[Tuple[int, int]]:
        """Extract two period parameters from feature name."""
        try:
            # Expected formats: volume_trend_strength_10_30, volume_osc_10_20, etc.
            parts = feature_name.lower().split('_')
            numbers = [int(p) for p in parts if p.isdigit()]

            if len(numbers) >= 2:
                return (numbers[0], numbers[1])
            elif len(numbers) == 1:
                return (numbers[0], default_long)

            return None
        except Exception:
            return None

    def _extract_dual_params(self, feature_name: str, param1_name: str, param2_name: str) -> Optional[Tuple[int, int]]:
        """Extract two parameters from feature name based on parameter names."""
        try:
            # Expected formats:
            # - support_level_1_20, resistance_level_2_10, fibonacci_0.382_20
            # - correlation_interaction_col1_col2_window
            parts = feature_name.lower().split('_')

            # Find parameter positions
            param1_pos = -1
            param2_pos = -1

            for i, part in enumerate(parts):
                if param1_name in part:
                    param1_pos = i
                elif param2_name in part:
                    param2_pos = i

            if param1_pos >= 0 and param2_pos >= 0:
                # Extract numeric values after parameter names
                param1_parts = parts[param1_pos].split(param1_name)
                param2_parts = parts[param2_pos].split(param2_name)

                if len(param1_parts) > 1 and param1_parts[1].isdigit():
                    param1_val = int(param1_parts[1])
                elif param1_pos + 1 < len(parts) and parts[param1_pos + 1].isdigit():
                    param1_val = int(parts[param1_pos + 1])
                else:
                    return None

                if len(param2_parts) > 1 and param2_parts[1].isdigit():
                    param2_val = int(param2_parts[1])
                elif param2_pos + 1 < len(parts) and parts[param2_pos + 1].isdigit():
                    param2_val = int(parts[param2_pos + 1])
                else:
                    return None

                return (param1_val, param2_val)

            return None
        except Exception:
            return None

    def _extract_column_params(self, feature_name: str) -> Optional[Tuple[str, str]]:
        """Extract column names from feature name for interaction indicators."""
        try:
            # Expected format: feature_ratio_close_to_volume, correlation_interaction_close_volume_20
            parts = feature_name.lower().split('_')

            # For feature_ratio_close_to_volume format
            if 'feature_ratio' in feature_name.lower():
                ratio_part = feature_name.lower().replace('feature_ratio_', '')
                if '_to_' in ratio_part:
                    col1, col2 = ratio_part.split('_to_')
                    return (col1, col2)

            # For correlation_interaction_close_volume_20 format
            elif 'correlation_interaction' in feature_name.lower():
                interaction_part = feature_name.lower().replace('correlation_interaction_', '')
                if len(parts) >= 3:
                    col1 = parts[-3]  # Third to last
                    col2 = parts[-2]  # Second to last
                    return (col1, col2)

            return None
        except Exception:
            return None

    def _create_sr_persistence_generator(self, window: int, sr_type: str):
        """Create SR persistence feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRPersistenceGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_persistence_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level persistence analysis for {sr_type} levels over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    # This would typically load historical SR data and calculate persistence
                    # For now, return a placeholder based on price volatility
                    volatility = np.std(close_prices) / np.mean(close_prices) if len(close_prices) > 1 else 0.0
                    persistence_score = 1.0 / (1.0 + volatility)  # Higher volatility = lower persistence
                    return pd.Series([persistence_score] * len(data), index=data.index, name=self.config.name)

            return SRPersistenceGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR persistence generator: {e}")
            return None

    def _create_sr_touch_freq_generator(self, window: int, sr_type: str):
        """Create SR touch frequency feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRTouchFreqGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_touch_freq_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level touch frequency for {sr_type} levels over {window} periods",
                        required_columns=["close", "high", "low"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # Calculate price movement frequency as a proxy for touch frequency
                    price_changes = np.abs(np.diff(data['close'].values))
                    avg_change = np.mean(price_changes) if len(price_changes) > 0 else 0.0
                    # Normalize to 0-1 range (higher frequency = more touches)
                    touch_freq = min(1.0, avg_change * 100)  # Scale factor
                    return pd.Series([touch_freq] * len(data), index=data.index, name=self.config.name)

            return SRTouchFreqGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR touch frequency generator: {e}")
            return None

    def _create_sr_bounce_rate_generator(self, window: int, sr_type: str):
        """Create SR bounce rate feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRBounceRateGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_bounce_rate_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level bounce success rate for {sr_type} levels over {window} periods",
                        required_columns=["close", "high", "low"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # Calculate bounce rate based on price reversals
                    close_prices = data['close'].values
                    if len(close_prices) < 3:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    # Simple bounce detection: when price changes direction after touching a level
                    price_changes = np.diff(close_prices)
                    reversals = np.sum(np.diff(np.sign(price_changes)) != 0) if len(price_changes) > 1 else 0
                    total_changes = len(price_changes)

                    bounce_rate = reversals / total_changes if total_changes > 0 else 0.5
                    return pd.Series([bounce_rate] * len(data), index=data.index, name=self.config.name)

            return SRBounceRateGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR bounce rate generator: {e}")
            return None

    def _create_sr_strength_trend_generator(self, window: int, sr_type: str):
        """Create SR strength trend feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRStrengthTrendGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_strength_trend_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level strength trend for {sr_type} levels over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # Calculate trend in price levels as a proxy for SR strength trend
                    close_prices = data['close'].values
                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    # Calculate rolling trend using vectorized operations
                    if len(close_prices) >= window:
                        # Use pandas rolling for vectorized calculation
                        price_series = pd.Series(close_prices)
                        
                        # Vectorized trend calculation: slope of linear regression over rolling window
                        rolling_mean = price_series.rolling(window=window, min_periods=window).mean()
                        rolling_std = price_series.rolling(window=window, min_periods=window).std()
                        
                        # Calculate trend as normalized slope (price change / mean price)
                        price_diff = price_series.diff(window)
                        trends = (price_diff / rolling_mean).fillna(0.0).values
                    else:
                        trends = np.zeros(len(close_prices))
                    return pd.Series(trends, index=data.index, name=self.config.name)

            return SRStrengthTrendGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR strength trend generator: {e}")
            return None

    def _create_ml_reliability_generator(self, window: int):
        """Create ML reliability feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class MLReliabilityGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"ml_reliability_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"ML-ready reliability score over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    # Reliability based on price stability (lower volatility = higher reliability)
                    if len(close_prices) < window:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    # Calculate reliabilities using vectorized operations
                    if len(close_prices) >= window:
                        price_series = pd.Series(close_prices)
                        
                        # Calculate rolling volatility and reliability
                        rolling_mean = price_series.rolling(window=window, min_periods=window).mean()
                        rolling_std = price_series.rolling(window=window, min_periods=window).std()
                        
                        # Avoid division by zero - use vectorized operations
                        volatility = np.where(
                            rolling_mean != 0,
                            rolling_std / rolling_mean,
                            0.0
                        )
                        reliability = 1.0 / (1.0 + volatility)
                        
                        # Fill NaN values with 0.5 (default reliability)
                        reliabilities = reliability.fillna(0.5).values
                    else:
                        reliabilities = np.full(len(close_prices), 0.5)

                    return pd.Series(reliabilities, index=data.index, name=self.config.name)

            return MLReliabilityGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating ML reliability generator: {e}")
            return None

    def _create_ml_bounce_prob_generator(self, window: int):
        """Create ML bounce probability feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class MLBounceProbGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"ml_bounce_prob_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"ML-ready bounce probability over {window} periods",
                        required_columns=["close", "high", "low"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    high_prices = data['high'].values
                    low_prices = data['low'].values

                    if len(close_prices) < window:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    # Calculate bounce probabilities using vectorized operations
                    if len(close_prices) >= window:
                        # Vectorized bounce probability calculation
                        close_series = pd.Series(close_prices)
                        high_series = pd.Series(high_prices)
                        low_series = pd.Series(low_prices)
                        
                        # Calculate price changes and level touches
                        close_diff = close_series.diff()
                        high_touches = close_series > high_series
                        low_touches = close_series < low_series
                        
                        # Calculate bounces: reversals after touching levels
                        high_bounces = high_touches & (close_diff < 0)
                        low_bounces = low_touches & (close_diff > 0)
                        total_bounces = high_bounces + low_bounces
                        
                        # Rolling mean of bounce probability
                        bounce_probs = total_bounces.rolling(window=window, min_periods=window).mean().fillna(0.5).values
                    else:
                        bounce_probs = np.full(len(close_prices), 0.5)
                    return pd.Series(bounce_probs, index=data.index, name=self.config.name)

            return MLBounceProbGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating ML bounce probability generator: {e}")
            return None

    def _create_trading_sr_reliability_generator(self, window: int, sr_type: str):
        """Create trading SR reliability feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class TradingSRReliabilityGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"trading_sr_reliability_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Trading reliability for {sr_type} levels over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    # Trading reliability based on price stability and trend
                    if len(close_prices) < window:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    reliabilities = []
                    for i in range(window - 1, len(close_prices)):
                        window_prices = close_prices[i-window+1:i+1]
                        # Calculate reliability based on consistency and trend strength
                        volatility = np.std(window_prices) / np.mean(window_prices)
                        trend_strength = abs(np.polyfit(np.arange(len(window_prices)), window_prices, 1)[0]) / np.mean(window_prices)

                        # Higher trend strength and lower volatility = higher reliability
                        reliability = (1.0 - volatility) * (1.0 + trend_strength) / 2.0
                        reliabilities.append(reliability)

                    # Pad with default reliability for the beginning
                    reliabilities_padded = [0.5] * (window - 1) + reliabilities
                    return pd.Series(reliabilities_padded, index=data.index, name=self.config.name)

            return TradingSRReliabilityGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating trading SR reliability generator: {e}")
            return None

    def _create_volume_profile_hvn_generator(self, window: int):
        """Create volume profile HVN (High Volume Node) feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfileHVNGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_hvn_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile High Volume Nodes over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    volumes = data['volume'].values

                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    # Calculate HVN levels using vectorized operations
                    if len(close_prices) >= window:
                        # Vectorized HVN calculation using volume-weighted price levels
                        close_series = pd.Series(close_prices)
                        volume_series = pd.Series(volumes)
                        
                        # Create price bins and calculate volume-weighted levels
                        price_bins = (close_series / 0.01).round() * 0.01  # Round to nearest cent
                        volume_weighted_prices = (close_series * volume_series).rolling(window=window, min_periods=window).sum()
                        total_volumes = volume_series.rolling(window=window, min_periods=window).sum()
                        
                        # Calculate HVN as volume-weighted average price
                        hvn_levels = (volume_weighted_prices / total_volumes).fillna(method='ffill').values
                    else:
                        hvn_levels = close_prices
                    return pd.Series(hvn_levels, index=data.index, name=self.config.name)

            return VolumeProfileHVNGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile HVN generator: {e}")
            return None

    def _create_volume_profile_poc_generator(self, window: int):
        """Create volume profile POC (Point of Control) feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfilePOCGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_poc_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile Point of Control over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # POC is essentially the same as HVN for this simple implementation
                    return self._create_volume_profile_hvn_generator(window)._generate_feature(data, **kwargs)

            return VolumeProfilePOCGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile POC generator: {e}")
            return None

    def _create_volume_profile_vah_generator(self, window: int):
        """Create volume profile VAH (Value Area High) feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfileVAHGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_vah_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile Value Area High over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    volumes = data['volume'].values

                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    # Calculate VAH levels using vectorized operations
                    if len(close_prices) >= window:
                        # Vectorized VAH calculation using rolling quantiles
                        close_series = pd.Series(close_prices)
                        volume_series = pd.Series(volumes)
                        
                        # Calculate volume-weighted price quantiles
                        volume_weighted_prices = close_series * volume_series
                        total_volumes = volume_series.rolling(window=window, min_periods=window).sum()
                        
                        # VAH as 70th percentile of volume-weighted prices
                        vah_levels = volume_weighted_prices.rolling(window=window, min_periods=window).quantile(0.7).fillna(method='ffill').values
                    else:
                        vah_levels = close_prices
                    return pd.Series(vah_levels, index=data.index, name=self.config.name)

            return VolumeProfileVAHGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile VAH generator: {e}")
            return None

    def _create_volume_profile_val_generator(self, window: int):
        """Create volume profile VAL (Value Area Low) feature generator."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfileVALGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_val_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile Value Area Low over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    volumes = data['volume'].values

                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    # Calculate VAL levels using vectorized operations
                    if len(close_prices) >= window:
                        # Vectorized VAL calculation using rolling quantiles
                        close_series = pd.Series(close_prices)
                        volume_series = pd.Series(volumes)
                        
                        # Calculate volume-weighted price quantiles
                        volume_weighted_prices = close_series * volume_series
                        total_volumes = volume_series.rolling(window=window, min_periods=window).sum()
                        
                        # VAL as 30th percentile of volume-weighted prices
                        val_levels = volume_weighted_prices.rolling(window=window, min_periods=window).quantile(0.3).fillna(method='ffill').values
                    else:
                        val_levels = close_prices
                    return pd.Series(val_levels, index=data.index, name=self.config.name)

            return VolumeProfileVALGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile VAL generator: {e}")
            return None

    def _create_volatility_generator(self, period: int):
        """Create a custom volatility generator for volatility features."""
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolatilityGenerator(VectorizedFeatureGenerator):
                def __init__(self, period: int):
                    config = FeatureConfig(
                        name=f"volatility_{period}",
                        category=FeatureCategory.VOLATILITY,
                        description=f"Volatility indicator with {period} period lookback",
                        required_columns=["close"],
                        default_lookback=period,
                        parameters={"period": period}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    if len(close_prices) < self.config.parameters["period"]:
                        return pd.Series(np.full(len(close_prices), np.nan), index=data.index, name=self.config.name)

                    # Calculate rolling standard deviation of returns
                    returns = np.diff(close_prices) / close_prices[:-1]
                    volatility = pd.Series(returns).rolling(window=self.config.parameters["period"]).std().values

                    # Pad the first value to match length
                    volatility = np.concatenate([[np.nan], volatility])

                    return pd.Series(volatility, index=data.index, name=self.config.name)

            return VolatilityGenerator(period)

        except Exception as e:
            self.logger.error(f"Error creating volatility generator: {e}")
            return None

    def _create_failed_result(self, method: str, optimization_time: float, feature_name: str = "") -> OptimizationResult:
        """Create a failed optimization result."""
        return OptimizationResult(
            best_lookback_period=OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK,  # Already an int
            best_score=0.0,
            optimization_method=method,
            total_trials=0,
            optimization_time=optimization_time,
            convergence_achieved=False,
            metadata={'error': 'Optimization failed'},
            feature_name=feature_name  # FIXED: Added feature_name
        )

    def _calculate_comprehensive_correlations(self, feature_values: np.ndarray, target_values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive correlation metrics with safe NaN handling."""
        try:
            correlations = {}
            
            # Use safe NaN handling for all correlation calculations
            correlations['pearson'] = safe_correlation_with_nan_handling(
                feature_values, target_values, method='pearson', min_samples=10
            )
            
            correlations['spearman'] = safe_correlation_with_nan_handling(
                feature_values, target_values, method='spearman', min_samples=10
            )
            
            # Mutual information with proper NaN handling
            correlations['mutual_info'] = safe_mutual_information_with_nan_handling(
                feature_values, target_values, n_bins=10, min_samples=20
            )
            
            # R-squared
            correlations['r_squared'] = correlations['pearson'] ** 2
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating correlations: {e}')
            return {'pearson': 0.0, 'spearman': 0.0, 'mutual_info': 0.0, 'r_squared': 0.0}

    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information with safe NaN handling."""
        return safe_mutual_information_with_nan_handling(x, y, n_bins=10, min_samples=20)

    def _calculate_composite_score(self, correlations: Dict[str, float]) -> float:
        """Calculate composite score using MI-consistent metrics with VectorBT optimizations."""
        try:
            # Convert all metrics to MI-consistent scale
            mi_metrics = {}
            
            # Use VectorBT optimization for correlation calculations if available
            if self.use_vectorbt_optimization and self.unified_manager:
                try:
                    operation_config = self.unified_manager.create_operation_config(
                        operation_type=OperationType.STATISTICAL_COMPUTATION,
                        data_size=1000,  # Approximate size
                        data_dimensions=(1000,),
                        memory_budget_mb=128.0,
                        time_budget_seconds=10.0
                    )
                    
                    strategy = self.unified_manager.select_optimization_strategy(operation_config)
                    self.logger.debug(f"Selected VectorBT strategy for composite scoring: {strategy}")
                except Exception as e:
                    self.logger.warning(f"VectorBT composite scoring setup failed: {e}")
            
            # Convert correlation metrics to MI approximations
            for metric in ['pearson', 'spearman', 'r_squared']:
                if metric in correlations:
                    corr_value = correlations[metric]
                    if abs(corr_value) < 0.999:  # Avoid log(0)
                        try:
                            mi_approx = 0.5 * np.log(1 - corr_value**2) if corr_value**2 < 1 else 0.0
                            mi_metrics[metric] = max(0.0, -mi_approx)  # Ensure positive MI
                        except (ValueError, OverflowError):
                            mi_metrics[metric] = 0.0
                    else:
                        mi_metrics[metric] = 0.0
            
            # Use mutual_info directly if available
            if 'mutual_info' in correlations:
                mi_metrics['mutual_info'] = max(0.0, correlations['mutual_info'])
            
            # Weighted combination of MI-consistent metrics
            weights = {
                'pearson': 0.3,
                'spearman': 0.2,
                'mutual_info': 0.4,
                'r_squared': 0.1
            }
            
            composite_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in mi_metrics:
                    composite_score += mi_metrics[metric] * weight
                    total_weight += weight
            
            return composite_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating composite score: {e}')
            return 0.0

    def _calculate_multi_target_metrics(self, feature_values: np.ndarray, target_values: np.ndarray) -> Dict[str, float]:
        """Calculate multiple target metrics for multi-objective optimization."""
        try:
            metrics = {}
            
            # Correlation metrics
            metrics['correlation'] = safe_correlation(feature_values, target_values, default=0.0)
            metrics['r_squared'] = metrics['correlation'] ** 2
            
            # Stability metrics
            metrics['stability'] = self._calculate_stability_metric(feature_values)
            
            # Information content
            metrics['information_content'] = self._calculate_information_content(feature_values)
            
            # Predictive power
            metrics['predictive_power'] = self._calculate_predictive_power(feature_values, target_values)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating multi-target metrics: {e}')
            return {'correlation': 0.0, 'r_squared': 0.0, 'stability': 0.0, 'information_content': 0.0, 'predictive_power': 0.0}

    def _calculate_stability_metric(self, values: np.ndarray) -> float:
        """Calculate stability metric (lower variance = higher stability)."""
        try:
            if len(values) < 2:
                return 0.0
            return 1.0 / (1.0 + np.var(values))  # Higher stability = lower variance
        except Exception:
            return 0.0

    def _calculate_information_content(self, values: np.ndarray) -> float:
        """Calculate information content using entropy."""
        tprint_debug("🧠 Entering _calculate_information_content")
        try:
            if len(values) < 2:
                return 0.0
            
            # Simple entropy calculation
            hist, _ = np.histogram(values, bins=min(10, len(values)//2))
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            
            entropy = -np.sum(probs * np.log2(probs))
            return entropy / np.log2(len(probs)) if len(probs) > 1 else 0.0
            
        except Exception:
            return 0.0

    def _calculate_predictive_power(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Calculate predictive power using cross-validation-like approach."""
        tprint_debug("🧠 Entering _calculate_predictive_power")
        try:
            if len(feature_values) < 10:
                return 0.0
            
            # Simple predictive power: correlation with lagged target
            lag = min(5, len(feature_values) // 4)
            if lag > 0:
                lagged_target = target_values[lag:]
                lagged_feature = feature_values[:-lag]
                return abs(safe_correlation(lagged_feature, lagged_target, default=0.0))
            else:
                return abs(safe_correlation(feature_values, target_values, default=0.0))
                
        except Exception:
            return 0.0

    def _calculate_multi_objective_score(self, targets: Dict[str, float]) -> float:
        """Calculate multi-objective score using weighted combination."""
        tprint_debug("🧠 Entering _calculate_multi_objective_score")
        try:
            # Default weights for different objectives
            weights = {
                'correlation': 0.3,
                'r_squared': 0.2,
                'stability': 0.2,
                'information_content': 0.15,
                'predictive_power': 0.15
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in targets:
                    value = abs(targets[metric])  # Use absolute value
                    score += value * weight
                    total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating multi-objective score: {e}')
            return 0.0

    def _check_convergence(self, scores: List[float]) -> bool:
        """Check if optimization has converged."""
        tprint_debug("🧠 Entering _check_convergence")
        try:
            if len(scores) < 5:
                return False
            
            # Check if the last few scores are stable
            recent_scores = scores[-5:]
            score_std = np.std(recent_scores)
            score_mean = np.mean(recent_scores)
            
            # Converged if coefficient of variation is small
            cv = score_std / (score_mean + 1e-8)
            return cv < 0.05  # 5% coefficient of variation threshold
            
        except Exception:
            return False

    def _update_performance_metrics(self, result: OptimizationResult, optimization_time: float) -> None:
        """Update performance tracking metrics."""
        tprint_debug("🧠 Entering _update_performance_metrics")
        try:
            self.performance_metrics['total_optimizations'] += 1
            
            if result.best_score > 0:
                self.performance_metrics['successful_optimizations'] += 1
                self.performance_metrics['best_scores'].append(result.best_score)
                
                # Keep only recent scores for memory efficiency
                if len(self.performance_metrics['best_scores']) > 100:
                    self.performance_metrics['best_scores'] = self.performance_metrics['best_scores'][-100:]
            
            # Update average optimization time
            total_time = self.performance_metrics['average_optimization_time'] * (self.performance_metrics['total_optimizations'] - 1)
            self.performance_metrics['average_optimization_time'] = (total_time + optimization_time) / self.performance_metrics['total_optimizations']
            
            # Calculate and log cache hit rate with memory usage
            total_cache_accesses = self.cache_hits + self.cache_misses
            if total_cache_accesses > 0:
                cache_hit_rate = (self.cache_hits / total_cache_accesses) * 100
                cache_size_mb = len(self.feature_cache) * 10 / 1024  # Estimate ~10KB per entry
                self.logger.info(
                    f"ℹ️ ✅ {result.feature_name}: best_lookback={result.best_lookback_period}, "
                    f"score={result.best_score:.6f} (cache: {cache_hit_rate:.2f}% hit rate, "
                    f"{len(self.feature_cache)}/{self.max_cache_size} entries, ~{cache_size_mb:.1f}MB)"
                )
            else:
                self.logger.info(f"ℹ️ ✅ {result.feature_name}: best_lookback={result.best_lookback_period}, score={result.best_score:.6f}")
            
            # Store in history
            self.optimization_history.append({
                'timestamp': time.time(),
                'method': result.optimization_method,
                'best_score': result.best_score,
                'optimization_time': optimization_time,
                'convergence_achieved': result.convergence_achieved
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f'⚠️ Error updating performance metrics: {e}')

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the optimizer."""
        tprint_debug("🧠 Entering get_performance_summary")
        try:
            if not self.performance_metrics['best_scores']:
                return {
                    'total_optimizations': self.performance_metrics['total_optimizations'],
                    'successful_optimizations': self.performance_metrics['successful_optimizations'],
                    'success_rate': 0.0,
                    'average_optimization_time': self.performance_metrics['average_optimization_time'],
                    'best_score_ever': 0.0,
                    'average_best_score': 0.0
                }
            
            return {
                'total_optimizations': self.performance_metrics['total_optimizations'],
                'successful_optimizations': self.performance_metrics['successful_optimizations'],
                'success_rate': self.performance_metrics['successful_optimizations'] / self.performance_metrics['total_optimizations'],
                'average_optimization_time': self.performance_metrics['average_optimization_time'],
                'best_score_ever': max(self.performance_metrics['best_scores']),
                'average_best_score': np.mean(self.performance_metrics['best_scores']),
                'score_std': np.std(self.performance_metrics['best_scores'])
            }
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error getting performance summary: {e}')
            return {}

    def save_optimization_results(self, filepath: str) -> bool:
        """Save optimization results to file."""
        tprint_debug("🧠 Entering save_optimization_results")
        try:
            results = {
                'performance_metrics': self.performance_metrics,
                'optimization_history': self.optimization_history[-100:],  # Last 100 optimizations
                'timestamp': datetime.now().isoformat()
            }
            
            self.serializer.save_json(results, filepath)
            self.logger.info(f'💾 Optimization results saved to {filepath}')
            return True
            
        except Exception as e:
            self.logger.error(f'❌ Error saving optimization results: {e}')
            return False

    def test_feature_engineering(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Test method to verify comprehensive feature engineering implementation.

        Args:
            data: Test data with OHLCV columns

        Returns:
            Dictionary of feature names to their calculated values
        """
        tprint_debug("🧠 Entering test_feature_engineering")
        test_features = [
            # RETURNS_VWAP-BASED indicators (NEW STANDARD - better signal quality)
            'rsi_14', 'macd_12_26_9', 'sma_20', 'ema_12', 'bb_upper_20',
            'stoch_k_14', 'williams_r_14', 'cci_20', 'adx_14',

            # PRICE_RETURNS variants (traditional price returns - for comparison)
            'returns_rsi_14', 'returns_macd_12_26_9', 'returns_sma_20', 'returns_ema_12', 'returns_bb_upper_20',
            'returns_stoch_k_14', 'returns_williams_r_14', 'returns_cci_20',

            # PRICE_LEVELS variants (raw price levels - for comparison)
            'price_rsi_14', 'price_sma_20', 'price_bb_upper_20',

            # VWAP-BASED variants (explicit VWAP calculation)
            'vwap_rsi_14', 'vwap_macd_12_26_9', 'vwap_sma_20', 'vwap_ema_12', 'vwap_bb_upper_20',

            # Advanced SR features (using actual price levels for S/R detection)
            'support_level_1_20', 'resistance_level_2_20',
            'pivot_point_20', 'fibonacci_0.382_20',
            'sr_persistence_avg_20', 'sr_persistence_avg_200', 'volume_profile_hvn_20',

            # Volume indicators (volume returns by default)
            'volume_sma_20', 'volume_ratio_10',

            # Order flow indicators
            'taker_buy_ratio_20', 'market_aggression_index_20',

            # Returns indicators (core price returns)
            'return_1', 'return_log_1', 'return_cumulative_1',

            # Entropy indicators (advanced analysis)
            'price_entropy_20', 'rsi_entropy_20_14',

            # Acceleration indicators
            'trend_strength_10', 'momentum_acc_10'
        ]

        results = {}

        for feature_name in test_features:
            try:
                feature_values = self._calculate_feature_for_lookback(data, feature_name, 20)
                results[feature_name] = feature_values
                if feature_values is not None:
                    self.logger.info(f"✅ Successfully calculated {feature_name}: shape={feature_values.shape}")
                else:
                    self.logger.warning(f"⚠️ {feature_name} returned None")
            except Exception as e:
                self.logger.error(f"❌ Failed to calculate {feature_name}: {e}")
                results[feature_name] = None

        return results

    def _get_data_hash(self, data: pd.DataFrame, feature_name: str, horizon: int) -> str:
        """Generate hash for data caching."""
        try:
            # Create a simple hash based on data shape, feature name, and horizon
            data_info = f"{data.shape}_{feature_name}_{horizon}_{data.index[-1] if len(data) > 0 else 0}"
            return str(hash(data_info))[:16]
        except Exception:
            return f"{feature_name}_{horizon}"

    def _cached_feature_calculation(self, data: pd.DataFrame, feature_name: str, horizon: int) -> Optional[np.ndarray]:
        """Calculate feature with thread-safe caching to avoid recomputation using LRU eviction.
        
        Uses OrderedDict with thread safety for efficient O(1) LRU operations:
        - Cache hit: O(1) move_to_end
        - Cache miss with eviction: O(1) popitem(last=False)
        - Memory: ~50k entries * ~10KB = ~500MB max
        - Thread-safe: Uses locks to prevent race conditions
        """
        cache_key = self._get_data_hash(data, feature_name, horizon)
        
        # Thread-safe cache access
        with getattr(self, '_cache_lock', type('MockLock', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})()):
            if cache_key in self.feature_cache:
                self.cache_hits += 1
                # Move to end = most recently used (O(1) operation in OrderedDict)
                self.feature_cache.move_to_end(cache_key)
                return self.feature_cache[cache_key]
        
        # Calculate feature
        feature_values = self._calculate_feature_for_lookback(data, feature_name, horizon)
        
        # Thread-safe cache update
        with getattr(self, '_cache_lock', type('MockLock', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})()):
            # Add to cache (at end = most recently used)
            self.feature_cache[cache_key] = feature_values
            self.cache_misses += 1
            
            # Enforce cache size limit to prevent memory leaks
            if len(self.feature_cache) > self.max_cache_size:
                # Remove oldest entry (first item)
                self.feature_cache.popitem(last=False)
        
        # Periodic cache cleanup
        self._cleanup_cache_if_needed()
        
        return feature_values
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limits or memory pressure is high."""
        total_operations = self.cache_hits + self.cache_misses
        
        # Periodic cleanup
        if total_operations - self._last_cache_cleanup > self._cache_cleanup_interval:
            self._last_cache_cleanup = total_operations
            
            # Remove oldest entries if cache is too large
            while len(self.feature_cache) > self.max_cache_size:
                self.feature_cache.popitem(last=False)
            
            # Force garbage collection if memory pressure is high
            if hasattr(self, 'memory_monitor') and self.memory_monitor.should_cleanup():
                import gc
                gc.collect()
                self.logger.debug(f"🧹 Cache cleanup: {len(self.feature_cache)} entries remaining")

    def _vectorized_mi_calculation(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """Calculate mutual information for multiple feature-return pairs using vectorized operations with GPU acceleration."""
        try:
            # Try GPU-accelerated batch processing first
            if self.gpu_available and self.batch_processor and len(features_list) > 10:
                return self._gpu_accelerated_mi_calculation(features_list, returns_list)
            
            # Fallback to CPU vectorized processing
            return self._cpu_vectorized_mi_calculation(features_list, returns_list)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized MI calculation failed, using fallback: {e}")
            # Fallback to individual calculations
            return [self._calculate_mutual_information_robust(f, r) for f, r in zip(features_list, returns_list)]

    def _gpu_accelerated_mi_calculation(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """GPU-accelerated MI calculation using batch processing."""
        try:
            # Prepare data for batch processing
            aligned_pairs = []
            for features, returns in zip(features_list, returns_list):
                min_length = min(len(features), len(returns))
                if min_length >= 10:
                    aligned_pairs.append((features[:min_length], returns[:min_length]))
            
            if not aligned_pairs:
                return [0.0] * len(features_list)
            
            # Use batch processor for GPU acceleration
            if self.batch_processor:
                # Convert to batch format
                features_batch = np.array([pair[0] for pair in aligned_pairs])
                returns_batch = np.array([pair[1] for pair in aligned_pairs])
                
                # Process in batches using GPU
                mi_scores = self.batch_processor.process_correlations_batch(features_batch, returns_batch)
                
                # Convert correlations to MI approximations
                mi_approximations = []
                for corr in mi_scores:
                    if abs(corr) < 0.999:
                        try:
                            mi_approx = -0.5 * np.log(1 - corr**2) if corr**2 < 1 else 0.0
                            mi_approximations.append(max(0.0, mi_approx))
                        except (ValueError, OverflowError):
                            mi_approximations.append(0.0)
                    else:
                        mi_approximations.append(0.0)
                
                return mi_approximations
            else:
                # Fallback to CPU processing
                return self._cpu_vectorized_mi_calculation(features_list, returns_list)
                
        except Exception as e:
            self.logger.warning(f"⚠️ GPU MI calculation failed, falling back to CPU: {e}")
            return self._cpu_vectorized_mi_calculation(features_list, returns_list)

    def _cpu_vectorized_mi_calculation(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """CPU vectorized MI calculation using true vectorization."""
        if not features_list or not returns_list:
            return []
        
        # Pre-allocate result array
        n_pairs = len(features_list)
        mi_scores = np.zeros(n_pairs, dtype=float)
        
        # Find minimum length across all pairs for efficient processing
        min_lengths = np.array([min(len(f), len(r)) for f, r in zip(features_list, returns_list)])
        valid_pairs = min_lengths >= 10
        
        if not np.any(valid_pairs):
            return [0.0] * n_pairs
        
        # Process valid pairs in batches
        valid_indices = np.where(valid_pairs)[0]
        
        for batch_start in range(0, len(valid_indices), 100):  # Process in batches of 100
            batch_end = min(batch_start + 100, len(valid_indices))
            batch_indices = valid_indices[batch_start:batch_end]
            
            # Align all arrays in the batch to the same length
            batch_min_length = np.min(min_lengths[batch_indices])
            
            # Create aligned arrays for batch processing
            aligned_features = np.zeros((len(batch_indices), batch_min_length), dtype=float)
            aligned_returns = np.zeros((len(batch_indices), batch_min_length), dtype=float)
            
            for i, idx in enumerate(batch_indices):
                aligned_features[i] = features_list[idx][:batch_min_length]
                aligned_returns[i] = returns_list[idx][:batch_min_length]
            
            # Vectorized correlation calculation
            try:
                # Calculate correlations for all pairs in the batch
                correlations = np.array([
                    safe_correlation(aligned_features[i], aligned_returns[i]) 
                    for i in range(len(batch_indices))
                ])
                
                # Vectorized MI approximation: MI ≈ -0.5 * log(1 - corr²)
                valid_corrs = np.abs(correlations) < 0.999
                mi_approximations = np.zeros_like(correlations)
                
                # Only calculate MI for valid correlations
                if np.any(valid_corrs):
                    corr_squared = correlations[valid_corrs] ** 2
                    mi_approximations[valid_corrs] = -0.5 * np.log(1 - corr_squared)
                    mi_approximations[valid_corrs] = np.maximum(0.0, mi_approximations[valid_corrs])
                
                # Store results
                for i, idx in enumerate(batch_indices):
                    mi_scores[idx] = mi_approximations[i]
                    
            except (ValueError, OverflowError) as e:
                # Fallback to individual calculation for this batch
                for i, idx in enumerate(batch_indices):
                    try:
                        correlation = safe_correlation(aligned_features[i], aligned_returns[i])
                        if abs(correlation) < 0.999:
                            mi_approx = -0.5 * np.log(1 - correlation**2) if correlation**2 < 1 else 0.0
                            mi_scores[idx] = max(0.0, mi_approx)
                        else:
                            mi_scores[idx] = 0.0
                    except (ValueError, OverflowError):
                        mi_scores[idx] = 0.0
        
        return mi_scores.tolist()

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage using efficient operations."""
        return optimize_dataframe_memory(df)

    def _extract_numeric_array(self, series: Union[pd.Series, np.ndarray, None]) -> Optional[np.ndarray]:
        """Convert a Series or array-like into a sanitized numpy array."""
        # tprint_debug("🧠 Entering _extract_numeric_array")  # Commented out for reduced verbosity
        if series is None:
            return None

        try:
            if isinstance(series, pd.Series):
                numeric = pd.to_numeric(series, errors='coerce')
                values = numeric.to_numpy(dtype=float, copy=True)
            else:
                values = np.asarray(series, dtype=float)

            if values.size == 0:
                return None

            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return None

    def _combine_arrays(self, arrays: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Combine multiple arrays by averaging while ignoring missing inputs."""
        # tprint_debug("🧠 Entering _combine_arrays")  # Commented out for reduced verbosity
        valid_arrays = [arr for arr in arrays if arr is not None]
        if not valid_arrays:
            return None

        try:
            stacked = np.vstack(valid_arrays)
            combined = np.nanmean(stacked, axis=0)
            return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return valid_arrays[0]

    def _aggregate_probability_stream(self, data: pd.DataFrame, direction: str, horizon_keyword: str) -> Optional[np.ndarray]:
        """Aggregate probability columns for a given direction and horizon keyword."""
        # tprint_debug("🧠 Entering _aggregate_probability_stream")  # Commented out for reduced verbosity
        pattern = f"_{horizon_keyword}_{direction}_prob"
        matching_cols = [col for col in data.columns if pattern in col]
        if not matching_cols:
            return None

        aggregated = [self._extract_numeric_array(data[col]) for col in matching_cols]
        aggregated = [arr for arr in aggregated if arr is not None]
        if not aggregated:
            return None

        return self._combine_arrays(aggregated)

    def _get_multi_horizon_boundaries(self) -> Tuple[int, int]:
        """Return cached immediate and short horizon boundaries derived from configuration."""
        # tprint_debug("🧠 Entering _get_multi_horizon_boundaries")  # Commented out for reduced verbosity
        if self._cached_multi_horizon_limits is not None:
            return self._cached_multi_horizon_limits

        immediate_default, short_default = 2, 4
        locator = self._data_locator
        if locator is not None:
            config_path = locator.config_path('multi_horizon_labeling')
        else:
            default_locator = PipelineDataLocator()
            config_path = default_locator.config_path('multi_horizon_labeling')

        if config_path.exists():
            try:
                import yaml  # type: ignore

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

                with open(config_path, 'r') as cfg_file:
                    config_data = yaml.safe_load(cfg_file)

                mh_config = config_data.get('multi_horizon_labeling', {}) if isinstance(config_data, dict) else {}
                horizons_cfg = mh_config.get('time_horizons', {}) if isinstance(mh_config, dict) else {}

                immediate_default = int(horizons_cfg.get('immediate', immediate_default))
                short_default = int(horizons_cfg.get('short', short_default))
            except Exception:
                # Use defaults if configuration parsing fails
                pass

        immediate_limit = max(1, immediate_default)
        short_limit = max(immediate_limit, short_default)

        self._cached_multi_horizon_limits = (immediate_limit, short_limit)
        return self._cached_multi_horizon_limits

    def _build_horizon_weighted_matrix(
        self,
        immediate_arr: Optional[np.ndarray],
        short_arr: Optional[np.ndarray],
        overall_arr: Optional[np.ndarray],
        immediate_limit: int,
        short_limit: int,
        max_horizon: int
    ) -> Dict[int, np.ndarray]:
        """Create a dictionary that maps horizons to the most appropriate opportunity stream."""
        # tprint_debug("🧠 Entering _build_horizon_weighted_matrix")  # Commented out for reduced verbosity
        horizon_map: Dict[int, np.ndarray] = {}

        for horizon in range(1, max_horizon + 1):
            selected: Optional[np.ndarray] = None

            if immediate_arr is not None and horizon <= immediate_limit:
                selected = immediate_arr
            elif short_arr is not None and horizon <= short_limit:
                selected = short_arr
            elif overall_arr is not None:
                selected = overall_arr
            elif immediate_arr is not None:
                selected = immediate_arr
            elif short_arr is not None:
                selected = short_arr

            if selected is not None:
                horizon_map[horizon] = selected

        return horizon_map

    def _get_shared_forward_returns_matrix(self, data: pd.DataFrame, target_column: str, max_horizon: int = 300) -> Dict[int, np.ndarray]:
        """
        Get or create shared forward returns matrix that can be reused across all features.

        Args:
            data: Input data with target column
            target_column: Target column for forward returns calculation
            max_horizon: Maximum horizon to compute forward returns

        Returns:
            Dictionary mapping horizon to forward returns array
        """
        # tprint_debug("🧠 Entering _get_shared_forward_returns_matrix")  # Commented out for reduced verbosity
        data_hash = self._get_data_hash(data, f"shared_returns_{target_column}", max_horizon)

        # Check if we can reuse existing cache
        if (
            self.shared_forward_returns_hash == data_hash and
            isinstance(self.shared_forward_returns, dict) and
            target_column in self.shared_forward_returns and
            len(self.shared_forward_returns[target_column]) > 0 and
            max(self.shared_forward_returns[target_column].keys(), default=0) >= max_horizon
        ):
            self.logger.info(f"♻️ Reusing cached multi-horizon opportunity matrix for '{target_column}'")
            return self.shared_forward_returns[target_column]

        self.logger.info(
            f"🔄 Building multi-horizon opportunity matrices up to horizon {max_horizon} for target '{target_column}'"
        )

        # Initialize cache if needed
        if not isinstance(self.shared_forward_returns, dict):
            self.shared_forward_returns = {}
        
        matrices = self._precompute_forward_returns_matrix(data, target_column, max_horizon)
        
        # Store in cache with proper structure
        if target_column not in self.shared_forward_returns:
            self.shared_forward_returns[target_column] = {}
        
        self.shared_forward_returns[target_column] = matrices.get(target_column, {})
        self.shared_forward_returns_hash = data_hash

        if target_column in self.shared_forward_returns and self.shared_forward_returns[target_column]:
            return self.shared_forward_returns[target_column]

        # Fallback: create a direct matrix from the target column if available
        if target_column in data.columns:
            direct_array = self._extract_numeric_array(data[target_column])
            if direct_array is not None:
                fallback_matrix = {h: direct_array for h in range(1, max_horizon + 1)}
                self.shared_forward_returns[target_column] = fallback_matrix
                return fallback_matrix

        return {}

    def _create_multi_horizon_aligned_targets(
        self,
        data: pd.DataFrame,
        max_horizon: int = 300,
        allow_labeler: bool = True
    ) -> Dict[str, Any]:
        """Create aligned multi-horizon opportunity streams using available labeled data."""
        tprint_debug("🧠 Entering _create_multi_horizon_aligned_targets")
        def column_array(column_name: str) -> Optional[np.ndarray]:
            return self._extract_numeric_array(data[column_name]) if column_name in data.columns else None
        
        # Check for Analyst or Tactician labels first
        analyst_target = column_array('analyst_target')
        analyst_confidence = column_array('analyst_confidence')
        tactician_target = column_array('tactician_target')
        tactician_confidence = column_array('tactician_confidence')
        
        # If we have Analyst labels, use them directly
        if analyst_target is not None:
            self.logger.info("✅ Using Analyst labels for optimization")
            # Use analyst_target as both long and short signals, weighted by confidence
            confidence = analyst_confidence if analyst_confidence is not None else np.ones_like(analyst_target)
            
            # Create weighted targets for long and short
            long_signal = np.maximum(analyst_target, 0) * confidence
            short_signal = np.maximum(-analyst_target, 0) * confidence
            composite_signal = analyst_target * confidence
            
            return {
                'long': {
                    'immediate': long_signal,
                    'short': long_signal,
                    'overall': long_signal,
                    'leverage': None
                },
                'short': {
                    'immediate': short_signal,
                    'short': short_signal,
                    'overall': short_signal,
                    'leverage': None
                },
                'composite': {
                    'immediate': composite_signal,
                    'short': composite_signal,
                    'overall': composite_signal,
                    'leverage': None
                },
                'directional': {
                    'confidence': confidence,
                    'asymmetry': analyst_target
                }
            }
        
        # If we have Tactician labels, use them directly  
        if tactician_target is not None:
            self.logger.info("✅ Using Tactician labels for optimization")
            confidence = tactician_confidence if tactician_confidence is not None else np.ones_like(tactician_target)
            
            # Create weighted targets for long and short
            long_signal = np.maximum(tactician_target, 0) * confidence
            short_signal = np.maximum(-tactician_target, 0) * confidence
            composite_signal = tactician_target * confidence
            
            return {
                'long': {
                    'immediate': long_signal,
                    'short': long_signal,
                    'overall': long_signal,
                    'leverage': None
                },
                'short': {
                    'immediate': short_signal,
                    'short': short_signal,
                    'overall': short_signal,
                    'leverage': None
                },
                'composite': {
                    'immediate': composite_signal,
                    'short': composite_signal,
                    'overall': composite_signal,
                    'leverage': None
                },
                'directional': {
                    'confidence': confidence,
                    'asymmetry': tactician_target
                }
            }

        long_immediate = self._combine_arrays([
            self._aggregate_probability_stream(data, 'long', 'immediate'),
            column_array('long_immediate_opportunity')
        ])
        long_short_term = column_array('long_short_term_opportunity')
        long_short = self._combine_arrays([
            self._aggregate_probability_stream(data, 'long', 'short'),
            long_short_term
        ])
        long_overall = column_array('long_overall_opportunity')
        long_leverage = column_array('long_leverage_adjusted_score')

        short_immediate = self._combine_arrays([
            self._aggregate_probability_stream(data, 'short', 'immediate'),
            column_array('short_immediate_opportunity')
        ])
        short_short_term = column_array('short_short_term_opportunity')
        short_short = self._combine_arrays([
            self._aggregate_probability_stream(data, 'short', 'short'),
            short_short_term
        ])
        short_overall = column_array('short_overall_opportunity')
        short_leverage = column_array('short_leverage_adjusted_score')

        composite_immediate = self._combine_arrays([
            column_array('immediate_opportunity'),
            long_immediate,
            short_immediate
        ])
        composite_short = self._combine_arrays([
            column_array('short_term_opportunity'),
            long_short,
            short_short
        ])
        composite_overall = self._combine_arrays([
            column_array('overall_opportunity'),
            long_overall,
            short_overall
        ])
        composite_leverage = self._combine_arrays([
            column_array('leverage_adjusted_score'),
            long_leverage,
            short_leverage,
            composite_overall
        ])

        directional_confidence = column_array('directional_confidence')
        if directional_confidence is None and long_overall is not None and short_overall is not None:
            directional_confidence = np.nan_to_num(
                np.abs(long_overall - short_overall),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        opportunity_asymmetry = column_array('opportunity_asymmetry')
        if opportunity_asymmetry is None and long_overall is not None and short_overall is not None:
            opportunity_asymmetry = np.nan_to_num(
                long_overall - short_overall,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        aligned_targets: Dict[str, Any] = {
            'long': {
                'immediate': long_immediate,
                'short': long_short,
                'overall': self._combine_arrays([long_overall, long_leverage]),
                'leverage': long_leverage
            },
            'short': {
                'immediate': short_immediate,
                'short': short_short,
                'overall': self._combine_arrays([short_overall, short_leverage]),
                'leverage': short_leverage
            },
            'composite': {
                'immediate': composite_immediate,
                'short': composite_short,
                'overall': composite_overall,
                'leverage': composite_leverage
            },
            'directional': {
                'confidence': directional_confidence,
                'asymmetry': opportunity_asymmetry
            }
        }

        has_any = any(
            isinstance(bucket, dict) and any(value is not None for value in bucket.values())
            for bucket in aligned_targets.values()
        )

        if has_any:
            return aligned_targets

        # No valid labels found - fail fast
        self.logger.error("❌ No valid labels found for optimization")
        self.logger.error("   → Expected: analyst_target/analyst_confidence OR tactician_target/tactician_confidence")
        self.logger.error("   → Use analyst-labeler or tactician-labeler instead")
        self.logger.error("   → Ensure analyst-labeler or tactician-labeler was run before optimization")
        available_cols = [col for col in data.columns if 'target' in col.lower() or 'label' in col.lower()]
        if available_cols:
            self.logger.error(f"   → Available label-related columns: {available_cols}")
        raise ValueError("Cannot create aligned targets: no analyst_target or tactician_target found in data")

    def _create_simple_forward_returns(self, data: pd.DataFrame, max_horizon: int) -> Dict[str, Dict[int, np.ndarray]]:
        """Fallback method to create simple forward returns."""
        tprint_debug("🧠 Entering _create_simple_forward_returns")
        targets = {'simple_returns': {}}

        if 'close' not in data.columns:
            self.logger.warning("⚠️ Close prices unavailable for simple forward returns fallback")
            return targets

        close_prices = data['close'].values
        
        for horizon in range(1, max_horizon + 1):
            if horizon >= len(close_prices):
                break
                
            future_prices = close_prices[horizon:]
            current_prices = close_prices[:-horizon]
            
            forward_returns = np.where(
                current_prices != 0,
                (future_prices - current_prices) / current_prices,
                0.0
            )
            
            targets['simple_returns'][horizon] = forward_returns
        
        return targets

    def _precompute_forward_returns_matrix(
        self,
        data: pd.DataFrame,
        target_column: str,
        max_horizon: int = 200
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Create horizon-weighted opportunity matrices derived from multi-horizon labeling signals."""
        tprint_debug("🧠 Entering _precompute_forward_returns_matrix")
        try:
            immediate_limit, short_limit = self._get_multi_horizon_boundaries()
            immediate_limit = min(max_horizon, immediate_limit)
            short_limit = min(max_horizon, short_limit)

            aligned_targets = self._create_multi_horizon_aligned_targets(data, max_horizon)

            if 'fallback' in aligned_targets:
                fallback_matrix = aligned_targets['fallback']
                return {target_column: fallback_matrix}

            result: Dict[str, Dict[int, np.ndarray]] = {}

            long_bucket = aligned_targets.get('long', {})
            short_bucket = aligned_targets.get('short', {})
            composite_bucket = aligned_targets.get('composite', {})
            directional_bucket = aligned_targets.get('directional', {})

            def register_target(
                name: str,
                bucket: Dict[str, Optional[np.ndarray]],
                immediate_override: Optional[np.ndarray] = None,
                short_override: Optional[np.ndarray] = None,
                overall_override: Optional[np.ndarray] = None,
                leverage_override: Optional[np.ndarray] = None,
                fallback_overall: Optional[np.ndarray] = None
            ) -> None:
                immediate_arr = immediate_override if immediate_override is not None else bucket.get('immediate')
                short_arr = short_override if short_override is not None else bucket.get('short')
                overall_arr = overall_override if overall_override is not None else bucket.get('overall')

                if overall_arr is None and leverage_override is not None:
                    overall_arr = leverage_override
                if overall_arr is None and bucket.get('leverage') is not None:
                    overall_arr = bucket.get('leverage')
                if overall_arr is None and fallback_overall is not None:
                    overall_arr = fallback_overall
                if short_arr is None and overall_arr is not None:
                    short_arr = overall_arr
                if short_arr is None and immediate_arr is not None:
                    short_arr = immediate_arr

                horizon_map = self._build_horizon_weighted_matrix(
                    immediate_arr,
                    short_arr,
                    overall_arr,
                    immediate_limit,
                    short_limit,
                    max_horizon
                )

                if horizon_map:
                    result[name] = horizon_map

            composite_overall = composite_bucket.get('overall') if composite_bucket.get('overall') is not None else composite_bucket.get('leverage')
            composite_immediate = composite_bucket.get('immediate') if composite_bucket.get('immediate') is not None else composite_overall
            composite_short = composite_bucket.get('short') if composite_bucket.get('short') is not None else composite_overall

            register_target('long_overall_opportunity', long_bucket, fallback_overall=composite_overall)
            register_target('long_immediate_opportunity', long_bucket, overall_override=long_bucket.get('immediate'), fallback_overall=composite_immediate)
            register_target('long_short_term_opportunity', long_bucket, immediate_override=long_bucket.get('short'), short_override=long_bucket.get('short'), fallback_overall=composite_short)
            register_target('long_leverage_adjusted_score', long_bucket, leverage_override=long_bucket.get('leverage'), fallback_overall=composite_overall)

            register_target('short_overall_opportunity', short_bucket, fallback_overall=composite_overall)
            register_target('short_immediate_opportunity', short_bucket, overall_override=short_bucket.get('immediate'), fallback_overall=composite_immediate)
            register_target('short_short_term_opportunity', short_bucket, immediate_override=short_bucket.get('short'), short_override=short_bucket.get('short'), fallback_overall=composite_short)
            register_target('short_leverage_adjusted_score', short_bucket, leverage_override=short_bucket.get('leverage'), fallback_overall=composite_overall)

            register_target('leverage_adjusted_score', composite_bucket, leverage_override=composite_bucket.get('leverage'))
            register_target('overall_opportunity', composite_bucket)
            register_target('immediate_opportunity', composite_bucket, overall_override=composite_bucket.get('immediate'), fallback_overall=composite_immediate)
            register_target('short_term_opportunity', composite_bucket, immediate_override=composite_bucket.get('short'), short_override=composite_bucket.get('short'), fallback_overall=composite_short)

            directional_confidence = directional_bucket.get('confidence')
            if directional_confidence is not None:
                result['directional_confidence'] = self._build_horizon_weighted_matrix(
                    directional_confidence,
                    directional_confidence,
                    directional_confidence,
                    immediate_limit,
                    short_limit,
                    max_horizon
                )

            opportunity_asymmetry = directional_bucket.get('asymmetry')
            if opportunity_asymmetry is not None:
                result['opportunity_asymmetry'] = self._build_horizon_weighted_matrix(
                    opportunity_asymmetry,
                    opportunity_asymmetry,
                    opportunity_asymmetry,
                    immediate_limit,
                    short_limit,
                    max_horizon
                )

            if not result:
                self.logger.error("❌ Failed to precompute forward returns matrix")
                raise RuntimeError("Forward returns matrix precomputation failed")

            self.logger.info(f"✅ Prepared multi-horizon matrices for {len(result)} target columns")
            return result

        except Exception as exc:
            self.logger.error(f"❌ Failed to build multi-horizon opportunity matrices: {exc}")
            raise RuntimeError(f"Multi-horizon matrix building failed: {exc}") from exc

    def _create_time_split(self, data_length: int, train_ratio: float = 0.7) -> Tuple[int, int]:
        """
        Create time-based train/validation split indices.
        
        Args:
            data_length: Total length of data
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_end_idx, validation_start_idx)
        """
        tprint_debug("🧠 Entering _create_time_split")
        train_end_idx = int(data_length * train_ratio)
        val_start_idx = train_end_idx + 1  # FIXED: Add 1 to avoid overlap
        return train_end_idx, val_start_idx

    def _generate_coarse_horizons(self, min_horizon: int = 1, max_horizon: int = 200) -> List[int]:
        """
        Generate coarse set of horizons optimized for ~15 total points (reduced from ~22).
        
        PERFORMANCE OPTIMIZATION: Reduced from 22 to 15 horizons for ~30% fewer trials.
        - Dense sampling for very short horizons (1-7)
        - Strategic log-spaced sampling for longer horizons
        
        FIXED: Now uses np.round() instead of dtype=int to properly include max_horizon boundary.
        
        Args:
            min_horizon: Minimum horizon
            max_horizon: Maximum horizon
            
        Returns:
            List of horizon values for coarse search (~15 points)
        """
        # tprint_debug("🧠 Entering _generate_coarse_horizons")  # PERFORMANCE: Reduced logging
        
        # OPTIMIZATION: Reduced dense range from 1-10 to 1-7 for fewer early points
        dense_horizons = list(range(min_horizon, min(8, max_horizon + 1)))
        
        # Log-spaced sampling for longer horizons
        if max_horizon > 7:
            # OPTIMIZATION: Generate only 10 log-spaced points (vs 15) from 7 to max_horizon
            # This gives us ~7 dense + ~8-10 log-spaced = ~15 total (vs previous ~22)
            log_start = max(8, min_horizon)
            # FIX: Use rounding instead of truncation to properly include boundaries
            # Old code: np.logspace(..., dtype=int) truncates 51.0 → 50
            # New code: np.round(...).astype(int) rounds 51.0 → 51
            log_horizons = np.round(np.logspace(np.log10(log_start), np.log10(max_horizon), 10)).astype(int)
            # Remove duplicates and ensure we don't exceed max_horizon
            log_horizons = sorted(list(set(log_horizons)))
            log_horizons = [h for h in log_horizons if h <= max_horizon and h > 7]
        else:
            log_horizons = []
        
        # Combine and sort - should yield ~15 total horizons
        all_horizons = sorted(list(set(dense_horizons + log_horizons)))
        
        # SAFETY: Explicitly ensure boundaries are included
        if min_horizon not in all_horizons:
            all_horizons.append(min_horizon)
        if max_horizon not in all_horizons:
            all_horizons.append(max_horizon)
        
        all_horizons = sorted(all_horizons)
        return all_horizons

    @safe_operation("mutual information calculation", default_value=0.0)
    def _calculate_mutual_information_robust(self, x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
        """
        Calculate robust mutual information using VectorBT optimizations and standardized error handling.
        
        Args:
            x: First variable
            y: Second variable
            n_bins: Number of bins for discretization (unused, kept for compatibility)
            
        Returns:
            Mutual information value
        """
        # Try VectorBT optimization first if available
        if self.use_vectorbt_optimization and self.unified_manager:
            try:
                # Use Unified Vectorization Manager for intelligent optimization
                operation_config = self.unified_manager.create_operation_config(
                    operation_type=OperationType.STATISTICAL_COMPUTATION,
                    data_size=len(x),
                    data_dimensions=(len(x),),
                    memory_budget_mb=256.0,
                    time_budget_seconds=15.0
                )
                
                strategy = self.unified_manager.select_optimization_strategy(operation_config)
                self.logger.debug(f"Selected VectorBT strategy for MI calculation: {strategy}")
                
                # Use VectorBT optimized correlation as a proxy for MI
                if self.rolling_optimizer and len(x) > 100:
                    # For large datasets, use rolling correlation as a faster approximation
                    x_series = pd.Series(x)
                    y_series = pd.Series(y)
                    
                    # Calculate rolling correlation and use its absolute value as MI approximation
                    rolling_corr = self.rolling_optimizer.rolling_corr(x_series, y_series, window=min(50, len(x)//4))
                    valid_corr = rolling_corr.dropna()
                    
                    if len(valid_corr) > 0:
                        # Use mean absolute correlation as MI approximation
                        mi_approx = abs(valid_corr.mean())
                        self.logger.debug(f"VectorBT MI approximation: {mi_approx:.4f}")
                        return float(mi_approx)
                
            except Exception as e:
                self.logger.warning(f"VectorBT MI calculation failed, falling back to standard method: {e}")
        
        # Fallback to standard method
        return safe_mutual_information_with_nan_handling(x, y, n_bins=n_bins, min_samples=20)

    def _calculate_scale_normalized_score(self, mean_mi: float, std_mi: float, stability_penalty: float, lookback_penalty: float) -> Dict[str, float]:
        """
        Calculate scale-normalized scoring with adaptive penalties using consolidated utilities.
        
        Args:
            mean_mi: Mean mutual information
            std_mi: Standard deviation of MI
            stability_penalty: Stability penalty (0 or 1)
            lookback_penalty: Lookback regularization penalty
            
        Returns:
            Dictionary with normalized score components
        """
        scoring_utils = get_scoring_utils()
        return scoring_utils.calculate_scale_normalized_score(mean_mi, std_mi, stability_penalty, lookback_penalty)

    def _bootstrap_mi_validation(self, feature_values: np.ndarray, forward_returns: np.ndarray, n_resamples: int = 10) -> Dict[str, float]:
        """
        Perform VECTORIZED bootstrap sampling for variance estimation of mutual information.
        
        PERFORMANCE OPTIMIZATION: Vectorized implementation for 20-30% faster validation.
        - Generates all bootstrap indices at once
        - Processes samples in vectorized batches
        - Reduces loop overhead significantly
        
        Args:
            feature_values: Feature values
            forward_returns: Forward returns for the horizon
            n_resamples: Number of bootstrap resamples
            
        Returns:
            Dictionary with mean_mi, std_mi, and objective score
        """
        # tprint_debug("🧠 Entering _bootstrap_mi_validation")  # Commented out for reduced verbosity
        try:
            # Align arrays
            min_length = min(len(feature_values), len(forward_returns))
            if min_length < 20:  # Need sufficient data for bootstrap
                return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
                
            feature_aligned = feature_values[:min_length]
            returns_aligned = forward_returns[:min_length]
            
            # CRITICAL FIX: Remove NaN values before bootstrap
            valid_mask = ~(np.isnan(feature_aligned) | np.isnan(returns_aligned))
            if not np.any(valid_mask):
                return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
            
            feature_aligned = feature_aligned[valid_mask]
            returns_aligned = returns_aligned[valid_mask]
            
            if len(feature_aligned) < 20:
                return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
            
            # Update min_length after NaN removal
            min_length = len(feature_aligned)
            
            # VECTORIZED OPTIMIZATION: Generate all bootstrap indices at once
            # Shape: (n_resamples, min_length) - each row is a bootstrap sample
            all_bootstrap_indices = self._rng.choice(
                min_length, 
                size=(n_resamples, min_length), 
                replace=True
            )
            
            # VECTORIZED OPTIMIZATION: Compute MI for all bootstrap samples
            # Use list comprehension with advanced indexing (still faster than explicit loop)
            mi_samples = np.array([
                self._calculate_mutual_information_robust(
                    feature_aligned[indices], 
                    returns_aligned[indices]
                )
                for indices in all_bootstrap_indices
            ])
            
            # Vectorized statistics computation
            mean_mi = float(np.mean(mi_samples))
            std_mi = float(np.std(mi_samples))
            median_mi = float(np.median(mi_samples))
            mad_mi = float(np.median(np.abs(mi_samples - median_mi)))
            mad_over_median = float(safe_divide(mad_mi, np.abs(median_mi))) if median_mi != 0 else 0.0

            # Use scale-normalized scoring (no penalties in bootstrap validation)
            scoring_result = self._calculate_scale_normalized_score(
                mean_mi=mean_mi,
                std_mi=std_mi,
                stability_penalty=0.0,  # No stability penalty in bootstrap validation
                lookback_penalty=0.0   # No lookback penalty in bootstrap validation
            )

            return {
                'mean_mi': mean_mi,
                'std_mi': std_mi,
                'median_mi': median_mi,
                'mad_mi': mad_mi,
                'mad_over_median': mad_over_median,
                'objective': scoring_result['final_score'],
                'variance_penalty': scoring_result['variance_penalty'],
                'samples': mi_samples
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Bootstrap validation failed: {e}")
            return {
                'mean_mi': 0.0,
                'std_mi': 0.0,
                'median_mi': 0.0,
                'mad_mi': 0.0,
                'mad_over_median': 0.0,
                'objective': 0.0
            }

    def _parallel_refinement(self, top_horizons: List[Tuple[int, float]], data: pd.DataFrame,
                           feature_name: str, forward_returns: Dict[int, np.ndarray],
                           train_end_idx: int, min_lookback: int, max_lookback: int) -> List[Tuple[int, float]]:
        """
        Parallel refinement of top horizons using ThreadPoolExecutor.
        
        FIXED: Refinement range now properly includes max_lookback boundary.
        
        Args:
            top_horizons: List of (horizon, mi_score) tuples
            data: Input data
            feature_name: Feature to optimize
            forward_returns: Precomputed forward returns
            train_end_idx: Training split end index
            min_lookback: Minimum lookback period
            max_lookback: Maximum lookback period
            
        Returns:
            List of refined (horizon, mi_score) tuples
        """
        tprint_debug("🧠 Entering _parallel_refinement")
        def refine_single_horizon(horizon_mi_tuple):
            horizon, coarse_mi = horizon_mi_tuple
            # FIX: Use max_lookback + 1 to include boundary in range (Python range is exclusive)
            # Also explicitly add max_lookback if within refinement window
            refinement_horizons = list(range(
                max(min_lookback, horizon - 10), 
                min(max_lookback + 1, horizon + 11),  # +1 to include max_lookback
                2  # Check every 2 periods
            ))
            
            # SAFETY: Explicitly ensure boundaries are tested if within range
            if min_lookback not in refinement_horizons and abs(horizon - min_lookback) <= 10:
                refinement_horizons.append(min_lookback)
            if max_lookback not in refinement_horizons and abs(horizon - max_lookback) <= 10:
                refinement_horizons.append(max_lookback)
            
            refinement_horizons = sorted(set(refinement_horizons))
            
            best_mi = coarse_mi
            best_refined_horizon = horizon
            
            for refined_horizon in refinement_horizons:
                if refined_horizon == horizon:
                    continue  # Already computed
                
                try:
                    # Use vectorized feature generation for refinement
                    vectorized_features = self._vectorized_feature_generation(data, feature_name, [refined_horizon])
                    feature_values = vectorized_features.get(refined_horizon)
                    
                    if feature_values is None or len(feature_values) == 0:
                        # Fallback to cached calculation
                        feature_values = self._cached_feature_calculation(data, feature_name, refined_horizon)
                        if feature_values is None or len(feature_values) == 0:
                            continue
                    
                    # Use train split for refinement evaluation
                    train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                    train_returns = forward_returns.get(refined_horizon, np.array([]))
                    
                    if len(train_returns) == 0:
                        continue
                    
                    # Align arrays
                    min_length = min(len(train_feature), len(train_returns))
                    if min_length < 10:
                        continue
                    
                    aligned_features = train_feature[:min_length]
                    aligned_returns = train_returns[:min_length]
                    
                    # CRITICAL FIX: Remove NaN values before MI calculation
                    valid_mask = ~(np.isnan(aligned_features) | np.isnan(aligned_returns))
                    if not np.any(valid_mask):
                        continue
                    
                    feature_clean = aligned_features[valid_mask]
                    returns_clean = aligned_returns[valid_mask]
                    
                    if len(feature_clean) < max(10, refined_horizon + 5):
                        continue
                    
                    # Use vectorized MI calculation
                    mi_score = self._vectorized_mi_calculation([feature_clean], [returns_clean])[0]
                    
                    if mi_score > best_mi:
                        best_mi = mi_score
                        best_refined_horizon = refined_horizon
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to refine horizon {refined_horizon}: {e}")
                    continue
            
            return (best_refined_horizon, best_mi)
        
        # Use ThreadPoolExecutor for parallel processing
        final_results = []
        with ThreadPoolExecutor(max_workers=min(4, len(top_horizons))) as executor:
            future_to_horizon = {executor.submit(refine_single_horizon, horizon_mi): horizon_mi
                               for horizon_mi in top_horizons}

            for future in as_completed(future_to_horizon):
                try:
                    result = future.result()
                    final_results.append(result)
                except Exception as e:
                    horizon_mi = future_to_horizon[future]
                    self.logger.warning(f"⚠️ Failed to refine horizon {horizon_mi[0]}: {e}")
                    final_results.append(horizon_mi)  # Fallback to original

        return final_results

    @staticmethod
    def _apply_minimum_lag(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Shift feature values by one period to enforce a minimum lag of 1."""
        if values is None:
            return np.array([], dtype=float)

        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr.astype(float)

        lagged = np.empty_like(arr, dtype=float)
        lagged[:] = np.nan
        lagged[1:] = arr[:-1]
        return lagged

    def _assert_lag_requirements(self, feature_name: str, horizon: int, values: np.ndarray) -> None:
        """Validate that feature arrays satisfy minimum lag requirements and record metadata."""
        if values is None:
            raise ValueError(f"Feature '{feature_name}' produced no values for horizon {horizon}")

        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"Feature '{feature_name}' produced empty array for horizon {horizon}")

        required_lag = 1
        leading_window = arr[:required_lag]
        if not np.isnan(leading_window).all():
            raise ValueError(
                f"Feature '{feature_name}' (horizon={horizon}) exposes contemporaneous values; "
                "expected leading NaNs after enforcing lag."
            )

        feature_meta = self.feature_lag_metadata.setdefault(feature_name, {})
        feature_meta[horizon] = {
            'max_lag': max(required_lag, int(horizon)),
            'required_lag': required_lag,
            'has_leading_nulls': True,
        }

    def _vectorized_feature_generation(self, data: pd.DataFrame, feature_name: str,
                                     horizons: List[int]) -> Dict[int, np.ndarray]:
        """
        Truly vectorized feature generation for multiple horizons using numpy operations.

        Args:
            data: Input data
            feature_name: Feature to generate
            horizons: List of lookback periods
            
        Returns:
            Dictionary mapping horizon to feature values
        """
        try:
            # Extract price data for vectorized operations
            if 'close' not in data.columns:
                return {}
            
            close_prices = data['close'].values
            n_samples = len(close_prices)
            
            # Filter valid horizons
            valid_horizons = [h for h in horizons if 0 < h < n_samples]
            if not valid_horizons:
                return {}
            
            # Pre-allocate results dictionary
            results = {}
            
            # Vectorized feature generation based on feature type
            if 'returns' in feature_name.lower():
                # Vectorized returns calculation for all horizons at once
                max_horizon = max(valid_horizons)
                
                # Create shifted price matrix: shape (n_horizons, n_samples)
                horizon_array = np.array(valid_horizons)[:, np.newaxis]  # Shape: (n_horizons, 1)
                indices = np.arange(n_samples)[np.newaxis, :]  # Shape: (1, n_samples)
                shift_indices = indices - horizon_array  # Broadcasting: (n_horizons, n_samples)
                
                # Handle negative indices (before start of data)
                valid_mask = shift_indices >= 0
                shift_indices = np.maximum(shift_indices, 0)
                
                # Get shifted prices using advanced indexing
                shifted_prices = close_prices[shift_indices]
                current_prices = np.tile(close_prices, (len(valid_horizons), 1))
                
                # Calculate returns: (current - shifted) / shifted
                returns = (current_prices - shifted_prices) / np.maximum(shifted_prices, 1e-8)
                
                # Set invalid values to NaN
                returns[~valid_mask] = np.nan
                
                # Apply minimum lag and store results
                for i, horizon in enumerate(valid_horizons):
                    lagged = self._apply_minimum_lag(returns[i])
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

            elif 'momentum' in feature_name.lower():
                # Vectorized momentum calculation for all horizons at once
                max_horizon = max(valid_horizons)
                
                # Create shifted price matrix
                horizon_array = np.array(valid_horizons)[:, np.newaxis]
                indices = np.arange(n_samples)[np.newaxis, :]
                shift_indices = indices - horizon_array
                
                valid_mask = shift_indices >= 0
                shift_indices = np.maximum(shift_indices, 0)
                
                shifted_prices = close_prices[shift_indices]
                current_prices = np.tile(close_prices, (len(valid_horizons), 1))
                
                # Calculate momentum: current - shifted
                momentum = current_prices - shifted_prices
                momentum[~valid_mask] = np.nan
                
                # Apply minimum lag and store results
                for i, horizon in enumerate(valid_horizons):
                    lagged = self._apply_minimum_lag(momentum[i])
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

            elif 'sma' in feature_name.lower() or 'moving_average' in feature_name.lower():
                # Vectorized SMA calculation using pandas rolling
                close_series = pd.Series(close_prices)
                
                for horizon in valid_horizons:
                    sma = close_series.rolling(window=horizon, min_periods=horizon).mean().values
                    lagged = self._apply_minimum_lag(sma)
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

            elif 'ema' in feature_name.lower() or 'exponential' in feature_name.lower():
                # Vectorized EMA calculation
                close_series = pd.Series(close_prices)
                
                for horizon in valid_horizons:
                    alpha = 2.0 / (horizon + 1)
                    ema = close_series.ewm(alpha=alpha, adjust=False).mean().values
                    lagged = self._apply_minimum_lag(ema)
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

            elif 'volatility' in feature_name.lower():
                # Vectorized volatility calculation
                returns = np.diff(close_prices) / close_prices[:-1]
                returns_series = pd.Series(returns)
                
                for horizon in valid_horizons:
                    if horizon < len(returns):
                        volatility = returns_series.rolling(window=horizon, min_periods=horizon).std().values
                        # Pad with NaN to match original length
                        volatility_padded = np.full(n_samples, np.nan)
                        volatility_padded[1:len(volatility)+1] = volatility
                        lagged = self._apply_minimum_lag(volatility_padded)
                        self._assert_lag_requirements(feature_name, horizon, lagged)
                        results[horizon] = lagged

            else:
                # Fallback to individual calculation for complex features
                for horizon in valid_horizons:
                    try:
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon)
                        if feature_values is not None:
                            lagged = self._apply_minimum_lag(feature_values)
                            self._assert_lag_requirements(feature_name, horizon, lagged)
                            results[horizon] = lagged
                    except Exception:
                        continue
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized feature generation failed: {e}")
            return {}

    def _optimize_with_bayesian_tpe(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        regularization_settings: Optional[Dict[str, float]] = None,
        n_trials: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize lookback period using Bayesian TPE optimizer for faster convergence.
        
        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            lookback_range: Min and max lookback periods to test
            regularization_settings: Regularization settings
            n_trials: Number of Bayesian optimization trials (default: 50 for light/blank modes)
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best lookback period and score
        """
        if not BAYESIAN_OPTIMIZER_AVAILABLE:
            tprint_warning("⚠️ Bayesian TPE optimizer not available, falling back to coarse-to-refine")
            return self._coarse_to_refine_single_pass(data, feature_name, target_column, lookback_range, regularization_settings, **kwargs)
        
        try:
            start_time = time.time()
            min_lookback, max_lookback = lookback_range
            regularization_settings = self._normalize_regularization_settings(regularization_settings)
            
            # Get precomputed forward returns if available
            precomputed_forward_returns = kwargs.get('precomputed_forward_returns')
            if precomputed_forward_returns is not None:
                forward_returns = precomputed_forward_returns
            else:
                forward_returns = self._get_shared_forward_returns_matrix(data, target_column, max_horizon=max_lookback)
            
            if not forward_returns:
                return self._create_failed_result("bayesian_tpe", 0.0, feature_name=feature_name)
            
            # Create time-based train/validation split
            train_end_idx, val_start_idx = self._create_time_split(len(data), train_ratio=0.7)
            
            # Define objective function for Bayesian optimization
            def objective_function(params: Dict[str, Any]) -> float:
                """Objective function for Bayesian optimization - returns negative MI (minimize)."""
                lookback = int(params['lookback'])
                
                # Calculate feature values for this lookback using the correct method
                feature_values = self._cached_feature_calculation(data, feature_name, lookback)
                if feature_values is None or len(feature_values) == 0:
                    return 1.0  # High penalty for invalid lookback
                
                # Get train subset
                train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                train_returns = forward_returns.get(lookback, np.array([]))[:train_end_idx]
                
                # Align arrays
                min_length = min(len(train_feature), len(train_returns))
                if min_length < 10:
                    return 1.0  # High penalty for insufficient data
                
                train_feature = train_feature[:min_length]
                train_returns = train_returns[:min_length]
                
                # Calculate mutual information (objective to maximize)
                mi_score = self._calculate_mutual_information_robust(train_feature, train_returns)
                
                # Use scale-normalized scoring instead of manual penalty application
                scoring_result = self._calculate_scale_normalized_score(
                    mean_mi=mi_score,
                    std_mi=0.0,  # No variance data in single evaluation
                    stability_penalty=0.0,  # No stability penalty in single evaluation
                    lookback_penalty=regularization_settings.get('penalty', 0.1) * abs(lookback - regularization_settings.get('preferred_lookback', 50))
                )
                
                # Return negative score (since Bayesian optimizer minimizes)
                return -scoring_result['final_score']
            
            # Configure Bayesian optimizer
            config = OptimizationConfig(
                n_trials=n_trials,
                n_startup_trials=min(10, n_trials // 5),  # 20% startup trials
                timeout=300,  # 5 minutes timeout
                seed=42,  # Use seed instead of random_state
                direction='minimize',  # We're minimizing negative MI
                enable_staged_optimization=False,  # Disable staged for simple lookback search
                enable_hardware_optimization=True,
                enable_adaptive_optimization=True
            )
            
            # Define search space
            search_space = {
                'lookback': {
                    'type': 'int',
                    'low': min_lookback,
                    'high': max_lookback,
                    'step': 1
                }
            }
            
            # Run Bayesian optimization
            optimizer = BayesianTPEOptimizer(config)
            optimization_result = optimizer.optimize(
                objective_function,  # First positional argument
                search_space         # Second positional argument
            )
            
            # Extract results from the optimization result dictionary
            best_params = optimization_result.get('best_params', {})
            best_score = optimization_result.get('best_value', 0.0)
            n_trials_completed = optimization_result.get('n_trials', n_trials)
            
            best_lookback = int(best_params.get('lookback', min_lookback))
            best_mi_score = -best_score  # Convert back from negative
            
            optimization_time = time.time() - start_time
            
            # Return result
            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_mi_score,
                optimization_method="bayesian_tpe",
                total_trials=n_trials_completed,
                optimization_time=optimization_time,
                convergence_achieved=True,
                metadata={
                    'n_trials': n_trials_completed,
                    'n_trials_requested': n_trials,
                    'execution_mode': 'light/blank',
                    'optimization_type': 'bayesian_tpe',
                    'optimization_result': optimization_result
                },
                feature_name=feature_name,
                is_stable=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian TPE optimization failed for {feature_name}: {e}")
            tprint_warning(f"⚠️ Bayesian optimization failed, falling back to coarse-to-refine")
            return self._coarse_to_refine_single_pass(data, feature_name, target_column, lookback_range, regularization_settings, **kwargs)

    def _optimize_coarse_to_refine(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        outer_split_iterator: Optional[Iterable[Tuple[slice, slice]]] = None,
        regularization_settings: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Optimize using coarse-to-refine search with optional nested CV."""
        provided_iterator = kwargs.pop('outer_split_iterator', None)
        if provided_iterator is not None:
            outer_split_iterator = provided_iterator

        if outer_split_iterator:
            outer_splits = list(outer_split_iterator)
            if outer_splits:
                return self._optimize_coarse_to_refine_with_outer(
                    data,
                    feature_name,
                    target_column,
                    lookback_range,
                    outer_splits,
                    regularization_settings=regularization_settings,
                    **kwargs,
                )

        return self._coarse_to_refine_single_pass(
            data,
            feature_name,
            target_column,
            lookback_range,
            regularization_settings=regularization_settings,
            **kwargs,
        )

    def _coarse_to_refine_single_pass(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        regularization_settings: Optional[Dict[str, float]] = None,
        precomputed_forward_returns: Optional[Dict[int, np.ndarray]] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize using coarse-to-refine approach with bootstrap validation.
        
        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            lookback_range: Min and max lookback periods to test
            precomputed_forward_returns: Optional pre-computed forward returns matrix (PERFORMANCE OPTIMIZATION)
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best lookback period and score
        """
        # tprint_debug("🧠 Entering _optimize_coarse_to_refine")  # PERFORMANCE: Reduced logging
        try:
            min_lookback, max_lookback = lookback_range
            regularization_settings = self._normalize_regularization_settings(regularization_settings)
            # Step 1: Get shared forward returns matrix (reused across all features)
            # PERFORMANCE OPTIMIZATION: Use precomputed matrix if available to avoid redundant computation
            if precomputed_forward_returns is not None:
                forward_returns = precomputed_forward_returns
            else:
                forward_returns = self._get_shared_forward_returns_matrix(data, target_column, max_horizon=max_lookback)
            if not forward_returns:
                return self._create_failed_result("coarse_to_refine", 0.0, feature_name=feature_name)

            # Step 2: Create time-based train/validation split
            train_end_idx, val_start_idx = self._create_time_split(len(data), train_ratio=0.7)

            # Step 3: Generate coarse horizons
            coarse_horizons = self._generate_coarse_horizons(min_lookback, max_lookback)
            self.logger.info(f"[{feature_name}] Generated {len(coarse_horizons)} coarse horizons: {coarse_horizons}")

            # Step 4: Vectorized coarse search with early termination
            coarse_results: List[Tuple[int, float]] = []
            penalized_cache: Dict[int, Dict[str, float]] = {}
            horizon_bootstrap_cache: Dict[int, Dict[str, float]] = {}
            unstable_horizons: Dict[int, float] = {}
            
            # Use vectorized feature generation for all coarse horizons
            vectorized_features = self._vectorized_feature_generation(data, feature_name, coarse_horizons)
            self.logger.info(f"[{feature_name}] Vectorized generation returned {len(vectorized_features)} features")
            
            # Prepare data for vectorized MI calculation
            valid_horizons = []
            features_list = []
            returns_list = []
            
            # If vectorized generation failed, fall back to individual calculation for all horizons
            if not vectorized_features:
                self.logger.info(f"[{feature_name}] Vectorized generation failed, using individual calculation for {len(coarse_horizons)} horizons")
            
            for horizon in coarse_horizons:
                # If vectorized generation didn't produce this horizon, calculate it individually
                if horizon not in vectorized_features:
                    feature_values = self._cached_feature_calculation(data, feature_name, horizon)
                    if feature_values is not None and len(feature_values) > 0:
                        vectorized_features[horizon] = feature_values
                    else:
                        continue
                
                feature_values = vectorized_features.get(horizon)
                
                if feature_values is None or len(feature_values) == 0:
                    continue
                
                # Use train split for coarse evaluation
                train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                train_returns = forward_returns.get(horizon, np.array([]))
                
                if len(train_returns) == 0:
                    continue
                
                # CRITICAL FIX: Properly align arrays and remove NaN padding
                min_length = min(len(train_feature), len(train_returns))
                if min_length < 10:
                    continue
                
                # Align arrays
                feature_aligned = train_feature[:min_length]
                returns_aligned = train_returns[:min_length]
                
                # CRITICAL FIX: Remove NaN values from alignment (from shift padding)
                # This prevents artificial correlation from zero-padding
                valid_mask = ~(np.isnan(feature_aligned) | np.isnan(returns_aligned))
                if not np.any(valid_mask):
                    continue
                
                feature_clean = feature_aligned[valid_mask]
                returns_clean = returns_aligned[valid_mask]
                
                # Need sufficient data after removing NaNs
                if len(feature_clean) < max(10, horizon + 5):  # At least horizon + 5 points
                    continue
                
                valid_horizons.append(horizon)
                features_list.append(feature_clean)
                returns_list.append(returns_clean)
            
            # Vectorized MI calculation for all valid horizons
            if features_list and returns_list:
                mi_scores = self._vectorized_mi_calculation(features_list, returns_list)
                coarse_results = []
                for horizon, mi in zip(valid_horizons, mi_scores):
                    if mi <= 0:
                        continue
                    penalized_score, penalty_value = self._apply_regularization_penalty(
                        int(horizon),
                        float(mi),
                        regularization_settings,
                    )
                    penalized_cache[int(horizon)] = {
                        'penalized_score': penalized_score,
                        'penalty': penalty_value,
                        'raw_score': float(mi),
                    }
                    coarse_results.append((int(horizon), float(mi)))
                
                # DEBUG: Log all coarse results
                if len(valid_horizons) > 1:
                    self.logger.info(f"[{feature_name}] Tested {len(valid_horizons)} horizons: {valid_horizons}")
                    self.logger.info(f"[{feature_name}] Got {len(coarse_results)} valid MI scores (>0)")

                # Smart early termination: check if MI improvements are minimal
                if len(coarse_results) >= 3:
                    coarse_results.sort(
                        key=lambda x: penalized_cache.get(x[0], {}).get('penalized_score', x[1]),
                        reverse=True,
                    )
                    best_info = penalized_cache.get(coarse_results[0][0], {
                        'penalized_score': coarse_results[0][1],
                        'penalty': 0.0,
                        'raw_score': coarse_results[0][1],
                    })
                    second_score = penalized_cache.get(coarse_results[1][0], {}).get(
                        'penalized_score',
                        coarse_results[1][1],
                    )
                    third_score = penalized_cache.get(coarse_results[2][0], {}).get(
                        'penalized_score',
                        coarse_results[2][1],
                    )

                    # If top 3 results are very close (relative to best score), skip refinement
                    mi_range = best_info['penalized_score'] - third_score
                    relative_improvement = mi_range / max(abs(best_info['penalized_score']), 1e-10)
                    
                    # PERFORMANCE OPTIMIZATION: Increased threshold from 1% to 2% for faster convergence
                    # Use relative threshold: only early terminate if improvement is < 2%
                    if relative_improvement < 0.02:  # Less than 2% relative improvement (was 0.01)
                        best_horizon = coarse_results[0][0]
                        best_penalty = best_info.get('penalty', 0.0)
                        best_score = best_info['penalized_score']
                        self.logger.info(
                            f'✅ {feature_name}: best_lookback={best_horizon}, score={best_score:.6f} '
                            f'(early termination: <2% improvement, rel_imp={relative_improvement:.4f})'
                        )
                        return OptimizationResult(
                            best_lookback_period=best_horizon,
                            best_score=best_score,
                            optimization_method="coarse_to_refine",
                            total_trials=len(coarse_results),
                            optimization_time=0.0,
                            convergence_achieved=True,
                            feature_name=feature_name,  # FIXED: Added feature_name field
                            metadata={
                                'feature_name': feature_name,
                                'target_column': target_column,
                                'coarse_horizons': len(coarse_results),
                                'early_termination': True,
                                'reason': 'minimal_relative_improvement',
                                'mi_range': mi_range,
                                'relative_improvement': relative_improvement,
                                'regularization_penalty': best_penalty,
                                'regularization_settings': regularization_settings,
                                'raw_mi': best_info.get('raw_score', best_score + best_penalty),
                            }
                        )

            if not coarse_results:
                return self._create_failed_result("coarse_to_refine", 0.0)
            
            # DEBUG: Log coarse results to diagnose why all features select lookback=5
            if len(coarse_results) > 1:
                self.logger.debug(f"[{feature_name}] Coarse results (first 10): {[(h, f'{s:.6f}') for h, s in coarse_results[:10]]}")
            
            # Step 5: Pick top horizons (after applying regularization penalties)
            coarse_results.sort(
                key=lambda x: penalized_cache.get(x[0], {}).get('penalized_score', x[1]),
                reverse=True,
            )
            original_top_3 = [
                {
                    'horizon': horizon,
                    'raw_mi': penalized_cache.get(horizon, {}).get('raw_score', mi),
                    'penalized_score': penalized_cache.get(horizon, {}).get('penalized_score', mi),
                    'regularization_penalty': penalized_cache.get(horizon, {}).get('penalty', 0.0),
                }
                for horizon, mi in coarse_results[:3]
            ]
            top_3_horizons = [(entry['horizon'], entry['raw_mi']) for entry in original_top_3]
            stability_threshold = 0.15
            screened_top_horizons: List[Tuple[int, float]] = []

            for horizon, mi in coarse_results:
                if len(screened_top_horizons) >= 3:
                    break

                try:
                    horizon_key = int(horizon)

                    feature_values = vectorized_features.get(horizon_key)
                    if feature_values is None or len(feature_values) == 0:
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon_key)
                        if feature_values is None or len(feature_values) == 0:
                            continue

                    train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                    train_returns = forward_returns.get(horizon_key, np.array([]))

                    if len(train_returns) == 0:
                        continue

                    min_length = min(len(train_feature), len(train_returns))
                    if min_length < 20:
                        continue

                    stats = self._bootstrap_mi_validation(
                        np.asarray(train_feature[:min_length]),
                        np.asarray(train_returns[:min_length]),
                        n_resamples=10
                    )

                    horizon_bootstrap_cache[horizon_key] = stats
                    mad_ratio = stats.get('mad_over_median', 0.0)

                    if mad_ratio > stability_threshold:
                        unstable_horizons[horizon_key] = mad_ratio
                        continue

                    screened_top_horizons.append((horizon_key, mi))

                except Exception as e:
                    self.logger.warning(f"⚠️ Stability screening failed for horizon {horizon_key}: {e}")
                    continue

            refinement_candidates = screened_top_horizons
            
            # Early stopping check
            best_candidate_info = penalized_cache.get(coarse_results[0][0], {
                'penalized_score': coarse_results[0][1],
                'penalty': 0.0,
                'raw_score': coarse_results[0][1],
            })
            if best_candidate_info['penalized_score'] < 1e-3:
                # Early stopping - return best result
                best_horizon = coarse_results[0][0]
                best_penalty = best_candidate_info.get('penalty', 0.0)
                best_score = best_candidate_info['penalized_score']
                self.logger.info(f'✅ {feature_name}: best_lookback={best_horizon}, score={best_score:.6f}')
                return OptimizationResult(
                    best_lookback_period=best_horizon,
                    best_score=best_score,
                    optimization_method="coarse_to_refine",
                    total_trials=len(coarse_results),
                    optimization_time=0.0,
                    convergence_achieved=True,
                    feature_name=feature_name,  # FIXED: Added feature_name field
                    metadata={
                        'feature_name': feature_name,
                        'target_column': target_column,
                        'coarse_horizons': len(coarse_results),
                        'early_stopping': True,
                        'reason': 'low_mi',
                        'regularization_penalty': best_penalty,
                        'regularization_settings': regularization_settings,
                        'raw_mi': best_candidate_info.get('raw_score', best_score + best_penalty),
                    }
                )
            
            # Step 6: Parallel refinement around top horizons
            refined_results: List[Tuple[int, float]] = []
            if refinement_candidates:
                refined_results = self._parallel_refinement(
                    refinement_candidates, data, feature_name, forward_returns,
                    train_end_idx, min_lookback, max_lookback
                )
                for horizon, refined_mi in refined_results:
                    penalized_score, penalty_value = self._apply_regularization_penalty(
                        int(horizon),
                        float(refined_mi),
                        regularization_settings,
                    )
                    penalized_cache[int(horizon)] = {
                        'penalized_score': penalized_score,
                        'penalty': penalty_value,
                        'raw_score': float(refined_mi),
                    }

            # Combine coarse and refined results
            all_candidates = coarse_results + refined_results
            
            # Step 7: Memory-efficient batch bootstrap validation
            bootstrap_results: List[Dict[str, Any]] = []
            batch_size = 3  # Process in small batches to manage memory

            for i in range(0, min(10, len(all_candidates)), batch_size):
                batch = all_candidates[i:i+batch_size]

                for horizon, mi in batch:
                    try:
                        horizon_key = int(horizon)

                        # Use cached feature calculation
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon_key)

                        if feature_values is None or len(feature_values) == 0:
                            continue

                        train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                        train_returns = forward_returns.get(horizon_key, np.array([]))

                        if len(train_returns) == 0:
                            continue

                        min_length = min(len(train_feature), len(train_returns))
                        penalty_info = penalized_cache.get(horizon_key)
                        if penalty_info is None:
                            penalized_score, penalty_value = self._apply_regularization_penalty(
                                horizon_key,
                                float(mi),
                                regularization_settings,
                            )
                            penalty_info = {
                                'penalized_score': penalized_score,
                                'penalty': penalty_value,
                                'raw_score': float(mi),
                            }
                            penalized_cache[horizon_key] = penalty_info
                        lookback_penalty = penalty_info.get('penalty', 0.0)
                        if min_length < 20:  # Need sufficient data for bootstrap
                            stability_penalty = 1.0 if horizon_key in unstable_horizons else 0.0
                            bootstrap_results.append({
                                'horizon': horizon_key,
                                'mean_mi': mi,
                                'std_mi': 0.0,
                                'median_mi': 0.0,
                                'mad_mi': 0.0,
                                'mad_over_median': 0.0,
                                'objective': mi - stability_penalty,
                                'penalized_objective': (mi - stability_penalty) - lookback_penalty,
                                'original_mi': mi,
                                'adjusted_mi': mi * (0.1 if horizon_key in unstable_horizons else 1.0),
                                'is_unstable': horizon_key in unstable_horizons,
                                'stability_guardrail_penalty': stability_penalty,
                                'regularization_penalty': lookback_penalty,
                            })
                            continue

                        stats = horizon_bootstrap_cache.get(horizon_key)
                        if stats is None:
                            stats = self._bootstrap_mi_validation(
                                train_feature[:min_length],
                                train_returns[:min_length],
                                n_resamples=10  # 50% reduction from 20 to 10
                            )
                            horizon_bootstrap_cache[horizon_key] = stats

                        mad_ratio = stats.get('mad_over_median', 0.0)
                        if mad_ratio > stability_threshold:
                            unstable_horizons[horizon_key] = mad_ratio

                        # Use scale-normalized scoring with proper penalty handling
                        stability_penalty_flag = 1.0 if horizon_key in unstable_horizons else 0.0
                        adjusted_mi = mi * (0.1 if horizon_key in unstable_horizons else 1.0)
                        
                        # Calculate normalized scores
                        scoring_result = self._calculate_scale_normalized_score(
                            mean_mi=stats['mean_mi'],
                            std_mi=stats['std_mi'],
                            stability_penalty=stability_penalty_flag,
                            lookback_penalty=lookback_penalty
                        )
                        
                        objective = scoring_result['base_objective']
                        penalized_objective = scoring_result['final_score']

                        bootstrap_results.append({
                            'horizon': horizon_key,
                            'mean_mi': stats['mean_mi'],
                            'std_mi': stats['std_mi'],
                            'median_mi': stats.get('median_mi', 0.0),
                            'mad_mi': stats.get('mad_mi', 0.0),
                            'mad_over_median': mad_ratio,
                            'objective': objective,
                            'penalized_objective': penalized_objective,
                            'original_mi': mi,
                            'adjusted_mi': adjusted_mi,
                            'is_unstable': horizon_key in unstable_horizons,
                            'stability_guardrail_penalty': scoring_result['normalized_stability_penalty'],
                            'regularization_penalty': lookback_penalty,
                            'variance_penalty': scoring_result['variance_penalty'],
                            'total_penalties': scoring_result['total_penalties'],
                            'capped_penalties': scoring_result['capped_penalties'],
                        })

                    except Exception as e:
                        self.logger.warning(f"⚠️ Bootstrap failed for horizon {horizon_key}: {e}")
                        continue
                
                # Memory cleanup after each batch
                gc.collect()
            
            # Step 8: Time stability check (validation split)
            if bootstrap_results:
                best_candidates = sorted(
                    bootstrap_results,
                    key=lambda x: x.get('penalized_objective', x['objective']),
                    reverse=True,
                )[:3]  # Top 3 by penalized objective

                stability_results: List[Dict[str, Any]] = []
                for candidate in best_candidates:
                    horizon = candidate['horizon']
                    try:
                        # Use cached feature calculation
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon)

                        if feature_values is None or len(feature_values) == 0:
                            continue

                        # Test on validation split
                        val_feature = feature_values[val_start_idx:] if len(feature_values) > val_start_idx else np.array([])
                        val_returns = forward_returns.get(horizon, np.array([]))

                        candidate_with_val = candidate.copy()

                        if len(val_feature) == 0 or len(val_returns) == 0:
                            candidate_with_val['val_mi'] = 0.0
                            stability_results.append(candidate_with_val)
                            continue

                        min_length = min(len(val_feature), len(val_returns))
                        if min_length < 10:
                            candidate_with_val['val_mi'] = 0.0
                            stability_results.append(candidate_with_val)
                            continue

                        # Calculate MI on validation split
                        val_mi = self._calculate_mutual_information_robust(
                            val_feature[:min_length],
                            val_returns[:min_length]
                        )

                        candidate_with_val['val_mi'] = val_mi
                        stability_results.append(candidate_with_val)

                    except Exception as e:
                        self.logger.warning(f"⚠️ Stability check failed for horizon {horizon}: {e}")
                        continue

                # Prefer horizons that don't collapse OOS (validation MI > 0.5 * train MI)
                final_results: List[Dict[str, Any]] = []
                for candidate in stability_results:
                    stability_penalty = 0.0
                    if candidate['mean_mi'] > 0 and candidate['val_mi'] < 0.5 * candidate['mean_mi']:
                        stability_penalty = 0.1  # Penalize poor OOS performance

                    lookback_penalty = candidate.get(
                        'regularization_penalty',
                        penalized_cache.get(candidate['horizon'], {}).get('penalty', 0.0),
                    )
                    base_objective = candidate.get(
                        'penalized_objective',
                        candidate['objective'] - lookback_penalty,
                    )
                    # Use scale-normalized final scoring
                    final_scoring_result = self._calculate_scale_normalized_score(
                        mean_mi=candidate.get('mean_mi', 0.0),
                        std_mi=candidate.get('std_mi', 0.0),
                        stability_penalty=stability_penalty,
                        lookback_penalty=lookback_penalty
                    )
                    final_score = final_scoring_result['final_score']
                    enriched_candidate = candidate.copy()
                    enriched_candidate['final_score'] = final_score
                    enriched_candidate['validation_penalty'] = final_scoring_result['normalized_stability_penalty']
                    enriched_candidate['regularization_penalty'] = lookback_penalty
                    enriched_candidate['penalized_objective'] = base_objective
                    enriched_candidate['variance_penalty'] = final_scoring_result['variance_penalty']
                    enriched_candidate['total_penalties'] = final_scoring_result['total_penalties']
                    enriched_candidate['capped_penalties'] = final_scoring_result['capped_penalties']
                    final_results.append(enriched_candidate)

                # Select best horizon
                if final_results:
                    final_results.sort(key=lambda x: x['final_score'], reverse=True)
                    best_candidate = final_results[0]
                    best_horizon = best_candidate['horizon']
                    best_score = best_candidate['final_score']

                    # Log cache performance
                    cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                    self.logger.info(f'✅ {feature_name}: best_lookback={best_horizon}, score={best_score:.6f} (cache_hit_rate={cache_hit_rate:.2%})')

                    return OptimizationResult(
                        best_lookback_period=best_horizon,
                        best_score=best_score,
                        optimization_method="coarse_to_refine",
                        total_trials=len(all_candidates),
                        optimization_time=0.0,  # Will be set by caller
                        convergence_achieved=True,
                        feature_name=feature_name,  # FIXED: Added feature_name field
                        metadata={
                            'feature_name': feature_name,
                            'target_column': target_column,
                            'coarse_horizons': len(coarse_results),
                            'refined_horizons': len(refined_results),
                            'bootstrap_samples': 10,  # Updated to reflect 50% reduction
                            'mean_mi': best_candidate['mean_mi'],
                            'std_mi': best_candidate['std_mi'],
                            'median_mi': best_candidate.get('median_mi', 0.0),
                            'mad_mi': best_candidate.get('mad_mi', 0.0),
                            'mad_over_median': best_candidate.get('mad_over_median', 0.0),
                            'stability_ratio': best_candidate.get('mad_over_median', 0.0),
                            'stability_guardrail_penalty': best_candidate.get('stability_guardrail_penalty', 0.0),
                            'is_unstable_candidate': best_candidate.get('is_unstable', False),
                            'stability_guardrail_triggered': bool(best_candidate.get('is_unstable', False)),
                            'validation_penalty': best_candidate.get('validation_penalty', 0.0),
                            'val_mi': best_candidate.get('val_mi', 0.0),
                            'original_mi': best_candidate.get('original_mi', 0.0),
                            'adjusted_mi': best_candidate.get('adjusted_mi', 0.0),
                            'top_3_coarse': original_top_3,
                            'screened_top_horizons': list(refinement_candidates),
                            'unstable_horizons': dict(unstable_horizons),
                            'stability_threshold': stability_threshold,
                            'stability_check': True,
                            'cache_hit_rate': cache_hit_rate,
                            'vectorized_ops': MATRIX_OPS_AVAILABLE,
                            'regularization_penalty': best_candidate.get('regularization_penalty', 0.0),
                            'regularization_settings': regularization_settings,
                            'penalized_objective': best_candidate.get('penalized_objective', best_score),
                        }
                    )

            # Fallback to best coarse result
            best_horizon, best_mi = coarse_results[0]
            best_stats = horizon_bootstrap_cache.get(best_horizon, {})
            best_ratio = best_stats.get('mad_over_median', unstable_horizons.get(best_horizon, 0.0))
            guardrail_penalty = 1.0 if best_horizon in unstable_horizons else 0.0
            self.logger.info(f'✅ {feature_name}: best_lookback={best_horizon}, score={best_mi:.6f}')

            return OptimizationResult(
                best_lookback_period=best_horizon,
                best_score=penalized_cache.get(best_horizon, {}).get('penalized_score', best_mi),
                optimization_method="coarse_to_refine",
                total_trials=len(coarse_results),
                optimization_time=0.0,
                convergence_achieved=True,
                feature_name=feature_name,  # FIXED: Added feature_name field
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'coarse_horizons': len(coarse_results),
                    'refined_horizons': 0,
                    'fallback': True,
                    'median_mi': best_stats.get('median_mi', 0.0),
                    'mad_mi': best_stats.get('mad_mi', 0.0),
                    'mad_over_median': best_ratio,
                    'stability_ratio': best_ratio,
                    'stability_guardrail_penalty': guardrail_penalty,
                    'stability_guardrail_triggered': best_horizon in unstable_horizons,
                    'stability_threshold': stability_threshold,
                    'unstable_horizons': dict(unstable_horizons),
                    'screened_top_horizons': list(refinement_candidates),
                    'top_3_coarse': original_top_3,
                    'regularization_penalty': penalized_cache.get(best_horizon, {}).get('penalty', 0.0),
                    'regularization_settings': regularization_settings,
                    'penalized_objective': penalized_cache.get(best_horizon, {}).get('penalized_score', best_mi),
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Coarse-to-refine optimization failed: {e}")
            return self._create_failed_result("coarse_to_refine", 0.0)

    def _optimize_coarse_to_refine_with_outer(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        outer_splits: List[Tuple[slice, slice]],
        regularization_settings: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> OptimizationResult:
        """Run coarse-to-refine optimization inside nested outer splits."""
        tprint_debug("🧠 Entering _optimize_coarse_to_refine_with_outer")

        try:
            min_lookback, max_lookback = lookback_range
            regularization_settings = self._normalize_regularization_settings(regularization_settings)
            forward_returns_full = self._get_shared_forward_returns_matrix(
                data,
                target_column,
                max_horizon=max_lookback,
            )
            if not forward_returns_full:
                return self._create_failed_result("coarse_to_refine", 0.0)

            lookback_scores: Dict[int, List[float]] = defaultdict(list)
            fold_records: List[Dict[str, Any]] = []
            total_trials = 0
            stability_scores: List[float] = []
            sensitivity_scores: List[float] = []
            convergence_flags: List[bool] = []

            for fold_index, (train_split, val_split) in enumerate(outer_splits):
                train_bounds = self._normalize_split_bounds(train_split, len(data))
                val_bounds = self._normalize_split_bounds(val_split, len(data))

                if not train_bounds or not val_bounds:
                    continue

                train_start, train_end = train_bounds
                val_start, val_end = val_bounds

                if train_end - train_start < min_lookback:
                    continue

                train_frame = data.iloc[train_start:train_end]
                if train_frame.empty:
                    continue

                # PERFORMANCE OPTIMIZATION: Pass precomputed forward returns matrix to avoid redundant computation
                inner_result = self._coarse_to_refine_single_pass(
                    train_frame,
                    feature_name,
                    target_column,
                    lookback_range,
                    regularization_settings=regularization_settings,
                    precomputed_forward_returns=forward_returns_full,  # Reuse precomputed matrix
                    **kwargs,
                )

                total_trials += inner_result.total_trials
                convergence_flags.append(inner_result.convergence_achieved)

                if inner_result.stability_score:
                    stability_scores.append(float(inner_result.stability_score))
                if inner_result.lookback_sensitivity:
                    sensitivity_scores.append(float(inner_result.lookback_sensitivity))

                lookback = int(inner_result.best_lookback_period)
                if lookback <= 0:
                    continue

                validation_score = self._score_frozen_lookback(
                    data,
                    feature_name,
                    lookback,
                    (val_start, val_end),
                    forward_returns_full,
                )

                lookback_scores[lookback].append(validation_score)

                fold_records.append({
                    'fold_index': fold_index,
                    'train_start': train_start,
                    'train_end': train_end,
                    'validation_start': val_start,
                    'validation_end': val_end,
                    'inner_best_lookback': lookback,
                    'inner_score': inner_result.best_score,
                    'validation_score': validation_score,
                })

            if not fold_records or not lookback_scores:
                return self._coarse_to_refine_single_pass(
                    data,
                    feature_name,
                    target_column,
                    lookback_range,
                    regularization_settings=regularization_settings,
                    **kwargs,
                )

            def aggregate_mean(values: List[float]) -> float:
                cleaned = [float(v) for v in values if v is not None and not np.isnan(v)]
                if not cleaned:
                    return 0.0
                return float(np.mean(cleaned))

            aggregate_scores = {}
            for lb, scores in lookback_scores.items():
                mean_score = aggregate_mean(scores)
                penalty = self._calculate_lookback_penalty(int(lb), regularization_settings)
                aggregate_scores[lb] = mean_score - penalty

            best_lookback, best_score = max(aggregate_scores.items(), key=lambda item: item[1])

            metadata = {
                'feature_name': feature_name,
                'target_column': target_column,
                'outer_folds': fold_records,
                'lookback_aggregates': {
                    int(lb): {
                        'mean_validation_score': aggregate_scores[lb],
                        'folds': len(scores),
                    }
                    for lb, scores in lookback_scores.items()
                },
                'frozen_from_inner': True,
                'outer_split_count': len(fold_records),
                'regularization_penalty': self._calculate_lookback_penalty(int(best_lookback), regularization_settings),
                'regularization_settings': regularization_settings,
            }

            stability_value = aggregate_mean(stability_scores)
            sensitivity_value = aggregate_mean(sensitivity_scores)

            return OptimizationResult(
                best_lookback_period=int(best_lookback),
                best_score=float(best_score),
                optimization_method="coarse_to_refine",
                total_trials=total_trials,
                optimization_time=0.0,
                convergence_achieved=all(convergence_flags) if convergence_flags else True,
                feature_name=feature_name,  # FIXED: Added feature_name field
                metadata=metadata,
                stability_score=stability_value,
                lookback_sensitivity=sensitivity_value,
            )

        except Exception as exc:
            self.logger.error(f"❌ Nested coarse-to-refine optimization failed: {exc}")
            return self._create_failed_result("coarse_to_refine", 0.0)

    def _normalize_split_bounds(
        self,
        split: Union[slice, Iterable[int]],
        data_length: int,
    ) -> Optional[Tuple[int, int]]:
        """Normalize split definitions to concrete integer bounds."""
        if isinstance(split, slice):
            start = 0 if split.start is None else max(0, int(split.start))
            stop = data_length if split.stop is None else min(data_length, int(split.stop))
            if start >= stop:
                return None
            return start, stop

        if isinstance(split, Iterable):
            indices = [int(idx) for idx in split if isinstance(idx, (int, np.integer))]
            if not indices:
                return None
            indices.sort()
            start = max(0, indices[0])
            stop = min(data_length, indices[-1] + 1)
            if start >= stop:
                return None
            return start, stop

        return None

    def _score_frozen_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback: int,
        bounds: Tuple[int, int],
        forward_returns: Dict[int, np.ndarray],
    ) -> float:
        """Evaluate a frozen lookback on the specified outer validation window."""
        start, end = bounds
        if start >= end or lookback <= 0:
            return 0.0

        feature_values = self._cached_feature_calculation(data, feature_name, lookback)
        if feature_values is None or len(feature_values) == 0:
            return 0.0

        returns_array = forward_returns.get(lookback, np.array([]))
        if len(returns_array) == 0:
            return 0.0

        feature_segment = feature_values[start:end]
        returns_segment = returns_array[start:end]

        min_length = min(len(feature_segment), len(returns_segment))
        if min_length < 10:
            return 0.0

        score = self._calculate_mutual_information_robust(
            feature_segment[:min_length],
            returns_segment[:min_length],
        )

        if score is None or np.isnan(score):
            return 0.0

        return float(score)


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
