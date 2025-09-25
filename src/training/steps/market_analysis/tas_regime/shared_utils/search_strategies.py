"""
Search Strategies for TAS Regime Analysis

This module provides comprehensive search strategies for market analysis using
the unified evaluation framework and various utility functions from the utils package.

Features:
- Grid Search with parallel processing
- Bayesian Optimization with TPE
- Random Search with intelligent sampling
- Hyperparameter optimization with cross-validation
- M1 hardware optimization integration
- Advanced memory management
- Comprehensive logging and monitoring
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from contextlib import contextmanager

# Import utility modules
from ...utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_float, safe_int, validate_finite, validate_positive, validate_range,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    math_safe, timed_operation, parallel_map, optimize_dataframe_dtypes,
    create_summary_statistics, safe_to_parquet, safe_read_parquet,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, cleanup_m1_optimizers, memory_checkpoint,
    gpu_context, optimize_memory, get_memory_usage
)

from ...utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_data_quality_report, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, CommonUtilities
)

from ...utils.math_validation import (
    safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, MathValidation
)

from ...utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from ...utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_with_level, tprint_batch, tprint_logged
)

# Import hardware optimization utilities
try:
    from ...utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array,
        m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    M1_GPU_AVAILABLE = True
except ImportError:
    M1_GPU_AVAILABLE = False
    tprint_warning("M1 GPU utilities not available")

try:
    from ...utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, start_m1_memory_monitoring,
        stop_m1_memory_monitoring, optimize_dataframe_memory
    )
    M1_MEMORY_AVAILABLE = True
except ImportError:
    M1_MEMORY_AVAILABLE = False
    tprint_warning("M1 memory optimizer not available")

try:
    from ...utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer
    )
    M1_CPU_AVAILABLE = True
except ImportError:
    M1_CPU_AVAILABLE = False
    tprint_warning("M1 CPU optimizer not available")

# Import ML utilities
try:
    from ...utils.ml_common.optimization import (
        BayesianOptimizer, GridSearchOptimizer, RandomSearchOptimizer
    )
    ML_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ML_OPTIMIZATION_AVAILABLE = False
    tprint_warning("ML optimization utilities not available")

try:
    from ...utils.ml_common.cvlsa import (
        CrossValidationManager, LookaheadBiasDetector
    )
    ML_CV_AVAILABLE = True
except ImportError:
    ML_CV_AVAILABLE = False
    tprint_warning("ML cross-validation utilities not available")

# Import matrix operations
try:
    from ...utils.matrix_operations.unified_operations import (
        MatrixOperationsManager, VectorizedOperations
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations utilities not available")

# Import data utilities
try:
    from ...utils.data.unified_data_utils import (
        DataLoader, DataProcessor, DataValidator
    )
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False
    tprint_warning("Data utilities not available")

# Setup logging
logger = logging.getLogger(__name__)

class SearchStrategyType(Enum):
    """Types of search strategies available."""
    GRID = "grid"
    BAYESIAN = "bayesian"
    RANDOM = "random"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

class OptimizationObjective(Enum):
    """Optimization objectives."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass
class SearchConfig:
    """Configuration for search strategies."""
    
    # Basic configuration
    strategy_type: SearchStrategyType = SearchStrategyType.GRID
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE
    n_trials: int = 100
    n_jobs: int = -1
    random_state: Optional[int] = None
    
    # Performance configuration
    timeout_seconds: Optional[int] = None
    memory_limit_gb: Optional[float] = None
    use_gpu: bool = True
    use_parallel: bool = True
    
    # Search space configuration
    param_space: Dict[str, Any] = field(default_factory=dict)
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    cv_folds: int = 5
    test_size: float = 0.2
    validation_metric: str = "accuracy"
    
    # Hardware optimization
    optimize_for_m1: bool = True
    memory_monitoring: bool = True
    gpu_acceleration: bool = True
    
    # Logging and monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_results: bool = True
    results_dir: Optional[str] = None

@dataclass
class SearchResult:
    """Results from a search strategy execution."""
    
    # Basic results
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_trial: int = 0
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_accelerated: bool = False
    
    # Search statistics
    n_trials_completed: int = 0
    n_trials_failed: int = 0
    convergence_iteration: Optional[int] = None
    
    # Detailed results
    all_scores: List[float] = field(default_factory=list)
    all_params: List[Dict[str, Any]] = field(default_factory=list)
    trial_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Hardware information
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    optimization_stats: Dict[str, Any] = field(default_factory=dict)

class SearchStrategies:
    """Comprehensive search strategies for TAS regime analysis."""
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize search strategies with configuration."""
        self.config = config or SearchConfig()
        self.logger = logger.getChild('SearchStrategies')
        
        # Initialize hardware optimizations
        self._setup_hardware_optimizations()
        
        # Initialize utilities
        self._setup_utilities()
        
        # Initialize search components
        self._setup_search_components()
        
        tprint_info("Search strategies initialized successfully")
    
    def _setup_hardware_optimizations(self):
        """Setup hardware-specific optimizations."""
        try:
            # M1 GPU optimization
            if M1_GPU_AVAILABLE and self.config.optimize_for_m1:
                self.gpu_manager = get_m1_gpu_manager()
                self.m1_available = is_m1_available()
                self.mps_available = is_mps_available()
                tprint_info(f"M1 GPU optimization enabled: M1={self.m1_available}, MPS={self.mps_available}")
            else:
                self.gpu_manager = None
                self.m1_available = False
                self.mps_available = False
            
            # M1 Memory optimization
            if M1_MEMORY_AVAILABLE and self.config.optimize_for_m1:
                self.memory_optimizer = get_m1_memory_optimizer(
                    memory_limit_gb=self.config.memory_limit_gb
                )
                if self.config.memory_monitoring:
                    start_m1_memory_monitoring()
                tprint_info("M1 memory optimization enabled")
            else:
                self.memory_optimizer = None
            
            # M1 CPU optimization
            if M1_CPU_AVAILABLE and self.config.optimize_for_m1:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_info("M1 CPU optimization enabled")
            else:
                self.cpu_optimizer = None
                
        except Exception as e:
            tprint_warning(f"Hardware optimization setup failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _setup_utilities(self):
        """Setup utility components."""
        try:
            # Math validation
            self.math_validator = MathValidation()
            
            # Common utilities
            self.common_utils = CommonUtilities()
            
            # Serialization
            self.serializer = UniversalSerializer()
            
            # Data utilities
            if DATA_UTILS_AVAILABLE:
                self.data_loader = DataLoader()
                self.data_processor = DataProcessor()
                self.data_validator = DataValidator()
            else:
                self.data_loader = None
                self.data_processor = None
                self.data_validator = None
            
            # Matrix operations
            if MATRIX_OPS_AVAILABLE:
                self.matrix_manager = MatrixOperationsManager()
                self.vectorized_ops = VectorizedOperations()
            else:
                self.matrix_manager = None
                self.vectorized_ops = None
                
        except Exception as e:
            tprint_warning(f"Utility setup failed: {e}")
    
    def _setup_search_components(self):
        """Setup search-specific components."""
        try:
            # ML optimization components
            if ML_OPTIMIZATION_AVAILABLE:
                self.bayesian_optimizer = BayesianOptimizer()
                self.grid_optimizer = GridSearchOptimizer()
                self.random_optimizer = RandomSearchOptimizer()
            else:
                self.bayesian_optimizer = None
                self.grid_optimizer = None
                self.random_optimizer = None
            
            # Cross-validation
            if ML_CV_AVAILABLE:
                self.cv_manager = CrossValidationManager()
                self.lookahead_detector = LookaheadBiasDetector()
            else:
                self.cv_manager = None
                self.lookahead_detector = None
                
        except Exception as e:
            tprint_warning(f"Search component setup failed: {e}")
    
    @tprint_logged(include_args=True, include_result=True)
    def grid_search(
        self,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> SearchResult:
        """Perform grid search optimization."""
        tprint_info("Starting grid search optimization")
        
        start_time = time.time()
        
        try:
            # Optimize data for M1 if available
            if self.m1_available and self.config.optimize_for_m1:
                with memory_checkpoint("grid_search_data_optimization"):
                    X = self._optimize_data_for_m1(X)
                    y = self._optimize_data_for_m1(y)
            
            # Use external grid optimizer if available
            if self.grid_optimizer and ML_OPTIMIZATION_AVAILABLE:
                result = self.grid_optimizer.search(
                    estimator=estimator,
                    param_grid=param_grid,
                    X=X,
                    y=y,
                    cv=self.config.cv_folds,
                    n_jobs=self.config.n_jobs,
                    **kwargs
                )
            else:
                # Fallback to manual grid search
                result = self._manual_grid_search(estimator, param_grid, X, y, **kwargs)
            
            # Create search result
            search_result = self._create_search_result(
                result, start_time, "grid_search"
            )
            
            tprint_success(f"Grid search completed: {search_result.best_score:.4f}")
            return search_result
            
        except Exception as e:
            tprint_error(f"Grid search failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    @tprint_logged(include_args=True, include_result=True)
    def bayesian_optimization(
        self,
        estimator: Any,
        param_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_trials: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """Perform Bayesian optimization with TPE."""
        tprint_info("Starting Bayesian optimization")
        
        start_time = time.time()
        n_trials = n_trials or self.config.n_trials
        
        try:
            # Optimize data for M1 if available
            if self.m1_available and self.config.optimize_for_m1:
                with memory_checkpoint("bayesian_optimization_data_optimization"):
                    X = self._optimize_data_for_m1(X)
                    y = self._optimize_data_for_m1(y)
            
            # Use external Bayesian optimizer if available
            if self.bayesian_optimizer and ML_OPTIMIZATION_AVAILABLE:
                result = self.bayesian_optimizer.optimize(
                    estimator=estimator,
                    param_space=param_space,
                    X=X,
                    y=y,
                    n_trials=n_trials,
                    cv=self.config.cv_folds,
                    **kwargs
                )
            else:
                # Fallback to manual Bayesian optimization
                result = self._manual_bayesian_optimization(
                    estimator, param_space, X, y, n_trials, **kwargs
                )
            
            # Create search result
            search_result = self._create_search_result(
                result, start_time, "bayesian_optimization"
            )
            
            tprint_success(f"Bayesian optimization completed: {search_result.best_score:.4f}")
            return search_result
            
        except Exception as e:
            tprint_error(f"Bayesian optimization failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    @tprint_logged(include_args=True, include_result=True)
    def random_search(
        self,
        estimator: Any,
        param_distributions: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_iter: Optional[int] = None,
        **kwargs
    ) -> SearchResult:
        """Perform random search optimization."""
        tprint_info("Starting random search optimization")
        
        start_time = time.time()
        n_iter = n_iter or self.config.n_trials
        
        try:
            # Optimize data for M1 if available
            if self.m1_available and self.config.optimize_for_m1:
                with memory_checkpoint("random_search_data_optimization"):
                    X = self._optimize_data_for_m1(X)
                    y = self._optimize_data_for_m1(y)
            
            # Use external random optimizer if available
            if self.random_optimizer and ML_OPTIMIZATION_AVAILABLE:
                result = self.random_optimizer.search(
                    estimator=estimator,
                    param_distributions=param_distributions,
                    X=X,
                    y=y,
                    n_iter=n_iter,
                    cv=self.config.cv_folds,
                    n_jobs=self.config.n_jobs,
                    **kwargs
                )
            else:
                # Fallback to manual random search
                result = self._manual_random_search(
                    estimator, param_distributions, X, y, n_iter, **kwargs
                )
            
            # Create search result
            search_result = self._create_search_result(
                result, start_time, "random_search"
            )
            
            tprint_success(f"Random search completed: {search_result.best_score:.4f}")
            return search_result
            
        except Exception as e:
            tprint_error(f"Random search failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    @tprint_logged(include_args=True, include_result=True)
    def hybrid_search(
        self,
        estimator: Any,
        param_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> SearchResult:
        """Perform hybrid search combining multiple strategies."""
        tprint_info("Starting hybrid search optimization")
        
        start_time = time.time()
        
        try:
            # Optimize data for M1 if available
            if self.m1_available and self.config.optimize_for_m1:
                with memory_checkpoint("hybrid_search_data_optimization"):
                    X = self._optimize_data_for_m1(X)
                    y = self._optimize_data_for_m1(y)
            
            # Phase 1: Random search for exploration
            tprint_info("Phase 1: Random exploration")
            random_result = self.random_search(
                estimator, param_space, X, y, n_iter=self.config.n_trials // 2
            )
            
            # Phase 2: Bayesian optimization for exploitation
            tprint_info("Phase 2: Bayesian exploitation")
            # Focus search around best random results
            focused_space = self._create_focused_param_space(
                param_space, random_result.best_params
            )
            
            bayesian_result = self.bayesian_optimization(
                estimator, focused_space, X, y, n_trials=self.config.n_trials // 2
            )
            
            # Combine results
            if bayesian_result.best_score > random_result.best_score:
                best_result = bayesian_result
                tprint_info("Bayesian optimization found better solution")
            else:
                best_result = random_result
                tprint_info("Random search found better solution")
            
            # Create combined search result
            search_result = self._create_hybrid_search_result(
                random_result, bayesian_result, best_result, start_time
            )
            
            tprint_success(f"Hybrid search completed: {search_result.best_score:.4f}")
            return search_result
            
        except Exception as e:
            tprint_error(f"Hybrid search failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _optimize_data_for_m1(self, data: np.ndarray) -> np.ndarray:
        """Optimize data for M1 hardware."""
        if not self.m1_available:
            return data
        
        try:
            if self.gpu_manager and self.mps_available:
                # Use GPU optimization
                return self.gpu_manager.optimize_tensor_operations(data)
            else:
                # Use CPU optimization
                return create_m1_optimized_array(data)
        except Exception as e:
            tprint_warning(f"Data optimization failed: {e}")
            return data
    
    def _manual_grid_search(
        self,
        estimator: Any,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Manual grid search implementation."""
        from sklearn.model_selection import GridSearchCV
        
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
            scoring=self.config.validation_metric,
            **kwargs
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def _manual_bayesian_optimization(
        self,
        estimator: Any,
        param_space: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Manual Bayesian optimization implementation using unified Bayesian TPE optimizer."""
        try:
            # Import Bayesian TPE optimizer
            from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
                BayesianTPEOptimizer,
                BayesianTPEConfig
            )
            
            # Define objective function for Bayesian TPE optimizer
            def objective_function(params: Dict[str, Any], **kwargs) -> float:
                try:
                    # Create estimator with parameters
                    est = estimator.set_params(**params)
                    
                    # Cross-validation
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(
                        est, X, y, cv=self.config.cv_folds, 
                        scoring=self.config.validation_metric
                    )
                    
                    return scores.mean()
                    
                except Exception as e:
                    tprint_warning(f"Objective function failed: {e}")
                    return -np.inf
            
            # Configure Bayesian TPE optimizer
            tpe_config = BayesianTPEConfig(
                n_trials=n_trials,
                timeout_seconds=self.config.timeout_seconds,
                enable_grid_search=True,
                coarse_grid_points=3,
                fine_grid_points=5,
                backend='optuna',
                enable_parallel=True,
                max_workers=self.config.n_jobs,
                enable_early_stopping=True,
                early_stopping_patience=10,
                log_level='INFO'
            )
            
            # Run optimization using new unified optimizer
            tprint_info("🎯 Starting Bayesian TPE optimization for search strategies")
            optimizer = BayesianTPEOptimizer(tpe_config)
            result = optimizer.optimize(objective_function, param_space)
            
            if not result.success:
                raise RuntimeError(f"Search strategy optimization failed: {result.error_message}")
            
            tprint_success(f"✅ Search strategy optimization completed")
            tprint_info(f"📊 Best score: {result.best_score:.4f}")
            tprint_info(f"📊 Optimization time: {result.optimization_time:.2f}s")
            tprint_info(f"📊 Trials: {result.n_trials}")
            
            return {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'n_trials': result.n_trials,
                'convergence_info': result.convergence_info,
                'grid_search_results': result.grid_search_results
            }
            
        except Exception as e:
            tprint_warning(f"Bayesian TPE optimization failed: {e}, falling back to random search")
            return self._manual_random_search(
                estimator, param_space, X, y, n_trials, **kwargs
            )
    
    def _manual_random_search(
        self,
        estimator: Any,
        param_distributions: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Manual random search implementation."""
        from sklearn.model_selection import RandomizedSearchCV
        
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
            scoring=self.config.validation_metric,
            random_state=self.config.random_state,
            **kwargs
        )
        
        random_search.fit(X, y)
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
    
    def _create_focused_param_space(
        self,
        original_space: Dict[str, Any],
        best_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create focused parameter space around best parameters."""
        focused_space = {}
        
        for param_name, param_value in best_params.items():
            if param_name in original_space:
                if isinstance(param_value, (int, float)):
                    # Create range around the value
                    if isinstance(param_value, int):
                        focused_space[param_name] = {
                            'type': 'int',
                            'low': max(1, int(param_value * 0.5)),
                            'high': int(param_value * 2)
                        }
                    else:
                        focused_space[param_name] = {
                            'type': 'float',
                            'low': param_value * 0.5,
                            'high': param_value * 2
                        }
                else:
                    # Keep categorical parameters as is
                    focused_space[param_name] = original_space[param_name]
        
        return focused_space
    
    def _create_search_result(
        self,
        result: Dict[str, Any],
        start_time: float,
        strategy_name: str
    ) -> SearchResult:
        """Create search result from optimization output."""
        execution_time = time.time() - start_time
        
        # Get memory usage
        memory_usage = 0.0
        if self.memory_optimizer:
            memory_stats = self.memory_optimizer.get_memory_stats()
            memory_usage = memory_stats.get('used_memory', 0) / (1024 * 1024)  # Convert to MB
        
        # Get hardware info
        hardware_info = self._get_hardware_info()
        
        # Create search result
        search_result = SearchResult(
            best_params=result.get('best_params', {}),
            best_score=result.get('best_score', 0.0),
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            gpu_accelerated=self.mps_available and self.config.gpu_acceleration,
            hardware_info=hardware_info,
            optimization_stats=self._get_optimization_stats()
        )
        
        # Add trial history if available
        if 'cv_results' in result:
            cv_results = result['cv_results']
            search_result.all_scores = cv_results.get('mean_test_score', []).tolist()
            search_result.n_trials_completed = len(search_result.all_scores)
        
        return search_result
    
    def _create_hybrid_search_result(
        self,
        random_result: SearchResult,
        bayesian_result: SearchResult,
        best_result: SearchResult,
        start_time: float
    ) -> SearchResult:
        """Create hybrid search result combining multiple strategies."""
        execution_time = time.time() - start_time
        
        # Combine all scores and parameters
        all_scores = random_result.all_scores + bayesian_result.all_scores
        all_params = random_result.all_params + bayesian_result.all_params
        
        # Create combined result
        combined_result = SearchResult(
            best_params=best_result.best_params,
            best_score=best_result.best_score,
            execution_time=execution_time,
            memory_usage_mb=max(random_result.memory_usage_mb, bayesian_result.memory_usage_mb),
            gpu_accelerated=best_result.gpu_accelerated,
            n_trials_completed=len(all_scores),
            all_scores=all_scores,
            all_params=all_params,
            hardware_info=self._get_hardware_info(),
            optimization_stats=self._get_optimization_stats()
        )
        
        return combined_result
    
    def _create_error_result(self, start_time: float, error_message: str) -> SearchResult:
        """Create error result."""
        execution_time = time.time() - start_time
        
        return SearchResult(
            best_params={},
            best_score=0.0,
            execution_time=execution_time,
            memory_usage_mb=0.0,
            gpu_accelerated=False,
            n_trials_failed=1,
            hardware_info=self._get_hardware_info(),
            optimization_stats={'error': error_message}
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        hardware_info = {
            'm1_available': self.m1_available,
            'mps_available': self.mps_available,
            'gpu_manager_available': self.gpu_manager is not None,
            'memory_optimizer_available': self.memory_optimizer is not None,
            'cpu_optimizer_available': self.cpu_optimizer is not None
        }
        
        if self.gpu_manager:
            hardware_info.update(self.gpu_manager.get_gpu_info())
        
        if self.memory_optimizer:
            hardware_info['memory_stats'] = self.memory_optimizer.get_memory_stats()
        
        return hardware_info
    
    def _get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'm1_optimization_enabled': self.config.optimize_for_m1,
            'gpu_acceleration_enabled': self.config.gpu_acceleration,
            'memory_monitoring_enabled': self.config.memory_monitoring,
            'parallel_processing_enabled': self.config.use_parallel
        }
        
        if self.memory_optimizer:
            stats['memory_optimization_stats'] = self.memory_optimizer.get_memory_stats()
        
        return stats
    
    def save_results(self, result: SearchResult, filename: Optional[str] = None) -> bool:
        """Save search results to file."""
        try:
            if not self.config.save_results:
                return True
            
            # Determine filename
            if not filename:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"search_results_{timestamp}.json"
            
            # Determine directory
            results_dir = Path(self.config.results_dir or "results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = results_dir / filename
            
            # Convert result to serializable format
            result_dict = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'execution_time': result.execution_time,
                'memory_usage_mb': result.memory_usage_mb,
                'gpu_accelerated': result.gpu_accelerated,
                'n_trials_completed': result.n_trials_completed,
                'n_trials_failed': result.n_trials_failed,
                'all_scores': result.all_scores,
                'hardware_info': result.hardware_info,
                'optimization_stats': result.optimization_stats
            }
            
            # Save using universal serializer
            success = self.serializer.save(result_dict, str(filepath))
            
            if success:
                tprint_success(f"Results saved to {filepath}")
            else:
                tprint_error(f"Failed to save results to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"Error saving results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[SearchResult]:
        """Load search results from file."""
        try:
            # Load using universal serializer
            result_dict = self.serializer.load(filepath)
            
            if not result_dict:
                tprint_error(f"Failed to load results from {filepath}")
                return None
            
            # Convert back to SearchResult
            result = SearchResult(
                best_params=result_dict.get('best_params', {}),
                best_score=result_dict.get('best_score', 0.0),
                execution_time=result_dict.get('execution_time', 0.0),
                memory_usage_mb=result_dict.get('memory_usage_mb', 0.0),
                gpu_accelerated=result_dict.get('gpu_accelerated', False),
                n_trials_completed=result_dict.get('n_trials_completed', 0),
                n_trials_failed=result_dict.get('n_trials_failed', 0),
                all_scores=result_dict.get('all_scores', []),
                hardware_info=result_dict.get('hardware_info', {}),
                optimization_stats=result_dict.get('optimization_stats', {})
            )
            
            tprint_success(f"Results loaded from {filepath}")
            return result
            
        except Exception as e:
            tprint_error(f"Error loading results: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop memory monitoring
            if self.memory_optimizer and self.config.memory_monitoring:
                stop_m1_memory_monitoring()
            
            # Cleanup M1 optimizers
            if self.config.optimize_for_m1:
                cleanup_m1_optimizers()
            
            tprint_info("Search strategies cleanup completed")
            
        except Exception as e:
            tprint_warning(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def evaluate_strategy(self, strategy, X_test, y_test, **kwargs):
        """Evaluate search strategy using unified framework."""
        return self.evaluator.evaluate_model(strategy, X_test, y_test, **kwargs)

# Convenience functions for backward compatibility
def create_search_strategies(config: Optional[SearchConfig] = None) -> SearchStrategies:
    """Create search strategies instance."""
    return SearchStrategies(config)

def grid_search(
    estimator: Any,
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[SearchConfig] = None,
    **kwargs
) -> SearchResult:
    """Convenience function for grid search."""
    with SearchStrategies(config) as strategies:
        return strategies.grid_search(estimator, param_grid, X, y, **kwargs)

def bayesian_optimization(
    estimator: Any,
    param_space: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[SearchConfig] = None,
    **kwargs
) -> SearchResult:
    """Convenience function for Bayesian optimization."""
    with SearchStrategies(config) as strategies:
        return strategies.bayesian_optimization(estimator, param_space, X, y, **kwargs)

def random_search(
    estimator: Any,
    param_distributions: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[SearchConfig] = None,
    **kwargs
) -> SearchResult:
    """Convenience function for random search."""
    with SearchStrategies(config) as strategies:
        return strategies.random_search(estimator, param_distributions, X, y, **kwargs)

def hybrid_search(
    estimator: Any,
    param_space: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[SearchConfig] = None,
    **kwargs
) -> SearchResult:
    """Convenience function for hybrid search."""
    with SearchStrategies(config) as strategies:
        return strategies.hybrid_search(estimator, param_space, X, y, **kwargs)

# Export main classes and functions
__all__ = [
    'SearchStrategies',
    'SearchConfig',
    'SearchResult',
    'SearchStrategyType',
    'OptimizationObjective',
    'create_search_strategies',
    'grid_search',
    'bayesian_optimization',
    'random_search',
    'hybrid_search'
]
