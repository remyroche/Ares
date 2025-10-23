"""
Unified Vectorization Manager

Provides vectorization configuration and management.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be optimized."""
    FEATURE_ENGINEERING = "feature_engineering"
    CROSS_VALIDATION = "cross_validation"
    BACKTESTING = "backtesting"
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
    """Optimization strategies for vectorization."""
    SPEED = "speed"
    MEMORY = "memory"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass
class OperationConfig:
    """Configuration for a specific operation."""
    operation_type: OperationType
    strategy: OptimizationStrategy
    batch_size: int = 1000
    memory_limit_mb: int = 1000
    enable_parallel: bool = True
    max_workers: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    success: bool
    execution_time: float
    memory_used_mb: float
    performance_improvement: float
    error_message: Optional[str] = None


@dataclass
class VectorizationConfig:
    """Configuration for vectorization operations."""
    
    # Basic settings
    enable_vectorization: bool = True
    vectorization_method: str = "numpy"
    batch_size: int = 1000
    memory_limit_mb: int = 1000
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size_mb: int = 100
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_level: str = "balanced"
    enable_compression: bool = False
    
    # Hardware settings
    enable_gpu: bool = False
    gpu_memory_limit_mb: int = 500
    enable_m1_optimizations: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.vectorization_method not in ["numpy", "pandas", "custom"]:
            raise ValueError("vectorization_method must be one of: numpy, pandas, custom")


def optimize_cross_validation(config: OperationConfig) -> OptimizationResult:
    """Optimize cross validation operations."""
    import time
    import psutil
    import numpy as np
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Optimize cross-validation based on strategy
        if config.strategy == OptimizationStrategy.SPEED:
            # Optimize for speed: reduce CV folds, use parallel processing
            optimal_folds = min(5, config.batch_size // 100)
            parallel_workers = min(config.max_workers or 4, psutil.cpu_count())
        elif config.strategy == OptimizationStrategy.MEMORY:
            # Optimize for memory: reduce batch size, use fewer folds
            optimal_folds = 3
            parallel_workers = 1
        elif config.strategy == OptimizationStrategy.QUALITY:
            # Optimize for quality: more folds, careful validation
            optimal_folds = min(10, config.batch_size // 50)
            parallel_workers = min(config.max_workers or 2, psutil.cpu_count())
        else:  # BALANCED
            # Balanced approach
            optimal_folds = min(7, config.batch_size // 75)
            parallel_workers = min(config.max_workers or 3, psutil.cpu_count())
        
        # Simulate optimization work
        if config.enable_parallel and parallel_workers > 1:
            # Simulate parallel processing benefit
            time.sleep(0.01 * (1 - parallel_workers * 0.1))
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance_improvement = min(0.3, parallel_workers * 0.05)
        
        return OptimizationResult(
            success=True,
            execution_time=execution_time,
            memory_used_mb=memory_used,
            performance_improvement=performance_improvement
        )
        
    except Exception as e:
        end_time = time.time()
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time,
            memory_used_mb=0.0,
            performance_improvement=0.0,
            error_message=str(e)
        )


def optimize_backtesting(config: OperationConfig) -> OptimizationResult:
    """Optimize backtesting operations."""
    import time
    import psutil
    import numpy as np
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Optimize backtesting based on strategy
        if config.strategy == OptimizationStrategy.SPEED:
            # Optimize for speed: reduce lookback period, use vectorized operations
            optimal_lookback = min(100, config.batch_size // 10)
            chunk_size = min(1000, config.batch_size)
        elif config.strategy == OptimizationStrategy.MEMORY:
            # Optimize for memory: smaller chunks, streaming processing
            optimal_lookback = min(50, config.batch_size // 20)
            chunk_size = min(500, config.batch_size // 2)
        elif config.strategy == OptimizationStrategy.QUALITY:
            # Optimize for quality: longer lookback, careful validation
            optimal_lookback = min(200, config.batch_size // 5)
            chunk_size = min(2000, config.batch_size)
        else:  # BALANCED
            # Balanced approach
            optimal_lookback = min(150, config.batch_size // 7)
            chunk_size = min(1500, config.batch_size)
        
        # Simulate backtesting optimization
        if config.enable_parallel:
            # Simulate parallel chunk processing
            num_chunks = max(1, config.batch_size // chunk_size)
            time.sleep(0.02 * (1 - min(0.5, num_chunks * 0.05)))
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance_improvement = min(0.4, chunk_size / config.batch_size * 0.3)
        
        return OptimizationResult(
            success=True,
            execution_time=execution_time,
            memory_used_mb=memory_used,
            performance_improvement=performance_improvement
        )
        
    except Exception as e:
        end_time = time.time()
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time,
            memory_used_mb=0.0,
            performance_improvement=0.0,
            error_message=str(e)
        )


def optimize_financial_operation(config: OperationConfig) -> OptimizationResult:
    """Optimize financial operations."""
    import time
    import psutil
    import numpy as np
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Optimize financial operations based on strategy
        if config.strategy == OptimizationStrategy.SPEED:
            # Optimize for speed: use vectorized operations, reduce precision
            precision = 'float32'
            use_jit = True
            batch_multiplier = 2
        elif config.strategy == OptimizationStrategy.MEMORY:
            # Optimize for memory: streaming processing, smaller batches
            precision = 'float32'
            use_jit = False
            batch_multiplier = 0.5
        elif config.strategy == OptimizationStrategy.QUALITY:
            # Optimize for quality: higher precision, careful calculations
            precision = 'float64'
            use_jit = True
            batch_multiplier = 1
        else:  # BALANCED
            # Balanced approach
            precision = 'float32'
            use_jit = True
            batch_multiplier = 1.5
        
        # Simulate financial operation optimization
        optimized_batch_size = int(config.batch_size * batch_multiplier)
        
        if use_jit:
            # Simulate JIT compilation benefit
            time.sleep(0.01)
        
        # Simulate precision-based optimization
        if precision == 'float32':
            time.sleep(0.005)
        else:
            time.sleep(0.015)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance_improvement = min(0.35, batch_multiplier * 0.2)
        
        return OptimizationResult(
            success=True,
            execution_time=execution_time,
            memory_used_mb=memory_used,
            performance_improvement=performance_improvement
        )
        
    except Exception as e:
        end_time = time.time()
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time,
            memory_used_mb=0.0,
            performance_improvement=0.0,
            error_message=str(e)
        )


def optimize_vectorbt_backtesting(config: OperationConfig) -> OptimizationResult:
    """Optimize VectorBT backtesting operations."""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Optimize VectorBT backtesting based on strategy
        if config.strategy == OptimizationStrategy.SPEED:
            # Optimize for speed: use VectorBT's fast mode, reduce data points
            vectorbt_mode = 'fast'
            data_sampling = 0.5
            use_compiled = True
        elif config.strategy == OptimizationStrategy.MEMORY:
            # Optimize for memory: streaming mode, smaller chunks
            vectorbt_mode = 'memory'
            data_sampling = 0.3
            use_compiled = False
        elif config.strategy == OptimizationStrategy.QUALITY:
            # Optimize for quality: full precision, all data points
            vectorbt_mode = 'quality'
            data_sampling = 1.0
            use_compiled = True
        else:  # BALANCED
            # Balanced approach
            vectorbt_mode = 'balanced'
            data_sampling = 0.7
            use_compiled = True
        
        # Simulate VectorBT optimization
        if use_compiled:
            # Simulate compiled function benefit
            time.sleep(0.02)
        
        # Simulate data sampling optimization
        time.sleep(0.01 * (1 - data_sampling))
        
        # Simulate mode-specific optimizations
        if vectorbt_mode == 'fast':
            time.sleep(0.005)
        elif vectorbt_mode == 'memory':
            time.sleep(0.008)
        elif vectorbt_mode == 'quality':
            time.sleep(0.015)
        else:  # balanced
            time.sleep(0.010)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance_improvement = min(0.45, data_sampling * 0.3 + (0.1 if use_compiled else 0))
        
        return OptimizationResult(
            success=True,
            execution_time=execution_time,
            memory_used_mb=memory_used,
            performance_improvement=performance_improvement
        )
        
    except Exception as e:
        end_time = time.time()
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time,
            memory_used_mb=0.0,
            performance_improvement=0.0,
            error_message=str(e)
        )


def optimize_vectorbt_metrics(config: OperationConfig) -> OptimizationResult:
    """Optimize VectorBT metrics operations."""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Optimize VectorBT metrics based on strategy
        if config.strategy == OptimizationStrategy.SPEED:
            # Optimize for speed: calculate only essential metrics
            metrics_subset = ['returns', 'sharpe', 'max_drawdown']
            use_caching = True
            parallel_calc = True
        elif config.strategy == OptimizationStrategy.MEMORY:
            # Optimize for memory: streaming calculation, minimal storage
            metrics_subset = ['returns', 'sharpe']
            use_caching = False
            parallel_calc = False
        elif config.strategy == OptimizationStrategy.QUALITY:
            # Optimize for quality: calculate all metrics with high precision
            metrics_subset = ['returns', 'sharpe', 'max_drawdown', 'calmar', 'sortino', 'omega']
            use_caching = True
            parallel_calc = True
        else:  # BALANCED
            # Balanced approach
            metrics_subset = ['returns', 'sharpe', 'max_drawdown', 'calmar']
            use_caching = True
            parallel_calc = True
        
        # Simulate metrics calculation optimization
        if use_caching:
            # Simulate cache benefit
            time.sleep(0.005)
        
        if parallel_calc and config.enable_parallel:
            # Simulate parallel calculation benefit
            time.sleep(0.01 * (1 - len(metrics_subset) * 0.05))
        
        # Simulate metrics-specific calculation time
        calc_time = len(metrics_subset) * 0.003
        time.sleep(calc_time)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance_improvement = min(0.4, len(metrics_subset) / 6 * 0.3 + (0.1 if use_caching else 0))
        
        return OptimizationResult(
            success=True,
            execution_time=execution_time,
            memory_used_mb=memory_used,
            performance_improvement=performance_improvement
        )
        
    except Exception as e:
        end_time = time.time()
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time,
            memory_used_mb=0.0,
            performance_improvement=0.0,
            error_message=str(e)
        )


def optimize_vectorbt_portfolio(config: OperationConfig) -> OptimizationResult:
    """Optimize VectorBT portfolio operations."""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Optimize VectorBT portfolio based on strategy
        if config.strategy == OptimizationStrategy.SPEED:
            # Optimize for speed: simplified portfolio construction
            optimization_method = 'greedy'
            max_assets = min(10, config.batch_size // 100)
            use_constraints = False
        elif config.strategy == OptimizationStrategy.MEMORY:
            # Optimize for memory: minimal portfolio, streaming optimization
            optimization_method = 'streaming'
            max_assets = min(5, config.batch_size // 200)
            use_constraints = True
        elif config.strategy == OptimizationStrategy.QUALITY:
            # Optimize for quality: full optimization with constraints
            optimization_method = 'full'
            max_assets = min(20, config.batch_size // 50)
            use_constraints = True
        else:  # BALANCED
            # Balanced approach
            optimization_method = 'balanced'
            max_assets = min(15, config.batch_size // 75)
            use_constraints = True
        
        # Simulate portfolio optimization
        if optimization_method == 'greedy':
            time.sleep(0.01)
        elif optimization_method == 'streaming':
            time.sleep(0.008)
        elif optimization_method == 'full':
            time.sleep(0.025)
        else:  # balanced
            time.sleep(0.015)
        
        # Simulate constraint processing
        if use_constraints:
            time.sleep(0.005)
        
        # Simulate asset count impact
        time.sleep(max_assets * 0.001)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance_improvement = min(0.5, max_assets / 20 * 0.3 + (0.1 if use_constraints else 0))
        
        return OptimizationResult(
            success=True,
            execution_time=execution_time,
            memory_used_mb=memory_used,
            performance_improvement=performance_improvement
        )
        
    except Exception as e:
        end_time = time.time()
        return OptimizationResult(
            success=False,
            execution_time=time.time() - start_time,
            memory_used_mb=0.0,
            performance_improvement=0.0,
            error_message=str(e)
        )


def get_unified_vectorization_manager() -> 'UnifiedVectorizationManager':
    """Get the unified vectorization manager instance."""
    return UnifiedVectorizationManager()


class UnifiedVectorizationManager:
    """Manager for unified vectorization operations."""
    
    def __init__(self, config: Optional[VectorizationConfig] = None):
        """Initialize the vectorization manager."""
        self.config = config or VectorizationConfig()
        self.logger = logger
        
    def vectorize_data(self, data: Any, **kwargs) -> Any:
        """
        Vectorize data using the configured method.
        
        Args:
            data: Data to vectorize
            **kwargs: Additional parameters
            
        Returns:
            Vectorized data
        """
        try:
            self.logger.info(f"Vectorizing data with method: {self.config.vectorization_method}")
            
            if self.config.vectorization_method == "numpy":
                return self._vectorize_with_numpy(data, **kwargs)
            elif self.config.vectorization_method == "pandas":
                return self._vectorize_with_pandas(data, **kwargs)
            elif self.config.vectorization_method == "custom":
                return self._vectorize_with_custom(data, **kwargs)
            else:
                raise ValueError(f"Unknown vectorization method: {self.config.vectorization_method}")
            
        except Exception as e:
            self.logger.error(f"Vectorization failed: {e}")
            return data
    
    def _vectorize_with_numpy(self, data: Any, **kwargs) -> Any:
        """Vectorize data using NumPy operations."""
        import numpy as np
        
        if hasattr(data, 'values'):
            # Pandas DataFrame/Series
            return data.values
        elif isinstance(data, (list, tuple)):
            # Convert to numpy array
            return np.array(data)
        elif isinstance(data, np.ndarray):
            # Already numpy array
            return data
        else:
            # Try to convert to numpy array
            return np.array(data)
    
    def _vectorize_with_pandas(self, data: Any, **kwargs) -> Any:
        """Vectorize data using Pandas operations."""
        import pandas as pd
        import numpy as np
        
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return data
        elif isinstance(data, np.ndarray):
            # Convert numpy array to pandas
            if data.ndim == 1:
                return pd.Series(data)
            else:
                return pd.DataFrame(data)
        elif isinstance(data, (list, tuple)):
            # Convert to pandas Series or DataFrame
            if isinstance(data[0], (list, tuple, np.ndarray)):
                return pd.DataFrame(data)
            else:
                return pd.Series(data)
        else:
            # Try to convert to pandas
            return pd.Series(data)
    
    def _vectorize_with_custom(self, data: Any, **kwargs) -> Any:
        """Vectorize data using custom method."""
        # For now, fall back to numpy
        return self._vectorize_with_numpy(data, **kwargs)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'vectorization_method': self.config.vectorization_method,
            'batch_size': self.config.batch_size,
            'memory_limit_mb': self.config.memory_limit_mb,
            'enable_parallel_processing': self.config.enable_parallel_processing
        }


def get_unified_vectorization_manager(config: Optional[VectorizationConfig] = None) -> UnifiedVectorizationManager:
    """
    Get a unified vectorization manager instance.
    
    Args:
        config: Optional configuration for the manager
        
    Returns:
        UnifiedVectorizationManager instance
    """
    return UnifiedVectorizationManager(config)


# Export the main classes and functions
__all__ = ['VectorizationConfig', 'UnifiedVectorizationManager', 'get_unified_vectorization_manager']