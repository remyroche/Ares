"""
Unified Optimization Wrapper

This module provides a unified interface that integrates all optimization components
for feature generation, including rolling operations, statistical calculations,
batch processing, and the Unified Vectorization Manager.

Key Features:
- Single interface for all optimizations
- Automatic optimization strategy selection
- Performance monitoring and reporting
- Consistent error handling and fallbacks
- Memory management and GPU acceleration
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

# Import optimization components
from .consolidated_rolling_optimizer import (
    ConsolidatedRollingOptimizer, 
    RollingOperationConfig, 
    RollingOperationType,
    BatchRollingConfig,
    get_global_rolling_optimizer
)
from .statistical_calculations_optimizer import (
    StatisticalCalculationsOptimizer,
    StatisticalOperationConfig,
    StatisticalOperationType,
    BatchStatisticalConfig,
    get_global_statistical_optimizer
)

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager, 
        OperationType, 
        OptimizationStrategy,
        OperationConfig,
        OptimizationResult
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    get_unified_vectorization_manager = None
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None
    OperationConfig = None
    OptimizationResult = None

# Batch processing
try:
    from ..core.vectorbt_batch_processor import (
        VectorBTBatchProcessor,
        BatchProcessingConfig,
        create_vectorbt_batch_processor
    )
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    BATCH_PROCESSING_AVAILABLE = False
    VectorBTBatchProcessor = None
    BatchProcessingConfig = None
    create_vectorbt_batch_processor = None

# Performance monitoring
try:
    from ...utils.ml_common.vectorbt_performance_monitor import (
        get_performance_monitor,
        PerformanceMonitor
    )
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    get_performance_monitor = None
    PerformanceMonitor = None

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes available."""
    AUTO = "auto"  # Automatically select best optimization
    ROLLING = "rolling"  # Focus on rolling operations
    STATISTICAL = "statistical"  # Focus on statistical calculations
    BATCH = "batch"  # Focus on batch processing
    UNIFIED = "unified"  # Use Unified Vectorization Manager
    FALLBACK = "fallback"  # Use fallback implementations


@dataclass
class UnifiedOptimizationConfig:
    """Configuration for unified optimization."""
    # Basic settings
    mode: OptimizationMode = OptimizationMode.AUTO
    enable_gpu: bool = True
    enable_parallel: bool = True
    memory_optimization: bool = True
    
    # Performance thresholds
    performance_threshold: int = 1000
    gpu_threshold: int = 2000
    batch_threshold: int = 500
    
    # Memory settings
    memory_limit_gb: float = 8.0
    chunk_size: int = 1000
    
    # Monitoring settings
    enable_performance_monitoring: bool = True
    enable_detailed_logging: bool = False
    
    # Fallback settings
    enable_fallbacks: bool = True
    fallback_timeout: float = 30.0


class UnifiedOptimizationWrapper:
    """
    Unified optimization wrapper that integrates all optimization components.
    
    This class provides a single interface for all feature generation optimizations,
    automatically selecting the best optimization strategy based on data characteristics
    and available hardware.
    """
    
    def __init__(self, config: Optional[UnifiedOptimizationConfig] = None):
        """
        Initialize the unified optimization wrapper.
        
        Args:
            config: Configuration for unified optimization
        """
        self.config = config or UnifiedOptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'rolling_operations': 0,
            'statistical_operations': 0,
            'batch_operations': 0,
            'unified_operations': 0,
            'fallback_operations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'average_time_per_operation': 0.0,
            'memory_savings_mb': 0.0,
            'optimization_hits': 0,
            'optimization_misses': 0
        }
    
    def _initialize_components(self):
        """Initialize all optimization components."""
        # Initialize rolling optimizer
        rolling_config = BatchRollingConfig(
            enable_gpu=self.config.enable_gpu,
            enable_parallel=self.config.enable_parallel,
            performance_threshold=self.config.performance_threshold
        )
        self.rolling_optimizer = ConsolidatedRollingOptimizer(rolling_config)
        
        # Initialize statistical optimizer
        statistical_config = BatchStatisticalConfig(
            enable_gpu=self.config.enable_gpu,
            enable_parallel=self.config.enable_parallel,
            performance_threshold=self.config.performance_threshold
        )
        self.statistical_optimizer = StatisticalCalculationsOptimizer(statistical_config)
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None
            self.logger.warning("Unified Vectorization Manager not available")
        
        # Initialize batch processor
        if BATCH_PROCESSING_AVAILABLE:
            batch_config = BatchProcessingConfig(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                chunk_size=self.config.chunk_size,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.batch_processor = create_vectorbt_batch_processor(batch_config)
        else:
            self.batch_processor = None
            self.logger.warning("Batch processing not available")
        
        # Initialize performance monitor
        if PERFORMANCE_MONITORING_AVAILABLE:
            self.performance_monitor = get_performance_monitor()
        else:
            self.performance_monitor = None
            self.logger.warning("Performance monitoring not available")
    
    def optimize_operation(self, 
                          operation_type: str,
                          data: Union[pd.Series, pd.DataFrame],
                          operation_func: Callable,
                          **kwargs) -> Any:
        """
        Optimize an operation using the best available strategy.
        
        Args:
            operation_type: Type of operation ('rolling', 'statistical', 'batch', etc.)
            data: Input data
            operation_func: Function to optimize
            **kwargs: Additional arguments for the operation
            
        Returns:
            Optimized result
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Determine optimization strategy
        strategy = self._select_optimization_strategy(operation_type, data, operation_func)
        
        try:
            if strategy == OptimizationMode.ROLLING:
                result = self._optimize_rolling_operation(data, operation_func, **kwargs)
                self.performance_stats['rolling_operations'] += 1
            elif strategy == OptimizationMode.STATISTICAL:
                result = self._optimize_statistical_operation(data, operation_func, **kwargs)
                self.performance_stats['statistical_operations'] += 1
            elif strategy == OptimizationMode.BATCH:
                result = self._optimize_batch_operation(data, operation_func, **kwargs)
                self.performance_stats['batch_operations'] += 1
            elif strategy == OptimizationMode.UNIFIED:
                result = self._optimize_unified_operation(data, operation_func, **kwargs)
                self.performance_stats['unified_operations'] += 1
            else:
                result = self._fallback_operation(data, operation_func, **kwargs)
                self.performance_stats['fallback_operations'] += 1
            
            # Update performance stats
            operation_time = time.time() - start_time
            self.performance_stats['total_time'] += operation_time
            self.performance_stats['average_time_per_operation'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_operations']
            )
            self.performance_stats['optimization_hits'] += 1
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Optimization failed: {e}, using fallback")
            self.performance_stats['optimization_misses'] += 1
            return self._fallback_operation(data, operation_func, **kwargs)
    
    def _select_optimization_strategy(self, 
                                    operation_type: str,
                                    data: Union[pd.Series, pd.DataFrame],
                                    operation_func: Callable) -> OptimizationMode:
        """Select the best optimization strategy based on operation and data characteristics."""
        data_size = len(data) if hasattr(data, '__len__') else data.shape[0]
        
        # Force specific mode if configured
        if self.config.mode != OptimizationMode.AUTO:
            return self.config.mode
        
        # Auto-selection logic
        if operation_type in ['rolling', 'rolling_mean', 'rolling_std', 'rolling_var']:
            return OptimizationMode.ROLLING
        elif operation_type in ['statistical', 'skew', 'kurt', 'correlation']:
            return OptimizationMode.STATISTICAL
        elif data_size >= self.config.batch_threshold and self.batch_processor:
            return OptimizationMode.BATCH
        elif data_size >= self.config.performance_threshold and self.unified_manager:
            return OptimizationMode.UNIFIED
        else:
            return OptimizationMode.FALLBACK
    
    def _optimize_rolling_operation(self, 
                                  data: Union[pd.Series, pd.DataFrame],
                                  operation_func: Callable,
                                  **kwargs) -> Any:
        """Optimize rolling operations."""
        # Extract rolling parameters from kwargs
        window = kwargs.get('window', 20)
        operation = kwargs.get('operation', 'mean')
        
        # Create rolling config
        config = RollingOperationConfig(
            operation=RollingOperationType(operation),
            window=window,
            min_periods=kwargs.get('min_periods'),
            center=kwargs.get('center', False)
        )
        
        return self.rolling_optimizer.single_rolling_operation(data, config)
    
    def _optimize_statistical_operation(self, 
                                      data: Union[pd.Series, pd.DataFrame],
                                      operation_func: Callable,
                                      **kwargs) -> Any:
        """Optimize statistical operations."""
        # Extract statistical parameters from kwargs
        operation = kwargs.get('operation', 'mean')
        window = kwargs.get('window')
        
        # Create statistical config
        config = StatisticalOperationConfig(
            operation=StatisticalOperationType(operation),
            window=window,
            min_periods=kwargs.get('min_periods'),
            axis=kwargs.get('axis', 0)
        )
        
        return self.statistical_optimizer.single_statistical_operation(data, config)
    
    def _optimize_batch_operation(self, 
                                data: Union[pd.Series, pd.DataFrame],
                                operation_func: Callable,
                                **kwargs) -> Any:
        """Optimize batch operations."""
        if not self.batch_processor:
            raise RuntimeError("Batch processor not available")
        
        # Use batch processor for multiple operations
        operations = kwargs.get('operations', [])
        if not operations:
            # Single operation fallback
            return operation_func(data)
        
        return self.batch_processor.process_features_batch(
            data=data,
            feature_generators=operations,
            parallel=self.config.enable_parallel
        )
    
    def _optimize_unified_operation(self, 
                                  data: Union[pd.Series, pd.DataFrame],
                                  operation_func: Callable,
                                  **kwargs) -> Any:
        """Optimize using Unified Vectorization Manager."""
        if not self.unified_manager:
            raise RuntimeError("Unified Vectorization Manager not available")
        
        # Determine operation type
        operation_type = kwargs.get('operation_type', OperationType.TECHNICAL_INDICATORS)
        
        # Create operation config
        op_config = OperationConfig(
            operation_type=operation_type,
            data_size=len(data),
            data_dimensions=data.shape if hasattr(data, 'shape') else (len(data),),
            memory_budget_mb=self.config.memory_limit_gb * 1024,
            parallel_workers=None
        )
        
        # Execute through unified manager
        result = self.unified_manager.optimize_operation(
            operation_type=operation_type,
            data=data,
            operation_func=operation_func,
            config=op_config
        )
        
        return result.result
    
    def _fallback_operation(self, 
                          data: Union[pd.Series, pd.DataFrame],
                          operation_func: Callable,
                          **kwargs) -> Any:
        """Fallback operation using standard implementations."""
        return operation_func(data, **kwargs)
    
    def batch_rolling_operations(self, 
                               data: Union[pd.Series, pd.DataFrame],
                               operations: List[str],
                               windows: List[int],
                               columns: Optional[List[str]] = None) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Convenience method for batch rolling operations."""
        return self.rolling_optimizer.batch_rolling_operations(
            data, 
            [RollingOperationConfig(
                operation=RollingOperationType(op),
                window=window
            ) for op in operations for window in windows]
        )
    
    def batch_statistical_operations(self, 
                                   data: Union[pd.Series, pd.DataFrame],
                                   operations: List[str],
                                   windows: Optional[List[int]] = None) -> Dict[str, Union[pd.Series, pd.DataFrame, float]]:
        """Convenience method for batch statistical operations."""
        configs = []
        for operation in operations:
            if windows:
                for window in windows:
                    configs.append(StatisticalOperationConfig(
                        operation=StatisticalOperationType(operation),
                        window=window
                    ))
            else:
                configs.append(StatisticalOperationConfig(
                    operation=StatisticalOperationType(operation)
                ))
        
        return self.statistical_optimizer.batch_statistical_operations(data, configs)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'unified_stats': self.performance_stats.copy(),
            'rolling_stats': self.rolling_optimizer.get_performance_stats(),
            'statistical_stats': self.statistical_optimizer.get_performance_stats()
        }
        
        # Calculate efficiency metrics
        total_ops = self.performance_stats['total_operations']
        if total_ops > 0:
            report['efficiency_metrics'] = {
                'optimization_hit_rate': self.performance_stats['optimization_hits'] / total_ops,
                'gpu_utilization': self.performance_stats['gpu_operations'] / total_ops,
                'average_operation_time': self.performance_stats['average_time_per_operation'],
                'memory_efficiency': self.performance_stats['memory_savings_mb'] / max(total_ops, 1)
            }
        
        return report
    
    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'rolling_operations': 0,
            'statistical_operations': 0,
            'batch_operations': 0,
            'unified_operations': 0,
            'fallback_operations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'average_time_per_operation': 0.0,
            'memory_savings_mb': 0.0,
            'optimization_hits': 0,
            'optimization_misses': 0
        }
        
        self.rolling_optimizer.reset_performance_stats()
        self.statistical_optimizer.reset_performance_stats()


# Convenience functions
def create_unified_optimizer(config: Optional[UnifiedOptimizationConfig] = None) -> UnifiedOptimizationWrapper:
    """Create a unified optimizer with specified configuration."""
    return UnifiedOptimizationWrapper(config)


def get_global_unified_optimizer() -> UnifiedOptimizationWrapper:
    """Get the global unified optimizer instance."""
    global _global_unified_optimizer
    if '_global_unified_optimizer' not in globals():
        _global_unified_optimizer = create_unified_optimizer()
    return _global_unified_optimizer


# Decorator for automatic optimization
def optimize_operation(operation_type: str = "auto", **config_kwargs):
    """
    Decorator for automatic operation optimization.
    
    Args:
        operation_type: Type of operation to optimize
        **config_kwargs: Configuration parameters
    """
    def decorator(func):
        def wrapper(self, data, *args, **kwargs):
            # Get or create optimizer
            if not hasattr(self, '_unified_optimizer'):
                config = UnifiedOptimizationConfig(**config_kwargs)
                self._unified_optimizer = create_unified_optimizer(config)
            
            # Optimize the operation
            return self._unified_optimizer.optimize_operation(
                operation_type=operation_type,
                data=data,
                operation_func=lambda x: func(self, x, *args, **kwargs),
                **kwargs
            )
        return wrapper
    return decorator