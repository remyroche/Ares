"""
Optimization mixin for automatic optimization selection and tuning.

This mixin provides automatic optimization capabilities including
VectorBT optimization, adaptive parameter tuning, and performance monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Union
import pandas as pd
import numpy as np

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class OptimizationMixin:
    """
    Mixin class providing automatic optimization capabilities.
    
    This mixin can be added to any class to provide automatic optimization
    selection, parameter tuning, and performance monitoring.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize optimization mixin."""
        super().__init__(*args, **kwargs)
        
        # Get unified configuration
        self.config = get_unified_config()
        
        # Optimization state
        self._optimization_enabled = True
        self._optimization_stats = {
            'total_operations': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'performance_improvements': [],
            'auto_tuning_decisions': 0
        }
        
        # Performance tracking
        self._performance_history = []
        self._last_performance_check = 0
        self._performance_check_interval = 100  # Check every 100 operations
    
    def enable_optimization(self) -> None:
        """Enable optimization features."""
        self._optimization_enabled = True
        logger.debug("Optimization enabled")
    
    def disable_optimization(self) -> None:
        """Disable optimization features."""
        self._optimization_enabled = False
        logger.debug("Optimization disabled")
    
    def is_optimization_enabled(self) -> bool:
        """Check if optimization is enabled."""
        return self._optimization_enabled
    
    def should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """Determine if VectorBT should be used for the given data."""
        if not self._optimization_enabled:
            return False
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        return self.config.should_use_vectorbt(data_size)
    
    def get_optimization_level(self) -> str:
        """Get the current optimization level."""
        return self.config.vectorbt.optimization_level
    
    def auto_optimize_operation(self, 
                               operation_func: Callable,
                               data: Union[pd.Series, pd.DataFrame],
                               *args, **kwargs) -> Any:
        """
        Automatically optimize an operation based on data characteristics.
        
        Args:
            operation_func: The operation function to optimize
            data: Input data
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation
            
        Returns:
            Result of the optimized operation
            
        Raises:
            RuntimeError: If optimization fails and fallback is not available
        """
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if not self._optimization_enabled:
            if TPRINT_AVAILABLE:
                tprint("⚠️  [OptimizationMixin] Optimization disabled, using direct operation", color="yellow")
            return operation_func(data, *args, **kwargs)
        
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [OptimizationMixin] Starting auto-optimization for {operation_func.__name__ if hasattr(operation_func, '__name__') else 'operation'}", color="cyan")
        
        start_time = time.time()
        self._optimization_stats['total_operations'] += 1
        
        try:
            # Determine optimization strategy
            if TPRINT_AVAILABLE:
                tprint("🔍 [OptimizationMixin] Determining optimization strategy", color="blue")
            
            strategy = self._determine_optimization_strategy(data)
            
            if TPRINT_AVAILABLE:
                tprint(f"📊 [OptimizationMixin] Selected strategy: {strategy}", color="green")
            
            # Apply optimization
            if strategy == 'vectorbt':
                if TPRINT_AVAILABLE:
                    tprint("🚀 [OptimizationMixin] Applying VectorBT optimization", color="green")
                result = self._apply_vectorbt_optimization(operation_func, data, *args, **kwargs)
                self._optimization_stats['optimized_operations'] += 1
            elif strategy == 'batch':
                if TPRINT_AVAILABLE:
                    tprint("📦 [OptimizationMixin] Applying batch optimization", color="green")
                result = self._apply_batch_optimization(operation_func, data, *args, **kwargs)
                self._optimization_stats['optimized_operations'] += 1
            elif strategy == 'memory':
                if TPRINT_AVAILABLE:
                    tprint("💾 [OptimizationMixin] Applying memory optimization", color="green")
                result = self._apply_memory_optimization(operation_func, data, *args, **kwargs)
                self._optimization_stats['optimized_operations'] += 1
            else:
                if TPRINT_AVAILABLE:
                    tprint("🔄 [OptimizationMixin] Using direct operation (no optimization)", color="yellow")
                result = operation_func(data, *args, **kwargs)
                self._optimization_stats['fallback_operations'] += 1
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_performance(execution_time, strategy)
            
            if TPRINT_AVAILABLE:
                tprint(f"✅ [OptimizationMixin] Operation completed in {execution_time:.4f}s using {strategy}", color="green")
            
            return result
            
        except Exception as e:
            error_msg = f"Optimization failed: {e}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [OptimizationMixin] {error_msg}, using fallback", color="red")
            logger.warning(error_msg)
            self._optimization_stats['fallback_operations'] += 1
            
            try:
                # Attempt fallback
                if TPRINT_AVAILABLE:
                    tprint("🔄 [OptimizationMixin] Attempting fallback operation", color="yellow")
                result = operation_func(data, *args, **kwargs)
                if TPRINT_AVAILABLE:
                    tprint("✅ [OptimizationMixin] Fallback operation successful", color="green")
                return result
            except Exception as fallback_error:
                error_msg = f"Both optimization and fallback failed: {fallback_error}"
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [OptimizationMixin] {error_msg}", color="red")
                self._log_error(error_msg)
                raise RuntimeError(error_msg) from fallback_error
    
    def _determine_optimization_strategy(self, data: Union[pd.Series, pd.DataFrame]) -> str:
        """Determine the best optimization strategy for the given data."""
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # VectorBT optimization for large datasets
        if self.should_use_vectorbt(data):
            return 'vectorbt'
        
        # Batch processing for medium datasets
        elif (data_size >= self.config.optimization.batch_size and 
              self.config.optimization.enable_batch_processing):
            return 'batch'
        
        # Memory optimization for any dataset
        elif self.config.optimization.memory_efficient:
            return 'memory'
        
        # No optimization
        else:
            return 'none'
    
    def _apply_vectorbt_optimization(self, operation_func: Callable, 
                                   data: Union[pd.Series, pd.DataFrame],
                                   *args, **kwargs) -> Any:
        """Apply VectorBT optimization to the operation."""
        # This would integrate with VectorBT optimization components
        # For now, we'll just call the original function
        # In a full implementation, this would use VectorBT's optimized functions
        return operation_func(data, *args, **kwargs)
    
    def _apply_batch_optimization(self, operation_func: Callable,
                                data: Union[pd.Series, pd.DataFrame],
                                *args, **kwargs) -> Any:
        """Apply batch processing optimization to the operation."""
        # Process data in batches for memory efficiency
        if isinstance(data, pd.DataFrame):
            batch_size = self.config.optimization.batch_size
            results = []
            
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                batch_result = operation_func(batch, *args, **kwargs)
                results.append(batch_result)
            
            # Combine results
            if results and isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif results and isinstance(results[0], pd.Series):
                return pd.concat(results, ignore_index=True)
            else:
                return results
        else:
            return operation_func(data, *args, **kwargs)
    
    def _apply_memory_optimization(self, operation_func: Callable,
                                 data: Union[pd.Series, pd.DataFrame],
                                 *args, **kwargs) -> Any:
        """Apply memory optimization to the operation."""
        # Optimize data types for memory efficiency
        if self.config.optimization.optimize_data_types:
            data = self._optimize_data_types(data)
        
        # Apply memory pooling if enabled
        if self.config.vectorbt.enable_memory_pooling:
            data = self._apply_memory_pooling(data)
        
        return operation_func(data, *args, **kwargs)
    
    def _optimize_data_types(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data types for memory efficiency."""
        if isinstance(data, pd.Series):
            if data.dtype == 'float64':
                # Check if float32 is sufficient
                if (data.min() >= np.finfo(np.float32).min and 
                    data.max() <= np.finfo(np.float32).max):
                    return data.astype(np.float32)
        elif isinstance(data, pd.DataFrame):
            optimized_data = data.copy()
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
            return optimized_data
        
        return data
    
    def _apply_memory_pooling(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply memory pooling to the data."""
        # This would integrate with a memory pooling system
        # For now, we'll just return the data as-is
        return data
    
    def _track_performance(self, execution_time: float, strategy: str) -> None:
        """Track performance metrics for optimization decisions."""
        self._performance_history.append({
            'execution_time': execution_time,
            'strategy': strategy,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]
        
        # Check if we should auto-tune parameters
        if (self.config.optimization.auto_tune_parameters and 
            len(self._performance_history) >= 10):
            self._auto_tune_parameters()
    
    def _auto_tune_parameters(self) -> None:
        """Automatically tune parameters based on performance history."""
        if not self._performance_history:
            return
        
        recent_history = self._performance_history[-10:]
        avg_time = np.mean([h['execution_time'] for h in recent_history])
        
        # If performance is below threshold, try more aggressive optimization
        if avg_time > self.config.optimization.performance_threshold:
            self._optimization_stats['auto_tuning_decisions'] += 1
            logger.debug(f"Auto-tuning parameters due to slow performance: {avg_time:.3f}s")
            
            # This would implement actual parameter tuning
            # For now, we'll just log the decision
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self._optimization_stats.copy()
        
        # Calculate optimization rate
        if stats['total_operations'] > 0:
            stats['optimization_rate'] = stats['optimized_operations'] / stats['total_operations']
            stats['fallback_rate'] = stats['fallback_operations'] / stats['total_operations']
        else:
            stats['optimization_rate'] = 0
            stats['fallback_rate'] = 0
        
        # Add performance metrics
        if self._performance_history:
            recent_times = [h['execution_time'] for h in self._performance_history[-10:]]
            stats['avg_execution_time'] = np.mean(recent_times)
            stats['std_execution_time'] = np.std(recent_times)
            stats['min_execution_time'] = np.min(recent_times)
            stats['max_execution_time'] = np.max(recent_times)
        
        return stats
    
    def reset_optimization_stats(self) -> None:
        """Reset optimization statistics."""
        self._optimization_stats = {
            'total_operations': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'performance_improvements': [],
            'auto_tuning_decisions': 0
        }
        self._performance_history = []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self._performance_history:
            return {'message': 'No performance data available'}
        
        recent_history = self._performance_history[-50:]  # Last 50 operations
        
        # Group by strategy
        strategy_stats = {}
        for record in recent_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(record['execution_time'])
        
        # Calculate statistics for each strategy
        summary = {}
        for strategy, times in strategy_stats.items():
            summary[strategy] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        
        return summary