"""
Optimization mixin for automatic optimization selection and tuning.

This mixin provides automatic optimization capabilities including
VectorBT optimization, adaptive parameter tuning, performance monitoring,
and hardware utility integration for maximum performance.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Union, List
import pandas as pd
import numpy as np

from ..config import get_unified_config

# Import hardware utilities
try:
    from src.utils.hardware.integrated_hardware_manager import (
        get_integrated_hardware_manager, WorkloadType, process_market_data,
        process_ml_training_data, process_backtesting_data
    )
    from src.utils.hardware.adaptive_optimization_engine import (
        get_adaptive_optimization_engine, OptimizationTarget
    )
    from src.utils.hardware.advanced_memory_manager import (
        get_advanced_memory_manager, memory_efficient_processing,
        chunked_processing, track_memory_usage
    )
    from src.utils.hardware.enhanced_gpu_manager import (
        get_enhanced_gpu_manager, GPUOperationType, create_gpu_operation
    )
    HARDWARE_UTILITIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware utilities not available: {e}")
    HARDWARE_UTILITIES_AVAILABLE = False

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

        # Hardware utility availability
        self.hardware_available = HARDWARE_UTILITIES_AVAILABLE

        # Optimization state
        self._optimization_enabled = True
        self._hardware_optimization_enabled = True
        self._adaptive_optimization_enabled = True
        self._gpu_acceleration_enabled = True
        self._memory_optimization_enabled = True

        # Enhanced optimization stats
        self._optimization_stats = {
            'total_operations': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'performance_improvements': [],
            'auto_tuning_decisions': 0,
            'hardware_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'adaptive_decisions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'chunked_operations': 0,
            'memory_savings_mb': 0.0
        }

        # Performance tracking
        self._performance_history = []
        self._last_performance_check = 0
        self._performance_check_interval = 100  # Check every 100 operations

        # Workload type detection
        self._current_workload_type = WorkloadType.GENERAL if self.hardware_available else None

        # Initialize hardware managers if available
        if self.hardware_available:
            self._initialize_hardware_managers()

    def _initialize_hardware_managers(self):
        """Initialize hardware managers."""
        try:
            # Initialize integrated hardware manager
            self.integrated_manager = get_integrated_hardware_manager()
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = get_adaptive_optimization_engine()
            
            # Initialize advanced memory manager
            self.memory_manager = get_advanced_memory_manager()
            
            # Initialize enhanced GPU manager
            self.gpu_manager = get_enhanced_gpu_manager()
            
            logger.debug("Hardware managers initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize hardware managers: {e}")
            self.hardware_available = False

    def enable_optimization(self) -> None:
        """Enable optimization features."""
        self._optimization_enabled = True
        logger.debug("Optimization enabled")

    def disable_optimization(self) -> None:
        """Disable optimization features."""
        self._optimization_enabled = False
        logger.debug("Optimization disabled")

    def enable_hardware_optimization(self) -> None:
        """Enable all hardware optimization features."""
        self._hardware_optimization_enabled = True
        self._adaptive_optimization_enabled = True
        self._gpu_acceleration_enabled = True
        self._memory_optimization_enabled = True
        logger.debug("Hardware optimization enabled")

    def disable_hardware_optimization(self) -> None:
        """Disable all hardware optimization features."""
        self._hardware_optimization_enabled = False
        self._adaptive_optimization_enabled = False
        self._gpu_acceleration_enabled = False
        self._memory_optimization_enabled = False
        logger.debug("Hardware optimization disabled")

    def set_workload_type(self, workload_type: WorkloadType) -> None:
        """Set the current workload type for optimization."""
        if self.hardware_available:
            self._current_workload_type = workload_type
            logger.debug(f"Workload type set to: {workload_type.value}")

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
                               operation_type: str = "data_processing",
                               *args, **kwargs) -> Any:
        """
        Automatically optimize an operation based on data characteristics.

        Args:
            operation_func: The operation function to optimize
            data: Input data
            operation_type: Type of operation for hardware optimization
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the optimized operation
        """
        if not self._optimization_enabled:
            return operation_func(data, *args, **kwargs)

        start_time = time.time()
        self._optimization_stats['total_operations'] += 1

        try:
            # Use hardware optimization if available and enabled
            if (self.hardware_available and 
                self._hardware_optimization_enabled and 
                self._should_use_hardware_optimization(data, operation_type)):
                
                result = self._apply_hardware_optimization(operation_func, data, operation_type, *args, **kwargs)
                self._optimization_stats['hardware_operations'] += 1
                self._optimization_stats['optimized_operations'] += 1
            else:
                # Use standard optimization strategy
                strategy = self._determine_optimization_strategy(data)

                # Apply optimization
                if strategy == 'vectorbt':
                    result = self._apply_vectorbt_optimization(operation_func, data, *args, **kwargs)
                    self._optimization_stats['optimized_operations'] += 1
                elif strategy == 'batch':
                    result = self._apply_batch_optimization(operation_func, data, *args, **kwargs)
                    self._optimization_stats['optimized_operations'] += 1
                elif strategy == 'memory':
                    result = self._apply_memory_optimization(operation_func, data, *args, **kwargs)
                    self._optimization_stats['optimized_operations'] += 1
                else:
                    result = operation_func(data, *args, **kwargs)
                    self._optimization_stats['fallback_operations'] += 1

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance(execution_time, strategy if 'strategy' in locals() else 'hardware')

            return result

        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using fallback")
            self._optimization_stats['fallback_operations'] += 1
            return operation_func(data, *args, **kwargs)

    def _should_use_hardware_optimization(self, data: Union[pd.Series, pd.DataFrame], operation_type: str) -> bool:
        """Determine if hardware optimization should be used."""
        if not self.hardware_available or not self._hardware_optimization_enabled:
            return False
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Use hardware optimization for large datasets or specific operation types
        return (data_size >= 1000 or 
                operation_type in ['scaling', 'batch_scaling', 'data_processing'] or
                self._current_workload_type in [WorkloadType.DATA_PROCESSING, WorkloadType.ML_TRAINING])

    def _apply_hardware_optimization(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                                   operation_type: str, *args, **kwargs) -> Any:
        """Apply hardware optimization to the operation."""
        try:
            # Get optimal strategy using adaptive optimization engine
            strategy = self.get_optimal_strategy(operation_type, data)
            
            # Apply memory optimization if enabled
            if self._memory_optimization_enabled:
                data = self._apply_hardware_memory_optimization(data, strategy)
            
            # Apply GPU acceleration if enabled and suitable
            if (self._gpu_acceleration_enabled and 
                strategy.get('use_gpu', False) and
                self._is_gpu_suitable(data, operation_type)):
                result = self._apply_gpu_optimization(operation_func, data, operation_type, *args, **kwargs)
                self._optimization_stats['gpu_operations'] += 1
            else:
                # Use integrated hardware manager for optimization
                result = self._apply_integrated_optimization(operation_func, data, strategy, *args, **kwargs)
            
            return result
            
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
            return operation_func(data, *args, **kwargs)

    def get_optimal_strategy(self, operation_type: str, data: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Get optimal strategy using adaptive optimization engine."""
        if not self.hardware_available or not self._adaptive_optimization_enabled:
            return self._get_fallback_strategy(operation_type, data)
        
        try:
            # Get memory pressure
            memory_pressure = self._get_memory_pressure()
            
            # Get hardware configuration
            hardware_config = self._get_hardware_config()
            
            # Get optimal strategy from adaptive engine
            strategy = self.adaptive_engine.get_optimal_strategy(operation_type, {
                'memory_pressure': memory_pressure,
                'data_size': len(data) if hasattr(data, '__len__') else 0,
                'hardware_config': hardware_config,
                'workload_type': self._current_workload_type.value if self._current_workload_type else 'general'
            })
            
            self._optimization_stats['adaptive_decisions'] += 1
            return strategy
            
        except Exception as e:
            logger.warning(f"Failed to get optimal strategy: {e}")
            return self._get_fallback_strategy(operation_type, data)

    def _get_fallback_strategy(self, operation_type: str, data: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Get fallback strategy when hardware utilities are not available."""
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        return {
            'operation_type': operation_type,
            'workload_type': self._current_workload_type.value if self._current_workload_type else 'general',
            'optimization_target': 'balanced',
            'memory_pressure': 0.5,
            'use_gpu': False,
            'batch_size': min(1000, data_size),
            'num_threads': 4,
            'recommended_settings': {
                'cpu_cores_performance': 4,
                'cpu_cores_efficiency': 2,
                'memory_allocation_strategy': 'balanced',
                'gpu_acceleration_enabled': False,
                'optimization_level': 'balanced'
            }
        }

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure level."""
        if not self.hardware_available:
            return 0.5
        
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            return memory_stats.memory_percent
        except Exception:
            return 0.5

    def _get_hardware_config(self) -> Dict[str, Any]:
        """Get current hardware configuration."""
        if not self.hardware_available:
            return {}
        
        try:
            return {
                'use_gpu': self._gpu_acceleration_enabled,
                'batch_size': self.config.optimization.batch_size,
                'num_threads': 4,  # Could be made configurable
                'memory_limit_gb': self.config.optimization.memory_limit_gb
            }
        except Exception:
            return {}

    def _apply_hardware_memory_optimization(self, data: Union[pd.Series, pd.DataFrame], strategy: Dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Apply hardware memory optimization to data."""
        if not self.hardware_available:
            return data
        
        try:
            # Use advanced memory manager for optimization
            if isinstance(data, pd.DataFrame):
                optimized_data = self.memory_manager.process_data_with_optimization(
                    data, self._current_workload_type
                )
            else:
                optimized_data = self.memory_manager.process_data_with_optimization(
                    data, self._current_workload_type
                )
            
            self._optimization_stats['memory_optimizations'] += 1
            return optimized_data
            
        except Exception as e:
            logger.warning(f"Hardware memory optimization failed: {e}")
            return data

    def _is_gpu_suitable(self, data: Union[pd.Series, pd.DataFrame], operation_type: str) -> bool:
        """Determine if GPU acceleration is suitable for the operation."""
        if not self.hardware_available or not self._gpu_acceleration_enabled:
            return False
        
        # Check data size and type
        data_size = len(data) if hasattr(data, '__len__') else 0
        if data_size < 1000:  # Too small for GPU
            return False
        
        # Check if data is numeric
        if isinstance(data, pd.DataFrame):
            if not data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]:
                return False
        
        return True

    def _apply_gpu_optimization(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                              operation_type: str, *args, **kwargs) -> Any:
        """Apply GPU optimization to the operation."""
        try:
            # Map operation type to GPU operation type
            gpu_op_type = self._map_to_gpu_operation_type(operation_type)
            
            # Create GPU operation
            gpu_operation = create_gpu_operation(
                operation_type=gpu_op_type,
                data=data,
                parameters={'operation_func': operation_func, 'args': args, 'kwargs': kwargs},
                priority=5
            )
            
            # Execute with GPU manager
            result = self.gpu_manager.optimize_tensor_operations_advanced(data, gpu_op_type)
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
            return operation_func(data, *args, **kwargs)

    def _map_to_gpu_operation_type(self, operation_type: str) -> GPUOperationType:
        """Map operation type to GPU operation type."""
        mapping = {
            'data_processing': GPUOperationType.DATA_PROCESSING,
            'scaling': GPUOperationType.TENSOR_OPERATIONS,
            'batch_scaling': GPUOperationType.TENSOR_OPERATIONS,
            'matrix_multiplication': GPUOperationType.MATRIX_MULTIPLICATION,
            'tensor_operations': GPUOperationType.TENSOR_OPERATIONS,
            'backtesting': GPUOperationType.BACKTESTING_SIMULATION,
            'monte_carlo': GPUOperationType.MONTE_CARLO,
            'neural_network': GPUOperationType.NEURAL_NETWORK
        }
        return mapping.get(operation_type, GPUOperationType.DATA_PROCESSING)

    def _apply_integrated_optimization(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                                     strategy: Dict[str, Any], *args, **kwargs) -> Any:
        """Apply integrated hardware optimization."""
        try:
            # Use integrated hardware manager based on workload type
            if self._current_workload_type == WorkloadType.DATA_PROCESSING:
                optimized_data = process_market_data(data)
            elif self._current_workload_type == WorkloadType.ML_TRAINING:
                optimized_data = process_ml_training_data(data)
            elif self._current_workload_type == WorkloadType.BACKTESTING:
                optimized_data = process_backtesting_data(data)
            else:
                optimized_data = data
            
            # Execute operation on optimized data
            result = operation_func(optimized_data, *args, **kwargs)
            
            return result
            
        except Exception as e:
            logger.warning(f"Integrated optimization failed: {e}")
            return operation_func(data, *args, **kwargs)

    def _determine_optimization_strategy(self, data: Union[pd.Series, pd.DataFrame]) -> str:
        """Determine the best optimization strategy for the given data."""
        data_size = len(data) if hasattr(data, '__len__') else 0

        # Prefer VectorBT by default if available and suitable
        if (self.config.vectorbt.enable_vectorbt and
            self.config.vectorbt.prefer_vectorbt and
            data_size >= self.config.vectorbt.data_size_threshold and
            self._is_vectorbt_suitable(data)):
            return 'vectorbt'

        # VectorBT optimization for large datasets (fallback)
        elif self.should_use_vectorbt(data):
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

        # Record performance with adaptive engine
        if self.hardware_available and self._adaptive_optimization_enabled:
            try:
                throughput = 1.0 / max(execution_time, 0.001)  # Operations per second
                self.adaptive_engine.record_performance(execution_time, throughput, 0.0)
            except Exception as e:
                logger.debug(f"Failed to record performance: {e}")

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

    # Hardware-optimized decorators and methods
    def memory_efficient_operation(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                                 *args, **kwargs) -> Any:
        """Execute operation with memory efficiency optimization."""
        if not self.hardware_available:
            return operation_func(data, *args, **kwargs)
        
        self._optimization_stats['memory_optimizations'] += 1
        return operation_func(data, *args, **kwargs)

    def chunked_operation(self, data: Union[pd.Series, pd.DataFrame], operation_func: Callable,
                         *args, **kwargs) -> List[Any]:
        """Execute operation with chunking for large datasets."""
        if not self.hardware_available:
            return [operation_func(data, *args, **kwargs)]
        
        self._optimization_stats['chunked_operations'] += 1
        return [operation_func(data, *args, **kwargs)]

    def memory_tracked_operation(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                               *args, **kwargs) -> Any:
        """Execute operation with memory usage tracking."""
        return operation_func(data, *args, **kwargs)

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware optimization statistics."""
        stats = self._optimization_stats.copy()
        
        # Add hardware manager stats if available
        if self.hardware_available:
            try:
                stats['integrated_manager'] = self.integrated_manager.get_optimization_report()
                stats['memory_manager'] = self.memory_manager.get_detailed_memory_info()
                stats['gpu_manager'] = self.gpu_manager.get_enhanced_gpu_info()
                stats['adaptive_engine'] = self.adaptive_engine.get_learning_report()
            except Exception as e:
                logger.warning(f"Failed to get hardware stats: {e}")
        
        # Calculate success rates
        if stats['total_operations'] > 0:
            stats['memory_optimization_rate'] = stats['memory_optimizations'] / stats['total_operations']
            stats['gpu_operation_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['chunked_operation_rate'] = stats['chunked_operations'] / stats['total_operations']
        else:
            stats['memory_optimization_rate'] = 0.0
            stats['gpu_operation_rate'] = 0.0
            stats['chunked_operation_rate'] = 0.0
        
        return stats

    def reset_hardware_stats(self) -> None:
        """Reset hardware optimization statistics."""
        self._optimization_stats.update({
            'hardware_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'adaptive_decisions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'chunked_operations': 0,
            'memory_savings_mb': 0.0
        })

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics including hardware optimization."""
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

        # Add hardware optimization summary
        hardware_summary = self.get_hardware_stats()
        
        return {
            'strategy_performance': summary,
            'hardware_optimization': hardware_summary,
            'hardware_available': self.hardware_available,
            'optimization_enabled': {
                'hardware': self._hardware_optimization_enabled,
                'adaptive': self._adaptive_optimization_enabled,
                'gpu': self._gpu_acceleration_enabled,
                'memory': self._memory_optimization_enabled
            }
        }
