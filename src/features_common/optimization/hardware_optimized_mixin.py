"""
Hardware-Optimized Mixin for features_common.

This module provides enhanced optimization capabilities by integrating
hardware utilities from src/utils/hardware/ for maximum performance.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, Union, List
import pandas as pd
import numpy as np

# Import hardware utilities
try:
    from src.utils.hardware.integrated_hardware_manager import (
        get_integrated_hardware_manager, IntegratedHardwareConfig,
        WorkloadType, process_market_data, process_ml_training_data,
        process_backtesting_data
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

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class HardwareOptimizedMixin:
    """
    Enhanced optimization mixin with hardware utility integration.
    
    This mixin provides advanced optimization capabilities by integrating
    with hardware utilities for intelligent memory management, adaptive
    optimization, GPU acceleration, and performance monitoring.
    """

    def __init__(self, *args, **kwargs):
        """Initialize hardware-optimized mixin."""
        super().__init__(*args, **kwargs)
        
        # Get unified configuration
        self.config = get_unified_config()
        
        # Hardware utility availability
        self.hardware_available = HARDWARE_UTILITIES_AVAILABLE
        
        # Initialize hardware managers if available
        if self.hardware_available:
            self._initialize_hardware_managers()
        
        # Enhanced optimization state
        self._hardware_optimization_enabled = True
        self._adaptive_optimization_enabled = True
        self._gpu_acceleration_enabled = True
        self._memory_optimization_enabled = True
        
        # Hardware-specific performance tracking
        self._hardware_stats = {
            'hardware_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'adaptive_decisions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'chunked_operations': 0,
            'memory_savings_mb': 0.0,
            'performance_improvements': []
        }
        
        # Workload type detection
        self._current_workload_type = WorkloadType.GENERAL
        
        logger.debug("Hardware-optimized mixin initialized")

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
        self._current_workload_type = workload_type
        logger.debug(f"Workload type set to: {workload_type.value}")

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
                'workload_type': self._current_workload_type.value
            })
            
            self._hardware_stats['adaptive_decisions'] += 1
            return strategy
            
        except Exception as e:
            logger.warning(f"Failed to get optimal strategy: {e}")
            return self._get_fallback_strategy(operation_type, data)

    def _get_fallback_strategy(self, operation_type: str, data: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Get fallback strategy when hardware utilities are not available."""
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        return {
            'operation_type': operation_type,
            'workload_type': self._current_workload_type.value,
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

    def hardware_optimized_operation(self,
                                   operation_func: Callable,
                                   data: Union[pd.Series, pd.DataFrame],
                                   operation_type: str = "data_processing",
                                   *args, **kwargs) -> Any:
        """
        Execute operation with full hardware optimization.
        
        Args:
            operation_func: The operation function to optimize
            data: Input data
            operation_type: Type of operation for optimization strategy
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation
            
        Returns:
            Result of the optimized operation
        """
        if not self.hardware_available or not self._hardware_optimization_enabled:
            return operation_func(data, *args, **kwargs)
        
        start_time = time.time()
        self._hardware_stats['hardware_operations'] += 1
        
        try:
            # Get optimal strategy
            strategy = self.get_optimal_strategy(operation_type, data)
            
            # Apply memory optimization if enabled
            if self._memory_optimization_enabled:
                data = self._apply_memory_optimization(data, strategy)
            
            # Apply GPU acceleration if enabled and suitable
            if (self._gpu_acceleration_enabled and 
                strategy.get('use_gpu', False) and
                self._is_gpu_suitable(data, operation_type)):
                result = self._apply_gpu_optimization(operation_func, data, operation_type, *args, **kwargs)
                self._hardware_stats['gpu_operations'] += 1
            else:
                # Use integrated hardware manager for optimization
                result = self._apply_integrated_optimization(operation_func, data, strategy, *args, **kwargs)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_hardware_performance(execution_time, strategy, operation_type)
            
            return result
            
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}, using fallback")
            return operation_func(data, *args, **kwargs)

    def _apply_memory_optimization(self, data: Union[pd.Series, pd.DataFrame], strategy: Dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """Apply memory optimization to data."""
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
            
            self._hardware_stats['memory_optimizations'] += 1
            return optimized_data
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
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

    def _track_hardware_performance(self, execution_time: float, strategy: Dict[str, Any], operation_type: str) -> None:
        """Track hardware performance metrics."""
        # Record performance with adaptive engine
        if self.hardware_available and self._adaptive_optimization_enabled:
            try:
                throughput = 1.0 / max(execution_time, 0.001)  # Operations per second
                self.adaptive_engine.record_performance(execution_time, throughput, 0.0)
            except Exception as e:
                logger.debug(f"Failed to record performance: {e}")
        
        # Update hardware stats
        self._hardware_stats['performance_improvements'].append({
            'execution_time': execution_time,
            'strategy': strategy.get('optimization_target', 'unknown'),
            'operation_type': operation_type,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self._hardware_stats['performance_improvements']) > 1000:
            self._hardware_stats['performance_improvements'] = self._hardware_stats['performance_improvements'][-500:]

    @memory_efficient_processing
    def memory_efficient_operation(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                                 *args, **kwargs) -> Any:
        """Execute operation with memory efficiency optimization."""
        if not self.hardware_available:
            return operation_func(data, *args, **kwargs)
        
        self._hardware_stats['memory_optimizations'] += 1
        return operation_func(data, *args, **kwargs)

    @chunked_processing(chunk_size_mb=50.0)
    def chunked_operation(self, data: Union[pd.Series, pd.DataFrame], operation_func: Callable,
                         *args, **kwargs) -> List[Any]:
        """Execute operation with chunking for large datasets."""
        if not self.hardware_available:
            return [operation_func(data, *args, **kwargs)]
        
        self._hardware_stats['chunked_operations'] += 1
        return [operation_func(data, *args, **kwargs)]

    @track_memory_usage
    def memory_tracked_operation(self, operation_func: Callable, data: Union[pd.Series, pd.DataFrame],
                               *args, **kwargs) -> Any:
        """Execute operation with memory usage tracking."""
        return operation_func(data, *args, **kwargs)

    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware optimization statistics."""
        stats = self._hardware_stats.copy()
        
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
        if stats['hardware_operations'] > 0:
            stats['memory_optimization_rate'] = stats['memory_optimizations'] / stats['hardware_operations']
            stats['gpu_operation_rate'] = stats['gpu_operations'] / stats['hardware_operations']
            stats['chunked_operation_rate'] = stats['chunked_operations'] / stats['hardware_operations']
        else:
            stats['memory_optimization_rate'] = 0.0
            stats['gpu_operation_rate'] = 0.0
            stats['chunked_operation_rate'] = 0.0
        
        return stats

    def reset_hardware_stats(self) -> None:
        """Reset hardware optimization statistics."""
        self._hardware_stats = {
            'hardware_operations': 0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'adaptive_decisions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'chunked_operations': 0,
            'memory_savings_mb': 0.0,
            'performance_improvements': []
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including hardware metrics."""
        # Get base performance summary
        base_summary = super().get_performance_summary() if hasattr(super(), 'get_performance_summary') else {}
        
        # Add hardware-specific metrics
        hardware_summary = self.get_hardware_stats()
        
        return {
            **base_summary,
            'hardware_optimization': hardware_summary,
            'hardware_available': self.hardware_available,
            'optimization_enabled': {
                'hardware': self._hardware_optimization_enabled,
                'adaptive': self._adaptive_optimization_enabled,
                'gpu': self._gpu_acceleration_enabled,
                'memory': self._memory_optimization_enabled
            }
        }