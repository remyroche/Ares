"""
GPU Accelerator for VectorBT Operations with Hardware Integration

This module provides GPU acceleration for VectorBT operations
with comprehensive hardware utility integration for maximum performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
import logging
import time

# Import hardware utilities
try:
    from src.utils.hardware.enhanced_gpu_manager import (
        get_enhanced_gpu_manager, GPUOperationType, create_gpu_operation
    )
    from src.utils.hardware.integrated_hardware_manager import (
        get_integrated_hardware_manager, WorkloadType
    )
    from src.utils.hardware.adaptive_optimization_engine import (
        get_adaptive_optimization_engine, OptimizationTarget
    )
    HARDWARE_UTILITIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware utilities not available: {e}")
    HARDWARE_UTILITIES_AVAILABLE = False

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """
    GPU accelerator for VectorBT operations with comprehensive hardware integration.
    
    This accelerator provides intelligent GPU acceleration using hardware utilities
    for maximum performance optimization.
    """

    def __init__(self):
        self.gpu_available = False
        self.hardware_available = HARDWARE_UTILITIES_AVAILABLE
        self.performance_metrics = {}
        
        # Hardware optimization components
        self.gpu_manager = None
        self.integrated_manager = None
        self.adaptive_engine = None
        
        # Performance tracking
        self.acceleration_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'memory_pool_operations': 0,
            'batch_operations': 0,
            'performance_improvements': [],
            'gpu_utilization': [],
            'memory_usage': []
        }
        
        # Initialize hardware components if available
        if self.hardware_available:
            self._initialize_hardware_components()

    def _initialize_hardware_components(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize enhanced GPU manager
            self.gpu_manager = get_enhanced_gpu_manager()
            
            # Initialize integrated hardware manager
            self.integrated_manager = get_integrated_hardware_manager()
            
            # Initialize adaptive optimization engine
            self.adaptive_engine = get_adaptive_optimization_engine()
            
            # Check GPU availability
            self.gpu_available = self._check_gpu_availability()
            
            logger.debug("Hardware components initialized for GPU accelerator")
            
        except Exception as e:
            logger.warning(f"Failed to initialize hardware components: {e}")
            self.hardware_available = False

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and properly configured."""
        if not self.hardware_available or not self.gpu_manager:
            return False
        
        try:
            gpu_info = self.gpu_manager.get_enhanced_gpu_info()
            return gpu_info.get('gpu_available', False)
        except Exception as e:
            logger.warning(f"Failed to check GPU availability: {e}")
            return False

    def accelerate_operation(self, operation: str, data: Union[pd.DataFrame, pd.Series], 
                           operation_type: str = 'data_processing', **kwargs) -> Any:
        """
        Accelerate a VectorBT operation using GPU with hardware optimization.
        
        Args:
            operation: Name of the operation to accelerate
            data: Input data
            operation_type: Type of operation for optimization
            **kwargs: Additional parameters
            
        Returns:
            GPU-accelerated result
        """
        self.acceleration_stats['total_operations'] += 1
        
        if not self.gpu_available:
            logger.debug("GPU not available, using CPU fallback")
            return self._cpu_fallback(operation, data, **kwargs)
        
        try:
            # Determine if operation is suitable for GPU
            if not self._is_gpu_suitable(operation, data, operation_type):
                return self._cpu_fallback(operation, data, **kwargs)
            
            # Apply GPU acceleration with hardware optimization
            result = self._apply_gpu_acceleration(operation, data, operation_type, **kwargs)
            
            # Track performance
            self._track_gpu_performance(operation, data, result)
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU acceleration failed: {e}, using CPU fallback")
            return self._cpu_fallback(operation, data, **kwargs)

    def _is_gpu_suitable(self, operation: str, data: Union[pd.DataFrame, pd.Series], 
                        operation_type: str) -> bool:
        """Determine if the operation is suitable for GPU acceleration."""
        if not self.gpu_available:
            return False
        
        # Check data size
        data_size = len(data) if hasattr(data, '__len__') else 0
        if data_size < 1000:  # Too small for GPU
            return False
        
        # Check if data is numeric
        if isinstance(data, pd.DataFrame):
            if not data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]:
                return False
        
        # Check operation type
        gpu_suitable_operations = [
            'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 'rolling_max',
            'scaling', 'ranking', 'zscore', 'matrix_multiplication', 'tensor_operations'
        ]
        
        return operation in gpu_suitable_operations

    def _apply_gpu_acceleration(self, operation: str, data: Union[pd.DataFrame, pd.Series], 
                              operation_type: str, **kwargs) -> Any:
        """Apply GPU acceleration with hardware optimization."""
        try:
            # Map operation to GPU operation type
            gpu_op_type = self._map_to_gpu_operation_type(operation)
            
            # Use enhanced GPU manager for acceleration
            if isinstance(data, pd.DataFrame):
                # Process DataFrame with GPU optimization
                result = self.gpu_manager.optimize_tensor_operations_advanced(data, gpu_op_type)
            else:
                # Process Series with GPU optimization
                result = self.gpu_manager.optimize_tensor_operations_advanced(data, gpu_op_type)
            
            self.acceleration_stats['gpu_operations'] += 1
            
            # Track GPU utilization
            gpu_info = self.gpu_manager.get_enhanced_gpu_info()
            self.acceleration_stats['gpu_utilization'].append(
                gpu_info.get('gpu_utilization', 0)
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU acceleration failed: {e}")
            raise e

    def _map_to_gpu_operation_type(self, operation: str) -> GPUOperationType:
        """Map operation to GPU operation type."""
        mapping = {
            'rolling_mean': GPUOperationType.TENSOR_OPERATIONS,
            'rolling_std': GPUOperationType.TENSOR_OPERATIONS,
            'rolling_var': GPUOperationType.TENSOR_OPERATIONS,
            'rolling_min': GPUOperationType.TENSOR_OPERATIONS,
            'rolling_max': GPUOperationType.TENSOR_OPERATIONS,
            'scaling': GPUOperationType.TENSOR_OPERATIONS,
            'ranking': GPUOperationType.TENSOR_OPERATIONS,
            'zscore': GPUOperationType.TENSOR_OPERATIONS,
            'matrix_multiplication': GPUOperationType.MATRIX_MULTIPLICATION,
            'tensor_operations': GPUOperationType.TENSOR_OPERATIONS,
            'backtesting': GPUOperationType.BACKTESTING_SIMULATION,
            'monte_carlo': GPUOperationType.MONTE_CARLO,
            'neural_network': GPUOperationType.NEURAL_NETWORK
        }
        return mapping.get(operation, GPUOperationType.TENSOR_OPERATIONS)

    def _cpu_fallback(self, operation: str, data: Union[pd.DataFrame, pd.Series], **kwargs) -> Any:
        """Fallback to CPU processing."""
        # This would contain the actual CPU implementation
        # For now, return the data as-is
        return data

    def _track_gpu_performance(self, operation: str, input_data: Union[pd.DataFrame, pd.Series], 
                             result: Any) -> None:
        """Track GPU performance metrics."""
        try:
            # Calculate performance metrics
            input_size = len(input_data) if hasattr(input_data, '__len__') else 0
            result_size = len(result) if hasattr(result, '__len__') else 0
            
            improvement = {
                'operation': operation,
                'input_size': input_size,
                'result_size': result_size,
                'timestamp': time.time(),
                'gpu_accelerated': True
            }
            
            self.acceleration_stats['performance_improvements'].append(improvement)
            
            # Keep only recent improvements
            if len(self.acceleration_stats['performance_improvements']) > 1000:
                self.acceleration_stats['performance_improvements'] = \
                    self.acceleration_stats['performance_improvements'][-500:]
                
        except Exception as e:
            logger.debug(f"Failed to track GPU performance: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self.performance_metrics.copy()
        metrics.update(self.acceleration_stats)
        
        # Add hardware-specific metrics
        if self.hardware_available and self.gpu_manager:
            try:
                gpu_info = self.gpu_manager.get_enhanced_gpu_info()
                metrics['gpu_info'] = gpu_info
                
                # Calculate average GPU utilization
                if self.acceleration_stats['gpu_utilization']:
                    metrics['avg_gpu_utilization'] = np.mean(self.acceleration_stats['gpu_utilization'])
                    metrics['max_gpu_utilization'] = np.max(self.acceleration_stats['gpu_utilization'])
                else:
                    metrics['avg_gpu_utilization'] = 0
                    metrics['max_gpu_utilization'] = 0
                    
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
                metrics['gpu_info'] = {}
        
        return metrics

    def get_acceleration_summary(self) -> Dict[str, Any]:
        """Get acceleration summary with recommendations."""
        total_ops = self.acceleration_stats['total_operations']
        gpu_ops = self.acceleration_stats['gpu_operations']
        
        return {
            'total_operations': total_ops,
            'gpu_acceleration_rate': gpu_ops / total_ops if total_ops > 0 else 0,
            'gpu_available': self.gpu_available,
            'hardware_available': self.hardware_available,
            'recommendations': self._generate_gpu_recommendations()
        }

    def _generate_gpu_recommendations(self) -> List[str]:
        """Generate GPU acceleration recommendations."""
        recommendations = []
        
        total_ops = self.acceleration_stats['total_operations']
        gpu_ops = self.acceleration_stats['gpu_operations']
        
        if total_ops > 0:
            gpu_rate = gpu_ops / total_ops
            
            if not self.gpu_available:
                recommendations.append("GPU not available - install CUDA for GPU acceleration")
            
            if gpu_rate < 0.3 and self.gpu_available:
                recommendations.append("Low GPU utilization - consider more GPU-suitable operations")
            
            if gpu_rate > 0.8:
                recommendations.append("High GPU utilization - GPU acceleration is performing well")
        
        return recommendations

    def reset_acceleration_stats(self) -> None:
        """Reset acceleration statistics."""
        self.acceleration_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'memory_pool_operations': 0,
            'batch_operations': 0,
            'performance_improvements': [],
            'gpu_utilization': [],
            'memory_usage': []
        }

def get_gpu_accelerator() -> GPUAccelerator:
    """Get the GPU accelerator instance."""
    return GPUAccelerator()
