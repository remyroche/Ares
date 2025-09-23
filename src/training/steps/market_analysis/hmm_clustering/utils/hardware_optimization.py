"""
Hardware optimization utilities for HMM clustering.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
import time

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareOptimizer:
    """Hardware optimization utilities for clustering operations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the hardware optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for optimization: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for optimization: {e}")

    def optimize_for_clustering(self, data_size: int, n_features: int) -> Dict[str, Any]:
        """Optimize hardware settings for clustering operations.

        Args:
            data_size: Size of the dataset
            n_features: Number of features

        Returns:
            Optimization configuration
        """
        try:
            optimization_config = {
                'data_size': data_size,
                'n_features': n_features,
                'optimization_applied': True,
                'hardware_status': {
                    'matrix_ops_available': self.matrix_ops is not None,
                    'hardware_acceleration_available': self.hardware_accelerator is not None,
                    'memory_manager_available': self.memory_manager is not None
                }
            }
            
            # Optimize batch size
            optimal_batch_size = self._calculate_optimal_batch_size(data_size, n_features)
            optimization_config['optimal_batch_size'] = optimal_batch_size
            
            # Optimize memory usage
            if self.memory_manager:
                memory_config = self._optimize_memory_usage(data_size, n_features)
                optimization_config['memory_config'] = memory_config
            
            # Optimize matrix operations
            if self.matrix_ops:
                matrix_config = self._optimize_matrix_operations(data_size, n_features)
                optimization_config['matrix_config'] = matrix_config
            
            self.logger.info(f"✅ Hardware optimization completed for dataset size {data_size}")
            return optimization_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization failed: {e}")
            return {
                'data_size': data_size,
                'n_features': n_features,
                'optimization_applied': False,
                'error': str(e)
            }

    def _calculate_optimal_batch_size(self, data_size: int, n_features: int) -> int:
        """Calculate optimal batch size for processing.

        Args:
            data_size: Size of the dataset
            n_features: Number of features

        Returns:
            Optimal batch size
        """
        try:
            # Use batch processor if available
            if self.batch_processor:
                return self.batch_processor.optimize_batch_size(data_size)
            
            # Calculate based on available memory and data characteristics
            if self.memory_manager:
                memory_info = self.memory_manager.get_memory_usage()
                available_memory_gb = memory_info.get('available_memory_gb', 4.0)
            else:
                available_memory_gb = 4.0  # Default assumption
            
            # Estimate memory per sample
            memory_per_sample = n_features * 8 / (1024**3)  # 8 bytes per float64
            
            # Calculate batch size to use 80% of available memory
            target_memory_usage = available_memory_gb * 0.8
            optimal_batch_size = int(target_memory_usage / memory_per_sample)
            
            # Apply reasonable bounds
            optimal_batch_size = max(100, min(optimal_batch_size, data_size))
            
            return optimal_batch_size
            
        except Exception as e:
            self.logger.warning(f"⚠️ Batch size calculation failed: {e}")
            return min(1000, data_size)

    def _optimize_memory_usage(self, data_size: int, n_features: int) -> Dict[str, Any]:
        """Optimize memory usage for clustering operations.

        Args:
            data_size: Size of the dataset
            n_features: Number of features

        Returns:
            Memory optimization configuration
        """
        try:
            memory_config = {
                'chunk_size': self._calculate_optimal_batch_size(data_size, n_features),
                'memory_efficient_mode': True,
                'garbage_collection_enabled': True
            }
            
            if self.memory_manager:
                # Get current memory usage
                memory_info = self.memory_manager.get_memory_usage()
                memory_config['current_memory_usage'] = memory_info
                
                # Set memory limits
                memory_config['memory_limit_gb'] = memory_info.get('total_memory_gb', 8.0) * 0.8
                memory_config['warning_threshold_gb'] = memory_config['memory_limit_gb'] * 0.9
            
            return memory_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return {'error': str(e)}

    def _optimize_matrix_operations(self, data_size: int, n_features: int) -> Dict[str, Any]:
        """Optimize matrix operations for clustering.

        Args:
            data_size: Size of the dataset
            n_features: Number of features

        Returns:
            Matrix operations optimization configuration
        """
        try:
            matrix_config = {
                'use_gpu_acceleration': True,
                'use_vectorized_operations': True,
                'batch_processing': True,
                'optimization_level': 'high'
            }
            
            if self.matrix_ops:
                # Configure matrix operations based on data characteristics
                if data_size > 10000:
                    matrix_config['use_sparse_operations'] = True
                else:
                    matrix_config['use_sparse_operations'] = False
                
                # Set operation-specific optimizations
                matrix_config['distance_calculation_method'] = 'euclidean'
                matrix_config['matrix_multiply_method'] = 'optimized'
                
                # Configure for clustering-specific operations
                matrix_config['clustering_optimizations'] = {
                    'centroid_calculation': 'vectorized',
                    'distance_matrix': 'chunked',
                    'silhouette_calculation': 'optimized'
                }
            
            return matrix_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Matrix operations optimization failed: {e}")
            return {'error': str(e)}

    def monitor_performance(self, operation_name: str) -> Dict[str, Any]:
        """Monitor performance of clustering operations.

        Args:
            operation_name: Name of the operation to monitor

        Returns:
            Performance monitoring configuration
        """
        try:
            monitoring_config = {
                'operation_name': operation_name,
                'monitoring_enabled': True,
                'hardware_monitoring': self.hardware_accelerator is not None,
                'matrix_ops_monitoring': self.matrix_ops is not None
            }
            
            if self.performance_monitor:
                # Start performance monitoring
                self.performance_monitor.start_monitoring(operation_name)
                monitoring_config['performance_monitor_active'] = True
            
            if self.memory_manager:
                # Start memory monitoring
                initial_memory = self.memory_manager.get_memory_usage()
                monitoring_config['initial_memory_usage'] = initial_memory
            
            return monitoring_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance monitoring setup failed: {e}")
            return {
                'operation_name': operation_name,
                'monitoring_enabled': False,
                'error': str(e)
            }

    def get_performance_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get performance summary for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Performance summary
        """
        try:
            summary = {
                'operation_name': operation_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if self.performance_monitor:
                # Get performance metrics
                performance_metrics = self.performance_monitor.stop_monitoring(operation_name)
                summary['performance_metrics'] = performance_metrics
            
            if self.memory_manager:
                # Get memory usage
                memory_usage = self.memory_manager.get_memory_usage()
                summary['memory_usage'] = memory_usage
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance summary generation failed: {e}")
            return {
                'operation_name': operation_name,
                'error': str(e)
            }


def create_hardware_optimizer(config: Dict[str, Any] = None) -> HardwareOptimizer:
    """Create a hardware optimizer instance.

    Args:
        config: Configuration dictionary

    Returns:
        HardwareOptimizer instance
    """
    return HardwareOptimizer(config)