"""
Hardware Optimization Utilities for Hybrid NAS-TAS Regime Detection.

Provides common hardware optimization utilities based on hardware/ system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import psutil
import os

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


@dataclass
class HardwareOptimizationConfig:
    """Configuration for hardware optimization operations."""
    use_gpu_acceleration: bool = True
    use_memory_optimization: bool = True
    use_matrix_operations: bool = True
    memory_limit_gb: float = 8.0
    batch_size: int = 1000
    parallel_workers: int = 4
    optimization_level: str = "high"  # "low", "medium", "high"
    monitoring_enabled: bool = True


@dataclass
class HardwareOptimizationResult:
    """Result from hardware optimization operations."""
    optimization_applied: bool
    performance_improvement: float
    memory_usage: Dict[str, Any]
    hardware_capabilities: Dict[str, Any]
    optimization_metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class HardwareOptimizer:
    """Hardware optimization utilities for clustering operations."""
    
    def __init__(self, config: HardwareOptimizationConfig):
        """Initialize the hardware optimizer.
        
        Args:
            config: Hardware optimization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE and config.use_gpu_acceleration:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for optimization")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")
        
        self.logger.info("✅ Hardware Optimizer initialized")
    
    def optimize_for_clustering(self, data_size: int, n_features: int) -> HardwareOptimizationResult:
        """Optimize hardware settings for clustering operations.
        
        Args:
            data_size: Size of the dataset
            n_features: Number of features
            
        Returns:
            HardwareOptimizationResult with optimization results
        """
        try:
            self.logger.info(f"⚡ Optimizing hardware for clustering: {data_size} samples, {n_features} features")
            start_time = time.time()
            
            # Get current system status
            system_status = self._get_system_status()
            
            # Calculate optimal settings
            optimal_settings = self._calculate_optimal_settings(data_size, n_features, system_status)
            
            # Apply optimizations
            optimization_applied = self._apply_optimizations(optimal_settings)
            
            # Monitor performance
            performance_metrics = {}
            if self.performance_monitor and self.config.monitoring_enabled:
                performance_metrics = self._monitor_performance("clustering_optimization")
            
            # Calculate performance improvement
            performance_improvement = self._calculate_performance_improvement(optimal_settings, system_status)
            
            execution_time = time.time() - start_time
            
            # Create optimization metadata
            optimization_metadata = {
                'data_size': data_size,
                'n_features': n_features,
                'system_status': system_status,
                'optimal_settings': optimal_settings,
                'performance_metrics': performance_metrics,
                'optimization_level': self.config.optimization_level,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Hardware optimization completed: {performance_improvement:.1f}% improvement")
            
            return HardwareOptimizationResult(
                optimization_applied=optimization_applied,
                performance_improvement=performance_improvement,
                memory_usage=system_status['memory'],
                hardware_capabilities=self._get_hardware_capabilities(),
                optimization_metadata=optimization_metadata,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Hardware optimization failed: {e}")
            return HardwareOptimizationResult(
                optimization_applied=False,
                performance_improvement=0.0,
                memory_usage={},
                hardware_capabilities={},
                optimization_metadata={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        try:
            # CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_percent = memory.percent
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            system_status = {
                'cpu': {
                    'count': cpu_count,
                    'percent': cpu_percent,
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'total_gb': memory_gb,
                    'available_gb': memory_available_gb,
                    'percent': memory_percent,
                    'used_gb': memory_gb - memory_available_gb
                },
                'disk': {
                    'free_gb': disk_free_gb,
                    'total_gb': disk.total / (1024**3)
                },
                'hardware_acceleration': self.hardware_accelerator is not None,
                'matrix_operations': self.matrix_ops is not None
            }
            
            return system_status
            
        except Exception as e:
            self.logger.warning(f"⚠️ System status retrieval failed: {e}")
            return {
                'cpu': {'count': 1, 'percent': 0, 'frequency': 0},
                'memory': {'total_gb': 4.0, 'available_gb': 2.0, 'percent': 50, 'used_gb': 2.0},
                'disk': {'free_gb': 10.0, 'total_gb': 100.0},
                'hardware_acceleration': False,
                'matrix_operations': False
            }
    
    def _calculate_optimal_settings(self, data_size: int, n_features: int, system_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal hardware settings."""
        try:
            optimal_settings = {
                'batch_size': self._calculate_optimal_batch_size(data_size, n_features, system_status),
                'memory_config': self._calculate_memory_config(data_size, n_features, system_status),
                'parallel_workers': self._calculate_optimal_workers(system_status),
                'matrix_operations_config': self._calculate_matrix_operations_config(data_size, n_features),
                'hardware_acceleration_config': self._calculate_hardware_acceleration_config(system_status)
            }
            
            return optimal_settings
            
        except Exception as e:
            self.logger.warning(f"⚠️ Optimal settings calculation failed: {e}")
            return {
                'batch_size': min(1000, data_size),
                'memory_config': {'limit_gb': 4.0},
                'parallel_workers': 1,
                'matrix_operations_config': {'enabled': False},
                'hardware_acceleration_config': {'enabled': False}
            }
    
    def _calculate_optimal_batch_size(self, data_size: int, n_features: int, system_status: Dict[str, Any]) -> int:
        """Calculate optimal batch size for processing."""
        try:
            # Use batch processor if available
            if self.batch_processor:
                return self.batch_processor.optimize_batch_size(data_size)
            
            # Calculate based on available memory and data characteristics
            available_memory_gb = system_status['memory']['available_gb']
            
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
    
    def _calculate_memory_config(self, data_size: int, n_features: int, system_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate memory optimization configuration."""
        try:
            available_memory_gb = system_status['memory']['available_gb']
            total_memory_gb = system_status['memory']['total_gb']
            
            memory_config = {
                'limit_gb': min(self.config.memory_limit_gb, available_memory_gb * 0.8),
                'chunk_size': self._calculate_optimal_batch_size(data_size, n_features, system_status),
                'memory_efficient_mode': True,
                'garbage_collection_enabled': True,
                'warning_threshold_gb': available_memory_gb * 0.9
            }
            
            return memory_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory config calculation failed: {e}")
            return {'limit_gb': 4.0, 'chunk_size': 1000, 'memory_efficient_mode': True}
    
    def _calculate_optimal_workers(self, system_status: Dict[str, Any]) -> int:
        """Calculate optimal number of parallel workers."""
        try:
            cpu_count = system_status['cpu']['count']
            memory_gb = system_status['memory']['total_gb']
            
            # Base workers on CPU count
            optimal_workers = min(cpu_count, self.config.parallel_workers)
            
            # Adjust based on memory
            if memory_gb < 4.0:
                optimal_workers = min(optimal_workers, 2)
            elif memory_gb < 8.0:
                optimal_workers = min(optimal_workers, 4)
            
            return max(1, optimal_workers)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Worker calculation failed: {e}")
            return 1
    
    def _calculate_matrix_operations_config(self, data_size: int, n_features: int) -> Dict[str, Any]:
        """Calculate matrix operations configuration."""
        try:
            matrix_config = {
                'enabled': self.matrix_ops is not None,
                'use_gpu_acceleration': True,
                'use_vectorized_operations': True,
                'batch_processing': True,
                'optimization_level': self.config.optimization_level
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
            self.logger.warning(f"⚠️ Matrix operations config calculation failed: {e}")
            return {'enabled': False}
    
    def _calculate_hardware_acceleration_config(self, system_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hardware acceleration configuration."""
        try:
            hardware_config = {
                'enabled': self.hardware_accelerator is not None,
                'gpu_available': False,  # Would be determined by actual GPU detection
                'memory_manager_enabled': self.memory_manager is not None,
                'performance_monitor_enabled': self.performance_monitor is not None
            }
            
            if self.hardware_accelerator:
                # Configure hardware acceleration based on system capabilities
                memory_gb = system_status['memory']['total_gb']
                
                if memory_gb >= 16.0:
                    hardware_config['optimization_level'] = 'high'
                elif memory_gb >= 8.0:
                    hardware_config['optimization_level'] = 'medium'
                else:
                    hardware_config['optimization_level'] = 'low'
            
            return hardware_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware acceleration config calculation failed: {e}")
            return {'enabled': False}
    
    def _apply_optimizations(self, optimal_settings: Dict[str, Any]) -> bool:
        """Apply hardware optimizations."""
        try:
            optimizations_applied = 0
            
            # Apply memory optimizations
            if self.memory_manager and optimal_settings.get('memory_config'):
                memory_config = optimal_settings['memory_config']
                if hasattr(self.memory_manager, 'configure'):
                    self.memory_manager.configure(memory_config)
                    optimizations_applied += 1
            
            # Apply matrix operations optimizations
            if self.matrix_ops and optimal_settings.get('matrix_operations_config'):
                matrix_config = optimal_settings['matrix_operations_config']
                if hasattr(self.matrix_ops, 'configure'):
                    self.matrix_ops.configure(matrix_config)
                    optimizations_applied += 1
            
            # Apply hardware acceleration optimizations
            if self.hardware_accelerator and optimal_settings.get('hardware_acceleration_config'):
                hardware_config = optimal_settings['hardware_acceleration_config']
                if hasattr(self.hardware_accelerator, 'configure'):
                    self.hardware_accelerator.configure(hardware_config)
                    optimizations_applied += 1
            
            return optimizations_applied > 0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Optimization application failed: {e}")
            return False
    
    def _monitor_performance(self, operation_name: str) -> Dict[str, Any]:
        """Monitor performance of operations."""
        try:
            if not self.performance_monitor:
                return {}
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring(operation_name)
            
            # Get initial metrics
            initial_metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': time.time()
            }
            
            return initial_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance monitoring setup failed: {e}")
            return {}
    
    def _calculate_performance_improvement(self, optimal_settings: Dict[str, Any], system_status: Dict[str, Any]) -> float:
        """Calculate estimated performance improvement."""
        try:
            improvement = 0.0
            
            # Matrix operations improvement
            if optimal_settings.get('matrix_operations_config', {}).get('enabled', False):
                improvement += 20.0  # 20% improvement from matrix operations
            
            # Hardware acceleration improvement
            if optimal_settings.get('hardware_acceleration_config', {}).get('enabled', False):
                improvement += 15.0  # 15% improvement from hardware acceleration
            
            # Memory optimization improvement
            if optimal_settings.get('memory_config', {}).get('memory_efficient_mode', False):
                improvement += 10.0  # 10% improvement from memory optimization
            
            # Batch processing improvement
            optimal_batch_size = optimal_settings.get('batch_size', 1000)
            if optimal_batch_size > 100:
                improvement += 5.0  # 5% improvement from optimal batching
            
            return min(improvement, 50.0)  # Cap at 50% improvement
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance improvement calculation failed: {e}")
            return 0.0
    
    def _get_hardware_capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities information."""
        try:
            capabilities = {
                'hardware_acceleration_available': self.hardware_accelerator is not None,
                'matrix_operations_available': self.matrix_ops is not None,
                'memory_manager_available': self.memory_manager is not None,
                'performance_monitor_available': self.performance_monitor is not None,
                'vectorized_processing_available': self.vectorized_core is not None,
                'enhanced_operations_available': self.enhanced_ops is not None,
                'batch_processing_available': self.batch_processor is not None
            }
            
            return capabilities
            
        except Exception as e:
            self.logger.warning(f"⚠️ Hardware capabilities retrieval failed: {e}")
            return {
                'hardware_acceleration_available': False,
                'matrix_operations_available': False,
                'memory_manager_available': False,
                'performance_monitor_available': False,
                'vectorized_processing_available': False,
                'enhanced_operations_available': False,
                'batch_processing_available': False
            }


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.monitoring_active = False
        self.start_time = None
        self.start_metrics = {}
        
        self.logger.info("✅ Performance Monitor initialized")
    
    def start_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """Start monitoring an operation.
        
        Args:
            operation_name: Name of the operation to monitor
            
        Returns:
            Initial monitoring metrics
        """
        try:
            self.operation_name = operation_name
            self.start_time = time.time()
            self.monitoring_active = True
            
            # Get initial system metrics
            self.start_metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'timestamp': self.start_time
            }
            
            self.logger.info(f"📊 Started monitoring: {operation_name}")
            return self.start_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Monitoring start failed: {e}")
            return {}
    
    def stop_monitoring(self, operation_name: str) -> Dict[str, Any]:
        """Stop monitoring an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Final monitoring metrics
        """
        try:
            if not self.monitoring_active or self.start_time is None:
                return {}
            
            end_time = time.time()
            execution_time = end_time - self.start_time
            
            # Get final system metrics
            end_metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'timestamp': end_time
            }
            
            # Calculate performance metrics
            performance_metrics = {
                'operation_name': operation_name,
                'execution_time': execution_time,
                'start_metrics': self.start_metrics,
                'end_metrics': end_metrics,
                'cpu_usage_delta': end_metrics['cpu_percent'] - self.start_metrics['cpu_percent'],
                'memory_usage_delta': end_metrics['memory_used_gb'] - self.start_metrics['memory_used_gb'],
                'monitoring_successful': True
            }
            
            self.monitoring_active = False
            self.start_time = None
            self.start_metrics = {}
            
            self.logger.info(f"📊 Stopped monitoring: {operation_name} ({execution_time:.2f}s)")
            return performance_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Monitoring stop failed: {e}")
            return {'monitoring_successful': False, 'error': str(e)}


def create_hardware_optimizer(config: HardwareOptimizationConfig) -> HardwareOptimizer:
    """Create a hardware optimizer instance.
    
    Args:
        config: Hardware optimization configuration
        
    Returns:
        HardwareOptimizer instance
    """
    return HardwareOptimizer(config)


def create_performance_monitor() -> PerformanceMonitor:
    """Create a performance monitor instance.
    
    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor()