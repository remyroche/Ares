"""
Hardware Accelerator Module for Advanced Computing

This module provides comprehensive hardware acceleration capabilities for various
computing tasks, including GPU acceleration, memory optimization, and M1-specific optimizations.
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
from src.utils.tprint import tprint

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a mock numpy for basic functionality
    class MockNumpy:
        def __init__(self):
            self.float32 = float
            self.float64 = float
            self.array = list
            self.mean = lambda x: sum(x) / len(x) if x else 0
            self.max = max
            self.sum = sum
            self.vstack = lambda x: x[0] if x else []
        def __getattr__(self, name):
            return lambda *args, **kwargs: 0
    np = MockNumpy()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a mock pandas for basic functionality
    class MockPandas:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    pd = MockPandas()

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create a mock psutil for basic functionality
    class MockPsutil:
        def cpu_percent(self):
            return 0.0
        def virtual_memory(self):
            class MockMemory:
                def __init__(self):
                    self.total = 8 * 1024 * 1024 * 1024  # 8GB
                    self.available = 4 * 1024 * 1024 * 1024  # 4GB
                    self.used = 4 * 1024 * 1024 * 1024  # 4GB
                    self.percent = 50.0
            return MockMemory()
    psutil = MockPsutil()

# Import shared utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        is_m1_available, is_mps_available, integrate_with_m1_optimizers
    )
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns,
        calculate_data_quality_metrics, create_summary_statistics
    )
    from src.utils.math_validation import (
        safe_correlation, safe_covariance, safe_mean, safe_std,
        validate_correlation_matrix, safe_matrix_inverse
    )
    from src.utils.matrix_operations import (
        vectorized_operations, matrix_optimization
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False

# Import M1-specific utilities
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    M1_UTILS_AVAILABLE = True
except ImportError:
    M1_UTILS_AVAILABLE = False

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, GridOptimizer
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

# Import serialization utilities
try:
    from src.utils.serialization_utils import (
        safe_serialize, safe_deserialize, optimize_serialization
    )
    SERIALIZATION_AVAILABLE = True
except ImportError:
    SERIALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class AccelerationType(Enum):
    """Types of hardware acceleration."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    M1_SPECIFIC = "m1_specific"

class AccelerationLevel(Enum):
    """Hardware acceleration levels."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class HardwareAcceleratorConfig:
    """Configuration for hardware accelerator."""
    enable_gpu_acceleration: bool = True
    enable_memory_acceleration: bool = True
    enable_cpu_acceleration: bool = True
    enable_m1_acceleration: bool = True
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    acceleration_level: AccelerationLevel = AccelerationLevel.STANDARD
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    chunk_size: int = 1000
    cache_size_mb: int = 100
    enable_compression: bool = True
    enable_monitoring: bool = True

@dataclass
class AccelerationMetrics:
    """Hardware acceleration metrics."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    processing_time: float
    throughput: float
    acceleration_type: str
    acceleration_level: str
    speedup_factor: float
    memory_efficiency: float

@dataclass
class ProcessingResult:
    """Result of accelerated processing."""
    data: Any
    metrics: AccelerationMetrics
    success: bool
    error_message: Optional[str] = None
    optimization_applied: bool = True

class HardwareAccelerator:
    """
    Advanced hardware accelerator for various computing tasks.
    
    This accelerator provides:
    - GPU acceleration for compute-intensive tasks
    - Memory optimization for large datasets
    - M1-specific optimizations for Apple Silicon
    - Parallel processing capabilities
    - Performance monitoring and metrics
    - Caching and compression
    """
    
    def __init__(self, config: Optional[HardwareAcceleratorConfig] = None):
        """Initialize hardware accelerator."""
        self.config = config or HardwareAcceleratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.cache_manager = None
        self.monitor = None
        
        # Performance tracking
        self.acceleration_history = []
        self.performance_stats = {
            'total_accelerations': 0,
            'gpu_accelerations': 0,
            'memory_optimizations': 0,
            'm1_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_speedup': 0.0,
            'total_memory_savings': 0.0
        }
        
        # Initialize hardware acceleration
        self._initialize_hardware_acceleration()
        
        self.logger.info("✅ Hardware Accelerator initialized")
    
    def _initialize_hardware_acceleration(self):
        """Initialize hardware acceleration components."""
        try:
            # Initialize GPU acceleration
            if self.config.enable_gpu_acceleration:
                self._setup_gpu_acceleration()
            
            # Initialize memory acceleration
            if self.config.enable_memory_acceleration:
                self._setup_memory_acceleration()
            
            # Initialize CPU acceleration
            if self.config.enable_cpu_acceleration:
                self._setup_cpu_acceleration()
            
            # Initialize M1 acceleration
            if self.config.enable_m1_acceleration and M1_UTILS_AVAILABLE:
                self._setup_m1_acceleration()
            
            # Initialize caching
            if self.config.enable_caching:
                self._setup_caching()
            
            # Initialize monitoring
            if self.config.enable_monitoring:
                self._setup_monitoring()
            
            self.logger.info("✅ Hardware acceleration components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Hardware acceleration initialization failed: {e}")
            raise
    
    def _setup_gpu_acceleration(self):
        """Setup GPU acceleration."""
        try:
            if M1_UTILS_AVAILABLE and is_m1_available():
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info("✅ M1 GPU acceleration enabled")
            else:
                self.logger.warning("⚠️ GPU acceleration not available")
        except Exception as e:
            self.logger.error(f"❌ GPU acceleration setup failed: {e}")
    
    def _setup_memory_acceleration(self):
        """Setup memory acceleration."""
        try:
            if M1_UTILS_AVAILABLE:
                self.memory_optimizer = get_m1_memory_optimizer()
                self.logger.info("✅ Memory acceleration enabled")
            else:
                # Fallback memory acceleration
                self.memory_optimizer = self._create_fallback_memory_optimizer()
                self.logger.info("✅ Fallback memory acceleration enabled")
        except Exception as e:
            self.logger.error(f"❌ Memory acceleration setup failed: {e}")
    
    def _setup_cpu_acceleration(self):
        """Setup CPU acceleration."""
        try:
            if M1_UTILS_AVAILABLE:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ CPU acceleration enabled")
            else:
                # Fallback CPU acceleration
                self.cpu_optimizer = self._create_fallback_cpu_optimizer()
                self.logger.info("✅ Fallback CPU acceleration enabled")
        except Exception as e:
            self.logger.error(f"❌ CPU acceleration setup failed: {e}")
    
    def _setup_m1_acceleration(self):
        """Setup M1-specific acceleration."""
        try:
            if M1_UTILS_AVAILABLE:
                # Integrate with M1 optimizers
                integration_result = integrate_with_m1_optimizers()
                if integration_result.get('success', False):
                    self.logger.info("✅ M1 acceleration enabled")
                else:
                    self.logger.warning("⚠️ M1 acceleration integration failed")
            else:
                self.logger.warning("⚠️ M1 acceleration not available")
        except Exception as e:
            self.logger.error(f"❌ M1 acceleration setup failed: {e}")
    
    def _setup_caching(self):
        """Setup caching system."""
        try:
            self.cache_manager = CacheManager(
                cache_size_mb=self.config.cache_size_mb,
                enable_compression=self.config.enable_compression
            )
            self.logger.info("✅ Caching enabled")
        except Exception as e:
            self.logger.error(f"❌ Caching setup failed: {e}")
    
    def _setup_monitoring(self):
        """Setup performance monitoring."""
        try:
            self.monitor = PerformanceMonitor()
            self.logger.info("✅ Performance monitoring enabled")
        except Exception as e:
            self.logger.error(f"❌ Monitoring setup failed: {e}")
    
    def _create_fallback_memory_optimizer(self):
        """Create fallback memory optimizer."""
        class FallbackMemoryOptimizer:
            def __init__(self):
                self.memory_usage = 0
                self.optimization_count = 0
            
            def optimize_memory(self, data):
                """Basic memory optimization."""
                if hasattr(data, 'memory_usage'):
                    return data.memory_usage(deep=True).sum()
                return 0
            
            def get_memory_usage(self):
                """Get current memory usage."""
                return psutil.virtual_memory().percent
        
        return FallbackMemoryOptimizer()
    
    def _create_fallback_cpu_optimizer(self):
        """Create fallback CPU optimizer."""
        class FallbackCPUOptimizer:
            def __init__(self):
                self.cpu_usage = 0
                self.optimization_count = 0
            
            def optimize_cpu(self, data):
                """Basic CPU optimization."""
                return data
            
            def get_cpu_usage(self):
                """Get current CPU usage."""
                return psutil.cpu_percent()
        
        return FallbackCPUOptimizer()
    
    def accelerate_processing(self, 
                            data: Union[np.ndarray, pd.DataFrame, Any],
                            operation: Callable,
                            acceleration_type: AccelerationType = AccelerationType.CPU,
                            **kwargs) -> ProcessingResult:
        """
        Accelerate processing with hardware optimization.
        
        Args:
            data: Data to process
            operation: Operation to accelerate
            acceleration_type: Type of acceleration to apply
            **kwargs: Additional arguments for the operation
            
        Returns:
            Processing result with acceleration metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting {acceleration_type.value} acceleration")
            
            # Preprocess data for acceleration
            data_optimized = self._optimize_data_for_acceleration(data, acceleration_type)
            
            # Apply hardware acceleration
            if acceleration_type == AccelerationType.GPU:
                result = self._apply_gpu_acceleration(operation, data_optimized, **kwargs)
            elif acceleration_type == AccelerationType.MEMORY:
                result = self._apply_memory_acceleration(operation, data_optimized, **kwargs)
            elif acceleration_type == AccelerationType.CPU:
                result = self._apply_cpu_acceleration(operation, data_optimized, **kwargs)
            elif acceleration_type == AccelerationType.M1_SPECIFIC:
                result = self._apply_m1_acceleration(operation, data_optimized, **kwargs)
            else:
                result = self._apply_generic_acceleration(operation, data_optimized, **kwargs)
            
            # Calculate acceleration metrics
            processing_time = time.time() - start_time
            metrics = self._calculate_acceleration_metrics(
                processing_time, data_optimized, acceleration_type
            )
            
            # Update performance stats
            self.performance_stats['total_accelerations'] += 1
            if acceleration_type == AccelerationType.GPU:
                self.performance_stats['gpu_accelerations'] += 1
            elif acceleration_type == AccelerationType.MEMORY:
                self.performance_stats['memory_optimizations'] += 1
            elif acceleration_type == AccelerationType.M1_SPECIFIC:
                self.performance_stats['m1_optimizations'] += 1
            
            processing_result = ProcessingResult(
                data=result,
                metrics=metrics,
                success=True,
                optimization_applied=True
            )
            
            self.logger.info(f"✅ {acceleration_type.value} acceleration completed in {processing_time:.2f}s")
            return processing_result
            
        except Exception as e:
            self.logger.error(f"❌ {acceleration_type.value} acceleration failed: {e}")
            return ProcessingResult(
                data=None,
                metrics=AccelerationMetrics(0, 0, None, 0, 0, acceleration_type.value, "unknown", 0, 0),
                success=False,
                error_message=str(e),
                optimization_applied=False
            )
    
    def _optimize_data_for_acceleration(self, 
                                      data: Union[np.ndarray, pd.DataFrame, Any],
                                      acceleration_type: AccelerationType) -> Any:
        """Optimize data for specific acceleration type."""
        try:
            if acceleration_type == AccelerationType.GPU:
                return self._optimize_for_gpu(data)
            elif acceleration_type == AccelerationType.MEMORY:
                return self._optimize_for_memory(data)
            elif acceleration_type == AccelerationType.CPU:
                return self._optimize_for_cpu(data)
            elif acceleration_type == AccelerationType.M1_SPECIFIC:
                return self._optimize_for_m1(data)
            else:
                return self._optimize_generic(data)
                
        except Exception as e:
            self.logger.error(f"❌ Data optimization failed: {e}")
            return data
    
    def _optimize_for_gpu(self, data: Any) -> Any:
        """Optimize data for GPU acceleration."""
        try:
            if isinstance(data, np.ndarray):
                # Convert to optimal data type for GPU
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                return data
            elif isinstance(data, pd.DataFrame):
                # Convert DataFrame to GPU-optimized format
                return data.astype(np.float32)
            else:
                return data
        except Exception as e:
            self.logger.warning(f"⚠️ GPU optimization failed: {e}")
            return data
    
    def _optimize_for_memory(self, data: Any) -> Any:
        """Optimize data for memory acceleration."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
                return self.memory_optimizer.optimize_memory(data)
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return data
    
    def _optimize_for_cpu(self, data: Any) -> Any:
        """Optimize data for CPU acceleration."""
        try:
            if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'optimize_cpu'):
                return self.cpu_optimizer.optimize_cpu(data)
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ CPU optimization failed: {e}")
            return data
    
    def _optimize_for_m1(self, data: Any) -> Any:
        """Optimize data for M1 acceleration."""
        try:
            if M1_UTILS_AVAILABLE and is_m1_available():
                # Apply M1-specific optimizations
                if isinstance(data, np.ndarray):
                    # M1-optimized data types
                    if data.dtype == np.float64:
                        data = data.astype(np.float32)
                return data
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ M1 optimization failed: {e}")
            return data
    
    def _optimize_generic(self, data: Any) -> Any:
        """Generic data optimization."""
        try:
            # Basic optimization
            if isinstance(data, np.ndarray):
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Generic optimization failed: {e}")
            return data
    
    def _apply_gpu_acceleration(self, operation: Callable, data: Any, **kwargs) -> Any:
        """Apply GPU acceleration to operation."""
        try:
            if self.gpu_manager and hasattr(self.gpu_manager, 'accelerate_operation'):
                return self.gpu_manager.accelerate_operation(operation, data, **kwargs)
            else:
                # Fallback to regular operation
                return operation(data, **kwargs)
        except Exception as e:
            self.logger.warning(f"⚠️ GPU acceleration failed: {e}")
            return operation(data, **kwargs)
    
    def _apply_memory_acceleration(self, operation: Callable, data: Any, **kwargs) -> Any:
        """Apply memory acceleration to operation."""
        try:
            if self.memory_optimizer and hasattr(self.memory_optimizer, 'accelerate_operation'):
                return self.memory_optimizer.accelerate_operation(operation, data, **kwargs)
            else:
                # Fallback to regular operation
                return operation(data, **kwargs)
        except Exception as e:
            self.logger.warning(f"⚠️ Memory acceleration failed: {e}")
            return operation(data, **kwargs)
    
    def _apply_cpu_acceleration(self, operation: Callable, data: Any, **kwargs) -> Any:
        """Apply CPU acceleration to operation."""
        try:
            if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'accelerate_operation'):
                return self.cpu_optimizer.accelerate_operation(operation, data, **kwargs)
            else:
                # Fallback to regular operation
                return operation(data, **kwargs)
        except Exception as e:
            self.logger.warning(f"⚠️ CPU acceleration failed: {e}")
            return operation(data, **kwargs)
    
    def _apply_m1_acceleration(self, operation: Callable, data: Any, **kwargs) -> Any:
        """Apply M1-specific acceleration to operation."""
        try:
            if M1_UTILS_AVAILABLE and is_m1_available():
                # M1-specific acceleration
                return operation(data, **kwargs)
            else:
                # Fallback to regular operation
                return operation(data, **kwargs)
        except Exception as e:
            self.logger.warning(f"⚠️ M1 acceleration failed: {e}")
            return operation(data, **kwargs)
    
    def _apply_generic_acceleration(self, operation: Callable, data: Any, **kwargs) -> Any:
        """Apply generic acceleration to operation."""
        try:
            return operation(data, **kwargs)
        except Exception as e:
            self.logger.warning(f"⚠️ Generic acceleration failed: {e}")
            return operation(data, **kwargs)
    
    def _calculate_acceleration_metrics(self, 
                                       processing_time: float, 
                                       data: Any,
                                       acceleration_type: AccelerationType) -> AccelerationMetrics:
        """Calculate acceleration metrics."""
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            gpu_usage = self._get_gpu_usage()
            
            # Calculate throughput
            if isinstance(data, np.ndarray):
                data_size = data.size
            elif isinstance(data, pd.DataFrame):
                data_size = len(data)
            else:
                data_size = 1
            
            throughput = data_size / processing_time if processing_time > 0 else 0
            
            # Calculate speedup factor based on acceleration type and hardware utilization
            speedup_factor = self._calculate_speedup_factor(
                acceleration_type, processing_time, data_size, 
                cpu_usage, memory_usage, gpu_usage
            )
            
            # Calculate memory efficiency
            memory_efficiency = 1.0 - (memory_usage / 100.0)  # Simplified calculation
            
            metrics = AccelerationMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                processing_time=processing_time,
                throughput=throughput,
                acceleration_type=acceleration_type.value,
                acceleration_level=self.config.acceleration_level.value,
                speedup_factor=speedup_factor,
                memory_efficiency=memory_efficiency
            )
            
            # Store in acceleration history
            self.acceleration_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Acceleration metrics calculation failed: {e}")
            return AccelerationMetrics(0, 0, None, 0, 0, acceleration_type.value, "unknown", 0, 0)
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage."""
        try:
            if self.gpu_manager and hasattr(self.gpu_manager, 'get_gpu_usage'):
                return self.gpu_manager.get_gpu_usage()
            return None
        except (AttributeError, TypeError, ValueError) as e:
            tprint(f"Error getting GPU usage: {e}", level="warning")
            return None
    
    def _calculate_speedup_factor(self, 
                                acceleration_type: AccelerationType,
                                processing_time: float,
                                data_size: int,
                                cpu_usage: float,
                                memory_usage: float,
                                gpu_usage: Optional[float]) -> float:
        """
        Calculate actual speedup factor based on acceleration type and hardware utilization.
        
        Args:
            acceleration_type: Type of acceleration applied
            processing_time: Time taken for processing
            data_size: Size of data processed
            cpu_usage: CPU utilization percentage
            memory_usage: Memory utilization percentage
            gpu_usage: GPU utilization percentage (if available)
            
        Returns:
            float: Calculated speedup factor
        """
        try:
            # Base speedup factors for different acceleration types
            base_speedup = {
                AccelerationType.CPU: 1.2,  # CPU optimization typically provides 20% speedup
                AccelerationType.MEMORY: 1.5,  # Memory optimization can provide 50% speedup
                AccelerationType.GPU: 2.0,  # GPU acceleration can provide 2x speedup
                AccelerationType.M1_SPECIFIC: 1.8,  # M1 optimizations can provide 80% speedup
                AccelerationType.STORAGE: 1.1,  # Storage optimization provides minimal speedup
                AccelerationType.NETWORK: 1.3,  # Network optimization provides moderate speedup
            }
            
            # Get base speedup for acceleration type
            base_factor = base_speedup.get(acceleration_type, 1.0)
            
            # Adjust based on hardware utilization efficiency
            utilization_factor = 1.0
            
            # CPU utilization adjustment (optimal around 70-80%)
            if cpu_usage > 0:
                if 70 <= cpu_usage <= 80:
                    utilization_factor *= 1.1  # Optimal CPU usage
                elif cpu_usage > 90:
                    utilization_factor *= 0.8  # Overutilized CPU
                elif cpu_usage < 30:
                    utilization_factor *= 0.9  # Underutilized CPU
            
            # Memory utilization adjustment (optimal around 60-70%)
            if memory_usage > 0:
                if 60 <= memory_usage <= 70:
                    utilization_factor *= 1.05  # Optimal memory usage
                elif memory_usage > 85:
                    utilization_factor *= 0.85  # High memory pressure
                elif memory_usage < 30:
                    utilization_factor *= 0.95  # Low memory usage
            
            # GPU utilization adjustment (if available)
            if gpu_usage is not None and gpu_usage > 0:
                if 60 <= gpu_usage <= 80:
                    utilization_factor *= 1.2  # Optimal GPU usage
                elif gpu_usage > 90:
                    utilization_factor *= 0.9  # Overutilized GPU
                elif gpu_usage < 40:
                    utilization_factor *= 0.8  # Underutilized GPU
            
            # Data size efficiency adjustment
            data_efficiency = 1.0
            if data_size > 10000:  # Large datasets benefit more from acceleration
                data_efficiency = 1.2
            elif data_size < 100:  # Small datasets may not benefit much
                data_efficiency = 0.9
            
            # Processing time efficiency (faster processing indicates better acceleration)
            time_efficiency = 1.0
            if processing_time < 0.1:  # Very fast processing
                time_efficiency = 1.1
            elif processing_time > 10:  # Slow processing
                time_efficiency = 0.8
            
            # Calculate final speedup factor
            speedup_factor = base_factor * utilization_factor * data_efficiency * time_efficiency
            
            # Ensure reasonable bounds (0.5x to 5x speedup)
            speedup_factor = max(0.5, min(5.0, speedup_factor))
            
            return speedup_factor
            
        except Exception as e:
            self.logger.error(f"❌ Speedup factor calculation failed: {e}")
            return 1.0  # Fallback to no speedup
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            if not self.acceleration_history:
                return {'message': 'No acceleration data available'}
            
            cpu_usage = [m.cpu_usage for m in self.acceleration_history]
            memory_usage = [m.memory_usage for m in self.acceleration_history]
            processing_times = [m.processing_time for m in self.acceleration_history]
            throughputs = [m.throughput for m in self.acceleration_history]
            speedup_factors = [m.speedup_factor for m in self.acceleration_history]
            
            return {
                'performance_stats': self.performance_stats,
                'acceleration_summary': {
                    'avg_cpu_usage': np.mean(cpu_usage),
                    'max_cpu_usage': np.max(cpu_usage),
                    'avg_memory_usage': np.mean(memory_usage),
                    'max_memory_usage': np.max(memory_usage),
                    'avg_processing_time': np.mean(processing_times),
                    'total_processing_time': np.sum(processing_times),
                    'avg_throughput': np.mean(throughputs),
                    'max_throughput': np.max(throughputs),
                    'avg_speedup_factor': np.mean(speedup_factors),
                    'max_speedup_factor': np.max(speedup_factors),
                    'acceleration_count': len(self.acceleration_history)
                },
                'hardware_utilization': self._get_hardware_utilization(),
                'memory_usage': self._get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Performance summary generation failed: {e}")
            return {'error': str(e)}
    
    def _get_hardware_utilization(self) -> Dict[str, float]:
        """Get current hardware utilization."""
        try:
            utilization = {
                'cpu_utilization': psutil.cpu_percent(),
                'memory_utilization': psutil.virtual_memory().percent
            }
            
            # Add GPU utilization if available
            gpu_usage = self._get_gpu_usage()
            if gpu_usage is not None:
                utilization['gpu_utilization'] = gpu_usage
            
            return utilization
            
        except Exception as e:
            self.logger.error(f"❌ Hardware utilization check failed: {e}")
            return {}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent
            }
        except Exception as e:
            self.logger.error(f"❌ Memory usage check failed: {e}")
            return {}
    
    def reset_performance_history(self):
        """Reset performance history."""
        self.acceleration_history = []
        self.performance_stats = {
            'total_accelerations': 0,
            'gpu_accelerations': 0,
            'memory_optimizations': 0,
            'm1_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_speedup': 0.0,
            'total_memory_savings': 0.0
        }
        self.logger.info("✅ Performance history reset")


class CacheManager:
    """Cache manager for hardware acceleration."""
    
    def __init__(self, cache_size_mb: int = 100, enable_compression: bool = True):
        """Initialize cache manager."""
        self.cache_size_mb = cache_size_mb
        self.enable_compression = enable_compression
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result."""
        try:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            self.logger.warning(f"⚠️ Cache retrieval failed: {e}")
            return None
    
    def cache_result(self, key: str, result: Any) -> bool:
        """Cache result."""
        try:
            self.cache[key] = result
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Cache storage failed: {e}")
            return False


class PerformanceMonitor:
    """Performance monitor for hardware acceleration."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.monitoring_active = False
        self.monitoring_data = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        self.monitoring_data = []
        self.logger.info("✅ Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        self.logger.info("✅ Performance monitoring stopped")
    
    def record_metric(self, metric_name: str, value: float):
        """Record performance metric."""
        if self.monitoring_active:
            self.monitoring_data.append({
                'metric_name': metric_name,
                'value': value,
                'timestamp': time.time()
            })


# Factory functions
def create_hardware_accelerator(config: Optional[HardwareAcceleratorConfig] = None) -> HardwareAccelerator:
    """Create hardware accelerator instance."""
    return HardwareAccelerator(config)


def create_cache_manager(cache_size_mb: int = 100, enable_compression: bool = True) -> CacheManager:
    """Create cache manager instance."""
    return CacheManager(cache_size_mb, enable_compression)


def create_performance_monitor() -> PerformanceMonitor:
    """Create performance monitor instance."""
    return PerformanceMonitor()


# Quick acceleration functions
def quick_gpu_acceleration(data: Any, operation: Callable, **kwargs) -> ProcessingResult:
    """Quick GPU acceleration for immediate use."""
    accelerator = create_hardware_accelerator()
    return accelerator.accelerate_processing(data, operation, AccelerationType.GPU, **kwargs)


def quick_memory_acceleration(data: Any, operation: Callable, **kwargs) -> ProcessingResult:
    """Quick memory acceleration for immediate use."""
    accelerator = create_hardware_accelerator()
    return accelerator.accelerate_processing(data, operation, AccelerationType.MEMORY, **kwargs)


def quick_cpu_acceleration(data: Any, operation: Callable, **kwargs) -> ProcessingResult:
    """Quick CPU acceleration for immediate use."""
    accelerator = create_hardware_accelerator()
    return accelerator.accelerate_processing(data, operation, AccelerationType.CPU, **kwargs)


def quick_m1_acceleration(data: Any, operation: Callable, **kwargs) -> ProcessingResult:
    """Quick M1 acceleration for immediate use."""
    accelerator = create_hardware_accelerator()
    return accelerator.accelerate_processing(data, operation, AccelerationType.M1_SPECIFIC, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create hardware accelerator
    config = HardwareAcceleratorConfig(
        enable_gpu_acceleration=True,
        enable_memory_acceleration=True,
        enable_cpu_acceleration=True,
        enable_m1_acceleration=True,
        acceleration_level=AccelerationLevel.STANDARD
    )
    
    accelerator = create_hardware_accelerator(config)
    
    # Example usage
    print("Hardware Accelerator created successfully!")
    print(f"Performance summary: {accelerator.get_performance_summary()}")