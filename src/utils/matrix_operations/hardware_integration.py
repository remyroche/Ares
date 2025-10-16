"""
Hardware Integration for Matrix Operations

This module integrates all hardware optimization tools to provide
a comprehensive computation toolbox with automatic hardware detection
and optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from contextlib import contextmanager

# Conditional imports for hardware optimization tools
try:
    from ..hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from ..hardware.m1_memory_optimizer import M1MemoryOptimizer
    from ..hardware.m1_cpu_optimizer import M1CPUOptimizer
    from ..hardware.memory_optimization import (
        MemoryMonitor, MemoryConfig, get_memory_manager,
        memory_efficient, optimize_dataframe_dtypes, chunk_dataframe
    )
    HARDWARE_TOOLS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_TOOLS_AVAILABLE = False
    get_m1_gpu_manager = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    MemoryMonitor = None
    MemoryConfig = None
    get_memory_manager = None
    memory_efficient = None
    optimize_dataframe_dtypes = None
    chunk_dataframe = None

# Conditional imports for computation libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Configuration for hardware optimization."""
    # Memory settings
    max_memory_gb: float = 8.0
    memory_warning_threshold: float = 0.75
    memory_critical_threshold: float = 0.90

    # GPU settings
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8

    # CPU settings
    max_cpu_cores: Optional[int] = None
    use_performance_cores: bool = True

    # Optimization settings
    auto_optimize_dtypes: bool = True
    auto_chunk_large_data: bool = True
    chunk_size_threshold: int = 100000  # rows

    # Monitoring settings
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True

class HardwareOptimizedMatrixProcessor:
    """
    Hardware-optimized matrix processor that automatically detects
    and utilizes available hardware for optimal performance.
    """

    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or HardwareConfig()
        self.logger = logger.getChild('HardwareOptimizedMatrixProcessor')

        # Initialize hardware managers
        self._initialize_hardware_managers()

        # Performance tracking
        self.performance_metrics = {
            'operations_count': 0,
            'total_time': 0.0,
            'memory_usage': [],
            'gpu_usage': [],
            'cpu_usage': []
        }

        self.logger.info("🚀 Hardware-Optimized Matrix Processor initialized")
        self._log_hardware_info()

    def _initialize_hardware_managers(self):
        """Initialize all available hardware managers."""
        if not HARDWARE_TOOLS_AVAILABLE:
            self.logger.warning("⚠️ Hardware optimization tools not available")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.memory_monitor = None
            return

        # Initialize GPU manager
        if self.config.enable_gpu and get_m1_gpu_manager:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                if self.gpu_manager.is_m1 and self.gpu_manager.mps_available:
                    self.logger.info("✅ M1 GPU acceleration available")
                else:
                    self.logger.info("ℹ️ M1 GPU not available, using CPU")
            except Exception as e:
                self.logger.warning(f"⚠️ GPU manager initialization failed: {e}")
                self.gpu_manager = None
        else:
            self.gpu_manager = None

        # Initialize memory optimizer
        try:
            self.memory_optimizer = M1MemoryOptimizer(
                memory_limit_gb=self.config.max_memory_gb
            )
            self.logger.info("✅ M1 Memory optimizer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimizer initialization failed: {e}")
            self.memory_optimizer = None

        # Initialize CPU optimizer
        try:
            self.cpu_optimizer = M1CPUOptimizer()
            self.logger.info("✅ M1 CPU optimizer initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ CPU optimizer initialization failed: {e}")
            self.cpu_optimizer = None

        # Initialize memory monitor
        try:
            memory_config = MemoryConfig(
                max_memory_mb=self.config.max_memory_gb * 1024,
                warning_threshold=self.config.memory_warning_threshold,
                critical_threshold=self.config.memory_critical_threshold
            )
            self.memory_monitor = get_memory_manager(memory_config)
            self.logger.info("✅ Memory monitor initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Memory monitor initialization failed: {e}")
            self.memory_monitor = None

    def _log_hardware_info(self):
        """Log available hardware information."""
        if self.gpu_manager:
            gpu_info = self.gpu_manager.get_gpu_info()
            self.logger.info(f"🖥️ GPU Info: {gpu_info}")

        if self.cpu_optimizer:
            cpu_info = self.cpu_optimizer.get_cpu_info()
            self.logger.info(f"💻 CPU Info: {cpu_info}")

        if self.memory_monitor:
            memory_stats = self.memory_monitor.get_memory_stats()
            self.logger.info(f"🧠 Memory Stats: {memory_stats}")

    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = self.memory_monitor.get_usage_mb() if self.memory_monitor else 0

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.memory_monitor.get_usage_mb() if self.memory_monitor else 0

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            # Update performance metrics
            self.performance_metrics['operations_count'] += 1
            self.performance_metrics['total_time'] += execution_time
            self.performance_metrics['memory_usage'].append(memory_delta)

            if self.config.log_performance_metrics:
                self.logger.info(
                    f"📊 {operation_name}: {execution_time:.3f}s, "
                    f"Memory: {memory_delta:+.1f}MB"
                )

    def optimize_data_for_processing(self, data: Union['np.ndarray', 'pd.DataFrame']) -> Union['np.ndarray', 'pd.DataFrame']:
        """Optimize data for processing based on available hardware."""
        if not HARDWARE_TOOLS_AVAILABLE:
            return data

        with self.performance_context("data_optimization"):
            # Memory optimization
            if self.memory_optimizer and hasattr(data, 'dtype'):
                if isinstance(data, pd.DataFrame):
                    data = self.memory_optimizer.optimize_dataframe(data)
                elif isinstance(data, np.ndarray):
                    data = self.memory_optimizer.optimize_dataframe_memory(data)

            # GPU optimization
            if self.gpu_manager and self.gpu_manager.is_m1 and isinstance(data, np.ndarray):
                data = self.gpu_manager.optimize_tensor_operations(data)

            # Data type optimization
            if self.config.auto_optimize_dtypes and optimize_dataframe_dtypes and isinstance(data, pd.DataFrame):
                data = optimize_dataframe_dtypes(data)

            return data

    def chunk_data_if_needed(self, data: Union['np.ndarray', 'pd.DataFrame']) -> List[Union['np.ndarray', 'pd.DataFrame']]:
        """Chunk data if it's too large for available memory."""
        if not self.config.auto_chunk_large_data:
            return [data]

        # Check if data needs chunking
        if isinstance(data, pd.DataFrame):
            if len(data) <= self.config.chunk_size_threshold:
                return [data]

            # Use memory monitor to determine optimal chunk size
            if self.memory_monitor:
                available_memory_mb = self.memory_monitor.get_memory_stats().get('available_mb', 1000)
                # Estimate chunk size based on available memory
                estimated_chunk_size = min(
                    self.config.chunk_size_threshold,
                    max(1000, int(available_memory_mb * 0.1))  # Use 10% of available memory
                )
            else:
                estimated_chunk_size = self.config.chunk_size_threshold

            if chunk_dataframe:
                return chunk_dataframe(data, estimated_chunk_size, self.memory_monitor)
            else:
                # Simple chunking fallback
                chunks = []
                for i in range(0, len(data), estimated_chunk_size):
                    chunks.append(data.iloc[i:i + estimated_chunk_size])
                return chunks

        elif isinstance(data, np.ndarray):
            if len(data) <= self.config.chunk_size_threshold:
                return [data]

            # Simple chunking for numpy arrays
            chunks = []
            for i in range(0, len(data), self.config.chunk_size_threshold):
                chunks.append(data[i:i + self.config.chunk_size_threshold])
            return chunks

        return [data]

    def process_with_hardware_optimization(self,
                                         data: Union['np.ndarray', 'pd.DataFrame'],
                                         operation_func: Callable,
                                         *args, **kwargs) -> Any:
        """Process data with full hardware optimization."""
        with self.performance_context(f"hardware_optimized_{operation_func.__name__}"):
            # Optimize data
            optimized_data = self.optimize_data_for_processing(data)

            # Check if chunking is needed
            chunks = self.chunk_data_if_needed(optimized_data)

            if len(chunks) == 1:
                # Single chunk - process directly
                return operation_func(chunks[0], *args, **kwargs)
            else:
                # Multiple chunks - process in parallel if CPU optimizer available
                if self.cpu_optimizer:
                    return self._process_chunks_parallel(chunks, operation_func, *args, **kwargs)
                else:
                    return self._process_chunks_sequential(chunks, operation_func, *args, **kwargs)

    def _process_chunks_parallel(self,
                               chunks: List[Union['np.ndarray', 'pd.DataFrame']],
                               operation_func: Callable,
                               *args, **kwargs) -> Any:
        """Process chunks in parallel using CPU optimizer."""
        if not self.cpu_optimizer:
            return self._process_chunks_sequential(chunks, operation_func, *args, **kwargs)

        def process_chunk(chunk):
            return operation_func(chunk, *args, **kwargs)

        # Use M1-optimized thread pool
        with self.cpu_optimizer.create_optimized_thread_pool() as executor:
            results = list(executor.map(process_chunk, chunks))

        # Combine results
        if isinstance(chunks[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(chunks[0], np.ndarray):
            return np.concatenate(results, axis=0)
        else:
            return results

    def _process_chunks_sequential(self,
                                 chunks: List[Union['np.ndarray', 'pd.DataFrame']],
                                 operation_func: Callable,
                                 *args, **kwargs) -> Any:
        """Process chunks sequentially."""
        results = []
        for chunk in chunks:
            result = operation_func(chunk, *args, **kwargs)
            results.append(result)

            # Trigger garbage collection if memory pressure
            if self.memory_monitor and self.memory_monitor.should_trigger_gc():
                self.memory_monitor.trigger_gc()

        # Combine results
        if isinstance(chunks[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(chunks[0], np.ndarray):
            return np.concatenate(results, axis=0)
        else:
            return results

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'hardware_info': {},
            'performance_metrics': self.performance_metrics.copy(),
            'optimization_recommendations': []
        }

        # Hardware info
        if self.gpu_manager:
            report['hardware_info']['gpu'] = self.gpu_manager.get_gpu_info()
        if self.cpu_optimizer:
            report['hardware_info']['cpu'] = self.cpu_optimizer.get_cpu_info()
        if self.memory_monitor:
            report['hardware_info']['memory'] = self.memory_monitor.get_memory_stats()

        # Performance metrics
        if self.performance_metrics['operations_count'] > 0:
            avg_time = self.performance_metrics['total_time'] / self.performance_metrics['operations_count']
            report['performance_metrics']['average_operation_time'] = avg_time

        # Optimization recommendations
        if self.performance_metrics['memory_usage']:
            avg_memory_usage = sum(self.performance_metrics['memory_usage']) / len(self.performance_metrics['memory_usage'])
            if avg_memory_usage > 1000:  # More than 1GB average
                report['optimization_recommendations'].append(
                    "Consider reducing chunk size or enabling more aggressive memory optimization"
                )

        return report

    def optimized_standard_scaling(self, data: Union['np.ndarray', 'pd.DataFrame']) -> 'np.ndarray':
        """
        Perform hardware-optimized standard scaling (z-score normalization).

        Args:
            data: Input data as numpy array or pandas DataFrame

        Returns:
            Standardized data as numpy array
        """
        with self.performance_context("optimized_standard_scaling"):
            # Validate input data type
            if isinstance(data, dict):
                self.logger.error(f"❌ Hardware-optimized processing failed: Expected DataFrame or array but got dict")
                return np.array([])  # Return empty array as fallback

            if not isinstance(data, (pd.DataFrame, np.ndarray)):
                self.logger.error(f"❌ Hardware-optimized processing failed: Expected DataFrame or array but got {type(data)}")
                try:
                    data_array = np.array(data, dtype=np.float32)
                except Exception as e:
                    self.logger.error(f"❌ Cannot convert {type(data)} to array: {e}")
                    return np.array([])
            elif isinstance(data, pd.DataFrame):
                data_array = data.values.astype(np.float32)
            else:
                data_array = np.array(data, dtype=np.float32)

            # Check for invalid values
            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                self.logger.warning("⚠️ Input data contains NaN or infinite values, cleaning...")
                data_array = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Use hardware-optimized operations if available
            if self.gpu_manager and self.gpu_manager.is_m1 and TORCH_AVAILABLE:
                # GPU-accelerated scaling using PyTorch
                try:
                    # Convert to tensor
                    tensor_data = torch.from_numpy(data_array)

                    # Move to GPU if available
                    if torch.backends.mps.is_available():
                        tensor_data = tensor_data.to('mps')

                    # Compute mean and std
                    mean = torch.mean(tensor_data, dim=0, keepdim=True)
                    std = torch.std(tensor_data, dim=0, keepdim=True)

                    # Avoid division by zero
                    std = torch.where(std == 0, torch.ones_like(std), std)

                    # Standardize
                    scaled_tensor = (tensor_data - mean) / std

                    # Convert back to numpy
                    scaled_data = scaled_tensor.cpu().numpy()

                    self.logger.info("✅ GPU-accelerated standard scaling completed")
                    return scaled_data

                except Exception as e:
                    self.logger.warning(f"⚠️ GPU scaling failed, falling back to CPU: {e}")

            # CPU-based scaling (fallback or primary)
            try:
                # Compute mean and standard deviation
                mean = np.mean(data_array, axis=0, keepdims=True)
                std = np.std(data_array, axis=0, keepdims=True)

                # Avoid division by zero
                std = np.where(std == 0, 1.0, std)

                # Standardize
                scaled_data = (data_array - mean) / std

                self.logger.info("✅ CPU-based standard scaling completed")
                return scaled_data

            except Exception as e:
                self.logger.error(f"❌ Standard scaling failed: {e}")
                # Return original data as fallback
                return data_array

    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()

        if self.memory_monitor:
            self.memory_monitor.trigger_gc()

        self.logger.info("🧹 Hardware optimization cleanup completed")

# Global instance
_hardware_processor = None

def get_hardware_optimized_processor(config: Optional[HardwareConfig] = None) -> HardwareOptimizedMatrixProcessor:
    """Get or create the global hardware-optimized processor."""
    global _hardware_processor
    if _hardware_processor is None:
        _hardware_processor = HardwareOptimizedMatrixProcessor(config)
    return _hardware_processor

def hardware_optimized(operation_name: str = None):
    """Decorator for hardware-optimized operations."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            processor = get_hardware_optimized_processor()
            operation = operation_name or func.__name__

            # Extract data from args (assuming first argument is data)
            if args:
                data = args[0]
                remaining_args = args[1:]
            else:
                data = None
                remaining_args = args

            if data is not None:
                return processor.process_with_hardware_optimization(
                    data, func, *remaining_args, **kwargs
                )
            else:
                # No data to optimize, just run the function
                return func(*args, **kwargs)

        return wrapper
    return decorator

# Convenience functions for common operations
def optimize_matrix_operation(data: Union['np.ndarray', 'pd.DataFrame'],
                            operation_func: Callable,
                            *args, **kwargs) -> Any:
    """Optimize a matrix operation using available hardware."""
    processor = get_hardware_optimized_processor()
    return processor.process_with_hardware_optimization(data, operation_func, *args, **kwargs)

def get_hardware_performance_report() -> Dict[str, Any]:
    """Get hardware performance report."""
    processor = get_hardware_optimized_processor()
    return processor.get_performance_report()

def cleanup_hardware_resources():
    """Cleanup hardware resources."""
    global _hardware_processor
    if _hardware_processor:
        _hardware_processor.cleanup()
        _hardware_processor = None

# Backward compatibility alias
HardwareOptimizedOperations = HardwareOptimizedMatrixProcessor
