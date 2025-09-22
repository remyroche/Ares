"""
Hardware Optimization Integration for Model Training

This module provides:
- Integration with M1 GPU, memory, and CPU optimizers
- Memory-efficient data loading patterns
- Resource monitoring and management
- Adaptive optimization based on hardware capabilities
- Performance optimization for training pipeline
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import time
import psutil
from contextlib import contextmanager
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_performance
from src.utils.common_operations import get_memory_usage, check_disk_space, timed_operation

logger = system_logger.getChild('HardwareOptimizer')

class HardwareOptimizer:
    """Centralized hardware optimization for training pipeline."""

    def __init__(self):
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self._initialize_optimizers()

    def _initialize_optimizers(self):
        """Initialize hardware optimizers if available."""
        try:
            # Try to import and initialize hardware optimizers
            from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
            self.gpu_manager = get_m1_gpu_manager()
            tprint_info("✅ M1 GPU manager initialized")
        except ImportError:
            tprint_warning("⚠️ M1 GPU manager not available")
            self.gpu_manager = None

        try:
            from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
            self.memory_optimizer = get_m1_memory_optimizer()
            tprint_info("✅ M1 memory optimizer initialized")
        except ImportError:
            tprint_warning("⚠️ M1 memory optimizer not available")
            self.memory_optimizer = None

        try:
            from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
            self.cpu_optimizer = get_m1_cpu_optimizer()
            tprint_info("✅ M1 CPU optimizer initialized")
        except ImportError:
            tprint_warning("⚠️ M1 CPU optimizer not available")
            self.cpu_optimizer = None

        # Initialize system monitoring
        self._setup_system_monitoring()

    def _setup_system_monitoring(self):
        """Setup system resource monitoring."""
        try:
            self.process = psutil.Process()
            self.system_memory_threshold = 0.8  # 80% memory usage threshold
            self.system_cpu_threshold = 0.9     # 90% CPU usage threshold
            tprint_info("✅ System monitoring initialized")
        except ImportError:
            tprint_warning("⚠️ psutil not available for system monitoring")
            self.process = None

    @contextmanager
    def memory_efficient_context(self, operation_name: str):
        """Context manager for memory-efficient operations."""
        start_memory = get_memory_usage() if self.memory_optimizer else 0
        start_time = time.time()

        tprint_info(f"🚀 Starting memory-efficient operation: {operation_name}")

        try:
            # Apply memory optimization if available
            if self.memory_optimizer:
                self.memory_optimizer.optimize_memory_usage()

            # Check system resources
            if self.process:
                memory_percent = psutil.virtual_memory().percent / 100
                cpu_percent = psutil.cpu_percent(interval=1) / 100

                if memory_percent > self.system_memory_threshold:
                    tprint_warning(f"⚠️ High memory usage: {memory_percent:.1%}")
                    if self.memory_optimizer:
                        self.memory_optimizer.force_garbage_collection()

                if cpu_percent > self.system_cpu_threshold:
                    tprint_warning(f"⚠️ High CPU usage: {cpu_percent:.1%}")
                    if self.cpu_optimizer:
                        self.cpu_optimizer.optimize_cpu_usage()

            yield

        except Exception as e:
            tprint_error(f"❌ Error in memory-efficient operation {operation_name}: {e}")
            raise
        finally:
            end_memory = get_memory_usage() if self.memory_optimizer else 0
            end_time = time.time()

            memory_delta = end_memory - start_memory
            duration = end_time - start_time

            if abs(memory_delta) > 100:  # Only report significant memory changes
                tprint_performance(f"📊 {operation_name}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")

    def optimize_data_loading(self, data_path: Union[str, Path], chunk_size: int = None) -> pd.DataFrame:
        """Load data with hardware optimization."""
        data_path = Path(data_path)

        # Determine optimal chunk size based on available memory
        if chunk_size is None:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            file_size_mb = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 100
            chunk_size = max(1000, int(available_memory_mb / (file_size_mb / 100)))

        tprint_info(f"📂 Loading data with chunk size: {chunk_size}")

        with self.memory_efficient_context("data_loading"):
            try:
                # Use hardware-optimized data loading if available
                if self.memory_optimizer:
                    df = self.memory_optimizer.load_dataframe_optimized(data_path, chunk_size=chunk_size)
                else:
                    # Fallback to standard pandas loading
                    df = pd.read_parquet(data_path)

                # Apply CPU optimization if available
                if self.cpu_optimizer and len(df) > 10000:
                    df = self.cpu_optimizer.optimize_dataframe(df)

                return df

            except Exception as e:
                tprint_error(f"❌ Failed to load data: {e}")
                raise

    def optimize_matrix_operations(self, matrices: List[np.ndarray], operation: str) -> np.ndarray:
        """Optimize matrix operations using available hardware acceleration."""
        with self.memory_efficient_context(f"matrix_{operation}"):
            try:
                # Use GPU acceleration if available
                if self.gpu_manager and len(matrices) > 0:
                    # Check if matrices are suitable for GPU processing
                    total_size = sum(matrix.nbytes for matrix in matrices)
                    if total_size < 100 * 1024 * 1024:  # Less than 100MB
                        result = self.gpu_manager.accelerate_matrix_operation(matrices, operation)
                        if result is not None:
                            return result

                # Fallback to CPU optimization
                if self.cpu_optimizer:
                    result = self.cpu_optimizer.optimize_matrix_operation(matrices, operation)
                    if result is not None:
                        return result

                # Standard operations as fallback
                if operation == "concatenate":
                    return np.concatenate(matrices, axis=0)
                elif operation == "stack":
                    return np.stack(matrices, axis=0)
                elif operation == "mean":
                    return np.mean(matrices, axis=0)
                elif operation == "std":
                    return np.std(matrices, axis=0)
                else:
                    raise ValueError(f"Unsupported matrix operation: {operation}")

            except Exception as e:
                tprint_error(f"❌ Matrix operation {operation} failed: {e}")
                raise

    def optimize_training_batch(self, X: np.ndarray, y: np.ndarray, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize training data batching for memory efficiency."""
        with self.memory_efficient_context("training_batch_optimization"):
            try:
                # Determine optimal batch size based on available memory
                if batch_size is None:
                    available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                    data_size_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
                    batch_size = max(100, int(available_memory_mb / (data_size_mb / 100)))

                # Apply memory optimization
                if self.memory_optimizer:
                    X_opt, y_opt = self.memory_optimizer.optimize_training_data(X, y, batch_size)
                    return X_opt, y_opt

                # Fallback to standard batching
                n_samples = X.shape[0]
                n_batches = max(1, n_samples // batch_size)

                # Shuffle data
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                # Create batches
                X_batches = []
                y_batches = []

                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_samples)
                    X_batches.append(X_shuffled[start_idx:end_idx])
                    y_batches.append(y_shuffled[start_idx:end_idx])

                return X_batches, y_batches

            except Exception as e:
                tprint_error(f"❌ Training batch optimization failed: {e}")
                raise

    def monitor_resources(self, operation_name: str) -> Dict[str, Any]:
        """Monitor system resources during operations."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_percent(interval=1)
            disk_info = psutil.disk_usage('/')

            resources = {
                'memory_used_mb': memory_info.used / (1024 * 1024),
                'memory_available_mb': memory_info.available / (1024 * 1024),
                'memory_percent': memory_info.percent,
                'cpu_percent': cpu_info,
                'disk_used_gb': disk_info.used / (1024 ** 3),
                'disk_available_gb': disk_info.free / (1024 ** 3),
                'operation': operation_name,
                'timestamp': time.time()
            }

            # Log resource usage
            if resources['memory_percent'] > 80:
                tprint_warning(f"⚠️ High memory usage during {operation_name}: {resources['memory_percent']:.1f}%")
            if resources['cpu_percent'] > 90:
                tprint_warning(f"⚠️ High CPU usage during {operation_name}: {resources['cpu_percent']:.1f}%")

            return resources

        except ImportError:
            tprint_warning("⚠️ psutil not available for resource monitoring")
            return {'operation': operation_name, 'error': 'psutil_not_available'}

    def optimize_dataframe_operations(self, df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """Optimize DataFrame operations using available hardware acceleration."""
        with self.memory_efficient_context("dataframe_operations"):
            try:
                optimized_df = df.copy()

                # Apply CPU optimization if available
                if self.cpu_optimizer:
                    optimized_df = self.cpu_optimizer.optimize_dataframe(optimized_df)

                # Apply memory optimization if available
                if self.memory_optimizer:
                    optimized_df = self.memory_optimizer.optimize_dataframe_memory(optimized_df)

                return optimized_df

            except Exception as e:
                tprint_error(f"❌ DataFrame optimization failed: {e}")
                raise

    def get_hardware_recommendations(self) -> Dict[str, Any]:
        """Get hardware optimization recommendations."""
        recommendations = {
            'available_optimizers': [],
            'memory_efficiency': 'unknown',
            'cpu_efficiency': 'unknown',
            'gpu_acceleration': 'unavailable',
            'recommended_batch_size': 1000,
            'recommended_chunk_size': 5000
        }

        # Check available optimizers
        if self.gpu_manager:
            recommendations['available_optimizers'].append('gpu_manager')
            recommendations['gpu_acceleration'] = 'available'

        if self.memory_optimizer:
            recommendations['available_optimizers'].append('memory_optimizer')

        if self.cpu_optimizer:
            recommendations['available_optimizers'].append('cpu_optimizer')

        # Assess memory efficiency
        try:
            memory_info = psutil.virtual_memory()
            memory_efficiency = memory_info.available / memory_info.total
            if memory_efficiency > 0.3:
                recommendations['memory_efficiency'] = 'good'
                recommendations['recommended_batch_size'] = 2000
                recommendations['recommended_chunk_size'] = 10000
            elif memory_efficiency > 0.1:
                recommendations['memory_efficiency'] = 'moderate'
                recommendations['recommended_batch_size'] = 1000
                recommendations['recommended_chunk_size'] = 5000
            else:
                recommendations['memory_efficiency'] = 'low'
                recommendations['recommended_batch_size'] = 500
                recommendations['recommended_chunk_size'] = 2000
        except:
            pass

        # Assess CPU efficiency
        try:
            cpu_count = psutil.cpu_count()
            if cpu_count >= 8:
                recommendations['cpu_efficiency'] = 'high'
            elif cpu_count >= 4:
                recommendations['cpu_efficiency'] = 'moderate'
            else:
                recommendations['cpu_efficiency'] = 'low'
        except:
            pass

        return recommendations

# Global hardware optimizer instance
_hardware_optimizer: Optional[HardwareOptimizer] = None

def get_hardware_optimizer() -> HardwareOptimizer:
    """Get or create the global hardware optimizer."""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizer()
    return _hardware_optimizer

def optimize_data_loading(data_path: Union[str, Path], chunk_size: int = None) -> pd.DataFrame:
    """Convenience function for optimized data loading."""
    optimizer = get_hardware_optimizer()
    return optimizer.optimize_data_loading(data_path, chunk_size)

def optimize_training_batch(X: np.ndarray, y: np.ndarray, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for optimized training batching."""
    optimizer = get_hardware_optimizer()
    return optimizer.optimize_training_batch(X, y, batch_size)

def monitor_training_resources(operation_name: str) -> Dict[str, Any]:
    """Convenience function for resource monitoring."""
    optimizer = get_hardware_optimizer()
    return optimizer.monitor_resources(operation_name)