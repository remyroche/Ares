"""
Hardware Optimization Manager for Analyst Models Training

Handles hardware optimization with caching and M1-specific optimizations.
"""

from typing import Dict, Any, Optional
import time
import numpy as np
import psutil

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
from src.utils.common_operations import (
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, get_memory_usage, check_disk_space
)
()
            self.memory_optimizer = get_integrated_hardware_manager()
            self.cpu_optimizer = get_comprehensive_optimizer()

            integration_result = integrate_with_m1_optimizers()

            if integration_result.get('success', False):
                tprint_success("✅ M1 optimizers initialized successfully")
            else:
                tprint_warning(f"⚠️ M1 optimizer initialization failed: {integration_result.get('error', 'Unknown')}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize M1 optimizers: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    @property
    def m1_available(self) -> bool:
        """Check if M1 hardware is available (cached)."""
        return self._m1_available

    @property
    def mps_available(self) -> bool:
        """Check if MPS is available (cached)."""
        return self._mps_available

    def get_hardware_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get hardware status with optional caching.

        Args:
            use_cache: Whether to use cached status (if valid)

        Returns:
            Dictionary with hardware status information
        """
        current_time = time.time()

        # Return cached status if valid
        if use_cache and self._status_cache is not None:
            cache_age = current_time - self._status_cache_time
            if cache_age < HARDWARE_STATUS_CACHE_TTL:
                return self._status_cache

        # Refresh status
        try:
            status = {
                'm1_available': self._m1_available,
                'mps_available': self._mps_available,
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None,
                'memory_usage_mb': get_memory_usage() / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(),
                'disk_space': check_disk_space('/', 1.0)
            }

            # Update cache
            self._status_cache = status
            self._status_cache_time = current_time

            return status

        except Exception as e:
            tprint_error(f"❌ Failed to get hardware status: {e}")
            return {'error': str(e)}

    def setup_hardware_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Setup hardware optimization for training data.

        Args:
            X: Input features
            y: Target values

        Returns:
            Dictionary with optimization results
        """
        tprint_info("🧠 Setting up hardware optimization")

        optimization_result = {
            'm1_available': self._m1_available,
            'mps_available': self._mps_available,
            'optimizations_applied': [],
            'performance_improvements': {}
        }

        # M1 GPU optimization
        if self._mps_available and self.gpu_manager:
            try:
                X_optimized = create_m1_optimized_array(X, dtype=np.float32)
                y_optimized = create_m1_optimized_array(y, dtype=np.float32)

                optimization_result['optimizations_applied'].append('m1_gpu_optimization')
                optimization_result['performance_improvements']['m1_gpu'] = {
                    'data_optimized': True,
                    'dtype_converted': 'float32'
                }

                tprint_success("✅ M1 GPU optimization applied")

            except Exception as e:
                tprint_warning(f"⚠️ M1 GPU optimization failed: {e}")
        else:
            tprint_info("ℹ️ M1 GPU optimization not available")

        # Memory optimization
        if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
            try:
                memory_result = self.memory_optimizer.get_integrated_hardware_manager().clear_all_caches()
                optimization_result['optimizations_applied'].append('memory_optimization')
                optimization_result['performance_improvements']['memory'] = memory_result

                tprint_success("✅ Memory optimization applied")

            except Exception as e:
                tprint_warning(f"⚠️ Memory optimization failed: {e}")
        else:
            tprint_info("ℹ️ Memory optimization not available")

        # CPU optimization
        if self.cpu_optimizer and hasattr(self.cpu_optimizer, 'optimize_numpy_operations'):
            try:
                self.cpu_optimizer.optimize_numpy_operations()
                optimization_result['optimizations_applied'].append('cpu_optimization')
                optimization_result['performance_improvements']['cpu'] = {'numpy_optimized': True}

                tprint_success("✅ CPU optimization applied")

            except Exception as e:
                tprint_warning(f"⚠️ CPU optimization failed: {e}")
        else:
            tprint_info("ℹ️ CPU optimization not available")

        num_optimizations = len(optimization_result['optimizations_applied'])
        tprint_success(f"✅ Hardware optimization completed: {num_optimizations} optimizations applied")

        return optimization_result

    def optimize_memory_if_needed(self, threshold_mb: Optional[float] = None) -> bool:
        """
        Optimize memory if usage exceeds threshold.

        Args:
            threshold_mb: Memory threshold in MB (uses default if None)

        Returns:
            True if optimization was performed, False otherwise
        """
        if threshold_mb is None:
            from .analyst_training_constants import MEMORY_OPTIMIZATION_THRESHOLD_MB
            threshold_mb = MEMORY_OPTIMIZATION_THRESHOLD_MB

        current_memory = get_memory_usage() / 1024 / 1024

        if current_memory > threshold_mb:
            tprint_warning(f"⚠️ High memory usage detected: {current_memory:.1f}MB")

            if self.memory_optimizer and hasattr(self.memory_optimizer, 'optimize_memory'):
                try:
                    opt_result = self.memory_optimizer.get_integrated_hardware_manager().clear_all_caches()
                    tprint_info(f"🧠 Memory optimization result: {opt_result}")
                    return True
                except Exception as e:
                    tprint_warning(f"⚠️ Memory optimization failed: {e}")
                    return False
            else:
                tprint_info("ℹ️ Memory optimizer not available")
                return False

        return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of hardware optimization status.

        Returns:
            Dictionary with optimization summary
        """
        return {
            'm1_available': self._m1_available,
            'mps_available': self._mps_available,
            'optimizers': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            },
            'cache_status': {
                'cached': self._status_cache is not None,
                'cache_age': time.time() - self._status_cache_time if self._status_cache else None
            }
        }
