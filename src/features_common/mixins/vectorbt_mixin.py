"""
VectorBT integration mixin for automatic VectorBT optimization.

This mixin provides seamless VectorBT integration with automatic
optimization selection, fallback handling, and performance monitoring.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable, Tuple, List
import pandas as pd
import numpy as np

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class VectorBTMixin:
    """
    Mixin class providing VectorBT integration and optimization.

    This mixin can be added to any class to provide automatic VectorBT
    optimization with intelligent fallback to pandas when needed.
    """

    def __init__(self, *args, **kwargs):
        """Initialize VectorBT mixin."""
        super().__init__(*args, **kwargs)

        # Get unified configuration
        self.config = get_unified_config()

        # VectorBT availability
        self._vectorbt_available = self._check_vectorbt_availability()
        self._vectorbt_optimizer = None
        self._vectorization_manager = None

        # VectorBT statistics
        self._vectorbt_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'optimization_failures': 0,
            'performance_improvements': []
        }

        # Initialize VectorBT components if available
        if self._vectorbt_available:
            self._initialize_vectorbt_components()

    def _check_vectorbt_availability(self) -> bool:
        """Check if VectorBT is available and properly configured."""
        try:
            import vectorbt as vbt
            # Check if VectorBT is properly installed
            vbt.__version__
            return True
        except ImportError:
            logger.warning("VectorBT not available - falling back to pandas")
            return False
        except Exception as e:
            logger.warning(f"VectorBT availability check failed: {e}")
            return False

    def _initialize_vectorbt_components(self) -> None:
        """Initialize VectorBT optimization components."""
        try:
            # Import VectorBT optimization components
            from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
            from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager

            # Initialize rolling optimizer
            self._vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.vectorbt.enable_gpu,
                enable_parallel=self.config.vectorbt.enable_parallel_processing,
                memory_efficient=self.config.vectorbt.enable_memory_efficient
            )

            # Initialize vectorization manager
            self._vectorization_manager = get_unified_vectorization_manager()

            logger.debug("VectorBT components initialized successfully")

        except ImportError as e:
            logger.warning(f"Failed to import VectorBT optimization components: {e}")
            self._vectorbt_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize VectorBT components: {e}")
            self._vectorbt_available = False

    def is_vectorbt_available(self) -> bool:
        """Check if VectorBT is available and ready to use."""
        return self._vectorbt_available

    def should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """Determine if VectorBT should be used for the given data."""
        if not self._vectorbt_available:
            return False

        data_size = len(data) if hasattr(data, '__len__') else 0

        # Prefer VectorBT by default if enabled and data meets threshold
        if (self.config.vectorbt.prefer_vectorbt and
            data_size >= self.config.vectorbt.data_size_threshold):
            return True

        # Fallback to original logic
        return self.config.should_use_vectorbt(data_size)

    def vectorbt_operation(self,
                          operation_func: Callable,
                          data: Union[pd.Series, pd.DataFrame],
                          *args, **kwargs) -> Any:
        """
        Execute an operation with VectorBT optimization.

        Args:
            operation_func: The operation function to execute
            data: Input data
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            RuntimeError: If both VectorBT and fallback operations fail
        """
        from ..utils import TPRINT_AVAILABLE, tprint

        if TPRINT_AVAILABLE:
            tprint(f"🔧 [VectorBTMixin] Starting VectorBT operation for {operation_func.__name__ if hasattr(operation_func, '__name__') else 'operation'}", color="cyan")

        self._vectorbt_stats['total_operations'] += 1

        if not self.should_use_vectorbt(data):
            # Use pandas fallback
            if TPRINT_AVAILABLE:
                tprint("⚠️  [VectorBTMixin] VectorBT not suitable, using pandas fallback", color="yellow")
            self._vectorbt_stats['pandas_fallbacks'] += 1
            return operation_func(data, *args, **kwargs)

        try:
            # Try VectorBT optimization
            if TPRINT_AVAILABLE:
                tprint("🚀 [VectorBTMixin] Attempting VectorBT optimization", color="green")

            result = self._execute_vectorbt_operation(operation_func, data, *args, **kwargs)
            self._vectorbt_stats['vectorbt_operations'] += 1

            # Track performance improvement
            self._track_performance_improvement(data, result, operation_func)

            if TPRINT_AVAILABLE:
                tprint("✅ [VectorBTMixin] VectorBT operation completed successfully", color="green")

            return result

        except Exception as e:
            error_msg = f"VectorBT operation failed: {e}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [VectorBTMixin] {error_msg}, using pandas fallback", color="red")
            logger.warning(error_msg)
            self._vectorbt_stats['optimization_failures'] += 1
            self._vectorbt_stats['pandas_fallbacks'] += 1

            try:
                # Attempt pandas fallback
                if TPRINT_AVAILABLE:
                    tprint("🔄 [VectorBTMixin] Attempting pandas fallback", color="yellow")
                result = operation_func(data, *args, **kwargs)
                if TPRINT_AVAILABLE:
                    tprint("✅ [VectorBTMixin] Pandas fallback successful", color="green")
                return result
            except Exception as fallback_error:
                error_msg = f"Both VectorBT and pandas fallback failed: {fallback_error}"
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [VectorBTMixin] {error_msg}", color="red")
                self._log_error(error_msg)
                raise RuntimeError(error_msg) from fallback_error

    def _execute_vectorbt_operation(self,
                                   operation_func: Callable,
                                   data: Union[pd.Series, pd.DataFrame],
                                   *args, **kwargs) -> Any:
        """Execute operation using VectorBT optimization."""
        # This is a placeholder for VectorBT-specific operations
        # In a full implementation, this would use VectorBT's optimized functions

        # For now, we'll use the unified vectorization manager if available
        if self._vectorization_manager is not None:
            try:
                return self._vectorization_manager.execute_operation(
                    operation_func, data, *args, **kwargs
                )
            except Exception as e:
                logger.warning(f"Unified vectorization manager failed: {e}")

        # Fallback to direct VectorBT functions
        return self._execute_direct_vectorbt(operation_func, data, *args, **kwargs)

    def _execute_direct_vectorbt(self,
                                operation_func: Callable,
                                data: Union[pd.Series, pd.DataFrame],
                                *args, **kwargs) -> Any:
        """Execute operation using direct VectorBT functions."""
        try:
            import vectorbt as vbt
            from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply
            from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile

            # Map common operations to VectorBT functions
            operation_name = operation_func.__name__ if hasattr(operation_func, '__name__') else str(operation_func)

            if 'rolling_mean' in operation_name or 'mean' in operation_name:
                window = kwargs.get('window', 20)
                return rolling_mean(data, window=window, **kwargs)
            elif 'rolling_std' in operation_name or 'std' in operation_name:
                window = kwargs.get('window', 20)
                return rolling_std(data, window=window, **kwargs)
            elif 'rolling_var' in operation_name or 'var' in operation_name:
                window = kwargs.get('window', 20)
                return rolling_var(data, window=window, **kwargs)
            elif 'rolling_min' in operation_name or 'min' in operation_name:
                window = kwargs.get('window', 20)
                return rolling_min(data, window=window, **kwargs)
            elif 'rolling_max' in operation_name or 'max' in operation_name:
                window = kwargs.get('window', 20)
                return rolling_max(data, window=window, **kwargs)
            elif 'rolling_sum' in operation_name or 'sum' in operation_name:
                window = kwargs.get('window', 20)
                return rolling_sum(data, window=window, **kwargs)
            elif 'zscore' in operation_name or 'normalize' in operation_name:
                return zscore(data, **kwargs)
            elif 'scale' in operation_name:
                return scale(data, **kwargs)
            elif 'rank' in operation_name:
                return rank(data, **kwargs)
            elif 'winsorize' in operation_name:
                return winsorize(data, **kwargs)
            elif 'clip' in operation_name:
                return clip(data, **kwargs)
            elif 'quantile' in operation_name:
                return quantile(data, **kwargs)
            else:
                # Generic VectorBT operation
                return operation_func(data, *args, **kwargs)

        except Exception as e:
            logger.warning(f"Direct VectorBT execution failed: {e}")
            raise

    def _track_performance_improvement(self,
                                     data: Union[pd.Series, pd.DataFrame],
                                     result: Any,
                                     operation_func: Callable) -> None:
        """Track performance improvement from VectorBT optimization."""
        # This would implement actual performance tracking
        # For now, we'll just log the operation
        data_size = len(data) if hasattr(data, '__len__') else 0
        logger.debug(f"VectorBT operation completed on {data_size} samples")

    def get_vectorbt_stats(self) -> Dict[str, Any]:
        """Get VectorBT operation statistics."""
        stats = self._vectorbt_stats.copy()

        # Calculate success rates
        if stats['total_operations'] > 0:
            stats['vectorbt_success_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['pandas_fallback_rate'] = stats['pandas_fallbacks'] / stats['total_operations']
            stats['optimization_failure_rate'] = stats['optimization_failures'] / stats['total_operations']
        else:
            stats['vectorbt_success_rate'] = 0.0
            stats['pandas_fallback_rate'] = 0.0
            stats['optimization_failure_rate'] = 0.0

        # Add availability status
        stats['vectorbt_available'] = self._vectorbt_available
        stats['optimizer_available'] = self._vectorbt_optimizer is not None
        stats['vectorization_manager_available'] = self._vectorization_manager is not None

        return stats

    def reset_vectorbt_stats(self) -> None:
        """Reset VectorBT statistics."""
        self._vectorbt_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'optimization_failures': 0,
            'performance_improvements': []
        }

    def get_vectorbt_recommendations(self) -> List[str]:
        """Get recommendations for VectorBT optimization."""
        recommendations = []
        stats = self.get_vectorbt_stats()

        # Check VectorBT availability
        if not self._vectorbt_available:
            recommendations.append("Install VectorBT for better performance")

        # Check success rate
        if stats['vectorbt_success_rate'] < 0.8:
            recommendations.append("Low VectorBT success rate - check configuration")

        # Check fallback rate
        if stats['pandas_fallback_rate'] > 0.5:
            recommendations.append("High pandas fallback rate - consider adjusting thresholds")

        # Check optimization failures
        if stats['optimization_failure_rate'] > 0.1:
            recommendations.append("High optimization failure rate - check error logs")

        return recommendations

    def optimize_vectorbt_settings(self) -> None:
        """Optimize VectorBT settings based on current performance."""
        stats = self.get_vectorbt_stats()

        # Adjust threshold based on success rate
        if stats['vectorbt_success_rate'] > 0.9:
            # High success rate - can lower threshold
            current_threshold = self.config.vectorbt.data_size_threshold
            new_threshold = max(100, int(current_threshold * 0.8))
            self.config.vectorbt.data_size_threshold = new_threshold
            logger.info(f"Lowered VectorBT threshold to {new_threshold} due to high success rate")

        elif stats['vectorbt_success_rate'] < 0.5:
            # Low success rate - increase threshold
            current_threshold = self.config.vectorbt.data_size_threshold
            new_threshold = int(current_threshold * 1.5)
            self.config.vectorbt.data_size_threshold = new_threshold
            logger.info(f"Increased VectorBT threshold to {new_threshold} due to low success rate")

    def get_vectorbt_config(self) -> Dict[str, Any]:
        """Get current VectorBT configuration."""
        return self.config.vectorbt.to_dict()

    def update_vectorbt_config(self, **kwargs) -> None:
        """Update VectorBT configuration."""
        self.config.vectorbt.update(**kwargs)

        # Reinitialize components if needed
        if self._vectorbt_available:
            self._initialize_vectorbt_components()
