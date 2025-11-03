"""
Unified VectorBT Manager for seamless optimization.

This module provides a unified interface for all VectorBT operations
with automatic optimization selection, fallback handling, and performance monitoring.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import pandas as pd
import numpy as np
import time

from ..config import get_unified_config
from ..mixins import OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin

logger = logging.getLogger(__name__)

class UnifiedVectorBTManager(OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin):
    """
    Unified VectorBT Manager with all optimizations.

    This manager provides a single interface for all VectorBT operations
    with automatic optimization selection, intelligent fallback, and
    comprehensive performance monitoring.
    """

    def __init__(self):
        """Initialize unified VectorBT manager."""
        # Initialize all mixins
        super().__init__()

        # Get unified configuration
        self.config = get_unified_config()

        # VectorBT availability and components
        self._vectorbt_available = self._check_vectorbt_availability()
        self._vectorbt_optimizer = None
        self._vectorization_manager = None

        # Operation registry
        self._operation_registry = {}
        self._registered_operations = {}

        # Performance tracking
        self._operation_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'optimization_failures': 0,
            'performance_improvements': []
        }

        # Initialize VectorBT components
        if self._vectorbt_available:
            self._initialize_vectorbt_components()
            self._register_default_operations()

        # Enable all optimizations by default
        self.enable_optimization()
        self.enable_performance_monitoring()

    def _check_vectorbt_availability(self) -> bool:
        """Check if VectorBT is available and properly configured."""
        try:
            import vectorbt as vbt
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

    def _register_default_operations(self) -> None:
        """Register default VectorBT operations."""
        try:
            import vectorbt as vbt
            
            # Try to import from vectorbt.generic (newer API)
            try:
                from src.utils.vectorbt_compat import (
                    rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply,
                    scale, rank, zscore, winsorize, clip, quantile
                )
                use_vectorbt_native = True
            except ImportError:
                # Fallback to pandas implementations
                use_vectorbt_native = False
                logger.info("VectorBT rolling functions not available, using pandas fallbacks")

            if use_vectorbt_native:
                # Register rolling operations from VectorBT
                self.register_operation('rolling_mean', rolling_mean)
                self.register_operation('rolling_std', rolling_std)
                self.register_operation('rolling_var', rolling_var)
                self.register_operation('rolling_min', rolling_min)
                self.register_operation('rolling_max', rolling_max)
                self.register_operation('rolling_sum', rolling_sum)
                self.register_operation('rolling_apply', rolling_apply)

                # Register scaling operations
                self.register_operation('scale', scale)
                self.register_operation('rank', rank)
                self.register_operation('zscore', zscore)
                self.register_operation('winsorize', winsorize)
                self.register_operation('clip', clip)
                self.register_operation('quantile', quantile)
            else:
                # Register pandas fallback implementations
                self._register_pandas_fallbacks()

            logger.debug("Default VectorBT operations registered")

        except ImportError as e:
            logger.warning(f"Failed to register default VectorBT operations: {e}")
            self._register_pandas_fallbacks()

    def _register_pandas_fallbacks(self) -> None:
        """Register pandas fallback implementations for VectorBT operations."""
        import pandas as pd
        import numpy as np
        
        # Rolling operations using pandas
        self.register_operation('rolling_mean', lambda data, window, **kwargs: data.rolling(window, **kwargs).mean())
        self.register_operation('rolling_std', lambda data, window, **kwargs: data.rolling(window, **kwargs).std())
        self.register_operation('rolling_var', lambda data, window, **kwargs: data.rolling(window, **kwargs).var())
        self.register_operation('rolling_min', lambda data, window, **kwargs: data.rolling(window, **kwargs).min())
        self.register_operation('rolling_max', lambda data, window, **kwargs: data.rolling(window, **kwargs).max())
        self.register_operation('rolling_sum', lambda data, window, **kwargs: data.rolling(window, **kwargs).sum())
        
        # Rolling apply with pandas
        def rolling_apply_fallback(data, window, func, **kwargs):
            return data.rolling(window, **kwargs).apply(func)
        self.register_operation('rolling_apply', rolling_apply_fallback)
        
        # Scaling operations using pandas/numpy
        self.register_operation('scale', lambda data, **kwargs: (data - data.mean()) / data.std())
        self.register_operation('rank', lambda data, **kwargs: data.rank(**kwargs))
        self.register_operation('zscore', lambda data, **kwargs: (data - data.mean()) / data.std())
        
        # Winsorize using numpy
        def winsorize_fallback(data, limits=0.05, **kwargs):
            data = data.copy()
            if isinstance(limits, (int, float)):
                limits = (limits, limits)
            lower = data.quantile(limits[0])
            upper = data.quantile(1 - limits[1])
            data = np.clip(data, lower, upper)
            return data
        self.register_operation('winsorize', winsorize_fallback)
        
        # Clip using numpy
        self.register_operation('clip', lambda data, a_min=None, a_max=None, **kwargs: np.clip(data, a_min, a_max))
        
        # Quantile using pandas
        self.register_operation('quantile', lambda data, q, **kwargs: data.quantile(q, **kwargs))
        
        logger.debug("Pandas fallback operations registered")

    def register_operation(self, name: str, operation_func: Callable) -> None:
        """Register a VectorBT operation."""
        self._operation_registry[name] = operation_func
        logger.debug(f"Registered VectorBT operation: {name}")

    def execute_operation(self,
                         operation_name: str,
                         data: Union[pd.Series, pd.DataFrame],
                         *args, **kwargs) -> Any:
        """
        Execute a VectorBT operation with full optimization.

        Args:
            operation_name: Name of the registered operation
            data: Input data
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            ValueError: If operation is not registered
            RuntimeError: If operation execution fails
        """
        from ..utils import TPRINT_AVAILABLE, tprint

        if TPRINT_AVAILABLE:
            tprint(f"🔧 [UnifiedVectorBTManager] Executing operation: {operation_name}", color="cyan")

        if operation_name not in self._operation_registry:
            error_msg = f"Unknown operation: {operation_name}. Available operations: {list(self._operation_registry.keys())}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [UnifiedVectorBTManager] {error_msg}", color="red")
            raise ValueError(error_msg)

        operation_func = self._operation_registry[operation_name]

        try:
            # Use the unified optimization system
            if TPRINT_AVAILABLE:
                tprint(f"🚀 [UnifiedVectorBTManager] Using unified optimization for {operation_name}", color="green")

            result = self.auto_optimize_operation(operation_func, data, *args, **kwargs)

            if TPRINT_AVAILABLE:
                tprint(f"✅ [UnifiedVectorBTManager] Operation {operation_name} completed successfully", color="green")

            return result

        except Exception as e:
            error_msg = f"Operation {operation_name} failed: {e}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [UnifiedVectorBTManager] {error_msg}", color="red")
            self._log_error(error_msg)
            raise RuntimeError(error_msg) from e

    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Optimized rolling mean operation."""
        return self.execute_operation('rolling_mean', data, window=window, **kwargs)

    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Optimized rolling standard deviation operation."""
        return self.execute_operation('rolling_std', data, window=window, **kwargs)

    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Optimized rolling variance operation."""
        return self.execute_operation('rolling_var', data, window=window, **kwargs)

    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Optimized rolling minimum operation."""
        return self.execute_operation('rolling_min', data, window=window, **kwargs)

    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Optimized rolling maximum operation."""
        return self.execute_operation('rolling_max', data, window=window, **kwargs)

    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Optimized rolling sum operation."""
        return self.execute_operation('rolling_sum', data, window=window, **kwargs)

    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], func: Callable, window: int = 20, **kwargs) -> Any:
        """Optimized rolling apply operation."""
        return self.execute_operation('rolling_apply', data, func, window=window, **kwargs)

    def scale_data(self, data: Union[pd.Series, pd.DataFrame], method: str = 'zscore', **kwargs) -> Any:
        """Optimized data scaling operation."""
        if method == 'zscore':
            return self.execute_operation('zscore', data, **kwargs)
        elif method == 'minmax':
            return self.execute_operation('scale', data, method='minmax', **kwargs)
        elif method == 'robust':
            return self.execute_operation('scale', data, method='robust', **kwargs)
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def rank_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Optimized data ranking operation."""
        return self.execute_operation('rank', data, **kwargs)

    def winsorize_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Optimized data winsorization operation."""
        return self.execute_operation('winsorize', data, **kwargs)

    def clip_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Optimized data clipping operation."""
        return self.execute_operation('clip', data, **kwargs)

    def quantile_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Optimized data quantile operation."""
        return self.execute_operation('quantile', data, **kwargs)

    def batch_operation(self,
                       operation_name: str,
                       data: Union[pd.Series, pd.DataFrame],
                       *args, **kwargs) -> Any:
        """
        Execute a batch operation with optimization.

        Args:
            operation_name: Name of the operation
            data: Input data
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Result of the batch operation
        """
        if not self.config.optimization.enable_batch_processing:
            return self.execute_operation(operation_name, data, *args, **kwargs)

        # Use batch processing for large datasets
        if isinstance(data, pd.DataFrame) and len(data) > self.config.optimization.batch_size:
            return self._execute_batch_operation(operation_name, data, *args, **kwargs)
        else:
            return self.execute_operation(operation_name, data, *args, **kwargs)

    def _execute_batch_operation(self,
                                operation_name: str,
                                data: pd.DataFrame,
                                *args, **kwargs) -> pd.DataFrame:
        """Execute operation in batches for large datasets."""
        batch_size = self.config.optimization.batch_size
        results = []

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            batch_result = self.execute_operation(operation_name, batch, *args, **kwargs)
            results.append(batch_result)

        # Combine results
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif results and isinstance(results[0], pd.Series):
            return pd.concat(results, ignore_index=True)
        else:
            return results

    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        stats = self._operation_stats.copy()

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
        stats['registered_operations'] = len(self._operation_registry)

        return stats

    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return list(self._operation_registry.keys())

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # Get operation stats
        op_stats = self.get_operation_stats()

        # Get optimization stats
        opt_stats = self.get_optimization_stats()

        # Get performance stats
        perf_stats = self.get_performance_stats()

        # Get VectorBT stats
        vectorbt_stats = self.get_vectorbt_stats()

        return {
            'operation_stats': op_stats,
            'optimization_stats': opt_stats,
            'performance_stats': perf_stats,
            'vectorbt_stats': vectorbt_stats,
            'overall_health': self._assess_overall_health()
        }

    def _assess_overall_health(self) -> str:
        """Assess overall system health."""
        op_stats = self.get_operation_stats()

        if not op_stats['vectorbt_available']:
            return 'degraded'

        if op_stats['optimization_failure_rate'] > 0.2:
            return 'critical'
        elif op_stats['pandas_fallback_rate'] > 0.5:
            return 'warning'
        else:
            return 'healthy'

    def optimize_settings(self) -> None:
        """Optimize settings based on current performance."""
        op_stats = self.get_operation_stats()

        # Adjust VectorBT threshold based on success rate
        if op_stats['vectorbt_success_rate'] > 0.9:
            # High success rate - can lower threshold
            current_threshold = self.config.vectorbt.data_size_threshold
            new_threshold = max(100, int(current_threshold * 0.8))
            self.config.vectorbt.data_size_threshold = new_threshold
            logger.info(f"Lowered VectorBT threshold to {new_threshold} due to high success rate")

        elif op_stats['vectorbt_success_rate'] < 0.5:
            # Low success rate - increase threshold
            current_threshold = self.config.vectorbt.data_size_threshold
            new_threshold = int(current_threshold * 1.5)
            self.config.vectorbt.data_size_threshold = new_threshold
            logger.info(f"Increased VectorBT threshold to {new_threshold} due to low success rate")

        # Optimize other components
        if hasattr(self, 'optimize_vectorbt_settings'):
            self.optimize_vectorbt_settings()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._operation_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'optimization_failures': 0,
            'performance_improvements': []
        }

        # Reset mixin stats
        if hasattr(self, 'reset_optimization_stats'):
            self.reset_optimization_stats()
        if hasattr(self, 'reset_performance_stats'):
            self.reset_performance_stats()
        if hasattr(self, 'reset_vectorbt_stats'):
            self.reset_vectorbt_stats()

# Global manager instance
_global_manager: Optional[UnifiedVectorBTManager] = None

def get_unified_vectorbt_manager() -> UnifiedVectorBTManager:
    """Get the global unified VectorBT manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedVectorBTManager()
    return _global_manager

def set_unified_vectorbt_manager(manager: UnifiedVectorBTManager) -> None:
    """Set the global unified VectorBT manager."""
    global _global_manager
    _global_manager = manager

def reset_unified_vectorbt_manager() -> None:
    """Reset the global unified VectorBT manager."""
    global _global_manager
    _global_manager = None
