"""
Hardware-Optimized VectorBT Manager.

This module provides a VectorBT manager that integrates with hardware utilities
for maximum performance optimization of demanding processes.
"""

import logging
import time
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import pandas as pd
import numpy as np

from .unified_manager import UnifiedVectorBTManager
from ..optimization.hardware_optimized_mixin import HardwareOptimizedMixin
from ..config import get_unified_config

logger = logging.getLogger(__name__)

class HardwareOptimizedVectorBTManager(UnifiedVectorBTManager, HardwareOptimizedMixin):
    """
    Hardware-optimized VectorBT Manager with full hardware utility integration.
    
    This manager extends the UnifiedVectorBTManager with advanced hardware
    optimization capabilities including intelligent memory management,
    adaptive optimization, GPU acceleration, and performance monitoring.
    """

    def __init__(self):
        """Initialize hardware-optimized VectorBT manager."""
        # Initialize parent classes
        super().__init__()
        
        # Enhanced performance tracking
        self._hardware_vectorbt_stats = {
            'vectorbt_hardware_operations': 0,
            'vectorbt_memory_optimizations': 0,
            'vectorbt_gpu_operations': 0,
            'vectorbt_adaptive_decisions': 0,
            'vectorbt_cache_hits': 0,
            'vectorbt_cache_misses': 0,
            'vectorbt_chunked_operations': 0,
            'vectorbt_performance_improvements': []
        }
        
        logger.info("Hardware-optimized VectorBT Manager initialized")

    def execute_operation(self,
                         operation_name: str,
                         data: Union[pd.Series, pd.DataFrame],
                         *args, **kwargs) -> Any:
        """
        Execute a VectorBT operation with full hardware optimization.
        
        Args:
            operation_name: Name of the registered operation
            data: Input data
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation
            
        Returns:
            Result of the optimized operation
        """
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🚀 [HardwareVectorBT] Executing {operation_name} with hardware optimization", color="cyan")
        
        if operation_name not in self._operation_registry:
            error_msg = f"Unknown operation: {operation_name}. Available operations: {list(self._operation_registry.keys())}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [HardwareVectorBT] {error_msg}", color="red")
            raise ValueError(error_msg)
        
        operation_func = self._operation_registry[operation_name]
        
        try:
            # Determine operation type for hardware optimization
            operation_type = self._determine_operation_type(operation_name, data)
            
            # Set workload type based on data characteristics
            self._set_workload_type_for_operation(operation_name, data)
            
            # Execute with hardware optimization
            result = self.hardware_optimized_operation(
                operation_func, data, operation_type, *args, **kwargs
            )
            
            # Update VectorBT-specific stats
            self._update_vectorbt_hardware_stats(operation_name, operation_type)
            
            if TPRINT_AVAILABLE:
                tprint(f"✅ [HardwareVectorBT] Operation {operation_name} completed with hardware optimization", color="green")
            
            return result
            
        except Exception as e:
            error_msg = f"Hardware-optimized operation {operation_name} failed: {e}"
            if TPRINT_AVAILABLE:
                tprint(f"❌ [HardwareVectorBT] {error_msg}", color="red")
            self._log_error(error_msg)
            
            # Fallback to standard operation
            return super().execute_operation(operation_name, data, *args, **kwargs)

    def _determine_operation_type(self, operation_name: str, data: Union[pd.Series, pd.DataFrame]) -> str:
        """Determine the operation type for hardware optimization."""
        # Map operation names to operation types
        operation_type_mapping = {
            'rolling_mean': 'data_processing',
            'rolling_std': 'data_processing',
            'rolling_var': 'data_processing',
            'rolling_min': 'data_processing',
            'rolling_max': 'data_processing',
            'rolling_sum': 'data_processing',
            'rolling_apply': 'data_processing',
            'scale': 'tensor_operations',
            'rank': 'tensor_operations',
            'zscore': 'tensor_operations',
            'winsorize': 'tensor_operations',
            'clip': 'tensor_operations',
            'quantile': 'tensor_operations'
        }
        
        return operation_type_mapping.get(operation_name, 'data_processing')

    def _set_workload_type_for_operation(self, operation_name: str, data: Union[pd.Series, pd.DataFrame]) -> None:
        """Set workload type based on operation and data characteristics."""
        from src.utils.hardware.integrated_hardware_manager import WorkloadType
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Determine workload type based on operation and data size
        if 'rolling' in operation_name:
            if data_size > 100000:  # Large datasets
                self.set_workload_type(WorkloadType.DATA_PROCESSING)
            else:
                self.set_workload_type(WorkloadType.GENERAL)
        elif operation_name in ['scale', 'rank', 'zscore']:
            self.set_workload_type(WorkloadType.ML_TRAINING)
        else:
            self.set_workload_type(WorkloadType.GENERAL)

    def _update_vectorbt_hardware_stats(self, operation_name: str, operation_type: str) -> None:
        """Update VectorBT-specific hardware statistics."""
        self._hardware_vectorbt_stats['vectorbt_hardware_operations'] += 1
        
        # Update operation-specific stats
        if 'memory' in operation_type:
            self._hardware_vectorbt_stats['vectorbt_memory_optimizations'] += 1
        if 'gpu' in operation_type:
            self._hardware_vectorbt_stats['vectorbt_gpu_operations'] += 1
        if 'adaptive' in operation_type:
            self._hardware_vectorbt_stats['vectorbt_adaptive_decisions'] += 1

    def batch_operation(self,
                       operation_name: str,
                       data: Union[pd.Series, pd.DataFrame],
                       *args, **kwargs) -> Any:
        """
        Execute a batch operation with hardware optimization.
        
        Args:
            operation_name: Name of the operation
            data: Input data
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Result of the batch operation
        """
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🔄 [HardwareVectorBT] Executing batch {operation_name} with hardware optimization", color="cyan")
        
        if not self.config.optimization.enable_batch_processing:
            return self.execute_operation(operation_name, data, *args, **kwargs)
        
        # Use hardware-optimized chunking for large datasets
        if isinstance(data, pd.DataFrame) and len(data) > self.config.optimization.batch_size:
            return self._execute_hardware_optimized_batch_operation(operation_name, data, *args, **kwargs)
        else:
            return self.execute_operation(operation_name, data, *args, **kwargs)

    def _execute_hardware_optimized_batch_operation(self,
                                                  operation_name: str,
                                                  data: pd.DataFrame,
                                                  *args, **kwargs) -> pd.DataFrame:
        """Execute batch operation with hardware optimization."""
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [HardwareVectorBT] Using hardware-optimized chunking for large dataset", color="blue")
        
        # Use chunked processing with hardware optimization
        def process_chunk(chunk):
            return self.execute_operation(operation_name, chunk, *args, **kwargs)
        
        # Apply chunked processing with memory optimization
        results = self.chunked_operation(data, process_chunk)
        
        # Combine results
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif results and isinstance(results[0], pd.Series):
            return pd.concat(results, ignore_index=True)
        else:
            return results

    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling mean operation."""
        return self.execute_operation('rolling_mean', data, window=window, **kwargs)

    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling standard deviation operation."""
        return self.execute_operation('rolling_std', data, window=window, **kwargs)

    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling variance operation."""
        return self.execute_operation('rolling_var', data, window=window, **kwargs)

    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling minimum operation."""
        return self.execute_operation('rolling_min', data, window=window, **kwargs)

    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling maximum operation."""
        return self.execute_operation('rolling_max', data, window=window, **kwargs)

    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling sum operation."""
        return self.execute_operation('rolling_sum', data, window=window, **kwargs)

    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], func: Callable, window: int = 20, **kwargs) -> Any:
        """Hardware-optimized rolling apply operation."""
        return self.execute_operation('rolling_apply', data, func, window=window, **kwargs)

    def scale_data(self, data: Union[pd.Series, pd.DataFrame], method: str = 'zscore', **kwargs) -> Any:
        """Hardware-optimized data scaling operation."""
        if method == 'zscore':
            return self.execute_operation('zscore', data, **kwargs)
        elif method == 'minmax':
            return self.execute_operation('scale', data, method='minmax', **kwargs)
        elif method == 'robust':
            return self.execute_operation('scale', data, method='robust', **kwargs)
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def rank_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Hardware-optimized data ranking operation."""
        return self.execute_operation('rank', data, **kwargs)

    def winsorize_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Hardware-optimized data winsorization operation."""
        return self.execute_operation('winsorize', data, **kwargs)

    def clip_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Hardware-optimized data clipping operation."""
        return self.execute_operation('clip', data, **kwargs)

    def quantile_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Any:
        """Hardware-optimized data quantile operation."""
        return self.execute_operation('quantile', data, **kwargs)

    def get_hardware_vectorbt_stats(self) -> Dict[str, Any]:
        """Get VectorBT-specific hardware statistics."""
        stats = self._hardware_vectorbt_stats.copy()
        
        # Calculate success rates
        if stats['vectorbt_hardware_operations'] > 0:
            stats['vectorbt_memory_optimization_rate'] = (
                stats['vectorbt_memory_optimizations'] / stats['vectorbt_hardware_operations']
            )
            stats['vectorbt_gpu_operation_rate'] = (
                stats['vectorbt_gpu_operations'] / stats['vectorbt_hardware_operations']
            )
            stats['vectorbt_adaptive_decision_rate'] = (
                stats['vectorbt_adaptive_decisions'] / stats['vectorbt_hardware_operations']
            )
        else:
            stats['vectorbt_memory_optimization_rate'] = 0.0
            stats['vectorbt_gpu_operation_rate'] = 0.0
            stats['vectorbt_adaptive_decision_rate'] = 0.0
        
        return stats

    def get_comprehensive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including all optimizations."""
        # Get base performance summary
        base_summary = self.get_performance_summary()
        
        # Get hardware optimization summary
        hardware_summary = self.get_hardware_stats()
        
        # Get VectorBT-specific hardware stats
        vectorbt_hardware_stats = self.get_hardware_vectorbt_stats()
        
        # Get operation stats
        operation_stats = self.get_operation_stats()
        
        return {
            'base_performance': base_summary,
            'hardware_optimization': hardware_summary,
            'vectorbt_hardware': vectorbt_hardware_stats,
            'operation_stats': operation_stats,
            'overall_health': self._assess_hardware_health()
        }

    def _assess_hardware_health(self) -> str:
        """Assess overall hardware optimization health."""
        vectorbt_stats = self.get_hardware_vectorbt_stats()
        hardware_stats = self.get_hardware_stats()
        
        # Check if hardware utilities are available
        if not self.hardware_available:
            return 'hardware_unavailable'
        
        # Check optimization rates
        memory_rate = vectorbt_stats.get('vectorbt_memory_optimization_rate', 0)
        gpu_rate = vectorbt_stats.get('vectorbt_gpu_operation_rate', 0)
        adaptive_rate = vectorbt_stats.get('vectorbt_adaptive_decision_rate', 0)
        
        # Assess health based on optimization rates
        if memory_rate > 0.8 and adaptive_rate > 0.5:
            return 'excellent'
        elif memory_rate > 0.5 and adaptive_rate > 0.3:
            return 'good'
        elif memory_rate > 0.2 or adaptive_rate > 0.1:
            return 'fair'
        else:
            return 'poor'

    def reset_hardware_vectorbt_stats(self) -> None:
        """Reset VectorBT-specific hardware statistics."""
        self._hardware_vectorbt_stats = {
            'vectorbt_hardware_operations': 0,
            'vectorbt_memory_optimizations': 0,
            'vectorbt_gpu_operations': 0,
            'vectorbt_adaptive_decisions': 0,
            'vectorbt_cache_hits': 0,
            'vectorbt_cache_misses': 0,
            'vectorbt_chunked_operations': 0,
            'vectorbt_performance_improvements': []
        }

    def optimize_hardware_settings(self) -> None:
        """Optimize hardware settings based on current performance."""
        if not self.hardware_available:
            return
        
        try:
            # Get current performance metrics
            vectorbt_stats = self.get_hardware_vectorbt_stats()
            hardware_stats = self.get_hardware_stats()
            
            # Optimize based on performance
            if vectorbt_stats.get('vectorbt_memory_optimization_rate', 0) < 0.5:
                # Increase memory optimization
                self._memory_optimization_enabled = True
                logger.info("Enabled memory optimization due to low optimization rate")
            
            if vectorbt_stats.get('vectorbt_gpu_operation_rate', 0) < 0.3:
                # Consider enabling GPU optimization
                if self._is_gpu_suitable(pd.Series([1, 2, 3]), 'data_processing'):
                    self._gpu_acceleration_enabled = True
                    logger.info("Enabled GPU acceleration for suitable operations")
            
            # Use adaptive optimization engine for fine-tuning
            if self.adaptive_engine:
                self.adaptive_engine.start_learning()
                logger.info("Started adaptive optimization learning")
                
        except Exception as e:
            logger.warning(f"Failed to optimize hardware settings: {e}")

# Global manager instance
_global_hardware_vectorbt_manager: Optional[HardwareOptimizedVectorBTManager] = None

def get_hardware_optimized_vectorbt_manager() -> HardwareOptimizedVectorBTManager:
    """Get the global hardware-optimized VectorBT manager."""
    global _global_hardware_vectorbt_manager
    if _global_hardware_vectorbt_manager is None:
        _global_hardware_vectorbt_manager = HardwareOptimizedVectorBTManager()
    return _global_hardware_vectorbt_manager

def set_hardware_optimized_vectorbt_manager(manager: HardwareOptimizedVectorBTManager) -> None:
    """Set the global hardware-optimized VectorBT manager."""
    global _global_hardware_vectorbt_manager
    _global_hardware_vectorbt_manager = manager

def reset_hardware_optimized_vectorbt_manager() -> None:
    """Reset the global hardware-optimized VectorBT manager."""
    global _global_hardware_vectorbt_manager
    _global_hardware_vectorbt_manager = None