"""
Hardware-Optimized Scaler for features_common.

This module provides scalers that integrate with hardware utilities
for maximum performance optimization of scaling operations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from ..optimization.hardware_optimized_mixin import HardwareOptimizedMixin

logger = logging.getLogger(__name__)

class HardwareOptimizedVectorBTScaler(VectorBTScaler, HardwareOptimizedMixin):
    """
    Hardware-optimized VectorBT scaler with full hardware utility integration.
    
    This scaler extends VectorBTScaler with advanced hardware optimization
    capabilities including intelligent memory management, adaptive optimization,
    GPU acceleration, and performance monitoring.
    """

    def __init__(self, method: str = 'zscore', enable_gpu: bool = False,
                 enable_batch: bool = True, memory_efficient: bool = True,
                 use_optimizer: bool = True, use_unified_manager: bool = True,
                 enable_hardware_optimization: bool = True, **kwargs):
        """
        Initialize hardware-optimized VectorBT scaler.
        
        Args:
            method: Scaling method
            enable_gpu: Enable GPU processing
            enable_batch: Enable batch processing
            memory_efficient: Enable memory optimization
            use_optimizer: Whether to use VectorBTRollingOptimizer
            use_unified_manager: Whether to use UnifiedVectorizationManager
            enable_hardware_optimization: Enable hardware optimization
            **kwargs: Additional parameters
        """
        # Initialize parent classes
        super().__init__(
            method=method, enable_gpu=enable_gpu, enable_batch=enable_batch,
            memory_efficient=memory_efficient, use_optimizer=use_optimizer,
            use_unified_manager=use_unified_manager, **kwargs
        )
        
        # Hardware optimization settings
        self.enable_hardware_optimization = enable_hardware_optimization
        
        # Enhanced performance tracking
        self._hardware_scaling_stats = {
            'hardware_scaling_operations': 0,
            'hardware_memory_optimizations': 0,
            'hardware_gpu_operations': 0,
            'hardware_adaptive_decisions': 0,
            'hardware_cache_hits': 0,
            'hardware_cache_misses': 0,
            'hardware_chunked_operations': 0,
            'hardware_performance_improvements': []
        }
        
        logger.debug(f"Hardware-optimized VectorBT scaler initialized with method={method}")

    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit scaler parameters and transform data with hardware optimization."""
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🚀 [HardwareScaler] Starting hardware-optimized fit_transform with method={self.method}", color="cyan")
        
        self.performance_stats['total_operations'] += 1
        self._hardware_scaling_stats['hardware_scaling_operations'] += 1
        
        # Set workload type for optimization
        self._set_workload_type_for_scaling(data)
        
        if self.enable_hardware_optimization and self.hardware_available:
            try:
                # Use hardware-optimized operation
                result = self.hardware_optimized_operation(
                    self._fit_transform_operation, data, 'scaling'
                )
                
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [HardwareScaler] Hardware-optimized fit_transform completed", color="green")
                
                return result
                
            except Exception as e:
                logger.warning(f"Hardware optimization failed: {e}, using standard method")
                return super().fit_transform(data)
        else:
            return super().fit_transform(data)

    def _fit_transform_operation(self, data: pd.Series) -> pd.Series:
        """Core fit_transform operation for hardware optimization."""
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        # Optimize data for processing
        if self.memory_efficient:
            data = self._apply_memory_optimization(data, {})
        
        # Remove NaN values for fitting
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            logger.warning("No valid data to fit, using defaults")
            return pd.Series(np.nan, index=data.index)
        
        # Apply scaling method
        result, params = self._apply_scaling_method(clean_data, self.method, **self.kwargs)
        self.scaling_params = params
        
        # Align result with original index
        result = result.reindex(data.index)
        self.fitted = True
        
        return result

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data using fitted parameters with hardware optimization."""
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🚀 [HardwareScaler] Starting hardware-optimized transform with method={self.method}", color="cyan")
        
        self._validate_fitted()
        
        if self.enable_hardware_optimization and self.hardware_available:
            try:
                # Use hardware-optimized operation
                result = self.hardware_optimized_operation(
                    self._transform_operation, data, 'scaling'
                )
                
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [HardwareScaler] Hardware-optimized transform completed", color="green")
                
                return result
                
            except Exception as e:
                logger.warning(f"Hardware optimization failed: {e}, using standard method")
                return super().transform(data)
        else:
            return super().transform(data)

    def _transform_operation(self, data: pd.Series) -> pd.Series:
        """Core transform operation for hardware optimization."""
        # Apply transform using fitted parameters
        if self.method == 'zscore':
            mean = self.scaling_params['mean']
            std = self.scaling_params['std']
            result = (data - mean) / std
        elif self.method == 'minmax':
            min_val = self.scaling_params['min']
            max_val = self.scaling_params['max']
            result = (data - min_val) / (max_val - min_val)
        elif self.method == 'robust':
            median = self.scaling_params['median']
            mad = self.scaling_params['mad']
            result = (data - median) / mad
        else:
            # For other methods, use VectorBT directly
            result = self._apply_scaling_method(data, self.method, **self.kwargs)[0]
        
        return result

    def _set_workload_type_for_scaling(self, data: pd.Series) -> None:
        """Set workload type based on scaling operation and data characteristics."""
        from src.utils.hardware.integrated_hardware_manager import WorkloadType
        
        data_size = len(data)
        
        # Determine workload type based on data size and method
        if data_size > 100000:  # Large datasets
            self.set_workload_type(WorkloadType.DATA_PROCESSING)
        elif self.method in ['zscore', 'minmax', 'robust']:
            self.set_workload_type(WorkloadType.ML_TRAINING)
        else:
            self.set_workload_type(WorkloadType.GENERAL)

    def get_hardware_scaling_stats(self) -> Dict[str, Any]:
        """Get hardware scaling statistics."""
        stats = self._hardware_scaling_stats.copy()
        
        # Calculate success rates
        if stats['hardware_scaling_operations'] > 0:
            stats['hardware_memory_optimization_rate'] = (
                stats['hardware_memory_optimizations'] / stats['hardware_scaling_operations']
            )
            stats['hardware_gpu_operation_rate'] = (
                stats['hardware_gpu_operations'] / stats['hardware_scaling_operations']
            )
            stats['hardware_adaptive_decision_rate'] = (
                stats['hardware_adaptive_decisions'] / stats['hardware_scaling_operations']
            )
        else:
            stats['hardware_memory_optimization_rate'] = 0.0
            stats['hardware_gpu_operation_rate'] = 0.0
            stats['hardware_adaptive_decision_rate'] = 0.0
        
        return stats

    def get_comprehensive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including hardware metrics."""
        # Get base performance summary
        base_summary = self.get_performance_summary()
        
        # Get hardware optimization summary
        hardware_summary = self.get_hardware_stats()
        
        # Get scaling-specific hardware stats
        scaling_hardware_stats = self.get_hardware_scaling_stats()
        
        return {
            'base_performance': base_summary,
            'hardware_optimization': hardware_summary,
            'scaling_hardware': scaling_hardware_stats,
            'overall_health': self._assess_scaling_health()
        }

    def _assess_scaling_health(self) -> str:
        """Assess scaling operation health."""
        scaling_stats = self.get_hardware_scaling_stats()
        
        # Check if hardware utilities are available
        if not self.hardware_available:
            return 'hardware_unavailable'
        
        # Check optimization rates
        memory_rate = scaling_stats.get('hardware_memory_optimization_rate', 0)
        adaptive_rate = scaling_stats.get('hardware_adaptive_decision_rate', 0)
        
        # Assess health based on optimization rates
        if memory_rate > 0.8 and adaptive_rate > 0.5:
            return 'excellent'
        elif memory_rate > 0.5 and adaptive_rate > 0.3:
            return 'good'
        elif memory_rate > 0.2 or adaptive_rate > 0.1:
            return 'fair'
        else:
            return 'poor'

    def reset_hardware_scaling_stats(self) -> None:
        """Reset hardware scaling statistics."""
        self._hardware_scaling_stats = {
            'hardware_scaling_operations': 0,
            'hardware_memory_optimizations': 0,
            'hardware_gpu_operations': 0,
            'hardware_adaptive_decisions': 0,
            'hardware_cache_hits': 0,
            'hardware_cache_misses': 0,
            'hardware_chunked_operations': 0,
            'hardware_performance_improvements': []
        }

class HardwareOptimizedVectorBTBatchScaler(VectorBTBatchScaler, HardwareOptimizedMixin):
    """
    Hardware-optimized VectorBT batch scaler with full hardware utility integration.
    
    This scaler extends VectorBTBatchScaler with advanced hardware optimization
    capabilities for processing multiple features efficiently.
    """

    def __init__(self, method: str = 'zscore', enable_gpu: bool = False,
                 memory_efficient: bool = True, enable_parallel: bool = True,
                 use_optimizer: bool = True, use_unified_manager: bool = True,
                 enable_hardware_optimization: bool = True, **kwargs):
        """
        Initialize hardware-optimized VectorBT batch scaler.
        
        Args:
            method: Scaling method
            enable_gpu: Enable GPU processing
            memory_efficient: Enable memory optimization
            enable_parallel: Enable parallel processing
            use_optimizer: Whether to use VectorBTRollingOptimizer
            use_unified_manager: Whether to use UnifiedVectorizationManager
            enable_hardware_optimization: Enable hardware optimization
            **kwargs: Additional parameters
        """
        # Initialize parent classes
        super().__init__(
            method=method, enable_gpu=enable_gpu, memory_efficient=memory_efficient,
            enable_parallel=enable_parallel, use_optimizer=use_optimizer,
            use_unified_manager=use_unified_manager, **kwargs
        )
        
        # Hardware optimization settings
        self.enable_hardware_optimization = enable_hardware_optimization
        
        # Enhanced performance tracking
        self._hardware_batch_stats = {
            'hardware_batch_operations': 0,
            'hardware_memory_optimizations': 0,
            'hardware_gpu_operations': 0,
            'hardware_adaptive_decisions': 0,
            'hardware_cache_hits': 0,
            'hardware_cache_misses': 0,
            'hardware_chunked_operations': 0,
            'hardware_performance_improvements': []
        }
        
        logger.debug(f"Hardware-optimized VectorBT batch scaler initialized with method={method}")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform multiple features using hardware optimization."""
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🚀 [HardwareBatchScaler] Starting hardware-optimized batch fit_transform", color="cyan")
        
        self.performance_stats['total_operations'] += 1
        self._hardware_batch_stats['hardware_batch_operations'] += 1
        
        # Set workload type for optimization
        self._set_workload_type_for_batch_scaling(data)
        
        if self.enable_hardware_optimization and self.hardware_available:
            try:
                # Use hardware-optimized operation
                result = self.hardware_optimized_operation(
                    self._fit_transform_batch_operation, data, 'batch_scaling'
                )
                
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [HardwareBatchScaler] Hardware-optimized batch fit_transform completed", color="green")
                
                return result
                
            except Exception as e:
                logger.warning(f"Hardware optimization failed: {e}, using standard method")
                return super().fit_transform(data)
        else:
            return super().fit_transform(data)

    def _fit_transform_batch_operation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Core fit_transform batch operation for hardware optimization."""
        # Optimize DataFrame for processing
        if self.memory_efficient:
            data = self._optimize_dataframe_types(data)
        
        # Apply batch scaling method
        result = self._apply_enhanced_vectorbt_batch_scaling(data)
        
        # Store scaling parameters for each column
        self._store_scaling_parameters(data)
        
        return result

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters with hardware optimization."""
        from ..utils import TPRINT_AVAILABLE, tprint
        
        if TPRINT_AVAILABLE:
            tprint(f"🚀 [HardwareBatchScaler] Starting hardware-optimized batch transform", color="cyan")
        
        if not self.scalers:
            raise ValueError("Batch scaler must be fitted before transform")
        
        if self.enable_hardware_optimization and self.hardware_available:
            try:
                # Use hardware-optimized operation
                result = self.hardware_optimized_operation(
                    self._transform_batch_operation, data, 'batch_scaling'
                )
                
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [HardwareBatchScaler] Hardware-optimized batch transform completed", color="green")
                
                return result
                
            except Exception as e:
                logger.warning(f"Hardware optimization failed: {e}, using standard method")
                return super().transform(data)
        else:
            return super().transform(data)

    def _transform_batch_operation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Core transform batch operation for hardware optimization."""
        # Apply transform using fitted parameters
        if self.method == 'zscore':
            result = data.copy()
            for col in data.columns:
                if col in self.scalers:
                    mean = self.scalers[col]['mean']
                    std = self.scalers[col]['std']
                    if std != 0:
                        result[col] = (data[col] - mean) / std
                    else:
                        result[col] = 0
        elif self.method == 'minmax':
            result = data.copy()
            for col in data.columns:
                if col in self.scalers:
                    min_val = self.scalers[col]['min']
                    max_val = self.scalers[col]['max']
                    if max_val != min_val:
                        result[col] = (data[col] - min_val) / (max_val - min_val)
                    else:
                        result[col] = 0
        elif self.method == 'robust':
            result = data.copy()
            for col in data.columns:
                if col in self.scalers:
                    median = self.scalers[col]['median']
                    mad = self.scalers[col]['mad']
                    if mad != 0:
                        result[col] = (data[col] - median) / mad
                    else:
                        result[col] = 0
        else:
            # For other methods, use VectorBT directly
            result = self._apply_enhanced_vectorbt_batch_scaling(data)
        
        return result

    def _set_workload_type_for_batch_scaling(self, data: pd.DataFrame) -> None:
        """Set workload type based on batch scaling operation and data characteristics."""
        from src.utils.hardware.integrated_hardware_manager import WorkloadType
        
        data_size = len(data)
        
        # Determine workload type based on data size and method
        if data_size > 50000:  # Large datasets
            self.set_workload_type(WorkloadType.DATA_PROCESSING)
        elif self.method in ['zscore', 'minmax', 'robust']:
            self.set_workload_type(WorkloadType.ML_TRAINING)
        else:
            self.set_workload_type(WorkloadType.GENERAL)

    def get_hardware_batch_stats(self) -> Dict[str, Any]:
        """Get hardware batch scaling statistics."""
        stats = self._hardware_batch_stats.copy()
        
        # Calculate success rates
        if stats['hardware_batch_operations'] > 0:
            stats['hardware_memory_optimization_rate'] = (
                stats['hardware_memory_optimizations'] / stats['hardware_batch_operations']
            )
            stats['hardware_gpu_operation_rate'] = (
                stats['hardware_gpu_operations'] / stats['hardware_batch_operations']
            )
            stats['hardware_adaptive_decision_rate'] = (
                stats['hardware_adaptive_decisions'] / stats['hardware_batch_operations']
            )
        else:
            stats['hardware_memory_optimization_rate'] = 0.0
            stats['hardware_gpu_operation_rate'] = 0.0
            stats['hardware_adaptive_decision_rate'] = 0.0
        
        return stats

    def get_comprehensive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including hardware metrics."""
        # Get base performance summary
        base_summary = self.get_performance_summary()
        
        # Get hardware optimization summary
        hardware_summary = self.get_hardware_stats()
        
        # Get batch-specific hardware stats
        batch_hardware_stats = self.get_hardware_batch_stats()
        
        return {
            'base_performance': base_summary,
            'hardware_optimization': hardware_summary,
            'batch_hardware': batch_hardware_stats,
            'overall_health': self._assess_batch_health()
        }

    def _assess_batch_health(self) -> str:
        """Assess batch scaling operation health."""
        batch_stats = self.get_hardware_batch_stats()
        
        # Check if hardware utilities are available
        if not self.hardware_available:
            return 'hardware_unavailable'
        
        # Check optimization rates
        memory_rate = batch_stats.get('hardware_memory_optimization_rate', 0)
        adaptive_rate = batch_stats.get('hardware_adaptive_decision_rate', 0)
        
        # Assess health based on optimization rates
        if memory_rate > 0.8 and adaptive_rate > 0.5:
            return 'excellent'
        elif memory_rate > 0.5 and adaptive_rate > 0.3:
            return 'good'
        elif memory_rate > 0.2 or adaptive_rate > 0.1:
            return 'fair'
        else:
            return 'poor'

    def reset_hardware_batch_stats(self) -> None:
        """Reset hardware batch scaling statistics."""
        self._hardware_batch_stats = {
            'hardware_batch_operations': 0,
            'hardware_memory_optimizations': 0,
            'hardware_gpu_operations': 0,
            'hardware_adaptive_decisions': 0,
            'hardware_cache_hits': 0,
            'hardware_cache_misses': 0,
            'hardware_chunked_operations': 0,
            'hardware_performance_improvements': []
        }

# Factory functions
def create_hardware_optimized_scaler(method: str = 'zscore', enable_gpu: bool = False,
                                   enable_batch: bool = True, memory_efficient: bool = True,
                                   use_optimizer: bool = True, use_unified_manager: bool = True,
                                   enable_hardware_optimization: bool = True, **kwargs) -> HardwareOptimizedVectorBTScaler:
    """Create a hardware-optimized VectorBT scaler."""
    return HardwareOptimizedVectorBTScaler(
        method=method, enable_gpu=enable_gpu, enable_batch=enable_batch,
        memory_efficient=memory_efficient, use_optimizer=use_optimizer,
        use_unified_manager=use_unified_manager, enable_hardware_optimization=enable_hardware_optimization,
        **kwargs
    )

def create_hardware_optimized_batch_scaler(method: str = 'zscore', enable_gpu: bool = False,
                                         memory_efficient: bool = True, enable_parallel: bool = True,
                                         use_optimizer: bool = True, use_unified_manager: bool = True,
                                         enable_hardware_optimization: bool = True, **kwargs) -> HardwareOptimizedVectorBTBatchScaler:
    """Create a hardware-optimized VectorBT batch scaler."""
    return HardwareOptimizedVectorBTBatchScaler(
        method=method, enable_gpu=enable_gpu, memory_efficient=memory_efficient,
        enable_parallel=enable_parallel, use_optimizer=use_optimizer,
        use_unified_manager=use_unified_manager, enable_hardware_optimization=enable_hardware_optimization,
        **kwargs
    )