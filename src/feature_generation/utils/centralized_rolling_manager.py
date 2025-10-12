"""
Centralized Rolling Operations Manager

This module provides a centralized interface for all rolling operations across
the feature generation system, eliminating code duplication and ensuring
consistent VectorBT optimization usage.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..core.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ..utils.unified_vectorization_manager import get_unified_vectorization_manager

logger = logging.getLogger(__name__)

class RollingOperation(Enum):
    """Enum for supported rolling operations."""
    MEAN = "mean"
    STD = "std"
    VAR = "var"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    SUM = "sum"
    COUNT = "count"
    SKEW = "skew"
    KURT = "kurt"
    QUANTILE = "quantile"
    RANK = "rank"

@dataclass
class RollingOperationConfig:
    """Configuration for rolling operations."""
    operation: RollingOperation
    window: int
    min_periods: Optional[int] = None
    center: bool = False
    win_type: Optional[str] = None
    on: Optional[str] = None
    axis: int = 0
    closed: Optional[str] = None
    method: str = 'single'
    numeric_only: bool = False
    quantile: Optional[float] = None  # For quantile operations
    rank_method: str = 'average'  # For rank operations
    ascending: bool = True  # For rank operations

@dataclass
class PerformanceStats:
    """Performance statistics for rolling operations."""
    vectorbt_operations: int = 0
    pandas_fallbacks: int = 0
    total_operations: int = 0
    average_execution_time: float = 0.0
    memory_optimizations: int = 0
    gpu_accelerations: int = 0

class CentralizedRollingManager:
    """
    Centralized manager for all rolling operations with VectorBT optimization.
    
    This class eliminates code duplication by providing a single interface
    for all rolling operations across the feature generation system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the centralized rolling manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        self.vectorization_manager = get_unified_vectorization_manager()
        self.performance_stats = PerformanceStats()
        
        # Operation mapping
        self._operation_map = {
            RollingOperation.MEAN: self._rolling_mean,
            RollingOperation.STD: self._rolling_std,
            RollingOperation.VAR: self._rolling_var,
            RollingOperation.MIN: self._rolling_min,
            RollingOperation.MAX: self._rolling_max,
            RollingOperation.MEDIAN: self._rolling_median,
            RollingOperation.SUM: self._rolling_sum,
            RollingOperation.COUNT: self._rolling_count,
            RollingOperation.SKEW: self._rolling_skew,
            RollingOperation.KURT: self._rolling_kurt,
            RollingOperation.QUANTILE: self._rolling_quantile,
            RollingOperation.RANK: self._rolling_rank,
        }
    
    def rolling_operation(self, operation: Union[str, RollingOperation], 
                         data: pd.Series, window: int, **kwargs) -> pd.Series:
        """
        Execute a rolling operation with automatic optimization selection.
        
        Args:
            operation: The rolling operation to perform
            data: Input data series
            window: Rolling window size
            **kwargs: Additional operation-specific parameters
            
        Returns:
            Resulting rolling operation series
        """
        if isinstance(operation, str):
            operation = RollingOperation(operation.lower())
        
        config = RollingOperationConfig(
            operation=operation,
            window=window,
            **kwargs
        )
        
        return self._execute_rolling_operation(data, config)
    
    def rolling_mean(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling mean with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.MEAN, data, window, **kwargs)
    
    def rolling_std(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling standard deviation with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.STD, data, window, **kwargs)
    
    def rolling_var(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling variance with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.VAR, data, window, **kwargs)
    
    def rolling_min(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling minimum with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.MIN, data, window, **kwargs)
    
    def rolling_max(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling maximum with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.MAX, data, window, **kwargs)
    
    def rolling_median(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling median with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.MEDIAN, data, window, **kwargs)
    
    def rolling_sum(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling sum with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.SUM, data, window, **kwargs)
    
    def rolling_count(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling count with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.COUNT, data, window, **kwargs)
    
    def rolling_skew(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling skewness with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.SKEW, data, window, **kwargs)
    
    def rolling_kurt(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling kurtosis with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.KURT, data, window, **kwargs)
    
    def rolling_quantile(self, data: pd.Series, window: int, quantile: float, **kwargs) -> pd.Series:
        """Calculate rolling quantile with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.QUANTILE, data, window, 
                                    quantile=quantile, **kwargs)
    
    def rolling_rank(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling rank with VectorBT optimization."""
        return self.rolling_operation(RollingOperation.RANK, data, window, **kwargs)
    
    def batch_rolling_operations(self, data: pd.DataFrame, operations: List[RollingOperation], 
                               window: int, **kwargs) -> Dict[str, pd.Series]:
        """
        Execute multiple rolling operations efficiently in batch.
        
        Args:
            data: Input DataFrame
            operations: List of rolling operations to perform
            window: Rolling window size
            **kwargs: Additional operation-specific parameters
            
        Returns:
            Dictionary mapping operation names to resulting series
        """
        results = {}
        
        for operation in operations:
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    key = f"{column}_{operation.value}_{window}"
                    results[key] = self.rolling_operation(operation, data[column], window, **kwargs)
        
        return results
    
    def _execute_rolling_operation(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """Execute the rolling operation with optimization selection."""
        if data.empty or len(data) < config.window:
            return pd.Series(dtype=float, index=data.index)
        
        # Update performance stats
        self.performance_stats.total_operations += 1
        
        # Get the appropriate operation function
        operation_func = self._operation_map.get(config.operation)
        if not operation_func:
            raise ValueError(f"Unsupported rolling operation: {config.operation}")
        
        try:
            # Try VectorBT optimization first
            result = operation_func(data, config)
            self.performance_stats.vectorbt_operations += 1
            return result
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
            self.performance_stats.pandas_fallbacks += 1
            return self._pandas_fallback(data, config)
    
    def _rolling_mean(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling mean."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_mean(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).mean()
    
    def _rolling_std(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling standard deviation."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_std(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).std()
    
    def _rolling_var(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling variance."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_var(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).var()
    
    def _rolling_min(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling minimum."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_min(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).min()
    
    def _rolling_max(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling maximum."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_max(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).max()
    
    def _rolling_median(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling median."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_median(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).median()
    
    def _rolling_sum(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling sum."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_sum(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).sum()
    
    def _rolling_count(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling count."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_count(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).count()
    
    def _rolling_skew(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling skewness."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_skew(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).skew()
    
    def _rolling_kurt(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling kurtosis."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_kurt(data, window=config.window)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).kurt()
    
    def _rolling_quantile(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling quantile."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_quantile(data, window=config.window, 
                                                         quantile=config.quantile)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).quantile(config.quantile)
    
    def _rolling_rank(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """VectorBT-optimized rolling rank."""
        if self.rolling_optimizer:
            return self.rolling_optimizer.rolling_rank(data, window=config.window, 
                                                     method=config.rank_method, 
                                                     ascending=config.ascending)
        else:
            return data.rolling(window=config.window, min_periods=config.min_periods).rank(
                method=config.rank_method, ascending=config.ascending)
    
    def _pandas_fallback(self, data: pd.Series, config: RollingOperationConfig) -> pd.Series:
        """Fallback to pandas rolling operations."""
        rolling_obj = data.rolling(
            window=config.window,
            min_periods=config.min_periods,
            center=config.center,
            win_type=config.win_type,
            on=config.on,
            axis=config.axis,
            closed=config.closed,
            method=config.method,
            numeric_only=config.numeric_only
        )
        
        if config.operation == RollingOperation.MEAN:
            return rolling_obj.mean()
        elif config.operation == RollingOperation.STD:
            return rolling_obj.std()
        elif config.operation == RollingOperation.VAR:
            return rolling_obj.var()
        elif config.operation == RollingOperation.MIN:
            return rolling_obj.min()
        elif config.operation == RollingOperation.MAX:
            return rolling_obj.max()
        elif config.operation == RollingOperation.MEDIAN:
            return rolling_obj.median()
        elif config.operation == RollingOperation.SUM:
            return rolling_obj.sum()
        elif config.operation == RollingOperation.COUNT:
            return rolling_obj.count()
        elif config.operation == RollingOperation.SKEW:
            return rolling_obj.skew()
        elif config.operation == RollingOperation.KURT:
            return rolling_obj.kurt()
        elif config.operation == RollingOperation.QUANTILE:
            return rolling_obj.quantile(config.quantile)
        elif config.operation == RollingOperation.RANK:
            return rolling_obj.rank(method=config.rank_method, ascending=config.ascending)
        else:
            raise ValueError(f"Unsupported operation for pandas fallback: {config.operation}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'vectorbt_operations': self.performance_stats.vectorbt_operations,
            'pandas_fallbacks': self.performance_stats.pandas_fallbacks,
            'total_operations': self.performance_stats.total_operations,
            'vectorbt_success_rate': (
                self.performance_stats.vectorbt_operations / 
                max(self.performance_stats.total_operations, 1)
            ),
            'average_execution_time': self.performance_stats.average_execution_time,
            'memory_optimizations': self.performance_stats.memory_optimizations,
            'gpu_accelerations': self.performance_stats.gpu_accelerations
        }
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = PerformanceStats()

# Global instance
_centralized_rolling_manager = None

def get_centralized_rolling_manager() -> CentralizedRollingManager:
    """Get the global centralized rolling manager instance."""
    global _centralized_rolling_manager
    if _centralized_rolling_manager is None:
        _centralized_rolling_manager = CentralizedRollingManager()
    return _centralized_rolling_manager