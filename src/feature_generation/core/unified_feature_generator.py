"""
Unified Feature Generator Base Class

This module provides a unified base class that integrates all centralized
utilities, eliminating code duplication across feature generators.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureResult
from .vectorbt_feature_generator import VectorBTFeatureGenerator
from ..utils.centralized_rolling_manager import get_centralized_rolling_manager, RollingOperation
from ..utils.scaler_factory import get_scaler_factory, ScalerType
from ..utils.common_operations import get_common_operations
from ..utils.unified_vectorization_manager import get_unified_vectorization_manager

logger = logging.getLogger(__name__)

@dataclass
class UnifiedFeatureConfig(FeatureConfig):
    """Enhanced configuration for unified feature generators."""
    # Rolling operations configuration
    rolling_optimization: bool = True
    rolling_window: Optional[int] = None
    
    # Normalization configuration
    auto_normalize: bool = False
    normalization_method: str = 'zscore'
    normalization_feature_type: str = 'default'
    
    # Batch processing configuration
    enable_batch_processing: bool = False
    batch_size: Optional[int] = None
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    detailed_logging: bool = False

class UnifiedFeatureGenerator(VectorBTFeatureGenerator):
    """
    Unified base class that integrates all centralized utilities.
    
    This class eliminates code duplication by providing a single interface
    for all common operations used across feature generators.
    """
    
    def __init__(self, config: UnifiedFeatureConfig):
        """
        Initialize the unified feature generator.
        
        Args:
            config: Configuration for the feature generator
        """
        super().__init__(config)
        
        # Initialize centralized utilities
        self.rolling_manager = get_centralized_rolling_manager()
        self.scaler_factory = get_scaler_factory()
        self.common_operations = get_common_operations()
        self.vectorization_manager = get_unified_vectorization_manager()
        
        # Enhanced configuration
        self.unified_config = config
        
        # Performance tracking
        self._unified_performance_stats = {
            'rolling_operations': 0,
            'normalization_operations': 0,
            'batch_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'memory_optimizations': 0
        }
    
    def rolling_operation(self, data: pd.Series, operation: Union[str, RollingOperation], 
                         window: int, **kwargs) -> pd.Series:
        """
        Execute a rolling operation using centralized manager.
        
        Args:
            data: Input data series
            operation: Rolling operation to perform
            window: Rolling window size
            **kwargs: Additional operation parameters
            
        Returns:
            Resulting rolling operation series
        """
        try:
            result = self.rolling_manager.rolling_operation(operation, data, window, **kwargs)
            self._unified_performance_stats['rolling_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"Rolling operation failed: {e}")
            self._unified_performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_fallback(data, operation, window, **kwargs)
    
    def rolling_mean(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling mean using centralized manager."""
        return self.rolling_operation(data, RollingOperation.MEAN, window, **kwargs)
    
    def rolling_std(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling standard deviation using centralized manager."""
        return self.rolling_operation(data, RollingOperation.STD, window, **kwargs)
    
    def rolling_var(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling variance using centralized manager."""
        return self.rolling_operation(data, RollingOperation.VAR, window, **kwargs)
    
    def rolling_min(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling minimum using centralized manager."""
        return self.rolling_operation(data, RollingOperation.MIN, window, **kwargs)
    
    def rolling_max(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling maximum using centralized manager."""
        return self.rolling_operation(data, RollingOperation.MAX, window, **kwargs)
    
    def rolling_median(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling median using centralized manager."""
        return self.rolling_operation(data, RollingOperation.MEDIAN, window, **kwargs)
    
    def rolling_sum(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling sum using centralized manager."""
        return self.rolling_operation(data, RollingOperation.SUM, window, **kwargs)
    
    def rolling_skew(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling skewness using centralized manager."""
        return self.rolling_operation(data, RollingOperation.SKEW, window, **kwargs)
    
    def rolling_kurt(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling kurtosis using centralized manager."""
        return self.rolling_operation(data, RollingOperation.KURT, window, **kwargs)
    
    def rolling_quantile(self, data: pd.Series, window: int, quantile: float, **kwargs) -> pd.Series:
        """Calculate rolling quantile using centralized manager."""
        return self.rolling_operation(data, RollingOperation.QUANTILE, window, 
                                    quantile=quantile, **kwargs)
    
    def rolling_rank(self, data: pd.Series, window: int, **kwargs) -> pd.Series:
        """Calculate rolling rank using centralized manager."""
        return self.rolling_operation(data, RollingOperation.RANK, window, **kwargs)
    
    def normalize_feature(self, data: pd.Series, method: str = None, 
                         feature_type: str = None, **kwargs) -> pd.Series:
        """
        Normalize feature data using centralized scaler factory.
        
        Args:
            data: Input data series
            method: Normalization method (uses config default if None)
            feature_type: Type of feature (uses config default if None)
            **kwargs: Additional scaler parameters
            
        Returns:
            Normalized data series
        """
        if not self.unified_config.auto_normalize:
            return data
        
        method = method or self.unified_config.normalization_method
        feature_type = feature_type or self.unified_config.normalization_feature_type
        
        try:
            normalized_data = self.common_operations.normalize_feature(
                data, method, feature_type, **kwargs
            )
            self._unified_performance_stats['normalization_operations'] += 1
            return normalized_data
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, returning original data")
            return data
    
    def calculate_technical_indicator(self, data: pd.DataFrame, indicator: str, 
                                    params: Dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate technical indicators using common operations.
        
        Args:
            data: Input DataFrame with OHLCV data
            indicator: Name of the technical indicator
            params: Parameters for the indicator
            
        Returns:
            Calculated indicator values
        """
        try:
            result = self.common_operations.calculate_technical_indicator(
                data, indicator, params
            )
            self._unified_performance_stats['vectorbt_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"Technical indicator calculation failed: {e}")
            return pd.Series(dtype=float, index=data.index)
    
    def calculate_rolling_statistics(self, data: pd.Series, window: int, 
                                   operations: List[Union[str, RollingOperation]] = None,
                                   **kwargs) -> Dict[str, pd.Series]:
        """
        Calculate multiple rolling statistics efficiently.
        
        Args:
            data: Input data series
            window: Rolling window size
            operations: List of operations to perform
            **kwargs: Additional operation parameters
            
        Returns:
            Dictionary mapping operation names to resulting series
        """
        try:
            results = self.common_operations.calculate_rolling_statistics(
                data, window, operations, **kwargs
            )
            self._unified_performance_stats['rolling_operations'] += len(results)
            return results
        except Exception as e:
            logger.warning(f"Rolling statistics calculation failed: {e}")
            return {}
    
    def calculate_price_levels(self, data: pd.DataFrame, 
                              levels: List[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate common price levels using common operations.
        
        Args:
            data: Input DataFrame with OHLCV data
            levels: List of price levels to calculate
            
        Returns:
            Dictionary mapping level names to calculated series
        """
        try:
            return self.common_operations.calculate_price_levels(data, levels)
        except Exception as e:
            logger.warning(f"Price levels calculation failed: {e}")
            return {}
    
    def calculate_returns(self, data: pd.Series, method: str = 'simple', 
                         periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate returns using common operations.
        
        Args:
            data: Input price series
            method: Return calculation method
            periods: List of periods for calculation
            
        Returns:
            Dictionary mapping period names to return series
        """
        try:
            return self.common_operations.calculate_returns(data, method, periods)
        except Exception as e:
            logger.warning(f"Returns calculation failed: {e}")
            return {}
    
    def calculate_volatility_measures(self, data: pd.DataFrame, 
                                    window: int = None) -> Dict[str, pd.Series]:
        """
        Calculate volatility measures using common operations.
        
        Args:
            data: Input DataFrame with OHLCV data
            window: Rolling window size (uses config default if None)
            
        Returns:
            Dictionary mapping measure names to calculated series
        """
        window = window or self.unified_config.rolling_window or 20
        try:
            return self.common_operations.calculate_volatility_measures(data, window)
        except Exception as e:
            logger.warning(f"Volatility measures calculation failed: {e}")
            return {}
    
    def calculate_momentum_indicators(self, data: pd.DataFrame, 
                                    window: int = None) -> Dict[str, pd.Series]:
        """
        Calculate momentum indicators using common operations.
        
        Args:
            data: Input DataFrame with OHLCV data
            window: Rolling window size (uses config default if None)
            
        Returns:
            Dictionary mapping indicator names to calculated series
        """
        window = window or self.unified_config.rolling_window or 14
        try:
            return self.common_operations.calculate_momentum_indicators(data, window)
        except Exception as e:
            logger.warning(f"Momentum indicators calculation failed: {e}")
            return {}
    
    def batch_process_features(self, data: pd.DataFrame, 
                              feature_configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """
        Process multiple features in batch for efficiency.
        
        Args:
            data: Input DataFrame
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            Dictionary mapping feature names to calculated series
        """
        if not self.unified_config.enable_batch_processing:
            logger.warning("Batch processing not enabled in config")
            return {}
        
        try:
            results = self.common_operations.batch_process_features(data, feature_configs)
            self._unified_performance_stats['batch_operations'] += 1
            return results
        except Exception as e:
            logger.warning(f"Batch processing failed: {e}")
            return {}
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for processing using vectorization manager.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        try:
            if self.vectorization_manager:
                optimized_data = self.vectorization_manager.optimize_dataframe(data)
                self._unified_performance_stats['memory_optimizations'] += 1
                return optimized_data
            else:
                return data
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return data
    
    def _pandas_rolling_fallback(self, data: pd.Series, operation: Union[str, RollingOperation], 
                                window: int, **kwargs) -> pd.Series:
        """Fallback to pandas rolling operations."""
        if isinstance(operation, str):
            operation = RollingOperation(operation.lower())
        
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == RollingOperation.MEAN:
            return rolling_obj.mean()
        elif operation == RollingOperation.STD:
            return rolling_obj.std()
        elif operation == RollingOperation.VAR:
            return rolling_obj.var()
        elif operation == RollingOperation.MIN:
            return rolling_obj.min()
        elif operation == RollingOperation.MAX:
            return rolling_obj.max()
        elif operation == RollingOperation.MEDIAN:
            return rolling_obj.median()
        elif operation == RollingOperation.SUM:
            return rolling_obj.sum()
        elif operation == RollingOperation.SKEW:
            return rolling_obj.skew()
        elif operation == RollingOperation.KURT:
            return rolling_obj.kurt()
        elif operation == RollingOperation.QUANTILE:
            return rolling_obj.quantile(kwargs.get('quantile', 0.5))
        elif operation == RollingOperation.RANK:
            return rolling_obj.rank(**kwargs)
        else:
            raise ValueError(f"Unsupported operation for pandas fallback: {operation}")
    
    def get_unified_performance_stats(self) -> Dict[str, Any]:
        """Get unified performance statistics."""
        base_stats = self.get_performance_stats()
        unified_stats = self._unified_performance_stats.copy()
        
        # Merge with base stats
        for key, value in unified_stats.items():
            if key in base_stats:
                base_stats[key] += value
            else:
                base_stats[key] = value
        
        return base_stats
    
    def reset_unified_performance_stats(self):
        """Reset unified performance statistics."""
        self._unified_performance_stats = {
            'rolling_operations': 0,
            'normalization_operations': 0,
            'batch_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'memory_optimizations': 0
        }
        self.reset_performance_stats()

# Convenience function for creating unified generators
def create_unified_generator(generator_class, config: UnifiedFeatureConfig, **kwargs):
    """
    Create a unified feature generator instance.
    
    Args:
        generator_class: The generator class to instantiate
        config: Unified feature configuration
        **kwargs: Additional initialization parameters
        
    Returns:
        Configured unified generator instance
    """
    if not issubclass(generator_class, UnifiedFeatureGenerator):
        raise ValueError("Generator class must inherit from UnifiedFeatureGenerator")
    
    return generator_class(config, **kwargs)