"""
Scaler factory for creating optimized scalers.

This factory provides intelligent scaler creation with automatic
optimization selection and configuration.
"""

import logging
from typing import Dict, Any, Optional, Union, Type, List
import pandas as pd

from ..config import get_unified_config
from ..mixins import OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin

logger = logging.getLogger(__name__)

class OptimizedScaler(OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin):
    """
    Optimized scaler with all mixins for maximum performance.
    
    This scaler automatically uses all available optimizations including
    VectorBT, caching, performance monitoring, and validation.
    """
    
    def __init__(self, method: str = 'zscore', **kwargs):
        """Initialize optimized scaler."""
        # Initialize all mixins
        super().__init__()
        
        self.method = method
        self.kwargs = kwargs
        self.fitted = False
        self.scaling_params = {}
        
        # Enable all optimizations by default
        self.enable_optimization()
        self.enable_performance_monitoring()
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit and transform data with all optimizations."""
        # Validate input
        sanitized_data, is_valid, warnings = self.validate_and_sanitize(data, "input data")
        if not is_valid and warnings:
            logger.warning(f"Data validation warnings: {warnings}")
        
        # Use cached operation if available
        if hasattr(self, 'cached_operation'):
            return self.cached_operation(self._fit_transform_impl, sanitized_data)
        else:
            return self._fit_transform_impl(sanitized_data)
    
    def _fit_transform_impl(self, data: pd.Series) -> pd.Series:
        """Implementation of fit_transform with all optimizations."""
        # Use auto-optimization
        return self.auto_optimize_operation(self._scaling_operation, data)
    
    def _scaling_operation(self, data: pd.Series) -> pd.Series:
        """Core scaling operation."""
        if self.method == 'zscore':
            mean = data.mean()
            std = data.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=data.index)
            result = (data - mean) / std
            self.scaling_params = {'mean': mean, 'std': std}
        elif self.method == 'minmax':
            min_val = data.min()
            max_val = data.max()
            if max_val == min_val or pd.isna(max_val) or pd.isna(min_val):
                return pd.Series(0, index=data.index)
            result = (data - min_val) / (max_val - min_val)
            self.scaling_params = {'min': min_val, 'max': max_val}
        elif self.method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            if mad == 0 or pd.isna(mad):
                return pd.Series(0, index=data.index)
            result = (data - median) / mad
            self.scaling_params = {'median': median, 'mad': mad}
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")
        
        self.fitted = True
        return result
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Validate input
        sanitized_data, is_valid, warnings = self.validate_and_sanitize(data, "input data")
        if not is_valid and warnings:
            logger.warning(f"Data validation warnings: {warnings}")
        
        # Use cached operation if available
        if hasattr(self, 'cached_operation'):
            return self.cached_operation(self._transform_impl, sanitized_data)
        else:
            return self._transform_impl(sanitized_data)
    
    def _transform_impl(self, data: pd.Series) -> pd.Series:
        """Implementation of transform with all optimizations."""
        # Use auto-optimization
        return self.auto_optimize_operation(self._transform_operation, data)
    
    def _transform_operation(self, data: pd.Series) -> pd.Series:
        """Core transform operation."""
        if self.method == 'zscore':
            mean = self.scaling_params['mean']
            std = self.scaling_params['std']
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=data.index)
            return (data - mean) / std
        elif self.method == 'minmax':
            min_val = self.scaling_params['min']
            max_val = self.scaling_params['max']
            if max_val == min_val or pd.isna(max_val) or pd.isna(min_val):
                return pd.Series(0, index=data.index)
            return (data - min_val) / (max_val - min_val)
        elif self.method == 'robust':
            median = self.scaling_params['median']
            mad = self.scaling_params['mad']
            if mad == 0 or pd.isna(mad):
                return pd.Series(0, index=data.index)
            return (data - median) / mad
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get scaler state for persistence."""
        return {
            'method': self.method,
            'kwargs': self.kwargs,
            'scaling_params': self.scaling_params,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set scaler state from persistence."""
        self.method = state.get('method', 'zscore')
        self.kwargs = state.get('kwargs', {})
        self.scaling_params = state.get('scaling_params', {})
        self.fitted = state.get('fitted', False)


class OptimizedBatchScaler(OptimizationMixin, PerformanceMixin, VectorBTMixin, ValidationMixin, CachingMixin, MonitoringMixin):
    """
    Optimized batch scaler for processing multiple features.
    
    This scaler can process multiple features simultaneously with
    all available optimizations.
    """
    
    def __init__(self, method: str = 'zscore', **kwargs):
        """Initialize optimized batch scaler."""
        # Initialize all mixins
        super().__init__()
        
        self.method = method
        self.kwargs = kwargs
        self.fitted = False
        self.scalers = {}
        
        # Enable all optimizations by default
        self.enable_optimization()
        self.enable_performance_monitoring()
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform multiple features with all optimizations."""
        # Validate input
        sanitized_data, is_valid, warnings = self.validate_and_sanitize(data, "input data")
        if not is_valid and warnings:
            logger.warning(f"Data validation warnings: {warnings}")
        
        # Use cached operation if available
        if hasattr(self, 'cached_operation'):
            return self.cached_operation(self._fit_transform_impl, sanitized_data)
        else:
            return self._fit_transform_impl(sanitized_data)
    
    def _fit_transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implementation of batch fit_transform with all optimizations."""
        result = data.copy()
        
        for column in data.columns:
            # Create individual scaler for each column
            scaler = OptimizedScaler(self.method, **self.kwargs)
            result[column] = scaler.fit_transform(data[column])
            self.scalers[column] = scaler.get_state()
        
        self.fitted = True
        return result
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        if not self.fitted:
            raise ValueError("Batch scaler must be fitted before transform")
        
        # Validate input
        sanitized_data, is_valid, warnings = self.validate_and_sanitize(data, "input data")
        if not is_valid and warnings:
            logger.warning(f"Data validation warnings: {warnings}")
        
        # Use cached operation if available
        if hasattr(self, 'cached_operation'):
            return self.cached_operation(self._transform_impl, sanitized_data)
        else:
            return self._transform_impl(sanitized_data)
    
    def _transform_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implementation of batch transform with all optimizations."""
        result = data.copy()
        
        for column in data.columns:
            if column in self.scalers:
                # Create scaler and restore state
                scaler = OptimizedScaler(self.method, **self.kwargs)
                scaler.set_state(self.scalers[column])
                result[column] = scaler.transform(data[column])
            else:
                logger.warning(f"No scaler found for column '{column}', using original values")
                result[column] = data[column]
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Get batch scaler state for persistence."""
        return {
            'method': self.method,
            'kwargs': self.kwargs,
            'scalers': self.scalers,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set batch scaler state from persistence."""
        self.method = state.get('method', 'zscore')
        self.kwargs = state.get('kwargs', {})
        self.scalers = state.get('scalers', {})
        self.fitted = state.get('fitted', False)


class ScalerFactory:
    """
    Factory for creating optimized scalers.
    
    This factory provides intelligent scaler creation with automatic
    optimization selection and configuration.
    """
    
    def __init__(self):
        """Initialize scaler factory."""
        self.config = get_unified_config()
        self._scaler_cache = {}
    
    def create_scaler(self, 
                     method: str = 'zscore',
                     use_optimization: bool = True,
                     use_caching: bool = True,
                     use_monitoring: bool = True,
                     **kwargs) -> Union[OptimizedScaler, OptimizedBatchScaler]:
        """
        Create an optimized scaler.
        
        Args:
            method: Scaling method ('zscore', 'minmax', 'robust')
            use_optimization: Whether to use optimization mixins
            use_caching: Whether to use caching
            use_monitoring: Whether to use monitoring
            **kwargs: Additional parameters for the scaler
            
        Returns:
            Optimized scaler instance
        """
        # Create cache key
        cache_key = f"scaler_{method}_{use_optimization}_{use_caching}_{use_monitoring}_{hash(tuple(sorted(kwargs.items())))}"
        
        if cache_key in self._scaler_cache:
            return self._scaler_cache[cache_key]
        
        # Create scaler
        scaler = OptimizedScaler(method=method, **kwargs)
        
        # Configure optimizations
        if not use_optimization:
            scaler.disable_optimization()
        
        if not use_caching:
            scaler.clear_cache()
        
        if not use_monitoring:
            scaler.disable_performance_monitoring()
        
        # Cache scaler
        self._scaler_cache[cache_key] = scaler
        
        return scaler
    
    def create_batch_scaler(self, 
                           method: str = 'zscore',
                           use_optimization: bool = True,
                           use_caching: bool = True,
                           use_monitoring: bool = True,
                           **kwargs) -> OptimizedBatchScaler:
        """
        Create an optimized batch scaler.
        
        Args:
            method: Scaling method
            use_optimization: Whether to use optimization mixins
            use_caching: Whether to use caching
            use_monitoring: Whether to use monitoring
            **kwargs: Additional parameters for the scaler
            
        Returns:
            Optimized batch scaler instance
        """
        # Create cache key
        cache_key = f"batch_scaler_{method}_{use_optimization}_{use_caching}_{use_monitoring}_{hash(tuple(sorted(kwargs.items())))}"
        
        if cache_key in self._scaler_cache:
            return self._scaler_cache[cache_key]
        
        # Create batch scaler
        scaler = OptimizedBatchScaler(method=method, **kwargs)
        
        # Configure optimizations
        if not use_optimization:
            scaler.disable_optimization()
        
        if not use_caching:
            scaler.clear_cache()
        
        if not use_monitoring:
            scaler.disable_performance_monitoring()
        
        # Cache scaler
        self._scaler_cache[cache_key] = scaler
        
        return scaler
    
    def get_available_methods(self) -> List[str]:
        """Get list of available scaling methods."""
        return ['zscore', 'minmax', 'robust']
    
    def get_recommended_method(self, data: pd.Series) -> str:
        """Get recommended scaling method for given data."""
        # Analyze data characteristics
        if data.skew() > 2 or data.kurtosis() > 3:
            return 'robust'  # For skewed or heavy-tailed data
        elif data.min() >= 0 and data.max() <= 1:
            return 'minmax'  # For data already in [0,1] range
        else:
            return 'zscore'  # Default for normal-like data
    
    def clear_cache(self) -> None:
        """Clear scaler cache."""
        self._scaler_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get scaler cache statistics."""
        return {
            'cached_scalers': len(self._scaler_cache),
            'cache_keys': list(self._scaler_cache.keys())
        }


# Global factory instance
_factory_instance: Optional[ScalerFactory] = None

def get_scaler_factory() -> ScalerFactory:
    """Get the global scaler factory."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = ScalerFactory()
    return _factory_instance

def create_optimized_scaler(method: str = 'zscore', **kwargs) -> OptimizedScaler:
    """Create an optimized scaler using the global factory."""
    return get_scaler_factory().create_scaler(method=method, **kwargs)

def create_batch_scaler(method: str = 'zscore', **kwargs) -> OptimizedBatchScaler:
    """Create an optimized batch scaler using the global factory."""
    return get_scaler_factory().create_batch_scaler(method=method, **kwargs)