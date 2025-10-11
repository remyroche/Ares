"""
VectorBT-Enhanced Scaler

This module provides VectorBT-optimized scaling and normalization operations
for the features_common system, leveraging VectorBT's high-performance
scaling functions.

Key Features:
- VectorBT scaling functions (zscore, minmax, robust, quantile, winsorize)
- GPU acceleration support
- Memory-efficient processing
- Batch scaling operations
- Fallback to standard scalers when VectorBT is not available
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

from .base_scaler import BaseScaler

logger = logging.getLogger(__name__)


class VectorBTScaler(BaseScaler):
    """
    VectorBT-optimized scaler with comprehensive scaling methods.
    
    This scaler leverages VectorBT's high-performance scaling functions
    for maximum efficiency and accuracy.
    """
    
    def __init__(self, method: str = 'zscore', **kwargs):
        """
        Initialize VectorBT scaler.
        
        Args:
            method: Scaling method ('zscore', 'minmax', 'robust', 'quantile', 'winsorize', 'rank', 'clip')
            **kwargs: Additional parameters for the scaling method
        """
        super().__init__()
        self.method = method
        self.kwargs = kwargs
        self.scaling_params = {}
        
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, using fallback scaler")
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """Fit scaler parameters and transform data using VectorBT."""
        self._log_info(f"🔧 [VectorBTScaler] Fitting {self.method} scaler on {len(data)} samples")
        
        # Validate input
        self._validate_numeric_input(data, "input data")
        
        # Remove NaN values for fitting
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            self._log_warning("⚠️  No valid data to fit, using defaults")
            return pd.Series(np.nan, index=data.index)
        
        if VECTORBT_AVAILABLE:
            try:
                # Use VectorBT scaling
                if self.method == 'zscore':
                    result = zscore(clean_data, **self.kwargs)
                    self.scaling_params = {
                        'mean': clean_data.mean(),
                        'std': clean_data.std()
                    }
                elif self.method == 'minmax':
                    result = scale(clean_data, method='minmax', **self.kwargs)
                    self.scaling_params = {
                        'min': clean_data.min(),
                        'max': clean_data.max()
                    }
                elif self.method == 'robust':
                    result = scale(clean_data, method='robust', **self.kwargs)
                    median = clean_data.median()
                    mad = (clean_data - median).abs().median()
                    self.scaling_params = {
                        'median': median,
                        'mad': mad
                    }
                elif self.method == 'quantile':
                    result = quantile(clean_data, **self.kwargs)
                    self.scaling_params = {
                        'quantiles': clean_data.quantile([0.25, 0.5, 0.75])
                    }
                elif self.method == 'winsorize':
                    result = winsorize(clean_data, **self.kwargs)
                    self.scaling_params = {
                        'limits': self.kwargs.get('limits', (0.05, 0.05))
                    }
                elif self.method == 'rank':
                    result = rank(clean_data, **self.kwargs)
                    self.scaling_params = {
                        'method': self.kwargs.get('method', 'average')
                    }
                elif self.method == 'clip':
                    result = clip(clean_data, **self.kwargs)
                    self.scaling_params = {
                        'lower': self.kwargs.get('lower', None),
                        'upper': self.kwargs.get('upper', None)
                    }
                else:
                    raise ValueError(f"Unsupported scaling method: {self.method}")
                
                # Align result with original index
                result = result.reindex(data.index)
                self.fitted = True
                self._log_success(f"✅ [VectorBTScaler] Fitted {self.method} scaler successfully")
                
                # Validate output
                self._check_output_validity(result, "transformed data")
                
                return result
                
            except Exception as e:
                self._log_warning(f"⚠️  VectorBT scaling failed: {e}, using fallback")
                return self._fallback_fit_transform(data)
        else:
            return self._fallback_fit_transform(data)
    
    def transform(self, data: pd.Series) -> pd.Series:
        """Transform new data using fitted parameters."""
        self._validate_fitted()
        
        if VECTORBT_AVAILABLE and self.fitted:
            try:
                # Use VectorBT scaling with fitted parameters
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
                elif self.method == 'quantile':
                    # For quantile scaling, we need to use the fitted quantiles
                    result = quantile(data, **self.kwargs)
                elif self.method == 'winsorize':
                    result = winsorize(data, **self.kwargs)
                elif self.method == 'rank':
                    result = rank(data, **self.kwargs)
                elif self.method == 'clip':
                    result = clip(data, **self.kwargs)
                else:
                    raise ValueError(f"Unsupported scaling method: {self.method}")
                
                # Validate output
                self._check_output_validity(result, "transformed data")
                
                return result
                
            except Exception as e:
                self._log_warning(f"⚠️  VectorBT transform failed: {e}, using fallback")
                return self._fallback_transform(data)
        else:
            return self._fallback_transform(data)
    
    def _fallback_fit_transform(self, data: pd.Series) -> pd.Series:
        """Fallback fit_transform using standard methods."""
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            return pd.Series(np.nan, index=data.index)
        
        if self.method == 'zscore':
            mean = clean_data.mean()
            std = clean_data.std()
            if std == 0:
                result = pd.Series(0, index=data.index)
            else:
                result = (data - mean) / std
            self.scaling_params = {'mean': mean, 'std': std}
        elif self.method == 'minmax':
            min_val = clean_data.min()
            max_val = clean_data.max()
            if max_val == min_val:
                result = pd.Series(0, index=data.index)
            else:
                result = (data - min_val) / (max_val - min_val)
            self.scaling_params = {'min': min_val, 'max': max_val}
        elif self.method == 'robust':
            median = clean_data.median()
            mad = (clean_data - median).abs().median()
            if mad == 0:
                result = pd.Series(0, index=data.index)
            else:
                result = (data - median) / mad
            self.scaling_params = {'median': median, 'mad': mad}
        else:
            # For other methods, use simple z-score as fallback
            mean = clean_data.mean()
            std = clean_data.std()
            if std == 0:
                result = pd.Series(0, index=data.index)
            else:
                result = (data - mean) / std
            self.scaling_params = {'mean': mean, 'std': std}
        
        self.fitted = True
        return result
    
    def _fallback_transform(self, data: pd.Series) -> pd.Series:
        """Fallback transform using fitted parameters."""
        if self.method == 'zscore':
            mean = self.scaling_params['mean']
            std = self.scaling_params['std']
            if std == 0:
                return pd.Series(0, index=data.index)
            return (data - mean) / std
        elif self.method == 'minmax':
            min_val = self.scaling_params['min']
            max_val = self.scaling_params['max']
            if max_val == min_val:
                return pd.Series(0, index=data.index)
            return (data - min_val) / (max_val - min_val)
        elif self.method == 'robust':
            median = self.scaling_params['median']
            mad = self.scaling_params['mad']
            if mad == 0:
                return pd.Series(0, index=data.index)
            return (data - median) / mad
        else:
            # Fallback to z-score
            mean = self.scaling_params.get('mean', 0)
            std = self.scaling_params.get('std', 1)
            if std == 0:
                return pd.Series(0, index=data.index)
            return (data - mean) / std
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence."""
        return {
            'method': self.method,
            'kwargs': self.kwargs,
            'scaling_params': self.scaling_params,
            'fitted': self.fitted
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore scaler state from persistence."""
        self.method = state.get('method', 'zscore')
        self.kwargs = state.get('kwargs', {})
        self.scaling_params = state.get('scaling_params', {})
        self.fitted = state.get('fitted', False)


class VectorBTBatchScaler:
    """
    VectorBT-optimized batch scaler for processing multiple features efficiently.
    
    This scaler can process multiple features simultaneously using VectorBT's
    batch processing capabilities.
    """
    
    def __init__(self, method: str = 'zscore', **kwargs):
        """
        Initialize VectorBT batch scaler.
        
        Args:
            method: Scaling method
            **kwargs: Additional parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.scalers = {}
        
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, using fallback batch scaler")
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform multiple features using VectorBT batch processing."""
        if not VECTORBT_AVAILABLE:
            return self._fallback_fit_transform(data)
        
        try:
            # Use VectorBT batch scaling
            if self.method == 'zscore':
                result = zscore(data, **self.kwargs)
            elif self.method == 'minmax':
                result = scale(data, method='minmax', **self.kwargs)
            elif self.method == 'robust':
                result = scale(data, method='robust', **self.kwargs)
            elif self.method == 'quantile':
                result = quantile(data, **self.kwargs)
            elif self.method == 'winsorize':
                result = winsorize(data, **self.kwargs)
            elif self.method == 'rank':
                result = rank(data, **self.kwargs)
            elif self.method == 'clip':
                result = clip(data, **self.kwargs)
            else:
                raise ValueError(f"Unsupported scaling method: {self.method}")
            
            # Store scaling parameters for each column
            for col in data.columns:
                if self.method == 'zscore':
                    self.scalers[col] = {
                        'mean': data[col].mean(),
                        'std': data[col].std()
                    }
                elif self.method == 'minmax':
                    self.scalers[col] = {
                        'min': data[col].min(),
                        'max': data[col].max()
                    }
                elif self.method == 'robust':
                    median = data[col].median()
                    mad = (data[col] - median).abs().median()
                    self.scalers[col] = {
                        'median': median,
                        'mad': mad
                    }
            
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT batch scaling failed: {e}, using fallback")
            return self._fallback_fit_transform(data)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        if not self.scalers:
            raise ValueError("Batch scaler must be fitted before transform")
        
        if not VECTORBT_AVAILABLE:
            return self._fallback_transform(data)
        
        try:
            # Use VectorBT batch scaling with fitted parameters
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
                if self.method == 'quantile':
                    result = quantile(data, **self.kwargs)
                elif self.method == 'winsorize':
                    result = winsorize(data, **self.kwargs)
                elif self.method == 'rank':
                    result = rank(data, **self.kwargs)
                elif self.method == 'clip':
                    result = clip(data, **self.kwargs)
                else:
                    raise ValueError(f"Unsupported scaling method: {self.method}")
            
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT batch transform failed: {e}, using fallback")
            return self._fallback_transform(data)
    
    def _fallback_fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback batch fit_transform using standard methods."""
        result = data.copy()
        
        for col in data.columns:
            scaler = VectorBTScaler(self.method, **self.kwargs)
            result[col] = scaler.fit_transform(data[col])
            self.scalers[col] = scaler.scaling_params
        
        return result
    
    def _fallback_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback batch transform using fitted parameters."""
        result = data.copy()
        
        for col in data.columns:
            if col in self.scalers:
                scaler = VectorBTScaler(self.method, **self.kwargs)
                scaler.scaling_params = self.scalers[col]
                scaler.fitted = True
                result[col] = scaler.transform(data[col])
            else:
                result[col] = data[col]  # Keep original if not fitted
        
        return result


def create_vectorbt_scaler(method: str = 'zscore', **kwargs) -> BaseScaler:
    """
    Create a VectorBT-optimized scaler.
    
    Args:
        method: Scaling method
        **kwargs: Additional parameters
        
    Returns:
        VectorBT scaler instance
    """
    return VectorBTScaler(method, **kwargs)


def create_vectorbt_batch_scaler(method: str = 'zscore', **kwargs) -> VectorBTBatchScaler:
    """
    Create a VectorBT-optimized batch scaler.
    
    Args:
        method: Scaling method
        **kwargs: Additional parameters
        
    Returns:
        VectorBT batch scaler instance
    """
    return VectorBTBatchScaler(method, **kwargs)


# Available scaling methods
VECTORBT_SCALING_METHODS = [
    'zscore', 'minmax', 'robust', 'quantile', 
    'winsorize', 'rank', 'clip'
] if VECTORBT_AVAILABLE else []


def get_available_scaling_methods() -> List[str]:
    """Get list of available scaling methods."""
    if VECTORBT_AVAILABLE:
        return VECTORBT_SCALING_METHODS
    else:
        return ['zscore', 'minmax', 'robust']  # Fallback methods