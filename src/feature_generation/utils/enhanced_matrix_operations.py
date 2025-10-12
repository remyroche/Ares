"""
Enhanced Matrix Operations for Feature Generation

This module provides enhanced matrix operations optimized for feature generation tasks.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class EnhancedMatrixOperations:
    """Enhanced matrix operations for feature generation."""
    
    def __init__(self, gpu_enabled: bool = True, memory_optimization: bool = True):
        self.gpu_enabled = gpu_enabled
        self.memory_optimization = memory_optimization
        self.logger = logger
        
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for matrix operations.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        try:
            optimized_data = data.copy()
            
            # Convert to optimal data types
            for col in optimized_data.select_dtypes(include=['float64']).columns:
                optimized_data[col] = optimized_data[col].astype('float32')
            
            for col in optimized_data.select_dtypes(include=['int64']).columns:
                optimized_data[col] = optimized_data[col].astype('int32')
            
            self.logger.info(f"✅ DataFrame optimized: {optimized_data.shape}")
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ DataFrame optimization failed: {e}")
            return data
    
    def enhanced_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform enhanced correlation analysis.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Correlation analysis results
        """
        try:
            # Validate input data type
            if not isinstance(data, pd.DataFrame):
                self.logger.warning(f"⚠️ Expected DataFrame but got {type(data)} for correlation analysis")
                return {}
            
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            correlation_matrix = numeric_data.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            results = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation': correlation_matrix.abs().max().max(),
                'mean_correlation': correlation_matrix.abs().mean().mean()
            }
            
            self.logger.info("✅ Enhanced correlation analysis completed")
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced correlation analysis failed: {e}")
            return {}
    
    def vectorized_feature_engineering(self, data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """
        Perform vectorized feature engineering operations.
        
        Args:
            data: Input DataFrame
            operations: List of operations to perform
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Validate input data type
            if not isinstance(data, pd.DataFrame):
                self.logger.warning(f"⚠️ Expected DataFrame but got {type(data)} for vectorized feature engineering")
                return pd.DataFrame()  # Return empty DataFrame instead of dict
            
            result_data = data.copy()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for operation in operations:
                if operation == 'rolling_mean':
                    for col in numeric_cols:
                        if col != 'timestamp':
                            result_data[f'{col}_rolling_mean'] = data[col].rolling(window=20).mean()
                
                elif operation == 'rolling_std':
                    for col in numeric_cols:
                        if col != 'timestamp':
                            result_data[f'{col}_rolling_std'] = data[col].rolling(window=20).std()
                
                elif operation == 'lag_features':
                    for col in numeric_cols:
                        if col != 'timestamp':
                            result_data[f'{col}_lag1'] = data[col].shift(1)
                            result_data[f'{col}_lag5'] = data[col].shift(5)
            
            self.logger.info(f"✅ Vectorized feature engineering completed: {result_data.shape}")
            return result_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized feature engineering failed: {e}")
            return data

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
