"""
Enhanced data operations for advanced data processing.

This module provides enhanced data processing operations including
vectorized operations, memory optimization, and advanced data transformations.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
import gc

# Setup logging
logger = logging.getLogger(__name__)

def vectorized_operation(data: Union[pd.DataFrame, pd.Series], operation: Callable, **kwargs) -> Union[pd.DataFrame, pd.Series]:
    """Perform vectorized operation on data."""
    try:
        return operation(data, **kwargs)
    except Exception as e:
        logger.warning(f"Error in vectorized operation: {e}")
        return data

def memory_optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    try:
        original_memory = df.memory_usage(deep=True).sum()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')
        
        new_memory = df.memory_usage(deep=True).sum()
        logger.info(f"Memory usage decreased from {original_memory / 1024**2:.2f} MB to {new_memory / 1024**2:.2f} MB")
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing DataFrame memory: {e}")
        return df

def chunked_processing(df: pd.DataFrame, chunk_size: int, operation: Callable, **kwargs) -> pd.DataFrame:
    """Process DataFrame in chunks to manage memory."""
    try:
        results = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            result_chunk = operation(chunk, **kwargs)
            results.append(result_chunk)
            
            # Force garbage collection
            del chunk
            gc.collect()
        
        return pd.concat(results, ignore_index=True)
    except Exception as e:
        logger.error(f"Error in chunked processing: {e}")
        return df

def parallel_apply(df: pd.DataFrame, func: Callable, axis: int = 0, n_jobs: int = -1) -> pd.DataFrame:
    """Apply function to DataFrame in parallel."""
    try:
        from joblib import Parallel, delayed

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
        
        if axis == 0:
            # Apply to columns
            results = Parallel(n_jobs=n_jobs)(delayed(func)(df[col]) for col in df.columns)
            return pd.DataFrame(dict(zip(df.columns, results)))
        else:
            # Apply to rows
            results = Parallel(n_jobs=n_jobs)(delayed(func)(row) for _, row in df.iterrows())
            return pd.DataFrame(results)
    except Exception as e:
        logger.warning(f"Error in parallel apply, falling back to regular apply: {e}")
        return df.apply(func, axis=axis)

def advanced_fillna(df: pd.DataFrame, method: str = "interpolate", **kwargs) -> pd.DataFrame:
    """Advanced missing value filling."""
    try:
        if method == "interpolate":
            return df.interpolate(**kwargs)
        elif method == "forward_backward":
            return df.fillna(method='ffill').fillna(method='bfill')
        elif method == "median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            return df
        elif method == "mode":
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                mode_value = df[col].mode()
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value[0])
            return df
        else:
            return df.fillna(method=method, **kwargs)
    except Exception as e:
        logger.warning(f"Error in advanced fillna: {e}")
        return df

def rolling_window_analysis(df: pd.DataFrame, window_sizes: List[int], operations: List[str]) -> pd.DataFrame:
    """Perform rolling window analysis with multiple window sizes."""
    try:
        result_df = df.copy()
        
        for window in window_sizes:
            for col in df.select_dtypes(include=[np.number]).columns:
                for op in operations:
                    if op == "mean":
                        result_df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
                    elif op == "std":
                        result_df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
                    elif op == "min":
                        result_df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
                    elif op == "max":
                        result_df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
                    elif op == "sum":
                        result_df[f"{col}_rolling_sum_{window}"] = df[col].rolling(window=window).sum()
        
        return result_df
    except Exception as e:
        logger.error(f"Error in rolling window analysis: {e}")
        return df

def feature_engineering_pipeline(df: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply a pipeline of feature engineering operations."""
    try:
        result_df = df.copy()
        
        for operation in operations:
            op_type = operation.get('type')
            params = operation.get('params', {})
            
            if op_type == "rolling":
                result_df = rolling_window_analysis(result_df, **params)
            elif op_type == "lag":
                for col in params.get('columns', []):
                    for lag in params.get('lags', [1]):
                        result_df[f"{col}_lag_{lag}"] = result_df[col].shift(lag)
            elif op_type == "diff":
                for col in params.get('columns', []):
                    result_df[f"{col}_diff"] = result_df[col].diff()
            elif op_type == "pct_change":
                for col in params.get('columns', []):
                    result_df[f"{col}_pct_change"] = result_df[col].pct_change()
            elif op_type == "log":
                for col in params.get('columns', []):
                    result_df[f"{col}_log"] = np.log1p(result_df[col])
            elif op_type == "sqrt":
                for col in params.get('columns', []):
                    result_df[f"{col}_sqrt"] = np.sqrt(result_df[col])
            elif op_type == "square":
                for col in params.get('columns', []):
                    result_df[f"{col}_square"] = result_df[col] ** 2
        
        return result_df
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        return df

def data_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data quality check."""
    try:
        quality_report = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'issues': []
        }
        
        # Check for issues
        if quality_report['missing_values']:
            high_missing = {k: v for k, v in quality_report['missing_values'].items() if v > len(df) * 0.5}
            if high_missing:
                quality_report['issues'].append(f"High missing values in columns: {list(high_missing.keys())}")
        
        if quality_report['duplicate_rows'] > len(df) * 0.1:
            quality_report['issues'].append(f"High duplicate rows: {quality_report['duplicate_rows']}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality_report['issues'].append(f"Constant columns: {constant_cols}")
        
        return quality_report
    except Exception as e:
        logger.error(f"Error in data quality check: {e}")
        return {}

def smart_sampling(df: pd.DataFrame, sample_size: int, method: str = "random") -> pd.DataFrame:
    """Smart sampling of DataFrame."""
    try:
        if len(df) <= sample_size:
            return df
        
        if method == "random":
            return df.sample(n=sample_size, random_state=42)
        elif method == "stratified":
            # Simple stratified sampling based on first categorical column
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                stratify_col = categorical_cols[0]
                return df.groupby(stratify_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // df[stratify_col].nunique()), random_state=42)
                )
            else:
                return df.sample(n=sample_size, random_state=42)
        elif method == "systematic":
            step = len(df) // sample_size
            return df.iloc[::step][:sample_size]
        else:
            return df.sample(n=sample_size, random_state=42)
    except Exception as e:
        logger.error(f"Error in smart sampling: {e}")
        return df


    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
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
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
