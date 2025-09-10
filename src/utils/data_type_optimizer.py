"""
Data type optimization utilities for memory efficiency.

This module provides utilities for optimizing data types in DataFrames
to reduce memory usage while maintaining data integrity.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

def optimize_integer_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize integer column data types."""
    try:
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing integer dtypes: {e}")
        return df

def optimize_float_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize float column data types."""
    try:
        for col in df.select_dtypes(include=['float64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.finfo(np.float16).min and col_max <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing float dtypes: {e}")
        return df

def optimize_object_dtypes(df: pd.DataFrame, min_categories: int = 2) -> pd.DataFrame:
    """Optimize object column data types by converting to category when appropriate."""
    try:
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5 and df[col].nunique() >= min_categories:
                df[col] = df[col].astype('category')
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing object dtypes: {e}")
        return df

def optimize_datetime_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize datetime column data types."""
    try:
        for col in df.select_dtypes(include=['datetime64[ns]']).columns:
            # Check if we can use a smaller datetime precision
            if df[col].dt.nanosecond.sum() == 0:
                df[col] = df[col].astype('datetime64[s]')
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing datetime dtypes: {e}")
        return df

def optimize_boolean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize boolean column data types."""
    try:
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype('bool')
        
        return df
    except Exception as e:
        logger.error(f"Error optimizing boolean dtypes: {e}")
        return df

def reduce_dataframe_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce DataFrame memory usage by optimizing data types."""
    try:
        start_memory = df.memory_usage(deep=True).sum()
        
        # Optimize each data type
        df = optimize_integer_dtypes(df)
        df = optimize_float_dtypes(df)
        df = optimize_object_dtypes(df)
        df = optimize_datetime_dtypes(df)
        df = optimize_boolean_dtypes(df)
        
        end_memory = df.memory_usage(deep=True).sum()
        
        if verbose:
            logger.info(f"Memory usage decreased from {start_memory / 1024**2:.2f} MB to {end_memory / 1024**2:.2f} MB")
            logger.info(f"Memory reduction: {(1 - end_memory / start_memory) * 100:.1f}%")
        
        return df
    except Exception as e:
        logger.error(f"Error reducing DataFrame memory: {e}")
        return df

def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed memory usage information for DataFrame."""
    try:
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_memory': memory_usage.sum(),
            'total_memory_mb': memory_usage.sum() / 1024**2,
            'per_column': memory_usage.to_dict(),
            'per_column_mb': (memory_usage / 1024**2).to_dict(),
            'largest_columns': memory_usage.nlargest(10).to_dict()
        }
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {}

def suggest_optimizations(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Suggest data type optimizations for DataFrame."""
    try:
        suggestions = {
            'integer_optimizations': [],
            'float_optimizations': [],
            'object_optimizations': [],
            'datetime_optimizations': [],
            'general_optimizations': []
        }
        
        # Check integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                suggestions['integer_optimizations'].append(f"{col}: int64 -> int8")
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                suggestions['integer_optimizations'].append(f"{col}: int64 -> int16")
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                suggestions['integer_optimizations'].append(f"{col}: int64 -> int32")
        
        # Check float columns
        for col in df.select_dtypes(include=['float64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= np.finfo(np.float16).min and col_max <= np.finfo(np.float16).max:
                suggestions['float_optimizations'].append(f"{col}: float64 -> float16")
            elif col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                suggestions['float_optimizations'].append(f"{col}: float64 -> float32")
        
        # Check object columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:
                suggestions['object_optimizations'].append(f"{col}: object -> category")
        
        # Check datetime columns
        for col in df.select_dtypes(include=['datetime64[ns]']).columns:
            if df[col].dt.nanosecond.sum() == 0:
                suggestions['datetime_optimizations'].append(f"{col}: datetime64[ns] -> datetime64[s]")
        
        # General optimizations
        if df.select_dtypes(include=['object']).shape[1] > 0:
            suggestions['general_optimizations'].append("Consider converting object columns to category for better performance")
        
        if df.select_dtypes(include=['int64', 'float64']).shape[1] > 0:
            suggestions['general_optimizations'].append("Consider downcasting numeric columns to smaller dtypes")
        
        return suggestions
    except Exception as e:
        logger.error(f"Error suggesting optimizations: {e}")
        return {}

def apply_optimizations(df: pd.DataFrame, optimizations: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply suggested optimizations to DataFrame."""
    try:
        result_df = df.copy()
        
        # Apply integer optimizations
        for opt in optimizations.get('integer_optimizations', []):
            col, dtype = opt.split(': ')[0], opt.split(' -> ')[1]
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(dtype)
        
        # Apply float optimizations
        for opt in optimizations.get('float_optimizations', []):
            col, dtype = opt.split(': ')[0], opt.split(' -> ')[1]
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(dtype)
        
        # Apply object optimizations
        for opt in optimizations.get('object_optimizations', []):
            col = opt.split(': ')[0]
            if col in result_df.columns:
                result_df[col] = result_df[col].astype('category')
        
        # Apply datetime optimizations
        for opt in optimizations.get('datetime_optimizations', []):
            col, dtype = opt.split(': ')[0], opt.split(' -> ')[1]
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(dtype)
        
        return result_df
    except Exception as e:
        logger.error(f"Error applying optimizations: {e}")
        return df
