# src/utils/enhanced_data_quality_decorators.py
"""
Enhanced Data Quality Decorators for Comprehensive Validation
Provides advanced data quality validation, memory optimization, and feature engineering fixes.
"""

import functools
import hashlib
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from enum import Enum
import warnings
import psutil
import gc

from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning, critical
from src.training.steps.raw_data_quality_checker import validate_raw_data_quality
from src.utils.feature_output_validator import validate_feature_output
from src.utils.lookahead_bias_detector import (
    detect_lookahead_bias,
    apply_feature_lagging,
)


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"  # Informational messages
    WARNING = "warning"  # Log issues but continue
    ERROR = "error"  # Error level issues
    CRITICAL = "critical"  # Critical issues
    STRICT = "strict"  # Stop on any critical issue
    SILENT = "silent"  # Only log summary


class MemoryOptimizer:
    """Memory optimization utilities for DataFrames."""
    
    def __init__(self):
        self.logger = system_logger.getChild("MemoryOptimizer")
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if df is None or df.empty:
            return df
            
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Optimize integers
            if df[col].dtype == 'int64':
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize floats
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == 'float64' and df[col].isnull().sum() == 0:
                df[col] = df[col].astype(np.float32)
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_saved = initial_memory - final_memory
        
        if memory_saved > 0:
            self.logger.info(f"Memory optimization: {initial_memory:.2f}MB -> {final_memory:.2f}MB (saved {memory_saved:.2f}MB)")
        
        return df
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }


class DataQualityCache:
    """Cache for data quality validation results to avoid redundant checks."""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.logger = system_logger.getChild("DataQualityCache")

    def _generate_cache_key(self, data: pd.DataFrame, method_name: str) -> str:
        """Generate cache key for data quality validation."""
        try:
            # Create a more stable hash based on data shape and column names
            data_signature = f"{data.shape[0]}_{data.shape[1]}_{'_'.join(sorted(data.columns))}"
            data_hash = hashlib.md5(data_signature.encode()).hexdigest()
            return f"{data_hash}_{method_name}"
        except Exception:
            # Fallback to simple hash
            return f"{hash(str(data.shape))}_{method_name}"

    def get(self, data: pd.DataFrame, method_name: str) -> Optional[Dict[str, Any]]:
        """Get cached validation result."""
        cache_key = self._generate_cache_key(data, method_name)
        result = self.cache.get(cache_key)

        if result:
            self.logger.info(f"✅ [CACHE] Cache hit for {method_name}")
        else:
            self.logger.debug(f"❌ [CACHE] Cache miss for {method_name}")

        return result

    def set(self, data: pd.DataFrame, method_name: str, result: Dict[str, Any]) -> None:
        """Set cached validation result."""
        cache_key = self._generate_cache_key(data, method_name)
        self.cache[cache_key] = result

        self.logger.info(f"💾 [CACHE] Cached validation result for {method_name}")

        # Limit cache size
        if len(self.cache) > self.max_size:
            # Remove oldest entries
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.logger.info(f"🗑️ [CACHE] Removed oldest cache entry due to size limit")

    def clear(self) -> None:
        """Clear the cache."""
        self.logger.debug(f"🗑️ [CACHE] Clearing data quality cache")
        cache_size = len(self.cache)
        self.cache.clear()
        self.logger.debug(f"✅ [CACHE] Cache cleared ({cache_size} entries removed)")


# Global instances
_data_quality_cache = DataQualityCache()
_memory_optimizer = MemoryOptimizer()


def extract_data_from_args(args: tuple, kwargs: dict) -> Optional[pd.DataFrame]:
    """Extract DataFrame from function arguments."""
    # Look for DataFrame in positional arguments
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg
    
    # Look for DataFrame in keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
            return value
    
    return None


def validate_constant_features(func):
    """Decorator to detect and remove constant features."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract data
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Check for constant features
            constant_features = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                system_logger.warning(f"Found {len(constant_features)} constant features: {constant_features}")
                data = data.drop(columns=constant_features)
                
                # Update the data in args/kwargs
                for i, arg in enumerate(args):
                    if isinstance(arg, pd.DataFrame):
                        args = list(args)
                        args[i] = data
                        args = tuple(args)
                        break
                else:
                    for key, value in kwargs.items():
                        if isinstance(value, pd.DataFrame):
                            kwargs[key] = data
                            break
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_low_variance_features(func):
    """Decorator to detect and remove low variance features."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Extract data
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Check for low variance features
            low_variance_features = []
            for col in data.select_dtypes(include=[np.number]).columns:
                if data[col].var() < 1e-8:  # Very low variance threshold
                    low_variance_features.append(col)
            
            if low_variance_features:
                system_logger.warning(f"Found {len(low_variance_features)} low variance features: {low_variance_features}")
                data = data.drop(columns=low_variance_features)
                
                # Update the data in args/kwargs
                for i, arg in enumerate(args):
                    if isinstance(arg, pd.DataFrame):
                        args = list(args)
                        args[i] = data
                        args = tuple(args)
                        break
                else:
                    for key, value in kwargs.items():
                        if isinstance(value, pd.DataFrame):
                            kwargs[key] = data
                            break
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_data_completeness(func):
    """Decorator to validate data completeness and handle missing data."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Check for missing data
            missing_data = data.isnull().sum()
            columns_with_missing = missing_data[missing_data > 0]
            
            if not columns_with_missing.empty:
                system_logger.warning(f"Found missing data in {len(columns_with_missing)} columns")
                
                # Handle missing data
                if isinstance(data.index, pd.DatetimeIndex):
                    data = data.fillna(method='ffill').fillna(method='bfill')
                else:
                    # For non-datetime data, use forward fill then backward fill
                    data = data.fillna(method='ffill').fillna(method='bfill')
                
                # Update the data in args/kwargs
                for i, arg in enumerate(args):
                    if isinstance(arg, pd.DataFrame):
                        args = list(args)
                        args[i] = data
                        args = tuple(args)
                        break
                else:
                    for key, value in kwargs.items():
                        if isinstance(value, pd.DataFrame):
                            kwargs[key] = data
                            break
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_datetime_index(func):
    """Decorator to validate and fix datetime index."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Check if data has proper datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                system_logger.warning("Data does not have datetime index, attempting to fix...")
                
                # Try to create datetime index from existing columns
                datetime_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
                
                if datetime_columns:
                    datetime_col = datetime_columns[0]
                    try:
                        data.index = pd.to_datetime(data[datetime_col])
                        data = data.drop(columns=[datetime_col])
                        system_logger.info(f"Created datetime index from column: {datetime_col}")
                    except Exception as e:
                        system_logger.error(f"Failed to create datetime index: {e}")
                        # Create synthetic datetime index as fallback
                        data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='1min')
                else:
                    # Create synthetic datetime index
                    data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='1min')
                
                # Update the data in args/kwargs
                for i, arg in enumerate(args):
                    if isinstance(arg, pd.DataFrame):
                        args = list(args)
                        args[i] = data
                        args = tuple(args)
                        break
                else:
                    for key, value in kwargs.items():
                        if isinstance(value, pd.DataFrame):
                            kwargs[key] = data
                            break
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_multi_timeframe_alignment(func):
    """Decorator to validate multi-timeframe data alignment."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Check for proper datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                system_logger.error("Multi-timeframe data missing datetime index")
                return func(self, *args, **kwargs)
            
            # Check for regular intervals
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0]
                irregular_intervals = time_diffs[time_diffs != expected_interval]
                irregular_ratio = len(irregular_intervals) / len(time_diffs)
                
                if irregular_ratio > 0.05:  # More than 5% irregular
                    system_logger.warning(f"High irregular interval ratio: {irregular_ratio:.3f}")
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_hmm_data_requirements(func):
    """Decorator to validate HMM data requirements."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None:
            # Check for empty data
            if data.empty:
                system_logger.error("HMM Regime Discovery: Empty data provided")
                raise ValueError("Empty data cannot be processed for HMM regime discovery")
            
            # Check for sufficient data points
            if len(data) < 100:
                system_logger.warning(f"HMM Regime Discovery: Insufficient data points ({len(data)})")
            
            # Check for proper OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                system_logger.error(f"HMM Regime Discovery: Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns for HMM: {missing_cols}")
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_data_structure(func):
    """Decorator to validate data structure and completeness."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Check column count consistency
            expected_columns = 19  # Based on expected column count
            if len(data.columns) != expected_columns:
                system_logger.warning(f"Column count mismatch: expected {expected_columns}, got {len(data.columns)}")
            
            # Check for data completeness
            completeness_ratio = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            if completeness_ratio < 0.95:
                system_logger.warning(f"Data completeness below 95%: {completeness_ratio:.2%}")
            
            # Check for price range anomalies
            if 'close' in data.columns:
                price_range = (data['close'].max() - data['close'].min()) / data['close'].mean()
                if price_range > 0.5:  # More than 50% range
                    system_logger.warning(f"Large price range detected: {price_range:.2%}")
        
        return func(self, *args, **kwargs)
    return wrapper


def optimize_memory_usage(func):
    """Decorator to optimize memory usage of DataFrames."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get memory usage before
        memory_before = _memory_optimizer.get_memory_usage()
        
        # Extract and optimize data
        data = extract_data_from_args(args, kwargs)
        if data is not None and not data.empty:
            optimized_data = _memory_optimizer.optimize_dataframe_memory(data.copy())
            
            # Update the data in args/kwargs
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    args = list(args)
                    args[i] = optimized_data
                    args = tuple(args)
                    break
            else:
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        kwargs[key] = optimized_data
                        break
        
        # Execute function
        result = func(self, *args, **kwargs)
        
        # Get memory usage after
        memory_after = _memory_optimizer.get_memory_usage()
        memory_diff = memory_after["rss_mb"] - memory_before["rss_mb"]
        
        if memory_diff > 0:
            system_logger.info(f"Memory usage increased by {memory_diff:.2f}MB during {func.__name__}")
        
        # Force garbage collection
        gc.collect()
        
        return result
    return wrapper


def comprehensive_data_validation(func):
    """Comprehensive data validation decorator combining multiple checks."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Apply all validation decorators
        validated_func = validate_datetime_index(
            validate_data_completeness(
                validate_constant_features(
                    validate_low_variance_features(
                        validate_data_structure(func)
                    )
                )
            )
        )
        
        return validated_func(self, *args, **kwargs)
    return wrapper


def validate_memory_optimized_data_quality(func):
    """Memory-optimized validation decorator."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Apply memory optimization and comprehensive validation
        optimized_func = optimize_memory_usage(
            comprehensive_data_validation(func)
        )
        
        return optimized_func(self, *args, **kwargs)
    return wrapper


def validate_feature_engineering_pipeline(func):
    """Specialized decorator for feature engineering pipeline validation."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Pre-validation checks
            initial_shape = data.shape
            initial_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            system_logger.info(f"Feature engineering pipeline: Input shape {initial_shape}, memory {initial_memory:.2f}MB")
            
            # Apply comprehensive validation
            validated_func = comprehensive_data_validation(func)
            result = validated_func(self, *args, **kwargs)
            
            # Post-validation checks
            if isinstance(result, pd.DataFrame):
                final_shape = result.shape
                final_memory = result.memory_usage(deep=True).sum() / 1024 / 1024
                
                system_logger.info(f"Feature engineering pipeline: Output shape {final_shape}, memory {final_memory:.2f}MB")
                
                # Check for reasonable output
                if final_shape[0] == 0:
                    system_logger.error("Feature engineering produced empty DataFrame")
                elif final_shape[1] < initial_shape[1] * 0.5:
                    system_logger.warning(f"Feature engineering significantly reduced columns: {initial_shape[1]} -> {final_shape[1]}")
            
            return result
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_hmm_regime_discovery(func):
    """Specialized decorator for HMM regime discovery validation."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        data = extract_data_from_args(args, kwargs)
        
        if data is not None and not data.empty:
            # Apply HMM-specific validation
            validated_func = validate_hmm_data_requirements(
                validate_datetime_index(
                    validate_data_completeness(func)
                )
            )
            
            return validated_func(self, *args, **kwargs)
        
        return func(self, *args, **kwargs)
    return wrapper


def validate_multi_timeframe_processing(func):
    """Specialized decorator for multi-timeframe processing validation."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Apply multi-timeframe specific validation
        validated_func = validate_multi_timeframe_alignment(
            validate_datetime_index(
                validate_data_completeness(func)
            )
        )
        
        return validated_func(self, *args, **kwargs)
    return wrapper


# Utility functions for external use
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    return _memory_optimizer.get_memory_usage()


def clear_data_quality_cache() -> None:
    """Clear the data quality cache."""
    _data_quality_cache.clear()


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize a DataFrame's memory usage."""
    return _memory_optimizer.optimize_dataframe_memory(df)


# Export all decorators for easy import
__all__ = [
    'validate_constant_features',
    'validate_low_variance_features',
    'validate_data_completeness',
    'validate_datetime_index',
    'validate_multi_timeframe_alignment',
    'validate_hmm_data_requirements',
    'validate_data_structure',
    'optimize_memory_usage',
    'comprehensive_data_validation',
    'validate_memory_optimized_data_quality',
    'validate_feature_engineering_pipeline',
    'validate_hmm_regime_discovery',
    'validate_multi_timeframe_processing',
    'get_memory_usage',
    'clear_data_quality_cache',
    'optimize_dataframe',
    'MemoryOptimizer',
    'DataQualityCache',
    'ValidationLevel'
]