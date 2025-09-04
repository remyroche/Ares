#!/usr/bin/env python3
"""
Backtesting Pipeline Decorators

This module provides specialized decorators for the backtesting pipeline,
ensuring data formatting, analysis integrity, and access protection.
"""

import asyncio
import functools
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import json
import hashlib
import threading
from contextlib import contextmanager

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory,
)
from src.core.domain.decorators import validate_data_quality, ValidationLevel
from src.utils.compat import handle_errors


class DataFormattingDecorator:
    """Decorators for data formatting operations."""
    
    @staticmethod
    def ensure_dataframe_format(
        required_columns: Optional[List[str]] = None,
        ensure_timestamp: bool = True,
        sort_by_timestamp: bool = True
    ) -> Callable:
        """Ensure DataFrame has proper format for backtesting."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find DataFrame arguments
                dataframes = []
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        dataframes.append(arg)
                for value in kwargs.values():
                    if isinstance(value, pd.DataFrame):
                        dataframes.append(value)
                
                # Process each DataFrame
                processed_dataframes = []
                for df in dataframes:
                    processed_df = df.copy()
                    
                    # Ensure required columns
                    if required_columns:
                        missing_cols = set(required_columns) - set(processed_df.columns)
                        if missing_cols:
                            raise ValueError(f"Missing required columns: {missing_cols}")
                    
                    # Ensure timestamp column
                    if ensure_timestamp and "timestamp" in processed_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(processed_df["timestamp"]):
                            processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"])
                    
                    # Sort by timestamp
                    if sort_by_timestamp and "timestamp" in processed_df.columns:
                        processed_df = processed_df.sort_values("timestamp").reset_index(drop=True)
                    
                    processed_dataframes.append(processed_df)
                
                # Replace original DataFrames with processed ones
                new_args = []
                df_index = 0
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        new_args.append(processed_dataframes[df_index])
                        df_index += 1
                    else:
                        new_args.append(arg)
                
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        new_kwargs[key] = processed_dataframes[df_index]
                        df_index += 1
                    else:
                        new_kwargs[key] = value
                
                return func(*new_args, **new_kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def validate_price_data(
        check_ohlc: bool = True,
        check_volume: bool = True,
        check_timestamps: bool = True
    ) -> Callable:
        """Validate price data integrity."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find price data
                price_data = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame) and "close" in arg.columns:
                        price_data = arg
                        break
                if price_data is None:
                    for value in kwargs.values():
                        if isinstance(value, pd.DataFrame) and "close" in value.columns:
                            price_data = value
                            break
                
                if price_data is not None:
                    # Check OHLC consistency
                    if check_ohlc and all(col in price_data.columns for col in ["open", "high", "low", "close"]):
                        invalid_ohlc = (
                            (price_data["high"] < price_data["low"]) |
                            (price_data["high"] < price_data["open"]) |
                            (price_data["high"] < price_data["close"]) |
                            (price_data["low"] > price_data["open"]) |
                            (price_data["low"] > price_data["close"])
                        )
                        if invalid_ohlc.any():
                            raise ValueError(f"Invalid OHLC relationships found in {invalid_ohlc.sum()} rows")
                    
                    # Check volume
                    if check_volume and "volume" in price_data.columns:
                        if (price_data["volume"] < 0).any():
                            raise ValueError("Negative volume values found")
                    
                    # Check timestamps
                    if check_timestamps and "timestamp" in price_data.columns:
                        if price_data["timestamp"].isnull().any():
                            raise ValueError("Null timestamps found")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def ensure_numeric_features(
        handle_infinite: bool = True,
        handle_nan: bool = True,
        fill_method: str = "forward"
    ) -> Callable:
        """Ensure feature data is properly formatted for analysis."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find feature DataFrames
                feature_data = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        feature_data = arg
                        break
                if feature_data is None:
                    for value in kwargs.values():
                        if isinstance(value, pd.DataFrame):
                            feature_data = value
                            break
                
                if feature_data is not None:
                    processed_df = feature_data.copy()
                    
                    # Handle infinite values
                    if handle_infinite:
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if np.isinf(processed_df[col]).any():
                                processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Handle NaN values
                    if handle_nan:
                        if fill_method == "forward":
                            processed_df = processed_df.fillna(method="ffill")
                        elif fill_method == "backward":
                            processed_df = processed_df.fillna(method="bfill")
                        elif fill_method == "zero":
                            processed_df = processed_df.fillna(0)
                        elif fill_method == "mean":
                            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                    
                    # Replace original DataFrame
                    new_args = []
                    for arg in args:
                        if isinstance(arg, pd.DataFrame) and arg is feature_data:
                            new_args.append(processed_df)
                        else:
                            new_args.append(arg)
                    
                    new_kwargs = {}
                    for key, value in kwargs.items():
                        if isinstance(value, pd.DataFrame) and value is feature_data:
                            new_kwargs[key] = processed_df
                        else:
                            new_kwargs[key] = value
                    
                    return func(*new_args, **new_kwargs)
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


class AnalysisProtectionDecorator:
    """Decorators for analysis integrity and protection."""
    
    @staticmethod
    def prevent_lookahead_bias(
        max_future_lookback: int = 0,
        strict_mode: bool = True
    ) -> Callable:
        """Prevent lookahead bias in analysis operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find timestamp data
                timestamp_data = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame) and "timestamp" in arg.columns:
                        timestamp_data = arg
                        break
                if timestamp_data is None:
                    for value in kwargs.values():
                        if isinstance(value, pd.DataFrame) and "timestamp" in value.columns:
                            timestamp_data = value
                            break
                
                if timestamp_data is not None and strict_mode:
                    # Check if data is sorted by timestamp
                    if not timestamp_data["timestamp"].is_monotonic_increasing:
                        raise ValueError("Data must be sorted by timestamp to prevent lookahead bias")
                    
                    # Check for future data access
                    current_time = get_current_datetime()
                    future_data = timestamp_data[timestamp_data["timestamp"] > current_time]
                    if not future_data.empty:
                        raise ValueError(f"Future data detected: {len(future_data)} rows with timestamps after current time")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def validate_analysis_inputs(
        min_data_points: int = 100,
        required_columns: Optional[List[str]] = None
    ) -> Callable:
        """Validate inputs for analysis operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find DataFrame arguments
                dataframes = []
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        dataframes.append(arg)
                for value in kwargs.values():
                    if isinstance(value, pd.DataFrame):
                        dataframes.append(value)
                
                for df in dataframes:
                    # Check minimum data points
                    if len(df) < min_data_points:
                        raise ValueError(f"Insufficient data points: {len(df)} (minimum: {min_data_points})")
                    
                    # Check required columns
                    if required_columns:
                        missing_cols = set(required_columns) - set(df.columns)
                        if missing_cols:
                            raise ValueError(f"Missing required columns: {missing_cols}")
                    
                    # Check for empty DataFrame
                    if df.empty:
                        raise ValueError("Empty DataFrame provided")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def cache_analysis_results(
        cache_dir: str = "cache/analysis",
        cache_key_func: Optional[Callable] = None,
        ttl_seconds: int = 3600
    ) -> Callable:
        """Cache analysis results to avoid redundant computation."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    # Default cache key generation
                    key_data = str(args) + str(sorted(kwargs.items()))
                    cache_key = hashlib.md5(key_data.encode()).hexdigest()
                
                cache_path = Path(cache_dir) / f"{func.__name__}_{cache_key}.json"
                ensure_directory(cache_path.parent)
                
                # Check if cache exists and is valid
                if cache_path.exists():
                    try:
                        cache_data = safe_json_load(cache_path)
                        cache_time = pd.to_datetime(cache_data.get("timestamp", "1970-01-01"))
                        if (get_current_datetime() - cache_time).total_seconds() < ttl_seconds:
                            logging.info(f"Using cached result for {func.__name__}")
                            return cache_data["result"]
                    except Exception as e:
                        logging.warning(f"Failed to load cache: {e}")
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                
                try:
                    cache_data = {
                        "result": result,
                        "timestamp": format_datetime(get_current_datetime()),
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                    safe_json_dump(cache_data, cache_path)
                    logging.info(f"Cached result for {func.__name__}")
                except Exception as e:
                    logging.warning(f"Failed to cache result: {e}")
                
                return result
            
            return wrapper
        return decorator


class DataAccessProtectionDecorator:
    """Decorators for data access protection and security."""
    
    @staticmethod
    def secure_file_access(
        allowed_extensions: List[str] = [".parquet", ".csv", ".json"],
        max_file_size_mb: int = 1000,
        read_only: bool = True
    ) -> Callable:
        """Secure file access operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find file path arguments
                file_paths = []
                for arg in args:
                    if isinstance(arg, (str, Path)):
                        file_paths.append(Path(arg))
                for value in kwargs.values():
                    if isinstance(value, (str, Path)):
                        file_paths.append(Path(value))
                
                for file_path in file_paths:
                    # Check file extension
                    if file_path.suffix.lower() not in allowed_extensions:
                        raise ValueError(f"File extension not allowed: {file_path.suffix}")
                    
                    # Check file size
                    if file_path.exists():
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        if file_size_mb > max_file_size_mb:
                            raise ValueError(f"File too large: {file_size_mb:.2f}MB (max: {max_file_size_mb}MB)")
                    
                    # Check read-only mode
                    if read_only and file_path.exists():
                        if not os.access(file_path, os.R_OK):
                            raise ValueError(f"No read permission for file: {file_path}")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def validate_data_integrity(
        checksum_validation: bool = True,
        backup_before_modify: bool = True
    ) -> Callable:
        """Validate data integrity before and after operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find file paths that might be modified
                file_paths = []
                for arg in args:
                    if isinstance(arg, (str, Path)):
                        file_paths.append(Path(arg))
                for value in kwargs.values():
                    if isinstance(value, (str, Path)):
                        file_paths.append(Path(value))
                
                # Calculate checksums before operation
                checksums_before = {}
                if checksum_validation:
                    for file_path in file_paths:
                        if file_path.exists():
                            with open(file_path, 'rb') as f:
                                checksums_before[str(file_path)] = hashlib.md5(f.read()).hexdigest()
                
                # Create backups if needed
                backups = {}
                if backup_before_modify:
                    for file_path in file_paths:
                        if file_path.exists():
                            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                            import shutil
                            shutil.copy2(file_path, backup_path)
                            backups[str(file_path)] = backup_path
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Validate checksums after operation
                    if checksum_validation:
                        for file_path in file_paths:
                            if file_path.exists() and str(file_path) in checksums_before:
                                with open(file_path, 'rb') as f:
                                    checksum_after = hashlib.md5(f.read()).hexdigest()
                                if checksum_after != checksums_before[str(file_path)]:
                                    logging.warning(f"File checksum changed: {file_path}")
                    
                    return result
                
                except Exception as e:
                    # Restore backups on error
                    if backup_before_modify:
                        for original_path, backup_path in backups.items():
                            if backup_path.exists():
                                import shutil
                                shutil.copy2(backup_path, original_path)
                                backup_path.unlink()
                    raise e
            
            return wrapper
        return decorator
    
    @staticmethod
    def rate_limit_operations(
        max_operations_per_minute: int = 60,
        operation_type: str = "file_access"
    ) -> Callable:
        """Rate limit operations to prevent system overload."""
        def decorator(func: Callable) -> Callable:
            # Thread-safe rate limiting
            lock = threading.Lock()
            operation_times = []
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with lock:
                    current_time = time.time()
                    # Remove operations older than 1 minute
                    operation_times[:] = [t for t in operation_times if current_time - t < 60]
                    
                    # Check if we're at the rate limit
                    if len(operation_times) >= max_operations_per_minute:
                        sleep_time = 60 - (current_time - operation_times[0])
                        if sleep_time > 0:
                            logging.info(f"Rate limiting {operation_type}: sleeping {sleep_time:.2f}s")
                            time.sleep(sleep_time)
                    
                    # Record this operation
                    operation_times.append(current_time)
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


class PerformanceMonitoringDecorator:
    """Decorators for performance monitoring and optimization."""
    
    @staticmethod
    def monitor_execution_time(
        log_threshold_seconds: float = 1.0,
        profile_memory: bool = False
    ) -> Callable:
        """Monitor execution time and optionally memory usage."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = None
                
                if profile_memory:
                    try:
                        import psutil
                        process = psutil.Process()
                        start_memory = process.memory_info().rss / 1024 / 1024  # MB
                    except ImportError:
                        logging.warning("psutil not available for memory profiling")
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    
                    if execution_time >= log_threshold_seconds:
                        logging.info(f"{func.__name__} executed in {execution_time:.2f}s")
                    
                    if profile_memory and start_memory is not None:
                        try:
                            end_memory = process.memory_info().rss / 1024 / 1024  # MB
                            memory_delta = end_memory - start_memory
                            if abs(memory_delta) > 10:  # Log if memory change > 10MB
                                logging.info(f"{func.__name__} memory delta: {memory_delta:+.2f}MB")
                        except Exception:
                            pass
            
            return wrapper
        return decorator
    
    @staticmethod
    def optimize_dataframe_operations(
        chunk_size: int = 10000,
        use_dask: bool = False
    ) -> Callable:
        """Optimize DataFrame operations for large datasets."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Find large DataFrames
                large_dataframes = []
                for arg in args:
                    if isinstance(arg, pd.DataFrame) and len(arg) > chunk_size:
                        large_dataframes.append(arg)
                for value in kwargs.values():
                    if isinstance(value, pd.DataFrame) and len(value) > chunk_size:
                        large_dataframes.append(value)
                
                if large_dataframes and use_dask:
                    try:
                        import dask.dataframe as dd
                        logging.info("Using Dask for large DataFrame operations")
                        
                        # Convert large DataFrames to Dask DataFrames
                        new_args = []
                        for arg in args:
                            if isinstance(arg, pd.DataFrame) and len(arg) > chunk_size:
                                new_args.append(dd.from_pandas(arg, npartitions=4))
                            else:
                                new_args.append(arg)
                        
                        new_kwargs = {}
                        for key, value in kwargs.items():
                            if isinstance(value, pd.DataFrame) and len(value) > chunk_size:
                                new_kwargs[key] = dd.from_pandas(value, npartitions=4)
                            else:
                                new_kwargs[key] = value
                        
                        result = func(*new_args, **new_kwargs)
                        
                        # Convert back to pandas if needed
                        if isinstance(result, dd.DataFrame):
                            result = result.compute()
                        
                        return result
                    except ImportError:
                        logging.warning("Dask not available, using standard pandas operations")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


# Convenience decorator combinations
class BacktestingDecorators:
    """Convenience class combining multiple decorators for common use cases."""
    
    @staticmethod
    def data_processing_pipeline(
        required_columns: Optional[List[str]] = None,
        validate_price_data: bool = True,
        handle_missing_data: bool = True
    ) -> Callable:
        """Combined decorator for data processing pipeline operations."""
        def decorator(func: Callable) -> Callable:
            # Apply multiple decorators
            decorated_func = func
            
            if required_columns or True:  # Always apply DataFrame formatting
                decorated_func = DataFormattingDecorator.ensure_dataframe_format(
                    required_columns=required_columns
                )(decorated_func)
            
            if validate_price_data:
                decorated_func = DataFormattingDecorator.validate_price_data()(decorated_func)
            
            if handle_missing_data:
                decorated_func = DataFormattingDecorator.ensure_numeric_features()(decorated_func)
            
            decorated_func = AnalysisProtectionDecorator.prevent_lookahead_bias()(decorated_func)
            decorated_func = AnalysisProtectionDecorator.validate_analysis_inputs()(decorated_func)
            decorated_func = PerformanceMonitoringDecorator.monitor_execution_time()(decorated_func)
            
            return decorated_func
        
        return decorator
    
    @staticmethod
    def secure_file_operations(
        allowed_extensions: List[str] = [".parquet", ".csv", ".json"],
        backup_before_modify: bool = True
    ) -> Callable:
        """Combined decorator for secure file operations."""
        def decorator(func: Callable) -> Callable:
            decorated_func = DataAccessProtectionDecorator.secure_file_access(
                allowed_extensions=allowed_extensions
            )(func)
            
            decorated_func = DataAccessProtectionDecorator.validate_data_integrity(
                backup_before_modify=backup_before_modify
            )(decorated_func)
            
            decorated_func = DataAccessProtectionDecorator.rate_limit_operations()(decorated_func)
            decorated_func = PerformanceMonitoringDecorator.monitor_execution_time()(decorated_func)
            
            return decorated_func
        
        return decorator
    
    @staticmethod
    def analysis_operations(
        cache_results: bool = True,
        prevent_lookahead: bool = True,
        min_data_points: int = 100
    ) -> Callable:
        """Combined decorator for analysis operations."""
        def decorator(func: Callable) -> Callable:
            decorated_func = func
            
            if prevent_lookahead:
                decorated_func = AnalysisProtectionDecorator.prevent_lookahead_bias()(decorated_func)
            
            decorated_func = AnalysisProtectionDecorator.validate_analysis_inputs(
                min_data_points=min_data_points
            )(decorated_func)
            
            if cache_results:
                decorated_func = AnalysisProtectionDecorator.cache_analysis_results()(decorated_func)
            
            decorated_func = PerformanceMonitoringDecorator.monitor_execution_time()(decorated_func)
            
            return decorated_func
        
        return decorator