"""Validation Decorators Component

Simplified decorators for data validation.
Extracted from raw_data_quality_checker.py
"""

import functools
from datetime import datetime
from typing import Any, Callable
import pandas as pd
import logging
import numpy as np
import time

from src.utils.logger import system_logger
from .data_utils import fix_datetime_index
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

def validate_data(func: Callable) -> Callable:
    """Comprehensive data validation decorator using proper quality tools.
    
    This decorator provides comprehensive data validation using the proper
    data quality tools from src/utils/data/quality/.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        logger = system_logger.getChild("ValidationDecorator")
        
        # Basic validation
        if data is None or data.empty:
            logger.error(f"❌ {func.__name__}: Empty or None data provided")
            if func.__name__ == "validate_raw_data":
                return self._create_error_result("Empty or None data provided", kwargs)
            return None
        
        # Comprehensive quality validation if tools are available
        try:
            from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
            quality_scorer = get_quality_scorer()
            
            # Perform comprehensive quality assessment
            quality_assessment = quality_scorer.assess_data_quality(
                data,
                context="data_collection",
                step_name=f"validation_decorator_{func.__name__}",
                data_type="klines"
            )
            
            # Log quality assessment results
            logger.info(f"📊 Data quality assessment: {quality_assessment.overall_score:.2f} ({quality_assessment.level.value})")
            
            # Handle quality issues
            if quality_assessment.level.value in ['poor', 'critical']:
                logger.warning(f"⚠️ Low data quality detected: {quality_assessment.issues}")
                
                # For critical quality issues, return error result
                if quality_assessment.level.value == 'critical' and func.__name__ == "validate_raw_data":
                    return self._create_error_result(f"Critical data quality issues: {quality_assessment.issues}", kwargs)
            
        except ImportError:
            logger.info("ℹ️ Comprehensive quality tools not available, using basic validation")
        except Exception as e:
            logger.warning(f"⚠️ Comprehensive quality validation failed: {e}")
        
        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"❌ {func.__name__}: Missing required columns: {missing_columns}")
            if func.__name__ == "validate_raw_data":
                return self._create_error_result(f"Missing required columns: {missing_columns}", kwargs)
            return None
            
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning(f"⚠️ {func.__name__}: Data does not have datetime index, attempting to fix...")
            fixed_data = fix_datetime_index(data)
            
            if fixed_data is not None:
                logger.info(f"✅ {func.__name__}: Successfully created datetime index")
                data = fixed_data
            else:
                logger.error(f"❌ {func.__name__}: Failed to create datetime index")
                if func.__name__ == "validate_raw_data":
                    return self._create_error_result("Failed to create datetime index", kwargs)
                return None
                
        return func(self, data, *args, **kwargs)
        
    return wrapper

def log_validation_progress(func: Callable) -> Callable:
    """Decorator to log validation progress and timing.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        logger = system_logger.getChild("ValidationProgress")
        start_time = datetime.now()
        
        logger.info(f'🚀 {func.__name__}: Starting validation...')
        
        try:
            result = func(self, data, *args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if func.__name__ == 'validate_raw_data' and isinstance(result, tuple):
                validation_results, _ = result
                status = '✅ PASSED' if validation_results.get('validation_passed', False) else '❌ FAILED'
                logger.info(f'{status} {func.__name__}: Completed in {duration:.2f}s')
            else:
                logger.info(f'✅ {func.__name__}: Completed in {duration:.2f}s')
                
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.exception(f'❌ {func.__name__}: Failed after {duration:.2f}s - {e}')
            raise
            
    return wrapper

def handle_validation_errors(func: Callable) -> Callable:
    """Decorator to handle validation errors gracefully.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        logger = system_logger.getChild("ErrorHandler")
        
        try:
            return func(self, data, *args, **kwargs)
        except Exception as e:
            logger.exception(f'❌ {func.__name__}: Validation error: {e}')
            
            if func.__name__ == 'validate_raw_data':
                return self._create_error_result(f"Validation error: {str(e)}", kwargs)
            return None
            
    return wrapper

def ensure_data_types(func: Callable) -> Callable:
    """Decorator to ensure proper data types for OHLCV columns.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        logger = system_logger.getChild("DataTypeValidator")
        
        if data is not None and not data.empty:
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in ohlcv_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f'⚠️ {func.__name__}: Failed to convert {col} to numeric: {e}')
                        
            # Handle NaN values after conversion
            if data[ohlcv_columns].isna().any().any():
                logger.warning(f'⚠️ {func.__name__}: NaN values detected after type conversion')
                data[ohlcv_columns] = data[ohlcv_columns].fillna(method='ffill').fillna(method='bfill')
                
        return func(self, data, *args, **kwargs)
        
    return wrapper

def auto_fix_data_quality_issues(func: Callable) -> Callable:
    """Decorator that automatically fixes data quality issues before calling the decorated function.
    
    This decorator is specifically designed to address irregular interval warnings.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = system_logger.getChild("AutoFixDecorator")
        
        # Find DataFrame in arguments
        data = None
        symbol = kwargs.get('symbol', 'UNKNOWN')
        exchange = kwargs.get('exchange', 'UNKNOWN')
        
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                data = arg
                break
                
        if data is None:
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    data = value
                    break
                    
        if data is not None and not data.empty:
            # Check for irregular intervals
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                tolerance_percentage = 0.15
                tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
                irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds = tolerance_seconds)]
                irregular_ratio = len(irregular_intervals) / len(time_diffs)
                
                time_diffs_seconds = time_diffs.dt.total_seconds()
                mean_interval = time_diffs_seconds.mean()
                std_interval = time_diffs_seconds.std()
                cv = std_interval / mean_interval if mean_interval > 0 else 0
                
                if irregular_ratio > 0.01 or cv > 0.2:
                    logger.info(f'🔧 Auto-fixing irregular intervals for {func.__name__} (ratio: {irregular_ratio:.3f}, CV: {cv:.3f})')
                    
                    # Try to fix using the object's method if available
                    self_obj = args[0] if len(args) > 0 else None
                    if hasattr(self_obj, 'fix_irregular_intervals_automatically'):
                        fixed_data = self_obj.fix_irregular_intervals_automatically(data, symbol, exchange)
                    else:
                        # Fallback: simple resampling
                        freq = f'{int(expected_interval.total_seconds())}S'
                        fixed_data = data.resample(freq).ffill()
                        
                    # Update arguments with fixed data
                    if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                        new_args = (fixed_data, *args[1:])
                        return func(*new_args, **kwargs)
                    else:
                        new_kwargs = kwargs.copy()
                        for key, value in kwargs.items():
                            if isinstance(value, pd.DataFrame):
                                new_kwargs[key] = fixed_data
                                break
                        return func(*args, **new_kwargs)
                        
        return func(*args, **kwargs)
        
    return wrapper