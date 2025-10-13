"""
Common utilities for period analysis to eliminate code duplication.

This module provides shared utilities for validation, error handling, and logging
across the period analysis components.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import logging
from contextlib import contextmanager
import time

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class AnalysisError(Exception):
    """Custom exception for analysis errors."""
    pass


class PeriodAnalysisUtils:
    """Common utilities for period analysis components."""
    
    @staticmethod
    def validate_dataframe(data: pd.DataFrame, 
                          min_length: int = 1,
                          required_columns: Optional[List[str]] = None,
                          operation_name: str = "operation") -> None:
        """
        Validate DataFrame with comprehensive error checking and detailed logging.
        
        Args:
            data: DataFrame to validate
            min_length: Minimum required length
            required_columns: List of required column names
            operation_name: Name of operation for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        tprint_debug(f"🔍 Validating DataFrame for {operation_name}...")
        
        # Type validation
        if not isinstance(data, pd.DataFrame):
            tprint_error(f"❌ {operation_name}: Invalid data type - expected pandas DataFrame, got {type(data).__name__}")
            raise ValidationError(f"{operation_name}: Expected pandas DataFrame, got {type(data).__name__}")
        
        # Empty DataFrame check
        if len(data) == 0:
            tprint_error(f"❌ {operation_name}: DataFrame is empty")
            raise ValidationError(f"{operation_name}: DataFrame cannot be empty")
        
        # Length validation
        if len(data) < min_length:
            tprint_error(f"❌ {operation_name}: Insufficient data - got {len(data)} rows, need at least {min_length}")
            raise ValidationError(f"{operation_name}: Insufficient data ({len(data)} < {min_length} required)")
        
        # Column validation
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                available_columns = list(data.columns)
                tprint_error(f"❌ {operation_name}: Missing required columns: {missing_columns}")
                tprint_error(f"📊 Available columns: {available_columns}")
                raise ValidationError(f"{operation_name}: Missing required columns: {missing_columns}")
        
        # Data quality checks
        if data.isnull().all().any():
            null_columns = data.columns[data.isnull().all()].tolist()
            tprint_warning(f"⚠️ {operation_name}: Found columns with all null values: {null_columns}")
        
        tprint_debug(f"✅ DataFrame validation passed - shape: {data.shape}, columns: {list(data.columns)}")
    
    @staticmethod
    def validate_series(series: pd.Series,
                       min_length: int = 1,
                       expected_dtype: Optional[type] = None,
                       operation_name: str = "operation") -> None:
        """
        Validate Series with comprehensive error checking.
        
        Args:
            series: Series to validate
            min_length: Minimum required length
            expected_dtype: Expected data type
            operation_name: Name of operation for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(series, pd.Series):
            raise ValidationError(f"{operation_name}: Expected pandas Series, got {type(series).__name__}")
        
        if len(series) == 0:
            raise ValidationError(f"{operation_name}: Series cannot be empty")
        
        if len(series) < min_length:
            raise ValidationError(f"{operation_name}: Insufficient data ({len(series)} < {min_length} required)")
        
        if expected_dtype and not isinstance(series.iloc[0], expected_dtype):
            raise ValidationError(f"{operation_name}: Expected {expected_dtype.__name__}, got {type(series.iloc[0]).__name__}")
    
    @staticmethod
    def validate_periods(periods: List[int],
                        min_period: int = 1,
                        max_period: int = 1000,
                        operation_name: str = "operation") -> None:
        """
        Validate list of periods.
        
        Args:
            periods: List of periods to validate
            min_period: Minimum allowed period
            max_period: Maximum allowed period
            operation_name: Name of operation for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(periods, list):
            raise ValidationError(f"{operation_name}: Expected list of periods, got {type(periods).__name__}")
        
        if not periods:
            raise ValidationError(f"{operation_name}: Periods list cannot be empty")
        
        for i, period in enumerate(periods):
            if not isinstance(period, int):
                raise ValidationError(f"{operation_name}: Period {i} must be integer, got {type(period).__name__}")
            
            if not (min_period <= period <= max_period):
                raise ValidationError(f"{operation_name}: Period {period} outside valid range [{min_period}, {max_period}]")
    
    @staticmethod
    def safe_operation(operation_func, 
                     operation_name: str,
                     *args, 
                     **kwargs) -> Any:
        """
        Safely execute an operation with comprehensive error handling and logging.
        
        Args:
            operation_func: Function to execute
            operation_name: Name of operation for logging
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the operation
            
        Raises:
            AnalysisError: If operation fails
        """
        tprint_debug(f"🔍 Starting {operation_name}...")
        tprint_debug(f"📊 Operation args: {len(args)} positional, {len(kwargs)} keyword")
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Validate result
            if result is None:
                tprint_warning(f"⚠️ {operation_name} returned None - this may indicate an issue")
            elif isinstance(result, (list, dict)) and len(result) == 0:
                tprint_warning(f"⚠️ {operation_name} returned empty {type(result).__name__}")
            
            tprint_success(f"✅ {operation_name} completed in {execution_time:.3f}s")
            tprint_debug(f"📊 Result type: {type(result).__name__}, size: {getattr(result, '__len__', lambda: 'N/A')()}")
            return result
            
        except ValidationError as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ {operation_name} validation failed after {execution_time:.3f}s: {e}")
            raise AnalysisError(f"{operation_name} validation failed: {e}") from e
        except AnalysisError as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ {operation_name} analysis failed after {execution_time:.3f}s: {e}")
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ {operation_name} failed after {execution_time:.3f}s: {e}")
            tprint_error(f"📊 Error type: {type(e).__name__}")
            tprint_error(f"📊 Error details: {str(e)}")
            raise AnalysisError(f"{operation_name} failed: {e}") from e
    
    @staticmethod
    def log_operation_start(operation_name: str, **kwargs) -> None:
        """Log the start of an operation with parameters."""
        tprint_debug(f"🔍 Starting {operation_name}...")
        if kwargs:
            params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            tprint_debug(f"📊 Parameters: {params}")
    
    @staticmethod
    def log_operation_success(operation_name: str, result_info: str = "", execution_time: float = 0.0) -> None:
        """Log successful completion of an operation."""
        if execution_time > 0:
            tprint_success(f"✅ {operation_name} completed in {execution_time:.3f}s")
        else:
            tprint_success(f"✅ {operation_name} completed")
        
        if result_info:
            tprint_debug(f"📊 Result: {result_info}")
    
    @staticmethod
    def log_operation_error(operation_name: str, error: Exception, execution_time: float = 0.0) -> None:
        """Log operation failure."""
        if execution_time > 0:
            tprint_error(f"❌ {operation_name} failed after {execution_time:.3f}s: {error}")
        else:
            tprint_error(f"❌ {operation_name} failed: {error}")
    
    @staticmethod
    def detect_frequency(data: pd.DataFrame) -> str:
        """
        Detect the frequency of the data.
        
        Args:
            data: DataFrame with DatetimeIndex
            
        Returns:
            Detected frequency string
            
        Raises:
            ValidationError: If data is invalid
        """
        PeriodAnalysisUtils.validate_dataframe(data, min_length=2, operation_name="frequency_detection")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValidationError("DataFrame must have DatetimeIndex to determine frequency")
        
        try:
            time_diffs = data.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            
            if pd.isna(median_diff):
                raise ValidationError("Failed to calculate time differences")
            
            # Convert to minutes and determine frequency
            if median_diff < pd.Timedelta(minutes=1):
                return 'sub-minute'
            elif median_diff < pd.Timedelta(minutes=5):
                return '1m'
            elif median_diff < pd.Timedelta(minutes=10):
                return '5m'
            elif median_diff < pd.Timedelta(minutes=20):
                return '15m'
            elif median_diff < pd.Timedelta(minutes=90):
                return '60m'
            elif median_diff < pd.Timedelta(hours=2):
                return '4h'
            elif median_diff < pd.Timedelta(hours=6):
                return '1d'
            else:
                return 'weekly'
        except Exception as e:
            raise AnalysisError(f"Frequency detection failed: {e}") from e
    
    @staticmethod
    def get_timeframe_minutes(data: pd.DataFrame) -> int:
        """
        Get timeframe in minutes from DataFrame.
        
        Args:
            data: DataFrame with DatetimeIndex
            
        Returns:
            Timeframe in minutes
            
        Raises:
            ValidationError: If data is invalid
        """
        PeriodAnalysisUtils.validate_dataframe(data, min_length=2, operation_name="timeframe_detection")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValidationError("DataFrame must have DatetimeIndex to determine timeframe")
        
        try:
            time_diffs = data.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            timeframe_minutes = int(median_diff.total_seconds() / 60)
            
            if timeframe_minutes <= 0:
                raise ValidationError(f"Invalid timeframe detected: {timeframe_minutes} minutes")
            
            return timeframe_minutes
        except Exception as e:
            raise AnalysisError(f"Timeframe detection failed: {e}") from e
    
    @staticmethod
    def find_pattern_periods(pattern: pd.Series) -> List[int]:
        """
        Find periods in a boolean pattern.
        
        Args:
            pattern: Boolean Series representing a pattern
            
        Returns:
            List of pattern lengths
            
        Raises:
            ValidationError: If pattern is invalid
        """
        PeriodAnalysisUtils.validate_series(pattern, expected_dtype=bool, operation_name="pattern_analysis")
        
        try:
            pattern_lengths = []
            in_pattern = False
            current_length = 0
            
            for is_true in pattern:
                if is_true:
                    if not in_pattern:
                        in_pattern = True
                        current_length = 1
                    else:
                        current_length += 1
                else:
                    if in_pattern:
                        pattern_lengths.append(current_length)
                        in_pattern = False
                        current_length = 0
            
            # Handle case where pattern ends with True values
            if in_pattern:
                pattern_lengths.append(current_length)
            
            return pattern_lengths
        except Exception as e:
            raise AnalysisError(f"Pattern analysis failed: {e}") from e
    
    @staticmethod
    def calculate_confidence_score(periods: List[int], 
                                 characteristics: Dict[str, Any]) -> float:
        """
        Calculate confidence score for selected periods.
        
        Args:
            periods: List of selected periods
            characteristics: Data characteristics
            
        Returns:
            Confidence score between 0 and 1
        """
        if not periods:
            return 0.0
        
        try:
            score = 0.0
            data_length = characteristics.get('data_length', 0)
            
            # Data sufficiency score
            if data_length > 1000:
                score += 0.3
            elif data_length > 500:
                score += 0.2
            elif data_length > 100:
                score += 0.1
            
            # Period diversity score
            if len(periods) >= 3:
                score += 0.2
            elif len(periods) >= 2:
                score += 0.1
            
            # Analysis completeness score
            analysis_components = ['volatility_clusters', 'trend_cycles', 'volume_patterns']
            completed_analyses = sum(1 for comp in analysis_components if comp in characteristics)
            score += (completed_analyses / len(analysis_components)) * 0.3
            
            # Period reasonableness score
            reasonable_periods = sum(1 for p in periods if isinstance(p, int) and 2 <= p <= data_length // 4)
            score += (reasonable_periods / len(periods)) * 0.2
            
            return min(score, 1.0)
        except Exception as e:
            tprint_warning(f"Confidence calculation failed: {e}")
            return 0.0


@contextmanager
def performance_monitoring(operation_name: str):
    """Context manager for performance monitoring."""
    tprint_debug(f"🔍 Starting performance monitoring for: {operation_name}")
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        tprint_performance(f"Operation {operation_name}: {execution_time:.3f}s")
        tprint_debug(f"✅ Performance monitoring complete for: {operation_name}")


def safe_validate_and_execute(validation_func, execution_func, operation_name: str, *args, **kwargs):
    """
    Safely validate inputs and execute operation with comprehensive error handling.
    
    Args:
        validation_func: Function to validate inputs
        execution_func: Function to execute
        operation_name: Name of operation for logging
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Result of execution
        
    Raises:
        ValidationError: If validation fails
        AnalysisError: If execution fails
    """
    # Validate inputs
    validation_func(*args, **kwargs)
    
    # Execute operation safely
    return PeriodAnalysisUtils.safe_operation(execution_func, operation_name, *args, **kwargs)