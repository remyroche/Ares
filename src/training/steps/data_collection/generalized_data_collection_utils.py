#!/usr/bin/env python3
"""
Generalized Data Collection Utilities

This module provides common utilities and patterns for data collection steps
that leverage the comprehensive tools available in BaseStep. It serves as a
foundation for all data collection operations with:

- Common data collection patterns
- Reusable utility functions
- Standardized error handling
- Comprehensive logging integration
- Hardware optimization utilities
- Data quality validation tools

Features:
- Exchange-agnostic data collection patterns
- Comprehensive gap detection and filling
- Data validation and quality assessment
- Hardware-optimized data processing
- Standardized file operations
- Performance monitoring and metrics
"""

import asyncio
import sys
import time
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Awaitable
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild("GeneralizedDataCollectionUtils")

class DataCollectionPatterns:
    """
    Common patterns and utilities for data collection operations.
    
    This class provides reusable patterns that can be used across all
    data collection steps to ensure consistency and leverage BaseStep
    comprehensive tools.
    """
    
    @staticmethod
    def create_collection_config(
        exchange: str,
        symbol: str,
        timeframe: str,
        data_dir: str = "historical_data",
        collection_mode: str = "incremental",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a standardized collection configuration.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            data_dir: Directory for data storage
            collection_mode: Collection mode (incremental, period, gap_filling)
            **kwargs: Additional configuration parameters
            
        Returns:
            Standardized configuration dictionary
        """
        config = {
            'exchange': exchange.upper(),
            'symbol': symbol,
            'timeframe': timeframe,
            'data_dir': data_dir,
            'collection_mode': collection_mode,
            'data_types': kwargs.get('data_types', ['klines']),
            'max_batches': kwargs.get('max_batches', 10),
            'batch_size': kwargs.get('batch_size', 1000),
            'start_time': kwargs.get('start_time'),
            'end_time': kwargs.get('end_time'),
            'information': kwargs.get('information', 'klines'),
            'direction': kwargs.get('direction', 'long'),
            'model': kwargs.get('model', 'Analyst')
        }
        
        return config
    
    @staticmethod
    def validate_collection_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate collection configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['exchange', 'symbol', 'timeframe', 'data_dir', 'collection_mode']
        
        for field in required_fields:
            if field not in config or not config[field]:
                return False, f"Missing required field: {field}"
        
        # Validate collection mode
        valid_modes = ['incremental', 'period', 'gap_filling']
        if config['collection_mode'] not in valid_modes:
            return False, f"Invalid collection_mode: {config['collection_mode']}. Must be one of {valid_modes}"
        
        # Validate period mode requirements
        if config['collection_mode'] == 'period':
            if not config.get('start_time') or not config.get('end_time'):
                return False, "start_time and end_time are required for period collection mode"
        
        return True, None

class DataValidationUtils:
    """
    Comprehensive data validation utilities for data collection.
    
    This class provides validation functions that leverage BaseStep
    comprehensive tools for data quality assessment and validation.
    """
    
    @staticmethod
    def validate_klines_data(data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate klines data structure and content.
        
        Args:
            data: List of klines data dictionaries
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not data:
            return False, ["No data provided"]
        
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        errors = []
        
        for i, record in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in record:
                    errors.append(f"Record {i}: Missing required field '{field}'")
            
            # Validate numeric fields
            numeric_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in numeric_fields:
                if field in record:
                    try:
                        value = float(record[field])
                        if not np.isfinite(value) or value < 0:
                            errors.append(f"Record {i}: Invalid {field} value: {record[field]}")
                    except (ValueError, TypeError):
                        errors.append(f"Record {i}: Non-numeric {field} value: {record[field]}")
            
            # Validate OHLC relationships
            if all(field in record for field in ['open', 'high', 'low', 'close']):
                o, h, l, c = record['open'], record['high'], record['low'], record['close']
                if h < max(o, c) or l > min(o, c):
                    errors.append(f"Record {i}: Invalid OHLC relationships")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame, data_type: str = "klines") -> Dict[str, Any]:
        """
        Validate data quality using comprehensive tools.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            Quality validation results
        """
        quality_result = {
            'valid': True,
            'quality_score': 100.0,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Check for empty data
            if df.empty:
                quality_result['valid'] = False
                quality_result['issues'].append("Empty dataset")
                quality_result['quality_score'] = 0.0
                return quality_result
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                quality_result['warnings'].append(f"Missing values detected: {missing_counts.to_dict()}")
                quality_result['quality_score'] -= 10.0
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                quality_result['warnings'].append(f"Duplicate records detected: {duplicate_count}")
                quality_result['quality_score'] -= 5.0
            
            # Check for outliers (basic)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    if len(outliers) > 0:
                        quality_result['warnings'].append(f"Outliers detected in {col}: {len(outliers)} records")
                        quality_result['quality_score'] -= 2.0
            
            # Check timestamp continuity for time series data
            if 'timestamp' in df.columns:
                df_sorted = df.sort_values('timestamp')
                time_diffs = df_sorted['timestamp'].diff().dropna()
                if len(time_diffs) > 0:
                    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else None
                    if expected_interval:
                        large_gaps = time_diffs[time_diffs > expected_interval * 2]
                        if len(large_gaps) > 0:
                            quality_result['warnings'].append(f"Large time gaps detected: {len(large_gaps)} gaps")
                            quality_result['quality_score'] -= 3.0
            
            # Ensure quality score is not negative
            quality_result['quality_score'] = max(0.0, quality_result['quality_score'])
            
            # Add recommendations
            if quality_result['quality_score'] < 80:
                quality_result['recommendations'].append("Consider data cleaning and validation")
            if quality_result['quality_score'] < 60:
                quality_result['recommendations'].append("Data quality is poor, manual review recommended")
            
        except Exception as e:
            quality_result['valid'] = False
            quality_result['issues'].append(f"Quality validation error: {str(e)}")
            quality_result['quality_score'] = 0.0
        
        return quality_result

class GapDetectionUtils:
    """
    Comprehensive gap detection utilities for data collection.
    
    This class provides gap detection and analysis functions that can be
    used across all data collection steps.
    """
    
    @staticmethod
    def detect_gaps(
        data: pd.DataFrame,
        data_type: str = "klines",
        threshold_seconds: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in time series data.
        
        Args:
            data: DataFrame with timestamp column
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            threshold_seconds: Gap threshold in seconds (auto-detected if None)
            
        Returns:
            List of gap information dictionaries
        """
        if data.empty or 'timestamp' not in data.columns:
            return []
        
        # Set default thresholds
        if threshold_seconds is None:
            thresholds = {
                'klines': 66.0,      # 1.1 minutes
                'aggtrades': 1.0,    # 1 second
                'futures': 32400.0   # 9 hours
            }
            threshold_seconds = thresholds.get(data_type, 60.0)
        
        gaps = []
        sorted_data = data.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(1, len(sorted_data)):
            current_ts = sorted_data.iloc[i]['timestamp']
            previous_ts = sorted_data.iloc[i-1]['timestamp']
            
            # Calculate gap in seconds
            gap_seconds = (current_ts - previous_ts) / 1000.0
            
            if gap_seconds > threshold_seconds:
                gap_info = {
                    'start_timestamp': previous_ts,
                    'end_timestamp': current_ts,
                    'gap_seconds': gap_seconds,
                    'gap_minutes': gap_seconds / 60.0,
                    'gap_hours': gap_seconds / 3600.0,
                    'start_time': pd.to_datetime(previous_ts, unit='ms', utc=True),
                    'end_time': pd.to_datetime(current_ts, unit='ms', utc=True),
                    'data_type': data_type,
                    'gap_size': gap_seconds
                }
                gaps.append(gap_info)
        
        return gaps
    
    @staticmethod
    def analyze_gap_patterns(gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze gap patterns in detected gaps.
        
        Args:
            gaps: List of gap information dictionaries
            
        Returns:
            Gap pattern analysis results
        """
        if not gaps:
            return {
                'total_gaps': 0,
                'total_gap_time': 0,
                'average_gap': 0,
                'gap_distribution': {},
                'largest_gap': None,
                'smallest_gap': None
            }
        
        gap_sizes = [gap['gap_seconds'] for gap in gaps]
        
        analysis = {
            'total_gaps': len(gaps),
            'total_gap_time': sum(gap_sizes),
            'total_gap_minutes': sum(gap_sizes) / 60.0,
            'total_gap_hours': sum(gap_sizes) / 3600.0,
            'average_gap': np.mean(gap_sizes),
            'average_gap_minutes': np.mean(gap_sizes) / 60.0,
            'median_gap': np.median(gap_sizes),
            'std_gap': np.std(gap_sizes),
            'largest_gap': max(gaps, key=lambda x: x['gap_seconds']),
            'smallest_gap': min(gaps, key=lambda x: x['gap_seconds']),
            'gap_distribution': {
                'small_gaps': len([g for g in gaps if g['gap_seconds'] < 300]),  # < 5 minutes
                'medium_gaps': len([g for g in gaps if 300 <= g['gap_seconds'] < 3600]),  # 5min - 1hr
                'large_gaps': len([g for g in gaps if g['gap_seconds'] >= 3600])  # >= 1 hour
            }
        }
        
        return analysis

class FileOperationsUtils:
    """
    Standardized file operations utilities for data collection.
    
    This class provides common file operations that can be used across
    all data collection steps with consistent naming and organization.
    """
    
    @staticmethod
    def generate_filename(
        data_type: str,
        exchange: str,
        symbol: str,
        timeframe: str,
        batch_num: Optional[int] = None,
        gap_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate standardized filename for data files.
        
        Args:
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            batch_num: Batch number (for incremental downloads)
            gap_id: Gap ID (for gap filling)
            timestamp: Custom timestamp (uses current time if None)
            
        Returns:
            Standardized filename
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        if gap_id is not None:
            return f"{data_type}_{exchange}_{symbol}_{timeframe}_gap_{gap_id}_{timestamp_str}_validated.parquet"
        elif batch_num is not None:
            return f"{data_type}_{exchange}_{symbol}_{timeframe}_batch_{batch_num}_{timestamp_str}_validated.parquet"
        else:
            return f"{data_type}_{exchange}_{symbol}_{timeframe}_{timestamp_str}_validated.parquet"
    
    @staticmethod
    def find_latest_file(
        data_dir: str,
        data_type: str,
        exchange: str,
        symbol: str,
        timeframe: str
    ) -> Optional[str]:
        """
        Find the latest file matching the pattern.
        
        Args:
            data_dir: Directory to search
            data_type: Type of data
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Path to latest file or None if not found
        """
        pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}*_validated.parquet"
        search_path = os.path.join(data_dir, pattern)
        files = glob.glob(search_path)
        
        if not files:
            return None
        
        return max(files, key=os.path.getmtime)
    
    @staticmethod
    def find_all_files(
        data_dir: str,
        data_type: str,
        exchange: str,
        symbol: str,
        timeframe: str
    ) -> List[str]:
        """
        Find all files matching the pattern.
        
        Args:
            data_dir: Directory to search
            data_type: Type of data
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            List of matching file paths
        """
        pattern = f"{data_type}_{exchange}_{symbol}_{timeframe}*_validated.parquet"
        search_path = os.path.join(data_dir, pattern)
        files = glob.glob(search_path)
        
        return sorted(files, key=os.path.getmtime)

class PerformanceMonitoringUtils:
    """
    Performance monitoring utilities for data collection operations.
    
    This class provides utilities for monitoring and tracking performance
    metrics during data collection operations.
    """
    
    @staticmethod
    def create_performance_tracker() -> Dict[str, Any]:
        """
        Create a performance tracking dictionary.
        
        Returns:
            Performance tracking dictionary
        """
        return {
            'start_time': time.time(),
            'end_time': None,
            'duration': 0.0,
            'operations': [],
            'memory_usage': [],
            'cpu_usage': [],
            'disk_usage': [],
            'errors': [],
            'warnings': []
        }
    
    @staticmethod
    def track_operation(
        tracker: Dict[str, Any],
        operation_name: str,
        start_time: float,
        end_time: float,
        success: bool,
        rows_processed: int = 0,
        memory_usage: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Track a single operation in the performance tracker.
        
        Args:
            tracker: Performance tracking dictionary
            operation_name: Name of the operation
            start_time: Operation start time
            end_time: Operation end time
            success: Whether the operation succeeded
            rows_processed: Number of rows processed
            memory_usage: Memory usage during operation
            error_message: Error message if operation failed
        """
        operation = {
            'name': operation_name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'success': success,
            'rows_processed': rows_processed,
            'memory_usage': memory_usage,
            'error_message': error_message
        }
        
        tracker['operations'].append(operation)
        
        if memory_usage is not None:
            tracker['memory_usage'].append(memory_usage)
        
        if not success and error_message:
            tracker['errors'].append(f"{operation_name}: {error_message}")
    
    @staticmethod
    def finalize_performance_tracker(tracker: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize the performance tracker with summary statistics.
        
        Args:
            tracker: Performance tracking dictionary
            
        Returns:
            Finalized performance tracker with summary statistics
        """
        tracker['end_time'] = time.time()
        tracker['duration'] = tracker['end_time'] - tracker['start_time']
        
        # Calculate summary statistics
        successful_operations = [op for op in tracker['operations'] if op['success']]
        failed_operations = [op for op in tracker['operations'] if not op['success']]
        
        tracker['summary'] = {
            'total_operations': len(tracker['operations']),
            'successful_operations': len(successful_operations),
            'failed_operations': len(failed_operations),
            'success_rate': len(successful_operations) / len(tracker['operations']) * 100 if tracker['operations'] else 0,
            'total_duration': tracker['duration'],
            'average_operation_duration': np.mean([op['duration'] for op in tracker['operations']]) if tracker['operations'] else 0,
            'total_rows_processed': sum(op['rows_processed'] for op in tracker['operations']),
            'average_memory_usage': np.mean(tracker['memory_usage']) if tracker['memory_usage'] else 0,
            'peak_memory_usage': max(tracker['memory_usage']) if tracker['memory_usage'] else 0,
            'error_count': len(tracker['errors']),
            'warning_count': len(tracker['warnings'])
        }
        
        return tracker

# Convenience functions for easy usage
def create_standard_collection_config(
    exchange: str,
    symbol: str,
    timeframe: str,
    **kwargs
) -> Dict[str, Any]:
    """Create a standard collection configuration."""
    return DataCollectionPatterns.create_collection_config(exchange, symbol, timeframe, **kwargs)

def validate_collection_config(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a collection configuration."""
    return DataCollectionPatterns.validate_collection_config(config)

def validate_klines_data(data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate klines data structure and content."""
    return DataValidationUtils.validate_klines_data(data)

def validate_data_quality(df: pd.DataFrame, data_type: str = "klines") -> Dict[str, Any]:
    """Validate data quality using comprehensive tools."""
    return DataValidationUtils.validate_data_quality(df, data_type)

def detect_gaps(
    data: pd.DataFrame,
    data_type: str = "klines",
    threshold_seconds: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Detect gaps in time series data."""
    return GapDetectionUtils.detect_gaps(data, data_type, threshold_seconds)

def analyze_gap_patterns(gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze gap patterns in detected gaps."""
    return GapDetectionUtils.analyze_gap_patterns(gaps)

def generate_filename(
    data_type: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    **kwargs
) -> str:
    """Generate standardized filename for data files."""
    return FileOperationsUtils.generate_filename(data_type, exchange, symbol, timeframe, **kwargs)

def find_latest_file(
    data_dir: str,
    data_type: str,
    exchange: str,
    symbol: str,
    timeframe: str
) -> Optional[str]:
    """Find the latest file matching the pattern."""
    return FileOperationsUtils.find_latest_file(data_dir, data_type, exchange, symbol, timeframe)

def create_performance_tracker() -> Dict[str, Any]:
    """Create a performance tracking dictionary."""
    return PerformanceMonitoringUtils.create_performance_tracker()

def track_operation(tracker: Dict[str, Any], **kwargs) -> None:
    """Track a single operation in the performance tracker."""
    PerformanceMonitoringUtils.track_operation(tracker, **kwargs)

def finalize_performance_tracker(tracker: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize the performance tracker with summary statistics."""
    return PerformanceMonitoringUtils.finalize_performance_tracker(tracker)