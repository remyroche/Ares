from src.utils.tprint import tprint

from ...core.decorators import handles_errors, traced
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

"""Unified Data Loader for Step1_5 Data.
from src.utils.logger import system_logger

This module provides secure, decorated access to data created by step1_5_data_converter.
It includes comprehensive validation for file paths, data formats, sizes, and string sanitization.
"""
import sys
import logging
import asyncio
from pathlib import Path
from typing import Any, Optional, Callable, Dict, List

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common_operations import (
    safe_read_parquet,
    list_parquet_files,
    safe_file_exists,
    validate_dataframe_schema,
    safe_copy
)

# Import logger with fallback
try:
    from src.utils.logger import get_logger
    system_logger = get_logger(__name__)
except ImportError:
    system_logger = logging.getLogger(__name__)

# Import core domain functions with fallbacks
try:
    from src.utils.common_operations import (
        guard_dataframe_nulls,
        secure_file_path,
        validate_dataframe_schema,
        validate_file_size,
        with_tracing_span,
        sanitize_string
    )
except ImportError:
    # Fallback decorators and functions
    def handles_errors(*args, **kwargs) -> Callable:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    def traced(*args, **kwargs) -> Callable:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    def secure_file_path(*args, **kwargs) -> Callable:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    def validate_dataframe_schema(*args, **kwargs) -> bool:
        return True

    def validate_file_size(*args, **kwargs) -> Callable:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    def guard_dataframe_nulls(*args, **kwargs) -> Any:
        return None

    def with_tracing_span(*args, **kwargs) -> Callable:
        def decorator(func: Callable) -> Callable:
            return func
        return decorator

    def sanitize_string(*args, **kwargs) -> str:
        return str(*args) if args else ""

    @log_important_calls
    class ParquetDatasetManager:
        def __init__(self, *args, **kwargs):
            pass

class UnifiedDataLoader:
    """Secure data loader for step1_5 unified data with comprehensive validation."""
    @log_important_calls

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the unified data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = system_logger.getChild('UnifiedDataLoader')
        self.expected_schema = {
            'timestamp': 'int64', 
            'open': 'float64', 
            'high': 'float64', 
            'low': 'float64', 
            'close': 'float64', 
            'volume': 'float64', 
            'exchange': 'string', 
            'symbol': 'string', 
            'timeframe': 'string', 
            'year': 'int16', 
            'month': 'int8', 
            'day': 'int8'
        }
        self.optional_columns = {
            'trade_volume': 'float64', 
            'trade_count': 'int64', 
            'avg_price': 'float64', 
            'min_price': 'float64',
            'max_price': 'float64',
            'volume_ratio': 'float64'
        }
        self.max_file_size = 100 * 1024 * 1024
        self.max_rows = 10000000

    @secure_file_path(allowed_dirs=['data_cache', 'data'])
    @validate_file_size(max_size_mb = 100)
    @traced(span_name='UnifiedDataLoader.load_unified_data')
    async def load_unified_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str = 'historical_data', 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Load unified data created by step1_5 with comprehensive validation.

        Args:
            symbol: Trading symbol (e.g. "ETHUSDT")
            exchange: Exchange name (e.g. "BINANCE")
            timeframe: Timeframe (e.g. "1m")
            data_dir: Data directory
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            columns: Specific columns to load

        Returns:
            DataFrame with unified data or None if loading fails
        """
        try:
            # Sanitize inputs
            symbol = sanitize_string(symbol)
            exchange = sanitize_string(exchange)
            timeframe = sanitize_string(timeframe)
            data_dir = sanitize_string(data_dir)

            # Construct data path
            data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            
            if not data_path.exists():
                self.logger.error(f"Data path does not exist: {data_path}")
                return None

            # Get list of parquet files
            parquet_files = list_parquet_files(data_path)
            if not parquet_files:
                self.logger.error(f"No parquet files found in {data_path}")
                return None

            # Load the most recent file
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)

            # Load data
            data = await self._load_data_file(latest_file, columns)
            if data is None:
                return None

            # Apply date filters if provided
            if start_date or end_date:
                data = self._apply_date_filters(data, start_date, end_date)

            # Validate schema
            if not validate_dataframe_schema(data, self.expected_schema):
                self.logger.warning("Schema validation failed, but continuing...")

            # Guard against nulls
            guard_dataframe_nulls(data)

            # Remove duplicate timestamps to prevent data quality issues
            if 'timestamp' in data.columns:
                initial_rows = len(data)
                data = data.drop_duplicates(subset=['timestamp'], keep='first')
                duplicates_removed = initial_rows - len(data)
                if duplicates_removed > 0:
                    self.logger.warning(f"🧹 Removed {duplicates_removed} duplicate timestamps during data loading")

            self.logger.info(f"Successfully loaded {len(data)} rows from {latest_file}")
            return data

        except Exception as e:
            self.logger.exception(f"Error loading unified data: {e}")
            return None

    async def _load_data_file(self, file_path: Path, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load data from a single file."""
        try:
            if columns:
                data = safe_read_parquet(file_path, columns = columns)
            else:
                data = safe_read_parquet(file_path)
            
            if data is None or len(data) == 0:
                self.logger.error(f"No data loaded from {file_path}")
                return None
            
            # Convert timestamp index to column if it exists
            if data.index.name == 'timestamp' or (hasattr(data.index, 'name') and data.index.name == 'timestamp'):
                data = data.reset_index()
                self.logger.info("Converted timestamp index to column")
            elif 'timestamp' not in data.columns and hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
                # If the index is datetime but not named 'timestamp', rename it
                data = data.reset_index()
                data = data.rename(columns={'index': 'timestamp'})
                self.logger.info("Converted datetime index to timestamp column")
            
            if len(data) > self.max_rows:
                self.logger.warning(f"Data has {len(data)} rows, exceeding limit of {self.max_rows}")
                data = data.head(self.max_rows)
            
            return data
            
        except Exception as e:
            self.logger.exception(f"Error loading file {file_path}: {e}")
            return None
    @log_all_calls

    def _apply_date_filters(self, data: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Apply date filters to the data."""
        try:
            if 'timestamp' not in data.columns:
                self.logger.warning("No timestamp column found, skipping date filters")
                return data

            # Convert timestamp to datetime if needed
            if data['timestamp'].dtype.kind in ['i', 'f']:
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            elif data['timestamp'].dtype.kind != 'M':
                data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Apply filters
            if start_date:
                start_dt = pd.to_datetime(start_date)
                data = data[data['timestamp'] >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                data = data[data['timestamp'] <= end_dt]

            return data

        except Exception as e:
            self.logger.exception(f"Error applying date filters: {e}")
            return data

    @traced(span_name='UnifiedDataLoader.get_data_info')
    async def get_data_info(self, symbol: str, exchange: str, timeframe: str, data_dir: str = 'historical_data') -> Dict[str, Any]:
        """Get information about available data without loading it.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            Dictionary with data information
        """
        try:
            data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            
            if not data_path.exists():
                return {'exists': False, 'error': f'Path does not exist: {data_path}'}

            parquet_files = list_parquet_files(data_path)
            
            if not parquet_files:
                return {'exists': False, 'error': 'No parquet files found'}

            # Get file information
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
            file_size = latest_file.stat().st_size
            file_count = len(parquet_files)

            return {
                'exists': True,
                'file_count': file_count,
                'latest_file': str(latest_file),
                'latest_file_size': file_size,
                'data_path': str(data_path)
            }

        except Exception as e:
            self.logger.exception(f"Error getting data info: {e}")
            return {'exists': False, 'error': str(e)}

    @traced(span_name='UnifiedDataLoader.validate_data_quality')
    async def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of loaded data.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'passed': True,
                'issues': [],
                'warnings': [],
                'stats': {}
            }

            # Check for required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                validation_results['passed'] = False
                validation_results['issues'].append(f'Missing required columns: {missing_columns}')

            # Check for null values
            null_counts = data[required_columns].isnull().sum()
            if null_counts.sum() > 0:
                validation_results['warnings'].append(f'Null values found: {null_counts.to_dict()}')

            # Check for negative prices
            price_columns = ['open', 'high', 'low', 'close']
            negative_prices = (data[price_columns] < 0).sum().sum()
            if negative_prices > 0:
                validation_results['issues'].append(f'Found {negative_prices} negative price values')
                validation_results['passed'] = False

            # Check OHLC consistency
            ohlc_errors = 0
            for idx, row in data.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and row['low'] <= row['close'] <= row['high']):
                    ohlc_errors += 1
            
            if ohlc_errors > 0:
                validation_results['warnings'].append(f'Found {ohlc_errors} OHLC consistency errors')

            # Generate statistics
            validation_results['stats'] = {
                'row_count': len(data),
                'column_count': len(data.columns),
                'null_counts': null_counts.to_dict(),
                'ohlc_errors': ohlc_errors,
                'negative_prices': negative_prices
            }

            return validation_results

        except Exception as e:
            self.logger.exception(f"Error validating data quality: {e}")
            return {
                'passed': False,
                'issues': [f'Validation error: {str(e)}'],
                'warnings': [],
                'stats': {}
            }

# Example usage
if __name__ == "__main__":
    async def main():
        loader = UnifiedDataLoader()
        
        # Get data info
        info = await loader.get_data_info("ETHUSDT", "BINANCE", "1m")
        tprint(f"Data info: {info}")
        
        # Load data
        data = await loader.load_unified_data("ETHUSDT", "BINANCE", "1m")
        if data is not None:
            tprint(f"Loaded data shape: {data.shape}")
            
            # Validate quality
            quality = await loader.validate_data_quality(data)
            tprint(f"Quality validation: {quality}")
        else:
            tprint("Failed to load data")

    asyncio.run(main())