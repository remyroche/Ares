from typing import Dict, List, Optional, Union, Any, Tuple
"""Unified Data Loader for Step1_5 Data.

This module provides secure, decorated access to data created by step1_5_data_converter.
It includes comprehensive validation for file paths, data formats, sizes, and string sanitization.
"""
from src.core.decorators import handles_errors, traced
import os
import sys
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import asyncio
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import safe_read_parquet, list_parquet_files, safe_file_exists, validate_dataframe_schema, safe_copy
try:
    from src.utils.logger import system_logger
except Exception as e:
    import logging
    system_logger = logging.getLogger(__name__)
try:
    from src.training.steps.step01_5_data_converter import ParquetDatasetManager
    from src.core.domain import guard_dataframe_nulls, secure_file_path, validate_dataframe_schema, validate_file_size, with_tracing_span, sanitize_string
from src.core.decorators.errors import handles_errors
except ImportError:

    def handles_errors(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def validates(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def traced(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def secure_file_path(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def validate_dataframe_schema(*args, **kwargs) -> bool:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def validate_file_size(*args, **kwargs) -> bool:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def sanitize_string(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator
    system_logger = logging.getLogger(__name__)

class UnifiedDataLoader:
    """Secure data loader for step1_5 unified data with comprehensive validation."""

    def __init__(self, config: Optional[dict[str, Any]]=None) -> None:
        """Initialize the unified data loader."

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = system_logger.getChild('UnifiedDataLoader')
        self.expected_schema = {'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64', 'exchange': 'string', 'symbol': 'string', 'timeframe': 'string', 'year': 'int16', 'month': 'int8', 'day': 'int8'}
        self.optional_columns = {'trade_volume': 'float64', 'trade_count': 'int64', 'avg_price': 'float64', 'min_price': 'float64', 'max_price': 'float64', 'volume_ratio': 'float64', 'funding_rate': 'float64'}
        self.max_file_size = 100 * 1024 * 1024
        self.max_rows = 10000000

    @secure_file_path(allowed_dirs=['data_cache', 'data'])
    @validate_file_size(max_size_mb=100)
    @traced(span_name='UnifiedDataLoader.load_unified_data')
    async def load_unified_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache', start_date: Optional[str]=None, end_date: Optional[str]=None, columns: Optional[list[str]]=None) -> Optional[pd.DataFrame]:
        """Load unified data created by step1_5 with comprehensive validation."

        Args:
            symbol: Trading symbol (e.g. = "ETHUSDT")
            exchange: Exchange name (e.g. = "BINANCE")
            timeframe: Timeframe (e.g. = "1m")
            data_dir: Data directory
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            columns: Specific columns to load

        Returns:
            DataFrame with unified data or None if failed
        """
        try:
            symbol = sanitize_string(symbol, max_length=20, allowed_chars='A-Z0-9')
            exchange = sanitize_string(exchange, max_length=20, allowed_chars='A-Z0-9')
            timeframe = sanitize_string(timeframe, max_length=10, allowed_chars='0-9mhdw')
            self.logger.info(f'📊 Loading unified data for {exchange}_{symbol}_{timeframe}')
            unified_path = self._get_unified_data_path(symbol, exchange, timeframe, data_dir)
            if not os.path.exists(unified_path):
                self.logger.error(f'❌ Unified data path does not exist: {unified_path}')
                return None
            try:
                pdm = ParquetDatasetManager(logger=self.logger)
                filters = None
                if start_date or end_date:
                    filters = []
                    if start_date:
                        start_ts = pd.Timestamp(start_date).timestamp() * 1000
                        filters.append(['timestamp', '>=', start_ts])
                    if end_date:
                        end_ts = pd.Timestamp(end_date).timestamp() * 1000
                        filters.append(['timestamp', '<=', end_ts])
                df = pdm.scan_dataset(base_dir=unified_path, filters=filters, columns=columns, batch_size=100000)
            except ImportError:
                self.logger.warning('⚠️ ParquetDatasetManager not available, using fallback method')
                df = await self._load_unified_data_fallback(unified_path, start_date, end_date, columns)
            if df is None or df.empty:
                self.logger.error('❌ No data loaded from unified dataset')
                return None
            validation_result = await self._validate_unified_data(df, symbol, exchange, timeframe)
            if not validation_result['valid']:
                self.logger.error(f"❌ Data validation failed: {validation_result['reason']}")
                return None
            self.logger.info(f'✅ Loaded {len(df)} rows of unified data')
            return df
        except Exception as e:
            self.logger.exception(f'❌ Failed to load unified data: {e}')
            return None

    @validates(mode='warn', arg_index=1)
    @validate_dataframe_schema(expected_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    async def _validate_unified_data(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Validate unified data against expected schema and constraints."

        Args:
            df: DataFrame to validate
            symbol: Expected symbol
            exchange: Expected exchange
            timeframe: Expected timeframe

        Returns:
            Validation result dictionary
        """
        try:
            validation_result = {'valid': True, 'reason': 'OK'}
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result['valid'] = False
                validation_result['reason'] = f'Missing required columns: {missing_columns}'
                return validation_result
            if not pd.api.types.is_numeric_dtype(df['timestamp']):
                validation_result['valid'] = False
                validation_result['reason'] = 'Timestamp column must be numeric'
                return validation_result
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_result['valid'] = False
                    validation_result['reason'] = f'Price column {col} must be numeric'
                    return validation_result
            for col in price_columns:
                if (df[col] < 0).any():
                    validation_result['valid'] = False
                    validation_result['reason'] = f'Negative prices found in {col}'
                    return validation_result
            if (df['volume'] < 0).any():
                validation_result['valid'] = False
                validation_result['reason'] = 'Negative volumes found'
                return validation_result
            if not df['timestamp'].is_monotonic_increasing:
                validation_result['valid'] = False
                validation_result['reason'] = 'Timestamps are not in ascending order'
                return validation_result
            if len(df) > self.max_rows:
                validation_result['valid'] = False
                validation_result['reason'] = f'Too many rows: {len(df)} > {self.max_rows}'
                return validation_result
            if 'symbol' in df.columns and df['symbol'].iloc[0] != symbol:
                validation_result['valid'] = False
                validation_result['reason'] = f"Symbol mismatch: expected {symbol}, got {df['symbol'].iloc[0]}"
                return validation_result
            if 'exchange' in df.columns and df['exchange'].iloc[0] != exchange:
                validation_result['valid'] = False
                validation_result['reason'] = f"Exchange mismatch: expected {exchange}, got {df['exchange'].iloc[0]}"
                return validation_result
            if 'timeframe' in df.columns and df['timeframe'].iloc[0] != timeframe:
                validation_result['valid'] = False
                validation_result['reason'] = f"Timeframe mismatch: expected {timeframe}, got {df['timeframe'].iloc[0]}"
                return validation_result
            self.logger.info('✅ Unified data validation passed')
            return validation_result
        except Exception as e:
            self.logger.exception(f'❌ Data validation error: {e}')
            return {'valid': False, 'reason': f'Validation error: {e}'}

    @secure_file_path(allowed_dirs=['data_cache', 'data'])
    async def _load_unified_data_fallback(self, unified_path: str, start_date: Optional[str]=None, end_date: Optional[str]=None, columns: Optional[list[str]]=None) -> Optional[pd.DataFrame]:
        """Fallback method to load unified data without ParquetDatasetManager."

        Args:
            unified_path: Path to unified data directory
            start_date: Start date filter
            end_date: End date filter
            columns: Specific columns to load

        Returns:
            DataFrame with unified data or None if failed
        """
        try:
            parquet_files = []
            for root, _dirs, files in os.walk(unified_path):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))
            if not parquet_files:
                self.logger.error(f'❌ No parquet files found in {unified_path}')
                return None
            dfs = []
            for file_path in sorted(parquet_files):
                try:
                    df = safe_read_parquet(file_path, columns=columns)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to load {file_path}: {e}')
                    continue
            if not dfs:
                self.logger.error('❌ No valid parquet files could be loaded')
                return None
            combined_df = pd.concat(dfs, ignore_index=True)
            if start_date or end_date:
                if 'timestamp' in combined_df.columns:
                    combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms', utc=True)
                    if start_date:
                        start_dt = pd.Timestamp(start_date)
                        combined_df = combined_df[combined_df['datetime'] >= start_dt]
                    if end_date:
                        end_dt = pd.Timestamp(end_date)
                        combined_df = combined_df[combined_df['datetime'] <= end_dt]
                    combined_df = combined_df.drop(columns=['datetime'])
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            return combined_df
        except Exception as e:
            self.logger.exception(f'❌ Fallback data loading failed: {e}')
            return None

    @sanitize_string(max_length=100, allowed_chars='A-Za-z0-9/_-')
    def _get_unified_data_path(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> str:
        """Get the path to unified data with input sanitization."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            Path to unified data directory
        """
        return os.path.join(data_dir, 'unified', exchange.lower(), symbol, timeframe)

    @handles_errors(exceptions=(Exception,), default_return=None, context='unified_data_loader.get_data_info')
    async def get_data_info(self, symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache') -> Optional[dict[str, Any]]:
        """Get information about available unified data."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            Dictionary with data information or None if failed
        """
        try:
            unified_path = self._get_unified_data_path(symbol, exchange, timeframe, data_dir)
            if not os.path.exists(unified_path):
                return None
            file_count = 0
            total_size = 0
            date_range = {'start': None, 'end': None}
            for root, _dirs, files in os.walk(unified_path):
                for file in files:
                    if file.endswith('.parquet'):
                        file_count += 1
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                        try:
                            path_parts = file_path.split('/')
                            for i, part in enumerate(path_parts):
                                if part.startswith('year='):
                                    year = int(part.split('=')[1])
                                    month = int(path_parts[i + 1].split('=')[1])
                                    day = int(path_parts[i + 2].split('=')[1])
                                    date = f'{year:04d}-{month:02d}-{day:02d}'
                                    if date_range['start'] is None or date < date_range['start']:
                                        date_range['start'] = date
                                    if date_range['end'] is None or date > date_range['end']:
                                        date_range['end'] = date
                                    break
                        except Exception:
                            pass
            return {'path': unified_path, 'file_count': file_count, 'total_size_bytes': total_size, 'total_size_mb': total_size / (1024 * 1024), 'date_range': date_range, 'exists': True}
        except Exception as e:
            self.logger.exception(f'❌ Failed to get data info: {e}')
            return None
_unified_data_loader = None

def get_unified_data_loader(config: Optional[dict[str, Any]]=None) -> UnifiedDataLoader:
    """Get or create a global unified data loader instance."

    Args:
        config: Configuration dictionary

    Returns:
        UnifiedDataLoader instance
    """
    global _unified_data_loader
    if _unified_data_loader is None:
        _unified_data_loader = UnifiedDataLoader(config)
    return _unified_data_loader

@handles_errors(exceptions=(Exception,), default_return=None, context='load_unified_data')
async def load_unified_data(symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache', start_date: Optional[str]=None, end_date: Optional[str]=None, columns: Optional[list[str]]=None) -> Optional[pd.DataFrame]:
    """Load unified data with global loader instance."

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        start_date: Start date filter
        end_date: End date filter
        columns: Specific columns to load

    Returns:
        DataFrame with unified data or None if failed
    """
    loader = get_unified_data_loader()
    return await loader.load_unified_data(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir, start_date=start_date, end_date=end_date, columns=columns)

@handles_errors(exceptions=(Exception,), default_return=None, context='get_unified_data_info')
async def get_unified_data_info(symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache') -> Optional[dict[str, Any]]:
    """Get information about unified data with global loader instance."

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory

    Returns:
        Dictionary with data information or None if failed
    """
    loader = get_unified_data_loader()
    return await loader.get_data_info(symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir)