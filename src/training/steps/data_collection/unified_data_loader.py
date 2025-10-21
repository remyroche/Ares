from src.utils.tprint import tprint

from ...core.decorators import handles_errors, traced
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
import numpy as np

"""Unified Data Loader for Step1_5 Data.

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

# Create a logger for UnifiedDataLoader
unified_data_loader_logger = logging.getLogger('UnifiedDataLoader')

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
        # Use the fallback logger
        self.logger = unified_data_loader_logger
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

            # Construct data path - data is in historical_data/storage/{exchange}/{symbol}/{timeframe}/
            data_path = Path(data_dir) / 'storage' / exchange.lower() / symbol.upper() / timeframe

            if not data_path.exists():
                self.logger.warning(f"Data path does not exist: {data_path}, trying consolidated files...")
                # Try to find consolidated files as fallback
                consolidated_path = Path(data_dir) / f"features_{exchange.lower()}_{symbol}_consolidated.parquet"
                if consolidated_path.exists():
                    self.logger.info(f"📁 Found consolidated file: {consolidated_path}")
                    parquet_files = [str(consolidated_path)]
                else:
                    self.logger.error(f"Data path does not exist and no consolidated file available: {consolidated_path}")
                    return None
            else:
                # Get list of parquet files
                parquet_files = list_parquet_files(data_path)
                if not parquet_files:
                    # Try to find consolidated files as fallback
                    self.logger.warning(f"No parquet files found in {data_path}, trying consolidated files...")
                    
                    # Try consolidated file at root level
                    consolidated_path = Path(data_dir) / f"features_{exchange.lower()}_{symbol}_consolidated.parquet"
                    if consolidated_path.exists():
                        self.logger.info(f"📁 Found consolidated file: {consolidated_path}")
                        parquet_files = [str(consolidated_path)]
                    else:
                        # Try 1m consolidated file
                        consolidated_1m_path = data_path.parent / f"{symbol.lower()}_1m" / f"features_{symbol.lower()}_1m_consolidated.parquet"
                        if consolidated_1m_path.exists():
                            self.logger.info(f"📁 Found 1m consolidated file: {consolidated_1m_path}")
                            parquet_files = [str(consolidated_1m_path)]
                        else:
                            self.logger.error(f"No parquet files found in {data_path} and no consolidated files available")
                            return None

            # Load data from multiple files if date filters are applied
            if start_date or end_date:
                # Smart loading: only load files that contain data in the required date range
                data = await self._load_data_with_date_filters(parquet_files, start_date, end_date, columns)
                if data is None or data.empty:
                    self.logger.error("No data loaded with date filters, trying fallback approach")
                    # Fallback: load the most recent file and apply date filters
                    latest_file = max(parquet_files, key=lambda x: Path(x).stat().st_mtime if isinstance(x, str) else x.stat().st_mtime)
                    data = await self._load_data_file(latest_file, columns)
                    if data is not None and not data.empty:
                        data = self._apply_date_filters(data, start_date, end_date)
                        if data.empty:
                            self.logger.error("No data found in the specified date range")
                            return None
                    else:
                        self.logger.error("Failed to load data from fallback file")
                        return None
                
                
            else:
                # Load the most recent file only
                latest_file = max(parquet_files, key=lambda x: Path(x).stat().st_mtime if isinstance(x, str) else x.stat().st_mtime)
                data = await self._load_data_file(latest_file, columns)
                if data is None:
                    return None


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

            if start_date or end_date:
                self.logger.info(f"Successfully loaded {len(data)} rows from {len(parquet_files)} files")
            else:
                self.logger.info(f"Successfully loaded {len(data)} rows from {latest_file}")
            return data

        except Exception as e:
            self.logger.exception(f"Error loading unified data: {e}")
            return None

    async def _load_data_with_date_filters(self, parquet_files: List[Path], 
                                         start_date: Optional[str], 
                                         end_date: Optional[str], 
                                         columns: Optional[List[str]]) -> Optional[pd.DataFrame]:
        """
        Efficiently load data with date filters by only loading relevant files.
        
        Args:
            parquet_files: List of parquet file paths
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            columns: Specific columns to load
            
        Returns:
            Filtered DataFrame or None if no data found
        """
        try:
            self.logger.info(f"🎯 Loading data for date range: {start_date} to {end_date}")
            
            # For efficiency, try to load from consolidated files first
            consolidated_files = [f for f in parquet_files if 'consolidated' in str(f)]
            if consolidated_files:
                self.logger.info(f"📁 Found {len(consolidated_files)} consolidated files, using most recent")
                # Use the most recent consolidated file - ensure it's a Path object
                latest_consolidated = max(consolidated_files, key=lambda x: Path(x).stat().st_mtime if isinstance(x, str) else x.stat().st_mtime)
                data = await self._load_data_file(latest_consolidated, columns)
                if data is not None and not data.empty:
                    # Apply date filters directly
                    data = self._apply_date_filters(data, start_date, end_date)
                    if not data.empty:
                        self.logger.info(f"✅ Loaded {len(data)} rows from consolidated file")
                        return data
            
            # If no consolidated files or they don't contain the required data,
            # load from partitioned files but be smart about it
            self.logger.info("📁 No consolidated files or insufficient data, loading from partitioned files")
            
            # Load files in reverse chronological order (most recent first)
            # and stop when we have enough data
            all_data = []
            total_rows = 0
            max_rows_needed = 50000  # Reasonable limit for 20-180 days of 15m data
            
            for file_path in sorted(parquet_files, key=lambda x: Path(x).stat().st_mtime if isinstance(x, str) else x.stat().st_mtime, reverse=True):
                if total_rows >= max_rows_needed:
                    break
                    
                file_data = await self._load_data_file(file_path, columns)
                if file_data is not None and not file_data.empty:
                    # Apply date filters to this file's data
                    filtered_data = self._apply_date_filters(file_data, start_date, end_date)
                    if not filtered_data.empty:
                        all_data.append(filtered_data)
                        total_rows += len(filtered_data)
                        file_name = Path(file_path).name if isinstance(file_path, str) else file_path.name
                        self.logger.info(f"📊 Loaded {len(filtered_data)} rows from {file_name} (total: {total_rows})")
            
            if not all_data:
                self.logger.error("No data found in any files for the specified date range")
                return None
            
            # Combine all filtered data
            if len(all_data) == 1:
                data = all_data[0]
            else:
                data = pd.concat(all_data, ignore_index=True)
                # Remove duplicates based on timestamp
                if 'timestamp' in data.columns:
                    data = data.drop_duplicates(subset=['timestamp'], keep='first')
            
            # Apply final data quality fixes
            data = self._fix_non_finite_values(data)
            
            self.logger.info(f"✅ Efficiently loaded {len(data)} rows for date range {start_date} to {end_date}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in efficient data loading: {e}")
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

            # Add data preview for troubleshooting data state before processing
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(
                data, 
                name=f"before_processing_{Path(file_path).name}",
                max_rows=3,
                level="DEBUG",
                include_metadata=True
            )

            # DEBUG: Check data quality immediately after parquet read
            import numpy as np
            non_finite = (~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()
            if non_finite > 0:
                print(f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name}: {non_finite} non-finite values IMMEDIATELY after parquet read")
                for col in data.select_dtypes(include=[np.number]).columns:
                    col_non_finite = (~np.isfinite(data[col])).sum()
                    if col_non_finite > 0:
                        print(f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name} {col}: {col_non_finite} non-finite values")
                        # Find the exact rows with non-finite values
                        non_finite_mask = ~np.isfinite(data[col])
                        non_finite_rows = data[non_finite_mask].index.tolist()
                        print(f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name} {col}: Non-finite values at rows: {non_finite_rows[:10]}")
                        # Show the actual values
                        non_finite_values = data.loc[non_finite_mask, col].tolist()
                        print(f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name} {col}: Non-finite values: {non_finite_values[:10]}")
                        # Show context around the first non-finite value
                        if len(non_finite_rows) > 0:
                            first_bad_row = non_finite_rows[0]
                            bad_positions = np.flatnonzero(non_finite_mask)
                            if bad_positions.size > 0:
                                first_bad_pos = int(bad_positions[0])
                                start_idx = max(0, first_bad_pos - 2)
                                end_idx = min(len(data), first_bad_pos + 3)
                                context_cols = [c for c in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] if c in data.columns]
                                context_window = data.iloc[start_idx:end_idx]
                                if context_cols:
                                    context_window = context_window[context_cols]
                                print(
                                    f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name} {col}: "
                                    f"Context around index {first_bad_row} (position {first_bad_pos}):"
                                )
                                print(f"🔍 [DEBUG] Context:\n{context_window}")

            # Apply forward/backward fill to handle non-finite values (up to 6 consecutive rows)
            data = self._fix_non_finite_values(data)

            # Convert timestamp index to column if it exists
            if data.index.name == 'timestamp' or (hasattr(data.index, 'name') and data.index.name == 'timestamp'):
                data = data.reset_index()
                self.logger.info("Converted timestamp index to column")
                # DEBUG: Check if timestamp conversion introduced non-finite values
                non_finite_after = (~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()
                if non_finite_after > non_finite:
                    print(f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name}: {non_finite_after - non_finite} NEW non-finite values introduced during timestamp conversion")
            elif 'timestamp' not in data.columns and hasattr(data.index, 'dtype') and 'datetime' in str(data.index.dtype):
                # If the index is datetime but not named 'timestamp', rename it
                data = data.reset_index()
                data = data.rename(columns={'index': 'timestamp'})
                self.logger.info("Converted datetime index to timestamp column")
                # DEBUG: Check if timestamp conversion introduced non-finite values
                non_finite_after = (~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()
                if non_finite_after > non_finite:
                    print(f"🔍 [DEBUG] UnifiedDataLoader._load_data_file - {Path(file_path).name}: {non_finite_after - non_finite} NEW non-finite values introduced during timestamp conversion")

            data = self._inject_partition_metadata(data, Path(file_path))

        if len(data) > self.max_rows:
            self.logger.warning(f"Data has {len(data)} rows, exceeding limit of {self.max_rows}")
            data = data.head(self.max_rows)
        
        # Add data preview for troubleshooting data state after loading
        from src.utils.tprint import tprint_data_preview
        tprint_data_preview(
            data, 
            name=f"loaded_from_{Path(file_path).name}",
            max_rows=3,
            level="DEBUG",
            include_metadata=True
        )

            return data

        except Exception as e:
            self.logger.exception(f"Error loading file {file_path}: {e}")
            return None

    def _fix_non_finite_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix non-finite values using forward/backward fill with 6-row consecutive limit.

        Args:
            data: DataFrame with potential non-finite values

        Returns:
            DataFrame with non-finite values handled
        """
        import numpy as np

        # Check for non-finite values before fixing
        non_finite_before = (~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()
        if non_finite_before == 0:
            return data

        print(f"🔧 [DEBUG] Fixing {non_finite_before} non-finite values using forward/backward fill (limit 6 consecutive)")

        # Process each numeric column
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        fixed_columns = []

        for col in numeric_columns:
            original_non_finite = (~np.isfinite(data[col])).sum()

            if original_non_finite == 0:
                fixed_columns.append(col)
                continue

            print(f"🔧 [DEBUG] Column '{col}': {original_non_finite} non-finite values to fix")

            # Create a copy of the column to work with
            col_data = data[col].copy()

            # Find sequences of consecutive non-finite values
            non_finite_mask = ~np.isfinite(col_data)
            consecutive_groups = []

            if non_finite_mask.any():
                # Group consecutive non-finite values
                diff = np.diff(np.concatenate(([False], non_finite_mask.values, [False])))
                start_indices = np.where(diff > 0)[0]
                end_indices = np.where(diff < 0)[0]

                for start, end in zip(start_indices, end_indices):
                    group_length = end - start
                    consecutive_groups.append((start, end, group_length))

                    # Handle groups larger than 6 consecutive missing values
                    if group_length > 6:
                        print(f"⚠️ [WARNING] Column '{col}' has {group_length} consecutive non-finite values at indices {start}-{end-1} (exceeds 6-row limit)")

            # Apply forward fill first, then backward fill for remaining gaps
            col_data = col_data.fillna(method='ffill', limit=6)
            col_data = col_data.fillna(method='bfill', limit=6)

            # Check how many non-finite values remain after fixing
            remaining_non_finite = (~np.isfinite(col_data)).sum()
            if remaining_non_finite > 0:
                print(f"⚠️ [WARNING] Column '{col}': {remaining_non_finite} non-finite values remain after forward/backward fill")

                # For remaining non-finite values, use the column mean as a fallback
                if col_data.dtype in [np.float64, np.float32]:
                    col_mean = col_data.mean()
                    if np.isfinite(col_mean):
                        col_data = col_data.fillna(col_mean)
                        print(f"🔧 [DEBUG] Column '{col}': Filled remaining {remaining_non_finite} values with column mean {col_mean}")
                    else:
                        print(f"⚠️ [ERROR] Column '{col}': Cannot use mean fallback - column mean is also non-finite")
                else:
                    print(f"⚠️ [ERROR] Column '{col}': Cannot apply fallback for non-float column")

            # Update the original data
            data[col] = col_data
            fixed_columns.append(col)

            # Verify the fix
            final_non_finite = (~np.isfinite(data[col])).sum()
            if final_non_finite == 0:
                print(f"✅ [SUCCESS] Column '{col}': Successfully fixed all {original_non_finite} non-finite values")
            elif final_non_finite < original_non_finite:
                print(f"✅ [PARTIAL] Column '{col}': Reduced non-finite values from {original_non_finite} to {final_non_finite}")
            else:
                print(f"❌ [FAILED] Column '{col}': Non-finite values unchanged or increased ({final_non_finite})")

        # Final check of all numeric columns
        non_finite_after = (~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()
        if non_finite_after == 0:
            print(f"✅ [SUCCESS] All non-finite values fixed! Reduced from {non_finite_before} to {non_finite_after}")
        else:
            print(f"⚠️ [PARTIAL] {non_finite_after} non-finite values remain after fixing (was {non_finite_before})")

        return data

    def _extract_metadata_from_path(self, file_path: Path) -> Dict[str, Any]:
        """Extract partition metadata (exchange, symbol, timeframe, year, month, day) from file path."""
        metadata: Dict[str, Any] = {}
        parts = file_path.parts

        try:
            root_idx = parts.index('historical_data')
        except ValueError:
            root_idx = None

        if root_idx is not None:
            if len(parts) > root_idx + 1:
                metadata['exchange'] = parts[root_idx + 1].upper()
            if len(parts) > root_idx + 2:
                metadata['symbol'] = parts[root_idx + 2].upper()
            if len(parts) > root_idx + 4:
                dataset_part = parts[root_idx + 4]
                timeframe_value = dataset_part.split('_', 1)[1] if '_' in dataset_part else dataset_part
                metadata['timeframe'] = timeframe_value

        for part in parts:
            if part.startswith('year='):
                value = part.split('=', 1)[1]
                try:
                    metadata['year'] = int(value)
                except ValueError:
                    self.logger.warning(f"Invalid year partition value '{value}' in {file_path}")
            elif part.startswith('month='):
                value = part.split('=', 1)[1]
                try:
                    metadata['month'] = int(value)
                except ValueError:
                    self.logger.warning(f"Invalid month partition value '{value}' in {file_path}")
            elif part.startswith('day='):
                value = part.split('=', 1)[1]
                try:
                    metadata['day'] = int(value)
                except ValueError:
                    self.logger.warning(f"Invalid day partition value '{value}' in {file_path}")

        return metadata

    def _inject_partition_metadata(self, data: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """Ensure required metadata columns are present using partition information and timestamps."""
        metadata = self._extract_metadata_from_path(file_path)

        for column, value in metadata.items():
            if value is None:
                continue

            value_to_assign = str(value) if column in {'exchange', 'symbol', 'timeframe'} else value
            if column in data.columns:
                if data[column].isna().all():
                    data[column] = value_to_assign
            else:
                data[column] = value_to_assign

        if 'day' not in data.columns or data['day'].isna().all():
            if 'timestamp' in data.columns:
                timestamp_series = data['timestamp']
                if timestamp_series.dtype.kind in {'i', 'u'}:
                    ts = pd.to_datetime(timestamp_series, unit='ms', errors='coerce', utc=True)
                else:
                    ts = pd.to_datetime(timestamp_series, errors='coerce', utc=True)

                if ts.notna().any():
                    day_values = ts.dt.day
                    if not day_values.isna().any():
                        data['day'] = day_values.astype('int8')
                    else:
                        data['day'] = day_values

        return data

    @log_all_calls
    def _apply_date_filters(self, data: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        """Apply date filters to the data."""
        try:
            if 'timestamp' not in data.columns:
                self.logger.warning("No timestamp column found, skipping date filters")
                return data

            # Convert timestamp to datetime with robust error handling
            data['timestamp'] = self._robust_timestamp_conversion(data['timestamp'])

            # Apply filters - ensure timezone consistency
            if start_date:
                # Handle numpy array inputs
                if isinstance(start_date, np.ndarray):
                    if start_date.size == 1:
                        start_date = start_date.item()
                    else:
                        self.logger.warning(f"Invalid start_date format: numpy array with {start_date.size} elements")
                        return data

                start_dt = pd.to_datetime(start_date, utc=True)
                data = data[data['timestamp'] >= start_dt]

            if end_date:
                # Handle numpy array inputs
                if isinstance(end_date, np.ndarray):
                    if end_date.size == 1:
                        end_date = end_date.item()
                    else:
                        self.logger.warning(f"Invalid end_date format: numpy array with {end_date.size} elements")
                        return data

                end_dt = pd.to_datetime(end_date, utc=True)
                data = data[data['timestamp'] <= end_dt]

            # Add data preview for troubleshooting filtered data correctness
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(
                data, 
                name=f"date_filtered_{start_date}_{end_date}",
                max_rows=3,
                level="DEBUG",
                include_metadata=True
            )

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

    def _robust_timestamp_conversion(self, timestamp_series: pd.Series) -> pd.Series:
        """
        Robust timestamp conversion that handles various corruption scenarios.
        
        Args:
            timestamp_series: Series containing timestamp data
            
        Returns:
            Series with converted timestamps
        """
        try:
            # First, try standard conversion
            if timestamp_series.dtype.kind in ['i', 'f']:
                # Try as Unix timestamp (seconds)
                try:
                    converted = pd.to_datetime(timestamp_series, unit='s', utc=True)
                    if self._validate_timestamps(converted):
                        return converted
                except (pd.errors.OutOfBoundsDatetime, ValueError):
                    pass
                
                # Try as Unix timestamp (milliseconds)
                try:
                    converted = pd.to_datetime(timestamp_series, unit='ms', utc=True)
                    if self._validate_timestamps(converted):
                        return converted
                except (pd.errors.OutOfBoundsDatetime, ValueError):
                    pass
                
                # Try as Unix timestamp (microseconds)
                try:
                    converted = pd.to_datetime(timestamp_series, unit='us', utc=True)
                    if self._validate_timestamps(converted):
                        return converted
                except (pd.errors.OutOfBoundsDatetime, ValueError):
                    pass
                
                # Try as Unix timestamp (nanoseconds)
                try:
                    converted = pd.to_datetime(timestamp_series, unit='ns', utc=True)
                    if self._validate_timestamps(converted):
                        return converted
                except (pd.errors.OutOfBoundsDatetime, ValueError):
                    pass
            else:
                # Try direct conversion
                try:
                    converted = pd.to_datetime(timestamp_series, utc=True)
                    if self._validate_timestamps(converted):
                        return converted
                except (pd.errors.OutOfBoundsDatetime, ValueError):
                    pass
            
            # If all standard methods fail, try corruption correction
            self.logger.warning("Standard timestamp conversion failed, attempting corruption correction")
            return self._correct_timestamp_corruption(timestamp_series)
            
        except Exception as e:
            self.logger.error(f"Timestamp conversion failed: {e}")
            # Return empty series as fallback
            return pd.Series(dtype='datetime64[ns, UTC]')

    def _validate_timestamps(self, timestamps: pd.Series) -> bool:
        """
        Validate that timestamps are within reasonable bounds.
        
        Args:
            timestamps: Series of datetime objects
            
        Returns:
            True if timestamps are valid, False otherwise
        """
        if timestamps.empty:
            return False
        
        # Check for reasonable date range (2000-2030)
        min_date = pd.Timestamp('2000-01-01', tz='UTC')
        max_date = pd.Timestamp('2030-12-31', tz='UTC')
        
        valid_timestamps = timestamps.dropna()
        if len(valid_timestamps) == 0:
            return False
        
        # Check if all timestamps are within reasonable bounds
        within_bounds = ((valid_timestamps >= min_date) & (valid_timestamps <= max_date)).all()
        
        # For small samples (less than 10 timestamps), be more lenient
        if len(valid_timestamps) < 10:
            # Just check if timestamps are reasonable (not too far in past/future)
            # Since we're in 2025, allow current year data
            reasonable_range = ((valid_timestamps >= pd.Timestamp('1990-01-01', tz='UTC')) & 
                              (valid_timestamps <= pd.Timestamp('2026-12-31', tz='UTC'))).all()
            return reasonable_range
        
        # Check for reasonable distribution (not all the same date)
        unique_dates = valid_timestamps.dt.date.nunique()
        has_variation = unique_dates > 1
        
        return within_bounds and has_variation

    def _correct_timestamp_corruption(self, timestamp_series: pd.Series) -> pd.Series:
        """
        Attempt to correct various timestamp corruption patterns.
        
        Args:
            timestamp_series: Series containing corrupted timestamp data
            
        Returns:
            Series with corrected timestamps
        """
        try:
            # Pattern 1: Check if timestamps are in milliseconds (common issue)
            if timestamp_series.dtype.kind in ['i', 'f']:
                # Check if values are in milliseconds range (13 digits)
                sample_values = timestamp_series.dropna().head(10)
                if len(sample_values) > 0:
                    # Check if values look like milliseconds (13+ digits)
                    max_val = sample_values.max()
                    min_val = sample_values.min()
                    
                    # If values are in the range of milliseconds (13+ digits)
                    if max_val > 1e12:  # Values are in milliseconds
                        self.logger.info("Detected millisecond timestamps, converting to seconds")
                        try:
                            # Convert milliseconds to seconds
                            corrected_values = timestamp_series / 1000
                            converted = pd.to_datetime(corrected_values, unit='s', utc=True)
                            self.logger.debug(f"Converted timestamps sample: {converted.head()}")
                            self.logger.debug(f"Converted timestamps range: {converted.min()} to {converted.max()}")
                            validation_result = self._validate_timestamps(converted)
                            self.logger.debug(f"Validation result: {validation_result}")
                            if validation_result:
                                self.logger.info("Successfully corrected millisecond timestamps")
                                return converted
                            else:
                                self.logger.warning("Millisecond conversion produced invalid timestamps")
                        except Exception as e:
                            self.logger.debug(f"Millisecond conversion failed: {e}")
                    
                    # If values are in the range of microseconds (16+ digits)
                    elif max_val > 1e15:  # Values are in microseconds
                        self.logger.info("Detected microsecond timestamps, converting to seconds")
                        try:
                            # Convert microseconds to seconds
                            corrected_values = timestamp_series / 1000000
                            converted = pd.to_datetime(corrected_values, unit='s', utc=True)
                            if self._validate_timestamps(converted):
                                self.logger.info("Successfully corrected microsecond timestamps")
                                return converted
                        except Exception as e:
                            self.logger.debug(f"Microsecond conversion failed: {e}")
                    
                    # If values are in the range of nanoseconds (19+ digits)
                    elif max_val > 1e18:  # Values are in nanoseconds
                        self.logger.info("Detected nanosecond timestamps, converting to seconds")
                        try:
                            # Convert nanoseconds to seconds
                            corrected_values = timestamp_series / 1000000000
                            converted = pd.to_datetime(corrected_values, unit='s', utc=True)
                            if self._validate_timestamps(converted):
                                self.logger.info("Successfully corrected nanosecond timestamps")
                                return converted
                        except Exception as e:
                            self.logger.debug(f"Nanosecond conversion failed: {e}")
            
            # Pattern 2: Try to extract reasonable timestamps from corrupted data
            # Look for patterns that might be Unix timestamps with corruption
            if timestamp_series.dtype.kind in ['i', 'f']:
                # Try dividing by various factors to get reasonable timestamps
                for factor in [1000, 1000000, 1000000000]:
                    try:
                        corrected_values = timestamp_series / factor
                        converted = pd.to_datetime(corrected_values, unit='s', utc=True)
                        if self._validate_timestamps(converted):
                            self.logger.info(f"Successfully corrected timestamp by dividing by {factor}")
                            return converted
                    except Exception:
                        continue
            
            # Pattern 3: Try to reconstruct from partial data
            # If we have some valid timestamps, use them as reference
            try:
                # Convert with errors='coerce' to get NaT for invalid ones
                converted = pd.to_datetime(timestamp_series, utc=True, errors='coerce')
                valid_mask = converted.notna()
                
                if valid_mask.any():
                    # Use valid timestamps to estimate the range
                    valid_timestamps = converted[valid_mask]
                    min_ts = valid_timestamps.min()
                    max_ts = valid_timestamps.max()
                    
                    # Fill invalid timestamps with interpolated values
                    if len(valid_timestamps) > 1:
                        # Interpolate missing timestamps
                        converted = converted.interpolate(method='time')
                        if self._validate_timestamps(converted):
                            self.logger.info("Successfully interpolated missing timestamps")
                            return converted
                
            except Exception:
                pass
            
            # If all correction attempts fail, return empty series
            self.logger.warning("All timestamp correction attempts failed")
            return pd.Series(dtype='datetime64[ns, UTC]')
            
        except Exception as e:
            self.logger.error(f"Timestamp corruption correction failed: {e}")
            return pd.Series(dtype='datetime64[ns, UTC]')

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
