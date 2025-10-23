"""
Unified Data Cleaning Module

This module consolidates missing value handling and outlier detection functionality
from multiple previous modules into a single, comprehensive framework.

Consolidated from:
- enhanced_missing_value_handler.py
- enhanced_outlier_handler.py
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import logging
from src.utils.tprint import tprint_data_format

class MissingValueStrategy(Enum):
    """Strategies for handling missing values."""
    INTERPOLATE = "interpolate"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    DROP = "drop"
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"

class OutlierStrategy(Enum):
    """Strategies for handling outliers."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    REMOVE = "remove"
    CAP = "cap"
    TRANSFORM = "transform"
    CLIP = "clip"

# Import UnifiedGapFiller for critical gap handling
try:
    from src.training.steps.data_collection.unified_gap_filler import UnifiedGapFiller
    UNIFIED_GAP_FILLER_AVAILABLE = True
except ImportError:
    UNIFIED_GAP_FILLER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    
    # Missing value handling
    missing_value_strategy: str = "interpolate"  # interpolate, forward_fill, backward_fill, drop
    max_missing_ratio: float = 0.1
    interpolate_method: str = "linear"
    
    # Outlier detection
    outlier_detection_enabled: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_strategy: str = "clip"  # clip, cap, remove, transform
    outlier_threshold: float = 3.0
    
    # Data quality thresholds
    min_data_quality_score: float = 0.8
    max_gap_seconds: int = 3600
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 1024
    
    # Validation settings
    strict_validation: bool = True
    skip_validation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'missing_value_strategy': self.missing_value_strategy,
            'max_missing_ratio': self.max_missing_ratio,
            'interpolate_method': self.interpolate_method,
            'outlier_detection_enabled': self.outlier_detection_enabled,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'min_data_quality_score': self.min_data_quality_score,
            'max_gap_seconds': self.max_gap_seconds,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'memory_limit_mb': self.memory_limit_mb,
            'strict_validation': self.strict_validation,
            'skip_validation': self.skip_validation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CleaningConfig':
        """Create config from dictionary."""
        return cls(**data)

class GapType(Enum):
    """Types of data gaps."""
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    CRITICAL = 'critical'

class OutlierSeverity(Enum):
    """Outlier severity levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class GapInfo:
    """Information about a data gap."""

    def __init__(self, start_time: int, end_time: int, gap_size: int, gap_type: GapType) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.gap_size = gap_size
        self.gap_type = gap_type
        self.filled = False
        self.fill_method = None
        self.downloaded_data = None

    def __str__(self) -> str:
        return f'Gap({self.start_time} -> {self.end_time}, size={self.gap_size}s, type={self.gap_type.value})'

class OutlierInfo:
    """Information about detected outliers."""

    def __init__(self, column: str, indices: List[int], values: List[Any], method: str, severity: OutlierSeverity, threshold: float) -> None:
        self.column = column
        self.indices = indices
        self.values = values
        self.method = method
        self.severity = severity
        self.threshold = threshold
        self.timestamp = datetime.now()
        self.context = {}

    def __str__(self) -> str:
        return f'OutlierInfo(column={self.column}, count={len(self.indices)}, severity={self.severity.value}, method={self.method})'

    def __repr__(self) -> str:
        return self.__str__()

class DataSchema:
    """Defines expected data schema for file operations."""

    def __init__(self, name: str, required_columns: List[str], optional_columns: List[str] = None, data_types: Dict[str, str] = None, constraints: Dict[str, Dict[str, Any]] = None) -> None:
        self.name = name
        self.required_columns = set(required_columns)
        self.optional_columns = set(optional_columns or [])
        self.data_types = data_types or {}
        self.constraints = constraints or {}
        self.all_columns = self.required_columns.union(self.optional_columns)

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe against schema."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'constraint_violations': []
        }

        df_columns = set(df.columns)
        missing_required = self.required_columns - df_columns
        if missing_required:
            results['valid'] = False
            results['missing_columns'] = list(missing_required)
            results['errors'].append(f'Missing required columns: {missing_required}')

        extra_columns = df_columns - self.all_columns
        if extra_columns:
            results['warnings'].append(f'Extra columns found: {extra_columns}')
            results['extra_columns'] = list(extra_columns)

        for column, expected_type in self.data_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type != expected_type:
                    results['type_mismatches'].append({'column': column, 'expected': expected_type, 'actual': actual_type})
                    results['warnings'].append(f'Type mismatch in {column}: expected {expected_type}, got {actual_type}')

        for column, constraint in self.constraints.items():
            if column in df.columns:
                if 'not_null' in constraint and constraint['not_null']:
                    if df[column].isnull().any():
                        results['constraint_violations'].append(f'Column {column} contains null values')
                        results['warnings'].append(f'Null values found in {column}')
                if 'unique' in constraint and constraint['unique']:
                    if df[column].duplicated().any():
                        results['constraint_violations'].append(f'Column {column} contains duplicate values')
                        results['warnings'].append(f'Duplicate values found in {column}')
                if 'min' in constraint:
                    min_val = constraint['min']
                    if (df[column] < min_val).any():
                        results['constraint_violations'].append(f'Column {column} contains values below minimum {min_val}')
                        results['warnings'].append(f'Values below minimum {min_val} found in {column}')
                if 'max' in constraint:
                    max_val = constraint['max']
                    if (df[column] > max_val).any():
                        results['constraint_violations'].append(f'Column {column} contains values above maximum {max_val}')
                        results['warnings'].append(f'Values above maximum {max_val} found in {column}')

        return results

class DataCleaner:
    """Unified data cleaning with missing value handling and outlier detection."""

    def __init__(self, config: CleaningConfig = None, max_forward_fill_gap: int = 5, download_threshold: int = 5, raise_errors: bool = True, log_details: bool = True, data_type: str = 'klines') -> None:
        """Initialize data cleaner with data-type specific gap thresholds.

        Args:
            config: CleaningConfig for data cleaning operations
            max_forward_fill_gap: Maximum gap size for forward fill (seconds)
            download_threshold: Threshold for triggering data download (seconds)
            raise_errors: Whether to raise errors on critical issues
            log_details: Whether to log detailed information
            data_type: Type of data ('klines', 'aggtrades', 'futures') for gap thresholds
        """
        start_time = time.time()
        self.logger = logging.getLogger('DataCleaner')
        
        # Initialize config
        self.config = config or CleaningConfig()
        
        self.max_forward_fill_gap = max_forward_fill_gap
        self.download_threshold = download_threshold
        self.raise_errors = raise_errors
        self.log_details = log_details
        self.data_type = data_type

        # Data-type specific gap thresholds (in seconds)
        # Large gaps trigger re-downloading of missing data
        # Thresholds are adjusted based on timeframe for relevance
        self.data_type_gap_thresholds = {
            'aggtrades': {
                GapType.SMALL: 1,      # 1 second
                GapType.MEDIUM: 5,     # 5 seconds
                GapType.LARGE: 10,     # 10 seconds - triggers re-download
                GapType.CRITICAL: 30   # 30 seconds
            },
            'klines': {
                GapType.SMALL: 65,     # 65 seconds - triggers download
                GapType.MEDIUM: 300,   # 5 minutes - triggers download
                GapType.LARGE: 1800,   # 30 minutes - triggers download
                GapType.CRITICAL: 3600 # 1 hour - triggers UnifiedGapFiller
            },
            'klines_1m': {  # Special thresholds for 1m timeframe data
                GapType.SMALL: 60,     # 1 minute (60s) - minimum meaningful gap for 1m data
                GapType.MEDIUM: 300,   # 5 minutes - triggers download
                GapType.LARGE: 1800,   # 30 minutes - triggers download
                GapType.CRITICAL: 3600 # 1 hour - triggers UnifiedGapFiller
            },
            'klines_5m': {  # Special thresholds for 5m timeframe data
                GapType.SMALL: 300,    # 5 minutes (300s) - minimum meaningful gap for 5m data
                GapType.MEDIUM: 900,   # 15 minutes - triggers download
                GapType.LARGE: 3600,   # 1 hour - triggers download
                GapType.CRITICAL: 7200 # 2 hours - triggers UnifiedGapFiller
            },
            'klines_15m': {  # Special thresholds for 15m timeframe data
                GapType.SMALL: 900,    # 15 minutes (900s) - minimum meaningful gap for 15m data
                GapType.MEDIUM: 1800,  # 30 minutes - triggers download
                GapType.LARGE: 7200,   # 2 hours - triggers download
                GapType.CRITICAL: 14400 # 4 hours - triggers UnifiedGapFiller
            },
            'klines_30m': {  # Special thresholds for 30m timeframe data
                GapType.SMALL: 1800,   # 30 minutes (1800s) - minimum meaningful gap for 30m data
                GapType.MEDIUM: 3600,  # 1 hour - triggers download
                GapType.LARGE: 14400,  # 4 hours - triggers download
                GapType.CRITICAL: 28800 # 8 hours - triggers UnifiedGapFiller
            },
            'klines_1h': {  # Special thresholds for 1h timeframe data
                GapType.SMALL: 3600,   # 1 hour (3600s) - minimum meaningful gap for 1h data
                GapType.MEDIUM: 7200,  # 2 hours - triggers download
                GapType.LARGE: 14400,  # 4 hours - triggers download
                GapType.CRITICAL: 28800 # 8 hours - triggers UnifiedGapFiller
            },
            'klines_4h': {  # Special thresholds for 4h timeframe data
                GapType.SMALL: 14400,  # 4 hours - minimum meaningful gap
                GapType.MEDIUM: 28800, # 8 hours - triggers download
                GapType.LARGE: 57600,  # 16 hours - triggers download
                GapType.CRITICAL: 86400 # 24 hours - triggers UnifiedGapFiller
            },
            'futures': {
                GapType.SMALL: 3600,   # 1 hour
                GapType.MEDIUM: 14400, # 4 hours
                GapType.LARGE: 32400,  # 9 hours - triggers re-download
                GapType.CRITICAL: 86400 # 24 hours
            },
            'unified': {
                GapType.SMALL: 65,     # 65 seconds (same as klines)
                GapType.MEDIUM: 300,   # 5 minutes (same as klines)
                GapType.LARGE: 900,    # 15 minutes - consistent monotonicity
                GapType.CRITICAL: 1800 # 30 minutes (same as klines)
            }
        }

        # Use data-type specific thresholds or fallback to generic
        if data_type in self.data_type_gap_thresholds:
            self.gap_thresholds = self.data_type_gap_thresholds[data_type]
            self.logger.info(f"Using {data_type}-specific gap thresholds: {self.gap_thresholds}")
        else:
            # Fallback to generic thresholds
            self.gap_thresholds = {
                GapType.SMALL: max_forward_fill_gap,
                GapType.MEDIUM: 60,
                GapType.LARGE: 300,
                GapType.CRITICAL: float('inf')
            }
            self.logger.warning(f"Unknown data type '{data_type}', using generic gap thresholds")

        self.fill_strategies = {
            GapType.SMALL: 'download',           # All gaps now trigger download
            GapType.MEDIUM: 'download',
            GapType.LARGE: 'download',
            GapType.CRITICAL: 'unified_gap_filler'  # Critical gaps use UnifiedGapFiller
        }

        self.outlier_history = []
        self.standard_schemas = {
            'klines': DataSchema(
                name='klines',
                required_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                data_types={
                    'timestamp': 'int64',
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'float64'
                },
                constraints={
                    'open': {'min': 0, 'not_null': True},
                    'high': {'min': 0, 'not_null': True},
                    'low': {'min': 0, 'not_null': True},
                    'close': {'min': 0, 'not_null': True},
                    'volume': {'min': 0, 'not_null': True}
                }
            ),
            'features': DataSchema(
                name='features',
                required_columns=['timestamp'],
                optional_columns=[],
                data_types={'timestamp': 'int64'},
                constraints={'timestamp': {'not_null': True}}
            ),
            'labels': DataSchema(
                name='labels',
                required_columns=['timestamp', 'label'],
                data_types={'timestamp': 'int64', 'label': 'object'},
                constraints={'timestamp': {'not_null': True}, 'label': {'not_null': True}}
            )
        }

        self.detection_methods = {
            'zscore': self._detect_zscore_outliers,
            'iqr': self._detect_iqr_outliers,
            'isolation_forest': self._detect_isolation_forest_outliers,
            'local_outlier_factor': self._detect_lof_outliers,
            'mahalanobis': self._detect_mahalanobis_outliers
        }

        # Reduce verbosity of initialization logging
        self.logger.debug(f'🧹 Data Cleaner initialized with {len(self.detection_methods)} outlier detection methods')

        # Add timing information (Numba-safe implementation)
        duration = time.time() - start_time
        try:
            from src.utils.tprint import tprint_performance
            tprint_performance(f"DataCleaner({data_type}) initialization", duration)
        except ImportError:
            # Fallback to basic logging (Numba-safe)
            self.logger.info(f"⏱️ DataCleaner({data_type}) initialized in {duration:.3f}s")

    async def handle_missing_values_intelligently(
        self,
        data: pd.DataFrame,
        timestamp_column: str = 'timestamp',
        symbol: str = None,
        exchange: str = None,
        timeframe: str = '1m'
    ) -> pd.DataFrame:
        """Handle missing values intelligently based on gap size."""
        if timestamp_column not in data.columns:
            self.logger.error(f"Timestamp column '{timestamp_column}' not found")
            return data

        # Add format debugging for troubleshooting
        tprint_data_format(data, f"missing_values_input_{symbol}_{exchange}_{timeframe}", level="DEBUG")

        data = data.sort_values(timestamp_column).reset_index(drop=True)
        gaps = self._analyze_gaps(data, timestamp_column)

        if not gaps:
            self.logger.info('No gaps detected in data')
            return data

        self._log_gap_analysis(gaps)
        filled_data = data.copy()

        for gap in gaps:
            # PRIORITY 1: Try to re-download data for ALL gaps (small, medium, large)
            if symbol and exchange and timeframe:
                download_success = False
                try:
                    filled_data = await self._handle_large_gap_with_download(filled_data, gap, timestamp_column, symbol, exchange, timeframe)
                    # Check if gap was actually filled by looking for the gap again
                    if self._is_gap_filled(filled_data, gap, timestamp_column):
                        self.logger.info(f'✅ Gap filled via re-download: {gap}')
                        download_success = True
                        gap.filled = True
                        gap.fill_method = 'download'
                    else:
                        self.logger.warning(f'⚠️ Re-download attempted but gap still exists: {gap}')
                except Exception as e:
                    self.logger.warning(f'⚠️ Re-download failed for gap {gap}: {e}')

                if not download_success:
                    # PRIORITY 2: Fallback strategies for when download fails
                    if gap.gap_type in [GapType.SMALL, GapType.MEDIUM, GapType.LARGE]:
                        # Use forward fill for smaller gaps when download fails
                        self.logger.info(f'🔄 Using forward fill fallback for {gap.gap_type.value} gap: {gap}')
                        filled_data = self._handle_small_gap(filled_data, gap, timestamp_column)
                        gap.fill_method = 'forward_fill_fallback'
                    else:
                        # For critical gaps, try UnifiedGapFiller
                        self.logger.info(f'🔄 Attempting UnifiedGapFiller for critical gap: {gap}')
                        filled_data = self._handle_critical_gap(filled_data, gap, timestamp_column, symbol, exchange, timeframe)
            else:
                # No symbol/exchange/timeframe available
                if gap.gap_type in [GapType.SMALL, GapType.MEDIUM, GapType.LARGE]:
                    # Forward fill for small/medium/large gaps when no download possible
                    self.logger.info(f'🔄 Using forward fill (no download possible): {gap}')
                    filled_data = self._handle_small_gap(filled_data, gap, timestamp_column)
                    gap.fill_method = 'forward_fill_no_download'
                else:
                    # Try UnifiedGapFiller for critical gaps even without parameters
                    self.logger.info(f'🔄 Attempting UnifiedGapFiller for critical gap (no params): {gap}')
                    filled_data = self._handle_critical_gap(filled_data, gap, timestamp_column, symbol, exchange, timeframe)

        final_gaps = self._analyze_gaps(filled_data, timestamp_column)
        if final_gaps:
            self.logger.warning(f'Remaining gaps after filling: {len(final_gaps)}')
        else:
            self.logger.info('All gaps successfully filled')

        return filled_data

    def _analyze_gaps(self, data: pd.DataFrame, timestamp_column: str) -> List[GapInfo]:
        """Analyze gaps in the data."""
        gaps = []
        timestamps = data[timestamp_column].values

        # Detect timestamp unit by checking the magnitude
        if len(timestamps) > 1:
            time_diff = timestamps[1] - timestamps[0]
            if time_diff > 1e10:  # nanoseconds
                time_unit = 'nanoseconds'
                seconds_per_unit = 1e9
            elif time_diff > 1e7:  # milliseconds
                time_unit = 'milliseconds'
                seconds_per_unit = 1000
            else:  # seconds
                time_unit = 'seconds'
                seconds_per_unit = 1
            self.logger.debug(f"Detected timestamp unit: {time_unit}")
        else:
            seconds_per_unit = 1  # fallback

        for i in range(len(timestamps) - 1):
            current_time = timestamps[i]
            next_time = timestamps[i + 1]
            expected_next_time = current_time + (60 * seconds_per_unit)  # 60 seconds in the detected unit

            if next_time > expected_next_time:
                gap_size_raw = next_time - expected_next_time
                gap_size_seconds = gap_size_raw / seconds_per_unit  # Convert to seconds for classification
                gap_type = self._classify_gap(gap_size_seconds)
                gap = GapInfo(
                    start_time=expected_next_time,
                    end_time=next_time,
                    gap_size=gap_size_seconds,  # Store gap size in seconds
                    gap_type=gap_type
                )
                gaps.append(gap)

        return gaps

    def _classify_gap(self, gap_size: int) -> GapType:
        """Classify gap based on size."""
        if gap_size <= self.gap_thresholds[GapType.SMALL]:
            return GapType.SMALL
        elif gap_size <= self.gap_thresholds[GapType.MEDIUM]:
            return GapType.MEDIUM
        elif gap_size <= self.gap_thresholds[GapType.LARGE]:
            return GapType.LARGE
        else:
            return GapType.CRITICAL

    def _log_gap_analysis(self, gaps: List[GapInfo]) -> None:
        """Log gap analysis results."""
        gap_counts = {}
        for gap in gaps:
            gap_type = gap.gap_type.value
            if gap_type not in gap_counts:
                gap_counts[gap_type] = 0
            gap_counts[gap_type] += 1

        self.logger.info(f'Gap analysis: {len(gaps)} total gaps')
        for gap_type, count in gap_counts.items():
            self.logger.info(f'  {gap_type}: {count} gaps')

    def _is_gap_filled(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str) -> bool:
        """Check if a gap has been filled by verifying data exists in the gap period."""
        try:
            gap_start = gap.start_time
            gap_end = gap.end_time

            # Check if we have data points within the gap period
            gap_data = data[(data[timestamp_column] >= gap_start) & (data[timestamp_column] <= gap_end)]

            # Consider gap filled if we have at least one data point in the gap period
            return len(gap_data) > 0
        except Exception as e:
            self.logger.warning(f'Error checking if gap is filled: {e}')
            return False

    def _handle_small_gap(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str) -> pd.DataFrame:
        """Handle small gap with forward fill."""
        self.logger.info(f'Handling small gap with forward fill: {gap}')
        before_gap_idx = data[data[timestamp_column] <= gap.start_time].index[-1]
        filled_data = data.copy()

        missing_timestamps = []
        current_time = gap.start_time
        while current_time < gap.end_time:
            missing_timestamps.append(current_time)
            current_time += 60

        new_rows = []
        for timestamp in missing_timestamps:
            new_row = data.iloc[before_gap_idx].copy()
            new_row[timestamp_column] = timestamp
            new_rows.append(new_row)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            filled_data = pd.concat([filled_data, new_df], ignore_index=True)
            filled_data = filled_data.sort_values(timestamp_column).reset_index(drop=True)

        gap.filled = True
        gap.fill_method = 'forward_fill'
        return filled_data

    async def _handle_large_gap_with_download(
        self,
        data: pd.DataFrame,
        gap: GapInfo,
        timestamp_column: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Handle large gap by downloading missing data."""
        self.logger.info(f'Downloading data for gap: {gap}')
        try:
            downloaded_data = await self._download_missing_data(symbol, exchange, timeframe, gap.start_time, gap.end_time)
            if downloaded_data is not None and len(downloaded_data) > 0:
                filled_data = self._insert_downloaded_data(data, downloaded_data, timestamp_column)
                gap.filled = True
                gap.fill_method = 'download'
                gap.downloaded_data = downloaded_data
                self.logger.info(f'Successfully downloaded and inserted {len(downloaded_data)} rows')
                return filled_data
            else:
                self.logger.warning(f'No data downloaded for gap {gap}, using fallback')
                return self._handle_large_gap_with_fallback(data, gap, timestamp_column)
        except Exception as e:
            self.logger.error(f'Failed to download data for gap {gap}: {e}')
            return self._handle_large_gap_with_fallback(data, gap, timestamp_column)

    async def _download_missing_data(self, symbol: str, exchange: str, timeframe: str, start_time: int, end_time: int) -> Optional[pd.DataFrame]:
        """Download missing data from exchange."""
        try:
            start_dt = datetime.fromtimestamp(start_time)
            end_dt = datetime.fromtimestamp(end_time)
            self.logger.info(f'Downloading {symbol} data from {exchange} for {start_dt} to {end_dt}')

            if exchange.lower() == 'binance':
                from src.training.steps.data_collection.unified_data_downloader import UnifiedDataDownloader

                downloader = UnifiedDataDownloader()
                success, downloaded_data, error = await downloader.download_klines(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    start_date=start_dt,
                    end_date=end_dt
                )

                if not success:
                    self.logger.error(f"Failed to download data: {error}")
                    return None

                if downloaded_data is not None and len(downloaded_data) > 0:
                    # Convert timestamp to unix timestamp (seconds since epoch)
                    if 'timestamp' in downloaded_data.columns:
                        # Convert datetime to unix timestamp properly
                        dt_series = pd.to_datetime(downloaded_data['timestamp'])
                        try:
                            # For datetime64 arrays, convert to int64 nanoseconds then to seconds
                            if dt_series.dtype == 'datetime64[ns]':
                                downloaded_data['timestamp'] = dt_series.astype('int64') // 10**9
                            elif dt_series.dtype == 'datetime64[ms]':
                                downloaded_data['timestamp'] = dt_series.astype('int64')
                            else:
                                # Improved fallback for other datetime formats - avoid 1970 reference issues
                                try:
                                    # Try direct conversion to unix timestamp
                                    downloaded_data['timestamp'] = dt_series.astype('int64') // 10**9
                                except:
                                    # Last resort: use pandas timestamp conversion without explicit 1970 reference
                                    downloaded_data['timestamp'] = pd.to_datetime(dt_series).astype('int64') // 10**9
                        except Exception as conv_e:
                            self.logger.warning(f'Failed to convert timestamp using standard method: {conv_e}')
                            # Enhanced fallback: multiple conversion strategies
                            try:
                                # Strategy 1: Direct int64 conversion for unix timestamps
                                if dt_series.dtype == 'int64':
                                    downloaded_data['timestamp'] = dt_series
                                else:
                                    # Strategy 2: Convert to datetime then to unix timestamp
                                    dt_converted = pd.to_datetime(dt_series, errors='coerce')
                                    mask = dt_converted.notna()
                                    downloaded_data.loc[mask, 'timestamp'] = dt_converted[mask].astype('int64') // 10**9
                                    # Keep original values for failed conversions
                                    downloaded_data.loc[~mask, 'timestamp'] = dt_series[~mask]
                            except Exception as fallback_e:
                                self.logger.error(f'All timestamp conversion methods failed: {fallback_e}')
                                raise
                    return downloaded_data
                else:
                    self.logger.warning('No data returned from downloader')
                    return None
            else:
                self.logger.warning(f'Exchange {exchange} not supported for data download')
                return None
        except Exception as e:
            self.logger.error(f'Error downloading data: {e}')
            return None

    def _insert_downloaded_data(self, data: pd.DataFrame, downloaded_data: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Insert downloaded data into the main dataset."""
        combined_data = pd.concat([data, downloaded_data], ignore_index=True)
        combined_data = combined_data.sort_values(timestamp_column).reset_index(drop=True)
        combined_data = combined_data.drop_duplicates(subset=[timestamp_column])
        return combined_data

    def _handle_large_gap_with_fallback(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str) -> pd.DataFrame:
        """Handle large gap with fallback strategy (interpolation)."""
        self.logger.info(f'Using fallback strategy for gap: {gap}')
        filled_data = data.copy()
        before_gap_idx = data[data[timestamp_column] <= gap.start_time].index[-1]
        after_gap_idx = data[data[timestamp_column] >= gap.end_time].index[0]

        missing_timestamps = []
        current_time = gap.start_time
        while current_time < gap.end_time:
            missing_timestamps.append(current_time)
            current_time += 60

        for timestamp in missing_timestamps:
            time_diff = timestamp - data.iloc[before_gap_idx][timestamp_column]
            total_gap = data.iloc[after_gap_idx][timestamp_column] - data.iloc[before_gap_idx][timestamp_column]
            weight = time_diff / total_gap if total_gap > 0 else 0

            new_row = data.iloc[before_gap_idx].copy()
            new_row[timestamp_column] = timestamp

            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != timestamp_column:
                    before_val = data.iloc[before_gap_idx][col]
                    after_val = data.iloc[after_gap_idx][col]
                    interpolated_val = before_val + weight * (after_val - before_val)
                    new_row[col] = interpolated_val

            filled_data = pd.concat([filled_data, pd.DataFrame([new_row])], ignore_index=True)

        filled_data = filled_data.sort_values(timestamp_column).reset_index(drop=True)
        gap.filled = True
        gap.fill_method = 'interpolation_fallback'
        return filled_data

    def _handle_critical_gap(self, data: pd.DataFrame, gap: GapInfo, timestamp_column: str,
                           symbol: str = None, exchange: str = None, timeframe: str = '1m') -> pd.DataFrame:
        """Handle critical gap using UnifiedGapFiller."""
        self.logger.warning(f'🚨 CRITICAL GAP DETECTED: {gap}')

        # First try UnifiedGapFiller if available and we have the required parameters
        if UNIFIED_GAP_FILLER_AVAILABLE and symbol and exchange:
            self.logger.info('🔄 Attempting to fill critical gap using UnifiedGapFiller...')

            try:
                gap_filler = UnifiedGapFiller()

                # Convert gap timestamps to datetime for UnifiedGapFiller
                gap_start_dt = datetime.fromtimestamp(gap.start_time)
                gap_end_dt = datetime.fromtimestamp(gap.end_time)

                # Determine data type from timeframe
                data_type = 'klines' if timeframe.endswith('m') else 'futures'

                # Use detect_and_fill_gaps method
                fill_result = asyncio.run(gap_filler.detect_and_fill_gaps(
                    symbol=symbol,
                    exchange=exchange,
                    data_type=data_type,
                    start_date=gap_start_dt - timedelta(hours=1),  # Add buffer
                    end_date=gap_end_dt + timedelta(hours=1),      # Add buffer
                    auto_fill=True
                ))

                if fill_result.get('success', False) and fill_result.get('gaps_filled', 0) > 0:
                    self.logger.info(f'✅ Critical gap filled successfully via UnifiedGapFiller: {fill_result}')

                    # Try to load the newly downloaded data and merge it
                    try:
                        # The gap filler saves data to data_cache, we need to reload and merge
                        # This is a simplified approach - in practice you might want to reload from files
                        gap.filled = True
                        gap.fill_method = 'unified_gap_filler'
                        self.logger.info(f'✅ Critical gap {gap} filled via UnifiedGapFiller')
                        return data  # Return original data - the gap filler handles file updates
                    except Exception as merge_e:
                        self.logger.warning(f'⚠️ Gap filled but merge failed: {merge_e}')
                        gap.filled = True
                        gap.fill_method = 'unified_gap_filler_partial'
                        return data
                else:
                    self.logger.warning(f'⚠️ UnifiedGapFiller failed to fill critical gap: {fill_result}')

            except Exception as e:
                self.logger.error(f'❌ Error using UnifiedGapFiller for critical gap: {e}')

        # Fallback: Manual intervention required message
        self.logger.error('🚨 MANUAL INTERVENTION REQUIRED - CRITICAL GAPS REQUIRE EXTERNAL DATA COLLECTION')
        self.logger.error('🚨 CRITICAL GAPS WILL REMAIN AS GAPS IN THE DATA')
        self.logger.warning('⚠️ CRITICAL GAP NOT FILLED - CONSIDER RUNNING DEDICATED DATA COLLECTION')
        gap.filled = False
        gap.fill_method = 'manual_intervention_required_external_collection'
        return data  # Return data COMPLETELY UNCHANGED - critical gaps require external handling

    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'zscore',
        threshold: float = 3.0,
        columns: List[str] = None,
        raise_errors: bool = None
    ) -> List[OutlierInfo]:
        """Detect outliers in data using specified method."""
        if raise_errors is None:
            raise_errors = self.raise_errors

        if method not in self.detection_methods:
            self.logger.error(f'Unknown detection method: {method}')
            return []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        all_outliers = []
        for column in columns:
            if column not in data.columns:
                self.logger.warning(f'Column {column} not found in data')
                continue
            if not np.issubdtype(data[column].dtype, np.number):
                self.logger.warning(f'Column {column} is not numeric, skipping')
                continue

            clean_data = data[column].dropna()
            if len(clean_data) == 0:
                self.logger.warning(f'Column {column} has no valid data')
                continue

            outliers = self.detection_methods[method](data, column, threshold)
            all_outliers.extend(outliers)

        if all_outliers:
            self._log_outlier_details(all_outliers)
            if raise_errors:
                self._handle_outlier_errors(all_outliers)
            self.outlier_history.extend(all_outliers)

        return all_outliers

    def _detect_zscore_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> List[OutlierInfo]:
        """Detect outliers using Z-score method."""
        outliers = []
        try:
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outlier_indices = np.where(z_scores > threshold)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()
                max_z_score = z_scores.max()

                # Adjust severity thresholds for cryptocurrency data
                # Crypto markets have higher volatility and more extreme movements
                if max_z_score > threshold * 5:  # Increased from 3 to 5 for crypto
                    severity = OutlierSeverity.CRITICAL
                elif max_z_score > threshold * 3:  # Increased from 2 to 3 for crypto
                    severity = OutlierSeverity.HIGH
                elif max_z_score > threshold * 2:  # Increased from 1.5 to 2 for crypto
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method='zscore',
                    severity=severity,
                    threshold=threshold
                )
                outlier_info.context = {
                    'z_scores': z_scores[outlier_indices].tolist(),
                    'max_z_score': max_z_score,
                    'mean': data[column].mean(),
                    'std': data[column].std()
                }
                outliers.append(outlier_info)
        except Exception as e:
            self.logger.exception(f'Error in Z-score outlier detection: {e}')
        return outliers

    def _detect_iqr_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> List[OutlierInfo]:
        """Detect outliers using IQR method."""
        outliers = []
        try:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = np.where((data[column] < lower_bound) | (data[column] > upper_bound))[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()
                distances = []
                for idx in outlier_indices:
                    val = data[column].iloc[idx]
                    if val < lower_bound:
                        distances.append((lower_bound - val) / IQR)
                    else:
                        distances.append((val - upper_bound) / IQR)

                max_distance = max(distances)
                if max_distance > threshold * 2:
                    severity = OutlierSeverity.CRITICAL
                elif max_distance > threshold * 1.5:
                    severity = OutlierSeverity.HIGH
                elif max_distance > threshold * 1.2:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method='iqr',
                    severity=severity,
                    threshold=threshold
                )
                outlier_info.context = {
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'max_distance': max_distance
                }
                outliers.append(outlier_info)
        except Exception as e:
            self.logger.exception(f'Error in IQR outlier detection: {e}')
        return outliers

    def _detect_isolation_forest_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> List[OutlierInfo]:
        """Detect outliers using Isolation Forest method."""
        outliers = []
        try:
            from sklearn.ensemble import IsolationForest
            X = data[column].values.reshape(-1, 1)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X)
            outlier_indices = np.where(predictions == -1)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()
                anomaly_scores = iso_forest.decision_function(X)
                outlier_scores = anomaly_scores[outlier_indices]
                min_score = min(outlier_scores)

                if min_score < -0.5:
                    severity = OutlierSeverity.CRITICAL
                elif min_score < -0.3:
                    severity = OutlierSeverity.HIGH
                elif min_score < -0.1:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method='isolation_forest',
                    severity=severity,
                    threshold=threshold
                )
                outlier_info.context = {
                    'anomaly_scores': outlier_scores.tolist(),
                    'min_score': min_score,
                    'contamination': 0.1
                }
                outliers.append(outlier_info)
        except ImportError:
            self.logger.warning('scikit-learn not available for isolation forest outlier detection')
        except Exception as e:
            self.logger.exception(f'Error in isolation forest outlier detection: {e}')
        return outliers

    def _detect_lof_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> List[OutlierInfo]:
        """Detect outliers using Local Outlier Factor method."""
        outliers = []
        try:
            from sklearn.neighbors import LocalOutlierFactor
            X = data[column].values.reshape(-1, 1)
            lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
            predictions = lof.fit_predict(X)
            outlier_indices = np.where(predictions == -1)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()
                lof_scores = lof.negative_outlier_factor_
                outlier_scores = lof_scores[outlier_indices]
                min_score = min(outlier_scores)

                if min_score < -1.5:
                    severity = OutlierSeverity.CRITICAL
                elif min_score < -1.2:
                    severity = OutlierSeverity.HIGH
                elif min_score < -1.0:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method='local_outlier_factor',
                    severity=severity,
                    threshold=threshold
                )
                outlier_info.context = {
                    'lof_scores': outlier_scores.tolist(),
                    'min_score': min_score,
                    'contamination': 0.1
                }
                outliers.append(outlier_info)
        except ImportError:
            self.logger.warning('scikit-learn not available for LOF outlier detection')
        except Exception as e:
            self.logger.exception(f'Error in LOF outlier detection: {e}')
        return outliers

    def _detect_mahalanobis_outliers(self, data: pd.DataFrame, column: str, threshold: float) -> List[OutlierInfo]:
        """Detect outliers using Mahalanobis distance method."""
        outliers = []
        try:
            median = data[column].median()
            mad = np.median(np.abs(data[column] - median))
            if mad == 0:
                return outliers

            modified_z_scores = 0.6745 * (data[column] - median) / mad
            outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]

            if len(outlier_indices) > 0:
                outlier_values = data[column].iloc[outlier_indices].tolist()
                max_score = np.abs(modified_z_scores).max()

                if max_score > threshold * 2:
                    severity = OutlierSeverity.CRITICAL
                elif max_score > threshold * 1.5:
                    severity = OutlierSeverity.HIGH
                elif max_score > threshold * 1.2:
                    severity = OutlierSeverity.MEDIUM
                else:
                    severity = OutlierSeverity.LOW

                outlier_info = OutlierInfo(
                    column=column,
                    indices=outlier_indices.tolist(),
                    values=outlier_values,
                    method='mahalanobis',
                    severity=severity,
                    threshold=threshold
                )
                outlier_info.context = {
                    'modified_z_scores': modified_z_scores[outlier_indices].tolist(),
                    'max_score': max_score,
                    'median': median,
                    'mad': mad
                }
                outliers.append(outlier_info)
        except Exception as e:
            self.logger.exception(f'Error in Mahalanobis outlier detection: {e}')
        return outliers

    def _log_outlier_details(self, outliers: List[OutlierInfo]) -> None:
        """Log detailed outlier information with crypto-specific context."""
        if not outliers:
            return
        self.logger.info(f'🔍 Detected {len(outliers)} outlier groups')

        # Group outliers by severity for better analysis
        critical_outliers = [o for o in outliers if o.severity == OutlierSeverity.CRITICAL]
        high_outliers = [o for o in outliers if o.severity == OutlierSeverity.HIGH]
        medium_outliers = [o for o in outliers if o.severity == OutlierSeverity.MEDIUM]
        low_outliers = [o for o in outliers if o.severity == OutlierSeverity.LOW]

        # Log summary with crypto context
        if critical_outliers:
            self.logger.warning(f'🚨 CRITICAL outliers detected: {len(critical_outliers)} groups - These may indicate major market events (flash crashes, pumps, whale movements)')
        if high_outliers:
            self.logger.warning(f'⚠️ HIGH severity outliers: {len(high_outliers)} groups - Common in volatile crypto markets')
        if medium_outliers:
            self.logger.info(f'📊 MEDIUM severity outliers: {len(medium_outliers)} groups - Normal crypto market volatility')
        if low_outliers:
            self.logger.info(f'📈 LOW severity outliers: {len(low_outliers)} groups - Minor market fluctuations')

        # Log detailed information for critical and high outliers
        for outlier in critical_outliers + high_outliers:
            self.logger.warning(f'Outlier in {outlier.column}: {len(outlier.indices)} values, severity={outlier.severity.value}, method={outlier.method}')

            # Add crypto-specific context
            if outlier.column in ['volume', 'quote_volume']:
                self.logger.info(f'  💰 Volume outliers may indicate whale movements or major news events')
            elif outlier.column in ['price_range', 'body_size']:
                self.logger.info(f'  📈 Price range outliers may indicate flash crashes or pumps')
            elif outlier.column in ['trades']:
                self.logger.info(f'  🔄 Trade count outliers may indicate high-frequency trading periods')

            if outlier.severity == OutlierSeverity.CRITICAL:
                self.logger.error(f'Critical outlier details: {outlier}')
                self.logger.error(f'  Values: {outlier.values[:5]}...')
                self.logger.error(f'  Context: {outlier.context}')

    def _handle_outlier_errors(self, outliers: List[OutlierInfo]) -> None:
        """Handle outlier errors by raising exceptions or logging."""
        critical_outliers = [o for o in outliers if o.severity == OutlierSeverity.CRITICAL]
        high_outliers = [o for o in outliers if o.severity == OutlierSeverity.HIGH]

        if critical_outliers:
            error_msg = f'Critical outliers detected: {len(critical_outliers)} groups'
            for outlier in critical_outliers:
                error_msg += f'\n  {outlier.column}: {len(outlier.indices)} values'
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if high_outliers:
            error_msg = f'High severity outliers detected: {len(high_outliers)} groups'
            for outlier in high_outliers:
                error_msg += f'\n  {outlier.column}: {len(outlier.indices)} values'
            self.logger.error(error_msg)
            if self.raise_errors:
                raise ValueError(error_msg)

    def handle_outliers_with_strategy(self, data: pd.DataFrame, strategy: OutlierStrategy, 
                                    threshold: float = 3.0, columns: List[str] = None) -> pd.DataFrame:
        """Handle outliers using specified strategy."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        cleaned_data = data.copy()
        
        if strategy == OutlierStrategy.CLIP:
            return self._clip_outliers(cleaned_data, threshold, columns)
        elif strategy == OutlierStrategy.CAP:
            return self._cap_outliers(cleaned_data, threshold, columns)
        elif strategy == OutlierStrategy.REMOVE:
            return self._remove_outliers(cleaned_data, threshold, columns)
        elif strategy == OutlierStrategy.TRANSFORM:
            return self._transform_outliers(cleaned_data, threshold, columns)
        else:
            self.logger.warning(f"Unsupported outlier strategy: {strategy}")
            return cleaned_data

    def _clip_outliers(self, data: pd.DataFrame, threshold: float, columns: List[str]) -> pd.DataFrame:
        """Clip outliers to threshold using z-score method."""
        cleaned_data = data.copy()
        
        for column in columns:
            if column not in data.columns or not np.issubdtype(data[column].dtype, np.number):
                continue
                
            # Calculate z-scores
            mean_val = data[column].mean()
            std_val = data[column].std()
            
            if std_val == 0:
                continue
                
            z_scores = np.abs((data[column] - mean_val) / std_val)
            
            # Clip values that exceed threshold
            upper_bound = mean_val + threshold * std_val
            lower_bound = mean_val - threshold * std_val
            
            # Apply clipping
            cleaned_data[column] = np.clip(data[column], lower_bound, upper_bound)
            
            # Log clipping information
            clipped_count = np.sum(z_scores > threshold)
            if clipped_count > 0:
                self.logger.info(f"📊 Clipped {clipped_count} outliers in column '{column}' using threshold {threshold}")
        
        return cleaned_data

    def _cap_outliers(self, data: pd.DataFrame, threshold: float, columns: List[str]) -> pd.DataFrame:
        """Cap outliers using IQR method."""
        cleaned_data = data.copy()
        
        for column in columns:
            if column not in data.columns or not np.issubdtype(data[column].dtype, np.number):
                continue
                
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Apply capping
            cleaned_data[column] = np.clip(data[column], lower_bound, upper_bound)
            
            # Log capping information
            capped_count = np.sum((data[column] < lower_bound) | (data[column] > upper_bound))
            if capped_count > 0:
                self.logger.info(f"📊 Capped {capped_count} outliers in column '{column}' using IQR method")
        
        return cleaned_data

    def _remove_outliers(self, data: pd.DataFrame, threshold: float, columns: List[str]) -> pd.DataFrame:
        """Remove rows containing outliers."""
        cleaned_data = data.copy()
        
        for column in columns:
            if column not in data.columns or not np.issubdtype(data[column].dtype, np.number):
                continue
                
            # Calculate z-scores
            mean_val = data[column].mean()
            std_val = data[column].std()
            
            if std_val == 0:
                continue
                
            z_scores = np.abs((data[column] - mean_val) / std_val)
            
            # Remove outliers
            outlier_mask = z_scores > threshold
            cleaned_data = cleaned_data[~outlier_mask]
            
            # Log removal information
            removed_count = np.sum(outlier_mask)
            if removed_count > 0:
                self.logger.info(f"📊 Removed {removed_count} rows with outliers in column '{column}'")
        
        return cleaned_data

    def _transform_outliers(self, data: pd.DataFrame, threshold: float, columns: List[str]) -> pd.DataFrame:
        """Transform outliers using log transformation."""
        cleaned_data = data.copy()
        
        for column in columns:
            if column not in data.columns or not np.issubdtype(data[column].dtype, np.number):
                continue
                
            # Apply log transformation to reduce outlier impact
            # Add small constant to avoid log(0)
            min_val = data[column].min()
            if min_val <= 0:
                constant = abs(min_val) + 1
            else:
                constant = 0
                
            cleaned_data[column] = np.log1p(data[column] + constant)
            
            self.logger.info(f"📊 Applied log transformation to column '{column}' to reduce outlier impact")
        
        return cleaned_data

    def validate_data_schema(self, data: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
        """Validate data against a standard schema."""
        if schema_name not in self.standard_schemas:
            self.logger.error(f'Unknown schema: {schema_name}')
            return {'valid': False, 'error': f'Unknown schema: {schema_name}'}
        schema = self.standard_schemas[schema_name]
        return schema.validate_dataframe(data)

    def get_gap_report(self, data: pd.DataFrame, timestamp_column: str = 'timestamp') -> Dict[str, Any]:
        """Generate gap analysis report."""
        gaps = self._analyze_gaps(data, timestamp_column)
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_gaps': len(gaps),
            'gap_summary': {},
            'gap_details': []
        }

        for gap_type in GapType:
            gap_type_gaps = [g for g in gaps if g.gap_type == gap_type]
            report['gap_summary'][gap_type.value] = {
                'count': len(gap_type_gaps),
                'total_size': sum((g.gap_size for g in gap_type_gaps)),
                'avg_size': np.mean([g.gap_size for g in gap_type_gaps]) if gap_type_gaps else 0
            }

        for gap in gaps:
            report['gap_details'].append({
                'start_time': gap.start_time,
                'end_time': gap.end_time,
                'gap_size': gap.gap_size,
                'gap_type': gap.gap_type.value,
                'filled': gap.filled,
                'fill_method': gap.fill_method
            })

        return report

    def get_outlier_report(self) -> Dict[str, Any]:
        """Generate comprehensive outlier report."""
        if not self.outlier_history:
            return {'message': 'No outliers detected'}

        severity_counts = {}
        column_counts = {}
        method_counts = {}

        for outlier in self.outlier_history:
            severity = outlier.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            column = outlier.column
            if column not in column_counts:
                column_counts[column] = {'count': 0, 'total_values': 0}
            column_counts[column]['count'] += 1
            column_counts[column]['total_values'] += len(outlier.indices)

            method = outlier.method
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'timestamp': datetime.now().isoformat(),
            'total_outlier_groups': len(self.outlier_history),
            'severity_distribution': severity_counts,
            'column_distribution': column_counts,
            'method_distribution': method_counts,
            'recent_outliers': [
                {
                    'column': o.column,
                    'count': len(o.indices),
                    'severity': o.severity.value,
                    'method': o.method,
                    'timestamp': o.timestamp.isoformat()
                }
                for o in self.outlier_history[-10:]
            ]
        }

    async def clean_dataframe(self, data: pd.DataFrame, remove_constant_features: bool = False,
                       remove_duplicates: bool = True, handle_missing_values: bool = True,
                       timestamp_column: str = 'timestamp', symbol: str = None,
                       exchange: str = None, timeframe: str = None) -> Optional[pd.DataFrame]:
        """Clean dataframe with comprehensive data quality improvements.

        Args:
            data: DataFrame to clean
            remove_constant_features: Whether to remove constant features
            remove_duplicates: Whether to remove duplicate rows
            handle_missing_values: Whether to handle missing values
            timestamp_column: Name of timestamp column
            symbol: Trading symbol for data collection hooks
            exchange: Exchange name for data collection hooks
            timeframe: Timeframe for data collection hooks

        Returns:
            Cleaned DataFrame or None if cleaning failed
        """
        if data is None or len(data) == 0:
            self.logger.warning("⚠️ Input data is None or empty, returning None")
            return None

        self.logger.info(f"🧹 Starting data cleaning for {len(data)} rows, {len(data.columns)} columns")
        self.logger.info(f"   Data type: {self.data_type}")
        self.logger.info(f"   Gap thresholds: {self.gap_thresholds}")

        cleaned_data = data.copy()
        original_columns = set(cleaned_data.columns)
        removed_columns = []

        try:
            # 1. Remove constant features with adequate warnings
            if remove_constant_features:
                self.logger.info("🔍 Checking for constant features...")
                constant_features = self._identify_constant_features(cleaned_data)

                if constant_features:
                    self.logger.warning(f"⚠️ CONSTANT FEATURES DETECTED: {len(constant_features)} features with no variation")
                    self.logger.warning(f"   Constant features: {constant_features}")
                    self.logger.warning("   ⚠️ WARNING: Removing constant features may impact model performance")
                    self.logger.warning("   ⚠️ WARNING: Consider investigating data source for proper feature calculation")
                    self.logger.warning("   ⚠️ WARNING: This may indicate data processing pipeline issues")

                    # Remove constant features
                    cleaned_data = cleaned_data.drop(columns=constant_features)
                    removed_columns.extend(constant_features)
                    self.logger.info(f"✅ Removed {len(constant_features)} constant features")
                else:
                    self.logger.info("✅ No constant features detected")

            # 2. Remove duplicates
            if remove_duplicates:
                initial_rows = len(cleaned_data)
                if timestamp_column in cleaned_data.columns:
                    cleaned_data = cleaned_data.drop_duplicates(subset=[timestamp_column])
                else:
                    cleaned_data = cleaned_data.drop_duplicates()
                duplicates_removed = initial_rows - len(cleaned_data)
                if duplicates_removed > 0:
                    self.logger.info(f"✅ Removed {duplicates_removed} duplicate rows")

            # 3. Handle missing values with gap detection
            if handle_missing_values and timestamp_column in cleaned_data.columns:
                self.logger.info("🔍 Handling missing values and detecting gaps...")
                cleaned_data = await self.handle_missing_values_intelligently(
                    cleaned_data, timestamp_column, symbol, exchange, timeframe
                )

                # Log gap detection results
                gap_report = self.get_gap_report(cleaned_data, timestamp_column)
                if gap_report['total_gaps'] > 0:
                    self.logger.warning(f"⚠️ GAP DETECTION: Found {gap_report['total_gaps']} gaps in data")
                    self.logger.warning(f"   Large gaps (triggering re-download): {gap_report.get('large_gaps', 0)}")
                    self.logger.warning(f"   Critical gaps: {gap_report.get('critical_gaps', 0)}")

                    # Hook with data collection for large gaps
                    if gap_report.get('large_gaps', 0) > 0 or gap_report.get('critical_gaps', 0) > 0:
                        self.logger.warning("🔄 LARGE GAPS DETECTED - Consider triggering data re-collection")
                        self.logger.warning("   Hook with data collection pipeline for missing data")

                        # Trigger gap collection hook if parameters are available
                        if symbol and exchange:
                            try:
                                from src.utils.data.quality.gap_collection_hook import trigger_gap_collection

                                # Create gap info for collection hook
                                gap_info = {
                                    'gap_size': gap_report.get('max_gap_seconds', 0),
                                    'gap_type': 'large' if gap_report.get('large_gaps', 0) > 0 else 'critical',
                                    'total_gaps': gap_report.get('total_gaps', 0)
                                }

                                collection_result = trigger_gap_collection(
                                    gap_info, self.data_type, symbol, exchange, timeframe
                                )

                                if collection_result.get('triggered', False):
                                    self.logger.info("✅ Data collection hook triggered successfully")
                                else:
                                    self.logger.warning(f"⚠️ Data collection hook not triggered: {collection_result.get('reason', 'Unknown')}")

                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to trigger gap collection hook: {e}")
                        else:
                            self.logger.warning("⚠️ Missing symbol/exchange parameters - cannot trigger data collection hook")

            # 4. Handle outliers if enabled in config
            if self.config.outlier_detection_enabled:
                self.logger.info("🔍 Handling outliers...")
                try:
                    # Get outlier strategy from config
                    outlier_strategy = getattr(self.config, 'outlier_strategy', OutlierStrategy.CLIP)
                    outlier_threshold = getattr(self.config, 'outlier_threshold', 3.0)
                    
                    # Apply outlier handling
                    cleaned_data = self.handle_outliers_with_strategy(
                        cleaned_data, 
                        outlier_strategy, 
                        outlier_threshold
                    )
                    self.logger.info(f"✅ Outlier handling completed using {outlier_strategy.value} strategy")
                except Exception as e:
                    self.logger.warning(f"⚠️ Outlier handling failed: {e}")

            # 5. Final validation
            final_columns = set(cleaned_data.columns)
            removed_columns.extend(original_columns - final_columns)

            if removed_columns:
                self.logger.warning(f"⚠️ SUMMARY: Removed {len(removed_columns)} columns during cleaning")
                self.logger.warning(f"   Removed columns: {removed_columns}")

            self.logger.info(f"✅ Data cleaning completed: {len(cleaned_data)} rows, {len(cleaned_data.columns)} columns")
            return cleaned_data

        except Exception as e:
            self.logger.error(f"❌ Data cleaning failed: {e}")
            if self.raise_errors:
                raise
            return None

    def _identify_constant_features(self, data: pd.DataFrame) -> List[str]:
        """Identify constant features in the data with detailed logging."""
        constant_features = []

        # Define features that may be constant due to missing data sources
        excluded_constant_features = {
            'trade_volume', 'trade_count', 'avg_price',
            'min_price', 'max_price', 'volume_ratio'
        }

        for col in data.columns:
            # Skip aggtrades-derived features that may be constant due to missing data
            if col in excluded_constant_features:
                unique_count = data[col].nunique()
                if unique_count <= 1:
                    self.logger.info(f"   ℹ️ Skipping constant check for {col} (likely missing aggtrades data)")
                    continue

            unique_count = data[col].nunique()
            non_null_count = data[col].notna().sum()
            total_count = len(data)

            if unique_count <= 1:
                constant_features.append(col)
                self.logger.warning(f"   🚨 CONSTANT: '{col}' has {unique_count} unique values ({non_null_count}/{total_count} non-null)")

                # Provide specific insights for known problematic features
                if col == 'volume_ratio':
                    self.logger.warning("      💡 volume_ratio constant - likely missing aggtrades data or constant trading volume")
                elif col == 'trade_volume':
                    self.logger.warning("      💡 trade_volume constant - likely missing aggtrades data")
                else:
                    # Show sample values for other constant features
                    sample_vals = data[col].dropna().head(3).tolist()
                    self.logger.warning(f"      📊 Sample values: {sample_vals}")

            elif data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # Check for very low variance (effectively constant)
                std_val = data[col].std()
                if not pd.isna(std_val) and std_val < 1e-10:
                    constant_features.append(col)
                    self.logger.warning(f"   🚨 NEAR-CONSTANT: '{col}' has very low std ({std_val:.2e}) despite {unique_count} unique values")

        if constant_features:
            self.logger.warning(f"   📋 Total constant/near-constant features: {len(constant_features)}")
            self.logger.warning("   💡 RECOMMENDATION: Check data sources and collection pipeline")

        return constant_features

# Convenience functions for backwards compatibility
async def handle_missing_values_intelligently(
    data: pd.DataFrame,
    timestamp_column: str = 'timestamp',
    symbol: str = None,
    exchange: str = None,
    timeframe: str = '1m'
) -> pd.DataFrame:
    """Handle missing values intelligently based on gap size."""
    cleaner = DataCleaner()
    return await cleaner.handle_missing_values_intelligently(data, timestamp_column, symbol, exchange, timeframe)

def detect_outliers(
    data: pd.DataFrame,
    method: str = 'zscore',
    threshold: float = 3.0,
    columns: List[str] = None,
    raise_errors: bool = True
) -> List[OutlierInfo]:
    """Detect outliers in data using specified method."""
    cleaner = DataCleaner()
    return cleaner.detect_outliers(data, method, threshold, columns, raise_errors)

def validate_data_schema(data: pd.DataFrame, schema_name: str) -> Dict[str, Any]:
    """Validate data against a standard schema."""
    cleaner = DataCleaner()
    return cleaner.validate_data_schema(data, schema_name)

# Create global instances for backwards compatibility
enhanced_missing_value_handler = DataCleaner()
enhanced_outlier_handler = DataCleaner()

class DataCleanerManager:
    """Centralized manager for DataCleaner instances to prevent duplicate initialization."""

    _instance = None
    _lock = threading.Lock()
    _init_done = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not DataCleanerManager._init_done:
            self.logger = logging.getLogger('DataCleanerManager')
            self._cleaners: Dict[str, DataCleaner] = {}
            self._initialized = True
            DataCleanerManager._init_done = True
            self.logger.info("🔧 DataCleanerManager initialized (singleton)")

    def get_cleaner(self, data_type: str = 'klines', **kwargs) -> DataCleaner:
        """Get or create DataCleaner instance for specific data type."""
        start_time = time.time()

        if data_type not in self._cleaners:
            with self._lock:
                if data_type not in self._cleaners:
                    self.logger.info(f"🏭 Creating DataCleaner for data_type='{data_type}'")
                    self._cleaners[data_type] = DataCleaner(data_type=data_type, **kwargs)
                    duration = time.time() - start_time
                    self.logger.info(f"✅ DataCleaner for '{data_type}' created in {duration:.3f}s")
                else:
                    duration = time.time() - start_time
                    self.logger.info(f"♻️ Reusing existing DataCleaner for '{data_type}' (took {duration:.3f}s)")
        else:
            duration = time.time() - start_time
            self.logger.info(f"♻️ Reusing existing DataCleaner for '{data_type}' (took {duration:.3f}s)")

        return self._cleaners[data_type]

    def get_all_cleaners(self) -> Dict[str, DataCleaner]:
        """Get all created cleaners."""
        return self._cleaners.copy()

    def clear_cleaners(self):
        """Clear all cleaners (for testing)."""
        with self._lock:
            self._cleaners.clear()
            self.logger.info("🧹 All DataCleaner instances cleared")

# Global manager instance
_data_cleaner_manager = DataCleanerManager()

def get_data_cleaner(data_type: str = 'klines', **kwargs) -> DataCleaner:
    """Get DataCleaner instance through centralized manager."""
    return _data_cleaner_manager.get_cleaner(data_type, **kwargs)

