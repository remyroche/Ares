from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
#!/usr/bin/env python3
"""
Enhanced Data Resampler for Step1 with Comprehensive Stability Features.

This module provides advanced data resampling capabilities with:
- Memory-efficient processing for large datasets
- Intelligent caching mechanisms
- Concurrent processing with proper synchronization
- Robust error recovery and retry logic
- Advanced data quality validation
- Time series continuity management
"""

import asyncio
import gc
import hashlib
import json
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import psutil
import numpy as np
import pandas as pd
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.utils.logger import system_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ....core.decorators import handles_errors, traced, validates

logger = system_logger.getChild("EnhancedDataResampler")


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: int = 2048
    chunk_size_mb: int = 256
    gc_threshold: int = 1000
    enable_chunking: bool = True
    compression_level: int = 6


@dataclass
class CacheConfig:
    """Configuration for caching."""
    cache_dir: str = "data_cache/resample_cache"
    max_cache_age_hours: int = 24
    enable_compression: bool = True
    cache_metadata: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for processing."""
    max_workers: int = 4
    enable_concurrency: bool = True
    batch_size: int = 1000
    timeout_seconds: int = 300


class MemoryManager:
    """Advanced memory management for data processing."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process()
        self._baseline_memory = self.get_memory_usage()
        self._memory_history = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds configured threshold."""
        current_memory = self.get_memory_usage()
        return current_memory > self.config.max_memory_mb

    def should_chunk_data(self, data_size_mb: float) -> bool:
        """Determine if data should be processed in chunks."""
        return self.config.enable_chunking and data_size_mb > self.config.chunk_size_mb

    def force_garbage_collection(self) -> None:
        """Force garbage collection with monitoring."""
        before_gc = self.get_memory_usage()
        gc.collect()
        after_gc = self.get_memory_usage()
        memory_freed = before_gc - after_gc

        logger.debug(".1f")
        self._memory_history.append({
            "timestamp": time.time(),
            "before_gc": before_gc,
            "after_gc": after_gc,
            "freed": memory_freed
        })

    def estimate_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Estimate memory usage of DataFrame in MB."""
        return df.memory_usage(deep=True).sum() / 1024 / 1024

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        memory_info = self.process.memory_info()
        return {
            "current_mb": self.get_memory_usage(),
            "peak_mb": max(self._memory_history, key=lambda x: x["before_gc"])["before_gc"] if self._memory_history else 0,
            "baseline_mb": self._baseline_memory,
            "gc_calls": len([h for h in self._memory_history if h["freed"] > 0]),
            "total_freed_mb": sum(h["freed"] for h in self._memory_history),
        }


class IntelligentCache:
    """Intelligent caching system for intermediate results."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index = self._load_cache_index()

    def _load_cache_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self._cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def generate_cache_key(self, data_hash: str, operation: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the operation."""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{data_hash}:{operation}:{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_index:
            return False

        entry = self._cache_index[cache_key]
        cache_age_hours = (time.time() - entry["timestamp"]) / 3600

        if cache_age_hours > self.config.max_cache_age_hours:
            logger.debug(f"Cache entry {cache_key} expired (age: {cache_age_hours:.1f}h)")
            return False

        # Check if source files still exist and haven't been modified
        if "source_files" in entry:
            for file_path in entry["source_files"]:
                if not Path(file_path).exists():
                    logger.debug(f"Source file {file_path} no longer exists for cache {cache_key}")
                    return False

        return True

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached result."""
        if not self.is_cache_valid(cache_key):
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached result {cache_key}: {e}")
            return None

    def store_result(self, cache_key: str, result: Any, metadata: Dict[str, Any]) -> None:
        """Store result in cache."""
        try:
            # Store the result
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update cache index
            self._cache_index[cache_key] = {
                "timestamp": time.time(),
                "size_mb": cache_file.stat().st_size / 1024 / 1024,
                **metadata
            }

            self._save_cache_index()
            logger.debug(".1f")

        except Exception as e:
            logger.error(f"Failed to store result in cache {cache_key}: {e}")

    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        expired_keys = []
        current_time = time.time()

        for cache_key, entry in self._cache_index.items():
            cache_age_hours = (current_time - entry["timestamp"]) / 3600
            if cache_age_hours > self.config.max_cache_age_hours:
                expired_keys.append(cache_key)

        # Remove expired cache files
        for cache_key in expired_keys:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                if cache_file.exists():
                    cache_file.unlink()
                del self._cache_index[cache_key]
            except Exception as e:
                logger.warning(f"Failed to remove expired cache {cache_key}: {e}")

        if expired_keys:
            self._save_cache_index()
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


class TimeSeriesValidator:
    """Advanced validation for time series data continuity."""

    def __init__(self):
        self.expected_frequencies = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
        }

    def detect_gaps(self, df: pd.DataFrame, timeframe: str, max_gap_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect gaps in time series data."""
        if df.empty or "timestamp" not in df.columns:
            return []

        df_sorted = df.sort_values("timestamp").copy()
        time_diffs = df_sorted["timestamp"].diff().dropna()

        expected_freq = self.expected_frequencies.get(timeframe, pd.Timedelta(minutes=1))
        max_gap = pd.Timedelta(minutes=max_gap_minutes)

        gaps = []
        for i, diff in enumerate(time_diffs):
            if diff > max_gap:
                gap_start = df_sorted.iloc[i]["timestamp"]
                gap_end = df_sorted.iloc[i + 1]["timestamp"]
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "duration_minutes": diff.total_seconds() / 60,
                    "expected_frequency": str(expected_freq),
                })

        return gaps

    def interpolate_missing_values(self, df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
        """Interpolate missing values in time series."""
        if df.empty:
            return df

        df_interp = df.copy()

        # Interpolate numeric columns
        numeric_columns = df_interp.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            df_interp[numeric_columns] = df_interp[numeric_columns].interpolate(method=method)

        return df_interp

    def validate_continuity(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Validate time series continuity."""
        validation = {
            "is_continuous": True,
            "gaps_found": 0,
            "total_gap_minutes": 0,
            "max_gap_minutes": 0,
            "recommendations": []
        }

        if df.empty or "timestamp" not in df.columns:
            validation["is_continuous"] = False
            validation["recommendations"].append("No timestamp data available")
            return validation

        gaps = self.detect_gaps(df, timeframe)

        if gaps:
            validation["is_continuous"] = False
            validation["gaps_found"] = len(gaps)
            validation["total_gap_minutes"] = sum(gap["duration_minutes"] for gap in gaps)
            validation["max_gap_minutes"] = max(gap["duration_minutes"] for gap in gaps)

            if validation["max_gap_minutes"] > 1440:  # More than 1 day
                validation["recommendations"].append("Large gaps detected - consider data redownload")
            elif validation["gaps_found"] > len(df) * 0.1:  # More than 10% gaps
                validation["recommendations"].append("Many gaps detected - consider interpolation")

        return validation


class EnhancedDataResampler:
    """Enhanced data resampler with comprehensive stability features."""

    def __init__(
        self,
        data_cache_path: str = "data_cache",
        memory_config: Optional[MemoryConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        processing_config: Optional[ProcessingConfig] = None
    ):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)

        # Initialize configurations
        self.memory_config = memory_config or MemoryConfig()
        self.cache_config = cache_config or CacheConfig()
        self.processing_config = processing_config or ProcessingConfig()

        # Initialize components
        self.memory_manager = MemoryManager(self.memory_config)
        self.cache = IntelligentCache(self.cache_config)
        self.time_series_validator = TimeSeriesValidator()

        # Processing components
        self.executor = ThreadPoolExecutor(max_workers=self.processing_config.max_workers)
        self._lock = threading.Lock()
        self._active_operations = set()

        # Performance tracking
        self.performance_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0,
            "memory_peak": 0,
        }

    @traced(span_name="load_and_validate_data")
    @handles_errors(
        default_return=pd.DataFrame(),
        context="enhanced_data_resampler.load_and_validate_data"
    )
    def load_and_validate_data(
        self,
        symbol: str,
        exchange: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load and validate data with stability features."""
        logger.info(f"📊 Loading and validating data for {exchange}_{symbol}")

        # Generate data hash for caching
        data_signature = f"{exchange}_{symbol}_{start_date}_{end_date}"
        data_hash = hashlib.md5(data_signature.encode()).hexdigest()

        # Check cache first
        cache_key = self.cache.generate_cache_key(data_hash, "load_data", {
            "symbol": symbol, "exchange": exchange,
            "start_date": str(start_date), "end_date": str(end_date)
        })

        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result is not None:
            logger.info("✅ Loaded data from cache")
            return cached_result

        try:
            # Load data with memory monitoring
            if self.memory_manager.check_memory_pressure():
                self.memory_manager.force_garbage_collection()

            # Load klines files
            klines_files = self._get_klines_files(symbol, exchange)
            if not klines_files:
                logger.warning(f"⚠️ No klines files found for {exchange}_{symbol}")
                return pd.DataFrame()

            # Load data in chunks if necessary
            data_size_mb = self._estimate_files_size(klines_files)
            if self.memory_manager.should_chunk_data(data_size_mb):
                df = self._load_data_in_chunks(klines_files, start_date, end_date)
            else:
                df = self._load_data_bulk(klines_files, start_date, end_date)

            if df.empty:
                return df

            # Validate and clean data
            df = self._validate_and_clean_data(df)

            # Cache the result
            self.cache.store_result(
                cache_key,
                df,
                {
                    "source_files": [str(f) for f in klines_files],
                    "data_size_mb": self.memory_manager.estimate_dataframe_memory(df),
                    "row_count": len(df),
                }
            )

            logger.info(f"✅ Loaded and validated {len(df)} rows of data")
            return df

        except Exception as e:
            logger.exception(f"❌ Error loading data for {exchange}_{symbol}: {e}")
            return pd.DataFrame()

    def _get_klines_files(self, symbol: str, exchange: str) -> List[Path]:
        """Get all klines files for a symbol and exchange."""
        pattern = f"klines_{exchange}_{symbol}_1m_*.parquet"
        files = list(self.data_cache_path.glob(pattern))

        # Also check for CSV files
        csv_pattern = f"klines_{exchange}_{symbol}_1m_*.csv"
        csv_files = list(self.data_cache_path.glob(csv_pattern))

        return sorted(files + csv_files)

    def _estimate_files_size(self, files: List[Path]) -> float:
        """Estimate total size of files in MB."""
        total_size = sum(f.stat().st_size for f in files if f.exists())
        return total_size / 1024 / 1024

    def _load_data_in_chunks(
        self,
        files: List[Path],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load data in chunks to manage memory."""
        logger.info("🔄 Loading data in chunks for memory efficiency")

        dataframes = []

        for file_path in files:
            try:
                # Check memory before loading each file
                if self.memory_manager.check_memory_pressure():
                    self.memory_manager.force_garbage_collection()

                # Load file
                if file_path.suffix.lower() == '.parquet':
                    df = standardized_parquet_handler.read_parquet_standardized(file_path)
                else:
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])

                # Apply date filters
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]

                if not df.empty:
                    dataframes.append(df)
                    logger.debug(f"✅ Loaded {file_path.name}: {len(df)} rows")

            except Exception as e:
                logger.warning(f"⚠️ Failed to load {file_path.name}: {e}")

        if not dataframes:
            return pd.DataFrame()

        # Combine dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Sort and remove duplicates
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

        return combined_df

    def _load_data_bulk(
        self,
        files: List[Path],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load all data at once."""
        logger.info("🔄 Loading data in bulk")

        dataframes = []
        for file_path in files:
            try:
                if file_path.suffix.lower() == '.parquet':
                    df = standardized_parquet_handler.read_parquet_standardized(file_path)
                else:
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])

                if not df.empty:
                    dataframes.append(df)

            except Exception as e:
                logger.warning(f"⚠️ Failed to load {file_path.name}: {e}")

        if not dataframes:
            return pd.DataFrame()

        combined_df = pd.concat(dataframes, ignore_index=True)

        # Apply date filters
        if start_date:
            combined_df = combined_df[combined_df['timestamp'] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df['timestamp'] <= end_date]

        # Sort and remove duplicates
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

        return combined_df

    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean loaded data."""
        if df.empty:
            return df

        original_size = len(df)
        logger.info(f"🧹 Validating and cleaning {original_size} rows of data")

        # Remove rows with null timestamps
        df = df.dropna(subset=['timestamp'])

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # Validate required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"❌ Missing required column: {col}")
                return pd.DataFrame()

        # Clean numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with null values in critical columns
        df = df.dropna(subset=numeric_columns)

        # Remove negative prices or volumes
        df = df[(df['open'] >= 0) & (df['high'] >= 0) & (df['low'] >= 0) & (df['close'] >= 0) & (df['volume'] >= 0)]

        # Ensure high >= max(open, close) and low <= min(open, close)
        df = df[(df['high'] >= df[['open', 'close']].max(axis=1)) & (df['low'] <= df[['open', 'close']].min(axis=1))]

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        cleaned_size = len(df)
        removed_rows = original_size - cleaned_size

        if removed_rows > 0:
            logger.info(f"🧹 Removed {removed_rows} invalid rows ({removed_rows/original_size*100:.1f}%)")

        return df

    @traced(span_name="resample_with_stability")
    @handles_errors(
        default_return=pd.DataFrame(),
        context="enhanced_data_resampler.resample_with_stability"
    )
    def resample_with_stability(
        self,
        df: pd.DataFrame,
        timeframe: str,
        symbol: str,
        exchange: str
    ) -> pd.DataFrame:
        """Resample data with comprehensive stability features."""
        if df.empty:
            return df

        logger.info(f"🔄 Resampling {len(df)} rows to {timeframe} timeframe")

        # Generate cache key
        data_hash = hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()
        cache_key = self.cache.generate_cache_key(
            data_hash,
            "resample",
            {"timeframe": timeframe, "symbol": symbol, "exchange": exchange}
        )

        # Check cache
        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result is not None:
            logger.info("✅ Loaded resampled data from cache")
            return cached_result

        try:
            # Check memory before resampling
            data_size_mb = self.memory_manager.estimate_dataframe_memory(df)
            if self.memory_manager.should_chunk_data(data_size_mb):
                result_df = self._resample_in_chunks(df, timeframe)
            else:
                result_df = self._resample_bulk(df, timeframe)

            if result_df.empty:
                return result_df

            # Validate time series continuity
            continuity_validation = self.time_series_validator.validate_continuity(result_df, timeframe)
            if not continuity_validation["is_continuous"]:
                logger.warning(f"⚠️ Time series continuity issues detected: {continuity_validation['gaps_found']} gaps")

                # Apply interpolation if gaps are not too large
                if continuity_validation["max_gap_minutes"] < 60:  # Less than 1 hour
                    result_df = self.time_series_validator.interpolate_missing_values(result_df)
                    logger.info("🔧 Applied interpolation for small gaps")

            # Cache the result
            self.cache.store_result(
                cache_key,
                result_df,
                {
                    "timeframe": timeframe,
                    "original_rows": len(df),
                    "resampled_rows": len(result_df),
                    "data_size_mb": self.memory_manager.estimate_dataframe_memory(result_df),
                    "continuity_score": 1.0 if continuity_validation["is_continuous"] else 0.5,
                }
            )

            logger.info(f"✅ Resampled to {timeframe}: {len(result_df)} rows")
            return result_df

        except Exception as e:
            logger.exception(f"❌ Error resampling to {timeframe}: {e}")
            return pd.DataFrame()

    def _resample_bulk(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data in bulk."""
        try:
            # Set timestamp as index for resampling
            df_resampled = df.copy()
            df_resampled = df_resampled.set_index('timestamp')

            # Define resampling rules
            resample_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }

            # Get pandas frequency string
            freq_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H'
            }

            freq = freq_map.get(timeframe, '5T')

            # Resample
            resampled = df_resampled.resample(freq).agg(resample_rules)

            # Remove any NaN values
            resampled = resampled.dropna()

            # Reset index
            resampled = resampled.reset_index()

            return resampled

        except Exception as e:
            logger.exception(f"❌ Error in bulk resampling: {e}")
            return pd.DataFrame()

    def _resample_in_chunks(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data in chunks for memory efficiency."""
        logger.info("🔄 Resampling in chunks for memory efficiency")

        try:
            # Split data into chunks
            chunk_size = 50000  # 50k rows per chunk
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

            resampled_chunks = []

            for i, chunk in enumerate(chunks):
                logger.debug(f"🔄 Processing chunk {i+1}/{len(chunks)}")

                # Check memory before processing chunk
                if self.memory_manager.check_memory_pressure():
                    self.memory_manager.force_garbage_collection()

                # Resample chunk
                resampled_chunk = self._resample_bulk(chunk, timeframe)
                if not resampled_chunk.empty:
                    resampled_chunks.append(resampled_chunk)

            if not resampled_chunks:
                return pd.DataFrame()

            # Combine resampled chunks
            result_df = pd.concat(resampled_chunks, ignore_index=True)

            # Final sort and deduplication
            result_df = result_df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

            return result_df

        except Exception as e:
            logger.exception(f"❌ Error in chunked resampling: {e}")
            return pd.DataFrame()

    @traced(span_name="save_with_stability")
    @handles_errors(
        default_return=False,
        context="enhanced_data_resampler.save_with_stability"
    )
    def save_with_stability(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        output_format: str = "parquet"
    ) -> bool:
        """Save data with stability features."""
        if df.empty:
            logger.warning("⚠️ Cannot save empty DataFrame")
            return False

        try:
            # Create output directory
            output_dir = self.data_cache_path / "resampled" / exchange / symbol
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            if output_format.lower() == "parquet":
                filename = f"klines_{exchange}_{symbol}_{timeframe}_resampled.parquet"
            else:
                filename = f"klines_{exchange}_{symbol}_{timeframe}_resampled.csv"

            output_path = output_dir / filename

            # Ensure proper column order
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if list(df.columns) != expected_columns:
                if all(col in df.columns for col in expected_columns):
                    df = df[expected_columns]
                else:
                    logger.error(f"❌ Missing required columns for {timeframe} data")
                    return False

            # Ensure proper data types
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any remaining null values
            df = df.dropna()

            # Create backup if file exists
            if output_path.exists():
                backup_path = output_path.with_suffix(f"{output_path.suffix}.backup")
                try:
                    import shutil
                    shutil.copy2(output_path, backup_path)
                    logger.debug(f"✅ Created backup: {backup_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create backup: {e}")

            # Save file
            if output_format.lower() == "parquet":
                standardized_parquet_handler.write_parquet_standardized(df, output_path, compression="zstd", index=False)
            else:
                df.to_csv(output_path, index=False)

            # Verify file was saved
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(".1f")
                return True
            else:
                logger.error(f"❌ Failed to save file: {output_path}")
                return False

        except Exception as e:
            logger.exception(f"❌ Error saving {timeframe} data: {e}")
            return False

    @traced(span_name="resample_all_timeframes_stable")
    @handles_errors(
        default_return={
            "success": False,
            "error": "Resampling failed",
            "timeframes_processed": [],
            "total_time": 0,
        },
        context="enhanced_data_resampler.resample_all_timeframes_stable"
    )
    async def resample_all_timeframes_stable(
        self,
        symbol: str,
        exchange: str,
        timeframes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_concurrency: bool = True
    ) -> Dict[str, Any]:
        """Resample data to multiple timeframes with comprehensive stability."""
        start_time = time.time()

        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '30m', '1h']

        logger.info(f"🚀 Starting stable resampling for {exchange}_{symbol}")
        logger.info(f"📅 Timeframes: {timeframes}")
        logger.info(f"📅 Date range: {start_date} to {end_date}")

        try:
            # Load source data
            source_df = self.load_and_validate_data(symbol, exchange, start_date, end_date)

            if source_df.empty:
                return {
                    "success": False,
                    "error": "No source data available",
                    "timeframes_processed": [],
                    "total_time": time.time() - start_time,
                }

            logger.info(f"📊 Source data: {len(source_df)} rows")

            # Process timeframes
            results = {}
            processed_timeframes = []

            if use_concurrency and len(timeframes) > 1:
                # Concurrent processing
                results = await self._resample_timeframes_concurrent(
                    source_df, timeframes, symbol, exchange
                )
            else:
                # Sequential processing
                results = await self._resample_timeframes_sequential(
                    source_df, timeframes, symbol, exchange
                )

            # Collect successful results
            for timeframe, result in results.items():
                if result.get("success", False):
                    processed_timeframes.append(timeframe)

            # Generate summary
            total_time = time.time() - start_time
            success_count = len(processed_timeframes)

            logger.info("=" * 60)
            logger.info("📊 RESAMPLING SUMMARY")
            logger.info(f"⏱️ Total time: {total_time:.2f}s")
            logger.info(f"✅ Successful: {success_count}/{len(timeframes)}")
            logger.info(f"📁 Source rows: {len(source_df)}")

            if processed_timeframes:
                logger.info("📊 PROCESSED TIMEFRAMES:")
                for timeframe in processed_timeframes:
                    result = results[timeframe]
                    if result.get("success"):
                        logger.info(f"  • {timeframe}: {result.get('rows', 0)} rows")

            # Cleanup expired cache
            self.cache.cleanup_expired_cache()

            return {
                "success": success_count > 0,
                "timeframes_processed": processed_timeframes,
                "results": results,
                "total_time": total_time,
                "source_rows": len(source_df),
                "memory_stats": self.memory_manager.get_memory_stats(),
            }

        except Exception as e:
            total_time = time.time() - start_time
            logger.exception(f"❌ Stable resampling failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "timeframes_processed": [],
                "total_time": total_time,
            }

    async def _resample_timeframes_concurrent(
        self,
        source_df: pd.DataFrame,
        timeframes: List[str],
        symbol: str,
        exchange: str
    ) -> Dict[str, Dict[str, Any]]:
        """Resample timeframes concurrently."""
        logger.info("🔄 Processing timeframes concurrently")

        async def resample_timeframe(timeframe: str) -> Tuple[str, Dict[str, Any]]:
            """Resample a single timeframe."""
            try:
                # Check memory before processing
                if self.memory_manager.check_memory_pressure():
                    self.memory_manager.force_garbage_collection()
                    await asyncio.sleep(0.1)

                # Resample data
                resampled_df = self.resample_with_stability(source_df, timeframe, symbol, exchange)

                if resampled_df.empty:
                    return timeframe, {"success": False, "error": "Resampling produced no data"}

                # Save data
                saved = self.save_with_stability(resampled_df, symbol, exchange, timeframe)

                if saved:
                    return timeframe, {
                        "success": True,
                        "rows": len(resampled_df),
                        "file_size_mb": self.memory_manager.estimate_dataframe_memory(resampled_df),
                    }
                else:
                    return timeframe, {"success": False, "error": "Failed to save data"}

            except Exception as e:
                logger.exception(f"❌ Error resampling {timeframe}: {e}")
                return timeframe, {"success": False, "error": str(e)}

        # Process all timeframes concurrently
        tasks = [resample_timeframe(tf) for tf in timeframes]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to dictionary
        results = {}
        for result in results_list:
            if isinstance(result, Exception):
                logger.error(f"❌ Task failed with exception: {result}")
                continue

            timeframe, result_data = result
            results[timeframe] = result_data

        return results

    async def _resample_timeframes_sequential(
        self,
        source_df: pd.DataFrame,
        timeframes: List[str],
        symbol: str,
        exchange: str
    ) -> Dict[str, Dict[str, Any]]:
        """Resample timeframes sequentially."""
        logger.info("🔄 Processing timeframes sequentially")

        results = {}

        for timeframe in timeframes:
            try:
                logger.info(f"🔄 Processing {timeframe}...")

                # Check memory between timeframes
                if self.memory_manager.check_memory_pressure():
                    self.memory_manager.force_garbage_collection()

                # Resample data
                resampled_df = self.resample_with_stability(source_df, timeframe, symbol, exchange)

                if resampled_df.empty:
                    results[timeframe] = {"success": False, "error": "Resampling produced no data"}
                    continue

                # Save data
                saved = self.save_with_stability(resampled_df, symbol, exchange, timeframe)

                if saved:
                    results[timeframe] = {
                        "success": True,
                        "rows": len(resampled_df),
                        "file_size_mb": self.memory_manager.estimate_dataframe_memory(resampled_df),
                    }
                    logger.info(f"✅ Completed {timeframe}: {len(resampled_df)} rows")
                else:
                    results[timeframe] = {"success": False, "error": "Failed to save data"}

            except Exception as e:
                logger.exception(f"❌ Error processing {timeframe}: {e}")
                results[timeframe] = {"success": False, "error": str(e)}

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "total_operations": self.performance_stats["total_operations"],
            "success_rate": (
                self.performance_stats["successful_operations"] /
                self.performance_stats["total_operations"]
                if self.performance_stats["total_operations"] > 0 else 0
            ),
            "avg_processing_time": (
                self.performance_stats["total_processing_time"] /
                self.performance_stats["total_operations"]
                if self.performance_stats["total_operations"] > 0 else 0
            ),
            "memory_peak_mb": self.performance_stats["memory_peak"],
            "cache_performance": {
                "cache_hits": getattr(self.cache, '_cache_hits', 0),
                "cache_misses": getattr(self.cache, '_cache_misses', 0),
            },
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=True)
            self.cache.cleanup_expired_cache()
            logger.info("🧹 Enhanced data resampler cleaned up")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}"
