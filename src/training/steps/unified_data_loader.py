# src/training/steps/unified_data_loader.py

import asyncio
import os
import sys
import gc
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union, List, Dict, Iterator
from functools import lru_cache

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    PYARROW_AVAILABLE = True
except ImportError:
    pa = None
    pq = None
    ds = None
    pc = None
    PYARROW_AVAILABLE = False

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import (
    error,
    warning,
    failed,
    missing,
)
from src.utils.decorators import with_tracing_span, guard_dataframe_nulls

# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    monitor_pipeline_step,
    validate_pipeline_input,
    monitor_pipeline_performance,
    PipelineValidationLevel,
)


class UnifiedDataLoader:
    """
    Optimized unified data loader for ML training with streaming, caching, and memory management.

    This loader provides efficient access to the unified Parquet partitioned data format
    with optimizations for large-scale ML training scenarios.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("UnifiedDataLoader")
        self.data_cache_dir = "data_cache"

        # Performance optimization settings
        self.optimization_config = config.get("computational_optimization", {})
        self.memory_config = self.optimization_config.get("memory_management", {})
        self.caching_config = self.optimization_config.get("caching", {})

        # Memory management
        self.memory_threshold = self.memory_config.get("memory_threshold", 0.8)
        self.chunk_size = self.optimization_config.get("data_streaming", {}).get(
            "chunk_size", 10000
        )

        # Caching
        self._data_cache = {}
        self.max_cache_size = self.caching_config.get("max_cache_size", 1000)

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.memory_threshold:
                self.logger.warning(f"⚠️ High memory usage: {memory_percent:.1%}")
                self._cleanup_memory()
                return False
            return True
        except Exception:
            return True  # Continue if memory check fails

    def _cleanup_memory(self):
        """Clean up memory by clearing cache and running garbage collection."""
        self._data_cache.clear()
        gc.collect()
        self.logger.info("🧹 Memory cleanup completed")

    def safe_read_parquet_with_logging(
        self,
        file_path: str,
        columns: List[str] = None,
        nrows: Optional[int] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Safely read parquet file with comprehensive logging and error handling.
        Equivalent to step3_regime_data_splitting.py's _safe_read_parquet_with_logging.

        Args:
            file_path: Path to the parquet file
            columns: Optional list of columns to read
            nrows: Optional number of rows to read
            **kwargs: Additional arguments for parquet reading

        Returns:
            DataFrame if successful, None if all attempts fail
        """
        self.logger.info(f"🔧 Attempting to read parquet file: {file_path}")

        if os.path.exists(file_path):
            self.logger.info(
                f"   File size: {os.path.getsize(file_path) / 1024:.2f} KB"
            )
        else:
            self.logger.error(f"❌ File does not exist: {file_path}")
            return None

        self.logger.info(f"   Columns requested: {columns if columns else 'ALL'}")
        if nrows:
            self.logger.info(f"   Rows requested: {nrows}")

        # Strategy 1: Fastparquet engine (avoid PyArrow completely)
        try:
            self.logger.info(
                "   Trying strategy 1/2: Fastparquet engine (PyArrow disabled)"
            )
            df = pd.read_parquet(
                file_path, columns=columns, engine="fastparquet", **kwargs
            )

            if nrows and len(df) > nrows:
                df = df.head(nrows)

            self.logger.info(f"✅ Successfully read with strategy 1: {df.shape}")
            return df
        except Exception as e:
            self.logger.warning(f"   Strategy 1 failed: {e}")

        # Strategy 2: Basic pandas read_parquet (as last resort)
        try:
            self.logger.info(
                "   Trying strategy 2/2: Basic pandas read_parquet (last resort)"
            )
            df = pd.read_parquet(file_path, columns=columns, **kwargs)

            if nrows and len(df) > nrows:
                df = df.head(nrows)

            self.logger.info(f"✅ Successfully read with strategy 2: {df.shape}")
            return df
        except Exception as e:
            self.logger.warning(f"   Strategy 2 failed: {e}")

        self.logger.error(f"❌ All strategies failed for file: {file_path}")
        return None

    @lru_cache(maxsize=100)
    def _get_partition_paths(
        self, base_path: str, start_date: datetime, end_date: datetime
    ) -> List[str]:
        """Get only the relevant partition paths for the date range."""
        relevant_paths = []

        try:
            # Calculate year range
            start_year = start_date.year
            end_year = end_date.year

            for year in range(start_year, end_year + 1):
                year_path = os.path.join(base_path, f"year={year}")
                if not os.path.exists(year_path):
                    continue

                # Calculate month range for this year
                start_month = start_date.month if year == start_year else 1
                end_month = end_date.month if year == end_year else 12

                for month in range(start_month, end_month + 1):
                    month_path = os.path.join(year_path, f"month={month:02d}")
                    if not os.path.exists(month_path):
                        continue

                    # Calculate day range for this month
                    if year == start_year and month == start_month:
                        start_day = start_date.day
                    else:
                        start_day = 1

                    if year == end_year and month == end_month:
                        end_day = end_date.day
                    else:
                        # Get last day of month
                        if month == 12:
                            end_day = 31
                        else:
                            next_month = datetime(year, month + 1, 1)
                            end_day = (next_month - timedelta(days=1)).day

                    for day in range(start_day, end_day + 1):
                        day_path = os.path.join(month_path, f"day={day:02d}")
                        if os.path.exists(day_path):
                            # Add all parquet files in this day directory
                            for file in os.listdir(day_path):
                                if file.endswith(".parquet"):
                                    relevant_paths.append(os.path.join(day_path, file))

            return relevant_paths

        except Exception as e:
            self.logger.error(f"❌ Error getting partition paths: {e}")
            return []

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="unified data loading",
    )
    async def load_unified_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: Optional[int] = None,
        use_streaming: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load unified data with comprehensive error handling and fallback strategies.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            start_date: Start date for data loading
            end_date: End date for data loading
            lookback_days: Number of days to look back
            use_streaming: Whether to use streaming (disabled to avoid segmentation faults)

        Returns:
            DataFrame with unified data or None if loading fails
        """
        try:
            self.logger.info(f"🔄 Loading unified data for {symbol} on {exchange}")
            self.logger.info(f"   Timeframe: {timeframe}")

            # Calculate date range with progressive loading to avoid segmentation faults
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                if lookback_days:
                    # Use the full lookback_days parameter (180 days)
                    start_date = end_date - timedelta(days=lookback_days)
                else:
                    # Default to 180 days if no lookback_days specified
                    start_date = end_date - timedelta(days=180)

            self.logger.info(
                f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )
            self.logger.info(f"   Streaming: {use_streaming}")

            # Disable streaming to avoid segmentation faults
            use_streaming = False
            self.logger.info("   ⚠️ Streaming disabled to avoid segmentation faults")

            # Try to load data using the optimized pyarrow approach first
            base_path = f"data_cache/unified/{exchange.lower()}/{symbol}/{timeframe}/exchange={exchange.upper()}/symbol={symbol}/timeframe={timeframe}"

            if os.path.exists(base_path):
                # Check for timestamp issues (year=1970 indicates incorrect timestamps)
                year_1970_exists = os.path.exists(os.path.join(base_path, "year=1970"))
                if year_1970_exists:
                    self.logger.warning(
                        f"   ⚠️ Detected timestamp issue in {timeframe} data (year=1970), attempting self-healing..."
                    )
                    # Remove the problematic data and regenerate
                    try:
                        import shutil

                        backup_dir = f"{base_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        shutil.copytree(base_path, backup_dir)
                        shutil.rmtree(base_path)
                        self.logger.info(
                            f"   📦 Backed up and removed problematic {timeframe} data"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"   ❌ Failed to remove problematic data: {e}"
                        )

                self.logger.info(
                    "   📁 Found unified data cache, using optimized pyarrow loading"
                )
                unified_data = await self._load_with_optimized_pyarrow(
                    base_path, start_date, end_date
                )
                if unified_data is not None and not unified_data.empty:
                    return unified_data
                else:
                    self.logger.warning(
                        "   ⚠️ Unified data loading failed, trying fallback"
                    )
            else:
                self.logger.warning(
                    f"   ⚠️ Unified data cache not found at: {base_path}"
                )

                # Try to create unified data structure using step1_5
                self.logger.info(
                    "   🔄 Attempting to create unified data structure using step1_5..."
                )
                unified_created = await self._create_unified_data_structure(
                    symbol, exchange, timeframe
                )

                if unified_created:
                    self.logger.info(
                        "   ✅ Unified data structure created successfully, retrying load..."
                    )
                    # Retry loading the unified data
                    if os.path.exists(base_path):
                        unified_data = await self._load_with_optimized_pyarrow(
                            base_path, start_date, end_date
                        )
                        if unified_data is not None and not unified_data.empty:
                            return unified_data
                    else:
                        self.logger.warning(
                            "   ⚠️ Unified data structure creation failed"
                        )

            # Try to resample from 1m data if the requested timeframe is not available
            if timeframe != "1m":
                self.logger.info(
                    f"   🔄 Timeframe {timeframe} not found, attempting to resample from 1m data"
                )
                resampled_data = await self._resample_from_1m_data(
                    symbol, exchange, timeframe, start_date, end_date
                )
                if resampled_data is not None and not resampled_data.empty:
                    self.logger.info(
                        f"   ✅ Successfully resampled {timeframe} data from 1m source"
                    )

                    # Save resampled data to unified structure for future use
                    save_success = await self._save_resampled_data_to_unified(
                        resampled_data, symbol, exchange, timeframe
                    )
                    if save_success:
                        self.logger.info(
                            f"   💾 Saved resampled {timeframe} data to unified structure"
                        )

                    return resampled_data

            self.logger.warning(
                "   ⚠️ Unified data cache not found, trying fallback approaches"
            )

            # Try fallback to legacy data
            return await self._fallback_to_legacy_data(
                symbol, exchange, timeframe, start_date, end_date
            )

        except Exception as e:
            self.logger.error(f"❌ Unified data loading failed: {e}")
            return None

    async def _create_unified_data_structure(
        self, symbol: str, exchange: str, timeframe: str
    ) -> bool:
        """
        Create unified data structure using step1_5 data converter.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(
                f"🔄 Creating unified data structure for {symbol} on {exchange} ({timeframe})..."
            )

            # Import step1_5 data converter
            try:
                from src.training.steps.step1_5_data_converter import run_step
            except ImportError as e:
                self.logger.error(f"❌ Could not import step1_5_data_converter: {e}")
                return False

            # Run step1_5 to create unified data structure
            success = await run_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir="data_cache",
                force_rerun=False,  # Don't force rerun, just create if missing
            )

            if success:
                self.logger.info(
                    f"✅ Unified data structure created successfully for {symbol} on {exchange} ({timeframe})"
                )
                return True
            else:
                self.logger.warning(
                    f"⚠️ Failed to create unified data structure for {symbol} on {exchange} ({timeframe})"
                )
                return False

        except Exception as e:
            self.logger.error(f"❌ Error creating unified data structure: {e}")
            return False

    def _estimate_dataset_size(self, start_date: datetime, end_date: datetime) -> int:
        """Estimate dataset size based on date range."""
        days_diff = (end_date - start_date).days
        # Rough estimate: 1440 rows per day for 1m data
        return days_diff * 1440

    async def _load_with_optimized_pyarrow(
        self,
        base_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Load data using optimized pyarrow approach."""
        try:
            self.logger.info("🔧 Using optimized pyarrow loading...")

            # Disable pyarrow to avoid segmentation faults
            self.logger.info("   ⚠️ PyArrow disabled to avoid segmentation faults")

            # Use simple pandas approach instead
            return await self._load_with_simple_pandas(base_path, start_date, end_date)

        except Exception as e:
            self.logger.error(f"❌ Optimized pyarrow loading failed: {e}")
            return None

    async def _load_with_simple_pandas(
        self,
        base_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Load data using memory-efficient streaming approach for large datasets."""
        try:
            self.logger.info(
                "🔧 Using memory-efficient streaming loading for 180 days..."
            )

            # Get relevant partition paths
            partition_paths = self._get_partition_paths(base_path, start_date, end_date)

            if not partition_paths:
                self.logger.warning("⚠️ No partition paths found")
                return None

            self.logger.info(f"📁 Found {len(partition_paths)} partition files")

            # Process files one at a time to minimize memory usage
            all_data = []
            total_rows = 0

            for i, file_path in enumerate(partition_paths):
                try:
                    # Force garbage collection before each file
                    gc.collect()

                    # Only log every 10th file to reduce verbosity
                    if (i + 1) % 10 == 0 or i == 0:
                        self.logger.info(
                            f"   📄 Processing file {i+1}/{len(partition_paths)}: {os.path.basename(file_path)}"
                        )

                    # Read single file with minimal memory footprint
                    chunk_df = pd.read_parquet(file_path, engine="fastparquet")

                    if chunk_df is not None and not chunk_df.empty:
                        # Only log every 10th file to reduce verbosity
                        if (i + 1) % 10 == 0 or i == 0:
                            self.logger.info(
                                f"   ✅ Loaded file {i+1}: {chunk_df.shape}"
                            )

                        # Filter by date range immediately to reduce memory
                        if "timestamp" in chunk_df.columns:
                            chunk_df["timestamp"] = pd.to_datetime(
                                chunk_df["timestamp"], unit="ms"
                            )
                            mask = (chunk_df["timestamp"] >= start_date) & (
                                chunk_df["timestamp"] <= end_date
                            )
                            chunk_df = chunk_df[mask]

                            # Only log every 10th file to reduce verbosity
                            if (i + 1) % 10 == 0 or i == 0:
                                self.logger.info(
                                    f"   🔍 After date filtering: {chunk_df.shape}"
                                )

                        if not chunk_df.empty:
                            # Convert to list of dictionaries to save memory
                            chunk_data = chunk_df.to_dict("records")
                            all_data.extend(chunk_data)
                            total_rows += len(chunk_data)

                            # Only log every 10th file to reduce verbosity
                            if (i + 1) % 10 == 0 or i == 0:
                                self.logger.info(
                                    f"   📊 Total rows so far: {total_rows}"
                                )

                            # Clear the dataframe immediately
                            del chunk_df
                            gc.collect()
                    else:
                        self.logger.warning(f"   ⚠️ Empty chunk from file {i+1}")

                except Exception as e:
                    self.logger.error(f"   ❌ Failed to load chunk {i+1}: {e}")
                    continue

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)

            if not all_data:
                self.logger.error("❌ No valid data loaded")
                return None

            # Convert back to DataFrame only at the end
            self.logger.info(f"🔄 Converting {len(all_data)} records to DataFrame...")
            final_df = pd.DataFrame(all_data)

            # Clear the list to free memory
            all_data.clear()
            gc.collect()

            self.logger.info(f"✅ Memory-efficient loading completed: {final_df.shape}")
            return final_df

        except Exception as e:
            self.logger.error(f"❌ Memory-efficient loading failed: {e}")
            return None

    async def _load_with_streaming(
        self,
        base_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Load data using streaming approach to handle large datasets."""
        try:
            self.logger.info("🌊 Using streaming data loading...")

            # Get relevant partition paths
            partition_paths = self._get_partition_paths(base_path, start_date, end_date)

            if not partition_paths:
                self.logger.warning("⚠️ No partition paths found for streaming")
                return None

            self.logger.info(f"📁 Found {len(partition_paths)} partition files")

            # Use a simpler approach without streaming to avoid segmentation faults
            all_chunks = []

            for i, file_path in enumerate(partition_paths):
                try:
                    # Use fastparquet engine to avoid PyArrow segmentation faults
                    chunk_df = pd.read_parquet(file_path, engine="fastparquet")

                    if chunk_df is not None and not chunk_df.empty:
                        # Filter by date range
                        if "timestamp" in chunk_df.columns:
                            chunk_df["timestamp"] = pd.to_datetime(
                                chunk_df["timestamp"], unit="ms"
                            )
                            mask = (chunk_df["timestamp"] >= start_date) & (
                                chunk_df["timestamp"] <= end_date
                            )
                            chunk_df = chunk_df[mask]

                        if not chunk_df.empty:
                            all_chunks.append(chunk_df)
                            # Only log every 10th chunk to reduce verbosity
                            if (i + 1) % 10 == 0 or i == 0:
                                self.logger.info(
                                    f"   ✅ Loaded chunk {i+1}/{len(partition_paths)}: {chunk_df.shape}"
                                )

                        # Memory management
                        if len(all_chunks) % 10 == 0:
                            gc.collect()

                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to load chunk {i+1}: {e}")
                    continue

            if not all_chunks:
                self.logger.error("❌ No valid chunks loaded")
                return None

            # Combine all chunks
            combined_df = pd.concat(all_chunks, ignore_index=True)
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

            self.logger.info(f"✅ Streaming completed: {combined_df.shape}")
            return combined_df

        except Exception as e:
            self.logger.error(f"❌ Streaming failed: {e}")
            return None

    async def _load_with_optimized_pandas(
        self,
        base_path: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Load data using optimized pandas with targeted directory traversal."""
        try:
            # Get only relevant partition paths
            partition_paths = self._get_partition_paths(base_path, start_date, end_date)

            if not partition_paths:
                self.logger.warning("⚠️ No partition paths found for date range")
                return None

            all_data = []

            for i, file_path in enumerate(partition_paths):
                if not self._check_memory_usage():
                    self.logger.warning("⚠️ Memory threshold exceeded, stopping loading")
                    break

                try:
                    # Use parquet_utils for safe reading
                    df_chunk = pd.read_parquet(file_path, engine="fastparquet")

                    if df_chunk is not None and not df_chunk.empty:
                        # Filter by timestamp if needed
                        if "timestamp" in df_chunk.columns:
                            if not pd.api.types.is_datetime64_any_dtype(
                                df_chunk["timestamp"]
                            ):
                                df_chunk["timestamp"] = pd.to_datetime(
                                    df_chunk["timestamp"], unit="ms"
                                )

                            # Filter by date range
                            df_chunk = df_chunk[
                                (df_chunk["timestamp"] >= start_date)
                                & (df_chunk["timestamp"] <= end_date)
                            ]

                        if not df_chunk.empty:
                            all_data.append(df_chunk)

                    if (i + 1) % 10 == 0:
                        self.logger.info(
                            f"📊 Processed {i + 1}/{len(partition_paths)} files"
                        )

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")
                    continue

            if not all_data:
                self.logger.warning("⚠️ No data found for specified date range")
                return None

            # Combine all data
            df = pd.concat(all_data, ignore_index=True)

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Optimize memory usage
            df = self._optimize_dataframe_memory(df)

            self.logger.info(f"✅ Loaded {len(df)} rows using optimized pandas")
            return df

        except Exception as e:
            self.logger.error(f"❌ Optimized pandas loading failed: {e}")
            return None

    @with_tracing_span("UnifiedDataLoader._optimize_dataframe_memory", log_args=False)
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        try:
            # Optimize numeric columns
            for col in df.select_dtypes(include=["int64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="integer")

            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

            # Optimize object columns
            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype("category")

            return df
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return df

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the data loader."""
        try:
            memory_info = psutil.virtual_memory()
            return {
                "memory_usage": {
                    "total_gb": memory_info.total / (1024**3),
                    "available_gb": memory_info.available / (1024**3),
                    "used_gb": memory_info.used / (1024**3),
                    "percent": memory_info.percent,
                },
                "cache_stats": {
                    "cache_size": len(self._data_cache),
                    "max_cache_size": self.max_cache_size,
                    "cache_hit_ratio": getattr(self, "_cache_hits", 0)
                    / max(getattr(self, "_cache_requests", 1), 1),
                },
                "optimization_config": {
                    "memory_threshold": self.memory_threshold,
                    "chunk_size": self.chunk_size,
                    "streaming_enabled": True,
                    "pyarrow_available": PYARROW_AVAILABLE,
                },
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting performance metrics: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        gc.collect()
        self.logger.info("🧹 Data cache cleared")

    async def _resample_from_1m_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Resample 1m data to the requested timeframe."""
        try:
            self.logger.info(f"🔄 Resampling 1m data to {timeframe}...")

            # Load 1m data
            base_path_1m = f"data_cache/unified/{exchange.lower()}/{symbol}/1m/exchange={exchange.upper()}/symbol={symbol}/timeframe=1m"

            if not os.path.exists(base_path_1m):
                self.logger.warning("   ⚠️ 1m data not available for resampling")
                return None

            # Load 1m data
            data_1m = await self._load_with_optimized_pyarrow(
                base_path_1m, start_date, end_date
            )

            if data_1m is None or data_1m.empty:
                self.logger.warning("   ⚠️ No 1m data available for resampling")
                return None

            self.logger.info(
                f"   📊 Loaded {len(data_1m)} rows of 1m data for resampling"
            )

            # Ensure timestamp is the index
            if "timestamp" in data_1m.columns:
                data_1m = data_1m.set_index("timestamp")
            elif not isinstance(data_1m.index, pd.DatetimeIndex):
                self.logger.error("   ❌ No timestamp column found for resampling")
                return None

            # Sort by timestamp
            data_1m = data_1m.sort_index()

            # Define timeframe mapping
            timeframe_map = {
                "5m": "5T",
                "15m": "15T",
                "30m": "30T",
                "1h": "1H",
                "4h": "4H",
                "1d": "1D",
            }

            if timeframe not in timeframe_map:
                self.logger.error(
                    f"   ❌ Unsupported timeframe for resampling: {timeframe}"
                )
                return None

            resample_freq = timeframe_map[timeframe]

            # Resample OHLCV data
            resampled_data = pd.DataFrame()

            # OHLCV columns
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_ohlcv = [col for col in ohlcv_cols if col in data_1m.columns]

            if available_ohlcv:
                # Resample OHLCV
                if "open" in data_1m.columns:
                    resampled_data["open"] = (
                        data_1m["open"].resample(resample_freq).first()
                    )
                if "high" in data_1m.columns:
                    resampled_data["high"] = (
                        data_1m["high"].resample(resample_freq).max()
                    )
                if "low" in data_1m.columns:
                    resampled_data["low"] = data_1m["low"].resample(resample_freq).min()
                if "close" in data_1m.columns:
                    resampled_data["close"] = (
                        data_1m["close"].resample(resample_freq).last()
                    )
                if "volume" in data_1m.columns:
                    resampled_data["volume"] = (
                        data_1m["volume"].resample(resample_freq).sum()
                    )

            # Handle other numeric columns (sum or mean as appropriate)
            numeric_cols = data_1m.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in available_ohlcv:
                    # For most numeric features, use mean
                    resampled_data[col] = data_1m[col].resample(resample_freq).mean()

            # Handle categorical/string columns (take first value)
            categorical_cols = data_1m.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_cols:
                resampled_data[col] = data_1m[col].resample(resample_freq).first()

            # Drop rows with all NaN values
            resampled_data = resampled_data.dropna(how="all")

            # Reset index to make timestamp a column
            resampled_data = resampled_data.reset_index()

            # Convert timestamp to milliseconds for proper partitioning
            if "timestamp" in resampled_data.columns:
                resampled_data["timestamp"] = (
                    resampled_data["timestamp"].astype(np.int64) // 10**6
                )

            self.logger.info(
                f"   ✅ Resampled to {timeframe}: {len(resampled_data)} rows"
            )
            return resampled_data

        except Exception as e:
            self.logger.error(f"   ❌ Error resampling data: {e}")
            return None

    async def _save_resampled_data_to_unified(
        self,
        resampled_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> bool:
        """Save resampled data to unified structure for future use."""
        try:
            self.logger.info(
                f"   💾 Saving resampled {timeframe} data to unified structure..."
            )

            # Import step1_5 data converter for saving
            try:
                from src.training.steps.step1_5_data_converter import (
                    ParquetDatasetManager,
                )
            except ImportError as e:
                self.logger.error(f"   ❌ Could not import step1_5_data_converter: {e}")
                return False

            # Create parquet dataset manager instance
            converter = ParquetDatasetManager(logger=self.logger)

            # Prepare the data for saving
            # Ensure timestamp is in milliseconds for proper partitioning
            if "timestamp" in resampled_data.columns:
                if pd.api.types.is_datetime64_any_dtype(resampled_data["timestamp"]):
                    resampled_data["timestamp"] = (
                        resampled_data["timestamp"].astype(np.int64) // 10**6
                    )

            # Add required columns for unified structure
            if "exchange" not in resampled_data.columns:
                resampled_data["exchange"] = exchange.upper()
            if "symbol" not in resampled_data.columns:
                resampled_data["symbol"] = symbol
            if "timeframe" not in resampled_data.columns:
                resampled_data["timeframe"] = timeframe

            # Save to unified structure
            base_dir = f"data_cache/unified/{exchange.lower()}/{symbol}/{timeframe}/exchange={exchange.upper()}/symbol={symbol}/timeframe={timeframe}"

            # Create directory if it doesn't exist
            os.makedirs(base_dir, exist_ok=True)

            # Save using the converter's write_partitioned_dataset method
            try:
                converter.write_partitioned_dataset(
                    df=resampled_data,
                    base_dir=base_dir,
                    partition_cols=["year", "month", "day"],
                    schema_name=None,
                    compression="snappy",
                    max_rows_per_file=100000,
                    auto_add_date_columns=True,
                    metadata={
                        "source": "resampled_from_1m",
                        "timeframe": timeframe,
                        "symbol": symbol,
                        "exchange": exchange,
                        "created_at": datetime.now().isoformat(),
                    },
                )

                # Verify the data was saved by checking if the directory exists and has files
                if os.path.exists(base_dir):
                    # Check for year directories
                    year_dirs = [
                        d for d in os.listdir(base_dir) if d.startswith("year=")
                    ]
                    if year_dirs and "year=1970" not in year_dirs:
                        self.logger.info(
                            f"   ✅ Successfully saved resampled {timeframe} data to unified structure"
                        )
                        return True
                    else:
                        self.logger.warning(
                            f"   ⚠️ Saved data has incorrect timestamps (year=1970)"
                        )
                        return False
                else:
                    self.logger.warning(
                        f"   ⚠️ Failed to save resampled {timeframe} data to unified structure"
                    )
                    return False

            except Exception as e:
                self.logger.error(f"   ❌ Error saving resampled data: {e}")
                return False

        except Exception as e:
            self.logger.error(
                f"   ❌ Error saving resampled data to unified structure: {e}"
            )
            return False

    async def _fallback_to_legacy_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fallback to legacy data sources if unified data is not available."""
        self.logger.info("🔄 Falling back to legacy data sources...")

        # First, try individual parquet files in data_cache directory
        try:
            import glob
            import os

            # Look for individual parquet files in data_cache directory
            pattern = f"data_cache/aggtrades_{exchange}_{symbol}_*.parquet"
            parquet_files = glob.glob(pattern)

            if parquet_files:
                # Sort files by date (newest first)
                parquet_files.sort(reverse=True)

                # Calculate how many files we need for the date range
                days_diff = (end_date - start_date).days
                max_files = min(days_diff + 10, 180)  # Add buffer, but cap at 180 files

                # Limit files for blank mode
                if os.environ.get("BLANK_TRAINING_MODE", "0") == "1":
                    parquet_files = parquet_files[:max_files]

                self.logger.info(
                    f"🔄 Found {len(parquet_files)} individual parquet files, loading up to {max_files}..."
                )

                # Load files one by one
                all_data = []
                total_rows = 0

                for i, file_path in enumerate(parquet_files[:max_files]):
                    try:
                        if i % 10 == 0:
                            self.logger.info(
                                f"📂 Loading file {i+1}/{len(parquet_files[:max_files])}: {os.path.basename(file_path)}"
                            )

                        # Load individual file
                        df = pd.read_parquet(file_path)

                        if not df.empty:
                            # Filter by date range, but be flexible if no data found
                            original_df = df.copy()
                            df_filtered = df[
                                (df["timestamp"] >= start_date)
                                & (df["timestamp"] <= end_date)
                            ]

                            if not df_filtered.empty:
                                all_data.append(df_filtered)
                                total_rows += len(df_filtered)
                                self.logger.info(
                                    f"✅ Loaded {len(df_filtered):,} rows from {os.path.basename(file_path)}"
                                )
                            else:
                                # If no data in requested range, use all available data
                                self.logger.warning(
                                    f"⚠️ No data in requested date range for {os.path.basename(file_path)}, using all available data"
                                )
                                all_data.append(original_df)
                                total_rows += len(original_df)
                                self.logger.info(
                                    f"✅ Loaded {len(original_df):,} rows from {os.path.basename(file_path)} (all available)"
                                )

                    except Exception as e:
                        self.logger.warning(f"❌ Failed to load {file_path}: {e}")
                        continue

                if all_data:
                    # Combine all data
                    combined_df = pd.concat(all_data, ignore_index=True)
                    combined_df = combined_df.sort_values("timestamp").reset_index(
                        drop=True
                    )

                    self.logger.info(
                        f"✅ Successfully loaded {len(combined_df):,} total rows from {len(all_data)} files"
                    )
                    return combined_df

        except Exception as e:
            self.logger.warning(f"⚠️ Individual parquet approach failed: {e}")

        # Try consolidated parquet files as fallback
        consolidated_paths = [
            f"data_cache/klines_{exchange}_{symbol}_1m_consolidated.csv",
            f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"data_cache/{exchange}_{symbol}_historical_data.pkl",
        ]

        for path in consolidated_paths:
            if os.path.exists(path):
                try:
                    if path.endswith(".csv"):
                        df = pd.read_csv(path)
                    elif path.endswith(".parquet"):
                        # Use parquet_utils for safe reading
                        df = pd.read_parquet(path, engine="fastparquet")
                    elif path.endswith(".pkl"):
                        import pickle

                        with open(path, "rb") as f:
                            payload = pickle.load(f)
                        if isinstance(payload, dict):
                            df = payload.get("klines") or next(
                                (
                                    v
                                    for v in payload.values()
                                    if isinstance(v, pd.DataFrame)
                                ),
                                None,
                            )
                        else:
                            df = payload if isinstance(payload, pd.DataFrame) else None

                    if df is not None and not df.empty:
                        # Convert timestamp if needed
                        if (
                            "timestamp" in df.columns
                            and not pd.api.types.is_datetime64_any_dtype(
                                df["timestamp"]
                            )
                        ):
                            if df["timestamp"].iloc[0] > 1e12:  # milliseconds
                                df["timestamp"] = pd.to_datetime(
                                    df["timestamp"], unit="ms"
                                )
                            else:
                                df["timestamp"] = pd.to_datetime(df["timestamp"])

                        # Filter by date range
                        df = df[
                            (df["timestamp"] >= start_date)
                            & (df["timestamp"] <= end_date)
                        ]

                        if not df.empty:
                            self.logger.info(
                                f"✅ Loaded {len(df)} rows from legacy source: {path}"
                            )
                            return df.sort_values("timestamp").reset_index(drop=True)

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load legacy source {path}: {e}")

        self.logger.error("❌ No usable data sources found")
        return None

    @guard_dataframe_nulls(mode="warn", arg_index=1)
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get information about the loaded data."""
        if df is None or df.empty:
            return {"status": "empty"}

        info = {
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df["timestamp"].min().isoformat()
                if "timestamp" in df.columns
                else None,
                "end": df["timestamp"].max().isoformat()
                if "timestamp" in df.columns
                else None,
            },
            "has_aggtrades_data": any(
                col in df.columns
                for col in ["trade_volume", "trade_count", "avg_price"]
            ),
            "has_futures_data": "funding_rate" in df.columns,
            "data_types": df.dtypes.to_dict(),
        }

        return info


# Global instance for easy access
_unified_data_loader = None


def get_unified_data_loader(config: dict[str, Any]) -> UnifiedDataLoader:
    """Get or create a global unified data loader instance."""
    global _unified_data_loader
    if _unified_data_loader is None:
        _unified_data_loader = UnifiedDataLoader(config)
    return _unified_data_loader
