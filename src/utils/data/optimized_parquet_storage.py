"""
Optimized Parquet Storage with Hardware Optimizations

This module provides optimized parquet storage capabilities using hardware-specific
optimizations for memory efficiency and performance.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from src.utils.logger import system_logger
from src.utils.parquet_utils import get_parquet_utils
from src.utils.hardware.memory_optimization import MemoryMonitor, optimize_dataframe_dtypes
from src.utils.hardware import (
    get_integrated_hardware_manager, 
    get_comprehensive_optimizer,
    memory_optimized, 
    comprehensive_memory_optimization,
    optimize_dataframe, 
    optimize_array,
    m1_optimized,
    WorkloadCategory,
    MemoryOptimizationLevel
)

class OptimizedParquetStorage:
    """Optimized parquet storage with hardware-specific optimizations."""

    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the optimized parquet storage.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "binance"
        self.processed_data_dir = self.data_dir / "binance"
        self.logger = system_logger.getChild("OptimizedParquetStorage")
        self.parquet_utils = get_parquet_utils()
        self.memory_monitor = MemoryMonitor()
        self.m1_optimizer = get_integrated_hardware_manager()
        self.m1_data_manager = M1DataManager(self.m1_optimizer)

        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def save_optimized_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        data_type: str = "raw",
        compression: str = "snappy",
        partition_by: List[str] = None
    ) -> bool:
        """Save data with hardware-optimized parquet storage.

        Args:
            df: DataFrame to save
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'
            compression: Compression algorithm
            partition_by: Columns to partition by

        Returns:
            True if successful, False otherwise
        """
        try:
            if df is None or len(df) == 0:
                self.logger.warning("Cannot save empty DataFrame")
                return False

            # Create directory structure
            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"

            data_dir.mkdir(parents=True, exist_ok=True)

            # Optimize DataFrame before saving
            optimized_df = self._optimize_dataframe_for_storage(df)

            # Add partitioning columns if specified
            if partition_by:
                optimized_df = self._add_partitioning_columns(optimized_df, partition_by)

            # Determine file path
            if data_type == "raw":
                # For raw data, save as monthly files
                filename = f"{symbol.lower()}_{interval}_{datetime.now().strftime('%Y_%m')}.parquet"
                filepath = data_dir / filename

                # Append to existing file if it exists
                if filepath.exists():
                    existing_df = self.parquet_utils.safe_read_parquet(str(filepath))
                    if existing_df is not None:
                        optimized_df = self._merge_dataframes(existing_df, optimized_df)

                # Save with M1 optimization
                with self.m1_optimizer.memory_checkpoint(f"save_raw_{symbol}_{interval}"):
                    optimized_df.to_parquet(
                        filepath,
                        index=True,
                        compression=compression,
                        engine='pyarrow'
                    )
            else:
                # For processed data, save as partitioned parquet
                output_path = data_dir / f"{symbol.lower()}_{interval}"

                # Use M1 data manager for efficient saving
                self.m1_data_manager.save_data_efficiently(
                    optimized_df,
                    str(output_path),
                    format='parquet',
                    compression=compression
                )

            self.logger.info(f"💾 Saved optimized data: {symbol} {interval} ({len(optimized_df)} records)")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to save optimized data: {e}")
            return False

    def load_optimized_data(
        self,
        symbol: str,
        interval: str,
        data_type: str = "raw",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Load data with hardware-optimized parquet reading.

        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'
            start_date: Start date for filtering
            end_date: End date for filtering
            columns: List of columns to read

        Returns:
            DataFrame with data or None if not found
        """
        try:
            # Create directory structure
            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
                pattern = f"{symbol.lower()}_{interval}_*.parquet"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"
                pattern = f"{symbol.lower()}_{interval}"

            if not data_dir.exists():
                self.logger.warning(f"No data directory found for {symbol} {interval}")
                return None

            # Find matching files
            files = list(data_dir.glob(f"{pattern}*"))

            if not files:
                self.logger.warning(f"No files found for {symbol} {interval}")
                return None

            # Load and combine data with M1 optimization
            with self.m1_optimizer.memory_checkpoint(f"load_{symbol}_{interval}"):
                dataframes = []

                for file_path in sorted(files):
                    try:
                        if data_type == "processed" and file_path.is_dir():
                            # For processed data, it might be partitioned
                            parquet_files = list(file_path.glob("*.parquet"))
                            for pf in sorted(parquet_files):
                                df = self.m1_data_manager.load_data_efficiently(
                                    str(pf), columns=columns
                                )
                                if df is not None and not len(df) == 0:
                                    dataframes.append(df)
                        else:
                            # For raw data or single files
                            df = self.m1_data_manager.load_data_efficiently(
                                str(file_path), columns=columns
                            )
                            if df is not None and not len(df) == 0:
                                dataframes.append(df)
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")

                if not dataframes:
                    self.logger.warning(f"No valid data found for {symbol} {interval}")
                    return None

                # Combine all dataframes efficiently
                combined_df = self.m1_optimizer.memory_efficient_concat(dataframes)
                combined_df = combined_df.sort_index()

                # Remove duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

                # Apply date filtering if specified
                if start_date is not None:
                    combined_df = combined_df[combined_df.index >= start_date]

                if end_date is not None:
                    combined_df = combined_df[combined_df.index <= end_date]

                self.logger.info(f"📊 Loaded optimized data: {symbol} {interval} ({len(combined_df)} records)")
                return combined_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to load optimized data: {e}")
            return None

    def _optimize_dataframe_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for efficient storage.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        try:
            # Use M1 optimizer for memory efficiency
            optimized_df = df.copy()

            # Optimize data types
            optimized_df = optimize_dataframe_dtypes(optimized_df)

            # Additional M1-specific optimizations
            optimized_df = self.m1_optimizer.create_memory_efficient_array(
                optimized_df, dtype=np.float32
            )
            optimized_df = pd.DataFrame(optimized_df, index=df.index, columns=df.columns)

            # Convert object columns to category if beneficial
            for col in optimized_df.select_dtypes(include=['object']):
                if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')

            return optimized_df

        except Exception as e:
            self.logger.exception(f"❌ DataFrame optimization failed: {e}")
            return df

    def _add_partitioning_columns(self, df: pd.DataFrame, partition_by: List[str]) -> pd.DataFrame:
        """Add partitioning columns to DataFrame.

        Args:
            df: Input DataFrame
            partition_by: List of columns to partition by

        Returns:
            DataFrame with partitioning columns
        """
        try:
            df_with_partitions = df.copy()

            for col in partition_by:
                if col == 'year':
                    df_with_partitions['year'] = df_with_partitions.index.year
                elif col == 'month':
                    df_with_partitions['month'] = df_with_partitions.index.month
                elif col == 'day':
                    df_with_partitions['day'] = df_with_partitions.index.day
                elif col == 'hour':
                    df_with_partitions['hour'] = df_with_partitions.index.hour
                elif col == 'day_of_week':
                    df_with_partitions['day_of_week'] = df_with_partitions.index.dayofweek
                elif col == 'is_weekend':
                    df_with_partitions['is_weekend'] = df_with_partitions.index.dayofweek.isin([5, 6]).astype(int)

            return df_with_partitions

        except Exception as e:
            self.logger.exception(f"❌ Failed to add partitioning columns: {e}")
            return df

    def _merge_dataframes(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge existing and new DataFrames efficiently.

        Args:
            existing_df: Existing DataFrame
            new_df: New DataFrame to merge

        Returns:
            Merged DataFrame
        """
        try:
            # Combine DataFrames
            combined_df = pd.concat([existing_df, new_df], ignore_index=False)
            combined_df = combined_df.sort_index()

            # Remove duplicates (keep last occurrence)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            return combined_df

        except Exception as e:
            self.logger.exception(f"❌ DataFrame merge failed: {e}")
            return new_df

    def get_storage_info(self, symbol: str, interval: str, data_type: str = "raw") -> Dict[str, Any]:
        """Get storage information for data.

        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'

        Returns:
            Dictionary with storage information
        """
        try:
            # Create directory structure
            if data_type == "raw":
                data_dir = self.raw_data_dir / symbol.lower() / "raw"
                pattern = f"{symbol.lower()}_{interval}_*.parquet"
            else:
                data_dir = self.processed_data_dir / symbol.lower() / "processed"
                pattern = f"{symbol.lower()}_{interval}"

            if not data_dir.exists():
                return {
                    "available": False,
                    "files_count": 0,
                    "total_size_mb": 0,
                    "total_records": 0,
                    "date_range": None
                }

            # Find matching files
            files = list(data_dir.glob(f"{pattern}*"))

            if not files:
                return {
                    "available": False,
                    "files_count": 0,
                    "total_size_mb": 0,
                    "total_records": 0,
                    "date_range": None
                }

            # Calculate total size
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            # Get record count and date range
            total_records = 0
            date_ranges = []

            for file_path in files:
                try:
                    if data_type == "processed" and file_path.is_dir():
                        # For processed data, it might be partitioned
                        parquet_files = list(file_path.glob("*.parquet"))
                        for pf in parquet_files:
                            df = self.parquet_utils.safe_read_parquet(str(pf))
                            if df is not None and not len(df) == 0:
                                total_records += len(df)
                                date_ranges.append((df.index.min(), df.index.max()))
                    else:
                        # For raw data or single files
                        df = self.parquet_utils.safe_read_parquet(str(file_path))
                        if df is not None and not len(df) == 0:
                            total_records += len(df)
                            date_ranges.append((df.index.min(), df.index.max()))
                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            info = {
                "available": True,
                "files_count": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "total_records": total_records,
                "date_range": None
            }

            if date_ranges:
                min_date = min(dt[0] for dt in date_ranges)
                max_date = max(dt[1] for dt in date_ranges)
                info["date_range"] = (min_date, max_date)

            return info

        except Exception as e:
            self.logger.exception(f"❌ Failed to get storage info: {e}")
            return {
                "available": False,
                "files_count": 0,
                "total_size_mb": 0,
                "total_records": 0,
                "date_range": None,
                "error": str(e)
            }

    def optimize_existing_data(self, symbol: str, interval: str, data_type: str = "raw") -> bool:
        """Optimize existing data files for better performance.

        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: 'raw' or 'processed'

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"🔧 Optimizing existing data for {symbol} {interval}")

            # Load data
            data = self.load_optimized_data(symbol, interval, data_type)

            if data is None or len(data) == 0:
                self.logger.warning(f"No data found to optimize for {symbol} {interval}")
                return False

            # Optimize and save
            success = self.save_optimized_data(data, symbol, interval, data_type)

            if success:
                self.logger.info(f"✅ Data optimization completed for {symbol} {interval}")
            else:
                self.logger.error(f"❌ Data optimization failed for {symbol} {interval}")

            return success

        except Exception as e:
            self.logger.exception(f"❌ Data optimization failed: {e}")
            return False

# Convenience functions
def get_optimized_storage(data_dir: str = "historical_data") -> OptimizedParquetStorage:
    """Get an optimized parquet storage instance.

    Args:
        data_dir: Base directory for data storage

    Returns:
        OptimizedParquetStorage instance
    """
    return OptimizedParquetStorage(data_dir)

def save_optimized_ethusdt_data(
    df: pd.DataFrame,
    interval: str = "1m",
    data_type: str = "raw",
    data_dir: str = "historical_data"
) -> bool:
    """Save ETHUSDT data with optimized storage.

    Args:
        df: DataFrame to save
        interval: Data interval
        data_type: 'raw' or 'processed'
        data_dir: Base directory for data storage

    Returns:
        True if successful, False otherwise
    """
    storage = get_optimized_storage(data_dir)
    return storage.save_optimized_data(df, "ETHUSDT", interval, data_type)

def load_optimized_ethusdt_data(
    interval: str = "1m",
    data_type: str = "raw",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_dir: str = "historical_data"
) -> Optional[pd.DataFrame]:
    """Load ETHUSDT data with optimized storage.

    Args:
        interval: Data interval
        data_type: 'raw' or 'processed'
        start_date: Start date for filtering
        end_date: End date for filtering
        data_dir: Base directory for data storage

    Returns:
        DataFrame with data or None if not found
    """
    storage = get_optimized_storage(data_dir)
    return storage.load_optimized_data("ETHUSDT", interval, data_type, start_date, end_date)

if __name__ == "__main__":
    # Example usage
    storage = get_optimized_storage()

    # Get storage info
    info = storage.get_storage_info("ETHUSDT", "1m", "raw")
    print(f"Storage info: {info}")

    # Load data
    data = storage.load_optimized_data("ETHUSDT", "1m", "raw")
    if data is not None:
        print(f"Loaded {len(data)} records")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
