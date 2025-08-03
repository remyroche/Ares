# src/training/data_efficiency_optimizer.py

import gc
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import system_logger


class DataEfficiencyOptimizer:
    """
    Comprehensive data efficiency optimizer for handling large datasets (2+ years of historical data).

    Implements multiple strategies:
    1. Intelligent caching with SQLite storage
    2. Time-based data segmentation
    3. Memory-efficient data loading
    4. Progressive data processing
    5. Database-backed feature storage
    6. Checkpoint and resume capabilities
    """

    def __init__(self, db_manager: SQLiteManager, symbol: str, timeframe: str, exchange: str = "BINANCE"):
        self.db_manager = db_manager
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = exchange
        self.logger = system_logger.getChild("DataEfficiencyOptimizer")

        # Cache configuration
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Database configuration
        self.db_path = f"data_cache/{exchange}_{symbol}_{timeframe}_efficiency.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)

        # Memory management
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.chunk_size = 10000  # Default chunk size for processing

        # Initialize database tables
        self._init_database()

        self.logger.info(
            f"DataEfficiencyOptimizer initialized for {exchange} {symbol} {timeframe}",
        )

    def _init_database(self):
        """Initialize SQLite database with optimized tables for large datasets."""
        with self.engine.connect() as conn:
            # Raw data table with partitioning by date
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    data_type TEXT NOT NULL,  -- 'klines', 'agg_trades', 'futures'
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """),
            )

            # Create indexes for efficient querying
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp 
                ON raw_data(timestamp)
            """),
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_raw_data_type 
                ON raw_data(data_type)
            """),
            )

            # Feature cache table
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS feature_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    feature_type TEXT,  -- 'technical', 'price', 'volume', 'regime'
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """),
            )

            # Create indexes for feature cache
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_feature_cache_timestamp 
                ON feature_cache(timestamp)
            """),
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_feature_cache_name 
                ON feature_cache(feature_name)
            """),
            )

            # Processing checkpoints
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS processing_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    status TEXT NOT NULL,  -- 'completed', 'in_progress', 'failed'
                    metadata TEXT,  -- JSON string with additional info
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """),
            )

            conn.commit()

    def get_memory_usage(self) -> float:
        """Get current memory usage as a percentage."""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        self.logger.debug(f"Current memory usage: {memory_percent:.2f}%")
        return memory_percent

    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed."""
        return self.get_memory_usage() > (self.memory_threshold * 100)

    def cleanup_memory(self):
        """Force garbage collection and memory cleanup."""
        self.logger.info("Performing memory cleanup...")
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup
        self.logger.info(f"Memory usage after cleanup: {self.get_memory_usage():.2f}%")

    async def load_data_with_caching(
        self,
        lookback_days: int,
        force_reload: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Load data with intelligent caching and memory management.

        Args:
            lookback_days: Number of days to look back
            force_reload: Force reload from source, bypassing cache

        Returns:
            Dictionary containing klines, agg_trades, and futures DataFrames
        """
        cache_key = f"{self.exchange}_{self.symbol}_{self.timeframe}_{lookback_days}"
        cache_file = self.cache_dir / f"{cache_key}_cached_data.pkl"

        # Check if cache exists and is valid
        if not force_reload and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            max_cache_age = 24 * 60 * 60  # 24 hours

            if cache_age < max_cache_age:
                self.logger.info(f"Loading data from cache: {cache_file}")
                try:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)

                    # Validate cache data
                    if self._validate_cached_data(data, lookback_days):
                        self.logger.info("Cache validation successful")
                        return data
                    self.logger.warning("Cache validation failed, reloading data")
                except Exception as e:
                    self.logger.warning(f"Cache loading failed: {e}")

        # Load data from source with memory management
        self.logger.info(f"Loading {lookback_days} days of data from source...")
        data = await self._load_data_from_source(lookback_days)

        # Cache the loaded data
        self._cache_data(data, cache_file)

        return data

    def _validate_cached_data(
        self,
        data: dict[str, pd.DataFrame],
        lookback_days: int,
    ) -> bool:
        """Validate cached data integrity and completeness."""
        try:
            # Check if all required data types are present
            required_types = ["klines", "agg_trades", "futures"]
            if not all(dtype in data for dtype in required_types):
                return False

            # Check data completeness
            cutoff_date = datetime.now() - timedelta(days=lookback_days)

            for dtype, df in data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return False

                # Check if data covers the required time period
                if "timestamp" in df.columns:
                    min_date = pd.to_datetime(df["timestamp"].min())
                    if min_date > cutoff_date:
                        return False

                # Check for reasonable data size
                if len(df) < 100:  # Minimum reasonable data points
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"Cache validation error: {e}")
            return False

    async def _load_data_from_source(
        self,
        lookback_days: int,
    ) -> dict[str, pd.DataFrame]:
        """Load data from source with memory-efficient processing."""
        self.logger.info(f"Loading data from source for {lookback_days} days...")

        # Try to load from existing pickle file first
        data_file = f"data/{self.exchange}_{self.symbol}_historical_data.pkl"
        if os.path.exists(data_file):
            self.logger.info(f"Loading data from existing file: {data_file}")
            try:
                with open(data_file, "rb") as f:
                    data = pickle.load(f)

                # Validate that we have the expected data structure
                if (
                    isinstance(data, dict)
                    and "klines" in data
                    and isinstance(data["klines"], pd.DataFrame)
                    and not data["klines"].empty
                ):
                    self.logger.info(
                        f"Successfully loaded {len(data['klines'])} klines records",
                    )
                    return data
                self.logger.warning("Existing data file has invalid structure")
            except Exception as e:
                self.logger.warning(f"Failed to load existing data file: {e}")

        # Fallback: return empty structure if no data available
        self.logger.warning("No valid data source found, returning empty structure")
        data = {
            "klines": pd.DataFrame(),
            "agg_trades": pd.DataFrame(),
            "futures": pd.DataFrame(),
        }

        return data

    def _cache_data(self, data: dict[str, pd.DataFrame], cache_file: Path):
        """Cache data to disk with compression."""
        try:
            self.logger.info(f"Caching data to {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info("Data cached successfully")
        except Exception as e:
            self.logger.error(f"Failed to cache data: {e}")

    def segment_data_by_time(
        self,
        data: pd.DataFrame,
        segment_days: int = 30,
    ) -> list[tuple[datetime, datetime, pd.DataFrame]]:
        """
        Segment large datasets by time periods for efficient processing.

        Args:
            data: Input DataFrame with timestamp index
            segment_days: Number of days per segment

        Returns:
            List of tuples: (start_date, end_date, segment_data)
        """
        if data.empty:
            return []

        # Ensure timestamp index
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")

        # Sort by timestamp
        data = data.sort_index()

        segments = []
        start_date = data.index.min()
        end_date = data.index.max()

        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=segment_days), end_date)

            # Extract segment data
            segment_mask = (data.index >= current_start) & (data.index <= current_end)
            segment_data = data[segment_mask].copy()

            if not segment_data.empty:
                segments.append((current_start, current_end, segment_data))

            current_start = current_end + timedelta(seconds=1)

        self.logger.info(f"Created {len(segments)} time segments")
        return segments

    def process_data_in_chunks(
        self,
        data: pd.DataFrame,
        chunk_size: int | None = None,
    ) -> pd.DataFrame:
        """
        Process large datasets in memory-efficient chunks.

        Args:
            data: Input DataFrame
            chunk_size: Size of each chunk (defaults to self.chunk_size)

        Returns:
            Processed DataFrame
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        if len(data) <= chunk_size:
            return self._process_chunk(data)

        processed_chunks = []
        total_chunks = (len(data) + chunk_size - 1) // chunk_size

        self.logger.info(f"Processing {len(data)} rows in {total_chunks} chunks")

        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i : i + chunk_size]
            processed_chunk = self._process_chunk(chunk)
            processed_chunks.append(processed_chunk)

            # Memory management
            if self.should_cleanup_memory():
                self.cleanup_memory()

            # Progress logging
            if (i // chunk_size + 1) % 10 == 0:
                self.logger.info(
                    f"Processed {i // chunk_size + 1}/{total_chunks} chunks",
                )

        # Combine processed chunks
        result = pd.concat(processed_chunks, ignore_index=True)
        self.logger.info(f"Completed processing {len(result)} rows")

        return result

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data with data validation and repair."""
        if chunk.empty:
            return chunk

        # Create a copy to avoid modifying the original
        processed_chunk = chunk.copy()

        # Fix timestamp columns if they're missing
        if (
            "open_time" in processed_chunk.columns
            and processed_chunk["open_time"].isna().all()
        ):
            # If open_time is all NaN, try to reconstruct from the index
            if processed_chunk.index.name == "timestamp":
                processed_chunk["open_time"] = processed_chunk.index
            else:
                # Try to convert timestamp index to open_time
                try:
                    processed_chunk["open_time"] = pd.to_datetime(processed_chunk.index)
                except:
                    self.logger.warning("Could not reconstruct open_time column")

        # Fix close_time if it's missing
        if (
            "close_time" in processed_chunk.columns
            and processed_chunk["close_time"].isna().all()
        ):
            if (
                "open_time" in processed_chunk.columns
                and not processed_chunk["open_time"].isna().all()
            ):
                # Set close_time to open_time + 1 hour (for 1h timeframe)
                processed_chunk["close_time"] = processed_chunk[
                    "open_time"
                ] + pd.Timedelta(hours=1)

        # Fix other missing columns with reasonable defaults
        numeric_columns = [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        for col in numeric_columns:
            if col in processed_chunk.columns and processed_chunk[col].isna().all():
                # Set to 0 for missing numeric values
                processed_chunk[col] = 0

        # Fix ignore column
        if (
            "ignore" in processed_chunk.columns
            and processed_chunk["ignore"].isna().all()
        ):
            processed_chunk["ignore"] = 0

        return processed_chunk

    def store_features_in_database(
        self,
        features: pd.DataFrame,
        feature_type: str = "technical",
    ):
        """
        Store computed features in SQLite database for efficient retrieval.

        Args:
            features: DataFrame with features (timestamp index + feature columns)
            feature_type: Type of features ('technical', 'price', 'volume', 'regime')
        """
        if features.empty:
            return

        # Ensure timestamp index
        if "timestamp" in features.columns:
            features = features.set_index("timestamp")

        self.logger.info(f"Storing {len(features.columns)} features in database")

        with self.Session() as session:
            # Prepare data for database insertion
            records = []
            for timestamp, row in features.iterrows():
                for feature_name, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        # Convert timestamp to string for SQLite compatibility
                        timestamp_str = (
                            timestamp.isoformat()
                            if hasattr(timestamp, "isoformat")
                            else str(timestamp)
                        )
                        records.append(
                            {
                                "timestamp": timestamp_str,
                                "feature_name": feature_name,
                                "feature_value": float(value),
                                "feature_type": feature_type,
                            },
                        )

            # Batch insert for efficiency
            if records:
                session.execute(
                    text("""
                    INSERT INTO feature_cache (timestamp, feature_name, feature_value, feature_type)
                    VALUES (:timestamp, :feature_name, :feature_value, :feature_type)
                """),
                    records,
                )
                session.commit()

        self.logger.info(f"Stored {len(records)} feature records")

    def load_features_from_database(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load features from database for a specific time period.

        Args:
            start_date: Start of time period
            end_date: End of time period
            feature_names: Specific features to load (None for all)

        Returns:
            DataFrame with features
        """
        with self.Session() as session:
            query = text("""
                SELECT timestamp, feature_name, feature_value
                FROM feature_cache
                WHERE timestamp BETWEEN :start_date AND :end_date
            """)

            # Convert pandas Timestamps to strings for SQLite compatibility
            start_date_str = (
                start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date)
            )
            end_date_str = (
                end_date.isoformat()
                if hasattr(end_date, "isoformat")
                else str(end_date)
            )
            params = {"start_date": start_date_str, "end_date": end_date_str}

            if feature_names:
                query = text("""
                    SELECT timestamp, feature_name, feature_value
                    FROM feature_cache
                    WHERE timestamp BETWEEN :start_date AND :end_date
                    AND feature_name IN :feature_names
                """)
                params["feature_names"] = tuple(feature_names)

            result = session.execute(query, params)
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=["timestamp", "feature_name", "feature_value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Pivot to wide format
        features_df = df.pivot_table(
            index="timestamp",
            columns="feature_name",
            values="feature_value",
        )

        self.logger.info(f"Loaded {len(features_df)} feature records")
        return features_df

    def create_processing_checkpoint(
        self,
        checkpoint_name: str,
        metadata: dict[str, Any],
    ):
        """Create a processing checkpoint for resume capability."""
        with self.Session() as session:
            session.execute(
                text("""
                INSERT INTO processing_checkpoints (checkpoint_name, timestamp, status, metadata)
                VALUES (:checkpoint_name, :timestamp, 'completed', :metadata)
            """),
                {
                    "checkpoint_name": checkpoint_name,
                    "timestamp": datetime.now(),
                    "metadata": str(metadata),
                },
            )
            session.commit()

        self.logger.info(f"Created checkpoint: {checkpoint_name}")

    def get_latest_checkpoint(self, checkpoint_name: str) -> dict[str, Any] | None:
        """Get the latest checkpoint for resume capability."""
        with self.Session() as session:
            result = session.execute(
                text("""
                SELECT timestamp, metadata
                FROM processing_checkpoints
                WHERE checkpoint_name = :checkpoint_name
                ORDER BY timestamp DESC
                LIMIT 1
            """),
                {"checkpoint_name": checkpoint_name},
            )

            row = result.fetchone()
            if row:
                return {"timestamp": row[0], "metadata": eval(row[1]) if row[1] else {}}

        return None

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.

        Args:
            df: Input DataFrame

        Returns:
            Memory-optimized DataFrame
        """
        # Check if input is actually a DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.warning(
                f"Expected DataFrame but got {type(df).__name__}, returning as-is",
            )
            return df

        # Check if DataFrame is empty
        if df.empty:
            self.logger.info("DataFrame is empty, skipping memory optimization")
            return df

        try:
            initial_memory = df.memory_usage(deep=True).sum()

            # Downcast numeric columns
            for col in df.select_dtypes(include=["int64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="integer")

            for col in df.select_dtypes(include=["float64"]).columns:
                df[col] = pd.to_numeric(df[col], downcast="float")

            # Optimize object columns
            for col in df.select_dtypes(include=["object"]).columns:
                if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                    df[col] = df[col].astype("category")

            final_memory = df.memory_usage(deep=True).sum()
            memory_reduction = (initial_memory - final_memory) / initial_memory * 100

            self.logger.info(
                f"Memory optimization: {memory_reduction:.1f}% reduction "
                f"({initial_memory / 1024**2:.1f}MB -> {final_memory / 1024**2:.1f}MB)",
            )

            return df
        except Exception as e:
            self.logger.warning(
                f"Memory optimization failed: {e}, returning original DataFrame",
            )
            return df

    def get_database_stats(self) -> dict[str, Any]:
        """Get statistics about the efficiency database."""
        with self.Session() as session:
            # Raw data stats
            raw_count = session.execute(text("SELECT COUNT(*) FROM raw_data")).scalar()

            # Feature cache stats
            feature_count = session.execute(
                text("SELECT COUNT(*) FROM feature_cache"),
            ).scalar()
            feature_types = session.execute(
                text("""
                SELECT feature_type, COUNT(*) as count
                FROM feature_cache
                GROUP BY feature_type
            """),
            ).fetchall()

            # Checkpoint stats
            checkpoint_count = session.execute(
                text("SELECT COUNT(*) FROM processing_checkpoints"),
            ).scalar()

            # Database file size
            db_size = (
                os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            )

        return {
            "raw_data_records": raw_count,
            "feature_cache_records": feature_count,
            "feature_types": dict(feature_types),
            "checkpoints": checkpoint_count,
            "database_size_mb": db_size / (1024 * 1024),
        }

    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to manage storage."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        with self.Session() as session:
            # Clean up old raw data
            deleted_raw = session.execute(
                text("""
                DELETE FROM raw_data 
                WHERE timestamp < :cutoff_date
            """),
                {"cutoff_date": cutoff_date},
            ).rowcount

            # Clean up old feature cache
            deleted_features = session.execute(
                text("""
                DELETE FROM feature_cache 
                WHERE timestamp < :cutoff_date
            """),
                {"cutoff_date": cutoff_date},
            ).rowcount

            session.commit()

        self.logger.info(
            f"Cleaned up {deleted_raw} raw records and {deleted_features} feature records",
        )

        # Vacuum database to reclaim space
        with self.engine.connect() as conn:
            conn.execute(text("VACUUM"))

        self.logger.info("Database vacuumed to reclaim space")
