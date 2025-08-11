# src/training/data_efficiency_optimizer.py

import gc
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Number

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    validation_error,
    warning,
)


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

    def __init__(
        self,
        db_manager: SQLiteManager,
        symbol: str,
        timeframe: str,
        exchange: str = "BINANCE",
    ):
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

            # Feature cache table (legacy format)
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

            # Feature cache table (wide format)
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS feature_cache_wide (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    feature_type TEXT NOT NULL,
                    feature_data TEXT,  -- JSON-like string with all features for timestamp
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, feature_type)
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

            # Create indexes for wide format feature cache
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_feature_cache_wide_timestamp
                ON feature_cache_wide(timestamp)
            """),
            )
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_feature_cache_wide_type
                ON feature_cache_wide(feature_type)
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

    def get_memory_usage(self) -> Number:
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
        cache_dir = self.cache_dir / f"{cache_key}_cached_data"

        # Check if cache exists and is valid
        if not force_reload and cache_dir.exists():
            # Check if any Parquet files exist in cache directory
            parquet_files = list(cache_dir.glob("*.parquet"))
            if parquet_files:
                # Use the oldest file's modification time as cache age
                cache_age = time.time() - min(f.stat().st_mtime for f in parquet_files)
                max_cache_age = 24 * 60 * 60  # 24 hours

                if cache_age < max_cache_age:
                    self.logger.info(f"Loading data from Parquet cache: {cache_dir}")
                    try:
                        data = {}

                        # Load each data type from its Parquet file
                        for data_type in ["klines", "agg_trades", "futures"]:
                            parquet_file = cache_dir / f"{data_type}.parquet"
                            if parquet_file.exists():
                                try:
                                    data[data_type] = pq.read_table(
                                        parquet_file,
                                    ).to_pandas()
                                    self.logger.info(
                                        f"Loaded {len(data[data_type])} {data_type} from cache",
                                    )
                                except Exception as e:
                                    self.logger.warning(
                                        f"Failed to load {data_type} from cache: {e}",
                                    )
                                    data[data_type] = pd.DataFrame()
                            else:
                                data[data_type] = pd.DataFrame()

                        # Validate cache data
                        if self._validate_cached_data(data, lookback_days):
                            self.logger.info("Cache validation successful")
                            return data
                        self.print(failed("Cache validation failed, reloading data"))
                    except Exception:
                        self.print(failed("Cache loading failed: {e}"))

        # Load data from source with memory management
        self.logger.info(f"Loading {lookback_days} days of data from source...")
        data = await self._load_data_from_source(lookback_days)

        # Cache the loaded data
        self._cache_data(
            data,
            cache_dir / "dummy.parquet",
        )  # Pass a dummy file, actual caching uses the directory

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

            for df in data.values():
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
            error_msg = f"Cache validation error: {e}"
            self.logger.error(error_msg)
            self.print(validation_error(error_msg))
            return False

    async def _load_data_from_source(
        self,
        lookback_days: int,
    ) -> dict[str, pd.DataFrame]:
        """Load data from source with memory-efficient processing."""
        self.logger.info(f"Loading data from source for {lookback_days} days...")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        data = {
            "klines": pd.DataFrame(),
            "agg_trades": pd.DataFrame(),
            "futures": pd.DataFrame(),
        }

        try:
            # First, try to load from Parquet files (modern format)
            parquet_dir = Path("data/parquet")
            if parquet_dir.exists():
                self.logger.info("Attempting to load from Parquet files...")

                # Try to load klines data
                klines_file = (
                    parquet_dir / f"{self.exchange}_{self.symbol}_klines.parquet"
                )
                if klines_file.exists():
                    try:
                        data["klines"] = pq.read_table(klines_file).to_pandas()
                        self.logger.info(
                            f"Loaded {len(data['klines'])} klines from Parquet",
                        )
                    except Exception:
                        self.print(failed("Failed to load klines from Parquet: {e}"))

                # Try to load aggregated trades
                trades_file = (
                    parquet_dir / f"{self.exchange}_{self.symbol}_agg_trades.parquet"
                )
                if trades_file.exists():
                    try:
                        data["agg_trades"] = pq.read_table(trades_file).to_pandas()
                        self.logger.info(
                            f"Loaded {len(data['agg_trades'])} aggregated trades from Parquet",
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load aggregated trades from Parquet: {e}",
                        )

                # Try to load futures data
                futures_file = (
                    parquet_dir / f"{self.exchange}_{self.symbol}_futures.parquet"
                )
                if futures_file.exists():
                    try:
                        data["futures"] = pq.read_table(futures_file).to_pandas()
                        self.logger.info(
                            f"Loaded {len(data['futures'])} futures records from Parquet",
                        )
                    except Exception:
                        self.print(failed("Failed to load futures from Parquet: {e}"))

            # If Parquet files don't exist or are empty, try database query
            if all(df.empty for df in data.values()) and self.db_manager:
                self.logger.info("Attempting to load from database...")
                try:
                    # Query the database for klines data
                    klines_query = f"""
                    SELECT * FROM klines
                    WHERE symbol = '{self.symbol}'
                    AND exchange = '{self.exchange}'
                    AND timestamp BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
                    ORDER BY timestamp
                    """

                    with self.db_manager.get_session() as session:
                        result = session.execute(text(klines_query))
                        if result:
                            data["klines"] = pd.DataFrame(result.fetchall())
                            if not data["klines"].empty:
                                self.logger.info(
                                    f"Loaded {len(data['klines'])} klines from database",
                                )

                    # Query for aggregated trades
                    trades_query = f"""
                    SELECT * FROM agg_trades
                    WHERE symbol = '{self.symbol}'
                    AND exchange = '{self.exchange}'
                    AND timestamp BETWEEN '{start_date.isoformat()}' AND '{end_date.isoformat()}'
                    ORDER BY timestamp
                    """

                    with self.db_manager.get_session() as session:
                        result = session.execute(text(trades_query))
                        if result:
                            data["agg_trades"] = pd.DataFrame(result.fetchall())
                            if not data["agg_trades"].empty:
                                self.logger.info(
                                    f"Loaded {len(data['agg_trades'])} aggregated trades from database",
                                )

                except Exception:
                    self.print(failed("Database query failed: {e}"))

            # Final fallback: try legacy pickle file
            if all(df.empty for df in data.values()):
                data_file = f"data/{self.exchange}_{self.symbol}_historical_data.pkl"
                if os.path.exists(data_file):
                    self.logger.info(
                        f"Loading data from legacy pickle file: {data_file}",
                    )
                    try:
                        import pickle

                        with open(data_file, "rb") as f:
                            legacy_data = pickle.load(f)

                        # Validate that we have the expected data structure
                        if (
                            isinstance(legacy_data, dict)
                            and "klines" in legacy_data
                            and isinstance(legacy_data["klines"], pd.DataFrame)
                            and not legacy_data["klines"].empty
                        ):
                            data = legacy_data
                            self.logger.info(
                                f"Successfully loaded {len(data['klines'])} klines records from legacy file",
                            )
                        else:
                            self.logger.warning(
                                "Legacy data file has invalid structure",
                            )
                    except Exception:
                        self.print(failed("Failed to load legacy data file: {e}"))

            # Filter data by date range if we have data
            for data_type, df in data.items():
                if not df.empty and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    mask = (df["timestamp"] >= start_date) & (
                        df["timestamp"] <= end_date
                    )
                    data[data_type] = df[mask].copy()

            # Log summary
            total_records = sum(len(df) for df in data.values())
            self.logger.info(f"Loaded {total_records} total records from source")

        except Exception as e:
            error_msg = f"Error loading data from source: {e}"
            self.logger.error(error_msg)
            self.print(error(error_msg))

        return data

    def _cache_data(self, data: dict[str, pd.DataFrame], cache_file: Path):
        """Cache data to disk using Parquet format for high performance."""
        try:
            self.logger.info(f"Caching data to {cache_file}")

            # Create cache directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Store each DataFrame as a separate Parquet file for better performance
            cache_base = cache_file.stem
            cache_dir = cache_file.parent / cache_base

            for data_type, df in data.items():
                if not df.empty:
                    # Create wide format: each row is a timestamp, each column is a feature
                    if "timestamp" in df.columns:
                        df_wide = df.set_index("timestamp")
                    else:
                        # If no timestamp column, use the index
                        df_wide = df.copy()

                    # Save as Parquet with compression
                    parquet_file = cache_dir / f"{data_type}.parquet"
                    parquet_file.parent.mkdir(exist_ok=True)

                    # Convert to PyArrow table and write with compression
                    table = pa.Table.from_pandas(df_wide)
                    pq.write_table(table, parquet_file, compression="snappy")

                    self.logger.info(
                        f"Cached {len(df_wide)} {data_type} records to {parquet_file}",
                    )

            self.logger.info("Data cached successfully in Parquet format")
        except Exception:
            self.print(failed("Failed to cache data: {e}"))

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
                    self.print(warning("Could not reconstruct open_time column"))

        # Fix close_time if it's missing
        if (
            "close_time" in processed_chunk.columns
            and processed_chunk["close_time"].isna().all()
        ) and (
            "open_time" in processed_chunk.columns
            and not processed_chunk["open_time"].isna().all()
        ):
            # Set close_time to open_time + 1 hour (for 1h timeframe)
            processed_chunk["close_time"] = processed_chunk["open_time"] + pd.Timedelta(
                hours=1,
            )

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
        Store computed features in SQLite database in wide format for efficient retrieval.

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
            # Store features in wide format: each row is a timestamp, each column is a feature
            for timestamp, row in features.iterrows():
                # Convert timestamp to string for SQLite compatibility
                timestamp_str = (
                    timestamp.isoformat()
                    if hasattr(timestamp, "isoformat")
                    else str(timestamp)
                )

                # Create a record with all features for this timestamp
                record = {
                    "timestamp": timestamp_str,
                    "feature_type": feature_type,
                }

                # Add each feature as a column
                for feature_name, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        record[f"feature_{feature_name}"] = float(value)

                # Insert the wide-format record
                session.execute(
                    text("""
                    INSERT INTO feature_cache_wide (timestamp, feature_type, feature_data)
                    VALUES (:timestamp, :feature_type, :feature_data)
                    ON CONFLICT(timestamp, feature_type)
                    DO UPDATE SET feature_data = :feature_data
                """),
                    {
                        "timestamp": timestamp_str,
                        "feature_type": feature_type,
                        "feature_data": str(
                            record,
                        ),  # Store as JSON-like string for now
                    },
                )

            session.commit()

        self.logger.info(f"Stored {len(features)} feature records in wide format")

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
            DataFrame with features in wide format
        """
        with self.Session() as session:
            # Try wide format first (more efficient)
            query = text("""
                SELECT timestamp, feature_type, feature_data
                FROM feature_cache_wide
                WHERE timestamp BETWEEN :start_date AND :end_date
            """)

            # SQLAlchemy can handle datetime objects directly
            params = {"start_date": start_date, "end_date": end_date}

            result = session.execute(query, params)
            rows = result.fetchall()

            if rows:
                # Process wide format data
                features_list = []
                for row in rows:
                    timestamp = pd.to_datetime(row[0])
                    feature_data_str = row[2]

                    # Parse the feature data (stored as string representation)
                    try:
                        # Simple parsing - in production you might want to use JSON
                        import ast

                        feature_dict = ast.literal_eval(feature_data_str)

                        # Extract features, excluding timestamp and feature_type
                        features = {
                            k: v
                            for k, v in feature_dict.items()
                            if k.startswith("feature_")
                            and k not in ["timestamp", "feature_type"]
                        }

                        if features:
                            features["timestamp"] = timestamp
                            features_list.append(features)
                    except Exception:
                        self.print(failed("Failed to parse feature data: {e}"))
                        continue

                if features_list:
                    features_df = pd.DataFrame(features_list)
                    features_df.set_index("timestamp", inplace=True)

                    # Filter by requested feature names if specified
                    if feature_names:
                        available_features = [
                            col
                            for col in features_df.columns
                            if col.startswith("feature_")
                        ]
                        requested_features = [
                            f"feature_{name}"
                            for name in feature_names
                            if f"feature_{name}" in available_features
                        ]
                        features_df = features_df[requested_features]

                    self.logger.info(
                        f"Loaded {len(features_df)} feature records from wide format",
                    )
                    return features_df

            # Fallback to legacy format if wide format is empty
            self.logger.info("Wide format empty, trying legacy format...")
            query = text("""
                SELECT timestamp, feature_name, feature_value
                FROM feature_cache
                WHERE timestamp BETWEEN :start_date AND :end_date
            """)

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

        self.logger.info(
            f"Loaded {len(features_df)} feature records from legacy format",
        )
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

            # Feature cache stats (legacy format)
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

            # Feature cache stats (wide format)
            feature_count_wide = session.execute(
                text("SELECT COUNT(*) FROM feature_cache_wide"),
            ).scalar()
            feature_types_wide = session.execute(
                text("""
                SELECT feature_type, COUNT(*) as count
                FROM feature_cache_wide
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
            "feature_cache_records_wide": feature_count_wide,
            "feature_types": dict(feature_types),
            "feature_types_wide": dict(feature_types_wide),
            "checkpoints": checkpoint_count,
            "database_size_mb": db_size / (1024 * 1024),
        }

    def migrate_pickle_to_parquet(self, pickle_file_path: str) -> bool:
        """
        Migrate existing pickle data to Parquet format.

        Args:
            pickle_file_path: Path to the pickle file to migrate

        Returns:
            True if migration was successful, False otherwise
        """
        try:
            self.logger.info(f"Migrating pickle file to Parquet: {pickle_file_path}")

            # Load pickle data
            import pickle

            with open(pickle_file_path, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                self.print(error("Pickle file does not contain a dictionary"))
                return False

            # Create Parquet directory
            parquet_dir = Path("data/parquet")
            parquet_dir.mkdir(parents=True, exist_ok=True)

            # Convert each DataFrame to Parquet
            for data_type, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Create wide format: each row is a timestamp, each column is a feature
                    if "timestamp" in df.columns:
                        df_wide = df.set_index("timestamp")
                    else:
                        df_wide = df.copy()

                    # Save as Parquet
                    parquet_file = (
                        parquet_dir
                        / f"{self.exchange}_{self.symbol}_{data_type}.parquet"
                    )
                    table = pa.Table.from_pandas(df_wide)
                    pq.write_table(table, parquet_file, compression="snappy")

                    self.logger.info(
                        f"Migrated {len(df_wide)} {data_type} records to {parquet_file}",
                    )

            self.logger.info("Migration completed successfully")
            return True

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            self.logger.exception(error_msg)
            self.print(failed(error_msg))
            return False

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

            # Clean up old feature cache (legacy format)
            deleted_features = session.execute(
                text("""
                DELETE FROM feature_cache
                WHERE timestamp < :cutoff_date
            """),
                {"cutoff_date": cutoff_date},
            ).rowcount

            # Clean up old feature cache (wide format)
            deleted_features_wide = session.execute(
                text("""
                DELETE FROM feature_cache_wide
                WHERE timestamp < :cutoff_date
            """),
                {"cutoff_date": cutoff_date},
            ).rowcount

            session.commit()

        self.logger.info(
            f"Cleaned up {deleted_raw} raw records, {deleted_features} legacy feature records, "
            f"and {deleted_features_wide} wide format feature records",
        )

        # Vacuum database to reclaim space
        with self.engine.connect() as conn:
            conn.execute(text("VACUUM"))

        self.logger.info("Database vacuumed to reclaim space")
