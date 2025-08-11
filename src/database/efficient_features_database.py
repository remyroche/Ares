# src/database/efficient_features_database.py

import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd

from src.config import CONFIG
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


class EfficientFeaturesDatabase:
    """
    Efficient database for storing and retrieving precomputed features with incremental updates.
    Uses naming convention: {token}_{exchange}_{date}_{timestamp}_historical_data_with_precomputed_features
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EfficientFeaturesDatabase")

        # Database configuration
        self.db_config = config.get("efficient_features_database", {})
        self.storage_format = self.db_config.get(
            "storage_format",
            "pickle",
        )  # pickle, parquet, hdf5
        self.compression = self.db_config.get("compression", True)
        self.chunk_size = self.db_config.get("chunk_size", 10000)  # rows per chunk

        # Storage paths
        self.base_storage_dir = self.db_config.get(
            "storage_directory",
            os.path.join(CONFIG.get("DATA_DIR", "data"), "precomputed_features"),
        )
        os.makedirs(self.base_storage_dir, exist_ok=True)

        # Cache for database metadata
        self.database_cache = {}
        self.metadata_cache = {}

        self.is_initialized = False

    @handle_errors(exceptions=(Exception,), default_return=False)
    async def initialize(self) -> bool:
        """Initialize the efficient features database."""
        try:
            self.logger.info("🚀 Initializing EfficientFeaturesDatabase...")

            # Scan existing databases
            await self._scan_existing_databases()

            self.is_initialized = True
            self.logger.info("✅ EfficientFeaturesDatabase initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error initializing EfficientFeaturesDatabase: {e}",
            )
            return False

    def _generate_database_name(
        self,
        symbol: str,
        exchange: str,
        start_date: str = None,
        timestamp: str = None,
    ) -> str:
        """
        Generate database name using the specified convention.

        Args:
            symbol: Trading symbol (token)
            exchange: Exchange name
            start_date: Start date (YYYY-MM-DD format)
            timestamp: Timestamp (YYYYMMDD_HHMMSS format)

        Returns:
            Database name following the convention
        """
        if start_date is None:
            start_date = datetime.now().strftime("%Y-%m-%d")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Remove any special characters from symbol
        clean_symbol = symbol.replace("/", "").replace("-", "").upper()
        clean_exchange = exchange.upper()

        return f"{clean_symbol}_{clean_exchange}_{start_date}_{timestamp}_historical_data_with_precomputed_features"

    def _get_database_path(self, database_name: str) -> str:
        """Get full path for database file."""
        if self.storage_format == "pickle":
            extension = ".pkl"
        elif self.storage_format == "parquet":
            extension = ".parquet"
        elif self.storage_format == "hdf5":
            extension = ".h5"
        else:
            extension = ".pkl"

        return os.path.join(self.base_storage_dir, f"{database_name}{extension}")

    @handle_errors(exceptions=(Exception,), default_return=[])
    async def _scan_existing_databases(self) -> list[str]:
        """Scan for existing databases and populate cache."""
        try:
            databases = []

            if not os.path.exists(self.base_storage_dir):
                return databases

            for filename in os.listdir(self.base_storage_dir):
                if (
                    filename.endswith((".pkl", ".parquet", ".h5"))
                    and "precomputed_features" in filename
                ):
                    # Parse database metadata from filename
                    db_name = filename.rsplit(".", 1)[0]  # Remove extension
                    parts = db_name.split("_")

                    if (
                        len(parts) >= 6
                    ):  # Expected format: TOKEN_EXCHANGE_DATE_TIMESTAMP_historical_data_with_precomputed_features
                        symbol = parts[0]
                        exchange = parts[1]
                        date = parts[2]
                        timestamp = parts[3]

                        db_path = os.path.join(self.base_storage_dir, filename)

                        # Get file metadata
                        file_stat = os.stat(db_path)
                        metadata = {
                            "symbol": symbol,
                            "exchange": exchange,
                            "date": date,
                            "timestamp": timestamp,
                            "file_path": db_path,
                            "file_size": file_stat.st_size,
                            "last_modified": datetime.fromtimestamp(file_stat.st_mtime),
                        }

                        # Try to get data range information
                        try:
                            data_info = await self._get_database_info(db_path)
                            metadata.update(data_info)
                        except Exception as e:
                            self.logger.warning(
                                f"Could not read info from {filename}: {e}",
                            )

                        self.metadata_cache[db_name] = metadata
                        databases.append(db_name)

            self.logger.info(
                f"Found {len(databases)} existing precomputed features databases",
            )
            return databases

        except Exception as e:
            self.print(error("Error scanning existing databases: {e}"))
            return []

    @handle_errors(exceptions=(Exception,), default_return={})
    async def _get_database_info(self, db_path: str) -> dict[str, Any]:
        """Get information about a database file."""
        try:
            if self.storage_format == "pickle":
                with open(db_path, "rb") as f:
                    data = pickle.load(f)
            elif self.storage_format == "parquet":
                try:
                    # If index is timestamp-like, loading only index can be heavy; project minimal columns if known
                    cols = getattr(self, "feature_columns", None)
                    if isinstance(cols, list) and len(cols) > 0:
                        data = pd.read_parquet(db_path, columns=cols)
                    else:
                        data = pd.read_parquet(db_path)
                except Exception:
                    data = pd.read_parquet(db_path)
            elif self.storage_format == "hdf5":
                data = pd.read_hdf(db_path, key="features")
            else:
                return {}

            if isinstance(data, pd.DataFrame) and not data.empty:
                return {
                    "start_time": data.index.min(),
                    "end_time": data.index.max(),
                    "num_records": len(data),
                    "num_features": len(data.columns),
                    "feature_categories": self._analyze_feature_categories(
                        data.columns,
                    ),
                }
            return {}

        except Exception as e:
            self.print(warning("Error reading database info: {e}"))
            return {}

    def _analyze_feature_categories(self, columns: list[str]) -> dict[str, int]:
        """Analyze feature categories from column names."""
        categories = {}
        for col in columns:
            if "_" in col:
                category = col.split("_")[0]
                categories[category] = categories.get(category, 0) + 1
        return categories

    @handle_errors(exceptions=(Exception,), default_return=(None, []))
    async def find_existing_database(
        self,
        symbol: str,
        exchange: str,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None,
    ) -> tuple[str | None, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Find existing database for symbol/exchange and determine missing time ranges.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            start_time: Desired start time
            end_time: Desired end time

        Returns:
            Tuple of (database_name, missing_time_ranges)
        """
        try:
            clean_symbol = symbol.replace("/", "").replace("-", "").upper()
            clean_exchange = exchange.upper()

            # Find matching databases
            matching_dbs = []
            for db_name, metadata in self.metadata_cache.items():
                if (
                    metadata.get("symbol", "").upper() == clean_symbol
                    and metadata.get("exchange", "").upper() == clean_exchange
                ):
                    matching_dbs.append((db_name, metadata))

            if not matching_dbs:
                # No existing database
                missing_ranges = (
                    [(start_time, end_time)] if start_time and end_time else []
                )
                return None, missing_ranges

            # Find the most recent database
            latest_db = max(
                matching_dbs,
                key=lambda x: x[1].get("last_modified", datetime.min),
            )
            db_name, metadata = latest_db

            # Check time coverage
            missing_ranges = []
            if start_time and end_time:
                db_start = metadata.get("start_time")
                db_end = metadata.get("end_time")

                if db_start is None or db_end is None:
                    # No time info available, assume we need to process everything
                    missing_ranges = [(start_time, end_time)]
                else:
                    # Convert to pandas timestamps if needed
                    if not isinstance(db_start, pd.Timestamp):
                        db_start = pd.Timestamp(db_start)
                    if not isinstance(db_end, pd.Timestamp):
                        db_end = pd.Timestamp(db_end)

                    # Check for gaps
                    if start_time < db_start:
                        missing_ranges.append((start_time, db_start))
                    if end_time > db_end:
                        missing_ranges.append((db_end, end_time))

            return db_name, missing_ranges

        except Exception as e:
            self.print(error("Error finding existing database: {e}"))
            return None, [(start_time, end_time)] if start_time and end_time else []

    @handle_errors(exceptions=(Exception,), default_return=pd.DataFrame())
    async def load_database(self, database_name: str) -> pd.DataFrame:
        """Load a precomputed features database."""
        try:
            if database_name in self.database_cache:
                self.logger.info(f"Loading database from cache: {database_name}")
                return self.database_cache[database_name].copy()

            if database_name not in self.metadata_cache:
                self.print(missing("Database not found: {database_name}"))
                return pd.DataFrame()

            db_path = self.metadata_cache[database_name]["file_path"]

            self.logger.info(f"Loading database from disk: {database_name}")

            if self.storage_format == "pickle":
                with open(db_path, "rb") as f:
                    data = pickle.load(f)
            elif self.storage_format == "parquet":
                data = pd.read_parquet(db_path)
            elif self.storage_format == "hdf5":
                data = pd.read_hdf(db_path, key="features")
            else:
                return pd.DataFrame()

            # Cache the data if it's not too large
            if len(data) < self.chunk_size * 10:  # Cache if less than 10 chunks
                self.database_cache[database_name] = data.copy()

            self.logger.info(
                f"Loaded {len(data)} records with {len(data.columns)} features",
            )
            return data

        except Exception as e:
            self.print(error("Error loading database {database_name}: {e}"))
            return pd.DataFrame()

    @handle_errors(exceptions=(Exception,), default_return=False)
    async def save_database(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        database_name: str = None,
    ) -> bool:
        """
        Save precomputed features to database.

        Args:
            data: DataFrame with precomputed features
            symbol: Trading symbol
            exchange: Exchange name
            database_name: Optional specific database name

        Returns:
            True if successful, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning("Cannot save empty database")
                return False

            # Generate database name if not provided
            if database_name is None:
                start_date = data.index.min().strftime("%Y-%m-%d")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                database_name = self._generate_database_name(
                    symbol,
                    exchange,
                    start_date,
                    timestamp,
                )

            db_path = self._get_database_path(database_name)

            self.logger.info(f"Saving {len(data)} records to database: {database_name}")

            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, unit="ms")

            # Save based on format
            if self.storage_format == "pickle":
                with open(db_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.storage_format == "parquet":
                if self.compression:
                    data.to_parquet(db_path, compression="snappy")
                else:
                    data.to_parquet(db_path)
            elif self.storage_format == "hdf5":
                if self.compression:
                    data.to_hdf(
                        db_path,
                        key="features",
                        mode="w",
                        complevel=9,
                        complib="zlib",
                    )
                else:
                    data.to_hdf(db_path, key="features", mode="w")

            # Update metadata cache
            file_stat = os.stat(db_path)
            metadata = {
                "symbol": symbol.replace("/", "").replace("-", "").upper(),
                "exchange": exchange.upper(),
                "date": data.index.min().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "file_path": db_path,
                "file_size": file_stat.st_size,
                "last_modified": datetime.fromtimestamp(file_stat.st_mtime),
                "start_time": data.index.min(),
                "end_time": data.index.max(),
                "num_records": len(data),
                "num_features": len(data.columns),
                "feature_categories": self._analyze_feature_categories(data.columns),
            }
            self.metadata_cache[database_name] = metadata

            # Update database cache
            if len(data) < self.chunk_size * 10:
                self.database_cache[database_name] = data.copy()

            self.logger.info(f"✅ Database saved successfully: {db_path}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error saving database: {e}")
            return False

    @handle_errors(exceptions=(Exception,), default_return=False)
    async def update_database(
        self,
        new_data: pd.DataFrame,
        existing_database_name: str,
    ) -> bool:
        """
        Update an existing database with new data - only processes new rows.

        Args:
            new_data: New data to append (only NEW rows that need features/labels)
            existing_database_name: Name of existing database

        Returns:
            True if successful, False otherwise
        """
        try:
            if new_data.empty:
                self.print(warning("No new data to update database"))
                return True

            # Load existing data
            existing_data = await self.load_database(existing_database_name)

            if existing_data.empty:
                self.logger.warning(
                    f"Could not load existing database: {existing_database_name}",
                )
                return False

            # Ensure indices are datetime
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data.index = pd.to_datetime(new_data.index, unit="ms")
            if not isinstance(existing_data.index, pd.DatetimeIndex):
                existing_data.index = pd.to_datetime(existing_data.index, unit="ms")

            # Identify truly new rows (timestamps not in existing data)
            new_timestamps = new_data.index.difference(existing_data.index)
            truly_new_data = new_data.loc[new_timestamps]

            if truly_new_data.empty:
                self.logger.info("No new timestamps found, database already up-to-date")
                return True

            self.logger.info(
                f"Identified {len(truly_new_data)} truly new rows to add to database",
            )

            # Combine existing data with only the new rows
            combined_data = pd.concat([existing_data, truly_new_data], axis=0)

            # Sort by index to maintain chronological order
            combined_data = combined_data.sort_index()

            # Save updated database with timestamp update
            metadata = self.metadata_cache.get(existing_database_name, {})
            symbol = metadata.get("symbol", "UNKNOWN")
            exchange = metadata.get("exchange", "UNKNOWN")

            success = await self._save_database_with_timestamp_update(
                combined_data,
                symbol,
                exchange,
                existing_database_name,
            )

            if success:
                self.logger.info(
                    f"✅ Database updated with {len(truly_new_data)} new records",
                )
                self.logger.info(
                    f"📝 File timestamp updated for database: {existing_database_name}",
                )

            return success

        except Exception as e:
            self.print(error("❌ Error updating database: {e}"))
            return False

    @handle_errors(exceptions=(Exception,), default_return=False)
    async def _save_database_with_timestamp_update(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        database_name: str,
    ) -> bool:
        """
        Save database and explicitly update file timestamp.

        Args:
            data: DataFrame to save
            symbol: Trading symbol
            exchange: Exchange name
            database_name: Database name

        Returns:
            True if successful, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning("Cannot save empty database")
                return False

            db_path = self._get_database_path(database_name)

            self.logger.info(f"Saving {len(data)} records to database: {database_name}")

            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index, unit="ms")

            # Save based on format
            if self.storage_format == "pickle":
                with open(db_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.storage_format == "parquet":
                if self.compression:
                    data.to_parquet(db_path, compression="snappy")
                else:
                    data.to_parquet(db_path)
            elif self.storage_format == "hdf5":
                if self.compression:
                    data.to_hdf(
                        db_path,
                        key="features",
                        mode="w",
                        complevel=9,
                        complib="zlib",
                    )
                else:
                    data.to_hdf(db_path, key="features", mode="w")

            # Explicitly update file timestamp to current time
            current_time = datetime.now()
            os.utime(db_path, (current_time.timestamp(), current_time.timestamp()))

            # Update metadata cache with new timestamp
            file_stat = os.stat(db_path)
            metadata = {
                "symbol": symbol.replace("/", "").replace("-", "").upper(),
                "exchange": exchange.upper(),
                "date": data.index.min().strftime("%Y-%m-%d"),
                "timestamp": current_time.strftime("%Y%m%d_%H%M%S"),
                "file_path": db_path,
                "file_size": file_stat.st_size,
                "last_modified": current_time,  # Use current time explicitly
                "start_time": data.index.min(),
                "end_time": data.index.max(),
                "num_records": len(data),
                "num_features": len(data.columns),
                "feature_categories": self._analyze_feature_categories(data.columns),
                "last_update": current_time.isoformat(),  # Track when it was last updated
            }
            self.metadata_cache[database_name] = metadata

            # Update database cache
            if len(data) < self.chunk_size * 10:
                self.database_cache[database_name] = data.copy()

            self.logger.info(f"✅ Database saved with updated timestamp: {db_path}")
            self.logger.info(f"📅 Last modified: {current_time.isoformat()}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error saving database with timestamp: {e}")
            return False

    def get_database_list(
        self,
        symbol: str = None,
        exchange: str = None,
    ) -> list[dict[str, Any]]:
        """Get list of available databases with optional filtering."""
        databases = []

        for db_name, metadata in self.metadata_cache.items():
            if symbol and metadata.get("symbol", "").upper() != symbol.upper():
                continue
            if exchange and metadata.get("exchange", "").upper() != exchange.upper():
                continue

            databases.append({"name": db_name, "metadata": metadata.copy()})

        # Sort by last modified date (newest first)
        databases.sort(
            key=lambda x: x["metadata"].get("last_modified", datetime.min),
            reverse=True,
        )

        return databases

    def get_database_stats(self) -> dict[str, Any]:
        """Get statistics about all databases."""
        total_databases = len(self.metadata_cache)
        total_size = sum(
            meta.get("file_size", 0) for meta in self.metadata_cache.values()
        )
        total_records = sum(
            meta.get("num_records", 0) for meta in self.metadata_cache.values()
        )

        symbols = {meta.get("symbol") for meta in self.metadata_cache.values()}
        exchanges = {meta.get("exchange") for meta in self.metadata_cache.values()}

        return {
            "total_databases": total_databases,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_records": total_records,
            "unique_symbols": len(symbols),
            "unique_exchanges": len(exchanges),
            "symbols": list(symbols),
            "exchanges": list(exchanges),
            "storage_format": self.storage_format,
            "compression_enabled": self.compression,
        }

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def cleanup_old_databases(self, keep_latest_n: int = 5) -> None:
        """Clean up old databases, keeping only the latest N for each symbol/exchange pair."""
        try:
            # Group databases by symbol/exchange
            symbol_exchange_groups = {}
            for db_name, metadata in self.metadata_cache.items():
                key = (metadata.get("symbol"), metadata.get("exchange"))
                if key not in symbol_exchange_groups:
                    symbol_exchange_groups[key] = []
                symbol_exchange_groups[key].append((db_name, metadata))

            deleted_count = 0
            for databases in symbol_exchange_groups.values():
                # Sort by last modified date (newest first)
                databases.sort(
                    key=lambda x: x[1].get("last_modified", datetime.min),
                    reverse=True,
                )

                # Delete old databases beyond keep_latest_n
                for db_name, metadata in databases[keep_latest_n:]:
                    try:
                        db_path = metadata.get("file_path")
                        if db_path and os.path.exists(db_path):
                            os.remove(db_path)
                            self.logger.info(f"Deleted old database: {db_name}")
                            deleted_count += 1

                        # Remove from cache
                        if db_name in self.metadata_cache:
                            del self.metadata_cache[db_name]
                        if db_name in self.database_cache:
                            del self.database_cache[db_name]

                    except Exception as e:
                        self.logger.error(f"Error deleting database {db_name}: {e}")

            self.logger.info(
                f"Cleanup completed. Deleted {deleted_count} old databases",
            )

        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
