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

    @handle_errors(exceptions=(OSError, PermissionError, ValueError), default_return=False)
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

    @handle_errors(exceptions=(OSError, PermissionError, ValueError), default_return=[])
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

    @handle_errors(exceptions=(OSError, ValueError, KeyError, pd.errors.EmptyDataError), default_return={})
    def _analyze_feature_categories(self, columns: list[str]) -> dict[str, int]:
        """Analyze feature categories from column names."""
        categories = {}
        for col in columns:
            if "_" in col:
                category = col.split("_")[0]
                categories[category] = categories.get(category, 0) + 1
        return categories

    @handle_errors(exceptions=(ValueError, KeyError, OSError), default_return=(None, []))
    @handle_errors(exceptions=(OSError, ValueError, KeyError, pd.errors.EmptyDataError), default_return=pd.DataFrame())
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

    @handle_errors(exceptions=(OSError, ValueError, PermissionError), default_return=False)
    @handle_errors(exceptions=(ValueError, KeyError, OSError, pd.errors.EmptyDataError), default_return=False)
    @handle_errors(exceptions=(OSError, ValueError, PermissionError), default_return=False)
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

    @handle_errors(exceptions=(OSError, PermissionError, ValueError), default_return=None)