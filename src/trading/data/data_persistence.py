"""
Data Persistence Layer

Provides persistence for collected trading data to various backends.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = system_logger.getChild('DataPersistence')

class PersistenceBackend(Enum):
    """Persistence backend types."""
    SQLITE = "sqlite"
    PARQUET = "parquet"
    CSV = "csv"
    MEMORY = "memory"  # For testing


class DataPersistence:
    """
    Data persistence layer for trading data.

    Supports multiple backends: SQLite, Parquet, CSV
    """

    def __init__(self, backend: PersistenceBackend = PersistenceBackend.SQLITE,
                 storage_path: Optional[str] = None) -> None:
        """
        Initialize data persistence.

        Args:
            backend: Persistence backend type
            storage_path: Path to storage location
        """
        tprint(f"DataPersistence.__init__: backend={backend.value}, storage_path={storage_path}")
        self.backend: PersistenceBackend = backend
        self.storage_path: Path = Path(storage_path) if storage_path else Path("./data_cache")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        tprint(f"DataPersistence.__init__: Created storage directory at {self.storage_path}")
        self.logger = logger.getChild(backend.value)

        # Backend-specific storage
        self._db_connection: Optional[Any] = None
        self._init_backend()
        tprint(f"DataPersistence.__init__: Initialized successfully with {backend.value} backend")

    def _init_backend(self) -> None:
        """Initialize backend-specific storage."""
        tprint(f"_init_backend: Initializing {self.backend.value} backend")
        if self.backend == PersistenceBackend.SQLITE:
            try:
                import sqlite3
                db_path: Path = self.storage_path / "trading_data.db"
                tprint(f"_init_backend: Connecting to SQLite database at {db_path}")
                self._db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
                self._create_tables()
                tprint(f"_init_backend: SQLite backend initialized successfully at {db_path}")
                self.logger.info(f"SQLite backend initialized at {db_path}")
            except Exception as e:
                tprint(f"_init_backend: Failed to initialize SQLite: {e}")
                self.logger.error(f"Failed to initialize SQLite: {e}")
                raise
        tprint(f"_init_backend: Backend initialization complete")

    def _create_tables(self) -> None:
        """Create database tables for SQLite backend."""
        tprint(f"_create_tables: Creating database tables")
        if self._db_connection:
            cursor = self._db_connection.cursor()
            tprint(f"_create_tables: Executing CREATE TABLE statement")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            tprint(f"_create_tables: Creating indexes")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON market_data(symbol)")
            self._db_connection.commit()
            tprint(f"_create_tables: Database tables created successfully")
        else:
            tprint(f"_create_tables: No database connection available")
        tprint(f"_create_tables: Returning from _create_tables")

    async def save_data_point(self, data_point: Dict[str, Any]) -> bool:
        """
        Save a single data point.

        Args:
            data_point: Data point dictionary

        Returns:
            bool: True if successful
        """
        symbol: str = data_point.get('symbol', 'unknown')
        tprint(f"save_data_point: symbol={symbol}, backend={self.backend.value}")
        try:
            if self.backend == PersistenceBackend.SQLITE:
                tprint(f"save_data_point: Saving to SQLite backend")
                result: bool = await self._save_sqlite(data_point)
                tprint(f"save_data_point: SQLite save result={result}")
                return result
            elif self.backend == PersistenceBackend.PARQUET:
                tprint(f"save_data_point: Saving to Parquet backend")
                result = await self._save_parquet(data_point)
                tprint(f"save_data_point: Parquet save result={result}")
                return result
            elif self.backend == PersistenceBackend.CSV:
                tprint(f"save_data_point: Saving to CSV backend")
                result = await self._save_csv(data_point)
                tprint(f"save_data_point: CSV save result={result}")
                return result
            else:
                tprint(f"save_data_point: Unsupported backend: {self.backend.value}, returning False")
                return False
        except Exception as e:
            tprint(f"save_data_point: Failed to save data point: {e}")
            self.logger.error(f"Failed to save data point: {e}")
            tprint(f"save_data_point: Returning False due to exception")
            return False

    async def _save_sqlite(self, data_point: Dict[str, Any]) -> bool:
        """Save to SQLite database."""
        symbol = data_point.get('symbol', 'unknown')
        tprint(f"_save_sqlite: Saving data for {symbol}")
        try:
            import json
            if not self._db_connection:
                tprint(f"_save_sqlite: SQLite connection not available, returning False")
                return False
            cursor = self._db_connection.cursor()
            tprint(f"_save_sqlite: Inserting data into market_data table")
            cursor.execute("""
                INSERT INTO market_data
                (timestamp, symbol, exchange, open, high, low, close, volume, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_point.get('timestamp'),
                data_point.get('symbol'),
                data_point.get('exchange'),
                data_point.get('open'),
                data_point.get('high'),
                data_point.get('low'),
                data_point.get('close'),
                data_point.get('volume'),
                json.dumps(data_point.get('metadata', {}))
            ))
            self._db_connection.commit()
            tprint(f"_save_sqlite: Successfully saved data for {symbol}, returning True")
            return True
        except Exception as e:
            tprint(f"_save_sqlite: SQLite save error: {e}, returning False")
            self.logger.error(f"SQLite save error: {e}")
            return False

    async def _save_parquet(self, data_point: Dict[str, Any]) -> bool:
        """Save to Parquet file."""
        symbol = data_point.get('symbol', 'unknown')
        tprint(f"_save_parquet: Saving data for {symbol}")
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            df: pd.DataFrame = pd.DataFrame([data_point])
            file_path: Path = self.storage_path / f"{symbol}_data.parquet"
            tprint(f"_save_parquet: File path: {file_path}")

            # Append to existing file or create new
            if file_path.exists():
                tprint(f"_save_parquet: Appending to existing Parquet file")
                existing_df: pd.DataFrame = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            else:
                tprint(f"_save_parquet: Creating new Parquet file: {file_path}")

            df.to_parquet(file_path, index=False)
            tprint(f"_save_parquet: Successfully saved to Parquet, returning True")
            return True
        except Exception as e:
            tprint(f"_save_parquet: Parquet save error: {e}, returning False")
            self.logger.error(f"Parquet save error: {e}")
            return False

    async def _save_csv(self, data_point: Dict[str, Any]) -> bool:
        """Save to CSV file."""
        symbol = data_point.get('symbol', 'unknown')
        tprint(f"_save_csv: Saving data for {symbol}")
        try:
            file_path: Path = self.storage_path / f"{symbol}_data.csv"
            tprint(f"_save_csv: File path: {file_path}")

            df: pd.DataFrame = pd.DataFrame([data_point])
            # Append mode
            is_new_file: bool = not file_path.exists()
            if is_new_file:
                tprint(f"_save_csv: Creating new CSV file: {file_path}")
            else:
                tprint(f"_save_csv: Appending to existing CSV file")
            df.to_csv(file_path, mode='a', header=is_new_file, index=False)
            tprint(f"_save_csv: Successfully saved to CSV, returning True")
            return True
        except Exception as e:
            tprint(f"_save_csv: CSV save error: {e}, returning False")
            self.logger.error(f"CSV save error: {e}")
            return False

    async def load_historical_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load historical data from persistence.

        Args:
            symbol: Trading symbol
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of records

        Returns:
            pd.DataFrame: Historical data
        """
        tprint(f"load_historical_data: symbol={symbol}, backend={self.backend.value}, limit={limit}")
        try:
            if self.backend == PersistenceBackend.SQLITE:
                tprint(f"load_historical_data: Loading from SQLite")
                df: pd.DataFrame = await self._load_sqlite(symbol, start_time, end_time, limit)
                tprint(f"load_historical_data: Loaded {len(df)} records from SQLite, returning df")
                return df
            elif self.backend == PersistenceBackend.PARQUET:
                tprint(f"load_historical_data: Loading from Parquet")
                df = await self._load_parquet(symbol, start_time, end_time, limit)
                tprint(f"load_historical_data: Loaded {len(df)} records from Parquet, returning df")
                return df
            elif self.backend == PersistenceBackend.CSV:
                tprint(f"load_historical_data: Loading from CSV")
                df = await self._load_csv(symbol, start_time, end_time, limit)
                tprint(f"load_historical_data: Loaded {len(df)} records from CSV, returning df")
                return df
            else:
                tprint(f"load_historical_data: Unsupported backend: {self.backend.value}, returning empty DataFrame")
                return pd.DataFrame()
        except Exception as e:
            tprint(f"load_historical_data: Failed to load historical data for {symbol}: {e}")
            self.logger.error(f"Failed to load historical data: {e}")
            tprint(f"load_historical_data: Returning empty DataFrame")
            return pd.DataFrame()

    async def _load_sqlite(self, symbol: str, start_time: Optional[datetime],
                          end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from SQLite."""
        tprint(f"_load_sqlite: Loading data for {symbol}")
        try:
            if not self._db_connection:
                tprint(f"_load_sqlite: SQLite connection not available, returning empty DataFrame")
                return pd.DataFrame()
            query: str = "SELECT * FROM market_data WHERE symbol = ?"
            params: List[Union[str, datetime]] = [symbol]
            tprint(f"_load_sqlite: Building query with filters")

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                tprint(f"_load_sqlite: Added start_time filter: {start_time}")
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                tprint(f"_load_sqlite: Added end_time filter: {end_time}")

            query += " ORDER BY timestamp DESC"

            if limit:
                query += f" LIMIT {limit}"
                tprint(f"_load_sqlite: Added limit: {limit}")

            tprint(f"_load_sqlite: Executing query")
            df: pd.DataFrame = pd.read_sql_query(query, self._db_connection, params=params)
            tprint(f"_load_sqlite: Query returned {len(df)} records, returning df")
            return df
        except Exception as e:
            tprint(f"_load_sqlite: SQLite load error: {e}, returning empty DataFrame")
            self.logger.error(f"SQLite load error: {e}")
            return pd.DataFrame()

    async def _load_parquet(self, symbol: str, start_time: Optional[datetime],
                           end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from Parquet."""
        tprint(f"_load_parquet: Loading data for {symbol}")
        try:
            file_path: Path = self.storage_path / f"{symbol}_data.parquet"
            if not file_path.exists():
                tprint(f"_load_parquet: Parquet file not found: {file_path}, returning empty DataFrame")
                return pd.DataFrame()

            tprint(f"_load_parquet: Reading Parquet file: {file_path}")
            df: pd.DataFrame = pd.read_parquet(file_path)
            tprint(f"_load_parquet: Loaded {len(df)} records from file")

            # Apply filters
            if 'timestamp' in df.columns:
                if start_time:
                    df = df[df['timestamp'] >= start_time]
                    tprint(f"_load_parquet: Applied start_time filter, {len(df)} records remaining")
                if end_time:
                    df = df[df['timestamp'] <= end_time]
                    tprint(f"_load_parquet: Applied end_time filter, {len(df)} records remaining")
                df = df.sort_values('timestamp', ascending=False)

            if limit:
                df = df.head(limit)
                tprint(f"_load_parquet: Applied limit, {len(df)} records remaining")

            tprint(f"_load_parquet: Returning {len(df)} records")
            return df
        except Exception as e:
            tprint(f"_load_parquet: Parquet load error: {e}, returning empty DataFrame")
            self.logger.error(f"Parquet load error: {e}")
            return pd.DataFrame()

    async def _load_csv(self, symbol: str, start_time: Optional[datetime],
                        end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from CSV."""
        tprint(f"_load_csv: Loading data for {symbol}")
        try:
            file_path: Path = self.storage_path / f"{symbol}_data.csv"
            if not file_path.exists():
                tprint(f"_load_csv: CSV file not found: {file_path}, returning empty DataFrame")
                return pd.DataFrame()

            tprint(f"_load_csv: Reading CSV file: {file_path}")
            df: pd.DataFrame = pd.read_csv(file_path)
            tprint(f"_load_csv: Loaded {len(df)} records from file")

            # Apply filters
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if start_time:
                    df = df[df['timestamp'] >= start_time]
                    tprint(f"_load_csv: Applied start_time filter, {len(df)} records remaining")
                if end_time:
                    df = df[df['timestamp'] <= end_time]
                    tprint(f"_load_csv: Applied end_time filter, {len(df)} records remaining")
                df = df.sort_values('timestamp', ascending=False)

            if limit:
                df = df.head(limit)
                tprint(f"_load_csv: Applied limit, {len(df)} records remaining")

            tprint(f"_load_csv: Returning {len(df)} records")
            return df
        except Exception as e:
            tprint(f"_load_csv: CSV load error: {e}, returning empty DataFrame")
            self.logger.error(f"CSV load error: {e}")
            return pd.DataFrame()

    async def cleanup(self) -> None:
        """Clean up resources."""
        tprint(f"cleanup: Cleaning up Data Persistence resources")
        if self._db_connection:
            tprint(f"cleanup: Closing database connection")
            self._db_connection.close()
            self._db_connection = None
        tprint(f"cleanup: Data persistence cleaned up successfully")
        self.logger.info("Data persistence cleaned up")
