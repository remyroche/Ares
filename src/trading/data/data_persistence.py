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
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

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
        tprint_info(f"🔄 Initializing Data Persistence with backend: {backend.value}")
        self.backend: PersistenceBackend = backend
        self.storage_path: Path = Path(storage_path) if storage_path else Path("./data_cache")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.getChild(backend.value)
        
        # Backend-specific storage
        self._db_connection: Optional[Any] = None
        self._init_backend()
        tprint_success(f"✅ Data Persistence initialized with {backend.value} backend")
    
    def _init_backend(self) -> None:
        """Initialize backend-specific storage."""
        if self.backend == PersistenceBackend.SQLITE:
            try:
                import sqlite3
                db_path: Path = self.storage_path / "trading_data.db"
                tprint_info(f"🔄 Initializing SQLite backend at {db_path}")
                self._db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
                self._create_tables()
                tprint_success(f"✅ SQLite backend initialized at {db_path}")
                self.logger.info(f"✅ SQLite backend initialized at {db_path}")
            except Exception as e:
                tprint_error(f"❌ Failed to initialize SQLite: {e}")
                self.logger.error(f"❌ Failed to initialize SQLite: {e}")
                raise
    
    def _create_tables(self) -> None:
        """Create database tables for SQLite backend."""
        if self._db_connection:
            tprint_info("🔄 Creating database tables...")
            cursor = self._db_connection.cursor()
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON market_data(symbol)")
            self._db_connection.commit()
            tprint_success("✅ Database tables created successfully")
    
    async def save_data_point(self, data_point: Dict[str, Any]) -> bool:
        """
        Save a single data point.
        
        Args:
            data_point: Data point dictionary
            
        Returns:
            bool: True if successful
        """
        try:
            symbol: str = data_point.get('symbol', 'unknown')
            if self.backend == PersistenceBackend.SQLITE:
                result: bool = await self._save_sqlite(data_point)
                if result:
                    tprint_info(f"💾 Saved data point for {symbol} to SQLite")
                return result
            elif self.backend == PersistenceBackend.PARQUET:
                result = await self._save_parquet(data_point)
                if result:
                    tprint_info(f"💾 Saved data point for {symbol} to Parquet")
                return result
            elif self.backend == PersistenceBackend.CSV:
                result = await self._save_csv(data_point)
                if result:
                    tprint_info(f"💾 Saved data point for {symbol} to CSV")
                return result
            else:
                tprint_warning(f"⚠️ Unsupported backend: {self.backend.value}")
                return False
        except Exception as e:
            tprint_error(f"❌ Failed to save data point: {e}")
            self.logger.error(f"❌ Failed to save data point: {e}")
            return False
    
    async def _save_sqlite(self, data_point: Dict[str, Any]) -> bool:
        """Save to SQLite database."""
        try:
            import json
            if not self._db_connection:
                tprint_error("❌ SQLite connection not available")
                return False
            cursor = self._db_connection.cursor()
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
            return True
        except Exception as e:
            tprint_error(f"❌ SQLite save error: {e}")
            self.logger.error(f"❌ SQLite save error: {e}")
            return False
    
    async def _save_parquet(self, data_point: Dict[str, Any]) -> bool:
        """Save to Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            df: pd.DataFrame = pd.DataFrame([data_point])
            file_path: Path = self.storage_path / f"{data_point.get('symbol', 'unknown')}_data.parquet"
            
            # Append to existing file or create new
            if file_path.exists():
                existing_df: pd.DataFrame = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            else:
                tprint_info(f"📄 Creating new Parquet file: {file_path}")
            
            df.to_parquet(file_path, index=False)
            return True
        except Exception as e:
            tprint_error(f"❌ Parquet save error: {e}")
            self.logger.error(f"❌ Parquet save error: {e}")
            return False
    
    async def _save_csv(self, data_point: Dict[str, Any]) -> bool:
        """Save to CSV file."""
        try:
            file_path: Path = self.storage_path / f"{data_point.get('symbol', 'unknown')}_data.csv"
            
            df: pd.DataFrame = pd.DataFrame([data_point])
            # Append mode
            is_new_file: bool = not file_path.exists()
            if is_new_file:
                tprint_info(f"📄 Creating new CSV file: {file_path}")
            df.to_csv(file_path, mode='a', header=is_new_file, index=False)
            return True
        except Exception as e:
            tprint_error(f"❌ CSV save error: {e}")
            self.logger.error(f"❌ CSV save error: {e}")
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
        try:
            tprint_info(f"📖 Loading historical data for {symbol} (backend: {self.backend.value})")
            if self.backend == PersistenceBackend.SQLITE:
                df: pd.DataFrame = await self._load_sqlite(symbol, start_time, end_time, limit)
                tprint_success(f"✅ Loaded {len(df)} records from SQLite for {symbol}")
                return df
            elif self.backend == PersistenceBackend.PARQUET:
                df = await self._load_parquet(symbol, start_time, end_time, limit)
                tprint_success(f"✅ Loaded {len(df)} records from Parquet for {symbol}")
                return df
            elif self.backend == PersistenceBackend.CSV:
                df = await self._load_csv(symbol, start_time, end_time, limit)
                tprint_success(f"✅ Loaded {len(df)} records from CSV for {symbol}")
                return df
            else:
                tprint_warning(f"⚠️ Unsupported backend: {self.backend.value}")
                return pd.DataFrame()
        except Exception as e:
            tprint_error(f"❌ Failed to load historical data for {symbol}: {e}")
            self.logger.error(f"❌ Failed to load historical data: {e}")
            return pd.DataFrame()
    
    async def _load_sqlite(self, symbol: str, start_time: Optional[datetime],
                          end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from SQLite."""
        try:
            if not self._db_connection:
                tprint_error("❌ SQLite connection not available")
                return pd.DataFrame()
            query: str = "SELECT * FROM market_data WHERE symbol = ?"
            params: List[Union[str, datetime]] = [symbol]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df: pd.DataFrame = pd.read_sql_query(query, self._db_connection, params=params)
            return df
        except Exception as e:
            tprint_error(f"❌ SQLite load error: {e}")
            self.logger.error(f"❌ SQLite load error: {e}")
            return pd.DataFrame()
    
    async def _load_parquet(self, symbol: str, start_time: Optional[datetime],
                           end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from Parquet."""
        try:
            file_path: Path = self.storage_path / f"{symbol}_data.parquet"
            if not file_path.exists():
                tprint_warning(f"⚠️ Parquet file not found: {file_path}")
                return pd.DataFrame()
            
            df: pd.DataFrame = pd.read_parquet(file_path)
            
            # Apply filters
            if 'timestamp' in df.columns:
                if start_time:
                    df = df[df['timestamp'] >= start_time]
                if end_time:
                    df = df[df['timestamp'] <= end_time]
                df = df.sort_values('timestamp', ascending=False)
            
            if limit:
                df = df.head(limit)
            
            return df
        except Exception as e:
            tprint_error(f"❌ Parquet load error: {e}")
            self.logger.error(f"❌ Parquet load error: {e}")
            return pd.DataFrame()
    
    async def _load_csv(self, symbol: str, start_time: Optional[datetime],
                        end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from CSV."""
        try:
            file_path: Path = self.storage_path / f"{symbol}_data.csv"
            if not file_path.exists():
                tprint_warning(f"⚠️ CSV file not found: {file_path}")
                return pd.DataFrame()
            
            df: pd.DataFrame = pd.read_csv(file_path)
            
            # Apply filters
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if start_time:
                    df = df[df['timestamp'] >= start_time]
                if end_time:
                    df = df[df['timestamp'] <= end_time]
                df = df.sort_values('timestamp', ascending=False)
            
            if limit:
                df = df.head(limit)
            
            return df
        except Exception as e:
            tprint_error(f"❌ CSV load error: {e}")
            self.logger.error(f"❌ CSV load error: {e}")
            return pd.DataFrame()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        tprint_info("🧹 Cleaning up Data Persistence resources...")
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None
        tprint_success("✅ Data persistence cleaned up successfully")
        self.logger.info("🧹 Data persistence cleaned up")
