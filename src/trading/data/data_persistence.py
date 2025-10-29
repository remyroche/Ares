"""
Data Persistence Layer

Provides persistence for collected trading data to various backends.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger

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
                 storage_path: Optional[str] = None):
        """
        Initialize data persistence.
        
        Args:
            backend: Persistence backend type
            storage_path: Path to storage location
        """
        self.backend = backend
        self.storage_path = Path(storage_path) if storage_path else Path("./data_cache")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.getChild(backend.value)
        
        # Backend-specific storage
        self._db_connection = None
        self._init_backend()
    
    def _init_backend(self):
        """Initialize backend-specific storage."""
        if self.backend == PersistenceBackend.SQLITE:
            try:
                import sqlite3
                db_path = self.storage_path / "trading_data.db"
                self._db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
                self._create_tables()
                self.logger.info(f"✅ SQLite backend initialized at {db_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize SQLite: {e}")
                raise
    
    def _create_tables(self):
        """Create database tables for SQLite backend."""
        if self._db_connection:
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
    
    async def save_data_point(self, data_point: Dict[str, Any]) -> bool:
        """
        Save a single data point.
        
        Args:
            data_point: Data point dictionary
            
        Returns:
            bool: True if successful
        """
        try:
            if self.backend == PersistenceBackend.SQLITE:
                return await self._save_sqlite(data_point)
            elif self.backend == PersistenceBackend.PARQUET:
                return await self._save_parquet(data_point)
            elif self.backend == PersistenceBackend.CSV:
                return await self._save_csv(data_point)
            else:
                return False
        except Exception as e:
            self.logger.error(f"❌ Failed to save data point: {e}")
            return False
    
    async def _save_sqlite(self, data_point: Dict[str, Any]) -> bool:
        """Save to SQLite database."""
        try:
            import json
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
            self.logger.error(f"❌ SQLite save error: {e}")
            return False
    
    async def _save_parquet(self, data_point: Dict[str, Any]) -> bool:
        """Save to Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            df = pd.DataFrame([data_point])
            file_path = self.storage_path / f"{data_point.get('symbol', 'unknown')}_data.parquet"
            
            # Append to existing file or create new
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(file_path, index=False)
            return True
        except Exception as e:
            self.logger.error(f"❌ Parquet save error: {e}")
            return False
    
    async def _save_csv(self, data_point: Dict[str, Any]) -> bool:
        """Save to CSV file."""
        try:
            file_path = self.storage_path / f"{data_point.get('symbol', 'unknown')}_data.csv"
            
            df = pd.DataFrame([data_point])
            # Append mode
            df.to_csv(file_path, mode='a', header=not file_path.exists(), index=False)
            return True
        except Exception as e:
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
            if self.backend == PersistenceBackend.SQLITE:
                return await self._load_sqlite(symbol, start_time, end_time, limit)
            elif self.backend == PersistenceBackend.PARQUET:
                return await self._load_parquet(symbol, start_time, end_time, limit)
            elif self.backend == PersistenceBackend.CSV:
                return await self._load_csv(symbol, start_time, end_time, limit)
            else:
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"❌ Failed to load historical data: {e}")
            return pd.DataFrame()
    
    async def _load_sqlite(self, symbol: str, start_time: Optional[datetime],
                          end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from SQLite."""
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, self._db_connection, params=params)
            return df
        except Exception as e:
            self.logger.error(f"❌ SQLite load error: {e}")
            return pd.DataFrame()
    
    async def _load_parquet(self, symbol: str, start_time: Optional[datetime],
                           end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from Parquet."""
        try:
            file_path = self.storage_path / f"{symbol}_data.parquet"
            if not file_path.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
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
            self.logger.error(f"❌ Parquet load error: {e}")
            return pd.DataFrame()
    
    async def _load_csv(self, symbol: str, start_time: Optional[datetime],
                        end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Load from CSV."""
        try:
            file_path = self.storage_path / f"{symbol}_data.csv"
            if not file_path.exists():
                return pd.DataFrame()
            
            df = pd.read_csv(file_path)
            
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
            self.logger.error(f"❌ CSV load error: {e}")
            return pd.DataFrame()
    
    async def cleanup(self):
        """Clean up resources."""
        if self._db_connection:
            self._db_connection.close()
        self.logger.info("🧹 Data persistence cleaned up")
