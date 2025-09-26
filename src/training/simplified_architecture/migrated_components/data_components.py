"""
Migrated Data Components

This module contains the migrated data loading and preprocessing components
that replace the monolithic data handling in the original architecture.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from .core.interfaces import (
    BasePipelineStep, IDataStep, StepResult, StepStatus, StepConfig
)
from ..modular_components import ExchangeDataSourceFactory
import pandas as pd
import logging
import numpy as np
import typing
from typing import Optional, Dict, Any

@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation."""
    total_rows: int
    total_columns: int
    null_percentage: float
    duplicate_rows: int
    memory_usage_mb: float
    date_range: Optional[tuple] = None
    price_range: Optional[tuple] = None
    volume_stats: Optional[Dict[str, float]] = None

class DataCollectionStep(BasePipelineStep, IDataStep):
    """
    Migrated data collection step that replaces step01_data_collection.py.
    
    This step handles data collection from various sources including:
    - Exchange APIs (Binance, Coinbase, Kraken)
    - File sources (CSV, Parquet, JSON)
    - Database connections
    """

    @property
    def version(self) -> str:
        return '2.0.0'

    @property
    def description(self) -> str:
        return "Collects market data from various sources with quality validation"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Data source identifier"},
                "symbol": {"type": "string", "description": "Trading symbol"},
                "timeframe": {"type": "string", "description": "Data timeframe"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"}
            },
            "required": ["source", "symbol"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "dataframe", "description": "Market data DataFrame"},
                "metadata": {"type": "object", "description": "Data collection metadata"},
                "quality_metrics": {"type": "object", "description": "Data quality metrics"}
            }
        }

    async def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from specified source."""
        source_type = self.config.parameters.get('source_type', 'file')
        
        if source_type == 'exchange':
            return await self._load_from_exchange(source, **kwargs)
        elif source_type == 'file':
            return await self._load_from_file(source, **kwargs)
        elif source_type == 'database':
            return await self._load_from_database(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    async def _load_from_exchange(self, exchange_name: str, **kwargs) -> pd.DataFrame:
        """Load data from exchange API."""
        symbol = kwargs.get('symbol', self.config.parameters.get('symbol'))
        timeframe = kwargs.get('timeframe', self.config.parameters.get('timeframe', '1h'))
        lookback_days = kwargs.get('lookback_days', self.config.parameters.get('lookback_days', 30))
        
        # Get exchange data source
        try:
            data_source = ExchangeDataSourceFactory.create_exchange(exchange_name)
        except Exception as e:
            raise ValueError(f"Failed to create exchange data source for {exchange_name}: {e}")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days = lookback_days)
        
        self.logger.info(f"Loading data from {exchange_name} for {symbol} ({timeframe}) from {start_date} to {end_date}")
        
        # Fetch data
        data = await data_source.fetch_data(symbol, start_date, end_date)
        
        # Add metadata
        data.attrs['exchange'] = exchange_name
        data.attrs['symbol'] = symbol
        data.attrs['timeframe'] = timeframe
        data.attrs['collection_time'] = datetime.now()
        
        return data

    async def _load_from_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from file: {file_path}")
        
        # Load based on file extension
        if file_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(file_path)
        elif file_path.suffix.lower() in ['.csv', '.txt']:
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            data = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Add metadata
        data.attrs['source_file'] = str(file_path)
        data.attrs['collection_time'] = datetime.now()
        
        return data

    async def _load_from_database(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from database."""
        try:
            # Parse connection string to determine database type
            if connection_string.startswith('postgresql://') or connection_string.startswith('postgres://'):
                return await self._load_from_postgresql(connection_string, **kwargs)
            elif connection_string.startswith('mysql://') or connection_string.startswith('mysql+pymysql://'):
                return await self._load_from_mysql(connection_string, **kwargs)
            elif connection_string.startswith('sqlite:///'):
                return await self._load_from_sqlite(connection_string, **kwargs)
            elif connection_string.startswith('mongodb://'):
                return await self._load_from_mongodb(connection_string, **kwargs)
            else:
                raise ValueError(f"Unsupported database connection string format: {connection_string}")
        except Exception as e:
            self.logger.error(f"Failed to load data from database: {e}")
            raise

    async def _load_from_postgresql(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from PostgreSQL database."""
        try:
            import asyncpg
            import asyncio
            
            # Parse connection parameters
            query = kwargs.get('query', 'SELECT * FROM market_data ORDER BY timestamp')
            symbol = kwargs.get('symbol', 'BTCUSDT')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            # Build parameterized query
            if start_date and end_date:
                query = f"SELECT * FROM market_data WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3 ORDER BY timestamp"
                params = [symbol, start_date, end_date]
            else:
                query = f"SELECT * FROM market_data WHERE symbol = $1 ORDER BY timestamp"
                params = [symbol]
            
            # Connect and fetch data
            conn = await asyncpg.connect(connection_string)
            try:
                rows = await conn.fetch(query, *params)
                if not rows:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = pd.DataFrame([dict(row) for row in rows])
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
                
                self.logger.info(f"Loaded {len(data)} rows from PostgreSQL for {symbol}")
                return data
            finally:
                await conn.close()
                
        except ImportError:
            self.logger.error("asyncpg not available for PostgreSQL connection")
            # Fallback to synchronous psycopg2 if available
            try:
                import psycopg2
                import psycopg2.extras
                
                # Parse connection string for psycopg2
                conn_params = self._parse_postgresql_connection_string(connection_string)
                
                # Build parameterized query
                if start_date and end_date:
                    query = "SELECT * FROM market_data WHERE symbol = %s AND timestamp BETWEEN %s AND %s ORDER BY timestamp"
                    params = [symbol, start_date, end_date]
                else:
                    query = "SELECT * FROM market_data WHERE symbol = %s ORDER BY timestamp"
                    params = [symbol]
                
                # Connect and fetch data synchronously
                with psycopg2.connect(**conn_params) as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                        cursor.execute(query, params)
                        rows = cursor.fetchall()
                        
                        if not rows:
                            self.logger.warning(f"No data found for symbol {symbol}")
                            return pd.DataFrame()
                        
                        # Convert to DataFrame
                        data = pd.DataFrame([dict(row) for row in rows])
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data = data.set_index('timestamp')
                        
                        self.logger.info(f"Loaded {len(data)} rows from PostgreSQL (sync) for {symbol}")
                        return data
                        
            except ImportError:
                self.logger.error("Neither asyncpg nor psycopg2 available for PostgreSQL connection")
                raise NotImplementedError("PostgreSQL support requires asyncpg or psycopg2 package")

    async def _load_from_mysql(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from MySQL database."""
        try:
            import aiomysql
            
            query = kwargs.get('query', 'SELECT * FROM market_data ORDER BY timestamp')
            symbol = kwargs.get('symbol', 'BTCUSDT')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            # Build parameterized query
            if start_date and end_date:
                query = "SELECT * FROM market_data WHERE symbol = %s AND timestamp BETWEEN %s AND %s ORDER BY timestamp"
                params = [symbol, start_date, end_date]
            else:
                query = "SELECT * FROM market_data WHERE symbol = %s ORDER BY timestamp"
                params = [symbol]
            
            # Connect and fetch data
            conn = await aiomysql.connect(connection_string)
            try:
                cursor = await conn.cursor(aiomysql.DictCursor)
                await cursor.execute(query, params)
                rows = await cursor.fetchall()
                
                if not rows:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = pd.DataFrame(rows)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
                
                self.logger.info(f"Loaded {len(data)} rows from MySQL for {symbol}")
                return data
            finally:
                await conn.close()
                
        except ImportError:
            self.logger.error("aiomysql not available for MySQL connection")
            # Fallback to synchronous PyMySQL if available
            try:
                import pymysql
                import pymysql.cursors
                
                # Parse connection string for PyMySQL
                conn_params = self._parse_mysql_connection_string(connection_string)
                
                # Build parameterized query
                if start_date and end_date:
                    query = "SELECT * FROM market_data WHERE symbol = %s AND timestamp BETWEEN %s AND %s ORDER BY timestamp"
                    params = [symbol, start_date, end_date]
                else:
                    query = "SELECT * FROM market_data WHERE symbol = %s ORDER BY timestamp"
                    params = [symbol]
                
                # Connect and fetch data synchronously
                with pymysql.connect(**conn_params) as conn:
                    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                        cursor.execute(query, params)
                        rows = cursor.fetchall()
                        
                        if not rows:
                            self.logger.warning(f"No data found for symbol {symbol}")
                            return pd.DataFrame()
                        
                        # Convert to DataFrame
                        data = pd.DataFrame(rows)
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data = data.set_index('timestamp')
                        
                        self.logger.info(f"Loaded {len(data)} rows from MySQL (sync) for {symbol}")
                        return data
                        
            except ImportError:
                self.logger.error("Neither aiomysql nor pymysql available for MySQL connection")
                raise NotImplementedError("MySQL support requires aiomysql or pymysql package")

    async def _load_from_sqlite(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from SQLite database."""
        try:
            import aiosqlite
            
            # Extract database path from connection string
            db_path = connection_string.replace('sqlite:///', '')
            
            query = kwargs.get('query', 'SELECT * FROM market_data ORDER BY timestamp')
            symbol = kwargs.get('symbol', 'BTCUSDT')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            # Build parameterized query
            if start_date and end_date:
                query = "SELECT * FROM market_data WHERE symbol = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp"
                params = [symbol, start_date, end_date]
            else:
                query = "SELECT * FROM market_data WHERE symbol = ? ORDER BY timestamp"
                params = [symbol]
            
            # Connect and fetch data
            async with aiosqlite.connect(db_path) as conn:
                conn.row_factory = aiosqlite.Row
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
                
                if not rows:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = pd.DataFrame([dict(row) for row in rows])
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
                
                self.logger.info(f"Loaded {len(data)} rows from SQLite for {symbol}")
                return data
                
        except ImportError:
            self.logger.error("aiosqlite not available for SQLite connection")
            # Fallback to synchronous sqlite3 if available
            try:
                import sqlite3
                
                # Extract database path from connection string
                db_path = connection_string.replace('sqlite:///', '')
                
                # Build parameterized query
                if start_date and end_date:
                    query = "SELECT * FROM market_data WHERE symbol = ? AND timestamp BETWEEN ? AND ? ORDER BY timestamp"
                    params = [symbol, start_date, end_date]
                else:
                    query = "SELECT * FROM market_data WHERE symbol = ? ORDER BY timestamp"
                    params = [symbol]
                
                # Connect and fetch data synchronously
                with sqlite3.connect(db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    if not rows:
                        self.logger.warning(f"No data found for symbol {symbol}")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    data = pd.DataFrame([dict(row) for row in rows])
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data = data.set_index('timestamp')
                    
                    self.logger.info(f"Loaded {len(data)} rows from SQLite (sync) for {symbol}")
                    return data
                    
            except ImportError:
                self.logger.error("Neither aiosqlite nor sqlite3 available for SQLite connection")
                raise NotImplementedError("SQLite support requires aiosqlite or sqlite3 package")

    async def _load_from_mongodb(self, connection_string: str, **kwargs) -> pd.DataFrame:
        """Load data from MongoDB database."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from bson import ObjectId
            
            # Parse connection string and collection
            collection_name = kwargs.get('collection', 'market_data')
            symbol = kwargs.get('symbol', 'BTCUSDT')
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            
            # Connect to MongoDB
            client = AsyncIOMotorClient(connection_string)
            db = client.get_default_database()
            collection = db[collection_name]
            
            # Build query
            query = {'symbol': symbol}
            if start_date and end_date:
                query['timestamp'] = {'$gte': start_date, '$lte': end_date}
            
            # Fetch data
            cursor = collection.find(query).sort('timestamp', 1)
            rows = await cursor.to_list(length=None)
            
            if not rows:
                self.logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame(rows)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
            
            # Remove MongoDB _id field if present
            if '_id' in data.columns:
                data = data.drop('_id', axis=1)
            
            self.logger.info(f"Loaded {len(data)} rows from MongoDB for {symbol}")
            return data
            
        except ImportError:
            self.logger.error("motor not available for MongoDB connection")
            # Fallback to synchronous pymongo if available
            try:
                import pymongo
                from pymongo import MongoClient
                
                # Parse connection string and collection
                collection_name = kwargs.get('collection', 'market_data')
                symbol = kwargs.get('symbol', 'BTCUSDT')
                start_date = kwargs.get('start_date')
                end_date = kwargs.get('end_date')
                
                # Connect to MongoDB
                client = MongoClient(connection_string)
                db = client.get_default_database()
                collection = db[collection_name]
                
                # Build query
                query = {'symbol': symbol}
                if start_date and end_date:
                    query['timestamp'] = {'$gte': start_date, '$lte': end_date}
                
                # Fetch data
                cursor = collection.find(query).sort('timestamp', 1)
                rows = list(cursor)
                
                if not rows:
                    self.logger.warning(f"No data found for symbol {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = pd.DataFrame(rows)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
                
                # Remove MongoDB _id field if present
                if '_id' in data.columns:
                    data = data.drop('_id', axis=1)
                
                self.logger.info(f"Loaded {len(data)} rows from MongoDB (sync) for {symbol}")
                return data
                
            except ImportError:
                self.logger.error("Neither motor nor pymongo available for MongoDB connection")
                raise NotImplementedError("MongoDB support requires motor or pymongo package")

    def _parse_postgresql_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """Parse PostgreSQL connection string for psycopg2."""
        from urllib.parse import urlparse
        
        parsed = urlparse(connection_string)
        return {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 5432,
            'database': parsed.path.lstrip('/') if parsed.path else 'postgres',
            'user': parsed.username or 'postgres',
            'password': parsed.password or ''
        }

    def _parse_mysql_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """Parse MySQL connection string for PyMySQL."""
        from urllib.parse import urlparse
        
        parsed = urlparse(connection_string)
        return {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 3306,
            'database': parsed.path.lstrip('/') if parsed.path else 'mysql',
            'user': parsed.username or 'root',
            'password': parsed.password or '',
            'charset': 'utf8mb4'
        }

    async def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data meets requirements."""
        required_columns = self.config.parameters.get('required_columns', [])
        
        # Check required columns
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            self.add_warning(f'Missing required columns: {missing_columns}')
            return False
        
        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        max_null_percentage = self.config.parameters.get('max_null_percentage', 5.0)
        
        for col, null_count in null_counts.items():
            null_percentage = (null_count / len(data)) * 100
            if null_percentage > max_null_percentage:
                self.add_warning(f'Column {col} has {null_percentage:.2f}% null values (max: {max_null_percentage}%)')
                return False
        
        # Check minimum rows
        min_rows = self.config.parameters.get('min_rows', 100)
        if len(data) < min_rows:
            self.add_warning(f'Insufficient data: {len(data)} rows (min: {min_rows})')
            return False
        
        # Check data types for price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    self.add_warning(f'Column {col} is not numeric')
                    return False
                
                # Check for negative prices
                if (data[col] <= 0).any():
                    self.add_warning(f'Column {col} contains non-positive values')
                    return False
        
        return True

    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for downstream steps."""
        # Remove duplicates
        initial_rows = len(data)
        data = data.drop_duplicates()
        if len(data) < initial_rows:
            self.add_metric('duplicates_removed', initial_rows - len(data))
        
        # Sort by index (usually timestamp)
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
            self.add_metric('data_sorted', True)
        
        # Handle missing values
        if data.isnull().any().any():
            # Forward fill for price data, backward fill for volume
            price_columns = ['open', 'high', 'low', 'close']
            volume_columns = ['volume']
            
            for col in price_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill')
            
            for col in volume_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(method='bfill')
            
            self.add_metric('missing_values_handled', True)
        
        # Add basic technical indicators if requested
        if self.config.parameters.get('add_basic_indicators', False):
            data = self._add_basic_indicators(data)
        
        return data

    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        if 'close' not in data.columns:
            return data
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            data[f'sma_{period}'] = data['close'].rolling(window = period).mean()
        
        # Price change
        data['price_change'] = data['close'].pct_change()
        
        # Volume change
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
        
        self.add_metric('basic_indicators_added', True)
        return data

    def get_data_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Get comprehensive data quality metrics."""
        metrics = DataQualityMetrics(
            total_rows = len(data),
            total_columns = len(data.columns),
            null_percentage=(data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            duplicate_rows = data.duplicated().sum(),
            memory_usage_mb = data.memory_usage(deep = True).sum() / 1024 / 1024
        )
        
        # Date range
        if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
            metrics.date_range = (data.index.min(), data.index.max())
        
        # Price range
        price_columns = ['open', 'high', 'low', 'close']
        existing_price_columns = [col for col in price_columns if col in data.columns]
        if existing_price_columns:
            min_price = data[existing_price_columns].min().min()
            max_price = data[existing_price_columns].max().max()
            metrics.price_range = (min_price, max_price)
        
        # Volume statistics
        if 'volume' in data.columns:
            metrics.volume_stats = {
                'mean': data['volume'].mean(),
                'std': data['volume'].std(),
                'min': data['volume'].min(),
                'max': data['volume'].max(),
                'median': data['volume'].median()
            }
        
        return metrics

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of data collection step."""
        source = kwargs.get('source', self.config.parameters.get('source'))
        if not source:
            raise ValueError("Data source not specified")
        
        # Load data
        data = await self.load_data(source, **kwargs)
        
        # Validate data
        if not await self.validate_data(data):
            raise ValueError('Data validation failed')
        
        # Preprocess data
        data = await self.preprocess_data(data)
        
        # Get quality metrics
        quality_metrics = self.get_data_quality_metrics(data)
        
        # Add metrics to step result
        self.add_metric('total_rows', quality_metrics.total_rows)
        self.add_metric('total_columns', quality_metrics.total_columns)
        self.add_metric('null_percentage', quality_metrics.null_percentage)
        self.add_metric('duplicate_rows', quality_metrics.duplicate_rows)
        self.add_metric('memory_usage_mb', quality_metrics.memory_usage_mb)
        
        # Save snapshot if requested
        if self.config.parameters.get('save_snapshot', False):
            snapshot_path = Path(f'data/snapshots/{self.name}_{int(time.time())}.parquet')
            snapshot_path.parent.mkdir(parents = True, exist_ok = True)
            data.to_parquet(snapshot_path)
            self.add_artifact('data_snapshot', snapshot_path)
        
        return {
            'data': data,
            'metadata': {
                'source': source,
                'collection_time': datetime.now().isoformat(),
                'data_shape': data.shape,
                'columns': list(data.columns)
            },
            'quality_metrics': quality_metrics
        }

class DataConverterStep(BasePipelineStep, IDataStep):
    """
    Migrated data converter step that replaces step01_5_data_converter.py.
    
    This step converts data from various formats to a unified format.
    """

    @property
    def version(self) -> str:
        return '2.0.0'

    @property
    def description(self) -> str:
        return "Converts data from various formats to unified format"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "dataframe", "description": "Input data to convert"},
                "target_format": {"type": "string", "description": "Target format specification"}
            },
            "required": ["data"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data": {"type": "dataframe", "description": "Converted data in unified format"},
                "conversion_metadata": {"type": "object", "description": "Conversion process metadata"}
            }
        }

    async def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data for conversion."""
        # This step typically receives data from previous step
        return kwargs.get('data')

    async def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data for conversion."""
        if data is None or data.empty:
            self.add_warning("No data provided for conversion")
            return False
        
        return True

    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data during conversion."""
        # Standardize column names
        column_mapping = self.config.parameters.get('column_mapping', {})
        if column_mapping:
            data = data.rename(columns = column_mapping)
            self.add_metric('columns_renamed', len(column_mapping))
        
        # Ensure required columns exist
        required_columns = self.config.parameters.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns after conversion: {missing_columns}")
        
        # Standardize data types
        data_types = self.config.parameters.get('data_types', {})
        for col, dtype in data_types.items():
            if col in data.columns:
                data[col] = data[col].astype(dtype)
        
        return data

    def get_data_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """Get data quality metrics after conversion."""
        return DataQualityMetrics(
            total_rows = len(data),
            total_columns = len(data.columns),
            null_percentage=(data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            duplicate_rows = data.duplicated().sum(),
            memory_usage_mb = data.memory_usage(deep = True).sum() / 1024 / 1024
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of data conversion step."""
        data = kwargs.get('data')
        if data is None:
            raise ValueError("No data provided for conversion")
        
        # Load and validate
        data = await self.load_data("", data = data)
        if not await self.validate_data(data):
            raise ValueError('Data validation failed')
        
        # Convert and preprocess
        converted_data = await self.preprocess_data(data)
        
        # Get quality metrics
        quality_metrics = self.get_data_quality_metrics(converted_data)
        
        # Add conversion metadata
        conversion_metadata = {
            'original_shape': data.shape,
            'converted_shape': converted_data.shape,
            'columns_mapped': self.config.parameters.get('column_mapping', {}),
            'conversion_time': datetime.now().isoformat()
        }
        
        # Add metrics
        self.add_metric('original_rows', data.shape[0])
        self.add_metric('converted_rows', converted_data.shape[0])
        self.add_metric('original_columns', data.shape[1])
        self.add_metric('converted_columns', converted_data.shape[1])
        
        return {
            'data': converted_data,
            'conversion_metadata': conversion_metadata,
            'quality_metrics': quality_metrics
        }

# Register the migrated components
from ..enhanced_interfaces import StepFactory

StepFactory.register_step('data_collection', DataCollectionStep)
StepFactory.register_step('data_converter', DataConverterStep)