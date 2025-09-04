"""
Migrated Data Components

This module contains the migrated data loading and preprocessing components
that replace the monolithic data handling in the original architecture.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..enhanced_interfaces import (
    BasePipelineStep, IDataStep, StepResult, StepStatus, StepConfig
)
from ..modular_components import IExchangeDataSource, ExchangeDataSourceFactory

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
        start_date = end_date - timedelta(days=lookback_days)
        
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
        # This would be implemented based on the specific database
        # For now, raise NotImplementedError
        raise NotImplementedError("Database loading not yet implemented")

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
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
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
            total_rows=len(data),
            total_columns=len(data.columns),
            null_percentage=(data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            duplicate_rows=data.duplicated().sum(),
            memory_usage_mb=data.memory_usage(deep=True).sum() / 1024 / 1024
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
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
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
            data = data.rename(columns=column_mapping)
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
            total_rows=len(data),
            total_columns=len(data.columns),
            null_percentage=(data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            duplicate_rows=data.duplicated().sum(),
            memory_usage_mb=data.memory_usage(deep=True).sum() / 1024 / 1024
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of data conversion step."""
        data = kwargs.get('data')
        if data is None:
            raise ValueError("No data provided for conversion")
        
        # Load and validate
        data = await self.load_data("", data=data)
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