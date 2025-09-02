"""
Data manager for pipeline data operations.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class DataManager:
    """
    Manages data operations for pipeline components.
    
    This class handles data loading, validation, processing, and storage
    operations within the trading pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataManager.
        
        Args:
            config: Configuration dictionary for data operations
        """
        self.config = config or {}
        self.data_cache = {}
        self.logger = logging.getLogger(__name__)
        self._setup_connections()
        
    def _setup_connections(self) -> None:
        """Set up database and API connections based on config."""
        # Database connection
        if 'database' in self.config:
            try:
                db_config = self.config['database']
                connection_string = f"postgresql://{db_config.get('user', '')}:{db_config.get('password', '')}@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('name', '')}"
                self.db_engine = create_engine(connection_string)
                self.logger.info("Database connection established")
            except Exception as e:
                self.logger.error(f"Failed to establish database connection: {e}")
                self.db_engine = None
        
        # API configuration
        self.api_config = self.config.get('api', {})
        self.api_headers = self.api_config.get('headers', {})
        self.api_timeout = self.api_config.get('timeout', 30)
        
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from specified source.
        
        Args:
            source: Data source identifier (file, database, api)
            **kwargs: Additional parameters for data loading
            
        Returns:
            Loaded data as pandas DataFrame
        """
        try:
            if source.startswith('file://'):
                return self._load_from_file(source[7:], **kwargs)
            elif source.startswith('db://'):
                return self._load_from_database(source[5:], **kwargs)
            elif source.startswith('api://'):
                return self._load_from_api(source[6:], **kwargs)
            elif source.startswith('cache://'):
                return self._load_from_cache(source[8:], **kwargs)
            else:
                raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            self.logger.error(f"Failed to load data from {source}: {e}")
            raise
            
    def _load_from_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from various file formats."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path, **kwargs)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif file_extension == '.json':
                return pd.read_json(file_path, **kwargs)
            elif file_extension == '.xlsx':
                return pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise Exception(f"Failed to read file {file_path}: {e}")
            
    def _load_from_database(self, query: str, **kwargs) -> pd.DataFrame:
        """Load data from database using SQL query."""
        if not self.db_engine:
            raise Exception("Database connection not available")
            
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except SQLAlchemyError as e:
            raise Exception(f"Database query failed: {e}")
            
    def _load_from_api(self, endpoint: str, **kwargs) -> pd.DataFrame:
        """Load data from API endpoint."""
        try:
            params = kwargs.get('params', {})
            response = requests.get(
                endpoint,
                headers=self.api_headers,
                params=params,
                timeout=self.api_timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
        except requests.RequestException as e:
            raise Exception(f"API request failed: {e}")
            
    def _load_from_cache(self, key: str, **kwargs) -> pd.DataFrame:
        """Load data from cache."""
        if key not in self.data_cache:
            raise KeyError(f"Cache key not found: {key}")
        return self.data_cache[key]
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and integrity.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.warning("Data is None or empty")
            return False
            
        try:
            # Check for required columns
            required_columns = self.config.get('required_columns', [])
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Check for null values in critical columns
            critical_columns = self.config.get('critical_columns', [])
            for col in critical_columns:
                if col in data.columns and data[col].isnull().any():
                    self.logger.warning(f"Critical column {col} contains null values")
                    
            # Check data types
            expected_types = self.config.get('expected_types', {})
            for col, expected_type in expected_types.items():
                if col in data.columns:
                    if not pd.api.types.is_dtype_equal(data[col].dtype, expected_type):
                        self.logger.warning(f"Column {col} has unexpected type: {data[col].dtype}")
                        
            # Check for duplicates
            if self.config.get('check_duplicates', True):
                duplicate_count = data.duplicated().sum()
                if duplicate_count > 0:
                    self.logger.warning(f"Found {duplicate_count} duplicate rows")
                    
            # Check data range/constraints
            constraints = self.config.get('constraints', {})
            for col, constraint in constraints.items():
                if col in data.columns:
                    if 'min' in constraint and data[col].min() < constraint['min']:
                        self.logger.warning(f"Column {col} has values below minimum: {constraint['min']}")
                    if 'max' in constraint and data[col].max() > constraint['max']:
                        self.logger.warning(f"Column {col} has values above maximum: {constraint['max']}")
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
            
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and transform data.
        
        Args:
            data: Raw data to process
            
        Returns:
            Processed data
        """
        if data is None or data.empty:
            return data
            
        try:
            processed_data = data.copy()
            
            # Apply data cleaning
            processed_data = self._clean_data(processed_data)
            
            # Apply transformations
            processed_data = self._apply_transformations(processed_data)
            
            # Apply aggregations
            processed_data = self._apply_aggregations(processed_data)
            
            # Apply filters
            processed_data = self._apply_filters(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
            
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data."""
        # Remove leading/trailing whitespace from string columns
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype(str).str.strip()
            
        # Convert date columns
        date_columns = self.config.get('date_columns', [])
        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                
        # Handle missing values
        missing_strategy = self.config.get('missing_value_strategy', 'drop')
        if missing_strategy == 'drop':
            data = data.dropna(subset=self.config.get('critical_columns', []))
        elif missing_strategy == 'fill':
            fill_values = self.config.get('fill_values', {})
            data = data.fillna(fill_values)
            
        return data
        
    def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data transformations."""
        transformations = self.config.get('transformations', {})
        
        for col, transform_config in transformations.items():
            if col in data.columns:
                transform_type = transform_config.get('type')
                
                if transform_type == 'log':
                    data[col] = np.log(data[col] + 1)
                elif transform_type == 'sqrt':
                    data[col] = np.sqrt(data[col])
                elif transform_type == 'normalize':
                    data[col] = (data[col] - data[col].mean()) / data[col].std()
                elif transform_type == 'scale':
                    min_val = transform_config.get('min', 0)
                    max_val = transform_config.get('max', 1)
                    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min()) * (max_val - min_val) + min_val
                    
        return data
        
    def _apply_aggregations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data aggregations."""
        aggregations = self.config.get('aggregations', {})
        
        if aggregations:
            group_by = aggregations.get('group_by', [])
            agg_functions = aggregations.get('functions', {})
            
            if group_by and agg_functions:
                data = data.groupby(group_by).agg(agg_functions).reset_index()
                
        return data
        
    def _apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply data filters."""
        filters = self.config.get('filters', [])
        
        for filter_config in filters:
            column = filter_config.get('column')
            operator = filter_config.get('operator')
            value = filter_config.get('value')
            
            if column in data.columns:
                if operator == '>':
                    data = data[data[column] > value]
                elif operator == '<':
                    data = data[data[column] < value]
                elif operator == '>=':
                    data = data[data[column] >= value]
                elif operator == '<=':
                    data = data[column] <= value
                elif operator == '==':
                    data = data[data[column] == value]
                elif operator == '!=':
                    data = data[data[column] != value]
                elif operator == 'in':
                    data = data[data[column].isin(value)]
                elif operator == 'not_in':
                    data = data[~data[column].isin(value)]
                    
        return data
        
    def store_data(self, data: pd.DataFrame, destination: str) -> bool:
        """
        Store data to specified destination.
        
        Args:
            data: Data to store
            destination: Storage destination identifier
            
        Returns:
            True if storage was successful
        """
        try:
            if destination.startswith('file://'):
                return self._store_to_file(data, destination[7:])
            elif destination.startswith('db://'):
                return self._store_to_database(data, destination[5:])
            elif destination.startswith('cache://'):
                return self._store_to_cache(data, destination[8:])
            else:
                raise ValueError(f"Unsupported destination: {destination}")
        except Exception as e:
            self.logger.error(f"Failed to store data to {destination}: {e}")
            return False
            
    def _store_to_file(self, data: pd.DataFrame, file_path: str) -> bool:
        """Store data to file."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                data.to_csv(file_path, index=False)
            elif file_extension == '.parquet':
                data.to_parquet(file_path, index=False)
            elif file_extension == '.json':
                data.to_json(file_path, orient='records', indent=2)
            elif file_extension == '.xlsx':
                data.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            self.logger.info(f"Data stored to file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store data to file {file_path}: {e}")
            return False
            
    def _store_to_database(self, data: pd.DataFrame, table_name: str) -> bool:
        """Store data to database table."""
        if not self.db_engine:
            self.logger.error("Database connection not available")
            return False
            
        try:
            data.to_sql(
                table_name,
                self.db_engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            self.logger.info(f"Data stored to database table: {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store data to database table {table_name}: {e}")
            return False
            
    def _store_to_cache(self, data: pd.DataFrame, key: str) -> bool:
        """Store data to cache."""
        try:
            self.data_cache[key] = data
            self.logger.info(f"Data stored to cache with key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store data to cache with key {key}: {e}")
            return False
            
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_info = {}
        for key, data in self.data_cache.items():
            cache_info[key] = {
                'shape': data.shape if data is not None else None,
                'memory_usage': data.memory_usage(deep=True).sum() if data is not None else 0,
                'dtypes': data.dtypes.to_dict() if data is not None else None
            }
        return cache_info
        
    def clear_cache(self, key: Optional[str] = None) -> None:
        """Clear cache or specific cache key."""
        if key:
            if key in self.data_cache:
                del self.data_cache[key]
                self.logger.info(f"Cleared cache key: {key}")
        else:
            self.data_cache.clear()
            self.logger.info("Cleared all cache")

