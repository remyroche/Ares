"""
Serialization Utilities

This module provides comprehensive serialization and deserialization utilities
for JSON, pickle, parquet, and other data formats.
"""

import json
import pickle
import gzip
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass

class JSONSerializer:
    """JSON serialization utilities with error handling."""
    
    @staticmethod
    def save(data: Any, file_path: Union[str, Path], indent: int = 2, 
             ensure_ascii: bool = False, compress: bool = False) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to serialize
            file_path: Path to save the file
            indent: JSON indentation
            ensure_ascii: Whether to ensure ASCII encoding
            compress: Whether to compress the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, 
                                default=str)  # Handle non-serializable objects
            
            if compress:
                with gzip.open(file_path.with_suffix(file_path.suffix + '.gz'), 'wt') as f:
                    f.write(json_str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            logger.debug(f"Successfully saved JSON to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            raise SerializationError(f"JSON save failed: {e}")
    
    @staticmethod
    def load(file_path: Union[str, Path], default: Any = None, 
             compressed: bool = False) -> Any:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to the JSON file
            default: Default value if file doesn't exist or fails to load
            compressed: Whether the file is compressed
            
        Returns:
            Loaded data or default value
        """
        try:
            file_path = Path(file_path)
            
            if compressed:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            if not file_path.exists():
                logger.warning(f"JSON file not found: {file_path}")
                return default
            
            if compressed:
                with gzip.open(file_path, 'rt') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            logger.debug(f"Successfully loaded JSON from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return default

class PickleSerializer:
    """Pickle serialization utilities with error handling."""
    
    @staticmethod
    def save(data: Any, file_path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL,
             compress: bool = False) -> bool:
        """
        Save data to pickle file.
        
        Args:
            data: Data to serialize
            file_path: Path to save the file
            protocol: Pickle protocol version
            compress: Whether to compress the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if compress:
                with gzip.open(file_path.with_suffix(file_path.suffix + '.gz'), 'wb') as f:
                    pickle.dump(data, f, protocol=protocol)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=protocol)
            
            logger.debug(f"Successfully saved pickle to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pickle to {file_path}: {e}")
            raise SerializationError(f"Pickle save failed: {e}")
    
    @staticmethod
    def load(file_path: Union[str, Path], default: Any = None,
             compressed: bool = False) -> Any:
        """
        Load data from pickle file.
        
        Args:
            file_path: Path to the pickle file
            default: Default value if file doesn't exist or fails to load
            compressed: Whether the file is compressed
            
        Returns:
            Loaded data or default value
        """
        try:
            file_path = Path(file_path)
            
            if compressed:
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            if not file_path.exists():
                logger.warning(f"Pickle file not found: {file_path}")
                return default
            
            if compressed:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.debug(f"Successfully loaded pickle from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load pickle from {file_path}: {e}")
            return default

class ParquetSerializer:
    """Parquet serialization utilities for DataFrames."""
    
    @staticmethod
    def save(df: pd.DataFrame, file_path: Union[str, Path], 
             compression: str = 'snappy', index: bool = False) -> bool:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the file
            compression: Compression algorithm
            index: Whether to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(file_path, compression=compression, index=index)
            
            logger.debug(f"Successfully saved parquet to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save parquet to {file_path}: {e}")
            raise SerializationError(f"Parquet save failed: {e}")
    
    @staticmethod
    def load(file_path: Union[str, Path], default: Optional[pd.DataFrame] = None,
             columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from parquet file.
        
        Args:
            file_path: Path to the parquet file
            default: Default DataFrame if file doesn't exist or fails to load
            columns: Specific columns to load
            
        Returns:
            Loaded DataFrame or default value
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"Parquet file not found: {file_path}")
                return default
            
            df = pd.read_parquet(file_path, columns=columns)
            
            logger.debug(f"Successfully loaded parquet from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load parquet from {file_path}: {e}")
            return default

class UniversalSerializer:
    """Universal serializer that automatically chooses the best format."""
    
    @staticmethod
    def save(data: Any, file_path: Union[str, Path], 
             format_type: Optional[str] = None, **kwargs) -> bool:
        """
        Save data using the most appropriate format.
        
        Args:
            data: Data to serialize
            file_path: Path to save the file
            format_type: Force specific format ('json', 'pickle', 'parquet')
            **kwargs: Additional arguments for the serializer
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Auto-detect format if not specified
        if format_type is None:
            if isinstance(data, pd.DataFrame):
                format_type = 'parquet'
            elif isinstance(data, (dict, list, str, int, float, bool)):
                format_type = 'json'
            else:
                format_type = 'pickle'
        
        # Choose serializer based on format
        if format_type == 'json':
            return JSONSerializer.save(data, file_path, **kwargs)
        elif format_type == 'pickle':
            return PickleSerializer.save(data, file_path, **kwargs)
        elif format_type == 'parquet':
            if not isinstance(data, pd.DataFrame):
                raise SerializationError("Parquet format requires DataFrame")
            return ParquetSerializer.save(data, file_path, **kwargs)
        else:
            raise SerializationError(f"Unsupported format: {format_type}")
    
    @staticmethod
    def load(file_path: Union[str, Path], default: Any = None,
             format_type: Optional[str] = None, **kwargs) -> Any:
        """
        Load data using the most appropriate format.
        
        Args:
            file_path: Path to the file
            default: Default value if file doesn't exist or fails to load
            format_type: Force specific format ('json', 'pickle', 'parquet')
            **kwargs: Additional arguments for the deserializer
            
        Returns:
            Loaded data or default value
        """
        file_path = Path(file_path)
        
        # Auto-detect format if not specified
        if format_type is None:
            if file_path.suffix == '.parquet':
                format_type = 'parquet'
            elif file_path.suffix == '.json' or file_path.suffix == '.json.gz':
                format_type = 'json'
            elif file_path.suffix == '.pkl' or file_path.suffix == '.pickle':
                format_type = 'pickle'
            else:
                # Try to detect based on content
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    format_type = 'json'
                except:
                    format_type = 'pickle'
        
        # Choose deserializer based on format
        if format_type == 'json':
            return JSONSerializer.load(file_path, default, **kwargs)
        elif format_type == 'pickle':
            return PickleSerializer.load(file_path, default, **kwargs)
        elif format_type == 'parquet':
            return ParquetSerializer.load(file_path, default, **kwargs)
        else:
            raise SerializationError(f"Unsupported format: {format_type}")

# Convenience functions
def save_json(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function to save JSON."""
    return JSONSerializer.save(data, file_path, **kwargs)

def load_json(file_path: Union[str, Path], default: Any = None, **kwargs) -> Any:
    """Convenience function to load JSON."""
    return JSONSerializer.load(file_path, default, **kwargs)

def save_pickle(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function to save pickle."""
    return PickleSerializer.save(data, file_path, **kwargs)

def load_pickle(file_path: Union[str, Path], default: Any = None, **kwargs) -> Any:
    """Convenience function to load pickle."""
    return PickleSerializer.load(file_path, default, **kwargs)

def save_parquet(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function to save parquet."""
    return ParquetSerializer.save(df, file_path, **kwargs)

def load_parquet(file_path: Union[str, Path], default: Optional[pd.DataFrame] = None, **kwargs) -> Optional[pd.DataFrame]:
    """Convenience function to load parquet."""
    return ParquetSerializer.load(file_path, default, **kwargs)

def save_data(data: Any, file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function to save data with auto-format detection."""
    return UniversalSerializer.save(data, file_path, **kwargs)

def load_data(file_path: Union[str, Path], default: Any = None, **kwargs) -> Any:
    """Convenience function to load data with auto-format detection."""
    return UniversalSerializer.load(file_path, default, **kwargs)

__all__ = [
    'SerializationError',
    'JSONSerializer',
    'PickleSerializer', 
    'ParquetSerializer',
    'UniversalSerializer',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'save_parquet',
    'load_parquet',
    'save_data',
    'load_data'
]