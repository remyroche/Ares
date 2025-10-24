"""
Serialization Utilities Module

This module provides various serialization utilities for data persistence.
Uses kline_parquet.py for efficient Parquet operations and prefers Parquet over Pickle.

Key Features:
- Automatic format detection with Parquet preference
- Optimized Parquet operations via KlinesParquetManager
- Lightweight JSON and Pickle serialization
- Fallback mechanisms for compatibility
- Enhanced error handling and logging

Usage:
    # Automatic format detection (prefers Parquet for DataFrames)
    safe_serialize(data, "data.parquet")  # Will use Parquet
    safe_serialize(data, "data.pkl")      # Will use Pickle
    safe_serialize(data, "data.json")     # Will use JSON
    
    # Direct format specification
    save_data(data, "output.parquet", format="parquet")
    load_data("input.parquet")
"""

import json
import pickle
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Import kline_parquet for efficient Parquet operations
try:
    from .kline_parquet import KlinesParquetManager, StorageConfig, get_klines_manager
    KLINE_PARQUET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"KlineParquetManager not available: {e}")
    KLINE_PARQUET_AVAILABLE = False

class JSONSerializer:
    """JSON serialization utilities."""

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data as JSON."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional[Any]:
        """Load data from JSON."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")
            return None

class PickleSerializer:
    """Pickle serialization utilities."""

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data as pickle."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle: {e}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional[Any]:
        """Load data from pickle."""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle: {e}")
            return None

class ParquetSerializer:
    """Parquet serialization utilities using KlinesParquetManager for efficiency."""

    def __init__(self):
        """Initialize with KlinesParquetManager if available."""
        if KLINE_PARQUET_AVAILABLE:
            self.klines_manager = get_klines_manager()
        else:
            self.klines_manager = None

    def save(self, data: Any, filepath: str) -> bool:
        """Save data as parquet using KlinesParquetManager for efficiency."""
        try:
            import pandas as pd
            if not isinstance(data, pd.DataFrame):
                logger.error("ParquetSerializer only supports pandas DataFrames")
                return False

            if self.klines_manager and KLINE_PARQUET_AVAILABLE:
                # Use KlinesParquetManager for optimized storage
                # Extract metadata from filepath if possible
                path_parts = Path(filepath).parts
                symbol = "UNKNOWN"
                exchange = "UNKNOWN" 
                interval = "1m"
                
                # Try to extract symbol/exchange from path
                if len(path_parts) >= 3:
                    exchange = path_parts[-3] if path_parts[-3] != "data" else "UNKNOWN"
                    symbol = path_parts[-2] if path_parts[-2] != "klines" else "UNKNOWN"
                
                # Use optimized storage
                return self.klines_manager.store_klines(
                    data, symbol, exchange, interval, 
                    batch_id=f"serialization_{Path(filepath).stem}"
                )
            else:
                # Fallback to direct pandas parquet
                data.to_parquet(filepath)
                return True
        except Exception as e:
            logger.error(f"Failed to save parquet: {e}")
            return False

    def load(self, filepath: str) -> Optional[Any]:
        """Load data from parquet using KlinesParquetManager for efficiency."""
        try:
            if self.klines_manager and KLINE_PARQUET_AVAILABLE:
                # Try to load using KlinesParquetManager first
                try:
                    # Extract metadata from filepath
                    path_parts = Path(filepath).parts
                    symbol = "UNKNOWN"
                    exchange = "UNKNOWN"
                    interval = "1m"
                    
                    if len(path_parts) >= 3:
                        exchange = path_parts[-3] if path_parts[-3] != "data" else "UNKNOWN"
                        symbol = path_parts[-2] if path_parts[-2] != "klines" else "UNKNOWN"
                    
                    # Try to load using manager
                    df = self.klines_manager.load_klines(symbol, exchange, interval)
                    if not df.empty:
                        return df
                except Exception:
                    pass  # Fall back to direct pandas
                
            # Fallback to direct pandas parquet
            import pandas as pd
            return pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Failed to load parquet: {e}")
            return None

class UniversalSerializer:
    """Universal serialization that tries multiple formats with Parquet preference."""

    def __init__(self):
        self.serializers = {
            'json': JSONSerializer,
            'parquet': ParquetSerializer(),
            'pickle': PickleSerializer
        }

    def save(self, data: Any, filepath: str, format: str = 'auto') -> bool:
        """Save data with automatic format detection, preferring Parquet."""
        if format == 'auto':
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.parquet'):
                format = 'parquet'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            else:
                # Prefer Parquet over Pickle for better performance and compatibility
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    format = 'parquet'
                    # Update filepath to have .parquet extension
                    if not filepath.endswith('.parquet'):
                        filepath = str(Path(filepath).with_suffix('.parquet'))
                else:
                    format = 'pickle'  # fallback for non-DataFrame data

        serializer = self.serializers.get(format)
        if serializer:
            if format == 'parquet':
                return serializer.save(data, filepath)
            else:
                return serializer.save(data, filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

    def load(self, filepath: str) -> Optional[Any]:
        """Load data with automatic format detection."""
        if filepath.endswith('.json'):
            return JSONSerializer.load(filepath)
        elif filepath.endswith('.parquet'):
            return self.serializers['parquet'].load(filepath)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            return PickleSerializer.load(filepath)
        else:
            # Try parquet first, then pickle as fallback
            try:
                return self.serializers['parquet'].load(filepath)
            except Exception:
                return PickleSerializer.load(filepath)

def safe_serialize(data: Any, filepath: str, format: str = 'auto') -> bool:
    """Safely serialize data to file with error handling."""
    try:
        serializer = UniversalSerializer()
        return serializer.save(data, filepath, format)
    except Exception as e:
        logger.error(f"Failed to serialize data: {e}")
        return False

def safe_deserialize(filepath: str) -> Optional[Any]:
    """Safely deserialize data from file with error handling."""
    try:
        serializer = UniversalSerializer()
        return serializer.load(filepath)
    except Exception as e:
        logger.error(f"Failed to deserialize data: {e}")
        return None


def save_pickle(data: Any, filepath: str) -> bool:
    """Save data as pickle file."""
    return PickleSerializer.save(data, filepath)


def load_pickle(filepath: str) -> Optional[Any]:
    """Load data from pickle file."""
    return PickleSerializer.load(filepath)


def save_parquet(data: Any, filepath: str) -> bool:
    """Save data as parquet file using optimized KlinesParquetManager."""
    serializer = ParquetSerializer()
    return serializer.save(data, filepath)


def load_parquet(filepath: str) -> Optional[Any]:
    """Load data from parquet file using optimized KlinesParquetManager."""
    serializer = ParquetSerializer()
    return serializer.load(filepath)


def save_json(data: Any, filepath: str) -> bool:
    """Save data as JSON file."""
    return JSONSerializer.save(data, filepath)


def load_json(filepath: str) -> Optional[Any]:
    """Load data from JSON file."""
    return JSONSerializer.load(filepath)


def save_data(data: Any, filepath: str, format: str = 'auto') -> bool:
    """Save data with automatic format detection, preferring Parquet."""
    serializer = UniversalSerializer()
    return serializer.save(data, filepath, format)


def load_data(filepath: str) -> Optional[Any]:
    """Load data with automatic format detection."""
    serializer = UniversalSerializer()
    return serializer.load(filepath)
