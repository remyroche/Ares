"""
Serialization Utilities Module

This module provides various serialization utilities for data persistence.
Uses kline_parquet.py for optimized Parquet operations with advanced features:
- Efficient compression and storage optimization
- Metadata management and data validation
- Batch management for incremental updates
- Comprehensive error handling and logging

Key Features:
- JSON, Pickle, and Parquet serialization support
- Automatic format detection
- Optimized Parquet operations via KlinesParquetManager
- Safe serialization with error handling
- Convenience functions for common operations
"""

import json
import pickle
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from .kline_parquet import KlinesParquetManager, StorageConfig
    KLINE_PARQUET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"kline_parquet not available: {e}. Using fallback Parquet implementation.")
    KlinesParquetManager = None
    StorageConfig = None
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
    """Parquet serialization utilities using KlinesParquetManager when available."""

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize with optional storage configuration."""
        if KLINE_PARQUET_AVAILABLE and KlinesParquetManager:
            self.manager = KlinesParquetManager(config)
        else:
            self.manager = None
            logger.warning("Using fallback Parquet implementation - kline_parquet not available")

    def save(self, data: Any, filepath: str, symbol: str = "UNKNOWN", 
             exchange: str = "unknown", interval: str = "1m") -> bool:
        """Save data as parquet using optimized kline_parquet manager or fallback."""
        try:
            import pandas as pd
            if not isinstance(data, pd.DataFrame):
                logger.error("ParquetSerializer only supports pandas DataFrames")
                return False
            
            if self.manager and KLINE_PARQUET_AVAILABLE:
                # Use the optimized kline_parquet manager for storage
                return self.manager.store_klines(data, symbol, exchange, interval)
            else:
                # Fallback to basic pandas parquet save
                data.to_parquet(filepath)
                return True
        except ImportError as e:
            logger.error(f"Pandas not available for Parquet operations: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save parquet: {e}")
            return False

    def load(self, filepath: str, symbol: str = "UNKNOWN", 
             exchange: str = "unknown", interval: str = "1m") -> Optional[Any]:
        """Load data from parquet using optimized kline_parquet manager or fallback."""
        try:
            import pandas as pd
            
            if self.manager and KLINE_PARQUET_AVAILABLE:
                # Try to load using kline_parquet manager first
                df = self.manager.load_klines(symbol, exchange, interval)
                if not df.empty:
                    return df
            
            # Fallback to direct pandas read
            return pd.read_parquet(filepath)
        except ImportError as e:
            logger.error(f"Pandas not available for Parquet operations: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load parquet: {e}")
            return None

class UniversalSerializer:
    """Universal serialization that tries multiple formats."""

    def __init__(self, parquet_config: Optional[StorageConfig] = None):
        self.serializers = {
            'json': JSONSerializer,
            'pickle': PickleSerializer,
            'parquet': ParquetSerializer(parquet_config)
        }

    def save(self, data: Any, filepath: str, format: str = 'auto', 
             symbol: str = "UNKNOWN", exchange: str = "unknown", interval: str = "1m") -> bool:
        """Save data with automatic format detection."""
        if format == 'auto':
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            elif filepath.endswith('.parquet'):
                format = 'parquet'
            else:
                format = 'pickle'  # default

        serializer = self.serializers.get(format)
        if serializer:
            if format == 'parquet':
                return serializer.save(data, filepath, symbol, exchange, interval)
            else:
                return serializer.save(data, filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

    def load(self, filepath: str, symbol: str = "UNKNOWN", 
             exchange: str = "unknown", interval: str = "1m") -> Optional[Any]:
        """Load data with automatic format detection."""
        if filepath.endswith('.json'):
            return JSONSerializer.load(filepath)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            return PickleSerializer.load(filepath)
        elif filepath.endswith('.parquet'):
            return self.serializers['parquet'].load(filepath, symbol, exchange, interval)
        else:
            # Try pickle as default
            return PickleSerializer.load(filepath)

def safe_serialize(data: Any, filepath: str, format: str = 'auto', 
                   symbol: str = "UNKNOWN", exchange: str = "unknown", 
                   interval: str = "1m", parquet_config: Optional[StorageConfig] = None) -> bool:
    """Safely serialize data to file with error handling."""
    try:
        serializer = UniversalSerializer(parquet_config)
        return serializer.save(data, filepath, format, symbol, exchange, interval)
    except Exception as e:
        logger.error(f"Failed to serialize data: {e}")
        return False

def safe_deserialize(filepath: str, symbol: str = "UNKNOWN", 
                     exchange: str = "unknown", interval: str = "1m",
                     parquet_config: Optional[StorageConfig] = None) -> Optional[Any]:
    """Safely deserialize data from file with error handling."""
    try:
        serializer = UniversalSerializer(parquet_config)
        return serializer.load(filepath, symbol, exchange, interval)
    except Exception as e:
        logger.error(f"Failed to deserialize data: {e}")
        return None


def save_pickle(data: Any, filepath: str) -> bool:
    """Save data as pickle file."""
    return PickleSerializer.save(data, filepath)


def load_pickle(filepath: str) -> Optional[Any]:
    """Load data from pickle file."""
    return PickleSerializer.load(filepath)


# Parquet convenience functions using kline_parquet
def save_parquet(data: Any, filepath: str, symbol: str = "UNKNOWN", 
                 exchange: str = "unknown", interval: str = "1m",
                 config: Optional[StorageConfig] = None) -> bool:
    """Save data as parquet file using optimized kline_parquet manager."""
    try:
        serializer = ParquetSerializer(config)
        return serializer.save(data, filepath, symbol, exchange, interval)
    except Exception as e:
        logger.error(f"Failed to save parquet: {e}")
        return False


def load_parquet(filepath: str, symbol: str = "UNKNOWN", 
                 exchange: str = "unknown", interval: str = "1m",
                 config: Optional[StorageConfig] = None) -> Optional[Any]:
    """Load data from parquet file using optimized kline_parquet manager."""
    try:
        serializer = ParquetSerializer(config)
        return serializer.load(filepath, symbol, exchange, interval)
    except Exception as e:
        logger.error(f"Failed to load parquet: {e}")
        return None


def get_parquet_manager(config: Optional[StorageConfig] = None):
    """Get a KlinesParquetManager instance for advanced operations."""
    if KLINE_PARQUET_AVAILABLE and KlinesParquetManager:
        return KlinesParquetManager(config)
    else:
        logger.warning("KlinesParquetManager not available - returning None")
        return None
