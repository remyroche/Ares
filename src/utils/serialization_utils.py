"""
Serialization Utilities Module

This module provides various serialization utilities for data persistence.
"""

import json
import pickle
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import pandas for Parquet operations
try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas not available. Parquet serialization will not work.")
    pd = None

class SerializationError(Exception):
    """Custom exception for serialization errors."""
    pass

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
    """Parquet serialization utilities."""

    @staticmethod
    def save(data: Any, filepath: str) -> bool:
        """Save data as parquet."""
        try:
            if pd is None:
                logger.error("Pandas not available. Cannot save parquet files.")
                return False
            if isinstance(data, pd.DataFrame):
                data.to_parquet(filepath)
                return True
            else:
                logger.error("ParquetSerializer only supports pandas DataFrames")
                return False
        except Exception as e:
            logger.error(f"Failed to save parquet: {e}")
            return False

    @staticmethod
    def load(filepath: str) -> Optional[Any]:
        """Load data from parquet."""
        try:
            if pd is None:
                logger.error("Pandas not available. Cannot load parquet files.")
                return None
            return pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Failed to load parquet: {e}")
            return None

class UniversalSerializer:
    """Universal serialization that tries multiple formats."""

    def __init__(self):
        self.serializers = {
            'json': JSONSerializer,
            'pickle': PickleSerializer,
            'parquet': ParquetSerializer
        }

    def save(self, data: Any, filepath: str, format: str = 'auto') -> bool:
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
            return serializer.save(data, filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

    def load(self, filepath: str) -> Optional[Any]:
        """Load data with automatic format detection."""
        if filepath.endswith('.json'):
            return JSONSerializer.load(filepath)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            return PickleSerializer.load(filepath)
        elif filepath.endswith('.parquet'):
            return ParquetSerializer.load(filepath)
        else:
            # Try pickle as default
            return PickleSerializer.load(filepath)

# Convenience functions for pickle operations
def save_pickle(data: Any, filepath: str) -> bool:
    """Save data as pickle file."""
    try:
        return PickleSerializer.save(data, filepath)
    except Exception as e:
        logger.error(f"Failed to save pickle: {e}")
        return False


def load_pickle(filepath: str) -> Optional[Any]:
    """Load data from pickle file."""
    try:
        return PickleSerializer.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load pickle: {e}")
        return None


# Convenience functions for JSON operations
def save_json(data: Any, filepath: str) -> bool:
    """Save data as JSON file."""
    try:
        return JSONSerializer.save(data, filepath)
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """Load data from JSON file."""
    try:
        return JSONSerializer.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return None


# Convenience functions for parquet operations
def save_parquet(data: Any, filepath: str) -> bool:
    """Save data as parquet file."""
    try:
        return ParquetSerializer.save(data, filepath)
    except Exception as e:
        logger.error(f"Failed to save parquet: {e}")
        return False


def load_parquet(filepath: str) -> Optional[Any]:
    """Load data from parquet file."""
    try:
        return ParquetSerializer.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load parquet: {e}")
        return None


# Convenience functions for universal operations
def save_data(data: Any, filepath: str, format: str = 'auto') -> bool:
    """Save data with automatic format detection."""
    try:
        serializer = UniversalSerializer()
        return serializer.save(data, filepath, format)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        return False


def load_data(filepath: str) -> Optional[Any]:
    """Load data with automatic format detection."""
    try:
        serializer = UniversalSerializer()
        return serializer.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None
