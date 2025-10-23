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
            import pandas as pd
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
