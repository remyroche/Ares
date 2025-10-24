"""
Serialization Utilities Module

This module provides various serialization utilities for data persistence.
Uses intelligent format selection: JSON for simple data, Parquet for large tabular data, Pickle for complex objects.

Key Features:
- Intelligent format selection based on data characteristics
- Optimized Parquet operations via KlinesParquetManager for large DataFrames
- JSON for simple, serializable data
- Pickle for complex Python objects and small datasets
- Backward compatibility for all formats
- Enhanced error handling and logging

Format Selection Logic:
- JSON: Simple data (strings, numbers, basic lists/dicts)
- Parquet: Large DataFrames (>1000 rows) or large convertible data
- Pickle: Complex Python objects, small DataFrames, functions, custom classes

Usage:
    # Intelligent format detection
    safe_serialize(simple_dict, "data.json")      # Will use JSON
    safe_serialize(large_df, "data.parquet")      # Will use Parquet
    safe_serialize(complex_obj, "data.pkl")       # Will use Pickle
    safe_serialize(data, "data")                  # Auto-selects best format
    
    # Direct format specification
    save_data(data, "output.parquet", format="parquet")
    load_data("input.parquet")  # Can load all formats
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
            
            # Convert non-DataFrame data to DataFrame if possible
            if not isinstance(data, pd.DataFrame):
                try:
                    # Try to convert to DataFrame
                    if isinstance(data, (list, tuple)):
                        data = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        data = pd.DataFrame([data])
                    elif hasattr(data, '__dict__'):
                        # Convert object to DataFrame
                        data = pd.DataFrame([data.__dict__])
                    else:
                        # Try to convert using pandas
                        data = pd.DataFrame([data])
                    logger.info(f"Converted non-DataFrame data to DataFrame for Parquet storage")
                except Exception as e:
                    logger.error(f"Could not convert data to DataFrame: {e}")
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
        """Save data with intelligent format detection based on data type."""
        if format == 'auto':
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.parquet'):
                format = 'parquet'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'  # Respect explicit .pkl/.pickle extensions
            else:
                # Intelligent format selection based on data type
                format = self._select_optimal_format(data, filepath)

        serializer = self.serializers.get(format)
        if serializer:
            if format == 'parquet':
                return serializer.save(data, filepath)
            else:
                return serializer.save(data, filepath)
        else:
            logger.error(f"Unsupported format: {format}")
            return False

    def _select_optimal_format(self, data: Any, filepath: str) -> str:
        """Select the optimal format based on data characteristics."""
        try:
            import pandas as pd
            pandas_available = True
        except ImportError:
            pandas_available = False
        
        # Always use JSON for simple, serializable data
        if self._is_json_serializable(data):
            return 'json'
        
        # Use Parquet for DataFrames and large tabular data
        if pandas_available and isinstance(data, pd.DataFrame):
            if len(data) > 1000:  # Large datasets benefit from Parquet
                return 'parquet'
            else:
                return 'pickle'  # Small DataFrames are better with Pickle
        
        # Use Pickle for complex Python objects
        if self._is_complex_python_object(data):
            return 'pickle'
        
        # Use Parquet for large data structures that can be converted to DataFrame
        if self._is_large_convertible_data(data):
            return 'parquet'
        
        # Default to Pickle for everything else
        return 'pickle'
    
    def _is_json_serializable(self, data: Any) -> bool:
        """Check if data is JSON serializable."""
        try:
            json.dumps(data)
            return True
        except (TypeError, ValueError):
            return False
    
    def _is_complex_python_object(self, data: Any) -> bool:
        """Check if data is a complex Python object that should use Pickle."""
        # Check for complex objects that Pickle handles well
        if hasattr(data, '__dict__') and not isinstance(data, (str, int, float, bool)):
            return True
        if isinstance(data, (list, dict, tuple, set)) and len(str(data)) < 1000:
            return True
        if callable(data) or hasattr(data, '__call__'):
            return True
        return False
    
    def _is_large_convertible_data(self, data: Any) -> bool:
        """Check if data is large and can be converted to DataFrame."""
        try:
            import pandas as pd
            if isinstance(data, (list, dict)) and len(str(data)) > 1000:
                # Try to convert to DataFrame to see if it's feasible
                if isinstance(data, list) and len(data) > 0:
                    pd.DataFrame(data)
                    return True
                elif isinstance(data, dict) and len(data) > 10:
                    pd.DataFrame([data])
                    return True
        except Exception:
            pass
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
    """Safely serialize data to file with intelligent format selection."""
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
    """Save data with intelligent format selection based on data characteristics."""
    serializer = UniversalSerializer()
    return serializer.save(data, filepath, format)


def load_data(filepath: str) -> Optional[Any]:
    """Load data with automatic format detection."""
    serializer = UniversalSerializer()
    return serializer.load(filepath)


def save_pickle_as_parquet(data: Any, filepath: str) -> bool:
    """Save data as Parquet file (replaces Pickle functionality)."""
    # Convert .pkl extension to .parquet if needed
    if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
        filepath = str(Path(filepath).with_suffix('.parquet'))
    
    serializer = ParquetSerializer()
    return serializer.save(data, filepath)
