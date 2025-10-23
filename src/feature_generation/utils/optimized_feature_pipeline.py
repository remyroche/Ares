"""
Optimized Feature Engineering Pipeline

This module provides a unified, hardware-optimized feature engineering pipeline
that ensures full compatibility between the feature bank, normalizer, and scaler
components with maximum vectorization and hardware utilization.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import hashlib
import gc
import psutil
import tempfile
import os
import shutil
from pathlib import Path

# Core imports (always needed)
from ..core.feature_bank import FeatureBank, FeatureBankConfig, get_global_feature_bank

# Lazy imports for optional components
def _lazy_import_intensity_scaler():
    from src.utils.intensity_scaler import get_intensity_config, apply_intensity_scaling
    return get_intensity_config, apply_intensity_scaling

def _lazy_import_matrix_operations():
    from src.utils.matrix_operations import get_unified_matrix_operations
    return get_unified_matrix_operations

def _lazy_import_hardware_manager():
    from src.utils.hardware.unified_hardware_manager import get_unified_hardware_manager, WorkloadType, OptimizationLevel
    return get_unified_hardware_manager, WorkloadType, OptimizationLevel

def _lazy_import_vectorized_core():
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    return get_vectorized_processing_core


def _lazy_import_incremental_processor():
    from .incremental_processor import get_incremental_processor, IncrementalConfig
    return get_incremental_processor, IncrementalConfig

def _lazy_import_intelligent_cache():
    try:
        from .intelligent_cache import get_feature_cache
        return get_feature_cache
    except ImportError as e:
        logger.warning(f"Intelligent cache not available: {e}")
        return None

def _lazy_import_rolling_optimizer():
    from .consolidated_rolling_optimizer import get_global_rolling_optimizer
    return get_global_rolling_optimizer

def _lazy_import_vectorbt_batcher():
    from .vectorbt_operation_batcher import VectorBTOperationBatcher
    return VectorBTOperationBatcher

logger = logging.getLogger(__name__)

class IntelligentFeatureCache:
    """Intelligent feature caching with dependency tracking, TTL, and disk persistence."""
    
    def __init__(self, ttl_seconds: int = 3600, enable_disk_cache: bool = True, cache_dir: str = None):
        self.cache = {}
        self.dependencies = {}
        self.data_hashes = {}
        self.timestamps = {}
        self.ttl_seconds = ttl_seconds
        self.enable_disk_cache = enable_disk_cache
        self.cache_dir = cache_dir or tempfile.gettempdir() / Path("ares_feature_cache")
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # LRU cache for rolling operations
        from functools import lru_cache
        self.rolling_cache = {}
        self.rolling_cache_maxsize = 100
    
    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """Generate a fast, stable hash for the data to detect changes."""
        try:
            # Use pandas hash_pandas_object for faster, more stable fingerprinting
            if hasattr(pd.util, 'hash_pandas_object'):
                # Use first/last N rows + index range for stability
                if len(data) > 0:
                    # Sample first 5 and last 5 rows for hash
                    sample_data = pd.concat([data.head(5), data.tail(5)])
                    hash_value = pd.util.hash_pandas_object(sample_data).sum()
                else:
                    hash_value = 0
                
                # Include shape and index range for additional stability
                index_hash = hash((data.shape, data.index.min(), data.index.max()))
                # Include column name fingerprint to disambiguate identical-shape frames
                try:
                    cols_fingerprint = hash(tuple(list(data.columns)[:10] + list(data.columns)[-10:]))
                except Exception:
                    cols_fingerprint = 0
                return str(hash_value + index_hash + cols_fingerprint)
            else:
                # Fallback to original method but with better sampling
                if len(data) > 0:
                    step = max(1, len(data) // 50)  # More frequent sampling
                    data_sample = data.iloc[::step]
                else:
                    data_sample = data
                try:
                    col_head = list(data.columns)[:10]
                    col_tail = list(data.columns)[-10:]
                except Exception:
                    col_head, col_tail = [], []
                hash_input = f"{data.shape}_{data.index.tolist()[:5]}_{col_head}_{col_tail}_{data_sample.values.tobytes()}"
                return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            # Ultimate fallback
            return hashlib.md5(f"{data.shape}_{len(data)}".encode()).hexdigest()
    
    def get_cached_features(self, data: pd.DataFrame, feature_specs: List[str]) -> Optional[pd.DataFrame]:
        """Get cached features if available and not expired."""
        data_hash = self._generate_data_hash(data)
        # Use a stable, order-independent fingerprint for feature specs (avoid built-in hash randomness)
        try:
            spec_fingerprint = hashlib.md5("||".join(sorted(map(str, feature_specs))).encode()).hexdigest()
        except Exception:
            # Fallback in case of unexpected types
            spec_fingerprint = hashlib.md5(str(sorted(feature_specs)).encode()).hexdigest()
        cache_key = f"{data_hash}_{spec_fingerprint}"
        
        # Check memory cache first
        if cache_key in self.cache:
            # Check TTL
            if time.time() - self.timestamps.get(cache_key, 0) < self.ttl_seconds:
                logger.debug(f"✅ Memory cache hit for {len(feature_specs)} features")
                cached_data = self.cache[cache_key]
                # Ensure we return a DataFrame, not a FeatureResult
                if hasattr(cached_data, 'data'):
                    return cached_data.data.copy()
                else:
                    return cached_data.copy()
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
                if cache_key in self.timestamps:
                    del self.timestamps[cache_key]
        
        # Check disk cache if enabled
        if self.enable_disk_cache:
            disk_cache_path = self.cache_dir / f"{cache_key}.parquet"
            if disk_cache_path.exists():
                try:
                    # Check if file is recent enough
                    file_age = time.time() - disk_cache_path.stat().st_mtime
                    if file_age < self.ttl_seconds:
                        logger.debug(f"✅ Disk cache hit for {len(feature_specs)} features")
                        cached_data = pd.read_parquet(disk_cache_path)
                        # Store in memory cache for faster access
                        self.cache[cache_key] = cached_data.copy()
                        self.timestamps[cache_key] = time.time()
                        return cached_data
                except Exception as e:
                    logger.warning(f"Failed to read disk cache: {e}")
        
        return None
    
    def cache_features(self, data: pd.DataFrame, feature_specs: List[str], features):
        """Cache features with dependency tracking and disk persistence."""
        data_hash = self._generate_data_hash(data)
        # Use the same stable, order-independent fingerprint for cache key
        try:
            spec_fingerprint = hashlib.md5("||".join(sorted(map(str, feature_specs))).encode()).hexdigest()
        except Exception:
            spec_fingerprint = hashlib.md5(str(sorted(feature_specs)).encode()).hexdigest()
        cache_key = f"{data_hash}_{spec_fingerprint}"
        
        # Extract actual feature data - handle various input types
        feature_data = None
        
        if isinstance(features, pd.DataFrame):
            # It's already a DataFrame
            feature_data = features
        elif hasattr(features, 'data'):
            # It's a FeatureResult object, extract the data
            if isinstance(features.data, pd.DataFrame):
                feature_data = features.data
            elif isinstance(features.data, pd.Series):
                feature_data = pd.DataFrame({features.name: features.data})
            else:
                logger.warning(f"FeatureResult data is not a DataFrame or Series: {type(features.data)}")
                return
        elif isinstance(features, list):
            # It's a list of FeatureResult objects
            feature_dict = {}
            for item in features:
                if hasattr(item, 'name') and hasattr(item, 'data'):
                    if isinstance(item.data, pd.Series):
                        feature_dict[item.name] = item.data
                    elif isinstance(item.data, pd.DataFrame):
                        if len(item.data.columns) == 1:
                            feature_dict[item.name] = item.data.iloc[:, 0]
                        else:
                            for col in item.data.columns:
                                feature_dict[f"{item.name}_{col}"] = item.data[col]
            if feature_dict:
                feature_data = pd.DataFrame(feature_dict)
        else:
            # Try to convert to DataFrame
            try:
                feature_data = pd.DataFrame(features)
            except Exception as e:
                logger.warning(f"Could not convert features to DataFrame: {e}")
                return
        
        if feature_data is None or feature_data.empty:
            logger.warning("No valid feature data to cache")
            return
        
        # Store in memory cache
        self.cache[cache_key] = feature_data.copy()
        self.timestamps[cache_key] = time.time()
        
        # Store in disk cache if enabled
        if self.enable_disk_cache:
            try:
                disk_cache_path = self.cache_dir / f"{cache_key}.parquet"
                # Ensure we only serialize the DataFrame data, not complex objects
                if isinstance(feature_data, pd.DataFrame):
                    # Clean the DataFrame to ensure it's serializable
                    clean_data = feature_data.copy()
                    # Convert any object columns that might contain non-serializable data
                    for col in clean_data.columns:
                        if clean_data[col].dtype == 'object':
                            # Try to convert to numeric if possible, otherwise skip
                            try:
                                clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
                            except:
                                # If conversion fails, drop the column
                                logger.warning(f"Dropping non-serializable column {col} from disk cache")
                                clean_data = clean_data.drop(columns=[col])
                    
                    clean_data.to_parquet(disk_cache_path, compression='snappy')
                    logger.debug(f"✅ Cached {len(feature_specs)} features to disk")
                else:
                    logger.warning("Feature data is not a DataFrame, skipping disk cache")
            except Exception as e:
                logger.warning(f"Failed to write disk cache: {e}")
        
        # Track dependencies
        for spec in feature_specs:
            if spec not in self.dependencies:
                self.dependencies[spec] = set()
            self.dependencies[spec].add(cache_key)
        
        logger.debug(f"💾 Cached {len(feature_specs)} features")
    
    def get_rolling_cache_key(self, series_id: str, operation: str, window: int, min_periods: int = None) -> str:
        """Generate cache key for rolling operations."""
        return f"{series_id}_{operation}_{window}_{min_periods or window}"
    
    def get_cached_rolling_result(self, series_id: str, operation: str, window: int, min_periods: int = None):
        """Get cached rolling operation result."""
        cache_key = self.get_rolling_cache_key(series_id, operation, window, min_periods)
        return self.rolling_cache.get(cache_key)
    
    def cache_rolling_result(self, series_id: str, operation: str, window: int, result, min_periods: int = None):
        """Cache rolling operation result with LRU eviction."""
        cache_key = self.get_rolling_cache_key(series_id, operation, window, min_periods)
        
        # Implement simple LRU eviction
        if len(self.rolling_cache) >= self.rolling_cache_maxsize:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.rolling_cache))
            del self.rolling_cache[oldest_key]
        
        self.rolling_cache[cache_key] = result
    
    def invalidate_dependencies(self, changed_specs: List[str]):
        """Invalidate cache entries that depend on changed specifications."""
        for spec in changed_specs:
            if spec in self.dependencies:
                for cache_key in self.dependencies[spec]:
                    if cache_key in self.cache:
                        del self.cache[cache_key]
                    if cache_key in self.timestamps:
                        del self.timestamps[cache_key]
                del self.dependencies[spec]
    
    def cleanup_expired(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
        
        if expired_keys:
            logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear_old_entries(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Clear old cache entries to free up memory.
        
        Args:
            max_age_seconds: Maximum age of entries to keep (None for default TTL)
            
        Returns:
            Number of entries cleared
        """
        try:
            if max_age_seconds is None:
                max_age_seconds = self.ttl_seconds
            
            current_time = time.time()
            expired_keys = []
            
            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > max_age_seconds:
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                self.cache.pop(key, None)
                self.dependencies.pop(key, None)
                self.data_hashes.pop(key, None)
                self.timestamps.pop(key, None)
            
            if expired_keys:
                logger.debug(f"🧹 Cleared {len(expired_keys)} old cache entries")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Error clearing old entries: {e}")
            return 0

class MemoryMappedDataFrame:
    """Memory-mapped DataFrame for large dataset handling."""
    
    def __init__(self, data: pd.DataFrame, temp_dir: Optional[str] = None, 
                 compression: str = "lz4", sparse_threshold: float = 0.5):
        self.original_data = data
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.compression = compression
        self.sparse_threshold = sparse_threshold
        self.memory_mapped_files = {}
        self.memory_mapped_dirs = set()
        self.is_memory_mapped = False
        
    def _get_data_size_mb(self, data: pd.DataFrame) -> float:
        """Calculate DataFrame size in MB."""
        return data.memory_usage(deep=True).sum() / (1024 * 1024)
    
    def _should_use_memory_mapping(self, data: pd.DataFrame, threshold_mb: int) -> bool:
        """Determine if memory mapping should be used."""
        # Check data size first
        if self._get_data_size_mb(data) <= threshold_mb:
            return False
        
        # Check for sparse data that might cause memory mapping issues
        sparse_columns = []
        
        # Define datetime-related column patterns that should not be considered sparse
        datetime_patterns = ['day', 'hour', 'day_of_week', 'month', 'year', 'weekday', 'time']
        
        for col in data.columns:
            col_data = data[col]
            nan_ratio = col_data.isna().sum() / len(col_data) if len(col_data) > 0 else 0
            
            # Skip datetime-related columns from sparse detection
            is_datetime_column = any(pattern in col.lower() for pattern in datetime_patterns)
            if is_datetime_column:
                continue
            
            # Check for boolean columns with high NaN ratio (common with sparse features)
            # Lower threshold for boolean columns as they're more problematic for memory mapping
            if col_data.dtype == 'bool' and nan_ratio > 0.1:  # Reduced from 0.3 to 0.1
                sparse_columns.append(col)
            
            # Check for object/category columns with high NaN ratio
            elif col_data.dtype in ['object', 'category'] and nan_ratio > 0.5:
                sparse_columns.append(col)
            
            # Check for numeric columns with very high NaN ratio
            elif col_data.dtype in ['float64', 'float32', 'int64', 'int32'] and nan_ratio > 0.7:
                sparse_columns.append(col)
            
            # Check for columns with very low variance (near-constant values)
            # But be more lenient for datetime features and integer columns
            elif (col_data.dtype in ['float64', 'float32'] and nan_ratio < 0.1 and 
                  col_data.dtype not in ['int64', 'int32', 'int16', 'int8']):
                try:
                    if len(col_data) > 0 and col_data.nunique() / len(col_data) < 0.01:  # Less than 1% unique values
                        sparse_columns.append(col)
                except:
                    pass  # Skip if calculation fails
        
        if sparse_columns:
            logger.info(f"💡 Detected sparse columns: {sparse_columns}")
            logger.info("💡 Skipping memory mapping due to sparse data - using in-memory processing")
            return False
        
        return True
    
    def _optimize_for_memory_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory mapping."""
        optimized_data = data.copy()
        
        # Convert to most efficient dtypes
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            if optimized_data[col].dtype == np.float64:
                if (optimized_data[col].max() < np.finfo(np.float32).max and
                    optimized_data[col].min() > np.finfo(np.float32).min):
                    optimized_data[col] = optimized_data[col].astype(np.float32)
            elif optimized_data[col].dtype == np.int64:
                if (optimized_data[col].max() < np.iinfo(np.int32).max and
                    optimized_data[col].min() > np.iinfo(np.int32).min):
                    optimized_data[col] = optimized_data[col].astype(np.int32)
        
        # Handle sparse data - for memory mapping, we need to avoid sparse arrays
        # as Parquet doesn't support them well. Instead, fill NaN values with appropriate defaults.
        datetime_patterns = ['day', 'hour', 'day_of_week', 'month', 'year', 'weekday', 'time']
        
        for col in optimized_data.columns:
            col_data = optimized_data[col]
            
            # Skip datetime-related columns from sparse optimization
            is_datetime_column = any(pattern in col.lower() for pattern in datetime_patterns)
            if is_datetime_column:
                # For datetime columns, just ensure they're properly typed
                if col_data.dtype == 'object':
                    # Try to convert to appropriate numeric type if it's a datetime feature
                    try:
                        if col_data.dropna().dtype in ['int64', 'int32']:
                            optimized_data[col] = col_data.astype('int32')
                    except:
                        pass
                continue
            
            # Check for various types of sparsity
            nan_ratio = col_data.isna().sum() / len(optimized_data) if len(optimized_data) > 0 else 0
            is_numeric_or_bool = (
                pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data)
            )
            zero_ratio = (col_data == 0).sum() / len(optimized_data) if is_numeric_or_bool and len(optimized_data) > 0 else 0
            empty_ratio = (col_data == '').sum() / len(optimized_data) if col_data.dtype == 'object' and len(optimized_data) > 0 else 0
            
            # Consider data sparse if it has high NaN ratio OR high zero/empty ratio
            is_sparse = nan_ratio > self.sparse_threshold or (zero_ratio + empty_ratio) > self.sparse_threshold
            
            if is_sparse:
                logger.debug(f"🔍 Column '{col}' is sparse: NaN={nan_ratio:.2%}, Zero={zero_ratio:.2%}, Empty={empty_ratio:.2%}")
                
                # Special handling for boolean columns - convert to int8 for better memory mapping
                if col_data.dtype == 'bool':
                    # Convert boolean to int8 (0/1) for better parquet compatibility
                    optimized_data[col] = col_data.fillna(False).astype('int8')
                    logger.debug(f"🔧 Converted boolean column '{col}' to int8 for memory mapping compatibility")
                elif col_data.dtype in [np.float32, np.float64]:
                    optimized_data[col] = col_data.fillna(0.0)
                elif col_data.dtype in [np.int32, np.int64]:
                    optimized_data[col] = col_data.fillna(0)
                elif col_data.dtype == 'object':
                    optimized_data[col] = col_data.fillna('')
                else:
                    # For any other dtype, try to fill with appropriate default
                    if col_data.dtype.kind in ['f']:  # float
                        optimized_data[col] = col_data.fillna(0.0)
                    elif col_data.dtype.kind in ['i', 'u']:  # integer
                        optimized_data[col] = col_data.fillna(0)
                    else:
                        optimized_data[col] = col_data.fillna('')
        
        return optimized_data
    
    def _convert_sparse_booleans(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert sparse boolean columns to int8 for better memory mapping compatibility."""
        converted_data = data.copy()
        converted_columns = []
        
        for col in converted_data.columns:
            col_data = converted_data[col]
            
            # Enhanced boolean column detection and conversion
            if col_data.dtype == 'bool':
                # Check for NaN values or sparse data patterns
                has_nan = col_data.isna().any()
                is_sparse = col_data.isna().sum() / len(col_data) > 0.01  # 1% threshold
                
                if has_nan or is_sparse:
                    # Convert to int8 (0/1) with False as default for NaN
                    converted_data[col] = col_data.fillna(False).astype('int8')
                    converted_columns.append(col)
                    
                    # Log detailed conversion info
                    nan_count = col_data.isna().sum()
                    total_count = len(col_data)
                    logger.debug(f"🔧 Converted boolean column '{col}' to int8 (NaN: {nan_count}/{total_count})")
            
            # Also handle object columns that might contain boolean-like values
            elif col_data.dtype == 'object' and col in ['is_weekend', 'is_market_hours', 'is_trading_session']:
                # Convert object boolean columns to int8
                try:
                    # Try to convert to boolean first, then to int8
                    bool_series = pd.to_numeric(col_data, errors='coerce').fillna(0).astype(bool)
                    converted_data[col] = bool_series.astype('int8')
                    converted_columns.append(col)
                    logger.debug(f"🔧 Converted object boolean column '{col}' to int8")
                except Exception as e:
                    logger.debug(f"⚠️ Could not convert object column '{col}' to boolean: {e}")
        
        if converted_columns:
            logger.info(f"🔧 Converted {len(converted_columns)} boolean/object columns to int8: {converted_columns}")
        
        return converted_data
    
    def _detect_sparse_columns(self, data: pd.DataFrame) -> list[str]:
        """Detect columns that might cause memory mapping issues."""
        sparse_columns = []
        
        for col in data.columns:
            col_data = data[col]
            
            # Check for various sparse patterns
            if col_data.dtype == 'bool':
                # Boolean columns with NaN values
                if col_data.isna().any():
                    sparse_columns.append(col)
            elif col_data.dtype == 'object':
                # Object columns that might contain mixed types
                if col in ['is_weekend', 'is_market_hours', 'is_trading_session']:
                    sparse_columns.append(col)
            elif col_data.dtype in [np.float64, np.float32]:
                # Float columns with high NaN ratio
                nan_ratio = col_data.isna().sum() / len(col_data)
                if nan_ratio > 0.05:  # 5% threshold
                    sparse_columns.append(col)
        
        return sparse_columns

    def _preprocess_for_memory_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data to maximize memory mapping compatibility."""
        processed_data = data.copy()
        
        # Convert all boolean columns to int8 for better parquet compatibility
        boolean_columns = processed_data.select_dtypes(include=['bool']).columns
        for col in boolean_columns:
            processed_data[col] = processed_data[col].fillna(False).astype('int8')
        
        # Handle object columns that might contain boolean-like values
        object_boolean_cols = ['is_weekend', 'is_market_hours', 'is_trading_session', 'is_business_day']
        for col in object_boolean_cols:
            if col in processed_data.columns:
                try:
                    # Convert to boolean first, then to int8
                    processed_data[col] = processed_data[col].fillna(False).astype(bool).astype('int8')
                except Exception:
                    # If conversion fails, leave as is
                    pass
        
        # Fill NaN values in numeric columns with appropriate defaults
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if processed_data[col].isna().any():
                if processed_data[col].dtype in [np.float64, np.float32]:
                    processed_data[col] = processed_data[col].fillna(0.0)
                elif processed_data[col].dtype in [np.int64, np.int32, np.int16, np.int8]:
                    processed_data[col] = processed_data[col].fillna(0)
        
        logger.debug(f"🔧 Preprocessed {len(boolean_columns)} boolean columns and {len(numeric_columns)} numeric columns for memory mapping")
        return processed_data

    def create_memory_mapped(self, data: pd.DataFrame, threshold_mb: int = 100) -> pd.DataFrame:
        """Create a disk-backed (true NumPy memmap) DataFrame for numeric columns.

        This avoids Parquet round-trips and maps numeric columns to on-disk
        arrays. Non-numeric columns are kept in-memory.
        """
        if not self._should_use_memory_mapping(data, threshold_mb):
            return data

        try:
            # Preprocess and optimize for mapping
            preprocessed = self._preprocess_for_memory_mapping(data)
            optimized = self._optimize_for_memory_mapping(preprocessed)

            # Directory to store per-column memmaps
            mm_dir = os.path.join(self.temp_dir, f"memmap_{id(data)}")
            os.makedirs(mm_dir, exist_ok=True)
            self.memory_mapped_dirs.add(mm_dir)

            mapped_cols = {}
            for col in optimized.columns:
                col_data = optimized[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    # Create memmap file for numeric columns
                    safe_name = str(col).replace(os.sep, "_")
                    fpath = os.path.join(mm_dir, f"{safe_name}.mmap")
                    mm = np.memmap(fpath, dtype=col_data.dtype, mode='w+', shape=(len(col_data),))
                    np.copyto(mm, col_data.to_numpy(copy=False), casting='safe')
                    self.memory_mapped_files[fpath] = mm
                    mapped_cols[col] = pd.Series(mm, index=optimized.index, name=col)
                else:
                    # Keep non-numeric columns as-is
                    mapped_cols[col] = col_data

            mem_df = pd.DataFrame(mapped_cols, index=optimized.index)
            self.is_memory_mapped = True
            logger.info(f"💾 Created disk-backed memmap DataFrame with {len(self.memory_mapped_files)} mapped columns @ {mm_dir}")
            return mem_df

        except Exception as e:
            logger.warning(f"⚠️ Memmap creation failed, falling back to in-memory: {e}")
            return data
    
    def cleanup(self):
        """Clean up memory-mapped files."""
        for temp_file in list(self.memory_mapped_files.keys()):
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove memory-mapped file {temp_file}: {e}")
            finally:
                self.memory_mapped_files.pop(temp_file, None)

        # Remove created directories if empty
        for d in list(self.memory_mapped_dirs):
            try:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to remove memmap directory {d}: {e}")
            finally:
                self.memory_mapped_dirs.discard(d)

class AdvancedDataTypeOptimizer:
    """Advanced data type optimization with categorical encoding and compression."""
    
    def __init__(self, enable_categorical: bool = True, enable_compression: bool = True,
                 compression_algorithm: str = "lz4", sparse_threshold: float = 0.5):
        self.enable_categorical = enable_categorical
        self.enable_compression = enable_compression
        self.compression_algorithm = compression_algorithm
        self.sparse_threshold = sparse_threshold
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive data type optimization."""
        optimized_data = data.copy()
        
        # Basic data type optimization
        optimized_data = self._optimize_numeric_types(optimized_data)
        
        # Categorical encoding
        if self.enable_categorical:
            optimized_data = self._optimize_categorical_columns(optimized_data)
        
        # Sparse data optimization
        optimized_data = self._optimize_sparse_data(optimized_data)
        
        # String optimization
        optimized_data = self._optimize_string_columns(optimized_data)
        
        return optimized_data
    
    def _optimize_numeric_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize numeric data types with early downcasting and copy=False."""
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].dtype == np.float64:
                # Check if float32 is sufficient
                if (data[col].max() < np.finfo(np.float32).max and
                    data[col].min() > np.finfo(np.float32).min):
                    data[col] = data[col].astype(np.float32, copy=False)
            elif data[col].dtype == np.int64:
                # Check if int32 is sufficient
                if (data[col].max() < np.iinfo(np.int32).max and
                    data[col].min() > np.iinfo(np.int32).min):
                    data[col] = data[col].astype(np.int32, copy=False)
            elif data[col].dtype == np.int32:
                # Check if int16 is sufficient
                if (data[col].max() < np.iinfo(np.int16).max and
                    data[col].min() > np.iinfo(np.int16).min):
                    data[col] = data[col].astype(np.int16, copy=False)
                # Check if int8 is sufficient
                elif (data[col].max() < np.iinfo(np.int8).max and
                      data[col].min() > np.iinfo(np.int8).min):
                    data[col] = data[col].astype(np.int8, copy=False)
        
        return data
    
    def _optimize_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert string columns to categorical when beneficial."""
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].dtype == 'object':
                # Check if conversion to categorical is beneficial
                unique_ratio = data[col].nunique() / len(data[col])
                if unique_ratio < 0.5:  # Less than 50% unique values
                    data[col] = data[col].astype('category')
        
        return data
    
    def _optimize_sparse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize sparse data using sparse arrays."""
        for col in data.select_dtypes(include=[np.number]).columns:
            # Check if column is sparse
            zero_ratio = (data[col] == 0).sum() / len(data[col])
            nan_ratio = data[col].isna().sum() / len(data[col])
            
            if (zero_ratio + nan_ratio) > self.sparse_threshold:
                # Convert to sparse representation
                try:
                    from pandas.arrays import SparseArray
                    data[col] = SparseArray(data[col], fill_value=0)
                except ImportError:
                    # Fallback for older pandas versions
                    try:
                        data[col] = pd.SparseArray(data[col], fill_value=0)
                    except AttributeError:
                        # Skip sparse conversion if not available
                        pass
        
        return data
    
    def _optimize_string_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize string columns with vectorized conversions."""
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].dtype == 'object':
                # Check if all values are numeric strings
                try:
                    pd.to_numeric(data[col], errors='raise')
                    # Convert to numeric with downcast and copy=False when feasible
                    data[col] = pd.to_numeric(data[col], downcast='integer', copy=False)
                except (ValueError, TypeError):
                    # Keep as object or convert to category
                    if data[col].nunique() / len(data[col]) < 0.5:
                        data[col] = data[col].astype('category', copy=False)
        
        return data
    
    def get_memory_usage_reduction(self, original_data: pd.DataFrame, optimized_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate memory usage reduction."""
        original_memory = original_data.memory_usage(deep=True).sum()
        optimized_memory = optimized_data.memory_usage(deep=True).sum()
        
        reduction_mb = (original_memory - optimized_memory) / (1024 * 1024)
        reduction_percentage = (reduction_mb / (original_memory / (1024 * 1024))) * 100
        
        return {
            'original_memory_mb': original_memory / (1024 * 1024),
            'optimized_memory_mb': optimized_memory / (1024 * 1024),
            'reduction_mb': reduction_mb,
            'reduction_percentage': reduction_percentage
        }

@dataclass
class PipelineConfig:
    """Configuration for the optimized feature pipeline."""
    # Feature Bank Configuration
    enable_matrix_operations: bool = True
    enable_gpu_acceleration: bool = True
    enable_lookback_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 6  # Optimized for M1 with better load balancing
    chunk_size: int = 1000  # M1-optimized chunk size for better memory management
    memory_efficient: bool = True
    cache_results: bool = True

    # Normalization Configuration
    auto_normalize: bool = True
    normalization_method: str = "zscore"  # "zscore", "minmax", "robust", "quantile"
    normalization_exclude_categories: List[str] = field(default_factory=list)
    normalization_exclude_features: List[str] = field(default_factory=list)
    normalization_rolling_windows: List[int] = field(default_factory=lambda: [20, 50, 100])

    # Scaling Configuration
    enable_intensity_scaling: bool = True
    intensity_percentage: Optional[float] = None  # Auto-detect from environment

    # Hardware Optimization
    enable_hardware_optimization: bool = True
    workload_type: str = "feature_engineering"  # WorkloadType.FEATURE_ENGINEERING
    optimization_level: str = "balanced"  # OptimizationLevel.BALANCED

    # Performance Monitoring
    enable_performance_monitoring: bool = True
    enable_memory_tracking: bool = True
    enable_profiling: bool = False
    
    # Processing Strategy
    enable_incremental_processing: bool = True
    processing_strategy: str = "auto"  # "auto", "incremental", "traditional"
    
    # Performance Optimizations
    enable_aggressive_memory_cleanup: bool = True
    enable_feature_dependency_optimization: bool = True
    enable_lazy_loading: bool = True
    
    # Memory Optimization Settings
    enable_streaming_processing: bool = True
    streaming_chunk_size: int = 5000  # Process data in smaller chunks
    memory_threshold_mb: float = 1000.0  # Trigger cleanup when memory exceeds this
    enable_adaptive_chunking: bool = True  # Dynamically adjust chunk size based on memory
    enable_memory_pooling: bool = True  # Reuse memory blocks
    max_memory_usage_mb: float = 2000.0  # Hard limit for memory usage
    enable_m1_optimizations: bool = True
    enable_mps_acceleration: bool = True
    enable_vectorbt_batch_processing: bool = True
    vectorbt_chunk_size: int = 5000
    enable_intelligent_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_incremental_caching: bool = True
    
    # Rolling and VectorBT Optimizations
    enable_rolling_optimization: bool = True
    enable_vectorbt_optimization: bool = True
    
    # Memory-Mapped DataFrames & Data Type Optimization
    enable_memory_mapping: bool = True
    memory_mapping_threshold_mb: int = 50  # Use memory mapping for datasets > 50MB (more aggressive)
    enable_advanced_data_type_optimization: bool = True
    enable_categorical_encoding: bool = True
    enable_data_compression: bool = True
    compression_algorithm: str = "lz4"  # "lz4", "zstd", "gzip", "bz2"
    enable_sparse_data_optimization: bool = True
    sparse_threshold: float = 0.3  # Consider data sparse if >30% zeros/NaNs (more aggressive)

@dataclass
class PipelineResult:
    """Result from feature pipeline execution."""
    features: pd.DataFrame
    normalization_params: Dict[str, Any]
    scaling_params: Dict[str, Any]
    performance_stats: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    memory_usage: float = 0.0

class OptimizedFeaturePipeline:
    """
    Optimized feature engineering pipeline with full hardware acceleration
    and vectorization support.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the optimized feature pipeline."""
        self.config = config or PipelineConfig()
        self.logger = logger.getChild('OptimizedFeaturePipeline')

        # Initialize components
        self.feature_bank = None
        self.normalizer = None
        self.scaler = None
        self.matrix_ops = None
        self.hardware_manager = None
        self.vectorized_core = None
        
        # Initialize processing components
        self.incremental_processor = None
        
        # Initialize rolling optimizer (singleton)
        self.rolling_optimizer = None
        self.vectorbt_batcher = None
        
        # Initialize intelligent caching
        self.feature_cache = None
        if self.config.cache_results:
            if self.config.enable_intelligent_caching:
                self.feature_cache = IntelligentFeatureCache(self.config.cache_ttl_seconds)
                self.logger.info("✅ Intelligent feature cache initialized")
            else:
                try:
                    get_feature_cache = _lazy_import_intelligent_cache()
                    if get_feature_cache is not None:
                        self.feature_cache = get_feature_cache()
                        self.logger.info("✅ Intelligent feature cache initialized")
                    else:
                        self.logger.warning("Intelligent cache not available, caching disabled")
                except Exception as e:
                    self.logger.warning(f"Feature cache initialization failed: {e}")
        
        # Initialize memory-mapped DataFrame handler
        self.memory_mapped_handler = None
        if self.config.enable_memory_mapping:
            # Create a dummy DataFrame for initialization
            dummy_data = pd.DataFrame({'dummy': [1, 2, 3]})
            self.memory_mapped_handler = MemoryMappedDataFrame(
                data=dummy_data,
                compression=self.config.compression_algorithm,
                sparse_threshold=self.config.sparse_threshold
            )
            self.logger.info("✅ Memory-mapped DataFrame handler initialized")
        
        # Initialize advanced data type optimizer
        self.data_type_optimizer = None
        if self.config.enable_advanced_data_type_optimization:
            self.data_type_optimizer = AdvancedDataTypeOptimizer(
                enable_categorical=self.config.enable_categorical_encoding,
                enable_compression=self.config.enable_data_compression,
                compression_algorithm=self.config.compression_algorithm,
                sparse_threshold=self.config.sparse_threshold
            )
            self.logger.info("✅ Advanced data type optimizer initialized")

        # Performance tracking
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_processing_time': 0.0,
            'peak_memory_usage': 0.0,
            'hardware_accelerations': 0,
            'vectorized_operations': 0
        }
        
        # Recursion guard to prevent circular calls
        self._processing = False

        # Throttle memory checks to reduce overhead
        self._last_memory_check: float = 0.0

        # Initialize all components
        self._initialize_components()

        # Reduced verbosity - only log once per session
        if not hasattr(OptimizedFeaturePipeline, '_logged_initialization'):
            self.logger.info("✅ Optimized Feature Pipeline initialized")
            OptimizedFeaturePipeline._logged_initialization = True

    def _initialize_components(self):
        """Initialize all pipeline components with optimal configuration."""
        try:
            # Initialize Feature Bank with optimized configuration
            feature_bank_config = FeatureBankConfig(
                enable_matrix_operations=self.config.enable_matrix_operations,
                enable_gpu_acceleration=self.config.enable_gpu_acceleration,
                enable_lookback_optimization=self.config.enable_lookback_optimization,
                enable_parallel_processing=self.config.enable_parallel_processing,
                max_workers=self.config.max_workers,
                chunk_size=self.config.chunk_size,
                memory_efficient=self.config.memory_efficient,
                cache_results=self.config.cache_results
            )

            self.feature_bank = FeatureBank(feature_bank_config)
            self.logger.info("✅ Feature Bank initialized")

            # Initialize Normalizer - using lazy import to avoid circular dependencies
            from ...training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.data_normalization import (
                NormalizationConfig, NormalizationMethod, create_data_normalizer
            )

            normalization_config = NormalizationConfig(
                method=getattr(NormalizationMethod, self.config.normalization_method.upper(), NormalizationMethod.Z_SCORE),
                use_hardware_acceleration=self.config.enable_hardware_optimization,
                use_matrix_operations=self.config.enable_matrix_operations,
                batch_size=self.config.chunk_size,
                memory_limit_gb=8.0
            )

            self.normalizer = create_data_normalizer(normalization_config)
            self.logger.info("✅ Normalizer initialized")

            # Initialize Scaler (Intensity Scaler) - lazy import
            if self.config.enable_intensity_scaling:
                get_intensity_config, apply_intensity_scaling = _lazy_import_intensity_scaler()
                self.scaler = get_intensity_config(self.config.intensity_percentage)
                self.logger.info("✅ Intensity Scaler initialized")

            # Initialize Matrix Operations - lazy import
            if self.config.enable_matrix_operations:
                get_unified_matrix_operations = _lazy_import_matrix_operations()
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.info("✅ Matrix Operations initialized")

            
            if self.config.enable_incremental_processing:
                get_incremental_processor, IncrementalConfig = _lazy_import_incremental_processor()
                self.incremental_processor = get_incremental_processor(IncrementalConfig(
                    max_window_size=1000,
                    memory_efficient=self.config.memory_efficient,
                    enable_thread_safety=True,
                    use_vectorbt_optimizer=True,
                    use_unified_vectorization=True
                ))
                self.logger.info("✅ Incremental Processor initialized")

            # Initialize Hardware Manager with error handling - lazy import
            if self.config.enable_hardware_optimization:
                try:
                    get_unified_hardware_manager, WorkloadType, OptimizationLevel = _lazy_import_hardware_manager()
                    self.hardware_manager = get_unified_hardware_manager()
                    # Try to optimize, but don't fail if it doesn't work
                    try:
                        # Convert string config to enum objects
                        workload_enum = getattr(WorkloadType, self.config.workload_type.upper(), WorkloadType.FEATURE_ENGINEERING)
                        optimization_enum = getattr(OptimizationLevel, self.config.optimization_level.upper(), OptimizationLevel.BALANCED)
                        
                        self.hardware_manager.optimize_for_workload(
                            workload_enum,
                            optimization_enum
                        )
                    except Exception as opt_e:
                        self.logger.warning(f"⚠️ Hardware optimization failed: {opt_e}")
                        self.logger.info("🔄 Continuing with standard hardware configuration")
                    self.logger.info("✅ Hardware Manager initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Hardware Manager initialization failed: {e}")
                    self.logger.info("🔄 Continuing without hardware optimization")
                    self.hardware_manager = None

            # Initialize Vectorized Core - lazy import
            get_vectorized_processing_core = _lazy_import_vectorized_core()
            self.vectorized_core = get_vectorized_processing_core()
            self.logger.info("✅ Vectorized Core initialized")

            # Initialize Rolling Optimizer (singleton)
            if self.config.enable_rolling_optimization:
                try:
                    get_global_rolling_optimizer = _lazy_import_rolling_optimizer()
                    self.rolling_optimizer = get_global_rolling_optimizer()
                    self.logger.info("✅ Rolling Optimizer initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Rolling Optimizer initialization failed: {e}")
                    self.rolling_optimizer = None

            # Initialize VectorBT Operation Batcher
            if self.config.enable_vectorbt_optimization:
                try:
                    VectorBTOperationBatcher = _lazy_import_vectorbt_batcher()
                    self.vectorbt_batcher = VectorBTOperationBatcher()
                    self.logger.info("✅ VectorBT Operation Batcher initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT Operation Batcher initialization failed: {e}")
                    self.vectorbt_batcher = None

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise

    @contextmanager
    def _hardware_optimization_context(self):
        """Context manager for hardware optimization."""
        if self.hardware_manager:
            # Convert string config to enum objects
            try:
                from ...utils.hardware.unified_hardware_manager import WorkloadType, OptimizationLevel
                workload_enum = getattr(WorkloadType, self.config.workload_type.upper(), WorkloadType.FEATURE_ENGINEERING)
                optimization_enum = getattr(OptimizationLevel, self.config.optimization_level.upper(), OptimizationLevel.BALANCED)
                
                with self.hardware_manager.optimization_context(
                        workload_enum,
                        optimization_enum
                ):
                    yield
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware context setup failed: {e}")
                try:
                    yield
                except Exception as inner_e:
                    self.logger.error(f"⚠️ Hardware context cleanup failed: {inner_e}")
                    # Don't re-raise to avoid generator issues
        else:
            yield

    def process_features(self,
                       data: pd.DataFrame,
                       categories: Optional[List[str]] = None,
                       features: Optional[List[str]] = None,
                       target_column: Optional[str] = None,
                       **kwargs) -> PipelineResult:
        """
        Process features through the complete pipeline with hardware optimization.

        Key Optimizations:
        - Intelligent feature batching by dependency and computation cost
        - Memory-efficient in-place operations where possible
        - Aggressive caching for repeated feature calculations
        - VectorBT operation batching for related features
        - Hardware-accelerated normalization and scaling

        Args:
            data: Input DataFrame
            categories: List of feature categories to generate
            features: List of specific features to generate
            target_column: Target column for lookback optimization
            **kwargs: Additional parameters

        Returns:
            PipelineResult with processed features and metadata
        """
        # Performance optimizations: intelligent batching and memory management
        optimized_data = self._optimize_input_data(data)

        # Recursion guard to prevent circular calls
        if self._processing:
            self.logger.warning("⚠️ Recursion detected, falling back to standard generation")
            return self._fallback_to_standard_generation(data, categories, features, target_column, **kwargs)
        
        self._processing = True
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Intelligent feature ordering and batching for optimal performance
        feature_batches = self._create_optimized_feature_batches(categories, features, optimized_data)

        try:
            # Ensure FeatureBank is initialized
            if not self.feature_bank:
                self.logger.info("🔧 Initializing FeatureBank...")
                self._initialize_components()
            
            self.logger.info("🚀 Starting optimized feature processing pipeline")
            self.logger.info(f"   Input shape: {data.shape}")
            self.logger.info(f"   Categories: {categories}")
            self.logger.info(f"   Features: {features}")

            with self._hardware_optimization_context():
                # Check if we should use streaming processing
                data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
                use_streaming = (self.config.enable_streaming_processing and 
                               data_size_mb > self.config.memory_threshold_mb)
                
                if use_streaming:
                    self.logger.info(f"🌊 Using streaming processing (data size: {data_size_mb:.1f}MB)")
                    # Step 1: Generate features using streaming processing with optimized batches
                    features_df = self._streaming_feature_generation_optimized(optimized_data, feature_batches)
                else:
                    self.logger.info(f"🔄 Using optimized batch processing (data size: {data_size_mb:.1f}MB)")
                    # Step 1: Generate features using optimized batch processing
                    # Map execution_mode to intensity for compatibility
                    if 'execution_mode' in kwargs and 'intensity' not in kwargs:
                        kwargs['intensity'] = kwargs['execution_mode']

                    features_df = self._generate_features_optimized_batch(
                        optimized_data, feature_batches, **kwargs
                    )

                # Monitor memory usage after feature generation
                self._monitor_memory_usage()
                
                # Step 2: Apply normalization
                normalized_df, normalization_params = self._apply_normalization_optimized(
                    features_df, categories
                )
                
                # Monitor memory usage after normalization
                self._monitor_memory_usage()

                # Step 3: Apply scaling
                scaled_df, scaling_params = self._apply_scaling_optimized(
                    normalized_df
                )
                
                # Monitor memory usage after scaling
                self._monitor_memory_usage()

                # Step 4: Final optimization
                final_df = self._finalize_features(scaled_df)
                
                # Final memory monitoring
                self._monitor_memory_usage()

                processing_time = time.time() - start_time
                memory_usage = self._get_memory_usage() - start_memory

                # Update performance stats
                if hasattr(self, '_update_performance_stats'):
                    self._update_performance_stats(processing_time, memory_usage, True)

                self.logger.info(f"✅ Feature processing completed in {processing_time:.3f}s")
                self.logger.info(f"   Generated features: {len(final_df.columns)}")
                self.logger.info(f"   Memory usage: {memory_usage:.2f}MB")

                # Report VectorBT consolidated optimizer stats if available
                try:
                    from .consolidated_rolling_optimizer import get_global_rolling_optimizer
                    try:
                        optimizer = get_global_rolling_optimizer()
                        stats = getattr(optimizer, 'performance_stats', None)
                        if isinstance(stats, dict) and stats:
                            summary = (
                                f"VectorBT stats — total_ops: {stats.get('total_operations', 0)}, "
                                f"vbt_ops: {stats.get('vectorbt_operations', 0)}, "
                                f"pandas_fallbacks: {stats.get('pandas_fallbacks', 0)}, "
                                f"gpu_ops: {stats.get('gpu_operations', 0)}, "
                                f"avg_time_s: {stats.get('average_time_per_operation', 0.0):.6f}"
                            )
                            self.logger.info(summary)
                    except Exception:
                        # Keep pipeline robust if stats are unavailable
                        pass
                except Exception:
                    # Optional dependency; ignore if import fails
                    pass

                return PipelineResult(
                    features=final_df,
                    normalization_params=normalization_params,
                    scaling_params=scaling_params,
                    performance_stats=self.performance_stats.copy(),
                    success=True,
                    processing_time=processing_time,
                    memory_usage=memory_usage
                )

        except Exception as e:
            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory

            self.logger.error(f"❌ Feature processing failed: {e}")
            # Update performance stats
            if hasattr(self, '_update_performance_stats'):
                self._update_performance_stats(processing_time, memory_usage, False)

            # Attempt to log VectorBT consolidated optimizer stats even on failure
            try:
                from .consolidated_rolling_optimizer import get_global_rolling_optimizer
                try:
                    optimizer = get_global_rolling_optimizer()
                    stats = getattr(optimizer, 'performance_stats', None)
                    if isinstance(stats, dict) and stats:
                        summary = (
                            f"VectorBT stats — total_ops: {stats.get('total_operations', 0)}, "
                            f"vbt_ops: {stats.get('vectorbt_operations', 0)}, "
                            f"pandas_fallbacks: {stats.get('pandas_fallbacks', 0)}, "
                            f"gpu_ops: {stats.get('gpu_operations', 0)}, "
                            f"avg_time_s: {stats.get('average_time_per_operation', 0.0):.6f}"
                        )
                        self.logger.info(summary)
                except Exception:
                    pass
            except Exception:
                pass

            return PipelineResult(
                features=pd.DataFrame(),
                normalization_params={},
                scaling_params={},
                performance_stats=self.performance_stats.copy(),
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                memory_usage=memory_usage
            )
        finally:
            # Clear recursion guard
            self._processing = False

    def _fallback_to_standard_generation(self, data: pd.DataFrame, categories: Optional[List[str]] = None,
                                        features: Optional[List[str]] = None, target_column: Optional[str] = None,
                                        **kwargs) -> PipelineResult:
        """Fallback to standard feature generation when recursion is detected."""
        try:
            self.logger.info("🔄 Using standard feature generation (recursion fallback)")
            
            # Use FeatureBank's internal methods directly to avoid recursion
            generators_to_use = self.feature_bank._select_generators(categories, features)
            
            if not generators_to_use:
                self.logger.warning("No generators selected for fallback generation")
                return PipelineResult(
                    features=pd.DataFrame(),
                    normalization_params={},
                    scaling_params={},
                    performance_stats=self.performance_stats.copy(),
                    success=False,
                    error_message="No generators selected"
                )
            
            # Optimize lookbacks if requested
            if target_column and self.feature_bank.lookback_optimizer:
                generators_to_use = self.feature_bank._optimize_lookbacks(generators_to_use, data, target_column)
            
            # Generate features using internal parallel generation
            results = self.feature_bank._generate_features_parallel(generators_to_use, data, **kwargs)
            
            # Combine results
            features_df = self.feature_bank._combine_results(results, data.index)
            
            # Apply automatic normalization if enabled
            if self.feature_bank.auto_normalize and not features_df.empty:
                features_df = self.feature_bank._apply_automatic_normalization(features_df, categories)
            
            return PipelineResult(
                features=features_df,
                normalization_params={},
                scaling_params={},
                performance_stats=self.performance_stats.copy(),
                success=True,
                processing_time=0.0,
                memory_usage=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Fallback generation failed: {e}")
            return PipelineResult(
                features=pd.DataFrame(),
                normalization_params={},
                scaling_params={},
                performance_stats=self.performance_stats.copy(),
                success=False,
                error_message=str(e)
            )

    def _optimize_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize input data for better memory usage and processing speed."""
        optimized_data = data.copy()

        # Convert to optimal data types for memory efficiency
        for col in optimized_data.columns:
            if optimized_data[col].dtype == 'float64':
                # Use float32 for better memory usage if precision allows
                if (optimized_data[col].min() >= np.finfo(np.float32).min and
                    optimized_data[col].max() <= np.finfo(np.float32).max):
                    optimized_data[col] = optimized_data[col].astype(np.float32)
            elif optimized_data[col].dtype == 'int64':
                # Use int32 for better memory usage if range allows
                if (optimized_data[col].min() >= np.iinfo(np.int32).min and
                    optimized_data[col].max() <= np.iinfo(np.int32).max):
                    optimized_data[col] = optimized_data[col].astype(np.int32)

        # Sort by index for better cache locality during processing
        optimized_data = optimized_data.sort_index()

        return optimized_data

    def _create_optimized_feature_batches(self, categories: Optional[List[str]],
                                        features: Optional[List[str]],
                                        data: pd.DataFrame) -> List[Dict]:
        """Create optimized feature batches based on dependencies and computation cost."""
        # Get all available features from FeatureBank
        all_features = self._get_all_available_features(categories, features)
        self.logger.info(f"🔍 Available features from FeatureBank: {len(all_features)}")
        
        if all_features:
            # Show breakdown by category
            category_counts = {}
            for feature in all_features:
                cat = feature.get('category', 'unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            self.logger.info(f"📊 Feature breakdown by category:")
            for cat, count in sorted(category_counts.items()):
                self.logger.info(f"   • {cat}: {count} features")

        # Group features by computation complexity and dependencies
        feature_groups = self._group_features_by_complexity(all_features)
        self.logger.info(f"📊 Features grouped by complexity:")
        for complexity, features_list in feature_groups.items():
            self.logger.info(f"   • {complexity}: {len(features_list)} features")

        # Sort each group by dependencies (simple features first)
        sorted_groups = []
        for complexity, feature_list in feature_groups.items():
            sorted_features = self._sort_features_by_dependencies(feature_list)
            sorted_groups.append({
                'complexity': complexity,
                'features': sorted_features,
                'batch_size': self._calculate_optimal_batch_size(complexity, len(data))
            })

        # Create final batches optimized for memory and processing efficiency
        batches = []
        total_features_in_batches = 0
        
        for group in sorted_groups:
            current_batch = []
            current_batch_size = 0

            for feature in group['features']:
                feature_size = self._estimate_feature_memory_usage(feature, len(data))

                # Start new batch if this feature would exceed memory threshold
                if current_batch and (current_batch_size + feature_size) > self._get_memory_threshold():
                    batches.append({
                        'features': current_batch,
                        'batch_size': len(current_batch),
                        'estimated_memory': current_batch_size,
                        'complexity': group['complexity']
                    })
                    total_features_in_batches += len(current_batch)
                    current_batch = [feature]
                    current_batch_size = feature_size
                else:
                    current_batch.append(feature)
                    current_batch_size += feature_size

            # Add remaining features in current batch
            if current_batch:
                batches.append({
                    'features': current_batch,
                    'batch_size': len(current_batch),
                    'estimated_memory': current_batch_size,
                    'complexity': group['complexity']
                })
                total_features_in_batches += len(current_batch)

        self.logger.info(f"📦 Created {len(batches)} batches with {total_features_in_batches} total features")
        self.logger.info(f"📊 Batch breakdown:")
        for i, batch in enumerate(batches):
            self.logger.info(f"   • Batch {i+1}: {batch['batch_size']} features ({batch['complexity']}, {batch['estimated_memory']:.1f}MB)")

        return batches

    def _get_all_available_features(self, categories: Optional[List[str]],
                                  features: Optional[List[str]]) -> List[Dict]:
        """Get all available features based on categories and feature filters."""
        if self.feature_bank:
            return self.feature_bank.get_available_features(categories, features)
        return []

    def _group_features_by_complexity(self, features: List[Dict]) -> Dict[str, List[Dict]]:
        """Group features by computational complexity."""
        groups = {
            'simple': [],      # Basic indicators like SMA, EMA
            'medium': [],      # Rolling operations, basic statistics
            'complex': [],     # VectorBT operations, complex calculations
            'expensive': []    # Features that require heavy computation
        }

        for feature in features:
            complexity = self._assess_feature_complexity(feature)
            groups[complexity].append(feature)

        return groups

    def _assess_feature_complexity(self, feature: Dict) -> str:
        """Assess computational complexity of a feature."""
        feature_name = feature.get('name', '').lower()

        # Simple features - basic indicators
        if any(keyword in feature_name for keyword in ['sma', 'ema', 'rsi', 'macd']):
            return 'simple'

        # Medium complexity - rolling operations
        if any(keyword in feature_name for keyword in ['rolling', 'std', 'var', 'corr']):
            return 'medium'

        # Complex features - VectorBT operations
        if any(keyword in feature_name for keyword in ['vectorbt', 'momentum', 'volatility']):
            return 'complex'

        # Expensive features - trend analysis, pattern recognition
        if any(keyword in feature_name for keyword in ['trend_strength', 'pattern', 'breakout']):
            return 'expensive'

        return 'medium'  # Default

    def _sort_features_by_dependencies(self, features: List[Dict]) -> List[Dict]:
        """Sort features by their dependencies (simple features first)."""
        # For now, return in original order - dependency resolution can be enhanced later
        return features

    def _calculate_optimal_batch_size(self, complexity: str, data_size: int) -> int:
        """Calculate optimal batch size based on complexity and data size."""
        base_sizes = {
            'simple': 50,
            'medium': 25,
            'complex': 10,
            'expensive': 5
        }

        base_size = base_sizes.get(complexity, 20)

        # Adjust based on data size
        if data_size > 100000:  # Large dataset
            return max(1, base_size // 2)
        elif data_size > 50000:  # Medium dataset
            return max(1, base_size * 3 // 4)
        else:  # Small dataset
            return base_size

    def _estimate_feature_memory_usage(self, feature: Dict, data_size: int) -> float:
        """Estimate memory usage for a feature in MB."""
        # Rough estimation based on feature complexity
        complexity = self._assess_feature_complexity(feature)
        multipliers = {
            'simple': 1.0,
            'medium': 2.0,
            'complex': 3.0,
            'expensive': 5.0
        }

        return (data_size * 8) / (1024 * 1024) * multipliers.get(complexity, 2.0)  # Assume 8 bytes per float

    def _get_memory_threshold(self) -> float:
        """Get current memory threshold for batching."""
        # Use 80% of available memory as threshold
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        return available_memory * 0.8

    def _choose_processing_strategy(self, data: pd.DataFrame, categories: Optional[List[str]],
                                  features: Optional[List[str]]) -> str:
        """Choose the optimal processing strategy based on data characteristics."""
        if self.config.processing_strategy != "auto":
            return self.config.processing_strategy
        
        # Auto-strategy selection logic
        data_size = len(data)
        num_columns = len(data.columns)
        
        # Use incremental for real-time streaming or very small datasets
        if data_size < 1000:
            return "incremental"
        
        
        # Use traditional for medium datasets
        return "traditional"

    def _generate_features_optimized(self,
                                   data: pd.DataFrame,
                                   categories: Optional[List[str]] = None,
                                   features: Optional[List[str]] = None,
                                   target_column: Optional[str] = None,
                                   **kwargs) -> pd.DataFrame:
        """Generate features with hardware optimization and processing strategy selection."""
        try:
            # Apply light mode restriction if needed (same logic as FeatureBank)
            data = self._apply_light_mode_restriction(data, **kwargs)
            
            # Apply M1 optimizations if enabled
            if self.config.enable_m1_optimizations:
                data = self._m1_optimizations(data)
            
            # Optimize data types for better memory usage
            data = self._optimize_data_types(data)
            
            # Apply memory mapping if beneficial
            if self.memory_mapped_handler and self.config.enable_memory_mapping:
                data = self._apply_memory_mapping(data)
            
            # Use vectorized core for optimization
            if self.vectorized_core:
                data = self.vectorized_core.optimize_dataframe_for_processing(data)

            # Check intelligent cache first
            if self.feature_cache and self.config.enable_intelligent_caching:
                feature_specs = self._create_feature_specs(categories, features)
                cached_features = self.feature_cache.get_cached_features(data, feature_specs)
                if cached_features is not None:
                    self.logger.info("✅ Using cached features")
                    return cached_features

            # Choose optimal processing strategy
            strategy = self._choose_processing_strategy(data, categories, features)
            self.logger.info(f"🚀 Using processing strategy: {strategy}")
            
            # Check if we have FeatureBank generators available
            has_featurebank_generators = (hasattr(self, 'feature_bank') and 
                                        self.feature_bank and 
                                        len(self.feature_bank.registry.get_all()) > 0)
            
            # Prioritize our optimizations over traditional FeatureBank approach
            if self.config.enable_parallel_processing and len(data) > 1000:
                # Use parallel processing for larger datasets (PRIORITY)
                self.logger.info("🚀 Using parallel feature generation for large dataset")
                feature_configs = self._create_feature_configs(categories, features)
                result = self._parallel_feature_generation(data, feature_configs)
            elif strategy == "incremental" and self.incremental_processor:
                result = self._generate_features_incremental(data, categories, features, **kwargs)
            elif has_featurebank_generators:
                # Use traditional FeatureBank approach for full generator access (FALLBACK)
                self.logger.info("🎯 Using traditional FeatureBank approach for full generator access")
                result = self._generate_features_traditional(data, categories, features, target_column, **kwargs)
            else:
                # Fallback to traditional FeatureBank approach
                result = self._generate_features_traditional(data, categories, features, target_column, **kwargs)
            
            # Cache the result if intelligent caching is enabled
            if self.feature_cache and self.config.enable_intelligent_caching and not result.empty:
                feature_specs = self._create_feature_specs(categories, features)
                # Extract actual feature data if result contains FeatureResult objects
                cache_data = self._extract_feature_data_for_cache(result)
                if cache_data is not None:
                    self.feature_cache.cache_features(data, feature_specs, cache_data)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Optimized feature generation failed: {e}")
            raise


    
    def _generate_features_incremental(self, data: pd.DataFrame, categories: Optional[List[str]], 
                                     features: Optional[List[str]], **kwargs) -> pd.DataFrame:
        """Generate features using incremental processing with fast paths and delta computations."""
        try:
            # Fast path for small windows and streaming data
            if len(data) < 1000 and self.config.processing_strategy == 'incremental':
                return self._generate_features_incremental_fast_path(data, categories, features, **kwargs)
            
            # Create feature specifications for incremental processing
            feature_specs = self._create_incremental_specs(categories, features)
            
            # Process using incremental processor with delta computations
            features_df = self._process_incremental_with_deltas(data, feature_specs, categories, features)
            
            self.logger.info(f"✅ Incremental processing completed: {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Incremental processing failed: {e}")
            # Fallback to traditional
            return self._generate_features_traditional(data, categories, features, None, **kwargs)
    
    def _generate_features_incremental_fast_path(self, data: pd.DataFrame, categories: Optional[List[str]], 
                                               features: Optional[List[str]], **kwargs) -> pd.DataFrame:
        """Fast path for incremental processing on small datasets."""
        try:
            # Bypass full pipeline for small windows
            if self.incremental_processor:
                feature_specs = self._create_incremental_specs(categories, features)
                return self.incremental_processor.process_dataframe_incremental(data, feature_specs)
            else:
                # Fallback to traditional
                return self._generate_features_traditional(data, categories, features, None, **kwargs)
        except Exception as e:
            self.logger.warning(f"Fast path incremental processing failed: {e}")
            return self._generate_features_traditional(data, categories, features, None, **kwargs)
    
    def _process_incremental_with_deltas(self, data: pd.DataFrame, feature_specs: List[Dict], categories: Optional[List[str]] = None, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Process incremental features with delta computations for rolling features."""
        try:
            # For rolling features on live feeds, only compute deltas for the last k rows
            if hasattr(self, 'incremental_processor') and self.incremental_processor:
                # Check if we can use delta computations
                if hasattr(self.incremental_processor, 'process_delta_features'):
                    # Process only the last k rows for rolling features
                    delta_window = min(100, len(data) // 10)  # Last 10% or 100 rows, whichever is smaller
                    if len(data) > delta_window:
                        # Process delta for recent data
                        recent_data = data.tail(delta_window)
                        delta_features = self.incremental_processor.process_delta_features(recent_data, feature_specs)
                        
                        # Combine with cached previous results if available
                        if hasattr(self, '_previous_features') and self._previous_features is not None:
                            # Merge delta with previous results
                            combined_features = pd.concat([self._previous_features, delta_features], ignore_index=True)
                            self._previous_features = combined_features.tail(len(data))
                            return combined_features
                        else:
                            # First run, process full dataset
                            full_features = self.incremental_processor.process_dataframe_incremental(data, feature_specs)
                            self._previous_features = full_features.tail(delta_window)
                            return full_features
                    else:
                        # Small dataset, process normally
                        return self.incremental_processor.process_dataframe_incremental(data, feature_specs)
                else:
                    # Fallback to normal incremental processing
                    return self.incremental_processor.process_dataframe_incremental(data, feature_specs)
            else:
                # No incremental processor available, use traditional
                return self._generate_features_traditional(data, categories, features, None)
        except Exception as e:
            self.logger.warning(f"Delta processing failed: {e}")
            # Fallback to traditional processing
            return self._generate_features_traditional(data, categories, features, None)
    
    def _compute_one_pass_stats(self, data: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        """Compute multiple statistics in one pass over the data."""
        try:
            # Precompute rolling window statistics in one pass
            rolling_data = data.rolling(window=window)
            
            # Compute all stats at once
            stats = {
                'mean': rolling_data.mean(),
                'std': rolling_data.std(),
                'min': rolling_data.min(),
                'max': rolling_data.max(),
                'sum': rolling_data.sum()
            }
            
            # Add derived stats that use the same base stats
            stats['zscore'] = (data - stats['mean']) / stats['std']
            stats['rsi'] = self._compute_rsi_from_stats(data, stats['mean'], stats['std'])
            
            return stats
        except Exception as e:
            self.logger.warning(f"One-pass stats computation failed: {e}")
            return {}
    
    def _compute_rsi_from_stats(self, data: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
        """Compute RSI using precomputed mean and std."""
        try:
            # Simplified RSI calculation using precomputed stats
            from ...utils.error_handling import safe_diff
            price_change = safe_diff(data)
            gain = price_change.where(price_change > 0, 0)
            loss = -price_change.where(price_change < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(index=data.index, dtype=float)
    
    def _compute_approximate_quantiles(self, data: pd.Series, quantiles: List[float] = [0.25, 0.5, 0.75]) -> Dict[str, pd.Series]:
        """Compute approximate quantiles for large series using t-digest approximation."""
        try:
            # For very large series, use approximate quantiles
            if len(data) > 10000:
                # Use pandas quantile with interpolation for approximation
                results = {}
                for q in quantiles:
                    results[f'q{int(q*100)}'] = data.rolling(window=min(100, len(data)//10)).quantile(q, interpolation='linear')
                return results
            else:
                # For smaller series, use exact quantiles
                results = {}
                for q in quantiles:
                    results[f'q{int(q*100)}'] = data.rolling(window=20).quantile(q)
                return results
        except Exception as e:
            self.logger.warning(f"Approximate quantiles computation failed: {e}")
            return {}
    
    def _extract_feature_data_for_cache(self, result) -> Optional[pd.DataFrame]:
        """Extract feature data from result for caching purposes."""
        try:
            # If result is already a DataFrame, return it
            if isinstance(result, pd.DataFrame):
                return result
            
            # If result is a list of FeatureResult objects, extract the data
            if isinstance(result, list):
                feature_data = {}
                for feature_result in result:
                    if hasattr(feature_result, 'name') and hasattr(feature_result, 'data'):
                        # Extract the feature name and data
                        feature_name = feature_result.name
                        feature_series = feature_result.data
                        if isinstance(feature_series, pd.Series):
                            feature_data[feature_name] = feature_series
                        elif isinstance(feature_series, pd.DataFrame):
                            # If it's a DataFrame, use the first column or merge all columns
                            if len(feature_series.columns) == 1:
                                feature_data[feature_name] = feature_series.iloc[:, 0]
                            else:
                                # Merge multiple columns with prefixed names
                                for col in feature_series.columns:
                                    feature_data[f"{feature_name}_{col}"] = feature_series[col]
                
                if feature_data:
                    return pd.DataFrame(feature_data, index=result[0].data.index if result else None)
            
            # If result is a single FeatureResult object
            if hasattr(result, 'data'):
                if isinstance(result.data, pd.Series):
                    return pd.DataFrame({result.name: result.data})
                elif isinstance(result.data, pd.DataFrame):
                    return result.data
            
            # If we can't extract data, return None to skip caching
            self.logger.warning("Could not extract feature data for caching")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract feature data for cache: {e}")
            return None
    
    def _generate_features_traditional(self, data: pd.DataFrame, categories: Optional[List[str]], 
                                     features: Optional[List[str]], target_column: Optional[str], 
                                     **kwargs) -> pd.DataFrame:
        """Generate features using traditional FeatureBank approach with VectorBT optimizations."""
        try:
            # Use FeatureBank with optimized pipeline enabled for VectorBT optimizations
            self.logger.info("🎯 Using FeatureBank with optimized pipeline for VectorBT optimizations")
            return self.feature_bank.generate_features(
                data=data,
                categories=categories,
                features=features,
                target_column=target_column,
                use_optimized_pipeline=True,  # Enable VectorBT optimizations
                **kwargs
            )

        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            raise
    
    def _create_enhanced_default_specs(self) -> Dict[str, Any]:
        """Create enhanced default specifications when FeatureBank is not available."""
        specs = {}
        
        # Enhanced columns to process
        enhanced_columns = ['close', 'volume', 'high', 'low', 'open', 'close_return', 'close_log_return']
        
        for column in enhanced_columns:
            specs[column] = {
                'rolling': {
                    'windows': [5, 10, 20, 50, 100, 200],
                    'functions': ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
                },
                'statistical': {
                    'functions': ['skew', 'kurt', 'quantile', 'rank', 'percentile']
                },
                'technical': {
                    'functions': ['rsi', 'macd', 'bollinger', 'stoch', 'williams', 'cci', 'adx', 'aroon']
                },
                'momentum': {
                    'functions': ['roc', 'momentum', 'rate_of_change', 'price_velocity']
                },
                'volatility': {
                    'functions': ['atr', 'volatility', 'garch', 'parkinson', 'garman_klass']
                },
                'volume': {
                    'functions': ['obv', 'ad_line', 'mfi', 'volume_rate', 'volume_oscillator']
                },
                'trend': {
                    'functions': ['ema', 'sma', 'wma', 'dema', 'tema', 'kama', 't3']
                },
                'oscillator': {
                    'functions': ['rsi', 'stoch', 'williams', 'cci', 'roc', 'momentum']
                },
                'pattern': {
                    'functions': ['doji', 'hammer', 'shooting_star', 'engulfing', 'harami']
                },
                'advanced': {
                    'functions': ['fourier', 'wavelet', 'entropy', 'fractal', 'hurst']
                }
            }
        
        return specs
    
    def _create_incremental_specs(self, categories: Optional[List[str]], features: Optional[List[str]]) -> Dict[str, Any]:
        """Create feature specifications for incremental processing."""
        specs = {}
        
        # Default columns to process
        default_columns = ['close', 'volume', 'high', 'low', 'open']
        
        for column in default_columns:
            specs[column] = {
                'rolling_mean': {
                    'windows': [20, 50, 100]
                },
                'rolling_std': {
                    'windows': [20, 50, 100]
                },
                'rsi': {
                    'windows': [14, 21]
                }
            }
        
        return specs
    
    def _create_feature_specs(self, categories: Optional[List[str]], features: Optional[List[str]]) -> List[str]:
        """Create feature specifications for caching."""
        if features:
            return features
        elif categories:
            return categories
        else:
            return ['default']
    
    def _create_feature_configs(self, categories: Optional[List[str]], features: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Create feature configurations for parallel processing using actual FeatureBank generators."""
        configs = []
        
        if features:
            # Create configs for specific features
            for feature in features:
                configs.append({
                    'name': feature,
                    'type': 'specific',
                    'category': 'custom'
                })
        elif categories:
            # Try to use FeatureBank generators if available
            if hasattr(self, 'feature_bank') and self.feature_bank:
                try:
                    # Debug: Check what's available in the registry
                    all_generators = self.feature_bank.registry.get_all()
                    self.logger.debug(f"🔍 FeatureBank registry has {len(all_generators)} total generators")
                    
                    # Check available categories
                    try:
                        from ..core.feature_bank import FeatureCategory
                        available_categories = [cat.name.lower() for cat in FeatureCategory]
                        self.logger.debug(f"🔍 Available categories: {available_categories}")
                    except Exception as e:
                        self.logger.debug(f"🔍 Could not get available categories: {e}")
                    
                    # Get generators for each category
                    for category in categories:
                        self.logger.debug(f"🔍 Processing category: {category}")
                        
                        # Convert string to FeatureCategory if needed
                        if isinstance(category, str):
                            try:
                                from ..core.feature_bank import FeatureCategory
                                category_enum = getattr(FeatureCategory, category.upper(), None)
                                self.logger.debug(f"🔍 Category enum: {category_enum}")
                                
                                if category_enum:
                                    generators = self.feature_bank.registry.get_by_category(category_enum)
                                    self.logger.debug(f"🔍 Found {len(generators)} generators for {category}")
                                    
                                    # If no generators found, try alternative approaches
                                    if not generators:
                                        self.logger.debug(f"🔍 No generators found for {category}, trying alternative approaches...")
                                        
                                        # Try with different case variations
                                        for alt_category in [category.lower(), category.upper(), category.capitalize()]:
                                            try:
                                                alt_enum = getattr(FeatureCategory, alt_category.upper(), None)
                                                if alt_enum:
                                                    alt_generators = self.feature_bank.registry.get_by_category(alt_enum)
                                                    if alt_generators:
                                                        self.logger.debug(f"🔍 Found {len(alt_generators)} generators using alternative case: {alt_category}")
                                                        generators = alt_generators
                                                        break
                                            except Exception:
                                                continue
                                else:
                                    self.logger.warning(f"⚠️ Category '{category}' not found in FeatureCategory enum")
                                    generators = []
                            except Exception as e:
                                self.logger.warning(f"⚠️ Failed to get category enum for '{category}': {e}")
                                generators = []
                        else:
                            generators = self.feature_bank.registry.get_by_category(category)
                            self.logger.debug(f"🔍 Found {len(generators)} generators for {category}")
                        
                        # Create configs for each generator
                        for i, generator in enumerate(generators):
                            configs.append({
                                'name': f'{category}_{generator.__class__.__name__}_{i}',
                                'type': 'generator',
                                'category': category,
                                'generator': generator
                            })
                    
                    self.logger.info(f"📊 Created {len(configs)} feature configs from {len(categories)} categories using FeatureBank generators")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get FeatureBank generators: {e}")
                    # Fallback to generic configs
                    for category in categories:
                        configs.append({
                            'name': f'{category}_features',
                            'type': 'category',
                            'category': category
                        })
            else:
                # FeatureBank must be available for optimized pipeline
                self.logger.error("❌ FeatureBank not available - optimized pipeline requires FeatureBank")
                raise RuntimeError("FeatureBank is required for optimized feature generation but is not available")
        else:
            # No categories specified - this should not happen in optimized pipeline
            self.logger.error("❌ No categories specified for feature generation")
            raise ValueError("Categories must be specified for optimized feature generation")
        
        return configs

    def _apply_normalization_optimized(self,
                                     features_df: pd.DataFrame,
                                     categories: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply normalization with hardware optimization."""
        try:
            if not self.config.auto_normalize or features_df.empty:
                return features_df, {}

            # Select features for normalization
            target_columns = self._select_normalization_targets(features_df, categories)

            if not target_columns:
                return features_df, {}

            # Apply normalization using the normalizer
            normalization_result = self.normalizer.normalize_data(
                features_df, target_columns=target_columns
            )

            if normalization_result.success:
                self.performance_stats['hardware_accelerations'] += 1
                return normalization_result.normalized_data, normalization_result.normalization_params
            else:
                self.logger.warning("Normalization failed, returning original features")
                return features_df, {}

        except Exception as e:
            self.logger.error(f"Normalization failed: {e}")
            return features_df, {}

    def _apply_scaling_optimized(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply intensity scaling with optimization."""
        try:
            if not self.config.enable_intensity_scaling or features_df.empty:
                return features_df, {}

            # Apply intensity scaling to configuration
            scaling_params = {
                'intensity_percentage': self.scaler.intensity_percentage,
                'training_mode': self.scaler.training_mode,
                'scaled_parameters': {}
            }

            # Scale feature generation parameters if needed
            if hasattr(self.feature_bank, 'config') and hasattr(self, 'scaler') and self.scaler:
                try:
                    get_intensity_config, apply_intensity_scaling = _lazy_import_intensity_scaler()
                    scaled_config = apply_intensity_scaling(
                        self.feature_bank.config.__dict__,
                        self.scaler.intensity_percentage
                    )
                    scaling_params['scaled_parameters'] = scaled_config
                except Exception as e:
                    self.logger.warning(f"Intensity scaling failed: {e}")

            return features_df, scaling_params

        except Exception as e:
            self.logger.error(f"Scaling failed: {e}")
            return features_df, {}

    def _finalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Finalize features with additional optimizations."""
        try:
            if features_df.empty:
                return features_df

            # Apply final vectorized optimizations
            if self.vectorized_core:
                features_df = self.vectorized_core.optimize_dataframe_for_processing(features_df)

            # Avoid whole-frame fillna - only fill selected numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                # Only fill NaN values in numeric columns that actually have NaN values
                cols_with_nan = numeric_cols[features_df[numeric_cols].isna().any()]
                if not cols_with_nan.empty:
                    features_df[cols_with_nan] = features_df[cols_with_nan].fillna(0)

            # Ensure numeric types are optimized with copy=False
            for col in features_df.select_dtypes(include=[np.number]).columns:
                if features_df[col].dtype == np.float64:
                    # Check if float32 is sufficient
                    if (features_df[col].max() < np.finfo(np.float32).max and
                        features_df[col].min() > np.finfo(np.float32).min):
                        features_df[col] = features_df[col].astype(np.float32, copy=False)

            return features_df

        except Exception as e:
            self.logger.error(f"Feature finalization failed: {e}")
            return features_df

    def _select_normalization_targets(self,
                                    features_df: pd.DataFrame,
                                    categories: Optional[List[str]] = None) -> List[str]:
        """Select which features should be normalized with strict criteria."""
        target_columns = []

        # Get numeric columns
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_columns:
            # Skip excluded features
            if col in self.config.normalization_exclude_features:
                continue

            # Skip features from excluded categories
            if categories and self._is_feature_in_excluded_category(col, categories):
                continue

            # Only normalize features that materially benefit from normalization
            if self._should_normalize_feature(features_df[col], col):
                target_columns.append(col)

        return target_columns

    def _is_feature_in_excluded_category(self, feature_name: str, categories: List[str]) -> bool:
        """Check if a feature belongs to an excluded category."""
        # Simple heuristic - in practice, you'd maintain a proper mapping
        excluded_indicators = ['zscore', 'normalized', 'scaled', 'rank']
        return any(indicator in feature_name.lower() for indicator in excluded_indicators)

    def _is_already_normalized(self, feature_name: str) -> bool:
        """Check if a feature is already normalized."""
        normalized_indicators = [
            'rsi', 'stoch', 'williams', 'macd_hist', 'bb_percent',
            'adx', 'cci', 'momentum', 'roc', 'zscore', 'normalized'
        ]
        return any(indicator in feature_name.lower() for indicator in normalized_indicators)
    
    def _should_normalize_feature(self, series: pd.Series, feature_name: str) -> bool:
        """Determine if a feature should be normalized based on strict criteria."""
        try:
            # Skip if already normalized
            if self._is_already_normalized(feature_name):
                return False
            
            # Skip if too many NaN values
            if series.isna().sum() / len(series) > 0.5:
                return False
            
            # Skip if constant or near-constant
            if series.std() < 1e-10:
                return False
            
            # Skip if already in a good range (between -1 and 1)
            if series.min() >= -1 and series.max() <= 1:
                return False
            
            # Only normalize if there's significant variance and the range is large
            coefficient_of_variation = series.std() / abs(series.mean()) if series.mean() != 0 else float('inf')
            if coefficient_of_variation < 0.1:  # Low variance
                return False
            
            # Check if normalization would materially improve the feature
            # (e.g., large range, high variance, not already bounded)
            range_ratio = (series.max() - series.min()) / abs(series.mean()) if series.mean() != 0 else float('inf')
            if range_ratio < 2:  # Small range relative to mean
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking normalization criteria for {feature_name}: {e}")
            return False

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_optimal_workers(self, data_size: int) -> int:
        """Get optimal number of workers based on data size and CPU cores."""
        cpu_count = os.cpu_count() or mp.cpu_count()
        if data_size < 1000:
            return min(2, cpu_count)
        elif data_size < 10000:
            return min(4, cpu_count)
        else:
            return min(self.config.max_workers, cpu_count)
    
    # NOTE: Unified _get_adaptive_chunk_size signature to accept DataFrame only.

    def get_vectorbt_rolling_optimizer(self):
        """Get the singleton rolling optimizer for cache locality."""
        if self.rolling_optimizer is None:
            try:
                get_global_rolling_optimizer = _lazy_import_rolling_optimizer()
                self.rolling_optimizer = get_global_rolling_optimizer()
            except Exception as e:
                self.logger.warning(f"Failed to get rolling optimizer: {e}")
                return None
        return self.rolling_optimizer

    def _batch_rolling_calculations(self, data: pd.DataFrame, window_sets: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Batch rolling calculations by coalescing per-operation across columns.

        Groups requests by operation and applies wide-matrix rolling to all columns
        and the union of requested windows in a single pass per operation.
        """
        if not self.vectorbt_batcher or not window_sets:
            return {}

        try:
            # Collect columns and windows per operation
            op_to_windows: Dict[str, set] = {}
            op_to_columns: Dict[str, set] = {}
            for cfg in window_sets:
                op = str(cfg.get('operation', 'mean')).lower()
                win = int(cfg.get('window', 20))
                col = cfg.get('column')
                op_to_windows.setdefault(op, set()).add(win)
                if col is not None:
                    op_to_columns.setdefault(op, set()).add(col)

            results: Dict[str, pd.DataFrame] = {}
            for op, windows in op_to_windows.items():
                # Select columns (default to all numeric if none specified)
                cols = list(op_to_columns.get(op, []))
                df_in = data[cols] if cols else data.select_dtypes(include=[np.number])
                if df_in.empty:
                    continue
                if hasattr(self.vectorbt_batcher, 'rolling_dataframe'):
                    out = self.vectorbt_batcher.rolling_dataframe(df_in, sorted(windows), operation=op)
                else:
                    # Fallback: pandas per window, but still coalesced across columns
                    frames = []
                    for w in sorted(windows):
                        rolled = df_in.rolling(window=w, min_periods=w)
                        if op == 'mean':
                            rolled = rolled.mean()
                        elif op == 'std':
                            rolled = rolled.std()
                        elif op == 'min':
                            rolled = rolled.min()
                        elif op == 'max':
                            rolled = rolled.max()
                        elif op == 'sum':
                            rolled = rolled.sum()
                        else:
                            continue
                        frames.append(rolled.rename(columns={c: f"{c}_{op}_{w}" for c in rolled.columns}))
                    out = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df_in.index)
                results[f"rolling_{op}"] = out

            return results

        except Exception as e:
            self.logger.warning(f"Batch rolling calculations failed: {e}")
            return {}
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for better memory usage and performance."""
        if self.data_type_optimizer:
            # Use advanced data type optimization
            optimized_data = self.data_type_optimizer.optimize_dataframe(data)
            
            # Log memory reduction
            memory_stats = self.data_type_optimizer.get_memory_usage_reduction(data, optimized_data)
            if memory_stats['reduction_percentage'] > 5:  # Only log if significant reduction
                self.logger.info(f"💾 Data type optimization: {memory_stats['reduction_percentage']:.1f}% memory reduction "
                               f"({memory_stats['reduction_mb']:.1f}MB saved)")
            
            return optimized_data
        else:
            # Enhanced fallback optimization
            data_optimized = data.copy()
            original_memory = data_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
            
            for col in data_optimized.select_dtypes(include=[np.number]).columns:
                col_data = data_optimized[col]
                
                # Skip if column has NaN values that might cause issues
                if col_data.isna().any():
                    continue
                
                # Float64 -> Float32 optimization
                if col_data.dtype == np.float64:
                    if (col_data.max() < np.finfo(np.float32).max and
                        col_data.min() > np.finfo(np.float32).min):
                        data_optimized[col] = col_data.astype(np.float32)
                
                # Int64 -> Int32 optimization
                elif col_data.dtype == np.int64:
                    if (col_data.max() < np.iinfo(np.int32).max and
                        col_data.min() > np.iinfo(np.int32).min):
                        data_optimized[col] = col_data.astype(np.int32)
                
                # Int64 -> Int16 optimization for small ranges
                elif col_data.dtype == np.int64:
                    if (col_data.max() < np.iinfo(np.int16).max and
                        col_data.min() > np.iinfo(np.int16).min):
                        data_optimized[col] = col_data.astype(np.int16)
                
                # Int64 -> Int8 optimization for very small ranges
                elif col_data.dtype == np.int64:
                    if (col_data.max() < np.iinfo(np.int8).max and
                        col_data.min() > np.iinfo(np.int8).min):
                        data_optimized[col] = col_data.astype(np.int8)
            
            # Log memory reduction
            optimized_memory = data_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
            memory_reduction = original_memory - optimized_memory
            if memory_reduction > 1.0:  # Only log if significant reduction
                self.logger.info(f"💾 Enhanced data type optimization: {memory_reduction:.1f}MB saved "
                               f"({(memory_reduction/original_memory)*100:.1f}% reduction)")
            
            return data_optimized
    
    def _m1_optimizations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply M1 Mac specific optimizations."""
        if not self.config.enable_m1_optimizations:
            return data
        
        try:
            # Use M1's unified memory architecture
            data_optimized = self._optimize_data_types(data)
            
            # M1-specific memory optimization
            if hasattr(self, 'm1_optimized') and self.m1_optimized:
                # Use Metal Performance Shaders (MPS) if available
                if self.config.enable_mps_acceleration:
                    try:
                        import torch
                        if torch.backends.mps.is_available():
                            # Convert to tensor for MPS operations
                            data_tensor = torch.tensor(data_optimized.values, device='mps')
                            # Perform operations on MPS
                            optimized_data = self._mps_operations(data_tensor)
                            return pd.DataFrame(optimized_data.cpu().numpy(), 
                                              index=data.index, columns=data.columns)
                    except ImportError:
                        pass
            
            return data_optimized
            
        except Exception as e:
            self.logger.warning(f"M1 optimizations failed: {e}")
            return data
    
    def _mps_operations(self, data_tensor):
        """Perform operations using Metal Performance Shaders."""
        # Placeholder for MPS-specific operations
        # In a full implementation, this would use MPS for matrix operations
        return data_tensor
    
    def _streaming_feature_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using streaming processing to reduce memory usage."""
        if not self.config.enable_streaming_processing:
            return self._traditional_feature_generation(data)
        
        self.logger.info("🌊 Starting streaming feature generation")
        
        # Calculate optimal chunk size based on available memory
        chunk_size = self._get_adaptive_chunk_size(data)
        total_rows = len(data)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        self.logger.info(f"📊 Processing {total_rows} rows in {num_chunks} chunks of {chunk_size} rows")
        
        # Initialize result storage
        feature_results = []
        processed_rows = 0
        
        try:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)
                
                # Extract chunk
                chunk_data = data.iloc[start_idx:end_idx].copy()
                
                self.logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{num_chunks} (rows {start_idx}-{end_idx-1})")
                
                # Process chunk
                chunk_features = self._process_chunk(chunk_data, chunk_idx)
                
                # Store results
                feature_results.append(chunk_features)
                processed_rows += len(chunk_data)
                
                # Memory cleanup after each chunk
                self._cleanup_chunk_memory(chunk_data, chunk_features)
                
                # Check memory usage and adjust if needed
                if self._should_trigger_cleanup():
                    self.force_memory_cleanup()
                
                # Progress update
                progress = (chunk_idx + 1) / num_chunks * 100
                self.logger.info(f"📈 Progress: {progress:.1f}% ({processed_rows}/{total_rows} rows)")
            
            # Combine all results
            self.logger.info("🔗 Combining feature results")
            final_features = self._combine_chunk_results(feature_results, data.index)
            
            # Final cleanup
            del feature_results
            self.force_memory_cleanup()
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"❌ Streaming processing failed: {e}")
            # Fallback to traditional processing
            return self._traditional_feature_generation(data)

    def _streaming_feature_generation_optimized(self, data: pd.DataFrame, feature_batches: List[Dict]) -> pd.DataFrame:
        """Generate features using optimized streaming processing with intelligent batching."""
        try:
            all_features = []

            # Process each batch in the optimized order
            for batch_idx, batch in enumerate(feature_batches):
                self.logger.info(f"📦 Processing batch {batch_idx + 1}/{len(feature_batches)} "
                               f"({len(batch['features'])} features, complexity: {batch['complexity']})")

                # Process this batch using chunked processing
                batch_features = self._process_feature_batch_streaming(data, batch['features'], batch['complexity'])

                if batch_features is not None and not batch_features.empty:
                    all_features.append(batch_features)

                # Memory cleanup after each batch
                gc.collect()

            # Combine all feature batches
            if all_features:
                combined_features = pd.concat(all_features, axis=1)
                return combined_features
            else:
                return pd.DataFrame(index=data.index)

        except Exception as e:
            self.logger.error(f"Streaming feature generation failed: {e}")
            return pd.DataFrame(index=data.index)

    def _generate_features_optimized_batch(self, data: pd.DataFrame, feature_batches: List[Dict], **kwargs) -> pd.DataFrame:
        """Generate features using optimized batch processing with intelligent memory management."""
        try:
            all_features = []

            # Process each batch in the optimized order
            for batch_idx, batch in enumerate(feature_batches):
                self.logger.info(f"📦 Processing batch {batch_idx + 1}/{len(feature_batches)} "
                               f"({len(batch['features'])} features, complexity: {batch['complexity']}, "
                               f"estimated memory: {batch['estimated_memory']:.2f}MB)")

                # Process this batch with optimized memory management
                batch_features = self._process_feature_batch_optimized(
                    data, batch['features'], batch['complexity']
                )

                if batch_features is not None and not batch_features.empty:
                    all_features.append(batch_features)

                # Aggressive memory cleanup after each batch
                gc.collect()

                # Monitor memory usage
                self._monitor_memory_usage()

            # Combine all feature batches
            if all_features:
                combined_features = pd.concat(all_features, axis=1)
                return combined_features
            else:
                return pd.DataFrame(index=data.index)

        except Exception as e:
            self.logger.error(f"Optimized batch feature generation failed: {e}")
            return pd.DataFrame(index=data.index)

    def _process_feature_batch_optimized(self, data: pd.DataFrame, features: List[Dict], complexity: str) -> pd.DataFrame:
        """Process a batch of features with optimized memory usage and VectorBT batching."""
        try:
            batch_results = {}

            # Group features by processing type for optimal batching
            vectorbt_features = []
            pandas_features = []

            for feature in features:
                feature_type = self._assess_feature_processing_type(feature)
                if feature_type == 'vectorbt':
                    vectorbt_features.append(feature)
                else:
                    pandas_features.append(feature)
            
            self.logger.info(f"🔍 Feature processing type breakdown:")
            self.logger.info(f"   📊 Total features: {len(features)}")
            self.logger.info(f"   🚀 VectorBT features: {len(vectorbt_features)}")
            self.logger.info(f"   📊 Pandas features: {len(pandas_features)}")

            # Process VectorBT features in batches for optimal performance
            if vectorbt_features:
                self.logger.debug(f"🚀 Processing {len(vectorbt_features)} VectorBT features")
                vectorbt_results = self._process_vectorbt_batch(data, vectorbt_features)
                batch_results.update(vectorbt_results)

            # Process pandas features individually or in small batches
            if pandas_features:
                self.logger.info(f"📊 Processing {len(pandas_features)} pandas features")
                pandas_results = self._process_pandas_batch(data, pandas_features)
                self.logger.info(f"   ✅ Pandas processing completed: {len(pandas_results)} results")
                batch_results.update(pandas_results)

            # Convert results to DataFrame
            if batch_results:
                return pd.DataFrame(batch_results, index=data.index)
            else:
                return pd.DataFrame(index=data.index)

        except Exception as e:
            self.logger.error(f"Optimized batch processing failed: {e}")
            return pd.DataFrame(index=data.index)

    def _assess_feature_processing_type(self, feature: Dict) -> str:
        """Determine optimal processing type for a feature."""
        feature_name = feature.get('name', '').lower()
        feature_category = feature.get('category', '').lower()

        # VectorBT-optimized features - expanded keyword list
        vectorbt_keywords = [
            'vectorbt', 'momentum', 'volatility', 'rsi', 'macd', 'bbands',
            'sma', 'ema', 'returns', 'volume', 'trend', 'oscillator',
            'support_resistance', 'candlestick', 'entropy', 'acceleration',
            'spectral', 'wavelet', 'cross_timeframe'
        ]
        
        # Check if feature name contains VectorBT keywords
        if any(keyword in feature_name for keyword in vectorbt_keywords):
            return 'vectorbt'
        
        # Check if feature category suggests VectorBT processing
        vectorbt_categories = [
            'momentum', 'volatility', 'returns', 'volume', 'trend', 
            'oscillator', 'support_resistance', 'candlestick_pattern',
            'entropy', 'acceleration', 'spectral_wavelet', 'cross_timeframe'
        ]
        
        if any(category in feature_category for category in vectorbt_categories):
            return 'vectorbt'

        # Pandas features (fallback)
        return 'pandas'

    def _process_vectorbt_batch(self, data: pd.DataFrame, features: List[Dict]) -> Dict[str, pd.Series]:
        """Process VectorBT features in optimized batches."""
        results = {}

        try:
            # Use VectorBT batcher if available
            if self.vectorbt_batcher:
                batch_configs = []
                for feature in features:
                    batch_configs.append({
                        'name': feature.get('name'),
                        'type': 'indicator',
                        'params': feature.get('parameters', {})
                    })

                batch_results = self.vectorbt_batcher.generate_features_batch_optimized(data, batch_configs)
                results.update(batch_results)
            else:
                # Fallback to individual processing
                for feature in features:
                    try:
                        feature_name = feature.get('name')
                        if feature_name:
                            # Generate individual feature
                            feature_series = self._generate_single_feature_optimized(data, feature)
                            if feature_series is not None:
                                results[feature_name] = feature_series
                    except Exception as e:
                        self.logger.warning(f"Failed to generate feature {feature.get('name', 'unknown')}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"VectorBT batch processing failed: {e}")

        return results

    def _process_pandas_batch(self, data: pd.DataFrame, features: List[Dict]) -> Dict[str, pd.Series]:
        """Process pandas features efficiently."""
        results = {}

        for feature in features:
            try:
                feature_name = feature.get('name')
                if feature_name:
                    # Generate individual feature
                    feature_series = self._generate_single_feature_optimized(data, feature)
                    if feature_series is not None:
                        results[feature_name] = feature_series
            except Exception as e:
                self.logger.warning(f"Failed to generate feature {feature.get('name', 'unknown')}: {e}")
                continue

        return results

    def _generate_single_feature_optimized(self, data: pd.DataFrame, feature: Dict) -> Optional[pd.Series]:
        """Generate a single feature with optimizations."""
        try:
            feature_name = feature.get('name')
            feature_type = feature.get('type', 'indicator')
            parameters = feature.get('parameters', {})

            # For pandas features, we need to implement basic feature generation
            # This is a simplified approach - in production, you'd want more sophisticated logic
            
            if feature_type == 'indicator':
                # Try to generate basic technical indicators using pandas
                if 'sma' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        return data[column].rolling(window=window).mean()
                
                elif 'ema' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        return data[column].ewm(span=window).mean()
                
                elif 'rsi' in feature_name.lower():
                    window = parameters.get('window', 14)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        delta = data[column].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                        rs = gain / loss
                        return 100 - (100 / (1 + rs))
                
                elif 'macd' in feature_name.lower():
                    fast = parameters.get('fast', 12)
                    slow = parameters.get('slow', 26)
                    signal = parameters.get('signal', 9)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        ema_fast = data[column].ewm(span=fast).mean()
                        ema_slow = data[column].ewm(span=slow).mean()
                        macd_line = ema_fast - ema_slow
                        return macd_line
                
                elif 'bbands' in feature_name.lower() or 'bollinger' in feature_name.lower():
                    window = parameters.get('window', 20)
                    std_dev = parameters.get('std_dev', 2)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        sma = data[column].rolling(window=window).mean()
                        std = data[column].rolling(window=window).std()
                        return sma + (std * std_dev)  # Upper band
                
                elif 'volume' in feature_name.lower():
                    # Volume-based features
                    if 'sma' in feature_name.lower():
                        window = parameters.get('window', 20)
                        if 'volume' in data.columns:
                            return data['volume'].rolling(window=window).mean()
                    elif 'ema' in feature_name.lower():
                        window = parameters.get('window', 20)
                        if 'volume' in data.columns:
                            return data['volume'].ewm(span=window).mean()
                
                elif 'returns' in feature_name.lower():
                    # Returns-based features
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        if 'log' in feature_name.lower():
                            return np.log(data[column] / data[column].shift(1))
                        else:
                            return data[column].pct_change()
                
                elif 'momentum' in feature_name.lower():
                    window = parameters.get('window', 10)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        return data[column] - data[column].shift(window)
                
                elif 'volatility' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        returns = data[column].pct_change()
                        return returns.rolling(window=window).std()
                
                elif 'trend' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        sma = data[column].rolling(window=window).mean()
                        return (data[column] - sma) / sma
                
                elif 'oscillator' in feature_name.lower():
                    window = parameters.get('window', 14)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        high_col = parameters.get('high_column', 'high')
                        low_col = parameters.get('low_column', 'low')
                        if high_col in data.columns and low_col in data.columns:
                            highest = data[high_col].rolling(window=window).max()
                            lowest = data[low_col].rolling(window=window).min()
                            return ((data[column] - lowest) / (highest - lowest)) * 100
                
                elif 'support_resistance' in feature_name.lower():
                    window = parameters.get('window', 20)
                    high_col = parameters.get('high_column', 'high')
                    low_col = parameters.get('low_column', 'low')
                    if high_col in data.columns and low_col in data.columns:
                        if 'resistance' in feature_name.lower():
                            return data[high_col].rolling(window=window).max()
                        elif 'support' in feature_name.lower():
                            return data[low_col].rolling(window=window).min()
                
                elif 'candlestick' in feature_name.lower():
                    # Basic candlestick patterns
                    if 'doji' in feature_name.lower():
                        if all(col in data.columns for col in ['open', 'close', 'high', 'low']):
                            body = abs(data['close'] - data['open'])
                            range_val = data['high'] - data['low']
                            return (body / range_val) < 0.1  # Doji pattern
                    
                    elif 'hammer' in feature_name.lower():
                        if all(col in data.columns for col in ['open', 'close', 'high', 'low']):
                            body = abs(data['close'] - data['open'])
                            lower_shadow = data[['open', 'close']].min(axis=1) - data['low']
                            upper_shadow = data['high'] - data[['open', 'close']].max(axis=1)
                            return (lower_shadow > 2 * body) & (upper_shadow < body)
                
                elif 'entropy' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        # Simple entropy calculation
                        returns = data[column].pct_change()
                        binned = pd.cut(returns, bins=10, labels=False)
                        entropy = binned.rolling(window=window).apply(
                            lambda x: -sum(p * np.log2(p) for p in x.value_counts(normalize=True) if p > 0)
                        )
                        return entropy
                
                elif 'acceleration' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        # Second derivative (acceleration)
                        first_derivative = data[column].diff()
                        return first_derivative.diff()
                
                elif 'spectral' in feature_name.lower() or 'wavelet' in feature_name.lower():
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        # Simple spectral analysis using FFT
                        rolling_data = data[column].rolling(window=window)
                        return rolling_data.apply(lambda x: np.abs(np.fft.fft(x)).mean() if len(x) == window else np.nan)
                
                elif 'cross_timeframe' in feature_name.lower():
                    # Cross-timeframe features
                    window = parameters.get('window', 20)
                    column = parameters.get('column', 'close')
                    if column in data.columns:
                        # Simple cross-timeframe momentum
                        short_ma = data[column].rolling(window=window//2).mean()
                        long_ma = data[column].rolling(window=window).mean()
                        return (short_ma - long_ma) / long_ma

            # If no specific pattern matched, fast fail
            self.logger.debug(f"⚠️ No pattern matched for {feature_name} - skipping")
            return None

        except Exception as e:
            self.logger.warning(f"Failed to generate single feature {feature.get('name', 'unknown')}: {e}")

        return None


    def _process_feature_batch_streaming(self, data: pd.DataFrame, features: List[Dict], complexity: str) -> pd.DataFrame:
        """Process a batch of features using streaming approach with chunking."""
        try:
            # For streaming, use chunked processing
            chunk_size = self._get_adaptive_chunk_size(data)
            total_rows = len(data)
            num_chunks = (total_rows + chunk_size - 1) // chunk_size

            all_batch_features = []

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_rows)

                # Extract chunk
                chunk_data = data.iloc[start_idx:end_idx].copy()

                # Process this chunk's features
                chunk_features = self._process_feature_batch_optimized(chunk_data, features, complexity)

                if chunk_features is not None and not chunk_features.empty:
                    all_batch_features.append(chunk_features)

                # Memory cleanup
                del chunk_data

            # Combine results from all chunks
            if all_batch_features:
                # For now, return the first chunk's results (can be enhanced to combine properly)
                return all_batch_features[0]
            else:
                return pd.DataFrame(index=data.index)

        except Exception as e:
            self.logger.error(f"Streaming batch processing failed: {e}")
            return pd.DataFrame(index=data.index)

    def _get_adaptive_chunk_size(self, data: pd.DataFrame) -> int:
        """Calculate optimal chunk size based on available memory and data characteristics."""
        if not self.config.enable_adaptive_chunking:
            return self.config.streaming_chunk_size
        
        try:
            # Get current memory usage
            current_memory = self._get_memory_usage()
            available_memory = self.config.max_memory_usage_mb - current_memory
            
            # M1-optimized chunk sizing based on memory pressure
            memory_usage_percent = psutil.virtual_memory().percent
            data_size = len(data)
            
            # Aggressive chunking for high memory pressure
            if memory_usage_percent > 80:
                return min(500, data_size)
            elif memory_usage_percent > 70:
                return min(1000, data_size)
            elif available_memory > 2000:  # > 2GB available
                return min(2000, data_size)
            else:
                return min(1000, data_size)
            
            # Estimate memory needed per row
            data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            memory_per_row = data_size_mb / len(data)
            
            # Calculate safe chunk size (use 50% of available memory)
            safe_memory = available_memory * 0.5
            adaptive_chunk_size = int(safe_memory / memory_per_row) if memory_per_row > 0 else self.config.streaming_chunk_size
            
            # Ensure chunk size is within reasonable bounds
            min_chunk_size = 1000
            max_chunk_size = 10000
            
            adaptive_chunk_size = max(min_chunk_size, min(adaptive_chunk_size, max_chunk_size))
            
            self.logger.info(f"🧠 Adaptive chunk size: {adaptive_chunk_size} (available memory: {available_memory:.1f}MB)")
            return adaptive_chunk_size
            
        except Exception as e:
            self.logger.warning(f"Adaptive chunking failed: {e}, using default size")
            return self.config.streaming_chunk_size
    
    def _process_chunk(self, chunk_data: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
        """Process a single chunk of data."""
        try:
            # Apply data type optimization
            optimized_chunk = self._optimize_data_types(chunk_data)
            
            # Generate features for this chunk
            chunk_features = self._generate_chunk_features(optimized_chunk, chunk_idx)
            
            return chunk_features
            
        except Exception as e:
            self.logger.error(f"Chunk processing failed for chunk {chunk_idx}: {e}")
            # Return empty DataFrame with same index
            return pd.DataFrame(index=chunk_data.index)
    
    def _generate_chunk_features(self, chunk_data: pd.DataFrame, chunk_idx: int) -> pd.DataFrame:
        """Generate features for a specific chunk."""
        # This would contain the actual feature generation logic
        # For now, return the chunk data as-is
        return chunk_data
    
    def _cleanup_chunk_memory(self, chunk_data: pd.DataFrame, chunk_features: pd.DataFrame):
        """Clean up memory after processing a chunk with M1 optimization."""
        try:
            # Delete chunk data
            del chunk_data
            del chunk_features
            
            # M1-optimized memory cleanup
            import gc
            gc.collect()
            
            # Additional cleanup for M1
            if hasattr(self, 'hardware_manager') and self.hardware_manager:
                try:
                    self.hardware_manager.optimize_memory()
                except Exception:
                    pass
            
            # Force garbage collection
            if self.config.enable_aggressive_memory_cleanup:
                gc.collect()
                
        except Exception as e:
            self.logger.warning(f"Chunk memory cleanup failed: {e}")
    
    def _should_trigger_cleanup(self) -> bool:
        """Check if memory cleanup should be triggered."""
        try:
            current_memory = self._get_memory_usage()
            return current_memory > self.config.memory_threshold_mb
        except:
            return False
    
    def _combine_chunk_results(self, feature_results: List[pd.DataFrame], original_index: pd.Index) -> pd.DataFrame:
        """Combine results from all chunks."""
        try:
            if not feature_results:
                return pd.DataFrame(index=original_index)
            
            # Concatenate all results
            combined_features = pd.concat(feature_results, ignore_index=False, sort=False)
            
            # Ensure index alignment
            combined_features = combined_features.reindex(original_index)
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Failed to combine chunk results: {e}")
            return pd.DataFrame(index=original_index)
    
    def _traditional_feature_generation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback to traditional feature generation."""
        self.logger.info("🔄 Using traditional feature generation")
        # This would contain the original feature generation logic
        return data
    
    def _vectorbt_batch_operations(self, data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """Perform VectorBT batch operations for better performance with progress updates."""
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt_batch_processing:
            self.logger.info("🔄 VectorBT not available, using fallback operations")
            return self._fallback_operations(data, operations)
        
        try:
            results = {}
            total_operations = len(operations) * 5  # 5 windows per operation
            completed_operations = 0

            self.logger.info(f"🚀 Starting VectorBT batch operations: {len(operations)} operation types")

            # Use VectorBT's native batch processing
            for i, operation in enumerate(operations):
                self.logger.info(f"🔄 Processing operation {i + 1}/{len(operations)}: {operation}")

                if operation == 'rolling_mean':
                    for w in [5, 10, 20, 50, 100]:
                        results[f'mean_{w}'] = rolling_mean(data['close'], window=w)
                        completed_operations += 1
                elif operation == 'rolling_std':
                    for w in [5, 10, 20, 50, 100]:
                        results[f'std_{w}'] = rolling_std(data['close'], window=w)
                        completed_operations += 1
                elif operation == 'rolling_min':
                    for w in [5, 10, 20, 50, 100]:
                        results[f'min_{w}'] = rolling_min(data['close'], window=w)
                        completed_operations += 1
                elif operation == 'rolling_max':
                    for w in [5, 10, 20, 50, 100]:
                        results[f'max_{w}'] = rolling_max(data['close'], window=w)
                        completed_operations += 1

                # Progress update
                progress_pct = (completed_operations / total_operations) * 100
                self.logger.info(f"✅ Operation {operation} completed ({progress_pct:.1f}%)")

            self.logger.info(f"🎉 VectorBT batch operations completed: {completed_operations} operations")
            return pd.DataFrame(results, index=data.index)

        except Exception as e:
            self.logger.warning(f"VectorBT batch operations failed: {e}")
            return self._fallback_operations(data, operations)
    
    def _fallback_operations(self, data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """Fallback operations when VectorBT is not available."""
        results = {}
        
        for operation in operations:
            if operation == 'rolling_mean':
                for w in [5, 10, 20, 50, 100]:
                    results[f'mean_{w}'] = data['close'].rolling(window=w).mean()
            elif operation == 'rolling_std':
                for w in [5, 10, 20, 50, 100]:
                    results[f'std_{w}'] = data['close'].rolling(window=w).std()
        
        return pd.DataFrame(results, index=data.index)
    
    def _parallel_feature_generation(self, data: pd.DataFrame, feature_configs: List[Dict]) -> pd.DataFrame:
        """Generate features using parallel processing with progress updates."""
        if not self.config.enable_parallel_processing or len(feature_configs) < 2:
            self.logger.info("🔄 Parallel processing disabled or insufficient features, using sequential processing")
            return self._sequential_feature_generation(data, feature_configs)
        
        try:
            # Create feature batches for parallel processing
            optimal_workers = self._get_optimal_workers(len(data))
            
            # Create more batches than workers for better load balancing
            # Use smaller batch sizes for better memory management and load distribution
            target_batches = optimal_workers * 2  # 2x workers for better load balancing (6 workers → 12 batches)
            batch_size = max(1, len(feature_configs) // target_batches)
            
            # Create batches with the calculated batch size
            feature_batches = [feature_configs[i:i + batch_size] 
                              for i in range(0, len(feature_configs), batch_size)]
            
            avg_features_per_batch = len(feature_configs) / len(feature_batches)
            self.logger.info(f"🚀 Starting parallel processing: {len(feature_batches)} batches, {optimal_workers} workers")
            self.logger.info(f"📊 Batch details: {len(feature_configs)} features → {len(feature_batches)} batches (avg {avg_features_per_batch:.1f} features/batch)")
            
            # Process batches in parallel with progress tracking
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickle issues
            results = []
            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_feature_batch_thread_safe, data, batch): (i, batch)
                    for i, batch in enumerate(feature_batches)
                }

                # Process results as they complete for better concurrency
                completed_batches = 0
                for future in as_completed(future_to_batch):
                    batch_idx, batch = future_to_batch[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed_batches += 1

                        # Progress update
                        progress_pct = (completed_batches / len(feature_batches)) * 100
                        features_in_batch = len(batch)
                        self.logger.info(
                            f"✅ Batch {batch_idx + 1}/{len(feature_batches)} completed "
                            f"({progress_pct:.1f}%) - {features_in_batch} features processed"
                        )

                    except Exception as e:
                        self.logger.warning(f"❌ Batch {batch_idx + 1} failed: {e}")
                        # Add empty result to maintain order
                        results.append(pd.DataFrame(index=data.index))
            
            self.logger.info(f"🎉 Parallel processing completed: {completed_batches}/{len(feature_batches)} batches successful")
            
            # Combine results
            return self._combine_feature_results(results, data.index)
            
        except Exception as e:
            self.logger.warning(f"Parallel processing failed: {e}")
            self.logger.info("🔄 Falling back to sequential processing...")
            return self._sequential_feature_generation(data, feature_configs)
    
    def _process_feature_batch(self, data: pd.DataFrame, feature_batch: List[Dict]) -> pd.DataFrame:
        """Process a batch of features with detailed progress tracking."""
        batch_results = []
        successful_features = 0
        failed_features = 0
        total_features_generated = 0
        
        for i, feature_config in enumerate(feature_batch):
            feature_name = feature_config.get('name', f'feature_{i}')
            feature_type = feature_config.get('type', 'unknown')
            category = feature_config.get('category', 'unknown')
            
            try:
                # Log feature processing start
                self.logger.debug(f"🔄 Processing generator: {feature_name} ({feature_type}/{category})")
                
                feature_result = self._compute_single_feature(data, feature_config)
                
                if not feature_result.empty:
                    batch_results.append(feature_result)
                    successful_features += 1
                    total_features_generated += len(feature_result.columns)
                    
                    # Log successful feature completion
                    self.logger.debug(f"✅ Generator {feature_name} completed: {len(feature_result.columns)} features")
                else:
                    failed_features += 1
                    self.logger.warning(f"⚠️ Generator {feature_name} returned empty result")
                
            except Exception as e:
                failed_features += 1
                self.logger.warning(f"❌ Generator {feature_name} failed: {e}")
                continue
        
        # Combine all results
        if batch_results:
            combined_result = pd.concat(batch_results, axis=1, sort=False)
        else:
            combined_result = pd.DataFrame(index=data.index)
        
        # Log batch summary
        total_generators = len(feature_batch)
        success_rate = (successful_features / total_generators) * 100 if total_generators > 0 else 0
        self.logger.info(f"📊 Batch summary: {successful_features}/{total_generators} generators successful "
                        f"({success_rate:.1f}%) - {total_features_generated} total features generated")
        
        return combined_result
    
    def _process_feature_batch_thread_safe(self, data: pd.DataFrame, feature_batch: List[Dict]) -> pd.DataFrame:
        """Thread-safe version of feature batch processing."""
        batch_results = []
        successful_features = 0
        failed_features = 0
        total_features_generated = 0
        
        for i, feature_config in enumerate(feature_batch):
            feature_name = feature_config.get('name', f'feature_{i}')
            feature_type = feature_config.get('type', 'unknown')
            category = feature_config.get('category', 'unknown')
            generator = feature_config.get('generator', None)
            
            try:
                # Log feature processing start
                self.logger.debug(f"🔄 Processing generator: {feature_name} ({feature_type}/{category})")
                
                # Use thread-safe feature computation
                feature_result = self._compute_single_feature_thread_safe(data, feature_config)
                
                if not feature_result.empty:
                    batch_results.append(feature_result)
                    successful_features += 1
                    total_features_generated += len(feature_result.columns)
                    
                    # Log successful feature completion
                    self.logger.debug(f"✅ Generator {feature_name} completed: {len(feature_result.columns)} features")
                else:
                    failed_features += 1
                    self.logger.warning(f"⚠️ Generator {feature_name} returned empty result")
                
            except Exception as e:
                failed_features += 1
                self.logger.warning(f"❌ Generator {feature_name} failed: {e}")
                continue
        
        # Combine all results
        if batch_results:
            combined_result = pd.concat(batch_results, axis=1, sort=False)
        else:
            combined_result = pd.DataFrame(index=data.index)
        
        # Log batch summary
        total_generators = len(feature_batch)
        success_rate = (successful_features / total_generators) * 100 if total_generators > 0 else 0
        self.logger.info(f"📊 Batch summary: {successful_features}/{total_generators} generators successful "
                        f"({success_rate:.1f}%) - {total_features_generated} total features generated")
        
        return combined_result
    
    def _compute_single_feature_thread_safe(self, data: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        """Thread-safe feature computation that avoids pickle issues."""
        feature_name = feature_config.get('name', 'unknown')
        feature_type = feature_config.get('type', 'unknown')
        category = feature_config.get('category', 'unknown')
        generator = feature_config.get('generator', None)
        
        try:
            if feature_type == 'generator' and generator is not None:
                # Use the actual FeatureBank generator (thread-safe)
                try:
                    # Call the generator's generate_features method (with fallback to generate)
                    if hasattr(generator, 'generate_features'):
                        generator_result = generator.generate_features(data)
                    elif hasattr(generator, 'generate'):
                        generator_result = generator.generate(data)
                    else:
                        self.logger.warning(f"Generator {generator.__class__.__name__} has no generate_features or generate method")
                        return pd.DataFrame(index=data.index)
                    
                    if isinstance(generator_result, dict):
                        # Convert dict to DataFrame
                        if generator_result:
                            return pd.DataFrame(generator_result, index=data.index)
                        else:
                            # Empty result, return empty DataFrame
                            return pd.DataFrame(index=data.index)
                    elif isinstance(generator_result, pd.DataFrame):
                        # Use DataFrame directly
                        return generator_result
                    else:
                        # If it returns a Series, convert to DataFrame
                        return pd.DataFrame({feature_name: generator_result}, index=data.index)
                        
                except Exception as gen_e:
                    self.logger.warning(f"Generator {generator.__class__.__name__} failed: {gen_e}")
                    return pd.DataFrame(index=data.index)
                    
            elif feature_type == 'category':
                # Use VectorBT batch operations for category-based features
                if self.config.enable_vectorbt_batch_processing:
                    operations = self._get_operations_for_category(category)
                    batch_result = self._vectorbt_batch_operations(data, operations)
                    return batch_result
            
            # Fallback to basic feature computation
            if 'close' in data.columns:
                # Simple rolling mean as example
                return pd.DataFrame({f'{feature_name}_mean': data['close'].rolling(window=20).mean()}, index=data.index)
            else:
                # Return empty DataFrame
                return pd.DataFrame(index=data.index)
                
        except Exception as e:
            self.logger.warning(f"Feature computation failed for {feature_name}: {e}")
            return pd.DataFrame(index=data.index)
    
    def _compute_single_feature(self, data: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        """Compute features using actual FeatureBank generators."""
        feature_name = feature_config.get('name', 'unknown')
        feature_type = feature_config.get('type', 'unknown')
        category = feature_config.get('category', 'unknown')
        generator = feature_config.get('generator', None)
        
        try:
            if feature_type == 'generator' and generator is not None:
                # Use the actual FeatureBank generator
                try:
                    # Call the generator's generate_features method (with fallback to generate)
                    if hasattr(generator, 'generate_features'):
                        generator_result = generator.generate_features(data)
                    elif hasattr(generator, 'generate'):
                        generator_result = generator.generate(data)
                    else:
                        self.logger.warning(f"Generator {generator.__class__.__name__} has no generate_features or generate method")
                        return pd.DataFrame(index=data.index)
                    
                    if isinstance(generator_result, dict):
                        # Convert dict to DataFrame
                        if generator_result:
                            return pd.DataFrame(generator_result, index=data.index)
                        else:
                            # Empty result, return empty DataFrame
                            return pd.DataFrame(index=data.index)
                    elif isinstance(generator_result, pd.DataFrame):
                        # Use DataFrame directly
                        return generator_result
                    else:
                        # If it returns a Series, convert to DataFrame
                        return pd.DataFrame({feature_name: generator_result}, index=data.index)
                        
                except Exception as gen_e:
                    self.logger.warning(f"Generator {generator.__class__.__name__} failed: {gen_e}")
                    return pd.DataFrame(index=data.index)
                    
            elif feature_type == 'category':
                # Use VectorBT batch operations for category-based features
                if self.config.enable_vectorbt_batch_processing:
                    operations = self._get_operations_for_category(category)
                    batch_result = self._vectorbt_batch_operations(data, operations)
                    return batch_result
            
            # Fallback to basic feature computation
            if 'close' in data.columns:
                # Simple rolling mean as example
                return pd.DataFrame({f'{feature_name}_mean': data['close'].rolling(window=20).mean()}, index=data.index)
            else:
                # Return empty DataFrame
                return pd.DataFrame(index=data.index)
                
        except Exception as e:
            self.logger.warning(f"Feature computation failed for {feature_name}: {e}")
            return pd.DataFrame(index=data.index)
    
    def _get_operations_for_category(self, category: str) -> List[str]:
        """Get operations for a specific category."""
        category_operations = {
            'technical': ['rolling_mean', 'rolling_std'],
            'statistical': ['rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max'],
            'momentum': ['rolling_mean'],
            'volatility': ['rolling_std'],
            'volume': ['rolling_mean', 'rolling_std']
        }
        return category_operations.get(category, ['rolling_mean'])
    
    def _sequential_feature_generation(self, data: pd.DataFrame, feature_configs: List[Dict]) -> pd.DataFrame:
        """Generate features sequentially as fallback with progress updates."""
        results = {}
        successful_features = 0
        failed_features = 0
        total_features = len(feature_configs)
        
        self.logger.info(f"🔄 Starting sequential processing: {total_features} features")
        
        for i, feature_config in enumerate(feature_configs):
            feature_name = feature_config.get('name', f'feature_{i}')
            feature_type = feature_config.get('type', 'unknown')
            category = feature_config.get('category', 'unknown')
            
            try:
                # Progress update every 10 features or at start/end
                if i % 10 == 0 or i == total_features - 1:
                    progress_pct = ((i + 1) / total_features) * 100
                    self.logger.info(f"🔄 Processing feature {i + 1}/{total_features} ({progress_pct:.1f}%): {feature_name}")
                
                feature_result = self._compute_single_feature(data, feature_config)
                results[feature_name] = feature_result
                successful_features += 1
                
            except Exception as e:
                failed_features += 1
                self.logger.warning(f"❌ Feature {feature_name} failed: {e}")
                # Add zero-filled result to maintain DataFrame structure
                results[feature_name] = pd.Series(0, index=data.index, dtype=np.float32)
                continue
        
        # Final summary
        success_rate = (successful_features / total_features) * 100 if total_features > 0 else 0
        self.logger.info(f"🎉 Sequential processing completed: {successful_features}/{total_features} features successful ({success_rate:.1f}%)")
        
        return pd.DataFrame(results, index=data.index)
    
    def _combine_feature_results(self, results: List[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
        """Combine feature results from parallel processing."""
        if not results:
            return pd.DataFrame(index=index)
        
        # Combine all results
        combined = pd.concat(results, axis=1, sort=False)
        return combined.reindex(index)
    
    def _apply_memory_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply memory mapping if beneficial for large datasets."""
        try:
            if self.memory_mapped_handler:
                memory_mapped_data = self.memory_mapped_handler.create_memory_mapped(
                    data, self.config.memory_mapping_threshold_mb
                )
                
                if self.memory_mapped_handler.is_memory_mapped:
                    self.logger.info(f"💾 Using memory-mapped DataFrame for {len(data)} rows")
                    return memory_mapped_data
                else:
                    self.logger.debug("📊 Dataset too small for memory mapping, using in-memory")
                    return data
            else:
                return data
                
        except Exception as e:
            self.logger.warning(f"⚠️ Memory mapping failed: {e}")
            self.logger.info("💡 Falling back to in-memory processing. This is normal for datasets with sparse data.")
            return data

    def _normalize_timestamps_for_comparison(self, ts_series: pd.Series) -> pd.Series:
        """Normalize timestamps to timezone-naive for safe comparison."""
        if hasattr(ts_series.dt, 'tz') and ts_series.dt.tz is not None:
            return ts_series.dt.tz_localize(None)
        return ts_series

    def _apply_light_mode_restriction(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply mode-based data restriction:
        - Light mode: 20 days
        - Blank mode (default): 180 days  
        - Full mode: Full lookback period as defined in ares_launcher
        
        This ensures consistent behavior between FeatureBank and optimized pipeline.
        """
        import os
        import pandas as pd
        import numpy as np
        
        # Simple approach: Check for execution mode flag directly
        # This catches the flag from ares_launcher (light/blank/full)
        exec_mode_str = None
        
        # Check kwargs first - support both 'execution_mode' and 'intensity' parameters
        if 'execution_mode' in kwargs:
            exec_mode = kwargs['execution_mode']
            if hasattr(exec_mode, 'name'):
                exec_mode_str = exec_mode.name.lower()
            elif hasattr(exec_mode, 'value'):
                exec_mode_str = exec_mode.value.lower()
            else:
                exec_mode_str = str(exec_mode).lower()
        elif 'intensity' in kwargs:
            # Handle intensity parameter from ares_launcher
            intensity = kwargs['intensity']
            if hasattr(intensity, 'name'):
                exec_mode_str = intensity.name.lower()
            elif hasattr(intensity, 'value'):
                exec_mode_str = intensity.value.lower()
            else:
                exec_mode_str = str(intensity).lower()
        
        # Check environment variables as fallback
        if not exec_mode_str:
            exec_mode_str = os.environ.get('ARES_EXECUTION_MODE', '').lower()
        
        if not exec_mode_str:
            exec_mode_str = os.environ.get('EXECUTION_MODE', '').lower()
        
        # Check command line arguments as final fallback
        if not exec_mode_str:
            import sys
            for i, arg in enumerate(sys.argv):
                if arg == '--execution-mode' and i + 1 < len(sys.argv):
                    exec_mode_str = sys.argv[i + 1].lower()
                    break
        
        # Debug logging
        self.logger.info(f"🔍 [Pipeline] Execution mode detected: {exec_mode_str}")
        self.logger.info(f"🔍 [Pipeline] Available kwargs: {list(kwargs.keys())}")
        if 'intensity' in kwargs:
            self.logger.info(f"🔍 [Pipeline] Intensity value: {kwargs['intensity']}")
        if 'execution_mode' in kwargs:
            self.logger.info(f"🔍 [Pipeline] Execution mode value: {kwargs['execution_mode']}")
        
        # Check for explicit mode environment variables
        light_mode = os.getenv('LIGHT_MODE', '').lower() in ('1', 'true', 'yes')
        full_mode = os.getenv('FULL_MODE', '').lower() in ('1', 'true', 'yes')
        
        # Determine restriction based on mode
        if full_mode or exec_mode_str == 'full':
            # Full mode: no restriction, use all data
            self.logger.info("📅 [Pipeline] FULL mode: using complete dataset (no restrictions)")
            return data
        elif light_mode or exec_mode_str == 'light':
            # Light mode: 20 days
            restriction_days = 20
            mode_name = "LIGHT"
        else:
            # Blank mode (default): 180 days
            restriction_days = 180
            mode_name = "BLANK"
            
        # Extract timestamp series (same logic as FeatureBank)
        ts_series = None
        if 'timestamp' in data.columns:
            ts_col = data['timestamp']
            try:
                # Check if it's already a datetime type (including timezone-aware)
                if pd.api.types.is_datetime64_any_dtype(ts_col):
                    ts_series = pd.to_datetime(ts_col)
                elif pd.api.types.is_integer_dtype(ts_col):
                    unit = 'ms' if ts_col.dropna().astype(np.int64).median() > 10**12 else 's'
                    ts_series = pd.to_datetime(ts_col, unit=unit, errors='coerce')
                else:
                    ts_series = pd.to_datetime(ts_col, errors='coerce')
                
                # Debug: log timestamp info
                if ts_series is not None and not ts_series.empty:
                    self.logger.info(f"📅 [Pipeline] Timestamp column found: min={ts_series.min()}, max={ts_series.max()}, dtype={ts_series.dtype}")
                    # Additional debug info for timezone-aware columns
                    if hasattr(ts_series.dt, 'tz') and ts_series.dt.tz is not None:
                        self.logger.info(f"📅 [Pipeline] Timezone-aware timestamps detected: {ts_series.dt.tz}")
            except Exception as e:
                self.logger.warning(f"⚠️ [Pipeline] Failed to parse timestamp column: {e}")
                ts_series = None
        
        # Check for other common timestamp column names
        if ts_series is None:
            for col_name in ['time', 'datetime', 'date', 'open_time', 'close_time']:
                if col_name in data.columns:
                    try:
                        ts_series = pd.to_datetime(data[col_name], errors='coerce')
                        if not ts_series.empty and not ts_series.isna().all():
                            self.logger.info(f"📅 [Pipeline] Using timestamp column: {col_name}")
                            break
                    except Exception:
                        continue
        
        if ts_series is None and isinstance(data.index, pd.DatetimeIndex):
            ts_series = data.index.to_series()
        
        # Apply restriction based on mode
        if ts_series is not None and not ts_series.empty:
            # Validate timestamp data - check for invalid dates (1970-01-01 epoch)
            valid_ts = ts_series.dropna()
            if len(valid_ts) == 0:
                self.logger.warning("⚠️ [Pipeline] No valid timestamps found, using fallback restriction")
                ts_series = None
            else:
                # Check if all dates are epoch (1970-01-01) which indicates invalid data
                epoch_date = pd.Timestamp('1970-01-01')
                if valid_ts.nunique() == 1 and valid_ts.iloc[0].date() == epoch_date.date():
                    self.logger.warning("⚠️ [Pipeline] Invalid timestamp data detected (all dates are 1970-01-01), using fallback restriction")
                    ts_series = None
                else:
                    end_dt = ts_series.max()
                    start_dt_full = ts_series.min()
                    
                    # Convert to naive timestamps for comparison if timezone-aware
                    if hasattr(end_dt, 'tz') and end_dt.tz is not None:
                        end_dt = end_dt.tz_localize(None)
                    if hasattr(start_dt_full, 'tz') and start_dt_full.tz is not None:
                        start_dt_full = start_dt_full.tz_localize(None)
                    
                    # Also ensure ts_series is timezone-naive for comparison
                    ts_series = self._normalize_timestamps_for_comparison(ts_series)
                    
                    # Additional validation: check if dates are reasonable (not before 2000)
                    if start_dt_full.year < 2000:
                        self.logger.warning(f"⚠️ [Pipeline] Suspicious start date detected: {start_dt_full.date()}, using fallback restriction")
                        ts_series = None
                    else:
                        cutoff_dt = end_dt - pd.Timedelta(days=restriction_days)
                        # Ensure both timestamps are timezone-naive for comparison
                        if hasattr(cutoff_dt, 'tz') and cutoff_dt.tz is not None:
                            cutoff_dt = cutoff_dt.tz_localize(None)
                        mask = ts_series >= cutoff_dt
                        if mask.any() and mask.sum() < len(ts_series):
                            rows_before = len(data)
                            data = data.loc[mask.values]
                            rows_after = len(data)
                            days_used = (end_dt.normalize() - cutoff_dt.normalize()).days + 1
                            self.logger.info(f"📅 [Pipeline] {mode_name} mode: restricting to last {restriction_days} days: {cutoff_dt.date()} → {end_dt.date()} ({days_used} days, {rows_after}/{rows_before} rows)")
                        else:
                            days_range = (end_dt.normalize() - start_dt_full.normalize()).days + 1
                            self.logger.info(f"📅 [Pipeline] {mode_name} mode: data already within {days_range} days: {start_dt_full.date()} → {end_dt.date()}")
        
        # Fallback when no valid timestamps are found
        if ts_series is None or ts_series.empty:
            self.logger.info(f"📅 [Pipeline] {mode_name} mode: no valid timestamps found, using row-based restriction")
            # Fallback: restrict to percentage of data based on mode
            total_rows = len(data)
            if mode_name == "LIGHT":
                fallback_percentage = 0.1  # 10% for light mode
            else:
                fallback_percentage = 0.3  # 30% for blank mode
                
            if total_rows > 1000:
                rows_to_keep = max(1000, int(total_rows * fallback_percentage))
                data = data.tail(rows_to_keep)
                self.logger.info(f"📅 [Pipeline] {mode_name} mode: no timestamp available, using last {rows_to_keep}/{total_rows} rows ({fallback_percentage*100:.0f}%)")
            else:
                self.logger.info(f"📅 [Pipeline] {mode_name} mode: no timestamp available, data already small ({total_rows} rows)")
        
        return data

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'pipeline_stats': self.performance_stats.copy(),
            'component_status': {
                'feature_bank': self.feature_bank is not None,
                'normalizer': self.normalizer is not None,
                'scaler': self.scaler is not None,
                'matrix_ops': self.matrix_ops is not None,
                'hardware_manager': self.hardware_manager is not None,
                'vectorized_core': self.vectorized_core is not None,
                'memory_mapped_handler': self.memory_mapped_handler is not None,
                'data_type_optimizer': self.data_type_optimizer is not None,
                'intelligent_cache': self.feature_cache is not None
            },
            'config': {
                'auto_normalize': self.config.auto_normalize,
                'normalization_method': self.config.normalization_method,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_matrix_operations': self.config.enable_matrix_operations,
                'workload_type': getattr(self.config.workload_type, 'value', self.config.workload_type),
                'optimization_level': getattr(self.config.optimization_level, 'value', self.config.optimization_level),
                'enable_memory_mapping': self.config.enable_memory_mapping,
                'enable_advanced_data_type_optimization': self.config.enable_advanced_data_type_optimization,
                'enable_intelligent_caching': self.config.enable_intelligent_caching
            }
        }
    
    def get_memory_optimization_stats(self) -> Dict[str, Any]:
        """Get detailed memory optimization statistics."""
        stats = {
            'memory_mapping_enabled': self.config.enable_memory_mapping,
            'advanced_data_type_optimization': self.config.enable_advanced_data_type_optimization,
            'intelligent_caching_enabled': self.config.enable_intelligent_caching,
            'compression_algorithm': self.config.compression_algorithm,
            'sparse_threshold': self.config.sparse_threshold,
            'memory_mapping_threshold_mb': self.config.memory_mapping_threshold_mb
        }
        
        # Add cache statistics if available
        if self.feature_cache:
            stats['cache_entries'] = len(self.feature_cache.cache)
            stats['cache_dependencies'] = len(self.feature_cache.dependencies)
        
        # Add memory-mapped file statistics if available
        if self.memory_mapped_handler:
            stats['memory_mapped_files'] = len(self.memory_mapped_handler.memory_mapped_files)
            stats['is_memory_mapped'] = self.memory_mapped_handler.is_memory_mapped
        
        return stats
    
    def verify_optimizations(self) -> Dict[str, Any]:
        """Verify that all optimizations are properly wired and working."""
        verification_results = {
            'parallel_processing': {
                'enabled': self.config.enable_parallel_processing,
                'max_workers': self.config.max_workers,
                'method_available': hasattr(self, '_parallel_feature_generation'),
                'optimal_workers_method': hasattr(self, '_get_optimal_workers'),
                'adaptive_chunking': hasattr(self, '_get_adaptive_chunk_size')
            },
            'memory_optimization': {
                'enabled': self.config.enable_advanced_data_type_optimization,
                'optimizer_initialized': self.data_type_optimizer is not None,
                'method_available': hasattr(self, '_optimize_data_types'),
                'memory_mapping_enabled': self.config.enable_memory_mapping,
                'memory_mapper_initialized': self.memory_mapped_handler is not None
            },
            'vectorbt_optimization': {
                'enabled': self.config.enable_vectorbt_batch_processing,
                'vectorbt_available': VECTORBT_AVAILABLE,
                'method_available': hasattr(self, '_vectorbt_batch_operations'),
                'chunk_size': self.config.vectorbt_chunk_size
            },
            'intelligent_caching': {
                'enabled': self.config.enable_intelligent_caching,
                'cache_initialized': self.feature_cache is not None,
                'ttl_seconds': self.config.cache_ttl_seconds,
                'incremental_caching': self.config.enable_incremental_caching
            },
            'm1_optimizations': {
                'enabled': self.config.enable_m1_optimizations,
                'mps_acceleration': self.config.enable_mps_acceleration,
                'method_available': hasattr(self, '_m1_optimizations')
            },
            'hardware_optimization': {
                'enabled': self.config.enable_hardware_optimization,
                'hardware_manager_initialized': self.hardware_manager is not None,
                'vectorized_core_initialized': self.vectorized_core is not None
            }
        }
        
        # Check if all critical methods are wired
        critical_methods = [
            '_parallel_feature_generation',
            '_optimize_data_types', 
            '_vectorbt_batch_operations',
            '_apply_memory_mapping',
            '_m1_optimizations'
        ]
        
        missing_methods = [method for method in critical_methods if not hasattr(self, method)]
        verification_results['missing_methods'] = missing_methods
        verification_results['all_methods_available'] = len(missing_methods) == 0
        
        return verification_results
    
    def force_optimization_usage(self) -> Dict[str, Any]:
        """Force the use of our optimizations by adjusting configuration."""
        original_config = {
            'enable_parallel_processing': self.config.enable_parallel_processing,
            'memory_mapping_threshold_mb': self.config.memory_mapping_threshold_mb,
            'sparse_threshold': self.config.sparse_threshold
        }
        
        # Force more aggressive optimization settings
        self.config.enable_parallel_processing = True
        self.config.memory_mapping_threshold_mb = 25  # Very aggressive
        self.config.sparse_threshold = 0.2  # Very aggressive
        
        self.logger.info("🚀 Forced optimization settings:")
        self.logger.info(f"   - Parallel processing: {self.config.enable_parallel_processing}")
        self.logger.info(f"   - Memory mapping threshold: {self.config.memory_mapping_threshold_mb}MB")
        self.logger.info(f"   - Sparse threshold: {self.config.sparse_threshold}")
        
        return {
            'original_config': original_config,
            'new_config': {
                'enable_parallel_processing': self.config.enable_parallel_processing,
                'memory_mapping_threshold_mb': self.config.memory_mapping_threshold_mb,
                'sparse_threshold': self.config.sparse_threshold
            },
            'optimization_forced': True
        }
    
    def get_batch_processing_stats(self) -> Dict[str, Any]:
        """Get detailed batch processing statistics."""
        stats = {
            'parallel_processing_enabled': self.config.enable_parallel_processing,
            'max_workers': self.config.max_workers,
            'chunk_size': self.config.chunk_size,
            'vectorbt_batch_processing': self.config.enable_vectorbt_batch_processing,
            'vectorbt_chunk_size': self.config.vectorbt_chunk_size,
            'memory_mapping_enabled': self.config.enable_memory_mapping,
            'memory_mapping_threshold_mb': self.config.memory_mapping_threshold_mb
        }
        
        # Add performance stats if available
        if hasattr(self, 'performance_stats'):
            stats.update({
                'total_executions': self.performance_stats.get('total_executions', 0),
                'successful_executions': self.performance_stats.get('successful_executions', 0),
                'failed_executions': self.performance_stats.get('failed_executions', 0),
                'average_processing_time': self.performance_stats.get('average_processing_time', 0.0),
                'peak_memory_usage': self.performance_stats.get('peak_memory_usage', 0.0)
            })
        
        return stats

    def _update_performance_stats(self, processing_time: float, memory_usage: float = 0.0, success: bool = True, **kwargs):
        """Update performance statistics."""
        # Safety check - ensure performance_stats is initialized
        if not hasattr(self, 'performance_stats'):
            self.performance_stats = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_processing_time': 0.0,
                'peak_memory_usage': 0.0,
                'vectorized_operations': 0,
                'hardware_accelerations': 0
            }
        
        self.performance_stats['total_executions'] += 1
        if success:
            self.performance_stats['successful_executions'] += 1
        else:
            self.performance_stats['failed_executions'] += 1

        # Update average processing time
        total_time = self.performance_stats['average_processing_time'] * (self.performance_stats['total_executions'] - 1)
        self.performance_stats['average_processing_time'] = (total_time + processing_time) / self.performance_stats['total_executions']

        # Update peak memory usage
        self.performance_stats['peak_memory_usage'] = max(
            self.performance_stats['peak_memory_usage'],
            memory_usage
        )
        
        # Handle additional stats from kwargs
        if 'features_count' in kwargs:
            self.performance_stats['last_features_count'] = kwargs['features_count']
        if 'strategy' in kwargs:
            self.performance_stats['last_strategy'] = kwargs['strategy']

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.hardware_manager:
                self.hardware_manager.shutdown()
            if self.feature_bank:
                self.feature_bank.clear_cache()
            
            # Cleanup intelligent cache
            if self.feature_cache:
                self.feature_cache.cleanup_expired()
            
            # Cleanup memory-mapped files
            if self.memory_mapped_handler:
                self.memory_mapped_handler.cleanup()
            
            # Force aggressive garbage collection
            collected = 0
            if self.config.enable_aggressive_memory_cleanup:
                for _ in range(5):  # More aggressive cleanup
                    collected += gc.collect()
            else:
                collected = gc.collect()
            
            # Clear any remaining references
            self.matrix_ops = None
            self.vectorized_core = None
            
            self.logger.info(f"🧹 Pipeline cleanup completed: {collected} objects collected")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def force_memory_cleanup(self):
        """Force aggressive memory cleanup."""
        try:
            # Multiple rounds of garbage collection
            total_collected = 0
            cleanup_rounds = 5 if self.config.enable_aggressive_memory_cleanup else 1
            
            for _ in range(cleanup_rounds):
                collected = gc.collect()
                total_collected += collected
            
            # Clear feature bank cache
            if self.feature_bank:
                self.feature_bank.clear_cache()
            
            # Clear intelligent cache if enabled
            if self.feature_cache:
                self.feature_cache.cleanup_expired()
            
            # Clear memory-mapped files if enabled
            if self.memory_mapped_handler:
                self.memory_mapped_handler.cleanup()
            
            # Clear any temporary data
            if hasattr(self, 'temp_data'):
                del self.temp_data
                self.temp_data = None
            
            self.logger.info(f"🧹 Forced memory cleanup: {total_collected} objects collected")
            return total_collected
            
        except Exception as e:
            self.logger.error(f"Forced cleanup error: {e}")
            return 0
    
    def _enhanced_memory_cleanup(self):
        """Enhanced memory cleanup with monitoring and adaptive strategies."""
        try:
            initial_memory = self._get_memory_usage()
            
            # Clear feature cache aggressively
            if self.feature_cache:
                self.feature_cache.cleanup_expired()
                # Clear additional cache entries if memory is high
                if initial_memory > self.config.memory_threshold_mb:
                    self.feature_cache.clear_old_entries(max_age_seconds=3600)  # 1 hour = 3600 seconds
            
            # Clear memory-mapped files
            if self.memory_mapped_handler:
                self.memory_mapped_handler.cleanup()
            
            # Clear temporary variables
            temp_vars = ['temp_data', 'cached_features', 'intermediate_results']
            for var in temp_vars:
                if hasattr(self, var):
                    delattr(self, var)
            
            # Multiple rounds of garbage collection
            total_collected = 0
            for round_num in range(3):
                collected = gc.collect()
                total_collected += collected
                
                # Check if we've freed enough memory
                current_memory = self._get_memory_usage()
                if current_memory < initial_memory * 0.8:  # 20% reduction
                    break
            
            final_memory = self._get_memory_usage()
            memory_freed = initial_memory - final_memory
            
            self.logger.info(f"🧹 Enhanced cleanup: {total_collected} objects, {memory_freed:.1f}MB freed")
            return total_collected, memory_freed
            
        except Exception as e:
            self.logger.error(f"Enhanced cleanup failed: {e}")
            return 0, 0
    
    def _monitor_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed."""
        try:
            # Throttle frequent checks (once every 2 seconds)
            now = time.time()
            if getattr(self, "_last_memory_check", 0.0) and now - self._last_memory_check < 2.0:
                return False
            self._last_memory_check = now

            current_memory = self._get_memory_usage()
            
            # Check if we're approaching memory limits
            if current_memory > self.config.max_memory_usage_mb:
                self.logger.warning(f"⚠️ Memory usage critical: {current_memory:.1f}MB > {self.config.max_memory_usage_mb}MB")
                self._enhanced_memory_cleanup()
                return True
            
            # Check if we should trigger cleanup
            elif current_memory > self.config.memory_threshold_mb:
                self.logger.info(f"🧠 Memory usage high: {current_memory:.1f}MB, triggering cleanup")
                self.force_memory_cleanup()
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Memory monitoring failed: {e}")
            return False

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    # Gracefully degrade when VectorBT is not installed
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None

    # GPU acceleration placeholder (not used here)
    cp = None


# Global instance
_optimized_pipeline: Optional[OptimizedFeaturePipeline] = None

def get_optimized_feature_pipeline(config: Optional[PipelineConfig] = None) -> OptimizedFeaturePipeline:
    """Get or create the global optimized feature pipeline instance."""
    global _optimized_pipeline

    if _optimized_pipeline is None:
        _optimized_pipeline = OptimizedFeaturePipeline(config)

    return _optimized_pipeline

def process_features_optimized(data: pd.DataFrame,
                             categories: Optional[List[str]] = None,
                             features: Optional[List[str]] = None,
                             target_column: Optional[str] = None,
                             config: Optional[PipelineConfig] = None,
                             **kwargs) -> PipelineResult:
    """
    Convenience function to process features through the optimized pipeline.

    Args:
        data: Input DataFrame
        categories: List of feature categories to generate
        features: List of specific features to generate
        target_column: Target column for lookback optimization
        config: Optional pipeline configuration
        **kwargs: Additional parameters

    Returns:
        PipelineResult with processed features and metadata
    """
    pipeline = get_optimized_feature_pipeline(config)
    return pipeline.process_features(data, categories, features, target_column, **kwargs)
