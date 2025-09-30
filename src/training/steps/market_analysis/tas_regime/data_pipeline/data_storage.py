"""
Data Storage for TAS

Comprehensive data storage system for tree architecture search including
data persistence, caching, and retrieval for historical data processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """Storage formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    FEATHER = "feather"


class StorageType(Enum):
    """Storage types."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    DATABASE = "database"
    MEMORY = "memory"


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    
    # Storage type and format
    storage_type: StorageType = StorageType.LOCAL
    storage_format: StorageFormat = StorageFormat.PARQUET
    
    # Storage paths
    base_directory: str = "tas_data"
    data_directory: str = "data"
    cache_directory: str = "cache"
    metadata_directory: str = "metadata"
    
    # Storage options
    enable_compression: bool = True
    compression_type: str = "gzip"  # "gzip", "snappy", "lz4", "brotli"
    enable_indexing: bool = True
    enable_partitioning: bool = True
    
    # Caching options
    enable_caching: bool = True
    cache_size_mb: int = 1000
    cache_ttl_hours: int = 24
    cache_eviction_policy: str = "lru"  # "lru", "lfu", "fifo"
    
    # Data organization
    organize_by_date: bool = True
    organize_by_symbol: bool = True
    organize_by_timeframe: bool = True
    date_format: str = "%Y%m%d"
    
    # Metadata storage
    enable_metadata: bool = True
    metadata_format: str = "json"
    metadata_compression: bool = True
    
    # Performance options
    enable_parallel_io: bool = True
    max_workers: int = 4
    chunk_size_mb: int = 100
    
    # Output configuration
    save_storage_info: bool = True
    output_directory: str = "storage_info"


@dataclass
class StorageResult:
    """Result of data storage operations."""
    
    # Storage information
    storage_path: str
    storage_format: str
    storage_size_mb: float
    compression_ratio: float
    
    # Data information
    data_shape: Tuple[int, int]
    data_columns: List[str]
    data_types: Dict[str, str]
    data_range: Tuple[datetime, datetime]
    
    # Storage metadata
    storage_metadata: Dict[str, Any]
    storage_time: float
    storage_success: bool
    
    # Performance metrics
    write_time: float
    read_time: float
    compression_time: float
    
    # Cache information
    cache_hit: bool
    cache_size_mb: float
    cache_ttl_hours: float
    
    # Metadata
    config: StorageConfig
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataStorageManager:
    """
    Comprehensive data storage manager for TAS.
    
    Provides data persistence, caching, and retrieval
    for tree architecture search.
    """
    
    def __init__(self, config: StorageConfig):
        """Initialize data storage manager.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize storage directories
        self._initialize_directories()
        
        # Initialize cache
        self.cache = {}
        self.cache_metadata = {}
        
        self.logger.info("✅ Data Storage Manager initialized")
        self.logger.info(f"📊 Storage type: {config.storage_type.value}")
        self.logger.info(f"📊 Storage format: {config.storage_format.value}")
        self.logger.info(f"📊 Base directory: {config.base_directory}")
        self.logger.info(f"📊 Caching enabled: {config.enable_caching}")
    
    def _initialize_directories(self):
        """Initialize storage directories."""
        try:
            base_dir = Path(self.config.base_directory)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (base_dir / self.config.data_directory).mkdir(exist_ok=True)
            (base_dir / self.config.cache_directory).mkdir(exist_ok=True)
            (base_dir / self.config.metadata_directory).mkdir(exist_ok=True)
            
            self.logger.info("✅ Storage directories initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize directories: {e}")
    
    def store_data(self, data: pd.DataFrame, 
                   data_type: str = "data",
                   symbol: str = "BTCUSDT",
                   timeframe: str = "1h",
                   metadata: Optional[Dict[str, Any]] = None) -> StorageResult:
        """
        Store data in storage system.
        
        Args:
            data: Data to store
            data_type: Type of data (e.g., "raw", "processed", "features", "regimes")
            symbol: Trading symbol
            timeframe: Data timeframe
            metadata: Optional metadata
            
        Returns:
            Storage result
        """
        self.logger.info(f"🚀 Storing {data_type} data for {symbol} {timeframe}")
        start_time = datetime.now()
        
        try:
            # Generate storage path
            storage_path = self._generate_storage_path(data_type, symbol, timeframe)
            
            # Check cache first
            cache_hit = False
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(data_type, symbol, timeframe)
                if cache_key in self.cache:
                    cache_hit = True
                    self.logger.info("📦 Data found in cache")
            
            # Store data
            if not cache_hit:
                storage_success = self._write_data(data, storage_path)
                if not storage_success:
                    raise Exception("Failed to write data to storage")
            
            # Calculate storage metrics
            storage_size_mb = self._calculate_storage_size(storage_path)
            compression_ratio = self._calculate_compression_ratio(data, storage_path)
            
            # Store metadata
            storage_metadata = self._create_storage_metadata(data, data_type, symbol, timeframe, metadata)
            if self.config.enable_metadata:
                self._store_metadata(storage_metadata, storage_path)
            
            # Update cache
            if self.config.enable_caching and not cache_hit:
                self._update_cache(cache_key, data, storage_path)
            
            # Calculate performance metrics
            storage_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = StorageResult(
                # Storage information
                storage_path=str(storage_path),
                storage_format=self.config.storage_format.value,
                storage_size_mb=storage_size_mb,
                compression_ratio=compression_ratio,
                
                # Data information
                data_shape=data.shape,
                data_columns=list(data.columns),
                data_types=data.dtypes.to_dict(),
                data_range=(data.index[0], data.index[-1]) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
                
                # Storage metadata
                storage_metadata=storage_metadata,
                storage_time=storage_time,
                storage_success=True,
                
                # Performance metrics
                write_time=storage_time,
                read_time=0.0,  # Would be measured during read operations
                compression_time=0.0,  # Would be measured during compression
                
                # Cache information
                cache_hit=cache_hit,
                cache_size_mb=len(self.cache) * 0.001,  # Approximate
                cache_ttl_hours=self.config.cache_ttl_hours,
                
                # Metadata
                config=self.config
            )
            
            # Save storage info if configured
            if self.config.save_storage_info:
                self._save_storage_info(result)
            
            self.logger.info(f"✅ Data stored successfully in {result.storage_time:.2f}s")
            self.logger.info(f"📊 Storage path: {result.storage_path}")
            self.logger.info(f"📊 Storage size: {result.storage_size_mb:.2f} MB")
            self.logger.info(f"📊 Compression ratio: {result.compression_ratio:.3f}")
            self.logger.info(f"📊 Cache hit: {result.cache_hit}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data storage failed: {e}")
            raise
    
    def retrieve_data(self, data_type: str = "data",
                     symbol: str = "BTCUSDT",
                     timeframe: str = "1h",
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> StorageResult:
        """
        Retrieve data from storage system.
        
        Args:
            data_type: Type of data to retrieve
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Storage result with retrieved data
        """
        self.logger.info(f"🚀 Retrieving {data_type} data for {symbol} {timeframe}")
        start_time = datetime.now()
        
        try:
            # Generate storage path
            storage_path = self._generate_storage_path(data_type, symbol, timeframe)
            
            # Check cache first
            cache_hit = False
            data = None
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(data_type, symbol, timeframe)
                if cache_key in self.cache:
                    cache_hit = True
                    data = self.cache[cache_key]
                    self.logger.info("📦 Data retrieved from cache")
            
            # Retrieve data from storage
            if not cache_hit:
                data = self._read_data(storage_path)
                if data is None:
                    raise Exception("Failed to read data from storage")
                
                # Update cache
                if self.config.enable_caching:
                    self._update_cache(cache_key, data, storage_path)
            
            # Apply date filters if provided
            if start_date or end_date:
                data = self._apply_date_filters(data, start_date, end_date)
            
            # Calculate storage metrics
            storage_size_mb = self._calculate_storage_size(storage_path)
            compression_ratio = self._calculate_compression_ratio(data, storage_path)
            
            # Load metadata
            storage_metadata = self._load_metadata(storage_path)
            
            # Calculate performance metrics
            storage_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive result
            result = StorageResult(
                # Storage information
                storage_path=str(storage_path),
                storage_format=self.config.storage_format.value,
                storage_size_mb=storage_size_mb,
                compression_ratio=compression_ratio,
                
                # Data information
                data_shape=data.shape,
                data_columns=list(data.columns),
                data_types=data.dtypes.to_dict(),
                data_range=(data.index[0], data.index[-1]) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
                
                # Storage metadata
                storage_metadata=storage_metadata,
                storage_time=storage_time,
                storage_success=True,
                
                # Performance metrics
                write_time=0.0,  # Not applicable for read operations
                read_time=storage_time,
                compression_time=0.0,  # Not applicable for read operations
                
                # Cache information
                cache_hit=cache_hit,
                cache_size_mb=len(self.cache) * 0.001,  # Approximate
                cache_ttl_hours=self.config.cache_ttl_hours,
                
                # Metadata
                config=self.config
            )
            
            # Store retrieved data in result for access
            result.retrieved_data = data
            
            self.logger.info(f"✅ Data retrieved successfully in {result.storage_time:.2f}s")
            self.logger.info(f"📊 Data shape: {result.data_shape}")
            self.logger.info(f"📊 Cache hit: {result.cache_hit}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Data retrieval failed: {e}")
            raise
    
    def _generate_storage_path(self, data_type: str, symbol: str, timeframe: str) -> Path:
        """Generate storage path for data."""
        try:
            base_dir = Path(self.config.base_directory)
            data_dir = base_dir / self.config.data_directory
            
            # Organize by symbol and timeframe
            if self.config.organize_by_symbol:
                data_dir = data_dir / symbol.lower()
            
            if self.config.organize_by_timeframe:
                data_dir = data_dir / timeframe
            
            # Create directory if it doesn't exist
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{symbol.lower()}_{timeframe}_{timestamp}"
            
            # Add file extension
            if self.config.storage_format == StorageFormat.PARQUET:
                filename += ".parquet"
            elif self.config.storage_format == StorageFormat.CSV:
                filename += ".csv"
            elif self.config.storage_format == StorageFormat.JSON:
                filename += ".json"
            elif self.config.storage_format == StorageFormat.PICKLE:
                filename += ".pkl"
            elif self.config.storage_format == StorageFormat.HDF5:
                filename += ".h5"
            elif self.config.storage_format == StorageFormat.FEATHER:
                filename += ".feather"
            
            return data_dir / filename
            
        except Exception as e:
            self.logger.warning(f"⚠️ Storage path generation failed: {e}")
            return Path(self.config.base_directory) / f"{data_type}_{symbol}_{timeframe}.{self.config.storage_format.value}"
    
    def _generate_cache_key(self, data_type: str, symbol: str, timeframe: str) -> str:
        """Generate cache key for data."""
        return f"{data_type}_{symbol}_{timeframe}"

    def generate_cache_key(self, data_type: str, symbol: str, timeframe: str,
                            suffix: Optional[str] = None) -> str:
        """Generate a cache key with an optional suffix."""
        base_key = self._generate_cache_key(data_type, symbol, timeframe)
        if suffix:
            return f"{base_key}_{suffix}"
        return base_key
    
    def _write_data(self, data: pd.DataFrame, storage_path: Path) -> bool:
        """Write data to storage."""
        try:
            if self.config.storage_format == StorageFormat.PARQUET:
                data.to_parquet(storage_path, compression=self.config.compression_type if self.config.enable_compression else None)
            elif self.config.storage_format == StorageFormat.CSV:
                data.to_csv(storage_path, compression='gzip' if self.config.enable_compression else None)
            elif self.config.storage_format == StorageFormat.JSON:
                data.to_json(storage_path, orient='index', date_format='iso')
            elif self.config.storage_format == StorageFormat.PICKLE:
                with open(storage_path, 'wb') as f:
                    pickle.dump(data, f)
            elif self.config.storage_format == StorageFormat.HDF5:
                data.to_hdf(storage_path, key='data', mode='w', complevel=9 if self.config.enable_compression else 0)
            elif self.config.storage_format == StorageFormat.FEATHER:
                data.to_feather(storage_path, compression='zstd' if self.config.enable_compression else None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data write failed: {e}")
            return False
    
    def _read_data(self, storage_path: Path) -> Optional[pd.DataFrame]:
        """Read data from storage."""
        try:
            if not storage_path.exists():
                self.logger.warning(f"⚠️ Storage path does not exist: {storage_path}")
                return None
            
            if self.config.storage_format == StorageFormat.PARQUET:
                return pd.read_parquet(storage_path)
            elif self.config.storage_format == StorageFormat.CSV:
                return pd.read_csv(storage_path, index_col=0, parse_dates=True)
            elif self.config.storage_format == StorageFormat.JSON:
                return pd.read_json(storage_path, orient='index', date_unit='s')
            elif self.config.storage_format == StorageFormat.PICKLE:
                with open(storage_path, 'rb') as f:
                    return pickle.load(f)
            elif self.config.storage_format == StorageFormat.HDF5:
                return pd.read_hdf(storage_path, key='data')
            elif self.config.storage_format == StorageFormat.FEATHER:
                return pd.read_feather(storage_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Data read failed: {e}")
            return None
    
    def _apply_date_filters(self, data: pd.DataFrame, start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.DataFrame:
        """Apply date filters to data."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return data
            
            if start_date:
                data = data[data.index >= start_date]
            
            if end_date:
                data = data[data.index <= end_date]
            
            return data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Date filtering failed: {e}")
            return data
    
    def _calculate_storage_size(self, storage_path: Path) -> float:
        """Calculate storage size in MB."""
        try:
            if storage_path.exists():
                return storage_path.stat().st_size / (1024 * 1024)
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"⚠️ Storage size calculation failed: {e}")
            return 0.0
    
    def _calculate_compression_ratio(self, data: pd.DataFrame, storage_path: Path) -> float:
        """Calculate compression ratio."""
        try:
            # Calculate original size (approximate)
            original_size = data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Calculate compressed size
            compressed_size = self._calculate_storage_size(storage_path)
            
            if compressed_size > 0:
                return original_size / compressed_size
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"⚠️ Compression ratio calculation failed: {e}")
            return 1.0
    
    def _create_storage_metadata(self, data: pd.DataFrame, data_type: str, symbol: str, 
                               timeframe: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create storage metadata."""
        try:
            storage_metadata = {
                'data_type': data_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'data_shape': data.shape,
                'data_columns': list(data.columns),
                'data_types': data.dtypes.to_dict(),
                'data_range': (data.index[0], data.index[-1]) if isinstance(data.index, pd.DatetimeIndex) else (None, None),
                'created_at': datetime.now().isoformat(),
                'storage_format': self.config.storage_format.value,
                'compression_enabled': self.config.enable_compression,
                'compression_type': self.config.compression_type
            }
            
            if metadata:
                storage_metadata.update(metadata)
            
            return storage_metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ Storage metadata creation failed: {e}")
            return {}
    
    def _store_metadata(self, metadata: Dict[str, Any], storage_path: Path):
        """Store metadata to file."""
        try:
            metadata_path = storage_path.with_suffix('.metadata.json')
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Metadata storage failed: {e}")
    
    def _load_metadata(self, storage_path: Path) -> Dict[str, Any]:
        """Load metadata from file."""
        try:
            metadata_path = storage_path.with_suffix('.metadata.json')
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Metadata loading failed: {e}")
            return {}
    
    def _update_cache(self, cache_key: str, data: Any,
                      storage_path: Optional[Path] = None,
                      ttl_hours: Optional[float] = None):
        """Update cache with data."""
        try:
            if self.config.enable_caching:
                # Check cache size limit
                if len(self.cache) >= self.config.cache_size_mb:
                    self._evict_cache()

                # Add to cache
                if hasattr(data, 'copy'):
                    try:
                        self.cache[cache_key] = data.copy()
                    except TypeError:
                        self.cache[cache_key] = data
                else:
                    self.cache[cache_key] = data
                self.cache_metadata[cache_key] = {
                    'storage_path': str(storage_path) if storage_path else None,
                    'created_at': datetime.now(),
                    'ttl_hours': ttl_hours if ttl_hours is not None else self.config.cache_ttl_hours
                }

        except Exception as e:
            self.logger.warning(f"⚠️ Cache update failed: {e}")

    def set_cache_entry(self, cache_key: str, data: Any,
                        ttl_hours: Optional[float] = None,
                        storage_path: Optional[Path] = None):
        """Store a cache entry with an optional TTL override."""
        self._update_cache(cache_key, data, storage_path, ttl_hours)

    def get_cache_entry(self, cache_key: str) -> Optional[Any]:
        """Retrieve a cache entry if it hasn't expired."""
        try:
            if not self.config.enable_caching:
                return None

            metadata = self.cache_metadata.get(cache_key)
            if not metadata:
                return None

            created_at = metadata.get('created_at')
            ttl_hours = metadata.get('ttl_hours', self.config.cache_ttl_hours)

            if isinstance(created_at, datetime) and ttl_hours:
                if datetime.now() - created_at > timedelta(hours=ttl_hours):
                    self.logger.info(f"🗑️ Cache entry expired for key: {cache_key}")
                    self.invalidate_cache_entry(cache_key)
                    return None

            return self.cache.get(cache_key)
        except Exception as e:
            self.logger.warning(f"⚠️ Cache retrieval failed for {cache_key}: {e}")
            return None

    def invalidate_cache_entry(self, cache_key: str):
        """Remove a cache entry and its metadata."""
        try:
            if cache_key in self.cache:
                del self.cache[cache_key]
            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]
        except Exception as e:
            self.logger.warning(f"⚠️ Cache invalidation failed for {cache_key}: {e}")

    def _evict_cache(self):
        """Evict data from cache based on eviction policy."""
        try:
            if self.config.cache_eviction_policy == "lru":
                # Remove least recently used
                oldest_key = min(self.cache_metadata.keys(), 
                                key=lambda k: self.cache_metadata[k]['created_at'])
                del self.cache[oldest_key]
                del self.cache_metadata[oldest_key]
            elif self.config.cache_eviction_policy == "lfu":
                # Remove least frequently used (simplified)
                if self.cache:
                    key_to_remove = list(self.cache.keys())[0]
                    del self.cache[key_to_remove]
                    if key_to_remove in self.cache_metadata:
                        del self.cache_metadata[key_to_remove]
            elif self.config.cache_eviction_policy == "fifo":
                # Remove first in, first out
                if self.cache:
                    key_to_remove = list(self.cache.keys())[0]
                    del self.cache[key_to_remove]
                    if key_to_remove in self.cache_metadata:
                        del self.cache_metadata[key_to_remove]
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Cache eviction failed: {e}")
    
    def _save_storage_info(self, result: StorageResult):
        """Save storage information."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"storage_info_{timestamp}.json"
            filepath = output_dir / filename
            
            storage_info = {
                'storage_path': result.storage_path,
                'storage_format': result.storage_format,
                'storage_size_mb': result.storage_size_mb,
                'compression_ratio': result.compression_ratio,
                'data_shape': result.data_shape,
                'data_columns': result.data_columns,
                'data_types': result.data_types,
                'data_range': result.data_range,
                'storage_metadata': result.storage_metadata,
                'storage_time': result.storage_time,
                'storage_success': result.storage_success,
                'write_time': result.write_time,
                'read_time': result.read_time,
                'compression_time': result.compression_time,
                'cache_hit': result.cache_hit,
                'cache_size_mb': result.cache_size_mb,
                'cache_ttl_hours': result.cache_ttl_hours
            }
            
            with open(filepath, 'w') as f:
                json.dump(storage_info, f, indent=2, default=str)
            
            self.logger.info(f"📁 Storage info saved to {filepath}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save storage info: {e}")
    
    def clear_cache(self):
        """Clear the cache."""
        try:
            self.cache.clear()
            self.cache_metadata.clear()
            self.logger.info("✅ Cache cleared")
        except Exception as e:
            self.logger.warning(f"⚠️ Cache clearing failed: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        try:
            return {
                'cache_size': len(self.cache),
                'cache_metadata_size': len(self.cache_metadata),
                'cache_keys': list(self.cache.keys()),
                'cache_metadata_keys': list(self.cache_metadata.keys())
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Cache info retrieval failed: {e}")
            return {}
    
    def export_storage_info(self, result: StorageResult, filepath: str):
        """Export storage information to file."""
        try:
            storage_info = {
                'storage_path': result.storage_path,
                'storage_format': result.storage_format,
                'storage_size_mb': result.storage_size_mb,
                'compression_ratio': result.compression_ratio,
                'data_shape': result.data_shape,
                'data_columns': result.data_columns,
                'data_types': result.data_types,
                'data_range': result.data_range,
                'storage_metadata': result.storage_metadata,
                'storage_time': result.storage_time,
                'storage_success': result.storage_success,
                'cache_hit': result.cache_hit,
                'cache_size_mb': result.cache_size_mb
            }
            
            with open(filepath, 'w') as f:
                json.dump(storage_info, f, indent=2, default=str)
            
            self.logger.info(f"📁 Storage info exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export storage info: {e}")