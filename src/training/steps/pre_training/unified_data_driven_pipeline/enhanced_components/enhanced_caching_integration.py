"""
Enhanced Caching Integration with Advanced Features

This module provides comprehensive caching integration with FeatureCacheService,
artifact management, and performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import json
import hashlib
import pickle
import joblib
from datetime import datetime, timedelta

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import FeatureCacheService
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    FEATURE_CACHE_AVAILABLE = True
    tprint_info("✅ FeatureCacheService available")
except ImportError:
    FEATURE_CACHE_AVAILABLE = False
    tprint_warning("⚠️ FeatureCacheService not available")

# Import serialization utilities
try:
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    SERIALIZATION_AVAILABLE = True
    tprint_info("✅ Serialization utilities available")
except ImportError:
    SERIALIZATION_AVAILABLE = False
    tprint_warning("⚠️ Serialization utilities not available")

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the cache."""
    
    key: str
    data: Any
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    def __post_init__(self):
        """Validate cache entry."""
        assert self.key, "Cache key is required"
        assert self.data is not None, "Cache data is required"
        assert isinstance(self.metadata, dict), "Metadata must be dict"
        assert isinstance(self.created_at, datetime), "created_at must be datetime"
        assert isinstance(self.last_accessed, datetime), "last_accessed must be datetime"
        assert self.access_count >= 0, "access_count must be non-negative"
        assert self.size_bytes >= 0, "size_bytes must be non-negative"


@dataclass
class CacheStats:
    """Cache statistics."""
    
    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    hit_rate: float
    average_access_time: float
    memory_usage_mb: float
    
    def __post_init__(self):
        """Validate cache stats."""
        assert self.total_entries >= 0, "total_entries must be non-negative"
        assert self.total_size_bytes >= 0, "total_size_bytes must be non-negative"
        assert self.hit_count >= 0, "hit_count must be non-negative"
        assert self.miss_count >= 0, "miss_count must be non-negative"
        assert 0 <= self.hit_rate <= 1, "hit_rate must be between 0 and 1"


@dataclass
class ArtifactMetadata:
    """Metadata for cached artifacts."""
    
    artifact_type: str
    schema_version: str
    data_hash: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    dependencies: List[str] = None
    tags: List[str] = None
    size_bytes: int = 0
    
    def __post_init__(self):
        """Validate artifact metadata."""
        assert self.artifact_type, "artifact_type is required"
        assert self.schema_version, "schema_version is required"
        assert self.data_hash, "data_hash is required"
        assert isinstance(self.created_at, datetime), "created_at must be datetime"
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []


class EnhancedCachingIntegration:
    """
    Enhanced caching integration with advanced features.
    
    Provides comprehensive caching with FeatureCacheService integration,
    artifact management, and performance optimization.
    """
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 enable_feature_cache: bool = True,
                 enable_serialization: bool = True,
                 enable_compression: bool = True,
                 max_cache_size_mb: int = 1000,
                 default_ttl_seconds: int = 3600):
        """Initialize the enhanced caching integration."""
        self.cache_dir = cache_dir or Path("artifacts") / "enhanced_cache"
        self.enable_feature_cache = enable_feature_cache and FEATURE_CACHE_AVAILABLE
        self.enable_serialization = enable_serialization and SERIALIZATION_AVAILABLE
        self.enable_compression = enable_compression
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl_seconds = default_ttl_seconds
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache components
        self._initialize_cache_components()
        
        # Cache storage
        self.cache_entries: Dict[str, CacheEntry] = {}
        self.artifact_metadata: Dict[str, ArtifactMetadata] = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_cache_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'total_cache_time': 0.0,
            'total_serialization_time': 0.0,
            'total_deserialization_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        tprint_info("Enhanced Caching Integration initialized")
        if self.enable_feature_cache:
            tprint_info("✅ FeatureCacheService integration enabled")
        if self.enable_serialization:
            tprint_info("✅ Serialization enabled")
        if self.enable_compression:
            tprint_info("✅ Compression enabled")
    
    def _initialize_cache_components(self):
        """Initialize cache components."""
        # Initialize FeatureCacheService
        if self.enable_feature_cache:
            try:
                self.feature_cache = FeatureCacheService(
                    base_dir=self.cache_dir / "features"
                )
                tprint_success("✅ FeatureCacheService initialized")
            except Exception as e:
                tprint_warning(f"⚠️ FeatureCacheService initialization failed: {e}")
                self.feature_cache = None
                self.enable_feature_cache = False
        else:
            self.feature_cache = None
        
        # Initialize serialization utilities
        if self.enable_serialization:
            try:
                self.universal_serializer = UniversalSerializer()
                self.json_serializer = JSONSerializer()
                self.pickle_serializer = PickleSerializer()
                tprint_success("✅ Serialization utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Serialization utilities initialization failed: {e}")
                self.universal_serializer = None
                self.json_serializer = None
                self.pickle_serializer = None
                self.enable_serialization = False
        else:
            self.universal_serializer = None
            self.json_serializer = None
            self.pickle_serializer = None
    
    def cache_data(self, 
                   key: str, 
                   data: Any, 
                   artifact_type: str = "data",
                   schema_version: str = "1.0",
                   ttl_seconds: Optional[int] = None,
                   dependencies: Optional[List[str]] = None,
                   tags: Optional[List[str]] = None) -> bool:
        """
        Cache data with metadata.
        
        Args:
            key: Cache key
            data: Data to cache
            artifact_type: Type of artifact
            schema_version: Schema version
            ttl_seconds: Time to live in seconds
            dependencies: List of dependency keys
            tags: List of tags
            
        Returns:
            True if caching successful
        """
        tprint_debug(f"Caching data with key '{key}'")
        
        start_time = time.time()
        
        try:
            # Calculate data hash
            data_hash = self._calculate_data_hash(data)
            
            # Create artifact metadata
            metadata = ArtifactMetadata(
                artifact_type=artifact_type,
                schema_version=schema_version,
                data_hash=data_hash,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl_seconds or self.default_ttl_seconds),
                dependencies=dependencies or [],
                tags=tags or [],
                size_bytes=self._calculate_data_size(data)
            )
            
            # Serialize data
            serialized_data = self._serialize_data(data)
            
            # Create cache entry
            cache_entry = CacheEntry(
                key=key,
                data=serialized_data,
                metadata=metadata.__dict__,
                created_at=metadata.created_at,
                last_accessed=metadata.created_at,
                access_count=0,
                size_bytes=metadata.size_bytes,
                ttl_seconds=ttl_seconds or self.default_ttl_seconds
            )
            
            # Store in cache
            self.cache_entries[key] = cache_entry
            self.artifact_metadata[key] = metadata
            
            # Check cache size and evict if necessary
            self._evict_if_necessary()
            
            # Update performance stats
            cache_time = time.time() - start_time
            self.performance_stats['total_cache_operations'] += 1
            self.performance_stats['total_cache_time'] += cache_time
            
            tprint_success(f"Data cached successfully with key '{key}' in {cache_time:.3f}s")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to cache data with key '{key}': {e}")
            return False
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Get cached data by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        tprint_debug(f"Retrieving cached data with key '{key}'")
        
        start_time = time.time()
        
        try:
            # Check if key exists
            if key not in self.cache_entries:
                self.performance_stats['cache_misses'] += 1
                tprint_debug(f"Cache miss for key '{key}'")
                return None
            
            cache_entry = self.cache_entries[key]
            
            # Check if expired
            if self._is_expired(cache_entry):
                self._evict_entry(key)
                self.performance_stats['cache_misses'] += 1
                tprint_debug(f"Cache entry expired for key '{key}'")
                return None
            
            # Update access statistics
            cache_entry.last_accessed = datetime.now()
            cache_entry.access_count += 1
            
            # Deserialize data
            data = self._deserialize_data(cache_entry.data)
            
            # Update performance stats
            cache_time = time.time() - start_time
            self.performance_stats['cache_hits'] += 1
            self.performance_stats['total_cache_time'] += cache_time
            
            tprint_success(f"Cache hit for key '{key}' in {cache_time:.3f}s")
            return data
            
        except Exception as e:
            tprint_error(f"Failed to retrieve cached data with key '{key}': {e}")
            self.performance_stats['cache_misses'] += 1
            return None
    
    def cache_features(self, 
                      symbol: str,
                      timeframe: str,
                      features: pd.DataFrame,
                      feature_bank_version: str = "1.0",
                      lookback_config_hash: str = "default") -> bool:
        """
        Cache features using FeatureCacheService.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            features: Features DataFrame
            feature_bank_version: Feature bank version
            lookback_config_hash: Lookback configuration hash
            
        Returns:
            True if caching successful
        """
        if not self.enable_feature_cache or not self.feature_cache:
            tprint_warning("FeatureCacheService not available, using fallback caching")
            return self.cache_data(
                key=f"features_{symbol}_{timeframe}",
                data=features,
                artifact_type="features",
                schema_version=feature_bank_version
            )
        
        try:
            # Build cache key
            cache_key = self.feature_cache.build_key(
                symbol=symbol,
                timeframe=timeframe,
                feature_bank_version=feature_bank_version,
                lookback_config_hash=lookback_config_hash
            )
            
            # Save features
            self.feature_cache.save(cache_key, features, "features")
            
            tprint_success(f"Features cached for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            tprint_error(f"Failed to cache features for {symbol} {timeframe}: {e}")
            return False
    
    def get_cached_features(self, 
                           symbol: str,
                           timeframe: str,
                           feature_bank_version: str = "1.0",
                           lookback_config_hash: str = "default") -> Optional[pd.DataFrame]:
        """
        Get cached features using FeatureCacheService.
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            feature_bank_version: Feature bank version
            lookback_config_hash: Lookback configuration hash
            
        Returns:
            Cached features DataFrame or None if not found
        """
        if not self.enable_feature_cache or not self.feature_cache:
            tprint_warning("FeatureCacheService not available, using fallback caching")
            return self.get_cached_data(f"features_{symbol}_{timeframe}")
        
        try:
            # Build cache key
            cache_key = self.feature_cache.build_key(
                symbol=symbol,
                timeframe=timeframe,
                feature_bank_version=feature_bank_version,
                lookback_config_hash=lookback_config_hash
            )
            
            # Load features
            features = self.feature_cache.load(cache_key, "features")
            
            if features is not None:
                tprint_success(f"Features retrieved from cache for {symbol} {timeframe}")
            else:
                tprint_debug(f"No cached features found for {symbol} {timeframe}")
            
            return features
            
        except Exception as e:
            tprint_error(f"Failed to retrieve cached features for {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash for data."""
        try:
            if isinstance(data, pd.DataFrame):
                # Use DataFrame hash
                return hashlib.sha256(data.to_string().encode()).hexdigest()
            elif isinstance(data, (list, dict, tuple)):
                # Use JSON serialization
                json_str = json.dumps(data, sort_keys=True, default=str)
                return hashlib.sha256(json_str.encode()).hexdigest()
            else:
                # Use string representation
                return hashlib.sha256(str(data).encode()).hexdigest()
        except Exception as e:
            tprint_debug(f"Hash calculation failed: {e}")
            return hashlib.sha256(str(id(data)).encode()).hexdigest()
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (list, dict, tuple)):
                return len(json.dumps(data, default=str).encode())
            else:
                return len(str(data).encode())
        except Exception as e:
            tprint_debug(f"Size calculation failed: {e}")
            return 0
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for caching."""
        if not self.enable_serialization:
            return data
        
        try:
            if isinstance(data, pd.DataFrame):
                # Use pickle for DataFrames
                return pickle.dumps(data)
            elif isinstance(data, (list, dict, tuple)):
                # Use JSON for simple data structures
                return json.dumps(data, default=str)
            else:
                # Use pickle for complex objects
                return pickle.dumps(data)
        except Exception as e:
            tprint_debug(f"Serialization failed: {e}")
            return data
    
    def _deserialize_data(self, serialized_data: Any) -> Any:
        """Deserialize data from cache."""
        if not self.enable_serialization:
            return serialized_data
        
        try:
            if isinstance(serialized_data, bytes):
                # Try pickle first
                try:
                    return pickle.loads(serialized_data)
                except:
                    # Try JSON
                    return json.loads(serialized_data.decode())
            elif isinstance(serialized_data, str):
                # Try JSON
                return json.loads(serialized_data)
            else:
                return serialized_data
        except Exception as e:
            tprint_debug(f"Deserialization failed: {e}")
            return serialized_data
    
    def _is_expired(self, cache_entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if cache_entry.ttl_seconds is None:
            return False
        
        now = datetime.now()
        return now > cache_entry.created_at + timedelta(seconds=cache_entry.ttl_seconds)
    
    def _evict_if_necessary(self):
        """Evict entries if cache size exceeds limit."""
        current_size_mb = sum(entry.size_bytes for entry in self.cache_entries.values()) / (1024 * 1024)
        
        if current_size_mb > self.max_cache_size_mb:
            # Evict least recently used entries
            sorted_entries = sorted(
                self.cache_entries.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Evict 20% of entries
            evict_count = max(1, len(sorted_entries) // 5)
            
            for i in range(evict_count):
                key, _ = sorted_entries[i]
                self._evict_entry(key)
                self.performance_stats['cache_evictions'] += 1
    
    def _evict_entry(self, key: str):
        """Evict a cache entry."""
        if key in self.cache_entries:
            del self.cache_entries[key]
        if key in self.artifact_metadata:
            del self.artifact_metadata[key]
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_entries = len(self.cache_entries)
        total_size_bytes = sum(entry.size_bytes for entry in self.cache_entries.values())
        hit_count = self.performance_stats['cache_hits']
        miss_count = self.performance_stats['cache_misses']
        eviction_count = self.performance_stats['cache_evictions']
        
        total_requests = hit_count + miss_count
        hit_rate = hit_count / total_requests if total_requests > 0 else 0.0
        
        average_access_time = (
            self.performance_stats['total_cache_time'] / 
            self.performance_stats['total_cache_operations']
            if self.performance_stats['total_cache_operations'] > 0 else 0.0
        )
        
        memory_usage_mb = total_size_bytes / (1024 * 1024)
        
        return CacheStats(
            total_entries=total_entries,
            total_size_bytes=total_size_bytes,
            hit_count=hit_count,
            miss_count=miss_count,
            eviction_count=eviction_count,
            hit_rate=hit_rate,
            average_access_time=average_access_time,
            memory_usage_mb=memory_usage_mb
        )
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries."""
        if pattern is None:
            # Clear all entries
            self.cache_entries.clear()
            self.artifact_metadata.clear()
            tprint_info("All cache entries cleared")
        else:
            # Clear entries matching pattern
            keys_to_remove = [key for key in self.cache_entries.keys() if pattern in key]
            for key in keys_to_remove:
                self._evict_entry(key)
            tprint_info(f"Cleared {len(keys_to_remove)} cache entries matching pattern '{pattern}'")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_stats.copy()


# Convenience functions
def create_enhanced_caching_integration(
    cache_dir: Optional[Path] = None,
    enable_feature_cache: bool = True,
    enable_serialization: bool = True,
    enable_compression: bool = True,
    max_cache_size_mb: int = 1000,
    default_ttl_seconds: int = 3600
) -> EnhancedCachingIntegration:
    """Create an enhanced caching integration."""
    return EnhancedCachingIntegration(
        cache_dir=cache_dir,
        enable_feature_cache=enable_feature_cache,
        enable_serialization=enable_serialization,
        enable_compression=enable_compression,
        max_cache_size_mb=max_cache_size_mb,
        default_ttl_seconds=default_ttl_seconds
    )
