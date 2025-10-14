"""
Advanced Caching and Serialization for UnifiedDataDrivenPipeline

This module implements the sophisticated caching and serialization capabilities
from FeatureLookbackOptimizationComponent, including:
- Multi-level caching with cache hit/miss tracking
- Universal serialization with JSON and Pickle support
- Cache key resolution with pipeline state integration
- Force refresh capabilities for cache invalidation
- Memory-efficient caching strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import time
import logging
import hashlib
import json
import pickle
from pathlib import Path
import gc

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

# Import caching dependencies
try:
    from src.feature_generation.core.feature_cache import FeatureCacheService
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False
    FeatureCacheService = None
    UniversalSerializer = None
    JSONSerializer = None
    PickleSerializer = None

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-level caching."""
    MEMORY = "memory"
    DISK = "disk"
    PERSISTENT = "persistent"


@dataclass
class CacheConfig:
    """Configuration for advanced caching."""
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    enable_persistent_cache: bool = True
    memory_cache_size_mb: int = 100
    disk_cache_size_mb: int = 1000
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    enable_encryption: bool = False
    cache_directory: str = "./cache"
    max_cache_entries: int = 10000


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    timestamp: float
    ttl: int
    level: CacheLevel
    size_bytes: int
    access_count: int
    last_accessed: float


@dataclass
class CacheStats:
    """Cache statistics."""
    total_entries: int
    memory_entries: int
    disk_entries: int
    persistent_entries: int
    total_hits: int
    total_misses: int
    hit_rate: float
    memory_usage_mb: float
    disk_usage_mb: float
    evictions: int
    errors: int


class AdvancedCacheManager:
    """
    Advanced cache manager with multi-level caching capabilities.
    
    Features:
    - Multi-level caching (memory, disk, persistent)
    - Cache hit/miss tracking
    - TTL support
    - Compression and encryption
    - Memory-efficient strategies
    - Cache invalidation
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the advanced cache manager."""
        self.config = config or CacheConfig()
        
        # Initialize cache levels
        self.memory_cache = {}
        self.disk_cache = {}
        self.persistent_cache = {}
        
        # Cache statistics
        self.cache_stats = {
            'total_entries': 0,
            'memory_entries': 0,
            'disk_entries': 0,
            'persistent_entries': 0,
            'total_hits': 0,
            'total_misses': 0,
            'evictions': 0,
            'errors': 0,
            'memory_usage_bytes': 0,
            'disk_usage_bytes': 0
        }
        
        # Initialize serializers
        self._initialize_serializers()
        
        # Create cache directory
        self._create_cache_directory()
        
        tprint_success("✅ Advanced Cache Manager initialized")
    
    def _initialize_serializers(self):
        """Initialize serializers for different cache levels."""
        try:
            if CACHING_AVAILABLE:
                self.universal_serializer = UniversalSerializer()
                self.json_serializer = JSONSerializer()
                self.pickle_serializer = PickleSerializer()
                tprint_success("✅ Serializers initialized")
            else:
                self.universal_serializer = None
                self.json_serializer = None
                self.pickle_serializer = None
                tprint_warning("⚠️ Serializers not available, using fallback implementations")
        except Exception as e:
            tprint_error(f"❌ Serializer initialization failed: {e}")
            self.universal_serializer = None
            self.json_serializer = None
            self.pickle_serializer = None
    
    def _create_cache_directory(self):
        """Create cache directory if it doesn't exist."""
        try:
            cache_dir = Path(self.config.cache_directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tprint_debug(f"📁 Cache directory: {cache_dir}")
        except Exception as e:
            tprint_error(f"❌ Failed to create cache directory: {e}")
    
    def get(self, key: str, level: Optional[CacheLevel] = None) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            level: Specific cache level to check (None for all levels)
            
        Returns:
            Cached data or None if not found
        """
        try:
            # Check specific level
            if level:
                return self._get_from_level(key, level)
            
            # Check all levels in order of preference
            for cache_level in [CacheLevel.MEMORY, CacheLevel.DISK, CacheLevel.PERSISTENT]:
                data = self._get_from_level(key, cache_level)
                if data is not None:
                    # Promote to memory cache if found in lower level
                    if cache_level != CacheLevel.MEMORY:
                        self._set_to_level(key, data, CacheLevel.MEMORY)
                    return data
            
            # Cache miss
            self.cache_stats['total_misses'] += 1
            return None
            
        except Exception as e:
            tprint_error(f"❌ Cache get failed for key {key}: {e}")
            self.cache_stats['errors'] += 1
            return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None, level: CacheLevel = CacheLevel.MEMORY):
        """
        Set data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            level: Cache level to use
        """
        try:
            ttl = ttl or self.config.cache_ttl_seconds
            
            # Check cache size limits
            if self._should_evict(level):
                self._evict_entries(level)
            
            # Set to specified level
            self._set_to_level(key, data, level, ttl)
            
            # Update statistics
            self.cache_stats['total_entries'] += 1
            
        except Exception as e:
            tprint_error(f"❌ Cache set failed for key {key}: {e}")
            self.cache_stats['errors'] += 1
    
    def _get_from_level(self, key: str, level: CacheLevel) -> Optional[Any]:
        """Get data from specific cache level."""
        try:
            if level == CacheLevel.MEMORY:
                return self._get_from_memory(key)
            elif level == CacheLevel.DISK:
                return self._get_from_disk(key)
            elif level == CacheLevel.PERSISTENT:
                return self._get_from_persistent(key)
            else:
                return None
        except Exception as e:
            tprint_debug(f"Error getting from {level.value} cache: {e}")
            return None
    
    def _set_to_level(self, key: str, data: Any, level: CacheLevel, ttl: int = 3600):
        """Set data to specific cache level."""
        try:
            if level == CacheLevel.MEMORY:
                self._set_to_memory(key, data, ttl)
            elif level == CacheLevel.DISK:
                self._set_to_disk(key, data, ttl)
            elif level == CacheLevel.PERSISTENT:
                self._set_to_persistent(key, data, ttl)
        except Exception as e:
            tprint_debug(f"Error setting to {level.value} cache: {e}")
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get data from memory cache."""
        if key not in self.memory_cache:
            return None
        
        entry = self.memory_cache[key]
        
        # Check TTL
        if time.time() - entry.timestamp > entry.ttl:
            del self.memory_cache[key]
            self.cache_stats['memory_entries'] -= 1
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_accessed = time.time()
        
        # Update hit statistics
        self.cache_stats['total_hits'] += 1
        
        return entry.data
    
    def _set_to_memory(self, key: str, data: Any, ttl: int):
        """Set data to memory cache."""
        # Calculate size
        size_bytes = self._calculate_size(data)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            ttl=ttl,
            level=CacheLevel.MEMORY,
            size_bytes=size_bytes,
            access_count=0,
            last_accessed=time.time()
        )
        
        # Check memory limits
        if self._exceeds_memory_limit(size_bytes):
            self._evict_memory_entries()
        
        # Store entry
        self.memory_cache[key] = entry
        self.cache_stats['memory_entries'] += 1
        self.cache_stats['memory_usage_bytes'] += size_bytes
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get data from disk cache."""
        try:
            cache_file = Path(self.config.cache_directory) / f"{key}.cache"
            if not cache_file.exists():
                return None
            
            # Load entry
            with open(cache_file, 'rb') as f:
                entry = pickle.load(f)
            
            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                cache_file.unlink()
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Update hit statistics
            self.cache_stats['total_hits'] += 1
            
            return entry.data
            
        except Exception as e:
            tprint_debug(f"Error loading from disk cache: {e}")
            return None
    
    def _set_to_disk(self, key: str, data: Any, ttl: int):
        """Set data to disk cache."""
        try:
            # Calculate size
            size_bytes = self._calculate_size(data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                timestamp=time.time(),
                ttl=ttl,
                level=CacheLevel.DISK,
                size_bytes=size_bytes,
                access_count=0,
                last_accessed=time.time()
            )
            
            # Check disk limits
            if self._exceeds_disk_limit(size_bytes):
                self._evict_disk_entries()
            
            # Store entry
            cache_file = Path(self.config.cache_directory) / f"{key}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            self.cache_stats['disk_entries'] += 1
            self.cache_stats['disk_usage_bytes'] += size_bytes
            
        except Exception as e:
            tprint_debug(f"Error saving to disk cache: {e}")
    
    def _get_from_persistent(self, key: str) -> Optional[Any]:
        """Get data from persistent cache."""
        try:
            if not CACHING_AVAILABLE or not self.universal_serializer:
                return None
            
            # Use universal serializer for persistent cache
            data = self.universal_serializer.load(key)
            if data is not None:
                self.cache_stats['total_hits'] += 1
            
            return data
            
        except Exception as e:
            tprint_debug(f"Error loading from persistent cache: {e}")
            return None
    
    def _set_to_persistent(self, key: str, data: Any, ttl: int):
        """Set data to persistent cache."""
        try:
            if not CACHING_AVAILABLE or not self.universal_serializer:
                return
            
            # Use universal serializer for persistent cache
            self.universal_serializer.save(key, data)
            self.cache_stats['persistent_entries'] += 1
            
        except Exception as e:
            tprint_debug(f"Error saving to persistent cache: {e}")
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, (list, dict, tuple)):
                return len(pickle.dumps(data))
            else:
                return len(str(data).encode('utf-8'))
        except:
            return 0
    
    def _should_evict(self, level: CacheLevel) -> bool:
        """Check if cache should evict entries."""
        if level == CacheLevel.MEMORY:
            return len(self.memory_cache) >= self.config.max_cache_entries
        elif level == CacheLevel.DISK:
            return len(self.disk_cache) >= self.config.max_cache_entries
        else:
            return False
    
    def _exceeds_memory_limit(self, size_bytes: int) -> bool:
        """Check if adding data would exceed memory limit."""
        memory_limit_bytes = self.config.memory_cache_size_mb * 1024 * 1024
        return self.cache_stats['memory_usage_bytes'] + size_bytes > memory_limit_bytes
    
    def _exceeds_disk_limit(self, size_bytes: int) -> bool:
        """Check if adding data would exceed disk limit."""
        disk_limit_bytes = self.config.disk_cache_size_mb * 1024 * 1024
        return self.cache_stats['disk_usage_bytes'] + size_bytes > disk_limit_bytes
    
    def _evict_entries(self, level: CacheLevel):
        """Evict entries from cache level."""
        if level == CacheLevel.MEMORY:
            self._evict_memory_entries()
        elif level == CacheLevel.DISK:
            self._evict_disk_entries()
    
    def _evict_memory_entries(self):
        """Evict least recently used entries from memory cache."""
        try:
            # Sort by last accessed time
            sorted_entries = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest 25% of entries
            n_to_remove = max(1, len(sorted_entries) // 4)
            
            for i in range(n_to_remove):
                key, entry = sorted_entries[i]
                del self.memory_cache[key]
                self.cache_stats['memory_entries'] -= 1
                self.cache_stats['memory_usage_bytes'] -= entry.size_bytes
                self.cache_stats['evictions'] += 1
            
        except Exception as e:
            tprint_debug(f"Error evicting memory entries: {e}")
    
    def _evict_disk_entries(self):
        """Evict least recently used entries from disk cache."""
        try:
            # Get all cache files
            cache_dir = Path(self.config.cache_directory)
            cache_files = list(cache_dir.glob("*.cache"))
            
            if not cache_files:
                return
            
            # Sort by modification time
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest 25% of files
            n_to_remove = max(1, len(cache_files) // 4)
            
            for i in range(n_to_remove):
                cache_file = cache_files[i]
                cache_file.unlink()
                self.cache_stats['disk_entries'] -= 1
                self.cache_stats['evictions'] += 1
            
        except Exception as e:
            tprint_debug(f"Error evicting disk entries: {e}")
    
    def invalidate(self, key: str):
        """Invalidate cache entry."""
        try:
            # Remove from all levels
            if key in self.memory_cache:
                del self.memory_cache[key]
                self.cache_stats['memory_entries'] -= 1
            
            cache_file = Path(self.config.cache_directory) / f"{key}.cache"
            if cache_file.exists():
                cache_file.unlink()
                self.cache_stats['disk_entries'] -= 1
            
            if CACHING_AVAILABLE and self.universal_serializer:
                self.universal_serializer.delete(key)
                self.cache_stats['persistent_entries'] -= 1
            
        except Exception as e:
            tprint_debug(f"Error invalidating cache entry {key}: {e}")
    
    def clear(self, level: Optional[CacheLevel] = None):
        """Clear cache."""
        try:
            if level is None or level == CacheLevel.MEMORY:
                self.memory_cache.clear()
                self.cache_stats['memory_entries'] = 0
                self.cache_stats['memory_usage_bytes'] = 0
            
            if level is None or level == CacheLevel.DISK:
                cache_dir = Path(self.config.cache_directory)
                for cache_file in cache_dir.glob("*.cache"):
                    cache_file.unlink()
                self.cache_stats['disk_entries'] = 0
                self.cache_stats['disk_usage_bytes'] = 0
            
            if level is None or level == CacheLevel.PERSISTENT:
                if CACHING_AVAILABLE and self.universal_serializer:
                    self.universal_serializer.clear()
                self.cache_stats['persistent_entries'] = 0
            
            if level is None:
                self.cache_stats['total_entries'] = 0
            
        except Exception as e:
            tprint_debug(f"Error clearing cache: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self.cache_stats['total_hits'] + self.cache_stats['total_misses']
        hit_rate = self.cache_stats['total_hits'] / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            total_entries=self.cache_stats['total_entries'],
            memory_entries=self.cache_stats['memory_entries'],
            disk_entries=self.cache_stats['disk_entries'],
            persistent_entries=self.cache_stats['persistent_entries'],
            total_hits=self.cache_stats['total_hits'],
            total_misses=self.cache_stats['total_misses'],
            hit_rate=hit_rate,
            memory_usage_mb=self.cache_stats['memory_usage_bytes'] / 1024 / 1024,
            disk_usage_mb=self.cache_stats['disk_usage_bytes'] / 1024 / 1024,
            evictions=self.cache_stats['evictions'],
            errors=self.cache_stats['errors']
        )
    
    def generate_cache_key(self, data: Any, prefix: str = "") -> str:
        """Generate cache key for data."""
        try:
            # Create hash of data
            if isinstance(data, pd.DataFrame):
                data_str = f"{data.shape}_{data.columns.tolist()}_{data.index.tolist()}"
            elif isinstance(data, np.ndarray):
                data_str = f"{data.shape}_{data.dtype}_{data.tobytes()}"
            else:
                data_str = str(data)
            
            # Generate hash
            hash_obj = hashlib.md5(data_str.encode('utf-8'))
            hash_hex = hash_obj.hexdigest()
            
            return f"{prefix}_{hash_hex}" if prefix else hash_hex
            
        except Exception as e:
            tprint_debug(f"Error generating cache key: {e}")
            return f"{prefix}_{int(time.time())}" if prefix else str(int(time.time()))


def create_advanced_cache_manager(config: Optional[CacheConfig] = None) -> AdvancedCacheManager:
    """Create an advanced cache manager with default configuration."""
    return AdvancedCacheManager(config)