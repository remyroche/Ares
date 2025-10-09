"""
Intelligent Caching System for Feature Lookback Optimization

Implements hierarchical caching with dependency tracking, warm start,
and compression for optimal performance.
"""

import os
import json
import pickle
import hashlib
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import zstandard as zstd
import numpy as np
import pandas as pd

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
from src.utils.logger import get_logger


@dataclass
class CacheKey:
    """Represents a cache key with all dependencies."""
    dataset_version: str
    symbol: str
    timeframe: str
    feature_signature: str
    label_spec: str
    search_space: str
    seed: int
    code_hash: str
    
    def to_string(self) -> str:
        """Convert to string representation."""
        return f"{self.dataset_version}_{self.symbol}_{self.timeframe}_{self.feature_signature}_{self.label_spec}_{self.search_space}_{self.seed}_{self.code_hash}"
    
    def to_hash(self) -> str:
        """Convert to hash for shorter keys."""
        key_str = self.to_string()
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: CacheKey
    data: Any
    timestamp: float
    size_bytes: int
    access_count: int
    last_accessed: float
    dependencies: Set[str]
    compression_ratio: float = 1.0


@dataclass
class WarmStartData:
    """Data for warm starting optimization."""
    top_k_lookbacks: List[int]
    local_curvature: Dict[str, float]  # MI around argmax
    convergence_info: Dict[str, Any]
    feature_importance: Dict[str, float]


class IntelligentCache:
    """
    Intelligent caching system with dependency tracking and warm start.
    
    Features:
    - Hierarchical caching (memory -> disk -> compressed)
    - Dependency tracking for smart invalidation
    - Warm start data for Bayesian optimization
    - Compression with zstd
    - Memory-mapped arrays for large data
    """
    
    def __init__(self, 
                 cache_dir: str = "feature_lookback_cache",
                 max_memory_mb: int = 1024,
                 max_disk_mb: int = 10240,
                 enable_compression: bool = True,
                 logger=None):
        """Initialize the intelligent cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_mb = max_memory_mb
        self.max_disk_mb = max_disk_mb
        self.enable_compression = enable_compression
        self.logger = logger or get_logger('IntelligentCache')
        
        # Cache layers
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = self.cache_dir / "disk"
        self.compressed_cache_dir = self.cache_dir / "compressed"
        self.disk_cache_dir.mkdir(exist_ok=True)
        self.compressed_cache_dir.mkdir(exist_ok=True)
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        
        # Warm start data
        self.warm_start_cache: Dict[str, WarmStartData] = {}
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'invalidations': 0,
            'compression_saves': 0,
            'memory_usage_mb': 0,
            'disk_usage_mb': 0
        }
        
        # Compression context
        if self.enable_compression:
            self.zstd_compressor = zstd.ZstdCompressor(level=3)
            self.zstd_decompressor = zstd.ZstdDecompressor()
        
        tprint("🧠 Initializing Intelligent Cache")
        tprint_info(f"   → Cache directory: {self.cache_dir}")
        tprint_info(f"   → Max memory: {max_memory_mb}MB")
        tprint_info(f"   → Max disk: {max_disk_mb}MB")
        tprint_info(f"   → Compression: {enable_compression}")
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        tprint_success("✅ Intelligent Cache initialized")
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get data from cache with intelligent lookup."""
        cache_key = key.to_hash()
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.stats['hits'] += 1
            tprint_debug(f"💾 Memory cache hit for {key.feature_signature}")
            return entry.data
        
        # Try disk cache
        disk_path = self.disk_cache_dir / f"{cache_key}.pkl"
        if disk_path.exists():
            try:
                data = self._load_from_disk(disk_path)
                if data is not None:
                    # Promote to memory cache
                    self._add_to_memory_cache(cache_key, key, data)
                    self.stats['hits'] += 1
                    tprint_debug(f"💾 Disk cache hit for {key.feature_signature}")
                    return data
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load from disk cache: {e}")
        
        # Try compressed cache
        compressed_path = self.compressed_cache_dir / f"{cache_key}.zst"
        if compressed_path.exists():
            try:
                data = self._load_from_compressed(compressed_path)
                if data is not None:
                    # Promote to memory cache
                    self._add_to_memory_cache(cache_key, key, data)
                    self.stats['hits'] += 1
                    tprint_debug(f"💾 Compressed cache hit for {key.feature_signature}")
                    return data
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load from compressed cache: {e}")
        
        self.stats['misses'] += 1
        tprint_debug(f"💾 Cache miss for {key.feature_signature}")
        return None
    
    def put(self, key: CacheKey, data: Any, dependencies: Optional[Set[str]] = None) -> bool:
        """Store data in cache with dependency tracking."""
        cache_key = key.to_hash()
        
        try:
            # Calculate data size
            size_bytes = self._calculate_size(data)
            
            # Add to memory cache
            self._add_to_memory_cache(cache_key, key, data, size_bytes, dependencies)
            
            # Store on disk for persistence
            self._store_to_disk(cache_key, data)
            
            # Store compressed version for large data
            if size_bytes > 1024 * 1024:  # > 1MB
                self._store_compressed(cache_key, data)
                self.stats['compression_saves'] += 1
            
            # Update dependency tracking
            if dependencies:
                self._update_dependencies(cache_key, dependencies)
            
            self.stats['writes'] += 1
            tprint_debug(f"💾 Cached {key.feature_signature} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Failed to cache {key.feature_signature}: {e}")
            return False
    
    def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate all cache entries that depend on the given dependency."""
        invalidated = 0
        
        if dependency in self.reverse_dependencies:
            for cache_key in self.reverse_dependencies[dependency]:
                if self._invalidate_entry(cache_key):
                    invalidated += 1
        
        self.stats['invalidations'] += invalidated
        tprint_info(f"🗑️ Invalidated {invalidated} entries for dependency: {dependency}")
        return invalidated
    
    def get_warm_start_data(self, feature_signature: str) -> Optional[WarmStartData]:
        """Get warm start data for Bayesian optimization."""
        return self.warm_start_cache.get(feature_signature)
    
    def put_warm_start_data(self, feature_signature: str, warm_start: WarmStartData):
        """Store warm start data for future use."""
        self.warm_start_cache[feature_signature] = warm_start
        tprint_debug(f"🔥 Stored warm start data for {feature_signature}")
    
    def _add_to_memory_cache(self, cache_key: str, key: CacheKey, data: Any, 
                           size_bytes: int = None, dependencies: Set[str] = None):
        """Add entry to memory cache with size management."""
        if size_bytes is None:
            size_bytes = self._calculate_size(data)
        
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            size_bytes=size_bytes,
            access_count=1,
            last_accessed=time.time(),
            dependencies=dependencies or set()
        )
        
        # Check memory limits
        if self._get_memory_usage() + size_bytes > self.max_memory_mb * 1024 * 1024:
            self._evict_memory_cache()
        
        self.memory_cache[cache_key] = entry
        self.stats['memory_usage_mb'] = self._get_memory_usage() / (1024 * 1024)
    
    def _store_to_disk(self, cache_key: str, data: Any):
        """Store data to disk cache."""
        disk_path = self.disk_cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(disk_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to store to disk: {e}")
    
    def _store_compressed(self, cache_key: str, data: Any):
        """Store data in compressed format."""
        if not self.enable_compression:
            return
        
        compressed_path = self.compressed_cache_dir / f"{cache_key}.zst"
        
        try:
            # Serialize data
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress
            compressed = self.zstd_compressor.compress(serialized)
            
            # Store
            with open(compressed_path, 'wb') as f:
                f.write(compressed)
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to store compressed: {e}")
    
    def _load_from_disk(self, disk_path: Path) -> Optional[Any]:
        """Load data from disk cache."""
        try:
            with open(disk_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load from disk: {e}")
            return None
    
    def _load_from_compressed(self, compressed_path: Path) -> Optional[Any]:
        """Load data from compressed cache."""
        if not self.enable_compression:
            return None
        
        try:
            with open(compressed_path, 'rb') as f:
                compressed = f.read()
            
            # Decompress
            serialized = self.zstd_decompressor.decompress(compressed)
            
            # Deserialize
            return pickle.loads(serialized)
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load compressed: {e}")
            return None
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            if isinstance(data, np.ndarray):
                return data.nbytes
            elif isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(self._calculate_size(v) for v in data.values())
            else:
                return len(pickle.dumps(data))
        except:
            return 1024  # Default estimate
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        total = 0
        for entry in self.memory_cache.values():
            total += entry.size_bytes
        return total
    
    def _evict_memory_cache(self):
        """Evict least recently used entries from memory cache."""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest 25% of entries
        to_remove = len(sorted_entries) // 4
        for i in range(to_remove):
            cache_key, _ = sorted_entries[i]
            del self.memory_cache[cache_key]
        
        tprint_debug(f"🧹 Evicted {to_remove} entries from memory cache")
    
    def _update_dependencies(self, cache_key: str, dependencies: Set[str]):
        """Update dependency tracking."""
        self.dependency_graph[cache_key] = dependencies
        
        for dep in dependencies:
            if dep not in self.reverse_dependencies:
                self.reverse_dependencies[dep] = set()
            self.reverse_dependencies[dep].add(cache_key)
    
    def _invalidate_entry(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry."""
        invalidated = False
        
        # Remove from memory
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            invalidated = True
        
        # Remove from disk
        disk_path = self.disk_cache_dir / f"{cache_key}.pkl"
        if disk_path.exists():
            disk_path.unlink()
            invalidated = True
        
        # Remove compressed version
        compressed_path = self.compressed_cache_dir / f"{cache_key}.zst"
        if compressed_path.exists():
            compressed_path.unlink()
            invalidated = True
        
        return invalidated
    
    def _load_cache_metadata(self):
        """Load existing cache metadata."""
        metadata_path = self.cache_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.stats.update(metadata.get('stats', {}))
            except Exception as e:
                tprint_warning(f"⚠️ Failed to load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """Save cache metadata."""
        metadata_path = self.cache_dir / "metadata.json"
        try:
            metadata = {
                'stats': self.stats,
                'timestamp': time.time()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save cache metadata: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(list(self.disk_cache_dir.glob("*.pkl"))),
            'compressed_entries': len(list(self.compressed_cache_dir.glob("*.zst"))),
            'hit_rate': self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        }
    
    def cleanup(self):
        """Cleanup cache resources."""
        self._save_cache_metadata()
        tprint("🧹 Cache cleanup completed")


# Utility functions for creating cache keys
def create_cache_key(dataset_version: str,
                    symbol: str,
                    timeframe: str,
                    feature_signature: str,
                    label_spec: str,
                    search_space: str,
                    seed: int,
                    code_hash: str) -> CacheKey:
    """Create a cache key with all dependencies."""
    return CacheKey(
        dataset_version=dataset_version,
        symbol=symbol,
        timeframe=timeframe,
        feature_signature=feature_signature,
        label_spec=label_spec,
        search_space=search_space,
        seed=seed,
        code_hash=code_hash
    )


def compute_code_hash(code_string: str) -> str:
    """Compute hash of code for dependency tracking."""
    return hashlib.sha256(code_string.encode()).hexdigest()[:16]


def compute_feature_signature(feature_name: str, feature_params: Dict[str, Any]) -> str:
    """Compute signature for feature with parameters."""
    signature = f"{feature_name}_{json.dumps(feature_params, sort_keys=True)}"
    return hashlib.sha256(signature.encode()).hexdigest()[:16]