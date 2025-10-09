"""
Content-Addressed Cache with Dependency Graph

This module implements a sophisticated caching system that:
- Uses content-addressed keys (blake3 hash of inputs)
- Maintains a dependency graph for intelligent invalidation
- Provides L1 (in-memory) and L2 (disk) cache layers
- Supports warm-start for optimization algorithms
- Implements smart cache eviction policies

Key Features:
- Content-addressed keys for cache consistency
- Dependency graph for targeted invalidation
- Multi-level caching (L1 + L2)
- Warm-start support for TPE optimization
- Intelligent eviction policies
"""

import hashlib
import pickle
import json
import time
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
try:
    import zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import networkx as nx

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for the content-addressed cache."""
    # Cache sizes
    l1_max_size_mb: float = 100.0
    l2_max_size_mb: float = 1000.0
    
    # Cache directories
    l2_cache_dir: str = "cache"
    warm_start_dir: str = "warm_start"
    
    # Compression settings
    enable_compression: bool = True
    compression_level: int = 3
    
    # TTL settings
    default_ttl_seconds: int = 3600  # 1 hour
    warm_start_ttl_seconds: int = 86400  # 24 hours
    
    # Eviction policies
    l1_eviction_policy: str = "lru"  # lru, lfu, ttl
    l2_eviction_policy: str = "lru"
    
    # Dependency tracking
    enable_dependency_tracking: bool = True
    max_dependency_depth: int = 10


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    data: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: int = 3600
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    cache_level: int = 1  # 1 for L1, 2 for L2
    compressed: bool = False


@dataclass
class WarmStartData:
    """Warm-start data for optimization algorithms."""
    best_parameters: Dict[str, Any]
    neighborhood_scores: List[Tuple[Dict[str, Any], float]]
    optimization_history: List[Dict[str, Any]]
    created_at: float
    algorithm_type: str
    search_space_hash: str


class ContentAddressedCache:
    """
    Content-addressed cache with dependency graph and multi-level storage.
    
    Uses blake3 hashing for content addressing and maintains a dependency
    graph for intelligent cache invalidation.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the content-addressed cache."""
        self.config = config or CacheConfig()
        
        # Cache storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache_dir = Path(self.config.l2_cache_dir)
        self.warm_start_dir = Path(self.config.warm_start_dir)
        
        # Create directories
        self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
        self.warm_start_dir.mkdir(parents=True, exist_ok=True)
        
        # Dependency graph
        self.dependency_graph = nx.DiGraph()
        self.reverse_dependency_graph = nx.DiGraph()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0,
            'compressions': 0,
            'decompressions': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        tprint_info(f"🚀 Content-addressed cache initialized")
        tprint_info(f"📊 L1 cache: {self.config.l1_max_size_mb} MB")
        tprint_info(f"📊 L2 cache: {self.config.l2_max_size_mb} MB")
        tprint_info(f"🗂️ L2 directory: {self.l2_cache_dir}")
    
    def _compute_content_hash(self, data_id: str, symbol: str, timeframe: str,
                            label_spec: Dict[str, Any], feature_signature: str,
                            code_hash: str, search_space: Dict[str, Any],
                            seed: int, gate_hash: str) -> str:
        """Compute content-addressed hash using blake3."""
        # Create hash input
        hash_input = {
            'data_id': data_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'label_spec': label_spec,
            'feature_signature': feature_signature,
            'code_hash': code_hash,
            'search_space': search_space,
            'seed': seed,
            'gate_hash': gate_hash
        }
        
        # Convert to JSON string for hashing
        hash_string = json.dumps(hash_input, sort_keys=True)
        
        # Compute blake3 hash
        return hashlib.blake2b(hash_string.encode(), digest_size=32).hexdigest()
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data with optional compression."""
        # Serialize to pickle
        serialized = pickle.dumps(data)
        
        if self.config.enable_compression and len(serialized) > 1024 and ZSTD_AVAILABLE:  # Only compress large data
            compressed = zstd.compress(serialized, self.config.compression_level)
            self.stats['compressions'] += 1
            return compressed
        else:
            return serialized
    
    def _deserialize_data(self, data: bytes, compressed: bool = False) -> Any:
        """Deserialize data with optional decompression."""
        if compressed and self.config.enable_compression and ZSTD_AVAILABLE:
            decompressed = zstd.decompress(data)
            self.stats['decompressions'] += 1
            return pickle.loads(decompressed)
        else:
            return pickle.loads(data)
    
    def _get_data_size(self, data: Any) -> int:
        """Get the size of data in bytes."""
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            return data.nbytes if hasattr(data, 'nbytes') else len(str(data))
        else:
            return len(str(data))
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if entry.ttl_seconds <= 0:
            return False
        
        age = time.time() - entry.created_at
        return age > entry.ttl_seconds
    
    def _evict_l1_cache(self) -> None:
        """Evict entries from L1 cache based on policy."""
        if not self.l1_cache:
            return
        
        # Calculate current size
        current_size = sum(entry.size_bytes for entry in self.l1_cache.values())
        target_size = self.config.l1_max_size_mb * 1024 * 1024
        
        if current_size <= target_size:
            return
        
        # Evict based on policy
        if self.config.l1_eviction_policy == "lru":
            # Remove least recently used entries
            while current_size > target_size and self.l1_cache:
                key, entry = self.l1_cache.popitem(last=False)
                current_size -= entry.size_bytes
                self.stats['evictions'] += 1
                tprint_debug(f"🗑️ Evicted L1 entry: {key}")
        
        elif self.config.l1_eviction_policy == "lfu":
            # Remove least frequently used entries
            sorted_entries = sorted(self.l1_cache.items(), key=lambda x: x[1].access_count)
            for key, entry in sorted_entries:
                if current_size <= target_size:
                    break
                del self.l1_cache[key]
                current_size -= entry.size_bytes
                self.stats['evictions'] += 1
                tprint_debug(f"🗑️ Evicted L1 entry: {key}")
        
        elif self.config.l1_eviction_policy == "ttl":
            # Remove expired entries
            expired_keys = [key for key, entry in self.l1_cache.items() if self._is_expired(entry)]
            for key in expired_keys:
                del self.l1_cache[key]
                self.stats['evictions'] += 1
                tprint_debug(f"🗑️ Evicted expired L1 entry: {key}")
    
    def _save_to_l2(self, key: str, entry: CacheEntry) -> None:
        """Save entry to L2 cache (disk)."""
        file_path = self.l2_cache_dir / f"{key}.cache"
        
        # Serialize entry
        entry_data = {
            'key': entry.key,
            'data': self._serialize_data(entry.data),
            'created_at': entry.created_at,
            'last_accessed': entry.last_accessed,
            'access_count': entry.access_count,
            'size_bytes': entry.size_bytes,
            'ttl_seconds': entry.ttl_seconds,
            'dependencies': list(entry.dependencies),
            'dependents': list(entry.dependents),
            'cache_level': entry.cache_level,
            'compressed': entry.compressed
        }
        
        # Save to disk
        with open(file_path, 'wb') as f:
            pickle.dump(entry_data, f)
        
        tprint_debug(f"💾 Saved to L2 cache: {key}")
    
    def _load_from_l2(self, key: str) -> Optional[CacheEntry]:
        """Load entry from L2 cache (disk)."""
        file_path = self.l2_cache_dir / f"{key}.cache"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                entry_data = pickle.load(f)
            
            # Deserialize data
            data = self._deserialize_data(entry_data['data'], entry_data.get('compressed', False))
            
            # Create entry
            entry = CacheEntry(
                key=entry_data['key'],
                data=data,
                created_at=entry_data['created_at'],
                last_accessed=entry_data['last_accessed'],
                access_count=entry_data['access_count'],
                size_bytes=entry_data['size_bytes'],
                ttl_seconds=entry_data['ttl_seconds'],
                dependencies=set(entry_data['dependencies']),
                dependents=set(entry_data['dependents']),
                cache_level=entry_data['cache_level'],
                compressed=entry_data.get('compressed', False)
            )
            
            tprint_debug(f"📂 Loaded from L2 cache: {key}")
            return entry
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load L2 cache entry {key}: {e}")
            return None
    
    def _remove_from_l2(self, key: str) -> None:
        """Remove entry from L2 cache."""
        file_path = self.l2_cache_dir / f"{key}.cache"
        if file_path.exists():
            file_path.unlink()
            tprint_debug(f"🗑️ Removed from L2 cache: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        with self.lock:
            # Try L1 cache first
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self.l1_cache[key]
                    self.stats['misses'] += 1
                    return None
                
                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Move to end (LRU)
                self.l1_cache.move_to_end(key)
                
                self.stats['hits'] += 1
                tprint_debug(f"✅ L1 cache hit: {key}")
                return entry.data
            
            # Try L2 cache
            entry = self._load_from_l2(key)
            if entry is not None:
                # Check if expired
                if self._is_expired(entry):
                    self._remove_from_l2(key)
                    self.stats['misses'] += 1
                    return None
                
                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Promote to L1 cache
                self._evict_l1_cache()
                self.l1_cache[key] = entry
                
                self.stats['hits'] += 1
                tprint_debug(f"✅ L2 cache hit: {key}")
                return entry.data
            
            self.stats['misses'] += 1
            tprint_debug(f"❌ Cache miss: {key}")
            return None
    
    def put(self, key: str, data: Any, dependencies: Optional[Set[str]] = None,
            ttl_seconds: Optional[int] = None) -> None:
        """Put data into cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._get_data_size(data)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                dependencies=dependencies or set(),
                cache_level=1
            )
            
            # Add to dependency graph
            if self.config.enable_dependency_tracking and dependencies:
                for dep in dependencies:
                    self.dependency_graph.add_edge(dep, key)
                    self.reverse_dependency_graph.add_edge(key, dep)
                    entry.dependencies.add(dep)
            
            # Evict L1 cache if needed
            self._evict_l1_cache()
            
            # Add to L1 cache
            self.l1_cache[key] = entry
            
            # Also save to L2 cache for persistence
            self._save_to_l2(key, entry)
            
            tprint_debug(f"💾 Cached: {key} ({size_bytes} bytes)")
    
    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry and its dependents."""
        with self.lock:
            # Remove from L1 cache
            if key in self.l1_cache:
                del self.l1_cache[key]
            
            # Remove from L2 cache
            self._remove_from_l2(key)
            
            # Invalidate dependents
            if self.config.enable_dependency_tracking:
                dependents = list(self.reverse_dependency_graph.successors(key))
                for dependent in dependents:
                    self.invalidate(dependent)
                    self.stats['invalidations'] += 1
            
            tprint_debug(f"🗑️ Invalidated: {key}")
    
    def invalidate_dependencies(self, dependencies: Set[str]) -> None:
        """Invalidate all entries that depend on the given dependencies."""
        with self.lock:
            if not self.config.enable_dependency_tracking:
                return
            
            # Find all entries that depend on any of the given dependencies
            to_invalidate = set()
            for dep in dependencies:
                # Find all entries that depend on this dependency
                dependents = nx.descendants(self.dependency_graph, dep)
                to_invalidate.update(dependents)
            
            # Invalidate all found entries
            for key in to_invalidate:
                self.invalidate(key)
                self.stats['invalidations'] += 1
            
            tprint_debug(f"🗑️ Invalidated {len(to_invalidate)} entries due to dependency changes")
    
    def save_warm_start(self, key: str, warm_start_data: WarmStartData) -> None:
        """Save warm-start data for optimization algorithms."""
        file_path = self.warm_start_dir / f"{key}.warm_start"
        
        # Serialize warm-start data
        data = {
            'best_parameters': warm_start_data.best_parameters,
            'neighborhood_scores': warm_start_data.neighborhood_scores,
            'optimization_history': warm_start_data.optimization_history,
            'created_at': warm_start_data.created_at,
            'algorithm_type': warm_start_data.algorithm_type,
            'search_space_hash': warm_start_data.search_space_hash
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        tprint_debug(f"🔥 Saved warm-start data: {key}")
    
    def load_warm_start(self, key: str) -> Optional[WarmStartData]:
        """Load warm-start data for optimization algorithms."""
        file_path = self.warm_start_dir / f"{key}.warm_start"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if expired
            age = time.time() - data['created_at']
            if age > self.config.warm_start_ttl_seconds:
                file_path.unlink()
                return None
            
            warm_start_data = WarmStartData(
                best_parameters=data['best_parameters'],
                neighborhood_scores=data['neighborhood_scores'],
                optimization_history=data['optimization_history'],
                created_at=data['created_at'],
                algorithm_type=data['algorithm_type'],
                search_space_hash=data['search_space_hash']
            )
            
            tprint_debug(f"🔥 Loaded warm-start data: {key}")
            return warm_start_data
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load warm-start data {key}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            l1_size_mb = sum(entry.size_bytes for entry in self.l1_cache.values()) / (1024 * 1024)
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'invalidations': self.stats['invalidations'],
                'compressions': self.stats['compressions'],
                'decompressions': self.stats['decompressions'],
                'l1_entries': len(self.l1_cache),
                'l1_size_mb': l1_size_mb,
                'dependency_graph_nodes': self.dependency_graph.number_of_nodes(),
                'dependency_graph_edges': self.dependency_graph.number_of_edges()
            }
    
    def cleanup(self) -> None:
        """Clean up expired entries and temporary files."""
        with self.lock:
            # Clean up expired L1 entries
            expired_keys = [key for key, entry in self.l1_cache.items() if self._is_expired(entry)]
            for key in expired_keys:
                del self.l1_cache[key]
            
            # Clean up expired L2 entries
            for file_path in self.l2_cache_dir.glob("*.cache"):
                try:
                    with open(file_path, 'rb') as f:
                        entry_data = pickle.load(f)
                    
                    age = time.time() - entry_data['created_at']
                    if age > entry_data.get('ttl_seconds', self.config.default_ttl_seconds):
                        file_path.unlink()
                except:
                    # Remove corrupted files
                    file_path.unlink()
            
            # Clean up expired warm-start data
            for file_path in self.warm_start_dir.glob("*.warm_start"):
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    age = time.time() - data['created_at']
                    if age > self.config.warm_start_ttl_seconds:
                        file_path.unlink()
                except:
                    # Remove corrupted files
                    file_path.unlink()
            
            tprint_debug("🧹 Cache cleanup completed")


# Convenience functions

def create_content_cache(config: Optional[CacheConfig] = None) -> ContentAddressedCache:
    """Create a content-addressed cache with the given configuration."""
    return ContentAddressedCache(config)


def compute_cache_key(data_id: str, symbol: str, timeframe: str,
                     label_spec: Dict[str, Any], feature_signature: str,
                     code_hash: str, search_space: Dict[str, Any],
                     seed: int, gate_hash: str) -> str:
    """Compute a content-addressed cache key."""
    cache = ContentAddressedCache()
    return cache._compute_content_hash(
        data_id, symbol, timeframe, label_spec, feature_signature,
        code_hash, search_space, seed, gate_hash
    )


# Example usage
if __name__ == "__main__":
    # Create cache
    config = CacheConfig(l1_max_size_mb=50.0, l2_max_size_mb=200.0)
    cache = create_content_cache(config)
    
    try:
        # Test basic caching
        print("Testing basic caching...")
        
        # Put some data
        cache.put("test_key", {"data": [1, 2, 3, 4, 5]}, ttl_seconds=60)
        
        # Get data
        data = cache.get("test_key")
        print(f"Retrieved data: {data}")
        
        # Test warm-start data
        print("Testing warm-start data...")
        
        warm_start = WarmStartData(
            best_parameters={"param1": 0.5, "param2": 1.2},
            neighborhood_scores=[({"param1": 0.4}, 0.8), ({"param1": 0.6}, 0.9)],
            optimization_history=[{"iteration": 1, "score": 0.7}],
            created_at=time.time(),
            algorithm_type="tpe",
            search_space_hash="abc123"
        )
        
        cache.save_warm_start("optimization_key", warm_start)
        loaded_warm_start = cache.load_warm_start("optimization_key")
        print(f"Loaded warm-start: {loaded_warm_start.best_parameters}")
        
        # Print statistics
        stats = cache.get_stats()
        print(f"Cache stats: {stats}")
        
    finally:
        cache.cleanup()