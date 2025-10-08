"""
Model Caching System for Warm-Start Training

This module provides comprehensive model caching capabilities for efficient
warm-start training and model reuse across training iterations.

Features:
- In-memory and disk-based caching
- Automatic cache invalidation
- Cache size management (LRU eviction)
- Model metadata tracking
- Warm-start capability
- Thread-safe operations
- Comprehensive cache statistics
"""

import hashlib
import pickle
import joblib
import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success, tprint_timer
)
from src.utils.common_operations import (
    ensure_directory, safe_file_exists, get_current_datetime, safe_json_dump, safe_json_load
)
from src.utils.math_validation import validate_positive

logger = system_logger.getChild('ModelCache')


@dataclass
class CachedModelMetadata:
    """Metadata for a cached model."""
    
    model_id: str
    model_type: str
    regime: str
    timestamp: str
    data_hash: str
    config_hash: str
    
    # Model metrics
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    n_samples: int = 0
    n_features: int = 0
    
    # Cache metadata
    access_count: int = 0
    last_accessed: Optional[str] = None
    size_bytes: int = 0
    
    # Training metadata
    training_duration: float = 0.0
    hyperparameters: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedModelMetadata':
        """Create from dictionary."""
        return cls(**data)


class ModelCache:
    """
    Thread-safe model caching system with LRU eviction.
    
    Features:
    - In-memory caching for fast access
    - Disk-based caching for persistence
    - Automatic cache invalidation based on data/config changes
    - LRU eviction when cache size exceeds limits
    - Warm-start capability for incremental training
    - Comprehensive cache statistics
    
    Example:
        cache = ModelCache(max_memory_models=10, cache_dir="./cache")
        
        # Cache a model
        cache.put_model(model, regime="volatile", data_hash="abc123")
        
        # Retrieve a model
        cached_model, metadata = cache.get_model(regime="volatile", data_hash="abc123")
        
        # Check if model exists
        if cache.has_model(regime="volatile", data_hash="abc123"):
            model, metadata = cache.get_model(regime="volatile")
    """
    
    def __init__(
        self,
        max_memory_models: int = 10,
        max_disk_models: int = 50,
        cache_dir: str = "./cache/models",
        enable_disk_cache: bool = True,
        cache_ttl_hours: float = 24.0,
        auto_cleanup: bool = True
    ):
        """
        Initialize model cache.
        
        Args:
            max_memory_models: Maximum number of models in memory
            max_disk_models: Maximum number of models on disk
            cache_dir: Directory for disk cache
            enable_disk_cache: Whether to enable disk caching
            cache_ttl_hours: Time-to-live for cached models in hours
            auto_cleanup: Whether to automatically cleanup expired models
        """
        validate_positive(max_memory_models, "max_memory_models")
        validate_positive(max_disk_models, "max_disk_models")
        validate_positive(cache_ttl_hours, "cache_ttl_hours")
        
        self.max_memory_models = max_memory_models
        self.max_disk_models = max_disk_models
        self.cache_dir = Path(cache_dir)
        self.enable_disk_cache = enable_disk_cache
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.auto_cleanup = auto_cleanup
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache (LRU using OrderedDict)
        self._memory_cache: OrderedDict[str, Tuple[Any, CachedModelMetadata]] = OrderedDict()
        
        # Metadata index
        self._metadata_index: Dict[str, CachedModelMetadata] = {}
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
            'disk_saves': 0,
            'disk_loads': 0,
            'errors': 0
        }
        
        # Initialize disk cache
        if self.enable_disk_cache:
            ensure_directory(str(self.cache_dir))
            self._load_metadata_index()
        
        logger.info(f"✅ ModelCache initialized (memory={max_memory_models}, disk={max_disk_models})")
    
    def _generate_cache_key(
        self,
        regime: str,
        model_type: str = "unknown",
        data_hash: Optional[str] = None,
        config_hash: Optional[str] = None
    ) -> str:
        """
        Generate unique cache key.
        
        Args:
            regime: Regime identifier
            model_type: Type of model
            data_hash: Hash of training data
            config_hash: Hash of model configuration
            
        Returns:
            Unique cache key
        """
        key_parts = [regime, model_type]
        if data_hash:
            key_parts.append(data_hash)
        if config_hash:
            key_parts.append(config_hash)
        
        key = "_".join(key_parts)
        return key
    
    def _hash_data(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Generate hash of training data.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Hash string
        """
        try:
            # Use sample of data for efficiency
            sample_size = min(1000, len(X))
            indices = np.linspace(0, len(X) - 1, sample_size, dtype=int)
            
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Create hash
            data_bytes = X_sample.tobytes() + y_sample.tobytes()
            return hashlib.md5(data_bytes).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to hash data: {e}")
            return "no_hash"
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """
        Generate hash of model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Hash string
        """
        try:
            # Sort keys for consistent hashing
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Failed to hash config: {e}")
            return "no_hash"
    
    def put_model(
        self,
        model: Any,
        regime: str,
        model_type: str = "unknown",
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[CachedModelMetadata] = None
    ) -> str:
        """
        Cache a model.
        
        Args:
            model: Model to cache
            regime: Regime identifier
            model_type: Type of model
            X: Training features (for data hash)
            y: Training targets (for data hash)
            config: Model configuration (for config hash)
            metadata: Optional pre-computed metadata
            
        Returns:
            Cache key
        """
        with self._lock:
            try:
                # Generate hashes
                data_hash = self._hash_data(X, y) if X is not None and y is not None else "no_data_hash"
                config_hash = self._hash_config(config) if config else "no_config_hash"
                
                # Generate cache key
                cache_key = self._generate_cache_key(regime, model_type, data_hash, config_hash)
                
                # Create or update metadata
                if metadata is None:
                    metadata = CachedModelMetadata(
                        model_id=cache_key,
                        model_type=model_type,
                        regime=regime,
                        timestamp=get_current_datetime().isoformat(),
                        data_hash=data_hash,
                        config_hash=config_hash,
                        n_samples=len(X) if X is not None else 0,
                        n_features=X.shape[1] if X is not None else 0,
                        hyperparameters=config
                    )
                
                # Store in memory cache (LRU)
                self._memory_cache[cache_key] = (model, metadata)
                self._memory_cache.move_to_end(cache_key)  # Mark as most recently used
                
                # Store metadata
                self._metadata_index[cache_key] = metadata
                
                # Evict if necessary
                if len(self._memory_cache) > self.max_memory_models:
                    self._evict_lru()
                
                # Save to disk if enabled
                if self.enable_disk_cache:
                    self._save_to_disk(cache_key, model, metadata)
                
                self._stats['puts'] += 1
                tprint_success(f"✅ Cached model: {cache_key}")
                
                return cache_key
                
            except Exception as e:
                self._stats['errors'] += 1
                tprint_error(f"❌ Failed to cache model: {e}")
                raise
    
    def get_model(
        self,
        regime: str,
        model_type: str = "unknown",
        data_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        allow_disk_load: bool = True
    ) -> Optional[Tuple[Any, CachedModelMetadata]]:
        """
        Retrieve cached model.
        
        Args:
            regime: Regime identifier
            model_type: Type of model
            data_hash: Hash of training data
            config_hash: Hash of model configuration
            allow_disk_load: Whether to load from disk if not in memory
            
        Returns:
            Tuple of (model, metadata) or None if not found
        """
        with self._lock:
            try:
                cache_key = self._generate_cache_key(regime, model_type, data_hash, config_hash)
                
                # Check memory cache
                if cache_key in self._memory_cache:
                    model, metadata = self._memory_cache[cache_key]
                    
                    # Check if expired
                    if self._is_expired(metadata):
                        tprint_warning(f"⚠️ Cached model expired: {cache_key}")
                        self._remove_model(cache_key)
                        self._stats['misses'] += 1
                        return None
                    
                    # Update access metadata
                    metadata.access_count += 1
                    metadata.last_accessed = get_current_datetime().isoformat()
                    self._memory_cache.move_to_end(cache_key)  # Mark as most recently used
                    
                    self._stats['hits'] += 1
                    tprint_success(f"✅ Cache hit: {cache_key}")
                    return (model, metadata)
                
                # Check disk cache
                if allow_disk_load and self.enable_disk_cache:
                    result = self._load_from_disk(cache_key)
                    if result is not None:
                        model, metadata = result
                        
                        # Check if expired
                        if self._is_expired(metadata):
                            tprint_warning(f"⚠️ Cached model expired: {cache_key}")
                            self._remove_from_disk(cache_key)
                            self._stats['misses'] += 1
                            return None
                        
                        # Store in memory cache
                        self._memory_cache[cache_key] = (model, metadata)
                        self._memory_cache.move_to_end(cache_key)
                        
                        # Update access metadata
                        metadata.access_count += 1
                        metadata.last_accessed = get_current_datetime().isoformat()
                        
                        self._stats['hits'] += 1
                        self._stats['disk_loads'] += 1
                        tprint_success(f"✅ Cache hit (disk): {cache_key}")
                        return (model, metadata)
                
                # Cache miss
                self._stats['misses'] += 1
                tprint_info(f"ℹ️ Cache miss: {cache_key}")
                return None
                
            except Exception as e:
                self._stats['errors'] += 1
                tprint_error(f"❌ Failed to retrieve cached model: {e}")
                return None
    
    def has_model(
        self,
        regime: str,
        model_type: str = "unknown",
        data_hash: Optional[str] = None,
        config_hash: Optional[str] = None
    ) -> bool:
        """
        Check if model exists in cache.
        
        Args:
            regime: Regime identifier
            model_type: Type of model
            data_hash: Hash of training data
            config_hash: Hash of model configuration
            
        Returns:
            True if model exists in cache
        """
        cache_key = self._generate_cache_key(regime, model_type, data_hash, config_hash)
        
        with self._lock:
            # Check memory
            if cache_key in self._memory_cache:
                metadata = self._memory_cache[cache_key][1]
                if not self._is_expired(metadata):
                    return True
            
            # Check disk
            if self.enable_disk_cache:
                disk_path = self._get_disk_path(cache_key)
                if disk_path.exists():
                    # Check metadata for expiration
                    if cache_key in self._metadata_index:
                        metadata = self._metadata_index[cache_key]
                        if not self._is_expired(metadata):
                            return True
            
            return False
    
    def invalidate(
        self,
        regime: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> int:
        """
        Invalidate cached models matching criteria.
        
        Args:
            regime: Regime identifier (None = all regimes)
            model_type: Model type (None = all types)
            
        Returns:
            Number of models invalidated
        """
        with self._lock:
            invalidated = 0
            
            # Find matching keys
            keys_to_remove = []
            for key in list(self._memory_cache.keys()):
                _, metadata = self._memory_cache[key]
                
                if regime is not None and metadata.regime != regime:
                    continue
                if model_type is not None and metadata.model_type != model_type:
                    continue
                
                keys_to_remove.append(key)
            
            # Remove from caches
            for key in keys_to_remove:
                self._remove_model(key)
                invalidated += 1
            
            tprint_info(f"🗑️ Invalidated {invalidated} cached models")
            return invalidated
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._memory_cache.clear()
            self._metadata_index.clear()
            
            if self.enable_disk_cache:
                # Clear disk cache
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                for metadata_file in self.cache_dir.glob("*.json"):
                    metadata_file.unlink()
            
            tprint_success("✅ Cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used model from memory cache."""
        if not self._memory_cache:
            return
        
        # Remove oldest (first) item
        cache_key, (model, metadata) = self._memory_cache.popitem(last=False)
        
        self._stats['evictions'] += 1
        tprint_info(f"🗑️ Evicted LRU model: {cache_key}")
    
    def _is_expired(self, metadata: CachedModelMetadata) -> bool:
        """Check if cached model is expired."""
        if not self.auto_cleanup:
            return False
        
        try:
            cached_time = datetime.fromisoformat(metadata.timestamp)
            age = datetime.now() - cached_time
            return age > self.cache_ttl
        except Exception:
            return False
    
    def _remove_model(self, cache_key: str) -> None:
        """Remove model from all caches."""
        # Remove from memory
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Remove from metadata index
        if cache_key in self._metadata_index:
            del self._metadata_index[cache_key]
        
        # Remove from disk
        if self.enable_disk_cache:
            self._remove_from_disk(cache_key)
    
    def _get_disk_path(self, cache_key: str) -> Path:
        """Get disk path for cached model."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get disk path for model metadata."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _save_to_disk(self, cache_key: str, model: Any, metadata: CachedModelMetadata) -> None:
        """Save model and metadata to disk."""
        try:
            # Save model
            model_path = self._get_disk_path(cache_key)
            joblib.dump(model, model_path, compress=3)
            
            # Update metadata with file size
            metadata.size_bytes = model_path.stat().st_size
            
            # Save metadata
            metadata_path = self._get_metadata_path(cache_key)
            safe_json_dump(metadata.to_dict(), metadata_path)
            
            self._stats['disk_saves'] += 1
            
            # Check disk cache size
            self._cleanup_disk_cache()
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save model to disk: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Tuple[Any, CachedModelMetadata]]:
        """Load model and metadata from disk."""
        try:
            model_path = self._get_disk_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            if not model_path.exists() or not metadata_path.exists():
                return None
            
            # Load metadata
            metadata_dict = safe_json_load(metadata_path)
            if metadata_dict is None:
                return None
            
            metadata = CachedModelMetadata.from_dict(metadata_dict)
            
            # Load model
            model = joblib.load(model_path)
            
            return (model, metadata)
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load model from disk: {e}")
            return None
    
    def _remove_from_disk(self, cache_key: str) -> None:
        """Remove model and metadata from disk."""
        try:
            model_path = self._get_disk_path(cache_key)
            metadata_path = self._get_metadata_path(cache_key)
            
            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to remove model from disk: {e}")
    
    def _cleanup_disk_cache(self) -> None:
        """Cleanup disk cache if it exceeds size limit."""
        try:
            # Get all cached models with metadata
            cached_models = []
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_key = cache_file.stem
                metadata_path = self._get_metadata_path(cache_key)
                
                if metadata_path.exists():
                    metadata_dict = safe_json_load(metadata_path)
                    if metadata_dict:
                        metadata = CachedModelMetadata.from_dict(metadata_dict)
                        cached_models.append((cache_key, metadata))
            
            # Remove if exceeds limit
            if len(cached_models) > self.max_disk_models:
                # Sort by last accessed (oldest first)
                cached_models.sort(
                    key=lambda x: x[1].last_accessed or x[1].timestamp
                )
                
                # Remove oldest
                to_remove = len(cached_models) - self.max_disk_models
                for cache_key, _ in cached_models[:to_remove]:
                    self._remove_from_disk(cache_key)
                    tprint_info(f"🗑️ Removed old cached model: {cache_key}")
                    
        except Exception as e:
            tprint_warning(f"⚠️ Disk cache cleanup failed: {e}")
    
    def _load_metadata_index(self) -> None:
        """Load metadata index from disk."""
        try:
            for metadata_file in self.cache_dir.glob("*.json"):
                metadata_dict = safe_json_load(metadata_file)
                if metadata_dict:
                    metadata = CachedModelMetadata.from_dict(metadata_dict)
                    self._metadata_index[metadata.model_id] = metadata
                    
            tprint_info(f"📚 Loaded {len(self._metadata_index)} model metadata entries")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load metadata index: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            hit_rate = self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
            
            return {
                'memory_models': len(self._memory_cache),
                'disk_models': len(list(self.cache_dir.glob("*.pkl"))) if self.enable_disk_cache else 0,
                'total_puts': self._stats['puts'],
                'total_hits': self._stats['hits'],
                'total_misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'disk_saves': self._stats['disk_saves'],
                'disk_loads': self._stats['disk_loads'],
                'errors': self._stats['errors']
            }
    
    def list_cached_models(self) -> List[CachedModelMetadata]:
        """
        List all cached models.
        
        Returns:
            List of model metadata
        """
        with self._lock:
            return list(self._metadata_index.values())


# Global cache instance
_global_model_cache: Optional[ModelCache] = None


def get_model_cache(
    max_memory_models: int = 10,
    max_disk_models: int = 50,
    cache_dir: str = "./cache/models"
) -> ModelCache:
    """
    Get or create global model cache instance.
    
    Args:
        max_memory_models: Maximum models in memory
        max_disk_models: Maximum models on disk
        cache_dir: Cache directory path
        
    Returns:
        Global ModelCache instance
    """
    global _global_model_cache
    
    if _global_model_cache is None:
        _global_model_cache = ModelCache(
            max_memory_models=max_memory_models,
            max_disk_models=max_disk_models,
            cache_dir=cache_dir
        )
    
    return _global_model_cache


def clear_global_cache() -> None:
    """Clear the global model cache."""
    global _global_model_cache
    
    if _global_model_cache is not None:
        _global_model_cache.clear()
        _global_model_cache = None
