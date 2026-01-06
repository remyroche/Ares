"""
Specialist Model Cache

This module provides intelligent caching for specialist models with warm-start capability
to avoid redundant training and improve computational efficiency.
"""

import pickle
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success
from src.utils.common_operations import ensure_directory, safe_file_exists

logger = system_logger.getChild('SpecialistCache')


@dataclass
class CacheMetadata:
    """Metadata for cached specialist model"""
    specialist_name: str
    model_type: str
    timestamp: str
    data_hash: str
    config_hash: str
    performance_metrics: Dict[str, float]
    training_time: float
    model_size_mb: float


class SpecialistModelCache:
    """Intelligent caching for specialist models with warm-start capability"""
    
    def __init__(self, cache_dir: Path = None, max_memory_gb: float = 2.0, max_disk_gb: float = 10.0):
        self.cache_dir = cache_dir or Path("cache/specialist_models")
        self.max_memory_gb = max_memory_gb
        self.max_disk_gb = max_disk_gb
        
        # Ensure cache directory exists
        ensure_directory(self.cache_dir)
        
        # In-memory cache (LRU)
        self.memory_cache = OrderedDict()
        self.memory_usage_mb = 0.0
        
        # Metadata cache
        self.metadata_cache = {}
        self._load_metadata_cache()
        
        tprint_info(f"📂 Specialist cache initialized: {self.cache_dir}")
        tprint_info(f"   Memory limit: {self.max_memory_gb:.1f}GB, Disk limit: {self.max_disk_gb:.1f}GB")
    
    def get_cached_specialist(self, specialist_name: str, config_hash: str, 
                             data_hash: Optional[str] = None) -> Optional[Any]:
        """Retrieve cached specialist model if valid"""
        
        cache_key = f"{specialist_name}_{config_hash}"
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            metadata = self.metadata_cache.get(cache_key)
            if metadata and self._is_cache_valid(metadata, data_hash):
                tprint_info(f"🎯 Memory cache hit: {specialist_name}")
                # Move to end (LRU)
                self.memory_cache.move_to_end(cache_key)
                return self.memory_cache[cache_key]
            else:
                # Remove invalid cache
                self._remove_from_memory_cache(cache_key)
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    model, metadata = pickle.load(f)
                
                if self._is_cache_valid(metadata, data_hash):
                    # Load into memory if space available
                    if self._can_fit_in_memory(metadata.model_size_mb):
                        self._add_to_memory_cache(cache_key, model, metadata)
                    
                    tprint_info(f"💾 Disk cache hit: {specialist_name}")
                    return model
                else:
                    # Remove invalid cache file
                    cache_file.unlink()
                    if cache_key in self.metadata_cache:
                        del self.metadata_cache[cache_key]
                    
            except Exception as e:
                logger.warning(f"Failed to load cached model {specialist_name}: {e}")
                cache_file.unlink()
        
        return None
    
    def cache_specialist(self, specialist_name: str, model: Any, metadata: CacheMetadata) -> bool:
        """Cache trained specialist with metadata"""
        
        cache_key = f"{specialist_name}_{metadata.config_hash}"
        
        try:
            # Calculate model size
            model_size_mb = self._estimate_model_size(model)
            metadata.model_size_mb = model_size_mb
            
            # Add to memory cache if possible
            if self._can_fit_in_memory(model_size_mb):
                self._add_to_memory_cache(cache_key, model, metadata)
                tprint_info(f"🧠 Cached {specialist_name} in memory ({model_size_mb:.1f}MB)")
            
            # Save to disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump((model, metadata), f)
            
            # Update metadata cache
            self.metadata_cache[cache_key] = metadata
            self._save_metadata_cache()
            
            # Check disk usage and clean if needed
            self._cleanup_disk_cache()
            
            tprint_success(f"✅ Cached {specialist_name} ({model_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {specialist_name}: {e}")
            return False
    
    def is_cached(self, specialist_name: str, config_hash: str, data_hash: Optional[str] = None) -> bool:
        """Check if specialist is cached and valid"""
        cache_key = f"{specialist_name}_{config_hash}"
        
        # Check memory cache
        if cache_key in self.memory_cache:
            metadata = self.metadata_cache.get(cache_key)
            return metadata and self._is_cache_valid(metadata, data_hash)
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            metadata = self.metadata_cache.get(cache_key)
            return metadata and self._is_cache_valid(metadata, data_hash)
        
        return False
    
    def invalidate_cache(self, specialist_name: Optional[str] = None, reason: str = "manual") -> int:
        """Invalidate cache entries"""
        
        invalidated_count = 0
        
        if specialist_name:
            # Invalidate specific specialist
            keys_to_remove = [key for key in self.metadata_cache.keys() if key.startswith(f"{specialist_name}_")]
        else:
            # Invalidate all cache
            keys_to_remove = list(self.metadata_cache.keys())
        
        for cache_key in keys_to_remove:
            # Remove from memory
            if cache_key in self.memory_cache:
                model = self.memory_cache.pop(cache_key)
                self.memory_usage_mb -= self._estimate_model_size(model)
            
            # Remove from disk
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove from metadata
            if cache_key in self.metadata_cache:
                del self.metadata_cache[cache_key]
            
            invalidated_count += 1
        
        # Save updated metadata
        self._save_metadata_cache()
        
        tprint_warning(f"🗑️  Invalidated {invalidated_count} cache entries ({reason})")
        return invalidated_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        
        # Memory statistics
        memory_entries = len(self.memory_cache)
        memory_usage_mb = self.memory_usage_mb
        memory_utilization = memory_usage_mb / (self.max_memory_gb * 1024) * 100
        
        # Disk statistics
        disk_entries = len(self.metadata_cache)
        disk_usage_mb = sum(meta.model_size_mb for meta in self.metadata_cache.values())
        disk_utilization = disk_usage_mb / (self.max_disk_gb * 1024) * 100
        
        # Cache hit rates (placeholder - would need tracking)
        cache_hits = getattr(self, '_cache_hits', 0)
        cache_misses = getattr(self, '_cache_misses', 0)
        hit_rate = cache_hits / (cache_hits + cache_misses) * 100 if (cache_hits + cache_misses) > 0 else 0
        
        return {
            'memory': {
                'entries': memory_entries,
                'usage_mb': memory_usage_mb,
                'utilization_percent': memory_utilization,
                'limit_mb': self.max_memory_gb * 1024
            },
            'disk': {
                'entries': disk_entries,
                'usage_mb': disk_usage_mb,
                'utilization_percent': disk_utilization,
                'limit_mb': self.max_disk_gb * 1024
            },
            'performance': {
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'hit_rate_percent': hit_rate
            }
        }
    
    def _is_cache_valid(self, metadata: CacheMetadata, data_hash: Optional[str] = None) -> bool:
        """Check if cached model is still valid"""
        
        # Check age (cache expires after 7 days)
        try:
            cache_time = datetime.fromisoformat(metadata.timestamp)
            age = datetime.now() - cache_time
            if age > timedelta(days=7):
                return False
        except:
            return False
        
        # Check data hash if provided
        if data_hash and metadata.data_hash != data_hash:
            return False
        
        return True
    
    def _can_fit_in_memory(self, model_size_mb: float) -> bool:
        """Check if model can fit in memory cache"""
        return (self.memory_usage_mb + model_size_mb) <= (self.max_memory_gb * 1024)
    
    def _add_to_memory_cache(self, cache_key: str, model: Any, metadata: CacheMetadata):
        """Add model to memory cache with LRU eviction"""
        
        # Remove oldest entries if needed
        while not self._can_fit_in_memory(metadata.model_size_mb) and self.memory_cache:
            oldest_key, oldest_model = self.memory_cache.popitem(last=False)
            oldest_size = self._estimate_model_size(oldest_model)
            self.memory_usage_mb -= oldest_size
            tprint_info(f"🔄 Evicted {oldest_key} from memory cache")
        
        # Add new entry
        self.memory_cache[cache_key] = model
        self.memory_usage_mb += metadata.model_size_mb
    
    def _remove_from_memory_cache(self, cache_key: str):
        """Remove entry from memory cache"""
        if cache_key in self.memory_cache:
            model = self.memory_cache.pop(cache_key)
            self.memory_usage_mb -= self._estimate_model_size(model)
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB"""
        try:
            # Use pickle size as estimate
            size_bytes = len(pickle.dumps(model))
            return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 10.0  # Default estimate
    
    def _cleanup_disk_cache(self):
        """Clean up disk cache if over limit"""
        
        total_size = sum(meta.model_size_mb for meta in self.metadata_cache.values())
        max_size_mb = self.max_disk_gb * 1024
        
        if total_size <= max_size_mb:
            return
        
        # Sort by age (oldest first)
        sorted_entries = sorted(
            self.metadata_cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest entries until under limit
        removed_count = 0
        for cache_key, metadata in sorted_entries:
            if total_size <= max_size_mb * 0.8:  # Leave 20% buffer
                break
            
            # Remove file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove from metadata
            del self.metadata_cache[cache_key]
            total_size -= metadata.model_size_mb
            removed_count += 1
        
        if removed_count > 0:
            tprint_info(f"🗑️  Cleaned up {removed_count} old disk cache entries")
    
    def _load_metadata_cache(self):
        """Load metadata cache from disk"""
        metadata_file = self.cache_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Convert back to CacheMetadata objects
                for cache_key, metadata_dict in data.items():
                    self.metadata_cache[cache_key] = CacheMetadata(**metadata_dict)
                
                tprint_info(f"📋 Loaded {len(self.metadata_cache)} cache metadata entries")
                
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self.metadata_cache = {}
    
    def _save_metadata_cache(self):
        """Save metadata cache to disk"""
        metadata_file = self.cache_dir / "metadata.json"
        
        try:
            # Convert to serializable format
            data = {
                cache_key: asdict(metadata)
                for cache_key, metadata in self.metadata_cache.items()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")


def calculate_data_hash(data: pd.DataFrame) -> str:
    """Calculate hash for DataFrame to detect changes"""
    # Use shape, columns, and sample of data for hash
    hash_input = f"{data.shape}_{list(data.columns)}_{data.iloc[:100].values.tobytes()}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def calculate_config_hash(config: Dict[str, Any]) -> str:
    """Calculate hash for configuration"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def create_specialist_cache(cache_dir: Path = None, max_memory_gb: float = 2.0, 
                          max_disk_gb: float = 10.0) -> SpecialistModelCache:
    """Factory function to create specialist model cache"""
    return SpecialistModelCache(cache_dir, max_memory_gb, max_disk_gb)
