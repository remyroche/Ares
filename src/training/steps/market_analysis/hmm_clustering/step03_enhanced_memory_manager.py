#!/usr/bin/env python3
"""Enhanced Memory Manager for Step03 with Chunked Processing and Memory-Aware Operations.

This module provides advanced memory management capabilities including:
1. Chunked data processing for large datasets
2. Memory-aware data loading with automatic chunk sizing
3. Memory monitoring and optimization
4. Intelligent garbage collection
5. Memory usage reporting and analytics
"""

import asyncio
import gc
import logging
import os
import psutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from contextlib import asynccontextmanager
import weakref
from functools import wraps
import tracemalloc

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_usage_percent: float = 80.0
    chunk_size_mb: int = 100
    max_chunk_size_mb: int = 500
    min_chunk_size_mb: int = 10
    gc_threshold_mb: int = 200
    enable_memory_monitoring: bool = True
    enable_chunked_processing: bool = True
    memory_cleanup_interval: int = 30  # seconds
    max_memory_warnings: int = 5

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_usage_percent: float
    process_memory_mb: float
    peak_memory_mb: float
    chunk_count: int
    last_cleanup_time: float
    gc_count: int

class MemoryMonitor:
    """Real-time memory monitoring and management."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MemoryMonitor")
        self.stats = MemoryStats(
            total_memory_mb=0,
            available_memory_mb=0,
            used_memory_mb=0,
            memory_usage_percent=0,
            process_memory_mb=0,
            peak_memory_mb=0,
            chunk_count=0,
            last_cleanup_time=0,
            gc_count=0
        )
        self.memory_warnings = 0
        self._monitoring = False
        self._monitor_task = None
        
        # Enable memory tracing if available
        if hasattr(tracemalloc, 'start'):
            tracemalloc.start()
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            self.stats.total_memory_mb = system_memory.total / (1024**2)
            self.stats.available_memory_mb = system_memory.available / (1024**2)
            self.stats.used_memory_mb = system_memory.used / (1024**2)
            self.stats.memory_usage_percent = system_memory.percent
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            self.stats.process_memory_mb = process_memory.rss / (1024**2)
            
            # Update peak memory
            if self.stats.process_memory_mb > self.stats.peak_memory_mb:
                self.stats.peak_memory_mb = self.stats.process_memory_mb
            
            return self.stats
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return self.stats
    
    def check_memory_availability(self, required_mb: float) -> bool:
        """Check if sufficient memory is available."""
        stats = self.get_memory_stats()
        available_mb = stats.available_memory_mb
        
        if available_mb < required_mb:
            self.logger.warning(f"Insufficient memory: {available_mb:.1f}MB available, {required_mb:.1f}MB required")
            return False
        
        if stats.memory_usage_percent > self.config.max_memory_usage_percent:
            self.logger.warning(f"High memory usage: {stats.memory_usage_percent:.1f}% > {self.config.max_memory_usage_percent}%")
            return False
        
        return True
    
    def force_cleanup(self) -> None:
        """Force memory cleanup."""
        try:
            self.logger.info("🧹 Forcing memory cleanup...")
            
            # Force garbage collection
            collected = gc.collect()
            self.stats.gc_count += 1
            
            # Clear any cached data
            if hasattr(self, '_cache'):
                self._cache.clear()
            
            self.stats.last_cleanup_time = time.time()
            
            self.logger.info(f"✅ Memory cleanup completed. Collected {collected} objects")
            
        except Exception as e:
            self.logger.error(f"Failed to force memory cleanup: {e}")
    
    async def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("🔍 Memory monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("⏹️ Memory monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Memory monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Check for high memory usage
                if stats.memory_usage_percent > self.config.max_memory_usage_percent:
                    self.memory_warnings += 1
                    self.logger.warning(f"⚠️ High memory usage: {stats.memory_usage_percent:.1f}%")
                    
                    if self.memory_warnings >= self.config.max_memory_warnings:
                        self.logger.critical("🚨 Critical memory usage - forcing cleanup")
                        self.force_cleanup()
                        self.memory_warnings = 0
                
                # Periodic cleanup
                if time.time() - self.stats.last_cleanup_time > self.config.memory_cleanup_interval:
                    if stats.process_memory_mb > self.config.gc_threshold_mb:
                        self.force_cleanup()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(10)

class ChunkedDataProcessor:
    """Chunked data processing for large datasets."""
    
    def __init__(self, memory_monitor: MemoryMonitor, config: MemoryConfig):
        self.memory_monitor = memory_monitor
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ChunkedDataProcessor")
    
    def calculate_optimal_chunk_size(self, file_size_mb: float, available_memory_mb: float) -> int:
        """Calculate optimal chunk size based on file size and available memory."""
        # Start with configured chunk size
        chunk_size_mb = self.config.chunk_size_mb
        
        # Adjust based on available memory
        if available_memory_mb < 1000:  # Less than 1GB
            chunk_size_mb = min(chunk_size_mb, 50)
        elif available_memory_mb > 8000:  # More than 8GB
            chunk_size_mb = min(chunk_size_mb * 2, self.config.max_chunk_size_mb)
        
        # Adjust based on file size
        if file_size_mb > 1000:  # Large file
            chunk_size_mb = min(chunk_size_mb, 200)
        elif file_size_mb < 100:  # Small file
            chunk_size_mb = max(chunk_size_mb, self.config.min_chunk_size_mb)
        
        # Ensure we don't exceed available memory
        max_safe_chunk = available_memory_mb * 0.3  # Use max 30% of available memory
        chunk_size_mb = min(chunk_size_mb, max_safe_chunk)
        
        self.logger.info(f"📊 Calculated optimal chunk size: {chunk_size_mb:.1f}MB")
        return int(chunk_size_mb)
    
    async def process_file_in_chunks(self, file_path: Path, 
                                   process_func: Callable[[pd.DataFrame], Any],
                                   chunk_size_mb: Optional[int] = None) -> List[Any]:
        """Process a file in chunks."""
        try:
            # Get file size
            file_size_mb = file_path.stat().st_size / (1024**2)
            self.logger.info(f"📁 Processing file: {file_path.name} ({file_size_mb:.1f}MB)")
            
            # Calculate chunk size
            if chunk_size_mb is None:
                stats = self.memory_monitor.get_memory_stats()
                chunk_size_mb = self.calculate_optimal_chunk_size(file_size_mb, stats.available_memory_mb)
            
            # Calculate number of rows per chunk
            # Estimate rows per MB (rough approximation)
            estimated_rows_per_mb = 10000  # Conservative estimate
            rows_per_chunk = int(chunk_size_mb * estimated_rows_per_mb)
            
            self.logger.info(f"🔄 Processing in chunks of {rows_per_chunk:,} rows (~{chunk_size_mb:.1f}MB)")
            
            results = []
            chunk_count = 0
            
            # Process file in chunks
            for chunk_df in pd.read_parquet(file_path, chunksize=rows_per_chunk):
                chunk_count += 1
                self.logger.info(f"📦 Processing chunk {chunk_count} ({len(chunk_df):,} rows)")
                
                # Check memory before processing
                if not self.memory_monitor.check_memory_availability(chunk_size_mb * 2):
                    self.logger.warning("⚠️ Low memory - forcing cleanup before chunk processing")
                    self.memory_monitor.force_cleanup()
                
                # Process chunk
                try:
                    result = await self._process_chunk_async(chunk_df, process_func)
                    results.append(result)
                    
                    # Update memory stats
                    self.memory_monitor.stats.chunk_count = chunk_count
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing chunk {chunk_count}: {e}")
                    raise
                
                # Cleanup chunk data
                del chunk_df
                gc.collect()
            
            self.logger.info(f"✅ File processing completed: {chunk_count} chunks processed")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process file in chunks: {e}")
            raise
    
    async def _process_chunk_async(self, chunk_df: pd.DataFrame, 
                                 process_func: Callable[[pd.DataFrame], Any]) -> Any:
        """Process a single chunk asynchronously."""
        # Run CPU-intensive processing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process_func, chunk_df)

class IntelligentCache:
    """Intelligent caching system with memory awareness."""
    
    def __init__(self, memory_monitor: MemoryMonitor, max_cache_size_mb: int = 500):
        self.memory_monitor = memory_monitor
        self.max_cache_size_mb = max_cache_size_mb
        self.logger = logging.getLogger(f"{__name__}.IntelligentCache")
        
        # Use weak references to avoid memory leaks
        self._cache = {}
        self._cache_metadata = {}
        self._access_times = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_mb = sum(meta.get('size_mb', 0) for meta in self._cache_metadata.values())
        hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        
        return {
            'total_entries': len(self._cache),
            'total_size_mb': total_size_mb,
            'hit_rate': hit_rate,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses
        }
    
    def _estimate_size_mb(self, obj: Any) -> float:
        """Estimate object size in MB."""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / (1024**2)
            elif isinstance(obj, np.ndarray):
                return obj.nbytes / (1024**2)
            else:
                import sys
                return sys.getsizeof(obj) / (1024**2)
        except:
            return 0.1  # Default estimate
    
    def _evict_old_entries(self) -> None:
        """Evict old cache entries to free memory."""
        if not self._cache:
            return
        
        # Sort by access time (oldest first)
        sorted_entries = sorted(self._access_times.items(), key=lambda x: x[1])
        
        # Remove oldest entries until we're under the limit
        current_size_mb = sum(meta.get('size_mb', 0) for meta in self._cache_metadata.values())
        
        for key, _ in sorted_entries:
            if current_size_mb <= self.max_cache_size_mb * 0.8:  # Keep 80% of max size
                break
            
            if key in self._cache:
                size_mb = self._cache_metadata.get(key, {}).get('size_mb', 0)
                del self._cache[key]
                del self._cache_metadata[key]
                del self._access_times[key]
                current_size_mb -= size_mb
                
                self.logger.info(f"🗑️ Evicted cache entry: {key} ({size_mb:.1f}MB)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            self._cache_hits += 1
            self._access_times[key] = time.time()
            return self._cache[key]
        else:
            self._cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache."""
        try:
            # Estimate size
            size_mb = self._estimate_size_mb(value)
            
            # Check if we need to evict entries
            current_size_mb = sum(meta.get('size_mb', 0) for meta in self._cache_metadata.values())
            if current_size_mb + size_mb > self.max_cache_size_mb:
                self._evict_old_entries()
            
            # Store value
            self._cache[key] = value
            self._cache_metadata[key] = {
                'size_mb': size_mb,
                'created_at': time.time(),
                'ttl_seconds': ttl_seconds
            }
            self._access_times[key] = time.time()
            
            self.logger.debug(f"💾 Cached: {key} ({size_mb:.1f}MB)")
            
        except Exception as e:
            self.logger.error(f"Failed to cache value for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._cache_metadata.clear()
        self._access_times.clear()
        self.logger.info("🧹 Cache cleared")

class EnhancedMemoryManager:
    """Enhanced memory manager with all optimization features."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(f"{__name__}.EnhancedMemoryManager")
        
        # Initialize components
        self.memory_monitor = MemoryMonitor(self.config)
        self.chunked_processor = ChunkedDataProcessor(self.memory_monitor, self.config)
        self.cache = IntelligentCache(self.memory_monitor)
        
        self.logger.info("🚀 Enhanced Memory Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the memory manager."""
        if self.config.enable_memory_monitoring:
            await self.memory_monitor.start_monitoring()
        
        self.logger.info("✅ Enhanced Memory Manager initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup memory manager resources."""
        await self.memory_monitor.stop_monitoring()
        self.cache.clear()
        self.memory_monitor.force_cleanup()
        
        self.logger.info("🧹 Enhanced Memory Manager cleaned up")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        stats = self.memory_monitor.get_memory_stats()
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'system_memory': {
                'total_mb': stats.total_memory_mb,
                'available_mb': stats.available_memory_mb,
                'used_mb': stats.used_memory_mb,
                'usage_percent': stats.memory_usage_percent
            },
            'process_memory': {
                'current_mb': stats.process_memory_mb,
                'peak_mb': stats.peak_memory_mb
            },
            'processing_stats': {
                'chunks_processed': stats.chunk_count,
                'gc_count': stats.gc_count,
                'last_cleanup': stats.last_cleanup_time
            },
            'cache_stats': cache_stats,
            'config': {
                'max_memory_percent': self.config.max_memory_usage_percent,
                'chunk_size_mb': self.config.chunk_size_mb,
                'monitoring_enabled': self.config.enable_memory_monitoring
            }
        }
    
    @asynccontextmanager
    async def memory_context(self, operation_name: str):
        """Context manager for memory-aware operations."""
        start_stats = self.memory_monitor.get_memory_stats()
        self.logger.info(f"🔍 Starting memory-aware operation: {operation_name}")
        
        try:
            yield self
        finally:
            end_stats = self.memory_monitor.get_memory_stats()
            memory_delta = end_stats.process_memory_mb - start_stats.process_memory_mb
            
            self.logger.info(f"✅ Completed operation: {operation_name}")
            self.logger.info(f"📊 Memory delta: {memory_delta:+.1f}MB")
            
            # Force cleanup if memory usage increased significantly
            if memory_delta > 100:  # More than 100MB increase
                self.logger.warning(f"⚠️ High memory increase ({memory_delta:.1f}MB) - forcing cleanup")
                self.memory_monitor.force_cleanup()

# Global instance
_enhanced_memory_manager = None

def get_enhanced_memory_manager(config: Optional[MemoryConfig] = None) -> EnhancedMemoryManager:
    """Get or create global enhanced memory manager instance."""
    global _enhanced_memory_manager
    if _enhanced_memory_manager is None:
        _enhanced_memory_manager = EnhancedMemoryManager(config)
    return _enhanced_memory_manager

# Decorator for memory-aware functions
def memory_aware(func):
    """Decorator for memory-aware function execution."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        memory_manager = get_enhanced_memory_manager()
        
        async with memory_manager.memory_context(func.__name__):
            return await func(*args, **kwargs)
    
    return wrapper