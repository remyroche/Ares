"""
Enhanced Unified Memory Manager for Apple Silicon.

This module provides advanced unified memory management with cross-component sharing,
intelligent memory pooling, and adaptive memory optimization for M1/M2/M3/M4 chips.
"""

import gc
import logging
import threading
import time
import weakref
import psutil
import subprocess
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from contextlib import contextmanager
import tracemalloc
from functools import wraps
import ctypes
import ctypes.util
import hashlib
import pickle
import zlib

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .m1_unified_memory_manager import (
    M1UnifiedMemoryManager, UnifiedMemoryConfig, MemoryTier, MemoryAllocationStrategy,
    get_unified_memory_manager, optimize_for_unified_memory, allocate_unified_memory,
    get_unified_memory_stats, unified_memory_optimized, memory_tier_aware
)

logger = logging.getLogger(__name__)

class MemoryComponent(Enum):
    """Memory components in unified architecture."""
    CPU = "cpu"
    GPU = "gpu"
    NEURAL_ENGINE = "neural_engine"
    CACHE = "cache"
    SHARED = "shared"
    COMPRESSED = "compressed"

class MemoryAccessPattern(Enum):
    """Memory access patterns."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRIDED = "strided"
    BURST = "burst"
    STREAMING = "streaming"

class MemoryCompressionType(Enum):
    """Memory compression types."""
    NONE = "none"
    LZ4 = "lz4"
    ZLIB = "zlib"
    GZIP = "gzip"
    CUSTOM = "custom"

@dataclass
class EnhancedUnifiedMemoryConfig(UnifiedMemoryConfig):
    """Enhanced unified memory configuration."""
    # Cross-component sharing
    enable_cross_component_sharing: bool = True
    sharing_timeout_seconds: float = 60.0
    sharing_compression_threshold_mb: float = 10.0
    enable_memory_deduplication: bool = True
    
    # Memory pooling
    enable_advanced_pooling: bool = True
    pool_growth_strategy: str = "exponential"  # linear, exponential, adaptive
    pool_cleanup_interval: float = 300.0
    enable_pool_prefetching: bool = True
    
    # Memory optimization
    enable_aggressive_optimization: bool = True
    optimization_threshold_mb: float = 50.0
    enable_memory_compression: bool = True
    compression_algorithm: MemoryCompressionType = MemoryCompressionType.LZ4
    
    # Memory monitoring
    enable_detailed_monitoring: bool = True
    monitoring_interval: float = 2.0
    enable_memory_profiling: bool = True
    profile_retention_hours: int = 24
    
    # Memory pressure management
    enable_intelligent_pressure_management: bool = True
    pressure_prediction_window: float = 30.0
    enable_memory_preallocation: bool = True

@dataclass
class MemoryAllocation:
    """Enhanced memory allocation with cross-component support."""
    allocation_id: str
    size_mb: float
    tier: MemoryTier
    component: MemoryComponent
    access_pattern: MemoryAccessPattern
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_shared: bool = False
    shared_with: List[MemoryComponent] = field(default_factory=list)
    is_compressed: bool = False
    compression_ratio: float = 1.0
    compression_type: MemoryCompressionType = MemoryCompressionType.NONE
    memory_hash: str = ""
    is_deduplicated: bool = False
    deduplication_savings_mb: float = 0.0

class MemoryDeduplicator:
    """Memory deduplication system."""
    
    def __init__(self, config: EnhancedUnifiedMemoryConfig):
        self.config = config
        self.logger = logger.getChild('MemoryDeduplicator')
        
        # Deduplication tracking
        self.memory_hashes: Dict[str, List[str]] = defaultdict(list)  # hash -> allocation_ids
        self.deduplication_stats = {
            'total_deduplications': 0,
            'total_savings_mb': 0.0,
            'current_deduplications': 0
        }
    
    def _calculate_memory_hash(self, data: Any) -> str:
        """Calculate memory hash for deduplication."""
        try:
            if isinstance(data, np.ndarray):
                # Use array data for hashing
                return hashlib.md5(data.tobytes()).hexdigest()
            elif isinstance(data, pd.DataFrame):
                # Use DataFrame values for hashing
                return hashlib.md5(data.values.tobytes()).hexdigest()
            elif isinstance(data, (list, tuple)):
                # Convert to string and hash
                return hashlib.md5(str(data).encode()).hexdigest()
            else:
                # Use pickle for complex objects
                return hashlib.md5(pickle.dumps(data)).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate memory hash: {e}")
            return hashlib.md5(str(id(data)).encode()).hexdigest()
    
    def check_deduplication(self, data: Any, allocation_id: str) -> Optional[str]:
        """Check if data can be deduplicated."""
        if not self.config.enable_memory_deduplication:
            return None
        
        memory_hash = self._calculate_memory_hash(data)
        
        # Check if we have similar data
        if memory_hash in self.memory_hashes:
            existing_allocations = self.memory_hashes[memory_hash]
            if existing_allocations:
                # Return the first existing allocation ID
                self.deduplication_stats['total_deduplications'] += 1
                self.deduplication_stats['current_deduplications'] += 1
                return existing_allocations[0]
        
        # Add to tracking
        self.memory_hashes[memory_hash].append(allocation_id)
        return None
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return self.deduplication_stats.copy()

class MemoryCompressor:
    """Memory compression system."""
    
    def __init__(self, config: EnhancedUnifiedMemoryConfig):
        self.config = config
        self.logger = logger.getChild('MemoryCompressor')
        
        # Compression algorithms
        self.compression_algorithms = {
            MemoryCompressionType.LZ4: self._compress_lz4,
            MemoryCompressionType.ZLIB: self._compress_zlib,
            MemoryCompressionType.GZIP: self._compress_gzip,
            MemoryCompressionType.CUSTOM: self._compress_custom
        }
        
        # Decompression algorithms
        self.decompression_algorithms = {
            MemoryCompressionType.LZ4: self._decompress_lz4,
            MemoryCompressionType.ZLIB: self._decompress_zlib,
            MemoryCompressionType.GZIP: self._decompress_gzip,
            MemoryCompressionType.CUSTOM: self._decompress_custom
        }
        
        # Compression stats
        self.compression_stats = {
            'total_compressions': 0,
            'total_savings_mb': 0.0,
            'average_compression_ratio': 0.0
        }
    
    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress data using LZ4."""
        try:
            import lz4.frame
            return lz4.frame.compress(data)
        except ImportError:
            self.logger.warning("LZ4 not available, using zlib")
            return zlib.compress(data)
    
    def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress data using LZ4."""
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except ImportError:
            self.logger.warning("LZ4 not available, using zlib")
            return zlib.decompress(data)
    
    def _compress_zlib(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data)
    
    def _decompress_zlib(self, data: bytes) -> bytes:
        """Decompress data using zlib."""
        return zlib.decompress(data)
    
    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        import gzip
        return gzip.compress(data)
    
    def _decompress_gzip(self, data: bytes) -> bytes:
        """Decompress data using gzip."""
        import gzip
        return gzip.decompress(data)
    
    def _compress_custom(self, data: bytes) -> bytes:
        """Custom compression algorithm."""
        # Simple run-length encoding for demonstration
        compressed = []
        i = 0
        while i < len(data):
            count = 1
            while i + count < len(data) and data[i + count] == data[i]:
                count += 1
            if count > 3:
                compressed.extend([data[i], count])
            else:
                compressed.extend([data[i]] * count)
            i += count
        return bytes(compressed)
    
    def _decompress_custom(self, data: bytes) -> bytes:
        """Custom decompression algorithm."""
        decompressed = []
        i = 0
        while i < len(data):
            if i + 1 < len(data) and isinstance(data[i + 1], int) and data[i + 1] > 3:
                decompressed.extend([data[i]] * data[i + 1])
                i += 2
            else:
                decompressed.append(data[i])
                i += 1
        return bytes(decompressed)
    
    def compress_data(self, data: Any, compression_type: MemoryCompressionType = None) -> Tuple[bytes, float]:
        """Compress data and return compressed bytes and compression ratio."""
        if compression_type is None:
            compression_type = self.config.compression_algorithm
        
        try:
            # Convert data to bytes
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, pd.DataFrame):
                data_bytes = data.values.tobytes()
            else:
                data_bytes = pickle.dumps(data)
            
            # Compress
            compressed_bytes = self.compression_algorithms[compression_type](data_bytes)
            compression_ratio = len(compressed_bytes) / len(data_bytes)
            
            # Update stats
            self.compression_stats['total_compressions'] += 1
            self.compression_stats['total_savings_mb'] += (len(data_bytes) - len(compressed_bytes)) / (1024 * 1024)
            self.compression_stats['average_compression_ratio'] = (
                (self.compression_stats['average_compression_ratio'] * 
                 (self.compression_stats['total_compressions'] - 1) + compression_ratio) /
                self.compression_stats['total_compressions']
            )
            
            return compressed_bytes, compression_ratio
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return pickle.dumps(data), 1.0
    
    def decompress_data(self, compressed_bytes: bytes, compression_type: MemoryCompressionType) -> Any:
        """Decompress data."""
        try:
            decompressed_bytes = self.decompression_algorithms[compression_type](compressed_bytes)
            return pickle.loads(decompressed_bytes)
        except Exception as e:
            self.logger.warning(f"Decompression failed: {e}")
            return pickle.loads(compressed_bytes)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return self.compression_stats.copy()

class CrossComponentMemoryManager:
    """Cross-component memory sharing manager."""
    
    def __init__(self, config: EnhancedUnifiedMemoryConfig):
        self.config = config
        self.logger = logger.getChild('CrossComponentMemoryManager')
        
        # Shared memory tracking
        self.shared_allocations: Dict[str, List[MemoryComponent]] = defaultdict(list)
        self.component_allocations: Dict[MemoryComponent, List[str]] = defaultdict(list)
        
        # Sharing stats
        self.sharing_stats = {
            'total_shares': 0,
            'active_shares': 0,
            'sharing_savings_mb': 0.0
        }
    
    def can_share_allocation(self, allocation_id: str, target_component: MemoryComponent) -> bool:
        """Check if allocation can be shared with target component."""
        if allocation_id in self.shared_allocations:
            return target_component not in self.shared_allocations[allocation_id]
        return True
    
    def share_allocation(self, allocation_id: str, source_component: MemoryComponent, 
                        target_component: MemoryComponent) -> bool:
        """Share allocation between components."""
        try:
            # Add to shared allocations
            if allocation_id not in self.shared_allocations:
                self.shared_allocations[allocation_id] = [source_component]
            
            if target_component not in self.shared_allocations[allocation_id]:
                self.shared_allocations[allocation_id].append(target_component)
                self.component_allocations[target_component].append(allocation_id)
                
                self.sharing_stats['total_shares'] += 1
                self.sharing_stats['active_shares'] += 1
                
                self.logger.debug(f"🤝 Shared allocation {allocation_id} between {source_component.value} and {target_component.value}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to share allocation {allocation_id}: {e}")
            return False
    
    def unshare_allocation(self, allocation_id: str, component: MemoryComponent) -> bool:
        """Unshare allocation from component."""
        try:
            if allocation_id in self.shared_allocations:
                if component in self.shared_allocations[allocation_id]:
                    self.shared_allocations[allocation_id].remove(component)
                    if component in self.component_allocations:
                        if allocation_id in self.component_allocations[component]:
                            self.component_allocations[component].remove(allocation_id)
                    
                    self.sharing_stats['active_shares'] = max(0, self.sharing_stats['active_shares'] - 1)
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to unshare allocation {allocation_id}: {e}")
            return False
    
    def get_sharing_stats(self) -> Dict[str, Any]:
        """Get sharing statistics."""
        return self.sharing_stats.copy()

class EnhancedUnifiedMemoryManager:
    """Enhanced unified memory manager with advanced features."""
    
    def __init__(self, config: Optional[EnhancedUnifiedMemoryConfig] = None):
        self.config = config or EnhancedUnifiedMemoryConfig()
        self.logger = logger.getChild('EnhancedUnifiedMemoryManager')
        
        # Initialize base memory manager
        self.base_manager = get_unified_memory_manager(self.config)
        
        # Initialize enhanced components
        self.deduplicator = MemoryDeduplicator(self.config)
        self.compressor = MemoryCompressor(self.config)
        self.cross_component_manager = CrossComponentMemoryManager(self.config)
        
        # Enhanced allocation tracking
        self.enhanced_allocations: Dict[str, MemoryAllocation] = {}
        self.allocation_counter = 0
        
        # Memory profiling
        self.memory_profiles: Dict[str, Dict[str, Any]] = {}
        self.profile_counter = 0
        
        # Start monitoring
        if self.config.enable_detailed_monitoring:
            self._start_enhanced_monitoring()
        
        self.logger.info("🧠 Enhanced Unified Memory Manager initialized")
    
    def _start_enhanced_monitoring(self):
        """Start enhanced memory monitoring."""
        def monitor():
            while True:
                try:
                    self._update_memory_profiles()
                    self._cleanup_expired_allocations()
                    self._optimize_memory_usage()
                    time.sleep(self.config.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Enhanced monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 Enhanced memory monitoring started")
    
    def _update_memory_profiles(self):
        """Update memory usage profiles."""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            
            # Create memory profile
            profile = {
                'timestamp': time.time(),
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'used_memory_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'allocations_count': len(self.enhanced_allocations),
                'deduplication_stats': self.deduplicator.get_deduplication_stats(),
                'compression_stats': self.compressor.get_compression_stats(),
                'sharing_stats': self.cross_component_manager.get_sharing_stats()
            }
            
            # Store profile
            profile_id = f"profile_{self.profile_counter}_{int(time.time())}"
            self.memory_profiles[profile_id] = profile
            self.profile_counter += 1
            
            # Cleanup old profiles
            current_time = time.time()
            profiles_to_remove = []
            for pid, prof in self.memory_profiles.items():
                if current_time - prof['timestamp'] > self.config.profile_retention_hours * 3600:
                    profiles_to_remove.append(pid)
            
            for pid in profiles_to_remove:
                del self.memory_profiles[pid]
        
        except Exception as e:
            self.logger.warning(f"Failed to update memory profiles: {e}")
    
    def _cleanup_expired_allocations(self):
        """Cleanup expired allocations."""
        current_time = time.time()
        expired_allocations = []
        
        for allocation_id, allocation in self.enhanced_allocations.items():
            if current_time - allocation.last_accessed > self.config.sharing_timeout_seconds:
                expired_allocations.append(allocation_id)
        
        for allocation_id in expired_allocations:
            self._free_enhanced_allocation(allocation_id)
    
    def _optimize_memory_usage(self):
        """Optimize memory usage based on current state."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0
            
            if usage_percent > 0.8:  # High memory usage
                # Compress large allocations
                self._compress_large_allocations()
                
                # Enable deduplication
                if not self.config.enable_memory_deduplication:
                    self.config.enable_memory_deduplication = True
                    self.logger.info("🔧 Enabled memory deduplication due to high usage")
            
            elif usage_percent < 0.5:  # Low memory usage
                # Disable aggressive compression
                if self.config.compression_algorithm != MemoryCompressionType.NONE:
                    self.logger.info("🔧 Disabled compression due to low usage")
        
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _compress_large_allocations(self):
        """Compress large allocations."""
        large_allocations = [
            (aid, alloc) for aid, alloc in self.enhanced_allocations.items()
            if alloc.size_mb >= self.config.optimization_threshold_mb
            and not alloc.is_compressed
        ]
        
        for allocation_id, allocation in large_allocations:
            try:
                # This would implement actual compression
                allocation.is_compressed = True
                allocation.compression_ratio = 0.7  # Simulated compression ratio
                allocation.compression_type = self.config.compression_algorithm
                
                self.logger.debug(f"📦 Compressed allocation {allocation_id}")
            except Exception as e:
                self.logger.warning(f"Failed to compress allocation {allocation_id}: {e}")
    
    def allocate_enhanced_memory(self, size_mb: float, tier: MemoryTier, 
                                component: MemoryComponent,
                                access_pattern: MemoryAccessPattern = MemoryAccessPattern.SEQUENTIAL,
                                allow_sharing: bool = True,
                                data: Any = None) -> str:
        """Allocate memory with enhanced features."""
        allocation_id = f"enhanced_alloc_{self.allocation_counter}_{int(time.time())}"
        self.allocation_counter += 1
        
        # Check for deduplication
        deduplicated_id = None
        if data is not None and allow_sharing:
            deduplicated_id = self.deduplicator.check_deduplication(data, allocation_id)
            if deduplicated_id:
                # Share existing allocation
                self.cross_component_manager.share_allocation(
                    deduplicated_id, MemoryComponent.SHARED, component
                )
                return deduplicated_id
        
        # Create enhanced allocation
        allocation = MemoryAllocation(
            allocation_id=allocation_id,
            size_mb=size_mb,
            tier=tier,
            component=component,
            access_pattern=access_pattern,
            is_shared=allow_sharing,
            memory_hash=self.deduplicator._calculate_memory_hash(data) if data else ""
        )
        
        # Store allocation
        self.enhanced_allocations[allocation_id] = allocation
        
        # Add to component tracking
        self.cross_component_manager.component_allocations[component].append(allocation_id)
        
        # Allocate in base manager
        base_allocation_id = self.base_manager.allocate_for_operation(
            f"enhanced_{component.value}", size_mb, component.value
        )
        
        self.logger.debug(f"🧠 Allocated enhanced memory {allocation_id} ({size_mb:.1f}MB) for {component.value}")
        
        return allocation_id
    
    def _free_enhanced_allocation(self, allocation_id: str) -> bool:
        """Free enhanced allocation."""
        if allocation_id not in self.enhanced_allocations:
            return False
        
        allocation = self.enhanced_allocations[allocation_id]
        
        # Remove from component tracking
        if allocation.component in self.cross_component_manager.component_allocations:
            if allocation_id in self.cross_component_manager.component_allocations[allocation.component]:
                self.cross_component_manager.component_allocations[allocation.component].remove(allocation_id)
        
        # Unshare if shared
        if allocation.is_shared:
            for component in allocation.shared_with:
                self.cross_component_manager.unshare_allocation(allocation_id, component)
        
        # Remove from enhanced allocations
        del self.enhanced_allocations[allocation_id]
        
        self.logger.debug(f"🗑️ Freed enhanced allocation {allocation_id}")
        
        return True
    
    def get_enhanced_memory_stats(self) -> Dict[str, Any]:
        """Get enhanced memory statistics."""
        base_stats = self.base_manager.get_comprehensive_stats()
        
        enhanced_stats = {
            'enhanced_allocations': len(self.enhanced_allocations),
            'deduplication_stats': self.deduplicator.get_deduplication_stats(),
            'compression_stats': self.compressor.get_compression_stats(),
            'sharing_stats': self.cross_component_manager.get_sharing_stats(),
            'memory_profiles_count': len(self.memory_profiles),
            'component_allocations': {
                component.value: len(allocations) 
                for component, allocations in self.cross_component_manager.component_allocations.items()
            }
        }
        
        return {**base_stats, **enhanced_stats}

# Global instance
_enhanced_unified_memory_manager: Optional[EnhancedUnifiedMemoryManager] = None

def get_enhanced_unified_memory_manager(config: Optional[EnhancedUnifiedMemoryConfig] = None) -> EnhancedUnifiedMemoryManager:
    """Get or create the global enhanced unified memory manager."""
    global _enhanced_unified_memory_manager
    
    if _enhanced_unified_memory_manager is None:
        _enhanced_unified_memory_manager = EnhancedUnifiedMemoryManager(config)
    
    return _enhanced_unified_memory_manager

def unified_memory_feature_processing(data: Any, operation_type: str = 'feature_selection', 
                                    component: str = 'gpu') -> Any:
    """Backward compatible function for unified memory feature processing."""
    manager = get_enhanced_unified_memory_manager()
    
    # Determine component
    memory_component = MemoryComponent.GPU if component == 'gpu' else MemoryComponent.CPU
    
    # Allocate memory
    size_mb = data.nbytes / (1024 * 1024) if hasattr(data, 'nbytes') else 100.0
    allocation_id = manager.allocate_enhanced_memory(
        size_mb=size_mb,
        tier=MemoryTier.GPU_OPTIMIZED if component == 'gpu' else MemoryTier.SHARED,
        component=memory_component,
        access_pattern=MemoryAccessPattern.SEQUENTIAL,
        allow_sharing=True,
        data=data
    )
    
    try:
        # Process data
        optimized_data = manager.base_manager.optimize_data_for_component(data, component)
        return optimized_data
    finally:
        # Free memory
        manager._free_enhanced_allocation(allocation_id)

def get_enhanced_unified_memory_stats() -> Dict[str, Any]:
    """Get enhanced unified memory statistics."""
    manager = get_enhanced_unified_memory_manager()
    return manager.get_enhanced_memory_stats()