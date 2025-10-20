"""
M1 Unified Memory Manager for Apple Silicon.

This module provides comprehensive memory management specifically optimized for
M1/M2/M3/M4 unified memory architecture, including intelligent memory pooling,
compression, and cross-component memory sharing.
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

# Optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_performance, LogLevel
)

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Memory tiers for unified memory architecture."""
    CPU_ONLY = "cpu_only"
    SHARED = "shared"
    GPU_OPTIMIZED = "gpu_optimized"
    NEURAL_ENGINE = "neural_engine"
    COMPRESSED = "compressed"

class MemoryAllocationStrategy(Enum):
    """Memory allocation strategies for unified memory."""
    UNIFIED_OPTIMIZED = "unified_optimized"
    TIER_BASED = "tier_based"
    WORKLOAD_AWARE = "workload_aware"
    ADAPTIVE = "adaptive"

@dataclass
class UnifiedMemoryConfig:
    """Configuration for unified memory management."""
    # Memory limits
    total_memory_gb: float = 16.0
    cpu_memory_limit_gb: float = 8.0
    gpu_memory_limit_gb: float = 8.0
    neural_engine_memory_limit_gb: float = 4.0
    
    # Memory compression
    enable_compression: bool = True
    compression_threshold_mb: float = 100.0
    compression_ratio: float = 0.7
    max_compression_ratio: float = 0.5
    
    # Memory pooling
    enable_memory_pools: bool = True
    pool_size_mb: float = 512.0
    pool_growth_factor: float = 1.5
    pool_cleanup_interval: float = 300.0
    
    # Cross-component sharing
    enable_cross_component_sharing: bool = True
    sharing_threshold_mb: float = 50.0
    sharing_timeout_seconds: float = 30.0
    
    # Memory pressure management
    enable_pressure_management: bool = True
    pressure_check_interval: float = 2.0
    low_pressure_threshold: float = 0.6
    high_pressure_threshold: float = 0.85
    critical_pressure_threshold: float = 0.95
    
    # Garbage collection
    enable_aggressive_gc: bool = True
    gc_threshold_mb: float = 200.0
    gc_interval_seconds: float = 30.0
    
    # Memory monitoring
    enable_detailed_monitoring: bool = True
    monitoring_interval: float = 5.0
    enable_memory_tracing: bool = False

@dataclass
class MemoryAllocation:
    """Represents a memory allocation."""
    allocation_id: str
    size_mb: float
    tier: MemoryTier
    component: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_shared: bool = False
    shared_with: List[str] = field(default_factory=list)
    is_compressed: bool = False
    compression_ratio: float = 1.0

class UnifiedMemoryPool:
    """Memory pool for unified memory architecture."""
    
    def __init__(self, config: UnifiedMemoryConfig):
        self.config = config
        self.logger = logger.getChild('UnifiedMemoryPool')
        
        # Memory pools by tier
        self.pools: Dict[MemoryTier, Dict[str, Any]] = {
            tier: {} for tier in MemoryTier
        }
        
        # Allocation tracking
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.allocation_counter = 0
        
        # Memory statistics
        self.stats = {
            'total_allocated_mb': 0.0,
            'total_freed_mb': 0.0,
            'current_usage_mb': 0.0,
            'peak_usage_mb': 0.0,
            'compression_savings_mb': 0.0,
            'cross_component_shares': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Memory pressure monitoring
        self.pressure_level = MemoryPressureLevel.LOW
        self.pressure_history = deque(maxlen=100)
        
        # Initialize memory pools
        self._initialize_memory_pools()
        
        # Start monitoring thread
        if self.config.enable_detailed_monitoring:
            self._start_monitoring()
    
    def _initialize_memory_pools(self):
        """Initialize memory pools for each tier."""
        for tier in MemoryTier:
            self.pools[tier] = {
                'available': [],
                'allocated': {},
                'compressed': {},
                'shared': {}
            }
        
        self.logger.info("🧠 Unified memory pools initialized")
    
    def _start_monitoring(self):
        """Start memory monitoring thread."""
        def monitor():
            while True:
                try:
                    self._update_memory_stats()
                    self._check_memory_pressure()
                    self._cleanup_expired_allocations()
                    time.sleep(self.config.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        self.logger.info("📊 Memory monitoring started")
    
    def _update_memory_stats(self):
        """Update memory statistics."""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            
            # Calculate current usage
            current_usage = sum(alloc.size_mb for alloc in self.allocations.values())
            self.stats['current_usage_mb'] = current_usage
            
            # Update peak usage
            if current_usage > self.stats['peak_usage_mb']:
                self.stats['peak_usage_mb'] = current_usage
            
            # Update pressure level
            usage_percent = memory.percent / 100.0
            self.pressure_history.append(usage_percent)
            
        except Exception as e:
            self.logger.warning(f"Failed to update memory stats: {e}")
    
    def _check_memory_pressure(self):
        """Check memory pressure and adjust accordingly."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100.0
            
            # Determine pressure level
            if usage_percent >= self.config.critical_pressure_threshold:
                self.pressure_level = MemoryPressureLevel.CRITICAL
            elif usage_percent >= self.config.high_pressure_threshold:
                self.pressure_level = MemoryPressureLevel.HIGH
            elif usage_percent >= self.config.low_pressure_threshold:
                self.pressure_level = MemoryPressureLevel.MEDIUM
            else:
                self.pressure_level = MemoryPressureLevel.LOW
            
            # Take action based on pressure level
            if self.pressure_level == MemoryPressureLevel.CRITICAL:
                self._handle_critical_pressure()
            elif self.pressure_level == MemoryPressureLevel.HIGH:
                self._handle_high_pressure()
            elif self.pressure_level == MemoryPressureLevel.MEDIUM:
                self._handle_medium_pressure()
                
        except Exception as e:
            self.logger.warning(f"Failed to check memory pressure: {e}")
    
    def _handle_critical_pressure(self):
        """Handle critical memory pressure."""
        self.logger.warning("🚨 Critical memory pressure detected")
        
        # Force garbage collection
        if self.config.enable_aggressive_gc:
            collected = gc.collect()
            self.logger.info(f"🧹 Collected {collected} objects")
        
        # Clear least recently used allocations
        self._clear_lru_allocations(0.3)  # Clear 30% of allocations
        
        # Compress large allocations
        self._compress_large_allocations()
    
    def _handle_high_pressure(self):
        """Handle high memory pressure."""
        self.logger.info("⚠️ High memory pressure detected")
        
        # Clear some LRU allocations
        self._clear_lru_allocations(0.1)  # Clear 10% of allocations
        
        # Compress medium allocations
        self._compress_medium_allocations()
    
    def _handle_medium_pressure(self):
        """Handle medium memory pressure."""
        # Light cleanup
        self._clear_expired_allocations()
    
    def _clear_lru_allocations(self, fraction: float):
        """Clear least recently used allocations."""
        if not self.allocations:
            return
        
        # Sort by last accessed time
        sorted_allocations = sorted(
            self.allocations.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Clear fraction of allocations
        num_to_clear = int(len(sorted_allocations) * fraction)
        
        for allocation_id, allocation in sorted_allocations[:num_to_clear]:
            self._free_allocation(allocation_id)
    
    def _compress_large_allocations(self):
        """Compress large allocations."""
        if not self.config.enable_compression:
            return
        
        large_allocations = [
            (aid, alloc) for aid, alloc in self.allocations.items()
            if alloc.size_mb >= self.config.compression_threshold_mb
            and not alloc.is_compressed
        ]
        
        for allocation_id, allocation in large_allocations:
            if self._compress_allocation(allocation_id):
                self.logger.info(f"📦 Compressed allocation {allocation_id} ({allocation.size_mb:.1f}MB)")
    
    def _compress_medium_allocations(self):
        """Compress medium allocations."""
        if not self.config.enable_compression:
            return
        
        medium_allocations = [
            (aid, alloc) for aid, alloc in self.allocations.items()
            if 50 <= alloc.size_mb < self.config.compression_threshold_mb
            and not alloc.is_compressed
        ]
        
        for allocation_id, allocation in medium_allocations:
            if self._compress_allocation(allocation_id):
                self.logger.debug(f"📦 Compressed medium allocation {allocation_id}")
    
    def _compress_allocation(self, allocation_id: str) -> bool:
        """Compress a specific allocation."""
        try:
            allocation = self.allocations[allocation_id]
            
            # Simulate compression (in real implementation, would use actual compression)
            compression_ratio = self.config.compression_ratio
            compressed_size = allocation.size_mb * compression_ratio
            
            # Update allocation
            allocation.is_compressed = True
            allocation.compression_ratio = compression_ratio
            allocation.size_mb = compressed_size
            
            # Update stats
            self.stats['compression_savings_mb'] += allocation.size_mb * (1 - compression_ratio)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to compress allocation {allocation_id}: {e}")
            return False
    
    def _clear_expired_allocations(self):
        """Clear expired allocations."""
        current_time = time.time()
        expired_allocations = []
        
        for allocation_id, allocation in self.allocations.items():
            if current_time - allocation.last_accessed > self.config.sharing_timeout_seconds:
                expired_allocations.append(allocation_id)
        
        for allocation_id in expired_allocations:
            self._free_allocation(allocation_id)
    
    def _cleanup_expired_allocations(self):
        """Cleanup expired allocations."""
        self._clear_expired_allocations()
    
    def allocate_memory(self, size_mb: float, tier: MemoryTier, 
                       component: str, allow_sharing: bool = True) -> str:
        """Allocate memory in the unified memory architecture."""
        allocation_id = f"alloc_{self.allocation_counter}_{int(time.time())}"
        self.allocation_counter += 1
        
        # Check if we can share existing allocation
        if allow_sharing and self.config.enable_cross_component_sharing:
            shared_allocation = self._find_shareable_allocation(size_mb, tier, component)
            if shared_allocation:
                return self._share_allocation(shared_allocation, component)
        
        # Create new allocation
        allocation = MemoryAllocation(
            allocation_id=allocation_id,
            size_mb=size_mb,
            tier=tier,
            component=component,
            is_shared=allow_sharing
        )
        
        # Check memory pressure
        if self.pressure_level == MemoryPressureLevel.CRITICAL:
            # Try to compress or clear space
            self._handle_critical_pressure()
        
        # Allocate memory
        self.allocations[allocation_id] = allocation
        self.stats['total_allocated_mb'] += size_mb
        self.stats['current_usage_mb'] += size_mb
        
        # Update pool
        self.pools[tier]['allocated'][allocation_id] = allocation
        
        self.logger.debug(f"🧠 Allocated {size_mb:.1f}MB for {component} in {tier.value} tier")
        
        return allocation_id
    
    def _find_shareable_allocation(self, size_mb: float, tier: MemoryTier, 
                                  component: str) -> Optional[str]:
        """Find a shareable allocation."""
        for allocation_id, allocation in self.allocations.items():
            if (allocation.tier == tier and 
                allocation.size_mb >= size_mb and 
                allocation.is_shared and
                component not in allocation.shared_with):
                return allocation_id
        return None
    
    def _share_allocation(self, allocation_id: str, component: str) -> str:
        """Share an existing allocation with a new component."""
        allocation = self.allocations[allocation_id]
        allocation.shared_with.append(component)
        allocation.access_count += 1
        allocation.last_accessed = time.time()
        
        self.stats['cross_component_shares'] += 1
        
        self.logger.debug(f"🤝 Shared allocation {allocation_id} with {component}")
        
        return allocation_id
    
    def free_memory(self, allocation_id: str) -> bool:
        """Free a memory allocation."""
        return self._free_allocation(allocation_id)
    
    def _free_allocation(self, allocation_id: str) -> bool:
        """Internal method to free an allocation."""
        if allocation_id not in self.allocations:
            return False
        
        allocation = self.allocations[allocation_id]
        
        # Update stats
        self.stats['total_freed_mb'] += allocation.size_mb
        self.stats['current_usage_mb'] -= allocation.size_mb
        
        # Remove from pools
        if allocation_id in self.pools[allocation.tier]['allocated']:
            del self.pools[allocation.tier]['allocated'][allocation_id]
        
        # Remove from allocations
        del self.allocations[allocation_id]
        
        self.logger.debug(f"🗑️ Freed allocation {allocation_id} ({allocation.size_mb:.1f}MB)")
        
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            'allocations': len(self.allocations),
            'total_allocated_mb': self.stats['total_allocated_mb'],
            'current_usage_mb': self.stats['current_usage_mb'],
            'peak_usage_mb': self.stats['peak_usage_mb'],
            'compression_savings_mb': self.stats['compression_savings_mb'],
            'cross_component_shares': self.stats['cross_component_shares'],
            'pressure_level': self.pressure_level.value,
            'pools': {
                tier.value: {
                    'allocated': len(self.pools[tier]['allocated']),
                    'available': len(self.pools[tier]['available']),
                    'compressed': len(self.pools[tier]['compressed']),
                    'shared': len(self.pools[tier]['shared'])
                }
                for tier in MemoryTier
            }
        }

class M1UnifiedMemoryManager:
    """Main manager for M1 unified memory architecture."""
    
    def __init__(self, config: Optional[UnifiedMemoryConfig] = None):
        self.config = config or UnifiedMemoryConfig()
        self.logger = logger.getChild('M1UnifiedMemoryManager')
        
        # Initialize memory pool
        self.memory_pool = UnifiedMemoryPool(self.config)
        
        # Component-specific optimizations
        self.component_optimizations = {
            'cpu': self._optimize_cpu_memory,
            'gpu': self._optimize_gpu_memory,
            'neural_engine': self._optimize_neural_engine_memory
        }
        
        # Memory tier mappings
        self.tier_mappings = {
            'matrix_operations': MemoryTier.GPU_OPTIMIZED,
            'neural_networks': MemoryTier.NEURAL_ENGINE,
            'data_processing': MemoryTier.SHARED,
            'caching': MemoryTier.COMPRESSED,
            'general': MemoryTier.CPU_ONLY
        }
        
        self.logger.info("🧠 M1 Unified Memory Manager initialized")
    
    def _optimize_cpu_memory(self, data: Any) -> Any:
        """Optimize memory for CPU operations."""
        # Convert to appropriate dtypes for CPU efficiency
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64:
                return data.astype(np.float32)
            elif data.dtype == np.int64:
                return data.astype(np.int32)
        elif isinstance(data, pd.DataFrame):
            # Optimize DataFrame for CPU
            for col in data.select_dtypes(include=['object']):
                if data[col].nunique() / len(data) < 0.5:
                    data[col] = data[col].astype('category')
        return data
    
    def _optimize_gpu_memory(self, data: Any) -> Any:
        """Optimize memory for GPU operations."""
        if TORCH_AVAILABLE and isinstance(data, np.ndarray):
            # Convert to PyTorch tensor for GPU
            return torch.from_numpy(data).float()
        return data
    
    def _optimize_neural_engine_memory(self, data: Any) -> Any:
        """Optimize memory for Neural Engine."""
        # Neural Engine prefers specific data layouts
        if isinstance(data, np.ndarray):
            # Ensure contiguous memory layout
            return np.ascontiguousarray(data)
        return data
    
    def allocate_for_operation(self, operation_type: str, size_mb: float, 
                              component: str = 'general') -> str:
        """Allocate memory for a specific operation type."""
        tier = self.tier_mappings.get(operation_type, MemoryTier.SHARED)
        return self.memory_pool.allocate_memory(size_mb, tier, component)
    
    def optimize_data_for_component(self, data: Any, component: str) -> Any:
        """Optimize data for a specific component."""
        if component in self.component_optimizations:
            return self.component_optimizations[component](data)
        return data
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        memory_stats = self.memory_pool.get_memory_stats()
        
        # Add system memory info
        try:
            system_memory = psutil.virtual_memory()
            memory_stats['system'] = {
                'total_gb': system_memory.total / (1024**3),
                'available_gb': system_memory.available / (1024**3),
                'used_percent': system_memory.percent,
                'pressure_level': self.memory_pool.pressure_level.value
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system memory info: {e}")
            memory_stats['system'] = {}
        
        return memory_stats
    
    def cleanup_all(self):
        """Cleanup all memory allocations."""
        self.memory_pool.allocations.clear()
        self.memory_pool.stats['current_usage_mb'] = 0.0
        
        # Force garbage collection
        if self.config.enable_aggressive_gc:
            gc.collect()
        
        self.logger.info("🧹 All memory allocations cleaned up")

# Global instance
_unified_memory_manager: Optional[M1UnifiedMemoryManager] = None

def get_unified_memory_manager(config: Optional[UnifiedMemoryConfig] = None) -> M1UnifiedMemoryManager:
    """Get or create the global unified memory manager."""
    global _unified_memory_manager
    
    if _unified_memory_manager is None:
        _unified_memory_manager = M1UnifiedMemoryManager(config)
    
    return _unified_memory_manager

def optimize_for_unified_memory(data: Any, operation_type: str = 'general', 
                               component: str = 'general') -> Any:
    """Optimize data for unified memory architecture."""
    manager = get_unified_memory_manager()
    return manager.optimize_data_for_component(data, component)

def allocate_unified_memory(size_mb: float, operation_type: str = 'general', 
                           component: str = 'general') -> str:
    """Allocate memory in unified memory architecture."""
    manager = get_unified_memory_manager()
    return manager.allocate_for_operation(operation_type, size_mb, component)

def get_unified_memory_stats() -> Dict[str, Any]:
    """Get unified memory statistics."""
    manager = get_unified_memory_manager()
    return manager.get_comprehensive_stats()

# Decorators for easy integration
def unified_memory_optimized(operation_type: str = 'general', component: str = 'general'):
    """Decorator to optimize function for unified memory."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optimize inputs
            optimized_args = []
            for arg in args:
                if hasattr(arg, 'shape') or isinstance(arg, (np.ndarray, pd.DataFrame)):
                    optimized_args.append(optimize_for_unified_memory(arg, operation_type, component))
                else:
                    optimized_args.append(arg)
            
            # Execute function
            result = func(*optimized_args, **kwargs)
            
            # Optimize output
            if hasattr(result, 'shape') or isinstance(result, (np.ndarray, pd.DataFrame)):
                result = optimize_for_unified_memory(result, operation_type, component)
            
            return result
        return wrapper
    return decorator

def memory_tier_aware(tier: MemoryTier):
    """Decorator to make function memory tier aware."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Allocate memory for this tier
            manager = get_unified_memory_manager()
            allocation_id = manager.allocate_for_operation(
                func.__name__, 100.0, 'decorated_function'  # Estimate 100MB
            )
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Free memory
                manager.memory_pool.free_memory(allocation_id)
        
        return wrapper
    return decorator