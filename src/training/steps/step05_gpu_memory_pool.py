"""
Step05 GPU Memory Pool Management Module

This module provides advanced GPU memory pool management for Step05 processing,
implementing intelligent memory allocation, reuse patterns, and defragmentation
to optimize GPU memory utilization for M1/M2/M3 chips.
"""

import torch
import numpy as np
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import gc
import weakref

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, validates

logger = system_logger.getChild('GPUMemoryPool')


@dataclass
class MemoryBlock:
    """Represents a block of GPU memory."""
    tensor: Optional[torch.Tensor] = None
    size_bytes: int = 0
    allocated_time: datetime = field(default_factory=datetime.now)
    last_access_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    operation_type: str = "general"
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    is_pinned: bool = False
    memory_pool_id: str = "default"


@dataclass
class MemoryPoolConfig:
    """Configuration for GPU memory pool management."""
    max_pool_size_gb: float = 4.0
    min_block_size_mb: float = 1.0
    max_block_size_mb: float = 512.0
    fragmentation_threshold: float = 0.3  # Defragment when >30% fragmented
    reuse_threshold_mb: float = 10.0  # Minimum size for reuse consideration
    cleanup_interval_seconds: int = 60
    enable_memory_tracking: bool = True
    enable_automatic_defragmentation: bool = True
    enable_memory_reuse: bool = True
    enable_pin_memory: bool = True
    memory_pressure_threshold: float = 0.8


@dataclass
class MemoryPoolStats:
    """Statistics for memory pool performance."""
    total_allocated_bytes: int = 0
    total_freed_bytes: int = 0
    peak_memory_usage_bytes: int = 0
    current_memory_usage_bytes: int = 0
    fragmentation_ratio: float = 0.0
    reuse_count: int = 0
    defragmentation_count: int = 0
    cache_hit_rate: float = 0.0
    memory_pressure_events: int = 0
    last_cleanup_time: datetime = field(default_factory=datetime.now)


class GPUMemoryPoolManager:
    """
    Advanced GPU memory pool manager with intelligent allocation,
    reuse patterns, and automatic defragmentation.
    """

    def __init__(self, config: Optional[MemoryPoolConfig] = None):
        self.config = config or MemoryPoolConfig()
        self.logger = logger

        # Memory pools by device and operation type
        self.memory_pools: Dict[str, Dict[str, List[MemoryBlock]]] = defaultdict(lambda: defaultdict(list))
        self.free_blocks: Dict[str, List[MemoryBlock]] = defaultdict(list)
        self.allocated_blocks: Dict[str, Dict[int, MemoryBlock]] = defaultdict(dict)

        # Statistics and tracking
        self.stats = MemoryPoolStats()
        self.allocation_history: deque[Tuple[str, int, datetime]] = deque(maxlen=1000)

        # Thread safety
        self._lock = threading.RLock()

        # GPU device detection
        self.device = self._detect_device()
        self.has_gpu = self.device.type in ['mps', 'cuda']

        # Memory tracking
        self.memory_tracker = {}
        self.cleanup_thread = None

        if self.has_gpu:
            self._start_cleanup_thread()

        self.logger.info("🚀 Initializing GPU Memory Pool Manager")
        self.logger.info(f"🎮 Device: {self.device}")
        self.logger.info(f"💾 Max pool size: {self.config.max_pool_size_gb:.1f}GB")
        self.logger.info(f"🔄 Memory reuse: {'Enabled' if self.config.enable_memory_reuse else 'Disabled'}")
        self.logger.info(f"🧹 Auto-defragmentation: {'Enabled' if self.config.enable_automatic_defragmentation else 'Disabled'}")

    def _detect_device(self) -> torch.device:
        """Detect the best available GPU device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self.config.enable_memory_tracking:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name="GPU-Memory-Cleanup"
            )
            self.cleanup_thread.start()

    def _cleanup_worker(self) -> None:
        """Background cleanup worker thread."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval_seconds)
                self._perform_cleanup()
            except Exception as e:
                self.logger.warning(f"⚠️ Cleanup worker error: {e}")

    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       operation_type: str = "general", pin_memory: bool = False) -> torch.Tensor:
        """
        Allocate a tensor with intelligent memory pool management.

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            operation_type: Type of operation for memory optimization
            pin_memory: Whether to pin memory for faster transfers

        Returns:
            Allocated tensor
        """
        if not self.has_gpu:
            # Fallback to regular allocation
            return torch.zeros(shape, dtype=dtype, device=self.device)

        with self._lock:
            try:
                # Calculate memory requirements
                size_bytes = self._calculate_tensor_size_bytes(shape, dtype)
                pool_id = f"{self.device.type}_{operation_type}"

                # Check memory pressure
                if self._is_memory_pressure_high():
                    self._handle_memory_pressure()

                # Try to reuse existing block
                if self.config.enable_memory_reuse:
                    reused_tensor = self._try_reuse_block(pool_id, size_bytes, shape, dtype)
                    if reused_tensor is not None:
                        self.stats.reuse_count += 1
                        return reused_tensor

                # Allocate new tensor
                tensor = self._allocate_new_tensor(shape, dtype, pin_memory)

                if tensor is not None:
                    # Track the allocation
                    self._track_allocation(tensor, pool_id, size_bytes, operation_type, shape, dtype)

                return tensor

            except Exception as e:
                self.logger.error(f"❌ Tensor allocation failed: {e}")
                # Fallback allocation
                return torch.zeros(shape, dtype=dtype, device=self.device)

    def _try_reuse_block(self, pool_id: str, required_size: int,
                        shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Try to reuse an existing memory block."""
        try:
            free_blocks = self.free_blocks[pool_id]

            # Find suitable block (best fit strategy)
            best_fit = None
            best_fit_idx = -1

            for idx, block in enumerate(free_blocks):
                if (block.size_bytes >= required_size and
                    block.size_bytes <= required_size * 1.5 and  # Allow 50% overhead
                    block.dtype == dtype):
                    if best_fit is None or block.size_bytes < best_fit.size_bytes:
                        best_fit = block
                        best_fit_idx = idx

            if best_fit is not None and best_fit.tensor is not None:
                # Reuse the block
                tensor = best_fit.tensor

                # Resize if necessary (within reasonable bounds)
                if tensor.shape != shape and tensor.numel() >= np.prod(shape):
                    try:
                        tensor = tensor.view(shape)
                    except RuntimeError:
                        # Can't reshape, create new tensor
                        return None

                # Update tracking
                best_fit.last_access_time = datetime.now()
                best_fit.access_count += 1
                best_fit.shape = shape

                # Move from free to allocated
                allocated_id = id(tensor)
                self.allocated_blocks[pool_id][allocated_id] = best_fit
                free_blocks.pop(best_fit_idx)

                # Update tensor reference
                tensor._memory_block = best_fit

                self.logger.debug(f"♻️ Reused memory block: {best_fit.size_bytes} bytes for {shape}")
                return tensor

        except Exception as e:
            self.logger.warning(f"⚠️ Memory reuse failed: {e}")

        return None

    def _allocate_new_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype,
                           pin_memory: bool = False) -> Optional[torch.Tensor]:
        """Allocate a new tensor."""
        try:
            # Check if we have enough memory
            size_bytes = self._calculate_tensor_size_bytes(shape, dtype)
            if not self._can_allocate(size_bytes):
                self.logger.warning(f"⚠️ Cannot allocate {size_bytes} bytes, attempting cleanup")
                self._perform_cleanup()
                if not self._can_allocate(size_bytes):
                    return None

            # Allocate tensor
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)

            if pin_memory and self.config.enable_pin_memory:
                try:
                    tensor = tensor.pin_memory()
                except RuntimeError:
                    pass  # Pinning not supported

            return tensor

        except RuntimeError as e:
            self.logger.warning(f"⚠️ Tensor allocation failed: {e}")
            return None

    def _track_allocation(self, tensor: torch.Tensor, pool_id: str, size_bytes: int,
                         operation_type: str, shape: Tuple[int, ...], dtype: torch.dtype) -> None:
        """Track a new memory allocation."""
        try:
            # Create memory block
            block = MemoryBlock(
                tensor=tensor,
                size_bytes=size_bytes,
                operation_type=operation_type,
                shape=shape,
                dtype=dtype,
                memory_pool_id=pool_id
            )

            # Add to allocated blocks
            tensor_id = id(tensor)
            self.allocated_blocks[pool_id][tensor_id] = block

            # Update statistics
            self.stats.total_allocated_bytes += size_bytes
            self.stats.current_memory_usage_bytes += size_bytes
            self.stats.peak_memory_usage_bytes = max(
                self.stats.peak_memory_usage_bytes,
                self.stats.current_memory_usage_bytes
            )

            # Add to allocation history
            self.allocation_history.append((pool_id, size_bytes, datetime.now()))

            # Set up cleanup callback
            weakref.ref(tensor, lambda ref: self._tensor_cleanup_callback(pool_id, tensor_id))

            # Store reference to block in tensor (for reuse tracking)
            tensor._memory_block = block

        except Exception as e:
            self.logger.warning(f"⚠️ Allocation tracking failed: {e}")

    def free_tensor(self, tensor: torch.Tensor) -> None:
        """
        Free a tensor and return its memory to the pool for reuse.

        Args:
            tensor: Tensor to free
        """
        if not self.has_gpu or tensor.device != self.device:
            return

        with self._lock:
            try:
                tensor_id = id(tensor)
                pool_id = None

                # Find the tensor in allocated blocks
                for pid, blocks in self.allocated_blocks.items():
                    if tensor_id in blocks:
                        pool_id = pid
                        block = blocks[tensor_id]
                        break

                if pool_id and block:
                    # Move to free blocks for reuse
                    if self.config.enable_memory_reuse:
                        self.free_blocks[pool_id].append(block)

                    # Update statistics
                    self.stats.total_freed_bytes += block.size_bytes
                    self.stats.current_memory_usage_bytes -= block.size_bytes

                    # Remove from allocated
                    del self.allocated_blocks[pool_id][tensor_id]

                    self.logger.debug(f"🗑️ Freed tensor: {block.size_bytes} bytes from pool {pool_id}")

                    # Clear tensor reference
                    if hasattr(tensor, '_memory_block'):
                        tensor._memory_block.tensor = None

            except Exception as e:
                self.logger.warning(f"⚠️ Tensor free failed: {e}")

    def _tensor_cleanup_callback(self, pool_id: str, tensor_id: int) -> None:
        """Callback when a tensor is garbage collected."""
        try:
            with self._lock:
                if pool_id in self.allocated_blocks and tensor_id in self.allocated_blocks[pool_id]:
                    block = self.allocated_blocks[pool_id][tensor_id]
                    self.stats.current_memory_usage_bytes -= block.size_bytes
                    del self.allocated_blocks[pool_id][tensor_id]
        except Exception:
            pass  # Ignore cleanup errors

    def _calculate_tensor_size_bytes(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Calculate the size of a tensor in bytes."""
        try:
            element_size = torch.tensor([], dtype=dtype).element_size()
            total_elements = np.prod(shape)
            return int(total_elements * element_size)
        except Exception:
            # Conservative estimate
            return int(np.prod(shape) * 4)  # Assume 4 bytes per element

    def _can_allocate(self, size_bytes: int) -> bool:
        """Check if we can allocate the requested memory."""
        try:
            # Check pool limits
            max_pool_bytes = int(self.config.max_pool_size_gb * 1024**3)
            if self.stats.current_memory_usage_bytes + size_bytes > max_pool_bytes:
                return False

            # Check system memory
            memory_info = psutil.virtual_memory()
            available_bytes = memory_info.available

            # Conservative check (leave some headroom)
            return size_bytes < available_bytes * 0.8

        except Exception:
            return True  # Allow allocation if check fails

    def _is_memory_pressure_high(self) -> bool:
        """Check if memory pressure is high."""
        try:
            current_usage_gb = self.stats.current_memory_usage_bytes / (1024**3)
            max_pool_gb = self.config.max_pool_size_gb

            pressure_ratio = current_usage_gb / max_pool_gb
            return pressure_ratio > self.config.memory_pressure_threshold

        except Exception:
            return False

    def _handle_memory_pressure(self) -> None:
        """Handle high memory pressure situations."""
        try:
            self.logger.info("⚠️ Memory pressure detected, performing emergency cleanup")

            # Force garbage collection
            collected = gc.collect()

            # Clear GPU cache
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Force defragmentation
            self._perform_defragmentation()

            # Update stats
            self.stats.memory_pressure_events += 1

            self.logger.info(f"🧹 Emergency cleanup completed: {collected} objects collected")

        except Exception as e:
            self.logger.warning(f"⚠️ Memory pressure handling failed: {e}")

    def _perform_cleanup(self) -> None:
        """Perform periodic cleanup and maintenance."""
        try:
            # Remove old free blocks
            cutoff_time = datetime.now() - timedelta(minutes=5)

            for pool_id, blocks in self.free_blocks.items():
                # Keep only recent blocks
                self.free_blocks[pool_id] = [
                    block for block in blocks
                    if block.last_access_time > cutoff_time
                ]

            # Check for defragmentation
            if self.config.enable_automatic_defragmentation:
                self._check_defragmentation()

            self.stats.last_cleanup_time = datetime.now()

        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")

    def _check_defragmentation(self) -> None:
        """Check if defragmentation is needed and perform it."""
        try:
            # Calculate fragmentation ratio
            total_allocated = self.stats.total_allocated_bytes
            if total_allocated == 0:
                return

            # Estimate fragmentation based on free block distribution
            total_free = sum(block.size_bytes for blocks in self.free_blocks.values()
                           for block in blocks)

            if total_free > 0:
                # Simple fragmentation metric
                avg_free_size = total_free / max(1, sum(len(blocks) for blocks in self.free_blocks.values()))
                fragmentation = 1.0 - (avg_free_size / total_allocated)

                if fragmentation > self.config.fragmentation_threshold:
                    self.logger.info(".1f"                    self._perform_defragmentation()

        except Exception as e:
            self.logger.warning(f"⚠️ Defragmentation check failed: {e}")

    def _perform_defragmentation(self) -> None:
        """Perform memory defragmentation."""
        try:
            self.logger.info("🔧 Performing memory defragmentation")

            # For each pool, consolidate free blocks
            for pool_id, blocks in self.free_blocks.items():
                if len(blocks) < 2:
                    continue

                # Sort by size (largest first)
                blocks.sort(key=lambda b: b.size_bytes, reverse=True)

                # Try to merge adjacent blocks (simplified)
                consolidated = []
                current_block = None

                for block in blocks:
                    if current_block is None:
                        current_block = block
                    elif (block.size_bytes + current_block.size_bytes <
                          self.config.max_block_size_mb * 1024 * 1024):
                        # Merge blocks
                        current_block.size_bytes += block.size_bytes
                        current_block.last_access_time = max(
                            current_block.last_access_time,
                            block.last_access_time
                        )
                    else:
                        consolidated.append(current_block)
                        current_block = block

                if current_block:
                    consolidated.append(current_block)

                self.free_blocks[pool_id] = consolidated

            self.stats.defragmentation_count += 1
            self.logger.info("✅ Defragmentation completed")

        except Exception as e:
            self.logger.warning(f"⚠️ Defragmentation failed: {e}")

    def optimize_memory_layout(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        with self._lock:
            try:
                results = {
                    'cleanup_performed': False,
                    'defragmentation_performed': False,
                    'memory_freed_mb': 0.0,
                    'optimization_timestamp': datetime.now().isoformat()
                }

                # Force cleanup
                old_usage = self.stats.current_memory_usage_bytes
                self._perform_cleanup()
                new_usage = self.stats.current_memory_usage_bytes
                freed_mb = (old_usage - new_usage) / (1024 * 1024)

                results['cleanup_performed'] = True
                results['memory_freed_mb'] = freed_mb

                # Force defragmentation
                self._perform_defragmentation()
                results['defragmentation_performed'] = True

                # Clear GPU cache
                if self.device.type == "mps":
                    torch.mps.empty_cache()
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()

                self.logger.info("🎯 Memory optimization completed"                self.logger.info(f"💾 Memory freed: {freed_mb:.1f} MB")

                return results

            except Exception as e:
                self.logger.error(f"❌ Memory optimization failed: {e}")
                return {'error': str(e)}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self._lock:
            try:
                total_free = sum(block.size_bytes for blocks in self.free_blocks.values()
                               for block in blocks)
                total_allocated = sum(block.size_bytes for blocks in self.allocated_blocks.values()
                                    for block in blocks.values())

                fragmentation_ratio = 0.0
                if total_allocated + total_free > 0:
                    fragmentation_ratio = 1.0 - (total_free / (total_allocated + total_free))

                return {
                    'total_allocated_bytes': total_allocated,
                    'total_free_bytes': total_free,
                    'current_usage_bytes': self.stats.current_memory_usage_bytes,
                    'peak_usage_bytes': self.stats.peak_memory_usage_bytes,
                    'fragmentation_ratio': fragmentation_ratio,
                    'reuse_count': self.stats.reuse_count,
                    'defragmentation_count': self.stats.defragmentation_count,
                    'memory_pressure_events': self.stats.memory_pressure_events,
                    'pools_count': len(self.memory_pools),
                    'free_blocks_count': sum(len(blocks) for blocks in self.free_blocks.values()),
                    'allocated_blocks_count': sum(len(blocks) for blocks in self.allocated_blocks.values()),
                    'cache_hit_rate': self.stats.cache_hit_rate,
                    'last_cleanup': self.stats.last_cleanup_time.isoformat()
                }

            except Exception as e:
                self.logger.error(f"❌ Error getting memory stats: {e}")
                return {'error': str(e)}

    @contextmanager
    def memory_context(self, operation_type: str = "general"):
        """Context manager for optimized memory usage."""
        try:
            # Setup for operation
            pool_id = f"{self.device.type}_{operation_type}"
            yield self

        finally:
            # Cleanup after operation
            try:
                # Force cleanup for this operation type
                if pool_id in self.free_blocks:
                    # Keep only recent blocks
                    cutoff_time = datetime.now() - timedelta(seconds=30)
                    self.free_blocks[pool_id] = [
                        block for block in self.free_blocks[pool_id]
                        if block.last_access_time > cutoff_time
                    ]
            except Exception:
                pass  # Ignore cleanup errors

    def reset_pools(self) -> None:
        """Reset all memory pools (for testing or emergency cleanup)."""
        with self._lock:
            self.memory_pools.clear()
            self.free_blocks.clear()
            self.allocated_blocks.clear()
            self.allocation_history.clear()

            # Reset statistics
            self.stats = MemoryPoolStats()

            self.logger.info("🔄 Memory pools reset")
