"""
M1 Memory Optimizer for Training Pipeline.

This module provides memory optimization utilities specifically designed for M1/M2/M3 Macs,
including intelligent memory management, chunked processing, and memory-efficient data structures.
"""

import gc
import logging
import psutil

import threading
import tracemalloc

import subprocess
from typing import Any, Dict, List, Optional, Generator, Callable, TypeVar, Set
from contextlib import contextmanager
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time

# Import M1CPUOptimizer for compatibility
try:
    from .m1_cpu_optimizer import M1CPUOptimizer
except ImportError:
    M1CPUOptimizer = None

logger = logging.getLogger(__name__)

T = TypeVar('T')

class M1MemoryOptimizer:
    """Memory optimizer for M1 Macs with intelligent resource management."""

    def __init__(self, memory_limit_gb: float = 8.0, enable_gc_tuning: bool = True,
                 enable_memory_leak_detection: bool = True, enable_swap_management: bool = True):
        """Initialize memory optimizer.

        Args:
            memory_limit_gb: Memory limit in GB
            enable_gc_tuning: Whether to tune garbage collection
            enable_memory_leak_detection: Whether to enable memory leak detection
            enable_swap_management: Whether to enable swap management
        """
        self.memory_limit_gb = memory_limit_gb
        self.enable_gc_tuning = enable_gc_tuning
        self.enable_memory_leak_detection = enable_memory_leak_detection
        self.enable_swap_management = enable_swap_management
        self.logger = logger.getChild('M1MemoryOptimizer')

        # Memory tracking
        self.memory_history = []
        self.peak_memory_usage = 0
        self.memory_checkpoints = {}

        # Memory leak detection
        self.object_registry: Set[int] = set()
        self.leak_detection_enabled = False
        self.leak_snapshots = []

        # Swap management
        self.swap_info = self._get_swap_info()
        self.memory_compression_enabled = self._check_memory_compression()

        # Optimize garbage collection for M1
        if self.enable_gc_tuning:
            self._optimize_gc_settings()

        # Initialize memory leak detection
        if self.enable_memory_leak_detection:
            self._init_memory_leak_detection()

        self.logger.info(f"🧠 M1 Memory Optimizer initialized (limit: {memory_limit_gb}GB)")
        self.logger.info(f"🔧 Memory compression: {'enabled' if self.memory_compression_enabled else 'disabled'}")

    def _optimize_gc_settings(self):
        """Optimize garbage collection settings for M1."""

        # Set GC thresholds optimized for M1 unified memory
        gc.set_threshold(700, 10, 10)  # More aggressive collection

        # Disable automatic GC during intensive operations
        gc.disable()  # We'll manage GC manually

        self.logger.info("🔧 GC settings optimized for M1")

    def _init_memory_leak_detection(self):
        """Initialize memory leak detection."""
        try:
            tracemalloc.start()
            self.leak_detection_enabled = True

            # Take initial snapshot
            self.leak_snapshots.append(tracemalloc.take_snapshot())

            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._memory_leak_monitor, daemon=True
            )
            self.monitoring_thread.start()

            self.logger.info("🔍 Memory leak detection initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize memory leak detection: {e}")
            self.leak_detection_enabled = False

    def _get_swap_info(self) -> Dict[str, Any]:
        """Get swap/memory compression information for M1."""
        try:
            # Get swap usage via sysctl
            result = subprocess.run(
                ['sysctl', 'vm.swapusage'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                swap_output = result.stdout.strip()
                # Parse output like: "vm.swapusage: total = 2048.00M used = 256.00M free = 1792.00M"
                # Remove "=" signs and clean up the output
                cleaned_output = swap_output.replace('=', '').replace(':', '')
                parts = cleaned_output.split()

                # Find the values after the labels
                total_idx = parts.index('total') + 1 if 'total' in parts else 2
                used_idx = parts.index('used') + 1 if 'used' in parts else 4
                free_idx = parts.index('free') + 1 if 'free' in parts else 6

                total = float(parts[total_idx].rstrip('M')) if 'M' in parts[total_idx] else float(parts[total_idx].rstrip('G')) * 1024
                used = float(parts[used_idx].rstrip('M')) if 'M' in parts[used_idx] else float(parts[used_idx].rstrip('G')) * 1024
                free = float(parts[free_idx].rstrip('M')) if 'M' in parts[free_idx] else float(parts[free_idx].rstrip('G')) * 1024

                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'usage_percent': (used / total) * 100 if total > 0 else 0
                }
            else:
                return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'usage_percent': 0}
        except Exception as e:
            self.logger.warning(f"Failed to get swap info: {e}")
            return {'total_mb': 0, 'used_mb': 0, 'free_mb': 0, 'usage_percent': 0}

    def _check_memory_compression(self) -> bool:
        """Check if memory compression is enabled on M1."""
        try:
            result = subprocess.run(
                ['sysctl', 'vm.compressor_mode'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                mode = result.stdout.strip().split()[-1]
                return mode != '0'  # 0 means disabled
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check memory compression: {e}")
            return False

    def _memory_leak_monitor(self):
        """Monitor for memory leaks in background thread."""
        while self.leak_detection_enabled:
            try:
                time.sleep(60)  # Check every minute

                if len(self.leak_snapshots) >= 2:
                    # Compare with previous snapshot
                    current = tracemalloc.take_snapshot()
                    previous = self.leak_snapshots[-1]

                    stats = current.compare_to(previous, 'lineno')

                    # Check for significant memory growth
                    total_growth = sum(stat.size_diff for stat in stats[:10])  # Top 10 growing locations

                    if total_growth > 50 * 1024 * 1024:  # 50MB growth
                        self.logger.warning(f"🚨 Potential memory leak detected: {total_growth / 1024**2:.1f}MB growth")

                        # Log top memory consumers
                        for stat in stats[:5]:
                            if stat.size_diff > 10 * 1024 * 1024:  # 10MB per location
                                self.logger.warning(
                                    f"   {stat.traceback.format()[0]}: +{stat.size_diff / 1024**2:.1f}MB"
                                )

                # Keep only last 10 snapshots
                if len(self.leak_snapshots) > 10:
                    self.leak_snapshots.pop(0)

                self.leak_snapshots.append(tracemalloc.take_snapshot())

            except Exception as e:
                self.logger.error(f"Memory leak monitoring error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'rss_gb': memory_info.rss / (1024**3),
                'vms_gb': memory_info.vms / (1024**3),
                'percentage': process.memory_percent(),
                'available_gb': psutil.virtual_memory().available / (1024**3)
            }
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return {'rss_gb': 0, 'vms_gb': 0, 'percentage': 0, 'available_gb': 8.0}

    def should_chunk_data(self, data_size_mb: float, operation_type: str = "general") -> bool:
        """Determine if data should be processed in chunks."""
        current_memory = self.get_memory_usage()

        # M1-specific chunking thresholds
        thresholds = {
            'matrix_mult': 500,  # MB
            'neural_net': 1000,
            'general': 2000
        }

        threshold = thresholds.get(operation_type, thresholds['general'])

        # Check if operation would exceed memory limits
        available_mb = current_memory['available_gb'] * 1024
        should_chunk = data_size_mb > threshold or data_size_mb > (available_mb * 0.7)

        if should_chunk:
            self.logger.debug(f"📦 Chunking recommended for {operation_type} operation ({data_size_mb:.1f}MB)")

        return should_chunk

    def calculate_optimal_chunk_size(self, data_shape: tuple, operation_type: str = "general") -> int:
        """Calculate optimal chunk size for data processing."""
        data_size_mb = np.prod(data_shape) * 8 / (1024**2)  # Assume float64

        if not self.should_chunk_data(data_size_mb, operation_type):
            return data_shape[0]  # No chunking needed

        current_memory = self.get_memory_usage()
        available_mb = current_memory['available_gb'] * 1024

        # M1-specific chunk size calculations
        base_chunk_sizes = {
            'matrix_mult': 1000,
            'neural_net': 500,
            'general': 2000
        }

        base_size = base_chunk_sizes.get(operation_type, base_chunk_sizes['general'])

        # Adjust based on available memory
        memory_factor = min(1.0, available_mb / (self.memory_limit_gb * 1024))
        optimal_size = int(base_size * memory_factor)

        # Ensure reasonable bounds
        optimal_size = max(100, min(optimal_size, data_shape[0]))

        self.logger.debug(f"📏 Optimal chunk size: {optimal_size} for {operation_type}")
        return optimal_size

    @contextmanager
    def memory_checkpoint(self, checkpoint_name: str):
        """Context manager for memory checkpointing."""
        start_memory = self.get_memory_usage()
        self.memory_checkpoints[checkpoint_name] = start_memory

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            memory_delta = end_memory['rss_gb'] - start_memory['rss_gb']

            self.logger.debug(
                f"📊 Memory checkpoint '{checkpoint_name}': {memory_delta:+.2f}GB "
                f"({start_memory['rss_gb']:.1f}GB → {end_memory['rss_gb']:.1f}GB)"
            )

    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization with aggressive MPS clearing."""
        results = {
            'gc_collected': 0,
            'memory_freed_mb': 0,
            'torch_cache_cleared': False,
            'numpy_cleanup': False,
            'mps_aggressive_clear': False,
            'swap_optimized': False,
            'memory_compression_optimized': False
        }

        start_memory = self.get_memory_usage()['rss_gb']

        try:
            # Force garbage collection
            if self.enable_gc_tuning:
                results['gc_collected'] = gc.collect()

            # Aggressive PyTorch/MPS cache clearing
            results.update(self._aggressive_cache_clearing())

            # Clear numpy memory
            try:
                import ctypes
                libc = ctypes.CDLL("libc.dylib")
                libc.malloc_trim(0)
                results['numpy_cleanup'] = True
            except Exception as e:
                self.logger.debug(f"NumPy cleanup failed: {e}")

            # Optimize swap usage
            if self.enable_swap_management:
                results['swap_optimized'] = self._optimize_swap_usage()

            # Optimize memory compression
            if self.memory_compression_enabled:
                results['memory_compression_optimized'] = self._optimize_memory_compression()

            # Track peak memory
            current_memory = self.get_memory_usage()
            if current_memory['rss_gb'] > self.peak_memory_usage:
                self.peak_memory_usage = current_memory['rss_gb']

            results['memory_freed_mb'] = (start_memory - current_memory['rss_gb']) * 1024

            self.logger.debug(f"🧹 Aggressive memory optimization completed: {results}")

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")

        return results

    def _aggressive_cache_clearing(self) -> Dict[str, Any]:
        """Perform aggressive cache clearing for MPS."""
        results = {
            'torch_cache_cleared': False,
            'mps_aggressive_clear': False,
            'cuda_cache_cleared': False
        }

        try:
            # Clear PyTorch caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
                results['cuda_cache_cleared'] = True
                results['torch_cache_cleared'] = True

            elif torch.backends.mps.is_available():
                # Aggressive MPS cache clearing
                torch.mps.empty_cache()

                # Multiple cache clears for thorough cleanup
                for _ in range(3):
                    torch.mps.empty_cache()
                    time.sleep(0.01)  # Small delay between clears

                results['mps_aggressive_clear'] = True
                results['torch_cache_cleared'] = True

                # Force MPS synchronization
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()

            # Clear any remaining GPU memory pools
            if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_emptyCache'):
                torch._C._cuda_emptyCache()

        except Exception as e:
            self.logger.warning(f"Cache clearing failed: {e}")

        return results

    def _optimize_swap_usage(self) -> bool:
        """Optimize swap usage for M1 memory management."""
        try:
            current_swap = self._get_swap_info()

            # If swap usage is high, trigger memory cleanup
            if current_swap['usage_percent'] > 70:
                self.logger.info(f"🔄 High swap usage ({current_swap['usage_percent']:.1f}%), triggering cleanup")

                # Force additional garbage collection
                gc.collect()

                # Trigger system memory cleanup
                try:
                    subprocess.run(
                        ['sudo', 'purge'],
                        capture_output=True,
                        timeout=30
                    )
                    self.logger.info("🧽 System memory purge completed")
                except Exception as e:
                    self.logger.debug(f"System purge failed: {e}")

                return True

            return False

        except Exception as e:
            self.logger.warning(f"Swap optimization failed: {e}")
            return False

    def _optimize_memory_compression(self) -> bool:
        """Optimize memory compression settings."""
        try:
            # Check current compression stats
            result = subprocess.run(
                ['sysctl', 'vm.compressor_bytes_used', 'vm.compressor_compressed_bytes'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                used = int(lines[0].split()[-1])
                compressed = int(lines[1].split()[-1])

                compression_ratio = compressed / used if used > 0 else 0

                if compression_ratio > 2.0:  # High compression ratio
                    self.logger.info(f"📦 High memory compression ratio: {compression_ratio:.2f}")
                    # Could trigger memory defragmentation here if needed

                return True

            return False

        except Exception as e:
            self.logger.warning(f"Memory compression optimization failed: {e}")
            return False

    def chunked_dataframe_processor(
        self,
        df: pd.DataFrame,
        processor_func: Callable[[pd.DataFrame], T],
        chunk_size: Optional[int] = None,
        operation_type: str = "general"
    ) -> Generator[T, None, None]:
        """Process DataFrame in memory-efficient chunks."""

        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(df.shape, operation_type)

        total_rows = len(df)

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]

            with self.memory_checkpoint(f"chunk_{start_idx}_{end_idx}"):
                result = processor_func(chunk)

                # Memory cleanup between chunks
                if start_idx % (chunk_size * 3) == 0:
                    self.optimize_memory()

            yield result

    def memory_efficient_concat(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Memory-efficient DataFrame concatenation."""
        if not dataframes:
            return pd.DataFrame()

        if len(dataframes) == 1:
            return dataframes[0]

        # Estimate memory requirements
        total_memory_mb = sum(df.memory_usage(deep=True).sum() for df in dataframes) / (1024**2)

        if not self.should_chunk_data(total_memory_mb, "general"):
            # Direct concatenation
            return pd.concat(dataframes, ignore_index=True)

        # Memory-efficient concatenation
        result = dataframes[0]

        for df in dataframes[1:]:
            # Concatenate in smaller chunks if needed
            if len(result) + len(df) > 100000:  # Arbitrary large size threshold
                self.optimize_memory()

            result = pd.concat([result, df], ignore_index=True)

        return result

    def create_memory_efficient_array(self, data: Any, dtype: np.dtype = np.float32) -> np.ndarray:
        """Create memory-efficient numpy array."""
        # Use float32 by default for M1 (better MPS performance)
        if isinstance(data, pd.DataFrame):
            # Convert to float32 for memory efficiency
            array = data.values.astype(dtype)
        elif isinstance(data, list):
            array = np.array(data, dtype=dtype)
        else:
            array = np.asarray(data, dtype=dtype)

        # Check if we should use memory mapping for large arrays
        array_size_mb = array.nbytes / (1024**2)

        if array_size_mb > 500:  # Large array threshold
            self.logger.warning(f"⚠️ Large array detected ({array_size_mb:.1f}MB), consider using memory mapping")

        return array

    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        current_memory = self.get_memory_usage()
        current_swap = self.swap_info if self.enable_swap_management else {}

        # Get memory leak info if available
        leak_info = {}
        if self.leak_detection_enabled and self.leak_snapshots:
            try:
                current_snapshot = tracemalloc.take_snapshot()
                if len(self.leak_snapshots) > 0:
                    stats = current_snapshot.compare_to(self.leak_snapshots[-1], 'lineno')
                    leak_info = {
                        'total_objects': len(stats),
                        'top_memory_locations': [
                            {
                                'file': stat.traceback.format()[0],
                                'size_mb': stat.size / 1024**2,
                                'count': stat.count
                            }
                            for stat in stats[:5]
                        ]
                    }
            except Exception as e:
                self.logger.debug(f"Failed to get leak info: {e}")

        return {
            'current_usage_gb': current_memory['rss_gb'],
            'peak_usage_gb': self.peak_memory_usage,
            'available_gb': current_memory['available_gb'],
            'memory_limit_gb': self.memory_limit_gb,
            'usage_percentage': current_memory['percentage'],
            'memory_efficiency': (self.memory_limit_gb - current_memory['rss_gb']) / self.memory_limit_gb,
            'checkpoints': self.memory_checkpoints,
            'gc_enabled': self.enable_gc_tuning,
            'memory_leak_detection_enabled': self.leak_detection_enabled,
            'swap_info': current_swap,
            'memory_compression_enabled': self.memory_compression_enabled,
            'leak_info': leak_info,
            'object_registry_size': len(self.object_registry),
            'leak_snapshots_count': len(self.leak_snapshots)
        }

class M1DataManager:
    """Data manager optimized for M1 memory architecture."""

    def __init__(self, memory_optimizer: M1MemoryOptimizer):
        self.memory_optimizer = memory_optimizer
        self.logger = logger.getChild('M1DataManager')
        self.cache = {}

    def load_data_efficiently(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Load data with memory efficiency considerations."""

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Estimate file size
        file_size_mb = file_path.stat().st_size / (1024**2)

        if file_size_mb > 1000:  # Large file
            self.logger.info(f"📂 Loading large file ({file_size_mb:.1f}MB) with chunking")

            if file_path.suffix == '.csv':
                # Use chunked CSV reading
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size or 50000, usecols=columns):
                    chunks.append(chunk)

                    # Memory management
                    if len(chunks) % 5 == 0:
                        self.memory_optimizer.optimize_memory()

                return self.memory_optimizer.memory_efficient_concat(chunks)

            elif file_path.suffix in ['.parquet', '.pq']:
                # Parquet files are already efficient
                return pd.read_parquet(file_path, columns=columns)

        # Normal loading for smaller files
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path, usecols=columns)
        elif file_path.suffix in ['.parquet', '.pq']:
            return pd.read_parquet(file_path, columns=columns)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def save_data_efficiently(
        self,
        df: pd.DataFrame,
        file_path: str,
        format: str = 'parquet',
        compression: str = 'snappy'
    ):
        """Save data with memory efficiency."""

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'parquet':
            df.to_parquet(file_path, compression=compression, index=False)
        elif format == 'csv':
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"💾 Saved data to {file_path} ({format}, {compression})")

# Global instance
_m1_memory_optimizer = None

def get_m1_memory_optimizer() -> M1MemoryOptimizer:
    """Get global M1 memory optimizer instance."""
    global _m1_memory_optimizer
    if _m1_memory_optimizer is None:
        _m1_memory_optimizer = M1MemoryOptimizer()
    return _m1_memory_optimizer

def create_memory_efficient_dataframe(*args, **kwargs) -> pd.DataFrame:
    """Create DataFrame with memory optimizations."""
    df = pd.DataFrame(*args, **kwargs)

    # Convert object columns to category if beneficial
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')

    return df

def memory_efficient_groupby(df: pd.DataFrame, by: str, agg_func: str = 'mean') -> pd.DataFrame:
    """Memory-efficient groupby operation with final re-aggregation."""
    memory_optimizer = get_m1_memory_optimizer()

    # Check if chunking is needed
    data_size_mb = df.memory_usage(deep=True).sum() / (1024**2)

    if memory_optimizer.should_chunk_data(data_size_mb, "general"):
        # Process in chunks
        intermediate_results = []
        chunk_size = memory_optimizer.calculate_optimal_chunk_size(df.shape)

        logger.info(f"🔄 Processing groupby in chunks (size: {chunk_size}) for {len(df)} rows")

        for chunk in memory_optimizer.chunked_dataframe_processor(
            df, lambda x: x.groupby(by).agg(agg_func), chunk_size
        ):
            intermediate_results.append(chunk)

        if not intermediate_results:
            return pd.DataFrame()

        # Combine intermediate results
        combined = memory_optimizer.memory_efficient_concat(intermediate_results)

        # Final re-aggregation step - crucial for correct results when groups span multiple chunks
        logger.debug(f"🔄 Performing final re-aggregation on {len(combined)} intermediate rows")

        # Reset index to avoid issues with groupby
        if isinstance(combined.index, pd.MultiIndex):
            combined = combined.reset_index()

        # Re-group and re-aggregate to ensure correctness
        final_result = combined.groupby(by).agg(agg_func)

        logger.info(f"✅ Groupby completed with final re-aggregation: {len(final_result)} groups")
        return final_result
    else:
        return df.groupby(by).agg(agg_func)
