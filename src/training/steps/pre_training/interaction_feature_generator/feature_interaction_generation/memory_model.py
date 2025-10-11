"""
Memory-Efficient Model for Interactive Feature Generation

This module implements a sophisticated memory management system using:
- PyArrow/Parquet for columnar storage (zero-copy into pandas/Polars)
- NumPy memmap for large 2D arrays
- Chunked operations for time and feature tiles
- Rolling stats reuse to avoid recomputation
- Shared read-only memmap across workers

Key Features:
- Zero-copy data access
- Memory-mapped large arrays
- Chunked processing for memory efficiency
- Rolling statistics optimization
- Shared memory across processes
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import logging
import mmap
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    # Memory limits
    max_memory_gb: float = 8.0
    chunk_size_mb: float = 100.0
    tile_size: int = 1000
    
    # PyArrow/Parquet settings
    use_parquet: bool = True
    compression: str = "snappy"
    use_dictionary_encoding: bool = True
    
    # Memmap settings
    use_memmap: bool = True
    memmap_threshold_mb: float = 50.0
    
    # Rolling stats settings
    enable_rolling_stats_reuse: bool = True
    rolling_window_cache_size: int = 1000
    
    # Shared memory settings
    enable_shared_memory: bool = True
    shared_memory_dir: Optional[str] = None


@dataclass
class RollingStatsCache:
    """Cache for rolling statistics to avoid recomputation."""
    sums: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    sumsq: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    counts: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)
    last_update: Dict[Tuple[str, int], int] = field(default_factory=dict)
    
    def get_rolling_stats(self, data: np.ndarray, window: int, start_idx: int) -> Tuple[float, float, int]:
        """Get rolling statistics for a window, reusing cached values where possible."""
        cache_key = (id(data), window)
        
        if cache_key not in self.sums:
            # Initialize cache
            self.sums[cache_key] = np.zeros(len(data))
            self.sumsq[cache_key] = np.zeros(len(data))
            self.counts[cache_key] = np.zeros(len(data), dtype=int)
            self.last_update[cache_key] = -1
        
        # Check if we can reuse cached values
        if start_idx > self.last_update[cache_key]:
            # Update cache from last position
            for i in range(max(0, self.last_update[cache_key] + 1), start_idx + window):
                if i < len(data):
                    self.sums[cache_key][i] = self.sums[cache_key][i-1] + data[i] if i > 0 else data[i]
                    self.sumsq[cache_key][i] = self.sumsq[cache_key][i-1] + data[i]**2 if i > 0 else data[i]**2
                    self.counts[cache_key][i] = min(i + 1, window)
            
            self.last_update[cache_key] = start_idx + window - 1
        
        # Calculate rolling stats for the window
        end_idx = min(start_idx + window, len(data))
        if start_idx == 0:
            sum_val = self.sums[cache_key][end_idx - 1]
            sumsq_val = self.sumsq[cache_key][end_idx - 1]
            count = self.counts[cache_key][end_idx - 1]
        else:
            sum_val = self.sums[cache_key][end_idx - 1] - self.sums[cache_key][start_idx - 1]
            sumsq_val = self.sumsq[cache_key][end_idx - 1] - self.sumsq[cache_key][start_idx - 1]
            count = min(window, end_idx - start_idx)
        
        return sum_val, sumsq_val, count


class MemoryEfficientProcessor:
    """
    Memory-efficient processor for large-scale feature generation.
    
    Uses PyArrow/Parquet for columnar storage and NumPy memmap for large arrays.
    Implements chunked processing and rolling statistics optimization.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory-efficient processor."""
        self.config = config or MemoryConfig()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="feature_gen_"))
        self.memmap_files: Dict[str, str] = {}
        self.parquet_files: Dict[str, str] = {}
        self.rolling_cache = RollingStatsCache()
        self.shared_memory_lock = threading.Lock()
        
        # Create shared memory directory
        if self.config.enable_shared_memory:
            self.shared_memory_dir = Path(self.config.shared_memory_dir or self.temp_dir / "shared")
            self.shared_memory_dir.mkdir(exist_ok=True)
        
        tprint_info(f"🚀 Memory-efficient processor initialized")
        tprint_info(f"📊 Max memory: {self.config.max_memory_gb} GB")
        tprint_info(f"📦 Chunk size: {self.config.chunk_size_mb} MB")
        tprint_info(f"🗂️ Temp directory: {self.temp_dir}")
    
    def to_columnar(self, data: pd.DataFrame, name: str) -> pa.Table:
        """Convert DataFrame to PyArrow Table for efficient storage."""
        tprint_debug(f"🔄 Converting {name} to columnar format")
        
        # Fast-fail: Validate input data
        if data.empty:
            raise RuntimeError(f"CRITICAL: Cannot convert empty DataFrame {name} to columnar format")
        
        # Check for duplicate columns and handle them
        if len(data.columns) != len(set(data.columns)):
            duplicate_cols = [col for col in data.columns if list(data.columns).count(col) > 1]
            unique_duplicates = list(set(duplicate_cols))
            tprint_warning(f"⚠️ Found duplicate columns: {unique_duplicates[:10]}{'...' if len(unique_duplicates) > 10 else ''}")
            
            # Remove duplicate columns (keep first occurrence)
            data = data.loc[:, ~data.columns.duplicated(keep='first')]
            tprint_debug(f"✅ Removed duplicate columns, now have {len(data.columns)} unique columns")
        
        try:
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(data, preserve_index=True)
            
            # Apply compression and dictionary encoding only if beneficial
            if self.config.use_dictionary_encoding and len(data) > 1000:
                # Dictionary encode string columns
                for i, field in enumerate(table.schema):
                    if pa.types.is_string(field.type):
                        table = table.set_column(i, field.name, pa.compute.dictionary_encode(table.column(i)))
            
            return table
            
        except Exception as e:
            raise RuntimeError(f"CRITICAL: Failed to convert {name} to columnar format: {e}")
    
    def save_columnar(self, table: pa.Table, name: str) -> str:
        """Save PyArrow Table to Parquet file."""
        file_path = self.temp_dir / f"{name}.parquet"
        
        # Write with compression
        pq.write_table(
            table,
            file_path,
            compression=self.config.compression,
            use_dictionary=self.config.use_dictionary_encoding
        )
        
        self.parquet_files[name] = str(file_path)
        tprint_debug(f"💾 Saved {name} to {file_path}")
        return str(file_path)
    
    def load_columnar(self, name: str) -> pd.DataFrame:
        """Load Parquet file as DataFrame with zero-copy where possible."""
        if name not in self.parquet_files:
            raise ValueError(f"Columnar data {name} not found")
        
        file_path = self.parquet_files[name]
        
        # Read with zero-copy where possible
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        tprint_debug(f"📂 Loaded {name} from {file_path}")
        return df
    
    def create_memmap(self, data: np.ndarray, name: str, dtype: np.dtype = None) -> np.ndarray:
        """Create memory-mapped array for large data."""
        if not self.config.use_memmap:
            return data
        
        # Check if data is large enough to benefit from memmap
        data_size_mb = data.nbytes / (1024 * 1024)
        if data_size_mb < self.config.memmap_threshold_mb:
            return data
        
        file_path = self.temp_dir / f"{name}.npy"
        
        # Save as memmap
        memmap_array = np.memmap(
            file_path,
            dtype=dtype or data.dtype,
            mode='w+',
            shape=data.shape
        )
        
        # Copy data to memmap
        memmap_array[:] = data[:]
        memmap_array.flush()
        
        # Reopen as read-only for sharing
        shared_memmap = np.memmap(
            file_path,
            dtype=dtype or data.dtype,
            mode='r',
            shape=data.shape
        )
        
        self.memmap_files[name] = str(file_path)
        tprint_debug(f"🗂️ Created memmap {name}: {data_size_mb:.1f} MB")
        return shared_memmap
    
    def load_memmap(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Load memory-mapped array."""
        if name not in self.memmap_files:
            raise ValueError(f"Memmap {name} not found")
        
        file_path = self.memmap_files[name]
        return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
    
    def process_in_time_tiles(self, data: pd.DataFrame, processor_func: callable, 
                            tile_size: int = None) -> Iterator[pd.DataFrame]:
        """Process data in time-based tiles for memory efficiency."""
        tile_size = tile_size or self.config.tile_size
        
        tprint_debug(f"🔄 Processing data in time tiles of size {tile_size}")
        
        for start_idx in range(0, len(data), tile_size):
            end_idx = min(start_idx + tile_size, len(data))
            tile_data = data.iloc[start_idx:end_idx]
            
            tprint_debug(f"📦 Processing tile {start_idx}:{end_idx}")
            yield processor_func(tile_data)
    
    def process_in_feature_tiles(self, data: pd.DataFrame, processor_func: callable,
                               tile_size: int = None) -> Iterator[pd.DataFrame]:
        """Process data in feature-based tiles for memory efficiency."""
        tile_size = tile_size or self.config.tile_size
        
        tprint_debug(f"🔄 Processing data in feature tiles of size {tile_size}")
        
        for start_col in range(0, len(data.columns), tile_size):
            end_col = min(start_col + tile_size, len(data.columns))
            tile_data = data.iloc[:, start_col:end_col]
            
            tprint_debug(f"📦 Processing feature tile {start_col}:{end_col}")
            yield processor_func(tile_data)
    
    def rolling_mean_optimized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized rolling mean using cached statistics."""
        if not self.config.enable_rolling_stats_reuse:
            return pd.Series(data).rolling(window=window).mean().values
        
        result = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            sum_val, _, count = self.rolling_cache.get_rolling_stats(data, window, i - window + 1)
            result[i] = sum_val / count if count > 0 else np.nan
        
        return result
    
    def rolling_std_optimized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Optimized rolling standard deviation using cached statistics."""
        if not self.config.enable_rolling_stats_reuse:
            return pd.Series(data).rolling(window=window).std().values
        
        result = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            sum_val, sumsq_val, count = self.rolling_cache.get_rolling_stats(data, window, i - window + 1)
            
            if count > 1:
                mean_val = sum_val / count
                variance = (sumsq_val / count) - (mean_val ** 2)
                result[i] = np.sqrt(max(0, variance))
            else:
                result[i] = 0.0
        
        return result
    
    def rolling_correlation_optimized(self, x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Optimized rolling correlation using cached statistics."""
        if not self.config.enable_rolling_stats_reuse:
            return pd.Series(x).rolling(window=window).corr(pd.Series(y)).values
        
        result = np.full(len(x), np.nan)
        
        for i in range(window - 1, len(x)):
            # Get rolling stats for both series
            sum_x, sumsq_x, count_x = self.rolling_cache.get_rolling_stats(x, window, i - window + 1)
            sum_y, sumsq_y, count_y = self.rolling_cache.get_rolling_stats(y, window, i - window + 1)
            
            if count_x > 1 and count_y > 1:
                mean_x = sum_x / count_x
                mean_y = sum_y / count_y
                
                # Calculate correlation
                sum_xy = np.sum(x[i-window+1:i+1] * y[i-window+1:i+1])
                cov_xy = (sum_xy / count_x) - (mean_x * mean_y)
                
                var_x = (sumsq_x / count_x) - (mean_x ** 2)
                var_y = (sumsq_y / count_y) - (mean_y ** 2)
                
                if var_x > 0 and var_y > 0:
                    result[i] = cov_xy / np.sqrt(var_x * var_y)
        
        return result
    
    def create_shared_memmap(self, data: np.ndarray, name: str) -> str:
        """Create shared memory-mapped array for multi-process access."""
        if not self.config.enable_shared_memory:
            return self.create_memmap(data, name)
        
        with self.shared_memory_lock:
            file_path = self.shared_memory_dir / f"{name}.npy"
            
            # Create shared memmap
            shared_memmap = np.memmap(
                file_path,
                dtype=data.dtype,
                mode='w+',
                shape=data.shape
            )
            
            # Copy data
            shared_memmap[:] = data[:]
            shared_memmap.flush()
            
            tprint_debug(f"🤝 Created shared memmap {name}: {file_path}")
            return str(file_path)
    
    def load_shared_memmap(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Load shared memory-mapped array."""
        if not self.config.enable_shared_memory:
            return self.load_memmap(name, shape, dtype)
        
        file_path = self.shared_memory_dir / f"{name}.npy"
        
        if not file_path.exists():
            raise ValueError(f"Shared memmap {name} not found")
        
        return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def cleanup(self) -> None:
        """Clean up temporary files and memory."""
        tprint_debug("🧹 Cleaning up memory-efficient processor")
        
        # Clear caches
        self.rolling_cache.sums.clear()
        self.rolling_cache.sumsq.clear()
        self.rolling_cache.counts.clear()
        self.rolling_cache.last_update.clear()
        
        # Remove temporary files
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Force garbage collection
        gc.collect()
        
        tprint_success("✅ Memory cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


# Convenience functions

def create_memory_efficient_processor(config: Optional[MemoryConfig] = None) -> MemoryEfficientProcessor:
    """Create a memory-efficient processor with the given configuration."""
    return MemoryEfficientProcessor(config)


def process_large_dataset(data: pd.DataFrame, processor_func: callable, 
                         config: Optional[MemoryConfig] = None) -> pd.DataFrame:
    """Process a large dataset using memory-efficient techniques."""
    processor = create_memory_efficient_processor(config)
    
    try:
        # Convert to columnar format
        table = processor.to_columnar(data, "input_data")
        
        # Process in chunks
        results = []
        for chunk in processor.process_in_time_tiles(data, processor_func):
            results.append(chunk)
        
        # Combine results
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    finally:
        processor.cleanup()


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'price': np.random.randn(10000).cumsum(),
        'volume': np.random.randint(1000, 10000, 10000),
        'feature1': np.random.randn(10000),
        'feature2': np.random.randn(10000)
    })
    
    # Create processor
    config = MemoryConfig(max_memory_gb=2.0, chunk_size_mb=50.0)
    processor = create_memory_efficient_processor(config)
    
    try:
        # Test rolling statistics optimization
        print("Testing rolling statistics optimization...")
        
        # Standard rolling mean
        start_time = time.time()
        standard_mean = data['price'].rolling(window=20).mean()
        standard_time = time.time() - start_time
        
        # Optimized rolling mean
        start_time = time.time()
        optimized_mean = processor.rolling_mean_optimized(data['price'].values, 20)
        optimized_time = time.time() - start_time
        
        print(f"Standard rolling mean: {standard_time:.3f}s")
        print(f"Optimized rolling mean: {optimized_time:.3f}s")
        print(f"Speedup: {standard_time / optimized_time:.1f}x")
        
        # Test memory usage
        memory_usage = processor.get_memory_usage()
        print(f"Memory usage: {memory_usage['rss_mb']:.1f} MB")
        
    finally:
        processor.cleanup()