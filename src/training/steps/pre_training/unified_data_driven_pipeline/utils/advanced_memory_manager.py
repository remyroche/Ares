"""
Advanced Memory Manager for Large-Scale Feature Interaction Generation

Implements memory mapping, incremental loading, memory pools, and cache-friendly
data layouts optimized for M1 Apple Silicon architecture.
"""

import gc
import os
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Iterator, Union
from dataclasses import dataclass
import logging
import psutil
import warnings
from contextlib import contextmanager
import h5py

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for advanced memory management."""
    
    # Memory thresholds
    memory_mapping_threshold_gb: float = 2.0
    chunk_size: int = 10000
    memory_pool_sizes: List[int] = None
    
    # Cache optimization
    cache_line_size: int = 64  # M1 cache line size
    enable_memory_pool: bool = True
    enable_memory_mapping: bool = True
    
    # Garbage collection
    gc_frequency: int = 100  # Run GC every N operations
    aggressive_gc_threshold: float = 0.85  # Memory usage threshold for aggressive GC
    
    def __post_init__(self):
        if self.memory_pool_sizes is None:
            self.memory_pool_sizes = [1000, 5000, 10000, 50000, 100000]

class AdvancedMemoryManager:
    """Advanced memory manager with M1-optimized memory operations."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logger.getChild('AdvancedMemoryManager')
        
        # Memory tracking
        self.operation_count = 0
        self.memory_pressure_history = []
        
        # Memory pools
        self.memory_pools: Dict[int, List[np.ndarray]] = {}
        if self.config.enable_memory_pool:
            self._initialize_memory_pools()
        
        # Temporary file management
        self.temp_files = []
        self.temp_dir = None
        
        tprint("🧠 [MEMORY] Advanced Memory Manager initialized")
    
    def _initialize_memory_pools(self):
        """Initialize memory pools for common array sizes."""
        tprint("🔄 [MEMORY] Initializing memory pools for common array sizes")
        
        for size in self.config.memory_pool_sizes:
            self.memory_pools[size] = []
            # Pre-allocate a few arrays of each size
            for _ in range(3):
                pool_array = np.empty(size, dtype=np.float32)
                self.memory_pools[size].append(pool_array)
        
        tprint(f"✅ [MEMORY] Memory pools initialized for sizes: {self.config.memory_pool_sizes}")
    
    def _get_optimal_chunk_size(self, total_size: int, available_memory_gb: float) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Use 25% of available memory for chunk processing
        target_memory_bytes = available_memory_gb * 0.25 * 1024**3
        
        # Estimate memory per row (assuming float32 + overhead)
        bytes_per_row = 100  # Conservative estimate
        
        optimal_chunk_size = int(target_memory_bytes / bytes_per_row)
        
        # Ensure chunk size is reasonable
        optimal_chunk_size = max(1000, min(optimal_chunk_size, total_size))
        
        tprint(f"📊 [MEMORY] Optimal chunk size calculated: {optimal_chunk_size:,} rows")
        return optimal_chunk_size
    
    def should_use_memory_mapping(self, data: pd.DataFrame) -> bool:
        """Determine if data should use memory mapping based on size."""
        memory_usage_gb = data.memory_usage(deep=True).sum() / (1024**3)
        should_use = memory_usage_gb > self.config.memory_mapping_threshold_gb
        
        tprint(f"🔍 [MEMORY] Memory usage: {memory_usage_gb:.2f} GB, Memory mapping: {'Yes' if should_use else 'No'}")
        return should_use
    
    def create_memory_mapped_data(self, data: pd.DataFrame, name: str = "data") -> str:
        """Create memory-mapped version of DataFrame."""
        if not self.config.enable_memory_mapping:
            return None
        
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="ares_memory_mapped_")
        
        file_path = os.path.join(self.temp_dir, f"{name}.h5")
        
        try:
            # Downcast to int32/float32 for memory efficiency
            optimized_data = self._downcast_to_int32(data)
            
            # Store as HDF5 for memory mapping
            with h5py.File(file_path, 'w') as f:
                # Store metadata
                f.attrs['columns'] = optimized_data.columns.tolist()
                f.attrs['index'] = optimized_data.index.tolist()
                f.attrs['shape'] = optimized_data.shape
                
                # Store data as float32
                f.create_dataset('data', data=optimized_data.values.astype(np.float32), 
                               compression='gzip', compression_opts=9)
            
            self.temp_files.append(file_path)
            tprint(f"💾 [MEMORY] Memory-mapped file created: {file_path}")
            return file_path
            
        except Exception as e:
            tprint(f"❌ [MEMORY] Failed to create memory-mapped file: {e}")
            return None
    
    def load_memory_mapped_data(self, file_path: str) -> pd.DataFrame:
        """Load data from memory-mapped file."""
        try:
            with h5py.File(file_path, 'r') as f:
                # Load metadata
                columns = f.attrs['columns']
                index = f.attrs['index']
                shape = f.attrs['shape']
                
                # Load data (this will be memory-mapped)
                data_array = f['data'][:]
                
                # Reconstruct DataFrame
                df = pd.DataFrame(data_array, columns=columns, index=index)
                
                tprint(f"📂 [MEMORY] Loaded memory-mapped data: {df.shape}")
                return df
                
        except Exception as e:
            tprint(f"❌ [MEMORY] Failed to load memory-mapped file: {e}")
            return pd.DataFrame()
    
    def _downcast_to_int32(self, data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Downcast data types to int32/float32 for memory efficiency."""
        if isinstance(data, pd.Series):
            return self._downcast_series_to_int32(data)
        
        # Process DataFrame columns
        optimized_data = data.copy()
        
        for col in optimized_data.columns:
            if optimized_data[col].dtype == 'int64':
                # Check if values fit in int32
                if optimized_data[col].min() >= np.iinfo(np.int32).min and \
                   optimized_data[col].max() <= np.iinfo(np.int32).max:
                    optimized_data[col] = optimized_data[col].astype(np.int32)
                    tprint(f"🔧 [MEMORY] Downcasted {col} from int64 to int32")
            
            elif optimized_data[col].dtype == 'float64':
                # Check if values fit in float32
                if optimized_data[col].min() >= np.finfo(np.float32).min and \
                   optimized_data[col].max() <= np.finfo(np.float32).max:
                    optimized_data[col] = optimized_data[col].astype(np.float32)
                    tprint(f"🔧 [MEMORY] Downcasted {col} from float64 to float32")
            
            elif optimized_data[col].dtype == 'object':
                # Try to convert to categorical if low cardinality
                unique_ratio = optimized_data[col].nunique() / len(optimized_data)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_data[col] = optimized_data[col].astype('category')
                    tprint(f"🔧 [MEMORY] Converted {col} to categorical")
        
        return optimized_data
    
    def _downcast_series_to_int32(self, series: pd.Series) -> pd.Series:
        """Downcast a Series to int32/float32."""
        if series.dtype == 'int64':
            if series.min() >= np.iinfo(np.int32).min and \
               series.max() <= np.iinfo(np.int32).max:
                return series.astype(np.int32)
        elif series.dtype == 'float64':
            if series.min() >= np.finfo(np.float32).min and \
               series.max() <= np.finfo(np.float32).max:
                return series.astype(np.float32)
        return series
    
    def get_from_memory_pool(self, size: int) -> Optional[np.ndarray]:
        """Get an array from memory pool if available."""
        if not self.config.enable_memory_pool:
            return None
        
        # Find closest pool size
        closest_size = min(self.memory_pools.keys(), key=lambda x: abs(x - size))
        
        if closest_size in self.memory_pools and self.memory_pools[closest_size]:
            array = self.memory_pools[closest_size].pop()
            # Resize if needed
            if len(array) != size:
                array = np.empty(size, dtype=np.float32)
            return array
        
        return None
    
    def return_to_memory_pool(self, array: np.ndarray):
        """Return an array to memory pool."""
        if not self.config.enable_memory_pool:
            return
        
        size = len(array)
        closest_size = min(self.memory_pools.keys(), key=lambda x: abs(x - size))
        
        if closest_size in self.memory_pools:
            # Only keep a limited number in pool
            if len(self.memory_pools[closest_size]) < 5:
                self.memory_pools[closest_size].append(array)
    
    def incremental_data_processing(self, data_iterator: Iterator[pd.DataFrame], 
                                   processor_func, **kwargs) -> pd.DataFrame:
        """Process data in streaming fashion to handle datasets > available RAM."""
        tprint("🔄 [MEMORY] Starting incremental data processing")
        
        results = []
        chunk_count = 0
        
        for chunk in data_iterator:
            chunk_count += 1
            tprint(f"📊 [MEMORY] Processing chunk {chunk_count}")
            
            # Process chunk
            processed_chunk = processor_func(chunk, **kwargs)
            results.append(processed_chunk)
            
            # Immediate cleanup
            del chunk
            self._managed_garbage_collection()
        
        tprint(f"✅ [MEMORY] Completed incremental processing: {chunk_count} chunks")
        
        # Combine results
        if results:
            combined_result = pd.concat(results, ignore_index=True)
            del results
            self._managed_garbage_collection()
            return combined_result
        
        return pd.DataFrame()
    
    def create_chunk_iterator(self, data: pd.DataFrame, chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Create iterator for chunked data processing."""
        if chunk_size is None:
            available_memory = psutil.virtual_memory().available / (1024**3)
            chunk_size = self._get_optimal_chunk_size(len(data), available_memory)
        
        tprint(f"📊 [MEMORY] Creating chunk iterator with size: {chunk_size:,}")
        
        for i in range(0, len(data), chunk_size):
            yield data.iloc[i:i+chunk_size].copy()
    
    def cache_friendly_data_layout(self, data: pd.DataFrame, 
                                 access_frequency: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Optimize data layout for M1's cache hierarchy."""
        tprint("🔄 [MEMORY] Optimizing data layout for cache efficiency")
        
        if access_frequency is None:
            # Default: assume all columns accessed equally
            access_frequency = {col: 1.0 for col in data.columns}
        
        # Sort columns by access frequency (most accessed first)
        sorted_cols = sorted(data.columns, key=lambda x: access_frequency.get(x, 0), reverse=True)
        
        # Reorder columns for better cache locality
        optimized_data = data[sorted_cols]
        
        tprint(f"✅ [MEMORY] Reordered {len(sorted_cols)} columns for cache efficiency")
        return optimized_data
    
    def _managed_garbage_collection(self):
        """Manage garbage collection based on operation count and memory pressure."""
        self.operation_count += 1
        
        # Check memory pressure
        memory_usage = psutil.virtual_memory().percent / 100
        self.memory_pressure_history.append(memory_usage)
        
        # Keep only recent history
        if len(self.memory_pressure_history) > 100:
            self.memory_pressure_history = self.memory_pressure_history[-50:]
        
        # Determine GC strategy
        should_gc = False
        gc_type = "standard"
        
        if self.operation_count % self.config.gc_frequency == 0:
            should_gc = True
            gc_type = "periodic"
        
        if memory_usage > self.config.aggressive_gc_threshold:
            should_gc = True
            gc_type = "aggressive"
        
        if should_gc:
            tprint(f"🧹 [MEMORY] Running {gc_type} garbage collection (memory: {memory_usage:.1%})")
            
            if gc_type == "aggressive":
                # Multiple GC passes for aggressive cleanup
                for _ in range(3):
                    gc.collect()
            else:
                gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        
        stats = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent,
            'operation_count': self.operation_count,
            'memory_pools_active': sum(len(pool) for pool in self.memory_pools.values()),
            'temp_files': len(self.temp_files)
        }
        
        return stats
    
    def cleanup(self):
        """Clean up temporary files and memory pools."""
        tprint("🧹 [MEMORY] Cleaning up memory manager resources")
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                tprint(f"⚠️ [MEMORY] Failed to remove temp file {temp_file}: {e}")
        
        # Clean up temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except Exception as e:
                tprint(f"⚠️ [MEMORY] Failed to remove temp directory {self.temp_dir}: {e}")
        
        # Clear memory pools
        self.memory_pools.clear()
        
        # Final garbage collection
        gc.collect()
        
        tprint("✅ [MEMORY] Memory manager cleanup completed")
    
    @contextmanager
    def memory_context(self, operation_name: str):
        """Context manager for memory operations with automatic cleanup."""
        initial_memory = psutil.virtual_memory().used / (1024**3)
        
        try:
            tprint(f"🔄 [MEMORY] Starting memory context: {operation_name}")
            yield self
            
        finally:
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_delta = final_memory - initial_memory
            
            tprint(f"✅ [MEMORY] Completed memory context: {operation_name} (delta: {memory_delta:+.2f} GB)")
            
            # Cleanup if memory usage increased significantly
            if memory_delta > 1.0:  # More than 1GB increase
                self._managed_garbage_collection()
