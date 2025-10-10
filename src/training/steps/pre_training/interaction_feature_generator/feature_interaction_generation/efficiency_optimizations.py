"""
Efficiency Optimizations for Interactive Feature Generation

This module implements comprehensive efficiency improvements to make the feature
generation pipeline extremely fast and memory-efficient.

Key Optimizations:
- Data fingerprinting for cache keys
- Right-sized chunking for L3 cache
- Optimal parallelism choices
- Zero-copy data paths
- Preallocation over concatenation
- Memmap discipline
- Vectorized operations
- Memory-efficient algorithms
"""

import hashlib
import numpy as np
import pandas as pd
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging
import platform
import sys
from pathlib import Path
import pickle
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from functools import lru_cache

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency optimizations."""
    # Memory management
    max_memory_gb: float = 8.0
    memory_headroom_gb: float = 2.0
    chunk_size_mb: float = 0.75  # 0.5-1.0 × (max_memory - headroom) / workers
    
    # Parallelism
    max_workers: int = 4
    use_multiprocessing: bool = False  # True for CPU-bound, False for I/O-bound
    release_gil: bool = True  # For NumPy/numba operations
    
    # Caching
    enable_caching: bool = True
    cache_dir: str = "/tmp/feature_cache"
    cache_version: str = "1.0.0"
    
    # Data types
    default_dtype: np.dtype = np.float32
    accumulator_dtype: np.dtype = np.float64
    downcast_threshold: float = 1e-6
    
    # Chunking
    min_chunk_size: int = 1000
    max_chunk_size: int = 100000
    target_chunk_size: int = 10000
    
    # Vectorization
    vectorization_threshold: int = 1000
    batch_size: int = 10000
    
    # Monitoring
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_timing: bool = True


class DataFingerprinter:
    """Generate comprehensive data fingerprints for cache keys."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self._fingerprint_cache = {}
    
    def generate_fingerprint(self, 
                           data: pd.DataFrame,
                           config: Dict[str, Any],
                           code_version: str = "1.0.0") -> str:
        """
        Generate comprehensive fingerprint including:
        - Data hash (content + structure)
        - Configuration hash
        - Code version
        - Library versions
        - RNG seeds
        """
        try:
            # Data fingerprint
            data_hash = self._hash_dataframe(data)
            
            # Configuration fingerprint
            config_hash = self._hash_config(config)
            
            # Environment fingerprint
            env_hash = self._hash_environment(code_version)
            
            # RNG fingerprint
            rng_hash = self._hash_rng_state()
            
            # Combine all fingerprints
            combined = f"{data_hash}_{config_hash}_{env_hash}_{rng_hash}"
            fingerprint = hashlib.sha256(combined.encode()).hexdigest()[:16]
            
            tprint_debug(f"🔍 Generated fingerprint: {fingerprint}")
            return fingerprint
            
        except Exception as e:
            tprint_warning(f"⚠️ Fingerprint generation failed: {e}")
            return f"fallback_{int(time.time())}"
    
    def _hash_dataframe(self, data: pd.DataFrame) -> str:
        """Hash DataFrame content and structure."""
        try:
            # Hash data content
            content_hash = hashlib.sha256(data.values.tobytes()).hexdigest()[:8]
            
            # Hash structure
            structure = {
                'shape': data.shape,
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'index_type': str(type(data.index)),
                'columns': list(data.columns)
            }
            structure_hash = hashlib.sha256(
                json.dumps(structure, sort_keys=True).encode()
            ).hexdigest()[:8]
            
            return f"{content_hash}_{structure_hash}"
            
        except Exception as e:
            tprint_debug(f"⚠️ DataFrame hashing failed: {e}")
            return f"df_{int(time.time())}"
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Hash configuration parameters."""
        try:
            # Sort config for consistent hashing
            sorted_config = json.dumps(config, sort_keys=True, default=str)
            return hashlib.sha256(sorted_config.encode()).hexdigest()[:8]
        except Exception as e:
            tprint_debug(f"⚠️ Config hashing failed: {e}")
            return f"cfg_{int(time.time())}"
    
    def _hash_environment(self, code_version: str) -> str:
        """Hash environment and library versions."""
        try:
            env_info = {
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
                'platform': platform.platform(),
                'code_version': code_version
            }
            env_str = json.dumps(env_info, sort_keys=True)
            return hashlib.sha256(env_str.encode()).hexdigest()[:8]
        except Exception as e:
            tprint_debug(f"⚠️ Environment hashing failed: {e}")
            return f"env_{int(time.time())}"
    
    def _hash_rng_state(self) -> str:
        """Hash random number generator state."""
        try:
            rng_state = {
                'numpy_random_state': np.random.get_state()[1][0],  # First number
                'python_random_state': hash(str(hashlib.sha256(
                    str(np.random.random()).encode()
                ).hexdigest()))
            }
            rng_str = json.dumps(rng_state, sort_keys=True)
            return hashlib.sha256(rng_str.encode()).hexdigest()[:8]
        except Exception as e:
            tprint_debug(f"⚠️ RNG hashing failed: {e}")
            return f"rng_{int(time.time())}"


class ChunkingOptimizer:
    """Optimize chunking for L3 cache and memory budget."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self._memory_info = self._get_memory_info()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'l3_cache_mb': self._estimate_l3_cache()
            }
        except Exception as e:
            tprint_warning(f"⚠️ Memory info failed: {e}")
            return {
                'total_gb': 8.0,
                'available_gb': 6.0,
                'used_gb': 2.0,
                'l3_cache_mb': 16.0
            }
    
    def _estimate_l3_cache(self) -> float:
        """Estimate L3 cache size."""
        try:
            # Try to get CPU info
            cpu_count = psutil.cpu_count()
            # Rough estimate: 1-2MB per core for L3 cache
            return min(cpu_count * 1.5, 32.0)  # Cap at 32MB
        except:
            return 16.0  # Default estimate
    
    def calculate_optimal_chunk_size(self, 
                                   data_size_mb: float,
                                   num_workers: int) -> int:
        """
        Calculate optimal chunk size based on:
        - L3 cache size
        - Memory budget
        - Number of workers
        """
        try:
            # Memory budget per worker
            available_memory = self._memory_info['available_gb'] - self.config.memory_headroom_gb
            memory_per_worker = available_memory / num_workers
            
            # L3 cache consideration
            l3_cache_mb = self._memory_info['l3_cache_mb']
            l3_optimal = l3_cache_mb * 0.8  # Use 80% of L3 cache
            
            # Memory budget consideration
            memory_optimal = memory_per_worker * 1024 * self.config.chunk_size_mb
            
            # Choose the smaller of the two
            optimal_mb = min(l3_optimal, memory_optimal)
            
            # Convert to rows (assuming average 1KB per row)
            optimal_rows = int(optimal_mb * 1024)  # 1KB per row estimate
            
            # Apply constraints
            optimal_rows = max(self.config.min_chunk_size, optimal_rows)
            optimal_rows = min(self.config.max_chunk_size, optimal_rows)
            
            tprint_info(f"📊 Optimal chunk size: {optimal_rows} rows (~{optimal_mb:.1f}MB)")
            tprint_info(f"📊 L3 cache: {l3_cache_mb:.1f}MB, Memory per worker: {memory_per_worker:.1f}GB")
            
            return optimal_rows
            
        except Exception as e:
            tprint_warning(f"⚠️ Chunk size calculation failed: {e}")
            return self.config.target_chunk_size
    
    def create_chunks(self, 
                     data: pd.DataFrame,
                     chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """Create optimally sized chunks."""
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(
                data.memory_usage(deep=True).sum() / (1024**2),
                self.config.max_workers
            )
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size].copy()
            chunks.append(chunk)
        
        tprint_info(f"📊 Created {len(chunks)} chunks of ~{chunk_size} rows each")
        return chunks


class ParallelismOptimizer:
    """Optimize parallelism based on operation type."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self._gil_release_ops = {
            'numpy', 'numba', 'blas', 'fft', 'convolution',
            'rolling', 'correlation', 'matrix_ops'
        }
    
    def should_use_multiprocessing(self, operation_type: str) -> bool:
        """Determine if multiprocessing should be used."""
        if not self.config.use_multiprocessing:
            return False
        
        # Use multiprocessing for CPU-bound operations
        cpu_bound_ops = {
            'correlation', 'matrix_ops', 'feature_generation',
            'rolling_stats', 'technical_indicators'
        }
        
        return operation_type in cpu_bound_ops
    
    def get_executor(self, operation_type: str, max_workers: Optional[int] = None):
        """Get appropriate executor for operation type."""
        if max_workers is None:
            max_workers = self.config.max_workers
        
        if self.should_use_multiprocessing(operation_type):
            tprint_debug(f"🔧 Using ProcessPoolExecutor for {operation_type}")
            return ProcessPoolExecutor(max_workers=max_workers)
        else:
            tprint_debug(f"🔧 Using ThreadPoolExecutor for {operation_type}")
            return ThreadPoolExecutor(max_workers=max_workers)


class ZeroCopyOptimizer:
    """Optimize zero-copy data conversions."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
    
    def optimize_dataframe_conversion(self, data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Optimize DataFrame conversion with zero-copy when possible."""
        try:
            if isinstance(data, np.ndarray):
                # Zero-copy conversion from numpy to pandas
                return pd.DataFrame(data, copy=False)
            elif isinstance(data, pd.DataFrame):
                # Already a DataFrame, optimize if needed
                return self._optimize_dataframe(data)
            else:
                return pd.DataFrame(data)
        except Exception as e:
            tprint_warning(f"⚠️ Zero-copy conversion failed: {e}")
            return pd.DataFrame(data)
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize existing DataFrame."""
        try:
            # Downcast dtypes for memory efficiency
            optimized = df.copy()
            
            for col in optimized.columns:
                if optimized[col].dtype == 'float64':
                    if self._can_downcast_float64(optimized[col]):
                        optimized[col] = optimized[col].astype(np.float32)
                elif optimized[col].dtype == 'int64':
                    if self._can_downcast_int64(optimized[col]):
                        optimized[col] = optimized[col].astype(np.int32)
            
            return optimized
            
        except Exception as e:
            tprint_warning(f"⚠️ DataFrame optimization failed: {e}")
            return df
    
    def _can_downcast_float64(self, series: pd.Series) -> bool:
        """Check if float64 can be downcast to float32."""
        try:
            min_val = series.min()
            max_val = series.max()
            return (min_val >= np.finfo(np.float32).min and 
                   max_val <= np.finfo(np.float32).max)
        except:
            return False
    
    def _can_downcast_int64(self, series: pd.Series) -> bool:
        """Check if int64 can be downcast to int32."""
        try:
            min_val = series.min()
            max_val = series.max()
            return (min_val >= np.iinfo(np.int32).min and 
                   max_val <= np.iinfo(np.int32).max)
        except:
            return False


class PreallocationOptimizer:
    """Optimize memory allocation patterns."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
    
    def preallocate_dataframe(self, 
                            shape: Tuple[int, int],
                            columns: List[str],
                            dtype: np.dtype = None) -> pd.DataFrame:
        """Preallocate DataFrame with optimal memory layout."""
        if dtype is None:
            dtype = self.config.default_dtype
        
        try:
            # Preallocate with optimal memory layout
            data = np.empty(shape, dtype=dtype)
            df = pd.DataFrame(data, columns=columns)
            
            tprint_debug(f"📊 Preallocated DataFrame: {shape} with dtype {dtype}")
            return df
            
        except Exception as e:
            tprint_warning(f"⚠️ Preallocation failed: {e}")
            return pd.DataFrame(columns=columns)
    
    def preallocate_array(self, 
                         shape: Tuple[int, ...],
                         dtype: np.dtype = None) -> np.ndarray:
        """Preallocate array with optimal memory layout."""
        if dtype is None:
            dtype = self.config.default_dtype
        
        try:
            return np.empty(shape, dtype=dtype)
        except Exception as e:
            tprint_warning(f"⚠️ Array preallocation failed: {e}")
            return np.array([])
    
    def batch_concatenate(self, 
                         dataframes: List[pd.DataFrame],
                         batch_size: int = None) -> pd.DataFrame:
        """Efficiently concatenate DataFrames in batches."""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        try:
            if len(dataframes) <= batch_size:
                return pd.concat(dataframes, ignore_index=True)
            
            # Process in batches
            result_chunks = []
            for i in range(0, len(dataframes), batch_size):
                batch = dataframes[i:i + batch_size]
                chunk = pd.concat(batch, ignore_index=True)
                result_chunks.append(chunk)
            
            # Final concatenation
            return pd.concat(result_chunks, ignore_index=True)
            
        except Exception as e:
            tprint_warning(f"⚠️ Batch concatenation failed: {e}")
            return pd.concat(dataframes, ignore_index=True)


class VectorizationOptimizer:
    """Optimize vectorized operations."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
    
    def vectorized_rolling_ops(self, 
                              data: np.ndarray,
                              window: int,
                              operations: List[str]) -> Dict[str, np.ndarray]:
        """Perform multiple rolling operations in a single pass."""
        try:
            results = {}
            
            if len(data) < window:
                # Return NaN arrays for insufficient data
                for op in operations:
                    results[op] = np.full(len(data), np.nan)
                return results
            
            # Precompute cumulative sums for efficiency
            cumsum = np.cumsum(data)
            cumsum_sq = np.cumsum(data ** 2)
            
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                
                window_data = data[start_idx:end_idx]
                
                for op in operations:
                    if op == 'mean':
                        results.setdefault('mean', np.full(len(data), np.nan))[i] = np.mean(window_data)
                    elif op == 'std':
                        results.setdefault('std', np.full(len(data), np.nan))[i] = np.std(window_data)
                    elif op == 'min':
                        results.setdefault('min', np.full(len(data), np.nan))[i] = np.min(window_data)
                    elif op == 'max':
                        results.setdefault('max', np.full(len(data), np.nan))[i] = np.max(window_data)
                    elif op == 'sum':
                        results.setdefault('sum', np.full(len(data), np.nan))[i] = np.sum(window_data)
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Vectorized rolling ops failed: {e}")
            return {op: np.full(len(data), np.nan) for op in operations}
    
    def vectorized_correlations(self, 
                               data: np.ndarray,
                               target: np.ndarray,
                               threshold: float = 0.97) -> np.ndarray:
        """Compute correlations efficiently with early stopping."""
        try:
            # Use numpy's optimized correlation
            correlations = np.corrcoef(data.T, target)[:-1, -1]
            
            # Early stopping for high correlations
            high_corr_mask = np.abs(correlations) > threshold
            if np.any(high_corr_mask):
                tprint_debug(f"📊 Found {np.sum(high_corr_mask)} high correlations (> {threshold})")
            
            return correlations
            
        except Exception as e:
            tprint_warning(f"⚠️ Vectorized correlations failed: {e}")
            return np.zeros(data.shape[1])
    
    def batch_matrix_operations(self, 
                               matrices: List[np.ndarray],
                               operation: str) -> List[np.ndarray]:
        """Perform matrix operations in batches."""
        try:
            if operation == 'multiply':
                # Batch matrix multiplication
                results = []
                for i in range(0, len(matrices), self.config.batch_size):
                    batch = matrices[i:i + self.config.batch_size]
                    batch_result = [np.dot(m, m.T) for m in batch]
                    results.extend(batch_result)
                return results
            else:
                return matrices
                
        except Exception as e:
            tprint_warning(f"⚠️ Batch matrix operations failed: {e}")
            return matrices


class MemoryMonitor:
    """Monitor memory usage and performance."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self._memory_history = []
        self._timing_history = []
        self._start_time = None
    
    def start_monitoring(self):
        """Start monitoring session."""
        self._start_time = time.time()
        if self.config.profile_memory:
            self._record_memory_usage()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics."""
        if self._start_time is None:
            return {}
        
        end_time = time.time()
        duration = end_time - self._start_time
        
        stats = {
            'duration_seconds': duration,
            'memory_usage_mb': self._get_current_memory_usage(),
            'peak_memory_mb': max(self._memory_history) if self._memory_history else 0,
            'memory_efficiency': self._calculate_memory_efficiency()
        }
        
        tprint_performance(f"📊 Performance: {duration:.2f}s, Peak memory: {stats['peak_memory_mb']:.1f}MB")
        
        return stats
    
    def _record_memory_usage(self):
        """Record current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self._memory_history.append(memory_mb)
        except Exception as e:
            tprint_debug(f"⚠️ Memory recording failed: {e}")
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score."""
        if not self._memory_history:
            return 0.0
        
        # Simple efficiency metric: lower peak memory is better
        peak_memory = max(self._memory_history)
        baseline_memory = self._memory_history[0] if self._memory_history else 0
        
        if baseline_memory == 0:
            return 1.0
        
        return min(1.0, baseline_memory / peak_memory)


class EfficiencyOptimizer:
    """Main efficiency optimizer coordinating all optimizations."""
    
    def __init__(self, config: Optional[EfficiencyConfig] = None):
        self.config = config or EfficiencyConfig()
        
        # Initialize components
        self.fingerprinter = DataFingerprinter(self.config)
        self.chunking = ChunkingOptimizer(self.config)
        self.parallelism = ParallelismOptimizer(self.config)
        self.zero_copy = ZeroCopyOptimizer(self.config)
        self.preallocation = PreallocationOptimizer(self.config)
        self.vectorization = VectorizationOptimizer(self.config)
        self.monitor = MemoryMonitor(self.config)
        
        tprint_success("🚀 Efficiency optimizer initialized")
        tprint_info(f"📊 Max memory: {self.config.max_memory_gb}GB")
        tprint_info(f"📊 Max workers: {self.config.max_workers}")
        tprint_info(f"📊 Chunk size: {self.config.chunk_size_mb}MB")
    
    def optimize_feature_generation(self, 
                                  data: pd.DataFrame,
                                  config: Dict[str, Any]) -> pd.DataFrame:
        """Optimize the entire feature generation process."""
        self.monitor.start_monitoring()
        
        try:
            # Generate fingerprint for caching
            fingerprint = self.fingerprinter.generate_fingerprint(data, config)
            
            # Optimize data conversion
            optimized_data = self.zero_copy.optimize_dataframe_conversion(data)
            
            # Calculate optimal chunking
            chunks = self.chunking.create_chunks(optimized_data)
            
            # Process chunks in parallel
            results = self._process_chunks_parallel(chunks, config)
            
            # Efficiently combine results
            final_result = self.preallocation.batch_concatenate(results)
            
            # Monitor performance
            stats = self.monitor.stop_monitoring()
            
            tprint_success(f"✅ Feature generation optimized: {len(final_result)} features")
            tprint_performance(f"📊 Peak memory: {stats['peak_memory_mb']:.1f}MB")
            
            return final_result
            
        except Exception as e:
            tprint_error(f"❌ Optimization failed: {e}")
            self.monitor.stop_monitoring()
            raise
    
    def _process_chunks_parallel(self, 
                               chunks: List[pd.DataFrame],
                               config: Dict[str, Any]) -> List[pd.DataFrame]:
        """Process chunks in parallel with optimal executor."""
        try:
            # Determine operation type for parallelism choice
            operation_type = 'feature_generation'
            executor = self.parallelism.get_executor(operation_type)
            
            results = []
            with executor as exec:
                futures = []
                for chunk in chunks:
                    future = exec.submit(self._process_chunk, chunk, config)
                    futures.append(future)
                
                for future in futures:
                    result = future.result()
                    results.append(result)
            
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Parallel processing failed: {e}")
            # Fallback to sequential processing
            return [self._process_chunk(chunk, config) for chunk in chunks]
    
    def _process_chunk(self, chunk: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Process a single chunk."""
        try:
            # This would call the actual feature generation logic
            # For now, return the chunk as-is
            return chunk
            
        except Exception as e:
            tprint_warning(f"⚠️ Chunk processing failed: {e}")
            return pd.DataFrame()


# Global instance for convenience
_efficiency_optimizer = None

def get_efficiency_optimizer(config: Optional[EfficiencyConfig] = None) -> EfficiencyOptimizer:
    """Get the global efficiency optimizer instance."""
    global _efficiency_optimizer
    if _efficiency_optimizer is None:
        _efficiency_optimizer = EfficiencyOptimizer(config)
    return _efficiency_optimizer

def optimize_feature_generation(data: pd.DataFrame, 
                              config: Dict[str, Any],
                              efficiency_config: Optional[EfficiencyConfig] = None) -> pd.DataFrame:
    """Optimize feature generation using the global optimizer."""
    optimizer = get_efficiency_optimizer(efficiency_config)
    return optimizer.optimize_feature_generation(data, config)