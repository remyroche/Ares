"""
Performance Optimizer for Unified Regime Detection

This module provides performance optimization capabilities including caching,
GPU acceleration, and memory optimization for regime detection operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import hashlib
import pickle
from functools import wraps
from pathlib import Path
import logging

# Import tprint for logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import GPU acceleration tools
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceCache:
    """High-performance caching system for regime detection operations."""
    
    def __init__(self, cache_dir: str = "regime_cache", max_size_mb: int = 1024):
        """Initialize performance cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size_mb': 0,
            'entries': 0
        }
        
        tprint_info(f"🗄️ Performance cache initialized: {cache_dir}")
        logger.info(f"Performance cache initialized: {cache_dir}")
    
    def _generate_cache_key(self, data_hash: str, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key from data hash, method, and parameters."""
        param_str = str(sorted(params.items()))
        key_string = f"{data_hash}_{method}_{param_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache entry is valid (not expired)."""
        if not cache_path.exists():
            return False
        
        file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        return file_age_hours <= max_age_hours
    
    def get(self, data_hash: str, method: str, params: Dict[str, Any], 
            max_age_hours: int = 24) -> Optional[Any]:
        """Get cached result if available and valid."""
        cache_key = self._generate_cache_key(data_hash, method, params)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path, max_age_hours):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                
                self.cache_stats['hits'] += 1
                tprint_debug(f"📦 Cache hit: {method}")
                return result
            except Exception as e:
                tprint_warning(f"⚠️ Cache read error: {e}")
                self.cache_stats['misses'] += 1
        else:
            self.cache_stats['misses'] += 1
        
        tprint_debug(f"📭 Cache miss: {method}")
        return None
    
    def set(self, data_hash: str, method: str, params: Dict[str, Any], result: Any) -> bool:
        """Store result in cache."""
        try:
            cache_key = self._generate_cache_key(data_hash, method, params)
            cache_path = self._get_cache_path(cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            self.cache_stats['entries'] += 1
            self.cache_stats['size_mb'] = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl")) / (1024 * 1024)
            
            tprint_debug(f"💾 Cached result: {method}")
            return True
        except Exception as e:
            tprint_error(f"❌ Cache write error: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size_mb': 0,
            'entries': 0
        }
        
        tprint_info("🗑️ Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses'])
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'cache_size_mb': self.cache_stats['size_mb'],
            'total_entries': self.cache_stats['entries']
        }

class GPUAccelerator:
    """GPU acceleration for regime detection operations."""
    
    def __init__(self, enable_gpu: bool = True):
        """Initialize GPU accelerator."""
        self.enable_gpu = enable_gpu and TORCH_AVAILABLE
        self.device = None
        self.gpu_available = False
        
        if self.enable_gpu:
            try:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.gpu_available = torch.cuda.is_available()
                
                if self.gpu_available:
                    tprint_success(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name()}")
                    logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name()}")
                else:
                    tprint_warning("⚠️ CUDA not available, using CPU")
                    logger.warning("CUDA not available, using CPU")
            except Exception as e:
                tprint_warning(f"⚠️ GPU initialization failed: {e}")
                self.gpu_available = False
        else:
            tprint_info("💻 GPU acceleration disabled")
    
    def to_tensor(self, data: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """Convert data to GPU tensor if available."""
        if not self.enable_gpu or not self.gpu_available:
            return torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)
        
        try:
            tensor = torch.tensor(
                data.values if isinstance(data, pd.DataFrame) else data,
                dtype=torch.float32,
                device=self.device
            )
            return tensor
        except Exception as e:
            tprint_warning(f"⚠️ GPU tensor conversion failed: {e}")
            return torch.tensor(data.values if isinstance(data, pd.DataFrame) else data)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor back to numpy array."""
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    
    def accelerate_matrix_operations(self, func: Callable) -> Callable:
        """Decorator to accelerate matrix operations with GPU."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enable_gpu or not self.gpu_available:
                return func(*args, **kwargs)
            
            try:
                # Convert inputs to GPU tensors
                gpu_args = []
                for arg in args:
                    if isinstance(arg, (np.ndarray, pd.DataFrame)):
                        gpu_args.append(self.to_tensor(arg))
                    else:
                        gpu_args.append(arg)
                
                # Execute on GPU
                result = func(*gpu_args, **kwargs)
                
                # Convert result back to numpy
                if isinstance(result, torch.Tensor):
                    return self.to_numpy(result)
                elif isinstance(result, tuple):
                    return tuple(self.to_numpy(r) if isinstance(r, torch.Tensor) else r for r in result)
                else:
                    return result
                    
            except Exception as e:
                tprint_warning(f"⚠️ GPU acceleration failed, falling back to CPU: {e}")
                return func(*args, **kwargs)
        
        return wrapper

class MemoryOptimizer:
    """Memory optimization for large-scale regime detection."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        """Initialize memory optimizer."""
        self.max_memory_gb = max_memory_gb
        self.memory_usage = 0.0
        
        tprint_info(f"💾 Memory optimizer initialized: {max_memory_gb}GB limit")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        original_size = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_type = df[col].dtype
            
            if col_type != np.object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        optimized_size = df.memory_usage(deep=True).sum() / 1024**2  # MB
        reduction = (original_size - optimized_size) / original_size * 100
        
        tprint_debug(f"💾 Memory optimization: {reduction:.1f}% reduction ({original_size:.1f}MB → {optimized_size:.1f}MB)")
        
        return df
    
    def chunk_processing(self, data: Union[np.ndarray, pd.DataFrame], 
                        chunk_size: int = 1000) -> List[Union[np.ndarray, pd.DataFrame]]:
        """Split data into chunks for memory-efficient processing."""
        if len(data) <= chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        
        tprint_debug(f"📦 Split data into {len(chunks)} chunks of size {chunk_size}")
        return chunks
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_usage_gb = memory_info.rss / 1024**3
            
            return {
                'memory_usage_gb': memory_usage_gb,
                'memory_limit_gb': self.max_memory_gb,
                'memory_usage_percent': (memory_usage_gb / self.max_memory_gb) * 100
            }
        except ImportError:
            return {'error': 'psutil not available'}

class PerformanceOptimizer:
    """Main performance optimizer combining all optimization strategies."""
    
    def __init__(self, enable_cache: bool = True, enable_gpu: bool = True, 
                 max_memory_gb: float = 8.0):
        """Initialize performance optimizer."""
        self.cache = PerformanceCache() if enable_cache else None
        self.gpu_accelerator = GPUAccelerator(enable_gpu)
        self.memory_optimizer = MemoryOptimizer(max_memory_gb)
        
        tprint_success("⚡ Performance optimizer initialized")
        logger.info("Performance optimizer initialized")
    
    def optimize_regime_detection(self, func: Callable) -> Callable:
        """Decorator to optimize regime detection functions."""
        @wraps(func)
        def wrapper(self, market_data, *args, **kwargs):
            start_time = time.time()
            
            # Generate data hash for caching
            data_hash = self._generate_data_hash(market_data)
            
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(data_hash, func.__name__, kwargs)
                if cached_result is not None:
                    execution_time = time.time() - start_time
                    tprint_performance(f"⚡ Cache hit for {func.__name__}: {execution_time:.3f}s")
                    return cached_result
            
            # Optimize memory usage
            if hasattr(market_data, 'memory_usage'):
                market_data = self.memory_optimizer.optimize_dataframe(market_data)
            
            # Monitor memory usage
            memory_before = self.memory_optimizer.monitor_memory()
            
            # Execute function with GPU acceleration if applicable
            if hasattr(self.gpu_accelerator, 'accelerate_matrix_operations'):
                func = self.gpu_accelerator.accelerate_matrix_operations(func)
            
            # Execute the function
            result = func(self, market_data, *args, **kwargs)
            
            # Cache result
            if self.cache:
                self.cache.set(data_hash, func.__name__, kwargs, result)
            
            # Monitor memory usage after
            memory_after = self.memory_optimizer.monitor_memory()
            
            execution_time = time.time() - start_time
            tprint_performance(f"⚡ {func.__name__} completed: {execution_time:.3f}s")
            
            if 'memory_usage_gb' in memory_before and 'memory_usage_gb' in memory_after:
                memory_delta = memory_after['memory_usage_gb'] - memory_before['memory_usage_gb']
                tprint_debug(f"💾 Memory usage: {memory_delta:+.2f}GB")
            
            return result
        
        return wrapper
    
    def _generate_data_hash(self, data: Union[np.ndarray, pd.DataFrame]) -> str:
        """Generate hash for data caching."""
        if isinstance(data, pd.DataFrame):
            # Use first and last few rows for hash to handle large datasets
            sample_data = pd.concat([data.head(5), data.tail(5)])
            data_str = sample_data.to_string()
        else:
            # Use first and last few elements for numpy arrays
            sample_data = np.concatenate([data[:5], data[-5:]])
            data_str = str(sample_data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'gpu_available': self.gpu_accelerator.gpu_available,
            'gpu_enabled': self.gpu_accelerator.enable_gpu,
            'cache_enabled': self.cache is not None,
            'memory_limit_gb': self.memory_optimizer.max_memory_gb
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        memory_stats = self.memory_optimizer.monitor_memory()
        if 'error' not in memory_stats:
            stats['memory_stats'] = memory_stats
        
        return stats
    
    def clear_cache(self):
        """Clear performance cache."""
        if self.cache:
            self.cache.clear()
    
    def optimize_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Optimize data for processing."""
        if isinstance(data, pd.DataFrame):
            return self.memory_optimizer.optimize_dataframe(data)
        else:
            return data

# Global performance optimizer instance
_global_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

def optimize_performance(enable_cache: bool = True, enable_gpu: bool = True, 
                        max_memory_gb: float = 8.0):
    """Decorator to optimize any regime detection function."""
    def decorator(func: Callable) -> Callable:
        optimizer = get_performance_optimizer()
        return optimizer.optimize_regime_detection(func)
    return decorator