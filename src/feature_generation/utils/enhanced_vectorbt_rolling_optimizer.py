"""
Enhanced VectorBT Rolling Operations Optimizer

This module provides enhanced rolling operations using VectorBT's high-performance
functions with adaptive chunking, advanced memory management, and M1 GPU optimization.

Key Features:
- Adaptive chunking based on memory pressure and data characteristics
- Advanced multi-level caching strategies
- Mac M1 GPU optimization with Metal Performance Shaders
- Backward compatibility with existing VectorBTRollingOptimizer
- Intelligent memory management and resource pooling
"""

import numpy as np
import pandas as pd
import logging
import psutil
import gc
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import warnings
from functools import wraps
import time
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
import os
from pathlib import Path

# Enhanced logging with tprint
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions for when tprint is not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)
    def tprint_timer(*args, **kwargs): print("TIMER:", *args, **kwargs)

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# M1 GPU optimization imports
try:
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        M1_GPU_AVAILABLE = True
        tprint_info("🍎 Mac M1 GPU (Metal Performance Shaders) detected")
    else:
        M1_GPU_AVAILABLE = False
        tprint_info("🍎 Mac M1 detected but MPS not available")
except ImportError:
    TORCH_AVAILABLE = False
    M1_GPU_AVAILABLE = False
    tprint_warning("⚠️ PyTorch not available for M1 GPU optimization")

# Optional GPU acceleration (CUDA)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

# Enhanced error handling with fast failing
class VectorBTOptimizationError(Exception):
    """Custom exception for VectorBT optimization errors with detailed context."""
    def __init__(self, message: str, operation: str = None, data_shape: tuple = None, 
                 window: int = None, strategy: str = None, original_error: Exception = None):
        self.operation = operation
        self.data_shape = data_shape
        self.window = window
        self.strategy = strategy
        self.original_error = original_error
        
        # Build detailed error message
        context_parts = []
        if operation:
            context_parts.append(f"Operation: {operation}")
        if data_shape:
            context_parts.append(f"Data shape: {data_shape}")
        if window:
            context_parts.append(f"Window: {window}")
        if strategy:
            context_parts.append(f"Strategy: {strategy}")
        
        context_str = ", ".join(context_parts)
        full_message = f"{message}"
        if context_str:
            full_message += f" (Context: {context_str})"
        if original_error:
            full_message += f" (Original: {str(original_error)})"
            
        super().__init__(full_message)

class VectorBTValidationError(Exception):
    """Custom exception for VectorBT validation errors."""
    def __init__(self, message: str, validation_type: str = None, value: Any = None):
        self.validation_type = validation_type
        self.value = value
        full_message = f"{message}"
        if validation_type:
            full_message += f" (Validation: {validation_type})"
        if value is not None:
            full_message += f" (Value: {value})"
        super().__init__(full_message)

class MemoryPressureLevel(Enum):
    """Memory pressure levels for adaptive chunking."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CacheLevel(Enum):
    """Cache levels for multi-level caching."""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_DISTRIBUTED = "l3_distributed"

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_gb: float = 8.0
    memory_pressure_threshold: float = 0.8
    adaptive_chunking: bool = True
    memory_pooling: bool = True
    gc_frequency: int = 100  # Run GC every N operations
    memory_monitoring: bool = True

@dataclass
class CacheConfig:
    """Configuration for caching strategies."""
    l1_cache_size: int = 1000
    l2_cache_size: int = 10000
    l2_cache_dir: str = "./cache"
    cache_ttl: float = 3600.0  # 1 hour
    cache_compression: bool = True
    cache_encryption: bool = False
    distributed_cache_url: Optional[str] = None

@dataclass
class ChunkingStrategy:
    """Strategy for adaptive chunking."""
    base_chunk_size: int
    memory_multiplier: float
    data_type_multiplier: float
    operation_complexity: float
    gpu_available: bool

class MemoryManager:
    """Advanced memory management with adaptive chunking."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_pool = {}
        self.memory_stats = {
            'allocations': 0,
            'deallocations': 0,
            'peak_usage': 0,
            'current_usage': 0,
            'gc_runs': 0
        }
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._monitoring = False
        
        if self.config.memory_monitoring:
            self._start_memory_monitoring()
    
    def get_memory_pressure(self) -> MemoryPressureLevel:
        """Get current memory pressure level."""
        try:
            memory = psutil.virtual_memory()
            usage_ratio = memory.used / memory.total
            
            if usage_ratio < 0.5:
                return MemoryPressureLevel.LOW
            elif usage_ratio < 0.7:
                return MemoryPressureLevel.MEDIUM
            elif usage_ratio < 0.9:
                return MemoryPressureLevel.HIGH
            else:
                return MemoryPressureLevel.CRITICAL
        except Exception as e:
            tprint_warning(f"⚠️ Failed to get memory pressure: {e}")
            return MemoryPressureLevel.MEDIUM
    
    def calculate_optimal_chunk_size(self, data_size: int, data_dtype: np.dtype, 
                                   operation_complexity: float = 1.0) -> int:
        """Calculate optimal chunk size based on memory pressure and data characteristics."""
        with self._lock:
            pressure = self.get_memory_pressure()
            
            # Base chunk size calculation
            base_size = min(data_size, 1000)  # Start with 1000
            
            # Adjust based on memory pressure
            pressure_multipliers = {
                MemoryPressureLevel.LOW: 2.0,
                MemoryPressureLevel.MEDIUM: 1.5,
                MemoryPressureLevel.HIGH: 1.0,
                MemoryPressureLevel.CRITICAL: 0.5
            }
            
            # Adjust based on data type size
            dtype_size = np.dtype(data_dtype).itemsize
            dtype_multiplier = 8.0 / dtype_size  # Normalize to float64
            
            # Adjust based on operation complexity
            complexity_multiplier = 1.0 / max(operation_complexity, 0.1)
            
            # Calculate final chunk size
            optimal_size = int(base_size * 
                             pressure_multipliers[pressure] * 
                             dtype_multiplier * 
                             complexity_multiplier)
            
            # Ensure reasonable bounds
            optimal_size = max(100, min(optimal_size, data_size))
            
            tprint_debug(f"🧠 Calculated optimal chunk size: {optimal_size} (pressure: {pressure.value}, dtype: {data_dtype})")
            return optimal_size
    
    def allocate_memory(self, size: int, dtype: np.dtype) -> np.ndarray:
        """Allocate memory with pooling if enabled."""
        if self.config.memory_pooling:
            pool_key = (size, dtype)
            if pool_key in self.memory_pool and len(self.memory_pool[pool_key]) > 0:
                array = self.memory_pool[pool_key].pop()
                tprint_debug(f"♻️ Reused memory from pool: {size} {dtype}")
                return array
        
        array = np.empty(size, dtype=dtype)
        self.memory_stats['allocations'] += 1
        self.memory_stats['current_usage'] += array.nbytes
        self.memory_stats['peak_usage'] = max(self.memory_stats['peak_usage'], 
                                            self.memory_stats['current_usage'])
        
        tprint_debug(f"📦 Allocated memory: {size} {dtype} ({array.nbytes / 1024**2:.2f} MB)")
        return array
    
    def deallocate_memory(self, array: np.ndarray, pool: bool = True):
        """Deallocate memory, optionally returning to pool."""
        if self.config.memory_pooling and pool:
            pool_key = (array.size, array.dtype)
            if pool_key not in self.memory_pool:
                self.memory_pool[pool_key] = []
            
            # Limit pool size to prevent memory bloat
            if len(self.memory_pool[pool_key]) < 10:
                self.memory_pool[pool_key].append(array)
                tprint_debug(f"♻️ Returned memory to pool: {array.size} {array.dtype}")
                return
        
        del array
        self.memory_stats['deallocations'] += 1
        self.memory_stats['current_usage'] -= array.nbytes
        tprint_debug(f"🗑️ Deallocated memory: {array.size} {array.dtype}")
    
    def run_gc_if_needed(self):
        """Run garbage collection if needed."""
        if self.memory_stats['allocations'] % self.config.gc_frequency == 0:
            gc.collect()
            self.memory_stats['gc_runs'] += 1
            tprint_debug("🧹 Ran garbage collection")
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        tprint_info("📊 Started memory monitoring")
    
    def _monitor_memory(self):
        """Background memory monitoring thread."""
        while self._monitoring:
            try:
                pressure = self.get_memory_pressure()
                if pressure == MemoryPressureLevel.CRITICAL:
                    tprint_warning("🚨 Critical memory pressure detected, running cleanup")
                    self.cleanup_pools()
                    gc.collect()
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                tprint_error(f"❌ Memory monitoring error: {e}")
                time.sleep(10)
    
    def cleanup_pools(self):
        """Clean up memory pools."""
        with self._lock:
            for pool_key, pool in self.memory_pool.items():
                for array in pool:
                    del array
                pool.clear()
            tprint_info("🧹 Cleaned up memory pools")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        return {
            'memory_stats': self.memory_stats.copy(),
            'pool_sizes': {str(k): len(v) for k, v in self.memory_pool.items()},
            'current_pressure': self.get_memory_pressure().value
        }
    
    def cleanup(self):
        """Clean up memory manager."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        self.cleanup_pools()
        tprint_info("🧹 Memory manager cleaned up")

class AdvancedCacheManager:
    """Advanced multi-level cache manager."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.l1_cache = {}  # In-memory cache
        self.l2_cache_dir = Path(config.l2_cache_dir)
        self.l2_cache_dir.mkdir(exist_ok=True)
        
        # Cache statistics
        self.cache_stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Access tracking for LRU
        self.access_times = {}
        self._lock = threading.Lock()
    
    def _generate_cache_key(self, data_hash: str, operation: str, 
                          window: int, **kwargs) -> str:
        """Generate cache key for operation."""
        params_str = str(sorted(kwargs.items()))
        key_string = f"{data_hash}_{operation}_{window}_{params_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - timestamp < self.config.cache_ttl
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self.cache_stats['total_requests'] += 1
            
            # Try L1 cache first
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                if self._is_cache_valid(entry['timestamp']):
                    self.cache_stats['l1_hits'] += 1
                    self.access_times[cache_key] = time.time()
                    tprint_debug(f"💾 L1 cache hit: {cache_key[:8]}...")
                    return entry['data']
                else:
                    # Remove expired entry
                    del self.l1_cache[cache_key]
                    self.cache_stats['evictions'] += 1
            
            # Try L2 cache
            l2_path = self.l2_cache_dir / f"{cache_key}.pkl"
            if l2_path.exists():
                try:
                    with open(l2_path, 'rb') as f:
                        entry = pickle.load(f)
                    if self._is_cache_valid(entry['timestamp']):
                        self.cache_stats['l2_hits'] += 1
                        # Promote to L1 cache
                        self._put_l1(cache_key, entry['data'])
                        tprint_debug(f"💾 L2 cache hit: {cache_key[:8]}...")
                        return entry['data']
                    else:
                        # Remove expired file
                        l2_path.unlink()
                except Exception as e:
                    tprint_warning(f"⚠️ L2 cache read error: {e}")
            
            self.cache_stats['l1_misses'] += 1
            self.cache_stats['l2_misses'] += 1
            return None
    
    def put(self, cache_key: str, data: Any):
        """Put value in cache."""
        with self._lock:
            # Put in L1 cache
            self._put_l1(cache_key, data)
            
            # Put in L2 cache if enabled
            if self.config.l2_cache_size > 0:
                self._put_l2(cache_key, data)
    
    def _put_l1(self, cache_key: str, data: Any):
        """Put value in L1 cache with LRU eviction."""
        # Evict if cache is full
        if len(self.l1_cache) >= self.config.l1_cache_size:
            self._evict_lru_l1()
        
        self.l1_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        self.access_times[cache_key] = time.time()
        tprint_debug(f"💾 Stored in L1 cache: {cache_key[:8]}...")
    
    def _put_l2(self, cache_key: str, data: Any):
        """Put value in L2 cache."""
        try:
            l2_path = self.l2_cache_dir / f"{cache_key}.pkl"
            entry = {
                'data': data,
                'timestamp': time.time()
            }
            
            with open(l2_path, 'wb') as f:
                pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            tprint_debug(f"💾 Stored in L2 cache: {cache_key[:8]}...")
        except Exception as e:
            tprint_warning(f"⚠️ L2 cache write error: {e}")
    
    def _evict_lru_l1(self):
        """Evict least recently used item from L1 cache."""
        if not self.access_times:
            # Remove arbitrary item if no access times
            key_to_remove = next(iter(self.l1_cache))
        else:
            key_to_remove = min(self.access_times.keys(), 
                              key=lambda k: self.access_times[k])
        
        if key_to_remove in self.l1_cache:
            del self.l1_cache[key_to_remove]
            if key_to_remove in self.access_times:
                del self.access_times[key_to_remove]
            self.cache_stats['evictions'] += 1
            tprint_debug(f"🗑️ Evicted from L1 cache: {key_to_remove[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['total_requests']
        l1_hit_rate = (self.cache_stats['l1_hits'] / total_requests * 100) if total_requests > 0 else 0
        l2_hit_rate = (self.cache_stats['l2_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_stats': self.cache_stats.copy(),
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'l1_size': len(self.l1_cache),
            'l2_size': len(list(self.l2_cache_dir.glob("*.pkl")))
        }
    
    def cleanup(self):
        """Clean up cache manager."""
        with self._lock:
            self.l1_cache.clear()
            self.access_times.clear()
            tprint_info("🧹 Cache manager cleaned up")

class M1GPUOptimizer:
    """Mac M1 GPU optimization using Metal Performance Shaders."""
    
    def __init__(self):
        self.device = None
        self.available = M1_GPU_AVAILABLE
        
        if self.available:
            try:
                self.device = torch.device("mps")
                tprint_success("🍎 M1 GPU optimizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ M1 GPU initialization failed: {e}")
                self.available = False
    
    def can_optimize(self, data_size: int, operation_complexity: float = 1.0) -> bool:
        """Check if M1 GPU optimization is beneficial."""
        if not self.available:
            return False
        
        # Use M1 GPU for larger datasets or complex operations
        return data_size > 1000 or operation_complexity > 2.0
    
    def rolling_mean_m1(self, data: Union[pd.Series, pd.DataFrame], window: int) -> Union[pd.Series, pd.DataFrame]:
        """M1 GPU optimized rolling mean."""
        if not self.available:
            raise VectorBTOptimizationError("M1 GPU not available")
        
        try:
            # Convert to torch tensor
            if isinstance(data, pd.Series):
                tensor = torch.tensor(data.values, dtype=torch.float32, device=self.device)
                result = self._rolling_mean_tensor(tensor, window)
                return pd.Series(result.cpu().numpy(), index=data.index, name=data.name)
            else:
                tensor = torch.tensor(data.values, dtype=torch.float32, device=self.device)
                result = self._rolling_mean_tensor(tensor, window)
                return pd.DataFrame(result.cpu().numpy(), index=data.index, columns=data.columns)
        except Exception as e:
            raise VectorBTOptimizationError(f"M1 GPU rolling mean failed: {e}")
    
    def rolling_std_m1(self, data: Union[pd.Series, pd.DataFrame], window: int) -> Union[pd.Series, pd.DataFrame]:
        """M1 GPU optimized rolling standard deviation."""
        if not self.available:
            raise VectorBTOptimizationError("M1 GPU not available")
        
        try:
            if isinstance(data, pd.Series):
                tensor = torch.tensor(data.values, dtype=torch.float32, device=self.device)
                result = self._rolling_std_tensor(tensor, window)
                return pd.Series(result.cpu().numpy(), index=data.index, name=data.name)
            else:
                tensor = torch.tensor(data.values, dtype=torch.float32, device=self.device)
                result = self._rolling_std_tensor(tensor, window)
                return pd.DataFrame(result.cpu().numpy(), index=data.index, columns=data.columns)
        except Exception as e:
            raise VectorBTOptimizationError(f"M1 GPU rolling std failed: {e}")
    
    def _rolling_mean_tensor(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        """Compute rolling mean on M1 GPU tensor."""
        # Use unfold for efficient rolling operations
        if tensor.dim() == 1:
            # 1D case
            unfolded = tensor.unfold(0, window, 1)
            return unfolded.mean(dim=1)
        else:
            # 2D case
            unfolded = tensor.unfold(0, window, 1)
            return unfolded.mean(dim=1)
    
    def _rolling_std_tensor(self, tensor: torch.Tensor, window: int) -> torch.Tensor:
        """Compute rolling standard deviation on M1 GPU tensor."""
        if tensor.dim() == 1:
            # 1D case
            unfolded = tensor.unfold(0, window, 1)
            return unfolded.std(dim=1)
        else:
            # 2D case
            unfolded = tensor.unfold(0, window, 1)
            return unfolded.std(dim=1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get M1 GPU optimization statistics."""
        return {
            'available': self.available,
            'device': str(self.device) if self.device else None
        }

class EnhancedVectorBTRollingOptimizer:
    """
    Enhanced VectorBT rolling optimizer with adaptive chunking, advanced caching,
    and M1 GPU optimization while maintaining backward compatibility.
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True, 
                 memory_efficient: bool = True, chunk_size: int = 1000, 
                 fast_fail: bool = True, enable_logging: bool = True,
                 # New enhanced parameters
                 memory_config: Optional[MemoryConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 enable_m1_gpu: bool = True,
                 enable_adaptive_chunking: bool = True,
                 enable_advanced_caching: bool = True):
        """
        Initialize enhanced VectorBT rolling optimizer.
        
        Args:
            enable_gpu: Enable GPU acceleration if available (backward compatible)
            enable_parallel: Enable parallel processing (backward compatible)
            memory_efficient: Enable memory optimization (backward compatible)
            chunk_size: Size of data chunks for processing (backward compatible)
            fast_fail: Enable fast failing instead of silent fallbacks (backward compatible)
            enable_logging: Enable comprehensive logging with tprint (backward compatible)
            memory_config: Advanced memory management configuration
            cache_config: Advanced caching configuration
            enable_m1_gpu: Enable Mac M1 GPU optimization
            enable_adaptive_chunking: Enable adaptive chunking
            enable_advanced_caching: Enable advanced caching
        """
        tprint_info("🚀 Initializing Enhanced VectorBTRollingOptimizer with advanced features")
        
        # Backward compatibility - maintain original parameters
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.use_vectorbt = VECTORBT_AVAILABLE
        self.fast_fail = fast_fail
        self.enable_logging = enable_logging
        
        # Enhanced features
        self.enable_m1_gpu = enable_m1_gpu
        self.enable_adaptive_chunking = enable_adaptive_chunking
        self.enable_advanced_caching = enable_advanced_caching
        
        # Initialize enhanced components
        self.memory_config = memory_config or MemoryConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize memory manager
        if self.enable_adaptive_chunking:
            self.memory_manager = MemoryManager(self.memory_config)
            tprint_success("✅ Memory manager initialized")
        else:
            self.memory_manager = None
        
        # Initialize cache manager
        if self.enable_advanced_caching:
            self.cache_manager = AdvancedCacheManager(self.cache_config)
            tprint_success("✅ Cache manager initialized")
        else:
            self.cache_manager = None
        
        # Initialize M1 GPU optimizer
        if self.enable_m1_gpu:
            self.m1_optimizer = M1GPUOptimizer()
            if self.m1_optimizer.available:
                tprint_success("✅ M1 GPU optimizer initialized")
            else:
                tprint_warning("⚠️ M1 GPU optimizer not available")
        else:
            self.m1_optimizer = None
        
        # Enhanced performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'numpy_fallbacks': 0,
            'gpu_operations': 0,
            'm1_gpu_operations': 0,
            'memory_optimizations': 0,
            'chunk_operations': 0,
            'adaptive_chunk_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_operations': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }
        
        # Configure VectorBT settings with error handling
        try:
            if self.use_vectorbt:
                tprint_info("🔧 Configuring VectorBT settings")
                vbt.settings.parallel['enabled'] = self.enable_parallel
                if self.enable_gpu:
                    vbt.settings.array_wrapper['freq'] = '1min'
                tprint_success("✅ VectorBT settings configured successfully")
            else:
                tprint_warning("⚠️ VectorBT not available, using fallback methods")
        except Exception as e:
            error_msg = f"Failed to configure VectorBT settings: {e}"
            tprint_error(error_msg)
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, strategy="initialization", original_error=e)
            else:
                tprint_warning("⚠️ Continuing with fallback configuration")
        
        tprint_success(f"✅ Enhanced VectorBTRollingOptimizer initialized: VectorBT={self.use_vectorbt}, GPU={self.enable_gpu}, M1GPU={self.enable_m1_gpu}, AdaptiveChunking={self.enable_adaptive_chunking}, AdvancedCaching={self.enable_advanced_caching}")
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling mean calculation with adaptive chunking and caching."""
        tprint_debug(f"🔄 Starting enhanced rolling mean calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs
        self._validate_rolling_inputs(data, window, 'mean')
        
        # Check cache first
        if self.cache_manager:
            cache_key = self._generate_cache_key(data, 'mean', window, **kwargs)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                tprint_debug("💾 Cache hit for rolling mean")
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        try:
            result = self._enhanced_rolling_operation(data, 'mean', window, **kwargs)
            
            # Cache result
            if self.cache_manager:
                self.cache_manager.put(cache_key, result)
            
            tprint_success(f"✅ Enhanced rolling mean completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Enhanced rolling mean calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='mean', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_mean(data, window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling standard deviation calculation."""
        tprint_debug(f"🔄 Starting enhanced rolling std calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        self._validate_rolling_inputs(data, window, 'std')
        
        # Check cache first
        if self.cache_manager:
            cache_key = self._generate_cache_key(data, 'std', window, **kwargs)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                tprint_debug("💾 Cache hit for rolling std")
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        try:
            result = self._enhanced_rolling_operation(data, 'std', window, **kwargs)
            
            # Cache result
            if self.cache_manager:
                self.cache_manager.put(cache_key, result)
            
            tprint_success(f"✅ Enhanced rolling std completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Enhanced rolling std calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='std', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_std(data, window, **kwargs)
    
    # Maintain all other rolling methods with enhanced implementations
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling variance calculation."""
        return self._enhanced_rolling_operation(data, 'var', window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling minimum calculation."""
        return self._enhanced_rolling_operation(data, 'min', window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling maximum calculation."""
        return self._enhanced_rolling_operation(data, 'max', window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling sum calculation."""
        return self._enhanced_rolling_operation(data, 'sum', window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling quantile calculation."""
        return self._enhanced_rolling_operation(data, 'quantile', window, q=q, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling skewness calculation."""
        return self._enhanced_rolling_operation(data, 'skew', window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling kurtosis calculation."""
        return self._enhanced_rolling_operation(data, 'kurt', window, **kwargs)
    
    def rolling_corr(self, data: Union[pd.Series, pd.DataFrame], other: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling correlation calculation."""
        return self._enhanced_rolling_operation(data, 'corr', window, other=other, **kwargs)
    
    def rolling_cov(self, data: Union[pd.Series, pd.DataFrame], other: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling covariance calculation."""
        return self._enhanced_rolling_operation(data, 'cov', window, other=other, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], func: callable, 
                     window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling apply calculation."""
        return self._enhanced_rolling_operation(data, 'apply', window, func=func, **kwargs)
    
    def rolling_median(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling median calculation."""
        return self.rolling_quantile(data, window, q=0.5, **kwargs)
    
    def rolling_percentile(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                          percentile: float, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling percentile calculation."""
        return self.rolling_quantile(data, window, q=percentile/100, **kwargs)
    
    def rolling_rank(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced rolling rank calculation."""
        return self._enhanced_rolling_operation(data, 'rank', window, **kwargs)
    
    def rolling_ewm(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                   alpha: float = None, span: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced exponentially weighted moving average."""
        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).mean()
        elif span is not None:
            return data.ewm(span=span, **kwargs).mean()
        else:
            return data.ewm(span=window, **kwargs).mean()
    
    def rolling_ewm_std(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                       alpha: float = None, span: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced exponentially weighted moving standard deviation."""
        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).std()
        elif span is not None:
            return data.ewm(span=span, **kwargs).std()
        else:
            return data.ewm(span=window, **kwargs).std()
    
    def rolling_ewm_var(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                       alpha: float = None, span: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced exponentially weighted moving variance."""
        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).var()
        elif span is not None:
            return data.ewm(span=span, **kwargs).var()
        else:
            return data.ewm(span=window, **kwargs).var()
    
    def rolling_correlation_matrix(self, data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
        """Enhanced rolling correlation matrix calculation."""
        if not self.use_vectorbt:
            return self._fallback_rolling_correlation_matrix(data, window, **kwargs)
        
        try:
            result = rolling_corr(data, window=window, **kwargs)
            self.performance_stats['vectorbt_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"VectorBT rolling correlation matrix failed: {e}, using fallback")
            return self._fallback_rolling_correlation_matrix(data, window, **kwargs)
    
    def rolling_covariance_matrix(self, data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
        """Enhanced rolling covariance matrix calculation."""
        if not self.use_vectorbt:
            return self._fallback_rolling_covariance_matrix(data, window, **kwargs)
        
        try:
            result = rolling_cov(data, window=window, **kwargs)
            self.performance_stats['vectorbt_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"VectorBT rolling covariance matrix failed: {e}, using fallback")
            return self._fallback_rolling_covariance_matrix(data, window, **kwargs)
    
    def _enhanced_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                  window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Enhanced rolling operation with adaptive chunking, caching, and M1 GPU optimization.
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        tprint_debug(f"🔄 Starting enhanced rolling operation: {operation}, window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs before processing
        self._validate_rolling_inputs(data, window, operation)
        
        # Optimize data for processing
        if self.memory_efficient:
            tprint_debug("🧠 Optimizing data types for memory efficiency")
            try:
                data = self._optimize_data_types(data)
                tprint_success("✅ Data type optimization completed")
            except Exception as e:
                error_msg = f"Data type optimization failed: {e}"
                tprint_warning(f"⚠️ {error_msg}")
                if self.fast_fail:
                    raise VectorBTOptimizationError(error_msg, operation=operation, original_error=e)
        
        try:
            # Check if M1 GPU optimization is beneficial
            if (self.m1_optimizer and self.m1_optimizer.can_optimize(len(data), self._get_operation_complexity(operation))):
                tprint_debug("🍎 Using M1 GPU optimization")
                if operation == 'mean':
                    result = self.m1_optimizer.rolling_mean_m1(data, window)
                elif operation == 'std':
                    result = self.m1_optimizer.rolling_std_m1(data, window)
                else:
                    # Fallback to other methods for unsupported M1 operations
                    result = self._adaptive_chunked_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['m1_gpu_operations'] += 1
                tprint_success("✅ M1 GPU processing completed")
            
            # Check if data is large enough for adaptive chunked processing
            elif len(data) > self.chunk_size and self.enable_adaptive_chunking:
                tprint_info(f"📦 Using adaptive chunked processing: data_size={len(data)}")
                result = self._adaptive_chunked_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['adaptive_chunk_operations'] += 1
                tprint_success("✅ Adaptive chunked processing completed")
            
            # Check if data is large enough for regular chunked processing
            elif len(data) > self.chunk_size and self.memory_efficient:
                tprint_info(f"📦 Using chunked processing: data_size={len(data)}, chunk_size={self.chunk_size}")
                result = self._chunked_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['chunk_operations'] += 1
                tprint_success("✅ Chunked processing completed")
            else:
                # Determine optimal processing method
                strategy = self._select_processing_strategy(data, window, operation)
                tprint_debug(f"🎯 Selected processing strategy: {strategy}")
                
                if strategy == 'vectorbt':
                    result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    tprint_success("✅ VectorBT processing completed")
                elif strategy == 'gpu':
                    result = self._gpu_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['gpu_operations'] += 1
                    tprint_success("✅ GPU processing completed")
                else:
                    result = self._pandas_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['pandas_fallbacks'] += 1
                    tprint_success("✅ Pandas processing completed")
            
            # Update timing and validate result
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            
            # Validate result
            self._validate_rolling_result(result, operation, window)
            
            tprint_performance(f"Enhanced rolling {operation}", execution_time)
            return result
            
        except Exception as e:
            error_msg = f"Enhanced rolling operation {operation} failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise VectorBTOptimizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting numpy fallback")
                try:
                    result = self._numpy_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['numpy_fallbacks'] += 1
                    tprint_success("✅ Numpy fallback completed")
                    return result
                except Exception as fallback_error:
                    error_msg = f"All rolling operation methods failed for {operation}"
                    tprint_error(f"❌ {error_msg}: {fallback_error}")
                    raise VectorBTOptimizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=fallback_error)
    
    def _adaptive_chunked_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Process data with adaptive chunking based on memory pressure."""
        if not self.memory_manager:
            return self._chunked_rolling_operation(data, operation, window, **kwargs)
        
        # Calculate optimal chunk size
        data_dtype = data.dtypes.iloc[0] if isinstance(data, pd.DataFrame) else data.dtype
        optimal_chunk_size = self.memory_manager.calculate_optimal_chunk_size(
            len(data), data_dtype, self._get_operation_complexity(operation)
        )
        
        tprint_debug(f"🧠 Using adaptive chunk size: {optimal_chunk_size}")
        
        if isinstance(data, pd.Series):
            return self._adaptive_chunked_series_operation(data, operation, window, optimal_chunk_size, **kwargs)
        else:
            return self._adaptive_chunked_dataframe_operation(data, operation, window, optimal_chunk_size, **kwargs)
    
    def _adaptive_chunked_series_operation(self, data: pd.Series, operation: str, 
                                         window: int, chunk_size: int, **kwargs) -> pd.Series:
        """Process Series with adaptive chunking."""
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap for rolling window
            
            # Allocate memory for chunk processing
            if self.memory_manager:
                chunk_array = self.memory_manager.allocate_memory(len(chunk), chunk.dtype)
                chunk_array[:] = chunk.values
                chunk = pd.Series(chunk_array, index=chunk.index, name=chunk.name)
            
            try:
                if self._should_use_vectorbt(chunk, window):
                    chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                elif self._should_use_gpu(chunk, window):
                    chunk_result = self._gpu_rolling_operation(chunk, operation, window, **kwargs)
                    self.performance_stats['gpu_operations'] += 1
                else:
                    chunk_result = self._pandas_rolling_operation(chunk, operation, window, **kwargs)
                    self.performance_stats['pandas_fallbacks'] += 1
                
                # Remove overlap from result (except for first chunk)
                if i == 0:
                    results.append(chunk_result)
                else:
                    results.append(chunk_result.iloc[window-1:])
            
            finally:
                # Deallocate memory
                if self.memory_manager:
                    self.memory_manager.deallocate_memory(chunk.values)
                    self.memory_manager.run_gc_if_needed()
        
        return pd.concat(results, ignore_index=False)
    
    def _adaptive_chunked_dataframe_operation(self, data: pd.DataFrame, operation: str, 
                                            window: int, chunk_size: int, **kwargs) -> pd.DataFrame:
        """Process DataFrame with adaptive chunking."""
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap for rolling window
            
            # Allocate memory for chunk processing
            if self.memory_manager:
                chunk_array = self.memory_manager.allocate_memory(chunk.values.size, chunk.values.dtype)
                chunk_array.flat[:] = chunk.values.flat
                chunk = pd.DataFrame(chunk_array, index=chunk.index, columns=chunk.columns)
            
            try:
                if self._should_use_vectorbt(chunk, window):
                    chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                elif self._should_use_gpu(chunk, window):
                    chunk_result = self._gpu_rolling_operation(chunk, operation, window, **kwargs)
                    self.performance_stats['gpu_operations'] += 1
                else:
                    chunk_result = self._pandas_rolling_operation(chunk, operation, window, **kwargs)
                    self.performance_stats['pandas_fallbacks'] += 1
                
                # Remove overlap from result (except for first chunk)
                if i == 0:
                    results.append(chunk_result)
                else:
                    results.append(chunk_result.iloc[window-1:])
            
            finally:
                # Deallocate memory
                if self.memory_manager:
                    self.memory_manager.deallocate_memory(chunk.values)
                    self.memory_manager.run_gc_if_needed()
        
        return pd.concat(results, ignore_index=False)
    
    def _get_operation_complexity(self, operation: str) -> float:
        """Get operation complexity for adaptive chunking."""
        complexity_map = {
            'mean': 1.0,
            'sum': 1.0,
            'min': 1.0,
            'max': 1.0,
            'std': 2.0,
            'var': 2.0,
            'quantile': 3.0,
            'skew': 4.0,
            'kurt': 4.0,
            'corr': 5.0,
            'cov': 5.0,
            'apply': 6.0
        }
        return complexity_map.get(operation, 2.0)
    
    def _generate_cache_key(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                          window: int, **kwargs) -> str:
        """Generate cache key for operation."""
        import hashlib
        
        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        
        return f"{operation}_{window}_{data_hash}_{params_hash}"
    
    # Include all the original methods for backward compatibility
    def _chunked_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Process large data in chunks for memory efficiency (original implementation)."""
        if isinstance(data, pd.Series):
            return self._chunked_series_operation(data, operation, window, **kwargs)
        else:
            return self._chunked_dataframe_operation(data, operation, window, **kwargs)
    
    def _chunked_series_operation(self, data: pd.Series, operation: str, 
                                window: int, **kwargs) -> pd.Series:
        """Process Series in chunks for memory efficiency (original implementation)."""
        results = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap for rolling window
            
            if self._should_use_vectorbt(chunk, window):
                chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif self._should_use_gpu(chunk, window):
                chunk_result = self._gpu_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            else:
                chunk_result = self._pandas_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
            
            # Remove overlap from result (except for first chunk)
            if i == 0:
                results.append(chunk_result)
            else:
                results.append(chunk_result.iloc[window-1:])
        
        return pd.concat(results, ignore_index=False)
    
    def _chunked_dataframe_operation(self, data: pd.DataFrame, operation: str, 
                                   window: int, **kwargs) -> pd.DataFrame:
        """Process DataFrame in chunks for memory efficiency (original implementation)."""
        results = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap for rolling window
            
            if self._should_use_vectorbt(chunk, window):
                chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif self._should_use_gpu(chunk, window):
                chunk_result = self._gpu_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            else:
                chunk_result = self._pandas_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
            
            # Remove overlap from result (except for first chunk)
            if i == 0:
                results.append(chunk_result)
            else:
                results.append(chunk_result.iloc[window-1:])
        
        return pd.concat(results, ignore_index=False)
    
    def _optimize_data_types(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data types for memory efficiency (original implementation)."""
        if self.memory_efficient:
            if isinstance(data, pd.Series):
                if data.dtype == 'float64':
                    if (data.min() >= np.finfo(np.float32).min and 
                        data.max() <= np.finfo(np.float32).max):
                        data = data.astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1
            elif isinstance(data, pd.DataFrame):
                optimized_data = data.copy()
                for column in optimized_data.columns:
                    if optimized_data[column].dtype == 'float64':
                        if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                            optimized_data[column].max() <= np.finfo(np.float32).max):
                            optimized_data[column] = optimized_data[column].astype(np.float32)
                            self.performance_stats['memory_optimizations'] += 1
                return optimized_data
        return data
    
    def _should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if VectorBT should be used for this operation (original implementation)."""
        if not self.use_vectorbt:
            return False
        
        # Use VectorBT for larger datasets or when parallel processing is beneficial
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size > 1000 or (self.enable_parallel and data_size > 100)
    
    def _should_use_gpu(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if GPU acceleration should be used (original implementation)."""
        if not self.enable_gpu or not CUPY_AVAILABLE:
            return False
        
        # Use GPU for very large datasets
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size > 10000
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT (original implementation)."""
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            elif operation == 'quantile':
                q = kwargs.get('q', 0.5)
                return rolling_quantile(data, window=window, q=q, **kwargs)
            elif operation == 'skew':
                return rolling_skew(data, window=window, **kwargs)
            elif operation == 'kurt':
                return rolling_kurt(data, window=window, **kwargs)
            elif operation == 'apply':
                func = kwargs.get('func')
                return rolling_apply(data, window=window, func=func, **kwargs)
            elif operation == 'corr':
                data2 = kwargs.get('data2')
                return rolling_corr(data, data2, window=window, **kwargs)
            elif operation == 'cov':
                data2 = kwargs.get('data2')
                return rolling_cov(data, data2, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported VectorBT operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT {operation} failed: {e}")
            raise
    
    def _gpu_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                              window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using GPU acceleration (original implementation)."""
        try:
            # Convert to CuPy arrays
            if isinstance(data, pd.Series):
                gpu_data = cp.asarray(data.values)
                result = self._gpu_rolling_series(gpu_data, operation, window, **kwargs)
                return pd.Series(result, index=data.index, name=data.name)
            else:
                gpu_data = cp.asarray(data.values)
                result = self._gpu_rolling_dataframe(gpu_data, operation, window, **kwargs)
                return pd.DataFrame(result, index=data.index, columns=data.columns)
        except Exception as e:
            logger.warning(f"GPU {operation} failed: {e}")
            raise
    
    def _gpu_rolling_series(self, data, operation: str, window: int, **kwargs):
        """GPU rolling operation for Series (original implementation)."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones(window) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones(window), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._numpy_rolling_operation(pd.Series(data.get()), operation, window, **kwargs).values
    
    def _gpu_rolling_dataframe(self, data, operation: str, window: int, **kwargs):
        """GPU rolling operation for DataFrame (original implementation)."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones((window, 1)), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._numpy_rolling_operation(pd.DataFrame(data.get()), operation, window, **kwargs).values
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using pandas (original implementation)."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        elif operation == 'corr':
            data2 = kwargs.get('data2')
            return rolling_obj.corr(data2)
        elif operation == 'cov':
            data2 = kwargs.get('data2')
            return rolling_obj.cov(data2)
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _numpy_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using numpy (fallback) (original implementation)."""
        if isinstance(data, pd.Series):
            values = data.values
            result = self._numpy_rolling_series(values, operation, window, **kwargs)
            return pd.Series(result, index=data.index, name=data.name)
        else:
            values = data.values
            result = self._numpy_rolling_dataframe(values, operation, window, **kwargs)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
    
    def _numpy_rolling_series(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """Numpy rolling operation for Series (original implementation)."""
        if operation == 'mean':
            return np.convolve(values, np.ones(window) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones(window), mode='same')
        else:
            # For complex operations, use pandas
            series = pd.Series(values)
            return series.rolling(window=window, **kwargs).agg(operation).values
    
    def _numpy_rolling_dataframe(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """Numpy rolling operation for DataFrame (original implementation)."""
        if operation == 'mean':
            return np.convolve(values, np.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones((window, 1)), mode='same')
        else:
            # For complex operations, use pandas
            df = pd.DataFrame(values)
            return df.rolling(window=window, **kwargs).agg(operation).values
    
    def _select_processing_strategy(self, data: Union[pd.Series, pd.DataFrame], 
                                   window: int, operation: str) -> str:
        """Select optimal processing strategy with detailed logging (original implementation)."""
        tprint_debug(f"🎯 Selecting processing strategy for {operation}")
        
        # VectorBT strategy
        if self._should_use_vectorbt(data, window):
            tprint_debug("✅ Selected VectorBT strategy")
            return 'vectorbt'
        
        # GPU strategy
        if self._should_use_gpu(data, window):
            tprint_debug("✅ Selected GPU strategy")
            return 'gpu'
        
        # Pandas fallback
        tprint_debug("✅ Selected Pandas strategy (fallback)")
        return 'pandas'
    
    def _validate_rolling_inputs(self, data: Union[pd.Series, pd.DataFrame], 
                                window: int, operation: str):
        """Validate rolling operation inputs with comprehensive checks (original implementation)."""
        tprint_debug(f"🔍 Validating rolling inputs for {operation}")
        
        # Check data type
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise VectorBTValidationError("Data must be a pandas Series or DataFrame", "type_check", type(data))
        
        # Check data is not empty
        if len(data) == 0:
            raise VectorBTValidationError("Data cannot be empty", "empty_check", len(data))
        
        # Check window size
        if not isinstance(window, int) or window <= 0:
            raise VectorBTValidationError("Window must be a positive integer", "range_check", window)
        
        if window > len(data):
            raise VectorBTValidationError(f"Window size ({window}) cannot be larger than data length ({len(data)})", "range_check", window)
        
        # Check for NaN values in critical columns
        if isinstance(data, pd.DataFrame):
            nan_counts = data.isnull().sum()
            if nan_counts.any():
                tprint_warning(f"⚠️ NaN values detected in data: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check data types for numeric operations
        if operation in ['mean', 'std', 'var', 'sum', 'quantile', 'skew', 'kurt']:
            if isinstance(data, pd.Series):
                if not pd.api.types.is_numeric_dtype(data):
                    raise VectorBTValidationError("Data must be numeric for this operation", "dtype_check", data.dtype)
            else:  # DataFrame
                non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_cols) > 0:
                    raise VectorBTValidationError(f"All columns must be numeric for {operation}, found: {list(non_numeric_cols)}", "dtype_check", list(non_numeric_cols))
        
        tprint_success(f"✅ Rolling inputs validated for {operation}")
    
    def _validate_rolling_result(self, result: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int):
        """Validate rolling operation result (original implementation)."""
        tprint_debug(f"🔍 Validating rolling result for {operation}")
        
        if result is None:
            raise VectorBTValidationError("Result cannot be None", "null_check", result)
        
        # Check result type matches input type
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            raise VectorBTValidationError("Result must be a pandas Series or DataFrame", "type_check", type(result))
        
        # Check for infinite values
        if isinstance(result, pd.Series):
            if np.isinf(result).any():
                tprint_warning(f"⚠️ Infinite values detected in {operation} result")
        else:  # DataFrame
            inf_counts = np.isinf(result).sum()
            if inf_counts.any():
                tprint_warning(f"⚠️ Infinite values detected in {operation} result: {inf_counts[inf_counts > 0].to_dict()}")
        
        tprint_success(f"✅ Rolling result validated for {operation}")
    
    # Include all fallback methods for backward compatibility
    def _fallback_rolling_mean(self, data: Union[pd.Series, pd.DataFrame], 
                              window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling mean with error handling (original implementation)."""
        tprint_warning("⚠️ Using fallback rolling mean implementation")
        try:
            return data.rolling(window=window, **kwargs).mean()
        except Exception as e:
            error_msg = f"Fallback rolling mean failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='mean', original_error=e)
    
    def _fallback_rolling_std(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling std with error handling (original implementation)."""
        tprint_warning("⚠️ Using fallback rolling std implementation")
        try:
            return data.rolling(window=window, **kwargs).std()
        except Exception as e:
            error_msg = f"Fallback rolling std failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='std', original_error=e)
    
    def _fallback_rolling_var(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling var with error handling (original implementation)."""
        tprint_warning("⚠️ Using fallback rolling var implementation")
        try:
            return data.rolling(window=window, **kwargs).var()
        except Exception as e:
            error_msg = f"Fallback rolling var failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='var', original_error=e)
    
    def _fallback_rolling_min(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling min with error handling (original implementation)."""
        tprint_warning("⚠️ Using fallback rolling min implementation")
        try:
            return data.rolling(window=window, **kwargs).min()
        except Exception as e:
            error_msg = f"Fallback rolling min failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='min', original_error=e)
    
    def _fallback_rolling_max(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling max with error handling (original implementation)."""
        tprint_warning("⚠️ Using fallback rolling max implementation")
        try:
            return data.rolling(window=window, **kwargs).max()
        except Exception as e:
            error_msg = f"Fallback rolling max failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='max', original_error=e)
    
    def _fallback_rolling_sum(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling sum with error handling (original implementation)."""
        tprint_warning("⚠️ Using fallback rolling sum implementation")
        try:
            return data.rolling(window=window, **kwargs).sum()
        except Exception as e:
            error_msg = f"Fallback rolling sum failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='sum', original_error=e)
    
    def _fallback_rolling_correlation_matrix(self, data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
        """Fallback rolling correlation matrix (original implementation)."""
        return data.rolling(window=window, **kwargs).corr()
    
    def _fallback_rolling_covariance_matrix(self, data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
        """Fallback rolling covariance matrix (original implementation)."""
        return data.rolling(window=window, **kwargs).cov()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics."""
        stats = self.performance_stats.copy()
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['m1_gpu_usage_rate'] = stats['m1_gpu_operations'] / stats['total_operations']
            stats['adaptive_chunk_usage_rate'] = stats['adaptive_chunk_operations'] / stats['total_operations']
            stats['cache_hit_rate'] = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        # Add memory manager stats
        if self.memory_manager:
            stats['memory_stats'] = self.memory_manager.get_stats()
        
        # Add cache manager stats
        if self.cache_manager:
            stats['cache_stats'] = self.cache_manager.get_stats()
        
        # Add M1 GPU stats
        if self.m1_optimizer:
            stats['m1_gpu_stats'] = self.m1_optimizer.get_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        tprint_info("🔄 Resetting enhanced performance statistics")
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'numpy_fallbacks': 0,
            'gpu_operations': 0,
            'm1_gpu_operations': 0,
            'memory_optimizations': 0,
            'chunk_operations': 0,
            'adaptive_chunk_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_operations': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }
        tprint_success("✅ Enhanced performance statistics reset")
    
    def cleanup(self) -> None:
        """Enhanced cleanup with resource management."""
        tprint("🧹 Cleaning up enhanced VectorBT rolling optimizer resources")
        
        try:
            # Cleanup memory manager
            if self.memory_manager:
                self.memory_manager.cleanup()
                tprint("✅ Memory manager cleaned up")
            
            # Cleanup cache manager
            if self.cache_manager:
                self.cache_manager.cleanup()
                tprint("✅ Cache manager cleaned up")
            
            # Clear any caches or temporary data
            if hasattr(self, '_operation_cache'):
                self._operation_cache.clear()
                tprint("✅ Operation cache cleared")
            
            # Reset performance stats
            self.reset_stats()
            
            # Force garbage collection
            gc.collect()
            tprint("✅ Garbage collection completed")
            
        except Exception as e:
            tprint_error(f"❌ ERROR: Enhanced VectorBT rolling optimizer cleanup failed: {e}")
            raise RuntimeError(f"Enhanced VectorBT rolling optimizer cleanup failed: {e}")
        
        tprint("✅ Enhanced VectorBT rolling optimizer cleanup completed")
    
    def __enter__(self) -> 'EnhancedVectorBTRollingOptimizer':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()


# Backward compatibility - create alias for original class name
VectorBTRollingOptimizer = EnhancedVectorBTRollingOptimizer

# Global optimizer instance
_global_optimizer = None

def get_vectorbt_rolling_optimizer(enable_gpu: bool = False, enable_parallel: bool = True, 
                                 memory_efficient: bool = True, chunk_size: int = 1000,
                                 fast_fail: bool = True, enable_logging: bool = True,
                                 # New enhanced parameters with defaults for backward compatibility
                                 memory_config: Optional[MemoryConfig] = None,
                                 cache_config: Optional[CacheConfig] = None,
                                 enable_m1_gpu: bool = True,
                                 enable_adaptive_chunking: bool = True,
                                 enable_advanced_caching: bool = True) -> EnhancedVectorBTRollingOptimizer:
    """Get global enhanced VectorBT rolling optimizer instance with backward compatibility."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = EnhancedVectorBTRollingOptimizer(
            enable_gpu=enable_gpu, 
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient,
            chunk_size=chunk_size,
            fast_fail=fast_fail,
            enable_logging=enable_logging,
            memory_config=memory_config,
            cache_config=cache_config,
            enable_m1_gpu=enable_m1_gpu,
            enable_adaptive_chunking=enable_adaptive_chunking,
            enable_advanced_caching=enable_advanced_caching
        )
    return _global_optimizer


# Backward compatibility functions
def optimized_rolling_mean(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling mean using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_mean(data, window, **kwargs)

def optimized_rolling_std(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling standard deviation using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_std(data, window, **kwargs)

def optimized_rolling_var(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling variance using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_var(data, window, **kwargs)

def optimized_rolling_min(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling minimum using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_min(data, window, **kwargs)

def optimized_rolling_max(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling maximum using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_max(data, window, **kwargs)

def optimized_rolling_sum(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling sum using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_sum(data, window, **kwargs)

def optimized_rolling_quantile(data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling quantile using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_quantile(data, window, q=q, **kwargs)

def optimized_rolling_apply(data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling apply using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_apply(data, window, func, **kwargs)

def optimized_rolling_corr(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling correlation using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_corr(data1, data2, window, **kwargs)

def optimized_rolling_cov(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                         window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling covariance using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_cov(data1, data2, window, **kwargs)

def optimized_rolling_correlation_matrix(data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Optimized rolling correlation matrix using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_correlation_matrix(data, window, **kwargs)

def optimized_rolling_covariance_matrix(data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Optimized rolling covariance matrix using enhanced VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_covariance_matrix(data, window, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
    np.random.seed(42)
    
    # Generate sample data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(5000) * 0.01),
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=dates)
    
    # Test enhanced optimizer with backward compatibility
    print("Testing Enhanced VectorBT rolling operations with backward compatibility...")
    
    # Test with original parameters (backward compatibility)
    optimizer = EnhancedVectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
    
    # Test various operations
    print("Testing enhanced rolling operations...")
    
    # Rolling mean
    mean_result = optimizer.rolling_mean(data['close'], window=20)
    print(f"Rolling mean shape: {mean_result.shape}")
    
    # Rolling std
    std_result = optimizer.rolling_std(data['close'], window=20)
    print(f"Rolling std shape: {std_result.shape}")
    
    # Rolling correlation
    corr_result = optimizer.rolling_corr(data['close'], data['volume'], window=20)
    print(f"Rolling correlation shape: {corr_result.shape}")
    
    # Performance stats
    stats = optimizer.get_performance_stats()
    print(f"Enhanced performance stats: {stats}")
    
    # Test with enhanced features
    print("\nTesting enhanced features...")
    
    # Create enhanced configuration
    memory_config = MemoryConfig(
        max_memory_gb=4.0,
        memory_pressure_threshold=0.7,
        adaptive_chunking=True,
        memory_pooling=True
    )
    
    cache_config = CacheConfig(
        l1_cache_size=500,
        l2_cache_size=2000,
        cache_ttl=1800.0  # 30 minutes
    )
    
    # Test with enhanced configuration
    enhanced_optimizer = EnhancedVectorBTRollingOptimizer(
        enable_gpu=False,
        enable_parallel=True,
        memory_efficient=True,
        chunk_size=1000,
        fast_fail=True,
        enable_logging=True,
        memory_config=memory_config,
        cache_config=cache_config,
        enable_m1_gpu=True,
        enable_adaptive_chunking=True,
        enable_advanced_caching=True
    )
    
    # Test enhanced operations
    enhanced_mean = enhanced_optimizer.rolling_mean(data['close'], window=20)
    print(f"Enhanced rolling mean shape: {enhanced_mean.shape}")
    
    # Test caching (second call should hit cache)
    cached_mean = enhanced_optimizer.rolling_mean(data['close'], window=20)
    print(f"Cached rolling mean shape: {cached_mean.shape}")
    
    # Enhanced performance stats
    enhanced_stats = enhanced_optimizer.get_performance_stats()
    print(f"Enhanced performance stats: {enhanced_stats}")
    
    print("\nEnhanced VectorBT rolling optimizer test completed successfully!")