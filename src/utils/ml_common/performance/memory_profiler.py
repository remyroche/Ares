"""
Advanced Memory Profiling and Optimization for ML Common Operations

This module provides comprehensive memory profiling, monitoring, and optimization
for ML operations, integrating with M1 hardware optimizations and VectorBT.
"""

import asyncio
import gc
import logging
import psutil
import threading
import time
import tracemalloc
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path
import json
import pickle

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profiler = None

# Import M1 optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryProfileLevel(Enum):
    """Memory profiling levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    REAL_TIME = "real_time"

class MemoryOptimizationStrategy(Enum):
    """Memory optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

@dataclass
class MemoryProfile:
    """Memory profile data structure."""
    
    timestamp: float
    process_memory_mb: float
    system_memory_mb: float
    memory_percent: float
    peak_memory_mb: float
    memory_available_mb: float
    gc_objects: int
    gc_garbage: int
    tracemalloc_peak_mb: Optional[float] = None
    tracemalloc_current_mb: Optional[float] = None
    numpy_memory_mb: Optional[float] = None
    pandas_memory_mb: Optional[float] = None
    vectorbt_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'process_memory_mb': self.process_memory_mb,
            'system_memory_mb': self.system_memory_mb,
            'memory_percent': self.memory_percent,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_available_mb': self.memory_available_mb,
            'gc_objects': self.gc_objects,
            'gc_garbage': self.gc_garbage,
            'tracemalloc_peak_mb': self.tracemalloc_peak_mb,
            'tracemalloc_current_mb': self.tracemalloc_current_mb,
            'numpy_memory_mb': self.numpy_memory_mb,
            'pandas_memory_mb': self.pandas_memory_mb,
            'vectorbt_memory_mb': self.vectorbt_memory_mb
        }

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization."""
    
    # Profiling settings
    enable_profiling: bool = True
    profile_level: MemoryProfileLevel = MemoryProfileLevel.DETAILED
    profile_interval: float = 1.0  # seconds
    enable_tracemalloc: bool = True
    tracemalloc_limit: int = 10  # MB
    
    # Optimization settings
    enable_optimization: bool = True
    optimization_strategy: MemoryOptimizationStrategy = MemoryOptimizationStrategy.BALANCED
    memory_threshold_mb: float = 1000.0  # 1GB
    gc_threshold: int = 100  # Trigger GC every N operations
    enable_weak_references: bool = True
    
    # M1 optimizations
    enable_m1_optimizations: bool = True
    use_m1_memory_optimizer: bool = True
    use_m1_cpu_optimizer: bool = True
    use_m1_gpu_optimizer: bool = True
    
    # VectorBT optimizations
    enable_vectorbt_optimizations: bool = True
    use_vectorbt_rolling: bool = True
    
    # Monitoring settings
    enable_real_time_monitoring: bool = False
    monitoring_interval: float = 0.5  # seconds
    max_profiles: int = 1000
    
    # Alerting settings
    enable_memory_alerts: bool = True
    critical_memory_threshold: float = 0.9  # 90% of system memory
    warning_memory_threshold: float = 0.75  # 75% of system memory
    
    # Reporting settings
    enable_memory_reports: bool = True
    report_interval: float = 60.0  # seconds
    save_profiles: bool = True
    profile_save_path: str = "data_cache/memory_profiles"

class MemoryProfiler:
    """Advanced memory profiler for ML operations."""
    
    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        self.config = config or MemoryOptimizationConfig()
        self.logger = logger.getChild('MemoryProfiler')
        self._profiles: List[MemoryProfile] = []
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Initialize optimizers
        self._m1_memory_optimizer = None
        self._m1_cpu_optimizer = None
        self._m1_gpu_manager = None
        self._vectorbt_optimizer = None
        
        if M1_OPTIMIZATIONS_AVAILABLE and self.config.enable_m1_optimizations:
            if self.config.use_m1_memory_optimizer:
                self._m1_memory_optimizer = get_m1_memory_optimizer()
            if self.config.use_m1_cpu_optimizer:
                self._m1_cpu_optimizer = get_m1_cpu_optimizer()
            if self.config.use_m1_gpu_optimizer:
                self._m1_gpu_manager = get_m1_gpu_manager()
        
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.enable_vectorbt_optimizations:
            self._vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        # Initialize tracemalloc if enabled
        if self.config.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(self.config.tracemalloc_limit)
        
        # Create profile save directory
        if self.config.save_profiles:
            Path(self.config.profile_save_path).mkdir(parents=True, exist_ok=True)
    
    def start_monitoring(self):
        """Start real-time memory monitoring."""
        if self._monitoring_active:
            return
        
        if not self.config.enable_real_time_monitoring:
            self.logger.warning("Real-time monitoring is disabled in config")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("🧠 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time memory monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        self.logger.info("🧠 Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                profile = self.get_memory_profile()
                self._add_profile(profile)
                
                # Check for memory alerts
                self._check_memory_alerts(profile)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval * 2)
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory profile."""
        try:
            # Get process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            process_memory_mb = process_memory.rss / (1024 * 1024)
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            system_memory_mb = system_memory.used / (1024 * 1024)
            memory_percent = system_memory.percent / 100.0
            memory_available_mb = system_memory.available / (1024 * 1024)
            
            # Get peak memory
            peak_memory_mb = process_memory_mb  # This would need to be tracked over time
            
            # Get GC stats
            gc_objects = len(gc.get_objects())
            gc_garbage = len(gc.garbage)
            
            # Get tracemalloc stats if available
            tracemalloc_peak_mb = None
            tracemalloc_current_mb = None
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current_mb = current / (1024 * 1024)
                tracemalloc_peak_mb = peak / (1024 * 1024)
            
            # Get library-specific memory usage
            numpy_memory_mb = self._get_numpy_memory_usage()
            pandas_memory_mb = self._get_pandas_memory_usage()
            vectorbt_memory_mb = self._get_vectorbt_memory_usage()
            
            return MemoryProfile(
                timestamp=time.time(),
                process_memory_mb=process_memory_mb,
                system_memory_mb=system_memory_mb,
                memory_percent=memory_percent,
                peak_memory_mb=peak_memory_mb,
                memory_available_mb=memory_available_mb,
                gc_objects=gc_objects,
                gc_garbage=gc_garbage,
                tracemalloc_peak_mb=tracemalloc_peak_mb,
                tracemalloc_current_mb=tracemalloc_current_mb,
                numpy_memory_mb=numpy_memory_mb,
                pandas_memory_mb=pandas_memory_mb,
                vectorbt_memory_mb=vectorbt_memory_mb
            )
            
        except Exception as e:
            self.logger.error(f"Error getting memory profile: {e}")
            # Return minimal profile
            return MemoryProfile(
                timestamp=time.time(),
                process_memory_mb=0.0,
                system_memory_mb=0.0,
                memory_percent=0.0,
                peak_memory_mb=0.0,
                memory_available_mb=0.0,
                gc_objects=0,
                gc_garbage=0
            )
    
    def _get_numpy_memory_usage(self) -> Optional[float]:
        """Get numpy memory usage in MB."""
        if not NUMPY_AVAILABLE:
            return None
        
        try:
            # This is a simplified approach - in practice, you'd want more sophisticated tracking
            total_memory = 0
            for obj in gc.get_objects():
                if isinstance(obj, np.ndarray):
                    total_memory += obj.nbytes
            
            return total_memory / (1024 * 1024)
        except Exception:
            return None
    
    def _get_pandas_memory_usage(self) -> Optional[float]:
        """Get pandas memory usage in MB."""
        if not PANDAS_AVAILABLE:
            return None
        
        try:
            total_memory = 0
            for obj in gc.get_objects():
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    total_memory += obj.memory_usage(deep=True).sum()
            
            return total_memory / (1024 * 1024)
        except Exception:
            return None
    
    def _get_vectorbt_memory_usage(self) -> Optional[float]:
        """Get VectorBT memory usage in MB."""
        if not VECTORBT_OPTIMIZATIONS_AVAILABLE:
            return None
        
        try:
            # This would need to be implemented based on VectorBT's memory tracking
            # For now, return None as a placeholder
            return None
        except Exception:
            return None
    
    def _add_profile(self, profile: MemoryProfile):
        """Add profile to the list."""
        with self._lock:
            self._profiles.append(profile)
            
            # Limit profile count
            if len(self._profiles) > self.config.max_profiles:
                self._profiles.pop(0)
    
    def _check_memory_alerts(self, profile: MemoryProfile):
        """Check for memory alerts."""
        if not self.config.enable_memory_alerts:
            return
        
        if profile.memory_percent >= self.config.critical_memory_threshold:
            self.logger.critical(f"🚨 CRITICAL: Memory usage at {profile.memory_percent:.1%}")
            self._trigger_memory_optimization(MemoryOptimizationStrategy.AGGRESSIVE)
        elif profile.memory_percent >= self.config.warning_memory_threshold:
            self.logger.warning(f"⚠️ WARNING: Memory usage at {profile.memory_percent:.1%}")
            self._trigger_memory_optimization(MemoryOptimizationStrategy.BALANCED)
    
    def _trigger_memory_optimization(self, strategy: MemoryOptimizationStrategy):
        """Trigger memory optimization."""
        try:
            if strategy == MemoryOptimizationStrategy.AGGRESSIVE:
                self._aggressive_memory_cleanup()
            elif strategy == MemoryOptimizationStrategy.BALANCED:
                self._balanced_memory_cleanup()
            elif strategy == MemoryOptimizationStrategy.CONSERVATIVE:
                self._conservative_memory_cleanup()
            else:  # ADAPTIVE
                self._adaptive_memory_cleanup()
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
    
    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup."""
        self.logger.info("🧹 Performing aggressive memory cleanup")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear caches
        if self._m1_memory_optimizer:
            self._m1_memory_optimizer.force_garbage_collection()
        
        # Clear library caches
        self._clear_library_caches()
    
    def _balanced_memory_cleanup(self):
        """Balanced memory cleanup."""
        self.logger.info("🧹 Performing balanced memory cleanup")
        
        # Single garbage collection
        gc.collect()
        
        # Clear some caches
        if self._m1_memory_optimizer:
            self._m1_memory_optimizer._moderate_memory_cleanup()
    
    def _conservative_memory_cleanup(self):
        """Conservative memory cleanup."""
        self.logger.info("🧹 Performing conservative memory cleanup")
        
        # Light garbage collection
        gc.collect(0)  # Young generation only
    
    def _adaptive_memory_cleanup(self):
        """Adaptive memory cleanup based on current state."""
        profile = self.get_memory_profile()
        
        if profile.memory_percent > 0.9:
            self._aggressive_memory_cleanup()
        elif profile.memory_percent > 0.75:
            self._balanced_memory_cleanup()
        else:
            self._conservative_memory_cleanup()
    
    def _clear_library_caches(self):
        """Clear library-specific caches."""
        try:
            # Clear pandas caches
            if PANDAS_AVAILABLE:
                # Clear any pandas internal caches
                pass
            
            # Clear numpy caches
            if NUMPY_AVAILABLE:
                # Clear any numpy internal caches
                pass
            
        except Exception as e:
            self.logger.debug(f"Error clearing library caches: {e}")
    
    @contextmanager
    def memory_checkpoint(self, name: str):
        """Context manager for memory checkpointing."""
        start_profile = self.get_memory_profile()
        start_time = time.time()
        
        try:
            self.logger.debug(f"🧠 Memory checkpoint '{name}' started: {start_profile.process_memory_mb:.1f} MB")
            yield
        finally:
            end_profile = self.get_memory_profile()
            end_time = time.time()
            
            memory_diff = end_profile.process_memory_mb - start_profile.process_memory_mb
            time_diff = end_time - start_time
            
            if memory_diff > 10:  # Log if memory increased by more than 10MB
                self.logger.info(f"🧠 Memory checkpoint '{name}' completed: +{memory_diff:.1f} MB in {time_diff:.3f}s")
            else:
                self.logger.debug(f"🧠 Memory checkpoint '{name}' completed: {memory_diff:+.1f} MB in {time_diff:.3f}s")
    
    def optimize_memory_usage(self, data: Any) -> Any:
        """Optimize memory usage for data."""
        try:
            # Apply M1 optimizations if available
            if self._m1_memory_optimizer and hasattr(data, 'memory_usage'):
                optimized_data = self._m1_memory_optimizer.optimize_dataframe_memory(data)
            else:
                optimized_data = data
            
            # Apply VectorBT optimizations if available
            if self._vectorbt_optimizer and hasattr(data, 'values'):
                optimized_data = self._vectorbt_optimizer.optimize_dataframe(optimized_data)
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
            return data
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_profile = self.get_memory_profile()
        
        with self._lock:
            if not self._profiles:
                return current_profile.to_dict()
            
            # Calculate statistics from all profiles
            process_memories = [p.process_memory_mb for p in self._profiles]
            system_memories = [p.system_memory_mb for p in self._profiles]
            memory_percents = [p.memory_percent for p in self._profiles]
            
            stats = {
                'current': current_profile.to_dict(),
                'history': {
                    'process_memory': {
                        'min': min(process_memories),
                        'max': max(process_memories),
                        'avg': sum(process_memories) / len(process_memories),
                        'current': current_profile.process_memory_mb
                    },
                    'system_memory': {
                        'min': min(system_memories),
                        'max': max(system_memories),
                        'avg': sum(system_memories) / len(system_memories),
                        'current': current_profile.system_memory_mb
                    },
                    'memory_percent': {
                        'min': min(memory_percents),
                        'max': max(memory_percents),
                        'avg': sum(memory_percents) / len(memory_percents),
                        'current': current_profile.memory_percent
                    }
                },
                'gc_stats': {
                    'objects': current_profile.gc_objects,
                    'garbage': current_profile.gc_garbage
                },
                'tracemalloc_stats': {
                    'peak_mb': current_profile.tracemalloc_peak_mb,
                    'current_mb': current_profile.tracemalloc_current_mb
                },
                'library_memory': {
                    'numpy_mb': current_profile.numpy_memory_mb,
                    'pandas_mb': current_profile.pandas_memory_mb,
                    'vectorbt_mb': current_profile.vectorbt_memory_mb
                },
                'profile_count': len(self._profiles)
            }
            
            return stats
    
    def save_profiles(self, filepath: Optional[str] = None) -> bool:
        """Save memory profiles to file."""
        if not self.config.save_profiles:
            return False
        
        try:
            if filepath is None:
                timestamp = int(time.time())
                filepath = f"{self.config.profile_save_path}/memory_profiles_{timestamp}.json"
            
            with self._lock:
                profiles_data = [profile.to_dict() for profile in self._profiles]
            
            with open(filepath, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            self.logger.info(f"💾 Memory profiles saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save memory profiles: {e}")
            return False
    
    def load_profiles(self, filepath: str) -> bool:
        """Load memory profiles from file."""
        try:
            with open(filepath, 'r') as f:
                profiles_data = json.load(f)
            
            with self._lock:
                self._profiles = [MemoryProfile(**data) for data in profiles_data]
            
            self.logger.info(f"📂 Memory profiles loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load memory profiles: {e}")
            return False
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        stats = self.get_memory_stats()
        
        report = {
            'timestamp': time.time(),
            'summary': {
                'current_memory_mb': stats['current']['process_memory_mb'],
                'memory_percent': stats['current']['memory_percent'],
                'peak_memory_mb': stats['history']['process_memory']['max'],
                'avg_memory_mb': stats['history']['process_memory']['avg'],
                'gc_objects': stats['gc_stats']['objects'],
                'gc_garbage': stats['gc_stats']['garbage']
            },
            'recommendations': self._generate_recommendations(stats),
            'detailed_stats': stats
        }
        
        return report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        current_memory = stats['current']['process_memory_mb']
        memory_percent = stats['current']['memory_percent']
        gc_objects = stats['gc_stats']['objects']
        gc_garbage = stats['gc_stats']['garbage']
        
        if memory_percent > 0.9:
            recommendations.append("🚨 CRITICAL: Memory usage is very high. Consider aggressive cleanup.")
        elif memory_percent > 0.75:
            recommendations.append("⚠️ WARNING: Memory usage is high. Consider memory optimization.")
        
        if gc_garbage > 100:
            recommendations.append("🧹 High garbage collection count. Check for memory leaks.")
        
        if gc_objects > 100000:
            recommendations.append("📊 High object count. Consider object pooling or weak references.")
        
        if current_memory > 1000:  # 1GB
            recommendations.append("💾 High memory usage. Consider data chunking or streaming.")
        
        if not recommendations:
            recommendations.append("✅ Memory usage appears healthy.")
        
        return recommendations

# Global profiler instance
_global_profiler: Optional[MemoryProfiler] = None

def get_memory_profiler(config: Optional[MemoryOptimizationConfig] = None) -> MemoryProfiler:
    """Get the global memory profiler instance."""
    global _global_profiler
    
    if _global_profiler is None:
        _global_profiler = MemoryProfiler(config)
    
    return _global_profiler

def memory_profile(level: MemoryProfileLevel = MemoryProfileLevel.DETAILED):
    """Decorator for memory profiling functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_memory_profiler()
            
            with profiler.memory_checkpoint(f"{func.__name__}"):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def memory_optimize(data: Any) -> Any:
    """Optimize memory usage for data."""
    profiler = get_memory_profiler()
    return profiler.optimize_memory_usage(data)

def start_memory_monitoring():
    """Start global memory monitoring."""
    profiler = get_memory_profiler()
    profiler.start_monitoring()

def stop_memory_monitoring():
    """Stop global memory monitoring."""
    profiler = get_memory_profiler()
    profiler.stop_monitoring()

def get_memory_stats() -> Dict[str, Any]:
    """Get global memory statistics."""
    profiler = get_memory_profiler()
    return profiler.get_memory_stats()

def generate_memory_report() -> Dict[str, Any]:
    """Generate global memory report."""
    profiler = get_memory_profiler()
    return profiler.generate_memory_report()