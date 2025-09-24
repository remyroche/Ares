"""
Performance & Memory Management for CVLSA

This module implements advanced performance and memory management with:
1. Memory-efficient processing for large datasets
2. Model caching to avoid retraining
3. Incremental learning with resource constraints
4. Resource monitoring and optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import pickle
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from contextlib import contextmanager
import queue
import weakref

# Import existing utilities
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer

logger = logging.getLogger(__name__)

@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    # Memory management
    max_memory_usage: float = 0.8  # Maximum memory usage (80% of available)
    chunk_size: int = 1000  # Chunk size for processing
    enable_memory_monitoring: bool = True
    memory_cleanup_threshold: float = 0.7  # Cleanup when memory usage exceeds this
    
    # Model caching
    enable_model_caching: bool = True
    cache_directory: str = "./model_cache"
    max_cache_size: int = 10  # Maximum number of cached models
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    
    # Incremental learning
    enable_incremental_learning: bool = True
    incremental_batch_size: int = 100
    max_incremental_samples: int = 10000
    learning_rate_decay: float = 0.95
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: int = 5  # Seconds
    performance_history_size: int = 1000
    
    # Resource optimization
    enable_auto_optimization: bool = True
    optimization_threshold: float = 0.8  # Optimize when resource usage exceeds this
    gpu_memory_fraction: float = 0.8  # Maximum GPU memory usage

class ModelCache:
    """Intelligent model caching system."""
    
    def __init__(self, cache_directory: str, max_size: int = 10, ttl: int = 3600):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        logger.info(f"💾 Model cache initialized: {cache_directory}")
    
    def _generate_cache_key(self, model_config: Dict[str, Any], data_hash: str) -> str:
        """Generate unique cache key for model and data."""
        config_str = str(sorted(model_config.items()))
        key_string = f"{config_str}_{data_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _calculate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Calculate hash of input data for cache key."""
        # Use data shape and sample statistics for hash
        data_info = {
            'shape': X.shape,
            'dtype': str(X.dtype),
            'mean': np.mean(X),
            'std': np.std(X),
            'y_mean': np.mean(y),
            'y_std': np.std(y)
        }
        return hashlib.md5(str(data_info).encode()).hexdigest()
    
    def get_cached_model(self, model_config: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Optional[Any]:
        """Retrieve cached model if available."""
        with self._lock:
            data_hash = self._calculate_data_hash(X, y)
            cache_key = self._generate_cache_key(model_config, data_hash)
            
            if cache_key in self.cache_metadata:
                metadata = self.cache_metadata[cache_key]
                
                # Check TTL
                if time.time() - metadata['timestamp'] > self.ttl:
                    self._remove_cache_entry(cache_key)
                    return None
                
                # Load model from disk
                try:
                    cache_file = self.cache_directory / f"{cache_key}.pkl"
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            model = pickle.load(f)
                        
                        # Update access time
                        self.access_times[cache_key] = time.time()
                        
                        logger.info(f"💾 Retrieved cached model: {cache_key}")
                        return model
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}")
                    self._remove_cache_entry(cache_key)
            
            return None
    
    def cache_model(self, model: Any, model_config: Dict[str, Any], 
                   X: np.ndarray, y: np.ndarray, performance_metrics: Dict[str, float]):
        """Cache a trained model."""
        with self._lock:
            data_hash = self._calculate_data_hash(X, y)
            cache_key = self._generate_cache_key(model_config, data_hash)
            
            # Check cache size limit
            if len(self.cache_metadata) >= self.max_size:
                self._evict_oldest_entry()
            
            try:
                # Save model to disk
                cache_file = self.cache_directory / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(model, f)
                
                # Store metadata
                self.cache_metadata[cache_key] = {
                    'timestamp': time.time(),
                    'model_config': model_config,
                    'data_hash': data_hash,
                    'performance_metrics': performance_metrics,
                    'file_size': cache_file.stat().st_size
                }
                
                self.access_times[cache_key] = time.time()
                
                logger.info(f"💾 Cached model: {cache_key}")
                
            except Exception as e:
                logger.error(f"Failed to cache model: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry and file."""
        if cache_key in self.cache_metadata:
            del self.cache_metadata[cache_key]
        
        if cache_key in self.access_times:
            del self.access_times[cache_key]
        
        cache_file = self.cache_directory / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
    
    def _evict_oldest_entry(self):
        """Evict the least recently used cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        self._remove_cache_entry(oldest_key)
        logger.info(f"🗑️ Evicted oldest cache entry: {oldest_key}")
    
    def cleanup_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, metadata in self.cache_metadata.items():
            if current_time - metadata['timestamp'] > self.ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)
        
        if expired_keys:
            logger.info(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(metadata['file_size'] for metadata in self.cache_metadata.values())
        
        return {
            'total_entries': len(self.cache_metadata),
            'max_entries': self.max_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_directory),
            'ttl_seconds': self.ttl
        }

class ResourceMonitor:
    """Advanced resource monitoring system."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.monitoring_active = False
        self.monitoring_thread = None
        self.performance_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Initialize hardware monitors
        self._init_hardware_monitors()
        
        logger.info("📊 Resource monitor initialized")
    
    def _init_hardware_monitors(self):
        """Initialize hardware monitoring components."""
        try:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
            self.matrix_ops = get_enhanced_matrix_operations()
        except Exception as e:
            logger.warning(f"Hardware monitors not available: {e}")
            self.memory_optimizer = None
            self.gpu_manager = None
            self.matrix_ops = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("📊 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("📊 Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect resource metrics
                metrics = self._collect_resource_metrics()
                
                with self._lock:
                    self.performance_history.append(metrics)
                    
                    # Limit history size
                    if len(self.performance_history) > self.config.performance_history_size:
                        self.performance_history = self.performance_history[-self.config.performance_history_size:]
                
                # Check for optimization triggers
                if self.config.enable_auto_optimization:
                    self._check_optimization_triggers(metrics)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.config.monitoring_interval * 2)
    
    def _collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
        
        # GPU metrics if available
        if self.gpu_manager:
            try:
                gpu_info = self.gpu_manager.get_gpu_info()
                metrics.update({
                    'gpu_available': gpu_info.get('mps_available', False),
                    'gpu_memory_available': gpu_info.get('gpu_memory', 'Unknown')
                })
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        # Memory optimizer metrics
        if self.memory_optimizer:
            try:
                memory_stats = self.memory_optimizer.get_memory_stats()
                metrics.update({
                    'memory_pressure': memory_stats.get('memory_pressure', 0),
                    'protected_objects': memory_stats.get('protected_objects', 0)
                })
            except Exception as e:
                logger.debug(f"Memory optimizer metrics failed: {e}")
        
        return metrics
    
    def _check_optimization_triggers(self, metrics: Dict[str, Any]):
        """Check if optimization is needed based on current metrics."""
        if metrics['memory_percent'] > self.config.optimization_threshold * 100:
            logger.warning(f"🚨 High memory usage: {metrics['memory_percent']:.1f}%")
            self._trigger_memory_optimization()
        
        if metrics['cpu_percent'] > 90:
            logger.warning(f"🚨 High CPU usage: {metrics['cpu_percent']:.1f}%")
    
    def _trigger_memory_optimization(self):
        """Trigger memory optimization."""
        if self.memory_optimizer:
            try:
                optimization_result = self.memory_optimizer.optimize_memory_usage(aggressive=True)
                if optimization_result.get('success', False):
                    logger.info(f"🧠 Memory optimization completed: {optimization_result.get('memory_saved_mb', 0):.1f} MB saved")
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from monitoring history."""
        if not self.performance_history:
            return {}
        
        with self._lock:
            recent_history = self.performance_history[-100:]  # Last 100 measurements
            
            summary = {
                'monitoring_duration': len(self.performance_history),
                'average_cpu_percent': np.mean([m['cpu_percent'] for m in recent_history]),
                'average_memory_percent': np.mean([m['memory_percent'] for m in recent_history]),
                'peak_memory_percent': max([m['memory_percent'] for m in recent_history]),
                'average_memory_available_gb': np.mean([m['memory_available_gb'] for m in recent_history]),
                'optimization_triggers': sum(1 for m in recent_history if m['memory_percent'] > self.config.optimization_threshold * 100)
            }
            
            return summary

class IncrementalLearner:
    """Incremental learning system with resource constraints."""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.learning_history: List[Dict[str, Any]] = []
        self.current_model = None
        self.learning_rate = 0.01
        self._lock = threading.Lock()
        
        logger.info("🔄 Incremental learner initialized")
    
    def can_learn_incrementally(self, new_samples: int) -> bool:
        """Check if incremental learning is feasible."""
        if not self.config.enable_incremental_learning:
            return False
        
        # Check if we have a current model
        if self.current_model is None:
            return False
        
        # Check resource constraints
        if new_samples > self.config.max_incremental_samples:
            logger.warning(f"Incremental learning rejected: too many samples ({new_samples})")
            return False
        
        # Check memory constraints
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.config.max_memory_usage * 100:
                logger.warning(f"Incremental learning rejected: high memory usage ({memory_percent:.1f}%)")
                return False
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")
        
        return True
    
    def learn_incrementally(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Perform incremental learning on new data."""
        if not self.can_learn_incrementally(len(X_new)):
            return {'success': False, 'reason': 'Incremental learning not feasible'}
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Update learning rate with decay
                self.learning_rate *= self.config.learning_rate_decay
                
                # Perform incremental learning (simplified example)
                if hasattr(self.current_model, 'partial_fit'):
                    # Models that support partial_fit
                    self.current_model.partial_fit(X_new, y_new)
                else:
                    # For models without partial_fit, retrain with combined data
                    logger.warning("Model doesn't support partial_fit, retraining with combined data")
                    # This would require storing previous data, which is memory-intensive
                    return {'success': False, 'reason': 'Model does not support incremental learning'}
                
                # Record learning session
                learning_session = {
                    'timestamp': time.time(),
                    'samples_added': len(X_new),
                    'learning_rate': self.learning_rate,
                    'learning_time': time.time() - start_time
                }
                
                self.learning_history.append(learning_session)
                
                logger.info(f"🔄 Incremental learning completed: {len(X_new)} samples in {learning_session['learning_time']:.2f}s")
                
                return {
                    'success': True,
                    'samples_added': len(X_new),
                    'learning_time': learning_session['learning_time'],
                    'new_learning_rate': self.learning_rate
                }
                
        except Exception as e:
            logger.error(f"Incremental learning failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    def set_model(self, model: Any):
        """Set the current model for incremental learning."""
        with self._lock:
            self.current_model = model
            logger.info("🔄 Model set for incremental learning")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get incremental learning statistics."""
        if not self.learning_history:
            return {}
        
        total_samples = sum(session['samples_added'] for session in self.learning_history)
        total_time = sum(session['learning_time'] for session in self.learning_history)
        
        return {
            'total_learning_sessions': len(self.learning_history),
            'total_samples_learned': total_samples,
            'total_learning_time': total_time,
            'average_samples_per_session': total_samples / len(self.learning_history),
            'current_learning_rate': self.learning_rate
        }

class PerformanceMemoryManager:
    """Main performance and memory management system."""
    
    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()
        
        # Initialize components
        self.model_cache = ModelCache(
            self.config.cache_directory,
            self.config.max_cache_size,
            self.config.cache_ttl
        ) if self.config.enable_model_caching else None
        
        self.resource_monitor = ResourceMonitor(self.config)
        self.incremental_learner = IncrementalLearner(self.config)
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info("🚀 Performance & Memory Manager initialized")
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        if self.config.enable_performance_monitoring:
            self.resource_monitor.start_monitoring()
        
        logger.info("📊 All monitoring systems started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.resource_monitor.stop_monitoring()
        
        logger.info("📊 All monitoring systems stopped")
    
    @contextmanager
    def memory_efficient_processing(self, operation_name: str = "processing"):
        """Context manager for memory-efficient processing."""
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        try:
            yield
        finally:
            end_memory = psutil.virtual_memory().used / (1024**3)
            memory_delta = end_memory - start_memory
            
            if memory_delta > 0.1:  # Log if memory increased by more than 100MB
                logger.info(f"🧠 {operation_name} memory usage: {memory_delta:.2f} GB")
    
    def process_large_dataset(self, X: np.ndarray, y: np.ndarray, 
                            processing_func: Callable, 
                            chunk_size: Optional[int] = None) -> Any:
        """Process large dataset in chunks to manage memory."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        logger.info(f"📊 Processing large dataset in chunks of {chunk_size}")
        
        results = []
        
        for i in range(0, len(X), chunk_size):
            end_idx = min(i + chunk_size, len(X))
            X_chunk = X[i:end_idx]
            y_chunk = y[i:end_idx]
            
            try:
                with self.memory_efficient_processing(f"chunk_{i//chunk_size}"):
                    chunk_result = processing_func(X_chunk, y_chunk)
                    results.append(chunk_result)
                
                # Force garbage collection between chunks
                gc.collect()
                
            except Exception as e:
                logger.error(f"Chunk processing failed at {i}: {e}")
                continue
        
        logger.info(f"✅ Processed {len(results)} chunks")
        return results
    
    def get_cached_model(self, model_config: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Optional[Any]:
        """Get cached model if available."""
        if not self.model_cache:
            return None
        
        return self.model_cache.get_cached_model(model_config, X, y)
    
    def cache_model(self, model: Any, model_config: Dict[str, Any], 
                   X: np.ndarray, y: np.ndarray, performance_metrics: Dict[str, float]):
        """Cache trained model."""
        if not self.model_cache:
            return
        
        self.model_cache.cache_model(model, model_config, X, y, performance_metrics)
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        if self.model_cache:
            self.model_cache.cleanup_expired_entries()
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance and memory analytics."""
        analytics = {
            'resource_monitor': self.resource_monitor.get_performance_summary(),
            'incremental_learner': self.incremental_learner.get_learning_stats(),
            'model_cache': self.model_cache.get_cache_stats() if self.model_cache else {},
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
            }
        }
        
        return analytics


# Factory functions
def create_performance_memory_manager(config: Optional[ResourceConfig] = None) -> PerformanceMemoryManager:
    """Create performance and memory manager."""
    return PerformanceMemoryManager(config)