"""
Advanced Memory Optimizer for Apple Silicon.

This module extends the basic M1MemoryOptimizer with advanced features including
intelligent memory pooling, predictive memory management, and advanced optimization strategies.
"""

import logging
import time
import threading
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import weakref
import queue
import asyncio
from collections import deque, defaultdict
import json
from pathlib import Path

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

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

from .m1_memory_optimizer import M1MemoryOptimizer

logger = logging.getLogger(__name__)

class MemoryPoolType(Enum):
    """Types of memory pools."""
    SMALL_OBJECTS = "small_objects"
    LARGE_OBJECTS = "large_objects"
    NUMPY_ARRAYS = "numpy_arrays"
    PANDAS_DATAFRAMES = "pandas_dataframes"
    TENSOR_DATA = "tensor_data"
    CACHE_DATA = "cache_data"

class MemoryStrategy(Enum):
    """Memory management strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"

class MemoryEventType(Enum):
    """Types of memory events."""
    ALLOCATION = "allocation"
    DEALLOCATION = "deallocation"
    GARBAGE_COLLECTION = "garbage_collection"
    MEMORY_PRESSURE = "memory_pressure"
    POOL_GROWTH = "pool_growth"
    POOL_SHRINK = "pool_shrink"

@dataclass
class MemoryPool:
    """Memory pool configuration."""
    pool_type: MemoryPoolType
    initial_size_mb: float
    max_size_mb: float
    min_size_mb: float
    growth_factor: float = 1.5
    shrink_threshold: float = 0.3
    enable_compression: bool = True
    compression_ratio: float = 0.7
    auto_cleanup: bool = True
    cleanup_interval: float = 300.0  # 5 minutes

@dataclass
class MemoryEvent:
    """Memory event record."""
    event_type: MemoryEventType
    timestamp: float
    size_bytes: int
    pool_type: Optional[MemoryPoolType] = None
    object_type: Optional[str] = None
    memory_usage_before: float = 0.0
    memory_usage_after: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryPrediction:
    """Memory usage prediction."""
    timestamp: float
    predicted_usage_mb: float
    confidence: float
    time_horizon_minutes: int
    factors: Dict[str, float]

class IntelligentMemoryPool:
    """Intelligent memory pool with advanced management."""

    def __init__(self, config: MemoryPool):
        self.config = config
        self.logger = logger.getChild(f'MemoryPool-{config.pool_type.value}')

        # Pool state
        self.current_size_mb = config.initial_size_mb
        self.allocated_mb = 0.0
        self.free_mb = config.initial_size_mb
        self.allocation_count = 0
        self.deallocation_count = 0

        # Object tracking
        self.allocated_objects: Dict[str, Dict[str, Any]] = {}
        self.object_lifecycle_stats: Dict[str, List[float]] = defaultdict(list)

        # Performance tracking
        self.allocation_times: deque = deque(maxlen=1000)
        self.deallocation_times: deque = deque(maxlen=1000)
        self.utilization_history: deque = deque(maxlen=100)

        # Cleanup
        self.last_cleanup = time.time()
        self.cleanup_thread: Optional[threading.Thread] = None

        if config.auto_cleanup:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start automatic cleanup thread."""
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()

    def _cleanup_loop(self):
        """Automatic cleanup loop."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval)
                self._perform_cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    def allocate(self, size_bytes: int, object_id: str,
                object_type: str = "unknown") -> bool:
        """Allocate memory from the pool."""
        try:
            size_mb = size_bytes / (1024 * 1024)

            # Check if we need to grow the pool
            if size_mb > self.free_mb:
                if not self._grow_pool(size_mb):
                    self.logger.warning(f"Insufficient memory in pool {self.config.pool_type.value}")
                    return False

            # Record allocation
            allocation_info = {
                'size_bytes': size_bytes,
                'size_mb': size_mb,
                'object_type': object_type,
                'allocated_at': time.time(),
                'access_count': 0,
                'last_accessed': time.time()
            }

            self.allocated_objects[object_id] = allocation_info
            self.allocated_mb += size_mb
            self.free_mb -= size_mb
            self.allocation_count += 1

            # Record allocation time
            self.allocation_times.append(time.time())

            # Update utilization history
            utilization = (self.allocated_mb / self.current_size_mb) * 100
            self.utilization_history.append(utilization)

            self.logger.debug(f"Allocated {size_mb:.2f}MB for {object_id} in {self.config.pool_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Allocation failed for {object_id}: {e}")
            return False

    def deallocate(self, object_id: str) -> bool:
        """Deallocate memory from the pool."""
        try:
            if object_id not in self.allocated_objects:
                self.logger.warning(f"Object {object_id} not found in pool")
                return False

            allocation_info = self.allocated_objects[object_id]
            size_mb = allocation_info['size_mb']

            # Record lifecycle statistics
            lifecycle_duration = time.time() - allocation_info['allocated_at']
            self.object_lifecycle_stats[allocation_info['object_type']].append(lifecycle_duration)

            # Update pool state
            del self.allocated_objects[object_id]
            self.allocated_mb -= size_mb
            self.free_mb += size_mb
            self.deallocation_count += 1

            # Record deallocation time
            self.deallocation_times.append(time.time())

            # Update utilization history
            utilization = (self.allocated_mb / self.current_size_mb) * 100
            self.utilization_history.append(utilization)

            # Check if we should shrink the pool
            if self._should_shrink_pool():
                self._shrink_pool()

            self.logger.debug(f"Deallocated {size_mb:.2f}MB for {object_id} from {self.config.pool_type.value}")
            return True

        except Exception as e:
            self.logger.error(f"Deallocation failed for {object_id}: {e}")
            return False

    def _grow_pool(self, required_size_mb: float) -> bool:
        """Grow the memory pool."""
        try:
            new_size = max(
                self.current_size_mb * self.config.growth_factor,
                self.current_size_mb + required_size_mb
            )

            if new_size > self.config.max_size_mb:
                self.logger.warning(f"Cannot grow pool beyond max size {self.config.max_size_mb}MB")
                return False

            size_increase = new_size - self.current_size_mb
            self.current_size_mb = new_size
            self.free_mb += size_increase

            self.logger.info(f"Grew {self.config.pool_type.value} pool by {size_increase:.2f}MB to {new_size:.2f}MB")
            return True

        except Exception as e:
            self.logger.error(f"Pool growth failed: {e}")
            return False

    def _should_shrink_pool(self) -> bool:
        """Check if pool should be shrunk."""
        utilization = (self.allocated_mb / self.current_size_mb) * 100
        return (utilization < self.config.shrink_threshold * 100 and
                self.current_size_mb > self.config.min_size_mb)

    def _shrink_pool(self):
        """Shrink the memory pool."""
        try:
            new_size = max(
                self.config.min_size_mb,
                self.allocated_mb * 1.2  # Keep 20% headroom
            )

            if new_size < self.current_size_mb:
                size_decrease = self.current_size_mb - new_size
                self.current_size_mb = new_size
                self.free_mb = max(0, self.free_mb - size_decrease)

                self.logger.info(f"Shrank {self.config.pool_type.value} pool by {size_decrease:.2f}MB to {new_size:.2f}MB")

        except Exception as e:
            self.logger.error(f"Pool shrink failed: {e}")

    def _perform_cleanup(self):
        """Perform pool cleanup."""
        try:
            current_time = time.time()
            cleanup_threshold = 3600  # 1 hour

            # Remove old unused objects
            objects_to_remove = []
            for object_id, info in self.allocated_objects.items():
                if (current_time - info['last_accessed'] > cleanup_threshold and
                    info['access_count'] == 0):
                    objects_to_remove.append(object_id)

            for object_id in objects_to_remove:
                self.deallocate(object_id)

            if objects_to_remove:
                self.logger.info(f"Cleaned up {len(objects_to_remove)} unused objects from {self.config.pool_type.value}")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        utilization = (self.allocated_mb / self.current_size_mb) * 100 if self.current_size_mb > 0 else 0

        # Calculate average lifecycle duration
        avg_lifecycle = {}
        for obj_type, durations in self.object_lifecycle_stats.items():
            if durations:
                avg_lifecycle[obj_type] = sum(durations) / len(durations)

        return {
            'pool_type': self.config.pool_type.value,
            'current_size_mb': self.current_size_mb,
            'allocated_mb': self.allocated_mb,
            'free_mb': self.free_mb,
            'utilization_percent': utilization,
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count,
            'active_objects': len(self.allocated_objects),
            'average_lifecycle_duration': avg_lifecycle,
            'utilization_trend': list(self.utilization_history)[-10:] if self.utilization_history else []
        }

class PredictiveMemoryManager:
    """Predictive memory management system."""

    def __init__(self):
        self.logger = logger.getChild('PredictiveMemoryManager')
        self.memory_history: deque = deque(maxlen=1000)
        self.prediction_models: Dict[str, Any] = {}
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)

    def record_memory_usage(self, usage_mb: float, context: Dict[str, Any] = None):
        """Record memory usage for prediction."""
        record = {
            'timestamp': time.time(),
            'usage_mb': usage_mb,
            'context': context or {}
        }
        self.memory_history.append(record)

    def predict_memory_usage(self, time_horizon_minutes: int = 30) -> MemoryPrediction:
        """Predict future memory usage."""
        try:
            if len(self.memory_history) < 10:
                return MemoryPrediction(
                    timestamp=time.time(),
                    predicted_usage_mb=0.0,
                    confidence=0.0,
                    time_horizon_minutes=time_horizon_minutes,
                    factors={}
                )

            # Simple linear regression prediction
            recent_data = list(self.memory_history)[-50:]  # Last 50 records
            timestamps = [r['timestamp'] for r in recent_data]
            usages = [r['usage_mb'] for r in recent_data]

            # Calculate trend
            if len(timestamps) > 1:
                time_span = timestamps[-1] - timestamps[0]
                usage_span = usages[-1] - usages[0]
                trend = usage_span / time_span if time_span > 0 else 0
            else:
                trend = 0

            # Predict future usage
            current_usage = usages[-1]
            future_time = time_horizon_minutes * 60
            predicted_usage = current_usage + (trend * future_time)

            # Calculate confidence based on data consistency
            if NUMPY_AVAILABLE and len(usages) > 1:
                usage_variance = np.var(usages)
            else:
                # Simple variance calculation without numpy
                if len(usages) > 1:
                    mean_usage = sum(usages) / len(usages)
                    usage_variance = sum((u - mean_usage) ** 2 for u in usages) / len(usages)
                else:
                    usage_variance = 0
            confidence = max(0, 1 - (usage_variance / max(1, current_usage)))

            # Identify contributing factors
            factors = {
                'trend': trend,
                'current_usage': current_usage,
                'data_points': len(recent_data),
                'variance': usage_variance
            }

            prediction = MemoryPrediction(
                timestamp=time.time(),
                predicted_usage_mb=max(0, predicted_usage),
                confidence=confidence,
                time_horizon_minutes=time_horizon_minutes,
                factors=factors
            )

            self.logger.debug(f"Predicted memory usage: {predicted_usage:.2f}MB in {time_horizon_minutes}min (confidence: {confidence:.2f})")
            return prediction

        except Exception as e:
            self.logger.error(f"Memory prediction failed: {e}")
            return MemoryPrediction(
                timestamp=time.time(),
                predicted_usage_mb=0.0,
                confidence=0.0,
                time_horizon_minutes=time_horizon_minutes,
                factors={'error': str(e)}
            )

    def get_memory_trends(self) -> Dict[str, Any]:
        """Get memory usage trends."""
        if len(self.memory_history) < 2:
            return {"error": "Insufficient data"}

        recent_data = list(self.memory_history)[-100:]  # Last 100 records
        usages = [r['usage_mb'] for r in recent_data]
        timestamps = [r['timestamp'] for r in recent_data]

        # Calculate trends
        if len(usages) > 1:
            time_span = timestamps[-1] - timestamps[0]
            usage_span = usages[-1] - usages[0]
            trend_per_minute = (usage_span / time_span) * 60 if time_span > 0 else 0
        else:
            trend_per_minute = 0

        return {
            'current_usage_mb': usages[-1],
            'average_usage_mb': sum(usages) / len(usages),
            'max_usage_mb': max(usages),
            'min_usage_mb': min(usages),
            'trend_per_minute_mb': trend_per_minute,
            'data_points': len(recent_data),
            'time_span_minutes': (timestamps[-1] - timestamps[0]) / 60 if len(timestamps) > 1 else 0
        }

class MemoryEventTracker:
    """Tracks memory events for analysis and optimization."""

    def __init__(self):
        self.logger = logger.getChild('MemoryEventTracker')
        self.events: deque = deque(maxlen=10000)
        self.event_callbacks: Dict[MemoryEventType, List[Callable]] = defaultdict(list)

    def record_event(self, event: MemoryEvent):
        """Record a memory event."""
        self.events.append(event)

        # Trigger callbacks
        for callback in self.event_callbacks[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")

    def add_event_callback(self, event_type: MemoryEventType, callback: Callable):
        """Add callback for specific event type."""
        self.event_callbacks[event_type].append(callback)

    def get_event_stats(self, event_type: Optional[MemoryEventType] = None) -> Dict[str, Any]:
        """Get event statistics."""
        if event_type:
            filtered_events = [e for e in self.events if e.event_type == event_type]
        else:
            filtered_events = list(self.events)

        if not filtered_events:
            return {"error": "No events found"}

        sizes = [e.size_bytes for e in filtered_events]
        timestamps = [e.timestamp for e in filtered_events]

        return {
            'event_type': event_type.value if event_type else 'all',
            'total_events': len(filtered_events),
            'total_size_bytes': sum(sizes),
            'average_size_bytes': sum(sizes) / len(sizes),
            'max_size_bytes': max(sizes),
            'min_size_bytes': min(sizes),
            'time_span_minutes': (max(timestamps) - min(timestamps)) / 60 if len(timestamps) > 1 else 0,
            'events_per_minute': len(filtered_events) / max(1, (max(timestamps) - min(timestamps)) / 60) if len(timestamps) > 1 else 0
        }

class AdvancedM1MemoryOptimizer(M1MemoryOptimizer):
    """Advanced M1 memory optimizer with intelligent pooling and predictive management."""

    def __init__(self, memory_limit_gb: Optional[float] = None,
                 strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE):
        super().__init__(memory_limit_gb)

        self.strategy = strategy
        self.logger = logger.getChild('AdvancedM1MemoryOptimizer')

        # Initialize advanced components
        self.memory_pools: Dict[MemoryPoolType, IntelligentMemoryPool] = {}
        self.predictive_manager = PredictiveMemoryManager()
        self.event_tracker = MemoryEventTracker()

        # Initialize memory pools
        self._initialize_memory_pools()

        # Set up event callbacks
        self._setup_event_callbacks()

        # Start predictive monitoring
        self._start_predictive_monitoring()

        self.logger.info(f"🧠 Advanced M1 Memory Optimizer initialized with {strategy.value} strategy")

    def _initialize_memory_pools(self):
        """Initialize memory pools for different object types."""
        pool_configs = {
            MemoryPoolType.SMALL_OBJECTS: MemoryPool(
                pool_type=MemoryPoolType.SMALL_OBJECTS,
                initial_size_mb=50.0,
                max_size_mb=200.0,
                min_size_mb=10.0
            ),
            MemoryPoolType.LARGE_OBJECTS: MemoryPool(
                pool_type=MemoryPoolType.LARGE_OBJECTS,
                initial_size_mb=200.0,
                max_size_mb=1000.0,
                min_size_mb=50.0
            ),
            MemoryPoolType.NUMPY_ARRAYS: MemoryPool(
                pool_type=MemoryPoolType.NUMPY_ARRAYS,
                initial_size_mb=300.0,
                max_size_mb=1500.0,
                min_size_mb=100.0
            ),
            MemoryPoolType.PANDAS_DATAFRAMES: MemoryPool(
                pool_type=MemoryPoolType.PANDAS_DATAFRAMES,
                initial_size_mb=400.0,
                max_size_mb=2000.0,
                min_size_mb=100.0
            ),
            MemoryPoolType.TENSOR_DATA: MemoryPool(
                pool_type=MemoryPoolType.TENSOR_DATA,
                initial_size_mb=500.0,
                max_size_mb=2500.0,
                min_size_mb=200.0
            ),
            MemoryPoolType.CACHE_DATA: MemoryPool(
                pool_type=MemoryPoolType.CACHE_DATA,
                initial_size_mb=100.0,
                max_size_mb=500.0,
                min_size_mb=20.0
            )
        }

        for pool_type, config in pool_configs.items():
            self.memory_pools[pool_type] = IntelligentMemoryPool(config)

        self.logger.info(f"🏊 Initialized {len(self.memory_pools)} memory pools")

    def _setup_event_callbacks(self):
        """Set up memory event callbacks."""
        self.event_tracker.add_event_callback(
            MemoryEventType.MEMORY_PRESSURE,
            self._handle_memory_pressure_event
        )
        self.event_tracker.add_event_callback(
            MemoryEventType.ALLOCATION,
            self._handle_allocation_event
        )

    def _start_predictive_monitoring(self):
        """Start predictive memory monitoring."""
        def monitoring_loop():
            while True:
                try:
                    # Record current memory usage
                    memory_stats = self.get_memory_stats()
                    self.predictive_manager.record_memory_usage(
                        memory_stats.get('used_memory', 0) / (1024**3),  # Convert to GB
                        {'strategy': self.strategy.value}
                    )

                    # Make predictions
                    prediction = self.predictive_manager.predict_memory_usage(30)  # 30 minutes

                    # Take action based on predictions
                    if (prediction and prediction.confidence and prediction.confidence > 0.7 and
                        prediction.predicted_usage_mb is not None and prediction.predicted_usage_mb > self.memory_limit_gb * 0.9):
                        self.logger.warning(f"🚨 Predicted memory pressure in 30min: {prediction.predicted_usage_mb:.2f}GB")
                        self._proactive_memory_cleanup()

                    time.sleep(60)  # Check every minute

                except Exception as e:
                    self.logger.error(f"Predictive monitoring error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error

        threading.Thread(target=monitoring_loop, daemon=True).start()
        self.logger.info("🔮 Predictive memory monitoring started")

    def _handle_memory_pressure_event(self, event: MemoryEvent):
        """Handle memory pressure events."""
        self.logger.warning(f"🚨 Memory pressure event: {event.size_bytes / (1024**2):.2f}MB")

        # Adjust strategy based on pressure level
        if event.size_bytes > self.memory_limit_bytes * 0.9:
            self.strategy = MemoryStrategy.AGGRESSIVE
            self._aggressive_memory_cleanup()
        elif event.size_bytes > self.memory_limit_bytes * 0.8:
            self.strategy = MemoryStrategy.BALANCED
            self._moderate_memory_cleanup()

    def _handle_allocation_event(self, event: MemoryEvent):
        """Handle allocation events."""
        # Update predictive models based on allocation patterns
        pass

    def aggressive_cleanup(self, force_cleanup: bool = False, clear_caches: bool = True,
                          compress_memory: bool = True, optimize_pools: bool = True) -> Dict[str, Any]:
        """Perform aggressive memory cleanup."""
        cleanup_results = {
            'success': False,
            'memory_freed_mb': 0.0,
            'methods_used': [],
            'errors': []
        }

        try:
            self.logger.info("🧹 Performing aggressive memory cleanup")

            # Clear caches if requested
            if clear_caches:
                self._clear_all_caches()
                cleanup_results['methods_used'].append('cache_clear')

            # Optimize pools if requested
            if optimize_pools:
                for pool_type, pool in self.memory_pools.items():
                    pool._perform_cleanup()
                cleanup_results['methods_used'].append('pool_optimization')

            # Compress memory if requested
            if compress_memory:
                self._compress_memory_pools()
                cleanup_results['methods_used'].append('memory_compression')

            # Force garbage collection
            import gc
            collected = gc.collect()
            cleanup_results['memory_freed_mb'] = collected * 0.001  # Rough estimate
            cleanup_results['methods_used'].append('garbage_collection')

            # Proactive cleanup
            self._proactive_memory_cleanup()
            cleanup_results['methods_used'].append('proactive_cleanup')

            cleanup_results['success'] = True
            self.logger.info(f"✅ Aggressive cleanup completed: {cleanup_results['memory_freed_mb']:.1f}MB freed")

        except Exception as e:
            cleanup_results['errors'].append(str(e))
            self.logger.error(f"❌ Aggressive cleanup failed: {e}")

        return cleanup_results

    def _clear_all_caches(self):
        """Clear all internal caches."""
        # Clear event tracker cache
        if hasattr(self, 'event_tracker') and hasattr(self.event_tracker, 'clear_cache'):
            self.event_tracker.clear_cache()
        elif hasattr(self, 'event_tracker'):
            # Fallback: clear events manually
            if hasattr(self.event_tracker, 'events'):
                self.event_tracker.events.clear()

        # Clear predictive manager cache
        if hasattr(self, 'predictive_manager') and hasattr(self.predictive_manager, 'clear_cache'):
            self.predictive_manager.clear_cache()
        elif hasattr(self, 'predictive_manager'):
            # Fallback: clear predictions manually
            if hasattr(self.predictive_manager, 'predictions'):
                self.predictive_manager.predictions.clear()

    def _compress_memory_pools(self):
        """Compress memory pools to free up space."""
        for pool_type, pool in self.memory_pools.items():
            if hasattr(pool, 'compress'):
                pool.compress()

    def _proactive_memory_cleanup(self):
        """Perform proactive memory cleanup based on predictions."""
        self.logger.info("🧹 Performing proactive memory cleanup")

        # Clean up unused pools
        for pool_type, pool in self.memory_pools.items():
            if pool.config.auto_cleanup:
                pool._perform_cleanup()

        # Force garbage collection
        collected = gc.collect()
        if collected > 0:
            self.logger.info(f"🧹 Proactive GC collected {collected} objects")

    def allocate_from_pool(self, pool_type: MemoryPoolType, size_bytes: int,
                          object_id: str, object_type: str = "unknown") -> bool:
        """Allocate memory from a specific pool."""
        if pool_type not in self.memory_pools:
            self.logger.error(f"Memory pool {pool_type.value} not found")
            return False

        pool = self.memory_pools[pool_type]
        success = pool.allocate(size_bytes, object_id, object_type)

        if success:
            # Record allocation event
            event = MemoryEvent(
                event_type=MemoryEventType.ALLOCATION,
                timestamp=time.time(),
                size_bytes=size_bytes,
                pool_type=pool_type,
                object_type=object_type,
                memory_usage_before=self.get_memory_stats().get('used_memory', 0),
                memory_usage_after=self.get_memory_stats().get('used_memory', 0)
            )
            self.event_tracker.record_event(event)

        return success

    def deallocate_from_pool(self, pool_type: MemoryPoolType, object_id: str) -> bool:
        """Deallocate memory from a specific pool."""
        if pool_type not in self.memory_pools:
            self.logger.error(f"Memory pool {pool_type.value} not found")
            return False

        pool = self.memory_pools[pool_type]
        success = pool.deallocate(object_id)

        if success:
            # Record deallocation event
            event = MemoryEvent(
                event_type=MemoryEventType.DEALLOCATION,
                timestamp=time.time(),
                size_bytes=0,  # Size will be determined by pool
                pool_type=pool_type,
                memory_usage_before=self.get_memory_stats().get('used_memory', 0),
                memory_usage_after=self.get_memory_stats().get('used_memory', 0)
            )
            self.event_tracker.record_event(event)

        return success

    def optimize_dataframe_advanced(self, df):
        """Advanced DataFrame optimization with intelligent pooling."""
        if df is None or df.empty:
            return df

        try:
            # Determine appropriate pool type
            size_mb = df.memory_usage(deep=True).sum() / (1024**2)
            if size_mb > 100:
                pool_type = MemoryPoolType.LARGE_OBJECTS
            else:
                pool_type = MemoryPoolType.PANDAS_DATAFRAMES

            # Allocate from pool
            object_id = f"dataframe_{id(df)}"
            self.allocate_from_pool(pool_type, df.memory_usage(deep=True).sum(),
                                  object_id, "pandas_dataframe")

            # Apply optimizations
            optimized_df = self.optimize_dataframe_memory(df)

            # Record optimization event
            event = MemoryEvent(
                event_type=MemoryEventType.ALLOCATION,
                timestamp=time.time(),
                size_bytes=optimized_df.memory_usage(deep=True).sum(),
                pool_type=pool_type,
                object_type="optimized_dataframe",
                metadata={'original_size_mb': size_mb}
            )
            self.event_tracker.record_event(event)

            return optimized_df

        except Exception as e:
            self.logger.error(f"Advanced DataFrame optimization failed: {e}")
            return df

    def get_advanced_memory_stats(self) -> Dict[str, Any]:
        """Get advanced memory statistics."""
        base_stats = self.get_memory_stats()

        # Pool statistics
        pool_stats = {}
        for pool_type, pool in self.memory_pools.items():
            pool_stats[pool_type.value] = pool.get_pool_stats()

        # Predictive statistics
        prediction = self.predictive_manager.predict_memory_usage(30)
        trends = self.predictive_manager.get_memory_trends()

        # Event statistics
        event_stats = {}
        for event_type in MemoryEventType:
            stats = self.event_tracker.get_event_stats(event_type)
            if 'error' not in stats:
                event_stats[event_type.value] = stats

        return {
            **base_stats,
            'strategy': self.strategy.value,
            'memory_pools': pool_stats,
            'predictive_analysis': {
                'prediction_30min': {
                    'usage_mb': prediction.predicted_usage_mb,
                    'confidence': prediction.confidence,
                    'factors': prediction.factors
                },
                'trends': trends
            },
            'event_statistics': event_stats,
            'advanced_features': {
                'intelligent_pooling': True,
                'predictive_management': True,
                'event_tracking': True,
                'proactive_cleanup': True
            }
        }

    def get_memory_pool_stats(self, pool_type: MemoryPoolType) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific memory pool."""
        if pool_type in self.memory_pools:
            return self.memory_pools[pool_type].get_pool_stats()
        return None

    def get_memory_predictions(self, time_horizon_minutes: int = 30) -> MemoryPrediction:
        """Get memory usage predictions."""
        return self.predictive_manager.predict_memory_usage(time_horizon_minutes)

    def get_memory_trends(self) -> Dict[str, Any]:
        """Get memory usage trends."""
        return self.predictive_manager.get_memory_trends()

    def set_memory_strategy(self, strategy: MemoryStrategy):
        """Set memory management strategy."""
        old_strategy = self.strategy
        self.strategy = strategy
        self.logger.info(f"🧠 Memory strategy changed: {old_strategy.value} -> {strategy.value}")

        # Adjust pool configurations based on strategy
        self._adjust_pools_for_strategy()

    def _adjust_pools_for_strategy(self):
        """Adjust memory pools based on strategy."""
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            # Reduce pool sizes for aggressive cleanup
            for pool in self.memory_pools.values():
                pool.config.shrink_threshold = 0.5
                pool.config.cleanup_interval = 60.0  # 1 minute
        elif self.strategy == MemoryStrategy.CONSERVATIVE:
            # Increase pool sizes for conservative approach
            for pool in self.memory_pools.values():
                pool.config.shrink_threshold = 0.2
                pool.config.cleanup_interval = 600.0  # 10 minutes
        else:  # BALANCED or ADAPTIVE
            # Default settings
            for pool in self.memory_pools.values():
                pool.config.shrink_threshold = 0.3
                pool.config.cleanup_interval = 300.0  # 5 minutes

# Global instance
_advanced_memory_optimizer: Optional[AdvancedM1MemoryOptimizer] = None

def get_advanced_memory_optimizer(memory_limit_gb: Optional[float] = None,
                                strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE) -> AdvancedM1MemoryOptimizer:
    """Get the global advanced memory optimizer instance."""
    global _advanced_memory_optimizer

    if _advanced_memory_optimizer is None:
        _advanced_memory_optimizer = AdvancedM1MemoryOptimizer(memory_limit_gb, strategy)

    return _advanced_memory_optimizer

def optimize_dataframe_advanced(df):
    """Convenience function for advanced DataFrame optimization."""
    optimizer = get_advanced_memory_optimizer()
    return optimizer.optimize_dataframe_advanced(df)

def get_memory_predictions(time_horizon_minutes: int = 30) -> MemoryPrediction:
    """Get memory usage predictions."""
    optimizer = get_advanced_memory_optimizer()
    return optimizer.get_memory_predictions(time_horizon_minutes)
