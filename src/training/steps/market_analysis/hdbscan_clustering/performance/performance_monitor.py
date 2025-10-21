"""
Performance Monitor and Caching System

This module provides real-time performance monitoring, intelligent caching,
and optimization recommendations for the data-driven clustering system.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import queue
import hashlib
import pickle
from functools import wraps
import gc
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import tprint utilities for extensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Tracks:
    - Execution times
    - Memory usage
    - CPU utilization
    - Cache hit rates
    - Optimization effectiveness
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, 
                 monitoring_interval: float = 1.0,  # seconds
                 max_history: int = 1000,
                 enable_alerts: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            monitoring_interval: Interval for monitoring (seconds)
            max_history: Maximum number of historical records
            enable_alerts: Whether to enable performance alerts
        """
        self.monitoring_interval = monitoring_interval
        self.max_history = max_history
        self.enable_alerts = enable_alerts
        
        # Performance tracking
        self.execution_times: deque = deque(maxlen=max_history)
        self.memory_usage: deque = deque(maxlen=max_history)
        self.cpu_usage: deque = deque(maxlen=max_history)
        self.cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.monitoring_active = False
        
        # Performance thresholds
        self.thresholds = {
            'max_execution_time': 60.0,  # seconds
            'max_memory_usage': 2000.0,  # MB
            'max_cpu_usage': 80.0,  # percentage
            'min_cache_hit_rate': 0.7  # 70%
        }
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.stop_monitoring.clear()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return
        
        self.monitoring_active = False
        self.stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                # Record performance metrics
                self._record_metrics()
                
                # Check thresholds
                if self.enable_alerts:
                    self._check_thresholds()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _record_metrics(self):
        """Record current performance metrics."""
        timestamp = datetime.now()
        
        # Memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_usage.append({
            'timestamp': timestamp,
            'memory_mb': memory_mb
        })
        
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.append({
            'timestamp': timestamp,
            'cpu_percent': cpu_percent
        })
    
    def _check_thresholds(self):
        """Check performance thresholds and generate alerts."""
        if not self.memory_usage or not self.cpu_usage:
            return
        
        current_memory = self.memory_usage[-1]['memory_mb']
        current_cpu = self.cpu_usage[-1]['cpu_percent']
        
        # Check memory threshold
        if current_memory > self.thresholds['max_memory_usage']:
            self._generate_alert('high_memory', {
                'current': current_memory,
                'threshold': self.thresholds['max_memory_usage'],
                'message': f"Memory usage {current_memory:.1f}MB exceeds threshold {self.thresholds['max_memory_usage']:.1f}MB"
            })
        
        # Check CPU threshold
        if current_cpu > self.thresholds['max_cpu_usage']:
            self._generate_alert('high_cpu', {
                'current': current_cpu,
                'threshold': self.thresholds['max_cpu_usage'],
                'message': f"CPU usage {current_cpu:.1f}% exceeds threshold {self.thresholds['max_cpu_usage']:.1f}%"
            })
    
    def _generate_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Generate performance alert."""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'data': alert_data
        }
        
        self.alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Performance alert: {alert_data['message']}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def record_execution_time(self, operation: str, execution_time: float):
        """Record execution time for an operation."""
        self.execution_times.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'execution_time': execution_time
        })
        
        # Check execution time threshold
        if execution_time > self.thresholds['max_execution_time']:
            self._generate_alert('slow_operation', {
                'operation': operation,
                'execution_time': execution_time,
                'threshold': self.thresholds['max_execution_time'],
                'message': f"Operation '{operation}' took {execution_time:.2f}s, exceeding threshold {self.thresholds['max_execution_time']:.2f}s"
            })
    
    def record_cache_stats(self, cache_key: str, hit: bool):
        """Record cache statistics."""
        if hit:
            self.cache_stats[cache_key]['hits'] += 1
        else:
            self.cache_stats[cache_key]['misses'] += 1
        
        # Check cache hit rate
        total_requests = self.cache_stats[cache_key]['hits'] + self.cache_stats[cache_key]['misses']
        if total_requests > 10:  # Only check after some requests
            hit_rate = self.cache_stats[cache_key]['hits'] / total_requests
            if hit_rate < self.thresholds['min_cache_hit_rate']:
                self._generate_alert('low_cache_hit_rate', {
                    'cache_key': cache_key,
                    'hit_rate': hit_rate,
                    'threshold': self.thresholds['min_cache_hit_rate'],
                    'message': f"Cache '{cache_key}' hit rate {hit_rate:.1%} below threshold {self.thresholds['min_cache_hit_rate']:.1%}"
                })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        summary = {
            'timestamp': datetime.now(),
            'monitoring_active': self.monitoring_active,
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-10:] if self.alerts else [],
            'execution_times': {
                'total_operations': len(self.execution_times),
                'avg_execution_time': np.mean([e['execution_time'] for e in self.execution_times]) if self.execution_times else 0,
                'max_execution_time': max([e['execution_time'] for e in self.execution_times]) if self.execution_times else 0,
                'recent_operations': list(self.execution_times)[-10:] if self.execution_times else []
            },
            'memory_usage': {
                'current_mb': self.memory_usage[-1]['memory_mb'] if self.memory_usage else 0,
                'avg_mb': np.mean([m['memory_mb'] for m in self.memory_usage]) if self.memory_usage else 0,
                'max_mb': max([m['memory_mb'] for m in self.memory_usage]) if self.memory_usage else 0
            },
            'cpu_usage': {
                'current_percent': self.cpu_usage[-1]['cpu_percent'] if self.cpu_usage else 0,
                'avg_percent': np.mean([c['cpu_percent'] for c in self.cpu_usage]) if self.cpu_usage else 0,
                'max_percent': max([c['cpu_percent'] for c in self.cpu_usage]) if self.cpu_usage else 0
            },
            'cache_stats': dict(self.cache_stats)
        }
        
        return summary


class IntelligentCache:
    """
    Intelligent caching system with automatic invalidation and optimization.
    
    Features:
    - Automatic cache key generation
    - TTL-based expiration
    - Memory-based eviction
    - Cache hit rate optimization
    - Performance monitoring integration
    """
    
    def __init__(self, 
                 max_size_mb: float = 500.0,
                 default_ttl: float = 3600.0,  # 1 hour
                 enable_compression: bool = True,
                 performance_monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize the intelligent cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            default_ttl: Default time-to-live in seconds
            enable_compression: Whether to enable compression
            performance_monitor: Performance monitor instance
        """
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.performance_monitor = performance_monitor
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.creation_times: Dict[str, datetime] = {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Memory tracking
        self.current_size_mb = 0.0
        
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create hash of arguments
        args_str = str(sorted(args)) if args else ""
        kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
        
        # Generate hash
        key_data = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_item_size_mb(self, item: Any) -> float:
        """Estimate item size in MB."""
        try:
            if hasattr(item, 'nbytes'):
                return item.nbytes / 1024 / 1024
            else:
                # Serialize to estimate size
                serialized = pickle.dumps(item)
                return len(serialized) / 1024 / 1024
        except:
            return 1.0  # Default estimate
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache item is expired."""
        if key not in self.creation_times:
            return True
        
        age = (datetime.now() - self.creation_times[key]).total_seconds()
        return age > self.default_ttl
    
    def _evict_oldest(self):
        """Evict oldest cache item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_item(oldest_key)
        self.evictions += 1
    
    def _evict_expired(self):
        """Evict expired cache items."""
        expired_keys = [key for key in self.cache.keys() if self._is_expired(key)]
        for key in expired_keys:
            self._remove_item(key)
    
    def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            # Update size
            item_size = self._get_item_size_mb(self.cache[key]['value'])
            self.current_size_mb -= item_size
            
            # Remove from all tracking
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.creation_times:
                del self.creation_times[key]
    
    def _make_room(self, required_size_mb: float):
        """Make room in cache for new item."""
        # First, evict expired items
        self._evict_expired()
        
        # If still not enough room, evict oldest items
        while (self.current_size_mb + required_size_mb > self.max_size_mb and 
               self.cache):
            self._evict_oldest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key not in self.cache:
            self.misses += 1
            if self.performance_monitor:
                self.performance_monitor.record_cache_stats(key, False)
            return None
        
        # Check if expired
        if self._is_expired(key):
            self._remove_item(key)
            self.misses += 1
            if self.performance_monitor:
                self.performance_monitor.record_cache_stats(key, False)
            return None
        
        # Update access time
        self.access_times[key] = datetime.now()
        
        # Record hit
        self.hits += 1
        if self.performance_monitor:
            self.performance_monitor.record_cache_stats(key, True)
        
        return self.cache[key]['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set item in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        # Estimate size
        item_size_mb = self._get_item_size_mb(value)
        
        # Make room if needed
        if item_size_mb > self.max_size_mb:
            logger.warning(f"Item size {item_size_mb:.2f}MB exceeds cache limit {self.max_size_mb:.2f}MB")
            return False
        
        self._make_room(item_size_mb)
        
        # Store item
        self.cache[key] = {
            'value': value,
            'ttl': ttl
        }
        self.access_times[key] = datetime.now()
        self.creation_times[key] = datetime.now()
        self.current_size_mb += item_size_mb
        
        return True
    
    def clear(self):
        """Clear all cache items."""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
        self.current_size_mb = 0.0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'current_size_mb': self.current_size_mb,
            'max_size_mb': self.max_size_mb,
            'utilization': self.current_size_mb / self.max_size_mb,
            'total_items': len(self.cache)
        }


def cached(ttl: Optional[float] = None, 
          cache_key_func: Optional[Callable] = None,
          cache: Optional[IntelligentCache] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        cache_key_func: Custom cache key function
        cache: Cache instance to use
    """
    def decorator(func):
        if cache is None:
            # Create default cache
            func_cache = IntelligentCache()
        else:
            func_cache = cache
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(func.__name__, args, kwargs)
            else:
                key = func_cache._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = func_cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            func_cache.set(key, result, ttl)
            
            return result
        
        # Add cache stats to function
        wrapper.cache_stats = func_cache.get_stats
        wrapper.cache_clear = func_cache.clear
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """
    Performance optimization system that provides recommendations
    based on monitoring data.
    """
    
    def __init__(self, 
                 performance_monitor: PerformanceMonitor,
                 cache: IntelligentCache):
        """
        Initialize the performance optimizer.
        
        Args:
            performance_monitor: Performance monitor instance
            cache: Cache instance
        """
        self.performance_monitor = performance_monitor
        self.cache = cache
        
        # Optimization recommendations
        self.recommendations: List[Dict[str, Any]] = []
        
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and generate recommendations."""
        summary = self.performance_monitor.get_performance_summary()
        cache_stats = self.cache.get_stats()
        
        recommendations = []
        
        # Memory usage analysis
        if summary['memory_usage']['current_mb'] > 1000:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'high',
                'message': f"High memory usage: {summary['memory_usage']['current_mb']:.1f}MB",
                'suggestions': [
                    "Consider reducing batch sizes",
                    "Enable garbage collection",
                    "Use memory-efficient data types",
                    "Implement data streaming"
                ]
            })
        
        # CPU usage analysis
        if summary['cpu_usage']['current_percent'] > 70:
            recommendations.append({
                'type': 'cpu_optimization',
                'priority': 'medium',
                'message': f"High CPU usage: {summary['cpu_usage']['current_percent']:.1f}%",
                'suggestions': [
                    "Enable parallel processing",
                    "Optimize algorithms",
                    "Use vectorized operations",
                    "Consider distributed computing"
                ]
            })
        
        # Cache efficiency analysis
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'medium',
                'message': f"Low cache hit rate: {cache_stats['hit_rate']:.1%}",
                'suggestions': [
                    "Review cache key generation",
                    "Increase cache size",
                    "Optimize TTL settings",
                    "Improve cache key uniqueness"
                ]
            })
        
        # Execution time analysis
        if summary['execution_times']['avg_execution_time'] > 10:
            recommendations.append({
                'type': 'execution_optimization',
                'priority': 'high',
                'message': f"Slow average execution time: {summary['execution_times']['avg_execution_time']:.2f}s",
                'suggestions': [
                    "Profile slow operations",
                    "Optimize algorithms",
                    "Use caching more effectively",
                    "Consider parallel processing"
                ]
            })
        
        # Cache size analysis
        if cache_stats['utilization'] > 0.8:
            recommendations.append({
                'type': 'cache_size_optimization',
                'priority': 'low',
                'message': f"High cache utilization: {cache_stats['utilization']:.1%}",
                'suggestions': [
                    "Increase cache size",
                    "Implement LRU eviction",
                    "Optimize stored data",
                    "Use compression"
                ]
            })
        
        self.recommendations = recommendations
        
        return {
            'timestamp': datetime.now(),
            'performance_summary': summary,
            'cache_stats': cache_stats,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'high_priority_recommendations': len([r for r in recommendations if r['priority'] == 'high']),
            'medium_priority_recommendations': len([r for r in recommendations if r['priority'] == 'medium']),
            'low_priority_recommendations': len([r for r in recommendations if r['priority'] == 'low'])
        }
    
    def get_optimization_plan(self) -> Dict[str, Any]:
        """Get detailed optimization plan."""
        analysis = self.analyze_performance()
        
        # Group recommendations by type
        recommendations_by_type = defaultdict(list)
        for rec in analysis['recommendations']:
            recommendations_by_type[rec['type']].append(rec)
        
        # Create optimization plan
        plan = {
            'timestamp': datetime.now(),
            'overview': {
                'total_recommendations': analysis['total_recommendations'],
                'high_priority': analysis['high_priority_recommendations'],
                'medium_priority': analysis['medium_priority_recommendations'],
                'low_priority': analysis['low_priority_recommendations']
            },
            'recommendations_by_type': dict(recommendations_by_type),
            'implementation_priority': self._get_implementation_priority(analysis['recommendations']),
            'expected_improvements': self._estimate_improvements(analysis)
        }
        
        return plan
    
    def _get_implementation_priority(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Get implementation priority order."""
        priority_order = ['high', 'medium', 'low']
        return sorted(recommendations, key=lambda x: priority_order.index(x['priority']))
    
    def _estimate_improvements(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate potential improvements."""
        improvements = {}
        
        # Memory improvements
        if any(r['type'] == 'memory_optimization' for r in analysis['recommendations']):
            improvements['memory_reduction'] = {
                'current_mb': analysis['performance_summary']['memory_usage']['current_mb'],
                'estimated_reduction': '20-40%',
                'estimated_savings_mb': analysis['performance_summary']['memory_usage']['current_mb'] * 0.3
            }
        
        # Cache improvements
        if any(r['type'] == 'cache_optimization' for r in analysis['recommendations']):
            improvements['cache_efficiency'] = {
                'current_hit_rate': analysis['cache_stats']['hit_rate'],
                'estimated_improvement': '30-50%',
                'estimated_new_hit_rate': min(0.9, analysis['cache_stats']['hit_rate'] + 0.3)
            }
        
        # Execution time improvements
        if any(r['type'] == 'execution_optimization' for r in analysis['recommendations']):
            improvements['execution_time'] = {
                'current_avg_time': analysis['performance_summary']['execution_times']['avg_execution_time'],
                'estimated_reduction': '25-50%',
                'estimated_new_time': analysis['performance_summary']['execution_times']['avg_execution_time'] * 0.6
            }
        
        return improvements


# Global instances
_performance_monitor = None
_intelligent_cache = None
_performance_optimizer = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_intelligent_cache() -> IntelligentCache:
    """Get global intelligent cache instance."""
    global _intelligent_cache
    if _intelligent_cache is None:
        _intelligent_cache = IntelligentCache(performance_monitor=get_performance_monitor())
    return _intelligent_cache


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer(
            performance_monitor=get_performance_monitor(),
            cache=get_intelligent_cache()
        )
    return _performance_optimizer


def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


def get_performance_summary() -> Dict[str, Any]:
    """Get current performance summary."""
    monitor = get_performance_monitor()
    cache = get_intelligent_cache()
    
    return {
        'monitor': monitor.get_performance_summary(),
        'cache': cache.get_stats(),
        'optimizer': get_performance_optimizer().analyze_performance()
    }


if __name__ == "__main__":
    # Example usage
    print("Performance monitoring and caching system example")
    
    # Start monitoring
    start_performance_monitoring()
    
    # Get cache
    cache = get_intelligent_cache()
    
    # Example cached function
    @cached(ttl=300, cache=cache)
    def expensive_computation(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * y + x ** 2
    
    # Test caching
    print("Testing caching...")
    
    # First call (cache miss)
    start_time = time.time()
    result1 = expensive_computation(5, 10)
    time1 = time.time() - start_time
    print(f"First call: {result1} in {time1:.3f}s")
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = expensive_computation(5, 10)
    time2 = time.time() - start_time
    print(f"Second call: {result2} in {time2:.3f}s")
    
    # Get performance summary
    summary = get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Cache hit rate: {summary['cache']['hit_rate']:.1%}")
    print(f"Memory usage: {summary['monitor']['memory_usage']['current_mb']:.1f}MB")
    print(f"CPU usage: {summary['monitor']['cpu_usage']['current_percent']:.1f}%")
    
    # Stop monitoring
    stop_performance_monitoring()