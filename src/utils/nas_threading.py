#!/usr/bin/env python3
"""
Thread-Safe Utilities for NAS Components

This module provides thread-safe utilities, synchronization primitives,
and concurrent execution patterns to prevent race conditions and ensure
thread safety across NAS operations.
"""

import threading
import time
import queue
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import logging
import weakref
from functools import wraps

from .nas_error_handling import (
    NASThreadingError, ErrorContext, error_context, 
    safe_execute, get_error_handler
)

T = TypeVar('T')


class LockType(Enum):
    """Types of locks available."""
    RLock = "rlock"
    Lock = "lock"
    Semaphore = "semaphore"
    Event = "event"
    Condition = "condition"


@dataclass
class LockInfo:
    """Information about a lock."""
    lock_id: str
    lock_type: LockType
    created_at: float
    acquired_count: int = 0
    last_acquired: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreadSafeCounter:
    """Thread-safe counter with overflow protection."""
    
    def __init__(self, initial_value: int = 0, max_value: Optional[int] = None):
        self._value = initial_value
        self._max_value = max_value
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            try:
                new_value = self._value + amount
                
                if self._max_value is not None and new_value > self._max_value:
                    context = ErrorContext("counter_overflow", "thread_safe_counter")
                    self._error_handler.handle_error(
                        NASThreadingError(f"Counter overflow: {new_value} > {self._max_value}"),
                        context,
                        reraise=False
                    )
                    new_value = self._max_value
                
                self._value = new_value
                return self._value
                
            except Exception as e:
                context = ErrorContext("increment_counter", "thread_safe_counter")
                self._error_handler.handle_error(e, context, reraise=False)
                return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            try:
                new_value = self._value - amount
                self._value = max(0, new_value)  # Prevent negative values
                return self._value
                
            except Exception as e:
                context = ErrorContext("decrement_counter", "thread_safe_counter")
                self._error_handler.handle_error(e, context, reraise=False)
                return self._value
    
    def get_value(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self) -> None:
        """Reset counter to initial value."""
        with self._lock:
            self._value = 0


class ThreadSafeCache(Generic[T]):
    """Thread-safe cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0):
        self._cache: Dict[str, T] = {}
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        with self._lock:
            try:
                if key not in self._cache:
                    return None
                
                # Check TTL
                if time.time() - self._timestamps[key] > self._ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                
                return self._cache[key]
                
            except Exception as e:
                context = ErrorContext("cache_get", "thread_safe_cache")
                self._error_handler.handle_error(e, context, reraise=False)
                return None
    
    def set(self, key: str, value: T) -> None:
        """Set value in cache."""
        with self._lock:
            try:
                # Check size limit
                if len(self._cache) >= self._max_size and key not in self._cache:
                    self._evict_oldest()
                
                self._cache[key] = value
                self._timestamps[key] = time.time()
                
            except Exception as e:
                context = ErrorContext("cache_set", "thread_safe_cache")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def _evict_oldest(self) -> None:
        """Evict oldest entry from cache."""
        try:
            if not self._timestamps:
                return
            
            oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
            
        except Exception as e:
            context = ErrorContext("cache_evict", "thread_safe_cache")
            self._error_handler.handle_error(e, context, reraise=False)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


class ThreadSafeQueue:
    """Thread-safe queue with timeout and error handling."""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
    
    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Put item in queue with timeout."""
        try:
            self._queue.put(item, timeout=timeout)
            return True
        except queue.Full:
            context = ErrorContext("queue_put_timeout", "thread_safe_queue")
            self._error_handler.handle_error(
                NASThreadingError("Queue put operation timed out"),
                context,
                reraise=False
            )
            return False
        except Exception as e:
            context = ErrorContext("queue_put", "thread_safe_queue")
            self._error_handler.handle_error(e, context, reraise=False)
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Get item from queue with timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception as e:
            context = ErrorContext("queue_get", "thread_safe_queue")
            self._error_handler.handle_error(e, context, reraise=False)
            return None
    
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()


class LockManager:
    """Manages locks to prevent deadlocks and provide monitoring."""
    
    def __init__(self):
        self._locks: Dict[str, LockInfo] = {}
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
        self._deadlock_detector = DeadlockDetector()
    
    def create_lock(self, lock_id: str, lock_type: LockType = LockType.RLock) -> Any:
        """Create a new lock with monitoring."""
        with self._lock:
            try:
                if lock_id in self._locks:
                    context = ErrorContext("duplicate_lock", "lock_manager")
                    self._error_handler.handle_error(
                        NASThreadingError(f"Lock {lock_id} already exists"),
                        context,
                        reraise=False
                    )
                    return self._get_lock_object(lock_id)
                
                lock_info = LockInfo(
                    lock_id=lock_id,
                    lock_type=lock_type,
                    created_at=time.time()
                )
                
                self._locks[lock_id] = lock_info
                return self._get_lock_object(lock_id)
                
            except Exception as e:
                context = ErrorContext("create_lock", "lock_manager")
                self._error_handler.handle_error(e, context, reraise=False)
                return threading.RLock()
    
    def _get_lock_object(self, lock_id: str) -> Any:
        """Get the actual lock object."""
        lock_info = self._locks[lock_id]
        
        if lock_info.lock_type == LockType.RLock:
            return threading.RLock()
        elif lock_info.lock_type == LockType.Lock:
            return threading.Lock()
        elif lock_info.lock_type == LockType.Semaphore:
            return threading.Semaphore()
        elif lock_info.lock_type == LockType.Event:
            return threading.Event()
        elif lock_info.lock_type == LockType.Condition:
            return threading.Condition()
        else:
            return threading.RLock()
    
    def acquire_lock(self, lock_id: str, timeout: Optional[float] = None) -> bool:
        """Acquire a lock with timeout."""
        with self._lock:
            try:
                if lock_id not in self._locks:
                    context = ErrorContext("lock_not_found", "lock_manager")
                    self._error_handler.handle_error(
                        NASThreadingError(f"Lock {lock_id} not found"),
                        context,
                        reraise=False
                    )
                    return False
                
                lock_info = self._locks[lock_id]
                lock_obj = self._get_lock_object(lock_id)
                
                # Check for potential deadlock
                if self._deadlock_detector.would_cause_deadlock(lock_id):
                    context = ErrorContext("potential_deadlock", "lock_manager")
                    self._error_handler.handle_error(
                        NASThreadingError(f"Acquiring lock {lock_id} would cause deadlock"),
                        context,
                        reraise=False
                    )
                    return False
                
                # Acquire lock
                acquired = lock_obj.acquire(timeout=timeout)
                if acquired:
                    lock_info.acquired_count += 1
                    lock_info.last_acquired = time.time()
                    self._deadlock_detector.record_lock_acquisition(lock_id)
                
                return acquired
                
            except Exception as e:
                context = ErrorContext("acquire_lock", "lock_manager")
                self._error_handler.handle_error(e, context, reraise=False)
                return False
    
    def release_lock(self, lock_id: str) -> bool:
        """Release a lock."""
        with self._lock:
            try:
                if lock_id not in self._locks:
                    return False
                
                lock_obj = self._get_lock_object(lock_id)
                lock_obj.release()
                self._deadlock_detector.record_lock_release(lock_id)
                return True
                
            except Exception as e:
                context = ErrorContext("release_lock", "lock_manager")
                self._error_handler.handle_error(e, context, reraise=False)
                return False
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get statistics about locks."""
        with self._lock:
            return {
                'total_locks': len(self._locks),
                'locks': {
                    lock_id: {
                        'type': info.lock_type.value,
                        'acquired_count': info.acquired_count,
                        'last_acquired': info.last_acquired,
                        'age_seconds': time.time() - info.created_at
                    }
                    for lock_id, info in self._locks.items()
                }
            }


class DeadlockDetector:
    """Detects potential deadlocks by tracking lock acquisition order."""
    
    def __init__(self):
        self._lock_order: Dict[int, List[str]] = {}  # thread_id -> list of acquired locks
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
    
    def record_lock_acquisition(self, lock_id: str) -> None:
        """Record that a lock was acquired."""
        with self._lock:
            try:
                thread_id = threading.get_ident()
                if thread_id not in self._lock_order:
                    self._lock_order[thread_id] = []
                
                self._lock_order[thread_id].append(lock_id)
                
            except Exception as e:
                context = ErrorContext("record_lock_acquisition", "deadlock_detector")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def record_lock_release(self, lock_id: str) -> None:
        """Record that a lock was released."""
        with self._lock:
            try:
                thread_id = threading.get_ident()
                if thread_id in self._lock_order:
                    if lock_id in self._lock_order[thread_id]:
                        self._lock_order[thread_id].remove(lock_id)
                    
                    # Clean up empty thread entries
                    if not self._lock_order[thread_id]:
                        del self._lock_order[thread_id]
                
            except Exception as e:
                context = ErrorContext("record_lock_release", "deadlock_detector")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def would_cause_deadlock(self, lock_id: str) -> bool:
        """Check if acquiring a lock would cause a deadlock."""
        with self._lock:
            try:
                thread_id = threading.get_ident()
                current_locks = self._lock_order.get(thread_id, [])
                
                # Check if any other thread is waiting for a lock we hold
                for other_thread_id, other_locks in self._lock_order.items():
                    if other_thread_id == thread_id:
                        continue
                    
                    if other_locks and other_locks[0] in current_locks:
                        return True
                
                return False
                
            except Exception as e:
                context = ErrorContext("check_deadlock", "deadlock_detector")
                self._error_handler.handle_error(e, context, reraise=False)
                return False


class ThreadPool:
    """Thread pool with error handling and resource management."""
    
    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "nas-worker"):
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
        self._active_tasks = ThreadSafeCounter()
    
    def start(self) -> None:
        """Start the thread pool."""
        with self._lock:
            try:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix=self.thread_name_prefix
                    )
            except Exception as e:
                context = ErrorContext("start_thread_pool", "thread_pool")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop the thread pool."""
        with self._lock:
            try:
                if self._executor is not None:
                    self._executor.shutdown(wait=True, timeout=timeout)
                    self._executor = None
            except Exception as e:
                context = ErrorContext("stop_thread_pool", "thread_pool")
                self._error_handler.handle_error(e, context, reraise=False)
    
    def submit(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task to the thread pool."""
        with self._lock:
            try:
                if self._executor is None:
                    self.start()
                
                self._active_tasks.increment()
                
                def wrapped_func(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        context = ErrorContext("thread_pool_task", "thread_pool")
                        self._error_handler.handle_error(e, context, reraise=False)
                        raise
                    finally:
                        self._active_tasks.decrement()
                
                return self._executor.submit(wrapped_func, *args, **kwargs)
                
            except Exception as e:
                context = ErrorContext("submit_task", "thread_pool")
                self._error_handler.handle_error(e, context, reraise=False)
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self._lock:
            return {
                'max_workers': self.max_workers,
                'active_tasks': self._active_tasks.get_value(),
                'executor_running': self._executor is not None
            }


# Global instances
_global_lock_manager = LockManager()
_global_thread_pool = ThreadPool()


def thread_safe(func: Callable) -> Callable:
    """Decorator to make a function thread-safe."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        lock = threading.RLock()
        with lock:
            return func(*args, **kwargs)
    return wrapper


@contextmanager
def thread_safe_context(lock_id: str, timeout: Optional[float] = None):
    """Context manager for thread-safe operations."""
    lock_manager = get_lock_manager()
    
    try:
        acquired = lock_manager.acquire_lock(lock_id, timeout)
        if not acquired:
            raise NASThreadingError(f"Failed to acquire lock {lock_id}")
        yield
    finally:
        lock_manager.release_lock(lock_id)


def run_in_thread_pool(func: Callable, *args, **kwargs) -> concurrent.futures.Future:
    """Run a function in the thread pool."""
    thread_pool = get_thread_pool()
    return thread_pool.submit(func, *args, **kwargs)


def get_lock_manager() -> LockManager:
    """Get the global lock manager."""
    return _global_lock_manager


def get_thread_pool() -> ThreadPool:
    """Get the global thread pool."""
    return _global_thread_pool


def create_thread_safe_counter(initial_value: int = 0, max_value: Optional[int] = None) -> ThreadSafeCounter:
    """Create a thread-safe counter."""
    return ThreadSafeCounter(initial_value, max_value)


def create_thread_safe_cache(max_size: int = 1000, ttl_seconds: float = 3600.0) -> ThreadSafeCache:
    """Create a thread-safe cache."""
    return ThreadSafeCache(max_size, ttl_seconds)


def create_thread_safe_queue(maxsize: int = 0) -> ThreadSafeQueue:
    """Create a thread-safe queue."""
    return ThreadSafeQueue(maxsize)


# Export main classes and functions
__all__ = [
    'LockType',
    'LockInfo',
    'ThreadSafeCounter',
    'ThreadSafeCache',
    'ThreadSafeQueue',
    'LockManager',
    'DeadlockDetector',
    'ThreadPool',
    'thread_safe',
    'thread_safe_context',
    'run_in_thread_pool',
    'get_lock_manager',
    'get_thread_pool',
    'create_thread_safe_counter',
    'create_thread_safe_cache',
    'create_thread_safe_queue'
]