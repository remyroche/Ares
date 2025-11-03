"""
Component Pool Manager

This module provides a singleton pool for reusable components like
VectorBTRollingOptimizer, SRPerformanceMonitor, etc. to prevent
repeated initialization overhead.

Author: Ares Trading System
Date: 2025-11-02
"""

import logging
import threading
from typing import Dict, Any, Optional, Type
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)


class ComponentPool:
    """
    Thread-safe singleton pool for reusable components.
    
    Prevents repeated initialization of expensive components by
    maintaining a pool of initialized instances that can be reused.
    """
    
    _instance: Optional['ComponentPool'] = None
    _lock = threading.Lock()
    _components: Dict[str, Any] = {}
    _weak_components: WeakValueDictionary = WeakValueDictionary()
    
    def __new__(cls):
        """Ensure only one instance exists (thread-safe singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(ComponentPool, cls).__new__(cls)
                    cls._instance._components = {}
                    cls._instance._weak_components = WeakValueDictionary()
        return cls._instance
    
    def get_or_create(
        self, 
        component_key: str, 
        factory_func: callable,
        use_weak_ref: bool = False,
        **factory_kwargs
    ) -> Any:
        """
        Get an existing component or create a new one.
        
        Args:
            component_key: Unique key for the component
            factory_func: Function to create the component if not exists
            use_weak_ref: Use weak reference (auto-cleanup when no longer referenced)
            **factory_kwargs: Arguments to pass to factory function
            
        Returns:
            Component instance (existing or newly created)
        """
        with self._lock:
            # Check weak references first if enabled
            if use_weak_ref:
                component = self._weak_components.get(component_key)
                if component is not None:
                    logger.debug(f"♻️ Reusing component from weak pool: {component_key}")
                    return component
            else:
                # Check strong references
                if component_key in self._components:
                    logger.debug(f"♻️ Reusing component from pool: {component_key}")
                    return self._components[component_key]
            
            # Component doesn't exist, create it
            logger.debug(f"🔧 Creating new component: {component_key}")
            try:
                component = factory_func(**factory_kwargs)
                
                # Store in appropriate pool
                if use_weak_ref:
                    self._weak_components[component_key] = component
                else:
                    self._components[component_key] = component
                
                logger.debug(f"✅ Component created and cached: {component_key}")
                return component
                
            except Exception as e:
                logger.error(f"❌ Failed to create component {component_key}: {e}")
                raise
    
    def get(self, component_key: str) -> Optional[Any]:
        """
        Get a component from the pool without creating it.
        
        Args:
            component_key: Unique key for the component
            
        Returns:
            Component instance or None if not found
        """
        with self._lock:
            # Check weak references first
            component = self._weak_components.get(component_key)
            if component is not None:
                return component
            
            # Check strong references
            return self._components.get(component_key)
    
    def set(self, component_key: str, component: Any, use_weak_ref: bool = False):
        """
        Manually add a component to the pool.
        
        Args:
            component_key: Unique key for the component
            component: Component instance to store
            use_weak_ref: Use weak reference
        """
        with self._lock:
            if use_weak_ref:
                self._weak_components[component_key] = component
            else:
                self._components[component_key] = component
            logger.debug(f"✅ Component manually cached: {component_key}")
    
    def remove(self, component_key: str):
        """
        Remove a component from the pool.
        
        Args:
            component_key: Unique key for the component
        """
        with self._lock:
            # Remove from both pools
            self._components.pop(component_key, None)
            # Weak references auto-cleanup, but we can try to remove
            if component_key in self._weak_components:
                del self._weak_components[component_key]
            logger.debug(f"🗑️ Component removed from pool: {component_key}")
    
    def clear(self):
        """Clear all components from the pool."""
        with self._lock:
            count = len(self._components) + len(self._weak_components)
            self._components.clear()
            self._weak_components.clear()
            logger.info(f"🗑️ Component pool cleared ({count} components)")
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            return {
                'strong_refs': len(self._components),
                'weak_refs': len(self._weak_components),
                'total': len(self._components) + len(self._weak_components)
            }
    
    @classmethod
    def reset(cls):
        """Reset the singleton (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._components.clear()
                cls._instance._weak_components.clear()
            cls._instance = None


# Global singleton instance accessor
_component_pool_instance: Optional[ComponentPool] = None


def get_component_pool() -> ComponentPool:
    """
    Get the singleton component pool.
    
    Returns:
        ComponentPool: Singleton instance
    """
    global _component_pool_instance
    if _component_pool_instance is None:
        _component_pool_instance = ComponentPool()
    return _component_pool_instance


# Convenience functions for common components

def get_or_create_vectorbt_optimizer(**kwargs) -> Any:
    """
    Get or create VectorBTRollingOptimizer instance.
    
    Returns:
        VectorBTRollingOptimizer instance (cached)
    """
    from ...training.steps.market_analysis.sr_detection.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    
    pool = get_component_pool()
    return pool.get_or_create(
        'vectorbt_rolling_optimizer',
        VectorBTRollingOptimizer,
        **kwargs
    )


def get_or_create_performance_monitor(**kwargs) -> Any:
    """
    Get or create SRPerformanceMonitor instance.
    
    Returns:
        SRPerformanceMonitor instance (cached)
    """
    try:
        from ...training.steps.market_analysis.sr_performance_monitor import SRPerformanceMonitor
        
        pool = get_component_pool()
        return pool.get_or_create(
            'sr_performance_monitor',
            SRPerformanceMonitor,
            **kwargs
        )
    except ImportError:
        logger.warning("SRPerformanceMonitor not available")
        return None


def get_or_create_unified_vectorization_manager(**kwargs) -> Any:
    """
    Get or create UnifiedVectorizationManager instance.
    
    Returns:
        UnifiedVectorizationManager instance (cached)
    """
    try:
        from ...training.steps.market_analysis.sr_detection.unified_vectorization_manager import UnifiedVectorizationManager
        
        pool = get_component_pool()
        return pool.get_or_create(
            'unified_vectorization_manager',
            UnifiedVectorizationManager,
            **kwargs
        )
    except ImportError:
        logger.warning("UnifiedVectorizationManager not available")
        return None


# Example usage:
# from src.utils.ml_common.component_pool import get_component_pool, get_or_create_vectorbt_optimizer
# 
# # Get or create a component
# optimizer = get_or_create_vectorbt_optimizer(enable_gpu=False)
# 
# # Manual pool management
# pool = get_component_pool()
# my_component = pool.get_or_create('my_key', MyComponentClass, param1=value1)
# 
# # Check pool stats
# stats = pool.get_stats()
# print(f"Pool contains {stats['total']} components")

