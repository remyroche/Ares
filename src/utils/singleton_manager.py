"""
Singleton Manager for Component Initialization

This module provides a centralized singleton manager to prevent repeated initialization
of components and resolve the initialization loop issues.
"""

import threading
import logging
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class SingletonManager:
    """Centralized singleton manager to prevent repeated initialization."""
    
    _instance = None
    _lock = threading.Lock()
    _initialized_components: Dict[str, Any] = {}
    _initialization_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._component_locks: Dict[str, threading.Lock] = {}
    
    def get_component(self, component_name: str, factory_func: Callable, *args, **kwargs) -> Any:
        """
        Get a component instance, creating it if it doesn't exist.
        
        Args:
            component_name: Unique name for the component
            factory_func: Function to create the component if it doesn't exist
            *args: Arguments to pass to factory function
            **kwargs: Keyword arguments to pass to factory function
            
        Returns:
            Component instance
        """
        if component_name in self._initialized_components:
            logger.debug(f"🔄 Reusing existing component: {component_name}")
            return self._initialized_components[component_name]
        
        # Create a lock for this specific component
        if component_name not in self._component_locks:
            with self._initialization_lock:
                if component_name not in self._component_locks:
                    self._component_locks[component_name] = threading.Lock()
        
        with self._component_locks[component_name]:
            # Double-check after acquiring the lock
            if component_name in self._initialized_components:
                logger.debug(f"🔄 Component {component_name} was initialized by another thread")
                return self._initialized_components[component_name]
            
            try:
                logger.info(f"🚀 Initializing component: {component_name}")
                component = factory_func(*args, **kwargs)
                self._initialized_components[component_name] = component
                logger.info(f"✅ Component initialized successfully: {component_name}")
                return component
            except Exception as e:
                logger.error(f"❌ Failed to initialize component {component_name}: {e}")
                raise
    
    def is_initialized(self, component_name: str) -> bool:
        """Check if a component is already initialized."""
        return component_name in self._initialized_components
    
    def reset_component(self, component_name: str):
        """Reset a component (for testing purposes)."""
        with self._initialization_lock:
            if component_name in self._initialized_components:
                del self._initialized_components[component_name]
                logger.info(f"🔄 Reset component: {component_name}")
    
    def get_initialized_components(self) -> Dict[str, Any]:
        """Get all initialized components."""
        return self._initialized_components.copy()
    
    def clear_all(self):
        """Clear all initialized components (for testing purposes)."""
        with self._initialization_lock:
            self._initialized_components.clear()
            logger.info("🧹 Cleared all initialized components")


# Global singleton manager instance
singleton_manager = SingletonManager()


def singleton_component(component_name: str):
    """
    Decorator to make a component a singleton.
    
    Args:
        component_name: Unique name for the component
    """
    def decorator(cls):
        original_new = cls.__new__
        
        def new_singleton(cls, *args, **kwargs):
            return singleton_manager.get_component(
                component_name,
                lambda: original_new(cls),
                *args,
                **kwargs
            )
        
        cls.__new__ = new_singleton
        return cls
    
    return decorator


def get_singleton_component(component_name: str, factory_func: Callable, *args, **kwargs) -> Any:
    """
    Get a singleton component instance.
    
    Args:
        component_name: Unique name for the component
        factory_func: Function to create the component if it doesn't exist
        *args: Arguments to pass to factory function
        **kwargs: Keyword arguments to pass to factory function
        
    Returns:
        Component instance
    """
    return singleton_manager.get_component(component_name, factory_func, *args, **kwargs)


def is_component_initialized(component_name: str) -> bool:
    """Check if a component is already initialized."""
    return singleton_manager.is_initialized(component_name)


def reset_singleton_components():
    """Reset all singleton components (for testing purposes)."""
    singleton_manager.clear_all()
