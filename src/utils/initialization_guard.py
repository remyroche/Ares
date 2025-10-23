"""
Initialization Guard

This module provides initialization guards to prevent repeated initialization
of components during module imports.
"""

import threading
import logging
from typing import Set, Dict, Any

logger = logging.getLogger(__name__)


class InitializationGuard:
    """Guard to prevent repeated initialization of components."""
    
    _instance = None
    _lock = threading.Lock()
    _initialized_components: Set[str] = set()
    _component_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def is_initialized(self, component_name: str) -> bool:
        """Check if a component is already initialized."""
        with self._component_lock:
            return component_name in self._initialized_components
    
    def mark_initialized(self, component_name: str):
        """Mark a component as initialized."""
        with self._component_lock:
            if component_name in self._initialized_components:
                logger.debug(f"🔄 Component {component_name} already initialized, skipping")
                return False
            self._initialized_components.add(component_name)
            logger.debug(f"✅ Marked component {component_name} as initialized")
            return True
    
    def reset(self, component_name: str = None):
        """Reset initialization status for a component or all components."""
        with self._component_lock:
            if component_name:
                self._initialized_components.discard(component_name)
                logger.debug(f"🔄 Reset initialization status for {component_name}")
            else:
                self._initialized_components.clear()
                logger.debug("🧹 Reset all initialization statuses")
    
    def get_initialized_components(self) -> Set[str]:
        """Get all initialized component names."""
        with self._component_lock:
            return self._initialized_components.copy()


# Global initialization guard instance
init_guard = InitializationGuard()


def initialization_guard(component_name: str):
    """
    Decorator to guard against repeated initialization.
    
    Args:
        component_name: Unique name for the component
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if init_guard.is_initialized(component_name):
                logger.debug(f"🔄 Skipping initialization of {component_name} - already initialized")
                return None
            
            if init_guard.mark_initialized(component_name):
                logger.info(f"🚀 Initializing {component_name}")
                return func(*args, **kwargs)
            else:
                logger.debug(f"🔄 {component_name} initialization skipped - already initialized")
                return None
        
        return wrapper
    return decorator


def check_initialization_status(component_name: str) -> bool:
    """Check if a component is initialized."""
    return init_guard.is_initialized(component_name)


def reset_initialization_status(component_name: str = None):
    """Reset initialization status."""
    init_guard.reset(component_name)


def get_all_initialized_components() -> Set[str]:
    """Get all initialized component names."""
    return init_guard.get_initialized_components()
