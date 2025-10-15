"""
Base Factory

This module provides the base factory class for creating various components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseFactory(ABC):
    """Base factory class for creating components."""
    
    def __init__(self):
        """Initialize the base factory."""
        self._components = {}
    
    @abstractmethod
    def create_component(self, component_type: str, **kwargs) -> Any:
        """Create a component of the specified type."""
        pass
    
    def register_component(self, name: str, component_class: type) -> None:
        """Register a component class."""
        self._components[name] = component_class
    
    def get_available_components(self) -> list:
        """Get list of available components."""
        return list(self._components.keys())