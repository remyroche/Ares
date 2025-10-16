"""
Unified Factory

This module provides a unified factory for creating optimized components.
"""

from typing import Any, Dict, Optional, Type
from .base_factory import BaseFactory

class UnifiedFactory(BaseFactory):
    """Unified factory for creating optimized components."""

    def __init__(self):
        """Initialize the unified factory."""
        super().__init__()
        self._optimized_components = {}

    def create_component(self, component_type: str, **kwargs) -> Any:
        """Create a component of the specified type."""
        if component_type not in self._optimized_components:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = self._optimized_components[component_type]
        return component_class(**kwargs)

    def register_optimized_component(self, name: str, component_class: Type) -> None:
        """Register an optimized component class."""
        self._optimized_components[name] = component_class

    def get_available_components(self) -> list:
        """Get list of available components."""
        return list(self._optimized_components.keys())

def create_optimized_component(component_type: str, **kwargs) -> Any:
    """Create an optimized component of the specified type."""
    factory = UnifiedFactory()
    return factory.create_component(component_type, **kwargs)
