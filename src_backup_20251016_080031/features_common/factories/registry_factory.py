"""
Registry Factory

This module provides factory functions for creating feature registries.
"""

from typing import Dict, Any, Optional, Type
from .base_factory import BaseFactory


class RegistryFactory(BaseFactory):
    """Factory for creating feature registries."""
    
    def __init__(self):
        """Initialize the registry factory."""
        super().__init__()
        self._registry_types = {}
    
    def register_registry_type(self, name: str, registry_class: Type) -> None:
        """Register a registry type."""
        self._registry_types[name] = registry_class
    
    def create_registry(self, registry_type: str, **kwargs) -> Any:
        """Create a registry of the specified type."""
        if registry_type not in self._registry_types:
            raise ValueError(f"Unknown registry type: {registry_type}")
        
        registry_class = self._registry_types[registry_type]
        return registry_class(**kwargs)
    
    def get_available_registry_types(self) -> list:
        """Get list of available registry types."""
        return list(self._registry_types.keys())


def create_registry(registry_type: str, **kwargs) -> Any:
    """Create a registry of the specified type."""
    factory = RegistryFactory()
    return factory.create_registry(registry_type, **kwargs)


def create_feature_registry(**kwargs) -> Any:
    """Create a feature registry."""
    # For now, return a simple dict-based registry
    return {"features": {}, "metadata": kwargs}