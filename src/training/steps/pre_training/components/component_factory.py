"""
Component Factory for Pre-Training Pipeline

This module provides the ComponentFactory for creating pre-training pipeline components.
"""

from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

# Import ModularComponent for enhanced component creation
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
        ModularComponent
    )
    MODULAR_COMPONENT_AVAILABLE = True
except ImportError:
    MODULAR_COMPONENT_AVAILABLE = False

@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = None
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.custom_params is None:
            self.custom_params = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for ModularComponent."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'parameters': self.parameters,
            'custom_params': self.custom_params
        }

class BaseComponent(ABC):
    """Base class for all components."""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the component."""
        pass

class ComponentFactory:
    """Factory for creating pre-training pipeline components."""
    
    _components: Dict[str, Type[BaseComponent]] = {}
    
    @classmethod
    def register_component(cls, name: str, component_class: Type[BaseComponent]):
        """Register a component class."""
        cls._components[name] = component_class
    
    @classmethod
    def create_component(cls, name: str, config: ComponentConfig) -> Optional[BaseComponent]:
        """Create a component instance."""
        if name not in cls._components:
            return None
        
        component_class = cls._components[name]
        return component_class(config)
    
    @classmethod
    def create_modular_component(cls, name: str, config: ComponentConfig, 
                                logger: Optional[logging.Logger] = None):
        """Create a ModularComponent instance with enhanced features."""
        if not MODULAR_COMPONENT_AVAILABLE:
            # Fallback to regular component creation
            return cls.create_component(name, config)
        
        if name not in cls._components:
            return None
        
        component_class = cls._components[name]
        
        # Check if component is already a ModularComponent
        if MODULAR_COMPONENT_AVAILABLE and issubclass(component_class, ModularComponent):
            return component_class(
                name=config.name,
                config=config.to_dict() if hasattr(config, 'to_dict') else config.__dict__,
                logger=logger or logging.getLogger(__name__)
            )
        else:
            # For non-ModularComponent classes, create regular instance
            return component_class(config)
    
    @classmethod
    def get_available_components(cls) -> list:
        """Get list of available component names."""
        return list(cls._components.keys())
