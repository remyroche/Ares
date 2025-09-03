"""
Dependency Injection Container for Supervisor Module.

This module provides a clean way to manage component dependencies
and their initialization, reducing coupling and improving testability.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.state_manager import StateManager


class DependencyContainer:
    """
    Dependency injection container for managing component dependencies.
    
    This container:
    - Centralizes dependency management
    - Provides lazy initialization of components
    - Handles dependency injection in a clean, testable way
    - Reduces coupling between components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the dependency container."""
        self.config = config or CONFIG
        self.logger = system_logger.getChild("DependencyContainer")
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        
    def register(self, name: str, factory: callable) -> None:
        """
        Register a component factory.
        
        Args:
            name: Component name
            factory: Callable that creates the component instance
        """
        self._factories[name] = factory
        self.logger.debug(f"Registered factory for {name}")
        
    def get(self, name: str) -> Any:
        """
        Get a component instance (lazy initialization).
        
        Args:
            name: Component name
            
        Returns:
            Component instance
            
        Raises:
            KeyError: If component is not registered
        """
        if name not in self._instances:
            if name not in self._factories:
                raise KeyError(f"Component '{name}' not registered")
            
            self._instances[name] = self._factories[name]()
            self.logger.debug(f"Created instance of {name}")
            
        return self._instances[name]
    
    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._factories
    
    def inject_dependencies(self, component: Any, dependencies: Dict[str, str]) -> None:
        """
        Inject dependencies into a component.
        
        Args:
            component: Component to inject dependencies into
            dependencies: Mapping of attribute names to dependency names
        """
        for attr_name, dep_name in dependencies.items():
            if hasattr(component, attr_name):
                dep_instance = self.get(dep_name)
                setattr(component, attr_name, dep_instance)
                self.logger.debug(f"Injected {dep_name} as {attr_name} into {component.__class__.__name__}")


class ComponentBuilder:
    """
    Builder for creating components with proper dependency injection.
    """
    
    def __init__(self, container: DependencyContainer):
        """Initialize the component builder."""
        self.container = container
        self.logger = system_logger.getChild("ComponentBuilder")
        
    def build_analyst(self, exchange_client: Any, state_manager: StateManager) -> Any:
        """
        Build Analyst component with dependencies.
        
        Args:
            exchange_client: Exchange client instance
            state_manager: State manager instance
            
        Returns:
            Configured Analyst instance
        """
        def factory():
            from src.utils.model_manager import ModelManager
            model_manager = ModelManager(self.container.config)
            analyst = model_manager.get_analyst()
            
            # Inject dependencies
            if hasattr(analyst, "exchange"):
                analyst.exchange = exchange_client
            if hasattr(analyst, "state_manager"):
                analyst.state_manager = state_manager
                
            return analyst
            
        return factory
    
    def build_strategist(self, exchange_client: Any, state_manager: StateManager) -> Any:
        """
        Build Strategist component with dependencies.
        
        Args:
            exchange_client: Exchange client instance
            state_manager: State manager instance
            
        Returns:
            Configured Strategist instance
        """
        def factory():
            from src.utils.model_manager import ModelManager
            model_manager = ModelManager(self.container.config)
            strategist = model_manager.get_strategist()
            
            # Inject dependencies
            if hasattr(strategist, "exchange"):
                strategist.exchange = exchange_client
            if hasattr(strategist, "state_manager"):
                strategist.state_manager = state_manager
                
            return strategist
            
        return factory
    
    def build_tactician(self, exchange_client: Any, state_manager: StateManager, performance_reporter: Any) -> Any:
        """
        Build Tactician component with dependencies.
        
        Args:
            exchange_client: Exchange client instance
            state_manager: State manager instance
            performance_reporter: Performance reporter instance
            
        Returns:
            Configured Tactician instance
        """
        def factory():
            from src.utils.model_manager import ModelManager
            model_manager = ModelManager(self.container.config)
            tactician = model_manager.get_tactician()
            
            # Inject dependencies
            if hasattr(tactician, "exchange"):
                tactician.exchange = exchange_client
            if hasattr(tactician, "state_manager"):
                tactician.state_manager = state_manager
            if hasattr(tactician, "performance_reporter"):
                tactician.performance_reporter = performance_reporter
                
            return tactician
            
        return factory
    
    def build_sentinel(self, exchange_client: Any, state_manager: StateManager) -> Any:
        """
        Build Sentinel component with dependencies.
        
        Args:
            exchange_client: Exchange client instance
            state_manager: State manager instance
            
        Returns:
            Configured Sentinel instance
        """
        def factory():
            from src.sentinel.sentinel import Sentinel
            return Sentinel(exchange_client, state_manager)
            
        return factory