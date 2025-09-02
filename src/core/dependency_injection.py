# src/core/dependency_injection.py

from collections.abc import Callable
from src.utils.logger import system_logger
from typing import Any, TypeVar

from dataclasses import dataclass
from src.interfaces import (
    IAnalyst,
    IStrategist,
    ISupervisor,
    ITactician,
)

T = TypeVar("T")


class ServiceLifetime:
    """Service lifetime constants compatible with enhanced DI usage."""
    
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceRegistration:
    """Enhanced service registration with configuration support."""
    
    service_type: type
    implementation: type | None = None
    singleton: bool = True
    config: dict[str, Any] | None = None
    # Backward-incompatible attributes replaced/extended for compatibility
    factory_method: str | None = None  # kept for backward compatibility (unused)
    dependencies: dict[str, str] | None = None
    # New attributes to align with enhanced DI usage
    lifetime: str = ServiceLifetime.SINGLETON
    factory: Callable | None = None
    instance: Any | None = None


class DependencyContainer:
    """
    Enhanced dependency injection container with configuration management.
    """
    
    def __init__(self, config=None):
        self._services: dict[Any, ServiceRegistration] = {}
        self._instances: dict[Any, Any] = {}
        self._scoped_instances: dict[str, dict[Any, Any]] = {}
        self._current_scope: str | None = None
        self._config: dict[str, Any] = config or {}
        self._factories: dict[Any, Callable] = {}
        self.logger = system_logger.getChild("DependencyContainer")

    def register(self, service_name, service_type, implementation=None, 
                singleton=True, config=None, dependencies=None, lifetime=None):
        """Register a service in the container."""
        # Map legacy singleton flag to lifetime if not explicitly provided
        if lifetime not in {
            ServiceLifetime.SINGLETON,
            ServiceLifetime.TRANSIENT,
            ServiceLifetime.SCOPED,
        }:
            lifetime = (
                ServiceLifetime.SINGLETON if singleton else ServiceLifetime.TRANSIENT
            )

        self._services[service_name] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation or service_type,
            singleton=singleton,
            config=config,
            dependencies=dependencies,
            lifetime=lifetime,
        )
        self.logger.debug(
            f"Registered service: {getattr(service_name, '__name__', str(service_name))} -> {service_type.__name__}",
        )

    def register_factory(self, service_name, factory_func, lifetime=ServiceLifetime.TRANSIENT, config=None):
        """Register a factory function for a service."""
        self._factories[service_name] = factory_func
        # Also create a registration placeholder so resolve() can work
        self._services[service_name] = ServiceRegistration(
            service_type=service_name
            if isinstance(service_name, type)
            else type(factory_func),
            implementation=None,
            singleton=(lifetime == ServiceLifetime.SINGLETON),
            config=config,
            dependencies=None,
            lifetime=lifetime,
            factory=factory_func,
        )
        self.logger.debug(
            f"Registered factory for: {getattr(service_name, '__name__', str(service_name))}",
        )

    def register_instance(self, service_name, instance, config=None):
        """Register an existing instance."""
        self._services[service_name] = ServiceRegistration(
            service_type=type(instance),
            implementation=type(instance),
            singleton=True,
            config=config,
            dependencies=None,
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance,
        )
        self._instances[service_name] = instance
        self.logger.debug(
            f"Registered instance: {getattr(service_name, '__name__', str(service_name))}",
        )

    def resolve(self, service_name):
        """Resolve a service from the container."""
        if service_name not in self._services:
            raise KeyError(f"Service not registered: {service_name}")
        
        registration = self._services[service_name]
        
        # Return existing instance if singleton
        if registration.singleton and service_name in self._instances:
            return self._instances[service_name]
        
        # Use factory if available
        if registration.factory:
            instance = registration.factory()
        else:
            # Create new instance
            instance = registration.implementation()
        
        # Store instance if singleton
        if registration.singleton:
            self._instances[service_name] = instance
        
        return instance

    def get_all_services(self):
        """Get all registered services."""
        return self._services

    def create_scope(self, scope_name: str):
        """Create a new scope for scoped services."""
        self._current_scope = scope_name
        if scope_name not in self._scoped_instances:
            self._scoped_instances[scope_name] = {}
        return self

    def end_scope(self):
        """End the current scope and clean up scoped instances."""
        if self._current_scope:
            del self._scoped_instances[self._current_scope]
            self._current_scope = None


class ComponentFactory:
    """Factory for creating components with dependency injection."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ComponentFactory")

    def create_component(self, component_type, **kwargs):
        """Create a component with the given type."""
        try:
            # Try to resolve from container first
            if component_type in self.container._services:
                return self.container.resolve(component_type)
            
            # Create new instance with kwargs
            return component_type(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create component {component_type}: {e}")
            raise


class ModularTradingSystem:
    """Modular trading system that uses dependency injection."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.components = {}
        self.logger = system_logger.getChild("ModularTradingSystem")

    def add_component(self, name: str, component_type):
        """Add a component to the system."""
        try:
            component = self.container.resolve(component_type)
            self.components[name] = component
            self.logger.info(f"Added component: {name}")
        except Exception as e:
            self.logger.error(f"Failed to add component {name}: {e}")
            raise

    def get_component(self, name: str):
        """Get a component by name."""
        return self.components.get(name)

    def initialize_all(self):
        """Initialize all components."""
        for name, component in self.components.items():
            if hasattr(component, 'initialize'):
                try:
                    component.initialize()
                    self.logger.info(f"Initialized component: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize component {name}: {e}")

    def shutdown_all(self):
        """Shutdown all components."""
        for name, component in self.components.items():
            if hasattr(component, 'shutdown'):
                try:
                    component.shutdown()
                    self.logger.info(f"Shutdown component: {name}")
                except Exception as e:
                    self.logger.error(f"Failed to shutdown component {name}: {e}")
