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

    def __init__(self, config: dict[str, Any] | None = None):
        self._services: dict[Any, ServiceRegistration] = {}
        self._instances: dict[Any, Any] = {}
        self._scoped_instances: dict[str, dict[Any, Any]] = {}
        self._current_scope: str | None = None
        self._config: dict[str, Any] = config or {}
        self._factories: dict[Any, Callable] = {}
        self.logger = system_logger.getChild("DependencyContainer")

    def register_factory(
        self,
        service_name: Any,
        factory_func: Callable,
        lifetime: str = ServiceLifetime.SINGLETON,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a factory function for service creation."""
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

    def resolve(self, service_name: Any) -> Any:
        """Resolve a service with enhanced error handling."""
        try:
            # Handle existing instances (singleton or scoped)
            if service_name in self._instances:
                return self._instances[service_name]

            # Scoped instances
            if self._current_scope and service_name in self._scoped_instances.get(
                self._current_scope, {},
            ):
                return self._scoped_instances[self._current_scope][service_name]

            # Get or create service registration
            service_reg = self._services.get(service_name)
            if not service_reg and service_name in self._factories:
                # Create a default registration for factory-only services
                self.register_factory(service_name, self._factories[service_name])
                service_reg = self._services.get(service_name)

            if not service_reg:
                msg = f"Service '{getattr(service_name, '__name__', service_name)}' not registered"
                raise ValueError(msg)

            # Instance already provided
            if service_reg.instance is not None:
                instance = service_reg.instance
            else:
                # Create instance
                instance = self._create_instance(service_reg)

            # Store instance based on lifetime
            if service_reg.lifetime == ServiceLifetime.SINGLETON:
                self._instances[service_name] = instance
            elif service_reg.lifetime == ServiceLifetime.SCOPED and self._current_scope:
                self._scoped_instances[self._current_scope][service_name] = instance

            return instance

        except Exception as e:
            self.logger.exception(
                f"Failed to resolve service '{getattr(service_name, '__name__', service_name)}': {e}",
            )
            raise

    def _create_instance(self, service_reg: ServiceRegistration) -> Any:
        """Create service instance with dependency injection."""
        try:
            # Use factory function if available
            if service_reg.factory:
                factory_func = service_reg.factory
                try:
                    # Try calling with container
                    return factory_func(self)
                except TypeError:
                    try:
                        # Try calling with config
                        return factory_func(self._config)
                    except TypeError:
                        # No-arg factory
                        return factory_func()

            # Get constructor parameters
            constructor_params = self._get_constructor_params(service_reg)

            # Create instance
            if constructor_params:
                instance = service_reg.implementation(**constructor_params)
            else:
                instance = service_reg.implementation()

            # Inject service-specific configuration if available
            if service_reg.config:
                self._inject_config(instance, service_reg.config)

            return instance

        except Exception as e:
            self.logger.exception(
                f"Failed to create instance for '{service_reg.service_type.__name__}': {e}",
            )
            raise

    def _inject_config(self, instance: Any, config: dict[str, Any]) -> None:
        """Inject configuration into an instance."""
        if hasattr(instance, "configure"):
            instance.configure(config)
        elif hasattr(instance, "config"):
            instance.config.update(config)


class ComponentFactory:
    """Factory for creating trading system components."""

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ComponentFactory")


class ModularTradingSystem:
    """Modular trading system using dependency injection."""

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.factory = ComponentFactory(container)
        self.logger = system_logger.getChild("ModularTradingSystem")
        self.components: dict[str, Any] = {}
