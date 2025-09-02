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

    def register(
        self,
        service_name: Any,
        service_type: type,
        implementation: type | None = None,
        singleton: bool = True,
        config: dict[str, Any] | None = None,
        dependencies: dict[str, str] | None = None,
        lifetime: str | None = None,
    ) -> None:
        """Register a service with the container."""
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

    def register_factory(
        self,
        service_name: Any,
        factory_func: Callable,
        lifetime: str = ServiceLifetime.SINGLETON,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a factory function for creating service instances."""
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

    def register_instance(
        self,
        service_name: Any,
        instance: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register an existing instance with the container."""
        self._services[service_name] = ServiceRegistration(
            service_type=type(instance),
            implementation=type(instance),
            singleton=True,
            config=None,
            dependencies=None,
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance,
        )
        self._instances[service_name] = instance
        self.logger.debug(
            f"Registered instance for: {getattr(service_name, '__name__', str(service_name))}",
        )

    def begin_scope(self, scope_id: str) -> None:
        """Begin a new scope for scoped services."""
        self._current_scope = scope_id
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
        self.logger.debug(f"Entered scope: {scope_id}")

    def end_scope(self, scope_id: str) -> None:
        """End a scope and clean up scoped instances."""
        if self._current_scope == scope_id:
            self._current_scope = None
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
        self.logger.debug(f"Exited scope: {scope_id}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        self._config[key] = value
        self.logger.debug(f"Set config: {key} = {value}")

    def get_service_config(self, service_name: Any) -> dict[str, Any]:
        """Get configuration for a specific service."""
        service = self._services.get(service_name)
        if service and service.config:
            return service.config
        return {}

    def resolve(self, service_name: Any) -> Any:
        """Resolve a service instance from the container."""
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
        """Create a new instance of a service."""
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

    def _get_constructor_params(self, service_reg: ServiceRegistration) -> dict[str, Any]:
        """Get constructor parameters for a service."""
        params = {}

        # Add service-specific config if available
        if service_reg.config:
            params["config"] = service_reg.config

        # Resolve dependencies if specified
        if service_reg.dependencies:
            for param_name, dep_service_name in service_reg.dependencies.items():
                try:
                    params[param_name] = self.resolve(dep_service_name)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to resolve dependency '{dep_service_name}' for '{param_name}': {e}",
                    )

        return params

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

    def create_analyst(self) -> IAnalyst:
        """Create an analyst component."""
        # Implementation would depend on specific analyst classes
        self.logger.warning("Analyst creation requested but not implemented")
        raise NotImplementedError("Analyst creation not implemented - register an analyst service first")

    def create_strategist(self) -> IStrategist:
        """Create a strategist component."""
        # Implementation would depend on specific strategist classes
        self.logger.warning("Strategist creation requested but not implemented")
        raise NotImplementedError("Strategist creation not implemented - register a strategist service first")

    def create_tactician(self) -> ITactician:
        """Create a tactician component."""
        # Implementation would depend on specific tactician classes
        self.logger.warning("Tactician creation requested but not implemented")
        raise NotImplementedError("Tactician creation not implemented - register a tactician service first")

    def create_supervisor(self) -> ISupervisor:
        """Create a supervisor component."""
        # Implementation would depend on specific supervisor classes
        self.logger.warning("Supervisor creation requested but not implemented")
        raise NotImplementedError("Supervisor creation not implemented - register a supervisor service first")


class ModularTradingSystem:
    """Modular trading system using dependency injection."""

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.factory = ComponentFactory(container)
        self.logger = system_logger.getChild("ModularTradingSystem")
        self.components: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the trading system."""
        self.logger.info("Initializing modular trading system")
        # Initialize components as needed
        try:
            # Create basic components if they're registered
            if hasattr(self.container, '_services') and self.container._services:
                self.logger.info(f"Found {len(self.container._services)} registered services")
            else:
                self.logger.warning("No services registered in container")
        except Exception as e:
            self.logger.error(f"Error during initialization: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the trading system."""
        self.logger.info("Shutting down modular trading system")
        # Cleanup components
        try:
            # Clear component references
            self.components.clear()
            self.logger.info("Cleared component references")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
