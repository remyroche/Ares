# src/core/dependency_injection.py

from collections.abc import Callable
from src.utils.logger import system_logger
from typing import Any, TypeVar

from dataclasses import dataclass
from src.interfaces import (
import IAnalyst,
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
    pass
    pass
    pass
    pass
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
        lifetime: str = ServiceLifetime.SINGLETON,
    ) -> None:
        """Register a service with enhanced configuration support."""
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

    def register_instance(self, service_name: Any, instance: Any) -> None:
    pass
    pass
    pass
    pass
        """Register an already-created service instance (always singleton)."""
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
    pass
    pass
    pass
    pass
        """Begin a scoped lifetime context."""
        self._current_scope = scope_id
        if scope_id not in self._scoped_instances:
    pass
    pass
    pass
    pass
            self._scoped_instances[scope_id] = {}
        self.logger.debug(f"Entered scope: {scope_id}")

    def end_scope(self, scope_id: str) -> None:
    pass
    pass
    pass
    pass
        """End a scoped lifetime context and cleanup scoped instances."""
        if self._current_scope == scope_id:
    pass
    pass
    pass
    pass
            self._current_scope = None
        if scope_id in self._scoped_instances:
    pass
    pass
    pass
    pass
            del self._scoped_instances[scope_id]
        self.logger.debug(f"Exited scope: {scope_id}")

    def get_config(self, key: str, default: Any = None) -> Any:
    pass
    pass
    pass
    pass
        """Get configuration value with fallback."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
    pass
    pass
    pass
    pass
        """Set configuration value."""
        self._config[key] = value
        self.logger.debug(f"Set config: {key} = {value}")

    def get_service_config(self, service_name: Any) -> dict[str, Any]:
    pass
    pass
    pass
    pass
        """Get service-specific configuration."""
        service = self._services.get(service_name)
        if service and service.config:
    pass
    pass
    pass
    pass
            return service.config
        return {}

    def resolve(self, service_name: Any) -> Any:
    pass
    pass
    pass
    pass
        """Resolve a service with enhanced error handling."""
        try:
            # Handle existing instances (singleton or scoped)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            if service_name in self._instances:
    pass
    pass
    pass
    pass
                return self._instances[service_name]

            # Scoped instances
            if self._current_scope and service_name in self._scoped_instances.get(
                self._current_scope, {},
            ):
                return self._scoped_instances[self._current_scope][service_name]

            # Get or create service registration
            service_reg = self._services.get(service_name)
            if not service_reg and service_name in self._factories:
    pass
    pass
    pass
    pass
                # Create a default registration for factory-only services
                self.register_factory(service_name, self._factories[service_name])
                service_reg = self._services.get(service_name)

            if not service_reg:
    pass
    pass
    pass
    pass
                msg = f"Service '{getattr(service_name, '__name__', service_name)}' not registered"
                raise ValueError(msg)

            # Instance already provided
            if service_reg.instance is not None:
    pass
    pass
    pass
    pass
                instance = service_reg.instance
            else:
                # Create instance
                instance = self._create_instance(service_reg)

            # Store instance based on lifetime
            if service_reg.lifetime == ServiceLifetime.SINGLETON:
    pass
    pass
    pass
    pass
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
    pass
    pass
    pass
    pass
        """Create service instance with dependency injection."""
        try:
            # Use factory function if available
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            if service_reg.factory:
    pass
    pass
    pass
    pass
                factory_func = service_reg.factory
                try:
                    # Try calling with container
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                    return factory_func(self)
                except TypeError:
                    try:
                        # Try calling with config
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                        return factory_func(self._config)
                    except TypeError:
                        # No-arg factory
                        return factory_func()

            # Get constructor parameters
            constructor_params = self._get_constructor_params(service_reg)

            # Create instance
            if constructor_params:
    pass
    pass
    pass
    pass
                instance = service_reg.implementation(**constructor_params)
            else:
                instance = service_reg.implementation()

            # Inject service-specific configuration if available
            if service_reg.config:
    pass
    pass
    pass
    pass
                self._inject_config(instance, service_reg.config)

            return instance

        except Exception as e:
            self.logger.exception(
                f"Failed to create instance for '{service_reg.service_type.__name__}': {e}",
            )
            raise

    def _get_constructor_params(self, service_reg: ServiceRegistration) -> dict[str, Any]:
    pass
    pass
    pass
    pass
        """Get constructor parameters for service creation."""
        params = {}

        # Add service-specific config if available
        if service_reg.config:
    pass
    pass
    pass
    pass
            params["config"] = service_reg.config

        # Resolve dependencies if specified
        if service_reg.dependencies:
    pass
    pass
    pass
    pass
            for param_name, dep_service_name in service_reg.dependencies.items():
    pass
    pass
    pass
    pass
                try:
                    params[param_name] = self.resolve(dep_service_name)
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
                except Exception as e:
                    self.logger.warning(
                        f"Failed to resolve dependency '{dep_service_name}' for '{param_name}': {e}",
                    )

        return params

    def _inject_config(self, instance: Any, config: dict[str, Any]) -> None:
    pass
    pass
    pass
    pass
        """Inject configuration into an instance."""
        if hasattr(instance, "configure"):
    pass
    pass
    pass
    pass
            instance.configure(config)
        elif hasattr(instance, "config"):
            instance.config.update(config)


class ComponentFactory:
    """Factory for creating trading system components."""

    def __init__(self, container: DependencyContainer):
    pass
    pass
    pass
    pass
        self.container = container
        self.logger = system_logger.getChild("ComponentFactory")

    def create_analyst(self, config: dict[str, Any] | None = None) -> IAnalyst:
    pass
    pass
    pass
    pass
        """Create an analyst component."""
        # Implementation would depend on specific analyst classes
        raise NotImplementedError("Analyst creation not implemented")

    def create_strategist(self, config: dict[str, Any] | None = None) -> IStrategist:
    pass
    pass
    pass
    pass
        """Create a strategist component."""
        # Implementation would depend on specific strategist classes
        raise NotImplementedError("Strategist creation not implemented")

    def create_tactician(self, config: dict[str, Any] | None = None) -> ITactician:
    pass
    pass
    pass
    pass
        """Create a tactician component."""
        # Implementation would depend on specific tactician classes
        raise NotImplementedError("Tactician creation not implemented")

    def create_supervisor(self, config: dict[str, Any] | None = None) -> ISupervisor:
    pass
    pass
    pass
    pass
        """Create a supervisor component."""
        # Implementation would depend on specific supervisor classes
        raise NotImplementedError("Supervisor creation not implemented")


class ModularTradingSystem:
    """Modular trading system using dependency injection."""

    def __init__(self, container: DependencyContainer):
    pass
    pass
    pass
    pass
        self.container = container
        self.factory = ComponentFactory(container)
        self.logger = system_logger.getChild("ModularTradingSystem")
        self.components: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the trading system."""
        self.logger.info("Initializing modular trading system")
        # Initialize components as needed
        pass

    async def shutdown(self) -> None:
        """Shutdown the trading system."""
        self.logger.info("Shutting down modular trading system")
        # Cleanup components
        pass
