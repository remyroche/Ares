from dataclasses import dataclass
from typing import TypeVar, Any, Callable
from datetime import datetime
from src.interfaces import IAnalyst, IStrategist, ISupervisor, ITactician
from src.utils.logger import system_logger
import logging
import time

T = TypeVar('T')

class ServiceLifetime:
    """Service lifetime constants compatible with enhanced DI usage."""
    SINGLETON = 'singleton'
    TRANSIENT = 'transient'
    SCOPED = 'scoped'

@dataclass
class ServiceRegistration:
    """Enhanced service registration with configuration support."""
    service_type: type
    implementation: type | None = None
    singleton: bool = True
    config: dict[str, Any] | None = None
    factory_method: str | None = None
    dependencies: dict[str, str] | None = None
    lifetime: str = ServiceLifetime.SINGLETON
    factory: Callable | None = None
    instance: Any | None = None

class DependencyContainer:
    """
    Enhanced dependency injection container with configuration management.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._services: dict[Any, ServiceRegistration] = {}
        self._instances: dict[Any, Any] = {}
        self._scoped_instances: dict[str, dict[Any, Any]] = {}
        self._current_scope: str | None = None
        self._config: dict[str, Any] = config or {}
        self._factories: dict[Any, Callable] = {}
        self._resolution_stack: list[Any] = []  # For circular dependency detection
        self._service_health: dict[Any, dict[str, Any]] = {}  # Service health monitoring
        self.logger = system_logger.getChild('DependencyContainer')

    def register(self, service_name: Any, service_type: type, implementation: type | None = None, singleton: bool = True, config: dict[str, Any] | None = None, dependencies: dict[str, str] | None = None, lifetime: str = ServiceLifetime.SINGLETON) -> None:
        """Register a service with enhanced configuration support."""
        if lifetime not in {ServiceLifetime.SINGLETON, ServiceLifetime.TRANSIENT, ServiceLifetime.SCOPED}:
            lifetime = ServiceLifetime.SINGLETON if singleton else ServiceLifetime.TRANSIENT
        self._services[service_name] = ServiceRegistration(service_type = service_type, implementation = implementation or service_type, singleton = singleton, config = config, dependencies = dependencies, lifetime = lifetime)
        self.logger.debug(f"Registered service: {getattr(service_name, '__name__', str(service_name))} -> {service_type.__name__}")

    def register_factory(self, service_name: Any, factory_func: Callable, lifetime: str = ServiceLifetime.SINGLETON, config: dict[str, Any] | None = None) -> None:
        """Register a factory function for service creation."""
        self._factories[service_name] = factory_func
        self._services[service_name] = ServiceRegistration(service_type = service_name if isinstance(service_name, type) else type(factory_func), implementation = None, singleton = lifetime == ServiceLifetime.SINGLETON, config = config, dependencies = None, lifetime = lifetime, factory = factory_func)
        self.logger.debug(f"Registered factory for: {getattr(service_name, '__name__', str(service_name))}")

    def register_instance(self, service_name: Any, instance: Any) -> None:
        """Register an already-created service instance (always singleton)."""
        self._services[service_name] = ServiceRegistration(service_type = type(instance), implementation = type(instance), singleton = True, config = None, dependencies = None, lifetime = ServiceLifetime.SINGLETON, instance = instance)
        self._instances[service_name] = instance
        self.logger.debug(f"Registered instance for: {getattr(service_name, '__name__', str(service_name))}")

    def begin_scope(self, scope_id: str) -> None:
        """Begin a scoped lifetime context."""
        self._current_scope = scope_id
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
        self.logger.debug(f'Entered scope: {scope_id}')

    def end_scope(self, scope_id: str) -> None:
        """End a scoped lifetime context and cleanup scoped instances."""
        if self._current_scope == scope_id:
            self._current_scope = None
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
        self.logger.debug(f'Exited scope: {scope_id}')

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
        self.logger.debug(f'Set config: {key} = {value}')

    def get_service_config(self, service_name: Any) -> dict[str, Any]:
        """Get service-specific configuration."""
        service = self._services.get(service_name)
        if service and service.config:
            return service.config
        return {}

    def resolve(self, service_name: Any) -> Any:
        """Resolve a service with enhanced error handling and circular dependency detection."""
        try:
            # Check for circular dependency
            if service_name in self._resolution_stack:
                circular_path = " -> ".join([str(s) for s in self._resolution_stack] + [str(service_name)])
                raise ValueError(f"Circular dependency detected: {circular_path}")
            
            # Add to resolution stack
            self._resolution_stack.append(service_name)
            
            try:
                if service_name in self._instances:
                    return self._instances[service_name]
                if self._current_scope and service_name in self._scoped_instances.get(self._current_scope, {}):
                    return self._scoped_instances[self._current_scope][service_name]
                service_reg = self._services.get(service_name)
                if not service_reg and service_name in self._factories:
                    self.register_factory(service_name, self._factories[service_name])
                    service_reg = self._services.get(service_name)
                if not service_reg:
                    msg = f"Service '{getattr(service_name, '__name__', service_name)}' not registered"
                    raise ValueError(msg)
                if service_reg.instance is not None:
                    instance = service_reg.instance
                else:
                    instance = self._create_instance(service_reg)
                if service_reg.lifetime == ServiceLifetime.SINGLETON:
                    self._instances[service_name] = instance
                elif service_reg.lifetime == ServiceLifetime.SCOPED and self._current_scope:
                    self._scoped_instances[self._current_scope][service_name] = instance
                return instance
            finally:
                # Remove from resolution stack
                self._resolution_stack.pop()
                
        except Exception as e:
            self.logger.exception(f"Failed to resolve service '{getattr(service_name, '__name__', service_name)}': {e}")
            # Clean up resolution stack on error
            if service_name in self._resolution_stack:
                self._resolution_stack.remove(service_name)
            raise

    def _create_instance(self, service_reg: ServiceRegistration) -> Any:
        """Create service instance with dependency injection."""
        try:
            if service_reg.factory:
                factory_func = service_reg.factory
                try:
                    return factory_func(self)
                except TypeError:
                    try:
                        return factory_func(self._config)
                    except TypeError:
                        return factory_func()
            constructor_params = self._get_constructor_params(service_reg)
            if constructor_params:
                instance = service_reg.implementation(**constructor_params)
            else:
                instance = service_reg.implementation()
            if service_reg.config:
                self._inject_config(instance, service_reg.config)
            return instance
        except Exception as e:
            self.logger.exception(f"Failed to create instance for '{service_reg.service_type.__name__}': {e}")
            raise

    def _get_constructor_params(self, service_reg: ServiceRegistration) -> dict[str, Any]:
        """Get constructor parameters for service creation."""
        params = {}
        if service_reg.config:
            params['config'] = service_reg.config
        if service_reg.dependencies:
            for param_name, dep_service_name in service_reg.dependencies.items():
                try:
                    params[param_name] = self.resolve(dep_service_name)
                except Exception as e:
                    self.logger.warning(f"Failed to resolve dependency '{dep_service_name}' for '{param_name}': {e}")
        return params

    def _inject_config(self, instance: Any, config: dict[str, Any]) -> None:
        """Inject configuration into an instance."""
        if hasattr(instance, 'configure'):
            instance.configure(config)
        elif hasattr(instance, 'config'):
            instance.config.update(config)

    def get_service_health(self, service_name: Any) -> dict[str, Any]:
        """Get health status of a service."""
        health = self._service_health.get(service_name, {})
        service_reg = self._services.get(service_name)
        
        if service_reg:
            health.update({
                'registered': True,
                'lifetime': service_reg.lifetime,
                'has_instance': service_name in self._instances,
                'has_factory': service_reg.factory is not None,
                'has_config': service_reg.config is not None
            })
        else:
            health.update({
                'registered': False,
                'error': 'Service not registered'
            })
        
        return health

    def get_all_service_health(self) -> dict[str, dict[str, Any]]:
        """Get health status of all registered services."""
        health_status = {}
        for service_name in self._services.keys():
            health_status[str(service_name)] = self.get_service_health(service_name)
        return health_status

    def update_service_health(self, service_name: Any, status: str, details: dict[str, Any] | None = None) -> None:
        """Update health status of a service."""
        if service_name not in self._service_health:
            self._service_health[service_name] = {}
        
        self._service_health[service_name].update({
            'status': status,
            'last_updated': datetime.now().isoformat(),
            'details': details or {}
        })
        
        self.logger.debug(f"Updated health for {service_name}: {status}")

    def check_service_health(self, service_name: Any) -> bool:
        """Check if a service is healthy."""
        try:
            # Try to resolve the service
            instance = self.resolve(service_name)
            if instance is not None:
                self.update_service_health(service_name, 'healthy')
                return True
            else:
                self.update_service_health(service_name, 'unhealthy', {'error': 'Resolved to None'})
                return False
        except Exception as e:
            self.update_service_health(service_name, 'unhealthy', {'error': str(e)})
            return False

    def get_unhealthy_services(self) -> list[str]:
        """Get list of unhealthy services."""
        unhealthy = []
        for service_name in self._services.keys():
            if not self.check_service_health(service_name):
                unhealthy.append(str(service_name))
        return unhealthy

class ComponentFactory:
    """Factory for creating trading system components."""

    def __init__(self, container: DependencyContainer) -> None:
        self.container = container
        self.logger = system_logger.getChild('ComponentFactory')

    def create_analyst(self, config: dict[str, Any] | None = None) -> IAnalyst:
        """Create an analyst component."""
        try:
            from src.analyst.analyst import Analyst
            analyst_config = config or self.container.get_config('analyst', {})
            return Analyst(analyst_config)
        except ImportError as e:
            self.logger.error(f"Failed to import Analyst: {e}")
            raise NotImplementedError(f"Analyst creation failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to create Analyst: {e}")
            raise

    def create_strategist(self, config: dict[str, Any] | None = None) -> IStrategist:
        """Create a strategist component."""
        try:
            from src.strategist.strategist import Strategist
            strategist_config = config or self.container.get_config('strategist', {})
            return Strategist(strategist_config)
        except ImportError as e:
            self.logger.error(f"Failed to import Strategist: {e}")
            raise NotImplementedError(f"Strategist creation failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to create Strategist: {e}")
            raise

    def create_tactician(self, config: dict[str, Any] | None = None) -> ITactician:
        """Create a tactician component."""
        msg = 'Tactician creation not implemented'
        raise NotImplementedError(msg)

    def create_supervisor(self, config: dict[str, Any] | None = None) -> ISupervisor:
        """Create a supervisor component."""
        msg = 'Supervisor creation not implemented'
        raise NotImplementedError(msg)

class ModularTradingSystem:
    """Modular trading system using dependency injection."""

    def __init__(self, container: DependencyContainer) -> None:
        self.container = container
        self.factory = ComponentFactory(container)
        self.logger = system_logger.getChild('ModularTradingSystem')
        self.components: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the trading system."""
        self.logger.info('Initializing modular trading system')

    async def shutdown(self) -> None:
        """Shutdown the trading system."""
        self.logger.info('Shutting down modular trading system')