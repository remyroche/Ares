# src/core/enhanced_dependency_injection.py

"""
Enhanced dependency injection container with improved type safety and protocol support.
"""

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

from src.types import ConfigDict, Configurable, Monitorable, Startable
from src.utils.logger import system_logger

T = TypeVar("T")
ServiceT = TypeVar("ServiceT")
ConfigT = TypeVar("ConfigT", bound=ConfigDict)

logger = system_logger.getChild("DependencyInjection")


@runtime_checkable
class Injectable(Protocol):
    """Protocol for injectable services."""
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the service."""
        ...


@runtime_checkable
class ServiceProvider(Protocol[ServiceT]):
    """Protocol for service providers."""
    
    def provide(self) -> ServiceT:
        """Provide the service instance."""
        ...
    
    def can_provide(self, service_type: Type[ServiceT]) -> bool:
        """Check if provider can provide the service type."""
        ...


class ServiceLifetime:
    """Service lifetime constants."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceRegistration(Generic[ServiceT]):
    """Type-safe service registration."""
    
    def __init__(
        self,
        service_type: Type[ServiceT],
        implementation: Optional[Type[ServiceT]] = None,
        factory: Optional[Callable[..., ServiceT]] = None,
        instance: Optional[ServiceT] = None,
        lifetime: str = ServiceLifetime.SINGLETON,
        config: Optional[ConfigDict] = None,
        dependencies: Optional[List[Type]] = None,
    ):
        self.service_type = service_type
        self.implementation = implementation or service_type
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        self.config = config or {}
        self.dependencies = dependencies or []
        
        # Validate registration
        self._validate_registration()
    
    def _validate_registration(self) -> None:
        """Validate the service registration."""
        if not (self.implementation or self.factory or self.instance):
            raise ValueError(f"Service {self.service_type} must have implementation, factory, or instance")
        
        if self.instance and self.lifetime != ServiceLifetime.SINGLETON:
            raise ValueError(f"Instance registration for {self.service_type} must use singleton lifetime")


class EnhancedDependencyContainer:
    """
    Enhanced dependency injection container with improved type safety.
    """
    
    def __init__(self, global_config: Optional[ConfigDict] = None):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._global_config = global_config or {}
        self._current_scope: Optional[str] = None
        self._service_providers: List[ServiceProvider] = []
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        lifetime: str = ServiceLifetime.SINGLETON,
        config: Optional[ConfigDict] = None,
    ) -> None:
        """Register a service with type safety."""
        if not self._is_valid_service_type(service_type):
            raise ValueError(f"Invalid service type: {service_type}")
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime=lifetime,
            config=config,
        )
        
        self._services[service_type] = registration
        logger.debug(f"Registered service: {service_type.__name__}")
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        lifetime: str = ServiceLifetime.SINGLETON,
        config: Optional[ConfigDict] = None,
    ) -> None:
        """Register a service with a factory function."""
        registration = ServiceRegistration(
            service_type=service_type,
            factory=factory,
            lifetime=lifetime,
            config=config,
        )
        
        self._services[service_type] = registration
        logger.debug(f"Registered factory for service: {service_type.__name__}")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register a service instance."""
        registration = ServiceRegistration(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
        )
        
        self._services[service_type] = registration
        self._instances[service_type] = instance
        logger.debug(f"Registered instance for service: {service_type.__name__}")
    
    def register_provider(self, provider: ServiceProvider) -> None:
        """Register a service provider."""
        self._service_providers.append(provider)
        logger.debug(f"Registered service provider: {type(provider).__name__}")
    
    async def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service with type safety."""
        # Check for existing instance based on lifetime
        if service_type in self._services:
            registration = self._services[service_type]
            
            # Handle singleton lifetime
            if registration.lifetime == ServiceLifetime.SINGLETON:
                if service_type in self._instances:
                    return cast(T, self._instances[service_type])
            
            # Handle scoped lifetime
            elif registration.lifetime == ServiceLifetime.SCOPED and self._current_scope:
                scope_instances = self._scoped_instances.get(self._current_scope, {})
                if service_type in scope_instances:
                    return cast(T, scope_instances[service_type])
            
            # Create new instance
            instance = await self._create_instance(registration)
            
            # Store instance based on lifetime
            if registration.lifetime == ServiceLifetime.SINGLETON:
                self._instances[service_type] = instance
            elif registration.lifetime == ServiceLifetime.SCOPED and self._current_scope:
                if self._current_scope not in self._scoped_instances:
                    self._scoped_instances[self._current_scope] = {}
                self._scoped_instances[self._current_scope][service_type] = instance
            
            return instance
        
        # Try service providers
        for provider in self._service_providers:
            if provider.can_provide(service_type):
                return provider.provide()
        
        raise ValueError(f"Service {service_type} not registered and no provider found")
    
    async def _create_instance(self, registration: ServiceRegistration[T]) -> T:
        """Create a service instance."""
        if registration.instance:
            return registration.instance
        
        if registration.factory:
            # Resolve factory dependencies
            factory_params = await self._resolve_factory_dependencies(registration.factory)
            return registration.factory(**factory_params)
        
        if registration.implementation:
            # Resolve constructor dependencies
            constructor_params = await self._resolve_constructor_dependencies(registration.implementation)
            instance = registration.implementation(**constructor_params)
            
            # Configure if configurable
            if isinstance(instance, Configurable):
                merged_config = {**self._global_config, **registration.config}
                instance.configure(merged_config)
            
            return instance
        
        raise ValueError(f"Cannot create instance for {registration.service_type}")
    
    async def _resolve_constructor_dependencies(self, implementation: Type[T]) -> Dict[str, Any]:
        """Resolve constructor dependencies."""
        params = {}
        
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Skip parameters with default values unless they're services
            if param.default != inspect.Parameter.empty:
                if not self._is_service_parameter(param):
                    continue
            
            # Resolve parameter
            if param.annotation and param.annotation != inspect.Parameter.empty:
                try:
                    resolved_param = await self.resolve(param.annotation)
                    params[param_name] = resolved_param
                except ValueError:
                    # If resolution fails and parameter has default, skip it
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise
        
        return params
    
    async def _resolve_factory_dependencies(self, factory: Callable) -> Dict[str, Any]:
        """Resolve factory function dependencies."""
        params = {}
        
        # Get factory signature
        sig = inspect.signature(factory)
        
        for param_name, param in sig.parameters.items():
            if param.annotation and param.annotation != inspect.Parameter.empty:
                try:
                    resolved_param = await self.resolve(param.annotation)
                    params[param_name] = resolved_param
                except ValueError:
                    if param.default != inspect.Parameter.empty:
                        continue
                    raise
        
        return params
    
    def _is_valid_service_type(self, service_type: Type) -> bool:
        """Check if the service type is valid."""
        # Must be a class or protocol
        return inspect.isclass(service_type) or hasattr(service_type, "__protocol__")
    
    def _is_service_parameter(self, param: inspect.Parameter) -> bool:
        """Check if a parameter should be resolved as a service."""
        if param.annotation == inspect.Parameter.empty:
            return False
        
        # Check if it's a registered service type
        return param.annotation in self._services
    
    def create_scope(self, scope_id: str) -> "ScopeContext":
        """Create a new dependency injection scope."""
        return ScopeContext(self, scope_id)
    
    def _enter_scope(self, scope_id: str) -> None:
        """Enter a dependency scope."""
        self._current_scope = scope_id
    
    def _exit_scope(self, scope_id: str) -> None:
        """Exit a dependency scope."""
        if self._current_scope == scope_id:
            self._current_scope = None
        
        # Clean up scoped instances
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
    
    def get_registered_services(self) -> List[Type]:
        """Get list of registered service types."""
        return list(self._services.keys())
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services


class ScopeContext:
    """Context manager for dependency injection scopes."""
    
    def __init__(self, container: EnhancedDependencyContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id
    
    def __enter__(self) -> "ScopeContext":
        self.container._enter_scope(self.scope_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.container._exit_scope(self.scope_id)


# Global container instance
_global_container: Optional[EnhancedDependencyContainer] = None


def get_container() -> EnhancedDependencyContainer:
    """Get the global dependency container."""
    global _global_container
    if _global_container is None:
        _global_container = EnhancedDependencyContainer()
    return _global_container


def register_service(
    service_type: Type[T],
    implementation: Optional[Type[T]] = None,
    lifetime: str = ServiceLifetime.SINGLETON,
    config: Optional[ConfigDict] = None,
) -> None:
    """Register a service in the global container."""
    container = get_container()
    container.register(service_type, implementation, lifetime, config)


async def resolve_service(service_type: Type[T]) -> T:
    """Resolve a service from the global container."""
    container = get_container()
    return await container.resolve(service_type)