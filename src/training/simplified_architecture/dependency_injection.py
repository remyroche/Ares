"""
Enhanced Dependency Injection System for Simplified ML Pipeline

This module implements a comprehensive dependency injection pattern to replace
hidden imports and complex dependency chains in the monolithic architecture.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
import pandas as pd
import asyncio
from enum import Enum

T = TypeVar('T')

class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

@dataclass
class ServiceDescriptor:
    """Enhanced service descriptor with lifecycle management."""
    service_type: Type
    factory: Callable[..., Any]
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    dependencies: List[str] = field(default_factory=list)
    instance: Optional[Any] = None
    scope_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []

class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""

class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""

class ServiceRegistrationError(Exception):
    """Raised when service registration fails."""

class EnhancedDIContainer:
    """
    Enhanced Dependency Injection Container for managing pipeline services.
    
    Features:
    - Explicit service registration with lifecycle management
    - Singleton, transient, and scoped lifetimes
    - Automatic dependency resolution with validation
    - Circular dependency detection and prevention
    - Service metadata and configuration
    - Async service support
    - Service health monitoring
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._services: Dict[str, ServiceDescriptor] = {}
        self._resolving: set[str] = set()
        self._scopes: Dict[str, Dict[str, Any]] = {}
        self._current_scope: Optional[str] = None
        self.logger = logger or logging.getLogger(__name__)
        self._service_health: Dict[str, bool] = {}

    def register_singleton(self, name: str, service_type: Type[T], factory: Optional[Callable[..., T]] = None, dependencies: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a singleton service."""
        self._register_service(name, service_type, factory, ServiceLifetime.SINGLETON, dependencies, metadata)

    def register_transient(self, name: str, service_type: Type[T], factory: Optional[Callable[..., T]] = None, dependencies: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a transient service."""
        self._register_service(name, service_type, factory, ServiceLifetime.TRANSIENT, dependencies, metadata)

    def register_scoped(self, name: str, service_type: Type[T], factory: Optional[Callable[..., T]] = None, dependencies: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a scoped service."""
        self._register_service(name, service_type, factory, ServiceLifetime.SCOPED, dependencies, metadata)

    def register_factory(self, name: str, factory: Callable[..., T], lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT, dependencies: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a service with a custom factory function."""
        self._register_service(name, None, factory, lifetime, dependencies, metadata)

    def register_instance(self, name: str, instance: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an existing instance as a singleton."""
        descriptor = ServiceDescriptor(
            service_type=type(instance),
            factory=lambda: instance,
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance,
            metadata=metadata or {}
        )
        self._services[name] = descriptor
        self._service_health[name] = True
        self.logger.debug(f'Registered instance: {name} (type: {type(instance).__name__})')

    def _register_service(self, name: str, service_type: Optional[Type[T]], factory: Optional[Callable[..., T]], lifetime: ServiceLifetime, dependencies: Optional[List[str]], metadata: Optional[Dict[str, Any]]) -> None:
        """Internal method to register a service."""
        if name in self._services:
            raise ServiceRegistrationError(f"Service '{name}' is already registered")
        
        if factory is None and service_type is None:
            raise ServiceRegistrationError(f"Either service_type or factory must be provided for service '{name}'")
        
        if factory is None:
            factory = service_type
        
        descriptor = ServiceDescriptor(
            service_type=service_type or type(factory()),
            factory=factory,
            lifetime=lifetime,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self._services[name] = descriptor
        self._service_health[name] = True
        self.logger.debug(f'Registered service: {name} (lifetime: {lifetime.value}, dependencies: {dependencies})')

    def get(self, name: str) -> Any:
        """
        Resolve and return a service instance.
        
        Args:
            name: Service identifier
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotFoundError: If service not registered
            CircularDependencyError: If circular dependencies detected
        """
        if name in self._resolving:
            raise CircularDependencyError(f"Circular dependency detected while resolving '{name}'")
        
        if name not in self._services:
            raise ServiceNotFoundError(f"Service '{name}' not registered")
        
        descriptor = self._services[name]
        
        # Handle different lifetimes
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if self._current_scope and self._current_scope in self._scopes:
                if name in self._scopes[self._current_scope]:
                    return self._scopes[self._current_scope][name]
        
        self._resolving.add(name)
        try:
            # Resolve dependencies
            dependencies = {}
            for dep_name in descriptor.dependencies:
                dependencies[dep_name] = self.get(dep_name)
            
            # Create instance
            instance = descriptor.factory(**dependencies)
            
            # Store based on lifetime
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                descriptor.instance = instance
            elif descriptor.lifetime == ServiceLifetime.SCOPED:
                if self._current_scope:
                    if self._current_scope not in self._scopes:
                        self._scopes[self._current_scope] = {}
                    self._scopes[self._current_scope][name] = instance
            
            self._service_health[name] = True
            return instance
            
        except Exception as e:
            self._service_health[name] = False
            self.logger.error(f"Failed to resolve service '{name}': {e}")
            raise
        finally:
            self._resolving.discard(name)

    def create_scope(self, scope_id: str) -> 'ServiceScope':
        """Create a new service scope."""
        return ServiceScope(self, scope_id)

    def get_health_status(self) -> Dict[str, bool]:
        """Get health status of all registered services."""
        return self._service_health.copy()

    def validate_dependencies(self) -> List[str]:
        """Validate that all service dependencies can be resolved."""
        errors = []
        for name in self._services:
            try:
                # Test resolution without actually creating instances
                self._validate_service_dependencies(name, set())
            except Exception as e:
                errors.append(f"Service '{name}': {e}")
        return errors

    def _validate_service_dependencies(self, name: str, visited: set[str]) -> None:
        """Recursively validate service dependencies."""
        if name in visited:
            raise CircularDependencyError(f"Circular dependency detected: {' -> '.join(visited)} -> {name}")
        
        if name not in self._services:
            raise ServiceNotFoundError(f"Service '{name}' not registered")
        
        visited.add(name)
        descriptor = self._services[name]
        
        for dep_name in descriptor.dependencies:
            self._validate_service_dependencies(dep_name, visited.copy())

    def get_service_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a registered service."""
        if name not in self._services:
            return None
        
        descriptor = self._services[name]
        return {
            'name': name,
            'type': descriptor.service_type.__name__ if descriptor.service_type else 'Unknown',
            'lifetime': descriptor.lifetime.value,
            'dependencies': descriptor.dependencies,
            'metadata': descriptor.metadata,
            'is_healthy': self._service_health.get(name, False),
            'has_instance': descriptor.instance is not None
        }

    def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services with their information."""
        return [self.get_service_info(name) for name in self._services.keys()]

class ServiceScope:
    """Context manager for service scopes."""
    
    def __init__(self, container: EnhancedDIContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id
        self._previous_scope = None

    def __enter__(self):
        self._previous_scope = self.container._current_scope
        self.container._current_scope = self.scope_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container._current_scope = self._previous_scope
        # Clean up scope instances
        if self.scope_id in self.container._scopes:
            del self.container._scopes[self.scope_id]

# Decorator for automatic service registration
def injectable(lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT, dependencies: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Decorator to mark a class as injectable."""
    def decorator(cls):
        cls._di_lifetime = lifetime
        cls._di_dependencies = dependencies or []
        cls._di_metadata = metadata or {}
        return cls
    return decorator

# Decorator for dependency injection
def inject(*service_names: str):
    """Decorator to inject dependencies into a method."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get DI container from self or global registry
            container = getattr(self, '_di_container', None)
            if container is None:
                raise ServiceNotFoundError("No DI container available for injection")
            
            # Inject services
            for service_name in service_names:
                if service_name not in kwargs:
                    kwargs[service_name] = container.get(service_name)
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._resolving.clear()

class ServiceLocator:
    """
    Global service locator pattern (use sparingly).
    Provides static access to a DI container.
    """
    _container: Optional[DIContainer] = None

    @classmethod
    def set_container(cls, container: DIContainer) -> None:
        """Set the global container instance."""
        cls._container = container

    @classmethod
    def get(cls, name: str) -> Any:
        """Get a service from the global container."""
        if cls._container is None:
            raise RuntimeError('ServiceLocator container not initialized')
        return cls._container.get(name)

def inject(**dependencies) -> None:
    """
    Decorator to automatically inject dependencies into a class.
    
    Usage:
        @inject(logger='logger', validator='validator')
        class MyService:
            def __init__(self, logger, validator):
                self.logger = logger
                self.validator = validator
    """

    def decorator(cls) -> None:
        original_init = cls.__init__

        def new_init(self, *args, **kwargs) -> None:
            container = kwargs.pop('_container', None)
            if container is None:
                container = ServiceLocator._container
            if container is not None:
                for param_name, service_name in dependencies.items():
                    if param_name not in kwargs:
                        kwargs[param_name] = container.get(service_name)
            original_init(self, *args, **kwargs)
        cls.__init__ = new_init
        return cls
    return decorator

class ILogger(ABC):
    """Logger interface."""

    @abstractmethod
    def info(self, message: str) -> None:
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        pass

class IDataValidator(ABC):
    """Data validator interface."""

    @abstractmethod
    def validate(self, data: Any) -> bool:
        pass

class IFeatureEngine(ABC):
    """Feature engineering interface."""

    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        pass

class ConsoleLogger(ILogger):
    """Simple console logger implementation."""

    def info(self, message: str) -> None:
        print(f'[INFO] {datetime.now()}: {message}')

    def error(self, message: str) -> None:
        print(f'[ERROR] {datetime.now()}: {message}')

class FileLogger(ILogger):
    """File-based logger implementation."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        self._write_log('INFO', message)

    def error(self, message: str) -> None:
        self._write_log('ERROR', message)

    def _write_log(self, level: str, message: str) -> None:
        with open(self.log_file, 'a') as f:
            f.write(f'[{level}] {datetime.now()}: {message}\n')

class DataValidator(IDataValidator):
    """Basic data validator implementation."""

    def __init__(self, logger: ILogger=None) -> None:
        self.logger = logger or ConsoleLogger()

    def validate(self, data: Any) -> bool:
        """Validate data structure and content."""
        if data is None:
            self.logger.error('Data is None')
            return False
        if hasattr(data, 'empty') and data.empty:
            self.logger.error('Data is empty')
            return False
        self.logger.info('Data validation passed')
        return True

def create_pipeline_container(config: dict) -> DIContainer:
    """
    Create and configure DI container for ML pipeline.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Configured DIContainer
    """
    container = DIContainer()
    if config.get('logging', {}).get('type') == 'file':
        log_file = Path(config['logging']['file'])
        container.register('logger', ILogger, lambda: FileLogger(log_file), singleton=True)
    else:
        container.register('logger', ILogger, ConsoleLogger, singleton=True)
    container.register('validator', IDataValidator, lambda logger: DataValidator(logger), singleton=True, dependencies=['logger'])
    if config.get('cache', {}).get('enabled', False):
        from .caching import MemoryCache
        container.register('cache', MemoryCache, lambda: MemoryCache(config['cache']), singleton=True)
    return container
if __name__ == '__main__':
    config = {'logging': {'type': 'console'}, 'cache': {'enabled': True}}
    container = create_pipeline_container(config)
    logger = container.get('logger')
    validator = container.get('validator')
    logger.info('Pipeline initialized')

    @inject(logger='logger', validator='validator')
    class MyService:

        def __init__(self, logger: ILogger, validator: IDataValidator) -> None:
            self.logger = logger
            self.validator = validator

        def process(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
            if self.validator.validate(data):
                self.logger.info('Processing data...')
            else:
                self.logger.error('Invalid data')
    ServiceLocator.set_container(container)
    service = MyService()
    service.process({'test': 'data'})