"""
Dependency Injection System for Simplified ML Pipeline

This module implements a clean dependency injection pattern to replace
hidden imports and complex dependency chains.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar('T')


@dataclass
class ServiceDescriptor:
    """Describes a service in the DI container."""
    service_type: Type
    factory: Callable[..., Any]
    singleton: bool = False
    dependencies: list[str] = None
    instance: Optional[Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""


class DIContainer:
    """
    Dependency Injection Container for managing pipeline services.
    
    Features:
    - Explicit service registration
    - Singleton and transient lifetimes
    - Automatic dependency resolution
    - Circular dependency detection
    """
    
    def __init__(self):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._resolving: set[str] = set()  # Track services being resolved
        self.logger = logging.getLogger(__name__)
    
    def register(
        self,
        name: str,
        service_type: Type[T],
        factory: Optional[Callable[..., T]] = None,
        singleton: bool = False,
        dependencies: Optional[list[str]] = None
    ) -> None:
        """
        Register a service in the container.
        
        Args:
            name: Service identifier
            service_type: Type/interface of the service
            factory: Factory function to create instances
            singleton: Whether to maintain single instance
            dependencies: List of required service names
        """
        if factory is None:
            # Default factory is the type constructor
            factory = service_type
        
        descriptor = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            singleton=singleton,
            dependencies=dependencies or []
        )
        
        self._services[name] = descriptor
        self.logger.debug(f"Registered service: {name} (singleton={singleton})")
    
    def register_instance(self, name: str, instance: Any) -> None:
        """Register an existing instance as a singleton."""
        descriptor = ServiceDescriptor(
            service_type=type(instance),
            factory=lambda: instance,
            singleton=True,
            instance=instance
        )
        self._services[name] = descriptor
    
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
            raise CircularDependencyError(
                f"Circular dependency detected while resolving '{name}'"
            )
        
        if name not in self._services:
            raise ServiceNotFoundError(f"Service '{name}' not registered")
        
        descriptor = self._services[name]
        
        # Return existing singleton instance if available
        if descriptor.singleton and descriptor.instance is not None:
            return descriptor.instance
        
        # Mark as being resolved
        self._resolving.add(name)
        
        try:
            # Resolve dependencies
            dependencies = {}
            for dep_name in descriptor.dependencies:
                dependencies[dep_name] = self.get(dep_name)
            
            # Create instance
            if dependencies:
                # Inject dependencies as kwargs
                instance = descriptor.factory(**dependencies)
            else:
                instance = descriptor.factory()
            
            # Store singleton instance
            if descriptor.singleton:
                descriptor.instance = instance
            
            return instance
            
        finally:
            self._resolving.remove(name)
    
    def get_all(self, service_type: Type[T]) -> list[T]:
        """Get all registered services of a specific type."""
        services = []
        for name, descriptor in self._services.items():
            if issubclass(descriptor.service_type, service_type):
                services.append(self.get(name))
        return services
    
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
            raise RuntimeError("ServiceLocator container not initialized")
        return cls._container.get(name)


# Decorator for dependency injection
def inject(**dependencies):
    """
    Decorator to automatically inject dependencies into a class.
    
    Usage:
        @inject(logger='logger', validator='validator')
        class MyService:
            def __init__(self, logger, validator):
                self.logger = logger
                self.validator = validator
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Get container from ServiceLocator if not provided
            container = kwargs.pop('_container', None)
            if container is None:
                container = ServiceLocator._container
            
            if container is not None:
                # Inject dependencies
                for param_name, service_name in dependencies.items():
                    if param_name not in kwargs:
                        kwargs[param_name] = container.get(service_name)
            
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# Example service interfaces
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


# Example implementations
class ConsoleLogger(ILogger):
    """Simple console logger implementation."""
    
    def info(self, message: str) -> None:
        print(f"[INFO] {datetime.now()}: {message}")
    
    def error(self, message: str) -> None:
        print(f"[ERROR] {datetime.now()}: {message}")


class FileLogger(ILogger):
    """File-based logger implementation."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def info(self, message: str) -> None:
        self._write_log("INFO", message)
    
    def error(self, message: str) -> None:
        self._write_log("ERROR", message)
    
    def _write_log(self, level: str, message: str) -> None:
        with open(self.log_file, 'a') as f:
            f.write(f"[{level}] {datetime.now()}: {message}\n")


class DataValidator(IDataValidator):
    """Basic data validator implementation."""
    
    def __init__(self, logger: ILogger = None):
        self.logger = logger or ConsoleLogger()
    
    def validate(self, data: Any) -> bool:
        """Validate data structure and content."""
        if data is None:
            self.logger.error("Data is None")
            return False
        
        if hasattr(data, 'empty') and data.empty:
            self.logger.error("Data is empty")
            return False
        
        self.logger.info("Data validation passed")
        return True


# Pipeline-specific container setup
def create_pipeline_container(config: dict) -> DIContainer:
    """
    Create and configure DI container for ML pipeline.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Configured DIContainer
    """
    container = DIContainer()
    
    # Register core services
    if config.get('logging', {}).get('type') == 'file':
        log_file = Path(config['logging']['file'])
        container.register(
            'logger',
            ILogger,
            lambda: FileLogger(log_file),
            singleton=True
        )
    else:
        container.register(
            'logger',
            ILogger,
            ConsoleLogger,
            singleton=True
        )
    
    # Register validator with logger dependency
    container.register(
        'validator',
        IDataValidator,
        lambda logger: DataValidator(logger),
        singleton=True,
        dependencies=['logger']
    )
    
    # Register other services based on config
    if config.get('cache', {}).get('enabled', False):
        from .caching import MemoryCache  # Would be implemented
        container.register(
            'cache',
            MemoryCache,
            lambda: MemoryCache(config['cache']),
            singleton=True
        )
    
    return container


# Usage example
if __name__ == "__main__":
    # Configure container
    config = {
        'logging': {'type': 'console'},
        'cache': {'enabled': True}
    }
    
    container = create_pipeline_container(config)
    
    # Use services
    logger = container.get('logger')
    validator = container.get('validator')
    
    logger.info("Pipeline initialized")
    
    # Example with injection decorator
    @inject(logger='logger', validator='validator')
    class MyService:
        def __init__(self, logger: ILogger, validator: IDataValidator):
            self.logger = logger
            self.validator = validator
        
        def process(self, data):
            if self.validator.validate(data):
                self.logger.info("Processing data...")
            else:
                self.logger.error("Invalid data")
    
    # Service will have dependencies automatically injected
    ServiceLocator.set_container(container)
    service = MyService()
    service.process({"test": "data"})