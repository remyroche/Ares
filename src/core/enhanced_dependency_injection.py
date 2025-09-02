# src/core/enhanced_dependency_injection.py

"""
Deprecated: Forward to src.core.dependency_injection
This module remains for backward compatibility and forwards to the canonical DI container.
"""

from typing import Any, TypeVar
from enum import Enum

T = TypeVar("T")


class ServiceLifetime(Enum):
    """Service lifetime constants compatible with enhanced DI usage."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class DependencyContainer:
    """
    Minimal dependency injection container for backward compatibility.
    This is a placeholder implementation since the main DI container is corrupted.
    """
    
    def __init__(self):
        self._services = {}
        self._instances = {}
    
    def register(
        self,
        service_type: type[T],
        implementation: type[T] | None = None,
        singleton: bool = True,
        config: dict[str, Any] | None = None,
        lifetime: str = ServiceLifetime.SINGLETON,
    ) -> None:
        """Register a service with the container."""
        # Store service registration
        self._services[service_type] = {
            'implementation': implementation or service_type,
            'singleton': singleton,
            'config': config,
            'lifetime': lifetime
        }
    
    def resolve(self, service_type: type[T]) -> T:
        """Resolve a service from the container."""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        service_info = self._services[service_type]
        
        # For singleton services, return existing instance or create new one
        if service_info['singleton']:
            if service_type not in self._instances:
                implementation = service_info['implementation']
                self._instances[service_type] = implementation()
            return self._instances[service_type]
        
        # For transient services, create new instance
        implementation = service_info['implementation']
        return implementation()


# Global container instance (backward compatibility)
_global_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    global _global_container

    if _global_container is None:
        _global_container = DependencyContainer()
    return _global_container


def register_service(
    service_type: type[T],
    implementation: type[T] | None = None,
    lifetime: str = ServiceLifetime.SINGLETON,
    config: dict[str, Any] | None = None,
) -> None:
    container = get_container()

    # Use type as key to align with canonical container usage
    container.register(
        service_type,
        implementation=implementation,
        singleton=(lifetime == ServiceLifetime.SINGLETON),
        config=config,
        lifetime=lifetime,
    )


async def resolve_service(service_type: type[T]) -> T:
    container = get_container()

    return container.resolve(service_type)
