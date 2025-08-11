# src/core/enhanced_dependency_injection.py

"""
Deprecated: Forward to src.core.dependency_injection
This module remains for backward compatibility and forwards to the canonical DI container.
"""

from typing import Any, TypeVar

from src.core.dependency_injection import (
    AsyncServiceContainer as _AsyncServiceContainer,
)
from src.core.dependency_injection import (
    DependencyContainer as _DependencyContainer,
)
from src.core.dependency_injection import (
    ServiceLifetime as _ServiceLifetime,
)

T = TypeVar("T")

# Re-export canonical classes
ServiceLifetime = _ServiceLifetime
DependencyContainer = _DependencyContainer
AsyncServiceContainer = _AsyncServiceContainer

# Global container instance (backward compatibility)
_global_container: _DependencyContainer | None = None


def get_container() -> _DependencyContainer:
    global _global_container
    if _global_container is None:
        _global_container = _DependencyContainer()
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
        service_type,
        implementation=implementation,
        singleton=(lifetime == ServiceLifetime.SINGLETON),
        config=config,
        lifetime=lifetime,
    )


async def resolve_service(service_type: type[T]) -> T:
    container = get_container()
    return container.resolve(service_type)
