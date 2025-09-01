# src/core/enhanced_dependency_injection.py

"""
Deprecated: Forward to src.core.dependency_injection
This module remains for backward compatibility and forwards to the canonical DI container.
"""

from src.core.dependency_injection import (
    DependencyContainer as _DependencyContainer,
    ServiceLifetime as _ServiceLifetime,
)
from typing import Any, TypeVar

T = TypeVar("T")

# Re-export canonical classes
ServiceLifetime = _ServiceLifetime
DependencyContainer = _DependencyContainer

# Global container instance (backward compatibility)
_global_container: _DependencyContainer | None = None



