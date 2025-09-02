# src/core/__init__.py

# Temporarily commented out due to corruption issues
# from .dependency_injection import (ComponentFactory, DependencyContainer,
#                                    ModularTradingSystem, ServiceRegistration)
from .enhanced_dependency_injection import DependencyContainer, ServiceLifetime

__all__ = [
    "DependencyContainer",
    "ServiceLifetime",
]
