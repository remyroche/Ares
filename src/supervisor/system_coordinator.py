"""
System Coordinator - Backward Compatibility Layer.

This module provides backward compatibility for the refactored system coordinator.
The actual implementation is now split across modules in src/supervisor/coordinator/.
"""


# Import all the components from the new structure
from .core.circuit_breaker import CircuitBreaker
from .core.component_monitor import ComponentMonitor
from .core.health_monitor import HealthMonitor
from .core.online_learning_manager import OnlineLearningManager
from .core.recovery_manager import RecoveryManager
from .core.system_coordinator import SystemCoordinator

# Re-export everything for backward compatibility
__all__ = [
    "CircuitBreaker",
    "OnlineLearningManager",
    "SystemCoordinator",
    "ComponentMonitor", 
    "HealthMonitor",
    "RecoveryManager",
]

# Create aliases for backward compatibility
# The old file had "Supervisor" class, now it's "SystemCoordinator"
Supervisor = SystemCoordinator