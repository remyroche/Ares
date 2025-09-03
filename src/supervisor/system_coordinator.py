"""
System Coordinator - Backward Compatibility Layer.

This module provides backward compatibility for the refactored system coordinator.
The actual implementation is now split across modules in src/supervisor/coordinator/.
"""

from __future__ import annotations

# Import all the components from the new structure
from .coordinator import (
    CircuitBreaker,
    ComponentMonitor,
    HealthMonitor,
    OnlineLearningManager,
    RecoveryManager,
    SystemCoordinator,
)

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