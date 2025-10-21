"""
Base component for pre-training steps.

This module provides the base component class and related data structures.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class ComponentStatus(Enum):
    """Status of a component."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ComponentResult:
    """Result of component execution."""
    success: bool
    status: ComponentStatus
    data: Any = None
    metrics: Dict[str, Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.status is None:
            self.status = ComponentStatus.COMPLETED if self.success else ComponentStatus.FAILED

@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None
    timeout: float = 300.0  # 5 minutes default timeout
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

class BaseComponent:
    """Base class for all components."""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.params = config.params
        self.timeout = config.timeout
        self.status = ComponentStatus.PENDING
    
    def initialize(self) -> ComponentResult:
        """Initialize the component."""
        try:
            self.status = ComponentStatus.RUNNING
            # TODO: Implement initialization logic
            return ComponentResult(
                success=True,
                status=ComponentStatus.COMPLETED,
                metrics={"initialized": True}
            )
        except Exception as e:
            self.status = ComponentStatus.FAILED
            return ComponentResult(
                success=False,
                status=ComponentStatus.FAILED,
                error=str(e)
            )
    
    def process(self, data: Any) -> ComponentResult:
        """Process data through the component."""
        if not self.enabled:
            return ComponentResult(
                success=True,
                status=ComponentStatus.SKIPPED,
                data=data,
                metrics={"skipped": True}
            )
        
        try:
            self.status = ComponentStatus.RUNNING
            # TODO: Implement processing logic
            processed_data = data  # Placeholder
            return ComponentResult(
                success=True,
                status=ComponentStatus.COMPLETED,
                data=processed_data,
                metrics={"processed": True}
            )
        except Exception as e:
            self.status = ComponentStatus.FAILED
            return ComponentResult(
                success=False,
                status=ComponentStatus.FAILED,
                error=str(e)
            )
    
    def cleanup(self) -> ComponentResult:
        """Cleanup component resources."""
        try:
            # TODO: Implement cleanup logic
            return ComponentResult(
                success=True,
                status=ComponentStatus.COMPLETED,
                metrics={"cleaned_up": True}
            )
        except Exception as e:
            return ComponentResult(
                success=False,
                status=ComponentStatus.FAILED,
                error=str(e)
            )
    
    def get_status(self) -> ComponentStatus:
        """Get current component status."""
        return self.status
    
    def is_enabled(self) -> bool:
        """Check if component is enabled."""
        return self.enabled

__all__ = [
    'ComponentStatus',
    'ComponentResult', 
    'ComponentConfig',
    'BaseComponent'
]