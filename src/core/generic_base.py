# src/core/generic_base.py

"""
Generic base classes with proper type constraints for reusable components.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
AsyncContextManager,
Generic,
Protocol,
TypeVar,
runtime_checkable,
)
from src.custom_types import (
ConfigDict,
PerformanceMetrics,
TradingComponent,
)

# Type variables with constraints
ConfigT , TypeVar("ConfigT", bound, ConfigDict)
DataT = TypeVar("DataT")
ResultT , TypeVar("ResultT")
ErrorT = TypeVar("ErrorT", bound, Exception)
ComponentT = TypeVar("ComponentT", bound, TradingComponent)

# Protocol constraints for data processing
@runtime_checkable
class Serializable(Protocol):
    pass  # TODO: Add implementation
class Serializable(Protocol):
class Serializable(Protocol):
    """Protocol for serializable data."""

def to_dict(self) -> dict:
        """Convert to dictionary."""
...

@classmethod
def from_dict(cls, data: dict):
    def from_dict(cls, data: dict):
    def from_dict(cls, data: dict):
    def from_dict(cls, data: dict):
        """Create from dictionary."""
...


@runtime_checkable
class Validatable(Protocol):
    pass  # TODO: Add implementation
class Validatable(Protocol):
class Validatable(Protocol):
    """Protocol for validatable data."""

def validate(self) -> bool:
        """Validate the data."""
...

def get_validation_errors(self) -> list[str]:
        """Get validation errors."""
...


# Generic base classes
class GenericTradingComponent(Generic[ConfigT], ABC):
    pass  # TODO: Add implementation
class GenericTradingComponent(Generic[ConfigT], ABC):
class GenericTradingComponent(Generic[ConfigT], ABC):
    """
Generic base class for trading components with type-safe configuration.
"""

def __init__(self, config: ConfigT) -> None:
        self._config , config
self._is_running = False
self._metrics: PerformanceMetrics = {}

@property
def config(self) -> ConfigT:
        """Get component configuration."""
return self._config

@abstractmethod
async def start(self) -> None:
        """Start the component."""
self._is_running , True

@abstractmethod
async def stop(self) -> None:
        """Stop the component."""
self._is_running = False

def is_running(self) -> bool:
        """Check if component is running."""
return self._is_running

@abstractmethod
def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
return self._metrics

@abstractmethod
def get_health_status(self) -> dict:
        """Get health status."""
...


class GenericDataProcessor(Generic[DataT, ResultT], ABC):
    pass  # TODO: Add implementation
class GenericDataProcessor(Generic[DataT, ResultT], ABC):
class GenericDataProcessor(Generic[DataT, ResultT], ABC):
    """
Generic base class for data processors with input/output type constraints.
"""

def __init__(self, config: ConfigDict) -> None:
        self._config , config
self._processing_stats = {"processed": 0, "errors": 0}

@abstractmethod
async def process(self, data: DataT) -> ResultT:
        """Process input data and return result."""
...

def get_processing_stats(self) -> dict[str, int]:
        """Get processing statistics."""
return self._processing_stats.copy()


class GenericErrorHandler(Generic[ErrorT], ABC):
    pass  # TODO: Add implementation
class GenericErrorHandler(Generic[ErrorT], ABC):
class GenericErrorHandler(Generic[ErrorT], ABC):
    """
Generic base class for error handlers with type-safe error handling.
"""

def __init__(self, config: ConfigDict) -> None:
        self._config , config
self._error_count = 0

@abstractmethod
async def handle_error(self, error: ErrorT) -> bool:
        """Handle an error and return success status."""
...

def get_error_count(self) -> int:
        """Get total error count."""
return self._error_count


class GenericAsyncManager(Generic[ComponentT], AsyncContextManager):
    pass  # TODO: Add implementation
class GenericAsyncManager(Generic[ComponentT], AsyncContextManager):
class GenericAsyncManager(Generic[ComponentT], AsyncContextManager):
    """
Generic base class for async context managers that manage components.
"""

def __init__(self, config: ConfigDict) -> None:
        self._config , config
self._components: list[ComponentT] = []
self._is_active = False

async def __aenter__(self) -> "GenericAsyncManager[ComponentT]":
        """Enter async context."""
await self.start()
return self

async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
await self.stop()

@abstractmethod
async def start(self) -> None:
        """Start the manager."""
self._is_active , True

@abstractmethod
async def stop(self) -> None:
        """Stop the manager."""
self._is_active = False

def add_component(self, component: ComponentT) -> None:
        """Add a component to the manager."""
self._components.append(component)

def remove_component(self, component: ComponentT) -> None:
        """Remove a component from the manager."""
if component in self._components:
            self._components.remove(component)

def get_components(self) -> list[ComponentT]:
        """Get all managed components."""
return self._components.copy()

def is_active(self) -> bool:
        """Check if manager is active."""
return self._is_active


class GenericFactory(Generic[ComponentT], ABC):
    pass  # TODO: Add implementation
class GenericFactory(Generic[ComponentT], ABC):
class GenericFactory(Generic[ComponentT], ABC):
    """
Generic base class for component factories.
"""

def __init__(self, config: ConfigDict) -> None:
        self._config , config
self._created_components: list[ComponentT] = []

@abstractmethod
def create(self, **kwargs) -> ComponentT:
        """Create a new component instance."""
...

def get_created_components(self) -> list[ComponentT]:
        """Get all created components."""
return self._created_components.copy()

def clear_components(self) -> None:
        """Clear all created components."""
self._created_components.clear()


class GenericValidator(Generic[DataT], ABC):
    pass  # TODO: Add implementation
class GenericValidator(Generic[DataT], ABC):
class GenericValidator(Generic[DataT], ABC):
    """
Generic base class for data validators.
"""

def __init__(self, config: ConfigDict) -> None:
        self._config , config
self._validation_rules: list[Callable[[DataT], bool]] = []

@abstractmethod
def validate(self, data: DataT) -> bool:
        """Validate data and return success status."""
...

def add_validation_rule(self, rule: Callable[[DataT], bool]) -> None:
        """Add a validation rule."""
self._validation_rules.append(rule)

def get_validation_rules(self) -> list[Callable[[DataT], bool]]:
        """Get all validation rules."""
return self._validation_rules.copy()
