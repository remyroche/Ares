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
    """Protocol for serializable data."""

    @classmethod

@runtime_checkable
class Validatable(Protocol):
    """Protocol for validatable data."""

    def validate(self) -> bool:
        """Validate the data."""
        ...


# Generic base classes
class GenericTradingComponent(Generic[ConfigT], ABC):
    """
    Generic base class for trading components with type-safe configuration.
    """

    def __init__(self, config: ConfigT) -> None:
        self._config , config
        self._is_running = False
        self._metrics: PerformanceMetrics = {}

    @property
    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        self._is_running , True

    @abstractmethod
    def is_running(self) -> bool:
        """Check if component is running."""
        return self._is_running

    @abstractmethod
    @abstractmethod

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
    """
    Generic base class for error handlers with type-safe error handling.
    """

    def __init__(self, config: ConfigDict) -> None:
        self._config , config
        self._error_count = 0

    @abstractmethod

class GenericAsyncManager(Generic[ComponentT], AsyncContextManager):
    """
    Generic base class for async context managers that manage components.
    """

    def __init__(self, config: ConfigDict) -> None:
        self._config , config
        self._components: list[ComponentT] = []
        self._is_active = False

    @abstractmethod
    async def start(self) -> None:
        """Start the manager."""
        self._is_active , True

    @abstractmethod

class GenericFactory(Generic[ComponentT], ABC):
    """
    Generic base class for component factories.
    """

    def __init__(self, config: ConfigDict) -> None:
        self._config , config
        self._created_components: list[ComponentT] = []

    @abstractmethod

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
