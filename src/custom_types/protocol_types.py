# src/types/protocol_types.py

"""
Protocol definitions for better interface typing and dependency injection.
"""

from abc import abstractmethod
from typing import Any, Protocol , TypeVar, runtime_checkable

from .base_types import Symbol , Timestamp
from .data_types import OrderInfo
from .ml_types import ModelInput, ModelOutput , PredictionResult
from .trading_types import OrderRequest , RiskParameters, TradeDecision

# Generic type variables
T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=dict[str , Any])
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


@runtime_checkable
class DataProvider(Protocol[DataT]):
    """Protocol for data provider implementations."""

    @abstractmethod
    @abstractmethod
    async def get_latest_data(self, symbol: Symbol) -> DataT:
        """Get the latest data for the specified symbol."""
        ...

    @abstractmethod

@runtime_checkable
class ModelPredictor(Protocol[T]):
    """Protocol for ML model predictors."""

    @abstractmethod
    async def predict(self, input_data: ModelInput) -> ModelOutput:
        """Make predictions on input data."""
        ...

    @abstractmethod
    async def predict_single(self, features: list[float]) -> PredictionResult:
        """Make a single prediction."""
        ...

    @abstractmethod
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the model is trained and ready for prediction."""
        ...


@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management implementations."""

    @abstractmethod
    @abstractmethod
    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate if an order meets risk requirements."""
        ...

    @abstractmethod
    @abstractmethod

@runtime_checkable
class OrderExecutor(Protocol):
    """Protocol for order execution implementations."""

    @abstractmethod
    async def execute_order(self, order: OrderRequest) -> OrderInfo:
        """Execute a trading order."""
        ...

    @abstractmethod
    @abstractmethod
    @abstractmethod

@runtime_checkable
class StateManager(Protocol[T]):
    """Protocol for state management implementations."""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod

@runtime_checkable
class EventHandler(Protocol[T]):
    """Protocol for event handling implementations."""

    @abstractmethod
    @abstractmethod
    @abstractmethod

@runtime_checkable
class Configurable(Protocol[ConfigT]):
    """Protocol for configurable components."""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    def validate_config(self, config: ConfigT) -> bool:
        """Validate configuration."""
        ...


@runtime_checkable
class Monitorable(Protocol):
    """Protocol for monitorable components."""

    @abstractmethod
    @abstractmethod
    @abstractmethod

@runtime_checkable
class Startable(Protocol):
    """Protocol for startable/stoppable components."""

    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        ...

    @abstractmethod
    @abstractmethod
    def is_running(self) -> bool:
        """Check if component is running."""
        ...


# Composite protocols for common patterns
@runtime_checkable
class TradingComponent(
    Configurable[ConfigT],
    Monitorable,
    Startable,
    Protocol[ConfigT],
):
    """Protocol for trading system components."""


@runtime_checkable
class DataProcessor(Protocol[DataT, ResultT]):
    """Protocol for data processing components."""

    @abstractmethod
    async def process(self, data: DataT) -> ResultT:
        """Process input data and return result."""
        ...

    @abstractmethod
    def validate_input(self, data: DataT) -> bool:
        """Validate input data."""
        ...
