# src/types/protocol_types.py

"""
Protocol definitions for better interface typing and dependency injection.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

from .base_types import Symbol, Timestamp
from .data_types import AccountInfo, MarketDataDict, OrderInfo, PositionInfo
from .ml_types import ModelInput, ModelOutput, PredictionResult
from .trading_types import OrderRequest, RiskParameters, TradeDecision

# Generic type variables
T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=Dict[str, Any])
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")


@runtime_checkable
class DataProvider(Protocol[DataT]):
    """Protocol for data provider implementations."""
    
    @abstractmethod
    async def get_data(self, symbol: Symbol, start: Timestamp, end: Timestamp) -> DataT:
        """Get data for the specified symbol and time range."""
        ...
    
    @abstractmethod
    async def get_latest_data(self, symbol: Symbol) -> DataT:
        """Get the latest data for the specified symbol."""
        ...
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the data provider is connected."""
        ...


@runtime_checkable
class ModelPredictor(Protocol[T]):
    """Protocol for ML model predictors."""
    
    @abstractmethod
    async def predict(self, input_data: ModelInput) -> ModelOutput:
        """Make predictions on input data."""
        ...
    
    @abstractmethod
    async def predict_single(self, features: List[float]) -> PredictionResult:
        """Make a single prediction."""
        ...
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        ...
    
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if the model is trained and ready for prediction."""
        ...


@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management implementations."""
    
    @abstractmethod
    async def assess_risk(self, trade_decision: TradeDecision) -> float:
        """Assess risk for a trade decision."""
        ...
    
    @abstractmethod
    async def validate_order(self, order: OrderRequest) -> bool:
        """Validate if an order meets risk requirements."""
        ...
    
    @abstractmethod
    def get_risk_parameters(self) -> RiskParameters:
        """Get current risk parameters."""
        ...
    
    @abstractmethod
    async def update_risk_parameters(self, params: RiskParameters) -> bool:
        """Update risk parameters."""
        ...


@runtime_checkable
class OrderExecutor(Protocol):
    """Protocol for order execution implementations."""
    
    @abstractmethod
    async def execute_order(self, order: OrderRequest) -> OrderInfo:
        """Execute a trading order."""
        ...
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        ...
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderInfo:
        """Get status of an order."""
        ...
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[OrderInfo]:
        """Get all open orders."""
        ...


@runtime_checkable
class StateManager(Protocol[T]):
    """Protocol for state management implementations."""
    
    @abstractmethod
    async def get_state(self, key: str) -> Optional[T]:
        """Get state value by key."""
        ...
    
    @abstractmethod
    async def set_state(self, key: str, value: T) -> bool:
        """Set state value by key."""
        ...
    
    @abstractmethod
    async def delete_state(self, key: str) -> bool:
        """Delete state by key."""
        ...
    
    @abstractmethod
    async def get_all_states(self) -> Dict[str, T]:
        """Get all states."""
        ...


@runtime_checkable
class EventHandler(Protocol[T]):
    """Protocol for event handling implementations."""
    
    @abstractmethod
    async def handle_event(self, event_type: str, data: T) -> None:
        """Handle an event."""
        ...
    
    @abstractmethod
    async def subscribe(self, event_type: str) -> None:
        """Subscribe to an event type."""
        ...
    
    @abstractmethod
    async def unsubscribe(self, event_type: str) -> None:
        """Unsubscribe from an event type."""
        ...


@runtime_checkable
class Configurable(Protocol[ConfigT]):
    """Protocol for configurable components."""
    
    @abstractmethod
    def configure(self, config: ConfigT) -> None:
        """Configure the component."""
        ...
    
    @abstractmethod
    def get_config(self) -> ConfigT:
        """Get current configuration."""
        ...
    
    @abstractmethod
    def validate_config(self, config: ConfigT) -> bool:
        """Validate configuration."""
        ...


@runtime_checkable
class Monitorable(Protocol):
    """Protocol for monitorable components."""
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        ...
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        ...
    
    @abstractmethod
    def get_status(self) -> str:
        """Get current status."""
        ...


@runtime_checkable
class Startable(Protocol):
    """Protocol for startable/stoppable components."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        ...
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the component."""
        ...
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if component is running."""
        ...


# Composite protocols for common patterns
@runtime_checkable
class TradingComponent(Configurable[ConfigT], Monitorable, Startable, Protocol[ConfigT]):
    """Protocol for trading system components."""
    pass


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