"""
Exchange Interface Definitions

This module defines the core interfaces and types for exchange operations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime


class ExchangeType(Enum):
    """Types of exchanges supported"""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"
    OPTIONS = "options"
    DERIVATIVES = "derivatives"


class ExchangeStatus(Enum):
    """Status of exchange connections"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class IExchange(ABC):
    """
    Core exchange interface that all exchanges must implement.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the exchange connection."""

    @abstractmethod
    async def close(self) -> None:
        """Close the exchange connection."""

    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @abstractmethod
    async def get_status(self) -> ExchangeStatus:
        """Get current exchange status."""

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""

    @abstractmethod
    async def get_balance(self, currency: str) -> Dict[str, Any]:
        """Get balance for a specific currency."""

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new order."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order."""

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information."""

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get kline/candlestick data."""


@dataclass
class ExchangeConfig:
    """Configuration for an exchange"""
    name: str
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    base_url: str
    sandbox: bool = False
    rate_limits: Dict[str, int] = field(default_factory=dict)
    supported_symbols: List[str] = field(default_factory=list)
    features: Dict[str, bool] = field(default_factory=dict)


@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResponse:
    """Order response structure"""
    order_id: str
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    commission_asset: str = ""
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Import standardized market data from shared module
try:
    from exchanges.shared import StandardizedMarketData as MarketDataPoint
except ImportError:
    @dataclass
    class MarketDataPoint:
        """Single market data point - fallback definition"""
        symbol: str
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
        interval: str = "1m"


@dataclass
class ExchangeMetrics:
    """Exchange performance metrics"""
    exchange_name: str
    connection_status: ExchangeStatus
    last_heartbeat: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    success_rate: float = 100.0
    active_orders: int = 0
    total_volume_24h: float = 0.0


class IExchangeAdapter(ABC):
    """
    Adapter interface for connecting different exchange implementations.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the exchange."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""

    @abstractmethod
    async def __aenter__(self):
        """Async context manager entry."""
        success = await self.connect()
        if not success:
            raise ConnectionError("Failed to connect to exchange")
        return self

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to the exchange."""

    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test the exchange connection."""

    @abstractmethod
    async def execute_order(self, order_request: OrderRequest) -> OrderResponse:
        """Execute an order on the exchange."""

    @abstractmethod
    async def query_order(self, order_id: str) -> OrderResponse:
        """Query order status from the exchange."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order on the exchange."""

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from the exchange."""

    @abstractmethod
    async def get_market_data(
        self,
        symbol: str,
        data_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get market data from the exchange."""


class IMessageRouter(ABC):
    """
    Interface for routing messages between exchanges and the trading system.
    """

    @abstractmethod
    async def route_order(
        self,
        order_request: OrderRequest,
        target_exchanges: List[str]
    ) -> Dict[str, OrderResponse]:
        """Route an order to target exchanges."""

    @abstractmethod
    async def route_data_request(
        self,
        data_type: str,
        symbol: str,
        target_exchanges: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Route a data request to target exchanges."""

    @abstractmethod
    async def route_response(
        self,
        response_type: str,
        response_data: Dict[str, Any],
        source_exchange: str
    ) -> None:
        """Route a response back to the appropriate handler."""

    @abstractmethod
    async def broadcast_message(
        self,
        message_type: str,
        message_data: Dict[str, Any],
        target_exchanges: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Broadcast a message to multiple exchanges."""


class IResponseHandler(ABC):
    """
    Interface for handling responses from exchanges.
    """

    @abstractmethod
    async def handle_order_response(
        self,
        order_response: OrderResponse,
        exchange_name: str
    ) -> None:
        """Handle an order response from an exchange."""

    @abstractmethod
    async def handle_data_response(
        self,
        data_type: str,
        data: Dict[str, Any],
        exchange_name: str
    ) -> None:
        """Handle a data response from an exchange."""

    @abstractmethod
    async def handle_error_response(
        self,
        error_type: str,
        error_data: Dict[str, Any],
        exchange_name: str
    ) -> None:
        """Handle an error response from an exchange."""

    @abstractmethod
    async def handle_status_update(
        self,
        status: ExchangeStatus,
        exchange_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle a status update from an exchange."""


class ExchangeEvent(Enum):
    """Exchange events"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    ORDER_EXECUTED = "order_executed"
    ORDER_FAILED = "order_failed"
    DATA_RECEIVED = "data_received"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    MAINTENANCE_MODE = "maintenance_mode"


@dataclass
class ExchangeEventData:
    """Data for exchange events"""
    event_type: ExchangeEvent
    exchange_name: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IEventPublisher(ABC):
    """
    Interface for publishing exchange events.
    """

    @abstractmethod
    async def publish_event(self, event_data: ExchangeEventData) -> None:
        """Publish an event."""

    @abstractmethod
    async def subscribe_to_events(
        self,
        event_types: List[ExchangeEvent],
        callback: Callable[[ExchangeEventData], Awaitable[None]]
    ) -> str:
        """Subscribe to specific event types."""

    @abstractmethod
    async def unsubscribe_from_events(self, subscription_id: str) -> None:
        """Unsubscribe from events."""


class IExchangeManager(ABC):
    """
    Interface for managing multiple exchanges.
    """

    @abstractmethod
    async def add_exchange(self, config: ExchangeConfig) -> bool:
        """Add a new exchange."""

    @abstractmethod
    async def remove_exchange(self, exchange_name: str) -> bool:
        """Remove an exchange."""

    @abstractmethod
    async def get_exchange(self, exchange_name: str) -> Optional[IExchange]:
        """Get an exchange by name."""

    @abstractmethod
    async def get_all_exchanges(self) -> Dict[str, IExchange]:
        """Get all configured exchanges."""

    @abstractmethod
    async def get_exchange_metrics(self, exchange_name: str) -> ExchangeMetrics:
        """Get metrics for an exchange."""

    @abstractmethod
    async def get_all_metrics(self) -> Dict[str, ExchangeMetrics]:
        """Get metrics for all exchanges."""