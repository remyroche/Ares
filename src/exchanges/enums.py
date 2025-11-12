"""
Énumérations pour le module exchanges.

Ce module définit les énumérations utilisées dans tout le système d'échanges.
"""

from enum import Enum
from typing import Any, Dict, Optional


class ExchangeStatus(Enum):
    """Statuts possibles pour un exchange."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class OrderStatus(Enum):
    """Statuts possibles pour un ordre."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Côtés d'ordre."""
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    """Types de durée de validité d'ordre."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Valid for the day
    GTD = "GTD"  # Good Till Date


class SignalStatus(Enum):
    """Statuts possibles pour un signal de trading."""
    RECEIVED = "received"
    PROCESSED = "processed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ReceiverState(Enum):
    """États possibles pour le récepteur de trading."""
    STOPPED = "stopped"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


class DispatchResult(Enum):
    """Résultats possibles d'un dispatch."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    TIMEOUT = "timeout"


class RiskLevel(Enum):
    """Niveaux de risque."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSide(Enum):
    """Côtés de position."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


# Classes de données pour les structures complexes
class TradingSignal:
    """Représente un signal de trading."""
    
    def __init__(self,
                 symbol: str,
                 side: str,
                 order_type: str,
                 quantity: float,
                 price: float,
                 exchange: Optional[str] = None,
                 timestamp: Optional[Any] = None,
                 confidence: float = 0.0,
                 strategy: Optional[str] = None,
                 signal_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.exchange = exchange
        self.timestamp = timestamp
        self.confidence = confidence
        self.strategy = strategy
        self.signal_id = signal_id
        self.metadata = metadata or {}


class RoutedOrder:
    """Représente un ordre routé."""
    
    def __init__(self,
                 id: str,
                 exchange: str,
                 symbol: str,
                 side: str,
                 order_type: str,
                 quantity: float,
                 price: float,
                 status: OrderStatus,
                 exchange_order_id: Optional[str] = None,
                 timestamp: Optional[Any] = None,
                 filled_quantity: float = 0.0,
                 average_price: float = 0.0,
                 fees: float = 0.0):
        self.id = id
        self.exchange = exchange
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.status = status
        self.exchange_order_id = exchange_order_id
        self.timestamp = timestamp
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.fees = fees