"""
Standardized Order Data Structure

Unified order structure that all exchanges must conform to.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

from exchanges.base_exchange.exchange_interface import OrderSide, OrderType, OrderStatus


@dataclass
class StandardizedOrder:
    """
    Unified order structure across all exchanges.
    
    This is the single source of truth for order data across the entire system.
    All exchanges must convert their order data to this exact format.
    """
    # Required fields
    order_id: str                    # Exchange-specific order ID
    symbol: str
    exchange: str
    side: OrderSide                  # BUY/SELL
    order_type: OrderType            # MARKET/LIMIT/STOP/etc
    status: OrderStatus              # PENDING/PARTIALLY_FILLED/FILLED/CANCELED/REJECTED
    quantity: float
    timestamp: datetime              # Order creation time
    update_time: datetime            # Last update time
    
    # Optional core fields
    client_order_id: Optional[str] = None   # Client-provided order ID
    price: Optional[float] = None           # None for market orders
    executed_quantity: float = 0.0          # Filled quantity
    remaining_quantity: Optional[float] = None  # Remaining quantity
    executed_price_avg: Optional[float] = None    # Average fill price
    
    # Order parameters
    stop_price: Optional[float] = None      # For stop orders
    time_in_force: Optional[str] = None     # GTC/IOC/FOK
    fee: Optional[float] = None
    fee_currency: Optional[str] = None
    
    # Exchange metadata
    raw_order_data: Optional[Dict[str, Any]] = None  # Original response
    exchange_order_id: Optional[str] = None         # Exchange-specific identifier
    source_exchange_type: Optional[str] = None      # ExchangeType enum value
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity - self.executed_quantity
        
        if not self.update_time:
            self.update_time = datetime.now(timezone.utc)
        
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
        
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate the order data for consistency and quality"""
        errors = []
        
        # Validate required fields
        if not self.order_id or not isinstance(self.order_id, str):
            errors.append("order_id must be a non-empty string")
        
        if not self.symbol or not isinstance(self.symbol, str):
            errors.append("symbol must be a non-empty string")
        
        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            errors.append("quantity must be a positive number")
        
        if self.price is not None and (not isinstance(self.price, (int, float)) or self.price <= 0):
            errors.append("price must be a positive number if provided")
        
        if not isinstance(self.executed_quantity, (int, float)) or self.executed_quantity < 0:
            errors.append("executed_quantity must be a non-negative number")
        
        if self.executed_quantity > self.quantity:
            errors.append("executed_quantity cannot exceed quantity")
        
        if self.remaining_quantity is not None and self.remaining_quantity < 0:
            errors.append("remaining_quantity cannot be negative")
        
        # Validate status consistency
        if self.status == OrderStatus.FILLED and self.executed_quantity < self.quantity:
            errors.append("FILLED status requires executed_quantity == quantity")
        
        if self.status == OrderStatus.PARTIALLY_FILLED and (
            self.executed_quantity == 0 or self.executed_quantity >= self.quantity
        ):
            errors.append("PARTIALLY_FILLED status requires 0 < executed_quantity < quantity")
        
        # Validate order type requirements
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            errors.append(f"{self.order_type.value} orders require a price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            errors.append(f"{self.order_type.value} orders require a stop_price")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        if not self.is_valid:
            self.quality_score = max(0.0, self.quality_score - len(errors) * 10.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side.value if isinstance(self.side, Enum) else str(self.side),
            'order_type': self.order_type.value if isinstance(self.order_type, Enum) else str(self.order_type),
            'status': self.status.value if isinstance(self.status, Enum) else str(self.status),
            'quantity': self.quantity,
            'price': self.price,
            'executed_quantity': self.executed_quantity,
            'remaining_quantity': self.remaining_quantity,
            'executed_price_avg': self.executed_price_avg,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'fee': self.fee,
            'fee_currency': self.fee_currency,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'update_time': self.update_time.isoformat() if isinstance(self.update_time, datetime) else str(self.update_time),
            'exchange_order_id': self.exchange_order_id,
            'source_exchange_type': self.source_exchange_type,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'quality_score': self.quality_score,
        }
    
    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert to single-row dictionary for DataFrame creation"""
        return self.to_dict()
    
    def __repr__(self) -> str:
        return (
            f"StandardizedOrder("
            f"order_id={self.order_id}, "
            f"symbol={self.symbol}, "
            f"side={self.side.value if isinstance(self.side, Enum) else self.side}, "
            f"type={self.order_type.value if isinstance(self.order_type, Enum) else self.order_type}, "
            f"status={self.status.value if isinstance(self.status, Enum) else self.status}, "
            f"quantity={self.quantity}, "
            f"executed={self.executed_quantity}/{self.quantity}"
            f")"
        )