"""
Standardized Trade Data Structure

Unified trade/transaction structure that all exchanges must conform to.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

from src.utils.tprint import tprint
from exchanges.base_exchange.exchange_interface import OrderSide


@dataclass
class StandardizedTrade:
    """
    Unified trade/transaction structure.
    
    This is the single source of truth for trade data across the entire system.
    All exchanges must convert their trade data to this exact format.
    """
    # Required fields
    trade_id: str                   # Exchange-specific trade ID
    order_id: str                   # Parent order ID
    symbol: str
    exchange: str
    side: OrderSide                 # BUY/SELL
    price: float
    quantity: float
    timestamp: datetime
    fee: float
    fee_currency: str
    
    # Optional fields
    is_maker: Optional[bool] = None         # Maker/taker flag
    is_buyer: Optional[bool] = None         # Buyer/seller flag
    trade_type: Optional[str] = None        # TRADE/FUNDING_FEE/etc
    
    # Exchange metadata
    raw_trade_data: Optional[Dict[str, Any]] = None
    source_exchange_type: Optional[str] = None
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        tprint(f"StandardizedTrade.__post_init__ called for trade_id={self.trade_id}, symbol={self.symbol}, side={self.side}", "INFO")

        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
            tprint(f"Set default timestamp: {self.timestamp}", "INFO")

        self._validate_data()
        tprint(f"Trade post-initialization complete for {self.trade_id}, is_valid={self.is_valid}", "SUCCESS" if self.is_valid else "WARNING")
    
    def _validate_data(self) -> None:
        """Validate the trade data for consistency and quality"""
        tprint(f"Validating trade: trade_id={self.trade_id}, symbol={self.symbol}, side={self.side}", "INFO")
        errors = []

        # Validate required fields
        if not self.trade_id or not isinstance(self.trade_id, str):
            errors.append("trade_id must be a non-empty string")
            tprint("Validation error: trade_id must be a non-empty string", "ERROR")

        if not self.order_id or not isinstance(self.order_id, str):
            errors.append("order_id must be a non-empty string")
            tprint("Validation error: order_id must be a non-empty string", "ERROR")

        if not self.symbol or not isinstance(self.symbol, str):
            errors.append("symbol must be a non-empty string")
            tprint("Validation error: symbol must be a non-empty string", "ERROR")

        if not isinstance(self.price, (int, float)) or self.price <= 0:
            errors.append("price must be a positive number")
            tprint(f"Validation error: price={self.price} must be positive", "ERROR")

        if not isinstance(self.quantity, (int, float)) or self.quantity <= 0:
            errors.append("quantity must be a positive number")
            tprint(f"Validation error: quantity={self.quantity} must be positive", "ERROR")

        if not isinstance(self.fee, (int, float)) or self.fee < 0:
            errors.append("fee must be a non-negative number")
            tprint(f"Validation error: fee={self.fee} must be non-negative", "ERROR")

        if not self.fee_currency or not isinstance(self.fee_currency, str):
            errors.append("fee_currency must be a non-empty string")
            tprint("Validation error: fee_currency must be a non-empty string", "ERROR")

        # Validate side consistency with is_buyer
        if self.is_buyer is not None:
            if self.side == OrderSide.BUY and not self.is_buyer:
                errors.append("BUY side should have is_buyer=True")
                tprint("Validation error: BUY side inconsistent with is_buyer=False", "ERROR")
            elif self.side == OrderSide.SELL and self.is_buyer:
                errors.append("SELL side should have is_buyer=False")
                tprint("Validation error: SELL side inconsistent with is_buyer=True", "ERROR")

        self.validation_errors = errors
        self.is_valid = len(errors) == 0

        if not self.is_valid:
            self.quality_score = max(0.0, self.quality_score - len(errors) * 10.0)
            tprint(f"Trade validation failed for {self.trade_id} with {len(errors)} errors, quality_score={self.quality_score}", "ERROR")
        else:
            tprint(f"Trade validation successful for {self.trade_id}: {self.symbol} {self.side.value} {self.quantity}@{self.price}, fee={self.fee}", "SUCCESS")
    
    def get_total_value(self) -> float:
        """Calculate total trade value (price * quantity)"""
        total_value = self.price * self.quantity
        tprint(f"Calculated total trade value for {self.trade_id}: {total_value}", "INFO")
        return total_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        tprint(f"Converting trade to dict: trade_id={self.trade_id}, symbol={self.symbol}", "INFO")

        result = {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side.value if isinstance(self.side, Enum) else str(self.side),
            'price': self.price,
            'quantity': self.quantity,
            'fee': self.fee,
            'fee_currency': self.fee_currency,
            'is_maker': self.is_maker,
            'is_buyer': self.is_buyer,
            'trade_type': self.trade_type,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'source_exchange_type': self.source_exchange_type,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'quality_score': self.quality_score,
        }

        tprint(f"Trade converted to dict: {self.trade_id} {self.quantity}@{self.price}", "SUCCESS")
        return result
    
    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert to single-row dictionary for DataFrame creation"""
        return self.to_dict()
    
    def __repr__(self) -> str:
        return (
            f"StandardizedTrade("
            f"trade_id={self.trade_id}, "
            f"order_id={self.order_id}, "
            f"symbol={self.symbol}, "
            f"side={self.side.value if isinstance(self.side, Enum) else self.side}, "
            f"price={self.price}, "
            f"quantity={self.quantity}, "
            f"fee={self.fee}"
            f")"
        )