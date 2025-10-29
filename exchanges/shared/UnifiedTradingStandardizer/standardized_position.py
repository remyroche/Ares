"""
Standardized Position Data Structure

Unified position structure that all exchanges must conform to.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from enum import Enum


PositionSide = Literal["long", "short", "neutral"]


@dataclass
class StandardizedPosition:
    """
    Unified position structure across all exchanges.
    
    This is the single source of truth for position data across the entire system.
    All exchanges must convert their position data to this exact format.
    """
    # Required fields
    symbol: str
    exchange: str
    side: str                        # "long"/"short"/"neutral"
    size: float                      # Position size (positive number)
    entry_price: float               # Average entry price
    timestamp: datetime
    update_time: datetime
    
    # Optional fields
    mark_price: Optional[float] = None       # Current mark price
    liquidation_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: Optional[float] = None          # 1x for spot, >1x for futures
    margin: Optional[float] = None           # Used margin
    isolated_margin: Optional[float] = None
    
    # Additional fields
    position_value: Optional[float] = None    # Position notional value
    margin_mode: Optional[str] = None        # ISOLATED/CROSSED
    position_mode: Optional[str] = None      # HEDGE/ONE_WAY
    
    # Exchange metadata
    exchange_position_id: Optional[str] = None
    raw_position_data: Optional[Dict[str, Any]] = None
    source_exchange_type: Optional[str] = None
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.update_time:
            self.update_time = datetime.now(timezone.utc)
        
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
        
        # Calculate unrealized PnL if mark_price is available
        if self.mark_price and self.entry_price:
            if self.side == "long":
                self.unrealized_pnl = (self.mark_price - self.entry_price) * self.size
            elif self.side == "short":
                self.unrealized_pnl = (self.entry_price - self.mark_price) * self.size
        
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate the position data for consistency and quality"""
        errors = []
        
        # Validate required fields
        if not self.symbol or not isinstance(self.symbol, str):
            errors.append("symbol must be a non-empty string")
        
        if not isinstance(self.size, (int, float)) or self.size < 0:
            errors.append("size must be a non-negative number")
        
        if not isinstance(self.entry_price, (int, float)) or self.entry_price <= 0:
            errors.append("entry_price must be a positive number")
        
        # Validate side
        valid_sides = ["long", "short", "neutral"]
        if self.side not in valid_sides:
            errors.append(f"side must be one of {valid_sides}, got {self.side}")
        
        # Validate leverage
        if self.leverage is not None and (not isinstance(self.leverage, (int, float)) or self.leverage < 1):
            errors.append("leverage must be >= 1 if provided")
        
        # Validate margin consistency
        if self.margin is not None and self.leverage and self.position_value:
            expected_margin = self.position_value / self.leverage
            if abs(self.margin - expected_margin) / expected_margin > 0.1:  # 10% tolerance
                errors.append("margin inconsistent with position_value and leverage")
        
        # Validate liquidation price
        if self.liquidation_price is not None and self.entry_price:
            if self.side == "long" and self.liquidation_price >= self.entry_price:
                errors.append("long position liquidation_price should be < entry_price")
            elif self.side == "short" and self.liquidation_price <= self.entry_price:
                errors.append("short position liquidation_price should be > entry_price")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        if not self.is_valid:
            self.quality_score = max(0.0, self.quality_score - len(errors) * 10.0)
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL based on current price"""
        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        elif self.side == "short":
            return (self.entry_price - current_price) * self.size
        else:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'mark_price': self.mark_price,
            'liquidation_price': self.liquidation_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'leverage': self.leverage,
            'margin': self.margin,
            'isolated_margin': self.isolated_margin,
            'position_value': self.position_value,
            'margin_mode': self.margin_mode,
            'position_mode': self.position_mode,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'update_time': self.update_time.isoformat() if isinstance(self.update_time, datetime) else str(self.update_time),
            'exchange_position_id': self.exchange_position_id,
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
            f"StandardizedPosition("
            f"symbol={self.symbol}, "
            f"side={self.side}, "
            f"size={self.size}, "
            f"entry_price={self.entry_price}, "
            f"unrealized_pnl={self.unrealized_pnl:.2f}"
            f")"
        )