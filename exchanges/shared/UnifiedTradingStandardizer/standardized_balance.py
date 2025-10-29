"""
Standardized Balance Data Structure

Unified balance structure that all exchanges must conform to.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class StandardizedBalance:
    """
    Unified balance structure across all exchanges.
    
    This is the single source of truth for balance data across the entire system.
    All exchanges must convert their balance data to this exact format.
    """
    # Required fields
    currency: str
    exchange: str
    free: float                     # Available balance
    used: float                     # Locked/in-use balance
    total: float                    # Total balance (free + used)
    timestamp: datetime
    
    # Optional fields
    available_balance: Optional[float] = None   # Exchange-specific available
    frozen_balance: Optional[float] = None      # Frozen/locked balance
    account_type: Optional[str] = None           # SPOT/MARGIN/FUTURES
    
    # Exchange metadata
    raw_balance_data: Optional[Dict[str, Any]] = None
    source_exchange_type: Optional[str] = None
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
        
        # Ensure total is consistent
        if self.total != (self.free + self.used):
            if abs(self.total - (self.free + self.used)) > 1e-8:  # Floating point tolerance
                self.total = self.free + self.used
        
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate the balance data for consistency and quality"""
        errors = []
        
        # Validate required fields
        if not self.currency or not isinstance(self.currency, str):
            errors.append("currency must be a non-empty string")
        
        if not isinstance(self.free, (int, float)) or self.free < 0:
            errors.append("free must be a non-negative number")
        
        if not isinstance(self.used, (int, float)) or self.used < 0:
            errors.append("used must be a non-negative number")
        
        if not isinstance(self.total, (int, float)) or self.total < 0:
            errors.append("total must be a non-negative number")
        
        # Validate consistency
        calculated_total = self.free + self.used
        if abs(self.total - calculated_total) > 1e-6:
            errors.append(f"total ({self.total}) should equal free + used ({calculated_total})")
        
        # Validate account type
        if self.account_type is not None:
            valid_types = ["SPOT", "MARGIN", "FUTURES", "OPTIONS"]
            if self.account_type.upper() not in valid_types:
                errors.append(f"account_type should be one of {valid_types}")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        if not self.is_valid:
            self.quality_score = max(0.0, self.quality_score - len(errors) * 10.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'currency': self.currency,
            'exchange': self.exchange,
            'free': self.free,
            'used': self.used,
            'total': self.total,
            'available_balance': self.available_balance,
            'frozen_balance': self.frozen_balance,
            'account_type': self.account_type,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
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
            f"StandardizedBalance("
            f"currency={self.currency}, "
            f"free={self.free}, "
            f"used={self.used}, "
            f"total={self.total}"
            f")"
        )