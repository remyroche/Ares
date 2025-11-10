"""
Standardized Account Info Data Structure

Unified account information structure that all exchanges must conform to.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.tprint import tprint
from .standardized_balance import StandardizedBalance


@dataclass
class StandardizedAccountInfo:
    """
    Unified account information structure.
    
    This is the single source of truth for account data across the entire system.
    All exchanges must convert their account data to this exact format.
    """
    # Required fields
    exchange: str
    account_type: str               # SPOT/MARGIN/FUTURES
    can_trade: bool
    can_withdraw: bool
    can_deposit: bool
    timestamp: datetime
    
    # Optional fields
    permissions: List[str] = field(default_factory=list)  # Trading permissions
    balances: List[StandardizedBalance] = field(default_factory=list)  # Account balances
    total_equity: Optional[float] = None
    available_margin: Optional[float] = None
    used_margin: Optional[float] = None
    margin_ratio: Optional[float] = None
    
    # Exchange metadata
    raw_account_data: Optional[Dict[str, Any]] = None
    source_exchange_type: Optional[str] = None
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    def __post_init__(self):
        """Validate data after initialization"""
        tprint(f"StandardizedAccountInfo.__post_init__ called for exchange={self.exchange}, account_type={self.account_type}", "INFO")

        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc)
            tprint(f"Set default timestamp for account info: {self.timestamp}", "INFO")

        self._validate_data()
        tprint(f"Account info post-initialization complete, is_valid={self.is_valid}", "SUCCESS" if self.is_valid else "WARNING")
    
    def _validate_data(self) -> None:
        """Validate the account info data for consistency and quality"""
        tprint(f"Validating account info for exchange={self.exchange}", "INFO")
        errors = []

        # Validate required fields
        if not self.exchange or not isinstance(self.exchange, str):
            errors.append("exchange must be a non-empty string")
            tprint("Validation error: exchange must be a non-empty string", "ERROR")

        # Validate account type
        valid_types = ["SPOT", "MARGIN", "FUTURES", "OPTIONS", "UNIFIED"]
        if self.account_type.upper() not in valid_types:
            errors.append(f"account_type should be one of {valid_types}")
            tprint(f"Validation error: Invalid account_type={self.account_type}", "ERROR")

        if not isinstance(self.can_trade, bool):
            errors.append("can_trade must be a boolean")
            tprint("Validation error: can_trade must be a boolean", "ERROR")

        if not isinstance(self.can_withdraw, bool):
            errors.append("can_withdraw must be a boolean")
            tprint("Validation error: can_withdraw must be a boolean", "ERROR")

        if not isinstance(self.can_deposit, bool):
            errors.append("can_deposit must be a boolean")
            tprint("Validation error: can_deposit must be a boolean", "ERROR")

        # Validate margin ratio if margin fields are present
        if self.used_margin is not None and self.available_margin is not None:
            if self.used_margin > 0 and self.margin_ratio is None:
                # Could calculate it, but it's optional
                pass

        self.validation_errors = errors
        self.is_valid = len(errors) == 0

        if not self.is_valid:
            self.quality_score = max(0.0, self.quality_score - len(errors) * 10.0)
            tprint(f"Account info validation failed with {len(errors)} errors, quality_score={self.quality_score}", "ERROR")
        else:
            tprint(f"Account info validation successful, quality_score={self.quality_score}", "SUCCESS")
    
    def get_balance(self, currency: str) -> Optional[StandardizedBalance]:
        """Get balance for a specific currency"""
        tprint(f"Getting balance for currency={currency} from {len(self.balances)} balances", "INFO")

        for balance in self.balances:
            if balance.currency.upper() == currency.upper():
                tprint(f"Found balance for {currency}: free={balance.free}, used={balance.used}, total={balance.total}", "SUCCESS")
                return balance

        tprint(f"No balance found for currency={currency}", "WARNING")
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        tprint(f"Converting account info to dict for exchange={self.exchange}", "INFO")

        result = {
            'exchange': self.exchange,
            'account_type': self.account_type,
            'can_trade': self.can_trade,
            'can_withdraw': self.can_withdraw,
            'can_deposit': self.can_deposit,
            'permissions': self.permissions,
            'balances': [balance.to_dict() for balance in self.balances],
            'total_equity': self.total_equity,
            'available_margin': self.available_margin,
            'used_margin': self.used_margin,
            'margin_ratio': self.margin_ratio,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'source_exchange_type': self.source_exchange_type,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
            'quality_score': self.quality_score,
        }

        tprint(f"Account info converted to dict with {len(self.balances)} balances", "SUCCESS")
        return result
    
    def to_dataframe_row(self) -> Dict[str, Any]:
        """Convert to single-row dictionary for DataFrame creation"""
        return {
            'exchange': self.exchange,
            'account_type': self.account_type,
            'can_trade': self.can_trade,
            'can_withdraw': self.can_withdraw,
            'can_deposit': self.can_deposit,
            'total_equity': self.total_equity,
            'available_margin': self.available_margin,
            'used_margin': self.used_margin,
            'margin_ratio': self.margin_ratio,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
        }
    
    def __repr__(self) -> str:
        return (
            f"StandardizedAccountInfo("
            f"exchange={self.exchange}, "
            f"account_type={self.account_type}, "
            f"can_trade={self.can_trade}, "
            f"balance_count={len(self.balances)}"
            f")"
        )