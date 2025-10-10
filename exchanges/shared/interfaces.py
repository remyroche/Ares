"""
High-level interfaces for shared exchange utilities.

This module provides high-level interfaces that abstract away implementation details
and provide consistent abstraction levels across all shared utilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum


class DataSource(Enum):
    """Data source enumeration for consistent data fetching."""
    CACHE = "cache"
    EXCHANGE = "exchange"
    FALLBACK = "fallback"


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)


class IDataManager(ABC):
    """High-level interface for data management operations."""
    
    @abstractmethod
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get data with automatic source selection."""
    
    @abstractmethod
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh data from exchange."""
    
    @abstractmethod
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached data."""
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""


class IValidationManager(ABC):
    """High-level interface for validation operations."""
    
    @abstractmethod
    def validate_data(self, data: Any, data_type: str) -> ValidationResult:
        """Validate data according to type-specific rules."""
    
    @abstractmethod
    def validate_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate API request parameters."""
    
    @abstractmethod
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate API response data."""


class IConfigurationManager(ABC):
    """High-level interface for configuration management."""
    
    @abstractmethod
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
    
    @abstractmethod
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
    
    @abstractmethod
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration."""
    
    @abstractmethod
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""


class IErrorHandler(ABC):
    """High-level interface for error handling."""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle an error and return whether it was recoverable."""
    
    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried."""
    
    @abstractmethod
    def get_retry_delay(self, error: Exception, attempt: int) -> float:
        """Get delay before retry."""


class IPerformanceMonitor(ABC):
    """High-level interface for performance monitoring."""
    
    @abstractmethod
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record operation performance."""
    
    @abstractmethod
    def get_performance_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
    
    @abstractmethod
    def is_performance_acceptable(self, operation: str) -> bool:
        """Check if performance is within acceptable limits."""


class IExchangeUtility(ABC):
    """Base interface for all exchange utilities."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the utility."""
    
    @abstractmethod
    def close(self) -> None:
        """Close the utility and cleanup resources."""
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get utility status."""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset utility to initial state."""


class IHighLevelAuthManager(IExchangeUtility):
    """High-level authentication management interface."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with exchange using credentials."""
    
    @abstractmethod
    async def reauthenticate(self) -> bool:
        """Re-authenticate if needed."""
    
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
    
    @abstractmethod
    def get_auth_headers(self, request_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Get authentication headers for request."""
    
    @abstractmethod
    def has_permission(self, permission: str) -> bool:
        """Check if has specific permission."""


class IHighLevelMarketManager(IExchangeUtility, IDataManager):
    """High-level market data management interface."""
    
    @abstractmethod
    async def get_instrument_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get instrument information."""
    
    @abstractmethod
    async def get_price(self, symbol: str, source: DataSource = DataSource.CACHE) -> Optional[float]:
        """Get current price for symbol."""
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data."""
    
    @abstractmethod
    def is_symbol_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
    
    @abstractmethod
    def search_instruments(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search instruments with filters."""


class IHighLevelOrderManager(IExchangeUtility, IValidationManager):
    """High-level order management interface."""
    
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        **kwargs
    ) -> Optional[str]:
        """Create and submit order, return order ID."""
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
    
    @abstractmethod
    def validate_order_params(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate order parameters."""


class IHighLevelRiskManager(IExchangeUtility):
    """High-level risk management interface."""
    
    @abstractmethod
    def calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        leverage: float
    ) -> Dict[str, Any]:
        """Calculate position risk metrics."""
    
    @abstractmethod
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
    
    @abstractmethod
    def validate_risk_limits(self, risk_data: Dict[str, Any]) -> ValidationResult:
        """Validate against risk limits."""
    
    @abstractmethod
    def get_max_position_size(
        self,
        symbol: str,
        available_margin: float,
        risk_tolerance: float
    ) -> float:
        """Calculate maximum position size."""


class IHighLevelBalanceManager(IExchangeUtility, IDataManager):
    """High-level balance management interface."""
    
    @abstractmethod
    async def get_balance(self, currency: str, account_type: str = "spot") -> Optional[float]:
        """Get balance for currency."""
    
    @abstractmethod
    async def get_all_balances(self, account_type: str = "spot") -> Dict[str, float]:
        """Get all balances for account type."""
    
    @abstractmethod
    def has_sufficient_balance(
        self,
        currency: str,
        amount: float,
        account_type: str = "spot"
    ) -> bool:
        """Check if has sufficient balance."""
    
    @abstractmethod
    def calculate_portfolio_value(
        self,
        prices: Dict[str, float],
        base_currency: str = "USDT"
    ) -> float:
        """Calculate total portfolio value."""


class IHighLevelRateLimitManager(IExchangeUtility):
    """High-level rate limiting interface."""
    
    @abstractmethod
    async def execute_with_limits(
        self,
        operation: str,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with rate limiting."""
    
    @abstractmethod
    def set_limits(self, operation: str, limits: Dict[str, int]) -> None:
        """Set rate limits for operation."""
    
    @abstractmethod
    def get_remaining_requests(self, operation: str) -> int:
        """Get remaining requests for operation."""
    
    @abstractmethod
    def is_limited(self, operation: str) -> bool:
        """Check if operation is currently limited."""