"""
High-level wrapper classes for shared exchange utilities with comprehensive type hints (Part 2).

These classes provide consistent, high-level interfaces that abstract away
implementation details and provide uniform abstraction levels.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, TypeVar
from datetime import datetime
import logging

from .interfaces_typed import (
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
    DataSource, ValidationResult, tprint, handle_errors, handle_async_errors
)

# Type variables
T = TypeVar('T')

# Logger for error handling
logger = logging.getLogger(__name__)

# Import low-level managers with error handling
try:
    from .risk.risk_calculator import RiskCalculator, RiskLevel
    from .wallet.balance_manager import BalanceManager, AccountType
    from .reliability.rate_limit_manager import RateLimitManager, RateLimit
except ImportError as e:
    tprint(f"Failed to import low-level managers: {e}", "ERROR")
    raise


class HighLevelRiskManager(IHighLevelRiskManager):
    """High-level wrapper for risk management with comprehensive type hints."""
    
    def __init__(self, exchange_name: str) -> None:
        """Initialize the high-level risk manager."""
        try:
            self.exchange_name: str = exchange_name
            self.risk_calculator: RiskCalculator = RiskCalculator(exchange_name)
            self._initialized: bool = False
            tprint(f"Initialized HighLevelRiskManager for {exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize HighLevelRiskManager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def initialize(self) -> None:
        """Initialize the risk manager."""
        try:
            self._initialized = True
            tprint(f"Risk manager initialized for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize risk manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def close(self) -> None:
        """Close the risk manager."""
        try:
            self._initialized = False
            tprint(f"Risk manager closed for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to close risk manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={"initialized": False, "warning_ratio": 0.8, "critical_ratio": 0.9, "liquidation_ratio": 0.95})
    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status."""
        try:
            return {
                "initialized": self._initialized,
                "warning_ratio": self.risk_calculator.margin_ratio_warning,
                "critical_ratio": self.risk_calculator.margin_ratio_critical,
                "liquidation_ratio": self.risk_calculator.margin_ratio_liquidation
            }
        except Exception as e:
            tprint(f"Failed to get risk status: {e}", "ERROR")
            return {"initialized": False, "warning_ratio": 0.8, "critical_ratio": 0.9, "liquidation_ratio": 0.95}
    
    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset to initial state."""
        try:
            self.risk_calculator.set_risk_thresholds()
            self.risk_calculator.set_default_margins()
            tprint(f"Risk manager reset for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to reset risk manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={})
    def calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        leverage: float
    ) -> Dict[str, Any]:
        """Calculate position risk metrics."""
        try:
            position_risk = self.risk_calculator.calculate_position_risk(
                symbol=symbol,
                position_size=position_size,
                entry_price=current_price,  # Using current price as entry for simplicity
                current_price=current_price,
                leverage=leverage
            )
            
            return {
                "symbol": position_risk.symbol,
                "position_size": position_risk.position_size,
                "current_price": position_risk.current_price,
                "leverage": position_risk.leverage,
                "margin_used": position_risk.margin_used,
                "unrealized_pnl": position_risk.unrealized_pnl,
                "margin_ratio": position_risk.margin_ratio,
                "liquidation_price": position_risk.liquidation_price,
                "risk_level": position_risk.risk_level.value,
                "notional_value": position_risk.notional_value
            }
        except Exception as e:
            tprint(f"Failed to calculate position risk: {e}", "ERROR")
            return {}
    
    @handle_errors(default_return={})
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
        try:
            # Convert positions to PositionRisk objects
            position_risks = []
            for pos in positions:
                position_risk = self.risk_calculator.calculate_position_risk(
                    symbol=pos["symbol"],
                    position_size=pos["position_size"],
                    entry_price=pos.get("entry_price", pos["current_price"]),
                    current_price=pos["current_price"],
                    leverage=pos["leverage"]
                )
                position_risks.append(position_risk)
            
            # Calculate total equity (simplified)
            total_equity = sum(pos.get("equity", 10000) for pos in positions)
            
            portfolio_risk = self.risk_calculator.calculate_portfolio_risk(
                position_risks, total_equity
            )
            
            return {
                "total_equity": portfolio_risk.total_equity,
                "total_margin_used": portfolio_risk.total_margin_used,
                "total_unrealized_pnl": portfolio_risk.total_unrealized_pnl,
                "portfolio_margin_ratio": portfolio_risk.portfolio_margin_ratio,
                "risk_level": portfolio_risk.risk_level.value,
                "max_leverage_used": portfolio_risk.max_leverage_used,
                "total_notional": portfolio_risk.total_notional,
                "position_count": len(positions)
            }
        except Exception as e:
            tprint(f"Failed to calculate portfolio risk: {e}", "ERROR")
            return {}
    
    @handle_errors(default_return=ValidationResult(False, ["Risk validation failed"]))
    def validate_risk_limits(self, risk_data: Dict[str, Any]) -> ValidationResult:
        """Validate against risk limits."""
        try:
            result = ValidationResult(True)
            
            margin_ratio = risk_data.get("margin_ratio", 0)
            
            if margin_ratio >= self.risk_calculator.margin_ratio_liquidation:
                result.add_error(f"Margin ratio {margin_ratio:.2%} exceeds liquidation limit")
            elif margin_ratio >= self.risk_calculator.margin_ratio_critical:
                result.add_warning(f"Margin ratio {margin_ratio:.2%} is at critical level")
            elif margin_ratio >= self.risk_calculator.margin_ratio_warning:
                result.add_warning(f"Margin ratio {margin_ratio:.2%} is high")
            
            leverage = risk_data.get("leverage", 1)
            if leverage > 10:
                result.add_warning(f"High leverage: {leverage}x")
            
            return result
        except Exception as e:
            tprint(f"Failed to validate risk limits: {e}", "ERROR")
            return ValidationResult(False, [f"Risk validation error: {e}"])
    
    @handle_errors(default_return=0.0)
    def get_max_position_size(
        self,
        symbol: str,
        available_margin: float,
        risk_tolerance: float
    ) -> float:
        """Calculate maximum position size."""
        try:
            # Use current price as entry price for calculation
            current_price = 50000.0  # This should be fetched from market data
            leverage = 2.0  # Default leverage
            
            return self.risk_calculator.calculate_max_position_size(
                symbol=symbol,
                entry_price=current_price,
                current_price=current_price,
                leverage=leverage,
                available_margin=available_margin,
                risk_tolerance=risk_tolerance
            )
        except Exception as e:
            tprint(f"Failed to calculate max position size: {e}", "ERROR")
            return 0.0


class HighLevelBalanceManager(IHighLevelBalanceManager):
    """High-level wrapper for balance management with comprehensive type hints."""
    
    def __init__(self, exchange_name: str) -> None:
        """Initialize the high-level balance manager."""
        try:
            self.exchange_name: str = exchange_name
            self.balance_manager: BalanceManager = BalanceManager(exchange_name)
            self._initialized: bool = False
            tprint(f"Initialized HighLevelBalanceManager for {exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize HighLevelBalanceManager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def initialize(self) -> None:
        """Initialize the balance manager."""
        try:
            self._initialized = True
            tprint(f"Balance manager initialized for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize balance manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def close(self) -> None:
        """Close the balance manager."""
        try:
            self._initialized = False
            tprint(f"Balance manager closed for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to close balance manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={"initialized": False, "currencies_count": 0, "last_fetch": None})
    def get_status(self) -> Dict[str, Any]:
        """Get balance manager status."""
        try:
            return {
                "initialized": self._initialized,
                "currencies_count": len(self.balance_manager.balances),
                "last_fetch": self.balance_manager.last_fetch.isoformat() if self.balance_manager.last_fetch else None
            }
        except Exception as e:
            tprint(f"Failed to get balance status: {e}", "ERROR")
            return {"initialized": False, "currencies_count": 0, "last_fetch": None}
    
    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset to initial state."""
        try:
            self.balance_manager.balances.clear()
            self.balance_manager.account_equities.clear()
            self.balance_manager.last_fetch = None
            tprint(f"Balance manager reset for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to reset balance manager: {e}", "ERROR")
            raise
    
    @handle_async_errors(default_return=None)
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get balance data."""
        try:
            account_type = AccountType(key.split("_")[0]) if "_" in key else AccountType.SPOT
            currency = key.split("_")[1] if "_" in key else key
            
            return self.balance_manager.get_balance(currency, account_type)
        except Exception as e:
            tprint(f"Failed to get balance data for key '{key}': {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=False)
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh balance data from exchange."""
        try:
            if keys:
                # Refresh specific currencies/account types
                for key in keys:
                    account_type = AccountType(key.split("_")[0]) if "_" in key else AccountType.SPOT
                    await self.balance_manager.fetch_balances(account_type)
            else:
                # Refresh all account types
                for account_type in AccountType:
                    await self.balance_manager.fetch_balances(account_type)
            return True
        except Exception as e:
            tprint(f"Failed to refresh balance data: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=0)
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached balance data."""
        try:
            if keys:
                count = 0
                for key in keys:
                    if "_" in key:
                        account_type = AccountType(key.split("_")[0])
                        currency = key.split("_")[1]
                        self.balance_manager.invalidate_cache(currency, account_type)
                        count += 1
                return count
            else:
                count = len(self.balance_manager.balances)
                self.balance_manager.invalidate_cache()
                return count
        except Exception as e:
            tprint(f"Failed to invalidate balance cache: {e}", "ERROR")
            return 0
    
    @handle_errors(default_return={})
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        try:
            return self.balance_manager.get_balance_statistics()
        except Exception as e:
            tprint(f"Failed to get balance statistics: {e}", "ERROR")
            return {}
    
    @handle_async_errors(default_return=None)
    async def get_balance(self, currency: str, account_type: str = "spot") -> Optional[float]:
        """Get balance for currency."""
        try:
            account_type_enum = AccountType(account_type.lower())
            balance = self.balance_manager.get_balance(currency, account_type_enum)
            return balance.total if balance else None
        except ValueError as e:
            tprint(f"Invalid account type '{account_type}': {e}", "WARNING")
            return None
        except Exception as e:
            tprint(f"Failed to get balance for {currency}: {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return={})
    async def get_all_balances(self, account_type: str = "spot") -> Dict[str, float]:
        """Get all balances for account type."""
        try:
            account_type_enum = AccountType(account_type.lower())
            balances = self.balance_manager.get_all_balances(account_type_enum)
            return {balance.currency: balance.total for balance in balances}
        except ValueError as e:
            tprint(f"Invalid account type '{account_type}': {e}", "WARNING")
            return {}
        except Exception as e:
            tprint(f"Failed to get all balances: {e}", "ERROR")
            return {}
    
    @handle_errors(default_return=False)
    def has_sufficient_balance(
        self,
        currency: str,
        amount: float,
        account_type: str = "spot"
    ) -> bool:
        """Check if has sufficient balance."""
        try:
            account_type_enum = AccountType(account_type.lower())
            return self.balance_manager.has_sufficient_balance(currency, amount, account_type_enum)
        except ValueError as e:
            tprint(f"Invalid account type '{account_type}': {e}", "WARNING")
            return False
        except Exception as e:
            tprint(f"Failed to check sufficient balance: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=0.0)
    def calculate_portfolio_value(
        self,
        prices: Dict[str, float],
        base_currency: str = "USDT"
    ) -> float:
        """Calculate total portfolio value."""
        try:
            account_type_enum = AccountType.SPOT
            return self.balance_manager.calculate_portfolio_value(account_type_enum, prices, base_currency)
        except Exception as e:
            tprint(f"Failed to calculate portfolio value: {e}", "ERROR")
            return 0.0


class HighLevelRateLimitManager(IHighLevelRateLimitManager):
    """High-level wrapper for rate limiting with comprehensive type hints."""
    
    def __init__(self, exchange_name: str) -> None:
        """Initialize the high-level rate limit manager."""
        try:
            self.exchange_name: str = exchange_name
            self.rate_limit_manager: RateLimitManager = RateLimitManager(exchange_name)
            self._initialized: bool = False
            tprint(f"Initialized HighLevelRateLimitManager for {exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize HighLevelRateLimitManager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def initialize(self) -> None:
        """Initialize the rate limit manager."""
        try:
            self._initialized = True
            tprint(f"Rate limit manager initialized for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize rate limit manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def close(self) -> None:
        """Close the rate limit manager."""
        try:
            self._initialized = False
            tprint(f"Rate limit manager closed for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to close rate limit manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={"initialized": False, "configured_limits": 0, "total_requests": 0})
    def get_status(self) -> Dict[str, Any]:
        """Get rate limit manager status."""
        try:
            return {
                "initialized": self._initialized,
                "configured_limits": len(self.rate_limit_manager.rate_limits),
                "total_requests": len(self.rate_limit_manager.request_history)
            }
        except Exception as e:
            tprint(f"Failed to get rate limit status: {e}", "ERROR")
            return {"initialized": False, "configured_limits": 0, "total_requests": 0}
    
    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset to initial state."""
        try:
            self.rate_limit_manager.reset_request_history()
            tprint(f"Rate limit manager reset for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to reset rate limit manager: {e}", "ERROR")
            raise
    
    @handle_async_errors(default_return=None)
    async def execute_with_limits(
        self,
        operation: str,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Execute function with rate limiting."""
        try:
            return await self.rate_limit_manager.execute_with_rate_limit(operation, func, *args, **kwargs)
        except Exception as e:
            tprint(f"Failed to execute with rate limits: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def set_limits(self, operation: str, limits: Dict[str, int]) -> None:
        """Set rate limits for operation."""
        try:
            rate_limit = RateLimit(
                requests_per_second=limits.get("per_second", 10),
                requests_per_minute=limits.get("per_minute", 600),
                requests_per_hour=limits.get("per_hour", 36000),
                burst_limit=limits.get("burst", 20)
            )
            self.rate_limit_manager.set_rate_limit(operation, rate_limit)
            tprint(f"Set rate limits for {operation}: {limits}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to set rate limits for {operation}: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=0)
    def get_remaining_requests(self, operation: str) -> int:
        """Get remaining requests for operation."""
        try:
            status = self.rate_limit_manager.get_rate_limit_status(operation)
            return status["remaining_requests"]
        except Exception as e:
            tprint(f"Failed to get remaining requests for {operation}: {e}", "ERROR")
            return 0
    
    @handle_errors(default_return=False)
    def is_limited(self, operation: str) -> bool:
        """Check if operation is currently limited."""
        try:
            status = self.rate_limit_manager.get_rate_limit_status(operation)
            return status["is_limited"]
        except Exception as e:
            tprint(f"Failed to check if {operation} is limited: {e}", "ERROR")
            return False