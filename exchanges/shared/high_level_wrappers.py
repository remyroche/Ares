"""
High-level wrapper classes for shared exchange utilities.

These classes provide consistent, high-level interfaces that abstract away
implementation details and provide uniform abstraction levels.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .interfaces import (
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
    DataSource, ValidationResult
)
from .auth.auth_manager import AuthenticationManager, AuthConfig, APIKeyPermission
from .market.market_metadata import MarketMetadataManager, InstrumentType
from .orders.order_manager import OrderManager, OrderSide, OrderType
from .risk.risk_calculator import RiskCalculator, RiskLevel
from .wallet.balance_manager import BalanceManager, AccountType
from .reliability.rate_limit_manager import RateLimitManager, RateLimit


class HighLevelAuthManager(IHighLevelAuthManager):
    """High-level wrapper for authentication management."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.auth_manager = AuthenticationManager(exchange_name)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the authentication manager."""
        self._initialized = True
    
    def close(self) -> None:
        """Close the authentication manager."""
        self.auth_manager.logout()
        self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get authentication status."""
        return {
            "initialized": self._initialized,
            "authenticated": self.auth_manager.is_authenticated,
            "permissions": list(self.auth_manager.get_current_permissions()),
            "time_synced": self.auth_manager.is_time_synced()
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.auth_manager.logout()
        self._initialized = False
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with exchange using credentials."""
        if not self._initialized:
            self.initialize()
        
        # Convert credentials to AuthConfig
        permissions = set()
        if credentials.get("permissions"):
            for perm in credentials["permissions"]:
                try:
                    permissions.add(APIKeyPermission(perm))
                except ValueError:
                    continue
        
        auth_config = AuthConfig(
            exchange_name=self.exchange_name,
            api_key=credentials["api_key"],
            api_secret=credentials["api_secret"],
            passphrase=credentials.get("passphrase"),
            permissions=permissions or {APIKeyPermission.READ},
            auto_sync_time=credentials.get("auto_sync_time", True)
        )
        
        return await self.auth_manager.authenticate(auth_config)
    
    async def reauthenticate(self) -> bool:
        """Re-authenticate if needed."""
        return await self.auth_manager.reauthenticate()
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self.auth_manager.is_authenticated_and_valid()
    
    def get_auth_headers(self, request_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Get authentication headers for request."""
        return self.auth_manager.get_auth_headers(
            method=request_data.get("method", "GET"),
            endpoint=request_data.get("endpoint", ""),
            body=request_data.get("body", ""),
            additional_headers=request_data.get("additional_headers")
        )
    
    def has_permission(self, permission: str) -> bool:
        """Check if has specific permission."""
        try:
            perm = APIKeyPermission(permission)
            return self.auth_manager.has_permission(perm)
        except ValueError:
            return False


class HighLevelMarketManager(IHighLevelMarketManager):
    """High-level wrapper for market data management."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.market_manager = MarketMetadataManager(exchange_name)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the market manager."""
        self._initialized = True
    
    def close(self) -> None:
        """Close the market manager."""
        self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get market manager status."""
        return {
            "initialized": self._initialized,
            "instruments_count": len(self.market_manager.instruments),
            "last_refresh": self.market_manager.last_refresh.isoformat() if self.market_manager.last_refresh else None
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.market_manager.instruments.clear()
        self.market_manager.market_data.clear()
        self.market_manager.last_refresh = None
    
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get data with automatic source selection."""
        if source == DataSource.CACHE and not force_refresh:
            return self.market_manager.get_instrument(key)
        elif source == DataSource.EXCHANGE or force_refresh:
            await self.refresh_data([key])
            return self.market_manager.get_instrument(key)
        return None
    
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh data from exchange."""
        if keys:
            return await self.market_manager.refresh_market_data(keys)
        else:
            return await self.market_manager.refresh_instruments()
    
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached data."""
        if keys:
            count = 0
            for key in keys:
                if key in self.market_manager.instruments:
                    del self.market_manager.instruments[key]
                    count += 1
            return count
        else:
            count = len(self.market_manager.instruments)
            self.market_manager.instruments.clear()
            return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return self.market_manager.get_statistics()
    
    async def get_instrument_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get instrument information."""
        instrument = self.market_manager.get_instrument(symbol)
        if instrument:
            return {
                "symbol": instrument.symbol,
                "base_currency": instrument.base_currency,
                "quote_currency": instrument.quote_currency,
                "type": instrument.instrument_type.value,
                "tick_size": instrument.tick_size,
                "lot_size": instrument.lot_size,
                "min_notional": instrument.min_notional,
                "max_leverage": instrument.max_leverage,
                "is_tradable": self.is_symbol_tradable(symbol)
            }
        return None
    
    async def get_price(self, symbol: str, source: DataSource = DataSource.CACHE) -> Optional[float]:
        """Get current price for symbol."""
        market_data = self.market_manager.get_market_data(symbol)
        if market_data and "ticker" in market_data:
            ticker = market_data["ticker"]
            return ticker.get("last") or ticker.get("close") or ticker.get("price")
        return None
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data."""
        return self.market_manager.get_market_data(symbol)
    
    def is_symbol_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        return self.market_manager.is_symbol_tradable(symbol)
    
    def search_instruments(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search instruments with filters."""
        instruments = self.market_manager.search_instruments(
            base_currency=filters.get("base_currency"),
            quote_currency=filters.get("quote_currency"),
            instrument_type=InstrumentType(filters["type"]) if filters.get("type") else None,
            min_leverage=filters.get("min_leverage"),
            max_leverage=filters.get("max_leverage")
        )
        
        return [
            {
                "symbol": inst.symbol,
                "base_currency": inst.base_currency,
                "quote_currency": inst.quote_currency,
                "type": inst.instrument_type.value,
                "tick_size": inst.tick_size,
                "lot_size": inst.lot_size,
                "min_notional": inst.min_notional,
                "max_leverage": inst.max_leverage
            }
            for inst in instruments
        ]


class HighLevelOrderManager(IHighLevelOrderManager):
    """High-level wrapper for order management."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.order_manager = OrderManager(exchange_name)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the order manager."""
        self._initialized = True
    
    def close(self) -> None:
        """Close the order manager."""
        self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get order manager status."""
        return {
            "initialized": self._initialized,
            "total_orders": len(self.order_manager.orders),
            "open_orders": len(self.order_manager.get_open_orders())
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.order_manager.orders.clear()
        self.order_manager.orders_by_symbol.clear()
        self.order_manager.orders_by_status.clear()
    
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get order data."""
        return self.order_manager.get_order(key)
    
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh order data from exchange."""
        return await self.order_manager.sync_orders_from_exchange() > 0
    
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached order data."""
        if keys:
            count = 0
            for key in keys:
                if key in self.order_manager.orders:
                    self.order_manager._remove_order(key)
                    count += 1
            return count
        else:
            count = len(self.order_manager.orders)
            self.order_manager.orders.clear()
            self.order_manager.orders_by_symbol.clear()
            self.order_manager.orders_by_status.clear()
            return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return self.order_manager.get_order_statistics()
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        **kwargs
    ) -> Optional[str]:
        """Create and submit order, return order ID."""
        try:
            # Convert string parameters to enums
            side_enum = OrderSide(side.lower())
            type_enum = OrderType(order_type.lower())
            
            # Create order
            order = self.order_manager.create_order(
                symbol=symbol,
                side=side_enum,
                order_type=type_enum,
                quantity=quantity,
                price=kwargs.get("price"),
                stop_price=kwargs.get("stop_price"),
                client_order_id=kwargs.get("client_order_id"),
                metadata=kwargs.get("metadata")
            )
            
            # Submit order
            success = await self.order_manager.submit_order(order)
            if success:
                return order.order_id
            return None
            
        except (ValueError, KeyError) as e:
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        return await self.order_manager.cancel_order(order_id)
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        order = self.order_manager.get_order(order_id)
        if order:
            return {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.remaining_quantity,
                "average_price": order.average_price,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "updated_at": order.updated_at.isoformat() if order.updated_at else None
            }
        return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        if symbol:
            orders = self.order_manager.get_orders_by_symbol(symbol)
        else:
            orders = self.order_manager.get_open_orders()
        
        return [
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.remaining_quantity
            }
            for order in orders
        ]
    
    def validate_order_params(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate order parameters."""
        result = ValidationResult(True)
        
        # Required fields
        required_fields = ["symbol", "side", "order_type", "quantity"]
        for field in required_fields:
            if field not in params:
                result.add_error(f"Missing required field: {field}")
        
        # Validate side
        if "side" in params:
            try:
                OrderSide(params["side"].lower())
            except ValueError:
                result.add_error(f"Invalid side: {params['side']}")
        
        # Validate order type
        if "order_type" in params:
            try:
                OrderType(params["order_type"].lower())
            except ValueError:
                result.add_error(f"Invalid order type: {params['order_type']}")
        
        # Validate quantity
        if "quantity" in params:
            try:
                qty = float(params["quantity"])
                if qty <= 0:
                    result.add_error("Quantity must be positive")
            except (ValueError, TypeError):
                result.add_error("Invalid quantity format")
        
        # Validate price for limit orders
        if params.get("order_type", "").lower() == "limit":
            if "price" not in params:
                result.add_error("Price required for limit orders")
            elif params.get("price") is not None:
                try:
                    price = float(params["price"])
                    if price <= 0:
                        result.add_error("Price must be positive")
                except (ValueError, TypeError):
                    result.add_error("Invalid price format")
        
        return result


class HighLevelRiskManager(IHighLevelRiskManager):
    """High-level wrapper for risk management."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.risk_calculator = RiskCalculator(exchange_name)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the risk manager."""
        self._initialized = True
    
    def close(self) -> None:
        """Close the risk manager."""
        self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status."""
        return {
            "initialized": self._initialized,
            "warning_ratio": self.risk_calculator.margin_ratio_warning,
            "critical_ratio": self.risk_calculator.margin_ratio_critical,
            "liquidation_ratio": self.risk_calculator.margin_ratio_liquidation
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.risk_calculator.set_risk_thresholds()
        self.risk_calculator.set_default_margins()
    
    def calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        leverage: float
    ) -> Dict[str, Any]:
        """Calculate position risk metrics."""
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
    
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics."""
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
    
    def validate_risk_limits(self, risk_data: Dict[str, Any]) -> ValidationResult:
        """Validate against risk limits."""
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
    
    def get_max_position_size(
        self,
        symbol: str,
        available_margin: float,
        risk_tolerance: float
    ) -> float:
        """Calculate maximum position size."""
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


class HighLevelBalanceManager(IHighLevelBalanceManager):
    """High-level wrapper for balance management."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.balance_manager = BalanceManager(exchange_name)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the balance manager."""
        self._initialized = True
    
    def close(self) -> None:
        """Close the balance manager."""
        self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get balance manager status."""
        return {
            "initialized": self._initialized,
            "currencies_count": len(self.balance_manager.balances),
            "last_fetch": self.balance_manager.last_fetch.isoformat() if self.balance_manager.last_fetch else None
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.balance_manager.balances.clear()
        self.balance_manager.account_equities.clear()
        self.balance_manager.last_fetch = None
    
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get balance data."""
        account_type = AccountType(key.split("_")[0]) if "_" in key else AccountType.SPOT
        currency = key.split("_")[1] if "_" in key else key
        
        return self.balance_manager.get_balance(currency, account_type)
    
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh balance data from exchange."""
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
    
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached balance data."""
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return self.balance_manager.get_balance_statistics()
    
    async def get_balance(self, currency: str, account_type: str = "spot") -> Optional[float]:
        """Get balance for currency."""
        try:
            account_type_enum = AccountType(account_type.lower())
            balance = self.balance_manager.get_balance(currency, account_type_enum)
            return balance.total if balance else None
        except ValueError:
            return None
    
    async def get_all_balances(self, account_type: str = "spot") -> Dict[str, float]:
        """Get all balances for account type."""
        try:
            account_type_enum = AccountType(account_type.lower())
            balances = self.balance_manager.get_all_balances(account_type_enum)
            return {balance.currency: balance.total for balance in balances}
        except ValueError:
            return {}
    
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
        except ValueError:
            return False
    
    def calculate_portfolio_value(
        self,
        prices: Dict[str, float],
        base_currency: str = "USDT"
    ) -> float:
        """Calculate total portfolio value."""
        try:
            account_type_enum = AccountType.SPOT
            return self.balance_manager.calculate_portfolio_value(account_type_enum, prices, base_currency)
        except Exception:
            return 0.0


class HighLevelRateLimitManager(IHighLevelRateLimitManager):
    """High-level wrapper for rate limiting."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.rate_limit_manager = RateLimitManager(exchange_name)
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the rate limit manager."""
        self._initialized = True
    
    def close(self) -> None:
        """Close the rate limit manager."""
        self._initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limit manager status."""
        return {
            "initialized": self._initialized,
            "configured_limits": len(self.rate_limit_manager.rate_limits),
            "total_requests": len(self.rate_limit_manager.request_history)
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.rate_limit_manager.reset_request_history()
    
    async def execute_with_limits(
        self,
        operation: str,
        func,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with rate limiting."""
        return await self.rate_limit_manager.execute_with_rate_limit(operation, func, *args, **kwargs)
    
    def set_limits(self, operation: str, limits: Dict[str, int]) -> None:
        """Set rate limits for operation."""
        rate_limit = RateLimit(
            requests_per_second=limits.get("per_second", 10),
            requests_per_minute=limits.get("per_minute", 600),
            requests_per_hour=limits.get("per_hour", 36000),
            burst_limit=limits.get("burst", 20)
        )
        self.rate_limit_manager.set_rate_limit(operation, rate_limit)
    
    def get_remaining_requests(self, operation: str) -> int:
        """Get remaining requests for operation."""
        status = self.rate_limit_manager.get_rate_limit_status(operation)
        return status["remaining_requests"]
    
    def is_limited(self, operation: str) -> bool:
        """Check if operation is currently limited."""
        status = self.rate_limit_manager.get_rate_limit_status(operation)
        return status["is_limited"]