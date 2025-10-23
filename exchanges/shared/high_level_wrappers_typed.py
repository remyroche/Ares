"""
High-level wrapper classes for shared exchange utilities with comprehensive type hints.

These classes provide consistent, high-level interfaces that abstract away
implementation details and provide uniform abstraction levels.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable, TypeVar, cast
from datetime import datetime
import logging

from .interfaces_typed import (
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
    DataSource, ValidationResult, tprint, handle_errors, handle_async_errors
)

# Type variables
T = TypeVar('T')

# Logger for error handling
logger = logging.getLogger(__name__)

# Import low-level managers with error handling
try:
    from .auth.auth_manager import AuthenticationManager, AuthConfig, APIKeyPermission
    from .market.market_metadata import MarketMetadataManager, InstrumentType
    from .orders.order_manager import OrderManager, OrderSide, OrderType
    from .risk.risk_calculator import RiskCalculator, RiskLevel
    from .wallet.balance_manager import BalanceManager, AccountType
    from .reliability.rate_limit_manager import RateLimitManager, RateLimit
except ImportError as e:
    tprint(f"Failed to import low-level managers: {e}", "ERROR")
    raise


class HighLevelAuthManager(IHighLevelAuthManager):
    """High-level wrapper for authentication management with comprehensive type hints."""
    
    def __init__(self, exchange_name: str) -> None:
        """Initialize the high-level auth manager."""
        try:
            self.exchange_name: str = exchange_name
            self.auth_manager: AuthenticationManager = AuthenticationManager(exchange_name)
            self._initialized: bool = False
            tprint(f"Initialized HighLevelAuthManager for {exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize HighLevelAuthManager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def initialize(self) -> None:
        """Initialize the authentication manager."""
        try:
            self._initialized = True
            tprint(f"Auth manager initialized for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize auth manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def close(self) -> None:
        """Close the authentication manager."""
        try:
            self.auth_manager.logout()
            self._initialized = False
            tprint(f"Auth manager closed for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to close auth manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={"initialized": False, "authenticated": False, "permissions": [], "time_synced": False})
    def get_status(self) -> Dict[str, Any]:
        """Get authentication status."""
        try:
            return {
                "initialized": self._initialized,
                "authenticated": self.auth_manager.is_authenticated,
                "permissions": list(self.auth_manager.get_current_permissions()),
                "time_synced": self.auth_manager.is_time_synced()
            }
        except Exception as e:
            tprint(f"Failed to get auth status: {e}", "ERROR")
            return {"initialized": False, "authenticated": False, "permissions": [], "time_synced": False}
    
    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset to initial state."""
        try:
            self.auth_manager.logout()
            self._initialized = False
            tprint(f"Auth manager reset for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to reset auth manager: {e}", "ERROR")
            raise
    
    @handle_async_errors(default_return=False)
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with exchange using credentials."""
        try:
            if not self._initialized:
                self.initialize()
            
            # Convert credentials to AuthConfig
            permissions: set = set()
            if credentials.get("permissions"):
                for perm in credentials["permissions"]:
                    try:
                        permissions.add(APIKeyPermission(perm))
                    except ValueError as e:
                        tprint(f"Invalid permission '{perm}': {e}", "WARNING")
                        continue
            
            auth_config = AuthConfig(
                exchange_name=self.exchange_name,
                api_key=credentials["api_key"],
                api_secret=credentials["api_secret"],
                passphrase=credentials.get("passphrase"),
                permissions=permissions or {APIKeyPermission.READ},
                auto_sync_time=credentials.get("auto_sync_time", True)
            )
            
            result = await self.auth_manager.authenticate(auth_config)
            if result:
                tprint(f"Successfully authenticated with {self.exchange_name}", "INFO")
            else:
                tprint(f"Failed to authenticate with {self.exchange_name}", "WARNING")
            return result
        except KeyError as e:
            tprint(f"Missing required credential: {e}", "ERROR")
            return False
        except Exception as e:
            tprint(f"Authentication error: {e}", "ERROR")
            return False
    
    @handle_async_errors(default_return=False)
    async def reauthenticate(self) -> bool:
        """Re-authenticate if needed."""
        try:
            result = await self.auth_manager.reauthenticate()
            if result:
                tprint(f"Successfully re-authenticated with {self.exchange_name}", "INFO")
            else:
                tprint(f"Failed to re-authenticate with {self.exchange_name}", "WARNING")
            return result
        except Exception as e:
            tprint(f"Re-authentication error: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=False)
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        try:
            return self.auth_manager.is_authenticated_and_valid()
        except Exception as e:
            tprint(f"Failed to check authentication status: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=None)
    def get_auth_headers(self, request_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Get authentication headers for request."""
        try:
            return self.auth_manager.get_auth_headers(
                method=request_data.get("method", "GET"),
                endpoint=request_data.get("endpoint", ""),
                body=request_data.get("body", ""),
                additional_headers=request_data.get("additional_headers")
            )
        except Exception as e:
            tprint(f"Failed to get auth headers: {e}", "ERROR")
            return None
    
    @handle_errors(default_return=False)
    def has_permission(self, permission: str) -> bool:
        """Check if has specific permission."""
        try:
            perm = APIKeyPermission(permission)
            return self.auth_manager.has_permission(perm)
        except ValueError as e:
            tprint(f"Invalid permission '{permission}': {e}", "WARNING")
            return False
        except Exception as e:
            tprint(f"Failed to check permission: {e}", "ERROR")
            return False


class HighLevelMarketManager(IHighLevelMarketManager):
    """High-level wrapper for market data management with comprehensive type hints."""
    
    def __init__(self, exchange_name: str) -> None:
        """Initialize the high-level market manager."""
        try:
            self.exchange_name: str = exchange_name
            self.market_manager: MarketMetadataManager = MarketMetadataManager(exchange_name)
            self._initialized: bool = False
            tprint(f"Initialized HighLevelMarketManager for {exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize HighLevelMarketManager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def initialize(self) -> None:
        """Initialize the market manager."""
        try:
            self._initialized = True
            tprint(f"Market manager initialized for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize market manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def close(self) -> None:
        """Close the market manager."""
        try:
            self._initialized = False
            tprint(f"Market manager closed for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to close market manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={"initialized": False, "instruments_count": 0, "last_refresh": None})
    def get_status(self) -> Dict[str, Any]:
        """Get market manager status."""
        try:
            return {
                "initialized": self._initialized,
                "instruments_count": len(self.market_manager.instruments),
                "last_refresh": self.market_manager.last_refresh.isoformat() if self.market_manager.last_refresh else None
            }
        except Exception as e:
            tprint(f"Failed to get market status: {e}", "ERROR")
            return {"initialized": False, "instruments_count": 0, "last_refresh": None}
    
    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset to initial state."""
        try:
            self.market_manager.instruments.clear()
            self.market_manager.market_data.clear()
            self.market_manager.last_refresh = None
            tprint(f"Market manager reset for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to reset market manager: {e}", "ERROR")
            raise
    
    @handle_async_errors(default_return=None)
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get data with automatic source selection."""
        try:
            if source == DataSource.CACHE and not force_refresh:
                return self.market_manager.get_instrument(key)
            elif source == DataSource.EXCHANGE or force_refresh:
                await self.refresh_data([key])
                return self.market_manager.get_instrument(key)
            return None
        except Exception as e:
            tprint(f"Failed to get data for key '{key}': {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=False)
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh data from exchange."""
        try:
            if keys:
                return await self.market_manager.refresh_market_data(keys)
            else:
                return await self.market_manager.refresh_instruments()
        except Exception as e:
            tprint(f"Failed to refresh data: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=0)
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached data."""
        try:
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
        except Exception as e:
            tprint(f"Failed to invalidate cache: {e}", "ERROR")
            return 0
    
    @handle_errors(default_return={})
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        try:
            return self.market_manager.get_statistics()
        except Exception as e:
            tprint(f"Failed to get market statistics: {e}", "ERROR")
            return {}
    
    @handle_async_errors(default_return=None)
    async def get_instrument_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get instrument information."""
        try:
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
        except Exception as e:
            tprint(f"Failed to get instrument info for '{symbol}': {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=None)
    async def get_price(self, symbol: str, source: DataSource = DataSource.CACHE) -> Optional[float]:
        """Get current price for symbol."""
        try:
            market_data = self.market_manager.get_market_data(symbol)
            if market_data and "ticker" in market_data:
                ticker = market_data["ticker"]
                return ticker.get("last") or ticker.get("close") or ticker.get("price")
            return None
        except Exception as e:
            tprint(f"Failed to get price for '{symbol}': {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=None)
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data."""
        try:
            return self.market_manager.get_market_data(symbol)
        except Exception as e:
            tprint(f"Failed to get market data for '{symbol}': {e}", "ERROR")
            return None
    
    @handle_errors(default_return=False)
    def is_symbol_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        try:
            return self.market_manager.is_symbol_tradable(symbol)
        except Exception as e:
            tprint(f"Failed to check if symbol '{symbol}' is tradable: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=[])
    def search_instruments(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search instruments with filters."""
        try:
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
        except Exception as e:
            tprint(f"Failed to search instruments: {e}", "ERROR")
            return []


class HighLevelOrderManager(IHighLevelOrderManager):
    """High-level wrapper for order management with comprehensive type hints."""
    
    def __init__(self, exchange_name: str) -> None:
        """Initialize the high-level order manager."""
        try:
            self.exchange_name: str = exchange_name
            self.order_manager: OrderManager = OrderManager(exchange_name)
            self._initialized: bool = False
            tprint(f"Initialized HighLevelOrderManager for {exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize HighLevelOrderManager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def initialize(self) -> None:
        """Initialize the order manager."""
        try:
            self._initialized = True
            tprint(f"Order manager initialized for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize order manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return=None)
    def close(self) -> None:
        """Close the order manager."""
        try:
            self._initialized = False
            tprint(f"Order manager closed for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to close order manager: {e}", "ERROR")
            raise
    
    @handle_errors(default_return={"initialized": False, "total_orders": 0, "open_orders": 0})
    def get_status(self) -> Dict[str, Any]:
        """Get order manager status."""
        try:
            return {
                "initialized": self._initialized,
                "total_orders": len(self.order_manager.orders),
                "open_orders": len(self.order_manager.get_open_orders())
            }
        except Exception as e:
            tprint(f"Failed to get order status: {e}", "ERROR")
            return {"initialized": False, "total_orders": 0, "open_orders": 0}
    
    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset to initial state."""
        try:
            self.order_manager.orders.clear()
            self.order_manager.orders_by_symbol.clear()
            self.order_manager.orders_by_status.clear()
            tprint(f"Order manager reset for {self.exchange_name}", "DEBUG")
        except Exception as e:
            tprint(f"Failed to reset order manager: {e}", "ERROR")
            raise
    
    @handle_async_errors(default_return=None)
    async def get_data(
        self,
        key: str,
        source: DataSource = DataSource.CACHE,
        force_refresh: bool = False
    ) -> Optional[Any]:
        """Get order data."""
        try:
            return self.order_manager.get_order(key)
        except Exception as e:
            tprint(f"Failed to get order data for key '{key}': {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=False)
    async def refresh_data(self, keys: Optional[List[str]] = None) -> bool:
        """Refresh order data from exchange."""
        try:
            return await self.order_manager.sync_orders_from_exchange() > 0
        except Exception as e:
            tprint(f"Failed to refresh order data: {e}", "ERROR")
            return False
    
    @handle_errors(default_return=0)
    def invalidate_cache(self, keys: Optional[List[str]] = None) -> int:
        """Invalidate cached order data."""
        try:
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
        except Exception as e:
            tprint(f"Failed to invalidate order cache: {e}", "ERROR")
            return 0
    
    @handle_errors(default_return={})
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        try:
            return self.order_manager.get_order_statistics()
        except Exception as e:
            tprint(f"Failed to get order statistics: {e}", "ERROR")
            return {}
    
    @handle_async_errors(default_return=None)
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        **kwargs: Any
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
                tprint(f"Successfully created order {order.order_id} for {symbol}", "INFO")
                return order.order_id
            else:
                tprint(f"Failed to create order for {symbol}", "WARNING")
                return None
            
        except (ValueError, KeyError) as e:
            tprint(f"Invalid order parameters: {e}", "ERROR")
            return None
        except Exception as e:
            tprint(f"Failed to create order: {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=False)
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        try:
            result = await self.order_manager.cancel_order(order_id)
            if result:
                tprint(f"Successfully cancelled order {order_id}", "INFO")
            else:
                tprint(f"Failed to cancel order {order_id}", "WARNING")
            return result
        except Exception as e:
            tprint(f"Failed to cancel order {order_id}: {e}", "ERROR")
            return False
    
    @handle_async_errors(default_return=None)
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        try:
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
        except Exception as e:
            tprint(f"Failed to get order status for {order_id}: {e}", "ERROR")
            return None
    
    @handle_async_errors(default_return=[])
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        try:
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
        except Exception as e:
            tprint(f"Failed to get open orders: {e}", "ERROR")
            return []
    
    @handle_errors(default_return=ValidationResult(False, ["Validation failed"]))
    def validate_order_params(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate order parameters."""
        try:
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
        except Exception as e:
            tprint(f"Failed to validate order parameters: {e}", "ERROR")
            return ValidationResult(False, [f"Validation error: {e}"])
    
    @handle_errors(default_return=ValidationResult(False, ["Validation failed"]))
    def validate_data(self, data: Any, data_type: str) -> ValidationResult:
        """Validate data according to type-specific rules."""
        try:
            result = ValidationResult(True)
            
            if data_type == "order":
                if not isinstance(data, dict):
                    result.add_error("Order data must be a dictionary")
                else:
                    required_fields = ["symbol", "side", "order_type", "quantity"]
                    for field in required_fields:
                        if field not in data:
                            result.add_error(f"Missing required field: {field}")
            
            return result
        except Exception as e:
            tprint(f"Failed to validate data: {e}", "ERROR")
            return ValidationResult(False, [f"Validation error: {e}"])
    
    @handle_errors(default_return=ValidationResult(False, ["Validation failed"]))
    def validate_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate API request parameters."""
        try:
            result = ValidationResult(True)
            
            if not isinstance(request, dict):
                result.add_error("Request must be a dictionary")
            
            return result
        except Exception as e:
            tprint(f"Failed to validate request: {e}", "ERROR")
            return ValidationResult(False, [f"Validation error: {e}"])
    
    @handle_errors(default_return=ValidationResult(False, ["Validation failed"]))
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate API response data."""
        try:
            result = ValidationResult(True)
            
            if not isinstance(response, dict):
                result.add_error("Response must be a dictionary")
            
            return result
        except Exception as e:
            tprint(f"Failed to validate response: {e}", "ERROR")
            return ValidationResult(False, [f"Validation error: {e}"])