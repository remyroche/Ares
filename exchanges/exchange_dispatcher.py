"""
Exchange Dispatcher

Exchange-agnostic dispatcher that routes trading operations to the appropriate exchange
implementation. Provides a unified interface for all exchange operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint_data_format, LogLevel
from src.interfaces.base_interfaces import MarketData

from .base_exchange import BaseExchange
# from .okx import create_okx_exchange, OkxExchange  # Commented out to avoid circular import
from .shared.auth import AuthenticationManager
from .shared.market import MarketMetadataManager
from .shared.pricing import PriceManager, OHLCVManager
from .shared.orders import OrderManager
from .shared.risk import RiskCalculator
from .shared.wallet import BalanceManager
from .shared.reliability import RateLimitManager


class ExchangeType(Enum):
    """Supported exchange types."""
    OKX = "okx"
    BINANCE = "binance"
    GATEIO = "gateio"
    MEXC = "mexc"
    PHEMEX = "phemex"


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    password: Optional[str] = None
    subaccount_id: Optional[str] = None
    use_testnet: bool = True
    trade_symbol: str = "BTCUSDT"
    additional_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_config is None:
            self.additional_config = {}


class ExchangeDispatcher:
    """
    Exchange-agnostic dispatcher that routes operations to the appropriate exchange.
    
    Provides a unified interface for:
    - Market data access
    - Order management
    - Account operations
    - Risk management
    - Authentication
    """
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.logger = system_logger.getChild(f"ExchangeDispatcher.{config.exchange_type.value}")
        
        # Exchange instance
        self.exchange: Optional[BaseExchange] = None
        
        # Shared utilities (will be initialized with exchange)
        self.auth_manager: Optional[AuthenticationManager] = None
        self.market_metadata: Optional[MarketMetadataManager] = None
        self.price_manager: Optional[PriceManager] = None
        self.ohlcv_manager: Optional[OHLCVManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.risk_calculator: Optional[RiskCalculator] = None
        self.balance_manager: Optional[BalanceManager] = None
        self.rate_limit_manager: Optional[RateLimitManager] = None
        
        # Initialization state
        self._initialized = False
        self._initializing = False
        
    async def initialize(self) -> bool:
        """
        Initialize the exchange dispatcher and underlying exchange.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        if self._initializing:
            # Wait for ongoing initialization
            while self._initializing:
                await asyncio.sleep(0.1)
            return self._initialized
            
        self._initializing = True
        
        try:
            # Create exchange instance
            self.exchange = await self._create_exchange()
            if not self.exchange:
                self.logger.error("Failed to create exchange instance")
                return False
            
            # Initialize exchange
            await self.exchange._initialize_exchange()
            
            # Initialize shared utilities
            await self._initialize_shared_utilities()
            
            self._initialized = True
            self.logger.info(f"✅ Exchange dispatcher initialized for {self.config.exchange_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize exchange dispatcher: {e}")
            return False
        finally:
            self._initializing = False
    
    async def _create_exchange(self) -> Optional[BaseExchange]:
        """Create the appropriate exchange instance."""
        try:
            if self.config.exchange_type == ExchangeType.OKX:
                return create_okx_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    password=self.config.password,
                    subaccount_id=self.config.subaccount_id,
                    use_testnet=self.config.use_testnet,
                    trade_symbol=self.config.trade_symbol
                )
            elif self.config.exchange_type == ExchangeType.BINANCE:
                from .binance import create_binance_exchange
                return create_binance_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    trade_symbol=self.config.trade_symbol,
                    password=self.config.password,
                    subaccount_id=self.config.subaccount_id,
                    use_testnet=self.config.use_testnet
                )
            elif self.config.exchange_type == ExchangeType.GATEIO:
                from .gateio import create_gateio_exchange
                return create_gateio_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    trade_symbol=self.config.trade_symbol,
                    password=self.config.password
                )
            elif self.config.exchange_type == ExchangeType.MEXC:
                from .mexc import create_mexc_exchange
                return create_mexc_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    trade_symbol=self.config.trade_symbol,
                    password=self.config.password,
                    subaccount_id=self.config.subaccount_id,
                    use_testnet=self.config.use_testnet
                )
            elif self.config.exchange_type == ExchangeType.PHEMEX:
                from .phemex import create_phemex_exchange
                return create_phemex_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    trade_symbol=self.config.trade_symbol,
                    password=self.config.password
                )
            else:
                self.logger.error(f"Unsupported exchange type: {self.config.exchange_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating exchange instance: {e}")
            return None
    
    async def _initialize_shared_utilities(self) -> None:
        """Initialize shared utilities from the exchange instance."""
        if not self.exchange or not hasattr(self.exchange, 'auth_manager'):
            return
            
        # Get shared utilities from exchange
        self.auth_manager = getattr(self.exchange, 'auth_manager', None)
        self.market_metadata = getattr(self.exchange, 'market_metadata', None)
        self.price_manager = getattr(self.exchange, 'price_manager', None)
        self.ohlcv_manager = getattr(self.exchange, 'ohlcv_manager', None)
        self.order_manager = getattr(self.exchange, 'order_manager', None)
        self.risk_calculator = getattr(self.exchange, 'risk_calculator', None)
        self.balance_manager = getattr(self.exchange, 'balance_manager', None)
        self.rate_limit_manager = getattr(self.exchange, 'rate_limit_manager', None)
    
    # Market Data Operations
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        if not self._ensure_initialized():
            return None
            
        if self.price_manager:
            price_data = await self.price_manager.get_price(symbol)
            return price_data.price if price_data else None
        
        # Fallback to exchange method
        if hasattr(self.exchange, 'get_price'):
            return await self.exchange.get_price(symbol)
        
        return None
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[MarketData]:
        """Get OHLCV data for symbol."""
        if not self._ensure_initialized():
            return []
            
        if hasattr(self.exchange, 'get_ohlcv'):
            ohlcv_data = await self.exchange.get_ohlcv(symbol, timeframe, limit)
            
            # Log OHLCV data format for debugging
            tprint_data_format(
                ohlcv_data, 
                f"dispatcher_ohlcv_{self.config.exchange_type.value}_{symbol}_{timeframe}", 
                level=LogLevel.DEBUG
            )
            
            return ohlcv_data
        
        return []
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for symbol."""
        if not self._ensure_initialized():
            return None
            
        if hasattr(self.exchange, 'get_ticker'):
            ticker_data = await self.exchange.get_ticker(symbol)
            
            # Log ticker data format for debugging
            tprint_data_format(
                ticker_data, 
                f"dispatcher_ticker_{self.config.exchange_type.value}_{symbol}", 
                level=LogLevel.DEBUG
            )
            
            return ticker_data
        
        return None
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book for symbol."""
        if not self._ensure_initialized():
            return None
            
        if hasattr(self.exchange, 'get_order_book'):
            orderbook_data = await self.exchange.get_order_book(symbol, limit)
            
            # Log order book data format for debugging
            tprint_data_format(
                orderbook_data, 
                f"dispatcher_orderbook_{self.config.exchange_type.value}_{symbol}", 
                level=LogLevel.DEBUG
            )
            
            return orderbook_data
        
        return None
    
    # Account Operations
    async def get_balance(self, currency: str = "USDT") -> float:
        """Get balance for currency."""
        if not self._ensure_initialized():
            return 0.0
            
        if hasattr(self.exchange, 'get_balance'):
            return await self.exchange.get_balance(currency)
        
        return 0.0
    
    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        if not self._ensure_initialized():
            return None
            
        if hasattr(self.exchange, 'get_account_info'):
            return await self.exchange.get_account_info()
        
        return None
    
    # Order Operations
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create order."""
        if not self._ensure_initialized():
            return None
            
        if hasattr(self.exchange, 'create_order_enhanced'):
            return await self.exchange.create_order_enhanced(
                symbol, side, order_type, quantity, price, stop_price, client_order_id
            )
        elif hasattr(self.exchange, 'create_order'):
            return await self.exchange.create_order(
                symbol, side, quantity, price, order_type
            )
        
        return None
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        if not self._ensure_initialized():
            return False
            
        if hasattr(self.exchange, 'cancel_order'):
            return await self.exchange.cancel_order(symbol, order_id)
        
        return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status."""
        if not self._ensure_initialized():
            return None
            
        if hasattr(self.exchange, 'get_order_status'):
            return await self.exchange.get_order_status(symbol, order_id)
        
        return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        if not self._ensure_initialized():
            return []
            
        if hasattr(self.exchange, 'get_open_orders'):
            return await self.exchange.get_open_orders(symbol)
        
        return []
    
    # Position Operations
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        if not self._ensure_initialized():
            return []
            
        if hasattr(self.exchange, 'get_positions'):
            return await self.exchange.get_positions()
        
        return []
    
    async def get_liquidation_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get liquidation risk for symbol."""
        if not self._ensure_initialized():
            return None
            
        if hasattr(self.exchange, 'get_liquidation_risk'):
            return await self.exchange.get_liquidation_risk(symbol)
        
        return None
    
    # Market Metadata Operations
    async def get_instrument_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get instrument information."""
        if not self._ensure_initialized():
            return None
            
        if self.market_metadata:
            instrument = self.market_metadata.get_instrument(symbol)
            if instrument:
                return {
                    "symbol": instrument.symbol,
                    "base_currency": instrument.base_currency,
                    "quote_currency": instrument.quote_currency,
                    "tick_size": instrument.tick_size,
                    "lot_size": instrument.lot_size,
                    "min_notional": instrument.min_notional,
                    "max_leverage": instrument.max_leverage,
                    "price_precision": instrument.price_precision,
                    "quantity_precision": instrument.quantity_precision
                }
        
        return None
    
    # Risk Management Operations
    async def calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        current_price: float,
        leverage: float
    ) -> Optional[Dict[str, Any]]:
        """Calculate position risk."""
        if not self._ensure_initialized():
            return None
            
        if self.risk_calculator:
            position_risk = self.risk_calculator.calculate_position_risk(
                symbol, position_size, entry_price, current_price, leverage
            )
            return {
                "symbol": position_risk.symbol,
                "margin_ratio": position_risk.margin_ratio,
                "liquidation_price": position_risk.liquidation_price,
                "risk_level": position_risk.risk_level.value,
                "unrealized_pnl": position_risk.unrealized_pnl,
                "margin_used": position_risk.margin_used
            }
        
        return None
    
    # Utility Operations
    async def is_connected(self) -> bool:
        """Check if connected to exchange."""
        if not self._initialized or not self.exchange:
            return False
            
        if hasattr(self.exchange, 'is_connected'):
            return await self.exchange.is_connected()
        
        return True
    
    async def get_exchange_status(self) -> Dict[str, Any]:
        """Get comprehensive exchange status."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        status = {
            "exchange_type": self.config.exchange_type.value,
            "initialized": self._initialized,
            "connected": await self.is_connected(),
            "testnet": self.config.use_testnet
        }
        
        # Add authentication status
        if self.auth_manager:
            auth_status = self.auth_manager.get_authentication_status()
            status["authentication"] = {
                "is_authenticated": auth_status["is_authenticated"],
                "time_synced": auth_status["time_synced"],
                "permissions": auth_status["permissions"]
            }
        
        # Add rate limiting status
        if self.rate_limit_manager:
            rate_limit_stats = self.rate_limit_manager.get_rate_limit_statistics()
            status["rate_limiting"] = {
                "total_requests": rate_limit_stats["total_requests"],
                "requests_last_minute": rate_limit_stats["requests_last_minute"]
            }
        
        return status
    
    def _ensure_initialized(self) -> bool:
        """Ensure dispatcher is initialized."""
        if not self._initialized:
            self.logger.warning("Exchange dispatcher not initialized. Call initialize() first.")
            return False
        return True
    
    async def close(self) -> None:
        """Close the exchange dispatcher."""
        if self.exchange and hasattr(self.exchange, 'close'):
            await self.exchange.close()
        
        self._initialized = False
        self.logger.info("Exchange dispatcher closed")


# Factory function for creating exchange dispatchers
def create_exchange_dispatcher(config: ExchangeConfig) -> ExchangeDispatcher:
    """Create an exchange dispatcher with the given configuration."""
    return ExchangeDispatcher(config)


# Convenience function for creating OKX dispatcher
def create_okx_dispatcher(
    api_key: str,
    api_secret: str,
    password: Optional[str] = None,
    subaccount_id: Optional[str] = None,
    use_testnet: bool = True,
    trade_symbol: str = "BTCUSDT"
) -> ExchangeDispatcher:
    """Create an OKX exchange dispatcher."""
    config = ExchangeConfig(
        exchange_type=ExchangeType.OKX,
        api_key=api_key,
        api_secret=api_secret,
        password=password,
        subaccount_id=subaccount_id,
        use_testnet=use_testnet,
        trade_symbol=trade_symbol
    )
    return create_exchange_dispatcher(config)


# Convenience function for creating Binance dispatcher
def create_binance_dispatcher(
    api_key: str,
    api_secret: str,
    trade_symbol: str,
    use_testnet: bool = True
) -> ExchangeDispatcher:
    """Create a Binance exchange dispatcher."""
    config = ExchangeConfig(
        exchange_type=ExchangeType.BINANCE,
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        trade_symbol=trade_symbol
    )
    return create_exchange_dispatcher(config)


# Convenience function for creating Gate.io dispatcher
def create_gateio_dispatcher(
    api_key: str,
    api_secret: str,
    trade_symbol: str
) -> ExchangeDispatcher:
    """Create a Gate.io exchange dispatcher."""
    config = ExchangeConfig(
        exchange_type=ExchangeType.GATEIO,
        api_key=api_key,
        api_secret=api_secret,
        trade_symbol=trade_symbol
    )
    return create_exchange_dispatcher(config)


# Convenience function for creating MEXC dispatcher
def create_mexc_dispatcher(
    api_key: str,
    api_secret: str,
    trade_symbol: str,
    use_testnet: bool = True
) -> ExchangeDispatcher:
    """Create a MEXC exchange dispatcher."""
    config = ExchangeConfig(
        exchange_type=ExchangeType.MEXC,
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        trade_symbol=trade_symbol
    )
    return create_exchange_dispatcher(config)


# Convenience function for creating Phemex dispatcher
def create_phemex_dispatcher(
    api_key: str,
    api_secret: str,
    trade_symbol: str
) -> ExchangeDispatcher:
    """Create a Phemex exchange dispatcher."""
    config = ExchangeConfig(
        exchange_type=ExchangeType.PHEMEX,
        api_key=api_key,
        api_secret=api_secret,
        trade_symbol=trade_symbol
    )
    return create_exchange_dispatcher(config)