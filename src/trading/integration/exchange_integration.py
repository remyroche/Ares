"""
Exchange Integration Module

This module provides comprehensive integration between exchanges/shared/ modules
and the trading system, ensuring proper wiring and type safety.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, Protocol
from dataclasses import dataclass, field

# Import shared exchange utilities
from exchanges.shared.interfaces_typed import (
    tprint, handle_errors, handle_async_errors, DataSource, ValidationResult,
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager
)
from exchanges.shared import (
    HighLevelAuthManager, HighLevelMarketManager, HighLevelOrderManager,
    HighLevelRiskManager, HighLevelBalanceManager, HighLevelRateLimitManager
)

# Import exchange implementations
from exchanges.binance import BinanceExchange, create_binance_exchange
from exchanges.bingx import BingXExchange, create_bingx_exchange

# Import trading interfaces
from ..execution.exchange_interface import ExchangeInterface, TickerData, KlineData

logger = logging.getLogger(__name__)

@dataclass
class ExchangeIntegrationConfig:
    """Configuration for exchange integration."""
    exchange_type: str
    api_key: str
    api_secret: str
    testnet: bool = True
    rate_limits: Dict[str, int] = field(default_factory=dict)
    trade_symbol: str = "BTCUSDT"
    password: Optional[str] = None
    enable_shared_utilities: bool = True
    enable_risk_management: bool = True
    enable_rate_limiting: bool = True

class ExchangeIntegrationManager:
    """
    Manages integration between exchanges/shared/ modules and trading system.

    Provides:
    - Unified exchange interface with shared utilities
    - Comprehensive error handling with tprint
    - Type-safe operations
    - Risk management integration
    - Rate limiting
    """

    def __init__(self, config: ExchangeIntegrationConfig):
        """Initialize exchange integration manager."""
        self.config = config
        self.exchange: Optional[Union[BinanceExchange, BingXExchange]] = None
        self.exchange_interface: Optional[ExchangeInterface] = None

        # Shared utilities
        self.auth_manager: Optional[IHighLevelAuthManager] = None
        self.market_manager: Optional[IHighLevelMarketManager] = None
        self.order_manager: Optional[IHighLevelOrderManager] = None
        self.risk_manager: Optional[IHighLevelRiskManager] = None
        self.balance_manager: Optional[IHighLevelBalanceManager] = None
        self.rate_limit_manager: Optional[IHighLevelRateLimitManager] = None

        # Status tracking
        self.is_initialized = False
        self.is_connected = False
        self.last_error: Optional[str] = None

        # Initialize the integration
        self._initialize_integration()

    @handle_errors(default_return=None)
    def _initialize_integration(self) -> None:
        """Initialize the exchange integration."""
        try:
            # Create exchange instance based on type
            if self.config.exchange_type.lower() == 'binance':
                self.exchange = create_binance_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    trade_symbol=self.config.trade_symbol,
                    password=self.config.password
                )
            elif self.config.exchange_type.lower() == 'bingx':
                self.exchange = create_bingx_exchange(
                    api_key=self.config.api_key,
                    api_secret=self.config.api_secret,
                    trade_symbol=self.config.trade_symbol,
                    password=self.config.password
                )
            else:
                raise ValueError(f"Unsupported exchange type: {self.config.exchange_type}")

            # Initialize shared utilities if enabled
            if self.config.enable_shared_utilities:
                self._initialize_shared_utilities()

            # Create exchange interface
            interface_config = {
                'exchange_type': self.config.exchange_type,
                'api_key': self.config.api_key,
                'api_secret': self.config.api_secret,
                'testnet': self.config.testnet,
                'rate_limits': self.config.rate_limits,
                'trade_symbol': self.config.trade_symbol
            }
            self.exchange_interface = ExchangeInterface(interface_config)

            self.is_initialized = True
            tprint(f"✅ Exchange integration initialized for {self.config.exchange_type}", "INFO")

        except Exception as e:
            self.last_error = str(e)
            tprint(f"❌ Failed to initialize exchange integration: {e}", "ERROR")
            raise

    @handle_errors(default_return=None)
    def _initialize_shared_utilities(self) -> None:
        """Initialize shared exchange utilities."""
        try:
            exchange_config = {
                'exchange_type': self.config.exchange_type,
                'api_key': self.config.api_key,
                'api_secret': self.config.api_secret,
                'testnet': self.config.testnet,
                'rate_limits': self.config.rate_limits
            }

            # Initialize each utility with correct parameters
            self.auth_manager = HighLevelAuthManager(self.config.exchange_type)
            self.market_manager = HighLevelMarketManager(self.config.exchange_type)
            self.order_manager = HighLevelOrderManager(self.config.exchange_type)

            if self.config.enable_risk_management:
                self.risk_manager = HighLevelRiskManager(self.config.exchange_type)

            self.balance_manager = HighLevelBalanceManager(self.config.exchange_type)

            if self.config.enable_rate_limiting:
                self.rate_limit_manager = HighLevelRateLimitManager(self.config.exchange_type)

            # Initialize all utilities
            for manager in [self.auth_manager, self.market_manager, self.order_manager,
                          self.risk_manager, self.balance_manager, self.rate_limit_manager]:
                if manager:
                    manager.initialize()

            tprint("✅ Shared utilities initialized successfully", "INFO")

        except Exception as e:
            tprint(f"❌ Failed to initialize shared utilities: {e}", "ERROR")
            raise

    @handle_async_errors(default_return=False)
    async def connect(self) -> bool:
        """Connect to the exchange."""
        try:
            if not self.is_initialized:
                tprint("❌ Integration not initialized", "ERROR")
                return False

            # Connect using exchange interface
            success = await self.exchange_interface.connect()

            if success:
                self.is_connected = True
                tprint(f"✅ Connected to {self.config.exchange_type}", "INFO")
            else:
                self.is_connected = False
                tprint(f"❌ Failed to connect to {self.config.exchange_type}", "ERROR")

            return success

        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            tprint(f"❌ Connection error: {e}", "ERROR")
            return False

    @handle_async_errors(default_return=None)
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        try:
            if self.exchange_interface:
                await self.exchange_interface.disconnect()

            # Close shared utilities
            for manager in [self.auth_manager, self.market_manager, self.order_manager,
                          self.risk_manager, self.balance_manager, self.rate_limit_manager]:
                if manager:
                    manager.close()

            self.is_connected = False
            tprint(f"📴 Disconnected from {self.config.exchange_type}", "INFO")

        except Exception as e:
            tprint(f"❌ Error during disconnect: {e}", "ERROR")

    @handle_async_errors(default_return=None)
    async def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get ticker data for symbol."""
        try:
            if not self.is_connected:
                tprint("❌ Not connected to exchange", "ERROR")
                return None

            return await self.exchange_interface.get_ticker(symbol)

        except Exception as e:
            tprint(f"❌ Error getting ticker for {symbol}: {e}", "ERROR")
            return None

    @handle_async_errors(default_return=[])
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[KlineData]:
        """Get kline data for symbol."""
        try:
            if not self.is_connected:
                tprint("❌ Not connected to exchange", "ERROR")
                return []

            return await self.exchange_interface.get_klines(
                symbol, interval, start_time, end_time, limit
            )

        except Exception as e:
            tprint(f"❌ Error getting klines for {symbol}: {e}", "ERROR")
            return []

    @handle_async_errors(default_return={})
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create order with risk management."""
        try:
            if not self.is_connected:
                tprint("❌ Not connected to exchange", "ERROR")
                return {'error': 'not_connected'}

            # Risk management check
            if self.risk_manager and price:
                risk_data = self.risk_manager.calculate_position_risk(
                    symbol, quantity, price, kwargs.get('leverage', 1.0)
                )

                # Check if risk is acceptable
                if not self.risk_manager.validate_risk_limits(risk_data).is_valid:
                    tprint(f"❌ Order rejected due to risk limits for {symbol}", "WARNING")
                    return {'error': 'risk_limit_exceeded', 'risk_data': risk_data}

            # Create order
            result = await self.exchange_interface.create_order(
                symbol, side, order_type, quantity, price, **kwargs
            )

            if 'error' not in result:
                tprint(f"✅ Order created for {symbol}: {result.get('orderId', 'N/A')}", "INFO")

            return result

        except Exception as e:
            tprint(f"❌ Error creating order for {symbol}: {e}", "ERROR")
            return {'error': str(e)}

    @handle_async_errors(default_return={})
    async def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """Get account balance."""
        try:
            if not self.is_connected:
                tprint("❌ Not connected to exchange", "ERROR")
                return {}

            return await self.exchange_interface.get_account_balance(asset)

        except Exception as e:
            tprint(f"❌ Error getting account balance: {e}", "ERROR")
            return {}

    @handle_async_errors(default_return={})
    async def get_risk_info(self, symbol: str, position_size: float, current_price: float, leverage: float = 1.0) -> Dict[str, Any]:
        """Get risk information for a position."""
        try:
            if not self.is_connected:
                tprint("❌ Not connected to exchange", "ERROR")
                return {}

            return await self.exchange_interface.get_risk_info(symbol, position_size, current_price, leverage)

        except Exception as e:
            tprint(f"❌ Error getting risk info for {symbol}: {e}", "ERROR")
            return {}

    @handle_async_errors(default_return=[])
    async def get_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get portfolio risk information."""
        try:
            if not self.is_connected:
                tprint("❌ Not connected to exchange", "ERROR")
                return {}

            return await self.exchange_interface.get_portfolio_risk(positions)

        except Exception as e:
            tprint(f"❌ Error getting portfolio risk: {e}", "ERROR")
            return {}

    @handle_errors(default_return={})
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            'is_initialized': self.is_initialized,
            'is_connected': self.is_connected,
            'exchange_type': self.config.exchange_type,
            'last_error': self.last_error,
            'shared_utilities_enabled': self.config.enable_shared_utilities,
            'risk_management_enabled': self.config.enable_risk_management,
            'rate_limiting_enabled': self.config.enable_rate_limiting
        }

    @handle_errors(default_return=None)
    def reset(self) -> None:
        """Reset the integration."""
        try:
            # Disconnect if connected
            if self.is_connected:
                asyncio.create_task(self.disconnect())

            # Reset state
            self.is_initialized = False
            self.is_connected = False
            self.last_error = None

            # Reinitialize
            self._initialize_integration()

            tprint("✅ Exchange integration reset", "INFO")

        except Exception as e:
            tprint(f"❌ Error resetting integration: {e}", "ERROR")

# Factory function for creating exchange integration
def create_exchange_integration(config: ExchangeIntegrationConfig) -> ExchangeIntegrationManager:
    """
    Create exchange integration manager.

    Args:
        config: Exchange integration configuration

    Returns:
        ExchangeIntegrationManager instance
    """
    try:
        return ExchangeIntegrationManager(config)
    except Exception as e:
        tprint(f"❌ Failed to create exchange integration: {e}", "ERROR")
        raise

# Convenience function for common configurations
def create_binance_integration(
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    trade_symbol: str = "BTCUSDT"
) -> ExchangeIntegrationManager:
    """Create Binance exchange integration."""
    config = ExchangeIntegrationConfig(
        exchange_type="binance",
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        trade_symbol=trade_symbol
    )
    return create_exchange_integration(config)

def create_bingx_integration(
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    trade_symbol: str = "BTCUSDT"
) -> ExchangeIntegrationManager:
    """Create BingX exchange integration."""
    config = ExchangeIntegrationConfig(
        exchange_type="bingx",
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        trade_symbol=trade_symbol
    )
    return create_exchange_integration(config)
