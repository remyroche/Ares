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
    handle_errors, handle_async_errors, DataSource, ValidationResult,
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

# Import tprint utilities for comprehensive logging
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug
)

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
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.exchange_type:
            tprint_error("❌ exchange_type is required")
            raise ValueError("exchange_type is required")
        if not self.api_key:
            tprint_error("❌ api_key is required")
            raise ValueError("api_key is required")
        if not self.api_secret:
            tprint_error("❌ api_secret is required")
            raise ValueError("api_secret is required")
        if self.exchange_type.lower() not in ['binance', 'bingx']:
            tprint_error(f"❌ Unsupported exchange type: {self.exchange_type}")
            raise ValueError(f"Unsupported exchange type: {self.exchange_type}")
        if self.connection_retry_attempts < 1:
            tprint_error("❌ connection_retry_attempts must be >= 1")
            raise ValueError("connection_retry_attempts must be >= 1")
        if self.connection_retry_delay < 0:
            tprint_error("❌ connection_retry_delay must be >= 0")
            raise ValueError("connection_retry_delay must be >= 0")
        tprint_info(f"✅ Exchange integration config validated for {self.exchange_type}")

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
        self.connection_attempts = 0

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
            tprint_success(f"✅ Exchange integration initialized for {self.config.exchange_type}")

        except Exception as e:
            self.last_error = str(e)
            tprint_error(f"❌ Failed to initialize exchange integration: {e}")
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

            tprint_success("✅ Shared utilities initialized successfully")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize shared utilities: {e}")
            raise

    @handle_async_errors(default_return=False)
    async def connect(self) -> bool:
        """Connect to the exchange with retry logic."""
        try:
            if not self.is_initialized:
                tprint_error("❌ Integration not initialized")
                return False

            tprint_info(f"🔄 Connecting to {self.config.exchange_type}...")

            # Retry connection logic
            last_error: Optional[str] = None
            for attempt in range(1, self.config.connection_retry_attempts + 1):
                try:
                    # Connect using exchange interface
                    success = await self.exchange_interface.connect()

                    if success:
                        self.is_connected = True
                        self.connection_attempts = attempt
                        tprint_success(f"✅ Connected to {self.config.exchange_type} (attempt {attempt})")
                        return True
                    else:
                        if attempt < self.config.connection_retry_attempts:
                            tprint_warning(f"⚠️ Connection attempt {attempt} failed, retrying...")
                            await asyncio.sleep(self.config.connection_retry_delay * attempt)
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.config.connection_retry_attempts:
                        tprint_warning(f"⚠️ Connection attempt {attempt} failed: {e}, retrying...")
                        await asyncio.sleep(self.config.connection_retry_delay * attempt)
                    else:
                        break

            # All attempts failed
            self.is_connected = False
            self.last_error = last_error or "Connection failed after all retry attempts"
            tprint_error(f"❌ Failed to connect to {self.config.exchange_type} after {self.config.connection_retry_attempts} attempts")
            return False

        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            tprint_error(f"❌ Connection error: {e}")
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
            tprint_info(f"📴 Disconnected from {self.config.exchange_type}")

        except Exception as e:
            tprint_error(f"❌ Error during disconnect: {e}")

    @handle_async_errors(default_return=None)
    async def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get ticker data for symbol."""
        try:
            if not self.is_connected:
                tprint_error("❌ Not connected to exchange")
                return None

            tprint_debug(f"🔍 Getting ticker for {symbol}")
            result = await self.exchange_interface.get_ticker(symbol)
            if result:
                tprint_debug(f"✅ Ticker data retrieved for {symbol}")
            return result

        except Exception as e:
            tprint_error(f"❌ Error getting ticker for {symbol}: {e}")
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
                tprint_error("❌ Not connected to exchange")
                return []

            tprint_debug(f"🔍 Getting klines for {symbol} (interval: {interval}, limit: {limit})")
            result = await self.exchange_interface.get_klines(
                symbol, interval, start_time, end_time, limit
            )
            if result:
                tprint_debug(f"✅ Retrieved {len(result)} klines for {symbol}")
            return result

        except Exception as e:
            tprint_error(f"❌ Error getting klines for {symbol}: {e}")
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
                tprint_error("❌ Not connected to exchange")
                return {'error': 'not_connected'}

            tprint_info(f"🔄 Creating {order_type} order for {symbol} ({side}, qty: {quantity})")

            # Risk management check
            if self.risk_manager and price:
                risk_data = self.risk_manager.calculate_position_risk(
                    symbol, quantity, price, kwargs.get('leverage', 1.0)
                )

                # Check if risk is acceptable
                if not self.risk_manager.validate_risk_limits(risk_data).is_valid:
                    tprint_warning(f"❌ Order rejected due to risk limits for {symbol}")
                    return {'error': 'risk_limit_exceeded', 'risk_data': risk_data}

            # Create order
            result = await self.exchange_interface.create_order(
                symbol, side, order_type, quantity, price, **kwargs
            )

            if 'error' not in result:
                tprint_success(f"✅ Order created for {symbol}: {result.get('orderId', 'N/A')}")
            else:
                tprint_error(f"❌ Order creation failed: {result.get('error', 'unknown error')}")

            return result

        except Exception as e:
            tprint_error(f"❌ Error creating order for {symbol}: {e}")
            return {'error': str(e)}

    @handle_async_errors(default_return={})
    async def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """Get account balance."""
        try:
            if not self.is_connected:
                tprint_error("❌ Not connected to exchange")
                return {}

            tprint_debug(f"🔍 Getting account balance for {asset or 'all assets'}")
            result = await self.exchange_interface.get_account_balance(asset)
            if result:
                tprint_debug(f"✅ Retrieved balance data for {len(result)} asset(s)")
            return result

        except Exception as e:
            tprint_error(f"❌ Error getting account balance: {e}")
            return {}

    @handle_async_errors(default_return={})
    async def get_risk_info(self, symbol: str, position_size: float, current_price: float, leverage: float = 1.0) -> Dict[str, Any]:
        """Get risk information for a position."""
        try:
            if not self.is_connected:
                tprint_error("❌ Not connected to exchange")
                return {}

            tprint_debug(f"🔍 Getting risk info for {symbol} (size: {position_size}, price: {current_price}, leverage: {leverage})")
            result = await self.exchange_interface.get_risk_info(symbol, position_size, current_price, leverage)
            if result:
                tprint_debug(f"✅ Retrieved risk info for {symbol}")
            return result

        except Exception as e:
            tprint_error(f"❌ Error getting risk info for {symbol}: {e}")
            return {}

    @handle_async_errors(default_return={})
    async def get_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get portfolio risk information."""
        try:
            if not self.is_connected:
                tprint_error("❌ Not connected to exchange")
                return {}

            tprint_debug(f"🔍 Getting portfolio risk for {len(positions)} position(s)")
            result = await self.exchange_interface.get_portfolio_risk(positions)
            if result:
                tprint_debug(f"✅ Retrieved portfolio risk information")
            return result

        except Exception as e:
            tprint_error(f"❌ Error getting portfolio risk: {e}")
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
    async def reset(self) -> None:
        """Reset the integration."""
        try:
            # Disconnect if connected (await properly)
            if self.is_connected:
                await self.disconnect()

            # Reset state
            self.is_initialized = False
            self.is_connected = False
            self.last_error = None
            self.connection_attempts = 0

            # Reinitialize
            self._initialize_integration()

            tprint_success("✅ Exchange integration reset")

        except Exception as e:
            tprint_error(f"❌ Error resetting integration: {e}")

    async def close(self) -> None:
        """Close and cleanup resources."""
        try:
            await self.disconnect()
            # Close all managers
            for manager in [self.auth_manager, self.market_manager, self.order_manager,
                          self.risk_manager, self.balance_manager, self.rate_limit_manager]:
                if manager:
                    try:
                        manager.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing manager: {e}")
            tprint_success("✅ Exchange integration closed")
        except Exception as e:
            tprint_error(f"❌ Error closing integration: {e}")

    def __enter__(self) -> "ExchangeIntegrationManager":
        """Context manager entry."""
        return self

    async def __aenter__(self) -> "ExchangeIntegrationManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        """Async context manager exit."""
        await self.close()

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
        tprint_info(f"🔄 Creating exchange integration for {config.exchange_type}")
        manager = ExchangeIntegrationManager(config)
        tprint_success(f"✅ Exchange integration created for {config.exchange_type}")
        return manager
    except Exception as e:
        tprint_error(f"❌ Failed to create exchange integration: {e}")
        raise

# Convenience function for common configurations
def create_binance_integration(
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    trade_symbol: str = "BTCUSDT"
) -> ExchangeIntegrationManager:
    """
    Create Binance exchange integration.
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Whether to use testnet (default: True)
        trade_symbol: Trading symbol (default: "BTCUSDT")
        
    Returns:
        ExchangeIntegrationManager instance configured for Binance
    """
    tprint_info("🔄 Creating Binance exchange integration")
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
    """
    Create BingX exchange integration.
    
    Args:
        api_key: BingX API key
        api_secret: BingX API secret
        testnet: Whether to use testnet (default: True)
        trade_symbol: Trading symbol (default: "BTCUSDT")
        
    Returns:
        ExchangeIntegrationManager instance configured for BingX
    """
    tprint_info("🔄 Creating BingX exchange integration")
    config = ExchangeIntegrationConfig(
        exchange_type="bingx",
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        trade_symbol=trade_symbol
    )
    return create_exchange_integration(config)
