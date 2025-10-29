"""
Exchange Interface

Abstract interface for different cryptocurrency exchanges.
Provides unified API for market data access and order execution.
Integrates with exchanges/shared/ modules for comprehensive functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)

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

from exchanges.exchange_dispatcher import ExchangeDispatcher, ExchangeConfig, ExchangeType
from exchanges.shared import (
    # Auth
    AuthenticationManager, APIKeyManager, TimeSyncManager, SubaccountManager,
    # Market
    MarketMetadataManager, InstrumentManager, PrecisionHelper, RiskTierManager,
    # Pricing
    PriceManager, OHLCVManager, MarketDataAggregator,
    # Orders
    OrderManager, IdempotencyManager, PositionManager,
    # Risk
    RiskCalculator, LiquidationRiskManager, MarginManager,
    # History
    TradeHistoryManager, PaginationManager,
    # Wallet
    BalanceManager, WalletManager,
    # Reliability
    RateLimitManager, RetryManager, AuditLogger, SystemStatusManager
)
from exchanges.shared.interfaces_typed import (
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
    tprint, DataSource, ValidationResult
)
from ..utils.error_handling import (
    ExecutionError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config

logger = system_logger.getChild('ExchangeInterface')

class ExchangeType(Enum):
    """Exchange types."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    SIMULATED = "simulated"

class MarketDataType(Enum):
    """Market data types."""
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    KLINES = "klines"
    AGGREGATE_TRADES = "agg_trades"

class ConnectionStatus(Enum):
    """Connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

@dataclass
class MarketData:
    """Market data container."""
    symbol: str
    exchange: str
    data_type: MarketDataType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TickerData:
    """Ticker data."""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    volume_24h: float
    price_change_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

@dataclass
class KlineData:
    """Kline (candlestick) data."""
    symbol: str
    interval: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    close_time: datetime
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float

class ExchangeInterface:
    """
    Exchange interface that uses the exchange dispatcher and shared utilities.

    Provides unified API for:
    - Market data access (ticker, order book, klines, trades)
    - Order execution and management
    - Account information and balances
    - Rate limiting and error handling
    - Authentication and risk management
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exchange interface.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.exchange_type = config.get('exchange_type', 'simulated')
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.testnet = config.get('testnet', True)
        self.rate_limits = config.get('rate_limits', {})

        # Exchange dispatcher
        self.dispatcher: Optional[ExchangeDispatcher] = None

        # Shared utilities
        self.auth_manager: Optional[IHighLevelAuthManager] = None
        self.market_manager: Optional[IHighLevelMarketManager] = None
        self.order_manager: Optional[IHighLevelOrderManager] = None
        self.risk_manager: Optional[IHighLevelRiskManager] = None
        self.balance_manager: Optional[IHighLevelBalanceManager] = None
        self.rate_limit_manager: Optional[IHighLevelRateLimitManager] = None
        self._initialize_shared_utilities()

        # Connection state
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_connection_attempt = None
        self.connection_errors = []

        # Data streams
        self.ticker_streams: Dict[str, Any] = {}
        self.order_book_streams: Dict[str, Any] = {}
        self.kline_streams: Dict[str, Dict[str, Any]] = {}

        # Rate limiting
        self.request_counts: Dict[str, int] = {}
        self.last_requests: Dict[str, datetime] = {}

        # Performance tracking
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0

        # Simulated exchange data
        self.price_feeds: Dict[str, Dict[str, float]] = {}
        self.simulated_orders: Dict[str, Dict[str, Any]] = {}

        # Initialize simulated data
        self._initialize_simulated_data()

        self.logger = logger.getChild(f'{self.exchange_type}')

    def _initialize_shared_utilities(self) -> None:
        """Initialize shared utilities for exchange operations."""
        try:
            # Authentication
            self.auth_manager = AuthenticationManager(self.exchange_type)
            self.api_key_manager = APIKeyManager(self.exchange_type)
            self.time_sync_manager = TimeSyncManager(self.exchange_type)
            self.subaccount_manager = SubaccountManager(self.exchange_type)

            # Market metadata
            self.market_metadata = MarketMetadataManager(self.exchange_type)
            self.instrument_manager = InstrumentManager(self.exchange_type)
            self.precision_helper = PrecisionHelper()
            self.risk_tier_manager = RiskTierManager(self.exchange_type)

            # Pricing
            self.price_manager = PriceManager(self.exchange_type)
            self.ohlcv_manager = OHLCVManager(self.exchange_type)

            # Orders
            self.order_manager = OrderManager(self.exchange_type)
            self.idempotency_manager = IdempotencyManager(self.exchange_type)

            # Risk
            self.risk_calculator = RiskCalculator(self.exchange_type)

            # Wallet
            self.balance_manager = BalanceManager(self.exchange_type)

            # Reliability
            self.rate_limit_manager = RateLimitManager(self.exchange_type)
            self.audit_logger = AuditLogger(self.exchange_type)

        except Exception as e:
            tprint(f"Failed to initialize shared utilities: {e}", "ERROR")
            raise

    def _initialize_simulated_data(self) -> None:
        """Initialize simulated exchange data."""
        self.price_feeds['ETHUSDT'] = {
            'price': 3000.0,
            'bid_price': 2999.5,
            'ask_price': 3000.5,
            'volume_24h': 1000000.0,
            'price_change_24h': 50.0,
            'high_24h': 3100.0,
            'low_24h': 2900.0
        }

        self.price_feeds['BTCUSDT'] = {
            'price': 50000.0,
            'bid_price': 49995.0,
            'ask_price': 50005.0,
            'volume_24h': 500000.0,
            'price_change_24h': 1000.0,
            'high_24h': 51000.0,
            'low_24h': 48000.0
        }


    @handle_async_errors(default_return=False)
    async def connect(self, max_retries: int = 5, initial_backoff: float = 1.0) -> bool:
        """Connect to exchange with retry logic and exponential backoff."""
        try:
            if self.exchange_type == 'simulated':
                self.connection_status = ConnectionStatus.CONNECTED
                tprint("✅ Connected to simulated exchange", "INFO")
                return True

            # Retry logic with exponential backoff
            backoff = initial_backoff
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    self.connection_status = ConnectionStatus.CONNECTING
                    self.last_connection_attempt = datetime.now()
                    
                    # Authenticate using shared auth manager
                    if self.auth_manager:
                        credentials = {
                            'api_key': self.api_key,
                            'api_secret': self.api_secret,
                            'testnet': self.testnet
                        }
                        auth_success = await self.auth_manager.authenticate(credentials)
                        if not auth_success:
                            raise Exception("Authentication failed")
                        tprint(f"✅ Connected to {self.exchange_type}", "INFO")
                        self.connection_status = ConnectionStatus.CONNECTED
                        return True

                    # Initialize shared utilities for real exchanges
                    await self._initialize_exchange_utilities()

                    # Create exchange dispatcher
                    exchange_type = ExchangeType.OKX if self.exchange_type == 'okx' else ExchangeType.BINANCE
                    config = ExchangeConfig(
                        exchange_type=exchange_type,
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        use_testnet=self.testnet,
                        trade_symbol=self.config.get('trade_symbol', 'BTCUSDT')
                    )

                    self.dispatcher = ExchangeDispatcher(config)
                    success = await self.dispatcher.initialize()

                    if success:
                        self.connection_status = ConnectionStatus.CONNECTED
                        tprint(f"✅ Connected to {self.exchange_type}", "INFO")
                        return True
                    else:
                        raise Exception(f"Failed to initialize dispatcher for {self.exchange_type}")

                except Exception as e:
                    last_error = e
                    self.connection_errors.append({
                        'timestamp': datetime.now(),
                        'operation': 'connect',
                        'attempt': attempt + 1,
                        'error': str(e)
                    })
                    
                    if attempt < max_retries - 1:
                        self.connection_status = ConnectionStatus.RECONNECTING
                        tprint_warning(f"⚠️ Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {backoff:.1f}s...")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60.0)  # Exponential backoff, max 60s
                    else:
                        # Last attempt failed
                        self.connection_status = ConnectionStatus.ERROR
                        tprint_error(f"❌ Failed to connect to {self.exchange_type} after {max_retries} attempts")
                        await self._handle_error(e, "connect")
                        return False

        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            await self._handle_error(e, "connect")
            return False

    async def _initialize_exchange_utilities(self) -> None:
        """Initialize exchange-specific utilities."""
        try:
            # Register exchange functions with shared utilities
            self._register_exchange_functions()

            # Set up rate limiting
            self._setup_rate_limiting()

            # Initialize market data
            await self._initialize_market_data()

        except Exception as e:
            tprint(f"Failed to initialize exchange utilities: {e}", "ERROR")
            raise

    def _register_exchange_functions(self) -> None:
        """Register exchange-specific functions with shared utilities."""
        try:
            # Auth functions
            self.auth_manager.register_auth_functions(
                get_server_time=self._get_server_time,
                test_connection=self._test_connection,
                get_account_info=self._get_account_info
            )

            # Market metadata functions
            self.market_metadata.register_refresh_functions(
                get_instruments=self._get_instruments,
                get_ticker=self._get_ticker,
                get_orderbook=self._get_order_book,
                get_funding_rate=self._get_funding_rate
            )

            # Price functions
            self.price_manager.register_price_functions(
                get_ticker=self._get_ticker,
                get_orderbook=self._get_order_book,
                get_recent_trades=self._get_recent_trades,
                get_klines=self._get_klines
            )

            # OHLCV functions
            self.ohlcv_manager.register_fetch_functions(
                get_klines=self._get_klines,
                get_historical_klines=self._get_historical_klines
            )

            # Order functions
            self.order_manager.register_execution_functions(
                create_order=self._create_order_exchange,
                cancel_order=self._cancel_order_exchange,
                get_order_status=self._get_order_status_exchange,
                get_open_orders=self._get_open_orders_exchange
            )

            # Balance functions
            self.balance_manager.register_fetch_functions(
                get_balances=self._get_balances_exchange,
                get_account_info=self._get_account_info
            )

        except Exception as e:
            tprint(f"Failed to register exchange functions: {e}", "ERROR")
            raise

    def _setup_rate_limiting(self) -> None:
        """Set up rate limiting for different endpoints."""
        try:
            from ...exchanges.shared.reliability.rate_limit_manager import RateLimit

            # General API rate limits
            general_limit = RateLimit(
                requests_per_second=20,
                requests_per_minute=1200,
                requests_per_hour=72000,
                burst_limit=50
            )

            # Trading rate limits (more restrictive)
            trading_limit = RateLimit(
                requests_per_second=10,
                requests_per_minute=600,
                requests_per_hour=36000,
                burst_limit=20
            )

            # Set rate limits for different endpoint categories
            self.rate_limit_manager.set_rate_limit("public", general_limit)
            self.rate_limit_manager.set_rate_limit("trading", trading_limit)
            self.rate_limit_manager.set_rate_limit("account", trading_limit)

        except Exception as e:
            tprint(f"Failed to setup rate limiting: {e}", "ERROR")

    async def _initialize_market_data(self) -> None:
        """Initialize market metadata and instruments."""
        try:
            # Refresh instruments
            await self.market_metadata.refresh_instruments()

            # Set up precision configurations
            await self._setup_precision_configs()

            # Set up risk tiers
            await self._setup_risk_tiers()

        except Exception as e:
            tprint(f"Failed to initialize market data: {e}", "ERROR")
            raise

    async def _setup_precision_configs(self) -> None:
        """Set up precision configurations for symbols."""
        try:
            from ...exchanges.shared.market.precision_helper import PrecisionConfig

            instruments = self.market_metadata.get_active_instruments()
            for instrument in instruments:
                config = PrecisionConfig(
                    symbol=instrument.symbol,
                    price_precision=instrument.price_precision,
                    quantity_precision=instrument.quantity_precision,
                    tick_size=instrument.tick_size,
                    lot_size=instrument.lot_size,
                    min_notional=instrument.min_notional,
                    max_notional=instrument.max_notional
                )
                self.precision_helper.set_precision_config(config)

        except Exception as e:
            tprint(f"Failed to setup precision configs: {e}", "ERROR")

    async def _setup_risk_tiers(self) -> None:
        """Set up risk tiers for symbols."""
        try:
            from ...exchanges.shared.market.risk_tier_manager import SymbolRiskProfile, RiskTier

            instruments = self.market_metadata.get_active_instruments()
            for instrument in instruments:
                # Determine risk tier based on leverage
                if instrument.max_leverage and instrument.max_leverage >= 20:
                    risk_tier = RiskTier.EXTREME
                elif instrument.max_leverage and instrument.max_leverage >= 10:
                    risk_tier = RiskTier.HIGH
                elif instrument.max_leverage and instrument.max_leverage >= 5:
                    risk_tier = RiskTier.MEDIUM
                else:
                    risk_tier = RiskTier.LOW

                profile = SymbolRiskProfile(
                    symbol=instrument.symbol,
                    risk_tier=risk_tier,
                    max_leverage=instrument.max_leverage or 1.0,
                    max_position_size=1000000.0,  # Default
                    max_notional=2000000.0,  # Default
                    margin_ratio=instrument.margin_ratio or 0.1,
                    liquidation_ratio=instrument.liquidation_ratio or 0.8,
                    maintenance_margin=0.05,  # Default
                    initial_margin=0.1  # Default
                )

                self.risk_tier_manager.set_symbol_risk_profile(profile)

        except Exception as e:
            tprint(f"Failed to setup risk tiers: {e}", "ERROR")

    @handle_async_errors(default_return=None)
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        try:
            # Close all data streams
            for symbol in list(self.ticker_streams.keys()):
                try:
                    stream = self.ticker_streams.pop(symbol, None)
                    if stream and hasattr(stream, 'close'):
                        await stream.close()
                except Exception as e:
                    tprint_warning(f"⚠️ Error closing ticker stream for {symbol}: {e}")
            
            for symbol in list(self.order_book_streams.keys()):
                try:
                    stream = self.order_book_streams.pop(symbol, None)
                    if stream and hasattr(stream, 'close'):
                        await stream.close()
                except Exception as e:
                    tprint_warning(f"⚠️ Error closing order book stream for {symbol}: {e}")
            
            for symbol in list(self.kline_streams.keys()):
                for interval in list(self.kline_streams[symbol].keys()):
                    try:
                        stream = self.kline_streams[symbol].pop(interval, None)
                        if stream and hasattr(stream, 'close'):
                            await stream.close()
                    except Exception as e:
                        tprint_warning(f"⚠️ Error closing kline stream for {symbol}/{interval}: {e}")
            
            # Close dispatcher
            if self.dispatcher:
                await self.dispatcher.close()
                self.dispatcher = None

            # Close shared utilities
            for manager in [self.auth_manager, self.market_manager, self.order_manager,
                          self.risk_manager, self.balance_manager, self.rate_limit_manager]:
                if manager and hasattr(manager, 'close'):
                    try:
                        manager.close()
                    except Exception as e:
                        tprint_warning(f"⚠️ Error closing manager: {e}")

            self.connection_status = ConnectionStatus.DISCONNECTED
            tprint(f"📴 Disconnected from {self.exchange_type}", "INFO")

        except Exception as e:
            tprint(f"❌ Error during disconnect: {e}", "ERROR")
            self.connection_status = ConnectionStatus.DISCONNECTED

    # Exchange-specific methods for shared utilities
    async def _get_server_time(self) -> Optional[int]:
        """Get server time in milliseconds."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_server_time()
        except Exception as e:
            tprint(f"Error getting server time: {e}", "ERROR")
        return None

    async def _test_connection(self) -> bool:
        """Test connection to exchange."""
        try:
            if self.dispatcher:
                return await self.dispatcher.test_connection()
        except Exception as e:
            tprint(f"Connection test failed: {e}", "ERROR")
        return False

    async def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_account_info()
        except Exception as e:
            tprint(f"Error getting account info: {e}", "ERROR")
        return None

    async def _get_instruments(self) -> List[Dict[str, Any]]:
        """Get instrument specifications."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_instruments()
        except Exception as e:
            tprint(f"Error getting instruments: {e}", "ERROR")
        return []

    async def _get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for symbol."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_ticker(symbol)
        except Exception as e:
            tprint(f"Error getting ticker for {symbol}: {e}", "ERROR")
        return None

    async def _get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book for symbol."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_order_book(symbol, limit)
        except Exception as e:
            tprint(f"Error getting order book for {symbol}: {e}", "ERROR")
        return None

    async def _get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for symbol."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_recent_trades(symbol, limit)
        except Exception as e:
            tprint(f"Error getting recent trades for {symbol}: {e}", "ERROR")
        return []

    async def _get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List[Any]]:
        """Get kline data for symbol."""
        try:
            if self.dispatcher:
                ohlcv_data = await self.dispatcher.get_ohlcv(symbol, interval, limit)
                return [[candle.timestamp.timestamp() * 1000, candle.open, candle.high,
                        candle.low, candle.close, candle.volume] for candle in ohlcv_data]
        except Exception as e:
            tprint(f"Error getting klines for {symbol}: {e}", "ERROR")
        return []

    async def _get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[List[Any]]:
        """Get historical kline data."""
        try:
            if self.dispatcher:
                ohlcv_data = await self.dispatcher.get_historical_ohlcv(
                    symbol, interval, start_time, end_time, limit
                )
                return [[candle.timestamp.timestamp() * 1000, candle.open, candle.high,
                        candle.low, candle.close, candle.volume] for candle in ohlcv_data]
        except Exception as e:
            tprint(f"Error getting historical klines for {symbol}: {e}", "ERROR")
        return []

    async def _get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get funding rate for symbol."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_funding_rate(symbol)
        except Exception as e:
            tprint(f"Error getting funding rate for {symbol}: {e}", "ERROR")
        return None

    async def _get_balances_exchange(self, account_type: str) -> List[Dict[str, Any]]:
        """Get balances for account type."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_balances(account_type)
        except Exception as e:
            tprint(f"Error getting balances: {e}", "ERROR")
        return []

    async def _create_order_exchange(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create order on exchange."""
        try:
            if self.dispatcher:
                return await self.dispatcher.create_order(
                    symbol, side, order_type, quantity, price, stop_price, client_order_id
                )
        except Exception as e:
            tprint(f"Error creating order: {e}", "ERROR")
        return None

    async def _cancel_order_exchange(self, order_id: str) -> bool:
        """Cancel order on exchange."""
        try:
            if self.dispatcher:
                return await self.dispatcher.cancel_order(order_id)
        except Exception as e:
            tprint(f"Error canceling order: {e}", "ERROR")
        return False

    async def _get_order_status_exchange(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status from exchange."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_order_status(order_id)
        except Exception as e:
            tprint(f"Error getting order status: {e}", "ERROR")
        return None

    async def _get_open_orders_exchange(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders from exchange."""
        try:
            if self.dispatcher:
                return await self.dispatcher.get_open_orders(symbol)
        except Exception as e:
            tprint(f"Error getting open orders: {e}", "ERROR")
        return []

    async def is_connected(self) -> bool:
        """Check if connected to exchange."""
        if self.exchange_type == 'simulated':
            return self.connection_status == ConnectionStatus.CONNECTED

        if self.dispatcher:
            return await self.dispatcher.is_connected()

        return False

    @handle_async_errors(default_return=None)
    async def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get ticker data for symbol."""
        try:
            if self.exchange_type == 'simulated':
                return await self._get_simulated_ticker(symbol)

            # Use shared market manager if available
            if self.market_manager:
                market_data = await self.market_manager.get_market_data(symbol)
                if market_data:
                    return TickerData(
                        symbol=symbol,
                        price=market_data.get('price', 0),
                        bid_price=market_data.get('bid', 0),
                        ask_price=market_data.get('ask', 0),
                        bid_quantity=market_data.get('bidQty', 0),
                        ask_quantity=market_data.get('askQty', 0),
                        volume_24h=market_data.get('volume', 0),
                        price_change_24h=market_data.get('change', 0),
                        price_change_percent_24h=market_data.get('changePercent', 0),
                        high_24h=market_data.get('high', 0),
                        low_24h=market_data.get('low', 0),
                        timestamp=datetime.now()
                    )

            # Fallback to dispatcher
            if self.dispatcher:
                ticker_data = await self.dispatcher.get_ticker(symbol)
                if ticker_data:
                    return TickerData(
                        symbol=symbol,
                        price=ticker_data.get('price', 0),
                        bid_price=ticker_data.get('bid', 0),
                        ask_price=ticker_data.get('ask', 0),
                        bid_quantity=ticker_data.get('bidQty', 0),
                        ask_quantity=ticker_data.get('askQty', 0),
                        volume_24h=ticker_data.get('volume', 0),
                        price_change_24h=ticker_data.get('change', 0),
                        price_change_percent_24h=ticker_data.get('changePercent', 0),
                        high_24h=ticker_data.get('high', 0),
                        low_24h=ticker_data.get('low', 0),
                        timestamp=datetime.now()
                    )

            return None

        except Exception as e:
            tprint(f"❌ Error getting ticker for {symbol}: {e}", "ERROR")
            return None

    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Get order book for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_order_book(symbol, limit)

        if self.dispatcher:
            return await self.dispatcher.get_order_book(symbol, limit)

        return None

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[KlineData]:
        """Get kline data for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_klines(symbol, interval, start_time, end_time, limit)

        if self.dispatcher:
            ohlcv_data = await self.dispatcher.get_ohlcv(symbol, interval, limit)
            klines = []
            for candle in ohlcv_data:
                klines.append(KlineData(
                    symbol=candle.symbol,
                    interval=interval,
                    timestamp=candle.timestamp,
                    open_price=candle.open,
                    high_price=candle.high,
                    low_price=candle.low,
                    close_price=candle.close,
                    volume=candle.volume,
                    close_time=candle.timestamp,
                    quote_asset_volume=candle.volume * candle.close,
                    number_of_trades=0,
                    taker_buy_base_asset_volume=candle.volume * 0.5,
                    taker_buy_quote_asset_volume=candle.volume * candle.close * 0.5
                ))
            return klines

        return []

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get recent trades for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_recent_trades(symbol, limit)

        # For real exchanges, this would be implemented in the dispatcher
        return []

    @handle_async_errors(default_return={})
    async def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """Get account balance."""
        try:
            if self.exchange_type == 'simulated':
                return self._get_simulated_balance(asset)

            # Use shared balance manager if available
            if self.balance_manager:
                if asset:
                    balance = await self.balance_manager.get_balance(asset)
                    return {asset: balance} if balance is not None else {}
                else:
                    return await self.balance_manager.get_all_balances()

            # Fallback to dispatcher
            if self.dispatcher:
                if asset:
                    balance = await self.dispatcher.get_balance(asset)
                    return {asset: balance}
                else:
                    # Get all balances - this would need to be implemented in dispatcher
                    return {}

            return {}

        except Exception as e:
            tprint(f"❌ Error getting account balance: {e}", "ERROR")
            return {}

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
        """Create order."""
        try:
            if self.exchange_type == 'simulated':
                return await self._create_simulated_order(symbol, side, order_type, quantity, price)

            # Use shared order manager if available
            if self.order_manager:
                # Validate order parameters first
                order_params = {
                    'symbol': symbol,
                    'side': side,
                    'order_type': order_type,
                    'quantity': quantity,
                    'price': price,
                    **kwargs
                }

                validation_result = self.order_manager.validate_order_params(order_params)
                if not validation_result.is_valid:
                    tprint(f"❌ Order validation failed: {validation_result.errors}", "ERROR")
                    return {'error': 'validation_failed', 'errors': validation_result.errors}

                # Create order using shared manager
                order_id = await self.order_manager.create_order(
                    symbol, side, order_type, quantity, **kwargs
                )

                if order_id:
                    return {
                        'orderId': order_id,
                        'symbol': symbol,
                        'side': side,
                        'type': order_type,
                        'quantity': quantity,
                        'price': price,
                        'status': 'NEW'
                    }
                else:
                    tprint(f"❌ Failed to create order for {symbol}", "ERROR")
                    return {'error': 'order_creation_failed'}

            # Fallback to dispatcher
            if self.dispatcher:
                result = await self.dispatcher.create_order(symbol, side, order_type, quantity, price)
                return result or {}

            return {'error': 'no_order_manager_available'}

        except Exception as e:
            tprint(f"❌ Error creating order for {symbol}: {e}", "ERROR")
            return {'error': str(e)}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        if self.exchange_type == 'simulated':
            return await self._cancel_simulated_order(symbol, order_id)

        if self.dispatcher:
            return await self.dispatcher.cancel_order(symbol, order_id)

        return False

    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        if self.exchange_type == 'simulated':
            return self._get_simulated_order_status(symbol, order_id)

        if self.dispatcher:
            result = await self.dispatcher.get_order_status(symbol, order_id)
            return result or {}

        return {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        if self.exchange_type == 'simulated':
            return self._get_simulated_open_orders(symbol)

        if self.dispatcher:
            return await self.dispatcher.get_open_orders(symbol)

        return []

    # Simulated exchange methods
    async def _get_simulated_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get simulated ticker data."""
        # Use the existing simulated exchange logic
        if symbol not in self.price_feeds:
            return None

        data = self.price_feeds[symbol]
        price_variation = np.random.normal(0, data['price'] * 0.001)
        current_price = data['price'] + price_variation

        return TickerData(
            symbol=symbol,
            price=current_price,
            bid_price=current_price - 0.5,
            ask_price=current_price + 0.5,
            bid_quantity=np.random.uniform(1, 10),
            ask_quantity=np.random.uniform(1, 10),
            volume_24h=data['volume_24h'],
            price_change_24h=data['price_change_24h'],
            price_change_percent_24h=(data['price_change_24h'] / data['price']) * 100,
            high_24h=data['high_24h'],
            low_24h=data['low_24h'],
            timestamp=datetime.now()
        )

    async def _get_simulated_order_book(self, symbol: str, limit: int) -> Optional[Dict[str, Any]]:
        """Get simulated order book."""
        if symbol not in self.price_feeds:
            return None

        data = self.price_feeds[symbol]
        base_price = data['price']

        bids = []
        asks = []

        for i in range(limit):
            bid_price = base_price - 0.5 - (i * 0.1)
            ask_price = base_price + 0.5 + (i * 0.1)

            bid_quantity = np.random.uniform(0.1, 5.0)
            ask_quantity = np.random.uniform(0.1, 5.0)

            bids.append([bid_price, bid_quantity])
            asks.append([ask_price, ask_quantity])

        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }

    async def _get_simulated_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> List[KlineData]:
        """Get simulated kline data."""
        if symbol not in self.price_feeds:
            return []

        data = self.price_feeds[symbol]
        base_price = data['price']
        klines = []
        current_time = datetime.now()

        for i in range(min(limit, 500)):
            timestamp = current_time - timedelta(minutes=i)
            open_price = base_price + np.random.normal(0, base_price * 0.02)
            high_price = open_price + abs(np.random.normal(0, base_price * 0.01))
            low_price = open_price - abs(np.random.normal(0, base_price * 0.01))
            close_price = low_price + np.random.uniform(0, high_price - low_price)
            volume = np.random.uniform(100, 1000)

            klines.append(KlineData(
                symbol=symbol,
                interval=interval,
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                close_time=timestamp + timedelta(minutes=1),
                quote_asset_volume=close_price * volume,
                number_of_trades=int(np.random.uniform(10, 100)),
                taker_buy_base_asset_volume=volume * np.random.uniform(0.3, 0.7),
                taker_buy_quote_asset_volume=close_price * volume * np.random.uniform(0.3, 0.7)
            ))

        return klines

    async def _get_simulated_recent_trades(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Get simulated recent trades."""
        if symbol not in self.price_feeds:
            return []

        data = self.price_feeds[symbol]
        base_price = data['price']
        trades = []

        for i in range(min(limit, 500)):
            timestamp = datetime.now() - timedelta(seconds=i)
            price = base_price + np.random.normal(0, base_price * 0.001)
            quantity = np.random.uniform(0.01, 1.0)

            trades.append({
                'id': f'sim_trade_{i}',
                'price': price,
                'qty': quantity,
                'quoteQty': price * quantity,
                'time': timestamp.isoformat(),
                'isBuyerMaker': np.random.choice([True, False]),
                'isBestMatch': True
            })

        return trades

    def _get_simulated_balance(self, asset: Optional[str]) -> Dict[str, float]:
        """Get simulated account balance."""
        if asset:
            return {asset: 1000.0 if asset == 'USDT' else 10.0}

        return {
            'USDT': 10000.0,
            'ETH': 10.0,
            'BTC': 1.0
        }

    async def _create_simulated_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Create simulated order."""
        order_id = f'sim_order_{len(self.simulated_orders)}'

        order_data = {
            'symbol': symbol,
            'orderId': order_id,
            'orderListId': -1,
            'clientOrderId': f'client_{order_id}',
            'price': price,
            'origQty': quantity,
            'executedQty': quantity,
            'cummulativeQuoteQty': price * quantity if price else 3000.0 * quantity,
            'status': 'FILLED',
            'timeInForce': 'GTC',
            'type': order_type,
            'side': side,
            'workingTime': datetime.now().isoformat(),
            'selfTradePreventionMode': 'NONE'
        }

        self.simulated_orders[order_id] = order_data
        return order_data

    async def _cancel_simulated_order(self, symbol: str, order_id: str) -> bool:
        """Cancel simulated order."""
        if order_id in self.simulated_orders:
            self.simulated_orders[order_id]['status'] = 'CANCELLED'
            return True
        return False

    def _get_simulated_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get simulated order status."""
        return self.simulated_orders.get(order_id, {})

    def _get_simulated_open_orders(self, symbol: Optional[str]) -> List[Dict[str, Any]]:
        """Get simulated open orders."""
        open_orders = []

        for order_id, order_data in self.simulated_orders.items():
            if order_data['status'] in ['NEW', 'PARTIALLY_FILLED']:
                open_orders.append(order_data)

        return open_orders

    @handle_errors(default_return=False)
    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if request is within rate limits."""
        try:
            # Use shared rate limit manager if available
            if self.rate_limit_manager:
                return not self.rate_limit_manager.is_limited(endpoint)

            # Fallback to time-windowed rate limiting implementation
            now = datetime.now()
            
            # Reset counters if time window has passed
            if endpoint in self.last_requests:
                time_since_last = (now - self.last_requests[endpoint]).total_seconds()
                # Reset every minute
                if time_since_last >= 60:
                    self.request_counts[endpoint] = 0
                    self.last_requests[endpoint] = now
            
            # Check if we're within limits
            limit = self.rate_limits.get(endpoint, 100)
            current_count = self.request_counts.get(endpoint, 0)
            
            if current_count >= limit:
                tprint_warning(f"⚠️ Rate limit exceeded for {endpoint}: {current_count}/{limit}")
                return False
            
            return True

        except Exception as e:
            tprint(f"❌ Error checking rate limit: {e}", "ERROR")
            return False  # Fail closed for safety

    @handle_async_errors(default_return={})
    async def get_risk_info(self, symbol: str, position_size: float, current_price: float, leverage: float = 1.0) -> Dict[str, Any]:
        """Get risk information for a position."""
        try:
            if self.risk_manager:
                return self.risk_manager.calculate_position_risk(
                    symbol, position_size, current_price, leverage
                )
            return {}

        except Exception as e:
            tprint(f"❌ Error getting risk info for {symbol}: {e}", "ERROR")
            return {}

    @handle_async_errors(default_return=[])
    async def get_portfolio_risk(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get portfolio risk information."""
        try:
            if self.risk_manager:
                return self.risk_manager.calculate_portfolio_risk(positions)
            return {}

        except Exception as e:
            tprint(f"❌ Error getting portfolio risk: {e}", "ERROR")
            return {}

    def _update_rate_limit(self, endpoint: str) -> None:
        """Update rate limit counters."""
        now = datetime.now()

        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = 0
            self.last_requests[endpoint] = now

        self.request_counts[endpoint] += 1
        self.last_requests[endpoint] = now
        self.total_requests += 1

    @handle_async_errors(default_return=None)
    async def _handle_error(self, error: Exception, operation: str) -> None:
        """Handle exchange errors."""
        try:
            self.connection_errors.append({
                'timestamp': datetime.now(),
                'operation': operation,
                'error': str(error)
            })

            self.failed_requests += 1

            if len(self.connection_errors) > 10:
                self.connection_status = ConnectionStatus.ERROR

            tprint(f"❌ Exchange error in {operation}: {str(error)}", "ERROR")

        except Exception as e:
            tprint(f"❌ Error in error handler: {e}", "ERROR")

class SimulatedExchange:
    """
    Stub class for SimulatedExchange - provides simulated trading functionality.
    
    This class simulates exchange operations for testing and development purposes.
    It maintains internal state for orders, balances, and positions without
    connecting to real exchanges.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simulated exchange with configuration."""
        self.config = config
        self.exchange_type = config.get('exchange_type', 'simulated')
        self.symbol = config.get('symbol', 'BTCUSDT')
        
        # Simulated account state
        self.balances = {
            'USDT': 10000.0,  # Starting with 10k USDT
            'BTC': 0.0,
            'ETH': 0.0
        }
        
        # Simulated order book
        self.orders = {}
        self.order_counter = 0
        
        # Simulated positions
        self.positions = {}
        
        # Simulated market data
        self.current_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0
        }
        
        # Connection state
        self.is_connected = False
        self.last_update = datetime.now()
        
        self.logger = logger.getChild('SimulatedExchange')
        self.logger.info(f"SimulatedExchange initialized for {self.symbol}")
    
    async def connect(self) -> bool:
        """Simulate connection to exchange."""
        try:
            self.is_connected = True
            self.logger.info("Connected to simulated exchange")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to simulated exchange: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Simulate disconnection from exchange."""
        self.is_connected = False
        self.logger.info("Disconnected from simulated exchange")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get simulated account information."""
        total_balance = sum(self.balances.values())
        return {
            'totalWalletBalance': str(total_balance),
            'balances': self.balances.copy(),
            'accountType': 'SPOT',
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True
        }
    
    async def get_balance(self, asset: str) -> float:
        """Get balance for specific asset."""
        return self.balances.get(asset, 0.0)
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get simulated open positions."""
        positions = []
        for symbol, position in self.positions.items():
            if position['amount'] != 0:
                positions.append({
                    'symbol': symbol,
                    'positionAmt': str(position['amount']),
                    'entryPrice': str(position['entry_price']),
                    'leverage': str(position.get('leverage', 1)),
                    'unrealizedPnl': str(position.get('unrealized_pnl', 0.0)),
                    'markPrice': str(self.current_prices.get(symbol, 0.0))
                })
        return positions
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create simulated order."""
        if not self.is_connected:
            raise Exception("Not connected to exchange")
        
        self.order_counter += 1
        order_id = f"sim_order_{self.order_counter}"
        
        # Simulate order execution
        execution_price = price or self.current_prices.get(symbol, 0.0)
        
        order = {
            'orderId': order_id,
            'symbol': symbol,
            'status': 'FILLED',  # Simulate immediate fill
            'type': order_type,
            'side': side,
            'origQty': str(quantity),
            'executedQty': str(quantity),
            'price': str(execution_price),
            'timeInForce': 'GTC',
            'time': int(datetime.now().timestamp() * 1000),
            'updateTime': int(datetime.now().timestamp() * 1000)
        }
        
        self.orders[order_id] = order
        
        # Update balances and positions
        await self._update_balances_and_positions(symbol, side, quantity, execution_price)
        
        self.logger.info(f"Created simulated order: {order_id} - {side} {quantity} {symbol} @ {execution_price}")
        return order
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel simulated order."""
        if order_id in self.orders:
            self.orders[order_id]['status'] = 'CANCELLED'
            self.logger.info(f"Cancelled simulated order: {order_id}")
            return True
        return False
    
    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get simulated order status."""
        return self.orders.get(order_id, {})
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get simulated open orders."""
        open_orders = []
        for order in self.orders.values():
            if order['status'] in ['NEW', 'PARTIALLY_FILLED']:
                if symbol is None or order['symbol'] == symbol:
                    open_orders.append(order)
        return open_orders
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get simulated ticker data."""
        base_price = self.current_prices.get(symbol, 0.0)
        # Add some random variation
        variation = np.random.normal(0, base_price * 0.001)
        current_price = base_price + variation
        
        return {
            'symbol': symbol,
            'price': str(current_price),
            'bidPrice': str(current_price - 0.5),
            'askPrice': str(current_price + 0.5),
            'volume': str(np.random.uniform(1000, 10000)),
            'change': str(variation),
            'changePercent': str((variation / base_price) * 100) if base_price > 0 else '0'
        }
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[List[Any]]:
        """Get simulated kline data."""
        base_price = self.current_prices.get(symbol, 0.0)
        klines = []
        current_time = datetime.now()
        
        for i in range(min(limit, 500)):
            timestamp = current_time - timedelta(minutes=i)
            open_price = base_price + np.random.normal(0, base_price * 0.02)
            high_price = open_price + abs(np.random.normal(0, base_price * 0.01))
            low_price = open_price - abs(np.random.normal(0, base_price * 0.01))
            close_price = low_price + np.random.uniform(0, high_price - low_price)
            volume = np.random.uniform(100, 1000)
            
            klines.append([
                int(timestamp.timestamp() * 1000),  # Open time
                str(open_price),                    # Open
                str(high_price),                    # High
                str(low_price),                     # Low
                str(close_price),                   # Close
                str(volume),                        # Volume
                int(timestamp.timestamp() * 1000),  # Close time
                str(close_price * volume),          # Quote asset volume
                int(np.random.uniform(10, 100)),    # Number of trades
                str(volume * np.random.uniform(0.3, 0.7)),  # Taker buy base asset volume
                str(close_price * volume * np.random.uniform(0.3, 0.7)),  # Taker buy quote asset volume
                '0'  # Ignore
            ])
        
        return klines
    
    async def _update_balances_and_positions(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> None:
        """Update simulated balances and positions after order execution."""
        try:
            # Determine base and quote assets
            if symbol.endswith('USDT'):
                base_asset = symbol[:-4]
                quote_asset = 'USDT'
            else:
                base_asset = symbol[:3]
                quote_asset = symbol[3:]
            
            # Calculate cost
            cost = quantity * price
            
            if side.upper() == 'BUY':
                # Buying: reduce quote asset, increase base asset
                self.balances[quote_asset] -= cost
                self.balances[base_asset] += quantity
                
                # Update position
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'amount': 0.0,
                        'entry_price': 0.0,
                        'leverage': 1,
                        'unrealized_pnl': 0.0
                    }
                
                # Update position (simple average)
                current_amount = self.positions[symbol]['amount']
                current_entry = self.positions[symbol]['entry_price']
                
                if current_amount >= 0:  # Adding to long position
                    new_amount = current_amount + quantity
                    new_entry = ((current_amount * current_entry) + cost) / new_amount if new_amount > 0 else 0
                else:  # Reducing short position
                    new_amount = current_amount + quantity
                    new_entry = current_entry if new_amount < 0 else price
                
                self.positions[symbol]['amount'] = new_amount
                self.positions[symbol]['entry_price'] = new_entry
                
            else:  # SELL
                # Selling: reduce base asset, increase quote asset
                self.balances[base_asset] -= quantity
                self.balances[quote_asset] += cost
                
                # Update position
                if symbol not in self.positions:
                    self.positions[symbol] = {
                        'amount': 0.0,
                        'entry_price': 0.0,
                        'leverage': 1,
                        'unrealized_pnl': 0.0
                    }
                
                # Update position
                current_amount = self.positions[symbol]['amount']
                current_entry = self.positions[symbol]['entry_price']
                
                if current_amount <= 0:  # Adding to short position
                    new_amount = current_amount - quantity
                    new_entry = ((abs(current_amount) * current_entry) + cost) / abs(new_amount) if new_amount < 0 else 0
                else:  # Reducing long position
                    new_amount = current_amount - quantity
                    new_entry = current_entry if new_amount > 0 else price
                
                self.positions[symbol]['amount'] = new_amount
                self.positions[symbol]['entry_price'] = new_entry
            
            # Update unrealized PnL
            if symbol in self.positions and self.positions[symbol]['amount'] != 0:
                current_price = self.current_prices.get(symbol, price)
                entry_price = self.positions[symbol]['entry_price']
                amount = self.positions[symbol]['amount']
                
                if amount > 0:  # Long position
                    self.positions[symbol]['unrealized_pnl'] = (current_price - entry_price) * amount
                else:  # Short position
                    self.positions[symbol]['unrealized_pnl'] = (entry_price - current_price) * abs(amount)
            
            self.logger.debug(f"Updated balances and positions for {symbol}: {side} {quantity} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error updating balances and positions: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get simulated exchange status."""
        return {
            'is_connected': self.is_connected,
            'exchange_type': self.exchange_type,
            'symbol': self.symbol,
            'balances': self.balances.copy(),
            'positions': self.positions.copy(),
            'orders_count': len(self.orders),
            'last_update': self.last_update.isoformat()
        }

# Factory function for creating exchange interfaces
def create_exchange_interface(config: Dict[str, Any]) -> ExchangeInterface:
    """
    Create exchange interface.

    Args:
        config: Configuration dictionary

    Returns:
        Exchange interface instance
    """
    return ExchangeInterface(config)

def get_exchange_interface(exchange_type: str) -> Optional[ExchangeInterface]:
    """Get exchange interface by type."""
    # Placeholder for getting cached interface
    return None
