"""
BingX Exchange Implementation

This module provides a complete BingX exchange implementation that follows
the BaseExchange interface and integrates with the data collection system.
This module provides a comprehensive BingX exchange implementation that uses
the shared utilities for common functionality and adds all required features.
"""

import asyncio
import hashlib
import hmac
import time
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None

from src.interfaces.base_interfaces import MarketData
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

from .base_exchange import BaseExchange
from .shared import (
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
from .shared.interfaces_typed import (
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
    tprint, DataSource, ValidationResult
)


class BingXExchange(BaseExchange):
    """
    BingX exchange implementation following the BaseExchange interface.
    
    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data)
    - Aggregated trades
    - Futures funding rates
    """

    Enhanced BingX exchange implementation with comprehensive functionality.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        trade_symbol: str,
        password: str | None = None,
        subaccount_id: str | None = None,
        use_testnet: bool = False
    ) -> None:
        super().__init__(api_key, api_secret, trade_symbol, password)
        self.logger = system_logger.getChild("BingXExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://open-api.bingx.com"
        self.use_testnet = False  # Set to True for testing

    async def _initialize_exchange(self) -> None:
        """Initialize the BingX exchange client."""
        try:
            if aiohttp is None:
                self.logger.warning("⚠️ aiohttp not available, using mock session")
                self.session = None
                return

            # Initialize aiohttp session with SSL configuration
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            # Test connection
            await self._test_connection()

            self.logger.info("✅ BingX exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize BingX exchange: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to BingX API."""
        try:
            url = f"{self.base_url}/openApi/spot/v1/common/serverTime"
        self.base_url = "https://open-api.bingx.com" if not use_testnet else "https://open-api.bingx.com"
        self.use_testnet = use_testnet
        self.subaccount_id = subaccount_id
        
        # Initialize shared utilities
        self._initialize_shared_utilities()
        
        # Exchange-specific settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30
        
    def _initialize_shared_utilities(self) -> None:
        """Initialize all shared utilities."""
        try:
            # Authentication
            self.auth_manager = AuthenticationManager("bingx")
            self.api_key_manager = APIKeyManager("bingx")
            self.time_sync_manager = TimeSyncManager("bingx")
            self.subaccount_manager = SubaccountManager("bingx")
            
            # Market metadata
            self.market_metadata = MarketMetadataManager("bingx")
            self.instrument_manager = InstrumentManager("bingx")
            self.precision_helper = PrecisionHelper()
            self.risk_tier_manager = RiskTierManager("bingx")
            
            # Pricing
            self.price_manager = PriceManager("bingx")
            self.ohlcv_manager = OHLCVManager("bingx")
            
            # Orders
            self.order_manager = OrderManager("bingx")
            self.idempotency_manager = IdempotencyManager("bingx")
            
            # Risk
            self.risk_calculator = RiskCalculator("bingx")
            
            # Wallet
            self.balance_manager = BalanceManager("bingx")
            
            # Reliability
            self.rate_limit_manager = RateLimitManager("bingx")
            self.audit_logger = AuditLogger("bingx")
            
            # Register exchange-specific functions
            self._register_exchange_functions()
            
        except Exception as e:
            tprint(f"Failed to initialize shared utilities: {e}", "ERROR")
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
            
            # Set up rate limiting
            self._setup_rate_limiting()
            
        except Exception as e:
            tprint(f"Failed to register exchange functions: {e}", "ERROR")
            raise
        
    def _setup_rate_limiting(self) -> None:
        """Set up rate limiting for different endpoints."""
        try:
            from .shared.reliability.rate_limit_manager import RateLimit, RateLimitStrategy
            
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
    
    async def _initialize_exchange(self) -> None:
        """Initialize the enhanced BingX exchange client."""
        try:
            if aiohttp is None:
                tprint("aiohttp not available, using mock session", "WARNING")
                self.session = None
                return

            # Initialize aiohttp session
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            # Authenticate
            await self._authenticate()
            
            # Initialize market data
            await self._initialize_market_data()
            
            # Start background tasks
            await self._start_background_tasks()

            tprint("BingX exchange initialized successfully", "INFO")

        except Exception as e:
            tprint(f"Failed to initialize BingX exchange: {e}", "ERROR")
            raise
    
    async def _authenticate(self) -> None:
        """Authenticate with BingX using shared auth manager."""
        try:
            from .shared.auth.auth_manager import AuthConfig, APIKeyPermission
            
            # Create auth config
            auth_config = AuthConfig(
                exchange_name="bingx",
                api_key=self.api_key,
                api_secret=self.api_secret,
                passphrase=self.password,
                subaccount_id=self.subaccount_id,
                permissions={APIKeyPermission.READ, APIKeyPermission.TRADE},
                auto_sync_time=True
            )
            
            # Authenticate
            success = await self.auth_manager.authenticate(auth_config)
            if not success:
                raise Exception("Failed to authenticate with BingX")
                
        except Exception as e:
            tprint(f"Authentication failed: {e}", "ERROR")
            raise
    
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
            from .shared.market.precision_helper import PrecisionConfig
            
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
            from .shared.market.risk_tier_manager import SymbolRiskProfile, RiskTier
            
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
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for data synchronization."""
        try:
            # Start time sync
            if self.auth_manager.is_authenticated_and_valid():
                await self.time_sync_manager.start_auto_sync(self._get_server_time)
            
            # Start market data refresh
            asyncio.create_task(self._background_market_data_refresh())
            
            # Start order sync
            asyncio.create_task(self._background_order_sync())
            
        except Exception as e:
            tprint(f"Failed to start background tasks: {e}", "ERROR")
    
    async def _background_market_data_refresh(self) -> None:
        """Background task to refresh market data."""
        while True:
            try:
                if self.session:
                    # Refresh market data for active symbols
                    symbols = [self.trade_symbol]
                    await self.market_metadata.refresh_market_data(symbols)
                
                await asyncio.sleep(30)  # Refresh every 30 seconds
            except Exception as e:
                tprint(f"Error in background market data refresh: {e}", "ERROR")
                await asyncio.sleep(60)
    
    async def _background_order_sync(self) -> None:
        """Background task to sync orders."""
        while True:
            try:
                if self.session:
                    await self.order_manager.sync_orders_from_exchange()
                
                await asyncio.sleep(10)  # Sync every 10 seconds
            except Exception as e:
                tprint(f"Error in background order sync: {e}", "ERROR")
                await asyncio.sleep(30)
    
    # Exchange-specific API methods
    async def _get_server_time(self) -> Optional[int]:
        """Get server time in milliseconds."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/server/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("data", {}).get("serverTime")
                    self.logger.info(f"Connected to BingX API (Server time: {server_time})")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise

    def _generate_signature(self, params: dict[str, Any]) -> str:
        """Generate HMAC signature for authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret not configured")
        
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to BingX API."""
        if aiohttp is None or not self.session:
            self.logger.warning("⚠️ aiohttp not available, returning mock data")
            return []

        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}

        headers = {}
        if signed and self.api_key:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)
            headers["X-BX-APIKEY"] = self.api_key

        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # BingX wraps responses in a data field
                    return data.get("data", data)
                else:
                    error_text = await response.text()
                    self.logger.error(f"API request failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return None

    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Convert raw BingX kline data to standardized MarketData format."""
        market_data_list = []
        
        for item in raw_data:
            try:
                # BingX klines format: [open_time, open, high, low, close, volume, close_time, ...]
                if isinstance(item, list):
                    timestamp = datetime.fromtimestamp(item[0] / 1000)
                    open_price = float(item[1])
                    high_price = float(item[2])
                    low_price = float(item[3])
                    close_price = float(item[4])
                    volume = float(item[5])
                else:
                    # Dict format
                    timestamp = self._convert_timestamp(item.get("timestamp", item.get("open_time", 0)))
                    open_price = float(item.get("open", 0))
                    high_price = float(item.get("high", 0))
                    low_price = float(item.get("low", 0))
                    close_price = float(item.get("close", 0))
                    volume = float(item.get("volume", 0))

                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    interval=interval
                )
                market_data_list.append(market_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert kline data: {e}")
                continue

        return market_data_list

    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol (BingX uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from BingX."""
        params = {
            "symbol": symbol.upper(),
            "interval": self._convert_interval(interval),
            "limit": min(limit, 1000)  # BingX max limit is 1000
        }
        
        data = await self._make_request("GET", "/openApi/spot/v1/market/klines", params)
        if data:
            # Convert list format to dict format for consistency
            klines = []
            for item in data:
                klines.append({
                    "timestamp": item[0],
                    "open_time": item[0],
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": item[6],
                    "quote_volume": item[7],
                    "trades": item[8],
                    "taker_buy_base": item[9],
                    "taker_buy_quote": item[10]
                })
            return klines
        return []

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical kline data from BingX."""
        params = {
            "symbol": symbol.upper(),
            "interval": self._convert_interval(interval),
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/openApi/spot/v1/market/klines", params)
        if data:
            # Convert list format to dict format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": item[0],
                    "open_time": item[0],
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": item[6],
                    "quote_volume": item[7],
                    "trades": item[8],
                    "taker_buy_base": item[9],
                    "taker_buy_quote": item[10]
                })
            return klines
        return []

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from BingX."""
        params = {
            "symbol": symbol.upper(),
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/openApi/spot/v1/market/aggTrades", params)
        if data:
            # Standardize field names
            trades = []
            for item in data:
                trades.append({
                    "timestamp": item.get("T", item.get("timestamp", 0)),
                    "price": item.get("p", item.get("price", 0)),
                    "quantity": item.get("q", item.get("quantity", 0)),
                    "is_buyer_maker": item.get("m", item.get("is_buyer_maker", False)),
                    "trade_id": item.get("a", item.get("trade_id", 0)),
                    "first_trade_id": item.get("f", item.get("first_trade_id", 0)),
                    "last_trade_id": item.get("l", item.get("last_trade_id", 0))
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from BingX."""
        return await self._make_request("GET", "/openApi/spot/v1/account", signed=True) or {}

    async def _create_order_raw(
                    return int(server_time) if server_time else None
        except Exception as e:
            tprint(f"Error getting server time: {e}", "ERROR")
        return None
    
    async def _test_connection(self) -> bool:
        """Test connection to BingX API."""
        try:
            server_time = await self._get_server_time()
            return server_time is not None
        except Exception as e:
            tprint(f"Connection test failed: {e}", "ERROR")
            return False
    
    async def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        try:
            headers = self.auth_manager.get_auth_headers("GET", "/openApi/swap/v2/user/balance")
            if not headers:
                return None
            
            url = f"{self.base_url}/openApi/swap/v2/user/balance"
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
        except Exception as e:
            tprint(f"Error getting account info: {e}", "ERROR")
        return None
    
    async def _get_instruments(self) -> List[Dict[str, Any]]:
        """Get instrument specifications."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/quote/contracts"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            tprint(f"Error getting instruments: {e}", "ERROR")
        return []
    
    async def _get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for symbol."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/quote/ticker"
            params = {"symbol": symbol.upper()}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            tprint(f"Error getting ticker for {symbol}: {e}", "ERROR")
        return None
    
    async def _get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book for symbol."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/quote/depth"
            params = {"symbol": symbol.upper(), "limit": str(limit)}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            tprint(f"Error getting order book for {symbol}: {e}", "ERROR")
        return None
    
    async def _get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for symbol."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/quote/trades"
            params = {"symbol": symbol.upper(), "limit": str(limit)}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            tprint(f"Error getting recent trades for {symbol}: {e}", "ERROR")
        return []
    
    async def _get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List[Any]]:
        """Get kline data for symbol."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/quote/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": self._convert_interval(interval),
                "limit": str(min(limit, 1000))
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
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
            url = f"{self.base_url}/openApi/swap/v2/quote/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": self._convert_interval(interval),
                "startTime": str(int(start_time.timestamp() * 1000)),
                "endTime": str(int(end_time.timestamp() * 1000)),
                "limit": str(min(limit, 1000))
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            tprint(f"Error getting historical klines for {symbol}: {e}", "ERROR")
        return []
    
    async def _get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get funding rate for symbol."""
        try:
            url = f"{self.base_url}/openApi/swap/v2/quote/fundingRate"
            params = {"symbol": symbol.upper()}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            tprint(f"Error getting funding rate for {symbol}: {e}", "ERROR")
        return None
    
    async def _get_balances_exchange(self, account_type: str) -> List[Dict[str, Any]]:
        """Get balances for account type."""
        try:
            headers = self.auth_manager.get_auth_headers("GET", "/openApi/swap/v2/user/balance")
            if not headers:
                return []
            
            url = f"{self.base_url}/openApi/swap/v2/user/balance"
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            tprint(f"Error getting balances: {e}", "ERROR")
        return []
    
    async def _create_order_exchange(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on BingX."""
        order_params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity
        }
        
        if price is not None:
            order_params["price"] = price
            
        if params:
            order_params.update(params)
            
        return await self._make_request("POST", "/openApi/spot/v1/trade/order", order_params, signed=True) or {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from BingX futures."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/swap/v2/user/positions", params, signed=True)
        
        if data and isinstance(data, list):
            # Return first matching position or first position if no symbol specified
            for position in data:
                if not symbol or position.get("symbol", "").upper() == symbol.upper():
                    return position
            return data[0] if data else {}
        
        return data or {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from BingX."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/spot/v1/trade/openOrders", params, signed=True)
        return data if isinstance(data, list) else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on BingX."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("DELETE", "/openApi/spot/v1/trade/order", params, signed=True) or {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from BingX."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("GET", "/openApi/spot/v1/trade/query", params, signed=True) or {}

        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create order on exchange."""
        try:
            # Generate idempotency key
            idempotency_key = self.idempotency_manager.create_order_key(
                symbol, side, order_type, quantity, price, client_order_id
            )
            
            # Prepare order parameters
            order_params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": str(quantity)
            }
            
            if price is not None:
                order_params["price"] = str(price)
            if stop_price is not None:
                order_params["stopPrice"] = str(stop_price)
            if client_order_id:
                order_params["clientOrderId"] = client_order_id
            
            # Get auth headers
            headers = self.auth_manager.get_auth_headers("POST", "/openApi/swap/v2/trade/order")
            if not headers:
                return None
            
            url = f"{self.base_url}/openApi/swap/v2/trade/order"
            async with self.session.post(url, json=order_params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
                else:
                    error_text = await response.text()
                    tprint(f"Order creation failed: {response.status} - {error_text}", "ERROR")
                    return None
                    
        except Exception as e:
            tprint(f"Error creating order: {e}", "ERROR")
        return None
    
    async def _cancel_order_exchange(self, order_id: str) -> bool:
        """Cancel order on exchange."""
        try:
            headers = self.auth_manager.get_auth_headers("DELETE", "/openApi/swap/v2/trade/order")
            if not headers:
                return False
            
            url = f"{self.base_url}/openApi/swap/v2/trade/order"
            params = {"orderId": order_id}
            async with self.session.delete(url, params=params, headers=headers) as response:
                if response.status == 200:
                    return True
                else:
                    error_text = await response.text()
                    tprint(f"Order cancellation failed: {response.status} - {error_text}", "ERROR")
                    return False
                    
        except Exception as e:
            tprint(f"Error canceling order: {e}", "ERROR")
        return False
    
    async def _get_order_status_exchange(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status from exchange."""
        try:
            headers = self.auth_manager.get_auth_headers("GET", "/openApi/swap/v2/trade/order")
            if not headers:
                return None
            
            url = f"{self.base_url}/openApi/swap/v2/trade/order"
            params = {"orderId": order_id}
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
                else:
                    error_text = await response.text()
                    tprint(f"Order status check failed: {response.status} - {error_text}", "ERROR")
                    return None
                    
        except Exception as e:
            tprint(f"Error getting order status: {e}", "ERROR")
        return None
    
    async def _get_open_orders_exchange(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders from exchange."""
        try:
            headers = self.auth_manager.get_auth_headers("GET", "/openApi/swap/v2/trade/openOrders")
            if not headers:
                return []
            
            url = f"{self.base_url}/openApi/swap/v2/trade/openOrders"
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    error_text = await response.text()
                    tprint(f"Open orders fetch failed: {response.status} - {error_text}", "ERROR")
                    return []
                    
        except Exception as e:
            tprint(f"Error getting open orders: {e}", "ERROR")
        return []
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to BingX format."""
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "8h": "8h",
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M"
        }
        return interval_map.get(interval, "1m")

    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("BingX exchange connection closed")
    
    async def close(self) -> None:
        """Close the exchange connection."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            tprint("BingX exchange connection closed", "INFO")
        except Exception as e:
            tprint(f"Error closing BingX exchange: {e}", "ERROR")


# Factory function for creating BingX exchange instances
def create_bingx_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> BingXExchange:
    """Create a new BingX exchange instance."""
    return BingXExchange(api_key, api_secret, trade_symbol, password)
    subaccount_id: str | None = None,
    use_testnet: bool = False
) -> BingXExchange:
    """Create a new BingX exchange instance."""
    return BingXExchange(api_key, api_secret, trade_symbol, password, subaccount_id, use_testnet)
