"""
Binance Exchange Implementation

This module provides a complete Binance exchange implementation that follows
the BaseExchange interface and integrates with the data collection system.
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import ccxt.async_support as ccxt
except ImportError:
    ccxt = None


# Custom exception classes for better error handling
class BinanceError(Exception):
    """Base exception for Binance exchange errors."""
    pass


class AuthenticationError(BinanceError):
    """Authentication related errors."""
    pass


class NetworkError(BinanceError):
    """Network related errors."""
    pass


class ValidationError(BinanceError):
    """Validation related errors."""
    pass


class RateLimitError(BinanceError):
    """Rate limit related errors."""
    pass

from src.interfaces.base_interfaces import MarketData
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

from exchanges.base_exchange import BaseExchange
from exchanges.shared.interfaces_typed import tprint, handle_async_errors, handle_errors
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
    RateLimitManager, AuditLogger, SystemStatusManager
)
from exchanges.shared.interfaces_typed import (
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
    tprint, DataSource, ValidationResult
)


class BinanceExchange(BaseExchange):
    """
    Binance exchange implementation following the BaseExchange interface.
    
    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data)
    - Aggregated trades
    - Futures funding rates
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
        self.logger = system_logger.getChild("BinanceExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://api.binance.com"
        self.futures_url = "https://fapi.binance.com"
        self.testnet_url = "https://testnet.binance.vision"
        self.testnet_futures_url = "https://testnet.binancefuture.com"
        self.use_testnet = use_testnet
        self.subaccount_id = subaccount_id
        
        # Initialize shared utilities
        self._initialize_shared_utilities()
        
        # Exchange-specific settings
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30
        
        # Background task management
        self._running = True
        self._background_tasks: List[asyncio.Task] = []
        
    def _initialize_shared_utilities(self) -> None:
        """Initialize all shared utilities."""
        try:
            # Authentication
            self.auth_manager = AuthenticationManager("binance")
            self.api_key_manager = APIKeyManager("binance")
            self.time_sync_manager = TimeSyncManager("binance")
            self.subaccount_manager = SubaccountManager("binance")
            
            # Market metadata
            self.market_metadata = MarketMetadataManager("binance")
            self.instrument_manager = InstrumentManager("binance")
            self.precision_helper = PrecisionHelper()
            self.risk_tier_manager = RiskTierManager("binance")
            
            # Pricing
            self.price_manager = PriceManager("binance")
            self.ohlcv_manager = OHLCVManager("binance")
            
            # Orders
            self.order_manager = OrderManager("binance")
            self.idempotency_manager = IdempotencyManager("binance")
            
            # Risk
            self.risk_calculator = RiskCalculator("binance")
            
            # Wallet
            self.balance_manager = BalanceManager("binance")
            
            # Reliability
            self.rate_limit_manager = RateLimitManager("binance")
            self.audit_logger = AuditLogger("binance")
            
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
            from exchanges.shared.reliability.rate_limit_manager import RateLimit, RateLimitStrategy
            
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
        """Initialize the Binance exchange client."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for Binance exchange but not available")
        
        try:
            # Initialize aiohttp session with SSL configuration
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            # Create SSL connector with certificate verification disabled for compatibility
            # In production, proper SSL certificates should be configured
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            # Authenticate
            await self._authenticate()
            
            # Initialize market data
            await self._initialize_market_data()
            
            # Start background tasks
            await self._start_background_tasks()

            tprint("✅ Binance exchange initialized successfully", "INFO")

        except Exception as e:
            tprint(f"❌ Failed to initialize Binance exchange: {e}", "ERROR")
            raise
    
    async def _authenticate(self) -> None:
        """Authenticate with Binance using shared auth manager."""
        try:
            from exchanges.shared.auth.auth_manager import AuthConfig, APIKeyPermission
            
            # Create auth config
            auth_config = AuthConfig(
                exchange_name="binance",
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
                raise Exception("Failed to authenticate with Binance")
                
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
            from exchanges.shared.market.precision_helper import PrecisionConfig
            
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
            from exchanges.shared.market.risk_tier_manager import SymbolRiskProfile, RiskTier
            
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
            
            # Start market data refresh and track the task
            task1 = asyncio.create_task(self._background_market_data_refresh())
            self._background_tasks.append(task1)
            
            # Start order sync and track the task
            task2 = asyncio.create_task(self._background_order_sync())
            self._background_tasks.append(task2)
            
        except Exception as e:
            tprint(f"Failed to start background tasks: {e}", "ERROR")
    
    async def _background_market_data_refresh(self) -> None:
        """Background task to refresh market data."""
        while self._running:
            try:
                if self.session and self._running:
                    # Refresh market data for active symbols
                    symbols = [self.trade_symbol]
                    await self.market_metadata.refresh_market_data(symbols)
                
                await asyncio.sleep(30)  # Refresh every 30 seconds
            except asyncio.CancelledError:
                break  # Task was cancelled, exit gracefully
            except Exception as e:
                if self._running:  # Only log if not shutting down
                    tprint(f"Error in background market data refresh: {e}", "ERROR")
                await asyncio.sleep(60)
    
    async def _background_order_sync(self) -> None:
        """Background task to sync orders."""
        while self._running:
            try:
                if self.session and self._running:
                    await self.order_manager.sync_orders_from_exchange()
                
                await asyncio.sleep(10)  # Sync every 10 seconds
            except asyncio.CancelledError:
                break  # Task was cancelled, exit gracefully
            except Exception as e:
                if self._running:  # Only log if not shutting down
                    tprint(f"Error in background order sync: {e}", "ERROR")
                await asyncio.sleep(30)

    async def _test_connection(self) -> bool:
        """Test connection to Binance API."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
            
        url = f"{self._get_base_url()}/api/v3/time"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                server_time = data.get("serverTime")
                tprint(f"Connected to Binance API (Server time: {server_time})", "INFO")
                return True
            else:
                error_text = await response.text()
                tprint(f"Connection test failed with status: {response.status} - {error_text}", "ERROR")
                raise NetworkError(f"Connection test failed: {response.status} - {error_text}")
    
    async def _get_server_time(self) -> int:
        """Get server time in milliseconds."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
            
        url = f"{self._get_base_url()}/api/v3/time"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                server_time = data.get("serverTime")
                if server_time:
                    return int(server_time)
                else:
                    raise ValueError("Invalid server time response")
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get server time: {response.status} - {error_text}")
    
    async def _get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
            
        headers = self.auth_manager.get_auth_headers("GET", "/api/v3/account")
        if not headers:
            raise AuthenticationError("Failed to get authentication headers")
        
        url = f"{self._get_base_url()}/api/v3/account"
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get account info: {response.status} - {error_text}")
    
    async def _get_instruments(self) -> List[Dict[str, Any]]:
        """Get instrument specifications."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
            
        url = f"{self._get_base_url()}/api/v3/exchangeInfo"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("symbols", [])
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get instruments: {response.status} - {error_text}")
    
    async def _get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data for symbol."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
            
        url = f"{self._get_base_url()}/api/v3/ticker/24hr"
        params = {"symbol": symbol.upper()}
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get ticker for {symbol}: {response.status} - {error_text}")
    
    async def _get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for symbol."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
        if limit <= 0 or limit > 5000:
            raise ValueError("Limit must be between 1 and 5000")
            
        url = f"{self._get_base_url()}/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": str(limit)}
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get order book for {symbol}: {response.status} - {error_text}")
    
    async def _get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for symbol."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
            
        url = f"{self._get_base_url()}/api/v3/trades"
        params = {"symbol": symbol.upper(), "limit": str(limit)}
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get recent trades for {symbol}: {response.status} - {error_text}")
    
    async def _get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[List[Any]]:
        """Get kline data for symbol."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
        if not interval:
            raise ValueError("Interval is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
            
        url = f"{self._get_base_url()}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": str(min(limit, 1000))
        }
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get klines for {symbol}: {response.status} - {error_text}")
    
    async def _get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> List[List[Any]]:
        """Get historical kline data."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
        if not interval:
            raise ValueError("Interval is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
            
        url = f"{self._get_base_url()}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": str(int(start_time.timestamp() * 1000)),
            "endTime": str(int(end_time.timestamp() * 1000)),
            "limit": str(min(limit, 1000))
        }
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get historical klines for {symbol}: {response.status} - {error_text}")
    
    async def _get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get funding rate for symbol."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
            
        url = f"{self._get_futures_url()}/fapi/v1/premiumIndex"
        params = {"symbol": symbol.upper()}
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get funding rate for {symbol}: {response.status} - {error_text}")
    
    async def _get_balances_exchange(self, account_type: str) -> List[Dict[str, Any]]:
        """Get balances for account type."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
            
        headers = self.auth_manager.get_auth_headers("GET", "/api/v3/account")
        if not headers:
            raise AuthenticationError("Failed to get authentication headers")
        
        url = f"{self._get_base_url()}/api/v3/account"
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("balances", [])
            else:
                error_text = await response.text()
                raise NetworkError(f"Failed to get balances: {response.status} - {error_text}")
    
    async def _create_order_exchange(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create order on exchange."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not symbol:
            raise ValueError("Symbol is required")
        if not side or side.upper() not in ["BUY", "SELL"]:
            raise ValueError("Side must be BUY or SELL")
        if not order_type:
            raise ValueError("Order type is required")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price is not None and price <= 0:
            raise ValueError("Price must be positive")
        if stop_price is not None and stop_price <= 0:
            raise ValueError("Stop price must be positive")
            
        # Generate idempotency key
        idempotency_key = self.idempotency_manager.create_order_key(
            symbol, side, order_type, quantity, price, client_order_id
        )
        
        # Prepare order parameters
        order_params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
            "timestamp": int(time.time() * 1000)
        }
        
        if price is not None:
            order_params["price"] = str(price)
        if stop_price is not None:
            order_params["stopPrice"] = str(stop_price)
        if client_order_id:
            order_params["newClientOrderId"] = client_order_id
        
        # Generate signature
        query_string = urlencode(order_params)
        signature = self._generate_signature(order_params)
        order_params["signature"] = signature
        
        # Prepare headers
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        url = f"{self._get_base_url()}/api/v3/order"
        async with self.session.post(url, data=order_params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                tprint(f"Order creation failed: {response.status} - {error_text}", "ERROR")
                raise NetworkError(f"Order creation failed: {response.status} - {error_text}")
    
    async def _cancel_order_exchange(self, order_id: str) -> bool:
        """Cancel order on exchange."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not order_id:
            raise ValueError("Order ID is required")
            
        headers = self.auth_manager.get_auth_headers("DELETE", "/api/v3/order")
        if not headers:
            raise AuthenticationError("Failed to get authentication headers")
        
        url = f"{self._get_base_url()}/api/v3/order"
        params = {"orderId": order_id}
        async with self.session.delete(url, params=params, headers=headers) as response:
            if response.status == 200:
                return True
            else:
                error_text = await response.text()
                tprint(f"Order cancellation failed: {response.status} - {error_text}", "ERROR")
                raise NetworkError(f"Order cancellation failed: {response.status} - {error_text}")
    
    async def _get_order_status_exchange(self, order_id: str) -> Dict[str, Any]:
        """Get order status from exchange."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
        if not order_id:
            raise ValueError("Order ID is required")
            
        headers = self.auth_manager.get_auth_headers("GET", "/api/v3/order")
        if not headers:
            raise AuthenticationError("Failed to get authentication headers")
        
        url = f"{self._get_base_url()}/api/v3/order"
        params = {"orderId": order_id}
        async with self.session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                tprint(f"Order status check failed: {response.status} - {error_text}", "ERROR")
                raise NetworkError(f"Order status check failed: {response.status} - {error_text}")
    
    async def _get_open_orders_exchange(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders from exchange."""
        if not self.session:
            raise RuntimeError("Exchange session not initialized")
            
        headers = self.auth_manager.get_auth_headers("GET", "/api/v3/openOrders")
        if not headers:
            raise AuthenticationError("Failed to get authentication headers")
        
        url = f"{self._get_base_url()}/api/v3/openOrders"
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        
        async with self.session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                tprint(f"Open orders fetch failed: {response.status} - {error_text}", "ERROR")
                raise NetworkError(f"Open orders fetch failed: {response.status} - {error_text}")

    def _get_base_url(self) -> str:
        """Get the base URL for API calls."""
        return self.testnet_url if self.use_testnet else self.base_url

    def _get_futures_url(self) -> str:
        """Get the futures URL for API calls."""
        return self.testnet_futures_url if self.use_testnet else self.futures_url

    def _generate_signature(self, params: dict[str, Any]) -> str:
        """Generate HMAC signature for authenticated requests."""
        try:
            if not self.api_secret:
                raise ValueError("API secret not configured")
            
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            return signature
        except Exception as e:
            tprint(f"Error generating signature: {e}", "ERROR")
            raise

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        futures: bool = False
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Make HTTP request to Binance API."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for Binance exchange but not available")
        
        if not self.session:
            raise RuntimeError("Exchange session not initialized")

        base_url = self._get_futures_url() if futures else self._get_base_url()
        url = f"{base_url}{endpoint}"
        
        if params is None:
            params = {}

        headers = {}
        if signed and self.api_key:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)
            headers["X-MBX-APIKEY"] = self.api_key

        async with self.session.request(method, url, params=params, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                # Rate limit exceeded
                tprint("Rate limit exceeded, waiting...", "WARNING")
                await asyncio.sleep(1)
                return await self._make_request(method, endpoint, params, signed, futures)  # Retry
            elif response.status == 401:
                # Authentication error
                error_text = await response.text()
                tprint(f"Authentication failed - check API credentials: {error_text}", "ERROR")
                raise AuthenticationError("Authentication failed - check API credentials")
            elif response.status == 400:
                # Bad request
                error_data = await response.json()
                tprint(f"Bad request: {error_data}", "ERROR")
                raise ValidationError(f"Bad request: {error_data}")
            else:
                error_text = await response.text()
                tprint(f"API request failed: {response.status} - {error_text}", "ERROR")
                raise NetworkError(f"API request failed: {response.status} - {error_text}")

    # _convert_to_market_data is now handled by the base class with standardized conversion

    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol (Binance uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from Binance."""
        if not symbol:
            raise ValueError("Symbol is required")
        if not interval:
            raise ValueError("Interval is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
            
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)  # Binance max limit is 1000
        }
        
        data = await self._make_request("GET", "/api/v3/klines", params)
        if data and not isinstance(data, dict) and "error" not in str(data):
            # Convert list format to dict format for consistency
            klines = []
            for item in data:
                if isinstance(item, list) and len(item) >= 11:
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
                else:
                    tprint(f"Invalid kline data format: {item}", "WARNING")
            return klines
        elif isinstance(data, dict) and "error" in data:
            tprint(f"API error in klines: {data['error']}", "ERROR")
            raise NetworkError(f"API error in klines: {data['error']}")
        return []

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical kline data from Binance."""
        if not symbol:
            raise ValueError("Symbol is required")
        if not interval:
            raise ValueError("Interval is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if start_time_ms >= end_time_ms:
            raise ValueError("Start time must be before end time")
            
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/api/v3/klines", params)
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
        """Get raw historical aggregated trades from Binance."""
        if not symbol:
            raise ValueError("Symbol is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
        if start_time_ms >= end_time_ms:
            raise ValueError("Start time must be before end time")
            
        params = {
            "symbol": symbol.upper(),
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/api/v3/aggTrades", params)
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
        """Get raw account information from Binance."""
        return await self._make_request("GET", "/api/v3/account", signed=True)

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on Binance."""
        if not symbol:
            raise ValueError("Symbol is required")
        if not side or side.upper() not in ["BUY", "SELL"]:
            raise ValueError("Side must be BUY or SELL")
        if not order_type:
            raise ValueError("Order type is required")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if price is not None and price <= 0:
            raise ValueError("Price must be positive")
            
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
            
        return await self._make_request("POST", "/api/v3/order", order_params, signed=True)

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from Binance futures."""
        if not symbol:
            raise ValueError("Symbol is required")
            
        params = {"symbol": symbol.upper()}
        data = await self._make_request("GET", "/fapi/v2/positionRisk", params, signed=True, futures=True)
        
        if data and isinstance(data, list):
            # Return first matching position or first position if no symbol specified
            for position in data:
                if position.get("symbol", "").upper() == symbol.upper():
                    return position
            if data:
                return data[0]
            else:
                raise ValueError(f"No position found for symbol {symbol}")
        
        return data

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from Binance."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v3/openOrders", params, signed=True)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("Invalid response format for open orders")

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on Binance."""
        if not symbol:
            raise ValueError("Symbol is required")
        if not order_id:
            raise ValueError("Order ID is required")
            
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("DELETE", "/api/v3/order", params, signed=True)

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from Binance."""
        if not symbol:
            raise ValueError("Symbol is required")
        if not order_id:
            raise ValueError("Order ID is required")
            
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("GET", "/api/v3/order", params, signed=True)

    # Additional Binance-specific methods for data collection

    async def get_24hr_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get 24hr ticker statistics."""
        endpoint = "/api/v3/ticker/24hr"
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._make_request("GET", endpoint, params)

    async def get_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Get order book data."""
        if not symbol:
            raise ValueError("Symbol is required")
        if limit <= 0 or limit > 5000:
            raise ValueError("Limit must be between 1 and 5000")
            
        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 5000)  # Binance max limit is 5000
        }
        return await self._make_request("GET", "/api/v3/depth", params)

    # Additional methods for live trading
    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get ticker information - implements IExchange interface."""
        return await self.get_24hr_ticker(symbol)
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades."""
        if not symbol:
            raise ValueError("Symbol is required")
        if limit <= 0 or limit > 1000:
            raise ValueError("Limit must be between 1 and 1000")
            
        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 1000)
        }
        data = await self._make_request("GET", "/api/v3/trades", params)
        if data:
            trades = []
            for item in data:
                trades.append({
                    "timestamp": item["time"],
                    "price": item["price"],
                    "quantity": item["qty"],
                    "side": "buy" if item["isBuyerMaker"] else "sell",
                    "trade_id": item["id"]
                })
            return trades
        return []
    
    async def get_funding_rate(self, symbol: str | None = None) -> dict[str, Any]:
        """Get funding rate information."""
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._make_request("GET", "/fapi/v1/premiumIndex", params, futures=True)
    
    async def get_open_interest(self, symbol: str | None = None) -> dict[str, Any]:
        """Get open interest information."""
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._make_request("GET", "/fapi/v1/openInterest", params, futures=True)
    
    async def get_server_time(self) -> dict[str, Any]:
        """Get server time."""
        return await self._make_request("GET", "/api/v3/time")
    
    async def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange information."""
        return await self._make_request("GET", "/api/v3/exchangeInfo")
    
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol information."""
        if not symbol:
            raise ValueError("Symbol is required")
            
        exchange_info = await self.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            for symbol_info in exchange_info["symbols"]:
                if symbol_info.get("symbol") == symbol.upper():
                    return symbol_info
        raise ValueError(f"Symbol {symbol} not found")
    
    async def get_klines_stream(self, symbol: str, interval: str, callback) -> None:
        """Stream klines data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                klines = await self.get_klines(symbol, interval, limit=1)
                if klines:
                    await callback(klines[0])
                await asyncio.sleep(60)  # Poll every minute
            except Exception as e:
                self.logger.error(f"Error in klines stream: {e}")
                await asyncio.sleep(60)
    
    async def get_ticker_stream(self, symbol: str, callback) -> None:
        """Stream ticker data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                ticker = await self.get_ticker(symbol)
                if ticker:
                    await callback(ticker)
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                self.logger.error(f"Error in ticker stream: {e}")
                await asyncio.sleep(1)
    
    async def get_trade_stream(self, symbol: str, callback) -> None:
        """Stream trade data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                trades = await self.get_recent_trades(symbol, limit=10)
                for trade in trades:
                    await callback(trade)
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                self.logger.error(f"Error in trade stream: {e}")
                await asyncio.sleep(1)
    
    async def get_orderbook_stream(self, symbol: str, callback) -> None:
        """Stream orderbook data (WebSocket implementation would go here)."""
        # This would implement WebSocket streaming
        # For now, we'll use polling
        while True:
            try:
                orderbook = await self.get_order_book(symbol, limit=20)
                if orderbook:
                    await callback(orderbook)
                await asyncio.sleep(1)  # Poll every second
            except Exception as e:
                self.logger.error(f"Error in orderbook stream: {e}")
                await asyncio.sleep(1)

    # Required IExchangeClient interface methods
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> list[MarketData]:
        """Get historical kline data - implements IExchangeClient interface."""
        raw_data = await self._get_klines_raw(symbol, interval, limit)
        return await self._convert_to_market_data(raw_data, symbol, interval)

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information - implements IExchangeClient interface."""
        return await self._get_account_info_raw()

    async def create_order(self, symbol: str, side: str, quantity: float, price: float | None = None, order_type: str = 'MARKET') -> dict[str, Any]:
        """Create a trading order - implements IExchangeClient interface."""
        return await self._create_order_raw(symbol, side, order_type, quantity, price, None)

    async def get_position_risk(self, symbol: str) -> dict[str, Any]:
        """Get position risk information - implements IExchangeClient interface."""
        return await self._get_position_risk_raw(symbol)

    # Required IExchange interface methods
    async def initialize(self) -> None:
        """Initialize the exchange connection - implements IExchange interface."""
        await self._initialize_exchange()

    async def get_status(self) -> str:
        """Get current exchange status - implements IExchange interface."""
        try:
            if not self.session:
                return "disconnected"
            
            # Test connection to determine status
            await self._test_connection()
            return "connected"
        except Exception:
            return "error"

    async def get_balance(self, currency: str) -> dict[str, Any]:
        """Get balance for a specific currency - implements IExchange interface."""
        try:
            balances = await self._get_balances_exchange("spot")
            for balance in balances:
                if balance.get("asset", "").upper() == currency.upper():
                    return balance
            return {"asset": currency, "free": "0.0", "locked": "0.0"}
        except Exception as e:
            raise NetworkError(f"Failed to get balance for {currency}: {e}")

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an existing order - implements IExchange interface."""
        return await self._cancel_order_raw(self.trade_symbol, order_id)

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get status of an order - implements IExchange interface."""
        return await self._get_order_status_raw(self.trade_symbol, order_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open orders - implements IExchange interface."""
        return await self._get_open_orders_raw(symbol)


    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the exchange connection."""
        # Signal background tasks to stop
        self._running = False
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete cancellation
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear task list
        self._background_tasks.clear()
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
        
        tprint("Binance exchange connection closed", "INFO")


# Factory function for creating Binance exchange instances
def create_binance_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
    subaccount_id: str | None = None,
    use_testnet: bool = False
) -> BinanceExchange:
    """
    Create a new Binance exchange instance.
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        trade_symbol: Trading symbol (default: BTCUSDT)
        password: API password (optional)
        subaccount_id: Subaccount ID (optional)
        use_testnet: Use testnet (default: False)
        
    Returns:
        BinanceExchange instance
    """
    return BinanceExchange(api_key, api_secret, trade_symbol, password, subaccount_id, use_testnet)
