"""
BingX Exchange Implementation

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
    subaccount_id: str | None = None,
    use_testnet: bool = False
) -> BingXExchange:
    """Create a new BingX exchange instance."""
    return BingXExchange(api_key, api_secret, trade_symbol, password, subaccount_id, use_testnet)