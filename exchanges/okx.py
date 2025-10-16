"""
Enhanced OKX Exchange Implementation

This module provides a comprehensive OKX exchange implementation that uses
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
    RateLimitManager, AuditLogger, SystemStatusManager
)


class OkxExchange(BaseExchange):
    """
    Enhanced OKX exchange implementation with comprehensive functionality.
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
        self.logger = system_logger.getChild("OkxExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://www.okx.com" if not use_testnet else "https://www.okx.com"
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
        # Authentication
        self.auth_manager = AuthenticationManager("okx")
        self.api_key_manager = APIKeyManager("okx")
        self.time_sync_manager = TimeSyncManager("okx")
        self.subaccount_manager = SubaccountManager("okx")
        
        # Market metadata
        self.market_metadata = MarketMetadataManager("okx")
        self.instrument_manager = InstrumentManager("okx")
        self.precision_helper = PrecisionHelper()
        self.risk_tier_manager = RiskTierManager("okx")
        
        # Pricing
        self.price_manager = PriceManager("okx")
        self.ohlcv_manager = OHLCVManager("okx")
        
        # Orders
        self.order_manager = OrderManager("okx")
        self.idempotency_manager = IdempotencyManager("okx")
        
        # Risk
        self.risk_calculator = RiskCalculator("okx")
        
        # Wallet
        self.balance_manager = BalanceManager("okx")
        
        # Reliability
        self.rate_limit_manager = RateLimitManager("okx")
        self.audit_logger = AuditLogger("okx")
        
        # Register exchange-specific functions
        self._register_exchange_functions()
        
    def _register_exchange_functions(self) -> None:
        """Register exchange-specific functions with shared utilities."""
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
        
    def _setup_rate_limiting(self) -> None:
        """Set up rate limiting for different endpoints."""
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
        
    async def _initialize_exchange(self) -> None:
        """Initialize the enhanced OKX exchange client."""
        try:
            if aiohttp is None:
                self.logger.warning("⚠️ aiohttp not available, using mock session")
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

            self.logger.info("✅ OKX exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OKX exchange: {e}")
            raise
    
    async def _authenticate(self) -> None:
        """Authenticate with OKX using shared auth manager."""
        from .shared.auth.auth_manager import AuthConfig, APIKeyPermission
        
        # Create auth config
        auth_config = AuthConfig(
            exchange_name="okx",
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
            raise Exception("Failed to authenticate with OKX")
    
    async def _initialize_market_data(self) -> None:
        """Initialize market metadata and instruments."""
        # Refresh instruments
        await self.market_metadata.refresh_instruments()
        
        # Set up precision configurations
        await self._setup_precision_configs()
        
        # Set up risk tiers
        await self._setup_risk_tiers()
    
    async def _setup_precision_configs(self) -> None:
        """Set up precision configurations for symbols."""
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
    
    async def _setup_risk_tiers(self) -> None:
        """Set up risk tiers for symbols."""
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
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for data synchronization."""
        # Start time sync
        if self.auth_manager.is_authenticated_and_valid():
            await self.time_sync_manager.start_auto_sync(self._get_server_time)
        
        # Start market data refresh
        asyncio.create_task(self._background_market_data_refresh())
        
        # Start order sync
        asyncio.create_task(self._background_order_sync())
    
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
                self.logger.error(f"Error in background market data refresh: {e}")
                await asyncio.sleep(60)
    
    async def _background_order_sync(self) -> None:
        """Background task to sync orders."""
        while True:
            try:
                if self.session:
                    await self.order_manager.sync_orders_from_exchange()
                
                await asyncio.sleep(10)  # Sync every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in background order sync: {e}")
                await asyncio.sleep(30)
    
    # Exchange-specific API methods
    async def _get_server_time(self) -> Optional[int]:
        """Get server time in milliseconds."""
        try:
            if not self._check_session():
                raise Exception("HTTP session not initialized")
                
            url = f"{self.base_url}/api/v5/public/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("data", [{}])[0].get("ts")
                    if server_time:
                        return int(server_time)
                    else:
                        raise Exception("No server time in response")
                else:
                    raise Exception(f"HTTP {response.status}: Failed to get server time")
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            raise
    
    async def _test_connection(self) -> bool:
        """Test connection to OKX API."""
        try:
            server_time = await self._get_server_time()
            return server_time is not None
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise
    
    async def _get_account_info_raw(self) -> Dict[str, Any]:
        """Get raw account information from exchange."""
        try:
            if not self._check_session():
                raise Exception("HTTP session not initialized")
                
            headers = self.auth_manager.get_auth_headers("GET", "/api/v5/account/balance")
            if not headers:
                raise Exception("Failed to get authentication headers")
            
            url = f"{self.base_url}/api/v5/account/balance"
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == "0" and data.get("data"):
                        return data.get("data", [{}])[0]
                    else:
                        raise Exception(f"API error: {data.get('msg', 'Unknown error')}")
                else:
                    raise Exception(f"HTTP {response.status}: Failed to get account info")
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    async def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        try:
            if not self._check_session():
                return None
                
            headers = self.auth_manager.get_auth_headers("GET", "/api/v5/account/balance")
            if not headers:
                return None
            
            url = f"{self.base_url}/api/v5/account/balance"
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
        return None
    
    async def _get_instruments(self) -> List[Dict[str, Any]]:
        """Get instrument specifications."""
        try:
            if not self._check_session():
                return []
                
            url = f"{self.base_url}/api/v5/public/instruments"
            params = {"instType": "SPOT"}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting instruments: {e}")
        return []
    
    async def _get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get ticker data for symbol."""
        try:
            if not self._check_session():
                return None
                
            url = f"{self.base_url}/api/v5/market/ticker"
            params = {"instId": symbol.upper()}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
        return None
    
    async def _get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book for symbol."""
        try:
            if not self._check_session():
                return None
                
            url = f"{self.base_url}/api/v5/market/books"
            params = {"instId": symbol.upper(), "sz": str(limit)}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}: {e}")
        return None
    
    async def _get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for symbol."""
        try:
            if not self._check_session():
                return []
                
            url = f"{self.base_url}/api/v5/market/trades"
            params = {"instId": symbol.upper(), "limit": str(limit)}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting recent trades for {symbol}: {e}")
        return []
    
    async def _get_klines_raw(self, symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]:
        """Get raw kline data from exchange."""
        try:
            if not self._check_session():
                raise Exception("HTTP session not initialized")
                
            url = f"{self.base_url}/api/v5/market/candles"
            params = {
                "instId": symbol.upper(),
                "bar": self._convert_interval(interval),
                "limit": str(min(limit, 300))
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == "0":
                        raw_klines = data.get("data", [])
                        if not raw_klines:
                            raise Exception(f"No kline data returned for {symbol}")
                        return [{"timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]), 
                               "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])} 
                               for k in raw_klines]
                    else:
                        raise Exception(f"API error: {data.get('msg', 'Unknown error')}")
                else:
                    raise Exception(f"HTTP {response.status}: Failed to get klines for {symbol}")
        except Exception as e:
            self.logger.error(f"Error getting klines for {symbol}: {e}")
            raise
    
    # _convert_to_market_data is now handled by the base class with standardized conversion
    
    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol."""
        # For OKX, symbols are already in the correct format (e.g., BTCUSDT)
        # Just ensure they're uppercase
        return symbol.upper()
    
    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Get raw historical kline data from exchange."""
        try:
            if not self._check_session():
                return []
                
            url = f"{self.base_url}/api/v5/market/history-candles"
            params = {
                "instId": symbol.upper(),
                "bar": self._convert_interval(interval),
                "before": str(end_time_ms),
                "after": str(start_time_ms),
                "limit": str(min(limit, 300))
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Convert list format to dict format for consistency
                    raw_klines = data.get("data", [])
                    return [{"timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]), 
                           "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])} 
                           for k in raw_klines]
        except Exception as e:
            self.logger.error(f"Error getting historical klines for {symbol}: {e}")
        return []
    
    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Get raw historical aggregated trades from exchange."""
        try:
            if not self._check_session():
                return []
                
            url = f"{self.base_url}/api/v5/market/history-trades"
            params = {
                "instId": symbol.upper(),
                "before": str(end_time_ms),
                "after": str(start_time_ms),
                "limit": str(min(limit, 100))
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting historical trades for {symbol}: {e}")
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
            if not self._check_session():
                return []
                
            url = f"{self.base_url}/api/v5/market/history-candles"
            params = {
                "instId": symbol.upper(),
                "bar": self._convert_interval(interval),
                "before": str(int(end_time.timestamp() * 1000)),
                "after": str(int(start_time.timestamp() * 1000)),
                "limit": str(min(limit, 300))
            }
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting historical klines for {symbol}: {e}")
        return []
    
    async def _get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get funding rate for symbol."""
        try:
            if not self._check_session():
                return None
                
            url = f"{self.base_url}/api/v5/public/funding-rate"
            params = {"instId": symbol.upper()}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [{}])[0] if data.get("data") else None
        except Exception as e:
            self.logger.error(f"Error getting funding rate for {symbol}: {e}")
        return None
    
    async def _get_balances_exchange(self, account_type: str) -> List[Dict[str, Any]]:
        """Get balances for account type."""
        try:
            if not self._check_session():
                return []
                
            headers = self.auth_manager.get_auth_headers("GET", "/api/v5/account/balance")
            if not headers:
                return []
            
            url = f"{self.base_url}/api/v5/account/balance"
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting balances: {e}")
        return []
    
    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create raw order on exchange."""
        try:
            if not self._check_session():
                return {}
                
            # Generate idempotency key
            idempotency_key = self.idempotency_manager.create_order_key(
                symbol, side, order_type, quantity, price, None
            )
            
            # Prepare order parameters
            order_params = {
                "instId": symbol.upper(),
                "tdMode": "cash",  # cash, cross, isolated
                "side": "buy" if side.lower() == "buy" else "sell",
                "ordType": "market" if order_type.upper() == "MARKET" else "limit",
                "sz": str(quantity),
                "clOrdId": idempotency_key
            }
            
            if price is not None and order_type.upper() != "MARKET":
                order_params["px"] = str(price)
            
            # Add additional params if provided
            if params:
                order_params.update(params)
            
            # Add subaccount if specified
            if self.subaccount_id:
                order_params["subAcct"] = self.subaccount_id
            
            # Execute with rate limiting
            headers = self.auth_manager.get_auth_headers("POST", "/api/v5/trade/order", str(order_params))
            if not headers:
                return {}
            
            url = f"{self.base_url}/api/v5/trade/order"
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "POST", url, order_params, headers):
                async with self.session.post(url, json=order_params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [{}])[0]
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return {}
    
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
                "instId": symbol.upper(),
                "tdMode": "cash",  # cash, cross, isolated
                "side": "buy" if side.lower() == "buy" else "sell",
                "ordType": "market" if order_type.upper() == "MARKET" else "limit",
                "sz": str(quantity),
                "clOrdId": client_order_id or idempotency_key
            }
            
            if price is not None and order_type.upper() != "MARKET":
                order_params["px"] = str(price)
            
            if stop_price is not None:
                order_params["slTriggerPx"] = str(stop_price)
            
            # Add subaccount if specified
            if self.subaccount_id:
                order_params["subAcct"] = self.subaccount_id
            
            # Execute with rate limiting
            headers = self.auth_manager.get_auth_headers("POST", "/api/v5/trade/order", str(order_params))
            if not headers:
                return None
            
            url = f"{self.base_url}/api/v5/trade/order"
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "POST", url, order_params, headers):
                async with self.session.post(url, json=order_params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [{}])[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None
    
    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> Dict[str, Any]:
        """Cancel raw order on exchange."""
        try:
            if not self._check_session():
                return {"success": False, "error": "Session not initialized"}
                
            # Generate idempotency key
            idempotency_key = self.idempotency_manager.create_cancel_key(str(order_id), symbol)
            
            # Prepare cancel parameters
            cancel_params = {
                "instId": symbol.upper(),
                "ordId": str(order_id),
                "clOrdId": idempotency_key
            }
            
            # Add subaccount if specified
            if self.subaccount_id:
                cancel_params["subAcct"] = self.subaccount_id
            
            headers = self.auth_manager.get_auth_headers("POST", "/api/v5/trade/cancel-order", str(cancel_params))
            if not headers:
                return {}
            
            url = f"{self.base_url}/api/v5/trade/cancel-order"
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "POST", url, cancel_params, headers):
                async with self.session.post(url, json=cancel_params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"success": data.get("code") == "0", "data": data}
            
            return {"success": False, "error": "Request failed"}
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _cancel_order_exchange(self, order_id: str, symbol: str) -> bool:
        """Cancel order on exchange."""
        try:
            # Generate idempotency key
            idempotency_key = self.idempotency_manager.create_cancel_key(order_id, symbol)
            
            # Prepare cancel parameters
            cancel_params = {
                "instId": symbol.upper(),
                "ordId": order_id,
                "clOrdId": idempotency_key
            }
            
            # Add subaccount if specified
            if self.subaccount_id:
                cancel_params["subAcct"] = self.subaccount_id
            
            headers = self.auth_manager.get_auth_headers("POST", "/api/v5/trade/cancel-order", str(cancel_params))
            if not headers:
                return False
            
            url = f"{self.base_url}/api/v5/trade/cancel-order"
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "POST", url, cancel_params, headers):
                async with self.session.post(url, json=cancel_params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("code") == "0"
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> Dict[str, Any]:
        """Get raw order status from exchange."""
        try:
            if not self._check_session():
                return {}
                
            headers = self.auth_manager.get_auth_headers("GET", f"/api/v5/trade/order?instId={symbol}&ordId={order_id}")
            if not headers:
                return {}
            
            url = f"{self.base_url}/api/v5/trade/order"
            params = {"instId": symbol.upper(), "ordId": str(order_id)}
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "GET", url, params, headers):
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [{}])[0]
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {}
    
    async def _get_order_status_exchange(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order status from exchange."""
        try:
            if not self._check_session():
                return None
                
            headers = self.auth_manager.get_auth_headers("GET", f"/api/v5/trade/order?instId={symbol}&ordId={order_id}")
            if not headers:
                return None
            
            url = f"{self.base_url}/api/v5/trade/order"
            params = {"instId": symbol.upper(), "ordId": order_id}
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "GET", url, params, headers):
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [{}])[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    async def _get_open_orders_raw(self, symbol: Optional[str]) -> List[Dict[str, Any]]:
        """Get raw open orders from exchange."""
        try:
            if not self._check_session():
                return []
                
            headers = self.auth_manager.get_auth_headers("GET", "/api/v5/trade/orders-pending")
            if not headers:
                return []
            
            url = f"{self.base_url}/api/v5/trade/orders-pending"
            params = {}
            if symbol:
                params["instId"] = symbol.upper()
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "GET", url, params, headers):
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [])
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []
    
    async def _get_open_orders_exchange(self) -> List[Dict[str, Any]]:
        """Get open orders from exchange."""
        try:
            if not self._check_session():
                return []
                
            headers = self.auth_manager.get_auth_headers("GET", "/api/v5/trade/orders-pending")
            if not headers:
                return []
            
            url = f"{self.base_url}/api/v5/trade/orders-pending"
            
            async with self.rate_limit_manager.execute_with_rate_limit("trading", self._make_request, "GET", url, {}, headers):
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [])
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []
    
    async def _make_request(self, method: str, url: str, params: Dict[str, Any], headers: Dict[str, str]) -> Any:
        """Make HTTP request with proper error handling."""
        try:
            if not self._check_session():
                raise Exception("Session not initialized")
                
            if method.upper() == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            else:
                async with self.session.post(url, json=params, headers=headers) as response:
                    return await response.json()
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def _check_session(self) -> bool:
        """Check if session is initialized and available."""
        if not self.session:
            self.logger.error("HTTP session not initialized. Call _initialize_exchange() first.")
            return False
        return True
    
    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to OKX format."""
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6Hutc",
            "12h": "12Hutc",
            "1d": "1Dutc",
            "3d": "3Dutc",
            "1w": "1Wutc",
            "1M": "1Mutc"
        }
        return interval_map.get(interval, "1m")
    
    # Enhanced public methods using shared utilities
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol using shared price manager."""
        price_data = await self.price_manager.get_price(symbol)
        return price_data.price if price_data else None
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[MarketData]:
        """Get OHLCV data using shared OHLCV manager."""
        from .shared.pricing.ohlcv_manager import Timeframe
        
        # Convert timeframe string to enum
        timeframe_enum = Timeframe(timeframe)
        
        # Get OHLCV data
        ohlcv_data = await self.ohlcv_manager.get_ohlcv(symbol, timeframe_enum, limit)
        
        # Convert to MarketData format
        market_data_list = []
        for candle in ohlcv_data:
            market_data = MarketData(
                symbol=candle.symbol,
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                interval=timeframe
            )
            market_data_list.append(market_data)
        
        return market_data_list
    
    async def create_order_enhanced(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Create order with enhanced validation and idempotency."""
        # Validate order parameters
        is_valid, errors = self.precision_helper.validate_order(
            symbol, side, order_type, price, quantity
        )
        
        if not is_valid:
            self.logger.error(f"Order validation failed: {errors}")
            return None
        
        # Round prices and quantities
        if price is not None:
            price = self.precision_helper.round_price(price, symbol)
        quantity = self.precision_helper.round_quantity(quantity, symbol)
        
        # Create order using order manager
        from .shared.orders.order_manager import OrderSide, OrderType
        
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        order_type_enum = OrderType.MARKET if order_type.upper() == "MARKET" else OrderType.LIMIT
        
        order = self.order_manager.create_order(
            symbol=symbol,
            side=order_side,
            order_type=order_type_enum,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            client_order_id=client_order_id
        )
        
        # Submit order
        success = await self.order_manager.submit_order(order)
        if success:
            return {
                "order_id": order.order_id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value
            }
        
        return None
    
    async def get_balance(self, currency: str = "USDT") -> float:
        """Get balance for currency using shared balance manager."""
        from .shared.wallet.balance_manager import AccountType
        
        balance = self.balance_manager.get_balance(currency, AccountType.SPOT)
        return balance.total if balance else 0.0
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        try:
            if not self._check_session():
                return []
                
            headers = self.auth_manager.get_auth_headers("GET", "/api/v5/account/positions")
            if not headers:
                return []
            
            url = f"{self.base_url}/api/v5/account/positions"
            
            async with self.rate_limit_manager.execute_with_rate_limit("account", self._make_request, "GET", url, {}, headers):
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("code") == "0":
                            return data.get("data", [])
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    async def _get_position_risk_raw(self, symbol: str) -> Dict[str, Any]:
        """Get raw position risk information from exchange."""
        try:
            positions = await self.get_positions()
            
            for position in positions:
                if position.get("instId") == symbol.upper():
                    # Calculate liquidation risk using shared risk calculator
                    from .shared.risk.risk_calculator import PositionRisk
                    
                    position_risk = self.risk_calculator.calculate_position_risk(
                        symbol=symbol,
                        position_size=float(position.get("pos", 0)),
                        entry_price=float(position.get("avgPx", 0)),
                        current_price=float(position.get("markPx", 0)),
                        leverage=float(position.get("lever", 1))
                    )
                    
                    return {
                        "symbol": symbol,
                        "margin_ratio": position_risk.margin_ratio,
                        "liquidation_price": position_risk.liquidation_price,
                        "risk_level": position_risk.risk_level.value,
                        "unrealized_pnl": position_risk.unrealized_pnl,
                        "position_size": position.get("pos", 0),
                        "entry_price": position.get("avgPx", 0),
                        "current_price": position.get("markPx", 0),
                        "leverage": position.get("lever", 1)
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting position risk for {symbol}: {e}")
            return {}
    
    async def get_liquidation_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get liquidation risk for symbol."""
        positions = await self.get_positions()
        
        for position in positions:
            if position.get("instId") == symbol.upper():
                # Calculate liquidation risk using shared risk calculator
                from .shared.risk.risk_calculator import PositionRisk
                
                position_risk = self.risk_calculator.calculate_position_risk(
                    symbol=symbol,
                    position_size=float(position.get("pos", 0)),
                    entry_price=float(position.get("avgPx", 0)),
                    current_price=float(position.get("markPx", 0)),
                    leverage=float(position.get("lever", 1))
                )
                
                return {
                    "symbol": symbol,
                    "margin_ratio": position_risk.margin_ratio,
                    "liquidation_price": position_risk.liquidation_price,
                    "risk_level": position_risk.risk_level.value,
                    "unrealized_pnl": position_risk.unrealized_pnl
                }
        
        return None
    
    # Required public interface methods from BaseExchange/IExchangeClient
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[MarketData]:
        """Get kline data - required by IExchangeClient interface."""
        try:
            raw_data = await self._get_klines_raw(symbol, interval, limit)
            if not raw_data:
                raise Exception(f"Failed to get klines for {symbol}")
            return await self._convert_to_market_data(raw_data, symbol, interval)
        except Exception as e:
            self.logger.error(f"Error getting klines for {symbol}: {e}")
            raise

    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | datetime,
        end_time: int | datetime,
        limit: int = 1000,
        **kwargs
    ) -> list[MarketData]:
        """Get historical kline data - required by BaseExchange interface."""
        try:
            # Handle both datetime and milliseconds parameters
            if isinstance(start_time, datetime):
                start_time_ms = int(start_time.timestamp() * 1000)
            else:
                start_time_ms = start_time

            if isinstance(end_time, datetime):
                end_time_ms = int(end_time.timestamp() * 1000)
            else:
                end_time_ms = end_time

            raw_data = await self._get_historical_klines_raw(
                symbol, interval, start_time_ms, end_time_ms, limit
            )
            if not raw_data:
                raise Exception(f"Failed to get historical klines for {symbol}")
            return await self._convert_to_market_data(raw_data, symbol, interval)
        except Exception as e:
            self.logger.error(f"Error getting historical klines for {symbol}: {e}")
            raise

    async def get_historical_agg_trades(
        self,
        symbol: str,
        start_time: int | datetime,
        end_time: int | datetime,
        limit: int = 1000,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Get historical aggregated trades - required by BaseExchange interface."""
        try:
            # Handle both datetime and milliseconds parameters
            if isinstance(start_time, datetime):
                start_time_ms = int(start_time.timestamp() * 1000)
            else:
                start_time_ms = start_time

            if isinstance(end_time, datetime):
                end_time_ms = int(end_time.timestamp() * 1000)
            else:
                end_time_ms = end_time

            return await self._get_historical_agg_trades_raw(
                symbol, start_time_ms, end_time_ms, limit
            )
        except Exception as e:
            self.logger.error(f"Error getting historical trades for {symbol}: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Create order - required by IExchangeClient interface."""
        try:
            result = await self._create_order_raw(symbol, side, order_type, quantity, price, None)
            if not result:
                raise Exception(f"Failed to create order for {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Error creating order for {symbol}: {e}")
            raise

    async def cancel_order(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel order - required by BaseExchange interface."""
        try:
            result = await self._cancel_order_raw(symbol, order_id)
            if not result:
                raise Exception(f"Failed to cancel order {order_id} for {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id} for {symbol}: {e}")
            raise

    async def get_order_status(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get order status - required by BaseExchange interface."""
        try:
            result = await self._get_order_status_raw(symbol, order_id)
            if not result:
                raise Exception(f"Failed to get order status for {order_id} on {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id} on {symbol}: {e}")
            raise

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open orders - required by BaseExchange interface."""
        try:
            result = await self._get_open_orders_raw(symbol)
            return result
        except Exception as e:
            self.logger.error(f"Error getting open orders for {symbol or 'all'}: {e}")
            raise

    async def close(self) -> None:
        """Close the enhanced exchange connection."""
        # Stop background tasks
        await self.time_sync_manager.stop_auto_sync()
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Close shared utilities
        await self.auth_manager.close()
        
        self.logger.info("OKX exchange connection closed")


# Factory function for creating OKX exchange instances
def create_okx_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
    subaccount_id: str | None = None,
    use_testnet: bool = False,
) -> OkxExchange:
    """Create a new OKX exchange instance."""
    return OkxExchange(
        api_key=api_key,
        api_secret=api_secret,
        trade_symbol=trade_symbol,
        password=password,
        subaccount_id=subaccount_id,
        use_testnet=use_testnet
    )