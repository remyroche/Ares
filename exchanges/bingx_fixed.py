"""
BingX Exchange Implementation - Production Ready

This module provides a complete BingX exchange implementation that follows
the BaseExchange interface with proper error handling and real API integration.
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None

from src.interfaces.base_interfaces import MarketData
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

from .base_exchange import BaseExchange


class BingXAPIError(Exception):
    """BingX API specific error."""
    pass


class BingXConnectionError(Exception):
    """BingX connection error."""
    pass


class BingXAuthenticationError(Exception):
    """BingX authentication error."""
    pass


class BingXExchange(BaseExchange):
    """
    BingX exchange implementation following the BaseExchange interface.
    
    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data)
    - Aggregated trades
    - Futures funding rates
    - Account information
    - Order management
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
        self.subaccount_id = subaccount_id
        self.use_testnet = use_testnet
        
        # API endpoints
        if use_testnet:
            self.base_url = "https://open-api-testnet.bingx.com"
        else:
            self.base_url = "https://open-api.bingx.com"
        
        # Rate limiting
        self.rate_limits = {
            "requests_per_second": 20,
            "requests_per_minute": 1200,
            "requests_per_hour": 72000
        }
        
        # Request tracking
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()

    @handles_errors
    async def _initialize_exchange(self) -> None:
        """Initialize the BingX exchange client."""
        if aiohttp is None:
            raise BingXConnectionError("aiohttp is required but not installed")
        
        # Initialize aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(verify_ssl=True)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        
        # Test connection
        await self._test_connection()
        
        self.logger.info("✅ BingX exchange initialized successfully")

    @handles_errors
    async def _test_connection(self) -> None:
        """Test connection to BingX API."""
        try:
            # Test with server time endpoint
            url = f"{self.base_url}/openApi/swap/v2/server/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == 0:
                        server_time = data.get("data", {}).get("serverTime")
                        self.logger.info(f"Connected to BingX API (Server time: {server_time})")
                    else:
                        raise BingXAPIError(f"API error: {data.get('msg', 'Unknown error')}")
                else:
                    raise BingXConnectionError(f"Connection test failed with status: {response.status}")
        except Exception as e:
            if isinstance(e, (BingXAPIError, BingXConnectionError)):
                raise
            raise BingXConnectionError(f"Connection test failed: {e}")

    def _generate_signature(self, params: dict[str, Any]) -> str:
        """Generate HMAC signature for authenticated requests."""
        if not self.api_secret:
            raise BingXAuthenticationError("API secret not configured")
        
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check per-minute limit
        if self.request_count >= self.rate_limits["requests_per_minute"]:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                raise BingXAPIError(f"Rate limit exceeded. Wait {sleep_time:.2f} seconds")
        
        # Check per-second limit
        if current_time - self.last_request_time < 1.0 / self.rate_limits["requests_per_second"]:
            sleep_time = 1.0 / self.rate_limits["requests_per_second"] - (current_time - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

    @handles_errors
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        futures: bool = False
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Make HTTP request to BingX API with proper error handling."""
        if not self.session:
            raise BingXConnectionError("Exchange not initialized")
        
        # Check rate limits
        self._check_rate_limits()
        
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if signed and self.api_key:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)
            headers["X-BX-APIKEY"] = self.api_key
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == 0:
                        return data.get("data", data)
                    else:
                        error_msg = data.get("msg", "Unknown API error")
                        if "signature" in error_msg.lower() or "auth" in error_msg.lower():
                            raise BingXAuthenticationError(f"Authentication failed: {error_msg}")
                        else:
                            raise BingXAPIError(f"API error: {error_msg}")
                elif response.status == 401:
                    raise BingXAuthenticationError("Invalid API credentials")
                elif response.status == 403:
                    raise BingXAPIError("API access forbidden")
                elif response.status == 429:
                    raise BingXAPIError("Rate limit exceeded")
                else:
                    error_text = await response.text()
                    raise BingXAPIError(f"HTTP {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise BingXConnectionError(f"Network error: {e}")
        except Exception as e:
            if isinstance(e, (BingXAPIError, BingXAuthenticationError, BingXConnectionError)):
                raise
            raise BingXAPIError(f"Request failed: {e}")

    @handles_errors
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
                # Handle both list and dict formats from BingX
                if isinstance(item, list):
                    # BingX klines format: [open_time, open, high, low, close, volume, close_time, ...]
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

    @handles_errors
    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol (BingX uses symbol as-is)."""
        return symbol.upper()

    @handles_errors
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
        
        data = await self._make_request("GET", "/openApi/swap/v2/quote/klines", params)
        
        if not data:
            raise BingXAPIError("No kline data received")
        
        # Convert list format to dict format for consistency
        klines = []
        for item in data:
            if isinstance(item, list) and len(item) >= 6:
                klines.append({
                    "timestamp": item[0],
                    "open_time": item[0],
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": item[6] if len(item) > 6 else item[0] + 3599999,
                    "quote_volume": item[7] if len(item) > 7 else None,
                    "trades": item[8] if len(item) > 8 else None,
                    "taker_buy_base": item[9] if len(item) > 9 else None,
                    "taker_buy_quote": item[10] if len(item) > 10 else None
                })
        
        return klines

    @handles_errors
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
        
        data = await self._make_request("GET", "/openApi/swap/v2/quote/klines", params)
        
        if not data:
            raise BingXAPIError("No historical kline data received")
        
        # Convert list format to dict format
        klines = []
        for item in data:
            if isinstance(item, list) and len(item) >= 6:
                klines.append({
                    "timestamp": item[0],
                    "open_time": item[0],
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": item[6] if len(item) > 6 else item[0] + 3599999,
                    "quote_volume": item[7] if len(item) > 7 else None,
                    "trades": item[8] if len(item) > 8 else None,
                    "taker_buy_base": item[9] if len(item) > 9 else None,
                    "taker_buy_quote": item[10] if len(item) > 10 else None
                })
        
        return klines

    @handles_errors
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
        
        if not data:
            raise BingXAPIError("No aggregated trades data received")
        
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

    @handles_errors
    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from BingX."""
        data = await self._make_request("GET", "/openApi/swap/v2/user/balance", signed=True)
        
        if not data:
            raise BingXAPIError("No account information received")
        
        return data

    @handles_errors
    async def _create_order_raw(
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
            "quantity": str(quantity)
        }
        
        if price is not None:
            order_params["price"] = str(price)
            
        if params:
            order_params.update(params)
        
        data = await self._make_request("POST", "/openApi/swap/v2/trade/order", order_params, signed=True)
        
        if not data:
            raise BingXAPIError("Order creation failed - no response")
        
        return data

    @handles_errors
    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from BingX futures."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/swap/v2/user/positions", params, signed=True)
        
        if not data:
            raise BingXAPIError("No position data received")
        
        if isinstance(data, list):
            # Return first matching position or first position if no symbol specified
            for position in data:
                if not symbol or position.get("symbol", "").upper() == symbol.upper():
                    return position
            return data[0] if data else {}
        
        return data

    @handles_errors
    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from BingX."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/swap/v2/trade/openOrders", params, signed=True)
        
        if not data:
            raise BingXAPIError("No open orders data received")
        
        return data if isinstance(data, list) else []

    @handles_errors
    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on BingX."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        
        data = await self._make_request("DELETE", "/openApi/swap/v2/trade/order", params, signed=True)
        
        if not data:
            raise BingXAPIError("Order cancellation failed - no response")
        
        return data

    @handles_errors
    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from BingX."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        
        data = await self._make_request("GET", "/openApi/swap/v2/trade/order", params, signed=True)
        
        if not data:
            raise BingXAPIError("Order status check failed - no response")
        
        return data

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

    # Public interface methods (inherited from BaseExchange but explicitly defined for clarity)
    
    @handles_errors
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[MarketData]:
        """Get historical kline data - public interface method."""
        raw_data = await self._get_klines_raw(symbol, interval, limit)
        return await self._convert_to_market_data(raw_data, symbol, interval)

    @handles_errors
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> list[MarketData]:
        """Get historical kline data with time range - public interface method."""
        # Convert datetime to milliseconds
        start_time_ms = int(start_time.timestamp() * 1000)
        end_time_ms = int(end_time.timestamp() * 1000)

        raw_data = await self._get_historical_klines_raw(
            symbol, interval, start_time_ms, end_time_ms, limit
        )
        return await self._convert_to_market_data(raw_data, symbol, interval)

    @handles_errors
    async def get_account_info(self) -> dict[str, Any]:
        """Get account information - public interface method."""
        return await self._get_account_info_raw()

    @handles_errors
    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Create a trading order - public interface method."""
        return await self._create_order_raw(symbol, side, order_type, quantity, price, None)

    @handles_errors
    async def get_position_risk(self, symbol: str) -> dict[str, Any]:
        """Get position risk information - public interface method."""
        return await self._get_position_risk_raw(symbol)

    @handles_errors
    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("BingX exchange connection closed")


# Factory function for creating BingX exchange instances
def create_bingx_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
    subaccount_id: str | None = None,
    use_testnet: bool = False
) -> BingXExchange:
    """
    Create a new BingX exchange instance.
    
    Args:
        api_key: BingX API key
        api_secret: BingX API secret
        trade_symbol: Trading symbol (default: BTCUSDT)
        password: API password (optional)
        subaccount_id: Subaccount ID (optional)
        use_testnet: Use testnet environment (default: False)
        
    Returns:
        BingXExchange instance
    """
    return BingXExchange(api_key, api_secret, trade_symbol, password, subaccount_id, use_testnet)