"""
BingX Exchange Implementation

This module provides a complete BingX exchange implementation that follows
the BaseExchange interface and integrates with the data collection system.
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
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
from .shared.interfaces_typed import tprint, handle_async_errors, handle_errors


class BingXExchange(BaseExchange):
    """
    BingX exchange implementation following the BaseExchange interface.
    
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
    ) -> None:
        super().__init__(api_key, api_secret, trade_symbol, password)
        self.logger = system_logger.getChild("BingXExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://open-api.bingx.com"
        self.futures_url = "https://open-api.bingx.com"
        self.testnet_url = "https://open-api-testnet.bingx.com"
        self.testnet_futures_url = "https://open-api-testnet.bingx.com"
        self.use_testnet = False  # Set to True for testing

    @handle_async_errors(default_return=None)
    async def _initialize_exchange(self) -> None:
        """Initialize the BingX exchange client."""
        try:
            if aiohttp is None:
                tprint("⚠️ aiohttp not available, using mock session", "WARNING")
                self.session = None
                return

            # Initialize aiohttp session with SSL configuration
            timeout = aiohttp.ClientTimeout(total=30)
            # Create SSL connector with certificate verification disabled for compatibility
            # In production, proper SSL certificates should be configured
            connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)

            # Test connection
            await self._test_connection()

            tprint("✅ BingX exchange initialized successfully", "INFO")

        except Exception as e:
            tprint(f"❌ Failed to initialize BingX exchange: {e}", "ERROR")
            raise

    @handle_async_errors(default_return=None)
    async def _test_connection(self) -> None:
        """Test connection to BingX API."""
        try:
            url = f"{self._get_base_url()}/openApi/spot/v1/common/server-time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("data", {}).get("serverTime")
                    tprint(f"Connected to BingX API (Server time: {server_time})", "INFO")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            tprint(f"Connection test failed: {e}", "ERROR")
            raise

    def _get_base_url(self) -> str:
        """Get the base URL for API calls."""
        return self.testnet_url if self.use_testnet else self.base_url

    def _get_futures_url(self) -> str:
        """Get the futures URL for API calls."""
        return self.testnet_futures_url if self.use_testnet else self.futures_url

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

    @handle_async_errors(default_return=None)
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        futures: bool = False
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to BingX API."""
        if aiohttp is None or not self.session:
            tprint("⚠️ aiohttp not available, returning mock data", "WARNING")
            return []

        base_url = self._get_futures_url() if futures else self._get_base_url()
        url = f"{base_url}{endpoint}"
        
        if params is None:
            params = {}

        headers = {
            "X-BX-APIKEY": self.api_key if self.api_key else "",
            "Content-Type": "application/json"
        }
        
        if signed and self.api_key:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)

        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    tprint(f"API request failed: {response.status} - {error_text}", "ERROR")
                    return None
        except Exception as e:
            tprint(f"Request failed: {e}", "ERROR")
            return None

    @handle_async_errors(default_return=[])
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
                    # BingX klines format: [open_time, open, high, low, close, volume, ...]
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
                tprint(f"Failed to convert kline data: {e}", "WARNING")
                continue

        return market_data_list

    @handle_async_errors(default_return="")
    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol (BingX uses symbol as-is)."""
        return symbol.upper()

    @handle_async_errors(default_return=[])
    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from BingX."""
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)  # BingX max limit is 1000
        }
        
        data = await self._make_request("GET", "/openApi/spot/v1/market/klines", params)
        if data and "data" in data:
            # Convert list format to dict format for consistency
            klines = []
            for item in data["data"]:
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

    @handle_async_errors(default_return=[])
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
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/openApi/spot/v1/market/klines", params)
        if data and "data" in data:
            # Convert list format to dict format
            klines = []
            for item in data["data"]:
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

    @handle_async_errors(default_return=[])
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
        if data and "data" in data:
            # Standardize field names
            trades = []
            for item in data["data"]:
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

    @handle_async_errors(default_return={})
    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from BingX."""
        return await self._make_request("GET", "/openApi/spot/v1/account", signed=True) or {}

    @handle_async_errors(default_return={})
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
            "quantity": quantity
        }
        
        if price is not None:
            order_params["price"] = price
            
        if params:
            order_params.update(params)
            
        return await self._make_request("POST", "/openApi/spot/v1/trade/order", order_params, signed=True) or {}

    @handle_async_errors(default_return={})
    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from BingX futures."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/futures/v1/positionRisk", params, signed=True, futures=True)
        
        if data and "data" in data and isinstance(data["data"], list):
            # Return first matching position or first position if no symbol specified
            for position in data["data"]:
                if not symbol or position.get("symbol", "").upper() == symbol.upper():
                    return position
            return data["data"][0] if data["data"] else {}
        
        return data or {}

    @handle_async_errors(default_return=[])
    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from BingX."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/spot/v1/trade/openOrders", params, signed=True)
        return data.get("data", []) if isinstance(data, dict) else []

    @handle_async_errors(default_return={})
    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on BingX."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("DELETE", "/openApi/spot/v1/trade/order", params, signed=True) or {}

    @handle_async_errors(default_return={})
    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from BingX."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("GET", "/openApi/spot/v1/trade/query", params, signed=True) or {}

    # Additional BingX-specific methods for data collection

    @handle_async_errors(default_return={})
    async def get_24hr_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get 24hr ticker statistics."""
        endpoint = "/openApi/spot/v1/market/ticker/24hr"
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", endpoint, params)
        return data.get("data", {}) if isinstance(data, dict) else {}

    @handle_async_errors(default_return={})
    async def get_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Get order book data."""
        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 5000)  # BingX max limit is 5000
        }
        data = await self._make_request("GET", "/openApi/spot/v1/market/depth", params)
        return data.get("data", {}) if isinstance(data, dict) else {}

    # Additional methods for live trading
    @handle_async_errors(default_return={})
    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get ticker information."""
        if symbol:
            return await self.get_24hr_ticker(symbol)
        else:
            # Get all tickers
            data = await self._make_request("GET", "/openApi/spot/v1/market/ticker/24hr")
            return data.get("data", {}) if isinstance(data, dict) else {}
    
    @handle_async_errors(default_return=[])
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades."""
        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 1000)
        }
        data = await self._make_request("GET", "/openApi/spot/v1/market/trades", params)
        if data and "data" in data:
            trades = []
            for item in data["data"]:
                trades.append({
                    "timestamp": item["time"],
                    "price": item["price"],
                    "quantity": item["qty"],
                    "side": "buy" if item["isBuyerMaker"] else "sell",
                    "trade_id": item["id"]
                })
            return trades
        return []
    
    @handle_async_errors(default_return={})
    async def get_funding_rate(self, symbol: str | None = None) -> dict[str, Any]:
        """Get funding rate information."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/futures/v1/market/premiumIndex", params, futures=True)
        return data.get("data", {}) if isinstance(data, dict) else {}
    
    @handle_async_errors(default_return={})
    async def get_open_interest(self, symbol: str | None = None) -> dict[str, Any]:
        """Get open interest information."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/openApi/futures/v1/market/openInterest", params, futures=True)
        return data.get("data", {}) if isinstance(data, dict) else {}
    
    @handle_async_errors(default_return={})
    async def get_server_time(self) -> dict[str, Any]:
        """Get server time."""
        data = await self._make_request("GET", "/openApi/spot/v1/common/server-time")
        return data.get("data", {}) if isinstance(data, dict) else {}
    
    @handle_async_errors(default_return={})
    async def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange information."""
        data = await self._make_request("GET", "/openApi/spot/v1/common/symbols")
        return data.get("data", {}) if isinstance(data, dict) else {}
    
    @handle_async_errors(default_return={})
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol information."""
        exchange_info = await self.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            for symbol_info in exchange_info["symbols"]:
                if symbol_info.get("symbol") == symbol.upper():
                    return symbol_info
        return {}
    
    @handle_async_errors(default_return=None)
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
                tprint(f"Error in klines stream: {e}", "ERROR")
                await asyncio.sleep(60)
    
    @handle_async_errors(default_return=None)
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
                tprint(f"Error in ticker stream: {e}", "ERROR")
                await asyncio.sleep(1)
    
    @handle_async_errors(default_return=None)
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
                tprint(f"Error in trade stream: {e}", "ERROR")
                await asyncio.sleep(1)
    
    @handle_async_errors(default_return=None)
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
                tprint(f"Error in orderbook stream: {e}", "ERROR")
                await asyncio.sleep(1)

    @handle_async_errors(default_return=None)
    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        tprint("BingX exchange connection closed", "INFO")


# Factory function for creating BingX exchange instances
def create_bingx_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> BingXExchange:
    """
    Create a new BingX exchange instance.
    
    Args:
        api_key: BingX API key
        api_secret: BingX API secret
        trade_symbol: Trading symbol (default: BTCUSDT)
        password: API password (optional)
        
    Returns:
        BingXExchange instance
    """
    try:
        return BingXExchange(api_key, api_secret, trade_symbol, password)
    except Exception as e:
        tprint(f"❌ Failed to create BingX exchange: {e}", "ERROR")
        raise