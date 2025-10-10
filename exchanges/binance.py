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
from typing import Any
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
    ) -> None:
        super().__init__(api_key, api_secret, trade_symbol, password)
        self.logger = system_logger.getChild("BinanceExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://api.binance.com"
        self.futures_url = "https://fapi.binance.com"
        self.testnet_url = "https://testnet.binance.vision"
        self.testnet_futures_url = "https://testnet.binancefuture.com"
        self.use_testnet = False  # Set to True for testing

    @handle_async_errors(default_return=None)
    async def _initialize_exchange(self) -> None:
        """Initialize the Binance exchange client."""
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

            tprint("✅ Binance exchange initialized successfully", "INFO")

        except Exception as e:
            tprint(f"❌ Failed to initialize Binance exchange: {e}", "ERROR")
            raise

    @handle_async_errors(default_return=None)
    async def _test_connection(self) -> None:
        """Test connection to Binance API."""
        try:
            url = f"{self._get_base_url()}/api/v3/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("serverTime")
                    tprint(f"Connected to Binance API (Server time: {server_time})", "INFO")
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
        """Make HTTP request to Binance API."""
        try:
            if aiohttp is None or not self.session:
                tprint("⚠️ aiohttp not available, returning mock data", "WARNING")
                return []

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
                    tprint("Authentication failed - check API credentials", "ERROR")
                    return {"error": "authentication_failed"}
                elif response.status == 400:
                    # Bad request
                    error_data = await response.json()
                    tprint(f"Bad request: {error_data}", "ERROR")
                    return {"error": "bad_request", "details": error_data}
                else:
                    error_text = await response.text()
                    tprint(f"API request failed: {response.status} - {error_text}", "ERROR")
                    return {"error": f"http_{response.status}", "details": error_text}
        except Exception as e:
            tprint(f"Request failed: {e}", "ERROR")
            return None

    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Convert raw Binance kline data to standardized MarketData format."""
        market_data_list = []
        
        for item in raw_data:
            try:
                # Handle both list and dict formats from Binance
                if isinstance(item, list):
                    # Binance klines format: [open_time, open, high, low, close, volume, ...]
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
        """Get the market ID for a given symbol (Binance uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from Binance."""
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
        return await self._make_request("GET", "/api/v3/account", signed=True) or {}

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
            
        return await self._make_request("POST", "/api/v3/order", order_params, signed=True) or {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from Binance futures."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/fapi/v2/positionRisk", params, signed=True, futures=True)
        
        if data and isinstance(data, list):
            # Return first matching position or first position if no symbol specified
            for position in data:
                if not symbol or position.get("symbol", "").upper() == symbol.upper():
                    return position
            return data[0] if data else {}
        
        return data or {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from Binance."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v3/openOrders", params, signed=True)
        return data if isinstance(data, list) else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on Binance."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("DELETE", "/api/v3/order", params, signed=True) or {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from Binance."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("GET", "/api/v3/order", params, signed=True) or {}

    # Additional Binance-specific methods for data collection

    async def get_24hr_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get 24hr ticker statistics."""
        endpoint = "/api/v3/ticker/24hr"
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._make_request("GET", endpoint, params) or {}

    async def get_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Get order book data."""
        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 5000)  # Binance max limit is 5000
        }
        return await self._make_request("GET", "/api/v3/depth", params) or {}

    # Additional methods for live trading
    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get ticker information."""
        if symbol:
            return await self.get_24hr_ticker(symbol)
        else:
            # Get all tickers
            return await self._make_request("GET", "/api/v3/ticker/24hr") or {}
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades."""
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
        return await self._make_request("GET", "/fapi/v1/premiumIndex", params, futures=True) or {}
    
    async def get_open_interest(self, symbol: str | None = None) -> dict[str, Any]:
        """Get open interest information."""
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._make_request("GET", "/fapi/v1/openInterest", params, futures=True) or {}
    
    async def get_server_time(self) -> dict[str, Any]:
        """Get server time."""
        return await self._make_request("GET", "/api/v3/time") or {}
    
    async def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange information."""
        return await self._make_request("GET", "/api/v3/exchangeInfo") or {}
    
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol information."""
        exchange_info = await self.get_exchange_info()
        if exchange_info and "symbols" in exchange_info:
            for symbol_info in exchange_info["symbols"]:
                if symbol_info.get("symbol") == symbol.upper():
                    return symbol_info
        return {}
    
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

    @handle_async_errors(default_return=None)
    async def close(self) -> None:
        """Close the exchange connection."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            tprint("Binance exchange connection closed", "INFO")
        except Exception as e:
            tprint(f"Error closing Binance exchange: {e}", "ERROR")


# Factory function for creating Binance exchange instances
def create_binance_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> BinanceExchange:
    """
    Create a new Binance exchange instance.
    
    Args:
        api_key: Binance API key
        api_secret: Binance API secret
        trade_symbol: Trading symbol (default: BTCUSDT)
        password: API password (optional)
        
    Returns:
        BinanceExchange instance
    """
    try:
        return BinanceExchange(api_key, api_secret, trade_symbol, password)
    except Exception as e:
        tprint(f"❌ Failed to create Binance exchange: {e}", "ERROR")
        raise