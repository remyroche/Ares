"""
OKX Exchange Implementation

This module provides a complete OKX exchange implementation that follows
the BaseExchange interface and integrates with the data collection system.
"""

import asyncio
import hashlib
import hmac
import time
import base64
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


class OkxExchange(BaseExchange):
    """
    OKX exchange implementation following the BaseExchange interface.
    
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
        self.logger = system_logger.getChild("OkxExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://www.okx.com"
        self.use_testnet = False  # Set to True for testing

    async def _initialize_exchange(self) -> None:
        """Initialize the OKX exchange client."""
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

            self.logger.info("✅ OKX exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OKX exchange: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to OKX API."""
        try:
            url = f"{self.base_url}/api/v5/public/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("data", [{}])[0].get("ts")
                    self.logger.info(f"Connected to OKX API (Server time: {server_time})")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise

    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate OKX signature."""
        if not self.api_secret:
            raise ValueError("API secret not configured")
        
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return signature

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        body: str = ""
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to OKX API."""
        if aiohttp is None or not self.session:
            self.logger.warning("⚠️ aiohttp not available, returning mock data")
            return []

        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}

        headers = {
            "Content-Type": "application/json"
        }
        
        if signed and self.api_key:
            timestamp = str(int(time.time() * 1000))
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            headers.update({
                "OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.password or "",
            })

        try:
            if method.upper() == "GET":
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", []) if data.get("code") == "0" else []
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API request failed: {response.status} - {error_text}")
                        return None
            else:
                async with self.session.request(method, url, json=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", []) if data.get("code") == "0" else []
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
        """Convert raw OKX kline data to standardized MarketData format."""
        market_data_list = []
        
        for item in raw_data:
            try:
                # OKX klines format: [timestamp, open, high, low, close, volume, ...]
                if isinstance(item, list):
                    timestamp = datetime.fromtimestamp(int(item[0]) / 1000)
                    open_price = float(item[1])
                    high_price = float(item[2])
                    low_price = float(item[3])
                    close_price = float(item[4])
                    volume = float(item[5])
                else:
                    # Dict format
                    timestamp = self._convert_timestamp(item.get("ts", item.get("timestamp", 0)))
                    open_price = float(item.get("open", 0))
                    high_price = float(item.get("high", 0))
                    low_price = float(item.get("low", 0))
                    close_price = float(item.get("close", 0))
                    volume = float(item.get("vol", item.get("volume", 0)))

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
        """Get the market ID for a given symbol (OKX uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from OKX."""
        params = {
            "instId": symbol.upper(),
            "bar": self._convert_interval(interval),
            "limit": min(limit, 300)  # OKX max limit is 300
        }
        
        data = await self._make_request("GET", "/api/v5/market/candles", params)
        if data:
            # Convert OKX format to standard format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": int(item[0]),
                    "open_time": int(item[0]),
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": int(item[0]) + self._get_interval_ms(interval),
                    "quote_volume": item[6],
                    "trades": item[7],
                    "taker_buy_base": item[8],
                    "taker_buy_quote": item[9]
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
        """Get raw historical kline data from OKX."""
        params = {
            "instId": symbol.upper(),
            "bar": self._convert_interval(interval),
            "before": str(end_time_ms),
            "after": str(start_time_ms),
            "limit": min(limit, 300)
        }
        
        data = await self._make_request("GET", "/api/v5/market/candles", params)
        if data:
            # Convert OKX format to standard format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": int(item[0]),
                    "open_time": int(item[0]),
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[5],
                    "close_time": int(item[0]) + self._get_interval_ms(interval),
                    "quote_volume": item[6],
                    "trades": item[7],
                    "taker_buy_base": item[8],
                    "taker_buy_quote": item[9]
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
        """Get raw historical aggregated trades from OKX."""
        params = {
            "instId": symbol.upper(),
            "before": str(end_time_ms),
            "after": str(start_time_ms),
            "limit": min(limit, 100)
        }
        
        data = await self._make_request("GET", "/api/v5/market/history-trades", params)
        if data:
            # Standardize field names
            trades = []
            for item in data:
                trades.append({
                    "timestamp": int(item["ts"]),
                    "price": item["px"],
                    "quantity": item["sz"],
                    "is_buyer_maker": item["side"] == "sell",
                    "trade_id": item["tradeId"]
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from OKX."""
        data = await self._make_request("GET", "/api/v5/account/balance", signed=True)
        if data and len(data) > 0:
            account = data[0]
            return {
                "accountId": account.get("accountId"),
                "totalBalance": account.get("totalEq"),
                "availableBalance": account.get("availEq"),
                "frozenBalance": account.get("frozenBal"),
                "details": account.get("details", [])
            }
        return {}

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on OKX."""
        order_params = {
            "instId": symbol.upper(),
            "tdMode": "cash",  # cash, cross, isolated
            "side": "buy" if side.lower() == "buy" else "sell",
            "ordType": "market" if order_type.upper() == "MARKET" else "limit",
            "sz": str(quantity)
        }
        
        if price is not None and order_type.upper() != "MARKET":
            order_params["px"] = str(price)
            
        if params:
            order_params.update(params)
            
        data = await self._make_request("POST", "/api/v5/trade/order", order_params, signed=True)
        if data and len(data) > 0:
            return data[0]
        return {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from OKX."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/account/positions", params, signed=True)
        
        if data and len(data) > 0:
            # Return first matching position or first position if no symbol specified
            for position in data:
                if not symbol or position.get("instId", "").upper() == symbol.upper():
                    return {
                        "symbol": position.get("instId"),
                        "size": position.get("pos"),
                        "side": position.get("posSide"),
                        "markPrice": position.get("markPx"),
                        "unrealizedPnl": position.get("upl"),
                        "liquidationPrice": position.get("liqPx"),
                        "margin": position.get("margin"),
                        "notionalUsd": position.get("notionalUsd")
                    }
            return data[0] if data else {}
        
        return {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from OKX."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/trade/orders-pending", params, signed=True)
        return data if isinstance(data, list) else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(order_id)
        }
        data = await self._make_request("POST", "/api/v5/trade/cancel-order", params, signed=True)
        if data and len(data) > 0:
            return data[0]
        return {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(order_id)
        }
        data = await self._make_request("GET", "/api/v5/trade/order", params, signed=True)
        if data and len(data) > 0:
            return data[0]
        return {}

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

    def _get_interval_ms(self, interval: str) -> int:
        """Get interval duration in milliseconds."""
        interval_map = {
            "1m": 60000,
            "3m": 180000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "2h": 7200000,
            "4h": 14400000,
            "6h": 21600000,
            "12h": 43200000,
            "1d": 86400000,
            "3d": 259200000,
            "1w": 604800000,
            "1M": 2592000000
        }
        return interval_map.get(interval, 60000)

    # Additional methods for live trading
    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get ticker information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/market/ticker", params)
        if data and len(data) > 0:
            ticker = data[0]
            return {
                "symbol": ticker.get("instId"),
                "last": ticker.get("last"),
                "bid": ticker.get("bidPx"),
                "ask": ticker.get("askPx"),
                "high": ticker.get("high24h"),
                "low": ticker.get("low24h"),
                "volume": ticker.get("vol24h"),
                "quoteVolume": ticker.get("volCcy24h"),
                "change": ticker.get("chg"),
                "changePercent": ticker.get("chgRate"),
                "timestamp": ticker.get("ts")
            }
        return {}
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades."""
        params = {
            "instId": symbol.upper(),
            "limit": min(limit, 100)
        }
        data = await self._make_request("GET", "/api/v5/market/trades", params)
        if data:
            trades = []
            for item in data:
                trades.append({
                    "timestamp": int(item["ts"]),
                    "price": item["px"],
                    "quantity": item["sz"],
                    "side": item["side"],
                    "trade_id": item["tradeId"]
                })
            return trades
        return []
    
    async def get_funding_rate(self, symbol: str | None = None) -> dict[str, Any]:
        """Get funding rate information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/public/funding-rate", params)
        if data and len(data) > 0:
            return data[0]
        return {}
    
    async def get_open_interest(self, symbol: str | None = None) -> dict[str, Any]:
        """Get open interest information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/public/open-interest", params)
        if data and len(data) > 0:
            return data[0]
        return {}
    
    async def get_server_time(self) -> dict[str, Any]:
        """Get server time."""
        data = await self._make_request("GET", "/api/v5/public/time")
        if data and len(data) > 0:
            return data[0]
        return {}
    
    async def get_instruments(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get instruments information."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v5/public/instruments", params)
        return data if isinstance(data, list) else []
    
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get symbol information."""
        instruments = await self.get_instruments(symbol)
        if instruments and len(instruments) > 0:
            return instruments[0]
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

    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("OKX exchange connection closed")


# Factory function for creating OKX exchange instances
def create_okx_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> OkxExchange:
    """Create a new OKX exchange instance."""
    return OkxExchange(api_key, api_secret, trade_symbol, password)
