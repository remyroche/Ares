"""
MEXC Exchange Implementation

This module provides a complete MEXC exchange implementation that follows
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


class MexcExchange(BaseExchange):
    """
    MEXC exchange implementation following the BaseExchange interface.
    
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
        self.logger = system_logger.getChild("MexcExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://api.mexc.com"
        self.use_testnet = False  # Set to True for testing

    async def _initialize_exchange(self) -> None:
        """Initialize the MEXC exchange client."""
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

            self.logger.info("✅ MEXC exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize MEXC exchange: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to MEXC API."""
        try:
            url = f"{self.base_url}/api/v3/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("serverTime")
                    self.logger.info(f"Connected to MEXC API (Server time: {server_time})")
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
        """Make HTTP request to MEXC API."""
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
            headers["X-MEXC-APIKEY"] = self.api_key

        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
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
        """Convert raw MEXC kline data to standardized MarketData format."""
        market_data_list = []
        
        for item in raw_data:
            try:
                # MEXC klines format: [open_time, open, high, low, close, volume, close_time, ...]
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
        """Get the market ID for a given symbol (MEXC uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from MEXC."""
        params = {
            "symbol": symbol.upper(),
            "interval": self._convert_interval(interval),
            "limit": min(limit, 1000)  # MEXC max limit is 1000
        }
        
        data = await self._make_request("GET", "/api/v3/klines", params)
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
        """Get raw historical kline data from MEXC."""
        params = {
            "symbol": symbol.upper(),
            "interval": self._convert_interval(interval),
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
        """Get raw historical aggregated trades from MEXC."""
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
        """Get raw account information from MEXC."""
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
        """Create raw order on MEXC."""
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
        """Get raw position risk information from MEXC futures."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v3/positionRisk", params, signed=True)
        
        if data and isinstance(data, list):
            # Return first matching position or first position if no symbol specified
            for position in data:
                if not symbol or position.get("symbol", "").upper() == symbol.upper():
                    return position
            return data[0] if data else {}
        
        return data or {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from MEXC."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v3/openOrders", params, signed=True)
        return data if isinstance(data, list) else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on MEXC."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("DELETE", "/api/v3/order", params, signed=True) or {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from MEXC."""
        params = {
            "symbol": symbol.upper(),
            "orderId": str(order_id)
        }
        return await self._make_request("GET", "/api/v3/order", params, signed=True) or {}

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to MEXC format."""
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
        self.logger.info("MEXC exchange connection closed")


# Factory function for creating MEXC exchange instances
def create_mexc_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> MexcExchange:
    """Create a new MEXC exchange instance."""
    return MexcExchange(api_key, api_secret, trade_symbol, password)
