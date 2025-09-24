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

from .base_exchange import BaseExchange


class OkxExchange(BaseExchange):
    """
    OKX exchange implementation following the BaseExchange interface.

    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data)
    - Aggregated trades
    - Funding rates
    - Account information
    - Order management
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
        self.api_url = "https://www.okx.com/api/v5"
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
            url = f"{self.api_url}/public/time"
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
        """Generate HMAC signature for OKX authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret not configured")

        message = f"{timestamp}{method.upper()}{request_path}{body}"

        # Create HMAC SHA256 signature
        hmac_key = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        )

        # Base64 encode the signature
        signature = base64.b64encode(hmac_key.digest()).decode("utf-8")
        return signature

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        is_public: bool = True
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to OKX API."""
        if aiohttp is None or not self.session:
            self.logger.warning("⚠️ aiohttp not available, returning mock data")
            return []

        url = f"{self.api_url}{endpoint}"

        if params is None:
            params = {}

        headers = {
            "OK-ACCESS-KEY": self.api_key if self.api_key else "",
            "Content-Type": "application/json"
        }

        if signed and self.api_key and self.api_secret:
            # Generate signature
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(params) if params else ""
            signature = self._generate_signature(timestamp, method, endpoint, body)

            headers.update({
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": self.password or ""
            })

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
        """Convert raw OKX kline data to standardized MarketData format."""
        market_data_list = []

        for item in raw_data:
            try:
                # OKX format: timestamp, open, high, low, close, volume, ...
                timestamp = datetime.fromtimestamp(int(item[0]) / 1000)
                open_price = float(item[1])
                high_price = float(item[2])
                low_price = float(item[3])
                close_price = float(item[4])
                volume = float(item[5])

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
        # Convert interval format (OKX uses different format)
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "1d": "1D",
            "1w": "1W"
        }

        okx_interval = interval_map.get(interval, "1m")

        params = {
            "instId": symbol.upper(),
            "bar": okx_interval,
            "limit": str(min(limit, 100))  # OKX max limit is 100
        }

        data = await self._make_request("GET", "/market/candles", params, signed=False, is_public=True)

        if data and data.get("code") == "0" and data.get("data"):
            return data["data"]
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
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "1d": "1D",
            "1w": "1W"
        }

        okx_interval = interval_map.get(interval, "1m")

        params = {
            "instId": symbol.upper(),
            "bar": okx_interval,
            "before": str(end_time_ms),
            "after": str(start_time_ms),
            "limit": str(min(limit, 100))
        }

        data = await self._make_request("GET", "/market/candles", params, signed=False, is_public=True)

        if data and data.get("code") == "0" and data.get("data"):
            return data["data"]
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
            "limit": str(min(limit, 100))
        }

        data = await self._make_request("GET", "/market/trades", params, signed=False, is_public=True)

        if data and data.get("code") == "0" and data.get("data"):
            # Standardize field names
            trades = []
            for item in data["data"]:
                trades.append({
                    "timestamp": item.get("ts", 0),
                    "price": item.get("px", 0),
                    "quantity": item.get("sz", 0),
                    "side": item.get("side", "buy"),
                    "trade_id": item.get("tradeId", 0)
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from OKX."""
        return await self._make_request("GET", "/account/balance", signed=True) or {}

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
            "tdMode": "cash",  # Cash mode for spot trading
            "side": side.lower(),
            "ordType": order_type.lower(),
            "sz": str(quantity)
        }

        if price is not None:
            order_params["px"] = str(price)

        if params:
            order_params.update(params)

        return await self._make_request("POST", "/trade/order", order_params, signed=True) or {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from OKX."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/account/positions", params, signed=True)

        if data and data.get("code") == "0" and data.get("data"):
            # Return first matching position or first position if no symbol specified
            for position in data["data"]:
                if not symbol or position.get("instId", "").upper() == symbol.upper():
                    return position
            return data["data"][0] if data["data"] else {}

        return data or {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from OKX."""
        params = {"instId": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/trade/orders-pending", params, signed=True)

        if data and data.get("code") == "0" and data.get("data"):
            return data["data"]
        return []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(order_id)
        }
        return await self._make_request("POST", "/trade/cancel-order", params, signed=True) or {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(order_id)
        }
        return await self._make_request("GET", "/trade/order", params, signed=True) or {}

    async def _open_position_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None
    ) -> dict[str, Any]:
        """Open raw position on OKX Futures."""
        order_params = {
            "instId": symbol.upper(),
            "tdMode": "cross",  # Cross margin mode for futures
            "side": side.lower(),
            "ordType": order_type.lower(),
            "sz": str(quantity)
        }

        if price is not None:
            order_params["px"] = str(price)

        return await self._make_request("POST", "/trade/order", order_params, signed=True) or {}

    async def _close_position_raw(
        self,
        symbol: str,
        side: str,
        quantity: float,
        trade_id: Any
    ) -> dict[str, Any]:
        """Close raw position on OKX."""
        order_params = {
            "instId": symbol.upper(),
            "tdMode": "cross",
            "side": side.lower(),
            "ordType": "market",
            "sz": str(quantity)
        }

        # Close position by creating opposite order
        result = await self._make_request("POST", "/trade/order", order_params, signed=True)

        if result and result.get("code") == "0":
            return {
                "success": True,
                "pnl": 0,  # Would need to calculate based on position
                "close_order_id": result.get("data", [{}])[0].get("ordId")
            }
        return {}

    async def _get_trade_info_raw(self, symbol: str, trade_id: Any) -> dict[str, Any]:
        """Get raw trade information from OKX."""
        params = {
            "instId": symbol.upper(),
            "ordId": str(trade_id)
        }

        return await self._make_request("GET", "/trade/order", params, signed=True) or {}

    # Additional OKX-specific methods for data collection

    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get ticker data from OKX."""
        endpoint = "/market/ticker"
        params = {"instId": symbol.upper()} if symbol else {}
        return await self._make_request("GET", endpoint, params, signed=False, is_public=True) or {}

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """Get order book data."""
        params = {
            "instId": symbol.upper(),
            "sz": str(limit)
        }
        return await self._make_request("GET", "/market/books", params, signed=False, is_public=True) or {}

    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Get funding rate for futures."""
        params = {"instId": symbol.upper()}
        return await self._make_request("GET", "/public/funding-rate", params, signed=False, is_public=True) or {}

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
    trade_symbol: str = "BTC-USDT",
    password: str | None = None,
) -> OkxExchange:
    """Create a new OKX exchange instance."""
    return OkxExchange(api_key, api_secret, trade_symbol, password)