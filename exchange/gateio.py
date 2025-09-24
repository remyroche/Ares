"""
GateIO Exchange Implementation

This module provides a complete GateIO exchange implementation that follows
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

from .base_exchange import BaseExchange


class GateioExchange(BaseExchange):
    """
    GateIO exchange implementation following the BaseExchange interface.

    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data)
    - Aggregated trades
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
        self.logger = system_logger.getChild("GateioExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://api.gateio.ws"
        self.api_url = "https://api.gateio.ws/api/v4"

    async def _initialize_exchange(self) -> None:
        """Initialize the GateIO exchange client."""
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

            self.logger.info("✅ GateIO exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize GateIO exchange: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to GateIO API."""
        try:
            url = f"{self.api_url}/spot/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("server_time")
                    self.logger.info(f"Connected to GateIO API (Server time: {server_time})")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise

    def _generate_signature(self, method: str, url: str, query_string: str = "", body: str = "") -> dict[str, str]:
        """Generate signature for GateIO authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret not configured")

        timestamp = str(int(time.time() * 1000))
        message = f"{method}\n{url}\n{query_string}\n{body}\n{timestamp}"

        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha512
        ).hexdigest()

        return {
            "KEY": self.api_key,
            "Timestamp": timestamp,
            "SIGN": signature
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to GateIO API."""
        if aiohttp is None or not self.session:
            self.logger.warning("⚠️ aiohttp not available, returning mock data")
            return []

        url = f"{self.api_url}{endpoint}"

        if params is None:
            params = {}

        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        if signed and self.api_key and self.api_secret:
            query_string = urlencode(params) if params else ""
            signature_data = self._generate_signature(method, endpoint, query_string)

            headers.update({
                "KEY": signature_data["KEY"],
                "Timestamp": signature_data["Timestamp"],
                "SIGN": signature_data["SIGN"]
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
        """Convert raw GateIO kline data to standardized MarketData format."""
        market_data_list = []

        for item in raw_data:
            try:
                # GateIO format: timestamp, volume, close, high, low, open, ...
                timestamp = datetime.fromtimestamp(int(item[0]) / 1000)
                volume = float(item[1])
                close_price = float(item[2])
                high_price = float(item[3])
                low_price = float(item[4])
                open_price = float(item[5])

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
        """Get the market ID for a given symbol (GateIO uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from GateIO."""
        # Convert interval format
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }

        gateio_interval = interval_map.get(interval, "1m")

        params = {
            "currency_pair": symbol.upper(),
            "interval": gateio_interval,
            "limit": min(limit, 1000)  # GateIO max limit is 1000
        }

        data = await self._make_request("GET", "/spot/candlesticks", params, signed=False)

        if data:
            return data
        return []

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical kline data from GateIO."""
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }

        gateio_interval = interval_map.get(interval, "1m")

        params = {
            "currency_pair": symbol.upper(),
            "interval": gateio_interval,
            "limit": min(limit, 1000),
            "_from": start_time_ms // 1000,  # Convert to seconds
            "to": end_time_ms // 1000
        }

        data = await self._make_request("GET", "/spot/candlesticks", params, signed=False)

        if data:
            return data
        return []

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from GateIO."""
        params = {
            "currency_pair": symbol.upper(),
            "limit": min(limit, 1000),
            "_from": start_time_ms // 1000,
            "to": end_time_ms // 1000
        }

        data = await self._make_request("GET", "/spot/trades", params, signed=False)

        if data:
            # Standardize field names
            trades = []
            for item in data:
                trades.append({
                    "timestamp": item.get("create_time_ms", 0),
                    "price": item.get("price", 0),
                    "quantity": item.get("amount", 0),
                    "side": item.get("side", "buy"),
                    "trade_id": item.get("id", 0)
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from GateIO."""
        return await self._make_request("GET", "/spot/accounts", signed=True) or {}

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on GateIO."""
        order_params = {
            "currency_pair": symbol.upper(),
            "side": side.lower(),
            "type": order_type.lower(),
            "amount": str(quantity)
        }

        if price is not None:
            order_params["price"] = str(price)

        if params:
            order_params.update(params)

        return await self._make_request("POST", "/spot/orders", order_params, signed=True) or {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from GateIO."""
        # GateIO spot doesn't have positions like futures, so return account balance
        return await self._make_request("GET", "/spot/accounts", signed=True) or {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from GateIO."""
        params = {"currency_pair": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/spot/orders", params, signed=True)

        if data:
            return data
        return []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on GateIO."""
        endpoint = f"/spot/orders/{order_id}"
        return await self._make_request("DELETE", endpoint, signed=True) or {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from GateIO."""
        endpoint = f"/spot/orders/{order_id}"
        return await self._make_request("GET", endpoint, signed=True) or {}

    # Additional GateIO-specific methods for data collection

    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get ticker data from GateIO."""
        endpoint = "/spot/tickers"
        params = {"currency_pair": symbol.upper()} if symbol else {}
        return await self._make_request("GET", endpoint, params, signed=False) or {}

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """Get order book data."""
        params = {
            "currency_pair": symbol.upper(),
            "limit": str(limit),
            "with_id": "true"
        }
        return await self._make_request("GET", "/spot/order_book", params, signed=False) or {}

    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("GateIO exchange connection closed")


# Factory function for creating GateIO exchange instances
def create_gateio_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTC_USDT",
    password: str | None = None,
) -> GateioExchange:
    """Create a new GateIO exchange instance."""
    return GateioExchange(api_key, api_secret, trade_symbol, password)