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
from src.core.decorators import handles_errors

from .base_exchange import BaseExchange


class GateioExchange(BaseExchange):
    """
    GateIO exchange implementation following the BaseExchange interface.
    
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
        self.logger = system_logger.getChild("GateioExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://api.gateio.ws"
        self.use_testnet = False  # Set to True for testing

    async def _initialize_exchange(self) -> None:
        """Initialize the GateIO exchange client."""
        try:
            if aiohttp is None:
                self.logger.warning("⚠️ aiohttp not available, using mock session")
                self.logger.warning("⚠️ Mock session will fast fail on API calls instead of returning mock data")
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
            url = f"{self.base_url}/api/v4/spot/currencies"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"Connected to GateIO API (Found {len(data)} currencies)")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise

    def _generate_signature(self, method: str, url_path: str, query_string: str = "", payload: str = "") -> tuple[str, str, str]:
        """Generate GateIO signature."""
        if not self.api_secret:
            raise ValueError("API secret not configured")
        
        timestamp = str(int(time.time()))
        
        # Create signature string
        sign_string = f"{method}\n{url_path}\n{query_string}\n{hashlib.sha512(payload.encode()).hexdigest()}\n{timestamp}"
        
        # Generate signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        
        return signature, timestamp, query_string

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        payload: str = ""
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to GateIO API."""
        if aiohttp is None or not self.session:
            self.logger.error("❌ aiohttp not available, fast failing API call")
            raise Exception("aiohttp not available - API call cannot be completed")

        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
            query_string = ""
        else:
            query_string = urlencode(params)

        headers = {
            "Content-Type": "application/json"
        }
        
        if signed and self.api_key:
            signature, timestamp, _ = self._generate_signature(method, endpoint, query_string, payload)
            
            headers.update({
                "KEY": self.api_key,
                "Timestamp": timestamp,
                "SIGN": signature
            })

        try:
            if method.upper() == "GET":
                full_url = f"{url}?{query_string}" if query_string else url
                async with self.session.get(full_url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        self.logger.error(f"API request failed: {response.status} - {error_text}")
                        return None
            else:
                async with self.session.request(method, url, json=params, headers=headers) as response:
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
                # GateIO klines format: [timestamp, volume, close, high, low, open]
                if isinstance(item, list):
                    timestamp = datetime.fromtimestamp(int(item[0]))
                    volume = float(item[1])
                    close_price = float(item[2])
                    high_price = float(item[3])
                    low_price = float(item[4])
                    open_price = float(item[5])
                else:
                    # Dict format
                    timestamp = self._convert_timestamp(item.get("t", item.get("timestamp", 0)))
                    volume = float(item.get("v", item.get("volume", 0)))
                    close_price = float(item.get("c", item.get("close", 0)))
                    high_price = float(item.get("h", item.get("high", 0)))
                    low_price = float(item.get("l", item.get("low", 0)))
                    open_price = float(item.get("o", item.get("open", 0)))

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
        params = {
            "currency_pair": symbol.upper(),
            "interval": self._convert_interval(interval),
            "limit": min(limit, 1000)  # GateIO max limit is 1000
        }
        
        data = await self._make_request("GET", "/api/v4/spot/candlesticks", params)
        if data:
            # Convert GateIO format to standard format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": int(item[0]),
                    "open_time": int(item[0]),
                    "open": item[5],
                    "high": item[3],
                    "low": item[4],
                    "close": item[2],
                    "volume": item[1],
                    "close_time": int(item[0]) + self._get_interval_ms(interval),
                    "quote_volume": 0,  # GateIO doesn't provide this
                    "trades": 0,  # GateIO doesn't provide this
                    "taker_buy_base": 0,  # GateIO doesn't provide this
                    "taker_buy_quote": 0  # GateIO doesn't provide this
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
        """Get raw historical kline data from GateIO."""
        params = {
            "currency_pair": symbol.upper(),
            "interval": self._convert_interval(interval),
            "from": str(start_time_ms),
            "to": str(end_time_ms),
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/api/v4/spot/candlesticks", params)
        if data:
            # Convert GateIO format to standard format
            klines = []
            for item in data:
                klines.append({
                    "timestamp": int(item[0]),
                    "open_time": int(item[0]),
                    "open": item[5],
                    "high": item[3],
                    "low": item[4],
                    "close": item[2],
                    "volume": item[1],
                    "close_time": int(item[0]) + self._get_interval_ms(interval),
                    "quote_volume": 0,
                    "trades": 0,
                    "taker_buy_base": 0,
                    "taker_buy_quote": 0
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
        """Get raw historical aggregated trades from GateIO."""
        params = {
            "currency_pair": symbol.upper(),
            "from": str(start_time_ms),
            "to": str(end_time_ms),
            "limit": min(limit, 1000)
        }
        
        data = await self._make_request("GET", "/api/v4/spot/trades", params)
        if data:
            # Standardize field names
            trades = []
            for item in data:
                trades.append({
                    "timestamp": int(item["create_time"]) * 1000,
                    "price": item["price"],
                    "quantity": item["amount"],
                    "is_buyer_maker": item["side"] == "sell",
                    "trade_id": item["id"]
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from GateIO."""
        data = await self._make_request("GET", "/api/v4/spot/accounts", signed=True)
        if data:
            total_balance = 0.0
            available_balance = 0.0
            
            for account in data:
                if account["currency"] == "USDT":
                    total_balance += float(account["available"]) + float(account["locked"])
                    available_balance += float(account["available"])
            
            return {
                "totalBalance": total_balance,
                "availableBalance": available_balance,
                "frozenBalance": total_balance - available_balance,
                "details": data
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
        """Create raw order on GateIO."""
        order_params = {
            "currency_pair": symbol.upper(),
            "side": side.lower(),
            "amount": str(quantity),
            "type": "market" if order_type.upper() == "MARKET" else "limit"
        }
        
        if price is not None and order_type.upper() != "MARKET":
            order_params["price"] = str(price)
            
        if params:
            order_params.update(params)
            
        data = await self._make_request("POST", "/api/v4/spot/orders", order_params, signed=True)
        return data if data else {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from GateIO (spot only)."""
        # GateIO spot trading doesn't have positions, return empty
        return {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from GateIO."""
        params = {"currency_pair": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/api/v4/spot/open_orders", params, signed=True)
        return data if isinstance(data, list) else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on GateIO."""
        data = await self._make_request("DELETE", f"/api/v4/spot/orders/{order_id}", signed=True)
        return data if data else {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from GateIO."""
        data = await self._make_request("GET", f"/api/v4/spot/orders/{order_id}", signed=True)
        return data if data else {}

    def _convert_interval(self, interval: str) -> str:
        """Convert standard interval to GateIO format."""
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
            "12h": "12h",
            "1d": "1d",
            "3d": "3d",
            "1w": "1w",
            "1M": "1M"
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
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> GateioExchange:
    """Create a new GateIO exchange instance."""
    return GateioExchange(api_key, api_secret, trade_symbol, password)
