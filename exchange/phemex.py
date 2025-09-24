"""
Phemex Exchange Implementation

This module provides a complete Phemex exchange implementation that follows
the BaseExchange interface and integrates with the data collection system.
Phemex offers futures and perpetual contracts trading with up to 100x leverage.
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


class PhemexExchange(BaseExchange):
    """
    Phemex exchange implementation following the BaseExchange interface.
    
    Provides comprehensive data download capabilities for:
    - Klines (OHLCV data) for both spot and perpetual contracts
    - Aggregated trades
    - Futures funding rates
    - Position and risk management
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
        self.logger = system_logger.getChild("PhemexExchange")
        self.session: aiohttp.ClientSession | None = None
        self.base_url = "https://api.phemex.com"
        self.testnet_url = "https://testnet.phemex.com"
        self.use_testnet = False  # Set to True for testing

    async def _initialize_exchange(self) -> None:
        """Initialize the Phemex exchange client."""
        try:
            if aiohttp is None:
                self.logger.warning("⚠️ aiohttp not available, using mock session")
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

            self.logger.info("✅ Phemex exchange initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Phemex exchange: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to Phemex API."""
        try:
            url = f"{self._get_base_url()}/public/time"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    server_time = data.get("data", {}).get("timestamp")
                    self.logger.info(f"Connected to Phemex API (Server time: {server_time})")
                else:
                    raise Exception(f"Connection test failed with status: {response.status}")
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            raise

    def _get_base_url(self) -> str:
        """Get the base URL for API calls."""
        return self.testnet_url if self.use_testnet else self.base_url

    def _generate_signature(self, params: dict[str, Any], method: str, path: str) -> str:
        """Generate HMAC signature for authenticated requests."""
        if not self.api_secret:
            raise ValueError("API secret not configured")
        
        # Phemex signature format: HMAC-SHA256(query + expiry + body)
        query_string = urlencode(params) if params else ""
        expiry = str(int(time.time() * 1000) + 5000)  # 5 seconds from now
        body = ""
        
        message = query_string + expiry + body
        return hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
        data: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Make HTTP request to Phemex API."""
        if aiohttp is None or not self.session:
            self.logger.warning("⚠️ aiohttp not available, returning mock data")
            return []

        url = f"{self._get_base_url()}{endpoint}"
        
        if params is None:
            params = {}

        headers = {
            "Content-Type": "application/json"
        }
        
        if signed and self.api_key:
            expiry = str(int(time.time() * 1000) + 5000)
            query_string = urlencode(params) if params else ""
            body = ""
            
            message = query_string + expiry + body
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            headers.update({
                "x-phemex-access-token": self.api_key,
                "x-phemex-request-expiry": expiry,
                "x-phemex-request-signature": signature
            })

        try:
            async with self.session.request(
                method, url, params=params, headers=headers, json=data
            ) as response:
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
        """Convert raw Phemex kline data to standardized MarketData format."""
        market_data_list = []
        
        for item in raw_data:
            try:
                # Handle both list and dict formats from Phemex
                if isinstance(item, list):
                    # Phemex klines format: [timestamp, interval, last_close, last_close_qty, turnover, volume, open, high, low, close]
                    timestamp = datetime.fromtimestamp(item[0] / 1000)
                    open_price = float(item[6])  # open
                    high_price = float(item[7])  # high
                    low_price = float(item[8])   # low
                    close_price = float(item[9]) # close
                    volume = float(item[5])      # volume
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
        """Get the market ID for a given symbol (Phemex uses symbol as-is)."""
        return symbol.upper()

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from Phemex."""
        # Convert interval format to Phemex format
        interval_map = {
            "1m": "60",
            "3m": "180", 
            "5m": "300",
            "15m": "900",
            "30m": "1800",
            "1h": "3600",
            "2h": "7200",
            "4h": "14400",
            "6h": "21600",
            "12h": "43200",
            "1d": "86400",
            "1w": "604800"
        }
        
        phemex_interval = interval_map.get(interval, "3600")  # Default to 1h
        
        params = {
            "symbol": symbol.upper(),
            "interval": phemex_interval,
            "limit": min(limit, 200)  # Phemex max limit is 200
        }
        
        data = await self._make_request("GET", "/public/md/kline", params)
        if data and data.get("data", {}).get("rows"):
            # Convert Phemex format to standardized format
            klines = []
            for item in data["data"]["rows"]:
                klines.append({
                    "timestamp": item[0],
                    "interval": item[1],
                    "last_close": item[2],
                    "last_close_qty": item[3],
                    "turnover": item[4],
                    "volume": item[5],
                    "open": item[6],
                    "high": item[7],
                    "low": item[8],
                    "close": item[9]
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
        """Get raw historical kline data from Phemex."""
        # Convert interval format to Phemex format
        interval_map = {
            "1m": "60",
            "3m": "180", 
            "5m": "300",
            "15m": "900",
            "30m": "1800",
            "1h": "3600",
            "2h": "7200",
            "4h": "14400",
            "6h": "21600",
            "12h": "43200",
            "1d": "86400",
            "1w": "604800"
        }
        
        phemex_interval = interval_map.get(interval, "3600")
        
        params = {
            "symbol": symbol.upper(),
            "interval": phemex_interval,
            "from": start_time_ms // 1000,  # Phemex uses seconds
            "to": end_time_ms // 1000,
            "limit": min(limit, 200)
        }
        
        data = await self._make_request("GET", "/public/md/kline", params)
        if data and data.get("data", {}).get("rows"):
            # Convert Phemex format to standardized format
            klines = []
            for item in data["data"]["rows"]:
                klines.append({
                    "timestamp": item[0],
                    "interval": item[1],
                    "last_close": item[2],
                    "last_close_qty": item[3],
                    "turnover": item[4],
                    "volume": item[5],
                    "open": item[6],
                    "high": item[7],
                    "low": item[8],
                    "close": item[9]
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
        """Get raw historical aggregated trades from Phemex."""
        params = {
            "symbol": symbol.upper(),
            "from": start_time_ms // 1000,  # Phemex uses seconds
            "to": end_time_ms // 1000,
            "limit": min(limit, 200)
        }
        
        data = await self._make_request("GET", "/public/md/trade", params)
        if data and data.get("data", {}).get("rows"):
            # Convert Phemex format to standardized format
            trades = []
            for item in data["data"]["rows"]:
                trades.append({
                    "timestamp": item[0] * 1000,  # Convert to milliseconds
                    "price": item[1],
                    "quantity": item[2],
                    "side": item[3]  # 0 for buy, 1 for sell
                })
            return trades
        return []

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from Phemex."""
        return await self._make_request("GET", "/accounts/accountPositions", signed=True) or {}

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on Phemex."""
        order_data = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "orderType": order_type.upper(),
            "orderQty": str(quantity),  # Phemex requires string for quantities
            "timeInForce": "GoodTillCancel"
        }
        
        if price is not None:
            order_data["priceEp"] = str(int(price * 10000))  # Phemex uses price in basis points
        
        if params:
            order_data.update(params)
            
        return await self._make_request("POST", "/orders", data=order_data, signed=True) or {}

    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from Phemex."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/accounts/positions", params, signed=True)
        
        if data and data.get("data", {}).get("positions"):
            positions = data["data"]["positions"]
            # Return first matching position or first position if no symbol specified
            for position in positions:
                if not symbol or position.get("symbol", "").upper() == symbol.upper():
                    return position
            return positions[0] if positions else {}
        
        return {}

    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from Phemex."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/orders/active", params, signed=True)
        return data.get("data", {}).get("rows", []) if data else []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on Phemex."""
        return await self._make_request(
            "DELETE", f"/orders/{order_id}", 
            data={"symbol": symbol.upper()}, 
            signed=True
        ) or {}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from Phemex."""
        params = {"symbol": symbol.upper()}
        return await self._make_request("GET", f"/orders/{order_id}", params, signed=True) or {}

    # Additional Phemex-specific methods for data collection

    async def get_24hr_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Get 24hr ticker statistics."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._make_request("GET", "/public/md/ticker/24hr", params)
        return data.get("data", {}) if data else {}

    async def get_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Get order book data."""
        params = {
            "symbol": symbol.upper(),
            "depth": min(limit, 100)  # Phemex max depth is 100
        }
        data = await self._make_request("GET", "/public/md/orderbook", params)
        return data.get("data", {}) if data else {}

    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Get funding rate for perpetual contracts."""
        params = {"symbol": symbol.upper()}
        data = await self._make_request("GET", "/public/md/funding", params)
        return data.get("data", {}) if data else {}

    async def get_funding_history(
        self, 
        symbol: str, 
        start_time_ms: int, 
        end_time_ms: int, 
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get historical funding rates."""
        params = {
            "symbol": symbol.upper(),
            "from": start_time_ms // 1000,
            "to": end_time_ms // 1000,
            "limit": min(limit, 200)
        }
        data = await self._make_request("GET", "/public/md/funding-history", params)
        return data.get("data", {}).get("rows", []) if data else []

    async def get_leverage_info(self, symbol: str) -> dict[str, Any]:
        """Get leverage information for a symbol."""
        params = {"symbol": symbol.upper()}
        data = await self._make_request("GET", "/public/md/leverage", params)
        return data.get("data", {}) if data else {}

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for a symbol."""
        try:
            data = {
                "symbol": symbol.upper(),
                "leverage": str(int(leverage))
            }
            result = await self._make_request("PUT", "/positions/leverage", data=data, signed=True)
            return result is not None and result.get("code") == 0
        except Exception as e:
            self.logger.error(f"Failed to set leverage: {e}")
            return False

    async def set_margin_mode(self, symbol: str, mode: str) -> bool:
        """Set margin mode for a symbol."""
        try:
            # Phemex margin modes: "Fixed" or "Cross"
            data = {
                "symbol": symbol.upper(),
                "marginType": mode
            }
            result = await self._make_request("PUT", "/positions/margin", data=data, signed=True)
            return result is not None and result.get("code") == 0
        except Exception as e:
            self.logger.error(f"Failed to set margin mode: {e}")
            return False

    async def get_risk_limits(self, symbol: str) -> dict[str, Any]:
        """Get risk limits for a symbol."""
        params = {"symbol": symbol.upper()}
        data = await self._make_request("GET", "/public/md/risk-limit", params)
        return data.get("data", {}) if data else {}

    async def get_contract_info(self, symbol: str) -> dict[str, Any]:
        """Get contract information for a symbol."""
        params = {"symbol": symbol.upper()}
        data = await self._make_request("GET", "/public/md/contract", params)
        return data.get("data", {}) if data else {}

    async def get_server_time(self) -> dict[str, Any]:
        """Get server time."""
        return await self._make_request("GET", "/public/time") or {}

    async def get_exchange_info(self) -> dict[str, Any]:
        """Get exchange information and supported symbols."""
        return await self._make_request("GET", "/public/md/symbols") or {}

    async def get_trading_fees(self) -> dict[str, Any]:
        """Get trading fees information."""
        return await self._make_request("GET", "/public/md/fees", signed=True) or {}

    async def get_account_balance(self) -> dict[str, Any]:
        """Get account balance."""
        return await self._make_request("GET", "/accounts/balances", signed=True) or {}

    async def get_trade_history(
        self, 
        symbol: str | None = None, 
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get trade history."""
        params = {"limit": min(limit, 200)}
        if symbol:
            params["symbol"] = symbol.upper()
        
        data = await self._make_request("GET", "/accounts/trades", params, signed=True)
        return data.get("data", {}).get("rows", []) if data else []

    async def get_order_history(
        self, 
        symbol: str | None = None, 
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get order history."""
        params = {"limit": min(limit, 200)}
        if symbol:
            params["symbol"] = symbol.upper()
        
        data = await self._make_request("GET", "/orders/history", params, signed=True)
        return data.get("data", {}).get("rows", []) if data else []

    async def close(self) -> None:
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("Phemex exchange connection closed")


# Factory function for creating Phemex exchange instances
def create_phemex_exchange(
    api_key: str = "",
    api_secret: str = "",
    trade_symbol: str = "BTCUSDT",
    password: str | None = None,
) -> PhemexExchange:
    """Create a new Phemex exchange instance."""
    return PhemexExchange(api_key, api_secret, trade_symbol, password)