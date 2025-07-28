import asyncio
import aiohttp
import hmac
import hashlib
import json
from typing import Any, Dict, List
from loguru import logger
import websockets
import time

# Assuming src.config is in the python path
# In a real project structure, you might need to adjust python path
# or use relative imports if the structure allows.
from src.config import settings

class BinanceExchange:
    """
    Asynchronous client for interacting with the Binance API.
    Handles both REST API calls and WebSocket streams with robust error handling.
    """
    BASE_URL = "https://fapi.binance.com" # Using fapi for futures
    WS_BASE_URL = "wss://fstream.binance.com"

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret.encode('utf-8') if api_secret else None
        self._session = aiohttp.ClientSession(base_url=self.BASE_URL)

    def _get_timestamp(self) -> int:
        """Returns the current timestamp in milliseconds."""
        return int(time.time() * 1000)

    async def _generate_signature(self, data: Dict[str, Any]) -> str:
        """Generates a HMAC SHA256 signature for signed requests."""
        if not self._api_secret:
            raise ValueError("API secret is not configured for signing requests.")
        query_string = '&'.join([f"{k}={v}" for k, v in data.items() if v is not None])
        return hmac.new(self._api_secret, query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    async def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = False, max_retries: int = 3):
        """
        Makes an asynchronous HTTP request to the Binance API with retry logic.
        """
        params = params or {}
        headers = {'X-MBX-APIKEY': self._api_key} if self._api_key else {}
        
        if signed:
            if not self._api_key or not self._api_secret:
                raise PermissionError("Signed endpoint requires API key and secret.")
            params['timestamp'] = self._get_timestamp()
            params['signature'] = await self._generate_signature(params)

        for attempt in range(max_retries):
            try:
                async with self._session.request(method.upper(), endpoint, params=params, headers=headers, timeout=10) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP Error {e.status} for {method} {endpoint}: {e.message}")
                if e.status >= 500 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    return {"error": e.message, "code": e.status}
            except asyncio.TimeoutError:
                logger.error(f"Request timed out for {method} {endpoint}")
                if attempt < max_retries -1:
                    await asyncio.sleep(1)
                else:
                    return {"error": "Request timed out"}
            except aiohttp.ClientError as e:
                logger.error(f"A non-HTTP client error occurred: {e}")
                return {"error": str(e)}
        return {"error": f"Failed request after {max_retries} retries."}

    # --- REST API Functions ---
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]] | None:
        """Fetches historical kline (candlestick) data."""
        endpoint = "/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return await self._request("GET", endpoint, params)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, stop_price: float = None, time_in_force: str = "GTC", reduce_only: bool = False):
        """Places a new order."""
        endpoint = "/fapi/v1/order"
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "newOrderRespType": "RESULT"
        }
        if price: params["price"] = price
        if stop_price: params["stopPrice"] = stop_price
        if order_type.upper() in ["LIMIT", "STOP", "TAKE_PROFIT"]:
            params["timeInForce"] = time_in_force
        if reduce_only: params["reduceOnly"] = "true"
            
        return await self._request("POST", endpoint, params, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None):
        """Cancel an open order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return await self._request("DELETE", endpoint, params, signed=True)

    async def get_account_info(self):
        """Fetches account information, including balances and positions."""
        endpoint = "/fapi/v2/account"
        return await self._request("GET", endpoint, signed=True)
        
    async def get_position_risk(self, symbol: str = None):
        """Get current position risk for all symbols or a specific symbol."""
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._request("GET", endpoint, signed=True, params=params)

    # --- WebSocket Handlers ---
    async def _websocket_handler(self, url: str, callback, name: str):
        """Generic WebSocket handler with reconnection logic."""
        logger.info(f"Connecting to {name} WebSocket: {url}")
        while True:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"{name} WebSocket connected.")
                    while True:
                        data = await ws.recv()
                        await callback(json.loads(data))
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError) as e:
                logger.warning(f"{name} WebSocket disconnected: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"An unexpected error occurred in {name} WebSocket: {e}. Reconnecting in 10 seconds...")
                await asyncio.sleep(10)

    async def kline_socket(self, symbol: str, interval: str, callback):
        """Connects to the kline WebSocket stream."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(url, callback, f"Kline ({symbol})")

    async def user_data_socket(self, callback):
        """
        Connects to the user data WebSocket stream for real-time account updates.
        Manages the listen key automatically.
        """
        while True:
            listen_key_resp = await self._request("POST", "/fapi/v1/listenKey", signed=True)
            if not listen_key_resp or 'listenKey' not in listen_key_resp:
                logger.error("Could not get listen key for user data stream. Retrying in 20 seconds...")
                await asyncio.sleep(20)
                continue
            
            listen_key = listen_key_resp['listenKey']
            url = f"{self.WS_BASE_URL}/ws/{listen_key}"
            
            async def keepalive():
                while True:
                    await asyncio.sleep(30 * 60) # Keepalive every 30 minutes
                    await self._request("PUT", "/fapi/v1/listenKey", signed=True)
                    logger.info("User data stream listen key keepalive sent.")

            keepalive_task = asyncio.create_task(keepalive())
            ws_handler_task = asyncio.create_task(self._websocket_handler(url, callback, "User Data"))

            await ws_handler_task
            keepalive_task.cancel()
            logger.info("Restarting user data socket connection...")

    async def close(self):
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("HTTP session closed.")

# Global exchange instance, initialized with settings from config
try:
    exchange = BinanceExchange(
        api_key=settings.binance_api_key, 
        api_secret=settings.binance_api_secret
    )
    logger.info("BinanceExchange client instantiated for Futures.")
except Exception as e:
    logger.critical(f"Failed to instantiate BinanceExchange: {e}")
    exchange = None
