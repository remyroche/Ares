import asyncio
import json
import hmac
import hashlib
import time
from typing import Any, Dict, List, Callable
from functools import wraps

import aiohttp
import websockets
from src.utils.logger import logger
from src.config import settings

class BinanceExchange:
    """
    Asynchronous client for interacting with the Binance Futures API.
    Handles REST API calls and WebSocket streams with robust error handling and reconnection logic.
    """
    BASE_URL = "https://fapi.binance.com"
    WS_BASE_URL = "wss://fstream.binance.com"

    def __init__(self, api_key: str, api_secret: str, trade_symbol: str):
        self._api_key = api_key
        self._api_secret = api_secret.encode('utf-8') if api_secret else None
        self._session = aiohttp.ClientSession(base_url=self.BASE_URL)
        self.trade_symbol = trade_symbol.upper()
        
        # Real-time data storage
        self.order_book = {'bids': {}, 'asks': {}}
        self.recent_trades = []
        self.kline_data = {}

    def _get_timestamp(self) -> int:
        """Returns the current timestamp in milliseconds."""
        return int(time.time() * 1000)

    async def _generate_signature(self, data: Dict[str, Any]) -> str:
        """Generates a HMAC SHA256 signature for signed requests."""
        if not self._api_secret:
            raise ValueError("API secret is not configured for signing requests.")
        query_string = '&'.join([f"{k}={v}" for k, v in data.items() if v is not None])
        return hmac.new(self._api_secret, query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    async def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None, signed: bool = False, max_retries: int = 5):
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
                    raise
            except asyncio.TimeoutError:
                logger.error(f"Request timed out for {method} {endpoint}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
            except aiohttp.ClientError as e:
                logger.error(f"A non-HTTP client error occurred: {e}")
                raise

    # --- REST API Functions ---
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetches historical kline (candlestick) data."""
        endpoint = "/fapi/v1/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        return await self._request("GET", endpoint, params)

    async def create_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, time_in_force: str = None):
        """Creates a new order."""
        endpoint = "/fapi/v1/order"
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if price:
            params["price"] = price
        if time_in_force:
            params["timeInForce"] = time_in_force
        elif order_type.upper() == 'LIMIT':
             params["timeInForce"] = "GTC"

        return await self._request("POST", endpoint, params, signed=True)

    async def get_order_status(self, symbol: str, order_id: int):
        """Retrieves the status of a specific order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper(), "orderId": order_id}
        return await self._request("GET", endpoint, params, signed=True)
        
    async def cancel_order(self, symbol: str, order_id: int):
        """Cancels an open order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper(), "orderId": order_id}
        return await self._request("DELETE", endpoint, params, signed=True)

    async def get_account_info(self):
        """Fetches account information, including balances and positions."""
        endpoint = "/fapi/v2/account"
        return await self._request("GET", endpoint, signed=True)
        
    async def get_position_risk(self, symbol: str = None):
        """Gets current position risk for all symbols or a specific symbol."""
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._request("GET", endpoint, params, signed=True)

    # --- WebSocket Handlers ---
    async def _websocket_handler(self, url: str, callback: Callable, name: str):
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

    async def _process_depth_message(self, data: Dict[str, Any]):
        """Callback to process incoming depth stream messages."""
        for bid in data.get('b', []):
            price, qty = bid
            if float(qty) == 0:
                if price in self.order_book['bids']:
                    del self.order_book['bids'][price]
            else:
                self.order_book['bids'][price] = float(qty)
        for ask in data.get('a', []):
            price, qty = ask
            if float(qty) == 0:
                if price in self.order_book['asks']:
                    del self.order_book['asks'][price]
            else:
                self.order_book['asks'][price] = float(qty)

    async def _process_trade_message(self, data: Dict[str, Any]):
        """Callback to process incoming trade stream messages."""
        self.recent_trades.append(data)
        if len(self.recent_trades) > 200:  # Keep last 200 trades
            self.recent_trades.pop(0)

    async def _process_kline_message(self, data: Dict[str, Any]):
        """Callback to process incoming kline stream messages."""
        kline = data.get('k', {})
        self.kline_data = {
            'open_time': kline.get('t'),
            'open': kline.get('o'),
            'high': kline.get('h'),
            'low': kline.get('l'),
            'close': kline.get('c'),
            'volume': kline.get('v'),
            'close_time': kline.get('T'),
            'is_closed': kline.get('x')
        }

    async def start_kline_socket(self, symbol: str, interval: str):
        """Connects to the kline WebSocket stream."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(url, self._process_kline_message, f"Kline ({symbol})")

    async def start_depth_socket(self, symbol: str):
        """Connects to the depth (order book) WebSocket stream."""
        stream_name = f"{symbol.lower()}@depth"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(url, self._process_depth_message, f"Depth ({symbol})")

    async def start_trade_socket(self, symbol: str):
        """Connects to the trade WebSocket stream."""
        stream_name = f"{symbol.lower()}@trade"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(url, self._process_trade_message, f"Trade ({symbol})")

    async def start_user_data_socket(self, callback: Callable):
        """Connects to the user data stream for account and order updates."""
        while True:
            try:
                listen_key_resp = await self._request("POST", "/fapi/v1/listenKey", signed=True)
                listen_key = listen_key_resp['listenKey']
                url = f"{self.WS_BASE_URL}/ws/{listen_key}"

                async def keepalive():
                    while True:
                        await asyncio.sleep(30 * 60) # Keepalive every 30 mins
                        await self._request("PUT", "/fapi/v1/listenKey", signed=True)
                        logger.info("User data stream listen key keepalive sent.")

                keepalive_task = asyncio.create_task(keepalive())
                ws_handler_task = asyncio.create_task(self._websocket_handler(url, callback, "User Data"))
                
                await ws_handler_task # This will run indefinitely until a disconnect
                keepalive_task.cancel()
            except Exception as e:
                logger.error(f"Failed to start or maintain user data socket: {e}. Retrying...")
                await asyncio.sleep(10)


    async def close(self):
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("HTTP session closed.")

# --- Global exchange instance ---
try:
    exchange = BinanceExchange(
        api_key=settings.binance_api_key, 
        api_secret=settings.binance_api_secret,
        trade_symbol=settings.trade_symbol
    )
    logger.info("BinanceExchange client instantiated for Futures.")
except Exception as e:
    logger.critical(f"Failed to instantiate BinanceExchange: {e}")
    exchange = None

