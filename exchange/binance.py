import asyncio
import json
import hmac
import hashlib
import time
from typing import Any, Dict, List, Callable
from functools import wraps
from datetime import datetime, timedelta

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
        self._session = None # Initialize session as None, create it in _get_session
        self.trade_symbol = trade_symbol.upper()
        
        # Real-time data storage
        self.order_book = {'bids': {}, 'asks': {}}
        self.recent_trades = []
        self.kline_data = {}

    async def _get_session(self):
        """Ensures an aiohttp session is active."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(base_url=self.BASE_URL)
        return self._session

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
        Includes enhanced error handling for network, HTTP, and parsing issues.
        """
        params = params or {}
        headers = {'X-MBX-APIKEY': self._api_key} if self._api_key else {}
        
        if signed:
            if not self._api_key or not self._api_secret:
                logger.error(f"Permission Error: API key or secret missing for signed endpoint {endpoint}.")
                raise PermissionError("Signed endpoint requires API key and secret.")
            params['timestamp'] = self._get_timestamp()
            params['signature'] = await self._generate_signature(params)

        session = await self._get_session()

        for attempt in range(max_retries):
            try:
                async with session.request(method.upper(), endpoint, params=params, headers=headers, timeout=10) as response:
                    # Check for HTTP status codes
                    response.raise_for_status() 
                    
                    # Attempt to parse JSON
                    try:
                        json_response = await response.json()
                        return json_response
                    except aiohttp.ContentTypeError:
                        logger.error(f"Content Type Error: Expected JSON but got {response.headers.get('Content-Type')} for {endpoint}. Response: {await response.text()}")
                        raise ValueError("Invalid content type in API response.")
                    except json.JSONDecodeError:
                        logger.error(f"JSON Decode Error: Could not parse response as JSON for {endpoint}. Response: {await response.text()}")
                        raise ValueError("Could not decode JSON response from API.")

            except aiohttp.ClientResponseError as e:
                # Handle specific HTTP errors (4xx, 5xx)
                logger.error(f"HTTP Error {e.status} for {method} {endpoint}: {e.message}. Response: {e.history or await response.text()}")
                if e.status in [400, 401, 403, 404, 429]: # Client errors, often not retryable or rate limit
                    if e.status == 429: # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 5)) # Get retry-after header
                        logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds.")
                        await asyncio.sleep(retry_after)
                        # Don't increment attempt for rate limit, allow more retries if needed
                        continue 
                    raise # Re-raise for non-retryable client errors
                elif e.status >= 500: # Server errors, often retryable
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Server error. Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.critical(f"Failed after {max_retries} attempts due to server error for {endpoint}.")
                        raise
                else: # Other HTTP errors
                    raise

            except asyncio.TimeoutError:
                logger.error(f"Request timed out for {method} {endpoint} (attempt {attempt + 1}/{max_retries}).")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2) # Short delay before retrying timeout
                else:
                    logger.critical(f"Failed after {max_retries} attempts due to timeout for {endpoint}.")
                    raise

            except aiohttp.ClientError as e:
                # Catch broader aiohttp client errors (e.g., connection issues)
                logger.error(f"Aiohttp Client Error for {method} {endpoint}: {e} (attempt {attempt + 1}/{max_retries}).")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.critical(f"Failed after {max_retries} attempts due to client error for {endpoint}.")
                    raise
            except Exception as e:
                # Catch any other unexpected errors during the request
                logger.error(f"An unexpected error occurred during API request to {endpoint}: {e}", exc_info=True)
                raise # Re-raise to ensure it's handled upstream

        logger.critical(f"Request to {endpoint} failed after {max_retries} attempts.")
        return None # Should not be reached if exceptions are re-raised

    # --- REST API Functions ---
    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetches historical kline (candlestick) data."""
        endpoint = "/fapi/v1/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        try:
            return await self._request("GET", endpoint, params)
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}-{interval}: {e}")
            return []

    async def create_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, time_in_force: str = None, params: Dict[str, Any] = None):
        """Creates a new order."""
        endpoint = "/fapi/v1/order"
        order_params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if price:
            order_params["price"] = price
        if time_in_force:
            order_params["timeInForce"] = time_in_force
        elif order_type.upper() == 'LIMIT':
             order_params["timeInForce"] = "GTC"
        
        # Merge additional params from the caller
        if params:
            order_params.update(params)

        try:
            return await self._request("POST", endpoint, order_params, signed=True)
        except Exception as e:
            logger.error(f"Failed to create {order_type} order for {symbol} ({side} {quantity}): {e}")
            # Return a structured error response or re-raise based on upstream needs
            return {"error": str(e), "status": "failed"}

    async def get_order_status(self, symbol: str, order_id: int):
        """Retrieves the status of a specific order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper(), "orderId": order_id}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            logger.error(f"Failed to get status for order {order_id} on {symbol}: {e}")
            return {"error": str(e)}
        
    async def cancel_order(self, symbol: str, order_id: int):
        """Cancels an open order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper(), "orderId": order_id}
        try:
            return await self._request("DELETE", endpoint, params, signed=True)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on {symbol}: {e}")
            return {"error": str(e)}

    async def get_account_info(self):
        """Fetches account information, including balances and positions."""
        endpoint = "/fapi/v2/account"
        try:
            return await self._request("GET", endpoint, signed=True)
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
        
    async def get_position_risk(self, symbol: str = None):
        """Gets current position risk for all symbols or a specific symbol."""
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            logger.error(f"Failed to get position risk for {symbol or 'all symbols'}: {e}")
            return []

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves all open orders for a given symbol or all symbols.
        """
        endpoint = "/fapi/v1/openOrders"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            logger.error(f"Failed to get open orders for {symbol or 'all symbols'}: {e}")
            return []

    async def close_all_positions(self, symbol: str = None):
        """
        Closes all open positions for a given symbol or all symbols.
        This will place market orders to close positions.
        """
        logger.warning(f"Attempting to close all open positions for {symbol if symbol else 'all symbols'}...")
        try:
            positions = await self.get_position_risk(symbol)
            for position in positions:
                position_amount = float(position.get('positionAmt', 0))
                if position_amount != 0:
                    current_symbol = position['symbol']
                    side = "SELL" if position_amount > 0 else "BUY"
                    quantity = abs(position_amount)

                    logger.info(f"Closing {side} position for {current_symbol} with quantity {quantity}...")
                    order_response = await self.create_order(
                        symbol=current_symbol,
                        side=side,
                        order_type="MARKET",
                        quantity=quantity
                    )
                    if order_response and order_response.get('status') == 'failed':
                        logger.error(f"Failed to place closing order for {current_symbol}: {order_response.get('error')}")
                    else:
                        logger.info(f"Closed position for {current_symbol}: {order_response}")
                else:
                    logger.debug(f"No open position for {position.get('symbol', 'N/A')}.")
        except Exception as e:
            logger.error(f"Error closing all positions: {e}", exc_info=True)

    async def cancel_all_orders(self, symbol: str = None):
        """
        Cancels all open orders for a given symbol or all symbols.
        """
        logger.warning(f"Attempting to cancel all open orders for {symbol if symbol else 'all symbols'}...")
        try:
            open_orders = await self.get_open_orders(symbol)
            if not open_orders:
                logger.info(f"No open orders found for {symbol if symbol else 'all symbols'}.")
                return

            for order in open_orders:
                order_id = order['orderId']
                order_symbol = order['symbol']
                logger.info(f"Cancelling order {order_id} for {order_symbol}...")
                cancel_response = await self.cancel_order(order_symbol, order_id)
                if cancel_response and cancel_response.get('status') == 'failed':
                    logger.error(f"Failed to cancel order {order_id} for {order_symbol}: {cancel_response.get('error')}")
                else:
                    logger.info(f"Cancelled order {order_id} for {order_symbol}: {cancel_response}")
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}", exc_info=True)

    async def get_historical_agg_trades(self, symbol: str, start_time_ms: int, end_time_ms: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetches historical aggregated trades for a symbol within a time range.
        Binance API limits: max 1000 trades per request.
        """
        endpoint = "/fapi/v1/aggTrades"
        all_trades = []
        current_start_time = start_time_ms

        while current_start_time < end_time_ms:
            params = {
                "symbol": symbol.upper(),
                "startTime": current_start_time,
                "endTime": end_time_ms,
                "limit": limit
            }
            try:
                trades = await self._request("GET", endpoint, params)
                if not trades:
                    break
                all_trades.extend(trades)
                current_start_time = trades[-1]['T'] + 1 # 'T' is the trade time in milliseconds

                if len(trades) < limit: # If fewer than limit trades, likely reached end of data for the period
                    break
                
                await asyncio.sleep(0.1) # Small delay to avoid hitting rate limits
            except Exception as e:
                logger.error(f"Error fetching historical agg trades for {symbol}: {e}")
                break # Stop fetching for this period on error
        
        return all_trades

    async def get_historical_futures_data(self, symbol: str, start_time_ms: int, end_time_ms: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetches historical futures-specific data (funding rates, open interest).
        """
        funding_rates = []
        open_interest = []

        # Fetch Funding Rates
        fr_endpoint = "/fapi/v1/fundingRate"
        current_fr_start = start_time_ms
        while current_fr_start < end_time_ms:
            params = {
                "symbol": symbol.upper(),
                "startTime": current_fr_start,
                "endTime": end_time_ms,
                "limit": 1000
            }
            try:
                rates = await self._request("GET", fr_endpoint, params)
                if not rates:
                    break
                funding_rates.extend(rates)
                current_fr_start = rates[-1]['fundingTime'] + 1
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error fetching historical funding rates for {symbol}: {e}")
                break

        # Fetch Open Interest History
        oi_endpoint = "/fapi/v1/openInterestHist"
        oi_params = {
            "symbol": symbol.upper(),
            "period": "5m",
            "limit": 500
        }
        try:
            oi_data = await self._request("GET", oi_endpoint, oi_params)
            if oi_data:
                open_interest.extend(oi_data)
        except Exception as e:
            logger.error(f"Error fetching historical open interest for {symbol}: {e}")
        
        return {"funding_rates": funding_rates, "open_interest": open_interest}


    # --- WebSocket Handlers ---
    async def _websocket_handler(self, url: str, callback: Callable, name: str):
        """Generic WebSocket handler with reconnection logic."""
        logger.info(f"Connecting to {name} WebSocket: {url}")
        while True:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"{name} WebSocket connected.")
                    while True:
                        try:
                            data = await ws.recv()
                            await callback(json.loads(data))
                        except json.JSONDecodeError as e:
                            logger.error(f"WebSocket JSON Decode Error for {name}: {e}. Raw data: {data[:200]}...")
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message for {name}: {e}", exc_info=True)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"{name} WebSocket disconnected: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except asyncio.TimeoutError:
                logger.warning(f"{name} WebSocket connection timed out. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"An unexpected error occurred in {name} WebSocket: {e}. Reconnecting in 10 seconds...", exc_info=True)
                await asyncio.sleep(10)

    async def _process_depth_message(self, data: Dict[str, Any]):
        """Callback to process incoming depth stream messages."""
        try:
            for bid in data.get('b', []):
                price, qty = float(bid[0]), float(bid[1])
                if qty == 0:
                    if price in self.order_book['bids']:
                        del self.order_book['bids'][price]
                else:
                    self.order_book['bids'][price] = qty
            for ask in data.get('a', []):
                price, qty = float(ask[0]), float(ask[1])
                if qty == 0:
                    if price in self.order_book['asks']:
                        del self.order_book['asks'][price]
                else:
                    self.order_book['asks'][price] = qty
        except Exception as e:
            logger.error(f"Error processing depth message: {e}. Data: {data}", exc_info=True)

    async def _process_trade_message(self, data: Dict[str, Any]):
        """Callback to process incoming trade stream messages."""
        try:
            self.recent_trades.append(data)
            if len(self.recent_trades) > 200:
                self.recent_trades.pop(0)
        except Exception as e:
            logger.error(f"Error processing trade message: {e}. Data: {data}", exc_info=True)

    async def _process_kline_message(self, data: Dict[str, Any]):
        """Callback to process incoming kline stream messages."""
        try:
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
        except Exception as e:
            logger.error(f"Error processing kline message: {e}. Data: {data}", exc_info=True)

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
                listen_key = listen_key_resp.get('listenKey')
                if not listen_key:
                    logger.error("Failed to get listenKey for user data stream. Retrying in 30 seconds.")
                    await asyncio.sleep(30)
                    continue

                url = f"{self.WS_BASE_URL}/ws/{listen_key}"

                async def keepalive():
                    while True:
                        await asyncio.sleep(30 * 60) # Keepalive every 30 mins
                        try:
                            await self._request("PUT", "/fapi/v1/listenKey", signed=True)
                            logger.info("User data stream listen key keepalive sent.")
                        except Exception as e:
                            logger.error(f"Failed to send user data stream keepalive: {e}")

                keepalive_task = asyncio.create_task(keepalive())
                ws_handler_task = asyncio.create_task(self._websocket_handler(url, callback, "User Data"))
                
                await ws_handler_task
                keepalive_task.cancel()
            except Exception as e:
                logger.error(f"Failed to start or maintain user data socket: {e}. Retrying in 10 seconds...", exc_info=True)
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

