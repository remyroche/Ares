import asyncio
import hashlib
import hmac
import json
import logging
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

import aiohttp
import ccxt.async_support as ccxt
import websockets
from ccxt.base.errors import (
    AuthenticationError,
    DDoSProtection,
    ExchangeError,
    ExchangeNotAvailable,
    RateLimitExceeded,
    RequestTimeout,
)

from src.interfaces.base_interfaces import MarketData
from src.utils.error_handler import (
    handle_errors,
    handle_network_operations,
)
from src.utils.logger import system_logger

from .base_exchange import BaseExchange

logger = logging.getLogger(__name__)


class BinanceRateLimiter:
    """
    Rate limiter for Binance API based on their documentation:
    - 6000 weight limit per minute per IP address
    - Aggregated trades have much stricter limits (5 requests per minute)
    - Handles 429 responses with retryAfter field
    - Prevents IP bans by respecting rate limits
    """

    def __init__(self):
        self.request_times = []
        self.weight_limit = 6000  # per minute
        self.current_weight = 0
        self.last_reset = time.time()

        # Special limits for specific endpoints
        self.endpoint_limits = {
            "/fapi/v1/aggTrades": {
                "requests_per_minute": 10,  # Increased from 5 to 10
                "last_request_time": 0,
                "request_count": 0,
            },
        }

    async def wait_if_needed(self, request_weight=1, endpoint=None):
        """Wait if we're approaching the rate limit"""
        current_time = time.time()

        # Handle special endpoint limits
        if endpoint and endpoint in self.endpoint_limits:
            limit_info = self.endpoint_limits[endpoint]

            # Reset counter if a minute has passed
            if current_time - limit_info["last_request_time"] >= 60:
                limit_info["request_count"] = 0
                limit_info["last_request_time"] = current_time

            # Check if we're at the limit
            if limit_info["request_count"] >= limit_info["requests_per_minute"]:
                wait_time = 60 - (current_time - limit_info["last_request_time"])
                if wait_time > 0:
                    logger.info(
                        f"Endpoint rate limit hit for {endpoint}, waiting {wait_time:.1f} seconds...",
                    )
                    await asyncio.sleep(wait_time)
                    limit_info["request_count"] = 0
                    limit_info["last_request_time"] = time.time()

            limit_info["request_count"] += 1
            limit_info["last_request_time"] = current_time
            return

        # General weight-based rate limiting
        # Reset weight counter every minute
        if current_time - self.last_reset >= 60:
            self.current_weight = 0
            self.last_reset = current_time

        # If we're approaching the limit, wait
        if (
            self.current_weight + request_weight > self.weight_limit * 0.9
        ):  # Increased to 90%
            wait_time = 60 - (current_time - self.last_reset)
            if wait_time > 0:
                logger.info(
                    f"Rate limit approaching, waiting {wait_time:.1f} seconds...",
                )
                await asyncio.sleep(wait_time)
                self.current_weight = 0
                self.last_reset = time.time()

        self.current_weight += request_weight

    async def handle_429_response(self, response_data):
        """Handle 429 response with retryAfter field"""
        if "retryAfter" in response_data:
            retry_after = response_data["retryAfter"]
            wait_time = (retry_after - time.time() * 1000) / 1000  # Convert to seconds
            if wait_time > 0:
                logger.warning(f"Rate limit hit, waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                return True
        return False


# Global rate limiter instance
rate_limiter = BinanceRateLimiter()


def retry_on_rate_limit(max_retries=5, initial_backoff=1.0):
    """
    A decorator to handle API rate limiting and other transient errors
    with exponential backoff.

    Args:
        max_retries (int): The maximum number of retries before giving up.
        initial_backoff (float): The initial wait time in seconds for the first retry.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            while retries < max_retries:
                try:
                    # Await the async function
                    return await func(*args, **kwargs)
                except AuthenticationError as e:
                    logger.error(
                        f"Authentication error in {func.__name__}. Not retrying. "
                        f"Please check your API keys. Error: {e}",
                    )
                    # Do not retry on authentication errors, as they are persistent.
                    raise
                except (RateLimitExceeded, DDoSProtection) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            f"API Rate Limit Exceeded. Max retries reached for {func.__name__}. Error: {e}",
                        )
                        raise e

                    logger.warning(
                        f"API Rate Limit Exceeded for {func.__name__}. "
                        f"Retrying in {backoff:.2f} seconds... (Attempt {retries}/{max_retries})",
                    )
                    await asyncio.sleep(backoff)
                    # Double the backoff time for the next potential retry.
                    backoff *= 2
                except (ExchangeNotAvailable, RequestTimeout) as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            f"Exchange not available or request timed out. Max retries reached for {func.__name__}. Error: {e}",
                        )
                        raise e

                    logger.warning(
                        f"Exchange not available or request timed out for {func.__name__}. "
                        f"Retrying in {backoff:.2f} seconds... (Attempt {retries}/{max_retries})",
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                except ExchangeError as e:
                    logger.warning(
                        f"Caught a transient exchange error in {func.__name__}: {e}. Retrying...",
                    )
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            f"Max retries reached for {func.__name__} after multiple exchange errors. Last error: {e}",
                        )
                        raise e
                    await asyncio.sleep(backoff)
                    backoff *= 2
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred in {func.__name__}: {e}",
                    )
                    raise e
            # This part should not be reached if max_retries is > 0
            # but as a fallback, raise an exception if the loop finishes.
            raise Exception(f"Exhausted retries for {func.__name__}")

        return wrapper

    return decorator


class BinanceExchange(BaseExchange):
    """
    Asynchronous client for interacting with the Binance Futures API.
    Handles REST API calls and WebSocket streams with robust error handling and reconnection logic.
    """

    BASE_URL = "https://fapi.binance.com"
    WS_BASE_URL = "wss://fstream.binance.com"

    def __init__(self, api_key: str, api_secret: str, trade_symbol: str):
        super().__init__(api_key, api_secret, trade_symbol)
        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8") if api_secret else None
        self._session = None  # Initialize session as None, create it in _get_session

        # Real-time data storage
        self.order_book = {"bids": {}, "asks": {}}
        self.recent_trades = []
        self.kline_data = {}
        self.mark_price_data = {}
        self.funding_rate_data = {}

    @handle_errors(exceptions=(Exception,), default_return=None, context="get_session")
    async def _get_session(self):
        """Ensures an aiohttp session is active."""
        if self._session is None or self._session.closed:
            # Create SSL context to handle certificate verification issues
            import ssl

            import certifi

            # Create SSL context with proper certificate verification
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            # Create connector with SSL context
            connector = aiohttp.TCPConnector(ssl=ssl_context)

            self._session = aiohttp.ClientSession(
                base_url=self.BASE_URL,
                connector=connector,
            )
        return self._session

    @handle_errors(
        exceptions=(Exception,),
        default_return=int(time.time() * 1000),
        context="get_timestamp",
    )
    def _get_timestamp(self) -> int:
        """Returns the current timestamp in milliseconds."""
        return int(time.time() * 1000)

    @handle_errors(
        exceptions=(ValueError, TypeError, AttributeError),
        default_return="",
        context="generate_signature",
    )
    async def _generate_signature(self, data: dict[str, Any]) -> str:
        """Generates a HMAC SHA256 signature for signed requests."""
        if not self._api_secret:
            raise ValueError("API secret is not configured for signing requests.")
        query_string = "&".join([f"{k}={v}" for k, v in data.items() if v is not None])
        return hmac.new(
            self._api_secret,
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @handle_network_operations(
        max_retries=5,
        default_return=None,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] = None,
        signed: bool = False,
        max_retries: int = 5,
        request_weight: int = 1,  # Default weight for most requests
    ):
        """
        Makes an asynchronous HTTP request to the Binance API with retry logic.
        Includes enhanced error handling for network, HTTP, and parsing issues.
        """
        params = params or {}
        headers = {"X-MBX-APIKEY": self._api_key} if self._api_key else {}

        if signed:
            if not self._api_key or not self._api_secret:
                system_logger.error(
                    f"Permission Error: API key or secret missing for signed endpoint {endpoint}.",
                )
                raise PermissionError("Signed endpoint requires API key and secret.")
            params["timestamp"] = self._get_timestamp()
            params["signature"] = await self._generate_signature(params)

        session = await self._get_session()

        for attempt in range(max_retries):
            try:
                # Rate limiting: Wait if we're approaching the limit
                await rate_limiter.wait_if_needed(request_weight, endpoint)

                async with session.request(
                    method.upper(),
                    endpoint,
                    params=params,
                    headers=headers,
                    timeout=10,
                ) as response:
                    # Check for HTTP status codes
                    response.raise_for_status()

                    # Attempt to parse JSON
                    try:
                        json_response = await response.json()
                        return json_response
                    except aiohttp.ContentTypeError:
                        system_logger.error(
                            f"Content Type Error: Expected JSON but got {response.headers.get('Content-Type')} for {endpoint}. Response: {await response.text()}",
                        )
                        raise ValueError("Invalid content type in API response.")
                    except json.JSONDecodeError:
                        system_logger.error(
                            f"JSON Decode Error: Could not parse response as JSON for {endpoint}. Response: {await response.text()}",
                        )
                        raise ValueError("Could not decode JSON response from API.")

            except aiohttp.ClientResponseError as e:
                # Handle specific HTTP errors (4xx, 5xx)
                response_text = ""
                try:
                    response_text = await response.text()
                except Exception:
                    response_text = "Could not read response body"

                system_logger.error(
                    f"HTTP Error {e.status} for {method} {endpoint}: {e.message}. Response: {response_text}",
                )
                if e.status in [
                    400,
                    401,
                    403,
                    404,
                    429,
                ]:  # Client errors, often not retryable or rate limit
                    if e.status == 429:  # Rate limit
                        # Try to get retry-after from headers first
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            retry_after = int(retry_after)
                        else:
                            # Try to parse from response body
                            try:
                                response_data = json.loads(response_text)
                                if "retryAfter" in response_data:
                                    retry_after = int(
                                        response_data["retryAfter"] / 1000,
                                    )  # Convert from ms to seconds
                                else:
                                    retry_after = 60  # Default to 60 seconds
                            except Exception:
                                retry_after = 60  # Default to 60 seconds

                        system_logger.warning(
                            f"Rate limit hit. Retrying after {retry_after} seconds.",
                        )
                        await asyncio.sleep(retry_after)
                        # Don't increment attempt for rate limit, allow more retries if needed
                        continue
                    raise  # Re-raise for non-retryable client errors
                if e.status >= 500:  # Server errors, often retryable
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        system_logger.warning(
                            f"Server error. Retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})...",
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        system_logger.critical(
                            f"Failed after {max_retries} attempts due to server error for {endpoint}.",
                        )
                        raise
                else:  # Other HTTP errors
                    raise

            except TimeoutError:
                system_logger.error(
                    f"Request timed out for {method} {endpoint} (attempt {attempt + 1}/{max_retries}).",
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Short delay before retrying timeout
                else:
                    system_logger.critical(
                        f"Failed after {max_retries} attempts due to timeout for {endpoint}.",
                    )
                    raise

            except aiohttp.ClientError as e:
                # Catch broader aiohttp client errors (e.g., connection issues)
                system_logger.error(
                    f"Aiohttp Client Error for {method} {endpoint}: {e} (attempt {attempt + 1}/{max_retries}).",
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    system_logger.critical(
                        f"Failed after {max_retries} attempts due to client error for {endpoint}.",
                    )
                    raise
            except Exception as e:
                # Catch any other unexpected errors during the request
                system_logger.error(
                    f"An unexpected error occurred during API request to {endpoint}: {e}",
                    exc_info=True,
                )
                raise  # Re-raise to ensure it's handled upstream

        system_logger.critical(
            f"Request to {endpoint} failed after {max_retries} attempts.",
        )
        return None  # Should not be reached if exceptions are re-raised

    # --- REST API Functions ---
    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Get kline/candlestick data for a symbol."""
        logger.info("🔧 Starting get_klines request:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Interval: {interval}")
        logger.info(f"   Limit: {limit}")

        start_time = time.time()
        try:
            endpoint = "/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }

            logger.info(f"🔧 Making API request to {endpoint}")
            logger.info(f"   Parameters: {params}")

            # Klines requests have weight 1 per request
            response = await self._request("GET", endpoint, params, request_weight=1)

            request_duration = time.time() - start_time
            logger.info(f"⏱️  API request completed in {request_duration:.2f} seconds")

            if response and isinstance(response, list):
                logger.info(f"✅ Successfully received {len(response)} klines")
                return response
            logger.error(f"💥 Invalid response format: {type(response)}")
            logger.error(f"   Response: {response}")
            return []

        except Exception as e:
            request_duration = time.time() - start_time
            logger.error(
                f"💥 Error in get_klines after {request_duration:.2f} seconds: {e}",
            )
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get historical kline/candlestick data for a symbol within a time range with pagination."""
        logger.info("🔧 Starting get_historical_klines request:")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Interval: {interval}")
        logger.info(f"   Start time: {datetime.fromtimestamp(start_time_ms / 1000)}")
        logger.info(f"   End time: {datetime.fromtimestamp(end_time_ms / 1000)}")
        logger.info(f"   Limit: {limit}")

        start_time = time.time()
        all_klines = []
        current_start_time = start_time_ms

        try:
            while current_start_time < end_time_ms:
                endpoint = "/fapi/v1/klines"
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start_time,
                    "endTime": end_time_ms,
                    "limit": limit,
                }

                logger.info(f"🔧 Making API request to {endpoint}")
                logger.info(f"   Parameters: {params}")

                # Historical klines requests have weight 1 per request
                response = await self._request(
                    "GET",
                    endpoint,
                    params,
                    request_weight=1,
                )

                if not response or not isinstance(response, list):
                    logger.error(f"💥 Invalid response format: {type(response)}")
                    logger.error(f"   Response: {response}")
                    break

                if not response:  # No more data
                    logger.info("No more klines available")
                    break

                all_klines.extend(response)
                logger.info(
                    f"✅ Received {len(response)} klines (total: {len(all_klines)})",
                )

                # Update start time for next request
                # The last kline's open time + interval duration
                last_kline = response[-1]
                last_open_time = last_kline[0]  # First element is open time

                # Calculate interval duration in milliseconds
                interval_ms = self._get_interval_ms(interval)
                current_start_time = last_open_time + interval_ms

                # If we got less than the limit, we've reached the end
                if len(response) < limit:
                    logger.info(
                        f"Received {len(response)} klines (less than {limit}), stopping",
                    )
                    break

                # Add a small delay between requests to respect rate limits
                await asyncio.sleep(0.1)

            request_duration = time.time() - start_time
            logger.info(
                f"⏱️  All API requests completed in {request_duration:.2f} seconds",
            )
            logger.info(
                f"✅ Successfully received {len(all_klines)} total historical klines",
            )
            return all_klines

        except Exception as e:
            request_duration = time.time() - start_time
            logger.error(
                f"💥 Error in get_historical_klines after {request_duration:.2f} seconds: {e}",
            )
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
            raise

    def _get_interval_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        interval_map = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "3d": 3 * 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000,  # Approximate
        }
        return interval_map.get(interval, 60 * 1000)  # Default to 1 minute

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to create order", "status": "failed"},
    )
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
        time_in_force: str = None,
        params: dict[str, Any] = None,
    ):
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
        elif order_type.upper() == "LIMIT":
            order_params["timeInForce"] = "GTC"

        # Merge additional params from the caller
        if params:
            order_params.update(params)

        try:
            return await self._request("POST", endpoint, order_params, signed=True)
        except Exception as e:
            system_logger.error(
                f"Failed to create {order_type} order for {symbol} ({side} {quantity}): {e}",
            )
            # Return a structured error response or re-raise based on upstream needs
            return {"error": str(e), "status": "failed"}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to get order status"},
    )
    async def get_order_status(self, symbol: str, order_id: int):
        """Retrieves the status of a specific order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper(), "orderId": order_id}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(
                f"Failed to get status for order {order_id} on {symbol}: {e}",
            )
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to cancel order"},
    )
    async def cancel_order(self, symbol: str, order_id: int):
        """Cancels an open order."""
        endpoint = "/fapi/v1/order"
        params = {"symbol": symbol.upper(), "orderId": order_id}
        try:
            return await self._request("DELETE", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to cancel order {order_id} on {symbol}: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to get account info"},
    )
    async def get_account_info(self):
        """Fetches account information, including balances and positions."""
        endpoint = "/fapi/v2/account"
        try:
            return await self._request("GET", endpoint, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}

    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_position_risk(self, symbol: str = None):
        """Gets current position risk for all symbols or a specific symbol."""
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(
                f"Failed to get position risk for {symbol or 'all symbols'}: {e}",
            )
            return []

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_open_orders(self, symbol: str = None) -> list[dict[str, Any]]:
        """
        Retrieves all open orders for a given symbol or all symbols.
        """
        endpoint = "/fapi/v1/openOrders"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(
                f"Failed to get open orders for {symbol or 'all symbols'}: {e}",
            )
            return []

    @retry_on_rate_limit()
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="close_all_positions",
    )
    async def close_all_positions(self, symbol: str = None):
        """
        Closes all open positions for a given symbol or all symbols.
        This will place market orders to close positions.
        """
        system_logger.warning(
            f"Attempting to close all open positions for {symbol if symbol else 'all symbols'}...",
        )
        try:
            positions = await self.get_position_risk(symbol)
            for position in positions:
                position_amount = float(position.get("positionAmt", 0))
                if position_amount != 0:
                    current_symbol = position["symbol"]
                    side = "SELL" if position_amount > 0 else "BUY"
                    quantity = abs(position_amount)

                    system_logger.info(
                        f"Closing {side} position for {current_symbol} with quantity {quantity}...",
                    )
                    order_response = await self.create_order(
                        symbol=current_symbol,
                        side=side,
                        order_type="MARKET",
                        quantity=quantity,
                    )
                    if order_response and order_response.get("status") == "failed":
                        system_logger.error(
                            f"Failed to place closing order for {current_symbol}: {order_response.get('error')}",
                        )
                    else:
                        system_logger.info(
                            f"Closed position for {current_symbol}: {order_response}",
                        )
                else:
                    system_logger.debug(
                        f"No open position for {position.get('symbol', 'N/A')}.",
                    )
        except Exception as e:
            system_logger.error(f"Error closing all positions: {e}", exc_info=True)

    @retry_on_rate_limit()
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="cancel_all_orders",
    )
    async def cancel_all_orders(self, symbol: str = None):
        """
        Cancels all open orders for a given symbol or all symbols.
        """
        system_logger.warning(
            f"Attempting to cancel all open orders for {symbol if symbol else 'all symbols'}...",
        )
        try:
            open_orders = await self.get_open_orders(symbol)
            if not open_orders:
                system_logger.info(
                    f"No open orders found for {symbol if symbol else 'all symbols'}.",
                )
                return

            for order in open_orders:
                order_id = order["orderId"]
                order_symbol = order["symbol"]
                system_logger.info(f"Cancelling order {order_id} for {order_symbol}...")
                cancel_response = await self.cancel_order(order_symbol, order_id)
                if cancel_response and cancel_response.get("status") == "failed":
                    system_logger.error(
                        f"Failed to cancel order {order_id} for {order_symbol}: {cancel_response.get('error')}",
                    )
                else:
                    system_logger.info(
                        f"Cancelled order {order_id} for {order_symbol}: {cancel_response}",
                    )
        except Exception as e:
            system_logger.error(f"Error cancelling all orders: {e}", exc_info=True)

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_aggregate_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Get aggregate trades for a symbol within a time range.
        This is an alias for get_historical_agg_trades for compatibility.
        """
        return await self.get_historical_agg_trades(
            symbol,
            start_time_ms,
            end_time_ms,
            limit,
        )

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_historical_agg_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Fetches historical aggregated trades for a symbol within a time range.
        Binance API limits: max 1000 trades per request.
        Aggregated trades have strict rate limits (10 requests per minute).
        """
        endpoint = "/fapi/v1/aggTrades"
        all_trades = []
        current_start_time = start_time_ms
        max_iterations = 100  # Prevent infinite loops
        iteration_count = 0

        while current_start_time < end_time_ms and iteration_count < max_iterations:
            params = {
                "symbol": symbol.upper(),
                "startTime": current_start_time,
                "endTime": end_time_ms,
                "limit": limit,
            }
            try:
                # Use weight 1 since the rate limiter handles aggTrades specially
                trades = await self._request("GET", endpoint, params, request_weight=1)

                # Handle empty or None response
                if not trades:
                    system_logger.info(
                        f"No more trades found for {symbol} at time {current_start_time}",
                    )
                    break

                all_trades.extend(trades)

                # Check if we have valid trade data
                if not trades or len(trades) == 0:
                    system_logger.info(f"Empty trade response for {symbol}, stopping")
                    break

                # Update start time for next iteration
                current_start_time = (
                    trades[-1]["T"] + 1
                )  # 'T' is the trade time in milliseconds

                # If fewer than limit trades, likely reached end of data for the period
                if len(trades) < limit:
                    system_logger.info(
                        f"Reached end of trade data for {symbol}, got {len(trades)} trades",
                    )
                    break

                # Add a small delay between requests to be extra careful with rate limits
                await asyncio.sleep(0.2)  # Reduced from 0.5 to 0.2 seconds
                iteration_count += 1

            except Exception as e:
                system_logger.error(
                    f"Error fetching historical agg trades for {symbol}: {e}",
                )
                break  # Stop fetching for this period on error

        if iteration_count >= max_iterations:
            system_logger.warning(
                f"Reached max iterations ({max_iterations}) for {symbol} agg trades",
            )

        return all_trades

    async def get_historical_agg_trades_ccxt(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Fetches historical aggregated trades using CCXT for better rate limiting.
        CCXT handles rate limits automatically and is more reliable.
        """
        try:
            # Initialize CCXT exchange
            exchange = ccxt.binance(
                {
                    "apiKey": self._api_key,
                    "secret": self._api_secret.decode("utf-8")
                    if self._api_secret
                    else None,
                    "sandbox": False,  # Use live environment
                    "enableRateLimit": True,  # Enable built-in rate limiting
                    "options": {
                        "defaultType": "future",  # Use futures API
                    },
                },
            )

            # Convert timestamps to datetime objects for CCXT
            start_time = datetime.fromtimestamp(start_time_ms / 1000)
            end_time = datetime.fromtimestamp(end_time_ms / 1000)

            system_logger.info(
                f"CCXT: Fetching agg trades for {symbol} from {start_time} to {end_time}",
            )

            # Use CCXT's direct API call to get aggregated trades
            # CCXT doesn't have a built-in aggregate trades method, so we'll use the raw API
            params = {
                "symbol": symbol.upper(),
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": limit,
            }

            # Use CCXT's direct API call method
            trades = await exchange.fapiPublicGetAggTrades(params)

            # Debug: Check what we're getting from the API
            if trades:
                system_logger.info(
                    f"CCXT Debug: First trade timestamp: {trades[0].get('T')}",
                )
                system_logger.info(
                    f"CCXT Debug: Last trade timestamp: {trades[-1].get('T')}",
                )
                system_logger.info(
                    f"CCXT Debug: Number of unique timestamps: {len(set(t.get('T') for t in trades))}",
                )

            # Convert CCXT format to our expected format
            formatted_trades = []
            for trade in trades:
                formatted_trade = {
                    "a": trade.get("a", 0),  # Aggregate trade ID
                    "p": float(trade.get("p", 0)),  # Price
                    "q": float(trade.get("q", 0)),  # Quantity
                    "f": trade.get("f", 0),  # First trade ID
                    "l": trade.get("l", 0),  # Last trade ID
                    "T": trade.get("T", 0),  # Trade time
                    "m": trade.get("m", False),  # Is buyer maker
                    "M": trade.get("M", False),  # Best match
                }
                formatted_trades.append(formatted_trade)

            system_logger.info(
                f"CCXT: Retrieved {len(formatted_trades)} aggregated trades for {symbol}",
            )

            # Close CCXT exchange
            await exchange.close()

            return formatted_trades

        except Exception as e:
            system_logger.error(
                f"CCXT Error fetching historical agg trades for {symbol}: {e}",
            )
            return []

    async def get_historical_klines_ccxt(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Fetches historical kline data using CCXT for better rate limiting.
        CCXT handles rate limits automatically and is more reliable.
        """
        try:
            # Initialize CCXT exchange
            exchange = ccxt.binance(
                {
                    "apiKey": self._api_key,
                    "secret": self._api_secret.decode("utf-8")
                    if self._api_secret
                    else None,
                    "sandbox": False,  # Use live environment
                    "enableRateLimit": True,  # Enable built-in rate limiting
                    "options": {
                        "defaultType": "future",  # Use futures API
                    },
                },
            )

            # Convert timestamps to datetime objects for CCXT
            start_time = datetime.fromtimestamp(start_time_ms / 1000)
            end_time = datetime.fromtimestamp(end_time_ms / 1000)

            system_logger.info(
                f"CCXT: Fetching klines for {symbol} from {start_time} to {end_time}",
            )

            # Use CCXT's fetch_ohlcv method for klines
            # Convert interval to CCXT format if needed
            ccxt_interval = interval  # Binance intervals are already in CCXT format

            # Fetch klines using CCXT
            klines = await exchange.fetch_ohlcv(
                symbol=symbol.upper(),
                timeframe=ccxt_interval,
                since=start_time_ms,
                limit=limit,
            )

            # Convert CCXT format to our expected format
            formatted_klines = []
            for kline in klines:
                # CCXT format: [timestamp, open, high, low, close, volume]
                formatted_kline = [
                    kline[0],  # timestamp
                    str(kline[1]),  # open
                    str(kline[2]),  # high
                    str(kline[3]),  # low
                    str(kline[4]),  # close
                    str(kline[5]),  # volume
                    kline[0] + self._get_interval_ms(interval),  # close_time
                    "0",  # quote_asset_volume
                    0,  # number_of_trades
                    "0",  # taker_buy_base_asset_volume
                    "0",  # taker_buy_quote_asset_volume
                    "0",  # ignore
                ]
                formatted_klines.append(formatted_kline)

            system_logger.info(
                f"CCXT: Retrieved {len(formatted_klines)} klines for {symbol}",
            )

            # Close CCXT exchange
            await exchange.close()

            return formatted_klines

        except Exception as e:
            system_logger.error(
                f"CCXT Error fetching historical klines for {symbol}: {e}",
            )
            return []

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def futures_funding_rate(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[dict[str, Any]]:
        """
        Get futures funding rates for a symbol within a time range.
        This is an alias for the funding rates part of get_historical_futures_data.
        """
        futures_data = await self.get_historical_futures_data(
            symbol,
            start_time_ms,
            end_time_ms,
        )
        return futures_data.get("funding_rates", [])

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"funding_rates": [], "open_interest": []},
    )
    async def get_historical_futures_data(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetches historical futures-specific data (funding rates).
        """
        funding_rates = []

        # Fetch Funding Rates
        fr_endpoint = "/fapi/v1/fundingRate"
        current_fr_start = start_time_ms
        while current_fr_start < end_time_ms:
            params = {
                "symbol": symbol.upper(),
                "startTime": current_fr_start,
                "endTime": end_time_ms,
                "limit": 1000,
            }
            try:
                # Funding rates requests have weight 1 per request
                rates = await self._request(
                    "GET",
                    fr_endpoint,
                    params,
                    request_weight=1,
                )
                if not rates:
                    break
                funding_rates.extend(rates)
                current_fr_start = rates[-1]["fundingTime"] + 1
                # Rate limiting is now handled by the _request method
            except Exception as e:
                system_logger.error(
                    f"Error fetching historical funding rates for {symbol}: {e}",
                )
                break

        return {"funding_rates": funding_rates}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_exchange_info(self):
        """Fetches exchange information including symbols, filters, and trading rules."""
        endpoint = "/fapi/v1/exchangeInfo"
        try:
            return await self._request("GET", endpoint)
        except Exception as e:
            system_logger.error(f"Failed to get exchange info: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return={})
    async def get_ticker(self, symbol: str = None):
        """Fetches 24hr ticker price change statistics."""
        endpoint = "/fapi/v1/ticker/24hr"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params)
        except Exception as e:
            system_logger.error(
                f"Failed to get ticker for {symbol or 'all symbols'}: {e}",
            )
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_funding_rate(self, symbol: str = None):
        """Fetches current funding rate for a symbol or all symbols."""
        endpoint = "/fapi/v1/fundingRate"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params)
        except Exception as e:
            system_logger.error(
                f"Failed to get funding rate for {symbol or 'all symbols'}: {e}",
            )
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_open_interest(self, symbol: str):
        """Fetches current open interest for a symbol."""
        endpoint = "/fapi/v1/openInterest"
        params = {"symbol": symbol.upper()}
        try:
            return await self._request("GET", endpoint, params)
        except Exception as e:
            system_logger.error(f"Failed to get open interest for {symbol}: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_all_orders(
        self,
        symbol: str,
        order_id: int = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500,
    ):
        """Fetches all orders for a symbol."""
        endpoint = "/fapi/v1/allOrders"
        params = {"symbol": symbol.upper(), "limit": limit}
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get all orders for {symbol}: {e}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_trade_history(
        self,
        symbol: str,
        order_id: int = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500,
    ):
        """Fetches trade history for a symbol."""
        endpoint = "/fapi/v1/userTrades"
        params = {"symbol": symbol.upper(), "limit": limit}
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get trade history for {symbol}: {e}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to create batch orders"},
    )
    async def create_batch_orders(self, orders: list[dict[str, Any]]):
        """Creates multiple orders in a single request."""
        endpoint = "/fapi/v1/batchOrders"
        params = {"batchOrders": json.dumps(orders)}
        try:
            return await self._request("POST", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to create batch orders: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to cancel batch orders"},
    )
    async def cancel_batch_orders(self, orders: list[dict[str, Any]]):
        """Cancels multiple orders in a single request."""
        endpoint = "/fapi/v1/batchOrders"
        params = {"batchOrders": json.dumps(orders)}
        try:
            return await self._request("DELETE", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to cancel batch orders: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_income_history(
        self,
        symbol: str = None,
        income_type: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500,
    ):
        """Fetches income history (realized PnL, funding, etc.)."""
        endpoint = "/fapi/v1/income"
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if income_type:
            params["incomeType"] = income_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get income history: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_commission_rate(self, symbol: str):
        """Fetches commission rate for a symbol."""
        endpoint = "/fapi/v1/commissionRate"
        params = {"symbol": symbol.upper()}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get commission rate for {symbol}: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=[],
    )
    async def get_adl_quantile(self, symbol: str = None):
        """Fetches ADL (Auto-Deleveraging) quantile for positions."""
        endpoint = "/fapi/v1/adlQuantile"
        params = {"symbol": symbol.upper()} if symbol else {}
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(
                f"Failed to get ADL quantile for {symbol or 'all symbols'}: {e}",
            )
            return []

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_force_orders(
        self,
        symbol: str = None,
        auto_close_type: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = 500,
    ):
        """Fetches force orders (liquidation, ADL, etc.)."""
        endpoint = "/fapi/v1/forceOrders"
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if auto_close_type:
            params["autoCloseType"] = auto_close_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        try:
            return await self._request("GET", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get force orders: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_position_side_dual(self):
        """Checks if dual side position mode is enabled."""
        endpoint = "/fapi/v1/positionSide/dual"
        try:
            return await self._request("GET", endpoint, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get position side dual status: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def change_position_side_dual(self, dual_side_position: bool):
        """Changes dual side position mode."""
        endpoint = "/fapi/v1/positionSide/dual"
        params = {"dualSidePosition": str(dual_side_position).lower()}
        try:
            return await self._request("POST", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to change position side dual mode: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def get_multi_assets_margin(self):
        """Gets multi-assets margin mode status."""
        endpoint = "/fapi/v1/multiAssetsMargin"
        try:
            return await self._request("GET", endpoint, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to get multi-assets margin status: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={},
    )
    async def change_multi_assets_margin(self, multi_assets_margin: bool):
        """Changes multi-assets margin mode."""
        endpoint = "/fapi/v1/multiAssetsMargin"
        params = {"multiAssetsMargin": str(multi_assets_margin).lower()}
        try:
            return await self._request("POST", endpoint, params, signed=True)
        except Exception as e:
            system_logger.error(f"Failed to change multi-assets margin mode: {e}")
            return {}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=0.0,
    )
    async def get_account_balance(self, asset: str = "USDT") -> float:
        """Gets the balance of a specific asset in the futures account."""
        try:
            account_info = await self.get_account_info()
            assets = account_info.get("assets", [])

            for asset_info in assets:
                if asset_info.get("asset") == asset:
                    return float(asset_info.get("walletBalance", 0))

            return 0.0
        except Exception as e:
            system_logger.error(f"Failed to get {asset} balance: {e}")
            return 0.0

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=0.0,
    )
    async def get_spot_balance(self, asset: str = "USDT") -> float:
        """Gets the balance of a specific asset in the spot account."""
        try:
            # Use spot API endpoint
            endpoint = "/api/v3/account"
            response = await self._request("GET", endpoint, signed=True)

            balances = response.get("balances", [])
            for balance in balances:
                if balance.get("asset") == asset:
                    return float(balance.get("free", 0))

            return 0.0
        except Exception as e:
            system_logger.error(f"Failed to get spot {asset} balance: {e}")
            return 0.0

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return=False,
    )
    async def transfer_to_spot(self, asset: str, amount: float) -> bool:
        """Transfers funds from futures to spot account."""
        try:
            # Use futures transfer endpoint
            endpoint = "/fapi/v1/transfer"
            params = {
                "asset": asset,
                "amount": amount,
                "type": 1,
            }  # 1 = futures to spot

            response = await self._request("POST", endpoint, params, signed=True)

            if response.get("status") == "CONFIRMED":
                system_logger.info(
                    f"Successfully transferred {amount} {asset} to spot account",
                )
                return True
            system_logger.error(f"Transfer failed: {response}")
            return False

        except Exception as e:
            system_logger.error(f"Failed to transfer {amount} {asset} to spot: {e}")
            return False

    # --- WebSocket Handlers ---
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="websocket_handler",
    )
    async def _websocket_handler(self, url: str, callback: Callable, name: str):
        """Generic WebSocket handler with reconnection logic."""
        system_logger.info(f"Connecting to {name} WebSocket: {url}")
        while True:
            try:
                async with websockets.connect(url) as ws:
                    system_logger.info(f"{name} WebSocket connected.")
                    while True:
                        try:
                            data = await ws.recv()
                            await callback(json.loads(data))
                        except json.JSONDecodeError as e:
                            system_logger.error(
                                f"WebSocket JSON Decode Error for {name}: {e}. Raw data: {data[:200]}...",
                            )
                        except Exception as e:
                            system_logger.error(
                                f"Error processing WebSocket message for {name}: {e}",
                                exc_info=True,
                            )
            except websockets.exceptions.ConnectionClosed as e:
                system_logger.warning(
                    f"{name} WebSocket disconnected: {e}. Reconnecting in 5 seconds...",
                )
                await asyncio.sleep(5)
            except TimeoutError:
                system_logger.warning(
                    f"{name} WebSocket connection timed out. Reconnecting in 5 seconds...",
                )
                await asyncio.sleep(5)
            except Exception as e:
                system_logger.error(
                    f"An unexpected error occurred in {name} WebSocket: {e}. Reconnecting in 10 seconds...",
                    exc_info=True,
                )
                await asyncio.sleep(10)

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=None,
        context="process_depth_message",
    )
    async def _process_depth_message(self, data: dict[str, Any]):
        """Callback to process incoming depth stream messages."""
        try:
            for bid in data.get("b", []):
                price, qty = float(bid[0]), float(bid[1])
                if qty == 0:
                    if price in self.order_book["bids"]:
                        del self.order_book["bids"][price]
                else:
                    self.order_book["bids"][price] = qty
            for ask in data.get("a", []):
                price, qty = float(ask[0]), float(ask[1])
                if qty == 0:
                    if price in self.order_book["asks"]:
                        del self.order_book["asks"][price]
                else:
                    self.order_book["asks"][price] = qty
        except Exception as e:
            system_logger.error(
                f"Error processing depth message: {e}. Data: {data}",
                exc_info=True,
            )

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=None,
        context="process_trade_message",
    )
    async def _process_trade_message(self, data: dict[str, Any]):
        """Callback to process incoming trade stream messages."""
        try:
            self.recent_trades.append(data)
            if len(self.recent_trades) > 200:
                self.recent_trades.pop(0)
        except Exception as e:
            system_logger.error(
                f"Error processing trade message: {e}. Data: {data}",
                exc_info=True,
            )

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=None,
        context="process_kline_message",
    )
    async def _process_kline_message(self, data: dict[str, Any]):
        """Callback to process incoming kline stream messages."""
        try:
            kline = data.get("k", {})
            self.kline_data = {
                "open_time": kline.get("t"),
                "open": kline.get("o"),
                "high": kline.get("h"),
                "low": kline.get("l"),
                "close": kline.get("c"),
                "volume": kline.get("v"),
                "close_time": kline.get("T"),
                "is_closed": kline.get("x"),
            }
        except Exception as e:
            system_logger.error(
                f"Error processing kline message: {e}. Data: {data}",
                exc_info=True,
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="start_kline_socket",
    )
    async def start_kline_socket(self, symbol: str, interval: str):
        """Connects to the kline WebSocket stream."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(
            url,
            self._process_kline_message,
            f"Kline ({symbol})",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="start_depth_socket",
    )
    async def start_depth_socket(self, symbol: str):
        """Connects to the depth (order book) WebSocket stream."""
        stream_name = f"{symbol.lower()}@depth"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(
            url,
            self._process_depth_message,
            f"Depth ({symbol})",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="start_trade_socket",
    )
    async def start_trade_socket(self, symbol: str):
        """Connects to the trade WebSocket stream."""
        stream_name = f"{symbol.lower()}@trade"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(
            url,
            self._process_trade_message,
            f"Trade ({symbol})",
        )

    # BaseExchange streaming hooks
    async def subscribe_trades(self, symbol: str, callback):
        async def _cb(msg):
            try:
                await callback(msg)
            except Exception as e:
                logger.error(f"Error in trade subscription callback for {symbol}: {e}", exc_info=True)
        await self._websocket_handler(
            f"{self.WS_BASE_URL}/ws/{symbol.lower()}@trade",
            _cb,
            f"Trade ({symbol})",
        )

    async def subscribe_ticker(self, symbol: str, callback):
        async def _cb(msg):
            try:
                await callback(msg)
            except Exception:
                pass
        await self._websocket_handler(
            f"{self.WS_BASE_URL}/ws/{symbol.lower()}@markPrice",
            _cb,
            f"Mark Price ({symbol})",
        )

    async def subscribe_order_book(self, symbol: str, callback):
        async def _cb(msg):
            try:
                await callback(msg)
            except Exception:
                pass
        await self._websocket_handler(
            f"{self.WS_BASE_URL}/ws/{symbol.lower()}@depth",
            _cb,
            f"Depth ({symbol})",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="start_user_data_socket",
    )
    async def start_user_data_socket(self, callback: Callable):
        """Connects to the user data stream for account and order updates."""
        while True:
            try:
                listen_key_resp = await self._request(
                    "POST",
                    "/fapi/v1/listenKey",
                    signed=True,
                )
                listen_key = listen_key_resp.get("listenKey")
                if not listen_key:
                    system_logger.error(
                        "Failed to get listenKey for user data stream. Retrying in 30 seconds.",
                    )
                    await asyncio.sleep(30)
                    continue

                url = f"{self.WS_BASE_URL}/ws/{listen_key}"

                async def keepalive():
                    while True:
                        await asyncio.sleep(30 * 60)  # Keepalive every 30 mins
                        try:
                            await self._request(
                                "PUT",
                                "/fapi/v1/listenKey",
                                signed=True,
                            )
                            system_logger.info(
                                "User data stream listen key keepalive sent.",
                            )
                        except Exception as e:
                            system_logger.error(
                                f"Failed to send user data stream keepalive: {e}",
                            )

                keepalive_task = asyncio.create_task(keepalive())
                ws_handler_task = asyncio.create_task(
                    self._websocket_handler(url, callback, "User Data"),
                )

                await ws_handler_task
                keepalive_task.cancel()
            except Exception as e:
                system_logger.error(
                    f"Failed to start or maintain user data socket: {e}. Retrying in 10 seconds...",
                    exc_info=True,
                )
                await asyncio.sleep(10)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="start_mark_price_socket",
    )
    async def start_mark_price_socket(self, symbol: str):
        """Connects to the mark price WebSocket stream."""
        stream_name = f"{symbol.lower()}@markPrice"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(
            url,
            self._process_mark_price_message,
            f"Mark Price ({symbol})",
        )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="start_funding_rate_socket",
    )
    async def start_funding_rate_socket(self, symbol: str):
        """Connects to the funding rate WebSocket stream."""
        stream_name = f"{symbol.lower()}@fundingRate"
        url = f"{self.WS_BASE_URL}/ws/{stream_name}"
        await self._websocket_handler(
            url,
            self._process_funding_rate_message,
            f"Funding Rate ({symbol})",
        )

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=None,
        context="process_mark_price_message",
    )
    async def _process_mark_price_message(self, data: dict[str, Any]):
        """Callback to process incoming mark price stream messages."""
        try:
            # Store mark price data for potential use
            self.mark_price_data = {
                "symbol": data.get("s"),
                "mark_price": data.get("p"),
                "index_price": data.get("i"),
                "settlement_price": data.get("P"),
                "funding_rate": data.get("r"),
                "next_funding_time": data.get("T"),
            }
        except Exception as e:
            system_logger.error(
                f"Error processing mark price message: {e}. Data: {data}",
                exc_info=True,
            )

    @handle_errors(
        exceptions=(KeyError, TypeError, ValueError),
        default_return=None,
        context="process_funding_rate_message",
    )
    async def _process_funding_rate_message(self, data: dict[str, Any]):
        """Callback to process incoming funding rate stream messages."""
        try:
            # Store funding rate data for potential use
            self.funding_rate_data = {
                "symbol": data.get("s"),
                "funding_rate": data.get("r"),
                "funding_time": data.get("T"),
                "next_funding_time": data.get("N"),
            }
        except Exception as e:
            system_logger.error(
                f"Error processing funding rate message: {e}. Data: {data}",
                exc_info=True,
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="close_session",
    )
    async def close(self):
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            system_logger.info("HTTP session closed.")

    # Implementation of abstract methods from BaseExchange

    async def _initialize_exchange(self) -> None:
        """Initialize the exchange client."""
        # Binance doesn't need special initialization beyond session creation
        await self._get_session()

    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Convert raw exchange data to standardized MarketData format."""
        market_data_list = []
        for candle in raw_data:
            try:
                # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=self._convert_timestamp(candle[0]),  # open_time
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    interval=interval,
                )
                market_data_list.append(market_data)
            except (IndexError, ValueError, TypeError) as e:
                system_logger.warning(
                    f"Failed to convert candle data: {e}. Candle: {candle}",
                )
                continue
        return market_data_list

    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol."""
        # For Binance, the symbol is the market ID
        return symbol

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from exchange."""
        # Use the existing get_klines_raw method but return raw data
        return await self.get_klines_raw(symbol, interval, limit)

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from exchange."""
        return await self.get_account_info()

    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create raw order on exchange."""
        # Handle the time_in_force parameter that Binance requires
        time_in_force = None
        if params and "time_in_force" in params:
            time_in_force = params.pop("time_in_force")

        return await self.create_order(
            symbol,
            side,
            order_type,
            quantity,
            price,
            time_in_force=time_in_force,
            params=params,
        )

    async def _get_position_risk_raw(
        self,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Get raw position risk information from exchange."""
        return await self.get_position_risk(symbol)

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical kline data from exchange."""
        return await self.get_historical_klines(
            symbol,
            interval,
            start_time_ms,
            end_time_ms,
            limit,
        )

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from exchange."""
        return await self.get_historical_agg_trades(
            symbol,
            start_time_ms,
            end_time_ms,
            limit,
        )

    async def _get_open_orders_raw(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get raw open orders from exchange."""
        return await self.get_open_orders(symbol)

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on exchange."""
        return await self.cancel_order(symbol, order_id)

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from exchange."""
        return await self.get_order_status(symbol, order_id)
