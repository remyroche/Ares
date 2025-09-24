"""
API Client for Live Trading

Handles API communication with exchanges and provides unified interface.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import hmac
import base64
from urllib.parse import urlencode

from .config import TradingConfig
from ..src.interfaces.base_interfaces import MarketData, TradeDecision


class APIMethod(Enum):
    """HTTP method enumeration"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class APIRequest:
    """API request structure"""
    method: APIMethod
    endpoint: str
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retries: int = 3
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class APIResponse:
    """API response structure"""
    success: bool
    data: Dict[str, Any]
    status_code: int
    headers: Dict[str, str]
    timestamp: datetime
    request_time: float
    error: Optional[str] = None
    raw_response: Optional[str] = None


class APIClient:
    """Unified API client for exchange communication"""
    
    def __init__(self, config: TradingConfig, exchange_name: str):
        self.config = config
        self.exchange_name = exchange_name
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.base_url = self._get_base_url()
        self.api_key = self._get_api_key()
        self.api_secret = self._get_api_secret()
        self.api_passphrase = self._get_api_passphrase()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.last_request_time: Dict[str, datetime] = {}
        
        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self.total_request_time = 0.0
        
        # Event handlers
        self.api_handlers: Dict[str, List[Callable[[APIResponse], Awaitable[None]]]] = {
            "on_request_success": [],
            "on_request_error": [],
            "on_rate_limit": []
        }
        
    def _get_base_url(self) -> str:
        """Get base URL for exchange"""
        urls = {
            "binance": "https://api.binance.com",
            "okx": "https://www.okx.com",
            "gateio": "https://api.gateio.ws",
            "mexc": "https://api.mexc.com",
            "phemex": "https://api.phemex.com"
        }
        return urls.get(self.exchange_name.lower(), "")
    
    def _get_api_key(self) -> str:
        """Get API key from config"""
        exchange_config = self.config.exchanges.get(self.exchange_name, {})
        return exchange_config.get("api_key", "")
    
    def _get_api_secret(self) -> str:
        """Get API secret from config"""
        exchange_config = self.config.exchanges.get(self.exchange_name, {})
        return exchange_config.get("api_secret", "")
    
    def _get_api_passphrase(self) -> str:
        """Get API passphrase from config"""
        exchange_config = self.config.exchanges.get(self.exchange_name, {})
        return exchange_config.get("password", "")
    
    async def start(self) -> None:
        """Start the API client"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30.0)
            self.session = aiohttp.ClientSession(timeout=timeout)
            self.logger.info(f"API client started for {self.exchange_name}")
    
    async def stop(self) -> None:
        """Stop the API client"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info(f"API client stopped for {self.exchange_name}")
    
    def register_handler(self, event_type: str, handler: Callable[[APIResponse], Awaitable[None]]) -> None:
        """Register API event handler"""
        if event_type in self.api_handlers:
            self.api_handlers[event_type].append(handler)
    
    async def make_request(self, request: APIRequest) -> APIResponse:
        """Make API request with retry logic and rate limiting"""
        start_time = time.time()
        
        # Check rate limits
        await self._check_rate_limits(request.endpoint)
        
        # Prepare request
        url = f"{self.base_url}{request.endpoint}"
        headers = self._prepare_headers(request)
        
        # Add authentication if needed
        if self._requires_auth(request.endpoint):
            headers.update(self._generate_auth_headers(request))
        
        # Make request with retries
        last_error = None
        for attempt in range(request.retries + 1):
            try:
                async with self.session.request(
                    method=request.method.value,
                    url=url,
                    headers=headers,
                    params=request.params if request.method == APIMethod.GET else None,
                    json=request.data if request.method != APIMethod.GET else None,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    
                    # Read response
                    response_text = await response.text()
                    response_data = {}
                    
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = {"raw": response_text}
                    
                    # Update rate limits
                    await self._update_rate_limits(response.headers)
                    
                    # Create response object
                    api_response = APIResponse(
                        success=200 <= response.status < 300,
                        data=response_data,
                        status_code=response.status,
                        headers=dict(response.headers),
                        timestamp=datetime.now(),
                        request_time=time.time() - start_time,
                        raw_response=response_text
                    )
                    
                    # Update statistics
                    self.request_count += 1
                    self.total_request_time += api_response.request_time
                    
                    if not api_response.success:
                        self.error_count += 1
                        api_response.error = f"HTTP {response.status}: {response_data.get('message', 'Unknown error')}"
                        await self._notify_handlers("on_request_error", api_response)
                    else:
                        await self._notify_handlers("on_request_success", api_response)
                    
                    return api_response
                    
            except asyncio.TimeoutError as e:
                last_error = f"Request timeout: {e}"
                self.logger.warning(f"Request timeout (attempt {attempt + 1}): {request.endpoint}")
                
            except aiohttp.ClientError as e:
                last_error = f"Client error: {e}"
                self.logger.warning(f"Client error (attempt {attempt + 1}): {e}")
                
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                self.logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < request.retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        self.error_count += 1
        error_response = APIResponse(
            success=False,
            data={},
            status_code=0,
            headers={},
            timestamp=datetime.now(),
            request_time=time.time() - start_time,
            error=last_error
        )
        
        await self._notify_handlers("on_request_error", error_response)
        return error_response
    
    async def get_account_info(self) -> APIResponse:
        """Get account information"""
        endpoint = self._get_account_endpoint()
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=endpoint,
            params={}
        )
        return await self.make_request(request)
    
    async def get_ticker(self, symbol: str) -> APIResponse:
        """Get ticker information"""
        endpoint = self._get_ticker_endpoint(symbol)
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=endpoint,
            params={"symbol": symbol}
        )
        return await self.make_request(request)
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> APIResponse:
        """Get order book"""
        endpoint = self._get_orderbook_endpoint(symbol)
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=endpoint,
            params={"symbol": symbol, "limit": limit}
        )
        return await self.make_request(request)
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> APIResponse:
        """Create order"""
        endpoint = self._get_create_order_endpoint()
        data = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity)
        }
        
        if price is not None:
            data["price"] = str(price)
        
        # Add exchange-specific parameters
        data.update(kwargs)
        
        request = APIRequest(
            method=APIMethod.POST,
            endpoint=endpoint,
            data=data
        )
        return await self.make_request(request)
    
    async def cancel_order(self, symbol: str, order_id: str) -> APIResponse:
        """Cancel order"""
        endpoint = self._get_cancel_order_endpoint(symbol, order_id)
        request = APIRequest(
            method=APIMethod.DELETE,
            endpoint=endpoint,
            params={"symbol": symbol, "orderId": order_id}
        )
        return await self.make_request(request)
    
    async def get_order_status(self, symbol: str, order_id: str) -> APIResponse:
        """Get order status"""
        endpoint = self._get_order_status_endpoint(symbol, order_id)
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=endpoint,
            params={"symbol": symbol, "orderId": order_id}
        )
        return await self.make_request(request)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> APIResponse:
        """Get open orders"""
        endpoint = self._get_open_orders_endpoint()
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=endpoint,
            params=params
        )
        return await self.make_request(request)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> APIResponse:
        """Get kline data"""
        endpoint = self._get_klines_endpoint()
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=endpoint,
            params=params
        )
        return await self.make_request(request)
    
    def _get_account_endpoint(self) -> str:
        """Get account endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/account",
            "okx": "/api/v5/account/balance",
            "gateio": "/api/v4/spot/accounts",
            "mexc": "/api/v3/account",
            "phemex": "/accounts/accountPositions"
        }
        return endpoints.get(self.exchange_name.lower(), "/account")
    
    def _get_ticker_endpoint(self, symbol: str) -> str:
        """Get ticker endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/ticker/24hr",
            "okx": "/api/v5/market/ticker",
            "gateio": "/api/v4/spot/tickers",
            "mexc": "/api/v3/ticker/24hr",
            "phemex": "/md/v1/ticker/24hr"
        }
        return endpoints.get(self.exchange_name.lower(), "/ticker")
    
    def _get_orderbook_endpoint(self, symbol: str) -> str:
        """Get orderbook endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/depth",
            "okx": "/api/v5/market/books",
            "gateio": "/api/v4/spot/order_book",
            "mexc": "/api/v3/depth",
            "phemex": "/md/v1/orderbook"
        }
        return endpoints.get(self.exchange_name.lower(), "/orderbook")
    
    def _get_create_order_endpoint(self) -> str:
        """Get create order endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/order",
            "okx": "/api/v5/trade/order",
            "gateio": "/api/v4/spot/orders",
            "mexc": "/api/v3/order",
            "phemex": "/orders"
        }
        return endpoints.get(self.exchange_name.lower(), "/order")
    
    def _get_cancel_order_endpoint(self, symbol: str, order_id: str) -> str:
        """Get cancel order endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/order",
            "okx": "/api/v5/trade/cancel-order",
            "gateio": "/api/v4/spot/orders",
            "mexc": "/api/v3/order",
            "phemex": "/orders"
        }
        return endpoints.get(self.exchange_name.lower(), "/order")
    
    def _get_order_status_endpoint(self, symbol: str, order_id: str) -> str:
        """Get order status endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/order",
            "okx": "/api/v5/trade/order",
            "gateio": "/api/v4/spot/orders",
            "mexc": "/api/v3/order",
            "phemex": "/orders"
        }
        return endpoints.get(self.exchange_name.lower(), "/order")
    
    def _get_open_orders_endpoint(self) -> str:
        """Get open orders endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/openOrders",
            "okx": "/api/v5/trade/orders-pending",
            "gateio": "/api/v4/spot/open_orders",
            "mexc": "/api/v3/openOrders",
            "phemex": "/orders/active"
        }
        return endpoints.get(self.exchange_name.lower(), "/openOrders")
    
    def _get_klines_endpoint(self) -> str:
        """Get klines endpoint for exchange"""
        endpoints = {
            "binance": "/api/v3/klines",
            "okx": "/api/v5/market/candles",
            "gateio": "/api/v4/spot/candlesticks",
            "mexc": "/api/v3/klines",
            "phemex": "/md/v1/klines"
        }
        return endpoints.get(self.exchange_name.lower(), "/klines")
    
    def _requires_auth(self, endpoint: str) -> bool:
        """Check if endpoint requires authentication"""
        auth_endpoints = [
            "/account", "/order", "/balance", "/positions"
        ]
        return any(auth_ep in endpoint for auth_ep in auth_endpoints)
    
    def _prepare_headers(self, request: APIRequest) -> Dict[str, str]:
        """Prepare request headers"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"TradingBot/{self.exchange_name}"
        }
        headers.update(request.headers)
        return headers
    
    def _generate_auth_headers(self, request: APIRequest) -> Dict[str, str]:
        """Generate authentication headers"""
        if self.exchange_name.lower() == "binance":
            return self._generate_binance_auth(request)
        elif self.exchange_name.lower() == "okx":
            return self._generate_okx_auth(request)
        elif self.exchange_name.lower() == "gateio":
            return self._generate_gateio_auth(request)
        elif self.exchange_name.lower() == "mexc":
            return self._generate_mexc_auth(request)
        elif self.exchange_name.lower() == "phemex":
            return self._generate_phemex_auth(request)
        else:
            return {}
    
    def _generate_binance_auth(self, request: APIRequest) -> Dict[str, str]:
        """Generate Binance authentication"""
        timestamp = str(int(time.time() * 1000))
        query_string = urlencode(request.params) if request.params else ""
        
        if request.data:
            query_string += "&" + urlencode(request.data) if query_string else urlencode(request.data)
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-MBX-APIKEY": self.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
    
    def _generate_okx_auth(self, request: APIRequest) -> Dict[str, str]:
        """Generate OKX authentication"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z'
        method = request.method.value
        request_path = request.endpoint
        
        # Create signature
        message = timestamp + method + request_path
        if request.data:
            message += json.dumps(request.data, separators=(',', ':'))
        
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.api_passphrase
        }
    
    def _generate_gateio_auth(self, request: APIRequest) -> Dict[str, str]:
        """Generate Gate.io authentication"""
        timestamp = str(int(time.time()))
        method = request.method.value
        request_path = request.endpoint
        
        # Create signature
        message = method + "\n" + request_path + "\n" + timestamp + "\n"
        if request.data:
            message += json.dumps(request.data, separators=(',', ':'))
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        
        return {
            "KEY": self.api_key,
            "Timestamp": timestamp,
            "SIGN": signature
        }
    
    def _generate_mexc_auth(self, request: APIRequest) -> Dict[str, str]:
        """Generate MEXC authentication"""
        timestamp = str(int(time.time() * 1000))
        query_string = urlencode(request.params) if request.params else ""
        
        if request.data:
            query_string += "&" + urlencode(request.data) if query_string else urlencode(request.data)
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "X-MEXC-APIKEY": self.api_key,
            "timestamp": timestamp,
            "signature": signature
        }
    
    def _generate_phemex_auth(self, request: APIRequest) -> Dict[str, str]:
        """Generate Phemex authentication"""
        timestamp = str(int(time.time() * 1000))
        method = request.method.value
        request_path = request.endpoint
        
        # Create signature
        message = method + request_path + timestamp
        if request.data:
            message += json.dumps(request.data, separators=(',', ':'))
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            "x-phemex-access-token": self.api_key,
            "x-phemex-request-signature": signature,
            "x-phemex-request-timestamp": timestamp
        }
    
    async def _check_rate_limits(self, endpoint: str) -> None:
        """Check and enforce rate limits"""
        if endpoint in self.rate_limits:
            rate_limit = self.rate_limits[endpoint]
            last_request = self.last_request_time.get(endpoint)
            
            if last_request:
                time_since_last = (datetime.now() - last_request).total_seconds()
                min_interval = rate_limit.get("interval", 1.0)
                
                if time_since_last < min_interval:
                    sleep_time = min_interval - time_since_last
                    await asyncio.sleep(sleep_time)
        
        self.last_request_time[endpoint] = datetime.now()
    
    async def _update_rate_limits(self, headers: Dict[str, str]) -> None:
        """Update rate limits from response headers"""
        # Common rate limit headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After"
        ]
        
        for header in rate_limit_headers:
            if header in headers:
                # Parse rate limit information
                # This would be exchange-specific implementation
                pass
    
    async def _notify_handlers(self, event_type: str, response: APIResponse) -> None:
        """Notify registered handlers"""
        if event_type in self.api_handlers:
            for handler in self.api_handlers[event_type]:
                try:
                    await handler(response)
                except Exception as e:
                    self.logger.error(f"Error in API handler: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get API client statistics"""
        return {
            "exchange": self.exchange_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0.0,
            "average_request_time": self.total_request_time / self.request_count if self.request_count > 0 else 0.0,
            "rate_limits": dict(self.rate_limits),
            "timestamp": datetime.now().isoformat()
        }