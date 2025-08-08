import hashlib
import hmac
import time
from typing import Any
from urllib.parse import urlencode

import aiohttp

from src.utils.error_handler import (
    handle_errors,
    handle_network_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class BinanceExchange:
    """
    Enhanced Binance exchange client with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize Binance exchange with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("BinanceExchange")

        # Exchange state
        self.is_connected: bool = False
        self.session: aiohttp.ClientSession | None = None
        self.base_url: str = "https://api.binance.com"
        self.testnet_url: str = "https://testnet.binance.vision"
        # Add futures endpoints
        self.futures_base_url: str = "https://fapi.binance.com"
        self.testnet_futures_url: str = "https://testnet.binancefuture.com"

        # Configuration
        self.exchange_config: dict[str, Any] = self.config.get("binance_exchange", {})
        self.api_key: str | None = self.exchange_config.get("api_key")
        self.api_secret: str | None = self.exchange_config.get("api_secret")
        self.use_testnet: bool = self.exchange_config.get("use_testnet", True)
        self.timeout: int = self.exchange_config.get("timeout", 30)
        self.max_retries: int = self.exchange_config.get("max_retries", 3)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid Binance exchange configuration"),
            AttributeError: (False, "Missing required exchange parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="Binance exchange initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize Binance exchange with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Binance Exchange...")

        # Load exchange configuration
        await self._load_exchange_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error("Invalid configuration for Binance exchange")
            return False

        # Initialize connection
        await self._initialize_connection()

        self.logger.info(
            "✅ Binance Exchange initialization completed successfully",
        )
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exchange configuration loading",
    )
    async def _load_exchange_configuration(self) -> None:
        """Load exchange configuration."""
        # Set default exchange parameters
        self.exchange_config.setdefault("use_testnet", True)
        self.exchange_config.setdefault("timeout", 30)
        self.exchange_config.setdefault("max_retries", 3)
        self.exchange_config.setdefault("rate_limit_enabled", True)
        self.exchange_config.setdefault("rate_limit_requests", 1200)
        self.exchange_config.setdefault("rate_limit_window", 60)

        # Update configuration
        self.api_key = self.exchange_config.get("api_key")
        self.api_secret = self.exchange_config.get("api_secret")
        self.use_testnet = self.exchange_config["use_testnet"]
        self.timeout = self.exchange_config["timeout"]
        self.max_retries = self.exchange_config["max_retries"]

        self.logger.info("Exchange configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate exchange configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate timeout
        if self.timeout <= 0:
            self.logger.error("Invalid timeout")
            return False

        # Validate max retries
        if self.max_retries < 0:
            self.logger.error("Invalid max retries")
            return False

        # Validate API credentials for live trading
        if not self.use_testnet:
            if not self.api_key or not self.api_secret:
                self.logger.error("API credentials required for live trading")
                return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_network_operations(
        max_retries=3,
        default_return=False,
    )
    async def _initialize_connection(self) -> bool:
        """
        Initialize connection to Binance API.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )

            # Test connection
            server_time = await self._get_server_time()
            if server_time:
                self.is_connected = True
                self.logger.info(
                    f"Connected to Binance API (Server time: {server_time})",
                )
                return True
            self.logger.error("Failed to connect to Binance API")
            return False

        except Exception as e:
            self.logger.error(f"Error initializing connection: {e}")
            return False

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def _get_server_time(self) -> int | None:
        """
        Get server time from Binance.

        Returns:
            Optional[int]: Server time in milliseconds or None
        """
        try:
            url = f"{self._get_base_url()}/api/v3/time"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("serverTime")
                self.logger.error(f"Failed to get server time: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            return None

    def _get_base_url(self) -> str:
        """Get base URL based on testnet setting."""
        return self.testnet_url if self.use_testnet else self.base_url

    def _get_futures_base_url(self) -> str:
        """Get futures base URL based on testnet setting."""
        return self.testnet_futures_url if self.use_testnet else self.futures_base_url

    def _generate_signature(self, params: dict[str, Any]) -> str:
        """
        Generate HMAC signature for authenticated requests.

        Args:
            params: Request parameters

        Returns:
            str: HMAC signature
        """
        try:
            if not self.api_secret:
                raise ValueError("API secret not configured")

            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            return signature

        except Exception as e:
            self.logger.error(f"Error generating signature: {e}")
            return ""

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_account_info(self) -> dict[str, Any] | None:
        """
        Get account information.

        Returns:
            Optional[Dict[str, Any]]: Account information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            if not self.api_key or not self.api_secret:
                self.logger.error("API credentials required for account info")
                return None

            # Prepare request
            params = {"timestamp": int(time.time() * 1000)}

            # Add signature
            signature = self._generate_signature(params)
            params["signature"] = signature

            # Make request
            url = f"{self._get_base_url()}/api/v3/account"
            headers = {"X-MBX-APIKEY": self.api_key}

            async with self.session.get(
                url,
                params=params,
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info("Account information retrieved successfully")
                    return data
                self.logger.error(f"Failed to get account info: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_position_risk(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """
        Get position risk information.

        Args:
            symbol: Optional symbol filter

        Returns:
            Optional[List[Dict[str, Any]]]: Position risk information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            if not self.api_key or not self.api_secret:
                self.logger.error("API credentials required for position risk")
                return None

            # Prepare request
            params = {"timestamp": int(time.time() * 1000)}
            params["recvWindow"] = 5000

            if symbol:
                params["symbol"] = symbol

            # Add signature
            signature = self._generate_signature(params)
            params["signature"] = signature

            # Make request
            url = f"{self._get_futures_base_url()}/fapi/v2/positionRisk"
            headers = {"X-MBX-APIKEY": self.api_key}

            async with self.session.get(
                url,
                params=params,
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info("Position risk information retrieved successfully")
                    return data if isinstance(data, list) else [data]
                self.logger.error(f"Failed to get position risk: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting position risk: {e}")
            return None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid order parameters"),
            AttributeError: (False, "Missing order components"),
            KeyError: (False, "Missing required order data"),
        },
        default_return=False,
        context="order creation",
    )
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        time_in_force: str | None = None,
        stop_price: float | None = None,
        new_client_order_id: str | None = None,
        reduce_only: bool | None = None,
        close_on_trigger: bool | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        post_only: bool | None = None,
    ) -> dict[str, Any] | None:
        """
        Create an order.

        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            order_type: Order type ('MARKET' or 'LIMIT')
            quantity: Order quantity
            price: Order price (required for LIMIT orders)

        Returns:
            Optional[Dict[str, Any]]: Order information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            if not self.api_key or not self.api_secret:
                self.logger.error("API credentials required for order creation")
                return None

            # Validate parameters
            if side not in ["BUY", "SELL"]:
                self.logger.error("Invalid order side")
                return None

            if order_type not in ["MARKET", "LIMIT"]:
                self.logger.error("Invalid order type")
                return None

            if order_type == "LIMIT" and price is None:
                self.logger.error("Price required for LIMIT orders")
                return None

            # Prepare request
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": quantity,
                "timestamp": int(time.time() * 1000),
            }

            if price is not None:
                params["price"] = price
            if time_in_force:
                params["timeInForce"] = time_in_force
            if stop_price is not None:
                params["stopPrice"] = stop_price
            if new_client_order_id:
                params["newClientOrderId"] = new_client_order_id
            # reduce_only/close_on_trigger are futures-only; include if supported downstream
            if reduce_only is not None:
                params["reduceOnly"] = str(reduce_only).lower()
            if close_on_trigger is not None:
                params["closePosition"] = str(close_on_trigger).lower()
            if post_only is not None:
                params["postOnly"] = str(post_only).lower()
            # take_profit/stop_loss are strategy-level; for spot we skip; for futures these may map to working orders

            # Add signature
            signature = self._generate_signature(params)
            params["signature"] = signature

            # Make request
            url = f"{self._get_base_url()}/api/v3/order"
            headers = {"X-MBX-APIKEY": self.api_key}

            async with self.session.post(url, data=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(
                        f"Order created successfully: {data.get('orderId')}",
                    )
                    return data
                error_data = await response.json()
                self.logger.error(f"Failed to create order: {error_data}")
                return None

        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None

    async def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | bool | None:
        """Make a signed request; returns JSON dict for GET, True/False for DELETE depending on status."""
        if not self.is_connected or not self.api_key or not self.api_secret:
            self.logger.error("Exchange not connected or missing credentials")
            return None
        params = {**params, "timestamp": int(time.time() * 1000)}
        params["signature"] = self._generate_signature(params)
        url = f"{self._get_base_url()}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}
        try:
            if method == "GET":
                async with self.session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    self.logger.error(f"GET {path} failed: {await resp.text()}")
                    return None
            if method == "DELETE":
                async with self.session.delete(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        await resp.read()
                        return True
                    self.logger.error(f"DELETE {path} failed: {await resp.text()}")
                    return False
            self.logger.error(f"Unsupported method {method} for {path}")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error calling {path}: {e}")
            return None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid cancel parameters"),
            AttributeError: (False, "Missing cancel components"),
            KeyError: (False, "Missing required cancel data"),
        },
        default_return=False,
        context="order cancellation",
    )
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order."""
        result = await self._signed_request(
            method="DELETE",
            path="/api/v3/order",
            params={"symbol": symbol, "orderId": order_id},
        )
        return bool(result)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]] | None:
        """Get all open orders, optionally filtered by symbol."""
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        result = await self._signed_request(
            method="GET",
            path="/api/v3/openOrders",
            params=params,
        )
        return result if isinstance(result, list) else None

    async def set_margin_mode(self, symbol: str, mode: str) -> bool:
        """Set margin mode (isolated/cross). Note: For futures endpoints; stubbed for spot."""
        try:
            # Spot API doesn't support margin mode here; return True for compatibility
            return True
        except Exception:
            return False

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Set leverage for symbol. Note: For futures endpoints; stubbed for spot."""
        try:
            # Spot API doesn't support leverage; return True for compatibility
            return True
        except Exception:
            return False

    # --- WebSocket fills support (skeleton) ---
    async def subscribe_fills(self, callback: callable) -> bool:
        """Subscribe to user trade/fill events and call callback(event_dict)."""
        try:
            # TODO: Implement Binance user data stream listenKey + ws connect
            self._fills_callback = callback
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing to fills: {e}")
            return False

    async def unsubscribe_fills(self) -> bool:
        try:
            self._fills_callback = None
            return True
        except Exception:
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid status parameters"),
            AttributeError: (None, "Missing status components"),
            KeyError: (None, "Missing required status data"),
        },
        default_return=None,
        context="order status",
    )
    async def get_order_status(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        """Get the status of an order."""
        return await self._signed_request(
            method="GET",
            path="/api/v3/order",
            params={"symbol": symbol, "orderId": order_id},
        )

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
    ) -> list[list[Any]] | None:
        """
        Get kline/candlestick data.

        Args:
            symbol: Trading symbol
            interval: Kline interval
            limit: Number of klines to retrieve

        Returns:
            Optional[List[List[Any]]]: Kline data or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            # Prepare request
            params = {"symbol": symbol, "interval": interval, "limit": limit}

            # Make request
            url = f"{self._get_base_url()}/api/v3/klines"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(
                        f"Klines retrieved successfully: {len(data)} records",
                    )
                    return data
                self.logger.error(f"Failed to get klines: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting klines: {e}")
            return None

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_ticker(self, symbol: str) -> dict[str, Any] | None:
        """
        Get ticker information.

        Args:
            symbol: Trading symbol

        Returns:
            Optional[Dict[str, Any]]: Ticker information or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            # Prepare request
            params = {"symbol": symbol}

            # Make request
            url = f"{self._get_base_url()}/api/v3/ticker/24hr"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"Ticker retrieved successfully: {symbol}")
                    return data
                self.logger.error(f"Failed to get ticker: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting ticker: {e}")
            return None

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, Any] | None:
        """
        Get order book.

        Args:
            symbol: Trading symbol
            limit: Number of orders to retrieve

        Returns:
            Optional[Dict[str, Any]]: Order book or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            # Prepare request
            params = {"symbol": symbol, "limit": limit}

            # Make request
            url = f"{self._get_base_url()}/api/v3/depth"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(f"Order book retrieved successfully: {symbol}")
                    return data
                self.logger.error(f"Failed to get order book: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            return None

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_aggregate_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[dict[str, Any]] | None:
        """
        Get aggregate trades for a symbol within a time range.

        Args:
            symbol: Trading symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            List of aggregate trades or None if failed
        """
        try:
            params = {
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000,
            }

            url = f"{self._get_base_url()}/api/v3/aggTrades"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                self.logger.error(
                    f"Failed to get aggregate trades: {response.status}",
                )
                return None

        except Exception as e:
            self.logger.error(f"Error getting aggregate trades: {e}")
            return None

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def get_historical_agg_trades_ccxt(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[dict[str, Any]] | None:
        """
        Get historical aggregated trades data.

        Args:
            symbol: Trading symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            Optional[List[Dict[str, Any]]]: Aggregated trades data or None
        """
        try:
            if not self.is_connected:
                self.logger.error("Exchange not connected")
                return None

            # Prepare request
            params = {
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000,
            }

            # Make request
            url = f"{self._get_base_url()}/api/v3/aggTrades"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.logger.info(
                        f"Aggregated trades retrieved successfully: {len(data)} records",
                    )
                    return data
                self.logger.error(
                    f"Failed to get aggregated trades: {response.status}",
                )
                return None

        except Exception as e:
            self.logger.error(f"Error getting aggregated trades: {e}")
            return None

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    async def futures_funding_rate(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[dict[str, Any]] | None:
        """
        Get futures funding rates for a symbol within a time range.

        Args:
            symbol: Trading symbol
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            List of funding rates or None if failed
        """
        try:
            params = {
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000,
            }

            url = f"{self._get_futures_base_url()}/fapi/v1/fundingRate"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                self.logger.error(f"Failed to get funding rates: {response.status}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting funding rates: {e}")
            return None

    def get_exchange_status(self) -> dict[str, Any]:
        """
        Get exchange status information.

        Returns:
            Dict[str, Any]: Exchange status
        """
        return {
            "is_connected": self.is_connected,
            "use_testnet": self.use_testnet,
            "base_url": self._get_base_url(),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "api_key_configured": bool(self.api_key),
            "api_secret_configured": bool(self.api_secret),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="Binance exchange cleanup",
    )
    async def stop(self) -> None:
        """Stop the Binance exchange."""
        self.logger.info("🛑 Stopping Binance Exchange...")

        try:
            if self.session:
                await self.session.close()
                self.session = None

            self.is_connected = False
            self.logger.info("✅ Binance Exchange stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping Binance exchange: {e}")


# Global Binance exchange instance
binance_exchange: BinanceExchange | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="Binance exchange setup",
)
async def setup_binance_exchange(
    config: dict[str, Any] | None = None,
) -> BinanceExchange | None:
    """
    Setup global Binance exchange.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[BinanceExchange]: Global Binance exchange instance
    """
    try:
        global binance_exchange

        if config is None:
            config = {
                "binance_exchange": {
                    "use_testnet": True,
                    "timeout": 30,
                    "max_retries": 3,
                    "rate_limit_enabled": True,
                    "rate_limit_requests": 1200,
                    "rate_limit_window": 60,
                },
            }

        # Create Binance exchange
        binance_exchange = BinanceExchange(config)

        # Initialize Binance exchange
        success = await binance_exchange.initialize()
        if success:
            return binance_exchange
        return None

    except Exception as e:
        print(f"Error setting up Binance exchange: {e}")
        return None
