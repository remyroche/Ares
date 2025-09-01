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
from src.utils.warning_symbols import (
    connection_error,
    error,
    failed,
    invalid,
    missing,
)


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
            self.print(invalid("Invalid timeout"))
            return False

        # Validate max retries
        if self.max_retries < 0:
            self.print(invalid("Invalid max retries"))
            return False

        # Validate API credentials for live trading
        if not self.use_testnet and (not self.api_key or not self.api_secret):
            self.print(error("API credentials required for live trading"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_network_operations(
        max_retries=3,
        default_return=False,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
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
                msg = "API secret not configured"
                raise ValueError(msg)

            query_string = urlencode(params)
            return hmac.new(
                self.api_secret.encode("utf-8"),
                query_string.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

        except Exception:
            self.print(error("Error generating signature: {e}"))
            return ""

    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid order parameters"),
            AttributeError: (False, "Missing order components"),
            KeyError: (False, "Missing required order data"),
        },
        default_return=False,
        context="order creation",
    )
    async def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | bool | None:
        """Make a signed request; returns JSON dict for GET, True/False for DELETE depending on status."""
        if not self.is_connected or not self.api_key or not self.api_secret:
            self.print(missing("Exchange not connected or missing credentials"))
            return None
        params = {**params, "timestamp": int(time.time() * 1000)}
        params["signature"] = self._generate_signature(params)
        url = f"{self._get_base_url()}{path}"
        headers = {"X-MBX-APIKEY": self.api_key}
        try:
            if method == "GET":
                async with self.session.get(
                    url,
                    params=params,
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    self.print(failed("GET {path} failed: {await resp.text()}"))
                    return None
            if method == "DELETE":
                async with self.session.delete(
                    url,
                    params=params,
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        await resp.read()
                        return True
                    self.print(failed("DELETE {path} failed: {await resp.text()}"))
                    return False
            self.print(error("Unsupported method {method} for {path}"))
            return None
        except aiohttp.ClientError:
            self.print(connection_error("Network error calling {path}: {e}"))
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
    # --- WebSocket fills support (skeleton) ---
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid status parameters"),
            AttributeError: (None, "Missing status components"),
            KeyError: (None, "Missing required status data"),
        },
        default_return=None,
        context="order status",
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_network_operations(
        max_retries=3,
        default_return=None,
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="Binance exchange cleanup",
    )

# Global Binance exchange instance
binance_exchange: BinanceExchange | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="Binance exchange setup",
)