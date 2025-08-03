
import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Optional

import ccxt.async_support as ccxt
from ccxt.base.errors import (
    AuthenticationError,
    DDoSProtection,
    ExchangeError,
    ExchangeNotAvailable,
    RateLimitExceeded,
    RequestTimeout,
)

from src.utils.error_handler import handle_network_operations
from src.utils.logger import system_logger
from .base_exchange import BaseExchange
from src.interfaces.base_interfaces import MarketData

logger = logging.getLogger(__name__)


def retry_on_rate_limit(max_retries=5, initial_backoff=1.0):
    """
    A decorator to handle API rate limiting and other transient errors
    with exponential backoff, similar to the one in binance.py.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
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
                    backoff *= 2
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
            raise Exception(f"Exhausted retries for {func.__name__}")

        return wrapper

    return decorator


class OkxExchange(BaseExchange):
    """
    Asynchronous client for interacting with the OKX Futures API using CCXT.
    """

    def __init__(self, api_key: str, api_secret: str, password: str, trade_symbol: str):
        super().__init__(api_key, api_secret, trade_symbol, password)
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': password,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
            },
        })

    async def _get_market_id(self, symbol: str) -> str:
        """Helper to get the market ID for a given symbol."""
        # OKX uses symbols like 'BTC-USDT-SWAP' for perpetuals
        return f"{symbol.replace('USDT', '')}-USDT-SWAP"

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_klines_raw(self, symbol: str, interval: str, limit: int = 500) -> list[dict[str, Any]]:
        """Get kline/candlestick data for a symbol."""
        try:
            market_id = await self._get_market_id(symbol)
            ohlcv = await self.exchange.fetch_ohlcv(market_id, timeframe=interval, limit=limit)
            return [
                {
                    "timestamp": k[0],
                    "open": k[1],
                    "high": k[2],
                    "low": k[3],
                    "close": k[4],
                    "volume": k[5],
                }
                for k in ohlcv
            ]
        except Exception as e:
            logger.error(f"Error fetching klines from OKX for {symbol}: {e}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return={"error": "Failed to create order", "status": "failed"})
    async def create_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, params: dict[str, Any] = None):
        """Creates a new order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.create_order(market_id, order_type, side, quantity, price, params)
        except Exception as e:
            logger.error(f"Error creating order on OKX for {symbol}: {e}")
            return {"error": str(e), "status": "failed"}

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return={"error": "Failed to get order status"})
    async def get_order_status(self, symbol: str, order_id: str):
        """Retrieves the status of a specific order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.fetch_order(order_id, market_id)
        except Exception as e:
            logger.error(f"Failed to get status for order {order_id} on OKX {symbol}: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return={"error": "Failed to cancel order"})
    async def cancel_order(self, symbol: str, order_id: str):
        """Cancels an open order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.cancel_order(order_id, market_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on OKX {symbol}: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return={"error": "Failed to get account info"})
    async def get_account_info(self):
        """Fetches account information, including balances and positions."""
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Failed to get account info from OKX: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_position_risk(self, symbol: str = None):
        """Gets current position risk for all symbols or a specific symbol."""
        try:
            market_id = await self._get_market_id(symbol) if symbol else None
            return await self.exchange.fetch_positions([market_id] if market_id else None)
        except Exception as e:
            logger.error(f"Failed to get position risk from OKX for {symbol or 'all symbols'}: {e}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_open_orders(self, symbol: str = None) -> list[dict[str, Any]]:
        """Retrieves all open orders for a given symbol or all symbols."""
        try:
            market_id = await self._get_market_id(symbol) if symbol else None
            return await self.exchange.fetch_open_orders(market_id)
        except Exception as e:
            logger.error(f"Failed to get open orders from OKX for {symbol or 'all symbols'}: {e}")
            return []

    async def close(self):
        """Closes the CCXT exchange instance."""
        if self.exchange:
            await self.exchange.close()
            system_logger.info("OKX CCXT session closed.")

    # Implementation of abstract methods from BaseExchange

    async def _initialize_exchange(self) -> None:
        """Initialize the exchange client."""
        # OKX doesn't need special initialization beyond CCXT setup
        pass

    async def _convert_to_market_data(self, raw_data: list[dict[str, Any]], symbol: str, interval: str) -> list[MarketData]:
        """Convert raw exchange data to standardized MarketData format."""
        market_data_list = []
        for candle in raw_data:
            try:
                # OKX format: [timestamp, open, high, low, close, volume, ...]
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=self._convert_timestamp(candle[0]),  # timestamp
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    interval=interval
                )
                market_data_list.append(market_data)
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Failed to convert candle data: {e}. Candle: {candle}")
                continue
        return market_data_list

    async def _get_klines_raw(self, symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        """Get raw kline data from exchange."""
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
        price: Optional[float] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create raw order on exchange."""
        return await self.create_order(symbol, side, order_type, quantity, price, params)

    async def _get_position_risk_raw(self, symbol: Optional[str] = None) -> dict[str, Any]:
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
        # OKX doesn't have a direct historical klines method, so we'll use the regular klines
        # This is a limitation of the current implementation
        return await self.get_klines(symbol, interval, limit)

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from exchange."""
        # OKX doesn't have a direct historical trades method, so we'll return empty
        # This is a limitation of the current implementation
        logger.warning("OKX doesn't support historical aggregated trades in current implementation")
        return []

    async def _get_open_orders_raw(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """Get raw open orders from exchange."""
        return await self.get_open_orders(symbol)

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on exchange."""
        return await self.cancel_order(symbol, order_id)

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from exchange."""
        return await self.get_order_status(symbol, order_id)
