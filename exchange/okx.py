import asyncio
import logging
from functools import wraps
from typing import Any

import ccxt.async_support as ccxt
from ccxt.base.errors import (
    DDoSProtection,
    ExchangeError,
    ExchangeNotAvailable,
    RateLimitExceeded,
    RequestTimeout,
)

from src.interfaces.base_interfaces import MarketData
from src.utils.error_handler import handle_network_operations
from src.utils.logger import system_logger

from .base_exchange import BaseExchange

import json
import websockets

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
        self.exchange = ccxt.okx(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "password": password,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",
                },
            },
        )

    async def _get_market_id(self, symbol: str) -> str:
        """Helper to get the market ID for a given symbol."""
        # OKX uses symbols like 'BTC-USDT-SWAP' for perpetuals
        return f"{symbol.replace('USDT', '')}-USDT-SWAP"

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Get kline/candlestick data for a symbol."""
        try:
            market_id = await self._get_market_id(symbol)
            ohlcv = await self.exchange.fetch_ohlcv(
                market_id,
                timeframe=interval,
                limit=limit,
            )
            # Return raw CCXT OHLCV (list of lists) for standardized conversion
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching klines from OKX for {symbol}: {e}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to create order", "status": "failed"},
    )
    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
        params: dict[str, Any] | None = None,
    ):
        """Creates a new order with standardized signature."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.create_order(
                market_id,
                order_type,
                side,
                quantity,
                price,
                params,
            )
        except Exception as e:
            logger.error(f"Error creating order on OKX for {symbol}: {e}")
            return {"error": str(e), "status": "failed"}


    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to get order status"},
    )
    async def get_order_status(self, symbol: str, order_id: str):
        """Retrieves the status of a specific order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.fetch_order(order_id, market_id)
        except Exception as e:
            logger.error(
                f"Failed to get status for order {order_id} on OKX {symbol}: {e}",
            )
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to cancel order"},
    )
    async def cancel_order(self, symbol: str, order_id: str):
        """Cancels an open order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.cancel_order(order_id, market_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on OKX {symbol}: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to get account info"},
    )
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
            return await self.exchange.fetch_positions(
                [market_id] if market_id else None,
            )
        except Exception as e:
            logger.error(
                f"Failed to get position risk from OKX for {symbol or 'all symbols'}: {e}",
            )
            return []

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_open_orders(self, symbol: str = None) -> list[dict[str, Any]]:
        """Retrieves all open orders for a given symbol or all symbols."""
        try:
            market_id = await self._get_market_id(symbol) if symbol else None
            return await self.exchange.fetch_open_orders(market_id)
        except Exception as e:
            logger.error(
                f"Failed to get open orders from OKX for {symbol or 'all symbols'}: {e}",
            )
            return []

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_historical_futures_data(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[dict[str, Any]]:
        """Get historical futures data (funding rates) for a symbol within a time range."""
        try:
            market_id = await self._get_market_id(symbol)

            # Prefer CCXT funding rate history if available
            if hasattr(self.exchange, "fetch_funding_rate_history"):
                try:
                    since = start_time_ms
                    all_rates: list[dict[str, Any]] = []
                    # CCXT typical limit defaults; paginate by adjusting since
                    while since < end_time_ms:
                        batch = await self.exchange.fetch_funding_rate_history(
                            market_id,
                            since=since,
                            limit=100,
                        )
                        if not batch:
                            break
                        for item in batch:
                            ts = item.get("timestamp") or item.get("fundingTime") or item.get("time")
                            if ts is None:
                                continue
                            if ts < start_time_ms or ts > end_time_ms:
                                continue
                            all_rates.append(
                                {
                                    "symbol": item.get("symbol", market_id),
                                    "funding_rate": item.get("fundingRate") or item.get("rate") or item.get("fundingRateDaily"),
                                    "funding_time": ts,
                                    "next_funding_time": item.get("nextFundingTime", 0),
                                }
                            )
                        # Filter for valid numeric timestamps to avoid TypeError in max()
                        valid_timestamps = [
                            i.get("timestamp") for i in batch 
                            if i.get("timestamp") is not None and isinstance(i.get("timestamp"), (int, float))
                        ]
                        since = max(valid_timestamps, default=since) + 1
                        await asyncio.sleep(0.1)
                    return all_rates
                except Exception as e:
                    logger.warning(f"CCXT funding rate history failed on OKX: {e}")

            # Fallback to current funding rate (single point) if history unavailable
            try:
                info = await self.exchange.fetch_funding_rate(market_id)
                if info:
                    ts = info.get("timestamp") or info.get("fundingTime") or info.get("time")
                    if ts and start_time_ms <= ts <= end_time_ms:
                        return [
                            {
                                "symbol": info.get("symbol", market_id),
                                "funding_rate": info.get("fundingRate") or info.get("rate"),
                                "funding_time": ts,
                                "next_funding_time": info.get("nextFundingTime", 0),
                            }
                        ]
            except Exception as e:
                logger.warning(f"CCXT fetch_funding_rate failed on OKX: {e}")

            logger.info("No funding rate data available for OKX in the requested range")
            return []
        except Exception as e:
            logger.error(f"Error fetching historical futures data from OKX for {symbol}: {e}")
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
                # CCXT OHLCV format: [timestamp, open, high, low, close, volume]
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=self._convert_timestamp(candle[0]),  # timestamp
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    interval=interval,
                )
                market_data_list.append(market_data)
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Failed to convert candle data: {e}. Candle: {candle}")
                continue
        return market_data_list

    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from exchange."""
        return await self.get_klines_raw(symbol, interval, limit)

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from exchange."""
        try:
            return await self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Failed to get account info from OKX: {e}")
            return {"error": str(e)}

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
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.create_order(
                market_id,
                order_type,
                side,
                quantity,
                price,
                params,
            )
        except Exception as e:
            logger.error(f"Error creating order on OKX for {symbol}: {e}")
            return {"error": str(e), "status": "failed"}

    async def _get_position_risk_raw(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get raw position risk information from exchange."""
        try:
            market_id = await self._get_market_id(symbol) if symbol else None
            return await self.exchange.fetch_positions(
                [market_id] if market_id else None,
            )
        except Exception as e:
            logger.error(
                f"Failed to get position risk from OKX for {symbol or 'all symbols'}: {e}",
            )
            return []

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[list[Any]]:
        """Get raw historical kline data from exchange with pagination."""
        try:
            market_id = await self._get_market_id(symbol)
            since = start_time_ms
            all_ohlcv: list[list[Any]] = []

            while since < end_time_ms:
                ohlcv = await self.exchange.fetch_ohlcv(
                    market_id,
                    timeframe=interval,
                    since=since,
                    limit=limit,
                )
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                # Advance since to 1 ms after last candle
                since = ohlcv[-1][0] + 1
                await asyncio.sleep(0.1)

            return all_ohlcv
        except Exception as e:
            logger.error(
                f"Error fetching historical klines from OKX for {symbol}: {e}",
            )
            return []

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from exchange (standardized format)."""
        try:
            market_id = await self._get_market_id(symbol)
            since = start_time_ms
            all_trades: list[dict[str, Any]] = []
            total_calls = 0
            max_calls = 100

            while since < end_time_ms and total_calls < max_calls:
                trades = await self.exchange.fetch_trades(
                    symbol=market_id,
                    since=since,
                    limit=min(limit, 100),
                )
                if not trades:
                    break

                for t in trades:
                    t_time = t.get("timestamp", 0)
                    if start_time_ms <= t_time <= end_time_ms:
                        all_trades.append(
                            {
                                "a": t.get("id", 0),
                                "p": t.get("price", 0),
                                "q": t.get("amount", 0),
                                "T": t_time,
                                "m": t.get("side", "buy") == "buy"
                                and t.get("takerOrMaker", "taker") == "maker",
                                "f": t.get("id", 0),
                                "l": t.get("id", 0),
                            }
                        )
                total_calls += 1
                since = max(t.get("timestamp", since) for t in trades) + 1
                await asyncio.sleep(0.1)

            return all_trades
        except Exception as e:
            logger.warning(
                "OKX historical aggregated trades not fully supported, returning partial data: %s",
                e,
            )
            return []

    async def _get_open_orders_raw(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get raw open orders from exchange."""
        try:
            market_id = await self._get_market_id(symbol) if symbol else None
            return await self.exchange.fetch_open_orders(market_id)
        except Exception as e:
            logger.error(
                f"Failed to get open orders from OKX for {symbol or 'all symbols'}: {e}",
            )
            return []

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on exchange."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.cancel_order(order_id, market_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on OKX {symbol}: {e}")
            return {"error": str(e)}

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """
        Get raw order status from exchange.
        Must be implemented by subclasses.
        """
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.fetch_order(order_id, market_id)
        except Exception as e:
            logger.error(
                f"Failed to get status for order {order_id} on OKX {symbol}: {e}",
            )
            return {"error": str(e)}

    # --- Streaming hooks (standardized) ---
    async def subscribe_trades(self, symbol: str, callback):
        market_id = await self._get_market_id(symbol)
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sub = {"op": "subscribe", "args": [{"channel": "trades", "instId": market_id}]}

        async def _run():
            while True:
                try:
                    async with websockets.connect(url) as ws:
                        await ws.send(json.dumps(sub))
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                                if msg.get("arg", {}).get("channel") == "trades" and msg.get("arg", {}).get("instId") == market_id:
                                    for t in msg.get("data", []):
                                        std = {
                                            "type": "trade",
                                            "symbol": symbol,
                                            "price": float(t.get("px")),
                                            "qty": float(t.get("sz")),
                                            "side": t.get("side"),
                                            "timestamp": int(t.get("ts", 0)),
                                        }
                                        await callback(std)
                            except Exception:
                                continue
                except Exception:
                    await asyncio.sleep(3)
        import asyncio
        await _run()

    async def subscribe_ticker(self, symbol: str, callback):
        market_id = await self._get_market_id(symbol)
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sub = {"op": "subscribe", "args": [{"channel": "tickers", "instId": market_id}]}

        async def _run():
            while True:
                try:
                    async with websockets.connect(url) as ws:
                        await ws.send(json.dumps(sub))
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                                if msg.get("arg", {}).get("channel") == "tickers" and msg.get("arg", {}).get("instId") == market_id:
                                    t = (msg.get("data") or [{}])[0]
                                    std = {
                                        "type": "ticker",
                                        "symbol": symbol,
                                        "last": float(t.get("last")) if t.get("last") is not None else None,
                                        "bid": float(t.get("bidPx")) if t.get("bidPx") is not None else None,
                                        "ask": float(t.get("askPx")) if t.get("askPx") is not None else None,
                                        "timestamp": int(t.get("ts", 0)),
                                    }
                                    await callback(std)
                            except Exception:
                                continue
                except Exception:
                    await asyncio.sleep(3)
        import asyncio
        await _run()

    async def subscribe_order_book(self, symbol: str, callback):
        market_id = await self._get_market_id(symbol)
        url = "wss://ws.okx.com:8443/ws/v5/public"
        sub = {"op": "subscribe", "args": [{"channel": "books", "instId": market_id}]}

        async def _run():
            while True:
                try:
                    async with websockets.connect(url) as ws:
                        await ws.send(json.dumps(sub))
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                                if msg.get("arg", {}).get("channel") == "books" and msg.get("arg", {}).get("instId") == market_id:
                                    d = (msg.get("data") or [{}])[0]
                                    bids = d.get("bids") or []
                                    asks = d.get("asks") or []
                                    best_bid = float(bids[0][0]) if bids else None
                                    best_ask = float(asks[0][0]) if asks else None
                                    std = {
                                        "type": "order_book",
                                        "symbol": symbol,
                                        "bid": best_bid,
                                        "ask": best_ask,
                                        "timestamp": int(d.get("ts", 0)),
                                    }
                                    await callback(std)
                            except Exception:
                                continue
                except Exception:
                    await asyncio.sleep(3)
        import asyncio
        await _run()
