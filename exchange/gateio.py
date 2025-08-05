import asyncio
import logging
from datetime import datetime
from functools import wraps
from typing import Any

import aiohttp
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


class GateioExchange(BaseExchange):
    """
    Asynchronous client for interacting with the Gate.io Futures API using CCXT.
    This class provides a consistent interface with other exchange clients in the project.
    """

    def __init__(self, api_key: str, api_secret: str, trade_symbol: str):
        super().__init__(api_key, api_secret, trade_symbol)
        self.exchange = ccxt.gateio(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",  # for perpetual futures
                },
            },
        )

    async def _get_market_id(self, symbol: str) -> str:
        """Helper to get the market ID for a given symbol."""
        return f"{symbol.replace('USDT', '')}_USDT"

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
            logger.error(f"Error fetching klines from Gate.io for {symbol}: {e}")
            return []

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get kline/candlestick data for a symbol (for compatibility with consolidation step)."""
        logger.info(f"🔧 GATEIO: get_klines called for {symbol}")
        logger.info(f"📊 Parameters: interval={interval}, limit={limit}")

        try:
            market_id = await self._get_market_id(symbol)
            logger.info(f"🔧 GATEIO: Market ID resolved to {market_id}")

            ohlcv = await self.exchange.fetch_ohlcv(
                market_id,
                timeframe=interval,
                limit=limit,
            )
            logger.info(
                f"📊 GATEIO: CCXT Response: Received {len(ohlcv) if ohlcv else 0} klines",
            )

            result = [
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

            logger.info(
                f"✅ GATEIO: get_klines completed, returned {len(result)} klines",
            )
            return result
        except Exception as e:
            logger.error(
                f"❌ GATEIO: Error fetching klines from Gate.io for {symbol}: {e}",
            )
            logger.error(f"🔍 GATEIO: Full error details: {type(e).__name__}: {str(e)}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get historical kline/candlestick data for a symbol within a time range."""
        logger.info(f"🔧 GATEIO: get_historical_klines called for {symbol}")
        logger.info(
            f"📊 Parameters: interval={interval}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit={limit}",
        )

        try:
            market_id = await self._get_market_id(symbol)
            logger.info(f"🔧 GATEIO: Market ID resolved to {market_id}")

            since = start_time_ms
            all_klines = []
            call_count = 0

            while since < end_time_ms:
                call_count += 1
                logger.info(
                    f"📡 GATEIO: CCXT Call {call_count}: Fetching klines from {datetime.fromtimestamp(since / 1000)}",
                )
                logger.info(
                    f"🔧 GATEIO: CCXT Parameters: market_id={market_id}, timeframe={interval}, since={since}, limit={limit}",
                )

                ohlcv = await self.exchange.fetch_ohlcv(
                    market_id,
                    timeframe=interval,
                    since=since,
                    limit=limit,
                )

                logger.info(
                    f"📊 GATEIO: CCXT Response: Received {len(ohlcv) if ohlcv else 0} klines",
                )

                if not ohlcv:
                    logger.info(
                        f"⚠️ GATEIO: No more klines available at {datetime.fromtimestamp(since / 1000)}",
                    )
                    break

                klines = [
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

                all_klines.extend(klines)
                logger.info(
                    f"📈 GATEIO: Added {len(klines)} klines, total: {len(all_klines)}",
                )

                # Update since for next iteration
                if len(ohlcv) > 0:
                    since = ohlcv[-1][0] + 1  # Next timestamp after the last kline
                    logger.info(
                        f"⏭️ GATEIO: Advancing to {datetime.fromtimestamp(since / 1000)}",
                    )
                else:
                    logger.info("⏭️ GATEIO: No klines, stopping")
                    break

                # Add small delay to respect rate limits
                await asyncio.sleep(0.1)

            logger.info(
                f"✅ GATEIO: get_historical_klines completed, returned {len(all_klines)} klines",
            )
            return all_klines
        except Exception as e:
            logger.error(
                f"❌ GATEIO: Error fetching historical klines from Gate.io for {symbol}: {e}",
            )
            logger.error(f"🔍 GATEIO: Full error details: {type(e).__name__}: {str(e)}")
            return []

    @retry_on_rate_limit()
    @handle_network_operations(max_retries=3, default_return=[])
    async def get_historical_agg_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get historical aggregated trades for a symbol within a time range."""
        try:
            market_id = await self._get_market_id(symbol)
            since = start_time_ms
            all_trades = []

            logger.info(
                f"   🔍 Fetching historical trades from {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}",
            )

            # Try direct HTTP requests to Gate.io's historical trades API endpoint
            try:
                logger.info(
                    "   🌐 Attempting direct HTTP request to Gate.io historical trades API",
                )

                # Define base headers for Gate.io API
                base_headers = {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Timestamp": str(
                        int(datetime.now().timestamp() * 1000),
                    ),  # Always required
                }

                # Gate.io public API has very limited historical access (only last 10,000 points)
                # Skip direct API calls and go straight to CCXT fallback for historical data
                logger.info(
                    "   ⚠️ Gate.io public API has limited historical access, using CCXT fallback",
                )
                logger.info(
                    f"   📊 Requesting data from {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}",
                )
                logger.info(
                    "   ⏰ This is likely beyond Gate.io's public API limits (10,000 points ago)",
                )

            except Exception as http_error:
                logger.warning(f"Direct HTTP API failed: {http_error}")

            # Fallback to CCXT fetch_trades with pagination
            logger.info("   🔄 Falling back to CCXT fetch_trades with pagination")

            total_calls = 0
            max_calls = 50  # Safety limit to prevent infinite loops

            while since < end_time_ms and total_calls < max_calls:
                try:
                    logger.info(
                        f"   📡 CCXT Call {total_calls + 1}/{max_calls}: Fetching trades from {datetime.fromtimestamp(since / 1000)}",
                    )
                    logger.info(
                        f"   🔧 CCXT Parameters: symbol={market_id}, since={since}, limit={min(limit, 100)}",
                    )

                    trades = await self.exchange.fetch_trades(
                        symbol=market_id,
                        since=since,
                        limit=min(limit, 100),  # CCXT limit for fetch_trades
                    )

                    logger.info(
                        f"   📊 CCXT Response: Received {len(trades) if trades else 0} trades",
                    )

                    if trades and len(trades) > 0:
                        # Log first few trades for debugging
                        for i, trade in enumerate(trades[:3]):
                            logger.info(f"   📋 Trade {i+1}: {trade}")

                    if not trades:
                        logger.info(
                            f"   ⚠️ No more trades available at {datetime.fromtimestamp(since / 1000)}",
                        )
                        break

                    # Filter trades within our time range
                    filtered_trades = []
                    for trade in trades:
                        trade_time = trade.get("timestamp", 0)
                        if start_time_ms <= trade_time <= end_time_ms:
                            formatted_trade = {
                                "agg_trade_id": trade.get("id", 0),
                                "price": trade.get("price", 0),
                                "quantity": trade.get("amount", 0),
                                "timestamp": trade_time,
                                "is_buyer_maker": trade.get("side", "buy") == "buy"
                                and trade.get("takerOrMaker", "taker") == "maker",
                            }
                            filtered_trades.append(formatted_trade)

                    all_trades.extend(filtered_trades)
                    logger.info(
                        f"   📊 Call {total_calls + 1}/{max_calls}: Got {len(trades)} trades, filtered to {len(filtered_trades)} in range",
                    )

                    # Advance to next batch (1ms after last trade)
                    if trades:
                        last_trade_time = max(
                            trade.get("timestamp", 0) for trade in trades
                        )
                        since = last_trade_time + 1
                        logger.info(
                            f"   ⏭️ Advancing to {datetime.fromtimestamp(since / 1000)}",
                        )
                    else:
                        since += 86400000  # Advance by 1 day if no trades
                        logger.info(
                            f"   ⏭️ No trades, advancing by 1 day to {datetime.fromtimestamp(since / 1000)}",
                        )

                    total_calls += 1
                    await asyncio.sleep(0.1)  # Rate limiting

                except Exception as e:
                    logger.error(f"   ❌ Error in CCXT fallback: {e}")
                    logger.error(
                        f"   🔍 Full error details: {type(e).__name__}: {str(e)}",
                    )
                    break

            logger.info(f"   📈 Total trades collected: {len(all_trades)}")
            logger.info(
                f"✅ GATEIO: get_historical_agg_trades completed, returned {len(all_trades)} trades",
            )
            return all_trades

        except Exception as e:
            logger.error(
                f"❌ GATEIO: Error fetching historical trades from Gate.io for {symbol}: {e}",
            )
            logger.error(f"🔍 GATEIO: Full error details: {type(e).__name__}: {str(e)}")
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
            since = start_time_ms

            logger.info(
                f"   🔍 Fetching historical futures data from {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}",
            )

            # Try direct HTTP requests to Gate.io's public API endpoints
            try:
                logger.info(
                    "   🌐 Attempting direct HTTP request to Gate.io public API",
                )

                # Gate.io public API endpoints for funding rate data
                endpoints = [
                    {
                        "name": "Gate.io Public Funding Rate",
                        "url": "https://api.gateio.ws/api/v4/futures/contracts/funding_rate",
                        "params": {
                            "contract": market_id,
                            "from": since // 1000,  # Convert to seconds
                            "to": end_time_ms // 1000,
                            "limit": 1000,
                        },
                    },
                ]

                for endpoint in endpoints:
                    try:
                        logger.info(f"   📡 Trying {endpoint['name']}...")

                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as session:
                            async with session.get(
                                endpoint["url"],
                                params=endpoint["params"],
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    logger.info(
                                        f"   📊 Got response from {endpoint['name']}: {len(str(data))} chars",
                                    )

                                    if data and len(data) > 0:
                                        logger.info(
                                            f"   📈 Found {len(data)} funding rate records from {endpoint['name']}",
                                        )

                                        # Convert to consistent format
                                        formatted_data = []
                                        for item in data:
                                            if isinstance(item, dict):
                                                formatted_item = {
                                                    "symbol": item.get(
                                                        "contract",
                                                        symbol,
                                                    ),
                                                    "funding_rate": item.get(
                                                        "funding_rate",
                                                        item.get("rate", 0),
                                                    ),
                                                    "funding_time": item.get(
                                                        "funding_time",
                                                        item.get("time", 0),
                                                    ),
                                                    "next_funding_time": item.get(
                                                        "next_funding_time",
                                                        0,
                                                    ),
                                                }
                                                formatted_data.append(formatted_item)

                                        logger.info(
                                            f"   ✅ Successfully collected {len(formatted_data)} funding rate records from {endpoint['name']}",
                                        )
                                        return formatted_data
                                    logger.warning(
                                        f"   ⚠️ {endpoint['name']} returned empty data",
                                    )
                                else:
                                    text = await response.text()
                                    logger.warning(
                                        f"   ⚠️ {endpoint['name']} failed with status {response.status}: {text[:200]}",
                                    )
                                    # If we get 401 (authentication required), skip futures data
                                    if response.status == 401:
                                        logger.info(
                                            "   ℹ️ Skipping futures data - authentication required but not available",
                                        )
                                        return []
                    except Exception as e:
                        logger.warning(f"   ⚠️ {endpoint['name']} failed: {e}")
                        continue

            except Exception as http_error:
                logger.warning(f"Direct HTTP API failed: {http_error}")

            # Fallback: Try to get funding rates through CCXT
            logger.info("   🔄 Falling back to CCXT for funding rates")
            try:
                # Try to get funding rate through CCXT
                funding_info = await self.exchange.fetch_funding_rate(market_id)
                if funding_info:
                    logger.info(f"   📊 Got current funding rate: {funding_info}")
                    return [funding_info]
            except Exception as e:
                logger.warning(f"   ⚠️ CCXT funding rate failed: {e}")

            logger.info(f"   ℹ️ No funding rate data available for {symbol}")
            return []

        except Exception as e:
            logger.error(
                f"Error fetching historical futures data from Gate.io for {symbol}: {e}",
            )
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
        order_type: str,
        quantity: float,
        price: float = None,
        params: dict[str, Any] = None,
    ):
        """Creates a new order."""
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
            logger.error(f"Error creating order on Gate.io for {symbol}: {e}")
            return {"error": str(e), "status": "failed"}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to get order status"},
    )
    async def get_order_status(self, symbol: str, order_id: int):
        """Retrieves the status of a specific order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.fetch_order(order_id, market_id)
        except Exception as e:
            logger.error(
                f"Failed to get status for order {order_id} on Gate.io {symbol}: {e}",
            )
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to cancel order"},
    )
    async def cancel_order(self, symbol: str, order_id: int):
        """Cancels an open order."""
        try:
            market_id = await self._get_market_id(symbol)
            return await self.exchange.cancel_order(order_id, market_id)
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} on Gate.io {symbol}: {e}")
            return {"error": str(e)}

    @retry_on_rate_limit()
    @handle_network_operations(
        max_retries=3,
        default_return={"error": "Failed to get account info"},
    )
    async def get_account_info(self):
        """Fetches account information, including balances and positions."""
        try:
            return await self.exchange.fetch_balance(params={"type": "swap"})
        except Exception as e:
            logger.error(f"Failed to get account info from Gate.io: {e}")
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
                f"Failed to get position risk from Gate.io for {symbol or 'all symbols'}: {e}",
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
                f"Failed to get open orders from Gate.io for {symbol or 'all symbols'}: {e}",
            )
            return []

    async def close(self):
        """Closes the CCXT exchange instance."""
        if self.exchange:
            await self.exchange.close()
            system_logger.info("Gate.io CCXT session closed.")

    # Implementation of abstract methods from BaseExchange

    async def _initialize_exchange(self) -> None:
        """Initialize the exchange client."""
        # GateIO doesn't need special initialization beyond CCXT setup

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
                # CCXT format: [timestamp, open, high, low, close, volume, ...]
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
        return await self.create_order(
            symbol,
            side,
            order_type,
            quantity,
            price,
            params,
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

    def _generate_signature(self, params: dict) -> str:
        """Generate signature for Gate.io API authentication."""
        import hashlib
        import hmac

        # Convert params to query string
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])

        # Create signature using HMAC-SHA512
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha512,
        ).hexdigest()

        return signature

    async def get_historical_agg_trades_ccxt(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """CCXT-specific method for historical aggregated trades (for compatibility with consolidation step)."""
        logger.info(f"🔧 GATEIO: get_historical_agg_trades_ccxt called for {symbol}")
        logger.info(
            f"📊 Parameters: start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit={limit}",
        )

        try:
            # Use the existing get_historical_agg_trades method
            result = await self.get_historical_agg_trades(
                symbol,
                start_time_ms,
                end_time_ms,
                limit,
            )
            logger.info(
                f"✅ GATEIO: get_historical_agg_trades_ccxt completed, returned {len(result)} trades",
            )
            return result
        except Exception as e:
            logger.error(f"❌ GATEIO: get_historical_agg_trades_ccxt failed: {e}")
            return []

    async def get_historical_klines_ccxt(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """CCXT-specific method for historical klines (for compatibility with consolidation step)."""
        logger.info(f"🔧 GATEIO: get_historical_klines_ccxt called for {symbol}")
        logger.info(
            f"📊 Parameters: interval={interval}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit={limit}",
        )

        try:
            # Use the existing get_historical_klines method
            result = await self.get_historical_klines(
                symbol,
                interval,
                start_time_ms,
                end_time_ms,
                limit,
            )
            logger.info(
                f"✅ GATEIO: get_historical_klines_ccxt completed, returned {len(result)} klines",
            )
            return result
        except Exception as e:
            logger.error(f"❌ GATEIO: get_historical_klines_ccxt failed: {e}")
            return []
