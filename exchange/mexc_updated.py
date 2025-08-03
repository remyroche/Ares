#!/usr/bin/env python3
"""
Updated MEXC Exchange implementation with proper pagination and detailed logging.
"""

import asyncio
import aiohttp
import ccxt.async_support as ccxt
from datetime import datetime
from typing import Any, Optional
from functools import wraps

from src.utils.logger import system_logger
from src.utils.error_handler import handle_network_operations
from src.interfaces.base_interfaces import MarketData
from exchange.base_exchange import BaseExchange

logger = system_logger.getChild("MexcExchange")


def retry_on_rate_limit(max_retries=5, initial_backoff=1.0):
    """Retry decorator with exponential backoff for rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        wait_time = initial_backoff * (2 ** attempt)
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
            raise last_exception
        return wrapper
    return decorator


class MexcExchangeUpdated(BaseExchange):
    """Updated MEXC Exchange implementation with proper pagination."""

    def __init__(self, api_key: str, api_secret: str, trade_symbol: str):
        super().__init__(api_key, api_secret, trade_symbol)
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })

    async def _get_market_id(self, symbol: str) -> str:
        """Get market ID for symbol."""
        return symbol

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from exchange with proper pagination."""
        # Call the actual implementation directly to avoid recursion
        try:
            market_id = await self._get_market_id(symbol)
            since = start_time_ms
            all_trades = []
            
            print(f"🔍 DEBUG: MEXC _get_historical_agg_trades_raw called")
            print(f"🔍 DEBUG: Market ID: {market_id}")
            print(f"🔍 DEBUG: Time range: {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}")
            
            logger.info(f"   🔍 Fetching historical trades from {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}")
            
            # Try direct HTTP requests to MEXC's official aggTrades API endpoint
            try:
                logger.info(f"   🌐 Attempting direct HTTP request to MEXC aggTrades API")
                
                # MEXC has a 1-hour limit, so we need to paginate in 1-hour chunks
                current_start = since
                hour_ms = 60 * 60 * 1000  # 1 hour in milliseconds
                total_hours = (end_time_ms - since) // hour_ms + 1
                current_hour = 1
                
                print(f"🔍 DEBUG: Starting MEXC pagination")
                print(f"🔍 DEBUG: Total time range: {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}")
                print(f"🔍 DEBUG: Total hours to process: {total_hours}")
                print(f"🔍 DEBUG: Processing in 1-hour chunks")
                
                logger.info(f"   📅 Total hours to process: {total_hours}")
                logger.info(f"   🔧 Network timeout: 10 seconds")
                logger.info(f"   🔧 Rate limiting: 0.1 seconds between requests")
                
                while current_start < end_time_ms:
                    current_end = min(current_start + hour_ms, end_time_ms)
                    
                    # MEXC official aggTrades API endpoint
                    url = "https://api.mexc.com/api/v3/aggTrades"
                    params = {
                        'symbol': symbol,
                        'startTime': current_start,
                        'endTime': current_end,
                        'limit': 1000
                    }
                    
                    print(f"🔍 DEBUG: [{current_hour}/{total_hours}] Processing hour: {datetime.fromtimestamp(current_start / 1000)} to {datetime.fromtimestamp(current_end / 1000)}")
                    logger.info(f"   📡 [{current_hour}/{total_hours}] Fetching trades for {datetime.fromtimestamp(current_start / 1000)} to {datetime.fromtimestamp(current_end / 1000)}")
                    logger.info(f"   🔗 URL: {url}")
                    logger.info(f"   📋 Params: {params}")
                    
                    hour_trades = []
                    page = 1
                    total_trades_in_hour = 0
                    
                    # Paginate within this hour
                    while True:
                        print(f"🔍 DEBUG:   Page {page} for hour {current_hour}")
                        
                        try:
                            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                                print(f"🔍 DEBUG:     Creating HTTP session...")
                                logger.info(f"   🔄 Creating HTTP session...")
                                
                                async with session.get(url, params=params) as response:
                                    print(f"🔍 DEBUG:     HTTP Response Status: {response.status}")
                                    logger.info(f"   📡 HTTP Response Status: {response.status}")
                                    
                                    if response.status == 200:
                                        print(f"🔍 DEBUG:     HTTP request successful")
                                        logger.info(f"   ✅ HTTP request successful")
                                        
                                        data = await response.json()
                                        print(f"🔍 DEBUG:     Got {len(data)} trades on page {page}")
                                        logger.info(f"   📊 Got response from MEXC aggTrades API: {len(str(data))} chars")
                                        logger.info(f"   📋 Response type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")
                                        
                                        if data and len(data) > 0:
                                            print(f"🔍 DEBUG:     Processing {len(data)} trades")
                                            logger.info(f"   📈 Found {len(data)} aggregated trades for page {page}")
                                            
                                            # Convert to Binance-compatible format
                                            for trade in data:
                                                if isinstance(trade, dict):
                                                    # MEXC uses the same format as Binance: a, p, q, T, m, f, l
                                                    formatted_trade = {
                                                        "a": trade.get("a", trade.get("id", 0)),  # aggregated trade ID
                                                        "p": trade.get("p", trade.get("price", 0)),  # price
                                                        "q": trade.get("q", trade.get("quantity", 0)),  # quantity
                                                        "T": trade.get("T", trade.get("time", 0)),  # timestamp
                                                        "m": trade.get("m", trade.get("isBuyerMaker", False)),  # is buyer maker
                                                        "f": trade.get("f", 0),  # first trade ID
                                                        "l": trade.get("l", 0)  # last trade ID
                                                    }
                                                    hour_trades.append(formatted_trade)
                                            
                                            total_trades_in_hour += len(data)
                                            print(f"🔍 DEBUG:     Total trades in hour so far: {total_trades_in_hour}")
                                            
                                            # Check if we need to paginate (if we got exactly 1000 trades, there might be more)
                                            if len(data) == 1000:
                                                print(f"🔍 DEBUG:     Got exactly 1000 trades, checking for more data...")
                                                # Update start time to the last trade timestamp + 1ms
                                                last_trade_time = data[-1].get('T', 0)
                                                params['startTime'] = last_trade_time + 1
                                                print(f"🔍 DEBUG:     Next page start time: {datetime.fromtimestamp((last_trade_time + 1) / 1000)}")
                                                page += 1
                                                await asyncio.sleep(0.1)  # Rate limiting between pages
                                                continue
                                            else:
                                                print(f"🔍 DEBUG:     Got {len(data)} trades (less than 1000), no more pages for this hour")
                                                break
                                        else:
                                            print(f"🔍 DEBUG:     No trades returned for this page")
                                            logger.info(f"   ⚠️ No trades found for page {page}")
                                            break
                                    else:
                                        text = await response.text()
                                        print(f"🔍 DEBUG:     HTTP error: {response.status} - {text[:200]}")
                                        logger.warning(f"   ⚠️ MEXC aggTrades API failed with status {response.status}")
                                        logger.warning(f"   📋 Error response: {text[:500]}")
                                        break
                        except Exception as http_error:
                            print(f"🔍 DEBUG:     Exception: {type(http_error).__name__}: {http_error}")
                            logger.error(f"   ❌ HTTP request failed for hour {current_hour}, page {page}: {http_error}")
                            logger.error(f"   🔍 Exception type: {type(http_error).__name__}")
                            logger.error(f"   📋 Exception details: {str(http_error)}")
                            break
                    
                    # Add all trades from this hour to the main collection
                    all_trades.extend(hour_trades)
                    print(f"🔍 DEBUG:   Hour {current_hour} completed: {len(hour_trades)} trades")
                    print(f"🔍 DEBUG:   Total trades collected so far: {len(all_trades)}")
                    logger.info(f"   ✅ Hour {current_hour} completed: {len(hour_trades)} trades")
                    
                    # Move to next hour
                    current_start = current_end
                    current_hour += 1
                    print(f"🔍 DEBUG:   Moving to next hour...")
                    logger.info(f"   ⏳ Waiting 0.1 seconds before next request...")
                    await asyncio.sleep(0.1)  # Rate limiting
                
                if all_trades:
                    print(f"🔍 DEBUG: Successfully collected {len(all_trades)} aggregated trades from MEXC API")
                    logger.info(f"   ✅ Successfully collected {len(all_trades)} aggregated trades from MEXC API")
                    return all_trades
                else:
                    print(f"🔍 DEBUG: No trades collected from MEXC API")
                    logger.warning(f"   ⚠️ No trades collected from MEXC API")
                    return []
                            
            except Exception as http_error:
                print(f"🔍 DEBUG: Direct HTTP API failed: {http_error}")
                logger.warning(f"Direct HTTP API failed: {http_error}")
                return []
            
        except Exception as e:
            print(f"🔍 DEBUG: Error in _get_historical_agg_trades_raw: {e}")
            logger.error(f"Error fetching historical trades from MEXC for {symbol}: {e}")
            return []

    async def get_historical_agg_trades(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get historical aggregated trades for a symbol within a time range."""
        print(f"🔍 DEBUG: MEXC get_historical_agg_trades called")
        print(f"🔍 DEBUG: Parameters: symbol={symbol}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit={limit}")
        # Call the raw method directly to avoid recursion
        result = await self._get_historical_agg_trades_raw(symbol, start_time_ms, end_time_ms, limit)
        print(f"🔍 DEBUG: MEXC get_historical_agg_trades returning {len(result)} trades")
        return result

    # Add other required methods from BaseExchange
    async def _initialize_exchange(self) -> None:
        """Initialize the exchange client."""
        await self.exchange.load_markets()

    async def _convert_to_market_data(self, raw_data: list[dict[str, Any]], symbol: str, interval: str) -> list[MarketData]:
        """Convert raw exchange data to standardized MarketData format."""
        market_data_list = []
        for candle in raw_data:
            market_data = MarketData(
                symbol=symbol,
                timestamp=self._convert_timestamp(candle[0]),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
                interval=interval
            )
            market_data_list.append(market_data)
        return market_data_list

    async def _get_klines_raw(self, symbol: str, interval: str, limit: int) -> list[dict[str, Any]]:
        """Get raw kline data from exchange."""
        return await self.get_klines_raw(symbol, interval, limit)

    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account info from exchange."""
        return await self.exchange.fetch_balance()

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
        return await self.exchange.create_order(symbol, side, order_type, quantity, price, params or {})

    async def _get_position_risk_raw(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Get raw position risk from exchange."""
        return await self.exchange.fetch_positions([symbol] if symbol else None)

    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical klines from exchange."""
        return await self.get_historical_klines(symbol, interval, start_time_ms, end_time_ms, limit)

    async def _get_open_orders_raw(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """Get raw open orders from exchange."""
        return await self.exchange.fetch_open_orders(symbol)

    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on exchange."""
        return await self.exchange.cancel_order(order_id, symbol)

    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from exchange."""
        return await self.exchange.fetch_order(order_id, symbol)

    async def close(self):
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close() 