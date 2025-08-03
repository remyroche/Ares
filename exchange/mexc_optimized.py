#!/usr/bin/env python3
"""
Optimized MEXC Exchange implementation with concurrent requests and better performance.
"""

import asyncio
import aiohttp
import ccxt.async_support as ccxt
from datetime import datetime
from typing import Any, Optional, List
from functools import wraps
import time

from src.utils.logger import system_logger
from src.utils.error_handler import handle_network_operations
from src.interfaces.base_interfaces import MarketData
from exchange.base_exchange import BaseExchange

logger = system_logger.getChild("MexcExchangeOptimized")


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


class MexcExchangeOptimized(BaseExchange):
    """Optimized MEXC Exchange implementation with concurrent requests."""

    def __init__(self, api_key: str, api_secret: str, trade_symbol: str):
        super().__init__(api_key, api_secret, trade_symbol)
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        self.session = None
        self.semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async def _get_market_id(self, symbol: str) -> str:
        """Get market ID for symbol."""
        return symbol

    async def _make_request(self, url: str, params: dict) -> List[dict]:
        """Make a single HTTP request with connection pooling."""
        async with self.semaphore:
            if not self.session:
                connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30),
                    connector=connector
                )
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data if isinstance(data, list) else []
                else:
                    text = await response.text()
                    print(f"🔍 DEBUG: HTTP error {response.status}: {text[:200]}")
                    return []

    async def _fetch_hour_data(self, symbol: str, start_time: int, end_time: int) -> List[dict]:
        """Fetch all data for a single hour with optimized pagination."""
        url = "https://api.mexc.com/api/v3/aggTrades"
        all_trades = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000
            }
            
            try:
                data = await self._make_request(url, params)
                
                if not data:
                    break
                
                # Convert to Binance-compatible format
                for trade in data:
                    if isinstance(trade, dict):
                        formatted_trade = {
                            "a": trade.get("a", trade.get("id", 0)),
                            "p": trade.get("p", trade.get("price", 0)),
                            "q": trade.get("q", trade.get("quantity", 0)),
                            "T": trade.get("T", trade.get("time", 0)),
                            "m": trade.get("m", trade.get("isBuyerMaker", False)),
                            "f": trade.get("f", 0),
                            "l": trade.get("l", 0)
                        }
                        all_trades.append(formatted_trade)
                
                # If we got exactly 1000 trades, there might be more
                if len(data) == 1000:
                    last_trade_time = data[-1].get('T', 0)
                    current_start = last_trade_time + 1
                    await asyncio.sleep(0.05)  # Reduced rate limiting
                else:
                    break
                    
            except Exception as e:
                print(f"🔍 DEBUG: Error fetching hour data: {e}")
                break
        
        return all_trades

    async def _fetch_hours_concurrent(self, symbol: str, hour_ranges: List[tuple]) -> List[dict]:
        """Fetch multiple hours concurrently."""
        print(f"🔍 DEBUG: Fetching {len(hour_ranges)} hours concurrently")
        
        tasks = []
        for start_time, end_time in hour_ranges:
            task = self._fetch_hour_data(symbol, start_time, end_time)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_trades = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"🔍 DEBUG: Hour {i+1} failed: {result}")
            else:
                all_trades.extend(result)
                print(f"🔍 DEBUG: Hour {i+1} completed: {len(result)} trades")
        
        return all_trades

    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades with optimized concurrent requests."""
        try:
            market_id = await self._get_market_id(symbol)
            since = start_time_ms
            
            print(f"🔍 DEBUG: Optimized MEXC _get_historical_agg_trades_raw called")
            print(f"🔍 DEBUG: Time range: {datetime.fromtimestamp(since / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}")
            
            # Calculate hour chunks for concurrent processing
            hour_ms = 60 * 60 * 1000
            hour_ranges = []
            current_start = since
            
            while current_start < end_time_ms:
                current_end = min(current_start + hour_ms, end_time_ms)
                hour_ranges.append((current_start, current_end))
                current_start = current_end
            
            print(f"🔍 DEBUG: Processing {len(hour_ranges)} hours concurrently")
            
            # Fetch all hours concurrently
            all_trades = await self._fetch_hours_concurrent(symbol, hour_ranges)
            
            print(f"🔍 DEBUG: Successfully collected {len(all_trades)} aggregated trades")
            return all_trades
            
        except Exception as e:
            print(f"🔍 DEBUG: Error in optimized _get_historical_agg_trades_raw: {e}")
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
        print(f"🔍 DEBUG: Optimized MEXC get_historical_agg_trades called")
        print(f"🔍 DEBUG: Parameters: symbol={symbol}, start_time_ms={start_time_ms}, end_time_ms={end_time_ms}, limit={limit}")
        
        start_time = time.time()
        result = await self._get_historical_agg_trades_raw(symbol, start_time_ms, end_time_ms, limit)
        end_time = time.time()
        
        print(f"🔍 DEBUG: Optimized MEXC get_historical_agg_trades completed in {end_time - start_time:.2f} seconds")
        print(f"🔍 DEBUG: Returning {len(result)} trades")
        return result

    async def close(self):
        """Close the exchange connection."""
        if self.session:
            await self.session.close()
        if self.exchange:
            await self.exchange.close()

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