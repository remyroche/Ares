"""
Price Management Utilities

Handles price fetching, ticker data, and price validation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint


class PriceSource(Enum):
    """Price source enumeration"""
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    KLINE = "kline"


@dataclass
class PriceData:
    """Price data structure"""
    symbol: str
    price: float
    source: PriceSource
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_percent_24h: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PriceManager:
    """
    Manages price data fetching, caching, and validation.
    """
    
    def __init__(self, exchange_name: str):
        tprint(f"🔧 PriceManager.__init__ called with exchange_name={exchange_name}", "INFO")
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"PriceManager.{exchange_name}")
        
        # Price data cache
        self.price_cache: Dict[str, PriceData] = {}
        self.cache_ttl = timedelta(seconds=30)  # 30 seconds cache
        
        # Price fetching functions
        self.price_functions: Dict[PriceSource, callable] = {}
        
        # Price validation settings
        self.max_price_deviation = 0.1  # 10% max deviation
        self.min_price = 0.000001  # Minimum valid price
        self.max_price = 1000000.0  # Maximum valid price

        tprint(f"✅ PriceManager initialized successfully for {exchange_name}", "SUCCESS")
        
    def register_price_functions(
        self,
        get_ticker: callable,
        get_orderbook: Optional[callable] = None,
        get_recent_trades: Optional[callable] = None,
        get_klines: Optional[callable] = None
    ) -> None:
        """
        Register exchange-specific price fetching functions.

        Args:
            get_ticker: Function to get ticker data
            get_orderbook: Optional function to get order book data
            get_recent_trades: Optional function to get recent trades
            get_klines: Optional function to get kline data
        """
        tprint(f"🔧 register_price_functions called for {self.exchange_name}", "INFO")
        self.price_functions = {
            PriceSource.TICKER: get_ticker,
            PriceSource.ORDER_BOOK: get_orderbook,
            PriceSource.TRADES: get_recent_trades,
            PriceSource.KLINE: get_klines
        }

        self.logger.info("Registered price fetching functions")
        tprint(f"✅ register_price_functions: registered successfully", "SUCCESS")
    
    async def get_price(
        self,
        symbol: str,
        source: PriceSource = PriceSource.TICKER,
        use_cache: bool = True
    ) -> Optional[PriceData]:
        """
        Get price data for symbol.

        Args:
            symbol: Trading symbol
            source: Price source to use
            use_cache: Whether to use cached data

        Returns:
            PriceData if successful, None otherwise
        """
        tprint(f"🔧 get_price called with symbol={symbol}, source={source.value}, use_cache={use_cache}", "INFO")
        try:
            # Check cache first
            if use_cache and symbol in self.price_cache:
                cached_data = self.price_cache[symbol]
                if datetime.now() - cached_data.timestamp < self.cache_ttl:
                    tprint(f"✅ get_price: cache hit for {symbol}", "SUCCESS")
                    return cached_data
            
            tprint(f"⚠️ get_price: cache miss for {symbol}, fetching fresh data", "WARNING")

            # Fetch fresh data
            price_function = self.price_functions.get(source)
            if not price_function:
                self.logger.warning(f"No price function registered for {source.value}")
                tprint(f"❌ get_price: no price function registered for {source.value}", "ERROR")
                return None
            
            raw_data = await price_function(symbol)
            if not raw_data:
                self.logger.warning(f"No price data received for {symbol}")
                tprint(f"⚠️ get_price: no data received for {symbol}", "WARNING")
                return None
            
            # Parse price data
            price_data = self._parse_price_data(symbol, raw_data, source)
            if not price_data:
                tprint(f"❌ get_price: failed to parse price data for {symbol}", "ERROR")
                return None

            # Validate price
            if not self._validate_price(price_data):
                self.logger.warning(f"Invalid price data for {symbol}: {price_data.price}")
                tprint(f"❌ get_price: invalid price data for {symbol}", "ERROR")
                return None

            # Cache the data
            self.price_cache[symbol] = price_data
            tprint(f"✅ get_price: successfully fetched and cached price for {symbol}: {price_data.price}", "SUCCESS")

            return price_data

        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            tprint(f"❌ get_price failed: {e}", "ERROR")
            return None
    
    def _parse_price_data(
        self,
        symbol: str,
        raw_data: Dict[str, Any],
        source: PriceSource
    ) -> Optional[PriceData]:
        """Parse raw price data into PriceData structure."""
        tprint(f"🔧 _parse_price_data called with symbol={symbol}, source={source.value}", "INFO")
        try:
            if source == PriceSource.TICKER:
                return self._parse_ticker_data(symbol, raw_data)
            elif source == PriceSource.ORDER_BOOK:
                return self._parse_orderbook_data(symbol, raw_data)
            elif source == PriceSource.TRADES:
                return self._parse_trades_data(symbol, raw_data)
            elif source == PriceSource.KLINE:
                return self._parse_kline_data(symbol, raw_data)
            else:
                tprint(f"❌ _parse_price_data: unknown source {source.value}", "ERROR")
                return None

        except Exception as e:
            self.logger.warning(f"Error parsing price data: {e}")
            tprint(f"❌ _parse_price_data failed: {e}", "ERROR")
            return None
    
    def _parse_ticker_data(self, symbol: str, data: Dict[str, Any]) -> Optional[PriceData]:
        """Parse ticker data."""
        tprint(f"🔧 _parse_ticker_data called for symbol={symbol}", "INFO")
        try:
            # Extract price (last, close, or mark price)
            price = (
                data.get("last") or
                data.get("close") or
                data.get("mark") or
                data.get("price")
            )
            
            if not price:
                tprint(f"❌ _parse_ticker_data: no price found in ticker data", "ERROR")
                return None

            # Extract additional data
            bid = data.get("bid")
            ask = data.get("ask")
            volume_24h = data.get("volume") or data.get("vol24h")
            change_24h = data.get("change") or data.get("chg")
            change_percent_24h = data.get("changePercent") or data.get("chgRate")
            high_24h = data.get("high") or data.get("high24h")
            low_24h = data.get("low") or data.get("low24h")

            tprint(f"✅ _parse_ticker_data: parsed ticker successfully, price={float(price)}", "SUCCESS")
            return PriceData(
                symbol=symbol,
                price=float(price),
                source=PriceSource.TICKER,
                timestamp=datetime.now(),
                bid=float(bid) if bid else None,
                ask=float(ask) if ask else None,
                volume_24h=float(volume_24h) if volume_24h else None,
                change_24h=float(change_24h) if change_24h else None,
                change_percent_24h=float(change_percent_24h) if change_percent_24h else None,
                high_24h=float(high_24h) if high_24h else None,
                low_24h=float(low_24h) if low_24h else None,
                metadata=data
            )
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error parsing ticker data: {e}")
            tprint(f"❌ _parse_ticker_data failed: {e}", "ERROR")
            return None
    
    def _parse_orderbook_data(self, symbol: str, data: Dict[str, Any]) -> Optional[PriceData]:
        """Parse order book data."""
        tprint(f"🔧 _parse_orderbook_data called for symbol={symbol}", "INFO")
        try:
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                tprint(f"❌ _parse_orderbook_data: empty bids or asks", "ERROR")
                return None
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2

            tprint(f"✅ _parse_orderbook_data: parsed orderbook, mid_price={mid_price}", "SUCCESS")
            return PriceData(
                symbol=symbol,
                price=mid_price,
                source=PriceSource.ORDER_BOOK,
                timestamp=datetime.now(),
                bid=best_bid,
                ask=best_ask,
                metadata=data
            )
            
        except (ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error parsing orderbook data: {e}")
            tprint(f"❌ _parse_orderbook_data failed: {e}", "ERROR")
            return None
    
    def _parse_trades_data(self, symbol: str, data: List[Dict[str, Any]]) -> Optional[PriceData]:
        """Parse recent trades data."""
        tprint(f"🔧 _parse_trades_data called for symbol={symbol}", "INFO")
        try:
            if not data:
                tprint(f"❌ _parse_trades_data: no trade data", "ERROR")
                return None
            
            # Use the most recent trade price
            latest_trade = data[0]
            price = latest_trade.get("price") or latest_trade.get("px")

            if not price:
                tprint(f"❌ _parse_trades_data: no price in trade data", "ERROR")
                return None

            tprint(f"✅ _parse_trades_data: parsed trades, price={float(price)}", "SUCCESS")
            return PriceData(
                symbol=symbol,
                price=float(price),
                source=PriceSource.TRADES,
                timestamp=datetime.now(),
                metadata={"trades": data}
            )
            
        except (ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error parsing trades data: {e}")
            tprint(f"❌ _parse_trades_data failed: {e}", "ERROR")
            return None
    
    def _parse_kline_data(self, symbol: str, data: List[List[Any]]) -> Optional[PriceData]:
        """Parse kline data."""
        tprint(f"🔧 _parse_kline_data called for symbol={symbol}", "INFO")
        try:
            if not data:
                tprint(f"❌ _parse_kline_data: no kline data", "ERROR")
                return None
            
            # Use the most recent kline close price
            latest_kline = data[0]
            close_price = latest_kline[4]  # Close price is typically at index 4

            tprint(f"✅ _parse_kline_data: parsed kline, close_price={float(close_price)}", "SUCCESS")
            return PriceData(
                symbol=symbol,
                price=float(close_price),
                source=PriceSource.KLINE,
                timestamp=datetime.now(),
                metadata={"klines": data}
            )
            
        except (ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error parsing kline data: {e}")
            tprint(f"❌ _parse_kline_data failed: {e}", "ERROR")
            return None
    
    def _validate_price(self, price_data: PriceData) -> bool:
        """Validate price data."""
        tprint(f"🔧 _validate_price called for symbol={price_data.symbol}, price={price_data.price}", "INFO")
        try:
            price = price_data.price

            # Check price range
            if price < self.min_price or price > self.max_price:
                tprint(f"❌ _validate_price: price {price} out of range [{self.min_price}, {self.max_price}]", "ERROR")
                return False
            
            # Check for NaN or infinity
            if not (price == price) or price == float('inf') or price == float('-inf'):
                tprint(f"❌ _validate_price: invalid price (NaN or infinity)", "ERROR")
                return False
            
            # Check bid-ask spread if available
            if price_data.bid and price_data.ask:
                if price_data.bid >= price_data.ask:
                    tprint(f"❌ _validate_price: bid >= ask", "ERROR")
                    return False

                spread = (price_data.ask - price_data.bid) / price_data.bid
                if spread > self.max_price_deviation:
                    tprint(f"⚠️ _validate_price: spread {spread:.2%} exceeds max deviation", "WARNING")
                    return False

            tprint(f"✅ _validate_price: price validation passed", "SUCCESS")
            return True

        except Exception as e:
            self.logger.warning(f"Error validating price: {e}")
            tprint(f"❌ _validate_price failed: {e}", "ERROR")
            return False
    
    async def get_best_price(self, symbol: str) -> Optional[PriceData]:
        """
        Get best available price using multiple sources.

        Args:
            symbol: Trading symbol

        Returns:
            Best available PriceData
        """
        tprint(f"🔧 get_best_price called for symbol={symbol}", "INFO")
        # Try different sources in order of preference
        sources = [PriceSource.TICKER, PriceSource.ORDER_BOOK, PriceSource.TRADES, PriceSource.KLINE]
        
        for source in sources:
            try:
                price_data = await self.get_price(symbol, source, use_cache=False)
                if price_data and self._validate_price(price_data):
                    tprint(f"✅ get_best_price: found valid price from {source.value}", "SUCCESS")
                    return price_data
            except Exception as e:
                self.logger.warning(f"Error fetching price from {source.value}: {e}")
                continue

        tprint(f"❌ get_best_price: no valid price found for {symbol}", "ERROR")
        return None
    
    async def get_prices_batch(self, symbols: List[str], source: PriceSource = PriceSource.TICKER) -> Dict[str, PriceData]:
        """
        Get prices for multiple symbols in batch.

        Args:
            symbols: List of trading symbols
            source: Price source to use

        Returns:
            Dictionary mapping symbols to PriceData
        """
        tprint(f"🔧 get_prices_batch called with {len(symbols)} symbols, source={source.value}", "INFO")
        results = {}
        
        # Create tasks for concurrent fetching
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self.get_price(symbol, source))
            tasks.append((symbol, task))
        
        # Wait for all tasks to complete
        for symbol, task in tasks:
            try:
                price_data = await task
                if price_data:
                    results[symbol] = price_data
            except Exception as e:
                self.logger.warning(f"Error fetching price for {symbol}: {e}")

        tprint(f"✅ get_prices_batch: fetched prices for {len(results)}/{len(symbols)} symbols", "SUCCESS")
        return results
    
    def get_cached_price(self, symbol: str) -> Optional[PriceData]:
        """Get cached price data if available and fresh."""
        if symbol not in self.price_cache:
            return None
        
        cached_data = self.price_cache[symbol]
        if datetime.now() - cached_data.timestamp > self.cache_ttl:
            return None
        
        return cached_data
    
    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Invalidate price cache."""
        tprint(f"🔧 invalidate_cache called with symbol={symbol}", "INFO")
        if symbol:
            self.price_cache.pop(symbol, None)
            self.logger.debug(f"Invalidated cache for {symbol}")
            tprint(f"✅ invalidate_cache: cleared cache for {symbol}", "SUCCESS")
        else:
            self.price_cache.clear()
            self.logger.debug("Invalidated all price cache")
            tprint(f"✅ invalidate_cache: cleared all price cache", "SUCCESS")
    
    def get_price_statistics(self) -> Dict[str, Any]:
        """Get price cache statistics."""
        total_cached = len(self.price_cache)
        fresh_cached = len([
            data for data in self.price_cache.values()
            if datetime.now() - data.timestamp < self.cache_ttl
        ])
        
        source_counts = {}
        for data in self.price_cache.values():
            source = data.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_cached": total_cached,
            "fresh_cached": fresh_cached,
            "stale_cached": total_cached - fresh_cached,
            "source_distribution": source_counts,
            "cache_ttl_seconds": self.cache_ttl.total_seconds()
        }
    
    def cleanup_stale_cache(self) -> int:
        """Clean up stale cache entries."""
        tprint(f"🔧 cleanup_stale_cache called", "INFO")
        now = datetime.now()
        stale_symbols = [
            symbol for symbol, data in self.price_cache.items()
            if now - data.timestamp > self.cache_ttl
        ]

        for symbol in stale_symbols:
            del self.price_cache[symbol]

        if stale_symbols:
            self.logger.info(f"Cleaned up {len(stale_symbols)} stale cache entries")
            tprint(f"✅ cleanup_stale_cache: cleaned {len(stale_symbols)} entries", "SUCCESS")

        return len(stale_symbols)
    
    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache TTL in seconds."""
        tprint(f"🔧 set_cache_ttl called with ttl_seconds={ttl_seconds}", "INFO")
        self.cache_ttl = timedelta(seconds=ttl_seconds)
        self.logger.info(f"Set cache TTL to {ttl_seconds} seconds")
        tprint(f"✅ set_cache_ttl: updated TTL to {ttl_seconds} seconds", "SUCCESS")
    
    def set_price_validation(self, min_price: float, max_price: float, max_deviation: float) -> None:
        """Set price validation parameters."""
        tprint(f"🔧 set_price_validation called with min_price={min_price}, max_price={max_price}, max_deviation={max_deviation}", "INFO")
        self.min_price = min_price
        self.max_price = max_price
        self.max_price_deviation = max_deviation
        self.logger.info(f"Set price validation: min={min_price}, max={max_price}, max_deviation={max_deviation}")
        tprint(f"✅ set_price_validation: updated validation parameters", "SUCCESS")