"""
Unit tests for PriceManager.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from exchanges.shared.pricing.price_manager import (
    PriceManager, PriceData, PriceSource
)


class TestPriceManager:
    """Test cases for PriceManager."""

    @pytest.fixture
    def price_manager(self):
        """Create PriceManager instance for testing."""
        return PriceManager("test_exchange")

    @pytest.fixture
    def mock_price_functions(self):
        """Create mock price functions."""
        return {
            "get_ticker": AsyncMock(return_value={
                "symbol": "BTCUSDT",
                "last": "50000.0",
                "bid": "49999.0",
                "ask": "50001.0",
                "volume": "1000.0",
                "change": "1000.0",
                "changePercent": "2.0",
                "high": "51000.0",
                "low": "49000.0"
            }),
            "get_orderbook": AsyncMock(return_value={
                "bids": [["49999.0", "1.0"], ["49998.0", "2.0"]],
                "asks": [["50001.0", "1.0"], ["50002.0", "2.0"]]
            }),
            "get_recent_trades": AsyncMock(return_value=[
                {"price": "50000.0", "quantity": "0.001", "time": 1234567890},
                {"price": "49999.0", "quantity": "0.002", "time": 1234567891}
            ]),
            "get_klines": AsyncMock(return_value=[
                [1234567890, "49000.0", "51000.0", "48500.0", "50000.0", "100.0"],
                [1234567800, "48000.0", "50000.0", "47500.0", "49000.0", "90.0"]
            ])
        }

    def test_initialization(self, price_manager):
        """Test PriceManager initialization."""
        assert price_manager.exchange_name == "test_exchange"
        assert len(price_manager.price_cache) == 0
        assert price_manager.cache_ttl == timedelta(seconds=30)
        assert price_manager.max_price_deviation == 0.1
        assert price_manager.min_price == 0.000001
        assert price_manager.max_price == 1000000.0

    def test_register_price_functions(self, price_manager, mock_price_functions):
        """Test registering price functions."""
        price_manager.register_price_functions(**mock_price_functions)
        
        assert PriceSource.TICKER in price_manager.price_functions
        assert PriceSource.ORDER_BOOK in price_manager.price_functions
        assert PriceSource.TRADES in price_manager.price_functions
        assert PriceSource.KLINE in price_manager.price_functions

    @pytest.mark.asyncio
    async def test_get_price_ticker_success(self, price_manager, mock_price_functions):
        """Test successful price fetching from ticker."""
        price_manager.register_price_functions(**mock_price_functions)
        
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.TICKER)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0
        assert price_data.source == PriceSource.TICKER
        assert price_data.bid == 49999.0
        assert price_data.ask == 50001.0
        assert price_data.volume_24h == 1000.0
        assert price_data.change_24h == 1000.0
        assert price_data.change_percent_24h == 2.0
        assert price_data.high_24h == 51000.0
        assert price_data.low_24h == 49000.0

    @pytest.mark.asyncio
    async def test_get_price_orderbook_success(self, price_manager, mock_price_functions):
        """Test successful price fetching from order book."""
        price_manager.register_price_functions(**mock_price_functions)
        
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.ORDER_BOOK)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0  # Mid price
        assert price_data.source == PriceSource.ORDER_BOOK
        assert price_data.bid == 49999.0
        assert price_data.ask == 50001.0

    @pytest.mark.asyncio
    async def test_get_price_trades_success(self, price_manager, mock_price_functions):
        """Test successful price fetching from trades."""
        price_manager.register_price_functions(**mock_price_functions)
        
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.TRADES)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0
        assert price_data.source == PriceSource.TRADES

    @pytest.mark.asyncio
    async def test_get_price_kline_success(self, price_manager, mock_price_functions):
        """Test successful price fetching from klines."""
        price_manager.register_price_functions(**mock_price_functions)
        
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.KLINE)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0  # Close price
        assert price_data.source == PriceSource.KLINE

    @pytest.mark.asyncio
    async def test_get_price_no_function(self, price_manager):
        """Test price fetching without registered function."""
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.TICKER)
        
        assert price_data is None

    @pytest.mark.asyncio
    async def test_get_price_no_data(self, price_manager):
        """Test price fetching with no data returned."""
        mock_functions = {
            "get_ticker": AsyncMock(return_value=None),
            "get_orderbook": AsyncMock(),
            "get_recent_trades": AsyncMock(),
            "get_klines": AsyncMock()
        }
        price_manager.register_price_functions(**mock_functions)
        
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.TICKER)
        
        assert price_data is None

    @pytest.mark.asyncio
    async def test_get_price_use_cache(self, price_manager, mock_price_functions):
        """Test price fetching with cache."""
        price_manager.register_price_functions(**mock_price_functions)
        
        # First call - should fetch from exchange
        price_data1 = await price_manager.get_price("BTCUSDT", PriceSource.TICKER, use_cache=True)
        
        # Second call - should use cache
        price_data2 = await price_manager.get_price("BTCUSDT", PriceSource.TICKER, use_cache=True)
        
        assert price_data1 == price_data2
        assert "BTCUSDT" in price_manager.price_cache

    @pytest.mark.asyncio
    async def test_get_price_cache_expired(self, price_manager, mock_price_functions):
        """Test price fetching with expired cache."""
        price_manager.register_price_functions(**mock_price_functions)
        price_manager.cache_ttl = timedelta(seconds=0)  # Immediate expiration
        
        # First call
        await price_manager.get_price("BTCUSDT", PriceSource.TICKER, use_cache=True)
        
        # Second call - cache should be expired
        price_data = await price_manager.get_price("BTCUSDT", PriceSource.TICKER, use_cache=True)
        
        assert price_data is not None

    def test_parse_ticker_data(self, price_manager):
        """Test parsing ticker data."""
        ticker_data = {
            "symbol": "BTCUSDT",
            "last": "50000.0",
            "bid": "49999.0",
            "ask": "50001.0",
            "volume": "1000.0",
            "change": "1000.0",
            "changePercent": "2.0",
            "high": "51000.0",
            "low": "49000.0"
        }
        
        price_data = price_manager._parse_ticker_data("BTCUSDT", ticker_data)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0
        assert price_data.bid == 49999.0
        assert price_data.ask == 50001.0

    def test_parse_ticker_data_alternative_fields(self, price_manager):
        """Test parsing ticker data with alternative field names."""
        ticker_data = {
            "symbol": "BTCUSDT",
            "close": "50000.0",  # Alternative to "last"
            "bid": "49999.0",
            "ask": "50001.0",
            "vol24h": "1000.0",  # Alternative to "volume"
            "chg": "1000.0",     # Alternative to "change"
            "chgRate": "2.0",    # Alternative to "changePercent"
            "high24h": "51000.0", # Alternative to "high"
            "low24h": "49000.0"   # Alternative to "low"
        }
        
        price_data = price_manager._parse_ticker_data("BTCUSDT", ticker_data)
        
        assert price_data is not None
        assert price_data.price == 50000.0
        assert price_data.volume_24h == 1000.0
        assert price_data.change_24h == 1000.0
        assert price_data.change_percent_24h == 2.0

    def test_parse_ticker_data_no_price(self, price_manager):
        """Test parsing ticker data with no price."""
        ticker_data = {
            "symbol": "BTCUSDT",
            "bid": "49999.0",
            "ask": "50001.0"
        }
        
        price_data = price_manager._parse_ticker_data("BTCUSDT", ticker_data)
        
        assert price_data is None

    def test_parse_orderbook_data(self, price_manager):
        """Test parsing order book data."""
        orderbook_data = {
            "bids": [["49999.0", "1.0"], ["49998.0", "2.0"]],
            "asks": [["50001.0", "1.0"], ["50002.0", "2.0"]]
        }
        
        price_data = price_manager._parse_orderbook_data("BTCUSDT", orderbook_data)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0  # Mid price
        assert price_data.bid == 49999.0
        assert price_data.ask == 50001.0

    def test_parse_orderbook_data_empty(self, price_manager):
        """Test parsing empty order book data."""
        orderbook_data = {
            "bids": [],
            "asks": []
        }
        
        price_data = price_manager._parse_orderbook_data("BTCUSDT", orderbook_data)
        
        assert price_data is None

    def test_parse_trades_data(self, price_manager):
        """Test parsing trades data."""
        trades_data = [
            {"price": "50000.0", "quantity": "0.001", "time": 1234567890},
            {"price": "49999.0", "quantity": "0.002", "time": 1234567891}
        ]
        
        price_data = price_manager._parse_trades_data("BTCUSDT", trades_data)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0  # Latest trade price

    def test_parse_trades_data_empty(self, price_manager):
        """Test parsing empty trades data."""
        trades_data = []
        
        price_data = price_manager._parse_trades_data("BTCUSDT", trades_data)
        
        assert price_data is None

    def test_parse_kline_data(self, price_manager):
        """Test parsing kline data."""
        kline_data = [
            [1234567890, "49000.0", "51000.0", "48500.0", "50000.0", "100.0"],
            [1234567800, "48000.0", "50000.0", "47500.0", "49000.0", "90.0"]
        ]
        
        price_data = price_manager._parse_kline_data("BTCUSDT", kline_data)
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0  # Close price of latest kline

    def test_parse_kline_data_empty(self, price_manager):
        """Test parsing empty kline data."""
        kline_data = []
        
        price_data = price_manager._parse_kline_data("BTCUSDT", kline_data)
        
        assert price_data is None

    def test_validate_price_valid(self, price_manager):
        """Test validating valid price."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now(),
            bid=49999.0,
            ask=50001.0
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is True

    def test_validate_price_too_low(self, price_manager):
        """Test validating price that's too low."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=0.0000001,  # Below min_price
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is False

    def test_validate_price_too_high(self, price_manager):
        """Test validating price that's too high."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=2000000.0,  # Above max_price
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is False

    def test_validate_price_nan(self, price_manager):
        """Test validating NaN price."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=float('nan'),
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is False

    def test_validate_price_infinity(self, price_manager):
        """Test validating infinity price."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=float('inf'),
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is False

    def test_validate_price_bad_spread(self, price_manager):
        """Test validating price with bad bid-ask spread."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now(),
            bid=50001.0,  # Bid higher than ask
            ask=50000.0
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is False

    def test_validate_price_large_spread(self, price_manager):
        """Test validating price with large spread."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now(),
            bid=40000.0,  # Very large spread
            ask=60000.0
        )
        
        is_valid = price_manager._validate_price(price_data)
        
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_best_price(self, price_manager, mock_price_functions):
        """Test getting best available price."""
        price_manager.register_price_functions(**mock_price_functions)
        
        price_data = await price_manager.get_best_price("BTCUSDT")
        
        assert price_data is not None
        assert price_data.symbol == "BTCUSDT"
        assert price_data.price == 50000.0

    @pytest.mark.asyncio
    async def test_get_best_price_no_valid_sources(self, price_manager):
        """Test getting best price when no sources are valid."""
        mock_functions = {
            "get_ticker": AsyncMock(return_value=None),
            "get_orderbook": AsyncMock(return_value=None),
            "get_recent_trades": AsyncMock(return_value=None),
            "get_klines": AsyncMock(return_value=None)
        }
        price_manager.register_price_functions(**mock_functions)
        
        price_data = await price_manager.get_best_price("BTCUSDT")
        
        assert price_data is None

    @pytest.mark.asyncio
    async def test_get_prices_batch(self, price_manager, mock_price_functions):
        """Test getting prices for multiple symbols."""
        price_manager.register_price_functions(**mock_price_functions)
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        results = await price_manager.get_prices_batch(symbols, PriceSource.TICKER)
        
        assert len(results) == 2
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert results["BTCUSDT"].symbol == "BTCUSDT"
        assert results["ETHUSDT"].symbol == "ETHUSDT"

    def test_get_cached_price_fresh(self, price_manager):
        """Test getting fresh cached price."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        price_manager.price_cache["BTCUSDT"] = price_data
        
        cached = price_manager.get_cached_price("BTCUSDT")
        
        assert cached == price_data

    def test_get_cached_price_stale(self, price_manager):
        """Test getting stale cached price."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now() - timedelta(minutes=1)
        )
        price_manager.price_cache["BTCUSDT"] = price_data
        
        cached = price_manager.get_cached_price("BTCUSDT")
        
        assert cached is None

    def test_get_cached_price_not_found(self, price_manager):
        """Test getting non-existent cached price."""
        cached = price_manager.get_cached_price("NONEXISTENT")
        
        assert cached is None

    def test_invalidate_cache_specific(self, price_manager):
        """Test invalidating specific symbol cache."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        price_manager.price_cache["BTCUSDT"] = price_data
        price_manager.price_cache["ETHUSDT"] = price_data
        
        price_manager.invalidate_cache("BTCUSDT")
        
        assert "BTCUSDT" not in price_manager.price_cache
        assert "ETHUSDT" in price_manager.price_cache

    def test_invalidate_cache_all(self, price_manager):
        """Test invalidating all cache."""
        price_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        price_manager.price_cache["BTCUSDT"] = price_data
        price_manager.price_cache["ETHUSDT"] = price_data
        
        price_manager.invalidate_cache()
        
        assert len(price_manager.price_cache) == 0

    def test_get_price_statistics(self, price_manager):
        """Test getting price statistics."""
        # Add some cached data
        fresh_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        stale_data = PriceData(
            symbol="ETHUSDT",
            price=3000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now() - timedelta(minutes=1)
        )
        
        price_manager.price_cache["BTCUSDT"] = fresh_data
        price_manager.price_cache["ETHUSDT"] = stale_data
        
        stats = price_manager.get_price_statistics()
        
        assert stats["total_cached"] == 2
        assert stats["fresh_cached"] == 1
        assert stats["stale_cached"] == 1
        assert "source_distribution" in stats
        assert stats["cache_ttl_seconds"] == 30

    def test_cleanup_stale_cache(self, price_manager):
        """Test cleaning up stale cache entries."""
        fresh_data = PriceData(
            symbol="BTCUSDT",
            price=50000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now()
        )
        stale_data = PriceData(
            symbol="ETHUSDT",
            price=3000.0,
            source=PriceSource.TICKER,
            timestamp=datetime.now() - timedelta(minutes=1)
        )
        
        price_manager.price_cache["BTCUSDT"] = fresh_data
        price_manager.price_cache["ETHUSDT"] = stale_data
        
        cleaned = price_manager.cleanup_stale_cache()
        
        assert cleaned == 1
        assert "BTCUSDT" in price_manager.price_cache
        assert "ETHUSDT" not in price_manager.price_cache

    def test_set_cache_ttl(self, price_manager):
        """Test setting cache TTL."""
        price_manager.set_cache_ttl(60)
        
        assert price_manager.cache_ttl == timedelta(seconds=60)

    def test_set_price_validation(self, price_manager):
        """Test setting price validation parameters."""
        price_manager.set_price_validation(
            min_price=0.001,
            max_price=100000.0,
            max_deviation=0.05
        )
        
        assert price_manager.min_price == 0.001
        assert price_manager.max_price == 100000.0
        assert price_manager.max_price_deviation == 0.05