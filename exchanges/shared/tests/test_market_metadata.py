"""
Unit tests for MarketMetadataManager.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from exchanges.shared.market.market_metadata import (
    MarketMetadataManager, InstrumentSpec, InstrumentType, OrderType
)


class TestMarketMetadataManager:
    """Test cases for MarketMetadataManager."""

    @pytest.fixture
    def market_manager(self):
        """Create MarketMetadataManager instance for testing."""
        return MarketMetadataManager("test_exchange")

    @pytest.fixture
    def sample_instrument_data(self):
        """Sample instrument data for testing."""
        return {
            "symbol": "BTCUSDT",
            "type": "spot",
            "baseCcy": "BTC",
            "quoteCcy": "USDT",
            "state": "live",
            "tickSz": "0.01",
            "lotSz": "0.00001",
            "minSz": "10",
            "maxSz": "1000000",
            "lever": "10",
            "ctVal": "0.01",
            "settleCcy": "USDT"
        }

    @pytest.fixture
    def mock_refresh_functions(self):
        """Create mock refresh functions."""
        return {
            "get_instruments": AsyncMock(return_value=[
                {
                    "symbol": "BTCUSDT",
                    "type": "spot",
                    "baseCcy": "BTC",
                    "quoteCcy": "USDT",
                    "state": "live",
                    "tickSz": "0.01",
                    "lotSz": "0.00001",
                    "minSz": "10"
                }
            ]),
            "get_ticker": AsyncMock(return_value={
                "symbol": "BTCUSDT",
                "last": "50000.0",
                "bid": "49999.0",
                "ask": "50001.0",
                "volume": "1000.0"
            }),
            "get_orderbook": AsyncMock(return_value={
                "bids": [["49999.0", "1.0"]],
                "asks": [["50001.0", "1.0"]]
            }),
            "get_funding_rate": AsyncMock(return_value={
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001"
            })
        }

    def test_initialization(self, market_manager):
        """Test MarketMetadataManager initialization."""
        assert market_manager.exchange_name == "test_exchange"
        assert len(market_manager.instruments) == 0
        assert len(market_manager.market_data) == 0
        assert market_manager.cache_ttl == timedelta(minutes=5)
        assert market_manager.last_refresh is None

    def test_register_refresh_functions(self, market_manager, mock_refresh_functions):
        """Test registering refresh functions."""
        market_manager.register_refresh_functions(**mock_refresh_functions)
        
        assert "get_instruments" in market_manager.refresh_functions
        assert "get_ticker" in market_manager.refresh_functions
        assert "get_orderbook" in market_manager.refresh_functions
        assert "get_funding_rate" in market_manager.refresh_functions

    @pytest.mark.asyncio
    async def test_refresh_instruments_success(self, market_manager, mock_refresh_functions):
        """Test successful instrument refresh."""
        market_manager.register_refresh_functions(**mock_refresh_functions)
        
        result = await market_manager.refresh_instruments()
        
        assert result is True
        assert len(market_manager.instruments) == 1
        assert "BTCUSDT" in market_manager.instruments
        assert market_manager.last_refresh is not None

    @pytest.mark.asyncio
    async def test_refresh_instruments_no_function(self, market_manager):
        """Test instrument refresh without registered function."""
        result = await market_manager.refresh_instruments()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_instruments_no_data(self, market_manager):
        """Test instrument refresh with no data."""
        mock_functions = {
            "get_instruments": AsyncMock(return_value=None),
            "get_ticker": AsyncMock(),
            "get_orderbook": AsyncMock(),
            "get_funding_rate": AsyncMock()
        }
        market_manager.register_refresh_functions(**mock_functions)
        
        result = await market_manager.refresh_instruments()
        
        assert result is False

    def test_parse_instrument_data_spot(self, market_manager, sample_instrument_data):
        """Test parsing spot instrument data."""
        spec = market_manager._parse_instrument_data(sample_instrument_data)
        
        assert spec is not None
        assert spec.symbol == "BTCUSDT"
        assert spec.base_currency == "BTC"
        assert spec.quote_currency == "USDT"
        assert spec.instrument_type == InstrumentType.SPOT
        assert spec.tick_size == 0.01
        assert spec.lot_size == 0.00001
        assert spec.min_notional == 10.0

    def test_parse_instrument_data_futures(self, market_manager):
        """Test parsing futures instrument data."""
        data = {
            "symbol": "BTCUSDT-SWAP",
            "type": "futures",
            "baseCcy": "BTC",
            "quoteCcy": "USDT",
            "state": "live",
            "tickSz": "0.1",
            "lotSz": "0.001",
            "minSz": "100",
            "lever": "20"
        }
        
        spec = market_manager._parse_instrument_data(data)
        
        assert spec is not None
        assert spec.instrument_type == InstrumentType.FUTURES
        assert spec.max_leverage == 20.0

    def test_parse_instrument_data_perpetual(self, market_manager):
        """Test parsing perpetual instrument data."""
        data = {
            "symbol": "BTCUSDT-PERP",
            "type": "perpetual",
            "baseCcy": "BTC",
            "quoteCcy": "USDT",
            "state": "live",
            "tickSz": "0.1",
            "lotSz": "0.001",
            "minSz": "100"
        }
        
        spec = market_manager._parse_instrument_data(data)
        
        assert spec is not None
        assert spec.instrument_type == InstrumentType.PERPETUAL

    def test_parse_instrument_data_invalid(self, market_manager):
        """Test parsing invalid instrument data."""
        data = {"invalid": "data"}
        
        spec = market_manager._parse_instrument_data(data)
        
        assert spec is None

    @pytest.mark.asyncio
    async def test_refresh_market_data_success(self, market_manager, mock_refresh_functions):
        """Test successful market data refresh."""
        # First add some instruments
        market_manager.instruments["BTCUSDT"] = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        
        market_manager.register_refresh_functions(**mock_refresh_functions)
        
        result = await market_manager.refresh_market_data(["BTCUSDT"])
        
        assert result is True
        assert "BTCUSDT" in market_manager.market_data
        assert "ticker" in market_manager.market_data["BTCUSDT"]

    @pytest.mark.asyncio
    async def test_refresh_market_data_no_function(self, market_manager):
        """Test market data refresh without registered function."""
        result = await market_manager.refresh_market_data(["BTCUSDT"])
        
        assert result is False

    def test_get_instrument_direct(self, market_manager):
        """Test getting instrument by direct symbol match."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        market_manager.instruments["BTCUSDT"] = spec
        
        result = market_manager.get_instrument("BTCUSDT")
        
        assert result == spec

    def test_get_instrument_case_insensitive(self, market_manager):
        """Test getting instrument with case-insensitive lookup."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        market_manager.instruments["BTCUSDT"] = spec
        market_manager.symbol_cache["BTCUSDT"] = "BTCUSDT"
        
        result = market_manager.get_instrument("btcusdt")
        
        assert result == spec

    def test_get_instrument_not_found(self, market_manager):
        """Test getting non-existent instrument."""
        result = market_manager.get_instrument("NONEXISTENT")
        
        assert result is None

    def test_get_instruments_by_type(self, market_manager):
        """Test getting instruments by type."""
        spot_spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        futures_spec = InstrumentSpec(
            symbol="BTCUSDT-SWAP",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.FUTURES,
            status="active",
            tick_size=0.1,
            lot_size=0.001,
            min_notional=100.0,
            price_precision=1,
            quantity_precision=3
        )
        
        market_manager.instruments["BTCUSDT"] = spot_spec
        market_manager.instruments["BTCUSDT-SWAP"] = futures_spec
        
        spot_instruments = market_manager.get_instruments_by_type(InstrumentType.SPOT)
        futures_instruments = market_manager.get_instruments_by_type(InstrumentType.FUTURES)
        
        assert len(spot_instruments) == 1
        assert len(futures_instruments) == 1
        assert spot_instruments[0].symbol == "BTCUSDT"
        assert futures_instruments[0].symbol == "BTCUSDT-SWAP"

    def test_get_active_instruments(self, market_manager):
        """Test getting active instruments."""
        active_spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        inactive_spec = InstrumentSpec(
            symbol="ETHUSDT",
            base_currency="ETH",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="suspended",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=False
        )
        
        market_manager.instruments["BTCUSDT"] = active_spec
        market_manager.instruments["ETHUSDT"] = inactive_spec
        
        active_instruments = market_manager.get_active_instruments()
        
        assert len(active_instruments) == 1
        assert active_instruments[0].symbol == "BTCUSDT"

    def test_get_trading_pairs(self, market_manager):
        """Test getting trading pairs for a base currency."""
        btc_spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        eth_spec = InstrumentSpec(
            symbol="ETHUSDT",
            base_currency="ETH",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        
        market_manager.instruments["BTCUSDT"] = btc_spec
        market_manager.instruments["ETHUSDT"] = eth_spec
        
        btc_pairs = market_manager.get_trading_pairs("BTC")
        eth_pairs = market_manager.get_trading_pairs("ETH")
        
        assert len(btc_pairs) == 1
        assert len(eth_pairs) == 1
        assert btc_pairs[0].symbol == "BTCUSDT"
        assert eth_pairs[0].symbol == "ETHUSDT"

    def test_get_quote_currencies(self, market_manager):
        """Test getting quote currencies."""
        spec1 = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        spec2 = InstrumentSpec(
            symbol="BTCUSD",
            base_currency="BTC",
            quote_currency="USD",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        
        market_manager.instruments["BTCUSDT"] = spec1
        market_manager.instruments["BTCUSD"] = spec2
        
        quote_currencies = market_manager.get_quote_currencies()
        
        assert "USDT" in quote_currencies
        assert "USD" in quote_currencies
        assert len(quote_currencies) == 2

    def test_get_base_currencies(self, market_manager):
        """Test getting base currencies."""
        spec1 = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        spec2 = InstrumentSpec(
            symbol="ETHUSDT",
            base_currency="ETH",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        
        market_manager.instruments["BTCUSDT"] = spec1
        market_manager.instruments["ETHUSDT"] = spec2
        
        base_currencies = market_manager.get_base_currencies()
        
        assert "BTC" in base_currencies
        assert "ETH" in base_currencies
        assert len(base_currencies) == 2

    def test_get_market_data(self, market_manager):
        """Test getting market data."""
        market_data = {
            "ticker": {"price": 50000.0},
            "last_updated": datetime.now()
        }
        market_manager.market_data["BTCUSDT"] = market_data
        
        result = market_manager.get_market_data("BTCUSDT")
        
        assert result == market_data

    def test_get_ticker(self, market_manager):
        """Test getting ticker data."""
        market_data = {
            "ticker": {"price": 50000.0, "volume": 1000.0},
            "last_updated": datetime.now()
        }
        market_manager.market_data["BTCUSDT"] = market_data
        
        ticker = market_manager.get_ticker("BTCUSDT")
        
        assert ticker == {"price": 50000.0, "volume": 1000.0}

    def test_is_symbol_tradable(self, market_manager):
        """Test checking if symbol is tradable."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        market_manager.instruments["BTCUSDT"] = spec
        
        assert market_manager.is_symbol_tradable("BTCUSDT") is True
        assert market_manager.is_symbol_tradable("NONEXISTENT") is False

    def test_get_minimum_order_size(self, market_manager):
        """Test getting minimum order size."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        market_manager.instruments["BTCUSDT"] = spec
        
        min_size = market_manager.get_minimum_order_size("BTCUSDT")
        
        assert min_size == 10.0

    def test_get_maximum_leverage(self, market_manager):
        """Test getting maximum leverage."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.FUTURES,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            max_leverage=20.0
        )
        market_manager.instruments["BTCUSDT"] = spec
        
        max_leverage = market_manager.get_maximum_leverage("BTCUSDT")
        
        assert max_leverage == 20.0

    def test_get_price_precision(self, market_manager):
        """Test getting price precision."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        market_manager.instruments["BTCUSDT"] = spec
        
        precision = market_manager.get_price_precision("BTCUSDT")
        
        assert precision == 2

    def test_get_quantity_precision(self, market_manager):
        """Test getting quantity precision."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5
        )
        market_manager.instruments["BTCUSDT"] = spec
        
        precision = market_manager.get_quantity_precision("BTCUSDT")
        
        assert precision == 5

    def test_should_refresh_no_refresh(self, market_manager):
        """Test should refresh when no previous refresh."""
        assert market_manager.should_refresh() is True

    def test_should_refresh_fresh(self, market_manager):
        """Test should refresh when data is fresh."""
        market_manager.last_refresh = datetime.now()
        
        assert market_manager.should_refresh() is False

    def test_should_refresh_stale(self, market_manager):
        """Test should refresh when data is stale."""
        market_manager.last_refresh = datetime.now() - timedelta(minutes=10)
        
        assert market_manager.should_refresh() is True

    @pytest.mark.asyncio
    async def test_ensure_fresh_data_refresh_needed(self, market_manager, mock_refresh_functions):
        """Test ensure fresh data when refresh is needed."""
        market_manager.last_refresh = datetime.now() - timedelta(minutes=10)
        market_manager.register_refresh_functions(**mock_refresh_functions)
        
        result = await market_manager.ensure_fresh_data()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_ensure_fresh_data_no_refresh_needed(self, market_manager):
        """Test ensure fresh data when no refresh is needed."""
        market_manager.last_refresh = datetime.now()
        
        result = await market_manager.ensure_fresh_data()
        
        assert result is True

    def test_search_instruments(self, market_manager):
        """Test searching instruments with filters."""
        spot_spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        futures_spec = InstrumentSpec(
            symbol="BTCUSDT-SWAP",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.FUTURES,
            status="active",
            tick_size=0.1,
            lot_size=0.001,
            min_notional=100.0,
            price_precision=1,
            quantity_precision=3,
            max_leverage=20.0,
            is_active=True
        )
        
        market_manager.instruments["BTCUSDT"] = spot_spec
        market_manager.instruments["BTCUSDT-SWAP"] = futures_spec
        
        # Search by base currency
        btc_instruments = market_manager.search_instruments(base_currency="BTC")
        assert len(btc_instruments) == 2
        
        # Search by instrument type
        spot_instruments = market_manager.search_instruments(instrument_type=InstrumentType.SPOT)
        assert len(spot_instruments) == 1
        
        # Search by leverage
        high_leverage = market_manager.search_instruments(min_leverage=15.0)
        assert len(high_leverage) == 1

    def test_get_statistics(self, market_manager):
        """Test getting statistics."""
        spec = InstrumentSpec(
            symbol="BTCUSDT",
            base_currency="BTC",
            quote_currency="USDT",
            instrument_type=InstrumentType.SPOT,
            status="active",
            tick_size=0.01,
            lot_size=0.00001,
            min_notional=10.0,
            price_precision=2,
            quantity_precision=5,
            is_active=True
        )
        market_manager.instruments["BTCUSDT"] = spec
        market_manager.last_refresh = datetime.now()
        
        stats = market_manager.get_statistics()
        
        assert stats["total_instruments"] == 1
        assert stats["active_instruments"] == 1
        assert stats["inactive_instruments"] == 0
        assert "type_distribution" in stats
        assert "last_refresh" in stats

    def test_cleanup_old_data(self, market_manager):
        """Test cleaning up old data."""
        # Add old market data
        old_data = {
            "ticker": {"price": 50000.0},
            "last_updated": datetime.now() - timedelta(hours=25)
        }
        market_manager.market_data["BTCUSDT"] = old_data
        
        cleaned = market_manager.cleanup_old_data(max_age_hours=24)
        
        assert cleaned == 1
        assert "BTCUSDT" not in market_manager.market_data