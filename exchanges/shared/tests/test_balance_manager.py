"""
Unit tests for BalanceManager.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from exchanges.shared.wallet.balance_manager import (
    BalanceManager, Balance, AccountEquity, AccountType
)


class TestBalanceManager:
    """Test cases for BalanceManager."""

    @pytest.fixture
    def balance_manager(self):
        """Create BalanceManager instance for testing."""
        return BalanceManager("test_exchange")

    @pytest.fixture
    def mock_fetch_functions(self):
        """Create mock fetch functions."""
        return {
            "get_balances": AsyncMock(return_value=[
                {
                    "currency": "BTC",
                    "available": "0.5",
                    "frozen": "0.1",
                    "total": "0.6"
                },
                {
                    "currency": "USDT",
                    "available": "1000.0",
                    "frozen": "200.0",
                    "total": "1200.0"
                }
            ]),
            "get_account_info": AsyncMock(return_value={
                "account_id": "123",
                "account_type": "spot"
            })
        }

    def test_initialization(self, balance_manager):
        """Test BalanceManager initialization."""
        assert balance_manager.exchange_name == "test_exchange"
        assert len(balance_manager.balances) == 0
        assert len(balance_manager.account_equities) == 0
        assert balance_manager.cache_ttl == timedelta(seconds=30)
        assert balance_manager.last_fetch is None

    def test_register_fetch_functions(self, balance_manager, mock_fetch_functions):
        """Test registering fetch functions."""
        balance_manager.register_fetch_functions(**mock_fetch_functions)
        
        assert "get_balances" in balance_manager.fetch_functions
        assert "get_account_info" in balance_manager.fetch_functions

    @pytest.mark.asyncio
    async def test_fetch_balances_success(self, balance_manager, mock_fetch_functions):
        """Test successful balance fetching."""
        balance_manager.register_fetch_functions(**mock_fetch_functions)
        
        balances = await balance_manager.fetch_balances(AccountType.SPOT)
        
        assert len(balances) == 2
        assert balances[0].currency == "BTC"
        assert balances[0].available == 0.5
        assert balances[0].frozen == 0.1
        assert balances[0].total == 0.6
        assert balances[0].account_type == AccountType.SPOT
        assert balance_manager.last_fetch is not None

    @pytest.mark.asyncio
    async def test_fetch_balances_no_function(self, balance_manager):
        """Test balance fetching without registered function."""
        balances = await balance_manager.fetch_balances(AccountType.SPOT)
        
        assert len(balances) == 0

    @pytest.mark.asyncio
    async def test_fetch_balances_no_data(self, balance_manager):
        """Test balance fetching with no data."""
        mock_functions = {
            "get_balances": AsyncMock(return_value=None),
            "get_account_info": AsyncMock()
        }
        balance_manager.register_fetch_functions(**mock_functions)
        
        balances = await balance_manager.fetch_balances(AccountType.SPOT)
        
        assert len(balances) == 0

    def test_parse_balance_data(self, balance_manager):
        """Test parsing balance data."""
        raw_data = [
            {
                "currency": "BTC",
                "available": "0.5",
                "frozen": "0.1",
                "total": "0.6"
            },
            {
                "ccy": "USDT",  # Alternative field name
                "avail": "1000.0",  # Alternative field name
                "frozenBal": "200.0",  # Alternative field name
                "totalEq": "1200.0"  # Alternative field name
            }
        ]
        
        balances = balance_manager._parse_balance_data(raw_data, AccountType.SPOT)
        
        assert len(balances) == 2
        assert balances[0].currency == "BTC"
        assert balances[0].available == 0.5
        assert balances[0].frozen == 0.1
        assert balances[0].total == 0.6
        assert balances[1].currency == "USDT"
        assert balances[1].available == 1000.0
        assert balances[1].frozen == 200.0
        assert balances[1].total == 1200.0

    def test_parse_balance_data_calculate_total(self, balance_manager):
        """Test parsing balance data with calculated total."""
        raw_data = [
            {
                "currency": "BTC",
                "available": "0.5",
                "frozen": "0.1"
                # No total field
            }
        ]
        
        balances = balance_manager._parse_balance_data(raw_data, AccountType.SPOT)
        
        assert len(balances) == 1
        assert balances[0].total == 0.6  # available + frozen

    def test_parse_balance_data_invalid(self, balance_manager):
        """Test parsing invalid balance data."""
        raw_data = [
            {
                "invalid": "data"
            },
            {
                "currency": "BTC",
                "available": "invalid_number",
                "frozen": "0.1"
            }
        ]
        
        balances = balance_manager._parse_balance_data(raw_data, AccountType.SPOT)
        
        assert len(balances) == 0

    def test_update_balance_cache(self, balance_manager):
        """Test updating balance cache."""
        balances = [
            Balance(
                currency="BTC",
                available=0.5,
                frozen=0.1,
                total=0.6,
                account_type=AccountType.SPOT
            ),
            Balance(
                currency="USDT",
                available=1000.0,
                frozen=200.0,
                total=1200.0,
                account_type=AccountType.SPOT
            )
        ]
        
        balance_manager._update_balance_cache(balances, AccountType.SPOT)
        
        assert "BTC" in balance_manager.balances
        assert "USDT" in balance_manager.balances
        assert AccountType.SPOT in balance_manager.balances["BTC"]
        assert AccountType.SPOT in balance_manager.balances["USDT"]

    def test_get_balance(self, balance_manager):
        """Test getting balance for specific currency and account type."""
        balance = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance}
        
        retrieved = balance_manager.get_balance("BTC", AccountType.SPOT)
        
        assert retrieved == balance

    def test_get_balance_not_found(self, balance_manager):
        """Test getting non-existent balance."""
        result = balance_manager.get_balance("NONEXISTENT", AccountType.SPOT)
        
        assert result is None

    def test_get_all_balances(self, balance_manager):
        """Test getting all balances for account type."""
        balance1 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance3 = Balance(
            currency="ETH",
            available=1.0,
            frozen=0.0,
            total=1.0,
            account_type=AccountType.FUTURES
        )
        
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance1}
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance2}
        balance_manager.balances["ETH"] = {AccountType.FUTURES: balance3}
        
        spot_balances = balance_manager.get_all_balances(AccountType.SPOT)
        futures_balances = balance_manager.get_all_balances(AccountType.FUTURES)
        
        assert len(spot_balances) == 2
        assert len(futures_balances) == 1
        assert balance1 in spot_balances
        assert balance2 in spot_balances
        assert balance3 in futures_balances

    def test_get_balance_summary(self, balance_manager):
        """Test getting balance summary."""
        balance1 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance1}
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance2}
        
        summary = balance_manager.get_balance_summary(AccountType.SPOT)
        
        assert summary["account_type"] == "spot"
        assert summary["total_equity"] == 1200.6
        assert summary["available_equity"] == 1000.5
        assert summary["frozen_equity"] == 200.1
        assert summary["currency_count"] == 2
        assert summary["non_zero_balances"] == 2

    def test_calculate_total_equity(self, balance_manager):
        """Test calculating total equity."""
        balance1 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance1}
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance2}
        
        total_equity = balance_manager.calculate_total_equity(AccountType.SPOT)
        
        assert total_equity == 1200.6

    def test_calculate_available_equity(self, balance_manager):
        """Test calculating available equity."""
        balance1 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance1}
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance2}
        
        available_equity = balance_manager.calculate_available_equity(AccountType.SPOT)
        
        assert available_equity == 1000.5

    def test_get_usdt_balance(self, balance_manager):
        """Test getting USDT balance."""
        usdt_balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: usdt_balance}
        
        usdt_total = balance_manager.get_usdt_balance(AccountType.SPOT)
        
        assert usdt_total == 1200.0

    def test_get_usdt_balance_not_found(self, balance_manager):
        """Test getting USDT balance when not found."""
        usdt_total = balance_manager.get_usdt_balance(AccountType.SPOT)
        
        assert usdt_total == 0.0

    def test_get_btc_balance(self, balance_manager):
        """Test getting BTC balance."""
        btc_balance = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["BTC"] = {AccountType.SPOT: btc_balance}
        
        btc_total = balance_manager.get_btc_balance(AccountType.SPOT)
        
        assert btc_total == 0.6

    def test_get_btc_balance_not_found(self, balance_manager):
        """Test getting BTC balance when not found."""
        btc_total = balance_manager.get_btc_balance(AccountType.SPOT)
        
        assert btc_total == 0.0

    def test_has_sufficient_balance(self, balance_manager):
        """Test checking sufficient balance."""
        balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        assert balance_manager.has_sufficient_balance("USDT", 500.0, AccountType.SPOT) is True
        assert balance_manager.has_sufficient_balance("USDT", 1500.0, AccountType.SPOT) is False
        assert balance_manager.has_sufficient_balance("NONEXISTENT", 100.0, AccountType.SPOT) is False

    def test_get_balance_utilization(self, balance_manager):
        """Test getting balance utilization ratio."""
        balance = Balance(
            currency="USDT",
            available=800.0,
            frozen=200.0,
            total=1000.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        utilization = balance_manager.get_balance_utilization("USDT", AccountType.SPOT)
        
        assert utilization == 0.2  # frozen / total

    def test_get_balance_utilization_zero_total(self, balance_manager):
        """Test getting balance utilization with zero total."""
        balance = Balance(
            currency="USDT",
            available=0.0,
            frozen=0.0,
            total=0.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        utilization = balance_manager.get_balance_utilization("USDT", AccountType.SPOT)
        
        assert utilization == 0.0

    def test_get_top_balances(self, balance_manager):
        """Test getting top balances by total value."""
        balance1 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        balance3 = Balance(
            currency="ETH",
            available=1.0,
            frozen=0.0,
            total=1.0,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance1}
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance2}
        balance_manager.balances["ETH"] = {AccountType.SPOT: balance3}
        
        top_balances = balance_manager.get_top_balances(AccountType.SPOT, limit=2)
        
        assert len(top_balances) == 2
        assert top_balances[0].currency == "USDT"  # Highest total
        assert top_balances[1].currency == "ETH"   # Second highest

    def test_get_non_zero_balances(self, balance_manager):
        """Test getting non-zero balances."""
        balance1 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="BTC",
            available=0.0,
            frozen=0.0,
            total=0.0,
            account_type=AccountType.SPOT
        )
        balance3 = Balance(
            currency="ETH",
            available=1.0,
            frozen=0.0,
            total=1.0,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance1}
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance2}
        balance_manager.balances["ETH"] = {AccountType.SPOT: balance3}
        
        non_zero_balances = balance_manager.get_non_zero_balances(AccountType.SPOT)
        
        assert len(non_zero_balances) == 2
        assert balance1 in non_zero_balances
        assert balance3 in non_zero_balances
        assert balance2 not in non_zero_balances

    def test_calculate_portfolio_value(self, balance_manager):
        """Test calculating portfolio value in base currency."""
        balance1 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance1}
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance2}
        
        prices = {"BTC": 50000.0}
        portfolio_value = balance_manager.calculate_portfolio_value(
            AccountType.SPOT, prices, "USDT"
        )
        
        # USDT: 1200.0 + BTC: 0.6 * 50000.0 = 1200.0 + 30000.0 = 31200.0
        assert portfolio_value == 31200.0

    def test_should_refresh_balances_no_fetch(self, balance_manager):
        """Test should refresh when no previous fetch."""
        assert balance_manager.should_refresh_balances() is True

    def test_should_refresh_balances_fresh(self, balance_manager):
        """Test should refresh when data is fresh."""
        balance_manager.last_fetch = datetime.now()
        
        assert balance_manager.should_refresh_balances() is False

    def test_should_refresh_balances_stale(self, balance_manager):
        """Test should refresh when data is stale."""
        balance_manager.last_fetch = datetime.now() - timedelta(minutes=1)
        
        assert balance_manager.should_refresh_balances() is True

    @pytest.mark.asyncio
    async def test_ensure_fresh_balances_refresh_needed(self, balance_manager, mock_fetch_functions):
        """Test ensure fresh balances when refresh is needed."""
        balance_manager.last_fetch = datetime.now() - timedelta(minutes=1)
        balance_manager.register_fetch_functions(**mock_fetch_functions)
        
        balances = await balance_manager.ensure_fresh_balances(AccountType.SPOT)
        
        assert len(balances) == 2

    @pytest.mark.asyncio
    async def test_ensure_fresh_balances_no_refresh_needed(self, balance_manager):
        """Test ensure fresh balances when no refresh is needed."""
        balance_manager.last_fetch = datetime.now()
        
        # Add some cached balances
        balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        balances = await balance_manager.ensure_fresh_balances(AccountType.SPOT)
        
        assert len(balances) == 1

    def test_get_balance_statistics(self, balance_manager):
        """Test getting balance statistics."""
        balance1 = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance2 = Balance(
            currency="BTC",
            available=0.5,
            frozen=0.1,
            total=0.6,
            account_type=AccountType.SPOT
        )
        
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance1}
        balance_manager.balances["BTC"] = {AccountType.SPOT: balance2}
        balance_manager.last_fetch = datetime.now()
        
        stats = balance_manager.get_balance_statistics()
        
        assert stats["total_currencies"] == 2
        assert stats["total_account_types"] == 0
        assert "account_summaries" in stats
        assert "last_fetch" in stats
        assert stats["cache_ttl_seconds"] == 30

    def test_cleanup_old_balances(self, balance_manager):
        """Test cleaning up old balance data."""
        # Add old balance
        old_balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT,
            timestamp=datetime.now() - timedelta(hours=25)
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: old_balance}
        
        cleaned = balance_manager.cleanup_old_balances(max_age_hours=24)
        
        assert cleaned == 1
        assert "USDT" not in balance_manager.balances

    def test_set_cache_ttl(self, balance_manager):
        """Test setting cache TTL."""
        balance_manager.set_cache_ttl(60)
        
        assert balance_manager.cache_ttl == timedelta(seconds=60)

    def test_invalidate_cache_specific(self, balance_manager):
        """Test invalidating specific currency and account type cache."""
        balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        balance_manager.invalidate_cache("USDT", AccountType.SPOT)
        
        assert "USDT" not in balance_manager.balances

    def test_invalidate_cache_currency(self, balance_manager):
        """Test invalidating cache for specific currency."""
        balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        balance_manager.invalidate_cache("USDT")
        
        assert "USDT" not in balance_manager.balances

    def test_invalidate_cache_account_type(self, balance_manager):
        """Test invalidating cache for specific account type."""
        balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        balance_manager.invalidate_cache(account_type=AccountType.SPOT)
        
        assert "USDT" not in balance_manager.balances

    def test_invalidate_cache_all(self, balance_manager):
        """Test invalidating all cache."""
        balance = Balance(
            currency="USDT",
            available=1000.0,
            frozen=200.0,
            total=1200.0,
            account_type=AccountType.SPOT
        )
        balance_manager.balances["USDT"] = {AccountType.SPOT: balance}
        
        balance_manager.invalidate_cache()
        
        assert len(balance_manager.balances) == 0