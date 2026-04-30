import pandas as pd

from extreme_price_movements.portfolio_manager import PortfolioManager


def test_portfolio_manager_caps_size_at_5000_usdt():
    pm = PortfolioManager(portfolio_value=100000.0, max_position_usdt=5000.0)
    can_enter, info = pm.can_enter_position(
        symbol="BTC/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=0.9,
        initial_threshold=0.5,
        current_time=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    assert can_enter
    assert info["position_size_cap"] == 5000.0


def test_portfolio_manager_defaults_to_15pct_position_cap_without_total_cap():
    pm = PortfolioManager(portfolio_value=10000.0)
    can_enter, info = pm.can_enter_position(
        symbol="BTC/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=0.9,
        initial_threshold=0.5,
        current_time=pd.Timestamp("2026-01-01", tz="UTC"),
        requested_position_size=10000.0,
    )
    assert can_enter
    assert info["position_size_cap"] == 1500.0
    state = pm.get_portfolio_state()
    assert state["max_invested_pct"] is None
    assert state["max_position_pct"] == 0.15


def test_portfolio_manager_allows_only_two_positions_per_strategy():
    pm = PortfolioManager(portfolio_value=10000.0)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(2):
        pm.record_position_open(
            symbol=f"SYM{i}/USDT",
            side="long",
            strategy_id="long_mr",
            position_size=1000.0,
            entry_price=100.0,
            entry_time=t0,
        )
    can_enter, info = pm.can_enter_position(
        symbol="NEXT/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=t0 + pd.Timedelta(minutes=1),
        requested_position_size=1000.0,
    )
    assert not can_enter
    assert info["reason"].startswith("max_same_strategy_reached")


def test_dynamic_threshold_widens_with_open_positions():
    pm = PortfolioManager(max_positions=4)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(2):
        pm.record_position_open(
            symbol=f"SYM{i}/USDT",
            side="long",
            strategy_id="long_mr",
            position_size=1000.0,
            entry_price=100.0,
            entry_time=t0,
        )
    thr = pm.calculate_dynamic_threshold(0.5)
    assert abs(thr - 0.75) < 1e-12


def test_portfolio_manager_fetches_wallet_and_exchange_positions():
    class _Exchange:
        def __init__(self):
            self.balance_params = None
            self.position_params = None

        def fetch_balance(self, params=None):
            self.balance_params = params
            return {
                "total": {"USDC": 12345.0},
                "free": {"USDC": 10000.0},
                "used": {"USDC": 2345.0},
            }

        def fetch_positions(self, params=None):
            self.position_params = params
            return [
                {"symbol": "BTC/USDT", "contracts": 0.25},
                {"symbol": "ETH/USDT", "contracts": 0.0},
                {"symbol": "SOL/USDT", "info": {"positionAmt": "-2.0"}},
            ]

    exchange = _Exchange()
    pm = PortfolioManager(portfolio_value=10000.0)
    pm.record_position_open(
        symbol="BTC/USDT",
        side="long",
        strategy_id="long_mr",
        position_size=1000.0,
        entry_price=100.0,
        entry_time=pd.Timestamp("2026-01-01", tz="UTC"),
    )

    snapshot = pm.fetch_exchange_snapshot(exchange)

    assert exchange.balance_params == {"type": "margin", "marginMode": "cross"}
    assert exchange.position_params == {"type": "margin", "marginMode": "cross"}
    assert snapshot["execution_account"] == "margin"
    assert snapshot["margin_mode"] == "cross"
    assert snapshot["total_balance"] == 12345.0
    assert snapshot["free_balance"] == 10000.0
    assert snapshot["used_balance"] == 2345.0
    assert snapshot["exchange_open_positions"] == 2
    assert snapshot["local_open_positions"] == 1
    assert snapshot["errors"] == []
    assert pm.portfolio_value == 12345.0


def test_portfolio_manager_records_private_api_failures():
    class _Exchange:
        def fetch_balance(self, params=None):
            raise TimeoutError("network timeout")

        def fetch_positions(self, params=None):
            raise RuntimeError("positions unavailable")

    pm = PortfolioManager(portfolio_value=10000.0)
    snapshot = pm.fetch_exchange_snapshot(_Exchange())

    assert len(snapshot["errors"]) == 2
    assert snapshot["error_categories"] == ["timeout", "api_error"]
    assert len(pm.failed_api_events) == 2
