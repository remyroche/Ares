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


def test_portfolio_manager_defaults_to_75pct_total_and_15pct_position_cap():
    pm = PortfolioManager(portfolio_value=10000.0)
    can_enter, info = pm.can_enter_position(
        symbol="BTC/USDT",
        side="long",
        strategy_id="long_mr",
        confidence_score=0.9,
        initial_threshold=0.5,
        current_time=pd.Timestamp("2026-01-01", tz="UTC"),
        requested_position_size=1500.0,
    )
    assert can_enter
    assert info["position_size_cap"] == 1500.0
    state = pm.get_portfolio_state()
    assert state["max_invested_pct"] == 0.75
    assert state["max_position_pct"] == 0.15


def test_portfolio_manager_leveraged_caps_apply_before_quote_notional():
    pm = PortfolioManager(
        portfolio_value=100.0,
        max_portfolio_pct=0.75,
        max_position_pct=0.15,
        max_position_usdt=5000.0,
        leverage_wallet_multiplier=10.0,
    )
    pm.update_margin_account_metrics(
        total_assets_quote=100.0,
        total_liabilities_quote=0.0,
    )

    cap = pm.get_portfolio_capacity(side="long", strategy_id="long_mr")
    assert cap["max_position_notional"] == 150.0
    assert cap["max_total_notional"] == 750.0
    assert cap["margin_surplus_notional"] == 1000.0

    can_enter, info = pm.can_enter_position(
        symbol="BTC/USD:USD",
        side="long",
        strategy_id="long_mr",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=pd.Timestamp("2026-01-01", tz="UTC"),
        requested_position_size=200.0,
    )

    assert can_enter
    assert info["position_size_cap"] == 150.0


def test_portfolio_manager_clips_oversized_request_to_remaining_capacity():
    pm = PortfolioManager(portfolio_value=10000.0)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    pm.record_position_open(
        symbol="OPEN/USDT",
        side="long",
        strategy_id="long_mr",
        position_size=6500.0,
        entry_price=100.0,
        entry_time=t0,
    )

    can_enter, info = pm.can_enter_position(
        symbol="NEXT/USDT",
        side="long",
        strategy_id="long_mr_2",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=t0 + pd.Timedelta(minutes=1),
        requested_position_size=2000.0,
    )

    assert can_enter
    assert info["reason"] == "allowed"
    assert info["position_size_cap"] == 1000.0
    assert info["requested_size_clipped_to_remaining_total_notional"] is True


def test_portfolio_manager_allows_six_positions_per_strategy_by_default():
    pm = PortfolioManager(portfolio_value=10000.0)
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(6):
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
    assert info["reason"] == "max_concurrent_per_side_reached"


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


def test_archetype_loss_streak_blocks_only_that_archetype():
    pm = PortfolioManager(
        portfolio_value=10000.0,
        max_consecutive_losing_trades=10,
        max_consecutive_losing_trades_per_archetype=5,
    )
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(5):
        symbol = f"LOSS{i}/USDT"
        pm.record_position_open(
            symbol=symbol,
            side="long",
            strategy_id="long_meta",
            position_size=100.0,
            entry_price=100.0,
            entry_time=t0 + pd.Timedelta(minutes=i),
            policy_archetype="long__weak_path",
        )
        result = pm.record_position_close(
            symbol=symbol,
            exit_price=99.0,
            exit_time=t0 + pd.Timedelta(minutes=i, seconds=1),
            exit_reason="test_loss",
        )

    assert result is not None
    assert result["consecutive_losing_trades"] == 5
    assert result["archetype_consecutive_losing_trades"] == 5
    assert result["risk_guard_events"][0]["event"] == "archetype_loss_streak_disabled"
    assert not pm.manual_reset_required

    blocked, blocked_info = pm.can_enter_position(
        symbol="BLOCKED/USDT",
        side="long",
        strategy_id="long_meta",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=t0 + pd.Timedelta(minutes=10),
        requested_position_size=100.0,
        policy_archetype="long__weak_path",
    )
    assert not blocked
    assert blocked_info["reason"] == "archetype_loss_streak_block"

    allowed, allowed_info = pm.can_enter_position(
        symbol="OTHER/USDT",
        side="long",
        strategy_id="long_meta",
        confidence_score=1.0,
        initial_threshold=0.5,
        current_time=t0 + pd.Timedelta(minutes=10),
        requested_position_size=100.0,
        policy_archetype="long__clean_pullback",
    )
    assert allowed
    assert allowed_info["reason"] == "allowed"


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


def test_portfolio_manager_uses_kraken_futures_flex_balance():
    class _Exchange:
        id = "krakenfutures"

        def __init__(self):
            self.balance_params = None
            self.position_params = None

        def fetch_balance(self, params=None):
            self.balance_params = params
            return {
                "info": {
                    "accounts": {
                        "flex": {
                            "marginEquity": "57.25",
                            "availableMargin": "55.50",
                            "initialMarginWithOrders": "1.75",
                            "maintenanceMargin": "0.50",
                        }
                    }
                }
            }

        def fetch_positions(self, symbols=None, params=None):
            self.position_params = params
            return []

    exchange = _Exchange()
    pm = PortfolioManager(portfolio_value=10000.0)

    snapshot = pm.fetch_exchange_snapshot(
        exchange,
        quote_currency="USD",
        execution_account="margin",
        margin_mode="cross",
    )

    assert exchange.balance_params == {"type": "flex"}
    assert exchange.position_params == {}
    assert snapshot["total_balance"] == 57.25
    assert snapshot["free_balance"] == 55.50
    assert snapshot["used_balance"] == 1.75
    assert snapshot["errors"] == []


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
