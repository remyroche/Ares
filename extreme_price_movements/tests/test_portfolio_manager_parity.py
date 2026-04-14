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
