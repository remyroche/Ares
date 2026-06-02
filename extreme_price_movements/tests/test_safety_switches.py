import pandas as pd

from extreme_price_movements.inference.safety_switches import (
    MarketKillSwitch,
    StrategyKillSwitch,
)


def test_market_kill_switch_triggers_on_usdc_depeg(tmp_path):
    switch = MarketKillSwitch(tmp_path / "market_kill_switch.json")
    decision = switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 0.97},
        btc_close=pd.Series([100.0, 101.0]),
        eth_close=pd.Series([100.0, 101.0]),
        basket_close=pd.DataFrame({"A": [100.0, 101.0]}),
    )
    assert decision.active is True
    assert decision.allow_new_entries is False
    assert decision.reason == "USDC_USDT_DEPEG"


def test_market_kill_switch_self_recovers_after_halt(tmp_path):
    switch = MarketKillSwitch(tmp_path / "market_kill_switch.json")
    switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 0.97},
        btc_close=pd.Series([100.0, 101.0]),
        eth_close=pd.Series([100.0, 101.0]),
        basket_close=pd.DataFrame({"A": [100.0, 101.0]}),
    )
    recovered = switch.evaluate(
        now=pd.Timestamp("2026-05-11T01:00:00Z"),
        usdc_usdt_ticker={"last": 1.0},
        btc_close=pd.Series([100.0, 101.0]),
        eth_close=pd.Series([100.0, 101.0]),
        basket_close=pd.DataFrame({"A": [100.0, 101.0]}),
    )
    assert recovered.allow_new_entries is True
    assert recovered.active is False


def test_market_kill_switch_ignores_single_asset_spike_when_average_is_safe(tmp_path):
    switch = MarketKillSwitch(
        tmp_path / "market_kill_switch.json",
        min_market_basket_assets=2,
    )
    decision = switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 1.0},
        btc_close=pd.Series([100.0, 115.0]),
        eth_close=pd.Series([100.0, 100.0]),
        basket_close=pd.DataFrame(
            {
                "BTC/USD:USD": [100.0, 115.0],
                "ETH/USD:USD": [100.0, 100.0],
                "SOL/USD:USD": [100.0, 100.0],
                "XRP/USD:USD": [100.0, 100.0],
            }
        ),
    )
    assert decision.allow_new_entries is True
    assert decision.active is False
    assert decision.reason == "allowed"
    assert "btc_1h_move" not in decision.details
    assert "eth_1h_move" not in decision.details
    assert decision.details["market_avg_1h_abs_move"] < 0.05


def test_market_kill_switch_ignores_btc_eth_only_market_move(tmp_path):
    switch = MarketKillSwitch(
        tmp_path / "market_kill_switch.json",
        min_market_basket_assets=2,
    )
    decision = switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 1.0},
        btc_close=pd.Series([100.0, 115.0]),
        eth_close=pd.Series([100.0, 112.0]),
        basket_close=pd.DataFrame(
            {
                "BTC/USD:USD": [100.0, 115.0],
                "ETH/USD:USD": [100.0, 112.0],
            }
        ),
    )
    assert decision.allow_new_entries is True
    assert decision.active is False
    assert decision.reason == "allowed"
    assert decision.details["market_basket_assets"] == 0
    assert decision.details["market_basket_status"] == "insufficient_breadth"


def test_market_kill_switch_triggers_on_average_market_spike(tmp_path):
    switch = MarketKillSwitch(
        tmp_path / "market_kill_switch.json",
        min_market_basket_assets=2,
    )
    decision = switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 1.0},
        btc_close=pd.Series([100.0, 106.0]),
        eth_close=pd.Series([100.0, 106.0]),
        basket_close=pd.DataFrame(
            {
                "BTC/USD:USD": [100.0, 106.0],
                "ETH/USD:USD": [100.0, 106.0],
                "SOL/USD:USD": [100.0, 106.0],
                "XRP/USD:USD": [100.0, 106.0],
            }
        ),
    )
    assert decision.allow_new_entries is False
    assert decision.active is True
    assert decision.reason == "MARKET_AVG_1H_MOVE_GT_5PCT"


def test_market_kill_switch_does_not_preserve_deprecated_per_asset_halt(tmp_path):
    path = tmp_path / "market_kill_switch.json"
    path.write_text(
        """
        {
          "active": true,
          "reason": "BTC_1H_MOVE_GT_7PCT",
          "halt_until": "2026-05-11T00:00:00+00:00",
          "details": {"btc_1h_move": 0.15}
        }
        """,
        encoding="utf-8",
    )
    switch = MarketKillSwitch(path, min_market_basket_assets=2)
    decision = switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 1.0},
        btc_close=pd.Series([100.0, 115.0]),
        eth_close=pd.Series([100.0, 100.0]),
        basket_close=pd.DataFrame(
            {
                "BTC/USD:USD": [100.0, 115.0],
                "ETH/USD:USD": [100.0, 100.0],
                "SOL/USD:USD": [100.0, 100.0],
                "XRP/USD:USD": [100.0, 100.0],
            }
        ),
    )
    assert decision.allow_new_entries is True
    assert decision.active is False
    assert decision.reason == "allowed"


def test_market_kill_switch_active_halt_reports_current_and_stored_details(tmp_path):
    path = tmp_path / "market_kill_switch.json"
    path.write_text(
        """
        {
          "active": true,
          "self_reversible": true,
          "reason": "MARKET_AVG_1H_MOVE_GT_5PCT",
          "triggered_at": "2026-05-10T11:00:00+00:00",
          "halt_until": "2026-05-11T00:00:00+00:00",
          "details": {
            "btc_1h_move": 0.149,
            "market_avg_1h_abs_move": 0.221
          }
        }
        """,
        encoding="utf-8",
    )
    switch = MarketKillSwitch(path, min_market_basket_assets=2)
    decision = switch.evaluate(
        now=pd.Timestamp("2026-05-10T12:00:00Z"),
        usdc_usdt_ticker={"last": 1.0},
        btc_close=pd.Series([100.0, 101.0]),
        eth_close=pd.Series([100.0, 101.0]),
        basket_close=pd.DataFrame(
            {
                "BTC/USD:USD": [100.0, 101.0],
                "ETH/USD:USD": [100.0, 101.0],
                "SOL/USD:USD": [100.0, 101.0],
                "XRP/USD:USD": [100.0, 101.0],
            }
        ),
    )

    assert decision.allow_new_entries is False
    assert decision.active is True
    assert decision.reason == "MARKET_AVG_1H_MOVE_GT_5PCT"
    assert decision.details["halt_source"] == "stored_state"
    assert decision.details["market_avg_1h_abs_move"] < 0.05
    assert decision.details["stored_halt_details"]["market_avg_1h_abs_move"] == 0.221


def test_strategy_kill_switch_observe_only_does_not_block(tmp_path):
    switch = StrategyKillSwitch(tmp_path / "strategy_kill_switches.json")
    switch.set_state("long_test", active=True, reason="weak_hit_rate")
    decision = switch.is_blocked("long_test")
    assert decision.allow_new_entries is True
    assert decision.active is True


def test_strategy_kill_switch_blocks_when_enabled(tmp_path):
    switch = StrategyKillSwitch(
        tmp_path / "strategy_kill_switches.json",
        observe_only=False,
    )
    switch.set_state("long_test", active=True, reason="weak_hit_rate")
    decision = switch.is_blocked("long_test")
    assert decision.allow_new_entries is False
    assert decision.active is True
