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
