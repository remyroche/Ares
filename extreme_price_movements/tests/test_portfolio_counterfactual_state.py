import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    OpenPosition,
    PortfolioPolicyParams,
    PortfolioState,
    fit_hierarchical_ev_curves,
    replay_candidates,
)


def _row(ts: str, symbol: str, strategy_id: str, rank: float, net_return: float, holding_hours: int = 2) -> dict:
    timestamp = pd.Timestamp(ts, tz="UTC")
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": "short" if strategy_id.startswith("short") else "long",
        "strategy_id": strategy_id,
        "normalized_rank_score": rank,
        "strategy_rank_pct": rank,
        "base_strategy_threshold": 0.50,
        "calibrated_score": rank,
        "entry_price": 100.0,
        "exit_timestamp": timestamp + pd.Timedelta(hours=holding_hours),
        "exit_price": 101.0,
        "net_return": net_return,
        "gross_return": net_return + 0.001,
        "fees_bps": 10.0,
        "slippage_bps": 0.0,
        "holding_bars": holding_hours * 4,
        "simple_policy_exit_reason": "tp" if net_return > 0 else "full_sl",
    }


def test_portfolio_state_clone_is_lossless_and_independent():
    state = PortfolioState(wallet=12345.0)
    pos = OpenPosition(
        symbol="BTC",
        side="long",
        strategy_id="long_demo",
        entry_timestamp=pd.Timestamp("2026-01-01", tz="UTC"),
        exit_timestamp=pd.Timestamp("2026-01-01 02:00", tz="UTC"),
        position_size=1000.0,
        net_return=0.01,
        gross_return=0.011,
        exit_reason="tp",
    )
    state.open_position(pos)
    state.cooldowns["ETH"] = pd.Timestamp("2026-01-02", tz="UTC")

    clone = state.clone()
    clone.wallet += 10.0
    clone.open_positions[0].position_size = 10.0
    clone.cooldowns["ETH"] = pd.Timestamp("2026-01-03", tz="UTC")

    assert state.wallet == 12345.0
    assert state.open_positions[0].position_size == 1000.0
    assert state.open_notional == 1000.0
    assert state.side_counts() == {"long": 1, "short": 0}
    assert state.strategy_counts() == {"long_demo": 1}
    assert state.symbol_counts() == {"BTC": 1}
    assert state.cooldowns["ETH"] == pd.Timestamp("2026-01-02", tz="UTC")


def test_cloned_pre_decision_state_replays_noop_suffix_identically():
    candidates = pd.DataFrame(
        [
            _row("2026-01-01 00:00", "BTC", "long_demo", 0.90, 0.01, holding_hours=4),
            _row("2026-01-01 01:00", "ETH", "short_demo", 0.95, -0.01, holding_hours=2),
            _row("2026-01-01 02:00", "SOL", "long_demo", 0.92, 0.02, holding_hours=2),
        ]
    )
    params = PortfolioPolicyParams(
        max_concurrent_positions=3,
        max_new_entries_per_bar=2,
        max_total_wallet_allocation_pct=0.80,
        global_threshold_floor=0.50,
        cooldown_hours_after_loss=0.0,
    )
    ev_curve = fit_hierarchical_ev_curves(candidates)
    snapshots = {}

    def callback(ts, state, _group_idx, _cache):
        snapshots[pd.Timestamp(ts)] = state

    baseline_decisions, _equity, _metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
        pre_decision_snapshot_callback=callback,
    )

    ts = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    suffix = candidates.loc[candidates["timestamp"] >= ts].copy()
    clone_decisions, _clone_equity, _clone_metrics = replay_candidates(
        suffix,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
        initial_state=snapshots[ts],
    )
    baseline_suffix = baseline_decisions.loc[baseline_decisions["timestamp"] >= ts].reset_index(drop=True)
    clone_suffix = clone_decisions.reset_index(drop=True)

    assert baseline_suffix[["timestamp", "symbol", "strategy_id", "accepted", "rejection_reason"]].equals(
        clone_suffix[["timestamp", "symbol", "strategy_id", "accepted", "rejection_reason"]]
    )
    assert baseline_suffix["position_size"].tolist() == clone_suffix["position_size"].tolist()
