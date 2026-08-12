from __future__ import annotations

from dataclasses import replace

import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    normalise_candidate_table,
    replay_candidates,
)


def _candidate(multiplier: float) -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    return normalise_candidate_table(
        pd.DataFrame(
            {
                "timestamp": [timestamp],
                "symbol": ["BTC/USD:USD"],
                "side": ["long"],
                "strategy_id": ["trust_sizing_long"],
                "policy_archetype": ["trust_sizing_long"],
                "normalized_rank_score": [1.0],
                "strategy_rank_pct": [1.0],
                "base_strategy_threshold": [0.0],
                "calibrated_score": [1.0],
                "entry_price": [100.0],
                "exit_timestamp": [timestamp + pd.Timedelta(hours=1)],
                "exit_price": [101.0],
                "net_return": [0.01],
                "gross_return": [0.02],
                "holding_bars": [4],
                "simple_policy_exit_reason": ["TIMEOUT"],
                "portfolio_size_multiplier": [multiplier],
            }
        )
    )


def _params() -> PortfolioPolicyParams:
    return PortfolioPolicyParams(
        capacity_mode="pre_leverage_wallet",
        enforce_position_count_cap=True,
        max_concurrent_positions=8,
        max_concurrent_per_side=8,
        max_concurrent_per_strategy=None,
        max_concurrent_per_symbol=1,
        max_new_entries_per_bar=2,
        max_new_entries_per_strategy_per_bar=2,
        max_total_wallet_allocation_pct=0.80,
        perp_default_leverage=7.0,
        max_position_quote_notional=1_000_000_000.0,
        margin_slot_wallet_fraction=0.10,
        global_threshold_floor=0.0,
        threshold_viability_margin=0.0,
        occupancy_threshold_alpha=0.0,
        allocation_threshold_alpha=0.0,
        rank_size_power=1.0,
        rank_multiplier_min=1.0,
        rank_multiplier_max=1.0,
        strategy_ids=("trust_sizing_long",),
    )


def test_margin_slot_preserves_candidate_trust_size_multiplier() -> None:
    half, _, _ = replay_candidates(
        _candidate(0.5), params=_params(), initial_wallet=1_000.0,
        market_mode="perp",
    )
    full, _, _ = replay_candidates(
        _candidate(1.0), params=_params(), initial_wallet=1_000.0,
        market_mode="perp",
    )
    half_size = float(half.loc[half["accepted"], "position_size"].iloc[0])
    full_size = float(full.loc[full["accepted"], "position_size"].iloc[0])
    assert half_size == 350.0
    assert full_size == 700.0


def test_entry_margin_cap_is_distinct_from_marked_exposure() -> None:
    """A winning open trade may mark above its entry cap without new margin."""
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for index in range(8):
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": f"ASSET{index}/USD:USD",
                "side": "long",
                "strategy_id": "trust_sizing_long",
                "policy_archetype": "trust_sizing_long",
                "normalized_rank_score": 1.0,
                "strategy_rank_pct": 1.0,
                "base_strategy_threshold": 0.0,
                "calibrated_score": 1.0,
                "entry_price": 100.0,
                "exit_timestamp": timestamp + pd.Timedelta(hours=2),
                "exit_price": 103.0,
                "net_return": 0.02,
                "gross_return": 0.03,
                "holding_bars": 8,
                "simple_policy_exit_reason": "TIMEOUT",
                "portfolio_size_multiplier": 1.0,
            }
        )
    # A later candidate forces an intermediate mark while the eight positions
    # are open.  It cannot enter because the count cap is already full.
    rows.append({**rows[0], "timestamp": timestamp + pd.Timedelta(hours=1), "symbol": "LATE/USD:USD"})
    _, equity, metrics = replay_candidates(
        normalise_candidate_table(pd.DataFrame(rows)),
        params=replace(
            _params(),
            max_new_entries_per_bar=8,
            max_new_entries_per_strategy_per_bar=8,
        ),
        initial_wallet=1_000.0,
        market_mode="perp",
    )

    assert float(equity["committed_wallet_cap_utilization"].max()) <= 1.0
    assert float(equity["wallet_cap_utilization"].max()) > 1.0
    assert float(metrics["max_entry_marked_margin_cap_utilization"]) <= 1.0
    assert metrics["guardrails"]["max_capital_allocation_ok"] is True
