from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.replay_strict_r3_forward_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _auction_candidates,
)
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _run
from extreme_price_movements.portfolio_policy_replay import (
    PortfolioState,
    replay_candidates,
)
from scripts.replay_strict_r3_policy_portfolio_2025_2026 import _params
from extreme_price_movements.strict_r3_shadow_portfolio import (
    ShadowPortfolioPolicy,
    ShadowPortfolioState,
    auction_admitted_snapshot,
)


def _ledger(net_bps: list[float]) -> pd.DataFrame:
    ts = pd.Timestamp("2026-08-01T00:00:00Z")
    n = len(net_bps)
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(n)],
        "__decision_ts__": [ts] * n,
        "__symbol__": [f"S{i}" for i in range(n)],
        "side_name": ["long"] * n,
        "causal_21d_side_admitted_ge_50bps": [True] * n,
        "causal_21d_side_expected_net_bps": [90.0, 80.0, 70.0][:n],
        "final_score": [0.9, 0.8, 0.7][:n],
        "policy_path_valid": [True] * n,
        "policy_net_bps": net_bps,
        "policy_gross_bps": (np.asarray(net_bps) + 100.0).tolist(),
        "policy_exit_bar_15m": [1] * n,
        "policy_entry_price": [1.0] * n,
        "policy_exit_price": [1.0] * n,
        "policy_exit_reason": ["TIMEOUT"] * n,
    })


def test_canonical_auction_selection_is_invariant_to_held_outcomes() -> None:
    def selected(values: list[float]) -> list[int]:
        candidates = _auction_candidates(_ledger(values))
        decisions, _, _, _ = _run(
            candidates,
            0.0,
            "test",
            initial_wallet=1000.0,
            perp_leverage=7.0,
            margin_slot_wallet_fraction=0.10,
            ev_curve=CAUSAL_AUCTION_CURVE,
        )
        return decisions.loc[decisions["accepted"], "candidate_index"].tolist()

    assert selected([500.0, -900.0, 100.0]) == selected([-900.0, 500.0, 100.0])


def test_shadow_auction_matches_replay_engine_on_empty_state() -> None:
    ledger = _ledger([500.0, -900.0, 100.0])
    replay_input = _auction_candidates(ledger)
    replay, _, _ = replay_candidates(
        replay_input,
        _params(
            0.0,
            perp_leverage=7.0,
            margin_slot_wallet_fraction=0.10,
            strategy_ids=("strict_r3_current_v5_long",),
        ),
        mode="global_auction",
        ev_curve=CAUSAL_AUCTION_CURVE,
        initial_wallet=1000.0,
        initial_state=PortfolioState(wallet=1000.0),
        market_mode="perps",
    )
    replay_ids = replay_input.iloc[
        replay.loc[replay["accepted"], "candidate_index"].astype(int)
    ]["candidate_id"].tolist()
    shadow_policy = ShadowPortfolioPolicy.from_payload({
        "schema": "strict_r3_cell_day_trim15_portfolio_v1",
        "max_concurrent_positions": 8,
        "max_concurrent_per_symbol": 1,
        "max_new_entries_per_bar": 2,
        "max_total_margin_fraction": 0.8,
        "margin_slot_fraction": 0.1,
        "leverage": 7.0,
        "minimum_gross_notional": 1.0,
    })
    state = ShadowPortfolioState.from_payload({
        "schema": "strict_r3_shadow_portfolio_snapshot_v1",
        "as_of_ts": "2026-08-01T00:00:00Z",
        "wallet": 1000.0,
        "open_positions": [],
    }, expected_as_of_ts=pd.Timestamp("2026-08-01T00:00:00Z"))
    shadow = auction_admitted_snapshot(ledger, state=state, policy=shadow_policy)
    shadow_ids = shadow.loc[shadow["portfolio_accepted"], "candidate_id"].tolist()
    assert shadow_ids == replay_ids
