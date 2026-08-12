from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.strict_r3_shadow_portfolio import (
    A5_POLICY_SCHEMA,
    POLICY_SCHEMA,
    POSTERIOR_POLICY_SCHEMA,
    SCHEMA,
    ShadowPortfolioPolicy,
    ShadowPortfolioState,
    auction_admitted_snapshot,
)


TS = pd.Timestamp("2026-08-01T00:00:00Z")


def _policy() -> ShadowPortfolioPolicy:
    return ShadowPortfolioPolicy.from_payload({
        "schema": POLICY_SCHEMA,
        "max_concurrent_positions": 8,
        "max_concurrent_per_symbol": 1,
        "max_new_entries_per_bar": 2,
        "max_total_margin_fraction": 0.8,
        "margin_slot_fraction": 0.1,
        "leverage": 7.0,
        "minimum_gross_notional": 1.0,
    })


def _posterior_policy() -> ShadowPortfolioPolicy:
    return ShadowPortfolioPolicy.from_payload({
        "schema": POSTERIOR_POLICY_SCHEMA,
        "max_concurrent_positions": 8,
        "max_concurrent_per_symbol": 1,
        "max_new_entries_per_bar": 2,
        "max_total_margin_fraction": 0.8,
        "margin_slot_fraction": 0.1,
        "leverage": 7.0,
        "minimum_gross_notional": 1.0,
        "admission_expected_net_field": "trust_posterior_expected_bps",
        "admission_threshold_bps": 50.0,
        "missing_posterior": "fail_closed",
    })


def _a5_policy() -> ShadowPortfolioPolicy:
    return ShadowPortfolioPolicy.from_payload({
        "schema": A5_POLICY_SCHEMA,
        "max_concurrent_positions": 8,
        "max_concurrent_per_symbol": 1,
        "max_new_entries_per_bar": 2,
        "max_total_margin_fraction": 0.8,
        "margin_slot_fraction": 0.1,
        "leverage": 7.0,
        "minimum_gross_notional": 1.0,
        "admission_expected_net_field": "a5_bounded10_expected_bps",
        "admission_boolean_field": "a5_bounded10_admitted",
        "anchor_expected_net_field": "trust_posterior_expected_bps",
        "anchor_threshold_bps": 50.0,
        "domain": "timestamp_local_top15_by_pretrust_final_score",
        "missing_component": "fail_closed",
    })


def _state(*, positions=()) -> ShadowPortfolioState:
    return ShadowPortfolioState.from_payload({
        "schema": SCHEMA,
        "as_of_ts": TS.isoformat(),
        "wallet": 1000.0,
        "open_positions": list(positions),
    }, expected_as_of_ts=TS)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "__decision_ts__": [TS] * 4,
        "__symbol__": ["A", "B", "C", "D"],
        "side_name": ["long"] * 4,
        "final_score": [0.80, 0.99, 0.70, 0.60],
        "causal_21d_side_expected_net_bps": [80.0, 70.0, 60.0, -5.0],
        "causal_21d_side_admitted_ge_50bps": [True, True, True, False],
    })


def test_shadow_auction_uses_mapped_ev_then_score_and_two_entry_cap() -> None:
    out = auction_admitted_snapshot(_candidates(), state=_state(), policy=_policy())
    accepted = out.loc[out["portfolio_accepted"], "candidate_id"].tolist()
    assert accepted == ["a", "b"]
    assert out.set_index("candidate_id").at["c", "portfolio_rejection_reason"] == "max_new_entries_per_bar_reached"
    assert out.set_index("candidate_id").at["d", "portfolio_rejection_reason"] == "ev_map_rejected"
    assert out.loc[out["portfolio_accepted"], "portfolio_initial_margin"].tolist() == [100.0, 100.0]
    assert out.loc[out["portfolio_accepted"], "portfolio_gross_notional"].tolist() == [700.0, 700.0]


def test_shadow_auction_respects_existing_symbol_and_margin_state() -> None:
    positions = [
        {"symbol": "A", "side": "long", "gross_notional": 700.0, "effective_leverage": 7.0},
        *[
            {"symbol": f"OPEN{i}", "side": "long", "gross_notional": 700.0, "effective_leverage": 7.0}
            for i in range(1, 7)
        ],
    ]
    out = auction_admitted_snapshot(_candidates(), state=_state(positions=positions), policy=_policy())
    by_id = out.set_index("candidate_id")
    assert by_id.at["a", "portfolio_rejection_reason"] == "symbol_already_open"
    assert by_id.at["b", "portfolio_accepted"]
    assert by_id.at["c", "portfolio_rejection_reason"] == "max_concurrent_positions_reached"


def test_shadow_state_must_match_decision_timestamp() -> None:
    with pytest.raises(ValueError, match="exact decision timestamp"):
        ShadowPortfolioState.from_payload({
            "schema": SCHEMA,
            "as_of_ts": "2026-08-01T01:00:00Z",
            "wallet": 1000.0,
            "open_positions": [],
        }, expected_as_of_ts=TS)


def test_zero_admission_snapshot_retains_state_provenance() -> None:
    candidates = _candidates()
    candidates["causal_21d_side_admitted_ge_50bps"] = False
    out = auction_admitted_snapshot(candidates, state=_state(), policy=_policy())
    assert not out["portfolio_accepted"].any()
    assert out["portfolio_wallet"].eq(1000.0).all()
    assert out["portfolio_open_positions_before"].eq(0).all()
    assert out["portfolio_margin_cap"].eq(800.0).all()


def test_posterior_policy_owns_admission_and_fails_closed() -> None:
    candidates = _candidates()
    # Posterior can reject a Cell-day admission, admit a Cell-day rejection,
    # and must reject a missing posterior.
    candidates["trust_posterior_expected_bps"] = [40.0, 120.0, float("nan"), 90.0]
    out = auction_admitted_snapshot(
        candidates, state=_state(), policy=_posterior_policy(),
    ).set_index("candidate_id")
    assert not out.at["a", "portfolio_accepted"]
    assert out.at["b", "portfolio_accepted"]
    assert not out.at["c", "portfolio_accepted"]
    assert out.at["d", "portfolio_accepted"]
    assert not out.at["c", "trust_posterior_admitted_ge_50bps"]


def test_a5_policy_uses_bounded_score_but_cannot_override_fixed_gate() -> None:
    candidates = _candidates()
    candidates["trust_posterior_expected_bps"] = [60.0, 70.0, 40.0, 90.0]
    candidates["a5_bounded10_expected_bps"] = [65.0, 120.0, 500.0, 110.0]
    candidates["a5_timestamp_top15"] = [True, True, True, False]
    candidates["a5_bounded10_admitted"] = [True, True, False, False]
    out = auction_admitted_snapshot(
        candidates, state=_state(), policy=_a5_policy(),
    ).set_index("candidate_id")
    assert out.at["b", "portfolio_accepted"]
    assert out.at["a", "portfolio_accepted"]
    assert not out.at["c", "portfolio_accepted"]
    assert not out.at["d", "portfolio_accepted"]
