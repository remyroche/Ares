from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.inference.prop_account_controller import (
    AccountSnapshot,
    ControllerState,
    L2Capacity,
    MarkedPosition,
    PropAccountController,
    PropAccountPolicy,
)

NOW = pd.Timestamp("2026-07-18T12:00:00Z")


def account(equity=5000.0, positions=()):
    return AccountSnapshot(NOW, equity, tuple(positions))


def candidate(**updates):
    row = {
        "timestamp": NOW,
        "symbol": "BTC/USD:USD",
        "side": "long",
        "threshold_basis_corrected_expected_ev_rank": 0.99,
        "threshold_basis_corrected_expected_ev": 0.012,
        "raw_bayesian_size_multiplier": 1.0,
        "policy_archetype": "long__long_breakout_precision",
        "entry_notional_quote": 500.0,
        "signal_price": 100.0,
        "policy_stop_price": 98.0,
        "requested_entry_leverage": 1.0,
    }
    row.update(updates)
    return row


def controller(**updates):
    policy = PropAccountPolicy(**updates)
    return PropAccountController(policy, ["BTC/USD:USD"])


def full_l2(capacity=10_000.0, weight=1.0, slippage=2.0):
    return L2Capacity(capacity, weight, slippage)


def test_policy_rejects_outside_40_to_50_percent():
    with pytest.raises(ValueError):
        controller(max_marked_notional_fraction=0.51)
    with pytest.raises(ValueError):
        controller(max_total_wallet_allocation_fraction=0.46)


def test_empty_whitelist_fails_closed():
    snap = account()
    state = ControllerState.initialise(snap)
    decision = PropAccountController(PropAccountPolicy(), []).evaluate_entry(
        candidate(), snap, state, full_l2()
    )
    assert decision.action == "reject"
    assert decision.reason == "symbol_not_whitelisted"


def test_rank_and_ev_are_stricter_than_deployed_gate():
    snap = account()
    state = ControllerState.initialise(snap)
    low_rank = controller().evaluate_entry(
        candidate(threshold_basis_corrected_expected_ev_rank=0.949),
        snap,
        state,
        full_l2(),
    )
    low_ev = controller().evaluate_entry(
        candidate(threshold_basis_corrected_expected_ev=0.0069), snap, state, full_l2()
    )
    assert low_rank.reason == "rank_below_prop_threshold"
    assert low_ev.reason == "net_ev_below_prop_threshold"


def test_marked_notional_capacity_reduces_entry():
    pos = MarkedPosition("ETH/USD:USD", "long", 1850.0, 100.0, 99.0)
    snap = account(5000.0, [pos])
    state = ControllerState.initialise(snap)
    decision = controller(base_entry_notional_fraction=0.20).evaluate_entry(
        candidate(), snap, state, full_l2()
    )
    assert decision.action == "enter"
    assert decision.approved_notional == pytest.approx(150.0)


def test_aggregate_stop_risk_reduces_entry():
    # Existing risk is $32.50; operational cap is $37.50. At 2% stop distance,
    # the remaining $5 stop budget permits $250 marked notional.
    pos = MarkedPosition("ETH/USD:USD", "long", 1083.3333333333, 100.0, 97.0)
    snap = account(5000.0, [pos])
    state = ControllerState.initialise(snap)
    decision = controller(base_entry_notional_fraction=0.20).evaluate_entry(
        candidate(), snap, state, full_l2()
    )
    assert decision.action == "enter"
    assert decision.approved_notional == pytest.approx(250.0)


def test_per_position_stop_risk_is_capped_at_020_percent_equity():
    snap = account(5000.0)
    decision = controller().evaluate_entry(
        candidate(policy_stop_price=96.0),
        snap,
        ControllerState.initialise(snap),
        full_l2(),
    )
    # $10 individual risk allowance / 4% stop distance = $250 notional.
    assert decision.action == "enter"
    assert decision.approved_notional == pytest.approx(250.0)


def test_l2_reduces_or_rejects():
    snap = account()
    state = ControllerState.initialise(snap)
    reduced = controller(base_entry_notional_fraction=0.20).evaluate_entry(
        candidate(), snap, state, full_l2(capacity=125.0)
    )
    assert reduced.action == "enter"
    assert reduced.approved_notional == pytest.approx(125.0)
    state = ControllerState.initialise(snap)
    rejected = controller().evaluate_entry(
        candidate(), snap, state, full_l2(weight=0.2)
    )
    assert rejected.reason == "l2_capacity_weight_below_min"


def test_pause_at_075_percent_and_flatten_at_125_percent():
    start = account(5000.0)
    state = ControllerState.initialise(start)
    pause = controller().account_action(account(4962.5), state)
    assert pause.action == "pause"
    assert pause.reason == "entry_drawdown_limit"

    state = ControllerState.initialise(start)
    flatten = controller().account_action(account(4937.5), state)
    assert flatten.action == "flatten"
    assert state.flatten_latched
    assert pd.Timestamp(state.cooldown_until) == NOW + pd.Timedelta(hours=24)


def test_drawdown_tiers_tighten_rank_and_capacity():
    start = account(5000.0)
    state = ControllerState.initialise(start)
    snap = account(4970.0)  # -0.6%, before the hard entry stop.
    decision = controller().evaluate_entry(
        candidate(threshold_basis_corrected_expected_ev_rank=0.975),
        snap,
        state,
        full_l2(),
    )
    assert decision.reason == "rank_below_prop_threshold"
    assert decision.effective_min_rank == pytest.approx(0.98)


def test_cooldown_and_daily_state_survive_restart(tmp_path):
    path = tmp_path / "state.json"
    start = account(5000.0)
    state = ControllerState.initialise(start)
    controller().account_action(account(4962.5), state)
    state.save(path)
    loaded = ControllerState.load(path, account(5000.0))
    assert loaded.day_start_equity == 5000.0
    assert loaded.cooldown_until == state.cooldown_until
    assert json.loads(path.read_text())["utc_day"] == "2026-07-18"


def test_duplicate_and_missing_l2_fail_closed():
    snap = account()
    state = ControllerState.initialise(snap)
    first = controller().evaluate_entry(candidate(), snap, state, full_l2())
    duplicate = controller().evaluate_entry(candidate(), snap, state, full_l2())
    assert first.action == "enter"
    assert duplicate.reason == "duplicate_signal"
    missing = controller().evaluate_entry(
        candidate(symbol="BTC/USD:USD", timestamp=NOW + pd.Timedelta(minutes=1)),
        AccountSnapshot(NOW + pd.Timedelta(minutes=1), 5000.0),
        state,
        None,
    )
    assert missing.reason == "missing_l2_capacity_check"


def test_prop_target_uses_rank_and_raw_bayesian_multiplier():
    snap = account()
    ctl = controller()
    base = ctl.requested_notional(
        candidate(
            threshold_basis_corrected_expected_ev_rank=0.97,
            raw_bayesian_size_multiplier=1.0,
        ),
        snap,
    )
    larger = ctl.requested_notional(
        candidate(
            threshold_basis_corrected_expected_ev_rank=0.97,
            raw_bayesian_size_multiplier=1.2,
        ),
        snap,
    )
    assert base > 0
    assert larger == pytest.approx(base * 1.2)


def test_rank_sizing_is_convex_and_caps_at_099():
    snap = account()
    ctl = controller()
    notionals = [
        ctl.requested_notional(
            candidate(threshold_basis_corrected_expected_ev_rank=rank), snap
        )
        for rank in (0.95, 0.96, 0.97, 0.98, 0.99, 1.0)
    ]
    increments = [b - a for a, b in zip(notionals, notionals[1:5])]
    assert increments == sorted(increments)
    assert notionals[4] == pytest.approx(notionals[5])
    audit = ctl.sizing_breakdown(
        candidate(threshold_basis_corrected_expected_ev_rank=1.0), snap
    )
    assert audit["capped_rank"] == pytest.approx(0.99)
    assert audit["rank_sizing_cap"] == pytest.approx(0.99)


def test_missing_raw_bayesian_multiplier_fails_closed():
    snap = account()
    row = candidate()
    row.pop("raw_bayesian_size_multiplier")
    decision = controller().evaluate_entry(
        row, snap, ControllerState.initialise(snap), full_l2()
    )
    assert decision.action == "reject"
    assert decision.reason == "missing_raw_bayesian_size_multiplier"


def test_sizing_audit_records_combined_bayesian_inputs():
    snap = account()
    decision = controller().evaluate_entry(
        candidate(), snap, ControllerState.initialise(snap), full_l2()
    )
    assert decision.sizing["calibrated_net_ev"] == pytest.approx(0.012)
    assert decision.sizing["bayesian_components"] == [
        "calibrated_ev",
        "posterior_uncertainty",
        "gmm_ood",
        "archetype_support",
    ]


def test_same_archetype_positions_increase_reserved_risk():
    archetype = "long__long_breakout_precision"
    positions = [
        MarkedPosition("ETH/USD:USD", "long", 500.0, 100.0, 98.0, archetype),
        MarkedPosition("SOL/USD:USD", "long", 500.0, 100.0, 98.0, archetype),
    ]
    snap = account(5000.0, positions)
    ctl = controller()
    same = ctl.evaluate_entry(
        candidate(policy_archetype=archetype),
        snap,
        ControllerState.initialise(snap),
        full_l2(),
    )
    different = ctl.evaluate_entry(
        candidate(policy_archetype="long__long_mixed_clean_path"),
        snap,
        ControllerState.initialise(snap),
        full_l2(),
    )
    # Raw existing risk is $20. Two same-archetype positions reserve $25
    # (1.25x). A third raises the group to 1.50x, consuming another $5 of
    # existing-position reserve and leaving $5 base stop risk for the candidate.
    assert same.reserved_stop_risk_before == pytest.approx(25.0)
    assert same.approved_notional == pytest.approx(250.0)
    assert different.approved_notional == pytest.approx(500.0)
    assert same.sizing["archetype_risk"]["same_archetype_positions_before"] == 2
    assert same.sizing["archetype_risk"][
        "prospective_archetype_multiplier"
    ] == pytest.approx(1.5)
    assert same.sizing["archetype_risk"][
        "existing_group_uplift_reserved"
    ] == pytest.approx(5.0)


def test_missing_candidate_archetype_fails_closed():
    snap = account()
    decision = controller().evaluate_entry(
        candidate(policy_archetype=""),
        snap,
        ControllerState.initialise(snap),
        full_l2(),
    )
    assert decision.action == "reject"
    assert decision.reason == "missing_policy_archetype_for_risk"


def test_legacy_open_positions_with_missing_archetype_are_grouped_conservatively():
    positions = [
        MarkedPosition("ETH/USD:USD", "long", 500.0, 100.0, 98.0),
        MarkedPosition("SOL/USD:USD", "short", 500.0, 100.0, 102.0),
    ]
    assert controller().reserved_stop_risk(positions) == pytest.approx(25.0)


def test_existing_symbol_cannot_be_stacked():
    pos = MarkedPosition("BTC/USD:USD", "long", 100.0, 100.0, 98.0)
    snap = account(5000.0, [pos])
    decision = controller().evaluate_entry(
        candidate(), snap, ControllerState.initialise(snap), full_l2()
    )
    assert decision.action == "reject"
    assert decision.reason == "symbol_already_open"


def test_convex_budget_is_75_percent_of_internal_headroom_and_contracts():
    policy = PropAccountPolicy(
        convex_loss_budget_enabled=True,
        max_stop_risk_fraction=0.0125,
        max_position_stop_risk_fraction=0.0025,
        stop_loss_risk_margin_multiplier=1.25,
    )
    ctl = PropAccountController(policy, ["BTC/USD:USD"])
    start = AccountSnapshot(
        NOW,
        5000.0,
        day_start_equity=5000.0,
        high_water_equity=5000.0,
    )
    state = ControllerState.initialise(start)
    full = ctl.loss_capacity_budget(start, state)
    # Internal boundary is 1.5% * $5,000 = $75; 75% is $56.25.
    assert full["reserved_risk_budget"] == pytest.approx(56.25)

    loss = AccountSnapshot(
        NOW,
        4975.0,
        day_start_equity=5000.0,
        high_water_equity=5000.0,
    )
    contracted = ctl.loss_capacity_budget(loss, state)
    expected = 0.75 * 75.0 * (50.0 / 75.0) ** 1.5
    assert contracted["reserved_risk_budget"] == pytest.approx(expected)
    assert contracted["reserved_risk_budget"] < full["reserved_risk_budget"]


def test_convex_entry_charges_full_notional_stop_with_margin():
    ctl = controller(
        convex_loss_budget_enabled=True,
        max_stop_risk_fraction=0.0125,
        max_position_stop_risk_fraction=0.0025,
        stop_loss_risk_margin_multiplier=1.25,
    )
    snap = account(5000.0)
    row = candidate(
        policy_stop_price=96.0,
        portfolio_risk_budget_share=0.25,
    )
    decision = ctl.evaluate_entry(
        row, snap, ControllerState.initialise(snap), full_l2()
    )
    # Raw stop is 4%, stressed to 5%. Position risk is capped at $12.50.
    assert decision.action == "enter"
    assert decision.approved_notional == pytest.approx(250.0)
    assert decision.sizing["raw_stop_distance_fraction"] == pytest.approx(0.04)
    assert decision.sizing["stressed_stop_distance_fraction"] == pytest.approx(0.05)


def test_diversity_allocator_caps_trade_archetype_and_side():
    ctl = controller(convex_loss_budget_enabled=True)
    rows = [
        candidate(symbol="BTC/USD:USD", side="long", policy_archetype="long_a"),
        candidate(symbol="ETH/USD:USD", side="long", policy_archetype="long_b"),
        candidate(symbol="SOL/USD:USD", side="short", policy_archetype="short_a"),
        candidate(symbol="XRP/USD:USD", side="short", policy_archetype="short_b"),
    ]
    shares = ctl.opportunity_risk_shares(rows)
    assert sum(shares) == pytest.approx(1.0)
    assert max(shares) <= 0.25 + 1e-12
    assert sum(shares[:2]) <= 0.65 + 1e-12
    assert sum(shares[2:]) <= 0.65 + 1e-12
    # When no diverse alternative exists, budget remains unallocated.
    assert ctl.opportunity_risk_shares(rows[:1]) == pytest.approx([0.25])


def test_wallet_capacity_counts_invested_margin_but_risk_uses_notional():
    existing = MarkedPosition(
        "ETH/USD:USD",
        "long",
        3500.0,
        100.0,
        100.0,
        "long_other",
        leverage=2.0,
    )
    snap = account(5000.0, [existing])
    ctl = controller(base_entry_notional_fraction=0.20)
    decision = ctl.evaluate_entry(
        candidate(
            policy_stop_price=99.9,
            requested_entry_leverage=2.0,
        ),
        snap,
        ControllerState.initialise(snap),
        full_l2(),
    )
    assert snap.marked_notional == pytest.approx(3500.0)
    assert snap.marked_invested_amount == pytest.approx(1750.0)
    # 40% wallet cap leaves $250 invested, hence $500 notional at 2x.
    assert decision.action == "enter"
    assert decision.approved_notional == pytest.approx(500.0)


def test_static_starting_balance_drawdown_does_not_trail_profits():
    snap = AccountSnapshot(
        NOW,
        4950.0,
        day_start_equity=4950.0,
        high_water_equity=5200.0,
        starting_equity=5000.0,
    )
    static_ctl = controller(
        drawdown_reference_mode="starting_balance",
        stop_entries_peak_drawdown=-0.015,
        flatten_peak_drawdown=-0.02,
        firm_max_drawdown_fraction=0.03,
    )
    static_decision = static_ctl.account_action(snap, ControllerState.initialise(snap))
    assert static_decision.action == "allow"
    assert static_decision.peak_drawdown == pytest.approx(-0.01)

    trailing_ctl = controller(drawdown_reference_mode="high_water")
    trailing_decision = trailing_ctl.account_action(
        snap, ControllerState.initialise(snap)
    )
    assert trailing_decision.action == "flatten"
    assert trailing_decision.peak_drawdown == pytest.approx(4950.0 / 5200.0 - 1.0)


def test_legacy_state_load_migrates_starting_equity(tmp_path):
    path = tmp_path / "legacy_state.json"
    path.write_text(
        json.dumps(
            {
                "utc_day": "2026-07-18",
                "day_start_equity": 5010.0,
                "high_water_equity": 5200.0,
            }
        )
    )
    snap = AccountSnapshot(NOW, 5050.0, starting_equity=5000.0)
    state = ControllerState.load(path, snap)
    assert state.starting_equity == pytest.approx(5000.0)
