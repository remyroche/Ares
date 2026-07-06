import pytest

from extreme_price_movements.inference.portfolio_policy import (
    PortfolioPolicyConfig,
    compute_rank_based_position_size,
)
from extreme_price_movements.inference.trade_executor import (
    _perp_liquidation_guard_for_stop,
)


def test_perp_liquidation_guard_caps_dydx_like_stop_before_order():
    guard = _perp_liquidation_guard_for_stop(
        side="short",
        entry_price=0.1678,
        stop_price=0.1786,
        requested_leverage=10.0,
        config={
            "market_mode": "perps",
            "perp_maintenance_margin_pct": 0.05,
            "perp_liquidation_fee_buffer_pct": 0.005,
            "perp_liquidation_safety_buffer_pct": 0.01,
        },
    )

    assert guard["perp_liquidation_guard_reject"] is False
    assert guard["perp_liquidation_leverage_capped"] is True
    assert guard["perp_liquidation_guarded_leverage"] < 10.0
    assert guard["perp_liquidation_guarded_leverage"] == pytest.approx(7.73)
    assert guard["perp_liquidation_distance_at_requested_pct"] == pytest.approx(0.045)
    assert guard["perp_liquidation_stop_distance_pct"] == pytest.approx(
        0.1786 / 0.1678 - 1.0
    )
    assert (
        guard["perp_liquidation_distance_at_guarded_pct"] + 0.002
        >= guard["perp_liquidation_required_distance_pct"]
    )


def test_perp_liquidation_guard_leaves_tight_stop_uncapped():
    guard = _perp_liquidation_guard_for_stop(
        side="long",
        entry_price=100.0,
        stop_price=99.0,
        requested_leverage=10.0,
        config={
            "market_mode": "perps",
            "perp_maintenance_margin_pct": 0.05,
            "perp_liquidation_fee_buffer_pct": 0.005,
            "perp_liquidation_safety_buffer_pct": 0.01,
        },
    )

    assert guard["perp_liquidation_guard_reject"] is False
    assert guard["perp_liquidation_leverage_capped"] is False
    assert guard["perp_liquidation_guarded_leverage"] == 10.0
    assert guard["perp_liquidation_stop_distance_pct"] == pytest.approx(0.01)


def test_perp_liquidation_guard_uses_entry_relative_long_stop_distance():
    guard = _perp_liquidation_guard_for_stop(
        side="long",
        entry_price=0.003891,
        stop_price=0.003173,
        requested_leverage=10.0,
        config={
            "market_mode": "perps",
            "perp_maintenance_margin_pct": 0.05,
            "perp_liquidation_fee_buffer_pct": 0.005,
            "perp_liquidation_safety_buffer_pct": 0.01,
        },
    )

    expected_stop_distance = (0.003891 - 0.003173) / 0.003891
    expected_safe_max = (
        int((1.0 / (expected_stop_distance + 0.01 + 0.05 + 0.005)) * 100.0) / 100.0
    )
    assert guard["perp_liquidation_guard_reject"] is False
    assert guard["perp_liquidation_leverage_capped"] is True
    assert guard["perp_liquidation_stop_distance_pct"] == pytest.approx(
        expected_stop_distance
    )
    assert guard["perp_liquidation_guarded_leverage"] == pytest.approx(
        expected_safe_max
    )
    assert guard["perp_liquidation_guarded_leverage"] == pytest.approx(4.0)


def test_perp_sizing_uses_full_stop_distance_not_raw_barrier():
    policy = PortfolioPolicyConfig(perp_default_leverage=10.0)
    sizing = compute_rank_based_position_size(
        wallet_value=100.0,
        open_notional=0.0,
        adjusted_rank_score=0.99,
        final_threshold=0.70,
        policy=policy,
        liquidity_capacity_weight=1.0,
        live_test_mode=False,
        market_mode="perps",
        available_wallet_value=100.0,
        stop_loss_pct=0.06436,
        rank_number=1,
        rank_x=5,
        orderbook_capacity_quote=1_000.0,
    )

    expected_legacy_cap = 100.0 / (1.5 * 6.436)
    expected_liquidation_cap = 1.0 / (0.06436 + 0.01 + 0.05 + 0.005)
    assert sizing["perp_legacy_risk_cap_leverage"] == pytest.approx(expected_legacy_cap)
    assert sizing["perp_liquidation_risk_cap_leverage"] == pytest.approx(
        expected_liquidation_cap
    )
    assert sizing["perp_risk_cap_leverage"] == pytest.approx(expected_liquidation_cap)
    assert sizing["perp_effective_leverage"] < 10.0
