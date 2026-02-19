import pytest

from extreme_price_movements.production_admissibility import (
    TRADEABLE_TP_MIN,
    compute_prod_aligned_tp_params,
)


def test_tradeable_tp_min_policy_reduced_to_1p5pct():
    assert TRADEABLE_TP_MIN == pytest.approx(0.015, rel=1e-12)


def test_compute_prod_aligned_tp_params_h2_floor_math():
    atr_samples = [0.01, 0.02, 0.03, 0.04, 0.05]

    out = compute_prod_aligned_tp_params(
        atr_pct_samples=atr_samples,
        fee_pct_total=0.005,
        horizon_scaling_fn=lambda h: (h / 4.0) ** 0.5,
        worst_horizon=2,
        q=0.25,
        alpha=0.45,
        margin_mult=4.0,
        hard_min_tp=0.02,
        inflate=1.10,
    )

    # tp_min_tradeable = max(4*0.005, 0.02) = 0.02
    # s(H2)=sqrt(2/4)=~0.7071 => pre-inflate floor for tp_base >= ~0.0283
    assert out["tp_min_tradeable"] == pytest.approx(0.02, rel=1e-9)
    assert out["s_worst"] == pytest.approx((2.0 / 4.0) ** 0.5, rel=1e-9)
    assert out["tp_base_pct"] >= 0.0282842712 * 1.10 * (1 - 1e-6)
    assert out["tp_abs_lo_pct"] == pytest.approx(0.02, rel=1e-9)


def test_compute_prod_aligned_tp_params_emits_h2_h4_h8_candidates():
    atr_samples = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    out = compute_prod_aligned_tp_params(
        atr_pct_samples=atr_samples,
        fee_pct_total=0.005,
        horizon_scaling_fn=lambda h: (h / 4.0) ** 0.5,
        hard_min_tp=0.015,
        margin_mult=3.0,
        inflate=1.0,
    )

    ladder = out["tp_base_candidates"]
    assert len(ladder) > 0
    first = ladder[0]
    assert "tp_eff_targets" in first and "H2" in first["tp_eff_targets"]
    assert "tp_eff_bands" in first and "H8" in first["tp_eff_bands"]

    # H2 target is explicitly clipped to [1%, 4%].
    assert 0.01 <= float(first["tp_eff_targets"]["H2"]) <= 0.04


def test_compute_prod_aligned_tp_params_raises_on_empty_samples():
    with pytest.raises(ValueError):
        compute_prod_aligned_tp_params(
            atr_pct_samples=[],
            fee_pct_total=0.005,
            horizon_scaling_fn=lambda _: 1.0,
        )
