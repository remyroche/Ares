import numpy as np

from extreme_price_movements.simple_policy_1m_contextual import (
    apply_robust_state,
    beta_binomial_lower_score,
    exposure_neutral_size_multiplier,
    fit_robust_state,
    geometry_scaled_params,
    normalized_atr_power,
    posterior_mixture_scale,
    support_shrink,
)


def test_normalized_atr_power_identity_and_reference_anchor():
    atr = np.array([0.005, 0.01, 0.02])
    assert np.allclose(normalized_atr_power(atr, 0.01, 1.0), atr)
    assert normalized_atr_power(np.array([0.01]), 0.01, 0.4)[0] == 0.01


def test_posterior_mixture_is_soft_not_argmax():
    p = np.array([[0.25, 0.75], [1.0, 0.0]])
    assert np.allclose(posterior_mixture_scale(p, np.array([0.8, 1.2])), [1.1, 0.8])


def test_robust_state_is_frozen_and_missing_is_neutral():
    state = fit_robust_state(np.array([1.0, 2.0, 3.0]))
    out = apply_robust_state(np.array([2.0, np.nan]), state)
    assert np.allclose(out, [0.0, 0.0])


def test_geometry_scale_preserves_capital_inside_full_stop():
    params = {
        "sl_mult": 3.0,
        "trailing_activation_mult": 2.0,
        "giveback_beta": 0.5,
        "entry_capital_ratio": 0.9,
        "transition_center": 2.0,
        "current_distance_sl_ratio": 0.95,
    }
    out = geometry_scaled_params(params, 1.3)
    assert out["entry_capital_ratio"] < 1.0
    assert out["current_distance_sl_ratio"] <= out["entry_capital_ratio"]


def test_bayesian_size_is_conservative_exposure_neutral_and_shrunk():
    score = beta_binomial_lower_score(np.array([1.0, 80.0]), np.array([2.0, 100.0]))
    assert score[1] > score[0]
    size = exposure_neutral_size_multiplier(score, reference_mean=float(score.mean()), strength=2.0)
    assert 0.85 < size.mean() < 1.15
    assert support_shrink(1.25, 0, 100.0) == 1.0
    assert support_shrink(1.25, 10_000, 100.0) > 1.2
