import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.simple_policy_1m_constrained import (
    ACTIVATION_CURVE_BLENDED,
    ACTIVATION_CURVE_POST_ACTIVATION,
    ACTIVATION_CURVE_TOTAL_MFE,
    FAMILY_CONSTANT,
    FAMILY_EXPONENTIAL,
    FAMILY_MULTILAYER,
    FAMILY_RATIONAL,
    FAMILY_SIGMOID,
    FAMILY_SPLINE,
    FAMILY_TRAILING_ONLY,
    REASON_CAPITAL,
    REASON_TRAILING,
    _activation_curve_u,
    _excess_ratio,
    constrained_params_to_vector,
    simulate_constrained_1m_paths,
)
from scripts.download_policy_execution_1m import _merge_windows
from scripts.run_simple_policy_1m_constrained_search import _bounded_product


def _params(**updates):
    base = {
        "sl_mult": 3.0,
        "trailing_activation_mult": 1.5,
        "trailing_activation_cap_pct": 0.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "entry_capital_ratio": 2.0 / 3.0,
        "terminal_excess_ratio": 0.25,
        "transition_center": 2.0,
        "transition_shape": 1.2,
        "mixture_logit_1": 0.1,
        "mixture_logit_2": -0.2,
        "spline_retains": [0.85, 0.70, 0.55, 0.40, 0.25],
    }
    base.update(updates)
    return base


def _run(side, high, low, close, *, family=FAMILY_RATIONAL, params=None):
    bars = len(high)
    return simulate_constrained_1m_paths(
        np.array([0], dtype=np.int64),
        np.array([100.0], dtype=np.float32),
        np.asarray(high, dtype=np.float32).reshape(1, bars),
        np.asarray(low, dtype=np.float32).reshape(1, bars),
        np.asarray(close, dtype=np.float32).reshape(1, bars),
        np.array([side], dtype=np.float64),
        np.array([0.01], dtype=np.float64),
        np.zeros(1),
        np.zeros(1),
        constrained_params_to_vector(params or _params()),
        family,
        0.005,
        15.0,
        0.05,
        75.0,
        0.02,
    )


def test_capital_is_active_at_entry_and_replaces_full_stop_collision():
    result = _run(1.0, [100.2], [97.5], [98.0])
    assert result[4][0] == REASON_CAPITAL
    assert result[7][0] == 0
    assert result[10][0]
    assert result[11][0]
    assert result[1][0] > 97.0  # filled from the tighter capital stop, not full SL.


def test_long_short_mirror_has_identical_timing_reason_and_return():
    long = _run(1.0, [100.2, 101.5, 101.0], [99.8, 100.4, 98.8], [100.0, 101.0, 99.0])
    short = _run(-1.0, [100.2, 99.6, 101.2], [99.8, 98.5, 99.0], [100.0, 99.0, 101.0])
    assert long[0][0] == short[0][0]
    assert long[4][0] == short[4][0]
    assert np.isclose(long[2][0], short[2][0], atol=5e-4)


def test_all_excess_curves_are_positive_and_nonincreasing():
    grid = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    vector = constrained_params_to_vector(_params())
    for family in (FAMILY_CONSTANT, FAMILY_MULTILAYER, FAMILY_SIGMOID, FAMILY_EXPONENTIAL, FAMILY_RATIONAL, FAMILY_SPLINE):
        ratios = np.array([_excess_ratio(family, float(u), vector) for u in grid])
        assert np.isclose(ratios[0], 1.0)
        assert np.all(ratios > 0.0)
        assert np.all(np.diff(ratios) <= 1e-10)


def test_capital_precedes_trailing_handover():
    high = np.linspace(100.1, 104.0, 12)
    low = high - 0.2
    close = high - 0.1
    result = _run(1.0, high, low, close, params=_params(trailing_activation_mult=1.0))
    assert result[7][0] == 0
    assert result[8][0] > result[7][0]
    assert result[11][0]


def test_previous_close_buffer_does_not_use_current_close():
    params = _params(current_distance_sl_ratio=0.9, trailing_activation_mult=10.0)
    a = _run(1.0, [100.2, 100.2], [99.9, 99.9], [100.0, 95.0], params=params)
    b = _run(1.0, [100.2, 100.2], [99.9, 99.9], [100.0, 105.0], params=params)
    assert a[0][0] == b[0][0]
    assert a[4][0] == b[4][0]


def test_side_pooling_transform_is_bounded_and_reciprocal_before_clipping():
    long = _bounded_product(2.0, 0.15, 1.0, 0.1, 10.0)
    short = _bounded_product(2.0, 0.15, -1.0, 0.1, 10.0)
    assert np.isclose(long * short, 4.0)


def test_download_warmup_extends_only_the_window_start():
    ts = pd.Timestamp("2026-05-01 12:00", tz="UTC")
    assert _merge_windows([ts], 1440, 2880) == [
        (pd.Timestamp("2026-04-29 12:00", tz="UTC"), pd.Timestamp("2026-05-02 12:00", tz="UTC"))
    ]


def test_activation_curve_coordinates_are_distinct_and_ordered():
    total = _activation_curve_u(3.0, 1.5, ACTIVATION_CURVE_TOTAL_MFE, 0.0)
    shifted = _activation_curve_u(3.0, 1.5, ACTIVATION_CURVE_POST_ACTIVATION, 1.0)
    blended = _activation_curve_u(3.0, 1.5, ACTIVATION_CURVE_BLENDED, 0.4)
    assert total == 3.0
    assert shifted == 1.5
    assert shifted < blended < total


def test_early_trailing_layer_arms_from_prior_bar_mfe_only():
    result = _run(
        1.0,
        [100.6, 100.7],
        [99.0, 99.3],
        [100.5, 99.5],
        family=FAMILY_TRAILING_ONLY,
        params=_params(
            trailing_layer_count=1,
            trailing_activation_mult=0.5,
            giveback_beta=2.0,
            trailing_power=1.0,
            trailing_squash_divisor=100.0,
        ),
    )
    assert result[0][0] == 1
    assert result[4][0] == REASON_TRAILING
    assert result[12][0, 0] == 1
    assert result[14][0] == 0


@pytest.mark.parametrize(
    ("layer_count", "expected_first_bars"),
    [
        (1, [1, -1, -1]),
        (2, [1, 2, -1]),
        (3, [1, 2, 3]),
    ],
)
def test_one_to_three_total_mfe_layers_arm_in_threshold_order(
    layer_count, expected_first_bars
):
    result = _run(
        1.0,
        [100.6, 101.1, 101.6, 101.7],
        [100.5, 101.0, 101.5, 101.6],
        [100.55, 101.05, 101.55, 101.65],
        family=FAMILY_TRAILING_ONLY,
        params=_params(
            trailing_layer_count=layer_count,
            trailing_activation_mult=0.5,
            trailing_activation_mult_2=1.0,
            trailing_activation_mult_3=1.5,
            giveback_beta=3.0,
            giveback_beta_2=3.0,
            giveback_beta_3=3.0,
            trailing_power=1.0,
            trailing_squash_divisor=100.0,
        ),
    )
    assert result[12][0].tolist() == expected_first_bars


def test_early_protective_layer_can_exit_before_main_trailing_activates():
    result = _run(
        1.0,
        [100.6, 100.7],
        [99.8, 99.3],
        [100.5, 99.5],
        family=FAMILY_TRAILING_ONLY,
        params=_params(
            trailing_layer_count=2,
            trailing_activation_mult=0.5,
            trailing_activation_mult_2=1.5,
            giveback_beta=2.0,
            giveback_beta_2=0.5,
            trailing_power=1.0,
            trailing_squash_divisor=100.0,
        ),
    )
    assert result[0][0] == 1
    assert result[4][0] == REASON_TRAILING
    assert result[12][0].tolist() == [1, -1, -1]
    assert result[14][0] == 0
    assert result[2][0] < 0.0


def test_layered_trailing_stop_never_loosens_when_new_candidate_is_wider():
    result = _run(
        1.0,
        [100.6, 101.5, 101.6],
        [100.5, 100.0, 99.2],
        [100.55, 101.4, 99.3],
        family=FAMILY_TRAILING_ONLY,
        params=_params(
            trailing_layer_count=1,
            trailing_activation_mult=0.5,
            giveback_beta=2.0,
            trailing_power=1.0,
            trailing_squash_divisor=100.0,
        ),
    )
    # At bar 1 the 0.6-ATR MFE establishes a roughly 99.4 stop.  The larger
    # MFE visible to bar 2 proposes a wider stop; the executable stop must
    # retain the prior, tighter level and therefore be hit on bar 2.
    assert result[0][0] == 2
    assert result[4][0] == REASON_TRAILING
    assert result[14][0] == 0


def test_three_ordered_layers_report_tightest_exit_layer():
    result = _run(
        1.0,
        [100.6, 101.1, 101.6, 101.7],
        [99.8, 100.0, 100.5, 101.0],
        [100.5, 101.0, 101.5, 101.1],
        family=FAMILY_TRAILING_ONLY,
        params=_params(
            trailing_layer_count=3,
            trailing_activation_mult=0.5,
            trailing_activation_mult_2=1.0,
            trailing_activation_mult_3=1.5,
            giveback_beta=1.5,
            giveback_beta_2=0.8,
            giveback_beta_3=0.2,
            trailing_power=1.0,
            trailing_squash_divisor=100.0,
        ),
    )
    assert result[12][0].tolist() == [1, 2, 3]
    assert result[14][0] == 2
