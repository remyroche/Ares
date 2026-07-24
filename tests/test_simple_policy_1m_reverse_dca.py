import numpy as np
import pytest

from extreme_price_movements.simple_policy_1m_constrained import (
    constrained_params_to_vector,
)
from extreme_price_movements.simple_policy_1m_reverse_dca import (
    EXIT_ANCHOR_INITIAL,
    SPACING_ABSOLUTE_FRACTION,
    SPACING_ATR_MULTIPLE,
    simulate_reverse_dca_1m_paths,
)


def _simulate(*, x: int, y: float, spacing_mode: int, fee: float = 0.0):
    prices = np.asarray([[100.0, 102.0, 103.0]], dtype=np.float64)
    return simulate_reverse_dca_1m_paths(
        np.asarray([0], dtype=np.int64),
        np.asarray([100.0], dtype=np.float64),
        prices,
        prices,
        prices,
        np.asarray([1.0], dtype=np.float64),
        np.asarray([0.01], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        constrained_params_to_vector(
            {
                "sl_mult": 20.0,
                "trailing_activation_mult": 100.0,
                "trailing_power": 1.0,
                "trailing_squash_divisor": 2.0,
                "giveback_beta": 0.5,
            }
        ),
        fee,
        0.0,
        0.0,
        0.0,
        x,
        y,
        spacing_mode,
        EXIT_ANCHOR_INITIAL,
        True,
    )


def test_unfilled_tranches_leave_unused_target_capital_neutral():
    result = _simulate(
        x=2,
        y=0.10,
        spacing_mode=SPACING_ABSOLUTE_FRACTION,
    )
    gross, filled, additions = result[2], result[7], result[8]
    assert additions[0] == 0
    assert filled[0] == pytest.approx(0.5)
    assert gross[0] == pytest.approx(0.5 * (103.0 / 100.0 - 1.0))


def test_absolute_and_atr_spacing_are_equivalent_at_matching_distance():
    absolute = _simulate(
        x=2,
        y=0.02,
        spacing_mode=SPACING_ABSOLUTE_FRACTION,
    )
    atr = _simulate(
        x=2,
        y=2.0,
        spacing_mode=SPACING_ATR_MULTIPLE,
    )
    for index in (2, 3, 7, 8, 9):
        np.testing.assert_allclose(absolute[index], atr[index], rtol=0.0, atol=1e-12)
    assert absolute[7][0] == pytest.approx(1.0)
    assert absolute[8][0] == 1


def test_each_filled_tranche_pays_round_trip_fee_once():
    fee = 0.005
    result = _simulate(
        x=2,
        y=0.01,
        spacing_mode=SPACING_ABSOLUTE_FRACTION,
        fee=fee,
    )
    gross, net, filled = result[2], result[3], result[7]
    assert filled[0] == pytest.approx(1.0)
    tranche_returns = np.asarray([103.0 / 100.0 - 1.0, 103.0 / 101.0 - 1.0])
    expected_gross = 0.5 * tranche_returns.sum()
    expected_net = 0.5 * np.sum(
        tranche_returns - fee - fee * (1.0 + tranche_returns)
    )
    assert gross[0] == pytest.approx(expected_gross)
    assert net[0] == pytest.approx(expected_net)
    assert gross[0] - net[0] == pytest.approx(
        0.5 * np.sum(fee + fee * (1.0 + tranche_returns))
    )
