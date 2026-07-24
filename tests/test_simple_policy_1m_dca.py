from __future__ import annotations

import numpy as np

from extreme_price_movements.simple_policy_1m_constrained import (
    REASON_TIMEOUT,
    REASON_TRAILING,
)
from extreme_price_movements.simple_policy_1m_dca import apply_dca_to_frozen_exits


def _run(
    *, low: list[float], exit_bar: int, reason: int, x: int, y: float,
    literal: bool = False, dca_first: bool = False,
):
    high = np.asarray([[101.0] * len(low)], dtype=np.float64)
    return apply_dca_to_frozen_exits(
        np.asarray([0], dtype=np.int64),
        np.asarray([100.0]),
        high,
        np.asarray([low], dtype=np.float64),
        np.asarray([1.0]),
        np.asarray([0.0]),
        np.asarray([exit_bar], dtype=np.int32),
        np.asarray([110.0]),
        np.asarray([reason], dtype=np.int8),
        0.005,
        x,
        y,
        literal,
        dca_first,
    )


def test_x1_reproduces_single_entry_return() -> None:
    gross, net, filled, additions, *_ = _run(
        low=[99.0, 99.0], exit_bar=1, reason=REASON_TRAILING, x=1, y=0.0
    )
    expected_gross = 110.0 / 100.0 - 1.0
    expected_net = expected_gross - 0.005 - 0.005 * (1.0 + expected_gross)
    np.testing.assert_allclose(gross, [expected_gross], atol=1e-14)
    np.testing.assert_allclose(net, [expected_net], atol=1e-14)
    np.testing.assert_allclose(filled, [1.0])
    np.testing.assert_array_equal(additions, [0])


def test_exposure_neutral_dca_fills_before_exit_bar() -> None:
    gross, _, filled, additions, average, level, *_ = _run(
        low=[99.0, 94.0, 90.0], exit_bar=2, reason=REASON_TRAILING, x=2, y=0.05
    )
    expected = 0.5 * (110.0 / 100.0 - 1.0) + 0.5 * (110.0 / 95.0 - 1.0)
    np.testing.assert_allclose(gross, [expected])
    np.testing.assert_allclose(filled, [1.0])
    np.testing.assert_array_equal(additions, [1])
    np.testing.assert_allclose(average, [97.5])
    np.testing.assert_allclose(level, [0.05])


def test_non_timeout_exit_wins_same_bar_collision() -> None:
    _, _, filled, additions, *_ = _run(
        low=[99.0, 94.0], exit_bar=1, reason=REASON_TRAILING, x=2, y=0.05
    )
    np.testing.assert_allclose(filled, [0.5])
    np.testing.assert_array_equal(additions, [0])


def test_dca_first_bound_fills_on_exit_bar_collision() -> None:
    _, _, filled, additions, *_ = _run(
        low=[99.0, 94.0], exit_bar=1, reason=REASON_TRAILING, x=2, y=0.05,
        dca_first=True,
    )
    np.testing.assert_allclose(filled, [1.0])
    np.testing.assert_array_equal(additions, [1])


def test_timeout_allows_final_bar_fill_before_close() -> None:
    _, _, filled, additions, *_ = _run(
        low=[99.0, 94.0], exit_bar=1, reason=REASON_TIMEOUT, x=2, y=0.05
    )
    np.testing.assert_allclose(filled, [1.0])
    np.testing.assert_array_equal(additions, [1])


def test_literal_mode_can_exceed_original_target() -> None:
    _, _, filled, additions, *_ = _run(
        low=[89.0, 89.0], exit_bar=1, reason=REASON_TRAILING,
        x=2, y=0.05, literal=True,
    )
    np.testing.assert_allclose(filled, [1.5])
    np.testing.assert_array_equal(additions, [2])
