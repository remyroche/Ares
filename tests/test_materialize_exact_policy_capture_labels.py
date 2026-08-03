from __future__ import annotations

import numpy as np

from scripts.materialize_exact_policy_capture_labels import compute_capture_labels


def test_capture_labels_respect_side_and_exact_exit() -> None:
    highs = np.array(
        [
            [101.0, 104.0, 110.0, 150.0],
            [101.0, 100.0, 99.0, 150.0],
        ]
    )
    lows = np.array(
        [
            [100.0, 100.0, 102.0, 20.0],
            [100.0, 98.0, 95.0, 50.0],
        ]
    )
    closes = np.array(
        [
            [100.5, 103.0, 102.0, 140.0],
            [99.5, 98.5, 98.0, 60.0],
        ]
    )
    labels = compute_capture_labels(
        highs,
        lows,
        closes,
        entry=np.array([100.0, 100.0]),
        side=np.array([1.0, -1.0]),
        exit_bar=np.array([2.0, 2.0]),
        gross=np.array([0.02, 0.01]),
        cost=np.array([0.01, 0.01]),
        atr_fraction=np.array([0.02, 0.02]),
    )
    np.testing.assert_allclose(labels["pre_exit_mfe_return"], [0.10, 0.05])
    np.testing.assert_allclose(labels["pre_exit_mae_return"], [0.0, 0.01])
    assert labels["policy_exit_bar_1m"].tolist() == [2, 2]
    assert labels["favorable_before_adverse_at_cost"].tolist() == [True, False]
    assert labels["adverse_before_favorable_at_cost"].tolist() == [False, True]
    assert labels["exact_net_positive"].tolist() == [True, False]


def test_same_minute_barrier_order_is_explicitly_ambiguous() -> None:
    labels = compute_capture_labels(
        np.array([[102.0, 103.0]]),
        np.array([[98.0, 99.0]]),
        np.array([[100.0, 101.0]]),
        entry=np.array([100.0]),
        side=np.array([1.0]),
        exit_bar=np.array([1.0]),
        gross=np.array([0.0]),
        cost=np.array([0.01]),
        atr_fraction=np.array([0.02]),
    )
    assert bool(labels.loc[0, "cost_barriers_same_minute"])
    assert not bool(labels.loc[0, "favorable_before_adverse_at_cost"])
    assert not bool(labels.loc[0, "adverse_before_favorable_at_cost"])
