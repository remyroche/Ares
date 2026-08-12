from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.ordinal_residual_meta import (
    classifier_diagnostics,
    cumulative_to_simplex,
    fit_residual_class_map,
    fit_soft_binary_residual_scale,
    ordinal_labels,
    policy_training_mask,
    reconstruct_expected_residual,
    sample_weights,
    soft_binary_residual_labels,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(8)],
        "side_name": ["long"] * 4 + ["short"] * 4,
        "net_bps": [-160, -70, 30, 180, -220, -20, 80, 260],
        "prequential_base_expected_net_bps": [0.] * 8,
        "base_side_rank": [.99, .80, .69, .20, .98, .75, .50, .10],
    })


def test_ordinal_simplex_is_monotonic_and_side_map_is_training_only() -> None:
    frame = _frame()
    labels = ordinal_labels(frame.net_bps, 100.)
    np.testing.assert_array_equal(labels, [0, 1, 1, 2, 0, 1, 1, 2])
    probability = cumulative_to_simplex([.20, .80], [.90, .30])
    np.testing.assert_allclose(probability.sum(axis=1), 1.)
    np.testing.assert_allclose(probability[0], [.80, 0., .20])
    mapping = fit_residual_class_map(frame, threshold_bps=100., shrinkage_support=1.)
    score = reconstruct_expected_residual(np.eye(3)[[0, 2]], ["long", "short"], mapping)
    assert score[0] < 0 < score[1]
    assert mapping.side_class_support["long"] == (1, 2, 1)


def test_policy_population_and_weights_are_deterministic_and_bounded() -> None:
    frame = _frame()
    first = policy_training_mask(frame, top_fraction=.30, lower_fraction=.25)
    second = policy_training_mask(frame.sample(frac=1., random_state=1).sort_index(), top_fraction=.30, lower_fraction=.25)
    np.testing.assert_array_equal(first, second)
    labels = ordinal_labels(frame.net_bps, 100.)
    weight = sample_weights(frame, labels, residual=frame.net_bps)
    assert np.isfinite(weight).all()
    assert (weight >= .25).all() and (weight <= 4.).all()


def test_classifier_diagnostics_include_ordinal_and_class_metrics() -> None:
    y = np.asarray([0, 1, 2])
    p = np.eye(3) * .8 + .2 / 3.
    result = classifier_diagnostics(y, p)
    assert set(result) >= {"rps", "log_loss", "brier_multiclass", "recall_c0", "recall_c1", "recall_c2"}
    assert result["rps"] >= 0.


def test_soft_binary_residual_labels_are_zero_centred_and_tail_clipped() -> None:
    lower, upper = fit_soft_binary_residual_scale(
        [-300, -100, -20, 0, 30, 80, 400], lower_percentile=5., upper_percentile=95.
    )
    y = soft_binary_residual_labels(
        [-1000., lower, 0., upper, 1000.], lower_bps=lower, upper_bps=upper
    )
    assert lower < 0 < upper
    np.testing.assert_allclose(y, [0., 0., .5, 1., 1.])
    fixed = fit_soft_binary_residual_scale([-1., 0., 1.], extrema_bps=75.)
    assert fixed == (-75., 75.)
