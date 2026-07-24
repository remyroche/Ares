from __future__ import annotations

import numpy as np

from scripts.run_short_default_uncertainty_ablation import (
    _adjust_rank,
    _percentile,
    _weight_templates,
)


def test_reverse_percentile_turns_low_values_into_high_risk() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    regular = _percentile(np.array([1.0, 4.0], dtype=np.float32), reference)
    reverse = _percentile(
        np.array([1.0, 4.0], dtype=np.float32), reference, reverse=True
    )
    assert regular[0] < regular[1]
    assert reverse[0] > reverse[1]


def test_uncertainty_adjustment_only_penalizes_above_threshold() -> None:
    rank = np.array([0.95, 0.95], dtype=np.float32)
    adjusted = _adjust_rank(
        rank,
        np.array([0.50, 1.00], dtype=np.float32),
        threshold=0.75,
        alpha=0.04,
    )
    np.testing.assert_allclose(adjusted, np.array([0.95, 0.91]), atol=1e-6)


def test_weight_templates_are_normalized() -> None:
    for weights in _weight_templates().values():
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0.0)
