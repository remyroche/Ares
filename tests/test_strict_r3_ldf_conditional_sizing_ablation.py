"""Focused invariants for causal train-bin conditional LDF sizing."""

from __future__ import annotations

import numpy as np

from scripts.run_strict_r3_ldf_conditional_sizing_ablation import (
    _conditional_multiplier,
    _relative_multiplier,
)


def test_conditional_bins_are_fit_only_from_training_scores() -> None:
    train_score = np.array([0.0, 0.1, 0.2, 0.7, 0.8, 0.9] * 20)
    train_quality = np.linspace(-2.0, 2.0, len(train_score))
    held_quality = np.array([-1.0, 0.0, 1.0])
    first = _conditional_multiplier(
        train_score=train_score, train_quality=train_quality,
        held_score=np.array([0.21, 0.50, 0.79]), held_quality=held_quality,
        bins=3, floor=0.25, cap=1.75,
    )
    # Arbitrarily mutating held scores does not alter the train bin boundaries
    # or the values assigned to an unchanged held score.
    second = _conditional_multiplier(
        train_score=train_score, train_quality=train_quality,
        held_score=np.array([0.21, -999.0, 999.0]), held_quality=held_quality,
        bins=3, floor=0.25, cap=1.75,
    )
    assert first[0] == second[0]


def test_relative_multiplier_is_neutral_when_local_and_global_agree() -> None:
    base = np.array([1.1, 1.3, 1.5])
    result = _relative_multiplier(
        base, np.array([1.2, 1.2, 1.2]), np.array([1.2, 1.2, 1.2]),
        alpha=0.5, floor=0.25, cap=1.75,
    )
    np.testing.assert_allclose(result, base)
