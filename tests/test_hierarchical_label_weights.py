from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.hierarchical_label_weights import (
    TARGET_EXPONENT_GRID,
    WEIGHT_RANGE_RATIO_MAX,
    WEIGHT_RANGE_RATIO_MIN,
    TargetStrengthWeightSpec,
    build_target_strength_weights,
)


def test_target_strength_weights_are_bounded_centered_and_monotone() -> None:
    target = np.linspace(0.0, 1.0, 1000)
    weights, diagnostics = build_target_strength_weights(
        target,
        timestamps=np.arange(len(target)),
        archetypes=np.repeat("a", len(target)),
        spec=TargetStrengthWeightSpec(exponent=1.5),
    )
    assert np.isclose(float(weights.mean()), 1.0, atol=1e-6)
    assert float(weights.min()) >= 0.50
    assert float(weights.max()) <= 2.00
    assert np.all(np.diff(weights) >= -1e-7)
    assert weights[-1] > weights[0]
    assert diagnostics["effective_sample_fraction"] >= 0.60


def test_every_requested_exponent_is_supported() -> None:
    target = np.linspace(0.0, 1.0, 100)
    for exponent in TARGET_EXPONENT_GRID:
        weights, _ = build_target_strength_weights(
            target,
            timestamps=np.arange(len(target)),
            archetypes=np.repeat(["a", "b"], len(target) // 2),
            spec=TargetStrengthWeightSpec(exponent=exponent),
        )
        assert np.isclose(float(weights.mean()), 1.0, atol=1e-6)


def test_every_requested_weight_range_ratio_is_supported() -> None:
    target = np.linspace(0.0, 1.0, 1000)
    for ratio in np.linspace(WEIGHT_RANGE_RATIO_MIN, WEIGHT_RANGE_RATIO_MAX, 7):
        weights, diagnostics = build_target_strength_weights(
            target,
            timestamps=np.arange(len(target)),
            archetypes=np.repeat("a", len(target)),
            spec=TargetStrengthWeightSpec(
                exponent=1.5,
                weight_range_ratio=ratio,
            ),
        )
        lower = 1.0 / np.sqrt(ratio)
        upper = np.sqrt(ratio)
        assert np.isclose(float(weights.mean()), 1.0, atol=1e-6)
        assert float(weights.min()) >= lower - 1e-7
        assert float(weights.max()) <= upper + 1e-7
        assert diagnostics["spec"]["weight_range_ratio"] == ratio
        assert np.isclose(diagnostics["derived_weight_min"], lower)
        assert np.isclose(diagnostics["derived_weight_max"], upper)


def test_zero_target_receives_floor_and_one_target_receives_ceiling() -> None:
    target = np.concatenate([np.zeros(500), np.ones(500)])
    weights, _ = build_target_strength_weights(
        target,
        timestamps=np.arange(len(target)),
        archetypes=np.repeat("a", len(target)),
        spec=TargetStrengthWeightSpec(exponent=2.0),
    )
    assert np.allclose(weights[:500], 0.50)
    assert np.allclose(weights[500:], 1.50)


def test_timestamp_and_archetype_rebalancing_remain_tempered() -> None:
    target = np.full(12, 0.8)
    timestamps = pd.to_datetime(
        ["2026-01-01"] * 8 + ["2026-01-02"] * 4, utc=True
    )
    archetypes = np.array(["large"] * 10 + ["small"] * 2)
    weights, diagnostics = build_target_strength_weights(
        target,
        timestamps=timestamps,
        archetypes=archetypes,
        spec=TargetStrengthWeightSpec(exponent=1.0),
    )
    assert np.isclose(float(weights.mean()), 1.0, atol=1e-6)
    assert float(weights[archetypes == "small"].mean()) > float(
        weights[archetypes == "large"].mean()
    )
    assert float(weights.max()) / float(weights.min()) <= 4.0
    assert diagnostics["archetype_count"] == 2
