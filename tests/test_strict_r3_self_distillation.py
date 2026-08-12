from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_self_distillation import (
    DistillationWeightSpec,
    build_distillation_weights,
    initial_screen_specs,
)


def test_initial_screen_has_predeclared_d0_to_d4() -> None:
    assert [spec.name for spec in initial_screen_specs()] == ["D0", "D1", "D2", "D3", "D4"]


def test_missing_teacher_keeps_existing_weight_semantics() -> None:
    frame = pd.DataFrame({"r3_class": [0, 1, 2], "teacher": [np.nan, 0.5, 0.9]})
    weights, audit = build_distillation_weights(
        frame, teacher_rank_column="teacher", layer="base",
        spec=DistillationWeightSpec("D1", use_score_weight=True),
    )
    assert audit["teacher_covered_rows"] == 2
    assert weights[0] > weights[1]
    assert np.isclose(weights.mean(), 1.0)


def test_base_tail_boosts_only_declared_r3_classes() -> None:
    frame = pd.DataFrame(
        {"r3_class": [2, 2, 1, 0, 0], "teacher": [0.5, 0.2, 0.9, 0.9, 0.6]}
    )
    weights, _ = build_distillation_weights(
        frame, teacher_rank_column="teacher", layer="base",
        spec=DistillationWeightSpec(
            "D4", positive_top_fraction=0.60, positive_boost=1.5,
            negative_top_fraction=0.20, negative_boost=1.5,
        ),
    )
    assert weights[0] > weights[1]
    assert weights[3] > weights[4]
    assert weights[2] < weights[3]


def test_residual_boundaries_match_economic_contract() -> None:
    frame = pd.DataFrame(
        {"policy_residual_bps": [101.0, 100.0, -149.9, -150.0], "teacher": [0.9] * 4}
    )
    weights, audit = build_distillation_weights(
        frame, teacher_rank_column="teacher", layer="residual",
        spec=DistillationWeightSpec(
            "tails", positive_top_fraction=0.6, positive_boost=2.0,
            negative_top_fraction=0.2, negative_boost=2.0,
        ),
    )
    assert audit["class_rows"] == {"adverse": 1, "weak": 2, "clear": 1}
    assert weights[0] > weights[1]
    assert weights[3] > weights[2]


def test_weight_projection_obeys_cap_and_mean_one() -> None:
    frame = pd.DataFrame({"r3_class": [2] * 100, "teacher": np.linspace(0, 1, 100)})
    weights, audit = build_distillation_weights(
        frame, teacher_rank_column="teacher", layer="base",
        spec=DistillationWeightSpec(
            "extreme", use_score_weight=True, score_power=3.0,
            positive_top_fraction=0.05, positive_boost=100.0,
        ),
    )
    assert weights.min() >= 0.25
    assert weights.max() <= 4.0
    assert np.isclose(weights.mean(), 1.0, atol=1e-6)
    assert audit["effective_sample_ratio"] > 0.0
