from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_canonical_global_book_component_hurdle_ablation import (
    _feature_arms,
    _fit_models,
)


def test_feature_arms_are_explicit_and_disjoint() -> None:
    arms = _feature_arms(
        ("context__global_a", "context__global_b"),
        ("context__global_common_ev_band_ordinal", "context__band_a"),
    )
    assert arms["global_only"] == (
        "context__global_a",
        "context__global_b",
    )
    assert arms["band_only"] == ("context__band_a",)
    assert arms["combined"] == (
        "context__global_a",
        "context__global_b",
        "context__band_a",
    )


def test_insufficient_support_falls_back_exactly_to_zero() -> None:
    train = pd.DataFrame(
        {"context__x": [0.0, 1.0], "target_delta": [1.0, -1.0]}
    )
    evaluation = pd.DataFrame({"context__x": [2.0, 3.0, 4.0]})
    predictions, status = _fit_models(
        train,
        evaluation,
        features=["context__x"],
        min_train_rows=3,
        min_conditional_rows=2,
        seed=1,
        threads=1,
    )
    assert np.array_equal(
        predictions["hurdle_signed_mean"], np.zeros(3)
    )
    assert np.array_equal(
        predictions["hurdle_sign_magnitude"], np.zeros(3)
    )
    assert status["occurrence_status"] == "zero_fallback"
