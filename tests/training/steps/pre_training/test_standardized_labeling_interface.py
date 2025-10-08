import pandas as pd

from src.training.steps.pre_training.standardized_labeling_interface import (
    LabelingMetadata,
    StandardizedLabelingResult,
)


def _make_standardized_result(weights):
    labels = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "target_long": [0.1],
            "target_micro": [0.2],
        }
    )

    metadata = LabelingMetadata(
        source_component="unit_test",
        creation_time="2024-01-01T00:00:00",
        pipeline_ready=True,
        symbol="TEST",
        exchange="TEST",
        timeframe="1m",
        n_samples=len(labels),
        n_targets=2,
        n_horizons=len(weights),
    )

    return StandardizedLabelingResult(
        labels=labels,
        weights=weights,
        target_columns=["target_long", "target_micro"],
        quality_scores={},
        confidence_scores=pd.DataFrame(),
        eligibility_masks=pd.DataFrame(),
        metadata=metadata,
    )


def test_get_best_target_uses_micro_and_long_weights():
    result = _make_standardized_result(
        {"micro": 0.9, "small": 0.1, "medium": 0.2, "high": 0.8}
    )
    assert result.get_best_target() == "target_micro"

    result = _make_standardized_result(
        {"micro": 0.2, "small": 0.1, "medium": 0.3, "high": 0.8}
    )
    assert result.get_best_target() == "target_long"
import pytest

from src.training.steps.pre_training.standardized_labeling_interface import (
    assert_labels_sigma_scaled,
)


def test_assert_labels_sigma_scaled_single_observation_does_not_raise():
    labels = pd.DataFrame(
        {
            "target_primary": [0.5],
        }
    )

    # Should not raise even though only a single non-null observation is present.
    assert_labels_sigma_scaled(labels)


def test_assert_labels_sigma_scaled_variance_out_of_bounds_raises():
    labels = pd.DataFrame(
        {
            "target_primary": [0.0, 3.0],
        }
    )

    with pytest.raises(ValueError):
        assert_labels_sigma_scaled(labels, tolerance=0.35)
