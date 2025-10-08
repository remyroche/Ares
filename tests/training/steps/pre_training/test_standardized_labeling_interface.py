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
