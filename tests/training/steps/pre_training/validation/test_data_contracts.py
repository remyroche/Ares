import pandas as pd
import pytest

from src.training.steps.pre_training.validation.data_contracts import (
    DataContractValidationError,
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)


def _make_labels() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    return pd.DataFrame(
        {
            "immediate_opportunity": [1, 0],
            "short_term_opportunity": [0, 1],
            "leverage_adjusted_score": [0.5, -0.1],
        },
        index=index,
    )


def _make_features() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    return pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [1.0, 2.0]}, index=index)


def _make_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10_000.0, 11_000.0],
        },
        index=index,
    )


def test_validate_multi_horizon_labeling_requires_dataframe():
    payload = {
        "labeled_data": _make_labels(),
        "labels": _make_labels(),
        "horizon_weights": {"t1": 1.0},
        "target_columns": ["target:t1"],
        "validation_results": {"is_valid": True, "issues": []},
        "metadata": {"downstream_ready": True},
        "smoothing_settings": {},
    }

    validated = validate_multi_horizon_labeling_result(payload, context="test")
    assert validated["labeled_data"].equals(payload["labeled_data"])

    payload["labeled_data"] = "not a dataframe"
    with pytest.raises(DataContractValidationError) as exc:
        validate_multi_horizon_labeling_result(payload, context="test.invalid")
    assert "test.invalid" in str(exc.value)


def test_validate_multi_horizon_market_batches_must_be_frames():
    payload = {
        "labeled_data": _make_labels(),
        "labels": _make_labels(),
        "horizon_weights": {"t1": 1.0},
        "target_columns": ["target:t1"],
        "validation_results": {"is_valid": True, "issues": []},
        "metadata": {},
        "smoothing_settings": {},
        "market_data": _make_market_data(),
        "market_data_batches": ["not-a-frame"],
    }

    with pytest.raises(DataContractValidationError) as exc:
        validate_multi_horizon_labeling_result(payload, context="test.market")
    assert "market_data_batches" in str(exc.value)


def test_validate_feature_artifact_requires_string_names():
    payload = {
        "features": _make_features(),
        "feature_names": ["feature_a", "feature_b"],
        "selected_features": ["feature_a"],
        "interaction_features": _make_features(),
    }

    validated = validate_feature_artifact(payload, context="test.features")
    assert list(validated["feature_names"]) == ["feature_a", "feature_b"]

    payload["feature_names"] = [1, 2]
    with pytest.raises(DataContractValidationError) as exc:
        validate_feature_artifact(payload, context="test.features.invalid")
    assert "test.features.invalid" in str(exc.value)


def test_validate_selection_artifact_enforces_strings():
    payload = {
        "final_features": ["f1", "f2"],
        "stage_1_features": ["f1"],
        "stage_2_features": ["f2"],
        "feature_counts": {"initial": 2, "final": 2},
        "stage_scores": {"final": {"cv_mean": 0.5}},
        "selection_time": 1.23,
    }

    validate_selection_artifact(payload, context="test.selection")

    payload["final_features"] = ["f1", 2]
    with pytest.raises(DataContractValidationError) as exc:
        validate_selection_artifact(payload, context="test.selection.invalid")
    assert "test.selection.invalid" in str(exc.value)
