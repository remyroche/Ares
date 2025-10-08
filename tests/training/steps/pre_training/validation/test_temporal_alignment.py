import numpy as np
import pandas as pd
import pytest

from src.training.steps.pre_training.validation.schemas import enforce_feature_temporal_alignment


def _build_index(length: int = 5) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=length, freq="1h")


def test_enforce_temporal_alignment_passes_for_lagged_features() -> None:
    index = _build_index()
    df = pd.DataFrame({"X_feature": [np.nan, 0.1, 0.2, 0.3, 0.4]}, index=index)

    metadata = enforce_feature_temporal_alignment(
        df,
        context="tests.lagged",
        target_shifts={"target": 2},
    )

    assert metadata["X_feature"]["observed_lag"] >= 1


def test_enforce_temporal_alignment_raises_on_contemporaneous_values() -> None:
    index = _build_index()
    df = pd.DataFrame({"X_feature": [0.0, 0.1, 0.2, 0.3, 0.4]}, index=index)

    with pytest.raises(ValueError):
        enforce_feature_temporal_alignment(
            df,
            context="tests.unlagged",
            target_shifts={"target": 1},
        )


def test_enforce_temporal_alignment_raises_when_metadata_reports_no_lag() -> None:
    index = _build_index()
    df = pd.DataFrame({"X_feature": [np.nan, 0.1, 0.2, 0.3, 0.4]}, index=index)
    metadata = {"features": {"X_feature": {"max_lag": 0}}}

    with pytest.raises(ValueError):
        enforce_feature_temporal_alignment(
            df,
            context="tests.metadata",
            target_shifts={"target": 1},
            feature_metadata=metadata,
        )


def test_enforce_temporal_alignment_validates_target_shift_metadata() -> None:
    index = _build_index()
    df = pd.DataFrame({"X_feature": [np.nan, 0.1, 0.2]}, index=index[:3])

    with pytest.raises(ValueError):
        enforce_feature_temporal_alignment(
            df,
            context="tests.target_shifts",
            target_shifts={"target": 0},
        )
