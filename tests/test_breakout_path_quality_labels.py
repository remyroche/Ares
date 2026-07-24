from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.breakout_path_quality_labels import (
    fit_breakout_path_quality_thresholds,
    fit_severe_retention_threshold,
    materialize_breakout_path_quality_labels,
    materialize_severe_retention_failure,
)


def _outcomes(rows: int = 200) -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "breakout_retention_outcome": values,
            "breakout_path_efficiency_outcome": values,
            "breakout_participation_outcome": values,
            "breakout_reversal_magnitude_outcome": values,
        }
    )


def test_thresholds_are_fitted_once_and_frozen_for_scoring() -> None:
    thresholds = fit_breakout_path_quality_thresholds(_outcomes())
    scored = materialize_breakout_path_quality_labels(_outcomes(120), thresholds)
    assert thresholds.fit_rows == 200
    assert scored["breakout_low_efficiency"].iloc[0] == 1
    assert scored["breakout_rapid_reversal"].iloc[-1] == 1


def test_soft_risk_is_failure_count_fraction() -> None:
    thresholds = fit_breakout_path_quality_thresholds(_outcomes())
    scored = materialize_breakout_path_quality_labels(_outcomes(), thresholds)
    expected = scored["breakout_path_quality_failure_count"] / 4.0
    np.testing.assert_allclose(scored["breakout_path_quality_soft_risk"], expected)


def test_severe_retention_uses_train_only_capture_tail() -> None:
    threshold = fit_severe_retention_threshold(pd.Series(range(-100, 100)))
    labels = materialize_severe_retention_failure(
        pd.Series([0.0, 1.0, 0.0]),
        pd.Series([-99.0, -99.0, 10.0]),
        threshold,
    )
    assert labels.tolist() == [1, 0, 0]
