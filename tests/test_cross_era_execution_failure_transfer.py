from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_cross_era_execution_failure_transfer import (  # noqa: E402
    causal_trailing_robust_z,
    common_health_columns,
    fit_transfer,
)


def test_common_catalog_excludes_semantically_incompatible_fields() -> None:
    catalog = pd.DataFrame(
        {
            "feature": [
                "health__mapped_ev_std",
                "health__alpha_uncertainty_mean",
                "health__catboost_entropy_mean",
            ],
            "cross_era_common": [True, False, False],
        }
    )
    assert common_health_columns(catalog) == ["health__mapped_ev_std"]


def test_causal_normalization_uses_only_strictly_earlier_same_era_rows() -> None:
    timestamp = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    original = pd.DataFrame(
        {
            "source_utc": list(timestamp) + list(timestamp),
            "era": ["old"] * 6 + ["new"] * 6,
            "health__value": [1, 2, 3, 4, 5, 6] + [10, 20, 30, 40, 50, 60],
        }
    )
    changed = original.copy()
    changed.loc[
        changed["era"].eq("old") & changed["source_utc"].eq(timestamp[-1]),
        "health__value",
    ] = 1_000_000
    first, columns = causal_trailing_robust_z(
        original, ["health__value"], lookback_days=21, min_history_hours=3
    )
    second, _ = causal_trailing_robust_z(
        changed, ["health__value"], lookback_days=21, min_history_hours=3
    )
    column = columns[0]
    earlier = original["source_utc"].lt(timestamp[-1])
    assert np.allclose(
        first.loc[earlier, column],
        second.loc[earlier, column],
        equal_nan=True,
    )
    new_era = original["era"].eq("new")
    isolated, _ = causal_trailing_robust_z(
        original.loc[new_era].copy(),
        ["health__value"],
        lookback_days=21,
        min_history_hours=3,
    )
    assert np.allclose(
        first.loc[new_era, column].to_numpy(),
        isolated[column].to_numpy(),
        equal_nan=True,
    )


def test_transfer_fit_does_not_require_evaluation_labels_for_training() -> None:
    train = pd.DataFrame(
        {
            "feature": [-2.0, -1.0, 1.0, 2.0] * 10,
            "target": [0, 0, 1, 1] * 10,
        }
    )
    evaluation = pd.DataFrame(
        {
            "feature": [-1.5, 1.5],
            "target": [1, 0],
        }
    )
    first = fit_transfer(
        train,
        evaluation,
        features=["feature"],
        target_column="target",
        seed=7,
    )
    evaluation["target"] = 1 - evaluation["target"]
    second = fit_transfer(
        train,
        evaluation,
        features=["feature"],
        target_column="target",
        seed=7,
    )
    assert np.allclose(first, second)
