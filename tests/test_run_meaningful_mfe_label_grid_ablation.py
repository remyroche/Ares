from __future__ import annotations

import pandas as pd

from scripts.run_meaningful_mfe_label_grid_ablation import (
    _available_expanding_month_folds,
)


def test_grid_folds_skip_leading_month_and_respect_resolution() -> None:
    decision = pd.Series(
        pd.to_datetime(
            [
                "2026-05-15T00:00:00Z",
                "2026-05-31T12:00:00Z",
                "2026-06-15T00:00:00Z",
                "2026-07-15T00:00:00Z",
            ]
        )
    )
    resolution = decision + pd.Timedelta(hours=24)
    folds = _available_expanding_month_folds(
        decision, resolution, purge_hours=24.0
    )
    assert [fold["month"] for fold in folds] == ["2026-06", "2026-07"]
    for fold in folds:
        train = fold["train_indices"]
        valid = fold["validation_indices"]
        validation_start = decision.iloc[valid].min().to_period("M").start_time.tz_localize("UTC")
        assert resolution.iloc[train].max() < validation_start
