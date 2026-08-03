from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import diagnose_marapr2025_direct_residual_regime_break_learnability as d


def test_feature_arms_exclude_forbidden_fields() -> None:
    forbidden = ("month", "time", "score", "outcome", "mfe", "mae", "state_id")
    for fields in d.ARMS.values():
        assert all(not any(token in field.lower() for token in forbidden) for field in fields)
    assert len(d.ARMS["combined20"]) == 20


def test_week_groups_are_shared_by_side_and_change_after_seven_days() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-03-03T00:00:00Z",
                "2025-03-03T00:00:00Z",
                "2025-03-10T00:00:00Z",
            ],
            utc=True,
        )
    )
    groups = d.week_group(timestamps)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]
    days = d.day_group(timestamps)
    assert days.iloc[0] == days.iloc[1]
    assert days.iloc[0] != days.iloc[2]


def test_hour_side_contribution_target_reconciles() -> None:
    context = {field: [0.1, 0.2] for field in d.ARMS["combined20"]}
    candidates = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2025-03-18", "2025-03-18"], utc=True),
            "side_name": ["long", "short"],
            "diagnostic_period": ["march03_19", "march03_19"],
            **context,
        }
    )
    books = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2025-03-18", "2025-03-18", "2025-03-18"], utc=True
            ),
            "side_name": ["long", "short", "short"],
            "diagnostic_period": ["march03_19"] * 3,
            "selection_source": ["direct_q25", "direct_q25", "residual"],
            "candidate_id": ["a", "b", "c"],
            d.NET: [0.02, -0.01, 0.03],
        }
    )
    result = d.build_hour_side_panel(candidates, books)
    long = result.loc[result.side_name.eq("long")].iloc[0]
    short = result.loc[result.side_name.eq("short")].iloc[0]
    assert np.isclose(long.direct_q25_contribution_bps, 100.0)
    assert np.isclose(short.direct_q25_contribution_bps, -50.0)
    assert np.isclose(short.residual_contribution_bps, 300.0)
    assert np.isclose(
        result.direct_advantage_bps.sum(),
        (0.02 - 0.01) / 2 * 1e4 - 0.03 * 1e4,
    )


def test_grouped_oof_covers_rows_without_group_leakage() -> None:
    rng = np.random.default_rng(7)
    rows = 240
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "side_name": np.where(np.arange(rows) % 2, "long", "short"),
            "diagnostic_period": "test",
            "cv_group": np.repeat(np.arange(12), 20),
            "target": np.tile([0, 1], rows // 2),
            "direct_advantage_bps": rng.normal(size=rows),
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
        }
    )
    predictions, folds, coefficients = d.grouped_oof(
        frame, ("x1", "x2"), "target", n_splits=4
    )
    assert len(predictions) == rows
    assert predictions.probability.between(0, 1).all()
    assert predictions["fold"].nunique() == 4
    assert folds.validation_rows.sum() == rows
    assert set(coefficients.feature) == {"x1", "x2"}
