from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_cross_era_regime_transition_classifier_ablation import (
    feature_sets,
    grouped_oof_predictions,
)


def _frame() -> pd.DataFrame:
    rows = []
    for group in range(12):
        for row in range(10):
            value = group + row / 10.0
            rows.append(
                {
                    "cohort_anchor_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=7 * group, hours=row),
                    "horizon_hours": 3,
                    "source_family": "fixture",
                    "economics_tier": "fixture",
                    "mapping_provenance_role": "strict_oof",
                    "cv_group_id": f"g{group:02d}",
                    "context__state_mean__x": value,
                    "context__past_delta_3h__x": row / 10.0,
                    "target": float((group + row) % 3 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_feature_family_partition_keeps_transition_context_explicit() -> None:
    families = feature_sets(
        [
            "context__state_mean__x",
            "context__mapping_current__mapped_mean",
            "context__past_delta_3h__x",
            "context__past_geometry_shift_3h",
        ]
    )
    assert "context__state_mean__x" in families["raw_state_only"]
    assert (
        "context__mapping_current__mapped_mean"
        in families["coordinates_only"]
    )
    assert (
        "context__past_delta_3h__x"
        in families["past_transitions_only"]
    )
    assert set(families["coordinates_plus_raw_state"]) == {
        "context__state_mean__x",
        "context__mapping_current__mapped_mean",
        "context__past_delta_3h__x",
        "context__past_geometry_shift_3h",
    }


def test_grouped_predictions_are_complete_and_group_disjoint() -> None:
    frame = _frame()
    result = grouped_oof_predictions(
        frame,
        columns=[
            "context__state_mean__x",
            "context__past_delta_3h__x",
        ],
        target="target",
        model_name="logistic",
        n_splits=4,
    )
    assert len(result) == len(frame)
    assert np.isfinite(result["prediction"]).all()
    assert result["cv_fold"].nunique() == 4
    assert result.groupby("cv_group_id")["cv_fold"].nunique().eq(1).all()
    assert result["selected_top10"].sum() > 0


def test_conditional_target_requires_and_preserves_exact_availability() -> None:
    frame = _frame().rename(
        columns={"target": "target__mechanism_upside_collapse"}
    )
    frame["target__mechanism_upside_collapse_available_utc"] = (
        frame["cohort_anchor_utc"] + pd.Timedelta(hours=18)
    )
    # Conditional mechanism heads must discard inactive/undefined rows rather
    # than treating them as negative examples.
    frame.loc[
        frame.index[::5], "target__mechanism_upside_collapse"
    ] = np.nan
    result = grouped_oof_predictions(
        frame,
        columns=[
            "context__state_mean__x",
            "context__past_delta_3h__x",
        ],
        target="target__mechanism_upside_collapse",
        model_name="logistic",
        n_splits=4,
    )
    assert len(result) == len(frame) - len(frame.index[::5])
    assert result["target_available_utc"].notna().all()
    assert (
        result["target_available_utc"]
        .sub(result["cohort_anchor_utc"])
        .eq(pd.Timedelta(hours=18))
        .all()
    )
