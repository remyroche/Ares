from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_canonical_global_book_conversion_context import (
    COORDINATE_COLUMNS,
    PANEL_CONTEXT,
    band_feature_columns,
    global_book_feature_columns,
    materialize_band_context,
    materialize_global_book_context,
    _validate_feature_surface,
)


def test_global_context_contract_is_unique_and_outcome_free() -> None:
    columns = (*global_book_feature_columns(), *band_feature_columns())
    _validate_feature_surface(global_book_feature_columns())
    _validate_feature_surface(band_feature_columns())
    assert len(global_book_feature_columns()) == len(
        set(global_book_feature_columns())
    )
    assert len(band_feature_columns()) == len(set(band_feature_columns()))
    assert set(global_book_feature_columns()).intersection(
        band_feature_columns()
    ) == set()
    assert not any("execution_net" in column for column in columns)
    assert not any("execution_gross" in column for column in columns)
    assert not any("execution_cost" in column for column in columns)
    assert not any("target" in column for column in columns)
    assert not any("selected" in column for column in columns)
    assert "context__book_fraction" not in columns
    assert "context__horizon_hours" not in columns


def _candidates() -> pd.DataFrame:
    stamps = pd.to_datetime(
        ["2025-01-01T00:00:00Z"] * 4 + ["2025-01-01T01:00:00Z"] * 4,
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index}" for index in range(8)],
            "execution_decision_utc": stamps,
            "side_name": ["long", "short"] * 4,
            "__symbol__": ["A", "B", "A", "C"] * 2,
            "causal_global_mapped_ev_band": ["B0", "B1", "B4", "B4"] * 2,
            "_coordinate_available": True,
        }
    )
    for column in PANEL_CONTEXT:
        frame[column] = np.arange(len(frame), dtype=float)
    coordinate_values = {
        "mapped_direct_net": np.tile([0.0, 1.0, 2.0, 3.0], 2),
        "map_reference_rows": 1000.0,
        "map_side_reference_rows": 500.0,
        "map_cell_reference_rows": 100.0,
        "causal_global_mapped_ev_percentile": np.tile(
            [0.25, 0.65, 0.96, 0.99], 2
        ),
        "causal_global_mapped_ev_reference_rows": 1000.0,
        "causal_global_mapped_ev_cutoff_p90": 1.5,
        "causal_global_mapped_ev_margin_to_p90": np.tile(
            [-1.5, -0.5, 0.5, 1.5], 2
        ),
    }
    for column in COORDINATE_COLUMNS:
        frame[column] = coordinate_values[column]
    return frame


def test_materialization_is_task_isolated_and_trailing_is_past_only() -> None:
    candidates = _candidates()
    anchor = pd.Timestamp("2025-01-01T01:00:00Z")
    audit = {
        "before_global_hour_complete_flag": True,
        "after_global_hour_complete_flag": True,
        "before_target_available_utc": anchor,
        "after_target_available_utc": anchor + pd.Timedelta(hours=12),
    }
    book_keys = pd.DataFrame(
        [
            {
                "cohort_anchor_utc": anchor,
                "horizon_hours": horizon,
                "book_fraction": fraction,
                **audit,
            }
            for horizon in (3, 12)
            for fraction in (0.1, 0.2)
        ]
    )
    book = materialize_global_book_context(candidates, book_keys)
    assert len(book) == 4
    assert tuple(
        column for column in book if column.startswith("context__")
    ) == global_book_feature_columns()
    assert book["context__current_population_support"].eq(4).all()
    assert book["context__trailing_3h__population_support"].eq(4).all()
    assert book["context__current_above_causal_p90_share"].eq(0.5).all()
    assert book.loc[:, global_book_feature_columns()].drop_duplicates().shape[0] == 1

    band_keys = pd.DataFrame(
        [
            {
                "cohort_anchor_utc": anchor,
                "horizon_hours": horizon,
                "global_common_ev_band": "B4",
                **audit,
            }
            for horizon in (3, 12)
        ]
    )
    band = materialize_band_context(candidates, band_keys)
    assert len(band) == 2
    assert tuple(
        column for column in band if column.startswith("context__")
    ) == band_feature_columns()
    assert band["context__current_band_support"].eq(2).all()
    assert band["context__trailing_3h_band__support"].eq(2).all()
    assert band.loc[:, band_feature_columns()].drop_duplicates().shape[0] == 1
