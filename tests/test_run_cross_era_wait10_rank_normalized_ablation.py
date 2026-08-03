from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_cross_era_wait10_rank_normalized_ablation import (
    RANK_FEATURES,
    add_complete_group_rank_coordinates,
)


def test_complete_group_rank_coordinates_are_side_timestamp_local() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "side_name": ["long", "long", "short", "short"],
            "__symbol__": ["A", "B", "A", "B"],
            "__ts__": pd.to_datetime(["2025-01-01"] * 4, utc=True),
            "score_base_alpha": [2.0, 1.0, 3.0, 4.0],
            "score_residual_expected_ev": [1.0, 2.0, 4.0, 3.0],
        }
    )
    result = add_complete_group_rank_coordinates(frame)
    assert result["base_rank_pct_timestamp_side_cross_era"].tolist() == [
        0.5,
        1.0,
        1.0,
        0.5,
    ]
    assert result["residual_rank_pct_timestamp_side_cross_era"].tolist() == [
        1.0,
        0.5,
        0.5,
        1.0,
    ]
    assert np.isfinite(result[list(RANK_FEATURES)].to_numpy()).all()


def test_rank_coordinates_do_not_include_raw_score_levels() -> None:
    assert "score_base_alpha" not in RANK_FEATURES
    assert "score_residual_expected_ev" not in RANK_FEATURES
