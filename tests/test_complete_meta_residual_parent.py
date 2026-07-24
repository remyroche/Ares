from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_complete_meta_residual_parent import (
    _causal_historical_ranks,
    _causal_trailing_day_ranks,
)
from scripts.materialize_causal_residual_episode_parent import materialize


def test_causal_historical_ranks_use_prior_months_only() -> None:
    evaluation = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-02T00:00:00Z",
                    "2026-05-01T00:00:00Z",
                ],
                utc=True,
            ),
            "score_meta_base_soft_label": [0.25, 0.75, 0.75],
        }
    )

    ranks, manifest = _causal_historical_ranks(
        np.asarray([0.0, 1.0]), evaluation
    )

    # Both April values are ranked against only [0, 1], so neither April row
    # can change the other row's rank.
    np.testing.assert_allclose(ranks[:2], [0.5, 0.5])
    # May sees the two April scores in addition to the original history.
    np.testing.assert_allclose(ranks[2], 0.625)
    assert manifest[0]["reference_rows"] == 2
    assert manifest[1]["reference_rows"] == 4


def test_causal_trailing_day_ranks_exclude_current_day() -> None:
    evaluation = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-01T12:00:00Z",
                    "2026-04-02T00:00:00Z",
                    "2026-04-11T00:00:00Z",
                ],
                utc=True,
            ),
            "score_meta_base_soft_label": [0.0, 1.0, 1.0, 0.5],
        }
    )

    ranks, manifest = _causal_trailing_day_ranks(
        pd.DataFrame(
            {
                "__ts__": pd.to_datetime(["2026-03-31T00:00:00Z"], utc=True),
                "score_meta_base_soft_label": [0.5],
            }
        ),
        evaluation,
        lookback_days=8,
    )

    # Both April 1 values see only the March 31 reference, proving that an
    # intraday earlier score cannot change the same day's rank.
    np.testing.assert_allclose(ranks[:2], [0.0, 1.0])
    # April 2 has the prior day plus March 31 in its reference.
    np.testing.assert_allclose(ranks[2], 5.0 / 6.0)
    # April 10 has no eligible history because the most recent prior day is
    # outside the strict trailing eight-day window.
    assert np.isnan(ranks[3])
    assert manifest[0]["reference_rows"] == 1
    assert manifest[1]["reference_rows"] == 3


def test_causal_candidate_parent_materialization_uses_raw_score_alias(tmp_path) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-03-01T00:00:00Z",
                    "2026-03-01T12:00:00Z",
                    "2026-03-02T00:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["A", "B", "A"],
            "side_name": ["long", "short", "long"],
            "archetype_policy_key": ["arch", "arch", "arch"],
            "score": [0.2, 0.8, 0.8],
            "ev_after_1pct": [0.01, -0.01, 0.02],
            "clean_exec": [1, 0, 1],
            "dirty_positive": [0, 1, 0],
            "full_path_bad_mae_1r": [0, 1, 0],
            "timeout": [0, 0, 0],
        }
    )
    frame.to_parquet(root / "candidates_2026-03.parquet", index=False)
    output = tmp_path / "parent"
    manifest = materialize(
        root,
        output,
        start="2026-03-02",
        end="2026-03-03",
        lookback_days=8,
        min_reference_rows=1,
    )
    parent = pd.read_parquet(output / "causal_parent_predictions.parquet")
    assert manifest["rank_coverage"] == 1.0
    np.testing.assert_allclose(parent["hit_probability"], [0.8])
    np.testing.assert_allclose(parent["historical_rank"], [0.75])
