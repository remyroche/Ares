import numpy as np
import pandas as pd

from scripts.materialize_canonical_opportunity_payoff_trust_panel import (
    add_score_context,
)


def test_score_context_uses_deterministic_timestamp_side_top40_cutoff():
    rows = []
    for side in ("long", "short"):
        for index, score in enumerate((5.0, 4.0, 3.0, 2.0, 1.0)):
            rows.append(
                {
                    "candidate_id": f"{side}-{index}",
                    "side_name": side,
                    "__symbol__": f"S{index}",
                    "__ts__": pd.Timestamp("2025-02-01T00:00:00Z"),
                    "base_oof_score": score,
                }
            )
    result, generated = add_score_context(pd.DataFrame(rows))
    assert "base_margin_to_top40_cutoff" in generated
    for side in ("long", "short"):
        group = result.loc[result["side_name"].eq(side)].sort_values(
            "base_rank_timestamp_side"
        )
        assert group["selected_top40_timestamp_side"].tolist() == [
            True,
            True,
            False,
            False,
            False,
        ]
        assert group["base_top40_cutoff_timestamp_side"].eq(4.0).all()
        assert np.allclose(
            group["base_margin_to_top40_cutoff"].to_numpy(),
            np.array([1.0, 0.0, -1.0, -2.0, -3.0]),
        )
        assert group["base_rank_decile_timestamp_side"].tolist() == [2, 4, 6, 8, 9]


def test_score_context_global_rank_is_pooled_across_sides():
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "side_name": ["long", "long", "short", "short"],
            "__symbol__": ["A", "B", "C", "D"],
            "__ts__": [pd.Timestamp("2025-02-01T00:00:00Z")] * 4,
            "base_oof_score": [0.4, 0.1, 0.3, 0.2],
        }
    )
    result, _ = add_score_context(frame)
    ordered = result.sort_values("base_rank_timestamp_global")
    assert ordered["candidate_id"].tolist() == ["a", "c", "d", "b"]
    assert ordered["base_group_rows_timestamp_global"].eq(4).all()
