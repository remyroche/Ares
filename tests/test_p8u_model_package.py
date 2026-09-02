from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_model_package import add_base_geometry, timestamp_desc_rank


def _base_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["b", "a", "c", "x"],
        "__decision_ts__": ["2026-08-28T00:00:00Z"] * 3 + ["2026-08-28T01:00:00Z"],
        "side_name": ["long"] * 4,
        "base_score": [2.0, 2.0, 1.0, 3.0],
    })


def test_timestamp_rank_uses_candidate_id_tie_break() -> None:
    frame = _base_rows()
    rank = timestamp_desc_rank(frame, "base_score")
    # a and b tie; a is first by the sealed candidate-ID tie break.
    assert np.allclose(rank, [0.5, 5 / 6, 1 / 6, 0.5])


def test_base_geometry_requires_the_exact_persisted_rank() -> None:
    frame = _base_rows()
    frame["base_rank_ts"] = timestamp_desc_rank(frame, "base_score")
    enriched = add_base_geometry(frame)
    assert set(enriched.columns).issuperset({
        "base_query_count", "base_query_mean", "base_query_std", "base_query_range",
        "base_score_z_ts", "base_top_gap", "base_top2_gap",
    })
    assert enriched.loc[enriched.candidate_id.eq("a"), "base_top_gap"].item() == pytest.approx(0.0)
    assert enriched.loc[enriched.candidate_id.eq("b"), "base_top2_gap"].item() == pytest.approx(1.0)
    frame.loc[0, "base_rank_ts"] = 0.0
    with pytest.raises(ValueError, match="does not match"):
        add_base_geometry(frame)
