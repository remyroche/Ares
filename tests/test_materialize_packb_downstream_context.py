from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_packb_downstream_context import (
    DownstreamContextError,
    build_context,
)


def _outer() -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-04-01T00:00:00Z")
    rows = []
    for side in ("long", "short"):
        for rank, score in enumerate(np.linspace(1.0, 0.1, 10), start=1):
            rows.append(
                {
                    "candidate_id": f"{side}-{rank}",
                    "side_name": side,
                    "__ts__": timestamp,
                    "__symbol__": f"S{rank:02d}",
                    "prediction": score,
                }
            )
    return pd.DataFrame(rows)


def _top40(outer: pd.DataFrame) -> pd.DataFrame:
    selected = outer.groupby(["__ts__", "side_name"], sort=False).head(4).copy()
    selected["base_candidate_rank_timestamp_side"] = (
        selected.groupby(["__ts__", "side_name"], sort=False).cumcount() + 1
    )
    selected["base_candidate_rank_pct_timestamp_side"] = (
        selected["base_candidate_rank_timestamp_side"] / 10.0
    )
    selected["base_candidate_group_rows"] = 10
    selected["selected_top40"] = True
    selected["prediction_source"] = "outer_oof_fold_model"
    return selected


def test_build_context_is_side_local_finite_and_preentry_only() -> None:
    outer = _outer()

    result = build_context(_top40(outer), outer)

    assert len(result) == 8
    assert set(result["side_name"]) == {"long", "short"}
    assert result["selected_top40"].all()
    assert result["candidate_id"].is_unique
    assert set(result["archetype"]) == {
        "base_rank_decile_1",
        "base_rank_decile_2",
        "base_rank_decile_3",
        "base_rank_decile_4",
    }
    assert np.isfinite(
        result[
            [
                "score",
                "base_margin_to_cutoff",
                "base_margin_to_cutoff_z",
                "base_signal_zscore_within_archetype",
            ]
        ].to_numpy(float)
    ).all()
    assert (
        result.groupby(["__ts__", "side_name"])["base_margin_to_cutoff"]
        .min()
        .eq(0.0)
        .all()
    )
    assert not any(column.startswith("path_arch_") for column in result.columns)


def test_build_context_rejects_score_lineage_mismatch() -> None:
    outer = _outer()
    selected = _top40(outer)
    selected.loc[selected.index[0], "prediction"] += 0.5

    with pytest.raises(DownstreamContextError, match="scores differ"):
        build_context(selected, outer)
