from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_v11_residual_context import _materialize


def test_context_materialization_preserves_frozen_predictions(tmp_path) -> None:
    keys = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"]),
            "__symbol__": ["A", "B"],
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_default_clean_path"] * 2,
            "frozen_rank": [0.91, 0.92],
        }
    )
    context = keys.drop(columns=["frozen_rank"]).assign(short_covering_score_market=[0.1, 0.2])
    source = tmp_path / "predictions.parquet"
    context_path = tmp_path / "context.parquet"
    output = tmp_path / "joined.parquet"
    keys.to_parquet(source, index=False)
    context.to_parquet(context_path, index=False)
    manifest = _materialize(source, context_path, output)
    joined = pd.read_parquet(output)
    assert manifest["match_rate"] == 1.0
    assert joined["frozen_rank"].tolist() == [0.91, 0.92]
    assert joined["short_covering_score_market"].tolist() == [0.1, 0.2]


def test_context_materialization_fails_closed_on_incomplete_required_coverage(tmp_path) -> None:
    keys = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"]),
            "__symbol__": ["A", "B"],
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_default_clean_path"] * 2,
        }
    )
    context = keys.iloc[:1].assign(short_covering_score_market=[0.1])
    source = tmp_path / "predictions.parquet"
    context_path = tmp_path / "context.parquet"
    keys.to_parquet(source, index=False)
    context.to_parquet(context_path, index=False)
    with pytest.raises(ValueError, match="coverage"):
        _materialize(
            source,
            context_path,
            tmp_path / "joined.parquet",
            required_columns=["short_covering_score_market"],
            minimum_match_rate=1.0,
        )
