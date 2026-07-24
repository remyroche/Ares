import numpy as np
import pandas as pd

from scripts.materialize_strict_oos_failure_taxonomy_source import (
    _causal_clean_probability,
    _top_fraction_by_timestamp,
    materialize,
)


def test_causal_meta_probability_ignores_same_and_future_day_outcomes() -> None:
    days = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"], utc=True)
    frame = pd.DataFrame(
        {
            "day": days,
            "side_name": "long",
            "archetype_policy_key": "long_test",
            "rank": [0.9, 0.9, 0.9],
            "base": [0.4, 0.4, 0.4],
            "clean_exec": [1.0, 0.0, 0.0],
        }
    )
    first, support = _causal_clean_probability(
        frame, rank_column="rank", base_column="base", bins=10, shrinkage=2.0
    )
    changed = frame.copy()
    changed.loc[2, "clean_exec"] = 1.0
    second, _ = _causal_clean_probability(
        changed, rank_column="rank", base_column="base", bins=10, shrinkage=2.0
    )
    np.testing.assert_allclose(first[:2], second[:2])
    assert first[0] == 0.4
    assert first[1] > first[0]
    assert support.tolist() == [0.0, 1.0, 2.0]


def test_monitor_flag_uses_timestamp_global_top_fraction() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-04-01T00:00:00Z"] * 4
                + ["2026-04-01T01:00:00Z"] * 4,
                utc=True,
            ),
            "rank": [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
        }
    )
    selected = _top_fraction_by_timestamp(frame, "rank", 0.25)
    assert selected.tolist() == [False, False, False, True, False, False, False, True]


def test_materialized_hit_probability_uses_causal_meta_probability(
    tmp_path,
) -> None:
    """The strict base+meta source must not silently fall back to base_score."""

    source = tmp_path / "meta_oos.parquet"
    output = tmp_path / "out"
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-04-01T00:00:00Z", "2026-04-02T00:00:00Z"], utc=True
            ),
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
            "side_name": ["long", "long"],
            "archetype_policy_key": ["long_test", "long_test"],
            "score_base": [0.20, 0.20],
            "score_base_residual_ev_rank_train_reference": [0.90, 0.90],
            "clean_exec": [1.0, 0.0],
            "ev_after_1pct": [0.02, -0.01],
            "dirty_positive": [0.0, 1.0],
            "full_path_bad_mae_1r": [0.0, 1.0],
            "timeout": [0.0, 0.0],
        }
    )
    rows.to_parquet(source, index=False)

    manifest = materialize(source, output, rank_bins=10, local_shrinkage=2.0)
    materialized = pd.read_parquet(output / "candidate_shards" / "candidates_202604.parquet")
    second = materialized.sort_values("__ts__", kind="stable").iloc[1]

    assert second["base_score"] == 0.20
    assert second["score_meta_base_soft_label"] > second["base_score"]
    assert second["hit_probability"] == second["score_meta_base_soft_label"]
    assert manifest["meta_probability_contract"]["output_column"] == "hit_probability"
