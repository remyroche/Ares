from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.materialize_path_archetype_candidates import materialize


def test_materializes_exact_top40_join(tmp_path: Path) -> None:
    population = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "__symbol__": ["A", "B"],
            "side_name": ["long", "short"],
            "prediction": [0.8, 0.7],
            "candidate_id": ["A-long", "B-short"],
            "base_candidate_rank_timestamp_side": [1, 1],
            "base_candidate_rank_pct_timestamp_side": [0.2, 0.2],
            "selected_top40": [True, True],
        }
    )
    labels = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01"], utc=True),
            "__symbol__": ["A"],
            "side_name": ["long"],
            "side": [1],
            "__barrier_pct__": [0.02],
            "__path_auxiliary_atr_fraction__": [0.01],
            "candidate_id": ["A-long"],
        }
    )
    population_path = tmp_path / "population.parquet"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    population.to_parquet(population_path, index=False)
    labels.to_parquet(labels_dir / "part.parquet", index=False)
    manifest = materialize(population_path, labels_dir, tmp_path / "output")
    result = pd.read_parquet(manifest["output"])
    assert manifest["population_rows"] == 2
    assert manifest["rows"] == 1
    assert result.loc[0, "__symbol__"] == "A"
    assert result.loc[0, "base_oof_score"] == 0.8
    assert result.loc[0, "path_cost_return"] == 0.01
    assert bool(result.loc[0, "selected_top40"])
    assert manifest["score_column"] == "prediction"
