from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import check_j4_j5_contextual_meta_fresh_oos_readiness as mod


def _write_freeze(path: Path, head: str = "short_asset") -> Path:
    freeze = path / "freeze.csv"
    pd.DataFrame(
        [
            {
                "head": head,
                "selected_contextual_feature_arm": "B_current_plus_model_state",
                "selected_capacity_config": "none_retain_context_arm",
                "selected_distillation_variant": "hard_label_context_arm",
                "effective_fresh_oos_after": "2026-06-15T04:00:00+00:00",
            }
        ]
    ).to_csv(freeze, index=False)
    return freeze


def test_readiness_requires_guarded_labels_and_scores(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    pd.DataFrame(
        {
            "__ts__": ["2026-06-15T05:00:00+00:00", "2026-06-15T06:00:00+00:00"],
            "__y_bin__": [1, 0],
        }
    ).to_parquet(label_dir / "train_asset_example_5.parquet")

    readiness, audit = mod.build_readiness(
        freeze,
        label_dir,
        [],
        min_later_hours=24.0,
        min_rows_per_head=1,
    )

    row = readiness.iloc[0]
    assert not bool(row["has_fresh_labels"])
    assert not bool(row["has_candidate_scores"])
    assert not bool(row["ready_for_fresh_oos_confirmation"])
    assert audit["status"] == "not_ready"


def test_readiness_passes_with_guarded_labels_and_scores(tmp_path: Path) -> None:
    freeze = _write_freeze(tmp_path)
    label_dir = tmp_path / "labels"
    score_dir = tmp_path / "scores"
    label_dir.mkdir()
    score_dir.mkdir()
    pd.DataFrame(
        {
            "__ts__": ["2026-06-16T05:00:00+00:00", "2026-06-16T06:00:00+00:00"],
            "__y_bin__": [1, 0],
        }
    ).to_parquet(label_dir / "train_asset_example_5.parquet")
    pd.DataFrame(
        {
            "timestamp": ["2026-06-16T05:00:00+00:00", "2026-06-16T06:00:00+00:00"],
            "score": [0.7, 0.2],
        }
    ).to_parquet(score_dir / "candidate_short_asset_scores.parquet")

    readiness, audit = mod.build_readiness(
        freeze,
        label_dir,
        [score_dir],
        min_later_hours=24.0,
        min_rows_per_head=2,
    )

    row = readiness.iloc[0]
    assert bool(row["has_fresh_labels"])
    assert bool(row["has_candidate_scores"])
    assert bool(row["ready_for_fresh_oos_confirmation"])
    assert audit["status"] == "ready"
