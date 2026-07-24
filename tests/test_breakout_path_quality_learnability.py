from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_breakout_path_quality_learnability import (
    _feature_columns,
    _partition_month,
    _sample_train,
)


def test_conservative_feature_variant_excludes_only_lagged_path_state() -> None:
    schema = [
        "__ts__", "side_name", "__archetype_policy_key__", "__first_touch_capture_net__",
        "dir_path_short_2h", "up_barrier_pressure_daily_vwap", "trend_z_t",
    ]
    assert _feature_columns(schema, "all_observable") == [
        "dir_path_short_2h", "trend_z_t", "up_barrier_pressure_daily_vwap",
    ]
    assert _feature_columns(schema, "exclude_lagged_path_state") == ["trend_z_t"]


def test_partition_month_and_stratified_sample_are_deterministic() -> None:
    assert _partition_month(Path("train_global_short_5_2026_04.parquet")).strftime("%Y-%m") == "2026-04"
    target = np.tile(np.array([0, 1], dtype=np.int8), 100)
    sample = _sample_train(len(target), target, maximum=30, seed=7)
    assert len(sample) == 30
    assert np.array_equal(sample, _sample_train(len(target), target, maximum=30, seed=7))
    assert target[sample].min() == 0 and target[sample].max() == 1
