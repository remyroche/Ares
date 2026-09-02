from pathlib import Path

import pandas as pd

import scripts.run_strict_r3_p8u_meta_target_query_grid_v1 as target_grid


def test_path_label_current_month_is_required_and_previous_is_optional(tmp_path: Path) -> None:
    month = pd.Timestamp("2025-03-01", tz="UTC")
    current = tmp_path / "month=2025-03" / "side=long.parquet"
    current.parent.mkdir(parents=True)
    current.touch()

    assert target_grid._path_label_paths(tmp_path, month) == (current,)


def test_path_label_includes_existing_previous_partition(tmp_path: Path) -> None:
    month = pd.Timestamp("2025-03-01", tz="UTC")
    previous = tmp_path / "month=2025-02" / "side=long.parquet"
    current = tmp_path / "month=2025-03" / "side=long.parquet"
    previous.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    previous.touch()
    current.touch()

    assert target_grid._path_label_paths(tmp_path, month) == (previous, current)
