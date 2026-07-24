from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import run_materialized_trailing_label_topk_lgbm_hpo as runner


def _write(path: Path, value: float) -> None:
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "__symbol__": ["BTC/USD:USD"],
            "side": [1],
            "target_soft": [value],
        }
    ).to_parquet(path, index=False)


def test_monthly_partitions_exclude_legacy_unsuffixed_parquet(tmp_path: Path) -> None:
    monthly_long = tmp_path / "train_global_long_5_2026_01.parquet"
    monthly_short = tmp_path / "train_global_short_5_2026_01.parquet"
    stale = tmp_path / "train_global_short_7.parquet"
    _write(monthly_long, 0.1)
    _write(monthly_short, 0.2)
    _write(stale, 0.9)

    files = runner._canonical_label_files(tmp_path)
    assert [path.name for path in files] == [monthly_long.name, monthly_short.name]
    loaded = runner._load_canonical_labels(tmp_path)
    assert sorted(loaded["target_soft"].tolist()) == [0.1, 0.2]
    identity = runner._label_source_identity(tmp_path)
    assert identity["file_count"] == 2
    assert stale.name not in identity["files"]


def test_legacy_only_store_retains_all_parquet_files(tmp_path: Path) -> None:
    first = tmp_path / "train_global_long_5.parquet"
    second = tmp_path / "train_global_short_5.parquet"
    _write(first, 0.1)
    _write(second, 0.2)
    assert runner._canonical_label_files(tmp_path) == [first, second]
