from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.representation_search_cache import (
    cached_side_conditioned_donor_map,
    load_reference_cache,
    prepare_reference_cache,
)


def _frame(rows: int = 90) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": [f"S{i % 5}" for i in range(rows)],
            "side": np.where(np.arange(rows) % 2, -1, 1).astype(np.int8),
            "ret_1h": np.linspace(-1.0, 1.0, rows),
            "oi_chg_4h": np.linspace(1.0, -1.0, rows),
        }
    )


def test_reference_cache_reuses_exact_nested_rows(tmp_path) -> None:
    manifest = prepare_reference_cache(
        _frame(),
        feature_names=["ret_1h", "oi_chg_4h"],
        output_dir=tmp_path,
        reference_rows=60,
        scaler_rows=30,
        sample_sizes=(15, 30, 60),
    )
    restored, values, keys = load_reference_cache(tmp_path)
    small = np.load(restored.sample_indices["15"])
    medium = np.load(restored.sample_indices["30"])
    assert manifest.row_identity_hash == restored.row_identity_hash
    assert values.shape == (60, 2)
    assert len(keys) == 60
    assert len(np.unique(medium)) == 30
    assert small.min() == 0 and small.max() == 59


def test_donor_map_never_crosses_side(tmp_path) -> None:
    sides = np.asarray([1, -1, 1, -1, 1, -1], dtype=np.int8)
    donors = cached_side_conditioned_donor_map(
        sides, seed=7, output_path=tmp_path / "donors.npy"
    )
    assert np.array_equal(sides, sides[donors])
    assert np.array_equal(donors, np.load(tmp_path / "donors.npy"))
