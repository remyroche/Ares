from __future__ import annotations

import pandas as pd

from extreme_price_movements.config import CANON_HORIZONS
from extreme_price_movements.offline_optimisers import compare_tbm_parameters as cmp
from extreme_price_movements.offline_optimisers import params_store
from extreme_price_movements.strategy_registry import normalize_strategy_horizon


def test_canonical_tbm_horizons_cover_3_to_7_hour_window() -> None:
    assert CANON_HORIZONS == [3, 5, 7]
    assert normalize_strategy_horizon(1) == 3
    assert normalize_strategy_horizon(3) == 3
    assert normalize_strategy_horizon(7) == 7


def test_compare_cell_keys_follow_runtime_horizons() -> None:
    keys = cmp._cell_keys_for_horizons([3, 7])
    assert "MR_long_H3" in keys
    assert "TF_short_H7" in keys
    assert "MR_long_H5" not in keys
    assert len(keys) == len(params_store.TBM_BUCKET_NAMES) * 2


def test_side_only_geometry_grid_aliases_bucket_cells(tmp_path, monkeypatch) -> None:
    grid_path = tmp_path / "tbm_geometry_grid.csv"
    pd.DataFrame(
        [
            {
                "cell_key": "long_H3",
                "k_tp": 1.1,
                "sl_as_tp_pct": 0.75,
                "base_atr_window": 840,
                "tp_abs_lo_pct": 0.012,
                "sl_abs_lo_pct": 0.005,
            }
        ]
    ).to_csv(grid_path, index=False)
    monkeypatch.setattr(params_store, "TBM_GEOMETRY_GRID_CSV", grid_path)

    loaded = params_store.load_tbm_geometry_grid()["per_cell"]
    assert loaded["long_H3"]["validated_triplets"] == [(1.1, 0.75, 840)]
    assert loaded["MR_long_H3"]["validated_triplets"] == [(1.1, 0.75, 840)]
    assert loaded["TF_long_H3"]["validated_triplets"] == [(1.1, 0.75, 840)]


def test_side_only_best_params_alias_bucket_cells(tmp_path, monkeypatch) -> None:
    cell_path = tmp_path / "tbm_best_params_per_cell.csv"
    pd.DataFrame(
        [
            {
                "cell_key": "short_H7",
                "rank_in_cell": 1,
                "config_id": "CFG_SIDE",
                "k_tp": 0.9,
                "sl_as_tp_pct": 0.6,
                "base_atr_window": 840,
            }
        ]
    ).to_csv(cell_path, index=False)
    monkeypatch.setattr(params_store, "TBM_BEST_PARAMS_PER_CELL_CSV", cell_path)

    best = params_store.load_tbm_best_params_per_cell()
    assert best["short_H7"]["config_id"] == "CFG_SIDE"
    assert best["MR_short_H7"]["config_id"] == "CFG_SIDE"
    assert best["TF_short_H7"]["config_id"] == "CFG_SIDE"

    all_rows = params_store.load_tbm_all_params_per_cell()
    assert all_rows["MR_short_H7"][0]["config_id"] == "CFG_SIDE"
    assert all_rows["TF_short_H7"][0]["config_id"] == "CFG_SIDE"
