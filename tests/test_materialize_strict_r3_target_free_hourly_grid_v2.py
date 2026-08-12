from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "materialize_strict_r3_target_free_hourly_grid_v2.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_target_free_grid", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_signal_hour_spread_is_point_in_time_and_missing_stays_missing(
    tmp_path, monkeypatch,
) -> None:
    root = tmp_path / "official"
    root.mkdir()
    index = pd.date_range("2026-08-12T08:00:00Z", periods=2, freq="1h")
    pd.DataFrame(
        {
            "ob_bid_bestPrice": [100.0, 50.0],
            "ob_ask_bestPrice": [101.0, 100.0],
        }, index=index,
    ).to_parquet(root / "BTC_USD_USD.parquet")
    monkeypatch.setattr(MODULE, "FROZEN_INPUT_BACKFILL_ROOT", root)

    out = MODULE._signal_hour_spread_panel(
        ["BTC/USD:USD", "MISSING/USD:USD"], index[:1],
    )

    assert np.isclose(out.loc[index[0], "BTC/USD:USD"], 10_000.0 / 100.5)
    assert pd.isna(out.loc[index[0], "MISSING/USD:USD"])
    # The much wider later hour must not affect the earlier candidate.
    assert out.loc[index[0], "BTC/USD:USD"] < 100.0
