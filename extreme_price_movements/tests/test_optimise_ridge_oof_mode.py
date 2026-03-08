import json

import numpy as np
import pandas as pd

from extreme_price_movements.optimise import run_optimise_from_ridge_oof


def test_run_optimise_from_ridge_oof_persists_grid_and_summary(tmp_path):
    run_id = "20260101_000000"
    oof_dir = tmp_path / "artifacts" / run_id / "ridge_sizer"
    oof_dir.mkdir(parents=True)

    n = 48
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    assets = np.array(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"] * (n // 4), dtype=object)
    entry = 100.0 + np.arange(n, dtype=float) * 0.1

    future_opens = []
    future_highs = []
    future_lows = []
    future_closes = []
    for i in range(n):
        base = entry[i]
        path = np.array([base * 1.001, base * 1.003, base * 1.005, base * 1.007], dtype=np.float32)
        future_opens.append(path)
        future_highs.append(path * 1.001)
        future_lows.append(path * 0.999)
        future_closes.append(path)

    df = pd.DataFrame(
        {
            "ts": ts,
            "asset": assets,
            "bucket": np.where((np.arange(n) // 4) % 2 == 0, "LONG_MR", "LONG_TF"),
            "close": entry,
            "side": "LONG",
            "sizer_score_oof": np.linspace(0.1, 2.0, n, dtype=np.float32),
            "opt_limit_offset_pct": np.linspace(0.0, 0.0015, n, dtype=np.float32),
            "future_opens": future_opens,
            "future_highs": future_highs,
            "future_lows": future_lows,
            "future_closes": future_closes,
            "entry_price": entry,
            "is_long": np.ones(n, dtype=np.int8),
            "label_policy_sl_atr_mult": np.full(n, 1.0, dtype=np.float32),
            "label_policy_tp_sl_ratio": np.full(n, 2.0, dtype=np.float32),
            "atr_12_15m": np.full(n, 0.25, dtype=np.float32),
            "label_policy_giveback_pct": np.zeros(n, dtype=np.float32),
            "label_policy_max_hold_bars": np.full(n, 4, dtype=np.int16),
        }
    )
    df.to_parquet(oof_dir / "ridge_sizer_oof_all.parquet", index=False)

    summary = run_optimise_from_ridge_oof(
        run_id=run_id,
        data_root=str(tmp_path),
        fee_roundtrip=0.003,
        cooldown_hours=0.0,
    )

    grid_path = oof_dir / "ridge_oof_optimise_grid.csv"
    best_path = oof_dir / "ridge_oof_optimise_best.json"

    assert summary["mode"] == "ridge_oof"
    assert grid_path.exists()
    assert best_path.exists()

    grid = pd.read_csv(grid_path)
    payload = json.loads(best_path.read_text())

    assert not grid.empty
    assert "bucket" in grid.columns
    assert payload["mode"] == "ridge_oof"
    assert payload["source_oof_path"].endswith("ridge_sizer_oof_all.parquet")
    assert "best" in payload and isinstance(payload["best"], dict)
    assert set(payload["best_by_bucket"]) == {"LONG_MR", "LONG_TF"}
