import json

import numpy as np
import pandas as pd

from extreme_price_movements.optimise import Policy, run_optimise_step


def test_optimise_persists_entry_policy_payload(tmp_path):
    n = 140
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(7)

    entry = 100.0 + rng.normal(0, 1, size=n)
    ret = rng.normal(0.001, 0.01, size=n)
    exit_px = entry * (1.0 + ret)
    reason = np.where(rng.uniform(0, 1, size=n) > 0.25, "trailing_stop", "limit_not_filled")
    filled = reason != "limit_not_filled"

    trades = pd.DataFrame(
        {
            "timestamp": ts,
            "bucket": ["LONG_MR"] * n,
            "confidence": np.clip(rng.uniform(0.1, 0.95, size=n), 0, 1),
            "entry_price": entry,
            "exit_price": exit_px,
            "is_long": np.ones(n, dtype=int),
            "score": rng.normal(0, 1, size=n),
            "reason": reason,
            "filled_via_limit": filled,
            "mae_pct": np.abs(rng.normal(0.01, 0.005, size=n)),
            "mfe_pct": np.abs(rng.normal(0.015, 0.008, size=n)),
            "duration": rng.integers(1, 20, size=n),
            "atr": np.clip(rng.normal(0.02, 0.003, size=n), 0.005, 0.05),
        }
    )
    trades.attrs["threaded_exit_stream"] = True
    trades.attrs["fee_pct"] = 0.003

    out_path = tmp_path / "bucket_params.json"
    out = run_optimise_step(
        trades=trades,
        atr_15m=pd.Series(0.02, index=trades.index),
        output_path=str(out_path),
        policy=Policy(mode="train_baseline", params_path=str(out_path)),
    )

    assert "LONG_MR" in out
    assert "entry_policy" in out["LONG_MR"]
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    bkt = payload["buckets"]["LONG_MR"]
    assert "entry_policy" in bkt
    assert "model" in bkt["entry_policy"]
    assert "objective" in bkt["entry_policy"]
