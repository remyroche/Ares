import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.breakdown_diagnostics import build_breakdown_events, run_breakdown_diagnostics


def _synthetic_ohlc(n=240, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0, 0.004, n)
    # inject a few deterministic shock segments to ensure event detection
    if n > 120:
        ret[60:66] += 0.02
        ret[140:146] -= 0.02
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    span = np.abs(rng.normal(0.002, 0.001, n))
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    atr = np.full(n, 0.02)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "atr_pct": atr}, index=idx)


def test_build_breakdown_events_basic():
    df = _synthetic_ohlc()
    out = build_breakdown_events(df, lookback_h=12, trigger=0.03, decluster_h=4, max_event_h=48)
    assert "event_id" in out.events.columns
    assert len(out.row_event_id) == len(df)
    assert (out.row_event_id >= -1).all()


def test_run_breakdown_diagnostics_smoke(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    df = _synthetic_ohlc()
    df.to_parquet(run_dir / "ohlc.parquet")

    cfg = {
        "lookback_h": 12,
        "baseline_trigger": 0.03,
        "trigger_sweep": [0.02, 0.03, 0.04],
        "decluster_h": 4,
        "max_event_h": 48,
        "entry_offsets_h": [-2, 0, 2],
        "directions": ["follow", "fade"],
        "cost_stress_multipliers": [1.0, 1.5],
        "optimise_run_dir": str(run_dir),
    }
    rep = run_breakdown_diagnostics(cfg, str(run_dir))
    assert "verdict" in rep
    out_dir = run_dir / "breakdown_diagnostics"
    assert (out_dir / "report.json").exists()
    assert (out_dir / "plots" / "offset_direction_mean_u.png").exists()
    assert (out_dir / "tables" / "events.parquet").exists()

    parsed = json.loads((out_dir / "report.json").read_text())
    assert "verdict" in parsed
