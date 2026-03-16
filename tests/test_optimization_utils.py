import numpy as np
import pandas as pd

from extreme_price_movements.optimization_utils import filter_low_variance_assets


class DummyCloseStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, symbol, columns=None, start_ts=None, end_ts=None):
        df = self.frames[symbol]
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end_ts is not None:
            df = df[df.index <= end_ts]
        return df


def test_filter_low_variance_assets_supports_sample_stride():
    idx = pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC")
    frames = {
        "FAST": pd.DataFrame(
            {"close": 100.0 + np.cumsum(np.sin(np.linspace(0.0, 30.0, len(idx))) + 1.5)},
            index=idx,
        ),
        "FLAT": pd.DataFrame({"close": np.ones(len(idx))}, index=idx),
    }
    store = DummyCloseStore(frames)

    kept = filter_low_variance_assets(
        store,
        ["FAST", "FLAT"],
        lookback_days=30,
        threshold_pct=0.5,
        ts_sig=idx[-1],
        sample_stride=100,
    )

    assert kept == ["FAST"]
