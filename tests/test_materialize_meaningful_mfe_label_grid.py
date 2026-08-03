from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.meaningful_mfe_label_grid import MeaningfulMFEGridSpec
from scripts.materialize_meaningful_mfe_label_grid import (
    HourlyBars,
    materialize_symbol_labels,
)


def test_materializer_requires_exact_contiguous_complete_path() -> None:
    timestamps = pd.date_range("2026-07-01", periods=30, freq="1h", tz="UTC")
    bars = HourlyBars(
        timestamp_ns=timestamps.astype("int64").to_numpy(),
        open=np.full(30, 100.0),
        high=np.linspace(100.0, 105.0, 30),
        low=np.linspace(100.0, 99.0, 30),
        close=np.linspace(100.0, 104.0, 30),
    )
    frame = pd.DataFrame(
        {
            "__ts__": [timestamps[0], timestamps[7]],
            "__symbol__": "BTC",
            "side_name": "long",
            "candidate_id": ["a", "b"],
            "execution_decision_utc": [timestamps[0], timestamps[7] + pd.Timedelta(minutes=1)],
            "oof_entry_atr_fraction": [0.01, 0.01],
        }
    )
    result = materialize_symbol_labels(
        frame,
        bars,
        [
            MeaningfulMFEGridSpec(horizon_hours=12, upper_atr=1.5),
            MeaningfulMFEGridSpec(horizon_hours=24, upper_atr=2.0),
        ],
        decision_column="execution_decision_utc",
        atr_column="oof_entry_atr_fraction",
    )
    assert set(result["candidate_id"]) == {"a"}
    assert set(result["grid_name"]) == {"h12_u1p5atr", "h24_u2p0atr"}
    assert (
        result["label_resolution_utc"]
        == result["execution_decision_utc"]
        + pd.to_timedelta(result["horizon_hours"], unit="h")
    ).all()
