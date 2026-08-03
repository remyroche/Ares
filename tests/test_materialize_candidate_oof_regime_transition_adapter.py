from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.materialize_candidate_oof_regime_transition_adapter import materialize_adapter


def _hourly() -> pd.DataFrame:
    source = pd.date_range("2026-01-01", periods=24 * 55, freq="h", tz="UTC")
    index = np.arange(len(source), dtype=float)
    phase = np.where(index % 19 == 0, "transition", np.where(index % 11 == 0, "approach", "stable"))
    return pd.DataFrame({
        "source_utc": source,
        "execution_decision_utc": source,
        "market_vol": np.sin(index / 11.0),
        "market_breadth": np.cos(index / 17.0),
        "market_liquidity": index / len(index),
        "target__phase": phase,
        "target__transition_active": (phase == "transition").astype(int),
        "target__available_utc": source + pd.Timedelta(hours=6),
    })


def test_adapter_builds_independent_hourly_layers_and_asof_joins(tmp_path: Path) -> None:
    hourly = _hourly()
    candidates = pd.DataFrame({
        "candidate_id": [f"c{index}" for index in range(20)],
        "__ts__": pd.date_range("2026-02-10", periods=20, freq="6h", tz="UTC"),
        "__symbol__": "BTC/USD:USD",
        "side_name": "long",
    })
    candidates_path, hourly_path = tmp_path / "candidates.parquet", tmp_path / "hourly.parquet"
    candidates.to_parquet(candidates_path, index=False)
    hourly.to_parquet(hourly_path, index=False)
    output = materialize_adapter(candidates_path=candidates_path, hourly_regime_path=hourly_path, hourly_transition_path=hourly_path, output_dir=tmp_path / "out", evaluation_start="2026-02-10T00:00:00Z", frequency="week", n_components=2, max_features=3)
    joined = pd.read_parquet(output / "candidate_oof_regime_transition.parquet")
    assert len(joined) == len(candidates)
    assert {"regime_state_p__0", "transition_state_p__stable", "transition_active_probability"}.issubset(joined.columns)
    assert (pd.to_datetime(joined["regime_train_end_utc"], utc=True) < pd.to_datetime(joined["__ts__"], utc=True)).all()
    assert (pd.to_datetime(joined["transition_train_end_utc"], utc=True) < pd.to_datetime(joined["__ts__"], utc=True)).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["coverage"]["exact_candidate_coverage"]
    assert manifest["contract"]["transition_layer"].startswith("phase simplex")
