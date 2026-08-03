from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_febapr2025_exact1m_path_head_labels import _output_batch


def _path(start: pd.Timestamp, *, rising: bool) -> str:
    steps = np.arange(720, dtype=np.float64)
    close = 100.0 + (steps / 100.0 if rising else -(steps / 100.0))
    return json.dumps(
        {
            "timestamp": [int(value) for value in pd.date_range(start, periods=720, freq="min", tz="UTC").astype("int64")],
            "open": close.tolist(),
            "high": (close + 0.1).tolist(),
            "low": (close - 0.1).tolist(),
            "close": close.tolist(),
        }, separators=(",", ":")
    )


def test_exact_1m_output_preserves_identity_timing_cost_and_side() -> None:
    decision = pd.Timestamp("2025-02-01T01:00:00Z")
    paths = pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2025-02-01T00:00:00Z")] * 2,
            "__symbol__": ["BTC/USD:USD"] * 2,
            "side_name": ["long", "short"],
            "candidate_id": ["long-id", "short-id"],
            "execution_future_path": [_path(decision, rising=True), _path(decision, rising=False)],
        }
    )
    context = pd.DataFrame(
        {
            "candidate_id": ["long-id", "short-id"],
            "__ts__": [paths.loc[0, "__ts__"], paths.loc[1, "__ts__"]],
            "__symbol__": ["BTC/USD:USD"] * 2,
            "side_name": ["long", "short"],
            "__decision_ts__": [decision] * 2,
            "__barrier_pct__": [0.01, 0.01],
            "atr_fraction": [0.01, 0.01],
            "execution_cost_return": [0.008, 0.012],
            "execution_entry_half_spread_bps": [10.0, 10.0],
            "execution_exit_half_spread_bps": [20.0, 20.0],
            "execution_decision_utc": [decision] * 2,
            "execution_entry_price": [100.0, 100.0],
            "policy_execution_cost_return": [0.008, 0.012],
            "policy_entry_half_spread_bps": [10.0, 10.0],
            "policy_exit_half_spread_bps": [20.0, 20.0],
            "execution_geometry_key": ["long__parent", "short__parent"],
            "policy_archetype": ["parent", "parent"],
            "execution_label_end_utc": [decision + pd.Timedelta(hours=12)] * 2,
            "execution_label_available_at": [decision + pd.Timedelta(hours=12)] * 2,
        }
    ).set_index("candidate_id")
    out = _output_batch(paths, context)
    assert out["candidate_id"].tolist() == ["long-id", "short-id"]
    assert (pd.to_datetime(out["__label_end_ts__"], utc=True) == decision + pd.Timedelta(hours=12)).all()
    assert out["path_arch_complete_12h"].tolist() == [1, 1]
    assert out["__path_auxiliary_target_valid__"].tolist() == [1, 1]
    assert out["__meaningful_mfe_reached_12h__"].tolist() == [1, 1]
    # The execution-adjusted CatBoost target receives each side's entry and
    # exit spread once, while the v6 auxiliary kernel remains raw-path only.
    assert out.loc[0, "execution_cost_return"] == np.float32(0.008)
    assert out.loc[1, "execution_cost_return"] == np.float32(0.012)
    assert out.loc[0, "path_arch_cost_atr"] == np.float32(0.8)
    assert out.loc[1, "path_arch_cost_atr"] == np.float32(1.2)
    assert out["path_archetype"].notna().all()
