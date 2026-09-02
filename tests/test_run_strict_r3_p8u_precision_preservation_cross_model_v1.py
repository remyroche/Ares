from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_strict_r3_p8u_precision_preservation_cross_model_v1.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p8u_precision_preservation_cross_model", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cross_model_families_fit_a_tiny_query_safe_panel() -> None:
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    train = pd.DataFrame({"__decision_ts__": np.repeat(timestamps, 5)})
    rng = np.random.default_rng(1729)
    x_train = rng.normal(size=(len(train), 4)).astype(np.float32)
    labels = np.tile(np.arange(5, dtype=np.int8), len(timestamps))
    held = rng.normal(size=(10, 4)).astype(np.float32)
    arm = MODULE.stage1.ARMS[2]
    for family in MODULE.MODEL_FAMILIES:
        values = MODULE._fit_predict(
            candidate=MODULE.Candidate(arm, "g3_clipped_economic", family), x_train=x_train,
            y=labels, train=train, x_held=held, seed=1729,
        )
        assert values.shape == (len(held),)
        assert np.isfinite(values).all()


def test_query_ids_are_timestamp_local_and_opaque() -> None:
    frame = pd.DataFrame({"__decision_ts__": pd.to_datetime([
        "2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z",
    ])})
    assert MODULE._qid(frame).tolist() == [1, 0, 1]
