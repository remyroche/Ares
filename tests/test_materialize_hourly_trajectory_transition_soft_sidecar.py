from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "trajectory_sidecar", ROOT / "scripts/materialize_hourly_trajectory_transition_soft_sidecar.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_trailing_features_match_exclusive_anchor_window() -> None:
    times = pd.date_range("2025-01-01", periods=180, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_utc": times,
            "source_segment_id": 1,
            "target__pooled_state": 0,
        }
    )
    for offset, field in enumerate(MODULE.BASE_SIGNALS):
        frame[field] = np.arange(len(frame), dtype=float) + offset
    result = MODULE.trailing_features(frame, MODULE.fields())
    row = result.iloc[169]  # window is hours 1..168, excluding anchor 169
    field = "sequence__breadth_dispersion__mean_168h"
    delta = "sequence__breadth_dispersion__delta_168h"
    assert row[field] == np.mean(np.arange(1, 169, dtype=float))
    assert row[delta] == 167.0
    assert pd.isna(result.iloc[167][field])  # incomplete 168h history fails closed


def test_entropy_is_highest_at_half_probability() -> None:
    values = MODULE.entropy(np.array([0.01, 0.5, 0.99]))
    assert values[1] > values[0]
    assert values[1] > values[2]
