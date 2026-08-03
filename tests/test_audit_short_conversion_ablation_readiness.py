from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_short_conversion_ablation_readiness.py"
SPEC = importlib.util.spec_from_file_location("short_conversion_readiness", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chronological_support_requires_both_target_classes_and_day_blocks() -> None:
    count = 1_001
    ts = pd.date_range("2025-03-01", periods=count, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "side_name": ["short"] * (2 * count), "__ts__": list(ts) + list(ts),
        "fold_id": ["fold"] * (2 * count), "fold_validation_start_utc": [ts[0]] * (2 * count), "fold_validation_end_utc": [ts[-1]] * (2 * count),
        "execution_net_ev_12h": np.r_[np.full(count, .01), np.full(count, -.01)],
    })
    result = MODULE.chronological_support(frame, scope="test")
    assert result.iloc[0].support_pass
    assert result.iloc[0].net_positive_rows_for_probability_and_favourable_magnitude == count
    assert result.iloc[0].net_nonpositive_rows_for_adverse_severity == count


def test_recommendation_grid_is_fixed_and_modest() -> None:
    assert MODULE.MIN_POSITIVE_ROWS == MODULE.MIN_NONPOSITIVE_ROWS == 1_000
    assert MODULE.MIN_POSITIVE_DAY_BLOCKS == MODULE.MIN_NONPOSITIVE_DAY_BLOCKS == 20
