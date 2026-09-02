from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_strict_r3_p8u_regular_feature_state_snapshot.py"
SPEC = importlib.util.spec_from_file_location("p8u_regular_snapshot_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _panel() -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    stamp = pd.Timestamp("2026-08-30T09:00:00Z")
    symbols = tuple(f"S{index:03d}" for index in range(160))
    index = pd.date_range(stamp - pd.Timedelta(hours=1), stamp, freq="h", tz="UTC")
    values = np.ones((2, len(symbols)), dtype=np.float32)
    return {
        "close": pd.DataFrame(values, index=index, columns=symbols),
        "volume": pd.DataFrame(values * 2, index=index, columns=symbols),
    }, stamp


def test_one_row_regular_snapshot_input_has_no_retained_history() -> None:
    panel, stamp = _panel()
    result = MODULE._one_row(panel, signal=stamp)
    assert all(len(value) == 1 for value in result.values() if isinstance(value, pd.DataFrame))
    assert result["close"].index.tolist() == [stamp]


def test_one_row_regular_snapshot_refuses_missing_source_field() -> None:
    panel, stamp = _panel()
    panel["volume"] = panel["volume"].iloc[:1]
    with pytest.raises(ValueError, match="lacks the signal row"):
        MODULE._one_row(panel, signal=stamp)
