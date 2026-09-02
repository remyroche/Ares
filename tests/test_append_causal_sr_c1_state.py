from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "append_causal_sr_c1_state", ROOT / "scripts" / "append_causal_sr_c1_state.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _bars(observed=True) -> pd.DataFrame:
    index = pd.date_range("2026-08-28T12:00:00Z", periods=4, freq="15min")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": .9, "close": 1.0, "volume": 1.0, "exchange_observed": observed},
        index=index,
    )


def test_fresh_observed_bars_returns_only_suffix() -> None:
    actual = MODULE.fresh_observed_bars(
        bars=_bars(), after=pd.Timestamp("2026-08-28T12:00:00Z"),
        through=pd.Timestamp("2026-08-28T12:45:00Z"),
    )
    assert list(actual.index) == list(pd.date_range("2026-08-28T12:15:00Z", periods=3, freq="15min"))


def test_fresh_observed_bars_rejects_synthetic_suffix() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        MODULE.fresh_observed_bars(
            bars=_bars(observed=[True, True, False, True]),
            after=pd.Timestamp("2026-08-28T12:00:00Z"),
            through=pd.Timestamp("2026-08-28T12:45:00Z"),
        )
