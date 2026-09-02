from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refresh_c1_lva_15m_source", ROOT / "scripts" / "refresh_c1_lva_15m_source.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _bars(*, observed: object = True) -> pd.DataFrame:
    index = pd.date_range("2026-08-28T12:00:00Z", periods=3, freq="15min")
    return pd.DataFrame(
        {"open": [1.0, 1.0, 1.0], "high": [1.1, 1.1, 1.1], "low": [0.9, 0.9, 0.9],
         "close": [1.0, 1.0, 1.0], "volume": [1.0, 1.0, 1.0], "exchange_observed": observed},
        index=index,
    )


def test_audit_requires_exchange_observed_bar() -> None:
    result = MODULE.audit_symbol_coverage(
        bars=_bars(observed=[True, False, True]),
        start=pd.Timestamp("2026-08-28T12:00:00Z"),
        end_exclusive=pd.Timestamp("2026-08-28T12:45:00Z"),
    )
    assert result["source_complete_exchange_observed"] is False
    assert result["synthetic_or_unknown_bars"] == 1


def test_audit_accepts_exact_observed_range() -> None:
    result = MODULE.audit_symbol_coverage(
        bars=_bars(), start=pd.Timestamp("2026-08-28T12:00:00Z"),
        end_exclusive=pd.Timestamp("2026-08-28T12:45:00Z"),
    )
    assert result["source_complete_exchange_observed"] is True
    assert result["expected_bars"] == 3
