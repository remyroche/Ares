from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_strict_r3_p8u_dual_mc1_prequential_v2.py"
    spec = importlib.util.spec_from_file_location("p8u_dual_mc1_rollover_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _six_month_score_panel() -> pd.DataFrame:
    start = pd.Timestamp("2026-03-01T00:00:00Z")
    end = pd.Timestamp("2026-09-01T00:00:00Z")
    return pd.DataFrame({"__decision_ts__": pd.date_range(start, end - pd.Timedelta(hours=1), freq="h", tz="UTC")})


def test_rollover_accepts_isolated_historical_hour_gap() -> None:
    runner = _runner_module()
    frame = _six_month_score_panel()
    # A preserved missing source hour is allowed; it is not reconstructed.
    frame = frame.loc[frame["__decision_ts__"].ne(pd.Timestamp("2026-06-15T16:00:00Z"))]

    runner._assert_full_train_coverage(
        frame, train_start=pd.Timestamp("2026-03-01T00:00:00Z"),
        held_start=pd.Timestamp("2026-09-01T00:00:00Z"),
    )


def test_rollover_rejects_partial_final_training_month() -> None:
    runner = _runner_module()
    frame = _six_month_score_panel()
    # The August partition still exists, but its final days are absent.  This
    # must fail rather than produce a nominally six-month September package.
    frame = frame.loc[frame["__decision_ts__"].lt(pd.Timestamp("2026-08-29T00:00:00Z"))]

    with pytest.raises(AssertionError, match="complete-enough"):
        runner._assert_full_train_coverage(
            frame, train_start=pd.Timestamp("2026-03-01T00:00:00Z"),
            held_start=pd.Timestamp("2026-09-01T00:00:00Z"),
        )
