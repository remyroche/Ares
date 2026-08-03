from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from run_regime_transition_hazard_matched_controls import (  # noqa: E402
    MATCH_FIELDS,
    matched_training_base,
)


def _frame() -> pd.DataFrame:
    rows = 12
    return pd.DataFrame(
        {
            "source_utc": pd.date_range(
                "2026-01-01", periods=rows, freq="h", tz="UTC"
            ),
            "state_context__current_state": [0] * 6 + [1] * 6,
            **{
                field: np.linspace(0.0, 1.0, rows)
                for field in MATCH_FIELDS
            },
        }
    )


def test_matching_keeps_events_and_uses_same_state_controls() -> None:
    frame = _frame()
    event_time = np.full(len(frame), np.inf)
    event_time[[1, 7]] = [2.0, 1.0]
    chosen, representation, report = matched_training_base(
        frame,
        event_time,
        np.arange(len(frame)),
        controls_per_positive=2,
        calendar_radius_days=30,
    )
    assert {1, 7}.issubset(set(chosen))
    controls = chosen[~np.isfinite(event_time[chosen])]
    assert set(frame.iloc[controls]["state_context__current_state"]) == {0, 1}
    assert report["state_fallback_event_rows"] == 0
    assert representation[controls].sum() >= len(controls)


def test_calendar_fallback_is_explicit_when_local_pool_is_too_small() -> None:
    frame = _frame()
    frame.loc[1, "source_utc"] = pd.Timestamp("2027-01-01", tz="UTC")
    event_time = np.full(len(frame), np.inf)
    event_time[1] = 2.0
    _, _, report = matched_training_base(
        frame,
        event_time,
        np.arange(len(frame)),
        controls_per_positive=3,
        calendar_radius_days=1,
    )
    assert report["calendar_fallback_event_rows"] == 1
