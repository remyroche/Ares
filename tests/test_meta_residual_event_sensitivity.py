from __future__ import annotations

import pandas as pd

from scripts.report_meta_residual_event_sensitivity import _event_ids


def test_event_ids_respect_group_and_gap() -> None:
    ts = pd.Series(
        pd.to_datetime(
            [
                "2026-01-01T00:00Z",
                "2026-01-01T01:00Z",
                "2026-01-01T05:00Z",
                "2026-01-01T00:00Z",
            ]
        )
    )
    side = pd.Series(["long", "long", "long", "short"])
    arch = pd.Series(["a", "a", "a", "a"])
    active = pd.Series([True, True, True, True])
    result = _event_ids(ts, side, arch, active, max_gap_hours=2)
    assert result.nunique() == 3
    assert result.iloc[0] == result.iloc[1]
    assert result.iloc[1] != result.iloc[2]
    assert result.iloc[0] != result.iloc[3]
