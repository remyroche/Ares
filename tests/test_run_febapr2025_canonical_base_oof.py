from __future__ import annotations

import pandas as pd

from scripts.run_febapr2025_canonical_base_oof import _deterministic_cap, _folds


def test_canonical_monthly_calendar_is_fixed() -> None:
    assert [(start.strftime("%Y_%m"), end.strftime("%Y_%m")) for start, end in _folds(pd.DataFrame())] == [
        ("2025_02", "2025_03"),
        ("2025_03", "2025_04"),
        ("2025_04", "2025_05"),
    ]


def test_deterministic_cap_preserves_calendar_month_support() -> None:
    rows = []
    for month, count in (("2025-01-01", 8), ("2025-02-01", 12), ("2025-03-01", 20)):
        for number in range(count):
            rows.append({"candidate_id": f"{month}-{number}", "__ts__": pd.Timestamp(month, tz="UTC")})
    frame = pd.DataFrame(rows)
    first = _deterministic_cap(frame, 20)
    second = _deterministic_cap(frame, 20)
    assert first["candidate_id"].tolist() == second["candidate_id"].tolist()
    assert set(first["__ts__"].dt.month) == {1, 2, 3}
    assert len(first) == 20
