from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.ablate_side_archetype_ev_portfolio_matrix import (
    _portfolio_replay,
    _trimmed_correction,
)


def test_symmetric_daily_trim_removes_both_residual_tails() -> None:
    stats = pd.DataFrame(
        {
            "sum": [-10.0, 0.0, 0.0, 0.0, 10.0],
            "count": [1, 1, 1, 1, 1],
            "mean": [-10.0, 0.0, 0.0, 0.0, 10.0],
        }
    )

    correction, support, retained_days, _ = _trimmed_correction(stats, 0.20)

    assert correction == 0.0
    assert support == 3
    assert retained_days == 3


def test_portfolio_replay_limits_entries_capacity_and_symbol_overlap() -> None:
    timestamps = pd.to_datetime(
        ["2026-06-01T00:00:00Z"] * 4 + ["2026-06-01T01:00:00Z"] * 4,
        utc=True,
    )
    source = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A", "A", "B", "C", "A", "D", "E", "F"],
            "ev_after_1pct": [0.01] * 8,
            "policy_parent_rank": np.linspace(0.9, 0.8, 8),
        }
    )
    corrected = np.linspace(0.02, 0.01, 8, dtype=np.float32)

    selected = _portfolio_replay(
        source,
        corrected,
        target_ev=0.0,
        max_new_entries_per_bar=2,
        max_concurrent_positions=3,
        outcome_horizon_hours=12,
    )

    chosen = source.iloc[selected]
    assert chosen.groupby("__ts__").size().max() <= 2
    assert len(chosen) == 3
    assert chosen["__symbol__"].nunique() == 3
