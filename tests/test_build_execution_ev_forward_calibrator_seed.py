from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_execution_ev_forward_calibrator_seed import (
    interaction_score,
    select_seed_history,
)


def _contract() -> dict[str, float]:
    return {
        "direction": 1.0,
        "threshold": 0.5,
        "robust_scale": 1.0,
        "interaction_weight": 0.25,
    }


def test_interaction_reuses_frozen_standardization() -> None:
    direct = np.array([-1.0, 0.0, 1.0])
    capture = np.array([0.1, 0.5, 0.9])
    margin = np.array([0.0, 0.5, 1.0])
    first, state = interaction_score(
        direct,
        capture,
        margin,
        contract=_contract(),
    )
    second, replay = interaction_score(
        direct,
        capture,
        margin,
        contract=_contract(),
        direct_center=state["direct_center"],
        direct_scale=state["direct_scale"],
        capture_center=state["capture_center"],
        capture_scale=state["capture_scale"],
    )
    assert replay == state
    assert second == pytest.approx(first)


def test_seed_history_uses_resolution_time_not_decision_time() -> None:
    frame = pd.DataFrame(
        {
            "execution_decision_utc": pd.to_datetime(
                ["2026-07-20", "2026-07-27"], utc=True
            ),
            "execution_label_end_utc": pd.to_datetime(
                ["2026-07-20T12:00:00Z", "2026-07-28T01:00:00Z"], utc=True
            ),
        }
    )
    selected = select_seed_history(
        frame,
        first_decision_exclusive=pd.Timestamp("2026-07-28T00:00:00Z"),
        lookback_days=21,
    )
    assert selected.index.tolist() == [0]


def test_seed_history_rejects_naive_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "execution_label_end_utc": pd.to_datetime(
                ["2026-07-20T12:00:00Z"], utc=True
            )
        }
    )
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        select_seed_history(
            frame,
            first_decision_exclusive=pd.Timestamp("2026-07-28"),
            lookback_days=21,
        )
