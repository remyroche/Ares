from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_exact_history_state_recurrence import (
    recurrence_gate,
    select_one_global_topk,
)


def test_global_topk_is_not_reranked_by_side_or_state() -> None:
    frame = pd.DataFrame(
        {
            "side_name": ["long"] * 5 + ["short"] * 5,
            "state_id": [0, 1] * 5,
            "score": np.arange(10, dtype=float),
        }
    )
    selected = select_one_global_topk(frame, "score", fraction=0.20)
    assert selected.sum() == 2
    assert frame.loc[selected, "score"].tolist() == [8.0, 9.0]


def test_recurrence_gate_requires_history_recent_support_and_july_sign() -> None:
    rows = []
    for week in ("h1", "h2", "h3"):
        rows.append(
            {
                "side_name": "long",
                "state_id": 0,
                "week": week,
                "era": "historical",
                "selected_rows": 120,
                "selected_mean_net_bps": 10.0,
            }
        )
    for week in ("m1", "m2"):
        rows.append(
            {
                "side_name": "long",
                "state_id": 0,
                "week": week,
                "era": "may_june",
                "selected_rows": 120,
                "selected_mean_net_bps": 5.0,
            }
        )
    rows.append(
        {
            "side_name": "long",
            "state_id": 0,
            "week": "j1",
            "era": "july",
            "selected_rows": 120,
            "selected_mean_net_bps": 2.0,
        }
    )
    gate = recurrence_gate(
        pd.DataFrame(rows),
        minimum_rows=100,
        minimum_historical_weeks=3,
        minimum_recent_weeks=2,
        minimum_sign_consistency=0.75,
    )
    assert gate.loc[0, "eligible_recurring_state"]

    reversed_july = pd.DataFrame(rows)
    reversed_july.loc[reversed_july["era"].eq("july"), "selected_mean_net_bps"] = -2.0
    rejected = recurrence_gate(
        reversed_july,
        minimum_rows=100,
        minimum_historical_weeks=3,
        minimum_recent_weeks=2,
        minimum_sign_consistency=0.75,
    )
    assert not rejected.loc[0, "eligible_recurring_state"]
