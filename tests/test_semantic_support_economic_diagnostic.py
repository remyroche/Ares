from __future__ import annotations

import pandas as pd

from scripts.run_semantic_support_economic_diagnostic import _bootstrap, _book_metrics


def test_day_bootstrap_uses_nonempty_utc_blocks() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "__ts__": pd.to_datetime([
                "2024-01-01 00:00Z", "2024-01-01 01:00Z",
                "2024-01-02 00:00Z", "2024-01-02 01:00Z",
            ], utc=True),
            "execution_net_ev_12h": [0.01, 0.02, -0.01, 0.03],
            "execution_gross_ev_12h": [0.02, 0.03, 0.0, 0.04],
            "execution_cost_return": [0.01, 0.01, 0.01, 0.01],
        }
    )
    result = _bootstrap(frame, reps=50, seed=7)
    assert result["utc_day_blocks"] == 2
    assert result["replicates"] == 50
    assert result["ci05_net_bps"] < result["ci95_net_bps"]
    metrics = _book_metrics(frame)
    assert metrics["selected_rows"] == 4
    assert metrics["months_selected"] == 1
