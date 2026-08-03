from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_execution_ev_long_ranking import (
    _metrics,
    _select_top_fraction,
)


def test_top_fraction_respects_score_orientation() -> None:
    frame = pd.DataFrame({"score": [0.3, 0.1, 0.2, 0.4]})
    low = _select_top_fraction(
        frame, "score", higher_is_better=False, fraction=0.5
    )
    high = _select_top_fraction(
        frame, "score", higher_is_better=True, fraction=0.5
    )
    assert low.tolist() == [False, True, True, False]
    assert high.tolist() == [True, False, False, True]


def test_metrics_reconcile_exact_cost() -> None:
    frame = pd.DataFrame(
        {
            "execution_net_ev_12h": [0.01, -0.02],
            "execution_gross_ev_12h": [0.02, -0.01],
            "execution_cost_return": [0.01, 0.01],
            "execution_mfe_return_12h": [0.03, 0.01],
            "execution_mae_return_12h": [0.01, 0.02],
            "execution_exit_reason": ["timeout", "full_stop"],
        }
    )
    metrics = _metrics(
        frame,
        np.array([True, False]),
        score_name="x",
        selection_scope="global",
        report_slice="long",
    )
    assert np.isclose(metrics["mean_gross_bps"], 200.0)
    assert np.isclose(metrics["mean_cost_bps"], 100.0)
    assert np.isclose(metrics["mean_net_bps"], 100.0)
