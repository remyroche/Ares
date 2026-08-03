from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_frozen_hurdle_blend_ablation import _choose_weight, _select, blend


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": [3, 2, 1, 0],
            "__symbol__": ["BTC"] * 4,
            "side_name": ["long", "short", "long", "short"],
            "candidate_id": ["z", "a", "b", "c"],
            "execution_decision_utc": pd.date_range("2026-05-01", periods=4, freq="h", tz="UTC"),
            "support_label_available_utc": pd.date_range("2026-05-01 12:00", periods=4, freq="h", tz="UTC"),
            "execution_net_ev_12h": [0.04, 0.03, -0.01, -0.02],
            "oof_fold": [1, 1, 2, 2],
            "side_causal_oof_ev_direct_net_residual": [0.8, 0.7, 0.1, 0.0],
            "side_causal_oof_ev_gross_cost_hurdle_ev": [0.0, 1.0, 0.2, 0.1],
        }
    )


def test_blend_endpoints_exactly_reproduce_controls() -> None:
    direct = np.array([0.1, -0.2])
    hurdle = np.array([-0.3, 0.4])
    assert np.array_equal(blend(direct, hurdle, 0.0), direct)
    assert np.array_equal(blend(direct, hurdle, 1.0), hurdle)


def test_selection_is_candidate_id_stable_under_ties() -> None:
    frame = _frame()
    selected = _select(frame, np.array([1.0, 1.0, 0.0, 0.0]), 0.5)
    assert selected.tolist() == [1, 0]


def test_weight_choice_uses_resolved_oof_and_returns_fixed_grid_member() -> None:
    weight, table = _choose_weight(_frame(), development_cutoff=pd.Timestamp("2026-05-02T00:00:00Z"))
    assert weight in {0.0, 0.25, 0.5, 0.75, 1.0}
    assert table["selected_frozen_weight"].sum() == 1
    assert table["development_rows"].iloc[0] == 4
