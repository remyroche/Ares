import pandas as pd
import pytest

from scripts.run_canonical_base_ic_ev_tail_diagnostic import (
    decile_monotonicity,
    stable_top,
)


def test_stable_top_is_global_and_candidate_id_breaks_score_ties():
    frame = pd.DataFrame(
        {
            "candidate_id": ["z", "b", "a", "x"],
            "score_raw": [0.1, 0.9, 0.9, 0.2],
        }
    )
    selected = stable_top(frame, 0.50)
    assert selected["candidate_id"].tolist() == ["a", "b"]


def test_decile_monotonicity_reports_no_violations_for_increasing_economics():
    rows = 100
    frame = pd.DataFrame(
        {
            "candidate_month": ["2025-02"] * rows,
            "side_name": ["long"] * rows,
            "score_raw": list(range(rows)),
            "execution_mfe_return_12h": list(range(rows)),
            "execution_gross_ev_12h": list(range(rows)),
            "execution_net_ev_12h": list(range(rows)),
            "opportunity_gross_above_cost_0bps": [index >= 50 for index in range(rows)],
            "opportunity_gross_above_cost_25bps": [index >= 60 for index in range(rows)],
            "exit_is_trailing": [index >= 50 for index in range(rows)],
            "exit_is_timeout": [False] * rows,
            "exit_is_full_stop": [False] * rows,
            "exit_is_adverse_exit": [index < 50 for index in range(rows)],
        }
    )
    _, summary = decile_monotonicity(frame)
    pooled = summary.loc[summary["scope"].eq("pooled_global")].iloc[0]
    assert pooled["execution_net_ev_12h__adjacent_violations"] == 0
    assert pooled["execution_net_ev_12h__decile_spearman"] == pytest.approx(1.0)
