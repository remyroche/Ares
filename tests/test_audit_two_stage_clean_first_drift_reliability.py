from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_two_stage_clean_first_drift_reliability import _psi, apply_reliability_rule, select_reliability_rule


def test_psi_is_zero_for_identical_distributions() -> None:
    values = np.arange(100, dtype=float)
    assert _psi(values, values) == 0.0


def _history() -> pd.DataFrame:
    rows = []
    for month in ("2026-05", "2026-06", "2026-07"):
        for i in range(100):
            rows.append({"candidate_id": f"{month}-{i}", "side_name": "short", "__ts__": pd.Timestamp(f"{month}-01", tz="UTC"), "catboost_hard_clean_first__probability": .9 - i / 200, "catboost_adverse_1atr_gate__probability": i / 100, "raw_ood_fraction": i / 100, "execution_net_ev_12h": .01 if i < 15 else -.01})
    return pd.DataFrame(rows)


def test_rule_is_selected_on_history_only_and_respects_coverage_floor() -> None:
    state, table = select_reliability_rule(_history())
    assert state["history_only"]
    assert state["rule_id"] in set(table.rule_id)
    assert table.loc[table.rule_id.eq(state["rule_id"]), "minimum_month_coverage"].iloc[0] >= .70


def test_apply_rule_is_short_only_and_never_creates_admissions() -> None:
    frame = pd.DataFrame({"side_name": ["short", "long"], "catboost_adverse_1atr_gate__probability": [.99, .99], "catboost_hard_clean_first__probability": [.5, .5], "raw_ood_fraction": [0., 0.], "catboost_hard_clean_first__historical_admission": [True, True]})
    result = apply_reliability_rule(frame, {"rule_id": "short_adverse_high", "thresholds": {"adverse_q80": .8, "clean_q20": .2, "ood_q80": .8}})
    assert result.frozen_reliability_abstain.tolist() == [True, False]
    assert result.frozen_clean_first_retained.tolist() == [False, True]
