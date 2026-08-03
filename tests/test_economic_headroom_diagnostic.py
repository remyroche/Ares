from __future__ import annotations

import pandas as pd

from extreme_price_movements.economic_headroom_diagnostic import model_headroom, oracle_headroom, ranking_diagnostics


def test_oracle_tail_can_be_positive_when_model_tail_is_not() -> None:
    primary = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "side": ["long", "long", "short", "short"],
        "execution_exact_h12_gross_bps": [300.0, -50.0, 250.0, -100.0],
        "execution_exact_h12_cost_bps": [100.0] * 4,
        "execution_exact_h12_net_bps": [200.0, -150.0, 150.0, -200.0],
    })
    result = oracle_headroom(primary)
    top = result[(result.scope == "all") & (result.oracle_score == "net") & (result.fraction == 0.1)]
    assert len(top) == 1
    assert top.iloc[0].net_bps > 0


def test_model_cost_sensitivity_exposes_pre_cost_ranking_failure() -> None:
    metrics = pd.DataFrame({
        "scope": ["pooled_global_top"], "fraction": [0.10], "arm": ["A"],
        "selected_rows": [1], "gross_bps": [-5.0], "cost_bps": [100.0], "net_bps": [-105.0],
    })
    result, top10 = model_headroom(metrics)
    assert top10.iloc[0]["net_at_cost_0bps"] == -5.0
    assert top10.iloc[0]["net_at_cost_100bps"] == -105.0


def test_ranking_diagnostic_reports_true_tail_recall() -> None:
    results = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"], "arm": ["A"] * 4,
        "raw_score": [0.9, 0.8, 0.2, 0.1],
        "calibrated_expected_net_bps": [0.9, 0.8, 0.2, 0.1],
        "exact_h12_gross_bps": [0.0, 0.0, 0.0, 0.0],
        "exact_h12_net_bps": [10.0, -1.0, 20.0, -2.0],
    })
    out = ranking_diagnostics(results)
    assert out.iloc[0]["top10_oracle_recall"] == 0.0
    assert out.iloc[0]["spearman_score_net"] < 0.5


def test_ranking_diagnostic_preserves_stable_ties_like_policy_selection() -> None:
    results = pd.DataFrame({
        "candidate_id": ["first", "second", "third", "fourth"], "arm": ["A"] * 4,
        "calibrated_expected_net_bps": [10.0, 10.0, 0.0, 0.0],
        "exact_h12_gross_bps": [40.0, -40.0, 0.0, 0.0],
        "exact_h12_net_bps": [-60.0, -140.0, -100.0, -100.0],
    })
    out = ranking_diagnostics(results)
    # The 10% diagnostic on four rows selects one row.  The two equal mapped
    # scores must retain materialized row order, so ``first`` is selected.
    assert out.iloc[0]["model_top10_rows"] == 1
    assert out.iloc[0]["top10_net_bps"] == -60.0
