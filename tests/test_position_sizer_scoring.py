import pytest
import numpy as np

from extreme_price_movements.position_sizer_v2 import (
    aggregate_candidate_fold_metrics,
    compute_candidate_utility,
    rank_candidates_with_tiebreak,
    run_candidate_pool_sensitivity_check,
)

def test_aggregate_candidate_fold_metrics():
    res = aggregate_candidate_fold_metrics(
        fold_pnl_days=[10.0, 20.0],
        fold_sortinos=[1.5, 2.5],
        fold_maxdds=[0.1, 0.3],
        fold_timeout_rates=[0.05, 0.15],
        candidate_params={"name": "test"},
    )

    assert res["net_pnl_day"] == 15.0
    assert res["sortino"] == 2.0
    assert res["maxDD"] == 0.2
    assert res["worst_fold_maxdd"] == 0.3
    assert res["mean_fold_maxdd"] == 0.2
    assert res["instability"] == 5.0
    assert res["mean_timeout_rate"] == 0.1
    assert res["name"] == "test"

def test_compute_candidate_utility_stable():
    candidates = [
        {"net_pnl_day": 1000.0, "sortino": 2.0, "maxDD": 0.1, "worst_fold_maxdd": 0.2, "instability": 100.0, "mean_timeout_rate": 0.05},
        {"net_pnl_day": 500.0, "sortino": 1.0, "maxDD": 0.2, "worst_fold_maxdd": 0.4, "instability": 200.0, "mean_timeout_rate": 0.1},
    ]

    scored = compute_candidate_utility(candidates, mode="stable_absolute", w_pnl=1.0, w_quality=1.0, w_dd=1.0, w_instab=1.0, w_to=1.0)

    # Candidate 1:
    # pnl_score = 1.0, quality_score = 2.0, dd_penalty = 0.2, instab = 0.1, to = 0.05
    # u = 1.0 + 2.0 - 0.2 - 0.1 - 0.05 = 2.65
    assert np.isclose(scored[0]["utility_abs"], 2.65)

    # Candidate 2:
    # pnl_score = 0.5, quality_score = 1.0, dd_penalty = 0.4, instab = 0.2, to = 0.1
    # u = 0.5 + 1.0 - 0.4 - 0.2 - 0.1 = 0.8
    assert np.isclose(scored[1]["utility_abs"], 0.8)

def test_rank_candidates_with_tiebreak():
    # Candidates with very similar utility
    eps = 1e-4
    candidates = [
        {"utility": 2.0 + eps/2, "worst_fold_maxdd": 0.3, "instability": 1.0, "mean_timeout_rate": 0.1, "net_pnl_day": 10.0},
        {"utility": 2.0, "worst_fold_maxdd": 0.2, "instability": 1.0, "mean_timeout_rate": 0.1, "net_pnl_day": 10.0}, # Better DD
        {"utility": 2.0, "worst_fold_maxdd": 0.3, "instability": 0.5, "mean_timeout_rate": 0.1, "net_pnl_day": 10.0}, # Better instability
    ]

    ranked = rank_candidates_with_tiebreak(candidates, eps_utility=eps)

    # Expected order: index 1 (best dd), index 2 (better instab), index 0 (worst)
    assert ranked[0]["worst_fold_maxdd"] == 0.2
    assert ranked[1]["instability"] == 0.5
    assert ranked[2]["worst_fold_maxdd"] == 0.3

def test_sensitivity_check():
    candidates = [
        {"net_pnl_day": 10.0, "sortino": 1.0, "maxDD": 0.1, "worst_fold_maxdd": 0.1, "instability": 1.0, "mean_timeout_rate": 0.0},
        {"net_pnl_day": 20.0, "sortino": 1.1, "maxDD": 0.1, "worst_fold_maxdd": 0.1, "instability": 1.0, "mean_timeout_rate": 0.0},
        {"net_pnl_day": 30.0, "sortino": 1.2, "maxDD": 0.1, "worst_fold_maxdd": 0.1, "instability": 1.0, "mean_timeout_rate": 0.0},
        {"net_pnl_day": 40.0, "sortino": 1.3, "maxDD": 0.1, "worst_fold_maxdd": 0.1, "instability": 1.0, "mean_timeout_rate": 0.0},
        {"net_pnl_day": 50.0, "sortino": 1.4, "maxDD": 0.1, "worst_fold_maxdd": 0.1, "instability": 1.0, "mean_timeout_rate": 0.0},
        # Outlier
        {"net_pnl_day": 500.0, "sortino": 5.0, "maxDD": 0.1, "worst_fold_maxdd": 0.1, "instability": 5.0, "mean_timeout_rate": 0.0},
    ]

    res = run_candidate_pool_sensitivity_check(candidates, drop_fraction=0.20)
    assert res["status"] == "success"
    # Since stable_absolute is independent of the pool, the winner should stay the same (or safely drop if it's the outlier).
    assert "absolute_stable" in res
