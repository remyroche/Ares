import pandas as pd

from extreme_price_movements.residual_lambdarank_hpo import (
    adjusted_hpo_score, complexity_penalty, era_portability_summary,
    conditional_downstream_summary, passes_conditional_promotion,
    make_pruned_study, materialize_lambdarank_params, portability_score,
    restore_broad_lambdarank_params, truncation_candidates,
    select_portability_winner,
)


def test_portability_penalizes_a_negative_worst_era():
    assert portability_score([10.0, 10.0, 10.0]) == 10.0
    assert portability_score([-10.0, 10.0, 10.0]) == 0.0


def test_complexity_penalty_only_applies_above_preferred_capacity():
    assert complexity_penalty(max_depth=4, num_leaves=15) == 0.0
    assert complexity_penalty(max_depth=5, num_leaves=30) > 0.0
    assert adjusted_hpo_score(era_evs=[10.0, 10.0], max_depth=5, num_leaves=30) < 10.0


def test_truncation_is_small_and_geometry_aware():
    values = truncation_candidates(retained_fraction=.05, median_candidates_per_query=40)
    assert values == sorted(set(values))
    assert all(3 <= value <= 32 for value in values)


def test_hpo_study_uses_median_pruning():
    assert type(make_pruned_study(seed=3).pruner).__name__ == "MedianPruner"


def test_fold_fraction_is_materialized_without_mutating_the_trial_contract():
    suggested = {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 2000,
        "min_child_samples_fraction": .01, "max_depth": 4, "num_leaves": 15,
    }
    params = materialize_lambdarank_params(suggested, training_rows=12_345)
    assert params["min_child_samples"] == 124
    assert "min_child_samples_fraction" not in params
    assert suggested["min_child_samples_fraction"] == .01


def test_restore_broad_trial_removes_search_only_switches():
    restored = restore_broad_lambdarank_params({
        "label_gain": "moderate_tail", "max_depth": 4, "num_leaves": 15,
        "min_data_in_leaf": .01, "min_sum_hessian_in_leaf": 1.0,
        "min_gain_to_split_zero": True, "feature_fraction": .8,
        "bagging_fraction": .8, "lambda_l1_zero": True,
        "lambda_l2": 2.0, "max_bin": 63,
        "lambdarank_truncation_level": 5,
    })
    assert restored["min_child_samples_fraction"] == .01
    assert restored["min_gain_to_split"] == 0.0
    assert restored["lambda_l1"] == 0.0
    assert restored["label_gain_name"] == "moderate_tail"


def test_era_summary_exposes_the_same_portability_terms():
    summary = era_portability_summary([10.0, -5.0, 20.0])
    assert summary["era_count"] == 3
    assert summary["era_ev_worst_bps"] == -5.0
    assert summary["portability_score_bps"] == portability_score([10.0, -5.0, 20.0])


def test_portability_tie_break_uses_monthly_stability_before_top1():
    table = pd.DataFrame({
        "arm": ["volatile", "stable"],
        "adjusted_hpo_score": [10.0, 9.4],
        "month_mad_net_bps": [20.0, 5.0],
        "month_worst_net_bps": [-5.0, -10.0],
        "top1_net_bps": [100.0, 1.0],
    })
    assert select_portability_winner(table, tie_tolerance_bps=1.0)["arm"] == "stable"


def test_conditional_downstream_utility_is_global_and_exposes_strict_gate_terms():
    frame = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(200)],
        "__ts__": pd.to_datetime(["2025-05-01"] * 100 + ["2025-06-01"] * 100, utc=True),
        "net_bps": list(range(-100, 100)),
        "gross_bps": list(range(0, 200)),
        "incumbent": list(range(200, 0, -1)),
        "candidate": list(range(200)),
    })
    summary = conditional_downstream_summary(
        frame, candidate_score_column="candidate", incumbent_score_column="incumbent",
    )
    assert summary["conditional_rows"] == 200
    assert set(("delta_top1_net_bps", "delta_top2_net_bps", "delta_top5_net_bps", "delta_top5_month_worst_net_bps")).issubset(summary)
    assert passes_conditional_promotion(summary)
