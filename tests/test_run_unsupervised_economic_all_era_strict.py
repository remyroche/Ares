import pandas as pd

from scripts.run_unsupervised_economic_all_era_strict import ARMS, _fixed_global_metrics


def test_strict_all_era_arms_exclude_nonidentical_diagonal_and_failure_context() -> None:
    assert set(ARMS) == {"baseline", "sticky_fullcov_gmm_geometry", "dae_to_gmm_geometry"}
    assert "sticky_ood_score" in ARMS["sticky_fullcov_gmm_geometry"]
    assert "dae_reconstruction_error" in ARMS["dae_to_gmm_geometry"]
    assert all("failure" not in field for fields in ARMS.values() for field in fields)


def test_period_metrics_decompose_one_fixed_global_book_without_period_reranking() -> None:
    rows = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "side_name": ["long"] * 10,
            "__symbol__": ["X"] * 10,
            "__ts__": pd.to_datetime(["2026-05-01T00:00Z"] * 9 + ["2026-06-01T00:00Z"]),
            "mapped_score": list(range(10, 0, -1)),
            "__first_touch_target_soft__": list(range(10)),
            "execution_net_ev_12h": [.01] * 10,
            "execution_gross_ev_12h": [.02] * 10,
            "execution_cost_return": [.01] * 10,
        }
    )
    summary, periods, _, _ = _fixed_global_metrics(rows, "baseline")
    assert summary["top10_support"] == 1
    monthly = periods.loc[periods.period_type.eq("month")]
    assert monthly.global_selected_rows.sum() == 1
    assert monthly.loc[monthly.period.eq("2026-06"), "global_selected_rows"].iloc[0] == 0
