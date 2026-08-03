import numpy as np
import pandas as pd

from scripts.diagnose_execution_ev_oracle_ceiling import (
    annotate_spread,
    opportunity_rows,
    oracle_topk_rows,
    variable_admission_rows,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z", "2026-06-01T00:00:00Z"]
            ),
            "__symbol__": ["AAA_USD:USD", "BBB/USD:USD", "AAA_USD:USD"],
            "side_name": ["long", "short", "long"],
            "candidate_id": ["a", "b", "c"],
            "execution_gross_ev_12h": [0.015, 0.008, 0.040],
            "execution_cost_return": [0.010, 0.010, 0.010],
            "execution_net_ev_12h": [0.005, -0.002, 0.030],
            "period_month": ["2026-05", "2026-05", "2026-06"],
            "period_week": ["2026-04-27/2026-05-03", "2026-04-27/2026-05-03", "2026-06-01/2026-06-07"],
        }
    )


def test_spread_annotation_uses_inference_normalization_and_fails_closed() -> None:
    baseline = pd.DataFrame(
        {
            "symbol_norm": ["AAA/USD:USD", "BBB/USD:USD"],
            "baseline_average_spread_bps": [5.0, 100.0],
            "inference_eligible": [True, False],
        }
    )
    actual = annotate_spread(_frame(), baseline)
    assert actual["inference_eligible"].tolist() == [True, False, True]
    assert actual["spread_bucket"].tolist() == ["<=10", "70-150", "<=10"]


def test_opportunity_and_oracle_metrics_keep_global_book_and_margin_contract() -> None:
    frame = _frame()
    frame["spread_bucket"] = pd.Series(["<=10", "70-150", "<=10"], dtype="string")
    opportunity = pd.DataFrame(opportunity_rows(frame, panel="test", universe="all"))
    overall_25 = opportunity.loc[
        (opportunity["grouping"] == "overall") & (opportunity["margin_bps"] == 25.0)
    ].iloc[0]
    # 50 and 300 bps net clear a 25 bps post-cost margin; -20 bps does not.
    assert overall_25["gross_above_cost_plus_margin_rows"] == 2
    assert overall_25["gross_above_cost_plus_margin_rate"] == 2 / 3
    assert overall_25["net_above_margin_rate"] == 2 / 3

    topk = pd.DataFrame(oracle_topk_rows(frame, panel="test", universe="all"))
    global_top10 = topk.loc[
        (topk["selection_scope"] == "one_global_book") & (topk["top_fraction"] == 0.10)
    ].iloc[0]
    assert global_top10["selected_rows"] == 1
    assert np.isclose(global_top10["mean_net_bps"], 300.0)

    admitted = pd.DataFrame(variable_admission_rows(frame, panel="test", universe="all"))
    assert admitted.loc[admitted["margin_bps"] == 50.0, "admitted_rows"].iloc[0] == 1
