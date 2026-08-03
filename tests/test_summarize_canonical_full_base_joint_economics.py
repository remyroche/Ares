import numpy as np
import pandas as pd

from scripts import summarize_canonical_full_base_joint_economics as summary


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["z", "a", "b", "c"],
            "side_name": ["long", "short", "long", "short"],
            "__symbol__": ["BTC"] * 4,
            "__ts__": pd.date_range("2025-04-01", periods=4, freq="h", tz="UTC"),
            "execution_gross_ev_12h": [0.03, 0.02, 0.01, 0.00],
            "execution_cost_return": [0.01] * 4,
            "execution_net_ev_12h": [0.02, 0.01, 0.00, -0.01],
            "opportunity_gross_above_cost_0bps": [True, True, False, False],
            "opportunity_gross_above_cost_25bps": [True, True, False, False],
        }
    )


def test_global_top_mask_uses_candidate_id_for_ties():
    rows = _rows()
    selected = summary.stable_global_top_mask(rows, [1.0, 1.0, 0.0, -1.0], 0.25)
    assert rows.loc[selected, "candidate_id"].tolist() == ["a"]


def test_selected_metrics_reconciles_gross_cost_net():
    metrics = summary.selected_metrics(_rows().iloc[:2])
    assert metrics["mean_gross_bps"] - metrics["mean_cost_bps"] == metrics["mean_net_bps"]
    assert metrics["long_share"] == 0.5
    assert metrics["short_share"] == 0.5


def test_promotion_gate_requires_all_conditions():
    tails = pd.DataFrame(
        [
            {
                "split": "april_reused_diagnostic",
                "slice_kind": "overall",
                "slice_value": "all",
                "arm": "S0",
                "score_name": "direct_primary",
                "fraction": 0.10,
                "mean_net_bps": -1.0,
                "long_share": 0.5,
                "short_share": 0.5,
            },
            {
                "split": "april_reused_diagnostic",
                "slice_kind": "latest_week",
                "slice_value": "2025-04-24T00:00:00+00:00",
                "arm": "S0",
                "score_name": "direct_primary",
                "fraction": 0.10,
                "mean_net_bps": 2.0,
                "long_share": 0.5,
                "short_share": 0.5,
            },
        ]
    )
    gate = summary.promotion_gates(tails)[0]
    assert not gate["global_top10_positive"]
    assert not gate["eligible_for_portfolio_replay"]
