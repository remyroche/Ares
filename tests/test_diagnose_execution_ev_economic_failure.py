from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_execution_ev_economic_failure import (
    IDENTITY,
    load_exact_population,
    period_local_global_topk_metrics,
    score_arm_metrics,
    score_pair_drift,
    selection_pair_metrics,
    sliced_topk_metrics,
)


def _frame() -> pd.DataFrame:
    rows = 20
    decision = pd.date_range("2026-07-01", periods=rows, freq="12h", tz="UTC")
    net = np.linspace(-0.02, 0.02, rows)
    return pd.DataFrame(
        {
            "__ts__": decision,
            "__symbol__": [f"S{i % 4}" for i in range(rows)],
            "side_name": np.where(np.arange(rows) % 2, "short", "long"),
            "candidate_id": [f"c{i}" for i in range(rows)],
            "execution_decision_utc": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
            "execution_net_ev_12h": net,
            "execution_gross_ev_12h": net + 0.01,
            "execution_mfe_return_12h": np.linspace(0.001, 0.04, rows),
            "execution_mae_return_12h": np.linspace(0.04, 0.001, rows),
            "execution_cost_return": 0.01,
            "execution_exit_reason": np.where(np.arange(rows) % 3, "trailing", "timeout"),
            "raw": np.linspace(1.0, 0.0, rows),
            "mapped": np.linspace(0.0, 1.0, rows),
            "evaluation_origin": "forward",
        }
    )


def test_global_topk_is_pooled_and_slices_do_not_rerank() -> None:
    frame = _frame()
    metrics, selections = score_arm_metrics(
        frame, ["raw", "mapped"], top_k_fraction=0.10
    )
    top = metrics.loc[metrics["scope"].eq("pooled_global_topk")].set_index("score_arm")
    assert top.loc["mapped", "rows"] == 2
    assert top.loc["mapped", "mean_net_ev_bps"] > 0
    assert top.loc["raw", "mean_net_ev_bps"] < 0

    slices = sliced_topk_metrics(frame, ["mapped"], top_k_fraction=0.10)
    side = slices.loc[slices["slice"].eq("side_name")]
    assert side["globally_selected_rows"].sum() == 2
    assert selections["mapped"].sum() == 2
    period = period_local_global_topk_metrics(
        frame, ["mapped"], top_k_fraction=0.10
    )
    months = period.loc[period["period"].eq("month")]
    assert months["selected_rows"].sum() == 2


def test_ascending_rank_orientation_is_explicit() -> None:
    frame = _frame()
    metrics, selections = score_arm_metrics(
        frame,
        ["raw"],
        top_k_fraction=0.10,
        lower_is_better=["raw"],
        non_return_unit=["raw"],
    )
    top = metrics.loc[metrics["scope"].eq("pooled_global_topk")].iloc[0]
    assert top["score_orientation"] == "lower_is_better"
    assert not top["return_unit_score"]
    assert np.isnan(top["score_net_mae_bps"])
    assert selections["raw"].sum() == 2


def test_mapping_pair_reports_changed_selection_economics() -> None:
    frame = _frame()
    _, selections = score_arm_metrics(frame, ["raw", "mapped"], top_k_fraction=0.20)
    pairs = selection_pair_metrics(frame, selections, [("raw", "mapped")]).iloc[0]
    assert pairs["intersection_rows"] == 0
    assert pairs["jaccard"] == 0.0
    assert pairs["mapped_minus_raw_topk_net_ev_bps"] > 0
    drift = score_pair_drift(frame, [("raw", "mapped")])
    overall = drift.loc[drift["slice"].eq("overall")].iloc[0]
    assert overall["score_spearman"] == pytest.approx(-1.0)
    assert overall["mean_abs_rank_percentile_delta"] > 0


def test_loader_rejects_conflicting_duplicate_outcomes(tmp_path) -> None:
    frame = _frame()
    ledger = tmp_path / "ledger.parquet"
    outcome = tmp_path / "outcome.parquet"
    frame.loc[:, [*IDENTITY, "raw"]].to_parquet(ledger, index=False)
    duplicated = pd.concat([frame, frame.iloc[[0]].assign(execution_net_ev_12h=9.0)])
    duplicated.drop(columns=["raw", "mapped", "evaluation_origin"]).to_parquet(
        outcome, index=False
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_exact_population(ledger, [outcome], ["raw"])
