from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_meaningful_mfe_exact_grid_causal_mapping import (
    evaluate_score,
    tail_row,
)


def _rows(
    count: int,
    *,
    start: str,
    partition: str,
    candidate_prefix: str,
) -> pd.DataFrame:
    decision = pd.date_range(start, periods=count, freq="min", tz="UTC")
    score = np.linspace(0.01, 0.99, count)
    net = (score - 0.55) * 0.04
    side = np.where(np.arange(count) % 2 == 0, "long", "short")
    return pd.DataFrame(
        {
            "__ts__": decision - pd.Timedelta(hours=1),
            "__symbol__": np.where(side == "long", "A", "B"),
            "side_name": side,
            "candidate_id": [
                f"{candidate_prefix}-{index:04d}" for index in range(count)
            ],
            "execution_decision_utc": decision,
            "execution_label_end_utc": decision + pd.Timedelta(hours=12),
            "execution_net_ev_12h": net,
            "execution_gross_ev_12h": net + 0.01,
            "execution_cost_return": np.full(count, 0.01),
            "source_partition": partition,
            "any_touch": (score > 0.4).astype(int),
            "clean_first": (score > 0.5).astype(int),
            "positive_net": (net > 0.0).astype(int),
            "timeout": (score < 0.2).astype(int),
            "test_score": score,
        }
    )


def test_tail_row_uses_global_deterministic_top_and_exact_cost_once() -> None:
    frame = _rows(
        100,
        start="2026-07-01T01:00:00Z",
        partition="june_to_july_frozen_forward_oos",
        candidate_prefix="eval",
    )

    result = tail_row(
        frame.sample(frac=1.0, random_state=7),
        score_column="test_score",
        score_name="test_score",
        arm="raw_common_eligible",
        fraction=0.10,
        eligibility="common_21d_history",
        common_eligible_rows=100,
    )

    assert result["selected_rows"] == 10
    assert result["cost_bps"] == pytest.approx(100.0)
    assert result["gross_ev_bps"] - result["cost_bps"] == pytest.approx(
        result["net_ev_bps"]
    )


def test_causal_mapping_uses_common_eligibility_and_separate_positive_admission() -> None:
    reference = _rows(
        600,
        start="2026-06-10T01:00:00Z",
        partition="june_calibration_oof",
        candidate_prefix="ref",
    )
    forward = _rows(
        100,
        start="2026-07-01T01:00:00Z",
        partition="june_to_july_frozen_forward_oos",
        candidate_prefix="eval",
    )
    ledger = pd.concat([reference, forward], ignore_index=True)

    predictions, metrics, coverage, audit = evaluate_score(
        ledger,
        score_name="test_score",
        window_days=21,
        min_reference_rows=500,
        side_support_target=500.0,
    )

    assert len(predictions) == len(forward)
    assert predictions["mapped_eligible"].all()
    common = metrics.loc[metrics["eligibility"].eq("common_21d_history")]
    assert set(common["arm"]) == {
        "raw_common_eligible",
        "causal_global",
        "side_calibrated_to_global",
    }
    assert common["common_eligible_rows"].nunique() == 1
    assert common["common_eligible_rows"].iloc[0] == len(forward)
    positive = metrics.loc[
        metrics["eligibility"].eq("mapped_ev_gt_zero_after_21d_history")
    ]
    assert set(positive["arm"]) == {
        "causal_global_positive_admission",
        "side_to_global_positive_admission",
    }
    assert coverage["mapped_fraction"].eq(1.0).all()
    assert not audit.empty
