from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import RegimeOOFStackError
from extreme_price_movements.regime_stack_evaluation import (
    EvaluationColumns,
    evaluate_matched_arms,
    global_top_k_mask,
)


def _frame(score_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for week in range(12):
        timestamp = pd.Timestamp("2026-01-05T00:00Z") + pd.Timedelta(days=7 * week)
        for idx in range(10):
            net = 0.01 if idx >= 8 else -0.01
            rows.append(
                {
                    "candidate_id": f"{week}-{idx}",
                    "__ts__": timestamp,
                    "__symbol__": f"S{idx}",
                    "side_name": "long" if idx % 2 else "short",
                    "mapped_score": float(week * 10 + idx) + score_shift,
                    "__first_touch_target_soft__": float(idx) / 9.0,
                    "execution_net_ev_12h": net,
                    "execution_gross_ev_12h": net + 0.01,
                    "execution_cost_12h": 0.01,
                    "state": "good" if idx >= 8 else "other",
                }
            )
    return pd.DataFrame(rows)


def test_global_top_k_is_pooled_not_timestamp_or_side_local() -> None:
    frame = _frame()
    mask = global_top_k_mask(frame, score_col="mapped_score", top_fraction=0.10)
    selected = frame.loc[mask]
    assert len(selected) == 12
    # The first week contains no selected row: selection is over the full pool,
    # not separately within each timestamp.
    assert selected["__ts__"].min() > frame["__ts__"].min()


def test_matched_evaluation_reports_global_economics_and_period_q10_q50() -> None:
    baseline = _frame()
    regime = _frame(score_shift=0.01)
    summary, periods, categories = evaluate_matched_arms(
        {"baseline": baseline, "regime_only": regime},
        columns=EvaluationColumns(),
        category_col="state",
    )
    assert set(summary["arm"]) == {"baseline", "regime_only"}
    assert set(summary["selection_basis"]) == {"pooled_global_post_mapping_top_k"}
    assert {"weekly_ic_q10", "weekly_ic_q50", "monthly_net_ev_q10", "monthly_net_ev_q50"}.issubset(summary.columns)
    assert set(periods["period_type"]) == {"week", "month"}
    assert not categories.empty


def test_matched_evaluation_rejects_row_mismatch() -> None:
    baseline = _frame()
    shorter = baseline.iloc[:-1].copy()
    with pytest.raises(RegimeOOFStackError, match="exact candidate rows"):
        evaluate_matched_arms({"baseline": baseline, "transition_only": shorter})
