from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_2022_2026_stack_regime_failures import (
    bh_fdr,
    composition_within_category_decomposition,
    asset_exit_attribution,
    exact_label_permutation_pvalues,
    identify_worst_weeks,
    side_state_transition_period_metrics,
)


def test_bh_fdr_is_monotone_and_bounded() -> None:
    adjusted = bh_fdr([0.01, 0.04, 0.03, 0.50])
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))
    assert adjusted[0] <= adjusted[1]


def test_exact_permutation_detects_large_period_shift() -> None:
    values = np.array(
        [
            [10.0, 0.0],
            [11.0, 1.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ]
    )
    labels = np.array([True, True, False, False, False, False])
    p = exact_label_permutation_pvalues(values, labels)
    assert p[0] < p[1]
    assert 0.0 <= p[0] <= 1.0


def test_worst_weeks_use_complete_periods_only() -> None:
    frame = pd.DataFrame(
        {
            "period_type": ["week"] * 9,
            "complete_for_percentage": [True] * 8 + [False],
            "mean_net_bps": [8, 7, 6, 5, 4, 3, 2, 1, -999],
            "period_start_utc": pd.date_range(
                "2025-01-06", periods=9, freq="7D", tz="UTC"
            ),
            "period_end_exclusive_utc": pd.date_range(
                "2025-01-13", periods=9, freq="7D", tz="UTC"
            ),
        }
    )
    result = identify_worst_weeks(frame, quantile=0.25)
    assert len(result) == 8
    assert result["worst_week"].sum() == 2
    assert result.loc[result["worst_week"], "mean_net_bps"].tolist() == [1, 2]


def test_side_state_transition_metrics_are_global_first_and_retain_unavailable() -> None:
    rows = []
    for start in (pd.Timestamp("2025-01-06", tz="UTC"), pd.Timestamp("2025-01-13", tz="UTC")):
        for position in range(20):
            rows.append(
                {
                    "candidate_id": f"{start.date()}-{position}",
                    "__ts__": start + pd.Timedelta(hours=position % 2),
                    "__symbol__": "BTC/USD:USD",
                    "side_name": "long" if position % 2 == 0 else "short",
                    "lineage_id": "research_lineage",
                    "evidence_grade": "B_RESEARCH",
                    "score_residual_expected_ev": float(100 - position),
                    "__first_touch_target_soft__": float(position % 3) / 2.0,
                    "execution_gross_ev_12h": 0.02,
                    "execution_cost_return": 0.01,
                    "execution_net_ev_12h": 0.01,
                }
            )
    hourly = pd.DataFrame(
        {
            "source_utc": pd.to_datetime(["2025-01-06T00:00:00Z", "2025-01-13T00:00:00Z"]),
            "state_context__current_state": [2, 3],
            "target__phase": ["approach", "active"],
            "target__transition_active": [True, True],
            "target__transition_archetype": ["a", "b"],
        }
    )
    metrics, stability, selected = side_state_transition_period_metrics(
        pd.DataFrame(rows), hourly
    )
    assert set(metrics["selection_scope"]) == {
        "one pooled-global top10 within period; category attribution after selection"
    }
    assert selected["period_selected_rows"].min() >= 2
    assert not metrics["regime_timeline_available"].all()
    assert "unavailable" in set(metrics["market_state"])
    assert "promotion_status" in stability.columns


def test_composition_within_category_decomposition_recomposes() -> None:
    selected = pd.DataFrame(
        {
            "period_type": ["week"] * 4,
            "period_start_utc": pd.to_datetime(
                ["2025-01-06", "2025-01-06", "2025-01-13", "2025-01-13"], utc=True
            ),
            "side_name": ["long", "short", "long", "short"],
            "market_state": ["0", "1", "0", "1"],
            "transition_phase_attribution": ["stable", "active", "stable", "active"],
            "execution_gross_ev_12h": [0.03, 0.01, 0.02, 0.01],
            "execution_cost_return": [0.01, 0.01, 0.01, 0.01],
            "execution_net_ev_12h": [0.02, 0.00, 0.01, 0.00],
        }
    )
    weeks = pd.DataFrame(
        {
            "period_start_utc": pd.to_datetime(["2025-01-06", "2025-01-13"], utc=True),
            "worst_week": [True, False],
        }
    )
    result = composition_within_category_decomposition(selected, weeks)
    assert set(result["metric"]) == {"gross", "cost", "net"}
    assert np.allclose(result["recomposition_error_bps"], 0.0)


def test_asset_exit_attribution_keeps_missing_exit_reasons_unavailable() -> None:
    rows = []
    for start, net in ((pd.Timestamp("2025-01-06", tz="UTC"), .02), (pd.Timestamp("2025-01-13", tz="UTC"), -.01)):
        for position in range(20):
            rows.append({"candidate_id":f"{start}-{position}","__ts__":start+pd.Timedelta(hours=position%2),"__symbol__":"BTC" if position<15 else "ETH","side_name":"long","lineage_id":"lineage","evidence_grade":"A","score_residual_expected_ev":float(20-position),"__first_touch_target_soft__":.5,"execution_gross_ev_12h":net+.01,"execution_cost_return":.01,"execution_net_ev_12h":net,"execution_exit_reason":"timeout" if start.day==6 else np.nan})
    weeks = pd.DataFrame({"period_start_utc":pd.to_datetime(["2025-01-06","2025-01-13"],utc=True),"lineage_id":["lineage"]*2,"evidence_grade":["A"]*2,"worst_week":[False,True]})
    period, book, decomposition = asset_exit_attribution(pd.DataFrame(rows), weeks)
    exit_rows = period.loc[period.attribution_kind.eq("exit_reason")]
    assert "unavailable" in set(exit_rows.attribution_value)
    assert not exit_rows.loc[exit_rows.attribution_value.eq("unavailable"), "exit_reason_attribution_available"].any()
    assert set(book.period_type) == {"week", "month"}
    assert set(decomposition.attribution_kind) == {"asset", "exit_reason"}
    assert np.allclose(decomposition.recomposition_error_bps, 0.0)


def test_asset_exit_attribution_uses_one_pooled_period_denominator_without_group_scans() -> None:
    """Shares must sum to one globally, with no per-group Series.eq scan."""
    rows = []
    starts = pd.date_range("2025-02-03", periods=4, freq="7D", tz="UTC")
    for week_number, start in enumerate(starts):
        for position in range(20):
            grade = "A" if position % 2 else "B"
            rows.append(
                {
                    "candidate_id": f"{week_number}-{position}",
                    "__ts__": start + pd.Timedelta(hours=position % 2),
                    "__symbol__": f"asset_{position}",
                    "side_name": "long",
                    "lineage_id": f"lineage_{grade}",
                    "evidence_grade": grade,
                    "score_residual_expected_ev": float(20 - position),
                    "__first_touch_target_soft__": .5,
                    "execution_gross_ev_12h": .02,
                    "execution_cost_return": .01,
                    "execution_net_ev_12h": .01,
                    "execution_exit_reason": "timeout",
                }
            )
    weeks = pd.DataFrame(
        {
            "period_start_utc": starts.repeat(2),
            "lineage_id": ["lineage_A", "lineage_B"] * len(starts),
            "evidence_grade": ["A", "B"] * len(starts),
            "worst_week": [False, False, True, True, False, False, True, True],
        }
    )
    calls = 0
    original_eq = pd.Series.eq

    def tracked_eq(self, other, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_eq(self, other, *args, **kwargs)

    pd.Series.eq = tracked_eq
    try:
        period, _, _ = asset_exit_attribution(pd.DataFrame(rows), weeks)
    finally:
        pd.Series.eq = original_eq

    asset_rows = period.loc[period["attribution_kind"].eq("asset")]
    pooled = asset_rows.groupby(
        ["period_type", "period", "period_start_utc"], dropna=False
    )["selected_book_share"].sum()
    assert np.allclose(pooled.to_numpy(), 1.0)
    # Only the post-attribution weekly filter needs ``Series.eq``.  The
    # former implementation made several full-table equality scans per group.
    assert calls <= 2
