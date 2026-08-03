from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.materialize_2022_2026_stack_performance_calendar import (
    _period_key,
    period_distribution_summary,
    positive_summary,
    stable_global_top_mask,
)


def test_global_top_mask_crosses_timestamps_and_sides() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "__ts__": pd.to_datetime(
                ["2025-03-01", "2025-03-01", "2025-03-02", "2025-03-02"],
                utc=True,
            ),
            "side_name": ["long", "short", "long", "short"],
        }
    )
    mask = stable_global_top_mask(frame, [0.1, 0.9, 0.8, 0.2], fraction=0.5)
    assert frame.loc[mask, "candidate_id"].tolist() == ["b", "c"]


def test_period_keys_are_utc_monday_weeks_and_calendar_months() -> None:
    timestamp = pd.Series(
        pd.to_datetime(["2025-03-02T23:00:00Z", "2025-03-03T00:00:00Z"])
    )
    weeks = _period_key(timestamp, "week")
    assert weeks.astype(str).tolist() == [
        "2025-02-24 00:00:00+00:00",
        "2025-03-03 00:00:00+00:00",
    ]
    assert _period_key(timestamp, "month").tolist() == ["2025-03", "2025-03"]


def test_positive_summary_excludes_partial_periods() -> None:
    metrics = pd.DataFrame(
        {
            "period_type": ["week", "week", "month"],
            "complete_for_percentage": [True, False, True],
            "meaningfully_positive_ic": [True, True, False],
            "meaningfully_positive_ev": [True, True, True],
            "meaningfully_positive_ic_and_ev": [True, True, False],
            "point_positive_ic_and_ev": [True, True, True],
        }
    )
    summary = positive_summary(metrics).set_index("period_type")
    assert summary.loc["week", "eligible_complete_periods"] == 1
    assert summary.loc["week", "meaningfully_positive_both_pct"] == 100.0
    assert summary.loc["month", "meaningfully_positive_both_pct"] == 0.0


def test_tie_break_is_candidate_id_stable() -> None:
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"]})
    mask = stable_global_top_mask(frame, np.ones(3), fraction=1 / 3)
    assert frame.loc[mask, "candidate_id"].tolist() == ["a"]


def test_period_distribution_uses_complete_reporting_periods_not_candidates() -> None:
    metrics = pd.DataFrame(
        {
            "period_type": ["week", "week", "week", "month"],
            "complete_for_percentage": [True, True, False, True],
            "mean_net_bps": [-10.0, 30.0, 9_999.0, 5.0],
            "alpha_rank_ic": [0.10, 0.30, 9.0, 0.20],
            "execution_net_rank_ic": [0.01, 0.03, 9.0, 0.02],
            "point_positive_ev": [False, True, True, True],
            "meaningfully_positive_ev": [False, True, True, True],
            "point_positive_ic": [True, True, True, True],
            "meaningfully_positive_ic": [True, True, True, True],
            "point_positive_ic_and_ev": [False, True, True, True],
            "meaningfully_positive_ic_and_ev": [False, True, True, True],
        }
    )
    report = period_distribution_summary(metrics).set_index("period_type")
    assert report.loc["week", "complete_periods"] == 2
    assert report.loc["week", "net_ev_bps_q50"] == 10.0
    assert report.loc["week", "point_positive_ev_period_share"] == 0.5
    assert report.loc["week", "quantile_unit"] == "complete reporting period, not candidate row"
