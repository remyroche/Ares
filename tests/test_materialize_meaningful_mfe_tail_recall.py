from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_meaningful_mfe_tail_recall import (
    bind_opportunity_labels,
    tail_event_metrics,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 10
    ts = pd.date_range("2026-05-01", periods=rows, freq="h", tz="UTC")
    identity = {
        "candidate_id": [f"asset|{value.isoformat()}|1h|long" for value in ts],
        "side_name": ["long"] * rows,
        "__symbol__": ["asset"] * rows,
        "__ts__": ts,
    }
    score = pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": ts,
            "execution_label_end_utc": ts + pd.Timedelta(hours=12),
            "execution_net_ev_12h": np.linspace(-0.02, 0.02, rows),
            "execution_gross_ev_12h": np.linspace(-0.01, 0.03, rows),
            "execution_cost_return": [0.01] * rows,
            "execution_mfe_return_12h": np.linspace(0.0, 0.04, rows),
            "score_base_alpha": np.arange(rows, dtype=float),
            "candidate_month": ["2026-05"] * rows,
        }
    )
    grid = pd.DataFrame(
        {
            **identity,
            "grid_name": ["h12_u1p5atr"] * rows,
            "horizon_hours": [12] * rows,
            "label_valid": [True] * rows,
            "label_resolution_utc": ts + pd.Timedelta(hours=12),
            "execution_decision_utc": ts,
            "execution_net_ev_12h": score["execution_net_ev_12h"],
            "soft_label": np.linspace(0, 1, rows),
            "favorable_first": [0] * 8 + [1, 1],
            "adverse_first": [1] * 8 + [0, 0],
            "timeout": [0] * rows,
            # One extra any-touch event was adverse first.
            "oof_entry_atr_fraction": [0.01] * rows,
            "upper_return": [0.015] * rows,
            "peak_mfe_atr": [0.0] * 7 + [1.6, 1.7, 1.8],
            "upper_atr": [1.5] * rows,
        }
    )
    return score, grid


def test_bind_distinguishes_any_touch_clean_event_and_exact_cost() -> None:
    score, grid = _frames()
    bound = bind_opportunity_labels(
        score, grid, grid_names=["h12_u1p5atr"], expected_rows=10
    )
    assert int(bound["meaningful_mfe_any_touch"].sum()) == 3
    assert int(bound["meaningful_mfe_clean_first"].sum()) == 2
    expected = (
        score["execution_mfe_return_12h"] > score["execution_cost_return"]
    ).astype(int)
    np.testing.assert_array_equal(
        bound["path_opportunity_above_exact_cost"], expected
    )


def test_tail_recall_uses_one_global_top_tail_and_candidate_tie_break() -> None:
    score, grid = _frames()
    # Add a short side without changing the globally highest two scores.
    score.loc[0:4, "side_name"] = "short"
    grid.loc[0:4, "side_name"] = "short"
    bound = bind_opportunity_labels(
        score, grid, grid_names=["h12_u1p5atr"], expected_rows=10
    )
    metrics = tail_event_metrics(bound, "score_base_alpha")
    global_top20 = metrics.loc[
        metrics["scope"].eq("pooled_global") & metrics["fraction"].eq(0.20)
    ].iloc[0]
    assert global_top20["selected_rows"] == 2
    assert global_top20["meaningful_mfe_clean_first_selected_events"] == 2
    assert global_top20["meaningful_mfe_clean_first_recall"] == pytest.approx(1.0)
    assert global_top20["meaningful_mfe_clean_first_selected_rate"] == pytest.approx(
        1.0
    )


def test_bind_rejects_policy_net_or_resolution_mismatch() -> None:
    score, grid = _frames()
    bad_net = grid.copy()
    bad_net.loc[0, "execution_net_ev_12h"] += 0.001
    with pytest.raises(ValueError, match="net does not match"):
        bind_opportunity_labels(
            score, bad_net, grid_names=["h12_u1p5atr"], expected_rows=10
        )
    bad_time = grid.copy()
    bad_time.loc[0, "label_resolution_utc"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="resolution disagrees"):
        bind_opportunity_labels(
            score, bad_time, grid_names=["h12_u1p5atr"], expected_rows=10
        )
