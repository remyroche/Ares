from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from materialize_historical_exact_model_health import (  # noqa: E402
    add_failure_labels,
    causal_resolved_health,
    stable_global_top_k,
)


def test_top_k_is_one_global_book_not_timestamp_quota() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "score": [4.0, 3.0, 2.0, 1.0],
            "timestamp": [0, 0, 1, 1],
        }
    )
    selected = stable_global_top_k(frame, score_column="score", fraction=0.5)
    assert selected["candidate_id"].tolist() == ["a", "b"]
    assert selected["timestamp"].nunique() == 1


def test_resolved_health_excludes_labels_resolving_at_same_decision() -> None:
    selected = pd.DataFrame(
        {
            "effective_label_resolution_utc": pd.to_datetime(
                ["2025-01-01 01:00", "2025-01-01 02:00"], utc=True
            ),
            "execution_net_ev_12h": [0.1, -0.1],
            "mapped_direct_net": [0.0, 0.0],
            "execution_cost_return": [0.01, 0.01],
            "execution_exit_class": ["trailing", "full_stop"],
        }
    )
    decisions = pd.Series(
        pd.to_datetime(
            ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00"],
            utc=True,
        )
    )
    health = causal_resolved_health(selected, decisions).set_index(
        "execution_decision_utc"
    )
    assert health.iloc[0]["health__recent_resolved_effective_rows_hl3d"] == 0.0
    assert np.isclose(
        health.iloc[1]["health__recent_resolved_net_ev_hl3d"], 0.1
    )
    assert health.iloc[1]["health__recent_resolved_full_stop_rate_hl3d"] == 0.0
    assert health.iloc[2]["health__recent_resolved_full_stop_rate_hl3d"] > 0.0


def test_failure_windows_use_prior_twelve_and_current_forward_twelve() -> None:
    timestamps = pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")
    net = np.r_[np.full(20, 0.02), np.full(20, -0.02)]
    frame = pd.DataFrame(
        {
            "source_utc": timestamps,
            "realized_net_mean": net,
            "mapping_residual_mean": net,
            "health__selected_rows": 10,
            "outcome_available_utc": timestamps + pd.Timedelta(hours=12),
        }
    )
    labelled, _ = add_failure_labels(frame, thresholds={"strict": -1.0})
    anchor = labelled.loc[labelled["source_utc"].eq(timestamps[20])].iloc[0]
    assert np.isclose(anchor["pre_12h_net_ev_mean"], 0.02)
    assert np.isclose(anchor["post_12h_net_ev_mean"], -0.02)
    assert anchor["target_available_utc"] >= timestamps[31] + pd.Timedelta(
        hours=12
    )


def test_failure_windows_do_not_cross_missing_calendar_hours() -> None:
    timestamps = pd.date_range("2025-01-01", periods=40, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_utc": timestamps.delete(10),
            "realized_net_mean": 0.01,
            "mapping_residual_mean": 0.01,
            "health__selected_rows": 10,
            "outcome_available_utc": timestamps.delete(10)
            + pd.Timedelta(hours=12),
        }
    )
    labelled, _ = add_failure_labels(frame, thresholds={"strict": -1.0})
    local = labelled.set_index("source_utc")
    assert not bool(local.loc[timestamps[12], "label_window_complete"])
    assert pd.isna(local.loc[timestamps[12], "pre_12h_net_ev_mean"])
    assert bool(local.loc[timestamps[23], "label_window_complete"])


def test_failure_windows_weight_economics_by_candidate_rows() -> None:
    timestamps = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    rows = np.ones(30)
    rows[12:18] = 9.0
    means = np.zeros(30)
    means[12:24] = np.r_[np.full(6, 0.10), np.full(6, -0.02)]
    frame = pd.DataFrame(
        {
            "source_utc": timestamps,
            "realized_net_mean": means,
            "realized_net_sum": means * rows,
            "mapping_residual_mean": means,
            "mapping_residual_sum": means * rows,
            "health__selected_rows": rows,
            "outcome_available_utc": timestamps + pd.Timedelta(hours=12),
        }
    )
    labelled, _ = add_failure_labels(frame, thresholds={"strict": -1.0})
    anchor = labelled.loc[labelled["source_utc"].eq(timestamps[12])].iloc[0]
    expected = float(np.average(means[12:24], weights=rows[12:24]))
    assert np.isclose(anchor["post_12h_net_ev_mean"], expected)
