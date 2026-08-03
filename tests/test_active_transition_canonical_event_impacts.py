from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from report_active_transition_canonical_event_impacts import (  # noqa: E402
    _episode_count,
    _stable_top_k,
    _window_metrics,
    _validation_disclosure,
    bootstrap_event_summary,
    build_event_report,
    destination_event_abstention_curve,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hours = pd.date_range("2025-02-01", periods=36, freq="h", tz="UTC")
    candidate_rows: list[dict[str, object]] = []
    for hour_index, hour in enumerate(hours):
        for rank in range(4):
            gross = 0.02 - 0.002 * rank
            candidate_rows.append(
                {
                    "__ts__": hour,
                    "__symbol__": f"A{rank}",
                    "side_name": "long" if rank % 2 == 0 else "short",
                    "candidate_id": f"{hour_index}-{rank}",
                    "execution_gross_ev_12h": gross,
                    "execution_cost_return": 0.01,
                    "execution_net_ev_12h": gross - 0.01,
                    "execution_exit_class": (
                        "full_stop" if rank == 3 else "trailing"
                    ),
                    "execution_mfe_return_12h": gross + 0.01,
                    "execution_mae_return_12h": -0.01,
                    "score_raw": 1.0 - 0.1 * rank,
                    "mapped_direct_net": 0.01 - 0.002 * rank,
                    "mapped_eligible": True,
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    anchor = hours[12]
    event_id = "event"
    active = pd.DataFrame(
        {
            "source_utc": hours,
            "target__event_id": [
                event_id if anchor <= hour < anchor + pd.Timedelta(hours=3) else None
                for hour in hours
            ],
            "target__transition_active": [
                int(anchor <= hour < anchor + pd.Timedelta(hours=3))
                for hour in hours
            ],
            "prediction": [
                0.9 if anchor <= hour < anchor + pd.Timedelta(hours=3) else 0.01
                for hour in hours
            ],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [event_id],
            "anchor_source_utc": [anchor],
            "transition_start_utc": [anchor],
            "transition_end_utc": [anchor + pd.Timedelta(hours=3)],
            "source_state": [0],
            "destination_state": [2],
            "transition_archetype": ["state_0_to_state_2"],
            "robust_pre_post_shift": [1.0],
        }
    )
    destination = pd.DataFrame(
        {
            "source_utc": [anchor],
            "target__event_id": [event_id],
            "destination_label": ["state_2"],
            "predicted_destination": ["state_2"],
            "p_destination__state_0": [0.05],
            "p_destination__state_1": [0.05],
            "p_destination__state_2": [0.80],
            "p_destination__state_3": [0.05],
            "p_destination__state_4": [0.05],
        }
    )
    return candidates, active, events, destination


def test_episode_count_collapses_consecutive_alert_hours() -> None:
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC"))
    mask = np.array([True, True, False, True, False, True])
    assert _episode_count(mask, timestamps) == 3


def test_stable_top_k_is_global_and_candidate_tie_broken() -> None:
    frame = pd.DataFrame(
        {"candidate_id": ["b", "a", "c"], "score": [1.0, 1.0, 0.0]}
    )
    selected = _stable_top_k(frame, score_column="score", fraction=1 / 3)
    assert selected["candidate_id"].tolist() == ["a"]


def test_empty_event_window_is_missing_not_zero() -> None:
    metrics = _window_metrics(
        pd.DataFrame(), score_column="score_raw", prefix="during"
    )
    assert metrics["during_rows"] == 0
    assert np.isnan(metrics["during_mean_gross_bps"])
    assert np.isnan(metrics["during_full_stop_rate"])


def test_event_report_preserves_frozen_books_and_reports_destination() -> None:
    candidates, active, events, destination = _inputs()
    report, operating, pair = build_event_report(
        candidates,
        active,
        events,
        destination,
        score_columns=("score_raw", "mapped_direct_net"),
        top_k_fraction=0.5,
        thresholds=(0.5,),
    )
    assert len(report) == 2
    assert set(report["score_stream"]) == {"score_raw", "mapped_direct_net"}
    assert report["destination_prediction"].eq("state_2").all()
    assert report["destination_prediction_correct"].all()
    assert np.allclose(report["destination_confidence"], 0.8)
    assert report["active_detected_threshold_0p5"].all()
    assert operating.loc[0, "event_recall"] == 1.0
    assert operating.loc[0, "damaging_score_raw_event_count"] == 0
    assert pair["event_count"].sum() == 2
    assert pair["destination_accuracy"].eq(1.0).all()


def test_event_bootstrap_is_deterministic_and_score_local() -> None:
    candidates, active, events, destination = _inputs()
    report, _, _ = build_event_report(
        candidates,
        active,
        events,
        destination,
        score_columns=("score_raw", "mapped_direct_net"),
        top_k_fraction=0.5,
        thresholds=(0.5,),
    )
    first = bootstrap_event_summary(report, draws=100, seed=7)
    second = bootstrap_event_summary(report, draws=100, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["score_stream"]) == {"score_raw", "mapped_direct_net"}


def test_mixed_validation_disclosure_is_explicit() -> None:
    disclosure = _validation_disclosure(
        "chronological_label_oos_pooled_geometry", "grouped_oof"
    )
    assert "chronological label OOS" in disclosure["active"]
    assert "grouped OOF" in disclosure["destination"]
    assert "only 13" in disclosure["blocker"]


def test_destination_event_abstention_uses_one_row_per_event() -> None:
    candidates, active, events, destination = _inputs()
    report, _, _ = build_event_report(
        candidates,
        active,
        events,
        destination,
        score_columns=("score_raw", "mapped_direct_net"),
        top_k_fraction=0.5,
        thresholds=(0.5,),
    )
    curve = destination_event_abstention_curve(report, (0.0, 0.9))
    assert curve.loc[0, "event_count"] == 1
    assert curve.loc[0, "accepted_events"] == 1
    assert curve.loc[0, "accuracy"] == 1.0
    assert curve.loc[1, "accepted_events"] == 0
