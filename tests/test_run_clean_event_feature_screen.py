from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_clean_event_feature_screen import (
    ADVERSE,
    EVENT_TARGET,
    FAVORABLE,
    SOFT_LABEL,
    TIMEOUT,
    add_clean_target,
    evaluate_promotion_gate,
    select_clean_features,
)


def test_clean_target_uses_canonical_soft_label_and_valid_hard_outcomes() -> None:
    frame = pd.DataFrame(
        {
            FAVORABLE: [True, False, False],
            ADVERSE: [False, True, False],
            TIMEOUT: [False, False, True],
            SOFT_LABEL: [1.0, 0.0, 0.65],
        }
    )
    result = add_clean_target(frame)
    np.testing.assert_allclose(result[EVENT_TARGET], [1.0, 0.0, 0.65])
    assert result["clean_order_resolved"].tolist() == [True, True, True]


def test_selector_requires_stable_clean_event_lift_in_both_halves() -> None:
    rows = 800
    decision = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    good = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), rows // 4)
    bad = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), rows // 4)
    clean_event = np.maximum(good, bad)
    frame = pd.DataFrame(
        {
            "execution_decision_utc": decision,
            EVENT_TARGET: clean_event,
            "execution_net_ev_12h": 0.02 * good - 0.02 * bad,
            "capture_candidate__good": good,
            "capture_candidate__constant": np.ones(rows),
        }
    )
    selected, report = select_clean_features(
        frame,
        [
            "capture_candidate__good",
            "capture_candidate__constant",
        ],
        max_features=4,
        minimum_coverage=0.99,
        maximum_per_family=4,
        correlation_cap=0.95,
    )
    assert selected == ["capture_candidate__good"]
    assert report["selected_count"] == 1


def test_promotion_gate_requires_both_windows_economics_and_coverage() -> None:
    rows = []
    for window in ("june", "july"):
        rows.extend(
            [
                {
                    "window": window,
                    "arm": "all_256",
                    "stage": "causal_global_recent_mapping",
                    "scope": "pooled_global",
                    "top10_net_bps": 2.0,
                    "latest_7d_candidate_rows": 500,
                    "latest_7d_selected_rows": 50,
                },
                {
                    "window": window,
                    "arm": "top_64",
                    "stage": "causal_global_recent_mapping",
                    "scope": "pooled_global",
                    "top10_net_bps": 3.0,
                    "latest_7d_candidate_rows": 500,
                    "latest_7d_selected_rows": 50,
                },
                {
                    "window": window,
                    "arm": "top_128",
                    "stage": "causal_global_recent_mapping",
                    "scope": "pooled_global",
                    "top10_net_bps": -1.0 if window == "july" else 3.0,
                    "latest_7d_candidate_rows": 500,
                    "latest_7d_selected_rows": 10 if window == "july" else 50,
                },
            ]
        )
    gate = evaluate_promotion_gate(
        pd.DataFrame(rows), minimum_latest_selected_rows=25
    )
    assert gate["challengers"]["top_64"]["eligible_for_mda_hpo"] is True
    assert gate["challengers"]["top_128"]["eligible_for_mda_hpo"] is False
    assert gate["eligible_arms"] == ["top_64"]
