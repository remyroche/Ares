from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_exact_h12_side_local_residual_oof import (
    TARGET,
    _feature_screen,
    _joint_development_contracts,
    _weekly_top10_objective,
    rank_ic,
    stable_top,
)


def test_stable_top_is_pooled_and_uses_declared_secondary_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["z", "a", "b", "c"],
            "side_name": ["long", "short", "long", "short"],
            "score": [1.0, 1.0, 0.0, 0.0],
            "secondary": [0.1, 0.9, 0.0, 0.0],
        }
    )
    selected = stable_top(
        frame, "score", 0.25, secondary_column="secondary"
    )
    assert selected["candidate_id"].tolist() == ["a"]
    assert selected["side_name"].tolist() == ["short"]


def test_weekly_objective_uses_global_rows_supplied_without_timestamp_quota() -> None:
    timestamps = pd.to_datetime(
        ["2026-05-04T00:00:00Z"] * 10 + ["2026-05-11T00:00:00Z"] * 10
    )
    target = np.r_[np.arange(10), np.arange(10)] / 10_000
    frame = pd.DataFrame(
        {
            "candidate_id": [f"id-{index:02d}" for index in range(20)],
            "__ts__": timestamps,
            TARGET: target,
        }
    )
    metrics = _weekly_top10_objective(
        frame,
        score=np.r_[np.arange(10), np.arange(10)],
        secondary=np.arange(20),
    )
    assert metrics["mean_week_top10_net_bps"] == 9.0
    assert metrics["worst_week_top10_net_bps"] == 9.0


def test_feature_screen_is_side_local_target_specific_and_keeps_context() -> None:
    rows = 200
    target = np.linspace(-1.0, 1.0, rows)
    frame = pd.DataFrame(
        {
            "base_oof_score": target,
            "base_rank_pct_timestamp_side": target**3,
            "base_score_z_timestamp_side": target * 2.0,
            "signal": target + 0.01 * np.sin(np.arange(rows)),
            "noise": np.sin(np.arange(rows) * 1.7),
            "sparse": np.where(np.arange(rows) < 30, np.nan, target),
        }
    )
    selected, report = _feature_screen(
        frame,
        target,
        list(frame.columns),
        max_count=4,
    )
    assert "base_oof_score" in selected
    assert "base_rank_pct_timestamp_side" in selected
    assert "base_score_z_timestamp_side" in selected
    assert "sparse" not in report["feature"].tolist()


def test_rank_ic_reports_exact_monotone_alignment() -> None:
    assert rank_ic([1, 2, 3], [10, 20, 30]) == 1.0
    assert rank_ic([1, 1, 1], [10, 20, 30]) != rank_ic(
        [1, 1, 1], [10, 20, 30]
    )


def test_joint_contract_selection_uses_pooled_global_book() -> None:
    timestamps = pd.to_datetime(["2025-04-07T00:00:00Z"] * 4)
    contracts = {
        "long": {
            "a": {"trial_id": "a", "config": {}, "features": [], "alpha": 1.0},
            "b": {"trial_id": "b", "config": {}, "features": [], "alpha": 1.0},
        },
        "short": {
            "c": {"trial_id": "c", "config": {}, "features": [], "alpha": 1.0},
            "d": {"trial_id": "d", "config": {}, "features": [], "alpha": 1.0},
        },
    }
    frames = {}
    for side, trials in {
        "long": {"a": ([10.0, 9.0], [0.02, -0.01]), "b": ([2.0, 1.0], [0.02, -0.01])},
        "short": {"c": ([8.0, 7.0], [0.03, -0.02]), "d": ([1.0, 0.0], [0.03, -0.02])},
    }.items():
        parts = []
        for trial, (scores, targets) in trials.items():
            parts.append(
                pd.DataFrame(
                    {
                        "candidate_id": [f"{side}-{trial}-0", f"{side}-{trial}-1"],
                        "side_name": side,
                        "__ts__": timestamps[:2],
                        TARGET: targets,
                        "base_oof_score": [0.2, 0.1],
                        "trial_id": trial,
                        "score_bps": scores,
                    }
                )
            )
        frames[side] = pd.concat(parts, ignore_index=True)
    selected, trials = _joint_development_contracts(contracts, frames)
    assert selected["long"]["joint_pair"] == selected["short"]["joint_pair"]
    assert len(trials) == 4
