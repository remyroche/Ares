from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_recent_residual_shrinkage_ablation import (
    _apply_side_shrink,
    policy_global_topk_mask,
    training_only_adaptive_shrinkage,
    weekly_forward_windows,
)


def _train() -> pd.DataFrame:
    ts = pd.date_range("2026-05-01", periods=1_500, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "execution_decision_utc": ts,
            "side_name": np.where(np.arange(len(ts)) % 2, "short", "long"),
            "existing_alpha_ev": np.sin(np.arange(len(ts)) / 20.0),
            "base_oof_score": np.cos(np.arange(len(ts)) / 30.0),
            "base_margin_to_cutoff_z": np.sin(np.arange(len(ts)) / 40.0),
            "oof_clean_favorable_probability": 0.5,
            "alpha_prediction_uncertainty": 0.2,
            "catboost_entropy": 0.3,
        }
    )


def test_weekly_windows_are_adjacent_and_latest_is_july13() -> None:
    windows = weekly_forward_windows()
    assert len(windows) == 7
    assert windows[0].cutoff.startswith("2026-06-01")
    assert windows[-1].cutoff.startswith("2026-07-13")
    for left, right in zip(windows, windows[1:]):
        assert pd.Timestamp(left.evaluation_end) == pd.Timestamp(right.cutoff)


def test_adaptive_shrink_has_explicit_zero_fallback() -> None:
    train = _train()
    unavailable = {
        "long": {"status": "insufficient_support", "weights": {}},
        "short": {"status": "insufficient_support", "weights": {}},
    }
    shrink, audit = training_only_adaptive_shrinkage(train, unavailable)
    assert shrink == {"long": 0.0, "short": 0.0}
    assert audit["long"]["fallback_reason"] == "adapter_unavailable"


def test_side_shrink_is_bounded_and_side_specific() -> None:
    frame = pd.DataFrame({"side_name": ["long", "short"]})
    out = _apply_side_shrink(
        frame,
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        {"long": 0.25, "short": 0.0},
    )
    np.testing.assert_allclose(out, [1.5, 1.0])


def test_portfolio_global_topk_matches_exact_replay_tie_semantics() -> None:
    frame = pd.DataFrame(
        {
            "score": [0.0, 1.0, 1.0, 1.0],
            "execution_decision_utc": pd.date_range(
                "2026-01-01", periods=4, freq="h", tz="UTC"
            ),
            "__symbol__": ["A"] * 4,
            "side_name": ["long"] * 4,
            "candidate_id": [f"c{i}" for i in range(4)],
        }
    )
    mask = policy_global_topk_mask(frame, "score", 0.50)
    assert mask.tolist() == [False, True, True, False]
