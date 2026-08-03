from __future__ import annotations

import pandas as pd
import pytest

from scripts.fit_execution_ev_forward_final_heads import prepare_training_frame
from scripts.run_execution_ev_mixed_period_remedies import IDENTITY_COLUMNS


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identity = {
        "__ts__": pd.to_datetime(
            ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"], utc=True
        ),
        "__symbol__": ["BTC", "ETH"],
        "side_name": ["long", "short"],
        "candidate_id": ["a", "b"],
    }
    frame = pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": pd.to_datetime(
                ["2026-07-01T01:00:00Z", "2026-07-02T01:00:00Z"], utc=True
            ),
            "execution_label_end_utc": pd.to_datetime(
                ["2026-07-01T13:00:00Z", "2026-07-02T13:00:00Z"], utc=True
            ),
            "execution_gross_ev_12h": [0.02, -0.01],
            "execution_cost_return": [0.01, 0.01],
            "execution_net_ev_12h": [0.01, -0.02],
        }
    )
    capture = pd.DataFrame(
        {
            **identity,
            "execution_gross_ev_12h": [0.02, -0.01],
            "execution_cost_return": [0.01, 0.01],
            "pre_exit_mfe_return": [0.03, 0.01],
            "pre_exit_mae_return": [-0.005, -0.03],
            "pre_exit_mfe_to_gross_gap": [0.01, 0.02],
            "pre_exit_gross_capture_ratio": [0.7, 0.0],
            "post_peak_close_giveback_ratio": [0.2, 1.0],
            "giveback_after_80pct_mfe_ratio": [0.1, 1.0],
            "favorable_before_adverse_at_cost": [True, False],
            "adverse_before_favorable_at_cost": [False, True],
            "exact_net_positive": [True, False],
            "exact_net_loss_worse_two_costs": [False, True],
        }
    )
    grid = pd.DataFrame(
        {
            **identity,
            "grid_name": ["h12_u1p5atr", "h12_u1p5atr"],
            "label_valid": [True, True],
            "favorable_first": [True, False],
            "adverse_first": [False, True],
            "timeout": [False, False],
        }
    )
    return frame, capture, grid


def test_training_frame_uses_only_strictly_resolved_rows() -> None:
    frame, capture, grid = _frames()
    result = prepare_training_frame(
        frame,
        capture,
        grid,
        grid_name="h12_u1p5atr",
        training_label_end_exclusive=pd.Timestamp("2026-07-02T00:00:00Z"),
    )
    assert result["candidate_id"].tolist() == ["a"]
    assert result["target_capture_ratio"].tolist() == pytest.approx([0.7])


def test_training_frame_rejects_duplicate_identity() -> None:
    frame, capture, grid = _frames()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate identities"):
        prepare_training_frame(
            frame,
            capture,
            grid,
            grid_name="h12_u1p5atr",
            training_label_end_exclusive=pd.Timestamp("2026-07-03T00:00:00Z"),
        )


def test_training_frame_requires_timezone_aware_cutoff() -> None:
    frame, capture, grid = _frames()
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        prepare_training_frame(
            frame,
            capture,
            grid,
            grid_name="h12_u1p5atr",
            training_label_end_exclusive=pd.Timestamp("2026-07-03"),
        )
