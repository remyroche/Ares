from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_within_july_learnability import (
    DECISION,
    RESOLVED,
    TARGET,
    aggregate_prediction_metrics,
    assert_forward_safe,
    build_temporal_splits,
    learnability_conclusion,
)


def _frame() -> pd.DataFrame:
    decision = pd.date_range(
        "2026-07-01",
        "2026-07-19 23:00",
        freq="6h",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            DECISION: decision,
            RESOLVED: decision + pd.Timedelta(hours=12),
            TARGET: np.linspace(-0.02, 0.02, len(decision)),
        }
    )


def test_weekly_forward_splits_purge_decisions_and_unresolved_labels() -> None:
    frame = _frame()
    splits = build_temporal_splits(frame, purge_hours=12.0)
    forward = [split for split in splits if split.mode == "forward_expanding"]
    assert len(forward) == 2
    for split in forward:
        assert_forward_safe(frame, split, purge_hours=12.0)
        train = frame.iloc[split.train_positions]
        assert (train[RESOLVED] < split.evaluation_start).all()
        assert (
            train[DECISION] < split.evaluation_start - pd.Timedelta(hours=12)
        ).all()


def test_directional_controls_have_exact_matched_sizes_and_labels() -> None:
    splits = build_temporal_splits(_frame(), purge_hours=12.0)
    for fold_id in {split.fold_id for split in splits}:
        forward = next(
            split
            for split in splits
            if split.fold_id == fold_id and split.mode == "forward_block_matched"
        )
        reverse = next(
            split
            for split in splits
            if split.fold_id == fold_id and split.mode == "reversed_block_matched"
        )
        assert len(forward.train_positions) == len(reverse.train_positions)
        assert len(forward.evaluation_positions) == len(reverse.evaluation_positions)
        assert len(forward.train_positions) == len(forward.evaluation_positions)
        assert forward.is_valid_forward_oos
        assert not reverse.is_valid_forward_oos
        assert reverse.training_direction == "future_to_past"


def test_conclusion_uses_only_forward_aggregate_and_uncertainty() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "mode": "forward_expanding",
                "scope": "pooled",
                "positive_ev_auc": 0.61,
                "spearman": 0.12,
                "top_k_mean_net_ev": 0.01,
                "top_k_lift_vs_unconditional": 0.012,
            },
            {
                "mode": "in_sample",
                "scope": "pooled",
                "positive_ev_auc": 0.99,
                "spearman": 0.95,
                "top_k_mean_net_ev": 0.20,
                "top_k_lift_vs_unconditional": 0.20,
            },
        ]
    )
    uncertainty = pd.DataFrame(
        [
            {
                "scope": "pooled",
                "metric": "top_k_mean_net_ev",
                "ci025": 0.001,
            },
            {
                "scope": "pooled",
                "metric": "top_k_mean_net_ev_delta_vs_baseline",
                "ci025": -0.001,
            },
        ]
    )
    conclusion = learnability_conclusion(aggregate, uncertainty)
    assert conclusion["status"] == "forward_oos_learnability_supported"
    assert conclusion["day_block_ci_excludes_zero"]
    assert not conclusion["day_block_delta_vs_baseline_ci_excludes_zero"]
    assert "in_sample" in conclusion["diagnostic_modes_excluded"]
