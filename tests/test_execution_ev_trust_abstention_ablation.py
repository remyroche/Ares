from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_execution_ev_trust_abstention_ablation import (
    DECISION,
    RESOLVED,
    apply_pooled_global_selection,
    causal_recent_unlabeled_shift,
    global_top_fraction_mask,
    strategy_composites,
    trust_soft_targets,
    weekly_purged_folds,
)


def test_soft_utility_targets_are_centered_on_abstention() -> None:
    targets = trust_soft_targets([-0.02, 0.0, 0.02])
    assert targets["hard_positive"].tolist() == [0.0, 0.0, 1.0]
    np.testing.assert_allclose(targets["logistic_50bps"][1], 0.5)
    np.testing.assert_allclose(targets["clipped_200bps"], [0.0, 0.5, 1.0])
    assert targets["logistic_50bps"][0] < 0.5
    assert targets["logistic_50bps"][2] > 0.5


def test_recent_shift_never_uses_current_or_future_days() -> None:
    decision = pd.date_range("2026-05-01", periods=12, freq="D", tz="UTC")
    frame = pd.DataFrame({DECISION: decision, "score": np.arange(12, dtype=float)})
    original = causal_recent_unlabeled_shift(
        frame,
        ["score"],
        min_reference_rows=2,
    )
    changed = frame.copy()
    changed.loc[changed[DECISION].ge("2026-05-10"), "score"] = 1_000_000.0
    revised = causal_recent_unlabeled_shift(
        changed,
        ["score"],
        min_reference_rows=2,
    )
    earlier = frame[DECISION].lt("2026-05-10")
    pd.testing.assert_frame_equal(
        original.loc[earlier].reset_index(drop=True),
        revised.loc[earlier].reset_index(drop=True),
    )


def test_weekly_folds_are_purged_and_label_resolved() -> None:
    decision = pd.date_range(
        "2026-05-01",
        periods=24 * 35,
        freq="h",
        tz="UTC",
    )
    frame = pd.DataFrame(
        {
            DECISION: decision,
            RESOLVED: decision + pd.Timedelta(hours=12),
        }
    )
    folds = weekly_purged_folds(frame, min_train_rows=200)
    assert folds
    for fold in folds:
        train = frame.iloc[fold["train_positions"]]
        assert (
            train[DECISION]
            < fold["week_start"] - pd.Timedelta(hours=12)
        ).all()
        assert (train[RESOLVED] < fold["week_start"]).all()


def test_trust_gate_can_abstain_below_the_global_quota() -> None:
    score = np.linspace(0.0, 1.0, 20)
    eligible = np.zeros(20, dtype=bool)
    eligible[-1] = True
    selected = global_top_fraction_mask(
        score,
        eligible=eligible,
        population_rows=20,
        fraction=0.10,
    )
    assert selected.sum() == 1
    assert selected[-1]


def test_no_trust_preserves_the_frozen_recent_mapped_score() -> None:
    frozen = np.array([-0.01, 0.0, 0.02])
    trust = np.array([0.9, 0.1, 0.8])
    strategies = strategy_composites(
        frozen,
        trust,
        train_frozen_rank_score=np.array([-0.02, -0.01, 0.01, 0.03]),
    )
    np.testing.assert_array_equal(strategies["no_trust"][0], frozen)
    assert strategies["no_trust"][1].all()
    assert strategies["trust_gate"][1].tolist() == [True, False, True]


def test_pooled_global_selection_is_one_quota_not_weekly_quotas() -> None:
    frame = pd.DataFrame(
        {
            "label_variant": "hard_positive",
            "strategy": "no_trust",
            "ranking_score": [100.0, 99.0, 1.0, 0.0],
            "eligible": True,
            "week_start": [
                "2026-06-01",
                "2026-06-01",
                "2026-06-08",
                "2026-06-08",
            ],
        }
    )
    selected = apply_pooled_global_selection(frame, top_k_fraction=0.25)
    assert selected["pooled_global_selected"].tolist() == [
        True,
        False,
        False,
        False,
    ]
