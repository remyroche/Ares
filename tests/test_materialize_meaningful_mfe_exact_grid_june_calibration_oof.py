from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_meaningful_mfe_exact_grid_june_calibration_oof import (
    JUNE_FOLD_STARTS,
    JuneFold,
    PREDECLARED_SCORES,
    _select_scores,
    _stamp,
    june_fold_masks,
    june_folds,
)


def _panel() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2026-06-01 00:00:00+00:00",
            "2026-06-09 10:00:00+00:00",
            "2026-06-09 11:00:00+00:00",
            "2026-06-10 00:00:00+00:00",
            "2026-06-16 23:00:00+00:00",
        ]
    )
    decision = ts + pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "candidate_id": [f"c{index}" for index in range(len(ts))],
            "__ts__": ts,
            "__symbol__": ["A"] * len(ts),
            "side_name": ["long"] * len(ts),
            "execution_decision_utc": decision,
            "label_resolution_utc": decision + pd.Timedelta(hours=12),
        }
    )


def test_june_fold_schedule_is_the_three_required_contiguous_blocks() -> None:
    folds = june_folds()
    assert [(fold.start, fold.end) for fold in folds] == [
        (JUNE_FOLD_STARTS[0], JUNE_FOLD_STARTS[1]),
        (JUNE_FOLD_STARTS[1], JUNE_FOLD_STARTS[2]),
        (JUNE_FOLD_STARTS[2], pd.Timestamp("2026-07-01T00:00:00Z")),
    ]


def test_june_fold_enforces_resolution_and_independent_12h_decision_purge() -> None:
    panel = _panel()
    fold = JuneFold(
        "june_oof_fold_0",
        pd.Timestamp("2026-06-10T00:00:00Z"),
        pd.Timestamp("2026-06-17T00:00:00Z"),
    )
    train, validation = june_fold_masks(panel, fold)

    # Jun 9 10:00 decision is 11:00 and remains before the 12:00 cutoff.
    assert train.tolist() == [0, 1]
    # Jun 9 11:00 decision is exactly the cutoff and must be excluded.
    assert 2 not in train.tolist()
    assert validation.tolist() == [3, 4]


def test_stamp_rejects_oof_lineage_reaching_the_decision() -> None:
    frame = pd.DataFrame(
        {
            "execution_decision_utc": [pd.Timestamp("2026-06-10T01:00:00Z")],
            "label_resolution_utc": [pd.Timestamp("2026-06-10T13:00:00Z")],
        }
    )
    with pytest.raises(ValueError, match="training outcomes reach"):
        _stamp(
            frame,
            source_partition="june_calibration_oof",
            is_oof=True,
            fold="fold",
            model_available_at=pd.Timestamp("2026-06-10T00:00:00Z"),
            training_decision_cutoff=pd.Timestamp("2026-06-10T00:00:00Z"),
            training_label_resolution_max=pd.Timestamp("2026-06-10T01:00:00Z"),
            recipe_hash="a" * 64,
        )


def test_predeclared_score_surface_excludes_unconditional_capture_head() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["c"],
            "__ts__": [pd.Timestamp("2026-06-10T00:00:00Z")],
            "__symbol__": ["A"],
            "side_name": ["long"],
            "execution_net_ev_12h": [0.01],
            "execution_gross_ev_12h": [0.02],
            "execution_cost_return": [0.01],
            "execution_exit_reason": ["trailing"],
            "execution_mfe_return_12h": [0.03],
            "execution_mae_return_12h": [0.01],
            "any_touch": [1],
            "clean_first": [1],
            "positive_net": [1],
            "timeout": [0],
            **{name: [0.5] for name in PREDECLARED_SCORES},
            "p_lightgbm_capture_given_touch": [0.99],
        }
    )
    selected = _select_scores(frame)
    assert set(PREDECLARED_SCORES).issubset(selected.columns)
    assert "p_lightgbm_capture_given_touch" not in selected.columns
    assert np.isfinite(selected.loc[0, list(PREDECLARED_SCORES)].to_numpy(float)).all()
