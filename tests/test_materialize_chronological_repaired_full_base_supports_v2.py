from __future__ import annotations

import pandas as pd

from scripts import materialize_chronological_repaired_full_base_supports_v2 as materialize


def _row(candidate: str, signal: str) -> dict[str, object]:
    timestamp = pd.Timestamp(signal)
    return {
        "candidate_id": candidate,
        "side_name": "long",
        "__symbol__": "BTCUSDT",
        "__ts__": timestamp,
        "execution_decision_utc": timestamp + pd.Timedelta(hours=1),
        "model_development_eligible": True,
        "candidate_score_is_oof": True,
        "upstream_scores_are_outer_oof": True,
        "residual_is_oof": True,
    }


def test_full_march_folds_are_contiguous_at_decision_time() -> None:
    folds = materialize.march_folds()
    assert len(folds) == 7
    assert folds[0][1] == pd.Timestamp("2025-03-13T00:00:00Z")
    assert folds[-1][2] == pd.Timestamp("2025-04-01T00:00:00Z")
    assert all(left[2] == right[1] for left, right in zip(folds, folds[1:]))


def test_boundary_hour_is_assigned_by_decision_not_signal_time() -> None:
    signal = pd.Timestamp("2025-03-15T23:00:00Z")
    frame = pd.DataFrame({"__decision_ts__": [signal + pd.Timedelta(hours=1)]})
    first = materialize.fold_mask(frame, *materialize.FOLDS[0][1:])
    second = materialize.fold_mask(frame, *materialize.FOLDS[1][1:])
    assert first.tolist() == [False]
    assert second.tolist() == [True]


def test_decision_time_train_mask_purges_boundary_path() -> None:
    start = pd.Timestamp("2025-03-16T00:00:00Z")
    frame = pd.DataFrame(
        {
            "__decision_ts__": [
                pd.Timestamp("2025-03-15T23:00:00Z"),
                pd.Timestamp("2025-03-16T00:00:00Z"),
                pd.Timestamp("2025-03-15T11:00:00Z"),
            ],
            "execution_label_end_utc": [
                pd.Timestamp("2025-03-16T11:00:00Z"),
                pd.Timestamp("2025-03-16T12:00:00Z"),
                pd.Timestamp("2025-03-15T23:00:00Z"),
            ],
        }
    )
    assert materialize.strict_train_mask(frame, start).tolist() == [False, False, True]
