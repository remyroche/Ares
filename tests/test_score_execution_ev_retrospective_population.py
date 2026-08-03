from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_execution_ev_forward_calibrator_seed import TARGET
from scripts.score_execution_ev_retrospective_population import (
    _validate_retro_preentry,
    apply_global_admission,
    causal_retrospective_mapping,
)


def _history(resolution: list[str], target: list[float]) -> pd.DataFrame:
    count = len(resolution)
    return pd.DataFrame(
        {
            "candidate_id": [f"h{i}" for i in range(count)],
            "__ts__": pd.date_range("2026-06-20", periods=count, freq="h", tz="UTC"),
            "__symbol__": ["BTC"] * count,
            "side_name": ["long"] * count,
            "execution_decision_utc": pd.date_range(
                "2026-06-20", periods=count, freq="h", tz="UTC"
            ),
            "execution_label_end_utc": pd.to_datetime(resolution, utc=True),
            TARGET: target,
            "frozen_margin_capture_interaction_raw": np.linspace(-1.0, 1.0, count),
        }
    )


def _candidates(timestamps: list[str], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(len(timestamps))],
            "__ts__": pd.to_datetime(timestamps, utc=True),
            "__symbol__": ["BTC"] * len(timestamps),
            "side_name": ["long"] * len(timestamps),
            "execution_decision_utc": pd.to_datetime(timestamps, utc=True),
            "frozen_margin_capture_interaction_raw": scores,
        }
    )


def test_retrospective_mapping_never_uses_same_time_or_future_labels() -> None:
    candidates = _candidates(["2026-08-22T00:00:00Z"], [0.0])
    # The Jul-31 observation falls outside the 21d window.  Aug-01 exactly at
    # the lower bound remains eligible; same-time and later rows must not.
    history = _history(
        [
            "2026-07-31T23:59:59Z",
            "2026-08-01T00:00:00Z",
            "2026-08-10T00:00:00Z",
            "2026-08-21T23:59:59Z",
            "2026-08-22T00:00:00Z",
            "2026-08-23T00:00:00Z",
        ],
        [-10.0, -0.02, 0.00, 0.02, 100.0, 200.0],
    )
    mapped, support = causal_retrospective_mapping(
        candidates, history, minimum_side_rows=3
    )
    allowed = history.iloc[1:4]
    reference, _ = causal_retrospective_mapping(
        candidates, allowed, minimum_side_rows=3
    )
    assert mapped.tolist() == reference.tolist()
    assert support.loc[0, "history_rows"] == 3
    assert support.loc[0, "history_resolution_min_utc"] == pd.Timestamp(
        "2026-08-01T00:00:00Z"
    )
    assert support.loc[0, "history_resolution_max_utc"] == pd.Timestamp(
        "2026-08-21T23:59:59Z"
    )
    assert bool(support.loc[0, "history_resolved_strictly_before_decision"])


def test_later_resolved_history_cannot_change_an_earlier_retro_score() -> None:
    candidates = _candidates(
        ["2026-08-10T00:00:00Z", "2026-08-12T00:00:00Z"], [0.0, 0.0]
    )
    baseline = _history(
        [
            "2026-08-07T00:00:00Z",
            "2026-08-08T00:00:00Z",
            "2026-08-09T00:00:00Z",
        ],
        [-0.01, 0.00, 0.01],
    )
    with_later_label = pd.concat(
        [
            baseline,
            _history(["2026-08-11T00:00:00Z"], [99.0]).assign(candidate_id="later"),
        ],
        ignore_index=True,
    )
    base_mapped, _ = causal_retrospective_mapping(
        candidates, baseline, minimum_side_rows=3
    )
    later_mapped, _ = causal_retrospective_mapping(
        candidates, with_later_label, minimum_side_rows=3
    )
    assert later_mapped[0] == base_mapped[0]


def test_retrospective_admission_is_one_pooled_global_book_after_mapping() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"id_{i:02d}" for i in range(20)],
            # The two best mapped values live in the same timestamp/side.  A
            # per-timestamp or per-side top-decile would select a different set.
            "mapped_execution_ev": [0.10, 0.09] + [0.01] * 18,
            "execution_decision_utc": pd.to_datetime(
                ["2026-08-01"] * 10 + ["2026-08-02"] * 10, utc=True
            ),
            "side_name": ["long"] * 10 + ["short"] * 10,
        }
    )
    admitted = apply_global_admission(frame)
    assert set(admitted.loc[admitted["global_top10_capacity_member"], "candidate_id"]) == {
        "id_00",
        "id_01",
    }
    assert admitted["globally_admitted_floor_50bps"].sum() == 2


def test_retrospective_preentry_rejects_outcomes_and_forward_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["c"],
            "__ts__": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "__symbol__": ["BTC"],
            "side_name": ["long"],
            "execution_decision_utc": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "feature_available_at": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "base_available_at": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "residual_available_at": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "peak_mfe_available_at": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "path_catboost_available_at": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            "clean_probability_available_at": pd.to_datetime(["2026-07-20T00:00:00Z"], utc=True),
            TARGET: [0.01],
        }
    )
    with pytest.raises(ValueError, match="resolved outcome"):
        _validate_retro_preentry(
            frame,
            first_decision_exclusive=pd.Timestamp("2026-07-27T23:59:59Z"),
        )
