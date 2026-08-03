from __future__ import annotations

import pandas as pd
import pytest

from scripts.score_execution_ev_forward_population import (
    apply_global_admission,
    causal_recent_isotonic_mapping,
    validate_resolved_updates,
)


def test_admission_is_one_global_book_not_per_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "mapped_execution_ev": [0.01, 0.02, 0.03, -0.01],
            "execution_decision_utc": pd.to_datetime(
                ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02"],
                utc=True,
            ),
            "side_name": ["long", "short", "long", "short"],
        }
    )
    result = apply_global_admission(frame)
    assert result["global_top10_capacity_member"].sum() == 1
    assert result.index[result["global_top10_capacity_member"]].tolist() == [2]


def test_global_ties_use_candidate_identity_not_input_order() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [f"id_{i:02d}" for i in range(20)],
            "mapped_execution_ev": 0.01,
        }
    )
    first = apply_global_admission(frame)
    second = apply_global_admission(
        frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    )
    assert set(
        first.loc[first["global_top10_capacity_member"], "candidate_id"]
    ) == {"id_00", "id_01"}
    assert set(
        second.loc[second["global_top10_capacity_member"], "candidate_id"]
    ) == {"id_00", "id_01"}


def test_causal_mapping_excludes_same_time_and_unresolved_updates() -> None:
    history = pd.DataFrame(
        {
            "side_name": ["long", "long", "long", "long"],
            "execution_label_end_utc": pd.to_datetime(
                [
                    "2026-07-31T00:00:00Z",
                    "2026-07-31T01:00:00Z",
                    "2026-08-01T00:00:00Z",
                    "2026-08-02T01:00:00Z",
                ],
                utc=True,
            ),
            "frozen_margin_capture_interaction_raw": [-1.0, 0.0, 1.0, 2.0],
            "execution_net_ev_12h": [-0.02, 0.00, 0.02, 1.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "side_name": ["long"],
            "execution_decision_utc": pd.to_datetime(
                ["2026-08-02T00:00:00Z"], utc=True
            ),
            "frozen_margin_capture_interaction_raw": [1.0],
        }
    )
    mapped, report = causal_recent_isotonic_mapping(
        candidates,
        history,
        lookback_days=21,
        minimum_side_rows=3,
    )
    assert mapped[0] == 0.02
    assert report[0]["history_rows"] == 3
    assert report[0]["history_resolution_max_utc"] == pd.Timestamp(
        "2026-08-01T00:00:00Z"
    )


def test_resolved_updates_require_known_unique_identity_and_exact_score() -> None:
    identity = {
        "candidate_id": ["new"],
        "__ts__": pd.to_datetime(["2026-08-01T00:00:00Z"], utc=True),
        "__symbol__": ["BTC"],
        "side_name": ["long"],
    }
    candidates = pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": pd.to_datetime(
                ["2026-08-01T01:00:00Z"], utc=True
            ),
            "frozen_margin_capture_interaction_raw": [0.2],
        }
    )
    updates = pd.DataFrame(
        {
            **identity,
            "execution_decision_utc": pd.to_datetime(
                ["2026-08-01T01:00:00Z"], utc=True
            ),
            "execution_label_end_utc": pd.to_datetime(
                ["2026-08-01T13:00:00Z"], utc=True
            ),
            "execution_net_ev_12h": [0.01],
            "frozen_margin_capture_interaction_raw": [0.2],
        }
    )
    seed = pd.DataFrame({"candidate_id": ["old"]})
    checked = validate_resolved_updates(updates, candidates, seed)
    assert len(checked) == 1
    without_score = updates.drop(
        columns=["frozen_margin_capture_interaction_raw"]
    )
    generated = validate_resolved_updates(without_score, candidates, seed)
    assert generated["frozen_margin_capture_interaction_raw"].tolist() == [0.2]
    bad = updates.copy()
    bad["frozen_margin_capture_interaction_raw"] = 0.3
    with pytest.raises(ValueError, match="does not match"):
        validate_resolved_updates(bad, candidates, seed)
