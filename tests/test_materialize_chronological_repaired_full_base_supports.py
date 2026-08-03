from __future__ import annotations

import pandas as pd
import pytest

from scripts import materialize_chronological_repaired_full_base_supports as materialize


def test_folds_are_contiguous_and_cover_march_20_to_30() -> None:
    folds = materialize.march_folds()
    assert folds[0][1] == pd.Timestamp("2025-03-20T00:00:00Z")
    assert folds[-1][2] == pd.Timestamp("2025-03-31T00:00:00Z")
    assert all(left[2] == right[1] for left, right in zip(folds, folds[1:]))


def test_strict_train_mask_excludes_unresolved_and_future_rows() -> None:
    start = pd.Timestamp("2025-03-23T00:00:00Z")
    frame = pd.DataFrame(
        {
            "__ts__": [
                pd.Timestamp("2025-03-22T00:00:00Z"),
                pd.Timestamp("2025-03-22T12:00:00Z"),
                pd.Timestamp("2025-03-23T00:00:00Z"),
                pd.Timestamp("2025-03-21T00:00:00Z"),
            ],
            "execution_label_end_utc": [
                pd.Timestamp("2025-03-22T12:00:00Z"),
                pd.Timestamp("2025-03-23T00:00:00Z"),
                pd.Timestamp("2025-03-23T12:00:00Z"),
                pd.Timestamp("2025-03-24T00:00:00Z"),
            ],
        }
    )
    assert materialize.strict_train_mask(frame, start).tolist() == [True, False, False, False]


def test_candidate_population_rejects_non_oof_and_excludes_april() -> None:
    rows = []
    for candidate, timestamp, good in (
        ("m", "2025-03-20T00:00:00Z", True),
        ("a", "2025-04-01T00:00:00Z", True),
    ):
        rows.append(
            {
                "candidate_id": candidate,
                "side_name": "long",
                "__symbol__": "BTCUSDT",
                "__ts__": pd.Timestamp(timestamp),
                "model_development_eligible": True,
                "candidate_score_is_oof": good,
                "upstream_scores_are_outer_oof": good,
                "residual_is_oof": good,
            }
        )
    result = materialize.candidate_population(pd.DataFrame(rows))
    assert result.candidate_id.tolist() == ["m"]
    bad = pd.DataFrame(rows)
    bad.loc[0, "residual_is_oof"] = False
    with pytest.raises(materialize.SupportMaterializationError, match="strict upstream OOF"):
        materialize.candidate_population(bad)


def test_frozen_configs_reject_feature_drift() -> None:
    repair = {
        "repair": {
            "selected_configs": [
                {"target": "hard0", "arm": "S0", "geometry": "fixed_d5"}
            ] * 8
        }
    }
    source = {"features": {"primary_arms": {"S0": {"long": ["wrong"], "short": ["wrong"]}}}}
    with pytest.raises(materialize.SupportMaterializationError, match="eight frozen configs"):
        materialize.frozen_configs(repair, source)
