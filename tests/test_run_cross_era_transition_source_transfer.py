from __future__ import annotations

import pandas as pd

from scripts.run_cross_era_transition_source_transfer import _source_frame
from scripts.run_cross_era_transition_source_transfer import (
    _prior_top10_selection,
)


def test_current_source_excludes_frozen_forward_rows() -> None:
    frame = pd.DataFrame(
        {
            "source_family": [
                "current_exact_spread_mayjul2026",
                "current_exact_spread_mayjul2026",
                "canonical_spread_febapr2025",
            ],
            "horizon_hours": [12, 12, 12],
            "context_available": [True, True, True],
            "mapping_provenance_role": [
                "strict_oof",
                "frozen_forward_oos",
                "strict_oof",
            ],
        }
    )
    result = _source_frame(frame, "current_exact_spread_strict_oof")
    assert len(result) == 1
    assert result["mapping_provenance_role"].eq("strict_oof").all()


def test_prior_top10_selection_is_deterministic_and_score_independent() -> None:
    frame = pd.DataFrame(
        {
            "cohort_anchor_utc": pd.date_range(
                "2026-01-01", periods=20, freq="h", tz="UTC"
            ),
            "source_family": ["source"] * 20,
            "mapping_provenance_role": ["strict_oof"] * 20,
            "prediction": range(20),
        }
    )
    selected = _prior_top10_selection(frame)
    reversed_scores = frame.copy()
    reversed_scores["prediction"] = list(reversed(range(20)))
    assert selected.sum() == 2
    assert selected.tolist() == _prior_top10_selection(
        reversed_scores
    ).tolist()
