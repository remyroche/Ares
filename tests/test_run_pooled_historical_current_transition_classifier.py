from __future__ import annotations

import pandas as pd

from scripts.run_pooled_historical_current_transition_classifier import CURRENT_SOURCE, TARGETS, eligible_rows


def test_eligible_rows_keep_historical_non_oof_but_exclude_current_forward() -> None:
    rows = []
    for source, provenance in [
        ("historical_backcast_2022_2023_non_oof", "historical_non_oof_backcast"),
        (CURRENT_SOURCE, "strict_oof"),
        (CURRENT_SOURCE, "frozen_forward_oos"),
    ]:
        row = {"source_family": source, "mapping_provenance_role": provenance, "context_available": True, "cv_group_id": "g"}
        for target in TARGETS:
            row[target] = 0.0
        rows.append(row)
    result = eligible_rows(pd.DataFrame(rows))
    assert len(result) == 2
    assert not ((result["source_family"].eq(CURRENT_SOURCE)) & result["mapping_provenance_role"].ne("strict_oof")).any()


def test_target_surface_covers_active_onset_recovery_and_reversal() -> None:
    assert TARGETS == (
        "target__active_adverse",
        "target__adverse_onset_within_3h",
        "target__lifecycle_recovery_within_3h",
        "target__lifecycle_reversal_after_recovery_within_3h",
    )
