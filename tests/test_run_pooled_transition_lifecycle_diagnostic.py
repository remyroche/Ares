from __future__ import annotations

import pandas as pd

from scripts.run_pooled_transition_lifecycle_diagnostic import (
    derive_lifecycle_targets,
    source_balanced_weights,
)


def _frame() -> pd.DataFrame:
    rows = []
    # A contiguous series supports exact timestamp shifts.  The first regime
    # is active, it recovers, and then it reverses to active.
    active = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    for hour, value in enumerate(active):
        stamp = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=hour)
        rows.append({
            "cohort_anchor_utc": stamp,
            "source_family": "fixture_a" if hour % 2 else "fixture_b",
            "horizon_hours": 12,
            "book_fraction": 0.10,
            "target__active_adverse": float(value),
            "target__active_adverse_available_utc": stamp + pd.Timedelta(hours=14),
            "target__adverse_onset_within_3h": float(any(active[hour:min(hour + 3, len(active))])),
            "target__adverse_onset_within_3h_available_utc": stamp + pd.Timedelta(hours=16),
        })
    # Targets are calculated per source lineage, so make the test a single
    # source after the two-source weight assertion below.
    result = pd.DataFrame(rows)
    result["source_family"] = "fixture"
    return result


def test_lifecycle_targets_have_exact_max_availability_and_null_conditioning() -> None:
    result = derive_lifecycle_targets(_frame())
    recovery = result.loc[result["cohort_anchor_utc"].eq(pd.Timestamp("2026-01-01 05:00", tz="UTC"))].iloc[0]
    assert recovery["target__lifecycle_recovery_within_3h"] == 1.0
    assert recovery["target__lifecycle_recovery_within_3h_available_utc"] == pd.Timestamp("2026-01-01 22:00", tz="UTC")
    inactive = result.loc[result["cohort_anchor_utc"].eq(pd.Timestamp("2026-01-01 07:00", tz="UTC"))].iloc[0]
    assert pd.isna(inactive["target__lifecycle_recovery_within_3h"])
    assert result["target__lifecycle_reversal_after_recovery_within_3h"].notna().any()


def test_source_weights_equalize_source_mass() -> None:
    frame = pd.DataFrame({"source_family": ["a", "a", "a", "b"]})
    weights = source_balanced_weights(frame)
    assert abs(weights[:3].sum() - weights[3]) < 1e-12
