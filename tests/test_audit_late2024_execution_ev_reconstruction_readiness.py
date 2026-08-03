from __future__ import annotations

import pandas as pd

from scripts.audit_late2024_execution_ev_reconstruction_readiness import (
    OLD55_NON_RAW_FIELDS,
    classify_readiness,
    parquet_timestamp_bounds,
)


def _coverage(start: str, end: str) -> dict[str, object]:
    return {
        "files": 1,
        "minimum_utc": pd.Timestamp(start),
        "maximum_utc": pd.Timestamp(end),
    }


def test_january_partial_and_february_forward_exact_1m_are_reconstructible() -> None:
    result = classify_readiness(
        execution_one_minute=_coverage("2025-01-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        source_labels=_coverage("2025-01-01T00:00:00Z", "2025-02-28T23:00:00Z"),
        pit_features=_coverage("2022-07-10T00:00:00Z", "2026-07-21T00:00:00Z"),
        hourly_ohlcv=_coverage("2022-01-01T00:00:00Z", "2026-07-01T00:00:00Z"),
        archived_candidates=_coverage("2025-03-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    )

    assert result["january_2025_exact_1m_12h_expanding_oof"]["status"] == (
        "reconstructible_strict_exact_1m_12h_oof_partial_after_warmup"
    )
    assert result["february_2025_exact_1m_12h_forward_oof"]["status"] == (
        "reconstructible_strict_exact_1m_12h_forward_oof"
    )
    assert result["late_2024_hourly_comparator"]["status"] == (
        "reconstructible_hourly_comparator_only_no_1m_policy_or_timing_parity"
    )
    assert result["old55_exact_score_contract"]["status"] == (
        "unavailable_must_use_fold_local_raw_pit_base_candidate_score"
    )


def test_missing_minute_resolution_fails_closed_even_when_hourly_history_exists() -> None:
    result = classify_readiness(
        execution_one_minute=_coverage("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"),
        source_labels=_coverage("2025-01-01T00:00:00Z", "2025-02-28T23:00:00Z"),
        pit_features=_coverage("2024-01-01T00:00:00Z", "2025-03-01T00:00:00Z"),
        hourly_ohlcv=_coverage("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        archived_candidates={},
    )
    assert result["february_2025_exact_1m_12h_forward_oof"]["status"].startswith("unavailable")
    assert result["late_2024_hourly_comparator"]["status"] == (
        "reconstructible_hourly_comparator_only_no_1m_policy_or_timing_parity"
    )


def test_february_requires_candidate_source_through_the_end_of_february() -> None:
    result = classify_readiness(
        execution_one_minute=_coverage("2025-01-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        source_labels=_coverage("2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z"),
        pit_features=_coverage("2022-07-10T00:00:00Z", "2025-03-01T00:00:00Z"),
        hourly_ohlcv=_coverage("2022-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        archived_candidates={},
    )
    assert result["february_2025_exact_1m_12h_forward_oof"]["status"].startswith("unavailable")


def test_late_2024_cannot_be_promoted_to_one_minute_parity() -> None:
    result = classify_readiness(
        execution_one_minute=_coverage("2025-01-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        source_labels=_coverage("2025-01-01T00:00:00Z", "2025-03-01T00:00:00Z"),
        pit_features=_coverage("2022-07-10T00:00:00Z", "2025-03-01T00:00:00Z"),
        hourly_ohlcv=_coverage("2022-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        archived_candidates={},
    )
    tier = result["late_2024_hourly_comparator"]
    assert "hourly_comparator_only" in tier["status"]
    assert "one-minute exit-geometry parity" in tier["forbidden_claims"]
    assert not tier["requirements_met"][
        "candidate_level_complete_one_minute_universe_certified"
    ]


def test_old55_is_rejected_even_if_a_static_store_contains_some_columns() -> None:
    result = classify_readiness(
        execution_one_minute=_coverage("2025-01-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        source_labels=_coverage("2025-01-01T00:00:00Z", "2025-03-01T00:00:00Z"),
        pit_features=_coverage("2022-01-01T00:00:00Z", "2025-03-01T00:00:00Z"),
        hourly_ohlcv=_coverage("2022-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        archived_candidates={},
        old55_feature_columns=OLD55_NON_RAW_FIELDS[:2],
    )
    contract = result["old55_exact_score_contract"]
    assert contract["status"] == "unavailable_must_use_fold_local_raw_pit_base_candidate_score"
    assert set(contract["static_feature_columns_missing"]) == set(OLD55_NON_RAW_FIELDS[2:])


def test_timestamp_inventory_rejects_files_without_requested_clock(tmp_path) -> None:
    good = tmp_path / "good.parquet"
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=3, freq="min", tz="UTC")}).to_parquet(good, index=False)
    pd.DataFrame({"not_time": [1, 2]}).to_parquet(bad, index=False)
    inventory = parquet_timestamp_bounds([good, bad], ("ts",))
    assert inventory["files"] == 1
    assert inventory["rows"] == 3
    assert str(bad) in inventory["unreadable_files"]
