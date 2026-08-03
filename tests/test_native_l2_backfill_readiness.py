from __future__ import annotations

from extreme_price_movements.native_l2_backfill_readiness import (
    assess_candidate_window,
    aggregate_inventory,
)


def test_proxy_inventory_cannot_pass_native_window_gate() -> None:
    inventory = aggregate_inventory(
        [
            {
                "rows": 100,
                "native_rows": 0,
                "proxy_rows": 100,
                "classification": "proxy_only",
                "native_min_ts": None,
                "native_max_ts": None,
            }
        ]
    )
    result = assess_candidate_window(
        inventory,
        [
            {
                "panel_id": "panel",
                "rows": 10,
                "symbols": 2,
                "min_candidate_ts": "2026-05-01T00:00:00+00:00",
                "max_candidate_ts": "2026-07-23T00:00:00+00:00",
            }
        ],
    )
    assert result["historical_native_backfill_required"] is True
    assert result["native_window_contains_declared_candidate_window"] is False


def test_native_window_gate_requires_both_bounds() -> None:
    inventory = aggregate_inventory(
        [
            {
                "rows": 20,
                "native_rows": 20,
                "proxy_rows": 0,
                "classification": "native_exact",
                "native_min_ts": "2026-05-01T00:00:00+00:00",
                "native_max_ts": "2026-07-23T00:00:00+00:00",
            }
        ]
    )
    result = assess_candidate_window(
        inventory,
        [
            {
                "panel_id": "panel",
                "rows": 10,
                "symbols": 2,
                "min_candidate_ts": "2026-05-01T00:00:00+00:00",
                "max_candidate_ts": "2026-07-23T00:00:00+00:00",
            }
        ],
    )
    assert result["native_window_contains_declared_candidate_window"] is True
    assert result["historical_native_backfill_required"] is False


def test_native_day_coverage_is_aggregated_without_filling_gaps() -> None:
    inventory = aggregate_inventory(
        [
            {
                "rows": 3,
                "native_rows": 3,
                "proxy_rows": 0,
                "classification": "native_exact",
                "native_min_ts": "2026-07-11T00:00:00+00:00",
                "native_max_ts": "2026-07-13T00:00:00+00:00",
                "native_day_counts": {"2026-07-11": 2, "2026-07-13": 1},
            }
        ]
    )
    assert inventory["native_day_counts"] == {
        "2026-07-11": 2,
        "2026-07-13": 1,
    }
    assert inventory["native_coverage_days"] == 2
    assert inventory["native_missing_calendar_days"] == ["2026-07-12"]
