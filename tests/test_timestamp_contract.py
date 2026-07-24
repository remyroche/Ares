import pandas as pd

from extreme_price_movements import pipeline_steps
from extreme_price_movements.data_store import (
    _apply_allowed_periods_mask,
    _normalize_feature_index,
)
from extreme_price_movements.timestamp_contract import (
    format_paris_display,
    to_utc_index,
    to_utc_timestamp,
)


def test_naive_legacy_timestamps_are_interpreted_as_utc() -> None:
    timestamp = to_utc_timestamp("2026-07-16 12:34:56")
    index = to_utc_index(["2026-07-16 12:34:56"])

    assert timestamp == pd.Timestamp("2026-07-16T12:34:56Z")
    assert str(index.dtype) == "datetime64[ns, UTC]"


def test_dst_offsets_normalize_to_distinct_utc_instants() -> None:
    before_jump = to_utc_timestamp("2026-03-29T01:30:00+01:00")
    after_jump = to_utc_timestamp("2026-03-29T03:30:00+02:00")

    assert before_jump == pd.Timestamp("2026-03-29T00:30:00Z")
    assert after_jump == pd.Timestamp("2026-03-29T01:30:00Z")
    assert format_paris_display(before_jump) == "2026-03-29 01:30:00 CET"
    assert format_paris_display(after_jump) == "2026-03-29 03:30:00 CEST"


def test_feature_store_preserves_utc_index_and_utc_period_filters() -> None:
    paris_index = to_utc_index(
        ["2026-10-25T02:30:00+02:00", "2026-10-25T02:30:00+01:00"]
    )
    normalized, _, reason = _normalize_feature_index(paris_index, [1.0, 2.0])
    frame = pd.DataFrame({"value": [1.0, 2.0]}, index=normalized)

    selected = _apply_allowed_periods_mask(
        frame,
        [("2026-10-25T00:00:00Z", "2026-10-25T01:00:00Z")],
    )

    assert reason is None
    assert str(normalized.dtype) == "datetime64[ns, UTC]"
    assert selected.index.tolist() == [pd.Timestamp("2026-10-25T00:30:00Z")]


def test_stage_feature_bounds_do_not_mix_naive_and_utc_timestamps(monkeypatch) -> None:
    captured = {}

    def _read_static_features(**kwargs):
        captured.update(kwargs)
        return {"ok": pd.DataFrame()}

    monkeypatch.setattr(pipeline_steps, "read_static_features", _read_static_features)
    pipeline_steps.load_features_for_stage_or_all(
        {
            "_active_stage_view": {
                "stage_name": "utc-boundary",
                "allowed_start_ts": "2026-03-29T01:00:00+01:00",
                "allowed_end_ts": "2026-03-29T03:00:00+02:00",
            }
        },
        pd.Timestamp("2026-03-29T02:00:00Z"),
        "data_perp",
        start_ts=pd.Timestamp("2026-03-29 00:15:00"),
        end_ts=pd.Timestamp("2026-03-29 01:45:00"),
    )

    assert captured["start_ts"] == pd.Timestamp("2026-03-29T00:15:00Z")
    assert captured["end_ts"] == pd.Timestamp("2026-03-29T01:00:00Z")
