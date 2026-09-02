from __future__ import annotations

import pandas as pd
import pytest

from src.execution.tardis_book import IncrementalL2Book, iter_complete_book_states, validate_snapshot5_top_of_book


def _rows(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def test_tardis_snapshot_replaces_book_and_updates_are_atomic() -> None:
    rows = _rows([
        {"timestamp": "2026-01-01T00:00:00Z", "local_timestamp": "2026-01-01T00:00:00Z", "is_snapshot": True, "side": "bid", "price": 100.0, "amount": 2.0},
        {"timestamp": "2026-01-01T00:00:00Z", "local_timestamp": "2026-01-01T00:00:00Z", "is_snapshot": True, "side": "ask", "price": 101.0, "amount": 3.0},
        # Four rows are one source message: no intermediate, partly-updated
        # book may be observed by the downstream state producer.
        {"timestamp": "2026-01-01T00:00:01Z", "local_timestamp": "2026-01-01T00:00:01Z", "is_snapshot": False, "side": "bid", "price": 100.0, "amount": 0.0},
        {"timestamp": "2026-01-01T00:00:01Z", "local_timestamp": "2026-01-01T00:00:01Z", "is_snapshot": False, "side": "bid", "price": 99.0, "amount": 5.0},
        {"timestamp": "2026-01-01T00:00:01Z", "local_timestamp": "2026-01-01T00:00:01Z", "is_snapshot": False, "side": "ask", "price": 101.0, "amount": 0.0},
        {"timestamp": "2026-01-01T00:00:01Z", "local_timestamp": "2026-01-01T00:00:01Z", "is_snapshot": False, "side": "ask", "price": 102.0, "amount": 1.0},
    ])
    states = list(iter_complete_book_states(rows))
    assert len(states) == 2
    assert states[0].best_bid == 100.0
    assert states[1].best_bid == 99.0
    assert states[1].best_ask == 102.0
    assert states[1].source_rows == 4
    assert states[1].valid


def test_tardis_drops_pre_snapshot_updates_and_rejects_reordered_messages() -> None:
    book = IncrementalL2Book()
    before = _rows([{"timestamp": "2026-01-01T00:00:00Z", "local_timestamp": "2026-01-01T00:00:00Z", "is_snapshot": False, "side": "bid", "price": 100.0, "amount": 1.0}])
    assert book.apply_message(before) is None
    snapshot = _rows([
        {"timestamp": "2026-01-01T00:00:02Z", "local_timestamp": "2026-01-01T00:00:02Z", "is_snapshot": True, "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": "2026-01-01T00:00:02Z", "local_timestamp": "2026-01-01T00:00:02Z", "is_snapshot": True, "side": "ask", "price": 101.0, "amount": 1.0},
    ])
    assert book.apply_message(snapshot) is not None
    with pytest.raises(ValueError, match="not ordered"):
        book.apply_message(before)


def test_tardis_microsecond_epoch_is_not_interpreted_as_nanoseconds() -> None:
    rows = _rows([
        {"timestamp": 1735689601615431, "local_timestamp": 1735689601730526, "is_snapshot": True, "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": 1735689601615431, "local_timestamp": 1735689601730526, "is_snapshot": True, "side": "ask", "price": 101.0, "amount": 1.0},
    ])
    state = list(iter_complete_book_states(rows))[0]
    assert state.local_timestamp.year == 2025
    assert state.exchange_timestamp.year == 2025


def test_stream_records_match_dataframe_message_semantics() -> None:
    records = [
        {"timestamp": "1735689601615431", "local_timestamp": "1735689601730526", "is_snapshot": "true", "side": "bid", "price": "100", "amount": "1"},
        {"timestamp": "1735689601615431", "local_timestamp": "1735689601730526", "is_snapshot": "true", "side": "ask", "price": "101", "amount": "2"},
    ]
    from_frame = IncrementalL2Book().apply_message(_rows(records))
    from_stream = IncrementalL2Book().apply_records(records)
    assert from_frame == from_stream


def test_snapshot5_validation_never_uses_future_reconstruction() -> None:
    reconstructed = pd.DataFrame({
        "local_timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:02Z"], utc=True),
        "best_bid": [100.0, 102.0], "best_ask": [101.0, 103.0],
    })
    snapshots = pd.DataFrame({
        "local_timestamp": pd.to_datetime(["2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"], utc=True),
        "best_bid": [100.0, 102.0], "best_ask": [101.0, 103.0],
    })
    audit = validate_snapshot5_top_of_book(reconstructed, snapshots)
    assert audit["matched_state_ts"].tolist() == reconstructed["local_timestamp"].tolist()
    assert audit["within_tolerance"].all()
