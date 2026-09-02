from __future__ import annotations

import pandas as pd

from scripts.execution.materialize_hf_kraken_orderbook_recap import SourceFile, _select_files, _snapshot_rows


def test_drifting_source_minutes_select_latest_snapshot_before_fixed_boundary() -> None:
    files = [
        SourceFile("2026-01-01", f"book/{minute:04d}.parquet", "", 1, minute)
        for minute in (1, 5, 12, 16, 25, 29)
    ]
    selected = _select_files(files, retained_cadence_minutes=15)
    assert [item.minute_of_day for item in selected] == [12, 29]
    assert [item.decision_ts for item in selected] == [
        "2026-01-01T00:15:00+00:00", "2026-01-01T00:30:00+00:00",
    ]


def test_snapshot_recap_discards_raw_book_columns_and_marks_spot_fallback() -> None:
    source = SourceFile(
        "2026-01-01", "data/crypto/book/2026-01-01/0001.parquet", "oid", 10, 1,
        "2026-01-01T00:15:00+00:00",
    )
    source_frame = pd.DataFrame({
        "ts": [1767226200000], "pair": ["BTC/USD"],
        "bids_json": ['[[100.0, 2.0]]'], "asks_json": ['[[101.0, 2.0]]'],
    })
    recap = _snapshot_rows(source_frame, source=source, notionals=(100.0,))
    assert not {"bids_json", "asks_json"}.intersection(recap.columns)
    assert recap.loc[0, "source_market"] == "spot"
    assert recap.loc[0, "market_selection"] == "spot_fallback_no_futures_in_abraxasccs_dataset"
    assert not bool(recap.loc[0, "raw_trade_data_retained"])
    assert bool(recap.loc[0, "source_available_by_decision"])

