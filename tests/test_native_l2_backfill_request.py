from __future__ import annotations

import pandas as pd

from extreme_price_movements.native_l2_backfill_request import build_backfill_request


def test_backfill_request_uses_only_symbol_day_and_marks_missing_pairs(tmp_path):
    candidate = pd.DataFrame(
        {
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD", "ETH/USD:USD"],
            "available_at": pd.to_datetime(
                ["2026-04-01T00:00:00Z", "2026-04-02T00:00:00Z", "2026-04-01T00:00:00Z"],
                utc=True,
            ),
            "label_like_future_field": [1, 2, 3],
        }
    )
    candidate_path = tmp_path / "candidate.parquet"
    candidate.to_parquet(candidate_path, index=False)
    native = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD"],
            "snapshot_ts": pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True),
            "l2_spread_bps": [10.0],
            "future_like_score": [999.0],
        }
    )
    native_path = tmp_path / "native.parquet"
    native.to_parquet(native_path, index=False)

    requirements, summary = build_backfill_request(
        [
            {
                "panel_id": "test_panel",
                "path": str(candidate_path),
                "time_column": "available_at",
            }
        ],
        root=tmp_path,
        native_sidecar=native_path,
    )

    assert requirements[["symbol", "utc_day"]].values.tolist() == [
        ["BTC/USD:USD", "2026-04-01"],
        ["ETH/USD:USD", "2026-04-01"],
        ["BTC/USD:USD", "2026-04-02"],
    ]
    assert requirements["native_coverage"].tolist() == [True, False, False]
    assert summary["candidate_symbol_count"] == 2
    assert summary["currently_covered_symbol_day_pairs"] == 1
    assert summary["missing_symbol_day_pairs"] == 2
