from __future__ import annotations

import pandas as pd

from scripts.materialize_native_l2_continuation_from_snapshots import _history_files


def test_history_file_discovery_is_limited_to_canonical_orderbook_history(tmp_path) -> None:
    root = tmp_path / "spread_snapshots"
    canonical = root / "orderbook_history" / "date=2026-07-11" / "snapshots.parquet"
    canonical.parent.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(canonical)
    pd.DataFrame({"x": [1]}).to_parquet(root / "latest_orderbooks.parquet")
    assert _history_files(root) == [canonical]
