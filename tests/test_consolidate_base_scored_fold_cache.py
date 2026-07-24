from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.consolidate_base_scored_fold_cache import consolidate_fold_cache


def test_consolidates_fold_cache_atomically(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    pq.write_table(pa.table({"row_id": [1, 2], "score": [0.1, 0.2]}), cache / "a.parquet")
    pq.write_table(pa.table({"score": [0.3], "row_id": [3]}), cache / "b.parquet")
    output = tmp_path / "ledger.parquet"

    manifest = consolidate_fold_cache(cache, output, expected_parts=2, batch_rows=1)

    assert manifest["part_count"] == 2
    assert manifest["row_count"] == 3
    assert pq.ParquetFile(output).metadata.num_rows == 3
