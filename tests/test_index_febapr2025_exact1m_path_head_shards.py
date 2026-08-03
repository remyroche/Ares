from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.index_febapr2025_exact1m_path_head_shards import _coverage


def _shard(offset: int, rows: int) -> dict:
    return {"source_slice": {"offset": offset, "expected_rows": rows}, "labels": {"rows": rows}}


def test_coverage_requires_exact_non_overlapping_source_offsets() -> None:
    complete = _coverage([_shard(0, 5), _shard(5, 3)], 8)
    assert complete["complete"] is True
    assert complete["gaps"] == []
    assert complete["overlap"] == []

    incomplete = _coverage([_shard(0, 5), _shard(6, 2)], 8)
    assert incomplete["complete"] is False
    assert incomplete["gaps"] == [[5, 6]]

    overlapping = _coverage([_shard(0, 5), _shard(4, 4)], 8)
    assert overlapping["complete"] is False
    assert overlapping["overlap"] == [[4, 5]]
