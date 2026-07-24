from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_materialized_trailing_label_topk_lgbm_hpo import (
    _timestamp_side_ranks,
)


def test_timestamp_side_rank_is_local_and_symbol_ties_are_deterministic() -> None:
    rows: list[dict[str, object]] = []
    scores: list[float] = []
    sides: list[int] = []
    for timestamp in ("2026-04-01T00:00:00Z", "2026-04-01T01:00:00Z"):
        for side in (-1, 1):
            for symbol in ("CCC", "AAA", "BBB", "DDD"):
                rows.append({"__ts__": timestamp, "__symbol__": symbol})
                scores.append(0.5 if symbol in {"AAA", "BBB"} else 0.1)
                sides.append(side)
    frame = pd.DataFrame(rows)
    ranked = _timestamp_side_ranks(frame, np.asarray(scores), np.asarray(sides))

    assert ranked["group_rows"].eq(4).all()
    for start in range(0, len(frame), 4):
        local = ranked.iloc[start : start + 4]
        by_symbol = dict(zip(frame.iloc[start : start + 4]["__symbol__"], local["rank"]))
        assert by_symbol["AAA"] == 1
        assert by_symbol["BBB"] == 2


def test_timestamp_side_top30_keeps_each_stream_represented() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": np.repeat(
                pd.to_datetime(["2026-04-01", "2026-04-02"], utc=True), 20
            ),
            "__symbol__": [f"S{i:02d}" for i in range(20)] * 2,
        }
    )
    sides = np.tile(np.repeat([-1, 1], 10), 2)
    scores = np.arange(len(frame), dtype=np.float64)
    ranked = _timestamp_side_ranks(frame, scores, sides)
    selected = ranked["rank"].to_numpy() <= np.ceil(
        ranked["group_rows"].to_numpy() * 0.30
    )
    keys = pd.DataFrame(
        {
            "ts": frame["__ts__"],
            "side": np.where(sides < 0, "short", "long"),
            "selected": selected,
        }
    )
    assert keys.groupby(["ts", "side"])["selected"].sum().eq(3).all()
