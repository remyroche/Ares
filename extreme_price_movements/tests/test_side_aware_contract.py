from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.side_aware import (
    add_side_contract_columns,
    candidate_id_series,
    expand_side_candidates,
    normalise_side_array,
    side_adjust_return,
    side_aware_path_metrics,
    validate_side_candidate_contract,
)


def test_side_adjust_return_makes_short_profit_positive() -> None:
    raw = np.array([0.04, -0.04], dtype=np.float32)
    adjusted = side_adjust_return(raw, [1, -1])

    assert adjusted.tolist() == pytest.approx([0.04, 0.04])


def test_path_metrics_mirror_long_and_short_adverse_favorable() -> None:
    out = side_aware_path_metrics(
        entry_price=[100.0, 100.0],
        future_high=[110.0, 110.0],
        future_low=[95.0, 95.0],
        future_close=[106.0, 94.0],
        side=[1, -1],
    )

    assert out.loc[0, "side_adjusted_return"] == pytest.approx(0.06)
    assert out.loc[0, "adverse_excursion"] == pytest.approx(100.0 / 95.0 - 1.0)
    assert out.loc[0, "favorable_excursion"] == pytest.approx(0.10)
    assert out.loc[1, "side_adjusted_return"] == pytest.approx(0.06)
    assert out.loc[1, "adverse_excursion"] == pytest.approx(0.10)
    assert out.loc[1, "favorable_excursion"] == pytest.approx(100.0 / 95.0 - 1.0)


def test_candidate_id_includes_side_and_timeframe() -> None:
    ts = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:00"], utc=True)
    ids = candidate_id_series(ts, ["BTC", "BTC"], "1h", [1, -1])

    assert ids.iloc[0] == "BTC|2026-01-01T00:00:00Z|1h|long"
    assert ids.iloc[1] == "BTC|2026-01-01T00:00:00Z|1h|short"
    assert ids.nunique() == 2


def test_add_side_contract_columns_and_validate() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"], utc=True),
            "__symbol__": ["BTC", "ETH"],
        }
    )

    out = add_side_contract_columns(
        frame,
        side="short",
        timestamp_col="__ts__",
        asset_col="__symbol__",
        timeframe="1h",
    )
    stats = validate_side_candidate_contract(out)

    assert normalise_side_array(out["side"]).tolist() == [-1, -1]
    assert out["side_name"].tolist() == ["short", "short"]
    assert out["candidate_id"].is_unique
    assert stats["short_rows"] == 2


def test_expand_side_candidates_duplicates_each_asset_timestamp_by_side() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"], utc=True),
            "__symbol__": ["BTC", "ETH"],
            "raw_score": [0.1, 0.2],
        }
    )

    out = expand_side_candidates(
        frame,
        timestamp_col="__ts__",
        asset_col="__symbol__",
        timeframe="1h",
    )
    stats = validate_side_candidate_contract(out)

    assert len(out) == 4
    assert stats["long_rows"] == 2
    assert stats["short_rows"] == 2
    assert out["candidate_id"].is_unique
    grouped_sides = out.groupby(["__symbol__", "__ts__"])["side_name"].apply(set).tolist()
    assert grouped_sides == [{"long", "short"}, {"long", "short"}]


def test_validate_side_candidate_contract_rejects_duplicate_ids() -> None:
    frame = pd.DataFrame({"side": [1, -1], "candidate_id": ["x", "x"]})

    with pytest.raises(ValueError, match="candidate_id is not unique"):
        validate_side_candidate_contract(frame)
