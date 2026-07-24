from __future__ import annotations

import pandas as pd

from scripts.monitor_short_default_uncertainty_blocks import _label_blocks


def test_block_monitor_requires_cooling_before_a_new_block() -> None:
    daily = pd.DataFrame(
        {
            "day": pd.date_range("2026-04-01", periods=7, freq="D", tz="UTC"),
            "conjunction_active": [True, False, True, False, False, True, False],
            "active_rows": [1, 0, 1, 0, 0, 1, 0],
        }
    )
    labeled = _label_blocks(daily, cooling_days=2)
    assert labeled.loc[:2, "block_id"].tolist() == [
        "forward_block_001",
        "forward_block_001",
        "forward_block_001",
    ]
    assert labeled.loc[5, "block_id"] == "forward_block_002"
