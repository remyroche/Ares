from __future__ import annotations

import numpy as np

from scripts.materialize_historical_exact_h12_postcost_events import classify_postcost_path


def test_clear_cost_before_adverse_is_a_clean_event() -> None:
    high = np.ones(720)
    low = np.ones(720)
    high[2] = 1.011
    low[7] = 0.97
    event, favorable, adverse, resolved = classify_postcost_path(
        high=high, low=low, entry_price=1.0, side="long", adverse_barrier_pct=0.02, cost_bps=100.0, hurdle_bps=0.0,
    )
    assert (event, favorable, adverse, resolved) == ("clear_cost_first", 2, 7, 2)


def test_same_minute_ohlc_dual_hit_is_conservative_conflict() -> None:
    high = np.ones(720)
    low = np.ones(720)
    high[4] = 1.02
    low[4] = 0.97
    event, favorable, adverse, resolved = classify_postcost_path(
        high=high, low=low, entry_price=1.0, side="long", adverse_barrier_pct=0.02, cost_bps=100.0, hurdle_bps=0.0,
    )
    assert (event, favorable, adverse, resolved) == ("adverse_first_or_conflict", 4, 4, 4)


def test_short_uses_low_for_favorable_and_high_for_adverse() -> None:
    high = np.ones(720)
    low = np.ones(720)
    low[3] = 0.985
    high[9] = 1.03
    event, favorable, adverse, resolved = classify_postcost_path(
        high=high, low=low, entry_price=1.0, side="short", adverse_barrier_pct=0.02, cost_bps=100.0, hurdle_bps=0.0,
    )
    assert (event, favorable, adverse, resolved) == ("clear_cost_first", 3, 9, 3)
