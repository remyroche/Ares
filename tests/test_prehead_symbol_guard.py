import pandas as pd

from extreme_price_movements.inference.prehead_symbol_guard import (
    prehead_symbol_guard_result,
)


def test_prehead_symbol_guard_blocks_matching_head_side_symbol():
    state = {
        "policy_name": "A1_l4of5_24h",
        "as_of": "2026-06-26T00:00:00Z",
        "blocked": {
            "long_bars": {
                "long": ["DASH/USD:USD"],
            }
        },
    }
    result = prehead_symbol_guard_result(
        symbol="DASH/USD:USD",
        strategy_id="long_bars_example",
        side="long",
        state=state,
        enabled=True,
        now="2026-06-30T00:00:00Z",
        max_state_age_days=7.0,
    )
    assert result.blocked is True
    assert result.reason == "prehead_symbol_guard_block"


def test_prehead_symbol_guard_stale_state_fails_open():
    state = {
        "policy_name": "A1_l4of5_24h",
        "as_of": "2026-06-20T00:00:00Z",
        "blocked": {
            "long_bars": {
                "long": ["DASH/USD:USD"],
            }
        },
    }
    result = prehead_symbol_guard_result(
        symbol="DASH/USD:USD",
        strategy_id="long_bars_example",
        side="long",
        state=state,
        enabled=True,
        now=pd.Timestamp("2026-06-30T00:00:00Z"),
        max_state_age_days=7.0,
    )
    assert result.blocked is False
    assert result.reason == "stale_state"
