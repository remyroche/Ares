import numpy as np
import pandas as pd

from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    SlicePlannerConfig,
    validate_events,
)


def test_validate_events_deduplicates_same_base_quote_variants():
    events = pd.DataFrame(
        {
            "event_id": np.arange(6, dtype=np.int64),
            "symbol": [
                "ETH/USDT",
                "ETH/USDT",
                "ETH/USDC",
                "BTC/USDT",
                "BTC/USDC",
                "SOL/EUR",
            ],
            "t0": pd.date_range("2026-01-01", periods=6, freq="H", tz="UTC"),
            "t1": pd.date_range("2026-01-01", periods=6, freq="H", tz="UTC"),
        }
    )
    cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
    out = validate_events(events, cfg.schema, cfg)

    assert "ETH/USDT" in set(out["symbol"].astype(str))
    assert "ETH/USDC" not in set(out["symbol"].astype(str))
    assert "BTC/USDT" in set(out["symbol"].astype(str))
    assert "BTC/USDC" not in set(out["symbol"].astype(str))
    assert "SOL/EUR" in set(out["symbol"].astype(str))
