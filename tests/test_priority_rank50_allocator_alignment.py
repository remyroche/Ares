import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_priority_rank50_capital_pressure_allocator import _accepted_with_candidates


def _candidate(
    *,
    timestamp: str,
    symbol: str,
    side: str = "long",
    net_return: float = 0.01,
) -> dict:
    ts = pd.Timestamp(timestamp, tz="UTC")
    return {
        "timestamp": ts,
        "symbol": symbol,
        "side": side,
        "strategy_id": f"{side}_candidate",
        "policy_archetype": f"{side}__test",
        "normalized_rank_score": 0.95,
        "base_strategy_threshold": 0.90,
        "calibrated_score": 0.95,
        "entry_price": 100.0,
        "exit_timestamp": ts + pd.Timedelta(hours=1),
        "exit_price": 101.0,
        "net_return": net_return,
        "gross_return": net_return + 0.01,
        "holding_bars": 4,
        "simple_policy_exit_reason": "tp",
        "fees_bps": 100.0,
        "slippage_bps": 0.0,
    }


def test_accepted_rows_use_replay_normalized_candidate_order() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(timestamp="2026-07-02T00:00:00Z", symbol="ZZZ/USD:USD"),
            _candidate(timestamp="2026-07-01T00:00:00Z", symbol="AAA/USD:USD"),
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "candidate_index": 0,
                "accepted": True,
                "position_size": 100.0,
                "position_exit_timestamp": pd.Timestamp("2026-07-01T01:00:00Z"),
                "position_net_return": 0.01,
                "position_gross_return": 0.02,
                "position_exit_reason": "tp",
                "position_exit_price": 101.0,
            },
            {
                "candidate_index": 1,
                "accepted": True,
                "position_size": 100.0,
                "position_exit_timestamp": pd.Timestamp("2026-07-02T01:00:00Z"),
                "position_net_return": 0.01,
                "position_gross_return": 0.02,
                "position_exit_reason": "tp",
                "position_exit_price": 101.0,
            },
        ]
    )

    accepted = _accepted_with_candidates(candidates, decisions)

    assert accepted["symbol"].tolist() == ["AAA/USD:USD", "ZZZ/USD:USD"]
    assert (
        pd.to_datetime(accepted["exit_timestamp"], utc=True)
        >= pd.to_datetime(accepted["timestamp"], utc=True)
    ).all()
