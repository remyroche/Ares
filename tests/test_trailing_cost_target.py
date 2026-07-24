from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.trailing_cost_target import (
    CausalSpreadP90Spec,
    build_trailing_cost_targets,
    causal_p90_spread_cost,
    pooled_asset_p90_spread_cost,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-06-10 03:00Z", "2026-06-11 03:00Z"], utc=True),
            "__symbol__": ["A/USD:USD", "A/USD:USD"],
            "side_name": ["long", "short"],
            "__first_touch_target_soft__": [0.4, 0.6],
            "__first_touch_capture_net__": [0.01, -0.01],
            "__first_touch_round_trip_cost__": [0.01, 0.01],
            "__trailing_profit_activated__": [1.0, 0.0],
            "__trailing_profit_activation_bar__": [2.0, 16.0],
            "__first_touch_timeout__": [0.0, 1.0],
            "__first_touch_full_path_mae_norm__": [0.4, 1.5],
        }
    )


def test_p90_cost_is_strictly_prior_and_does_not_use_same_timestamp() -> None:
    history = pd.DataFrame(
        {
            "observed_ts": pd.to_datetime([
                "2026-06-10 01:00Z", "2026-06-10 02:00Z", "2026-06-10 03:00Z", "2026-06-11 02:00Z"
            ], utc=True),
            "symbol": ["A/USD:USD"] * 4,
            "spread_bps": [10.0, 20.0, 1000.0, 30.0],
        }
    )
    cost = causal_p90_spread_cost(
        _rows(), history,
        spec=CausalSpreadP90Spec(lookback_days=28, min_observations=2, min_distinct_days=1),
    )
    # First row can see only 10/20 bps; its contemporaneous 1000 bps print is excluded.
    assert np.isclose(cost.loc[0, "p90_spread_bps"], 19.0)
    assert not cost.loc[0, "p90_round_trip_cost"] == 0.003 + 1000.0 / 10_000.0
    assert cost.loc[1, "p90_spread_bps"] > 20.0


def test_cost_target_replaces_old_cost_once_and_falls_back_without_support() -> None:
    rows = _rows()
    cost = pd.DataFrame(
        {
            "p90_round_trip_cost": [0.0035, np.nan],
            "p90_spread_bps": [20.0, np.nan],
        }
    )
    target = build_trailing_cost_targets(rows, cost)
    # gross = 1% old net + 1% embedded old cost = 2%; new net is 2% - 0.35%.
    assert np.isclose(target.loc[0, "capture_gross_reconstructed"], 0.02)
    assert np.isclose(target.loc[0, "capture_net_p90_spread_fee30bps"], 0.0165)
    assert target.loc[0, "p90_cost_observed"] == 1.0
    assert target.loc[1, "p90_cost_observed"] == 0.0
    assert np.isclose(target.loc[1, "target_soft_p90_trailing_blend"], 0.6)


def test_pooled_asset_p90_provides_full_history_proxy_without_claiming_causality() -> None:
    history = pd.DataFrame(
        {
            "observed_ts": pd.to_datetime(["2026-07-01 00:00Z"] * 3, utc=True),
            "symbol": ["A/USD:USD"] * 3,
            "spread_bps": [10.0, 20.0, 30.0],
        }
    )
    cost = pooled_asset_p90_spread_cost(
        _rows(), history,
        spec=CausalSpreadP90Spec(min_observations=3, min_distinct_days=1),
    )
    assert np.all(cost["p90_spread_cost_available"])
    assert np.allclose(cost["p90_spread_bps"], 28.0)
