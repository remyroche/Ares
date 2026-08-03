from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import (
    materialize_historical_backcast_exact1m_execution_path_labels as labels,
)


def _path(decision: pd.Timestamp) -> str:
    index = pd.date_range(decision, periods=720, freq="1min", tz="UTC")
    return json.dumps(
        {
            "timestamp": index.astype("int64").tolist(),
            "open": np.full(720, 100.0).tolist(),
            "high": np.linspace(100.1, 103.0, 720).tolist(),
            "low": np.full(720, 99.5).tolist(),
            "close": np.linspace(100.0, 102.5, 720).tolist(),
        }
    )


def test_batch_keeps_physical_and_policy_targets_separate() -> None:
    signal = pd.Timestamp("2024-01-01T00:00:00Z")
    decision = signal + pd.Timedelta(hours=1)
    identity = {
        "__ts__": signal,
        "__symbol__": "BTC/USD:USD",
        "side_name": "long",
        "candidate_id": "candidate-1",
    }
    context = pd.DataFrame(
        [
            {
                **identity,
                "__decision_ts__": decision,
                "__barrier_pct__": 0.02,
                "__path_auxiliary_atr_fraction__": 0.01,
                "atr_fraction": 0.01,
                "fee": 0.002,
                "entry_spread": 4.0,
                "exit_spread": 6.0,
                "execution_decision_utc": decision,
                "policy_archetype": "long_default",
                "execution_geometry_key": "long__parent",
                "execution_geometry_source": "side_parent_fallback",
                "execution_gross_ev_12h": 0.01,
                "execution_cost_return": 0.002,
                "execution_net_ev_12h": 0.008,
                "execution_exit_reason": "timeout",
                "execution_exit_hour": 12.0,
                "execution_mfe_return_12h": 0.03,
                "execution_mae_return_12h": 0.005,
                "execution_entry_price": 100.04,
                "execution_exit_price": 102.4,
                "execution_expected_spread_bps": 10.0,
                "execution_entry_half_spread_bps": 4.0,
                "execution_exit_half_spread_bps": 6.0,
                "execution_label_end_utc": decision + pd.Timedelta(hours=12),
                "execution_label_available_at": decision + pd.Timedelta(hours=12),
            }
        ]
    ).set_index("candidate_id")
    paths = pd.DataFrame(
        [{**identity, "execution_future_path": _path(decision)}]
    )
    result = labels._batch(paths, context)
    assert result.loc[0, "__opportunity_occurred_12h__"] == 1
    assert result.loc[0, "__soft_tb_first_event__"] == "favorable_first"
    assert result.loc[0, "execution_net_ev_12h"] == pytest.approx(0.008)
    assert result.loc[0, "__exit_conversion_loss_return_12h__"] > 0.0
    physical = labels._physical_columns(result.columns.tolist())
    assert "__peak_mfe_atr_12h__" in physical
    assert "execution_net_ev_12h" not in physical
