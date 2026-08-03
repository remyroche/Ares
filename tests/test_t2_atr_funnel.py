import json

import numpy as np
import pandas as pd

from extreme_price_movements.t2_atr_funnel import BarrierGeometry, materialize_geometry_events, soft_event_targets
from extreme_price_movements.config import T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS
from scripts.run_t2_atr_sequential_funnel import _resolved_before


def _path(high, low, close):
    return json.dumps({"timestamp": list(range(720)), "high": high + [1.0] * (720 - len(high)), "low": low + [1.0] * (720 - len(low)), "close": close + [1.0] * (720 - len(close))})


def test_short_uses_signed_not_reciprocal_return_and_ties_are_adverse():
    rows = pd.DataFrame({"candidate_id": ["a"], "side_name": ["short"], "atr_1h": [.01], "decision_price": [1.0], "execution_future_path": [_path([1.02], [.98], [1.0])]})
    event = materialize_geometry_events(rows, BarrierGeometry(2.0, 1.0)).iloc[0]
    assert event.lower_first == 1.0
    assert event.upper_first == 0.0
    assert bool(event.same_minute_conflict)


def test_soft_target_is_a_distribution():
    rows = pd.DataFrame({"candidate_id": ["a"], "side_name": ["long"], "atr_1h": [.01], "decision_price": [1.0], "execution_future_path": [_path([1.01], [.995], [1.005])]})
    event = materialize_geometry_events(rows, BarrierGeometry(2.0, 1.0))
    target = soft_event_targets(event, BarrierGeometry(2.0, 1.0), temperature_atr=.25)
    assert np.allclose(target.sum(axis=1), 1.0)
    assert target.shape == (1, 3)


def test_t2_does_not_alias_realised_cost_as_a_feature():
    assert "execution_cost_return" not in T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS
    assert "causal_entry_cost_bps" not in T2_FUNNEL_BASE_CONTEXT_FEATURE_KEYS


def test_train_labels_are_purged_by_resolution_not_feature_timestamp():
    test = pd.DataFrame({"__decision_ts__": pd.to_datetime(["2024-04-01 01:00:00Z"])})
    train = pd.DataFrame(
        {
            "candidate_id": ["resolved", "leaked"],
            "__decision_ts__": pd.to_datetime(["2024-03-31 00:00:00Z", "2024-03-31 23:00:00Z"]),
            "__label_available_at__": pd.to_datetime(["2024-03-31 12:00:00Z", "2024-04-01 12:00:00Z"]),
        }
    )
    assert _resolved_before(train, test).candidate_id.tolist() == ["resolved"]
