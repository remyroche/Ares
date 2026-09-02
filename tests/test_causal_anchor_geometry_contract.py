from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.causal_anchor_geometry import (
    ANCHOR_FEATURE_GROUPS,
    AnchorGeometryConfig,
    CausalAnchorGeometryEngine,
)


def _bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=420, freq="15min", tz="UTC")
    trend = 100.0 + np.linspace(0.0, 8.0, len(index)) + np.sin(np.arange(len(index)) / 4.0)
    return pd.DataFrame({
        "open": trend - .10,
        "high": trend + .35,
        "low": trend - .35,
        "close": trend,
        "volume": 100.0 + np.arange(len(index)) % 7,
        "open_interest": 1_000.0 + np.arange(len(index)) * .5,
    }, index=index)


def test_anchor_feature_hierarchy_is_nested() -> None:
    groups = list(ANCHOR_FEATURE_GROUPS.values())
    for previous, current in zip(groups, groups[1:]):
        assert set(previous).issubset(current)
    assert "m4_kalman_transition" in ANCHOR_FEATURE_GROUPS


def test_anchor_rows_are_causal_and_snapshot_identity_is_unique() -> None:
    bars = _bars()
    target = bars.index[300]
    events, snapshots, states = CausalAnchorGeometryEngine(
        "TEST/USD:USD", bars,
        output_start=bars.index[180], output_end=bars.index[360],
        snapshot_targets={target: [{"target_kind": "entry", "target_id": "test", "candidate_id": "test"}]},
        config=AnchorGeometryConfig(max_active_anchors=24),
    ).run()
    assert not events.empty
    assert (pd.to_datetime(events.label_available_ts, utc=True) > pd.to_datetime(events.event_ts, utc=True)).all()
    assert not snapshots.duplicated(["candidate_id", "snapshot_ts", "anchor_role"]).any()
    assert not states.duplicated(["__symbol__", "state_ts", "anchor_role"]).any()
    assert pd.to_datetime(states.state_ts, utc=True).eq(target).all()
    assert events.loc[:, ["anchor_kf_distance_level", "anchor_kf_innovation_z"]].notna().all().all()
