from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.causal_profile_geometry import (
    ANCHORED_VWAP_FEATURES,
    CausalProfileGeometryEngine,
    PROFILE_FEATURES,
    PROFILE_STATE_FEATURES,
    VOLATILITY_PARTICIPATION_FEATURES,
)


def _bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3_200, freq="15min", tz="UTC")
    path = 100.0 + np.linspace(0.0, 8.0, len(index)) + np.sin(np.arange(len(index)) / 13.0)
    return pd.DataFrame({
        "open": path - .08, "high": path + .25, "low": path - .30,
        "close": path, "volume": 10.0 + (np.arange(len(index)) % 11),
        "open_interest": 1_000.0 + np.arange(len(index)) * .5,
    }, index=index)


def test_profile_snapshot_is_future_invariant_and_oi_missing_is_optional() -> None:
    bars = _bars()
    snapshot_ts = bars.index[2_800].floor("1h")
    output_start, output_end = snapshot_ts - pd.Timedelta(days=2), snapshot_ts + pd.Timedelta(days=1)
    targets = {snapshot_ts: [{"target_kind": "entry", "target_id": "x", "candidate_id": "x"}]}
    _, baseline, states = CausalProfileGeometryEngine(
        "X/USD:USD", bars, output_start=output_start, output_end=output_end, snapshot_targets=targets,
    ).run()
    changed = bars.copy()
    changed.loc[changed.index > snapshot_ts + pd.Timedelta(hours=2), ["high", "low", "close"]] *= 1.25
    _, perturbed, _ = CausalProfileGeometryEngine(
        "X/USD:USD", changed, output_start=output_start, output_end=output_end, snapshot_targets=targets,
    ).run()
    assert len(baseline) == len(perturbed) == 1
    assert baseline.candidate_id.tolist() == perturbed.candidate_id.tolist() == ["x"]
    np.testing.assert_allclose(
        baseline.loc[:, PROFILE_STATE_FEATURES].to_numpy(float), perturbed.loc[:, PROFILE_STATE_FEATURES].to_numpy(float),
        equal_nan=True, rtol=0.0, atol=0.0,
    )
    assert not states.empty
    assert pd.to_datetime(states.state_ts, utc=True).le(snapshot_ts + pd.Timedelta(days=1)).all()

    no_oi = bars.drop(columns="open_interest")
    _, without_oi, _ = CausalProfileGeometryEngine(
        "X/USD:USD", no_oi, output_start=output_start, output_end=output_end, snapshot_targets=targets,
    ).run()
    assert len(without_oi) == 1
    assert without_oi.loc[:, ["profile_oi_at_price_z", "profile_oi_positioning_imbalance"]].isna().all(axis=None)
    assert without_oi.loc[:, ["profile_poc_distance_atr", "bb_zscore", "donchian_position"]].notna().all(axis=None)


def test_volatility_and_anchored_vwap_blocks_are_causal_state_features() -> None:
    bars = _bars()
    snapshot_ts = bars.index[2_600].floor("1h")
    targets = {snapshot_ts: [{"target_kind": "entry", "target_id": "v", "candidate_id": "v"}]}
    _, snapshots, _ = CausalProfileGeometryEngine(
        "X/USD:USD", bars, output_start=snapshot_ts - pd.Timedelta(days=2),
        output_end=snapshot_ts + pd.Timedelta(hours=1), snapshot_targets=targets,
    ).run()
    assert len(snapshots) == 1
    assert set(VOLATILITY_PARTICIPATION_FEATURES).issubset(snapshots.columns)
    assert set(ANCHORED_VWAP_FEATURES).issubset(snapshots.columns)
    assert set(PROFILE_FEATURES).issubset(PROFILE_STATE_FEATURES)
    assert snapshots.loc[:, [
        "profile_atr_percentile_21d", "profile_relative_volume_4h_24h",
        "profile_session_vwap_distance_atr", "profile_week_vwap_distance_atr",
    ]].notna().all(axis=None)
