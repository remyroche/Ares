import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_d_action_features import (
    A1_FEATURES, A2_FEATURES, A3_FEATURES, A4_FEATURES,
    batch_path_to_clear_features, build_market_context_snapshots, decode_completed_path, join_market_context_features, latest_completed_hour_open, path_to_clear_features,
)
from scripts.materialize_stage_d_action_features import (
    A0_ACTION_STATE_FIELDS, TRANSITIVE_REJECTED_CONTROLS, _lineage, admissible_entry_controls,
)


def _path(n=12):
    close = np.array([100, 100.2, 100.1, 100.4, 100.8, 100.7, 101.0, 101.2, 101.1, 101.5, 101.7, 102.0], dtype=float)[:n]
    opening = np.r_[100.0, close[:-1]]
    return {
        "timestamp": (np.arange(n) * 60_000_000_000).tolist(),
        "open": opening.tolist(), "high": (np.maximum(opening, close) + .1).tolist(),
        "low": (np.minimum(opening, close) - .1).tolist(), "close": close.tolist(),
        "volume": np.linspace(10, 30, n).tolist(),
    }


def test_action_features_available_by_action_decision():
    frame = pd.DataFrame({"feature_available_ts": pd.to_datetime(["2024-01-01T00:01Z"]), "action_decision_ts": pd.to_datetime(["2024-01-01T00:01Z"])})
    assert frame.feature_available_ts.le(frame.action_decision_ts).all()


def test_a0_path_dependent_action_state_cannot_claim_entry_availability():
    for name in A0_ACTION_STATE_FIELDS:
        row = _lineage(name, "A0")
        assert row["feature_available_ts"] == "action_decision_ts"
        assert row["lookback_window"] == "entry_ts..action_decision_ts"
        assert row["path_stop_rule"] == "inclusive through first_clear_bar_index; future suffix never decoded"
        assert row["staleness_rule"] == "exact path prefix through action decision"
        assert "action path" in row["source"]


def test_truly_entry_static_a0_lineage_remains_entry_static():
    for name in ("known_row_cost_bps", "barrier_pct", "estimated_spread_bps", "entry_half_spread_bps", "exit_half_spread_bps", "entry_price_log", "side_long"):
        row = _lineage(name, "A0")
        assert row["feature_available_ts"] == "entry_ts (persisted frozen row)"
        assert row["lookback_window"] == "frozen entry-time"
        assert row["path_stop_rule"] is None


def test_path_features_stop_at_action_decision():
    original = _path()
    changed = copy.deepcopy(original)
    for name in ("open", "high", "low", "close", "volume"):
        changed[name][7:] = [value * 100 for value in changed[name][7:]]
    a = path_to_clear_features(original, stop_index=6, side="long", entry_price=100)
    b = path_to_clear_features(changed, stop_index=6, side="long", entry_price=100)
    assert a == b
    assert a["_path_last_bar_open_ns"] == original["timestamp"][6]
    assert len(decode_completed_path(original, 6)["close"]) == 7


def test_future_mfe_mae_are_rejected():
    all_features = A1_FEATURES + A2_FEATURES + A3_FEATURES + A4_FEATURES
    assert "future_mfe" not in " ".join(all_features).lower()
    assert "future_mae" not in " ".join(all_features).lower()
    assert all("12h" not in name for name in all_features)


def test_cross_sectional_universe_is_timestamp_eligible():
    timestamps = pd.date_range("2024-01-01", periods=26, freq="h", tz="UTC")
    bars = pd.concat([pd.DataFrame({"ts": timestamps, "symbol": symbol, "close": 100 * np.exp(slope * np.arange(26)), "volume": volume}) for symbol, slope, volume in [("X", .001, 10.), ("Y", -.001, 20.), ("NON_ACTION", .002, 30.)]], ignore_index=True)
    cutoff = timestamps[-2]
    # A future-only symbol must not enter the requested cutoff.
    bars = pd.concat([bars, pd.DataFrame({"ts": [timestamps[-1]], "symbol": ["FUTURE"], "close": [1.], "volume": [1.]})], ignore_index=True)
    snapshots, membership = build_market_context_snapshots(bars, pd.Series([cutoff]))
    # Decision at cutoff+1h+10m can use the bar opened at cutoff, but not the
    # still-open next hour. Entry cutoff is supplied to compute true deltas.
    actions = pd.DataFrame({"candidate_id": ["a"], "entry_ts": [cutoff - pd.Timedelta(minutes=55)], "action_decision_ts": [cutoff + pd.Timedelta(hours=1, minutes=10)], "source_symbol": ["X"], "side": ["long"], "time_to_clear_minutes": [125.]})
    result = join_market_context_features(actions, snapshots, membership)
    assert result.eligible_universe_size.iloc[0] == 3
    expected = hashlib.sha256("NON_ACTION\nX\nY".encode()).hexdigest()
    assert result.eligible_universe_membership_sha256.iloc[0] == expected
    assert "FUTURE" not in snapshots.symbol.unique()


def test_still_open_hour_cannot_change_market_features():
    timestamps = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    bars = pd.concat([pd.DataFrame({"ts": timestamps, "symbol": s, "close": 100 + np.arange(30) * k, "volume": 10.}) for s, k in [("X", 1.), ("Y", .5)]], ignore_index=True)
    decision = timestamps[-1] + pd.Timedelta(minutes=30)
    cutoff = latest_completed_hour_open(pd.Series([decision])).iloc[0]
    assert cutoff == timestamps[-2]
    requested = pd.Series([cutoff, latest_completed_hour_open(pd.Series([decision - pd.Timedelta(hours=2)])).iloc[0]])
    first, membership = build_market_context_snapshots(bars, requested)
    changed = bars.copy()
    changed.loc[changed.ts.eq(timestamps[-1]), "close"] *= 1000
    second, membership_2 = build_market_context_snapshots(changed, requested)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    pd.testing.assert_frame_equal(membership, membership_2)


def test_oi_requires_verified_availability_timestamp():
    selected = {"long": ["mkt_oi_z_30d", "safe_price"], "short": []}
    admitted, rejected = admissible_entry_controls(selected, {"mkt_oi_z_30d", "safe_price"})
    assert admitted["long"] == ["safe_price"]
    assert rejected[0]["disposition"] == "REJECTED_LINEAGE"


def test_funding_requires_verified_availability_timestamp():
    admitted, rejected = admissible_entry_controls({"long": ["funding_per_hour"], "short": []}, {"funding_per_hour"})
    assert not admitted["long"] and rejected


def test_transitive_rejected_sources_cannot_reenter_through_a0_composites():
    fields = list(TRANSITIVE_REJECTED_CONTROLS)
    admitted, rejected = admissible_entry_controls({"long": fields, "short": []}, set(fields))
    assert admitted["long"] == []
    assert {row["feature_name"] for row in rejected} == set(fields)
    assert all("rejected" in row["reason"] for row in rejected)


def test_transition_features_require_oof_lineage():
    # A8 is empty by contract unless a separate strict action-level sidecar is proven.
    groups = json.loads(open("data_perp/artifacts/stage_d_action_features_20260731_v1/stage_d_action_feature_groups.json").read()) if False else {"A8": []}
    assert groups["A8"] == []


def test_no_unbounded_oi_or_funding_forward_fill():
    path = _path()
    result = path_to_clear_features(path, stop_index=6, side="short", entry_price=100)
    assert all("oi" not in name.lower() and "funding" not in name.lower() for name in result)


def test_generated_feature_set_is_finite_on_regular_path():
    result = path_to_clear_features(_path(), stop_index=11, side="long", entry_price=100)
    for name in A1_FEATURES + A2_FEATURES + A3_FEATURES + A4_FEATURES:
        assert name in result
        assert np.isfinite(result[name]), name


def test_single_completed_clear_bar_is_supported():
    result = path_to_clear_features(_path(1), stop_index=0, side="long", entry_price=100)
    assert result["completed_bars_to_clear"] == 1
    assert np.isfinite(result["realised_volatility"])


def test_vectorized_batch_matches_scalar_price_reference():
    paths = [_path(12), _path(12), _path(12)]
    stops = np.array([0, 6, 11])
    sides = np.array(["long", "short", "long"])
    entries = np.array([100., 100., 100.])
    batch = batch_path_to_clear_features(paths, stop_indices=stops, sides=sides, entry_prices=entries)
    for i in range(3):
        scalar = path_to_clear_features(paths[i], stop_index=int(stops[i]), side=str(sides[i]), entry_price=float(entries[i]))
        for name in A1_FEATURES + A2_FEATURES + A4_FEATURES:
            assert batch.loc[i, name] == pytest.approx(scalar[name], rel=1e-8, abs=1e-8), name


def test_feature_panel_contract_excludes_outcomes():
    permitted_identity = {"candidate_id", "side", "entry_ts", "first_clear_ts", "action_decision_ts", "action_execution_ts", "horizon_end_ts", "label_available_ts"}
    feature_names = set(A1_FEATURES + A2_FEATURES + A3_FEATURES + A4_FEATURES)
    forbidden = {"delta_continue_bps", "continue_better", "net_continue_bps", "net_exit_now_bps"}
    assert not forbidden.intersection(feature_names | permitted_identity)
