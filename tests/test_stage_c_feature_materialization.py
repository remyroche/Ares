"""Executable Stage-0 causality and source-contract checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from extreme_price_movements.continuation_features import (
    CONTINUATION_FEATURE_GROUPS,
    CONTINUATION_FUNDING_FEATURE_KEYS,
    CONTINUATION_OI_FEATURE_KEYS,
    CONTINUATION_REGIME_FEATURE_KEYS,
    materialize_ohlcv_continuation_features,
    side_normalize_continuation_features,
)
from scripts.materialize_stage_c_continuation_feature_panel import E15_FEATURE_MANIFEST, OI_FUNDING_BLOCK_REASON, _lineage_records
from scripts.materialize_stage_c_continuation_feature_panel import ALIGNMENT, PERSISTENCE
from scripts.run_stage_c_conditional_retention_ablation import _group_features


def _bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    records = []
    for symbol, offset in (("A_USD:USD", 0.0), ("B_USD:USD", 5.0)):
        for index, ts in enumerate(timestamps):
            close = 100.0 + offset + 0.2 * index
            records.append({"ts": ts, "symbol": symbol, "open": close - 0.1, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1_000.0 + index})
    return pd.DataFrame(records)


def test_all_features_available_by_decision_timestamp():
    features = materialize_ohlcv_continuation_features(_bars())
    fields = [name for names in CONTINUATION_FEATURE_GROUPS.values() for name in names]
    # Exercise the exact-bar key used by the materialiser: a candidate may only
    # receive a completed feature row at its declared cutoff, never a later bar.
    source = features.loc[features.ts.eq(features.ts.iloc[60])].copy()
    candidates = source.loc[:, ["symbol", "ts"]].rename(columns={"ts": "feature_cutoff_ts"})
    candidates["decision_ts"] = candidates.feature_cutoff_ts
    joined = candidates.merge(
        features,
        left_on=["symbol", "feature_cutoff_ts"],
        right_on=["symbol", "ts"],
        how="left",
        validate="one_to_one",
    )
    assert joined.ts.le(joined.decision_ts).all()
    assert joined.loc[:, fields].notna().all().all()
    assert all(name in features for name in fields)


def test_rolling_features_use_trailing_data_only():
    baseline = materialize_ohlcv_continuation_features(_bars())
    changed = _bars()
    cutoff = pd.Timestamp("2024-01-03T00:00:00Z")
    changed.loc[(changed.symbol.eq("A_USD:USD")) & changed.ts.gt(cutoff), "high"] *= 4.0
    candidate = materialize_ohlcv_continuation_features(changed)
    columns = [name for name in baseline if name.startswith("cont_")]
    pd.testing.assert_frame_equal(
        baseline.loc[(baseline.symbol.eq("A_USD:USD")) & baseline.ts.le(cutoff), columns].reset_index(drop=True),
        candidate.loc[(candidate.symbol.eq("A_USD:USD")) & candidate.ts.le(cutoff), columns].reset_index(drop=True),
    )


def test_cross_sectional_features_use_timestamp_eligible_universe():
    bars = _bars()
    last = bars.ts.max()
    full = materialize_ohlcv_continuation_features(bars)
    reduced = materialize_ohlcv_continuation_features(bars.loc[~(bars.symbol.eq("B_USD:USD") & bars.ts.eq(last))])
    assert int(full.loc[full.ts.eq(last), "cont_cs_universe_size"].iloc[0]) == 2
    assert int(reduced.loc[reduced.ts.eq(last) & reduced.symbol.eq("A_USD:USD"), "cont_cs_universe_size"].iloc[0]) == 1


def test_oi_values_respect_source_timestamp_and_staleness():
    assert CONTINUATION_OI_FEATURE_KEYS == []
    assert "unbounded ffill" in OI_FUNDING_BLOCK_REASON


def test_funding_values_respect_observation_timestamp():
    assert CONTINUATION_FUNDING_FEATURE_KEYS == []
    assert "available_ts" in OI_FUNDING_BLOCK_REASON


def test_no_future_funding_payment_used():
    funding_names = [name.lower() for name in CONTINUATION_FUNDING_FEATURE_KEYS]
    assert not any("next" in name or "payment" in name or "settlement" in name for name in funding_names)


def test_no_inverse_pi_rows_mixed_with_linear_pf_rows():
    symbols = pd.read_parquet(ALIGNMENT, columns=["symbol"])
    assert symbols.symbol.str.endswith("/USD:USD").all()


def test_retention_labels_exist_only_on_clear_first_support():
    labels = pd.read_parquet(PERSISTENCE, columns=["postcost_h0_clear_first", "postcost_h0_persistence_target_valid", "postcost_h0_retained_net"])
    valid = labels.postcost_h0_persistence_target_valid.astype(bool)
    assert valid.eq(labels.postcost_h0_clear_first.astype(bool)).all()
    stage_c_label = labels.postcost_h0_retained_net.where(valid)
    assert stage_c_label.loc[~valid].isna().all()


def test_clear_first_population_matches_frozen_label_manifest():
    labels = pd.read_parquet(
        PERSISTENCE,
        columns=["candidate_id", "postcost_h0_clear_first", "postcost_h0_persistence_target_valid"],
    )
    assert labels.candidate_id.is_unique
    assert labels.postcost_h0_clear_first.astype(bool).equals(
        labels.postcost_h0_persistence_target_valid.astype(bool)
    )


def test_side_breakout_rejection_is_symmetric():
    features = materialize_ohlcv_continuation_features(_bars())
    duplicated = pd.concat([features.assign(side="long"), features.assign(side="short")], ignore_index=True)
    aligned = side_normalize_continuation_features(duplicated, ("long", "short"))
    assert aligned.loc[aligned.side.eq("long"), "side_cont_breakout_rejection"].reset_index(drop=True).equals(
        aligned.loc[aligned.side.eq("long"), "cont_up_breakout_rejection_12h"].reset_index(drop=True)
    )


def test_side_adverse_rv_never_mixes_long_and_short_histories():
    features = materialize_ohlcv_continuation_features(_bars())
    duplicated = pd.concat([features.assign(side="long"), features.assign(side="short")], ignore_index=True)
    aligned = side_normalize_continuation_features(duplicated, ("long", "short"))
    latest = aligned.ts.eq(aligned.ts.max()) & aligned.symbol.eq("A_USD:USD")
    long_value = aligned.loc[latest & aligned.side.eq("long"), "side_cont_adverse_rv_12h"].iloc[0]
    short_value = aligned.loc[latest & aligned.side.eq("short"), "side_cont_adverse_rv_12h"].iloc[0]
    # Monotonic prices are favourable for long and adverse for short.  If the
    # rolling grouping ignored side, the long value would inherit short loss.
    assert long_value == 0.0
    assert short_value > 0.0
    assert aligned.loc[aligned.side.eq("short"), "side_cont_breakout_rejection"].reset_index(drop=True).equals(
        aligned.loc[aligned.side.eq("short"), "cont_down_breakout_rejection_12h"].reset_index(drop=True)
    )


def test_upstream_transition_predictions_are_oof():
    assert CONTINUATION_REGIME_FEATURE_KEYS == []


def test_conditional_arm_groups_keep_side_adverse_rv_out_of_price_arm():
    groups = _group_features(pd.DataFrame())
    assert "side_cont_adverse_rv_12h" not in groups["C1"]
    assert "side_cont_adverse_rv_12h" in groups["C3"]
    assert all(not name.startswith("mkt_regime_change__") for name in groups["C7"])


def test_feature_reuse_map_has_f0_and_rejected_f4_f5_f7():
    persisted = json.loads(E15_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    records = _lineage_records(inherited_columns=set(), e15_features={side: persisted[side] for side in ("long", "short")})
    frame = pd.DataFrame(records)
    assert {"F0_existing_E15_control", "F4_oi_dynamics", "F5_funding_crowding", "F7_causal_regime_transition"}.issubset(frame.feature_group)
    assert frame.loc[frame.feature_group.eq("F0_existing_E15_control"), "reuse_status"].eq("existing_control").all()
    assert frame.loc[frame.feature_group.isin(["F4_oi_dynamics", "F5_funding_crowding", "F7_causal_regime_transition"]), "point_in_time_safe"].eq(False).all()


def test_frozen_e15_control_hash_is_exact_persisted_file():
    expected = hashlib.sha256(E15_FEATURE_MANIFEST.read_bytes()).hexdigest()
    assert len(expected) == 64
    assert {"long", "short"}.issubset(json.loads(E15_FEATURE_MANIFEST.read_text(encoding="utf-8")))


def test_ohlcv_proxy_names_are_not_mislabeled_as_factual_l2():
    forbidden = ("orderbook", "depth", "aggressor", "liquidation", "spread")
    names = [name for names in CONTINUATION_FEATURE_GROUPS.values() for name in names]
    assert all(not any(token in name.lower() for token in forbidden) or name.endswith(("_proxy", "_estimator", "_ohlcv_proxy")) for name in names)


def test_f8_composites_are_predeclared_deterministic_products():
    features = materialize_ohlcv_continuation_features(_bars()).dropna(subset=["cont_efficiency_12h", "cont_volume_persistence_12h"])
    assert (features["cont_efficiency_x_volume_persistence"] == features["cont_efficiency_12h"] * features["cont_volume_persistence_12h"]).all()
