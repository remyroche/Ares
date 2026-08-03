from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from scripts.materialize_stage_i_2024_2026_surface import (
    LABEL_COLUMNS,
    R3_PRIMITIVES,
    _finalize_coverage,
    _feature_availability,
    _join_packb_exact_labels,
    _join_pit_store_by_symbol_exact,
    _require_completed_exact_tp6_sidecar,
    _store_features_at_signal_close,
    _validate_exact_join,
    audit_2024_pit_reference_readiness,
    declared_feature_contract,
    expand_feature_refs,
    packb_month_preflight,
)
from extreme_price_movements.stage_i_feature_selection import (
    resolve_stage_i_feature_universe,
    stage_i_active_contracts,
)


def test_expansion_preserves_layer_and_side_separation() -> None:
    cfg = {
        "base_shared_feature_keys": ["BASE_SHARED"], "base_long_feature_keys": ["LONG"],
        "base_short_feature_keys": ["SHORT"], "meta_shared_feature_keys": ["META"],
        "BASE_SHARED": ["base_common"], "LONG": ["long_only"], "SHORT": ["short_only"], "META": ["meta_only"],
    }
    result = declared_feature_contract(cfg)
    assert result["base_long"] == ["base_common", "long_only"]
    assert result["base_short"] == ["base_common", "short_only"]
    assert "long_only" not in result["meta_shared_residual"]
    assert "meta_only" not in result["base_long"]


def test_meta_surface_contract_is_exact_selector_union_with_explicit_provenance() -> None:
    cfg = {
        "base_shared_feature_keys": ["BASE"], "base_long_feature_keys": [], "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["META_RAW", "META_BASE_PERFORMANCE_FEATURE_KEYS"],
        "meta_product_feature_keys": ["PRODUCT"],
        "STAGE_I_M6_SHARED_UNION_META_FEATURE_KEYS": ["BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS"],
        "BASE": ["base"], "META_RAW": ["market_context"], "PRODUCT": ["trend_slope_24h"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": ["base_oof_score"],
        "META_MODEL_UNCERTAINTY_FEATURE_KEYS": [], "META_RECENT_EFFECTIVENESS_FEATURE_KEYS": [],
        "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS": [],
        "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS": ["leaf_support_mean"],
    }
    meta_head = next(item for item in stage_i_active_contracts() if item.layer == "meta")
    expected = resolve_stage_i_feature_universe(cfg, layer="meta", side="long", head=meta_head.head)
    contract = declared_feature_contract(cfg)
    assert contract["meta_selector"] == expected
    assert contract["meta_shared_residual"] == expected
    assert contract["meta_raw_store"] == ["market_context", "trend_slope_24h"]
    assert contract["meta_generated_later"] == ["base_oof_score", "leaf_support_mean"]
    assert set(contract["meta_raw_store"]).isdisjoint(contract["meta_generated_later"])
    assert set(contract["meta_raw_store"]) | set(contract["meta_generated_later"]) == set(expected)


def test_expansion_rejects_cyclic_config_lists() -> None:
    with pytest.raises(ValueError, match="cyclic"):
        expand_feature_refs(["A"], {"A": ["B"], "B": ["A"]})


def test_availability_reports_missing_without_synthesising_values() -> None:
    result = _feature_availability({"base_long": ["present", "missing"]}, {"present"})
    assert result["base_long"]["present"] == ["present"]
    assert result["base_long"]["missing"] == ["missing"]


def test_exact_join_enforces_h12_and_cost_once() -> None:
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    panel = pd.DataFrame({"candidate_id": ["x"], "__ts__": [ts], "__symbol__": ["BTC_USD:USD"], "side_name": ["long"], "__decision_ts__": [ts + pd.Timedelta(hours=1)]})
    labels = panel.copy()
    labels["tp6_sl4_entry_price"] = 100.
    labels["t2_tp6_sl4_event"] = 0
    labels["t2_tp6_sl4_exit_minute"] = 2
    labels["t4_tp6_sl4_exit_pnl_atr"] = 6.
    labels["t4_tp6_sl4_gross_bps"] = 150.
    labels["t4_tp6_sl4_net_bps"] = 50.
    labels["t4_tp6_sl4_terminal_pnl_atr"] = 7.
    labels["__label_available_at__"] = ts + pd.Timedelta(hours=13)
    robust = panel.copy()
    robust["label_valid"] = True
    robust["pre_adverse_mfe_atr"] = 3.0
    robust["lower_touch_minute"] = -1
    robust["robust_clear_event_b25"] = True
    robust["robust_clear_soft_b25_t50"] = .8
    output = _validate_exact_join(panel, labels, robust)
    assert len(output) == 1
    assert set(R3_PRIMITIVES).issubset(output)
    assert output.loc[0, "label_available_ts"] == ts + pd.Timedelta(hours=13)
    labels.loc[0, "t4_tp6_sl4_net_bps"] = 51.
    with pytest.raises(ValueError, match="cost"):
        _validate_exact_join(panel, labels, robust)


def test_completed_sidecar_requires_exact_contract_and_r3_attestation(tmp_path) -> None:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    (sidecar / "manifest.json").write_text(json.dumps({
        "complete": True,
        "contract": {"geometry": "TP6/SL4/H12", "cost_bps": 100, "robust_clear": "B25 T50"},
    }))
    assert _require_completed_exact_tp6_sidecar(sidecar, require_r3=True)["complete"] is True
    (sidecar / "manifest.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="not complete"):
        _require_completed_exact_tp6_sidecar(sidecar, require_r3=True)


def test_coverage_gate_never_treats_null_or_constant_as_eligible() -> None:
    result = _finalize_coverage([
        {"source_id": "packb", "month": "2025-01", "layer": "base", "side": "long", "feature": "good", "rows": 10, "present": True, "nonnull": 9, "min": 1., "max": 2.},
        {"source_id": "packb", "month": "2025-01", "layer": "base", "side": "long", "feature": "sparse", "rows": 10, "present": True, "nonnull": 8, "min": 1., "max": 2.},
        {"source_id": "packb", "month": "2025-01", "layer": "base", "side": "long", "feature": "flat", "rows": 10, "present": True, "nonnull": 10, "min": 1., "max": 1.},
    ]).set_index("feature")
    assert bool(result.loc["good", "fit_eligible"])
    assert not bool(result.loc["sparse", "fit_eligible"])
    assert not bool(result.loc["flat", "fit_eligible"])


def test_packb_store_join_is_exact_timestamp_not_asof(tmp_path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    pd.DataFrame({"ts": [ts, ts + pd.Timedelta(hours=2)], "causal_feature": [1.0, 3.0]}).to_parquet(
        store / "symbol=BTC_USD:USD.parquet", index=False
    )
    candidates = pd.DataFrame({"candidate_id": ["a", "b"], "__ts__": [ts, ts + pd.Timedelta(hours=1)], "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"]})
    joined = _store_features_at_signal_close(candidates, store=store, fields=["causal_feature"], start=ts, end=ts + pd.Timedelta(hours=3))
    assert joined.causal_feature.tolist()[0] == 1.0
    assert pd.isna(joined.causal_feature.tolist()[1])


def test_indexed_pit_store_join_is_exact_and_preserves_candidate_identity(tmp_path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    indexed = pd.DataFrame({"causal_feature": [1.0, 3.0]}, index=pd.DatetimeIndex([ts, ts + pd.Timedelta(hours=2)], name="ts"))
    indexed.to_parquet(store / "symbol=BTC_USD:USD.parquet")
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b"], "__ts__": [ts, ts + pd.Timedelta(hours=1)],
        "__symbol__": ["BTC_USD:USD", "BTC_USD:USD"], "side_name": ["long", "short"],
        "__decision_ts__": [ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
    })
    joined = _join_pit_store_by_symbol_exact(candidates, store=store, fields=["causal_feature"])
    assert joined.causal_feature.tolist()[0] == 1.0
    assert pd.isna(joined.causal_feature.tolist()[1])
    assert joined.candidate_id.tolist() == ["a", "b"]


def test_2024_reference_readiness_requires_exact_signal_store_contract(tmp_path) -> None:
    panel = tmp_path / "panel"
    (panel / "parts").mkdir(parents=True)
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b"], "__ts__": [ts, ts + pd.Timedelta(hours=1)],
        "__symbol__": ["BTC_USD:USD", "BTC_USD:USD"], "side_name": ["long", "short"],
        "__decision_ts__": [ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=2)],
    })
    candidates.to_parquet(panel / "parts" / "symbol=BTC_USD:USD.parquet", index=False)
    store = tmp_path / "store"
    store.mkdir()
    pd.DataFrame({"f1": [1.0, 2.0]}, index=pd.DatetimeIndex([ts, ts + pd.Timedelta(hours=1)], name="ts")).to_parquet(
        store / "symbol=BTC_USD:USD.parquet"
    )
    (store / "symbol=BTC_USD:USD.meta.json").write_text(json.dumps({"first_ts": ts.isoformat(), "last_ts": (ts + pd.Timedelta(hours=1)).isoformat()}))
    contract = {"base_long": ["f1", "f2"], "base_short": ["f1"], "meta_raw_store": ["f1"], "meta_generated_later": []}
    report = audit_2024_pit_reference_readiness(panel_2024=panel, feature_store=store, contract=contract)
    assert report[0]["status"] == "READY_REFERENCE_EXACT_JOIN_REQUIRED"
    assert report[0]["signal_close_to_decision_plus_1h"]
    assert report[0]["raw_feature_schema"]["base_long"] == {"declared": 2, "present_any_store": 1, "present_every_store": 1}
    assert "never asof" in report[0]["no_lookahead_rule"]


def test_packb_preflight_estimates_raw_matrix_and_keeps_reference_default(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame({"__symbol__": ["BTC/USD:USD", "BTC/USD:USD"]}).to_parquet(
        labels / "train_global_long_5_2025_01.parquet", index=False
    )
    pd.DataFrame({"__symbol__": ["BTC/USD:USD"]}).to_parquet(
        labels / "train_global_short_5_2025_01.parquet", index=False
    )
    store = tmp_path / "store"
    store.mkdir()
    pd.DataFrame({"ts": [pd.Timestamp("2025-01-01T00:00:00Z")], "f1": [1.0]}).to_parquet(
        store / "symbol=BTC_USD:USD.parquet", index=False
    )
    contract = {"base_long": ["f1"], "base_short": ["f2"], "meta_raw_store": ["f1", "f3"]}
    result = packb_month_preflight("2025-01", contract=contract, labels_root=labels, feature_store=store, disk_path=tmp_path)
    assert result["rows"] == 3
    assert result["raw_store_fields_present_any_symbol"] == ["f1"]
    assert result["estimated_uncompressed_float32_bytes"] == 12
    assert result["free_disk_bytes"] > 0
    assert result["default_layout"] == "reference_identity_labels"


def test_packb_exact_join_requires_identical_candidates_and_emits_label_alias() -> None:
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    candidates = pd.DataFrame({"candidate_id": ["x"], "__ts__": [ts], "__symbol__": ["BTC/USD:USD"], "side_name": ["long"], "__decision_ts__": [ts + pd.Timedelta(hours=1)]})
    labels = candidates.copy()
    labels["tp6_sl4_entry_price"] = 100.
    labels["t2_tp6_sl4_event"] = 0
    labels["t2_tp6_sl4_exit_minute"] = 1
    labels["t4_tp6_sl4_exit_pnl_atr"] = 6.
    labels["t4_tp6_sl4_gross_bps"] = 150.
    labels["t4_tp6_sl4_net_bps"] = 50.
    labels["t4_tp6_sl4_terminal_pnl_atr"] = 6.
    labels["__label_available_at__"] = ts + pd.Timedelta(hours=13)
    labels["label_valid"] = True
    labels["pre_adverse_mfe_atr"] = 3.
    labels["lower_touch_minute"] = -1
    labels["robust_clear_event_b25"] = True
    labels["robust_clear_soft_b25_t50"] = .8
    result = _join_packb_exact_labels(candidates, labels)
    assert result.loc[0, "label_available_ts"] == labels.loc[0, "__label_available_at__"]
    labels.loc[0, "candidate_id"] = "other"
    with pytest.raises(ValueError, match="match exactly"):
        _join_packb_exact_labels(candidates, labels)


def test_packb_exact_join_binds_decision_time_and_only_prices_valid_rows() -> None:
    ts = pd.Timestamp("2025-01-01T00:00:00Z")
    candidates = pd.DataFrame({"candidate_id": ["x"], "__ts__": [ts], "__symbol__": ["BTC/USD:USD"], "side_name": ["long"], "__decision_ts__": [ts + pd.Timedelta(hours=1)]})
    labels = candidates.copy()
    labels["tp6_sl4_entry_price"] = np.nan
    labels["t2_tp6_sl4_event"] = np.nan
    labels["t2_tp6_sl4_exit_minute"] = np.nan
    labels["t4_tp6_sl4_exit_pnl_atr"] = np.nan
    labels["t4_tp6_sl4_gross_bps"] = np.nan
    labels["t4_tp6_sl4_net_bps"] = np.nan
    labels["t4_tp6_sl4_terminal_pnl_atr"] = np.nan
    labels["__label_available_at__"] = ts + pd.Timedelta(hours=13)
    labels["label_valid"] = False
    labels["pre_adverse_mfe_atr"] = np.nan
    labels["lower_touch_minute"] = np.nan
    labels["robust_clear_event_b25"] = np.nan
    labels["robust_clear_soft_b25_t50"] = np.nan
    result = _join_packb_exact_labels(candidates, labels)
    assert not bool(result.loc[0, "label_valid"])
    labels.loc[0, "__decision_ts__"] = ts + pd.Timedelta(hours=2)
    with pytest.raises(ValueError, match="__decision_ts__"):
        _join_packb_exact_labels(candidates, labels)
