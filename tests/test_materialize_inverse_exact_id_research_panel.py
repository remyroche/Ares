from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.materialize_inverse_exact_id_research_panel import (
    ECONOMICS_CONTRACT,
    ECONOMICS_LABELS,
    EVIDENCE_SCOPE,
    LINEAGE,
    POPULATION_LINEAGE,
    PRODUCT_LINEAGE,
    build_panel,
    causal_feature_contract,
    run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _features() -> dict[str, float]:
    asset = [
        "ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_72h", "ret_168h",
        "rv_6h", "rv_24h", "rv_72h", "downside_rv_24h", "atr_fraction_14h",
        "range_12h_fraction", "drawdown_from_12h_high", "recovery_from_12h_low",
        "trend_slope_12h", "path_efficiency_12h", "range_24h_fraction",
        "drawdown_from_24h_high", "recovery_from_24h_low", "trend_slope_24h",
        "path_efficiency_24h", "range_72h_fraction", "drawdown_from_72h_high",
        "recovery_from_72h_low", "trend_slope_72h", "path_efficiency_72h",
        "volume_z_24h", "volume_z_72h", "jump_intensity_24h",
    ]
    market = [
        "market_median_ret_1h", "market_median_ret_4h", "market_median_ret_24h",
        "market_dispersion_1h", "market_dispersion_4h", "market_median_rv_24h",
        "market_median_atr_fraction", "market_breadth_up_1h", "market_negative_breadth_1h",
        "market_breadth_up_4h", "market_negative_breadth_4h", "market_breadth_up_24h",
        "market_negative_breadth_24h", "market_average_pair_corr_24h",
        "btc_minus_alt_median_ret_24h",
    ]
    transition = [f"transition_raw__x{i}__delta_1h" for i in range(25)]
    return {key: float(index) for index, key in enumerate([*asset, *market, *transition])}


def _manifest(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2))


def _fixture(tmp_path: Path) -> argparse.Namespace:
    stage_root, labels_root, coverage_root = (tmp_path / "stage", tmp_path / "labels", tmp_path / "coverage")
    for root in (stage_root, labels_root, coverage_root):
        root.mkdir()
    ts = pd.to_datetime(["2022-01-02T00:00:00Z", "2022-01-02T01:00:00Z"])
    stage = pd.DataFrame([
        {**_features(), "candidate_id": "a", "signal_timestamp": ts[0], "decision_timestamp": ts[0] + pd.Timedelta(hours=1), "path_end_exclusive": ts[0] + pd.Timedelta(hours=13), "symbol": "BTC/USD:BTC", "side_name": "long"},
        {**_features(), "candidate_id": "b", "signal_timestamp": ts[1], "decision_timestamp": ts[1] + pd.Timedelta(hours=1), "path_end_exclusive": ts[1] + pd.Timedelta(hours=13), "symbol": "ETH/USD:ETH", "side_name": "short"},
    ])
    labels = pd.DataFrame([
        {"candidate_id": "a", "__ts__": ts[0], "__symbol__": "BTC/USD:BTC", "side_name": "long", "__decision_ts__": ts[0] + pd.Timedelta(hours=1), "__label_end_ts__": ts[0] + pd.Timedelta(hours=13), "__label_available_at__": ts[0] + pd.Timedelta(hours=13), "execution_net_ev_12h": -0.01, "__opportunity_occurred_12h__": 0.0},
        {"candidate_id": "b", "__ts__": ts[1], "__symbol__": "ETH/USD:ETH", "side_name": "short", "__decision_ts__": ts[1] + pd.Timedelta(hours=1), "__label_end_ts__": ts[1] + pd.Timedelta(hours=13), "__label_available_at__": ts[1] + pd.Timedelta(hours=13), "execution_net_ev_12h": 0.01, "__opportunity_occurred_12h__": 1.0},
    ])
    coverage = pd.DataFrame({"candidate_id": ["a", "b"], "complete": [True, True]})
    stage_path = stage_root / "staged_candidates.parquet"; stage.to_parquet(stage_path, index=False)
    labels_path = labels_root / "joined_multitask_labels.parquet"; labels.to_parquet(labels_path, index=False)
    coverage_path = coverage_root / "candidate_coverage.parquet"; coverage.to_parquet(coverage_path, index=False)
    common = {"lineage": LINEAGE, "candidate_population_lineage": POPULATION_LINEAGE, "evidence_scope": EVIDENCE_SCOPE, "product_lineage": PRODUCT_LINEAGE, "execution_parity_claim": False, "promotion_eligible": False}
    stage_manifest = {**common, "schema": "historical_backcast_exact1m_request_stage_v2", "economics_contract": ECONOMICS_CONTRACT, "return_unit": "quote_notional_price_return_not_inverse_collateral_roe", "path_horizon_minutes": 720, "signal_to_decision_hours": 1, "selected_rows": 2, "outputs": {"staged_candidates": {"path": str(stage_path), "sha256": _sha(stage_path)}}}
    stage_manifest_path = stage_root / "manifest.json"; _manifest(stage_manifest_path, stage_manifest)
    coverage_manifest = {**common, "schema": "historical_exact1m_candidate_coverage_v1", "status": "complete", "candidate_coverage_fraction": 1.0, "incomplete_candidates": 0, "complete_candidates": 2, "required_minutes_per_candidate": 720, "stage_manifest": {"sha256": _sha(stage_manifest_path)}, "outputs": {"candidate_coverage": {"path": str(coverage_path), "sha256": _sha(coverage_path)}}}
    coverage_manifest_path = coverage_root / "manifest.json"; _manifest(coverage_manifest_path, coverage_manifest)
    labels_manifest = {**common, "economics": ECONOMICS_LABELS, "oof_status": "not_oof", "rows": 2, "outputs": {"joined_multitask_labels": {"path": str(labels_path), "sha256": _sha(labels_path)}}, "sources": {"candidate_coverage_manifest": {"sha256": _sha(coverage_manifest_path)}}}
    _manifest(labels_root / "manifest.json", labels_manifest)
    return argparse.Namespace(stage_root=stage_root, labels_root=labels_root, coverage_root=coverage_root, output_dir=tmp_path / "out")


def test_materializes_exact_hash_bound_panel_and_excludes_outcomes_from_features(tmp_path: Path) -> None:
    outputs = run(_fixture(tmp_path))
    manifest = json.loads(outputs["manifest"].read_text())
    panel = pd.read_parquet(outputs["panel"])
    assert len(panel) == 2
    assert manifest["status"] == "research_panel_ready_not_oof_not_promotable"
    assert manifest["feature_families"] and len(manifest["feature_columns"]) == 69
    assert "execution_net_ev_12h" not in manifest["feature_columns"]
    assert panel["candidate_id"].is_unique
    assert panel["__label_signal_timestamp__"].equals(panel["signal_timestamp"])
    assert manifest["input_hashes"]["stage_manifest"]["sha256"]


def test_fails_closed_on_identity_mismatch(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    labels_path = args.labels_root / "joined_multitask_labels.parquet"
    labels = pd.read_parquet(labels_path); labels.loc[0, "__symbol__"] = "XRP/USD:XRP"; labels.to_parquet(labels_path, index=False)
    manifest_path = args.labels_root / "manifest.json"; manifest = json.loads(manifest_path.read_text()); manifest["outputs"]["joined_multitask_labels"]["sha256"] = _sha(labels_path); _manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="signal/symbol/side mismatch"):
        run(args)


def test_fails_closed_if_coverage_is_not_bound_to_stage(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    manifest_path = args.coverage_root / "manifest.json"; manifest = json.loads(manifest_path.read_text()); manifest["stage_manifest"]["sha256"] = "wrong"; _manifest(manifest_path, manifest)
    labels_manifest_path = args.labels_root / "manifest.json"; labels_manifest = json.loads(labels_manifest_path.read_text()); labels_manifest["sources"]["candidate_coverage_manifest"]["sha256"] = _sha(manifest_path); _manifest(labels_manifest_path, labels_manifest)
    with pytest.raises(ValueError, match="not bound to the supplied stage"):
        run(args)


def test_feature_contract_rejects_future_fields_and_wrong_family_count() -> None:
    features = _features()
    selected, families = causal_feature_contract([*features, "execution_net_ev_12h"])
    assert len(selected) == 69 and len(families["transition"]) == 25
    with pytest.raises(ValueError, match="expected 69 causal fields"):
        causal_feature_contract([key for key in features if key != "ret_1h"])
