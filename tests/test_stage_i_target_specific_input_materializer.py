from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_adapter_winner_bundle import (
    StageIAdapterWinnerBundle,
    StageIAdapterWinnerCell,
)
from extreme_price_movements.stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    SOFT_SCALAR_S,
    bind_target_contract,
    canonical_sha256,
    file_sha256,
)
from extreme_price_movements.stage_i_target_specific_input_materializer import (
    TargetSpecificInputMaterializationError,
    TargetSpecificInputMaterializationSpec,
    _r3_contract_source,
    materialize_stage_i_target_specific_inputs,
)
from extreme_price_movements.stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path, *, reserved_feature: bool = False) -> TargetSpecificInputMaterializationSpec:
    selector = tmp_path / "selector"
    base_root, meta_root = tmp_path / "base", tmp_path / "meta"
    target_root, output = tmp_path / "target", tmp_path / "output"
    selector.mkdir(parents=True)
    all_features, all_ledger, all_handoff, cells = [], [], [], []
    base_manifests: dict[str, dict] = {}
    meta_manifests: dict[str, dict] = {}
    for side_index, side in enumerate(("long", "short")):
        n = 96
        ts = pd.date_range("2026-03-01", periods=n, freq="12h", tz="UTC")
        ids = [f"{side}-{i}" for i in range(n)]
        signal = np.sin(np.arange(n) / 9.0 + side_index)
        feature_name = "future_return" if reserved_feature and side == "long" else "base_feature"
        features = pd.DataFrame({
            "candidate_id": ids, "__ts__": ts, "__symbol__": np.resize(["BTC", "ETH"], n),
            feature_name: signal, "context": np.cos(np.arange(n) / 7.0),
            "regime": np.resize([0.0, 1.0, 2.0], n),
        })
        if feature_name != "base_feature":
            features["base_feature"] = signal
        net = signal * 200.0
        handoff = pd.DataFrame({
            "candidate_id": ids, "__ts__": ts, "__symbol__": features.__symbol__, "side_name": side,
            "decision_ts": ts + pd.Timedelta(hours=1),
            "label_available_ts": ts + pd.Timedelta(hours=13),
            "target_valid": True, "gross_bps": net + 100.0, "net_bps": net,
            "target_value": np.clip((signal + 1.0) / 2.0, 0.0, 1.0),
            "sample_weight_base_component": 1.0,
            "contract_certainty": np.clip(0.5 + 0.5 * np.abs(signal), 0.0, 1.0),
        })
        ledger = handoff.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts", "label_available_ts"]].copy()
        ledger["label_valid"] = True
        # The selector target contracts deliberately bind fewer training rows
        # than the evaluation handoff.  This proves OOS content is rebound,
        # not falsely compared to training hashes.
        training = handoff.iloc[:60].copy()
        training["direct_fq3_exact_net_basis"] = training.net_bps
        training["sample_weight"] = 1.0
        base_contract = bind_target_contract(
            training, family=SOFT_SCALAR_S, layer="base", target_name="S", geometry="TP7_SL2_H12",
            target_columns=("target_value",), weight_column="sample_weight_base_component",
            metadata={"training_weight_contract": {"mode": "contract_certainty"}},
        )
        meta_contract = bind_target_contract(
            training, family=FOLD_QUANTILE_RESIDUAL3, layer="meta", target_name="FQ3", geometry="TP7_SL2_H12",
            target_columns=("direct_fq3_exact_net_basis",), weight_column="sample_weight",
            metadata={
                "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
                "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
                "required_regime_features": ["regime"], "required_context_features": ["context"],
                "required_trust_features": ["base_output_entropy"],
            },
        )
        base_selected = (feature_name,)
        meta_selected = (
            "context", "regime", "base_raw_score", "base_state_p0", "base_state_p1",
            "base_output_entropy", "base_output_top2_margin", "base_output_max_probability",
        )
        base_manifest = {
            "schema": "stage_i_base_feature_selection_v2", "status": "complete", "side": side,
            "target_contract": base_contract.to_dict(), "target_contract_sha256": base_contract.sha256,
            "selected_feature_contract": list(base_selected), "input_feature_contract": list(base_selected),
            "feature_universe_lineage": {"schema": "stage_i_feature_universe_lineage_v1", "layer": "base", "side": side},
            "correlation_policy": "grouped-preserve", "best_params": {"objective": "regression_l1"},
        }
        meta_manifest = {
            "schema": "stage_i_adapter_meta_feature_selection_v2", "status": "complete", "side": side,
            "target_contract": meta_contract.to_dict(), "target_contract_sha256": meta_contract.sha256,
            "base_target_contract": base_contract.to_dict(),
            "selected_feature_contract": list(meta_selected), "input_feature_contract": ["context", "regime"],
            "feature_universe_lineage": {"schema": "stage_i_feature_universe_lineage_v1", "layer": "meta", "side": side},
            "required_same_side_base_oof_handoff_features": ["base_raw_score", "base_state_p0", "base_state_p1"],
            "correlation_policy": "grouped-preserve", "best_params": {"objective": "multiclass", "num_class": 3},
        }
        base_path, meta_path = base_root / side / "manifest.json", meta_root / side / "manifest.json"
        _write_json(base_path, base_manifest); _write_json(meta_path, meta_manifest)
        base_manifests[side], meta_manifests[side] = base_manifest, meta_manifest
        cells.append(StageIAdapterWinnerCell(
            side=side, base_features=base_selected, meta_features=meta_selected,
            base_params=base_manifest["best_params"], meta_params=meta_manifest["best_params"],
            base_target_contract=base_contract, meta_target_contract=meta_contract,
            base_selector_manifest_sha256=file_sha256(base_path),
            meta_selector_manifest_sha256=file_sha256(meta_path),
            required_same_side_base_handoff_features=("base_raw_score", "base_state_p0", "base_state_p1"),
        ))
        all_features.append(features); all_ledger.append(ledger); all_handoff.append(handoff)
    feature_frame = pd.concat(all_features, ignore_index=True)
    ledger_frame = pd.concat(all_ledger, ignore_index=True)
    feature_path, ledger_path = selector / "selector_features.parquet", selector / "selector_ledger.parquet"
    feature_frame.to_parquet(feature_path, index=False); ledger_frame.to_parquet(ledger_path, index=False)
    feature_contract = {"schema": "packb_static_point_feature_loader_v1", "feature_contract_sha256": "f" * 64}
    _write_json(selector / "selector_feature_contract.json", feature_contract)
    selector_manifest = {
        "schema": "stage_i_selector_sample_v1", "status": "complete", "rows": len(feature_frame),
        "feature_contract_sha256": feature_contract["feature_contract_sha256"],
        "artifact_integrity": {
            "schema": "stage_i_selector_artifact_integrity_v1",
            "selector_features_sha256": file_sha256(feature_path),
            "selector_ledger_sha256": file_sha256(ledger_path),
        },
    }
    _write_json(selector / "manifest.json", selector_manifest)
    target_root.mkdir()
    handoff_path = target_root / "winner_target_handoff.parquet"
    pd.concat(all_handoff, ignore_index=True).to_parquet(handoff_path, index=False)
    _write_json(target_root / "manifest.json", {
        "schema": "stage_i_base_target_winner_bundle_v1", "status": "complete",
        "artifact_sha256": {handoff_path.name: file_sha256(handoff_path)},
    })
    bundle = StageIAdapterWinnerBundle(cells=tuple(cells), code_revision="test-revision")
    bundle_path = tmp_path / "winner_bundle.json"
    _write_json(bundle_path, bundle.to_dict())
    return TargetSpecificInputMaterializationSpec(
        selector_dir=selector, base_selector_dir=base_root, meta_selector_dir=meta_root,
        winner_bundle_path=bundle_path, target_winner_dir=target_root, output_dir=output,
        n_validation_folds=3, min_train_rows=12,
    )


def test_materializer_rebinds_evaluation_rows_and_runner_preflight_accepts(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    result = materialize_stage_i_target_specific_inputs(spec)
    assert result["status"] == "complete" and result["model_fit_performed"] is False
    for side in ("long", "short"):
        manifest = json.loads((spec.output_dir / side / "manifest.json").read_text())
        assert manifest["rows"] == 96
        assert manifest["source_feature_contract"]["generated_fields_present"] is False
        assert manifest["evaluation_target_contracts"]["base"]["rows"] == 96
        assert manifest["training_target_semantics"]["base"]["geometry"] == "TP7_SL2_H12"
        coverage = manifest["evaluation_month_contract"]["source_availability"]
        assert len(coverage) == 36 and coverage["2026-03"]["source_available"]
        assert not coverage["2024-01"]["source_available"] and coverage["2024-01"]["source_gap_reason"]
        assert not coverage["2026-12"]["source_available"] and coverage["2026-12"]["source_gap_reason"]
        contract = pd.read_parquet(spec.output_dir / side / "contract.parquet")
        assert "contract_certainty" in contract
        assert contract.contract_certainty.between(0.0, 1.0).all()
    completed = subprocess.run([
        sys.executable, "scripts/run_stage_i_target_specific_oos.py",
        "--winner-bundle", str(spec.winner_bundle_path), "--input-root", str(spec.output_dir),
        "--base-selector-dir", str(spec.base_selector_dir), "--meta-selector-dir", str(spec.meta_selector_dir),
        "--preflight",
    ], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    payload_start = completed.stdout.rfind('{\n  "status"')
    assert payload_start >= 0, completed.stdout
    assert json.loads(completed.stdout[payload_start:])["status"] == "preflight_complete_no_fit"


def test_materializer_rejects_reserved_future_feature(tmp_path: Path) -> None:
    spec = _fixture(tmp_path, reserved_feature=True)
    with pytest.raises(TargetSpecificInputMaterializationError, match="reserved namespace"):
        materialize_stage_i_target_specific_inputs(spec)


def test_materializer_is_immutable_and_detects_source_hash_drift(tmp_path: Path) -> None:
    spec = _fixture(tmp_path)
    materialize_stage_i_target_specific_inputs(spec)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_stage_i_target_specific_inputs(spec)
    drift_spec = _fixture(tmp_path / "drift")
    manifest_path = drift_spec.selector_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_integrity"]["selector_features_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(TargetSpecificInputMaterializationError, match="hash drift"):
        materialize_stage_i_target_specific_inputs(drift_spec)


def test_frozen_r3_contract_source_materializes_direct_fq3_net_basis(tmp_path: Path) -> None:
    side, n = "long", 12
    ts = pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")
    identity = pd.DataFrame({
        "candidate_id": [f"r3-{i}" for i in range(n)], "__ts__": ts,
        "__symbol__": np.resize(["BTC", "ETH"], n), "side_name": side,
        "decision_ts": ts + pd.Timedelta(hours=1),
        "label_available_ts": ts + pd.Timedelta(hours=13),
    })
    net = np.linspace(-250.0, 250.0, n)
    oof = identity.copy()
    oof["exact_gross_bps"], oof["exact_net_bps"] = net + 100.0, net
    oof_path = tmp_path / "base" / side / "selector_base_oof.parquet"
    oof_path.parent.mkdir(parents=True)
    oof.to_parquet(oof_path, index=False)
    _write_json(oof_path.parent / "manifest.json", {
        "selector_base_oof_sha256": file_sha256(oof_path),
    })
    ledger = identity.copy()
    ledger["r3_class"] = np.resize([0, 1, 2], n)
    training = identity.copy()
    training["legacy_runtime_vector"] = ledger.r3_class
    training["direct_fq3_exact_net_basis"] = net
    training["gross_bps"], training["net_bps"] = net + 100.0, net
    training["target_valid"], training["sample_weight"] = True, 1.0
    base_contract = bind_target_contract(
        training, family="legacy_R3_multiclass3_control", layer="base",
        target_name="R3", geometry="TP6_SL4_H12", target_columns=("legacy_runtime_vector",),
    )
    meta_contract = bind_target_contract(
        training, family=FOLD_QUANTILE_RESIDUAL3, layer="meta",
        target_name="FQ3", geometry="TP6_SL4_H12", target_columns=("direct_fq3_exact_net_basis",),
        metadata={
            "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
            "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
        },
    )
    contract, frozen_path, hashes = _r3_contract_source(
        side=side, base_root=tmp_path / "base", ledger=ledger,
        base_contract=base_contract, meta_contract=meta_contract,
    )
    assert frozen_path == oof_path.resolve()
    assert hashes["frozen_r3_oof"] == file_sha256(oof_path)
    assert np.allclose(contract.direct_fq3_exact_net_basis, contract.net_bps)
    assert np.array_equal(contract.legacy_runtime_vector, ledger.r3_class)
