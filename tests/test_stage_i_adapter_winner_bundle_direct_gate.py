from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.stage_i_adapter_winner_bundle import build_stage_i_adapter_winner_bundle
from extreme_price_movements.stage_i_target_adapter import (
    FOLD_QUANTILE_RESIDUAL3,
    SOFT_SCALAR_S,
    bind_target_contract,
)
from extreme_price_movements.stage_i_target_specific_oos import (
    DIRECT_BASE_INPUT_SEMANTICS,
    DIRECT_FQ3_SEMANTICS,
)
from extreme_price_movements.stage_i_shared_population import canonical_sha256, file_sha256


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _selectors(tmp_path, *, direct: bool, include_map: bool = False):
    base_root, meta_root = tmp_path / "base", tmp_path / "meta"
    for side in ("long", "short"):
        frame = pd.DataFrame({
            "candidate_id": [f"{side}-1", f"{side}-2"],
            "__ts__": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "__symbol__": ["BTC", "ETH"], "side_name": side,
            "base_target": [0.2, 0.8], "meta_target": [-10.0, 10.0],
            "gross_bps": [90.0, 110.0], "net_bps": [-10.0, 10.0],
            "target_valid": True, "sample_weight": 1.0,
        })
        base_contract = bind_target_contract(
            frame, family=SOFT_SCALAR_S, layer="base", target_name="S", geometry="TP6_SL4_H12",
            target_columns=("base_target",),
        )
        metadata = ({
            "meta_target_semantics": DIRECT_FQ3_SEMANTICS,
            "base_input_semantics": DIRECT_BASE_INPUT_SEMANTICS,
        } if direct else {})
        meta_contract = bind_target_contract(
            frame, family=FOLD_QUANTILE_RESIDUAL3, layer="meta", target_name="FQ3", geometry="TP6_SL4_H12",
            target_columns=("meta_target",), metadata=metadata,
        )
        meta_features = ["context", "base_raw_score", "base_state_p0", "base_state_p1"]
        if include_map:
            meta_features.append("prequential_base_expected_net_bps")
        _write(base_root / side / "manifest.json", {
            "schema": "stage_i_base_feature_selection_v2", "status": "complete", "side": side,
            "selected_feature_contract": ["base_feature"], "best_params": {"objective": "regression_l1"},
            "target_contract": base_contract.to_dict(), "target_contract_sha256": base_contract.sha256,
        })
        _write(meta_root / side / "manifest.json", {
            "schema": "stage_i_adapter_meta_feature_selection_v2", "status": "complete", "side": side,
            "selected_feature_contract": meta_features,
            "required_same_side_base_oof_handoff_features": ["base_raw_score", "base_state_p0", "base_state_p1"],
            "best_params": {"objective": "multiclass", "num_class": 3},
            "target_contract": meta_contract.to_dict(), "target_contract_sha256": meta_contract.sha256,
            "base_target_contract": {"target_sha256": base_contract.target_sha256},
        })
    return base_root, meta_root


def test_bundle_builder_rejects_legacy_premapped_fq3_semantics(tmp_path) -> None:
    base, meta = _selectors(tmp_path, direct=False)
    with pytest.raises(ValueError, match="not direct FQ3"):
        build_stage_i_adapter_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, code_revision="test",
        )


def test_bundle_builder_rejects_map_feature_even_when_direct_metadata_is_forged(tmp_path) -> None:
    base, meta = _selectors(tmp_path, direct=True, include_map=True)
    with pytest.raises(ValueError, match="pre-mapped"):
        build_stage_i_adapter_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, code_revision="test",
        )


def test_bundle_builder_accepts_direct_fq3_contract(tmp_path) -> None:
    base, meta = _selectors(tmp_path, direct=True)
    bundle = build_stage_i_adapter_winner_bundle(
        base_selection_dir=base, meta_selection_dir=meta, code_revision="test",
    )
    assert {cell.side for cell in bundle.cells} == {"long", "short"}


def _write_signed_shared_population(root: Path) -> dict:
    rows = []
    for side in ("long", "short"):
        for index in range(2):
            rows.append({
                "candidate_id": f"{side}-{index}", "side_name": side,
                "__ts__": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=index),
                "decision_ts": pd.Timestamp("2024-01-01 01:00", tz="UTC") + pd.Timedelta(hours=index),
                "label_available_ts": pd.Timestamp("2024-01-01 13:00", tz="UTC") + pd.Timedelta(hours=index),
                "candidate_key": f"{side}::{side}-{index}",
            })
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True)
    population = root / "shared_population.parquet"
    frame.to_parquet(population, index=False)
    per_side = {}
    for side in ("long", "short"):
        part = frame.loc[frame.side_name.eq(side)].sort_values(
            ["side_name", "decision_ts", "candidate_key"], kind="stable",
        )
        per_side[side] = {
            "rows": len(part),
            "candidate_ids_sha256": canonical_sha256(sorted(part.candidate_id.tolist())),
            "identity_sha256": canonical_sha256(part.loc[:, [
                "candidate_id", "side_name", "__ts__", "decision_ts", "label_available_ts", "candidate_key",
            ]].astype(str).to_dict(orient="records")),
        }
    ordered = frame.sort_values(["side_name", "decision_ts", "candidate_key"], kind="stable")
    manifest = {
        "schema": "stage_i_joint_finalist_shared_population_v2", "status": "complete", "rows": len(frame),
        "identity_columns": ["candidate_id", "side_name", "__ts__", "decision_ts", "label_available_ts"],
        "population_sha256": canonical_sha256(ordered.loc[:, [
            "candidate_id", "side_name", "__ts__", "decision_ts", "label_available_ts", "candidate_key",
        ]].astype(str).to_dict(orient="records")),
        "source_lineage_sha256": {}, "counts": {}, "per_side": per_side,
        "finalist_families": ["R3_control", "scalar_S", "ordinal_O"],
        "selection": "per-side side-qualified intersection of target_valid candidate IDs across R3, scalar_S and ordinal_O before final OOS fitting/mapping",
        "files": {population.name: file_sha256(population)},
    }
    manifest["contract_sha256"] = canonical_sha256({key: value for key, value in manifest.items() if key != "contract_sha256"})
    _write(root / "manifest.json", manifest)
    return manifest


def test_publish_cli_binds_authorized_family_arm_and_signed_shared_universe(tmp_path, capsys) -> None:
    from scripts.publish_stage_i_adapter_winner_bundle import main

    base, meta = _selectors(tmp_path, direct=True)
    shared_root = tmp_path / "shared"
    shared = _write_signed_shared_population(shared_root)
    target_root = tmp_path / "target"
    _write(target_root / "manifest.json", {
        "schema": "stage_i_base_target_winner_bundle_v1", "status": "complete",
        "family": "scalar_S", "target_name": "S",
    })
    source = {
        "kind": "target_specific_winner_bundle_requires_new_base_mda",
        "bundle_manifest_sha256": file_sha256(target_root / "manifest.json"),
    }
    contract = {
        "schema": "stage_i_base_target_joint_finalists_v2", "status": "complete",
        "finalists": [{
            "family": "scalar_S", "arm": "S", "must_advance_to_joint_base_meta_evaluation": True,
            "source": source,
        }],
        "shared_population_contract_sha256": shared["contract_sha256"],
    }
    contract["contract_sha256"] = canonical_sha256({key: value for key, value in contract.items() if key != "contract_sha256"})
    contract_path = tmp_path / "target_finalist_contracts.json"
    _write(contract_path, contract)
    output = tmp_path / "bundle"
    assert main([
        "--target-finalists-contract", str(contract_path),
        "--shared-population-dir", str(shared_root),
        "--family", "scalar_S", "--arm", "S",
        "--base-selection-dir", str(base), "--meta-selection-dir", str(meta),
        "--target-winner-dir", str(target_root),
        "--code-revision", "test", "--run-id", "test-run", "--output-dir", str(output),
    ]) == 0
    payload = json.loads((output / "winner_bundle.json").read_text())
    assert payload["joint_finalist_authorization"]["family"] == "scalar_S"
    assert payload["joint_finalist_authorization"]["arm"] == "S"
    assert payload["joint_finalist_authorization"]["shared_population_contract_sha256"] == shared["contract_sha256"]
    assert json.loads(capsys.readouterr().out)["status"] == "created_immutable_bundle"
