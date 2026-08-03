from __future__ import annotations

import json
from pathlib import Path

import pytest

from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_ACTIVE_CONTRACTS,
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
)
from extreme_price_movements.stage_i_production_oos import StageIProductionWinnerBundle
from extreme_price_movements.stage_i_winner_bundle import (
    StageIWinnerBundleFreezeError,
    freeze_stage_i_winner_bundle,
)


RAW_FEATURES = ["base_signal", "base_context", "meta_regime", "meta_trust"]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base, meta, source = tmp_path / "base", tmp_path / "meta", tmp_path / "source"
    feature_hash = "f" * 64
    _write(source / "manifest.json", {
        "schema": "stage_i_production_input_contract_v1",
        "status": "complete",
        "rows": 1000,
        "min_signal_ts": "2022-08-31T01:00:00+00:00",
        "max_signal_ts": "2026-07-10T21:00:00+00:00",
        "feature_contract_sha256": feature_hash,
    })
    _write(source / "frozen_feature_contract.json", {
        "schema": "packb_static_point_causal_feature_contract_v1",
        "feature_contract_sha256": feature_hash,
        "feature_columns": RAW_FEATURES,
    })
    for side in ("long", "short"):
        base_features = ["base_signal", "base_context"]
        _write(base / side / "manifest.json", {
            "schema": "stage_i_base_feature_selection_v1",
            "status": "complete", "side": side, "rows": 500,
            "selected_features": base_features,
            "selected_feature_contract": base_features,
            "selected_feature_count": len(base_features),
            # Objective fields were fixed by the selector mode, not HPO.
            "best_params": {"n_estimators": 50, "learning_rate": 0.03},
        })
        meta_features = ["meta_regime", "meta_trust", *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES]
        _write(meta / side / "manifest.json", {
            "schema": "stage_i_meta_feature_selection_v1",
            "status": "complete", "side": side, "rows": 450,
            "selected_features": meta_features,
            "selected_feature_contract": meta_features,
            "selected_feature_count": len(meta_features),
            "required_same_side_base_oof_handoff_features": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
            "best_params": {"n_estimators": 35, "objective": "huber"},
        })
    return base, meta, source


def _freeze(tmp_path: Path):
    base, meta, source = _fixture(tmp_path)
    output = tmp_path / "winner.json"
    bundle, status = freeze_stage_i_winner_bundle(
        base_selection_dir=base,
        meta_selection_dir=meta,
        input_contract_dir=source,
        output_path=output,
        code_revision="a" * 40,
    )
    return bundle, status, output, base, meta, source


def test_freezes_exact_four_cells_runtime_semantics_and_calendar(tmp_path: Path) -> None:
    bundle, status, output, *_ = _freeze(tmp_path)
    assert status == "created_immutable_bundle"
    assert [cell.contract for cell in bundle.cells] == list(STAGE_I_ACTIVE_CONTRACTS)
    assert all(cell.lgbm_params["objective"] == "multiclass" for cell in bundle.cells if cell.contract.layer == "base")
    assert all(cell.lgbm_params["num_class"] == 3 for cell in bundle.cells if cell.contract.layer == "base")
    assert all(cell.lgbm_params["objective"] == "huber" for cell in bundle.cells if cell.contract.layer == "meta")
    assert bundle.calendar.evaluation_start_utc == "2024-01-01T00:00:00Z"
    assert bundle.calendar.evaluation_end_utc == "2026-07-10T21:00:00+00:00"
    assert bundle.feature_selection_exception.approved
    assert "reused backward" in bundle.feature_selection_exception.rationale
    loaded = StageIProductionWinnerBundle.from_dict(json.loads(output.read_text()))
    assert loaded.sha256 == bundle.sha256
    source_hashes = {cell.source_manifest_sha256 for cell in bundle.cells}
    assert len(source_hashes) == 1
    assert all(len(cell.selector_manifest_sha256) == 64 for cell in bundle.cells)


def test_exact_existing_bundle_is_reused_but_conflict_is_never_overwritten(tmp_path: Path) -> None:
    bundle, _, output, base, meta, source = _freeze(tmp_path)
    before = output.read_bytes()
    reused, status = freeze_stage_i_winner_bundle(
        base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
        output_path=output, code_revision="a" * 40,
    )
    assert status == "reused_verified_immutable_bundle"
    assert reused.sha256 == bundle.sha256
    with pytest.raises(FileExistsError, match="conflicting immutable"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=output, code_revision="b" * 40,
        )
    assert output.read_bytes() == before


def test_partial_cell_and_feature_or_parameter_disagreement_fail_closed(tmp_path: Path) -> None:
    base, meta, source = _fixture(tmp_path)
    (meta / "short" / "manifest.json").unlink()
    with pytest.raises(StageIWinnerBundleFreezeError, match="missing"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=tmp_path / "partial.json", code_revision="a" * 40,
        )

    base, meta, source = _fixture(tmp_path / "disagree")
    path = base / "long" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["selected_features"] = ["base_signal"]
    _write(path, manifest)
    with pytest.raises(StageIWinnerBundleFreezeError, match="fields disagree"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=tmp_path / "feature-disagreement.json", code_revision="a" * 40,
        )

    base, meta, source = _fixture(tmp_path / "params")
    path = meta / "long" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["lgbm_params"] = {"n_estimators": 99, "objective": "huber"}
    _write(path, manifest)
    with pytest.raises(StageIWinnerBundleFreezeError, match="parameter fields disagree"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=tmp_path / "param-disagreement.json", code_revision="a" * 40,
        )


def test_changed_source_hash_cannot_replace_existing_bundle(tmp_path: Path) -> None:
    _, _, output, base, meta, source = _freeze(tmp_path)
    before = output.read_bytes()
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rows"] += 1
    _write(path, manifest)
    with pytest.raises(FileExistsError, match="conflicting immutable"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=output, code_revision="a" * 40,
        )
    assert output.read_bytes() == before


def test_rejects_unavailable_raw_feature_and_wrong_runtime_objective(tmp_path: Path) -> None:
    base, meta, source = _fixture(tmp_path)
    path = base / "short" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["selected_features"] = manifest["selected_feature_contract"] = ["base_signal", "god_feature"]
    manifest["selected_feature_count"] = 2
    _write(path, manifest)
    with pytest.raises(StageIWinnerBundleFreezeError, match="absent from its frozen production source"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=tmp_path / "missing-feature.json", code_revision="a" * 40,
        )

    base, meta, source = _fixture(tmp_path / "objective")
    path = base / "short" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["best_params"]["objective"] = "binary"
    _write(path, manifest)
    with pytest.raises(StageIWinnerBundleFreezeError, match="objective disagrees"):
        freeze_stage_i_winner_bundle(
            base_selection_dir=base, meta_selection_dir=meta, input_contract_dir=source,
            output_path=tmp_path / "wrong-objective.json", code_revision="a" * 40,
        )
