from __future__ import annotations

import json
from pathlib import Path
import shutil

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
    StageITargetContract,
    canonical_sha256,
    file_sha256,
)
from extreme_price_movements.stage_iv_native_artifact_runner import NativeBasePrediction
from extreme_price_movements.stage_iv_native_materializer import (
    MATERIALIZER_SCHEMA,
    StageIVNativeMaterializationError,
    load_stage_iv_native_launch,
)
from scripts import run_stage_iv_native_sweep as cli


def _contract(family: str, layer: str, rows: int) -> StageITargetContract:
    return StageITargetContract(
        family=family, layer=layer, target_name=family, geometry="TP6_SL4_H12",
        identity_sha256="12" * 32, target_sha256="23" * 32,
        economics_sha256="34" * 32, validity_sha256="45" * 32,
        weight_sha256="56" * 32, rows=rows, target_columns=("target",),
        metadata=(
            {
                "meta_target_semantics": "same_side_direct_base_output_correctness_q33_v1",
                "base_input_semantics": "same_side_direct_base_output_without_bps_conversion_v1",
            }
            if layer == "meta" else {}
        ),
    )


def _bundle(rows: int) -> StageIAdapterWinnerBundle:
    base_features = ("signal",)
    meta_features = (
        "context", "base_raw_score", "base_state_p0", "base_state_p1",
        "base_output_entropy", "base_output_top2_margin",
        "base_output_max_probability",
    )
    cells = []
    for side in ("long", "short"):
        cells.append(StageIAdapterWinnerCell(
            side=side, base_features=base_features, meta_features=meta_features,
            base_params={"objective": "regression_l1", "n_estimators": 4},
            meta_params={"objective": "multiclass", "num_class": 3, "n_estimators": 4},
            base_target_contract=_contract(SOFT_SCALAR_S, "base", rows),
            meta_target_contract=_contract(FOLD_QUANTILE_RESIDUAL3, "meta", rows),
            base_selector_manifest_sha256="67" * 32,
            meta_selector_manifest_sha256="78" * 32,
            required_same_side_base_handoff_features=meta_features[1:],
        ))
    return StageIAdapterWinnerBundle(tuple(cells), code_revision="frozen-test-revision")


def _winner_hashes(bundle: StageIAdapterWinnerBundle, side: str) -> tuple[str, str]:
    cell = bundle.cell(side)
    return (
        canonical_sha256({"base": list(cell.base_features), "meta": list(cell.meta_features)}),
        canonical_sha256({
            "base": dict(cell.base_params), "meta": dict(cell.meta_params),
            "base_target_contract_sha256": cell.base_target_contract.sha256,
            "meta_target_contract_sha256": cell.meta_target_contract.sha256,
        }),
    )


def _write_launch(tmp_path: Path) -> Path:
    rows = 360
    bundle = _bundle(rows)
    winner_path = tmp_path / "winner.json"
    winner_path.write_text(json.dumps(bundle.to_dict(), sort_keys=True), encoding="utf-8")
    sources = {}
    for side, seed in (("long", 4), ("short", 8)):
        rng = np.random.default_rng(seed)
        decision = pd.date_range("2024-01-01", periods=rows, freq="3h", tz="UTC")
        signal = np.clip(rng.normal(0.5, 0.2, rows), 0.02, 0.98)
        ledger = pd.DataFrame({
            "candidate_id": [f"{side}-{index}" for index in range(rows)],
            "symbol": np.where(np.arange(rows) % 3, "BTC", "ETH"),
            "decision_ts": decision,
            "label_available_ts": decision + pd.Timedelta(hours=13),
            "base_target": signal,
            "exact_net_bps": 180.0 * signal - 90.0 + rng.normal(0.0, 20.0, rows),
            "signal": signal,
            "context": rng.normal(size=rows),
        })
        ledger_path = tmp_path / f"{side}.parquet"
        ledger.to_parquet(ledger_path, index=False)
        feature_sha, parameter_sha = _winner_hashes(bundle, side)
        sources[side] = {
            "path": ledger_path.name, "sha256": file_sha256(ledger_path),
            "winner_feature_contract_sha256": feature_sha,
            "winner_parameter_contract_sha256": parameter_sha,
            "columns": {
                "candidate_id": "candidate_id", "symbol": "symbol",
                "decision_ts": "decision_ts",
                "label_available_ts": "label_available_ts",
                "base_target": "base_target", "exact_net_bps": "exact_net_bps",
            },
        }
    cells = []
    settings = (
        ("control_x20", 0.20, "both", (24, 12, 10)),
        ("x30", 0.30, "tail", (28, 14, 11)),
        ("x40", 0.40, "meta", (32, 16, 12)),
        ("x50", 0.50, "neither", (24, 12, 10)),
    )
    for cell_id, fraction, route, burns in settings:
        cells.append({
            "cell_id": cell_id, "tail_fraction": fraction,
            "broad_output_route": route, "n_validation_folds": 2,
            "burn_ins": {
                "broad": burns[0], "tail": burns[1], "meta": burns[2],
                "handoff_history": 10,
            },
        })
    spec = {
        "schema": MATERIALIZER_SCHEMA,
        "winner_bundle": {
            "path": winner_path.name, "file_sha256": file_sha256(winner_path),
            "contract_sha256": bundle.sha256,
        },
        "side_ledgers": sources, "cells": cells,
        "runner": {
            "control_cell_id": "control_x20", "top_fractions": [0.10, 0.20],
            "selection_top_fraction": 0.10, "min_selected_rows": 2,
            "min_paired_months": 1,
            "admission_spec": {
                "min_reference_rows": 12, "min_side_reference_rows": 4,
                "bins": 4,
            },
        },
    }
    path = tmp_path / "cells.json"
    path.write_text(json.dumps(spec, sort_keys=True, indent=2), encoding="utf-8")
    return path


class _NativeModel:
    def predict_native(self, frame: pd.DataFrame) -> NativeBasePrediction:
        score = np.clip(frame.signal.to_numpy(float), 0.02, 0.98)
        return NativeBasePrediction(score, np.column_stack([1.0 - score, score]))


class _MetaModel:
    classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.base_raw_score.to_numpy(float)
        logits = np.column_stack([1.0 - raw, np.full(len(raw), 0.4), raw])
        logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return logits / logits.sum(axis=1, keepdims=True)


def _base_fitter(*_args, **_kwargs):
    return _NativeModel()


def _meta_fitter(*_args, **_kwargs):
    return _MetaModel()


def test_materializer_loads_only_explicit_hash_bound_cells(tmp_path: Path) -> None:
    path = _write_launch(tmp_path)
    launch = load_stage_iv_native_launch(path)
    assert [cell.cell_id for cell in launch.cells] == ["control_x20", "x30", "x40", "x50"]
    assert launch.launch_manifest["factorial_generation"] is False
    assert launch.cells[0].plans[0].meta_feature_names == ("context",)
    raw = json.loads(path.read_text())
    raw["winner_bundle"]["file_sha256"] = "00" * 32
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StageIVNativeMaterializationError, match="SHA-256 drift"):
        load_stage_iv_native_launch(path)


def test_cli_manifest_and_resume_reuse_verified_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_launch(tmp_path)
    monkeypatch.setattr(cli, "native_winner_base_fitter", _base_fitter)
    monkeypatch.setattr(cli, "direct_fq3_winner_meta_fitter", _meta_fitter)
    checkpoint = tmp_path / "checkpoints"
    first = tmp_path / "first"
    assert cli.main([
        "--cell-spec", str(path), "--output-dir", str(first),
        "--checkpoint-dir", str(checkpoint),
    ]) == 0
    manifest = json.loads((first / "run_manifest.json").read_text())
    assert manifest["launch_manifest"]["factorial_generation"] is False
    assert manifest["resume"]["executed_cell_count"] == 4
    assert manifest["resume"]["resumed_cell_count"] == 0

    partial_checkpoint = tmp_path / "partial_checkpoints"
    partial_checkpoint.mkdir()
    for completed in sorted(checkpoint.iterdir())[:2]:
        shutil.copytree(completed, partial_checkpoint / completed.name)
    partial = tmp_path / "partial_resume"
    assert cli.main([
        "--cell-spec", str(path), "--output-dir", str(partial),
        "--checkpoint-dir", str(partial_checkpoint), "--resume",
    ]) == 0
    partial_manifest = json.loads((partial / "run_manifest.json").read_text())
    assert partial_manifest["resume"]["resumed_cell_count"] == 2
    assert partial_manifest["resume"]["executed_cell_count"] == 2

    def forbidden(*_args, **_kwargs):
        raise AssertionError("verified checkpoints must prevent model refits")

    monkeypatch.setattr(cli, "native_winner_base_fitter", forbidden)
    monkeypatch.setattr(cli, "direct_fq3_winner_meta_fitter", forbidden)
    second = tmp_path / "second"
    assert cli.main([
        "--cell-spec", str(path), "--output-dir", str(second),
        "--checkpoint-dir", str(checkpoint), "--resume",
    ]) == 0
    resumed = json.loads((second / "run_manifest.json").read_text())
    assert resumed["resume"]["resumed_cell_count"] == 4
    assert resumed["resume"]["executed_cell_count"] == 0
    assert all(cell["resumed"] for cell in resumed["cells"])
