#!/usr/bin/env python3
"""Materialize the frozen V9 + market-state MLP policy without an 8-day gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from extreme_price_movements.meta_postprocessor_pipeline import (
    ARTIFACT_NAME,
    POLICY_ID,
    validate_meta_postprocessor_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data_perp/artifacts/s59_s52_frozen_inference_bundle_20260709"
DEFAULT_POSTPROCESSOR = REPO_ROOT / (
    "data_perp/reports/"
    "meta_market_state_encoder_ablation_mlp_direct_hierev_hpo_20260713_v14"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "data_perp/artifacts/"
    "s59_s52_frozen_inference_bundle_v9_tail95_mlp_hierev_20260713"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patch_policy_block(payload: dict[str, Any], calibration_path: str) -> None:
    payload.update(
        {
            "policy_name": "s52_v9_tail95_market_state_mlp_hier_ev_top10_v1",
            "regime_ev_calibration_enabled": True,
            "regime_ev_calibration_artifact_path": calibration_path,
            "regime_ev_calibration_policy_id": POLICY_ID,
            "regime_ev_calibration_rank_source": POLICY_ID,
            "threshold_basis_policy_enabled": False,
            "threshold_basis_policy_id": "",
            "threshold_basis_policy_path": "",
            "threshold_basis_family": "fixed_historical_expected_ev_rank_top10",
            "threshold_basis_window_days": 0,
            "source_threshold_basis_policy": "",
        }
    )
    selection = payload.get("selection")
    if isinstance(selection, dict):
        _patch_policy_block(selection, calibration_path)
        selection["initial_rank_threshold"] = 0.90
        selection["initial_rank_threshold_floor"] = 0.90


def promote(source: Path, postprocessor: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")
    validate_meta_postprocessor_bundle(postprocessor)
    shutil.copytree(source, output)
    policy_dir = output / "policy_params"
    calibration_path = policy_dir / ARTIFACT_NAME
    shutil.copy2(postprocessor / ARTIFACT_NAME, calibration_path)
    calibration = _load(calibration_path)
    calibration["predecessor_policy_id"] = (
        "meta_residual_extreme_local_champion_overlay_ooftrain_"
        "tieaware_downonly_20260712_v9::forced_local_tail_0.950"
    )
    calibration["postprocessor_archetype_contract"] = (
        "side_name||archetype_policy_key"
    )
    _write(calibration_path, calibration)
    policy_models_dir = policy_dir / "policy_models"
    shutil.rmtree(policy_models_dir, ignore_errors=True)
    shutil.copytree(postprocessor / "policy_models", policy_models_dir)
    handoff = postprocessor / "composite_policy_feature_handoff.parquet"
    if handoff.exists():
        shutil.copy2(handoff, policy_dir / handoff.name)

    relative_calibration = str(calibration_path.relative_to(REPO_ROOT))
    portfolio_path = policy_dir / "optimized_portfolio_policy_config.json"
    portfolio = _load(portfolio_path)
    _patch_policy_block(portfolio, relative_calibration)
    _write(portfolio_path, portfolio)

    hit_path = policy_dir / "hit_surprise_archetype_portfolio_policy.json"
    if hit_path.exists():
        hit = _load(hit_path)
        _patch_policy_block(hit, relative_calibration)
        hit["archetype_hit_surprise_enabled"] = False
        hit["archetype_hit_surprise_mode"] = "disabled_hr_off"
        _write(hit_path, hit)

    manifest_path = policy_dir / "promoted_policy_manifest.json"
    manifest = _load(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "policy_id": POLICY_ID,
            "policy_name": portfolio["policy_name"],
            "regime_ev_calibration_policy_id": POLICY_ID,
            "regime_ev_calibration_artifact_path": relative_calibration,
            "threshold_basis_policy_enabled": False,
            "threshold_basis_policy_id": "",
            "threshold_basis_policy_path": "",
            "predecessor_policy_id": (
                "meta_residual_extreme_local_champion_overlay_ooftrain_"
                "tieaware_downonly_20260712_v9::forced_local_tail_0.950"
            ),
            "admission_contract": "expected_ev_rank_score >= 0.90",
            "rolling_8d_modulator_enabled": False,
        }
    )
    manifest["file_sha256"] = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(policy_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    _write(manifest_path, manifest)

    pointer = {
        "schema": "meta_postprocessor_pointer_v1",
        "policy_id": POLICY_ID,
        "artifact_path": relative_calibration,
        "source_postprocessor": str(postprocessor.relative_to(REPO_ROOT)),
        "rolling_8d_modulator_enabled": False,
    }
    _write(output / "meta_postprocessor_pointer.json", pointer)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--postprocessor", type=Path, default=DEFAULT_POSTPROCESSOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = promote(args.source.resolve(), args.postprocessor.resolve(), args.output.resolve())
    print(output)


if __name__ == "__main__":
    main()
