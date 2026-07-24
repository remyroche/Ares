#!/usr/bin/env python3
"""Promote a validated side/archetype exit geometry into a live policy bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_NAME = "s52_v9_tail95_mlp_hierev_ev70_trim10_21d_evaware_geometry_v3"
ADMISSION_POLICY_ID = "side_archetype_hier_ev_fixed70_trim10_21d_v1"
ADMISSION_FAMILY = "side_archetype_expected_ev_recent_correction"
POSTPROCESSOR_POLICY_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
ADMISSION_STATUS = "promoted_default_threshold_basis_ev70_trim10_21d"
ADMISSION_LAYER = (
    "side_archetype_hierarchical_ev + "
    "causal_21d_trim10_recent_ev_correction + fixed_0.70pct_net_ev_admission"
)


def _patch_active_policy_mapping(
    value: Any,
    *,
    admission_path: str,
    regime_path: str,
    portfolio_path: str,
) -> None:
    """Canonicalize active policy pointers recursively in deployment payloads."""
    if isinstance(value, list):
        for item in value:
            _patch_active_policy_mapping(
                item,
                admission_path=admission_path,
                regime_path=regime_path,
                portfolio_path=portfolio_path,
            )
        return
    if not isinstance(value, dict):
        return
    if any(str(key).startswith("threshold_basis_") for key in value):
        value.update(
            {
                "threshold_basis_policy_enabled": True,
                "threshold_basis_policy_id": ADMISSION_POLICY_ID,
                "threshold_basis_family": ADMISSION_FAMILY,
                "threshold_basis_window_days": 21,
                "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
                "threshold_basis_fixed_target_net_ev": 0.007,
                "threshold_basis_robust_daily_residual_trim_fraction": 0.10,
                "threshold_basis_hr_rank50": False,
                "threshold_basis_policy_path": admission_path,
                "source_threshold_basis_policy": admission_path,
            }
        )
    if any(str(key).startswith("regime_ev_calibration_") for key in value):
        value.update(
            {
                "regime_ev_calibration_enabled": True,
                "regime_ev_calibration_policy_id": POSTPROCESSOR_POLICY_ID,
                "regime_ev_calibration_rank_source": POSTPROCESSOR_POLICY_ID,
                "regime_ev_calibration_artifact_path": regime_path,
            }
        )
    if "threshold_rank_score_source" in value:
        value["threshold_rank_score_source"] = (
            f"threshold_basis:{ADMISSION_POLICY_ID}"
        )
    if "threshold_rank_score_source_reason" in value:
        value["threshold_rank_score_source_reason"] = ADMISSION_POLICY_ID
    if "source_portfolio_policy" in value:
        value["source_portfolio_policy"] = portfolio_path
    if "archetype_hit_surprise_enabled" in value:
        value["archetype_hit_surprise_enabled"] = False
        value["archetype_hit_surprise_mode"] = "disabled_hr_off"
        if "archetype_hit_surprise_policy_path" in value:
            value["archetype_hit_surprise_policy_path"] = ""
    if "source_hit_surprise_policy" in value:
        value["source_hit_surprise_policy"] = ""
    for child in value.values():
        _patch_active_policy_mapping(
            child,
            admission_path=admission_path,
            regime_path=regime_path,
            portfolio_path=portfolio_path,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _geometry(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("shrinkage_final_geometry is not a dictionary")
    return dict(parsed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_params(row: pd.Series) -> dict[str, Any]:
    return {
        str(column)[len("param_") :]: value
        for column, value in row.items()
        if str(column).startswith("param_") and not pd.isna(value)
    }


def _template_by_side(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in payload.get("strategies", []):
        if isinstance(row, dict) and str(row.get("side")) in {"long", "short"}:
            output[str(row["side"])] = row
    missing = {"long", "short"} - set(output)
    if missing:
        raise ValueError(f"source deployment is missing side templates: {sorted(missing)}")
    return output


def _deployment_row(
    template: dict[str, Any],
    *,
    strategy_id: str,
    params: dict[str, Any],
    source: str,
    archetype: str = "",
) -> dict[str, Any]:
    row = deepcopy(template)
    row.update(_json_safe(params))
    row.update(
        {
            "strategy_id": strategy_id,
            "strategy_for_inference": strategy_id,
            "canonical_strategy_id": strategy_id,
            "selected": True,
            "generated_by": "simple_policy_optimiser",
            "schema": "simple_policy_v1",
            "policy_name": POLICY_NAME,
            "params_source": source,
            "round_trip_cost_pct": 0.01,
            "cost_pct_per_side": 0.005,
            "capital_protect_spread_lock_mult": 1.5,
            "exit_geometry_scope": "side_archetype" if archetype else "side_parent",
            "policy_archetype": archetype,
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployment_dir = args.artifact_root / "simple_policy_optimiser" / "deployment"
    canonical_path = deployment_dir / "best_policy_params.json"
    payload = json.loads(canonical_path.read_text())
    portfolio_config_path = (
        args.artifact_root / "policy_params" / "optimized_portfolio_policy_config.json"
    )
    portfolio_config = json.loads(portfolio_config_path.read_text())
    admission_path = (
        args.artifact_root
        / "policy_params"
        / "threshold_basis_policy_sidearch_ev70_trim10_21d.json"
    )
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if str(admission.get("policy_id") or "") != ADMISSION_POLICY_ID:
        raise ValueError(
            "admission policy ID mismatch: "
            f"{admission.get('policy_id')!r} != {ADMISSION_POLICY_ID!r}"
        )
    canonical_admission_path = (
        args.artifact_root / "policy_params" / "threshold_basis_policy.json"
    )
    canonical_admission_path.write_text(
        json.dumps(_json_safe(admission), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    admission_summary = {
        "policy_id": str(admission.get("policy_id") or ""),
        "family": str(admission.get("family") or ""),
        "window_days": int(admission.get("window_days") or 0),
        "selection_mode": str(admission.get("selection_mode") or ""),
        "fixed_target_net_ev": float(admission.get("fixed_target_net_ev") or 0.0),
        "robust_daily_residual_trim_fraction": float(
            admission.get("robust_daily_residual_trim_fraction") or 0.0
        ),
        "reference_candidates_path": str(
            admission.get("reference_candidates_path") or ""
        ),
    }
    regime_path = (
        args.artifact_root
        / "policy_params"
        / "composite_policy_regime_ev_calibration.json"
    )
    templates = _template_by_side(payload)
    parent = pd.read_csv(args.geometry_dir / "best_side_parent_policy_summary.csv")
    local = pd.read_csv(args.geometry_dir / "best_side_archetype_policy_summary.csv")
    parent_params = {
        str(row["side"]): _parent_params(row) for _, row in parent.iterrows()
    }
    source = canonical_path.relative_to(args.artifact_root.parent.parent).as_posix()
    strategies: list[dict[str, Any]] = []
    for side in ("long", "short"):
        parent_row = _deployment_row(
            templates[side],
            strategy_id=f"{side}__parent",
            params=parent_params[side],
            source=source,
        )
        strategies.append(parent_row)
        # The model-coverage contract is keyed by the exact side-specific model
        # route. Keep that route as an explicit alias of the side-parent policy;
        # archetype routing may replace it later, but inference must never rely
        # on an implicit policy fallback before the archetype is known.
        strategies.append(
            _deployment_row(
                templates[side],
                strategy_id=f"{side}_s52_meta_threshold_handoff",
                params=parent_params[side],
                source=source,
            )
        )
    for _, row in local.iterrows():
        side = str(row["side"])
        archetype = str(row["policy_archetype"])
        params = dict(parent_params[side])
        params.update(_geometry(row["shrinkage_final_geometry"]))
        strategies.append(
            _deployment_row(
                templates[side],
                strategy_id=f"{side}__{archetype}",
                params=params,
                source=source,
                archetype=archetype,
            )
        )

    timestamp = datetime.now(timezone.utc)
    payload["strategies"] = strategies
    payload["policy_name"] = POLICY_NAME
    payload["policy_contract"] = (
        "V9 tail95 predecessor + market-state MLP/hierarchical side-archetype EV "
        "mapping + causal 21-day trim10 side-archetype EV70 admission + "
        "side/archetype exit geometry + global auction; 8-day HR modulation disabled"
    )
    payload.update(
        {
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": ADMISSION_POLICY_ID,
            "threshold_basis_family": ADMISSION_FAMILY,
            "threshold_basis_window_days": 21,
            "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
            "threshold_basis_fixed_target_net_ev": 0.007,
            "threshold_basis_robust_daily_residual_trim_fraction": 0.10,
            "threshold_basis_hr_rank50": False,
            "threshold_basis_policy_path": admission_path.as_posix(),
            "source_threshold_basis_policy": admission_path.as_posix(),
            "threshold_basis_policy_summary": admission_summary,
            "regime_ev_calibration_enabled": True,
            "regime_ev_calibration_policy_id": POSTPROCESSOR_POLICY_ID,
        }
    )
    for row in strategies:
        row.update(
            {
                "threshold_basis_policy_enabled": True,
                "threshold_basis_policy_id": ADMISSION_POLICY_ID,
                "threshold_basis_family": ADMISSION_FAMILY,
                "threshold_basis_window_days": 21,
                "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
                "threshold_basis_fixed_target_net_ev": 0.007,
                "threshold_basis_robust_daily_residual_trim_fraction": 0.10,
                "threshold_basis_hr_rank50": False,
                "threshold_basis_policy_path": admission_path.as_posix(),
                "regime_ev_calibration_enabled": True,
                "regime_ev_calibration_policy_id": POSTPROCESSOR_POLICY_ID,
            }
        )
    payload["created_at_utc"] = timestamp.isoformat()
    payload["created_at_ns"] = int(timestamp.timestamp() * 1e9)
    payload["exit_geometry_contract"] = {
        "policy_name": POLICY_NAME,
        "scope": "side_x_policy_archetype_with_side_parent_fallback",
        "capital_protection": "max(policy_floor, 1.5 * asset_full_spread)",
        "timeout": "last_executable_close_at_horizon",
        "round_trip_cost_pct": 0.01,
        "validation_manifest": args.evidence_manifest.as_posix(),
    }
    payload.update(
        {
            "run_id": args.artifact_root.name,
            "regime_ev_predecessor_bundle_path": (
                args.artifact_root / "policy_params" / "v9_tail95_predecessor_bundle.joblib"
            ).resolve().as_posix(),
            "regime_ev_residual_event_state_path": (
                args.artifact_root / "policy_params" / "residual_event_state.joblib"
            ).resolve().as_posix(),
            "regime_ev_calibration_artifact_path": regime_path.resolve().as_posix(),
            "side_residual_expert_artifact_path": (
                args.artifact_root / "policy_params" / "side_residual_expert.joblib"
            ).resolve().as_posix(),
            "threshold_rank_score_source": f"threshold_basis:{ADMISSION_POLICY_ID}",
            "threshold_rank_score_source_reason": ADMISSION_POLICY_ID,
            "rank_policy": {
                "mode": "side_archetype_hier_ev_fixed70_trim10_21d",
                "base_rank_threshold": 0.90,
                "hr_rank50": False,
                "threshold_basis_policy_id": ADMISSION_POLICY_ID,
                "threshold_basis_family": ADMISSION_FAMILY,
                "threshold_basis_window_days": 21,
                "fixed_target_net_ev": 0.007,
                "robust_daily_residual_trim_fraction": 0.10,
            },
        }
    )
    _patch_active_policy_mapping(
        payload,
        admission_path=admission_path.resolve().as_posix(),
        regime_path=regime_path.resolve().as_posix(),
        portfolio_path=portfolio_config_path.resolve().as_posix(),
    )
    deployment_text = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    destinations = [canonical_path, deployment_dir / "best_policy_params_perps.json"]
    backup_paths: list[str] = []
    for destination in destinations:
        if destination.exists():
            backup = destination.with_suffix(destination.suffix + ".pre_evaware_geometry_v2")
            if not backup.exists():
                shutil.copy2(destination, backup)
            backup_paths.append(backup.as_posix())
        destination.write_text(deployment_text)

    portfolio_config.update(
        {
            "policy_name": POLICY_NAME,
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": ADMISSION_POLICY_ID,
            "threshold_basis_family": ADMISSION_FAMILY,
            "threshold_basis_window_days": 21,
            "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
            "threshold_basis_fixed_target_net_ev": 0.007,
            "threshold_basis_robust_daily_residual_trim_fraction": 0.10,
            "threshold_basis_hr_rank50": False,
            "threshold_basis_policy_path": admission_path.as_posix(),
            "source_threshold_basis_policy": admission_path.as_posix(),
            "threshold_basis_policy_summary": admission_summary,
            "regime_ev_calibration_enabled": True,
            "regime_ev_calibration_policy_id": POSTPROCESSOR_POLICY_ID,
            "regime_ev_calibration_rank_source": POSTPROCESSOR_POLICY_ID,
            "regime_ev_calibration_artifact_path": regime_path.as_posix(),
        }
    )
    _patch_active_policy_mapping(
        portfolio_config,
        admission_path=admission_path.resolve().as_posix(),
        regime_path=regime_path.resolve().as_posix(),
        portfolio_path=portfolio_config_path.resolve().as_posix(),
    )
    portfolio_config_path.write_text(
        json.dumps(_json_safe(portfolio_config), indent=2, sort_keys=True) + "\n"
    )

    promoted_manifest_path = args.artifact_root / "policy_params" / "promoted_policy_manifest.json"
    promoted = json.loads(promoted_manifest_path.read_text())
    promoted.update(
        {
            "status": ADMISSION_STATUS,
            "archetype_dynamic_layer": ADMISSION_LAYER,
            "exit_geometry_policy_name": POLICY_NAME,
            "exit_geometry_status": "promoted_canonical",
            "exit_geometry_scope": "side_x_policy_archetype_with_side_parent_fallback",
            "exit_geometry_rows": len(strategies),
            "exit_geometry_promoted_at_utc": timestamp.isoformat(),
            "exit_geometry_evidence_manifest": args.evidence_manifest.as_posix(),
            "exit_geometry_deployment_sha256": _sha256(canonical_path),
            "policy_name": POLICY_NAME,
            "portfolio_policy_path": portfolio_config_path.as_posix(),
            "threshold_basis_policy_path": admission_path.as_posix(),
            "threshold_basis_policy_enabled": True,
            "threshold_basis_policy_id": ADMISSION_POLICY_ID,
            "threshold_basis_family": ADMISSION_FAMILY,
            "threshold_basis_window_days": 21,
            "threshold_basis_selection_mode": "fixed_corrected_ev_threshold",
            "threshold_basis_fixed_target_net_ev": 0.007,
            "threshold_basis_robust_daily_residual_trim_fraction": 0.10,
            "threshold_basis_hr_rank50": False,
            "source_threshold_basis_policy": admission_path.as_posix(),
            "threshold_basis_reference_candidates": str(
                admission.get("reference_candidates_path") or ""
            ),
            "threshold_basis_policy_summary": admission_summary,
            "regime_ev_calibration_artifact_path": regime_path.as_posix(),
            "exit_geometry_previous_artifacts": backup_paths,
        }
    )
    promoted.update(
        {
            "regime_ev_predecessor_bundle_path": payload[
                "regime_ev_predecessor_bundle_path"
            ],
            "regime_ev_residual_event_state_path": payload[
                "regime_ev_residual_event_state_path"
            ],
            "regime_ev_calibration_artifact_path": regime_path.resolve().as_posix(),
            "side_residual_expert_artifact_path": payload[
                "side_residual_expert_artifact_path"
            ],
        }
    )
    _patch_active_policy_mapping(
        promoted,
        admission_path=admission_path.resolve().as_posix(),
        regime_path=regime_path.resolve().as_posix(),
        portfolio_path=portfolio_config_path.resolve().as_posix(),
    )
    promoted_manifest_path.write_text(
        json.dumps(_json_safe(promoted), indent=2, sort_keys=True) + "\n"
    )
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
    from extreme_price_movements.inference.training_live_parity_contract import (
        build_training_live_parity_contract,
        load_training_live_parity_contract,
        persist_training_live_parity_contract,
    )
    from extreme_price_movements.model_loader import load_full_state

    data_root = args.artifact_root.parent.parent
    run_id = args.artifact_root.name
    previous_contract = load_training_live_parity_contract(
        data_root=data_root.as_posix(),
        run_id=run_id,
        require=False,
    )
    previous_feature_source = previous_contract.get("feature_source") or {}
    model_bundle = load_full_state(run_id, data_root.as_posix())
    orchestrator = ModelOrchestrator(
        model_bundle,
        {
            "inference_model_timing_enabled": False,
            "preserve_logged_meta_model_derived_features": True,
        },
    )
    strategy_ids = (
        (portfolio_config.get("strategy_contract") or {}).get("strategy_ids")
        or [
            "long_s52_meta_threshold_handoff",
            "short_s52_meta_threshold_handoff",
        ]
    )
    parity_contract = build_training_live_parity_contract(
        data_root=data_root.as_posix(),
        run_id=run_id,
        market_mode="perps",
        orchestrator=orchestrator,
        model_bundle=model_bundle,
        strategy_ids=strategy_ids,
        deployment_payload=payload,
        portfolio_payload=portfolio_config,
        feature_source_run_id=previous_feature_source.get("run_id"),
        feature_source_data_root=previous_feature_source.get("data_root"),
    )
    parity_paths = persist_training_live_parity_contract(
        parity_contract,
        data_root=data_root.as_posix(),
        run_id=run_id,
    )
    print(
        json.dumps(
            {
                "policy_name": POLICY_NAME,
                "strategies": len(strategies),
                "deployment": canonical_path.as_posix(),
                "sha256": _sha256(canonical_path),
                "backups": backup_paths,
                "parity_contracts": [path.as_posix() for path in parity_paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
