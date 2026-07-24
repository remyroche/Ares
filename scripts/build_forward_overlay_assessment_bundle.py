#!/usr/bin/env python3
"""Materialize the frozen, shadow-only residual-overlay assessment bundle.

The bundle deliberately does not modify model ranks, policy thresholds, live
inference, sizing, or portfolio allocation.  It records the exact contracts
needed to assess four independently attributable components on future resolved
rows:

* PCA8 residual overlay plus sparse global shock composite (reference);
* short-default uncertainty challenger;
* nested short-default leverage-rebuild residual-state family;
* the combined shadow comparison.

Large source artifacts are referenced and hashed rather than copied.  This
keeps the bundle lightweight while making provenance checks fail closed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.residual_state_family_features import (
    ResidualStateFamilyContract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPARSE_SHOCK = ROOT / (
    "data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1/"
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_"
    "baseline_globaloverlay_sparse_shock_composite"
)
DEFAULT_UNCERTAINTY = ROOT / (
    "data_perp/reports/short_default_uncertainty_"
    "forward_challenger_20260713_v2_schema_contract"
)
DEFAULT_RESIDUAL_FAMILIES = ROOT / (
    "data_perp/reports/nested_residual_state_families_20260712_v1"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/forward_overlay_assessment_bundle_20260714_v1"

REFERENCE_ARM = (
    "lifecycle_residual_aware_ae_gmm_overlay_pca8_clip8_"
    "baseline_globaloverlay_sparse_shock_composite"
)
SHORT_DEFAULT_SIDE = "short"
SHORT_DEFAULT_ARCHETYPE = "short_default_clean_path"
LEVERAGE_BASE = "short_covering_score_market"
LEVERAGE_GATE = "funding_confirmed_long_flush"
LEVERAGE_FORM = "positive"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": _relative_or_absolute(path), "sha256": _sha256(path)}


def _validated_leverage_definition(contract_path: Path) -> dict[str, Any]:
    contract = ResidualStateFamilyContract.from_dict(_read_json(contract_path))
    matches = [
        definition
        for definition in contract.definitions
        if definition.side_name == SHORT_DEFAULT_SIDE
        and definition.archetype_policy_key == SHORT_DEFAULT_ARCHETYPE
        and definition.base_feature == LEVERAGE_BASE
        and definition.gate_feature == LEVERAGE_GATE
        and definition.form == LEVERAGE_FORM
        and definition.status == "validated_production_candidate"
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one validated short-default leverage-rebuild definition; "
            f"found {len(matches)}."
        )
    return {
        **matches[0].__dict__,
        "contract_hash": contract.contract_hash,
        "source_feature_contract_hash": contract.source_feature_contract_hash,
        "fit_end": contract.fit_end,
    }


def _registry_metrics(registry_path: Path) -> dict[str, Any]:
    registry = pd.read_csv(registry_path)
    matched = registry.loc[
        registry["side_name"].eq(SHORT_DEFAULT_SIDE)
        & registry["archetype_policy_key"].eq(SHORT_DEFAULT_ARCHETYPE)
        & registry["base_feature"].eq(LEVERAGE_BASE)
        & registry["gate_feature"].eq(LEVERAGE_GATE)
        & registry["form"].eq(LEVERAGE_FORM)
        & registry["nested_promoted"].eq(True)
    ]
    if len(matched) != 1:
        raise ValueError(
            "Expected one nested-promoted leverage registry row; "
            f"found {len(matched)}."
        )
    row = matched.iloc[0]
    columns = (
        "mean_lift",
        "mean_fpr",
        "mean_precision_gain",
        "mean_auc_gain",
        "fold_lift_std",
        "positive_folds",
        "evaluated_folds",
        "adverse_support",
        "search_score",
        "status",
    )
    return {
        column: (row[column].item() if hasattr(row[column], "item") else row[column])
        for column in columns
    }


def _write_csv_template(path: Path) -> None:
    fields = [
        "assessment_timestamp",
        "decision_timestamp",
        "resolved_timestamp",
        "symbol",
        "side_name",
        "archetype_policy_key",
        "baseline_selector",
        "baseline_rank",
        "baseline_selected",
        "sparse_global_shock_score",
        "short_default_uncertainty_score",
        "short_default_uncertainty_rank",
        "short_default_uncertainty_candidate_selected",
        "residual_state_family_leverage_rebuild_pct",
        "residual_state_family_leverage_rebuild_active",
        "residual_state_family_leverage_rebuild_computable",
        "leverage_rebuild_shadow_alert",
        "combined_shadow_alert",
        "realized_ev_after_1pct",
        "realized_clean_exec",
        "realized_bad_mae",
        "realized_timeout",
        "realized_stop_or_adverse",
        "outcome_resolved",
        "feature_schema_hash",
        "uncertainty_normalization_array_hash",
        "leverage_family_contract_hash",
    ]
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def run(args: argparse.Namespace) -> dict[str, Any]:
    sparse_manifest_path = _require(args.sparse_shock / "manifest.json")
    sparse_predictions_path = _require(args.sparse_shock / "oos_predictions_historical_rank.parquet")
    uncertainty_manifest_path = _require(args.uncertainty / "manifest.json")
    uncertainty_normalization_path = _require(args.uncertainty / "normalization_references.npz")
    uncertainty_replication_path = _require(args.uncertainty / "oos_replication_predictions.parquet")
    family_contract_path = _require(args.residual_families / "residual_state_family_contract.json")
    family_registry_path = _require(args.residual_families / "definition_registry.csv")

    sparse = _read_json(sparse_manifest_path)
    uncertainty = _read_json(uncertainty_manifest_path)
    if sparse.get("arm") != REFERENCE_ARM:
        raise ValueError(f"Unexpected sparse-shock reference arm: {sparse.get('arm')!r}")
    if uncertainty.get("status") != "frozen_research_challenger_not_live":
        raise ValueError("The uncertainty challenger must remain frozen research-only.")
    if uncertainty.get("activation") != "none":
        raise ValueError("The uncertainty challenger must not be active in live policy.")

    leverage_definition = _validated_leverage_definition(family_contract_path)
    leverage_metrics = _registry_metrics(family_registry_path)
    args.output.mkdir(parents=True, exist_ok=True)

    component_contract = {
        "status": "shadow_forward_assessment_only",
        "reference_component": {
            "name": "pca8_residual_overlay_plus_sparse_global_shock_composite",
            "selector": REFERENCE_ARM,
            "parameters": sparse["selected_side_parameters"],
            "historical_replication": sparse["challenger"],
            "required_runtime_columns": ["sparse_global_shock_score", "baseline_rank"],
        },
        "uncertainty_component": {
            "name": uncertainty["candidate_id"],
            "scope": {
                "side_name": SHORT_DEFAULT_SIDE,
                "archetype_policy_key": SHORT_DEFAULT_ARCHETYPE,
            },
            "parameters": uncertainty["candidate"],
            "feature_schema_hash": uncertainty["feature_schema"]["hash"],
            "normalization_array_hash": uncertainty["provenance_hashes"]["normalization_array_hash"],
            "historical_replication": uncertainty["replication"]["short_default_challenger"],
            "frozen_parent": uncertainty["parent_v11"],
            "required_runtime_columns": [
                "short_default_uncertainty_score",
                "short_default_uncertainty_rank",
            ],
            "activation": "shadow_only",
        },
        "leverage_rebuild_component": {
            "name": "short_default_leverage_rebuild_nested_validated",
            "scope": {
                "side_name": SHORT_DEFAULT_SIDE,
                "archetype_policy_key": SHORT_DEFAULT_ARCHETYPE,
            },
            "definition": leverage_definition,
            "nested_validation": leverage_metrics,
            "required_runtime_columns": [
                LEVERAGE_BASE,
                LEVERAGE_GATE,
                "residual_state_family_leverage_rebuild_pct",
                "residual_state_family_leverage_rebuild_active",
                "residual_state_family_leverage_rebuild_computable",
            ],
            "activation": "shadow_only",
        },
        "combined_component": {
            "name": "pca8_sparse_shock_plus_short_default_uncertainty_plus_leverage_rebuild",
            "rule": (
                "Record all components jointly.  The uncertainty challenger was "
                "validated against its frozen V11 parent, not the PCA8 sparse-shock "
                "rank, so it must remain a parallel shadow until a new causal "
                "cross-parent calibration is validated."
            ),
            "activation": "shadow_only",
        },
    }
    (args.output / "component_contract.json").write_text(
        json.dumps(component_contract, indent=2, sort_keys=True) + "\n"
    )
    _write_csv_template(args.output / "forward_assessment_rows_template.csv")

    manifest = {
        "schema": "forward_overlay_assessment_bundle_v1",
        "status": "shadow_forward_assessment_only",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Forward assessment of PCA8+sparse-shock reference versus frozen "
            "short-default uncertainty and validated leverage-rebuild diagnostics."
        ),
        "live_policy_change": False,
        "rank_or_threshold_change": False,
        "sizing_or_portfolio_change": False,
        "component_contract": "component_contract.json",
        "assessment_template": "forward_assessment_rows_template.csv",
        "assessment_materializer": "scripts/materialize_forward_overlay_assessment_rows.py",
        "comparison": {
            "reference": "PCA8 residual overlay plus sparse global shock composite",
            "shadow_a": "short-default uncertainty challenger versus its frozen V11 parent",
            "shadow_b": "reference plus leverage-rebuild diagnostic",
            "shadow_c": "all components recorded jointly, without a merged rank",
            "policy_instruction": (
                "Do not promote, tune, or merge the shadows on the same forward period. "
                "A combined score requires a separate causal calibration and evaluation."
            ),
        },
        "forward_metrics": [
            "selected_rows",
            "activity_retained",
            "mean_ev_after_1pct",
            "sum_ev_after_1pct",
            "clean_exec_precision",
            "bad_mae_rate",
            "timeout_rate",
            "stop_or_adverse_rate",
            "worst_day_ev",
            "largest_positive_day_share",
            "by_side_archetype_and_day",
        ],
        "required_outcome_rule": (
            "Materialize realized metrics only after each path has resolved; no "
            "outcome may influence same-period component scores or admission."
        ),
        "source_artifacts": {
            "sparse_shock_manifest": _artifact(sparse_manifest_path),
            "sparse_shock_historical_predictions": _artifact(sparse_predictions_path),
            "uncertainty_manifest": _artifact(uncertainty_manifest_path),
            "uncertainty_normalization": _artifact(uncertainty_normalization_path),
            "uncertainty_historical_predictions": _artifact(uncertainty_replication_path),
            "residual_state_family_contract": _artifact(family_contract_path),
            "residual_state_definition_registry": _artifact(family_registry_path),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-shock", type=Path, default=DEFAULT_SPARSE_SHOCK)
    parser.add_argument("--uncertainty", type=Path, default=DEFAULT_UNCERTAINTY)
    parser.add_argument("--residual-families", type=Path, default=DEFAULT_RESIDUAL_FAMILIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
