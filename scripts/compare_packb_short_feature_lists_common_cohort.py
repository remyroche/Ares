#!/usr/bin/env python3
"""Compare Pack-B short feature-list winners on identical complete-case rows.

The v1 short winner is raw-only while the v2 short winner includes frozen
side-local AE/GMM outputs.  Their normal HPO reports therefore cover slightly
different complete-case rows.  This gate loads the union once per fixed fold,
uses its joint-complete rows for both candidates, refits each already-selected
parameter set with the same seed, and ranks the candidates by the production
cost-aware objective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.packb_side_local_fs_hpo_stage import (
    _prepare_dataset_pair,
    _stage_dataset_sha256,
)
from extreme_price_movements.training_resource_guard import (
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts import run_packb_pre_march_side_fs_hpo as production
from scripts.prepare_packb_pre_march_side_contracts import parse_locked_dec09
from scripts.run_packb_pre_march_side_ae import (
    DEFAULT_DECISIONS,
    DEFAULT_FEATURE_INVENTORY,
    DEFAULT_FEATURE_STORE,
    DEFAULT_POPULATION_ROOT,
    _feature_inventory_binding,
    _source_contracts,
)

DEFAULT_V1_ROOT = ROOT / "data_perp/artifacts/packb_side_local_fs_hpo_20260724_v1"
DEFAULT_V2_ROOT = ROOT / "data_perp/artifacts/packb_side_local_fs_hpo_20260724_v2"
DEFAULT_AE_ROOT = production.DEFAULT_AE_ROOT
DEFAULT_LABELS = production.DEFAULT_LABELS
DEFAULT_RECENT_COLUMNS = production.RECENT_WINNER_FEATURE_CONTRACT
DEFAULT_RECENT_MANIFEST = production.RECENT_WINNER_PROCESS_MANIFEST
DEFAULT_OUTPUT = (
    ROOT / "docs/pipeline_roadmap/20260724/r3/"
    "packb_short_feature_list_common_cohort_gate_v1.json"
)
DEFAULT_TELEMETRY = (
    ROOT / "data_perp/artifacts/"
    "packb_short_feature_list_common_cohort_gate_20260724_v1/"
    "training_resource_telemetry.jsonl"
)


class CommonCohortGateError(RuntimeError):
    """Raised when the comparator cannot prove a common-cohort result."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CommonCohortGateError(f"JSON object required: {path}")
    return value


def _candidate(root: Path) -> dict[str, Any]:
    path = root / "short/hpo_parameters.json"
    payload = _json(path)
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise CommonCohortGateError(f"missing HPO selection: {path}")
    features = tuple(map(str, payload.get("selected_features", ())))
    params = selection.get("selected_params")
    trial_id = str(selection.get("selected_trial_id") or "")
    if not features or not isinstance(params, Mapping) or not trial_id:
        raise CommonCohortGateError(f"incomplete winner contract: {path}")
    return {
        "root": root,
        "path": path,
        "sha256": stage_manifest.sha256_file(path),
        "fixed_calendar_sha256": str(payload["fixed_calendar_sha256"]),
        "features": features,
        "params": dict(params),
        "trial_id": trial_id,
    }


def _aggregate(candidate_id: str, folds: list[dict[str, Any]]) -> dict[str, Any]:
    if len(folds) != 3:
        raise CommonCohortGateError("each candidate requires exactly three folds")
    objectives = np.asarray([row["objective"] for row in folds], dtype=np.float64)
    return {
        "candidate_id": candidate_id,
        "mean_objective": float(np.mean(objectives)),
        "worst_fold_objective": float(np.min(objectives)),
        "objective_std": float(np.std(objectives, ddof=1)),
        "fold_results": folds,
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["mean_objective"]),
            -float(row["worst_fold_objective"]),
            float(row["objective_std"]),
            str(row["candidate_id"]),
        ),
    )


def _historical_pair_audit(
    *,
    admitted_features: set[str],
    columns_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    columns = _json(columns_path)
    manifest = _json(manifest_path)
    historical = tuple(map(str, columns["feature_names_by_side"]["short"]))
    missing = sorted(set(historical) - admitted_features)
    return {
        "feature_contract_path": str(columns_path.relative_to(ROOT)),
        "feature_contract_sha256": stage_manifest.sha256_file(columns_path),
        "process_manifest_path": str(manifest_path.relative_to(ROOT)),
        "process_manifest_sha256": stage_manifest.sha256_file(manifest_path),
        "feature_count": len(historical),
        "available_in_locked_inventory_count": len(historical) - len(missing),
        "missing_from_locked_inventory": missing,
        "selection_scope": manifest.get("feature_selection_scope"),
        "selection_calibration_fold": manifest.get(
            "feature_selection_calibration_fold"
        ),
        "backward_reuse_note": manifest.get(
            "feature_selection_global_calibration_note"
        ),
        "strict_pre_march_eligibility": "REJECTED_FUTURE_SELECTED_AND_INCOMPLETE",
        "comparison_disposition": (
            "not comparable on the locked pre-March common cohort; retain as "
            "a post-June serving/reference contract only"
        ),
    }


def run(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    v1_root: Path = DEFAULT_V1_ROOT,
    v2_root: Path = DEFAULT_V2_ROOT,
    ae_root: Path = DEFAULT_AE_ROOT,
    labels_dir: Path = DEFAULT_LABELS,
    population_root: Path = DEFAULT_POPULATION_ROOT,
    feature_store: Path = DEFAULT_FEATURE_STORE,
    feature_inventory_path: Path = DEFAULT_FEATURE_INVENTORY,
    decisions_path: Path = DEFAULT_DECISIONS,
    telemetry_path: Path = DEFAULT_TELEMETRY,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise CommonCohortGateError(f"refusing to overwrite gate: {output}")
    telemetry_path = Path(telemetry_path)
    if telemetry_path.exists():
        raise CommonCohortGateError(
            f"refusing to overwrite gate telemetry: {telemetry_path}"
        )
    candidates = {
        "v1_short_8": _candidate(Path(v1_root)),
        "v2_short_36": _candidate(Path(v2_root)),
    }
    calendar_hashes = {value["fixed_calendar_sha256"] for value in candidates.values()}
    if len(calendar_hashes) != 1:
        raise CommonCohortGateError("candidate calendars differ")

    population_manifest, source_hashes, calendar_sha256, _binding = _source_contracts(
        population_root=Path(population_root),
        feature_inventory_path=Path(feature_inventory_path),
        decisions_path=Path(decisions_path),
    )
    if calendar_hashes != {calendar_sha256}:
        raise CommonCohortGateError("candidate calendar is not the locked calendar")
    dec09 = parse_locked_dec09(Path(decisions_path))
    if stage_manifest.canonical_json_sha256(dec09["calendar"]) != calendar_sha256:
        raise CommonCohortGateError("fixed calendar binding changed")

    expected_tree = _feature_inventory_binding(Path(feature_inventory_path))
    current_tree = production.hash_path(Path(feature_store))
    if (
        current_tree.get("sha256") != expected_tree["tree_sha256"]
        or current_tree.get("bytes") != expected_tree["bytes"]
        or current_tree.get("files") != expected_tree["files"]
    ):
        raise CommonCohortGateError("canonical feature store changed")

    ae_summary = _json(Path(ae_root) / "summary.json")
    ae_revision = str(ae_summary["source_revision"])
    loader_root = Path(ae_root) / "short/loader_evidence"
    contract, bundle, _extra_hashes = production._load_loader_contract(
        loader_root, source_revision=ae_revision
    )
    ae_manifest_path = Path(ae_root) / "short/ae_gmm/side_stage_manifest.json"
    ae_manifest = stage_manifest.validate_side_stage_manifest(
        ae_manifest_path,
        expected_side="short",
        expected_stage="ae_gmm",
        expected_source_hashes=source_hashes,
        expected_fixed_calendar_sha256=calendar_sha256,
    )
    state_path = Path(ae_root) / "short/ae_gmm" / str(ae_manifest["artifact"]["path"])
    state = production._load_side_ae_state(
        state_path,
        expected_side="short",
        expected_sha256=str(ae_manifest["artifact"]["sha256"]),
        raw_features=contract["feature_columns"],
    )
    generated = production._active_ae_gmm_columns(state)
    available = set(map(str, contract["feature_columns"])) | set(generated)
    union = tuple(
        dict.fromkeys(
            feature
            for candidate in candidates.values()
            for feature in candidate["features"]
        )
    )
    unavailable = sorted(set(union) - available)
    if unavailable:
        raise CommonCohortGateError(
            "candidate features unavailable: " + ", ".join(unavailable)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=output.parent,
        telemetry_path=telemetry_path,
    )
    guard.preflight("packb_short_common_cohort:preflight")
    raw_loader = production.make_fs_hpo_raw_feature_loader(
        feature_store_dir=feature_store,
        feature_contract=contract,
        evidence_bundle=bundle,
        resource_guard=guard,
    )
    feature_loader = production.SideRepresentationFeatureLoader(
        raw_loader=raw_loader,
        raw_features=contract["feature_columns"],
        state=state,
        generated_features=generated,
    )
    label_files = production._canonical_label_files(
        Path(labels_dir), population_manifest
    )
    labels = production.ExactLabelLoader(label_files, resource_guard=guard)
    fold_metrics: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidates
    }
    common_dataset_hashes: dict[str, dict[str, str]] = {}

    for fold_index, fold in enumerate(
        production._folds(Path(population_root), "short"), start=1
    ):
        guard.checkpoint(f"packb_short_common_cohort:{fold.name}:before_union_load")
        train, valid, admitted, coverage = _prepare_dataset_pair(
            train_ledger=fold.train_ledger,
            valid_ledger=fold.valid_ledger,
            features=union,
            feature_loader=feature_loader,
            target_loader=labels.target,
            weight_loader=labels.weights,
            name=f"short common cohort {fold.name}",
            allow_feature_pruning=False,
        )
        if set(admitted) != set(union):
            raise CommonCohortGateError("common-cohort union was altered")
        common_dataset_hashes[fold.name] = {
            "train": _stage_dataset_sha256(train),
            "validation": _stage_dataset_sha256(valid),
        }
        for candidate_id, candidate in candidates.items():
            guard.checkpoint(
                f"packb_short_common_cohort:{fold.name}:{candidate_id}:before_fit"
            )
            features = list(candidate["features"])
            _model, prediction, best_iteration = production._fit_predict(
                train.features.loc[:, features],
                train.target,
                train.weights,
                valid.features.loc[:, features],
                valid.target,
                valid.weights,
                candidate["params"],
                # Same seed across feature-list candidates within each fold.
                seed=20269724 + 1_000 * fold_index,
            )
            metrics = production._economic_objective(
                prediction,
                valid.target.to_numpy(dtype=np.float64),
                valid.weights.to_numpy(dtype=np.float64),
                labels.economic(valid.ledger),
                timestamps=valid.ledger["__ts__"],
                symbols=valid.ledger["__symbol__"],
            )
            fold_metrics[candidate_id].append(
                {
                    "fold_name": fold.name,
                    **metrics,
                    "best_iteration": int(best_iteration),
                    "train_rows": len(train.ledger),
                    "validation_rows": len(valid.ledger),
                    "selected_feature_count": len(features),
                    "common_union_feature_count": len(union),
                    "common_train_joint_complete_fraction": coverage["train"][
                        "joint_complete_fraction"
                    ],
                    "common_validation_joint_complete_fraction": coverage["validation"][
                        "joint_complete_fraction"
                    ],
                }
            )
        del train, valid
        production._release_memory()

    ranking = _rank(
        [_aggregate(candidate_id, rows) for candidate_id, rows in fold_metrics.items()]
    )
    winner_id = str(ranking[0]["candidate_id"])
    historical_audit = _historical_pair_audit(
        admitted_features=available,
        columns_path=DEFAULT_RECENT_COLUMNS,
        manifest_path=DEFAULT_RECENT_MANIFEST,
    )
    report = {
        "schema": "packb_short_feature_list_common_cohort_gate_v1",
        "status": "PASS_COMMON_COHORT_MODEL_SELECTION",
        "source_revision": production._git_revision(),
        "selection_scope": "short_side_only",
        "selection_metric": (
            "mean_three_fold_cost_aware_economic_objective_then_"
            "worst_fold_then_stability"
        ),
        "fixed_calendar_sha256": calendar_sha256,
        "hpo_validation_months": ["2025-12", "2026-01", "2026-02"],
        "common_cohort_contract": {
            "features": "union_of_v1_and_v2_short_lists",
            "feature_count": len(union),
            "joint_complete_rows": True,
            "same_rows_per_candidate": True,
            "same_seed_per_fold_per_candidate": True,
            "cost_column": production.ECONOMIC_COLUMN,
            "cost_accounting": "canonical first-touch net return, applied once",
            "common_dataset_sha256_by_fold": common_dataset_hashes,
        },
        "candidates": {
            candidate_id: {
                "root": str(candidate["root"].relative_to(ROOT)),
                "hpo_parameters_path": str(candidate["path"].relative_to(ROOT)),
                "hpo_parameters_sha256": candidate["sha256"],
                "selected_trial_id": candidate["trial_id"],
                "selected_feature_count": len(candidate["features"]),
                "selected_features": list(candidate["features"]),
                "selected_params": candidate["params"],
            }
            for candidate_id, candidate in candidates.items()
        },
        "ranking": ranking,
        "winner": winner_id,
        "promotion": {
            "long": {
                "source": "data_perp/artifacts/packb_side_local_fs_hpo_20260724_v2/long",
                "reason": (
                    "v1 and v2 long winners already use identical complete-case "
                    "rows; v2 wins the primary mean objective"
                ),
            },
            "short": {
                "source": str(candidates[winner_id]["root"].relative_to(ROOT))
                + "/short",
                "reason": "winner of this identical-row common-cohort gate",
            },
        },
        "recent_55_long_37_short_audit": historical_audit,
        "resource_telemetry": {
            "path": str(telemetry_path.relative_to(ROOT)),
        },
    }
    guard.checkpoint("packb_short_common_cohort:complete")
    report["resource_telemetry"]["sha256"] = stage_manifest.sha256_file(telemetry_path)
    stage_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(stage_path, output)
    report["output_path"] = str(output)
    report["output_sha256"] = stage_manifest.sha256_file(output)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run(output_path=args.output)
    print(
        json.dumps(
            {
                "output_path": report["output_path"],
                "output_sha256": report["output_sha256"],
                "winner": report["winner"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
