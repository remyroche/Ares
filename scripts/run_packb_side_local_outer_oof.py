#!/usr/bin/env python3
"""Train routed side-local Pack-B models and emit strict April--July OOF.

The runner consumes the immutable outer-fold population plus the frozen
promotion contract.  Feature selection and HPO are never repeated here.  Each
side/fold receives its own fitted LightGBM model, and the final refit is stored
separately and explicitly excluded from OOF metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.packb_side_local_fs_hpo_stage import (
    _bounded_beginning_middle_end_sample,
)
from extreme_price_movements.training_resource_guard import (
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.run_packb_historical_feature_hpo import (
    CachedRepresentationFeatureLoader,
    ExactCandidateFeatureLoader,
    HistoricalCompositeFeatureLoader,
    _label_schema,
)
from scripts.run_packb_pre_march_side_ae import (
    DEFAULT_DECISIONS,
    DEFAULT_FEATURE_INVENTORY,
    DEFAULT_FEATURE_STORE,
    DEFAULT_POPULATION_ROOT,
    _source_contracts,
)
from scripts.run_packb_pre_march_side_fs_hpo import (
    ECONOMIC_COLUMN,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    ExactLabelLoader,
    SideRepresentationFeatureLoader,
    _active_ae_gmm_columns,
    _canonical_label_files,
    _economic_objective,
    _git_revision,
    _lgbm_regressor,
    _load_loader_contract,
    _load_side_ae_state,
    _release_memory,
    make_fs_hpo_raw_feature_loader,
)

SCHEMA = "packb_side_local_outer_oof_runner_v1"
SIDES = ("long", "short")
DEFAULT_OUTER_POPULATION = (
    ROOT / "data_perp/artifacts/packb_outer_oof_population_20260724_v1"
)
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_PROMOTION = (
    ROOT / "docs/pipeline_roadmap/20260724/r3/packb_side_fs_hpo_promotion_v1.json"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/packb_side_local_outer_oof_20260724_v1"
DEFAULT_LABELS = (
    ROOT / "data_perp/artifacts/"
    "20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_TRAIN_MAX_ROWS = 100_000
DEFAULT_FINAL_MAX_ROWS = 150_000
REQUIRED_LEDGER_COLUMNS = (
    "candidate_id",
    "side_name",
    "__ts__",
    "__decision_ts__",
    "__label_resolution_ts__",
    "__symbol__",
)


class PackBOuterOOFRunnerError(RuntimeError):
    """Raised when routed outer-OOF evidence cannot be proven."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBOuterOOFRunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackBOuterOOFRunnerError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _validate_outer_manifest(
    outer_root: Path, decisions_path: Path
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    manifest_path = outer_root / "manifest.json"
    manifest = _json(manifest_path)
    if (
        manifest.get("schema") != "packb_outer_oof_population_materialization_v1"
        or manifest.get("status") != "MATERIALIZED_IMMUTABLE"
    ):
        raise PackBOuterOOFRunnerError("immutable outer population is required")
    if manifest.get("dec09", {}).get("sha256") != stage_manifest.sha256_file(
        decisions_path
    ):
        raise PackBOuterOOFRunnerError("outer population DEC-09 binding changed")
    folds = manifest.get("calendar", {}).get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise PackBOuterOOFRunnerError("exactly four fixed outer folds are required")
    names: set[str] = set()
    normalised: list[dict[str, Any]] = []
    previous_end: pd.Timestamp | None = None
    for item in folds:
        if not isinstance(item, Mapping):
            raise PackBOuterOOFRunnerError("outer fold record is invalid")
        name = str(item.get("name") or "")
        start = pd.Timestamp(item.get("validation_start_utc"))
        end = pd.Timestamp(item.get("validation_end_utc"))
        if (
            not name
            or name in names
            or start.tzinfo is None
            or end.tzinfo is None
            or start >= end
            or (previous_end is not None and start != previous_end)
        ):
            raise PackBOuterOOFRunnerError("outer folds are not unique and contiguous")
        names.add(name)
        previous_end = end
        normalised.append(
            {
                "name": name,
                "start": start.tz_convert("UTC"),
                "end": end.tz_convert("UTC"),
            }
        )
    return manifest, tuple(normalised)


def _load_bound_ledger(
    outer_root: Path,
    outer_manifest: Mapping[str, Any],
    *,
    fold: Mapping[str, Any],
    side: str,
    role: str,
) -> tuple[pd.DataFrame, Path, Mapping[str, Any]]:
    key = f"{fold['name']}/{side}/{role}"
    record = outer_manifest.get("ledgers", {}).get(key)
    if not isinstance(record, Mapping):
        raise PackBOuterOOFRunnerError(f"outer ledger record is missing: {key}")
    path = outer_root / str(record.get("path") or "")
    if not path.is_file() or stage_manifest.sha256_file(path) != record.get("sha256"):
        raise PackBOuterOOFRunnerError(f"outer ledger changed: {key}")
    frame = pd.read_parquet(path)
    if list(frame.columns) != list(REQUIRED_LEDGER_COLUMNS):
        raise PackBOuterOOFRunnerError(f"outer ledger schema changed: {key}")
    if len(frame) != int(record.get("rows", -1)):
        raise PackBOuterOOFRunnerError(f"outer ledger row count changed: {key}")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise PackBOuterOOFRunnerError(f"outer ledger has duplicate IDs: {key}")
    if set(frame["side_name"].astype(str)) != {side}:
        raise PackBOuterOOFRunnerError(f"outer ledger mixed sides: {key}")
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    resolution = pd.to_datetime(
        frame["__label_resolution_ts__"], utc=True, errors="raise"
    )
    if (
        not decision.eq(signal + pd.Timedelta(hours=1)).all()
        or not resolution.eq(decision + pd.Timedelta(hours=24)).all()
    ):
        raise PackBOuterOOFRunnerError(f"outer ledger timing changed: {key}")
    start, end = fold["start"], fold["end"]
    if role == "train":
        valid = signal.lt(start - pd.Timedelta(hours=25)) & resolution.lt(start)
    else:
        valid = signal.ge(start) & signal.lt(end)
    if not valid.all():
        raise PackBOuterOOFRunnerError(f"outer ledger violates cutoff: {key}")
    return frame, path, record


def _load_promotion(
    path: Path, *, fixed_calendar_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    promotion = _json(path)
    schema = promotion.get("schema")
    status = promotion.get("status")
    if (
        schema
        not in {
            "packb_side_fs_hpo_promotion_v1",
            "packb_side_fs_hpo_promotion_v2",
        }
        or status
        not in {
            "FROZEN_SIDE_ROUTED_FEATURE_SELECTION_AND_HPO",
            "FROZEN_HISTORICAL_FEATURE_EXCEPTION_WITH_STRICT_PRE_MARCH_HPO",
        }
        or promotion.get("fixed_calendar_sha256") != fixed_calendar_sha256
    ):
        raise PackBOuterOOFRunnerError("frozen routed promotion is required")
    result: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        route = promotion.get("sides", {}).get(side)
        if not isinstance(route, Mapping):
            raise PackBOuterOOFRunnerError(f"promotion route missing for {side}")
        source_root = ROOT / str(route.get("source_root") or "")
        feature_path = source_root / "feature_contract.json"
        parameter_path = source_root / "hpo_parameters.json"
        if stage_manifest.sha256_file(feature_path) != route.get(
            "feature_contract_sha256"
        ) or stage_manifest.sha256_file(parameter_path) != route.get(
            "hpo_parameters_sha256"
        ):
            raise PackBOuterOOFRunnerError(f"promoted {side} artifacts changed")
        feature_artifact = _json(feature_path)
        parameter_artifact = _json(parameter_path)
        features = tuple(map(str, feature_artifact.get("selected_features", ())))
        selection = parameter_artifact.get("selection")
        if not isinstance(selection, Mapping):
            raise PackBOuterOOFRunnerError(f"promoted {side} HPO selection is absent")
        params = selection.get("selected_params")
        if (
            feature_artifact.get("side") != side
            or parameter_artifact.get("side") != side
            or tuple(map(str, parameter_artifact.get("selected_features", ())))
            != features
            or len(features) != int(route.get("selected_feature_count", -1))
            or selection.get("selected_trial_id") != route.get("selected_trial_id")
            or not isinstance(params, Mapping)
            or not params
        ):
            raise PackBOuterOOFRunnerError(f"promoted {side} route is inconsistent")
        result[side] = {
            "features": features,
            "params": dict(params),
            "trial_id": str(selection["selected_trial_id"]),
            "feature_contract_path": feature_path,
            "parameter_path": parameter_path,
            "route": dict(route),
            "loader_kind": str(route.get("loader_kind") or "ae_gmm_only"),
            "missing_value_policy": str(
                route.get("missing_value_policy") or "joint_complete"
            ),
            "min_per_feature_finite_fraction": float(
                route.get("min_per_feature_finite_fraction", 1.0)
            ),
        }
        if result[side]["loader_kind"] not in {
            "ae_gmm_only",
            "historical_candidate_static_ae_gmm",
        }:
            raise PackBOuterOOFRunnerError(
                f"promoted {side} loader kind is unsupported"
            )
        if result[side]["missing_value_policy"] not in {
            "joint_complete",
            "lightgbm_native_nan",
        }:
            raise PackBOuterOOFRunnerError(
                f"promoted {side} missing-value policy is unsupported"
            )
    if (
        result["long"]["feature_contract_path"]
        == result["short"]["feature_contract_path"]
    ):
        raise PackBOuterOOFRunnerError("long and short promotion artifacts are shared")
    return promotion, result


def _admit_complete(
    ledger: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(ledger) != len(features) or len(ledger) != len(labels):
        raise PackBOuterOOFRunnerError("feature/label loader changed row alignment")
    matrix = features.replace([np.inf, -np.inf], np.nan)
    finite = np.isfinite(matrix.to_numpy(dtype=np.float32, copy=False)).all(axis=1)
    target = pd.to_numeric(labels[TARGET_COLUMN], errors="coerce").to_numpy()
    weight = pd.to_numeric(labels[WEIGHT_COLUMN], errors="coerce").to_numpy()
    economic = pd.to_numeric(labels[ECONOMIC_COLUMN], errors="coerce").to_numpy()
    admitted = (
        finite
        & np.isfinite(target)
        & np.isfinite(weight)
        & np.isfinite(economic)
        & (weight >= 0.0)
    )
    count = int(admitted.sum())
    if count < 1 or float(weight[admitted].sum()) <= 0.0:
        raise PackBOuterOOFRunnerError("no positive-weight joint-complete rows")
    evidence = {
        "raw_rows": int(len(ledger)),
        "admitted_rows": count,
        "attrited_rows": int(len(ledger) - count),
        "joint_complete_fraction": float(count / len(ledger)),
        "policy": "no_imputation_joint_complete_features_and_labels",
    }
    return (
        ledger.loc[admitted].reset_index(drop=True),
        matrix.loc[admitted].reset_index(drop=True),
        labels.loc[admitted].reset_index(drop=True),
        evidence,
    )


def _admit_native_missing(
    ledger: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    min_per_feature_finite_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(ledger) != len(features) or len(ledger) != len(labels):
        raise PackBOuterOOFRunnerError("feature/label loader changed row alignment")
    matrix = features.replace([np.inf, -np.inf], np.nan)
    finite_fraction = matrix.notna().mean()
    rejected = finite_fraction.loc[
        finite_fraction.lt(float(min_per_feature_finite_fraction))
    ]
    if not rejected.empty:
        raise PackBOuterOOFRunnerError(
            "historical feature coverage fell below the frozen native-missing floor: "
            + ", ".join(f"{name}={value:.6f}" for name, value in rejected.items())
        )
    target = pd.to_numeric(labels[TARGET_COLUMN], errors="coerce").to_numpy()
    weight = pd.to_numeric(labels[WEIGHT_COLUMN], errors="coerce").to_numpy()
    economic = pd.to_numeric(labels[ECONOMIC_COLUMN], errors="coerce").to_numpy()
    admitted = (
        np.isfinite(target)
        & np.isfinite(weight)
        & np.isfinite(economic)
        & (weight >= 0.0)
    )
    count = int(admitted.sum())
    if count < 1 or float(weight[admitted].sum()) <= 0.0:
        raise PackBOuterOOFRunnerError("no positive-weight label-complete rows")
    evidence = {
        "raw_rows": int(len(ledger)),
        "admitted_rows": count,
        "attrited_rows": int(len(ledger) - count),
        "joint_complete_fraction": float(
            matrix.notna().all(axis=1).loc[admitted].mean()
        ),
        "minimum_per_feature_finite_fraction": float(finite_fraction.min()),
        "per_feature_finite_fraction": {
            str(name): float(value) for name, value in finite_fraction.items()
        },
        "policy": "lightgbm_native_nan_no_imputation_label_complete_rows",
    }
    return (
        ledger.loc[admitted].reset_index(drop=True),
        matrix.loc[admitted].reset_index(drop=True),
        labels.loc[admitted].reset_index(drop=True),
        evidence,
    )


def _admit_route(
    ledger: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    route: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if route["missing_value_policy"] == "lightgbm_native_nan":
        return _admit_native_missing(
            ledger,
            features,
            labels,
            min_per_feature_finite_fraction=float(
                route["min_per_feature_finite_fraction"]
            ),
        )
    return _admit_complete(ledger, features, labels)


def _precompute_outer_representations(
    representation_loader: SideRepresentationFeatureLoader,
    ledgers: Sequence[pd.DataFrame],
    generated_features: Sequence[str],
    *,
    batch_rows: int = 350_000,
) -> tuple[CachedRepresentationFeatureLoader, dict[str, Any], pd.DataFrame]:
    if batch_rows < 1:
        raise PackBOuterOOFRunnerError("representation batch_rows must be positive")
    union = (
        pd.concat(list(ledgers), ignore_index=True, copy=False)
        .drop_duplicates("candidate_id", keep="first")
        .sort_values(["__ts__", "__symbol__", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    generated = tuple(map(str, generated_features))
    values = pd.concat(
        [
            representation_loader(
                union.iloc[start : start + int(batch_rows)].reset_index(drop=True),
                generated,
            )
            for start in range(0, len(union), int(batch_rows))
        ],
        ignore_index=True,
        copy=False,
    )
    cache = CachedRepresentationFeatureLoader(union, values)
    return (
        cache,
        {
            "schema": "packb_outer_representation_union_cache_v1",
            "union_rows": int(len(union)),
            "generated_features": list(generated),
            "batch_rows": int(batch_rows),
            "batch_count": int((len(union) + int(batch_rows) - 1) // int(batch_rows)),
            "candidate_stream_sha256": hashlib.sha256(
                "\n".join(union["candidate_id"].astype(str)).encode("utf-8")
            ).hexdigest(),
            "values_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(values, index=True)
                .to_numpy(dtype=np.uint64, copy=False)
                .tobytes()
            ).hexdigest(),
            "outcome_columns_loaded": False,
        },
        union,
    )


def _fit_model(
    train_x: pd.DataFrame,
    train_labels: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> Any:
    model = _lgbm_regressor(params, seed=seed)
    model.fit(
        train_x,
        train_labels[TARGET_COLUMN],
        sample_weight=train_labels[WEIGHT_COLUMN],
    )
    return model


def _metrics(
    prediction: np.ndarray, ledger: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, Any]:
    return _economic_objective(
        prediction,
        labels[TARGET_COLUMN].to_numpy(dtype=np.float64),
        labels[WEIGHT_COLUMN].to_numpy(dtype=np.float64),
        labels[ECONOMIC_COLUMN].to_numpy(dtype=np.float64),
        timestamps=pd.to_datetime(ledger["__ts__"], utc=True).to_numpy(),
        symbols=ledger["__symbol__"].astype(str).to_numpy(),
    )


def _model_sha256(path: Path) -> str:
    return stage_manifest.sha256_file(path)


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    outer_population_root: Path = DEFAULT_OUTER_POPULATION,
    inner_population_root: Path = DEFAULT_POPULATION_ROOT,
    ae_root: Path = DEFAULT_AE_ROOT,
    promotion_path: Path = DEFAULT_PROMOTION,
    labels_dir: Path = DEFAULT_LABELS,
    feature_store: Path = DEFAULT_FEATURE_STORE,
    feature_inventory_path: Path = DEFAULT_FEATURE_INVENTORY,
    decisions_path: Path = DEFAULT_DECISIONS,
    train_max_rows: int = DEFAULT_TRAIN_MAX_ROWS,
    final_max_rows: int = DEFAULT_FINAL_MAX_ROWS,
    validation_max_rows: int | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise PackBOuterOOFRunnerError(f"refusing to overwrite output: {destination}")
    if (
        train_max_rows < 1
        or final_max_rows < 1
        or (validation_max_rows is not None and validation_max_rows < 1)
    ):
        raise PackBOuterOOFRunnerError("row caps must be positive")
    revision = _git_revision()
    (
        inner_manifest,
        source_hashes,
        fixed_calendar_sha256,
        _feature_binding,
    ) = _source_contracts(
        population_root=Path(inner_population_root),
        feature_inventory_path=Path(feature_inventory_path),
        decisions_path=Path(decisions_path),
    )
    outer_manifest, folds = _validate_outer_manifest(
        Path(outer_population_root), Path(decisions_path)
    )
    if outer_manifest.get("input", {}).get("causal_audit_sha256") != inner_manifest.get(
        "input", {}
    ).get("causal_audit_sha256"):
        raise PackBOuterOOFRunnerError("inner and outer label audits differ")
    promotion, routes = _load_promotion(
        Path(promotion_path), fixed_calendar_sha256=fixed_calendar_sha256
    )
    label_files = _canonical_label_files(Path(labels_dir), inner_manifest)
    ae_summary = _json(Path(ae_root) / "summary.json")
    ae_revision = str(ae_summary.get("source_revision") or "")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ae_revision, revision],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackBOuterOOFRunnerError("AE source is not an ancestor") from exc

    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    guard.preflight("packb_outer_oof:preflight")
    reports: dict[str, Any] = {}
    all_predictions: list[pd.DataFrame] = []
    try:
        for side_index, side in enumerate(SIDES):
            guard.checkpoint(f"packb_outer_oof:{side}:setup")
            contract, bundle, loader_hashes = _load_loader_contract(
                Path(ae_root) / side / "loader_evidence",
                source_revision=ae_revision,
            )
            raw_loader = make_fs_hpo_raw_feature_loader(
                feature_store_dir=Path(feature_store),
                feature_contract=contract,
                evidence_bundle=bundle,
                resource_guard=guard,
            )
            ae_manifest_path = (
                Path(ae_root) / side / "ae_gmm" / "side_stage_manifest.json"
            )
            ae_manifest = stage_manifest.validate_side_stage_manifest(
                ae_manifest_path,
                expected_side=side,
                expected_stage="ae_gmm",
                expected_source_hashes=source_hashes,
                expected_fixed_calendar_sha256=fixed_calendar_sha256,
            )
            state_path = (
                Path(ae_root) / side / "ae_gmm" / str(ae_manifest["artifact"]["path"])
            )
            state = _load_side_ae_state(
                state_path,
                expected_side=side,
                expected_sha256=str(ae_manifest["artifact"]["sha256"]),
                raw_features=contract["feature_columns"],
            )
            representation_loader = SideRepresentationFeatureLoader(
                raw_loader=raw_loader,
                raw_features=contract["feature_columns"],
                state=state,
                generated_features=_active_ae_gmm_columns(state),
            )
            labels = ExactLabelLoader(label_files, resource_guard=guard)
            route = routes[side]
            features = route["features"]
            side_root = stage / side
            fold_reports: list[dict[str, Any]] = []
            side_predictions: list[pd.DataFrame] = []
            final_sources: list[pd.DataFrame] = []
            fold_inputs: list[
                tuple[
                    Mapping[str, Any],
                    pd.DataFrame,
                    Path,
                    Mapping[str, Any],
                    pd.DataFrame,
                    Path,
                    Mapping[str, Any],
                ]
            ] = []
            for fold in folds:
                train, train_path, train_record = _load_bound_ledger(
                    Path(outer_population_root),
                    outer_manifest,
                    fold=fold,
                    side=side,
                    role="train",
                )
                validation, valid_path, valid_record = _load_bound_ledger(
                    Path(outer_population_root),
                    outer_manifest,
                    fold=fold,
                    side=side,
                    role="validation",
                )
                if validation_max_rows is not None:
                    validation = _bounded_beginning_middle_end_sample(
                        validation,
                        max_rows=int(validation_max_rows),
                        name=f"{side}_{fold['name']}_validation_diagnostic",
                    )
                train = _bounded_beginning_middle_end_sample(
                    train,
                    max_rows=int(train_max_rows),
                    name=f"{side}_{fold['name']}_train",
                )
                fold_inputs.append(
                    (
                        fold,
                        train,
                        train_path,
                        train_record,
                        validation,
                        valid_path,
                        valid_record,
                    )
                )
                final_sources.append(validation)

            last_train, _, _ = _load_bound_ledger(
                Path(outer_population_root),
                outer_manifest,
                fold=folds[-1],
                side=side,
                role="train",
            )
            final_sources.append(
                _bounded_beginning_middle_end_sample(
                    last_train,
                    max_rows=int(final_max_rows),
                    name=f"{side}_final_base",
                )
            )
            final_ledger = (
                pd.concat(final_sources, ignore_index=True)
                .drop_duplicates("candidate_id", keep="last")
                .reset_index(drop=True)
            )
            final_ledger = _bounded_beginning_middle_end_sample(
                final_ledger,
                max_rows=int(final_max_rows),
                name=f"{side}_final_refit",
            )
            generated = [
                feature for feature in features if feature.startswith(("dae_", "gmm_"))
            ]
            representation_features = (
                list(features) if route["loader_kind"] == "ae_gmm_only" else generated
            )
            cache_ledgers = [
                item
                for fold_input in fold_inputs
                for item in (fold_input[1], fold_input[4])
            ]
            cache_ledgers.append(final_ledger)
            guard.checkpoint(f"packb_outer_oof:{side}:before_representation_union")
            cached_representation, representation_cache_evidence, feature_union = (
                _precompute_outer_representations(
                    representation_loader,
                    cache_ledgers,
                    representation_features,
                )
            )
            guard.checkpoint(f"packb_outer_oof:{side}:representation_union_complete")
            if route["loader_kind"] == "historical_candidate_static_ae_gmm":
                label_schema = _label_schema(label_files, side=side)
                candidate = [
                    feature
                    for feature in features
                    if feature in label_schema
                    and not feature.startswith(("dae_", "gmm_"))
                ]
                composite_loader = HistoricalCompositeFeatureLoader(
                    side=side,
                    all_features=features,
                    candidate_features=candidate,
                    candidate_loader=ExactCandidateFeatureLoader(
                        label_files,
                        available=candidate,
                        resource_guard=guard,
                    ),
                    representation_loader=cached_representation,
                    generated_features=_active_ae_gmm_columns(state),
                    feature_store=Path(feature_store),
                    resource_guard=guard,
                )
                composite_values = composite_loader(feature_union, features)
                feature_loader = CachedRepresentationFeatureLoader(
                    feature_union,
                    composite_values,
                )
                representation_cache_evidence["full_composite_cached"] = True
                representation_cache_evidence["full_composite_values_sha256"] = (
                    hashlib.sha256(
                        pd.util.hash_pandas_object(composite_values, index=True)
                        .to_numpy(dtype=np.uint64, copy=False)
                        .tobytes()
                    ).hexdigest()
                )
            else:
                feature_loader = cached_representation
                representation_cache_evidence["full_composite_cached"] = True

            admission_route = dict(route)
            if validation_max_rows is not None:
                admission_route["min_per_feature_finite_fraction"] = 0.0

            for fold_index, (
                fold,
                train,
                train_path,
                train_record,
                validation,
                valid_path,
                valid_record,
            ) in enumerate(fold_inputs):
                guard.checkpoint(f"packb_outer_oof:{side}:{fold['name']}:load")
                train_ledger, train_x, train_labels, train_coverage = _admit_route(
                    train,
                    feature_loader(train, features),
                    labels.load(train),
                    admission_route,
                )
                valid_ledger, valid_x, valid_labels, valid_coverage = _admit_route(
                    validation,
                    feature_loader(validation, features),
                    labels.load(validation),
                    admission_route,
                )
                seed = 20260724 + side_index * 1_000 + fold_index
                model = _fit_model(train_x, train_labels, route["params"], seed=seed)
                prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
                metrics = _metrics(prediction, valid_ledger, valid_labels)
                fold_root = side_root / "folds" / str(fold["name"])
                fold_root.mkdir(parents=True, exist_ok=True)
                model_path = fold_root / "model.txt"
                model.booster_.save_model(str(model_path))
                scored = valid_ledger.loc[
                    :, ["candidate_id", "side_name", "__ts__", "__symbol__"]
                ].copy()
                scored["outer_fold"] = str(fold["name"])
                scored["prediction"] = prediction
                scored[TARGET_COLUMN] = valid_labels[TARGET_COLUMN].to_numpy()
                scored[WEIGHT_COLUMN] = valid_labels[WEIGHT_COLUMN].to_numpy()
                scored[ECONOMIC_COLUMN] = valid_labels[ECONOMIC_COLUMN].to_numpy()
                scored["prediction_source"] = "outer_oof_fold_model"
                scored.to_parquet(fold_root / "predictions.parquet", index=False)
                fold_report = {
                    "fold": str(fold["name"]),
                    "validation_start_utc": fold["start"].isoformat(),
                    "validation_end_utc": fold["end"].isoformat(),
                    "train_ledger": {
                        "path": str(train_path),
                        "sha256": train_record["sha256"],
                        "authorized_rows": int(train_record["rows"]),
                        "sampled_rows": int(len(train)),
                    },
                    "validation_ledger": {
                        "path": str(valid_path),
                        "sha256": valid_record["sha256"],
                        "authorized_rows": int(valid_record["rows"]),
                        "sampled_rows": int(len(validation)),
                    },
                    "train_coverage": train_coverage,
                    "validation_coverage": valid_coverage,
                    "model_sha256": _model_sha256(model_path),
                    "metrics": metrics,
                    "final_refit_prediction_used": False,
                }
                _atomic_json(fold_root / "manifest.json", fold_report)
                fold_reports.append(fold_report)
                side_predictions.append(scored)
                all_predictions.append(scored)
                del model, prediction, valid_x, train_x, train_labels, valid_labels
                del validation, train
                _release_memory()
                guard.checkpoint(f"packb_outer_oof:{side}:{fold['name']}:complete")

            final_ledger, final_x, final_labels, final_coverage = _admit_route(
                final_ledger,
                feature_loader(final_ledger, features),
                labels.load(final_ledger),
                admission_route,
            )
            final_model = _fit_model(
                final_x,
                final_labels,
                route["params"],
                seed=20260724 + side_index * 1_000 + 999,
            )
            final_root = side_root / "final_refit"
            final_root.mkdir(parents=True, exist_ok=True)
            final_model_path = final_root / "model.txt"
            final_model.booster_.save_model(str(final_model_path))
            side_oof = pd.concat(side_predictions, ignore_index=True)
            side_oof.to_parquet(side_root / "oof_predictions.parquet", index=False)
            aggregate_metrics = _metrics(
                side_oof["prediction"].to_numpy(dtype=np.float64),
                side_oof.rename(columns={"outer_fold": "__unused__"}),
                side_oof,
            )
            report = {
                "side": side,
                "model_side_scope": "per_side",
                "features": list(features),
                "feature_count": len(features),
                "selected_trial_id": route["trial_id"],
                "parameters": route["params"],
                "loader_kind": route["loader_kind"],
                "missing_value_policy": route["missing_value_policy"],
                "min_per_feature_finite_fraction": route[
                    "min_per_feature_finite_fraction"
                ],
                "diagnostic_coverage_floor_bypassed": validation_max_rows is not None,
                "representation_union_cache": representation_cache_evidence,
                "folds": fold_reports,
                "aggregate_oof_metrics": aggregate_metrics,
                "oof_rows": int(len(side_oof)),
                "final_refit": {
                    "model_sha256": _model_sha256(final_model_path),
                    "coverage": final_coverage,
                    "training_rows": int(len(final_ledger)),
                    "excluded_from_oof_metrics": True,
                    "predictions_persisted": False,
                },
                "loader_evidence_hashes": loader_hashes,
                "ae_state_sha256": stage_manifest.sha256_file(state_path),
                "feature_contract_sha256": stage_manifest.sha256_file(
                    route["feature_contract_path"]
                ),
                "hpo_parameters_sha256": stage_manifest.sha256_file(
                    route["parameter_path"]
                ),
            }
            _atomic_json(side_root / "manifest.json", report)
            reports[side] = report
            del final_model, final_x, final_labels, final_ledger, last_train
            del labels, feature_loader, cached_representation, feature_union, raw_loader
            if route["loader_kind"] == "historical_candidate_static_ae_gmm":
                del composite_loader, composite_values
            del representation_loader, state, contract, bundle
            _release_memory()
            guard.checkpoint(f"packb_outer_oof:{side}:released")

        combined = pd.concat(all_predictions, ignore_index=True)
        if combined["candidate_id"].astype(str).duplicated().any():
            raise PackBOuterOOFRunnerError("combined OOF stream has duplicate IDs")
        combined.to_parquet(stage / "oof_predictions.parquet", index=False)
        guard.checkpoint("packb_outer_oof:publication")
        summary = {
            "schema": SCHEMA,
            "status": "COMPLETE_STRICT_SIDE_LOCAL_OUTER_OOF_AND_FINAL_REFITS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision,
            "model_side_scope": "per_side",
            "shared_fitted_model": False,
            "sides": reports,
            "oof_rows": int(len(combined)),
            "outer_population_manifest_sha256": stage_manifest.sha256_file(
                Path(outer_population_root) / "manifest.json"
            ),
            "promotion_contract_sha256": stage_manifest.sha256_file(
                Path(promotion_path)
            ),
            "fixed_calendar_sha256": fixed_calendar_sha256,
            "promotion_role": promotion["selection_evidence_role"],
            "final_refit_predictions_used_in_oof": False,
            "validation_sampling": (
                "full_authorized_outer_rows"
                if validation_max_rows is None
                else "bounded_diagnostic_only_not_promotion_evidence"
            ),
        }
        _atomic_json(stage / "summary.json", summary)
        os.replace(stage, destination)
        return summary
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--outer-population-root", type=Path, default=DEFAULT_OUTER_POPULATION
    )
    parser.add_argument(
        "--inner-population-root", type=Path, default=DEFAULT_POPULATION_ROOT
    )
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--promotion", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument(
        "--feature-inventory", type=Path, default=DEFAULT_FEATURE_INVENTORY
    )
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--train-max-rows", type=int, default=DEFAULT_TRAIN_MAX_ROWS)
    parser.add_argument("--final-max-rows", type=int, default=DEFAULT_FINAL_MAX_ROWS)
    parser.add_argument("--validation-max-rows", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(
            output_dir=args.output_dir,
            outer_population_root=args.outer_population_root,
            inner_population_root=args.inner_population_root,
            ae_root=args.ae_root,
            promotion_path=args.promotion,
            labels_dir=args.labels_dir,
            feature_store=args.feature_store,
            feature_inventory_path=args.feature_inventory,
            decisions_path=args.decisions,
            train_max_rows=args.train_max_rows,
            final_max_rows=args.final_max_rows,
            validation_max_rows=args.validation_max_rows,
        )
    except (PackBOuterOOFRunnerError, ValueError, FileExistsError) as exc:
        print(
            json.dumps({"status": "BLOCKED_PRECONDITION_FAILED", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
