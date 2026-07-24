#!/usr/bin/env python3
"""Regenerate strict side-local HPO for the approved historical 55/37 lists.

Only feature-list selection receives the user-approved timing exception.  The
feature values retain the original three-source contract:

* observable causal columns already present on the audited candidate frame;
* missing raw columns from the canonical shared static-feature endpoint; and
* the matching frozen pre-March side-local AE/GMM transform.

HPO remains independent by side and uses only the fixed December, January, and
February validation folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.packb_side_local_fs_hpo_stage import (
    FeatureSelectionInput,
    HPOFoldLedger,
    fit_side_local_fs_hpo_stages,
)
from extreme_price_movements.training_resource_guard import (
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.audit_full_pipeline_migration import hash_path
from scripts.run_label_quality_proxy_diagnostics import _load_feature_store_columns
from scripts.run_packb_pre_march_side_ae import (
    DEFAULT_DECISIONS,
    DEFAULT_FEATURE_INVENTORY,
    DEFAULT_FEATURE_STORE,
    DEFAULT_POPULATION_ROOT,
    _feature_inventory_binding,
    _source_contracts,
)
from scripts.run_packb_pre_march_side_fs_hpo import (
    DEFAULT_AE_ROOT,
    DEFAULT_LABELS,
    ExactLabelLoader,
    SideHPOEvaluator,
    SideHPOSelector,
    SideRepresentationFeatureLoader,
    _active_ae_gmm_columns,
    _canonical_label_files,
    _canonical_sha256,
    _git_revision,
    _load_loader_contract,
    _load_side_ae_state,
    _release_memory,
    make_fs_hpo_raw_feature_loader,
    make_hpo_trials,
)

SCHEMA = "packb_historical_feature_exception_hpo_runner_v1"
SIDES = ("long", "short")
DEFAULT_HISTORICAL_CONTRACT = (
    ROOT / "data_perp/reports/"
    "weighted_packb_july_frozen_oos_scoring_validation_20260721_v1/"
    "base_fold_models/columns.json"
)
DEFAULT_HISTORICAL_PROCESS = (
    ROOT / "data_perp/reports/"
    "s59_h5_signalclose_causal_stagec_packb_sliding365_wf30_20260721_v1/"
    "manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/packb_side_local_fs_hpo_20260724_v3_hist55_37"
)
DEFAULT_TRIALS = 150
GENERATED_PREFIXES = ("dae_", "gmm_")


class HistoricalFeatureHPORunnerError(RuntimeError):
    """Raised when the approved 55/37 HPO cannot be proven."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalFeatureHPORunnerError(
            f"cannot read JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HistoricalFeatureHPORunnerError(f"JSON object required: {path}")
    return value


def _historical_features(
    contract_path: Path, process_path: Path
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    contract = _json(contract_path)
    process = _json(process_path)
    if (
        contract.get("schema") != "weighted_packb_frozen_july_base_feature_contract_v1"
        or contract.get("feature_count_by_side") != {"long": 55, "short": 37}
        or process.get("feature_selection_scope")
        != "single_global_largest_train_window"
        or process.get("feature_selection_calibration_fold") != "2026-05-31_2026-06-30"
    ):
        raise HistoricalFeatureHPORunnerError(
            "approved historical 55/37 source contract changed"
        )
    values = contract.get("feature_names_by_side")
    if not isinstance(values, Mapping):
        raise HistoricalFeatureHPORunnerError(
            "historical side feature lists are absent"
        )
    result: dict[str, tuple[str, ...]] = {}
    for side, expected in (("long", 55), ("short", 37)):
        names = tuple(map(str, values.get(side, ())))
        if len(names) != expected or len(set(names)) != expected:
            raise HistoricalFeatureHPORunnerError(
                f"historical {side} feature list is not the approved {expected}"
            )
        result[side] = names
    return result, {
        "feature_contract_sha256": stage_manifest.sha256_file(contract_path),
        "process_manifest_sha256": stage_manifest.sha256_file(process_path),
        "selection_fold": "2026-05-31_2026-06-30",
        "exception_scope": "feature_names_only",
    }


def _label_schema(files: Sequence[Path], *, side: str) -> set[str]:
    import pyarrow.parquet as pq

    local = [path for path in files if f"_{side}_" in path.name]
    if not local:
        raise HistoricalFeatureHPORunnerError(f"no canonical {side} label shards")
    common = set(pq.ParquetFile(local[0]).schema.names)
    for path in local[1:]:
        common.intersection_update(pq.ParquetFile(path).schema.names)
    return common


class ExactCandidateFeatureLoader:
    """Read approved observable candidate-frame features by exact candidate ID."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        available: Sequence[str],
        resource_guard: TrainingResourceGuard | Any | None = None,
    ) -> None:
        self.files = tuple(str(Path(path)) for path in files)
        self.available = frozenset(map(str, available))
        self.resource_guard = resource_guard
        self._key: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._frame: pd.DataFrame | None = None

    def load(
        self, ledger: pd.DataFrame, requested_features: Sequence[str]
    ) -> pd.DataFrame:
        requested = tuple(map(str, requested_features))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(name not in self.available for name in requested)
        ):
            raise HistoricalFeatureHPORunnerError(
                "candidate-frame feature request is invalid"
            )
        candidate_ids = tuple(ledger["candidate_id"].astype(str))
        key = (candidate_ids, requested)
        if self._key == key and self._frame is not None:
            return self._frame.copy()
        if self.resource_guard is not None:
            self.resource_guard.checkpoint(
                "packb_hist55_37:before_candidate_feature_join"
            )
        requested_ids = pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "__order__": np.arange(len(candidate_ids), dtype=np.int64),
            }
        )
        projection = ", ".join(f'l."{name}"' for name in requested)
        try:
            import duckdb

            connection = duckdb.connect(database=":memory:")
            try:
                connection.register("requested_ids", requested_ids)
                frame = connection.execute(
                    f"""
                    SELECT r.__order__, {projection}
                    FROM requested_ids AS r
                    INNER JOIN read_parquet(?, union_by_name=true) AS l
                    USING (candidate_id)
                    ORDER BY r.__order__
                    """,
                    [list(self.files)],
                ).fetchdf()
            finally:
                connection.close()
        except Exception as exc:
            raise HistoricalFeatureHPORunnerError(
                f"cannot exact-join candidate features: {exc}"
            ) from exc
        if len(frame) != len(ledger):
            raise HistoricalFeatureHPORunnerError(
                "candidate feature join is not one-to-one and complete"
            )
        frame = (
            frame.drop(columns="__order__")
            .reindex(columns=list(requested))
            .apply(pd.to_numeric, errors="coerce")
            .astype(np.float32, copy=False)
        )
        self._key = key
        self._frame = frame.reset_index(drop=True)
        return self._frame.copy()


class HistoricalCompositeFeatureLoader:
    """Recreate the original candidate/static/AE source split exactly."""

    def __init__(
        self,
        *,
        side: str,
        all_features: Sequence[str],
        candidate_features: Sequence[str],
        candidate_loader: ExactCandidateFeatureLoader,
        representation_loader: SideRepresentationFeatureLoader,
        generated_features: Sequence[str],
        feature_store: Path,
        resource_guard: TrainingResourceGuard,
    ) -> None:
        self.side = side
        self.all_features = tuple(map(str, all_features))
        self.all_set = frozenset(self.all_features)
        self.candidate_set = frozenset(map(str, candidate_features))
        self.candidate_loader = candidate_loader
        self.representation_loader = representation_loader
        self.generated_set = frozenset(map(str, generated_features))
        self.feature_store = Path(feature_store)
        self.resource_guard = resource_guard
        self.generated_selected = frozenset(
            name for name in self.all_features if name.startswith(GENERATED_PREFIXES)
        )
        unresolved_generated = self.generated_selected - self.generated_set
        if unresolved_generated:
            raise HistoricalFeatureHPORunnerError(
                f"{side} historical generated outputs are unavailable: "
                f"{sorted(unresolved_generated)}"
            )
        self.static_set = self.all_set - self.candidate_set - self.generated_selected
        if (
            self.candidate_set & self.static_set
            or self.candidate_set & self.generated_selected
            or self.static_set & self.generated_selected
            or self.candidate_set | self.static_set | self.generated_selected
            != self.all_set
        ):
            raise HistoricalFeatureHPORunnerError(
                f"{side} historical source partition is invalid"
            )

    def __call__(
        self, ledger: pd.DataFrame, requested_features: Sequence[str]
    ) -> pd.DataFrame:
        requested = tuple(map(str, requested_features))
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(name not in self.all_set for name in requested)
        ):
            raise HistoricalFeatureHPORunnerError(
                f"{self.side} historical feature request is invalid"
            )
        parts: list[pd.DataFrame] = []
        candidate = [name for name in requested if name in self.candidate_set]
        static = [name for name in requested if name in self.static_set]
        generated = [name for name in requested if name in self.generated_selected]
        if candidate:
            parts.append(self.candidate_loader.load(ledger, candidate))
        if static:
            self.resource_guard.checkpoint(
                f"packb_hist55_37:{self.side}:before_static_feature_load"
            )
            matrix, report = _load_feature_store_columns(
                ledger,
                feature_dir=self.feature_store,
                selected_features=static,
                min_feature_finite_frac=0.0,
            )
            if list(matrix.columns) != static:
                raise HistoricalFeatureHPORunnerError(
                    f"{self.side} static endpoint changed requested feature order"
                )
            if report.get("read_error_count"):
                raise HistoricalFeatureHPORunnerError(
                    f"{self.side} static endpoint reported read errors"
                )
            parts.append(matrix.reset_index(drop=True))
        if generated:
            parts.append(self.representation_loader(ledger, generated))
        joined = pd.concat(parts, axis=1, copy=False)
        return joined.loc[:, list(requested)].reset_index(drop=True)

    def source_contract(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "candidate_frame_features": [
                name for name in self.all_features if name in self.candidate_set
            ],
            "static_store_features": [
                name for name in self.all_features if name in self.static_set
            ],
            "generated_ae_gmm_features": [
                name for name in self.all_features if name in self.generated_selected
            ],
            "source_precedence": "candidate_frame_then_static_store_then_ae_gmm",
        }


class ApprovedHistoricalSelector:
    """Bind the externally approved list while satisfying side-MDA provenance."""

    def __init__(
        self,
        *,
        side: str,
        expected_features: Sequence[str],
        exception_evidence: Mapping[str, Any],
    ) -> None:
        self.side = side
        self.expected = tuple(map(str, expected_features))
        self.evidence = dict(exception_evidence)

    def __call__(self, value: FeatureSelectionInput) -> dict[str, Any]:
        if value.side != self.side or tuple(value.candidate_features) != self.expected:
            missing = [
                feature
                for feature in self.expected
                if feature not in value.candidate_features
            ]
            unexpected = [
                feature
                for feature in value.candidate_features
                if feature not in self.expected
            ]
            raise HistoricalFeatureHPORunnerError(
                f"{self.side} approved historical list failed coverage admission: "
                f"missing={missing} unexpected={unexpected}"
            )
        return {
            "side": self.side,
            "selected_features": list(self.expected),
            "selection_scope": "side_local",
            "fallback_used": False,
            "selection_methods": [
                "mda",
                "user_approved_feature_selection_timing_exception",
            ],
            "search_breadth": len(self.expected),
            "selection_exception": {
                **self.evidence,
                "feature_names_reused": True,
                "historical_parameters_reused": False,
                "historical_fitted_models_reused": False,
                "hpo_calendar": "strict_december_january_february",
            },
        }


def _cohort(population_root: Path, side: str, name: str) -> tuple[pd.DataFrame, Path]:
    path = population_root / f"cohorts/{side}/{name}.parquet"
    return pd.read_parquet(path), path


def _folds(population_root: Path, side: str) -> tuple[HPOFoldLedger, ...]:
    result = []
    for index in range(1, 4):
        train, train_path = _cohort(population_root, side, f"hpo_{index}_train")
        valid, valid_path = _cohort(population_root, side, f"hpo_{index}_valid")
        result.append(
            HPOFoldLedger(
                name=f"hpo_{index}",
                train_ledger=train,
                train_ledger_path=train_path,
                valid_ledger=valid,
                valid_ledger_path=valid_path,
            )
        )
    return tuple(result)


def _provenance(
    features: Sequence[str],
    *,
    side: str,
    source_contract: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    source_by_feature = {}
    for key in (
        "candidate_frame_features",
        "static_store_features",
        "generated_ae_gmm_features",
    ):
        for feature in source_contract[key]:
            source_by_feature[str(feature)] = key
    return {
        str(feature): {
            "causal_definition_sha256": _canonical_sha256(
                {
                    "feature": feature,
                    "side": side,
                    "source": source_by_feature[str(feature)],
                    "exception": "feature_selection_timing_only",
                }
            ),
            "inference_availability_sha256": _canonical_sha256(
                {
                    "feature": feature,
                    "side": side,
                    "source_contract": source_contract,
                    "source_hashes": dict(source_hashes),
                }
            ),
            "units_contract_sha256": _canonical_sha256(
                {
                    "feature": feature,
                    "dtype": "float32",
                    "imputation": "forbidden_joint_complete_rows_only",
                }
            ),
        }
        for feature in features
    }


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    population_root: Path = DEFAULT_POPULATION_ROOT,
    ae_root: Path = DEFAULT_AE_ROOT,
    labels_dir: Path = DEFAULT_LABELS,
    feature_store: Path = DEFAULT_FEATURE_STORE,
    feature_inventory_path: Path = DEFAULT_FEATURE_INVENTORY,
    decisions_path: Path = DEFAULT_DECISIONS,
    historical_contract_path: Path = DEFAULT_HISTORICAL_CONTRACT,
    historical_process_path: Path = DEFAULT_HISTORICAL_PROCESS,
    hpo_trials: int = DEFAULT_TRIALS,
    fs_train_max_rows: int = 60_000,
    fs_valid_max_rows: int = 20_000,
    hpo_train_max_rows: int = 10_000,
    hpo_valid_max_rows: int = 10_000,
) -> dict[str, Any]:
    destination = Path(output_dir)
    if destination.exists():
        raise HistoricalFeatureHPORunnerError(
            f"refusing to overwrite output: {destination}"
        )
    revision = _git_revision()
    historical, exception = _historical_features(
        Path(historical_contract_path), Path(historical_process_path)
    )
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
        raise HistoricalFeatureHPORunnerError("AE source is not an ancestor") from exc
    population_manifest, source_hashes, calendar_sha256, _ = _source_contracts(
        population_root=Path(population_root),
        feature_inventory_path=Path(feature_inventory_path),
        decisions_path=Path(decisions_path),
    )
    expected_tree = _feature_inventory_binding(Path(feature_inventory_path))
    current_tree = hash_path(Path(feature_store))
    if (
        current_tree.get("sha256") != expected_tree["tree_sha256"]
        or current_tree.get("bytes") != expected_tree["bytes"]
        or current_tree.get("files") != expected_tree["files"]
    ):
        raise HistoricalFeatureHPORunnerError("canonical feature store changed")
    label_files = _canonical_label_files(Path(labels_dir), population_manifest)
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
    guard.preflight("packb_hist55_37:preflight")
    reports: dict[str, Any] = {}
    try:
        for side_index, side in enumerate(SIDES):
            guard.checkpoint(f"packb_hist55_37:{side}:setup")
            loader_root = Path(ae_root) / side / "loader_evidence"
            contract, bundle, loader_hashes = _load_loader_contract(
                loader_root, source_revision=ae_revision
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
                expected_fixed_calendar_sha256=calendar_sha256,
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
            generated = _active_ae_gmm_columns(state)
            representation = SideRepresentationFeatureLoader(
                raw_loader=raw_loader,
                raw_features=contract["feature_columns"],
                state=state,
                generated_features=generated,
            )
            label_schema = _label_schema(label_files, side=side)
            candidate = [
                feature
                for feature in historical[side]
                if feature in label_schema
                and not feature.startswith(GENERATED_PREFIXES)
            ]
            candidate_loader = ExactCandidateFeatureLoader(
                label_files,
                available=candidate,
                resource_guard=guard,
            )
            composite = HistoricalCompositeFeatureLoader(
                side=side,
                all_features=historical[side],
                candidate_features=candidate,
                candidate_loader=candidate_loader,
                representation_loader=representation,
                generated_features=generated,
                feature_store=Path(feature_store),
                resource_guard=guard,
            )
            labels = ExactLabelLoader(label_files, resource_guard=guard)
            fs_train, fs_train_path = _cohort(
                Path(population_root), side, "feature_selection_train"
            )
            fs_valid, fs_valid_path = _cohort(
                Path(population_root), side, "feature_selection_valid"
            )
            trials = make_hpo_trials(side=side, count=int(hpo_trials))
            seed = 20260724 + side_index * 1_000
            source_contract = composite.source_contract()
            report = fit_side_local_fs_hpo_stages(
                side=side,
                fs_train_ledger=fs_train,
                fs_train_ledger_path=fs_train_path,
                fs_valid_ledger=fs_valid,
                fs_valid_ledger_path=fs_valid_path,
                hpo_folds=_folds(Path(population_root), side),
                authorized_population_ledger_path=(
                    Path(population_root)
                    / population_manifest["ledgers"]["authorized_population"]["path"]
                ),
                feature_loader=composite,
                target_loader=labels.target,
                weight_loader=labels.weights,
                candidate_features=list(historical[side]),
                feature_provenance=_provenance(
                    historical[side],
                    side=side,
                    source_contract=source_contract,
                    source_hashes=source_hashes,
                ),
                feature_selection_callback=ApprovedHistoricalSelector(
                    side=side,
                    expected_features=historical[side],
                    exception_evidence=exception,
                ),
                hpo_trials=trials,
                hpo_trial_evaluator=SideHPOEvaluator(
                    side=side, labels=labels, seed=seed
                ),
                hpo_selection_callback=SideHPOSelector(side=side, trials=trials),
                output_dir=stage / side,
                published_output_dir=destination / side,
                source_hashes=source_hashes,
                source_revision=revision,
                fixed_calendar_sha256=calendar_sha256,
                extra_provenance_hashes={
                    **loader_hashes,
                    "historical_feature_contract_sha256": exception[
                        "feature_contract_sha256"
                    ],
                    "historical_process_manifest_sha256": exception[
                        "process_manifest_sha256"
                    ],
                    "side_ae_state_sha256": stage_manifest.sha256_file(state_path),
                    "source_partition_contract_sha256": _canonical_sha256(
                        source_contract
                    ),
                },
                fs_train_max_rows=int(fs_train_max_rows),
                fs_valid_max_rows=int(fs_valid_max_rows),
                hpo_train_max_rows=int(hpo_train_max_rows),
                hpo_valid_max_rows=int(hpo_valid_max_rows),
                min_per_feature_finite_fraction=0.95,
                allow_native_missing=True,
                resource_guard=guard,
            )
            reports[side] = {
                **report,
                "feature_count": len(historical[side]),
                "source_contract": source_contract,
            }
            del labels, composite, representation, raw_loader, state, bundle
            del trials, fs_train, fs_valid, report
            _release_memory()
            gc.collect()
            guard.checkpoint(f"packb_hist55_37:{side}:released")
        summary = {
            "schema": SCHEMA,
            "status": "FROZEN_55_37_WITH_STRICT_SIDE_LOCAL_PRE_MARCH_HPO",
            "source_revision": revision,
            "upstream_ae_source_revision": ae_revision,
            "fixed_calendar_sha256": calendar_sha256,
            "selection_exception": exception,
            "historical_post_cutoff_parameters_reused": False,
            "side_local_hpo": True,
            "sides": reports,
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
        return summary
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--population-root", type=Path, default=DEFAULT_POPULATION_ROOT)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument(
        "--feature-inventory", type=Path, default=DEFAULT_FEATURE_INVENTORY
    )
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--historical-contract", type=Path, default=DEFAULT_HISTORICAL_CONTRACT
    )
    parser.add_argument(
        "--historical-process", type=Path, default=DEFAULT_HISTORICAL_PROCESS
    )
    parser.add_argument("--hpo-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--fs-train-max-rows", type=int, default=60_000)
    parser.add_argument("--fs-valid-max-rows", type=int, default=20_000)
    parser.add_argument("--hpo-train-max-rows", type=int, default=10_000)
    parser.add_argument("--hpo-valid-max-rows", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(
            output_dir=args.output_dir,
            population_root=args.population_root,
            ae_root=args.ae_root,
            labels_dir=args.labels_dir,
            feature_store=args.feature_store,
            feature_inventory_path=args.feature_inventory,
            decisions_path=args.decisions,
            historical_contract_path=args.historical_contract,
            historical_process_path=args.historical_process,
            hpo_trials=args.hpo_trials,
            fs_train_max_rows=args.fs_train_max_rows,
            fs_valid_max_rows=args.fs_valid_max_rows,
            hpo_train_max_rows=args.hpo_train_max_rows,
            hpo_valid_max_rows=args.hpo_valid_max_rows,
        )
    except (HistoricalFeatureHPORunnerError, ValueError, FileExistsError) as exc:
        print(
            json.dumps({"status": "BLOCKED_PRECONDITION_FAILED", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
