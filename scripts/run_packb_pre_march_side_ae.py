#!/usr/bin/env python3
"""Freeze and fit the two strict pre-March Pack-B side-local AE/GMM states."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.packb_side_local_ae_stage import (
    REQUIRED_LEDGER_COLUMNS,
    fit_side_local_ae_gmm_stage,
)
from extreme_price_movements.packb_static_point_feature_loader import (
    build_fresh_causal_feature_contract,
    make_packb_static_feature_loader,
    write_loader_evidence_bundle,
)
from extreme_price_movements.training_resource_guard import (
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.audit_full_pipeline_migration import hash_path
from scripts.prepare_packb_pre_march_side_contracts import parse_locked_dec09

DEFAULT_POPULATION_ROOT = (
    ROOT / "data_perp/artifacts/packb_pre_march_population_20260724_v1"
)
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_FEATURE_INVENTORY = (
    ROOT / "docs/pipeline_roadmap/20260724/r0/migration_inventory.json"
)
DEFAULT_DECISIONS = ROOT / "config/full_pipeline_decisions_20260724.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_COVERAGE_SAMPLE_ROWS = 20_000
DEFAULT_MAX_FEATURE_COLUMNS = 256
DEFAULT_MAX_PROFILE_FEATURE_COLUMNS = 512


class PackBSideAERunnerError(RuntimeError):
    """Raised when production side-local AE evidence cannot be proven."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBSideAERunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackBSideAERunnerError(f"JSON object required: {path}")
    return value


def _git_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackBSideAERunnerError("cannot resolve the source revision") from exc
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PackBSideAERunnerError("source revision is not a full Git SHA")
    if dirty:
        raise PackBSideAERunnerError(
            "production AE fitting requires a clean tracked source revision"
        )
    return revision


def _feature_inventory_binding(path: Path) -> dict[str, Any]:
    payload = _json(path)
    inventory = payload.get("inventory")
    items = inventory.get("items") if isinstance(inventory, Mapping) else None
    matches = [
        item
        for item in (items or [])
        if isinstance(item, Mapping) and item.get("id") == "canonical_feature_store"
    ]
    if len(matches) != 1:
        raise PackBSideAERunnerError(
            "feature inventory must contain exactly one canonical_feature_store"
        )
    item = dict(matches[0])
    digest = str(item.get("sha256") or "").lower()
    if len(digest) != 64:
        raise PackBSideAERunnerError("feature-store inventory hash is invalid")
    return {
        "tree_sha256": digest,
        "bytes": int(item.get("bytes", -1)),
        "files": int(item.get("files", -1)),
        "directories": int(item.get("directories", -1)),
        "evidence_sha256": stage_manifest.sha256_file(path),
        "evidence_path": str(path),
    }


def _revalidate_feature_store(
    feature_store: Path, inventory_binding: Mapping[str, Any]
) -> dict[str, Any]:
    current = hash_path(feature_store)
    for current_key, expected_key in (
        ("sha256", "tree_sha256"),
        ("bytes", "bytes"),
        ("files", "files"),
        ("directories", "directories"),
    ):
        if current.get(current_key) != inventory_binding.get(expected_key):
            raise PackBSideAERunnerError(
                "canonical feature store changed since the R0 inventory: "
                f"{current_key}={current.get(current_key)!r}, "
                f"expected={inventory_binding.get(expected_key)!r}"
            )
    return current


def _source_contracts(
    *,
    population_root: Path,
    feature_inventory_path: Path,
    decisions_path: Path,
) -> tuple[dict[str, Any], dict[str, str], str, dict[str, Any]]:
    population_manifest_path = population_root / "manifest.json"
    population_manifest = _json(population_manifest_path)
    if population_manifest.get("status") != "MATERIALIZED_IMMUTABLE":
        raise PackBSideAERunnerError("pre-March population is not immutable")
    population_preflight = population_manifest.get("population_preflight")
    if not isinstance(population_preflight, Mapping):
        raise PackBSideAERunnerError("population manifest has no preflight evidence")
    population_ledger = population_root / str(
        population_manifest["ledgers"]["authorized_population"]["path"]
    )
    if not population_ledger.is_file():
        raise PackBSideAERunnerError("authorized population ledger is missing")
    feature_binding = _feature_inventory_binding(feature_inventory_path)
    dec09 = parse_locked_dec09(decisions_path)
    label_inventory = population_preflight.get("label_inventory")
    if not isinstance(label_inventory, Mapping):
        raise PackBSideAERunnerError("population preflight has no label inventory")
    source_hashes = {
        "dec09_decisions_sha256": str(dec09["sha256"]),
        "canonical_shard_inventory_sha256": str(
            label_inventory["canonical_shard_inventory_sha256"]
        ),
        "causal_audit_sha256": str(label_inventory["causal_audit_sha256"]),
        "population_preflight_sha256": stage_manifest.canonical_json_sha256(
            population_preflight
        ),
        "authorized_population_ledger_sha256": stage_manifest.sha256_file(
            population_ledger
        ),
        "feature_store_inventory_sha256": str(feature_binding["tree_sha256"]),
        "feature_store_inventory_evidence_sha256": str(
            feature_binding["evidence_sha256"]
        ),
    }
    fixed_calendar_sha256 = stage_manifest.canonical_json_sha256(dec09["calendar"])
    return (
        population_manifest,
        source_hashes,
        fixed_calendar_sha256,
        feature_binding,
    )


def _coverage_segments(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    signal = pd.to_datetime(ledger["__ts__"], utc=True, errors="raise")
    intervals = {
        "beginning": (
            pd.Timestamp("2025-01-01T00:00:00Z"),
            pd.Timestamp("2025-02-01T00:00:00Z"),
        ),
        "middle": (
            pd.Timestamp("2025-06-01T00:00:00Z"),
            pd.Timestamp("2025-07-01T00:00:00Z"),
        ),
        "end": (
            pd.Timestamp("2025-10-01T00:00:00Z"),
            pd.Timestamp("2025-11-01T00:00:00Z"),
        ),
    }
    result: dict[str, pd.DataFrame] = {}
    for name, (start, end) in intervals.items():
        segment = ledger.loc[
            signal.ge(start) & signal.lt(end),
            ["candidate_id", "__ts__", "__symbol__"],
        ].copy()
        if segment.empty:
            raise PackBSideAERunnerError(f"AE coverage segment {name!r} is empty")
        result[name] = segment
    return result


def _release_memory() -> None:
    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def run(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    population_root: Path = DEFAULT_POPULATION_ROOT,
    feature_store: Path = DEFAULT_FEATURE_STORE,
    feature_inventory_path: Path = DEFAULT_FEATURE_INVENTORY,
    decisions_path: Path = DEFAULT_DECISIONS,
    coverage_sample_rows: int = DEFAULT_COVERAGE_SAMPLE_ROWS,
    max_feature_columns: int = DEFAULT_MAX_FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Fit long then short, publishing only after both immutable stages pass."""

    destination = Path(output_dir)
    if destination.exists():
        raise PackBSideAERunnerError(
            f"refusing to overwrite production AE output: {destination}"
        )
    revision = _git_revision()
    population_manifest, source_hashes, calendar_sha256, feature_binding = (
        _source_contracts(
            population_root=Path(population_root),
            feature_inventory_path=Path(feature_inventory_path),
            decisions_path=Path(decisions_path),
        )
    )
    stage_root = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage_root.mkdir(parents=True, exist_ok=False)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage_root / "training_resource_telemetry.jsonl",
    )
    guard.preflight("packb_side_ae:feature_store_revalidation")
    current_feature_store = _revalidate_feature_store(
        Path(feature_store), feature_binding
    )
    side_reports: dict[str, Any] = {}
    try:
        for side in ("long", "short"):
            guard.checkpoint(f"packb_side_ae:{side}:before_ledger")
            cohort_path = Path(population_root) / f"cohorts/{side}/ae_reference.parquet"
            ledger = pd.read_parquet(cohort_path, columns=list(REQUIRED_LEDGER_COLUMNS))
            segments = _coverage_segments(ledger)
            identity = ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].copy()
            guard.checkpoint(f"packb_side_ae:{side}:before_contract")
            universe, coverage, contract = build_fresh_causal_feature_contract(
                identity,
                feature_store_dir=feature_store,
                schema_evidence_path=feature_inventory_path,
                coverage_sample_rows=int(coverage_sample_rows),
                min_exact_key_coverage=1.0,
                min_non_null_feature_coverage=0.99,
                max_feature_columns=int(max_feature_columns),
                max_profile_feature_columns=DEFAULT_MAX_PROFILE_FEATURE_COLUMNS,
                coverage_segments=segments,
                min_segment_exact_key_coverage=1.0,
                min_segment_non_null_feature_coverage=0.99,
                min_segment_joint_complete_coverage=None,
                min_segment_variance=1e-6,
                binary_prevalence_bounds=(0.005, 0.995),
                required_segment_names=("beginning", "middle", "end"),
                max_rows_per_batch=2_048,
                max_columns_per_read=64,
                reprofile_survivors=False,
                resource_guard=guard,
            )
            side_root = stage_root / side
            bundle = write_loader_evidence_bundle(
                output_dir=side_root / "loader_evidence",
                published_output_dir=destination / side / "loader_evidence",
                universe=universe,
                feature_contract=contract,
                coverage_profile=coverage,
                source_revision=revision,
                max_rows_per_batch=2_048,
                max_columns_per_read=64,
                max_output_bytes=512 * 1024**2,
            )
            feature_loader = make_packb_static_feature_loader(
                feature_store_dir=feature_store,
                feature_contract=contract,
                max_rows_per_batch=2_048,
                max_columns_per_read=64,
                max_output_bytes=512 * 1024**2,
                evidence_bundle=bundle,
                resource_guard=guard,
            )
            report = fit_side_local_ae_gmm_stage(
                side=side,
                cohort_ledger=ledger,
                cohort_ledger_path=cohort_path,
                authorized_population_ledger_path=(
                    Path(population_root)
                    / population_manifest["ledgers"]["authorized_population"]["path"]
                ),
                feature_loader=feature_loader,
                input_features=list(contract.feature_columns),
                output_dir=side_root / "ae_gmm",
                published_output_dir=destination / side / "ae_gmm",
                source_hashes=source_hashes,
                source_revision=revision,
                fixed_calendar_sha256=calendar_sha256,
                seed=41,
                max_train_rows=50_000,
                gmm_max_train_rows=50_000,
                ae_max_iter=80,
                min_reference_rows=20_000,
                min_joint_complete_fraction=0.98,
                resource_guard=guard,
            )
            side_reports[side] = {
                "feature_columns": len(contract.feature_columns),
                "feature_contract_sha256": contract.feature_contract_sha256,
                "coverage_profile_sha256": coverage.profile_sha256,
                "loader_evidence": bundle.to_dict(),
                "ae_gmm": report,
            }
            del feature_loader, bundle, contract, coverage, universe
            del identity, segments, ledger
            _release_memory()
            guard.checkpoint(f"packb_side_ae:{side}:released")
        summary = {
            "schema": "packb_pre_march_side_ae_runner_v1",
            "status": "FROZEN_LONG_AND_SHORT_AE_GMM",
            "source_revision": revision,
            "source_hashes": source_hashes,
            "fixed_calendar_sha256": calendar_sha256,
            "feature_store_revalidation": current_feature_store,
            "feature_store_inventory": feature_binding,
            "coverage_policy": {
                "sample_rows_per_surface": int(coverage_sample_rows),
                "per_feature_finite_min": 0.99,
                "joint_complete_min": 0.98,
                "joint_complete_scope": (
                    "exact_final_50000_row_ae_matrix_across_full_"
                    "beginning_middle_end_periods"
                ),
                "coverage_snapshots": {
                    "beginning": ["2025-01-01", "2025-02-01"],
                    "middle": ["2025-06-01", "2025-07-01"],
                    "end": ["2025-10-01", "2025-11-01"],
                },
                "variance_min_exclusive": 1e-6,
                "binary_prevalence_bounds": [0.005, 0.995],
                "max_feature_columns": int(max_feature_columns),
                "max_profile_feature_columns": DEFAULT_MAX_PROFILE_FEATURE_COLUMNS,
                "selection": "coverage_then_generator_family_round_robin_no_outcomes",
            },
            "sides": side_reports,
        }
        summary_path = stage_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        guard.checkpoint("packb_side_ae:complete")
        os.replace(stage_root, destination)
        return {
            **summary,
            "summary_path": str(destination / "summary.json"),
            "summary_sha256": stage_manifest.sha256_file(destination / "summary.json"),
        }
    except Exception:
        # Keep the hidden staging directory for diagnosis. It is never a
        # canonical result and a rerun always selects a fresh UUID.
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coverage-sample-rows",
        type=int,
        default=DEFAULT_COVERAGE_SAMPLE_ROWS,
    )
    parser.add_argument(
        "--max-feature-columns",
        type=int,
        default=DEFAULT_MAX_FEATURE_COLUMNS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run(
        output_dir=args.output_dir,
        coverage_sample_rows=args.coverage_sample_rows,
        max_feature_columns=args.max_feature_columns,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
