#!/usr/bin/env python3
"""Append frozen side-local AE/GMM outputs to canonical downstream context."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    candidate_identity_sha256,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.run_packb_pre_march_side_ae import (  # noqa: E402
    DEFAULT_FEATURE_STORE,
)
from scripts.run_packb_pre_march_side_fs_hpo import _git_revision  # noqa: E402
from scripts.run_packb_side_local_residual_oof import _side_loader  # noqa: E402

SCHEMA = "packb_downstream_frozen_side_representation_v1"
SIDES = ("long", "short")
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
DEFAULT_CONTEXT_ROOT = (
    ROOT / "data_perp/artifacts/packb_downstream_context_20260724_v1_31_8"
)
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm"
)
RESOURCE_TELEMETRY_FILENAME = "training_resource_telemetry.jsonl"
REPRESENTATION_AVAILABLE_FEATURE = "gmm_representation_available"


class DownstreamRepresentationError(RuntimeError):
    """Raised when frozen representation provenance or identity is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_context(
    context: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    context_path: Path,
) -> pd.DataFrame:
    required = {
        *IDENTITY_COLUMNS,
        "side",
        "selected_top40",
        "prediction_source",
    }
    missing = sorted(required.difference(context.columns))
    if missing:
        raise DownstreamRepresentationError(
            f"canonical downstream context is missing columns: {missing}"
        )
    output = context.copy()
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
    output["__symbol__"] = output["__symbol__"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["side"] = output["side"].astype(str).str.lower()
    expected = manifest.get("output", {})
    if (
        expected.get("sha256") != _sha256(context_path)
        or int(expected.get("rows", -1)) != len(output)
        or int(expected.get("columns", -1)) != len(output.columns)
        or expected.get("candidate_identity_sha256")
        != candidate_identity_sha256(output, columns=IDENTITY_COLUMNS)
        or output["candidate_id"].duplicated().any()
        or set(output["side_name"]) != set(SIDES)
        or not output["side"].equals(output["side_name"])
        or not output["selected_top40"].astype(bool).all()
        or set(output["prediction_source"].astype(str)) != {"outer_oof_fold_model"}
    ):
        raise DownstreamRepresentationError(
            "canonical downstream context binding or identity changed"
        )
    return output


def append_side_representation(
    context: pd.DataFrame,
    *,
    side_frames: Mapping[str, pd.DataFrame],
    generated_features_by_side: Mapping[str, Sequence[str]],
    minimum_joint_finite_fraction: float,
    minimum_monthly_joint_finite_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append already aligned per-side representations with strict coverage gates."""

    if not 0.0 < minimum_joint_finite_fraction <= 1.0:
        raise ValueError("minimum_joint_finite_fraction must be in (0, 1]")
    if not 0.0 < minimum_monthly_joint_finite_fraction <= 1.0:
        raise ValueError("minimum_monthly_joint_finite_fraction must be in (0, 1]")
    output = context.copy()
    if REPRESENTATION_AVAILABLE_FEATURE in output:
        raise DownstreamRepresentationError(
            f"context already contains {REPRESENTATION_AVAILABLE_FEATURE}"
        )
    output[REPRESENTATION_AVAILABLE_FEATURE] = np.float32(0.0)
    generated_contract: tuple[str, ...] | None = None
    reports: dict[str, Any] = {}
    for side in SIDES:
        side_mask = output["side_name"].astype(str).eq(side)
        positions = np.flatnonzero(side_mask.to_numpy())
        generated = side_frames.get(side)
        features = tuple(map(str, generated_features_by_side.get(side, ())))
        if (
            generated is None
            or not features
            or len(set(features)) != len(features)
            or list(generated.columns) != list(features)
            or len(generated) != len(positions)
        ):
            raise DownstreamRepresentationError(
                f"{side} generated representation contract is misaligned"
            )
        if generated_contract is None:
            generated_contract = features
            overlap = set(features).intersection(output.columns)
            if overlap:
                raise DownstreamRepresentationError(
                    f"generated features overlap downstream context: {sorted(overlap)}"
                )
            for feature in features:
                output[feature] = np.nan
        elif features != generated_contract:
            raise DownstreamRepresentationError(
                "long and short generated feature contracts differ"
            )
        values = generated.loc[:, list(features)].to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(values)
        joint = finite.all(axis=1)
        joint_fraction = float(joint.mean()) if len(joint) else 0.0
        if joint_fraction < minimum_joint_finite_fraction:
            raise DownstreamRepresentationError(
                f"{side} joint finite representation coverage "
                f"{joint_fraction:.6f} is below {minimum_joint_finite_fraction:.6f}"
            )
        month = pd.to_datetime(
            output.loc[side_mask, "__ts__"], utc=True, errors="raise"
        ).dt.strftime("%Y-%m")
        monthly_coverage = {
            str(key): float(joint[month.to_numpy() == key].mean())
            for key in sorted(month.unique())
        }
        failing_months = {
            key: value
            for key, value in monthly_coverage.items()
            if value < minimum_monthly_joint_finite_fraction
        }
        if failing_months:
            raise DownstreamRepresentationError(
                f"{side} monthly joint finite representation coverage is below "
                f"{minimum_monthly_joint_finite_fraction:.6f}: {failing_months}"
            )
        output.loc[side_mask, list(features)] = values
        output.loc[side_mask, REPRESENTATION_AVAILABLE_FEATURE] = joint.astype(
            np.float32
        )
        reports[side] = {
            "rows": int(len(generated)),
            "joint_finite_rows": int(joint.sum()),
            "joint_finite_fraction": joint_fraction,
            "minimum_feature_finite_fraction": float(
                finite.mean(axis=0).min(initial=1.0)
            ),
            "maximum_feature_finite_fraction": float(
                finite.mean(axis=0).max(initial=0.0)
            ),
            "monthly_joint_finite_fraction": monthly_coverage,
        }
    if generated_contract is None:
        raise DownstreamRepresentationError("no generated representation was appended")
    for feature in generated_contract:
        output[feature] = output[feature].astype(np.float32)
    return output, {
        "generated_features": list(generated_contract),
        "generated_feature_count": len(generated_contract),
        "availability_feature": REPRESENTATION_AVAILABLE_FEATURE,
        "model_feature_count": len(generated_contract) + 1,
        "coverage_by_side": reports,
        "minimum_joint_finite_fraction": float(minimum_joint_finite_fraction),
        "minimum_monthly_joint_finite_fraction": float(
            minimum_monthly_joint_finite_fraction
        ),
    }


def run(
    *,
    context_root: Path,
    ae_root: Path,
    feature_store: Path,
    destination: Path,
    minimum_joint_finite_fraction: float = 0.85,
    minimum_monthly_joint_finite_fraction: float = 0.70,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite downstream representation: {destination}"
        )
    context_path = context_root / "context.parquet"
    context_manifest_path = context_root / "manifest.json"
    ae_summary_path = ae_root / "summary.json"
    context_manifest = json.loads(context_manifest_path.read_text(encoding="utf-8"))
    ae_summary = json.loads(ae_summary_path.read_text(encoding="utf-8"))
    if (
        context_manifest.get("status")
        != "MATERIALIZED_STRICT_BASE_OOF_PREENTRY_CONTEXT"
        or ae_summary.get("status") != "FROZEN_LONG_AND_SHORT_AE_GMM"
    ):
        raise DownstreamRepresentationError(
            "canonical context and frozen side-local AE/GMM are required"
        )
    context = _validate_context(
        pd.read_parquet(context_path),
        manifest=context_manifest,
        context_path=context_path,
    )
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage / RESOURCE_TELEMETRY_FILENAME,
    )
    loader_evidence: dict[str, Any] = {}
    side_frames: dict[str, pd.DataFrame] = {}
    generated_by_side: dict[str, list[str]] = {}
    try:
        guard.preflight("packb_downstream_representation:preflight")
        for side in SIDES:
            guard.checkpoint(f"packb_downstream_representation:{side}:start")
            side_context = context.loc[context["side_name"].eq(side)].reset_index(
                drop=True
            )
            loader, candidates, evidence = _side_loader(
                side=side,
                ae_root=ae_root,
                feature_store=feature_store,
                guard=guard,
            )
            raw_count = int(evidence["raw_candidate_features"])
            generated = list(candidates[raw_count:])
            if not generated or len(generated) != int(
                evidence["generated_candidate_features"]
            ):
                raise DownstreamRepresentationError(
                    f"{side} generated feature inventory is invalid"
                )
            side_frames[side] = loader(side_context, generated)
            generated_by_side[side] = generated
            loader_evidence[side] = evidence
            guard.checkpoint(f"packb_downstream_representation:{side}:loaded")
            gc.collect()
        output, representation = append_side_representation(
            context,
            side_frames=side_frames,
            generated_features_by_side=generated_by_side,
            minimum_joint_finite_fraction=minimum_joint_finite_fraction,
            minimum_monthly_joint_finite_fraction=(
                minimum_monthly_joint_finite_fraction
            ),
        )
        if (
            len(output) != len(context)
            or output["candidate_id"].duplicated().any()
            or candidate_identity_sha256(output, columns=IDENTITY_COLUMNS)
            != context_manifest["output"]["candidate_identity_sha256"]
        ):
            raise DownstreamRepresentationError(
                "representation append changed canonical row identity"
            )
        guard.checkpoint("packb_downstream_representation:write")
        output_path = stage / "context.parquet"
        output.to_parquet(
            output_path, index=False, compression="zstd", compression_level=5
        )
        result = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_CANONICAL_CONTEXT_WITH_FROZEN_SIDE_AE_GMM",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": _git_revision(),
            "context": {
                "path": str(context_path),
                "sha256": _sha256(context_path),
                "manifest_sha256": _sha256(context_manifest_path),
            },
            "ae_gmm": {
                "root": str(ae_root),
                "summary_sha256": _sha256(ae_summary_path),
                "source_revision": ae_summary.get("source_revision"),
                "representation_selection_exception": (
                    "outcome-free frozen cycle representation fitted once on "
                    "the designated pre-March reference contract"
                ),
                "loader_evidence_by_side": loader_evidence,
            },
            "feature_store": {
                "path": str(feature_store),
                "immutable_point_lookup": True,
            },
            "representation": representation,
            "downstream_acceptance_requirements": {
                "missing_values": (
                    "retain rows; LightGBM native missing handling plus "
                    f"{REPRESENTATION_AVAILABLE_FEATURE}"
                ),
                "oof_reporting": (
                    "report every head by side and representation-available "
                    "versus representation-missing support"
                ),
                "promotion_gate": (
                    "no material objective or economic collapse in the "
                    "representation-missing slice"
                ),
            },
            "output": {
                "path": str(destination / "context.parquet"),
                "sha256": _sha256(output_path),
                "rows": int(len(output)),
                "columns": int(len(output.columns)),
                "candidate_identity_sha256": candidate_identity_sha256(
                    output, columns=IDENTITY_COLUMNS
                ),
            },
            "resource_guard": {
                "max_process_rss_bytes": guard.limits.max_process_rss_bytes,
                "min_free_ram_bytes": guard.limits.min_free_ram_bytes,
                "min_free_disk_bytes": guard.limits.min_free_disk_bytes,
                "telemetry": str(destination / RESOURCE_TELEMETRY_FILENAME),
            },
            "outcome_columns_added": [],
        }
        _write_json(stage / "manifest.json", result)
        guard.checkpoint("packb_downstream_representation:complete")
        os.replace(stage, destination)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-joint-finite-fraction", type=float, default=0.85)
    parser.add_argument(
        "--minimum-monthly-joint-finite-fraction", type=float, default=0.70
    )
    args = parser.parse_args()
    result = run(
        context_root=args.context_root,
        ae_root=args.ae_root,
        feature_store=args.feature_store,
        destination=args.output_dir,
        minimum_joint_finite_fraction=args.minimum_joint_finite_fraction,
        minimum_monthly_joint_finite_fraction=(
            args.minimum_monthly_joint_finite_fraction
        ),
    )
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
