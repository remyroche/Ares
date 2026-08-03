#!/usr/bin/env python3
"""Materialize an exact, outcome-free Primary100 context sidecar.

This sidecar is deliberately a data-only handoff for the Primary100
competing-risk context ablation.  It aligns the canonical 134,889 exact
identities with the frozen base-candidate and frozen AE/GMM representation
streams, plus two raw causal transition-entropy inputs from the Primary100
feature universe.  It never reads labels, paths, actions, or realised returns.

The output is suitable for a downstream runner to load with an exact four-key
one-to-one join.  It does not itself decide a model, an entry action, or a
portfolio policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
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


SCHEMA = "primary100_exact_outcome_free_context_sidecar_v1"
EXPECTED_ROWS = 134_889
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")

DEFAULT_FEATURES = (
    ROOT
    / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2"
    / "capture_feature_universe.parquet"
)
DEFAULT_FEATURE_MANIFEST = DEFAULT_FEATURES.with_name("manifest.json")
DEFAULT_CANDIDATE_CONTEXT = (
    ROOT
    / "data_perp/artifacts/packb_downstream_context_july20_20260726_v1_31_8"
    / "context.parquet"
)
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_CANDIDATE_CONTEXT.with_name("manifest.json")
DEFAULT_REPRESENTATION_CONTEXT = (
    ROOT
    / "data_perp/artifacts/packb_downstream_representation_july20_20260726_v1_31_8"
    / "context.parquet"
)
DEFAULT_REPRESENTATION_MANIFEST = DEFAULT_REPRESENTATION_CONTEXT.with_name(
    "manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/primary100_exact_context_sidecar_20260730_v1"
)

# This is an explicit whitelist.  Do not derive a wider output schema by
# selecting every numeric source column: that would turn future additions into
# an unaudited handoff change.
CANDIDATE_FIELDS = (
    "base_oof_score",
    "base_cutoff_score",
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_score_z_timestamp_side",
    "base_candidate_rank_timestamp_side",
    "base_candidate_rank_pct_timestamp_side",
    "base_signal_zscore_within_archetype",
    "base_rank_decile",
    "base_candidate_group_rows",
)
TRANSITION_SOURCE_FIELDS = (
    "capture_candidate__regime_transition_entropy_12h",
    "capture_candidate__regime_transition_entropy_48h",
)
TRANSITION_OUTPUT_FIELDS = (
    "raw_regime_transition_entropy_12h",
    "raw_regime_transition_entropy_48h",
)
DAE_FIELDS = tuple(f"dae_b16_{index:02d}" for index in range(16))
GMM_POSTERIOR_FIELDS = tuple(f"gmm_cluster_posterior_{index}" for index in range(12))
GMM_GEOMETRY_FIELDS = tuple(
    [*(f"gmm_dist_center_{index}" for index in range(12)),
     *(f"gmm_mahal_{index}" for index in range(12)),
     "gmm_cluster_id", "gmm_posterior_max", "gmm_posterior_margin",
     "gmm_unknown_probability", "gmm_ood_score", "gmm_entropy",
     "cluster_entropy_norm", "mahalanobis_distance", "expected_mahalanobis"]
)
GMM_RISK_FIELDS = ("dae_reconstruction_error", "dae_reconstruction_error_zscore")
REPRESENTATION_AVAILABILITY = "gmm_representation_available"
REPRESENTATION_FIELDS = (
    *DAE_FIELDS,
    *GMM_POSTERIOR_FIELDS,
    *GMM_GEOMETRY_FIELDS,
    *GMM_RISK_FIELDS,
)

FORBIDDEN_OUTPUT_TOKENS = (
    "label",
    "outcome",
    "execution_",
    "return",
    "gross",
    "net_ev",
    "mfe",
    "mae",
    "timing",
    "time_to",
    "wait",
    "target_price",
    "targetprice",
    "entry_price",
    "action_",
    "path_",
)


class Primary100ContextSidecarError(RuntimeError):
    """Raised when the exact, point-in-time sidecar cannot be proven."""


def _git_revision() -> str:
    """Return a best-effort source revision without importing training modules."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path, datetime)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
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


def _canonical_identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise Primary100ContextSidecarError(f"{source} lacks identity fields: {missing}")
    result = frame.copy()
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.strip().str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    if not result["side_name"].isin(("long", "short")).all():
        raise Primary100ContextSidecarError(f"{source} has a noncanonical side")
    if result[list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise Primary100ContextSidecarError(f"{source} identity is null or duplicated")
    return result


def _exact_join(anchor: pd.DataFrame, source: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Perform an order-preserving exact four-key one-to-one join."""

    left = _canonical_identity(anchor, source="Primary100 feature universe")
    right = _canonical_identity(source, source=name)
    if len(left) != len(anchor):  # defensive: canonicalization must preserve rows
        raise Primary100ContextSidecarError("identity canonicalization changed feature rows")
    payload = right.copy()
    payload["__exact_source_match__"] = 1
    joined = left[list(IDENTITY)].merge(payload, on=list(IDENTITY), how="left", validate="one_to_one", sort=False)
    if len(joined) != len(left):
        raise Primary100ContextSidecarError(f"{name} exact join changed Primary100 row count")
    if not joined[list(IDENTITY)].equals(left[list(IDENTITY)].reset_index(drop=True)):
        raise Primary100ContextSidecarError(f"{name} exact join changed Primary100 identity order")
    if not joined["__exact_source_match__"].eq(1).all():
        raise Primary100ContextSidecarError(f"{name} lacks complete Primary100 identity coverage")
    return joined.drop(columns="__exact_source_match__")


def _required(frame: pd.DataFrame, fields: Sequence[str], *, source: str) -> None:
    missing = sorted(set(fields).difference(frame.columns))
    if missing:
        raise Primary100ContextSidecarError(f"{source} lacks required fields: {missing}")


def _finite(frame: pd.DataFrame, fields: Sequence[str], *, name: str) -> None:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise Primary100ContextSidecarError(f"{name} must be finite")


def _assert_no_forbidden_output(fields: Sequence[str]) -> None:
    forbidden = [
        field
        for field in fields
        if any(token in str(field).lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if forbidden:
        raise Primary100ContextSidecarError(
            "sidecar attempted to emit outcome/action fields: " + ", ".join(sorted(forbidden))
        )


def _manifest_binding(
    path: Path,
    manifest_path: Path,
    *,
    output_key: str | None,
    source_name: str,
) -> dict[str, str]:
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{source_name} source or manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output_key is None:
        declared = manifest.get("output", {})
    else:
        declared = manifest.get("outputs", {}).get(output_key, {})
    expected = declared.get("sha256")
    actual = _sha256(path)
    if not isinstance(expected, str) or expected != actual:
        raise Primary100ContextSidecarError(
            f"{source_name} manifest does not bind the source parquet hash"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _representation_context_binding(
    candidate_path: Path,
    representation_manifest_path: Path,
) -> None:
    manifest = json.loads(representation_manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("context", {}).get("sha256")
    actual = _sha256(candidate_path)
    if not isinstance(expected, str) or expected != actual:
        raise Primary100ContextSidecarError(
            "representation manifest is not bound to the declared frozen candidate context"
        )


def build_sidecar(
    feature_universe: pd.DataFrame,
    candidate_context: pd.DataFrame,
    representation_context: pd.DataFrame,
    *,
    expected_rows: int | None = EXPECTED_ROWS,
) -> pd.DataFrame:
    """Build a whitelist-only sidecar, preserving exact feature-universe order."""

    feature = _canonical_identity(feature_universe, source="Primary100 feature universe")
    if expected_rows is not None and len(feature) != int(expected_rows):
        raise Primary100ContextSidecarError(
            f"Primary100 feature universe must contain exactly {expected_rows} rows; got {len(feature)}"
        )
    _required(feature, TRANSITION_SOURCE_FIELDS, source="Primary100 feature universe")
    _finite(feature, TRANSITION_SOURCE_FIELDS, name="raw transition entropy fields")

    candidate = _canonical_identity(candidate_context, source="candidate context")
    _required(candidate, (*CANDIDATE_FIELDS, "selected_top40", "prediction_source"), source="candidate context")
    if not candidate["selected_top40"].astype(bool).all():
        raise Primary100ContextSidecarError("candidate context includes rows outside the frozen candidate frontier")
    if not candidate["prediction_source"].astype(str).eq("outer_oof_fold_model").all():
        raise Primary100ContextSidecarError("candidate context is not strict outer-OOF provenance")
    _finite(candidate, CANDIDATE_FIELDS, name="candidate context fields")

    representation = _canonical_identity(representation_context, source="representation context")
    _required(
        representation,
        (REPRESENTATION_AVAILABILITY, *REPRESENTATION_FIELDS),
        source="representation context",
    )
    availability = pd.to_numeric(representation[REPRESENTATION_AVAILABILITY], errors="coerce")
    if not availability.isin((0.0, 1.0)).all():
        raise Primary100ContextSidecarError("representation availability must be binary 0/1")
    rep_values = representation.loc[:, list(REPRESENTATION_FIELDS)].apply(pd.to_numeric, errors="coerce")
    missing = rep_values.isna().any(axis=1)
    if (missing & availability.eq(1.0)).any():
        raise Primary100ContextSidecarError(
            "representation values may be missing only when availability=0"
        )
    if np.isinf(rep_values.to_numpy(float)).any():
        raise Primary100ContextSidecarError("representation context contains infinite values")

    candidate_joined = _exact_join(feature, candidate, name="candidate context")
    representation_joined = _exact_join(feature, representation, name="representation context")
    # Candidate lineage appears in both sources.  Prove that representation was
    # constructed from the same frozen base handoff, not merely matching IDs.
    shared = [field for field in CANDIDATE_FIELDS if field in representation_joined.columns]
    if shared:
        _finite(representation_joined, shared, name="representation candidate lineage")
        for field in shared:
            if not np.allclose(
                candidate_joined[field].to_numpy(float),
                representation_joined[field].to_numpy(float),
                rtol=0.0,
                atol=1e-7,
            ):
                raise Primary100ContextSidecarError(
                    f"candidate and representation contexts disagree on frozen {field}"
                )

    output = feature.loc[:, list(IDENTITY)].copy()
    for field in CANDIDATE_FIELDS:
        output[field] = candidate_joined[field].to_numpy()
    output["candidate_prediction_source"] = candidate_joined["prediction_source"].astype(str).to_numpy()
    output["candidate_selected_top40"] = candidate_joined["selected_top40"].astype(bool).to_numpy()
    output[REPRESENTATION_AVAILABILITY] = pd.to_numeric(
        representation_joined[REPRESENTATION_AVAILABILITY], errors="raise"
    ).to_numpy(dtype=np.int8)
    for field in REPRESENTATION_FIELDS:
        # Crucially do not fill unavailable representation values.  The
        # availability flag is emitted with the raw NaNs for native missing
        # handling / an explicit downstream missingness arm.
        output[field] = representation_joined[field].to_numpy()
    for source, target in zip(TRANSITION_SOURCE_FIELDS, TRANSITION_OUTPUT_FIELDS, strict=True):
        output[target] = feature[source].to_numpy()

    _assert_no_forbidden_output(output.columns)
    if output.duplicated(list(IDENTITY)).any() or len(output) != len(feature):
        raise Primary100ContextSidecarError("sidecar no longer has exactly one row per Primary100 identity")
    if not output[list(IDENTITY)].equals(feature[list(IDENTITY)].reset_index(drop=True)):
        raise Primary100ContextSidecarError("sidecar changed frozen Primary100 identity order")
    _finite(output, (*CANDIDATE_FIELDS, *TRANSITION_OUTPUT_FIELDS), name="candidate and transition sidecar fields")
    out_available = output[REPRESENTATION_AVAILABILITY].eq(1)
    if output.loc[out_available, list(REPRESENTATION_FIELDS)].isna().any().any():
        raise Primary100ContextSidecarError("available representation rows contain missing values")
    if np.isinf(output.loc[:, list(REPRESENTATION_FIELDS)].to_numpy(float)).any():
        raise Primary100ContextSidecarError("sidecar representation fields contain infinite values")
    return output.reset_index(drop=True)


def run(
    *,
    features_path: Path,
    features_manifest_path: Path,
    candidate_context_path: Path,
    candidate_manifest_path: Path,
    representation_context_path: Path,
    representation_manifest_path: Path,
    destination: Path,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, Any]:
    """Materialize a new immutable sidecar artifact, failing closed on lineage."""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Primary100 context sidecar: {destination}")
    sources = {
        "feature_universe": _manifest_binding(
            features_path, features_manifest_path, output_key="universe", source_name="feature universe"
        ),
        "candidate_context": _manifest_binding(
            candidate_context_path, candidate_manifest_path, output_key=None, source_name="candidate context"
        ),
        "representation_context": _manifest_binding(
            representation_context_path, representation_manifest_path, output_key=None, source_name="representation context"
        ),
    }
    _representation_context_binding(candidate_context_path, representation_manifest_path)
    feature_columns = [*IDENTITY, *TRANSITION_SOURCE_FIELDS]
    candidate_columns = [*IDENTITY, *CANDIDATE_FIELDS, "selected_top40", "prediction_source"]
    # Read the overlapping candidate fields too, although they are not copied
    # from this source.  Their equality proves the representation context was
    # generated from the same frozen base handoff on the actual selected rows.
    representation_columns = [
        *IDENTITY,
        *CANDIDATE_FIELDS,
        REPRESENTATION_AVAILABILITY,
        *REPRESENTATION_FIELDS,
    ]
    feature = pd.read_parquet(features_path, columns=feature_columns)
    candidate = pd.read_parquet(candidate_context_path, columns=candidate_columns)
    representation = pd.read_parquet(representation_context_path, columns=representation_columns)
    sidecar = build_sidecar(feature, candidate, representation, expected_rows=expected_rows)

    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        output_path = stage / "context.parquet"
        sidecar.to_parquet(output_path, index=False, compression="zstd", compression_level=5)
        output_binding = {
            "path": str(destination / output_path.name),
            "sha256": _sha256(output_path),
            "rows": int(len(sidecar)),
            "columns": int(len(sidecar.columns)),
            "candidate_identity_sha256": candidate_identity_sha256(sidecar, columns=IDENTITY),
        }
        report = {
            "schema": SCHEMA,
            "status": "MATERIALIZED_EXACT_OUTCOME_FREE_PRIMARY100_CONTEXT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": _git_revision(),
            "sources": sources,
            "output": output_binding,
            "identity_contract": "exact ordered four-key (__ts__, __symbol__, side_name, candidate_id) one-to-one join to the 134,889-row Primary100 feature universe",
            "candidate_context": {
                "fields": list(CANDIDATE_FIELDS),
                "provenance": "selected_top40 outer_oof_fold_model only; finite",
            },
            "representation_context": {
                "availability_field": REPRESENTATION_AVAILABILITY,
                "dae_fields": list(DAE_FIELDS),
                "gmm_posterior_fields": list(GMM_POSTERIOR_FIELDS),
                "gmm_geometry_fields": list(GMM_GEOMETRY_FIELDS),
                "gmm_risk_fields": list(GMM_RISK_FIELDS),
                "missingness": "NaNs are preserved and permitted only when gmm_representation_available=0",
            },
            "raw_transition_context": {
                "source_fields": list(TRANSITION_SOURCE_FIELDS),
                "output_fields": list(TRANSITION_OUTPUT_FIELDS),
                "provenance": "immutable Primary100 point-in-time feature panel",
            },
            "prohibited": "no realised outcomes, labels, path/MFE/MAE, timing, wait, target-price, or actions",
        }
        _write_json(stage / "report.json", report)
        manifest = {
            "schema": SCHEMA,
            "status": report["status"],
            "report": {
                "path": str(destination / "report.json"),
                "sha256": _sha256(stage / "report.json"),
            },
            "sources": sources,
            "output": output_binding,
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    value.add_argument("--features-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    value.add_argument("--candidate-context", type=Path, default=DEFAULT_CANDIDATE_CONTEXT)
    value.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    value.add_argument("--representation-context", type=Path, default=DEFAULT_REPRESENTATION_CONTEXT)
    value.add_argument("--representation-manifest", type=Path, default=DEFAULT_REPRESENTATION_MANIFEST)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(_jsonable(run(
        features_path=arguments.features,
        features_manifest_path=arguments.features_manifest,
        candidate_context_path=arguments.candidate_context,
        candidate_manifest_path=arguments.candidate_manifest,
        representation_context_path=arguments.representation_context,
        representation_manifest_path=arguments.representation_manifest,
        destination=arguments.output_dir,
        expected_rows=arguments.expected_rows,
    )), indent=2, sort_keys=True))
