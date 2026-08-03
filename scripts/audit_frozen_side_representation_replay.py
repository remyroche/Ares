#!/usr/bin/env python3
"""Read-only parity audit for the frozen Pack-B side-local AE/GMM states.

This is deliberately *not* a materializer.  It exists to answer one narrow
question when the current static-store registry has drifted: do the exact
current-store reads, fed through the bytes-bound frozen state, reproduce an
already materialized historical representation surface?  The sole schema
bypass is local to this diagnostic; it never writes a candidate/context output
and cannot be imported as a production replay path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The legacy package emits initialization logs during import.  Keep a CLI
# report machine-readable by routing only those import-time messages to stderr.
with contextlib.redirect_stdout(sys.stderr):
    from extreme_price_movements.packb_static_point_feature_loader import (  # noqa: E402
        FrozenFeatureContract,
        _provenance_backed_raw_allowlist,
        iter_point_in_time_feature_batches,
        point_feature_matrix_sha256,
    )
    from scripts.run_packb_pre_march_side_fs_hpo import (  # noqa: E402
        SideRepresentationFeatureLoader,
        _active_ae_gmm_columns,
        _load_loader_contract,
        _load_side_ae_state,
    )


SIDES = ("long", "short")
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
# The eleven frozen fields consumed by the July execution-EV contracts.  The
# first ten are state transforms; availability is deterministically one when
# the full 63-field state output is finite.
EXECUTION_EV_REQUIRED_REPRESENTATION = (
    "dae_b16_00",
    "dae_b16_02",
    "dae_b16_04",
    "dae_b16_08",
    "dae_b16_14",
    "expected_mahalanobis",
    "gmm_cluster_posterior_4",
    "gmm_dist_center_4",
    "gmm_dist_center_9",
    "gmm_ood_score",
    "gmm_representation_available",
)
DEFAULT_HISTORICAL = (
    ROOT
    / "data_perp/artifacts/packb_downstream_representation_july20_20260726_v1_31_8"
)
DEFAULT_CANDIDATES = (
    ROOT
    / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v1"
    / "candidates/candidate_features.parquet"
)
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_STORE = ROOT / "data_perp/features/20260711_070000"


class FrozenRepresentationAuditError(RuntimeError):
    """Raised when the audit cannot make a bounded, exact comparison."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically preserve only the diagnostic JSON, never feature values."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"audit report parent does not exist: {path.parent}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(IDENTITY).difference(frame.columns)
    if missing:
        raise FrozenRepresentationAuditError(
            f"identity columns missing: {sorted(missing)}"
        )
    output = frame.copy()
    output["__ts__"] = pd.to_datetime(output["__ts__"], utc=True, errors="raise")
    for column in ("__symbol__", "side_name", "candidate_id"):
        output[column] = output[column].astype(str)
    output["side_name"] = output["side_name"].str.lower()
    if output["candidate_id"].duplicated().any() or output.duplicated(list(IDENTITY)).any():
        raise FrozenRepresentationAuditError("candidate identity is not one-to-one")
    if not set(output["side_name"]).issubset(SIDES):
        raise FrozenRepresentationAuditError("unexpected side in candidate identity")
    return output


def exact_identity_overlap(
    historical: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Return deterministic exact identity overlap, never an as-of join."""

    left = _normalise_identity(historical).loc[:, list(IDENTITY)]
    right = _normalise_identity(candidates).loc[:, list(IDENTITY)]
    return left.merge(right, on=list(IDENTITY), how="inner", validate="one_to_one").sort_values(
        ["__ts__", "__symbol__", "side_name", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def deterministic_side_sample(
    frame: pd.DataFrame,
    *,
    rows_per_side: int,
    finite_columns: Sequence[str],
) -> pd.DataFrame:
    """Hash-sample finite saved rows stratified by side and calendar month."""

    if rows_per_side < 1:
        raise ValueError("rows_per_side must be positive")
    output: list[pd.DataFrame] = []
    for side in SIDES:
        local = frame.loc[frame["side_name"].eq(side)].copy()
        values = local.loc[:, list(finite_columns)].to_numpy(dtype=np.float32, copy=False)
        local = local.loc[np.isfinite(values).all(axis=1)].copy()
        if local.empty:
            raise FrozenRepresentationAuditError(
                f"historical context has no finite {side} representation rows"
            )
        local["__digest__"] = local["candidate_id"].map(
            lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        )
        local["__month__"] = local["__ts__"].dt.strftime("%Y-%m")
        months = sorted(local["__month__"].unique())
        base, remainder = divmod(rows_per_side, len(months))
        for index, month in enumerate(months):
            take = base + int(index < remainder)
            available = local.loc[local["__month__"].eq(month)]
            if len(available) < take:
                raise FrozenRepresentationAuditError(
                    f"historical context has only {len(available)} finite {side} rows in {month}; "
                    f"cannot draw deterministic {take}-row parity sample"
                )
            output.append(
                available.sort_values(["__digest__", "candidate_id"], kind="stable")
                .head(take)
                .drop(columns=["__digest__", "__month__"])
            )
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["side_name", "__ts__", "__symbol__", "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )


def _canonical_representation_sha256(
    identities: pd.DataFrame, values: pd.DataFrame, columns: Sequence[str]
) -> str:
    """Hash exact identities and canonical float32 representation values."""

    normalized = _normalise_identity(identities)
    array = values.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=True)
    bits = array.view(np.uint32)
    bits[np.isnan(array)] = np.uint32(0x7FC00000)
    bits[array == 0.0] = np.uint32(0)
    digest = hashlib.sha256()
    payload = {
        "identity": [
            [str(candidate_id), timestamp.isoformat(), str(symbol), str(side)]
            for timestamp, symbol, side, candidate_id in normalized.loc[:, list(IDENTITY)].itertuples(
                index=False, name=None
            )
        ],
        "columns": list(columns),
        "shape": list(array.shape),
        "dtype": "float32_le",
    }
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())
    return digest.hexdigest()


def _identity_sha256(identities: pd.DataFrame) -> str:
    normalized = _normalise_identity(identities)
    payload = [
        [str(candidate_id), timestamp.isoformat(), str(symbol), str(side)]
        for timestamp, symbol, side, candidate_id in normalized.loc[:, list(IDENTITY)].itertuples(
            index=False, name=None
        )
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _diagnostic_side_loader(
    *, side: str, ae_root: Path, feature_store: Path
) -> tuple[SideRepresentationFeatureLoader, tuple[str, ...], dict[str, Any]]:
    """Build an exact-read loader with schema revalidation bypassed only here."""

    summary = json.loads((ae_root / "summary.json").read_text(encoding="utf-8"))
    contract_mapping, _bundle, evidence_hashes = _load_loader_contract(
        ae_root / side / "loader_evidence", source_revision=str(summary["source_revision"])
    )
    contract = FrozenFeatureContract.from_mapping(contract_mapping)
    _allowlist, _rejected, current_allowlist_hash, current_registry_hash = (
        _provenance_backed_raw_allowlist()
    )
    if current_allowlist_hash != contract.raw_allowlist_sha256:
        raise FrozenRepresentationAuditError(
            f"{side} current raw allowlist differs from frozen contract"
        )
    raw_features = tuple(contract.feature_columns)
    matrix_evidence: dict[str, Any] = {}

    def raw_loader(ledger: pd.DataFrame, requested: Sequence[str]) -> pd.DataFrame:
        requested_tuple = tuple(map(str, requested))
        if not requested_tuple or not set(requested_tuple).issubset(raw_features):
            raise FrozenRepresentationAuditError("invalid diagnostic raw feature request")
        matrix = np.empty((len(ledger), len(raw_features)), dtype=np.float32)
        matched = np.zeros(len(ledger), dtype=bool)
        # This is the only schema bypass: exact identity and finite values are
        # still mandatory, and the frozen raw allowlist was checked above.
        for batch in iter_point_in_time_feature_batches(
            ledger,
            feature_store_dir=feature_store,
            feature_contract=contract,
            verify_frozen_schema=False,
            coverage_discovery=False,
            max_rows_per_batch=2_000,
            max_columns_per_read=64,
        ):
            matrix[batch.ledger_row_positions] = batch.features.to_numpy(
                dtype=np.float32, copy=False
            )
            matched[batch.ledger_row_positions] = batch.matched_exact_keys
        if not matched.all() or not np.isfinite(matrix).all():
            raise FrozenRepresentationAuditError(
                f"{side} diagnostic current-store raw matrix is incomplete/non-finite"
            )
        raw = pd.DataFrame(matrix, columns=list(raw_features))
        matrix_evidence.update(
            {
                "rows": int(len(ledger)),
                "exact_key_rows": int(matched.sum()),
                "point_feature_matrix_sha256": point_feature_matrix_sha256(
                    ledger, raw, feature_contract=contract
                ),
            }
        )
        return raw.loc[:, list(requested_tuple)].reset_index(drop=True)

    ae_manifest_path = ae_root / side / "ae_gmm/side_stage_manifest.json"
    ae_manifest = json.loads(ae_manifest_path.read_text(encoding="utf-8"))
    state_path = ae_root / side / "ae_gmm" / str(ae_manifest["artifact"]["path"])
    state = _load_side_ae_state(
        state_path,
        expected_side=side,
        expected_sha256=str(ae_manifest["artifact"]["sha256"]),
        raw_features=raw_features,
    )
    generated = tuple(_active_ae_gmm_columns(state))
    return (
        SideRepresentationFeatureLoader(
            raw_loader=raw_loader,
            raw_features=raw_features,
            state=state,
            generated_features=generated,
        ),
        generated,
        {
            **evidence_hashes,
            "frozen_ae_state_sha256": str(ae_manifest["artifact"]["sha256"]),
            "frozen_ae_manifest_sha256": _sha256(ae_manifest_path),
            "frozen_raw_allowlist_sha256": contract.raw_allowlist_sha256,
            "current_raw_allowlist_sha256": current_allowlist_hash,
            "frozen_generator_registry_sha256": contract.generator_registry_sha256,
            "current_generator_registry_sha256": current_registry_hash,
            "frozen_store_scan_manifest_sha256": contract.store_scan_manifest_sha256,
            "raw_matrix": matrix_evidence,
        },
    )


def _validate_historical_binding(
    historical_root: Path, ae_root: Path
) -> tuple[pd.DataFrame, Mapping[str, Any], tuple[str, ...]]:
    context_path = historical_root / "context.parquet"
    manifest_path = historical_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("output", {})
    if expected.get("sha256") != _sha256(context_path):
        raise FrozenRepresentationAuditError("historical representation context hash changed")
    frame = _normalise_identity(pd.read_parquet(context_path))
    if int(expected.get("rows", -1)) != len(frame):
        raise FrozenRepresentationAuditError("historical representation row count changed")
    columns = tuple(map(str, manifest.get("representation", {}).get("generated_features", ())))
    if (
        not columns
        or set(columns).difference(frame.columns)
        or set(EXECUTION_EV_REQUIRED_REPRESENTATION).difference(frame.columns)
    ):
        raise FrozenRepresentationAuditError("historical representation columns are unavailable")
    summary = json.loads((ae_root / "summary.json").read_text(encoding="utf-8"))
    for side in SIDES:
        historical_state = str(
            manifest["ae_gmm"]["loader_evidence_by_side"][side]["ae_state_sha256"]
        )
        current_state = str(summary["sides"][side]["ae_gmm"]["state_sha256"])
        state_path = Path(summary["sides"][side]["ae_gmm"]["state_path"])
        if historical_state != current_state or _sha256(state_path) != current_state:
            raise FrozenRepresentationAuditError(
                f"{side} frozen AE state does not bind historical context"
            )
    return frame, manifest, columns


def run(
    *,
    historical_root: Path = DEFAULT_HISTORICAL,
    candidate_ledger: Path = DEFAULT_CANDIDATES,
    ae_root: Path = DEFAULT_AE_ROOT,
    feature_store: Path = DEFAULT_STORE,
    rows_per_side: int = 256,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    verify_candidate_surface: bool = False,
) -> dict[str, Any]:
    """Return a read-only parity report; it never materializes a replay output."""

    historical, manifest, columns = _validate_historical_binding(historical_root, ae_root)
    candidates = _normalise_identity(pd.read_parquet(candidate_ledger))
    overlap = exact_identity_overlap(historical, candidates)
    sample = deterministic_side_sample(
        historical, rows_per_side=rows_per_side, finite_columns=columns
    )
    report: dict[str, Any] = {
        "schema": "frozen_side_representation_replay_audit_v1",
        "mode": "read_only_diagnostic_only_schema_bypass",
        "production_output_written": False,
        "historical_context": {
            "path": str(historical_root / "context.parquet"),
            "sha256": _sha256(historical_root / "context.parquet"),
            "rows": int(len(historical)),
            "timestamp_min": str(historical["__ts__"].min()),
            "timestamp_max": str(historical["__ts__"].max()),
        },
        "candidate_ledger": {
            "path": str(candidate_ledger),
            "sha256": _sha256(candidate_ledger),
            "rows": int(len(candidates)),
            "timestamp_min": str(candidates["__ts__"].min()),
            "timestamp_max": str(candidates["__ts__"].max()),
            "exact_historical_identity_overlap_rows": int(len(overlap)),
        },
        "historical_manifest_sha256": _sha256(historical_root / "manifest.json"),
        "historical_state_hashes": {
            side: manifest["ae_gmm"]["loader_evidence_by_side"][side]["ae_state_sha256"]
            for side in SIDES
        },
        "comparison_sample": {
            "method": "per-side/month smallest SHA256(candidate_id), finite saved values only",
            "rows_per_side": int(rows_per_side),
            "rows": int(len(sample)),
            "identity_sha256": _identity_sha256(sample),
            "rows_by_side_month": {
                side: {
                    str(month): int(count)
                    for month, count in sample.loc[sample["side_name"].eq(side)]
                    .groupby(sample.loc[sample["side_name"].eq(side), "__ts__"].dt.strftime("%Y-%m"))
                    .size()
                    .items()
                }
                for side in SIDES
            },
        },
        "representation_columns": list(columns),
        "tolerance": {"atol": float(atol), "rtol": float(rtol)},
        "sides": {},
    }
    for side in SIDES:
        local = sample.loc[sample["side_name"].eq(side)].reset_index(drop=True)
        loader, generated, evidence = _diagnostic_side_loader(
            side=side, ae_root=ae_root, feature_store=feature_store
        )
        if tuple(generated) != columns:
            raise FrozenRepresentationAuditError(
                f"{side} generated representation contract differs from historical artifact"
            )
        observed = loader(local, generated)
        saved = local.loc[:, list(columns)].reset_index(drop=True).astype(np.float32)
        observed = observed.loc[:, list(columns)].astype(np.float32)
        saved_array = saved.to_numpy(dtype=np.float32, copy=False)
        observed_array = observed.to_numpy(dtype=np.float32, copy=False)
        finite_pair = np.isfinite(saved_array) & np.isfinite(observed_array)
        nonfinite_mismatch = int(np.logical_xor(np.isfinite(saved_array), np.isfinite(observed_array)).sum())
        difference = np.abs(saved_array.astype(np.float64) - observed_array.astype(np.float64))
        required_generated = tuple(
            column
            for column in EXECUTION_EV_REQUIRED_REPRESENTATION
            if column != "gmm_representation_available"
        )
        saved_required = local.loc[:, list(EXECUTION_EV_REQUIRED_REPRESENTATION)].reset_index(
            drop=True
        ).astype(np.float32)
        observed_required = observed.loc[:, list(required_generated)].copy()
        observed_required["gmm_representation_available"] = np.float32(1.0)
        observed_required = observed_required.loc[:, list(EXECUTION_EV_REQUIRED_REPRESENTATION)]
        required_saved_array = saved_required.to_numpy(dtype=np.float32, copy=False)
        required_observed_array = observed_required.to_numpy(dtype=np.float32, copy=False)
        required_difference = np.abs(
            required_saved_array.astype(np.float64)
            - required_observed_array.astype(np.float64)
        )
        required_finite = np.isfinite(required_saved_array) & np.isfinite(
            required_observed_array
        )
        report["sides"][side] = {
            "rows": int(len(local)),
            "state_sha256": evidence["frozen_ae_state_sha256"],
            "loader_evidence": evidence,
            "historical_representation_sha256": _canonical_representation_sha256(local, saved, columns),
            "current_store_representation_sha256": _canonical_representation_sha256(local, observed, columns),
            "bitwise_equal_values": bool(np.array_equal(saved_array, observed_array, equal_nan=True)),
            "within_tolerance": bool(np.allclose(saved_array, observed_array, atol=atol, rtol=rtol, equal_nan=True)),
            "nonfinite_mismatch_values": nonfinite_mismatch,
            "finite_compared_values": int(finite_pair.sum()),
            "max_abs_error": float(difference[finite_pair].max()) if finite_pair.any() else None,
            "mean_abs_error": float(difference[finite_pair].mean()) if finite_pair.any() else None,
            "required_execution_ev_columns": list(EXECUTION_EV_REQUIRED_REPRESENTATION),
            "required_execution_ev_bitwise_equal": bool(
                np.array_equal(
                    required_saved_array, required_observed_array, equal_nan=True
                )
            ),
            "required_execution_ev_within_tolerance": bool(
                np.allclose(
                    required_saved_array,
                    required_observed_array,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=True,
                )
            ),
            "required_execution_ev_max_abs_error": (
                float(required_difference[required_finite].max())
                if required_finite.any()
                else None
            ),
        }
    report["overall_within_tolerance"] = bool(
        all(item["within_tolerance"] for item in report["sides"].values())
    )
    report["overall_bitwise_equal"] = bool(
        all(item["bitwise_equal_values"] for item in report["sides"].values())
    )
    if verify_candidate_surface:
        candidate_surface: dict[str, Any] = {
            "mode": "read_only_feasibility_no_candidate_or_representation_output",
            "sides": {},
        }
        for side in SIDES:
            local = candidates.loc[candidates["side_name"].eq(side)].reset_index(
                drop=True
            )
            loader, generated, evidence = _diagnostic_side_loader(
                side=side, ae_root=ae_root, feature_store=feature_store
            )
            values = loader(local, generated)
            array = values.to_numpy(dtype=np.float32, copy=False)
            candidate_surface["sides"][side] = {
                "rows": int(len(local)),
                "generated_columns": int(len(generated)),
                "all_finite": bool(np.isfinite(array).all()),
                "finite_values": int(np.isfinite(array).sum()),
                "total_values": int(array.size),
                "representation_sha256": _canonical_representation_sha256(
                    local, values, generated
                ),
                "state_sha256": evidence["frozen_ae_state_sha256"],
                "raw_matrix": evidence["raw_matrix"],
            }
        candidate_surface["all_finite"] = bool(
            all(item["all_finite"] for item in candidate_surface["sides"].values())
        )
        report["candidate_surface_feasibility"] = candidate_surface
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--candidate-ledger", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--rows-per-side", type=int, default=256)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--verify-candidate-surface",
        action="store_true",
        help="read July candidates exactly and report finite frozen representations",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="new JSON diagnostic report path; refuses to overwrite and never writes features",
    )
    args = parser.parse_args()
    report_path = args.report
    del args.report
    result = run(**vars(args))
    if report_path is not None:
        write_json_report(report_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
