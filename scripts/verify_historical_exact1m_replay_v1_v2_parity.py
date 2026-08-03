#!/usr/bin/env python3
"""Fail-closed immutable parity audit for the full-2024 exact-1m replay.

This is deliberately a *replay-artifact* verifier, not an OOF or execution
promotion test.  It proves that a fresh v2 materialisation is semantically
identical to the frozen v1 materialisation: exact hourly candidate identities,
policy labels, decoded one-minute paths, and the final physical/multitask
labels.  It also binds the four-partition coverage seal that made this replay
possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
PATH_COLUMN = "execution_future_path"
PATH_KEYS = ("timestamp", "open", "high", "low", "close")
EXPECTED_PATH_MINUTES = 720


class ParityError(ValueError):
    """A deterministic replay contract or value comparison failed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ParityError(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_sha(path: Path, expected: str, *, what: str) -> None:
    if not path.is_file():
        raise ParityError(f"{what} is missing: {path}")
    actual = sha256(path)
    if actual != str(expected):
        raise ParityError(f"{what} hash mismatch: {path}")


def _output_record(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name == "output":
        record = manifest.get("output")
    else:
        record = (manifest.get("outputs") or {}).get(name)
    if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
        raise ParityError(f"manifest has no hash-bound {name} output")
    return record


def _verify_output(manifest: Mapping[str, Any], name: str) -> tuple[Path, int | None, str]:
    record = _output_record(manifest, name)
    path = _resolve(str(record["path"]))
    _require_sha(path, str(record["sha256"]), what=f"{name} output")
    rows = record.get("rows")
    actual_rows = (
        int(pq.ParquetFile(path).metadata.num_rows)
        if path.suffix.lower() == ".parquet"
        else int(len(pd.read_csv(path)))
    )
    if rows is not None and actual_rows != int(rows):
        raise ParityError(f"{name} row count disagrees with its manifest")
    return path, None if rows is None else int(rows), str(record["sha256"])


def _identity_values(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(IDENTITY) - set(frame.columns))
    if missing:
        raise ParityError(f"candidate identity columns missing: {missing}")
    timestamps = pd.to_datetime(frame["__ts__"], utc=True, errors="raise").astype("int64")
    return np.column_stack(
        (
            timestamps.to_numpy(dtype=np.int64),
            frame["__symbol__"].astype(str).to_numpy(),
            frame["side_name"].astype(str).str.lower().to_numpy(),
            frame["candidate_id"].astype(str).to_numpy(),
        )
    )


def _first_identity(frame: pd.DataFrame, index: int) -> dict[str, str]:
    row = frame.iloc[int(index)]
    return {
        "__ts__": pd.Timestamp(row["__ts__"]).isoformat(),
        "__symbol__": str(row["__symbol__"]),
        "side_name": str(row["side_name"]),
        "candidate_id": str(row["candidate_id"]),
    }


def _decode_path(value: Any) -> tuple[np.ndarray, ...]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ParityError("serialized 1m path is not JSON") from exc
    if set(payload) != set(PATH_KEYS):
        raise ParityError("serialized 1m path has an unexpected schema")
    arrays: list[np.ndarray] = []
    for key in PATH_KEYS:
        dtype = np.int64 if key == "timestamp" else np.float64
        values = np.asarray(payload[key], dtype=dtype)
        if values.shape != (EXPECTED_PATH_MINUTES,):
            raise ParityError(f"serialized 1m path has wrong {key} length")
        arrays.append(values)
    return tuple(arrays)


def _compare_paths(
    reference: pd.Series,
    replay: pd.Series,
    identities: pd.DataFrame,
    *,
    float_atol: float,
) -> dict[str, Any]:
    mismatches = 0
    first: dict[str, str] | None = None
    max_delta = 0.0
    for position, (left, right) in enumerate(zip(reference, replay, strict=True)):
        left_arrays, right_arrays = _decode_path(left), _decode_path(right)
        timestamp_equal = np.array_equal(left_arrays[0], right_arrays[0])
        numeric_delta = max(
            (
                float(np.max(np.abs(a - b)))
                for a, b in zip(left_arrays[1:], right_arrays[1:], strict=True)
            ),
            default=0.0,
        )
        equal = timestamp_equal and bool(
            all(
                np.allclose(a, b, rtol=0.0, atol=float_atol, equal_nan=True)
                for a, b in zip(left_arrays[1:], right_arrays[1:], strict=True)
            )
        )
        max_delta = max(max_delta, numeric_delta)
        if not equal:
            mismatches += 1
            if first is None:
                first = _first_identity(identities, position)
    return {
        "kind": "decoded_1m_path",
        "atol": float(float_atol),
        "mismatch_rows": int(mismatches),
        "max_abs_ohlc_delta": float(max_delta),
        "first_mismatch_identity": first,
        "pass": mismatches == 0,
    }


def _compare_values(
    name: str,
    reference: pd.Series,
    replay: pd.Series,
    identities: pd.DataFrame,
    *,
    float_atol: float,
) -> dict[str, Any]:
    if pd.api.types.is_float_dtype(reference.dtype):
        left = pd.to_numeric(reference, errors="raise").to_numpy(dtype=np.float64)
        right = pd.to_numeric(replay, errors="raise").to_numpy(dtype=np.float64)
        mismatch = ~np.isclose(left, right, rtol=0.0, atol=float_atol, equal_nan=True)
        delta = np.abs(left - right)
        kind = "float"
        max_delta = float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0
    elif pd.api.types.is_datetime64_any_dtype(reference.dtype):
        left = pd.to_datetime(reference, utc=True, errors="raise").astype("int64").to_numpy()
        right = pd.to_datetime(replay, utc=True, errors="raise").astype("int64").to_numpy()
        mismatch = left != right
        kind = "datetime_exact"
        max_delta = 0.0
    elif pd.api.types.is_numeric_dtype(reference.dtype) or pd.api.types.is_bool_dtype(reference.dtype):
        left = reference.to_numpy()
        right = replay.to_numpy()
        mismatch = left != right
        kind = "integer_or_boolean_exact"
        max_delta = 0.0
    else:
        left = reference.astype("string").fillna("<NA>").to_numpy()
        right = replay.astype("string").fillna("<NA>").to_numpy()
        mismatch = left != right
        kind = "categorical_exact"
        max_delta = 0.0
    count = int(np.count_nonzero(mismatch))
    first = _first_identity(identities, int(np.flatnonzero(mismatch)[0])) if count else None
    return {
        "column": name,
        "kind": kind,
        "atol": float(float_atol) if kind == "float" else None,
        "mismatch_rows": count,
        "max_abs_delta": max_delta,
        "first_mismatch_identity": first,
        "pass": count == 0,
    }


def compare_parquet_files(
    reference_path: Path,
    replay_path: Path,
    *,
    float_atol: float,
    path_atol: float,
    batch_rows: int = 4096,
) -> dict[str, Any]:
    """Compare ordered immutable rows, including decoded serialized paths.

    The materializers are deterministic and write an ordered candidate stream.
    Requiring ordered four-key equality is intentionally stronger than a set
    comparison; any hidden reorder is a parity failure rather than a join risk.
    """
    reference_file, replay_file = pq.ParquetFile(reference_path), pq.ParquetFile(replay_path)
    if reference_file.metadata.num_rows != replay_file.metadata.num_rows:
        raise ParityError("parquet row counts differ")
    reference_columns = reference_file.schema.names
    if reference_columns != replay_file.schema.names:
        raise ParityError("parquet schemas/column ordering differ")
    if not set(IDENTITY).issubset(reference_columns):
        raise ParityError("parquet file has no full four-key candidate identity")
    column_summaries: dict[str, dict[str, Any]] = {}
    rows = 0
    left_batches: Iterable[Any] = reference_file.iter_batches(batch_size=int(batch_rows))
    right_batches: Iterable[Any] = replay_file.iter_batches(batch_size=int(batch_rows))
    for pair in zip_longest(left_batches, right_batches, fillvalue=None):
        left_batch, right_batch = pair
        if left_batch is None or right_batch is None or left_batch.num_rows != right_batch.num_rows:
            raise ParityError("parquet batch structure differs")
        left, right = left_batch.to_pandas(), right_batch.to_pandas()
        identities = _identity_values(left)
        if not np.array_equal(identities, _identity_values(right)):
            raise ParityError("ordered four-key candidate identity differs")
        for name in reference_columns:
            if name in IDENTITY:
                continue
            summary = (
                _compare_paths(left[name], right[name], left.loc[:, list(IDENTITY)], float_atol=path_atol)
                if name == PATH_COLUMN
                else _compare_values(name, left[name], right[name], left.loc[:, list(IDENTITY)], float_atol=float_atol)
            )
            existing = column_summaries.get(name)
            if existing is None:
                column_summaries[name] = summary
            else:
                existing["mismatch_rows"] += summary["mismatch_rows"]
                existing["max_abs_delta"] = max(
                    float(existing.get("max_abs_delta", existing.get("max_abs_ohlc_delta", 0.0))),
                    float(summary.get("max_abs_delta", summary.get("max_abs_ohlc_delta", 0.0))),
                )
                if existing.get("first_mismatch_identity") is None:
                    existing["first_mismatch_identity"] = summary.get("first_mismatch_identity")
                existing["pass"] = bool(existing["pass"] and summary["pass"])
        rows += len(left)
    return {
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "replay": {"path": str(replay_path), "sha256": sha256(replay_path)},
        "rows": int(rows),
        "columns": column_summaries,
        "pass": all(item["pass"] for item in column_summaries.values()),
    }


def verify_aggregate_seal(path: Path, *, coverage_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _json(path)
    sidecar = path.with_name("manifest.sha256")
    tokens = sidecar.read_text(encoding="utf-8").split() if sidecar.is_file() else []
    if not tokens or tokens[0] != sha256(path):
        raise ParityError("aggregate four-partition seal lacks a valid detached checksum")
    partitions = manifest.get("partitions") or {}
    if (
        manifest.get("schema") != "failure_2024_exact1m_download_verification_v1"
        or manifest.get("status") != "SEALED_COMPLETE"
        or int(manifest.get("partition_count", -1)) != 4
        or set(partitions) != {"0", "1", "2", "3"}
        or int(manifest.get("required_minutes", -1)) != int(manifest.get("covered_minutes", -2))
        or float(manifest.get("coverage_fraction", 0.0)) != 1.0
        or int(manifest.get("failed_symbols", -1)) != 0
        or int(manifest.get("incomplete_symbols", -1)) != 0
    ):
        raise ParityError("aggregate four-partition verification seal is incomplete")
    stage = coverage_manifest.get("stage_manifest") or {}
    request = manifest.get("request_manifest") or {}
    if stage.get("sha256") != request.get("sha256"):
        raise ParityError("coverage audit and aggregate seal bind different request stages")
    stage_manifest_path = _resolve(str(request.get("path", "")))
    stage_manifest = _json(stage_manifest_path)
    candidate = manifest.get("candidate_request") or {}
    expected = (stage_manifest.get("outputs") or {}).get("download_candidates") or {}
    if candidate.get("sha256") != expected.get("sha256") or int(candidate.get("rows", -1)) != int(expected.get("rows", -2)):
        raise ParityError("aggregate seal does not bind the frozen request candidate bytes")
    return {"path": str(path), "sha256": sha256(path), "partitions": 4, "pass": True}


@dataclass(frozen=True)
class ArtifactRoots:
    label_inputs: Path
    policy_labels: Path
    timing_candidates: Path
    paths: Path
    multitask: Path


def _manifests(roots: ArtifactRoots) -> dict[str, dict[str, Any]]:
    return {name: _json(getattr(roots, name) / "manifest.json") for name in roots.__dataclass_fields__}


def _verify_artifact_set(roots: ArtifactRoots, manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Path]:
    if manifests["label_inputs"].get("schema") != "historical_backcast_exact1m_label_inputs_v1":
        raise ParityError("label-input schema mismatch")
    if manifests["policy_labels"].get("schema") != "execution_ev_deployed_policy_1m_labels_v1":
        raise ParityError("policy-label schema mismatch")
    if manifests["timing_candidates"].get("schema") != "execution_entry_timing_candidates_v1":
        raise ParityError("timing-candidate schema mismatch")
    if manifests["paths"].get("schema") != "execution_entry_timing_1m_paths_v1":
        raise ParityError("exact-path schema mismatch")
    if manifests["multitask"].get("schema") != "historical_backcast_exact1m_execution_path_labels_v1" or manifests["multitask"].get("status") != "materialized":
        raise ParityError("multitask v2 artifact is not sealed/materialized")
    files = {
        "candidates": _verify_output(manifests["label_inputs"], "candidates")[0],
        "context": _verify_output(manifests["label_inputs"], "context")[0],
        "path_targets": _verify_output(manifests["label_inputs"], "path_targets")[0],
        "policy": _verify_output(manifests["policy_labels"], "output")[0],
        "paths": _verify_output(manifests["paths"], "output")[0] if isinstance(manifests["paths"].get("output"), Mapping) else _resolve(str((roots.paths / "paths.parquet"))),
        "physical": _verify_output(manifests["multitask"], "physical_path_labels")[0],
        "joined": _verify_output(manifests["multitask"], "joined_multitask_labels")[0],
        "support": _verify_output(manifests["multitask"], "support_by_month_side")[0],
    }
    # The timing/path manifests bind their artefact using source_artifact_sha256
    # rather than the generic outputs structure.
    timing_path = roots.timing_candidates / "candidates.parquet"
    _require_sha(timing_path, str(manifests["timing_candidates"].get("source_artifact_sha256", "")), what="timing candidates")
    files["timing"] = timing_path
    path_file = roots.paths / "paths.parquet"
    _require_sha(path_file, str(manifests["paths"].get("source_artifact_sha256", "")), what="exact 1m paths")
    files["paths"] = path_file
    return files


def compare_exact_bytes(reference_path: Path, replay_path: Path) -> dict[str, Any]:
    """Compare deterministic non-candidate summary outputs byte-for-byte."""
    reference_hash, replay_hash = sha256(reference_path), sha256(replay_path)
    return {
        "reference": {"path": str(reference_path), "sha256": reference_hash},
        "replay": {"path": str(replay_path), "sha256": replay_hash},
        "comparison": "byte_exact",
        "pass": reference_hash == replay_hash,
    }


def _assert_cross_bindings(
    manifests: Mapping[str, Mapping[str, Any]],
    files: Mapping[str, Path],
    coverage: Mapping[str, Any],
    *,
    coverage_manifest_path: Path,
) -> None:
    policy_source = manifests["policy_labels"].get("source") or {}
    for name, source_key in (("candidates", "candidates_sha256"), ("context", "context_sha256"), ("path_targets", "path_targets_sha256")):
        if policy_source.get(source_key) != sha256(files[name]):
            raise ParityError(f"policy labels do not bind {name} bytes")
    exit_contract = manifests["policy_labels"].get("exit_policy_contract") or {}
    if int(exit_contract.get("horizon_minutes", -1)) != 720 or exit_contract.get("replay_timeframe") != "1m":
        raise ParityError("policy labels do not bind signed 1m/720m replay")
    timing_sources = manifests["timing_candidates"].get("sources") or {}
    if (timing_sources.get("execution_ev_labels") or {}).get("sha256") != sha256(files["policy"]):
        raise ParityError("timing candidates do not bind policy labels")
    if manifests["paths"].get("source_artifact_sha256") != sha256(files["paths"]):
        raise ParityError("paths manifest does not bind serialized path bytes")
    path_timing = manifests["paths"].get("timing") or {}
    path_rows = manifests["paths"].get("rows") or {}
    if int(path_timing.get("cadence_minutes", -1)) != 1 or int(path_timing.get("path_minutes", -1)) != 720 or bool(path_rows.get("subset")):
        raise ParityError("paths are not complete 1m nested 720m paths")
    multi_sources = manifests["multitask"].get("sources") or {}
    for name, source_name in (("paths", "exact_paths"), ("path_targets", "path_targets"), ("policy", "policy_labels")):
        if (multi_sources.get(source_name) or {}).get("sha256") != sha256(files[name]):
            raise ParityError(f"multitask labels do not bind {name} bytes")
    coverage_source = multi_sources.get("candidate_coverage_manifest") or {}
    if coverage_source.get("sha256") != sha256(coverage_manifest_path):
        raise ParityError("multitask labels do not bind the audited coverage manifest")
    if coverage.get("status") != "complete" or int(coverage.get("complete_candidates", -1)) != int(coverage.get("candidate_rows", -2)) or int(coverage.get("required_minutes_per_candidate", -1)) != 720:
        raise ParityError("candidate-level 720m coverage gate is not complete")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    roots = {
        "v1": ArtifactRoots(*[getattr(args, f"v1_{name}") for name in ArtifactRoots.__dataclass_fields__]),
        "v2": ArtifactRoots(*[getattr(args, f"v2_{name}") for name in ArtifactRoots.__dataclass_fields__]),
    }
    manifests = {version: _manifests(value) for version, value in roots.items()}
    files = {version: _verify_artifact_set(roots[version], manifests[version]) for version in roots}
    coverage_paths = {
        "v1": args.v1_coverage_manifest,
        "v2": args.v2_coverage_manifest,
    }
    coverage = {version: _json(path) for version, path in coverage_paths.items()}
    # The full four-partition seal is the fresh-v2 acquisition proof.  The v1
    # bundle still verifies against its own immutable candidate coverage file.
    seal = verify_aggregate_seal(
        args.aggregate_verification_manifest,
        coverage_manifest=coverage["v2"],
    )
    for version in roots:
        _assert_cross_bindings(
            manifests[version], files[version], coverage[version],
            coverage_manifest_path=coverage_paths[version],
        )
    comparisons = {
        name: compare_parquet_files(files["v1"][name], files["v2"][name], float_atol=args.float_atol, path_atol=args.path_atol)
        for name in ("candidates", "context", "path_targets", "policy", "timing", "paths", "physical", "joined")
    }
    comparisons["support_by_month_side"] = compare_exact_bytes(files["v1"]["support"], files["v2"]["support"])
    report = {
        "schema": "historical_exact1m_replay_v1_v2_immutable_parity_v1",
        "scope": "frozen 2024 research-only replay; not OOF and not promotion evidence",
        "identity": list(IDENTITY),
        "float_atol": float(args.float_atol),
        "decoded_path_float_atol": float(args.path_atol),
        "aggregate_four_partition_verification": seal,
        "comparisons": comparisons,
        "pass": bool(all(item["pass"] for item in comparisons.values())),
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "parity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "parity_report.sha256").write_text(f"{sha256(report_path)}  parity_report.json\n", encoding="utf-8")
    if not report["pass"]:
        raise ParityError(f"immutable v1/v2 parity failed; inspect {report_path}")
    return report


def _default(name: str, version: str) -> Path:
    artifact_name = {
        "multitask": "multitask_labels",
    }.get(name, name)
    return ROOT / "data_perp/artifacts" / f"failure_2024_exact1m_{artifact_name}_20260730_{version}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for version in ("v1", "v2"):
        for name in ArtifactRoots.__dataclass_fields__:
            result.add_argument(f"--{version}-{name.replace('_', '-')}", type=Path, default=_default(name, version))
    result.add_argument("--v1-coverage-manifest", type=Path, default=_default("candidate_coverage", "v1") / "manifest.json")
    result.add_argument("--v2-coverage-manifest", type=Path, default=_default("candidate_coverage", "v2") / "manifest.json")
    result.add_argument("--aggregate-verification-manifest", type=Path, default=ROOT / "data_perp/artifacts/failure_2024_exact1m_download_verify_20260730_v1/manifest.json")
    result.add_argument("--output-dir", type=Path, default=ROOT / "data_perp/artifacts/failure_2024_exact1m_replay_v1_v2_parity_20260730_v1")
    result.add_argument("--float-atol", type=float, default=1e-10)
    result.add_argument("--path-atol", type=float, default=0.0)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    if options.float_atol < 0.0 or options.path_atol < 0.0:
        raise ValueError("comparison tolerances must be non-negative")
    print(json.dumps(run(options), indent=2, sort_keys=True))
