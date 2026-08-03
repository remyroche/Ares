#!/usr/bin/env python3
"""Prove exact candidate-level coverage for a frozen historical 1m stage.

The downloader reports merged symbol-window coverage.  This independent audit
returns to the immutable candidate ledger and proves that every candidate has
each of the 720 expected UTC-minute observations in its half-open decision
window.  It also binds the strict frozen-product verification manifest and
hashes every canonical store part used as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import canonical_kraken_execution_1m_root


OHLCV = ("open", "high", "low", "close", "volume")
MINUTE_NS = int(pd.Timedelta(minutes=1).value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _symbol_parts(root: Path, symbol: str) -> list[Path]:
    directory = root / "ohlcv" / f"symbol={str(symbol).replace('/', '_')}"
    return sorted(directory.glob("year=*/*.parquet"))


def _load_symbol_evidence(
    root: Path,
    symbol: str,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for part in _symbol_parts(root, symbol):
        frame = pd.read_parquet(part, columns=["ts", *OHLCV])
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
        frame = frame.loc[(frame["ts"] >= start) & (frame["ts"] < end_exclusive)]
        if frame.empty:
            continue
        frames.append(frame)
        sources.append(
            {
                "path": str(part.resolve()),
                "sha256": _sha256(part),
                "rows_in_audit_range": int(len(frame)),
            }
        )
    if not frames:
        return np.empty(0, dtype=np.int64), sources, 0
    evidence = pd.concat(frames, ignore_index=True)
    ts_ns = evidence["ts"].astype("int64").to_numpy(dtype=np.int64)
    if np.any(ts_ns % MINUTE_NS):
        raise ValueError(f"{symbol}: canonical evidence is not minute-aligned")
    values = evidence.loc[:, OHLCV].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{symbol}: canonical evidence contains non-finite OHLCV")
    if (
        (values[:, :4] <= 0.0).any()
        or (values[:, 1] < values[:, 2]).any()
        or (values[:, 4] < 0.0).any()
    ):
        raise ValueError(f"{symbol}: canonical evidence contains invalid OHLCV")

    evidence["_ts_ns"] = ts_ns
    duplicate_rows = int(evidence["_ts_ns"].duplicated(keep=False).sum())
    if duplicate_rows:
        for timestamp, rows in evidence.groupby("_ts_ns", sort=False):
            if len(rows) <= 1:
                continue
            reference = rows.iloc[0].loc[list(OHLCV)].to_numpy(dtype=np.float64)
            compared = rows.loc[:, OHLCV].to_numpy(dtype=np.float64)
            if not np.equal(compared, reference).all():
                raise ValueError(
                    f"{symbol}: conflicting immutable rows at "
                    f"{pd.Timestamp(int(timestamp), tz='UTC').isoformat()}"
                )
    unique_ns = np.unique(ts_ns)
    return unique_ns, sources, duplicate_rows


def _validate_strict_download_manifest(
    path: Path,
    *,
    expected_candidate_sha: str,
    expected_stage_manifest_sha: str,
    horizon_minutes: int,
) -> dict[str, Any]:
    payload = _json(path)
    if not payload.get("verify_only"):
        raise ValueError("strict download manifest must be verify-only")
    if payload.get("candidate_sha256") != expected_candidate_sha:
        raise ValueError("strict download manifest does not bind product requests")
    if (
        (payload.get("stage_manifest") or {}).get("sha256")
        != expected_stage_manifest_sha
    ):
        raise ValueError("strict download manifest does not bind request stage")
    if payload.get("product_mapping_contract") != (
        "frozen_product_id_from_candidate_input"
    ):
        raise ValueError("strict verification did not use frozen product IDs")
    if int(payload.get("horizon_minutes", -1)) != int(horizon_minutes):
        raise ValueError("strict verification horizon mismatch")
    summary = payload.get("summary") or {}
    if (
        int(summary.get("incomplete_symbols", -1)) != 0
        or int(summary.get("failed_symbols", -1)) != 0
        or int(summary.get("ok_symbols", 0)) != int(payload.get("symbols", -1))
    ):
        raise ValueError("strict verification manifest is not complete")
    return payload


def _validate_aggregate_download_verification(
    path: Path,
    *,
    expected_stage_manifest_sha: str,
    expected_raw_candidate_sha: str,
    expected_symbols: int,
) -> dict[str, Any]:
    payload = _json(path)
    if (
        payload.get("schema")
        != "failure_2024_exact1m_download_verification_v1"
        or payload.get("status") != "SEALED_COMPLETE"
        or payload.get("verification_only") is not True
    ):
        raise ValueError("aggregate download verification is not sealed/verify-only")
    if (
        (payload.get("request_manifest") or {}).get("sha256")
        != expected_stage_manifest_sha
        or (payload.get("candidate_request") or {}).get("sha256")
        != expected_raw_candidate_sha
    ):
        raise ValueError("aggregate verification does not bind the request stage")
    if (
        int(payload.get("partition_count", -1)) != 4
        or int(payload.get("symbols", -1)) != int(expected_symbols)
        or int(payload.get("required_minutes", -1))
        != int(payload.get("covered_minutes", -2))
        or int(payload.get("required_minutes", 0)) <= 0
        or int(payload.get("incomplete_symbols", -1)) != 0
        or int(payload.get("failed_symbols", -1)) != 0
        or len(payload.get("partitions") or {}) != 4
    ):
        raise ValueError("aggregate four-partition verification is incomplete")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--product-map-manifest", type=Path, required=True)
    parser.add_argument("--strict-download-manifest", type=Path, required=True)
    parser.add_argument(
        "--aggregate-download-verification-manifest",
        type=Path,
        default=None,
        help=(
            "Optional sealed four-partition verification; when supplied it is "
            "validated and hash-bound alongside the frozen-product verifier."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage_manifest_path = args.stage_dir / "manifest.json"
    path_map_path = args.stage_dir / "candidate_path_map.parquet"
    stage_manifest = _json(stage_manifest_path)
    if stage_manifest.get("schema") != "historical_backcast_exact1m_request_stage_v2":
        raise ValueError("stage schema must be historical_backcast_exact1m_request_stage_v2")
    horizon = int(stage_manifest.get("path_horizon_minutes", -1))
    if horizon <= 0:
        raise ValueError("stage horizon is invalid")
    expected_path_map_sha = (
        stage_manifest.get("outputs", {}).get("candidate_path_map", {}).get("sha256")
    )
    if _sha256(path_map_path) != expected_path_map_sha:
        raise ValueError("stage manifest does not bind candidate path map bytes")

    product_manifest = _json(args.product_map_manifest)
    if product_manifest.get("schema") != "kraken_historical_product_map_v1":
        raise ValueError("product map schema is invalid")
    for lineage_key in ("candidate_population_lineage", "product_lineage"):
        stage_value = stage_manifest.get(lineage_key)
        product_value = product_manifest.get(lineage_key)
        if stage_value is not None and product_value != stage_value:
            raise ValueError(
                f"product map {lineage_key} does not match the request stage"
            )
    expected_stage_candidates_sha = (
        stage_manifest.get("outputs", {}).get("staged_candidates", {}).get("sha256")
    )
    if (
        product_manifest.get("stage_candidates", {}).get("sha256")
        != expected_stage_candidates_sha
    ):
        raise ValueError("product map does not bind this stage")
    product_requests = Path(
        product_manifest["outputs"]["download_candidates_with_product"]["path"]
    )
    expected_product_requests_sha = product_manifest["outputs"][
        "download_candidates_with_product"
    ]["sha256"]
    if _sha256(product_requests) != expected_product_requests_sha:
        raise ValueError("product request bytes do not match product manifest")
    strict_manifest = _validate_strict_download_manifest(
        args.strict_download_manifest,
        expected_candidate_sha=expected_product_requests_sha,
        expected_stage_manifest_sha=_sha256(stage_manifest_path),
        horizon_minutes=horizon,
    )
    aggregate_manifest = None
    if args.aggregate_download_verification_manifest is not None:
        aggregate_manifest = _validate_aggregate_download_verification(
            args.aggregate_download_verification_manifest,
            expected_stage_manifest_sha=_sha256(stage_manifest_path),
            expected_raw_candidate_sha=stage_manifest["outputs"][
                "download_candidates"
            ]["sha256"],
            expected_symbols=int(stage_manifest["distinct_symbols"]),
        )

    paths = pd.read_parquet(path_map_path)
    required_columns = {
        "candidate_id",
        "timestamp",
        "symbol",
        "path_end_exclusive",
    }
    missing = sorted(required_columns - set(paths.columns))
    if missing:
        raise ValueError(f"candidate path map missing columns: {missing}")
    if paths["candidate_id"].duplicated().any():
        raise ValueError("candidate path map contains duplicate candidate IDs")
    for column in ("timestamp", "path_end_exclusive"):
        paths[column] = pd.to_datetime(paths[column], utc=True, errors="raise")
    duration = (
        paths["path_end_exclusive"] - paths["timestamp"]
    ) / pd.Timedelta(minutes=1)
    if not duration.eq(horizon).all():
        raise ValueError("candidate path map contains a horizon mismatch")

    covered = np.zeros(len(paths), dtype=np.int32)
    first_missing = np.full(len(paths), np.datetime64("NaT"), dtype="datetime64[ns]")
    source_parts: list[dict[str, Any]] = []
    duplicate_evidence_rows = 0
    store_root = canonical_kraken_execution_1m_root(args.data_root)
    for symbol, index in paths.groupby("symbol", sort=True).groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        starts = paths.loc[positions, "timestamp"]
        ends = paths.loc[positions, "path_end_exclusive"]
        unique_ns, sources, duplicates = _load_symbol_evidence(
            store_root,
            str(symbol),
            start=starts.min(),
            end_exclusive=ends.max(),
        )
        source_parts.extend([{"symbol": str(symbol), **row} for row in sources])
        duplicate_evidence_rows += int(duplicates)
        start_ns = pd.DatetimeIndex(starts).asi8
        end_ns = pd.DatetimeIndex(ends).asi8
        left = np.searchsorted(unique_ns, start_ns, side="left")
        right = np.searchsorted(unique_ns, end_ns, side="left")
        counts = (right - left).astype(np.int32)
        covered[positions] = counts
        for local_position in np.flatnonzero(counts != horizon):
            position = positions[local_position]
            expected = np.arange(
                start_ns[local_position],
                end_ns[local_position],
                MINUTE_NS,
                dtype=np.int64,
            )
            observed = unique_ns[
                left[local_position] : right[local_position]
            ]
            missing_ns = np.setdiff1d(expected, observed, assume_unique=True)
            if len(missing_ns):
                first_missing[position] = np.datetime64(
                    int(missing_ns[0]), "ns"
                )

    audit = paths.copy()
    audit["expected_minutes"] = np.int32(horizon)
    audit["covered_minutes"] = covered
    audit["complete_1m_path"] = covered == horizon
    audit["first_missing_timestamp"] = pd.to_datetime(first_missing, utc=True)
    incomplete = audit.loc[~audit["complete_1m_path"]].copy()

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "candidate_coverage.parquet"
    incomplete_path = output / "incomplete_candidates.parquet"
    audit.to_parquet(audit_path, index=False)
    incomplete.to_parquet(incomplete_path, index=False)
    complete = bool(audit["complete_1m_path"].all())
    manifest = {
        "schema": "historical_exact1m_candidate_coverage_v1",
        "status": "complete" if complete else "incomplete",
        "candidate_rows": int(len(audit)),
        "complete_candidates": int(audit["complete_1m_path"].sum()),
        "incomplete_candidates": int((~audit["complete_1m_path"]).sum()),
        "candidate_coverage_fraction": float(audit["complete_1m_path"].mean()),
        "required_minutes_per_candidate": horizon,
        "path_interval": "[decision_timestamp, decision_timestamp+12h)",
        "store_root": str(store_root.resolve()),
        "immutable_duplicate_rows": int(duplicate_evidence_rows),
        "conflicting_duplicate_rows": 0,
        "stage_manifest": {
            "path": str(stage_manifest_path.resolve()),
            "sha256": _sha256(stage_manifest_path),
        },
        "product_map_manifest": {
            "path": str(args.product_map_manifest.resolve()),
            "sha256": _sha256(args.product_map_manifest),
        },
        "strict_download_manifest": {
            "path": str(args.strict_download_manifest.resolve()),
            "sha256": _sha256(args.strict_download_manifest),
            "verified_symbols": int(strict_manifest["symbols"]),
        },
        **(
            {
                "aggregate_download_verification_manifest": {
                    "path": str(
                        args.aggregate_download_verification_manifest.resolve()
                    ),
                    "sha256": _sha256(
                        args.aggregate_download_verification_manifest
                    ),
                    "partitions": int(aggregate_manifest["partition_count"]),
                    "verified_symbols": int(aggregate_manifest["symbols"]),
                    "required_minutes": int(
                        aggregate_manifest["required_minutes"]
                    ),
                    "covered_minutes": int(
                        aggregate_manifest["covered_minutes"]
                    ),
                }
            }
            if aggregate_manifest is not None
            else {}
        ),
        "source_parts": source_parts,
        "outputs": {
            "candidate_coverage": {
                "path": str(audit_path.resolve()),
                "rows": int(len(audit)),
                "sha256": _sha256(audit_path),
            },
            "incomplete_candidates": {
                "path": str(incomplete_path.resolve()),
                "rows": int(len(incomplete)),
                "sha256": _sha256(incomplete_path),
            },
        },
        "evidence_scope": stage_manifest.get(
            "evidence_scope", "frozen_backcast_diagnostic_not_oof"
        ),
        "lineage": stage_manifest.get(
            "lineage", "historical_frozen_backcast_exact1m_research_only"
        ),
        "candidate_population_lineage": stage_manifest.get(
            "candidate_population_lineage"
        ),
        "product_lineage": stage_manifest.get("product_lineage"),
        "oof_status": "not_oof",
        "execution_parity_claim": False,
        "promotion_eligible": False,
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
