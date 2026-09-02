#!/usr/bin/env python3
"""Resumable, exact-PIT historical context materializer for path heads.

One shard is written atomically per feature-store symbol.  The final validator
reads only identity columns from those shards, never the full feature matrix.
Future path labels are deliberately excluded from every shard and index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.packb_static_point_feature_loader import (
    iter_point_in_time_feature_batches,
)

DEFAULT_POPULATION = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_top40_20260727_v1/population.parquet"
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_CONTRACT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1/long/loader_evidence/frozen_feature_contract.json"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_historical_path_head_context_20260727_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
BASE_CONTEXT = ("base_oof_score", "base_rank_timestamp_side", "base_group_rows", "base_rank_pct_timestamp_side")
SCHEMA = "febapr2025_historical_path_head_context_v2_partitioned"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _identity_hash(frame: pd.DataFrame, columns: Sequence[str] = IDENTITY) -> str:
    values = frame.loc[:, list(columns)].copy()
    if "__ts__" in values:
        values["__ts__"] = pd.to_datetime(values["__ts__"], utc=True, errors="raise").astype(str)
    values = values.astype(str).sort_values(list(columns), kind="stable")
    digest = hashlib.sha256()
    for row in values.itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _feature_symbol(frame: pd.DataFrame) -> pd.Series:
    return frame["candidate_id"].astype(str).str.split("|", n=1).str[0]


def _symbol_stem(symbol: str) -> str:
    return hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:20]


def _load_raw_features(
    ledger: pd.DataFrame, *, feature_store: Path, feature_contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, int]:
    """Read one bounded symbol ledger with exact, no-as-of feature joins."""

    pieces: list[pd.DataFrame] = []
    matched = 0
    for batch in iter_point_in_time_feature_batches(
        ledger,
        feature_store_dir=feature_store,
        feature_contract=feature_contract,
        verify_frozen_schema=False,
        max_rows_per_batch=2_048,
        max_columns_per_read=64,
    ):
        piece = batch.features.copy()
        piece["__row__"] = batch.ledger_row_positions
        pieces.append(piece)
        matched += int(batch.matched_exact_keys.sum())
    if not pieces:
        raise RuntimeError("point-in-time loader returned no feature batches")
    raw = pd.concat(pieces, ignore_index=True).sort_values("__row__", kind="stable")
    positions = raw.pop("__row__").to_numpy(dtype=np.int64)
    if not np.array_equal(positions, np.arange(len(ledger), dtype=np.int64)):
        raise RuntimeError("point-in-time loader did not preserve exact ledger order")
    return raw.reset_index(drop=True), matched


def _load_population(path: Path) -> pd.DataFrame:
    population = pd.read_parquet(path, columns=[*IDENTITY, *BASE_CONTEXT, "__decision_ts__"])
    population["__ts__"] = pd.to_datetime(population["__ts__"], utc=True, errors="raise")
    population["__decision_ts__"] = pd.to_datetime(population["__decision_ts__"], utc=True, errors="raise")
    population["side_name"] = population["side_name"].astype(str).str.lower()
    if len(population) != 205_194 or population.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("unexpected or duplicate frozen top40 population")
    if not population["__decision_ts__"].eq(population["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("base handoff decision timestamp changed")
    return population


def _shard_paths(output_dir: Path, symbol: str) -> tuple[Path, Path]:
    stem = _symbol_stem(symbol)
    return output_dir / "shards" / f"{stem}.parquet", output_dir / "shards" / f"{stem}.manifest.json"


def _completed_shard(manifest_path: Path, data_path: Path, *, input_hash: str, rows: int) -> bool:
    if not manifest_path.is_file() or not data_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(
        manifest.get("schema") == SCHEMA
        and manifest.get("input_identity_sha256") == input_hash
        and int(manifest.get("rows", -1)) == int(rows)
        and manifest.get("output_sha256") == _sha256(data_path)
    )


def _materialize_symbol(
    frame: pd.DataFrame,
    *,
    symbol: str,
    raw_features: list[str],
    feature_store: Path,
    feature_contract: Mapping[str, Any],
    output_dir: Path,
) -> tuple[str, bool]:
    """Materialize or reuse one atomic historical symbol shard."""

    data_path, manifest_path = _shard_paths(output_dir, symbol)
    input_hash = _identity_hash(frame)
    if _completed_shard(manifest_path, data_path, input_hash=input_hash, rows=len(frame)):
        return str(manifest_path), True
    unique = frame.loc[:, ["candidate_id", "__ts__"]].copy()
    unique["__feature_symbol__"] = _feature_symbol(unique)
    unique = unique.drop_duplicates(["__feature_symbol__", "__ts__"], keep="first").reset_index(drop=True)
    ledger = unique.rename(columns={"__feature_symbol__": "__symbol__"})
    raw, matched = _load_raw_features(ledger, feature_store=feature_store, feature_contract=feature_contract)
    if list(raw.columns) != raw_features or matched != len(ledger):
        raise ValueError(f"exact PIT feature contract failed for {symbol}")
    keyed = pd.concat([unique.loc[:, ["__ts__"]], raw], axis=1)
    context = frame.merge(keyed, on="__ts__", how="left", validate="many_to_one")
    if len(context) != len(frame) or context.loc[:, raw_features].isna().all(axis=1).any():
        raise ValueError(f"incomplete feature context for {symbol}")
    context["hour_sin"] = np.sin(2.0 * np.pi * context["__ts__"].dt.hour.to_numpy() / 24.0).astype(np.float32)
    context["hour_cos"] = np.cos(2.0 * np.pi * context["__ts__"].dt.hour.to_numpy() / 24.0).astype(np.float32)
    context = context.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    _write_parquet_atomic(context, data_path)
    finite = raw.notna().mean()
    _write_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "status": "COMPLETE_EXACT_PIT_PREENTRY_SHARD",
            "feature_symbol": symbol,
            "input_identity_sha256": input_hash,
            "rows": int(len(context)),
            "unique_symbol_signal_keys": int(len(ledger)),
            "exact_key_rows": int(matched),
            "signal_start_utc": str(context["__ts__"].min()),
            "signal_end_utc": str(context["__ts__"].max()),
            "output_path": str(data_path),
            "output_sha256": _sha256(data_path),
            "low_coverage_fields_below_0_99": {str(name): float(value) for name, value in finite.items() if value < 0.99},
            "forbidden": "no path label, future return, realised execution outcome, target-derived weight, or fitted downstream score",
        },
    )
    return str(manifest_path), False


def _finalize_index(
    *, output_dir: Path, population: pd.DataFrame, symbols: Sequence[str], raw_features: Sequence[str]
) -> tuple[Path, dict[str, Any]]:
    """Validate coverage using shard identity columns only, then atomically index it."""

    # Compute each source shard's immutable input contract once.  Recomputing
    # a string split/filter for every symbol was unnecessary and can make a
    # finalization pass less bounded than the materialization itself.
    feature_symbols = _feature_symbol(population)
    input_contract = {
        symbol: (int(mask.sum()), _identity_hash(population.loc[mask]))
        for symbol in symbols
        for mask in (feature_symbols.eq(symbol),)
    }
    required_columns = set((*IDENTITY, *BASE_CONTEXT, "__decision_ts__", "hour_sin", "hour_cos", *raw_features))
    forbidden_exact = {
        "__first_touch_target_soft__", "__first_touch_capture_net__", "__w__",
        "execution_net_ev_12h", "execution_label_end_utc", "native_label_resolution_utc",
    }
    identity_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for symbol in symbols:
        data_path, manifest_path = _shard_paths(output_dir, symbol)
        expected_rows, expected_hash = input_contract[symbol]
        if not _completed_shard(manifest_path, data_path, input_hash=expected_hash, rows=expected_rows):
            raise ValueError(f"incomplete or changed shard for {symbol}")
        manifest = json.loads(manifest_path.read_text())
        import pyarrow.parquet as pq
        schema_names = set(pq.ParquetFile(data_path).schema_arrow.names)
        if schema_names != required_columns:
            raise ValueError(f"context schema mismatch for {symbol}")
        forbidden = sorted(
            name for name in schema_names
            if name in forbidden_exact or name.startswith("__path_auxiliary_")
        )
        if forbidden:
            raise ValueError(f"forbidden outcome field(s) in {symbol}: {forbidden}")
        # The final pass intentionally loads only a compact identity index.
        identity = pd.read_parquet(data_path, columns=list(IDENTITY))
        identity["shard_manifest"] = str(manifest_path)
        identity_parts.append(identity)
        manifests.append(manifest)
    index = pd.concat(identity_parts, ignore_index=True)
    if len(index) != len(population) or index.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("partitioned context identity coverage is incomplete or duplicated")
    if _identity_hash(index) != _identity_hash(population):
        raise ValueError("partitioned context identities differ from frozen population")
    index = index.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    index_path = output_dir / "context_index.parquet"
    _write_parquet_atomic(index, index_path)
    month = pd.to_datetime(index["__ts__"], utc=True, errors="raise").dt.strftime("%Y-%m")
    return index_path, {
        "shards": len(manifests),
        "rows": int(len(index)),
        "identity_sha256": _identity_hash(index),
        "unique_symbol_signal_keys": int(sum(item["unique_symbol_signal_keys"] for item in manifests)),
        "exact_key_rows": int(sum(item["exact_key_rows"] for item in manifests)),
        "raw_feature_count": int(len(raw_features)),
        "rows_by_side": index["side_name"].value_counts().sort_index().astype(int).to_dict(),
        "rows_by_month": month.value_counts().sort_index().astype(int).to_dict(),
        "schema_consistency": "all shard schema names exactly equal the frozen pre-entry contract",
        "forbidden_outcome_scan": "pass: no native/execution/path auxiliary target or target weight columns in any shard schema",
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = json.loads(args.feature_contract.read_text())
    raw_features = list(map(str, contract.get("feature_columns", ())))
    if len(raw_features) != 256 or len(set(raw_features)) != len(raw_features):
        raise ValueError("the frozen raw point-in-time contract must contain 256 unique fields")
    population = _load_population(args.population)
    population["__feature_symbol__"] = _feature_symbol(population)
    all_symbols = sorted(population["__feature_symbol__"].unique())
    requested = set(args.symbol or all_symbols)
    unknown = sorted(requested.difference(all_symbols))
    if unknown:
        raise ValueError(f"requested symbols are absent from frozen population: {unknown[:3]}")
    symbols = [symbol for symbol in all_symbols if symbol in requested]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reused = 0
    written = 0
    if not args.finalize_only:
        for symbol in symbols:
            frame = population.loc[population["__feature_symbol__"].eq(symbol)].drop(columns="__feature_symbol__").copy()
            _manifest, was_reused = _materialize_symbol(frame, symbol=symbol, raw_features=raw_features, feature_store=args.feature_store, feature_contract=contract, output_dir=args.output_dir)
            reused += int(was_reused)
            written += int(not was_reused)
    partial_path = args.output_dir / "progress.json"
    _write_json(partial_path, {"schema": SCHEMA, "requested_shards": len(symbols), "total_required_shards": len(all_symbols), "written_this_run": written, "reused_this_run": reused, "complete": len(symbols) == len(all_symbols)})
    if len(symbols) != len(all_symbols) or (args.finalize_only and requested != set(all_symbols)):
        return {"progress": partial_path}
    index_path, coverage = _finalize_index(output_dir=args.output_dir, population=population.drop(columns="__feature_symbol__"), symbols=all_symbols, raw_features=raw_features)
    manifest_path = args.output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": SCHEMA,
            "status": "MATERIALIZED_PARTITIONED_PREENTRY_ONLY_EXACT_PIT_CONTEXT",
            "context_index": {"path": str(index_path), "sha256": _sha256(index_path), **coverage},
            "identity": list(IDENTITY),
            "population": {"path": str(args.population), "sha256": _sha256(args.population), "rows": int(len(population)), "rows_by_side": population["side_name"].value_counts().sort_index().astype(int).to_dict()},
            "feature_store": {"path": str(args.feature_store), "raw_contract": str(args.feature_contract), "raw_contract_sha256": _sha256(args.feature_contract), "exact_key_fraction": 1.0},
            "preentry_fields": [*BASE_CONTEXT, "__decision_ts__", "hour_sin", "hour_cos", *raw_features],
            "forbidden": "no path labels, outcomes, future returns, realised execution fields, target-derived weights, or fitted downstream scores",
            "timing": {"feature_time": "signal __ts__", "decision": "__ts__ + 1h", "join": "exact (__feature_symbol__, __ts__) only; no as-of or fill"},
            "final_validation": "loaded identity columns only from each shard; no full feature matrix was assembled",
        },
    )
    return {"index": index_path, "manifest": manifest_path, "progress": partial_path}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbol", action="append", help="materialize only this exact feature-store symbol; repeatable")
    parser.add_argument("--finalize-only", action="store_true", help="skip feature reads and publish the identity/schema coverage index from existing atomic shards")
    return parser


if __name__ == "__main__":
    print(json.dumps({key: str(value) for key, value in run(parser().parse_args()).items()}, indent=2))
