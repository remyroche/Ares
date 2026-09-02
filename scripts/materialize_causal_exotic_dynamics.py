#!/usr/bin/env python3
"""Materialise target-free five-family dynamic-state features for MC1 research.

This producer has one narrow responsibility: it reads the matched BCF/current
candidate identities and completed local 15-minute OHLCV history, then writes
the causal F1--F5 feature matrix in symbol partitions.  It does not read any
outcome, policy path, MC1 target, admission decision, or exchange endpoint.

Each partition contains one row for every candidate in the dual score-family
intersection, even when the bar source is absent or unsuitable.  Such rows
remain explicit missing values with a source-status field; they are never
silently imputed or removed from the target-free identity contract.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_exotic_dynamics import (
    FEATURE_COLUMNS,
    feature_metadata_frame,
    materialize_symbol,
)
from scripts.run_strict_r3_rich_policy_hpo import _symbol_filename


BCF_DEFAULT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
CURRENT_DEFAULT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
)
BARS_DEFAULT = ROOT / "15m_ohlcv_perp"
OUT_DEFAULT = ROOT / "data_perp/artifacts/causal_exotic_dynamics_2025train_2026confirm_20260831_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: object) -> pd.Series | pd.DatetimeIndex:
    return pd.to_datetime(values, utc=True, errors="raise")


def _load_source(path: Path, family: str) -> pd.DataFrame:
    required = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"{family}: score source lacks target-free identity fields {missing}")
    frame = pd.read_parquet(path, columns=required)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    if frame.candidate_id.duplicated().any():
        raise AssertionError(f"{family}: duplicate candidate identities")
    if not frame.side_name.astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{family}: expected long-only research source")
    return frame


def _target_free_intersection(bcf: Path, current: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    left, right = _load_source(bcf, "bcf"), _load_source(current, "current")
    columns = ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]
    result = left.merge(right, on=columns, how="inner", validate="one_to_one")
    result = result.loc[
        result["__decision_ts__"].ge(start) & result["__decision_ts__"].lt(end)
    ].copy()
    result = result.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if result.empty or result.candidate_id.duplicated().any():
        raise AssertionError("target-free BCF/current identity intersection is invalid")
    if result.duplicated(["__symbol__", "__decision_ts__"]).any():
        raise AssertionError("expected one long candidate per symbol and decision timestamp")
    return result


def _empty_feature_rows(group: pd.DataFrame, status: str) -> pd.DataFrame:
    result = group.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].copy()
    result = result.rename(columns={"__decision_ts__": "snapshot_ts"})
    result["dynamic_source_status"] = status
    for field in FEATURE_COLUMNS:
        result[field] = np.nan
    return result


def _read_completed_bars(path: Path, *, attempts: int = 3) -> pd.DataFrame:
    """Read one immutable local source with bounded retries.

    A few otherwise-valid parquet files can transiently raise ``ArrowInvalid``
    while several worker processes open distinct local files concurrently on
    this filesystem.  A retry reads the identical immutable file; it cannot
    change point-in-time semantics.  A genuinely corrupt source still fails
    explicitly after the bounded attempts.
    """
    last: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            return pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
        except Exception as exc:  # retry only immutable local read faults
            last = exc
            if attempt + 1 < attempts:
                time.sleep(.15 * (attempt + 1))
    assert last is not None
    raise last


def _materialize_one(args: tuple[str, pd.DataFrame, Path, Path]) -> tuple[str, pd.DataFrame, dict[str, object]]:
    symbol, group, bars_root, parts_root = args
    started = time.perf_counter()
    source = bars_root / _symbol_filename(symbol)
    if not source.is_file():
        result = _empty_feature_rows(group, "missing_15m_source")
    else:
        try:
            bars = _read_completed_bars(source)
            dynamic = materialize_symbol(bars, pd.DatetimeIndex(group["__decision_ts__"]))
            result = group.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].copy()
            result = result.rename(columns={"__decision_ts__": "snapshot_ts"})
            result = pd.concat([result.reset_index(drop=True), dynamic.drop(columns="snapshot_ts").reset_index(drop=True)], axis=1)
        except Exception as exc:  # local source faults are explicit missingness
            result = _empty_feature_rows(group, f"unreadable_or_invalid_15m_source:{type(exc).__name__}")
    if len(result) != len(group) or result.candidate_id.duplicated().any():
        raise AssertionError(f"{symbol}: feature materialisation changed target-free identities")
    # Kraken symbols contain ``/``; use a stable opaque partition key rather
    # than allowing the symbol string to alter the directory topology.
    partition = hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:16]
    part = parts_root / f"symbol_hash={partition}" / "features.parquet"
    part.parent.mkdir(parents=True, exist_ok=False)
    result.to_parquet(part, index=False, compression="zstd")
    feature_counts = result.loc[:, list(FEATURE_COLUMNS)].notna().sum().to_dict()
    return symbol, result.loc[:, ["candidate_id", "dynamic_source_status"]], {
        "symbol": symbol,
        "rows": int(len(result)),
        "source_status": result.dynamic_source_status.value_counts(dropna=False).to_dict(),
        "finite_counts": {name: int(value) for name, value in feature_counts.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "partition": partition,
        "bar_source": str(source),
        "bar_source_sha256": _sha256(source) if source.is_file() else None,
    }


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if not start < end:
        raise ValueError("--start must precede --end")
    route = _target_free_intersection(args.bcf.resolve(), args.current.resolve(), start, end)
    out.mkdir(parents=True, exist_ok=False)
    parts_root = out / "feature_parts"
    jobs = [
        (str(symbol), group.copy(), args.bars_root.resolve(), parts_root)
        for symbol, group in route.groupby("__symbol__", sort=True)
    ]
    workers = max(1, min(int(args.workers), len(jobs), 8))
    # Feature construction includes forward-only Kalman/CUSUM recursions in
    # Python.  Processes, rather than threads, let independent symbols use
    # separate cores without changing ordering, source values, or outputs.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_materialize_one, jobs))
    status = pd.concat([item[1] for item in results], ignore_index=True)
    status = route.loc[:, ["candidate_id"]].merge(status, on="candidate_id", how="left", validate="one_to_one")
    if len(status) != len(route) or status.dynamic_source_status.isna().any():
        raise AssertionError("feature partitions do not exactly cover the target-free route")
    feature_metadata_frame().to_parquet(out / "feature_metadata.parquet", index=False)
    coverage_rows = []
    total = len(route)
    for field in FEATURE_COLUMNS:
        finite = int(sum(int(item[2]["finite_counts"].get(field, 0)) for item in results))
        coverage_rows.append({"feature_name": field, "finite_rows": finite, "coverage": finite / total, "missing_rows": total - finite})
    pd.DataFrame(coverage_rows).to_parquet(out / "feature_coverage.parquet", index=False)
    pd.DataFrame([item[2] for item in results]).to_parquet(out / "source_coverage_and_cost.parquet", index=False)
    route.to_parquet(out / "target_free_candidate_intersection.parquet", index=False, compression="zstd")
    status.to_parquet(out / "candidate_source_status.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-exotic-dynamics-v1",
        "scope": "offline target-free F1--F5 source materialisation only; no MC1, policy, live, or exchange mutation",
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "source": {
            "bcf": {"path": str(args.bcf.resolve()), "sha256": _sha256(args.bcf.resolve())},
            "current": {"path": str(args.current.resolve()), "sha256": _sha256(args.current.resolve())},
            "bars_root": str(args.bars_root.resolve()),
        },
        "candidate_contract": "exact BCF/current target-free identity intersection; one long candidate per symbol/timestamp",
        "candidate_rows": int(len(route)),
        "symbols": int(route["__symbol__"].nunique()),
        "features": feature_metadata_frame().to_dict(orient="records"),
        "source_status_rows": status.dynamic_source_status.value_counts(dropna=False).to_dict(),
        "workers": workers,
        "no_exchange_calls": True,
        "causality": "every state maps a decision to the final 15m bar strictly before it; all statistics are trailing or forward-filtered; missing source remains explicit null state",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf", type=Path, default=BCF_DEFAULT)
    parser.add_argument("--current", type=Path, default=CURRENT_DEFAULT)
    parser.add_argument("--bars-root", type=Path, default=BARS_DEFAULT)
    parser.add_argument("--start", default="2025-04-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parsed = parser.parse_args()
    print(run(parsed))


if __name__ == "__main__":
    main()
