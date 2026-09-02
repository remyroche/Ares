#!/usr/bin/env python3
"""Verify raw strict-R3 source panels are invariant to a later source cutoff.

This audit is deliberately upstream of feature generation.  For each bounded
symbol batch it materialises the same half-open raw panel twice, once ending at
the held-period cutoff and once at a later cutoff, then compares the earlier
prefix exactly (including finite masks).  It never opens labels, outcomes,
models, admission maps, portfolio state, or exchange I/O.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SCRIPT = ROOT / "scripts" / "run_tp6_sl4_exact170_canonical_consensus.py"


def _load_canonical_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "strict_r3_source_horizon_canonical", CANONICAL_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CANONICAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_ts(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _batches(values: list[str], size: int) -> list[list[str]]:
    return [values[offset : offset + size] for offset in range(0, len(values), size)]


def _manifest_symbols(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    symbols = sorted({str(value) for value in payload.get("symbols", []) if str(value)})
    if len(symbols) < 20:
        raise AssertionError(f"{path}: requires a complete market context")
    return symbols


def _compare_prefix(
    short: dict[str, pd.DataFrame], long: dict[str, pd.DataFrame]
) -> tuple[dict[str, dict[str, float | int | bool]], bool]:
    if set(short) != set(long):
        raise AssertionError("raw source keys differ across horizons")
    result: dict[str, dict[str, float | int | bool]] = {}
    all_exact = True
    for field in sorted(short):
        left = short[field]
        right = long[field].reindex(index=left.index, columns=left.columns)
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            raise AssertionError(f"{field}: raw source identity changed across horizons")
        a = left.to_numpy(dtype=np.float32, copy=False)
        b = right.to_numpy(dtype=np.float32, copy=False)
        finite_mismatch = int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        max_delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        exact = bool(np.array_equal(a, b, equal_nan=True))
        result[field] = {
            "exact": exact,
            "finite_mismatch_cells": finite_mismatch,
            "max_abs_delta": max_delta,
        }
        all_exact = all_exact and exact
    return result, all_exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--symbols-manifest", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--short-end", required=True)
    parser.add_argument("--long-end", required=True)
    parser.add_argument("--symbol-batch-size", type=int, default=20)
    parser.add_argument(
        "--source-io-workers", type=int, default=1,
        help="Bounded per-batch source-reader threads; one is the robust default for this audit.",
    )
    args = parser.parse_args()
    if args.symbol_batch_size < 1:
        raise ValueError("--symbol-batch-size must be positive")
    if args.source_io_workers < 1:
        raise ValueError("--source-io-workers must be positive")
    start = _parse_ts(args.start)
    short_end = _parse_ts(args.short_end)
    long_end = _parse_ts(args.long_end)
    if not start < short_end < long_end:
        raise ValueError("require start < short-end < long-end")
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")
    args.out.mkdir(parents=True)
    symbols = _manifest_symbols(args.symbols_manifest)
    # The audit compares many independent source batches.  The canonical
    # reader defaults to four workers for a single production panel, but that
    # parallelism can retain several large Arrow frames at once between audit
    # batches.  Bound it here without changing the source precedence or any
    # generated value.
    os.environ["STRICT_R3_SOURCE_IO_WORKERS"] = str(args.source_io_workers)
    canonical = _load_canonical_module()
    batches = _batches(symbols, args.symbol_batch_size)
    report: dict[str, Any] = {
        "schema": "strict_r3_raw_source_horizon_invariance_v1",
        "scope": "research-only raw source provenance audit; no labels/outcomes/models/admission/portfolio/exchange I/O",
        "start": start.isoformat(),
        "short_end_exclusive": short_end.isoformat(),
        "long_end_exclusive": long_end.isoformat(),
        "symbols": len(symbols),
        "symbol_batch_size": int(args.symbol_batch_size),
        "source_io_workers": int(args.source_io_workers),
        "batches": [],
    }
    progress_path = args.out / "progress.json"
    for ordinal, batch in enumerate(batches):
        print(json.dumps({"event": "batch_start", "batch": ordinal, "symbols": len(batch)}), flush=True)
        short, short_map = canonical._make_panel(
            batch, start, short_end, allow_minute_fallback=False, bar_phase_minutes=0
        )
        long, long_map = canonical._make_panel(
            batch, start, long_end, allow_minute_fallback=False, bar_phase_minutes=0
        )
        fields, exact = _compare_prefix(short, long)
        batch_row = {
            "batch": int(ordinal),
            "symbols": list(batch),
            "source_map_exact": bool(short_map == long_map),
            "fields": fields,
            "all_prefix_values_exact": bool(exact and short_map == long_map),
        }
        report["batches"].append(batch_row)
        progress_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps({
            "event": "batch_complete", "batch": ordinal,
            "all_prefix_values_exact": batch_row["all_prefix_values_exact"],
        }), flush=True)
        del short, long
        gc.collect()
    report["all_source_prefixes_exact"] = bool(
        all(row["all_prefix_values_exact"] for row in report["batches"])
    )
    (args.out / "raw_source_horizon_invariance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    progress_path.unlink(missing_ok=True)
    (args.out / "run_manifest.json").write_text(json.dumps({
        "schema": report["schema"],
        "scope": report["scope"],
        "symbols_manifest": str(args.symbols_manifest),
        "symbols_manifest_sha256": hashlib.sha256(args.symbols_manifest.read_bytes()).hexdigest(),
        "all_source_prefixes_exact": report["all_source_prefixes_exact"],
    }, indent=2, sort_keys=True))
    print(json.dumps({
        "event": "audit_complete",
        "all_source_prefixes_exact": report["all_source_prefixes_exact"],
    }), flush=True)
    if not report["all_source_prefixes_exact"]:
        raise SystemExit("raw source horizon-invariance failure")


if __name__ == "__main__":
    main()
