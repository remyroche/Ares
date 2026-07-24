#!/usr/bin/env python3
"""Incrementally append causal market-transition features to a compact store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.features_negative_residuals import (
    calculate_market_regime_change_series,
)
from extreme_price_movements.market_regime_change_contract import (
    MARKET_REGIME_CHANGE_FEATURE_KEYS,
    MARKET_REGIME_CHANGE_SCHEMA_VERSION,
)


def _read_series(path: Path, name: str) -> pd.Series:
    table = pq.read_table(path, columns=["ts", name])
    frame = table.to_pandas()
    if "ts" in frame.columns:
        index = pd.to_datetime(frame.pop("ts"), utc=True, errors="coerce")
    else:
        index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    return pd.Series(
        pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32),
        index=index,
        dtype=np.float32,
    )


def _market_median_from_symbol_store(
    root: Path,
    name: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    columns: list[pd.Series] = []
    for path in sorted(root.glob("symbol=*.parquet")):
        if name not in pq.read_schema(path).names:
            continue
        series = _read_series(path, name).reindex(index)
        series.name = path.stem
        columns.append(series)
    if not columns:
        raise ValueError(f"No {name!r} columns found under {root}")
    values = pd.concat(columns, axis=1, copy=False)
    return values.median(axis=1, skipna=True).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, object]:
    target_files = sorted(args.target_root.glob("symbol=*.parquet"))
    if not target_files:
        raise FileNotFoundError(f"No compact feature files under {args.target_root}")
    reference = pd.read_parquet(target_files[0])
    if "ts" in reference.columns:
        index = pd.DatetimeIndex(
            pd.to_datetime(reference.pop("ts"), utc=True, errors="coerce")
        )
    else:
        index = pd.DatetimeIndex(
            pd.to_datetime(reference.index, utc=True, errors="coerce")
        )
    reference = reference.set_index(index)
    levels = {
        "negative_breadth": pd.to_numeric(
            reference["negative_breadth_pct"], errors="coerce"
        ).astype(np.float32),
        "btc_alt_relative_strength": pd.to_numeric(
            reference["median_alt_minus_btc"], errors="coerce"
        ).astype(np.float32),
        "short_covering": pd.to_numeric(
            reference["short_covering_score_market"], errors="coerce"
        ).astype(np.float32),
        "flush_recovery": pd.to_numeric(
            reference["flush_recovery_state"], errors="coerce"
        ).astype(np.float32),
        "funding": _market_median_from_symbol_store(
            args.source_root, "funding_1d_chg_ts_resid", index
        ),
        "eth_correlation": _market_median_from_symbol_store(
            args.source_root, "corr_eth_24h", index
        ),
        # The full store already contains the exact per-asset robust-z used by
        # features_oi before its cross-sectional market median is broadcast.
        "oi_contraction": -_market_median_from_symbol_store(
            args.source_root, "oi_chg_4h_robust_z", index
        ),
    }
    transitions = pd.DataFrame(
        calculate_market_regime_change_series(levels, bars_per_hour=1), index=index
    ).astype(np.float32)
    if list(transitions.columns) != list(MARKET_REGIME_CHANGE_FEATURE_KEYS):
        raise RuntimeError("transition feature contract/order mismatch")
    written = 0
    skipped = 0
    for path in target_files:
        schema_names = set(pq.read_schema(path).names)
        if set(MARKET_REGIME_CHANGE_FEATURE_KEYS).issubset(schema_names) and not args.force:
            skipped += 1
            continue
        frame = pd.read_parquet(path)
        if "ts" in frame.columns:
            frame_index = pd.DatetimeIndex(
                pd.to_datetime(frame.pop("ts"), utc=True, errors="coerce")
            )
        else:
            frame_index = pd.DatetimeIndex(
                pd.to_datetime(frame.index, utc=True, errors="coerce")
            )
        frame.index = frame_index
        frame.index.name = "ts"
        aligned = transitions.reindex(frame_index)
        for name in MARKET_REGIME_CHANGE_FEATURE_KEYS:
            frame[name] = aligned[name].to_numpy(dtype=np.float32, copy=False)
        frame.to_parquet(path, index=True, compression="zstd")
        written += 1
    manifest = {
        "schema": "market_regime_change_incremental_backfill_v1",
        "schema_version": MARKET_REGIME_CHANGE_SCHEMA_VERSION,
        "target_root": str(args.target_root),
        "source_root": str(args.source_root),
        "target_files": len(target_files),
        "written_files": written,
        "skipped_files": skipped,
        "rows": len(transitions),
        "timestamp_min": str(index.min()),
        "timestamp_max": str(index.max()),
        "feature_keys": list(MARKET_REGIME_CHANGE_FEATURE_KEYS),
        "finite_fraction": {
            name: float(transitions[name].notna().mean())
            for name in MARKET_REGIME_CHANGE_FEATURE_KEYS
        },
    }
    (args.target_root / "market_regime_change_backfill_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
