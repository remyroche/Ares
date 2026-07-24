#!/usr/bin/env python3
"""Causally backfill two V11 residual-market fields from frozen primitives.

The V11 train-OOF ledger predates residual-market-composite materialization.
This script recomputes only those composites from V11's point-in-time primitive
feature store. It never reads outcomes, and fails closed unless the rebuilt OOS
values agree with the frozen V11 OOS context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.kraken_actual_data import safe_symbol


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]
CONTEXT_COLUMNS = [
    "short_covering_score_market",
    "funding_confirmed_long_flush",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_utc(value: pd.Series | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(value, utc=True, errors="coerce"))


def _raw_symbol_map(ohlcv_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in ohlcv_root.glob("symbol=*"):
        symbol = path.name.removeprefix("symbol=")
        result[safe_symbol(symbol)] = symbol
    return result


def _load_feature_store_context(
    feature_store_paths: dict[str, Path],
    *,
    index: pd.DatetimeIndex,
    raw_symbol_map: dict[str, str],
    required_symbols: set[str],
) -> dict[str, pd.DataFrame]:
    """Load the exact point-in-time V11 context fields by symbol.

    This is a causal backfill, not a reinterpretation of the feature contract:
    V11's OOS ledger was built from these frozen pre-entry feature columns.
    Recomputing their upstream primitives from a different historical universe
    changes the cross-sectional state and cannot establish train/OOS parity.
    """

    values = {column: {} for column in CONTEXT_COLUMNS}
    for safe in sorted(set(safe_symbol(symbol) for symbol in required_symbols)):
        path = feature_store_paths.get(safe)
        if path is None or safe not in raw_symbol_map:
            continue
        available = set(pq.read_schema(path).names)
        if "ts" not in available or not set(CONTEXT_COLUMNS).issubset(available):
            continue
        frame = pd.read_parquet(path, columns=["ts", *CONTEXT_COLUMNS])
        # Pandas restores Parquet's persisted timestamp index instead of a
        # regular column for some historical feature-store shards.
        timestamps = _to_utc(frame.pop("ts") if "ts" in frame else frame.index)
        frame.index = timestamps
        frame = frame.loc[~frame.index.isna()].groupby(level=0).last()
        symbol = raw_symbol_map[safe]
        for column in CONTEXT_COLUMNS:
            values[column][symbol] = pd.to_numeric(
                frame[column], errors="coerce"
            ).reindex(index).astype(np.float32)
    if min(len(series) for series in values.values()) < 3:
        raise RuntimeError(
            "Frozen V11 context store has fewer than three usable symbol series"
        )
    return {
        column: pd.DataFrame(series, index=index, dtype=np.float32).median(axis=1, skipna=True).astype(np.float32)
        for column, series in values.items()
    }


def _context_for_rows(
    rows: pd.DataFrame,
    signals: dict[str, pd.Series],
) -> pd.DataFrame:
    result = rows.loc[:, KEYS].copy()
    timestamp_index = signals[CONTEXT_COLUMNS[0]].index
    positions = timestamp_index.get_indexer(result["__ts__"])
    for column in CONTEXT_COLUMNS:
        values = np.full(len(result), np.nan, dtype=np.float32)
        valid = positions >= 0
        values[valid] = signals[column].to_numpy(np.float32, copy=False)[positions[valid]]
        result[column] = values
    return result


def _parity_report(rebuilt: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    joined = frozen.loc[:, KEYS + CONTEXT_COLUMNS].merge(
        rebuilt, on=KEYS, suffixes=("_frozen", "_rebuilt"), how="inner", validate="one_to_one"
    )
    rows: list[dict[str, object]] = []
    for column in CONTEXT_COLUMNS:
        left = pd.to_numeric(joined[f"{column}_frozen"], errors="coerce").to_numpy(np.float64)
        right = pd.to_numeric(joined[f"{column}_rebuilt"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(left) & np.isfinite(right)
        if valid.sum() >= 2:
            correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
            mae = float(np.mean(np.abs(left[valid] - right[valid])))
        else:
            correlation, mae = np.nan, np.nan
        rows.append(
            {
                "feature": column,
                "matched_rows": int(valid.sum()),
                "match_rate": float(valid.mean()) if len(valid) else 0.0,
                "pearson_correlation": correlation,
                "mean_absolute_error": mae,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.v11_dir / "train_oof_predictions.parquet", columns=KEYS)
    oos = pd.read_parquet(args.v11_dir / "oos_predictions.parquet")
    for frame in (train, oos):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame.dropna(subset=["__ts__", "__symbol__"], inplace=True)
    requested_start = min(train["__ts__"].min(), oos["__ts__"].min())
    requested_end = max(train["__ts__"].max(), oos["__ts__"].max())
    start = (requested_start - pd.Timedelta(days=args.warmup_days)).floor("h")
    end = requested_end.ceil("h")
    index = pd.date_range(start, end, freq="h", tz="UTC", name="__ts__")
    if not args.feature_store_dir.exists():
        raise FileNotFoundError(f"Frozen V11 feature store is missing: {args.feature_store_dir}")
    feature_store_paths = {
        safe_symbol(path.stem.removeprefix("symbol=")): path
        for path in args.feature_store_dir.glob("symbol=*.parquet")
    }
    raw_symbols = _raw_symbol_map(args.raw_root / "ohlcv")
    required_symbols = set(train["__symbol__"].dropna()).union(oos["__symbol__"].dropna())
    signals = _load_feature_store_context(
        feature_store_paths,
        index=index,
        raw_symbol_map=raw_symbols,
        required_symbols=required_symbols,
    )
    missing = [column for column in CONTEXT_COLUMNS if column not in signals]
    if missing:
        raise RuntimeError(f"Residual context reconstruction failed to create: {missing}")
    train_context = _context_for_rows(train, signals)
    oos_context = _context_for_rows(oos, signals)
    train_context.to_parquet(args.output / "train_oof_residual_context.parquet", index=False, compression="zstd")
    oos_context.to_parquet(args.output / "oos_residual_context.parquet", index=False, compression="zstd")
    coverage = pd.DataFrame(
        {
            "symbol": sorted(required_symbols),
            "in_frozen_primitive_store": [
                safe_symbol(symbol) in feature_store_paths for symbol in sorted(required_symbols)
            ],
        }
    )
    coverage.to_csv(args.output / "frozen_primitive_source_coverage.csv", index=False)
    parity = _parity_report(oos_context, oos)
    parity.to_csv(args.output / "oos_parity_report.csv", index=False)
    passed = bool(
        len(parity) == len(CONTEXT_COLUMNS)
        and parity["match_rate"].ge(args.minimum_parity_match_rate).all()
        and parity["pearson_correlation"].ge(args.minimum_parity_correlation).all()
        and parity["mean_absolute_error"].le(args.maximum_parity_mae).all()
    )
    manifest = {
        "schema": "v11_causal_residual_market_context_backfill_v1",
        "status": "parity_passed" if passed else "parity_failed_do_not_use_train_context",
        "v11_dir": str(args.v11_dir),
        "raw_root": str(args.raw_root),
        "feature_store_dir": str(args.feature_store_dir),
        "source_contract": (
            "Exact frozen point-in-time V11 residual-market context fields; their upstream "
            "transforms were generated from Kraken OHLCV, open interest, and funding only."
        ),
        "requested_range": {"start": str(requested_start), "end": str(requested_end)},
        "reconstruction_range": {"start": str(start), "end": str(end), "warmup_days": int(args.warmup_days)},
        "included_symbols": int(coverage["in_frozen_primitive_store"].sum()),
        "candidate_symbols": int(len(coverage)),
        "frozen_feature_store_symbols": int(len(feature_store_paths)),
        "context_columns": CONTEXT_COLUMNS,
        "train_rows": int(len(train_context)),
        "train_complete_context_rate": float(train_context[CONTEXT_COLUMNS].notna().all(axis=1).mean()),
        "oos_rows": int(len(oos_context)),
        "oos_complete_context_rate": float(oos_context[CONTEXT_COLUMNS].notna().all(axis=1).mean()),
        "parity_requirements": {
            "minimum_match_rate": float(args.minimum_parity_match_rate),
            "minimum_correlation": float(args.minimum_parity_correlation),
            "maximum_mae": float(args.maximum_parity_mae),
        },
        "parity": parity.to_dict(orient="records"),
        "source_hashes": {
            "train_oof_predictions": _sha256(args.v11_dir / "train_oof_predictions.parquet"),
            "oos_predictions": _sha256(args.v11_dir / "oos_predictions.parquet"),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data_perp/exchanges/krakenfutures"))
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=Path("data_perp/features/20260710_170000"),
        help="Frozen causal feature store used by V11 for the funding-residual input.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--minimum-parity-match-rate", type=float, default=0.995)
    parser.add_argument("--minimum-parity-correlation", type=float, default=0.995)
    parser.add_argument("--maximum-parity-mae", type=float, default=0.03)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
