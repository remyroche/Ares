#!/usr/bin/env python3
"""Build a concrete T1 live-scoring sample ledger from a feature store.

The live final-fit scorer expects timestamp/symbol/strategy_id rows.  This
utility creates that ledger from feature-store symbol files without using model
scores, ranks, portfolio decisions or outcomes.  Policy outcomes are still
materialized later by the existing replay path.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ACTIVE_HEADS = ("short_asset", "short_boll")
STRATEGY_IDS = {
    "short_asset": (
        "short_asset_minus_mkt_oi_1d_peer_resid_0_34164831"
        "_oi_expansion_compression_balance_24h_0_42287597"
    ),
    "short_boll": (
        "short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385"
        "_price_rv_15d_robust_z_0_060036644"
    ),
}
SIDES = {
    "short_asset": "short",
    "short_boll": "short",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _symbol_from_feature_path(path: Path) -> str:
    stem = path.name
    if not stem.startswith("symbol=") or not stem.endswith(".parquet"):
        raise ValueError(f"Unexpected feature filename: {path}")
    raw = stem[len("symbol=") : -len(".parquet")]
    return raw.replace("_", "/")


def _read_feature_timestamps(path: Path) -> pd.Series:
    try:
        frame = pd.read_parquet(path, columns=["ts"])
        ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    except Exception:
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        elif "ts" in frame.columns:
            ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(frame.index, utc=True, errors="coerce")
    return pd.Series(ts).dropna().drop_duplicates().sort_values(ignore_index=True)


def _parse_timestamp(value: str, *, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_csv_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def build_sample_ledger(
    *,
    feature_store_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    heads: tuple[str, ...] = DEFAULT_ACTIVE_HEADS,
    symbols: set[str] | None = None,
    max_symbols: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not feature_store_dir.exists():
        raise FileNotFoundError(feature_store_dir)
    unknown = [head for head in heads if head not in STRATEGY_IDS]
    if unknown:
        raise ValueError(f"Unknown head(s): {unknown}")

    symbol_filter = set(symbols or set())
    rows: list[pd.DataFrame] = []
    symbol_rows: list[dict[str, Any]] = []
    feature_paths = sorted(feature_store_dir.glob("symbol=*.parquet"), key=lambda p: p.name)
    if max_symbols > 0:
        feature_paths = feature_paths[: int(max_symbols)]

    for path in feature_paths:
        symbol = _symbol_from_feature_path(path)
        if symbol_filter and symbol not in symbol_filter:
            continue
        ts = _read_feature_timestamps(path)
        ts = ts.loc[(ts >= start) & (ts <= end)].reset_index(drop=True)
        symbol_rows.append(
            {
                "symbol": symbol,
                "feature_file": str(path),
                "timestamp_count": int(len(ts)),
                "timestamp_min": ts.min() if len(ts) else None,
                "timestamp_max": ts.max() if len(ts) else None,
            }
        )
        if ts.empty:
            continue
        base = pd.DataFrame({"timestamp": ts})
        base["symbol"] = symbol
        for head in heads:
            part = base.copy()
            part["head"] = head
            part["strategy_id"] = STRATEGY_IDS[head]
            part["side"] = SIDES.get(head, "")
            rows.append(part)

    if rows:
        ledger = pd.concat(rows, axis=0, ignore_index=True, copy=False)
        ledger = ledger.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
        ledger = ledger.drop_duplicates(["timestamp", "strategy_id", "symbol"], keep="last")
        ledger = ledger.reset_index(drop=True)
    else:
        ledger = pd.DataFrame(columns=["timestamp", "symbol", "head", "strategy_id", "side"])

    ts = pd.to_datetime(ledger.get("timestamp"), utc=True, errors="coerce")
    summary = {
        "generated_by": "build_t1_feature_store_sample_ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_store_dir": str(feature_store_dir),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "heads": list(heads),
        "symbol_filter_count": len(symbol_filter),
        "max_symbols": int(max_symbols),
        "feature_files_seen": int(len(feature_paths)),
        "symbols_with_rows": int(pd.DataFrame(symbol_rows).query("timestamp_count > 0").shape[0])
        if symbol_rows
        else 0,
        "rows": int(len(ledger)),
        "timestamp_min": ts.min() if len(ts) else None,
        "timestamp_max": ts.max() if len(ts) else None,
        "timestamp_count": int(ts.nunique()) if len(ts) else 0,
        "rows_by_head": ledger.groupby("head").size().to_dict() if not ledger.empty else {},
        "symbols": int(ledger["symbol"].nunique()) if not ledger.empty else 0,
    }
    return ledger, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store-dir", type=Path, default=Path("data_perp/features/20260627_120000"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", action="append", default=[])
    parser.add_argument("--symbols", default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = _parse_timestamp(str(args.start), name="start")
    end = _parse_timestamp(str(args.end), name="end")
    if end < start:
        raise SystemExit("--end must be >= --start")
    heads = tuple(args.head or DEFAULT_ACTIVE_HEADS)
    ledger, summary = build_sample_ledger(
        feature_store_dir=args.feature_store_dir,
        start=start,
        end=end,
        heads=heads,
        symbols=_parse_csv_filter(args.symbols),
        max_symbols=int(args.max_symbols),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(args.output, index=False)
    ledger.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
