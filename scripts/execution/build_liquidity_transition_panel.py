#!/usr/bin/env python3
"""Materialise causal L2 liquidity-transition panels from compact surfaces.

This is deliberately a second stage.  The state builder works symbol/day by
symbol/day to keep L2 reconstruction memory bounded; market breadth and BTC
context can only be formed after the complete observed market universe for a
date is available.  The producer therefore reads a bounded date at a time,
calculates cross-sectional values before any candidate filtering, and writes
one immutable date partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.liquidity_transition import (  # noqa: E402
    add_actual_position_book_cost,
    add_causal_cross_asset_features,
    add_causal_liquidity_transition_features,
    add_causal_trade_activity_features,
    join_causal_btc_benchmark_context,
    join_causal_context,
    join_causal_trade_activity_recap,
)


def _read_date_surfaces(root: Path, date: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"year=*/date={date}/symbol=*/surface.parquet"))
    if not paths:
        raise FileNotFoundError(f"no compact L2 surfaces for {date} under {root}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", required=True, help="UTC YYYY-MM-DD partitions")
    parser.add_argument("--context", type=Path, help="Optional causal OI/funding/context parquet")
    parser.add_argument("--symbol-mapping", type=Path, help="Explicit dataset_symbol/context_symbol CSV")
    parser.add_argument("--btc-context", type=Path, help="Optional completed-candle BTC benchmark parquet")
    parser.add_argument("--activity-context", type=Path, help="Optional completed-minute aggregate trade-activity recap")
    parser.add_argument(
        "--restrict-to-activity-symbols",
        action="store_true",
        help="Build only symbols represented in --activity-context, for a dense activity-feature cohort.",
    )
    parser.add_argument("--position-notional", type=float, help="Static research notional for depth-ratio fields")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    context = pd.read_parquet(args.context) if args.context else None
    mapping = pd.read_csv(args.symbol_mapping) if args.symbol_mapping else None
    btc_context = pd.read_parquet(args.btc_context) if args.btc_context else None
    activity_context = pd.read_parquet(args.activity_context) if args.activity_context else None
    if args.restrict_to_activity_symbols and activity_context is None:
        raise ValueError("--restrict-to-activity-symbols requires --activity-context")
    activity_symbols = set(activity_context["symbol"].dropna().astype(str)) if activity_context is not None else set()
    receipt_rows: list[dict[str, object]] = []
    for date in sorted(set(args.dates)):
        output = args.out_root / f"date={date}" / "surface.parquet"
        if args.skip_existing and output.exists():
            receipt_rows.append({"date": date, "status": "existing", "path": str(output), "sha256": _sha256(output)})
            continue
        panel = _read_date_surfaces(args.surface_root, date)
        if args.restrict_to_activity_symbols:
            panel = panel.loc[panel["symbol"].isin(activity_symbols)].copy()
            if panel.empty:
                raise ValueError(f"no activity-context symbols are present in source panel for {date}")
        if args.position_notional is not None:
            if args.position_notional <= 0.0:
                raise ValueError("--position-notional must be positive")
            panel["position_notional"] = float(args.position_notional)
            panel = add_actual_position_book_cost(panel, side="sell")
        panel = add_causal_liquidity_transition_features(panel)
        panel = add_causal_cross_asset_features(panel)
        if btc_context is not None:
            panel = join_causal_btc_benchmark_context(panel, btc_context)
        if activity_context is not None:
            # The recap covers a completed source minute and is accepted only
            # when its own availability timestamp precedes this row's next-bar
            # decision boundary.  Derived volume ratios are then computed from
            # contiguous completed aggregate minutes only.
            panel = join_causal_trade_activity_recap(panel, activity_context)
            panel = add_causal_trade_activity_features(panel)
        if context is not None:
            # Avoid sorting an entire multi-year context store for every
            # one-day L2 partition.  The as-of contract has a two-hour max
            # age, so this bounded slice is logically equivalent.
            date_start = pd.Timestamp(date, tz="UTC")
            date_context = context.copy()
            context_time = "available_ts" if "available_ts" in date_context.columns else "timestamp"
            date_context[context_time] = pd.to_datetime(date_context[context_time], utc=True, errors="coerce")
            date_context = date_context.loc[
                date_context[context_time].between(date_start - pd.Timedelta(hours=2), date_start + pd.Timedelta(days=1), inclusive="left")
            ]
            panel = join_causal_context(panel, date_context, symbol_mapping=mapping)
        # No target values influence point-in-time features.  The label
        # columns remain as separately named offline supervision only.
        panel = panel.sort_values(["state_minute", "symbol"], kind="stable").reset_index(drop=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_name(f".{output.name}.partial")
        panel.to_parquet(staged, index=False)
        os.replace(staged, output)
        receipt_rows.append({
            "date": date,
            "status": "materialized",
            "path": str(output),
            "sha256": _sha256(output),
            "rows": int(len(panel)),
            "symbols": int(panel["symbol"].nunique()),
            "feature_columns": int(len(panel.columns)),
            "context_rows": int(panel["__context_available_ts"].notna().sum()) if "__context_available_ts" in panel else 0,
        })
    receipt = {
        "schema": "ares.liquidity_transition_panel.v1",
        "surface_root": str(args.surface_root),
        "context": str(args.context) if args.context else None,
        "btc_context": str(args.btc_context) if args.btc_context else None,
        "activity_context": str(args.activity_context) if args.activity_context else None,
        "activity_contract": "per-minute aggregates only; exact symbol/minute join; source availability must precede next-bar decision timestamp",
        "restricted_to_activity_symbols": bool(args.restrict_to_activity_symbols),
        "context_contract": "backward-only as-of by explicit symbol mapping; stale context becomes null",
        "btc_context_contract": "completed 1-minute BTC OHLCV; returns joined only once their candle-close available_ts is reached",
        "market_contract": "cross-sectional values use the full observed L2 surface for each UTC minute before filtering",
        "partitions": receipt_rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    pd.DataFrame.from_records(receipt_rows).to_parquet(args.out_root / "materialization_audit.parquet", index=False)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
