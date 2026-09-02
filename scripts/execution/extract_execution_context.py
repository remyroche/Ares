#!/usr/bin/env python3
"""Extract a compact, causally lagged OI/volume/volatility context panel.

The source feature store timestamps its hourly feature rows at the bar start.
This adapter therefore declares a one-hour availability lag by default rather
than assuming the value was known at that start.  It emits an explicit mapping
from Tardis spot names to the canonical perp names; no ticker inference is
performed at join time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONTEXT_COLUMNS = (
    "mkt_ret_15m", "mkt_rv_4h", "mkt_oi_chg_15m", "mkt_oi_chg_4h", "mkt_oi_chg_24h",
    "mkt_oi_flush_z_30d", "mkt_oi_dispersion_1h", "mkt_oi_dispersion_24h",
    "mkt_oi_chg_accel_1h", "mkt_ret_per_oi_change_1h", "mkt_ret_per_oi_change_4h",
    "mkt_abs_ret_per_oi_drop_4h", "mkt_oi_breadth_rising_24h",
    "volume_percentile", "volume_price_corr_ts_resid", "volume_entropy_24", "volume_trend_48",
    "q_lower_tail__volume_z_24", "q_tail_width__volume_z_12", "prior_volatility",
    "q_tail_width__volatility_zscore", "price_rv_15d_robust_z", "ob_spread_bps_z_24h",
    "ob_depth_l20_to_qv_z_7d", "xasset_mkt_depth_to_qv_z", "negative_breadth_pct", "pct_assets_up_15m",
    "mkt_pct_price_up_oi_up_1h", "mkt_pct_price_down_oi_down_1h",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Canonical feature parquet with __ts__/__symbol__")
    parser.add_argument("--mapping", type=Path, required=True, help="Exact Tardis mapping CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", required=True, help="UTC inclusive score date")
    parser.add_argument("--end", required=True, help="UTC exclusive score date")
    parser.add_argument("--availability-lag-minutes", type=int, default=60)
    parser.add_argument("--columns", nargs="+", default=list(DEFAULT_CONTEXT_COLUMNS))
    args = parser.parse_args()
    if args.availability_lag_minutes < 0:
        raise ValueError("availability lag must be non-negative")
    requested = ["__ts__", "__symbol__", *args.columns]
    available = set(pq.ParquetFile(args.source).schema.names)
    columns = [column for column in requested if column in available]
    if not {"__ts__", "__symbol__"}.issubset(columns):
        raise ValueError("source must contain __ts__ and __symbol__")
    source = pd.read_parquet(args.source, columns=columns)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    start, end = pd.to_datetime(args.start, utc=True), pd.to_datetime(args.end, utc=True)
    # Retain a small preceding interval so an as-of join at ``start`` can use
    # a fully resolved prior bar, but never retain context from after ``end``.
    margin = pd.Timedelta(minutes=int(args.availability_lag_minutes) + 120)
    source = source.loc[source["__ts__"].between(start - margin, end, inclusive="left")].copy()
    source = source.rename(columns={"__symbol__": "symbol"})
    source["available_ts"] = source["__ts__"] + pd.Timedelta(minutes=int(args.availability_lag_minutes))
    source = source.drop(columns=["__ts__"]).sort_values(["symbol", "available_ts"], kind="stable")
    mapping = pd.read_csv(args.mapping)
    required = {"dataset_symbol", "internal_symbol"}
    if required.difference(mapping.columns):
        raise ValueError(f"mapping lacks {sorted(required.difference(mapping.columns))}")
    explicit = mapping.loc[:, ["dataset_symbol", "internal_symbol"]].drop_duplicates().rename(columns={"internal_symbol": "context_symbol"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source.to_parquet(args.out, index=False)
    explicit.to_csv(args.out.with_name(args.out.stem + "_symbol_mapping.csv"), index=False)
    receipt = {
        "schema": "ares.execution_context_extract.v1",
        "source": str(args.source), "rows": int(len(source)), "symbols": int(source["symbol"].nunique()),
        "start": str(start), "end": str(end), "availability_lag_minutes": int(args.availability_lag_minutes),
        "requested_columns": list(args.columns), "available_columns": [column for column in args.columns if column in source.columns],
        "causality": "context available_ts equals its source bar timestamp plus declared availability lag",
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
