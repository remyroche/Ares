#!/usr/bin/env python3
"""Drop stale Kraken perp OI-derived feature columns from an existing snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extreme_price_movements.features_oi import get_oi_feature_names
from extreme_price_movements.perp_features import get_perp_feature_names


OI_DERIVED_EXACT = {
    "leverage_build",
    "leverage_build_score",
    "unwind",
    "unwind_score",
    "squeeze_prob",
    "oi_chg_8h_mkt_resid",
    "oi_rel_vol_8h_peer_resid",
}


def _columns_to_drop(existing: list[str]) -> list[str]:
    stale = set(get_oi_feature_names())
    stale.update(
        name
        for name in get_perp_feature_names()
        if name.startswith("oi_") or name in OI_DERIVED_EXACT
    )
    stale.update(OI_DERIVED_EXACT)
    return [name for name in existing if name in stale]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data_perp/features/20260520_004500"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(args.feature_root.glob("symbol=*.parquet"))
    if not files:
        raise FileNotFoundError(args.feature_root)

    touched = 0
    dropped_total = 0
    drop_counts: dict[str, int] = {}
    for path in files:
        df = pd.read_parquet(path)
        drops = _columns_to_drop(list(df.columns))
        if not drops:
            continue
        touched += 1
        dropped_total += len(drops)
        for col in drops:
            drop_counts[col] = drop_counts.get(col, 0) + 1
        if not args.dry_run:
            out = df.drop(columns=drops)
            tmp = path.with_suffix(path.suffix + ".tmp")
            out.to_parquet(tmp, compression="zstd")
            tmp.replace(path)
    print(
        {
            "feature_root": str(args.feature_root),
            "files": len(files),
            "touched": touched,
            "dropped_total": dropped_total,
            "dry_run": bool(args.dry_run),
        }
    )
    print("top_dropped_columns")
    for name, count in sorted(drop_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:80]:
        print(f"{name},{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
