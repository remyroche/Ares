"""Audit raw canonical OI/funding source coverage by PF asset and month.

This intentionally audits sidecars, not derived feature columns.  A derived
feature can be absent because of its own eligibility rules; this report answers
the upstream question: how much of the feature-index horizon has a canonical
availability-timestamped OI/funding observation?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backfill_kraken_oi_funding_sidecars import _parse_pf_product, _safe_key


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _index(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        return pd.DatetimeIndex([], tz="UTC")
    try:
        frame = pd.read_parquet(path, columns=[])
        idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
        return pd.DatetimeIndex(idx[~idx.isna()]).drop_duplicates().sort_values()
    except Exception:
        return pd.DatetimeIndex([], tz="UTC")


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = start.normalize().replace(day=1)
    last = end.normalize().replace(day=1)
    return list(pd.date_range(first, last, freq="MS", tz="UTC"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--perp-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--backfill-manifest", type=Path)
    ap.add_argument("--exclude-finalization-json", type=Path)
    ap.add_argument("--start-ts", required=True)
    ap.add_argument("--end-ts", required=True)
    args = ap.parse_args()

    start, end = _utc(args.start_ts), _utc(args.end_ts)
    if end < start:
        raise SystemExit("end-ts must be >= start-ts")
    excluded: set[str] = set()
    if args.exclude_finalization_json and args.exclude_finalization_json.exists():
        excluded = set(map(str, json.loads(args.exclude_finalization_json.read_text()).get("stale_files_cleared", [])))

    rows: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    families = (("open_interest_hourly", "open_interest"), ("funding_hourly", "funding_rate"))
    for feature_path in sorted(args.feature_dir.glob("symbol=*.parquet")):
        symbol = feature_path.stem.removeprefix("symbol=")
        product = _parse_pf_product(symbol)
        if product.status != "READY":
            continue
        if symbol in excluded:
            continue
        feature_idx = _index(feature_path)
        feature_idx = feature_idx[(feature_idx >= start) & (feature_idx <= end)]
        sidecar_key = _safe_key(symbol)
        products.append({"feature_symbol": symbol, "product_id": product.product_id, "sidecar_key": sidecar_key})
        for family, column in families:
            sidecar_path = args.perp_root / family / f"{sidecar_key}.parquet"
            source_idx = _index(sidecar_path)
            source_idx = source_idx[(source_idx >= start) & (source_idx <= end)]
            source_first = source_idx.min() if len(source_idx) else pd.NaT
            for month_start in _month_starts(start, end):
                month_end = month_start + pd.offsets.MonthBegin(1)
                expected = feature_idx[(feature_idx >= month_start) & (feature_idx < month_end)]
                observed = source_idx[(source_idx >= month_start) & (source_idx < month_end)]
                overlap = expected.intersection(observed)
                post_source_expected = expected[expected >= source_first] if pd.notna(source_first) else expected[:0]
                post_source_overlap = post_source_expected.intersection(observed)
                rows.append({
                    "feature_symbol": symbol,
                    "product_id": product.product_id,
                    "sidecar_key": sidecar_key,
                    "family": family,
                    "column": column,
                    "month": month_start.strftime("%Y-%m"),
                    "feature_rows": int(len(expected)),
                    "observed_rows": int(len(observed)),
                    "overlap_rows": int(len(overlap)),
                    "coverage_full": float(len(overlap) / len(expected)) if len(expected) else np.nan,
                    "post_source_expected_rows": int(len(post_source_expected)),
                    "post_source_overlap_rows": int(len(post_source_overlap)),
                    "coverage_post_source": float(len(post_source_overlap) / len(post_source_expected)) if len(post_source_expected) else np.nan,
                    "feature_first_ts": expected.min().isoformat() if len(expected) else None,
                    "feature_last_ts": expected.max().isoformat() if len(expected) else None,
                    "source_first_ts": source_idx.min().isoformat() if len(source_idx) else None,
                    "source_last_ts": source_idx.max().isoformat() if len(source_idx) else None,
                })

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "oi_funding_source_coverage_by_asset_month.parquet"
    table.to_parquet(parquet_path, index=False, compression="zstd")

    summary_rows = []
    for (family, month), group in table.groupby(["family", "month"], sort=True):
        valid = group[group.feature_rows > 0]
        summary_rows.append({
            "family": family,
            "month": month,
            "assets": int(len(valid)),
            "feature_rows": int(valid.feature_rows.sum()),
            "overlap_rows": int(valid.overlap_rows.sum()),
            "weighted_coverage_full": float(valid.overlap_rows.sum() / valid.feature_rows.sum()) if valid.feature_rows.sum() else None,
            "assets_ge_90pct_full": int((valid.coverage_full >= .90).sum()),
            "assets_ge_50pct_full": int((valid.coverage_full >= .50).sum()),
            "weighted_coverage_post_source": float(valid.post_source_overlap_rows.sum() / valid.post_source_expected_rows.sum()) if valid.post_source_expected_rows.sum() else None,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "oi_funding_source_coverage_monthly_summary.csv", index=False)

    manifest = {
        "schema": "kraken_oi_funding_source_coverage_v1",
        "feature_dir": str(args.feature_dir),
        "perp_root": str(args.perp_root),
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "products": len(products),
        "families": [x[0] for x in families],
        "excluded_symbols": sorted(excluded),
        "coverage_definition": "exact intersection of feature-index timestamps and canonical sidecar availability_ts timestamps",
        "coverage_post_source_definition": "intersection after the first observed sidecar timestamp; diagnostic only",
        "backfill_manifest": str(args.backfill_manifest) if args.backfill_manifest else None,
        "rows": int(len(table)),
        "artifacts": [parquet_path.name, "oi_funding_source_coverage_monthly_summary.csv"],
    }
    (args.out_dir / "source_coverage_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "products": len(products), "rows": len(table), "out_dir": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
