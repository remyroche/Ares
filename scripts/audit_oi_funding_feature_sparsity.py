"""Classify OI/funding feature sparsity after canonical source repair.

The audit is deliberately source-aware: low finite coverage is only an
upstream source problem when the relevant canonical sidecar is also sparse.
Features with a complete source but no finite output are constructor or
dependency failures and should not be silently retained in model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backfill_kraken_oi_funding_sidecars import _parse_pf_product, _safe_key
from finalize_oi_funding_feature_store import SKIPPED_KEYS


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _idx(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        return pd.DatetimeIndex([], tz="UTC")
    try:
        frame = pd.read_parquet(path, columns=[])
        idx = pd.to_datetime(frame.index, utc=True, errors="coerce")
        return pd.DatetimeIndex(idx[~idx.isna()]).drop_duplicates().sort_values()
    except Exception:
        return pd.DatetimeIndex([], tz="UTC")


def _family(key: str) -> str:
    text = str(key).lower()
    has_oi = "oi" in text or "open_interest" in text
    has_fund = "fund" in text or "carry" in text or "basis" in text
    if has_oi and has_fund:
        return "oi_funding_interaction"
    if has_oi:
        return "open_interest"
    if has_fund:
        return "funding"
    return "other"


def _coverage(expected: pd.DatetimeIndex, source: pd.DatetimeIndex) -> tuple[int, float, int, float]:
    overlap = expected.intersection(source)
    full = float(len(overlap) / len(expected)) if len(expected) else np.nan
    first = source.min() if len(source) else pd.NaT
    post = expected[expected >= first] if pd.notna(first) else expected[:0]
    post_overlap = post.intersection(source)
    post_cov = float(len(post_overlap) / len(post)) if len(post) else np.nan
    return int(len(overlap)), full, int(len(post_overlap)), post_cov


def _diagnosis(row: dict[str, Any]) -> str:
    key = row["feature"]
    if key in SKIPPED_KEYS:
        return "unsupported_declared"
    if not row["present"]:
        return "missing_column"
    if row["finite_rows"] == 0:
        return "constructor_or_dependency_failure" if (row["source_post_coverage"] >= .90) else "source_or_eligibility_gap"
    if row["source_post_coverage"] < .90:
        return "upstream_source_sparse"
    if row["finite_post_coverage"] < .90:
        return "derived_warmup_or_eligibility"
    if row["finite_coverage"] < .90:
        return "pre_source_unavailable"
    return "healthy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--perp-root", type=Path, required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exclude-finalization-json", type=Path)
    ap.add_argument("--start-ts", required=True)
    ap.add_argument("--end-ts", required=True)
    args = ap.parse_args()

    start, end = _utc(args.start_ts), _utc(args.end_ts)
    excluded: set[str] = set()
    if args.exclude_finalization_json and args.exclude_finalization_json.exists():
        excluded = set(map(str, json.loads(args.exclude_finalization_json.read_text()).get("stale_files_cleared", [])))
    keys = list(dict.fromkeys(x.strip() for x in args.keys.read_text().splitlines() if x.strip()))
    rows: list[dict[str, Any]] = []
    symbols = []
    for path in sorted(args.feature_dir.glob("symbol=*.parquet")):
        symbol = path.stem.removeprefix("symbol=")
        if symbol in excluded or _parse_pf_product(symbol).status != "READY":
            continue
        feature_idx = _idx(path)
        feature_idx = feature_idx[(feature_idx >= start) & (feature_idx <= end)]
        sidecar_key = _safe_key(symbol)
        oi_idx = _idx(args.perp_root / "open_interest_hourly" / f"{sidecar_key}.parquet")
        fund_idx = _idx(args.perp_root / "funding_hourly" / f"{sidecar_key}.parquet")
        oi_idx = oi_idx[(oi_idx >= start) & (oi_idx <= end)]
        fund_idx = fund_idx[(fund_idx >= start) & (fund_idx <= end)]
        source_cache = {"open_interest": oi_idx, "funding": fund_idx, "oi_funding_interaction": oi_idx.intersection(fund_idx), "other": feature_idx}
        try:
            frame = pd.read_parquet(path, columns=["ts", *keys])
        except Exception:
            frame = pd.read_parquet(path)
        if "ts" in frame.columns:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
            frame = frame.set_index("ts")
        else:
            frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[~frame.index.isna()].sort_index()
        frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
        symbols.append(symbol)
        for key in keys:
            fam = _family(key)
            source_idx = source_cache[fam]
            overlap_rows, source_full, source_post_rows, source_post = _coverage(feature_idx, source_idx)
            present = key in frame.columns
            values = pd.to_numeric(frame[key], errors="coerce") if present else pd.Series(dtype=float)
            finite = np.isfinite(values.to_numpy(dtype=float, copy=False)) if present else np.zeros(len(frame), dtype=bool)
            finite_idx = frame.index[finite] if present else pd.DatetimeIndex([], tz="UTC")
            first_source = source_idx.min() if len(source_idx) else pd.NaT
            post_expected = feature_idx[feature_idx >= first_source] if pd.notna(first_source) else feature_idx[:0]
            post_finite = finite_idx[finite_idx >= first_source] if pd.notna(first_source) else finite_idx[:0]
            rows.append({
                "feature_symbol": symbol,
                "feature": key,
                "family": fam,
                "present": bool(present),
                "feature_rows": int(len(frame)),
                "finite_rows": int(finite.sum()),
                "finite_coverage": float(finite.mean()) if len(frame) else np.nan,
                "finite_post_rows": int(len(post_finite)),
                "finite_post_coverage": float(len(post_finite) / len(post_expected)) if len(post_expected) else np.nan,
                "source_rows": int(len(source_idx)),
                "source_overlap_rows": overlap_rows,
                "source_full_coverage": source_full,
                "source_post_overlap_rows": source_post_rows,
                "source_post_coverage": source_post,
                "source_first_ts": source_idx.min().isoformat() if len(source_idx) else None,
                "source_last_ts": source_idx.max().isoformat() if len(source_idx) else None,
                "diagnosis": None,
            })
    table = pd.DataFrame(rows)
    table["diagnosis"] = table.apply(_diagnosis, axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out_dir / "oi_funding_feature_sparsity.parquet", index=False, compression="zstd")
    summary = table.groupby(["feature", "family", "diagnosis"], as_index=False).agg(
        assets=("feature_symbol", "nunique"),
        present_assets=("present", "sum"),
        median_finite_coverage=("finite_coverage", "median"),
        median_finite_post_coverage=("finite_post_coverage", "median"),
        median_source_post_coverage=("source_post_coverage", "median"),
        weighted_finite_coverage=("finite_rows", "sum"),
        total_feature_rows=("feature_rows", "sum"),
    )
    summary["weighted_finite_coverage"] = summary["weighted_finite_coverage"] / summary["total_feature_rows"].clip(lower=1)
    summary.to_parquet(args.out_dir / "oi_funding_feature_sparsity_summary.parquet", index=False, compression="zstd")
    diagnosis_counts = table["diagnosis"].value_counts().to_dict()
    payload = {
        "schema": "oi_funding_feature_sparsity_audit_v1",
        "feature_dir": str(args.feature_dir),
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "symbols": len(symbols),
        "keys": len(keys),
        "rows": int(len(table)),
        "diagnosis_counts": {str(k): int(v) for k, v in diagnosis_counts.items()},
        "classification": {
            "upstream_source_sparse": "relevant canonical source is <90% complete after its first observation",
            "derived_warmup_or_eligibility": "source is >=90% complete post-first-source but derived feature is not",
            "constructor_or_dependency_failure": "feature is absent/all-null despite >=90% relevant source coverage",
            "pre_source_unavailable": "finite only after source begins; pre-source rows are not failures",
            "unsupported_declared": "explicitly fail-closed unsupported key",
        },
        "artifacts": ["oi_funding_feature_sparsity.parquet", "oi_funding_feature_sparsity_summary.parquet"],
    }
    (args.out_dir / "sparsity_audit_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "symbols": len(symbols), "keys": len(keys), "rows": len(table), "diagnosis_counts": payload["diagnosis_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
