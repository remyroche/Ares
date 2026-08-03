#!/usr/bin/env python3
"""Materialize causal, observed liquidity inputs for the regime panel.

The output is an hourly sidecar.  It deliberately keeps market-wide fields
and cross-product aggregates separate: a replicated market field is not
presented as an independently aggregated asset statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MARKET_FIELDS = (
    "median_spread_bps",
    "mkt_quote_volume_z_24h",
    "mkt_volume_z_24h",
    "pct_assets_high_rvol",
    "pct_assets_wide_spread",
    "market_dispersion_1h",
    "market_dispersion_4h",
    "market_dispersion_24h",
    "xs_dispersion__amihud_illiq",
    "xs_dispersion__rvol_z",
    "xasset_mkt_spread_bps",
    "xasset_mkt_depth_z",
    "xasset_mkt_depth_to_qv_z",
)

ASSET_FIELDS = (
    "amihud_illiq",
    "range_per_volume",
    "log_quote_volume",
    "quote_volume_z_30d",
    "rvol_z",
    "ob_spread_bps_z_24h",
    "ob_depth_usd_l20_z",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        source = frame.index
    elif "ts" in frame:
        source = frame["ts"]
    else:
        raise ValueError(
            f"feature frame has neither a DatetimeIndex nor ts column: "
            f"index={frame.index.name!r}"
        )
    return pd.DatetimeIndex(pd.to_datetime(source, utc=True))


def build_liquidity_sidecar(
    *,
    calendar: pd.DataFrame,
    feature_paths: list[Path],
    market_reference_path: Path,
) -> tuple[pd.DataFrame, dict]:
    if "source_utc" not in calendar:
        raise ValueError("calendar must contain source_utc")
    target = pd.DatetimeIndex(pd.to_datetime(calendar["source_utc"], utc=True))
    if target.has_duplicates or not target.is_monotonic_increasing:
        raise ValueError("source_utc must be unique and increasing")

    available = set(pq.ParquetFile(market_reference_path).schema_arrow.names)
    missing = sorted(set(MARKET_FIELDS) - available)
    if missing:
        raise ValueError(f"market reference lacks required fields: {missing}")
    market = pd.read_parquet(market_reference_path, columns=list(MARKET_FIELDS))
    market.index = _utc_index(market)
    market = market.reindex(target)
    market.columns = [f"liquidity_market__{c}" for c in market.columns]

    n = len(target)
    sums = {c: np.zeros(n, dtype=np.float64) for c in ASSET_FIELDS}
    sumsq = {c: np.zeros(n, dtype=np.float64) for c in ASSET_FIELDS}
    counts = {c: np.zeros(n, dtype=np.int32) for c in ASSET_FIELDS}
    maxima = {c: np.full(n, -np.inf, dtype=np.float64) for c in ASSET_FIELDS}
    used: list[str] = []
    skipped: dict[str, list[str]] = {}

    for path in sorted(feature_paths):
        names = set(pq.ParquetFile(path).schema_arrow.names)
        absent = sorted(set(ASSET_FIELDS) - names)
        if absent:
            skipped[str(path)] = absent
            continue
        frame = pd.read_parquet(path, columns=list(ASSET_FIELDS))
        frame.index = _utc_index(frame)
        frame = frame.reindex(target)
        for col in ASSET_FIELDS:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            finite = np.isfinite(values)
            clipped = np.where(finite, values, 0.0)
            sums[col] += clipped
            sumsq[col] += clipped * clipped
            counts[col] += finite
            maxima[col] = np.maximum(maxima[col], np.where(finite, values, -np.inf))
        used.append(str(path))

    if not used:
        raise ValueError("no feature files contain the required asset fields")

    out = pd.DataFrame({"source_utc": target})
    out["calendar_segment_id"] = calendar["calendar_segment_id"].to_numpy()
    for col in ASSET_FIELDS:
        count = counts[col]
        denom = np.maximum(count, 1)
        mean = sums[col] / denom
        variance = np.maximum(sumsq[col] / denom - mean * mean, 0.0)
        out[f"liquidity_xs__{col}__mean"] = np.where(count > 0, mean, np.nan)
        out[f"liquidity_xs__{col}__std"] = np.where(
            count > 1, np.sqrt(variance), np.nan
        )
        out[f"liquidity_xs__{col}__max"] = np.where(
            count > 0, maxima[col], np.nan
        )
        out[f"liquidity_xs__{col}__coverage"] = count / float(len(used))
    out = out.join(market.reset_index(drop=True))

    manifest = {
        "contract": {
            "causality": "exact timestamp joins only; no backward/forward fill",
            "market_fields": list(MARKET_FIELDS),
            "asset_fields": list(ASSET_FIELDS),
            "aggregate_statistics": ["mean", "std", "max", "coverage"],
            "regime_transition_separation": (
                "liquidity inputs are shared observable predictors; they do not "
                "merge regime-state and transition-state targets or probabilities"
            ),
        },
        "counts": {
            "rows": int(len(out)),
            "feature_files_seen": int(len(feature_paths)),
            "feature_files_used": int(len(used)),
            "feature_files_skipped": int(len(skipped)),
            "output_fields": int(out.shape[1] - 2),
        },
        "sources": {
            "market_reference": str(market_reference_path),
            "market_reference_sha256": _sha256(market_reference_path),
            "used_feature_files": used,
            "skipped_feature_files": skipped,
        },
    }
    return out, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--market-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    calendar_path = Path(args.calendar).resolve()
    feature_dir = Path(args.feature_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet(
        calendar_path, columns=["source_utc", "calendar_segment_id"]
    )
    paths = sorted(feature_dir.glob("symbol=*.parquet"))
    sidecar, manifest = build_liquidity_sidecar(
        calendar=calendar,
        feature_paths=paths,
        market_reference_path=Path(args.market_reference).resolve(),
    )
    output_path = output_dir / "hourly_liquidity_enrichment.parquet"
    sidecar.to_parquet(output_path, index=False)
    manifest["sources"]["calendar"] = str(calendar_path)
    manifest["sources"]["calendar_sha256"] = _sha256(calendar_path)
    manifest["output"] = {
        "path": str(output_path),
        "sha256": _sha256(output_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.sha256").write_text(_sha256(manifest_path) + "\n")


if __name__ == "__main__":
    main()
