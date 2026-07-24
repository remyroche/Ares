#!/usr/bin/env python3
"""Append causal temporal-mechanism context to an existing compact store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
    NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
    add_negative_residual_features,
)


SOURCE_FEATURES = (
    "ret4h",
    "ret_resid_btc_4h",
    "mkt_ret_1h",
    "mkt_ret_4h",
    "log_realized_vol_cp_absratio_8_32",
    "market_dispersion_4h",
    "corr_eth_24h",
    "corr_btc_24h",
    "mkt_median_oi_chg_4h_rz",
    "mkt_median_oi_drawdown_from_peak_24h",
    "mkt_pct_oi_chg_4h_rz_lt_minus1",
    "funding_1d_chg_ts_resid",
    "range_per_volume",
    "price_recovery_from_low_24h_atr",
    "breadth_recovery_from_6h_min",
    "price_down_oi_down_4h_rz",
    "mkt_pct_price_up_oi_up_4h",
    "pct_assets_new_low_24h",
    "mkt_flush_exhaustion_score",
    "range_climax_decay_4h",
    "rv_24h_peer_resid",
    "market_downside_pairwise_corr_24h",
    "symbol_minus_mkt_ret_1h",
    "asset_short_covering_score",
    "oi_value_z_30d",
    "oi_value_1d_chg_z_90d",
)


def _symbol(path: Path) -> str:
    return path.stem.split("=", 1)[-1].replace("_", "/", 1)


def _extract_index(frame: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    if "ts" in frame.columns:
        raw = frame.pop("ts")
    else:
        raw = frame.index
    return (
        pd.DatetimeIndex(pd.to_datetime(raw, utc=True, errors="coerce")),
        frame,
    )


def _load_source_panels(
    root: Path,
    *,
    start: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    paths = sorted(root.glob("symbol=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No symbol feature files under {root}")
    reference = pd.read_parquet(paths[0], columns=["ts"])
    index, _ = _extract_index(reference)
    index = index.dropna().sort_values().unique()
    if len(index) < 2:
        raise ValueError(f"Insufficient timestamp coverage in {paths[0]}")
    # A single low-liquidity symbol can have intermittent feature rows. It must
    # not define the market-wide clock: use its cadence only, then construct a
    # regular range and let every symbol contribute at each available point.
    deltas = np.diff(index.asi8)
    deltas = deltas[deltas > 0]
    step_ns = int(np.median(deltas)) if len(deltas) else 0
    if step_ns <= 0:
        raise ValueError(f"Could not infer source cadence from {paths[0]}")
    index = pd.date_range(
        start=index.min(),
        end=index.max(),
        freq=pd.to_timedelta(step_ns, unit="ns"),
        tz="UTC",
    )
    if start is not None:
        index = index[index >= pd.Timestamp(start)]
    if index.empty:
        raise ValueError(f"No source rows remain after start={start}")
    symbols = [_symbol(path) for path in paths]
    arrays = {
        name: np.full((len(index), len(paths)), np.nan, dtype=np.float32)
        for name in SOURCE_FEATURES
    }
    for column, path in enumerate(paths):
        available = set(pq.read_schema(path).names)
        names = [name for name in SOURCE_FEATURES if name in available]
        if not names:
            continue
        frame = pd.read_parquet(path, columns=["ts", *names])
        local_index, frame = _extract_index(frame)
        if not local_index.equals(index):
            frame.index = local_index
            frame = frame.reindex(index)
        for name in names:
            arrays[name][:, column] = pd.to_numeric(
                frame[name], errors="coerce"
            ).to_numpy(np.float32, copy=False)
    columns = pd.Index(symbols)
    return {
        name: pd.DataFrame(values, index=index, columns=columns, copy=False)
        for name, values in arrays.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    target_files = sorted(args.target_root.glob("symbol=*.parquet"))
    if not target_files:
        raise FileNotFoundError(f"No compact files under {args.target_root}")
    if not args.force:
        complete = all(
            set(NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS).issubset(
                pq.read_schema(path).names
            )
            for path in target_files
        )
        if complete:
            return {
                "schema": "negative_residual_temporal_mechanism_backfill_v1",
                "skipped": True,
                "reason": "all target files already contain the feature contract",
            }
    panels = _load_source_panels(args.source_root)
    generated = add_negative_residual_features(
        panels,
        requested_feature_keys=NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS,
        cfg={"feature_bars_per_hour": 1},
    )
    expected = set(NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS)
    if generated != expected:
        raise RuntimeError(f"Generated feature mismatch: {sorted(expected - generated)}")
    written = 0
    for path in target_files:
        frame = pd.read_parquet(path)
        if "ts" in frame.columns:
            index = pd.DatetimeIndex(
                pd.to_datetime(frame.pop("ts"), utc=True, errors="coerce")
            )
        else:
            index = pd.DatetimeIndex(
                pd.to_datetime(frame.index, utc=True, errors="coerce")
            )
        frame.index = index
        frame.index.name = "ts"
        symbol = _symbol(path)
        for name in NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS:
            source = panels[name]
            if symbol not in source.columns:
                raise KeyError(f"{symbol} is absent from source universe")
            frame[name] = source[symbol].reindex(index).to_numpy(
                dtype=np.float32, copy=False
            )
        frame.to_parquet(path, index=True, compression="zstd")
        written += 1
    reference_symbol = _symbol(target_files[0])
    manifest = {
        "schema": "negative_residual_temporal_mechanism_backfill_v1",
        "schema_version": NEGATIVE_RESIDUAL_FEATURE_SCHEMA_VERSION,
        "source_root": str(args.source_root),
        "target_root": str(args.target_root),
        "source_symbols": len(panels[SOURCE_FEATURES[0]].columns),
        "target_files": len(target_files),
        "written_files": written,
        "rows": len(panels[SOURCE_FEATURES[0]]),
        "feature_keys": list(NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS),
        "finite_fraction": {
            name: float(panels[name][reference_symbol].notna().mean())
            for name in NEGATIVE_RESIDUAL_TEMPORAL_MECHANISM_FEATURE_KEYS
        },
    }
    (args.target_root / "temporal_mechanism_backfill_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
