#!/usr/bin/env python3
"""Build a broad timestamp/symbol/strategy sample ledger for final-fit scoring.

This ledger is the input to ``generate_live_finalfit_oos_predictions.py
--sample-ledger``.  It deliberately contains only causal decision keys:
timestamp, symbol, strategy_id, and head.  Scores and realized outcomes are
added by the downstream final-fit scorer and native candidate materializer.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGY_IDS = {
    "long_bars": (
        "long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115"
        "_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039"
        "_variance_ratio_10_48_0_92117828"
    ),
    "long_dist": (
        "long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579"
        "_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735"
    ),
    "short_asset": (
        "short_asset_minus_mkt_oi_1d_peer_resid_0_34164831"
        "_oi_expansion_compression_balance_24h_0_42287597"
    ),
    "short_boll": (
        "short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385"
        "_price_rv_15d_robust_z_0_060036644"
    ),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _symbol_from_path(path: Path) -> str:
    name = path.name
    if not name.startswith("symbol=") or not name.endswith(".parquet"):
        return path.stem
    return name[len("symbol=") : -len(".parquet")].replace("_", "/", 1)


def _normalise_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _index_range(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    # Reading one lightweight column preserves the parquet datetime index.
    try:
        frame = pd.read_parquet(path, columns=[])
    except Exception:
        frame = pd.read_parquet(path)
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce")).dropna()
    if idx.empty:
        return None, None, 0
    return pd.Timestamp(idx.min()), pd.Timestamp(idx.max()), int(idx.nunique())


def _load_symbol_timestamps(path: Path, *, start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    try:
        frame = pd.read_parquet(path, columns=[])
    except Exception:
        frame = pd.read_parquet(path)
    idx = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True, errors="coerce")).dropna().unique().sort_values()
    if idx.empty:
        return idx
    idx = idx[(idx >= start) & (idx <= end)]
    if not freq:
        return idx
    step = pd.Timedelta(freq)
    origin = start.floor(freq)
    elapsed = ((idx - origin).total_seconds() / max(step.total_seconds(), 1.0)).round(9)
    keep = np.isclose(elapsed, np.round(elapsed), atol=1e-9)
    return idx[keep]


def build_sample_ledger(
    *,
    feature_store: Path,
    output_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    heads: list[str],
    freq: str,
    symbol_limit: int,
    min_symbol_coverage_days: float,
) -> dict[str, Any]:
    paths = sorted(feature_store.glob("symbol=*.parquet"))
    if not paths:
        raise RuntimeError(f"No feature parquet files found in {feature_store}")
    head_to_strategy = {head: STRATEGY_IDS[head] for head in heads}
    symbol_rows: list[dict[str, Any]] = []
    eligible_paths: list[Path] = []
    for path in paths:
        ts_min, ts_max, ts_count = _index_range(path)
        if ts_min is None or ts_max is None:
            continue
        overlap_start = max(ts_min, start)
        overlap_end = min(ts_max, end)
        overlap_days = max((overlap_end - overlap_start).total_seconds() / 86400.0, 0.0)
        eligible = overlap_days >= float(min_symbol_coverage_days)
        symbol_rows.append(
            {
                "symbol": _symbol_from_path(path),
                "path": str(path),
                "timestamp_min": ts_min,
                "timestamp_max": ts_max,
                "timestamp_count": ts_count,
                "overlap_days": overlap_days,
                "eligible": eligible,
            }
        )
        if eligible:
            eligible_paths.append(path)
    eligible_paths = sorted(
        eligible_paths,
        key=lambda path: next(
            row["overlap_days"] for row in symbol_rows if row["path"] == str(path)
        ),
        reverse=True,
    )
    if symbol_limit > 0:
        eligible_paths = eligible_paths[: int(symbol_limit)]
    if not eligible_paths:
        raise RuntimeError("No symbols met the requested coverage constraints")

    parts: list[pd.DataFrame] = []
    for path in eligible_paths:
        symbol = _symbol_from_path(path)
        timestamps = _load_symbol_timestamps(path, start=start, end=end, freq=freq)
        if timestamps.empty:
            continue
        base = pd.DataFrame({"timestamp": timestamps})
        base["symbol"] = symbol
        for head, strategy_id in head_to_strategy.items():
            part = base.copy()
            part["head"] = head
            part["strategy_id"] = strategy_id
            parts.append(part)
    if not parts:
        raise RuntimeError("No sample rows were generated")
    ledger = pd.concat(parts, ignore_index=True, copy=False)
    ledger = ledger.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    ledger = ledger.drop_duplicates(["timestamp", "strategy_id", "symbol"], keep="last")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(output_path, index=False)
    ledger.to_csv(output_path.with_suffix(".csv"), index=False)
    coverage = pd.DataFrame(symbol_rows)
    coverage_path = output_path.with_name(output_path.stem + "_symbol_coverage.parquet")
    coverage.to_parquet(coverage_path, index=False)
    coverage.to_csv(coverage_path.with_suffix(".csv"), index=False)
    ts = pd.to_datetime(ledger["timestamp"], utc=True, errors="coerce")
    manifest = {
        "generated_by": "build_finalfit_broad_sample_ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_store": str(feature_store),
        "output_path": str(output_path),
        "symbol_coverage_path": str(coverage_path),
        "start": start,
        "end": end,
        "frequency": freq,
        "requested_heads": heads,
        "strategy_ids": head_to_strategy,
        "symbol_limit": int(symbol_limit),
        "min_symbol_coverage_days": float(min_symbol_coverage_days),
        "feature_symbol_files": int(len(paths)),
        "eligible_symbol_files": int(len(eligible_paths)),
        "rows": int(len(ledger)),
        "timestamp_min": pd.Timestamp(ts.min()),
        "timestamp_max": pd.Timestamp(ts.max()),
        "timestamp_count": int(ts.nunique()),
        "symbol_count": int(ledger["symbol"].nunique()),
        "head_count": int(ledger["head"].nunique()),
        "heads": sorted(ledger["head"].dropna().astype(str).unique().tolist()),
    }
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--head", action="append", default=[])
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--symbol-limit", type=int, default=0)
    parser.add_argument("--min-symbol-coverage-days", type=float, default=150.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads = args.head or list(STRATEGY_IDS)
    unknown = sorted(set(heads) - set(STRATEGY_IDS))
    if unknown:
        raise SystemExit(f"Unknown heads: {unknown}")
    manifest = build_sample_ledger(
        feature_store=args.feature_store,
        output_path=args.output_path,
        start=_normalise_ts(args.start),
        end=_normalise_ts(args.end),
        heads=list(heads),
        freq=str(args.freq),
        symbol_limit=int(args.symbol_limit),
        min_symbol_coverage_days=float(args.min_symbol_coverage_days),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
