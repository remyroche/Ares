#!/usr/bin/env python3
"""Create the full-universe, barrier-exit panel required for T2/T4 ablations.

Unlike the retired selected-monitor ledgers, this emits both long and short
rows at every eligible hourly feature cutoff for every symbol with continuous
one-minute price data.  Entries occur one completed hour after the feature
close.  Outcomes are settled at the first TP/SL hit, or at H12 timeout only
when neither barrier is reached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import load_ae_gmm_state_artifact, transform_ae_gmm_features
from extreme_price_movements.config import (
    DAILY_SR_BASE_FEATURE_KEYS,
    LONG_HORIZON_PERP_META_FEATURE_KEYS,
    MODEL_DIRECT_BASE_FEATURE_KEYS,
    MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS,
    ORDERBOOK_BASE_FEATURE_KEYS,
    ORDERBOOK_META_FEATURE_KEYS,
    RESIDUAL_BASE_FEATURE_KEYS,
    RESIDUAL_META_FEATURE_KEYS,
    VOLUME_FREE_PERP_BASE_FEATURE_KEYS,
    VOLUME_FREE_PERP_META_FEATURE_KEYS,
)


ONE_MINUTE = ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv"
FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
CONTRACT = ROOT / "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/frozen_raw_causal_features.json"
LONG_STATE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1/long/ae_gmm/ae_gmm_state.pkl"
SHORT_STATE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1/short/ae_gmm/ae_gmm_state.pkl"
GEOMETRIES = ((2.0, 1.0), (2.0, 2.0), (3.0, 1.0), (3.0, 2.0))
# The historical studies use a 1% all-in round trip as the conservative cost
# floor.  Minute OHLC has no contemporaneous quote/spread series, therefore
# this is deliberately an explicit *assumption*, never a fake causal field.
DEFAULT_ROUND_TRIP_COST_BPS = 100.0
LATENT = ("AE_reconstruction_error", "mahalanobis_distance", "cluster_acceleration", "cluster_entropy", "cluster_entropy_accel_1", "cluster_entropy_delta_1", "cluster_entropy_norm", "cluster_flip_count_20", "cluster_speed", "cluster_t")


def _layer_feature_universe() -> tuple[list[str], list[str]]:
    """Return only existing config-owned base/meta pools, never a generic list."""
    base = list(dict.fromkeys(MODEL_DIRECT_BASE_FEATURE_KEYS + RESIDUAL_BASE_FEATURE_KEYS + ORDERBOOK_BASE_FEATURE_KEYS + VOLUME_FREE_PERP_BASE_FEATURE_KEYS + DAILY_SR_BASE_FEATURE_KEYS))
    meta = list(dict.fromkeys(MODEL_REGIME_COMPOSITE_META_FEATURE_KEYS + RESIDUAL_META_FEATURE_KEYS + ORDERBOOK_META_FEATURE_KEYS + VOLUME_FREE_PERP_META_FEATURE_KEYS + LONG_HORIZON_PERP_META_FEATURE_KEYS))
    return base, meta


def _atr(hourly: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = hourly.close.shift(1)
    true_range = pd.concat([(hourly.high - hourly.low), (hourly.high - previous).abs(), (hourly.low - previous).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _temporal_latents(point: pd.DataFrame) -> pd.DataFrame:
    out = point.copy()
    ts = pd.DatetimeIndex(out.index)
    contiguous = np.r_[False, np.diff(ts.asi8) == pd.Timedelta(hours=1).value]
    entropy = out["cluster_entropy"].to_numpy(np.float32, copy=False)
    labels = out["cluster_t"].to_numpy(np.float32, copy=False)
    delta = np.zeros(len(out), np.float32); delta[1:] = entropy[1:] - entropy[:-1]; delta[~contiguous] = 0.0
    accel = np.zeros(len(out), np.float32); accel[1:] = delta[1:] - delta[:-1]; accel[~contiguous] = 0.0
    probs = out[[f"gmm_prob_{i}" for i in range(12)]].to_numpy(np.float32, copy=False)
    pdiff = np.zeros_like(probs); pdiff[1:] = probs[1:] - probs[:-1]
    speed = np.sqrt((pdiff * pdiff).sum(axis=1)).astype(np.float32); speed[~contiguous] = 0.0
    speed_accel = np.zeros(len(out), np.float32); speed_accel[1:] = speed[1:] - speed[:-1]; speed_accel[~contiguous] = 0.0
    changed = np.zeros(len(out), np.int32); changed[1:] = (labels[1:] != labels[:-1]) & contiguous[1:]
    starts = np.maximum(0, np.arange(len(out)) - 19); csum = np.r_[0, np.cumsum(changed, dtype=np.int64)]
    out["cluster_entropy_delta_1"] = delta; out["cluster_entropy_accel_1"] = accel
    out["cluster_speed"] = speed; out["cluster_acceleration"] = speed_accel
    out["cluster_flip_count_20"] = (csum[np.arange(len(out)) + 1] - csum[starts + 1]).astype(np.float32)
    return out


@njit(cache=True)
def _first_hits_all(high, low, close, starts, entry, atr, side):
    """Evaluate the four fixed contracts in one compiled H12 path pass."""
    tps = np.array([2.0, 2.0, 3.0, 3.0])
    sls = np.array([1.0, 2.0, 1.0, 2.0])
    n = len(starts)
    events = np.full((4, n), 2, np.int8)
    exits = np.full((4, n), 720, np.int16)
    pnl = np.empty((4, n), np.float32)
    pnl[:] = np.nan
    path_mfe = np.empty(n, np.float32)
    path_mae = np.empty(n, np.float32)
    path_mfe[:] = np.nan
    path_mae[:] = np.nan
    for row in range(n):
        start = starts[row]
        if start < 0 or start + 719 >= len(close) or not np.isfinite(atr[row]) or atr[row] <= 0.0 or not np.isfinite(entry[row]):
            continue
        unresolved = np.ones(4, np.uint8)
        max_favorable = -np.inf
        max_adverse = -np.inf
        for offset in range(720):
            pos = start + offset
            if side > 0:
                favorable = (high[pos] - entry[row]) / atr[row]
                adverse = (entry[row] - low[pos]) / atr[row]
            else:
                favorable = (entry[row] - low[pos]) / atr[row]
                adverse = (high[pos] - entry[row]) / atr[row]
            if favorable > max_favorable:
                max_favorable = favorable
            if adverse > max_adverse:
                max_adverse = adverse
            for geometry in range(4):
                if unresolved[geometry] == 1:
                    # Same-minute TP/SL conflicts are adverse by contract.
                    if adverse >= sls[geometry]:
                        events[geometry, row] = 1
                        exits[geometry, row] = offset + 1
                        pnl[geometry, row] = -sls[geometry]
                        unresolved[geometry] = 0
                    elif favorable >= tps[geometry]:
                        events[geometry, row] = 0
                        exits[geometry, row] = offset + 1
                        pnl[geometry, row] = tps[geometry]
                        unresolved[geometry] = 0
        for geometry in range(4):
            if unresolved[geometry] == 1:
                pnl[geometry, row] = side * (close[start + 719] - entry[row]) / atr[row]
        path_mfe[row] = max_favorable
        path_mae[row] = max_adverse
    return events, exits, pnl, path_mfe, path_mae


def _symbols() -> list[str]:
    result = []
    for directory in ONE_MINUTE.glob("symbol=*"):
        symbol = directory.name.removeprefix("symbol=")
        if all((directory / f"year={year}").exists() for year in ("2023", "2024", "2025")) and (FEATURE_STORE / f"symbol={symbol}.parquet").exists():
            result.append(symbol)
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-04-01")
    parser.add_argument("--end", default="2024-12-01", help="exclusive feature-close timestamp")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-symbols", type=int, default=0, help="nonzero is a smoke-test cap")
    parser.add_argument("--symbol", action="append", default=[], help="process an explicit symbol; repeatable")
    parser.add_argument("--resume", action="store_true", help="resume a previously interrupted output directory")
    parser.add_argument("--round-trip-cost-bps", type=float, default=DEFAULT_ROUND_TRIP_COST_BPS)
    args = parser.parse_args()
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(args.out_dir)
    start, end = (pd.Timestamp(value, tz="UTC") for value in (args.start, args.end))
    frozen_fields = json.loads(CONTRACT.read_text())["raw_feature_columns"]
    base_pool, meta_pool = _layer_feature_universe()
    # The former 361-field contract was insufficient for the required
    # 30–40-feature base subset.  Materialise the pre-existing, causal config
    # pools; selection still happens strictly inside each layer afterwards.
    fields = list(dict.fromkeys(frozen_fields + base_pool + meta_pool))
    static = [field for field in fields if field not in LATENT]
    states = {"long": load_ae_gmm_state_artifact(LONG_STATE), "short": load_ae_gmm_state_artifact(SHORT_STATE)}
    source_cols = set(static)
    for state in states.values(): source_cols.update(str(c) for c in state.get("feature_columns", []) if str(c) != "side")
    all_symbols = _symbols()
    symbols = all_symbols
    if args.symbol:
        requested = set(args.symbol)
        symbols = [symbol for symbol in symbols if symbol in requested]
        missing = requested.difference(symbols)
        if missing:
            raise ValueError(f"requested symbols are not full-universe eligible: {sorted(missing)}")
    if args.max_symbols: symbols = symbols[:int(args.max_symbols)]
    stage = args.out_dir
    stage.mkdir(parents=True, exist_ok=True)
    parts = stage / "parts"; parts.mkdir(exist_ok=True)
    reports = []
    try:
        for number, symbol in enumerate(symbols, start=1):
            destination = parts / f"symbol={symbol.replace('/', '_')}.parquet"
            if args.resume and destination.exists():
                try:
                    existing = pd.read_parquet(destination, columns=["candidate_id", "t2_path_mfe_atr", "t2_path_mae_atr", "assumed_round_trip_cost_bps"])
                    if len(existing):
                        reports.append({"symbol": symbol, "rows": int(len(existing)), "status": "reused"})
                        continue
                except Exception:
                    destination.unlink()
            minute_root = ONE_MINUTE / f"symbol={symbol}"
            minute_set = ds.dataset(minute_root, format="parquet", partitioning="hive")
            required_years = list(range((start - pd.Timedelta(hours=15)).year, (end + pd.Timedelta(hours=12)).year + 1))
            minute = minute_set.to_table(
                filter=(ds.field("year").isin(required_years)) & (ds.field("ts") >= start - pd.Timedelta(hours=15)) & (ds.field("ts") < end + pd.Timedelta(hours=13)),
                columns=["ts", "open", "high", "low", "close"],
            ).to_pandas()
            minute["ts"] = pd.to_datetime(minute.ts, utc=True); minute = minute.drop_duplicates("ts").set_index("ts").sort_index()
            hourly = minute.resample("1h", label="left", closed="left").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
            hourly["atr_1h"] = _atr(hourly)
            feature_path = FEATURE_STORE / f"symbol={symbol}.parquet"
            # The historical feature store is not perfectly rectangular.
            # Retain the frozen output schema below; absent source fields stay
            # missing and are caught by the later >=90% coverage gate instead
            # of making the full-universe build arbitrarily exclude a symbol.
            available_source_cols = set(pq.ParquetFile(feature_path).schema_arrow.names)
            read_cols = sorted(source_cols.intersection(available_source_cols))
            feature = pd.read_parquet(feature_path, columns=read_cols, filters=[("ts", ">=", start), ("ts", "<", end)])
            feature.index = pd.to_datetime(feature.index, utc=True)
            feature = feature.reindex(hourly.index.intersection(feature.index)).sort_index()
            feature["atr_1h"] = hourly.reindex(feature.index).atr_1h
            eligible = feature.loc[feature.atr_1h.gt(0) & feature.index.to_series().between(start, end - pd.Timedelta(hours=13))].copy()
            if eligible.empty: continue
            rows = []
            for side_name, side in (("long", 1.0), ("short", -1.0)):
                state = states[side_name]; inputs = [str(c) for c in state.get("feature_columns", [])]
                x = feature.reindex(columns=[c for c in inputs if c != "side"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if "side" in inputs: x["side"] = np.float32(side)
                transformed = transform_ae_gmm_features(x.reindex(columns=inputs), state, index=x.index)
                point = pd.DataFrame(index=x.index)
                for column in ("AE_reconstruction_error", "mahalanobis_distance", "cluster_entropy", "cluster_entropy_norm", "cluster_t"): point[column] = transformed[column]
                for index in range(12): point[f"gmm_prob_{index}"] = transformed[f"gmm_prob_{index}"]
                point = _temporal_latents(point)
                base = eligible.reindex(columns=static).copy(); latent = point.reindex(base.index).reindex(columns=LATENT)
                result = pd.concat([base, latent], axis=1)
                decision = result.index + pd.Timedelta(hours=1)
                entry = minute.reindex(decision).open.to_numpy(np.float64)
                atr = eligible.atr_1h.to_numpy(np.float64)
                result.insert(0, "candidate_id", [f"{symbol}|{ts.isoformat()}|1h|{side_name}" for ts in result.index])
                result.insert(1, "__ts__", result.index); result.insert(2, "__symbol__", symbol); result.insert(3, "side_name", side_name)
                result.insert(4, "__decision_ts__", decision); result["atr_1h"] = atr; result["decision_price"] = entry
                result["assumed_round_trip_cost_bps"] = np.float32(args.round_trip_cost_bps)
                event_grid, exit_grid, pnl_grid, path_mfe, path_mae = _first_hits_all(
                    minute.high.to_numpy(np.float64, copy=False), minute.low.to_numpy(np.float64, copy=False), minute.close.to_numpy(np.float64, copy=False),
                    minute.index.get_indexer(pd.DatetimeIndex(decision)), entry, atr, side,
                )
                for geometry_index, (tp, sl) in enumerate(GEOMETRIES):
                    key = f"tp{tp:g}_sl{sl:g}".replace(".", "p")
                    result[f"t2_{key}_event"] = event_grid[geometry_index]
                    result[f"t2_{key}_exit_minute"] = exit_grid[geometry_index]
                    result[f"t4_{key}_exit_pnl_atr"] = pnl_grid[geometry_index]
                    gross_bps = pnl_grid[geometry_index] * atr / entry * 10_000.0
                    result[f"t4_{key}_gross_bps"] = gross_bps.astype(np.float32)
                    result[f"t4_{key}_net_bps"] = (gross_bps - args.round_trip_cost_bps).astype(np.float32)
                # These entry-time-normalised path summaries make the soft
                # triple-barrier target auditable: near misses and adverse
                # excursions are observed inside exactly the same H12 path.
                result["t2_path_mfe_atr"] = path_mfe
                result["t2_path_mae_atr"] = path_mae
                result["__label_available_at__"] = decision + pd.Timedelta(hours=12)
                # Full-universe means every eligible *observable* row, not
                # fabricated timeout labels across missing one-minute history.
                complete = result["decision_price"].notna()
                for tp, sl in GEOMETRIES:
                    key = f"tp{tp:g}_sl{sl:g}".replace(".", "p")
                    complete &= result[f"t4_{key}_exit_pnl_atr"].notna()
                result = result.loc[complete].copy()
                rows.append(result.reset_index(drop=True))
            out = pd.concat(rows, ignore_index=True)
            out.to_parquet(destination, index=False, compression="zstd")
            reports.append({"symbol": symbol, "rows": len(out), "signal_start": str(eligible.index.min()), "signal_end": str(eligible.index.max())})
            print(json.dumps({"event": "symbol_complete", "number": number, **reports[-1]}), flush=True)
        complete = not args.symbol and not args.max_symbols and len(reports) == len(all_symbols)
        manifest = {"schema": "full_universe_t2_t4_panel_v2", "complete": complete, "universe": "all hourly long+short rows for symbols with continuous 2023-25 1m data and static features", "not_selected_for_monitor": True, "entry": "signal close + 1h, next one-minute open", "exit": "first TP/SL hit; adverse tie precedence; H12 timeout only if neither hit", "cost": {"assumed_round_trip_cost_bps": args.round_trip_cost_bps, "source": "declared fixed conservative 1% assumption; no synthetic row-level spread estimate"}, "geometries": [{"tp_atr": tp, "sl_atr": sl} for tp, sl in GEOMETRIES], "features": {"count": len(fields), "causal_static": len(static), "base_config_pool": len(base_pool), "meta_config_pool": len(meta_pool), "approved_later_aegmm": True}, "start": str(start), "end_exclusive": str(end), "symbols": reports, "parts": str(parts)}
        # A one-symbol smoke run must never masquerade as the completed
        # universe.  The final manifest is emitted only after every eligible
        # symbol has been processed; checkpoints get their own status file.
        manifest_name = "manifest.json" if complete else "checkpoint_manifest.json"
        (stage / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        # Completed partitions are intentional checkpoints; retain them for
        # ``--resume`` rather than discarding hours of full-universe work.
        raise


if __name__ == "__main__":
    main()
