#!/usr/bin/env python3
"""Materialise causal market-profile and channel geometry research inputs.

The producer reads archived 15-minute OHLCV and strictly-prior OI only.  It
does not import any live code, fit a model, call an exchange, or qualify a
candidate by the availability of profile data.  A missing profile/OI source is
recorded as an unavailable optional snapshot, not repaired from a later bar.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_profile_geometry import (
    ANCHORED_VWAP_FEATURES,
    CausalProfileGeometryEngine,
    ProfileGeometryConfig,
    VOLATILITY_PARTICIPATION_FEATURES,
)
from extreme_price_movements.causal_sr_engine import read_symbol_bars
from scripts.materialize_causal_oi_positioning import _oi_path, _strict_prior_oi


ENTRY_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_vwap_target_free_20260830_v1/target_free_15m_features.parquet"
BARS_ROOT = ROOT / "15m_ohlcv_perp"
OI_ROOT = ROOT / "data_perp/exchanges/krakenfutures/open_interest_hourly"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_profile_geometry_2025_train_2026_score_20260831_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _targets(entry: pd.DataFrame) -> dict[str, dict[pd.Timestamp, list[dict[str, object]]]]:
    result: dict[str, dict[pd.Timestamp, list[dict[str, object]]]] = {}
    for candidate_id, decision_ts, symbol in entry.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]].itertuples(index=False, name=None):
        ts = pd.Timestamp(decision_ts)
        result.setdefault(str(symbol), {}).setdefault(ts, []).append(
            {"target_kind": "entry", "target_id": str(candidate_id), "candidate_id": str(candidate_id)}
        )
    return result


def _worker(payload: tuple[str, dict[pd.Timestamp, list[dict[str, object]]], str, str, str, str, str, dict[str, object]]) -> dict[str, object]:
    symbol, targets, bars_root_raw, oi_root_raw, output_start_raw, output_end_raw, parts_raw, config_raw = payload
    try:
        bars = read_symbol_bars(Path(bars_root_raw), symbol)
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        config = ProfileGeometryConfig(**config_raw)
        oi, oi_rows = _strict_prior_oi(bars, _oi_path(Path(oi_root_raw), symbol), max_staleness_bars=4)
        bars["open_interest"] = oi
        events, snapshots, states = CausalProfileGeometryEngine(
            symbol, bars, output_start=pd.Timestamp(output_start_raw), output_end=pd.Timestamp(output_end_raw),
            snapshot_targets=targets, config=config,
        ).run()
        token = symbol.lower().replace("/", "_").replace(":", "_")
        parts = Path(parts_raw)
        if not events.empty:
            events.to_parquet(parts / f"{token}__events.parquet", index=False, compression="zstd")
        if not snapshots.empty:
            snapshots.to_parquet(parts / f"{token}__snapshots.parquet", index=False, compression="zstd")
        if not states.empty:
            states.to_parquet(parts / f"{token}__states.parquet", index=False, compression="zstd")
        return {
            "__symbol__": symbol, "source_available": True, "bar_rows": len(bars),
            "strict_prior_oi_rows": oi_rows, "oi_coverage": float(oi_rows / max(len(bars), 1)),
            "event_rows": len(events), "snapshot_rows": len(snapshots), "state_rows": len(states),
        }
    except Exception as exc:
        return {"__symbol__": symbol, "source_available": False, "exception_type": type(exc).__name__, "exception": str(exc)}


def _concat(parts: Path, suffix: str, output: Path) -> pd.DataFrame:
    paths = sorted(parts.glob(f"*__{suffix}.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()
    frame.to_parquet(output, index=False, compression="zstd")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-panel", type=Path, default=ENTRY_PANEL)
    parser.add_argument("--bars-root", type=Path, default=BARS_ROOT)
    parser.add_argument("--oi-root", type=Path, default=OI_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--output-end")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--config-json", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config_raw = json.loads(args.config_json.resolve().read_text(encoding="utf-8")) if args.config_json else {}
    config = ProfileGeometryConfig(**config_raw)
    entry = pd.read_parquet(args.entry_panel.resolve())
    entry["__decision_ts__"] = pd.to_datetime(entry.__decision_ts__, utc=True, errors="raise")
    target_map = _targets(entry)
    symbols = sorted(target_map)
    if args.max_symbols is not None:
        symbols = symbols[:args.max_symbols]
        target_map = {symbol: target_map[symbol] for symbol in symbols}
    output_start = pd.Timestamp(args.output_start)
    output_end = pd.Timestamp(args.output_end) if args.output_end else entry.__decision_ts__.max()
    output.mkdir(parents=True, exist_ok=False)
    parts = output / "parts"; parts.mkdir()
    payloads = [
        (symbol, target_map[symbol], str(args.bars_root.resolve()), str(args.oi_root.resolve()), str(output_start), str(output_end), str(parts), config_raw)
        for symbol in symbols
    ]
    coverage: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            item = future.result(); coverage.append(item); print(json.dumps(item, default=str), flush=True)
    coverage_frame = pd.DataFrame(coverage).sort_values("__symbol__", kind="stable")
    coverage_frame.to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    events = _concat(parts, "events", output / "profile_events.parquet")
    snapshots = _concat(parts, "snapshots", output / "profile_snapshots.parquet")
    states = _concat(parts, "states", output / "profile_hourly_states.parquet")
    if not snapshots.empty:
        snapshots.to_parquet(output / "entry_profile_snapshots.parquet", index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "schema": "causal-profile-geometry-v1",
        "scope": "offline causal research only; no live-model, execution, or exchange mutation",
        "contract": {
            "profile": "rolling completed-hour 21-day volume/time/OI-at-price profile on a fixed 25-bps logarithmic price grid",
            "levels": ["POC", "VAH", "VAL", "nearest HVN", "nearest LVN", "time-at-price balance region"],
            "positioning": "strictly-prior OI, max one-hour old, accumulated as signed dOI at price; missing OI stays unavailable",
            "geometry": "trailing completed-hour Bollinger(20,2), Keltner(EMA20,2ATR), and Donchian(20) geometry",
            "volatility_participation": {
                "features": list(VOLATILITY_PARTICIPATION_FEATURES),
                "definition": "trailing completed-hour 21-day ATR/volume percentiles plus 4h-to-24h realised-volatility, range and participation state",
            },
            "anchored_vwap": {
                "features": list(ANCHORED_VWAP_FEATURES),
                "definition": "completed-hour UTC-session and UTC-week volume-weighted anchors, their slopes, cross age and close-above-session-VWAP fraction",
            },
            "source_target": "next-8h long MFE-minus-MAE in ATR units plus adverse-break classification",
            "source_event_schedule": "every fourth completed hour; event labels resolve after eight completed future hours",
            "snapshots": "target-free candidate identities only; missing optional profile output never removes a candidate",
            "no_future_data": True,
            "config": config.__dict__,
        },
        "inputs": {
            "entry_panel": str(args.entry_panel.resolve()), "entry_panel_sha256": _sha256(args.entry_panel.resolve()),
            "bars_root": str(args.bars_root.resolve()), "oi_root": str(args.oi_root.resolve()), "symbols_requested": len(symbols),
        },
        "outputs": {"events": len(events), "snapshots": len(snapshots), "states": len(states), "source_ready_symbols": int(coverage_frame.source_available.sum())},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
