#!/usr/bin/env python3
"""Materialise causal directional OI-positioning zones from archived inputs.

The OI observation used at a 15-minute timestamp is strictly earlier than the
timestamp and expires after one hour.  Missing or stale OI therefore produces
an unavailable positioning snapshot rather than an imputed zone.  This script
is offline research only and never imports live execution code.
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

from extreme_price_movements.causal_oi_positioning import CausalOIPositioningEngine, OIPositioningConfig
from extreme_price_movements.causal_sr_engine import read_symbol_bars


ENTRY_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_vwap_target_free_20260830_v1/target_free_15m_features.parquet"
BARS_ROOT = ROOT / "15m_ohlcv_perp"
OI_ROOT = ROOT / "data_perp/exchanges/krakenfutures/open_interest_hourly"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_oi_positioning_2025_train_2026_score_20260831_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _oi_path(root: Path, symbol: str) -> Path:
    return root / f"{symbol.replace('/', '_').replace(':', '_').upper()}.parquet"


def _targets(entry: pd.DataFrame) -> dict[str, dict[pd.Timestamp, list[dict[str, object]]]]:
    result: dict[str, dict[pd.Timestamp, list[dict[str, object]]]] = {}
    for candidate_id, decision_ts, symbol in entry.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]].itertuples(index=False, name=None):
        ts = pd.Timestamp(decision_ts)
        result.setdefault(str(symbol), {}).setdefault(ts, []).append(
            {"target_kind": "entry", "target_id": str(candidate_id), "candidate_id": str(candidate_id)}
        )
    return result


def _strict_prior_oi(bars: pd.DataFrame, oi_path: Path, max_staleness_bars: int) -> tuple[pd.Series, int]:
    if not oi_path.exists():
        return pd.Series(np.nan, index=bars.index, dtype="float64"), 0
    oi = pd.read_parquet(oi_path)
    if "open_interest" not in oi.columns:
        return pd.Series(np.nan, index=bars.index, dtype="float64"), 0
    oi = oi.loc[:, ["open_interest"]].copy()
    oi.index = pd.to_datetime(oi.index, utc=True, errors="coerce")
    oi = oi.loc[oi.index.notna()].sort_index().loc[~oi.index.duplicated(keep="last")]
    oi["open_interest"] = pd.to_numeric(oi.open_interest, errors="coerce")
    oi = oi.loc[oi.open_interest.gt(0.0)]
    if oi.empty:
        return pd.Series(np.nan, index=bars.index, dtype="float64"), 0
    left = pd.DataFrame({"bar_ts": pd.to_datetime(bars.index, utc=True)})
    right = oi.reset_index(names="oi_ts").sort_values("oi_ts", kind="stable")
    merged = pd.merge_asof(
        left.sort_values("bar_ts", kind="stable"), right, left_on="bar_ts", right_on="oi_ts",
        direction="backward", allow_exact_matches=False,
    )
    age = merged.bar_ts - merged.oi_ts
    valid = age.le(pd.Timedelta(minutes=15 * max_staleness_bars))
    values = pd.to_numeric(merged.open_interest.where(valid), errors="coerce").to_numpy(float)
    return pd.Series(values, index=bars.index, dtype="float64"), int(valid.sum())


def _worker(payload: tuple[str, dict[pd.Timestamp, list[dict[str, object]]], str, str, str, str, str, dict[str, object]]) -> dict[str, object]:
    symbol, targets, bars_root_raw, oi_root_raw, output_start_raw, output_end_raw, parts_raw, config_raw = payload
    try:
        bars = read_symbol_bars(Path(bars_root_raw), symbol)
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        config = OIPositioningConfig(**config_raw)
        oi_path = _oi_path(Path(oi_root_raw), symbol)
        bars["open_interest"], oi_rows = _strict_prior_oi(bars, oi_path, config.oi_max_staleness_bars)
        engine = CausalOIPositioningEngine(
            symbol, bars, output_start=pd.Timestamp(output_start_raw), output_end=pd.Timestamp(output_end_raw),
            snapshot_targets=targets, config=config,
        )
        events, snapshots, zones = engine.run()
        token = symbol.lower().replace("/", "_").replace(":", "_")
        parts = Path(parts_raw)
        if not events.empty:
            events.to_parquet(parts / f"{token}__events.parquet", index=False, compression="zstd")
        if not snapshots.empty:
            snapshots.to_parquet(parts / f"{token}__snapshots.parquet", index=False, compression="zstd")
        if not zones.empty:
            zones.to_parquet(parts / f"{token}__zones.parquet", index=False, compression="zstd")
        return {"__symbol__": symbol, "source_available": True, "bar_rows": len(bars), "strict_prior_oi_rows": oi_rows,
                "oi_coverage": float(oi_rows / max(len(bars), 1)), "event_rows": len(events), "snapshot_rows": len(snapshots)}
    except Exception as exc:
        return {"__symbol__": symbol, "source_available": False, "exception_type": type(exc).__name__, "exception": str(exc)}


def _concat(parts: Path, suffix: str, output: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(parts.glob(f"*__{suffix}.parquet"))]
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_parquet(output, index=False, compression="zstd")
    return result


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
    config_raw: dict[str, object] = {}
    if args.config_json:
        config_raw = json.loads(args.config_json.resolve().read_text(encoding="utf-8"))
    config = OIPositioningConfig(**config_raw)
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
    parts = output / "parts"
    parts.mkdir()
    payloads = [
        (symbol, target_map[symbol], str(args.bars_root.resolve()), str(args.oi_root.resolve()), str(output_start), str(output_end), str(parts), config_raw)
        for symbol in symbols
    ]
    coverage: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            item = future.result()
            coverage.append(item)
            print(json.dumps(item, default=str), flush=True)
    coverage_frame = pd.DataFrame(coverage).sort_values("__symbol__", kind="stable")
    coverage_frame.to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    events = _concat(parts, "events", output / "positioning_interactions.parquet")
    snapshots = _concat(parts, "snapshots", output / "positioning_snapshots.parquet")
    _concat(parts, "zones", output / "positioning_zone_tape.parquet")
    manifest: dict[str, Any] = {
        "schema": "causal-oi-positioning-v1",
        "scope": "offline causal research only; no live mutation or exchange I/O",
        "contract": {
            "directional_lifecycle": "build -> active -> revisit -> defended or failed/trapped/unwound -> expired; never role-reverses",
            "oi_observation": "strictly earlier than 15m feature timestamp; max staleness is configured",
            "position_proxy": "price/OI regime proxy, not observed literal long or short inventory",
            "labels": "future outcome labels are separated from point-in-time snapshots",
            "config": config.__dict__,
        },
        "inputs": {"entry_panel": str(args.entry_panel.resolve()), "entry_panel_sha256": _sha256(args.entry_panel.resolve()),
                   "bars_root": str(args.bars_root.resolve()), "oi_root": str(args.oi_root.resolve())},
        "outputs": {"events": len(events), "snapshots": len(snapshots), "symbols": len(symbols)},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
