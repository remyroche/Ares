#!/usr/bin/env python3
"""Materialise a standalone, causal Support/Resistance research engine.

The job reads only archived 15-minute OHLCV.  It discovers structural levels
and labels resolved interactions from 2025 onward, then writes causal S/R
snapshots for the existing 2026 entry and continuation panels.  It performs no
exchange I/O, model fitting, or mutation of the trading stack.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_sr_engine import CausalSREngine, SREngineConfig, read_symbol_bars, symbol_filename


ENTRY_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_vwap_target_free_20260830_v1/target_free_15m_features.parquet"
CONTINUATION_PANEL = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_activation50_advantage_20260830_v1/activation50_advantage_states.parquet"
BARS_ROOT = ROOT / "15m_ohlcv_perp"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_token(symbol: str) -> str:
    return symbol.lower().replace("/", "_").replace(":", "_")


def _targets(entry: pd.DataFrame, continuation: pd.DataFrame) -> dict[str, dict[pd.Timestamp, list[dict[str, object]]]]:
    targets: dict[str, dict[pd.Timestamp, list[dict[str, object]]]] = {}
    for _, row in entry.iterrows():
        ts = pd.Timestamp(row["__decision_ts__"])
        symbol = str(row["__symbol__"])
        targets.setdefault(symbol, {}).setdefault(ts, []).append(
            {"target_kind": "entry", "target_id": str(row.candidate_id), "candidate_id": str(row.candidate_id)}
        )
    for _, row in continuation.iterrows():
        ts = pd.Timestamp(row.state_decision_ts)
        symbol = str(row["__symbol__"])
        candidate_id = str(row.candidate_id)
        state_bar = int(row.state_bar_15m)
        targets.setdefault(symbol, {}).setdefault(ts, []).append(
            {
                "target_kind": "continuation", "target_id": f"{candidate_id}|{state_bar}",
                "candidate_id": candidate_id, "state_bar_15m": state_bar,
            }
        )
    return targets


def _worker(payload: tuple[str, dict[pd.Timestamp, list[dict[str, object]]], str, str, str, str, dict[str, object], bool]) -> dict[str, object]:
    symbol, targets, bars_root_raw, output_start_raw, output_end_raw, parts_raw, ontology, compact = payload
    bars_root, output_start, output_end, parts = Path(bars_root_raw), pd.Timestamp(output_start_raw), pd.Timestamp(output_end_raw), Path(parts_raw)
    try:
        path = bars_root / symbol_filename(symbol)
        bars = read_symbol_bars(bars_root, symbol)
        engine = CausalSREngine(
            symbol, bars, output_start=output_start, output_end=output_end, snapshot_targets=targets,
            config=SREngineConfig(**ontology), record_tape=not compact,
        )
        candidates, zones, interactions, snapshots = engine.run()
        token = _safe_token(symbol)
        if not candidates.empty:
            candidates.to_parquet(parts / f"{token}__candidates.parquet", index=False, compression="zstd")
        if not zones.empty:
            zones.to_parquet(parts / f"{token}__zones.parquet", index=False, compression="zstd")
        if not interactions.empty:
            interactions.to_parquet(parts / f"{token}__interactions.parquet", index=False, compression="zstd")
        if not snapshots.empty:
            snapshots.to_parquet(parts / f"{token}__snapshots.parquet", index=False, compression="zstd")
        ts = pd.to_datetime(bars.index, utc=True)
        return {
            "__symbol__": symbol, "source_path": str(path), "source_available": True,
            "bar_start": ts.min(), "bar_end": ts.max(), "bar_count": len(bars),
            "candidate_rows": len(candidates), "zone_rows": len(zones),
            "interaction_rows": len(interactions), "snapshot_rows": len(snapshots),
        }
    except Exception as exc:  # candidate-local fail-closed: never repair from another source.
        return {"__symbol__": symbol, "source_path": str(bars_root / symbol_filename(symbol)), "source_available": False,
                "exception_type": type(exc).__name__, "exception": str(exc)}


def _concat(parts: Path, suffix: str, output: Path) -> pd.DataFrame:
    files = sorted(parts.glob(f"*__{suffix}.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    result = pd.concat(frames, ignore_index=True)
    result.to_parquet(output, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-panel", type=Path, default=ENTRY_PANEL)
    parser.add_argument("--continuation-panel", type=Path, default=CONTINUATION_PANEL)
    parser.add_argument("--bars-root", type=Path, default=BARS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--output-end", help="defaults to the latest target timestamp")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-symbols", type=int, help="deterministic smoke/research cap")
    parser.add_argument("--compact", action="store_true", help="omit diagnostic candidate/zone tapes; interactions and snapshots are unchanged")
    parser.add_argument(
        "--ontology-json", type=Path,
        help="optional JSON object passed to SREngineConfig; omitted preserves the V1 geometry exactly",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    ontology: dict[str, object] = {}
    if args.ontology_json is not None:
        loaded = json.loads(args.ontology_json.resolve().read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--ontology-json must contain one JSON object")
        for name in ("reaction_barriers", "penetration_barriers"):
            if name in loaded:
                loaded[name] = tuple(float(value) for value in loaded[name])
        # Validate before parallel workers make an immutable output directory.
        SREngineConfig(**loaded)
        ontology = loaded
    entry = pd.read_parquet(args.entry_panel.resolve())
    continuation = pd.read_parquet(args.continuation_panel.resolve())
    entry["__decision_ts__"] = pd.to_datetime(entry["__decision_ts__"], utc=True, errors="raise")
    continuation["state_decision_ts"] = pd.to_datetime(continuation.state_decision_ts, utc=True, errors="raise")
    target_map = _targets(entry, continuation)
    symbols = sorted(target_map)
    if args.max_symbols is not None:
        symbols = symbols[:args.max_symbols]
        target_map = {symbol: target_map[symbol] for symbol in symbols}
    output_start = pd.Timestamp(args.output_start)
    output_end = pd.Timestamp(args.output_end) if args.output_end else max(
        max(values) for values in target_map.values() if values
    )
    output.mkdir(parents=True, exist_ok=False)
    parts = output / "parts"
    parts.mkdir()
    payloads = [
        (symbol, target_map[symbol], str(args.bars_root.resolve()), str(output_start), str(output_end), str(parts), ontology, bool(args.compact))
        for symbol in symbols
    ]
    coverage: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            coverage.append(future.result())
            print(json.dumps(coverage[-1], default=str), flush=True)
    coverage_frame = pd.DataFrame(coverage).sort_values("__symbol__", kind="stable")
    coverage_frame.to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    candidates = _concat(parts, "candidates", output / "candidate_levels.parquet") if not args.compact else pd.DataFrame()
    zones = _concat(parts, "zones", output / "merged_zone_tape_daily.parquet") if not args.compact else pd.DataFrame()
    interactions = _concat(parts, "interactions", output / "interaction_events.parquet")
    snapshots = _concat(parts, "snapshots", output / "sr_snapshots.parquet")
    if not snapshots.empty:
        snapshots.loc[snapshots.target_kind.eq("entry")].to_parquet(output / "entry_sr_snapshots.parquet", index=False, compression="zstd")
        snapshots.loc[snapshots.target_kind.eq("continuation")].to_parquet(output / "continuation_sr_snapshots.parquet", index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "schema": "causal-sr-engine-v1",
        "scope": "standalone offline research; no live-trading model or execution mutation; no exchange I/O",
        "contract": {
            "output_start": str(output_start), "output_end": str(output_end),
            "pivot_confirmation": "1h pivots delayed three completed 1h bars; 4h pivots delayed two completed 4h bars",
            "sources": ["swing_1h", "swing_4h", "rolling_extreme", "prior_day", "prior_week", "vwap", "range_boundary", "role_reversal"],
            "ontology": {
                "merge_radius_atr": SREngineConfig(**ontology).merge_radius_atr,
                "touch_radius_atr": SREngineConfig(**ontology).touch_radius_atr,
                "reset_distance_atr": SREngineConfig(**ontology).reset_distance_atr,
                "reset_bars": SREngineConfig(**ontology).reset_bars,
                "reset_mode": SREngineConfig(**ontology).reset_mode,
                "reaction_barriers": list(SREngineConfig(**ontology).reaction_barriers),
                "penetration_barriers": list(SREngineConfig(**ontology).penetration_barriers),
                "horizon_bars": SREngineConfig(**ontology).horizon_bars,
                "speed_tau_bars": SREngineConfig(**ontology).speed_tau_bars,
            },
            "qualified_reset": "state-specific causal reset; parameters are recorded under ontology",
            "target": "causal reaction-before-penetration multi-barrier strength plus accepted-break target",
            "future_data": "labels resolve only after the 8h path; all level and feature rows are built before it",
        },
        "inputs": {
            "entry_panel": str(args.entry_panel.resolve()), "entry_panel_sha256": _sha256(args.entry_panel.resolve()),
            "continuation_panel": str(args.continuation_panel.resolve()), "continuation_panel_sha256": _sha256(args.continuation_panel.resolve()),
            "bars_root": str(args.bars_root.resolve()), "symbols_requested": len(symbols),
        },
        "outputs": {"candidate_rows": len(candidates), "zone_rows": len(zones), "interaction_rows": len(interactions), "snapshot_rows": len(snapshots), "compact": bool(args.compact)},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
