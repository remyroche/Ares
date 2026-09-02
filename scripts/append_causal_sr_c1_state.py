#!/usr/bin/env python3
"""Advance a completed C1 append-state universe from freshly observed 15m bars.

The state is intentionally long lived: a call advances each symbol exactly
once across a contiguous fresh interval.  It never rebuilds, resets, or
refits the S/R geometry, and never creates candidate, score, mapper, policy,
portfolio, private-account, or order state.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_sr_engine import CausalSREngine, read_symbol_bars
from extreme_price_movements.inference.causal_sr_c1_state import CausalSRC1AppendState


SCHEMA = "causal-sr-c1-state-advance-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _symbols(source_map: Path) -> list[str]:
    payload = json.loads(source_map.read_text(encoding="utf-8"))
    mapping = payload.get("source_map") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("C1 source map lacks source_map")
    return sorted(map(str, mapping))


def fresh_observed_bars(
    *, bars: pd.DataFrame, after: pd.Timestamp, through: pd.Timestamp,
) -> pd.DataFrame:
    """Extract a contiguous exchange-observed suffix, rejecting source fills."""
    frame = bars.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="raise")
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    expected_start = after + pd.Timedelta(minutes=15)
    expected = pd.date_range(expected_start, through, freq="15min", tz="UTC")
    fresh = frame.reindex(expected)
    required = ("open", "high", "low", "close", "volume")
    if set(required).difference(fresh.columns) or fresh.loc[:, list(required)].isna().any(axis=None):
        raise ValueError("C1 state append lacks a complete fresh 15-minute OHLCV suffix")
    if "exchange_observed" not in fresh.columns:
        raise ValueError("C1 state append source lacks exchange-observed provenance")
    observed = fresh["exchange_observed"].astype("boolean").fillna(False)
    if not bool(observed.all()):
        raise ValueError("C1 state append contains a synthetic or provenance-unknown bar")
    return fresh.loc[:, list(required)].copy()


def _worker(payload: tuple[str, str, str, str, str, str, str]) -> dict[str, object]:
    symbol, state_root_raw, staging_root_raw, bars_root_raw, end_raw, source_origin_raw, engine_source_raw = payload
    state_root, staging_root, bars_root = Path(state_root_raw), Path(staging_root_raw), Path(bars_root_raw)
    end, origin = _utc(end_raw), _utc(source_origin_raw)
    try:
        store = CausalSRC1AppendState(
            state_root, source_origin=origin, engine_source_path=Path(engine_source_raw),
        )
        checkpoint = store.checkpoint_path(symbol)
        if not checkpoint.is_file():
            raise FileNotFoundError("C1 bootstrap checkpoint is unavailable")
        before = _sha256(checkpoint)
        engine = CausalSREngine.load_checkpoint(checkpoint, record_tape=False)
        prior = pd.Timestamp(engine.last_processed_ts)
        if prior >= end:
            raise ValueError("C1 checkpoint is already at or after requested append endpoint")
        fresh = fresh_observed_bars(
            bars=read_symbol_bars(bars_root, symbol), after=prior, through=end,
        )
        engine.advance(fresh, snapshot_targets={})
        if pd.Timestamp(engine.last_processed_ts) != end:
            raise AssertionError("C1 append did not reach requested completed endpoint")
        staged_checkpoint = staging_root / checkpoint.name
        engine.save_checkpoint(staged_checkpoint)
        return {
            "__symbol__": symbol, "advanced": True, "prior_processed_ts": prior,
            "end_processed_ts": end, "fresh_bars": len(fresh),
            "checkpoint_before_sha256": before,
            "checkpoint_after_sha256": _sha256(staged_checkpoint),
            "staged_checkpoint": str(staged_checkpoint),
        }
    except Exception as exc:
        return {
            "__symbol__": symbol, "advanced": False,
            "exception_type": type(exc).__name__, "exception": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--end-inclusive", required=True, help="last completed UTC 15-minute bar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    state_root, source_map, output = args.state_root.resolve(), args.source_map.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("C1 state-advance output must be immutable")
    if not (state_root / "state_manifest.json").is_file() or not (state_root / "run_manifest.json").is_file():
        raise FileNotFoundError("C1 state root has not completed its immutable bootstrap")
    if not source_map.is_file():
        raise FileNotFoundError("C1 source map is unavailable")
    state_manifest = json.loads((state_root / "state_manifest.json").read_text(encoding="utf-8"))
    origin = _utc(state_manifest.get("source_origin"))
    end = _utc(args.end_inclusive)
    if end != end.floor("15min") or end > pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=15):
        raise ValueError("C1 state append endpoint must be a completed 15-minute bar")
    symbols = _symbols(source_map)
    output.mkdir(parents=True, exist_ok=False)
    # Advance every symbol into an output-local staging area.  A broken source
    # or engine may not leave a partially advanced live state root: the parent
    # publishes replacements only after all frozen symbols succeeded.
    staging_root = output / "staged_checkpoints"
    staging_root.mkdir(parents=True, exist_ok=False)
    payloads = [
        (
            symbol, str(state_root), str(staging_root), str(args.bars_root.resolve()),
            end.isoformat(), origin.isoformat(), str(ROOT / "extreme_price_movements/causal_sr_engine.py"),
        )
        for symbol in symbols
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_worker, item): item[0] for item in payloads}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); print(json.dumps(row, default=str), flush=True)
    coverage = pd.DataFrame(rows).sort_values("__symbol__", kind="stable")
    coverage_path = output / "source_coverage.parquet"
    coverage.to_parquet(coverage_path, index=False, compression="zstd")
    passed = coverage["advanced"].fillna(False).astype(bool)
    promoted = False
    if bool(passed.all()):
        for symbol in symbols:
            checkpoint = CausalSRC1AppendState(
                state_root, source_origin=origin,
                engine_source_path=ROOT / "extreme_price_movements/causal_sr_engine.py",
            ).checkpoint_path(symbol)
            staged_checkpoint = staging_root / checkpoint.name
            staged_bars = staged_checkpoint.with_suffix(staged_checkpoint.suffix + ".bars.parquet")
            target_bars = checkpoint.with_suffix(checkpoint.suffix + ".bars.parquet")
            if not staged_checkpoint.is_file() or not staged_bars.is_file():
                raise FileNotFoundError(f"C1 staged checkpoint is incomplete for {symbol}")
            # Each file replacement is atomic.  This loop runs only after all
            # symbols have produced and hash-reported a complete staged state.
            os.replace(staged_bars, target_bars)
            os.replace(staged_checkpoint, checkpoint)
        promoted = True
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass_append_only_c1_state" if promoted else "failed_c1_state_coverage",
        "state_root": str(state_root), "state_manifest_sha256": _sha256(state_root / "state_manifest.json"),
        "bootstrap_manifest_sha256": _sha256(state_root / "run_manifest.json"),
        "source_map": str(source_map), "source_map_sha256": _sha256(source_map),
        "end_inclusive": end.isoformat(), "symbols_requested": len(symbols),
        "advanced_symbols": int(passed.sum()), "failed_symbols": int((~passed).sum()),
        "source_coverage_sha256": _sha256(coverage_path),
        "staging": "all symbols are advanced into output-local staged checkpoints; state-root replacement occurs only after complete symbol coverage",
        "state_root_promoted": promoted,
        "causality": "only fresh completed exchange-observed 15-minute bars; checkpoint overlap is never rewritten; no target snapshots emitted",
        "outcome_columns_consumed": [], "private_account_called": False,
        "exchange_order_submission_called": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest["status"] != "pass_append_only_c1_state":
        raise RuntimeError("C1 append-state coverage failed; no current C1 snapshot is available")
    print(output)


if __name__ == "__main__":
    main()
