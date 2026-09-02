#!/usr/bin/env python3
"""Build initial, append-only C1 S/R source checkpoints from archived bars.

This is a no-order preparation job.  It seeds the persistent S/R lifecycle
from the declared historical origin, then future inference needs only append
new completed 15-minute bars.  It never joins policy outcomes or opens a
candidate based on source availability.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_sr_engine import CausalSREngine, read_symbol_bars, symbol_filename
from extreme_price_movements.inference.causal_sr_c1_state import CausalSRC1AppendState


DEFAULT_BARS = ROOT / "15m_ohlcv_perp"
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _stable_read_symbol_bars(
    root: Path, symbol: str, *, attempts: int = 10,
) -> pd.DataFrame:
    """Read one local archive only when its file identity stays unchanged.

    The live data collector may atomically replace an append-only Parquet
    archive while this offline bootstrap is reading it.  A transient Parquet
    footer error must not create a permanent C1-unavailable symbol.  This
    helper retries a *read-only* local read only when the file stabilises; it
    never repairs, rewrites, or fills a bar.
    """
    path = root / symbol_filename(symbol)
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        before = path.stat()
        try:
            bars = read_symbol_bars(root, symbol)
            after = path.stat()
            if (
                before.st_ino == after.st_ino
                and before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
            ):
                return bars
            last_error = RuntimeError("archive changed during local Parquet read")
        except Exception as exc:  # Retry only a local read; never substitute data.
            last_error = exc
        if attempt < attempts:
            # A collector's atomic replace can still leave a short interval
            # where the replacement is visible but its footer is not yet
            # readable.  Back off without delaying a healthy source by more
            # than one successful stat/read cycle.
            time.sleep(min(2.0, 0.5 * attempt))
    assert last_error is not None
    raise last_error


def _worker(payload: tuple[str, str, str, str, str, str, bool]) -> dict[str, object]:
    symbol, root_raw, bars_raw, origin_raw, cutoff_raw, engine_raw, resume = payload
    root, bars_root = Path(root_raw), Path(bars_raw)
    origin, cutoff = _utc(origin_raw), _utc(cutoff_raw)
    try:
        store = CausalSRC1AppendState(
            root, source_origin=origin, engine_source_path=Path(engine_raw),
        )
        engine_path = store.checkpoint_path(symbol)
        if engine_path.exists():
            if not resume:
                raise FileExistsError(f"immutable bootstrap already has {symbol}")
            existing = CausalSREngine.load_checkpoint(engine_path, record_tape=False)
            if existing.symbol != symbol or existing.last_processed_ts != cutoff:
                raise ValueError("existing checkpoint does not match this symbol/cutoff")
            return {
                "__symbol__": symbol, "source_ready": True,
                "checkpoint": str(engine_path), "checkpoint_sha256": _sha256(engine_path),
                "bar_start": existing.processed_source_start,
                "bar_end": existing.last_processed_ts,
                "bar_rows": int(len(existing.bars)),
                "last_processed_ts": existing.last_processed_ts,
                "status": "resumed_verified_checkpoint",
            }
        bars = _stable_read_symbol_bars(bars_root, symbol)
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        bars = bars.loc[bars.index <= cutoff]
        # A target-free sentinel is intentionally not emitted.  This run only
        # establishes deterministic structural state through the exact
        # completed cutoff; future decision snapshots are produced by append.
        if len(bars) < 2:
            raise ValueError("fewer than two completed source bars")
        engine = CausalSREngine(
            symbol, bars, output_start=origin, output_end=cutoff, record_tape=False,
        )
        engine.bootstrap_append_state(through=cutoff)
        engine.save_checkpoint(engine_path)
        return {
            "__symbol__": symbol, "source_ready": True, "checkpoint": str(engine_path),
            "checkpoint_sha256": _sha256(engine_path), "bar_start": bars.index.min(),
            "bar_end": bars.index.max(), "bar_rows": int(len(bars)),
            "last_processed_ts": engine.last_processed_ts,
        }
    except Exception as exc:  # Candidate-local and source-local fail closed.
        return {
            "__symbol__": symbol, "source_ready": False,
            "exception_type": type(exc).__name__, "exception": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols", type=Path, required=True,
        help=(
            "Parquet/CSV with one __symbol__ column, or a frozen JSON "
            "manifest containing either symbols or source_map"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume an incomplete identical bootstrap, verifying every existing checkpoint",
    )
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE / "run_manifest.json")
    parser.add_argument("--origin", help="defaults to source manifest output_start")
    parser.add_argument("--cutoff", required=True, help="inclusive completed UTC 15-minute bar")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"bootstrap output must be new: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError("--resume requires an existing incomplete bootstrap output")
    source_manifest = args.source_manifest.resolve()
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    origin = _utc(args.origin or dict(source_payload.get("contract") or {})["output_start"])
    cutoff = _utc(args.cutoff)
    if cutoff < origin:
        raise ValueError("cutoff precedes source origin")
    symbols_path = args.symbols.resolve()
    if symbols_path.suffix.lower() == ".json":
        symbols_payload = json.loads(symbols_path.read_text(encoding="utf-8"))
        raw_symbols = (
            symbols_payload.get("symbols")
            if isinstance(symbols_payload, dict)
            else None
        )
        if raw_symbols is None and isinstance(symbols_payload, dict):
            source_map = symbols_payload.get("source_map")
            raw_symbols = list(source_map) if isinstance(source_map, dict) else None
        if not isinstance(raw_symbols, list) or not raw_symbols:
            raise ValueError("JSON --symbols requires a non-empty symbols list or source_map")
        symbols = sorted({str(symbol) for symbol in raw_symbols})
    else:
        symbols_frame = pd.read_parquet(symbols_path) if symbols_path.suffix == ".parquet" else pd.read_csv(symbols_path)
        if "__symbol__" not in symbols_frame:
            raise ValueError("--symbols requires __symbol__")
        symbols = sorted(symbols_frame["__symbol__"].astype(str).drop_duplicates())
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    # Seal the common state geometry before workers begin creating per-symbol
    # checkpoints.  This avoids a manifest-write race and makes a mismatched
    # ontology fail before any potentially expensive replay starts.
    CausalSRC1AppendState(
        output, source_origin=origin,
        engine_source_path=ROOT / "extreme_price_movements/causal_sr_engine.py",
    )
    payloads = [
        (
            symbol, str(output), str(args.bars_root.resolve()), origin.isoformat(),
            cutoff.isoformat(), str(ROOT / "extreme_price_movements/causal_sr_engine.py"),
            bool(args.resume),
        )
        for symbol in symbols
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); print(json.dumps(row, default=str), flush=True)
    coverage = pd.DataFrame(rows).sort_values("__symbol__", kind="stable")
    coverage.to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "schema": "causal-sr-c1-append-bootstrap-v1",
        "scope": "no-order source-state bootstrap; no candidates, outcomes, MC1, portfolio, exchange, or execution authority",
        "source_origin": origin.isoformat(), "cutoff": cutoff.isoformat(),
        "source_manifest": str(source_manifest.relative_to(ROOT)),
        "source_manifest_sha256": _sha256(source_manifest),
        "engine_source": "extreme_price_movements/causal_sr_engine.py",
        "engine_source_sha256": _sha256(ROOT / "extreme_price_movements/causal_sr_engine.py"),
        "symbols_requested": len(symbols),
        "source_ready_symbols": int(coverage.source_ready.sum()),
        "resumed": bool(args.resume),
        "source_coverage_sha256": _sha256(output / "source_coverage.parquet"),
        "causality": {
            "bars": "only completed 15-minute archived bars at or before cutoff",
            "pending": "unresolved reactions remain in checkpoint and resolve only after their own later 8-hour path",
            "profile": "not created by this S/R bootstrap; bounded LVA context is separately materialised from completed bars",
            "fail_closed": "a missing/corrupt symbol creates no checkpoint and must remain C1-unavailable",
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
