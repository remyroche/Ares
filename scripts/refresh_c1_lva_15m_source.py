#!/usr/bin/env python3
"""Append a frozen C1-LVA 15-minute public source universe and audit it.

This is deliberately upstream-only.  It delegates retrieval and atomic
timestamp-level cache merges to :mod:`download_kraken_15m_hf`, then verifies
that every requested bar was observed by Kraken.  A locally synthesised flat
bar is useful cache provenance, but is not valid input to the C1 structural
state.  The result has no model, mapper, portfolio, private-account, or
order-submission authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_sr_engine import read_symbol_bars


SCHEMA = "strict-r3-c1-lva-15m-source-refresh-v1"
DEFAULT_PARTITIONS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _symbols(source_map: Path) -> list[str]:
    payload = json.loads(source_map.read_text(encoding="utf-8"))
    mapping = payload.get("source_map") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("C1 source map lacks a non-empty source_map")
    symbols = sorted(str(item) for item in mapping)
    declared = payload.get("symbols")
    if declared is not None and int(declared) != len(symbols):
        raise ValueError("C1 source map symbol count does not match source_map")
    return symbols


def audit_symbol_coverage(
    *, bars: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp,
) -> dict[str, object]:
    """Return exact completed-bar coverage for one source series.

    The audit is intentionally stronger than the cache's continuity check:
    every bar in the appended interval must carry ``exchange_observed=True``.
    Missing source candles that were locally regularised into flat bars remain
    visible and fail the C1 source receipt.
    """
    expected = pd.date_range(start, end_exclusive - pd.Timedelta(minutes=15), freq="15min", tz="UTC")
    frame = bars.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna() & ~frame.index.duplicated(keep="last")].sort_index()
    observed = frame.reindex(expected)
    required = ("open", "high", "low", "close", "volume")
    missing_ohlcv = observed.loc[:, list(required)].isna().any(axis=1) if set(required).issubset(observed.columns) else pd.Series(True, index=expected)
    if "exchange_observed" in observed.columns:
        exchange_observed = observed["exchange_observed"].astype("boolean").fillna(False).astype(bool)
    else:
        exchange_observed = pd.Series(False, index=expected)
    missing = expected[missing_ohlcv.to_numpy()]
    synthetic_or_unknown = expected[(~exchange_observed).to_numpy()]
    return {
        "expected_bars": int(len(expected)),
        "present_ohlcv_bars": int((~missing_ohlcv).sum()),
        "exchange_observed_bars": int(exchange_observed.sum()),
        "missing_ohlcv_bars": int(len(missing)),
        "synthetic_or_unknown_bars": int(len(synthetic_or_unknown)),
        "missing_ohlcv_first": None if len(missing) == 0 else missing[0].isoformat(),
        "synthetic_or_unknown_first": None if len(synthetic_or_unknown) == 0 else synthetic_or_unknown[0].isoformat(),
        "source_complete_exchange_observed": bool(len(missing) == 0 and len(synthetic_or_unknown) == 0),
    }


def _launch_partitions(*, source_map: Path, start: pd.Timestamp, end_exclusive: pd.Timestamp, output: Path, partitions: int) -> list[int]:
    children: list[tuple[int, Any, subprocess.Popen[str]]] = []
    for partition in range(partitions):
        log = output / f"ohlcv15m_partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        command = [
            sys.executable, str(ROOT / "scripts/download_kraken_15m_hf.py"),
            "--target-free-manifest", str(source_map),
            "--force-start", start.isoformat(), "--force-end", end_exclusive.isoformat(),
            "--hf-data-dir", "15m_ohlcv_perp", "--sleep-seconds", "0",
            "--rate-limit-ms", "1000", "--partition-count", str(partitions),
            "--partition-id", str(partition),
        ]
        children.append((partition, handle, subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)))
    failed: list[int] = []
    for partition, handle, process in children:
        code = process.wait(); handle.close()
        if code:
            failed.append(partition)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--start", required=True, help="inclusive completed UTC 15-minute bar")
    parser.add_argument("--end-exclusive", required=True, help="exclusive completed UTC 15-minute boundary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    parser.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    args = parser.parse_args()

    source_map, output, bars_root = args.source_map.resolve(), args.output.resolve(), args.bars_root.resolve()
    if output.exists():
        raise FileExistsError("C1 source-refresh output must be immutable")
    if not source_map.is_file():
        raise FileNotFoundError("C1 source map is unavailable")
    start, end_exclusive = _utc(args.start), _utc(args.end_exclusive)
    if start != start.floor("15min") or end_exclusive != end_exclusive.floor("15min") or end_exclusive <= start:
        raise ValueError("source range must be non-empty and aligned to 15 minutes")
    if end_exclusive > pd.Timestamp.now(tz="UTC").floor("15min"):
        raise ValueError("source refresh cannot request an incomplete future bar")
    partitions = max(1, int(args.partitions))
    symbols = _symbols(source_map)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    failed_partitions = _launch_partitions(
        source_map=source_map, start=start, end_exclusive=end_exclusive,
        output=output, partitions=partitions,
    )
    if failed_partitions:
        _atomic_json(output / "run_manifest.json", {
            "schema": SCHEMA, "status": "failed_partition_refresh",
            "source_map": str(source_map), "source_map_sha256": _sha256(source_map),
            "start": start.isoformat(), "end_exclusive": end_exclusive.isoformat(),
            "partitions": partitions, "failed_partitions": failed_partitions,
            "outcome_columns_consumed": [], "private_account_called": False,
            "exchange_order_submission_called": False,
        })
        raise RuntimeError(f"C1 public source refresh failed partitions={failed_partitions}")

    coverage: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            row = audit_symbol_coverage(
                bars=read_symbol_bars(bars_root, symbol), start=start, end_exclusive=end_exclusive,
            )
            row["__symbol__"] = symbol
        except Exception as exc:
            row = {
                "__symbol__": symbol, "source_complete_exchange_observed": False,
                "exception_type": type(exc).__name__, "exception": str(exc),
            }
        coverage.append(row)
    coverage_frame = pd.DataFrame(coverage).sort_values("__symbol__", kind="stable")
    coverage_path = output / "source_coverage.parquet"
    coverage_frame.to_parquet(coverage_path, index=False, compression="zstd")
    complete = coverage_frame["source_complete_exchange_observed"].fillna(False).astype(bool)
    status = "pass_complete_exchange_observed_source" if bool(complete.all()) else "failed_source_coverage"
    _atomic_json(output / "run_manifest.json", {
        "schema": SCHEMA, "status": status,
        "source_map": str(source_map), "source_map_sha256": _sha256(source_map),
        "bars_root": str(bars_root), "start": start.isoformat(), "end_exclusive": end_exclusive.isoformat(),
        "symbols_requested": len(symbols), "source_ready_symbols": int(complete.sum()),
        "source_unavailable_symbols": int((~complete).sum()), "partitions": partitions,
        "source_coverage_sha256": _sha256(coverage_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "causality": "completed public 15-minute bars only; source-unavailable symbols must remain C1-unavailable",
        "outcome_columns_consumed": [], "private_account_called": False,
        "exchange_order_submission_called": False,
    })
    if status != "pass_complete_exchange_observed_source":
        raise RuntimeError("C1 source coverage is incomplete; no current C1 state may be published")
    print(output)


if __name__ == "__main__":
    main()
