#!/usr/bin/env python3
"""Audit raw strict-R3 15-minute parquet shards in bounded subprocesses.

Arrow can block indefinitely while opening metadata from an unhealthy local
parquet shard.  This audit deliberately opens each raw shard in an isolated
child process with a timeout, then verifies that the pre-existing shared
15-minute mirror is readable before recommending that the raw shard be
quarantined.  It does not mutate source data or model contracts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data_perp/exchanges/krakenfutures/raw/ohlcv_15m"
MIRROR_ROOT = ROOT / "15m_ohlcv_perp"
CHILD = r"""
import sys
import pandas as pd

path, start, end = sys.argv[1:]
fields = ['open', 'high', 'low', 'close', 'volume']
filters = [
    ('__index_level_0__', '>=', pd.Timestamp(start).to_pydatetime()),
    ('__index_level_0__', '<', pd.Timestamp(end).to_pydatetime()),
]
try:
    pd.read_parquet(path, columns=[*fields, 'exchange_observed'], filters=filters)
except Exception:
    pd.read_parquet(path, columns=fields, filters=filters)
"""


def _name(symbol: str) -> str:
    return f"{symbol.lower().replace('/', '')}_15m.parquet"


def _check(
    path: Path,
    timeout_seconds: float,
    start: str,
    end: str,
) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, str(path), start, end],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"path": str(path), "status": "timeout"}
    if proc.returncode:
        return {
            "path": str(path),
            "status": "error",
            "stderr": proc.stderr.strip()[-500:],
        }
    return {"path": str(path), "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--history-start", default="2025-11-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-08-01T00:00:00Z")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.workers <= 0:
        raise ValueError("timeout-seconds and workers must be positive")

    candidates = pd.read_parquet(args.candidates, columns=["__symbol__"])
    symbols = sorted(candidates["__symbol__"].astype(str).unique())
    if len(symbols) != 170:
        raise ValueError(f"expected frozen 170-symbol universe, got {len(symbols)}")

    names = [_name(symbol) for symbol in symbols]
    raw_paths = [RAW_ROOT / name for name in names]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        raw_results = list(pool.map(
            lambda path: _check(
                path, args.timeout_seconds, args.history_start, args.end_exclusive,
            ),
            raw_paths,
        ))

    entries: dict[str, dict[str, object]] = {}
    audit_rows: list[dict[str, object]] = []
    for symbol, name, raw in zip(symbols, names, raw_results, strict=True):
        row = {"symbol": symbol, "file": name, "raw_status": raw["status"]}
        if raw["status"] == "ok":
            row["mirror_status"] = "not_needed"
        else:
            mirror = _check(
                MIRROR_ROOT / name,
                args.timeout_seconds,
                args.history_start,
                args.end_exclusive,
            )
            row["mirror_status"] = mirror["status"]
            if mirror["status"] == "ok":
                entries[name] = {
                    "source": str(RAW_ROOT / name),
                    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": (
                        "Raw 15-minute parquet failed bounded Arrow health "
                        f"preflight with status={raw['status']}; the existing "
                        "same-interval shared mirror passed."
                    ),
                    "fallback": "shared_15m_only",
                }
            else:
                row["failure"] = "no_healthy_same_interval_mirror"
        audit_rows.append(row)

    payload = {
        "schema": "strict_r3_raw_15m_source_health_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_universe": str(args.candidates),
        "symbols": len(symbols),
        "timeout_seconds": args.timeout_seconds,
        "workers": args.workers,
        "history_start": args.history_start,
        "end_exclusive": args.end_exclusive,
        "rows": audit_rows,
        "recommended_quarantine_entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    unhealthy = sum(row["raw_status"] != "ok" for row in audit_rows)
    unavailable = sum("failure" in row for row in audit_rows)
    print(json.dumps({
        "status": "pass" if not unavailable else "fail",
        "symbols": len(symbols),
        "unhealthy_raw": unhealthy,
        "recommended_quarantines": len(entries),
        "unavailable_without_mirror": unavailable,
        "out": str(args.out),
    }, sort_keys=True))
    if unavailable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
