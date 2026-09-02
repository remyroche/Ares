#!/usr/bin/env python3
"""Refresh and append one immutable, complete P8U live source successor.

This is deliberately an upstream-only producer.  It fetches public Kraken
inputs into the existing append-only caches, validates the full frozen 160
symbol OHLCV contract, then delegates publication to the canonical P8U source
append script.  It has no model, admission, portfolio, private-account, or
order-submission authority.

The initial invocation can repair a historical gap by requesting every missing
hour after a complete predecessor.  Subsequent live invocations append only
the next completed source hour.  A partial Kraken response never becomes a
source successor: it remains a refresh receipt and the scorer must fail closed.
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

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.append_strict_r3_p8u_canonical_source_state import _load_source


SCHEMA = "strict_r3_p8u_live_source_refresh_v1"
PARTITIONS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _launch_partitions(
    *,
    command_prefix: list[str],
    refresh_root: Path,
    name: str,
) -> list[int]:
    """Run deterministic network partitions and preserve every child log."""
    children: list[tuple[int, Any, subprocess.Popen[str]]] = []
    for partition in range(PARTITIONS):
        log = refresh_root / f"{name}_partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        command = [
            *command_prefix,
            "--partition-count", str(PARTITIONS),
            "--partition-id", str(partition),
        ]
        children.append((
            partition,
            handle,
            subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            ),
        ))
    failed: list[int] = []
    for partition, handle, process in children:
        result = process.wait()
        handle.close()
        if result:
            failed.append(partition)
    return failed


def _run_checked(*, command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"source refresh command failed: {log}")


def _source_start(path: Path) -> pd.Timestamp:
    state = _load_source(path)
    close = state["panel"]["close"]
    return _utc(close.index[-1]) + pd.Timedelta(hours=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--refresh-root", type=Path, required=True)
    parser.add_argument(
        "--allow-reconstructed-source-map",
        action="store_true",
        help=(
            "Pass the append producer's narrowly audited recovery mode for a "
            "deleted original manifest. The supplied manifest must still be "
            "bound to the exact predecessor source-map identity."
        ),
    )
    parser.add_argument(
        "--allow-historical-catchup",
        action="store_true",
        help="Required when more than one completed source hour must be fetched.",
    )
    args = parser.parse_args()

    predecessor = args.source_state.resolve()
    manifest = args.canonical_manifest.resolve()
    out_dir = args.out_dir.resolve()
    refresh_root = args.refresh_root.resolve()
    if out_dir.exists() or refresh_root.exists():
        raise FileExistsError("live source refresh outputs must be immutable")
    if not predecessor.is_file() or not manifest.is_file():
        raise FileNotFoundError("source predecessor or frozen manifest is unavailable")
    end = _utc(args.end_exclusive)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    if end > now:
        raise ValueError("source refresh may not request an incomplete future hour")
    start = _source_start(predecessor)
    if end <= start:
        raise ValueError("source state is already current through requested end")
    hours = int((end - start) / pd.Timedelta(hours=1))
    if hours > 1 and not args.allow_historical_catchup:
        raise ValueError("historical source catch-up requires --allow-historical-catchup")

    refresh_root.mkdir(parents=True)
    started = time.monotonic()
    base = [sys.executable]
    fifteen_prefix = [
        *base,
        str(ROOT / "scripts/download_kraken_15m_hf.py"),
        "--target-free-manifest", str(manifest),
        "--force-start", start.isoformat(),
        "--force-end", end.isoformat(),
        "--hf-data-dir", "15m_ohlcv_perp",
        "--sleep-seconds", "0",
        "--rate-limit-ms", "1000",
    ]
    failed_15m = _launch_partitions(
        command_prefix=fifteen_prefix,
        refresh_root=refresh_root,
        name="ohlcv15m",
    )
    if failed_15m:
        _atomic_json(refresh_root / "run_manifest.json", {
            "schema": SCHEMA,
            "status": "failed_15m_refresh",
            "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "failed_partitions": failed_15m,
            "exchange_or_order_submission_called": False,
        })
        raise RuntimeError(f"15m source refresh failed partitions={failed_15m}")

    _run_checked(
        command=[
            *base,
            str(ROOT / "scripts/backfill_kraken_frozen_contract_inputs.py"),
            "--symbols-json", str(manifest),
            "--out-dir", "data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly",
            "--start", start.isoformat(), "--end", end.isoformat(),
            "--workers", "16", "--include-trade-ohlcv", "--include-orderbook-analytics",
        ],
        log=refresh_root / "frozen_inputs.log",
    )

    oi_prefix = [
        *base,
        str(ROOT / "scripts/backfill_kraken_oi_funding_sidecars.py"),
        "--feature-dir", "data_perp/features",
        "--symbols-file", str(manifest),
        "--perp-root", "data_perp/exchanges/krakenfutures",
        "--out-dir", str(refresh_root / "oi_funding_partitions"),
        "--quarantine-corrupt-sidecars-dir", "data_perp/exchanges/krakenfutures/corrupt_sidecars",
        "--start-ts", start.isoformat(), "--end-ts", end.isoformat(),
        "--workers", "2", "--batch-append",
    ]
    failed_oi = _launch_partitions(
        command_prefix=oi_prefix,
        refresh_root=refresh_root,
        name="oi_funding",
    )
    if failed_oi:
        _atomic_json(refresh_root / "run_manifest.json", {
            "schema": SCHEMA,
            "status": "failed_oi_funding_refresh",
            "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "failed_partitions": failed_oi,
            "exchange_or_order_submission_called": False,
        })
        raise RuntimeError(f"OI/funding source refresh failed partitions={failed_oi}")

    append_log = refresh_root / "append.log"
    append_command = [
        *base,
        str(ROOT / "scripts/append_strict_r3_p8u_canonical_source_state.py"),
        "--source-state", str(predecessor),
        "--canonical-manifest", str(manifest),
        "--end-exclusive", end.isoformat(),
        "--out-dir", str(out_dir),
        "--require-complete-ohlcv",
    ]
    if args.allow_reconstructed_source_map:
        append_command.append("--allow-reconstructed-source-map")
    _run_checked(
        command=append_command,
        log=append_log,
    )
    append_receipt = json.loads((out_dir / "receipt.json").read_text())
    _atomic_json(refresh_root / "run_manifest.json", {
        "schema": SCHEMA,
        "status": "pass_target_free_complete_source_append",
        "predecessor": str(predecessor),
        "predecessor_sha256": _sha256(predecessor),
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "hours": hours, "partitions": PARTITIONS,
        "source_successor": str(out_dir / "source_panel_state.joblib"),
        "source_successor_sha256": append_receipt["source_panel_sha256"],
        "source_coverage": append_receipt.get("live_ohlcv_coverage"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "outcome_columns_consumed": [],
        "exchange_or_order_submission_called": False,
        "private_account_called": False,
    })
    print(json.dumps(json.loads((refresh_root / "run_manifest.json").read_text()), sort_keys=True))


if __name__ == "__main__":
    main()
