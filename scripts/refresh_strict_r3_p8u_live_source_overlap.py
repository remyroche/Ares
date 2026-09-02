#!/usr/bin/env python3
"""Append one complete P8U live source successor with bounded I/O overlap.

The source contract is unchanged from ``refresh_strict_r3_p8u_live_source``:
completed 15-minute OHLCV must be complete for the frozen 160 symbols before
any successor is published.  The only optimisation is that the independent
frozen trade/book refresh and OI/funding refresh run together after that
coverage gate.  OI/funding partitions have one bounded, partition-local retry
when their manifest records a transport error.  A second error fails closed.

This producer has no model, account, admission, or order-submission authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd

# This module is launched by an absolute script path from the persistent live
# session.  In that invocation Python puts ``scripts/`` (not the repository
# root) on ``sys.path``; make the package root explicit before importing the
# shared source-refresh contract.  Importing this file as a package remains
# unchanged.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refresh_strict_r3_p8u_live_source import (
    PARTITIONS,
    ROOT,
    SCHEMA,
    _atomic_json,
    _launch_partitions,
    _run_checked,
    _sha256,
    _source_start,
    _utc,
)


OVERLAP_SCHEMA = "strict_r3_p8u_live_source_refresh_overlap_v1"


def _start_frozen_refresh(*, command: list[str], log: Path) -> tuple[subprocess.Popen[str], Any]:
    """Start the independent frozen trade/book refresh without waiting."""
    handle = log.open("w", encoding="utf-8")
    return (
        subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True),
        handle,
    )


def _launch_oi_partitions(
    *,
    command_prefix: list[str],
    refresh_root: Path,
    name: str,
) -> list[int]:
    """Launch OI partitions with isolated manifests; return nonzero children."""
    children: list[tuple[int, Any, subprocess.Popen[str]]] = []
    partition_root = refresh_root / "oi_funding_partitions"
    for partition in range(PARTITIONS):
        log = refresh_root / f"{name}_partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        command = [
            *command_prefix,
            "--out-dir", str(partition_root / f"part_{partition:02d}"),
            "--partition-count", str(PARTITIONS),
            "--partition-id", str(partition),
        ]
        children.append((
            partition,
            handle,
            subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True),
        ))
    failed: list[int] = []
    for partition, handle, process in children:
        result = process.wait()
        handle.close()
        if result:
            failed.append(partition)
    return failed


def _oi_error_count(path: Path) -> int | None:
    """Return manifest error count, or ``None`` when receipt is absent/invalid."""
    manifest = path / "backfill_manifest.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return int(dict(payload.get("result_counts") or {}).get("error", 0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _retry_oi_partition(
    *,
    partition: int,
    command_prefix: list[str],
    refresh_root: Path,
) -> tuple[bool, int | None]:
    """Retry exactly one original partition once, preserving an immutable receipt."""
    retry_root = refresh_root / "oi_funding_retries" / f"part_{partition:02d}"
    log = refresh_root / f"oi_funding_retry_partition_{partition:02d}.log"
    command = [
        *command_prefix,
        "--out-dir", str(retry_root),
        "--partition-count", str(PARTITIONS),
        "--partition-id", str(partition),
    ]
    try:
        _run_checked(command=command, log=log)
    except RuntimeError:
        return False, None
    errors = _oi_error_count(retry_root)
    return errors == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--refresh-root", type=Path, required=True)
    parser.add_argument("--allow-historical-catchup", action="store_true")
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
        *base, str(ROOT / "scripts/download_kraken_15m_hf.py"),
        "--target-free-manifest", str(manifest),
        "--force-start", start.isoformat(), "--force-end", end.isoformat(),
        "--hf-data-dir", "15m_ohlcv_perp", "--sleep-seconds", "0", "--rate-limit-ms", "1000",
    ]
    failed_15m = _launch_partitions(
        command_prefix=fifteen_prefix, refresh_root=refresh_root, name="ohlcv15m",
    )
    if failed_15m:
        _atomic_json(refresh_root / "run_manifest.json", {
            "schema": OVERLAP_SCHEMA, "status": "failed_15m_refresh",
            "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "failed_partitions": failed_15m, "exchange_or_order_submission_called": False,
        })
        raise RuntimeError(f"15m source refresh failed partitions={failed_15m}")

    frozen_command = [
        *base, str(ROOT / "scripts/backfill_kraken_frozen_contract_inputs.py"),
        "--symbols-json", str(manifest),
        "--out-dir", "data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly",
        "--start", start.isoformat(), "--end", end.isoformat(),
        "--workers", "16", "--include-trade-ohlcv", "--include-orderbook-analytics",
    ]
    oi_prefix = [
        *base, str(ROOT / "scripts/backfill_kraken_oi_funding_sidecars.py"),
        "--feature-dir", "data_perp/features", "--symbols-file", str(manifest),
        "--perp-root", "data_perp/exchanges/krakenfutures",
        "--quarantine-corrupt-sidecars-dir", "data_perp/exchanges/krakenfutures/corrupt_sidecars",
        "--start-ts", start.isoformat(), "--end-ts", end.isoformat(),
        "--workers", "1", "--batch-append",
    ]

    frozen, frozen_handle = _start_frozen_refresh(
        command=frozen_command, log=refresh_root / "frozen_inputs.log",
    )
    failed_oi_processes = _launch_oi_partitions(
        command_prefix=oi_prefix, refresh_root=refresh_root, name="oi_funding",
    )
    frozen_rc = frozen.wait()
    frozen_handle.close()
    oi_root = refresh_root / "oi_funding_partitions"
    oi_manifest_errors = {
        partition: _oi_error_count(oi_root / f"part_{partition:02d}")
        for partition in range(PARTITIONS)
    }
    initial_retry_partitions = sorted({
        *failed_oi_processes,
        *(partition for partition, errors in oi_manifest_errors.items() if errors is None or errors > 0),
    })
    retry_results = {
        partition: _retry_oi_partition(
            partition=partition, command_prefix=oi_prefix, refresh_root=refresh_root,
        )
        for partition in initial_retry_partitions
    }
    unrecovered = [
        partition for partition, (passed, _errors) in retry_results.items() if not passed
    ]
    if frozen_rc or unrecovered:
        _atomic_json(refresh_root / "run_manifest.json", {
            "schema": OVERLAP_SCHEMA,
            "status": "failed_parallel_refresh",
            "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "frozen_exit_code": frozen_rc,
            "oi_initial_process_failures": failed_oi_processes,
            "oi_initial_error_counts": oi_manifest_errors,
            "oi_retry_partitions": initial_retry_partitions,
            "oi_unrecovered_partitions": unrecovered,
            "exchange_or_order_submission_called": False,
        })
        raise RuntimeError("parallel frozen/OI source refresh failed closed")

    append_log = refresh_root / "append.log"
    _run_checked(
        command=[
            *base, str(ROOT / "scripts/append_strict_r3_p8u_canonical_source_state.py"),
            "--source-state", str(predecessor), "--canonical-manifest", str(manifest),
            "--end-exclusive", end.isoformat(), "--out-dir", str(out_dir),
            "--require-complete-ohlcv",
        ], log=append_log,
    )
    append_receipt = json.loads((out_dir / "receipt.json").read_text(encoding="utf-8"))
    _atomic_json(refresh_root / "run_manifest.json", {
        "schema": OVERLAP_SCHEMA,
        "status": "pass_target_free_complete_source_append",
        "predecessor": str(predecessor), "predecessor_sha256": _sha256(predecessor),
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "hours": hours, "partitions": PARTITIONS,
        "source_successor": str(out_dir / "source_panel_state.joblib"),
        "source_successor_sha256": append_receipt["source_panel_sha256"],
        "source_coverage": append_receipt.get("live_ohlcv_coverage"),
        "frozen_exit_code": frozen_rc,
        "oi_initial_process_failures": failed_oi_processes,
        "oi_initial_error_counts": oi_manifest_errors,
        "oi_retry_partitions": initial_retry_partitions,
        "oi_retry_results": {
            str(partition): {"passed": passed, "error_count": errors}
            for partition, (passed, errors) in retry_results.items()
        },
        "stage_elapsed_seconds": {
            "total": round(time.monotonic() - started, 3),
        },
        "outcome_columns_consumed": [],
        "exchange_or_order_submission_called": False,
        "private_account_called": False,
    })
    print(json.dumps(json.loads((refresh_root / "run_manifest.json").read_text()), sort_keys=True))


if __name__ == "__main__":
    main()
