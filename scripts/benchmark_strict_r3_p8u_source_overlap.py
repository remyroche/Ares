#!/usr/bin/env python3
"""Isolated, no-order Kraken source-overlap benchmark for P8U.

This tool never touches production caches or source state.  It evaluates the
only proposed source-speed change: after the normal 15-minute OHLCV coverage
refresh succeeds, run the independent frozen-input and OI/funding refreshes
concurrently under a bounded network budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = 16


def _run_partitions(
    *,
    prefix: list[str],
    logs: Path,
    name: str,
    partitioned_out_dir: Path | None = None,
) -> dict[str, object]:
    processes: list[tuple[int, subprocess.Popen[object], object]] = []
    for partition in range(PARTITIONS):
        log = (logs / f"{name}_{partition:02d}.log").open("w", encoding="utf-8")
        command = [*prefix]
        if partitioned_out_dir is not None:
            command.extend(["--out-dir", str(partitioned_out_dir / f"part_{partition}")])
        command.extend(["--partition-count", str(PARTITIONS), "--partition-id", str(partition)])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((partition, process, log))
    failures: list[int] = []
    for partition, process, log in processes:
        if process.wait() != 0:
            failures.append(partition)
        log.close()
    return {"name": name, "failures": failures}


def _contains_transport_error(logs: Path) -> list[str]:
    terms = ("429", "rate limit", "too many", "connectionreset", "connection reset", "traceback")
    hits: list[str] = []
    for path in sorted(logs.glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if any(term in text for term in terms):
            hits.append(path.name)
    return hits


def _oi_summary(root: Path) -> dict[str, int]:
    manifests = sorted(root.glob("part_*/backfill_manifest.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    counts = [dict(payload.get("result_counts") or {}) for payload in payloads]
    return {
        "manifests": len(payloads),
        "jobs": sum(int(item.get("jobs", 0)) for item in counts),
        "ok": sum(int(item.get("ok", 0)) for item in counts),
        "empty": sum(int(item.get("empty", 0)) for item in counts),
        "error": sum(int(item.get("error", 0)) for item in counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"benchmark output must be immutable: {out}")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    started = time.monotonic()
    python = sys.executable

    ohlcv = _run_partitions(
        prefix=[
            python, str(ROOT / "scripts/download_kraken_15m_hf.py"),
            "--target-free-manifest", str(args.manifest.resolve()),
            "--force-start", str(args.start), "--force-end", str(args.end),
            "--hf-data-dir", str(out / "hf15m"),
            "--sleep-seconds", "0", "--rate-limit-ms", "1000",
        ],
        logs=logs,
        name="ohlcv15m",
    )
    ohlcv_files = len(list((out / "hf15m").glob("*.parquet")))

    frozen_log = (logs / "frozen.log").open("w", encoding="utf-8")
    frozen = subprocess.Popen(
        [
            python, str(ROOT / "scripts/backfill_kraken_frozen_contract_inputs.py"),
            "--symbols-json", str(args.manifest.resolve()),
            "--out-dir", str(out / "frozen"),
            "--start", str(args.start), "--end", str(args.end),
            "--workers", "16", "--include-trade-ohlcv", "--include-orderbook-analytics",
        ],
        cwd=ROOT,
        stdout=frozen_log,
        stderr=subprocess.STDOUT,
    )
    oi = _run_partitions(
        prefix=[
            python, str(ROOT / "scripts/backfill_kraken_oi_funding_sidecars.py"),
            "--feature-dir", str(ROOT / "data_perp/features"),
            "--symbols-file", str(args.manifest.resolve()),
            "--perp-root", str(out / "perp"),
            "--out-dir", str(out / "oi"),
            "--quarantine-corrupt-sidecars-dir", str(out / "quarantine"),
            "--start-ts", str(args.start), "--end-ts", str(args.end),
            "--workers", "1", "--batch-append",
        ],
        logs=logs,
        name="oi_funding",
        partitioned_out_dir=out / "oi",
    )
    frozen_rc = frozen.wait()
    frozen_log.close()
    frozen_text = (logs / "frozen.log").read_text(encoding="utf-8", errors="replace")
    summary = {
        "schema": "strict_r3_p8u_source_overlap_benchmark_v1",
        "no_model_or_account_or_order_io": True,
        "start": str(args.start),
        "end": str(args.end),
        "ohlcv_partition_failures": ohlcv["failures"],
        "ohlcv_15m_symbol_files": ohlcv_files,
        "frozen_exit_code": frozen_rc,
        "frozen_completed_160": '"completed": 160' in frozen_text,
        "oi_partition_failures": oi["failures"],
        "oi": _oi_summary(out / "oi"),
        "transport_error_logs": _contains_transport_error(logs),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    summary["status"] = "pass" if (
        not summary["ohlcv_partition_failures"]
        and summary["ohlcv_15m_symbol_files"] == 160
        and summary["frozen_exit_code"] == 0
        and summary["frozen_completed_160"]
        and not summary["oi_partition_failures"]
        and summary["oi"]["manifests"] == 16
        and summary["oi"]["error"] == 0
        and not summary["transport_error_logs"]
    ) else "fail"
    (out / "run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
