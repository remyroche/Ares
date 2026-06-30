#!/usr/bin/env python3
"""Run a planned feature repair only when memory/process safety gates pass."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PLAN = Path(
    "data_perp/reports/"
    "market_state_next_no_backfill_feature_repair_plan_activefs_20260628_v1/"
    "feature_repair_plan.json"
)


@dataclass(frozen=True)
class MemorySnapshot:
    page_size: int
    free_pages: int
    speculative_pages: int
    compressor_pages: int

    @property
    def free_mb(self) -> float:
        return (self.free_pages + self.speculative_pages) * self.page_size / 1024**2

    @property
    def compressor_mb(self) -> float:
        return self.compressor_pages * self.page_size / 1024**2


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    rss_mb: float
    command: str


def parse_vm_stat(text: str) -> MemorySnapshot:
    page_size = 4096
    values: dict[str, int] = {}
    for line in text.splitlines():
        if "page size of" in line:
            parts = line.replace(")", " ").split()
            for idx, part in enumerate(parts):
                if part == "of" and idx + 1 < len(parts):
                    try:
                        page_size = int(parts[idx + 1])
                    except ValueError:
                        pass
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw_digits = "".join(ch for ch in raw if ch.isdigit())
        if raw_digits:
            values[key.strip()] = int(raw_digits)
    return MemorySnapshot(
        page_size=page_size,
        free_pages=values.get("Pages free", 0),
        speculative_pages=values.get("Pages speculative", 0),
        compressor_pages=values.get("Pages occupied by compressor", 0),
    )


def collect_memory_snapshot() -> MemorySnapshot:
    try:
        return parse_vm_stat(subprocess.check_output(["vm_stat"], text=True))
    except Exception:
        # Process/memory introspection failure should never green-light a repair.
        return MemorySnapshot(
            page_size=4096,
            free_pages=0,
            speculative_pages=0,
            compressor_pages=10**9,
        )


def parse_process_table(text: str) -> list[ProcessSnapshot]:
    processes: list[ProcessSnapshot] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            rss_kb = float(parts[1])
        except ValueError:
            continue
        processes.append(
            ProcessSnapshot(pid=pid, rss_mb=rss_kb / 1024.0, command=parts[2])
        )
    return processes


def collect_process_snapshots() -> list[ProcessSnapshot]:
    try:
        out = subprocess.check_output(
            ["ps", "ax", "-o", "pid=,rss=,command="], text=True
        )
    except Exception as exc:
        return [
            ProcessSnapshot(
                pid=-1,
                rss_mb=1_000_000_000.0,
                command=(
                    "extreme_price_movements.inference.run_inference "
                    f"process_snapshot_unavailable:{exc}"
                ),
            )
        ]
    return parse_process_table(out)


def trusted_process_snapshots(rss_mb: float, description: str) -> list[ProcessSnapshot]:
    return [
        ProcessSnapshot(
            pid=0,
            rss_mb=float(rss_mb),
            command=(
                "extreme_price_movements.inference.run_inference "
                f"trusted_external_preflight:{description}"
            ),
        )
    ]


def relevant_processes(processes: Sequence[ProcessSnapshot]) -> list[ProcessSnapshot]:
    markers = (
        "extreme_price_movements.inference.run_inference",
        "extreme_price_movements/run_pipeline.py features",
        "extreme_price_movements.run_pipeline.py features",
    )
    return [proc for proc in processes if any(marker in proc.command for marker in markers)]


def evaluate_safety(
    memory: MemorySnapshot,
    processes: Sequence[ProcessSnapshot],
    *,
    min_free_mb: float,
    max_compressor_mb: float,
    max_relevant_process_rss_mb: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    relevant = relevant_processes(processes)
    max_relevant_rss = max((proc.rss_mb for proc in relevant), default=0.0)
    reasons: list[str] = []
    if memory.free_mb < min_free_mb:
        reasons.append(
            f"free_memory_mb {memory.free_mb:.1f} < required {min_free_mb:.1f}"
        )
    if memory.compressor_mb > max_compressor_mb:
        reasons.append(
            f"compressor_mb {memory.compressor_mb:.1f} > limit {max_compressor_mb:.1f}"
        )
    if max_relevant_rss > max_relevant_process_rss_mb:
        reasons.append(
            "relevant_process_rss_mb "
            f"{max_relevant_rss:.1f} > limit {max_relevant_process_rss_mb:.1f}"
        )
    payload = {
        "memory": {
            "free_mb": round(memory.free_mb, 2),
            "compressor_mb": round(memory.compressor_mb, 2),
            "page_size": memory.page_size,
            "free_pages": memory.free_pages,
            "speculative_pages": memory.speculative_pages,
            "compressor_pages": memory.compressor_pages,
        },
        "relevant_processes": [
            {"pid": p.pid, "rss_mb": round(p.rss_mb, 2), "command": p.command}
            for p in relevant
        ],
        "limits": {
            "min_free_mb": min_free_mb,
            "max_compressor_mb": max_compressor_mb,
            "max_relevant_process_rss_mb": max_relevant_process_rss_mb,
        },
    }
    return not reasons, reasons, payload


def load_command(plan_path: Path, command_name: str) -> list[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    commands = dict(plan.get("commands") or {})
    command = commands.get(command_name)
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise SystemExit(f"Command {command_name!r} missing from {plan_path}")
    return list(command)


def write_status(output_dir: Path, status: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_repair_runner_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Feature Repair Runner Status",
        "",
        f"- Status: `{status['status']}`",
        f"- Command: `{status['command_name']}`",
        f"- Generated at: `{status['generated_at_utc']}`",
        f"- Free memory MB: `{status['safety']['memory']['free_mb']}`",
        f"- Compressor MB: `{status['safety']['memory']['compressor_mb']}`",
        "",
        "## Reasons",
        "",
    ]
    reasons = status.get("reasons") or []
    if reasons:
        lines.extend([f"- {reason}" for reason in reasons])
    else:
        lines.append("- none")
    (output_dir / "feature_repair_runner_status.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_repair(args: argparse.Namespace) -> int:
    command = load_command(args.plan_json, args.command_name)
    memory = collect_memory_snapshot()
    if args.trusted_relevant_process_rss_mb is not None:
        processes = trusted_process_snapshots(
            args.trusted_relevant_process_rss_mb,
            args.trusted_relevant_process_description,
        )
    else:
        processes = collect_process_snapshots()
    safe, reasons, safety = evaluate_safety(
        memory,
        processes,
        min_free_mb=args.min_free_mb,
        max_compressor_mb=args.max_compressor_mb,
        max_relevant_process_rss_mb=args.max_relevant_process_rss_mb,
    )
    output_dir = args.output_dir or args.plan_json.parent
    status: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_json": str(args.plan_json),
        "command_name": args.command_name,
        "command": command,
        "dry_run": bool(args.dry_run),
        "safety": safety,
        "reasons": reasons,
    }
    if not safe and not args.allow_low_memory:
        status["status"] = "deferred_memory_safety"
        write_status(output_dir, status)
        print(
            "[feature_repair_runner] deferred by memory safety: "
            + "; ".join(reasons),
            flush=True,
        )
        return 0 if args.exit_zero_on_defer else 2

    if args.dry_run:
        status["status"] = "dry_run_ready"
        write_status(output_dir, status)
        print("[feature_repair_runner] dry-run ready", flush=True)
        return 0

    log_path = output_dir / f"{args.command_name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    status["status"] = "completed" if proc.returncode == 0 else "failed"
    status["returncode"] = int(proc.returncode)
    status["log_path"] = str(log_path)
    write_status(output_dir, status)
    print(
        f"[feature_repair_runner] {status['status']} returncode={proc.returncode} "
        f"log={log_path}",
        flush=True,
    )
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--command-name", default="repair_minimum_scoreable_window")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-free-mb", type=float, default=1024.0)
    parser.add_argument("--max-compressor-mb", type=float, default=4096.0)
    parser.add_argument("--max-relevant-process-rss-mb", type=float, default=2048.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-low-memory", action="store_true")
    parser.add_argument("--exit-zero-on-defer", action="store_true")
    parser.add_argument("--trusted-relevant-process-rss-mb", type=float)
    parser.add_argument(
        "--trusted-relevant-process-description",
        default="external_shell_preflight",
    )
    raise SystemExit(run_repair(parser.parse_args()))


if __name__ == "__main__":
    main()
