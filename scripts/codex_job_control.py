#!/usr/bin/env python3
"""Start and stop Codex-managed Ares Python jobs through a PID registry."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "logs" / "codex_job_control" / "registry.json"


class JobControlError(RuntimeError):
    """Raised when a job-control operation is unsafe or invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(value: Any) -> str:
    return str(value)


def _registry_path() -> Path:
    raw = os.environ.get("CODEX_JOB_CONTROL_REGISTRY")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_REGISTRY_PATH


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workspace": str(ROOT),
        "updated_at": _utc_now(),
        "jobs": {},
    }


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or _registry_path()
    if not registry_path.exists():
        return _empty_registry()
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JobControlError(
            f"Invalid registry JSON at {registry_path}: {exc}"
        ) from exc
    if not isinstance(registry, dict):
        raise JobControlError(
            f"Invalid registry payload at {registry_path}: not an object"
        )
    jobs = registry.setdefault("jobs", {})
    if not isinstance(jobs, dict):
        raise JobControlError(
            f"Invalid registry jobs at {registry_path}: not an object"
        )
    registry.setdefault("schema_version", 1)
    registry.setdefault("workspace", str(ROOT))
    return registry


def save_registry(registry: dict[str, Any], path: Path | None = None) -> None:
    registry_path = path or _registry_path()
    registry["updated_at"] = _utc_now()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry_path.with_suffix(registry_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(registry, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(registry_path)


def _resolve_under_root(path: str | os.PathLike[str], *, kind: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise JobControlError(f"{kind} must be inside {ROOT}: {resolved}") from exc
    return resolved


def _clean_remainder(command: Sequence[str]) -> list[str]:
    out = list(command)
    if out and out[0] == "--":
        out = out[1:]
    if not out:
        raise JobControlError("Missing command after --")
    return out


def _python_argv_index(argv: Sequence[str]) -> int | None:
    if not argv:
        return None
    if Path(argv[0]).name.lower().startswith("python"):
        return 0
    if Path(argv[0]).name.lower() == "env":
        for idx, token in enumerate(argv[1:], start=1):
            if token == "--":
                continue
            if token.startswith("-"):
                continue
            if "=" in token and not token.startswith("="):
                continue
            if Path(token).name.lower().startswith("python"):
                return idx
            return None
    return None


def command_is_python(argv: Sequence[str]) -> bool:
    return _python_argv_index(argv) is not None


def _command_fingerprint(argv: Sequence[str]) -> list[str]:
    py_idx = _python_argv_index(argv)
    if py_idx is None:
        return []
    fingerprint = [Path(argv[py_idx]).name]
    fingerprint.extend(str(token) for token in argv[py_idx + 1 :])
    return fingerprint


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _run_text(command: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _process_command(pid: int) -> str | None:
    return _run_text(["ps", "-p", str(pid), "-o", "command="])


def _process_lstart(pid: int) -> str | None:
    return _run_text(["ps", "-p", str(pid), "-o", "lstart="])


def _process_lstart_retry(
    pid: int, attempts: int = 10, delay_seconds: float = 0.1
) -> str | None:
    for _ in range(max(1, int(attempts))):
        lstart = _process_lstart(pid)
        if lstart:
            return lstart
        time.sleep(max(0.0, float(delay_seconds)))
    return None


def _current_pgid(pid: int) -> int | None:
    try:
        return int(os.getpgid(pid))
    except OSError:
        return None


def _process_cwd(pid: int) -> Path | None:
    proc_cwd = Path(f"/proc/{pid}/cwd")
    if proc_cwd.exists():
        try:
            return proc_cwd.resolve()
        except OSError:
            return None

    out = _run_text(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"])
    if not out:
        return None
    for line in out.splitlines():
        if line.startswith("n"):
            try:
                return Path(line[1:]).resolve()
            except OSError:
                return None
    return None


def _command_matches(entry: dict[str, Any], live_command: str | None) -> bool:
    if not live_command:
        return False
    expected = _command_fingerprint(entry.get("argv") or [])
    if not expected:
        return False
    first, *rest = expected
    tokens = shlex.split(live_command)
    if not tokens:
        return False
    live_basenames = {Path(token).name.lower() for token in tokens[:3]}
    first_lower = first.lower()
    if first_lower.startswith("python"):
        first_matches = any(token.startswith("python") for token in live_basenames)
    else:
        first_matches = first_lower in live_basenames
    if not first_matches and first not in live_command:
        return False
    return all(str(token) in live_command for token in rest)


def validate_entry_for_stop(entry: dict[str, Any]) -> dict[str, Any]:
    pid = int(entry.get("pid") or 0)
    if pid <= 0:
        raise JobControlError(f"Invalid registered pid: {entry.get('pid')!r}")
    if entry.get("started_by") != "codex_job_control":
        raise JobControlError(f"Refusing pid={pid}: not started by codex_job_control")
    if not command_is_python(entry.get("argv") or []):
        raise JobControlError(f"Refusing pid={pid}: registered command is not Python")
    if not _process_alive(pid):
        return {"pid": pid, "alive": False, "safe": True, "reason": "not_running"}

    entry_cwd = _resolve_under_root(entry.get("cwd") or ".", kind="registered cwd")
    live_cwd = _process_cwd(pid)
    cwd_verified = False
    if live_cwd is not None:
        try:
            live_cwd.relative_to(ROOT)
        except ValueError as exc:
            raise JobControlError(
                f"Refusing pid={pid}: live cwd is outside {ROOT}: {live_cwd}"
            ) from exc
        if live_cwd != entry_cwd:
            raise JobControlError(
                f"Refusing pid={pid}: live cwd changed from {entry_cwd} to {live_cwd}"
            )
        cwd_verified = True

    live_lstart = _process_lstart(pid)
    registered_lstart = str(entry.get("lstart") or "").strip()
    if registered_lstart and live_lstart and live_lstart != registered_lstart:
        raise JobControlError(
            f"Refusing pid={pid}: process start time changed; pid was reused"
        )
    if registered_lstart and not live_lstart:
        raise JobControlError(
            f"Refusing pid={pid}: could not verify process start time"
        )

    live_command = _process_command(pid)
    if not _command_matches(entry, live_command):
        raise JobControlError(
            f"Refusing pid={pid}: live command does not match registry entry"
        )

    live_pgid = _current_pgid(pid)
    registered_pgid = int(entry.get("pgid") or 0)
    if registered_pgid > 0 and live_pgid is not None and live_pgid != registered_pgid:
        raise JobControlError(
            f"Refusing pid={pid}: process group changed from {registered_pgid} to {live_pgid}"
        )

    return {
        "pid": pid,
        "pgid": registered_pgid,
        "alive": True,
        "safe": True,
        "cwd_verified": cwd_verified,
        "command": live_command,
    }


def _send_signal(pid: int, pgid: int | None, sig: signal.Signals) -> str:
    if pgid and pgid == pid:
        os.killpg(pgid, sig)
        return f"pgid:{pgid}"
    os.kill(pid, sig)
    return f"pid:{pid}"


def _wait_until_stopped(pid: int, timeout_seconds: float) -> bool:
    deadline = time.time() + max(0.0, float(timeout_seconds))
    while time.time() < deadline:
        if not _process_alive(pid):
            return True
        time.sleep(0.2)
    return not _process_alive(pid)


def _selected_entries(
    registry: dict[str, Any],
    *,
    pid: int | None = None,
    name: str | None = None,
    all_jobs: bool = False,
) -> list[dict[str, Any]]:
    jobs = registry.get("jobs") or {}
    entries = [job for job in jobs.values() if isinstance(job, dict)]
    if all_jobs:
        return entries
    if pid is not None:
        entry = jobs.get(str(pid))
        if not isinstance(entry, dict):
            raise JobControlError(f"PID {pid} is not registered")
        return [entry]
    if name:
        selected = [entry for entry in entries if entry.get("name") == name]
        if not selected:
            raise JobControlError(f"No registered job named {name!r}")
        return selected
    raise JobControlError("Specify --pid, --name, or --all")


def start_job(args: argparse.Namespace) -> int:
    argv = _clean_remainder(args.command)
    if not command_is_python(argv):
        raise JobControlError("codex_job_control only starts Python commands")
    cwd = _resolve_under_root(args.cwd, kind="cwd")
    log_path = _resolve_under_root(args.log, kind="log path")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", ".")
    with log_path.open("ab") as log_fh:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    time.sleep(0.05)
    pid = int(proc.pid)
    pgid = _current_pgid(pid) or pid
    entry = {
        "pid": pid,
        "pgid": pgid,
        "name": args.name,
        "status": "running",
        "started_at": _utc_now(),
        "started_by": "codex_job_control",
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "log": str(log_path),
        "argv": argv,
        "lstart": _process_lstart_retry(pid),
    }
    registry = load_registry()
    registry.setdefault("jobs", {})[str(pid)] = entry
    save_registry(registry)
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def register_job(args: argparse.Namespace) -> int:
    pid = int(args.pid)
    if pid <= 0:
        raise JobControlError("--pid must be positive")
    if not _process_alive(pid):
        raise JobControlError(f"PID {pid} is not running")
    argv = _clean_remainder(args.command) if args.command else None
    if argv is not None and not command_is_python(argv):
        raise JobControlError("registered command must be Python")
    live_command = _process_command(pid)
    if argv is None:
        argv = shlex.split(live_command or "")
    if not command_is_python(argv):
        raise JobControlError(f"PID {pid} does not look like a Python job")
    cwd = _process_cwd(pid) or _resolve_under_root(args.cwd, kind="cwd")
    try:
        cwd.relative_to(ROOT)
    except ValueError as exc:
        raise JobControlError(f"PID {pid} cwd is outside {ROOT}: {cwd}") from exc
    pgid = _current_pgid(pid) or pid
    entry = {
        "pid": pid,
        "pgid": pgid,
        "name": args.name,
        "status": "running",
        "started_at": _utc_now(),
        "started_by": "codex_job_control",
        "registered_existing_process": True,
        "workspace": str(ROOT),
        "cwd": str(cwd),
        "log": str(_resolve_under_root(args.log, kind="log path")) if args.log else "",
        "argv": argv,
        "lstart": _process_lstart_retry(pid),
    }
    registry = load_registry()
    registry.setdefault("jobs", {})[str(pid)] = entry
    save_registry(registry)
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def list_jobs(args: argparse.Namespace) -> int:
    registry = load_registry()
    jobs = registry.get("jobs") or {}
    out = []
    for entry in jobs.values():
        if not isinstance(entry, dict):
            continue
        pid = int(entry.get("pid") or 0)
        alive = _process_alive(pid) if pid > 0 else False
        if args.active and not alive:
            continue
        out.append({**entry, "alive": alive})
    print(json.dumps({"jobs": out}, indent=2, sort_keys=True))
    return 0


def stop_jobs(args: argparse.Namespace) -> int:
    registry = load_registry()
    entries = _selected_entries(
        registry,
        pid=args.pid,
        name=args.name,
        all_jobs=bool(args.all),
    )
    results: list[dict[str, Any]] = []
    for entry in entries:
        pid = int(entry.get("pid") or 0)
        validation = validate_entry_for_stop(entry)
        if not validation.get("alive"):
            entry["status"] = "exited"
            entry["stopped_at"] = entry.get("stopped_at") or _utc_now()
            results.append({"pid": pid, "status": "not_running"})
            continue
        if args.dry_run:
            results.append({"pid": pid, "status": "dry_run", **validation})
            continue

        target = _send_signal(
            pid,
            int(validation.get("pgid") or 0),
            signal.SIGTERM,
        )
        stopped = _wait_until_stopped(pid, args.timeout)
        escalated = False
        if not stopped:
            _send_signal(pid, int(validation.get("pgid") or 0), signal.SIGKILL)
            escalated = True
            stopped = _wait_until_stopped(pid, 2.0)
        entry["status"] = "stopped" if stopped else "stop_requested"
        entry["stopped_at"] = _utc_now()
        entry["stop_target"] = target
        entry["stop_escalated_to_sigkill"] = escalated
        results.append(
            {
                "pid": pid,
                "status": entry["status"],
                "target": target,
                "sigkill": escalated,
            }
        )
    save_registry(registry)
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


def prune_jobs(args: argparse.Namespace) -> int:
    registry = load_registry()
    jobs = registry.get("jobs") or {}
    removed: list[int] = []
    for pid_key, entry in list(jobs.items()):
        if not isinstance(entry, dict):
            continue
        pid = int(entry.get("pid") or 0)
        if pid <= 0 or not _process_alive(pid):
            if args.remove:
                jobs.pop(pid_key, None)
            else:
                entry["status"] = "exited"
                entry["stopped_at"] = entry.get("stopped_at") or _utc_now()
            removed.append(pid)
    save_registry(registry)
    print(json.dumps({"pruned": removed, "removed": bool(args.remove)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)

    start = sub.add_parser("start", help="start a detached Python job and register it")
    start.add_argument("--name", required=True)
    start.add_argument("--cwd", default=str(ROOT))
    start.add_argument("--log", required=True)
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(func=start_job)

    register = sub.add_parser("register", help="register an existing Python job")
    register.add_argument("--pid", type=int, required=True)
    register.add_argument("--name", required=True)
    register.add_argument("--cwd", default=str(ROOT))
    register.add_argument("--log", default="")
    register.add_argument("command", nargs=argparse.REMAINDER)
    register.set_defaults(func=register_job)

    list_cmd = sub.add_parser("list", help="list registered jobs")
    list_cmd.add_argument("--active", action="store_true")
    list_cmd.set_defaults(func=list_jobs)

    stop = sub.add_parser("stop", help="stop registered jobs after safety checks")
    target = stop.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int)
    target.add_argument("--name")
    target.add_argument("--all", action="store_true")
    stop.add_argument("--timeout", type=float, default=10.0)
    stop.add_argument("--dry-run", action="store_true")
    stop.set_defaults(func=stop_jobs)

    prune = sub.add_parser("prune", help="mark or remove exited jobs")
    prune.add_argument("--remove", action="store_true")
    prune.set_defaults(func=prune_jobs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except JobControlError as exc:
        print(f"codex_job_control: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
