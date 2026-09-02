#!/usr/bin/env python3
"""Launch one sealed strict-R3 service outside the caller's process group.

This contains no model, admission, policy, portfolio or exchange logic.  It
only gives an already-reviewed service command its own session so a short
operator terminal command cannot terminate a live singleton on return.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("provide a service command after --")
    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    pid_path = args.pid_file if args.pid_file.is_absolute() else ROOT / args.pid_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    child = os.fork()
    if child:
        print(child, flush=True)
        return 0
    os.setsid()
    os.chdir(ROOT)
    stdin_fd = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(stdin_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    finally:
        os.close(stdin_fd)
        os.close(log_fd)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("NUMBA_CACHE_DIR", "/private/tmp/ares_numba_cache")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/ares_matplotlib")
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
