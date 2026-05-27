#!/usr/bin/env python3
"""Persistent supervisor for the Kraken perps live-test inference loop."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_stamp(fmt: str) -> str:
    return datetime.now(timezone.utc).strftime(fmt)


def _log(path: Path, msg: str) -> None:
    line = f"[{_utc_stamp('%Y-%m-%dT%H:%M:%SZ')}] {msg}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
    print(line, end="", flush=True)


def main() -> int:
    root = Path("/Users/remyroche/Documents/Ares")
    os.chdir(root)

    run_id = os.environ.get("RUN_ID", "20260523_015947")
    data_root = os.environ.get("DATA_ROOT", "data_perp")
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    restart_delay = float(os.environ.get("RESTART_DELAY_SECONDS", "30"))
    max_restarts = int(os.environ.get("MAX_RESTARTS", "0"))
    inference_interval = os.environ.get("INFERENCE_INTERVAL", "60")
    challenger_interval = os.environ.get("CHALLENGER_INTERVAL", "60")
    exchange = os.environ.get("EPM_EXCHANGE", "kraken")
    live_data_root = os.environ.get("LIVE_DATA_ROOT", "")
    python_bin = os.environ.get(
        "PYTHON_BIN", "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = log_dir / f"live_test_supervisor_{_utc_stamp('%Y%m%d_%H%M%S')}.log"
    (log_dir / "live_test_supervisor.pid").write_text(
        str(os.getpid()), encoding="utf-8"
    )
    _log(
        supervisor_log,
        "supervisor_start "
        f"pid={os.getpid()} run_id={run_id} data_root={data_root} "
        f"exchange={exchange} inference_interval={inference_interval} "
        f"challenger_interval={challenger_interval}",
    )

    restart_count = 0
    while True:
        child_log = log_dir / f"live_test_kraken_perps_{_utc_stamp('%Y%m%d_%H%M%S')}.log"
        _log(
            supervisor_log,
            f"child_start restart_count={restart_count} log={child_log}",
        )
        cmd = [
            python_bin,
            "-u",
            "-m",
            "extreme_price_movements.inference.run_inference",
            "--live-test",
            "--perps",
            "--data-root",
            data_root,
            "--run-id",
            run_id,
            "--inference-interval",
            inference_interval,
            "--challenger-interval",
            challenger_interval,
        ]
        if live_data_root:
            cmd.extend(["--live-data-root", live_data_root])
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": ".",
                "MPLCONFIGDIR": "/private/tmp/ares_mplconfig",
                "EPM_EXCHANGE": exchange,
            }
        )
        with child_log.open("ab") as fh:
            proc = subprocess.run(cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT)
        _log(
            supervisor_log,
            f"child_exit restart_count={restart_count} exit_code={proc.returncode} log={child_log}",
        )
        restart_count += 1
        if max_restarts and restart_count >= max_restarts:
            _log(
                supervisor_log,
                f"supervisor_stop reason=max_restarts restart_count={restart_count}",
            )
            return int(proc.returncode)
        time.sleep(restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
