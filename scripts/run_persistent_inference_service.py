#!/usr/bin/env python3
"""Run an approved inference service script under Codex job control."""

from __future__ import annotations

import argparse
import signal
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICES = {
    "live": ROOT / "scripts" / "run_kraken_perps_live_supervisor.sh",
    "parity-monitor": ROOT / "scripts" / "run_live_parity_monitor_loop.sh",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=sorted(SERVICES))
    args = parser.parse_args()

    child = subprocess.Popen(["/bin/bash", str(SERVICES[args.service])], cwd=ROOT)

    def _forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    return int(child.wait())


if __name__ == "__main__":
    raise SystemExit(main())
