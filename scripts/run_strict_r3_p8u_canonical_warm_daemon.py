#!/usr/bin/env python3
"""Run the offline canonical P8U feature-and-model process warm and resumable.

This service is deliberately research-only.  It loads one sealed P8U stack
once, then consumes immutable single-hour requests from a directory in order.
The Router/Base/Under/MC1 objects, full primitive source panel, and transform
state remain warm for the lifetime of the process.  A restart resumes from the
last atomically committed state receipt; it never rebuilds history or advances
over a missing hour.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_warm_runtime import (
    P8UCanonicalWarmRequest,
    P8UCanonicalWarmRuntime,
)
from extreme_price_movements.inference.p8u_warm_feature_state import utc_timestamp


def _request_paths(directory: Path, runtime: P8UCanonicalWarmRuntime) -> list[Path]:
    ledger = runtime._ledger()  # Read-only inspection; advance owns all mutation.
    last = utc_timestamp(ledger["last_signal_ts"]) if ledger is not None else None
    prepared: list[tuple[object, Path]] = []
    for path in sorted(directory.glob("request_*.json")):
        request = P8UCanonicalWarmRequest.load(path, root=runtime.config.root)
        if last is None or request.signal_ts > last:
            prepared.append((request.signal_ts, path))
    return [path for _timestamp, path in sorted(prepared)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request", action="append", default=[])
    parser.add_argument("--request-dir", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    if not args.request and args.request_dir is None:
        raise ValueError("provide --request or --request-dir")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    runtime = P8UCanonicalWarmRuntime.load(config_path=args.config, bundle_path=args.bundle, root=ROOT)
    root = runtime.root
    lock_path = root / ".canonical_warm_daemon.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another P8U canonical warm daemon is already active") from exc
        while True:
            queued = [Path(path).resolve() for path in args.request]
            if args.request_dir is not None:
                directory = args.request_dir.resolve()
                if not directory.is_dir():
                    raise NotADirectoryError(directory)
                queued.extend(_request_paths(directory, runtime))
            # Duplicate immutable request paths are always a caller error;
            # deduplicate only exact repeated CLI values before validation.
            queued = list(dict.fromkeys(queued))
            if queued:
                parsed = [P8UCanonicalWarmRequest.load(path, root=runtime.config.root) for path in queued]
                for request in sorted(parsed, key=lambda item: item.signal_ts):
                    ledger = runtime._ledger()
                    if ledger is not None and request.signal_ts <= utc_timestamp(ledger["last_signal_ts"]):
                        continue
                    print(json.dumps(runtime.advance(request), sort_keys=True), flush=True)
                    completed += 1
                    if args.max_requests is not None and completed >= args.max_requests:
                        return
                # Explicit request arguments are one-shot; a follower only
                # keeps consuming newly arrived directory requests.
                args.request = []
            if not args.follow:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
