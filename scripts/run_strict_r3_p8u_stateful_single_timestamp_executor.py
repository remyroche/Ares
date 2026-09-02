#!/usr/bin/env python3
"""Advance and score one P8U source hour from saved state only.

This is offline/preproduction infrastructure.  It has no account, exchange,
policy, portfolio, or order-submission path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_stateful_single_timestamp_executor import (
    P8UStatefulSingleTimestampConfig,
    P8UStatefulSingleTimestampExecutor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--regular-bootstrap-root", type=Path, required=True)
    parser.add_argument("--direct-bootstrap-root", type=Path, required=True)
    parser.add_argument("--regular-state-scope", required=True)
    parser.add_argument("--regular-state-components", nargs="+", required=True)
    parser.add_argument("--timestamp", required=True, help="One completed UTC source hour.")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    executor = P8UStatefulSingleTimestampExecutor(
        P8UStatefulSingleTimestampConfig(
            root=args.out_root.resolve(),
            bundle=args.bundle.resolve(),
            source_state=args.source_state.resolve(),
            regular_bootstrap_root=args.regular_bootstrap_root.resolve(),
            direct_bootstrap_root=args.direct_bootstrap_root.resolve(),
            regular_state_scope=str(args.regular_state_scope),
            regular_state_components=tuple(map(str, args.regular_state_components)),
        )
    )
    print(json.dumps(executor.advance(timestamp=args.timestamp), sort_keys=True))


if __name__ == "__main__":
    main()
