#!/usr/bin/env python3
"""Create or validate an immutable strict-R3 live-runtime checkpoint.

This is deliberately offline.  It only reads local run artifacts and never
opens an exchange connection or writes to a live portfolio state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.strict_r3_runtime_checkpoint import (  # noqa: E402
    RuntimeCheckpointRequest,
    build_runtime_checkpoint,
    utc,
    validate_runtime_checkpoint,
    write_runtime_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a new immutable checkpoint")
    create.add_argument("--run-dir", type=Path, required=True)
    create.add_argument("--inference-bundle", type=Path, required=True)
    create.add_argument("--feature-state-bundle", type=Path, required=True)
    create.add_argument("--portfolio-state", type=Path, required=True)
    create.add_argument("--out-dir", type=Path, required=True)
    create.add_argument(
        "--feature-state-as-of-ts",
        help="UTC signal-bar timestamp; defaults to the run manifest signal_ts",
    )
    validate = commands.add_parser("validate", help="rehash and validate a checkpoint")
    validate.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "validate":
        result = validate_runtime_checkpoint(root=ROOT, checkpoint_dir=args.checkpoint_dir)
        print(json.dumps(result, sort_keys=True))
        return

    top = json.loads((args.run_dir / "run_manifest.json").read_text())
    state_ts = utc(args.feature_state_as_of_ts or top["signal_ts"])
    checkpoint = build_runtime_checkpoint(
        RuntimeCheckpointRequest(
            root=ROOT,
            run_dir=args.run_dir,
            inference_bundle=args.inference_bundle,
            feature_state_bundle=args.feature_state_bundle,
            feature_state_as_of_ts=state_ts,
            portfolio_state=args.portfolio_state,
        )
    )
    target = write_runtime_checkpoint(checkpoint=checkpoint, out_dir=args.out_dir)
    result = validate_runtime_checkpoint(root=ROOT, checkpoint_dir=target.parent)
    print(json.dumps({"checkpoint": str(target), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
