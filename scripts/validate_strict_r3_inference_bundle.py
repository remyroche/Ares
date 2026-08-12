#!/usr/bin/env python3
"""Validate every sealed input and the active window of a strict-R3 bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import (  # noqa: E402
    StrictR3InferenceBundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    bundle = StrictR3InferenceBundle.load(args.bundle, root=ROOT)
    audit = bundle.validate(decision_ts=args.decision_ts)
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(f"immutable validation output exists: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"event": "verified", **audit}))


if __name__ == "__main__":
    main()
