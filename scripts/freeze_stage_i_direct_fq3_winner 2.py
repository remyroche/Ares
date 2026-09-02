#!/usr/bin/env python3
"""Publish one immutable Stage-I base + direct-FQ3 meta winner bundle.

The publisher is deliberately bound to the three-family joint-shortlist
contract. It cannot package an arbitrary completed selector pair as a Stage-I
finalist.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_adapter_winner_bundle import (
    build_stage_i_adapter_winner_bundle,
    freeze_stage_i_adapter_winner_bundle,
)


FAMILY_TO_TARGET = {
    "R3_control": "legacy_R3_multiclass3_control",
    "scalar_S": "soft_scalar_S",
    "ordinal_O": "cumulative_ordinal5_O",
}


def _canonical_sha(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-finalists-contract", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(FAMILY_TO_TARGET), required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.joint_finalists_contract.read_text())
    if raw.get("status") != "complete" or raw.get("contract_sha256") != _canonical_sha({k: v for k, v in raw.items() if k != "contract_sha256"}):
        raise ValueError("joint-finalist contract checksum/status drift")
    finalist = next((item for item in raw.get("finalists", ()) if item.get("family") == args.family), None)
    if finalist is None or finalist.get("must_advance_to_joint_base_meta_evaluation") is not True:
        raise ValueError(f"{args.family} is not an authorized joint finalist")
    bundle = build_stage_i_adapter_winner_bundle(
        base_selection_dir=args.base_selection_dir,
        meta_selection_dir=args.meta_selection_dir,
        code_revision=args.code_revision,
        run_id=args.run_id,
    )
    expected = FAMILY_TO_TARGET[args.family]
    observed = {cell.base_target_contract.family for cell in bundle.cells}
    if observed != {expected}:
        raise ValueError(f"winner bundle base family drift: expected {expected}, got {sorted(observed)}")
    status = freeze_stage_i_adapter_winner_bundle(bundle, args.output)
    print(json.dumps({
        "status": status, "output": str(args.output.resolve()), "bundle_sha256": bundle.sha256,
        "family": args.family,
        "joint_finalists_contract_sha256": raw["contract_sha256"],
        "shared_population_contract_sha256": raw.get("shared_population_contract_sha256"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
