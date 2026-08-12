#!/usr/bin/env python3
"""Materialize the immutable Stage-I meta feature-count ladder for both sides."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_meta_feature_challenger import (
    checkpoint_meta_feature_plan,
    load_completed_stage_i_meta_selection,
    materialize_meta_feature_challenge,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--required-protected-json",
        type=Path,
        help=(
            "Optional JSON object with required_features and protected_features. "
            "The completed selector's base/trust handoff is mandatory regardless."
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite meta challenger plan: {args.output_dir}"
        )
    policy = (
        json.loads(args.required_protected_json.read_text(encoding="utf-8"))
        if args.required_protected_json
        else {}
    )
    if not isinstance(policy, dict):
        raise ValueError("required/protected contract must be a JSON object")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent
        )
    )
    try:
        side_manifests = {}
        for side in ("long", "short"):
            source = load_completed_stage_i_meta_selection(
                args.meta_selection_dir / side,
                side=side,
                selector_dir=args.selector_dir,
                base_selection_dir=args.base_selection_dir,
            )
            plan = materialize_meta_feature_challenge(
                source,
                required_features=policy.get("required_features", ()),
                protected_features=policy.get("protected_features", ()),
            )
            checkpoint_meta_feature_plan(plan, temporary / side)
            side_manifests[side] = {
                "plan_sha256": plan.plan_hash,
                "manifest_sha256": _sha(temporary / side / "manifest.json"),
                "frozen_base_oof_sha256": plan.frozen_base_oof_sha256,
                "automatic_feature_count": len(plan.feature_sets[0].features),
                "full_input_feature_count": len(plan.feature_sets[1].features),
            }
        payload = {
            "schema": "stage_i_meta_feature_challenger_bundle_v1",
            "status": "materialized",
            "comparison_scope": (
                "sequential_frozen_same_side_base_oof; vary_meta_features_only"
            ),
            "selector_manifest_sha256": _sha(args.selector_dir / "manifest.json"),
            "selector_feature_contract_sha256": _sha(
                args.selector_dir / "selector_feature_contract.json"
            ),
            "required_protected_contract_sha256": (
                _sha(args.required_protected_json)
                if args.required_protected_json
                else None
            ),
            "sides": side_manifests,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, args.output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

