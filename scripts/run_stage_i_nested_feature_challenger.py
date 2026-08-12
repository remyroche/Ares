#!/usr/bin/env python3
"""Materialise and evaluate Stage-I nested feature challengers through explicit hooks.

No default trainer is provided.  Callers must name importable base and meta
stack callbacks so this diagnostic cannot accidentally launch against a
different split, target, or OOF row population.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_nested_feature_challenger import (
    MetaTargetMetricSpec,
    checkpoint_nested_feature_plan,
    evaluate_nested_feature_challenge,
    load_completed_stage_i_base_selection,
    materialize_nested_feature_challenge,
)


def _callback(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("callbacks must use module_path:callable_name")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise ValueError(f"callback is not callable: {reference}")
    return value


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--required-protected-json", type=Path, required=True,
                        help="JSON object with optional required_features/protected_features arrays.")
    parser.add_argument("--meta-specs-json", type=Path, required=True,
                        help="JSON array of target-specific MetaTargetMetricSpec objects.")
    # Required by design: without both hooks this command cannot even write a
    # materialisation checkpoint, avoiding misleading half-runs.
    parser.add_argument("--base-hook", required=True, help="module_path:callable_name")
    parser.add_argument("--meta-hook", required=True, help="module_path:callable_name")
    args = parser.parse_args()
    policy = _json_object(args.required_protected_json)
    specs_raw = json.loads(args.meta_specs_json.read_text(encoding="utf-8"))
    if not isinstance(specs_raw, list):
        raise ValueError("meta-specs JSON must be an array")
    specs = tuple(MetaTargetMetricSpec(**item) for item in specs_raw)
    source = load_completed_stage_i_base_selection(args.base_selection_dir, side=args.side)
    plan = materialize_nested_feature_challenge(
        source,
        required_features=policy.get("required_features", ()),
        protected_features=policy.get("protected_features", ()),
    )
    checkpoint_nested_feature_plan(plan, args.checkpoint_dir)
    result = evaluate_nested_feature_challenge(
        plan, base_hook=_callback(args.base_hook), meta_hook=_callback(args.meta_hook), meta_specs=specs,
    )
    result_path = args.checkpoint_dir / "evaluation.json"
    if result_path.exists():
        raise FileExistsError(f"evaluation checkpoint already exists: {result_path}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
