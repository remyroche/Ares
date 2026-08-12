#!/usr/bin/env python3
"""Run the frozen, checkpointed Stage-I guarded meta-target funnel.

Both sides are run by default. A two-side invocation additionally publishes
the only production-interpretable comparison: causal 21-day side-local bps
mapping/admission followed by one pooled-global ranking. A one-side invocation
is explicitly diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extreme_price_movements.stage_i_meta_target_execution import (
    run_pooled_global_meta_target_evaluation,
    run_side_meta_target_funnel,
)
from extreme_price_movements.stage_i_meta_target_funnel import (
    default_meta_target_specs,
    focused_quantile_meta_target_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("long", "short"), action="append", default=[])
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument(
        "--arm-set",
        choices=("focused-tercile", "full"),
        default="focused-tercile",
        help=(
            "focused-tercile runs only the fold-q33/q67 classifier plus current "
            "Huber and mandatory model-free controls; full runs the broad legacy funnel"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.validation_folds < 1 or args.min_train_rows < 3:
        parser.error("validation folds/minimum training rows must be positive and >=3")
    sides = tuple(dict.fromkeys(args.side or ("long", "short")))
    specs = (
        focused_quantile_meta_target_specs()
        if args.arm_set == "focused-tercile"
        else default_meta_target_specs()
    )
    results = {}
    for side in sides:
        results[side] = run_side_meta_target_funnel(
            selector_dir=args.selector_dir,
            base_selection_dir=args.base_selection_dir,
            meta_selection_dir=args.meta_selection_dir,
            output_dir=args.output_dir / side,
            side=side,
            n_validation_folds=args.validation_folds,
            min_train_rows=args.min_train_rows,
            resume=args.resume,
            specs=specs,
        )
    pooled = None
    if set(sides) == {"long", "short"}:
        pooled = run_pooled_global_meta_target_evaluation(
            long_dir=args.output_dir / "long",
            short_dir=args.output_dir / "short",
            output_dir=args.output_dir / "pooled_global",
            resume=args.resume,
        )
    print(json.dumps({
        "status": "complete", "sides": list(sides),
        "arm_set": args.arm_set,
        "side_decisions": {side: value["decision"] for side, value in results.items()},
        "pooled_global_decision": pooled["decision"] if pooled else None,
        "production_interpretation": (
            "pooled_global_after_causal_common_bps_mapping"
            if pooled else "blocked_single_side_diagnostic_only"
        ),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
