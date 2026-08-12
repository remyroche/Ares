#!/usr/bin/env python3
"""Execute the strict Stage-I meta automatic/20/30/40/60/full ladder.

The command never chooses or freezes a production winner.  Every scored count
emits a mandatory count-specific target-HPO/refit request; only a later
hash-bound refit artifact can make a count eligible for winner freezing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_meta_feature_challenger import (
    load_completed_stage_i_meta_selection,
    materialize_meta_feature_challenge,
)
from extreme_price_movements.stage_i_meta_feature_ladder import (
    run_pooled_meta_feature_ladder,
    run_side_meta_feature_ladder,
)
from extreme_price_movements.stage_i_meta_target_funnel import (
    MetaTargetSpec,
    focused_quantile_meta_target_specs,
)


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _specs(path: Path | None) -> tuple[MetaTargetSpec, ...]:
    if path is None:
        return focused_quantile_meta_target_specs()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("--meta-specs-json must be a non-empty JSON array")
    return tuple(MetaTargetSpec(**dict(item)) for item in value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-protected-json", type=Path)
    parser.add_argument("--meta-specs-json", type=Path)
    parser.add_argument("--n-validation-folds", type=int, default=4)
    parser.add_argument("--min-train-candidate-rows", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n_validation_folds < 1 or args.min_train_candidate_rows < 3:
        parser.error("validation folds/minimum candidate rows must be positive")
    policy = _json_object(args.required_protected_json) if args.required_protected_json else {}
    if not set(policy).issubset({"required_features", "protected_features"}):
        parser.error("required/protected JSON permits only required_features/protected_features")
    specs = _specs(args.meta_specs_json)
    results = {}
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
        results[side] = run_side_meta_feature_ladder(
            selector_dir=args.selector_dir,
            base_selection_dir=args.base_selection_dir,
            meta_selection_dir=args.meta_selection_dir,
            plan=plan,
            output_dir=args.output_dir / side,
            specs=specs,
            n_validation_folds=args.n_validation_folds,
            min_train_candidate_rows=args.min_train_candidate_rows,
            resume=args.resume,
        )
    pooled = run_pooled_meta_feature_ladder(
        long_dir=args.output_dir / "long",
        short_dir=args.output_dir / "short",
        output_dir=args.output_dir / "pooled_global",
        resume=args.resume,
    )
    print(json.dumps({
        "status": "complete",
        "output_dir": str(args.output_dir.resolve()),
        "sides": list(results),
        "sets": ["automatic_sparse", "top20", "top30", "top40", "top60", "full_input_control"],
        "target_arms": [spec.arm_id for spec in specs],
        "pooled_global_ranking": "after_side_local_21d_common_bps_mapping",
        "freeze_disposition": "count_specific_target_HPO_and_refit_required",
        "pooled_request_sha256": pooled["request_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
