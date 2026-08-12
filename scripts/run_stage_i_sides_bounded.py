#!/usr/bin/env python3
"""Run long/short Stage-I selection with bounded process/thread parallelism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_side_orchestrator import (
    SideOrchestratorRequest,
    orchestrate_sides,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("base", "meta"), required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path)
    parser.add_argument("--target-winner-dir", type=Path)
    parser.add_argument("--meta-mode", choices=("legacy", "direct_fq3"), default="legacy")
    parser.add_argument("--required-regime-feature", action="append", default=[])
    parser.add_argument("--required-context-feature", action="append", default=[])
    parser.add_argument("--base-candidate-fraction", type=float, default=1.0)
    parser.add_argument("--target-neutral-cache-dir", type=Path)
    parser.add_argument("--feature-sidecar", type=Path)
    parser.add_argument("--feature-sidecar-field", action="append", default=[])
    parser.add_argument("--feature-sidecar-min-coverage", type=float, default=0.90)
    parser.add_argument("--hpo-trials", type=int, default=60)
    parser.add_argument("--hpo-patience", type=int, default=15)
    parser.add_argument(
        "--correlation-policy",
        choices=("grouped-preserve", "pre-mda-spearman-representative"),
        default="grouped-preserve",
    )
    parser.add_argument(
        "--dedicated-mda-reference", choices=("none", "full-selector-side"),
        default="full-selector-side",
    )
    parser.add_argument(
        "--mda-support-mode", choices=("full", "target-only"), default="full",
        help="Use full realised-path MDA support or the target-only control.",
    )
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    parser.add_argument("--worker-memory-gib", type=float)
    parser.add_argument(
        "--side", action="append", choices=("long", "short"),
        help="Run only the specified side(s); useful for an auditable interrupted-side restart.",
    )
    args = parser.parse_args()
    request = SideOrchestratorRequest(
        layer=args.layer,
        selector_dir=str(args.selector_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        base_selection_dir=(
            str(args.base_selection_dir.resolve()) if args.base_selection_dir else None
        ),
        target_winner_dir=(
            str(args.target_winner_dir.resolve()) if args.target_winner_dir else None
        ),
        meta_mode=args.meta_mode,
        required_regime_features=tuple(dict.fromkeys(args.required_regime_feature)),
        required_context_features=tuple(dict.fromkeys(args.required_context_feature)),
        base_candidate_fraction=float(args.base_candidate_fraction),
        target_neutral_cache_dir=(
            str(args.target_neutral_cache_dir.resolve()) if args.target_neutral_cache_dir else None
        ),
        feature_sidecar=(str(args.feature_sidecar.resolve()) if args.feature_sidecar else None),
        feature_sidecar_fields=tuple(dict.fromkeys(map(str, args.feature_sidecar_field))),
        feature_sidecar_min_coverage=float(args.feature_sidecar_min_coverage),
        hpo_trials=int(args.hpo_trials),
        hpo_patience=int(args.hpo_patience),
        correlation_policy=args.correlation_policy,
        dedicated_mda_reference=args.dedicated_mda_reference,
        mda_support_mode=args.mda_support_mode,
        reserve_gib=float(args.reserve_gib),
        worker_memory_gib=args.worker_memory_gib,
        sides=tuple(args.side or ("long", "short")),
    )
    result = orchestrate_sides(request)
    print(json.dumps({
        "status": result["status"],
        "execution_mode": result["execution_mode"],
        "request_sha256": result["request_sha256"],
        "side_manifest_sha256": result.get("side_manifest_sha256", {}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
