#!/usr/bin/env python3
"""Run the bounded post-materialisation diagnostics for one strict-OOF transport.

The driver is deliberately diagnostic-only.  It refuses a partial
side/fold/head collection, then writes the fold-local health/G2-G3 recurrence
and cluster sweep separately from the candidate-level scalar covariance audit.
Neither component fits a model, chooses a feature, or touches final OOS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_strict_oof_leaf_reasoning import (  # noqa: E402
    discover_reasoning_artifacts,
    run as run_health_cluster,
)
from extreme_price_movements.strict_oof_reasoning_covariance import (  # noqa: E402
    StrictOOFReasoningCovarianceError,
    discover_strict_oof_reasoning_artifacts,
    run_strict_oof_reasoning_covariance,
)


REQUIRED_HEADS = frozenset(("p_adverse", "p_weak", "p_clear"))


def _validate_complete_collection(inputs: list[Path]) -> list[Path]:
    health_paths = discover_reasoning_artifacts(inputs)
    covariance_paths = discover_strict_oof_reasoning_artifacts(inputs)
    if set(health_paths) != set(covariance_paths):
        raise StrictOOFReasoningCovarianceError(
            "health and covariance discovery disagree on strict-OOF artifact paths"
        )
    cells: dict[tuple[str, str], set[str]] = {}
    for path in health_paths:
        payload = json.loads((path / "base_reasoning_manifest.json").read_text(encoding="utf-8"))
        if payload.get("status") != "MATERIALIZED_STRICT_OOF":
            raise StrictOOFReasoningCovarianceError(f"not materialised strict OOF: {path}")
        key = (str(payload.get("side_name", "")).lower(), str(payload.get("fold_id", "")))
        head = str(payload.get("head_name", ""))
        if not key[0] or not key[1] or not head:
            raise StrictOOFReasoningCovarianceError(f"missing side/fold/head in {path}")
        cells.setdefault(key, set()).add(head)
    incomplete = {
        f"{side}/{fold}": sorted(REQUIRED_HEADS.difference(heads))
        for (side, fold), heads in cells.items()
        if heads != REQUIRED_HEADS
    }
    if incomplete:
        raise StrictOOFReasoningCovarianceError(
            f"refusing incomplete strict-OOF side/fold/head collection: {incomplete}"
        )
    sides = {side for side, _fold in cells}
    if sides != {"long", "short"}:
        raise StrictOOFReasoningCovarianceError(
            f"strict-OOF transport requires both canonical sides, got {sorted(sides)}"
        )
    return health_paths


def run(*, inputs: list[Path], output_dir: Path) -> Path:
    """Produce two fresh immutable diagnostic sub-artifacts for a completed transport."""

    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = _validate_complete_collection(inputs)
    output_dir.mkdir(parents=True)
    try:
        health_cluster = run_health_cluster(paths, output_dir / "health_recurrence_cluster")
        covariance = run_strict_oof_reasoning_covariance(paths, output_dir / "covariance")
        manifest = {
            "schema": "feature_leaf_reasoning_transport_diagnostics_v1",
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "input_artifact_count": len(paths),
            "side_fold_cells": len(paths) // len(REQUIRED_HEADS),
            "required_heads": sorted(REQUIRED_HEADS),
            "inputs": [str(path) for path in paths],
            "outputs": {
                "health_recurrence_cluster": str(health_cluster.relative_to(output_dir)),
                "covariance": str(covariance.relative_to(output_dir)),
            },
            "inference_or_selection": "none; diagnostics require a separately predeclared nested meta arm before use",
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return output_dir
    except Exception:
        # Preserve the source artifacts.  The failed destination contains no
        # complete manifest and is intentionally left for the caller to audit.
        raise


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="completed strict_oof_base_reasoning transport root or artifact directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    print(run(inputs=args.input, output_dir=args.output_dir))
