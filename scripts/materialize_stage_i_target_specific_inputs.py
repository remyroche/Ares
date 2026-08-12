#!/usr/bin/env python3
"""Build immutable, model-free inputs for direct-FQ3 Stage-I OOS replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_target_specific_input_materializer import (
    TargetSpecificInputMaterializationSpec,
    materialize_stage_i_target_specific_inputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selector-dir", type=Path, required=True)
    parser.add_argument("--meta-selector-dir", type=Path, required=True)
    parser.add_argument("--winner-bundle", type=Path, required=True)
    parser.add_argument(
        "--target-winner-dir", type=Path,
        help="S/O winner bundle containing winner_target_handoff.parquet; omit only for frozen R3",
    )
    parser.add_argument(
        "--shared-population-dir", type=Path,
        help="signed R3/scalar/ordinal common valid population for final joint comparison",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-validation-folds", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=500)
    args = parser.parse_args(argv)
    result = materialize_stage_i_target_specific_inputs(TargetSpecificInputMaterializationSpec(
        selector_dir=args.selector_dir,
        base_selector_dir=args.base_selector_dir,
        meta_selector_dir=args.meta_selector_dir,
        winner_bundle_path=args.winner_bundle,
        target_winner_dir=args.target_winner_dir,
        shared_population_dir=args.shared_population_dir,
        output_dir=args.output_dir,
        n_validation_folds=args.n_validation_folds,
        min_train_rows=args.min_train_rows,
    ))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
