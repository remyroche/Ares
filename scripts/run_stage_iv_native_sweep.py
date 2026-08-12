#!/usr/bin/env python3
"""Run explicit hash-bound Stage-IV native broad-to-tail cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_iv_native_artifact_runner import (
    run_stage_iv_native_artifact_sweep,
)
from extreme_price_movements.stage_iv_native_materializer import (
    direct_fq3_winner_meta_fitter,
    load_stage_iv_native_launch,
    native_winner_base_fitter,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="verify and reuse completed atomic cell checkpoints",
    )
    args = parser.parse_args(argv)
    launch = load_stage_iv_native_launch(args.cell_spec)
    result = run_stage_iv_native_artifact_sweep(
        launch.cells, output_directory=args.output_dir,
        checkpoint_directory=args.checkpoint_dir, resume=args.resume,
        base_fitter=native_winner_base_fitter,
        meta_fitter=direct_fq3_winner_meta_fitter,
        spec=launch.runner_spec, launch_manifest=launch.launch_manifest,
    )
    print(json.dumps({
        "status": "complete", "output_directory": str(result.output_directory),
        "winner_cell_id": result.winner["cell_id"],
        "decision": result.winner["decision"],
        "resumed_cell_count": result.manifest["resume"]["resumed_cell_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
