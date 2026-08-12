#!/usr/bin/env python3
"""Score only declared, checksummed frozen Stage-IV native model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_iv_native_artifact_runner import run_stage_iv_native_frozen_oos
from extreme_price_movements.stage_iv_native_materializer import load_stage_iv_native_frozen_oos_launch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-oos-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    launch = load_stage_iv_native_frozen_oos_launch(args.frozen_oos_spec)
    result = run_stage_iv_native_frozen_oos(
        launch.plans, output_directory=args.output_dir,
        admission_spec=launch.admission_spec,
    )
    print(json.dumps({
        "status": result.manifest["status"], "output_directory": str(result.output_directory),
        "frozen_only": True, "refit_forbidden": result.manifest["refit_forbidden"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
