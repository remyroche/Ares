#!/usr/bin/env python3
"""Materialise the frozen causal Stage-I-to-II direct-FQ3 handoff."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_to_stage_ii_handoff import StageIToStageIIHandoffSpec, materialize_stage_i_to_stage_ii_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-oos-dir", type=Path, required=True)
    parser.add_argument("--stage-i-inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_stage_i_to_stage_ii_handoff(StageIToStageIIHandoffSpec(
        stage_i_oos_dir=args.stage_i_oos_dir,
        stage_i_inputs_dir=args.stage_i_inputs_dir,
        output_dir=args.output_dir,
    )), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
