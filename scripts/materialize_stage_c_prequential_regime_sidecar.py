#!/usr/bin/env python3
"""Materialize the fail-closed candidate-level Stage-C F7 sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_c_prequential_regime_sidecar import Inputs, run


ART = ROOT / "data_perp/artifacts"
SIDE = ART / "authoritative_soft_regime_transition_sidecars_20260730_v1"
DEFAULT_OUTPUT = ART / "stage_c_candidate_prequential_regime_sidecar_20260801_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=ART / "stage_c_continuation_feature_panel_20260731_v2/stage_c_candidate_population.parquet")
    parser.add_argument("--sidecar-root", type=Path, default=SIDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    side = args.sidecar_root
    inputs = Inputs(args.candidates, side / "soft_regime_hourly.parquet", side / "soft_transition_hourly.parquet", side / "manifest.json")
    print(json.dumps(run(inputs=inputs, output=args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
