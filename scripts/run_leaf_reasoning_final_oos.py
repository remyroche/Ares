#!/usr/bin/env python3
"""Run the one-time frozen November-2024 leaf-reasoning final OOS replay.

This CLI has no arguments for training, feature selection, HPO, clustering,
successor selection, calibration fitting, or refitting.  All of those must be
sealed in --frozen-contract before the final candidate panel is opened.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.leaf_reasoning_final_oos import (  # noqa: E402
    FinalOOSReplayContract,
    run_leaf_reasoning_final_oos_replay,
)


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("--input-panel must be parquet or csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-contract", required=True, type=Path, help="sealed development-selected final-OOS contract JSON")
    parser.add_argument("--input-panel", required=True, type=Path, help="November candidate panel with causal feature/state availability timestamps and realized H12 labels")
    parser.add_argument("--output-dir", required=True, type=Path, help="new immutable replay artifact directory")
    parser.add_argument("--consumption-registry", required=True, type=Path, help="new exclusive registry path; prevents a second use of this final OOS contract")
    parser.add_argument("--min-feature-coverage", type=float, default=0.99)
    parser.add_argument("--top-fraction", type=float, action="append", help="repeatable global top-k fraction; default .01,.05,.10")
    args = parser.parse_args()
    contract = FinalOOSReplayContract.from_json_path(args.frozen_contract)
    result = run_leaf_reasoning_final_oos_replay(
        contract, _read_panel(args.input_panel), output_dir=args.output_dir,
        consumption_registry=args.consumption_registry,
        min_feature_coverage=args.min_feature_coverage,
        top_fractions=tuple(args.top_fraction) if args.top_fraction else (0.01, 0.05, 0.10),
    )
    print(result.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
