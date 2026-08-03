#!/usr/bin/env python3
"""Freeze four completed Stage-I selector cells; never train or materialise."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_winner_bundle import freeze_stage_i_winner_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--input-contract-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--run-id", default="stage_i_production_oos_2024_2026")
    args = parser.parse_args()
    bundle, status = freeze_stage_i_winner_bundle(
        base_selection_dir=args.base_selection_dir,
        meta_selection_dir=args.meta_selection_dir,
        input_contract_dir=args.input_contract_dir,
        output_path=args.output,
        code_revision=args.code_revision,
        run_id=args.run_id,
    )
    print(json.dumps({
        "status": status,
        "output": str(args.output.resolve()),
        "winner_bundle_sha256": bundle.sha256,
        "cells": [cell.contract.artifact_key for cell in bundle.cells],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
