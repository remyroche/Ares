#!/usr/bin/env python3
"""Execute the frozen Stage-I selected-panel -> strict-OOF production path."""

import argparse
import json
from pathlib import Path
import sys


# Allow direct invocation from ``scripts/`` without relying on an editable
# package install.  This only changes import resolution; the CLI remains a
# non-materialising preflight.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extreme_price_movements.stage_i_production_execution import (
    execute_stage_i_production_oos,
)
from extreme_price_movements.stage_i_production_oos import (
    StageIProductionWinnerBundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-bundle", type=Path, required=True)
    parser.add_argument("--input-contract-dir", type=Path, required=True)
    parser.add_argument("--selected-panel-cache", type=Path, required=True)
    parser.add_argument("--strict-oof-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=4_000)
    parser.add_argument("--max-read-columns", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_rows < 1 or args.max_read_columns < 1:
        parser.error("batch/read-column limits must be positive")
    bundle = StageIProductionWinnerBundle.from_dict(json.loads(
        args.winner_bundle.read_text(encoding="utf-8")
    ))
    result = execute_stage_i_production_oos(
        bundle,
        input_contract_dir=args.input_contract_dir,
        selected_panel_dir=args.selected_panel_cache,
        strict_oof_cache_dir=args.strict_oof_cache,
        output_dir=args.output_dir,
        resume=args.resume,
        max_rows_per_batch=args.batch_rows,
        max_columns_per_read=args.max_read_columns,
    )
    print(json.dumps({
        "status": result.get("status"),
        "schema": result.get("schema"),
        "winner_bundle_sha256": result.get("winner_bundle_sha256"),
        "output_dir": str(args.output_dir.resolve()),
        "restart_status": result.get("restart_status", "created"),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
