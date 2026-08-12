#!/usr/bin/env python3
"""Materialise the immutable Stage-II path/context ledger after Stage-I freeze.

The source map must declare the three canonical date-routed path substrates.
This command does not train a model, derive a label from a score, or fill a
missing source row.  A partial materialisation can only be resumed with the
identical frozen request.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extreme_price_movements.stage_ii_enriched_materializer import (
    materialize_stage_ii_enriched_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-oos-dir", type=Path, required=True)
    parser.add_argument("--stage-i-selected-panel-dir", type=Path, required=True)
    parser.add_argument("--candidate-spec-json", type=Path, required=True)
    parser.add_argument("--path-source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = materialize_stage_ii_enriched_ledger(
        stage_i_oos_dir=args.stage_i_oos_dir,
        selected_panel=args.stage_i_selected_panel_dir,
        candidate_spec=args.candidate_spec_json,
        source_map=args.path_source_manifest,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
