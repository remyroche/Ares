#!/usr/bin/env python3
"""Publish the Stage-I target-information versus selector-sparsity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.stage_i_feature_selection import (
    resolve_stage_i_feature_universe,
)
from extreme_price_movements.stage_i_target_selector_audit import (
    publish_stage_i_target_selector_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    feature_columns = pq.ParquetFile(
        args.selector_dir / "selector_features.parquet"
    ).schema_arrow.names
    universes = {
        side: resolve_stage_i_feature_universe(
            CFG,
            layer="base",
            side=side,
            head="R3_economic_simplex_b25",
            available_columns=feature_columns,
        )
        for side in ("long", "short")
    }
    manifest = publish_stage_i_target_selector_audit(
        args.selector_dir, args.output_dir, side_feature_universes=universes
    )
    print(json.dumps({"status": "complete", "summary": manifest["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
