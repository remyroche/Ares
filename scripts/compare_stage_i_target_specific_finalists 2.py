#!/usr/bin/env python3
"""Compare frozen R3/S/O finalist joint-stack OOS ledgers on identical rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_target_specific_oos import (
    compare_target_specific_finalists,
    load_target_specific_finalist_artifact,
)


def _parse_item(value: str) -> tuple[str, Path]:
    name, separator, path = str(value).partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--finalist must be NAME=/immutable/oos/artifact")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalist", action="append", type=_parse_item, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.finalist) < 2:
        parser.error("at least two --finalist values are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite finalist comparison: {args.output_dir}")
    finalists = []
    for name, root in args.finalist:
        finalists.append(load_target_specific_finalist_artifact(root, name=name))
    score, attribution = compare_target_specific_finalists(finalists)
    args.output_dir.mkdir(parents=True)
    score.to_parquet(args.output_dir / "joint_stack_promotion_score.parquet", index=False, compression="zstd")
    attribution.to_parquet(args.output_dir / "joint_stack_promotion_attribution.parquet", index=False, compression="zstd")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "comparison": "joint reconstructed meta stack only",
        "finalists": [name for name, _ in args.finalist],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
