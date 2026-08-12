#!/usr/bin/env python3
"""Materialise the immutable, six-field strict predecessor-meta OOF ledger.

The input must be an already-completed immutable base-to-meta ledger.  The
source map is deliberately explicit and maps each mandated predecessor
semantic to one causal scalar already present in that ledger; this command
does not perform feature selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_predecessor_meta_oof import (  # noqa: E402
    StrictPredecessorMetaOOFConfig,
    StrictPredecessorMetaOOFError,
    load_immutable_meta_ledger_for_predecessor,
    materialize_strict_predecessor_meta_oof,
    write_immutable_strict_predecessor_meta_oof,
)


def _source_map(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid --source-feature-map JSON: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise ValueError("--source-feature-map must be a JSON object of string semantic-to-column pairs")
    return {str(key): str(item) for key, item in value.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-root", type=Path, required=True,
        help="completed immutable base-to-meta ledger directory",
    )
    parser.add_argument(
        "--source-feature-map", type=Path, required=True,
        help="JSON map for exactly upgrade/downgrade portability, unstable upgrade share, covariance-break share, support score and reasoning entropy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--refit-interval-hours", type=int, default=24)
    args = parser.parse_args()
    try:
        ledger, _, source_hash = load_immutable_meta_ledger_for_predecessor(args.ledger_root)
        config = StrictPredecessorMetaOOFConfig(
            source_feature_map=_source_map(args.source_feature_map),
            min_train_rows=int(args.min_train_rows),
            ridge_alpha=float(args.ridge_alpha),
            refit_interval_hours=int(args.refit_interval_hours),
        )
        result = materialize_strict_predecessor_meta_oof(
            ledger, config=config, source_ledger_sha256=source_hash,
        )
        print(write_immutable_strict_predecessor_meta_oof(result, args.output_dir))
    except (StrictPredecessorMetaOOFError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
