#!/usr/bin/env python3
"""Attach point-in-time residual market context to frozen V11 prediction ledgers.

The source context must already have been generated causally.  This utility
only joins it by the complete prediction key and records coverage; it does not
recompute market features from outcomes or fill missing values from the future.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _materialize(
    predictions: Path,
    context: Path,
    output: Path,
    *,
    required_columns: list[str] | None = None,
    minimum_match_rate: float = 1.0,
) -> dict[str, object]:
    prediction_schema = set(pq.read_schema(predictions).names)
    context_schema = set(pq.read_schema(context).names)
    missing = sorted(set(KEYS).difference(context_schema))
    if missing:
        raise ValueError(f"Context is missing complete prediction keys: {missing}")
    base = pd.read_parquet(predictions)
    addition = pd.read_parquet(context)
    for frame in (base, addition):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    context_columns = [column for column in addition.columns if column not in KEYS]
    required = list(dict.fromkeys(required_columns or []))
    missing_required = sorted(set(required).difference(context_columns))
    if missing_required:
        raise ValueError(
            "Context does not provide the required parity columns: "
            f"{missing_required}"
        )
    collision = sorted(set(context_columns).intersection(prediction_schema))
    if collision:
        raise ValueError(f"Context would overwrite frozen prediction columns: {collision}")
    addition = addition.drop_duplicates(KEYS, keep="last")
    result = base.merge(addition, on=KEYS, how="left", validate="one_to_one")
    if len(result) != len(base):
        raise AssertionError("Context join changed the prediction row count")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    matched = result[context_columns].notna().any(axis=1) if context_columns else pd.Series(False, index=result.index)
    required_matched = (
        result[required].notna().all(axis=1)
        if required
        else matched
    )
    match_rate = float(required_matched.mean())
    if match_rate < minimum_match_rate:
        raise ValueError(
            "Context parity coverage is below the required floor: "
            f"{match_rate:.4%} < {minimum_match_rate:.4%}"
        )
    return {
        "predictions": str(predictions),
        "context": str(context),
        "output": str(output),
        "rows": int(len(result)),
        "context_columns": context_columns,
        "matched_rows": int(matched.sum()),
        "match_rate": float(matched.mean()),
        "required_columns": required,
        "required_matched_rows": int(required_matched.sum()),
        "required_match_rate": match_rate,
        "minimum_match_rate": float(minimum_match_rate),
        "timestamp_min": str(result["__ts__"].min()),
        "timestamp_max": str(result["__ts__"].max()),
        "leakage_contract": (
            "The context source must be point-in-time materialized before this join. "
            "This script rejects duplicate keys and source columns that overwrite frozen predictions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--train-context", type=Path, required=True)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--oos-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--required-columns",
        default="",
        help="Comma-separated causal context columns required on both ledgers.",
    )
    parser.add_argument(
        "--minimum-match-rate",
        type=float,
        default=1.0,
        help="Required complete-context coverage on each ledger.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.minimum_match_rate <= 1.0:
        raise ValueError("--minimum-match-rate must be in [0, 1]")
    required_columns = [
        token.strip() for token in args.required_columns.split(",") if token.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "v11_residual_context_materialization_v1",
        "train": _materialize(
            args.train_predictions,
            args.train_context,
            args.output_dir / "train_oof_with_context.parquet",
            required_columns=required_columns,
            minimum_match_rate=args.minimum_match_rate,
        ),
        "oos": _materialize(
            args.oos_predictions,
            args.oos_context,
            args.output_dir / "oos_with_context.parquet",
            required_columns=required_columns,
            minimum_match_rate=args.minimum_match_rate,
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
