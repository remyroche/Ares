#!/usr/bin/env python3
"""Score frozen forward failure detectors on fully observable daily state."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    FrozenFailureDetector,
)


def run(args: argparse.Namespace) -> pd.DataFrame:
    state = pd.read_parquet(args.state)
    state["day"] = pd.to_datetime(state["day"], utc=True).dt.floor("D")
    bundles: dict[str, FrozenFailureDetector] = joblib.load(args.bundle)
    parts = []
    for (side, archetype), rows in state.groupby(
        ["side_name", "archetype_policy_key"], observed=True, sort=True
    ):
        bundle = bundles.get(f"{side}::{archetype}")
        if bundle is None:
            continue
        part = bundle.score(rows)
        part["bundle_key"] = f"{side}::{archetype}"
        part["score_source"] = "frozen_final_forward_bundle"
        parts.append(part)
    output = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(
        {
            "rows": int(len(output)),
            "complete_rows": int(output.get("state_complete", pd.Series(dtype=bool)).sum()),
            "alerts": int(output.get("alert", pd.Series(dtype="boolean")).fillna(False).sum()),
            "output": str(Path(args.output).resolve()),
        },
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
