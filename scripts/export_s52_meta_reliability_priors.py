#!/usr/bin/env python3
"""Export frozen S52 meta reliability priors for live inference.

The meta model selected ``rel_rankband_*`` and ``rel_marginband_*`` features.
Those are train-derived priors by side/source-tag/base-score band.  Live rows
must receive frozen priors from the training fold, not same-row outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.inference.live_meta_feature_overlays import (
    reliability_prior_payload_from_training_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--train-month",
        action="append",
        default=[],
        help="Training month to include, e.g. 2026-05. May be repeated.",
    )
    parser.add_argument("--selected-col", default="selected_top30")
    parser.add_argument("--shrinkage-k", type=float, default=60.0)
    parser.add_argument(
        "--train-end-exclusive",
        default="",
        help="Optional UTC cutoff applied to __ts__ before fitting priors.",
    )
    parser.add_argument(
        "--exact-groups-only",
        action="store_true",
        help=(
            "Match the historical fold helper: unseen side/archetype/band "
            "groups fall back to global statistics, not side/band groups."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.scored_ledger.exists():
        raise FileNotFoundError(args.scored_ledger)
    cols = None
    frame = pd.read_parquet(args.scored_ledger, columns=cols)
    months = [str(m) for m in (args.train_month or []) if str(m).strip()]
    if months:
        if "month" not in frame.columns:
            raise ValueError("Scored ledger has no month column; cannot apply train-month filter")
        frame = frame[frame["month"].astype(str).isin(months)].copy()
    if str(args.train_end_exclusive).strip():
        if "__ts__" not in frame.columns:
            raise ValueError("Scored ledger has no __ts__ column")
        cutoff = pd.Timestamp(args.train_end_exclusive)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        timestamps = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        frame = frame.loc[timestamps.lt(cutoff)].copy()
    if frame.empty:
        raise ValueError("No rows remain after train-month filtering")
    payload = reliability_prior_payload_from_training_frame(
        frame,
        selected_col=str(args.selected_col),
        shrinkage_k=float(args.shrinkage_k),
    )
    payload["exact_groups_only"] = bool(args.exact_groups_only)
    payload["source"] = {
        "scored_ledger": str(args.scored_ledger),
        "train_months": months,
        "train_end_exclusive": str(args.train_end_exclusive or ""),
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": payload.get("rows"),
                "groups": len(payload.get("groups", {})),
                "train_months": months,
                "feature_names": payload.get("feature_names", []),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
