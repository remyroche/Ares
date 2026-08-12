#!/usr/bin/env python3
"""Verify exact v4 LDF producer parity against a frozen HPO OOF fold.

This intentionally refits one historical LDF bundle from the source ledger,
not from a persisted research model.  A successful check proves that the
production trainer, target-free scorer, equal-month sampling, parent map,
CMI interactions, two forests, and activation gate reproduce the evaluated
OOF contract under the same pre-cutoff inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_n5_canonical import (  # noqa: E402
    load_n5_contract,
    score_canonical_n5_bundle,
    train_canonical_n5_bundle,
)
from scripts.run_strict_r3_n5_canonical_selection import _load_selection_input  # noqa: E402


CHECK_COLUMNS = (
    "n5_expected_bps",
    "n5_predictive_sd_bps",
    "n5_shrinkage_lambda",
    "n5_effective_support",
    "n5_p_ev_positive",
    "n5_p_adverse_200",
    "trust_size_multiplier",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--feature-sidecar", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--cutoff", default="2025-01-01T00:00:00Z")
    parser.add_argument("--held-end", default="2025-04-01T00:00:00Z")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable output already exists: {args.out}")

    frame, _fields, _audit = _load_selection_input(
        args.source,
        feature_sidecar=args.feature_sidecar,
        feature_contract=args.feature_contract,
    )
    contract = load_n5_contract()
    cutoff = pd.Timestamp(args.cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    held_end = pd.Timestamp(args.held_end)
    held_end = held_end.tz_localize("UTC") if held_end.tzinfo is None else held_end.tz_convert("UTC")
    bundle = train_canonical_n5_bundle(frame, cutoff=cutoff)
    held = frame.loc[
        frame["__decision_ts__"].ge(cutoff)
        & frame["__decision_ts__"].lt(held_end)
        & frame["mapped_ev_available"].fillna(False).astype(bool),
        ["candidate_id", "final_score", "raw_expected_bps", *contract["features"]],
    ].copy()
    replay = score_canonical_n5_bundle(bundle, held)
    replay = replay.rename(columns={"portfolio_size_multiplier": "trust_size_multiplier"})
    stored = pd.read_parquet(args.oof_predictions)
    stored = stored.loc[
        stored["candidate_id"].isin(set(held["candidate_id"])),
        ["candidate_id", *CHECK_COLUMNS],
    ].copy()
    joined = replay.loc[:, ["candidate_id", *CHECK_COLUMNS]].merge(
        stored,
        on="candidate_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_producer", "_stored"),
    )
    if len(joined) != len(held):
        raise AssertionError(
            f"parity intersection {len(joined)} does not cover all held mapped rows {len(held)}",
        )
    rows: list[dict[str, object]] = []
    for column in CHECK_COLUMNS:
        left = pd.to_numeric(joined[f"{column}_producer"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(joined[f"{column}_stored"], errors="coerce").to_numpy(float)
        difference = np.abs(left - right)
        maximum = float(np.nanmax(difference)) if len(difference) else 0.0
        rows.append(
            {
                "column": column,
                "rows": int(len(difference)),
                "max_abs_difference": maximum,
                "mean_abs_difference": float(np.nanmean(difference)) if len(difference) else 0.0,
                "passes": bool(maximum <= float(args.tolerance)),
            },
        )
    audit = pd.DataFrame(rows)
    if not audit["passes"].all():
        raise AssertionError(
            "v4 producer does not reproduce stored OOF output: "
            + audit.loc[~audit["passes"], ["column", "max_abs_difference"]].to_json(orient="records"),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(args.out.with_suffix(".parquet"), index=False)
    args.out.write_text(
        json.dumps(
            {
                "schema": "strict_r3_ldf_v4_replay_parity_v1",
                "status": "pass",
                "cutoff": cutoff.isoformat(),
                "held_end": held_end.isoformat(),
                "mapped_held_rows": int(len(held)),
                "tolerance": float(args.tolerance),
                "fields": contract["features"],
                "audit_parquet": args.out.with_suffix(".parquet").name,
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
