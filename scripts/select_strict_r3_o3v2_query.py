#!/usr/bin/env python3
"""Freeze one O3-v2 LambdaRank query geometry from a development screen.

The paired query screen persists held scores before it joins policy outcomes.
This selector consumes only its declared development metrics and records a
single deterministic winner.  A later target/support screen must receive the
frozen query name; it must never choose from the subsequent forward block.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "strict_r3_o3v2_query_selection_v1"


def _months(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("at least one YYYY-MM development month is required")
    return values


def _exclusive_json(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run(*, metrics_path: Path, out: Path, development_months: tuple[str, ...]) -> None:
    if out.exists():
        raise FileExistsError(out)
    metrics = pd.read_parquet(metrics_path)
    required = {"query_mode", "month", "utility", "rank_ic", "top1", "top2", "top5"}
    if missing := required - set(metrics.columns):
        raise AssertionError(f"query metrics missing required fields: {sorted(missing)}")
    local = metrics.loc[metrics["month"].isin(development_months)].copy()
    expected = len(development_months)
    counts = local.groupby("query_mode", sort=True)["month"].nunique()
    if counts.empty or not (counts == expected).all():
        raise AssertionError("every declared query mode must cover every development month")
    rows: list[dict[str, object]] = []
    for mode, part in local.groupby("query_mode", sort=True):
        utility = pd.to_numeric(part["utility"], errors="coerce").to_numpy(float)
        if not np.isfinite(utility).all():
            raise AssertionError(f"{mode}: non-finite development utility")
        rows.append({
            "query_mode": str(mode),
            "development_months": ",".join(development_months),
            "utility_mean": float(utility.mean()),
            "utility_std": float(utility.std(ddof=0)),
            "utility_worst_month": float(utility.min()),
            "selection_score": float(utility.mean() - .25 * utility.std(ddof=0) - max(0.0, -utility.min())),
            "top1_mean": float(pd.to_numeric(part["top1"], errors="coerce").mean()),
            "top2_mean": float(pd.to_numeric(part["top2"], errors="coerce").mean()),
            "top5_mean": float(pd.to_numeric(part["top5"], errors="coerce").mean()),
            "rank_ic_mean": float(pd.to_numeric(part["rank_ic"], errors="coerce").mean()),
        })
    table = pd.DataFrame(rows).sort_values(
        ["selection_score", "utility_worst_month", "top1_mean", "query_mode"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    winner = table.iloc[0]
    out.mkdir(parents=True)
    table.to_parquet(out / "query_development_selection.parquet", index=False, compression="zstd")
    _exclusive_json(out / "selected_query_contract.json", {
        "schema": SCHEMA,
        "development_months": list(development_months),
        "selection": "mean utility minus 0.25 standard deviation, then worst month and top-1 mean as deterministic ties",
        "selected_query_mode": str(winner["query_mode"]),
        "selection_score": float(winner["selection_score"]),
        "held_out_rule": "later target/support and forward blocks are not read by this selector",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-months", default="2025-11,2025-12,2026-01")
    args = parser.parse_args()
    run(metrics_path=args.metrics, out=args.out, development_months=_months(args.development_months))


if __name__ == "__main__":
    main()
