#!/usr/bin/env python3
"""Report a causal global top-10% MLP-rank comparator for policy ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ablate_meta_recent_ev_target_mapping import _metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-start", default="2026-04-01")
    args = parser.parse_args()
    rows = pd.read_parquet(
        args.rank_history,
        columns=["__ts__", "rank_mlp_direct", "ev_after_1pct"],
    )
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True)
    rows = rows.dropna().sort_values("__ts__", kind="stable").reset_index(drop=True)
    rows["__day__"] = rows["__ts__"].dt.floor("D")
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    records: list[pd.DataFrame] = []
    for day in sorted(rows.loc[rows["__day__"].ge(eval_start), "__day__"].unique()):
        current = rows.loc[rows["__day__"].eq(day)].copy()
        prior = rows.loc[rows["__day__"].lt(day), "rank_mlp_direct"]
        if len(prior) < 200:
            continue
        cutoff = float(np.quantile(prior.to_numpy(dtype=np.float64), 0.90))
        current["selected"] = current["rank_mlp_direct"] >= cutoff
        records.append(current)
    scored = pd.concat(records, ignore_index=True)
    scored["month"] = scored["__ts__"].dt.strftime("%Y-%m")
    report: list[dict[str, object]] = []
    for month, source in scored.groupby("month", observed=True, sort=True):
        report.append(
            {"name": "causal_global_mlp_top10", "month": month,
             **_metrics(source.loc[source["selected"]], source)}
        )
    report.append(
        {"name": "causal_global_mlp_top10", "month": "overall",
         **_metrics(scored.loc[scored["selected"]], scored)}
    )
    pd.DataFrame(report).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
