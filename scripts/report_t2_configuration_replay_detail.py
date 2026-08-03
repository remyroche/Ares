#!/usr/bin/env python3
"""Detailed attribution for the archived T2 configuration replay.

This is reporting only.  It explicitly labels the result as previously opened
and does not convert it into an independent OOS claim.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _book(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    ordered = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="mergesort")
    return ordered.head(int(len(ordered) * .10 + .999999)).copy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cols = ["candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", "side_name", "__symbol__", "oof_fold", "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"]
    ledger = pd.read_parquet(a.ledger, columns=cols)
    pred = pd.read_parquet(a.predictions)
    x = ledger.merge(pred, on="candidate_id", how="inner", validate="one_to_one")
    x = x.loc[x.oof_fold.eq("meta_oos")].copy()
    for c in ("__ts__", "__decision_ts__", "__label_available_at__"):
        x[c] = pd.to_datetime(x[c], utc=True, errors="raise")
    x["week"] = x["__ts__"].dt.to_period("W-SUN").astype(str)
    x["month"] = x["__ts__"].dt.to_period("M").astype(str)
    x["hour_utc"] = x["__ts__"].dt.hour
    book = _book(x, "final_score_bps")
    weekly = book.groupby(["week", "side_name"], observed=True).agg(
        trades=("candidate_id", "size"), gross_bps_per_trade=("execution_gross_ev_12h", lambda v: v.mean() * 1e4), cost_bps_per_trade=("execution_cost_return", lambda v: v.mean() * 1e4), net_bps_per_trade=("execution_net_ev_12h", lambda v: v.mean() * 1e4), positive_net_rate=("execution_net_ev_12h", lambda v: (v > 0).mean()), assets=("__symbol__", "nunique"),
    ).reset_index()
    monthly = book.groupby(["month", "side_name"], observed=True).agg(
        trades=("candidate_id", "size"), gross_bps_per_trade=("execution_gross_ev_12h", lambda v: v.mean() * 1e4), cost_bps_per_trade=("execution_cost_return", lambda v: v.mean() * 1e4), net_bps_per_trade=("execution_net_ev_12h", lambda v: v.mean() * 1e4), positive_net_rate=("execution_net_ev_12h", lambda v: (v > 0).mean()),
    ).reset_index()
    hours = book.groupby(["hour_utc", "side_name"], observed=True).agg(trades=("candidate_id", "size"), net_bps_per_trade=("execution_net_ev_12h", lambda v: v.mean() * 1e4)).reset_index()
    shares = book.groupby("week", observed=True).size() / len(book)
    side = book.groupby("side_name", observed=True).agg(trades=("candidate_id", "size"), gross_bps_per_trade=("execution_gross_ev_12h", lambda v: v.mean() * 1e4), cost_bps_per_trade=("execution_cost_return", lambda v: v.mean() * 1e4), net_bps_per_trade=("execution_net_ev_12h", lambda v: v.mean() * 1e4), positive_net_rate=("execution_net_ev_12h", lambda v: (v > 0).mean())).reset_index()
    stage = Path(tempfile.mkdtemp(prefix=f".{a.output.name}.", dir=a.output.parent))
    try:
        book.to_parquet(stage / "global_top10_selected_trades.parquet", index=False)
        weekly.to_parquet(stage / "global_top10_weekly_side_metrics.parquet", index=False)
        monthly.to_parquet(stage / "global_top10_monthly_side_metrics.parquet", index=False)
        hours.to_parquet(stage / "global_top10_hour_utc_metrics.parquet", index=False)
        side.to_parquet(stage / "global_top10_side_metrics.parquet", index=False)
        summary = {"status": "ARCHIVED_PREVIOUSLY_OPENED_CONFIGURATION_REPLAY_NOT_FRESH_OOS", "candidate_rows": len(x), "global_top10_trades": len(book), "gross_bps_per_trade": float(book.execution_gross_ev_12h.mean() * 1e4), "cost_bps_per_trade": float(book.execution_cost_return.mean() * 1e4), "net_bps_per_trade": float(book.execution_net_ev_12h.mean() * 1e4), "week_hhi": float((shares**2).sum()), "largest_week_share": float(shares.max()), "top_three_weeks_share": float(shares.nlargest(3).sum()), "entry_timing": "feature cutoff / completed hourly bar close + 1h", "label_timing": "entry + 12h", "known_boundary_issue": "the archived model fit used source-time boundaries rather than a strict label-resolution purge; do not use this report as promotion evidence"}
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(stage, a.output)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
