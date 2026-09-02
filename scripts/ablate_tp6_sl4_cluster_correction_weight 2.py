#!/usr/bin/env python3
"""Causal post-fit weight ablation for canonical cluster corrections.

The path taxonomy and per-cluster residual models are frozen.  This script only
changes the declared correction multiplier in
``base_expected_bps + lambda * correction`` and reports global top-k plus
monthly stability, so it cannot tune on held-out labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def _metrics(frame: pd.DataFrame, score: str, period: str, tail: float, *, per_month: bool = False) -> dict[str, object]:
    block = frame if not per_month else frame.loc[frame.month_key.eq(period)]
    if block.empty:
        return {"arm": score, "period": period, "tail": tail, "trades": 0, "gross_bps_per_trade": np.nan, "net_bps_per_trade": np.nan, "rank_ic": np.nan}
    n = max(1, int(np.ceil(len(block) * tail)))
    top = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {
        "arm": score,
        "period": period,
        "tail": tail,
        "trades": int(n),
        "gross_bps_per_trade": float(top.gross_bps.mean()),
        "net_bps_per_trade": float(top.net_bps.mean()),
        "rank_ic": float(block[[score, "net_bps"]].corr(method="spearman").iloc[0, 1]),
    }


def run(source: Path, out: Path) -> Path:
    frame = pd.read_parquet(source).copy()
    frame["month_key"] = pd.to_datetime(frame["decision_ts"], utc=True).dt.strftime("%Y-%m")
    rows: list[dict[str, object]] = []
    weights = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
    for source_arm, label in (("cluster_only_score", "cluster_only"), ("cluster_context_score", "cluster_context")):
        correction = frame[source_arm].to_numpy(float) - frame.base_expected_bps.to_numpy(float)
        for weight in weights:
            arm = f"{label}_lambda_{weight:g}"
            frame[arm] = frame.base_expected_bps.to_numpy(float) + float(weight) * correction
            for tail in TAILS:
                rows.append(_metrics(frame, arm, "all", tail))
            for month in sorted(frame.month_key.unique()):
                rows.append(_metrics(frame, arm, month, 0.05, per_month=True))
    metrics = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out / "cluster_correction_weight_metrics.parquet", index=False, compression="zstd")
    monthly = metrics.loc[(metrics["period"] != "all") & metrics["tail"].eq(0.05)].copy()
    stability = monthly.groupby("arm", sort=True).net_bps_per_trade.agg(
        mean_top5_net_bps="mean", median_top5_net_bps="median", worst_month_top5_net_bps="min",
        positive_months_top5=lambda x: int((x > 0).sum()),
    ).reset_index()
    stability.to_parquet(out / "cluster_correction_weight_stability.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_cluster_correction_weight_ablation_v1",
        "source": str(source),
        "weights": list(weights),
        "target": "exact TP6/SL4 net bps; correction is frozen cluster prediction minus base anchor",
        "selection": "descriptive only; no new model fitting or held-label tuning",
        "global_ranking": True,
    }
    (out / "run_manifest.json").write_text(__import__("json").dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.source, args.out))
