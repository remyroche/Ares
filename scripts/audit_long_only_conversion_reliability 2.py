#!/usr/bin/env python3
"""Long-only audit for the historical-support conversion learner."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_perp/artifacts/frozen_conversion_reliability_learner_ablation_20260810_v2/predictions.parquet"
OUT = ROOT / "data_perp/artifacts/frozen_conversion_reliability_learner_ablation_20260810_v2/long_only"
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
ARMS = {
    "raw_control": "score",
    "regression_a0.25": "score_regression_a0.25",
    "regression_a0.50": "score_regression_a0.5",
    "regression_a1.00": "score_regression_a1",
    "ordinal_a0.25": "score_ordinal_a0.25",
    "ordinal_a0.50": "score_ordinal_a0.5",
    "ordinal_a1.00": "score_ordinal_a1",
}


def tail_row(frame: pd.DataFrame, arm: str, score: str, tail: float, period: str, selection: str) -> dict[str, object]:
    n = max(1, int(np.ceil(len(frame) * tail)))
    chosen = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {
        "arm": arm, "period": period, "selection": selection, "tail": tail,
        "population_rows": int(len(frame)), "selected_rows": int(len(chosen)),
        "gross_bps": float(chosen.gross_bps.mean()), "net_bps": float(chosen.net_bps.mean()),
        "rank_ic": float(frame[score].rank().corr(frame.net_bps.rank())),
    }


def run() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(INPUT)
    frame = frame.loc[frame.fold.astype(str).str.startswith("transport") & frame.side_name.eq("long")].copy()
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    rows: list[dict[str, object]] = []
    for arm, score in ARMS.items():
        for tail in TAILS:
            rows.append(tail_row(frame, arm, score, tail, "all_transport", "pooled_global_long_only"))
        dev = frame.loc[frame.month.lt("2024-11")]
        nov = frame.loc[frame.month.eq("2024-11")]
        rows.append(tail_row(dev, arm, score, .05, "jul_oct", "selection_dev"))
        rows.append(tail_row(nov, arm, score, .05, "november", "untouched_oos"))
        for month, group in frame.groupby("month", sort=True):
            rows.append(tail_row(group, arm, score, .05, month, "monthly_diagnostic"))
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(OUT / "long_only_metrics.parquet", index=False, compression="zstd")
    selection = metrics[(metrics["selection"].eq("selection_dev")) & (metrics["tail"].eq(.05))].sort_values("net_bps", ascending=False).iloc[0]
    manifest = {
        "schema": "long_only_conversion_reliability_audit_v1",
        "input": str(INPUT), "side": "long_only", "short_rows_used": 0,
        "transport_rows": int(len(frame)), "months": sorted(frame.month.unique().tolist()),
        "selection": "highest July-October long-only top-5 net; diagnostics report monthly and untouched November",
        "selected_dev_arm": str(selection.arm), "selected_dev_top5_net_bps": float(selection.net_bps),
        "arms": ARMS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return OUT


if __name__ == "__main__":
    print(run())
