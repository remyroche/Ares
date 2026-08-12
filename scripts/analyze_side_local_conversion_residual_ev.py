#!/usr/bin/env python3
"""Materialise true globally-ranked metrics for the conversion/residual ablation.

The training runner also emits per-side diagnostic rows.  This helper is the
authoritative evaluator for the requested global top-k comparison: each tail is
ranked over the complete candidate population, then side/month/fold slices are
reported without duplicating observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LAMBDA_COLS = {
    "base_only": "score_lambda_000",
    "lambda_025": "score_lambda_025",
    "lambda_050": "score_lambda_050",
    "lambda_075": "score_lambda_075",
    "lambda_100": "score_lambda_100",
    "selected_oof": "score_selected",
}
TAILS = (0.01, 0.05, 0.10)


def _tail_rows(frame: pd.DataFrame, score_col: str, scope: str, period: str, system: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    ordered = frame.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable")
    rows: list[dict[str, object]] = []
    for tail in TAILS:
        n = max(1, int(np.ceil(len(ordered) * tail)))
        top = ordered.head(n)
        rows.append(
            {
                "system": system,
                "scope": scope,
                "period": period,
                "tail": tail,
                "rows": len(frame),
                "trades": n,
                "gross_bps": float(top.gross_bps.mean()),
                "net_bps": float(top.net_bps.mean()),
                "rank_ic_net": float(frame[score_col].corr(frame.net_bps, method="spearman")),
            }
        )
    return rows


def build_metrics(artifact: Path) -> pd.DataFrame:
    data = pd.read_parquet(artifact / "predictions.parquet").copy()
    data["month"] = pd.to_datetime(data["__ts__"], utc=True).dt.strftime("%Y-%m")
    selected = pd.read_parquet(artifact / "lambda_selection.parquet")
    chosen = dict(zip(zip(selected.fold, selected.side), selected.selected_lambda))
    data["selected_lambda"] = [chosen.get((fold, side), 0.0) for fold, side in zip(data.fold, data.side_name)]
    data["score_selected"] = data["base_ev_bps"] + data["selected_lambda"] * data["meta_residual_ev_bps"].fillna(0.0)

    rows: list[dict[str, object]] = []
    for system, score_col in LAMBDA_COLS.items():
        rows.extend(_tail_rows(data, score_col, "global", "all", system))
        for side, side_frame in data.groupby("side_name", sort=True):
            rows.extend(_tail_rows(side_frame, score_col, f"side:{side}", "all", system))
        for fold, fold_frame in data.groupby("fold", sort=True):
            rows.extend(_tail_rows(fold_frame, score_col, "global", str(fold), system))
        for month, month_frame in data.groupby("month", sort=True):
            rows.extend(_tail_rows(month_frame, score_col, "global", str(month), system))
    out = pd.DataFrame(rows)
    out.to_parquet(artifact / "global_metrics.parquet", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(build_metrics(args.artifact).to_string(index=False))


if __name__ == "__main__":
    main()
