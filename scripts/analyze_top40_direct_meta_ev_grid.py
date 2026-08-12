#!/usr/bin/env python3
"""Persist true pooled-global metrics for a direct-meta EV-grid replay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GRID_COLUMNS = {
    "base_only": "score_wb100_wm000",
    "ev_mix_0.75_0.25": "score_wb075_wm025",
    "ev_mix_0.50_0.50": "score_wb050_wm050",
    "ev_mix_0.25_0.75": "score_wb025_wm075",
    "meta_only": "score_wb000_wm100",
}
TAILS = (.01, .05, .10)


def _rows(frame: pd.DataFrame, col: str, system: str, fold: str, period: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for side, sub in [("pooled", frame), *[(s, frame[frame.side_name.eq(s)]) for s in ("long", "short")]]:
        if len(sub) == 0:
            continue
        for tail in TAILS:
            n = max(1, int(np.ceil(len(sub) * tail)))
            top = sub.sort_values([col, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            out.append({"scope": "global", "system": system, "fold": fold, "period": period, "side": side, "tail": tail, "rows": len(sub), "trades": n, "gross_bps": float(top.gross_bps.mean()), "net_bps": float(top.net_bps.mean()), "rank_ic": float(sub[col].rank().corr(sub.net_bps.rank()))})
    return out


def run(artifact: Path) -> Path:
    d = pd.read_parquet(artifact / "predictions.parquet")
    rows: list[dict[str, object]] = []
    for system, col in GRID_COLUMNS.items():
        for fold, sub in d.groupby("fold", sort=True):
            rows.extend(_rows(sub, col, system, fold, "fold"))
            sub = sub.copy(); sub["month"] = pd.to_datetime(sub["__ts__"], utc=True).dt.strftime("%Y-%m")
            for month, month_frame in sub.groupby("month", sort=True):
                rows.extend(_rows(month_frame, col, system, fold, month))
        rows.extend(_rows(d, col, system, "all_oos", "all_oos"))
    out = artifact / "global_metrics.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(run(args.artifact))
