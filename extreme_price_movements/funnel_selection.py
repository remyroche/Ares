"""Deterministic selection rules for the frozen specialist/residual funnel."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def global_tail_metrics(frame: pd.DataFrame, score_column: str = "score", tails: Sequence[float] = (.01, .05, .10)) -> dict[str, float]:
    """Compute globally ranked gross/net tails across all sides and folds."""
    required = {score_column, "candidate_id", "net_bps", "gross_bps"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"global-tail frame missing {sorted(missing)}")
    x = frame.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable")
    out: dict[str, float] = {}
    for tail in tails:
        n = max(1, int(np.ceil(len(x) * float(tail))))
        top = x.head(n)
        out[f"top{int(round(100 * tail))}_net_bps"] = float(top.net_bps.mean())
        out[f"top{int(round(100 * tail))}_gross_bps"] = float(top.gross_bps.mean())
    return out


def monthly_stability(frame: pd.DataFrame, score_column: str = "score", tail: float = .05) -> dict[str, float]:
    """Summarise month-level global tail stability."""
    if "__ts__" not in frame:
        raise KeyError("monthly stability requires __ts__")
    x = frame.copy()
    x["_month"] = pd.to_datetime(x["__ts__"], utc=True).dt.strftime("%Y-%m")
    values: list[float] = []
    monthly: dict[str, float] = {}
    for month, group in x.groupby("_month", sort=True, observed=True):
        n = max(1, int(np.ceil(len(group) * float(tail))))
        value = float(group.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable").head(n).net_bps.mean())
        monthly[str(month)] = value
        values.append(value)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"month_count": 0, "month_mean_net_bps": np.nan, "month_std_net_bps": np.nan, "month_worst_net_bps": np.nan, "month_mad_net_bps": np.nan}
    median = float(np.median(arr))
    return {
        "month_count": int(len(arr)),
        "month_mean_net_bps": float(arr.mean()),
        "month_std_net_bps": float(arr.std(ddof=0)),
        "month_worst_net_bps": float(arr.min()),
        "month_mad_net_bps": float(np.median(np.abs(arr - median))),
        **{f"month_{month}_top{int(round(100 * tail))}_net_bps": value for month, value in monthly.items()},
    }


def selection_key(metrics: Mapping[str, float], *, tie_tolerance_bps: float = 1.0) -> tuple[float, float, float]:
    """Return a sortable key: pooled EV, stability, then top-1 EV.

    Stability is represented by the negative monthly dispersion and worst-month
    penalty.  ``tie_tolerance_bps`` is applied by ``select_winner`` rather than
    hidden in this numeric key.
    """
    stability = -float(metrics.get("month_std_net_bps", np.inf)) + .25 * float(metrics.get("month_worst_net_bps", -np.inf))
    return (float(metrics.get("top5_net_bps", metrics.get("top5_net_bps_per_trade", -np.inf))), stability, float(metrics.get("top1_net_bps", -np.inf)))


def select_winner(table: pd.DataFrame, *, tie_tolerance_bps: float = 1.0) -> pd.Series:
    """Select one arm using pooled top-5 EV, monthly stability, then top-1 EV."""
    required = {"arm", "top5_net_bps", "month_std_net_bps", "month_worst_net_bps", "top1_net_bps"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"selection table missing {sorted(missing)}")
    x = table.copy()
    best = float(x.top5_net_bps.max())
    x = x[x.top5_net_bps.ge(best - float(tie_tolerance_bps))].copy()
    x["_stability_key"] = -x.month_std_net_bps + .25 * x.month_worst_net_bps
    stable = float(x._stability_key.max())
    x = x[x._stability_key.ge(stable - float(tie_tolerance_bps))].copy()
    return x.sort_values(["top1_net_bps", "arm"], ascending=[False, True], kind="stable").iloc[0]


__all__ = ["global_tail_metrics", "monthly_stability", "selection_key", "select_winner"]
