#!/usr/bin/env python3
"""Materialise side/month tail metrics for the additional frozen funnel arms."""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from extreme_price_movements.funnel_selection import global_tail_metrics, monthly_stability


def _metrics(x: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in x.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        base = dict(zip(keys, values))
        rows.append({**base, "scope": "global", **global_tail_metrics(group), **monthly_stability(group)})
        for side, side_group in group.groupby("side_name", sort=False):
            rows.append({**base, "scope": "side", "side": side, **global_tail_metrics(side_group), **monthly_stability(side_group)})
        tmp = group.assign(_month=pd.to_datetime(group["__ts__"], utc=True).dt.strftime("%Y-%m"))
        for month, month_group in tmp.groupby("_month", sort=True):
            rows.append({**base, "scope": "month", "month": month, **global_tail_metrics(month_group), **monthly_stability(month_group)})
    return pd.DataFrame(rows)


def main() -> None:
    # Longer residual query arm.
    p = ROOT / "data_perp/artifacts/frozen_longer_meta_query_ablation_20260810_v1/predictions.parquet"
    _metrics(pd.read_parquet(p), ["query"]).to_parquet(p.parent / "side_month_metrics.parquet", index=False)

    # Larger specialists with regime-based query grouping.
    p = ROOT / "data_perp/artifacts/regime_grouped_larger_specialists_20260810_v1/predictions.parquet"
    _metrics(pd.read_parquet(p), ["regime_query", "level"]).to_parquet(p.parent / "side_month_metrics.parquet", index=False)

    # Incremental CMI, retaining the feature-added step as the arm identifier.
    p = ROOT / "data_perp/artifacts/meta_incremental_cmi_20260810_v1/predictions.parquet"
    # The feature name is fold-local; step is the comparable arm identifier.
    _metrics(pd.read_parquet(p), ["step"]).to_parquet(p.parent / "side_month_metrics.parquet", index=False)

    # Non-residual EV-mapped combinations are reconstructed from the two mapped scores.
    p = ROOT / "data_perp/artifacts/nonresidual_ev_combination_20260810_v1/mapped_predictions.parquet"
    pred = pd.read_parquet(p)
    weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    # Side metrics are sufficient here; pooled monthly/worst-period metrics are
    # already in combination_metrics.parquet. Avoid constructing a 14m-row
    # Cartesian frame just to repeat those pooled calculations.
    rows = []
    for wb in weights:
        for wm in weights:
            score = wb * pred.base_ev.to_numpy(float) + wm * pred.meta_ev.to_numpy(float)
            for side, idx in pred.groupby("side_name", sort=False).groups.items():
                ix = np.asarray(idx)
                order = ix[np.argsort(-score[ix], kind="stable")]
                rows.append({
                    "base_weight": wb, "meta_weight": wm, "scope": "side", "side": side,
                    "top1_net_bps": float(pred.net_bps.iloc[order[:max(1, int(np.ceil(len(order)*.01)))]].to_numpy(float).mean()),
                    "top5_net_bps": float(pred.net_bps.iloc[order[:max(1, int(np.ceil(len(order)*.05)))]].to_numpy(float).mean()),
                    "top10_net_bps": float(pred.net_bps.iloc[order[:max(1, int(np.ceil(len(order)*.10)))]].to_numpy(float).mean()),
                })
    pd.DataFrame(rows).to_parquet(p.parent / "side_metrics.parquet", index=False)


if __name__ == "__main__":
    main()
