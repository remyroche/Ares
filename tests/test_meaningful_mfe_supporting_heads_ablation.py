from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_meaningful_mfe_supporting_heads_ablation import (
    causal_incremental_stack,
    selection_replacement_economics,
)


def test_incremental_stack_is_june_to_july_and_pooled() -> None:
    rows = []
    for month, count in (("2026-06-01", 80), ("2026-07-01", 40)):
        start = pd.Timestamp(month, tz="UTC")
        for index in range(count):
            favorable = int(index % 3 == 0)
            rows.append(
                {
                    "__ts__": start + pd.Timedelta(hours=index),
                    "__symbol__": f"S{index % 4}",
                    "side_name": "long" if index % 2 else "short",
                    "candidate_id": f"{month}-{index}",
                    "label_resolution_utc": start + pd.Timedelta(hours=index + 12),
                    "favorable_first": favorable,
                    "execution_net_ev_12h": 0.01 if favorable else -0.01,
                    "hard_probability": 0.7 if favorable else 0.3,
                    "soft_probability": 0.65 if favorable else 0.35,
                    "competing_favorable_probability": 0.6 if favorable else 0.4,
                    "pred_early_path_quality": 0.8 if favorable else 0.2,
                    "pred_economic_barrier_time_quality": 0.7 if favorable else 0.3,
                    "pred_slope_quality": 0.75 if favorable else 0.25,
                }
            )
    predictions, metrics = causal_incremental_stack(pd.DataFrame(rows), seed=42)
    assert predictions["__ts__"].min() >= pd.Timestamp("2026-07-01T00:00:00Z")
    assert predictions["arm"].nunique() == 5
    assert set(metrics.loc[metrics["scope"].eq("pooled_global"), "top10_rows"]) == {4}
    assert set(metrics["scope"]) == {"pooled_global", "side_long", "side_short"}
    assert np.isfinite(metrics["top10_mean_net_ev_bps"]).all()
    replacement = selection_replacement_economics(predictions)
    baseline = replacement.loc[replacement["arm"].eq("event_only")].iloc[0]
    assert baseline["overlap_rows"] == 4
    assert baseline["incremental_net_ev_bps"] == 0.0
