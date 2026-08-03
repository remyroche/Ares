import numpy as np
import pandas as pd

from scripts.run_fixed_horizon_action_ablation import (
    bootstrap_rows,
    metric_rows,
)


def _frame():
    rows = []
    for day in range(2):
        for side in ("long", "short"):
            row = {
                "candidate_id": f"{day}-{side}",
                "candidate_month": "2025-03",
                "side_name": side,
                "execution_decision_utc": pd.Timestamp(
                    "2025-03-01", tz="UTC"
                )
                + pd.Timedelta(days=day),
                "weight_top_01": 1.0,
                "weight_top_05": 1.0,
                "weight_top_10": 1.0,
                "weight_top_20": 1.0,
                "gross__deployed": 0.01,
                "net__deployed": 0.0,
                "cost__deployed": 0.01,
                "positive__deployed": 0,
            }
            for hours in (1, 2, 4, 8, 12):
                row[f"gross__fixed_{hours}h"] = 0.01 + hours / 1000
                row[f"net__fixed_{hours}h"] = hours / 1000
                row[f"cost__fixed_{hours}h"] = 0.01
                row[f"positive__fixed_{hours}h"] = 1
            rows.append(row)
    return pd.DataFrame(rows)


def test_metrics_preserve_global_and_side_scopes():
    metrics = pd.DataFrame(metric_rows(_frame()))
    top10 = metrics.loc[metrics.top_fraction.eq(0.10)]
    assert set(top10.scope) == {"global", "side_long", "side_short"}
    fixed12 = top10.loc[
        top10.scope.eq("global") & top10.arm.eq("fixed_12h")
    ].iloc[0]
    assert np.isclose(fixed12.net_bps, 120.0)
    assert np.isclose(fixed12.paired_delta_vs_deployed_bps, 120.0)


def test_bootstrap_is_paired_and_deployed_delta_is_zero():
    result = pd.DataFrame(bootstrap_rows(_frame(), draws=100))
    deployed = result.loc[result.arm.eq("deployed")]
    assert np.allclose(deployed.paired_delta_bps, 0.0)
    assert np.allclose(deployed.paired_delta_ci_low_bps, 0.0)
    assert np.allclose(deployed.paired_delta_ci_high_bps, 0.0)
