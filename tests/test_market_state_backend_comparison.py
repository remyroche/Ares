from pathlib import Path

import pandas as pd

from scripts import compare_market_state_controller_walkforward_backends as compare


def test_backend_aggregate_metrics_backfills_safety_columns(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "arm": ["S0_baseline_static_thresholds", "S1_observed_axes_shared_response"],
            "folds": [2, 2],
            "median_delta_net_pnl": [0.0, 10.0],
            "mean_delta_net_pnl": [0.0, 12.0],
            "q25_delta_net_pnl": [0.0, 4.0],
            "positive_delta_share": [0.0, 1.0],
            "median_delta_max_drawdown": [0.0, 0.01],
            "median_delta_worst_24h": [0.0, 2.0],
            "median_trade_count": [10.0, 9.0],
        }
    ).to_csv(tmp_path / "walkforward_aggregate_delta.csv", index=False)
    pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "arm": [
                "S0_baseline_static_thresholds",
                "S1_observed_axes_shared_response",
                "S0_baseline_static_thresholds",
                "S1_observed_axes_shared_response",
            ],
            "trade_count": [10, 9, 20, 18],
            "full_sl_rate": [0.30, 0.25, 0.20, 0.15],
        }
    ).to_csv(tmp_path / "walkforward_summary.csv", index=False)

    out = compare._aggregate_metrics(tmp_path, "lgbm")

    row = out.loc[out["arm"].eq("S1_observed_axes_shared_response")].iloc[0]
    assert row["backend"] == "lgbm"
    assert abs(float(row["median_trade_retention_share"]) - 0.9) < 1e-12
    assert abs(float(row["median_delta_full_sl_rate"]) + 0.05) < 1e-12
