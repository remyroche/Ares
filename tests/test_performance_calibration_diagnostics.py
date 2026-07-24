import numpy as np
import pandas as pd

from extreme_price_movements.diagnostics.performance_calibration import (
    calibration_diagnostics,
    calibration_metrics,
    daily_performance_decomposition,
    meta_score_tail_diagnostics,
    monthly_performance_comparison,
)


def test_daily_decomposition_is_side_relative_and_reports_execution_metrics():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"),
            "side": ["long", "short", "long", "short"],
            "gross_return": [0.02, -0.01, -0.02, 0.01],
            "net_return": [0.01, -0.02, -0.03, 0.005],
            "bankroll_pnl": [10.0, 20.0, -30.0, -5.0],
            "mfe": [0.03, -0.02, 0.01, -0.01],
            "mae": [-0.01, 0.02, -0.04, 0.01],
            "holding_hours": [1.0, 2.0, 3.0, 4.0],
            "exit_reason": ["tp", "tp", "sl", "timeout"],
        }
    )

    daily = daily_performance_decomposition(trades, returns_are_side_relative=False)

    assert daily["date"].dt.tz is not None
    assert daily["net_return_sum"].tolist() == [0.01, 0.02, -0.03, -0.005]
    assert daily["bankroll_pnl_sum"].tolist() == [10.0, 20.0, -30.0, -5.0]
    assert daily.loc[0, "exit_tp_share"] == 1.0
    assert daily.loc[0, "win_rate"] == 1.0
    assert np.isfinite(daily.loc[0, "annualized_sharpe"])

    monthly = monthly_performance_comparison(trades, returns_are_side_relative=False)
    assert monthly.loc[0, "trade_count"] == 4
    assert monthly.loc[0, "month"] == "2026-01"


def test_calibration_supports_soft_targets_and_grouped_tables_without_apply():
    frame = pd.DataFrame(
        {
            "target": [0.0, 1.0, 0.5, 1.0, 0.0, np.nan],
            "score": [0.1, 0.9, 0.6, 0.8, 0.2, 0.5],
            "side": ["long", "long", "short", "short", "short", "long"],
        }
    )

    result = calibration_diagnostics(frame, group_cols="side", n_bins=5)
    overall = result["metrics"].query("scope == 'overall'").iloc[0]

    assert overall["support"] == 5
    assert overall["brier"] == np.mean((np.array([0.1, 0.9, 0.6, 0.8, 0.2]) - np.array([0.0, 1.0, 0.5, 1.0, 0.0])) ** 2)
    assert overall["top_1_precision"] == 1.0
    assert set(result["metrics"]["scope"]) == {"overall", "grouped"}
    assert len(result["reliability"].query("scope == 'grouped'")) == 10

    degenerate = calibration_metrics([1.0, 1.0], [0.8, 0.9])
    assert np.isnan(degenerate["auc"])


def test_meta_score_tails_are_disjoint_and_keep_economic_metrics():
    size = 100
    frame = pd.DataFrame(
        {
            "meta_score": np.linspace(1.0, 0.0, size, endpoint=False),
            "net_ev": np.r_[0.10, np.full(size - 1, -0.01)],
            "mfe": np.arange(size, dtype=float),
            "mae": -np.arange(size, dtype=float),
            "target": np.r_[1.0, np.zeros(size - 1)],
        }
    )

    tails = meta_score_tail_diagnostics(frame, target_col="target")

    assert tails["tail"].tolist() == ["top_1", "top_1_2", "top_2_5", "top_5_10"]
    assert tails["trade_count"].tolist() == [1, 1, 3, 5]
    assert tails["trade_count"].sum() == 10
    assert tails.loc[0, "ev_mean"] == 0.10
    assert tails.loc[0, "precision"] == 1.0


def test_missing_economics_and_empty_tails_are_safe():
    trades = pd.DataFrame({"timestamp": ["not-a-date"], "net_return": [np.nan]})
    daily = daily_performance_decomposition(trades)
    assert daily.empty

    tails = meta_score_tail_diagnostics(pd.DataFrame({"meta_score": [np.nan], "net_return": [0.1]}))
    assert tails["trade_count"].tolist() == [0, 0, 0, 0]
    assert tails["ev_mean"].isna().all()
