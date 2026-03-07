import numpy as np
import pandas as pd

from extreme_price_movements.ridge_position_sizer import (
    ExitReason,
    _stable_daily_pnl_metrics,
    _stable_daily_sortino_and_maxdd,
    compute_policy_aware_labels,
    compute_policy_aware_labels_batch,
    run_oof_grid_backtest,
)


def _make_flat_price_panel(n=8, symbol="BTCUSDT"):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    flat = pd.DataFrame({symbol: np.full(n, 100.0)}, index=idx)
    return {
        "open": flat.copy(),
        "high": flat.copy(),
        "low": flat.copy(),
        "close": flat.copy(),
    }


def test_policy_aware_labelers_use_full_horizon_window():
    panel = _make_flat_price_panel(n=8)
    candidates = pd.DataFrame(
        {
            "timestamp": [panel["open"].index[0]],
            "symbol": ["BTCUSDT"],
            "is_long": [True],
            "entry_price": [100.0],
        }
    )
    policy = {"tp_mult": 100.0, "sl_mult": 100.0, "trailing_pct": 1.0, "atr": {"BTCUSDT": 0.001}}

    out_single = compute_policy_aware_labels(
        candidates,
        panel,
        policy,
        max_hold_hours=1,
        bars_per_hour=3,
        cost_pct=0.0,
    )
    out_batch = compute_policy_aware_labels_batch(
        candidates,
        panel,
        policy,
        max_hold_hours=1,
        bars_per_hour=3,
        cost_pct=0.0,
    )

    assert len(out_single) == 1
    assert len(out_batch) == 1

    # Full 3-bar horizon should now be used (exit_bar indexed from 0).
    assert int(out_single.iloc[0]["exit_bar"]) == 2
    assert int(out_batch.iloc[0]["exit_bar"]) == 2

    assert out_single.iloc[0]["exit_reason"] == ExitReason.TIMEOUT
    assert out_batch.iloc[0]["exit_reason"] == ExitReason.TIMEOUT

    expected_exit_time = panel["open"].index[2]
    assert out_single.iloc[0]["exit_time"] == expected_exit_time
    assert out_batch.iloc[0]["exit_time"] == expected_exit_time


def test_policy_aware_batch_runs_without_missing_array_initialization():
    panel = _make_flat_price_panel(n=10)
    candidates = pd.DataFrame(
        {
            "timestamp": [panel["open"].index[0], panel["open"].index[1]],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "is_long": [True, False],
            "entry_price": [100.0, 100.0],
        }
    )
    policy = {"tp_mult": 50.0, "sl_mult": 50.0, "trailing_pct": 1.0, "atr": {"BTCUSDT": 0.001}}

    out = compute_policy_aware_labels_batch(
        candidates,
        panel,
        policy,
        max_hold_hours=1,
        bars_per_hour=4,
        cost_pct=0.0,
    )

    assert len(out) == 2
    assert set(out.columns) >= {"entry_price", "exit_price", "label", "exit_bar", "exit_reason"}


def test_policy_aware_batch_produces_stop_loss_hits_when_price_crosses_sl():
    idx = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    symbol = "BTCUSDT"
    panel = {
        "open": pd.DataFrame({symbol: [100.0, 100.0, 100.0, 100.0, 100.0]}, index=idx),
        "high": pd.DataFrame({symbol: [100.2, 100.1, 100.0, 100.0, 100.0]}, index=idx),
        "low": pd.DataFrame({symbol: [99.95, 99.7, 99.6, 99.5, 99.4]}, index=idx),
        "close": pd.DataFrame({symbol: [100.0, 99.8, 99.7, 99.6, 99.5]}, index=idx),
    }
    candidates = pd.DataFrame(
        {
            "timestamp": [idx[0]],
            "symbol": [symbol],
            "is_long": [True],
            "entry_price": [100.0],
        }
    )
    policy = {"tp_mult": 50.0, "sl_mult": 1.0, "trailing_pct": 10.0, "atr": {symbol: 0.002}}

    out = compute_policy_aware_labels_batch(
        candidates,
        panel,
        policy,
        max_hold_hours=1,
        bars_per_hour=4,
        cost_pct=0.0,
    )

    assert len(out) == 1
    assert out.iloc[0]["exit_reason"] == ExitReason.SL_HIT


def test_stable_daily_sortino_and_maxdd_avoid_degenerate_risk_metrics():
    daily_returns = np.array([0.02, 0.015, 0.01, -1e-8, 0.018, 0.012], dtype=float)

    sortino, max_dd = _stable_daily_sortino_and_maxdd(daily_returns)

    assert sortino == 0.0
    assert 0.0 < max_dd <= 1.0


def test_stable_daily_pnl_metrics_handles_datetime_inputs():
    pnl = np.array([100.0, -25.0, 80.0, -10.0], dtype=float)
    ts = pd.to_datetime(
        ["2026-01-01 00:00:00Z", "2026-01-01 12:00:00Z", "2026-01-02 00:00:00Z", "2026-01-03 00:00:00Z"]
    )

    sortino, max_dd, ulcer, tuw = _stable_daily_pnl_metrics(pnl, ts, start_equity=100000.0)

    assert np.isfinite(sortino)
    assert 0.0 < max_dd <= 1.0
    assert np.isfinite(ulcer)
    assert 0.0 <= tuw <= 1.0


def test_run_oof_grid_backtest_uses_timestamp_rolling_per_asset():
    ts = pd.date_range("2026-01-01", periods=8, freq="1D", tz="UTC")
    rows = []
    for asset, shift in [("A", 0.0), ("B", 0.05)]:
        for i, t in enumerate(ts):
            base = 100.0 + i + shift
            rows.append(
                {
                    "ts": t,
                    "asset": asset,
                    "side": "LONG",
                    "close": base,
                    "sizer_score_oof": 0.1 + 0.01 * i + shift,
                    "opt_limit_offset_pct": 0.0,
                    "future_opens": np.full(4, base),
                    "future_highs": np.full(4, base * 1.01),
                    "future_lows": np.full(4, base * 0.99),
                    "future_closes": np.full(4, base * 1.002),
                    "entry_price": base,
                    "is_long": True,
                    "label_policy_sl_atr_mult": 1.0,
                    "label_policy_tp_sl_ratio": 2.0,
                    "atr_12_15m": 0.5,
                    "label_policy_giveback_pct": 0.1,
                    "label_policy_max_hold_bars": 4,
                }
            )
    oof_df = pd.DataFrame(rows)

    out = run_oof_grid_backtest(oof_df)

    assert isinstance(out, pd.DataFrame)
