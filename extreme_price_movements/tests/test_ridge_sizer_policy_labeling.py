import numpy as np
import pandas as pd

from extreme_price_movements.ridge_position_sizer import (
    ExitReason,
    compute_policy_aware_labels,
    compute_policy_aware_labels_batch,
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
