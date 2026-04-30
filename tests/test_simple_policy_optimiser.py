import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_optimiser import (
    _build_top5_validation_diagnostic,
    compute_position_size,
    simulate_and_score,
)


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank_pct": [0.99],
            "barrier_pct": [0.02],
            "side": [1.0],
            "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")],
        }
    )


def test_capital_protect_zero_is_true_noop():
    df = _simple_df()
    f_opens = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    f_highs = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    f_lows = np.array([[100.0, 99.0, 97.0]], dtype=np.float32)
    f_closes = np.array([[100.0, 99.0, 97.0]], dtype=np.float32)

    no_cap = simulate_and_score(
        df,
        f_opens,
        f_highs,
        f_lows,
        f_closes,
        capital_protect_mfe_mult=0.0,
    )
    default_cap = simulate_and_score(
        df,
        f_opens,
        f_highs,
        f_lows,
        f_closes,
    )
    size = compute_position_size(np.array([0.99], dtype=np.float32), 1.0)[0]
    expected_exit_ret = -0.02
    expected_fees = size * 0.0015 + size * (1.0 + expected_exit_ret) * 0.0015
    expected_net_gain = size * expected_exit_ret - expected_fees

    assert no_cap["total_trades"] == 1
    np.testing.assert_allclose(no_cap["raw_gains"], default_cap["raw_gains"])
    np.testing.assert_allclose(no_cap["raw_gains"][0], expected_net_gain)


def test_top5_validation_diagnostic_uses_raw_gains_and_skips_length_mismatch(caplog):
    rows = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-02T00:00:00Z"),
            ]
        }
    )

    diag = _build_top5_validation_diagnostic(
        rows,
        {"raw_gains": np.array([0.1, -0.2], dtype=np.float32)},
    )
    skipped = _build_top5_validation_diagnostic(
        rows,
        {"raw_gains": np.array([0.1], dtype=np.float32)},
    )

    np.testing.assert_allclose(diag["net_gain"].to_numpy(), np.array([0.1, -0.2]))
    assert skipped is None
    assert "length mismatch" in caplog.text
