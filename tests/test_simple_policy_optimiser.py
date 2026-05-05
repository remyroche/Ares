import numpy as np
import optuna
import pandas as pd

from extreme_price_movements.simple_policy_optimiser import (
    _build_top5_validation_diagnostic,
    _suggest_policy_params,
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
    f_opens = np.array([[100.0, 100.0, 101.0]], dtype=np.float32)
    f_highs = np.array([[100.0, 100.0, 101.0]], dtype=np.float32)
    f_lows = np.array([[100.0, 99.5, 99.0]], dtype=np.float32)
    f_closes = np.array([[100.0, 100.0, 101.0]], dtype=np.float32)

    no_cap = simulate_and_score(
        df,
        f_opens,
        f_highs,
        f_lows,
        f_closes,
        capital_protect_mfe_mult=0.0,
    )
    with_cap = simulate_and_score(
        df,
        f_opens,
        f_highs,
        f_lows,
        f_closes,
    )
    size = compute_position_size(np.array([0.99], dtype=np.float32), 1.0)[0]
    expected_exit_ret = 0.01
    expected_fees = size * 0.0015 + size * (1.0 + expected_exit_ret) * 0.0015
    expected_net_gain = size * expected_exit_ret - expected_fees

    assert no_cap["total_trades"] == 1
    np.testing.assert_allclose(no_cap["raw_gains"], with_cap["raw_gains"])
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


def test_suggested_policy_params_do_not_optimize_max_concurrent_trades():
    trial = optuna.trial.FixedTrial(
        {
            "sl_mult": 0.8,
            "trailing_activation_mult": 1.0,
            "trailing_power": 1.5,
            "trailing_squash_divisor": 2.0,
            "giveback_beta": 0.7,
            "capital_protect_mfe_mult": 0.75,
            "capital_protect_regression_frac": 0.45,
        }
    )
    params = _suggest_policy_params(trial)
    assert "max_concurrent_trades" not in params
