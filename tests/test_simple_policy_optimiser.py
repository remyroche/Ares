import numpy as np
import optuna
import pandas as pd
import pytest

from extreme_price_movements.simple_policy_optimiser import (
    _build_top5_validation_diagnostic,
    _load_slice_plan_source_validation,
    _suggest_policy_params,
    apply_asset_weights,
    compute_position_size,
    discover_deployment_rank_threshold_simple_grid,
    optimise_deployment_rank_threshold,
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


def test_policy_oos_validation_uses_policy_plan_temporal_disjointness(tmp_path):
    slice_plan = {
        "version": 2,
        "materialized_views": {
            "train_base": {"n_plans": 1},
            "utility_policy_optimisation": {"n_plans": 1},
        },
        "consumer_plans": {
            "base_model_fit": [
                {
                    "fit_idx": [1, 2, 3],
                    "predict_idx": [4, 5],
                    "metadata": {},
                }
            ],
            "policy_optimiser": [
                {
                    # These indices intentionally overlap the base plan. They are
                    # local to the consumer plan and must not invalidate OOS.
                    "fit_idx": [1, 2, 3],
                    "predict_idx": [4, 5],
                    "metadata": {
                        "predict_role": "policy_holdout_tail",
                        "fit_end": "2026-01-01T00:00:00Z",
                        "predict_actual_start": "2026-01-02T00:00:00Z",
                    },
                }
            ],
        },
    }
    path = tmp_path / "slice_plan.json"
    path.write_text(__import__("json").dumps(slice_plan))

    validation = _load_slice_plan_source_validation(path)

    assert validation["oos_policy_slice_verified"] is True
    assert validation["policy_holdout_temporal_disjoint"] is True
    assert validation["policy_holdout_train_base_meta_fit_overlap_rows"] == 0


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


def test_deployment_rank_threshold_uses_concurrency_limited_pnl():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00Z",
                    "2026-01-01 00:00:00Z",
                    "2026-01-01 00:15:00Z",
                    "2026-01-01 00:30:00Z",
                    "2026-01-01 00:45:00Z",
                    "2026-01-01 01:00:00Z",
                ]
            ),
            "symbol": ["BTC", "ETH", "BTC", "SOL", "ETH", "XRP"],
            "calibrated_score": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "rank_pct": [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0],
            "exit_bars": [4, 4, 4, 1, 1, 1],
            "net_gain": [-1.0, -1.0, -1.0, -0.2, -0.3, 0.4],
        }
    )

    best = optimise_deployment_rank_threshold(
        rows,
        max_concurrent_per_asset=1,
        max_concurrent_per_strategy=3,
        lo=0.85,
        hi=0.99,
        precision=0.01,
    )

    assert best["deployment_rank_threshold"] >= 0.85
    assert best["n_trades"] == 1
    assert best["net_pnl"] == 0.4
    assert best["threshold_search"]["max_concurrent_per_asset"] == 1
    assert best["threshold_search"]["max_concurrent_per_strategy"] == 3


def test_simple_grid_threshold_discovery_uses_full_rank_population_below_top15():
    ranks = [0.50, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01T00:00:00Z", periods=len(ranks), freq="h"
            ),
            "symbol": list("ABCDEFG"),
            "strategy_id": ["long_a"] * len(ranks),
            "side": [1.0] * len(ranks),
            "rank_pct": ranks,
            "barrier_pct": [0.01] * len(ranks),
        }
    )
    f_opens = np.full((len(ranks), 3), 100.0, dtype=np.float32)
    f_highs = np.full((len(ranks), 3), 100.0, dtype=np.float32)
    f_lows = np.full((len(ranks), 3), 100.0, dtype=np.float32)
    f_closes = np.full((len(ranks), 3), 100.0, dtype=np.float32)
    f_lows[0, 1] = 99.0
    f_highs[1:, 1] = 103.0

    best = discover_deployment_rank_threshold_simple_grid(
        rows,
        (f_opens, f_highs, f_lows, f_closes),
        cost_pct=0.0,
        lo=0.5,
        hi=0.7,
        precision=0.02,
        sl_mults=(1.0,),
        tp_mults=(1.0,),
        local_band_width=0.02,
        confirmation_bands=5,
        confirmation_min_positive=4,
    )

    assert best["deployment_rank_threshold"] < 0.85
    assert best["deployment_rank_threshold"] == pytest.approx(0.6)
    assert best["mean_net_trade"] > 0.0
    assert best["local_confirmation_passed"] is True
    assert best["next_band_positive_count"] >= 4
    assert best["threshold_search"]["method"].startswith("full_policy_rank_grid")
    assert best["threshold_search"]["profitable_threshold_min"] == pytest.approx(0.6)


def test_asset_weights_blacklist_only_reliable_harmful_assets():
    logs = []
    metrics = pd.DataFrame(
        {
            "strategy_id": ["long_a", "long_a", "long_a"],
            "symbol": ["BAD", "MID", "GOOD"],
            "n_trades": [1000, 1000, 1000],
            "n_candidates": [2000, 2000, 2000],
            "mean_net_gain": [-0.01, 0.001, 0.002],
            "sortino": [-5.0, 1.0, 2.0],
        }
    )

    out = apply_asset_weights(
        metrics,
        policy_col="strategy_id",
        symbol_col="symbol",
        tprint_fn=logs.append,
    )

    by_symbol = out.set_index("symbol")
    assert by_symbol.loc["BAD", "asset_decision"] == "blacklist"
    assert by_symbol.loc["BAD", "asset_weight_multiplier"] == 0.0
    assert by_symbol.loc["GOOD", "asset_decision"] == "keep"
    assert by_symbol.loc["GOOD", "asset_weight_multiplier"] == 1.0
    assert any("group=blacklist" in line for line in logs)
