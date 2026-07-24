import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from extreme_price_movements import simple_policy_optimiser as spo


@pytest.fixture(autouse=True)
def _isolate_policy_spread_baseline(monkeypatch):
    monkeypatch.setenv("EPM_SIMPLE_POLICY_USE_SPREAD_MODEL", "0")
    monkeypatch.delenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", raising=False)
    monkeypatch.delenv("EPM_SIMPLE_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS", raising=False)
    spo._SPREAD_MODEL_COST_CACHE.clear()
    spo._SPREAD_BASELINE_CACHE.clear()
    yield
    spo._SPREAD_MODEL_COST_CACHE.clear()
    spo._SPREAD_BASELINE_CACHE.clear()


def test_policy_spread_fallback_quantile_defaults_to_p85():
    assert spo._spread_fallback_quantile() == pytest.approx(0.85)


def test_worse_negative_archetype_score_gets_no_geometry_foundation():
    _, diagnostics = spo._blend_archetype_geometry_with_parent(
        archetype_params={"sl_mult": 4.0},
        archetype_size_power=1.5,
        parent_params={"sl_mult": 2.0},
        parent_size_power=1.0,
        rows=10_000,
        k=1,
        mean_score_archetype=-2.0,
        std_score=0.1,
        mean_score_parent=-1.0,
    )

    assert diagnostics["performance_stability_confidence"] == 0.0
    assert diagnostics["archetype_foundation"] == 0.0


def test_improving_negative_archetype_score_gets_partial_confidence():
    _, diagnostics = spo._blend_archetype_geometry_with_parent(
        archetype_params={"sl_mult": 4.0},
        archetype_size_power=1.5,
        parent_params={"sl_mult": 2.0},
        parent_size_power=1.0,
        rows=10_000,
        k=1,
        mean_score_archetype=-0.75,
        std_score=1.0,
        mean_score_parent=-1.0,
    )

    assert diagnostics["performance_stability_confidence"] == pytest.approx(0.25)
    assert diagnostics["archetype_foundation"] == pytest.approx(0.25)


def test_policy_candidate_export_includes_entry_slippage_in_friction(monkeypatch):
    captured_kwargs = {}

    def fake_simulate_and_score(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "selected_mask": np.array([True]),
            "raw_gains": np.array([0.009], dtype=np.float64),
            "gross_gains": np.array([0.010], dtype=np.float64),
            "sizes": np.array([1.0], dtype=np.float64),
            "exit_bars": np.array([2], dtype=np.int32),
            "exit_reason": np.array(["trailing"], dtype=object),
        }

    monkeypatch.setattr(spo, "simulate_and_score", fake_simulate_and_score)
    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "symbol": ["BTC/USD:USD"],
            "side": [1.0],
            "rank_pct": [0.91],
            "calibrated_score": [0.8],
            "barrier_pct": [0.02],
            "theoretical_entry_price": [100.0],
            "entry_gap_bps": [15.0],
            "entry_slippage_proxy_bps": [4.5],
            "simple_grid_ev_bucket": [18],
            "simple_grid_gross_ev_bps": [140.0],
            "simple_grid_execution_friction_bps": [70.0],
            "simple_grid_net_ev_bps": [70.0],
            "simple_grid_selected_sl_mult": [1.0],
            "simple_grid_selected_tp_mult": [1.5],
            "meta_lgbm_uncertainty_score": [0.37],
            "meta_lgbm_inference_drift_score": [0.42],
            "meta_lgbm_feature_drift_psi_core": [0.18],
            "meta_lgbm_predictive_atlas_hit_rate_surprise": [0.07],
        }
    )
    paths = (
        np.array([[100.15, 101.0]], dtype=np.float32),
        np.array([[101.0, 102.0]], dtype=np.float32),
        np.array([[99.0, 100.0]], dtype=np.float32),
        np.array([[100.5, 101.0]], dtype=np.float32),
    )

    out = spo._build_simple_policy_candidate_rows(
        strategy_id="long_test",
        df_top=rows,
        paths=paths,
        cost_pct=0.001,
        best_params={
            "sl_mult": 1.0,
            "trailing_activation_mult": 2.5,
            "trailing_activation_cap_pct": 0.03,
        },
        best_size_power=1.0,
        base_strategy_threshold=0.75,
        market_mode="perps",
    )

    assert out["slippage_bps"].iloc[0] == 4.5
    assert out["orderbook_slippage_bps"].iloc[0] == 4.5
    assert out["expected_spread_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS
    )
    assert out["spread_cost_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert out["exit_quote_half_spread_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert out["exit_spread_cost_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert out["net_return_before_spread"].iloc[0] == pytest.approx(0.009)
    assert out["net_return_before_legacy_entry_spread_haircut"].iloc[0] == pytest.approx(
        0.009
    )
    assert out["legacy_posthoc_entry_spread_haircut_bps"].iloc[0] == pytest.approx(0.0)
    assert out["spread_adjustment_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS
    )
    assert out["net_return"].iloc[0] == pytest.approx(0.009)
    assert captured_kwargs["max_concurrent_trades"] == 1_000_000
    assert captured_kwargs["max_concurrent_per_asset"] == 1_000_000
    assert captured_kwargs["max_new_entries_per_bar"] == 1_000_000
    assert out["policy_executable_entry_price"].iloc[0] == pytest.approx(100.15)
    assert out["theoretical_entry_price"].iloc[0] == pytest.approx(100.15)
    assert out["entry_reanchor_bps"].iloc[0] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0 + 4.5
    )
    assert out["policy_trailing_activation_mult"].iloc[0] == pytest.approx(2.5)
    assert out["policy_trailing_activation_cap_pct"].iloc[0] == pytest.approx(0.03)
    assert out["policy_uncapped_trailing_activation_return"].iloc[0] == pytest.approx(
        0.05
    )
    assert out["policy_trailing_activation_return"].iloc[0] == pytest.approx(0.03)
    assert out["expected_friction_bps"].iloc[0] == pytest.approx(
        14.5
        + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
        + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert out["entry_gap_bps"].iloc[0] == 15.0
    assert out["simple_grid_ev_bucket"].iloc[0] == 18
    assert out["simple_grid_net_ev_bps"].iloc[0] == pytest.approx(70.0)
    assert out["meta_lgbm_uncertainty_score"].iloc[0] == pytest.approx(0.37)
    assert out["meta_lgbm_inference_drift_score"].iloc[0] == pytest.approx(0.42)
    assert out["meta_lgbm_feature_drift_psi_core"].iloc[0] == pytest.approx(0.18)
    assert out["meta_lgbm_predictive_atlas_hit_rate_surprise"].iloc[0] == pytest.approx(0.07)


def test_simulator_applies_trailing_activation_cap_pct():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=1,
                tz="UTC",
            ),
            "symbol": ["TEST/USD:USD"],
            "side": [1.0],
            "rank_pct": [0.95],
            "barrier_pct": [0.02],
        }
    )
    paths = (
        np.array([[100.0, 100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 104.0, 104.0]], dtype=np.float32),
        np.array([[100.0, 103.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 103.8, 103.0]], dtype=np.float32),
    )

    uncapped = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        sl_mult=1.0,
        trailing_activation_mult=2.5,
        trailing_activation_cap_pct=0.0,
        capital_protect_mfe_mult=0.0,
    )
    capped = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        sl_mult=1.0,
        trailing_activation_mult=2.5,
        trailing_activation_cap_pct=0.03,
        capital_protect_mfe_mult=0.0,
    )

    assert list(uncapped["exit_reason"]) == ["timeout"]
    assert list(capped["exit_reason"]) == ["trailing"]


def test_simulator_timeout_uses_final_close_net_of_costs():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=1,
                tz="UTC",
            ),
            "symbol": ["TEST/USD:USD"],
            "side": [1.0],
            "rank_pct": [1.0],
            "barrier_pct": [0.02],
        }
    )
    paths = (
        np.array([[100.0, 100.8, 102.0]], dtype=np.float32),
        np.array([[100.0, 101.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 100.2, 101.0]], dtype=np.float32),
        np.array([[100.0, 100.6, 102.0]], dtype=np.float32),
    )

    out = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.001,
        sl_mult=10.0,
        trailing_activation_mult=99.0,
        trailing_activation_cap_pct=0.0,
        capital_protect_mfe_mult=0.0,
    )

    size = float(out["sizes"][0])
    gross_return = float(out["gross_gains"][0]) / size
    net_return = float(out["raw_gains"][0]) / size

    assert list(out["exit_reason"]) == ["timeout"]
    assert int(out["exit_bars"][0]) == 2
    assert gross_return > 0.0
    assert gross_return > -float(out["sl_return"][0])
    assert net_return == pytest.approx(
        gross_return - 0.001 - (1.0 + gross_return) * 0.001
    )
    assert net_return > 0.0


def test_simulator_capital_protection_cannot_arm_and_exit_same_bar():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=1, tz="UTC"),
            "symbol": ["TEST/USD:USD"],
            "side": [1.0],
            "rank_pct": [1.0],
            "barrier_pct": [0.02],
        }
    )
    paths = (
        np.array([[100.0, 100.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 103.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 99.0, 100.5]], dtype=np.float32),
        np.array([[100.0, 102.0, 100.8]], dtype=np.float32),
    )

    out = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        sl_mult=5.0,
        trailing_activation_mult=99.0,
        capital_protect_mfe_mult=1.0,
        capital_protect_lock_frac=0.5,
        capital_protect_min_lock_bps=0.0,
        capital_protect_spread_lock_mult=0.0,
    )

    assert list(out["exit_reason"]) == ["capital_protect"]
    assert int(out["exit_bars"][0]) == 2


def test_policy_candidate_export_adds_regime_ae_features_without_raw_source_passthrough(monkeypatch):
    from extreme_price_movements.regime_ae_features import CURRENT_REGIME_AE_FEATURE_COLUMNS

    def fake_simulate_and_score(df, *args, **kwargs):
        n = len(df)
        return {
            "selected_mask": np.ones(n, dtype=bool),
            "raw_gains": np.linspace(-0.01, 0.02, n, dtype=np.float64),
            "gross_gains": np.linspace(-0.008, 0.024, n, dtype=np.float64),
            "sizes": np.ones(n, dtype=np.float64),
            "exit_bars": np.ones(n, dtype=np.int32),
            "exit_reason": np.repeat("time", n).astype(object),
        }

    monkeypatch.setattr(spo, "simulate_and_score", fake_simulate_and_score)
    monkeypatch.setitem(spo.CFG, "regime_ae_min_rows", 10)
    monkeypatch.setitem(spo.CFG, "regime_ae_min_features", 4)
    monkeypatch.setitem(spo.CFG, "regime_ae_max_features", 12)
    monkeypatch.setitem(spo.CFG, "regime_ae_max_train_rows", 64)
    monkeypatch.setitem(spo.CFG, "regime_ae_max_epochs", 1)
    monkeypatch.setitem(spo.CFG, "regime_ae_batch_size", 16)
    monkeypatch.setitem(spo.CFG, "regime_ae_candidate_generation", "walk_forward_prior_only")
    monkeypatch.setitem(spo.CFG, "regime_ae_oof_block_hours", 12)
    monkeypatch.setitem(spo.CFG, "regime_ae_walk_forward_min_prior_rows", 20)

    n = 40
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "symbol": np.where(np.arange(n) % 2 == 0, "BTC/USD:USD", "ETH/USD:USD"),
            "side": np.ones(n, dtype=np.float32),
            "rank_pct": np.linspace(0.80, 1.0, n),
            "calibrated_score": np.linspace(0.30, 0.90, n),
            "barrier_pct": np.full(n, 0.02),
            **{
                f"selected_feature_{i}": np.sin(np.arange(n) / (i + 2.0))
                for i in range(10)
            },
        }
    )
    paths = (
        np.full((n, 2), 100.0, dtype=np.float32),
        np.full((n, 2), 101.0, dtype=np.float32),
        np.full((n, 2), 99.0, dtype=np.float32),
        np.full((n, 2), 100.5, dtype=np.float32),
    )

    out = spo._build_simple_policy_candidate_rows(
        strategy_id="long_test",
        df_top=rows,
        paths=paths,
        cost_pct=0.001,
        best_params={"sl_mult": 1.0, "trailing_activation_mult": 1.0},
        best_size_power=1.0,
        base_strategy_threshold=0.80,
        market_mode="perps",
        regime_ae_feature_columns=[f"selected_feature_{i}" for i in range(10)],
        regime_ae_fit_frame=rows,
    )

    assert set(CURRENT_REGIME_AE_FEATURE_COLUMNS).issubset(out.columns)
    assert np.isfinite(out[list(CURRENT_REGIME_AE_FEATURE_COLUMNS)].to_numpy()).all()
    assert "selected_feature_0" not in out.columns
    assert out.attrs["current_regime_ae_state"]["enabled"] is True
    assert out.attrs["current_regime_ae_diagnostics"]["source_feature_count"] >= 4
    generation = out.attrs["current_regime_ae_diagnostics"]["candidate_generation"]
    assert generation["mode"] == "walk_forward_prior_only"
    assert generation["enabled_blocks"] >= 1


def test_alpha_uncertainty_context_exports_base_lgbm_drift_features():
    class AlphaModel:
        def transform_meta_features(self, frame):
            return pd.DataFrame(
                {
                    "lgbm_prob": [0.7, 0.6],
                    "entropy": [0.4, 0.5],
                    "uncertainty_score": [0.21, 0.31],
                    "inference_drift_score": [0.41, 0.51],
                    "feature_drift_psi_core_80": [0.12, 0.22],
                    "feature_drift_ks_bin_mean": [0.13, 0.23],
                },
                index=frame.index,
            )

    meta_base = pd.DataFrame(
        {
            "long_test": [0.7, 0.6],
            "feature_a": [1.0, 2.0],
        },
        index=pd.Index([10, 11]),
    )

    out = spo._add_alpha_model_uncertainty_context(
        meta_base,
        alpha_model=AlphaModel(),
        strategy_id="long_test",
        horizon=10,
    )

    assert np.allclose(out["base_lgbm_uncertainty_score"], [0.21, 0.31])
    assert np.allclose(out["base_lgbm_inference_drift_score"], [0.41, 0.51])
    assert np.allclose(out["base_lgbm_feature_drift_psi_core"], [0.12, 0.22])
    assert np.allclose(out["base_lgbm_feature_drift_ks_core"], [0.13, 0.23])
    assert "base_H10_pred_std" in out.columns


def test_replay_threshold_selector_can_lower_threshold_without_trade_count_floor(tmp_path):
    n = 12
    timestamps = pd.date_range("2026-01-01", periods=n, freq="6h", tz="UTC")
    ranks = np.linspace(0.70, 0.99, n)
    # Lower-rank candidates are still positive, so the replay objective should
    # prefer throughput over the previous high threshold.
    net_returns = np.linspace(0.018, 0.010, n)
    candidates = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [f"SYM{i}/USD:USD" for i in range(n)],
            "side": ["short"] * n,
            "strategy_id": ["short_test"] * n,
            "base_strategy_threshold": [0.92] * n,
            "calibrated_score": ranks,
            "normalized_rank_score": ranks,
            "strategy_rank_pct": ranks,
            "entry_price": [100.0] * n,
            "exit_timestamp": timestamps + pd.Timedelta(hours=2),
            "exit_price": [98.0] * n,
            "net_return": net_returns,
            "gross_return": net_returns + 0.002,
            "holding_bars": [8] * n,
            "simple_policy_exit_reason": ["trailing"] * n,
        }
    )
    payload = {
        "strategies": [
            {
                "strategy_id": "short_test",
                "selected": True,
                "deployment_rank_threshold": 0.92,
                "deployment_threshold_metrics": {},
            }
        ],
        "selection_rules": {},
    }

    updated_candidates, updated_replay, report = (
        spo._select_deployment_threshold_by_portfolio_replay(
            payload,
            candidates.copy(),
            candidates.copy(),
            output_dir=tmp_path,
            market_mode="perps",
        )
    )

    assert report["updated"] is True
    assert report["selected_threshold"] < 0.92
    assert report["deployment_threshold_updated"] is False
    assert report["deployment_threshold_updates_are_diagnostic_only"] is True
    assert report["global_min_trade_count_is_diagnostic_only"] is True
    assert report["selected"]["global_min_trade_count_met"] is False
    assert payload["strategies"][0]["deployment_rank_threshold"] == pytest.approx(
        0.92
    )
    selector = payload["strategies"][0]["deployment_threshold_metrics"][
        "portfolio_replay_threshold_selector"
    ]
    assert selector["applied_to_deployment_threshold"] is False
    assert selector["applied_threshold"] == pytest.approx(0.92)
    assert updated_candidates["base_strategy_threshold"].eq(0.92).all()
    assert updated_replay["base_strategy_threshold"].eq(0.92).all()
    assert (tmp_path / "deployment_threshold_sensitivity.csv").exists()
    assert (tmp_path / "deployment_threshold_sensitivity.json").exists()


def test_replay_threshold_selector_blocks_negative_recent_windows(tmp_path):
    timestamps = pd.date_range("2026-01-01", periods=42, freq="1D", tz="UTC")
    rows = []
    for i, ts in enumerate(timestamps):
        in_recent_week = ts > timestamps.max() - pd.Timedelta(days=7)
        high_ts = ts + pd.Timedelta(hours=12)
        rows.append(
            {
                "timestamp": ts,
                "symbol": f"LOW{i}/USD:USD",
                "side": "short",
                "strategy_id": "short_test",
                "base_strategy_threshold": 0.92,
                "calibrated_score": 0.72,
                "normalized_rank_score": 0.72,
                "strategy_rank_pct": 0.72,
                "entry_price": 100.0,
                "exit_timestamp": ts + pd.Timedelta(hours=2),
                "exit_price": 98.0,
                "net_return": -0.02 if in_recent_week else 0.03,
                "gross_return": -0.018 if in_recent_week else 0.032,
                "holding_bars": 8,
                "simple_policy_exit_reason": "trailing",
            }
        )
        rows.append(
            {
                "timestamp": high_ts,
                "symbol": f"HIGH{i}/USD:USD",
                "side": "short",
                "strategy_id": "short_test",
                "base_strategy_threshold": 0.92,
                "calibrated_score": 0.94,
                "normalized_rank_score": 0.94,
                "strategy_rank_pct": 0.94,
                "entry_price": 100.0,
                "exit_timestamp": high_ts + pd.Timedelta(hours=2),
                "exit_price": 98.0,
                "net_return": 0.01,
                "gross_return": 0.012,
                "holding_bars": 8,
                "simple_policy_exit_reason": "trailing",
            }
        )
    candidates = pd.DataFrame(rows)
    payload = {
        "strategies": [
            {
                "strategy_id": "short_test",
                "selected": True,
                "deployment_rank_threshold": 0.92,
                "deployment_threshold_metrics": {},
            }
        ],
        "selection_rules": {},
    }

    _, _, report = spo._select_deployment_threshold_by_portfolio_replay(
        payload,
        candidates.copy(),
        candidates.copy(),
        output_dir=tmp_path,
        market_mode="perps",
    )

    sensitivity = pd.read_csv(tmp_path / "deployment_threshold_sensitivity.csv")
    row_070 = sensitivity.loc[
        sensitivity["deployment_rank_threshold"].round(2).eq(0.70)
    ].iloc[0]
    assert bool(row_070["full_hard_floor_pass"]) is True
    assert bool(row_070["hard_floor_window_7d_pass"]) is False
    assert bool(row_070["hard_floor_pass"]) is False
    assert report["updated"] is True
    assert report["selected_threshold"] > 0.72
    assert report["selected"]["hard_floor_window_7d_pass"] is True
    assert report["selected"]["hard_floor_window_14d_pass"] is True
    assert report["selected"]["hard_floor_window_28d_pass"] is True


def test_simulate_and_score_reanchors_entry_spread_into_path_geometry(monkeypatch):
    monkeypatch.setenv("EPM_SIMPLE_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS", "0")
    rows = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
            ],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "market_mode": ["perps", "perps"],
            "side": [1.0, -1.0],
            "rank_pct": [0.95, 0.95],
            "barrier_pct": [0.01, 0.01],
            "expected_half_spread_bps": [10.0, 10.0],
            "entry_slippage_proxy_bps": [5.0, 5.0],
        }
    )
    paths = (
        np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 102.0], [100.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 98.0], [100.0, 98.0]], dtype=np.float32),
        np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32),
    )

    metrics = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        size_power=1.0,
        sl_mult=10.0,
        trailing_activation_mult=10.0,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
        market_mode="perps",
    )

    assert metrics["theoretical_entry_prices"].tolist() == pytest.approx(
        [100.0, 100.0]
    )
    assert metrics["entry_reanchor_bps"].tolist() == pytest.approx([15.0, 15.0])
    assert metrics["entry_prices"].tolist() == pytest.approx([100.15, 99.85])


def test_simulate_and_score_defaults_perp_exit_quote_gap(monkeypatch):
    monkeypatch.delenv("EPM_SIMPLE_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS", raising=False)
    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "symbol": ["PNUT/USD:USD"],
            "market_mode": ["perps"],
            "side": [-1.0],
            "rank_pct": [0.95],
            "barrier_pct": [0.01],
        }
    )
    paths = (
        np.array([[100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 101.5]], dtype=np.float32),
        np.array([[100.0, 99.5]], dtype=np.float32),
        np.array([[100.0, 101.0]], dtype=np.float32),
    )

    metrics = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        size_power=1.0,
        sl_mult=1.0,
        trailing_activation_mult=10.0,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
    )

    assert metrics["exit_quote_half_spread_bps"] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert metrics["full_sl_exit_count"] == 1


def test_policy_spread_columns_use_asset_average_with_global_fallback(monkeypatch, tmp_path):
    baseline_path = tmp_path / "per_asset_spread_baseline_latest.csv"
    baseline_path.write_text(
        "symbol,rows,average_spread_bps,median_spread_bps,p75_spread_bps,average_spread_ticks\n"
        "BTC/USD:USD,3,6.0,5.0,7.0,1.0\n"
        "ETH/USD:USD,1,18.0,16.0,20.0,3.0\n"
    )
    monkeypatch.setenv("EPM_SIMPLE_POLICY_USE_SPREAD_MODEL", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", str(baseline_path))
    spo._SPREAD_MODEL_COST_CACHE.clear()
    spo._SPREAD_BASELINE_CACHE.clear()
    rows = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "MISSING/USD:USD"],
            "market_mode": ["perps", "perps"],
        }
    )

    out = spo._with_policy_spread_cost_columns(rows, market_mode="perps")

    assert out["expected_spread_bps"].tolist() == pytest.approx([6.0, 9.0])
    assert out["expected_half_spread_bps"].tolist() == pytest.approx([3.0, 4.5])
    assert out["exit_spread_cost_bps"].tolist() == pytest.approx([3.0, 4.5])


def test_policy_spread_columns_prefer_asset_p90(monkeypatch, tmp_path):
    baseline_path = tmp_path / "per_asset_spread_baseline_latest.csv"
    baseline_path.write_text(
        "symbol,rows,average_spread_bps,p90_spread_bps\n"
        "BTC/USD:USD,100,6.0,11.0\n"
        "ETH/USD:USD,100,18.0,27.0\n"
    )
    monkeypatch.setenv("EPM_SIMPLE_POLICY_USE_SPREAD_MODEL", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", str(baseline_path))
    spo._SPREAD_MODEL_COST_CACHE.clear()
    spo._SPREAD_BASELINE_CACHE.clear()
    rows = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "market_mode": ["perps", "perps"],
        }
    )

    out = spo._with_policy_spread_cost_columns(rows, market_mode="perps")

    assert out["expected_spread_bps"].tolist() == pytest.approx([11.0, 27.0])
    assert out["expected_half_spread_bps"].tolist() == pytest.approx([5.5, 13.5])


def test_policy_spread_columns_accept_baseline_json_summary(monkeypatch, tmp_path):
    baseline_path = tmp_path / "per_asset_spread_baseline_latest.json"
    baseline_path.write_text(
        """
        {
          "schema": "per_asset_spread_baseline_v1",
          "global_average_spread_bps": 11.0,
          "per_asset_spread_baseline": [
            {"symbol": "BTC/USD:USD", "rows": 3, "average_spread_bps": 6.0},
            {"symbol": "ETH/USD:USD", "rows": 1, "average_spread_bps": 18.0}
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("EPM_SIMPLE_POLICY_USE_SPREAD_MODEL", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", str(baseline_path))
    spo._SPREAD_MODEL_COST_CACHE.clear()
    spo._SPREAD_BASELINE_CACHE.clear()
    rows = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "MISSING/USD:USD"],
            "market_mode": ["perps", "perps"],
        }
    )

    out = spo._with_policy_spread_cost_columns(rows, market_mode="perps")
    audit = spo._policy_spread_baseline_audit()

    assert out["expected_spread_bps"].tolist() == pytest.approx([6.0, 11.0])
    assert out["expected_half_spread_bps"].tolist() == pytest.approx([3.0, 5.5])
    assert out["exit_spread_cost_bps"].tolist() == pytest.approx([3.0, 5.5])
    assert audit["loaded"] is True
    assert audit["format"] == "json"
    assert audit["symbol_count"] == 2
    assert audit["global_average_spread_bps"] == pytest.approx(11.0)


def test_policy_spread_fallback_trims_illiquid_tail_and_uses_slice_universe(
    monkeypatch, tmp_path
):
    baseline_path = tmp_path / "per_asset_spread_baseline_latest.csv"
    rows = [
        "symbol,rows,average_spread_bps,median_spread_bps,p75_spread_bps,average_spread_ticks",
        "LOW/USD:USD,1,10.0,10.0,10.0,1.0",
        "MID/USD:USD,1,30.0,30.0,30.0,1.0",
    ]
    rows.extend(
        f"OK{i}/USD:USD,1,50.0,50.0,50.0,1.0"
        for i in range(27)
    )
    rows.append("WIDE/USD:USD,1,1000.0,1000.0,1000.0,1.0")
    baseline_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_USE_SPREAD_MODEL", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", str(baseline_path))
    monkeypatch.setattr(spo, "POLICY_SPREAD_FALLBACK_MAX_QUANTILE", 0.75)
    monkeypatch.setattr(spo, "POLICY_SPREAD_FALLBACK_MIN_SYMBOLS", 20)
    spo._SPREAD_MODEL_COST_CACHE.clear()
    spo._SPREAD_BASELINE_CACHE.clear()

    rows = pd.DataFrame(
        {
            "symbol": ["LOW/USD:USD", "MID/USD:USD", "MISSING/USD:USD"],
            "market_mode": ["perps", "perps", "perps"],
        }
    )

    out = spo._with_policy_spread_cost_columns(rows, market_mode="perps")
    audit = spo._policy_spread_baseline_audit()

    assert out["expected_spread_bps"].tolist() == pytest.approx([10.0, 30.0, 20.0])
    assert spo._policy_expected_spread_bps("perps") == pytest.approx(
        (10.0 + 30.0 + 27.0 * 50.0) / 29.0
    )
    assert audit["raw_global_average_spread_bps"] == pytest.approx(
        (10.0 + 30.0 + 27.0 * 50.0 + 1000.0) / 30.0
    )
    assert audit["global_average_spread_bps"] == pytest.approx(
        (10.0 + 30.0 + 27.0 * 50.0) / 29.0
    )
    assert audit["fallback_method"] == "liquidity_tail_trimmed_average"
    assert audit["fallback_symbol_count"] == 29


def test_simple_rank_net_ev_prefilter_subtracts_row_level_spread(monkeypatch):
    rows = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 00:15:00", tz="UTC"),
            ],
            "symbol": ["LOW/USD:USD", "WIDE/USD:USD"],
            "market_mode": ["perps", "perps"],
            "side": [1.0, 1.0],
            "rank_pct": [0.91, 0.81],
            "calibrated_score": [0.9, 0.8],
            "barrier_pct": [0.01, 0.01],
            "expected_half_spread_bps": [5.0, 75.0],
            "exit_spread_cost_bps": [5.0, 75.0],
            "entry_slippage_proxy_bps": [0.0, 0.0],
        }
    )
    paths = (
        np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 101.1], [100.0, 101.1]], dtype=np.float32),
        np.array([[100.0, 99.9], [100.0, 99.9]], dtype=np.float32),
        np.array([[100.0, 101.0], [100.0, 101.0]], dtype=np.float32),
    )

    fit = spo._fit_simple_rank_net_ev_prefilter(
        rows,
        paths,
        cost_pct=0.0,
        market_mode="perps",
        sl_mults=(1.0,),
        tp_mults=(1.0,),
        bucket_count=10,
        min_bucket_rows=1,
        min_net_ev_bps=0.0,
    )
    filtered, keep_mask, summary = spo._apply_simple_rank_net_ev_prefilter(
        rows,
        fit,
        cost_pct=0.0,
        market_mode="perps",
        context="unit_test",
    )

    assert fit["status"] == "fit"
    assert keep_mask.tolist() == [True, True]
    assert summary["binding"] is False
    assert summary["status"] == "diagnostic_only"
    assert summary["diagnostic_rows_after"] == 1
    assert summary["rows_after"] == 2
    assert filtered["symbol"].tolist() == ["LOW/USD:USD", "WIDE/USD:USD"]
    assert filtered["simple_grid_gross_ev_bps"].iloc[0] > 80.0
    assert filtered["simple_grid_net_ev_bps"].iloc[0] > 70.0

    monkeypatch.setattr(spo, "SIMPLE_NET_EV_PREFILTER_BINDING", True)
    binding_filtered, binding_keep_mask, binding_summary = (
        spo._apply_simple_rank_net_ev_prefilter(
            rows,
            {**fit, "uses_final_exit_policy": True},
            cost_pct=0.0,
            market_mode="perps",
            context="unit_test_final_policy",
        )
    )
    assert binding_keep_mask.tolist() == [True, False]
    assert binding_summary["binding"] is True
    assert binding_summary["rows_after"] == 1
    assert binding_filtered["symbol"].tolist() == ["LOW/USD:USD"]


def test_simulate_and_score_uses_row_level_exit_half_spread():
    rows = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
            ],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "market_mode": ["perps", "perps"],
            "side": [1.0, 1.0],
            "rank_pct": [1.0, 1.0],
            "barrier_pct": [0.10, 0.10],
            "expected_half_spread_bps": [0.0, 0.0],
            "exit_spread_cost_bps": [0.0, 100.0],
        }
    )
    paths = (
        np.array([[100.0], [100.0]], dtype=np.float32),
        np.array([[100.0], [100.0]], dtype=np.float32),
        np.array([[100.0], [100.0]], dtype=np.float32),
        np.array([[100.0], [100.0]], dtype=np.float32),
    )

    metrics = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        size_power=1.0,
        sl_mult=10.0,
        trailing_activation_mult=10.0,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
        market_mode="perps",
    )

    assert metrics["exit_spread_cost_bps"].tolist() == pytest.approx([0.0, 100.0])
    raw_returns = np.asarray(metrics["raw_gains"]) / np.asarray(metrics["sizes"])
    assert raw_returns[1] < raw_returns[0] - 0.009


def test_exit_pressure_tightens_hard_take_profit_threshold():
    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "symbol": ["BTC/USD"],
            "market_mode": ["spot"],
            "side": [1.0],
            "rank_pct": [0.95],
            "barrier_pct": [0.02],
        }
    )
    paths = (
        np.array([[100.0, 100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 101.0, 101.0]], dtype=np.float32),
        np.array([[100.0, 99.0, 99.0]], dtype=np.float32),
        np.array([[100.0, 100.5, 100.5]], dtype=np.float32),
    )

    baseline = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        sl_mult=10.0,
        trailing_activation_mult=10.0,
        hard_tp_abs_pct=0.02,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
        market_mode="spot",
    )
    tightened = spo.simulate_and_score(
        rows,
        *paths,
        cost_pct=0.0,
        sl_mult=10.0,
        trailing_activation_mult=10.0,
        hard_tp_abs_pct=0.02,
        exit_pressure_enabled=True,
        exit_pressure_alpha=1.0,
        exit_pressure_beta=1.0,
        exit_pressure_delta=1.0,
        exit_pressure_kappa=1.0,
        exit_pressure_min_multiplier=0.25,
        target_holding_hours=1.0 / 60.0,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
        market_mode="spot",
    )

    assert baseline["hard_tp_exit_count"] == 0
    assert tightened["hard_tp_exit_count"] == 1
    assert tightened["exit_pressure_p75"] > 0.0
    assert tightened["tightening_multiplier_p25"] < 1.0


def test_policy_round_trip_friction_fills_missing_spread_defaults():
    rows = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "ETH/USD:USD"],
            "market_mode": ["perps", "perps"],
            "expected_half_spread_bps": [np.nan, 6.0],
            "exit_spread_cost_bps": [np.nan, 8.0],
            "entry_slippage_proxy_bps": [1.0, 2.0],
        }
    )

    friction = spo._policy_round_trip_friction_bps(
        rows,
        cost_pct=0.001,
        exit_quote_half_spread_bps=7.0,
        market_mode="perps",
    )

    assert friction[0] == pytest.approx(
        20.0 + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0 + 7.0 + 1.0
    )
    assert friction[1] == pytest.approx(20.0 + 6.0 + 8.0 + 2.0)


def test_policy_band_metrics_include_execution_cost_assumptions():
    rows = pd.DataFrame(
        {
            "rank_pct": [0.82, 0.87],
            "net_return": [0.003, 0.001],
            "gross_return": [0.008, 0.006],
            "expected_spread_bps": [spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS] * 2,
            "expected_half_spread_bps": [
                spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
            ]
            * 2,
            "spread_cost_bps": [spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0]
            * 2,
            "exit_quote_half_spread_bps": [
                spo.DEFAULT_PERP_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS
            ]
            * 2,
            "exit_spread_cost_bps": [
                spo.DEFAULT_PERP_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS
            ]
            * 2,
            "entry_delay_target_minutes": [float(spo.POLICY_DELAYED_ENTRY_MINUTES)] * 2,
            "entry_delay_actual_minutes": [5.0, 6.0],
            "entry_slippage_proxy_bps": [4.0, 8.0],
            "fees_bps": [20.0, 20.0],
            "expected_friction_bps": [
                20.0
                + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
                + spo.DEFAULT_PERP_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS
                + 4.0,
                20.0
                + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
                + spo.DEFAULT_PERP_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS
                + 8.0,
            ],
            "entry_execution_source": [
                "delayed_1m_intraminute_proxy",
                "delayed_1m_intraminute_proxy",
            ],
        }
    )

    metrics = spo._candidate_band_metrics(
        rows,
        rank_col="rank_pct",
        group_name="strategy",
        strategy_id="long_test",
        band_lo=0.8,
        band_hi=0.85,
        selection_type="local_band",
    )

    assert spo.POLICY_DELAYED_ENTRY_MINUTES == 5
    assert metrics["mean_expected_spread_bps"] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS
    )
    assert metrics["mean_spread_cost_bps"] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
    )
    assert metrics["mean_exit_spread_cost_bps"] == pytest.approx(
        spo.DEFAULT_PERP_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS
    )
    assert metrics["mean_entry_delay_target_minutes"] == pytest.approx(5.0)
    assert metrics["mean_entry_delay_actual_minutes"] == pytest.approx(5.5)
    assert metrics["mean_entry_slippage_proxy_bps"] == pytest.approx(6.0)
    assert metrics["mean_expected_friction_bps"] == pytest.approx(
        20.0
        + spo.DEFAULT_PERP_POLICY_EXPECTED_SPREAD_BPS / 2.0
        + spo.DEFAULT_PERP_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS
        + 6.0
    )
    assert metrics["entry_execution_source_counts"] == {
        "delayed_1m_intraminute_proxy": 2
    }


def test_deployment_payload_exports_asset_exclusion_map(monkeypatch):
    monkeypatch.setenv("EPM_POLICY_DEPLOYMENT_SELECTION_METRIC", "top_5")
    monkeypatch.setenv("EPM_POLICY_MAX_DEPLOYMENT_STRATEGIES", "1")
    monkeypatch.setenv("EPM_POLICY_MAX_DEPLOYMENT_STRATEGIES_PER_SIDE", "1")
    monkeypatch.setattr(
        spo,
        "_load_lgbm_mask_contracts_for_deployment",
        lambda market_mode="spot": {},
    )
    monkeypatch.setattr(
        spo,
        "_require_lgbm_mask_contracts_for_deployment",
        lambda: False,
    )

    metrics = {
        "top_1": {
            "n_trades": 10,
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
        },
        "top_5": {
            "avg_pnl_bankroll": 0.01,
            "w_std": 0.001,
            "m_std": 0.001,
            "n_trades": 20,
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
        },
    }
    payload = spo._build_deployment_payload(
        run_id="unit",
        oos_results_json={
            "strategies": {
                "long_test": {
                    "validation_metrics": metrics,
                    "best_params": {"sl_mult": 1.0},
                    "asset_metrics": [
                        {
                            "symbol": "BAD/USD:USD",
                            "asset_decision": spo.ASSET_DECISION_BLACKLIST,
                        },
                        {
                            "symbol": "OK/USD:USD",
                            "asset_decision": spo.ASSET_DECISION_KEEP,
                        },
                    ],
                }
            }
        },
        available_strategy_ids={"long_test"},
        market_mode="perps",
    )

    assert payload["asset_exclusions"] == {"long_test": ["BAD/USD:USD"]}
    assert payload["strategies"][0]["excluded_symbols"] == ["BAD/USD:USD"]


def test_local_candidate_guard_uses_per_strategy_rank_and_filters_replay(monkeypatch):
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_HIT_GUARD_ENABLED", True)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 2)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.02)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_NET_HIT_RATE", 0.50)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_GROSS_HIT_RATE", 0.50)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_BPS_WEIGHTED_HIT", 0.50)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.0)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_BANDS", 0)
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_MIN_POSITIVE",
        0,
    )

    payload = {
        "strategies": [
            {
                "strategy_id": "long_test",
                "selected": True,
                "deployment_rank_threshold": 0.50,
            }
        ],
        "rejected_strategies": [],
    }
    candidates = pd.DataFrame(
        {
            "strategy_id": ["long_test"] * 4,
            "strategy_rank_pct": [0.50, 0.51, 0.54, 0.55],
            "auction_rank_score": [0.99, 0.98, 0.10, 0.11],
            "net_return": [-0.010, -0.020, 0.010, 0.020],
            "gross_return": [-0.005, -0.010, 0.015, 0.025],
        }
    )

    summary = spo._apply_local_candidate_hit_rate_guard(
        payload,
        candidates,
        candidate_path=Path("unit.parquet"),
    )

    guard = payload["strategies"][0]["local_candidate_hit_rate_guard"]
    assert summary["rank_col"] == "strategy_rank_pct"
    assert summary["rank_scope"] == "per_strategy"
    assert payload["strategies"][0]["deployment_rank_threshold"] == pytest.approx(0.54)
    assert guard["selected_mean_net_return"] > 0.0
    assert guard["selected_bps_weighted_hit"] == pytest.approx(1.0)

    replay = spo._filter_candidates_to_deployment_strategies(
        candidates,
        ["long_test"],
        deployment_payload=payload,
    )

    assert replay["strategy_rank_pct"].min() == pytest.approx(0.54)
    assert replay["base_strategy_threshold"].tolist() == pytest.approx([0.54, 0.54])
    assert replay["deployment_rank_threshold"].tolist() == pytest.approx([0.54, 0.54])
    assert replay["threshold_rank_score_source"].tolist() == [
        "policy_rank_pct",
        "policy_rank_pct",
    ]
    assert len(replay) == 2
