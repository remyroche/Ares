import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_specialist_similarity import (
    RegimeSimilarityConfig,
    SpecialistWeightConfig,
    build_regime_specialist_training_frame,
    compute_regime_similarity_to_current,
    compute_specialist_sample_weights,
    current_regime_recency_weights,
    shrink_self_distillation_towards_one,
    weighted_drift_baseline,
)


def _synthetic_regime_frame() -> pd.DataFrame:
    n_days = 70
    rows = []
    for day in range(n_days):
        for sym_i, symbol in enumerate(("BTC/USD:USD", "ETH/USD:USD")):
            t = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=day)
            phase = day / 8.0 + sym_i * 0.3
            current_bump = 1.0 if day >= n_days - 21 else 0.0
            rows.append(
                {
                    "timestamp": t,
                    "symbol": symbol,
                    "volatility_percentile": 0.4 + 0.2 * np.sin(phase) + 0.1 * current_bump,
                    "volume_percentile": 0.5 + 0.2 * np.cos(phase),
                    "correlation_percentile": 0.3 + 0.15 * np.sin(day / 13.0),
                    "cross_sectional_dispersion": 0.2 + 0.1 * np.cos(day / 10.0),
                    "funding_average": 0.01 * np.sin(day / 5.0),
                    "funding_dispersion": 0.02 + 0.01 * np.cos(day / 4.0),
                    "aggregate_oi_growth": 0.03 * np.sin(day / 6.0),
                    "oi_over_volume": 0.5 + 0.1 * np.cos(day / 9.0),
                    "breadth": 0.45 + 0.25 * np.sin(day / 11.0),
                    "trend_strength": 0.3 + 0.3 * np.cos(day / 7.0),
                    "price_entropy": 0.6 + 0.1 * np.sin(day / 12.0),
                    "feature_drift_psi_core": 0.1 + 0.05 * current_bump + 0.02 * np.sin(phase),
                    "feature_drift_ks_core": 0.08 + 0.03 * np.cos(phase),
                    "feature_covariance_drift": 0.05 + 0.04 * current_bump,
                    "base_model_feature_drift": 0.07 + 0.01 * sym_i,
                    "meta_model_feature_drift": 0.06 + 0.02 * np.sin(day / 3.0),
                    "prediction_distribution_drift": 0.04 + 0.02 * current_bump,
                    "feature_0": np.sin(phase),
                    "feature_1": np.cos(phase),
                    "feature_2": np.sin(day / 3.0) + sym_i * 0.1,
                    "feature_3": np.cos(day / 5.0),
                    "return_1h": 0.01 * np.sin(day / 6.0 + sym_i),
                }
            )
    return pd.DataFrame(rows)


def test_current_regime_recency_weights_sum_to_one_and_favor_recent_rows():
    ts = pd.date_range("2026-01-01", periods=4, freq="7D", tz="UTC")
    weights = current_regime_recency_weights(ts, current_end=ts[-1])
    assert weights.sum() == pytest.approx(1.0)
    assert weights.iloc[-1] > weights.iloc[0]


def test_regime_specialist_training_frame_outputs_similarity_and_weights():
    frame = _synthetic_regime_frame()
    cfg = RegimeSimilarityConfig(
        ae_enabled=False,
        min_candidate_rows=8,
        min_current_rows=8,
        knn_k=3,
        max_knn_current_rows=40,
        max_knn_candidate_rows=40,
    )
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=cfg,
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    assert diag["similarity"]["enabled"] is True
    assert out["similarity_to_current"].between(0.0, 1.0).all()
    assert out["window_similarity"].between(0.0, 1.0).all()
    assert out["day_similarity"].between(0.0, 1.0).all()
    assert out["regime_specialist_sample_weight"].mean() == pytest.approx(1.0)
    assert np.isfinite(out["regime_specialist_sample_weight"].to_numpy()).all()
    assert set(out["regime_specialist_bucket"]).issubset(
        {"current", "analogue", "normal", "irrelevant"}
    )
    current = out["regime_specialist_bucket"] == "current"
    assert current.any()
    assert out.loc[current, "current_regime_recency_weight"].sum() == pytest.approx(1.0)
    assert out.loc[~current, "current_regime_recency_weight"].sum() == pytest.approx(0.0)
    assert diag["similarity"]["block_scaling"]["combined_from_normalized_distances"] is True
    assert diag["similarity"]["block_scaling"]["tau"] > 0.0
    assert diag["similarity"]["weights"]["feature_drift_distance"] == pytest.approx(0.40)
    assert diag["similarity"]["weights"]["covariance_distance"] == pytest.approx(0.35)
    assert diag["similarity"]["scaling"]["source"] == "pre_current_history"
    assert diag["similarity"]["autoencoder"]["used"] is False
    assert diag["similarity"]["autoencoder"]["reason"] == "disabled"
    assert diag["similarity"]["knn"]["mode"] == "global_knn"
    assert diag["weighted_drift_baseline"]["enabled"] is True
    assert diag["weighted_drift_baseline"]["feature_count"] > 0
    assert "feature_drift_psi_core" in diag["weighted_drift_baseline"]["stats"]


def test_current_end_excludes_future_rows_from_similarity_and_weights():
    frame = _synthetic_regime_frame()
    current_end = frame["timestamp"].max() - pd.Timedelta(days=10)
    out, diag = build_regime_specialist_training_frame(
        frame,
        current_end=current_end,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )
    future = frame["timestamp"] > current_end
    assert diag["similarity"]["future_excluded_rows"] == int(future.sum())
    assert (out.loc[future, "regime_specialist_bucket"] == "future_excluded").all()
    assert out.loc[future, "similarity_to_current"].eq(0.0).all()
    assert out.loc[future, "regime_specialist_sample_weight"].eq(0.0).all()
    active = ~future
    assert out.loc[active, "regime_specialist_sample_weight"].mean() == pytest.approx(1.0)


def test_disabled_similarity_returns_non_trainable_neutral_weights():
    frame = _synthetic_regime_frame().tail(6).copy()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1"],
        similarity_config=RegimeSimilarityConfig(
            min_current_rows=100,
            min_candidate_rows=100,
            ae_enabled=False,
        ),
    )
    assert diag["similarity"]["enabled"] is False
    assert diag["sample_weight"]["should_train_specialist"] is False
    assert out["regime_specialist_sample_weight"].mean() == pytest.approx(1.0)


def test_similarity_output_preserves_original_index():
    frame = _synthetic_regime_frame().iloc[:80].copy()
    frame.index = pd.Index(np.arange(1000, 1000 + len(frame)), name="row_id")
    sim, _diag = compute_regime_similarity_to_current(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        config=RegimeSimilarityConfig(
            ae_enabled=False,
            min_candidate_rows=8,
            min_current_rows=8,
        ),
    )
    assert sim.index.equals(frame.index)


def test_explicit_column_diagnostics_and_asset_covariance():
    frame = _synthetic_regime_frame()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        market_columns=["volatility_percentile", "missing_market_feature"],
        asset_return_col="return_1h",
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=True,
            ae_min_windows=99,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    sim_diag = diag["similarity"]
    assert out["similarity_to_current"].between(0.0, 1.0).all()
    assert sim_diag["column_selection"]["market"]["source"] == "explicit"
    assert sim_diag["column_selection"]["market"]["missing_requested"] == [
        "missing_market_feature"
    ]
    assert sim_diag["asset_covariance"]["enabled"] is True
    assert sim_diag["asset_covariance"]["return_col"] == "return_1h"
    assert sim_diag["autoencoder"]["used"] is False
    assert sim_diag["autoencoder"]["reason"] == "insufficient_candidate_windows"


def test_day_similarity_is_blended_not_a_hard_multiplier():
    frame = _synthetic_regime_frame()
    strength = 0.35
    out, _diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            day_similarity_strength=strength,
            day_similarity_min_rows=2,
            min_candidate_rows=8,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_candidate_rows=40,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    historical = out["regime_specialist_bucket"] != "current"
    floor = out.loc[historical, "window_similarity"] * (1.0 - strength)
    assert (out.loc[historical, "similarity_to_current"] + 1e-6 >= floor).all()


def test_low_memory_output_and_capped_window_diagnostics():
    frame = _synthetic_regime_frame()
    out, diag = build_regime_specialist_training_frame(
        frame,
        selected_feature_columns=["feature_0", "feature_1", "feature_2", "feature_3"],
        include_input_columns=False,
        similarity_config=RegimeSimilarityConfig(
            ae_enabled=False,
            candidate_window_days=5,
            max_window_diagnostics=1,
            min_candidate_rows=4,
            min_current_rows=8,
            knn_k=3,
            max_knn_current_rows=40,
            max_knn_historical_rows=80,
        ),
        weight_config=SpecialistWeightConfig(
            tau_current=10.0,
            tau_analogue=10.0,
            tau_normal=10.0,
            tau_irrelevant=10.0,
        ),
    )

    assert "feature_0" not in out.columns
    assert "regime_specialist_sample_weight" in out.columns
    assert diag["weighted_drift_baseline"]["enabled"] is True
    assert diag["similarity"]["window_diagnostics_count"] >= len(
        diag["similarity"]["window_diagnostics"]
    )
    assert len(diag["similarity"]["window_diagnostics"]) <= 1


def test_lgbm_regime_specialist_shadow_and_active_weight_hooks():
    from extreme_price_movements import lgbm_pipeline as lp

    frame = _synthetic_regime_frame()
    selected = ["feature_0", "feature_1", "feature_2", "feature_3", "return_1h"]
    base_cfg = {
        "lgbm_regime_specialist_enabled": True,
        "lgbm_regime_specialist_objectives": ["train_base", "train_meta"],
        "lgbm_regime_specialist_ae_enabled": False,
        "lgbm_regime_specialist_min_candidate_rows": 8,
        "lgbm_regime_specialist_min_current_rows": 8,
        "lgbm_regime_specialist_knn_k": 3,
        "lgbm_regime_specialist_max_knn_current_rows": 40,
        "lgbm_regime_specialist_max_knn_historical_rows": 80,
        "lgbm_regime_specialist_tau_current": 1.0,
        "lgbm_regime_specialist_tau_analogue": 1.0,
        "lgbm_regime_specialist_tau_normal": 1.0,
        "lgbm_regime_specialist_tau_irrelevant": 1.0,
    }
    shadow_bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_base",
        cfg={**base_cfg, "lgbm_regime_specialist_shadow_only": True},
        random_state=7,
        label="unit",
    )
    assert shadow_bundle["metrics"]["regime_specialist_enabled"] is True
    assert shadow_bundle["metrics"]["regime_specialist_sample_weight_applied"] is False
    unchanged, shadow_diag = lp._apply_lgbm_regime_specialist_weights(
        np.ones(len(frame), dtype=np.float32),
        shadow_bundle,
    )
    assert shadow_diag["applied"] is False
    assert unchanged.mean() == pytest.approx(1.0)

    active_bundle = lp._build_lgbm_regime_specialist_bundle(
        frame,
        selected,
        timestamps=frame["timestamp"].to_numpy(),
        assets=frame["symbol"].to_numpy(),
        objective_mode="train_meta",
        cfg={
            **base_cfg,
            "lgbm_regime_specialist_shadow_only": False,
            "lgbm_regime_specialist_apply_sample_weight": True,
            "lgbm_regime_specialist_apply_distillation_shrink": True,
        },
        random_state=11,
        label="unit",
    )
    adjusted, active_diag = lp._apply_lgbm_regime_specialist_weights(
        np.ones(len(frame), dtype=np.float32),
        active_bundle,
    )
    assert active_bundle["apply_sample_weight"] is True
    assert active_bundle["apply_distillation_shrink"] is True
    assert active_bundle["metrics"]["regime_specialist_weighted_drift_baseline_enabled"] is True
    assert active_bundle["metrics"]["regime_specialist_weighted_drift_baseline_feature_count"] > 0
    assert active_diag["applied"] is True
    assert adjusted.mean() == pytest.approx(1.0)
    wide_base_weight = np.geomspace(0.05, 20.0, len(frame)).astype(np.float32)
    adjusted_wide, wide_diag = lp._apply_lgbm_regime_specialist_weights(
        wide_base_weight,
        active_bundle,
    )
    assert wide_diag["applied"] is True
    assert wide_diag["base_weight_preconditioned_policy"] == "unit_mean_compress_0.7_1.3"
    assert wide_diag["base_weight_preconditioned_min"] >= 0.7 - 1e-6
    assert wide_diag["base_weight_preconditioned_max"] <= 1.3 + 1e-6
    assert wide_diag["base_weight_preconditioned_mean"] == pytest.approx(1.0, abs=1e-6)
    assert adjusted_wide.mean() == pytest.approx(1.0)
    sim = lp._lgbm_regime_specialist_similarity_for_idx(active_bundle)
    assert sim is not None
    assert len(sim) == len(frame)
    current_metrics = lp._lgbm_regime_specialist_current_metrics(
        np.asarray([0, 1] * (len(frame) // 2), dtype=np.float32),
        np.linspace(0.0, 1.0, len(frame), dtype=np.float32),
        active_bundle,
        classifier=True,
    )
    assert current_metrics["current_regime_metrics_available"] is True
    assert current_metrics["current_regime_metric_rows"] > 0
    assert "current_regime_precision10" in current_metrics


def test_lgbm_regime_specialist_distillation_shrink_hook():
    from extreme_price_movements import lgbm_pipeline as lp

    adjusted = lp._regime_specialist_shrink_weight_towards_one(
        np.asarray([2.0, 0.5, 3.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.5], dtype=np.float32),
        cfg={"lgbm_regime_specialist_distillation_power": 1.0},
    )
    assert adjusted[0] == pytest.approx(2.0)
    assert adjusted[1] == pytest.approx(1.0)
    assert adjusted[2] == pytest.approx(2.0)


def test_specialist_sample_weights_cap_bucket_masses_and_normalize():
    df = pd.DataFrame(
        {
            "regime_specialist_bucket": (
                ["current"] * 20
                + ["analogue"] * 20
                + ["normal"] * 80
                + ["irrelevant"] * 80
            ),
            "similarity_to_current": (
                [1.0] * 20
                + [0.8] * 20
                + [0.3] * 80
                + [0.05] * 80
            ),
        }
    )
    weights, diag = compute_specialist_sample_weights(
        df,
        config=SpecialistWeightConfig(
            tau_current=1.0,
            tau_analogue=1.0,
            tau_normal=1.0,
            tau_irrelevant=1.0,
        ),
    )
    assert weights.mean() == pytest.approx(1.0)
    assert diag["normal_mass"] <= 0.25 + 1e-9
    assert diag["irrelevant_mass"] <= 0.05 + 1e-9
    assert diag["current_mass"] + diag["analogue_mass"] >= 0.70 - 1e-9
    assert diag["should_train_specialist"] is True


def test_self_distillation_shrinks_towards_one_for_low_similarity():
    adjusted = shrink_self_distillation_towards_one(
        [2.0, 0.5, 3.0],
        [1.0, 0.0, 0.5],
        power=1.0,
    )
    assert adjusted[0] == pytest.approx(2.0)
    assert adjusted[1] == pytest.approx(1.0)
    assert adjusted[2] == pytest.approx(2.0)


def test_weighted_drift_baseline_uses_current_regime_weights():
    frame = pd.DataFrame(
        {
            "feature_drift_psi_core": [1.0, 2.0, 10.0],
            "feature_drift_ks_core": [0.5, 1.5, 5.0],
            "current_regime_recency_weight": [0.25, 0.75, 0.0],
        }
    )
    baseline = weighted_drift_baseline(
        frame,
        drift_columns=["feature_drift_psi_core", "feature_drift_ks_core"],
    )
    assert baseline["enabled"] is True
    assert baseline["stats"]["feature_drift_psi_core"]["weighted_mean"] == pytest.approx(1.75)
    assert baseline["stats"]["feature_drift_ks_core"]["weighted_mean"] == pytest.approx(1.25)
