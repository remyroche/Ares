import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.feature_generator import (
    get_inference_required_feature_keys,
    _latest_matrix_low_finite_repair_incidents,
    _live_training_path_sync_feature_keys,
    _synthesize_gated_feature_keys,
)
from extreme_price_movements.inference.feature_parity import FeatureParityError
from extreme_price_movements.inference.model_orchestrator import (
    ModelOrchestrator,
    _fill_optional_generated_model_features,
    _fill_live_sparse_meta_context_features,
    _strict_finite_model_matrix,
)
from extreme_price_movements.inference.run_inference import (
    _sparse_selected_feature_source_attribution,
)
from extreme_price_movements.lgbm_pipeline import LGBMStabilityModel, score_for_trading


def test_lgbm_inference_contract_refuses_missing_features():
    model = LGBMStabilityModel(selected_features=["ret24h", "range_24h_pct"])
    X = pd.DataFrame({"ret24h": [0.1]}, index=["AAA/USDC"])

    with pytest.raises(ValueError, match="contracted features are missing"):
        model.predict(X)


def test_lgbm_inference_contract_refuses_nonfinite_features():
    model = LGBMStabilityModel(selected_features=["ret24h", "range_24h_pct"])
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [0.02, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )

    diagnostics = model.inference_schema_diagnostics(X)
    assert diagnostics["nonfinite_selected_features_count"] == 1

    with pytest.raises(ValueError, match="non-finite contracted features"):
        model.predict(X)


def test_lgbm_native_missing_inference_preserves_nan_features():
    model = LGBMStabilityModel(
        selected_features=["ret24h", "range_24h_pct"],
        allow_native_missing_at_inference=True,
    )
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [0.02, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )

    frame = model._frame(X)

    assert list(frame.columns) == ["ret24h", "range_24h_pct"]
    assert np.isnan(frame.loc["BBB/USDC", "ret24h"])


def test_lgbm_native_missing_inference_rejects_infinite_features():
    model = LGBMStabilityModel(
        selected_features=["ret24h", "range_24h_pct"],
        allow_native_missing_at_inference=True,
    )
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [np.inf, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )

    with pytest.raises(ValueError, match="infinite contracted features"):
        model._frame(X)


def test_score_for_trading_refuses_nonfinite_features_before_prediction():
    model = LGBMStabilityModel(selected_features=["ret24h"])
    X = pd.DataFrame({"ret24h": [np.inf]}, index=["AAA/USDC"])

    with pytest.raises(ValueError, match="Non-finite live features"):
        score_for_trading(model, X)


def test_lgbm_optional_generated_features_are_neutral_not_required():
    model = LGBMStabilityModel(
        selected_features=["ret24h", "dae_b16_00", "gmm_prob_0", "cluster_entropy_norm"]
    )
    X = pd.DataFrame({"ret24h": [0.1]}, index=["AAA/USDC"])

    diagnostics = model.inference_schema_diagnostics(X)
    assert diagnostics["missing_selected_features_count"] == 0
    assert diagnostics["missing_optional_generated_features_count"] == 3

    frame = model._frame(X)

    assert list(frame.columns) == model.selected_features
    assert frame.loc["AAA/USDC", "ret24h"] == pytest.approx(0.1)
    assert frame.loc["AAA/USDC", "dae_b16_00"] == 0.0
    assert frame.loc["AAA/USDC", "gmm_prob_0"] == 0.0
    assert frame.loc["AAA/USDC", "cluster_entropy_norm"] == 0.0


def test_lgbm_optional_generated_features_do_not_hide_core_gaps():
    model = LGBMStabilityModel(selected_features=["ret24h", "dae_b16_00"])
    X = pd.DataFrame({"dae_b16_00": [0.2]}, index=["AAA/USDC"])

    diagnostics = model.inference_schema_diagnostics(X)
    assert diagnostics["missing_selected_features_count"] == 1
    assert diagnostics["missing_selected_features_preview"] == ["ret24h"]

    with pytest.raises(ValueError, match="contracted features are missing"):
        model._frame(X)


def test_strict_final_matrix_refuses_nonfinite_without_dropping_rows():
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [0.02, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )

    with pytest.raises(FeatureParityError, match="non-finite"):
        _strict_finite_model_matrix(
            X,
            model_feature_cols=["ret24h", "range_24h_pct"],
            model_key="alpha",
        )


def test_alpha_contract_strict_mode_rejects_nonfinite_adapter_input():
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [np.inf, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )
    orchestrator = ModelOrchestrator({}, {"strict_feature_parity": True})

    aligned = orchestrator._align_alpha_feature_contract(
        X,
        ["ret24h", "range_24h_pct"],
    )

    assert list(aligned.index) == ["AAA/USDC", "BBB/USDC"]
    assert aligned.empty


def test_alpha_contract_legacy_neutral_fill_adapter_is_explicit():
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [np.inf, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )
    orchestrator = ModelOrchestrator(
        {},
        {
            "strict_feature_parity": True,
            "strict_feature_parity_neutral_fill_nonfinite": True,
        },
    )

    aligned = orchestrator._align_alpha_feature_contract(
        X,
        ["ret24h", "range_24h_pct"],
    )

    assert list(aligned.index) == ["AAA/USDC", "BBB/USDC"]
    assert list(aligned.columns) == ["ret24h", "range_24h_pct"]
    assert np.isfinite(aligned.to_numpy(dtype=np.float32)).all()
    assert aligned.loc["AAA/USDC", "range_24h_pct"] == 0.0
    assert aligned.loc["BBB/USDC", "ret24h"] == 0.0


def test_alpha_contract_native_lgbm_missing_preserves_nan_inputs():
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [0.02, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )
    orchestrator = ModelOrchestrator(
        {},
        {
            "strict_feature_parity": True,
            "simple_policy_allow_lgbm_native_missing": True,
        },
    )

    aligned = orchestrator._align_alpha_feature_contract(
        X,
        ["ret24h", "range_24h_pct"],
    )

    assert list(aligned.index) == ["AAA/USDC", "BBB/USDC"]
    assert list(aligned.columns) == ["ret24h", "range_24h_pct"]
    assert np.isnan(aligned.loc["BBB/USDC", "ret24h"])
    assert aligned.loc["AAA/USDC", "range_24h_pct"] == pytest.approx(0.02)


def test_alpha_contract_native_lgbm_missing_still_blocks_infinite_rows():
    X = pd.DataFrame(
        {"ret24h": [0.1, np.nan], "range_24h_pct": [np.inf, 0.03]},
        index=["AAA/USDC", "BBB/USDC"],
    )
    orchestrator = ModelOrchestrator(
        {},
        {
            "strict_feature_parity": True,
            "simple_policy_allow_lgbm_native_missing": True,
        },
    )

    aligned = orchestrator._align_alpha_feature_contract(
        X,
        ["ret24h", "range_24h_pct"],
    )

    assert list(aligned.index) == ["BBB/USDC"]
    assert list(aligned.columns) == ["ret24h", "range_24h_pct"]
    assert np.isnan(aligned.loc["BBB/USDC", "ret24h"])


def test_meta_optional_generated_features_are_neutral_before_strict_validation():
    X = pd.DataFrame(
        {"ret24h": [0.1], "dae_b16_00": [np.inf]},
        index=["AAA/USDC"],
    )

    filled, added, repaired = _fill_optional_generated_model_features(
        X,
        model_feature_cols=[
            "ret24h",
            "dae_b16_00",
            "gmm_prob_0",
            "cluster_entropy_norm",
        ],
    )

    assert added == ["gmm_prob_0", "cluster_entropy_norm"]
    assert repaired == ["dae_b16_00"]
    assert filled.loc["AAA/USDC", "dae_b16_00"] == 0.0
    assert filled.loc["AAA/USDC", "gmm_prob_0"] == 0.0
    assert filled.loc["AAA/USDC", "cluster_entropy_norm"] == 0.0

    strict = _strict_finite_model_matrix(
        filled.reindex(columns=["ret24h", "dae_b16_00", "gmm_prob_0"]),
        model_feature_cols=["ret24h", "dae_b16_00", "gmm_prob_0"],
        model_key="meta",
    )
    assert np.isfinite(strict.to_numpy(dtype=np.float32)).all()


def test_inference_required_keys_exclude_optional_generated_representations():
    class DummyMeta:
        selected_features = [
            "ret24h",
            "dae_b16_00",
            "meta_dae_b16_00",
            "gmm_prob_0",
            "meta_gmm_prob_0",
            "cluster_entropy_norm",
            "meta_cluster_entropy_norm",
            "pred_demo_raw_state_svd_03",
        ]

    required = get_inference_required_feature_keys(
        {"bundle": {"meta_models": {"long_demo": DummyMeta()}}},
    )

    assert "ret24h" in required
    assert "dae_b16_00" not in required
    assert "meta_dae_b16_00" not in required
    assert "gmm_prob_0" not in required
    assert "meta_gmm_prob_0" not in required
    assert "cluster_entropy_norm" not in required
    assert "meta_cluster_entropy_norm" not in required
    assert "pred_demo_raw_state_svd_03" not in required


def test_live_sparse_meta_context_features_remain_strict():
    X = pd.DataFrame(
        {
            "oi_7d_chg_z": [np.nan, 1.25],
            "price_trend_7d_vol_norm": [np.inf, -0.3],
            "ret24h": [0.1, np.nan],
        },
        index=["AAA/USDC", "BBB/USDC"],
    )

    out, filled = _fill_live_sparse_meta_context_features(
        X,
        [
            "oi_7d_chg_z",
            "price_trend_7d_vol_norm",
            "ret24h",
            "missing_sparse_feature",
        ],
    )

    assert np.isnan(out.loc["AAA/USDC", "oi_7d_chg_z"])
    assert np.isinf(out.loc["AAA/USDC", "price_trend_7d_vol_norm"])
    assert np.isnan(out.loc["BBB/USDC", "ret24h"])
    assert "missing_sparse_feature" not in out.columns
    assert filled == []


def test_gated_selected_feature_missing_base_materializes_nan_frame():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    panel = {
        "close": pd.DataFrame(
            {
                "AAA/USD:USD": np.linspace(100.0, 110.0, len(idx)),
                "BBB/USD:USD": np.linspace(50.0, 48.0, len(idx)),
            },
            index=idx,
        )
    }

    feats = _synthesize_gated_feature_keys(
        {},
        panel,
        ["AAA/USD:USD", "BBB/USD:USD"],
        {"atr_percentile_G_VOL_0"},
    )

    out = feats["atr_percentile_G_VOL_0"]
    assert list(out.columns) == ["AAA/USD:USD", "BBB/USD:USD"]
    assert out.index.equals(idx)
    assert out.isna().all().all()


def test_live_training_path_sync_skips_deterministic_live_synthesized_keys():
    sync_keys, skipped = _live_training_path_sync_feature_keys(
        [
            "ret1h_G_VOL_0",
            "ret1h_G_VOL_1",
            "barrier_pct",
            "ret12h",
            "lr_12h",
            "oi_3d_chg_z",
        ],
        {"live_model_feature_store_strict": True},
    )

    assert sync_keys == ["lr_12h", "oi_3d_chg_z", "ret12h"]
    assert skipped == [
        "barrier_pct",
        "ret1h_G_VOL_0",
        "ret1h_G_VOL_1",
    ]


def test_live_training_path_sync_can_skip_source_derived_keys_when_not_strict():
    sync_keys, skipped = _live_training_path_sync_feature_keys(
        [
            "distance_to_resistance_daily_donchian_atr",
            "mom_slow",
            "oi_3d_chg_z",
            "ret12h",
        ],
        {"live_model_feature_store_strict": False},
    )

    assert sync_keys == ["ret12h"]
    assert skipped == [
        "distance_to_resistance_daily_donchian_atr",
        "mom_slow",
        "oi_3d_chg_z",
    ]


def test_latest_matrix_low_finite_exempts_history_dependent_features():
    low_finite = [
        {"feature": "efficiency_ratio_20", "finite": 10, "total": 20},
        {"feature": "ker_16", "finite": 10, "total": 20},
        {"feature": "lr_24h", "finite": 10, "total": 20},
        {"feature": "oi_1d_x_funding", "finite": 10, "total": 20},
        {"feature": "cs_rank_oi_chg_1d_z_90d", "finite": 10, "total": 20},
    ]

    incidents = _latest_matrix_low_finite_repair_incidents(low_finite)

    assert incidents == []


@pytest.mark.parametrize(
    ("feature", "expected_hours"),
    [
        ("efficiency_ratio_20", 20.0),
        ("ker_16", 16.0),
        ("lr_24h", 24.0),
    ],
)
def test_latest_matrix_history_dependent_sparsity_is_attributed_to_symbol_history(
    feature, expected_hours
):
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    symbols = [f"S{i}/USD:USD" for i in range(20)]
    close = pd.DataFrame(
        np.linspace(100.0, 110.0, len(idx) * len(symbols)).reshape(
            len(idx), len(symbols)
        ),
        index=idx,
        columns=symbols,
    )
    panel = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
    }

    attribution = _sparse_selected_feature_source_attribution(
        key=feature,
        panel=panel,
        symbols=symbols,
        signal_bar_ts=idx[-1],
        reason="low_finite",
        finite=10,
        total=20,
        missing_symbols_count=0,
        latest_feature_ts=idx[-1],
        stale_hours=0.0,
    )

    assert attribution["source_attribution"] == "insufficient_symbol_history"
    assert attribution["required_history_hours"] == pytest.approx(expected_hours)


def test_gated_selected_feature_overwrites_nan_placeholder_from_base():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    cols = ["AAA/USD:USD", "BBB/USD:USD"]
    close = pd.DataFrame(
        {
            cols[0]: np.linspace(100.0, 110.0, len(idx)),
            cols[1]: np.linspace(50.0, 48.0, len(idx)),
        },
        index=idx,
    )
    base = close.pct_change().astype(np.float32)
    nan_gate = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=np.float32)

    feats = _synthesize_gated_feature_keys(
        {
            "ret1h": base,
            "ret1h_G_VOL_0": nan_gate,
            "ret1h_G_VOL_1": nan_gate,
        },
        {"close": close},
        cols,
        {"ret1h_G_VOL_0", "ret1h_G_VOL_1"},
    )

    combined = feats["ret1h_G_VOL_0"] + feats["ret1h_G_VOL_1"]
    assert np.isfinite(feats["ret1h_G_VOL_0"].tail(1).to_numpy()).any()
    assert np.isfinite(feats["ret1h_G_VOL_1"].tail(1).to_numpy()).any()
    np.testing.assert_allclose(
        combined.tail(24).to_numpy(dtype=np.float32),
        base.tail(24).to_numpy(dtype=np.float32),
        rtol=1e-6,
        atol=1e-8,
    )


def test_gated_return_feature_synthesizes_missing_return_base_from_close():
    idx = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    cols = ["AAA/USD:USD", "BBB/USD:USD"]
    close = pd.DataFrame(
        {
            cols[0]: np.linspace(100.0, 110.0, len(idx)),
            cols[1]: np.linspace(50.0, 48.0, len(idx)),
        },
        index=idx,
    )
    expected = close.pct_change().astype(np.float32)

    feats = _synthesize_gated_feature_keys(
        {},
        {"close": close},
        cols,
        {"ret1h_G_VOL_0", "ret1h_G_VOL_1"},
    )

    combined = feats["ret1h_G_VOL_0"] + feats["ret1h_G_VOL_1"]
    assert np.isfinite(combined.tail(1).to_numpy()).all()
    np.testing.assert_allclose(
        combined.tail(24).to_numpy(dtype=np.float32),
        expected.tail(24).to_numpy(dtype=np.float32),
        rtol=1e-6,
        atol=1e-8,
    )


def test_alpha_contract_overwrites_nan_gated_feature_when_base_and_gate_exist():
    X = pd.DataFrame(
        {
            "ret1h": [0.01, -0.02],
            "G_VOL": [1.0, 0.0],
            "ret1h_G_VOL_1": [np.nan, np.nan],
        },
        index=["AAA/USDC", "BBB/USDC"],
    )
    orchestrator = ModelOrchestrator({}, {"strict_feature_parity": True})

    aligned = orchestrator._align_alpha_feature_contract(
        X,
        ["ret1h", "G_VOL", "ret1h_G_VOL_1"],
    )

    assert aligned.loc["AAA/USDC", "ret1h_G_VOL_1"] == pytest.approx(0.01)
    assert aligned.loc["BBB/USDC", "ret1h_G_VOL_1"] == pytest.approx(-0.0)
