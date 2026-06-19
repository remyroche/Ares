import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.feature_generator import (
    _live_training_path_sync_feature_keys,
    _synthesize_gated_feature_keys,
)
from extreme_price_movements.inference.feature_parity import FeatureParityError
from extreme_price_movements.inference.model_orchestrator import (
    ModelOrchestrator,
    _fill_live_sparse_meta_context_features,
    _strict_finite_model_matrix,
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


def test_score_for_trading_refuses_nonfinite_features_before_prediction():
    model = LGBMStabilityModel(selected_features=["ret24h"])
    X = pd.DataFrame({"ret24h": [np.inf]}, index=["AAA/USDC"])

    with pytest.raises(ValueError, match="Non-finite live features"):
        score_for_trading(model, X)


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


def test_live_sparse_meta_context_fill_is_narrow_and_column_preserving():
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

    assert out.loc["AAA/USDC", "oi_7d_chg_z"] == 0.0
    assert out.loc["AAA/USDC", "price_trend_7d_vol_norm"] == 0.0
    assert np.isnan(out.loc["BBB/USDC", "ret24h"])
    assert "missing_sparse_feature" not in out.columns
    assert {item["feature"] for item in filled} == {
        "oi_7d_chg_z",
        "price_trend_7d_vol_norm",
    }


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
