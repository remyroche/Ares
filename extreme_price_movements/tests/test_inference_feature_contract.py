import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.feature_generator import (
    _synthesize_gated_feature_keys,
)
from extreme_price_movements.inference.feature_parity import FeatureParityError
from extreme_price_movements.inference.model_orchestrator import (
    ModelOrchestrator,
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


def test_alpha_contract_uses_training_equivalent_nonfinite_adapter():
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
    assert list(aligned.columns) == ["ret24h", "range_24h_pct"]
    assert np.isfinite(aligned.to_numpy(dtype=np.float32)).all()
    assert aligned.loc["AAA/USDC", "range_24h_pct"] == 0.0
    assert aligned.loc["BBB/USDC", "ret24h"] == 0.0


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
