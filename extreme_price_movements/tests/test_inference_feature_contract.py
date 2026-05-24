import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.feature_parity import FeatureParityError
from extreme_price_movements.inference.model_orchestrator import _strict_finite_model_matrix
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
