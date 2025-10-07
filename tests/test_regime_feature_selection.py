import importlib.util
import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "feature_generation"

feature_pkg = sys.modules.setdefault("src.feature_generation", types.ModuleType("src.feature_generation"))
feature_pkg.__path__ = [str(PACKAGE_ROOT)]

categories_pkg = sys.modules.setdefault("src.feature_generation.categories", types.ModuleType("src.feature_generation.categories"))
categories_pkg.__path__ = [str(PACKAGE_ROOT / "categories")]

utils_pkg = sys.modules.setdefault("src.feature_generation.utils", types.ModuleType("src.feature_generation.utils"))
utils_pkg.__path__ = [str(PACKAGE_ROOT / "utils")]

vectorization_stub = types.ModuleType("src.feature_generation.utils.vectorization_optimizer")
vectorization_stub.get_vectorization_optimizer = lambda *args, **kwargs: None
sys.modules.setdefault("src.feature_generation.utils.vectorization_optimizer", vectorization_stub)

pipeline_stub = types.ModuleType("src.feature_generation.utils.optimized_feature_pipeline")
pipeline_stub.get_optimized_feature_pipeline = lambda *args, **kwargs: None
sys.modules.setdefault("src.feature_generation.utils.optimized_feature_pipeline", pipeline_stub)

def _make_regime_stub(module_name: str, class_name: str) -> None:
    stub_module = types.ModuleType(module_name)

    class _StubGenerator:
        def generate_features(self, *args, **kwargs):
            return {}

    setattr(stub_module, class_name, _StubGenerator)
    sys.modules.setdefault(module_name, stub_module)


_make_regime_stub("src.feature_generation.categories.regime_volatility", "RegimeVolatilityFeatureGenerator")
_make_regime_stub("src.feature_generation.categories.regime_volume", "RegimeVolumeFeatureGenerator")
_make_regime_stub("src.feature_generation.categories.regime_structural_trend", "RegimeStructuralTrendFeatureGenerator")
_make_regime_stub("src.feature_generation.categories.regime_statistical", "RegimeStatisticalFeatureGenerator")

module_name = "src.feature_generation.categories.regime_feature_integration"
module_path = PACKAGE_ROOT / "categories" / "regime_feature_integration.py"
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
assert spec.loader is not None
spec.loader.exec_module(module)

RegimeFeatureConfig = module.RegimeFeatureConfig
RegimeFeatureIntegration = module.RegimeFeatureIntegration
generate_regime_features = module.generate_regime_features

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


def _make_market_data(rows: int = 512) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="min")
    base_price = np.linspace(100, 110, rows)
    return pd.DataFrame(
        {
            "open": base_price + 0.1,
            "high": base_price + 0.2,
            "low": base_price - 0.2,
            "close": base_price,
            "volume": np.linspace(1_000, 2_000, rows),
        },
        index=index,
    )


def _make_feature(seed: int, length: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0.5, 1.5, length)
    noise = rng.normal(0.0, 0.02, size=length)
    return base + noise


def _build_parallel_features() -> dict:
    categories = {
        "volatility": "volatility_feature",
        "volume": "volume_regime_feature",
        "structural": "structural_trend_feature",
        "statistical": "statistical_regime_feature",
        "economic": "economic_feature",
        "trading": "trading_feature",
        "quality": "regime_quality_feature",
        "misc": "misc_feature",
    }

    features = {}
    seed = 0
    for generator, prefix in categories.items():
        bucket = {}
        for i in range(30):
            bucket[f"{prefix}_{i}"] = _make_feature(seed + i)
        features[generator] = bucket
        seed += 31
    return features


def _build_market_config() -> RegimeFeatureConfig:
    return RegimeFeatureConfig(
        include_regime_quality_metrics=False,
        include_economic_significance=False,
        include_trading_viability=False,
        max_features_per_category=16,
        total_max_features=100,
        persistence_weight=0.6,
        noise_penalty_weight=0.25,
        stability_weight=0.15,
    )


@patch.object(RegimeFeatureIntegration, "_generate_regime_quality_features", return_value={})
@patch.object(RegimeFeatureIntegration, "_parallel_feature_generation")
def test_generate_regime_features_enforces_caps_and_target(mock_parallel, _):
    """The selector should respect category caps while returning 100 features."""

    mock_parallel.return_value = _build_parallel_features()
    market_data = _make_market_data()
    config = _build_market_config()

    features, summary = generate_regime_features(market_data, config=config)

    assert len(features) == config.total_max_features == 100
    quotas = summary["selection"]["category_quota"]
    for info in quotas.values():
        assert info["count"] <= info["max"]
    assert sum(info["count"] for info in quotas.values()) == len(features)

    # Validate weights are propagated for NAS/TAS tuning
    weights = summary["selection"]["weights"]
    assert pytest.approx(weights["persistence"]) == config.persistence_weight
    assert pytest.approx(weights["noise_penalty"]) == config.noise_penalty_weight
    assert pytest.approx(weights["stability"]) == config.stability_weight

    # Composite score metadata should line up with selected features
    composite_scores = summary["selection"]["composite_scores"]
    assert set(composite_scores.keys()) == set(features.keys())
    assert summary["selection"]["target"] == config.total_max_features

    # Intensity metadata should match returned features
    scalers = summary["selection"].get("intensity_scalers", {})
    assert set(scalers.keys()) == set(features.keys())
    assert all(scale > 0 for scale in scalers.values())


@patch.object(RegimeFeatureIntegration, "_generate_regime_quality_features", return_value={})
@patch.object(RegimeFeatureIntegration, "_parallel_feature_generation")
def test_generate_regime_features_ranks_top_features(mock_parallel, _):
    """Top ranked metadata should surface ordered feature scores."""

    mock_parallel.return_value = _build_parallel_features()
    market_data = _make_market_data()
    config = _build_market_config()

    features, summary = generate_regime_features(market_data, config=config)

    top_ranked = summary["selection"].get("top_ranked_features", [])
    assert top_ranked, "Expected non-empty ranking metadata"
    scores = [score for _, score in top_ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(top_ranked) <= 10
    assert all(name in features for name, _ in top_ranked)


def test_intensity_weighting_updates_scalers_and_stats():
    """Intensity weighting should scale features and persist metadata."""

    config = _build_market_config()
    generator = RegimeFeatureIntegration(config)

    features = {
        "regime_feature": np.ones(8),
        "prob_feature": np.full(8, 0.8),
    }

    quality_stats = {
        "regime_feature": {
            "persistence": 0.6,
            "noise_ratio": 1.0,
            "temporal_stability": 0.2,
        },
        "prob_feature": {
            "persistence": 0.4,
            "noise_ratio": 1.2,
            "temporal_stability": 0.25,
            "probability": 0.7,
        },
    }

    scaled_features, scalers, updated_stats = generator._apply_intensity_weighting(features, quality_stats)

    expected_regime_scale = 1.0 + config.persistence_scale * 0.6
    np.testing.assert_allclose(scaled_features["regime_feature"], features["regime_feature"] * expected_regime_scale)
    assert scalers["regime_feature"] == pytest.approx(expected_regime_scale)
    assert updated_stats["regime_feature"]["intensity_scaler"] == pytest.approx(expected_regime_scale)

    prob_multiplier = max(0.7 - 0.5, 0.0)
    expected_prob_scale = (1.0 + config.persistence_scale * 0.4) * (1.0 + config.probability_scale * prob_multiplier)
    np.testing.assert_allclose(scaled_features["prob_feature"], features["prob_feature"] * expected_prob_scale)
    assert scalers["prob_feature"] == pytest.approx(expected_prob_scale)
    assert updated_stats["prob_feature"]["probability"] == pytest.approx(0.7)
    assert updated_stats["prob_feature"]["intensity_scaler"] == pytest.approx(expected_prob_scale)
