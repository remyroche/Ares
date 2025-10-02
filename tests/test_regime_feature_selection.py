import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.feature_generation.categories.regime_feature_integration import (
    generate_regime_features,
    RegimeFeatureConfig,
    RegimeFeatureIntegration,
)


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
