import asyncio

import numpy as np

from src.training.steps.pre_training.pid_based_feature_generation.cross_timeframe_feature_generator import (
    CrossTimeframeConfig,
    CrossTimeframeFeatureGenerator,
)


def test_synthetic_timeframe_features_generated_for_basic_columns():
    """Ensure synthetic timeframe features are generated for single-token inputs."""

    config = CrossTimeframeConfig(
        max_timeframe_pairs=5,
        max_cross_timeframe_features=15,
        max_correlation_threshold=0.999,
    )
    generator = CrossTimeframeFeatureGenerator(config=config)

    timesteps = 120
    rng = np.random.default_rng(42)
    time_index = np.arange(timesteps)

    close = np.sin(time_index / 5.0) + rng.normal(scale=0.25, size=timesteps)
    volume = np.cos(time_index / 7.0) + rng.normal(scale=0.35, size=timesteps)

    X = np.column_stack([close, volume])
    feature_names = ["close", "volume"]

    result = asyncio.run(
        generator.generate_cross_timeframe_features(X, feature_names)
    )

    assert result.total_features_generated > 0
    assert any(
        indicator in name
        for name in result.feature_names
        for indicator in ("short_tf", "medium_tf", "long_tf")
    )


def test_optimized_lookback_periods_transform_and_update_metadata():
    """Optimized lookback windows should update both data and feature names."""

    config = CrossTimeframeConfig()
    generator = CrossTimeframeFeatureGenerator(config=config)

    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
    )
    feature_names = ["feat_a", "feat_b"]
    optimized_periods = {
        "feat_a": 3,
        "feat_b": 0,  # Invalid period should fall back to original column
        "feat_missing": 5,
    }

    transformed_X, transformed_names = generator._apply_optimized_lookback_periods(
        X,
        feature_names,
        optimized_periods,
    )

    expected_feat_a = np.array([1.0, 1.5, 2.0, 3.0, 4.0])

    assert transformed_X.shape == X.shape
    assert transformed_names == ["feat_a_lb3", "feat_b"]
    np.testing.assert_allclose(transformed_X[:, 0], expected_feat_a, rtol=1e-6)
    np.testing.assert_array_equal(transformed_X[:, 1], X[:, 1])
