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
