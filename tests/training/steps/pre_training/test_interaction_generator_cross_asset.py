import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.interaction_templates import (
    InteractionGenerator,
)


def _build_feature(name: str, asset: str, index: pd.DatetimeIndex, values: np.ndarray):
    series = pd.Series(values, index=index, name=name)
    return SimpleNamespace(
        feature_name=name,
        feature_series=series,
        metadata={'asset': asset},
        family='trend_level_vol',
    )


def test_cross_asset_interactions_generated_when_enabled():
    config = {
        'enable_cross_asset': True,
        'cross_asset_lags': [1],
    }
    generator = InteractionGenerator(config)

    index = pd.date_range('2023-01-01', periods=20, freq='D')
    feature_a = _build_feature('asset_a_trend', 'asset_a', index, np.linspace(0.0, 1.0, len(index)))
    feature_b = _build_feature('asset_b_trend', 'asset_b', index, np.linspace(1.0, 2.0, len(index)))

    materialized_htfs = {
        feature_a.feature_name: feature_a,
        feature_b.feature_name: feature_b,
    }

    interactions = generator._generate_cross_asset_interactions(
        materialized_htfs,
        targets=None,
        budget=5,
    )

    assert interactions, "Expected at least one cross-asset interaction to be generated"
    assert any(
        set(interaction.parent_features) == {feature_a.feature_name, feature_b.feature_name}
        for interaction in interactions
    )
