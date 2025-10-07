import pandas as pd

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.interaction_templates import (
    HTFInteractionTemplates,
)


def test_core_interaction_generated_from_ohlcv_dataframe():
    """Ensure that OHLCV inputs surface at least one core interaction."""
    config = {}
    templates = HTFInteractionTemplates(config)

    index = pd.date_range("2024-01-01", periods=10, freq="min")
    base_features = pd.DataFrame(
        {
            "open": pd.Series(range(10), index=index),
            "high": pd.Series(range(1, 11), index=index),
            "low": pd.Series(range(10), index=index),
            "close": pd.Series(range(2, 12), index=index),
            "volume": pd.Series(range(10, 20), index=index),
        }
    )

    interactions = templates.generate_interactions({}, base_features, targets=None)

    assert any(inter.interaction_type == "core" for inter in interactions), (
        "Expected at least one core interaction when OHLCV columns are available."
    )
