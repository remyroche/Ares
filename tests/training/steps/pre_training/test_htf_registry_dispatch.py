import pandas as pd
from types import SimpleNamespace

from src.feature_engineering.feature_registry import (
    FeatureRegistry,
    PriceReturnsFeatures,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.phase1_probe import (  # noqa: E501
    HTFFeatureGenerator as Phase1HTFFeatureGenerator,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_materialization import (  # noqa: E501
    HTFFeatureGenerator as MaterializationHTFFeatureGenerator,
    UpdateStyle,
)


def _build_price_frame(rows: int = 180) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="1min")
    base = pd.Series(range(rows), index=index, dtype=float)
    data = pd.DataFrame(
        {
            "open": base + 0.5,
            "high": base + 1.0,
            "low": base,
            "close": base + 0.75,
            "volume": 1_000 + base * 2,
        }
    )
    return data


def test_feature_registry_dispatch_matches_static_implementation():
    registry = FeatureRegistry()
    frame = _build_price_frame()

    expected = PriceReturnsFeatures.price_ema10_pct(frame)
    result = registry.compute_feature("p/price_ema10_pct", frame)

    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_phase1_and_materialization_use_shared_dispatch():
    frame = _build_price_frame()

    config = SimpleNamespace()
    phase1_generator = Phase1HTFFeatureGenerator(config)
    materialization_generator = MaterializationHTFFeatureGenerator(config)

    feature_name = "p/price_ema10_pct"
    lookback = 60
    family = "trend_level_vol"

    phase1_series = phase1_generator.generate_htf_feature(
        frame, feature_name, lookback, family
    )

    materialized = materialization_generator.generate_htf_feature(
        frame,
        feature_name,
        family,
        lookback,
        UpdateStyle.RIH,
    )

    pd.testing.assert_series_equal(
        phase1_series.dropna(), materialized.feature_series.dropna()
    )
    assert materialized.transform_applied == "ewz12"
    assert phase1_series.name == materialized.feature_series.name
