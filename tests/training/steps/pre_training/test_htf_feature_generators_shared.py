import pandas as pd
import numpy as np

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation import (
    htf_base_features,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.phase1_probe import (  # noqa: E501
    HTFFeatureGenerator as Phase1HTFFeatureGenerator,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.htf_materialization import (  # noqa: E501
    HTFFeatureGenerator as MaterializationHTFFeatureGenerator,
    MaterializedHTF,
    UpdateStyle,
)


def _sample_data():
    index = pd.date_range("2024-01-01", periods=6, freq="min")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 105, 6),
            "high": np.linspace(101, 106, 6),
            "low": np.linspace(99, 104, 6),
            "close": np.linspace(100, 105, 6),
            "volume": np.arange(1, 7),
        },
        index=index,
    )


def test_phase1_generator_uses_shared_utilities(monkeypatch):
    generator = Phase1HTFFeatureGenerator(config={})
    data = _sample_data()

    stub_base_series = pd.Series(np.arange(len(data)), index=data.index)
    expected_series = pd.Series([10.0, 11.0], index=pd.date_range("2024-01-01", periods=2, freq="60min"))
    expected_series.name = "p/price_ema10_pct"

    get_calls = []
    resample_calls = []

    def fake_get_base_feature(feature_name):
        get_calls.append(feature_name)
        assert feature_name == "p/price_ema10_pct"
        return lambda frame: stub_base_series

    def fake_resample(series, lookback, family):
        resample_calls.append((lookback, family))
        assert series is stub_base_series
        assert lookback == 60
        assert family == "trend_level_vol"
        return expected_series

    monkeypatch.setattr(htf_base_features, "get_base_feature_func", fake_get_base_feature)
    monkeypatch.setattr(htf_base_features, "resample_to_htf", fake_resample)

    result = generator.generate_htf_feature(
        data,
        base_feature="p/price_ema10_pct",
        lookback_minutes=60,
        family="trend_level_vol",
    )

    pd.testing.assert_series_equal(result, expected_series)
    assert get_calls == ["p/price_ema10_pct"]
    assert resample_calls == [(60, "trend_level_vol")]


def test_materialization_generator_uses_shared_utilities(monkeypatch):
    generator = MaterializationHTFFeatureGenerator(config={})
    data = _sample_data()

    stub_base_series = pd.Series(np.arange(len(data)), index=data.index)
    expected_series = pd.Series([10.0, 11.0], index=pd.date_range("2024-01-01", periods=2, freq="60min"))
    expected_series.name = "p/price_ema10_pct"

    get_calls = []
    resample_calls = []

    def fake_get_base_feature(feature_name):
        get_calls.append(feature_name)
        assert feature_name == "p/price_ema10_pct"
        return lambda frame: stub_base_series

    def fake_resample(series, lookback, family):
        resample_calls.append((lookback, family))
        assert series is stub_base_series
        assert lookback == 60
        assert family == "trend_level_vol"
        return expected_series

    class DummyTransformRouter:
        def __init__(self, config):
            self.config = config

        def fit_transform(self, train_df, _):
            assert train_df.columns.tolist() == ["p/price_ema10_pct"]
            pd.testing.assert_series_equal(train_df["p/price_ema10_pct"], expected_series)
            return {"p/price_ema10_pct": {"train": expected_series}}

    monkeypatch.setattr(htf_base_features, "get_base_feature_func", fake_get_base_feature)
    monkeypatch.setattr(htf_base_features, "resample_to_htf", fake_resample)
    monkeypatch.setattr(
        MaterializationHTFFeatureGenerator,
        "_create_transform_router",
        lambda self, feature_names: DummyTransformRouter({"features": feature_names}),
    )

    result = generator.generate_htf_feature(
        data=data,
        feature_name="p/price_ema10_pct",
        family="trend_level_vol",
        lookback=60,
        update_style=UpdateStyle.EHU,
    )

    assert isinstance(result, MaterializedHTF)
    assert result.feature_name == "t/p/price_ema10_pct_htf60/ewz12"
    pd.testing.assert_series_equal(result.feature_series, expected_series)
    assert get_calls == ["p/price_ema10_pct"]
    assert resample_calls == [(60, "trend_level_vol")]
