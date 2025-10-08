import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[4] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


label_balancing = _load_module(
    "profit_labeling_label_balancing",
    "src/training/steps/pre_training/profit_labeling/label_balancing.py",
)
bar_construction = _load_module(
    "profit_labeling_bar_construction",
    "src/training/steps/pre_training/profit_labeling/bar_construction.py",
)

SampleWeighter = label_balancing.SampleWeighter
WeightingConfig = label_balancing.WeightingConfig
WeightingScheme = label_balancing.WeightingScheme
BarConstructionConfig = bar_construction.BarConstructionConfig
BarType = bar_construction.BarType
EventBasedBarConstructor = bar_construction.EventBasedBarConstructor

# Disable optional integrations that require unavailable dependencies during tests
bar_construction.MATRIX_OPS_AVAILABLE = False
bar_construction.HARDWARE_OPTIMIZATION_AVAILABLE = False


class _InspectingBarConstructor(EventBasedBarConstructor):
    """Capture intermediate bar data for testing trailing behaviour."""

    def __init__(self, config: BarConstructionConfig):
        super().__init__(config)
        self.captured_bars = []
        # Ensure we exercise the fallback path where bars are built row-by-row
        self.matrix_ops = None

    def _create_single_bar(self, bar_data: pd.DataFrame):
        self.captured_bars.append(bar_data.copy())
        return {
            "timestamp": bar_data.index[-1],
            "volume": bar_data["volume"].sum(),
            "ticks": len(bar_data),
        }


def test_confidence_weights_ignore_future_mutations():
    index = pd.RangeIndex(5)
    y = pd.Series([0, 1, 0, 1, 0], index=index)
    confidence = pd.Series([0.5, 0.6, 10.0, 0.7, 12.0], index=index)

    config = WeightingConfig(
        weighting_scheme=WeightingScheme.CONFIDENCE,
        confidence_scale=1.0,
        confidence_smoothing=0.1,
    )

    weighter = SampleWeighter(config)

    base_weights = weighter._compute_confidence_weights(y, {"confidence": confidence})

    mutated_confidence = confidence.copy()
    mutated_confidence.iloc[2] = 8.0  # Future element relative to index 1
    mutated_weights = weighter._compute_confidence_weights(
        y, {"confidence": mutated_confidence}
    )

    pd.testing.assert_series_equal(base_weights.iloc[:2], mutated_weights.iloc[:2])


def test_uncertainty_weights_ignore_future_mutations():
    index = pd.RangeIndex(5)
    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 20.0, 4.0, 5.0],
        },
        index=index,
    )
    y = pd.Series([0, 1, 0, 1, 0], index=index)

    config = WeightingConfig(weighting_scheme=WeightingScheme.CONFIDENCE)
    weighter = SampleWeighter(config)

    base_weights = weighter._compute_uncertainty_weights(X, y)

    mutated_X = X.copy()
    mutated_X.loc[4, "feature"] = 50.0  # Mutate future observation
    mutated_weights = weighter._compute_uncertainty_weights(mutated_X, y)

    pd.testing.assert_series_equal(base_weights.iloc[:3], mutated_weights.iloc[:3])


def test_information_content_smoothing_is_trailing():
    index = pd.RangeIndex(5)
    X = pd.DataFrame(
        {
            "close": [10.0, 10.5, 11.0, 11.5, 12.0],
            "feature": [1.0, 2.0, 5.0, 2.5, 2.0],
            "regime": [0, 0, 0, 0, 0],
        },
        index=index,
    )
    y = pd.Series([0, 1, 0, 1, 0], index=index)

    config = WeightingConfig(
        weighting_scheme=WeightingScheme.INFORMATION_CONTENT,
        weight_smoothing=1.0,
        weight_normalization="none",
        information_volatility_weight=0.0,
        information_entropy_weight=0.0,
        information_uncertainty_weight=1.0,
        information_regime_weight=0.0,
        volatility_window=2,
        overlap_window=2,
        time_decay_half_life=1,
    )

    weighter = SampleWeighter(config)

    ones_series = pd.Series(1.0, index=index)
    weighter._compute_volatility_weights = lambda *args, **kwargs: ones_series.copy()
    weighter._compute_event_overlap_weights = lambda *args, **kwargs: ones_series.copy()
    weighter._compute_time_decay_weights = lambda *args, **kwargs: ones_series.copy()
    weighter._compute_regime_aware_weights = lambda *args, **kwargs: ones_series.copy()
    weighter._compute_entropy_weights = lambda *args, **kwargs: ones_series.copy()

    base_weights = pd.Series(
        weighter._compute_information_content_weights(X, y), index=index
    )

    mutated_X = X.copy()
    mutated_X.loc[2, "feature"] = 50.0  # Only affects future relative to indices 0 and 1
    mutated_weights = pd.Series(
        weighter._compute_information_content_weights(mutated_X, y), index=index
    )

    pd.testing.assert_series_equal(base_weights.iloc[:2], mutated_weights.iloc[:2])


def test_dollar_bar_median_volume_is_trailing():
    config = BarConstructionConfig(bar_type=BarType.DOLLAR, bar_size=4)
    constructor = _InspectingBarConstructor(config)

    periods = 30
    index = pd.RangeIndex(periods)

    base_volumes = [1.0, 5.0, 1.0] + [1.0] * (periods - 3)
    market_data = pd.DataFrame(
        {
            "open": [100.0] * periods,
            "high": [100.0] * periods,
            "low": [99.0] * periods,
            "close": [100.0] * periods,
            "volume": base_volumes,
        },
        index=index,
    )

    constructor._create_dollar_bars(market_data)
    base_effective = constructor.captured_bars[0]["effective_volume"].iloc[:2]

    mutated_market_data = market_data.copy()
    mutated_market_data.iloc[2, mutated_market_data.columns.get_loc("volume")] = 100.0

    constructor.captured_bars = []
    constructor._create_dollar_bars(mutated_market_data)
    mutated_effective = constructor.captured_bars[0]["effective_volume"].iloc[:2]

    pd.testing.assert_series_equal(base_effective, mutated_effective)
