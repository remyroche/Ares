import numpy as np
import sys
import types

import numpy as np
import pandas as pd


def _ensure_stubbed_module(module_name: str, *attributes: str) -> None:
    """Ensure a lightweight stub module exists for the given attributes."""
    if module_name in sys.modules:
        return

    stub_module = types.ModuleType(module_name)

    class _Stub:  # pragma: no cover - simple placeholder
        def __init__(self, *args, **kwargs):
            pass

    if not attributes:
        attributes = ("Stub",)

    for attribute in attributes:
        setattr(stub_module, attribute, _Stub)
    sys.modules[module_name] = stub_module


_ensure_stubbed_module("src.utils.data.klines_parquet", "KlineParquetManager")
_ensure_stubbed_module("src.utils.data.unified_data_utils", "UnifiedDataUtils")
_ensure_stubbed_module(
    "src.utils.ml_common.optimization.bayesian_tpe_optimizer",
    "BayesianTPEOptimizer",
)
_ensure_stubbed_module(
    "src.utils.ml_common.optimization.pareto",
    "ParetoOptimizer",
    "ParetoFront",
    "Solution",
)
_ensure_stubbed_module(
    "src.utils.ml_common.validation.cross_validation",
    "CrossValidator",
)
_ensure_stubbed_module(
    "src.utils.ml_common.ensembles.oof_stacking_ensemble_manager",
    "OOFStackingEnsembleManager",
)
_ensure_stubbed_module("src.utils.matrix_operations", "UnifiedMatrixOperations")


def _register_profit_labeling_stubs() -> None:
    """Register lightweight stubs for heavy profit labeling modules."""
    base = "src.training.steps.pre_training.profit_labeling"

    bar_module_name = f"{base}.bar_construction"
    if bar_module_name not in sys.modules:
        bar_module = types.ModuleType(bar_module_name)

        class BarConstructionConfig:
            def __init__(self, *args, **kwargs):
                pass

            def _validate_config(self) -> None:
                pass

        class EventBasedBarConstructor:
            def __init__(self, config=None):
                self.config = config

        class BarConstructionResult:
            pass

        bar_module.BarConstructionConfig = BarConstructionConfig
        bar_module.EventBasedBarConstructor = EventBasedBarConstructor
        bar_module.BarConstructionResult = BarConstructionResult
        sys.modules[bar_module_name] = bar_module

    noise_module_name = f"{base}.noise_gating"
    if noise_module_name not in sys.modules:
        noise_module = types.ModuleType(noise_module_name)

        class NoiseGatingConfig:
            def __init__(self, *args, **kwargs):
                pass

            def _validate_config(self) -> None:
                pass

        class NoiseGatingFilter:
            def __init__(self, config=None):
                self.config = config

        class EligibilityResult:
            pass

        noise_module.NoiseGatingConfig = NoiseGatingConfig
        noise_module.NoiseGatingFilter = NoiseGatingFilter
        noise_module.EligibilityResult = EligibilityResult
        sys.modules[noise_module_name] = noise_module

    quality_module_name = f"{base}.quality_scoring"
    if quality_module_name not in sys.modules:
        quality_module = types.ModuleType(quality_module_name)

        class QualityScoringConfig:
            def __init__(self, *args, **kwargs):
                pass

            def _validate_config(self) -> None:
                pass

        class LabelQualityScorer:
            def __init__(self, config=None):
                self.config = config

        class QualityMetrics:
            pass

        quality_module.QualityScoringConfig = QualityScoringConfig
        quality_module.LabelQualityScorer = LabelQualityScorer
        quality_module.QualityMetrics = QualityMetrics
        sys.modules[quality_module_name] = quality_module

    multi_module_name = f"{base}.multi_target_scheme"
    if multi_module_name not in sys.modules:
        multi_module = types.ModuleType(multi_module_name)

        class MultiTargetConfig:
            def __init__(self, *args, **kwargs):
                pass

            def _validate_config(self) -> None:
                pass

        class MultiTargetScheme:
            def __init__(self, config=None):
                self.config = config

        class TargetSelectionResult:
            pass

        class BandHorizonRule:
            pass

        multi_module.MultiTargetConfig = MultiTargetConfig
        multi_module.MultiTargetScheme = MultiTargetScheme
        multi_module.TargetSelectionResult = TargetSelectionResult
        multi_module.BandHorizonRule = BandHorizonRule
        sys.modules[multi_module_name] = multi_module

    enhanced_module_name = f"{base}.enhanced_label_definitions"
    if enhanced_module_name not in sys.modules:
        enhanced_module = types.ModuleType(enhanced_module_name)

        class EnhancedLabelDefinitions:
            def __init__(self, *args, **kwargs):
                pass

        class LabelDefinitionType:
            ANALYST = "analyst"

        enhanced_module.EnhancedLabelDefinitions = EnhancedLabelDefinitions
        enhanced_module.LabelDefinitionType = LabelDefinitionType
        enhanced_module.AnalystLabelConfig = object
        enhanced_module.TacticianLabelConfig = object
        enhanced_module.RegimeConditionedConfig = object
        enhanced_module.RiskAwareConfig = object
        enhanced_module.DataCleaningConfig = object
        enhanced_module.StabilityCheckConfig = object
        enhanced_module.TradingCosts = object
        enhanced_module.create_trading_aware_config = lambda *args, **kwargs: None
        sys.modules[enhanced_module_name] = enhanced_module


_register_profit_labeling_stubs()

from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareConfig,
    VolatilityAwareMultiHorizonLabeler,
)


class _DummyTargetResult:
    def __init__(self, raw_payoffs: pd.DataFrame, target_parameters: dict | None = None):
        self.raw_payoffs = raw_payoffs
        self.target_parameters = target_parameters or {}
        self.sigma_payoffs = pd.DataFrame()
        self.training_labels = pd.DataFrame()


def test_sigma_normalization_respects_horizon_shifts_and_embargo():
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    raw_payoffs = pd.DataFrame(
        {
            "target_small": np.linspace(0.1, 0.8, len(index)),
            "target_large": np.linspace(0.2, 1.6, len(index)),
        },
        index=index,
    )
    volatility_series = pd.Series(np.linspace(1.0, 2.4, len(index)), index=index)

    target_parameters = {
        "target_small": {"horizon": 2},
        "target_large": {"horizon": 3},
    }

    temporal_config = _TemporalValidationStub(
        enable_temporal_validation=True,
        enable_purging=True,
        purge_window_hours=1,
        embargo_window_hours=1,
    )

    config = VolatilityAwareConfig(
        enable_enhanced_labels=False,
        enable_quality_scoring=False,
        temporal_validation=temporal_config,
    )

    labeler = VolatilityAwareMultiHorizonLabeler(config)
    target_result = _DummyTargetResult(raw_payoffs.copy(), target_parameters)

    labeler._ensure_sigma_normalization(target_result, volatility_series)

    expected = pd.DataFrame(index=index, columns=raw_payoffs.columns, dtype=float)
    for column, params in target_parameters.items():
        shift = int(params["horizon"])
        expected[column] = raw_payoffs[column] / volatility_series.shift(shift)

    valid_index = index[1:-1]
    expected = expected.loc[valid_index].reindex(index)

    expected = expected.replace([np.inf, -np.inf], np.nan)

    pd.testing.assert_frame_equal(target_result.sigma_payoffs, expected)

    dropped_index = index.difference(valid_index)
    pd.testing.assert_index_equal(
        target_result.raw_payoffs.index,
        raw_payoffs.index,
    )
    assert target_result.raw_payoffs.loc[dropped_index].isna().all().all()
class _TemporalValidationStub:
    def __init__(
        self,
        enable_temporal_validation: bool = True,
        enable_purging: bool = True,
        purge_window_hours: int = 0,
        embargo_window_hours: int = 0,
    ) -> None:
        self.enable_temporal_validation = enable_temporal_validation
        self.enable_purging = enable_purging
        self.purge_window_hours = purge_window_hours
        self.embargo_window_hours = embargo_window_hours

