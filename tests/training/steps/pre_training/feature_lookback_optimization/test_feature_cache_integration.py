import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path
from enum import Enum
from typing import Optional

import pandas as pd
import pytest


class _DummyFeatureBank:
    VERSION = "test-version"
    call_count = 0

    def __init__(self):
        type(self).call_count += 1

    def generate_features(self, data, categories=None):
        frame = pd.DataFrame(
            {
                "RETURNS_alpha": pd.Series(range(len(data)), index=data.index),
                "MOMENTUM_beta": pd.Series(range(len(data)), index=data.index),
            }
        )
        return frame


_ROOT = Path(__file__).resolve().parents[5]
_FEATURE_CACHE_PATH = _ROOT / "src" / "feature_generation" / "core" / "feature_cache.py"
_FEATURE_CACHE_SPEC = importlib.util.spec_from_file_location(
    "src.feature_generation.core.feature_cache", _FEATURE_CACHE_PATH
)
_FEATURE_CACHE_MODULE = importlib.util.module_from_spec(_FEATURE_CACHE_SPEC)
assert _FEATURE_CACHE_SPEC.loader is not None
_FEATURE_CACHE_SPEC.loader.exec_module(_FEATURE_CACHE_MODULE)
FeatureCacheService = _FEATURE_CACHE_MODULE.FeatureCacheService

_feature_pkg = sys.modules.setdefault("src.feature_generation", types.ModuleType("src.feature_generation"))
core_pkg = types.ModuleType("src.feature_generation.core")
core_pkg.__path__ = []  # Mark as package for submodule imports
_feature_pkg.core = core_pkg
sys.modules["src.feature_generation.core"] = core_pkg
sys.modules["src.feature_generation.core.feature_cache"] = _FEATURE_CACHE_MODULE

feature_bank_stub = types.ModuleType("src.feature_generation.core.feature_bank")
feature_bank_stub.FeatureBank = _DummyFeatureBank
sys.modules["src.feature_generation.core.feature_bank"] = feature_bank_stub


def _enable_pickle_backend(service: FeatureCacheService) -> None:
    """Switch the cache service to a pickle-based backend for tests."""

    def _path(cache_key: str, artifact_type: str = "features") -> Path:
        safe_type = artifact_type.replace("/", "_")
        return service.base_dir / safe_type / f"{cache_key}.pkl"

    def _save(cache_key: str, data: pd.DataFrame, artifact_type: str = "features") -> None:
        if data is None or data.empty:
            return
        path = _path(cache_key, artifact_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_pickle(path)

    def _load(cache_key: str, artifact_type: str = "features") -> Optional[pd.DataFrame]:
        path = _path(cache_key, artifact_type)
        if not path.exists():
            return None
        return pd.read_pickle(path)

    service.save = _save  # type: ignore[assignment]
    service.load = _load  # type: ignore[assignment]


class _DummyEngine:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _DummyProcessType(Enum):
    FEATURE_LOOKBACK = "feature_lookback"


optimized_process_engines = types.ModuleType(
    "src.training.steps.market_analysis.optimized_process_engines"
)
optimized_process_engines.OptimizedFeatureLookbackEngine = _DummyEngine
optimized_process_engines.ProcessType = _DummyProcessType


def _noop(*_args, **_kwargs):
    return None


def _get_logger(name: str):
    return logging.getLogger(name)


class _LoggingContext:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


logging_standards = types.ModuleType(
    "src.training.steps.market_analysis.logging_standards"
)
logging_standards.get_logger = _get_logger
logging_standards.log_info = _noop
logging_standards.log_warning = _noop
logging_standards.log_error = _noop
logging_standards.log_success = _noop
logging_standards.log_debug = _noop
logging_standards.LoggingContext = _LoggingContext
logging_standards.log_step_progress = _noop
logging_standards.log_data_info = _noop
logging_standards.log_validation_result = _noop


market_analysis_pkg = types.ModuleType("src.training.steps.market_analysis")
market_analysis_pkg.optimized_process_engines = optimized_process_engines
market_analysis_pkg.logging_standards = logging_standards

sys.modules["src.training.steps.market_analysis"] = market_analysis_pkg
sys.modules["src.training.steps.market_analysis.optimized_process_engines"] = optimized_process_engines
sys.modules["src.training.steps.market_analysis.logging_standards"] = logging_standards


_pymc_stub = types.ModuleType("pymc")


class _DummyModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


_pymc_stub.Model = _DummyModel
sys.modules.setdefault("pymc", _pymc_stub)

aesara_stub = types.ModuleType("aesara")
aesara_tensor_stub = types.ModuleType("aesara.tensor")
aesara_stub.tensor = aesara_tensor_stub
sys.modules.setdefault("aesara", aesara_stub)
sys.modules.setdefault("aesara.tensor", aesara_tensor_stub)

dependency_manager_stub = types.ModuleType(
    "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.dependency_manager"
)


def _fake_get_dependency(name: str):
    if name == "pymc":
        return _pymc_stub, None
    return None, ImportError(f"stubbed missing dependency: {name}")


dependency_manager_stub.get_dependency = _fake_get_dependency
dependency_manager_stub.is_dependency_available = lambda name: name == "pymc"
sys.modules[
    "src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.dependency_manager"
] = dependency_manager_stub

_FLO_PATH = _ROOT / "src" / "training" / "steps" / "pre_training" / "feature_lookback_optimization" / "feature_lookback_optimization.py"
_FLO_PACKAGE_NAME = "src.training.steps.pre_training.feature_lookback_optimization"
package_module = sys.modules.setdefault(_FLO_PACKAGE_NAME, types.ModuleType(_FLO_PACKAGE_NAME))
package_module.__path__ = [str(_FLO_PATH.parent)]
_FLO_SPEC = importlib.util.spec_from_file_location(
    f"{_FLO_PACKAGE_NAME}.feature_lookback_optimization",
    _FLO_PATH,
)
_FLO_MODULE = importlib.util.module_from_spec(_FLO_SPEC)
_FLO_MODULE.__package__ = _FLO_PACKAGE_NAME
sys.modules[_FLO_SPEC.name] = _FLO_MODULE
assert _FLO_SPEC.loader is not None
_FLO_SPEC.loader.exec_module(_FLO_MODULE)
FeatureLookbackOptimizationComponent = _FLO_MODULE.FeatureLookbackOptimizationComponent
from src.training.config.data_locator import DataLocator, DataLocatorConfig
from src.training.steps.pre_training.components.base_component import ComponentConfig
from src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization import (
    FeatureLookbackOptimizationComponent,
)
from src.training.steps.pre_training.feature_lookback_optimization.core.optimizer import CoreOptimizer
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.optimized_interaction_orchestrator import (
    OptimizedInteractionOrchestrator,
    OptimizedInteractionResult,
    PipelineStage,
)


class _DummyFeatureCategory:
    RETURNS = "returns"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    TREND = "trend"
    OSCILLATOR = "oscillator"
    SUPPORT_RESISTANCE = "support_resistance"
    CANDLESTICK_PATTERN = "candlestick"
    MICROSTRUCTURE = "microstructure"
    ENTROPY = "entropy"
    ORDER_FLOW = "order_flow"
    ACCELERATION = "acceleration"
    TIME = "time"


def test_feature_bank_cache_skips_regeneration(monkeypatch, tmp_path):
    async def _run_test():
        module_name = "src.feature_generation.core.feature_bank"
        monkeypatch.setitem(
            sys.modules, module_name, types.SimpleNamespace(FeatureBank=_DummyFeatureBank)
        )
        generator_module = "src.feature_generation.core.feature_generator"
        monkeypatch.setitem(
            sys.modules, generator_module, types.SimpleNamespace(FeatureCategory=_DummyFeatureCategory)
        )

        _DummyFeatureBank.call_count = 0

        component = FeatureLookbackOptimizationComponent(
            ComponentConfig(symbol="TEST", timeframe="1h")
        )
        component.feature_cache = FeatureCacheService(base_dir=tmp_path / "bank_cache")
        _enable_pickle_backend(component.feature_cache)
        component.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0,
        }

        pipeline_state = {
            'symbol': 'TEST',
            'timeframe': '1h',
            'lookback_config': {'window': 10},
        }
        component._resolve_cache_key(pipeline_state, pipeline_state['lookback_config'])

        base_data = pd.DataFrame(
            {
                'open': [1, 2, 3],
                'high': [2, 3, 4],
                'low': [0, 1, 2],
                'close': [1, 2, 3],
                'volume': [100, 110, 120],
            },
            index=pd.RangeIndex(3),
        )

        cols_first = await component._generate_features_for_optimization(
            base_data.copy(), pipeline_state
        )
        assert _DummyFeatureBank.call_count == 1
        assert component.cache_metrics['writes'] == 1
        cache_key = component._current_cache_key
        assert component.feature_cache.load(cache_key) is not None

        cols_second = await component._generate_features_for_optimization(
            base_data.copy(), pipeline_state
        )
        assert _DummyFeatureBank.call_count == 1, (
            "FeatureBank should not be reinstantiated on cache hit"
        )
        assert component.cache_metrics['hits'] == 1
        assert cols_first == cols_second

    asyncio.run(_run_test())


def test_aligns_market_data_using_locator_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    component = FeatureLookbackOptimizationComponent()
    market_data = pd.DataFrame(
        {
            'open': [1.0, 2.0, 3.0],
            'high': [1.0, 2.0, 3.0],
            'low': [1.0, 2.0, 3.0],
            'close': [1.0, 2.0, 3.0],
            'volume': [10.0, 11.0, 12.0],
        }
    )

    locator = DataLocator(
        DataLocatorConfig(
            base_cache_dir=str(tmp_path / 'regime_cache'),
        )
    )

    pipeline_state = {
        'symbol': 'ETHUSDT',
        'custom_params': {},
        'data_locator': locator,
        'cache_dir_key': 'default',
    }

    with caplog.at_level("WARNING"):
        aligned = component._align_data_with_regime_assignments(market_data.copy(), pipeline_state)

    pd.testing.assert_frame_equal(aligned, market_data)
    assert str(locator.base_cache_dir) in caplog.text


def test_core_optimizer_uses_locator_for_configuration(tmp_path: Path) -> None:
    config_root = tmp_path / 'configs'
    config_root.mkdir()
    config_file = config_root / 'multi_horizon_labeling_config.yaml'
    config_file.write_text(
        '\n'.join(
            [
                'multi_horizon_labeling:',
                '  time_horizons:',
                '    immediate: 3',
                '    short: 6',
            ]
        ),
        encoding='utf-8',
    )

    locator = DataLocator(
        DataLocatorConfig(
            base_config_dir=str(config_root),
        )
    )

    optimizer = CoreOptimizer()
    optimizer.set_data_locator(locator)

    immediate, short = optimizer._get_multi_horizon_boundaries()
    assert immediate == 3
    assert short == 6


def test_interaction_orchestrator_reuses_cached_artifacts(monkeypatch, tmp_path):
    async def _run_test():
        orchestrator = OptimizedInteractionOrchestrator.__new__(
            OptimizedInteractionOrchestrator
        )
        orchestrator.config = types.SimpleNamespace(
            symbol="TEST",
            exchange="binance",
            timeframe="1h",
            feature_budget_pre=10,
            feature_budget_post=5,
            interactions_cap=5,
            transforms_per_parent=1,
            lookback_ceiling_minutes=60,
            latency_budget_ms=100,
            lookback_config={'window': 10},
        )
        orchestrator.logger = logging.getLogger("test-orchestrator")
        orchestrator.feature_cache = FeatureCacheService(
            base_dir=tmp_path / "orchestrator_cache"
        )
        _enable_pickle_backend(orchestrator.feature_cache)
        orchestrator.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'force_refreshes': 0,
        }
        orchestrator.performance_metrics = {}
        orchestrator.stage_start_times = {}
        orchestrator.memory_usage_history = []
        orchestrator.gpu_usage_history = []
        orchestrator.stage_results = {}
        orchestrator.m1_memory_optimizer = None
        orchestrator.m1_gpu_manager = None
        orchestrator.m1_cpu_optimizer = None
        orchestrator.vectorized_core = None
        orchestrator.feature_registry = None
        orchestrator._active_cache_key = None
        orchestrator._current_lookback_hash = None
        orchestrator._force_cache_refresh = False

        call_counts = {'interactions': 0, 'cross': 0}

        async def _stage_initialization(self, training_input, pipeline_state):
            data = training_input['data']
            result = {'data': data, 'stage_time': 0.0, 'success': True}
            self.stage_results[PipelineStage.INITIALIZATION] = result
            return result

        async def _stage_feature_engineering(self, training_input, pipeline_state):
            data = training_input['data']
            result = {'transformed_features': data, 'stage_time': 0.0, 'success': True}
            self.stage_results[PipelineStage.FEATURE_ENGINEERING] = result
            return result

        async def _stage_lookback(self, feature_result, pipeline_state):
            result = {
                'transformed_features': feature_result['transformed_features'],
                'stage_time': 0.0,
                'success': True,
            }
            self.stage_results[PipelineStage.LOOKBACK_OPTIMIZATION] = result
            return result

        async def _stage_transform(self, lookback_result, pipeline_state):
            result = {
                'transformed_features': lookback_result['transformed_features'],
                'stage_time': 0.0,
                'success': True,
            }
            self.stage_results[PipelineStage.TRANSFORM_APPLICATION] = result
            return result

        async def _stage_interactions(self, transform_result, pipeline_state):
            cached = pipeline_state.get('_cached_interaction_features')
            if cached is not None and not self._force_cache_refresh:
                result = {'interactions': cached, 'stage_time': 0.0, 'success': True}
                self.stage_results[PipelineStage.INTERACTION_GENERATION] = result
                return result

            call_counts['interactions'] += 1
            interactions = pd.DataFrame(
                {'i1': [1, 2, 3]},
                index=transform_result['transformed_features'].index,
            )
            if self._active_cache_key:
                self.feature_cache.save(
                    self._active_cache_key, interactions, artifact_type="interactions"
                )
            result = {'interactions': interactions, 'stage_time': 0.0, 'success': True}
            self.stage_results[PipelineStage.INTERACTION_GENERATION] = result
            return result

        async def _stage_cross(self, interaction_result, pipeline_state):
            cached = pipeline_state.get('_cached_cross_timeframe_features')
            transformed = self.stage_results[PipelineStage.TRANSFORM_APPLICATION][
                'transformed_features'
            ]
            if cached is not None and not self._force_cache_refresh:
                result = {
                    'cross_timeframe_features': cached,
                    'all_features': pd.concat(
                        [transformed, interaction_result['interactions']], axis=1
                    ),
                    'stage_time': 0.0,
                    'success': True,
                }
                self.stage_results[PipelineStage.CROSS_TIMEFRAME] = result
                return result

            call_counts['cross'] += 1
            cross = pd.DataFrame({'ctf1': [10, 11, 12]}, index=transformed.index)
            if self._active_cache_key:
                self.feature_cache.save(
                    self._active_cache_key, cross, artifact_type="cross_timeframe"
                )
                self.cache_metrics['writes'] += 1
            result = {
                'cross_timeframe_features': cross,
                'all_features': pd.concat(
                    [transformed, interaction_result['interactions']], axis=1
                ),
                'stage_time': 0.0,
                'success': True,
            }
            self.stage_results[PipelineStage.CROSS_TIMEFRAME] = result
            return result

        async def _stage_final(self, cross_result, pipeline_state):
            final_features = pd.concat(
                [cross_result['all_features'], cross_result['cross_timeframe_features']],
                axis=1,
            )
            result = {
                'final_features': final_features,
                'selected_features': list(final_features.columns),
                'all_feature_names': list(final_features.columns),
                'stage_time': 0.0,
                'success': True,
            }
            self.stage_results[PipelineStage.FINAL_ASSEMBLY] = result
            return result

        async def _stage_validation(self, final_result, pipeline_state):
            result = {'memory_usage_mb': 0.0, 'stage_time': 0.0, 'success': True}
            self.stage_results[PipelineStage.VALIDATION] = result
            return result

        async def _stage_completion(self, validation_result, pipeline_state):
            final_result = self.stage_results[PipelineStage.FINAL_ASSEMBLY]
            interaction_result = self.stage_results[PipelineStage.INTERACTION_GENERATION]
            cross_timeframe_result = self.stage_results[PipelineStage.CROSS_TIMEFRAME]
            return OptimizedInteractionResult(
                features=final_result['final_features'],
                feature_names=final_result['all_feature_names'],
                selected_features=final_result['selected_features'],
                interaction_features=interaction_result['interactions'],
                cross_timeframe_features=cross_timeframe_result['cross_timeframe_features'],
                execution_time=0.0,
                success=True,
                memory_usage_mb=0.0,
            )

        orchestrator._stage_initialization = types.MethodType(
            _stage_initialization, orchestrator
        )
        orchestrator._stage_feature_engineering = types.MethodType(
            _stage_feature_engineering, orchestrator
        )
        orchestrator._stage_lookback_optimization = types.MethodType(
            _stage_lookback, orchestrator
        )
        orchestrator._stage_transform_application = types.MethodType(
            _stage_transform, orchestrator
        )
        orchestrator._stage_interaction_generation = types.MethodType(
            _stage_interactions, orchestrator
        )
        orchestrator._stage_cross_timeframe_features = types.MethodType(
            _stage_cross, orchestrator
        )
        orchestrator._stage_final_assembly = types.MethodType(
            _stage_final, orchestrator
        )
        orchestrator._stage_validation = types.MethodType(
            _stage_validation, orchestrator
        )
        orchestrator._stage_completion = types.MethodType(
            _stage_completion, orchestrator
        )

        training_input = {'data': pd.DataFrame({'open': [1, 2, 3]}, index=pd.RangeIndex(3))}
        pipeline_state = {
            'symbol': 'TEST',
            'timeframe': '1h',
            'lookback_config': {'window': 10},
        }

        orchestrator.stage_results = {}
        await orchestrator.generate_features(training_input, pipeline_state.copy())
        assert call_counts['interactions'] == 1
        assert call_counts['cross'] == 1
        assert orchestrator.cache_metrics['writes'] == 1
        cache_key = orchestrator._active_cache_key
        assert orchestrator.feature_cache.load(cache_key, artifact_type="interactions") is not None
        assert orchestrator.feature_cache.load(cache_key, artifact_type="cross_timeframe") is not None

        orchestrator.stage_results = {}
        await orchestrator.generate_features(training_input, pipeline_state.copy())
        assert call_counts['interactions'] == 1, (
            "Cached run should skip interaction regeneration"
        )
        assert call_counts['cross'] == 1, (
            "Cached run should skip cross-timeframe regeneration"
        )
        assert orchestrator.cache_metrics['hits'] == 1

    asyncio.run(_run_test())
