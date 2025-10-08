import sys
import types
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.training.steps.pre_training.multi_horizon_profit_labeler import validate_and_prepare_dataframe


def _load_pre_training_sub_pipeline_module():
    module_name = 'src.training.steps.pre_training.sub_pipeline'
    if module_name in sys.modules:
        return sys.modules[module_name]

    market_module_name = 'src.training.steps.market_analysis'
    if market_module_name not in sys.modules:
        market_stub = types.ModuleType(market_module_name)
        market_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules[market_module_name] = market_stub
        steps_parent = sys.modules.get('src.training.steps')
        if steps_parent is None:
            import src.training.steps as steps_parent  # type: ignore
        setattr(steps_parent, 'market_analysis', market_stub)

    optimized_engines_module = 'src.training.steps.market_analysis.optimized_process_engines'
    if optimized_engines_module not in sys.modules:
        optimized_stub = types.ModuleType(optimized_engines_module)
        class _DummyEngine:  # pragma: no cover - test stub
            pass

        class _DummyProcessType:  # pragma: no cover - test stub
            pass

        optimized_stub.OptimizedFeatureLookbackEngine = _DummyEngine
        optimized_stub.OptimizedFeatureSelectionEngine = _DummyEngine
        optimized_stub.ProcessType = _DummyProcessType
        sys.modules[optimized_engines_module] = optimized_stub

    logging_standards_module = 'src.training.steps.market_analysis.logging_standards'
    if logging_standards_module not in sys.modules:
        logging_stub = types.ModuleType(logging_standards_module)

        def _noop_logger(*_args, **_kwargs):  # pragma: no cover - test stub
            return None

        logging_stub.get_logger = _noop_logger
        logging_stub.log_info = _noop_logger
        logging_stub.log_warning = _noop_logger
        logging_stub.log_error = _noop_logger
        logging_stub.log_success = _noop_logger
        logging_stub.log_debug = _noop_logger
        logging_stub.LoggingContext = object  # type: ignore[assignment]
        logging_stub.log_step_progress = _noop_logger
        logging_stub.log_data_info = _noop_logger
        logging_stub.log_validation_result = _noop_logger
        sys.modules[logging_standards_module] = logging_stub

    modular_feature_module = 'src.training.steps.pre_training.feature_lookback_optimization.feature_lookback_optimization_modular'
    if modular_feature_module not in sys.modules:
        modular_stub = types.ModuleType(modular_feature_module)
        class _ModularComponent:  # pragma: no cover - test stub
            pass

        modular_stub.FeatureLookbackOptimizationComponent = _ModularComponent
        sys.modules[modular_feature_module] = modular_stub

    component_factory_module = 'src.training.steps.pre_training.components.component_factory'
    if component_factory_module not in sys.modules:
        component_factory_stub = types.ModuleType(component_factory_module)

        class _StubComponentConfig:
            def __init__(self, **kwargs):
                self.custom_params = kwargs.get('custom_params', {})

        class _StubComponentFactory:
            @staticmethod
            def create_component(*_args, **_kwargs):  # pragma: no cover - test helper
                raise NotImplementedError("ComponentFactory is not available in test stub")

        component_factory_stub.ComponentFactory = _StubComponentFactory
        component_factory_stub.ComponentConfig = _StubComponentConfig
        sys.modules[component_factory_module] = component_factory_stub

    sys.modules.pop('src.training.steps.pre_training.components', None)
    parent_module = sys.modules.get('src.training.steps.pre_training')
    if parent_module is not None and hasattr(parent_module, 'components'):
        delattr(parent_module, 'components')

    import src.training.steps.pre_training.multi_horizon_profit_labeler as mh_module

    if not hasattr(mh_module, 'create_multi_horizon_labeler'):
        def _stub_create_multi_horizon_labeler(*_args, **_kwargs):  # pragma: no cover - test helper
            return None

        mh_module.create_multi_horizon_labeler = _stub_create_multi_horizon_labeler  # type: ignore[attr-defined]

    if not hasattr(mh_module, 'apply_multi_horizon_labeling'):
        def _stub_apply_multi_horizon_labeling(*_args, **_kwargs):  # pragma: no cover - test helper
            return {}

        mh_module.apply_multi_horizon_labeling = _stub_apply_multi_horizon_labeling  # type: ignore[attr-defined]

    from src.training.steps.pre_training import sub_pipeline as module

    return module


def test_component_quality_alerts_triggered(caplog):
    sub_pipeline_module = _load_pre_training_sub_pipeline_module()
    pipeline = sub_pipeline_module.PreTrainingSubPipeline()
    config = sub_pipeline_module.SubPipelineConfig(
        label_imbalance_warning_threshold=0.6,
        nan_rate_warning_threshold=0.1,
        duplicate_index_warning_threshold=0.1,
    )

    labeled_df = pd.DataFrame({
        'label': [1, 1, 1, 0, np.nan],
    }, index=[0, 0, 1, 2, 3])

    artifacts = {
        'multi_horizon_labeling_result': {
            'labeled_data': labeled_df,
        }
    }

    with caplog.at_level('WARNING', logger=pipeline.logger.name):
        metrics, alerts = pipeline._analyze_component_quality(
            'multi_horizon_profit_labeler',
            artifacts,
            config,
        )

    dataset_metrics = metrics['multi_horizon_labeling_result.labeled_data']

    assert dataset_metrics['duplicate_index_share'] == pytest.approx(0.2, rel=1e-6)
    assert dataset_metrics['nan_rate'] == pytest.approx(0.2, rel=1e-6)
    assert 'label_balance' in dataset_metrics
    assert dataset_metrics['label_balance']['columns']['label']['dominant_share'] == pytest.approx(0.75)

    assert any('duplicate index share' in alert for alert in alerts)
    assert any('NaN rate' in alert for alert in alerts)
    assert any('dominant label share' in alert for alert in alerts)

    logged_messages = ' '.join(caplog.messages)
    for alert in alerts:
        assert alert in logged_messages


def test_validate_dataframe_respects_duplicate_threshold():
    df = pd.DataFrame({'value': [1, 2, 3]}, index=[0, 0, 1])

    metrics_low: Dict[str, float] = {}
    cleaned_low = validate_and_prepare_dataframe(
        df.copy(),
        "TestFrame",
        duplicate_threshold=0.5,
        metrics=metrics_low,
    )

    assert cleaned_low.index.duplicated().any()
    assert metrics_low['deduplicated'] is False
    assert metrics_low['duplicate_index_share'] == pytest.approx(1 / 3, rel=1e-6)

    metrics_high: Dict[str, float] = {}
    cleaned_high = validate_and_prepare_dataframe(
        df.copy(),
        "TestFrame",
        duplicate_threshold=0.2,
        metrics=metrics_high,
    )

    assert not cleaned_high.index.duplicated().any()
    assert metrics_high['deduplicated'] is True
    assert metrics_high['duplicate_index_share'] == pytest.approx(1 / 3, rel=1e-6)


def test_prepare_interactive_training_input_uses_batches():
    sub_pipeline_module = _load_pre_training_sub_pipeline_module()
    pipeline = sub_pipeline_module.PreTrainingSubPipeline()
    config = sub_pipeline_module.SubPipelineConfig()

    index = pd.date_range('2024-01-01', periods=10, freq='H')
    market_data = pd.DataFrame(
        {
            'open': np.arange(10, dtype=float),
            'high': np.arange(10, dtype=float) + 1,
            'low': np.arange(10, dtype=float) - 1,
            'close': np.arange(10, dtype=float) + 0.5,
            'volume': np.arange(10, dtype=float) + 100,
        },
        index=index,
    )
    batches = [market_data.iloc[:5], market_data.iloc[5:]]
    labels = pd.DataFrame({'target': np.arange(10)}, index=index)

    pipeline._current_pipeline_state['multi_horizon_labeling_result'] = {
        'market_data': market_data,
        'market_data_batches': batches,
        'labeled_data': labels,
    }

    pipeline_state = pipeline._prepare_component_pipeline_state(config)
    training_input = pipeline._prepare_interactive_training_input(pipeline_state)

    assert 'data_batches' in training_input
    assert len(training_input['data_batches']) == 2
    pd.testing.assert_frame_equal(training_input['data'], market_data)
