import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# Stub modules required during import that may not be available in the test environment
mock_labeler_module = types.ModuleType('src.training.steps.pre_training.multi_horizon_profit_labeler')
mock_labeler_module.create_multi_horizon_labeler = lambda *args, **kwargs: None
class _DummyLabeler:
    pass

mock_labeler_module.MultiHorizonProfitLabeler = _DummyLabeler
mock_labeler_module.MultiHorizonConfig = type('MultiHorizonConfig', (), {})
mock_labeler_module.apply_multi_horizon_labeling = lambda *args, **kwargs: None
sys.modules.setdefault('src.training.steps.pre_training.multi_horizon_profit_labeler', mock_labeler_module)

mock_pid_module = types.ModuleType('src.training.steps.pre_training.pid_based_feature_generation')
mock_pid_module.PIDBasedFeatureOrchestrator = type('PIDBasedFeatureOrchestrator', (), {})
mock_pid_module.OrchestratorConfig = type('OrchestratorConfig', (), {})
mock_pid_module.InteractionFeatureGenerator = type('InteractionFeatureGenerator', (), {})
mock_pid_module.InteractionConfig = type('InteractionConfig', (), {})
mock_pid_module.CrossTimeframeFeatureGenerator = type('CrossTimeframeFeatureGenerator', (), {})
mock_pid_module.CrossTimeframeConfig = type('CrossTimeframeConfig', (), {})
mock_pid_module.OptimizedLookbackIntegration = type('OptimizedLookbackIntegration', (), {})
mock_pid_module.FeatureSelectionMechanism = type('FeatureSelectionMechanism', (), {})
mock_pid_module.FeatureSelectionConfig = type('FeatureSelectionConfig', (), {})
mock_pid_module.SelectionStrategy = type('SelectionStrategy', (), {})
sys.modules.setdefault('src.training.steps.pre_training.pid_based_feature_generation', mock_pid_module)

feature_generation_module = types.ModuleType('src.feature_generation')
feature_generation_core = types.ModuleType('src.feature_generation.core')
feature_generation_core.feature_generator = types.ModuleType('src.feature_generation.core.feature_generator')
feature_generation_core.feature_generator.FeatureGenerator = type('FeatureGenerator', (), {})
feature_generation_core.feature_generator.FeatureResult = type('FeatureResult', (), {})
feature_generation_module.core = feature_generation_core
sys.modules.setdefault('src.feature_generation', feature_generation_module)
sys.modules.setdefault('src.feature_generation.core', feature_generation_core)
sys.modules.setdefault('src.feature_generation.core.feature_generator', feature_generation_core.feature_generator)
sys.modules.setdefault('src.feature_generation.categories', types.ModuleType('src.feature_generation.categories'))
torch_module = types.ModuleType('torch')
torch_nn_module = types.ModuleType('torch.nn')
torch_optim_module = types.ModuleType('torch.optim')
torch_nn_utils_module = types.ModuleType('torch.nn.utils')
torch_nn_functional_module = types.ModuleType('torch.nn.functional')
torch_utils_module = types.ModuleType('torch.utils')
torch_utils_data_module = types.ModuleType('torch.utils.data')
torch_module.nn = torch_nn_module
torch_module.optim = torch_optim_module
torch_module.Tensor = type('Tensor', (), {})
torch_module.device = type('device', (), {})
torch_module.utils = torch_utils_module
torch_nn_module.Module = type('Module', (), {})
torch_nn_module.Sequential = type('Sequential', (), {})
torch_nn_module.Linear = type('Linear', (), {})
torch_nn_module.utils = torch_nn_utils_module
torch_nn_module.functional = torch_nn_functional_module
torch_nn_utils_module.prune = types.ModuleType('torch.nn.utils.prune')
torch_optim_module.Optimizer = type('Optimizer', (), {})
torch_utils_module.data = torch_utils_data_module
torch_utils_data_module.DataLoader = type('DataLoader', (), {})
sys.modules.setdefault('torch', torch_module)
sys.modules.setdefault('torch.nn', torch_nn_module)
sys.modules.setdefault('torch.optim', torch_optim_module)
sys.modules.setdefault('torch.nn.utils', torch_nn_utils_module)
sys.modules.setdefault('torch.nn.utils.prune', torch_nn_utils_module.prune)
sys.modules.setdefault('torch.nn.functional', torch_nn_functional_module)
sys.modules.setdefault('torch.utils', torch_utils_module)
sys.modules.setdefault('torch.utils.data', torch_utils_data_module)
gymnasium_module = types.ModuleType('gymnasium')
gymnasium_spaces = types.ModuleType('gymnasium.spaces')
gymnasium_module.spaces = gymnasium_spaces
gymnasium_module.Env = type('Env', (), {})
sys.modules.setdefault('gymnasium', gymnasium_module)
sys.modules.setdefault('gymnasium.spaces', gymnasium_spaces)

tas_regime_module = types.ModuleType('src.training.steps.market_analysis.tas_regime')
tas_regime_core_module = types.ModuleType('src.training.steps.market_analysis.tas_regime.core')
tas_regime_detector_module = types.ModuleType('src.training.steps.market_analysis.tas_regime.core.enhanced_tas_regime_detector')
tas_regime_detector_module.EnhancedTASRegimeDetector = type('EnhancedTASRegimeDetector', (), {})
tas_regime_detector_module.EnhancedTASResult = type('EnhancedTASResult', (), {})
tas_regime_core_module.enhanced_tas_regime_detector = tas_regime_detector_module
tas_regime_config_module = types.ModuleType('src.training.steps.market_analysis.tas_regime.core.tas_config')
tas_regime_config_module.TASConfig = type('TASConfig', (), {})
tas_regime_module.core = tas_regime_core_module
sys.modules.setdefault('src.training.steps.market_analysis.tas_regime', tas_regime_module)
sys.modules.setdefault('src.training.steps.market_analysis.tas_regime.core', tas_regime_core_module)
sys.modules.setdefault('src.training.steps.market_analysis.tas_regime.core.enhanced_tas_regime_detector', tas_regime_detector_module)
sys.modules.setdefault('src.training.steps.market_analysis.tas_regime.core.tas_config', tas_regime_config_module)

monitoring_module = types.ModuleType('src.trading.monitoring.enhanced_monitoring_orchestrator')
monitoring_module.EnhancedMonitoringOrchestrator = type('EnhancedMonitoringOrchestrator', (), {})
sys.modules.setdefault('src.trading.monitoring.enhanced_monitoring_orchestrator', monitoring_module)

trade_monitor_module = types.ModuleType('src.trading.monitoring.comprehensive_trade_monitor')
trade_monitor_module.ComprehensiveTradeMonitor = type('ComprehensiveTradeMonitor', (), {})
trade_monitor_module.DetailedTradeMetrics = type('DetailedTradeMetrics', (), {})
trade_monitor_module.TradingSessionMetrics = type('TradingSessionMetrics', (), {})
trade_monitor_module.comprehensive_trade_monitor = None
trade_monitor_module.initialize_comprehensive_monitoring = lambda *args, **kwargs: None
trade_monitor_module.record_detailed_trade = lambda *args, **kwargs: None
trade_monitor_module.update_trade_outcome = lambda *args, **kwargs: None
sys.modules.setdefault('src.trading.monitoring.comprehensive_trade_monitor', trade_monitor_module)


src_trading_module = sys.modules.setdefault('src.trading', types.ModuleType('src.trading'))
if not hasattr(src_trading_module, '__path__'):
    src_trading_module.__path__ = []

model_selection_pkg = sys.modules.setdefault('src.trading.model_selection', types.ModuleType('src.trading.model_selection'))
model_selection_pkg.__path__ = [
    str(Path(__file__).resolve().parents[2] / 'src' / 'trading' / 'model_selection')
]

module_path = Path(__file__).resolve().parents[2] / 'src' / 'trading' / 'model_selection' / 'model_selector_service.py'
spec = importlib.util.spec_from_file_location(
    'src.trading.model_selection.model_selector_service',
    module_path
)
model_selector_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_selector_module
spec.loader.exec_module(model_selector_module)

ModelSelectorService = model_selector_module.ModelSelectorService
TradingModelConfig = model_selector_module.TradingModelConfig


class DummyModelSelector:
    def __init__(self):
        self.calls = []

    def select_model_for_regime(self, regime_id, available_models):
        self.calls.append((regime_id, list(available_models)))
        selected_model = available_models[0]
        return selected_model, {selected_model: 1.0}


class DummyRegimeResult:
    def __init__(self, regime_id: int = 0, confidence: float = 1.0):
        self.success = True
        self.regime_predictions = [regime_id]
        n_regimes = max(regime_id + 1, 1)
        probs = np.zeros((1, n_regimes))
        probs[0, regime_id] = confidence
        self.regime_probabilities = probs
        self.execution_time = 0.01


def _basic_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'open': [1.0, 1.1, 1.2],
            'high': [1.05, 1.15, 1.25],
            'low': [0.95, 1.05, 1.15],
            'close': [1.02, 1.12, 1.22],
            'volume': [1000, 1100, 1200],
        }
    )


def test_select_models_for_trading_mixed_categories(monkeypatch):
    config = TradingModelConfig(
        analyst_models=['analyst_rf', 'analyst_xgb'],
        tactician_models=['tactician_rf', 'tactician_xgb'],
    )

    service = ModelSelectorService(config)
    service.model_selector = DummyModelSelector()

    monkeypatch.setattr(
        service,
        "_detect_current_regime",
        lambda data: DummyRegimeResult(regime_id=1, confidence=0.9),
    )

    result = service.select_models_for_trading(
        market_data=_basic_market_data(),
        model_types=['analyst', 'tactician'],
        timeframe='5m',
    )

    assert set(result.selected_models.keys()) == {'analyst', 'tactician'}
    assert result.selected_models['analyst'] == 'analyst_rf_5m'
    assert result.selected_models['tactician'] == 'tactician_rf_5m'

    expected_available = {
        'analyst': ['analyst_rf_5m', 'analyst_xgb_5m'],
        'tactician': ['tactician_rf_5m', 'tactician_xgb_5m'],
    }

    for model_type, call in zip(['analyst', 'tactician'], service.model_selector.calls):
        _, available_models = call
        assert available_models == expected_available[model_type]

    assert result.ensemble_weights['analyst'] == {'analyst_rf_5m': 1.0}
    assert result.ensemble_weights['tactician'] == {'tactician_rf_5m': 1.0}
    assert result.confirmation_status == 'single_timeframe'
    assert result.confirmation_details == {}


def test_select_models_for_trading_preserves_direct_model_keys(monkeypatch):
    config = TradingModelConfig(
        tactician_models=['rf', 'xgb'],
    )

    service = ModelSelectorService(config)
    service.model_selector = DummyModelSelector()

    monkeypatch.setattr(
        service,
        "_detect_current_regime",
        lambda data: DummyRegimeResult(regime_id=2, confidence=0.85),
    )

    result = service.select_models_for_trading(
        market_data=_basic_market_data(),
        model_types=None,
        timeframe='5m',
    )

    assert set(result.selected_models.keys()) == {'rf', 'xgb'}
    assert result.selected_models['rf'] == 'rf_5m'
    assert result.selected_models['xgb'] == 'xgb_5m'

    expected_calls = [
        (2, ['rf_5m']),
        (2, ['xgb_5m']),
    ]

    assert service.model_selector.calls == expected_calls
    assert result.confirmation_status == 'single_timeframe'
    assert result.confirmation_details == {}
