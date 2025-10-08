import json
import importlib.util
import json
import pickle
import sys
import types
from pathlib import Path

import importlib.util
import json
import pickle
import sys
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _TorchStub(types.ModuleType):
    def __getattr__(self, name):
        module_name = f"torch.{name}"
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            module.__path__ = []
            sys.modules[module_name] = module
        return module


if 'torch' not in sys.modules:
    torch_stub = _TorchStub('torch')
    torch_stub.__path__ = []
    sys.modules['torch'] = torch_stub
    submodules = [
        'torch.nn',
        'torch.nn.utils',
        'torch.nn.utils.prune',
        'torch.optim',
        'torch.optim.lr_scheduler',
        'torch.cuda',
        'torch.utils',
        'torch.utils.data',
    ]

    for submodule_name in submodules:
        submodule = types.ModuleType(submodule_name)
        submodule.__path__ = []
        if submodule_name == 'torch.nn':
            submodule.Module = type('Module', (), {})
        sys.modules[submodule_name] = submodule
        parent_name, attr_name = submodule_name.rsplit('.', 1)
        parent_module = sys.modules[parent_name]
        setattr(parent_module, attr_name, submodule)

if 'mlflow' not in sys.modules:
    sys.modules['mlflow'] = types.ModuleType('mlflow')


def _install_base_component_stub() -> None:
    package_name = 'src.training.steps.market_analysis.components'
    module_name = f'{package_name}.base_component'

    if module_name in sys.modules:
        return

    stub_module = types.ModuleType(module_name)

    class ComponentConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

            self.symbol = kwargs.get('symbol', 'ETHUSDT')
            self.exchange = kwargs.get('exchange', 'binance')
            self.timeframe = kwargs.get('timeframe', '1h')

    class ComponentResult:
        def __init__(
            self,
            success: bool,
            artifacts: dict | None = None,
            error: Exception | None = None,
            metrics: dict | None = None,
            warnings: list | None = None,
            execution_time: float = 0.0,
            metadata: dict | None = None,
        ):
            self.success = success
            self.artifacts = artifacts or {}
            self.error = error
            self.metrics = metrics or {}
            self.warnings = warnings or []
            self.execution_time = execution_time
            self.metadata = metadata or {}

            if (self.success and self.error is not None) or (not self.success and self.error is None):
                raise ValueError("Invalid ComponentResult state for stub")

    class BaseMarketAnalysisComponent:
        def __init__(self, config: ComponentConfig | None = None):
            self.config = config or ComponentConfig()
            logger_methods = {
                'info': lambda *args, **kwargs: None,
                'warning': lambda *args, **kwargs: None,
                'error': lambda *args, **kwargs: None,
                'debug': lambda *args, **kwargs: None,
            }
            self.logger = types.SimpleNamespace(**logger_methods)

    stub_module.ComponentConfig = ComponentConfig
    stub_module.ComponentResult = ComponentResult
    stub_module.BaseMarketAnalysisComponent = BaseMarketAnalysisComponent

    sys.modules[module_name] = stub_module

    package = types.ModuleType(package_name)
    package.__path__ = []
    package.base_component = stub_module
    sys.modules[package_name] = package


_install_base_component_stub()

_CONFIG_UTILS_PATH = Path(__file__).resolve().parents[5] / 'src/training/steps/market_analysis/regime_data_splitting/config_utils.py'
_config_spec = importlib.util.spec_from_file_location(
    'src.training.steps.market_analysis.regime_data_splitting.config_utils',
    _CONFIG_UTILS_PATH,
)
_config_module = importlib.util.module_from_spec(_config_spec)
assert _config_spec.loader is not None
_config_spec.loader.exec_module(_config_module)

RegimeDataSplittingConfig = _config_module.RegimeDataSplittingConfig
get_config_manager = _config_module.get_config_manager
reset_global_config = _config_module.reset_global_config
get_path_manager = _config_module.get_path_manager

_COMPONENT_PATH = Path(__file__).resolve().parents[5] / 'src/training/steps/market_analysis/regime_data_splitting/regime_data_splitting_component.py'
_component_spec = importlib.util.spec_from_file_location(
    'src.training.steps.market_analysis.regime_data_splitting.regime_data_splitting_component',
    _COMPONENT_PATH,
)
_component_module = importlib.util.module_from_spec(_component_spec)
assert _component_spec.loader is not None
_component_spec.loader.exec_module(_component_module)
RegimeDataSplittingComponent = _component_module.RegimeDataSplittingComponent


@pytest.fixture
def configured_component(tmp_path):
    reset_global_config()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    config = RegimeDataSplittingConfig(artifacts_dir=str(artifacts_dir))
    get_config_manager(config)

    component = object.__new__(RegimeDataSplittingComponent)
    component.config = types.SimpleNamespace(symbol="ETHUSDT", exchange="binance", timeframe="1h")
    component.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    component.path_manager = get_path_manager(config)

    yield component, artifacts_dir

    reset_global_config()


def _write_assignment_artifacts(artifacts_dir: Path, assignments: np.ndarray) -> Path:
    clustering_dir = artifacts_dir / "regime_data_splitting"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = clustering_dir / "nas_tas_clustering_results_test.pkl"
    with open(artifact_path, "wb") as handle:
        pickle.dump({"results": {"cluster_assignments": assignments}}, handle)

    metadata = {
        "artifact_path": str(artifact_path.relative_to(artifacts_dir)),
        "expected_length": int(len(assignments)),
    }
    metadata_path = clustering_dir / "cluster_assignments_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return artifact_path


def test_truncated_assignments_recovered_from_artifacts(configured_component):
    component, artifacts_dir = configured_component
    assignments = np.arange(12, dtype=np.int32)
    _write_assignment_artifacts(artifacts_dir, assignments)

    truncated_preview = "[0 1 2 ... 10 11]"
    regime_discovery = {
        "clustering_result": {"cluster_assignments": truncated_preview}
    }

    result = component._extract_regime_states(regime_discovery)

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, assignments)


def test_truncated_assignments_without_artifacts_raise(configured_component):
    component, artifacts_dir = configured_component
    clustering_dir = artifacts_dir / "regime_data_splitting"
    clustering_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = clustering_dir / "cluster_assignments_metadata.json"
    metadata_path.write_text(json.dumps({"expected_length": 6}), encoding="utf-8")

    truncated_preview = "[0 1 2 ... 3 4 5]"
    regime_discovery = {
        "clustering_result": {"cluster_assignments": truncated_preview}
    }

    with pytest.raises(ValueError):
        component._extract_regime_states(regime_discovery)
