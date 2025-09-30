import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_clustering_module():
    """Load the NAS-TAS clustering module without importing the entire components package."""
    package_specs = [
        ("src", ROOT_DIR / "src"),
        ("src.training", ROOT_DIR / "src/training"),
        ("src.training.steps", ROOT_DIR / "src/training/steps"),
        ("src.training.steps.market_analysis", ROOT_DIR / "src/training/steps/market_analysis"),
        ("src.training.steps.market_analysis.components", ROOT_DIR / "src/training/steps/market_analysis/components"),
    ]

    for name, path in package_specs:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    module_name = "src.training.steps.market_analysis.components.nas_tas_clustering"
    module_path = ROOT_DIR / "src/training/steps/market_analysis/components/nas_tas_clustering.py"

    if module_name in sys.modules:
        # Remove cached module to get a clean import each time
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_component_module(monkeypatch):
    """Provide a fresh view of the clustering module with clean state for each test."""
    module = load_clustering_module()

    if 'get_logger' not in module.__dict__:
        class _DummyLogger:
            def __getattr__(self, _):  # pragma: no cover - simple no-op logger
                return lambda *args, **kwargs: None

        monkeypatch.setattr(module, 'get_logger', lambda *_, **__: _DummyLogger(), raising=False)

    return module


def test_initialization_with_available_resources(monkeypatch, fresh_component_module):
    module = fresh_component_module

    resources = types.SimpleNamespace(
        matrix_ops=object(),
        vectorized_core=object(),
        batch_processor=object(),
        hardware_manager=object(),
        m1_gpu_optimizer=object(),
        m1_memory_optimizer=object(),
        m1_cpu_optimizer=object(),
    )

    class _DummySetup:
        def __init__(self):
            self.initialize_called = False

        def initialize(self):
            self.initialize_called = True
            return resources

    dummy_setup = _DummySetup()
    monkeypatch.setattr(module, 'HardwareSetup', lambda: dummy_setup, raising=False)

    component = module.NASTASClusteringComponent()

    assert dummy_setup.initialize_called is True
    assert component.hardware_setup is dummy_setup
    assert component.hardware_resources is resources
    assert component.matrix_ops is resources.matrix_ops
    assert component.vectorized_core is resources.vectorized_core
    assert component.batch_processor is resources.batch_processor
    assert component.hardware_manager is resources.hardware_manager
    assert component.m1_gpu_optimizer is resources.m1_gpu_optimizer
    assert component.m1_memory_optimizer is resources.m1_memory_optimizer
    assert component.m1_cpu_optimizer is resources.m1_cpu_optimizer


def test_initialization_without_available_resources(monkeypatch, fresh_component_module):
    module = fresh_component_module

    resources = types.SimpleNamespace(
        matrix_ops=None,
        vectorized_core=None,
        batch_processor=None,
        hardware_manager=None,
        m1_gpu_optimizer=None,
        m1_memory_optimizer=None,
        m1_cpu_optimizer=None,
    )

    class _DummySetup:
        def initialize(self):
            return resources

    monkeypatch.setattr(module, 'HardwareSetup', lambda: _DummySetup(), raising=False)

    component = module.NASTASClusteringComponent()

    assert component.hardware_resources is resources
    assert component.matrix_ops is None
    assert component.vectorized_core is None
    assert component.batch_processor is None
    assert component.hardware_manager is None
    assert component.m1_gpu_optimizer is None
    assert component.m1_memory_optimizer is None
    assert component.m1_cpu_optimizer is None
