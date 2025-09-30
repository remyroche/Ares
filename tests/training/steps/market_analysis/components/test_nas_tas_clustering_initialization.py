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

    # Ensure defaults before each test
    monkeypatch.setattr(module, 'matrix_ops', None, raising=False)
    monkeypatch.setattr(module, 'vectorized_core', None, raising=False)
    monkeypatch.setattr(module, 'batch_processor', None, raising=False)
    monkeypatch.setattr(module, 'hardware_manager', None, raising=False)
    monkeypatch.setattr(module, 'm1_gpu_optimizer', None, raising=False)
    monkeypatch.setattr(module, 'm1_memory_optimizer', None, raising=False)
    monkeypatch.setattr(module, 'm1_cpu_optimizer', None, raising=False)

    if 'get_logger' not in module.__dict__:
        class _DummyLogger:
            def __getattr__(self, _):  # pragma: no cover - simple no-op logger
                return lambda *args, **kwargs: None

        monkeypatch.setattr(module, 'get_logger', lambda *_, **__: _DummyLogger(), raising=False)

    return module


def test_initialization_with_available_resources(monkeypatch, fresh_component_module):
    module = fresh_component_module

    matrix_ops = object()
    vectorized_core = object()
    batch_processor = object()
    hardware_manager = object()
    m1_gpu_optimizer = object()
    m1_memory_optimizer = object()
    m1_cpu_optimizer = object()

    monkeypatch.setattr(module, 'MATRIX_OPERATIONS_AVAILABLE', True, raising=False)
    monkeypatch.setattr(module, 'HARDWARE_OPTIMIZATION_AVAILABLE', True, raising=False)
    monkeypatch.setattr(module, 'M1_HARDWARE_AVAILABLE', True, raising=False)

    monkeypatch.setattr(module, 'get_unified_matrix_operations', lambda: matrix_ops, raising=False)
    monkeypatch.setattr(module, 'get_vectorized_processing_core', lambda: vectorized_core, raising=False)
    monkeypatch.setattr(module, 'get_batch_matrix_processor', lambda: batch_processor, raising=False)

    monkeypatch.setattr(module, 'get_unified_hardware_manager', lambda: hardware_manager, raising=False)
    monkeypatch.setattr(module, 'get_m1_gpu_optimizer', lambda: m1_gpu_optimizer, raising=False)
    monkeypatch.setattr(module, 'get_m1_memory_optimizer', lambda: m1_memory_optimizer, raising=False)
    monkeypatch.setattr(module, 'get_m1_cpu_optimizer', lambda: m1_cpu_optimizer, raising=False)

    component = module.NASTASClusteringComponent()

    assert component.matrix_ops is matrix_ops
    assert component.vectorized_core is vectorized_core
    assert component.batch_processor is batch_processor
    assert component.hardware_manager is hardware_manager
    assert component.m1_gpu_optimizer is m1_gpu_optimizer
    assert component.m1_memory_optimizer is m1_memory_optimizer
    assert component.m1_cpu_optimizer is m1_cpu_optimizer


def test_initialization_without_available_resources(monkeypatch, fresh_component_module):
    module = fresh_component_module

    monkeypatch.setattr(module, 'MATRIX_OPERATIONS_AVAILABLE', False, raising=False)
    monkeypatch.setattr(module, 'HARDWARE_OPTIMIZATION_AVAILABLE', False, raising=False)
    monkeypatch.setattr(module, 'M1_HARDWARE_AVAILABLE', False, raising=False)

    def _should_not_be_called(*args, **kwargs):  # pragma: no cover - safety net
        raise AssertionError('Resource initializer should not be called when unavailable')

    monkeypatch.setattr(module, 'get_unified_matrix_operations', _should_not_be_called, raising=False)
    monkeypatch.setattr(module, 'get_vectorized_processing_core', _should_not_be_called, raising=False)
    monkeypatch.setattr(module, 'get_batch_matrix_processor', _should_not_be_called, raising=False)
    monkeypatch.setattr(module, 'get_unified_hardware_manager', _should_not_be_called, raising=False)
    monkeypatch.setattr(module, 'get_m1_gpu_optimizer', _should_not_be_called, raising=False)
    monkeypatch.setattr(module, 'get_m1_memory_optimizer', _should_not_be_called, raising=False)
    monkeypatch.setattr(module, 'get_m1_cpu_optimizer', _should_not_be_called, raising=False)

    component = module.NASTASClusteringComponent()

    assert component.matrix_ops is None
    assert component.vectorized_core is None
    assert component.batch_processor is None
    assert component.hardware_manager is None
    assert component.m1_gpu_optimizer is None
    assert component.m1_memory_optimizer is None
    assert component.m1_cpu_optimizer is None
