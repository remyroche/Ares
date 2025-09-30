import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_hardware_module():
    package_specs = [
        ("src", ROOT_DIR / "src"),
        ("src.training", ROOT_DIR / "src/training"),
        ("src.training.steps", ROOT_DIR / "src/training/steps"),
        ("src.training.steps.market_analysis", ROOT_DIR / "src/training/steps/market_analysis"),
        (
            "src.training.steps.market_analysis.components",
            ROOT_DIR / "src/training/steps/market_analysis/components",
        ),
    ]

    for name, path in package_specs:
        if name not in sys.modules:
            module = importlib.util.module_from_spec(
                importlib.machinery.ModuleSpec(name, loader=None)
            )
            module.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = module

    module_name = "src.training.steps.market_analysis.components.hardware_setup"
    module_path = ROOT_DIR / "src/training/steps/market_analysis/components/hardware_setup.py"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader  # pragma: no cover - ensure loader exists
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.fixture
def hardware_module():
    return load_hardware_module()


def test_initialize_returns_resources_when_available(monkeypatch, hardware_module):
    matrix_ops = object()
    vectorized_core = object()
    batch_processor = object()
    hardware_manager = object()
    m1_gpu_optimizer = object()
    m1_memory_optimizer = object()
    m1_cpu_optimizer = object()

    monkeypatch.setattr(hardware_module, "MATRIX_OPERATIONS_AVAILABLE", True, raising=False)
    monkeypatch.setattr(hardware_module, "HARDWARE_OPTIMIZATION_AVAILABLE", True, raising=False)
    monkeypatch.setattr(hardware_module, "M1_HARDWARE_AVAILABLE", True, raising=False)

    monkeypatch.setattr(hardware_module, "get_unified_matrix_operations", lambda: matrix_ops, raising=False)
    monkeypatch.setattr(hardware_module, "get_vectorized_processing_core", lambda: vectorized_core, raising=False)
    monkeypatch.setattr(hardware_module, "get_batch_matrix_processor", lambda: batch_processor, raising=False)
    monkeypatch.setattr(hardware_module, "get_unified_hardware_manager", lambda: hardware_manager, raising=False)
    monkeypatch.setattr(hardware_module, "get_m1_gpu_optimizer", lambda: m1_gpu_optimizer, raising=False)
    monkeypatch.setattr(hardware_module, "get_m1_memory_optimizer", lambda: m1_memory_optimizer, raising=False)
    monkeypatch.setattr(hardware_module, "get_m1_cpu_optimizer", lambda: m1_cpu_optimizer, raising=False)

    setup = hardware_module.HardwareSetup()
    resources = setup.initialize()

    assert isinstance(resources, hardware_module.HardwareResources)
    assert resources.matrix_ops is matrix_ops
    assert resources.vectorized_core is vectorized_core
    assert resources.batch_processor is batch_processor
    assert resources.hardware_manager is hardware_manager
    assert resources.m1_gpu_optimizer is m1_gpu_optimizer
    assert resources.m1_memory_optimizer is m1_memory_optimizer
    assert resources.m1_cpu_optimizer is m1_cpu_optimizer


def test_initialize_handles_missing_resources(monkeypatch, hardware_module):
    monkeypatch.setattr(hardware_module, "MATRIX_OPERATIONS_AVAILABLE", False, raising=False)
    monkeypatch.setattr(hardware_module, "HARDWARE_OPTIMIZATION_AVAILABLE", False, raising=False)
    monkeypatch.setattr(hardware_module, "M1_HARDWARE_AVAILABLE", False, raising=False)

    def _should_not_be_called(*_args, **_kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("initializer should not be called when unavailable")

    monkeypatch.setattr(hardware_module, "get_unified_matrix_operations", _should_not_be_called, raising=False)
    monkeypatch.setattr(hardware_module, "get_vectorized_processing_core", _should_not_be_called, raising=False)
    monkeypatch.setattr(hardware_module, "get_batch_matrix_processor", _should_not_be_called, raising=False)
    monkeypatch.setattr(hardware_module, "get_unified_hardware_manager", _should_not_be_called, raising=False)
    monkeypatch.setattr(hardware_module, "get_m1_gpu_optimizer", _should_not_be_called, raising=False)
    monkeypatch.setattr(hardware_module, "get_m1_memory_optimizer", _should_not_be_called, raising=False)
    monkeypatch.setattr(hardware_module, "get_m1_cpu_optimizer", _should_not_be_called, raising=False)

    setup = hardware_module.HardwareSetup()
    resources = setup.initialize()

    assert resources.matrix_ops is None
    assert resources.vectorized_core is None
    assert resources.batch_processor is None
    assert resources.hardware_manager is None
    assert resources.m1_gpu_optimizer is None
    assert resources.m1_memory_optimizer is None
    assert resources.m1_cpu_optimizer is None
