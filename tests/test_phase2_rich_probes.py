"""Tests for Phase 2 rich probe gating logic."""

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_PACKAGE_NAME = "tmp_phase2_pkg"


def _ensure_package(package_name: str) -> None:
    """Ensure a namespace package exists in ``sys.modules`` for dynamic imports."""

    parts = package_name.split(".")
    for idx in range(1, len(parts) + 1):
        package = ".".join(parts[:idx])
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package] = module


def _load_module(module_name: str, relative_path: str):
    """Load a module directly from a file path under the temporary package."""

    full_module_name = f"{_PACKAGE_NAME}.{module_name}"
    _ensure_package(_PACKAGE_NAME)

    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(full_module_name, module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)
    return module


config_module = _load_module(
    "config",
    "src/training/steps/pre_training/interaction_feature_generator/data_driven_feature_selection/config.py",
)

# Provide a lightweight stub for the utils module to satisfy relative imports in
# ``phase2_rich_probes`` without triggering heavyweight dependencies.
utils_module_name = f"{_PACKAGE_NAME}.utils"
if utils_module_name not in sys.modules:
    utils_stub = types.ModuleType(utils_module_name)

    class _StubFeatureGeneratorWrapper:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class _StubUtilityEstimator:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    class _StubCostEstimator:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    utils_stub.FeatureGeneratorWrapper = _StubFeatureGeneratorWrapper
    utils_stub.UtilityEstimator = _StubUtilityEstimator
    utils_stub.CostEstimator = _StubCostEstimator
    sys.modules[utils_module_name] = utils_stub

phase2_module = _load_module(
    "phase2_rich_probes",
    "src/training/steps/pre_training/interaction_feature_generator/data_driven_feature_selection/phase2_rich_probes.py",
)


Phase2Config = config_module.Phase2Config
Phase2RichProbes = phase2_module.Phase2RichProbes


class DummyWrapper:
    """Minimal stand-in for a feature wrapper used during gating tests."""

    def __init__(self, name: str, stability: float):
        self.name = name
        self.family = "momentum"
        self.category = "momentum"
        self.phase2_utility = 0.1
        self.phase2_uncertainty = 0.5
        self.phase2_stability = stability


def _make_wrapper(name: str, stability: float) -> DummyWrapper:
    """Helper to build a wrapper with the desired stability score."""

    return DummyWrapper(name=name, stability=stability)


def test_apply_phase2_gating_rejects_features_with_excessive_sign_instability():
    """Wrappers with >30% sign flips (stability < 0.7) should be rejected."""

    config = Phase2Config(stability_threshold=0.7)
    phase2 = Phase2RichProbes(config)

    unstable_wrapper = _make_wrapper("unstable_feature", stability=0.69)

    selected, rejected = phase2._apply_phase2_gating([unstable_wrapper])

    assert unstable_wrapper in rejected
    assert unstable_wrapper not in selected


def test_apply_phase2_gating_accepts_wrappers_meeting_sign_stability_requirement():
    """Wrappers at or above the 70% stability threshold should pass gating."""

    config = Phase2Config(stability_threshold=0.7)
    phase2 = Phase2RichProbes(config)

    stable_wrapper = _make_wrapper("stable_feature", stability=0.7)

    selected, rejected = phase2._apply_phase2_gating([stable_wrapper])

    assert stable_wrapper in selected
    assert stable_wrapper not in rejected
