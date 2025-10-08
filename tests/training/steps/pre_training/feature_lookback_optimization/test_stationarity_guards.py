import importlib
import importlib.machinery as machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

import pytest


def _ensure_stub_package(package: str, path: Path) -> None:
    if package in sys.modules:
        return

    module = types.ModuleType(package)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__spec__ = machinery.ModuleSpec(package, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [str(path)]
    sys.modules[package] = module


ROOT = Path(__file__).resolve().parents[5]

_ensure_stub_package("src", ROOT / "src")
_ensure_stub_package("src.training", ROOT / "src/training")
_ensure_stub_package("src.training.steps", ROOT / "src/training/steps")
_ensure_stub_package("src.training.steps.pre_training", ROOT / "src/training/steps/pre_training")
_ensure_stub_package(
    "src.training.steps.pre_training.feature_lookback_optimization",
    ROOT / "src/training/steps/pre_training/feature_lookback_optimization",
)
_ensure_stub_package(
    "src.training.steps.pre_training.feature_lookback_optimization.core",
    ROOT / "src/training/steps/pre_training/feature_lookback_optimization/core",
)

def _load_module(module_name: str, relative_path: Path, package: str):
    spec = importlib.util.spec_from_file_location(module_name, relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


optimization_strategy = _load_module(
    "src.training.steps.pre_training.feature_lookback_optimization.optimization_strategy",
    ROOT / "src/training/steps/pre_training/feature_lookback_optimization/optimization_strategy.py",
    "src.training.steps.pre_training.feature_lookback_optimization",
)
core_optimizer_module = _load_module(
    "src.training.steps.pre_training.feature_lookback_optimization.core.optimizer",
    ROOT / "src/training/steps/pre_training/feature_lookback_optimization/core/optimizer.py",
    "src.training.steps.pre_training.feature_lookback_optimization.core",
)

GridSearchStrategy = optimization_strategy.GridSearchStrategy
CoreOptimizer = core_optimizer_module.CoreOptimizer
OptimizationMethod = core_optimizer_module.OptimizationMethod


def _build_trending_frame(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="h")
    close = 100.0 * np.exp(np.linspace(0.0, 0.5, rows))
    target = pd.Series(close, index=index).pct_change().shift(-1).fillna(0.0)
    return pd.DataFrame({
        "close": close,
        "target": target,
    }, index=index)


def test_grid_search_stationary_fallback_uses_transformed_close():
    data = _build_trending_frame()
    strategy = GridSearchStrategy({"min_lookback": 5, "max_lookback": 10})

    transformed = strategy._generate_feature_with_lookback(data, "missing_feature", 10)
    raw_mean = data["close"].rolling(window=10).mean().values

    assert not np.allclose(np.nan_to_num(transformed), np.nan_to_num(raw_mean))


def test_mrmr_stationarity_guard_transforms_trending_features(monkeypatch: pytest.MonkeyPatch):
    data = _build_trending_frame()

    monkeypatch.setattr(CoreOptimizer, "_create_feature_generator", lambda self, feature_name, lookback: None)

    optimizer = CoreOptimizer()
    result = optimizer.optimize_single_feature(
        data=data,
        feature_name="close",
        target_column="target",
        method=OptimizationMethod.MRMR,
        lookback_range=(5, 6),
    )

    audit = result.metadata.get("stationarity_audit")
    assert audit, "Expected stationarity audit metadata to be populated"

    assert any(
        entry["train"].get("transformed") or entry["test"].get("transformed")
        for entry in audit.values()
    ), "Trending features should require a stationary transform"

    assert result.metadata.get("non_stationary_lookbacks", 0) >= 1
