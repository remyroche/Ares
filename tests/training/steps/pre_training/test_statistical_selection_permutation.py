import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[4]
PACKAGE_PATHS = {
    "src": ROOT / "src",
    "src.training": ROOT / "src" / "training",
    "src.training.steps": ROOT / "src" / "training" / "steps",
    "src.training.steps.pre_training": ROOT / "src" / "training" / "steps" / "pre_training",
    "src.training.steps.pre_training.interaction_feature_generator": ROOT
    / "src"
    / "training"
    / "steps"
    / "pre_training"
    / "interaction_feature_generator",
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation": ROOT
    / "src"
    / "training"
    / "steps"
    / "pre_training"
    / "interaction_feature_generator"
    / "cross_timeframe_generation",
}

for name, path in PACKAGE_PATHS.items():
    if name not in sys.modules:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

MODULE_PATH = PACKAGE_PATHS[
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation"
] / "statistical_selection.py"

spec = importlib.util.spec_from_file_location(
    "src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.statistical_selection",
    MODULE_PATH,
)
statistical_selection = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = statistical_selection
assert spec.loader is not None
spec.loader.exec_module(statistical_selection)

PermutationTester = statistical_selection.PermutationTester


@pytest.fixture
def base_series():
    index = pd.RangeIndex(60)
    features = pd.DataFrame({"feat": np.arange(60, dtype=float)}, index=index)
    targets = pd.Series(np.linspace(0.0, 1.0, 60), index=index)
    return features, targets


def test_permutation_blocks_respect_validation_window(monkeypatch, base_series):
    features, targets = base_series
    config = SimpleNamespace(
        permutation_block_size=4,
        permutation_n_permutations=5,
        permutation_cv_folds=5,
        permutation_random_state=None,
        label_horizon=1,
    )
    tester = PermutationTester(config)

    original_block_permute = tester._block_permute
    observed = []

    def wrapped(values, block_size, rng):
        permuted = original_block_permute(values, block_size, rng)
        observed.append((values.copy(), permuted.copy()))
        # All permuted values must remain within the validation window bounds
        assert permuted.min() >= values.min()
        assert permuted.max() <= values.max()
        return permuted

    monkeypatch.setattr(tester, "_block_permute", wrapped)

    splitter = TimeSeriesSplit(n_splits=4)
    tester.calculate_p_values(
        features,
        targets,
        cv_splitter=splitter,
        block_size=4,
        n_permutations=5,
        random_state=42,
    )

    assert observed, "Expected block permutations to be executed"


def test_block_size_impacts_permutation_results():
    index = pd.RangeIndex(120)
    base_signal = np.sin(np.linspace(0, 8 * np.pi, 120))
    targets = pd.Series(base_signal, index=index)
    features = pd.DataFrame({"feat": np.roll(base_signal, -1)}, index=index)

    config = SimpleNamespace(
        permutation_block_size=None,
        permutation_n_permutations=40,
        permutation_cv_folds=4,
        permutation_random_state=None,
        label_horizon=1,
    )
    tester = PermutationTester(config)
    splitter = TimeSeriesSplit(n_splits=4)

    small_block = tester.calculate_p_values(
        features,
        targets,
        cv_splitter=splitter,
        block_size=1,
        n_permutations=40,
        random_state=0,
    )

    large_block = tester.calculate_p_values(
        features,
        targets,
        cv_splitter=splitter,
        block_size=12,
        n_permutations=40,
        random_state=0,
    )

    assert small_block["feat"] < large_block["feat"], "Block size should influence p-values"
