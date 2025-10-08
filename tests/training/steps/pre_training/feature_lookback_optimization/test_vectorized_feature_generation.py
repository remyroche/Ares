import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_CORE_OPTIMIZER_PATH = (
    Path(__file__)
    .resolve()
    .parents[5]
    / "src"
    / "training"
    / "steps"
    / "pre_training"
    / "feature_lookback_optimization"
    / "core"
    / "optimizer.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "src.training.steps.pre_training.feature_lookback_optimization.core.optimizer",
    _CORE_OPTIMIZER_PATH,
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
CoreOptimizer = _MODULE.CoreOptimizer


def test_vectorized_feature_generation_enforces_minimum_lag() -> None:
    optimizer = CoreOptimizer()
    data = pd.DataFrame({'close': np.arange(1, 21, dtype=float)})

    features = optimizer._vectorized_feature_generation(data, 'returns', [1, 3])

    assert features, "Expected vectorized feature generation to return results"
    assert 'returns' in optimizer.feature_lag_metadata

    for horizon, values in features.items():
        assert np.isnan(values[0]), f"First value for horizon {horizon} should be NaN after lag enforcement"
        metadata = optimizer.feature_lag_metadata['returns'][horizon]
        assert metadata['max_lag'] >= 1
        assert metadata['has_leading_nulls'] is True


def test_assert_lag_requirements_raises_on_contemporaneous_values() -> None:
    optimizer = CoreOptimizer()

    with pytest.raises(ValueError):
        optimizer._assert_lag_requirements('test_feature', 1, np.ones(5, dtype=float))
