import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

mock_torch = types.ModuleType("torch")
mock_torch.nn = types.ModuleType("torch.nn")
mock_torch.optim = types.ModuleType("torch.optim")
sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("torch.nn", mock_torch.nn)
sys.modules.setdefault("torch.optim", mock_torch.optim)

MODULE_PATH = Path(__file__).resolve().parents[5] / "src/training/steps/market_analysis/tas_regime/search/advanced_search.py"
spec = importlib.util.spec_from_file_location("advanced_tas_search", MODULE_PATH)
advanced_search_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(advanced_search_module)

AdvancedSearchConfig = advanced_search_module.AdvancedSearchConfig
AdvancedTASSearch = advanced_search_module.AdvancedTASSearch


def _build_sample_data(num_rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2022-01-01", periods=num_rows, freq="D")
    feature_a = np.linspace(0.0, 1.0, num_rows)
    feature_b = np.linspace(1.0, 0.0, num_rows)
    feature_c = np.sin(np.linspace(0.0, 3.14, num_rows))
    market_data = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
        },
        index=index,
    )
    target = pd.Series(0.6 * feature_a + 0.3 * feature_b + 0.1 * feature_c, index=index)
    return market_data, target


def test_search_returns_best_architecture_and_score():
    np.random.seed(0)
    market_data, target = _build_sample_data()
    config = AdvancedSearchConfig(
        n_iterations=5,
        population_size=6,
        elite_size=2,
        mutation_rate=0.0,
        crossover_rate=0.0,
    )
    search_space = {
        "scaling_factor": [0.5, 1.0, 1.5],
        "feature_subset": [1, 2, 3],
        "regularization_strength": [0.0, 0.02],
    }

    search_engine = AdvancedTASSearch(config)
    result = search_engine.search(
        market_data=market_data,
        target_returns=target,
        market_regimes={"Bull": {"confidence": 0.8}},
        micro_regimes={},
        architecture_type="test",
        search_space=search_space,
    )

    assert "best_architecture" in result
    assert isinstance(result["best_architecture"], dict)
    assert result["best_architecture"], "best architecture should not be empty"
    assert result["best_score"] == pytest.approx(search_engine.best_score)
    assert result["best_architecture"] == search_engine.best_individual
    assert len(result["best_score_history"]) >= 1
    assert result["generations_evaluated"] > 0

    # Verify that the reported best score matches manual evaluation
    manual_score = search_engine._evaluate_individual(result["best_architecture"])
    assert manual_score == pytest.approx(result["best_score"])


def test_search_requires_market_data_and_targets():
    search_engine = AdvancedTASSearch(AdvancedSearchConfig(n_iterations=1, population_size=2))

    with pytest.raises(ValueError):
        search_engine.search(market_data=None, target_returns=pd.Series(dtype=float))

    market_data, _ = _build_sample_data()
    with pytest.raises(ValueError):
        search_engine.search(market_data=market_data, target_returns=None)
