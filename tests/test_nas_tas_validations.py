import pytest

np = pytest.importorskip("numpy")

from src.utils.nas_tas.data_processing import (
    DataProcessingError,
    DataProcessorConfigurationError,
    DataSplitError,
    UnifiedDataProcessor,
)
from src.utils.nas_tas.unified_search_engine import SearchConfig


class _FailingPreprocessor:
    def preprocess_data(self, _):  # pragma: no cover - simple failure stub
        raise RuntimeError("forced failure")


def test_unified_data_processor_validates_configuration():
    with pytest.raises(DataProcessorConfigurationError):
        UnifiedDataProcessor({'validation_split': 1.0})


def test_unified_data_processor_raises_on_processing_failure():
    processor = UnifiedDataProcessor({'validation_split': 0.2}, preprocessor=_FailingPreprocessor())
    X = np.ones((4, 2))
    y = np.zeros(4)

    with pytest.raises(DataProcessingError):
        processor.process_data(X, y)


def test_unified_data_processor_raises_on_split_failure(monkeypatch):
    processor = UnifiedDataProcessor({'validation_split': 0.2}, preprocessor=_FailingPreprocessor())

    def _fail(*args, **kwargs):  # pragma: no cover - testing failure path
        raise RuntimeError("split failure")

    monkeypatch.setattr('src.utils.nas_tas.data_processing.train_test_split', _fail)

    X = np.ones((10, 2))
    y = np.zeros(10)

    with pytest.raises(DataSplitError):
        processor.split_data(X, y)


def test_search_config_rejects_invalid_weights():
    with pytest.raises(ValueError):
        SearchConfig(objective_weights=[0.5, 0.4, 0.1 + 1e-3])


def test_search_config_enforces_population_constraints():
    with pytest.raises(ValueError):
        SearchConfig(population_size=5, max_candidates_per_batch=10)


def test_search_config_requires_valid_objectives():
    with pytest.raises(TypeError):
        SearchConfig(objectives=["accuracy"])  # type: ignore[list-item]
