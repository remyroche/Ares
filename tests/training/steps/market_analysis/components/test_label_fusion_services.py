import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PACKAGE_SPECS = [
    ("src", ROOT_DIR / "src"),
    ("src.training", ROOT_DIR / "src/training"),
    ("src.training.steps", ROOT_DIR / "src/training/steps"),
    ("src.training.steps.market_analysis", ROOT_DIR / "src/training/steps/market_analysis"),
    (
        "src.training.steps.market_analysis.regime_analysis",
        ROOT_DIR / "src/training/steps/market_analysis/regime_analysis",
    ),
]

for name, path in PACKAGE_SPECS:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

MODULE_NAME = "src.training.steps.market_analysis.regime_analysis.label_fusion"
MODULE_PATH = ROOT_DIR / "src/training/steps/market_analysis/regime_analysis/label_fusion.py"

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
assert spec.loader is not None
spec.loader.exec_module(module)

LabelFusionService = module.LabelFusionService
RegimeOptimizationService = module.RegimeOptimizationService


def silent_logger(*args, **kwargs):
    pass


def simple_score(features: np.ndarray, assignments: np.ndarray) -> float:
    return float(np.mean(assignments))


class DummyHMM:
    def __init__(self, predicted: np.ndarray, transmat: np.ndarray):
        self._predicted = predicted
        self.transmat_ = transmat
        self.n_components = transmat.shape[0]
        self.fit_calls = 0

    def fit(self, features: np.ndarray) -> None:
        self.fit_calls += 1

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._predicted


def test_map_labels_handles_abstain_mapping():
    service = LabelFusionService(logger=silent_logger)
    tas = np.array([0, 1, 5, -1])
    nas = np.array([1, 2, 7, 0])

    tas_mapped, nas_mapped, info = service.map_labels_to_k_space(tas, nas, target_k=3)

    assert info["mapping_needed"] is True
    assert info["method"] == "abstain_column"
    assert info["abstain_value"] == 3
    assert set(tas_mapped.tolist()) == {0, 1, 3}
    assert set(nas_mapped.tolist()) == {0, 1, 2, 3}


def test_dawid_skene_converges_on_consistent_labels():
    rng = np.random.default_rng(42)
    n_samples = 60
    features = np.vstack(
        [
            rng.normal(0, 0.1, size=(n_samples // 2, 3)),
            rng.normal(2, 0.1, size=(n_samples // 2, 3)),
        ]
    )
    true_labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    tas = true_labels.copy()
    nas = true_labels.copy()

    noise_idx = rng.choice(n_samples, size=6, replace=False)
    tas[noise_idx[:3]] = 1 - tas[noise_idx[:3]]
    nas[noise_idx[3:]] = 1 - nas[noise_idx[3:]]

    service = LabelFusionService(logger=silent_logger)
    result = service.run_dawid_skene(tas, nas, target_k=2, features=features)

    accuracy = np.mean(result.assignments == true_labels)
    assert accuracy >= 0.8
    assert result.metadata["converged"] is True
    assert len(result.metadata["log_likelihoods"]) <= 50


def test_apply_hmm_smoothing_uses_model_metadata():
    label_service = LabelFusionService(logger=silent_logger)
    regime_service = RegimeOptimizationService(label_service, simple_score, logger=silent_logger)

    assignments = np.array([0, 1, 1, 0])
    features = np.random.normal(size=(4, 2))

    predicted = np.array([0, 0, 1, 1])
    transmat = np.array([[0.7, 0.3], [0.2, 0.8]])
    dummy_model = DummyHMM(predicted=predicted, transmat=transmat)

    with patch.object(regime_service, "_initialize_hmm", return_value=dummy_model):
        smoothed, metadata = regime_service.apply_hmm_smoothing(features, assignments)

    assert np.array_equal(smoothed, predicted)
    assert metadata["method"] == "hmm"
    assert metadata["transmat"] == transmat.tolist()
    assert metadata["changed_points"] == [1, 3]
    assert dummy_model.fit_calls == 1


def test_apply_hmm_smoothing_fallback_corrects_isolated_point():
    label_service = LabelFusionService(logger=silent_logger)
    regime_service = RegimeOptimizationService(label_service, simple_score, logger=silent_logger)

    assignments = np.array([0, 0, 1, 0, 0])
    features = np.random.normal(size=(5, 2))

    with patch.object(regime_service, "_initialize_hmm", side_effect=RuntimeError("boom")):
        smoothed, metadata = regime_service.apply_hmm_smoothing(features, assignments)

    assert metadata["method"] == "simple_fallback"
    assert smoothed[2] == 0
