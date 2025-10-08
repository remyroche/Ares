import asyncio
import os
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest


if "src.training.steps.models_training.negative_learning_training_integration" not in sys.modules:
    integration_stub = types.ModuleType(
        "src.training.steps.models_training.negative_learning_training_integration"
    )
    integration_stub.initialize_negative_learning_integration = lambda *args, **kwargs: None
    integration_stub.get_negative_learning_integration = lambda: None
    sys.modules[
        "src.training.steps.models_training.negative_learning_training_integration"
    ] = integration_stub

if "src.training.steps.models_training.negative_learning_training_patches" not in sys.modules:
    patches_stub = types.ModuleType(
        "src.training.steps.models_training.negative_learning_training_patches"
    )

    def _apply_negative_learning_patches():
        return None

    patches_stub.apply_negative_learning_patches = _apply_negative_learning_patches
    sys.modules[
        "src.training.steps.models_training.negative_learning_training_patches"
    ] = patches_stub

if "src.training.steps.models_training.nas_tas.training_orchestrator" not in sys.modules:
    orchestrator_stub = types.ModuleType(
        "src.training.steps.models_training.nas_tas.training_orchestrator"
    )

    class _DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            self.training_result = None

        async def orchestrate_async(self, **kwargs):
            return types.SimpleNamespace(
                success=False,
                training_result=None,
                error=RuntimeError("stub"),
                warnings=["stub"],
                metrics={},
                execution_time=0.0,
            )

    class _DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyMode:
        TRAINING_ONLY = "training_only"

    orchestrator_stub.TrainingOrchestrator = _DummyOrchestrator
    orchestrator_stub.OrchestratorConfig = _DummyConfig
    orchestrator_stub.OrchestrationMode = _DummyMode
    sys.modules[
        "src.training.steps.models_training.nas_tas.training_orchestrator"
    ] = orchestrator_stub

from src.training.steps.models_training.analyst_models_training import AnalystModelsTrainingStep
from src.training.steps.models_training.tactician_models_training import TacticianModelsTrainingStep


def test_analyst_stacker_requires_oof():
    step = AnalystModelsTrainingStep()
    X = np.random.rand(30, 4)
    y = np.random.randint(0, 2, size=(30, 1))
    sample_weight = np.ones(30)

    with pytest.raises(ValueError):
        asyncio.run(step._train_stacker_lgbm_calibrated(X, y, sample_weight))


def test_tactician_stacker_requires_oof():
    step = TacticianModelsTrainingStep()
    X = np.random.rand(30, 5)
    y = np.random.randint(0, 2, size=(30, 1))
    sample_weight = np.ones(30)

    with pytest.raises(ValueError):
        asyncio.run(step._train_stacker_lgbm_calibrated(X, y, sample_weight))


def test_stacker_calibration_with_shallow_tree(tmp_path):
    step = AnalystModelsTrainingStep()
    X = np.random.rand(80, 6)
    y = np.random.randint(0, 2, size=(80, 1))
    sample_weight = np.ones(80)

    base_model_oof_predictions = {
        'lgbm': np.random.rand(80),
        'catboost': np.random.rand(80),
    }

    class _CalibratedStub:
        def __init__(self, base_estimator, method="isotonic", cv=5):
            self.base_estimator = base_estimator
            self.method = method
            self.cv = cv
            self.fitted = False

        def fit(self, X, y):
            self.fitted = True
            return self

        def predict_proba(self, X):
            preds = np.clip(self.base_estimator.regressor.predict(X), 0, 1)
            return np.column_stack([1 - preds, preds])

    with patch("src.models.stacker_lgbm_calibrated.CalibratedClassifierCV", _CalibratedStub):
        result = asyncio.run(
            step._train_stacker_lgbm_calibrated(
                X,
                y,
                sample_weight,
                base_model_oof_predictions=base_model_oof_predictions,
                output_directory=str(tmp_path),
            )
        )

    model = result['models']['stacker_lgbm_calibrated']
    assert model.calibrated_model is not None

    params = model.lgbm_model.get_params()
    assert params['max_depth'] == 2
    assert params['num_leaves'] <= 4

    meta_oof_path = result['metrics']['meta_oof_path']
    assert os.path.exists(meta_oof_path)

    saved_meta = np.load(meta_oof_path)
    assert saved_meta.shape[0] == X.shape[0]
    assert np.array_equal(saved_meta, np.array(result['meta_oof_predictions']))
