import importlib.util
import sys
import warnings

import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "src" / "utils" / "ml_common" / "optimization" / "tas" / "models" / "enhanced_tree_models.py"

spec = importlib.util.spec_from_file_location("enhanced_tree_models", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load enhanced_tree_models module for testing")

etm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = etm
spec.loader.exec_module(etm)


@pytest.mark.skipif(not etm.SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_cross_validate_produces_scores_without_warnings():
    config = etm.TreeModelConfig(
        model_type="bart",
        task_type="regression",
        bart_n_trees=5,
        max_depth=3,
        n_jobs=1,
        random_state=0,
    )

    model = etm.EnhancedTreeModelFactory.create_model(config)
    evaluator = etm.TreeModelEvaluator(task_type="regression")

    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        scores = evaluator._cross_validate(model, X, y, cv=3)

    assert not recorded_warnings, "Expected no warnings during cross-validation"
    assert isinstance(scores, list) and len(scores) == 3
    assert all(np.isfinite(scores))
