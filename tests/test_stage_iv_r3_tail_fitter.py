from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iv_r3_tail_fitter import (
    _R3OpportunityPredictor,
    StageIVR3FitterError,
    current_r3_class,
    current_r3_tree_params,
)


class _ThreeClassProbabilityModel:
    classes_ = np.array([0, 1, 2], dtype=np.int8)

    def predict_proba(self, X):
        return np.tile(np.array([[0.2, 0.3, 0.5]], dtype=np.float32), (len(X), 1))


def test_current_r3_classes_preserve_adverse_weak_clear_and_invalid() -> None:
    result = current_r3_class(
        robust_clear_event=np.array([0.0, 0.0, 1.0, np.nan]),
        lower_touch_minute=np.array([2.0, -1.0, 3.0, np.nan]),
        label_valid=np.array([True, True, True, False]),
    )
    assert result.tolist() == [0, 1, 2, -1]


def test_current_r3_target_rejects_missing_valid_path_input() -> None:
    with pytest.raises(StageIVR3FitterError, match="valid R3 rows"):
        current_r3_class(
            robust_clear_event=np.array([np.nan]),
            lower_touch_minute=np.array([-1.0]),
            label_valid=np.array([True]),
        )


def test_current_r3_tree_params_bind_multiclass_semantics() -> None:
    assert current_r3_tree_params({"objective": "multiclass", "num_class": 3, "num_leaves": 16}) == {"num_leaves": 16}
    with pytest.raises(StageIVR3FitterError, match="three-class R3"):
        current_r3_tree_params({"objective": "regression"})


def test_current_r3_context_preserves_simplex_and_invariant_confidence() -> None:
    predictor = _R3OpportunityPredictor(_ThreeClassProbabilityModel())
    context = predictor.predict_context(pd.DataFrame({"x": [1.0, 2.0]}))
    assert context.columns.tolist() == [
        "r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score",
        "base_r3_entropy", "base_r3_top2_margin", "base_r3_max_probability",
    ]
    assert np.allclose(context[["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].sum(axis=1), 1.0)
    assert np.allclose(context.r3_opportunity_score, 0.3)
    assert np.allclose(context.base_r3_top2_margin, 0.2)
