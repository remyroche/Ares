import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

import src.training.steps.labeling.label_based_layer_2 as label_layer


def test_huber_residual_stack_check_array_fallback(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 5))
    y = (X[:, 0] + rng.normal(scale=0.1, size=40) > 0).astype(float)

    real_check_array = label_layer.check_array

    def patched_check_array(*args, **kwargs):
        if "ensure_all_finite" in kwargs:
            raise TypeError("unexpected keyword argument 'ensure_all_finite'")
        return real_check_array(*args, **kwargs)

    monkeypatch.setattr(label_layer, "check_array", patched_check_array)

    model = label_layer.HuberResidualStack(
        huber_params={
            "loss_type": "huber",
            "huber_epsilon": 1.1,
            "alpha": 1.0,
            "irm_lambda": 1.0,
            "max_iter": 50,
        },
        fashion_estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),
        n_splits=3,
    )

    model.fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (40, 2)
    assert np.isfinite(probs).all()
