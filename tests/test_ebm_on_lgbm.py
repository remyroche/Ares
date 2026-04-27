import numpy as np
import pandas as pd

import extreme_price_movements.ebm_on_lgbm as lor
from extreme_price_movements.ebm_on_lgbm import (
    SplinePostProcessor,
    _feature_shape_scores,
    _prescreen_features,
    _select_smallest_within_one_se,
    train_ebm_on_lgbm_candidate,
)


class FakeEBMClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.coef_ = None
        self.term_scores_ = []
        self.term_features_ = []

    def fit(self, X, y, sample_weight=None):
        x = np.asarray(X, dtype=np.float32)
        yy = np.asarray(y, dtype=np.float32)
        yy = yy - float(np.mean(yy))
        denom = np.std(x, axis=0) * max(float(np.std(yy)), 1e-6)
        coef = np.where(
            denom > 1e-9,
            np.mean((x - np.mean(x, axis=0)) * yy[:, None], axis=0) / denom,
            0.0,
        )
        self.coef_ = np.nan_to_num(coef, nan=0.0).astype(np.float32)
        self.term_features_ = [(i,) for i in range(x.shape[1])]
        grid = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        self.term_scores_ = [grid * float(c) for c in self.coef_]
        return self

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        z = x @ self.coef_
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        return np.column_stack([1.0 - p, p]).astype(np.float32)


class FakeEBMRegressor(FakeEBMClassifier):
    def predict(self, X):
        return np.asarray(X, dtype=np.float32) @ self.coef_


def test_spline_postprocessor_identity_fallback_is_finite():
    pp = SplinePostProcessor(mode="classifier").fit(
        np.ones(12, dtype=np.float32),
        np.array([0, 1] * 6, dtype=np.float32),
    )

    out = pp.predict(np.array([0.2, 0.5, 0.8], dtype=np.float32))

    assert pp.identity is True
    assert np.all(np.isfinite(out))
    assert np.all((out > 0.0) & (out < 1.0))


def test_feature_shape_scores_are_finite_non_negative():
    m1 = FakeEBMClassifier().fit(
        pd.DataFrame(np.random.default_rng(1).normal(size=(40, 3))),
        np.array([0, 1] * 20),
    )
    m2 = FakeEBMClassifier().fit(
        pd.DataFrame(np.random.default_rng(2).normal(size=(40, 3))),
        np.array([0, 1] * 20),
    )

    scores = _feature_shape_scores([m1, m2], ["a", "b", "c"])

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_feature_shape_scores_zero_out_negative_shape_correlation():
    class OppositeShapeModel:
        term_features_ = [(0,)]

        def __init__(self, sign):
            self.term_scores_ = [sign * np.linspace(-1.0, 1.0, 16)]

    scores = _feature_shape_scores(
        [OppositeShapeModel(1.0), OppositeShapeModel(-1.0)], ["x"]
    )

    assert scores.shape == (1,)
    assert scores[0] == 0.0


def test_select_smallest_within_one_se_prefers_smaller_model():
    history = [
        {"round": 1, "J_final": 1.00, "J_se": 0.05, "n_features_end": 90},
        {"round": 2, "J_final": 0.97, "J_se": 0.04, "n_features_end": 55},
        {"round": 3, "J_final": 0.90, "J_se": 0.03, "n_features_end": 30},
    ]

    chosen = _select_smallest_within_one_se(history)

    assert chosen["round"] == 2


def test_prescreen_features_reduces_to_configured_cap(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(180, 24)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.2, size=180) > 0).astype(np.int8)
    names = [f"f{i}" for i in range(x.shape[1])]
    monkeypatch.setattr(lor, "EBM_PRESCREEN_MAX_FEATURES", 10)
    monkeypatch.setattr(lor, "EBM_MIN_FEATURES", 4)

    active = _prescreen_features(x, y, names, classifier=True, random_state=42)

    assert 1 <= len(active) <= 10
    assert active.dtype == np.int32


def test_train_ebm_on_lgbm_candidate_with_fake_ebm(monkeypatch):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(240, 12)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(scale=0.3, size=240) > 0).astype(np.int8)
    X = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    monkeypatch.setattr(
        lor, "_load_ebm_classes", lambda: (FakeEBMClassifier, FakeEBMRegressor)
    )
    monkeypatch.setattr(lor, "EBM_MAX_ROUNDS", 2)
    monkeypatch.setattr(lor, "EBM_PRESCREEN_MAX_FEATURES", 10)
    monkeypatch.setattr(lor, "EBM_MIN_FEATURES", 4)
    monkeypatch.setattr(lor, "EBM_FOLD_SUBSAMPLE_ROWS", 80)

    result = train_ebm_on_lgbm_candidate(X, y, random_state=42, mode="classifier")

    assert result is not None
    assert result["full_fit_needed"] is True
    assert np.isfinite(result["oof_probs"]).sum() > 20
    assert result["selected_features_from_cv"].dtype == np.int32
    assert result["metrics"]["feature_count"] > 0
