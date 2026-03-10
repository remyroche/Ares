import unittest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from extreme_price_movements.model_race import ModelRace, calculate_selection_score


class DummyCandidate(BaseEstimator, ClassifierMixin):
    def __init__(self, min_samples_leaf=50, min_child_weight=40):
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.classes_ = np.array([0, 1], dtype=np.int64)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        z = 1.0 / (1.0 + np.exp(-0.25 * X_arr[:, 0]))
        return np.column_stack([1.0 - z, z])


class TrackingModelRace(ModelRace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_leaf = []
        self.seen_child_weight = []

    def _get_candidates(self, race_mode=True):
        from extreme_price_movements.model_race import Float64Wrapper
        return {"stub": Float64Wrapper(DummyCandidate())}

    def _fit_model(self, model, X_tr, y_tr, X_val=None, y_val=None, sample_weight=None):
        inner = model.estimator if hasattr(model, "estimator") else model
        self.seen_leaf.append(getattr(inner, "min_samples_leaf", None))
        self.seen_child_weight.append(getattr(inner, "min_child_weight", None))
        model.fit(X_tr, y_tr, sample_weight=sample_weight)


class TestModelRace(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10

        self.X = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f"f_{i}" for i in range(self.n_features)]
        )

        # Target: related to f_0 + f_1
        logit = self.X["f_0"] + self.X["f_1"]
        prob = 1 / (1 + np.exp(-logit))
        self.y = (np.random.rand(self.n_samples) < prob).astype(int)

        # Returns: correlated with f_0 + noise
        self.returns = self.X["f_0"] * 0.1 + np.random.randn(self.n_samples) * 0.01

        # Sample weights
        self.weights = np.ones(self.n_samples, dtype=np.float32)

    def test_calculate_selection_score(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        y_ret = np.array([-0.01, -0.005, 0.005, 0.01])

        scores = calculate_selection_score(y_true, y_prob, y_ret)

        self.assertIn("Selection_Score", scores)
        self.assertIn("AUC", scores)
        self.assertIn("BSS", scores)
        self.assertIn("IC", scores)

        # AUC should be 1.0
        self.assertEqual(scores["AUC"], 1.0)
        # BSS should be positive
        self.assertGreater(scores["BSS"], 0.0)

    def test_model_race_fit_predict(self):
        race = ModelRace(kind="mr", n_splits=3)

        # Fit
        race.fit(self.X, self.y, sample_weight=self.weights, returns=self.returns)

        # Check winner
        self.assertIsNotNone(race.best_model_name)
        self.assertIsNotNone(race.best_model)

        # Predict
        preds = race.predict_proba(self.X)
        self.assertEqual(preds.shape, (self.n_samples, 2))

        # Ensure predictions are somewhat reasonable (not all same)
        self.assertTrue(np.std(preds[:, 1]) > 0.0)

    def test_fit_model_adjusts_class_weight_from_fold_prevalence(self):
        race = ModelRace(kind="mr", n_splits=2)
        model = race._get_candidates(race_mode=True)["extratrees"]
        inner = model.estimator if hasattr(model, "estimator") else model

        X = np.random.randn(100, 4)
        y = np.zeros(100, dtype=np.int8)
        y[:10] = 1

        race._fit_model(model, X, y)

        expected_pos_weight = (len(y) - y.sum()) / max(1, y.sum())
        self.assertIsInstance(inner, ExtraTreesClassifier)
        self.assertEqual(inner.class_weight, {0: 1.0, 1: expected_pos_weight})

    def test_model_race_dynamic_regularization_scales_with_positive_count(self):
        n = 25000
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"f0": np.linspace(-1.0, 1.0, n, dtype=np.float32)})
        y = np.zeros(n, dtype=np.int8)
        y[-22500:] = 1
        returns = X["f0"].to_numpy(dtype=np.float64)
        perm = rng.permutation(n)
        X = X.iloc[perm].reset_index(drop=True)
        y = y[perm]
        returns = returns[perm]

        race = TrackingModelRace(kind="mr", n_splits=2, max_label_horizon_hours=1)
        race.fit(X, y, returns=returns)

        self.assertTrue(len(race.seen_leaf) > 0)
        self.assertTrue(len(race.seen_child_weight) > 0)
        self.assertIn(225, [v for v in race.seen_leaf if v is not None])
        self.assertIn(57, [v for v in race.seen_child_weight if v is not None])

if __name__ == '__main__':
    unittest.main()
