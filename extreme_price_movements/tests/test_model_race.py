import unittest
import numpy as np
import pandas as pd
from extreme_price_movements.model_race import ModelRace, calculate_selection_score

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

if __name__ == '__main__':
    unittest.main()
