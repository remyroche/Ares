
import unittest
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
# Suppress LightGBM verbose warnings
warnings.filterwarnings("ignore")

from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2, RobustFocalLoss

class TestFeatureSelectionModels(unittest.TestCase):
    def setUp(self):
        self.layer = LabelBasedLayer2(verbose=False)

    def test_rfe_model_instantiation(self):
        """Verify that _run_titan_rfe runs with the custom objective."""
        # Create dummy data
        X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 100))
        vol = pd.Series(np.random.rand(100) * 0.02 + 0.01, index=X.index)

        # Mocking the split generator
        splits = [([0, 1, 2], [3, 4])] * 2

        # We can't easily mock the internal lgb.LGBMClassifier inside _run_titan_rfe
        # without dependency injection or heavy patching.
        # But we can try running it with very minimal settings to ensure it doesn't crash
        # due to the custom objective.

        try:
            # We set min_features high to force it to return quickly or loop once
            selected = self.layer._run_titan_rfe(X, y, splits, vol, min_features=8)
            self.assertTrue(len(selected) > 0)
        except Exception as e:
            self.fail(f"_run_titan_rfe crashed with custom objective: {e}")

    def test_robust_focal_loss(self):
        """Verify RobustFocalLoss callable structure matches expectation."""
        obj = RobustFocalLoss(gamma_pos=0.5, gamma_neg=1.25, alpha=0.65)

        # Test call with arrays
        preds = np.array([0.1, 0.9])
        labels = np.array([0, 1])

        grad, hess = obj(preds, labels)

        self.assertEqual(grad.shape, preds.shape)
        self.assertEqual(hess.shape, preds.shape)

        # Test basic property: hessian should be positive
        self.assertTrue(np.all(hess >= 0))

if __name__ == '__main__':
    unittest.main()
