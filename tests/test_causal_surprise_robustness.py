import unittest
import numpy as np
import pandas as pd
from src.training.steps.labeling.causal_surprise_events import CausalSurpriseDetector

class TestCausalSurpriseRobustness(unittest.TestCase):
    def setUp(self):
        self.detector = CausalSurpriseDetector(verbose=False)
        self.n_samples = 100
        # Fix seed for reproducibility
        np.random.seed(42)
        self.targets = pd.Series(np.random.randn(self.n_samples))

    def test_normal_data(self):
        """Test with clean, finite data."""
        preds = pd.Series(np.random.randn(self.n_samples))
        self.detector.register_specialist("normal", preds, self.targets)
        meta = self.detector.specialist_metadata_["normal"]
        self.assertTrue(np.isfinite(meta["global_mad"]))
        self.assertGreater(meta["global_mad"], 0)

    def test_nan_data(self):
        """Test with data containing NaNs."""
        preds = pd.Series(np.random.randn(self.n_samples))
        preds.iloc[:10] = np.nan
        self.detector.register_specialist("nan_input", preds, self.targets)
        meta = self.detector.specialist_metadata_["nan_input"]
        self.assertTrue(np.isfinite(meta["global_mad"]), f"MAD should be finite, got {meta['global_mad']}")

    def test_inf_data(self):
        """Test with data containing Infinity."""
        preds = pd.Series(np.random.randn(self.n_samples))
        preds.iloc[:10] = np.inf
        self.detector.register_specialist("inf_input", preds, self.targets)
        meta = self.detector.specialist_metadata_["inf_input"]
        self.assertTrue(np.isfinite(meta["global_mad"]), f"MAD should be finite, got {meta['global_mad']}")

    def test_mixed_nan_inf(self):
        """Test with mixed NaN, Inf, and -Inf."""
        preds = pd.Series(np.random.randn(self.n_samples))
        preds.iloc[0:10] = np.nan
        preds.iloc[10:20] = np.inf
        preds.iloc[20:30] = -np.inf
        self.detector.register_specialist("mixed", preds, self.targets)
        meta = self.detector.specialist_metadata_["mixed"]
        self.assertTrue(np.isfinite(meta["global_mad"]), f"MAD should be finite, got {meta['global_mad']}")

    def test_all_non_finite(self):
        """Test with data containing only NaNs/Infs (should default)."""
        preds = pd.Series([np.nan] * self.n_samples)
        # Should not crash, should default
        self.detector.register_specialist("all_nan", preds, self.targets)
        meta = self.detector.specialist_metadata_["all_nan"]
        self.assertEqual(meta["global_mad"], 1.0)
        self.assertEqual(meta["mean_error"], 0.0)

    def test_compute_specialist_surprise_robustness(self):
        """Test robustness of rolling MAD calculation in surprise computation."""
        preds = pd.Series(np.random.randn(self.n_samples))
        # Insert a block of Infs in the middle to trigger rolling window issues
        preds.iloc[40:60] = np.inf

        self.detector.register_specialist("rolling_check", preds, self.targets)

        # This calls get_mad which we modified
        surprise = self.detector.compute_specialist_surprise("rolling_check", method="zscore")

        # Should return a series of finite values (NaNs replaced by 0 or valid scores)
        # Note: compute_specialist_surprise does .fillna(0) at the end
        self.assertTrue(np.all(np.isfinite(surprise)), "Surprise scores should be finite")
        self.assertEqual(len(surprise), self.n_samples)

if __name__ == '__main__':
    unittest.main()
