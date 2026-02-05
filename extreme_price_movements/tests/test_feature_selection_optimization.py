
import unittest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

# Import the module under test
from extreme_price_movements.feature_selection_extreme_events import (
    purged_embargoed_splits,
    mdi_feature_selection_v3
)

class MockModel(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.estimators_ = []
        self.feature_importances_ = None

    def fit(self, X, y, sample_weight=None):
        # Create a dummy forest with one tree for testing extraction
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.random.rand(self.n_features_in_)
        self.feature_importances_ /= self.feature_importances_.sum()

        # Create a real DecisionTree to have valid tree_ structure
        dt = DecisionTreeClassifier(random_state=self.random_state, max_depth=3)
        dt.fit(X, y, sample_weight=sample_weight)
        self.estimators_ = [dt]
        return self

    def predict(self, X):
        return np.zeros(len(X))

class TestFeatureSelectionOptimization(unittest.TestCase):
    def test_purged_embargoed_splits_embargo(self):
        n_samples = 100
        n_splits = 2
        purge = 5
        embargo = 10

        splits = purged_embargoed_splits(n_samples, n_splits, purge=purge, embargo=embargo)

        # Check first split: val [0, 50)
        train_idx, val_idx = splits[0]
        val_end = val_idx[-1] + 1 # 50

        # Expect training data after embargo: [60, 100)
        # Current code returns empty train_idx because it ignores post-validation data

        # Assert we have data
        self.assertTrue(len(train_idx) > 0, "First split should have training data from post-embargo period")

        # Check embargo respect
        embargo_violation = np.any((train_idx >= val_end) & (train_idx < val_end + embargo))
        self.assertFalse(embargo_violation, "Embargo period violated in training set")

        # Check post-embargo presence
        has_post_embargo = np.any(train_idx >= val_end + embargo)
        self.assertTrue(has_post_embargo, "Training set should include post-embargo data")

    def test_mdi_feature_selection_v3_run(self):
        # synthetic data
        N = 200
        P = 10
        X = pd.DataFrame(np.random.randn(N, P), columns=[f"f{i}" for i in range(P)])
        y = pd.Series(np.random.randint(0, 2, N))

        # Ensure some correlation for dedupe testing
        X['f1'] = X['f0'] * 0.99 + np.random.normal(0, 0.01, N)

        base_model = MockModel(random_state=42)

        result = mdi_feature_selection_v3(
            X, y,
            base_model=base_model,
            n_splits=2,
            purge=2,
            analysis_n_estimators=2, # Keep it fast
            end_features=5
        )

        self.assertIsInstance(result.metrics_table, pd.DataFrame)
        self.assertIsInstance(result.selected_features, list)
        self.assertIsInstance(result.kept_after_dedupe, list)

        # Check float32 dtype
        if not result.metrics_table.empty:
             self.assertTrue(result.metrics_table['share_mu'].dtype == np.float32 or result.metrics_table['share_mu'].dtype == 'float32', "Metrics should be float32")

if __name__ == "__main__":
    unittest.main()
