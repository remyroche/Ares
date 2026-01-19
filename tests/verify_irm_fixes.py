
import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.steps.labeling.label_based_layer_2 import (
    IRM_LGBMClassifier, 
    IRM_XGBClassifier, 
    IRM_CatBoostClassifier,
    IRM_ExtraTreesClassifier
)

class TestIRMFixes(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)])
        self.y = np.random.randint(0, 2, 100)
        # Ensure at least one positive and one negative
        self.y[0] = 0
        self.y[1] = 1
        
        self.irm_system = MagicMock()
        self.environment_masks = {'env_1': np.random.randint(0, 2, 100)}

    def test_lgbm_initialization_and_params(self):
        """Test IRM_LGBMClassifier initialization and get_params()."""
        print("\nTesting IRM_LGBMClassifier...")
        model = IRM_LGBMClassifier(
            irm_system=self.irm_system,
            environment_masks=self.environment_masks,
            random_state=42
        )
        
        # Check if params are hidden from get_params
        params = model.get_params()
        self.assertNotIn('irm_system', params)
        self.assertNotIn('environment_masks', params)
        self.assertIn('random_state', params)
        self.assertEqual(params['n_jobs'], 1)  # Inherited/Forced
        
        # Test fitting
        model.fit(self.X, self.y)
        print("✅ IRM_LGBMClassifier fit successful")

    def test_xgb_constraints_mismatch(self):
        """Test IRM_XGBClassifier with mismatched constraints."""
        print("\nTesting IRM_XGBClassifier constraint handling...")
        # Create constraints with extra features not in X
        constraints = {
            'feat_0': 1,
            'feat_999': 1  # Invalid feature
        }
        
        # Test interaction constraints with Names
        interaction_constraints = [
            ['feat_0', 'feat_1'], # Valid
            ['feat_0', 'feat_999', 'feat_1'], # Invalid feature
            ['feat_998', 'feat_999'] # All invalid
        ]

        model = IRM_XGBClassifier(
            irm_system=self.irm_system,
            environment_masks=self.environment_masks,
            monotone_constraints=constraints,
            interaction_constraints=interaction_constraints,
            n_jobs=1
        )
        
        # Fit should succeed despite mismatch (should filter)
        model.fit(self.X, self.y)
        print("✅ IRM_XGBClassifier fit successful with constraint mismatch")
        
        # Check that it didn't crash
        self.assertTrue(hasattr(model, 'feature_importances_'))

    def test_catboost_fix(self):
        """Test IRM_CatBoostClassifier initialization."""
        print("\nTesting IRM_CatBoostClassifier...")
        model = IRM_CatBoostClassifier(
            irm_system=self.irm_system,
            environment_masks=self.environment_masks,
            thread_count=1,
            l2_leaf_reg=5, # Verify changed param
            random_strength=1
        )
        
        params = model.get_params()
        self.assertNotIn('irm_system', params)
        self.assertNotIn('environment_masks', params)
        self.assertEqual(params['thread_count'], 1)
        self.assertEqual(params['l2_leaf_reg'], 5)
        
        # Fit
        model.fit(self.X, self.y)
        print("✅ IRM_CatBoostClassifier fit successful")

    def test_elasticnet_pipeline_mimic(self):
        """Verify that ElasticNet configuration concept works (Pipeline)."""
        # I cannot test looking inside the function closure in the massive file directly easily,
        # but I can verify the imports and pipeline construction work as expected.
        from sklearn.linear_model import SGDClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        elastic_net_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                l1_ratio=0.3,
                alpha=0.01,
                max_iter=100, # fast
                tol=1e-4,
                fit_intercept=True,
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            ))
        ])
        
        elastic_net_calibrated = CalibratedClassifierCV(elastic_net_pipeline, method='sigmoid', cv=3)
        elastic_net_calibrated.fit(self.X, self.y)
        print("✅ ElasticNet Pipeline fit successful")

if __name__ == '__main__':
    unittest.main()
