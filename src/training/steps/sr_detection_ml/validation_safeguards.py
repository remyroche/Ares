"""
Validation Safeguards

Additional validation checks to prevent common ML errors in SR system.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from sklearn.metrics import r2_score, accuracy_score

logger = logging.getLogger(__name__)


class ValidationSafeguards:
    """
    Validation safeguards to detect suspicious results.
    
    Checks for:
    - Unrealistic performance (R² = 1.0)
    - Invalid hyperparameters
    - SHAP calculation failures
    - Overfitting indicators
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_results(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        best_params: Dict[str, Any],
        shap_values: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate training results for common issues.
        
        Returns:
            Dictionary with validation results and warnings
        """
        self.logger.info("🛡️ Running validation safeguards...")
        
        issues = []
        warnings = []
        
        # Check 1: Unrealistic performance
        perf_check = self._check_unrealistic_performance(
            model, X_train, y_train, X_val, y_val
        )
        if perf_check['is_suspicious']:
            issues.append(perf_check['message'])
        
        # Check 2: Invalid hyperparameters
        param_check = self._check_invalid_hyperparameters(
            best_params, len(X_train)
        )
        if param_check['has_issues']:
            issues.extend(param_check['issues'])
        
        # Check 3: SHAP calculation
        shap_check = self._check_shap_values(shap_values)
        if shap_check['has_issues']:
            warnings.append(shap_check['message'])
        
        # Check 4: Overfitting indicators
        overfit_check = self._check_overfitting(
            model, X_train, y_train, X_val, y_val
        )
        if overfit_check['is_overfitting']:
            warnings.append(overfit_check['message'])
        
        return {
            'has_critical_issues': len(issues) > 0,
            'critical_issues': issues,
            'warnings': warnings,
            'safe_to_use': len(issues) == 0
        }
    
    def _check_unrealistic_performance(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val
    ) -> Dict:
        """Check for unrealistically perfect performance."""
        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)
        
        # Perfect or near-perfect score
        if val_r2 >= 0.999:
            return {
                'is_suspicious': True,
                'message': (
                    f"🚨 UNREALISTIC PERFORMANCE: Val R²={val_r2:.4f} "
                    f"suggests data leakage or overfitting"
                )
            }
        
        # Train/val R² too close (should have some gap)
        if abs(train_r2 - val_r2) < 0.01 and train_r2 > 0.9:
            return {
                'is_suspicious': True,
                'message': (
                    f"🚨 SUSPICIOUS: Train R²={train_r2:.4f}, Val R²={val_r2:.4f} "
                    f"are too similar (suggests memorization)"
                )
            }
        
        return {'is_suspicious': False, 'message': None}
    
    def _check_invalid_hyperparameters(
        self,
        params: Dict[str, Any],
        n_train_samples: int
    ) -> Dict:
        """Check for invalid hyperparameter configurations."""
        issues = []
        
        # Check min_data_in_leaf
        min_leaf = params.get('min_data_in_leaf', 0)
        if min_leaf >= n_train_samples:
            issues.append(
                f"🚨 INVALID: min_data_in_leaf ({min_leaf}) >= n_train_samples ({n_train_samples})"
            )
        
        if min_leaf > n_train_samples * 0.8:
            issues.append(
                f"⚠️ WARNING: min_data_in_leaf ({min_leaf}) > 80% of training data ({n_train_samples})"
            )
        
        # Check num_leaves vs samples
        num_leaves = params.get('num_leaves', 31)
        if num_leaves > n_train_samples:
            issues.append(
                f"⚠️ WARNING: num_leaves ({num_leaves}) > n_samples ({n_train_samples})"
            )
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues
        }
    
    def _check_shap_values(self, shap_values: np.ndarray) -> Dict:
        """Check if SHAP values are calculated correctly."""
        # Check if all zeros
        if np.allclose(shap_values, 0):
            return {
                'has_issues': True,
                'message': "🚨 SHAP ERROR: All SHAP values are zero (calculation failed)"
            }
        
        # Check for NaN/inf
        if np.any(~np.isfinite(shap_values)):
            return {
                'has_issues': True,
                'message': "🚨 SHAP ERROR: SHAP values contain NaN or inf"
            }
        
        return {'has_issues': False, 'message': None}
    
    def _check_overfitting(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val
    ) -> Dict:
        """Check for overfitting indicators."""
        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)
        
        # Large gap between train and val
        gap = train_r2 - val_r2
        
        if gap > 0.2:  # More than 20% gap
            return {
                'is_overfitting': True,
                'message': (
                    f"⚠️ OVERFITTING: Train R²={train_r2:.4f}, Val R²={val_r2:.4f} "
                    f"(gap={gap:.4f} > 0.20)"
                )
            }
        
        return {'is_overfitting': False, 'message': None}

