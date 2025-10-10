"""
Method Settings for Reproducible Comparisons

This module provides standardized method settings for reproducible
feature comparison across different algorithms.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import shap
import warnings

logger = logging.getLogger(__name__)

class MethodSettings:
    """
    Standardized method settings for reproducible comparisons.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize method settings.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def get_lgbm_settings(self, task_type: str = 'regression') -> Dict[str, Any]:
        """
        Get standardized LightGBM settings.
        
        Args:
            task_type: 'regression' or 'classification'
            
        Returns:
            LightGBM settings dictionary
        """
        if task_type == 'regression':
            objective = 'regression'
            metric = 'rmse'
        else:
            objective = 'binary'
            metric = 'binary_logloss'
        
        settings = {
            'objective': objective,
            'metric': metric,
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # Conservative setting for comparability
            'learning_rate': 0.1,
            'feature_fraction': 0.8,  # Regularization
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,  # Tuned for stability
            'min_sum_hessian_in_leaf': 1e-3,
            'lambda_l1': 0.1,  # L1 regularization
            'lambda_l2': 0.1,  # L2 regularization
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1,
            'force_col_wise': True,  # For reproducibility
            'deterministic': True
        }
        
        return settings
    
    def get_lgbm_cv_settings(self, n_splits: int = 5) -> Dict[str, Any]:
        """
        Get LightGBM cross-validation settings.
        
        Args:
            n_splits: Number of CV splits
            
        Returns:
            CV settings dictionary
        """
        return {
            'n_splits': n_splits,
            'shuffle': False,  # Time series
            'random_state': self.random_state,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse',
            'return_train_score': True
        }
    
    def get_shap_settings(self, max_depth: int = 3) -> Dict[str, Any]:
        """
        Get standardized SHAP settings.
        
        Args:
            max_depth: Maximum depth for SHAP tree explainer
            
        Returns:
            SHAP settings dictionary
        """
        return {
            'max_depth': max_depth,  # Capped for comparability
            'feature_perturbation': 'tree_path_dependent',
            'model_output': 'raw',
            'use_cuda': False,  # For reproducibility
            'random_state': self.random_state
        }
    
    def get_lasso_settings(self, alphas: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Get standardized LASSO settings.
        
        Args:
            alphas: List of alpha values for regularization path
            
        Returns:
            LASSO settings dictionary
        """
        if alphas is None:
            alphas = np.logspace(-4, 1, 50)  # 0.0001 to 10
        
        return {
            'alphas': alphas,
            'cv': 5,
            'random_state': self.random_state,
            'max_iter': 2000,
            'tol': 1e-4,
            'selection': 'cyclic',
            'fit_intercept': True,
            'normalize': False,  # We'll standardize manually
            'copy_X': True
        }
    
    def get_ridge_settings(self, alphas: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Get standardized Ridge settings.
        
        Args:
            alphas: List of alpha values for regularization path
            
        Returns:
            Ridge settings dictionary
        """
        if alphas is None:
            alphas = np.logspace(-2, 3, 50)  # 0.01 to 1000
        
        return {
            'alphas': alphas,
            'cv': 5,
            'random_state': self.random_state,
            'fit_intercept': True,
            'normalize': False,  # We'll standardize manually
            'copy_X': True
        }
    
    def get_mutual_info_settings(self, n_neighbors: int = 3, 
                                discrete_features: str = 'auto') -> Dict[str, Any]:
        """
        Get standardized Mutual Information settings.
        
        Args:
            n_neighbors: Number of neighbors for kNN MI
            discrete_features: Strategy for discrete features
            
        Returns:
            MI settings dictionary
        """
        return {
            'n_neighbors': n_neighbors,
            'discrete_features': discrete_features,
            'random_state': self.random_state,
            'copy': True
        }
    
    def get_permutation_importance_settings(self, n_repeats: int = 10,
                                          scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Get standardized Permutation Importance settings.
        
        Args:
            n_repeats: Number of repeats for permutation
            scoring: Scoring function
            
        Returns:
            Permutation importance settings dictionary
        """
        return {
            'n_repeats': n_repeats,
            'scoring': scoring,
            'random_state': self.random_state,
            'n_jobs': -1
        }
    
    def create_lgbm_model(self, task_type: str = 'regression', 
                         custom_params: Optional[Dict[str, Any]] = None) -> lgb.LGBMRegressor:
        """
        Create standardized LightGBM model.
        
        Args:
            task_type: 'regression' or 'classification'
            custom_params: Custom parameters to override defaults
            
        Returns:
            LightGBM model
        """
        params = self.get_lgbm_settings(task_type)
        
        if custom_params:
            params.update(custom_params)
        
        if task_type == 'regression':
            return lgb.LGBMRegressor(**params)
        else:
            return lgb.LGBMClassifier(**params)
    
    def create_lasso_model(self, alphas: Optional[List[float]] = None,
                          custom_params: Optional[Dict[str, Any]] = None) -> LassoCV:
        """
        Create standardized LASSO model.
        
        Args:
            alphas: List of alpha values
            custom_params: Custom parameters to override defaults
            
        Returns:
            LASSO model
        """
        params = self.get_lasso_settings(alphas)
        
        if custom_params:
            params.update(custom_params)
        
        return LassoCV(**params)
    
    def create_ridge_model(self, alphas: Optional[List[float]] = None,
                          custom_params: Optional[Dict[str, Any]] = None) -> RidgeCV:
        """
        Create standardized Ridge model.
        
        Args:
            alphas: List of alpha values
            custom_params: Custom parameters to override defaults
            
        Returns:
            Ridge model
        """
        params = self.get_ridge_settings(alphas)
        
        if custom_params:
            params.update(custom_params)
        
        return RidgeCV(**params)
    
    def calculate_shap_importance(self, model: Any, X: pd.DataFrame, 
                                 max_depth: int = 3) -> pd.Series:
        """
        Calculate SHAP importance with standardized settings.
        
        Args:
            model: Trained model
            X: Feature matrix
            max_depth: Maximum depth for SHAP
            
        Returns:
            SHAP importance series
        """
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model, max_depth=max_depth)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values
            if len(shap_values.shape) == 2:
                importance = np.abs(shap_values).mean(axis=0)
            else:
                importance = np.abs(shap_values).mean(axis=0).mean(axis=0)
            
            return pd.Series(importance, index=X.columns)
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return pd.Series(0, index=X.columns)
    
    def calculate_mutual_info_importance(self, X: pd.DataFrame, y: pd.Series,
                                       n_neighbors: int = 3) -> pd.Series:
        """
        Calculate Mutual Information importance with standardized settings.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_neighbors: Number of neighbors for kNN MI
            
        Returns:
            MI importance series
        """
        try:
            # Calculate MI for each feature
            mi_scores = mutual_info_regression(
                X, y, 
                n_neighbors=n_neighbors,
                random_state=self.random_state
            )
            
            return pd.Series(mi_scores, index=X.columns)
            
        except Exception as e:
            logger.warning(f"MI calculation failed: {e}")
            return pd.Series(0, index=X.columns)
    
    def calculate_permutation_importance(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                       n_repeats: int = 10) -> pd.Series:
        """
        Calculate Permutation Importance with standardized settings.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            n_repeats: Number of repeats
            
        Returns:
            Permutation importance series
        """
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            return pd.Series(perm_importance.importances_mean, index=X.columns)
            
        except Exception as e:
            logger.warning(f"Permutation importance calculation failed: {e}")
            return pd.Series(0, index=X.columns)
    
    def get_regularization_path(self, model: Any) -> Dict[str, Any]:
        """
        Get regularization path for linear models.
        
        Args:
            model: Trained linear model
            
        Returns:
            Regularization path information
        """
        if hasattr(model, 'alphas_') and hasattr(model, 'coef_path_'):
            return {
                'alphas': model.alphas_,
                'coef_path': model.coef_path_,
                'alpha_optimal': model.alpha_,
                'coef_optimal': model.coef_,
                'feature_entry_order': self._get_feature_entry_order(model)
            }
        else:
            return {'error': 'Model does not support regularization path'}
    
    def _get_feature_entry_order(self, model: Any) -> List[str]:
        """
        Get feature entry order from regularization path.
        
        Args:
            model: Trained linear model
            
        Returns:
            List of features in entry order
        """
        if not hasattr(model, 'coef_path_'):
            return []
        
        # Find when each feature first becomes non-zero
        coef_path = model.coef_path_
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else range(coef_path.shape[1])
        
        entry_order = []
        for i, feature in enumerate(feature_names):
            non_zero_indices = np.where(np.abs(coef_path[:, i]) > 1e-8)[0]
            if len(non_zero_indices) > 0:
                entry_order.append((feature, non_zero_indices[0]))
        
        # Sort by entry index
        entry_order.sort(key=lambda x: x[1])
        return [feature for feature, _ in entry_order]
    
    def get_method_comparison_settings(self) -> Dict[str, Any]:
        """
        Get standardized settings for method comparison.
        
        Returns:
            Method comparison settings
        """
        return {
            'lgbm': self.get_lgbm_settings(),
            'lasso': self.get_lasso_settings(),
            'ridge': self.get_ridge_settings(),
            'mutual_info': self.get_mutual_info_settings(),
            'permutation_importance': self.get_permutation_importance_settings(),
            'shap': self.get_shap_settings(),
            'random_state': self.random_state,
            'cv_folds': 5,
            'early_stopping_rounds': 50,
            'n_bootstrap': 10,
            'n_permutation_repeats': 10
        }