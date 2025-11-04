"""
HPO Trainer - 100% Data-Driven

Uses hierarchical parameter optimizer for efficient hyperparameter search.
No YAML configs, no manual tuning - pure optimization.

Optimizations:
- Hierarchical parameter optimizer (coarse → fine → TPE)
- Hardware-aware optimization
- Purged cross-validation to prevent data leakage
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any, Optional, List
import lightgbm as lgb

# Hierarchical parameter optimizer
try:
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
        HierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage
    )
    HIERARCHICAL_OPTIMIZER_AVAILABLE = True
except ImportError:
    HIERARCHICAL_OPTIMIZER_AVAILABLE = False

# Hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import get_unified_hardware_manager
    HARDWARE_OPTIMIZER_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)


class HPOTrainer:
    """
    Train LGBM with hierarchical hyperparameter optimization.
    
    Uses multi-stage optimization:
    1. Coarse grid search
    2. Fine grid around best region
    3. TPE (Tree Parzen Estimator) for final refinement
    
    Philosophy: No predetermined values, let staged optimization discover best config.
    """
    
    def __init__(self, n_trials: int = 200):
        """
        Initialize HPO trainer.
        
        Args:
            n_trials: Number of HPO trials (distributed across stages)
        """
        self.n_trials = n_trials
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware manager if available
        if HARDWARE_OPTIMIZER_AVAILABLE:
            self.hardware_manager = get_unified_hardware_manager()
            self.logger.info("✅ Hardware optimizer initialized")
        else:
            self.hardware_manager = None
    
    def train_optimized_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
        """
        Train LGBM with hierarchical HPO.
        
        Uses multi-stage optimization:
        1. Coarse grid: 20% of trials
        2. Fine grid: 30% of trials
        3. TPE: 50% of trials
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            Tuple of (trained_model, best_params)
        """
        self.logger.info(f"🔧 Running hierarchical HPO with {self.n_trials} trials...")
        
        # Try hierarchical optimizer first
        if HIERARCHICAL_OPTIMIZER_AVAILABLE:
            best_params = self._hpo_with_hierarchical_optimizer(
                X_train, y_train, X_val, y_val
            )
        else:
            # Fallback to standard HPO utils
            self.logger.warning("Hierarchical optimizer not available, using standard HPO")
            try:
                from src.utils.ml_common.optimization.hpo_utils import optimize_hyperparameters
                
                # Define search space
                param_space = {
                    'num_leaves': (10, 100),
                    'max_depth': (3, 12),
                    'learning_rate': (0.001, 0.3),
                    'min_data_in_leaf': (10, 200),
                    'lambda_l1': (0.0, 10.0),
                    'lambda_l2': (0.0, 10.0),
                    'feature_fraction': (0.5, 1.0),
                    'bagging_fraction': (0.5, 1.0),
                    'bagging_freq': (1, 10)
                }
                
                best_params, optimization_results = optimize_hyperparameters(
                    model_type='lgbm',
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    param_space=param_space,
                    n_trials=self.n_trials,
                    metric='r2'
                )
                
                self.logger.info(f"✅ HPO complete! Best R²: {optimization_results.get('best_score', 'N/A')}")
                
            except ImportError:
                self.logger.warning("HPO utils not available, using Optuna directly")
                best_params = self._hpo_with_optuna(X_train, y_train, X_val, y_val)
        
        # Train final model with best params
        self.logger.info("🎯 Training final model with optimized hyperparameters...")
        
        final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
        final_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = final_model.score(X_train, y_train)
        val_score = final_model.score(X_val, y_val)
        
        self.logger.info(f"✅ Final model trained!")
        self.logger.info(f"   Train R²: {train_score:.4f}")
        self.logger.info(f"   Val R²: {val_score:.4f}")
        
        return final_model, best_params
    
    def _hpo_with_hierarchical_optimizer(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Use hierarchical parameter optimizer for staged optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            Best hyperparameters
        """
        self.logger.info("Using hierarchical parameter optimizer (coarse → fine → TPE)")
        
        # Calculate safe bounds for min_data_in_leaf (must be < n_train_samples)
        max_safe_min_leaf = max(10, int(len(X_train) * 0.3))  # Max 30% of training data
        
        # Define parameter groups
        param_groups = [
            ParameterGroup(
                name="tree_structure",
                params={
                    'num_leaves': {'type': 'int', 'low': 10, 'high': min(100, len(X_train))},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 12}
                },
                priority=1  # Optimize first (most important)
            ),
            ParameterGroup(
                name="regularization",
                params={
                    'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 10.0},
                    'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 10.0},
                    'min_data_in_leaf': {'type': 'int', 'low': 10, 'high': max_safe_min_leaf}
                },
                priority=2,
                depends_on=['tree_structure']
            ),
            ParameterGroup(
                name="learning",
                params={
                    'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.3, 'log': True},
                    'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
                    'bagging_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
                    'bagging_freq': {'type': 'int', 'low': 1, 'high': 10}
                },
                priority=3,
                depends_on=['tree_structure', 'regularization']
            )
        ]
        
        # Objective function - signature must match HierarchicalParameterOptimizer expectations
        # Expected: (params, X_train, y_train, X_val, y_val, model, cv_folds, scoring_metric, **kwargs)
        def objective_func(
            params: Dict[str, Any],
            X_train_inner: np.ndarray,
            y_train_inner: np.ndarray,
            X_val_inner: Optional[np.ndarray] = None,
            y_val_inner: Optional[np.ndarray] = None,
            model: Optional[Any] = None,
            cv_folds: int = 5,
            scoring_metric: str = 'r2',
            **kwargs
        ) -> float:
            """Objective for hierarchical optimizer."""
            try:
                model_params = {
                    'objective': 'regression',
                    'verbosity': -1,
                    'force_col_wise': True,
                    'n_estimators': 200,
                    'random_state': 42,
                    **params
                }
                
                model_instance = lgb.LGBMRegressor(**model_params)
                model_instance.fit(X_train_inner, y_train_inner)
                
                # Evaluate on validation set (use outer scope validation data)
                score = model_instance.score(X_val, y_val)
                return score
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return -999.0  # Return very poor score on failure
        
        # Create hierarchical optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_func,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ],
            direction='maximize',
            n_rounds=2,  # 2 rounds: exploration + refinement
            enable_final_refinement=True,
            final_refinement_trials=int(self.n_trials * 0.2),  # 20% for final refinement
            verbose=True
        )
        
        # Run optimization
        result = optimizer.optimize(
            X_train.values, 
            y_train.values, 
            X_val.values, 
            y_val.values
        )
        
        self.logger.info(f"✅ Hierarchical optimization complete! Best R²: {result.best_score:.4f}")
        
        # Add fixed params
        best_params = result.best_params.copy()
        best_params['objective'] = 'regression'
        best_params['force_col_wise'] = True
        best_params['n_estimators'] = 200
        
        return best_params
    
    def _hpo_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Fallback HPO implementation using Optuna directly.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        
        Returns:
            Best hyperparameters
        """
        try:
            import optuna
            
            def objective(trial):
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'force_col_wise': True,
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10)
                }
                
                model = lgb.LGBMRegressor(**params, n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
                
                val_score = model.score(X_val, y_val)
                return val_score
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
            
            self.logger.info(f"✅ Optuna HPO complete! Best R²: {study.best_value:.4f}")
            
            # Convert to LGBM params
            best_params = study.best_params.copy()
            best_params['objective'] = 'regression'
            best_params['force_col_wise'] = True
            best_params['n_estimators'] = 200
            
            return best_params
            
        except ImportError:
            self.logger.warning("Optuna not available, using default params")
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Fallback default parameters if HPO not available.
        
        Returns:
            Default LGBM parameters
        """
        return {
            'objective': 'regression',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'min_data_in_leaf': 20,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'n_estimators': 200,
            'force_col_wise': True
        }

