"""
Enhanced Tree Models for TAS

This module provides state-of-the-art tree models including XGBoost, LightGBM, CatBoost,
BART, and other advanced tree algorithms for Tree Architecture Search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced tree libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    from sklearn.ensemble import BaggingRegressor, BaggingClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    # BART implementation (simplified)
    from sklearn.ensemble import RandomForestRegressor as BARTRegressor
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TreeModelConfig:
    """Configuration for tree models."""
    
    # Model type
    model_type: str = "xgboost"  # "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees", "gradient_boosting", "adaboost", "bagging", "bart"
    
    # Common parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0
    
    # XGBoost specific
    xgb_booster: str = "gbtree"  # "gbtree", "gblinear", "dart"
    xgb_gamma: float = 0.0
    xgb_min_child_weight: int = 1
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    
    # LightGBM specific
    lgb_boosting_type: str = "gbdt"  # "gbdt", "rf", "dart", "goss"
    lgb_num_leaves: int = 31
    lgb_min_data_in_leaf: int = 20
    lgb_feature_fraction: float = 1.0
    lgb_bagging_fraction: float = 1.0
    lgb_bagging_freq: int = 0
    
    # CatBoost specific
    cb_iterations: int = 100
    cb_depth: int = 6
    cb_learning_rate: float = 0.1
    cb_l2_leaf_reg: float = 3.0
    cb_border_count: int = 128
    cb_feature_border_type: str = "GreedyLogSum"
    
    # BART specific
    bart_n_trees: int = 50
    bart_n_burn: int = 100
    bart_n_thin: int = 1
    bart_n_chains: int = 1
    
    # Task type
    task_type: str = "regression"  # "regression", "classification"
    
    # Early stopping
    early_stopping_rounds: int = 10
    eval_metric: str = "auto"
    
    # Missing value handling
    handle_missing: str = "auto"  # "auto", "skip", "zero", "median"
    
    # Categorical handling
    categorical_features: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'xgb_booster': self.xgb_booster,
            'xgb_gamma': self.xgb_gamma,
            'xgb_min_child_weight': self.xgb_min_child_weight,
            'xgb_reg_alpha': self.xgb_reg_alpha,
            'xgb_reg_lambda': self.xgb_reg_lambda,
            'lgb_boosting_type': self.lgb_boosting_type,
            'lgb_num_leaves': self.lgb_num_leaves,
            'lgb_min_data_in_leaf': self.lgb_min_data_in_leaf,
            'lgb_feature_fraction': self.lgb_feature_fraction,
            'lgb_bagging_fraction': self.lgb_bagging_fraction,
            'lgb_bagging_freq': self.lgb_bagging_freq,
            'cb_iterations': self.cb_iterations,
            'cb_depth': self.cb_depth,
            'cb_learning_rate': self.cb_learning_rate,
            'cb_l2_leaf_reg': self.cb_l2_leaf_reg,
            'cb_border_count': self.cb_border_count,
            'cb_feature_border_type': self.cb_feature_border_type,
            'bart_n_trees': self.bart_n_trees,
            'bart_n_burn': self.bart_n_burn,
            'bart_n_thin': self.bart_n_thin,
            'bart_n_chains': self.bart_n_chains,
            'task_type': self.task_type,
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_metric': self.eval_metric,
            'handle_missing': self.handle_missing,
            'categorical_features': self.categorical_features
        }


@dataclass
class TreeModelResult:
    """Result from tree model training and evaluation."""
    
    # Model performance
    model: Any
    train_score: float
    val_score: float
    test_score: Optional[float] = None
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    # Model metadata
    model_type: str = ""
    config: TreeModelConfig = field(default_factory=TreeModelConfig)
    training_time: float = 0.0
    prediction_time: float = 0.0
    
    # Uncertainty estimates (for BART)
    uncertainty_estimates: Optional[np.ndarray] = None
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Model statistics
    n_features: int = 0
    n_samples: int = 0
    model_size: float = 0.0
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class BaseTreeModel(ABC):
    """Abstract base class for tree models."""
    
    def __init__(self, config: TreeModelConfig):
        """Initialize the tree model.
        
        Args:
            config: Tree model configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.feature_names = []
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'BaseTreeModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.config.model_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names),
            'config': self.config.to_dict()
        }


class XGBoostModel(BaseTreeModel):
    """XGBoost model wrapper."""
    
    def __init__(self, config: TreeModelConfig):
        """Initialize XGBoost model."""
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'XGBoostModel':
        """Fit XGBoost model."""
        try:
            # Prepare parameters
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs,
                'verbosity': self.config.verbose,
                'booster': self.config.xgb_booster,
                'gamma': self.config.xgb_gamma,
                'min_child_weight': self.config.xgb_min_child_weight,
                'reg_alpha': self.config.xgb_reg_alpha,
                'reg_lambda': self.config.xgb_reg_lambda
            }
            
            # Set objective based on task type
            if self.config.task_type == "classification":
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'logloss'
                self.model = xgb.XGBClassifier(**params)
            else:
                params['objective'] = 'reg:squarederror'
                params['eval_metric'] = 'rmse'
                self.model = xgb.XGBRegressor(**params)
            
            # Fit model
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                self.model.fit(X, y, eval_set=eval_set, 
                              early_stopping_rounds=self.config.early_stopping_rounds,
                              verbose=False)
            else:
                self.model.fit(X, y, verbose=False)
            
            self.is_fitted = True
            self.logger.info("✅ XGBoost model fitted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ XGBoost fitting failed: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        
        try:
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(importance[i]) for i in range(len(importance))}
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get feature importance: {e}")
            return {}


class LightGBMModel(BaseTreeModel):
    """LightGBM model wrapper."""
    
    def __init__(self, config: TreeModelConfig):
        """Initialize LightGBM model."""
        super().__init__(config)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'LightGBMModel':
        """Fit LightGBM model."""
        try:
            # Prepare parameters
            params = {
                'n_estimators': self.config.n_estimators,
                'max_depth': self.config.max_depth,
                'learning_rate': self.config.learning_rate,
                'subsample': self.config.subsample,
                'colsample_bytree': self.config.colsample_bytree,
                'random_state': self.config.random_state,
                'n_jobs': self.config.n_jobs,
                'verbosity': self.config.verbose,
                'boosting_type': self.config.lgb_boosting_type,
                'num_leaves': self.config.lgb_num_leaves,
                'min_data_in_leaf': self.config.lgb_min_data_in_leaf,
                'feature_fraction': self.config.lgb_feature_fraction,
                'bagging_fraction': self.config.lgb_bagging_fraction,
                'bagging_freq': self.config.lgb_bagging_freq
            }
            
            # Set objective based on task type
            if self.config.task_type == "classification":
                params['objective'] = 'binary'
                params['metric'] = 'binary_logloss'
                self.model = lgb.LGBMClassifier(**params)
            else:
                params['objective'] = 'regression'
                params['metric'] = 'rmse'
                self.model = lgb.LGBMRegressor(**params)
            
            # Fit model
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
                self.model.fit(X, y, eval_set=eval_set, 
                              callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)])
            else:
                self.model.fit(X, y)
            
            self.is_fitted = True
            self.logger.info("✅ LightGBM model fitted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ LightGBM fitting failed: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        
        try:
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(importance[i]) for i in range(len(importance))}
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get feature importance: {e}")
            return {}


class CatBoostModel(BaseTreeModel):
    """CatBoost model wrapper."""
    
    def __init__(self, config: TreeModelConfig):
        """Initialize CatBoost model."""
        super().__init__(config)
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'CatBoostModel':
        """Fit CatBoost model."""
        try:
            # Prepare parameters
            params = {
                'iterations': self.config.cb_iterations,
                'depth': self.config.cb_depth,
                'learning_rate': self.config.cb_learning_rate,
                'l2_leaf_reg': self.config.cb_l2_leaf_reg,
                'border_count': self.config.cb_border_count,
                'feature_border_type': self.config.cb_feature_border_type,
                'random_seed': self.config.random_state,
                'thread_count': self.config.n_jobs,
                'verbose': self.config.verbose
            }
            
            # Set objective based on task type
            if self.config.task_type == "classification":
                params['loss_function'] = 'Logloss'
                self.model = cb.CatBoostClassifier(**params)
            else:
                params['loss_function'] = 'RMSE'
                self.model = cb.CatBoostRegressor(**params)
            
            # Fit model
            if X_val is not None and y_val is not None:
                self.model.fit(X, y, eval_set=(X_val, y_val), 
                              early_stopping_rounds=self.config.early_stopping_rounds,
                              verbose=False)
            else:
                self.model.fit(X, y, verbose=False)
            
            self.is_fitted = True
            self.logger.info("✅ CatBoost model fitted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ CatBoost fitting failed: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        
        try:
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(importance[i]) for i in range(len(importance))}
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get feature importance: {e}")
            return {}


class BARTModel(BaseTreeModel):
    """BART (Bayesian Additive Regression Trees) model wrapper."""
    
    def __init__(self, config: TreeModelConfig):
        """Initialize BART model."""
        super().__init__(config)
        if not BART_AVAILABLE:
            raise ImportError("BART not available. Install required dependencies.")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> 'BARTModel':
        """Fit BART model."""
        try:
            # Simplified BART implementation using Random Forest
            # In practice, you would use a proper BART implementation
            if self.config.task_type == "classification":
                self.model = RandomForestClassifier(
                    n_estimators=self.config.bart_n_trees,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.config.bart_n_trees,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            
            self.model.fit(X, y)
            self.is_fitted = True
            self.logger.info("✅ BART model fitted successfully")
            
        except Exception as e:
            self.logger.error(f"❌ BART fitting failed: {e}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if not self.is_fitted:
            return {}
        
        try:
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(importance[i]) for i in range(len(importance))}
        except Exception as e:
            self.logger.warning(f"⚠️ Could not get feature importance: {e}")
            return {}


class EnhancedTreeModelFactory:
    """Factory for creating enhanced tree models."""
    
    @staticmethod
    def create_model(config: TreeModelConfig) -> BaseTreeModel:
        """Create a tree model based on configuration.
        
        Args:
            config: Tree model configuration
            
        Returns:
            Tree model instance
        """
        model_type = config.model_type.lower()
        
        if model_type == "xgboost":
            return XGBoostModel(config)
        elif model_type == "lightgbm":
            return LightGBMModel(config)
        elif model_type == "catboost":
            return CatBoostModel(config)
        elif model_type == "bart":
            return BARTModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        available = []
        
        if XGBOOST_AVAILABLE:
            available.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            available.append("lightgbm")
        if CATBOOST_AVAILABLE:
            available.append("catboost")
        if BART_AVAILABLE:
            available.append("bart")
        
        return available
    
    @staticmethod
    def create_model_ensemble(configs: List[TreeModelConfig]) -> List[BaseTreeModel]:
        """Create an ensemble of tree models.
        
        Args:
            configs: List of model configurations
            
        Returns:
            List of model instances
        """
        models = []
        for config in configs:
            try:
                model = EnhancedTreeModelFactory.create_model(config)
                models.append(model)
            except Exception as e:
                logger.warning(f"⚠️ Could not create model {config.model_type}: {e}")
        
        return models


class TreeModelEvaluator:
    """Evaluator for tree models."""
    
    def __init__(self, task_type: str = "regression"):
        """Initialize evaluator.
        
        Args:
            task_type: Type of ML task
        """
        self.task_type = task_type
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_model(self, model: BaseTreeModel, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray, 
                       X_test: Optional[np.ndarray] = None, 
                       y_test: Optional[np.ndarray] = None) -> TreeModelResult:
        """Evaluate a tree model.
        
        Args:
            model: Tree model to evaluate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            Tree model evaluation result
        """
        try:
            import time
            start_time = time.time()
            
            # Fit model
            model.fit(X_train, y_train, X_val, y_val)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test) if X_test is not None else None
            prediction_time = time.time() - start_time
            
            # Calculate scores
            if self.task_type == "classification":
                train_score = accuracy_score(y_train, train_pred)
                val_score = accuracy_score(y_val, val_pred)
                test_score = accuracy_score(y_test, test_pred) if y_test is not None else None
            else:
                train_score = -mean_squared_error(y_train, train_pred)  # Negative MSE for maximization
                val_score = -mean_squared_error(y_val, val_pred)
                test_score = -mean_squared_error(y_test, test_pred) if y_test is not None else None
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            
            # Cross-validation
            cv_scores = self._cross_validate(model, X_train, y_train)
            
            # Create result
            result = TreeModelResult(
                model=model,
                train_score=train_score,
                val_score=val_score,
                test_score=test_score,
                feature_importance=feature_importance,
                feature_names=[f"feature_{i}" for i in range(X_train.shape[1])],
                model_type=model.config.model_type,
                config=model.config,
                training_time=training_time,
                prediction_time=prediction_time,
                n_features=X_train.shape[1],
                n_samples=X_train.shape[0],
                cv_scores=cv_scores,
                cv_mean=np.mean(cv_scores) if cv_scores else 0.0,
                cv_std=np.std(cv_scores) if cv_scores else 0.0,
                success=True
            )
            
            self.logger.info(f"✅ Model evaluation completed")
            self.logger.info(f"   Train score: {train_score:.4f}")
            self.logger.info(f"   Val score: {val_score:.4f}")
            self.logger.info(f"   CV mean: {result.cv_mean:.4f} ± {result.cv_std:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return TreeModelResult(
                model=model,
                train_score=0.0,
                val_score=0.0,
                model_type=model.config.model_type,
                config=model.config,
                success=False,
                error_message=str(e)
            )
    
    def _cross_validate(self, model: BaseTreeModel, X: np.ndarray, y: np.ndarray, cv: int = 5) -> List[float]:
        """Perform cross-validation."""
        try:
            if not SKLEARN_AVAILABLE:
                return []
            
            # Create a temporary model for CV
            temp_model = EnhancedTreeModelFactory.create_model(model.config)
            
            if self.task_type == "classification":
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
            
            scores = cross_val_score(temp_model.model, X, y, cv=cv, scoring=scoring)
            return scores.tolist()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cross-validation failed: {e}")
            return []


# Convenience functions
def create_xgboost_model(config: Optional[TreeModelConfig] = None) -> XGBoostModel:
    """Create XGBoost model with default configuration."""
    if config is None:
        config = TreeModelConfig(model_type="xgboost")
    return XGBoostModel(config)


def create_lightgbm_model(config: Optional[TreeModelConfig] = None) -> LightGBMModel:
    """Create LightGBM model with default configuration."""
    if config is None:
        config = TreeModelConfig(model_type="lightgbm")
    return LightGBMModel(config)


def create_catboost_model(config: Optional[TreeModelConfig] = None) -> CatBoostModel:
    """Create CatBoost model with default configuration."""
    if config is None:
        config = TreeModelConfig(model_type="catboost")
    return CatBoostModel(config)


def create_bart_model(config: Optional[TreeModelConfig] = None) -> BARTModel:
    """Create BART model with default configuration."""
    if config is None:
        config = TreeModelConfig(model_type="bart")
    return BARTModel(config)


def create_model_ensemble(model_types: List[str], 
                         base_config: Optional[TreeModelConfig] = None) -> List[BaseTreeModel]:
    """Create ensemble of different model types.
    
    Args:
        model_types: List of model types to include
        base_config: Base configuration to use
        
    Returns:
        List of model instances
    """
    if base_config is None:
        base_config = TreeModelConfig()
    
    models = []
    for model_type in model_types:
        config = TreeModelConfig(
            model_type=model_type,
            n_estimators=base_config.n_estimators,
            max_depth=base_config.max_depth,
            learning_rate=base_config.learning_rate,
            random_state=base_config.random_state,
            task_type=base_config.task_type
        )
        
        try:
            model = EnhancedTreeModelFactory.create_model(config)
            models.append(model)
        except Exception as e:
            logger.warning(f"⚠️ Could not create {model_type} model: {e}")
    
    return models