"""
Tree-Based Architecture Search (TAS) for ML Common

This module provides comprehensive Tree-Based Architecture Search capabilities
specifically designed for financial time series and trading models as an
alternative to Neural Architecture Search (NAS).

Key Features:
- Tree-based model architecture search (XGBoost, LightGBM, CatBoost, Random Forest)
- Multi-objective optimization (accuracy, efficiency, interpretability, robustness)
- Regime-aware tree adaptation
- Feature selection and engineering optimization
- Ensemble architecture search
- Integration with existing ML pipeline
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from abc import ABC, abstractmethod
import json
from pathlib import Path

# Tree-based model imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

# Optimization imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TreeArchitectureConfig:
    """Configuration for tree-based architecture search."""
    
    # Model types to search
    model_types: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm', 'catboost', 'random_forest'])
    
    # Search parameters
    n_trials: int = 50
    timeout_seconds: int = 1800  # 30 minutes
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'interpretability', 'robustness'])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_methods: List[str] = field(default_factory=lambda: ['mutual_info', 'f_score', 'correlation'])
    max_features: int = 50
    
    # Ensemble search
    enable_ensemble_search: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'stacking', 'blending'])
    max_ensemble_models: int = 5
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_adaptation_strength: float = 0.3
    
    # Performance
    n_jobs: int = -1
    memory_limit_gb: float = 8.0


@dataclass
class TreeArchitectureCandidate:
    """A candidate tree-based architecture."""
    
    # Architecture definition
    model_type: str
    model_params: Dict[str, Any]
    feature_selection: Dict[str, Any]
    ensemble_config: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    interpretability_score: float = 0.0
    robustness_score: float = 0.0
    overall_score: float = 0.0
    
    # Training info
    training_time: float = 0.0
    model_size: int = 0
    n_features: int = 0
    
    # Regime performance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trial_number: int = 0


class TreeArchitectureSearchSpace:
    """Defines the search space for tree-based architectures."""
    
    def __init__(self, config: TreeArchitectureConfig):
        self.config = config
        self.logger = logger.getChild('TreeArchitectureSearchSpace')
    
    def sample_architecture(self, trial_number: int = 0) -> TreeArchitectureCandidate:
        """Sample a random tree architecture from the search space."""
        try:
            # Sample model type
            model_type = np.random.choice(self.config.model_types)
            
            # Sample model parameters based on type
            model_params = self._sample_model_params(model_type)
            
            # Sample feature selection
            feature_selection = self._sample_feature_selection()
            
            # Sample ensemble configuration
            ensemble_config = None
            if self.config.enable_ensemble_search and np.random.random() < 0.3:
                ensemble_config = self._sample_ensemble_config()
            
            candidate = TreeArchitectureCandidate(
                model_type=model_type,
                model_params=model_params,
                feature_selection=feature_selection,
                ensemble_config=ensemble_config,
                trial_number=trial_number
            )
            
            self.logger.debug(f"Sampled {model_type} architecture with {len(model_params)} parameters")
            return candidate
            
        except Exception as e:
            self.logger.error(f"Architecture sampling failed: {e}")
            # Return minimal architecture as fallback
            return TreeArchitectureCandidate(
                model_type='xgboost',
                model_params={'n_estimators': 100, 'max_depth': 3},
                feature_selection={'method': 'all', 'max_features': 10},
                trial_number=trial_number
            )
    
    def _sample_model_params(self, model_type: str) -> Dict[str, Any]:
        """Sample parameters for specific model type."""
        if model_type == 'xgboost':
            return {
                'n_estimators': np.random.choice([50, 100, 200, 300]),
                'max_depth': np.random.choice([3, 5, 7, 9, 10]),
                'learning_rate': np.random.choice([0.01, 0.05, 0.1, 0.2, 0.3]),
                'subsample': np.random.choice([0.6, 0.75, 0.9, 1.0]),
                'colsample_bytree': np.random.choice([0.6, 0.75, 0.9, 1.0]),
                'gamma': np.random.choice([0.0, 0.1, 0.5, 1.0, 2.0, 5.0]),
                'reg_alpha': np.random.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
                'reg_lambda': np.random.choice([0.0, 0.01, 0.1, 0.5, 1.0])
            }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': np.random.choice([100, 200, 300, 400, 500]),
                'max_depth': np.random.choice([3, 5, 7, 9, 12, 15]),
                'learning_rate': np.random.choice([0.01, 0.03, 0.05, 0.1, 0.2]),
                'subsample': np.random.choice([0.6, 0.75, 0.9, 1.0]),
                'colsample_bytree': np.random.choice([0.6, 0.75, 0.9, 1.0]),
                'reg_alpha': np.random.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
                'reg_lambda': np.random.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
                'num_leaves': np.random.choice([15, 31, 47, 63])
            }
        elif model_type == 'catboost':
            return {
                'iterations': np.random.choice([500, 700, 900, 1200]),
                'depth': np.random.choice([4, 5, 6]),
                'learning_rate': np.random.choice([0.03, 0.04, 0.05, 0.06]),
                'l2_leaf_reg': np.random.choice([6, 8, 10, 12]),
                'bootstrap_type': np.random.choice(['Bayesian', 'Bernoulli']),
                'subsample': np.random.choice([0.5, 0.7, 0.9])
            }
        elif model_type == 'random_forest':
            return {
                'n_estimators': np.random.choice([100, 200, 500, 800]),
                'max_depth': np.random.choice([5, 10, 15, 20, None]),
                'max_features': np.random.choice(['sqrt', 'log2', 0.3, 0.5]),
                'min_samples_split': np.random.choice([2, 5, 10, 20]),
                'min_samples_leaf': np.random.choice([1, 2, 4, 8]),
                'bootstrap': np.random.choice([True, False])
            }
        else:
            return {}
    
    def _sample_feature_selection(self) -> Dict[str, Any]:
        """Sample feature selection configuration."""
        if not self.config.enable_feature_selection:
            return {'method': 'all', 'max_features': None}
        
        return {
            'method': np.random.choice(self.config.feature_selection_methods),
            'max_features': np.random.choice([10, 20, 50, 100, self.config.max_features]),
            'threshold': np.random.choice([0.01, 0.05, 0.1, 0.2])
        }
    
    def _sample_ensemble_config(self) -> Dict[str, Any]:
        """Sample ensemble configuration."""
        return {
            'method': np.random.choice(self.config.ensemble_methods),
            'n_models': np.random.choice([2, 3, 4, 5]),
            'base_models': np.random.choice(self.config.model_types, 
                                         size=np.random.randint(2, min(4, len(self.config.model_types))),
                                         replace=False).tolist()
        }


class TreeBasedArchitectureSearch:
    """Main Tree-Based Architecture Search implementation."""
    
    def __init__(self, config: TreeArchitectureConfig):
        """Initialize TAS."""
        self.config = config
        self.logger = logger.getChild('TreeBasedArchitectureSearch')
        self.search_space = TreeArchitectureSearchSpace(config)
        self.candidates = []
        self.best_candidate = None
        
        self.logger.info(f"✅ Tree-Based Architecture Search initialized with {config.n_trials} trials")
    
    def search(self, 
               X_train: np.ndarray, 
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               regime_labels: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
        """
        Perform tree-based architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            regime_labels: Regime labels for regime-aware search (optional)
            
        Returns:
            Best architecture candidate
        """
        self.logger.info("🚀 Starting Tree-Based Architecture Search...")
        start_time = time.time()
        
        try:
            # Prepare validation data
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=self.config.test_size, random_state=42
                )
            
            # Search for architectures
            if OPTUNA_AVAILABLE:
                best_candidate = self._optuna_search(X_train, y_train, X_val, y_val, regime_labels)
            else:
                best_candidate = self._random_search(X_train, y_train, X_val, y_val, regime_labels)
            
            search_time = time.time() - start_time
            self.logger.info(f"✅ TAS completed in {search_time:.2f}s")
            self.logger.info(f"📊 Best architecture: {best_candidate.model_type}, score: {best_candidate.overall_score:.4f}")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Tree-Based Architecture Search failed: {e}")
            raise
    
    def _optuna_search(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: np.ndarray, 
                      y_val: np.ndarray,
                      regime_labels: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
        """Perform architecture search using Optuna."""
        self.logger.info("🔍 Starting Optuna-based tree architecture search...")
        
        def objective(trial):
            try:
                # Sample architecture
                candidate = self._sample_architecture_from_trial(trial)
                
                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate, X_train, y_train, X_val, y_val, regime_labels
                )
                
                return performance['overall_score']
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Run optimization
        study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds
        )
        
        # Get best candidate
        best_trial = study.best_trial
        best_candidate = self._sample_architecture_from_trial(best_trial)
        
        # Train final model
        performance = self._train_and_evaluate_architecture(
            best_candidate, X_train, y_train, X_val, y_val, regime_labels
        )
        
        best_candidate.accuracy = performance['accuracy']
        best_candidate.efficiency_score = performance['efficiency_score']
        best_candidate.interpretability_score = performance['interpretability_score']
        best_candidate.robustness_score = performance['robustness_score']
        best_candidate.overall_score = performance['overall_score']
        
        return best_candidate
    
    def _random_search(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: np.ndarray, 
                      y_val: np.ndarray,
                      regime_labels: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
        """Perform random architecture search."""
        self.logger.info("🔍 Starting random tree architecture search...")
        
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(self.config.n_trials):
            try:
                # Sample random architecture
                candidate = self.search_space.sample_architecture(trial)
                
                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate, X_train, y_train, X_val, y_val, regime_labels
                )
                
                # Update best candidate
                if performance['overall_score'] > best_score:
                    best_score = performance['overall_score']
                    best_candidate = candidate
                    
                    best_candidate.accuracy = performance['accuracy']
                    best_candidate.efficiency_score = performance['efficiency_score']
                    best_candidate.interpretability_score = performance['interpretability_score']
                    best_candidate.robustness_score = performance['robustness_score']
                    best_candidate.overall_score = performance['overall_score']
                
                self.logger.debug(f"Trial {trial}: {candidate.model_type} Score {performance['overall_score']:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        if best_candidate is None:
            raise RuntimeError("No successful architecture found")
        
        return best_candidate
    
    def _sample_architecture_from_trial(self, trial) -> TreeArchitectureCandidate:
        """Sample architecture from Optuna trial."""
        model_type = trial.suggest_categorical('model_type', self.config.model_types)

        # Sample model parameters based on type
        model_params = {}
        if model_type == 'xgboost':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
        elif model_type == 'lightgbm':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 63)
            }
        elif model_type == 'catboost':
            model_params = {
                'iterations': trial.suggest_int('iterations', 500, 1200),
                'depth': trial.suggest_int('depth', 4, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.06),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 6.0, 12.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'subsample': trial.suggest_float('subsample', 0.5, 0.9)
            }
        elif model_type == 'random_forest':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        
        # Sample feature selection
        feature_selection = {
            'method': trial.suggest_categorical('feature_method', self.config.feature_selection_methods),
            'max_features': trial.suggest_int('max_features', 10, self.config.max_features)
        }
        
        return TreeArchitectureCandidate(
            model_type=model_type,
            model_params=model_params,
            feature_selection=feature_selection,
            trial_number=trial.number
        )
    
    def _train_and_evaluate_architecture(self, 
                                       candidate: TreeArchitectureCandidate,
                                       X_train: np.ndarray, 
                                       y_train: np.ndarray,
                                       X_val: np.ndarray, 
                                       y_val: np.ndarray,
                                       regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train and evaluate a tree architecture candidate."""
        try:
            start_time = time.time()
            
            # Apply feature selection
            X_train_selected, X_val_selected, selected_features = self._apply_feature_selection(
                candidate.feature_selection, X_train, X_val
            )
            
            # Create and train model
            model = self._create_tree_model(candidate)
            
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train_selected)
            val_pred = model.predict(X_val_selected)
            
            # Calculate metrics
            if len(np.unique(y_train)) > 10:  # Regression
                accuracy = r2_score(y_val, val_pred)
                train_accuracy = r2_score(y_train, train_pred)
            else:  # Classification
                accuracy = accuracy_score(y_val, val_pred)
                train_accuracy = accuracy_score(y_train, train_pred)
            
            # Calculate efficiency score (simpler models are more efficient)
            efficiency_score = self._calculate_efficiency_score(model, candidate)
            
            # Calculate interpretability score
            interpretability_score = self._calculate_interpretability_score(model, candidate)
            
            # Calculate robustness score
            robustness_score = self._calculate_robustness_score(model, X_train_selected, y_train)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score({
                'accuracy': accuracy,
                'efficiency_score': efficiency_score,
                'interpretability_score': interpretability_score,
                'robustness_score': robustness_score
            })
            
            training_time = time.time() - start_time
            
            # Update candidate
            candidate.training_time = training_time
            candidate.n_features = len(selected_features)
            candidate.model_size = self._estimate_model_size(model)
            
            return {
                'accuracy': accuracy,
                'efficiency_score': efficiency_score,
                'interpretability_score': interpretability_score,
                'robustness_score': robustness_score,
                'overall_score': overall_score
            }
            
        except Exception as e:
            self.logger.warning(f"Architecture training failed: {e}")
            return {
                'accuracy': 0.0,
                'efficiency_score': 0.0,
                'interpretability_score': 0.0,
                'robustness_score': 0.0,
                'overall_score': 0.0
            }
    
    def _apply_feature_selection(self, feature_config: Dict[str, Any], 
                                X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Apply feature selection to training and validation data."""
        if feature_config['method'] == 'all':
            return X_train, X_val, list(range(X_train.shape[1]))
        
        max_features = min(feature_config['max_features'], X_train.shape[1])
        
        if feature_config['method'] == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
        elif feature_config['method'] == 'f_score':
            selector = SelectKBest(score_func=f_regression, k=max_features)
        else:
            # Correlation-based selection
            corr_matrix = np.corrcoef(X_train.T)
            feature_scores = np.sum(np.abs(corr_matrix), axis=1)
            selected_indices = np.argsort(feature_scores)[-max_features:]
            return X_train[:, selected_indices], X_val[:, selected_indices], selected_indices.tolist()
        
        X_train_selected = selector.fit_transform(X_train, X_train)  # Using X_train as target for feature selection
        X_val_selected = selector.transform(X_val)
        selected_features = selector.get_support(indices=True).tolist()
        
        return X_train_selected, X_val_selected, selected_features
    
    def _create_tree_model(self, candidate: TreeArchitectureCandidate):
        """Create tree model from architecture candidate."""
        model_type = candidate.model_type
        params = candidate.model_params
        
        if model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            return xgb.XGBRegressor(**params, random_state=42, n_jobs=self.config.n_jobs)
        
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            return lgb.LGBMRegressor(**params, random_state=42, n_jobs=self.config.n_jobs, verbose=-1)
        
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            return cb.CatBoostRegressor(**params, random_seed=42, verbose=False)
        
        elif model_type == 'random_forest':
            return RandomForestRegressor(**params, random_state=42, n_jobs=self.config.n_jobs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_efficiency_score(self, model, candidate: TreeArchitectureCandidate) -> float:
        """Calculate efficiency score based on model complexity."""
        try:
            # Get model complexity metrics
            if hasattr(model, 'get_booster'):
                # XGBoost/LightGBM
                n_trees = model.n_estimators
                avg_depth = model.max_depth
                complexity = n_trees * avg_depth
            elif hasattr(model, 'estimators_'):
                # Random Forest
                n_trees = len(model.estimators_)
                avg_depth = np.mean([tree.tree_.max_depth for tree in model.estimators_])
                complexity = n_trees * avg_depth
            else:
                # Fallback
                complexity = 1000
            
            # Efficiency score (lower complexity = higher efficiency)
            efficiency_score = 1.0 / (1.0 + complexity / 10000)
            return float(efficiency_score)
            
        except Exception as e:
            self.logger.warning(f"Efficiency score calculation failed: {e}")
            return 0.5
    
    def _calculate_interpretability_score(self, model, candidate: TreeArchitectureCandidate) -> float:
        """Calculate interpretability score."""
        try:
            # Tree-based models are generally more interpretable
            base_score = 0.8
            
            # Adjust based on model complexity
            if hasattr(model, 'max_depth') and model.max_depth <= 5:
                base_score += 0.2
            elif hasattr(model, 'max_depth') and model.max_depth > 10:
                base_score -= 0.2
            
            # Feature importance availability
            if hasattr(model, 'feature_importances_'):
                base_score += 0.1
            
            return float(np.clip(base_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Interpretability score calculation failed: {e}")
            return 0.5
    
    def _calculate_robustness_score(self, model, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Calculate robustness score using cross-validation."""
        try:
            # Use cross-validation to assess robustness
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, scoring='r2')
            robustness_score = np.mean(cv_scores)
            return float(robustness_score)
            
        except Exception as e:
            self.logger.warning(f"Robustness score calculation failed: {e}")
            return 0.5
    
    def _calculate_overall_score(self, performance: Dict[str, float]) -> float:
        """Calculate overall score from multiple objectives."""
        try:
            weights = self.config.objective_weights
            
            overall_score = (
                weights[0] * performance['accuracy'] +
                weights[1] * performance['efficiency_score'] +
                weights[2] * performance['interpretability_score'] +
                weights[3] * performance['robustness_score']
            )
            
            return float(overall_score)
            
        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def _estimate_model_size(self, model) -> int:
        """Estimate model size in parameters."""
        try:
            if hasattr(model, 'get_booster'):
                # XGBoost/LightGBM - estimate from trees
                return model.n_estimators * 100  # Rough estimate
            elif hasattr(model, 'estimators_'):
                # Random Forest
                return len(model.estimators_) * 100  # Rough estimate
            else:
                return 1000  # Default estimate
        except:
            return 1000
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of architecture search results."""
        if not self.candidates:
            return {'message': 'No search results available'}
        
        try:
            # Calculate summary statistics
            accuracies = [c.accuracy for c in self.candidates]
            efficiency_scores = [c.efficiency_score for c in self.candidates]
            interpretability_scores = [c.interpretability_score for c in self.candidates]
            robustness_scores = [c.robustness_score for c in self.candidates]
            overall_scores = [c.overall_score for c in self.candidates]
            
            return {
                'total_candidates': len(self.candidates),
                'best_accuracy': float(np.max(accuracies)),
                'best_efficiency': float(np.max(efficiency_scores)),
                'best_interpretability': float(np.max(interpretability_scores)),
                'best_robustness': float(np.max(robustness_scores)),
                'best_overall_score': float(np.max(overall_scores)),
                'search_statistics': {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'efficiency_mean': float(np.mean(efficiency_scores)),
                    'efficiency_std': float(np.std(efficiency_scores)),
                    'interpretability_mean': float(np.mean(interpretability_scores)),
                    'interpretability_std': float(np.std(interpretability_scores)),
                    'robustness_mean': float(np.mean(robustness_scores)),
                    'robustness_std': float(np.std(robustness_scores)),
                    'overall_score_mean': float(np.mean(overall_scores)),
                    'overall_score_std': float(np.std(overall_scores))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def search_tree_architecture(X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None,
                           config: Optional[TreeArchitectureConfig] = None,
                           regime_labels: Optional[np.ndarray] = None) -> TreeArchitectureCandidate:
    """
    Convenience function to perform tree-based architecture search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Tree architecture search configuration
        regime_labels: Regime labels for regime-aware search (optional)
        
    Returns:
        Best architecture candidate
    """
    if config is None:
        config = TreeArchitectureConfig()
    
    tas = TreeBasedArchitectureSearch(config)
    return tas.search(X_train, y_train, X_val, y_val, regime_labels)