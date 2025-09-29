"""
Tree AutoML Framework for TAS

This module provides automated machine learning capabilities for tree models,
including hyperparameter optimization, model selection, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import AutoML libraries
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import our enhanced tree models
try:
    from ..models.enhanced_tree_models import (
        EnhancedTreeModelFactory, TreeModelConfig, TreeModelResult,
        TreeModelEvaluator
    )
    TREE_MODELS_AVAILABLE = True
except ImportError:
    TREE_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for Tree AutoML."""
    
    # Optimization settings
    optimization_method: str = "optuna"  # "optuna", "grid", "random", "bayesian"
    max_trials: int = 100
    timeout_seconds: int = 3600
    n_jobs: int = -1
    random_state: int = 42
    
    # Model selection
    model_types: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"
    ])
    enable_ensemble: bool = True
    ensemble_method: str = "voting"  # "voting", "stacking", "blending"
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "time_series"
    test_size: float = 0.2
    
    # Hyperparameter search space
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    min_improvement: float = 0.001
    
    # Performance constraints
    max_training_time: int = 300  # 5 minutes per model
    max_memory_usage: float = 0.8  # 80% of available memory
    min_score_threshold: float = 0.0
    
    # Output settings
    save_models: bool = True
    save_results: bool = True
    output_dir: str = "automl_results"
    verbose: bool = True


@dataclass
class AutoMLResult:
    """Result from AutoML optimization."""
    
    # Best model
    best_model: Any
    best_config: TreeModelConfig
    best_score: float
    
    # Model comparison
    model_results: List[TreeModelResult]
    model_rankings: List[Tuple[str, float]]
    
    # Optimization details
    optimization_history: List[Dict[str, Any]]
    total_trials: int
    successful_trials: int
    failed_trials: int
    
    # Performance metrics
    training_time: float
    prediction_time: float
    memory_usage: float
    
    # Ensemble results (if enabled)
    ensemble_model: Optional[Any] = None
    ensemble_score: Optional[float] = None
    ensemble_weights: Optional[Dict[str, float]] = None
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class BaseAutoMLOptimizer(ABC):
    """Abstract base class for AutoML optimizers."""
    
    def __init__(self, config: AutoMLConfig):
        """Initialize AutoML optimizer.
        
        Args:
            config: AutoML configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluator = TreeModelEvaluator()
        
        # Optimization state
        self.trial_results = []
        self.best_score = float('-inf')
        self.best_model = None
        self.best_config = None
        
        self.logger.info(f"✅ {self.__class__.__name__} initialized")
        self.logger.info(f"   Optimization method: {config.optimization_method}")
        self.logger.info(f"   Max trials: {config.max_trials}")
        self.logger.info(f"   Model types: {config.model_types}")
    
    @abstractmethod
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None) -> AutoMLResult:
        """Optimize tree models using AutoML.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            AutoMLResult with optimization results
        """
        pass
    
    def _create_search_space(self, model_type: str) -> Dict[str, Any]:
        """Create search space for a specific model type."""
        try:
            if model_type == "xgboost":
                return {
                    'n_estimators': {'type': 'integer', 'min': 50, 'max': 1000},
                    'max_depth': {'type': 'integer', 'min': 3, 'max': 15},
                    'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3},
                    'subsample': {'type': 'continuous', 'min': 0.6, 'max': 1.0},
                    'colsample_bytree': {'type': 'continuous', 'min': 0.6, 'max': 1.0},
                    'reg_alpha': {'type': 'continuous', 'min': 0.0, 'max': 1.0},
                    'reg_lambda': {'type': 'continuous', 'min': 0.0, 'max': 1.0}
                }
            elif model_type == "lightgbm":
                return {
                    'n_estimators': {'type': 'integer', 'min': 50, 'max': 1000},
                    'max_depth': {'type': 'integer', 'min': 3, 'max': 15},
                    'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3},
                    'num_leaves': {'type': 'integer', 'min': 10, 'max': 100},
                    'subsample': {'type': 'continuous', 'min': 0.6, 'max': 1.0},
                    'colsample_bytree': {'type': 'continuous', 'min': 0.6, 'max': 1.0},
                    'reg_alpha': {'type': 'continuous', 'min': 0.0, 'max': 1.0},
                    'reg_lambda': {'type': 'continuous', 'min': 0.0, 'max': 1.0}
                }
            elif model_type == "catboost":
                return {
                    'iterations': {'type': 'integer', 'min': 50, 'max': 1000},
                    'depth': {'type': 'integer', 'min': 3, 'max': 10},
                    'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3},
                    'l2_leaf_reg': {'type': 'continuous', 'min': 1.0, 'max': 10.0},
                    'border_count': {'type': 'integer', 'min': 32, 'max': 255}
                }
            else:
                # Default search space
                return {
                    'n_estimators': {'type': 'integer', 'min': 50, 'max': 500},
                    'max_depth': {'type': 'integer', 'min': 3, 'max': 15},
                    'learning_rate': {'type': 'continuous', 'min': 0.01, 'max': 0.3}
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create search space for {model_type}: {e}")
            return {}
    
    def _evaluate_model(self, config: TreeModelConfig, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> TreeModelResult:
        """Evaluate a model configuration."""
        try:
            # Create model
            model = EnhancedTreeModelFactory.create_model(config)
            
            # Evaluate model
            result = self.evaluator.evaluate_model(
                model, X_train, y_train, X_val, y_val
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model evaluation failed: {e}")
            return TreeModelResult(
                model=None,
                train_score=0.0,
                val_score=0.0,
                model_type=config.model_type,
                config=config,
                success=False,
                error_message=str(e)
            )
    
    def _create_ensemble(self, model_results: List[TreeModelResult]) -> Optional[Any]:
        """Create ensemble from model results."""
        try:
            if not self.config.enable_ensemble or len(model_results) < 2:
                return None
            
            # Filter successful models
            successful_models = [r for r in model_results if r.success and r.model is not None]
            
            if len(successful_models) < 2:
                return None
            
            # Create ensemble based on method
            if self.config.ensemble_method == "voting":
                return self._create_voting_ensemble(successful_models)
            elif self.config.ensemble_method == "stacking":
                return self._create_stacking_ensemble(successful_models)
            else:
                return self._create_voting_ensemble(successful_models)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble creation failed: {e}")
            return None
    
    def _create_voting_ensemble(self, model_results: List[TreeModelResult]) -> Any:
        """Create voting ensemble."""
        try:
            from sklearn.ensemble import VotingRegressor, VotingClassifier
            
            # Prepare models for ensemble
            estimators = []
            for i, result in enumerate(model_results):
                if result.model is not None:
                    estimators.append((f"model_{i}", result.model))
            
            if len(estimators) < 2:
                return None
            
            # Determine if classification or regression
            task_type = model_results[0].config.task_type
            
            if task_type == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=estimators)
            
            return ensemble
            
        except Exception as e:
            self.logger.warning(f"⚠️ Voting ensemble creation failed: {e}")
            return None
    
    def _create_stacking_ensemble(self, model_results: List[TreeModelResult]) -> Any:
        """Create stacking ensemble."""
        try:
            from sklearn.ensemble import StackingRegressor, StackingClassifier
            
            # Prepare models for ensemble
            estimators = []
            for i, result in enumerate(model_results):
                if result.model is not None:
                    estimators.append((f"model_{i}", result.model))
            
            if len(estimators) < 2:
                return None
            
            # Determine if classification or regression
            task_type = model_results[0].config.task_type
            
            if task_type == "classification":
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=None,  # Use default
                    cv=3
                )
            else:
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=None,  # Use default
                    cv=3
                )
            
            return ensemble
            
        except Exception as e:
            self.logger.warning(f"⚠️ Stacking ensemble creation failed: {e}")
            return None


class OptunaAutoMLOptimizer(BaseAutoMLOptimizer):
    """Optuna-based AutoML optimizer."""
    
    def __init__(self, config: AutoMLConfig):
        """Initialize Optuna AutoML optimizer."""
        super().__init__(config)
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None) -> AutoMLResult:
        """Optimize using Optuna."""
        try:
            self.logger.info("🚀 Starting Optuna AutoML optimization...")
            start_time = time.time()
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.random_state),
                pruner=MedianPruner()
            )
            
            # Define objective function
            def objective(trial):
                # Select model type
                model_type = trial.suggest_categorical('model_type', self.config.model_types)
                
                # Get search space for model type
                search_space = self._create_search_space(model_type)
                
                # Sample parameters
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'integer':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['min'], param_config['max']
                        )
                    elif param_config['type'] == 'continuous':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['min'], param_config['max']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                
                # Create model config
                config = TreeModelConfig(
                    model_type=model_type,
                    task_type="regression",  # Assume regression for now
                    **params
                )
                
                # Evaluate model
                result = self._evaluate_model(config, X_train, y_train, X_val, y_val)
                
                # Store trial result
                self.trial_results.append(result)
                
                # Return validation score
                return result.val_score if result.success else 0.0
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.config.max_trials,
                timeout=self.config.timeout_seconds
            )
            
            # Get best result
            best_trial = study.best_trial
            best_score = study.best_value
            
            # Find best model result
            best_result = None
            for result in self.trial_results:
                if abs(result.val_score - best_score) < 1e-6:
                    best_result = result
                    break
            
            if best_result is None:
                raise ValueError("Could not find best model result")
            
            # Create ensemble if enabled
            ensemble_model = None
            ensemble_score = None
            if self.config.enable_ensemble:
                ensemble_model = self._create_ensemble(self.trial_results)
                if ensemble_model is not None:
                    # Evaluate ensemble
                    ensemble_model.fit(X_train, y_train)
                    ensemble_pred = ensemble_model.predict(X_val)
                    ensemble_score = -mean_squared_error(y_val, ensemble_pred)
            
            # Create optimization history
            optimization_history = []
            for i, trial in enumerate(study.trials):
                optimization_history.append({
                    'trial': i,
                    'score': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                })
            
            # Calculate performance metrics
            training_time = time.time() - start_time
            successful_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            failed_trials = len(study.trials) - successful_trials
            
            self.logger.info(f"✅ Optuna optimization completed in {training_time:.2f}s")
            self.logger.info(f"   Best score: {best_score:.4f}")
            self.logger.info(f"   Successful trials: {successful_trials}")
            self.logger.info(f"   Failed trials: {failed_trials}")
            
            return AutoMLResult(
                best_model=best_result.model,
                best_config=best_result.config,
                best_score=best_score,
                model_results=self.trial_results,
                model_rankings=self._rank_models(),
                optimization_history=optimization_history,
                total_trials=len(study.trials),
                successful_trials=successful_trials,
                failed_trials=failed_trials,
                training_time=training_time,
                prediction_time=0.0,  # Not measured
                memory_usage=0.0,  # Not measured
                ensemble_model=ensemble_model,
                ensemble_score=ensemble_score,
                success=True
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Optuna optimization failed: {e}")
            return AutoMLResult(
                best_model=None,
                best_config=TreeModelConfig(),
                best_score=0.0,
                model_results=[],
                model_rankings=[],
                optimization_history=[],
                total_trials=0,
                successful_trials=0,
                failed_trials=0,
                training_time=training_time,
                prediction_time=0.0,
                memory_usage=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _rank_models(self) -> List[Tuple[str, float]]:
        """Rank models by performance."""
        try:
            rankings = []
            for result in self.trial_results:
                if result.success:
                    rankings.append((result.model_type, result.val_score))
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model ranking failed: {e}")
            return []


class GridSearchAutoMLOptimizer(BaseAutoMLOptimizer):
    """Grid search AutoML optimizer."""
    
    def __init__(self, config: AutoMLConfig):
        """Initialize Grid search AutoML optimizer."""
        super().__init__(config)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None) -> AutoMLResult:
        """Optimize using Grid Search."""
        try:
            self.logger.info("🚀 Starting Grid Search AutoML optimization...")
            start_time = time.time()
            
            # Combine training and validation data for grid search
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            best_score = float('-inf')
            best_result = None
            
            # Test each model type
            for model_type in self.config.model_types:
                try:
                    # Create search space
                    search_space = self._create_search_space(model_type)
                    
                    # Convert to sklearn format
                    param_grid = {}
                    for param_name, param_config in search_space.items():
                        if param_config['type'] == 'integer':
                            param_grid[param_name] = list(range(
                                param_config['min'], 
                                param_config['max'] + 1, 
                                max(1, (param_config['max'] - param_config['min']) // 5)
                            ))
                        elif param_config['type'] == 'continuous':
                            param_grid[param_name] = np.linspace(
                                param_config['min'], 
                                param_config['max'], 
                                5
                            ).tolist()
                        elif param_config['type'] == 'categorical':
                            param_grid[param_name] = param_config['choices']
                    
                    # Create model config
                    base_config = TreeModelConfig(model_type=model_type)
                    
                    # Create model
                    model = EnhancedTreeModelFactory.create_model(base_config)
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        model.model,
                        param_grid,
                        cv=self.config.cv_folds,
                        scoring='neg_mean_squared_error',
                        n_jobs=self.config.n_jobs,
                        verbose=0
                    )
                    
                    grid_search.fit(X_combined, y_combined)
                    
                    # Create result
                    result = TreeModelResult(
                        model=grid_search.best_estimator_,
                        train_score=-grid_search.best_score_,
                        val_score=-grid_search.best_score_,
                        model_type=model_type,
                        config=base_config,
                        success=True
                    )
                    
                    self.trial_results.append(result)
                    
                    # Update best result
                    if -grid_search.best_score_ > best_score:
                        best_score = -grid_search.best_score_
                        best_result = result
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Grid search failed for {model_type}: {e}")
                    continue
            
            # Create ensemble if enabled
            ensemble_model = None
            ensemble_score = None
            if self.config.enable_ensemble:
                ensemble_model = self._create_ensemble(self.trial_results)
            
            # Calculate performance metrics
            training_time = time.time() - start_time
            
            self.logger.info(f"✅ Grid Search optimization completed in {training_time:.2f}s")
            self.logger.info(f"   Best score: {best_score:.4f}")
            
            return AutoMLResult(
                best_model=best_result.model if best_result else None,
                best_config=best_result.config if best_result else TreeModelConfig(),
                best_score=best_score,
                model_results=self.trial_results,
                model_rankings=self._rank_models(),
                optimization_history=[],
                total_trials=len(self.trial_results),
                successful_trials=len([r for r in self.trial_results if r.success]),
                failed_trials=0,
                training_time=training_time,
                prediction_time=0.0,
                memory_usage=0.0,
                ensemble_model=ensemble_model,
                ensemble_score=ensemble_score,
                success=True
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Grid Search optimization failed: {e}")
            return AutoMLResult(
                best_model=None,
                best_config=TreeModelConfig(),
                best_score=0.0,
                model_results=[],
                model_rankings=[],
                optimization_history=[],
                total_trials=0,
                successful_trials=0,
                failed_trials=0,
                training_time=training_time,
                prediction_time=0.0,
                memory_usage=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _rank_models(self) -> List[Tuple[str, float]]:
        """Rank models by performance."""
        try:
            rankings = []
            for result in self.trial_results:
                if result.success:
                    rankings.append((result.model_type, result.val_score))
            
            # Sort by score (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model ranking failed: {e}")
            return []


class TreeAutoMLManager:
    """Manager for Tree AutoML optimization."""
    
    def __init__(self, config: AutoMLConfig):
        """Initialize Tree AutoML manager."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize optimizer based on method
        if config.optimization_method == "optuna" and OPTUNA_AVAILABLE:
            self.optimizer = OptunaAutoMLOptimizer(config)
        elif config.optimization_method == "grid" and SKLEARN_AVAILABLE:
            self.optimizer = GridSearchAutoMLOptimizer(config)
        else:
            # Fallback to grid search
            self.optimizer = GridSearchAutoMLOptimizer(config)
        
        self.logger.info("✅ Tree AutoML Manager initialized")
        self.logger.info(f"   Optimization method: {config.optimization_method}")
        self.logger.info(f"   Model types: {config.model_types}")
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: Optional[np.ndarray] = None,
                 y_test: Optional[np.ndarray] = None) -> AutoMLResult:
        """Optimize tree models using AutoML."""
        return self.optimizer.optimize(X_train, y_train, X_val, y_val, X_test, y_test)


# Convenience functions
def create_tree_automl_manager(config: Optional[AutoMLConfig] = None) -> TreeAutoMLManager:
    """Create Tree AutoML manager instance."""
    if config is None:
        config = AutoMLConfig()
    return TreeAutoMLManager(config)


def quick_automl_optimization(X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             model_types: List[str] = None,
                             max_trials: int = 50) -> AutoMLResult:
    """Quick AutoML optimization with default settings.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_types: List of model types to test
        max_trials: Maximum number of trials
        
    Returns:
        AutoMLResult with optimization results
    """
    if model_types is None:
        model_types = ["xgboost", "lightgbm", "catboost"]
    
    config = AutoMLConfig(
        model_types=model_types,
        max_trials=max_trials,
        optimization_method="optuna" if OPTUNA_AVAILABLE else "grid"
    )
    
    manager = create_tree_automl_manager(config)
    return manager.optimize(X_train, y_train, X_val, y_val)