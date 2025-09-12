"""
Hierarchical Hyperparameter Optimization for Multi-Output Stacking Ensemble

This module implements the recommended HPO strategy:
1. Phase 1: Optimize base models first
2. Phase 2: Optimize meta models with fixed base models
3. Ensures proper timing and prevents meta model overfitting to poor base models
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import json
from pathlib import Path

# HPO imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from src.utils.logger import system_logger

logger = system_logger.getChild('HierarchicalHPO')

@dataclass
class HPOPhaseConfig:
    """Configuration for each HPO phase."""
    phase_name: str
    models: Dict[str, Any]
    search_spaces: Dict[str, Dict[str, Any]]
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    enable_pruning: bool = True
    cv_folds: int = 5
    scoring_metric: str = 'neg_mean_squared_error'
    direction: str = 'maximize'

@dataclass
class HPOPhaseResult:
    """Result of a single HPO phase."""
    phase_name: str
    best_models: Dict[str, Any]
    best_scores: Dict[str, float]
    optimization_time: float
    n_trials: int
    best_params: Dict[str, Dict[str, Any]]
    optimization_history: List[Dict[str, Any]]

@dataclass
class HierarchicalHPOConfig:
    """Configuration for hierarchical HPO."""
    # Phase 1: Base Model HPO
    phase1_config: HPOPhaseConfig
    
    # Phase 2: Meta Model HPO
    phase2_config: HPOPhaseConfig
    
    # General settings
    enable_caching: bool = True
    cache_dir: str = "./hpo_cache"
    enable_parallel: bool = True
    max_workers: int = 4
    random_state: int = 42
    
    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_time_series_cv: bool = True

class HierarchicalHPO:
    """
    Hierarchical Hyperparameter Optimization for Multi-Output Stacking Ensemble.
    
    This class implements the recommended two-phase HPO strategy:
    1. Phase 1: Optimize base models individually
    2. Phase 2: Optimize meta models with fixed base models
    """
    
    def __init__(self, config: HierarchicalHPOConfig):
        """Initialize hierarchical HPO."""
        self.config = config
        self.logger = logger.getChild('HierarchicalHPO')
        
        # Validate dependencies
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for HPO functionality")
        
        # Initialize results
        self.phase1_result: Optional[HPOPhaseResult] = None
        self.phase2_result: Optional[HPOPhaseResult] = None
        self.final_models: Dict[str, Any] = {}
        
        # Create cache directory
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Hierarchical HPO initialized")
    
    def optimize_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform hierarchical hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (optional)
            
        Returns:
            Dictionary containing optimized models and results
        """
        self.logger.info("🚀 Starting hierarchical HPO optimization")
        start_time = time.time()
        
        # Prepare data
        X_val, y_val = self._prepare_validation_data(X_train, y_train, X_val, y_val)
        
        # Phase 1: Base Model HPO
        self.logger.info("🔄 Phase 1: Optimizing base models...")
        phase1_start = time.time()
        
        self.phase1_result = self._optimize_phase(
            phase_config=self.config.phase1_config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_names
        )
        
        phase1_time = time.time() - phase1_start
        self.logger.info(f"✅ Phase 1 completed in {phase1_time:.2f}s")
        
        # Phase 2: Meta Model HPO with optimized base models
        self.logger.info("🔄 Phase 2: Optimizing meta models with fixed base models...")
        phase2_start = time.time()
        
        # Create meta features using optimized base models
        meta_features = self._create_meta_features(X_val, self.phase1_result.best_models)
        
        self.phase2_result = self._optimize_phase(
            phase_config=self.config.phase2_config,
            X_train=X_train,
            y_train=y_train,
            X_val=meta_features,
            y_val=y_val,
            feature_names=feature_names,
            base_models=self.phase1_result.best_models
        )
        
        phase2_time = time.time() - phase2_start
        self.logger.info(f"✅ Phase 2 completed in {phase2_time:.2f}s")
        
        # Combine results
        total_time = time.time() - start_time
        self.final_models = {
            'base_models': self.phase1_result.best_models,
            'meta_models': self.phase2_result.best_models,
            'optimization_time': total_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time
        }
        
        self.logger.info(f"✅ Hierarchical HPO completed in {total_time:.2f}s")
        self.logger.info(f"📊 Phase 1: {len(self.phase1_result.best_models)} base models optimized")
        self.logger.info(f"📊 Phase 2: {len(self.phase2_result.best_models)} meta models optimized")
        
        return self.final_models
    
    def _optimize_phase(
        self,
        phase_config: HPOPhaseConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        base_models: Optional[Dict[str, Any]] = None
    ) -> HPOPhaseResult:
        """Optimize a single phase."""
        
        self.logger.info(f"🔄 Optimizing phase: {phase_config.phase_name}")
        start_time = time.time()
        
        best_models = {}
        best_scores = {}
        best_params = {}
        optimization_history = []
        
        # Create study for each model
        for model_name, model in phase_config.models.items():
            self.logger.info(f"🔄 Optimizing {model_name}...")
            
            # Create Optuna study
            study = optuna.create_study(
                direction=phase_config.direction,
                sampler=TPESampler(seed=self.config.random_state),
                pruner=MedianPruner() if phase_config.enable_pruning else None
            )
            
            # Define objective function
            def objective(trial):
                return self._objective_function(
                    trial=trial,
                    model=model,
                    model_name=model_name,
                    search_space=phase_config.search_spaces[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    cv_folds=phase_config.cv_folds,
                    scoring_metric=phase_config.scoring_metric,
                    base_models=base_models
                )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=phase_config.n_trials,
                timeout=phase_config.timeout_seconds
            )
            
            # Get best result
            best_trial = study.best_trial
            best_models[model_name] = self._create_optimized_model(
                model, best_trial.params, base_models
            )
            best_scores[model_name] = best_trial.value
            best_params[model_name] = best_trial.params
            
            # Record history
            optimization_history.append({
                'model_name': model_name,
                'n_trials': len(study.trials),
                'best_score': best_trial.value,
                'best_params': best_trial.params,
                'optimization_time': time.time() - start_time
            })
            
            self.logger.info(f"✅ {model_name} optimized: {best_trial.value:.4f}")
        
        return HPOPhaseResult(
            phase_name=phase_config.phase_name,
            best_models=best_models,
            best_scores=best_scores,
            optimization_time=time.time() - start_time,
            n_trials=sum(len(study.trials) for study in [optuna.create_study()] * len(phase_config.models)),
            best_params=best_params,
            optimization_history=optimization_history
        )
    
    def _objective_function(
        self,
        trial: optuna.Trial,
        model: Any,
        model_name: str,
        search_space: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_folds: int,
        scoring_metric: str,
        base_models: Optional[Dict[str, Any]] = None
    ) -> float:
        """Objective function for Optuna optimization."""
        
        try:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial, search_space)
            
            # Create model with sampled parameters
            optimized_model = self._create_optimized_model(model, params, base_models)
            
            # Perform cross-validation
            if cv_folds > 1:
                scores = self._cross_validate_model(
                    optimized_model, X_train, y_train, cv_folds, scoring_metric
                )
                return np.mean(scores)
            else:
                # Single validation
                optimized_model.fit(X_train, y_train)
                y_pred = optimized_model.predict(X_val)
                
                if scoring_metric == 'neg_mean_squared_error':
                    from sklearn.metrics import mean_squared_error
                    return -mean_squared_error(y_val, y_pred)
                elif scoring_metric == 'neg_mean_absolute_error':
                    from sklearn.metrics import mean_absolute_error
                    return -mean_absolute_error(y_val, y_pred)
                elif scoring_metric == 'r2':
                    from sklearn.metrics import r2_score
                    return r2_score(y_val, y_pred)
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring_metric}")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Trial failed for {model_name}: {e}")
            return float('-inf')
    
    def _sample_hyperparameters(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'], log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")
        
        return params
    
    def _create_optimized_model(self, base_model: Any, params: Dict[str, Any], base_models: Optional[Dict[str, Any]] = None) -> Any:
        """Create model with optimized parameters."""
        
        # Clone the base model
        from sklearn.base import clone
        optimized_model = clone(base_model)
        
        # Set parameters
        optimized_model.set_params(**params)
        
        return optimized_model
    
    def _cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int, scoring_metric: str) -> List[float]:
        """Perform cross-validation on model."""
        
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring=scoring_metric, n_jobs=1
        )
        
        return scores.tolist()
    
    def _create_meta_features(self, X: np.ndarray, base_models: Dict[str, Any]) -> np.ndarray:
        """Create meta features using base model predictions."""
        
        meta_features = []
        
        for model_name, model in base_models.items():
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                meta_features.append(pred)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                # Add zero predictions as fallback
                meta_features.append(np.zeros((len(X), 1)))
        
        return np.hstack(meta_features)
    
    def _prepare_validation_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare validation data."""
        
        if X_val is not None and y_val is not None:
            return X_val, y_val
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_split,
            random_state=self.config.random_state
        )
        
        return X_val_split, y_val_split
    
    def save_results(self, filepath: str) -> None:
        """Save optimization results to file."""
        
        results = {
            'phase1_result': {
                'phase_name': self.phase1_result.phase_name,
                'best_scores': self.phase1_result.best_scores,
                'best_params': self.phase1_result.best_params,
                'optimization_time': self.phase1_result.optimization_time,
                'n_trials': self.phase1_result.n_trials
            },
            'phase2_result': {
                'phase_name': self.phase2_result.phase_name,
                'best_scores': self.phase2_result.best_scores,
                'best_params': self.phase2_result.best_params,
                'optimization_time': self.phase2_result.optimization_time,
                'n_trials': self.phase2_result.n_trials
            },
            'final_models': self.final_models
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to {filepath}")

# Convenience functions
def create_hierarchical_hpo_config(
    base_models: Dict[str, Any],
    meta_models: Dict[str, Any],
    base_search_spaces: Dict[str, Dict[str, Any]],
    meta_search_spaces: Dict[str, Dict[str, Any]],
    n_trials_base: int = 100,
    n_trials_meta: int = 50
) -> HierarchicalHPOConfig:
    """Create hierarchical HPO configuration."""
    
    phase1_config = HPOPhaseConfig(
        phase_name="base_models",
        models=base_models,
        search_spaces=base_search_spaces,
        n_trials=n_trials_base
    )
    
    phase2_config = HPOPhaseConfig(
        phase_name="meta_models",
        models=meta_models,
        search_spaces=meta_search_spaces,
        n_trials=n_trials_meta
    )
    
    return HierarchicalHPOConfig(
        phase1_config=phase1_config,
        phase2_config=phase2_config
    )

def optimize_stacking_ensemble(
    base_models: Dict[str, Any],
    meta_models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_search_spaces: Dict[str, Dict[str, Any]],
    meta_search_spaces: Dict[str, Dict[str, Any]],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Optimize a stacking ensemble using hierarchical HPO."""
    
    config = create_hierarchical_hpo_config(
        base_models, meta_models, base_search_spaces, meta_search_spaces
    )
    
    hpo = HierarchicalHPO(config)
    return hpo.optimize_ensemble(X_train, y_train, X_val, y_val)