#!/usr/bin/env python3
"""
Probabilistic Bayesian Optimizer for Tactician and Analyst Models

This module provides Bayesian optimization specifically designed for probabilistic models
that output probability distributions, confidence intervals, and uncertainty estimates.
It optimizes both model hyperparameters and probabilistic output calibration.
"""

import logging
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ProbabilisticOptimizationConfig:
    """Configuration for probabilistic Bayesian optimization."""
    
    # Optimization objectives
    objectives: List[str] = None  # ['calibration', 'sharpness', 'discrimination']
    
    # Calibration metrics
    calibration_bins: int = 10
    reliability_threshold: float = 0.1
    
    # Uncertainty quantification
    uncertainty_weight: float = 0.3
    confidence_calibration_weight: float = 0.4
    prediction_accuracy_weight: float = 0.3
    
    # Optimization parameters
    n_trials: int = 100
    n_jobs: int = 1
    timeout: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: int = 10
    min_trials: int = 20
    
    # Sampling strategy
    sampler_type: str = "tpe"  # 'tpe', 'cmaes', 'random'
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['calibration', 'sharpness', 'discrimination']


class ProbabilisticBayesianOptimizer:
    """
    Bayesian optimizer specifically designed for probabilistic models.
    
    This optimizer focuses on:
    1. Calibration: Ensuring predicted probabilities match observed frequencies
    2. Sharpness: Making predictions as precise as possible
    3. Discrimination: Maximizing the difference between positive and negative predictions
    4. Uncertainty quantification: Optimizing confidence intervals and uncertainty estimates
    """
    
    def __init__(
        self,
        config: ProbabilisticOptimizationConfig,
        model_type: str = "tactician",  # 'tactician' or 'analyst'
        storage_url: str = "sqlite:///probabilistic_optuna.db"
    ):
        self.config = config
        self.model_type = model_type
        self.storage_url = storage_url
        self.logger = logging.getLogger(__name__)
        
        # Initialize Optuna study
        self.study = self._create_study()
        
        # Model-specific configurations
        self.model_configs = self._get_model_configurations()
        
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner."""
        
        # Choose sampler based on configuration
        if self.config.sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler(seed=42)
        elif self.config.sampler_type == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)
        
        # Create study with multi-objective optimization
        study = optuna.create_study(
            study_name=f"probabilistic_{self.model_type}_optimization",
            storage=self.storage_url,
            sampler=sampler,
            directions=["maximize"] * len(self.config.objectives),
            load_if_exists=True
        )
        
        return study
    
    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get model-specific hyperparameter search spaces."""
        
        if self.model_type == "tactician":
            return {
                "base_model": {
                    "n_estimators": (100, 1000),
                    "max_depth": (3, 15),
                    "learning_rate": (0.01, 0.3),
                    "subsample": (0.6, 1.0),
                    "colsample_bytree": (0.6, 1.0),
                    "reg_alpha": (0.0, 1.0),
                    "reg_lambda": (0.0, 1.0)
                },
                "probabilistic_calibration": {
                    "calibration_method": ["isotonic", "sigmoid", "platt"],
                    "calibration_cv_folds": (3, 10),
                    "uncertainty_estimation": ["mc_dropout", "ensemble", "gaussian"]
                },
                "barrier_system": {
                    "upper_barrier_multiplier": (0.3, 0.8),
                    "lower_barrier_multiplier": (0.1, 0.5),
                    "confidence_threshold": (0.6, 0.9),
                    "precision_threshold": (0.7, 0.95)
                }
            }
        else:  # analyst
            return {
                "base_model": {
                    "n_estimators": (200, 2000),
                    "max_depth": (5, 20),
                    "learning_rate": (0.005, 0.2),
                    "subsample": (0.7, 1.0),
                    "colsample_bytree": (0.7, 1.0),
                    "reg_alpha": (0.0, 2.0),
                    "reg_lambda": (0.0, 2.0)
                },
                "probabilistic_calibration": {
                    "calibration_method": ["isotonic", "sigmoid", "platt", "temperature"],
                    "calibration_cv_folds": (5, 15),
                    "uncertainty_estimation": ["mc_dropout", "ensemble", "gaussian", "conformal"]
                },
                "regime_detection": {
                    "regime_threshold": (0.5, 0.8),
                    "regime_confidence_threshold": (0.6, 0.9),
                    "regime_transition_smoothing": (0.1, 0.5)
                }
            }
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial."""
        
        params = {}
        
        # Base model parameters
        base_config = self.model_configs["base_model"]
        for param, (low, high) in base_config.items():
            if isinstance(low, int):
                params[param] = trial.suggest_int(param, low, high)
            else:
                params[param] = trial.suggest_float(param, low, high, log=True)
        
        # Probabilistic calibration parameters
        calib_config = self.model_configs["probabilistic_calibration"]
        params["calibration_method"] = trial.suggest_categorical(
            "calibration_method", calib_config["calibration_method"]
        )
        params["calibration_cv_folds"] = trial.suggest_int(
            "calibration_cv_folds", 
            calib_config["calibration_cv_folds"][0],
            calib_config["calibration_cv_folds"][1]
        )
        params["uncertainty_estimation"] = trial.suggest_categorical(
            "uncertainty_estimation", calib_config["uncertainty_estimation"]
        )
        
        # Model-specific parameters
        if self.model_type == "tactician":
            barrier_config = self.model_configs["barrier_system"]
            params["upper_barrier_multiplier"] = trial.suggest_float(
                "upper_barrier_multiplier",
                barrier_config["upper_barrier_multiplier"][0],
                barrier_config["upper_barrier_multiplier"][1]
            )
            params["lower_barrier_multiplier"] = trial.suggest_float(
                "lower_barrier_multiplier",
                barrier_config["lower_barrier_multiplier"][0],
                barrier_config["lower_barrier_multiplier"][1]
            )
            params["confidence_threshold"] = trial.suggest_float(
                "confidence_threshold",
                barrier_config["confidence_threshold"][0],
                barrier_config["confidence_threshold"][1]
            )
            params["precision_threshold"] = trial.suggest_float(
                "precision_threshold",
                barrier_config["precision_threshold"][0],
                barrier_config["precision_threshold"][1]
            )
        else:
            regime_config = self.model_configs["regime_detection"]
            params["regime_threshold"] = trial.suggest_float(
                "regime_threshold",
                regime_config["regime_threshold"][0],
                regime_config["regime_threshold"][1]
            )
            params["regime_confidence_threshold"] = trial.suggest_float(
                "regime_confidence_threshold",
                regime_config["regime_confidence_threshold"][0],
                regime_config["regime_confidence_threshold"][1]
            )
            params["regime_transition_smoothing"] = trial.suggest_float(
                "regime_transition_smoothing",
                regime_config["regime_transition_smoothing"][0],
                regime_config["regime_transition_smoothing"][1]
            )
        
        return params
    
    def evaluate_probabilistic_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        confidence_intervals: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate probabilistic model performance metrics."""
        
        metrics = {}
        
        # Calibration metrics
        if "calibration" in self.config.objectives:
            metrics["calibration"] = self._calculate_calibration_score(y_true, y_pred_proba)
        
        # Sharpness metrics
        if "sharpness" in self.config.objectives:
            metrics["sharpness"] = self._calculate_sharpness_score(y_pred_proba)
        
        # Discrimination metrics
        if "discrimination" in self.config.objectives:
            metrics["discrimination"] = self._calculate_discrimination_score(y_true, y_pred_proba)
        
        # Uncertainty quantification metrics
        if confidence_intervals is not None:
            metrics["uncertainty_quality"] = self._calculate_uncertainty_quality(
                y_true, y_pred_proba, confidence_intervals
            )
        
        return metrics
    
    def _calculate_calibration_score(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> float:
        """Calculate calibration score (lower is better)."""
        try:
            # Use Brier score for calibration
            return brier_score_loss(y_true, y_pred_proba)
        except:
            return 1.0  # Worst possible score
    
    def _calculate_sharpness_score(self, y_pred_proba: np.ndarray) -> float:
        """Calculate sharpness score (higher is better)."""
        try:
            # Sharpness is the negative entropy of predictions
            # We want predictions to be confident (low entropy)
            entropy = -np.mean(y_pred_proba * np.log(y_pred_proba + 1e-10))
            return -entropy  # Negative because we want to maximize
        except:
            return 0.0
    
    def _calculate_discrimination_score(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> float:
        """Calculate discrimination score (higher is better)."""
        try:
            # Use ROC AUC for discrimination
            return roc_auc_score(y_true, y_pred_proba)
        except:
            return 0.5  # Random performance
    
    def _calculate_uncertainty_quality(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        confidence_intervals: np.ndarray
    ) -> float:
        """Calculate uncertainty quantification quality."""
        try:
            # Check if true values fall within confidence intervals
            coverage = np.mean(
                (y_true >= confidence_intervals[:, 0]) & 
                (y_true <= confidence_intervals[:, 1])
            )
            return coverage
        except:
            return 0.0
    
    def create_objective_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        validation_split: float = 0.2
    ) -> Callable:
        """Create the objective function for optimization."""
        
        def objective(trial: optuna.Trial) -> Tuple[float, ...]:
            """Objective function for multi-objective optimization."""
            
            try:
                # Get hyperparameters for this trial
                params = self.suggest_hyperparameters(trial)
                
                # Split data for validation
                n_val = int(len(X) * validation_split)
                X_train, X_val = X[:-n_val], X[-n_val:]
                y_train, y_val = y[:-n_val], y[-n_val:]
                
                # Create and train model
                model = model_factory(params)
                model.fit(X_train, y_train)
                
                # Get probabilistic predictions
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Get confidence intervals if available
                confidence_intervals = None
                if hasattr(model, 'predict_proba_with_confidence'):
                    confidence_intervals = model.predict_proba_with_confidence(X_val)
                
                # Calculate metrics
                metrics = self.evaluate_probabilistic_metrics(
                    y_val, y_pred_proba, confidence_intervals
                )
                
                # Return objectives in the order specified
                objectives = []
                for obj_name in self.config.objectives:
                    if obj_name in metrics:
                        objectives.append(metrics[obj_name])
                    else:
                        objectives.append(0.0)  # Default value
                
                return tuple(objectives)
                
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {e}")
                # Return worst possible scores
                return tuple([0.0] * len(self.config.objectives))
        
        return objective
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Run the Bayesian optimization."""
        
        self.logger.info(f"Starting probabilistic Bayesian optimization for {self.model_type}")
        self.logger.info(f"Objectives: {self.config.objectives}")
        self.logger.info(f"Number of trials: {self.config.n_trials}")
        
        # Create objective function
        objective = self.create_objective_function(X, y, model_factory, validation_split)
        
        # Set up callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                optuna.callbacks.EarlyStoppingCallback(
                    self.config.early_stopping_patience,
                    directions=["maximize"] * len(self.config.objectives)
                )
            )
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            timeout=self.config.timeout,
            callbacks=callbacks
        )
        
        # Extract results
        results = self._extract_optimization_results()
        
        self.logger.info("Probabilistic Bayesian optimization completed successfully!")
        
        return results
    
    def _extract_optimization_results(self) -> Dict[str, Any]:
        """Extract and format optimization results."""
        
        # Get Pareto front solutions
        pareto_front = self.study.best_trials
        
        # Get best solution for each objective
        best_solutions = {}
        for i, objective in enumerate(self.config.objectives):
            best_trial = min(pareto_front, key=lambda t: t.values[i])
            best_solutions[objective] = {
                "params": best_trial.params,
                "value": best_trial.values[i],
                "trial_number": best_trial.number
            }
        
        # Get parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
        except:
            param_importance = {}
        
        # Get optimization history
        optimization_history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                optimization_history.append({
                    "trial_number": trial.number,
                    "values": trial.values,
                    "params": trial.params,
                    "duration": trial.duration.total_seconds()
                })
        
        return {
            "best_solutions": best_solutions,
            "pareto_front": pareto_front,
            "parameter_importance": param_importance,
            "optimization_history": optimization_history,
            "study": self.study,
            "config": self.config
        }
    
    def get_recommended_hyperparameters(self, objective_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Get recommended hyperparameters based on objective weights."""
        
        if objective_weights is None:
            # Default weights: equal importance
            objective_weights = {obj: 1.0 for obj in self.config.objectives}
        
        # Calculate weighted score for each trial
        best_trial = None
        best_weighted_score = float('-inf')
        
        for trial in self.study.best_trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                weighted_score = sum(
                    objective_weights[obj] * trial.values[i] 
                    for i, obj in enumerate(self.config.objectives)
                )
                
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_trial = trial
        
        if best_trial:
            return {
                "hyperparameters": best_trial.params,
                "objective_values": dict(zip(self.config.objectives, best_trial.values)),
                "weighted_score": best_weighted_score,
                "trial_number": best_trial.number
            }
        else:
            return {}
    
    def plot_optimization_results(self, save_path: Optional[str] = None):
        """Plot optimization results using Optuna's visualization tools."""
        
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for each objective
            fig, axes = plt.subplots(1, len(self.config.objectives), figsize=(5*len(self.config.objectives), 5))
            if len(self.config.objectives) == 1:
                axes = [axes]
            
            for i, objective in enumerate(self.config.objectives):
                # Plot optimization history for this objective
                values = [trial.values[i] for trial in self.study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                trial_numbers = [trial.number for trial in self.study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                
                axes[i].plot(trial_numbers, values, 'b-', alpha=0.6)
                axes[i].set_title(f'{objective.capitalize()} Optimization History')
                axes[i].set_xlabel('Trial Number')
                axes[i].set_ylabel(objective.capitalize())
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Optimization plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting optimization results: {e}")


# Example usage and model factories
def create_tactician_model(params: Dict[str, Any]):
    """Factory function for creating Tactician models."""
    # This would integrate with your existing Tactician model creation
    # For now, returning a placeholder
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 10),
        random_state=42,
        n_jobs=1
    )
    
    return model


def create_analyst_model(params: Dict[str, Any]):
    """Factory function for creating Analyst models."""
    # This would integrate with your existing Analyst model creation
    # For now, returning a placeholder
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=params.get('n_estimators', 200),
        max_depth=params.get('max_depth', 15),
        random_state=42,
        n_jobs=1
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    config = ProbabilisticOptimizationConfig(
        objectives=['calibration', 'sharpness', 'discrimination'],
        n_trials=50,
        n_jobs=1
    )
    
    # Create optimizer for Tactician
    tactician_optimizer = ProbabilisticBayesianOptimizer(
        config=config,
        model_type="tactician"
    )
    
    # Create optimizer for Analyst
    analyst_optimizer = ProbabilisticBayesianOptimizer(
        config=config,
        model_type="analyst"
    )
    
    print("✅ Probabilistic Bayesian Optimizer created successfully!")
    print(f"Tactician optimizer: {tactician_optimizer}")
    print(f"Analyst optimizer: {analyst_optimizer}")