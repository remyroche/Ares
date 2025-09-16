"""
Tactician Directional Optimization - Enhanced Model Training

This module enhances the Tactician model training to directly optimize for:
1. directional_accuracy: How often the entry direction is correct
2. adverse_movement_minimization: Minimize adverse price movement
3. directional_profit_efficiency: Profit from correct directional moves
4. risk_adjusted_performance: Risk-adjusted returns

The enhancements include:
- Custom loss functions for directional optimization
- Multi-objective training with Pareto front analysis
- Advanced feature engineering for directional prediction
- Risk-aware model architecture
- Enhanced evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core ML imports
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNetCV
import optuna
from optuna.samplers import TPESampler

# Enhanced utilities
from src.utils.ml_common.optimization.pareto import (
    ParetoFront, Solution, ObjectiveDirection, 
    scalarize_financial_goals, select_knee_point
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite
)

# Base training imports
from .tactician_models_training_refactored import TacticianModelsTrainingStepRefactored
from src.utils.ml_common.config import TacticianTrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class DirectionalOptimizationResult:
    """Result of directional optimization for Tactician."""
    model: Any
    directional_accuracy: float
    adverse_movement_minimization: float
    directional_profit_efficiency: float
    risk_adjusted_performance: float
    composite_score: float
    optimization_time: float
    n_trials: int
    optimization_history: List[Dict[str, Any]]


class DirectionalLossFunction:
    """Custom loss functions for directional optimization."""
    
    @staticmethod
    def directional_accuracy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Loss function that maximizes directional accuracy.
        
        Args:
            y_true: True directional labels (1 for up, -1 for down, 0 for neutral)
            y_pred: Predicted directional probabilities
            
        Returns:
            Negative directional accuracy (for minimization)
        """
        # Convert predictions to directional labels
        y_pred_direction = np.sign(y_pred)
        
        # Calculate directional accuracy
        correct_directions = np.sum(y_true * y_pred_direction > 0)
        total_predictions = np.sum(y_true != 0)
        
        if total_predictions == 0:
            return 1.0  # Worst possible loss
        
        directional_accuracy = correct_directions / total_predictions
        return 1.0 - directional_accuracy  # Return loss (1 - accuracy)
    
    @staticmethod
    def adverse_movement_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                            adverse_threshold: float = 0.005) -> float:  # 0.5% threshold
        """
        Loss function that minimizes adverse price movement for short-term 0.5% movements.
        
        Args:
            y_true: True price movements
            y_pred: Predicted price movements
            adverse_threshold: Threshold for adverse movement (0.5% for short-term focus)
            
        Returns:
            Adverse movement loss
        """
        # Calculate prediction errors
        errors = np.abs(y_true - y_pred)
        
        # Identify adverse movements (large errors in wrong direction)
        adverse_mask = (np.sign(y_true) != np.sign(y_pred)) & (errors > adverse_threshold)
        adverse_movements = np.sum(adverse_mask)
        total_movements = len(y_true)
        
        if total_movements == 0:
            return 1.0
        
        adverse_ratio = adverse_movements / total_movements
        return adverse_ratio
    
    @staticmethod
    def directional_profit_efficiency_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Loss function that maximizes profit efficiency from correct directional moves.
        
        Args:
            y_true: True price movements
            y_pred: Predicted price movements
            
        Returns:
            Negative profit efficiency (for minimization)
        """
        # Calculate profit from correct directional predictions
        correct_direction_mask = np.sign(y_true) == np.sign(y_pred)
        correct_movements = y_true[correct_direction_mask]
        
        if len(correct_movements) == 0:
            return 1.0  # No correct predictions
        
        # Calculate profit efficiency
        total_profit = np.sum(np.abs(correct_movements))
        max_possible_profit = np.sum(np.abs(y_true))
        
        if max_possible_profit == 0:
            return 1.0
        
        profit_efficiency = total_profit / max_possible_profit
        return 1.0 - profit_efficiency  # Return loss (1 - efficiency)
    
    @staticmethod
    def risk_adjusted_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Loss function that optimizes risk-adjusted performance.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            
        Returns:
            Negative risk-adjusted performance (for minimization)
        """
        # Calculate returns
        returns = y_pred - y_true
        
        # Calculate risk-adjusted performance (Sharpe-like ratio)
        mean_return = np.mean(returns)
        return_std = np.std(returns)
        
        if return_std == 0:
            return 1.0  # No risk, but also no return
        
        risk_adjusted_performance = mean_return / return_std
        return -risk_adjusted_performance  # Return negative for minimization


# DirectionalFeatureEngineer removed - using existing features from base training


class DirectionalTacticianOptimizer:
    """
    Enhanced Tactician optimizer that directly optimizes for directional goals.
    """
    
    def __init__(self, 
                 config: Optional[TacticianTrainingConfig] = None,
                 use_gpu: bool = True,
                 use_memory_optimization: bool = True):
        """
        Initialize directional Tactician optimizer.
        
        Args:
            config: Tactician training configuration
            use_gpu: Whether to use GPU acceleration
            use_memory_optimization: Whether to use memory optimization
        """
        self.config = config or TacticianTrainingConfig()
        self.use_gpu = use_gpu
        self.use_memory_optimization = use_memory_optimization
        
        self.logger = logger.getChild('DirectionalTacticianOptimizer')
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager() if use_gpu else None
        self.memory_optimizer = get_m1_memory_optimizer() if use_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Using existing features from base training - no additional feature engineering
        
        # Initialize Pareto front analyzer
        self.pareto_front = ParetoFront()
        
        # Optimization history
        self.optimization_history = []
        
        self.logger.info("🚀 Directional Tactician optimizer initialized")
    
    def optimize_tactician_directionally(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       regime_labels: np.ndarray,
                                       feature_names: Optional[List[str]] = None,
                                       hmm_states: Optional[np.ndarray] = None,
                                       analyst_signals: Optional[np.ndarray] = None,
                                       analyst_model_outputs: Optional[np.ndarray] = None,
                                       hmm_regime_features: Optional[np.ndarray] = None,
                                       all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
                                       hmm_model_outputs: Optional[np.ndarray] = None,
                                       analyst_ensemble_outputs: Optional[np.ndarray] = None,
                                       max_trials: int = 100) -> DirectionalOptimizationResult:
        """
        Optimize Tactician model for directional objectives.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Feature names
            hmm_states: HMM states
            analyst_signals: Analyst signals
            analyst_model_outputs: Analyst model outputs
            hmm_regime_features: HMM regime features
            all_analyst_models_outputs: All analyst model outputs
            hmm_model_outputs: HMM model outputs
            analyst_ensemble_outputs: Analyst ensemble outputs
            max_trials: Maximum optimization trials
            
        Returns:
            DirectionalOptimizationResult with optimized model
        """
        start_time = time.time()
        
        self.logger.info("🔄 Starting directional Tactician optimization...")
        
        # Step 1: Use existing features (no additional feature engineering)
        self.logger.info("📊 Step 1: Using existing features for directional optimization")
        X_enhanced = X  # Use existing features as-is
        
        # Step 2: Multi-objective optimization
        self.logger.info("📊 Step 2: Multi-objective directional optimization")
        optimization_result = self._multi_objective_directional_optimization(
            X_enhanced, y, regime_labels, max_trials
        )
        
        # Step 3: Train final model with best parameters
        self.logger.info("📊 Step 3: Training final directional model")
        final_model = self._train_final_directional_model(
            X_enhanced, y, optimization_result
        )
        
        # Step 4: Evaluate directional performance
        self.logger.info("📊 Step 4: Evaluating directional performance")
        directional_metrics = self._evaluate_directional_performance(final_model, X_enhanced, y)
        
        # Create result
        result = DirectionalOptimizationResult(
            model=final_model,
            directional_accuracy=directional_metrics['directional_accuracy'],
            adverse_movement_minimization=directional_metrics['adverse_movement_minimization'],
            directional_profit_efficiency=directional_metrics['directional_profit_efficiency'],
            risk_adjusted_performance=directional_metrics['risk_adjusted_performance'],
            composite_score=directional_metrics['composite_score'],
            optimization_time=time.time() - start_time,
            n_trials=max_trials,
            optimization_history=self.optimization_history.copy()
        )
        
        self.logger.info(f"✅ Directional optimization completed in {result.optimization_time:.2f}s")
        self.logger.info(f"   Directional accuracy: {result.directional_accuracy:.4f}")
        self.logger.info(f"   Adverse movement min: {result.adverse_movement_minimization:.4f}")
        self.logger.info(f"   Profit efficiency: {result.directional_profit_efficiency:.4f}")
        self.logger.info(f"   Risk-adjusted perf: {result.risk_adjusted_performance:.4f}")
        self.logger.info(f"   Composite score: {result.composite_score:.4f}")
        
        return result
    
    # Feature enhancement removed - using existing features from base training
    
    def _multi_objective_directional_optimization(self,
                                                X: np.ndarray,
                                                y: np.ndarray,
                                                regime_labels: np.ndarray,
                                                max_trials: int) -> Dict[str, Any]:
        """Multi-objective optimization for directional goals."""
        solutions = []
        
        def objective(trial):
            # Suggest model parameters
            model_type = trial.suggest_categorical('model_type', ['ElasticNetCV', 'Ridge', 'Lasso'])
            
            if model_type == 'ElasticNetCV':
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
                alpha = trial.suggest_float('alpha', 0.001, 1.0, log=True)
                model = ElasticNetCV(l1_ratio=[l1_ratio], alphas=[alpha], cv=5)
            elif model_type == 'Ridge':
                alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=alpha)
            else:  # Lasso
                alpha = trial.suggest_float('alpha', 0.001, 1.0, log=True)
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=alpha)
            
            # Train model
            model.fit(X, y)
            
            # Evaluate directional metrics
            metrics = self._evaluate_directional_metrics(model, X, y)
            
            # Store solution
            solution = Solution(
                metrics=metrics,
                params={
                    'model_type': model_type,
                    'l1_ratio': l1_ratio if model_type == 'ElasticNetCV' else None,
                    'alpha': alpha
                }
            )
            solutions.append(solution)
            
            # Return composite score for Optuna
            return metrics['composite_score']
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=max_trials, n_jobs=-1)
        
        # Find best solution
        best_solution = max(solutions, key=lambda s: s.metrics['composite_score'])
        
        return {
            'best_solution': best_solution,
            'all_solutions': solutions,
            'study': study
        }
    
    def _evaluate_directional_metrics(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate directional metrics for a model."""
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate directional accuracy
        y_direction = np.sign(y)
        y_pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(y_direction == y_pred_direction)
        
        # Calculate adverse movement minimization (focused on short-term 0.5% movements)
        adverse_movements = np.sum((y_direction != y_pred_direction) & (np.abs(y) > 0.005))  # 0.5% threshold
        total_movements = np.sum(y_direction != 0)
        adverse_movement_minimization = 1 - (adverse_movements / total_movements) if total_movements > 0 else 1.0
        
        # Calculate directional profit efficiency
        correct_direction_mask = y_direction == y_pred_direction
        if np.sum(correct_direction_mask) > 0:
            correct_returns = np.abs(y[correct_direction_mask])
            total_returns = np.abs(y[y_direction != 0])
            directional_profit_efficiency = np.sum(correct_returns) / np.sum(total_returns) if np.sum(total_returns) > 0 else 0.0
        else:
            directional_profit_efficiency = 0.0
        
        # Calculate risk-adjusted performance
        returns = y_pred - y
        mean_return = np.mean(returns)
        return_std = np.std(returns)
        risk_adjusted_performance = mean_return / return_std if return_std > 0 else 0.0
        
        # Calculate composite score
        composite_score = (
            0.4 * directional_accuracy +
            0.3 * adverse_movement_minimization +
            0.2 * directional_profit_efficiency +
            0.1 * risk_adjusted_performance
        )
        
        return {
            'directional_accuracy': directional_accuracy,
            'adverse_movement_minimization': adverse_movement_minimization,
            'directional_profit_efficiency': directional_profit_efficiency,
            'risk_adjusted_performance': risk_adjusted_performance,
            'composite_score': composite_score
        }
    
    def _train_final_directional_model(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     optimization_result: Dict[str, Any]) -> Any:
        """Train final model with best parameters."""
        best_solution = optimization_result['best_solution']
        params = best_solution.params
        
        # Create model with best parameters
        if params['model_type'] == 'ElasticNetCV':
            model = ElasticNetCV(
                l1_ratio=[params['l1_ratio']],
                alphas=[params['alpha']],
                cv=5
            )
        elif params['model_type'] == 'Ridge':
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=params['alpha'])
        else:  # Lasso
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=params['alpha'])
        
        # Train model
        model.fit(X, y)
        
        return model
    
    def _evaluate_directional_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate final directional performance."""
        return self._evaluate_directional_metrics(model, X, y)


# Convenience functions
def optimize_tactician_for_direction(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
    hmm_model_outputs: Optional[np.ndarray] = None,
    analyst_ensemble_outputs: Optional[np.ndarray] = None,
    max_trials: int = 100
) -> DirectionalOptimizationResult:
    """
    Optimize Tactician model for directional objectives.
    
    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels
        config: Training configuration
        feature_names: Feature names
        hmm_states: HMM states
        analyst_signals: Analyst signals
        analyst_model_outputs: Analyst model outputs
        hmm_regime_features: HMM regime features
        all_analyst_models_outputs: All analyst model outputs
        hmm_model_outputs: HMM model outputs
        analyst_ensemble_outputs: Analyst ensemble outputs
        max_trials: Maximum optimization trials
        
    Returns:
        DirectionalOptimizationResult with optimized model
    """
    optimizer = DirectionalTacticianOptimizer(config)
    
    return optimizer.optimize_tactician_directionally(
        X=X, y=y, regime_labels=regime_labels,
        feature_names=feature_names, hmm_states=hmm_states,
        analyst_signals=analyst_signals, analyst_model_outputs=analyst_model_outputs,
        hmm_regime_features=hmm_regime_features, 
        all_analyst_models_outputs=all_analyst_models_outputs,
        hmm_model_outputs=hmm_model_outputs, 
        analyst_ensemble_outputs=analyst_ensemble_outputs,
        max_trials=max_trials
    )


if __name__ == '__main__':
    # Test the directional optimization
    print("🎯 Testing Directional Tactician Optimization")
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)  # Directional targets
    
    # Create regime labels
    regime_labels = np.random.choice([0, 1, 2], n_samples)
    
    # Test directional optimization
    print("\n📊 Testing directional optimization...")
    result = optimize_tactician_for_direction(
        X=X, y=y, regime_labels=regime_labels, max_trials=50
    )
    
    print(f"✅ Directional optimization completed:")
    print(f"   Directional accuracy: {result.directional_accuracy:.4f}")
    print(f"   Adverse movement min: {result.adverse_movement_minimization:.4f}")
    print(f"   Profit efficiency: {result.directional_profit_efficiency:.4f}")
    print(f"   Risk-adjusted perf: {result.risk_adjusted_performance:.4f}")
    print(f"   Composite score: {result.composite_score:.4f}")
    print(f"   Optimization time: {result.optimization_time:.2f}s")
    
    print('✅ Directional Tactician Optimization test completed!')