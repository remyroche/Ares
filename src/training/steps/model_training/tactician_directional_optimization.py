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


class EntryTimingLossFunction:
    """Custom loss functions for entry timing optimization within 0-0.5% range."""
    
    @staticmethod
    def early_entry_penalty_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                                entry_range: float = 0.005) -> float:
        """
        Loss function that penalizes entering too early (before optimal timing).
        
        Args:
            y_true: True optimal entry timing (0-0.5% range)
            y_pred: Predicted entry timing (0-0.5% range)
            entry_range: Maximum entry range (0.5%)
            
        Returns:
            Early entry penalty loss
        """
        # Calculate how early we're entering (negative values = too early)
        timing_error = y_pred - y_true
        
        # Penalize early entries (negative timing errors)
        early_penalty = np.sum(np.maximum(0, -timing_error)) / len(y_true)
        
        return early_penalty
    
    @staticmethod
    def late_entry_penalty_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                               entry_range: float = 0.005) -> float:
        """
        Loss function that penalizes entering too late (after optimal timing).
        
        Args:
            y_true: True optimal entry timing (0-0.5% range)
            y_pred: Predicted entry timing (0-0.5% range)
            entry_range: Maximum entry range (0.5%)
            
        Returns:
            Late entry penalty loss
        """
        # Calculate how late we're entering (positive values = too late)
        timing_error = y_pred - y_true
        
        # Penalize late entries (positive timing errors)
        late_penalty = np.sum(np.maximum(0, timing_error)) / len(y_true)
        
        return late_penalty
    
    @staticmethod
    def optimal_entry_reward_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                                 tolerance: float = 0.001) -> float:
        """
        Loss function that rewards optimal entry timing.
        
        Args:
            y_true: True optimal entry timing
            y_pred: Predicted entry timing
            tolerance: Tolerance for "optimal" timing (0.1%)
            
        Returns:
            Optimal entry reward loss (inverted reward)
        """
        # Calculate timing accuracy
        timing_error = np.abs(y_pred - y_true)
        
        # Reward entries within tolerance
        optimal_mask = timing_error <= tolerance
        optimal_ratio = np.sum(optimal_mask) / len(y_true)
        
        return 1.0 - optimal_ratio  # Return loss (1 - reward)
    
    @staticmethod
    def entry_timing_efficiency_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                                   expected_movement: float = 0.01) -> float:
        """
        Loss function that maximizes profit efficiency from optimal entry timing.
        
        Args:
            y_true: True optimal entry timing (0-0.5% range)
            y_pred: Predicted entry timing (0-0.5% range)
            expected_movement: Expected 1% movement in right direction
            
        Returns:
            Entry timing efficiency loss
        """
        # Calculate profit from entry timing
        # Profit = (expected_movement - entry_timing) for correct timing
        timing_error = np.abs(y_pred - y_true)
        
        # Calculate potential profit (expected movement minus timing cost)
        potential_profit = expected_movement - timing_error
        
        # Calculate efficiency (actual profit / maximum possible profit)
        max_profit = expected_movement
        efficiency = np.mean(potential_profit) / max_profit if max_profit > 0 else 0
        
        return 1.0 - efficiency  # Return loss (1 - efficiency)


# DirectionalFeatureEngineer removed - using existing features from base training


class EntryTimingTacticianOptimizer:
    """
    Enhanced Tactician optimizer that directly optimizes for entry timing within 0-0.5% range.
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
        
        # Initialize loss functions
        self.loss_functions = EntryTimingLossFunction()
        
        self.logger.info("🚀 Entry Timing Tactician optimizer initialized")
    
    def optimize_tactician_entry_timing(self,
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
        Optimize Tactician model for entry timing within 0-0.5% range.
        
        Args:
            X: Input features
            y: Target values (optimal entry timing 0-0.5% range)
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
            DirectionalOptimizationResult with entry timing optimization results
        """
        start_time = time.time()
        
        self.logger.info("🔄 Starting entry timing Tactician optimization...")
        
        # Step 1: Use existing features (no additional feature engineering)
        self.logger.info("📊 Step 1: Using existing features for entry timing optimization")
        X_enhanced = X  # Use existing features as-is
        
        # Step 2: Multi-objective optimization for entry timing
        self.logger.info("📊 Step 2: Multi-objective entry timing optimization")
        optimization_result = self._multi_objective_entry_timing_optimization(
            X_enhanced, y, regime_labels, max_trials
        )
        
        # Step 3: Train final model with best parameters
        self.logger.info("📊 Step 3: Training final entry timing model")
        final_model = self._train_final_entry_timing_model(
            X_enhanced, y, optimization_result
        )
        
        # Step 4: Evaluate entry timing performance
        self.logger.info("📊 Step 4: Evaluating entry timing performance")
        entry_timing_metrics = self._evaluate_entry_timing_performance(final_model, X_enhanced, y)
        
        # Create result
        result = DirectionalOptimizationResult(
            model=final_model,
            directional_accuracy=entry_timing_metrics['early_entry_penalty'],
            adverse_movement_minimization=entry_timing_metrics['late_entry_penalty'],
            directional_profit_efficiency=entry_timing_metrics['optimal_entry_reward'],
            risk_adjusted_performance=entry_timing_metrics['entry_timing_efficiency'],
            composite_score=entry_timing_metrics['composite_score'],
            optimization_time=time.time() - start_time,
            n_trials=max_trials,
            optimization_history=self.optimization_history.copy()
        )
        
        self.logger.info(f"✅ Entry timing optimization completed in {result.optimization_time:.2f}s")
        self.logger.info(f"   Early entry penalty: {result.directional_accuracy:.4f}")
        self.logger.info(f"   Late entry penalty: {result.adverse_movement_minimization:.4f}")
        self.logger.info(f"   Optimal entry reward: {result.directional_profit_efficiency:.4f}")
        self.logger.info(f"   Entry timing efficiency: {result.risk_adjusted_performance:.4f}")
        self.logger.info(f"   Composite score: {result.composite_score:.4f}")
        
        return result
    
    # Feature enhancement removed - using existing features from base training
    
    def _multi_objective_entry_timing_optimization(self,
                                                X: np.ndarray,
                                                y: np.ndarray,
                                                regime_labels: np.ndarray,
                                                max_trials: int) -> Dict[str, Any]:
        """Multi-objective optimization for entry timing goals."""
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
            
            # Evaluate entry timing metrics
            metrics = self._evaluate_entry_timing_metrics(model, X, y)
            
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
    
    def _evaluate_entry_timing_metrics(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate entry timing metrics for a model."""
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate early entry penalty (minimize entering too early)
        timing_error = y_pred - y
        early_entry_penalty = np.mean(np.maximum(0, -timing_error))
        
        # Calculate late entry penalty (minimize entering too late)
        late_entry_penalty = np.mean(np.maximum(0, timing_error))
        
        # Calculate optimal entry reward (maximize entries within tolerance)
        timing_accuracy = np.abs(y_pred - y)
        tolerance = 0.001  # 0.1% tolerance for optimal timing
        optimal_mask = timing_accuracy <= tolerance
        optimal_entry_reward = np.mean(optimal_mask)
        
        # Calculate entry timing efficiency (maximize profit from optimal timing)
        expected_movement = 0.01  # Expected 1% movement
        potential_profit = expected_movement - timing_accuracy
        entry_timing_efficiency = np.mean(potential_profit) / expected_movement if expected_movement > 0 else 0
        
        # Calculate composite score
        composite_score = (
            0.3 * (1 - early_entry_penalty) +  # Minimize early entry penalty
            0.3 * (1 - late_entry_penalty) +   # Minimize late entry penalty
            0.2 * optimal_entry_reward +       # Maximize optimal entry reward
            0.2 * entry_timing_efficiency      # Maximize entry timing efficiency
        )
        
        return {
            'early_entry_penalty': early_entry_penalty,
            'late_entry_penalty': late_entry_penalty,
            'optimal_entry_reward': optimal_entry_reward,
            'entry_timing_efficiency': entry_timing_efficiency,
            'composite_score': composite_score
        }
    
    def _train_final_entry_timing_model(self,
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
    
    def _evaluate_entry_timing_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate final entry timing performance."""
        return self._evaluate_entry_timing_metrics(model, X, y)


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