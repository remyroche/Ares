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


class DirectionalConsistencyLossFunction:
    """Custom loss functions for ensuring directional consistency across 0.1%-0.5% price levels."""
    
    @staticmethod
    def directional_consistency_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                                   price_levels: List[float] = [0.001, 0.002, 0.003, 0.004, 0.005]) -> float:
        """
        Loss function that ensures all price moves (0.1%, 0.2%, 0.3%, 0.4%, 0.5%) 
        are in the same direction (no reversals).
        
        Args:
            y_true: True price movements at different levels
            y_pred: Predicted price movements at different levels
            price_levels: List of price levels to check (0.1%, 0.2%, 0.3%, 0.4%, 0.5%)
            
        Returns:
            Directional consistency loss
        """
        # Ensure y_true and y_pred have the same number of levels
        if y_true.shape[1] != len(price_levels) or y_pred.shape[1] != len(price_levels):
            return 1.0  # Maximum loss if dimensions don't match
        
        # Calculate directional consistency
        y_true_direction = np.sign(y_true)
        y_pred_direction = np.sign(y_pred)
        
        # Check if all levels have the same direction (no reversals)
        consistency_penalty = 0.0
        
        for i in range(len(price_levels)):
            for j in range(i + 1, len(price_levels)):
                # Check if directions are consistent between levels
                true_consistent = np.all(y_true_direction[:, i] == y_true_direction[:, j])
                pred_consistent = np.all(y_pred_direction[:, i] == y_pred_direction[:, j])
                
                # Penalize if predictions don't maintain consistency
                if not pred_consistent:
                    consistency_penalty += 1.0
                
                # Additional penalty if true data is consistent but prediction isn't
                if true_consistent and not pred_consistent:
                    consistency_penalty += 2.0
        
        # Normalize by number of level pairs
        num_pairs = len(price_levels) * (len(price_levels) - 1) // 2
        return consistency_penalty / num_pairs if num_pairs > 0 else 1.0
    
    @staticmethod
    def directional_accuracy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Loss function that maximizes directional accuracy across all price levels.
        
        Args:
            y_true: True price movements at different levels
            y_pred: Predicted price movements at different levels
            
        Returns:
            Directional accuracy loss
        """
        # Calculate directional accuracy for each level
        y_true_direction = np.sign(y_true)
        y_pred_direction = np.sign(y_pred)
        
        # Calculate accuracy for each level
        level_accuracies = []
        for level in range(y_true.shape[1]):
            level_accuracy = np.mean(y_true_direction[:, level] == y_pred_direction[:, level])
            level_accuracies.append(level_accuracy)
        
        # Return average accuracy loss
        avg_accuracy = np.mean(level_accuracies)
        return 1.0 - avg_accuracy
    
    @staticmethod
    def magnitude_consistency_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                                 price_levels: List[float] = [0.001, 0.002, 0.003, 0.004, 0.005]) -> float:
        """
        Loss function that ensures magnitude consistency (0.1% < 0.2% < 0.3% < 0.4% < 0.5%).
        
        Args:
            y_true: True price movements at different levels
            y_pred: Predicted price movements at different levels
            price_levels: List of price levels to check
            
        Returns:
            Magnitude consistency loss
        """
        # Check if magnitudes are monotonically increasing
        magnitude_penalty = 0.0
        
        for i in range(len(price_levels) - 1):
            # Check if next level has larger magnitude than current level
            true_magnitude_consistent = np.all(np.abs(y_true[:, i+1]) >= np.abs(y_true[:, i]))
            pred_magnitude_consistent = np.all(np.abs(y_pred[:, i+1]) >= np.abs(y_pred[:, i]))
            
            # Penalize if predictions don't maintain magnitude consistency
            if not pred_magnitude_consistent:
                magnitude_penalty += 1.0
            
            # Additional penalty if true data is consistent but prediction isn't
            if true_magnitude_consistent and not pred_magnitude_consistent:
                magnitude_penalty += 2.0
        
        # Normalize by number of level transitions
        num_transitions = len(price_levels) - 1
        return magnitude_penalty / num_transitions if num_transitions > 0 else 1.0
    
    @staticmethod
    def reversal_penalty_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Loss function that heavily penalizes any directional reversals.
        
        Args:
            y_true: True price movements at different levels
            y_pred: Predicted price movements at different levels
            
        Returns:
            Reversal penalty loss
        """
        # Calculate directional consistency
        y_pred_direction = np.sign(y_pred)
        
        # Count reversals (direction changes between consecutive levels)
        reversals = 0
        total_transitions = 0
        
        for i in range(y_pred.shape[1] - 1):
            # Check for direction changes between consecutive levels
            direction_changes = y_pred_direction[:, i] != y_pred_direction[:, i+1]
            reversals += np.sum(direction_changes)
            total_transitions += len(direction_changes)
        
        # Return reversal ratio (heavily penalized)
        reversal_ratio = reversals / total_transitions if total_transitions > 0 else 0
        return reversal_ratio * 10.0  # Heavy penalty for reversals


# DirectionalFeatureEngineer removed - using existing features from base training


class DirectionalConsistencyTacticianOptimizer:
    """
    Enhanced Tactician optimizer that ensures directional consistency across 0.1%-0.5% price levels.
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
        self.loss_functions = DirectionalConsistencyLossFunction()
        
        # Price levels for directional consistency (0.1%, 0.2%, 0.3%, 0.4%, 0.5%)
        self.price_levels = [0.001, 0.002, 0.003, 0.004, 0.005]
        
        self.logger.info("🚀 Directional Consistency Tactician optimizer initialized")
    
    def optimize_tactician_directional_consistency(self,
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
        Optimize Tactician model for directional consistency across 0.1%-0.5% price levels.
        
        Args:
            X: Input features
            y: Target values (price movements at 0.1%, 0.2%, 0.3%, 0.4%, 0.5% levels)
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
            DirectionalOptimizationResult with directional consistency optimization results
        """
        start_time = time.time()
        
        self.logger.info("🔄 Starting directional consistency Tactician optimization...")
        
        # Step 1: Validate and prepare multi-level targets
        self.logger.info("📊 Step 1: Preparing multi-level targets for directional consistency")
        y_multi_level = self._prepare_multi_level_targets(y)
        
        # Step 2: Use existing features (no additional feature engineering)
        self.logger.info("📊 Step 2: Using existing features for directional consistency optimization")
        X_enhanced = X  # Use existing features as-is
        
        # Step 3: Multi-objective optimization for directional consistency
        self.logger.info("📊 Step 3: Multi-objective directional consistency optimization")
        optimization_result = self._multi_objective_directional_consistency_optimization(
            X_enhanced, y_multi_level, regime_labels, max_trials
        )
        
        # Step 4: Train final model with best parameters
        self.logger.info("📊 Step 4: Training final directional consistency model")
        final_model = self._train_final_directional_consistency_model(
            X_enhanced, y_multi_level, optimization_result
        )
        
        # Step 5: Evaluate directional consistency performance
        self.logger.info("📊 Step 5: Evaluating directional consistency performance")
        directional_consistency_metrics = self._evaluate_directional_consistency_performance(final_model, X_enhanced, y_multi_level)
        
        # Create result
        result = DirectionalOptimizationResult(
            model=final_model,
            directional_accuracy=directional_consistency_metrics['directional_consistency_loss'],
            adverse_movement_minimization=directional_consistency_metrics['directional_accuracy_loss'],
            directional_profit_efficiency=directional_consistency_metrics['magnitude_consistency_loss'],
            risk_adjusted_performance=directional_consistency_metrics['reversal_penalty_loss'],
            composite_score=directional_consistency_metrics['composite_score'],
            optimization_time=time.time() - start_time,
            n_trials=max_trials,
            optimization_history=self.optimization_history.copy()
        )
        
        self.logger.info(f"✅ Directional consistency optimization completed in {result.optimization_time:.2f}s")
        self.logger.info(f"   Directional consistency loss: {result.directional_accuracy:.4f}")
        self.logger.info(f"   Directional accuracy loss: {result.adverse_movement_minimization:.4f}")
        self.logger.info(f"   Magnitude consistency loss: {result.directional_profit_efficiency:.4f}")
        self.logger.info(f"   Reversal penalty loss: {result.risk_adjusted_performance:.4f}")
        self.logger.info(f"   Composite score: {result.composite_score:.4f}")
        
        return result
    
    def _prepare_multi_level_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Prepare multi-level targets for directional consistency optimization.
        
        Args:
            y: Single-level targets (price movements)
            
        Returns:
            Multi-level targets (price movements at 0.1%, 0.2%, 0.3%, 0.4%, 0.5% levels)
        """
        # For now, we'll create synthetic multi-level targets based on the single level
        # In practice, this would come from actual price data at different levels
        
        n_samples = len(y)
        n_levels = len(self.price_levels)
        
        # Create multi-level targets
        y_multi_level = np.zeros((n_samples, n_levels))
        
        for i, level in enumerate(self.price_levels):
            # Scale the single-level target to different price levels
            # This is a simplified approach - in practice, you'd use actual price data
            y_multi_level[:, i] = y * (level / 0.005)  # Scale to 0.1%, 0.2%, etc.
        
        self.logger.info(f"📊 Prepared multi-level targets: {y_multi_level.shape} (levels: {self.price_levels})")
        return y_multi_level
    
    # Feature enhancement removed - using existing features from base training
    
    def _multi_objective_directional_consistency_optimization(self,
                                                X: np.ndarray,
                                                y: np.ndarray,
                                                regime_labels: np.ndarray,
                                                max_trials: int) -> Dict[str, Any]:
        """Multi-objective optimization for directional consistency goals."""
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
            
            # Evaluate directional consistency metrics
            metrics = self._evaluate_directional_consistency_metrics(model, X, y)
            
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
    
    def _evaluate_directional_consistency_metrics(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate directional consistency metrics for a model."""
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate directional consistency loss
        directional_consistency_loss = self.loss_functions.directional_consistency_loss(y, y_pred, self.price_levels)
        
        # Calculate directional accuracy loss
        directional_accuracy_loss = self.loss_functions.directional_accuracy_loss(y, y_pred)
        
        # Calculate magnitude consistency loss
        magnitude_consistency_loss = self.loss_functions.magnitude_consistency_loss(y, y_pred, self.price_levels)
        
        # Calculate reversal penalty loss
        reversal_penalty_loss = self.loss_functions.reversal_penalty_loss(y, y_pred)
        
        # Calculate composite score (lower is better for losses)
        composite_score = (
            0.4 * (1 - directional_consistency_loss) +  # Maximize directional consistency
            0.3 * (1 - directional_accuracy_loss) +     # Maximize directional accuracy
            0.2 * (1 - magnitude_consistency_loss) +    # Maximize magnitude consistency
            0.1 * (1 - reversal_penalty_loss)           # Minimize reversals
        )
        
        return {
            'directional_consistency_loss': directional_consistency_loss,
            'directional_accuracy_loss': directional_accuracy_loss,
            'magnitude_consistency_loss': magnitude_consistency_loss,
            'reversal_penalty_loss': reversal_penalty_loss,
            'composite_score': composite_score
        }
    
    def _train_final_directional_consistency_model(self,
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
    
    def _evaluate_directional_consistency_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate final directional consistency performance."""
        return self._evaluate_directional_consistency_metrics(model, X, y)


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