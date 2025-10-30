"""
Hierarchical Hyperparameter Optimization for Autoencoder + TCN.

This module uses the hierarchical parameter optimizer from src/utils/ml_common/optimization/
to efficiently optimize the Autoencoder + TCN architecture for analyst/tactician models.

Parameter Groups (optimized in order):
1. Autoencoder structure (latent_dim, hidden_dim) - MOST CRITICAL
2. TCN structure (num_filters, num_layers, kernel_size, dilation_base)
3. Learning rates (autoencoder_lr, tcn_lr)
4. Regularization (dropout rates)
5. Training parameters (batch_size, epochs)

Uses 2 rounds of optimization by default to capture parameter interactions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import json
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    HierarchicalOptimizationResult
)
from src.models.causal_dilated_tcn import CausalTCNConfig, CausalDilatedTCNModel

logger = get_logger(__name__)


def create_autoencoder_tcn_param_groups(
    input_dim: int,
    role: str = "analyst"
) -> List[ParameterGroup]:
    """
    Create parameter groups for hierarchical optimization of Autoencoder + TCN.
    
    Args:
        input_dim: Number of input features (determines autoencoder input size)
        role: "analyst" or "tactician" (affects default ranges)
    
    Returns:
        List of ParameterGroup objects in priority order
    """
    logger.info(f"🔧 Creating parameter groups for {role} with {input_dim} input features")
    
    # Adjust ranges based on role
    if role.lower() == "analyst":
        latent_choices = [12, 16, 20, 24, 32]
        tcn_filters = [48, 64, 80, 96]
        tcn_layers = (3, 5)
        dropout_range = (0.15, 0.30)
    else:  # tactician
        latent_choices = [16, 20, 24, 32, 40]
        tcn_filters = [64, 80, 96, 128]
        tcn_layers = (3, 6)
        dropout_range = (0.10, 0.25)
    
    param_groups = [
        # ============================================
        # GROUP 1: AUTOENCODER STRUCTURE (Priority 1)
        # ============================================
        ParameterGroup(
            name="autoencoder_structure",
            params={
                "latent_dim": {
                    "type": "categorical",
                    "choices": latent_choices
                },
                "ae_hidden_dim": {
                    "type": "categorical",
                    "choices": [32, 64, 96, 128]
                }
            },
            priority=1,
            description=f"Autoencoder compression: {input_dim} features → latent_dim"
        ),
        
        # ============================================
        # GROUP 2: TCN STRUCTURE (Priority 2)
        # ============================================
        ParameterGroup(
            name="tcn_structure",
            params={
                "num_filters": {
                    "type": "categorical",
                    "choices": tcn_filters
                },
                "num_layers": {
                    "type": "int",
                    "low": tcn_layers[0],
                    "high": tcn_layers[1]
                },
                "kernel_size": {
                    "type": "categorical",
                    "choices": [3, 5, 7]
                },
                "dilation_base": {
                    "type": "categorical",
                    "choices": [2, 3, 4]
                }
            },
            priority=2,
            depends_on=["autoencoder_structure"],
            description="TCN architecture - processes compressed features"
        ),
        
        # ============================================
        # GROUP 3: LEARNING RATES (Priority 3)
        # ============================================
        ParameterGroup(
            name="learning_rates",
            params={
                "ae_learning_rate": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.01,
                    "log": True
                },
                "tcn_learning_rate": {
                    "type": "float",
                    "low": 0.0001,
                    "high": 0.01,
                    "log": True
                }
            },
            priority=3,
            depends_on=["autoencoder_structure", "tcn_structure"],
            description="Learning rates for autoencoder pre-training and TCN training"
        ),
        
        # ============================================
        # GROUP 4: REGULARIZATION (Priority 4)
        # ============================================
        ParameterGroup(
            name="regularization",
            params={
                "ae_dropout": {
                    "type": "float",
                    "low": 0.1,
                    "high": 0.5
                },
                "tcn_dropout": {
                    "type": "float",
                    "low": dropout_range[0],
                    "high": dropout_range[1]
                }
            },
            priority=4,
            depends_on=["autoencoder_structure", "tcn_structure", "learning_rates"],
            description="Dropout regularization for both models"
        ),
        
        # ============================================
        # GROUP 5: TRAINING PARAMETERS (Priority 5)
        # ============================================
        ParameterGroup(
            name="training_params",
            params={
                "batch_size": {
                    "type": "categorical",
                    "choices": [16, 32, 64]
                },
                "ae_epochs": {
                    "type": "int",
                    "low": 30,
                    "high": 80
                },
                "tcn_epochs": {
                    "type": "int",
                    "low": 50,
                    "high": 120
                },
                "early_stopping_patience": {
                    "type": "int",
                    "low": 8,
                    "high": 15
                }
            },
            priority=5,
            depends_on=["learning_rates"],
            description="Training configuration parameters"
        )
    ]
    
    logger.info(f"✅ Created {len(param_groups)} parameter groups")
    for group in param_groups:
        logger.info(f"   {group.priority}. {group.name}: {len(group.params)} params")
    
    return param_groups


def create_objective_function(
    metric: str = "accuracy",
    use_validation_split: bool = False
):
    """
    Create objective function for hyperparameter optimization.
    
    Args:
        metric: Metric to optimize ("accuracy", "f1", "auc")
        use_validation_split: If True, split train into train/val
    
    Returns:
        Objective function compatible with hierarchical optimizer
    """
    def objective(params: Dict[str, Any], X: np.ndarray, y: np.ndarray, 
                 X_val: Optional[np.ndarray] = None, 
                 y_val: Optional[np.ndarray] = None) -> float:
        """
        Objective function that trains Autoencoder + TCN with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        
        Returns:
            Score (higher is better)
        """
        try:
            # Extract autoencoder-specific params
            ae_params = {
                'input_dim': X.shape[1],
                'latent_dim': params['latent_dim'],
                'hidden_dim': params.get('ae_hidden_dim', 64)
            }
            
            # Build TCN config
            tcn_config = CausalTCNConfig(
                # TCN structure
                num_filters=params['num_filters'],
                num_layers=params['num_layers'],
                kernel_size=params['kernel_size'],
                dilation_base=params['dilation_base'],
                
                # Training params
                learning_rate=params['tcn_learning_rate'],
                batch_size=params['batch_size'],
                epochs=params['tcn_epochs'],
                early_stopping_patience=params['early_stopping_patience'],
                dropout=params['tcn_dropout'],
                
                # Autoencoder integration
                use_autoencoder=True,
                latent_dim=params['latent_dim'],
                train_autoencoder_if_missing=True,
                autoencoder_epochs=params['ae_epochs'],
                
                # Note: We can't easily pass ae_learning_rate and ae_dropout
                # to the current implementation, but we can enhance it later
            )
            
            # Create and train model
            model = CausalDilatedTCNModel(config=tcn_config)
            
            # Split data if needed
            if use_validation_split and X_val is None:
                from sklearn.model_selection import train_test_split
                X_train, X_val_split, y_train, y_val_split = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train, y_train = X, y
                X_val_split, y_val_split = X_val, y_val
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            if X_val_split is not None and y_val_split is not None:
                preds = model.predict(X_val_split)
                y_eval = y_val_split
            else:
                preds = model.predict(X_train)
                y_eval = y_train
            
            # Calculate score based on metric
            if metric == "accuracy":
                from sklearn.metrics import accuracy_score
                binary_preds = (preds > 0.5).astype(int)
                score = accuracy_score(y_eval, binary_preds)
            elif metric == "f1":
                from sklearn.metrics import f1_score
                binary_preds = (preds > 0.5).astype(int)
                score = f1_score(y_eval, binary_preds)
            elif metric == "auc":
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_eval, preds)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            logger.info(f"   Trial result: {metric}={score:.4f}, latent_dim={params['latent_dim']}, "
                       f"filters={params['num_filters']}, layers={params['num_layers']}")
            
            return score
            
        except Exception as e:
            logger.error(f"   ❌ Trial failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # Return worst score on failure
    
    return objective


class AutoencoderTCNHPO:
    """
    Hierarchical hyperparameter optimizer for Autoencoder + TCN models.
    
    Uses the hierarchical parameter optimizer to efficiently search the
    hyperparameter space in stages:
    1. Autoencoder structure
    2. TCN structure  
    3. Learning rates
    4. Regularization
    5. Training parameters
    """
    
    def __init__(
        self,
        role: str = "analyst",
        metric: str = "accuracy",
        n_rounds: int = 2,
        stages: Optional[List[OptimizationStage]] = None,
        enable_final_refinement: bool = True,
        final_refinement_trials: int = 50,
        save_results: bool = True,
        results_dir: str = "artifacts/hpo/autoencoder_tcn",
        verbose: bool = True
    ):
        """
        Initialize Autoencoder + TCN hyperparameter optimizer.
        
        Args:
            role: "analyst" or "tactician" (affects parameter ranges)
            metric: Optimization metric ("accuracy", "f1", "auc")
            n_rounds: Number of optimization rounds (default: 2)
                     Round 1: Full exploration
                     Round 2: Refinement around best
            stages: Optimization stages (default: COARSE_GRID, FINE_GRID, TPE)
            enable_final_refinement: Whether to do final joint optimization
            final_refinement_trials: Number of trials for final refinement
            save_results: Whether to save optimization results
            results_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.role = role.lower()
        self.metric = metric
        self.n_rounds = n_rounds
        self.stages = stages or [
            OptimizationStage.COARSE_GRID,
            OptimizationStage.FINE_GRID,
            OptimizationStage.TPE
        ]
        self.enable_final_refinement = enable_final_refinement
        self.final_refinement_trials = final_refinement_trials
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.verbose = verbose
        
        # Create results directory
        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 Initialized Autoencoder+TCN HPO for {role}")
        logger.info(f"   Metric: {metric}")
        logger.info(f"   Rounds: {n_rounds}")
        logger.info(f"   Stages: {[s.value for s in self.stages]}")
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        use_validation_split: bool = False
    ) -> HierarchicalOptimizationResult:
        """
        Run hierarchical hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            use_validation_split: If True and X_val is None, split train data
        
        Returns:
            HierarchicalOptimizationResult with best parameters and history
        """
        logger.info("="*80)
        logger.info("🚀 STARTING AUTOENCODER + TCN HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Input features: {X_train.shape[1]}")
        logger.info(f"   Validation samples: {len(X_val) if X_val is not None else 'Using train split'}")
        logger.info(f"   Target classes: {len(np.unique(y_train))}")
        logger.info("")
        
        # Create parameter groups
        param_groups = create_autoencoder_tcn_param_groups(
            input_dim=X_train.shape[1],
            role=self.role
        )
        
        # Create objective function
        objective_func = create_objective_function(
            metric=self.metric,
            use_validation_split=use_validation_split
        )
        
        # Create hierarchical optimizer
        optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_func,
            stages=self.stages,
            n_rounds=self.n_rounds,
            enable_final_refinement=self.enable_final_refinement,
            final_refinement_trials=self.final_refinement_trials,
            direction='maximize',  # Higher score is better
            cv_folds=None,  # Use holdout validation
            verbose=self.verbose
        )
        
        # Run optimization
        logger.info("🏃 Running hierarchical optimization...")
        logger.info("")
        
        result = optimizer.optimize(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        # Log results
        logger.info("")
        logger.info("="*80)
        logger.info("✅ OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"   Best {self.metric}: {result.best_score:.4f}")
        logger.info(f"   Total trials: {result.total_trials}")
        logger.info(f"   Total time: {result.total_time:.1f}s")
        logger.info("")
        logger.info("📊 Best Parameters:")
        for group_name, group_result in result.group_results.items():
            logger.info(f"   {group_name}:")
            for param, value in group_result.best_params.items():
                logger.info(f"      {param}: {value}")
        
        # Save results
        if self.save_results:
            self._save_results(result)
        
        return result
    
    def _save_results(self, result: HierarchicalOptimizationResult):
        """Save optimization results to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hpo_results_{self.role}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            # Convert result to dictionary
            result_dict = result.to_dict()
            
            # Add metadata
            result_dict['metadata'] = {
                'role': self.role,
                'metric': self.metric,
                'n_rounds': self.n_rounds,
                'stages': [s.value for s in self.stages],
                'timestamp': timestamp
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"💾 Results saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def optimize_analyst_autoencoder_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    metric: str = "accuracy",
    n_rounds: int = 2,
    save_results: bool = True
) -> Tuple[Dict[str, Any], float]:
    """
    Convenience function to optimize Analyst Autoencoder + TCN.
    
    Args:
        X_train: Training features
        y_train: Training targets  
        X_val: Validation features
        y_val: Validation targets
        metric: Optimization metric
        n_rounds: Number of optimization rounds
        save_results: Whether to save results
    
    Returns:
        (best_params, best_score)
    """
    hpo = AutoencoderTCNHPO(
        role="analyst",
        metric=metric,
        n_rounds=n_rounds,
        save_results=save_results
    )
    
    result = hpo.optimize(X_train, y_train, X_val, y_val)
    return result.best_params, result.best_score


def optimize_tactician_autoencoder_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    metric: str = "accuracy",
    n_rounds: int = 2,
    save_results: bool = True
) -> Tuple[Dict[str, Any], float]:
    """
    Convenience function to optimize Tactician Autoencoder + TCN.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        metric: Optimization metric
        n_rounds: Number of optimization rounds
        save_results: Whether to save results
    
    Returns:
        (best_params, best_score)
    """
    hpo = AutoencoderTCNHPO(
        role="tactician",
        metric=metric,
        n_rounds=n_rounds,
        save_results=save_results
    )
    
    result = hpo.optimize(X_train, y_train, X_val, y_val)
    return result.best_params, result.best_score


if __name__ == "__main__":
    # Example usage
    logger.info("Example: Optimize Autoencoder + TCN for Analyst")
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(1000, 120)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 120)
    y_val = np.random.randint(0, 2, 200)
    
    # Run optimization
    best_params, best_score = optimize_analyst_autoencoder_tcn(
        X_train, y_train, X_val, y_val,
        metric="accuracy",
        n_rounds=2
    )
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")

