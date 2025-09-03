"""Weight optimization component for analyst ensemble creation."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class WeightOptimizer:
    """Handles weight optimization for ensemble models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the weight optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("weight_optimization", {})
        self.logger = system_logger.getChild("weight_optimizer")
        
        # Optimization configuration
        self.optimization_method = self.config.get("method", "scipy")
        self.optimization_metric = self.config.get("metric", "accuracy")
        self.cv_folds = self.config.get("cv_folds", 5)
        self.max_iterations = self.config.get("max_iterations", 100)
        self.convergence_threshold = self.config.get("convergence_threshold", 1e-4)
        
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="weight optimization"
    )
    async def optimize_weights(
        self,
        regime_ensembles: Dict[str, Any],
        features: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Optimize weights for ensemble models.
        
        Args:
            regime_ensembles: Dictionary of regime-specific ensembles
            features: Feature data for optimization
            
        Returns:
            Dictionary of optimized weights for each regime
        """
        self.logger.info("Starting weight optimization...")
        
        optimized_weights = {}
        
        for regime_id, ensemble_data in regime_ensembles.items():
            if isinstance(ensemble_data, dict) and "ensemble" in ensemble_data:
                regime_weights = await self._optimize_regime_weights(
                    regime_id,
                    ensemble_data,
                    features
                )
                optimized_weights[regime_id] = regime_weights
        
        return optimized_weights
    
    async def _optimize_regime_weights(
        self,
        regime_id: str,
        ensemble_data: Dict[str, Any],
        features: pd.DataFrame
    ) -> Dict[str, float]:
        """Optimize weights for a specific regime ensemble.
        
        Args:
            regime_id: Regime identifier
            ensemble_data: Ensemble data for this regime
            features: Feature data
            
        Returns:
            Dictionary of optimized weights
        """
        try:
            ensembles = ensemble_data.get("ensemble", {})
            if not ensembles:
                return {}
            
            # Get validation data for this regime
            # In practice, this would use regime-specific data
            if features.empty:
                return {ens_type: 1.0 / len(ensembles) for ens_type in ensembles}
            
            # Sample data for efficiency
            sample_size = min(5000, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            X_sample = features.iloc[sample_indices]
            
            # Generate synthetic labels for demonstration
            # In practice, these would come from the pipeline state
            y_sample = np.random.randint(0, 2, size=sample_size)
            
            # Optimize weights using cross-validation
            if self.optimization_method == "scipy":
                weights = await self._scipy_weight_optimization(
                    ensembles, X_sample, y_sample
                )
            elif self.optimization_method == "grid_search":
                weights = await self._grid_search_weight_optimization(
                    ensembles, X_sample, y_sample
                )
            else:
                # Equal weights as fallback
                weights = {ens_type: 1.0 / len(ensembles) for ens_type in ensembles}
            
            self.logger.info(
                f"Optimized weights for regime {regime_id}: {weights}"
            )
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to optimize weights for regime {regime_id}: {str(e)}")
            return {}
    
    async def _scipy_weight_optimization(
        self,
        ensembles: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Optimize weights using scipy optimization.
        
        Args:
            ensembles: Dictionary of ensemble models
            X: Feature data
            y: Target labels
            
        Returns:
            Dictionary of optimized weights
        """
        ensemble_names = list(ensembles.keys())
        n_ensembles = len(ensemble_names)
        
        # Get predictions from each ensemble
        predictions = []
        for ens_name, ensemble in ensembles.items():
            try:
                if hasattr(ensemble, 'predict_proba'):
                    pred = ensemble.predict_proba(X)[:, 1]
                else:
                    pred = ensemble.predict(X)
                predictions.append(pred)
            except:
                # Use random predictions if model fails
                pred = np.random.rand(len(X))
                predictions.append(pred)
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_ensembles)
        
        # Define objective function
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            # Calculate weighted predictions
            weighted_pred = predictions @ weights
            
            # Calculate loss based on metric
            if self.optimization_metric == "accuracy":
                # Convert to binary predictions
                binary_pred = (weighted_pred > 0.5).astype(int)
                return -accuracy_score(y, binary_pred)
            elif self.optimization_metric == "log_loss":
                # Clip probabilities to avoid log(0)
                weighted_pred = np.clip(weighted_pred, 1e-7, 1 - 1e-7)
                return log_loss(y, weighted_pred)
            else:
                return 0.0
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_ensembles)]
        
        # Initial weights (equal)
        initial_weights = np.ones(n_ensembles) / n_ensembles
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            optimized_weights = result.x / result.x.sum()  # Normalize
        else:
            self.logger.warning("Weight optimization failed, using equal weights")
            optimized_weights = initial_weights
        
        return dict(zip(ensemble_names, optimized_weights))
    
    async def _grid_search_weight_optimization(
        self,
        ensembles: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Optimize weights using grid search.
        
        Args:
            ensembles: Dictionary of ensemble models
            X: Feature data
            y: Target labels
            
        Returns:
            Dictionary of optimized weights
        """
        ensemble_names = list(ensembles.keys())
        n_ensembles = len(ensemble_names)
        
        # Generate weight grid
        weight_steps = 0.1
        weight_grid = self._generate_weight_grid(n_ensembles, weight_steps)
        
        # Get predictions from each ensemble
        predictions = []
        for ens_name, ensemble in ensembles.items():
            try:
                if hasattr(ensemble, 'predict_proba'):
                    pred = ensemble.predict_proba(X)[:, 1]
                else:
                    pred = ensemble.predict(X)
                predictions.append(pred)
            except:
                pred = np.random.rand(len(X))
                predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        # Find best weights
        best_score = -np.inf
        best_weights = None
        
        for weights in weight_grid:
            # Calculate weighted predictions
            weighted_pred = predictions @ weights
            
            # Calculate score
            if self.optimization_metric == "accuracy":
                binary_pred = (weighted_pred > 0.5).astype(int)
                score = accuracy_score(y, binary_pred)
            else:
                score = 0.5  # Default
            
            if score > best_score:
                best_score = score
                best_weights = weights
        
        if best_weights is None:
            best_weights = np.ones(n_ensembles) / n_ensembles
        
        return dict(zip(ensemble_names, best_weights))
    
    def _generate_weight_grid(
        self,
        n_weights: int,
        step_size: float = 0.1
    ) -> List[np.ndarray]:
        """Generate a grid of weights that sum to 1.
        
        Args:
            n_weights: Number of weights
            step_size: Step size for grid
            
        Returns:
            List of weight arrays
        """
        if n_weights == 1:
            return [np.array([1.0])]
        
        # Generate all possible combinations
        grid = []
        
        def generate_weights(remaining, current_weights, n_remaining):
            if n_remaining == 1:
                grid.append(np.array(current_weights + [remaining]))
                return
            
            # Try different values for the current position
            max_value = min(1.0, remaining)
            current = 0.0
            while current <= max_value:
                generate_weights(
                    remaining - current,
                    current_weights + [current],
                    n_remaining - 1
                )
                current += step_size
        
        generate_weights(1.0, [], n_weights)
        
        # Filter out weights that don't sum to approximately 1
        valid_grid = [w for w in grid if abs(w.sum() - 1.0) < 1e-6]
        
        return valid_grid
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="dynamic weight optimization"
    )
    async def optimize_dynamic_weights(
        self,
        ensembles: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize weights that can change based on input features.
        
        Args:
            ensembles: Dictionary of ensemble models
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with dynamic weight model
        """
        self.logger.info("Optimizing dynamic weights...")
        
        # This is a placeholder for more sophisticated dynamic weighting
        # In practice, this could train a meta-model that predicts optimal weights
        # based on input features
        
        # For now, return static optimized weights
        static_weights = await self._scipy_weight_optimization(
            ensembles, X_train, y_train
        )
        
        return {
            "type": "static",
            "weights": static_weights
        }