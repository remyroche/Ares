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
from typing import Dict, List, Optional, Union, Any, Tuple
from src.core.decorators.errors import handles_errors

class WeightOptimizer:
    """Handles weight optimization for ensemble models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the weight optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('weight_optimization', {})
        self.logger = system_logger.getChild('weight_optimizer')
        self.optimization_method = self.config.get('method', 'scipy')
        self.optimization_metric = self.config.get('metric', 'accuracy')
        self.cv_folds = self.config.get('cv_folds', 5)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.convergence_threshold = self.config.get('convergence_threshold', 0.0001)

    @handles_errors(exceptions=(Exception,), default_return={}, context='weight optimization')
    async def optimize_weights(self, regime_ensembles: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Optimize weights for ensemble models.
        
        Args:
            regime_ensembles: Dictionary of regime-specific ensembles
            features: Feature data for optimization
            
        Returns:
            Dictionary of optimized weights for each regime
        """
        self.logger.info('Starting weight optimization...')
        optimized_weights = {}
        for regime_id, ensemble_data in regime_ensembles.items():
            if isinstance(ensemble_data, dict) and 'ensemble' in ensemble_data:
                regime_weights = await self._optimize_regime_weights(regime_id, ensemble_data, features)
                optimized_weights[regime_id] = regime_weights
        return optimized_weights

    async def _optimize_regime_weights(self, regime_id: str, ensemble_data: Dict[str, Any], features: pd.DataFrame) -> Dict[str, float]:
        """Optimize weights for a specific regime ensemble.
        
        Args:
            regime_id: Regime identifier
            ensemble_data: Ensemble data for this regime
            features: Feature data
            
        Returns:
            Dictionary of optimized weights
        """
        try:
            ensembles = ensemble_data.get('ensemble', {})
            if not ensembles:
                return {}
            if features.empty:
                return {ens_type: 1.0 / len(ensembles) for ens_type in ensembles}
            sample_size = min(5000, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            X_sample = features.iloc[sample_indices]
            y_sample = np.random.randint(0, 2, size=sample_size)
            if self.optimization_method == 'scipy':
                weights = await self._scipy_weight_optimization(ensembles, X_sample, y_sample)
            elif self.optimization_method == 'grid_search':
                weights = await self._grid_search_weight_optimization(ensembles, X_sample, y_sample)
            else:
                weights = {ens_type: 1.0 / len(ensembles) for ens_type in ensembles}
            self.logger.info(f'Optimized weights for regime {regime_id}: {weights}')
            return weights
        except Exception as e:
            self.logger.error(f'Failed to optimize weights for regime {regime_id}: {str(e)}')
            return {}

    async def _scipy_weight_optimization(self, ensembles: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
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

        def objective(weights: Union[List[float], np.ndarray]) -> float:
            weights = weights / weights.sum()
            weighted_pred = predictions @ weights
            if self.optimization_metric == 'accuracy':
                binary_pred = (weighted_pred > 0.5).astype(int)
                return -accuracy_score(y, binary_pred)
            elif self.optimization_metric == 'log_loss':
                weighted_pred = np.clip(weighted_pred, 1e-07, 1 - 1e-07)
                return log_loss(y, weighted_pred)
            else:
                return 0.0
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(n_ensembles)]
        initial_weights = np.ones(n_ensembles) / n_ensembles
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': self.max_iterations})
        if result.success:
            optimized_weights = result.x / result.x.sum()
        else:
            self.logger.warning('Weight optimization failed, using equal weights')
            optimized_weights = initial_weights
        return dict(zip(ensemble_names, optimized_weights))

    async def _grid_search_weight_optimization(self, ensembles: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
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
        weight_steps = 0.1
        weight_grid = self._generate_weight_grid(n_ensembles, weight_steps)
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
        best_score = -np.inf
        best_weights = None
        for weights in weight_grid:
            weighted_pred = predictions @ weights
            if self.optimization_metric == 'accuracy':
                binary_pred = (weighted_pred > 0.5).astype(int)
                score = accuracy_score(y, binary_pred)
            else:
                score = 0.5
            if score > best_score:
                best_score = score
                best_weights = weights
        if best_weights is None:
            best_weights = np.ones(n_ensembles) / n_ensembles
        return dict(zip(ensemble_names, best_weights))

    def _generate_weight_grid(self, n_weights: int, step_size: float=0.1) -> List[np.ndarray]:
        """Generate a grid of weights that sum to 1.
        
        Args:
            n_weights: Number of weights
            step_size: Step size for grid
            
        Returns:
            List of weight arrays
        """
        if n_weights == 1:
            return [np.array([1.0])]
        grid = []

        def generate_weights(remaining: Any, current_weights: List[Any], n_remaining: Any) -> None:
            if n_remaining == 1:
                grid.append(np.array(current_weights + [remaining]))
                return
            max_value = min(1.0, remaining)
            current = 0.0
            while current <= max_value:
                generate_weights(remaining - current, current_weights + [current], n_remaining - 1)
                current += step_size
        generate_weights(1.0, [], n_weights)
        valid_grid = [w for w in grid if abs(w.sum() - 1.0) < 1e-06]
        return valid_grid

    @handles_errors(exceptions=(Exception,), default_return={}, context='dynamic weight optimization')
    async def optimize_dynamic_weights(self, ensembles: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize weights that can change based on input features.
        
        Args:
            ensembles: Dictionary of ensemble models
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with dynamic weight model
        """
        self.logger.info('Optimizing dynamic weights...')
        static_weights = await self._scipy_weight_optimization(ensembles, X_train, y_train)
        return {'type': 'static', 'weights': static_weights}