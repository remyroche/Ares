"""
Multi-Fidelity Objective Functions

This module provides multi-fidelity objective function implementations for different
use cases in the optimization system. Multi-fidelity optimization allows for
efficient resource allocation by evaluating configurations at different resource levels.

Phase 6: Multi-Fidelity Objective Functions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)


@dataclass
class MultiFidelityConfig:
    """Configuration for multi-fidelity optimization."""
    
    # Resource settings
    resource_name: str = "iteration"
    min_resource: int = 1
    max_resource: int = 10
    
    # Multi-fidelity parameters
    resource_scaling_factor: float = 1.0
    early_stopping_threshold: float = 0.01
    min_improvement_threshold: float = 0.001
    
    # Performance tracking
    track_performance: bool = True
    performance_history_size: int = 100


class MultiFidelityObjective(ABC):
    """Abstract base class for multi-fidelity objectives."""
    
    def __init__(self, config: MultiFidelityConfig):
        self.config = config
        self.performance_history = []
        self.resource_efficiency_history = []
    
    @abstractmethod
    def evaluate(self, params: Dict[str, Any], resource: int) -> float:
        """Evaluate objective at given resource level."""
        pass
    
    def get_resource_efficiency(self) -> float:
        """Calculate resource efficiency from history."""
        if len(self.resource_efficiency_history) < 2:
            return 1.0
        
        recent_efficiency = np.mean(self.resource_efficiency_history[-10:])
        return recent_efficiency


class ModelTrainingMultiFidelityObjective(MultiFidelityObjective):
    """Multi-fidelity objective for model training optimization."""
    
    def __init__(self, config: MultiFidelityConfig, 
                 model_factory: Callable,
                 X: np.ndarray, 
                 y: np.ndarray,
                 cv_folds: int = 5):
        super().__init__(config)
        self.model_factory = model_factory
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
    
    def evaluate(self, params: Dict[str, Any], resource: int) -> float:
        """Evaluate model training objective at given resource level."""
        try:
            # Create model with parameters
            model = self.model_factory()
            if hasattr(model, 'set_params'):
                model.set_params(**params)
            
            # Adjust model complexity based on resource level
            if hasattr(model, 'n_estimators') and resource < self.config.max_resource:
                # For tree-based models, reduce n_estimators for lower resource
                model.set_params(n_estimators=max(10, int(resource * 10)))
            elif hasattr(model, 'max_iter') and resource < self.config.max_resource:
                # For iterative models, reduce max_iter for lower resource
                model.set_params(max_iter=max(50, int(resource * 50)))
            elif hasattr(model, 'epochs') and resource < self.config.max_resource:
                # For neural networks, reduce epochs for lower resource
                model.set_params(epochs=max(1, int(resource * 2)))
            
            # Perform cross-validation with limited resource
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                model, self.X, self.y,
                cv=min(self.cv_folds, resource),  # Limit CV folds based on resource
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Calculate resource efficiency
            score = np.mean(scores)
            if self.config.track_performance:
                self.performance_history.append(score)
                if len(self.performance_history) > self.config.performance_history_size:
                    self.performance_history.pop(0)
            
            return score
            
        except Exception as e:
            logger.debug(f"Model training evaluation failed: {e}")
            return -np.inf


class ClusteringMultiFidelityObjective(MultiFidelityObjective):
    """Multi-fidelity objective for clustering optimization."""
    
    def __init__(self, config: MultiFidelityConfig,
                 clustering_func: Callable,
                 features: np.ndarray,
                 validation_func: Callable):
        super().__init__(config)
        self.clustering_func = clustering_func
        self.features = features
        self.validation_func = validation_func
    
    def evaluate(self, params: Dict[str, Any], resource: int) -> float:
        """Evaluate clustering objective at given resource level."""
        try:
            # Adjust clustering parameters based on resource level
            if resource < self.config.max_resource:
                # Reduce complexity for lower resource levels
                if 'n_clusters' in params:
                    params = params.copy()
                    params['n_clusters'] = max(2, int(params['n_clusters'] * (resource / self.config.max_resource)))
                
                if 'min_samples' in params:
                    params = params.copy()
                    params['min_samples'] = max(1, int(params['min_samples'] * (resource / self.config.max_resource)))
            
            # Perform clustering with limited iterations
            if hasattr(self.clustering_func, 'max_iter'):
                clustering_func = lambda data: self.clustering_func(data, max_iter=resource)
            else:
                clustering_func = self.clustering_func
            
            # Run clustering
            labels = clustering_func(self.features)
            
            # Validate clustering quality
            if self.validation_func:
                score = self.validation_func(labels, self.features)
            else:
                # Default validation using silhouette score
                from sklearn.metrics import silhouette_score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(self.features, labels)
                else:
                    score = -1.0
            
            # Track performance
            if self.config.track_performance:
                self.performance_history.append(score)
                if len(self.performance_history) > self.config.performance_history_size:
                    self.performance_history.pop(0)
            
            return score
            
        except Exception as e:
            logger.debug(f"Clustering evaluation failed: {e}")
            return -np.inf


class BacktestingMultiFidelityObjective(MultiFidelityObjective):
    """Multi-fidelity objective for backtesting optimization."""
    
    def __init__(self, config: MultiFidelityConfig,
                 backtesting_func: Callable,
                 market_data: pd.DataFrame,
                 evaluation_metrics: List[str] = None):
        super().__init__(config)
        self.backtesting_func = backtesting_func
        self.market_data = market_data
        self.evaluation_metrics = evaluation_metrics or ['sharpe_ratio', 'max_drawdown']
    
    def evaluate(self, params: Dict[str, Any], resource: int) -> float:
        """Evaluate backtesting objective at given resource level."""
        try:
            # Adjust backtesting parameters based on resource level
            if resource < self.config.max_resource:
                params = params.copy()
                
                # Reduce data size for lower resource levels
                if 'lookback_period' in params:
                    params['lookback_period'] = max(10, int(params['lookback_period'] * (resource / self.config.max_resource)))
                
                if 'rebalance_frequency' in params:
                    params['rebalance_frequency'] = max(1, int(params['rebalance_frequency'] * (resource / self.config.max_resource)))
            
            # Limit data size based on resource
            data_size = int(len(self.market_data) * (resource / self.config.max_resource))
            limited_data = self.market_data.tail(data_size)
            
            # Run backtesting with limited data
            results = self.backtesting_func(limited_data, **params)
            
            # Calculate composite score
            score = self._calculate_composite_score(results)
            
            # Track performance
            if self.config.track_performance:
                self.performance_history.append(score)
                if len(self.performance_history) > self.config.performance_history_size:
                    self.performance_history.pop(0)
            
            return score
            
        except Exception as e:
            logger.debug(f"Backtesting evaluation failed: {e}")
            return -np.inf
    
    def _calculate_composite_score(self, results: Dict[str, Any]) -> float:
        """Calculate composite score from backtesting results."""
        try:
            scores = []
            
            for metric in self.evaluation_metrics:
                if metric in results:
                    value = results[metric]
                    if metric == 'max_drawdown':
                        # Lower is better for drawdown
                        scores.append(-value)
                    else:
                        # Higher is better for other metrics
                        scores.append(value)
            
            if scores:
                return np.mean(scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Composite score calculation failed: {e}")
            return 0.0


class EnsembleMultiFidelityObjective(MultiFidelityObjective):
    """Multi-fidelity objective for ensemble training optimization."""
    
    def __init__(self, config: MultiFidelityConfig,
                 ensemble_factory: Callable,
                 X: np.ndarray,
                 y: np.ndarray,
                 base_models: List[str]):
        super().__init__(config)
        self.ensemble_factory = ensemble_factory
        self.X = X
        self.y = y
        self.base_models = base_models
    
    def evaluate(self, params: Dict[str, Any], resource: int) -> float:
        """Evaluate ensemble objective at given resource level."""
        try:
            # Create ensemble with parameters
            ensemble = self.ensemble_factory()
            if hasattr(ensemble, 'set_params'):
                ensemble.set_params(**params)
            
            # Adjust ensemble complexity based on resource level
            if resource < self.config.max_resource:
                # Reduce number of base models for lower resource
                limited_models = self.base_models[:max(1, int(len(self.base_models) * (resource / self.config.max_resource)))]
                if hasattr(ensemble, 'base_models'):
                    ensemble.set_params(base_models=limited_models)
                
                # Reduce training iterations
                if hasattr(ensemble, 'n_estimators'):
                    ensemble.set_params(n_estimators=max(10, int(resource * 5)))
            
            # Perform cross-validation with limited resource
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(
                ensemble, self.X, self.y,
                cv=min(3, resource),  # Limit CV folds based on resource
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Calculate resource efficiency
            score = np.mean(scores)
            
            # Track performance
            if self.config.track_performance:
                self.performance_history.append(score)
                if len(self.performance_history) > self.config.performance_history_size:
                    self.performance_history.pop(0)
            
            return score
            
        except Exception as e:
            logger.debug(f"Ensemble evaluation failed: {e}")
            return -np.inf


class MultiFidelityObjectiveFactory:
    """Factory for creating multi-fidelity objectives."""
    
    @staticmethod
    def create_objective(objective_type: str, 
                        config: MultiFidelityConfig,
                        **kwargs) -> MultiFidelityObjective:
        """Create multi-fidelity objective based on type."""
        
        objective_map = {
            'model_training': ModelTrainingMultiFidelityObjective,
            'clustering': ClusteringMultiFidelityObjective,
            'backtesting': BacktestingMultiFidelityObjective,
            'ensemble': EnsembleMultiFidelityObjective,
        }
        
        objective_class = objective_map.get(objective_type)
        if not objective_class:
            raise ValueError(f"Unknown objective type: {objective_type}")
        
        return objective_class(config, **kwargs)
    
    @staticmethod
    def create_model_training_objective(model_factory: Callable,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      config: MultiFidelityConfig = None,
                                      **kwargs) -> ModelTrainingMultiFidelityObjective:
        """Create model training multi-fidelity objective."""
        if config is None:
            config = MultiFidelityConfig(resource_name="epoch", min_resource=1, max_resource=10)
        
        return ModelTrainingMultiFidelityObjective(
            config, model_factory, X, y, **kwargs
        )
    
    @staticmethod
    def create_clustering_objective(clustering_func: Callable,
                                  features: np.ndarray,
                                  validation_func: Callable = None,
                                  config: MultiFidelityConfig = None,
                                  **kwargs) -> ClusteringMultiFidelityObjective:
        """Create clustering multi-fidelity objective."""
        if config is None:
            config = MultiFidelityConfig(resource_name="iteration", min_resource=1, max_resource=5)
        
        return ClusteringMultiFidelityObjective(
            config, clustering_func, features, validation_func, **kwargs
        )
    
    @staticmethod
    def create_backtesting_objective(backtesting_func: Callable,
                                   market_data: pd.DataFrame,
                                   config: MultiFidelityConfig = None,
                                   **kwargs) -> BacktestingMultiFidelityObjective:
        """Create backtesting multi-fidelity objective."""
        if config is None:
            config = MultiFidelityConfig(resource_name="iteration", min_resource=1, max_resource=5)
        
        return BacktestingMultiFidelityObjective(
            config, backtesting_func, market_data, **kwargs
        )
    
    @staticmethod
    def create_ensemble_objective(ensemble_factory: Callable,
                                X: np.ndarray,
                                y: np.ndarray,
                                base_models: List[str],
                                config: MultiFidelityConfig = None,
                                **kwargs) -> EnsembleMultiFidelityObjective:
        """Create ensemble multi-fidelity objective."""
        if config is None:
            config = MultiFidelityConfig(resource_name="epoch", min_resource=1, max_resource=8)
        
        return EnsembleMultiFidelityObjective(
            config, ensemble_factory, X, y, base_models, **kwargs
        )


# Convenience functions for easy creation
def create_model_training_objective(model_factory: Callable,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  **kwargs) -> ModelTrainingMultiFidelityObjective:
    """Create model training multi-fidelity objective."""
    return MultiFidelityObjectiveFactory.create_model_training_objective(
        model_factory, X, y, **kwargs
    )


def create_clustering_objective(clustering_func: Callable,
                              features: np.ndarray,
                              validation_func: Callable = None,
                              **kwargs) -> ClusteringMultiFidelityObjective:
    """Create clustering multi-fidelity objective."""
    return MultiFidelityObjectiveFactory.create_clustering_objective(
        clustering_func, features, validation_func, **kwargs
    )


def create_backtesting_objective(backtesting_func: Callable,
                               market_data: pd.DataFrame,
                               **kwargs) -> BacktestingMultiFidelityObjective:
    """Create backtesting multi-fidelity objective."""
    return MultiFidelityObjectiveFactory.create_backtesting_objective(
        backtesting_func, market_data, **kwargs
    )


def create_ensemble_objective(ensemble_factory: Callable,
                            X: np.ndarray,
                            y: np.ndarray,
                            base_models: List[str],
                            **kwargs) -> EnsembleMultiFidelityObjective:
    """Create ensemble multi-fidelity objective."""
    return MultiFidelityObjectiveFactory.create_ensemble_objective(
        ensemble_factory, X, y, base_models, **kwargs
    )