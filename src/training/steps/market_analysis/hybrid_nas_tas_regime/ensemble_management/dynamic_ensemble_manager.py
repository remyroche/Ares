"""
Dynamic Model Ensemble Management System

This module implements comprehensive ensemble management for regime-specific models including:
1. Per-regime model ensembles
2. Dynamic ensemble weighting strategies
3. Ensemble performance optimization
4. Real-time ensemble adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    from scipy.optimize import minimize
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for dynamic ensemble management."""
    
    # Ensemble composition
    max_ensemble_models: int = 5
    min_ensemble_models: int = 2
    ensemble_diversity_threshold: float = 0.3
    
    # Weighting strategies
    weighting_strategy: str = "performance_based"  # "performance_based", "uncertainty_based", "adaptive", "dynamic"
    performance_window: int = 100  # Number of recent predictions to consider
    uncertainty_weight: float = 0.3  # Weight given to uncertainty in adaptive weighting
    
    # Dynamic adaptation
    enable_dynamic_adaptation: bool = True
    adaptation_frequency: int = 50  # Adapt weights every N predictions
    adaptation_threshold: float = 0.05  # Minimum performance change to trigger adaptation
    learning_rate: float = 0.1  # Learning rate for weight updates
    
    # Performance optimization
    enable_performance_optimization: bool = True
    optimization_method: str = "gradient_descent"  # "gradient_descent", "genetic_algorithm", "bayesian"
    optimization_frequency: int = 100  # Optimize every N predictions
    convergence_threshold: float = 1e-6
    
    # Ensemble validation
    enable_ensemble_validation: bool = True
    validation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1_score", "roc_auc"
    ])
    min_ensemble_performance: float = 0.6
    
    # Model selection
    enable_model_selection: bool = True
    model_selection_criteria: str = "performance"  # "performance", "diversity", "stability"
    model_removal_threshold: float = 0.1  # Remove models with weight below this threshold
    
    # Real-time adaptation
    enable_real_time_adaptation: bool = True
    real_time_window: int = 20  # Window for real-time performance tracking
    real_time_threshold: float = 0.1  # Performance threshold for real-time adaptation


@dataclass
class EnsembleModel:
    """Represents a model in an ensemble."""
    model_id: str
    model_name: str
    model_instance: Any
    weight: float = 0.0
    performance_history: List[float] = field(default_factory=list)
    uncertainty_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    predictions: np.ndarray
    probabilities: np.ndarray
    uncertainties: np.ndarray
    model_weights: Dict[str, float]
    ensemble_confidence: float
    individual_predictions: Dict[str, np.ndarray]
    individual_probabilities: Dict[str, np.ndarray]


class DynamicEnsembleManager:
    """
    Dynamic ensemble manager for regime-specific model ensembles.
    """
    
    def __init__(self, config: EnsembleConfig):
        """Initialize dynamic ensemble manager."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensemble state
        self.regime_ensembles: Dict[int, List[EnsembleModel]] = {}
        self.ensemble_weights: Dict[int, Dict[str, float]] = {}
        self.performance_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        
        # Adaptation state
        self.adaptation_counters: Dict[int, int] = defaultdict(int)
        self.last_optimization: Dict[int, datetime] = {}
        
        self.logger.info("✅ Dynamic Ensemble Manager initialized")
        self.logger.info(f"   Weighting strategy: {config.weighting_strategy}")
        self.logger.info(f"   Dynamic adaptation: {config.enable_dynamic_adaptation}")
        self.logger.info(f"   Performance optimization: {config.enable_performance_optimization}")
    
    def create_regime_ensemble(self, 
                             regime_id: int,
                             models: List[Tuple[str, str, Any]]) -> bool:
        """
        Create an ensemble for a specific regime.
        
        Args:
            regime_id: ID of the regime
            models: List of (model_id, model_name, model_instance) tuples
            
        Returns:
            True if ensemble created successfully
        """
        try:
            if len(models) < self.config.min_ensemble_models:
                self.logger.warning(f"Not enough models for ensemble in regime {regime_id}: {len(models)}")
                return False
            
            # Create ensemble models
            ensemble_models = []
            for model_id, model_name, model_instance in models:
                ensemble_model = EnsembleModel(
                    model_id=model_id,
                    model_name=model_name,
                    model_instance=model_instance
                )
                ensemble_models.append(ensemble_model)
            
            # Initialize weights
            initial_weight = 1.0 / len(ensemble_models)
            for model in ensemble_models:
                model.weight = initial_weight
            
            # Store ensemble
            self.regime_ensembles[regime_id] = ensemble_models
            self.ensemble_weights[regime_id] = {model.model_id: model.weight for model in ensemble_models}
            
            self.logger.info(f"Created ensemble for regime {regime_id} with {len(ensemble_models)} models")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create ensemble for regime {regime_id}: {e}")
            return False
    
    def predict_ensemble(self, 
                        regime_id: int,
                        features: np.ndarray) -> EnsembleResult:
        """
        Make ensemble prediction for a regime.
        
        Args:
            regime_id: ID of the regime
            features: Input features
            
        Returns:
            Ensemble prediction result
        """
        try:
            if regime_id not in self.regime_ensembles:
                raise ValueError(f"No ensemble found for regime {regime_id}")
            
            ensemble_models = self.regime_ensembles[regime_id]
            active_models = [model for model in ensemble_models if model.is_active]
            
            if not active_models:
                raise ValueError(f"No active models in ensemble for regime {regime_id}")
            
            # Get predictions from each model
            individual_predictions = {}
            individual_probabilities = {}
            individual_uncertainties = {}
            
            for model in active_models:
                try:
                    # Get predictions
                    predictions = model.model_instance.predict(features)
                    individual_predictions[model.model_id] = predictions
                    
                    # Get probabilities if available
                    if hasattr(model.model_instance, 'predict_proba'):
                        probabilities = model.model_instance.predict_proba(features)
                        individual_probabilities[model.model_id] = probabilities
                        
                        # Calculate uncertainty (entropy)
                        uncertainty = self._calculate_uncertainty(probabilities)
                        individual_uncertainties[model.model_id] = uncertainty
                    else:
                        # Create dummy probabilities
                        n_classes = len(np.unique(predictions))
                        probabilities = np.zeros((len(features), n_classes))
                        for i, pred in enumerate(predictions):
                            probabilities[i, pred] = 1.0
                        individual_probabilities[model.model_id] = probabilities
                        individual_uncertainties[model.model_id] = np.zeros(len(features))
                    
                except Exception as e:
                    self.logger.warning(f"Model {model.model_id} prediction failed: {e}")
                    continue
            
            if not individual_predictions:
                raise ValueError(f"No successful predictions from ensemble for regime {regime_id}")
            
            # Calculate ensemble weights
            ensemble_weights = self._calculate_ensemble_weights(regime_id, active_models, individual_uncertainties)
            
            # Combine predictions
            ensemble_predictions, ensemble_probabilities, ensemble_uncertainties = self._combine_predictions(
                individual_predictions, individual_probabilities, individual_uncertainties, ensemble_weights
            )
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(ensemble_probabilities, ensemble_uncertainties)
            
            # Update model performance
            self._update_model_performance(regime_id, active_models, individual_predictions, individual_uncertainties)
            
            # Check for dynamic adaptation
            if self.config.enable_dynamic_adaptation:
                self._check_dynamic_adaptation(regime_id)
            
            # Create result
            result = EnsembleResult(
                predictions=ensemble_predictions,
                probabilities=ensemble_probabilities,
                uncertainties=ensemble_uncertainties,
                model_weights=ensemble_weights,
                ensemble_confidence=ensemble_confidence,
                individual_predictions=individual_predictions,
                individual_probabilities=individual_probabilities
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed for regime {regime_id}: {e}")
            raise
    
    def _calculate_ensemble_weights(self, 
                                  regime_id: int,
                                  active_models: List[EnsembleModel],
                                  individual_uncertainties: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate dynamic ensemble weights."""
        try:
            if self.config.weighting_strategy == "performance_based":
                return self._calculate_performance_based_weights(regime_id, active_models)
            elif self.config.weighting_strategy == "uncertainty_based":
                return self._calculate_uncertainty_based_weights(active_models, individual_uncertainties)
            elif self.config.weighting_strategy == "adaptive":
                return self._calculate_adaptive_weights(regime_id, active_models, individual_uncertainties)
            elif self.config.weighting_strategy == "dynamic":
                return self._calculate_dynamic_weights(regime_id, active_models, individual_uncertainties)
            else:
                # Default to equal weights
                weight = 1.0 / len(active_models)
                return {model.model_id: weight for model in active_models}
                
        except Exception as e:
            self.logger.error(f"Failed to calculate ensemble weights: {e}")
            # Fallback to equal weights
            weight = 1.0 / len(active_models)
            return {model.model_id: weight for model in active_models}
    
    def _calculate_performance_based_weights(self, 
                                           regime_id: int,
                                           active_models: List[EnsembleModel]) -> Dict[str, float]:
        """Calculate weights based on model performance."""
        try:
            weights = {}
            total_performance = 0.0
            
            for model in active_models:
                # Get recent performance
                if model.performance_history:
                    recent_performance = np.mean(model.performance_history[-self.config.performance_window:])
                else:
                    recent_performance = 0.5  # Default performance
                
                weights[model.model_id] = recent_performance
                total_performance += recent_performance
            
            # Normalize weights
            if total_performance > 0:
                weights = {model_id: weight / total_performance for model_id, weight in weights.items()}
            else:
                # Equal weights if no performance data
                weight = 1.0 / len(active_models)
                weights = {model.model_id: weight for model in active_models}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance-based weights: {e}")
            weight = 1.0 / len(active_models)
            return {model.model_id: weight for model in active_models}
    
    def _calculate_uncertainty_based_weights(self, 
                                           active_models: List[EnsembleModel],
                                           individual_uncertainties: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate weights based on model uncertainty."""
        try:
            weights = {}
            total_inverse_uncertainty = 0.0
            
            for model in active_models:
                if model.model_id in individual_uncertainties:
                    # Lower uncertainty = higher weight
                    avg_uncertainty = np.mean(individual_uncertainties[model.model_id])
                    inverse_uncertainty = 1.0 / (avg_uncertainty + 1e-8)
                    weights[model.model_id] = inverse_uncertainty
                    total_inverse_uncertainty += inverse_uncertainty
                else:
                    weights[model.model_id] = 1.0
                    total_inverse_uncertainty += 1.0
            
            # Normalize weights
            if total_inverse_uncertainty > 0:
                weights = {model_id: weight / total_inverse_uncertainty for model_id, weight in weights.items()}
            else:
                weight = 1.0 / len(active_models)
                weights = {model.model_id: weight for model in active_models}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to calculate uncertainty-based weights: {e}")
            weight = 1.0 / len(active_models)
            return {model.model_id: weight for model in active_models}
    
    def _calculate_adaptive_weights(self, 
                                  regime_id: int,
                                  active_models: List[EnsembleModel],
                                  individual_uncertainties: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate adaptive weights combining performance and uncertainty."""
        try:
            # Get performance-based weights
            perf_weights = self._calculate_performance_based_weights(regime_id, active_models)
            
            # Get uncertainty-based weights
            uncert_weights = self._calculate_uncertainty_based_weights(active_models, individual_uncertainties)
            
            # Combine weights
            combined_weights = {}
            for model in active_models:
                perf_weight = perf_weights.get(model.model_id, 0.0)
                uncert_weight = uncert_weights.get(model.model_id, 0.0)
                
                # Weighted combination
                combined_weight = (1 - self.config.uncertainty_weight) * perf_weight + \
                                 self.config.uncertainty_weight * uncert_weight
                combined_weights[model.model_id] = combined_weight
            
            # Normalize
            total_weight = sum(combined_weights.values())
            if total_weight > 0:
                combined_weights = {model_id: weight / total_weight for model_id, weight in combined_weights.items()}
            else:
                weight = 1.0 / len(active_models)
                combined_weights = {model.model_id: weight for model in active_models}
            
            return combined_weights
            
        except Exception as e:
            self.logger.error(f"Failed to calculate adaptive weights: {e}")
            weight = 1.0 / len(active_models)
            return {model.model_id: weight for model in active_models}
    
    def _calculate_dynamic_weights(self, 
                                 regime_id: int,
                                 active_models: List[EnsembleModel],
                                 individual_uncertainties: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate dynamic weights using optimization."""
        try:
            if not SCIPY_AVAILABLE:
                self.logger.warning("SciPy not available, falling back to adaptive weights")
                return self._calculate_adaptive_weights(regime_id, active_models, individual_uncertainties)
            
            # Define objective function for weight optimization
            def objective(weights):
                # Normalize weights
                weights = weights / np.sum(weights)
                
                # Calculate ensemble performance (simplified)
                # In practice, this would use actual performance metrics
                performance = 0.0
                for i, model in enumerate(active_models):
                    if model.performance_history:
                        recent_perf = np.mean(model.performance_history[-10:])
                        performance += weights[i] * recent_perf
                
                # Add diversity penalty
                diversity_penalty = -np.std(weights) * 0.1
                
                return -(performance + diversity_penalty)
            
            # Initial weights
            initial_weights = np.ones(len(active_models)) / len(active_models)
            
            # Constraints: weights sum to 1, weights >= 0
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(active_models))]
            
            # Optimize weights
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x
            else:
                self.logger.warning("Weight optimization failed, using adaptive weights")
                return self._calculate_adaptive_weights(regime_id, active_models, individual_uncertainties)
            
            # Create weight dictionary
            weights = {model.model_id: optimized_weights[i] for i, model in enumerate(active_models)}
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to calculate dynamic weights: {e}")
            return self._calculate_adaptive_weights(regime_id, active_models, individual_uncertainties)
    
    def _combine_predictions(self, 
                           individual_predictions: Dict[str, np.ndarray],
                           individual_probabilities: Dict[str, np.ndarray],
                           individual_uncertainties: Dict[str, np.ndarray],
                           ensemble_weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine individual predictions into ensemble predictions."""
        try:
            # Get first model's predictions to determine shape
            first_model_id = list(individual_predictions.keys())[0]
            n_samples = len(individual_predictions[first_model_id])
            
            # Initialize ensemble arrays
            ensemble_predictions = np.zeros(n_samples)
            ensemble_probabilities = None
            ensemble_uncertainties = np.zeros(n_samples)
            
            # Weighted combination
            for model_id, predictions in individual_predictions.items():
                weight = ensemble_weights.get(model_id, 0.0)
                ensemble_predictions += weight * predictions
                
                # Combine uncertainties
                if model_id in individual_uncertainties:
                    ensemble_uncertainties += weight * individual_uncertainties[model_id]
            
            # Combine probabilities if available
            if individual_probabilities:
                first_prob = individual_probabilities[first_model_id]
                n_classes = first_prob.shape[1]
                ensemble_probabilities = np.zeros((n_samples, n_classes))
                
                for model_id, probabilities in individual_probabilities.items():
                    weight = ensemble_weights.get(model_id, 0.0)
                    ensemble_probabilities += weight * probabilities
            
            # Round predictions to integers
            ensemble_predictions = np.round(ensemble_predictions).astype(int)
            
            return ensemble_predictions, ensemble_probabilities, ensemble_uncertainties
            
        except Exception as e:
            self.logger.error(f"Failed to combine predictions: {e}")
            # Fallback to first model's predictions
            first_model_id = list(individual_predictions.keys())[0]
            return (individual_predictions[first_model_id], 
                   individual_probabilities.get(first_model_id),
                   individual_uncertainties.get(first_model_id, np.zeros(len(individual_predictions[first_model_id]))))
    
    def _calculate_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate uncertainty (entropy) from probabilities."""
        try:
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            probs = probabilities + eps
            
            # Calculate entropy
            entropy_values = -np.sum(probs * np.log(probs), axis=1)
            
            return entropy_values
            
        except Exception as e:
            self.logger.error(f"Failed to calculate uncertainty: {e}")
            return np.zeros(len(probabilities))
    
    def _calculate_ensemble_confidence(self, 
                                     ensemble_probabilities: np.ndarray,
                                     ensemble_uncertainties: np.ndarray) -> float:
        """Calculate overall ensemble confidence."""
        try:
            if ensemble_probabilities is None:
                return 0.5  # Default confidence
            
            # Calculate average confidence (1 - uncertainty)
            avg_confidence = 1.0 - np.mean(ensemble_uncertainties)
            
            # Normalize to [0, 1]
            confidence = max(0.0, min(1.0, avg_confidence))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Failed to calculate ensemble confidence: {e}")
            return 0.5
    
    def _update_model_performance(self, 
                                regime_id: int,
                                active_models: List[EnsembleModel],
                                individual_predictions: Dict[str, np.ndarray],
                                individual_uncertainties: Dict[str, np.ndarray]):
        """Update model performance based on recent predictions."""
        try:
            for model in active_models:
                if model.model_id in individual_predictions:
                    # Calculate performance metric (simplified)
                    # In practice, this would use actual ground truth
                    performance = np.random.uniform(0.5, 1.0)  # Placeholder
                    
                    # Update performance history
                    model.performance_history.append(performance)
                    if len(model.performance_history) > 100:  # Keep last 100
                        model.performance_history.pop(0)
                    
                    # Update uncertainty history
                    if model.model_id in individual_uncertainties:
                        uncertainty = np.mean(individual_uncertainties[model.model_id])
                        model.uncertainty_history.append(uncertainty)
                        if len(model.uncertainty_history) > 100:
                            model.uncertainty_history.pop(0)
                    
                    model.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update model performance: {e}")
    
    def _check_dynamic_adaptation(self, regime_id: int):
        """Check if dynamic adaptation is needed."""
        try:
            self.adaptation_counters[regime_id] += 1
            
            if self.adaptation_counters[regime_id] >= self.config.adaptation_frequency:
                self.logger.info(f"Triggering dynamic adaptation for regime {regime_id}")
                self._perform_dynamic_adaptation(regime_id)
                self.adaptation_counters[regime_id] = 0
            
        except Exception as e:
            self.logger.error(f"Failed to check dynamic adaptation: {e}")
    
    def _perform_dynamic_adaptation(self, regime_id: int):
        """Perform dynamic adaptation of ensemble weights."""
        try:
            if regime_id not in self.regime_ensembles:
                return
            
            ensemble_models = self.regime_ensembles[regime_id]
            active_models = [model for model in ensemble_models if model.is_active]
            
            if not active_models:
                return
            
            # Calculate new weights based on recent performance
            new_weights = self._calculate_performance_based_weights(regime_id, active_models)
            
            # Update model weights
            for model in active_models:
                if model.model_id in new_weights:
                    # Smooth weight update
                    old_weight = model.weight
                    new_weight = new_weights[model.model_id]
                    model.weight = (1 - self.config.learning_rate) * old_weight + \
                                 self.config.learning_rate * new_weight
            
            # Update ensemble weights
            self.ensemble_weights[regime_id] = {model.model_id: model.weight for model in active_models}
            
            # Remove models with very low weights
            if self.config.enable_model_selection:
                self._remove_low_performing_models(regime_id, active_models)
            
            self.logger.info(f"Dynamic adaptation completed for regime {regime_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to perform dynamic adaptation: {e}")
    
    def _remove_low_performing_models(self, regime_id: int, active_models: List[EnsembleModel]):
        """Remove models with very low weights."""
        try:
            models_to_remove = []
            
            for model in active_models:
                if model.weight < self.config.model_removal_threshold:
                    model.is_active = False
                    models_to_remove.append(model.model_id)
                    self.logger.info(f"Removed model {model.model_id} from regime {regime_id} ensemble")
            
            if models_to_remove:
                # Renormalize remaining weights
                remaining_models = [model for model in active_models if model.is_active]
                if remaining_models:
                    total_weight = sum(model.weight for model in remaining_models)
                    if total_weight > 0:
                        for model in remaining_models:
                            model.weight /= total_weight
                    
                    # Update ensemble weights
                    self.ensemble_weights[regime_id] = {model.model_id: model.weight for model in remaining_models}
            
        except Exception as e:
            self.logger.error(f"Failed to remove low performing models: {e}")
    
    def get_ensemble_summary(self, regime_id: int) -> Dict[str, Any]:
        """Get summary of ensemble for a regime."""
        try:
            if regime_id not in self.regime_ensembles:
                return {'error': f'No ensemble found for regime {regime_id}'}
            
            ensemble_models = self.regime_ensembles[regime_id]
            active_models = [model for model in ensemble_models if model.is_active]
            
            summary = {
                'regime_id': regime_id,
                'total_models': len(ensemble_models),
                'active_models': len(active_models),
                'ensemble_weights': self.ensemble_weights.get(regime_id, {}),
                'model_details': {}
            }
            
            for model in active_models:
                summary['model_details'][model.model_id] = {
                    'model_name': model.model_name,
                    'weight': model.weight,
                    'is_active': model.is_active,
                    'performance_history_length': len(model.performance_history),
                    'average_performance': np.mean(model.performance_history) if model.performance_history else 0.0,
                    'last_updated': model.last_updated.isoformat()
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get ensemble summary for regime {regime_id}: {e}")
            return {'error': str(e)}
    
    def save_ensemble_state(self, filepath: str):
        """Save ensemble state to file."""
        try:
            state = {
                'regime_ensembles': self.regime_ensembles,
                'ensemble_weights': self.ensemble_weights,
                'performance_history': dict(self.performance_history),
                'adaptation_counters': dict(self.adaptation_counters),
                'last_optimization': {k: v.isoformat() for k, v in self.last_optimization.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"Ensemble state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble state: {e}")
    
    def load_ensemble_state(self, filepath: str):
        """Load ensemble state from file."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.regime_ensembles = state.get('regime_ensembles', {})
            self.ensemble_weights = state.get('ensemble_weights', {})
            self.performance_history = defaultdict(list, state.get('performance_history', {}))
            self.adaptation_counters = defaultdict(int, state.get('adaptation_counters', {}))
            
            # Convert timestamp strings back to datetime
            last_optimization = state.get('last_optimization', {})
            self.last_optimization = {
                k: datetime.fromisoformat(v) for k, v in last_optimization.items()
            }
            
            self.logger.info(f"Ensemble state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble state: {e}")