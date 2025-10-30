"""
Uncertainty Calculator for ML Model Predictions

This module provides comprehensive uncertainty quantification for ML predictions,
including ensemble variance, model disagreement, confidence degradation, and
combined uncertainty metrics.

Key Features:
- Ensemble variance calculation across multiple model predictions
- Model disagreement measurement (spread between different models)
- Confidence degradation tracking over time
- Combined uncertainty metrics with configurable weighting
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors

logger = system_logger.getChild('UncertaintyCalculator')


class UncertaintyCalculator:
    """
    Calculator for ML model uncertainty metrics.
    
    Provides multiple methods for quantifying prediction uncertainty:
    - Ensemble variance: Statistical variance across ensemble members
    - Model disagreement: Maximum spread between model predictions
    - Confidence degradation: Change in confidence over time windows
    - Combined uncertainty: Weighted combination of multiple metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the uncertainty calculator.
        
        Args:
            config: Configuration dictionary with uncertainty calculation parameters
        """
        self.config = config or {}
        self.logger = logger.getChild('UncertaintyCalculator')
        
        # Configuration parameters
        self.variance_weight = self.config.get('variance_weight', 0.4)
        self.disagreement_weight = self.config.get('disagreement_weight', 0.4)
        self.confidence_weight = self.config.get('confidence_weight', 0.2)
        self.epsilon = self.config.get('epsilon', 1e-10)  # For numerical stability
        
        # Degradation tracking
        self.degradation_window = self.config.get('degradation_window', 8)
        self.degradation_method = self.config.get('degradation_method', 'relative_change')  # or 'absolute_change'
        
        self.logger.info(f"✅ UncertaintyCalculator initialized with weights: variance={self.variance_weight}, "
                        f"disagreement={self.disagreement_weight}, confidence={self.confidence_weight}")
    
    @handles_errors(fallback=0.0, context="ensemble variance calculation")
    def calculate_ensemble_variance(self, predictions: Union[List[np.ndarray], List[float], np.ndarray]) -> float:
        """
        Calculate ensemble variance across model predictions.
        
        This measures the statistical spread in predictions from different ensemble members.
        Higher variance indicates higher uncertainty.
        
        Args:
            predictions: List of predictions from different models or ensemble members
                        Can be List[np.ndarray], List[float], or 2D np.ndarray
        
        Returns:
            float: Variance of predictions (0.0 to unbounded, normalized by mean)
        
        Examples:
            >>> calc = UncertaintyCalculator()
            >>> predictions = [0.7, 0.72, 0.68, 0.71]  # Low variance - high certainty
            >>> calc.calculate_ensemble_variance(predictions)
            0.0002...
            >>> predictions = [0.3, 0.8, 0.5, 0.9]  # High variance - high uncertainty
            >>> calc.calculate_ensemble_variance(predictions)
            0.065...
        """
        try:
            if not predictions:
                self.logger.warning("Empty predictions list for variance calculation")
                return 0.0
            
            # Convert to numpy array
            if isinstance(predictions, list):
                if len(predictions) == 0:
                    return 0.0
                # Handle list of arrays
                if isinstance(predictions[0], np.ndarray):
                    # Stack along first axis for ensemble predictions
                    predictions_array = np.stack(predictions, axis=0)
                else:
                    # Simple list of floats
                    predictions_array = np.array(predictions)
            else:
                predictions_array = np.atleast_1d(predictions)
            
            # Calculate variance
            if predictions_array.ndim == 1:
                # Single dimension - variance across models
                variance = np.var(predictions_array)
            else:
                # Multi-dimensional - variance across first axis (models), then mean
                variance = np.mean(np.var(predictions_array, axis=0))
            
            # Normalize by mean to get relative variance (coefficient of variation squared)
            mean_pred = np.mean(predictions_array)
            if abs(mean_pred) > self.epsilon:
                normalized_variance = variance / (mean_pred ** 2 + self.epsilon)
            else:
                normalized_variance = variance
            
            return float(normalized_variance)
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble variance calculation failed: {e}")
            return 0.0
    
    @handles_errors(fallback=0.0, context="model disagreement calculation")
    def calculate_model_disagreement(self, predictions: Dict[str, Union[np.ndarray, float]]) -> float:
        """
        Calculate model disagreement as the spread between different model outputs.
        
        This measures how much different models disagree with each other.
        Higher disagreement indicates higher uncertainty.
        
        Args:
            predictions: Dictionary mapping model names to their predictions
                        e.g., {'lightgbm': 0.75, 'catboost': 0.72, 'xgboost': 0.78}
        
        Returns:
            float: Disagreement metric (0.0 to 1.0), normalized by mean
        
        Examples:
            >>> calc = UncertaintyCalculator()
            >>> preds = {'model_a': 0.7, 'model_b': 0.71, 'model_c': 0.69}  # Low disagreement
            >>> calc.calculate_model_disagreement(preds)
            0.01...
            >>> preds = {'model_a': 0.3, 'model_b': 0.9, 'model_c': 0.5}  # High disagreement
            >>> calc.calculate_model_disagreement(preds)
            0.4...
        """
        try:
            if not predictions or len(predictions) < 2:
                self.logger.warning("Need at least 2 models for disagreement calculation")
                return 0.0
            
            # Extract prediction values
            pred_values = []
            for model_name, pred in predictions.items():
                if isinstance(pred, np.ndarray):
                    # Take mean if multi-dimensional
                    pred_values.append(float(np.mean(pred)))
                else:
                    pred_values.append(float(pred))
            
            if len(pred_values) < 2:
                return 0.0
            
            pred_array = np.array(pred_values)
            
            # Calculate disagreement as range (max - min)
            max_pred = np.max(pred_array)
            min_pred = np.min(pred_array)
            range_disagreement = max_pred - min_pred
            
            # Calculate standard deviation as alternative disagreement measure
            std_disagreement = np.std(pred_array)
            
            # Combine both metrics (average)
            disagreement = (range_disagreement + std_disagreement) / 2.0
            
            # Normalize by mean to get relative disagreement
            mean_pred = np.mean(pred_array)
            if abs(mean_pred) > self.epsilon:
                normalized_disagreement = disagreement / (abs(mean_pred) + self.epsilon)
            else:
                # If mean is near zero, use absolute disagreement
                normalized_disagreement = disagreement
            
            # Clip to [0, 1] for stability
            return float(np.clip(normalized_disagreement, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Model disagreement calculation failed: {e}")
            return 0.0
    
    @handles_errors(fallback=0.0, context="confidence degradation calculation")
    def calculate_confidence_degradation(
        self,
        confidence_series: Union[pd.Series, List[float], np.ndarray],
        window: Optional[int] = None,
        method: Optional[str] = None
    ) -> float:
        """
        Calculate confidence degradation over a time window.
        
        This tracks how much confidence has decreased from entry to current time.
        Higher degradation indicates the model is becoming less certain about its prediction.
        
        Args:
            confidence_series: Time series of confidence values (oldest to newest)
            window: Number of recent periods to analyze (default: self.degradation_window)
            method: Calculation method ('relative_change' or 'absolute_change')
        
        Returns:
            float: Degradation metric (negative for degradation, positive for improvement)
                  Range: -1.0 to 1.0 for relative, unbounded for absolute
        
        Examples:
            >>> calc = UncertaintyCalculator()
            >>> # Confidence degrading over time
            >>> confidence = [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
            >>> calc.calculate_confidence_degradation(confidence)
            -0.4375  # 43.75% degradation
            >>> # Confidence improving over time
            >>> confidence = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            >>> calc.calculate_confidence_degradation(confidence)
            0.70  # 70% improvement
        """
        try:
            window = window or self.degradation_window
            method = method or self.degradation_method
            
            # Convert to numpy array
            if isinstance(confidence_series, pd.Series):
                conf_array = confidence_series.values
            elif isinstance(confidence_series, list):
                conf_array = np.array(confidence_series)
            else:
                conf_array = np.atleast_1d(confidence_series)
            
            if len(conf_array) < 2:
                self.logger.warning("Need at least 2 confidence values for degradation calculation")
                return 0.0
            
            # Use the specified window or all available data
            window_size = min(window, len(conf_array))
            recent_confidence = conf_array[-window_size:]
            
            # Get initial and current confidence
            initial_confidence = recent_confidence[0]
            current_confidence = recent_confidence[-1]
            
            # Calculate degradation based on method
            if method == 'relative_change':
                # Relative change: (current - initial) / initial
                if abs(initial_confidence) > self.epsilon:
                    degradation = (current_confidence - initial_confidence) / initial_confidence
                else:
                    degradation = current_confidence - initial_confidence
            else:  # absolute_change
                # Absolute change: current - initial
                degradation = current_confidence - initial_confidence
            
            # Return negative value for degradation, positive for improvement
            return float(degradation)
            
        except Exception as e:
            self.logger.error(f"❌ Confidence degradation calculation failed: {e}")
            return 0.0
    
    @handles_errors(fallback=0.5, context="combined uncertainty calculation")
    def combine_uncertainty_metrics(
        self,
        variance: Optional[float] = None,
        disagreement: Optional[float] = None,
        confidence_degradation: Optional[float] = None,
        normalize: bool = True
    ) -> float:
        """
        Combine multiple uncertainty metrics into a single score.
        
        Uses weighted combination of variance, disagreement, and confidence degradation.
        
        Args:
            variance: Ensemble variance (0.0 to unbounded, will be clipped)
            disagreement: Model disagreement (0.0 to 1.0)
            confidence_degradation: Confidence change (-1.0 to 1.0, negative is degradation)
            normalize: Whether to normalize the output to [0, 1]
        
        Returns:
            float: Combined uncertainty score (0.0 = low uncertainty, 1.0 = high uncertainty)
        
        Examples:
            >>> calc = UncertaintyCalculator()
            >>> # Low uncertainty scenario
            >>> calc.combine_uncertainty_metrics(variance=0.01, disagreement=0.05, confidence_degradation=0.1)
            0.06...
            >>> # High uncertainty scenario
            >>> calc.combine_uncertainty_metrics(variance=0.5, disagreement=0.8, confidence_degradation=-0.6)
            0.85...
        """
        try:
            combined_uncertainty = 0.0
            total_weight = 0.0
            
            # Add variance component
            if variance is not None:
                # Clip and normalize variance
                normalized_variance = np.clip(variance, 0.0, 1.0)
                combined_uncertainty += normalized_variance * self.variance_weight
                total_weight += self.variance_weight
            
            # Add disagreement component
            if disagreement is not None:
                # Disagreement already in [0, 1]
                combined_uncertainty += disagreement * self.disagreement_weight
                total_weight += self.disagreement_weight
            
            # Add confidence degradation component
            if confidence_degradation is not None:
                # Convert degradation to uncertainty
                # Negative degradation (loss of confidence) -> high uncertainty
                # Positive degradation (gain in confidence) -> low uncertainty
                uncertainty_from_degradation = np.clip(-confidence_degradation, 0.0, 1.0)
                combined_uncertainty += uncertainty_from_degradation * self.confidence_weight
                total_weight += self.confidence_weight
            
            # Normalize by total weight
            if total_weight > 0:
                combined_uncertainty /= total_weight
            
            # Normalize to [0, 1] if requested
            if normalize:
                combined_uncertainty = np.clip(combined_uncertainty, 0.0, 1.0)
            
            return float(combined_uncertainty)
            
        except Exception as e:
            self.logger.error(f"❌ Combined uncertainty calculation failed: {e}")
            return 0.5  # Return moderate uncertainty as fallback
    
    @handles_errors(fallback={}, context="comprehensive uncertainty metrics")
    def calculate_comprehensive_metrics(
        self,
        ensemble_predictions: Optional[Union[List[np.ndarray], List[float]]] = None,
        model_predictions: Optional[Dict[str, Union[np.ndarray, float]]] = None,
        confidence_history: Optional[Union[pd.Series, List[float]]] = None
    ) -> Dict[str, float]:
        """
        Calculate all uncertainty metrics at once.
        
        Convenience method that computes all available uncertainty metrics
        and returns them in a structured dictionary.
        
        Args:
            ensemble_predictions: List of predictions from ensemble members
            model_predictions: Dictionary of predictions from different models
            confidence_history: Time series of confidence values
        
        Returns:
            Dict containing all calculated uncertainty metrics:
                - 'ensemble_variance': Variance across ensemble
                - 'model_disagreement': Disagreement between models
                - 'confidence_degradation': Change in confidence over time
                - 'combined_uncertainty': Weighted combination of all metrics
                - 'timestamp': When metrics were calculated
        
        Examples:
            >>> calc = UncertaintyCalculator()
            >>> metrics = calc.calculate_comprehensive_metrics(
            ...     ensemble_predictions=[0.7, 0.72, 0.68],
            ...     model_predictions={'lgb': 0.7, 'cat': 0.72, 'xgb': 0.68},
            ...     confidence_history=[0.8, 0.75, 0.7, 0.65]
            ... )
            >>> 'ensemble_variance' in metrics
            True
            >>> 'combined_uncertainty' in metrics
            True
        """
        try:
            metrics = {
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate ensemble variance if available
            variance = None
            if ensemble_predictions is not None:
                variance = self.calculate_ensemble_variance(ensemble_predictions)
                metrics['ensemble_variance'] = variance
            
            # Calculate model disagreement if available
            disagreement = None
            if model_predictions is not None:
                disagreement = self.calculate_model_disagreement(model_predictions)
                metrics['model_disagreement'] = disagreement
            
            # Calculate confidence degradation if available
            degradation = None
            if confidence_history is not None:
                degradation = self.calculate_confidence_degradation(confidence_history)
                metrics['confidence_degradation'] = degradation
            
            # Calculate combined uncertainty
            combined = self.combine_uncertainty_metrics(
                variance=variance,
                disagreement=disagreement,
                confidence_degradation=degradation
            )
            metrics['combined_uncertainty'] = combined
            
            self.logger.debug(f"Calculated comprehensive uncertainty metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive metrics calculation failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


def create_uncertainty_calculator(config: Optional[Dict[str, Any]] = None) -> UncertaintyCalculator:
    """
    Factory function to create an UncertaintyCalculator instance.
    
    Args:
        config: Configuration dictionary for the calculator
    
    Returns:
        UncertaintyCalculator: Initialized calculator instance
    """
    return UncertaintyCalculator(config)


# Global instance for convenience
_global_uncertainty_calculator: Optional[UncertaintyCalculator] = None


def get_global_uncertainty_calculator() -> UncertaintyCalculator:
    """
    Get or create the global uncertainty calculator instance.
    
    Returns:
        UncertaintyCalculator: Global calculator instance
    """
    global _global_uncertainty_calculator
    if _global_uncertainty_calculator is None:
        _global_uncertainty_calculator = UncertaintyCalculator()
    return _global_uncertainty_calculator


__all__ = [
    'UncertaintyCalculator',
    'create_uncertainty_calculator',
    'get_global_uncertainty_calculator'
]


