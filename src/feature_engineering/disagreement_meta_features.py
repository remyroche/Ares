"""
Disagreement Meta-Features for Ensemble Models

This module implements comprehensive disagreement meta-features for ensemble models
to capture model disagreement and uncertainty in predictions.

Features implemented:
1. Prediction Dispersion
Variance of predicted returns across models.
Std. deviation of probability of "up" vs "down" across models.
👉 If variance is high → models disagree strongly → signal less reliable.
2. Direction Conflict
Fraction of models long vs short (hard votes).
Example: 4 models → 3 long, 1 short ⇒ disagreement rate = 25%.
👉 Useful as a filter: trade only if ≥70% of models agree on direction.
3. Ensemble Confidence Gap (Margin)
Difference between highest and second-highest aggregated probability.
High margin = conviction trade.
Low margin = market regime uncertain, avoid or size down.
4. Uncertainty / Entropy
Entropy of the average probability distribution.
H=−∑kpˉklog⁡pˉkH = -\sum_k \bar p_k \log \bar p_kH=−∑k​pˉ​k​logpˉ​k​.
👉 High entropy = scattered belief → uncertain trade environment.
5. Model Spread Indicators
Range: max⁡(plong(m))−min⁡(plong(m))\max(p^{(m)}_{long}) - \min(p^{(m)}_{long})max(plong(m)​)−min(plong(m)​).
IQR (interquartile range): of predicted returns/probs across models.
👉 Captures disagreement magnitude on trade strength.
6. Pairwise Divergence
Jensen–Shannon divergence or KL divergence between model probability distributions.
Large divergence = models view market very differently.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon
import logging

class DisagreementMetaFeatures:
    """
    Comprehensive disagreement meta-features for ensemble models.
    
    This class implements all the disagreement meta-features that help
    identify when ensemble models disagree and signal uncertainty.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the disagreement meta-features calculator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_all_disagreement_features(
        self, 
        model_predictions: Dict[str, np.ndarray],
        model_probabilities: Dict[str, np.ndarray],
        model_confidences: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate all disagreement meta-features from model predictions.
        
        Args:
            model_predictions: Dict mapping model names to prediction arrays
            model_probabilities: Dict mapping model names to probability arrays
            model_confidences: Optional dict mapping model names to confidence arrays
            
        Returns:
            Dict containing all disagreement meta-features
        """
        try:
            disagreement_features = {}
            
            # 1. Prediction Dispersion
            disagreement_features.update(self._calculate_prediction_dispersion(model_predictions))
            
            # 2. Direction Conflict
            disagreement_features.update(self._calculate_direction_conflict(model_predictions))
            
            # 3. Ensemble Confidence Gap
            disagreement_features.update(self._calculate_confidence_gap(model_probabilities))
            
            # 4. Uncertainty/Entropy
            disagreement_features.update(self._calculate_entropy_uncertainty(model_probabilities))
            
            # 5. Model Spread Indicators
            disagreement_features.update(self._calculate_spread_indicators(model_predictions, model_probabilities))
            
            # 6. Pairwise Divergence
            disagreement_features.update(self._calculate_pairwise_divergence(model_probabilities))
            
            return disagreement_features
            
        except Exception as e:
            self.logger.error(f"Error calculating disagreement features: {e}")
            return self._get_default_disagreement_features()
    
    def _calculate_prediction_dispersion(self, model_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate prediction dispersion meta-features.
        
        Measures:
        - Variance of predicted returns across models
        - Standard deviation of probability of "up" vs "down" across models
        
        Args:
            model_predictions: Dict mapping model names to prediction arrays
            
        Returns:
            Dict containing dispersion features
        """
        try:
            if not model_predictions:
                return {"prediction_dispersion": 0.0, "prediction_std": 0.0}
            
            # Convert predictions to numpy array
            pred_array = np.array(list(model_predictions.values()))
            
            # Calculate variance and standard deviation
            prediction_variance = np.var(pred_array, axis=0).mean()
            prediction_std = np.std(pred_array, axis=0).mean()
            
            return {
                "prediction_dispersion": float(prediction_variance),
                "prediction_std": float(prediction_std)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction dispersion: {e}")
            return {"prediction_dispersion": 0.0, "prediction_std": 0.0}
    
    def _calculate_direction_conflict(self, model_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate direction conflict meta-features.
        
        Measures:
        - Fraction of models long vs short (hard votes)
        - Disagreement rate as a filter for trading decisions
        
        Args:
            model_predictions: Dict mapping model names to prediction arrays
            
        Returns:
            Dict containing direction conflict features
        """
        try:
            if not model_predictions:
                return {"direction_conflict": 0.0, "long_ratio": 0.5, "disagreement_rate": 0.0}
            
            # Convert predictions to binary decisions (long/short)
            pred_array = np.array(list(model_predictions.values()))
            
            # Count long vs short predictions
            long_predictions = np.sum(pred_array > 0.5, axis=0)
            total_predictions = len(model_predictions)
            
            # Calculate ratios
            long_ratio = long_predictions.mean() / total_predictions
            short_ratio = 1.0 - long_ratio
            
            # Calculate disagreement rate (how much models disagree)
            disagreement_rate = 1.0 - max(long_ratio, short_ratio)
            
            return {
                "direction_conflict": float(disagreement_rate),
                "long_ratio": float(long_ratio),
                "short_ratio": float(short_ratio),
                "disagreement_rate": float(disagreement_rate)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating direction conflict: {e}")
            return {"direction_conflict": 0.0, "long_ratio": 0.5, "disagreement_rate": 0.0}
    
    def _calculate_confidence_gap(self, model_probabilities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate ensemble confidence gap meta-features.
        
        Measures:
        - Difference between highest and second-highest aggregated probability
        - High margin = conviction trade, Low margin = uncertain market regime
        
        Args:
            model_probabilities: Dict mapping model names to probability arrays
            
        Returns:
            Dict containing confidence gap features
        """
        try:
            if not model_probabilities:
                return {"confidence_gap": 0.0, "max_confidence": 0.0, "second_max_confidence": 0.0}
            
            # Aggregate probabilities across models
            prob_array = np.array(list(model_probabilities.values()))
            aggregated_probs = np.mean(prob_array, axis=0)
            
            # Sort probabilities to find top two
            sorted_probs = np.sort(aggregated_probs)
            max_confidence = sorted_probs[-1]
            second_max_confidence = sorted_probs[-2] if len(sorted_probs) > 1 else 0.0
            
            # Calculate confidence gap
            confidence_gap = max_confidence - second_max_confidence
            
            return {
                "confidence_gap": float(confidence_gap),
                "max_confidence": float(max_confidence),
                "second_max_confidence": float(second_max_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence gap: {e}")
            return {"confidence_gap": 0.0, "max_confidence": 0.0, "second_max_confidence": 0.0}
    
    def _calculate_entropy_uncertainty(self, model_probabilities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate uncertainty/entropy meta-features.
        
        Measures:
        - Entropy of the average probability distribution
        - High entropy = scattered belief = uncertain trade environment
        
        Args:
            model_probabilities: Dict mapping model names to probability arrays
            
        Returns:
            Dict containing entropy features
        """
        try:
            if not model_probabilities:
                return {"entropy": 0.0, "uncertainty": 0.0}
            
            # Calculate average probability distribution
            prob_array = np.array(list(model_probabilities.values()))
            avg_probs = np.mean(prob_array, axis=0)
            
            # Calculate entropy: H = -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            avg_probs_safe = avg_probs + epsilon
            entropy = -np.sum(avg_probs_safe * np.log(avg_probs_safe))
            
            # Normalize entropy (max entropy for uniform distribution)
            max_entropy = np.log(len(avg_probs))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return {
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "uncertainty": float(normalized_entropy)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy uncertainty: {e}")
            return {"entropy": 0.0, "uncertainty": 0.0}
    
    def _calculate_spread_indicators(self, model_predictions: Dict[str, np.ndarray], 
                                   model_probabilities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate model spread indicators.
        
        Measures:
        - Range: max(p_long) - min(p_long) across models
        - IQR (interquartile range) of predicted returns/probs across models
        
        Args:
            model_predictions: Dict mapping model names to prediction arrays
            model_probabilities: Dict mapping model names to probability arrays
            
        Returns:
            Dict containing spread indicator features
        """
        try:
            spread_features = {}
            
            # Calculate spread for predictions
            if model_predictions:
                pred_array = np.array(list(model_predictions.values()))
                pred_range = np.max(pred_array, axis=0) - np.min(pred_array, axis=0)
                pred_iqr = np.percentile(pred_array, 75, axis=0) - np.percentile(pred_array, 25, axis=0)
                
                spread_features.update({
                    "prediction_range": float(pred_range.mean()),
                    "prediction_iqr": float(pred_iqr.mean())
                })
            
            # Calculate spread for probabilities
            if model_probabilities:
                prob_array = np.array(list(model_probabilities.values()))
                prob_range = np.max(prob_array, axis=0) - np.min(prob_array, axis=0)
                prob_iqr = np.percentile(prob_array, 75, axis=0) - np.percentile(prob_array, 25, axis=0)
                
                spread_features.update({
                    "probability_range": float(prob_range.mean()),
                    "probability_iqr": float(prob_iqr.mean())
                })
            
            return spread_features
            
        except Exception as e:
            self.logger.error(f"Error calculating spread indicators: {e}")
            return {"prediction_range": 0.0, "prediction_iqr": 0.0, "probability_range": 0.0, "probability_iqr": 0.0}
    
    def _calculate_pairwise_divergence(self, model_probabilities: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate pairwise divergence meta-features.
        
        Measures:
        - Jensen-Shannon divergence between model probability distributions
        - KL divergence between model probability distributions
        - Large divergence = models view market very differently
        
        Args:
            model_probabilities: Dict mapping model names to probability arrays
            
        Returns:
            Dict containing pairwise divergence features
        """
        try:
            if len(model_probabilities) < 2:
                return {"js_divergence": 0.0, "kl_divergence": 0.0, "avg_divergence": 0.0}
            
            model_names = list(model_probabilities.keys())
            prob_arrays = list(model_probabilities.values())
            
            js_divergences = []
            kl_divergences = []
            
            # Calculate pairwise divergences
            for i in range(len(prob_arrays)):
                for j in range(i + 1, len(prob_arrays)):
                    prob1 = prob_arrays[i]
                    prob2 = prob_arrays[j]
                    
                    # Jensen-Shannon divergence
                    js_div = jensenshannon(prob1, prob2)
                    js_divergences.append(js_div)
                    
                    # KL divergence (symmetric)
                    kl_div = self._calculate_kl_divergence(prob1, prob2)
                    kl_divergences.append(kl_div)
            
            # Calculate average divergences
            avg_js_divergence = np.mean(js_divergences) if js_divergences else 0.0
            avg_kl_divergence = np.mean(kl_divergences) if kl_divergences else 0.0
            avg_divergence = (avg_js_divergence + avg_kl_divergence) / 2.0
            
            return {
                "js_divergence": float(avg_js_divergence),
                "kl_divergence": float(avg_kl_divergence),
                "avg_divergence": float(avg_divergence)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pairwise divergence: {e}")
            return {"js_divergence": 0.0, "kl_divergence": 0.0, "avg_divergence": 0.0}
    
    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate KL divergence between two probability distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence value
        """
        try:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_safe = p + epsilon
            q_safe = q + epsilon
            
            # Calculate KL divergence: KL(p||q) = sum(p * log(p/q))
            kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
            
            return float(kl_div)
            
        except Exception as e:
            self.logger.error(f"Error calculating KL divergence: {e}")
            return 0.0
    
    def _get_default_disagreement_features(self) -> Dict[str, float]:
        """
        Get default disagreement features when calculation fails.
        
        Returns:
            Dict containing default disagreement features
        """
        return {
            "prediction_dispersion": 0.0,
            "prediction_std": 0.0,
            "direction_conflict": 0.0,
            "long_ratio": 0.5,
            "short_ratio": 0.5,
            "disagreement_rate": 0.0,
            "confidence_gap": 0.0,
            "max_confidence": 0.0,
            "second_max_confidence": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "uncertainty": 0.0,
            "prediction_range": 0.0,
            "prediction_iqr": 0.0,
            "probability_range": 0.0,
            "probability_iqr": 0.0,
            "js_divergence": 0.0,
            "kl_divergence": 0.0,
            "avg_divergence": 0.0
        }
    
    def calculate_disagreement_features_for_ensemble(
        self, 
        ensemble_predictions: Dict[str, Any],
        is_live: bool = False
    ) -> Dict[str, float]:
        """
        Calculate disagreement features for an ensemble of models.
        
        Args:
            ensemble_predictions: Dict containing ensemble prediction data
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            Dict containing disagreement features
        """
        try:
            # Extract model predictions and probabilities
            model_predictions = {}
            model_probabilities = {}
            model_confidences = {}
            
            for model_name, prediction_data in ensemble_predictions.items():
                if isinstance(prediction_data, dict):
                    # Handle dict format
                    if 'prediction' in prediction_data:
                        model_predictions[model_name] = np.array([prediction_data['prediction']])
                    if 'probability' in prediction_data:
                        model_probabilities[model_name] = np.array([prediction_data['probability']])
                    if 'confidence' in prediction_data:
                        model_confidences[model_name] = np.array([prediction_data['confidence']])
                elif isinstance(prediction_data, (int, float, np.ndarray)):
                    # Handle direct numeric format
                    model_predictions[model_name] = np.array([float(prediction_data)])
                    model_probabilities[model_name] = np.array([float(prediction_data)])
            
            # Calculate disagreement features
            disagreement_features = self.calculate_all_disagreement_features(
                model_predictions, model_probabilities, model_confidences
            )
            
            return disagreement_features
            
        except Exception as e:
            self.logger.error(f"Error calculating disagreement features for ensemble: {e}")
            return self._get_default_disagreement_features()