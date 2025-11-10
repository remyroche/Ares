"""
Ensemble Disagreement Features

This module provides centralized disagreement features for all ensemble models
(regime_ensemble_training, train_analyst_ensemble, train_tactician_ensemble).

Disagreement features capture model uncertainty and disagreement to improve
ensemble predictions and risk management.

Core Features (6 features used by all ensemble models):
1. prediction_dispersion: Variance of predictions across models
2. confidence_gap: Margin between top predictions
3. uncertainty: Normalized entropy (uncertainty measure)
4. prediction_range: Range of predictions (max - min)
5. avg_divergence: Average pairwise model divergence
6. max_confidence: Highest confidence among models
7. disagreement_rate: Proportion of models disagreeing on direction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats
import logging


class EnsembleDisagreementFeatures:
    """
    Centralized disagreement features calculator for ensemble models.

    This class provides a consistent set of disagreement features across
    all ensemble models in the system.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the ensemble disagreement features calculator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_disagreement_features(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_probabilities: Dict[str, np.ndarray],
        model_confidences: Optional[Dict[str, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate core disagreement features from model predictions.

        This is the main entry point for calculating disagreement features.
        Returns the 7 core features used by all ensemble models.

        Args:
            model_predictions: Dict mapping model names to prediction arrays
            model_probabilities: Dict mapping model names to probability arrays
            model_confidences: Optional dict mapping model names to confidence arrays
            feature_names: Optional list of specific features to calculate
                          (if None, calculates all 7 core features)

        Returns:
            Dict containing disagreement features as pandas Series
        """
        try:
            # Infer length from inputs
            length = self._infer_length(model_predictions, model_probabilities, model_confidences)
            index = pd.RangeIndex(length)

            # Calculate all disagre features
            features = {}

            # 1. Prediction Dispersion (variance of predictions)
            features['prediction_dispersion'] = self._calculate_prediction_dispersion(
                model_predictions, index
            )

            # 2. Confidence Gap (margin between top predictions)
            features['confidence_gap'] = self._calculate_confidence_gap(
                model_probabilities, index
            )

            # 3. Uncertainty (normalized entropy)
            features['uncertainty'] = self._calculate_uncertainty(
                model_probabilities, index
            )

            # 4. Prediction Range (max - min)
            features['prediction_range'] = self._calculate_prediction_range(
                model_predictions, index
            )

            # 5. Average Divergence (pairwise model divergence)
            features['avg_divergence'] = self._calculate_avg_divergence(
                model_probabilities, index
            )

            # 6. Max Confidence (highest confidence among models)
            features['max_confidence'] = self._calculate_max_confidence(
                model_probabilities, index
            )

            # 7. Disagreement Rate (proportion disagreeing on direction)
            features['disagreement_rate'] = self._calculate_disagreement_rate(
                model_predictions, index
            )

            # Filter by requested feature names if provided
            if feature_names is not None:
                features = {k: v for k, v in features.items() if k in feature_names}

            return features

        except Exception as e:
            self.logger.error(f"Error calculating disagreement features: {e}")
            length = self._infer_length(model_predictions, model_probabilities, model_confidences)
            index = pd.RangeIndex(length)
            return self._get_default_features(index)

    def _infer_length(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_probabilities: Dict[str, np.ndarray],
        model_confidences: Optional[Dict[str, np.ndarray]] = None
    ) -> int:
        """Infer the length of the data from the inputs."""
        for container in (model_predictions, model_probabilities, model_confidences or {}):
            if not container:
                continue
            for values in container.values():
                arr = np.asarray(values)
                if arr.ndim == 0:
                    continue
                return arr.shape[0] if arr.ndim > 1 else arr.size
        return 1

    def _stack_prediction_matrix(
        self,
        model_predictions: Dict[str, np.ndarray],
        index: pd.Index
    ) -> np.ndarray:
        """Stack prediction arrays into a matrix (n_models, n_samples)."""
        arrays = []
        target_length = len(index)

        for model_name, values in model_predictions.items():
            arr = np.asarray(values, dtype=float)
            arr = np.atleast_1d(arr)

            if arr.ndim > 1:
                if 1 in arr.shape:
                    arr = arr.reshape(-1)
                else:
                    raise ValueError(f"Prediction array for {model_name} has unexpected shape {arr.shape}")

            if arr.shape[0] != target_length:
                raise ValueError(
                    f"Prediction array for {model_name} length {arr.shape[0]} does not match expected {target_length}"
                )

            arrays.append(arr)

        if not arrays:
            return np.empty((0, target_length))

        return np.vstack(arrays)

    def _stack_probability_tensor(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> np.ndarray:
        """Stack probability arrays into a tensor (n_models, n_samples, n_classes)."""
        tensors = []
        target_length = len(index)

        for model_name, values in model_probabilities.items():
            arr = np.asarray(values, dtype=float)

            if arr.ndim == 1:
                # Convert single probability to binary probabilities
                arr = np.column_stack([1.0 - arr, arr])
            elif arr.ndim == 2:
                pass
            elif arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            else:
                raise ValueError(f"Probability array for {model_name} has unexpected shape {arr.shape}")

            if arr.shape[0] != target_length:
                raise ValueError(
                    f"Probability array for {model_name} length {arr.shape[0]} does not match expected {target_length}"
                )

            tensors.append(arr)

        if not tensors:
            return np.empty((0, target_length, 0))

        tensor = np.stack(tensors, axis=0)

        # Normalize per sample to ensure valid probabilities
        normalised = np.zeros_like(tensor)
        n_classes = tensor.shape[2] if tensor.ndim == 3 else 0

        if n_classes == 0:
            return tensor

        for model_idx in range(tensor.shape[0]):
            for sample_idx in range(tensor.shape[1]):
                row = tensor[model_idx, sample_idx]
                total = row.sum()
                if total <= 0:
                    normalised[model_idx, sample_idx] = np.full(n_classes, 1.0 / n_classes)
                else:
                    normalised[model_idx, sample_idx] = row / total

        return normalised

    def _calculate_prediction_dispersion(
        self,
        model_predictions: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate prediction dispersion (variance of predictions across models).

        High dispersion indicates models strongly disagree on the prediction magnitude.
        """
        try:
            if not model_predictions:
                return pd.Series(0.0, index=index)

            matrix = self._stack_prediction_matrix(model_predictions, index)
            if matrix.size == 0:
                return pd.Series(0.0, index=index)

            prediction_variance = np.var(matrix, axis=0)
            return pd.Series(prediction_variance, index=index)

        except Exception as e:
            self.logger.error(f"Error calculating prediction dispersion: {e}")
            return pd.Series(0.0, index=index)

    def _calculate_confidence_gap(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate confidence gap (margin between highest and second-highest probability).

        High gap = conviction trade, Low gap = uncertain market regime.
        """
        try:
            if not model_probabilities:
                return pd.Series(0.0, index=index)

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                return pd.Series(0.0, index=index)

            # Average probabilities across models
            avg_probs = tensor.mean(axis=0)

            if avg_probs.shape[1] < 2:
                return pd.Series(0.0, index=index)

            sorted_probs = np.sort(avg_probs, axis=1)
            max_confidence = sorted_probs[:, -1]
            second_max = sorted_probs[:, -2]
            confidence_gap = max_confidence - second_max

            return pd.Series(confidence_gap, index=index)

        except Exception as e:
            self.logger.error(f"Error calculating confidence gap: {e}")
            return pd.Series(0.0, index=index)

    def _calculate_uncertainty(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate uncertainty (normalized entropy of average probability distribution).

        High entropy = scattered belief = uncertain trade environment.
        """
        try:
            if not model_probabilities:
                return pd.Series(0.0, index=index)

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                return pd.Series(0.0, index=index)

            # Average probabilities across models
            avg_probs = tensor.mean(axis=0)

            # Calculate entropy
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = stats.entropy(avg_probs, axis=1)

            # Normalize by maximum possible entropy
            n_classes = avg_probs.shape[1] if avg_probs.ndim == 2 else 1
            max_entropy = np.log(n_classes) if n_classes > 0 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy

            series_normalized = pd.Series(normalized_entropy, index=index)
            return series_normalized.fillna(0.0)

        except Exception as e:
            self.logger.error(f"Error calculating uncertainty: {e}")
            return pd.Series(0.0, index=index)

    def _calculate_prediction_range(
        self,
        model_predictions: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate prediction range (max - min prediction across models).

        Large range indicates models have very different views on the trade.
        """
        try:
            if not model_predictions:
                return pd.Series(0.0, index=index)

            matrix = self._stack_prediction_matrix(model_predictions, index)
            if matrix.size == 0:
                return pd.Series(0.0, index=index)

            pred_range = matrix.max(axis=0) - matrix.min(axis=0)
            return pd.Series(pred_range, index=index)

        except Exception as e:
            self.logger.error(f"Error calculating prediction range: {e}")
            return pd.Series(0.0, index=index)

    def _calculate_avg_divergence(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate average divergence (mean pairwise Jensen-Shannon divergence).

        Large divergence = models view market very differently.
        """
        try:
            if len(model_probabilities) < 2:
                return pd.Series(0.0, index=index)

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                return pd.Series(0.0, index=index)

            # Calculate pairwise JS divergence
            js_values = []
            for i in range(tensor.shape[0]):
                for j in range(i + 1, tensor.shape[0]):
                    p = tensor[i]
                    q = tensor[j]
                    m = 0.5 * (p + q)
                    js = 0.5 * (stats.entropy(p, m, axis=1) + stats.entropy(q, m, axis=1))
                    js_values.append(js)

            if not js_values:
                return pd.Series(0.0, index=index)

            avg_divergence = np.mean(js_values, axis=0)
            return pd.Series(avg_divergence, index=index)

        except Exception as e:
            self.logger.error(f"Error calculating average divergence: {e}")
            return pd.Series(0.0, index=index)

    def _calculate_max_confidence(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate max confidence (highest probability among all models).

        Represents the most confident prediction across the ensemble.
        """
        try:
            if not model_probabilities:
                return pd.Series(0.0, index=index)

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                return pd.Series(0.0, index=index)

            # Average probabilities across models
            avg_probs = tensor.mean(axis=0)
            max_confidence = np.max(avg_probs, axis=1)

            return pd.Series(max_confidence, index=index)

        except Exception as e:
            self.logger.error(f"Error calculating max confidence: {e}")
            return pd.Series(0.0, index=index)

    def _calculate_disagreement_rate(
        self,
        model_predictions: Dict[str, np.ndarray],
        index: pd.Index
    ) -> pd.Series:
        """
        Calculate disagreement rate (proportion of models disagreeing on direction).

        Example: 4 models → 3 long, 1 short ⇒ disagreement rate = 25%.
        Useful as a filter: trade only if ≥70% of models agree on direction.
        """
        try:
            if not model_predictions:
                return pd.Series(0.0, index=index)

            matrix = self._stack_prediction_matrix(model_predictions, index)
            if matrix.size == 0:
                return pd.Series(0.0, index=index)

            total_predictions = matrix.shape[0]
            if total_predictions == 0:
                return pd.Series(0.0, index=index)

            # Count long and short votes
            long_votes = (matrix > 0).sum(axis=0) / total_predictions
            short_votes = (matrix < 0).sum(axis=0) / total_predictions

            # Disagreement rate = 1 - max(long_ratio, short_ratio)
            disagreement_rate = 1.0 - np.maximum(long_votes, short_votes)

            return pd.Series(disagreement_rate, index=index)

        except Exception as e:
            self.logger.error(f"Error calculating disagreement rate: {e}")
            return pd.Series(0.0, index=index)

    def _get_default_features(self, index: pd.Index) -> Dict[str, pd.Series]:
        """Get default disagreement features when calculation fails."""
        zero = pd.Series(0.0, index=index)
        return {
            'prediction_dispersion': zero.copy(),
            'confidence_gap': zero.copy(),
            'uncertainty': zero.copy(),
            'prediction_range': zero.copy(),
            'avg_divergence': zero.copy(),
            'max_confidence': zero.copy(),
            'disagreement_rate': zero.copy()
        }


# Convenience functions for easy imports

def calculate_ensemble_disagreement_features(
    model_predictions: Dict[str, np.ndarray],
    model_probabilities: Dict[str, np.ndarray],
    model_confidences: Optional[Dict[str, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, pd.Series]:
    """
    Calculate ensemble disagreement features (convenience function).

    Args:
        model_predictions: Dict mapping model names to prediction arrays
        model_probabilities: Dict mapping model names to probability arrays
        model_confidences: Optional dict mapping model names to confidence arrays
        feature_names: Optional list of specific features to calculate
        logger: Optional logger instance

    Returns:
        Dict containing disagreement features as pandas Series
    """
    calculator = EnsembleDisagreementFeatures(logger=logger)
    return calculator.calculate_disagreement_features(
        model_predictions, model_probabilities, model_confidences, feature_names
    )


def get_core_feature_names() -> List[str]:
    """
    Get list of core disagreement feature names.

    Returns:
        List of 7 core feature names used by all ensemble models
    """
    return [
        'prediction_dispersion',
        'confidence_gap',
        'uncertainty',
        'prediction_range',
        'avg_divergence',
        'max_confidence',
        'disagreement_rate'
    ]
