"""
import warnings
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
from typing import Dict, Any, Optional
from scipy import stats
import logging

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

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

    def _infer_length(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_probabilities: Dict[str, np.ndarray],
        model_confidences: Optional[Dict[str, np.ndarray]] = None
    ) -> int:
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
        tensors = []
        target_length = len(index)

        for model_name, values in model_probabilities.items():
            arr = np.asarray(values, dtype=float)

            if arr.ndim == 1:
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

        # Normalise per sample
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
        
    def calculate_all_disagreement_features(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_probabilities: Dict[str, np.ndarray],
        model_confidences: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, pd.Series]:
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
            length = self._infer_length(model_predictions, model_probabilities, model_confidences)
            index = pd.RangeIndex(length)

            disagreement_features: Dict[str, pd.Series] = {}

            # 1. Prediction Dispersion
            disagreement_features.update(
                self._calculate_prediction_dispersion(model_predictions, index)
            )

            # 2. Direction Conflict
            disagreement_features.update(
                self._calculate_direction_conflict(model_predictions, index)
            )

            # 3. Ensemble Confidence Gap
            disagreement_features.update(
                self._calculate_confidence_gap(model_probabilities, index)
            )

            # 4. Uncertainty/Entropy
            disagreement_features.update(
                self._calculate_entropy_uncertainty(model_probabilities, index)
            )

            # 5. Model Spread Indicators
            disagreement_features.update(
                self._calculate_spread_indicators(model_predictions, model_probabilities, index)
            )

            # 6. Pairwise Divergence
            disagreement_features.update(
                self._calculate_pairwise_divergence(model_probabilities, index)
            )

            return disagreement_features

        except Exception as e:
            self.logger.exception("Error calculating disagreement features")
            length = self._infer_length(model_predictions, model_probabilities, model_confidences)
            index = pd.RangeIndex(length)
            return self._get_default_disagreement_features(index)
    
    def _calculate_prediction_dispersion(
        self,
        model_predictions: Dict[str, np.ndarray],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
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
                zero = pd.Series(0.0, index=index)
                return {"prediction_dispersion": zero, "prediction_std": zero}

            matrix = self._stack_prediction_matrix(model_predictions, index)
            if matrix.size == 0:
                zero = pd.Series(0.0, index=index)
                return {"prediction_dispersion": zero, "prediction_std": zero}

            prediction_variance = np.var(matrix, axis=0)
            prediction_std = np.sqrt(prediction_variance)

            return {
                "prediction_dispersion": pd.Series(prediction_variance, index=index),
                "prediction_std": pd.Series(prediction_std, index=index)
            }

        except Exception as e:
            self.logger.error(f"Error calculating prediction dispersion: {e}")
            zero = pd.Series(0.0, index=index)
            return {"prediction_dispersion": zero, "prediction_std": zero}
    
    def _calculate_direction_conflict(
        self,
        model_predictions: Dict[str, np.ndarray],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
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
                default = pd.Series(0.0, index=index)
                half = pd.Series(0.5, index=index)
                return {
                    "direction_conflict": default,
                    "long_ratio": half,
                    "short_ratio": half,
                    "disagreement_rate": default
                }

            matrix = self._stack_prediction_matrix(model_predictions, index)
            if matrix.size == 0:
                default = pd.Series(0.0, index=index)
                half = pd.Series(0.5, index=index)
                return {
                    "direction_conflict": default,
                    "long_ratio": half,
                    "short_ratio": half,
                    "disagreement_rate": default
                }

            total_predictions = matrix.shape[0]
            if total_predictions == 0:
                default = pd.Series(0.0, index=index)
                half = pd.Series(0.5, index=index)
                return {
                    "direction_conflict": default,
                    "long_ratio": half,
                    "short_ratio": half,
                    "disagreement_rate": default
                }

            long_votes = (matrix > 0).sum(axis=0) / total_predictions
            short_votes = (matrix < 0).sum(axis=0) / total_predictions
            disagreement_rate = 1.0 - np.maximum(long_votes, short_votes)

            return {
                "direction_conflict": pd.Series(disagreement_rate, index=index),
                "long_ratio": pd.Series(long_votes, index=index),
                "short_ratio": pd.Series(short_votes, index=index),
                "disagreement_rate": pd.Series(disagreement_rate, index=index)
            }

        except Exception as e:
            self.logger.error(f"Error calculating direction conflict: {e}")
            default = pd.Series(0.0, index=index)
            half = pd.Series(0.5, index=index)
            return {
                "direction_conflict": default,
                "long_ratio": half,
                "short_ratio": half,
                "disagreement_rate": default
            }
    
    def _calculate_confidence_gap(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
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
                zero = pd.Series(0.0, index=index)
                return {
                    "confidence_gap": zero,
                    "max_confidence": zero,
                    "second_max_confidence": zero
                }

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                zero = pd.Series(0.0, index=index)
                return {
                    "confidence_gap": zero,
                    "max_confidence": zero,
                    "second_max_confidence": zero
                }

            avg_probs = tensor.mean(axis=0)
            if avg_probs.shape[1] < 2:
                max_confidence = avg_probs[:, 0]
                second_max = np.zeros_like(max_confidence)
            else:
                sorted_probs = np.sort(avg_probs, axis=1)
                max_confidence = sorted_probs[:, -1]
                second_max = sorted_probs[:, -2]

            confidence_gap = max_confidence - second_max

            return {
                "confidence_gap": pd.Series(confidence_gap, index=index),
                "max_confidence": pd.Series(max_confidence, index=index),
                "second_max_confidence": pd.Series(second_max, index=index)
            }

        except Exception as e:
            self.logger.error(f"Error calculating confidence gap: {e}")
            zero = pd.Series(0.0, index=index)
            return {
                "confidence_gap": zero,
                "max_confidence": zero,
                "second_max_confidence": zero
            }
    
    def _calculate_entropy_uncertainty(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
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
                zero = pd.Series(0.0, index=index)
                return {"entropy": zero, "normalized_entropy": zero, "uncertainty": zero}

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                zero = pd.Series(0.0, index=index)
                return {"entropy": zero, "normalized_entropy": zero, "uncertainty": zero}

            avg_probs = tensor.mean(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                entropy = stats.entropy(avg_probs, axis=1)

            n_classes = avg_probs.shape[1] if avg_probs.ndim == 2 else 1
            max_entropy = np.log(n_classes) if n_classes > 0 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy

            series_entropy = pd.Series(entropy, index=index)
            series_normalized = pd.Series(normalized_entropy, index=index)

            return {
                "entropy": series_entropy.fillna(0.0),
                "normalized_entropy": series_normalized.fillna(0.0),
                "uncertainty": series_normalized.fillna(0.0)
            }

        except Exception as e:
            self.logger.error(f"Error calculating entropy uncertainty: {e}")
            zero = pd.Series(0.0, index=index)
            return {"entropy": zero, "normalized_entropy": zero, "uncertainty": zero}
    
    def _calculate_spread_indicators(
        self,
        model_predictions: Dict[str, np.ndarray],
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
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
            spread_features: Dict[str, pd.Series] = {}

            if model_predictions:
                matrix = self._stack_prediction_matrix(model_predictions, index)
                if matrix.size > 0:
                    pred_range = matrix.max(axis=0) - matrix.min(axis=0)
                    pred_iqr = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
                    spread_features["prediction_range"] = pd.Series(pred_range, index=index)
                    spread_features["prediction_iqr"] = pd.Series(pred_iqr, index=index)

            if model_probabilities:
                tensor = self._stack_probability_tensor(model_probabilities, index)
                if tensor.size > 0:
                    positive_class = tensor[:, :, -1]
                    prob_range = positive_class.max(axis=0) - positive_class.min(axis=0)
                    prob_iqr = np.percentile(positive_class, 75, axis=0) - np.percentile(positive_class, 25, axis=0)
                    spread_features["probability_range"] = pd.Series(prob_range, index=index)
                    spread_features["probability_iqr"] = pd.Series(prob_iqr, index=index)

            if not spread_features:
                zero = pd.Series(0.0, index=index)
                return {
                    "prediction_range": zero,
                    "prediction_iqr": zero,
                    "probability_range": zero,
                    "probability_iqr": zero
                }

            return spread_features

        except Exception as e:
            self.logger.error(f"Error calculating spread indicators: {e}")
            zero = pd.Series(0.0, index=index)
            return {
                "prediction_range": zero,
                "prediction_iqr": zero,
                "probability_range": zero,
                "probability_iqr": zero
            }
    
    def _calculate_pairwise_divergence(
        self,
        model_probabilities: Dict[str, np.ndarray],
        index: pd.Index
    ) -> Dict[str, pd.Series]:
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
                zero = pd.Series(0.0, index=index)
                return {"js_divergence": zero, "kl_divergence": zero, "avg_divergence": zero}

            tensor = self._stack_probability_tensor(model_probabilities, index)
            if tensor.size == 0:
                zero = pd.Series(0.0, index=index)
                return {"js_divergence": zero, "kl_divergence": zero, "avg_divergence": zero}

            js_values = []
            kl_values = []

            for i in range(tensor.shape[0]):
                for j in range(i + 1, tensor.shape[0]):
                    p = tensor[i]
                    q = tensor[j]
                    m = 0.5 * (p + q)
                    js = 0.5 * (stats.entropy(p, m, axis=1) + stats.entropy(q, m, axis=1))
                    kl = 0.5 * (self._calculate_kl_divergence(p, q) + self._calculate_kl_divergence(q, p))
                    js_values.append(js)
                    kl_values.append(kl)

            if not js_values:
                zero = pd.Series(0.0, index=index)
                return {"js_divergence": zero, "kl_divergence": zero, "avg_divergence": zero}

            js_avg = np.mean(js_values, axis=0)
            kl_avg = np.mean(kl_values, axis=0)
            avg_divergence = 0.5 * (js_avg + kl_avg)

            return {
                "js_divergence": pd.Series(js_avg, index=index),
                "kl_divergence": pd.Series(kl_avg, index=index),
                "avg_divergence": pd.Series(avg_divergence, index=index)
            }

        except Exception as e:
            self.logger.error(f"Error calculating pairwise divergence: {e}")
            zero = pd.Series(0.0, index=index)
            return {"js_divergence": zero, "kl_divergence": zero, "avg_divergence": zero}
    
    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Calculate KL divergence between two probability distributions.
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence value
        """
        try:
            epsilon = 1e-10
            p_safe = np.clip(p, epsilon, None)
            q_safe = np.clip(q, epsilon, None)

            kl_div = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=1)

            return kl_div

        except Exception as e:
            self.logger.error(f"Error calculating KL divergence: {e}")
            return np.zeros(p.shape[0] if p.ndim > 1 else 1)
    
    def _get_default_disagreement_features(self, index: pd.Index) -> Dict[str, pd.Series]:
        """Get default disagreement features when calculation fails."""

        zero = pd.Series(0.0, index=index)
        half = pd.Series(0.5, index=index)

        return {
            "prediction_dispersion": zero,
            "prediction_std": zero,
            "direction_conflict": zero,
            "long_ratio": half,
            "short_ratio": half,
            "disagreement_rate": zero,
            "confidence_gap": zero,
            "max_confidence": zero,
            "second_max_confidence": zero,
            "entropy": zero,
            "normalized_entropy": zero,
            "uncertainty": zero,
            "prediction_range": zero,
            "prediction_iqr": zero,
            "probability_range": zero,
            "probability_iqr": zero,
            "js_divergence": zero,
            "kl_divergence": zero,
            "avg_divergence": zero
        }
    
    def calculate_disagreement_features_for_ensemble(
        self,
        ensemble_predictions: Dict[str, Any],
        is_live: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Calculate disagreement features for an ensemble of models.
        
        Args:
            ensemble_predictions: Dict containing ensemble prediction data
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            Dict containing disagreement features
        """
        try:
            model_predictions: Dict[str, np.ndarray] = {}
            model_probabilities: Dict[str, np.ndarray] = {}
            model_confidences: Dict[str, np.ndarray] = {}
            candidate_index: Optional[pd.Index] = None

            for model_name, prediction_data in ensemble_predictions.items():
                if isinstance(prediction_data, dict):
                    if 'prediction' in prediction_data:
                        value = prediction_data['prediction']
                        if isinstance(value, pd.Series):
                            candidate_index = value.index
                            model_predictions[model_name] = value.to_numpy(dtype=float)
                        else:
                            model_predictions[model_name] = np.asarray(value, dtype=float)
                    if 'probability' in prediction_data:
                        value = prediction_data['probability']
                        if isinstance(value, pd.Series):
                            candidate_index = value.index
                            model_probabilities[model_name] = value.to_numpy(dtype=float)
                        elif isinstance(value, pd.DataFrame):
                            candidate_index = value.index
                            model_probabilities[model_name] = value.to_numpy(dtype=float)
                        else:
                            model_probabilities[model_name] = np.asarray(value, dtype=float)
                    if 'confidence' in prediction_data:
                            conf_value = prediction_data['confidence']
                            if isinstance(conf_value, pd.Series):
                                candidate_index = conf_value.index
                                model_confidences[model_name] = conf_value.to_numpy(dtype=float)
                            else:
                                model_confidences[model_name] = np.asarray(conf_value, dtype=float)
                else:
                    array_value = np.asarray(prediction_data, dtype=float)
                    model_predictions[model_name] = array_value
                    model_probabilities[model_name] = array_value

            length = self._infer_length(model_predictions, model_probabilities, model_confidences)
            index = candidate_index if candidate_index is not None and len(candidate_index) == length else pd.RangeIndex(length)

            for container in (model_predictions, model_probabilities, model_confidences):
                keys = list(container.keys())
                for key in keys:
                    arr = np.asarray(container[key], dtype=float)
                    if arr.ndim == 0:
                        container[key] = np.full(length, float(arr))
                    elif arr.ndim > 1 and arr.shape[0] != length:
                        if arr.shape[-1] == length and arr.shape[0] == 1:
                            container[key] = arr.reshape(-1)
                        else:
                            raise ValueError(
                                f"Array for {key} has incompatible shape {arr.shape} for expected length {length}"
                            )
                    elif arr.ndim == 1 and arr.shape[0] != length:
                        raise ValueError(
                            f"Array for {key} has length {arr.shape[0]} but expected {length}"
                        )
                    else:
                        container[key] = arr

            disagreement_features = self.calculate_all_disagreement_features(
                model_predictions, model_probabilities, model_confidences
            )

            # Align returned series to chosen index
            aligned_features: Dict[str, pd.Series] = {}
            for name, series in disagreement_features.items():
                aligned_features[name] = series.set_axis(index, copy=False)

            return aligned_features

        except Exception as e:
            self.logger.error(f"Error calculating disagreement features for ensemble: {e}")
            length = self._infer_length({}, {}, {})
            index = pd.RangeIndex(length)
            return self._get_default_disagreement_features(index)
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
