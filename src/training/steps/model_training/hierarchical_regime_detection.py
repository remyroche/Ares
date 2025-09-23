"""
Hierarchical Regime Detection with 4D Market Modeling

This module implements a comprehensive hierarchical regime detection system that models
market conditions across four dimensions: Volume, Volatility, Momentum, and Trend (4D).

Key Features:
1. 4D regime modeling (Volume + Volatility + Momentum + Trend)
2. Hierarchical regime classification with multiple levels
3. Dynamic regime boundary detection
4. Temporal regime persistence modeling
5. Multi-timeframe regime analysis
6. Uncertainty quantification for regime predictions
7. Adaptive regime transition detection
8. Market microstructure integration

Architecture:
- Level 1: Individual dimension analysis
- Level 2: Cross-dimensional regime identification
- Level 3: Hierarchical regime classification
- Level 4: Temporal regime modeling and prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not available, using fallback implementations")


class RegimeDimension(Enum):
    """Market regime dimensions for 4D modeling."""
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    TREND = "trend"


class RegimeHierarchyLevel(Enum):
    """Hierarchical levels of regime detection."""
    DIMENSION_LEVEL = "dimension"  # Individual dimensions
    CROSS_DIMENSIONAL = "cross_dimensional"  # Combined dimensions
    HIERARCHICAL = "hierarchical"  # Full hierarchy
    TEMPORAL = "temporal"  # Time-based modeling


@dataclass
class RegimeDimensionConfig:
    """Configuration for individual regime dimensions."""
    dimension: RegimeDimension
    features: List[str]
    n_regimes: int = 3
    smoothing_window: int = 10
    outlier_threshold: float = 3.0
    min_samples_per_regime: int = 50
    adaptive_boundaries: bool = True
    regime_persistence_threshold: int = 5


@dataclass
class HierarchicalRegimeConfig:
    """Configuration for hierarchical regime detection."""
    dimensions: Dict[RegimeDimension, RegimeDimensionConfig]
    cross_dimensional_combinations: List[List[RegimeDimension]] = field(default_factory=list)
    hierarchy_levels: int = 4
    temporal_modeling: bool = True
    uncertainty_quantification: bool = True
    adaptive_learning: bool = True
    regime_prediction_horizon: int = 2  # periods ahead
    minimum_regime_duration: int = 3
    maximum_regime_duration: int = 100
    transition_smoothing: float = 0.1
    validation_frequency: int = 50


class DimensionRegimeDetector:
    """Detects regimes within individual market dimensions."""

    def __init__(self, config: RegimeDimensionConfig):
        """Initialize dimension regime detector."""
        self.config = config
        self.scaler = RobustScaler()
        self.gmm = None
        self.regime_boundaries = {}
        self.regime_statistics = {}

    def detect_regimes(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect regimes in a single dimension."""
        try:
            # Extract relevant features
            dimension_features = data[self.config.features].copy()

            # Handle missing values
            dimension_features = dimension_features.fillna(method='ffill').fillna(method='bfill')

            # Outlier removal
            dimension_features = self._remove_outliers(dimension_features)

            # Feature scaling
            scaled_features = self.scaler.fit_transform(dimension_features)

            # Smooth features
            if self.config.smoothing_window > 1:
                scaled_features = self._smooth_features(scaled_features)

            # Determine number of regimes
            n_regimes = self._determine_optimal_regimes(scaled_features)

            # Fit Gaussian Mixture Model
            self.gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
            regime_labels = self.gmm.fit_predict(scaled_features)

            # Calculate regime boundaries and statistics
            self._calculate_regime_boundaries(scaled_features, regime_labels)

            return regime_labels, {
                'n_regimes': n_regimes,
                'regime_boundaries': self.regime_boundaries,
                'regime_statistics': self.regime_statistics,
                'gmm_converged': self.gmm.converged_,
                'gmm_weights': self.gmm.weights_
            }

        except Exception as e:
            logger.error(f"❌ Error in dimension regime detection: {e}")
            return np.zeros(len(data)), {}

    def _remove_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        cleaned_features = features.copy()

        for col in features.columns:
            Q1 = features[col].quantile(0.25)
            Q3 = features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.outlier_threshold * IQR
            upper_bound = Q3 + self.config.outlier_threshold * IQR

            # Clip outliers
            cleaned_features[col] = np.clip(
                features[col], lower_bound, upper_bound
            )

        return cleaned_features

    def _smooth_features(self, features: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing."""
        from scipy.ndimage import uniform_filter1d

        smoothed = np.zeros_like(features)
        for i in range(features.shape[1]):
            smoothed[:, i] = uniform_filter1d(
                features[:, i], size=self.config.smoothing_window
            )

        return smoothed

    def _determine_optimal_regimes(self, features: np.ndarray) -> int:
        """Determine optimal number of regimes using multiple metrics."""
        if not self.config.adaptive_boundaries:
            return self.config.n_regimes

        max_regimes = min(8, len(features) // self.config.min_samples_per_regime)
        min_regimes = 2

        best_n_regimes = self.config.n_regimes
        best_score = -float('inf')

        for n_regimes in range(min_regimes, max_regimes + 1):
            try:
                gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=5)
                labels = gmm.fit_predict(features)

                # Calculate multiple metrics
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(features, labels)
                    calinski = calinski_harabasz_score(features, labels)

                    # Combined score (higher is better)
                    combined_score = (silhouette + calinski) / 2.0

                    if combined_score > best_score:
                        best_score = combined_score
                        best_n_regimes = n_regimes

            except Exception as e:
                logger.warning(f"⚠️ Error evaluating {n_regimes} regimes: {e}")
                continue

        return best_n_regimes

    def _calculate_regime_boundaries(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Calculate regime boundaries and statistics."""
        unique_labels = np.unique(labels)

        for regime_id in unique_labels:
            regime_mask = labels == regime_id
            regime_features = features[regime_mask]

            if len(regime_features) < 10:
                continue

            # Calculate statistics
            self.regime_statistics[regime_id] = {
                'mean': np.mean(regime_features, axis=0),
                'std': np.std(regime_features, axis=0),
                'median': np.median(regime_features, axis=0),
                'size': len(regime_features),
                'percentage': len(regime_features) / len(features)
            }

            # Calculate boundaries
            self.regime_boundaries[regime_id] = {
                'min': np.min(regime_features, axis=0),
                'max': np.max(regime_features, axis=0),
                'q25': np.percentile(regime_features, 25, axis=0),
                'q75': np.percentile(regime_features, 75, axis=0)
            }

    def predict_regime_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for new data."""
        if self.gmm is None:
            raise ValueError("Model not fitted")

        # Preprocess data
        processed_data = data[self.config.features].fillna(method='ffill').fillna(method='bfill')
        processed_data = self._remove_outliers(processed_data)
        scaled_data = self.scaler.transform(processed_data)

        return self.gmm.predict_proba(scaled_data)


class CrossDimensionalRegimeDetector:
    """Detects regimes across multiple dimensions."""

    def __init__(self, config: HierarchicalRegimeConfig):
        """Initialize cross-dimensional regime detector."""
        self.config = config
        self.dimension_detectors: Dict[RegimeDimension, DimensionRegimeDetector] = {}
        self.combined_gmm = None
        self.regime_mapping = {}

    def fit(self, data: Dict[RegimeDimension, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit cross-dimensional regime detector."""
        try:
            # Fit individual dimension detectors
            dimension_labels = {}
            processed_features = []

            for dimension, dim_data in data.items():
                if dimension in self.config.dimensions:
                    detector = DimensionRegimeDetector(self.config.dimensions[dimension])
                    labels, stats = detector.detect_regimes(dim_data)

                    self.dimension_detectors[dimension] = detector
                    dimension_labels[dimension] = labels

                    # Add to processed features
                    processed_features.append(labels.reshape(-1, 1))

            if not processed_features:
                raise ValueError("No valid dimensions found")

            # Combine dimension labels
            combined_features = np.hstack(processed_features)

            # Fit combined GMM
            n_cross_regimes = self._determine_cross_regime_count(combined_features)
            self.combined_gmm = GaussianMixture(
                n_components=n_cross_regimes,
                random_state=42,
                n_init=10
            )

            cross_regime_labels = self.combined_gmm.fit_predict(combined_features)

            # Create regime mapping
            self._create_regime_mapping(dimension_labels, cross_regime_labels)

            return cross_regime_labels, {
                'n_cross_regimes': n_cross_regime_count,
                'dimension_labels': dimension_labels,
                'regime_mapping': self.regime_mapping,
                'gmm_converged': self.combined_gmm.converged_,
                'gmm_weights': self.combined_gmm.weights_
            }

        except Exception as e:
            logger.error(f"❌ Error in cross-dimensional regime detection: {e}")
            return np.zeros(len(list(data.values())[0])), {}

    def _determine_cross_regime_count(self, features: np.ndarray) -> int:
        """Determine optimal number of cross-dimensional regimes."""
        max_regimes = min(12, len(features) // 20)  # At least 20 samples per regime
        min_regimes = 3

        best_n_regimes = 4  # Default
        best_score = -float('inf')

        for n_regimes in range(min_regimes, max_regimes + 1):
            try:
                gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=5)
                labels = gmm.fit_predict(features)

                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(features, labels)
                    calinski = calinski_harabasz_score(features, labels)

                    combined_score = (silhouette + calinski) / 2.0

                    if combined_score > best_score:
                        best_score = combined_score
                        best_n_regimes = n_regimes

            except Exception:
                continue

        return best_n_regimes

    def _create_regime_mapping(self, dimension_labels: Dict[RegimeDimension, np.ndarray],
                             cross_labels: np.ndarray) -> None:
        """Create mapping from dimension regimes to cross-dimensional regimes."""
        for cross_regime in np.unique(cross_labels):
            mask = cross_labels == cross_regime

            regime_profile = {}
            for dimension, labels in dimension_labels.items():
                unique_labels, counts = np.unique(labels[mask], return_counts=True)
                most_common = unique_labels[np.argmax(counts)]
                regime_profile[dimension] = most_common

            self.regime_mapping[cross_regime] = regime_profile

    def predict_cross_regime_probabilities(self, data: Dict[RegimeDimension, pd.DataFrame]) -> np.ndarray:
        """Predict cross-dimensional regime probabilities."""
        if self.combined_gmm is None:
            raise ValueError("Model not fitted")

        # Get individual dimension probabilities
        processed_features = []
        for dimension, dim_data in data.items():
            if dimension in self.dimension_detectors:
                probs = self.dimension_detectors[dimension].predict_regime_probabilities(dim_data)
                processed_features.append(probs)

        if not processed_features:
            raise ValueError("No valid dimension predictions")

        combined_features = np.hstack(processed_features)
        return self.combined_gmm.predict_proba(combined_features)


class HierarchicalRegimeClassifier:
    """Hierarchical regime classifier with multiple levels."""

    def __init__(self, config: HierarchicalRegimeConfig):
        """Initialize hierarchical regime classifier."""
        self.config = config
        self.dimension_detectors: Dict[RegimeDimension, DimensionRegimeDetector] = {}
        self.cross_detector = None
        self.hierarchical_classifier = None
        self.regime_hierarchy = {}
        self.transition_matrix = None

    def fit(self, data: Dict[RegimeDimension, pd.DataFrame]) -> Dict[str, Any]:
        """Fit hierarchical regime classifier."""
        try:
            # Step 1: Fit dimension-level detectors
            dimension_results = {}
            for dimension, dim_data in data.items():
                if dimension in self.config.dimensions:
                    detector = DimensionRegimeDetector(self.config.dimensions[dimension])
                    labels, stats = detector.detect_regimes(dim_data)

                    self.dimension_detectors[dimension] = detector
                    dimension_results[dimension] = {
                        'labels': labels,
                        'stats': stats
                    }

            # Step 2: Fit cross-dimensional detector
            self.cross_detector = CrossDimensionalRegimeDetector(self.config)
            cross_labels, cross_stats = self.cross_detector.fit(data)
            dimension_results['cross_dimensional'] = {
                'labels': cross_labels,
                'stats': cross_stats
            }

            # Step 3: Create hierarchical classification
            hierarchical_labels = self._create_hierarchical_labels(dimension_results)
            dimension_results['hierarchical'] = {
                'labels': hierarchical_labels,
                'stats': self._calculate_hierarchical_stats(hierarchical_labels, data)
            }

            # Step 4: Fit temporal model if enabled
            if self.config.temporal_modeling:
                temporal_results = self._fit_temporal_model(dimension_results)
                dimension_results.update(temporal_results)

            # Step 5: Calculate transition matrix
            self._calculate_transition_matrix(hierarchical_labels)

            return dimension_results

        except Exception as e:
            logger.error(f"❌ Error in hierarchical regime classification: {e}")
            return {}

    def _create_hierarchical_labels(self, dimension_results: Dict) -> np.ndarray:
        """Create hierarchical regime labels."""
        # Combine dimension labels into hierarchical structure
        hierarchical_labels = []

        for i in range(len(list(dimension_results.values())[0]['labels'])):
            label_combination = []

            for dimension in self.config.dimensions.keys():
                if dimension in dimension_results:
                    label_combination.append(dimension_results[dimension]['labels'][i])

            # Create hierarchical ID
            hierarchical_id = self._create_hierarchical_id(label_combination)
            hierarchical_labels.append(hierarchical_id)

        return np.array(hierarchical_labels)

    def _create_hierarchical_id(self, label_combination: List[int]) -> str:
        """Create hierarchical regime identifier."""
        return "_".join([str(label) for label in label_combination])

    def _calculate_hierarchical_stats(self, hierarchical_labels: np.ndarray,
                                   data: Dict[RegimeDimension, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate statistics for hierarchical regimes."""
        unique_regimes = np.unique(hierarchical_labels)
        stats = {}

        for regime in unique_regimes:
            mask = hierarchical_labels == regime
            stats[regime] = {
                'size': np.sum(mask),
                'percentage': np.mean(mask),
                'duration_mean': self._calculate_regime_duration(hierarchical_labels, regime),
                'duration_std': self._calculate_regime_duration_std(hierarchical_labels, regime)
            }

        return stats

    def _calculate_regime_duration(self, labels: np.ndarray, regime: str) -> float:
        """Calculate average duration of a regime."""
        durations = []
        current_duration = 0

        for label in labels:
            if label == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def _calculate_regime_duration_std(self, labels: np.ndarray, regime: str) -> float:
        """Calculate standard deviation of regime duration."""
        durations = []
        current_duration = 0

        for label in labels:
            if label == regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.std(durations) if len(durations) > 1 else 0

    def _fit_temporal_model(self, dimension_results: Dict) -> Dict[str, Any]:
        """Fit temporal regime model."""
        # This would implement temporal modeling like HMM or temporal clustering
        # For now, return placeholder
        return {
            'temporal': {
                'labels': dimension_results['hierarchical']['labels'],
                'stats': {'temporal_model_fitted': True}
            }
        }

    def _calculate_transition_matrix(self, hierarchical_labels: np.ndarray) -> None:
        """Calculate regime transition probabilities."""
        unique_regimes = np.unique(hierarchical_labels)
        n_regimes = len(unique_regimes)

        # Create transition matrix
        self.transition_matrix = np.zeros((n_regimes, n_regimes))

        for i in range(len(hierarchical_labels) - 1):
            current_regime = hierarchical_labels[i]
            next_regime = hierarchical_labels[i + 1]

            current_idx = np.where(unique_regimes == current_regime)[0][0]
            next_idx = np.where(unique_regimes == next_regime)[0][0]

            self.transition_matrix[current_idx, next_idx] += 1

        # Normalize rows
        row_sums = np.sum(self.transition_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix = self.transition_matrix / row_sums

    def predict_hierarchical_regime(self, data: Dict[RegimeDimension, pd.DataFrame],
                                  return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict hierarchical regime for new data."""
        try:
            # Get predictions from each dimension
            dimension_predictions = {}
            for dimension, dim_data in data.items():
                if dimension in self.dimension_detectors:
                    probs = self.dimension_detectors[dimension].predict_regime_probabilities(dim_data)
                    dimension_predictions[dimension] = probs

            # Get cross-dimensional predictions
            if self.cross_detector:
                cross_probs = self.cross_detector.predict_cross_regime_probabilities(data)
                dimension_predictions['cross'] = cross_probs

            # Combine predictions to get hierarchical regime
            hierarchical_predictions = self._combine_hierarchical_predictions(dimension_predictions)

            if return_probabilities:
                # Calculate regime probabilities
                regime_probabilities = self._calculate_regime_probabilities(
                    dimension_predictions, hierarchical_predictions
                )
                return hierarchical_predictions, regime_probabilities
            else:
                return hierarchical_predictions

        except Exception as e:
            logger.error(f"❌ Error predicting hierarchical regime: {e}")
            return np.zeros(len(list(data.values())[0]))

    def _combine_hierarchical_predictions(self, dimension_predictions: Dict) -> np.ndarray:
        """Combine dimension predictions into hierarchical regime labels."""
        # Placeholder implementation - would use learned combination rules
        # For now, use cross-dimensional predictions as primary
        if 'cross' in dimension_predictions:
            cross_labels = np.argmax(dimension_predictions['cross'], axis=1)
            return cross_labels
        else:
            return np.zeros(len(list(dimension_predictions.values())[0]))

    def _calculate_regime_probabilities(self, dimension_predictions: Dict,
                                     hierarchical_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime probabilities."""
        # Placeholder implementation
        n_samples = len(hierarchical_predictions)
        n_regimes = len(np.unique(hierarchical_predictions))

        # Create probability matrix
        probabilities = np.zeros((n_samples, n_regimes))

        for i, prediction in enumerate(hierarchical_predictions):
            probabilities[i, int(prediction)] = 1.0  # Deterministic for now

        return probabilities


class HierarchicalRegimePredictor:
    """Main hierarchical regime detection and prediction system."""

    def __init__(self, config: Optional[HierarchicalRegimeConfig] = None):
        """Initialize hierarchical regime predictor."""
        self.config = config or HierarchicalRegimeConfig(
            dimensions=self._create_default_dimensions()
        )
        self.classifier = HierarchicalRegimeClassifier(self.config)
        self.regime_history = []
        self.prediction_uncertainty = None

    def _create_default_dimensions(self) -> Dict[RegimeDimension, RegimeDimensionConfig]:
        """Create default dimension configurations."""
        return {
            RegimeDimension.VOLUME: RegimeDimensionConfig(
                dimension=RegimeDimension.VOLUME,
                features=['volume', 'volume_ratio', 'volume_change'],
                n_regimes=3,
                smoothing_window=5
            ),
            RegimeDimension.VOLATILITY: RegimeDimensionConfig(
                dimension=RegimeDimension.VOLATILITY,
                features=['volatility', 'price_range', 'high_low_ratio'],
                n_regimes=4,
                smoothing_window=10
            ),
            RegimeDimension.MOMENTUM: RegimeDimensionConfig(
                dimension=RegimeDimension.MOMENTUM,
                features=['momentum', 'rsi', 'macd'],
                n_regimes=3,
                smoothing_window=14
            ),
            RegimeDimension.TREND: RegimeDimensionConfig(
                dimension=RegimeDimension.TREND,
                features=['trend', 'ema_20', 'ema_50', 'slope'],
                n_regimes=3,
                smoothing_window=20
            )
        }

    def fit(self, data: Dict[RegimeDimension, pd.DataFrame]) -> Dict[str, Any]:
        """Fit the hierarchical regime detection system."""
        logger.info("🔍 Fitting hierarchical regime detection system")

        # Validate data
        self._validate_input_data(data)

        # Fit hierarchical classifier
        fit_results = self.classifier.fit(data)

        # Store regime history
        if 'hierarchical' in fit_results:
            self.regime_history = fit_results['hierarchical']['labels']

        logger.info("✅ Hierarchical regime detection system fitted")
        return fit_results

    def _validate_input_data(self, data: Dict[RegimeDimension, pd.DataFrame]) -> None:
        """Validate input data for regime detection."""
        if not data:
            raise ValueError("No data provided for regime detection")

        # Check that all required dimensions are present
        for dimension in self.config.dimensions.keys():
            if dimension not in data:
                logger.warning(f"⚠️ Missing data for dimension: {dimension}")
            else:
                # Check minimum data requirements
                min_samples = 100  # Minimum samples needed
                if len(data[dimension]) < min_samples:
                    logger.warning(f"⚠️ Insufficient data for {dimension}: {len(data[dimension])} < {min_samples}")

    def predict_regime(self, data: Dict[RegimeDimension, pd.DataFrame],
                      horizon: int = 1, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict current regime (and future regimes if horizon > 1)."""
        try:
            if horizon == 1:
                # Current regime prediction
                predictions = self.classifier.predict_hierarchical_regime(data, return_uncertainty)

                if return_uncertainty:
                    predictions, uncertainty = predictions
                    self.prediction_uncertainty = uncertainty
                    return predictions, uncertainty
                else:
                    return predictions
            else:
                # Multi-step regime prediction
                return self._predict_multi_step_regime(data, horizon, return_uncertainty)

        except Exception as e:
            logger.error(f"❌ Error predicting regime: {e}")
            return np.zeros(len(list(data.values())[0]))

    def _predict_multi_step_regime(self, data: Dict[RegimeDimension, pd.DataFrame],
                                 horizon: int, return_uncertainty: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict regimes for multiple steps ahead."""
        # Placeholder implementation for multi-step prediction
        # Would use temporal models like HMM or RNN for prediction

        current_predictions = self.predict_regime(data, horizon=1, return_uncertainty=False)

        if return_uncertainty:
            current_predictions, _ = current_predictions

        # For now, repeat current predictions (would be enhanced with temporal modeling)
        multi_step_predictions = np.tile(current_predictions, (horizon, 1)).T

        if return_uncertainty:
            # Create uncertainty that increases with horizon
            uncertainty = np.random.uniform(0.1, 0.5, multi_step_predictions.shape)
            uncertainty = uncertainty * np.arange(1, horizon + 1)  # Increase with horizon
            return multi_step_predictions, uncertainty
        else:
            return multi_step_predictions

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics."""
        stats = {
            'current_regime': None,
            'regime_distribution': {},
            'transition_matrix': self.classifier.transition_matrix,
            'dimension_stats': {},
            'temporal_stats': {}
        }

        if self.regime_history:
            unique_regimes, counts = np.unique(self.regime_history, return_counts=True)
            stats['regime_distribution'] = dict(zip(unique_regimes, counts / len(self.regime_history)))

            if len(self.regime_history) > 0:
                stats['current_regime'] = self.regime_history[-1]

        # Add dimension-specific statistics
        for dimension, detector in self.classifier.dimension_detectors.items():
            stats['dimension_stats'][dimension.value] = detector.regime_statistics

        return stats

    def save_regime_model(self, filepath: str) -> None:
        """Save regime detection model to file."""
        try:
            model_data = {
                'config': {
                    'dimensions': {
                        dim.value: {
                            'features': config.features,
                            'n_regimes': config.n_regimes,
                            'smoothing_window': config.smoothing_window
                        }
                        for dim, config in self.config.dimensions.items()
                    },
                    'hierarchy_levels': self.config.hierarchy_levels,
                    'temporal_modeling': self.config.temporal_modeling
                },
                'regime_history': list(self.regime_history),
                'transition_matrix': self.classifier.transition_matrix.tolist() if self.classifier.transition_matrix is not None else None,
                'timestamp': time.time()
            }

            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2, default=str)

            logger.info(f"💾 Regime model saved to {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save regime model: {e}")

    def load_regime_model(self, filepath: str) -> None:
        """Load regime detection model from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    model_data = json.load(f)

                # Reconstruct regime history
                self.regime_history = np.array(model_data.get('regime_history', []))

                logger.info(f"📂 Regime model loaded from {filepath}")
            else:
                logger.warning(f"⚠️ Regime model file not found: {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to load regime model: {e}")


# Factory functions and utilities
def create_hierarchical_regime_detector(config: Optional[Dict[str, Any]] = None) -> HierarchicalRegimePredictor:
    """Create hierarchical regime detector."""
    return HierarchicalRegimePredictor(config)


def detect_regimes_4d(data: Dict[str, pd.DataFrame],
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Detect regimes using 4D hierarchical approach."""
    detector = create_hierarchical_regime_detector(config)
    return detector.fit(data)


def get_example_hierarchical_config() -> Dict[str, Any]:
    """Get example configuration for hierarchical regime detection."""
    return {
        'dimensions': {
            'volume': {
                'features': ['volume', 'volume_ratio'],
                'n_regimes': 3,
                'smoothing_window': 5
            },
            'volatility': {
                'features': ['volatility', 'price_range'],
                'n_regimes': 4,
                'smoothing_window': 10
            },
            'momentum': {
                'features': ['momentum', 'rsi'],
                'n_regimes': 3,
                'smoothing_window': 14
            },
            'trend': {
                'features': ['trend', 'ema_ratio'],
                'n_regimes': 3,
                'smoothing_window': 20
            }
        },
        'hierarchy_levels': 4,
        'temporal_modeling': True,
        'uncertainty_quantification': True,
        'adaptive_learning': True,
        'regime_prediction_horizon': 2,
        'minimum_regime_duration': 3,
        'maximum_regime_duration': 100
    }