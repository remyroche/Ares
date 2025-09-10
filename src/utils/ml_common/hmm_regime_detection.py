"""
HMM Regime Detection Utilities

This module provides comprehensive HMM-based regime detection and analysis utilities,
extending the existing HMM composite manager with additional regime detection capabilities.

Key Features:
- HMM model training and optimization
- Regime state identification and validation
- Regime transition probability analysis
- Multi-timeframe HMM ensemble logic
- Regime quality assessment and metrics
- Regime continuity validation
- Cross-regime data validation
- Performance tracking and analytics
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass, field
from enum import Enum

from ..math_validation import safe_divide, safe_log, safe_sqrt, validate_positive, validate_range
from ..common_operations import create_fallback_logger
from ..m1_gpu_utils import M1GPUManager
from ..parallel_processing_optimizer import ParallelProcessor
from ..hmm_composite_manager import HMMCompositeManager

logger = logging.getLogger(__name__)

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("HMM libraries not available - limited regime detection functionality")

class RegimeDetectionMethod(Enum):
    """Available regime detection methods."""
    HMM_GAUSSIAN = "hmm_gaussian"
    HMM_MIXTURE = "hmm_mixture"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    KMEANS = "kmeans"
    ENSEMBLE = "ensemble"

@dataclass
class HMMRegimeConfig:
    """Configuration for HMM regime detection."""
    n_regimes: int = 3
    method: RegimeDetectionMethod = RegimeDetectionMethod.HMM_GAUSSIAN
    n_iterations: int = 100
    tolerance: float = 1e-6
    random_state: int = 42
    min_regime_duration: int = 5
    max_regime_duration: int = 1000
    transition_threshold: float = 0.1
    stability_threshold: float = 0.8
    ensemble_methods: List[RegimeDetectionMethod] = field(default_factory=lambda: [
        RegimeDetectionMethod.HMM_GAUSSIAN,
        RegimeDetectionMethod.GAUSSIAN_MIXTURE
    ])
    multi_timeframe: bool = True
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h'])
    parallel_processing: bool = True
    max_workers: int = 4

@dataclass
class RegimeDetectionResult:
    """Result of regime detection operation."""
    regime_ids: np.ndarray
    regime_probabilities: np.ndarray
    transition_matrix: np.ndarray
    regime_means: np.ndarray
    regime_covariances: np.ndarray
    regime_qualities: Dict[str, float]
    method_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class HMMRegimeDetector:
    """
    Enhanced HMM regime detection utilities.
    
    This class extends the existing HMM composite manager with additional
    regime detection capabilities, quality assessment, and multi-timeframe analysis.
    """
    
    def __init__(self, config: Optional[HMMRegimeConfig] = None):
        """Initialize the HMM regime detector."""
        self.config = config or HMMRegimeConfig()
        self.logger = logger.getChild('HMMRegimeDetector')
        
        # Initialize components
        self.gpu_manager = M1GPUManager()
        self.parallel_processor = ParallelProcessor(max_workers=self.config.max_workers)
        self.hmm_manager = HMMCompositeManager()
        
        # Validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the regime detection configuration."""
        validate_positive(self.config.n_regimes, "n_regimes")
        validate_positive(self.config.n_iterations, "n_iterations")
        validate_positive(self.config.tolerance, "tolerance")
        validate_positive(self.config.min_regime_duration, "min_regime_duration")
        validate_positive(self.config.max_regime_duration, "max_regime_duration")
        validate_range(self.config.transition_threshold, 0.0, 1.0, "transition_threshold")
        validate_range(self.config.stability_threshold, 0.0, 1.0, "stability_threshold")
        
        if not HMM_AVAILABLE and self.config.method in [RegimeDetectionMethod.HMM_GAUSSIAN, RegimeDetectionMethod.HMM_MIXTURE]:
            self.logger.warning("HMM libraries not available, falling back to Gaussian Mixture")
            self.config.method = RegimeDetectionMethod.GAUSSIAN_MIXTURE
    
    def detect_regimes(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        method: Optional[RegimeDetectionMethod] = None
    ) -> RegimeDetectionResult:
        """
        Detect regimes in the data using the specified method.
        
        Args:
            data: Input data DataFrame
            features: List of feature columns to use for regime detection
            method: Optional method override
            
        Returns:
            RegimeDetectionResult with regime information
        """
        method = method or self.config.method
        self.logger.info(f"Detecting regimes using method: {method.value}")
        
        # Prepare features
        if features is None:
            features = self._select_features(data)
        
        feature_data = data[features].values
        
        try:
            if method == RegimeDetectionMethod.HMM_GAUSSIAN:
                return self._hmm_gaussian_regime_detection(feature_data, data)
            elif method == RegimeDetectionMethod.HMM_MIXTURE:
                return self._hmm_mixture_regime_detection(feature_data, data)
            elif method == RegimeDetectionMethod.GAUSSIAN_MIXTURE:
                return self._gaussian_mixture_regime_detection(feature_data, data)
            elif method == RegimeDetectionMethod.KMEANS:
                return self._kmeans_regime_detection(feature_data, data)
            elif method == RegimeDetectionMethod.ENSEMBLE:
                return self._ensemble_regime_detection(feature_data, data)
            else:
                raise ValueError(f"Unsupported regime detection method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            raise
    
    def _select_features(self, data: pd.DataFrame) -> List[str]:
        """Select appropriate features for regime detection."""
        # Default feature selection logic
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp and target columns
        exclude_columns = ['timestamp', 'target', 'label', 'regime']
        features = [col for col in numeric_columns if col not in exclude_columns]
        
        # Limit to most relevant features
        if len(features) > 20:
            # Select features with highest variance
            feature_vars = data[features].var()
            features = feature_vars.nlargest(20).index.tolist()
        
        self.logger.info(f"Selected {len(features)} features for regime detection")
        return features
    
    def _hmm_gaussian_regime_detection(self, feature_data: np.ndarray, data: pd.DataFrame) -> RegimeDetectionResult:
        """Perform HMM Gaussian regime detection."""
        self.logger.info("Performing HMM Gaussian regime detection")
        
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available for Gaussian HMM detection")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create and train HMM
        model = hmm.GaussianHMM(
            n_components=self.config.n_regimes,
            n_iter=self.config.n_iterations,
            tol=self.config.tolerance,
            random_state=self.config.random_state
        )
        
        model.fit(scaled_data)
        
        # Get regime predictions
        regime_ids = model.predict(scaled_data)
        regime_probabilities = model.predict_proba(scaled_data)
        
        # Get model parameters
        transition_matrix = model.transmat_
        regime_means = model.means_
        regime_covariances = model.covars_
        
        # Calculate regime qualities
        regime_qualities = self._calculate_regime_qualities(regime_ids, regime_probabilities, data)
        
        # Create metadata
        metadata = {
            'method': 'hmm_gaussian',
            'n_regimes': self.config.n_regimes,
            'n_iterations': self.config.n_iterations,
            'tolerance': self.config.tolerance,
            'features_used': len(feature_data[0]) if len(feature_data) > 0 else 0,
            'model_score': model.score(scaled_data),
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        }
        
        return RegimeDetectionResult(
            regime_ids=regime_ids,
            regime_probabilities=regime_probabilities,
            transition_matrix=transition_matrix,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            regime_qualities=regime_qualities,
            method_used='hmm_gaussian',
            metadata=metadata
        )
    
    def _hmm_mixture_regime_detection(self, feature_data: np.ndarray, data: pd.DataFrame) -> RegimeDetectionResult:
        """Perform HMM mixture regime detection."""
        self.logger.info("Performing HMM mixture regime detection")
        
        if not HMM_AVAILABLE:
            raise ImportError("HMM libraries not available for mixture HMM detection")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create and train mixture HMM
        model = hmm.GMMHMM(
            n_components=self.config.n_regimes,
            n_mix=self.config.n_regimes,  # Number of mixture components per state
            n_iter=self.config.n_iterations,
            tol=self.config.tolerance,
            random_state=self.config.random_state
        )
        
        model.fit(scaled_data)
        
        # Get regime predictions
        regime_ids = model.predict(scaled_data)
        regime_probabilities = model.predict_proba(scaled_data)
        
        # Get model parameters
        transition_matrix = model.transmat_
        regime_means = model.means_
        regime_covariances = model.covars_
        
        # Calculate regime qualities
        regime_qualities = self._calculate_regime_qualities(regime_ids, regime_probabilities, data)
        
        # Create metadata
        metadata = {
            'method': 'hmm_mixture',
            'n_regimes': self.config.n_regimes,
            'n_iterations': self.config.n_iterations,
            'tolerance': self.config.tolerance,
            'features_used': len(feature_data[0]) if len(feature_data) > 0 else 0,
            'model_score': model.score(scaled_data),
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        }
        
        return RegimeDetectionResult(
            regime_ids=regime_ids,
            regime_probabilities=regime_probabilities,
            transition_matrix=transition_matrix,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            regime_qualities=regime_qualities,
            method_used='hmm_mixture',
            metadata=metadata
        )
    
    def _gaussian_mixture_regime_detection(self, feature_data: np.ndarray, data: pd.DataFrame) -> RegimeDetectionResult:
        """Perform Gaussian mixture regime detection."""
        self.logger.info("Performing Gaussian mixture regime detection")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create and train Gaussian mixture model
        model = GaussianMixture(
            n_components=self.config.n_regimes,
            max_iter=self.config.n_iterations,
            tol=self.config.tolerance,
            random_state=self.config.random_state
        )
        
        model.fit(scaled_data)
        
        # Get regime predictions
        regime_ids = model.predict(scaled_data)
        regime_probabilities = model.predict_proba(scaled_data)
        
        # Create transition matrix (simplified)
        transition_matrix = self._estimate_transition_matrix(regime_ids)
        
        # Get model parameters
        regime_means = model.means_
        regime_covariances = model.covariances_
        
        # Calculate regime qualities
        regime_qualities = self._calculate_regime_qualities(regime_ids, regime_probabilities, data)
        
        # Create metadata
        metadata = {
            'method': 'gaussian_mixture',
            'n_regimes': self.config.n_regimes,
            'n_iterations': self.config.n_iterations,
            'tolerance': self.config.tolerance,
            'features_used': len(feature_data[0]) if len(feature_data) > 0 else 0,
            'model_score': model.score(scaled_data),
            'aic': model.aic(scaled_data),
            'bic': model.bic(scaled_data),
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        }
        
        return RegimeDetectionResult(
            regime_ids=regime_ids,
            regime_probabilities=regime_probabilities,
            transition_matrix=transition_matrix,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            regime_qualities=regime_qualities,
            method_used='gaussian_mixture',
            metadata=metadata
        )
    
    def _kmeans_regime_detection(self, feature_data: np.ndarray, data: pd.DataFrame) -> RegimeDetectionResult:
        """Perform K-means regime detection."""
        self.logger.info("Performing K-means regime detection")
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create and train K-means model
        model = KMeans(
            n_clusters=self.config.n_regimes,
            max_iter=self.config.n_iterations,
            tol=self.config.tolerance,
            random_state=self.config.random_state
        )
        
        model.fit(scaled_data)
        
        # Get regime predictions
        regime_ids = model.labels_
        
        # Create regime probabilities (distance-based)
        distances = model.transform(scaled_data)
        regime_probabilities = self._distances_to_probabilities(distances)
        
        # Create transition matrix
        transition_matrix = self._estimate_transition_matrix(regime_ids)
        
        # Get model parameters
        regime_means = model.cluster_centers_
        regime_covariances = self._estimate_covariances(scaled_data, regime_ids, regime_means)
        
        # Calculate regime qualities
        regime_qualities = self._calculate_regime_qualities(regime_ids, regime_probabilities, data)
        
        # Create metadata
        metadata = {
            'method': 'kmeans',
            'n_regimes': self.config.n_regimes,
            'n_iterations': self.config.n_iterations,
            'tolerance': self.config.tolerance,
            'features_used': len(feature_data[0]) if len(feature_data) > 0 else 0,
            'inertia': model.inertia_,
            'silhouette_score': silhouette_score(scaled_data, regime_ids),
            'scaler_params': {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        }
        
        return RegimeDetectionResult(
            regime_ids=regime_ids,
            regime_probabilities=regime_probabilities,
            transition_matrix=transition_matrix,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            regime_qualities=regime_qualities,
            method_used='kmeans',
            metadata=metadata
        )
    
    def _ensemble_regime_detection(self, feature_data: np.ndarray, data: pd.DataFrame) -> RegimeDetectionResult:
        """Perform ensemble regime detection using multiple methods."""
        self.logger.info("Performing ensemble regime detection")
        
        # Get results from multiple methods
        ensemble_results = []
        methods = self.config.ensemble_methods
        
        for method in methods:
            try:
                result = self.detect_regimes(data, method=method)
                ensemble_results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in ensemble method {method}: {e}")
                continue
        
        if not ensemble_results:
            raise ValueError("No ensemble methods succeeded")
        
        # Combine results using voting
        combined_regime_ids = self._combine_regime_predictions(ensemble_results)
        combined_probabilities = self._combine_regime_probabilities(ensemble_results)
        combined_transition_matrix = self._combine_transition_matrices(ensemble_results)
        
        # Calculate average regime parameters
        regime_means = np.mean([r.regime_means for r in ensemble_results], axis=0)
        regime_covariances = np.mean([r.regime_covariances for r in ensemble_results], axis=0)
        
        # Calculate regime qualities
        regime_qualities = self._calculate_regime_qualities(combined_regime_ids, combined_probabilities, data)
        
        # Create metadata
        metadata = {
            'method': 'ensemble',
            'ensemble_methods': [r.method_used for r in ensemble_results],
            'n_regimes': self.config.n_regimes,
            'features_used': len(feature_data[0]) if len(feature_data) > 0 else 0,
            'ensemble_scores': [r.metadata.get('model_score', 0) for r in ensemble_results],
            'individual_results': [r.metadata for r in ensemble_results]
        }
        
        return RegimeDetectionResult(
            regime_ids=combined_regime_ids,
            regime_probabilities=combined_probabilities,
            transition_matrix=combined_transition_matrix,
            regime_means=regime_means,
            regime_covariances=regime_covariances,
            regime_qualities=regime_qualities,
            method_used='ensemble',
            metadata=metadata
        )
    
    def _calculate_regime_qualities(self, regime_ids: np.ndarray, regime_probabilities: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for detected regimes."""
        qualities = {}
        
        # Regime stability (how consistent the regime assignments are)
        regime_stability = np.mean(np.max(regime_probabilities, axis=1))
        qualities['stability'] = regime_stability
        
        # Regime duration analysis
        regime_durations = self._calculate_regime_durations(regime_ids)
        qualities['avg_duration'] = np.mean(regime_durations)
        qualities['duration_std'] = np.std(regime_durations)
        
        # Regime balance (how evenly distributed regimes are)
        unique_regimes, counts = np.unique(regime_ids, return_counts=True)
        regime_balance = 1 - (np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0
        qualities['balance'] = regime_balance
        
        # Transition smoothness
        transition_smoothness = self._calculate_transition_smoothness(regime_ids)
        qualities['transition_smoothness'] = transition_smoothness
        
        # Regime separation (how distinct regimes are)
        if 'close' in data.columns:
            regime_separation = self._calculate_regime_separation(regime_ids, data['close'].values)
            qualities['separation'] = regime_separation
        
        return qualities
    
    def _calculate_regime_durations(self, regime_ids: np.ndarray) -> List[int]:
        """Calculate durations of each regime."""
        durations = []
        current_regime = regime_ids[0]
        current_duration = 1
        
        for i in range(1, len(regime_ids)):
            if regime_ids[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regime_ids[i]
                current_duration = 1
        
        durations.append(current_duration)
        return durations
    
    def _calculate_transition_smoothness(self, regime_ids: np.ndarray) -> float:
        """Calculate how smooth regime transitions are."""
        transitions = 0
        for i in range(1, len(regime_ids)):
            if regime_ids[i] != regime_ids[i-1]:
                transitions += 1
        
        return 1 - (transitions / len(regime_ids))
    
    def _calculate_regime_separation(self, regime_ids: np.ndarray, prices: np.ndarray) -> float:
        """Calculate how well separated regimes are in terms of price behavior."""
        unique_regimes = np.unique(regime_ids)
        regime_means = []
        
        for regime in unique_regimes:
            regime_prices = prices[regime_ids == regime]
            regime_means.append(np.mean(regime_prices))
        
        if len(regime_means) < 2:
            return 0.0
        
        # Calculate coefficient of variation of regime means
        regime_means = np.array(regime_means)
        separation = np.std(regime_means) / np.mean(regime_means) if np.mean(regime_means) > 0 else 0
        return separation
    
    def _estimate_transition_matrix(self, regime_ids: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from regime sequence."""
        n_regimes = len(np.unique(regime_ids))
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_ids) - 1):
            current_regime = regime_ids[i]
            next_regime = regime_ids[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1)
        for i in range(n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i] /= row_sums[i]
            else:
                transition_matrix[i] = 1.0 / n_regimes
        
        return transition_matrix
    
    def _distances_to_probabilities(self, distances: np.ndarray) -> np.ndarray:
        """Convert distances to probabilities using softmax."""
        # Use negative distances as logits
        logits = -distances
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities
    
    def _estimate_covariances(self, data: np.ndarray, regime_ids: np.ndarray, regime_means: np.ndarray) -> np.ndarray:
        """Estimate covariances for each regime."""
        n_regimes = len(regime_means)
        n_features = data.shape[1]
        covariances = np.zeros((n_regimes, n_features, n_features))
        
        for regime in range(n_regimes):
            regime_data = data[regime_ids == regime]
            if len(regime_data) > 1:
                covariances[regime] = np.cov(regime_data.T)
            else:
                covariances[regime] = np.eye(n_features)
        
        return covariances
    
    def _combine_regime_predictions(self, results: List[RegimeDetectionResult]) -> np.ndarray:
        """Combine regime predictions using voting."""
        # Get the most common prediction for each point
        all_predictions = np.array([r.regime_ids for r in results])
        combined_predictions = np.zeros(len(results[0].regime_ids), dtype=int)
        
        for i in range(len(combined_predictions)):
            votes = all_predictions[:, i]
            combined_predictions[i] = np.bincount(votes).argmax()
        
        return combined_predictions
    
    def _combine_regime_probabilities(self, results: List[RegimeDetectionResult]) -> np.ndarray:
        """Combine regime probabilities by averaging."""
        all_probabilities = np.array([r.regime_probabilities for r in results])
        return np.mean(all_probabilities, axis=0)
    
    def _combine_transition_matrices(self, results: List[RegimeDetectionResult]) -> np.ndarray:
        """Combine transition matrices by averaging."""
        all_transitions = np.array([r.transition_matrix for r in results])
        return np.mean(all_transitions, axis=0)
    
    def validate_regime_continuity(self, regime_ids: np.ndarray) -> Dict[str, Any]:
        """Validate regime continuity and detect anomalies."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check for minimum regime duration
        durations = self._calculate_regime_durations(regime_ids)
        short_durations = [d for d in durations if d < self.config.min_regime_duration]
        
        if short_durations:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Found {len(short_durations)} regimes with duration < {self.config.min_regime_duration}")
        
        # Check for maximum regime duration
        long_durations = [d for d in durations if d > self.config.max_regime_duration]
        if long_durations:
            validation_results['issues'].append(f"Found {len(long_durations)} regimes with duration > {self.config.max_regime_duration}")
        
        # Check transition frequency
        transitions = sum(1 for i in range(1, len(regime_ids)) if regime_ids[i] != regime_ids[i-1])
        transition_rate = transitions / len(regime_ids)
        
        if transition_rate > self.config.transition_threshold:
            validation_results['issues'].append(f"High transition rate: {transition_rate:.3f} > {self.config.transition_threshold}")
        
        # Statistics
        validation_results['statistics'] = {
            'total_regimes': len(np.unique(regime_ids)),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'transition_rate': transition_rate,
            'short_regimes': len(short_durations),
            'long_regimes': len(long_durations)
        }
        
        return validation_results
    
    def detect_regime_changes(self, regime_ids: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and analyze regime changes."""
        changes = []
        
        for i in range(1, len(regime_ids)):
            if regime_ids[i] != regime_ids[i-1]:
                changes.append({
                    'index': i,
                    'from_regime': regime_ids[i-1],
                    'to_regime': regime_ids[i],
                    'change_type': 'transition'
                })
        
        return changes
    
    def analyze_regime_performance(self, regime_ids: np.ndarray, returns: np.ndarray) -> Dict[str, Any]:
        """Analyze performance of different regimes."""
        unique_regimes = np.unique(regime_ids)
        regime_performance = {}
        
        for regime in unique_regimes:
            regime_returns = returns[regime_ids == regime]
            regime_performance[str(regime)] = {
                'count': len(regime_returns),
                'mean_return': np.mean(regime_returns),
                'std_return': np.std(regime_returns),
                'sharpe_ratio': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0,
                'positive_ratio': np.sum(regime_returns > 0) / len(regime_returns) if len(regime_returns) > 0 else 0
            }
        
        return regime_performance

# Convenience functions
def get_hmm_regime_detector(config: Optional[HMMRegimeConfig] = None) -> HMMRegimeDetector:
    """Get a configured HMM regime detector."""
    return HMMRegimeDetector(config)

def detect_regimes(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: RegimeDetectionMethod = RegimeDetectionMethod.HMM_GAUSSIAN,
    config: Optional[HMMRegimeConfig] = None
) -> RegimeDetectionResult:
    """Convenience function for regime detection."""
    detector = get_hmm_regime_detector(config)
    return detector.detect_regimes(data, features, method)

def detect_ensemble_regimes(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    methods: Optional[List[RegimeDetectionMethod]] = None,
    config: Optional[HMMRegimeConfig] = None
) -> RegimeDetectionResult:
    """Convenience function for ensemble regime detection."""
    if config is None:
        config = HMMRegimeConfig()
    if methods:
        config.ensemble_methods = methods
    config.method = RegimeDetectionMethod.ENSEMBLE
    
    detector = get_hmm_regime_detector(config)
    return detector.detect_regimes(data, features)