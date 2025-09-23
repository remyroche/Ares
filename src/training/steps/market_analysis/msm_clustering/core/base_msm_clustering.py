"""
Base MSM clustering classes and interfaces.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
from enum import Enum

from src.utils.logger import system_logger

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MSMRegimeType(Enum):
    """MSM regime types based on data characteristics."""
    HIGH_RETURN_LOW_VOL = "high_return_low_vol"
    HIGH_RETURN_HIGH_VOL = "high_return_high_vol"
    LOW_RETURN_LOW_VOL = "low_return_low_vol"
    LOW_RETURN_HIGH_VOL = "low_return_high_vol"
    EXTREME_VOLATILITY = "extreme_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"


@dataclass
class MSMClusteringResult:
    """Result of MSM clustering operation."""
    
    # Core results
    regime_labels: np.ndarray
    regime_centers: np.ndarray
    transition_matrix: np.ndarray
    regime_probabilities: np.ndarray
    
    # Structural breaks
    break_points: List[int]
    segment_labels: List[int]
    
    # Regime characteristics
    regime_statistics: Dict[str, Any]
    regime_durations: Dict[int, float]
    
    # Model performance
    log_likelihood: float
    aic: float
    bic: float
    
    # Clustering metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # Processing info
    processing_time: float
    n_regimes: int
    convergence_achieved: bool
    
    # Additional metadata
    metadata: Dict[str, Any]


class BaseMSMClusterer(ABC):
    """Base class for MSM clustering algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MSM clusterer with configuration."""
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        
    @abstractmethod
    def fit(self, data: np.ndarray) -> MSMClusteringResult:
        """Fit MSM model to data."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict regime labels for new data."""
        pass
    
    def _detect_structural_breaks(self, data: np.ndarray) -> List[int]:
        """Detect structural breaks in the data."""
        if not RUPTURES_AVAILABLE:
            self.logger.warning("Ruptures not available, using simple break detection")
            return self._simple_break_detection(data)
        
        try:
            # Configure break detection
            method = self.config.get('break_detection_method', 'pelt')
            min_segment_length = self.config.get('min_segment_length', 50)
            penalty = self.config.get('break_penalty', 'bic')
            
            # Detect breaks
            if method == 'pelt':
                model = rpt.Pelt(model=penalty, jump=5, min_size=min_segment_length)
            elif method == 'binseg':
                model = rpt.Binseg(model=penalty, jump=5, min_size=min_segment_length)
            else:
                model = rpt.Window(model=penalty, width=min_segment_length, jump=5)
            
            # Fit and predict
            model.fit(data)
            break_points = model.predict(pen=1.0)
            
            return break_points[:-1]  # Remove last point (end of data)
            
        except Exception as e:
            self.logger.warning(f"Structural break detection failed: {e}")
            return self._simple_break_detection(data)
    
    def _simple_break_detection(self, data: np.ndarray) -> List[int]:
        """Simple break detection fallback."""
        n_points = len(data)
        n_breaks = min(5, n_points // 100)  # Maximum 5 breaks
        
        if n_breaks == 0:
            return []
        
        # Simple uniform spacing
        break_points = [i * n_points // (n_breaks + 1) for i in range(1, n_breaks + 1)]
        return break_points
    
    def _identify_regimes(self, data: np.ndarray, break_points: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Identify regimes using clustering on segments."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for regime identification")
        
        # Create segments
        segments = []
        segment_labels = []
        
        start_idx = 0
        for i, break_point in enumerate(break_points):
            segment = data[start_idx:break_point]
            segments.append(segment)
            segment_labels.extend([i] * len(segment))
            start_idx = break_point
        
        # Add final segment
        if start_idx < len(data):
            segment = data[start_idx:]
            segments.append(segment)
            segment_labels.extend([len(break_points)] * len(segment))
        
        # Extract features for clustering
        features = self._extract_regime_features(segments)
        
        # Cluster segments
        n_regimes = min(self.config.get('n_regimes', 3), len(segments))
        clustering_method = self.config.get('clustering_method', 'gaussian_mixture')
        
        if clustering_method == 'gaussian_mixture':
            clusterer = GaussianMixture(
                n_components=n_regimes,
                covariance_type='full',
                random_state=42
            )
        else:
            clusterer = KMeans(n_clusters=n_regimes, random_state=42)
        
        regime_labels = clusterer.fit_predict(features)
        
        # Map segment labels to regime labels
        final_labels = np.array([regime_labels[segment_labels[i]] for i in range(len(data))])
        
        # Calculate regime statistics
        regime_stats = self._calculate_regime_statistics(data, final_labels)
        
        return final_labels, regime_stats
    
    def _extract_regime_features(self, segments: List[np.ndarray]) -> np.ndarray:
        """Extract features for regime clustering."""
        features = []
        
        for segment in segments:
            if len(segment) == 0:
                continue
                
            # Basic statistics
            mean_val = np.mean(segment)
            std_val = np.std(segment)
            skew_val = self._safe_skew(segment)
            kurt_val = self._safe_kurtosis(segment)
            
            # Volatility features
            returns = np.diff(segment) if len(segment) > 1 else [0]
            volatility = np.std(returns) if len(returns) > 0 else 0
            
            # Trend features
            trend = self._calculate_trend(segment)
            
            features.append([mean_val, std_val, skew_val, kurt_val, volatility, trend])
        
        return np.array(features)
    
    def _safe_skew(self, data: np.ndarray) -> float:
        """Calculate skewness safely."""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            return 0.0
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend strength."""
        if len(data) < 2:
            return 0.0
        
        try:
            x = np.arange(len(data))
            slope, _ = np.polyfit(x, data, 1)
            return float(slope)
        except:
            return 0.0
    
    def _calculate_regime_statistics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for each regime."""
        stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            regime_data = data[mask]
            
            if len(regime_data) == 0:
                continue
            
            stats[f'regime_{label}'] = {
                'count': len(regime_data),
                'mean': float(np.mean(regime_data)),
                'std': float(np.std(regime_data)),
                'min': float(np.min(regime_data)),
                'max': float(np.max(regime_data)),
                'duration': len(regime_data)
            }
        
        return stats
    
    def _calculate_transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        """Calculate transition matrix from regime labels."""
        unique_labels = np.unique(labels)
        n_regimes = len(unique_labels)
        
        # Create label mapping
        label_map = {label: i for i, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_map[label] for label in labels])
        
        # Count transitions
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(mapped_labels) - 1):
            current = mapped_labels[i]
            next_label = mapped_labels[i + 1]
            transition_counts[current, next_label] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        # Handle zero rows
        zero_rows = row_sums == 0
        transition_matrix[zero_rows] = 1.0 / n_regimes
        
        return transition_matrix
    
    def _calculate_clustering_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        if not SKLEARN_AVAILABLE:
            return {}
        
        try:
            from sklearn.metrics import (
                silhouette_score, calinski_harabasz_score, davies_bouldin_score
            )
            
            metrics = {}
            
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(data.reshape(-1, 1), labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data.reshape(-1, 1), labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(data.reshape(-1, 1), labels)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate clustering metrics: {e}")
            return {}