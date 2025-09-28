"""
NAS Regime Analyzer

Analyzes regime characteristics and transitions for neural architecture search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

@dataclass
class RegimeAnalysisResult:
    """Result of regime analysis."""
    regime_characteristics: Dict[int, Dict[str, Any]]
    transition_matrix: np.ndarray
    regime_durations: Dict[int, float]
    regime_stability: Dict[str, float]
    execution_time: float

class NASRegimeAnalyzer:
    """
    NAS Regime Analyzer for analyzing regime characteristics and transitions.
    """
    
    def __init__(self, enable_hardware_optimization: bool = True, enable_matrix_optimization: bool = True):
        """
        Initialize the NAS Regime Analyzer.
        
        Args:
            enable_hardware_optimization: Whether to enable hardware optimization
            enable_matrix_optimization: Whether to enable matrix optimization
        """
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_matrix_optimization = enable_matrix_optimization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info("NAS Regime Analyzer initialized")
    
    def analyze_regimes(self, data: np.ndarray, regime_predictions: np.ndarray, 
                       timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze regime characteristics and transitions.
        
        Args:
            data: Input data
            regime_predictions: Regime labels for each data point
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary containing regime analysis results
        """
        start_time = time.time()
        self.logger.info(f"Starting regime analysis for {len(data)} data points")
        
        try:
            # Validate inputs
            if len(data) != len(regime_predictions):
                raise ValueError("Data and regime predictions must have the same length")
            
            # Analyze regime characteristics
            regime_characteristics = self._analyze_regime_characteristics(data, regime_predictions)
            
            # Calculate transition matrix
            transition_matrix = self._calculate_transition_matrix(regime_predictions)
            
            # Analyze regime durations
            regime_durations = self._analyze_regime_durations(regime_predictions, timestamps)
            
            # Calculate regime stability metrics
            regime_stability = self._calculate_regime_stability(data, regime_predictions)
            
            execution_time = time.time() - start_time
            
            result = {
                'regime_characteristics': regime_characteristics,
                'transition_matrix': transition_matrix.tolist(),
                'regime_durations': regime_durations,
                'regime_stability': regime_stability,
                'execution_time': execution_time,
                'n_regimes': len(np.unique(regime_predictions)),
                'total_data_points': len(data)
            }
            
            self.logger.info(f"Regime analysis completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Regime analysis failed: {e}")
            return {
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _analyze_regime_characteristics(self, data: np.ndarray, regime_predictions: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each regime."""
        try:
            unique_regimes = np.unique(regime_predictions)
            characteristics = {}
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    regime_stats = {
                        'count': len(regime_data),
                        'percentage': len(regime_data) / len(data) * 100,
                        'mean': np.mean(regime_data, axis=0).tolist(),
                        'std': np.std(regime_data, axis=0).tolist(),
                        'min': np.min(regime_data, axis=0).tolist(),
                        'max': np.max(regime_data, axis=0).tolist(),
                        'median': np.median(regime_data, axis=0).tolist(),
                        'q25': np.percentile(regime_data, 25, axis=0).tolist(),
                        'q75': np.percentile(regime_data, 75, axis=0).tolist()
                    }
                    
                    # Calculate additional statistics
                    if len(regime_data) > 1:
                        regime_stats['skewness'] = self._calculate_skewness(regime_data)
                        regime_stats['kurtosis'] = self._calculate_kurtosis(regime_data)
                        regime_stats['volatility'] = np.mean(np.std(regime_data, axis=0))
                    else:
                        regime_stats['skewness'] = 0.0
                        regime_stats['kurtosis'] = 0.0
                        regime_stats['volatility'] = 0.0
                    
                    characteristics[int(regime)] = regime_stats
                else:
                    characteristics[int(regime)] = {
                        'count': 0,
                        'percentage': 0.0,
                        'error': 'No data points for this regime'
                    }
            
            self.logger.debug(f"Analyzed characteristics for {len(characteristics)} regimes")
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Error analyzing regime characteristics: {e}")
            return {}
    
    def _calculate_transition_matrix(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime transition matrix."""
        try:
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)
            
            # Create regime mapping
            regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
            
            # Initialize transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            # Count transitions
            for i in range(len(regime_predictions) - 1):
                current_regime = regime_map[regime_predictions[i]]
                next_regime = regime_map[regime_predictions[i + 1]]
                transition_matrix[current_regime, next_regime] += 1
            
            # Normalize rows to get probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_matrix = transition_matrix / row_sums
            
            self.logger.debug(f"Calculated transition matrix for {n_regimes} regimes")
            return transition_matrix
            
        except Exception as e:
            self.logger.warning(f"Error calculating transition matrix: {e}")
            # Return identity matrix as fallback
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)
            return np.eye(n_regimes)
    
    def _analyze_regime_durations(self, regime_predictions: np.ndarray, 
                                 timestamps: Optional[np.ndarray] = None) -> Dict[int, float]:
        """Analyze regime durations."""
        try:
            unique_regimes = np.unique(regime_predictions)
            durations = {}
            
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) > 0:
                    # Calculate continuous segments
                    segments = self._find_continuous_segments(regime_indices)
                    segment_lengths = [len(segment) for segment in segments]
                    
                    durations[int(regime)] = {
                        'total_duration': len(regime_indices),
                        'n_segments': len(segments),
                        'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
                        'max_segment_length': max(segment_lengths) if segment_lengths else 0,
                        'min_segment_length': min(segment_lengths) if segment_lengths else 0,
                        'segment_lengths': segment_lengths
                    }
                    
                    # Add time-based duration if timestamps provided
                    if timestamps is not None and len(timestamps) == len(regime_predictions):
                        time_durations = []
                        for segment in segments:
                            if len(segment) > 1:
                                duration = timestamps[segment[-1]] - timestamps[segment[0]]
                                time_durations.append(duration)
                        
                        if time_durations:
                            durations[int(regime)]['avg_time_duration'] = np.mean(time_durations)
                            durations[int(regime)]['max_time_duration'] = max(time_durations)
                            durations[int(regime)]['min_time_duration'] = min(time_durations)
                else:
                    durations[int(regime)] = {
                        'total_duration': 0,
                        'n_segments': 0,
                        'avg_segment_length': 0,
                        'max_segment_length': 0,
                        'min_segment_length': 0,
                        'segment_lengths': []
                    }
            
            self.logger.debug(f"Analyzed durations for {len(durations)} regimes")
            return durations
            
        except Exception as e:
            self.logger.warning(f"Error analyzing regime durations: {e}")
            return {}
    
    def _calculate_regime_stability(self, data: np.ndarray, regime_predictions: np.ndarray) -> Dict[str, float]:
        """Calculate regime stability metrics."""
        try:
            unique_regimes = np.unique(regime_predictions)
            n_regimes = len(unique_regimes)
            
            stability_metrics = {}
            
            # Calculate within-regime stability (how consistent data is within each regime)
            within_regime_stability = []
            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_data = data[regime_mask]
                
                if len(regime_data) > 1:
                    # Calculate coefficient of variation as stability measure
                    regime_mean = np.mean(regime_data, axis=0)
                    regime_std = np.std(regime_data, axis=0)
                    
                    # Avoid division by zero
                    regime_mean[regime_mean == 0] = 1e-8
                    cv = np.mean(regime_std / np.abs(regime_mean))
                    
                    # Lower CV means higher stability
                    stability = 1.0 / (1.0 + cv)
                    within_regime_stability.append(stability)
            
            stability_metrics['within_regime_stability'] = np.mean(within_regime_stability) if within_regime_stability else 0.0
            
            # Calculate between-regime separation (how well separated regimes are)
            if n_regimes > 1:
                regime_centers = []
                for regime in unique_regimes:
                    regime_mask = regime_predictions == regime
                    regime_data = data[regime_mask]
                    if len(regime_data) > 0:
                        regime_centers.append(np.mean(regime_data, axis=0))
                
                if len(regime_centers) > 1:
                    regime_centers = np.array(regime_centers)
                    center_distances = euclidean_distances(regime_centers)
                    
                    # Average distance between regime centers
                    off_diagonal_distances = center_distances[np.triu_indices_from(center_distances, k=1)]
                    stability_metrics['between_regime_separation'] = np.mean(off_diagonal_distances)
                else:
                    stability_metrics['between_regime_separation'] = 0.0
            else:
                stability_metrics['between_regime_separation'] = 0.0
            
            # Calculate regime persistence (how often regimes change)
            regime_changes = np.sum(regime_predictions[1:] != regime_predictions[:-1])
            stability_metrics['regime_persistence'] = 1.0 - (regime_changes / max(1, len(regime_predictions) - 1))
            
            # Calculate regime balance (how evenly distributed regimes are)
            regime_counts = [np.sum(regime_predictions == regime) for regime in unique_regimes]
            regime_proportions = np.array(regime_counts) / len(regime_predictions)
            
            # Entropy-based balance measure
            entropy = -np.sum(regime_proportions * np.log(regime_proportions + 1e-8))
            max_entropy = np.log(n_regimes)
            stability_metrics['regime_balance'] = entropy / max_entropy if max_entropy > 0 else 0.0
            
            self.logger.debug("Calculated regime stability metrics")
            return stability_metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime stability: {e}")
            return {
                'within_regime_stability': 0.0,
                'between_regime_separation': 0.0,
                'regime_persistence': 0.0,
                'regime_balance': 0.0
            }
    
    def _find_continuous_segments(self, indices: np.ndarray) -> List[List[int]]:
        """Find continuous segments in a list of indices."""
        if len(indices) == 0:
            return []
        
        segments = []
        current_segment = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_segment.append(indices[i])
            else:
                segments.append(current_segment)
                current_segment = [indices[i]]
        
        segments.append(current_segment)
        return segments
    
    def _calculate_skewness(self, data: np.ndarray) -> List[float]:
        """Calculate skewness for each feature."""
        try:
            from scipy import stats
            return [stats.skew(data[:, i]) for i in range(data.shape[1])]
        except ImportError:
            # Simple skewness calculation
            result = []
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                if std_val > 0:
                    skewness = np.mean(((feature_data - mean_val) / std_val) ** 3)
                else:
                    skewness = 0.0
                result.append(skewness)
            return result
        except Exception:
            return [0.0] * data.shape[1]
    
    def _calculate_kurtosis(self, data: np.ndarray) -> List[float]:
        """Calculate kurtosis for each feature."""
        try:
            from scipy import stats
            return [stats.kurtosis(data[:, i]) for i in range(data.shape[1])]
        except ImportError:
            # Simple kurtosis calculation
            result = []
            for i in range(data.shape[1]):
                feature_data = data[:, i]
                mean_val = np.mean(feature_data)
                std_val = np.std(feature_data)
                if std_val > 0:
                    kurtosis = np.mean(((feature_data - mean_val) / std_val) ** 4) - 3
                else:
                    kurtosis = 0.0
                result.append(kurtosis)
            return result
        except Exception:
            return [0.0] * data.shape[1]
