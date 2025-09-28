"""
Micro Regime Detector

Detects micro-regimes within larger regime structures for neural architecture search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class MicroRegimeResult:
    """Result of micro-regime detection."""
    micro_types: List[str]
    micro_scores: List[float]
    detection_accuracy: float
    micro_regime_characteristics: Dict[str, Any]
    execution_time: float

class MicroRegimeDetector:
    """
    Micro Regime Detector for detecting fine-grained regime patterns.
    """
    
    def __init__(self, enable_hardware_optimization: bool = True, enable_matrix_optimization: bool = True):
        """
        Initialize the Micro Regime Detector.
        
        Args:
            enable_hardware_optimization: Whether to enable hardware optimization
            enable_matrix_optimization: Whether to enable matrix optimization
        """
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_matrix_optimization = enable_matrix_optimization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize detection parameters
        self.window_size = 10
        self.min_micro_regime_size = 3
        self.volatility_threshold = 0.02
        self.momentum_threshold = 0.01
        
        self.logger.info("Micro Regime Detector initialized")
    
    def detect_micro_regimes(self, data: np.ndarray, regime_predictions: np.ndarray, 
                           timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect micro-regimes within the given data and regime predictions.
        
        Args:
            data: Input data
            regime_predictions: Regime labels for each data point
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary containing micro-regime detection results
        """
        start_time = time.time()
        self.logger.info(f"Starting micro-regime detection for {len(data)} data points")
        
        try:
            # Detect micro-regimes based on volatility
            volatility_micro_regimes = self._detect_volatility_micro_regimes(data)
            
            # Detect micro-regimes based on momentum
            momentum_micro_regimes = self._detect_momentum_micro_regimes(data)
            
            # Detect micro-regimes based on clustering
            clustering_micro_regimes = self._detect_clustering_micro_regimes(data)
            
            # Combine micro-regime detections
            combined_micro_regimes = self._combine_micro_regime_detections(
                volatility_micro_regimes, momentum_micro_regimes, clustering_micro_regimes
            )
            
            # Calculate detection accuracy
            detection_accuracy = self._calculate_detection_accuracy(combined_micro_regimes, regime_predictions)
            
            # Analyze micro-regime characteristics
            micro_regime_characteristics = self._analyze_micro_regime_characteristics(
                data, combined_micro_regimes
            )
            
            execution_time = time.time() - start_time
            
            result = {
                'types': combined_micro_regimes['types'],
                'scores': combined_micro_regimes['scores'],
                'detection_accuracy': detection_accuracy,
                'micro_regime_characteristics': micro_regime_characteristics,
                'execution_time': execution_time,
                'volatility_detection': volatility_micro_regimes,
                'momentum_detection': momentum_micro_regimes,
                'clustering_detection': clustering_micro_regimes
            }
            
            self.logger.info(f"Micro-regime detection completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Micro-regime detection failed: {e}")
            return {
                'types': ['normal'] * len(data),
                'scores': [0.5] * len(data),
                'detection_accuracy': 0.0,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _detect_volatility_micro_regimes(self, data: np.ndarray) -> Dict[str, List]:
        """Detect micro-regimes based on volatility patterns."""
        try:
            micro_types = []
            micro_scores = []
            
            for i in range(len(data)):
                # Calculate local volatility
                start_idx = max(0, i - self.window_size // 2)
                end_idx = min(len(data), i + self.window_size // 2 + 1)
                window_data = data[start_idx:end_idx]
                
                if len(window_data) > 1:
                    volatility = np.std(window_data)
                else:
                    volatility = 0.0
                
                # Classify micro-regime based on volatility
                if volatility > self.volatility_threshold * 2:
                    micro_type = 'high_volatility'
                    micro_score = min(volatility * 20, 1.0)
                elif volatility < self.volatility_threshold * 0.5:
                    micro_type = 'low_volatility'
                    micro_score = 0.3
                elif volatility > self.volatility_threshold:
                    micro_type = 'moderate_volatility'
                    micro_score = 0.7
                else:
                    micro_type = 'normal_volatility'
                    micro_score = 0.5
                
                micro_types.append(micro_type)
                micro_scores.append(micro_score)
            
            return {
                'types': micro_types,
                'scores': micro_scores,
                'method': 'volatility_based'
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility micro-regime detection failed: {e}")
            return {
                'types': ['normal_volatility'] * len(data),
                'scores': [0.5] * len(data),
                'method': 'volatility_based'
            }
    
    def _detect_momentum_micro_regimes(self, data: np.ndarray) -> Dict[str, List]:
        """Detect micro-regimes based on momentum patterns."""
        try:
            micro_types = []
            micro_scores = []
            
            for i in range(len(data)):
                # Calculate local momentum
                start_idx = max(0, i - self.window_size)
                end_idx = i + 1
                window_data = data[start_idx:end_idx]
                
                if len(window_data) > 1:
                    # Calculate price momentum
                    price_change = (window_data[-1] - window_data[0]) / (window_data[0] + 1e-8)
                    momentum = np.mean(price_change)
                else:
                    momentum = 0.0
                
                # Classify micro-regime based on momentum
                if momentum > self.momentum_threshold * 2:
                    micro_type = 'strong_uptrend'
                    micro_score = min(abs(momentum) * 50, 1.0)
                elif momentum < -self.momentum_threshold * 2:
                    micro_type = 'strong_downtrend'
                    micro_score = min(abs(momentum) * 50, 1.0)
                elif momentum > self.momentum_threshold:
                    micro_type = 'uptrend'
                    micro_score = 0.7
                elif momentum < -self.momentum_threshold:
                    micro_type = 'downtrend'
                    micro_score = 0.7
                else:
                    micro_type = 'sideways'
                    micro_score = 0.5
                
                micro_types.append(micro_type)
                micro_scores.append(micro_score)
            
            return {
                'types': micro_types,
                'scores': micro_scores,
                'method': 'momentum_based'
            }
            
        except Exception as e:
            self.logger.warning(f"Momentum micro-regime detection failed: {e}")
            return {
                'types': ['sideways'] * len(data),
                'scores': [0.5] * len(data),
                'method': 'momentum_based'
            }
    
    def _detect_clustering_micro_regimes(self, data: np.ndarray) -> Dict[str, List]:
        """Detect micro-regimes using clustering approach."""
        try:
            # Use mini-batch K-means for efficiency
            n_micro_clusters = min(5, max(2, len(data) // 50))
            
            # Normalize data for clustering
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data)
            
            # Perform clustering
            kmeans = MiniBatchKMeans(n_clusters=n_micro_clusters, random_state=42, batch_size=1000)
            cluster_labels = kmeans.fit_predict(normalized_data)
            
            # Convert cluster labels to micro-regime types
            micro_types = []
            micro_scores = []
            
            for i, cluster_label in enumerate(cluster_labels):
                # Assign micro-regime type based on cluster
                if cluster_label == 0:
                    micro_type = 'cluster_0'
                elif cluster_label == 1:
                    micro_type = 'cluster_1'
                elif cluster_label == 2:
                    micro_type = 'cluster_2'
                else:
                    micro_type = f'cluster_{cluster_label}'
                
                # Calculate confidence score based on distance to cluster center
                distance_to_center = np.linalg.norm(normalized_data[i] - kmeans.cluster_centers_[cluster_label])
                max_distance = np.max(np.linalg.norm(kmeans.cluster_centers_, axis=1))
                micro_score = max(0.1, 1.0 - (distance_to_center / (max_distance + 1e-8)))
                
                micro_types.append(micro_type)
                micro_scores.append(micro_score)
            
            return {
                'types': micro_types,
                'scores': micro_scores,
                'method': 'clustering_based',
                'n_clusters': n_micro_clusters
            }
            
        except Exception as e:
            self.logger.warning(f"Clustering micro-regime detection failed: {e}")
            return {
                'types': ['cluster_0'] * len(data),
                'scores': [0.5] * len(data),
                'method': 'clustering_based'
            }
    
    def _combine_micro_regime_detections(self, volatility_detection: Dict[str, List],
                                       momentum_detection: Dict[str, List],
                                       clustering_detection: Dict[str, List]) -> Dict[str, List]:
        """Combine multiple micro-regime detection methods."""
        try:
            combined_types = []
            combined_scores = []
            
            for i in range(len(volatility_detection['types'])):
                # Get detections from each method
                vol_type = volatility_detection['types'][i]
                vol_score = volatility_detection['scores'][i]
                
                mom_type = momentum_detection['types'][i]
                mom_score = momentum_detection['scores'][i]
                
                cluster_type = clustering_detection['types'][i]
                cluster_score = clustering_detection['scores'][i]
                
                # Combine scores (weighted average)
                combined_score = (vol_score * 0.4 + mom_score * 0.3 + cluster_score * 0.3)
                
                # Determine combined type based on highest scoring method
                if vol_score >= mom_score and vol_score >= cluster_score:
                    combined_type = vol_type
                elif mom_score >= cluster_score:
                    combined_type = mom_type
                else:
                    combined_type = cluster_type
                
                # Add prefix to distinguish from main regimes
                combined_type = f"micro_{combined_type}"
                
                combined_types.append(combined_type)
                combined_scores.append(combined_score)
            
            return {
                'types': combined_types,
                'scores': combined_scores,
                'method': 'combined'
            }
            
        except Exception as e:
            self.logger.warning(f"Micro-regime combination failed: {e}")
            return {
                'types': ['micro_normal'] * len(volatility_detection['types']),
                'scores': [0.5] * len(volatility_detection['scores']),
                'method': 'combined'
            }
    
    def _calculate_detection_accuracy(self, micro_regimes: Dict[str, List], 
                                    regime_predictions: np.ndarray) -> float:
        """Calculate detection accuracy of micro-regimes."""
        try:
            # Simple accuracy based on consistency
            micro_types = micro_regimes['types']
            micro_scores = micro_regimes['scores']
            
            # Calculate consistency score
            consistency_score = np.mean(micro_scores)
            
            # Calculate diversity score (how many different micro-regime types)
            unique_types = len(set(micro_types))
            expected_diversity = min(5, len(micro_types) // 10)  # Expected number of micro-regime types
            diversity_score = min(1.0, unique_types / max(1, expected_diversity))
            
            # Combined accuracy
            accuracy = (consistency_score * 0.7 + diversity_score * 0.3)
            
            return accuracy
            
        except Exception as e:
            self.logger.warning(f"Detection accuracy calculation failed: {e}")
            return 0.5
    
    def _analyze_micro_regime_characteristics(self, data: np.ndarray, 
                                            micro_regimes: Dict[str, List]) -> Dict[str, Any]:
        """Analyze characteristics of detected micro-regimes."""
        try:
            micro_types = micro_regimes['types']
            micro_scores = micro_regimes['scores']
            
            unique_types = list(set(micro_types))
            characteristics = {
                'n_micro_regime_types': len(unique_types),
                'micro_regime_type_distribution': {},
                'micro_regime_quality_scores': {},
                'micro_regime_size_distribution': {}
            }
            
            # Analyze each micro-regime type
            for micro_type in unique_types:
                type_mask = np.array(micro_types) == micro_type
                type_indices = np.where(type_mask)[0]
                
                if len(type_indices) > 0:
                    # Distribution
                    characteristics['micro_regime_type_distribution'][micro_type] = len(type_indices)
                    
                    # Quality scores
                    type_scores = [micro_scores[i] for i in type_indices]
                    characteristics['micro_regime_quality_scores'][micro_type] = {
                        'mean_score': np.mean(type_scores),
                        'std_score': np.std(type_scores),
                        'min_score': np.min(type_scores),
                        'max_score': np.max(type_scores)
                    }
                    
                    # Size distribution (continuous segments)
                    segments = self._find_continuous_segments(type_indices)
                    segment_lengths = [len(segment) for segment in segments]
                    
                    characteristics['micro_regime_size_distribution'][micro_type] = {
                        'n_segments': len(segments),
                        'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
                        'max_segment_length': max(segment_lengths) if segment_lengths else 0,
                        'min_segment_length': min(segment_lengths) if segment_lengths else 0
                    }
            
            # Overall statistics
            characteristics['overall_quality'] = {
                'mean_score': np.mean(micro_scores),
                'std_score': np.std(micro_scores),
                'high_quality_ratio': np.mean(np.array(micro_scores) > 0.7),
                'low_quality_ratio': np.mean(np.array(micro_scores) < 0.3)
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Micro-regime characteristics analysis failed: {e}")
            return {}
    
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
