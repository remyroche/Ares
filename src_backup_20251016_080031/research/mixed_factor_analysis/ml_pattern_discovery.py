"""
ML-Based Pattern Discovery Framework

This module uses machine learning techniques to automatically discover new price patterns
that may not be captured by traditional technical analysis. It complements the mathematical
pattern definitions with data-driven pattern discovery.

Key Approaches:
1. Unsupervised Pattern Discovery (clustering, autoencoders)
2. Time Series Motif Discovery (matrix profile, SAX)
3. Anomaly Detection Patterns
4. Sequence Pattern Mining
5. Deep Learning Pattern Discovery (LSTM autoencoders)
6. Evolutionary Pattern Discovery (genetic algorithms)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings

from src.utils.logger import system_logger


class MLPatternDiscoveryMethod(Enum):
    """ML-based pattern discovery methods."""
    CLUSTERING_BASED = "clustering_based"
    AUTOENCODER_BASED = "autoencoder_based"
    MATRIX_PROFILE = "matrix_profile"
    ANOMALY_DETECTION = "anomaly_detection"
    SEQUENCE_MINING = "sequence_mining"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    EVOLUTIONARY_DISCOVERY = "evolutionary_discovery"
    CHANGE_POINT_DETECTION = "change_point_detection"


@dataclass
class MLDiscoveredPattern:
    """Result of ML-based pattern discovery."""
    pattern_id: str
    discovery_method: MLPatternDiscoveryMethod
    pattern_description: str
    pattern_labels: pd.Series  # Binary labels
    pattern_strength: float  # 0-1 confidence score
    frequency: float
    statistical_significance: Dict[str, float]
    pattern_characteristics: Dict[str, Any]
    mathematical_approximation: str  # Attempt to express as mathematical formula
    
    @property
    def is_significant_pattern(self) -> bool:
        """Check if discovered pattern is statistically significant."""
        return (
            self.frequency >= 0.02 and  # At least 2% frequency
            self.pattern_strength > 0.3 and  # At least 30% confidence
            self.statistical_significance.get('p_value', 1.0) < 0.05
        )


class ClusteringBasedPatternDiscovery:
    """Discover patterns using clustering techniques on price sequences."""
    
    def __init__(self):
        self.logger = system_logger.getChild('ClusteringPatternDiscovery')
    
    def discover_patterns(self, 
                         market_data: pd.DataFrame,
                         sequence_length: int = 20,
                         n_clusters: int = 8) -> List[MLDiscoveredPattern]:
        """
        Discover patterns by clustering price sequences.
        
        Method:
        1. Create overlapping price sequences of fixed length
        2. Normalize sequences for shape comparison
        3. Cluster sequences to find common patterns
        4. Validate clusters as meaningful patterns
        """
        
        self.logger.info(f"🔍 Discovering patterns via clustering (sequence_length={sequence_length})")
        
        prices = market_data['close']
        
        # Create price sequences
        sequences = []
        sequence_indices = []
        
        for i in range(len(prices) - sequence_length + 1):
            sequence = prices.iloc[i:i+sequence_length].values
            
            # Normalize sequence (remove trend, focus on shape)
            if sequence[-1] != sequence[0]:
                normalized_sequence = (sequence - sequence[0]) / (sequence[-1] - sequence[0])
            else:
                normalized_sequence = sequence - sequence[0]
            
            sequences.append(normalized_sequence)
            sequence_indices.append(i)
        
        if len(sequences) < n_clusters * 10:
            self.logger.warning("Insufficient sequences for clustering")
            return []
        
        # Cluster sequences
        sequences_array = np.array(sequences)
        
        # Standardize
        scaler = StandardScaler()
        sequences_scaled = scaler.fit_transform(sequences_array)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(sequences_scaled)
        
        # Analyze each cluster as potential pattern
        discovered_patterns = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_sequences = sequences_array[cluster_mask]
            cluster_indices = [sequence_indices[i] for i in range(len(sequence_indices)) if cluster_mask[i]]
            
            if len(cluster_sequences) < 10:  # Need minimum occurrences
                continue
            
            # Calculate cluster characteristics
            pattern_characteristics = self._analyze_cluster_characteristics(
                cluster_sequences, cluster_indices, prices, sequence_length
            )
            
            # Create pattern labels
            pattern_labels = pd.Series(0, index=prices.index)
            for idx in cluster_indices:
                pattern_labels.iloc[idx] = 1
            
            # Statistical significance
            statistical_significance = self._test_cluster_significance(
                cluster_sequences, sequences_array
            )
            
            # Generate mathematical approximation
            mathematical_approximation = self._approximate_cluster_formula(
                cluster_sequences, sequence_length
            )
            
            pattern = MLDiscoveredPattern(
                pattern_id=f"cluster_{cluster_id}",
                discovery_method=MLPatternDiscoveryMethod.CLUSTERING_BASED,
                pattern_description=pattern_characteristics['description'],
                pattern_labels=pattern_labels,
                pattern_strength=pattern_characteristics['strength'],
                frequency=len(cluster_indices) / len(sequences),
                statistical_significance=statistical_significance,
                pattern_characteristics=pattern_characteristics,
                mathematical_approximation=mathematical_approximation
            )
            
            if pattern.is_significant_pattern:
                discovered_patterns.append(pattern)
                self.logger.info(f"   ✅ Discovered significant pattern: {pattern.pattern_id}")
        
        return discovered_patterns
    
    def _analyze_cluster_characteristics(self, 
                                       cluster_sequences: np.ndarray,
                                       cluster_indices: List[int],
                                       prices: pd.Series,
                                       sequence_length: int) -> Dict[str, Any]:
        """Analyze characteristics of a sequence cluster."""
        
        # Calculate cluster centroid
        centroid = np.mean(cluster_sequences, axis=0)
        
        # Analyze centroid shape
        centroid_diff = np.diff(centroid)
        
        # Determine pattern type based on centroid characteristics
        if np.mean(centroid_diff) > 0.1:
            pattern_type = "upward_trend"
            description = f"Upward trending sequences over {sequence_length} periods"
        elif np.mean(centroid_diff) < -0.1:
            pattern_type = "downward_trend"
            description = f"Downward trending sequences over {sequence_length} periods"
        elif np.std(centroid_diff) < 0.05:
            pattern_type = "consolidation"
            description = f"Sideways consolidation sequences over {sequence_length} periods"
        elif np.std(centroid_diff) > 0.2:
            pattern_type = "high_volatility"
            description = f"High volatility sequences over {sequence_length} periods"
        else:
            pattern_type = "mixed"
            description = f"Mixed pattern sequences over {sequence_length} periods"
        
        # Calculate pattern strength (intra-cluster similarity)
        intra_cluster_distances = []
        for i in range(len(cluster_sequences)):
            for j in range(i+1, len(cluster_sequences)):
                distance = np.linalg.norm(cluster_sequences[i] - cluster_sequences[j])
                intra_cluster_distances.append(distance)
        
        if intra_cluster_distances:
            avg_intra_distance = np.mean(intra_cluster_distances)
            max_possible_distance = np.sqrt(sequence_length * 4)  # Rough estimate
            pattern_strength = max(0, 1 - avg_intra_distance / max_possible_distance)
        else:
            pattern_strength = 0.0
        
        return {
            'pattern_type': pattern_type,
            'description': description,
            'strength': pattern_strength,
            'centroid': centroid.tolist(),
            'avg_intra_distance': avg_intra_distance if intra_cluster_distances else 0,
            'cluster_size': len(cluster_sequences)
        }
    
    def _test_cluster_significance(self, 
                                 cluster_sequences: np.ndarray,
                                 all_sequences: np.ndarray) -> Dict[str, float]:
        """Test statistical significance of cluster."""
        
        # Test if cluster sequences are significantly different from random
        cluster_mean = np.mean(cluster_sequences.flatten())
        all_mean = np.mean(all_sequences.flatten())
        
        cluster_flat = cluster_sequences.flatten()
        all_flat = all_sequences.flatten()
        
        # Two-sample t-test
        try:
            t_stat, p_value = stats.ttest_ind(cluster_flat, all_flat)
            return {
                'p_value': float(p_value),
                't_statistic': float(t_stat),
                'cluster_mean': float(cluster_mean),
                'overall_mean': float(all_mean)
            }
        except:
            return {'p_value': 1.0, 't_statistic': 0.0}
    
    def _approximate_cluster_formula(self, 
                                   cluster_sequences: np.ndarray,
                                   sequence_length: int) -> str:
        """Attempt to express cluster as mathematical formula."""
        
        centroid = np.mean(cluster_sequences, axis=0)
        centroid_diff = np.diff(centroid)
        
        # Simple pattern recognition
        if np.all(centroid_diff > 0):
            return f"Monotonic increasing over {sequence_length} periods"
        elif np.all(centroid_diff < 0):
            return f"Monotonic decreasing over {sequence_length} periods"
        elif np.std(centroid_diff) < 0.05:
            return f"Approximately flat over {sequence_length} periods"
        else:
            # Try to identify peaks/valleys
            peaks = []
            valleys = []
            
            for i in range(1, len(centroid) - 1):
                if centroid[i] > centroid[i-1] and centroid[i] > centroid[i+1]:
                    peaks.append(i)
                elif centroid[i] < centroid[i-1] and centroid[i] < centroid[i+1]:
                    valleys.append(i)
            
            if len(peaks) == 1 and len(valleys) == 0:
                return f"Single peak at position {peaks[0]} over {sequence_length} periods"
            elif len(valleys) == 1 and len(peaks) == 0:
                return f"Single valley at position {valleys[0]} over {sequence_length} periods"
            elif len(peaks) > 0 and len(valleys) > 0:
                return f"Oscillating pattern with {len(peaks)} peaks, {len(valleys)} valleys"
            else:
                return f"Complex pattern over {sequence_length} periods"


class AnomalyPatternDiscovery:
    """Discover patterns using anomaly detection techniques."""
    
    def __init__(self):
        self.logger = system_logger.getChild('AnomalyPatternDiscovery')
    
    def discover_anomaly_patterns(self, 
                                 market_data: pd.DataFrame,
                                 feature_columns: List[str] = None,
                                 contamination: float = 0.1) -> List[MLDiscoveredPattern]:
        """
        Discover patterns using anomaly detection.
        
        Method:
        1. Create feature matrix from market data
        2. Use Isolation Forest to identify anomalous periods
        3. Analyze anomaly characteristics to define patterns
        4. Validate patterns for significance and trading utility
        """
        
        self.logger.info(f"🔍 Discovering anomaly patterns (contamination={contamination})")
        
        # Create feature matrix
        if feature_columns is None:
            features = self._create_default_features(market_data)
        else:
            features = market_data[feature_columns].fillna(0)
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)
        
        # Convert to binary (1 = anomaly, 0 = normal)
        anomaly_binary = (anomaly_labels == -1).astype(int)
        anomaly_series = pd.Series(anomaly_binary, index=market_data.index)
        
        # Analyze anomaly characteristics
        anomaly_characteristics = self._analyze_anomaly_characteristics(
            market_data, anomaly_series, features
        )
        
        # Statistical significance
        statistical_significance = self._test_anomaly_significance(
            market_data, anomaly_series
        )
        
        # Mathematical approximation
        mathematical_approximation = self._approximate_anomaly_conditions(
            features, anomaly_binary
        )
        
        pattern = MLDiscoveredPattern(
            pattern_id="anomaly_pattern",
            discovery_method=MLPatternDiscoveryMethod.ANOMALY_DETECTION,
            pattern_description=anomaly_characteristics['description'],
            pattern_labels=anomaly_series,
            pattern_strength=anomaly_characteristics['strength'],
            frequency=anomaly_series.sum() / len(anomaly_series),
            statistical_significance=statistical_significance,
            pattern_characteristics=anomaly_characteristics,
            mathematical_approximation=mathematical_approximation
        )
        
        return [pattern] if pattern.is_significant_pattern else []
    
    def _create_default_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create default feature set for anomaly detection."""
        
        features = pd.DataFrame(index=market_data.index)
        
        # Price-based features
        returns = market_data['close'].pct_change().fillna(0)
        features['return'] = returns
        features['abs_return'] = abs(returns)
        features['return_squared'] = returns ** 2
        
        # Volatility features
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_20'] = returns.rolling(20).std()
        features['vol_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # Momentum features
        features['momentum_5'] = returns.rolling(5).mean()
        features['momentum_20'] = returns.rolling(20).mean()
        
        # Price level features
        features['price_ma_ratio'] = market_data['close'] / market_data['close'].rolling(20).mean()
        
        # Volume features (if available)
        if 'volume' in market_data.columns:
            features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
            features['volume_price_corr'] = returns.rolling(20).corr(market_data['volume'])
        
        # Range features (if available)
        if all(col in market_data.columns for col in ['high', 'low']):
            features['daily_range'] = (market_data['high'] - market_data['low']) / market_data['close']
        
        return features.fillna(0)
    
    def _analyze_anomaly_characteristics(self, 
                                       market_data: pd.DataFrame,
                                       anomaly_labels: pd.Series,
                                       features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze characteristics of detected anomalies."""
        
        anomaly_periods = anomaly_labels[anomaly_labels == 1].index
        normal_periods = anomaly_labels[anomaly_labels == 0].index
        
        if len(anomaly_periods) == 0:
            return {'description': 'No anomalies detected', 'strength': 0.0}
        
        # Compare feature values during anomalies vs normal periods
        anomaly_features = features.loc[anomaly_periods]
        normal_features = features.loc[normal_periods]
        
        feature_differences = {}
        for col in features.columns:
            anomaly_mean = anomaly_features[col].mean()
            normal_mean = normal_features[col].mean()
            
            if normal_features[col].std() > 0:
                # Z-score difference
                difference = (anomaly_mean - normal_mean) / normal_features[col].std()
                feature_differences[col] = difference
        
        # Find most distinctive features
        top_features = sorted(feature_differences.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Generate description
        if top_features:
            main_feature, main_difference = top_features[0]
            if main_difference > 2:
                description = f"Anomalies characterized by high {main_feature} (>{main_difference:.1f} std above normal)"
            elif main_difference < -2:
                description = f"Anomalies characterized by low {main_feature} (>{abs(main_difference):.1f} std below normal)"
            else:
                description = f"Anomalies with mixed characteristics involving {main_feature}"
        else:
            description = "Anomalies with unclear characteristics"
        
        # Calculate pattern strength (how different anomalies are from normal)
        if top_features:
            pattern_strength = min(abs(top_features[0][1]) / 5.0, 1.0)  # Normalize by 5 std
        else:
            pattern_strength = 0.0
        
        return {
            'description': description,
            'strength': pattern_strength,
            'feature_differences': feature_differences,
            'top_distinctive_features': top_features,
            'anomaly_count': len(anomaly_periods)
        }
    
    def _test_anomaly_significance(self, 
                                 market_data: pd.DataFrame,
                                 anomaly_labels: pd.Series) -> Dict[str, float]:
        """Test statistical significance of anomaly pattern."""
        
        returns = market_data['close'].pct_change().fillna(0)
        
        anomaly_returns = returns[anomaly_labels == 1]
        normal_returns = returns[anomaly_labels == 0]
        
        if len(anomaly_returns) < 5 or len(normal_returns) < 5:
            return {'p_value': 1.0, 't_statistic': 0.0}
        
        try:
            # Test if returns are different during anomalies
            t_stat, p_value = stats.ttest_ind(anomaly_returns, normal_returns)
            
            # Test if volatility is different
            anomaly_vol = anomaly_returns.std()
            normal_vol = normal_returns.std()
            
            return {
                'p_value': float(p_value),
                't_statistic': float(t_stat),
                'anomaly_return_mean': float(anomaly_returns.mean()),
                'normal_return_mean': float(normal_returns.mean()),
                'anomaly_volatility': float(anomaly_vol),
                'normal_volatility': float(normal_vol)
            }
        except:
            return {'p_value': 1.0, 't_statistic': 0.0}
    
    def _approximate_anomaly_conditions(self, 
                                      features: pd.DataFrame,
                                      anomaly_labels: np.ndarray) -> str:
        """Approximate anomaly conditions as mathematical formula."""
        
        # Find feature thresholds that best separate anomalies
        anomaly_mask = anomaly_labels == 1
        
        if anomaly_mask.sum() == 0:
            return "No anomalies detected"
        
        conditions = []
        
        for col in features.columns:
            anomaly_values = features.loc[anomaly_mask, col]
            normal_values = features.loc[~anomaly_mask, col]
            
            if len(anomaly_values) > 0 and len(normal_values) > 0:
                # Find threshold that best separates
                anomaly_median = anomaly_values.median()
                normal_median = normal_values.median()
                
                if abs(anomaly_median - normal_median) > normal_values.std():
                    if anomaly_median > normal_median:
                        threshold = normal_median + normal_values.std()
                        conditions.append(f"{col} > {threshold:.4f}")
                    else:
                        threshold = normal_median - normal_values.std()
                        conditions.append(f"{col} < {threshold:.4f}")
        
        if conditions:
            return "Anomaly IF: " + " AND ".join(conditions[:3])  # Top 3 conditions
        else:
            return "Complex anomaly conditions - no simple threshold"


class ChangePointPatternDiscovery:
    """Discover patterns using change point detection."""
    
    def __init__(self):
        self.logger = system_logger.getChild('ChangePointPatternDiscovery')
    
    def discover_change_point_patterns(self, 
                                     market_data: pd.DataFrame,
                                     window_size: int = 50,
                                     min_segment_length: int = 10) -> List[MLDiscoveredPattern]:
        """
        Discover patterns using change point detection.
        
        Method:
        1. Detect change points in time series using statistical tests
        2. Analyze segments between change points
        3. Cluster similar segments to find recurring patterns
        4. Define patterns based on segment characteristics
        """
        
        self.logger.info(f"📊 Discovering change point patterns (window={window_size})")
        
        prices = market_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Detect change points using rolling window variance test
        change_points = self._detect_change_points(returns, window_size)
        
        if len(change_points) < 3:
            self.logger.warning("Insufficient change points detected")
            return []
        
        # Analyze segments between change points
        segments = self._extract_segments(returns, change_points, min_segment_length)
        
        if len(segments) < 5:
            self.logger.warning("Insufficient segments for pattern analysis")
            return []
        
        # Cluster segments to find patterns
        segment_patterns = self._cluster_segments(segments)
        
        # Convert to ML patterns
        discovered_patterns = []
        
        for pattern_id, segment_cluster in segment_patterns.items():
            pattern_labels = self._create_segment_pattern_labels(
                segment_cluster, market_data.index
            )
            
            pattern_characteristics = self._analyze_segment_characteristics(segment_cluster)
            
            statistical_significance = self._test_segment_significance(segment_cluster)
            
            mathematical_approximation = self._approximate_segment_formula(segment_cluster)
            
            pattern = MLDiscoveredPattern(
                pattern_id=f"changepoint_{pattern_id}",
                discovery_method=MLPatternDiscoveryMethod.CHANGE_POINT_DETECTION,
                pattern_description=pattern_characteristics['description'],
                pattern_labels=pattern_labels,
                pattern_strength=pattern_characteristics['strength'],
                frequency=pattern_labels.sum() / len(pattern_labels),
                statistical_significance=statistical_significance,
                pattern_characteristics=pattern_characteristics,
                mathematical_approximation=mathematical_approximation
            )
            
            if pattern.is_significant_pattern:
                discovered_patterns.append(pattern)
        
        return discovered_patterns
    
    def _detect_change_points(self, returns: pd.Series, window_size: int) -> List[int]:
        """Detect change points using rolling variance test."""
        
        change_points = []
        
        for i in range(window_size, len(returns) - window_size):
            # Compare variance before and after potential change point
            before_window = returns.iloc[i-window_size:i]
            after_window = returns.iloc[i:i+window_size]
            
            # F-test for variance equality
            try:
                f_stat = before_window.var() / after_window.var()
                
                # Critical value for F-test (approximate)
                critical_value = 2.0  # Simplified threshold
                
                if f_stat > critical_value or f_stat < 1/critical_value:
                    change_points.append(i)
            except:
                continue
        
        # Remove close change points
        filtered_change_points = []
        for cp in change_points:
            if not filtered_change_points or cp - filtered_change_points[-1] > window_size:
                filtered_change_points.append(cp)
        
        return filtered_change_points
    
    def _extract_segments(self, 
                         returns: pd.Series,
                         change_points: List[int],
                         min_length: int) -> List[Dict[str, Any]]:
        """Extract segments between change points."""
        
        segments = []
        
        # Add start and end points
        all_points = [0] + change_points + [len(returns)]
        all_points = sorted(set(all_points))
        
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]
            
            if end_idx - start_idx >= min_length:
                segment_returns = returns.iloc[start_idx:end_idx]
                
                segment = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'length': end_idx - start_idx,
                    'returns': segment_returns,
                    'mean_return': segment_returns.mean(),
                    'volatility': segment_returns.std(),
                    'skewness': segment_returns.skew(),
                    'kurtosis': segment_returns.kurtosis(),
                    'cumulative_return': segment_returns.sum()
                }
                
                segments.append(segment)
        
        return segments
    
    def _cluster_segments(self, segments: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Cluster segments to find recurring patterns."""
        
        if len(segments) < 5:
            return {}
        
        # Create feature matrix for segments
        segment_features = []
        for segment in segments:
            features = [
                segment['mean_return'],
                segment['volatility'],
                segment['skewness'],
                segment['kurtosis'],
                segment['length'],
                segment['cumulative_return']
            ]
            segment_features.append(features)
        
        segment_features_array = np.array(segment_features)
        
        # Standardize features
        scaler = StandardScaler()
        segment_features_scaled = scaler.fit_transform(segment_features_array)
        
        # Cluster segments
        n_clusters = min(4, len(segments) // 3)  # Reasonable number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(segment_features_scaled)
        
        # Group segments by cluster
        clustered_segments = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clustered_segments:
                clustered_segments[cluster_id] = []
            clustered_segments[cluster_id].append(segments[i])
        
        # Filter clusters with sufficient segments
        filtered_clusters = {
            cluster_id: segment_list 
            for cluster_id, segment_list in clustered_segments.items()
            if len(segment_list) >= 3
        }
        
        return filtered_clusters
    
    def _create_segment_pattern_labels(self, 
                                     segment_cluster: List[Dict[str, Any]],
                                     full_index: pd.Index) -> pd.Series:
        """Create binary labels for segment pattern."""
        
        labels = pd.Series(0, index=full_index)
        
        for segment in segment_cluster:
            start_idx = segment['start_idx']
            end_idx = segment['end_idx']
            
            # Mark segment periods as pattern periods
            if start_idx < len(labels) and end_idx <= len(labels):
                labels.iloc[start_idx:end_idx] = 1
        
        return labels
    
    def _analyze_segment_characteristics(self, segment_cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of segment cluster."""
        
        # Calculate average characteristics
        mean_returns = [seg['mean_return'] for seg in segment_cluster]
        volatilities = [seg['volatility'] for seg in segment_cluster]
        lengths = [seg['length'] for seg in segment_cluster]
        
        avg_return = np.mean(mean_returns)
        avg_volatility = np.mean(volatilities)
        avg_length = np.mean(lengths)
        
        # Determine pattern type
        if avg_return > 0.01:
            pattern_type = "bullish_segment"
            description = f"Bullish segments with avg return {avg_return:.3f}, avg length {avg_length:.1f}"
        elif avg_return < -0.01:
            pattern_type = "bearish_segment"
            description = f"Bearish segments with avg return {avg_return:.3f}, avg length {avg_length:.1f}"
        elif avg_volatility > np.median(volatilities) * 1.5:
            pattern_type = "high_volatility_segment"
            description = f"High volatility segments with avg vol {avg_volatility:.3f}, avg length {avg_length:.1f}"
        else:
            pattern_type = "consolidation_segment"
            description = f"Consolidation segments with low movement, avg length {avg_length:.1f}"
        
        # Calculate pattern strength (consistency within cluster)
        return_consistency = 1.0 - np.std(mean_returns) / (abs(avg_return) + 0.001)
        vol_consistency = 1.0 - np.std(volatilities) / (avg_volatility + 0.001)
        pattern_strength = (return_consistency + vol_consistency) / 2
        
        return {
            'pattern_type': pattern_type,
            'description': description,
            'strength': max(0, min(pattern_strength, 1.0)),
            'avg_return': avg_return,
            'avg_volatility': avg_volatility,
            'avg_length': avg_length,
            'cluster_size': len(segment_cluster)
        }
    
    def _test_segment_significance(self, segment_cluster: List[Dict[str, Any]]) -> Dict[str, float]:
        """Test statistical significance of segment pattern."""
        
        # Test if segment returns are significantly different from zero
        segment_returns = []
        for segment in segment_cluster:
            segment_returns.extend(segment['returns'].tolist())
        
        if len(segment_returns) < 10:
            return {'p_value': 1.0, 't_statistic': 0.0}
        
        try:
            t_stat, p_value = stats.ttest_1samp(segment_returns, 0)
            return {
                'p_value': float(p_value),
                't_statistic': float(t_stat),
                'mean_return': float(np.mean(segment_returns)),
                'return_std': float(np.std(segment_returns))
            }
        except:
            return {'p_value': 1.0, 't_statistic': 0.0}
    
    def _approximate_segment_formula(self, segment_cluster: List[Dict[str, Any]]) -> str:
        """Approximate segment pattern as mathematical formula."""
        
        if not segment_cluster:
            return "No pattern detected"
        
        avg_length = np.mean([seg['length'] for seg in segment_cluster])
        avg_return = np.mean([seg['mean_return'] for seg in segment_cluster])
        avg_volatility = np.mean([seg['volatility'] for seg in segment_cluster])
        
        if abs(avg_return) > 0.005:
            direction = "positive" if avg_return > 0 else "negative"
            return f"Segment with {direction} drift: mean_return ≈ {avg_return:.4f}, length ≈ {avg_length:.1f} periods"
        elif avg_volatility > 0.02:
            return f"High volatility segment: volatility ≈ {avg_volatility:.4f}, length ≈ {avg_length:.1f} periods"
        else:
            return f"Low activity segment: length ≈ {avg_length:.1f} periods, minimal movement"


class MLPatternDiscoveryOrchestrator:
    """Main orchestrator for ML-based pattern discovery."""
    
    def __init__(self):
        self.logger = system_logger.getChild('MLPatternDiscovery')
        
        # Initialize discovery methods
        self.discovery_methods = {
            'clustering': ClusteringBasedPatternDiscovery(),
            'anomaly_detection': AnomalyPatternDiscovery(),
            'change_point': ChangePointPatternDiscovery()
        }
    
    def discover_all_ml_patterns(self, 
                                market_data: pd.DataFrame,
                                methods: List[str] = None) -> Dict[str, List[MLDiscoveredPattern]]:
        """
        Discover patterns using all available ML methods.
        
        Args:
            market_data: OHLCV market data
            methods: List of methods to use (default: all available)
            
        Returns:
            Dictionary mapping method names to discovered patterns
        """
        
        if methods is None:
            methods = list(self.discovery_methods.keys())
        
        self.logger.info(f"🤖 Starting ML-based pattern discovery with methods: {methods}")
        
        all_discovered_patterns = {}
        
        for method_name in methods:
            if method_name not in self.discovery_methods:
                self.logger.warning(f"Unknown method: {method_name}")
                continue
            
            self.logger.info(f"🔍 Running {method_name} pattern discovery")
            
            try:
                discovery_method = self.discovery_methods[method_name]
                
                if method_name == 'clustering':
                    patterns = discovery_method.discover_patterns(market_data)
                elif method_name == 'anomaly_detection':
                    patterns = discovery_method.discover_anomaly_patterns(market_data)
                elif method_name == 'change_point':
                    patterns = discovery_method.discover_change_point_patterns(market_data)
                else:
                    patterns = []
                
                all_discovered_patterns[method_name] = patterns
                
                self.logger.info(f"   ✅ {method_name}: {len(patterns)} significant patterns discovered")
                
            except Exception as e:
                self.logger.error(f"   ❌ {method_name} failed: {e}")
                all_discovered_patterns[method_name] = []
        
        total_patterns = sum(len(patterns) for patterns in all_discovered_patterns.values())
        self.logger.info(f"🎯 ML pattern discovery completed: {total_patterns} total patterns")
        
        return all_discovered_patterns
    
    def generate_ml_pattern_report(self, 
                                 discovered_patterns: Dict[str, List[MLDiscoveredPattern]]) -> str:
        """Generate comprehensive report of ML-discovered patterns."""
        
        report = []
        report.append("# ML-Based Pattern Discovery Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        total_patterns = sum(len(patterns) for patterns in discovered_patterns.values())
        significant_patterns = sum(
            sum(1 for p in patterns if p.is_significant_pattern)
            for patterns in discovered_patterns.values()
        )
        
        report.append("## Discovery Summary")
        report.append("")
        report.append(f"- **Total Patterns Discovered**: {total_patterns}")
        report.append(f"- **Statistically Significant**: {significant_patterns}")
        report.append(f"- **Discovery Methods Used**: {len(discovered_patterns)}")
        report.append("")
        
        # Method-specific results
        for method_name, patterns in discovered_patterns.items():
            if not patterns:
                continue
            
            report.append(f"## {method_name.replace('_', ' ').title()} Method")
            report.append("")
            
            significant_patterns_method = [p for p in patterns if p.is_significant_pattern]
            
            report.append(f"**Patterns Found**: {len(patterns)}")
            report.append(f"**Significant Patterns**: {len(significant_patterns_method)}")
            report.append("")
            
            for pattern in significant_patterns_method:
                report.append(f"### {pattern.pattern_id}")
                report.append("")
                report.append(f"**Description**: {pattern.pattern_description}")
                report.append(f"**Frequency**: {pattern.frequency:.3f} ({pattern.frequency*100:.1f}% of periods)")
                report.append(f"**Pattern Strength**: {pattern.pattern_strength:.3f}")
                
                if pattern.statistical_significance.get('p_value'):
                    p_val = pattern.statistical_significance['p_value']
                    report.append(f"**Statistical Significance**: p={p_val:.3f}")
                
                report.append(f"**Mathematical Approximation**: {pattern.mathematical_approximation}")
                report.append("")
        
        # Recommendations
        report.append("## ML Pattern Discovery Recommendations")
        report.append("")
        
        if significant_patterns > 0:
            report.append("✅ **Significant ML-Discovered Patterns Found**")
            report.append("- These patterns complement traditional technical analysis")
            report.append("- Consider incorporating into trading strategy development")
            report.append("- Validate economic significance through backtesting")
            report.append("")
            
            report.append("**Next Steps:**")
            report.append("1. Combine ML-discovered patterns with mathematical pattern definitions")
            report.append("2. Test predictive power using market dimension features")
            report.append("3. Develop trading strategies based on pattern combinations")
            report.append("4. Validate on out-of-sample data")
        else:
            report.append("❌ **No Significant ML Patterns Found**")
            report.append("**Possible Reasons:**")
            report.append("- Market data may not contain discoverable patterns")
            report.append("- Pattern discovery parameters may need adjustment")
            report.append("- Longer time series or different timeframes may be needed")
            report.append("")
            
            report.append("**Recommendations:**")
            report.append("1. Adjust discovery parameters (window sizes, thresholds)")
            report.append("2. Try different timeframes (hourly, daily, weekly)")
            report.append("3. Include additional market data (volume, volatility)")
            report.append("4. Focus on mathematical pattern definitions")
        
        return "\n".join(report)
    
    def combine_with_mathematical_patterns(self, 
                                         ml_patterns: Dict[str, List[MLDiscoveredPattern]],
                                         mathematical_patterns: Dict[str, Any]) -> pd.DataFrame:
        """Combine ML-discovered patterns with mathematical pattern definitions."""
        
        all_pattern_labels = {}
        
        # Add mathematical patterns
        if 'pattern_labels' in mathematical_patterns:
            for pattern_name, labels in mathematical_patterns['pattern_labels'].items():
                all_pattern_labels[f"math_{pattern_name}"] = labels
        
        # Add ML-discovered patterns
        for method_name, patterns in ml_patterns.items():
            for pattern in patterns:
                if pattern.is_significant_pattern:
                    all_pattern_labels[f"ml_{pattern.pattern_id}"] = pattern.pattern_labels
        
        # Combine into single DataFrame
        if all_pattern_labels:
            combined_df = pd.DataFrame(all_pattern_labels)
            
            # Add interaction patterns
            math_patterns = [col for col in combined_df.columns if col.startswith('math_')]
            ml_patterns = [col for col in combined_df.columns if col.startswith('ml_')]
            
            # Create combination patterns
            if len(math_patterns) > 0 and len(ml_patterns) > 0:
                combined_df['math_and_ml'] = (
                    combined_df[math_patterns].max(axis=1) * 
                    combined_df[ml_patterns].max(axis=1)
                )
                
                combined_df['math_or_ml'] = (
                    combined_df[math_patterns].max(axis=1) + 
                    combined_df[ml_patterns].max(axis=1)
                ).clip(0, 1)
            
            return combined_df
        else:
            return pd.DataFrame()


# Example usage
def run_ml_pattern_discovery_example():
    """Example of ML-based pattern discovery."""
    
    print("ML-Based Pattern Discovery Framework")
    print("===================================")
    print()
    print("This framework automatically discovers patterns using:")
    print("1. Clustering-Based Discovery - Find recurring price sequence shapes")
    print("2. Anomaly Detection - Identify unusual market conditions")
    print("3. Change Point Detection - Find structural breaks and regime changes")
    print()
    print("Benefits:")
    print("- Discovers patterns not captured by traditional technical analysis")
    print("- Data-driven approach reduces human bias")
    print("- Automatically validates pattern significance")
    print("- Generates mathematical approximations of discovered patterns")
    print()
    print("Usage:")
    print("```python")
    print("orchestrator = MLPatternDiscoveryOrchestrator()")
    print("ml_patterns = orchestrator.discover_all_ml_patterns(market_data)")
    print("report = orchestrator.generate_ml_pattern_report(ml_patterns)")
    print("combined = orchestrator.combine_with_mathematical_patterns(ml_patterns, math_patterns)")
    print("```")


if __name__ == "__main__":
    run_ml_pattern_discovery_example()