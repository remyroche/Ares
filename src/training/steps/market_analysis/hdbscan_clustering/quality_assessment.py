"""
Comprehensive Clustering Quality Assessment

This module provides proper DBCV calculation and comprehensive quality assessment
for HDBSCAN clustering results with temporal stability validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# Import HDBSCAN for proper DBCV calculation
try:
    import hdbscan
    from hdbscan import validity
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None
    validity = None

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for clustering quality metrics."""
    # Core clustering metrics
    dbcv_score: Optional[float] = None
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    
    # Temporal and economic metrics
    temporal_stability: Optional[float] = None
    economic_separation: Optional[float] = None
    cluster_persistence: Optional[float] = None
    
    # Predictive power and composite metrics
    predictive_power: Optional[float] = None
    composite_quality_score: Optional[float] = None
    
    # Cluster composition
    noise_ratio: Optional[float] = None
    n_clusters: int = 0
    n_noise_points: int = 0
    cluster_sizes: Optional[List[int]] = None
    cluster_size_ratios: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'dbcv_score': self.dbcv_score,
            'silhouette_score': self.silhouette_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'temporal_stability': self.temporal_stability,
            'economic_separation': self.economic_separation,
            'cluster_persistence': self.cluster_persistence,
            'predictive_power': self.predictive_power,
            'composite_quality_score': self.composite_quality_score,
            'noise_ratio': self.noise_ratio,
            'n_clusters': self.n_clusters,
            'n_noise_points': self.n_noise_points,
            'cluster_sizes': self.cluster_sizes,
            'cluster_size_ratios': self.cluster_size_ratios
        }


class DBCVCalculator:
    """Calculator for Density-Based Clustering Validation (DBCV) score."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_dbcv(self, clusterer: Any, cluster_labels: np.ndarray) -> Optional[float]:
        """
        Calculate DBCV score for HDBSCAN clustering results.
        
        Args:
            clusterer: Fitted HDBSCAN clusterer with condensed_tree and minimum_spanning_tree
            cluster_labels: Cluster labels from HDBSCAN
            
        Returns:
            DBCV score or None if calculation fails
        """
        if not HDBSCAN_AVAILABLE:
            self.logger.warning("HDBSCAN not available for DBCV calculation")
            return None
        
        try:
            # Check if clusterer has required attributes
            if not hasattr(clusterer, 'condensed_tree_') or clusterer.condensed_tree_ is None:
                self.logger.warning("Clusterer missing condensed_tree_ attribute")
                return None
            
            if not hasattr(clusterer, 'minimum_spanning_tree_') or clusterer.minimum_spanning_tree_ is None:
                self.logger.warning("Clusterer missing minimum_spanning_tree_ attribute")
                return None
            
            # Calculate DBCV using HDBSCAN's validity module
            dbcv_score = validity.validity_index(
                clusterer.condensed_tree_,
                clusterer.minimum_spanning_tree_,
                cluster_labels
            )
            
            self.logger.info(f"DBCV score calculated: {dbcv_score:.4f}")
            return dbcv_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate DBCV: {e}")
            return None
    
    def calculate_approximate_dbcv(self, features: np.ndarray, cluster_labels: np.ndarray) -> Optional[float]:
        """
        Calculate approximate DBCV score when full HDBSCAN artifacts are not available.
        
        Args:
            features: Feature matrix used for clustering
            cluster_labels: Cluster labels from clustering
            
        Returns:
            Approximate DBCV score or None if calculation fails
        """
        try:
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            
            if n_clusters < 2:
                return None
            
            # Calculate density-based metrics
            densities = self._calculate_cluster_densities(features, cluster_labels)
            separations = self._calculate_cluster_separations(features, cluster_labels)
            
            # Approximate DBCV as ratio of average density to average separation
            avg_density = np.mean(densities)
            avg_separation = np.mean(separations)
            
            if avg_separation == 0:
                return None
            
            approximate_dbcv = avg_density / avg_separation
            self.logger.info(f"Approximate DBCV score calculated: {approximate_dbcv:.4f}")
            return approximate_dbcv
            
        except Exception as e:
            self.logger.error(f"Failed to calculate approximate DBCV: {e}")
            return None
    
    def _calculate_cluster_densities(self, features: np.ndarray, cluster_labels: np.ndarray) -> List[float]:
        """Calculate density for each cluster."""
        densities = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == label
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) < 2:
                densities.append(0.0)
                continue
            
            # Calculate average distance to nearest neighbors within cluster
            nbrs = NearestNeighbors(n_neighbors=min(5, len(cluster_features))).fit(cluster_features)
            distances, _ = nbrs.kneighbors(cluster_features)
            avg_distance = np.mean(distances[:, 1:])  # Exclude self-distance
            
            # Density is inverse of average distance
            density = 1.0 / (avg_distance + 1e-8)
            densities.append(density)
        
        return densities
    
    def _calculate_cluster_separations(self, features: np.ndarray, cluster_labels: np.ndarray) -> List[float]:
        """Calculate separation between clusters."""
        separations = []
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]
        
        if len(valid_labels) < 2:
            return [0.0]
        
        # Calculate pairwise separations
        for i, label1 in enumerate(valid_labels):
            for label2 in valid_labels[i+1:]:
                cluster1_mask = cluster_labels == label1
                cluster2_mask = cluster_labels == label2
                
                cluster1_features = features[cluster1_mask]
                cluster2_features = features[cluster2_mask]
                
                # Calculate minimum distance between clusters
                from sklearn.metrics.pairwise import pairwise_distances
                distances = pairwise_distances(cluster1_features, cluster2_features)
                min_distance = np.min(distances)
                
                separations.append(min_distance)
        
        return separations


class TemporalStabilityValidator:
    """Validator for temporal stability of clustering results."""
    
    def __init__(self, min_regime_duration: int = 20, stability_window: int = 100):
        self.min_regime_duration = min_regime_duration
        self.stability_window = stability_window
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_temporal_stability(self, cluster_labels: np.ndarray, 
                                   timestamps: Optional[pd.Series] = None) -> float:
        """
        Calculate temporal stability of clustering results.
        
        Args:
            cluster_labels: Cluster labels from clustering
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Temporal stability score (0-1, higher is better)
        """
        try:
            # Calculate regime persistence
            persistence_score = self._calculate_regime_persistence(cluster_labels)
            
            # Calculate transition stability
            transition_score = self._calculate_transition_stability(cluster_labels)
            
            # Calculate overall temporal consistency
            consistency_score = self._calculate_temporal_consistency(cluster_labels)
            
            # Combine scores
            temporal_stability = (persistence_score + transition_score + consistency_score) / 3.0
            
            self.logger.info(f"Temporal stability calculated: {temporal_stability:.4f}")
            return temporal_stability
            
        except Exception as e:
            self.logger.error(f"Failed to calculate temporal stability: {e}")
            return 0.0
    
    def _calculate_regime_persistence(self, cluster_labels: np.ndarray) -> float:
        """Calculate how long regimes persist on average."""
        if len(cluster_labels) == 0:
            return 0.0
        
        # Find regime changes
        regime_changes = np.diff(cluster_labels) != 0
        change_indices = np.where(regime_changes)[0]
        
        if len(change_indices) == 0:
            # No changes - perfect persistence
            return 1.0
        
        # Calculate regime durations
        durations = []
        start_idx = 0
        
        for change_idx in change_indices:
            duration = change_idx - start_idx + 1
            durations.append(duration)
            start_idx = change_idx + 1
        
        # Add final duration
        final_duration = len(cluster_labels) - start_idx
        durations.append(final_duration)
        
        # Calculate persistence score based on minimum duration requirement
        avg_duration = np.mean(durations)
        persistence_score = min(1.0, avg_duration / self.min_regime_duration)
        
        return persistence_score
    
    def _calculate_transition_stability(self, cluster_labels: np.ndarray) -> float:
        """Calculate stability of regime transitions."""
        if len(cluster_labels) < 2:
            return 1.0
        
        # Count transitions
        transitions = np.sum(np.diff(cluster_labels) != 0)
        total_possible_transitions = len(cluster_labels) - 1
        
        # Calculate transition frequency
        transition_frequency = transitions / total_possible_transitions
        
        # Stability is inverse of transition frequency (lower is better)
        stability = max(0.0, 1.0 - transition_frequency)
        
        return stability
    
    def _calculate_temporal_consistency(self, cluster_labels: np.ndarray) -> float:
        """Calculate temporal consistency using sliding window analysis."""
        if len(cluster_labels) < self.stability_window:
            return 1.0
        
        consistency_scores = []
        
        # Slide window across the data
        for i in range(len(cluster_labels) - self.stability_window + 1):
            window_labels = cluster_labels[i:i + self.stability_window]
            
            # Calculate consistency within window
            unique_labels = np.unique(window_labels)
            n_clusters = len(unique_labels[unique_labels != -1])
            
            if n_clusters <= 1:
                consistency_scores.append(1.0)
            else:
                # Calculate entropy-based consistency
                label_counts = [np.sum(window_labels == label) for label in unique_labels if label != -1]
                probabilities = np.array(label_counts) / len(window_labels)
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                max_entropy = np.log(n_clusters)
                
                # Consistency is inverse of normalized entropy
                consistency = 1.0 - (entropy / max_entropy)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0


class EconomicSeparationCalculator:
    """Calculator for economic separation between regimes."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_economic_separation(self, cluster_labels: np.ndarray, 
                                    features_df: pd.DataFrame,
                                    returns: Optional[np.ndarray] = None) -> float:
        """
        Calculate economic separation between regimes.
        
        Args:
            cluster_labels: Cluster labels from clustering
            features_df: Feature DataFrame
            returns: Optional returns data for economic analysis
            
        Returns:
            Economic separation score (0-1, higher is better)
        """
        try:
            unique_labels = np.unique(cluster_labels)
            valid_labels = unique_labels[unique_labels != -1]
            
            if len(valid_labels) < 2:
                return 0.0
            
            # Calculate returns if not provided
            if returns is None:
                returns = self._extract_returns(features_df)
            
            if returns is None or len(returns) != len(cluster_labels):
                self.logger.warning("Could not extract returns for economic separation calculation")
                return 0.0
            
            # Calculate regime statistics
            regime_stats = self._calculate_regime_statistics(cluster_labels, returns)
            
            # Calculate pairwise economic differences
            economic_differences = []
            for i, regime1 in enumerate(regime_stats):
                for regime2 in regime_stats[i+1:]:
                    # Calculate return difference
                    return_diff = abs(regime1['avg_return'] - regime2['avg_return'])
                    
                    # Calculate volatility difference
                    vol_diff = abs(regime1['volatility'] - regime2['volatility'])
                    
                    # Calculate Sharpe ratio difference
                    sharpe_diff = abs(regime1['sharpe_ratio'] - regime2['sharpe_ratio'])
                    
                    # Normalize by combined volatility
                    combined_vol = (regime1['volatility'] + regime2['volatility']) / 2
                    if combined_vol > 0:
                        normalized_diff = return_diff / combined_vol
                        economic_differences.append(normalized_diff)
            
            # Economic separation is average normalized difference
            economic_separation = np.mean(economic_differences) if economic_differences else 0.0
            
            self.logger.info(f"Economic separation calculated: {economic_separation:.4f}")
            return economic_separation
            
        except Exception as e:
            self.logger.error(f"Failed to calculate economic separation: {e}")
            return 0.0
    
    def _extract_returns(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract returns from features DataFrame."""
        # Look for common return column names
        return_columns = ['returns', 'return', 'pct_change', 'close_pct_change']
        
        for col in return_columns:
            if col in features_df.columns:
                return features_df[col].values
        
        # Look for close price to calculate returns
        if 'close' in features_df.columns:
            close_prices = features_df['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]
            return np.concatenate([[0], returns])  # Add 0 for first observation
        
        return None
    
    def _calculate_regime_statistics(self, cluster_labels: np.ndarray, 
                                   returns: np.ndarray) -> List[Dict[str, float]]:
        """Calculate economic statistics for each regime."""
        regime_stats = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == label
            cluster_returns = returns[cluster_mask]
            
            if len(cluster_returns) == 0:
                continue
            
            avg_return = np.mean(cluster_returns)
            volatility = np.std(cluster_returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
            
            regime_stats.append({
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'cluster_size': len(cluster_returns)
            })
        
        return regime_stats


class PredictivePowerCalculator:
    """Calculator for predictive power of clustering results."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_predictive_power(self, cluster_labels: np.ndarray,
                                   forward_returns: np.ndarray) -> float:
        """
        Calculate predictive power: can current regime predict future returns?
        
        Uses Random Forest classifier to predict return sign from regime labels.
        
        Args:
            cluster_labels: Cluster labels from clustering
            forward_returns: Forward returns for prediction
            
        Returns:
            Predictive power score (cross-validation accuracy)
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            # Use current regime to predict next period's return sign
            if len(cluster_labels) < 10 or len(forward_returns) < 10:
                return 0.0
            
            # Ensure arrays are aligned and valid
            min_len = min(len(cluster_labels), len(forward_returns))
            if min_len < 10:
                return 0.0
            
            X = pd.get_dummies(cluster_labels[:min_len-1])
            y = (forward_returns[1:min_len] > 0).astype(int)
            
            if len(X) != len(y):
                return 0.0
            
            # Check if we have enough samples and variation
            if len(y) < 10 or len(set(y)) < 2:
                return 0.0
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_score = cross_val_score(rf, X, y, cv=min(5, len(y) // 2)).mean()
            
            self.logger.info(f"Predictive power calculated: {cv_score:.4f}")
            return float(cv_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate predictive power: {e}")
            return 0.0


class CompositeScoreCalculator:
    """Calculator for composite quality score."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_composite_score(self, metrics: 'QualityMetrics') -> float:
        """
        Calculate overall composite quality score (0 to 1, higher is better).
        
        Combines multiple metrics into a single score:
        - Silhouette score (weight: 0.25)
        - Davies-Bouldin Index (weight: 0.20)
        - Calinski-Harabasz Index (weight: 0.20)
        - DBCV score (weight: 0.15)
        - Temporal stability (weight: 0.10)
        - Noise ratio (weight: 0.10)
        
        Args:
            metrics: QualityMetrics object with computed metrics
            
        Returns:
            Composite quality score (0 to 1)
        """
        try:
            score_components = []
            weights = []
            
            # 1. Silhouette score (normalize to 0-1, already in [-1, 1])
            if metrics.silhouette_score is not None:
                silhouette_normalized = (metrics.silhouette_score + 1) / 2
                score_components.append(silhouette_normalized)
                weights.append(0.25)
            
            # 2. Davies-Bouldin Index (lower is better, normalize inversely)
            if metrics.davies_bouldin_score is not None and not np.isinf(metrics.davies_bouldin_score):
                dbi_normalized = 1.0 / (1.0 + metrics.davies_bouldin_score)
                score_components.append(dbi_normalized)
                weights.append(0.20)
            
            # 3. Calinski-Harabasz Index (higher is better, normalize)
            if metrics.calinski_harabasz_score is not None:
                ch_normalized = np.tanh(metrics.calinski_harabasz_score / 100)
                score_components.append(ch_normalized)
                weights.append(0.20)
            
            # 4. DBCV score (if available, already in [0, 1])
            if metrics.dbcv_score is not None:
                # DBCV is typically in range [-1, 1], normalize to [0, 1]
                dbcv_normalized = (metrics.dbcv_score + 1) / 2
                score_components.append(dbcv_normalized)
                weights.append(0.15)
            
            # 5. Temporal stability (already in [0, 1])
            if metrics.temporal_stability is not None:
                score_components.append(metrics.temporal_stability)
                weights.append(0.10)
            
            # 6. Noise ratio (lower is better, invert)
            if metrics.noise_ratio is not None:
                noise_score = 1.0 - metrics.noise_ratio
                score_components.append(noise_score)
                weights.append(0.10)
            
            # Calculate weighted average
            if len(score_components) > 0:
                total_weight = sum(weights)
                weighted_score = sum(s * w for s, w in zip(score_components, weights)) / total_weight
                
                self.logger.info(f"Composite quality score calculated: {weighted_score:.4f}")
                return float(weighted_score)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Failed to calculate composite score: {e}")
            return 0.0


class ComprehensiveQualityAssessor:
    """Comprehensive quality assessment for HDBSCAN clustering results."""
    
    def __init__(self):
        self.dbcv_calculator = DBCVCalculator()
        self.temporal_validator = TemporalStabilityValidator()
        self.economic_calculator = EconomicSeparationCalculator()
        self.predictive_power_calculator = PredictivePowerCalculator()
        self.composite_score_calculator = CompositeScoreCalculator()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def assess_clustering_quality(self, 
                                cluster_labels: np.ndarray,
                                features: np.ndarray,
                                clusterer: Optional[Any] = None,
                                timestamps: Optional[pd.Series] = None,
                                returns: Optional[np.ndarray] = None) -> QualityMetrics:
        """
        Perform comprehensive quality assessment of clustering results.
        
        Args:
            cluster_labels: Cluster labels from HDBSCAN
            features: Feature matrix used for clustering
            clusterer: Fitted HDBSCAN clusterer (for DBCV calculation)
            timestamps: Optional timestamps for temporal analysis
            returns: Optional returns data for economic analysis
            
        Returns:
            QualityMetrics object with all calculated metrics
        """
        start_time = time.time()
        
        # Initialize metrics
        metrics = QualityMetrics()
        
        # Basic cluster information
        unique_labels = np.unique(cluster_labels)
        metrics.n_clusters = len(unique_labels[unique_labels != -1])
        metrics.n_noise_points = np.sum(cluster_labels == -1)
        metrics.noise_ratio = metrics.n_noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
        
        # Calculate cluster sizes
        if metrics.n_clusters > 0:
            cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels if label != -1]
            metrics.cluster_sizes = cluster_sizes
            metrics.cluster_size_ratios = [size / len(cluster_labels) for size in cluster_sizes]
        
        # Calculate DBCV score
        if clusterer is not None:
            metrics.dbcv_score = self.dbcv_calculator.calculate_dbcv(clusterer, cluster_labels)
        
        if metrics.dbcv_score is None:
            # Fallback to approximate DBCV
            metrics.dbcv_score = self.dbcv_calculator.calculate_approximate_dbcv(features, cluster_labels)
        
        # Calculate standard clustering metrics
        if metrics.n_clusters > 1:
            try:
                metrics.silhouette_score = silhouette_score(features, cluster_labels)
                metrics.calinski_harabasz_score = calinski_harabasz_score(features, cluster_labels)
                metrics.davies_bouldin_score = davies_bouldin_score(features, cluster_labels)
            except Exception as e:
                self.logger.warning(f"Failed to calculate standard clustering metrics: {e}")
        
        # Calculate temporal stability
        metrics.temporal_stability = self.temporal_validator.calculate_temporal_stability(
            cluster_labels, timestamps
        )
        
        # Calculate economic separation
        if returns is not None:
            metrics.economic_separation = self.economic_calculator.calculate_economic_separation(
                cluster_labels, pd.DataFrame(features), returns
            )
        
        # Calculate cluster persistence
        metrics.cluster_persistence = self._calculate_cluster_persistence(cluster_labels)
        
        # Calculate predictive power (if returns provided)
        if returns is not None and len(returns) > 0:
            metrics.predictive_power = self.predictive_power_calculator.calculate_predictive_power(
                cluster_labels, returns
            )
        
        # Calculate composite quality score
        metrics.composite_quality_score = self.composite_score_calculator.calculate_composite_score(metrics)
        
        assessment_time = time.time() - start_time
        self.logger.info(f"Quality assessment completed in {assessment_time:.2f}s")
        self.logger.info(f"Composite quality score: {metrics.composite_quality_score:.4f}")
        
        return metrics
    
    def _calculate_cluster_persistence(self, cluster_labels: np.ndarray) -> float:
        """Calculate cluster persistence score."""
        if len(cluster_labels) == 0:
            return 0.0
        
        # Calculate how often clusters appear consecutively
        persistence_scores = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == label
            cluster_positions = np.where(cluster_mask)[0]
            
            if len(cluster_positions) < 2:
                persistence_scores.append(0.0)
                continue
            
            # Calculate average gap between consecutive cluster appearances
            gaps = np.diff(cluster_positions)
            avg_gap = np.mean(gaps)
            
            # Persistence is inverse of average gap (normalized)
            max_possible_gap = len(cluster_labels) - 1
            persistence = max(0.0, 1.0 - (avg_gap / max_possible_gap))
            persistence_scores.append(persistence)
        
        return np.mean(persistence_scores) if persistence_scores else 0.0


def create_quality_assessor() -> ComprehensiveQualityAssessor:
    """Factory function to create a quality assessor."""
    return ComprehensiveQualityAssessor()