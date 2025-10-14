"""
Similarity Matrix Clustering with CV Confirmation

This module replaces traditional KMeans/GMM clustering with a similarity matrix approach
that uses coefficient of variation (CV) confirmation for regime discovery.

Key Features:
- Feature similarity matrix calculation
- Hierarchical similarity-based clustering
- CV-based cluster validation and merging
- Data-driven threshold discovery
- Economic relevance validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

from src.utils.logger import system_logger


class SimilarityMethod(Enum):
    """Methods for calculating feature similarity."""
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"
    DISTANCE_CORRELATION = "distance_correlation"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    COMBINED = "combined"


@dataclass
class SimilarityClusteringConfig:
    """Configuration for similarity matrix clustering."""
    # Similarity calculation
    similarity_method: SimilarityMethod = SimilarityMethod.CORRELATION
    similarity_threshold: float = 0.7
    min_similarity: float = 0.3
    
    # CV validation
    cv_threshold: float = 0.3
    max_cv_threshold: float = 0.8
    min_samples_per_cluster: int = 50
    
    # Hierarchical clustering
    linkage_method: str = "ward"
    distance_threshold: Optional[float] = None
    
    # Economic validation
    enable_economic_validation: bool = True
    min_economic_significance: float = 0.1
    
    # Data preprocessing
    standardize_features: bool = True
    handle_missing: str = "drop"  # "drop", "fill", "interpolate"


@dataclass
class ClusterValidationResult:
    """Result container for cluster validation."""
    cluster_id: int
    n_samples: int
    cv_score: float
    similarity_score: float
    economic_significance: float
    is_valid: bool
    merge_candidates: List[int]
    validation_metrics: Dict[str, float]


@dataclass
class SimilarityClusteringResult:
    """Result container for similarity clustering."""
    labels: np.ndarray
    n_clusters: int
    similarity_matrix: np.ndarray
    cluster_validations: List[ClusterValidationResult]
    final_cv_scores: Dict[int, float]
    final_similarity_scores: Dict[int, float]
    economic_significance_scores: Dict[int, float]
    metadata: Dict[str, Any]


class SimilarityMatrixClusterer:
    """
    Similarity matrix clustering with CV confirmation.
    
    This class implements a data-driven approach to clustering that:
    1. Calculates feature similarity matrices
    2. Performs hierarchical clustering based on similarity
    3. Validates clusters using coefficient of variation
    4. Merges clusters that exceed CV thresholds
    5. Validates economic significance
    """
    
    def __init__(self, config: Optional[SimilarityClusteringConfig] = None):
        self.config = config or SimilarityClusteringConfig()
        self.logger = system_logger.getChild('SimilarityMatrixClusterer')
        
    def fit_predict(self, 
                   features: pd.DataFrame, 
                   price_data: Optional[pd.DataFrame] = None) -> SimilarityClusteringResult:
        """
        Perform similarity matrix clustering with CV confirmation.
        
        Args:
            features: Feature matrix for clustering
            price_data: Optional price data for economic validation
            
        Returns:
            Clustering result with validation metrics
        """
        self.logger.info("🔍 Starting similarity matrix clustering with CV confirmation")
        
        # Preprocess features
        processed_features = self._preprocess_features(features)
        
        # Calculate similarity matrix
        self.logger.info("📊 Calculating feature similarity matrix")
        similarity_matrix = self._calculate_similarity_matrix(processed_features)
        
        # Initial hierarchical clustering
        self.logger.info("🌳 Performing hierarchical similarity clustering")
        initial_labels = self._hierarchical_similarity_clustering(similarity_matrix)
        
        # CV validation and cluster merging
        self.logger.info("✅ Validating clusters with CV confirmation")
        validated_labels, cluster_validations = self._cv_confirmation_and_merging(
            processed_features, initial_labels, price_data
        )
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(processed_features, validated_labels)
        
        # Create result
        result = SimilarityClusteringResult(
            labels=validated_labels,
            n_clusters=len(np.unique(validated_labels)),
            similarity_matrix=similarity_matrix,
            cluster_validations=cluster_validations,
            final_cv_scores=final_metrics['cv_scores'],
            final_similarity_scores=final_metrics['similarity_scores'],
            economic_significance_scores=final_metrics.get('economic_scores', {}),
            metadata={
                'config': self.config,
                'n_original_features': len(features.columns),
                'n_processed_features': len(processed_features.columns),
                'similarity_method': self.config.similarity_method.value,
                'linkage_method': self.config.linkage_method
            }
        )
        
        self.logger.info(f"✅ Clustering completed: {result.n_clusters} final clusters")
        return result
    
    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for clustering."""
        processed = features.copy()
        
        # Handle missing values
        if self.config.handle_missing == "drop":
            processed = processed.dropna(axis=1, how='any')
        elif self.config.handle_missing == "fill":
            processed = processed.fillna(processed.mean())
        elif self.config.handle_missing == "interpolate":
            processed = processed.interpolate().fillna(processed.mean())
        
        # Standardize if requested
        if self.config.standardize_features:
            scaler = StandardScaler()
            processed = pd.DataFrame(
                scaler.fit_transform(processed),
                columns=processed.columns,
                index=processed.index
            )
        
        # Remove constant features
        constant_features = processed.columns[processed.std() == 0]
        if len(constant_features) > 0:
            self.logger.warning(f"Removing {len(constant_features)} constant features")
            processed = processed.drop(columns=constant_features)
        
        return processed
    
    def _calculate_similarity_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate similarity matrix between features."""
        
        if self.config.similarity_method == SimilarityMethod.CORRELATION:
            similarity_matrix = features.corr().abs().values
            
        elif self.config.similarity_method == SimilarityMethod.SPEARMAN:
            similarity_matrix = features.corr(method='spearman').abs().values
            
        elif self.config.similarity_method == SimilarityMethod.KENDALL:
            similarity_matrix = features.corr(method='kendall').abs().values
            
        elif self.config.similarity_method == SimilarityMethod.MUTUAL_INFORMATION:
            similarity_matrix = self._calculate_mutual_information_matrix(features)
            
        elif self.config.similarity_method == SimilarityMethod.DISTANCE_CORRELATION:
            similarity_matrix = self._calculate_distance_correlation_matrix(features)
            
        elif self.config.similarity_method == SimilarityMethod.COMBINED:
            # Combine multiple similarity measures
            corr_sim = features.corr().abs().values
            spearman_sim = features.corr(method='spearman').abs().values
            
            # Weighted combination
            similarity_matrix = 0.6 * corr_sim + 0.4 * spearman_sim
            
        else:
            raise ValueError(f"Unsupported similarity method: {self.config.similarity_method}")
        
        # Handle NaN values
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        
        # Ensure diagonal is 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _calculate_mutual_information_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate mutual information similarity matrix."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            n_features = features.shape[1]
            mi_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        mi_matrix[i, j] = 1.0
                    elif i < j:  # Calculate only upper triangle
                        mi_score = mutual_info_regression(
                            features.iloc[:, [i]], features.iloc[:, j], 
                            discrete_features=False, random_state=42
                        )[0]
                        # Normalize MI score to [0, 1] range
                        mi_matrix[i, j] = mi_matrix[j, i] = min(1.0, mi_score)
            
            return mi_matrix
            
        except ImportError:
            self.logger.warning("scikit-learn not available for MI, falling back to correlation")
            return features.corr().abs().values
    
    def _calculate_distance_correlation_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate distance correlation similarity matrix."""
        try:
            
            # Use distance correlation approximation
            n_features = features.shape[1]
            dc_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        dc_matrix[i, j] = 1.0
                    elif i < j:
                        # Simplified distance correlation
                        x, y = features.iloc[:, i], features.iloc[:, j]
                        dc_score = self._distance_correlation(x, y)
                        dc_matrix[i, j] = dc_matrix[j, i] = dc_score
            
            return dc_matrix
            
        except Exception as e:
            self.logger.warning(f"Distance correlation failed: {e}, falling back to correlation")
            return features.corr().abs().values
    
    def _distance_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate distance correlation between two series."""
        try:
            x_vals, y_vals = x.values, y.values
            n = len(x_vals)
            
            if n < 4:  # Need minimum samples
                return 0.0
            
            # Distance matrices
            a = np.abs(x_vals[:, None] - x_vals[None, :])
            b = np.abs(y_vals[:, None] - y_vals[None, :])
            
            # Centered distance matrices
            A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
            
            # Distance covariance and variances
            dcov_xy = np.sqrt((A * B).mean())
            dcov_xx = np.sqrt((A * A).mean())
            dcov_yy = np.sqrt((B * B).mean())
            
            # Distance correlation
            if dcov_xx > 0 and dcov_yy > 0:
                dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
                return min(1.0, max(0.0, dcor))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _hierarchical_similarity_clustering(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering based on similarity matrix."""
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Ensure distance matrix is valid
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        
        # Convert to condensed form for scipy
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method=self.config.linkage_method)
        
        # Determine number of clusters
        if self.config.distance_threshold is not None:
            labels = fcluster(linkage_matrix, self.config.distance_threshold, criterion='distance')
        else:
            # Use similarity threshold to determine clusters
            distance_threshold = 1.0 - self.config.similarity_threshold
            labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        
        # Convert to 0-based indexing
        labels = labels - 1
        
        return labels
    
    def _cv_confirmation_and_merging(self, 
                                   features: pd.DataFrame,
                                   initial_labels: np.ndarray,
                                   price_data: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, List[ClusterValidationResult]]:
        """Validate clusters using CV and merge those exceeding thresholds."""
        
        labels = initial_labels.copy()
        cluster_validations = []
        merge_operations = []
        
        # Validate each cluster
        unique_clusters = np.unique(labels)
        self.logger.info(f"Validating {len(unique_clusters)} initial clusters")
        
        for cluster_id in unique_clusters:
            validation = self._validate_cluster(features, labels, cluster_id, price_data)
            cluster_validations.append(validation)
            
            if not validation.is_valid:
                merge_operations.append((cluster_id, validation.merge_candidates))
        
        # Perform merge operations
        labels = self._perform_merge_operations(labels, merge_operations)
        
        # Re-validate after merging
        final_validations = []
        for cluster_id in np.unique(labels):
            validation = self._validate_cluster(features, labels, cluster_id, price_data)
            final_validations.append(validation)
        
        self.logger.info(f"Final clusters after merging: {len(np.unique(labels))}")
        
        return labels, final_validations
    
    def _validate_cluster(self, 
                         features: pd.DataFrame,
                         labels: np.ndarray,
                         cluster_id: int,
                         price_data: Optional[pd.DataFrame] = None) -> ClusterValidationResult:
        """Validate a single cluster using multiple criteria."""
        
        cluster_mask = labels == cluster_id
        cluster_features = features[cluster_mask]
        n_samples = len(cluster_features)
        
        # Calculate CV score
        cv_score = self._calculate_cluster_cv(cluster_features)
        
        # Calculate similarity score
        similarity_score = self._calculate_cluster_similarity(cluster_features)
        
        # Calculate economic significance
        economic_significance = 0.0
        if price_data is not None and self.config.enable_economic_validation:
            economic_significance = self._calculate_economic_significance(
                cluster_features, price_data[cluster_mask], labels, cluster_id
            )
        
        # Determine if cluster is valid
        is_valid = (
            n_samples >= self.config.min_samples_per_cluster and
            cv_score <= self.config.cv_threshold and
            similarity_score >= self.config.min_similarity
        )
        
        # Find merge candidates if not valid
        merge_candidates = []
        if not is_valid:
            merge_candidates = self._find_merge_candidates(
                features, labels, cluster_id, cv_score, similarity_score
            )
        
        return ClusterValidationResult(
            cluster_id=cluster_id,
            n_samples=n_samples,
            cv_score=cv_score,
            similarity_score=similarity_score,
            economic_significance=economic_significance,
            is_valid=is_valid,
            merge_candidates=merge_candidates,
            validation_metrics={
                'cv_score': cv_score,
                'similarity_score': similarity_score,
                'economic_significance': economic_significance,
                'n_samples': n_samples
            }
        )
    
    def _calculate_cluster_cv(self, cluster_features: pd.DataFrame) -> float:
        """Calculate within-cluster coefficient of variation."""
        
        if len(cluster_features) < 2:
            return float('inf')  # Invalid cluster
        
        # Calculate CV for each feature within the cluster
        feature_cvs = []
        for col in cluster_features.columns:
            feature_values = cluster_features[col]
            if feature_values.std() > 0 and feature_values.mean() != 0:
                cv = abs(feature_values.std() / feature_values.mean())
                feature_cvs.append(cv)
        
        # Return mean within-cluster CV across features
        return np.mean(feature_cvs) if feature_cvs else float('inf')
    
    def _calculate_between_cluster_cv(self, 
                                    features: pd.DataFrame,
                                    labels: np.ndarray) -> Dict[str, float]:
        """Calculate between-cluster coefficient of variation for each feature."""
        
        between_cluster_cvs = {}
        
        for col in features.columns:
            # Calculate cluster centroids for this feature
            cluster_centroids = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = features[col][cluster_mask]
                if len(cluster_data) > 0:
                    cluster_centroids.append(cluster_data.mean())
            
            # Calculate CV of cluster centroids
            if len(cluster_centroids) > 1:
                centroids_array = np.array(cluster_centroids)
                if centroids_array.std() > 0 and centroids_array.mean() != 0:
                    between_cv = abs(centroids_array.std() / centroids_array.mean())
                    between_cluster_cvs[col] = between_cv
                else:
                    between_cluster_cvs[col] = 0.0
            else:
                between_cluster_cvs[col] = 0.0
        
        return between_cluster_cvs
    
    def _calculate_comprehensive_cv_metrics(self,
                                          features: pd.DataFrame,
                                          labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive CV metrics: within-cluster, between-cluster, and ratios."""
        
        cv_metrics = {
            'within_cluster_cvs': {},
            'between_cluster_cvs': {},
            'cv_ratios': {},
            'overall_within_cv': 0.0,
            'overall_between_cv': 0.0,
            'cv_separation_score': 0.0
        }
        
        # Calculate within-cluster CVs for each cluster
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            within_cv = self._calculate_cluster_cv(cluster_features)
            cv_metrics['within_cluster_cvs'][cluster_id] = within_cv
        
        # Calculate between-cluster CVs for each feature
        cv_metrics['between_cluster_cvs'] = self._calculate_between_cluster_cv(features, labels)
        
        # Calculate CV ratios (between/within) - higher is better for clustering
        for col in features.columns:
            between_cv = cv_metrics['between_cluster_cvs'].get(col, 0.0)
            
            # Calculate average within-cluster CV for this feature across clusters
            feature_within_cvs = []
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = features[col][cluster_mask]
                if len(cluster_data) > 1 and cluster_data.std() > 0 and cluster_data.mean() != 0:
                    within_cv = abs(cluster_data.std() / cluster_data.mean())
                    feature_within_cvs.append(within_cv)
            
            avg_within_cv = np.mean(feature_within_cvs) if feature_within_cvs else float('inf')
            
            # CV ratio: between/within (higher = better separation)
            if avg_within_cv > 0 and avg_within_cv != float('inf'):
                cv_ratio = between_cv / avg_within_cv
                cv_metrics['cv_ratios'][col] = cv_ratio
            else:
                cv_metrics['cv_ratios'][col] = 0.0
        
        # Overall metrics
        valid_within_cvs = [cv for cv in cv_metrics['within_cluster_cvs'].values() if cv != float('inf')]
        cv_metrics['overall_within_cv'] = np.mean(valid_within_cvs) if valid_within_cvs else float('inf')
        
        valid_between_cvs = [cv for cv in cv_metrics['between_cluster_cvs'].values() if cv > 0]
        cv_metrics['overall_between_cv'] = np.mean(valid_between_cvs) if valid_between_cvs else 0.0
        
        # CV separation score: how well clusters separate in CV space
        valid_ratios = [ratio for ratio in cv_metrics['cv_ratios'].values() if ratio > 0]
        cv_metrics['cv_separation_score'] = np.mean(valid_ratios) if valid_ratios else 0.0
        
        return cv_metrics
    
    def _calculate_cluster_similarity(self, cluster_features: pd.DataFrame) -> float:
        """Calculate internal similarity within cluster."""
        
        if len(cluster_features) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        corr_matrix = cluster_features.corr().abs()
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        ).stack()
        
        # Return mean correlation
        return upper_triangle.mean() if not upper_triangle.empty else 0.0
    
    def _calculate_economic_significance(self, 
                                       cluster_features: pd.DataFrame,
                                       cluster_price_data: pd.DataFrame,
                                       all_labels: np.ndarray,
                                       cluster_id: int) -> float:
        """Calculate economic significance of cluster."""
        
        try:
            if 'close' not in cluster_price_data.columns:
                return 0.0
            
            # Calculate cluster returns
            cluster_returns = cluster_price_data['close'].pct_change().dropna()
            
            if len(cluster_returns) < 10:
                return 0.0
            
            # Compare with other clusters
            other_clusters_mask = all_labels != cluster_id
            if not np.any(other_clusters_mask):
                return 0.0
            
            # Calculate information ratio difference
            cluster_ir = cluster_returns.mean() / cluster_returns.std() if cluster_returns.std() > 0 else 0
            
            # Simplified economic significance
            return abs(cluster_ir)
            
        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return 0.0
    
    def _find_merge_candidates(self, 
                              features: pd.DataFrame,
                              labels: np.ndarray,
                              cluster_id: int,
                              cv_score: float,
                              similarity_score: float) -> List[int]:
        """Find best candidates for merging with invalid cluster."""
        
        cluster_mask = labels == cluster_id
        cluster_features = features[cluster_mask]
        
        candidates = []
        unique_clusters = np.unique(labels)
        
        for other_id in unique_clusters:
            if other_id == cluster_id:
                continue
                
            other_mask = labels == other_id
            other_features = features[other_mask]
            
            # Calculate similarity between clusters
            inter_cluster_similarity = self._calculate_inter_cluster_similarity(
                cluster_features, other_features
            )
            
            # Calculate merged CV
            merged_features = pd.concat([cluster_features, other_features])
            merged_cv = self._calculate_cluster_cv(merged_features)
            
            # Check if merging would improve validation
            if (inter_cluster_similarity >= self.config.min_similarity and 
                merged_cv <= self.config.cv_threshold):
                candidates.append(other_id)
        
        # Sort by similarity (descending)
        candidates.sort(key=lambda x: self._calculate_inter_cluster_similarity(
            cluster_features, features[labels == x]
        ), reverse=True)
        
        return candidates[:3]  # Return top 3 candidates
    
    def _calculate_inter_cluster_similarity(self, 
                                          cluster1_features: pd.DataFrame,
                                          cluster2_features: pd.DataFrame) -> float:
        """Calculate similarity between two clusters."""
        
        try:
            # Calculate mean features for each cluster
            mean1 = cluster1_features.mean()
            mean2 = cluster2_features.mean()
            
            # Calculate correlation between cluster centroids
            correlation = np.corrcoef(mean1, mean2)[0, 1]
            
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _perform_merge_operations(self, 
                                labels: np.ndarray,
                                merge_operations: List[Tuple[int, List[int]]]) -> np.ndarray:
        """Perform cluster merge operations."""
        
        merged_labels = labels.copy()
        
        for cluster_id, candidates in merge_operations:
            if not candidates:
                continue
                
            # Merge with best candidate
            best_candidate = candidates[0]
            merged_labels[merged_labels == cluster_id] = best_candidate
            
            self.logger.info(f"Merged cluster {cluster_id} into cluster {best_candidate}")
        
        # Relabel clusters to be consecutive
        unique_labels = np.unique(merged_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        for old_label, new_label in label_mapping.items():
            merged_labels[merged_labels == old_label] = new_label
        
        return merged_labels
    
    def _calculate_final_metrics(self, 
                               features: pd.DataFrame,
                               labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive final metrics including within/between-cluster CV."""
        
        # Basic cluster metrics
        basic_metrics = {
            'cv_scores': {},
            'similarity_scores': {},
            'sample_counts': {}
        }
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            
            basic_metrics['cv_scores'][cluster_id] = self._calculate_cluster_cv(cluster_features)
            basic_metrics['similarity_scores'][cluster_id] = self._calculate_cluster_similarity(cluster_features)
            basic_metrics['sample_counts'][cluster_id] = len(cluster_features)
        
        # Comprehensive CV analysis
        comprehensive_cv = self._calculate_comprehensive_cv_metrics(features, labels)
        
        # Combine all metrics
        final_metrics = {
            **basic_metrics,
            'comprehensive_cv_analysis': comprehensive_cv,
            'within_cluster_cvs': comprehensive_cv['within_cluster_cvs'],
            'between_cluster_cvs': comprehensive_cv['between_cluster_cvs'],
            'cv_ratios': comprehensive_cv['cv_ratios'],
            'overall_within_cv': comprehensive_cv['overall_within_cv'],
            'overall_between_cv': comprehensive_cv['overall_between_cv'],
            'cv_separation_score': comprehensive_cv['cv_separation_score']
        }
        
        return final_metrics


def similarity_matrix_clustering(features: pd.DataFrame,
                                price_data: Optional[pd.DataFrame] = None,
                                config: Optional[SimilarityClusteringConfig] = None) -> SimilarityClusteringResult:
    """
    Convenience function for similarity matrix clustering.
    
    Args:
        features: Feature matrix for clustering
        price_data: Optional price data for economic validation
        config: Optional configuration
        
    Returns:
        Clustering result
    """
    clusterer = SimilarityMatrixClusterer(config)
    return clusterer.fit_predict(features, price_data)


# Example usage
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # Create correlated feature groups (simulating market dimensions)
    data = []
    
    # Group 1: High correlation (momentum features)
    base_momentum = np.random.randn(n_samples)
    for i in range(5):
        feature = base_momentum + np.random.randn(n_samples) * 0.3
        data.append(feature)
    
    # Group 2: High correlation (volatility features)  
    base_vol = np.random.randn(n_samples)
    for i in range(5):
        feature = base_vol + np.random.randn(n_samples) * 0.3
        data.append(feature)
    
    # Group 3: Medium correlation (volume features)
    base_volume = np.random.randn(n_samples)
    for i in range(5):
        feature = base_volume + np.random.randn(n_samples) * 0.6
        data.append(feature)
    
    # Group 4: Low correlation (noise features)
    for i in range(5):
        feature = np.random.randn(n_samples)
        data.append(feature)
    
    # Create DataFrame
    features = pd.DataFrame(
        np.column_stack(data),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create price data
    returns = np.random.randn(n_samples) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    price_data = pd.DataFrame({
        'close': prices,
        'returns': returns
    })
    
    # Test clustering
    config = SimilarityClusteringConfig(
        similarity_method=SimilarityMethod.CORRELATION,
        similarity_threshold=0.6,
        cv_threshold=0.4,
        min_samples_per_cluster=50
    )
    
    result = similarity_matrix_clustering(features, price_data, config)
    
    print(f"🎯 Similarity Matrix Clustering Results:")
    print(f"Number of clusters: {result.n_clusters}")
    print(f"Cluster sizes: {np.bincount(result.labels)}")
    
    print(f"\n📊 Cluster Validation:")
    for validation in result.cluster_validations:
        print(f"Cluster {validation.cluster_id}:")
        print(f"  - Samples: {validation.n_samples}")
        print(f"  - CV Score: {validation.cv_score:.3f}")
        print(f"  - Similarity: {validation.similarity_score:.3f}")
        print(f"  - Valid: {validation.is_valid}")
        if validation.merge_candidates:
            print(f"  - Merge candidates: {validation.merge_candidates}")