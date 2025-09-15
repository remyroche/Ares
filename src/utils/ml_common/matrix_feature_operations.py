"""
Matrix Operations for Feature Selection

This module provides optimized matrix operations specifically designed for feature selection
tasks, leveraging hardware acceleration and efficient algorithms.

Author: AI Assistant
Date: 2024-01-XX
Version: 1.0.0
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr, spearmanr

# Import matrix operations from existing system
try:
    from ...utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    from ...utils.matrix_operations.vectorized_core import VectorizedCore
    from ...utils.matrix_operations.enhanced_operations import EnhancedOperations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class MatrixFeatureOperations:
    """
    Optimized matrix operations for feature selection.
    
    This class provides efficient implementations of common feature selection
    operations using optimized matrix computations.
    """
    
    def __init__(self, use_gpu: bool = True, use_parallel: bool = True):
        """Initialize matrix feature operations."""
        self.logger = logger.getChild('MatrixFeatureOperations')
        self.use_gpu = use_gpu
        self.use_parallel = use_parallel
        
        # Initialize matrix operations from existing system
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.unified_ops = UnifiedMatrixOperations()
                self.vectorized_core = VectorizedCore()
                self.enhanced_ops = EnhancedOperations()
                self.logger.info("✅ Matrix operations initialized from existing system")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize matrix operations: {e}")
                self.unified_ops = None
                self.vectorized_core = None
                self.enhanced_ops = None
        else:
            self.unified_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            self.logger.warning("⚠️ Matrix operations not available, using fallback implementations")
    
    def correlation_matrix(
        self,
        X: np.ndarray,
        method: str = "pearson",
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute correlation matrix efficiently.
        
        Args:
            X: Feature matrix
            method: Correlation method ("pearson" or "spearman")
            feature_names: List of feature names
            
        Returns:
            Correlation matrix
        """
        self.logger.debug(f"Computing {method} correlation matrix for {X.shape[1]} features")
        
        if self.matrix_ops and hasattr(self.matrix_ops, 'correlation_matrix'):
            try:
                # Use optimized matrix operations
                if isinstance(X, pd.DataFrame):
                    return self.matrix_ops.correlation_matrix(X, method=method)
                else:
                    # Convert to DataFrame for matrix operations
                    df = pd.DataFrame(X, columns=feature_names or [f'feature_{i}' for i in range(X.shape[1])])
                    return self.matrix_ops.correlation_matrix(df, method=method)
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations failed, using fallback: {e}")
        
        # Fallback implementation
        if method == "pearson":
            return np.corrcoef(X.T)
        elif method == "spearman":
            # Use pandas for spearman correlation
            df = pd.DataFrame(X, columns=feature_names or [f'feature_{i}' for i in range(X.shape[1])])
            return df.corr(method='spearman').values
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    
    def mutual_information_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute mutual information matrix between features and target.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Mutual information scores
        """
        self.logger.debug(f"Computing mutual information for {X.shape[1]} features")
        
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            # Determine if classification or regression
            unique_y = np.unique(y)
            if len(unique_y) <= 10:  # Classification
                mi_scores = mutual_info_classif(X, y)
            else:  # Regression
                mi_scores = mutual_info_regression(X, y)
            
            return mi_scores
            
        except ImportError:
            self.logger.warning("⚠️ sklearn not available, using correlation fallback")
            # Fallback to correlation
            mi_scores = []
            for i in range(X.shape[1]):
                corr, _ = pearsonr(X[:, i], y)
                mi_scores.append(abs(corr))
            return np.array(mi_scores)
    
    def hierarchical_clustering_correlation(
        self,
        X: np.ndarray,
        correlation_threshold: float = 0.95,
        method: str = "ward",
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform hierarchical clustering based on correlation matrix.
        
        Args:
            X: Feature matrix
            correlation_threshold: Threshold for correlation
            method: Clustering method
            feature_names: List of feature names
            
        Returns:
            Dictionary with clustering results
        """
        self.logger.debug(f"Performing hierarchical clustering with threshold {correlation_threshold}")
        
        # Compute correlation matrix
        corr_matrix = self.correlation_matrix(X, feature_names=feature_names)
        
        # Convert to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method=method)
        
        # Get clusters
        clusters = fcluster(linkage_matrix, 1 - correlation_threshold, criterion='distance')
        
        # Group features by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            
            feature_name = feature_names[i] if feature_names else f'feature_{i}'
            cluster_groups[cluster_id].append(feature_name)
        
        # Select representative features from each cluster
        representative_features = []
        for cluster_id, features in cluster_groups.items():
            if len(features) > 1:
                # Select feature with highest variance
                feature_indices = [feature_names.index(f) if feature_names else int(f.split('_')[1]) for f in features]
                variances = np.var(X[:, feature_indices], axis=0)
                best_idx = np.argmax(variances)
                representative_features.append(features[best_idx])
            else:
                representative_features.extend(features)
        
        return {
            'clusters': cluster_groups,
            'representative_features': representative_features,
            'n_clusters': len(cluster_groups),
            'n_representatives': len(representative_features)
        }
    
    def feature_importance_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "random_forest",
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Compute feature importance matrix using various methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Importance method
            feature_names: List of feature names
            **kwargs: Additional parameters
            
        Returns:
            Feature importance scores
        """
        self.logger.debug(f"Computing feature importance using {method}")
        
        try:
            if method == "random_forest":
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Determine if classification or regression
                unique_y = np.unique(y)
                if len(unique_y) <= 10:  # Classification
                    model = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
                else:  # Regression
                    model = RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
                
                model.fit(X, y)
                return model.feature_importances_
            
            elif method == "lasso":
                from sklearn.linear_model import LassoCV
                
                model = LassoCV(cv=5, random_state=42, **kwargs)
                model.fit(X, y)
                return np.abs(model.coef_)
            
            elif method == "elastic_net":
                from sklearn.linear_model import ElasticNetCV
                
                model = ElasticNetCV(cv=5, random_state=42, **kwargs)
                model.fit(X, y)
                return np.abs(model.coef_)
            
            else:
                raise ValueError(f"Unknown importance method: {method}")
                
        except ImportError as e:
            self.logger.warning(f"⚠️ sklearn not available for {method}: {e}")
            # Fallback to correlation
            importance_scores = []
            for i in range(X.shape[1]):
                corr, _ = pearsonr(X[:, i], y)
                importance_scores.append(abs(corr))
            return np.array(importance_scores)
    
    def variance_threshold_matrix(
        self,
        X: np.ndarray,
        threshold: float = 0.0,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Apply variance threshold to remove low-variance features.
        
        Args:
            X: Feature matrix
            threshold: Variance threshold
            feature_names: List of feature names
            
        Returns:
            Dictionary with variance analysis results
        """
        self.logger.debug(f"Applying variance threshold {threshold}")
        
        # Calculate variances
        variances = np.var(X, axis=0)
        
        # Find features above threshold
        above_threshold = variances > threshold
        
        # Get feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        selected_features = [name for i, name in enumerate(feature_names) if above_threshold[i]]
        removed_features = [name for i, name in enumerate(feature_names) if not above_threshold[i]]
        
        return {
            'selected_features': selected_features,
            'removed_features': removed_features,
            'variances': {name: variances[i] for i, name in enumerate(feature_names)},
            'n_selected': len(selected_features),
            'n_removed': len(removed_features)
        }
    
    def correlation_filter_matrix(
        self,
        X: np.ndarray,
        correlation_threshold: float = 0.95,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Filter highly correlated features.
        
        Args:
            X: Feature matrix
            correlation_threshold: Correlation threshold
            feature_names: List of feature names
            
        Returns:
            Dictionary with filtering results
        """
        self.logger.debug(f"Filtering features with correlation > {correlation_threshold}")
        
        # Compute correlation matrix
        corr_matrix = self.correlation_matrix(X, feature_names=feature_names)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > correlation_threshold:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for i, j, corr in high_corr_pairs:
            # Keep the feature with higher variance
            var_i = np.var(X[:, i])
            var_j = np.var(X[:, j])
            
            if var_i < var_j:
                features_to_remove.add(i)
            else:
                features_to_remove.add(j)
        
        # Get feature names
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        selected_features = [name for i, name in enumerate(feature_names) if i not in features_to_remove]
        removed_features = [name for i, name in enumerate(feature_names) if i in features_to_remove]
        
        return {
            'selected_features': selected_features,
            'removed_features': removed_features,
            'high_correlation_pairs': [(feature_names[i], feature_names[j], corr) for i, j, corr in high_corr_pairs],
            'n_selected': len(selected_features),
            'n_removed': len(removed_features)
        }
    
    def batch_feature_operations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        operations: List[str],
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform multiple feature operations in batch.
        
        Args:
            X: Feature matrix
            y: Target vector
            operations: List of operations to perform
            feature_names: List of feature names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results from all operations
        """
        self.logger.info(f"Performing batch operations: {operations}")
        
        results = {}
        
        for operation in operations:
            try:
                if operation == "correlation_matrix":
                    results[operation] = self.correlation_matrix(X, feature_names=feature_names)
                
                elif operation == "mutual_information":
                    results[operation] = self.mutual_information_matrix(X, y, feature_names)
                
                elif operation == "hierarchical_clustering":
                    threshold = kwargs.get('correlation_threshold', 0.95)
                    results[operation] = self.hierarchical_clustering_correlation(
                        X, threshold, feature_names=feature_names
                    )
                
                elif operation == "feature_importance":
                    method = kwargs.get('importance_method', 'random_forest')
                    results[operation] = self.feature_importance_matrix(
                        X, y, method, feature_names, **kwargs
                    )
                
                elif operation == "variance_threshold":
                    threshold = kwargs.get('variance_threshold', 0.0)
                    results[operation] = self.variance_threshold_matrix(
                        X, threshold, feature_names
                    )
                
                elif operation == "correlation_filter":
                    threshold = kwargs.get('correlation_threshold', 0.95)
                    results[operation] = self.correlation_filter_matrix(
                        X, threshold, feature_names
                    )
                
                else:
                    self.logger.warning(f"⚠️ Unknown operation: {operation}")
                    results[operation] = None
                    
            except Exception as e:
                self.logger.error(f"❌ Operation {operation} failed: {e}")
                results[operation] = None
        
        return results
    
    def optimize_feature_selection_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_features: int,
        feature_names: Optional[List[str]] = None,
        pipeline_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize feature selection pipeline using matrix operations.
        
        Args:
            X: Feature matrix
            y: Target vector
            target_features: Target number of features
            feature_names: List of feature names
            pipeline_config: Pipeline configuration
            
        Returns:
            Dictionary with optimized feature selection results
        """
        self.logger.info(f"Optimizing feature selection pipeline for {target_features} features")
        
        if pipeline_config is None:
            pipeline_config = {
                'variance_threshold': 0.0,
                'correlation_threshold': 0.95,
                'importance_method': 'random_forest',
                'clustering_method': 'ward'
            }
        
        # Step 1: Remove low variance features
        variance_result = self.variance_threshold_matrix(
            X, pipeline_config['variance_threshold'], feature_names
        )
        X_filtered = X[:, [feature_names.index(f) if feature_names else int(f.split('_')[1]) 
                           for f in variance_result['selected_features']]]
        filtered_names = variance_result['selected_features']
        
        # Step 2: Remove highly correlated features
        corr_result = self.correlation_filter_matrix(
            X_filtered, pipeline_config['correlation_threshold'], filtered_names
        )
        X_corr_filtered = X_filtered[:, [filtered_names.index(f) 
                                        for f in corr_result['selected_features']]]
        corr_filtered_names = corr_result['selected_features']
        
        # Step 3: Compute feature importance
        importance_scores = self.feature_importance_matrix(
            X_corr_filtered, y, pipeline_config['importance_method'], corr_filtered_names
        )
        
        # Step 4: Select top features
        top_indices = np.argsort(importance_scores)[::-1][:target_features]
        final_features = [corr_filtered_names[i] for i in top_indices]
        final_scores = {corr_filtered_names[i]: importance_scores[i] for i in top_indices}
        
        return {
            'selected_features': final_features,
            'feature_scores': final_scores,
            'pipeline_steps': {
                'variance_filtering': variance_result,
                'correlation_filtering': corr_result,
                'importance_ranking': {
                    'scores': importance_scores.tolist(),
                    'feature_names': corr_filtered_names
                }
            },
            'n_selected': len(final_features),
            'selection_ratio': len(final_features) / X.shape[1]
        }


# Convenience functions
def create_matrix_feature_operations(use_gpu: bool = True, use_parallel: bool = True) -> MatrixFeatureOperations:
    """Create matrix feature operations instance."""
    return MatrixFeatureOperations(use_gpu, use_parallel)


def optimize_feature_selection_with_matrix_ops(
    X: np.ndarray,
    y: np.ndarray,
    target_features: int,
    feature_names: Optional[List[str]] = None,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for optimized feature selection using matrix operations.
    
    Args:
        X: Feature matrix
        y: Target vector
        target_features: Target number of features
        feature_names: List of feature names
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with feature selection results
    """
    matrix_ops = MatrixFeatureOperations(use_gpu=use_gpu)
    return matrix_ops.optimize_feature_selection_pipeline(X, y, target_features, feature_names)


# Export key classes and functions
__all__ = [
    'MatrixFeatureOperations',
    'create_matrix_feature_operations',
    'optimize_feature_selection_with_matrix_ops'
]