"""
CV-Aware Feature Weighting System

This module implements coefficient of variation (CV) aware feature weighting
to improve clustering quality by reweighting features based on their stability.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import clustering components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class CVWeightingResult:
    """Result of CV-aware feature weighting."""
    original_features_shape: Tuple[int, int]
    weighted_features_shape: Tuple[int, int]
    feature_weights: np.ndarray
    pca_cv_scores: np.ndarray
    avg_pca_cv: float
    n_components: int
    explained_variance_ratio: np.ndarray
    clustering_quality_improvement: float
    timestamp: datetime


@dataclass
class CVWeightingConfig:
    """Configuration for CV-aware feature weighting."""
    min_cv_threshold: float = 0.01  # Minimum CV to avoid division by zero
    max_cv_threshold: float = 2.0   # Maximum CV to avoid extreme weights
    cv_epsilon: float = 1e-8        # Small epsilon for numerical stability
    n_components: int = 10          # Number of PCA components
    enable_adaptive_components: bool = True  # Whether to adapt component count
    min_explained_variance: float = 0.8     # Minimum explained variance
    weight_normalization: str = 'l2'        # Weight normalization method
    enable_cv_penalty: bool = True          # Whether to add CV penalty to composite score


class CVAwareFeatureWeighting:
    """
    CV-aware feature weighting system.
    
    Implements:
    - Coefficient of variation calculation per PCA component
    - Feature reweighting by 1/(CV + ε)
    - Adaptive PCA component selection
    - CV penalty in composite scoring
    """
    
    def __init__(self, config: CVWeightingConfig = None):
        """
        Initialize CV-aware feature weighting system.
        
        Args:
            config: Configuration object
        """
        self.config = config or CVWeightingConfig()
        
        # Results storage
        self.weighting_results: List[CVWeightingResult] = []
        self.feature_weights_history: List[np.ndarray] = []
        
        # Performance tracking
        self.performance_metrics = {
            'weighting_time': 0.0,
            'clustering_time': 0.0,
            'quality_improvement': 0.0
        }
        
    def calculate_feature_cv(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate coefficient of variation for each feature.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            CV scores for each feature
        """
        # Calculate mean and standard deviation
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Calculate CV = std / mean
        # Add epsilon to avoid division by zero
        cv_scores = feature_stds / (np.abs(feature_means) + self.config.cv_epsilon)
        
        # Clip CV scores to reasonable range
        cv_scores = np.clip(cv_scores, self.config.min_cv_threshold, self.config.max_cv_threshold)
        
        return cv_scores
    
    def calculate_pca_cv(self, features: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate CV for PCA components.
        
        Args:
            features: Feature matrix
            n_components: Number of PCA components
            
        Returns:
            Tuple of (pca_components, cv_scores, explained_variance_ratio)
        """
        if n_components is None:
            n_components = min(self.config.n_components, features.shape[1])
        
        # Perform PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca_components = pca.fit_transform(features)
        
        # Calculate CV for each PCA component
        pca_cv_scores = self.calculate_feature_cv(pca_components)
        
        return pca_components, pca_cv_scores, pca.explained_variance_ratio_
    
    def calculate_feature_weights(self, features: np.ndarray) -> np.ndarray:
        """
        Calculate feature weights based on CV.
        
        Args:
            features: Feature matrix
            
        Returns:
            Feature weights
        """
        # Calculate CV for each feature
        feature_cv = self.calculate_feature_cv(features)
        
        # Calculate weights as 1/(CV + ε)
        feature_weights = 1.0 / (feature_cv + self.config.cv_epsilon)
        
        # Normalize weights
        if self.config.weight_normalization == 'l2':
            feature_weights = feature_weights / np.linalg.norm(feature_weights)
        elif self.config.weight_normalization == 'l1':
            feature_weights = feature_weights / np.sum(feature_weights)
        elif self.config.weight_normalization == 'max':
            feature_weights = feature_weights / np.max(feature_weights)
        
        return feature_weights
    
    def apply_feature_weights(self, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Apply feature weights to the feature matrix.
        
        Args:
            features: Feature matrix
            weights: Feature weights
            
        Returns:
            Weighted feature matrix
        """
        # Ensure weights have the same shape as features
        if len(weights) != features.shape[1]:
            raise ValueError(f"Weights length {len(weights)} doesn't match features shape {features.shape[1]}")
        
        # Apply weights
        weighted_features = features * weights[np.newaxis, :]
        
        return weighted_features
    
    def adaptive_component_selection(self, 
                                   features: np.ndarray,
                                   min_explained_variance: float = None) -> int:
        """
        Select optimal number of PCA components based on explained variance.
        
        Args:
            features: Feature matrix
            min_explained_variance: Minimum explained variance threshold
            
        Returns:
            Optimal number of components
        """
        if min_explained_variance is None:
            min_explained_variance = self.config.min_explained_variance
        
        # Try different numbers of components
        max_components = min(features.shape[1], 50)  # Limit to 50 components
        best_n_components = 1
        
        for n_components in range(1, max_components + 1):
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(features)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            
            if explained_variance >= min_explained_variance:
                best_n_components = n_components
                break
        
        return best_n_components
    
    def calculate_cv_penalty(self, pca_cv_scores: np.ndarray) -> float:
        """
        Calculate CV penalty term for composite scoring.
        
        Args:
            pca_cv_scores: CV scores for PCA components
            
        Returns:
            CV penalty score
        """
        # Calculate average CV across PCA components
        avg_pca_cv = np.mean(pca_cv_scores)
        
        # Convert to penalty (higher CV = higher penalty)
        cv_penalty = avg_pca_cv / (1.0 + avg_pca_cv)  # Normalize to 0-1
        
        return cv_penalty
    
    def optimize_feature_weighting(self, 
                                 features: np.ndarray,
                                 feature_names: List[str] = None,
                                 market_data: pd.DataFrame = None) -> CVWeightingResult:
        """
        Optimize feature weighting using CV-aware approach.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            market_data: Market data for validation
            
        Returns:
            CVWeightingResult
        """
        start_time = datetime.now()
        
        logger.info("Starting CV-aware feature weighting optimization...")
        
        # Store original features shape
        original_shape = features.shape
        
        # Calculate feature weights
        feature_weights = self.calculate_feature_weights(features)
        
        # Apply weights to features
        weighted_features = self.apply_feature_weights(features, feature_weights)
        
        # Adaptive component selection
        if self.config.enable_adaptive_components:
            n_components = self.adaptive_component_selection(weighted_features)
        else:
            n_components = self.config.n_components
        
        # Calculate PCA CV scores
        pca_components, pca_cv_scores, explained_variance_ratio = self.calculate_pca_cv(
            weighted_features, n_components
        )
        
        # Calculate average PCA CV
        avg_pca_cv = np.mean(pca_cv_scores)
        
        # Calculate clustering quality improvement
        clustering_quality_improvement = 0.0
        if market_data is not None:
            clustering_quality_improvement = self._evaluate_clustering_quality(
                weighted_features, market_data
            )
        
        # Create result
        result = CVWeightingResult(
            original_features_shape=original_shape,
            weighted_features_shape=weighted_features.shape,
            feature_weights=feature_weights,
            pca_cv_scores=pca_cv_scores,
            avg_pca_cv=avg_pca_cv,
            n_components=n_components,
            explained_variance_ratio=explained_variance_ratio,
            clustering_quality_improvement=clustering_quality_improvement,
            timestamp=datetime.now()
        )
        
        # Store results
        self.weighting_results.append(result)
        self.feature_weights_history.append(feature_weights.copy())
        
        # Update performance metrics
        end_time = datetime.now()
        self.performance_metrics['weighting_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"CV-aware weighting completed in {self.performance_metrics['weighting_time']:.3f}s")
        logger.info(f"Average PCA CV: {avg_pca_cv:.4f}")
        logger.info(f"Selected components: {n_components}")
        logger.info(f"Explained variance: {np.sum(explained_variance_ratio):.4f}")
        
        return result
    
    def _evaluate_clustering_quality(self, 
                                   features: np.ndarray,
                                   market_data: pd.DataFrame) -> float:
        """
        Evaluate clustering quality improvement.
        
        Args:
            features: Feature matrix
            market_data: Market data
            
        Returns:
            Quality improvement score
        """
        try:
            # Perform clustering
            clusterer = HDBSCAN(min_cluster_size=50, min_samples=10)
            cluster_labels = clusterer.fit_predict(features)
            
            # Calculate silhouette score
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
                silhouette = silhouette_score(features[valid_mask], cluster_labels[valid_mask])
            else:
                silhouette = 0.0
            
            # Calculate economic validity if market data available
            economic_score = 0.0
            if 'returns' in market_data.columns and len(market_data) > 0:
                returns = market_data['returns'].dropna()
                if len(returns) > 0 and len(cluster_labels) > 0:
                    valid_returns = returns.iloc[valid_mask] if len(returns) >= len(cluster_labels) else returns
                    valid_labels = cluster_labels[valid_mask] if len(cluster_labels) >= len(valid_returns) else cluster_labels[:len(valid_returns)]
                    
                    if len(np.unique(valid_labels)) > 1 and len(valid_returns) > 0:
                        groups = [valid_returns[valid_labels == label].values 
                                 for label in np.unique(valid_labels) if label != -1]
                        if len(groups) > 1 and all(len(g) > 0 for g in groups):
                            from scipy import stats
                            f_stat, p_value = stats.f_oneway(*groups)
                            economic_score = f_stat if not np.isnan(f_stat) else 0.0
            
            # Combined quality score
            quality_score = 0.6 * silhouette + 0.4 * min(economic_score / 10.0, 1.0)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Clustering quality evaluation failed: {e}")
            return 0.0
    
    def get_weighted_features(self, features: np.ndarray) -> np.ndarray:
        """
        Get weighted features using the latest weights.
        
        Args:
            features: Feature matrix
            
        Returns:
            Weighted feature matrix
        """
        if not self.feature_weights_history:
            logger.warning("No feature weights available. Run optimization first.")
            return features
        
        latest_weights = self.feature_weights_history[-1]
        return self.apply_feature_weights(features, latest_weights)
    
    def get_cv_penalty_score(self) -> float:
        """
        Get CV penalty score for composite scoring.
        
        Returns:
            CV penalty score
        """
        if not self.weighting_results:
            return 0.0
        
        latest_result = self.weighting_results[-1]
        return self.calculate_cv_penalty(latest_result.pca_cv_scores)
    
    def get_weighting_summary(self) -> Dict[str, Any]:
        """Get feature weighting summary."""
        if not self.weighting_results:
            return {'message': 'No weighting results available'}
        
        latest_result = self.weighting_results[-1]
        
        return {
            'timestamp': latest_result.timestamp,
            'original_shape': latest_result.original_features_shape,
            'weighted_shape': latest_result.weighted_features_shape,
            'n_components': latest_result.n_components,
            'avg_pca_cv': latest_result.avg_pca_cv,
            'explained_variance': float(np.sum(latest_result.explained_variance_ratio)),
            'clustering_quality': latest_result.clustering_quality_improvement,
            'cv_penalty': self.get_cv_penalty_score(),
            'weight_stats': {
                'mean': float(np.mean(latest_result.feature_weights)),
                'std': float(np.std(latest_result.feature_weights)),
                'min': float(np.min(latest_result.feature_weights)),
                'max': float(np.max(latest_result.feature_weights))
            },
            'performance_metrics': self.performance_metrics
        }
    
    def save_weighting_results(self, output_file: str = None):
        """Save weighting results to file."""
        if not self.weighting_results:
            logger.warning("No results to save")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"cv_weighting_results_{timestamp}.json"
        
        output_path = Path(output_file)
        
        # Prepare data for saving
        save_data = {
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.weighting_results],
            'performance_metrics': self.performance_metrics,
            'summary': self.get_weighting_summary()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        save_data = recursive_convert(save_data)
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"CV weighting results saved to {output_path}")


def run_cv_aware_weighting(features: np.ndarray,
                          feature_names: List[str] = None,
                          market_data: pd.DataFrame = None,
                          config: CVWeightingConfig = None) -> CVWeightingResult:
    """
    Run CV-aware feature weighting.
    
    Args:
        features: Feature matrix
        feature_names: List of feature names
        market_data: Market data for validation
        config: Configuration object
        
    Returns:
        CVWeightingResult
    """
    weighting_system = CVAwareFeatureWeighting(config)
    return weighting_system.optimize_feature_weighting(features, feature_names, market_data)


if __name__ == "__main__":
    # Example usage
    print("CV-aware feature weighting example")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create features with different CV characteristics
    features = np.random.randn(n_samples, n_features)
    
    # Add some features with high CV (noisy)
    features[:, 10:20] += np.random.randn(n_samples, 10) * 2.0
    
    # Add some features with low CV (stable)
    features[:, 30:40] += np.random.randn(n_samples, 10) * 0.1
    
    # Create sample market data
    market_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, n_samples),
        'volatility': np.random.uniform(0.01, 0.05, n_samples)
    })
    
    # Run CV-aware weighting
    config = CVWeightingConfig(
        min_cv_threshold=0.01,
        max_cv_threshold=2.0,
        n_components=10,
        enable_adaptive_components=True
    )
    
    result = run_cv_aware_weighting(features, market_data=market_data, config=config)
    
    print(f"Original shape: {result.original_features_shape}")
    print(f"Weighted shape: {result.weighted_features_shape}")
    print(f"Average PCA CV: {result.avg_pca_cv:.4f}")
    print(f"Selected components: {result.n_components}")
    print(f"Explained variance: {np.sum(result.explained_variance_ratio):.4f}")
    print(f"Clustering quality: {result.clustering_quality_improvement:.4f}")