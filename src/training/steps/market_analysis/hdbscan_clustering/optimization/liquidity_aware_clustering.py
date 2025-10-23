"""
Liquidity-Aware Clustering System

This module implements liquidity-aware clustering with RVOL separation as a first-class metric
and blocks cluster merges when KS(volume) shows distinct distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import clustering and statistical components
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import umap
from scipy import stats
from scipy.stats import ks_2samp, kruskal
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


@dataclass
class LiquidityMetrics:
    """Liquidity-related metrics for clustering."""
    rvol_separation: float
    volume_ks_stat: float
    volume_ks_p: float
    volume_discrimination: float
    vol_price_correlation: float
    liquidity_diversity: float
    volume_momentum_separation: float
    timestamp: datetime


@dataclass
class LiquidityAwareConfig:
    """Configuration for liquidity-aware clustering."""
    min_rvol_separation: float = 0.1  # Minimum RVOL separation threshold
    max_volume_ks_p: float = 0.05     # Maximum p-value for volume KS test
    min_volume_discrimination: float = 0.2  # Minimum volume discrimination
    enable_volume_blocking: bool = True     # Whether to block merges based on volume
    enable_rvol_penalty: bool = True        # Whether to add RVOL penalty to score
    volume_weight: float = 0.3              # Weight for volume features
    liquidity_weight: float = 0.2           # Weight for liquidity metrics
    min_cluster_size: int = 50              # Minimum cluster size
    min_samples: int = 10                   # Minimum samples for HDBSCAN


@dataclass
class LiquidityAwareResult:
    """Result of liquidity-aware clustering."""
    cluster_labels: np.ndarray
    liquidity_metrics: LiquidityMetrics
    merge_decisions: List[Dict[str, Any]]
    blocked_merges: List[Dict[str, Any]]
    composite_score: float
    rvol_penalty: float
    n_clusters: int
    n_noise: int
    timestamp: datetime


class LiquidityAwareClustering:
    """
    Liquidity-aware clustering system.
    
    Features:
    - RVOL separation as first-class metric
    - Volume distribution blocking for merges
    - Liquidity-aware distance metrics
    - Volume-weighted feature engineering
    - Comprehensive liquidity validation
    """
    
    def __init__(self, config: LiquidityAwareConfig = None):
        """
        Initialize liquidity-aware clustering system.
        
        Args:
            config: Configuration object
        """
        self.config = config or LiquidityAwareConfig()
        
        # Results storage
        self.clustering_results: List[LiquidityAwareResult] = []
        self.liquidity_metrics_history: List[LiquidityMetrics] = []
        
        # Performance tracking
        self.performance_metrics = {
            'clustering_time': 0.0,
            'liquidity_evaluation_time': 0.0,
            'merge_blocking_time': 0.0,
            'n_blocked_merges': 0,
            'n_allowed_merges': 0
        }
        
    def calculate_rvol_separation(self, 
                                cluster_labels: np.ndarray,
                                market_data: pd.DataFrame) -> float:
        """
        Calculate RVOL separation across clusters.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume information
            
        Returns:
            RVOL separation score
        """
        if 'volume' not in market_data.columns or 'volume_ma' not in market_data.columns:
            logger.warning("Volume data not available for RVOL calculation")
            return 0.0
        
        # Calculate RVOL
        rvol = market_data['volume'] / market_data['volume_ma']
        
        # Filter valid data
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        valid_rvol = rvol.iloc[valid_mask]
        
        if len(np.unique(valid_labels)) < 2 or len(valid_rvol) == 0:
            return 0.0
        
        # Calculate RVOL separation using ANOVA
        rvol_groups = [valid_rvol[valid_labels == label].values 
                      for label in np.unique(valid_labels) if label != -1]
        
        if len(rvol_groups) < 2 or any(len(g) == 0 for g in rvol_groups):
            return 0.0
        
        try:
            f_stat, p_value = stats.f_oneway(*rvol_groups)
            return f_stat if not np.isnan(f_stat) else 0.0
        except Exception as e:
            logger.error(f"RVOL separation calculation failed: {e}")
            return 0.0
    
    def calculate_volume_ks_test(self, 
                               cluster_labels: np.ndarray,
                               market_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate KS test for volume distributions between clusters.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume information
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        if 'volume' not in market_data.columns:
            logger.warning("Volume data not available for KS test")
            return 0.0, 1.0
        
        volume = market_data['volume']
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        valid_volume = volume.iloc[valid_mask]
        
        if len(np.unique(valid_labels)) < 2 or len(valid_volume) == 0:
            return 0.0, 1.0
        
        # Get volume groups for each cluster
        volume_groups = [valid_volume[valid_labels == label].values 
                        for label in np.unique(valid_labels) if label != -1]
        
        if len(volume_groups) < 2 or any(len(g) == 0 for g in volume_groups):
            return 0.0, 1.0
        
        try:
            # Perform KS test between all pairs of clusters
            ks_stats = []
            p_values = []
            
            for i in range(len(volume_groups)):
                for j in range(i + 1, len(volume_groups)):
                    ks_stat, p_value = ks_2samp(volume_groups[i], volume_groups[j])
                    ks_stats.append(ks_stat)
                    p_values.append(p_value)
            
            # Return average KS statistic and minimum p-value
            avg_ks_stat = np.mean(ks_stats) if ks_stats else 0.0
            min_p_value = np.min(p_values) if p_values else 1.0
            
            return avg_ks_stat, min_p_value
            
        except Exception as e:
            logger.error(f"Volume KS test calculation failed: {e}")
            return 0.0, 1.0
    
    def calculate_volume_discrimination(self, 
                                      cluster_labels: np.ndarray,
                                      market_data: pd.DataFrame) -> float:
        """
        Calculate volume discrimination across clusters.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume information
            
        Returns:
            Volume discrimination score
        """
        if 'volume' not in market_data.columns:
            return 0.0
        
        volume = market_data['volume']
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        valid_volume = volume.iloc[valid_mask]
        
        if len(np.unique(valid_labels)) < 2 or len(valid_volume) == 0:
            return 0.0
        
        # Calculate volume discrimination using Kruskal-Wallis test
        volume_groups = [valid_volume[valid_labels == label].values 
                        for label in np.unique(valid_labels) if label != -1]
        
        if len(volume_groups) < 2 or any(len(g) == 0 for g in volume_groups):
            return 0.0
        
        try:
            h_stat, p_value = kruskal(*volume_groups)
            return h_stat if not np.isnan(h_stat) else 0.0
        except Exception as e:
            logger.error(f"Volume discrimination calculation failed: {e}")
            return 0.0
    
    def calculate_vol_price_correlation(self, 
                                      cluster_labels: np.ndarray,
                                      market_data: pd.DataFrame) -> float:
        """
        Calculate volume-price correlation by cluster.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume and price information
            
        Returns:
            Average volume-price correlation
        """
        if 'volume' not in market_data.columns or 'returns' not in market_data.columns:
            return 0.0
        
        volume = market_data['volume']
        returns = market_data['returns'].dropna()
        
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        
        if len(valid_labels) == 0 or len(returns) == 0:
            return 0.0
        
        # Align data lengths
        min_len = min(len(valid_labels), len(returns))
        valid_labels = valid_labels[:min_len]
        valid_volume = volume.iloc[:min_len]
        valid_returns = returns.iloc[:min_len]
        
        correlations = []
        
        for label in np.unique(valid_labels):
            if label != -1:
                cluster_mask = valid_labels == label
                cluster_volume = valid_volume[cluster_mask]
                cluster_returns = valid_returns[cluster_mask]
                
                if len(cluster_volume) > 1 and len(cluster_returns) > 1:
                    correlation = np.corrcoef(cluster_volume, cluster_returns)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_liquidity_diversity(self, 
                                    cluster_labels: np.ndarray,
                                    market_data: pd.DataFrame) -> float:
        """
        Calculate liquidity diversity across clusters.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume information
            
        Returns:
            Liquidity diversity score
        """
        if 'volume' not in market_data.columns:
            return 0.0
        
        volume = market_data['volume']
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        valid_volume = volume.iloc[valid_mask]
        
        if len(np.unique(valid_labels)) < 2 or len(valid_volume) == 0:
            return 0.0
        
        # Calculate volume statistics for each cluster
        cluster_volumes = []
        for label in np.unique(valid_labels):
            if label != -1:
                cluster_volume = valid_volume[valid_labels == label]
                if len(cluster_volume) > 0:
                    cluster_volumes.append({
                        'mean': np.mean(cluster_volume),
                        'std': np.std(cluster_volume),
                        'cv': np.std(cluster_volume) / np.mean(cluster_volume) if np.mean(cluster_volume) > 0 else 0
                    })
        
        if len(cluster_volumes) < 2:
            return 0.0
        
        # Calculate diversity as coefficient of variation of cluster means
        cluster_means = [cv['mean'] for cv in cluster_volumes]
        diversity = np.std(cluster_means) / np.mean(cluster_means) if np.mean(cluster_means) > 0 else 0.0
        
        return diversity
    
    def calculate_volume_momentum_separation(self, 
                                           cluster_labels: np.ndarray,
                                           market_data: pd.DataFrame) -> float:
        """
        Calculate volume momentum separation across clusters.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data with volume information
            
        Returns:
            Volume momentum separation score
        """
        if 'volume' not in market_data.columns:
            return 0.0
        
        # Calculate volume momentum (short-term / long-term volume ratio)
        if 'volume_ma' in market_data.columns:
            volume_momentum = market_data['volume'] / market_data['volume_ma']
        else:
            # Calculate rolling means if not available
            volume_short = market_data['volume'].rolling(5).mean()
            volume_long = market_data['volume'].rolling(20).mean()
            volume_momentum = volume_short / volume_long
        
        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        valid_momentum = volume_momentum.iloc[valid_mask]
        
        if len(np.unique(valid_labels)) < 2 or len(valid_momentum) == 0:
            return 0.0
        
        # Calculate momentum separation using ANOVA
        momentum_groups = [valid_momentum[valid_labels == label].values 
                          for label in np.unique(valid_labels) if label != -1]
        
        if len(momentum_groups) < 2 or any(len(g) == 0 for g in momentum_groups):
            return 0.0
        
        try:
            f_stat, p_value = stats.f_oneway(*momentum_groups)
            return f_stat if not np.isnan(f_stat) else 0.0
        except Exception as e:
            logger.error(f"Volume momentum separation calculation failed: {e}")
            return 0.0
    
    def calculate_liquidity_metrics(self, 
                                  cluster_labels: np.ndarray,
                                  market_data: pd.DataFrame) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics.
        
        Args:
            cluster_labels: Cluster labels
            market_data: Market data
            
        Returns:
            LiquidityMetrics object
        """
        start_time = datetime.now()
        
        # Calculate all liquidity metrics
        rvol_separation = self.calculate_rvol_separation(cluster_labels, market_data)
        volume_ks_stat, volume_ks_p = self.calculate_volume_ks_test(cluster_labels, market_data)
        volume_discrimination = self.calculate_volume_discrimination(cluster_labels, market_data)
        vol_price_correlation = self.calculate_vol_price_correlation(cluster_labels, market_data)
        liquidity_diversity = self.calculate_liquidity_diversity(cluster_labels, market_data)
        volume_momentum_separation = self.calculate_volume_momentum_separation(cluster_labels, market_data)
        
        # Create metrics object
        metrics = LiquidityMetrics(
            rvol_separation=rvol_separation,
            volume_ks_stat=volume_ks_stat,
            volume_ks_p=volume_ks_p,
            volume_discrimination=volume_discrimination,
            vol_price_correlation=vol_price_correlation,
            liquidity_diversity=liquidity_diversity,
            volume_momentum_separation=volume_momentum_separation,
            timestamp=datetime.now()
        )
        
        # Update performance metrics
        end_time = datetime.now()
        self.performance_metrics['liquidity_evaluation_time'] = (end_time - start_time).total_seconds()
        
        # Store metrics
        self.liquidity_metrics_history.append(metrics)
        
        return metrics
    
    def should_block_merge(self, 
                          cluster1_labels: np.ndarray,
                          cluster2_labels: np.ndarray,
                          market_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if cluster merge should be blocked based on volume distributions.
        
        Args:
            cluster1_labels: Labels for first cluster
            cluster2_labels: Labels for second cluster
            market_data: Market data
            
        Returns:
            Tuple of (should_block, reason_dict)
        """
        if not self.config.enable_volume_blocking:
            return False, {}
        
        start_time = datetime.now()
        
        # Calculate volume KS test between clusters
        volume_ks_stat, volume_ks_p = self.calculate_volume_ks_test(
            np.concatenate([cluster1_labels, cluster2_labels]), market_data
        )
        
        # Check if distributions are significantly different
        should_block = volume_ks_p < self.config.max_volume_ks_p
        
        reason = {
            'volume_ks_stat': volume_ks_stat,
            'volume_ks_p': volume_ks_p,
            'threshold': self.config.max_volume_ks_p,
            'blocked': should_block
        }
        
        # Update performance metrics
        end_time = datetime.now()
        self.performance_metrics['merge_blocking_time'] += (end_time - start_time).total_seconds()
        
        if should_block:
            self.performance_metrics['n_blocked_merges'] += 1
        else:
            self.performance_metrics['n_allowed_merges'] += 1
        
        return should_block, reason
    
    def calculate_rvol_penalty(self, liquidity_metrics: LiquidityMetrics) -> float:
        """
        Calculate RVOL penalty for composite scoring.
        
        Args:
            liquidity_metrics: Liquidity metrics
            
        Returns:
            RVOL penalty score
        """
        if not self.config.enable_rvol_penalty:
            return 0.0
        
        # Calculate penalty based on RVOL separation
        rvol_penalty = 0.0
        
        if liquidity_metrics.rvol_separation < self.config.min_rvol_separation:
            rvol_penalty = (self.config.min_rvol_separation - liquidity_metrics.rvol_separation) / self.config.min_rvol_separation
        
        return rvol_penalty
    
    def create_liquidity_aware_features(self, 
                                      features: np.ndarray,
                                      market_data: pd.DataFrame) -> np.ndarray:
        """
        Create liquidity-aware features by weighting volume-related features.
        
        Args:
            features: Original feature matrix
            market_data: Market data
            
        Returns:
            Liquidity-aware feature matrix
        """
        # Create volume-weighted features
        if 'volume' in market_data.columns:
            # Calculate volume weights
            volume = market_data['volume']
            volume_weights = volume / volume.mean()  # Normalize by mean volume
            
            # Apply volume weighting to features
            # This is a simplified approach - in practice, you'd identify volume-related features
            liquidity_aware_features = features.copy()
            
            # Weight features by volume (assuming first few features are volume-related)
            n_volume_features = min(10, features.shape[1])
            for i in range(n_volume_features):
                liquidity_aware_features[:, i] *= volume_weights.iloc[:len(features)].values
            
            return liquidity_aware_features
        else:
            return features
    
    def perform_liquidity_aware_clustering(self, 
                                         features: np.ndarray,
                                         market_data: pd.DataFrame,
                                         feature_names: List[str] = None) -> LiquidityAwareResult:
        """
        Perform liquidity-aware clustering.
        
        Args:
            features: Feature matrix
            market_data: Market data
            feature_names: List of feature names
            
        Returns:
            LiquidityAwareResult
        """
        start_time = datetime.now()
        
        logger.info("Starting liquidity-aware clustering...")
        
        # Create liquidity-aware features
        liquidity_aware_features = self.create_liquidity_aware_features(features, market_data)
        
        # Perform dimensionality reduction
        if liquidity_aware_features.shape[1] > 10:
            pca = PCA(n_components=10, random_state=42)
            features_reduced = pca.fit_transform(liquidity_aware_features)
        else:
            features_reduced = liquidity_aware_features
        
        # UMAP for non-linear dimensionality reduction
        if features_reduced.shape[1] > 2:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            features_umap = umap_reducer.fit_transform(features_reduced)
        else:
            features_umap = features_reduced
        
        # HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples
        )
        cluster_labels = clusterer.fit_predict(features_umap)
        
        # Calculate liquidity metrics
        liquidity_metrics = self.calculate_liquidity_metrics(cluster_labels, market_data)
        
        # Calculate RVOL penalty
        rvol_penalty = self.calculate_rvol_penalty(liquidity_metrics)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(cluster_labels, features_umap, liquidity_metrics, rvol_penalty)
        
        # Simulate merge decisions (in practice, this would be part of the clustering algorithm)
        merge_decisions = self._simulate_merge_decisions(cluster_labels, market_data)
        
        # Create result
        result = LiquidityAwareResult(
            cluster_labels=cluster_labels,
            liquidity_metrics=liquidity_metrics,
            merge_decisions=merge_decisions['allowed'],
            blocked_merges=merge_decisions['blocked'],
            composite_score=composite_score,
            rvol_penalty=rvol_penalty,
            n_clusters=len(np.unique(cluster_labels[cluster_labels != -1])),
            n_noise=np.sum(cluster_labels == -1),
            timestamp=datetime.now()
        )
        
        # Store result
        self.clustering_results.append(result)
        
        # Update performance metrics
        end_time = datetime.now()
        self.performance_metrics['clustering_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Liquidity-aware clustering completed in {self.performance_metrics['clustering_time']:.3f}s")
        logger.info(f"Clusters: {result.n_clusters}, Noise: {result.n_noise}")
        logger.info(f"RVOL separation: {liquidity_metrics.rvol_separation:.4f}")
        logger.info(f"Volume KS p-value: {liquidity_metrics.volume_ks_p:.4f}")
        logger.info(f"Blocked merges: {len(result.blocked_merges)}")
        
        return result
    
    def _calculate_composite_score(self, 
                                 cluster_labels: np.ndarray,
                                 features: np.ndarray,
                                 liquidity_metrics: LiquidityMetrics,
                                 rvol_penalty: float) -> float:
        """Calculate composite score including liquidity metrics."""
        # Clustering quality score
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
            silhouette = silhouette_score(features[valid_mask], cluster_labels[valid_mask])
        else:
            silhouette = 0.0
        
        # Liquidity score
        liquidity_score = (
            self.config.volume_weight * liquidity_metrics.volume_discrimination +
            self.config.liquidity_weight * liquidity_metrics.rvol_separation
        )
        
        # Normalize scores
        silhouette_norm = max(0, min(1, silhouette))
        liquidity_norm = max(0, min(1, liquidity_score / 10.0))  # Normalize F-statistic
        
        # Composite score with RVOL penalty
        composite_score = 0.6 * silhouette_norm + 0.4 * liquidity_norm - rvol_penalty
        
        return max(0, composite_score)  # Ensure non-negative
    
    def _simulate_merge_decisions(self, 
                                cluster_labels: np.ndarray,
                                market_data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Simulate merge decisions for demonstration."""
        unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
        
        allowed_merges = []
        blocked_merges = []
        
        # Simulate pairwise merge decisions
        for i in range(len(unique_clusters)):
            for j in range(i + 1, len(unique_clusters)):
                cluster1 = unique_clusters[i]
                cluster2 = unique_clusters[j]
                
                cluster1_mask = cluster_labels == cluster1
                cluster2_mask = cluster_labels == cluster2
                
                should_block, reason = self.should_block_merge(
                    cluster_labels[cluster1_mask],
                    cluster_labels[cluster2_mask],
                    market_data
                )
                
                merge_decision = {
                    'cluster1': int(cluster1),
                    'cluster2': int(cluster2),
                    'reason': reason
                }
                
                if should_block:
                    blocked_merges.append(merge_decision)
                else:
                    allowed_merges.append(merge_decision)
        
        return {
            'allowed': allowed_merges,
            'blocked': blocked_merges
        }
    
    def get_liquidity_summary(self) -> Dict[str, Any]:
        """Get liquidity clustering summary."""
        if not self.clustering_results:
            return {'message': 'No clustering results available'}
        
        latest_result = self.clustering_results[-1]
        latest_metrics = latest_result.liquidity_metrics
        
        return {
            'timestamp': latest_result.timestamp,
            'n_clusters': latest_result.n_clusters,
            'n_noise': latest_result.n_noise,
            'composite_score': latest_result.composite_score,
            'rvol_penalty': latest_result.rvol_penalty,
            'liquidity_metrics': {
                'rvol_separation': latest_metrics.rvol_separation,
                'volume_ks_p': latest_metrics.volume_ks_p,
                'volume_discrimination': latest_metrics.volume_discrimination,
                'vol_price_correlation': latest_metrics.vol_price_correlation,
                'liquidity_diversity': latest_metrics.liquidity_diversity,
                'volume_momentum_separation': latest_metrics.volume_momentum_separation
            },
            'merge_decisions': {
                'allowed_merges': len(latest_result.merge_decisions),
                'blocked_merges': len(latest_result.blocked_merges)
            },
            'performance_metrics': self.performance_metrics
        }
    
    def save_liquidity_results(self, output_file: str = None):
        """Save liquidity clustering results."""
        if not self.clustering_results:
            logger.warning("No results to save")
            return
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"liquidity_clustering_results_{timestamp}.json"
        
        output_path = Path(output_file)
        
        # Prepare data for saving
        save_data = {
            'config': asdict(self.config),
            'results': [asdict(result) for result in self.clustering_results],
            'performance_metrics': self.performance_metrics,
            'summary': self.get_liquidity_summary()
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
        
        logger.info(f"Liquidity clustering results saved to {output_path}")


def run_liquidity_aware_clustering(features: np.ndarray,
                                 market_data: pd.DataFrame,
                                 feature_names: List[str] = None,
                                 config: LiquidityAwareConfig = None) -> LiquidityAwareResult:
    """
    Run liquidity-aware clustering.
    
    Args:
        features: Feature matrix
        market_data: Market data
        feature_names: List of feature names
        config: Configuration object
        
    Returns:
        LiquidityAwareResult
    """
    clustering_system = LiquidityAwareClustering(config)
    return clustering_system.perform_liquidity_aware_clustering(features, market_data, feature_names)


if __name__ == "__main__":
    # Example usage
    print("Liquidity-aware clustering example")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create features
    features = np.random.randn(n_samples, n_features)
    
    # Create sample market data with volume
    market_data = pd.DataFrame({
        'returns': np.random.normal(0, 0.01, n_samples),
        'volume': np.random.lognormal(5, 0.5, n_samples),
        'volume_ma': np.random.lognormal(5, 0.3, n_samples)
    })
    
    # Run liquidity-aware clustering
    config = LiquidityAwareConfig(
        min_rvol_separation=0.1,
        max_volume_ks_p=0.05,
        enable_volume_blocking=True,
        enable_rvol_penalty=True
    )
    
    result = run_liquidity_aware_clustering(features, market_data, config=config)
    
    print(f"Clusters: {result.n_clusters}")
    print(f"Noise: {result.n_noise}")
    print(f"Composite score: {result.composite_score:.4f}")
    print(f"RVOL penalty: {result.rvol_penalty:.4f}")
    print(f"RVOL separation: {result.liquidity_metrics.rvol_separation:.4f}")
    print(f"Volume KS p-value: {result.liquidity_metrics.volume_ks_p:.4f}")
    print(f"Blocked merges: {len(result.blocked_merges)}")
    print(f"Allowed merges: {len(result.merge_decisions)}")