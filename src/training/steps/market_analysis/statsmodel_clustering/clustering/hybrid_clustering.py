"""
Hybrid Clustering Combining Static and Temporal Approaches

This module implements hybrid clustering that combines static asset clustering
with temporal regime modeling for improved robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

# Import statsmodels
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    MarkovRegression = None

# Import sklearn for clustering
try:
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import community detection
try:
    import networkx as nx
    import community as community_louvain
    COMMUNITY_DETECTION_AVAILABLE = True
except ImportError:
    COMMUNITY_DETECTION_AVAILABLE = False

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')

# Import feature engineering
try:
    from ..feature_engineering.covariance_stabilization import CovarianceStabilizer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False


@dataclass
class HybridClusteringConfig:
    """Configuration for hybrid clustering."""
    # Static clustering parameters
    static_method: str = 'hierarchical'  # 'hierarchical', 'spectral', 'louvain'
    n_asset_clusters: int = 5
    linkage_method: str = 'ward'  # For hierarchical
    affinity_method: str = 'correlation'  # For spectral
    
    # Temporal modeling parameters
    n_regimes: int = 3
    switching_variance: bool = True
    switching_trend: bool = True
    
    # Aggregation method
    aggregation_method: str = 'pca'  # 'pca', 'mean', 'weighted_mean'
    n_pca_components: int = 3
    
    # Covariance stabilization
    covariance_method: str = 'ledoit_wolf'
    
    # Consensus parameters
    consensus_method: str = 'majority_voting'  # 'majority_voting', 'weighted_voting'
    ensemble_weights: Optional[List[float]] = None


class HybridClusteringEngine:
    """
    Hybrid clustering engine combining static asset clustering with temporal modeling.
    
    This approach reduces noise and dimensionality by first clustering assets
    statically, then modeling temporal dynamics on aggregated cluster series.
    """
    
    def __init__(self, config: Optional[HybridClusteringConfig] = None):
        """
        Initialize hybrid clustering engine.
        
        Args:
            config: Configuration for hybrid clustering
        """
        self.config = config or HybridClusteringConfig()
        
        tprint_info("🔧 Initialized Hybrid Clustering Engine")
        tprint_info(f"📊 Static method: {self.config.static_method}, Temporal regimes: {self.config.n_regimes}")
        
        # Initialize components
        if FEATURE_ENGINEERING_AVAILABLE:
            self.cov_stabilizer = CovarianceStabilizer(
                method=self.config.covariance_method
            )
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate clustering configuration."""
        tprint_info("🔍 Validating clustering configuration")
        
        if self.config.n_asset_clusters < 2:
            tprint_error("❌ n_asset_clusters must be >= 2")
            raise ValueError("n_asset_clusters must be >= 2")
        
        if self.config.n_regimes < 2:
            tprint_error("❌ n_regimes must be >= 2")
            raise ValueError("n_regimes must be >= 2")
        
        if self.config.static_method not in ['hierarchical', 'spectral', 'louvain']:
            tprint_error(f"❌ Unknown static method: {self.config.static_method}")
            raise ValueError(f"Unknown static method: {self.config.static_method}")
        
        if self.config.aggregation_method not in ['pca', 'mean', 'weighted_mean']:
            tprint_error(f"❌ Unknown aggregation method: {self.config.aggregation_method}")
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
        
        tprint_success("✅ Configuration validation passed")
    
    def fit_predict(self, 
                   returns: pd.DataFrame,
                   features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit hybrid clustering model and predict regimes.
        
        Args:
            returns: Asset returns matrix (T x N)
            features: Optional additional features
            
        Returns:
            Dictionary with clustering results
        """
        tprint_info("🔍 Fitting hybrid clustering model")
        
        try:
            # Step 1: Static asset clustering
            asset_clusters = self._static_asset_clustering(returns)
            
            # Step 2: Aggregate series per cluster
            cluster_series = self._aggregate_cluster_series(returns, asset_clusters)
            
            # Step 3: Temporal modeling on aggregated series
            temporal_results = self._temporal_modeling(cluster_series)
            
            # Step 4: Map back to asset-level predictions
            asset_regimes = self._map_to_asset_regimes(
                temporal_results['regime_labels'], 
                asset_clusters
            )
            
            # Compile results
            results = {
                'asset_clusters': asset_clusters,
                'cluster_series': cluster_series,
                'temporal_model': temporal_results['model'],
                'regime_labels': temporal_results['regime_labels'],
                'regime_probabilities': temporal_results['regime_probabilities'],
                'transition_matrix': temporal_results['transition_matrix'],
                'asset_regimes': asset_regimes,
                'model_metrics': temporal_results['model_metrics'],
                'clustering_config': self.config.__dict__
            }
            
            tprint_success("✅ Hybrid clustering complete")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Hybrid clustering failed: {e}")
            raise
    
    def _static_asset_clustering(self, returns: pd.DataFrame) -> np.ndarray:
        """Perform static clustering on assets."""
        tprint_info(f"📊 Static asset clustering using {self.config.static_method}")
        
        # Calculate correlation matrix with stabilization
        if FEATURE_ENGINEERING_AVAILABLE:
            _, corr_matrix = self.cov_stabilizer.stabilize_covariance(returns)
        else:
            corr_matrix = returns.corr()
        
        # Convert to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Apply clustering method
        if self.config.static_method == 'hierarchical':
            asset_clusters = self._hierarchical_clustering(distance_matrix)
        elif self.config.static_method == 'spectral':
            asset_clusters = self._spectral_clustering(distance_matrix)
        elif self.config.static_method == 'louvain':
            asset_clusters = self._louvain_clustering(distance_matrix)
        else:
            raise ValueError(f"Unknown static method: {self.config.static_method}")
        
        return asset_clusters
    
    def _hierarchical_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering."""
        tprint_info(f"🔗 Performing hierarchical clustering with {self.config.linkage_method} linkage")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ sklearn not available, using simple clustering")
            return np.arange(distance_matrix.shape[0])
        
        clustering = AgglomerativeClustering(
            n_clusters=self.config.n_asset_clusters,
            linkage=self.config.linkage_method,
            affinity='precomputed'
        )
        
        result = clustering.fit_predict(distance_matrix)
        tprint_success(f"✅ Hierarchical clustering completed with {len(np.unique(result))} clusters")
        return result
    
    def _spectral_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Perform spectral clustering."""
        tprint_info("🌟 Performing spectral clustering")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ sklearn not available, using simple clustering")
            return np.arange(distance_matrix.shape[0])
        
        clustering = SpectralClustering(
            n_clusters=self.config.n_asset_clusters,
            affinity='precomputed'
        )
        
        result = clustering.fit_predict(distance_matrix)
        tprint_success(f"✅ Spectral clustering completed with {len(np.unique(result))} clusters")
        return result
    
    def _louvain_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Perform Louvain community detection."""
        tprint_info("🔍 Performing Louvain community detection")
        
        if not COMMUNITY_DETECTION_AVAILABLE:
            tprint_warning("⚠️ community detection not available, using simple clustering")
            return np.arange(distance_matrix.shape[0])
        
        # Convert distance to similarity
        tprint_info("🔄 Converting distance matrix to similarity matrix")
        similarity_matrix = 1 - distance_matrix
        similarity_matrix[similarity_matrix < 0] = 0
        
        # Create graph
        tprint_info("🕸️ Creating network graph from similarity matrix")
        G = nx.from_numpy_array(similarity_matrix)
        
        # Apply Louvain algorithm
        tprint_info("🎯 Applying Louvain community detection algorithm")
        partition = community_louvain.best_partition(G)
        
        # Convert to cluster labels
        asset_labels = np.zeros(distance_matrix.shape[0], dtype=int)
        for node, cluster_id in partition.items():
            asset_labels[node] = cluster_id
        
        tprint_success(f"✅ Louvain clustering completed with {len(np.unique(asset_labels))} communities")
        return asset_labels
    
    def _aggregate_cluster_series(self, 
                              returns: pd.DataFrame, 
                              asset_clusters: np.ndarray) -> pd.DataFrame:
        """Aggregate returns series per asset cluster."""
        tprint_info(f"📈 Aggregating series using {self.config.aggregation_method}")
        
        cluster_series = pd.DataFrame(index=returns.index)
        
        for cluster_id in range(self.config.n_asset_clusters):
            # Get assets in this cluster
            cluster_assets = np.where(asset_clusters == cluster_id)[0]
            
            if len(cluster_assets) == 0:
                continue
            
            cluster_returns = returns.iloc[:, cluster_assets]
            
            # Apply aggregation method
            if self.config.aggregation_method == 'mean':
                aggregated = cluster_returns.mean(axis=1)
            elif self.config.aggregation_method == 'weighted_mean':
                # Weight by inverse volatility
                vol_weights = 1 / (cluster_returns.rolling(20).std() + 1e-8)
                vol_weights = vol_weights.div(vol_weights.sum(axis=1), axis=0)
                aggregated = (cluster_returns * vol_weights).sum(axis=1)
            elif self.config.aggregation_method == 'pca':
                if SKLEARN_AVAILABLE:
                    pca = PCA(n_components=min(self.config.n_pca_components, len(cluster_assets)))
                    aggregated_returns = pca.fit_transform(cluster_returns.fillna(0))
                    # Use first principal component
                    aggregated = pd.Series(
                        aggregated_returns[:, 0], 
                        index=returns.index
                    )
                else:
                    tprint_warning("⚠️ sklearn not available, using mean aggregation")
                    aggregated = cluster_returns.mean(axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
            
            cluster_series[f'cluster_{cluster_id}'] = aggregated
        
        return cluster_series
    
    def _temporal_modeling(self, cluster_series: pd.DataFrame) -> Dict[str, Any]:
        """Perform temporal modeling on aggregated cluster series."""
        tprint_info("⏰ Temporal modeling using MarkovRegression")
        
        if not STATSMODELS_AVAILABLE:
            tprint_error("❌ statsmodels not available for temporal modeling")
            raise ImportError("statsmodels is required for temporal modeling")
        
        # Use first cluster series for modeling (or combine them)
        tprint_info(f"📊 Using cluster series: {cluster_series.columns[0]} for temporal modeling")
        endog = cluster_series.iloc[:, 0]  # Use first cluster
        
        # Create and fit MarkovRegression model
        tprint_info(f"🔧 Creating MarkovRegression model with {self.config.n_regimes} regimes")
        model = MarkovRegression(
            endog=endog,
            k_regimes=self.config.n_regimes,
            switching_variance=self.config.switching_variance,
            switching_trend=self.config.switching_trend,
            trend='c'  # Constant trend
        )
        
        tprint_info("🔄 Fitting MarkovRegression model")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit()
        
        # Get predictions
        tprint_info("📈 Extracting regime predictions and probabilities")
        regime_labels = fitted_model.smoothed_marginal_probabilities.argmax(axis=1)
        regime_probabilities = fitted_model.smoothed_marginal_probabilities
        
        # Get transition matrix
        tprint_info("🔄 Extracting transition matrix")
        transition_matrix = fitted_model.regime_transition_matrix
        
        # Calculate model metrics
        tprint_info("📊 Calculating model metrics")
        model_metrics = {
            'log_likelihood': fitted_model.llf,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'converged': fitted_model.mle_retvals.get('converged', False)
        }
        
        tprint_success(f"✅ Temporal modeling completed (converged: {model_metrics['converged']})")
        return {
            'model': fitted_model,
            'regime_labels': regime_labels,
            'regime_probabilities': regime_probabilities,
            'transition_matrix': transition_matrix,
            'model_metrics': model_metrics
        }
    
    def _map_to_asset_regimes(self,
                             temporal_regimes: np.ndarray,
                             asset_clusters: np.ndarray) -> np.ndarray:
        """Map temporal regimes back to asset-level predictions."""
        tprint_info("🗺️ Mapping temporal regimes to asset-level predictions")
        
        n_assets = len(asset_clusters)
        n_periods = len(temporal_regimes)
        
        tprint_info(f"📊 Creating asset-level regime matrix ({n_periods} periods × {n_assets} assets)")
        
        # Create asset-level regime matrix
        asset_regimes = np.zeros((n_periods, n_assets))
        
        for t in range(n_periods):
            regime = temporal_regimes[t]
            for asset in range(n_assets):
                asset_cluster = asset_clusters[asset]
                asset_regimes[t, asset] = regime
        
        tprint_success("✅ Asset-level regime mapping completed")
        return asset_regimes


def create_hybrid_clustering_engine(
    static_method: str = 'hierarchical',
    n_asset_clusters: int = 5,
    n_regimes: int = 3,
    aggregation_method: str = 'pca',
    covariance_method: str = 'ledoit_wolf'
) -> HybridClusteringEngine:
    """
    Factory function to create hybrid clustering engine.
    
    Args:
        static_method: Static clustering method
        n_asset_clusters: Number of asset clusters
        n_regimes: Number of temporal regimes
        aggregation_method: Method for aggregating cluster series
        covariance_method: Covariance stabilization method
        
    Returns:
        HybridClusteringEngine instance
    """
    tprint_info("🏭 Creating Hybrid Clustering Engine with factory function")
    
    config = HybridClusteringConfig(
        static_method=static_method,
        n_asset_clusters=n_asset_clusters,
        n_regimes=n_regimes,
        aggregation_method=aggregation_method,
        covariance_method=covariance_method
    )
    
    tprint_success("✅ Hybrid Clustering Engine created successfully")
    return HybridClusteringEngine(config)