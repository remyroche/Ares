"""
Candidate Clustering - 100% Data-Driven

Clusters raw candidate levels into S/R zones using fast 1D DBSCAN.
Reduces thousands of raw extrema into meaningful support/resistance zones.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class CandidateClustering:
    """
    Cluster raw candidate levels into S/R zones using 1D DBSCAN.
    
    Philosophy: Reduce noise by grouping nearby extrema, but preserve
    the temporal contract by using earliest timestamp in each cluster.
    """
    
    def __init__(
        self, 
        eps_ratio: float = 0.0025,  # 0.25% of median price
        min_samples: int = 3,        # Minimum 3 points to form a zone
        enable_clustering: bool = False  # Disabled by default (for now)
    ):
        """
        Initialize candidate clustering.
        
        Args:
            eps_ratio: Ratio of median price for epsilon (default: 0.0025 = 0.25%)
            min_samples: Minimum samples to form a cluster (default: 3)
            enable_clustering: Whether to enable clustering (default: False)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.eps_ratio = eps_ratio
        self.min_samples = min_samples
        self.enable_clustering = enable_clustering
    
    def cluster_candidates(
        self, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cluster candidate levels into S/R zones.
        
        Args:
            candidates: List of candidate dicts with keys: price, idx, type, timestamp
        
        Returns:
            List of clustered candidates (or original if clustering disabled)
        """
        if not self.enable_clustering:
            self.logger.info("⚡ Clustering disabled - using all raw candidates")
            return candidates
        
        if len(candidates) < self.min_samples:
            self.logger.warning(
                f"Too few candidates ({len(candidates)}) for clustering "
                f"(min: {self.min_samples})"
            )
            return candidates
        
        self.logger.info(f"📍 Clustering {len(candidates)} candidates...")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(candidates)
        
        # Prepare data for 1D clustering
        prices = df['price'].values.reshape(-1, 1)
        
        # Determine epsilon dynamically based on median price
        median_price = np.median(prices)
        eps_value = median_price * self.eps_ratio
        
        self.logger.info(
            f"   Using DBSCAN with eps={eps_value:.2f} ({self.eps_ratio*100:.2f}% of price), "
            f"min_samples={self.min_samples}"
        )
        
        # Apply DBSCAN clustering
        db = DBSCAN(eps=eps_value, min_samples=self.min_samples, metric='euclidean')
        cluster_labels = db.fit_predict(prices)
        
        df['cluster'] = cluster_labels
        
        # Count clusters and noise
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        self.logger.info(f"   Found {n_clusters} clusters, {n_noise} noise points")
        
        # Aggregate clusters into zones
        clustered_candidates = []
        
        # Process each cluster
        for cluster_id in sorted(set(cluster_labels)):
            cluster_data = df[df['cluster'] == cluster_id]
            
            if cluster_id == -1:
                # Noise points - keep individually
                for _, row in cluster_data.iterrows():
                    clustered_candidates.append({
                        'price': float(row['price']),
                        'idx': int(row['idx']),
                        'type': row['type'],
                        'timestamp': row['timestamp'],
                        'cluster_size': 1,
                        'is_cluster': False
                    })
            else:
                # Real cluster - aggregate
                # Use mean price and earliest timestamp (TIMESTAMP CONTRACT)
                clustered_candidates.append({
                    'price': float(cluster_data['price'].mean()),
                    'idx': int(cluster_data['idx'].min()),  # Earliest index
                    'type': cluster_data['type'].mode()[0],  # Most common type
                    'timestamp': cluster_data['timestamp'].min(),  # EARLIEST timestamp
                    'cluster_size': len(cluster_data),
                    'is_cluster': True
                })
        
        self.logger.info(
            f"✅ Reduced {len(candidates)} candidates to {len(clustered_candidates)} zones "
            f"({n_clusters} clusters + {n_noise} noise)"
        )
        
        return clustered_candidates
    
    def get_clustering_report(self, candidates_before: int, candidates_after: int) -> Dict[str, Any]:
        """
        Get clustering statistics report.
        
        Args:
            candidates_before: Number of candidates before clustering
            candidates_after: Number of candidates after clustering
        
        Returns:
            Dictionary with clustering statistics
        """
        reduction_pct = (1 - candidates_after / candidates_before) * 100 if candidates_before > 0 else 0
        
        return {
            'candidates_before': candidates_before,
            'candidates_after': candidates_after,
            'reduction_count': candidates_before - candidates_after,
            'reduction_pct': reduction_pct,
            'enabled': self.enable_clustering
        }
