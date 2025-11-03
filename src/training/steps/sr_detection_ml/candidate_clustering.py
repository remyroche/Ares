# src/training/steps/sr_detection_ml/candidate_clustering.py

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from src.training.base_step import BaseStep
from src.utils.artifact_manager import ArtifactManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CandidateClustering(BaseStep):
    """
    Clusters raw candidate levels into S/R zones using 1D DBSCAN.
    """
    def __init__(self, config: dict, artifact_manager: ArtifactManager):
        super().__init__(config, artifact_manager)
        # Define clustering parameters from config, e.g.:
        # self.eps_ratio = self.config.get("clustering_eps_ratio", 0.005) # 0.5% of price
        self.min_samples = self.config.get("clustering_min_samples", 5)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the clustering step.
        Expects a DataFrame with 'price' and 'creation_timestamp' columns.
        """
        logger.info("Starting candidate clustering...")
        
        # 1. Prepare data for 1D clustering
        levels = data[['price']].values
        
        # 2. Determine 'eps' (epsilon) for DBSCAN
        # This is the most critical parameter. It should be a dynamic value,
        # not a fixed one. Example: 0.5% of the median price.
        median_price = np.median(levels)
        # You MUST tune this eps value; it's the max distance between points in a cluster
        eps_value = median_price * self.config.get("clustering_eps_ratio", 0.0025) 
        
        logger.info(f"Running 1D DBSCAN with eps={eps_value} and min_samples={self.min_samples}")
        
        # 3. Use 1D DBSCAN
        db = DBSCAN(eps=eps_value, min_samples=self.min_samples, metric='euclidean').fit(levels)
        
        data['cluster'] = db.labels_
        
        # 4. Filter out noise (cluster label -1)
        clustered_data = data[data['cluster'] != -1].copy()
        
        if clustered_data.empty:
            logger.warning("DBSCAN clustering resulted in no S/R zones. Adjust eps or min_samples.")
            return pd.DataFrame(columns=['level', 'creation_timestamp'])
            
        # 5. Aggregate clusters into S/R Zones
        # The 'level' of the zone is the mean price of all points in that cluster.
        # The 'creation_timestamp' respects the Timestamp Contract: it's the *earliest*
        # timestamp of any point in that cluster.
        agg_rules = {
            'price': 'mean',
            'creation_timestamp': 'min' 
        }
        
        sr_zones = clustered_data.groupby('cluster').agg(agg_rules).reset_index()
        sr_zones = sr_zones.rename(columns={'price': 'level'})
        
        logger.info(f"Clustered {len(data)} candidates into {len(sr_zones)} S/R zones.")
        
        # Return only the essential columns for the next step
        return sr_zones[['level', 'creation_timestamp']]
