"""
SR Clustering Component.

This component clusters Support/Resistance levels using optimized parameters.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class SRClusteringComponent(BaseMarketAnalysisComponent):
    """
    SR Clustering Component.
    
    Clusters Support/Resistance levels using optimized parameters.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the SR clustering component."""
        super().__init__(config)
        self.logger = system_logger.getChild('SRClustering')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_clustering_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute SR clustering.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with clustering results
        """
        self.logger.info('🔗 Starting SR Clustering')
        
        try:
            # Import SR clustering utilities
            from src.utils.sr_clustering.parameter_optimization_engine import ParameterOptimizationEngine, ParameterOptimizationConfig
            from dataclasses import dataclass
            
            @dataclass
            class ClusteringConfig:
                """Simple configuration for SR clustering."""
                clustering_method: str = 'proximity_based'
                distance_threshold: float = 0.005
                min_cluster_size: int = 2
                quality_threshold: float = 0.6
                enable_gpu_acceleration: bool = True
                memory_limit_gb: float = 8.0
            
            # Get SR levels from previous stage
            # Check both direct pipeline state and artifacts from previous steps
            sr_levels = pipeline_state.get('sr_levels', [])
            
            # Debug: Log pipeline state keys
            self.logger.info(f"Pipeline state keys: {list(pipeline_state.keys())}")
            
            # If not found in direct state, check artifacts from sr_detection
            if not sr_levels:
                sr_detection_artifacts = pipeline_state.get('artifacts', {})
                self.logger.info(f"Artifacts keys: {list(sr_detection_artifacts.keys())}")
                sr_detection_result = sr_detection_artifacts.get('sr_detection_result', {})
                self.logger.info(f"SR detection result keys: {list(sr_detection_result.keys())}")
                sr_levels = sr_detection_result.get('sr_levels', [])
            
            self.logger.info(f"Found {len(sr_levels)} SR levels for clustering")
            
            if not sr_levels:
                raise ValueError("No SR levels available for clustering")
            
            # Get optimized parameters
            optimized_parameters = pipeline_state.get('optimized_parameters', {})
            quality_thresholds = pipeline_state.get('quality_thresholds', {})
            
            # Configure clustering
            clustering_config = ClusteringConfig(
                clustering_method='proximity_based',
                distance_threshold=optimized_parameters.get('clustering_distance', 0.005),
                min_cluster_size=optimized_parameters.get('min_cluster_size', 2),
                quality_threshold=quality_thresholds.get('min_cluster_quality', 0.6),
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Perform simple proximity-based clustering
            self.logger.info(f'🔗 Clustering {len(sr_levels)} SR levels using {clustering_config.clustering_method}')
            clustering_result = await self._simple_proximity_clustering(sr_levels, clustering_config)
            
            # Extract results
            clustered_levels = clustering_result.get('clustered_levels', [])
            cluster_metrics = clustering_result.get('cluster_metrics', {})
            
            # Validate that we have clustered levels
            if not clustered_levels:
                raise ValueError("SR clustering completed but no clusters were created")
            
            # Create single consolidated artifact
            artifacts = {
                'sr_clustering_result': {
                    'clustered_levels': clustered_levels,
                    'cluster_metrics': cluster_metrics,
                    'clustering_summary': {
                        'total_clusters': len(clustered_levels),
                        'total_original_levels': len(sr_levels),
                        'clustering_efficiency': len(clustered_levels) / len(sr_levels) if sr_levels else 0.0,
                        'clustering_time': clustering_result.get('clustering_time', 0.0)
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'original_levels': len(sr_levels),
                        'execution_timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            self.logger.info(f'✅ SR Clustering completed: {len(clustered_levels)} clusters created')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'original_levels': len(sr_levels),
                    'clustered_levels': len(clustered_levels)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ SR Clustering failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _simple_proximity_clustering(
        self, 
        sr_levels: List[Dict[str, Any]], 
        config: Any
    ) -> Dict[str, Any]:
        """Perform simple proximity-based clustering."""
        try:
            start_time = datetime.now()
            
            # Group levels by proximity
            clusters = []
            used_indices = set()
            
            for i, level in enumerate(sr_levels):
                if i in used_indices:
                    continue
                    
                # Start a new cluster with this level
                cluster = [level]
                used_indices.add(i)
                level_price = level.get('price', 0.0)
                
                # Find nearby levels
                for j, other_level in enumerate(sr_levels):
                    if j in used_indices or j == i:
                        continue
                        
                    other_price = other_level.get('price', 0.0)
                    price_diff = abs(level_price - other_price) / level_price
                    
                    # If within distance threshold, add to cluster
                    if price_diff <= config.distance_threshold:
                        cluster.append(other_level)
                        used_indices.add(j)
                
                # Only keep clusters that meet minimum size requirement
                if len(cluster) >= config.min_cluster_size:
                    # Calculate cluster representative (strongest level)
                    best_level = max(cluster, key=lambda x: x.get('strength', 0.0))
                    clusters.append(best_level)
            
            # If no clusters meet minimum size, return all levels as individual clusters
            if not clusters:
                clusters = sr_levels
            
            end_time = datetime.now()
            clustering_time = (end_time - start_time).total_seconds()
            
            return {
                'clustered_levels': clusters,
                'cluster_metrics': {
                    'total_clusters': len(clusters),
                    'original_levels': len(sr_levels),
                    'reduction_ratio': len(clusters) / len(sr_levels) if sr_levels else 0.0
                },
                'clustering_time': clustering_time
            }
            
        except Exception as e:
            self.logger.error(f"Clustering process failed: {e}")
            return {
                'clustered_levels': sr_levels,  # Fallback to original levels
                'cluster_metrics': {
                    'total_clusters': len(sr_levels),
                    'original_levels': len(sr_levels),
                    'reduction_ratio': 1.0
                },
                'clustering_time': 0.0
            }