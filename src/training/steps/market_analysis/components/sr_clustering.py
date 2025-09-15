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
            from src.utils.sr_clustering.sr_clustering_engine import SRClusteringEngine, ClusteringConfig
            
            # Get SR levels from previous stage
            sr_levels = pipeline_state.get('sr_levels', [])
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
                
                # Hardware optimization settings
                enable_parallel_processing=True,
                enable_gpu_acceleration=True,
                memory_limit_gb=8.0
            )
            
            # Create clustering engine
            clustering_engine = SRClusteringEngine(clustering_config)
            
            # Perform clustering
            clustering_result = await self._perform_clustering(
                clustering_engine, sr_levels, clustering_config
            )
            
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
    
    async def _perform_clustering(
        self, 
        clustering_engine: Any, 
        sr_levels: List[Dict[str, Any]], 
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual clustering process."""
        try:
            # Convert SR levels to format expected by clustering engine
            level_data = []
            for level in sr_levels:
                level_data.append({
                    'price': level.get('price', 0.0),
                    'type': level.get('type', 'unknown'),
                    'strength': level.get('strength', 0.0),
                    'touches': level.get('touches', 0),
                    'timestamp': level.get('timestamp', datetime.now().isoformat())
                })
            
            # Perform clustering
            clustering_result = await clustering_engine.cluster_levels(level_data, config)
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Clustering process failed: {e}")
            # Return fallback clustering result
            return {
                'clustered_levels': [],
                'cluster_metrics': {
                    'clustering_method': 'fallback',
                    'error': str(e)
                },
                'clustering_time': 0.0
            }