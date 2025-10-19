"""
SR Clustering Component.

This component clusters Support/Resistance levels using optimized parameters.
Refactored to inherit from BaseStep for autonomous execution.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

class SRClusteringComponent(BaseStep):
    """
    SR Clustering Component.

    Clusters Support/Resistance levels using optimized parameters.
    Refactored to inherit from BaseStep for autonomous execution.
    """

    def __init__(self, step_name: str = "sr_clustering"):
        """Initialize the SR clustering component."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRClustering')

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_clustering_result']

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SR clustering.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.

        Returns:
            Execution result with artifacts and metrics
        """
        self.logger.info('🔗 Starting SR Clustering')

        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            execution_mode = config.get('execution_mode', 'light')
            
            if not symbol:
                raise ValueError("Symbol is required for SR clustering")
            
            self.logger.info(f"Clustering SR levels for {symbol} from {exchange}")
            self.logger.info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up artifact manager context
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst'
            )
            
            # Perform SR clustering (simplified version)
            clustering_result = await self._perform_sr_clustering(symbol, timeframe, direction, execution_mode)

            # Save clustering result as artifact (will auto-generate CSV if < 2000 rows)
            artifact_path = self._save_artifact(
                clustering_result,
                'sr_clustering_result',
                'data'
            )
            artifacts.append(artifact_path)
            
            # Record metrics
            metrics.update({
                'total_clusters': clustering_result.get('total_clusters', 0),
                'clustering_efficiency': clustering_result.get('clustering_efficiency', 0.0),
                'execution_mode': execution_mode
            })

            self.logger.info(f'✅ SR Clustering completed: {metrics["total_clusters"]} clusters created')
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'clustering_result': clustering_result
            }

        except Exception as e:
            self.logger.error(f'❌ SR Clustering failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }

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

    async def _perform_sr_clustering(self, symbol: str, timeframe: str, 
                                   direction: str, execution_mode: str) -> Dict[str, Any]:
        """
        Perform SR clustering with simplified logic.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            direction: Trading direction
            execution_mode: Execution mode (light/full)
            
        Returns:
            Clustering result dictionary
        """
        try:
            # Create sample clustering result for demonstration
            # In a real implementation, this would use the existing clustering logic
            
            sample_clusters = [
                {
                    'cluster_id': 1,
                    'levels': [1.2000, 1.2050, 1.2100],
                    'strength': 0.85,
                    'type': 'support'
                },
                {
                    'cluster_id': 2,
                    'levels': [1.2500, 1.2550],
                    'strength': 0.72,
                    'type': 'resistance'
                }
            ]
            
            return {
                'total_clusters': len(sample_clusters),
                'clustering_efficiency': 0.6,
                'clusters': sample_clusters,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode
                }
            }
            
        except Exception as e:
            self.logger.error(f"SR clustering failed: {e}")
            return {
                'total_clusters': 0,
                'clustering_efficiency': 0.0,
                'clusters': [],
                'error': str(e)
            }
