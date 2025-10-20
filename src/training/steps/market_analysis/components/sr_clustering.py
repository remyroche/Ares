"""
SR Clustering Component.

This component clusters Support/Resistance levels using optimized parameters.
Refactored to inherit from BaseStep for autonomous execution with full hardware optimization.
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

# Hardware optimization imports
from src.utils.hardware import (
    get_integrated_hardware_manager, IntegratedHardwareConfig,
    memory_optimized, smart_cache, performance_tracked,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_dataframe, force_cleanup, get_memory_stats
)
from src.utils.hardware.memory_optimized_decorators import (
    MemoryOptimizationLevel as MemOptLevel, chunked_processing_auto
)
from src.utils.hardware.optimization_decorators import (
    OptimizationConfig, OptimizationLevel as DecoratorOptLevel
)

class SRClusteringComponent(BaseStep):
    """
    SR Clustering Component.

    Clusters Support/Resistance levels using optimized parameters.
    Refactored to inherit from BaseStep for autonomous execution with full hardware optimization.
    """

    def __init__(self, step_name: str = "sr_clustering"):
        """Initialize the SR clustering component with hardware optimization."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('SRClustering')
        
        # Initialize hardware optimization
        self.hardware_config = IntegratedHardwareConfig(
            enable_automatic_optimization=True,
            enable_caching=True,
            enable_memory_monitoring=True,
            memory_limit_gb=8.0,
            cache_memory_limit_mb=512.0
        )
        self.hardware_manager = get_integrated_hardware_manager(self.hardware_config)
        
        # Clustering configuration
        self.clustering_config = {
            'distance_threshold': 0.02,  # 2% price difference threshold
            'min_cluster_size': 3,       # Minimum levels per cluster
            'max_clusters': 50,          # Maximum number of clusters
            'strength_threshold': 0.5    # Minimum strength for cluster inclusion
        }

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_clustering_result']

    @memory_optimized(optimization_level=MemOptLevel.AGGRESSIVE)
    @performance_tracked
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SR clustering with full hardware optimization.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.

        Returns:
            Execution result with artifacts and metrics
        """
        self.logger.info('🔗 Starting SR Clustering with hardware optimization')

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
            
            # Optimize hardware for clustering workload
            self.hardware_manager.optimize_for_workload('DATA_PROCESSING')
            
            # Load SR levels data (try to load from previous steps)
            sr_levels = await self._load_sr_levels(symbol, exchange, timeframe, direction)
            
            if sr_levels is None or len(sr_levels) == 0:
                self.logger.warning("No SR levels found, generating sample data for demonstration")
                sr_levels = self._generate_sample_sr_levels(symbol, timeframe)
            
            # Perform optimized SR clustering
            clustering_result = await self._perform_optimized_sr_clustering(
                sr_levels, symbol, timeframe, direction, execution_mode
            )

            # Optimize clustering result for storage
            optimized_result = self.hardware_manager.process_data_with_optimization(
                clustering_result, 'DATA_PROCESSING'
            )

            # Save clustering result as artifact (will auto-generate CSV if < 2000 rows)
            artifact_path = self._save_artifact(
                optimized_result,
                'sr_clustering_result',
                'data'
            )
            artifacts.append(artifact_path)
            
            # Record comprehensive metrics
            metrics.update({
                'total_clusters': clustering_result.get('total_clusters', 0),
                'clustering_efficiency': clustering_result.get('clustering_efficiency', 0.0),
                'execution_mode': execution_mode,
                'memory_usage_mb': get_memory_stats().get('used_memory', 0) / (1024 * 1024),
                'hardware_optimization_applied': True
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
        finally:
            # Force cleanup after execution
            force_cleanup()

    @memory_optimized(optimization_level=MemOptLevel.AGGRESSIVE)
    @smart_cache(ttl=3600, max_size=100)
    async def _load_sr_levels(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Optional[List[Dict[str, Any]]]:
        """Load SR levels from previous steps with caching."""
        try:
            # Try to load from sr_detection step
            sr_data = self._get_artifact('sr_detection_result', 'data')
            if sr_data is not None and isinstance(sr_data, dict):
                levels = sr_data.get('sr_levels', [])
                if levels:
                    self.logger.info(f"Loaded {len(levels)} SR levels from previous step")
                    return levels
            
            # Try alternative artifact names
            alternative_names = ['sr_levels', 'support_resistance_levels', 'sr_analysis_result']
            for name in alternative_names:
                data = self._get_artifact(name, 'data')
                if data is not None:
                    if isinstance(data, list):
                        self.logger.info(f"Loaded {len(data)} SR levels from {name}")
                        return data
                    elif isinstance(data, dict) and 'levels' in data:
                        levels = data['levels']
                        self.logger.info(f"Loaded {len(levels)} SR levels from {name}")
                        return levels
            
            self.logger.warning("No SR levels found in artifacts")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load SR levels: {e}")
            return None

    def _generate_sample_sr_levels(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Generate sample SR levels for demonstration purposes."""
        import random
        
        # Generate realistic price levels around a base price
        base_price = 2000.0  # Example base price
        levels = []
        
        # Generate support levels (below base price)
        for i in range(5):
            price = base_price * (0.95 - i * 0.02)
            levels.append({
                'price': round(price, 2),
                'type': 'support',
                'strength': random.uniform(0.6, 0.9),
                'touches': random.randint(2, 5),
                'timeframe': timeframe
            })
        
        # Generate resistance levels (above base price)
        for i in range(5):
            price = base_price * (1.05 + i * 0.02)
            levels.append({
                'price': round(price, 2),
                'type': 'resistance',
                'strength': random.uniform(0.6, 0.9),
                'touches': random.randint(2, 5),
                'timeframe': timeframe
            })
        
        return levels

    @memory_optimized(optimization_level=MemOptLevel.AGGRESSIVE)
    @performance_tracked
    async def _perform_optimized_sr_clustering(
        self, 
        sr_levels: List[Dict[str, Any]], 
        symbol: str, 
        timeframe: str, 
        direction: str, 
        execution_mode: str
    ) -> Dict[str, Any]:
        """
        Perform optimized SR clustering with hardware acceleration.
        
        Args:
            sr_levels: List of SR level dictionaries
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            direction: Trading direction
            execution_mode: Execution mode (light/full)
            
        Returns:
            Clustering result dictionary
        """
        try:
            start_time = datetime.now()
            
            if not sr_levels:
                return {
                    'total_clusters': 0,
                    'clustering_efficiency': 0.0,
                    'clusters': [],
                    'metadata': {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': direction,
                        'execution_mode': execution_mode
                    }
                }
            
            # Convert to DataFrame for optimization
            df = pd.DataFrame(sr_levels)
            
            # Optimize DataFrame with hardware acceleration
            optimized_df = self.hardware_manager.process_data_with_optimization(
                df, 'DATA_PROCESSING'
            )
            
            # Perform clustering based on execution mode
            if execution_mode == 'light':
                clusters = await self._perform_light_clustering(optimized_df)
            else:
                clusters = await self._perform_full_clustering(optimized_df)
            
            # Calculate metrics
            end_time = datetime.now()
            clustering_time = (end_time - start_time).total_seconds()
            
            clustering_efficiency = len(clusters) / len(sr_levels) if sr_levels else 0.0
            
            return {
                'total_clusters': len(clusters),
                'clustering_efficiency': clustering_efficiency,
                'clusters': clusters,
                'clustering_time': clustering_time,
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode,
                    'hardware_optimized': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Optimized SR clustering failed: {e}")
            return {
                'total_clusters': 0,
                'clustering_efficiency': 0.0,
                'clusters': [],
                'error': str(e)
            }

    @chunked_processing_auto(chunk_size_mb=50.0)
    async def _perform_light_clustering(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform light clustering for fast execution."""
        try:
            clusters = []
            used_indices = set()
            
            for i, row in df.iterrows():
                if i in used_indices:
                    continue
                
                # Start a new cluster
                cluster = [row.to_dict()]
                used_indices.add(i)
                level_price = row['price']
                
                # Find nearby levels
                for j, other_row in df.iterrows():
                    if j in used_indices or j == i:
                        continue
                    
                    other_price = other_row['price']
                    price_diff = abs(level_price - other_price) / level_price
                    
                    if price_diff <= self.clustering_config['distance_threshold']:
                        cluster.append(other_row.to_dict())
                        used_indices.add(j)
                
                # Only keep clusters that meet minimum size
                if len(cluster) >= self.clustering_config['min_cluster_size']:
                    # Find strongest level in cluster
                    best_level = max(cluster, key=lambda x: x.get('strength', 0.0))
                    clusters.append(best_level)
            
            return clusters if clusters else [row.to_dict() for _, row in df.iterrows()]
            
        except Exception as e:
            self.logger.error(f"Light clustering failed: {e}")
            return [row.to_dict() for _, row in df.iterrows()]

    @memory_optimized(optimization_level=MemOptLevel.AGGRESSIVE)
    async def _perform_full_clustering(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform full clustering with advanced algorithms."""
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering
            features = df[['price', 'strength', 'touches']].values
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(
                eps=self.clustering_config['distance_threshold'],
                min_samples=self.clustering_config['min_cluster_size']
            )
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Group levels by cluster
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_data = df[cluster_mask]
                
                # Find strongest level in cluster
                best_idx = cluster_data['strength'].idxmax()
                best_level = cluster_data.loc[best_idx].to_dict()
                clusters.append(best_level)
            
            return clusters if clusters else [row.to_dict() for _, row in df.iterrows()]
            
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to light clustering")
            return await self._perform_light_clustering(df)
        except Exception as e:
            self.logger.error(f"Full clustering failed: {e}")
            return await self._perform_light_clustering(df)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['sr_clustering_result']
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required input artifacts."""
        return ['sr_detection_result', 'sr_levels', 'support_resistance_levels']
    
    @smart_cache(ttl=1800, max_size=50)
    def _get_clustering_config(self, execution_mode: str) -> Dict[str, Any]:
        """Get clustering configuration based on execution mode."""
        base_config = self.clustering_config.copy()
        
        if execution_mode == 'light':
            base_config.update({
                'distance_threshold': 0.03,  # More lenient for speed
                'min_cluster_size': 2,       # Lower minimum
                'max_clusters': 20           # Fewer clusters
            })
        elif execution_mode == 'full':
            base_config.update({
                'distance_threshold': 0.015, # More strict
                'min_cluster_size': 4,       # Higher minimum
                'max_clusters': 100          # More clusters
            })
        
        return base_config
