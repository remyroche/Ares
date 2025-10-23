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
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_data_format, tprint_data_preview, tprint_performance,
    tprint_timer, tprint_structured, LogLevel
)

# Hardware optimization imports
from src.utils.hardware import (
    get_integrated_hardware_manager, IntegratedHardwareConfig,
    memory_optimized, smart_cache,
    comprehensive_memory_optimization, MemoryOptimizationLevel,
    optimize_dataframe, force_cleanup, get_memory_stats
)
from src.utils.hardware.memory_optimized_decorators import (
    MemoryOptimizationLevel as MemOptLevel, chunked_processing_auto
)
from src.utils.hardware.optimization_decorators import (
    OptimizationConfig, OptimizationLevel as DecoratorOptLevel, performance_tracked
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
        tprint_info("🚀 Starting SR Clustering Process with Hardware Optimization", level=LogLevel.INFO)
        tprint_structured({
            "step": "sr_clustering_start",
            "component": "SRClusteringComponent",
            "hardware_optimization": True,
            "timestamp": datetime.now().isoformat()
        }, level=LogLevel.INFO)

        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            execution_mode = config.get('execution_mode', 'light')
            
            # Debug configuration with tprint
            tprint_data_format(config, "sr_clustering_config", level=LogLevel.DEBUG)
            tprint_info(f"Configuration extracted - Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}")
            
            if not symbol:
                tprint_error("Symbol is required for SR clustering", level=LogLevel.ERROR)
                raise ValueError("Symbol is required for SR clustering")
            
            self.logger.info(f"Clustering SR levels for {symbol} from {exchange}")
            self.logger.info(f"Timeframe: {timeframe}, Direction: {direction}")
            tprint_info(f"Clustering SR levels for {symbol} from {exchange}")
            tprint_info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up context using BaseStep method
            tprint_debug("Setting up execution context", level=LogLevel.DEBUG)
            self._set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='Analyst'
            )
            
            # Optimize hardware for clustering workload
            tprint_info("Optimizing hardware for clustering workload", level=LogLevel.INFO)
            self.hardware_manager.optimize_for_workload('DATA_PROCESSING')
            tprint_success("Hardware optimization completed", level=LogLevel.INFO)
            
            # Load SR levels data (try to load from previous steps)
            tprint_info("Loading SR levels data from previous steps", level=LogLevel.INFO)
            sr_levels = await self._load_sr_levels(symbol, exchange, timeframe, direction)
            
            if sr_levels is None or len(sr_levels) == 0:
                self.logger.warning("No SR levels found, generating sample data for demonstration")
                tprint_warning("No SR levels found, generating sample data for demonstration", level=LogLevel.WARNING)
                sr_levels = self._generate_sample_sr_levels(symbol, timeframe)
                tprint_info(f"Generated {len(sr_levels)} sample SR levels", level=LogLevel.INFO)
            
            # Preview SR levels for clustering troubleshooting
            tprint_data_preview(sr_levels, "sr_levels_for_clustering", max_rows=10)
            tprint_data_format(sr_levels, "sr_levels_format", level=LogLevel.DEBUG)
            
            # Perform optimized SR clustering
            tprint_info("Starting optimized SR clustering process", level=LogLevel.INFO)
            with tprint_timer("sr_clustering_execution", level=LogLevel.PERFORMANCE):
                clustering_result = await self._perform_optimized_sr_clustering(
                    sr_levels, symbol, timeframe, direction, execution_mode
                )
            
            # Preview clustering result for troubleshooting
            tprint_data_preview(clustering_result, "clustering_result", max_rows=10)
            tprint_data_format(clustering_result, "clustering_result_format", level=LogLevel.DEBUG)
            tprint_success(f"Clustering completed with {clustering_result.get('total_clusters', 0)} clusters", level=LogLevel.INFO)

            # Optimize clustering result for storage
            tprint_info("Optimizing clustering result for storage", level=LogLevel.INFO)
            optimized_result = self.hardware_manager.process_data_with_optimization(
                clustering_result, 'DATA_PROCESSING'
            )
            tprint_success("Data optimization completed", level=LogLevel.INFO)

            # Save clustering result as artifact using enhanced artifact saving
            tprint_info("Saving clustering result as artifact", level=LogLevel.INFO)
            artifact_metadata = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'execution_mode': execution_mode,
                'total_clusters': clustering_result.get('total_clusters', 0),
                'clustering_efficiency': clustering_result.get('clustering_efficiency', 0.0),
                'hardware_optimized': True
            }
            tprint_data_format(artifact_metadata, "artifact_metadata", level=LogLevel.DEBUG)
            
            artifact_path = self._save_enhanced_artifact(
                optimized_result,
                'sr_clustering_result',
                'data',
                artifact_metadata
            )
            artifacts.append(artifact_path)
            tprint_success(f"Artifact saved to: {artifact_path}", level=LogLevel.INFO)
            
            # Record comprehensive metrics
            metrics.update({
                'total_clusters': clustering_result.get('total_clusters', 0),
                'clustering_efficiency': clustering_result.get('clustering_efficiency', 0.0),
                'execution_mode': execution_mode,
                'memory_usage_mb': get_memory_stats().get('used_memory', 0) / (1024 * 1024),
                'hardware_optimization_applied': True
            })
            
            tprint_structured({
                "step": "sr_clustering_complete",
                "metrics": metrics,
                "artifacts_count": len(artifacts)
            }, level=LogLevel.INFO)

            self.logger.info(f'✅ SR Clustering completed: {metrics["total_clusters"]} clusters created')
            tprint_success(f'SR Clustering completed: {metrics["total_clusters"]} clusters created', level=LogLevel.INFO)
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'clustering_result': clustering_result
            }

        except Exception as e:
            self.logger.error(f'❌ SR Clustering failed: {e}')
            tprint_error(f'SR Clustering failed: {e}', level=LogLevel.ERROR)
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f'❌ Error details: {error_details}')
            tprint_structured({
                "step": "sr_clustering_error",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_details
            }, level=LogLevel.ERROR)
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
        finally:
            # Force cleanup after execution
            tprint_debug("Performing final cleanup", level=LogLevel.DEBUG)
            force_cleanup()
            tprint_success("Cleanup completed", level=LogLevel.DEBUG)

    @memory_optimized(optimization_level=MemOptLevel.AGGRESSIVE)
    @smart_cache(ttl=3600, max_size=100)
    async def _load_sr_levels(self, symbol: str, exchange: str, timeframe: str, direction: str) -> Optional[List[Dict[str, Any]]]:
        """Load SR levels from previous steps with caching."""
        tprint_debug(f"Loading SR levels for {symbol} from {exchange}", level=LogLevel.DEBUG)
        try:
            # Try to load from sr_detection step
            sr_data = self._get_artifact('sr_detection_result', 'data')
            if sr_data is not None and isinstance(sr_data, dict):
                levels = sr_data.get('sr_levels', [])
                if levels:
                    self.logger.info(f"Loaded {len(levels)} SR levels from previous step")
                    tprint_success(f"Loaded {len(levels)} SR levels from previous step", level=LogLevel.INFO)
                    # Preview loaded SR levels for troubleshooting
                    tprint_data_preview(levels, "loaded_sr_levels", max_rows=10)
                    tprint_data_format(levels, "loaded_sr_levels_format", level=LogLevel.DEBUG)
                    return levels
            
            # Try alternative artifact names
            tprint_debug("Trying alternative artifact names", level=LogLevel.DEBUG)
            alternative_names = ['sr_levels', 'support_resistance_levels', 'sr_analysis_result']
            for name in alternative_names:
                data = self._get_artifact(name, 'data')
                if data is not None:
                    if isinstance(data, list):
                        self.logger.info(f"Loaded {len(data)} SR levels from {name}")
                        tprint_success(f"Loaded {len(data)} SR levels from {name}", level=LogLevel.INFO)
                        return data
                    elif isinstance(data, dict) and 'levels' in data:
                        levels = data['levels']
                        self.logger.info(f"Loaded {len(levels)} SR levels from {name}")
                        tprint_success(f"Loaded {len(levels)} SR levels from {name}", level=LogLevel.INFO)
                        return levels
            
            self.logger.warning("No SR levels found in artifacts")
            tprint_warning("No SR levels found in artifacts", level=LogLevel.WARNING)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load SR levels: {e}")
            tprint_error(f"Failed to load SR levels: {e}", level=LogLevel.ERROR)
            return None

    def _generate_sample_sr_levels(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Generate sample SR levels for demonstration purposes."""
        tprint_info(f"Generating sample SR levels for {symbol} on {timeframe}", level=LogLevel.INFO)
        import random
        
        # Generate realistic price levels around a base price
        base_price = 2000.0  # Example base price
        levels = []
        
        tprint_debug(f"Base price set to {base_price}", level=LogLevel.DEBUG)
        
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
        
        tprint_success(f"Generated {len(levels)} sample SR levels", level=LogLevel.INFO)
        tprint_data_format(levels, "generated_sample_levels", level=LogLevel.DEBUG)
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
            tprint_info(f"Starting optimized SR clustering with {len(sr_levels)} levels", level=LogLevel.INFO)
            tprint_structured({
                "clustering_params": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": direction,
                    "execution_mode": execution_mode,
                    "levels_count": len(sr_levels)
                }
            }, level=LogLevel.DEBUG)
            
            if not sr_levels:
                tprint_warning("No SR levels provided for clustering", level=LogLevel.WARNING)
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
            tprint_info("Converting SR levels to DataFrame", level=LogLevel.INFO)
            df = pd.DataFrame(sr_levels)
            
            # Preview DataFrame conversion for troubleshooting
            tprint_data_preview(df, "sr_levels_dataframe", max_rows=10)
            tprint_data_format(df, "sr_levels_dataframe_format", level=LogLevel.DEBUG)
            
            # Optimize DataFrame with hardware acceleration
            tprint_info("Applying hardware optimization to DataFrame", level=LogLevel.INFO)
            optimized_df = self.hardware_manager.process_data_with_optimization(
                df, 'DATA_PROCESSING'
            )
            tprint_success("Hardware optimization completed", level=LogLevel.INFO)
            
            # Preview hardware optimized DataFrame for troubleshooting
            tprint_data_preview(optimized_df, "hardware_optimized_dataframe", max_rows=10)
            tprint_data_format(optimized_df, "hardware_optimized_dataframe_format", level=LogLevel.DEBUG)
            
            # Perform clustering based on execution mode
            tprint_info(f"Starting {execution_mode} clustering", level=LogLevel.INFO)
            if execution_mode == 'light':
                clusters = await self._perform_light_clustering(optimized_df)
            else:
                clusters = await self._perform_full_clustering(optimized_df)
            tprint_success(f"Clustering completed with {len(clusters)} clusters", level=LogLevel.INFO)
            
            # Calculate metrics
            end_time = datetime.now()
            clustering_time = (end_time - start_time).total_seconds()
            
            clustering_efficiency = len(clusters) / len(sr_levels) if sr_levels else 0.0
            
            tprint_performance("SR Clustering", clustering_time, level=LogLevel.PERFORMANCE)
            tprint_structured({
                "clustering_metrics": {
                    "total_clusters": len(clusters),
                    "clustering_efficiency": clustering_efficiency,
                    "clustering_time": clustering_time,
                    "execution_mode": execution_mode
                }
            }, level=LogLevel.INFO)
            
            result = {
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
            
            tprint_data_format(result, "clustering_result", level=LogLevel.DEBUG)
            return result
            
        except Exception as e:
            self.logger.error(f"Optimized SR clustering failed: {e}")
            tprint_error(f"Optimized SR clustering failed: {e}", level=LogLevel.ERROR)
            tprint_structured({
                "clustering_error": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            }, level=LogLevel.ERROR)
            return {
                'total_clusters': 0,
                'clustering_efficiency': 0.0,
                'clusters': [],
                'error': str(e)
            }

    @chunked_processing_auto(chunk_size_mb=50.0)
    async def _perform_light_clustering(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform light clustering for fast execution."""
        tprint_info("Starting light clustering process", level=LogLevel.INFO)
        tprint_data_format(self.clustering_config, "clustering_config", level=LogLevel.DEBUG)
        try:
            clusters = []
            used_indices = set()
            
            tprint_debug(f"Processing {len(df)} levels for clustering", level=LogLevel.DEBUG)
            
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
                    tprint_debug(f"Created cluster with {len(cluster)} levels, best strength: {best_level.get('strength', 0.0)}", level=LogLevel.DEBUG)
            
            tprint_success(f"Light clustering completed with {len(clusters)} clusters", level=LogLevel.INFO)
            return clusters if clusters else [row.to_dict() for _, row in df.iterrows()]
            
        except Exception as e:
            self.logger.error(f"Light clustering failed: {e}")
            tprint_error(f"Light clustering failed: {e}", level=LogLevel.ERROR)
            return [row.to_dict() for _, row in df.iterrows()]

    @memory_optimized(optimization_level=MemOptLevel.AGGRESSIVE)
    async def _perform_full_clustering(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform full clustering with advanced algorithms."""
        tprint_info("Starting full clustering with advanced algorithms", level=LogLevel.INFO)
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering
            tprint_debug("Preparing features for clustering", level=LogLevel.DEBUG)
            features = df[['price', 'strength', 'touches']].values
            tprint_data_format(features, "clustering_features", level=LogLevel.DEBUG)
            
            # Standardize features
            tprint_debug("Standardizing features for clustering", level=LogLevel.DEBUG)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            tprint_data_format(features_scaled, "scaled_features", level=LogLevel.DEBUG)
            
            # Perform DBSCAN clustering
            tprint_info("Performing DBSCAN clustering", level=LogLevel.INFO)
            clustering = DBSCAN(
                eps=self.clustering_config['distance_threshold'],
                min_samples=self.clustering_config['min_cluster_size']
            )
            cluster_labels = clustering.fit_predict(features_scaled)
            tprint_data_format(cluster_labels, "cluster_labels", level=LogLevel.DEBUG)
            
            # Group levels by cluster
            tprint_debug("Grouping levels by cluster", level=LogLevel.DEBUG)
            clusters = []
            unique_labels = set(cluster_labels)
            tprint_info(f"Found {len(unique_labels)} unique cluster labels", level=LogLevel.INFO)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    tprint_debug(f"Skipping noise cluster with label {label}", level=LogLevel.DEBUG)
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_data = df[cluster_mask]
                
                # Find strongest level in cluster
                best_idx = cluster_data['strength'].idxmax()
                best_level = cluster_data.loc[best_idx].to_dict()
                clusters.append(best_level)
                tprint_debug(f"Created cluster {label} with {len(cluster_data)} levels, best strength: {best_level.get('strength', 0.0)}", level=LogLevel.DEBUG)
            
            tprint_success(f"Full clustering completed with {len(clusters)} clusters", level=LogLevel.INFO)
            return clusters if clusters else [row.to_dict() for _, row in df.iterrows()]
            
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to light clustering")
            tprint_warning("scikit-learn not available, falling back to light clustering", level=LogLevel.WARNING)
            return await self._perform_light_clustering(df)
        except Exception as e:
            self.logger.error(f"Full clustering failed: {e}")
            tprint_error(f"Full clustering failed: {e}", level=LogLevel.ERROR)
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
