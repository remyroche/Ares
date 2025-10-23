"""
Regime Clustering Step

BaseStep-based implementation for regime clustering using the clusters/ folder components.
This step integrates the comprehensive clustering pipeline for regime detection.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import time
import gc

# Import BaseStep and step registry
from src.training.steps.base_step import BaseStep

# Import clustering components
from src.training.steps.market_analysis.clusters import (
    ClusteringOrchestrator,
    ClusteringService,
    ClusteringResult,
    FeaturePreparationStep,
    ClusteringContext
)

# Import utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, tprint_data_preview, LogLevel
)
from src.utils.hardware import get_memory_usage, optimize_dataframe_default
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.serialization_utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class RegimeClusteringStep(BaseStep):
    """
    Regime clustering step using the comprehensive clustering pipeline.
    
    Features:
    - Advanced 3-step iterative clustering with risk mitigation
    - Feature preparation and selection
    - Hardware optimization and memory management
    - Comprehensive validation and reporting
    - Integration with existing regime detection pipeline
    """
    
    def __init__(self, step_name: str = "regime_clustering", config: Optional[Dict[str, Any]] = None):
        """Initialize the regime clustering step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")
        
        # Initialize clustering orchestrator
        self.orchestrator = ClusteringOrchestrator(verbose=True)
        self.clustering_service = ClusteringService(verbose=True)
        
        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "end_time": None,
            "clustering_time": 0.0,
            "feature_preparation_time": 0.0,
            "validation_time": 0.0,
            "memory_usage": [],
            "n_clusters": 0,
            "convergence_achieved": False
        }
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the regime clustering step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Data timeframe
                - features: Optional pre-computed features
                - market_data: Optional market data
                - clustering_config: Optional clustering configuration
        
        Returns:
            Dict containing:
                - success: bool indicating if step completed successfully
                - artifacts: list of artifact paths created
                - metrics: dict of performance metrics
                - cluster_assignments: numpy array of cluster assignments
                - n_clusters: number of clusters found
                - clustering_metrics: detailed clustering metrics
        """
        try:
            start_time = time.time()
            self.performance_metrics["start_time"] = start_time
            
            tprint("🚀 Starting Regime Clustering Step", "INFO")
            tprint(f"📊 Symbol: {config.get('symbol', 'Unknown')}", "INFO")
            tprint(f"📈 Exchange: {config.get('exchange', 'Unknown')}", "INFO")
            
            # Set context for artifact management
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information', 'regime_clustering'),
                direction=config.get('direction', 'both'),
                model=config.get('model', 'RegimeClustering')
            )
            
            # Load or prepare features and market data
            features, market_data = await self._prepare_data(config)
            
            if features is None or market_data is None:
                return {
                    'success': False,
                    'error': 'Failed to prepare features or market data',
                    'artifacts': [],
                    'metrics': self.performance_metrics
                }
            
            tprint(f"📊 Features shape: {features.shape}", "INFO")
            tprint(f"📈 Market data shape: {market_data.shape}", "INFO")
            
            # Prepare clustering configuration
            clustering_config = self._prepare_clustering_config(config)
            
            # Execute clustering pipeline
            clustering_result = await self._execute_clustering_pipeline(
                features, market_data, clustering_config
            )
            
            if clustering_result is None:
                return {
                    'success': False,
                    'error': 'Clustering pipeline failed',
                    'artifacts': [],
                    'metrics': self.performance_metrics
                }
            
            # Save results as artifacts
            artifacts = await self._save_clustering_results(clustering_result, config)
            
            # Calculate final metrics
            end_time = time.time()
            self.performance_metrics["end_time"] = end_time
            self.performance_metrics["clustering_time"] = end_time - start_time
            
            tprint(f"✅ Regime clustering completed in {self.performance_metrics['clustering_time']:.2f}s", "SUCCESS")
            tprint(f"🎯 Found {clustering_result.n_clusters} clusters", "SUCCESS")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': self.performance_metrics,
                'cluster_assignments': clustering_result.cluster_assignments,
                'n_clusters': clustering_result.n_clusters,
                'clustering_metrics': clustering_result.metrics,
                'optimization_history': clustering_result.optimization_history,
                'validation_results': clustering_result.validation_results,
                'convergence_status': clustering_result.convergence_status
            }
            
        except Exception as e:
            error_msg = f"Regime clustering step failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': self.performance_metrics
            }
        finally:
            # Cleanup
            gc.collect()
    
    async def _prepare_data(self, config: Dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Prepare features and market data for clustering."""
        try:
            tprint("🔍 Preparing data for clustering...", "INFO")
            
            # Check if features are provided in config
            if 'features' in config and config['features'] is not None:
                features = config['features']
                tprint("✅ Using provided features", "INFO")
            else:
                # Load features from artifacts
                features = self._load_dataframe('regime_features')
                if features is None:
                    tprint("⚠️ No features found, attempting to generate from market data", "WARNING")
                    features = await self._generate_features_from_market_data(config)
            
            # Check if market data is provided in config
            if 'market_data' in config and config['market_data'] is not None:
                market_data = config['market_data']
                tprint("✅ Using provided market data", "INFO")
            else:
                # Load market data from artifacts
                market_data = self._load_dataframe('market_data')
                if market_data is None:
                    tprint("⚠️ No market data found, attempting to load from klines", "WARNING")
                    market_data = await self._load_market_data_from_klines(config)
            
            # Validate data
            if features is not None and not isinstance(features, np.ndarray):
                if isinstance(features, pd.DataFrame):
                    features = features.values
                else:
                    raise ValueError(f"Invalid features type: {type(features)}")
            
            if market_data is not None and not isinstance(market_data, pd.DataFrame):
                raise ValueError(f"Invalid market_data type: {type(market_data)}")
            
            # Add data preview logging for troubleshooting
            if features is not None:
                tprint_data_preview(features, "features_loaded", max_rows=5, max_cols=10, level=LogLevel.DEBUG)
            
            if market_data is not None:
                tprint_data_preview(market_data, "market_data_loaded", max_rows=5, max_cols=10, level=LogLevel.DEBUG)
            
            return features, market_data
            
        except Exception as e:
            tprint(f"❌ Data preparation failed: {e}", "ERROR")
            return None, None
    
    async def _generate_features_from_market_data(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate features from market data if not available."""
        try:
            # This is a placeholder - in practice, you would implement feature generation
            # based on your specific requirements
            tprint("🔧 Generating features from market data...", "INFO")
            
            # For now, return None to indicate feature generation is not implemented
            # In a real implementation, you would:
            # 1. Load market data
            # 2. Calculate technical indicators
            # 3. Generate regime features
            # 4. Return as numpy array
            
            return None
            
        except Exception as e:
            tprint(f"❌ Feature generation failed: {e}", "ERROR")
            return None
    
    async def _load_market_data_from_klines(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load market data from klines if not available."""
        try:
            symbol = config.get('symbol')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            
            if not symbol:
                tprint("❌ No symbol provided for market data loading", "ERROR")
                return None
            
            tprint(f"📊 Loading market data for {symbol} from {exchange}", "INFO")
            
            # Use klines manager to load data
            klines_manager = get_klines_manager()
            market_data = await klines_manager.get_klines(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe
            )
            
            if market_data is not None and not market_data.empty:
                tprint(f"✅ Loaded {len(market_data)} rows of market data", "SUCCESS")
                # Add data preview logging for troubleshooting
                tprint_data_preview(market_data, "klines_market_data", max_rows=5, max_cols=10, level=LogLevel.DEBUG)
                return market_data
            else:
                tprint("❌ Failed to load market data from klines", "ERROR")
                return None
                
        except Exception as e:
            tprint(f"❌ Market data loading failed: {e}", "ERROR")
            return None
    
    def _prepare_clustering_config(self, config: Dict[str, Any]) -> Any:
        """Prepare clustering configuration."""
        try:
            # Get clustering config from config or use defaults
            clustering_config = config.get('clustering_config', {})
            
            # Create a simple config object
            class ClusteringConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            # Set default parameters
            default_config = {
                'max_clusters': 10,
                'min_clusters': 2,
                'n_iterations': 100,
                'convergence_threshold': 1e-4,
                'random_state': 42,
                'verbose': True
            }
            
            # Merge with provided config
            final_config = {**default_config, **clustering_config}
            
            return ClusteringConfig(**final_config)
            
        except Exception as e:
            tprint(f"❌ Config preparation failed: {e}", "ERROR")
            return None
    
    async def _execute_clustering_pipeline(
        self, 
        features: np.ndarray, 
        market_data: pd.DataFrame, 
        clustering_config: Any
    ) -> Optional[ClusteringResult]:
        """Execute the clustering pipeline."""
        try:
            tprint("🚀 Executing clustering pipeline...", "INFO")
            
            # Add data preview logging before clustering
            tprint_data_preview(features, "clustering_input_features", max_rows=5, max_cols=10, level=LogLevel.DEBUG)
            tprint_data_preview(market_data, "clustering_input_market_data", max_rows=5, max_cols=10, level=LogLevel.DEBUG)
            
            # Use the clustering service to run clustering
            clustering_result = await self.clustering_service.run_clustering(
                features=features,
                market_data=market_data,
                config=clustering_config
            )
            
            if clustering_result is not None:
                self.performance_metrics["n_clusters"] = clustering_result.n_clusters
                self.performance_metrics["convergence_achieved"] = clustering_result.convergence_status == "converged"
                
                tprint(f"✅ Clustering completed: {clustering_result.n_clusters} clusters", "SUCCESS")
                tprint(f"📊 Convergence: {clustering_result.convergence_status}", "INFO")
                
                # Add data preview logging after clustering
                tprint_data_preview(clustering_result.cluster_assignments, "clustering_output_assignments", max_rows=10, level=LogLevel.DEBUG)
                tprint_data_preview(clustering_result.metrics, "clustering_output_metrics", level=LogLevel.DEBUG)
            
            return clustering_result
            
        except Exception as e:
            tprint(f"❌ Clustering pipeline failed: {e}", "ERROR")
            return None
    
    async def _save_clustering_results(
        self, 
        clustering_result: ClusteringResult, 
        config: Dict[str, Any]
    ) -> List[str]:
        """Save clustering results as artifacts."""
        try:
            artifacts = []
            
            # Add data preview logging before saving
            tprint_data_preview(clustering_result.cluster_assignments, "final_cluster_assignments", max_rows=10, level=LogLevel.DEBUG)
            tprint_data_preview(clustering_result.metrics, "final_clustering_metrics", level=LogLevel.DEBUG)
            
            # Save cluster assignments
            cluster_assignments_path = self._save_dataframe(
                pd.DataFrame({'cluster_assignments': clustering_result.cluster_assignments}),
                'cluster_assignments'
            )
            artifacts.append(cluster_assignments_path)
            
            # Save clustering metrics
            metrics_data = {
                'n_clusters': clustering_result.n_clusters,
                'metrics': clustering_result.metrics,
                'optimization_history': clustering_result.optimization_history,
                'validation_results': clustering_result.validation_results,
                'convergence_status': clustering_result.convergence_status,
                'execution_time': clustering_result.execution_time
            }
            
            metrics_path = self._save_metadata(metrics_data, 'clustering_metrics')
            artifacts.append(metrics_path)
            
            # Save performance metrics
            performance_path = self._save_metadata(self.performance_metrics, 'clustering_performance')
            artifacts.append(performance_path)
            
            tprint(f"✅ Saved {len(artifacts)} clustering artifacts", "SUCCESS")
            
            return artifacts
            
        except Exception as e:
            tprint(f"❌ Failed to save clustering results: {e}", "ERROR")
            return []


# Register the step
if __name__ == "__main__":
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_clustering", RegimeClusteringStep)
    print("✅ RegimeClusteringStep registered successfully")