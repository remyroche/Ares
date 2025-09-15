"""
HMM Clustering Component.

This component performs HMM-based regime clustering.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class HMMClusteringComponent(BaseMarketAnalysisComponent):
    """
    HMM Clustering Component.
    
    Performs HMM-based regime clustering.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM clustering component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMClustering')
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_clustering_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM clustering.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with clustering results
        """
        self.logger.info('🔄 Starting HMM Clustering')
        
        try:
            # Import HMM clustering utilities
            from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for HMM clustering")
            
            # Get regime discovery results from previous stage
            hmm_regime_discovery = pipeline_state.get('hmm_regime_discovery_result', {})
            if not hmm_regime_discovery:
                raise ValueError("No HMM regime discovery results available for clustering")
            
            # Configure HMM clustering
            clustering_config = {
                'n_clusters': 3,  # Bull, Bear, Sideways
                'clustering_method': 'hmm_based',
                'min_cluster_size': 10,
                'convergence_tolerance': 1e-6,
                'max_iterations': 100,
                
                # Hardware optimization
                'enable_parallel_processing': True,
                'enable_gpu_acceleration': True,
                'memory_limit_gb': 8.0
            }
            
            # Create HMM composite manager
            hmm_manager = EnhancedHMMCompositeManager()
            
            # Perform HMM clustering
            clustering_result = await self._perform_hmm_clustering(
                hmm_manager, market_data, hmm_regime_discovery, clustering_config
            )
            
            # Extract results
            hmm_models = clustering_result.get('hmm_models', [])
            cluster_assignments = clustering_result.get('cluster_assignments', [])
            cluster_metrics = clustering_result.get('cluster_metrics', {})
            
            # Validate that we have clustering results
            if not hmm_models or not cluster_assignments:
                raise ValueError("HMM clustering completed but no clusters were created")
            
            # Create single consolidated artifact
            artifacts = {
                'hmm_clustering_result': {
                    'hmm_models': hmm_models,
                    'cluster_assignments': cluster_assignments,
                    'cluster_metrics': cluster_metrics,
                    'clustering_summary': {
                        'total_clusters': len(hmm_models),
                        'total_assignments': len(cluster_assignments),
                        'cluster_distribution': self._calculate_cluster_distribution(cluster_assignments),
                        'clustering_time': clustering_result.get('clustering_time', 0.0)
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Clustering completed: {len(hmm_models)} clusters created')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'cluster_count': len(hmm_models)
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ HMM Clustering failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for clustering."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_hmm_clustering(
        self, 
        hmm_manager: Any, 
        market_data: Any, 
        regime_discovery: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual HMM clustering process."""
        try:
            # Prepare data for clustering
            prepared_data = self._prepare_data_for_clustering(market_data, regime_discovery)
            
            # Perform HMM clustering
            clustering_result = await hmm_manager.perform_hmm_clustering(prepared_data, config)
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"HMM clustering process failed: {e}")
            # Return fallback clustering result
            return {
                'hmm_models': [],
                'cluster_assignments': [],
                'cluster_metrics': {
                    'clustering_method': 'fallback',
                    'error': str(e)
                },
                'clustering_time': 0.0
            }
    
    def _prepare_data_for_clustering(self, data: Any, regime_discovery: Dict[str, Any]) -> Any:
        """Prepare market data and regime discovery results for clustering."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'regime_discovery': regime_discovery
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for clustering: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'regime_discovery': regime_discovery
        }
    
    def _calculate_cluster_distribution(self, cluster_assignments: List[int]) -> Dict[str, float]:
        """Calculate the distribution of cluster assignments."""
        if not cluster_assignments:
            return {}
        
        total_assignments = len(cluster_assignments)
        cluster_counts = {}
        
        for assignment in cluster_assignments:
            cluster_counts[assignment] = cluster_counts.get(assignment, 0) + 1
        
        # Convert to percentages
        cluster_distribution = {}
        for cluster, count in cluster_counts.items():
            cluster_distribution[f'cluster_{cluster}'] = (count / total_assignments) * 100
        
        return cluster_distribution