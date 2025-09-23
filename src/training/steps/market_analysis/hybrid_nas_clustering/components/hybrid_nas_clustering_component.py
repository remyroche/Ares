"""
Hybrid NAS Clustering Component - Pipeline Integration

This component integrates hybrid NAS clustering into the existing pipeline
as a complementary approach to the existing neural NAS system.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime

from ..core.hybrid_nas_clusterer import HybridNASClusterer
from ..core.hybrid_nas_config import HybridNASClusteringConfig

logger = logging.getLogger(__name__)


class HybridNASClusteringComponent:
    """Hybrid NAS Clustering Component for pipeline integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid NAS clustering component."""
        self.config = config
        self.logger = logger.getChild('HybridNASClusteringComponent')
        
        # Create hybrid NAS configuration
        self.hybrid_config = HybridNASClusteringConfig.from_dict(config)
        
        # Initialize hybrid clusterer
        self.hybrid_clusterer = HybridNASClusterer(self.hybrid_config)
        
        self.logger.info("✅ Hybrid NAS Clustering Component initialized")
    
    async def execute(self, data: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hybrid NAS clustering as part of the pipeline.
        
        Args:
            data: Input data containing market data and timestamps
            pipeline_state: Current pipeline state
            
        Returns:
            Hybrid clustering results
        """
        self.logger.info("🚀 Executing Hybrid NAS Clustering...")
        start_time = time.time()
        
        try:
            # Extract data from pipeline
            market_data = data.get('market_data')
            timestamps = data.get('timestamps')
            
            if market_data is None or timestamps is None:
                raise ValueError("Missing required data: market_data or timestamps")
            
            # Convert to DataFrame if needed
            if not isinstance(market_data, pd.DataFrame):
                market_data = pd.DataFrame(market_data)
            
            # Convert timestamps to numpy array if needed
            if not isinstance(timestamps, np.ndarray):
                timestamps = np.array(timestamps)
            
            # Run hybrid NAS clustering
            results = self.hybrid_clusterer.cluster(
                market_data=market_data,
                timestamps=timestamps,
                optimize_parameters=True,
                generate_report=True
            )
            
            # Add pipeline metadata
            results['pipeline_metadata'] = {
                'component': 'hybrid_nas_clustering',
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'pipeline_state': pipeline_state.get('current_step', 'unknown')
            }
            
            # Update pipeline state
            pipeline_state['hybrid_nas_results'] = results
            pipeline_state['last_execution_time'] = time.time()
            
            self.logger.info(f"✅ Hybrid NAS Clustering completed in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid NAS Clustering execution failed: {e}")
            raise
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'name': 'HybridNASClusteringComponent',
            'version': '1.0.0',
            'description': 'Hybrid NAS Clustering Component combining tree-based and neural approaches',
            'config': self.hybrid_config.to_dict(),
            'capabilities': [
                'complementary_clustering',
                'ensemble_clustering', 
                'routing_clustering',
                'sequential_clustering',
                'feature_selection',
                'regime_detection',
                'micro_regime_detection'
            ]
        }
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data."""
        try:
            required_fields = ['market_data', 'timestamps']
            for field in required_fields:
                if field not in data:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate market data
            market_data = data['market_data']
            if not isinstance(market_data, (pd.DataFrame, np.ndarray, list)):
                self.logger.error("market_data must be DataFrame, numpy array, or list")
                return False
            
            # Validate timestamps
            timestamps = data['timestamps']
            if not isinstance(timestamps, (np.ndarray, list)):
                self.logger.error("timestamps must be numpy array or list")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the component."""
        try:
            if hasattr(self.hybrid_clusterer, 'clustering_results') and self.hybrid_clusterer.clustering_results:
                results = self.hybrid_clusterer.clustering_results
                return {
                    'accuracy': results.get('quality_metrics', {}).get('accuracy', 0.0),
                    'efficiency': results.get('quality_metrics', {}).get('efficiency', 0.0),
                    'interpretability': results.get('quality_metrics', {}).get('interpretability', 0.0),
                    'n_clusters': results.get('statistics', {}).get('n_clusters', 0),
                    'silhouette_score': results.get('statistics', {}).get('silhouette_score', 0.0)
                }
            else:
                return {'message': 'No performance metrics available'}
                
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {'error': str(e)}
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update component configuration."""
        try:
            self.hybrid_config.update_config(new_config)
            self.hybrid_clusterer = HybridNASClusterer(self.hybrid_config)
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from tree-based models."""
        try:
            if hasattr(self.hybrid_clusterer, 'clustering_results') and self.hybrid_clusterer.clustering_results:
                results = self.hybrid_clusterer.clustering_results
                if 'tree_results' in results and 'tree_architecture' in results['tree_results']:
                    tree_architecture = results['tree_results']['tree_architecture']
                    if hasattr(tree_architecture, 'feature_importance'):
                        return {
                            'feature_importance': tree_architecture.feature_importance,
                            'selected_features': tree_architecture.selected_features,
                            'feature_ranking': tree_architecture.feature_ranking
                        }
            
            return {'message': 'No feature importance available'}
            
        except Exception as e:
            self.logger.error(f"Feature importance retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get regime analysis results."""
        try:
            if hasattr(self.hybrid_clusterer, 'clustering_results') and self.hybrid_clusterer.clustering_results:
                results = self.hybrid_clusterer.clustering_results
                return {
                    'regime_labels': results.get('labels', []),
                    'regime_centers': results.get('cluster_centers', []),
                    'regime_statistics': results.get('statistics', {}),
                    'regime_quality': results.get('quality_metrics', {}),
                    'regime_transitions': results.get('transition_matrix', []),
                    'micro_regimes': results.get('micro_regimes', {})
                }
            else:
                return {'message': 'No regime analysis available'}
                
        except Exception as e:
            self.logger.error(f"Regime analysis retrieval failed: {e}")
            return {'error': str(e)}
    
    def compare_with_neural_nas(self, neural_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare hybrid results with neural NAS results."""
        try:
            if not hasattr(self.hybrid_clusterer, 'clustering_results') or not self.hybrid_clusterer.clustering_results:
                return {'message': 'No hybrid results available for comparison'}
            
            hybrid_results = self.hybrid_clusterer.clustering_results
            
            comparison = {
                'accuracy_comparison': {
                    'hybrid': hybrid_results.get('quality_metrics', {}).get('accuracy', 0.0),
                    'neural': neural_results.get('quality_metrics', {}).get('accuracy', 0.0),
                    'improvement': hybrid_results.get('quality_metrics', {}).get('accuracy', 0.0) - 
                                 neural_results.get('quality_metrics', {}).get('accuracy', 0.0)
                },
                'efficiency_comparison': {
                    'hybrid': hybrid_results.get('quality_metrics', {}).get('efficiency', 0.0),
                    'neural': neural_results.get('quality_metrics', {}).get('efficiency', 0.0),
                    'improvement': hybrid_results.get('quality_metrics', {}).get('efficiency', 0.0) - 
                                 neural_results.get('quality_metrics', {}).get('efficiency', 0.0)
                },
                'interpretability_comparison': {
                    'hybrid': hybrid_results.get('quality_metrics', {}).get('interpretability', 0.0),
                    'neural': 0.3,  # Neural networks are less interpretable
                    'improvement': hybrid_results.get('quality_metrics', {}).get('interpretability', 0.0) - 0.3
                },
                'execution_time_comparison': {
                    'hybrid': hybrid_results.get('hybrid_metadata', {}).get('execution_time', 0.0),
                    'neural': neural_results.get('execution_time', 0.0),
                    'improvement': neural_results.get('execution_time', 0.0) - 
                                 hybrid_results.get('hybrid_metadata', {}).get('execution_time', 0.0)
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Comparison failed: {e}")
            return {'error': str(e)}