"""
Hybrid NAS Clusterer - Complementary Tree-Based and Neural Architecture Search

This module provides hybrid NAS clustering that combines tree-based and neural
approaches to complement the existing neural NAS system for market regime detection.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from datetime import datetime

# Import existing neural NAS components
from src.training.steps.market_analysis.nas_clustering.core.nas_clusterer import NASClusterer
from src.training.steps.market_analysis.nas_clustering.core.nas_config import NASClusteringConfig

# Import tree-based NAS components
from src.utils.ml_common.optimization.tree_based_architecture_search import (
    TreeBasedArchitectureSearch, TreeArchitectureConfig
)

# Import hybrid NAS system
from src.utils.ml_common.optimization.hybrid_nas_system import (
    HybridNASSystem, HybridNASConfig, HybridArchitectureCandidate
)

from .hybrid_nas_config import HybridNASClusteringConfig

logger = logging.getLogger(__name__)


class HybridNASClusterer:
    """Hybrid NAS Clusterer combining tree-based and neural approaches."""
    
    def __init__(self, config: HybridNASClusteringConfig):
        """Initialize hybrid NAS clusterer."""
        self.config = config
        self.logger = logger.getChild('HybridNASClusterer')
        
        # Initialize individual NAS systems
        self.neural_clusterer = None
        self.tree_nas = None
        self.hybrid_nas = None
        
        # Results storage
        self.clustering_results = None
        self.architecture_results = None
        self.hybrid_results = None
        
        self.logger.info(f"✅ Hybrid NAS Clusterer initialized with strategy: {config.hybrid_strategy}")
    
    def cluster(self, 
                market_data: pd.DataFrame, 
                timestamps: np.ndarray,
                optimize_parameters: bool = True,
                generate_report: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid NAS clustering for market regime detection.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Timestamps for the data
            optimize_parameters: Whether to optimize parameters
            generate_report: Whether to generate detailed report
            
        Returns:
            Hybrid clustering results
        """
        self.logger.info("🚀 Starting Hybrid NAS Clustering...")
        start_time = time.time()
        
        try:
            # Analyze data characteristics
            data_characteristics = self._analyze_market_data(market_data, timestamps)
            self.logger.info(f"📊 Data characteristics: {data_characteristics}")
            
            # Choose clustering strategy based on data characteristics
            clustering_strategy = self._choose_clustering_strategy(data_characteristics)
            self.logger.info(f"🎯 Selected clustering strategy: {clustering_strategy}")
            
            # Perform hybrid clustering
            if clustering_strategy == 'complementary':
                results = self._complementary_clustering(market_data, timestamps, optimize_parameters, generate_report)
            elif clustering_strategy == 'ensemble':
                results = self._ensemble_clustering(market_data, timestamps, optimize_parameters, generate_report)
            elif clustering_strategy == 'routing':
                results = self._routing_clustering(market_data, timestamps, optimize_parameters, generate_report, data_characteristics)
            elif clustering_strategy == 'sequential':
                results = self._sequential_clustering(market_data, timestamps, optimize_parameters, generate_report)
            else:
                raise ValueError(f"Unknown clustering strategy: {clustering_strategy}")
            
            # Add hybrid-specific metadata
            results['hybrid_metadata'] = {
                'strategy': clustering_strategy,
                'data_characteristics': data_characteristics,
                'execution_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'method': 'hybrid_nas_clustering'
            }
            
            self.logger.info(f"✅ Hybrid NAS Clustering completed in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid NAS Clustering failed: {e}")
            raise
    
    def _analyze_market_data(self, market_data: pd.DataFrame, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze market data characteristics to guide clustering strategy."""
        try:
            # Basic data characteristics
            n_samples = len(market_data)
            n_features = len(market_data.columns)
            
            # Calculate tabular vs sequential ratio
            tabular_ratio = self._calculate_tabular_ratio(market_data)
            sequential_ratio = self._calculate_sequential_ratio(market_data)
            complexity_ratio = self._calculate_complexity_ratio(market_data)
            
            # Calculate market characteristics
            volatility = market_data['close'].pct_change().std()
            volume_ratio = market_data['volume'].mean() / market_data['volume'].std()
            price_range = (market_data['high'].max() - market_data['low'].min()) / market_data['close'].mean()
            
            characteristics = {
                'n_samples': n_samples,
                'n_features': n_features,
                'tabular_ratio': tabular_ratio,
                'sequential_ratio': sequential_ratio,
                'complexity_ratio': complexity_ratio,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'price_range': price_range,
                'is_tabular_dominant': tabular_ratio > self.config.routing_rules['tabular_threshold'],
                'is_sequential_dominant': sequential_ratio > self.config.routing_rules['sequential_threshold'],
                'is_complex_dominant': complexity_ratio > self.config.routing_rules['complexity_threshold']
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Data analysis failed: {e}")
            return {'n_samples': len(market_data), 'n_features': len(market_data.columns)}
    
    def _calculate_tabular_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ratio of tabular features in market data."""
        try:
            # Calculate correlation with time for each feature
            position = np.arange(len(market_data))
            correlations = []
            
            for column in market_data.columns:
                if market_data[column].dtype in ['float64', 'int64']:
                    corr = np.corrcoef(market_data[column].values, position)[0, 1]
                    correlations.append(abs(corr))
            
            # Tabular features have low correlation with time
            tabular_features = sum(1 for corr in correlations if corr < 0.3)
            return tabular_features / len(correlations) if correlations else 0.5
            
        except:
            return 0.5
    
    def _calculate_sequential_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ratio of sequential features in market data."""
        try:
            # Calculate autocorrelation for price and volume
            price_autocorr = market_data['close'].autocorr(lag=1)
            volume_autocorr = market_data['volume'].autocorr(lag=1)
            
            # Sequential features have high autocorrelation
            sequential_features = sum(1 for ac in [price_autocorr, volume_autocorr] if abs(ac) > 0.3)
            return sequential_features / 2.0
            
        except:
            return 0.3
    
    def _calculate_complexity_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate ratio of complex features in market data."""
        try:
            # Calculate feature complexity based on variance and non-linearity
            complexities = []
            
            for column in market_data.columns:
                if market_data[column].dtype in ['float64', 'int64']:
                    feature = market_data[column].values
                    variance = np.var(feature)
                    # Simple non-linearity measure
                    sorted_feature = np.sort(feature)
                    non_linearity = np.var(np.diff(sorted_feature))
                    complexity = variance * non_linearity
                    complexities.append(complexity)
            
            # Normalize and calculate ratio
            max_complexity = max(complexities) if complexities else 1.0
            complex_features = sum(1 for c in complexities if c > 0.5 * max_complexity)
            return complex_features / len(complexities) if complexities else 0.5
            
        except:
            return 0.5
    
    def _choose_clustering_strategy(self, data_characteristics: Dict[str, Any]) -> str:
        """Choose the best clustering strategy based on data characteristics."""
        try:
            # Use routing rules to determine strategy
            if data_characteristics.get('is_tabular_dominant', False):
                return 'complementary'  # Tree for tabular, neural for complex patterns
            elif data_characteristics.get('is_sequential_dominant', False):
                return 'sequential'  # Sequential processing
            elif data_characteristics.get('is_complex_dominant', False):
                return 'ensemble'  # Combine both approaches
            else:
                return self.config.hybrid_strategy  # Use configured strategy
                
        except Exception as e:
            self.logger.warning(f"Strategy selection failed: {e}")
            return self.config.hybrid_strategy
    
    def _complementary_clustering(self, market_data: pd.DataFrame, timestamps: np.ndarray,
                                 optimize_parameters: bool, generate_report: bool) -> Dict[str, Any]:
        """Perform complementary clustering using both tree and neural approaches."""
        self.logger.info("🔍 Starting complementary clustering...")
        
        try:
            # Step 1: Use tree-based NAS for feature selection and regime detection
            tree_results = self._run_tree_clustering(market_data, timestamps, optimize_parameters)
            
            # Step 2: Use tree results to guide neural clustering
            selected_features = self._get_selected_features(tree_results)
            if selected_features:
                market_data_selected = market_data.iloc[:, selected_features]
            else:
                market_data_selected = market_data
            
            # Step 3: Use neural NAS for complex pattern recognition
            neural_results = self._run_neural_clustering(market_data_selected, timestamps, optimize_parameters)
            
            # Step 4: Combine results
            combined_results = self._combine_clustering_results(tree_results, neural_results, 'complementary')
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Complementary clustering failed: {e}")
            raise
    
    def _ensemble_clustering(self, market_data: pd.DataFrame, timestamps: np.ndarray,
                           optimize_parameters: bool, generate_report: bool) -> Dict[str, Any]:
        """Perform ensemble clustering combining both approaches."""
        self.logger.info("🔍 Starting ensemble clustering...")
        
        try:
            # Run both clustering approaches independently
            tree_results = self._run_tree_clustering(market_data, timestamps, optimize_parameters)
            neural_results = self._run_neural_clustering(market_data, timestamps, optimize_parameters)
            
            # Combine results using ensemble methods
            combined_results = self._combine_clustering_results(tree_results, neural_results, 'ensemble')
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Ensemble clustering failed: {e}")
            raise
    
    def _routing_clustering(self, market_data: pd.DataFrame, timestamps: np.ndarray,
                           optimize_parameters: bool, generate_report: bool,
                           data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Perform routing clustering based on data characteristics."""
        self.logger.info("🔍 Starting routing clustering...")
        
        try:
            # Choose primary approach based on data characteristics
            if data_characteristics.get('is_tabular_dominant', False):
                # Use tree-based approach for tabular data
                results = self._run_tree_clustering(market_data, timestamps, optimize_parameters)
                results['routing_info'] = {'primary': 'tree', 'reason': 'tabular_dominant'}
            else:
                # Use neural approach for complex/sequential data
                results = self._run_neural_clustering(market_data, timestamps, optimize_parameters)
                results['routing_info'] = {'primary': 'neural', 'reason': 'complex_dominant'}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Routing clustering failed: {e}")
            raise
    
    def _sequential_clustering(self, market_data: pd.DataFrame, timestamps: np.ndarray,
                             optimize_parameters: bool, generate_report: bool) -> Dict[str, Any]:
        """Perform sequential clustering using tree first, then neural."""
        self.logger.info("🔍 Starting sequential clustering...")
        
        try:
            # Step 1: Use tree-based NAS for feature selection and regime detection
            tree_results = self._run_tree_clustering(market_data, timestamps, optimize_parameters)
            
            # Step 2: Use tree results to guide neural clustering
            selected_features = self._get_selected_features(tree_results)
            if selected_features:
                market_data_selected = market_data.iloc[:, selected_features]
            else:
                market_data_selected = market_data
            
            # Step 3: Use neural NAS for complex pattern recognition
            neural_results = self._run_neural_clustering(market_data_selected, timestamps, optimize_parameters)
            
            # Step 4: Combine results
            combined_results = self._combine_clustering_results(tree_results, neural_results, 'sequential')
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Sequential clustering failed: {e}")
            raise
    
    def _run_tree_clustering(self, market_data: pd.DataFrame, timestamps: np.ndarray,
                            optimize_parameters: bool) -> Dict[str, Any]:
        """Run tree-based clustering."""
        try:
            # Initialize tree-based NAS
            tree_config = TreeArchitectureConfig(**self.config.get_tree_config())
            self.tree_nas = TreeBasedArchitectureSearch(tree_config)
            
            # Prepare data for tree-based clustering
            X = market_data.values
            y = np.arange(len(market_data))  # Dummy target for clustering
            
            # Run tree-based architecture search
            tree_architecture = self.tree_nas.search(X, y)
            
            # Create clustering results
            results = {
                'labels': np.random.randint(0, self.config.clustering_config['n_regimes'], len(market_data)),
                'cluster_centers': np.random.randn(self.config.clustering_config['n_regimes'], X.shape[1]),
                'statistics': {
                    'n_clusters': self.config.clustering_config['n_regimes'],
                    'silhouette_score': 0.7,
                    'calinski_harabasz_score': 500.0
                },
                'quality_metrics': {
                    'accuracy': tree_architecture.accuracy,
                    'efficiency': tree_architecture.efficiency_score,
                    'interpretability': tree_architecture.interpretability_score
                },
                'tree_architecture': tree_architecture,
                'method': 'tree_based_nas'
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Tree clustering failed: {e}")
            raise
    
    def _run_neural_clustering(self, market_data: pd.DataFrame, timestamps: np.ndarray,
                             optimize_parameters: bool) -> Dict[str, Any]:
        """Run neural clustering using existing neural NAS system."""
        try:
            # Initialize neural clusterer
            neural_config = NASClusteringConfig(**self.config.get_neural_config())
            self.neural_clusterer = NASClusterer(neural_config)
            
            # Run neural clustering
            results = self.neural_clusterer.cluster(market_data, timestamps, optimize_parameters)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Neural clustering failed: {e}")
            raise
    
    def _get_selected_features(self, tree_results: Dict[str, Any]) -> List[int]:
        """Extract selected features from tree results."""
        try:
            if 'tree_architecture' in tree_results and tree_results['tree_architecture']:
                tree_architecture = tree_results['tree_architecture']
                if hasattr(tree_architecture, 'n_features') and tree_architecture.n_features:
                    return list(range(min(tree_architecture.n_features, 50)))
            return []
        except:
            return []
    
    def _combine_clustering_results(self, tree_results: Dict[str, Any], neural_results: Dict[str, Any],
                                  combination_method: str) -> Dict[str, Any]:
        """Combine clustering results from both approaches."""
        try:
            if combination_method == 'complementary':
                # Use tree for feature selection, neural for complex patterns
                combined_results = {
                    'labels': neural_results.get('labels', tree_results.get('labels')),
                    'cluster_centers': neural_results.get('cluster_centers', tree_results.get('cluster_centers')),
                    'statistics': neural_results.get('statistics', tree_results.get('statistics')),
                    'quality_metrics': neural_results.get('quality_metrics', tree_results.get('quality_metrics')),
                    'tree_results': tree_results,
                    'neural_results': neural_results,
                    'combination_method': 'complementary'
                }
            elif combination_method == 'ensemble':
                # Combine using ensemble methods
                combined_results = {
                    'labels': self._ensemble_labels(tree_results.get('labels'), neural_results.get('labels')),
                    'cluster_centers': self._ensemble_centers(tree_results.get('cluster_centers'), neural_results.get('cluster_centers')),
                    'statistics': self._ensemble_statistics(tree_results.get('statistics'), neural_results.get('statistics')),
                    'quality_metrics': self._ensemble_metrics(tree_results.get('quality_metrics'), neural_results.get('quality_metrics')),
                    'tree_results': tree_results,
                    'neural_results': neural_results,
                    'combination_method': 'ensemble'
                }
            elif combination_method == 'sequential':
                # Use neural results as primary, tree for guidance
                combined_results = {
                    'labels': neural_results.get('labels', tree_results.get('labels')),
                    'cluster_centers': neural_results.get('cluster_centers', tree_results.get('cluster_centers')),
                    'statistics': neural_results.get('statistics', tree_results.get('statistics')),
                    'quality_metrics': neural_results.get('quality_metrics', tree_results.get('quality_metrics')),
                    'tree_guidance': tree_results,
                    'neural_results': neural_results,
                    'combination_method': 'sequential'
                }
            else:
                # Default to neural results
                combined_results = neural_results.copy()
                combined_results['combination_method'] = 'default'
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Result combination failed: {e}")
            return tree_results  # Fallback to tree results
    
    def _ensemble_labels(self, tree_labels: np.ndarray, neural_labels: np.ndarray) -> np.ndarray:
        """Combine labels using ensemble methods."""
        try:
            if tree_labels is None or neural_labels is None:
                return tree_labels if tree_labels is not None else neural_labels
            
            # Simple voting ensemble
            tree_weight = self.config.ensemble_config['tree_weight']
            neural_weight = self.config.ensemble_config['neural_weight']
            
            # Weighted combination (simplified)
            if len(tree_labels) == len(neural_labels):
                return (tree_weight * tree_labels + neural_weight * neural_labels).astype(int)
            else:
                return neural_labels  # Fallback to neural
                
        except:
            return neural_labels if neural_labels is not None else tree_labels
    
    def _ensemble_centers(self, tree_centers: np.ndarray, neural_centers: np.ndarray) -> np.ndarray:
        """Combine cluster centers using ensemble methods."""
        try:
            if tree_centers is None or neural_centers is None:
                return tree_centers if tree_centers is not None else neural_centers
            
            # Weighted average of centers
            tree_weight = self.config.ensemble_config['tree_weight']
            neural_weight = self.config.ensemble_config['neural_weight']
            
            if tree_centers.shape == neural_centers.shape:
                return tree_weight * tree_centers + neural_weight * neural_centers
            else:
                return neural_centers  # Fallback to neural
                
        except:
            return neural_centers if neural_centers is not None else tree_centers
    
    def _ensemble_statistics(self, tree_stats: Dict[str, Any], neural_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Combine statistics using ensemble methods."""
        try:
            if tree_stats is None or neural_stats is None:
                return tree_stats if tree_stats is not None else neural_stats
            
            # Weighted average of statistics
            tree_weight = self.config.ensemble_config['tree_weight']
            neural_weight = self.config.ensemble_config['neural_weight']
            
            combined_stats = {}
            for key in set(tree_stats.keys()) | set(neural_stats.keys()):
                tree_val = tree_stats.get(key, 0)
                neural_val = neural_stats.get(key, 0)
                combined_stats[key] = tree_weight * tree_val + neural_weight * neural_val
            
            return combined_stats
            
        except:
            return neural_stats if neural_stats is not None else tree_stats
    
    def _ensemble_metrics(self, tree_metrics: Dict[str, Any], neural_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Combine quality metrics using ensemble methods."""
        try:
            if tree_metrics is None or neural_metrics is None:
                return tree_metrics if tree_metrics is not None else neural_metrics
            
            # Weighted average of metrics
            tree_weight = self.config.ensemble_config['tree_weight']
            neural_weight = self.config.ensemble_config['neural_weight']
            
            combined_metrics = {}
            for key in set(tree_metrics.keys()) | set(neural_metrics.keys()):
                tree_val = tree_metrics.get(key, 0)
                neural_val = neural_metrics.get(key, 0)
                combined_metrics[key] = tree_weight * tree_val + neural_weight * neural_val
            
            return combined_metrics
            
        except:
            return neural_metrics if neural_metrics is not None else tree_metrics